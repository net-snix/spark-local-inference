# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ripple's residual updates fused with their RMSNorms for decode-sized row counts.

The staged forward runs each RMSNorm as three launches: an FP32 square, ATen's mean
(`flare::mean`, kept for its rounding) and the scaling pointwise kernel. Here one program per
row reproduces that arithmetic bit for bit: it sums the squares in the exact order ATen's
`reduce_kernel` uses for a contiguous FP32 row of HIDDEN values, multiplies by ATen's MeanOps
factor, and repeats Inductor's pointwise rounding (BF16 normalized value, BF16 weight product,
BF16 residual sum, FMA lerp).

ATen order for one output row (Reduce.cuh, vectorized input of four, 512-thread blocks): a block
of BW = 512 / min(floor_pow2(rows), 16) lanes reads the row as float4 chunks; lane t keeps four
accumulators, accumulator i summing elements 4 (t + BW k) + i over k in order. The four combine
as ((a0 + a1) + a2) + a3, shared memory halves the lanes down to a warp (lane t adds lane
t + BW / 2, ...), and shuffle-down by 1, 2, 4, 8, 16 finishes in lane 0. FP32 addition is
commutative, so each pairwise step is one exactly reproduced addition.
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

# Rows at most this many take the fused path; larger (prefill) graphs keep the staged one.
MAX_ROWS = 16
_BLOCK = 1024


def aten_block_width(rows: int, hidden: int) -> int:
    """Lanes ATen's reduce_kernel assigns to one row of a contiguous FP32 mean over `hidden`."""
    if hidden % 4 or hidden // 4 < 512 or rows < 1:
        raise ValueError("the reproduced reduce configuration needs at least 2048 aligned columns")
    return 512 // min(1 << (rows.bit_length() - 1), 16)


@triton.jit
def _square_sum(row, VECTORS: tl.constexpr, BW: tl.constexpr):
    lanes = tl.arange(0, BW)
    a0 = tl.zeros([BW], tl.float32)
    a1 = tl.zeros([BW], tl.float32)
    a2 = tl.zeros([BW], tl.float32)
    a3 = tl.zeros([BW], tl.float32)
    for k in tl.static_range((VECTORS + BW - 1) // BW):
        vector = lanes + k * BW
        valid = vector < VECTORS
        x0 = tl.load(row + vector * 4, valid, other=0.).to(tl.float32)
        x1 = tl.load(row + vector * 4 + 1, valid, other=0.).to(tl.float32)
        x2 = tl.load(row + vector * 4 + 2, valid, other=0.).to(tl.float32)
        x3 = tl.load(row + vector * 4 + 3, valid, other=0.).to(tl.float32)
        a0 = a0 + x0 * x0
        a1 = a1 + x1 * x1
        a2 = a2 + x2 * x2
        a3 = a3 + x3 * x3
    total = ((a0 + a1) + a2) + a3
    if BW >= 512:
        total = tl.sum(tl.reshape(total, [2, 256]), axis=0)
    if BW >= 256:
        total = tl.sum(tl.reshape(total, [2, 128]), axis=0)
    if BW >= 128:
        total = tl.sum(tl.reshape(total, [2, 64]), axis=0)
    if BW >= 64:
        total = tl.sum(tl.reshape(total, [2, 32]), axis=0)
    total = tl.sum(tl.reshape(total, [16, 2]), axis=1)
    total = tl.sum(tl.reshape(total, [8, 2]), axis=1)
    total = tl.sum(tl.reshape(total, [4, 2]), axis=1)
    total = tl.sum(tl.reshape(total, [2, 2]), axis=1)
    return tl.sum(total, axis=0)


@triton.jit
def _residual_norm(Delta, Hidden, PostWeight, NextWeight, Start, Blend, HiddenOut, NormedOut, LerpOut,
                   eps, factor, HIDDEN: tl.constexpr, BW: tl.constexpr, DELTA: tl.constexpr,
                   LERP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * HIDDEN
    columns = tl.arange(0, BLOCK)
    if DELTA:
        mean = _square_sum(Delta + row, HIDDEN // 4, BW) * factor
        inverse = libdevice.rsqrt(mean + eps)
        for chunk in tl.static_range(0, HIDDEN, BLOCK):
            index = chunk + columns
            valid = index < HIDDEN
            delta = tl.load(Delta + row + index, valid, other=0.).to(tl.float32)
            normalized = (delta * inverse).to(tl.bfloat16).to(tl.float32)
            weight = tl.load(PostWeight + index, valid, other=0.).to(tl.float32)
            update = (weight * normalized).to(tl.bfloat16).to(tl.float32)
            residual = tl.load(Hidden + row + index, valid, other=0.).to(tl.float32)
            tl.store(HiddenOut + row + index, (residual + update).to(tl.bfloat16), valid)
        # The updated row is reread by other lanes of this program.
        tl.debug_barrier()
        source = HiddenOut
    else:
        source = Hidden
    mean = _square_sum(source + row, HIDDEN // 4, BW) * factor
    inverse = libdevice.rsqrt(mean + eps)
    if LERP:
        blend = tl.load(Blend).to(tl.float32)
        far = tl.abs(blend) >= 0.5
        coefficient = tl.where(far, -(1.0 - blend), blend)
    for chunk in tl.static_range(0, HIDDEN, BLOCK):
        index = chunk + columns
        valid = index < HIDDEN
        value = tl.load(source + row + index, valid, other=0.).to(tl.float32)
        normalized = (value * inverse).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(NextWeight + index, valid, other=0.).to(tl.float32)
        output = (weight * normalized).to(tl.bfloat16)
        tl.store(NormedOut + row + index, output, valid)
        if LERP:
            end = output.to(tl.float32)
            start = tl.load(Start + row + index, valid, other=0.).to(tl.float32)
            lerped = tl.fma(coefficient, end - start, tl.where(far, end, start))
            tl.store(LerpOut + row + index, lerped.to(tl.bfloat16), valid)


def residual_norm(delta, hidden, post_weight, weight, eps, *, start=None, blend=None):
    """`hidden + rms_norm(delta, post_weight)` (when delta is given) and its `rms_norm(., weight)`.

    With `start` and `blend`, also returns `torch.lerp(start, normalized, blend)`. Returns the
    residual (the input hidden without delta), the normalized rows and the lerp (or None).
    """
    rows, width = hidden.shape
    if not 0 < rows <= MAX_ROWS or not hidden.is_contiguous() or hidden.dtype != torch.bfloat16:
        raise ValueError("fused residual RMSNorm takes at most MAX_ROWS contiguous BF16 rows")
    for tensor in (delta, start):
        if tensor is not None and (tensor.shape != hidden.shape or not tensor.is_contiguous()
                                   or tensor.dtype != torch.bfloat16):
            raise ValueError("fused residual RMSNorm operands must match the hidden rows")
    lerp = start is not None
    if lerp != (blend is not None):
        raise ValueError("a value lerp needs both its start rows and its blend")
    updated = hidden if delta is None else torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    lerped = torch.empty_like(hidden) if lerp else normed
    # ATen's MeanOps factor float(outputs) / numel is fl32(1 / width); the kernel's FP32 scalar
    # argument rounds this double to the same value (checked for 5120 and powers of two).
    factor = 1.0 / width
    torch.library.wrap_triton(_residual_norm)[(rows,)](
        hidden if delta is None else delta, hidden, weight if post_weight is None else post_weight, weight,
        start if lerp else hidden, blend if lerp else weight, updated, normed, lerped,
        float(eps), factor, width, aten_block_width(rows, width), delta is not None, lerp, _BLOCK,
        num_warps=4, enable_fp_fusion=False)
    return updated, normed, lerped if lerp else None
