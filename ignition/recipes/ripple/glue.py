# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ripple's per-layer glue as three row-count-invariant Triton kernels (2026-11-09).

The staged forward runs each RMSNorm, the q/k norms with RoPE, the KV copies and the attention gate
as about 25 small launches per layer, and ATen's mean picks its reduction order from the row count.
These kernels do the same arithmetic as `modeling_ricursive.py` (FP32 sum of squares times 1/width,
rsqrt, BF16 normalized value, BF16 gain product, BF16 residual sum, FP32 RoPE on BF16-rounded tables,
FP32 sigmoid gate rounded to BF16) with one fixed FP32 summation order per row that does not depend
on how many rows the launch has. They are not bit-identical to ATen's mean (accumulation order only:
numerics class B), but a row's result is the same at C1, C4 and in prefill.

- `residual_norm`: optional `hidden + norm(delta, post)`, then `norm(., weight)` and the value lerp.
- `qk_rope`: q/k RMSNorm per head, RoPE on sliding layers, and the q, key and value buffer writes.
- `gate`: `attended * bf16(sigmoid(fp32(gate_logits)))`.
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

_BLOCK = 1024


@triton.jit
def _sum_squares(row, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    columns = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for chunk in tl.static_range(0, HIDDEN, BLOCK):
        index = chunk + columns
        x = tl.load(row + index, index < HIDDEN, other=0.).to(tl.float32)
        acc = acc + x * x
    return tl.sum(acc, axis=0)


@triton.jit
def _residual_norm(Delta, Hidden, PostWeight, NextWeight, Start, Blend, HiddenOut, NormedOut, LerpOut,
                   eps, factor, HIDDEN: tl.constexpr, DELTA: tl.constexpr, LERP: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * HIDDEN
    columns = tl.arange(0, BLOCK)
    if DELTA:
        inverse = libdevice.rsqrt(_sum_squares(Delta + row, HIDDEN, BLOCK) * factor + eps)
        for chunk in tl.static_range(0, HIDDEN, BLOCK):
            index = chunk + columns
            valid = index < HIDDEN
            delta = tl.load(Delta + row + index, valid, other=0.).to(tl.float32)
            normalized = (delta * inverse).to(tl.bfloat16).to(tl.float32)
            weight = tl.load(PostWeight + index, valid, other=0.).to(tl.float32)
            update = (weight * normalized).to(tl.bfloat16).to(tl.float32)
            residual = tl.load(Hidden + row + index, valid, other=0.).to(tl.float32)
            tl.store(HiddenOut + row + index, (residual + update).to(tl.bfloat16), valid)
        # The updated row is reread below by other threads of this program.
        tl.debug_barrier()
        source = HiddenOut
    else:
        source = Hidden
    inverse = libdevice.rsqrt(_sum_squares(source + row, HIDDEN, BLOCK) * factor + eps)
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
    """Same contract as `decoder.staged_residual_norm`: (hidden after the update, normed, lerp or None)."""
    rows, width = hidden.shape
    for tensor in (hidden, delta, start):
        if tensor is not None and (tensor.shape != hidden.shape or tensor.stride(-1) != 1
                                   or tensor.stride(0) != width or tensor.dtype != torch.bfloat16):
            raise ValueError("fused residual RMSNorm takes contiguous BF16 rows of one shape")
    lerp = start is not None
    if lerp != (blend is not None):
        raise ValueError("a value lerp needs both its start rows and its blend")
    updated = hidden if delta is None else torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    lerped = torch.empty_like(hidden) if lerp else normed
    torch.library.wrap_triton(_residual_norm)[(rows,)](
        hidden if delta is None else delta, hidden, weight if post_weight is None else post_weight, weight,
        start if lerp else hidden, blend if lerp else weight, updated, normed, lerped,
        float(eps), 1.0 / width, width, delta is not None, lerp, _BLOCK, num_warps=4, enable_fp_fusion=False)
    return updated, normed, lerped if lerp else None


@triton.jit
def _qk_rope(QGK, V, QNorm, KNorm, Cos, Sin, Q, Key, Value, eps, factor, qgk_stride,
             HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr, K_OFFSET: tl.constexpr,
             ROPE: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    HALF: tl.constexpr = DIM // 2
    half = tl.arange(0, HALF)
    if head < HEADS:
        source = QGK + token * qgk_stride + head * DIM
        gain = QNorm
        target = Q + (token * HEADS + head) * DIM
    else:
        kv = head - HEADS
        source = QGK + token * qgk_stride + K_OFFSET + kv * DIM
        gain = KNorm
        target = Key + (token * KV_HEADS + kv) * DIM
        value = V + (token * KV_HEADS + kv) * DIM
        out = Value + (token * KV_HEADS + kv) * DIM
        tl.store(out + half, tl.load(value + half))
        tl.store(out + HALF + half, tl.load(value + HALF + half))
    left = tl.load(source + half).to(tl.float32)
    right = tl.load(source + HALF + half).to(tl.float32)
    total = tl.sum(left * left, axis=0) + tl.sum(right * right, axis=0)
    inverse = libdevice.rsqrt(total * factor + eps)
    left = (tl.load(gain + half).to(tl.float32) * (left * inverse).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    right = (tl.load(gain + HALF + half).to(tl.float32)
             * (right * inverse).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    if ROPE:
        lf = left.to(tl.float32)
        rf = right.to(tl.float32)
        table = token * DIM
        left = (lf * tl.load(Cos + table + half) + (-rf) * tl.load(Sin + table + half)).to(tl.bfloat16)
        right = (rf * tl.load(Cos + table + HALF + half) + lf * tl.load(Sin + table + HALF + half)).to(tl.bfloat16)
    tl.store(target + half, left)
    tl.store(target + HALF + half, right)


def qk_rope(qgk, v, q_norm, k_norm, cos, sin, q, key, value, eps, *, rope):
    """Writes q = rope(norm(q)), key = rope(norm(k)) (no RoPE when `rope` is False) and value = v."""
    tokens, heads, dim = q.shape
    kv_heads = key.shape[1]
    if qgk.stride(-1) != 1 or v.stride(-1) != 1 or v.stride(0) != kv_heads * dim:
        raise ValueError("q/k RoPE reads row-contiguous qgk and v")
    if rope and (cos.dtype != torch.float32 or cos.stride(0) != dim or sin.stride(0) != dim):
        raise ValueError("RoPE tables are contiguous FP32 rows of the head dimension")
    torch.library.wrap_triton(_qk_rope)[(tokens, heads + kv_heads)](
        qgk, v, q_norm, k_norm, cos if rope else q_norm, sin if rope else q_norm, q, key, value,
        float(eps), 1.0 / dim, qgk.stride(0), heads, kv_heads, dim, 2 * heads * dim, rope,
        num_warps=1, enable_fp_fusion=False)


@triton.jit
def _gate(Attended, QGK, Out, a_stride, qgk_stride, G_OFFSET: tl.constexpr, WIDTH: tl.constexpr,
          BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = index < WIDTH
    logits = tl.load(QGK + token * qgk_stride + G_OFFSET + index, valid, other=0.).to(tl.float32)
    gate = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-logits)).to(tl.bfloat16).to(tl.float32)
    attended = tl.load(Attended + token * a_stride + index, valid, other=0.).to(tl.float32)
    tl.store(Out + token * WIDTH + index, (attended * gate).to(tl.bfloat16), valid)


def gate(attended, qgk, offset):
    """`attended * bf16(sigmoid(fp32(qgk[:, offset:offset + width])))` as new contiguous rows."""
    tokens, width = attended.shape
    out = torch.empty((tokens, width), dtype=attended.dtype, device=attended.device)
    block = 1024
    torch.library.wrap_triton(_gate)[(tokens, triton.cdiv(width, block))](
        attended, qgk, out, attended.stride(0), qgk.stride(0), offset, width, block, num_warps=4,
        enable_fp_fusion=False)
    return out
