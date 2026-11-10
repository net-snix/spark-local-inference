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

On SM90+ every kernel launches programmatically (PDL, 2026-11-09): it may start while the previous kernel
in the stream still runs. Before its `griddepcontrol.wait` it only warms the L2 lines of the inputs a
latency-bound row walk would otherwise miss one after another (residual rows, norm gains, RoPE tables);
L2 is the coherence point, so warming a line another grid may still write is a hint, never a stale read.
Every load and store follows the wait, then the kernel releases the next programmatic launch. Values are
unchanged (bitwise) with or without PDL.

The launches are opaque custom ops with declared outputs: Inductor's mutation analysis of user Triton
kernels cannot read the `griddepcontrol`/prefetch asm and would then treat every argument as written
(extra clones, and wrong results where one buffer fills two parameters).
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

_BLOCK = 1024


# Decode-sized launches only (2026-11-10): prefill chunks gain nothing from an early launch, and the
# pre-wait warming of thousands of rows competes with the GEMM still running (WS PDL build: prefill -1.2%).
_PDL_MAX_ROWS = 64


def _pdl(tensor):
    """Programmatic launches (and their pre-wait L2 warming) need SM90+, for decode-sized row counts."""
    return tensor.shape[0] <= _PDL_MAX_ROWS and torch.cuda.get_device_capability(tensor.device)[0] >= 9


@triton.jit
def _warm(address):
    tl.inline_asm_elementwise("prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l", [address.to(tl.int64)],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _warm_row(row, BYTES: tl.constexpr, ELEMENT: tl.constexpr, WIDTH: tl.constexpr):
    """One L2 prefetch per 128-byte line of a contiguous row of BYTES (ELEMENT bytes each), WIDTH per pass."""
    LINES: tl.constexpr = (BYTES + 127) // 128
    for first in tl.static_range(0, LINES, WIDTH):
        line = tl.minimum(first + tl.arange(0, WIDTH), LINES - 1)
        _warm(row + line.to(tl.int64) * (128 // ELEMENT))


@triton.jit
def _enter(PDL: tl.constexpr):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


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
                   BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64) * HIDDEN
    if PDL:
        # The residual row, the gains and the lerp start are each walked chunk by chunk below.
        _warm_row(Hidden + row, HIDDEN * 2, 2, 128)
        _warm_row(NextWeight, HIDDEN * 2, 2, 128)
        if DELTA:
            _warm_row(PostWeight, HIDDEN * 2, 2, 128)
        if LERP:
            _warm_row(Start + row, HIDDEN * 2, 2, 128)
    _enter(PDL)
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
    out = _residual_norm_rows(delta, hidden, post_weight, weight, start, blend, float(eps))
    return hidden if delta is None else out[0], out[-2 if lerp else -1], out[-1] if lerp else None


def _new_rows(delta, hidden, start):
    return [torch.empty_like(hidden) for _ in range(1 + (delta is not None) + (start is not None))]


@torch.library.custom_op("flare::ripple_residual_norm", mutates_args=())
def _residual_norm_rows(delta: torch.Tensor | None, hidden: torch.Tensor, post_weight: torch.Tensor | None,
                        weight: torch.Tensor, start: torch.Tensor | None, blend: torch.Tensor | None,
                        eps: float) -> list[torch.Tensor]:
    """New rows: the updated hidden (with a delta), the normed rows, then the lerp (with a start)."""
    rows, width = hidden.shape
    out = _new_rows(delta, hidden, start)
    updated = hidden if delta is None else out[0]
    normed = out[0 if delta is None else 1]
    lerped = normed if start is None else out[-1]
    pdl = _pdl(hidden)
    _residual_norm[(rows,)](
        hidden if delta is None else delta, hidden, weight if post_weight is None else post_weight, weight,
        hidden if start is None else start, weight if blend is None else blend, updated, normed, lerped,
        eps, 1.0 / width, width, delta is not None, start is not None, _BLOCK, PDL=pdl, num_warps=4,
        enable_fp_fusion=False, launch_pdl=pdl)
    return out


@_residual_norm_rows.register_fake
def _fake_residual_norm_rows(delta, hidden, post_weight, weight, start, blend, eps):
    return _new_rows(delta, hidden, start)


@triton.jit
def _qk_rope(QGK, V, QNorm, KNorm, Cos, Sin, Q, Key, Value, eps, factor, qgk_stride,
             HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr, K_OFFSET: tl.constexpr,
             ROPE: tl.constexpr, PDL: tl.constexpr = False):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    HALF: tl.constexpr = DIM // 2
    half = tl.arange(0, HALF)
    if PDL:
        if head < HEADS:
            _warm_row(QNorm, DIM * 2, 2, 4)
        else:
            _warm_row(KNorm, DIM * 2, 2, 4)
        if ROPE:
            _warm_row(Cos + token * DIM, DIM * 4, 4, 4)
            _warm_row(Sin + token * DIM, DIM * 4, 4, 4)
    _enter(PDL)
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
    _qk_rope_rows(qgk, v, q_norm, k_norm, cos if rope else None, sin if rope else None, q, key, value,
                  float(eps))


@torch.library.custom_op("flare::ripple_qk_rope", mutates_args=("q", "key", "value"))
def _qk_rope_rows(qgk: torch.Tensor, v: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor,
                  cos: torch.Tensor | None, sin: torch.Tensor | None, q: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, eps: float) -> None:
    """The q/k RoPE launch (RoPE when the tables are given)."""
    tokens, heads, dim = q.shape
    kv_heads = key.shape[1]
    rope = cos is not None
    pdl = _pdl(qgk)
    _qk_rope[(tokens, heads + kv_heads)](
        qgk, v, q_norm, k_norm, cos if rope else q_norm, sin if rope else q_norm, q, key, value,
        eps, 1.0 / dim, qgk.stride(0), heads, kv_heads, dim, 2 * heads * dim, rope, PDL=pdl,
        num_warps=1, enable_fp_fusion=False, launch_pdl=pdl)


@_qk_rope_rows.register_fake
def _fake_qk_rope_rows(qgk, v, q_norm, k_norm, cos, sin, q, key, value, eps):
    pass


@triton.jit
def _gate(Attended, QGK, Out, a_stride, qgk_stride, G_OFFSET: tl.constexpr, WIDTH: tl.constexpr,
          BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    _enter(PDL)
    token = tl.program_id(0).to(tl.int64)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = index < WIDTH
    logits = tl.load(QGK + token * qgk_stride + G_OFFSET + index, valid, other=0.).to(tl.float32)
    gate = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-logits)).to(tl.bfloat16).to(tl.float32)
    attended = tl.load(Attended + token * a_stride + index, valid, other=0.).to(tl.float32)
    tl.store(Out + token * WIDTH + index, (attended * gate).to(tl.bfloat16), valid)


@torch.library.custom_op("flare::ripple_gate", mutates_args=())
def gate(attended: torch.Tensor, qgk: torch.Tensor, offset: int) -> torch.Tensor:
    """`attended * bf16(sigmoid(fp32(qgk[:, offset:offset + width])))` as new contiguous rows."""
    tokens, width = attended.shape
    out = attended.new_empty((tokens, width))
    block = 1024
    pdl = _pdl(attended)
    _gate[(tokens, triton.cdiv(width, block))](
        attended, qgk, out, attended.stride(0), qgk.stride(0), offset, width, block, PDL=pdl, num_warps=4,
        enable_fp_fusion=False, launch_pdl=pdl)
    return out


@gate.register_fake
def _fake_gate(attended, qgk, offset):
    return attended.new_empty(attended.shape)
