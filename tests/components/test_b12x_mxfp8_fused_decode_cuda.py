# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""b12x's fused MXFP8 A16 GEMM decodes weights exactly as its FP32-path predecessor did.

The fused kernel converts E4M3 pairs with ``cvt.rn.bf16x2.e4m3x2`` and applies the UE8M0 scale
with ``mul.rn.bf16x2``. The previous kernel computed ``fp32(e4m3) * ex2.approx.ftz(e - 127)`` and
rounded to BF16. These tests run both decodes over every code/exponent pair and both GEMMs over
the Ripple decode configurations, and demand bitwise equality.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="set FLARE_RUN_CUDA_TESTS=1 to run CUDA kernels")

triton = pytest.importorskip("triton")
tl = triton.language
fused = pytest.importorskip("flashinfer.experimental.b12x.gemm.blockscaled._fused")
_decode_tile = fused._decode_tile
_scale_offset = pytest.importorskip("flashinfer.experimental.b12x.gemm.blockscaled._quantize")._scale_offset


@triton.jit
def _fp32_decode(w, e, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, G: tl.constexpr):
    # The FP32 decode the fused kernel shipped with before (b12x 9538ff2).
    scale = tl.exp2(e.to(tl.float32) - 127.0)
    decoded = tl.reshape(w.to(tl.float32), (BLOCK_N, G, 32)) * scale[:, :, None]
    return tl.reshape(decoded, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)


@triton.jit
def _table(OLD, NEW):
    # all 256 E4M3 codes (columns) x all 256 UE8M0 exponents (rows, 8 groups of 32 codes each)
    rn = tl.arange(0, 256)
    rk = tl.arange(0, 256)
    w = (rk[None, :] + 0 * rn[:, None]).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    e = (rn[:, None] + 0 * tl.arange(0, 8)[None, :]).to(tl.uint8)
    tl.store(OLD + rn[:, None] * 256 + rk[None, :], _fp32_decode(w, e, 256, 256, 8))
    tl.store(NEW + rn[:, None] * 256 + rk[None, :], _decode_tile(w, e, 256, 256, 8))


def test_decode_table_is_bitwise_equal():
    old = torch.empty(256, 256, device="cuda", dtype=torch.bfloat16)
    new = torch.empty_like(old)
    _table[(1,)](old, new, num_warps=8)
    codes = torch.arange(256, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(codes)[None, :].expand(256, 256)
    exps = torch.arange(256, device="cuda")[:, None].expand(256, 256)
    usable = finite & (exps < 255)  # 255 is UE8M0 NaN
    o, n = old.view(torch.int16), new.view(torch.int16)
    assert torch.equal(o[usable], n[usable])
    # Away from the flushed e = 0 row both equal the exactly rounded product.
    exact = (codes.double()[None, :] * torch.pow(2.0, exps.double() - 127)).to(torch.bfloat16)
    rows = usable & (exps > 0)
    assert torch.equal(n[rows], exact.view(torch.int16)[rows])


@triton.jit(do_not_specialize=["M"])
def _fp32_path_gemm(A, W, S, P, M, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(2)
    split = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    G: tl.constexpr = BLOCK_K // 32
    rg = tl.arange(0, G)
    KS: tl.constexpr = K // SPLIT
    k_first = split * KS
    a_ptrs = A + rm.to(tl.int64)[:, None] * K + (k_first + rk)[None, :]
    w_ptrs = W + rn.to(tl.int64)[:, None] * K + (k_first + rk)[None, :]
    rows = rm[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_first, k_first + KS, BLOCK_K):
        a = tl.load(a_ptrs, mask=rows, other=0.0)
        w = tl.load(w_ptrs)
        e = tl.load(S + _scale_offset(rn[:, None], (k0 // 32 + rg)[None, :], K // 32))
        acc = tl.dot(a, tl.trans(_fp32_decode(w, e, BLOCK_N, BLOCK_K, G)), acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    tl.store(P + (split * M + rm.to(tl.int64))[:, None] * N + rn[None, :], acc, mask=rows)


# (rows, N, K, tile_m, tile_n, tile_k, split, warps): the served Ripple decode/verify choices.
CASES = [(16, 81920, 5120, 16, 128, 128, 1, 4), (16, 5120, 40960, 16, 128, 128, 8, 4),
         (13, 5120, 10240, 16, 128, 128, 4, 4), (25, 81920, 5120, 32, 64, 128, 1, 4),
         (64, 5120, 40960, 64, 64, 128, 4, 4), (47, 21504, 5120, 64, 64, 128, 1, 4)]


@pytest.mark.parametrize("rows,n,k,tile_m,tile_n,tile_k,split,warps", CASES)
def test_fused_gemm_matches_fp32_decode_path(rows, n, k, tile_m, tile_n, tile_k, split, warps):
    g = torch.Generator(device="cuda").manual_seed(rows * 31 + n)
    a = (torch.randn(rows, k, device="cuda", generator=g) * 3).bfloat16()
    w = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.uint8, generator=g)
    w[(w & 0x7F) == 0x7F] = 0x3C  # no NaN codes; zeros and subnormals stay
    nt, kt = triton.cdiv(n, 128), triton.cdiv(k // 32, 4)
    s = torch.randint(110, 136, (nt * kt * 512,), device="cuda", dtype=torch.uint8, generator=g)
    w = w.view(torch.float8_e4m3fn)
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    parts = torch.empty(split * rows * n, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(rows, tile_m) * (n // tile_n), split, 1)
    fused._a16_mxfp8_fused[grid](a, w, s, out, parts, rows, n, k, tile_m, tile_n, tile_k,
                                 fused.FUSED_GROUP_M, split, num_warps=warps, num_stages=3)
    if split > 1:
        fused._sum_split_partials[(rows, triton.cdiv(n, fused.FUSED_REDUCE_BLOCK), 1)](
            parts, out, rows, n, split, fused.FUSED_REDUCE_BLOCK, num_warps=4, num_stages=1)
    ref_parts = torch.empty(split, rows, n, device="cuda", dtype=torch.float32)
    _fp32_path_gemm[(n // tile_n, split, triton.cdiv(rows, tile_m))](
        a, w, s, ref_parts, rows, n, k, tile_m, tile_n, tile_k, split,
        num_warps=warps, num_stages=3)
    ref = ref_parts[0].clone()
    for part in ref_parts[1:]:
        ref += part
    assert torch.equal(out.view(torch.int16), ref.bfloat16().view(torch.int16))
