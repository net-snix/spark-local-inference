# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""b12x's fused MXFP8 A16 GEMM gives the same bits for every launch geometry with the same split-K.

Each output element is one FP32 chain of BF16 mma.sync steps over its K slice in k16 order, with the
operands' K slots fixed by the decoded weight's dot layout (kWidth 4 for every tile). Tile M/N/K, warp
count and pipeline depth only change which CTA and warp run a chain and how loads are staged, so a
selection-pin retune that keeps split-K (and the fused backend) keeps decode class A. Split-K changes
the chain and is covered as the negative control.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="set FLARE_RUN_CUDA_TESTS=1 to run CUDA kernels")

triton = pytest.importorskip("triton")
fused = pytest.importorskip("flashinfer.experimental.b12x.gemm.blockscaled._fused")
tuning = pytest.importorskip("flashinfer.experimental.b12x.gemm.blockscaled._fused_tuning")


def operands(rows, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = torch.exp(torch.randn(rows, 1, device="cuda", generator=g))
    a = (torch.randn(rows, k, device="cuda", generator=g) * scale).bfloat16()
    w = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.uint8, generator=g)
    w[(w & 0x7F) == 0x7F] = 0x3C  # no NaN codes; zeros and subnormals stay
    s = torch.randint(110, 136, (triton.cdiv(n, 128) * triton.cdiv(k // 32, 4) * 512,), device="cuda",
                      dtype=torch.uint8, generator=g)
    return a, w.view(torch.float8_e4m3fn), s


def run(a, w, s, tile_m, tile_n, tile_k, split, warps, stages=None):
    rows, k = a.shape
    n = w.shape[0]
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    parts = torch.empty(split * rows * n, device="cuda", dtype=torch.float32)
    tile = type("Tile", (), dict(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, split_k=split))
    grid = (triton.cdiv(rows, tile_m) * (n // tile_n), split, 1)
    fused._a16_mxfp8_fused[grid](a, w, s, out, parts, rows, n, k, tile_m, tile_n, tile_k, fused.FUSED_GROUP_M,
                                 split, num_warps=warps, num_stages=stages or tuning.fused_stages(tile))
    if split > 1:
        fused._sum_split_partials[(rows, triton.cdiv(n, fused.FUSED_REDUCE_BLOCK), 1)](
            parts, out, rows, n, split, fused.FUSED_REDUCE_BLOCK, num_warps=4, num_stages=1)
    return out.view(torch.int16)


# (rows, N, K, split, served tile) from the Ripple verify buckets; the alternatives span tile_m, tile_n,
# tile_k (scale groups of 2/4/8 per tile), warps and stages.
CASES = [(16, 4096, 5120, 1, (16, 128, 128, 4)), (16, 5120, 40960, 8, (16, 128, 128, 4)),
         (64, 4096, 5120, 1, (64, 64, 128, 4)), (47, 5120, 40960, 4, (64, 64, 128, 4)),
         (25, 2048, 5120, 1, (32, 64, 128, 4)), (13, 5120, 10240, 4, (16, 128, 128, 4))]
# All within b12x FUSED_SMEM_LIMIT (stages x (tm tk 2 + tn tk) + tn tk 2 <= 96 KiB).
ALTERNATIVES = [(16, 64, 64, 4), (16, 256, 64, 4), (32, 128, 64, 8), (64, 128, 64, 4), (32, 256, 64, 8),
                (32, 128, 128, 4, 2), (64, 64, 128, 8, 3)]


@pytest.mark.parametrize("rows,n,k,split,served", CASES)
def test_launch_geometry_keeps_bits(rows, n, k, split, served):
    a, w, s = operands(rows, n, k, seed=rows * 7 + n)
    reference = run(a, w, s, *served[:3], split, served[3])
    checked = 0
    for alt in ALTERNATIVES:
        tile_m, tile_n, tile_k, warps = alt[:4]
        if n % tile_n or (k // split) % tile_k:
            continue
        got = run(a, w, s, tile_m, tile_n, tile_k, split, warps, alt[4] if len(alt) > 4 else None)
        assert torch.equal(got, reference), (alt, int((got != reference).sum()))
        checked += 1
    assert checked >= 4


def test_split_k_changes_the_chain():
    # Negative control: another K split reorders the FP32 sums, so some outputs must differ.
    a, w, s = operands(16, 4096, 40960, seed=3)
    assert not torch.equal(run(a, w, s, 16, 128, 128, 8, 4), run(a, w, s, 16, 128, 128, 4, 4))
