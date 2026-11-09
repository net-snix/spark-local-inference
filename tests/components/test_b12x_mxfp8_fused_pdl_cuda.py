# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""b12x's fused MXFP8 A16 GEMM and its split-K sum launch programmatically (PDL) with the same bits.

The plan path (compile_fused/run_fused) compiles both kernels with launch_pdl and a griddepcontrol.wait
before their first access. Outputs must equal the non-PDL launch bitwise, also inside a CUDA graph where
the activation producer releases the GEMM at its start and only then writes the activation rows (a read
before the wait would see stale rows), and an earlier kernel still reads the output buffer the GEMM
overwrites (a write before the transitive wait would corrupt its copy).
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="set FLARE_RUN_CUDA_TESTS=1 to run CUDA kernels")

triton = pytest.importorskip("triton")
tl = triton.language
fused = pytest.importorskip("flashinfer.experimental.b12x.gemm.blockscaled._fused")
fused_stages = pytest.importorskip("flashinfer.experimental.b12x.gemm.blockscaled._fused_tuning").fused_stages

# (rows, N, K, tile_m, tile_n, tile_k, split, warps): Ripple decode pins (C1 = 6 rows) and verify shapes.
CASES = [(6, 81920, 5120, 16, 128, 128, 1, 4), (6, 5120, 40960, 16, 128, 128, 8, 4),
         (16, 5120, 10240, 16, 128, 128, 4, 4), (25, 81920, 5120, 32, 64, 128, 1, 4),
         (47, 21504, 5120, 64, 64, 128, 1, 4), (64, 5120, 40960, 64, 64, 128, 4, 4)]


@triton.jit
def _spin(NS: tl.constexpr):
    t0 = tl.extra.cuda.globaltimer()
    t = t0
    while t - t0 < NS:
        t = tl.extra.cuda.globaltimer()


@triton.jit
def _late_copy(X, Y, NUMEL: tl.constexpr, BLOCK: tl.constexpr, NS: tl.constexpr):
    """Release the next programmatic launch at once, wait for this grid's own producer, then copy X to Y late."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    _spin(NS)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < NUMEL), i < NUMEL)


def _data(rows, n, k, split, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(rows, k, device="cuda", generator=g) * 3).bfloat16()
    w = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.uint8, generator=g)
    w[(w & 0x7F) == 0x7F] = 0x3C
    s = torch.randint(110, 136, (triton.cdiv(n, 128) * triton.cdiv(k // 32, 4) * 512,), device="cuda",
                      dtype=torch.uint8, generator=g)
    return a, w.view(torch.float8_e4m3fn), s, torch.zeros(split * rows * n * 4, device="cuda", dtype=torch.uint8)


def _plan(rows, n, k, tile_m, tile_n, tile_k, split, warps):
    query = SimpleNamespace(out_features=n, padded_in_features=k, num_tokens=rows)
    config = SimpleNamespace(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, split_k=split, mma_warps=warps)
    return fused.compile_fused(query, config), config


def _reference(a, w, s, rows, n, k, tile_m, tile_n, tile_k, split, warps):
    """The non-PDL launch (the kernels' default)."""
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    parts = torch.empty(split * rows * n, device="cuda", dtype=torch.float32)
    stages = fused_stages(SimpleNamespace(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, split_k=split))
    fused._a16_mxfp8_fused[(triton.cdiv(rows, tile_m) * (n // tile_n), split, 1)](
        a, w, s, out, parts, rows, n, k, tile_m, tile_n, tile_k, fused.FUSED_GROUP_M, split,
        num_warps=warps, num_stages=stages)
    if split > 1:
        fused._sum_split_partials[(rows, triton.cdiv(n, fused.FUSED_REDUCE_BLOCK), 1)](
            parts, out, rows, n, split, fused.FUSED_REDUCE_BLOCK, num_warps=4, num_stages=1)
    return out


def test_plan_compiles_programmatic_launches():
    programs, _ = _plan(6, 5120, 40960, 16, 128, 128, 8, 4)
    assert programs["fused"].metadata.launch_pdl and programs["fused_reduce"].metadata.launch_pdl


@pytest.mark.parametrize("rows,n,k,tile_m,tile_n,tile_k,split,warps", CASES)
def test_pdl_launch_is_bitwise_equal(rows, n, k, tile_m, tile_n, tile_k, split, warps):
    a, w, s, workspace = _data(rows, n, k, split, rows * 31 + n)
    want = _reference(a, w, s, rows, n, k, tile_m, tile_n, tile_k, split, warps)
    programs, config = _plan(rows, n, k, tile_m, tile_n, tile_k, split, warps)
    got = torch.empty_like(want)
    fused.run_fused(programs, config, a, w, s, got, workspace, rows)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("rows,n,k,tile_m,tile_n,tile_k,split,warps", [CASES[0], CASES[1], CASES[2]])
def test_pdl_graph_orders_late_producer_and_reader(rows, n, k, tile_m, tile_n, tile_k, split, warps):
    a, w, s, workspace = _data(rows, n, k, split, rows * 17 + n)
    want = _reference(a, w, s, rows, n, k, tile_m, tile_n, tile_k, split, warps)
    programs, config = _plan(rows, n, k, tile_m, tile_n, tile_k, split, warps)
    staged = torch.zeros_like(a)                       # the producer writes the GEMM's activation late
    out = torch.full((rows, n), 7.0, device="cuda", dtype=torch.bfloat16)
    seen = torch.zeros_like(out)                       # an earlier reader copies the old output late
    block = 1024

    def chain():
        staged.zero_()
        out.fill_(7.0)
        _late_copy[(triton.cdiv(out.numel(), block),)](out, seen, out.numel(), block, 20000, launch_pdl=True)
        _late_copy[(triton.cdiv(a.numel(), block),)](a, staged, a.numel(), block, 20000, launch_pdl=True)
        fused.run_fused(programs, config, staged, w, s, out, workspace, rows)

    chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    for _ in range(3):
        seen.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), want.view(torch.int16))
        assert bool((seen == 7.0).all())
