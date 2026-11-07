# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 full-context draft attention against an eager FP32 reference."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ignition.components import dflash2_full_attention as full

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                          reason="requires an explicitly assigned CUDA device")


def _case(device, kv_dtype, starts, lengths, *, dtype=torch.bfloat16, heads=20, kv_heads=2, dim=128, page=128, rows=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    requests = len(starts)
    blocks = max(-(-(start + rows) // page) for start in starts)
    pages = requests * blocks + 3
    order = torch.randperm(pages, generator=generator)
    table = torch.full((requests, blocks), -1, dtype=torch.int32)
    for request, start in enumerate(starts):
        used = -(-start // page)
        table[request, :used] = order[request * blocks:request * blocks + used].int()
    tokens = requests * rows
    first = torch.zeros(requests + 1, dtype=torch.int32)
    first[1:] = torch.tensor(lengths).cumsum(0)

    def normal(*shape):
        return torch.randn(*shape, generator=generator) * 2
    keys = normal(pages, page, kv_heads, dim).to(kv_dtype)
    values = normal(pages, page, kv_heads, dim).to(kv_dtype)
    move = dict(device=device)
    return dict(query=normal(tokens, heads, dim).to(dtype).to(**move), key_pages=keys.to(**move),
                value_pages=values.to(**move), page_table=table.to(**move),
                starts=torch.tensor(starts, dtype=torch.int64, device=device),
                lengths=torch.tensor(lengths, dtype=torch.int32, device=device), query_start=first.to(**move),
                block_key=normal(tokens, kv_heads, dim).to(dtype).to(**move),
                block_value=normal(tokens, kv_heads, dim).to(dtype).to(**move))


def _check(device, kv_dtype, starts, lengths, *, dtype=torch.bfloat16):
    case = _case(device, kv_dtype, starts, lengths, dtype=dtype)
    query = case["query"]
    tokens, heads, dim = query.shape
    output = torch.full_like(query, 7.)
    accumulators = torch.full((full.SPLITS + 1, tokens, heads, dim), float("nan"), device=device)
    lse = torch.full((full.SPLITS + 1, tokens, heads), float("nan"), device=device)
    full.attend(**case, output=output, accumulators=accumulators, lse=lse, rows=16)
    expected = full.reference(**case)
    for request, count in enumerate(lengths):
        first = int(case["query_start"][request])
        rows = slice(first, first + count)
        torch.testing.assert_close(output[rows].float(), expected[rows], atol=3e-2, rtol=3e-2)
    # Rows past the live queries are untouched.
    assert (output[int(case["query_start"][-1]):] == 7.).all()


def test_full_attention_matches_reference_on_the_interpreter():
    # The interpreter patches triton.language at import and lacks BF16 arithmetic, so it runs
    # the same kernels on FP32 operands in a fresh process.
    code = ("import torch, test_dflash2_full_attention as t; "
            "t._check('cpu', torch.float32, (0, 5, 127, 300), (16, 16, 16, 9), dtype=torch.float32); "
            "t._check('cpu', torch.float32, (2049, 4100), (16, 0), dtype=torch.float32)")
    path = os.pathsep.join((str(Path(__file__).parent), str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH", "")))
    result = subprocess.run([sys.executable, "-c", code], env=os.environ | {"TRITON_INTERPRET": "1", "PYTHONPATH": path},
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]


@CUDA
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("starts,lengths", [((0, 5, 127, 300), (16, 16, 16, 9)),
                                            ((2049, 4100, 33000, 1), (16, 0, 16, 16))])
def test_full_attention_matches_reference_on_cuda(kv_dtype, starts, lengths):
    _check("cuda", kv_dtype, starts, lengths)

