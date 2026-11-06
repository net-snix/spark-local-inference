# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Continuation-row copies cover rows longer than one CUDA grid dimension holds."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="explicit CUDA assignment required")


def test_row_copies_and_resets_loop_over_tiles_beyond_the_launched_programs():
    from ignition.components.state_transfer import _ROW_BLOCK, _copy_rows

    row_bytes = 10 * _ROW_BLOCK + 123
    generator = torch.Generator(device="cuda").manual_seed(0)
    backing = torch.randint(0, 256, (5, row_bytes), dtype=torch.uint8, device="cuda", generator=generator)
    expected = backing.clone()
    sources = torch.tensor([1, 2, 4], dtype=torch.int64, device="cuda")
    destinations = torch.tensor([3, 0, 2], dtype=torch.int64, device="cuda")
    operations = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda")
    count = torch.tensor([2], dtype=torch.int32, device="cuda")
    errors = torch.zeros(3, dtype=torch.int32, device="cuda")
    # Three programs per row walk all eleven tiles; the third command lies beyond the count.
    _copy_rows[(3, 3)](backing, sources, destinations, operations, count, errors, row_bytes, _ROW_BLOCK,
                       ((5, 9),))
    expected[3, :5], expected[3, 9:] = expected[1, :5], expected[1, 9:]
    expected[0, :5], expected[0, 9:] = 0, 0
    torch.cuda.synchronize()
    assert torch.equal(backing, expected)


def test_ripple_rows_launch_within_the_grid_limit():
    import triton

    from ignition.components.state_transfer import _ROW_BLOCK, _ROW_PROGRAMS

    # A Ripple TP1 row with BF16 rings and DFlash2 is 368 MiB: 94,272 tiles of 4,096 bytes.
    assert triton.cdiv(386_138_112, _ROW_BLOCK) > 65_535 >= _ROW_PROGRAMS
