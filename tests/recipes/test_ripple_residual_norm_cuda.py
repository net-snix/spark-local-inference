# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The fused Ripple residual RMSNorm equals the staged ATen/Inductor arithmetic bit for bit."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="set FLARE_RUN_CUDA_TESTS=1 to run CUDA kernels")

HIDDEN, EPS = 5120, 1e-6


def _rows(rows, scale, generator):
    return (torch.randn(rows, HIDDEN, device="cuda", generator=generator) * scale).bfloat16()


def _gain(generator):
    return (torch.rand(HIDDEN, device="cuda", generator=generator) * 3 - 0.5).bfloat16()


def test_aten_block_width_follows_reduce_config():
    from ignition.recipes.ripple.residual_norm import aten_block_width

    widths = [aten_block_width(rows, HIDDEN) for rows in (1, 2, 3, 4, 7, 8, 15, 16, 17)]
    assert widths == [512, 256, 256, 128, 128, 64, 64, 32, 32]


@pytest.mark.parametrize("rows", range(1, 17))
@pytest.mark.parametrize("scale", (1e-3, 1.0, 300.0))
def test_fused_matches_staged(rows, scale):
    from ignition.recipes.ripple.decoder import staged_residual_norm
    from ignition.recipes.ripple.residual_norm import residual_norm

    generator = torch.Generator(device="cuda").manual_seed(rows * 7 + int(scale > 1))
    hidden, delta, start = _rows(rows, 4.0, generator), _rows(rows, scale, generator), _rows(rows, 1.0, generator)
    post, weight = _gain(generator), _gain(generator)
    for blend in (0.3, 0.75, -0.6):
        blend = torch.tensor(blend, device="cuda").bfloat16()
        cases = (((delta, hidden, post, weight, EPS), dict(start=start, blend=blend)),
                 ((delta, hidden, post, weight, EPS), {}),
                 ((None, hidden, None, weight, EPS), dict(start=start, blend=blend)))
        for arguments, keywords in cases:
            expected = staged_residual_norm(*arguments, **keywords)
            actual = residual_norm(*arguments, **keywords)
            for got, want in zip(actual, expected):
                assert (got is None) == (want is None)
                if want is not None:
                    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
