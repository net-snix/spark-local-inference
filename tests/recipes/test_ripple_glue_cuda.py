# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""glue.py kernels: close to the staged eager arithmetic, and invariant to the launch's row count."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="set FLARE_RUN_CUDA_TESTS=1 to run CUDA kernels")

HIDDEN, EPS, HEADS, KV_HEADS, DIM = 5120, 1e-6, 80, 8, 128


def _rows(rows, width, scale, generator):
    return (torch.randn(rows, width, device="cuda", generator=generator) * scale).bfloat16()


def _gain(width, generator):
    return (torch.rand(width, device="cuda", generator=generator) * 3 - 0.5).bfloat16()


def _ulps(got, want):
    """Largest distance in BF16 units in the last place (same-sign values)."""
    a, b = got.float(), want.float()
    scale = torch.maximum(a.abs(), b.abs()).clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(scale)) - 7)
    return ((a - b).abs() / ulp).max().item()


@pytest.mark.parametrize("rows", (1, 3, 16, 64, 2048))
@pytest.mark.parametrize("scale", (1e-3, 1.0, 300.0))
def test_residual_norm_close_to_staged(rows, scale):
    from ignition.recipes.ripple.decoder import staged_residual_norm
    from ignition.recipes.ripple.glue import residual_norm

    generator = torch.Generator(device="cuda").manual_seed(rows * 7 + int(scale > 1))
    hidden, delta = _rows(rows, HIDDEN, 4.0, generator), _rows(rows, HIDDEN, scale, generator)
    start = _rows(rows, HIDDEN, 1.0, generator)
    post, weight = _gain(HIDDEN, generator), _gain(HIDDEN, generator)
    for blend in (0.3, 0.75, -0.6):
        blend = torch.tensor(blend, device="cuda").bfloat16()
        for arguments, keywords in (((delta, hidden, post, weight, EPS), dict(start=start, blend=blend)),
                                    ((delta, hidden, post, weight, EPS), {}),
                                    ((None, hidden, None, weight, EPS), dict(start=start, blend=blend))):
            expected = staged_residual_norm(*arguments, **keywords)
            actual = residual_norm(*arguments, **keywords)
            for got, want in zip(actual, expected):
                assert (got is None) == (want is None)
                if want is not None:
                    # Only the FP32 sum order differs: at most one BF16 rounding flip per value, rarely two.
                    assert _ulps(got, want) <= 2
                    assert (got != want).float().mean().item() < 0.02


def test_residual_norm_row_invariant():
    from ignition.recipes.ripple.glue import residual_norm

    generator = torch.Generator(device="cuda").manual_seed(11)
    hidden, delta, start = (_rows(64, HIDDEN, s, generator) for s in (4.0, 2.0, 1.0))
    post, weight = _gain(HIDDEN, generator), _gain(HIDDEN, generator)
    blend = torch.tensor(0.3, device="cuda").bfloat16()
    full = residual_norm(delta, hidden, post, weight, EPS, start=start, blend=blend)
    for rows in (1, 2, 5, 16, 33):
        part = residual_norm(delta[:rows].contiguous(), hidden[:rows].contiguous(), post, weight, EPS,
                             start=start[:rows].contiguous(), blend=blend)
        for got, want in zip(part, full):
            assert torch.equal(got.view(torch.int16), want[:rows].view(torch.int16))


def _staged_qk(qgk, v, q_norm, k_norm, cos, sin, rope):
    from ignition.recipes.ripple.decoder import rms_norm, rotate

    tokens = qgk.shape[0]
    q = rms_norm(qgk[:, :HEADS * DIM].view(tokens, HEADS, DIM), q_norm, EPS)
    k = rms_norm(qgk[:, 2 * HEADS * DIM:].view(tokens, KV_HEADS, DIM), k_norm, EPS)
    if rope:
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    return q.bfloat16(), k.bfloat16(), v.view(tokens, KV_HEADS, DIM)


@pytest.mark.parametrize("tokens", (1, 16, 64, 2048))
@pytest.mark.parametrize("rope", (False, True))
def test_qk_rope_close_to_staged(tokens, rope):
    from ignition.recipes.ripple.glue import qk_rope

    generator = torch.Generator(device="cuda").manual_seed(tokens + rope)
    qgk = _rows(tokens, (2 * HEADS + KV_HEADS) * DIM, 3.0, generator)
    v = _rows(tokens, KV_HEADS * DIM, 1.0, generator)
    q_norm, k_norm = _gain(DIM, generator), _gain(DIM, generator)
    positions = torch.randint(0, 200_000, (tokens,), device="cuda", generator=generator).float()
    inverse = 1.0 / (10000.0 ** (torch.arange(0, DIM, 2, device="cuda").float() / DIM))
    table = torch.cat([positions[:, None] * inverse[None, :]] * 2, dim=-1)
    cos, sin = table.cos().bfloat16().float(), table.sin().bfloat16().float()
    q = torch.empty(tokens, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    key = torch.empty(tokens, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    value = torch.empty_like(key)
    qk_rope(qgk, v, q_norm, k_norm, cos, sin, q, key, value, EPS, rope=rope)
    want_q, want_k, want_v = _staged_qk(qgk, v, q_norm, k_norm, cos, sin, rope)
    assert torch.equal(value, want_v)
    for got, want in ((q, want_q), (key, want_k)):
        assert _ulps(got, want) <= 2
        assert (got != want).float().mean().item() < 0.02
    # Row invariance: the first token alone gives the same bits.
    q1, k1, v1 = q[:1].clone(), key[:1].clone(), value[:1].clone()
    qk_rope(qgk[:1], v[:1], q_norm, k_norm, cos[:1], sin[:1], q1, k1, v1, EPS, rope=rope)
    assert torch.equal(q1, q[:1]) and torch.equal(k1, key[:1])


@pytest.mark.parametrize("tokens", (1, 16, 2048))
def test_gate_matches_staged(tokens):
    from ignition.recipes.ripple.glue import gate

    generator = torch.Generator(device="cuda").manual_seed(tokens)
    qgk = _rows(tokens, (2 * HEADS + KV_HEADS) * DIM, 4.0, generator)
    attended = _rows(tokens, HEADS * DIM, 1.0, generator)
    want = attended * torch.sigmoid(qgk[:, HEADS * DIM:2 * HEADS * DIM].float()).bfloat16()
    got = gate(attended, qgk, HEADS * DIM)
    assert _ulps(got, want) <= 1
