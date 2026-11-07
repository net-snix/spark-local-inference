# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ripple decoder-layer wiring: projections read the forward's temporaries, not staged copies."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ignition.recipes.ripple.decoder import PreparedLayer, rms_norm, rotate, staged_residual_norm

TOKENS, HIDDEN, HEADS, KV_HEADS, DIM, INTER, EPS = 3, 32, 2, 1, 8, 24, 1e-6


class _Linear:
    def __init__(self, weight, output):
        self.weight, self.output = weight, output

    def __call__(self, source):
        self.output.copy_((source.float() @ self.weight.float().t()).to(source.dtype))
        return self.output


def _bf16(*shape, generator):
    return torch.randn(*shape, generator=generator).bfloat16()


def _layer(generator, clamp):
    q_width, k_width = HEADS * DIM, KV_HEADS * DIM
    layer = object.__new__(PreparedLayer)
    layer.index, layer.eps, layer.is_sliding, layer.clamp = 0, EPS, True, clamp
    layer.heads, layer.kv_heads, layer.dim = HEADS, KV_HEADS, DIM
    layer.q_width, layer.k_width, layer.inter = q_width, k_width, INTER
    norms = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
             "post_feedforward_layernorm")
    layer.weights = {name: _bf16(HIDDEN, generator=generator) for name in norms}
    layer.weights.update(q_norm=_bf16(DIM, generator=generator), k_norm=_bf16(DIM, generator=generator))
    layer.blend = torch.sigmoid(torch.tensor(0.3)).bfloat16()
    # The next layer's input norm and value blend.
    layer.following = _bf16(HIDDEN, generator=generator), torch.sigmoid(torch.tensor(-0.2)).bfloat16()

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.bfloat16)

    # Prepared input buffers: they shape and tune the projections and must stay untouched.
    layer.normed = layer.value_input = zeros(TOKENS, HIDDEN)
    layer.qgk_out, layer.v_out = zeros(TOKENS, 2 * q_width + k_width), zeros(TOKENS, k_width)
    layer.q, layer.key, layer.value = zeros(TOKENS, HEADS, DIM), zeros(TOKENS, KV_HEADS, DIM), zeros(
        TOKENS, KV_HEADS, DIM)
    layer.attended, layer.attn_partial = zeros(TOKENS, q_width), zeros(TOKENS, HIDDEN)
    layer.gate_up_out, layer.mlp_partial = zeros(TOKENS, 2 * INTER), zeros(TOKENS, HIDDEN)
    layer.attn_out, layer.mlp_out = layer.attn_partial, layer.mlp_partial
    weights = {"qgk": (2 * q_width + k_width, HIDDEN), "v": (k_width, HIDDEN), "o": (HIDDEN, q_width),
               "gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}
    layer.matrices = {name: _bf16(*shape, generator=generator) * 0.2 for name, shape in weights.items()}
    layer.qgk = _Linear(layer.matrices["qgk"], layer.qgk_out)
    layer.v_proj = _Linear(layer.matrices["v"], layer.v_out)
    layer.o_proj = _Linear(layer.matrices["o"], layer.attn_partial)
    layer.gate_up = _Linear(layer.matrices["gate_up"], layer.gate_up_out)
    layer.down = _Linear(layer.matrices["down"], layer.mlp_partial)
    layer.writer = lambda: None
    # A sliding layer stages its KV ring around attention (window.py); staging moves pages, no arithmetic.
    layer.window = SimpleNamespace(gather=lambda: None, commit=lambda: None)
    # Stand-in attention: every query head attends to itself.
    layer.attention = lambda: layer.attended.view(TOKENS, HEADS, DIM).copy_(layer.q)
    # A single rank's reduction is never called: it would fail here.
    layer.attn_reduce = layer.mlp_reduce = SimpleNamespace(single=True)
    return layer


def _reference(layer, hidden, value_residual, cos, sin):
    """The staged arithmetic the layer had before it fed temporaries to its projections."""
    w, m, q_width = layer.weights, layer.matrices, layer.q_width
    following, blend = layer.following

    def linear(source, weight):
        return (source.float() @ weight.float().t()).to(source.dtype)

    normed = rms_norm(hidden, w["input_layernorm"], EPS)
    qgk = linear(normed, m["qgk"])
    value_input = torch.lerp(value_residual, normed, layer.blend)
    linear(value_input, m["v"])
    q = rms_norm(qgk[:, :q_width].view(TOKENS, HEADS, DIM), w["q_norm"], EPS)
    q = rotate(q, cos, sin).bfloat16()
    attended = q.view(TOKENS, q_width).clone()
    attended.mul_(torch.sigmoid(qgk[:, q_width:2 * q_width].float()).bfloat16())
    hidden = hidden + rms_norm(linear(attended, m["o"]), w["post_attention_layernorm"], EPS)
    gate_up = linear(rms_norm(hidden, w["pre_feedforward_layernorm"], EPS), m["gate_up"])
    gate, up = gate_up[:, :INTER], gate_up[:, INTER:]
    if layer.clamp is not None:
        gate, up = gate.clamp(max=layer.clamp), up.clamp(min=-layer.clamp, max=layer.clamp)
    mlp = linear(F.silu(gate) * up, m["down"])
    hidden = hidden + rms_norm(mlp, w["post_feedforward_layernorm"], EPS)
    normed = rms_norm(hidden, following, EPS)
    return hidden, normed, torch.lerp(value_residual, normed, blend)


def test_layer_projections_read_forward_temporaries():
    for clamp in (None, 0.5):
        generator = torch.Generator().manual_seed(7)
        layer = _layer(generator, clamp)
        hidden, value_residual = _bf16(TOKENS, HIDDEN, generator=generator), _bf16(TOKENS, HIDDEN, generator=generator)
        angles = torch.rand(TOKENS, DIM // 2, generator=generator) * 3
        table = torch.cat((angles, angles), dim=-1)
        cos, sin = table.cos().bfloat16().float(), table.sin().bfloat16().float()
        expected = _reference(layer, hidden, value_residual, cos, sin)
        _, normed, value_input = staged_residual_norm(None, hidden, None, layer.weights["input_layernorm"], EPS,
                                                      start=value_residual, blend=layer.blend)
        actual = layer(hidden, normed, value_input, value_residual, cos, sin)
        assert all(torch.equal(got, want) for got, want in zip(actual, expected, strict=True))
        assert not layer.normed.any()
