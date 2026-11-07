# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Ripple decoder layers over paged and ring KV and tensor-parallel projections.

Arithmetic follows the checkpoint's `modeling_ricursive.py`: RMSNorm rounds the
normalized FP32 activation to BF16 before its BF16 weight multiply, RoPE runs in
FP32 on BF16-rounded cos/sin tables, the attention gate's sigmoid runs in FP32, and
the value residual lerps the scaled embeddings toward each layer's normalized input.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.collectives import PreparedAllReduce
from ignition.components.dflash2 import PreparedContextCapture
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import prepare_projection
from ignition.components.paged_attention import (
    PagedAttentionMetadata, PreparedPagedAttention, PreparedPagedKVWriter,
)
from ignition.components.parallel import TensorParallelShard
from ignition.components.vocabulary import PreparedVocabularyProjection

from .config import PAGE_TOKENS
from .window import WindowMetadata


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x.float()
    left, right = x.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def staged_residual_norm(delta, hidden, post_weight, weight, eps, *, start=None, blend=None):
    """`hidden + rms_norm(delta, post_weight)` (with a delta), its `rms_norm(., weight)` and value lerp."""
    if delta is not None:
        hidden = hidden + rms_norm(delta, post_weight, eps)
    normed = rms_norm(hidden, weight, eps)
    return hidden, normed, None if start is None else torch.lerp(start, normed, blend)


def residual_norm(delta, hidden, post_weight, weight, eps, *, start=None, blend=None):
    # Every row count runs the staged Inductor chain. The fused decode kernel in residual_norm.py matches it
    # bit for bit in isolation, but served decode with it diverged from the baseline (rarely, from one verify
    # round on), so it stays unwired until that is explained (NOTES.md, 2026-11-07).
    return staged_residual_norm(delta, hidden, post_weight, weight, eps, start=start, blend=blend)


def _reduced(reduction: PreparedAllReduce, partial: torch.Tensor) -> torch.Tensor:
    # Traced forwards test the flag, never the runtime (see PreparedAllReduce).
    return partial if reduction.single else reduction(partial)


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    residual: torch.Tensor
    normalized: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, buffers, *, shard: TensorParallelShard = TensorParallelShard(0, 1)):
        def zeros(shape):
            return buffers.zeros(shape, torch.bfloat16)

        return cls(activations=zeros((tokens, config.hidden)), residual=zeros((tokens, config.hidden)),
                   normalized=zeros((tokens, config.hidden)), hidden=zeros((rows, config.hidden)),
                   local_logits=zeros((rows, config.vocabulary_partition(shard)[2])),
                   logits=zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return Scratch(self.activations[:tokens], self.residual[:tokens], self.normalized[:tokens],
                       self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


class PreparedLayer:
    def __init__(self, index, weights, config, source, metadata, pool, *, shard, runtime, coordinator, mode,
                 session, name, workspace, buffers, cache_metadata, windows):
        self.index, self.eps = index, config.rms_norm_eps
        self.is_sliding = config.is_sliding(index)
        self.weights = weights
        self.clamp = config.clamp
        tokens = source.shape[0]
        self.heads, self.kv_heads, self.dim = config.local_heads(shard), config.local_kv_heads(shard), config.head_dim
        self.q_width, self.k_width = self.heads * self.dim, self.kv_heads * self.dim
        r0, r1 = config.intermediate_range(shard)
        self.inter = r1 - r0
        width = metadata.max_query_tokens if mode == "verify" else 1

        def zeros(shape):
            return buffers.zeros(shape, torch.bfloat16)

        # A checkpoint that stores MXFP8 projections runs them weight-only: BF16 activations times
        # E4M3 x 2^e weights, exactly the dequantized checkpoint whose KL divergence against BF16
        # was measured (SM120 kernels; the SM103 planner chooses its own mode). BF16 projections,
        # and those converted by --online-mxfp8, let the tuner choose per shape.
        activation_mode = "a16" if config.mxfp8_projections else "auto"

        def projection(inputs, weight, output, field):
            return prepare_projection(inputs, weight, output, session=session, name=f"{name}.{field}",
                                      workspace=workspace, activation_mode=activation_mode)

        # Projections read their BF16 rows from the forward's own temporaries. A copy into these
        # prepared buffers lowers to an opaque memcpy node per layer (a partial arena region
        # cannot fuse into its producer), so the buffers only shape and tune the projections.
        self.normed = self.value_input = source
        self.qgk_out = zeros((tokens, 2 * self.q_width + self.k_width))
        self.v_out = zeros((tokens, self.k_width))
        self.q = zeros((tokens, self.heads, self.dim))
        self.key = zeros((tokens, self.kv_heads, self.dim))
        self.value = zeros((tokens, self.kv_heads, self.dim))
        self.attended = zeros((tokens, self.q_width))
        self.attn_partial = zeros((tokens, config.hidden))
        # A single rank's reduction is the identity: its partial is already the sum.
        self.attn_out = self.attn_partial if runtime is None else zeros((tokens, config.hidden))
        self.gate_up_out = zeros((tokens, 2 * self.inter))
        self.activation = zeros((tokens, self.inter))
        self.mlp_partial = zeros((tokens, config.hidden))
        self.mlp_out = self.mlp_partial if runtime is None else zeros((tokens, config.hidden))
        self.blend = torch.sigmoid(weights["value_blend"][0])
        # The norm after this layer: the next layer's input norm and value blend, or the final norm.
        self.following = None
        self.qgk = projection(source, weights["qgk"], self.qgk_out, "qgk")
        self.v_proj = projection(self.value_input, weights["v_proj"], self.v_out, "v_proj")
        self.o_proj = projection(self.attended, weights["o_proj"], self.attn_partial, "o_proj")
        self.gate_up = projection(source, weights["gate_up"], self.gate_up_out, "gate_up")
        self.down = projection(self.activation, weights["down_proj"], self.mlp_partial, "down_proj")
        self.attn_reduce = PreparedAllReduce(self.attn_partial, self.attn_out, runtime=runtime, session=session,
                                             coordinator=coordinator, name=f"{name}.attn_reduce")
        self.mlp_reduce = PreparedAllReduce(self.mlp_partial, self.mlp_out, runtime=runtime, session=session,
                                            coordinator=coordinator, name=f"{name}.mlp_reduce")
        # Sliding layers write and read their ring, or a prefill chunk's staged window (window.py).
        self.window = windows.layer(index) if self.is_sliding else None
        if self.window is None:
            key_cache, value_cache = pool.attention(index, config, shard=shard)
            pages, slots, tuning_pages = metadata.page_table, cache_metadata.slots, int(pool.memory_capacity[1].item())
        else:
            w = self.window
            key_cache, value_cache, pages, slots, tuning_pages = w.key, w.value, w.pages, w.slots, w.tuning_page_count
        self.writer = PreparedPagedKVWriter(self.key, self.value, key_cache, value_cache, slots,
                                            session=session, name=f"{name}.cache_write")
        self.attention = PreparedPagedAttention(self.q, key_cache, value_cache, pages,
            cache_metadata.lengths, metadata.query_start_loc, self.attended.view(tokens, self.heads, self.dim),
            mode=mode, window=config.sliding_window - 1 if self.is_sliding else -1, width=width,
            session=session, name=f"{name}.attention", workspace=workspace, tuning_page_count=tuning_pages)

    def __call__(self, hidden, normed, value_input, value_residual, cos, sin):
        """One layer from its normalized input rows; returns the residual stream and the following norm's
        rows (with the next layer's value lerp, or None after the last layer)."""
        w, eps, tokens = self.weights, self.eps, hidden.shape[0]
        self.qgk(normed)
        self.v_proj(value_input)
        q = rms_norm(self.qgk_out[:, :self.q_width].view(tokens, self.heads, self.dim), w["q_norm"], eps)
        k = rms_norm(self.qgk_out[:, 2 * self.q_width:].view(tokens, self.kv_heads, self.dim), w["k_norm"], eps)
        if self.is_sliding:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        self.q.copy_(q)
        self.key.copy_(k)
        self.value.copy_(self.v_out.view(tokens, self.kv_heads, self.dim))
        if self.window is not None:
            self.window.gather()
        self.writer()
        self.attention()
        if self.window is not None:
            self.window.commit()
        gate = torch.sigmoid(self.qgk_out[:, self.q_width:2 * self.q_width].float()).to(hidden.dtype)
        attn = _reduced(self.attn_reduce, self.o_proj(self.attended * gate))
        hidden, normed, _ = residual_norm(attn, hidden, w["post_attention_layernorm"],
                                          w["pre_feedforward_layernorm"], eps)
        self.gate_up(normed)
        gate, up = self.gate_up_out[:, :self.inter], self.gate_up_out[:, self.inter:]
        if self.clamp is not None:
            gate, up = gate.clamp(max=self.clamp), up.clamp(min=-self.clamp, max=self.clamp)
        mlp = _reduced(self.mlp_reduce, self.down(F.silu(gate) * up))
        weight, blend = self.following
        return residual_norm(mlp, hidden, w["post_feedforward_layernorm"], weight, eps,
                             start=None if blend is None else value_residual, blend=blend)


class PreparedDecoder:
    def __init__(self, weights, scratch: Scratch, metadata, pool, *, mode, session, name, workspace, layer_buffers,
                 runtime=None, coordinator=None, vocabulary_runtime=None, buffers=None, scoring=None, staging=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        config = self.config = weights.config
        self.eps = config.rms_norm_eps
        device = scratch.activations.device
        self.cache_metadata = PagedAttentionMetadata(metadata, page_tokens=PAGE_TOKENS, buffers=buffers)
        self.windows = WindowMetadata(metadata, pool, config, weights.shard, mode=mode, buffers=buffers,
                                      staging=staging)
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.inverse_frequency = config.rotary_inverse(device)
        self.layers = [PreparedLayer(index, layer_weights, config, scratch.normalized, metadata, pool,
                                     shard=weights.shard, runtime=runtime, coordinator=coordinator, mode=mode,
                                     session=session, name=f"{name}.layer{index}", workspace=workspace,
                                     buffers=layer_buffers.cursor(), cache_metadata=self.cache_metadata,
                                     windows=self.windows)
                       for index, layer_weights in enumerate(weights.layers)]
        for layer, following in zip(self.layers, self.layers[1:]):
            layer.following = following.weights["input_layernorm"], following.blend
        self.layers[-1].following = weights.norm, None
        self.scoring = None if scoring is None else scoring.prepare(scratch.normalized, weights.head,
            vocabulary=config.vocabulary, session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            activation_mode="auto", buffers=buffers)
        # With a drafter, the target layers it reads are captured for its accepted-context rings.
        draft = config.draft if weights.draft is not None else None
        self.taps = () if draft is None else draft.target_layers
        self.context_capture = None if draft is None else PreparedContextCapture(metadata,
            config.hidden * len(self.taps), draft.attention.window, compact_prefill=mode == "prefill",
            buffers=buffers)
        self.auxiliary = None if self.context_capture is None else self.context_capture.output

    def commit(self, counts):
        # Paged KV beyond the accepted prefix is rewritten by later rounds; nothing else is staged.
        pass

    def __call__(self):
        m = self.metadata
        self.cache_metadata()
        self.windows()
        frequency = m.positions.clamp_min(0).float()[:, None] * self.inverse_frequency[None, :]
        table = torch.cat((frequency, frequency), dim=-1)
        cos, sin = table.cos().bfloat16().float(), table.sin().bfloat16().float()
        embedded = self.embedding()
        scale = self.config.embed_scale
        self.scratch.residual.copy_(embedded if scale is None else embedded * scale)
        hidden = residual = self.scratch.residual
        if self.context_capture is not None:
            self.context_capture.begin()
        first = self.layers[0]
        _, normed, value_input = residual_norm(None, hidden, None, first.weights["input_layernorm"], self.eps,
                                               start=residual, blend=first.blend)
        for index, layer in enumerate(self.layers):
            hidden, normed, value_input = layer(hidden, normed, value_input, residual, cos, sin)
            if index in self.taps:
                self.context_capture(hidden, self.taps.index(index) * hidden.shape[-1])
        # After the last layer, `normed` is the final norm.
        self.scratch.normalized.copy_(normed)
        if self.scoring is not None:
            for source, projection, reduction in self.scoring.parts:
                projection(source)
                reduction()
        self.scratch.hidden.copy_(self.scratch.normalized[m.sample_indices])
        self.head(self.scratch.hidden)
