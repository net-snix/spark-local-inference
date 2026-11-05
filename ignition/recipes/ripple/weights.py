# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ripple checkpoint transfers with tensor-parallel slices and fused projections.

BF16 checkpoints load projections as BF16, optionally converted to MXFP8 while loading.
Checkpoints that store ModelOpt MXFP8 projections load their E4M3 values and UE8M0 scales
unchanged through the shared quantized loader.

Projections sharing an input are packed once while loading: each rank's query, output
gate, and key rows form one `qgk` projection of the normalized hidden state, and its
gate and up rows form one `gate_up` projection. The value projection reads the
value-residual blend, a different input, and stays separate.
"""

from dataclasses import dataclass, replace

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader, quantize_bf16_linear_as_mxfp8
from ignition.preparation.tensor_metadata import record_numeric_tensor

from .config import ModelConfig

PREFIX = "model"


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[dict, ...]
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)
    draft: object = None
    draft_embedding: torch.Tensor | None = None

    @classmethod
    def load(cls, manifest, device, *, shard: TensorParallelShard = TensorParallelShard(0, 1), progress=None,
             draft_snapshot=None, online_mxfp8=False):
        config = ModelConfig.load(manifest.snapshot)
        config.validate_shard(shard)
        if draft_snapshot is not None:
            from .draft_config import load as load_draft_config

            config = replace(config, draft=load_draft_config(draft_snapshot, config))
            config.draft.validate_shard(shard)
        v0, v1, local_vocab = config.vocabulary_partition(shard)
        before = torch.cuda.memory_allocated(device)
        with RippleWeightLoader(manifest, device, config=config, shard=shard, progress=progress,
                                online_mxfp8=online_mxfp8) as loader:
            embedding, norm = loader.load_tensors((f"{PREFIX}.embed_tokens.weight", f"{PREFIX}.norm.weight"))
            if embedding.dtype != torch.bfloat16 or embedding.shape != (config.vocabulary, config.hidden):
                raise ValueError("Ripple requires a BF16 embedding matching the vocabulary and hidden size")
            if norm.shape != (config.hidden,):
                raise ValueError("Ripple final norm weight shape differs from hidden_size")
            head = loader.load_tensor("lm_head.weight", extent=(slice(v0, v1), slice(None)))
            if head.dtype != torch.bfloat16 or head.shape != (local_vocab, config.hidden):
                raise ValueError("Ripple vocabulary projection shape differs from the model")
            layers = tuple(loader.load_layer(index) for index in range(config.num_layers))
        draft = None
        if draft_snapshot is not None:
            from ignition.components.dflash2_weights import DraftWeights

            draft = DraftWeights.load(draft_snapshot, config.draft, device, shard, progress=progress,
                                      online_mxfp8=online_mxfp8)
        draft_embedding = None
        if draft is not None:
            # Ranks share compiled forwards, which bind exact storage offsets; the drafter reads its
            # own column shard instead of a rank-dependent view into the replicated table.
            first, last = shard.partition(config.hidden, alignment=64)
            draft_embedding = embedding[:, first:last].contiguous()
        return cls(config=config, embedding=embedding, norm=norm, head=head, layers=layers,
                   allocated_bytes=torch.cuda.memory_allocated(device) - before, shard=shard, draft=draft,
                   draft_embedding=draft_embedding)


class RippleWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, config: ModelConfig, shard: TensorParallelShard, progress=None,
                 online_mxfp8=False):
        self.config = config
        self.shard = shard
        self.online_mxfp8 = online_mxfp8
        super().__init__(manifest, device, prefixes=(PREFIX + ".", "lm_head."), progress=progress)

    def _source(self, name: str, shape: tuple[int, int]):
        source = self._sources[name]
        if source.dtype != torch.bfloat16 or tuple(source.shape) != shape:
            raise ValueError(f"{name}: expected BF16 {shape}, got {source.dtype} {tuple(source.shape)}")
        return source

    def _mxfp8_source(self, prefix: str, shape: tuple[int, int]) -> None:
        """Require a stored MXFP8 projection's E4M3 values and UE8M0 scales to match the model."""
        weight, scale = self._sources[f"{prefix}.weight"], self._sources[f"{prefix}.weight_scale"]
        if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != shape:
            raise ValueError(f"{prefix}.weight: expected F8_E4M3 {shape}, got {weight.dtype} {tuple(weight.shape)}")
        if scale.dtype != torch.uint8 or tuple(scale.shape) != (shape[0], shape[1] // 32):
            raise ValueError(f"{prefix}.weight_scale: expected U8 {(shape[0], shape[1] // 32)}, "
                             f"got {scale.dtype} {tuple(scale.shape)}")

    def _rows(self, parts) -> torch.Tensor:
        """Pack row slices of several checkpoint matrices into one contiguous projection."""
        rows = sum(last - first for _, first, last in parts)
        columns = parts[0][0].shape[1]
        fused = torch.empty((rows, columns), dtype=torch.bfloat16, device=self.device)
        offset = 0
        for source, first, last in parts:
            self._copy(fused[offset:offset + last - first], source[first:last])
            offset += last - first
        self._session.flush()
        record_numeric_tensor(fused)
        return fused

    def _columns(self, name: str, shape: tuple[int, int], first: int, last: int) -> torch.Tensor:
        source = self._source(name, shape)
        result = torch.empty((shape[0], last - first), dtype=torch.bfloat16, device=self.device)
        self._copy(result, source[:, first:last])
        self._session.flush()
        return result

    def load_layer(self, index: int) -> dict:
        c, shard = self.config, self.shard
        prefix = f"{PREFIX}.layers.{index}"
        hidden, dim = c.hidden, c.head_dim
        q0, q1 = c.q_range(shard)
        k0, k1 = c.kv_range(shard)
        r0, r1 = c.intermediate_range(shard)
        names = tuple(f"{prefix}.{name}.weight" for name in (
            "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
            "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"))
        norms = dict(zip(("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                          "post_feedforward_layernorm", "q_norm", "k_norm"), self.load_tensors(names), strict=True))
        for name, tensor in norms.items():
            expected = (dim,) if name in ("q_norm", "k_norm") else (hidden,)
            if tensor.dtype != torch.bfloat16 or tensor.shape != expected:
                raise ValueError(f"{prefix}.{name}: expected BF16 {expected}")
        blend = self.load_tensor(f"{prefix}.self_attn.value_res_lambda")
        if blend.shape != (1,):
            raise ValueError(f"{prefix}.self_attn.value_res_lambda: expected shape (1,)")
        if c.mxfp8_projections:
            attention, mlp = f"{prefix}.self_attn", f"{prefix}.mlp"
            q_rows, kv_rows, inner = c.heads * dim, c.kv_heads * dim, c.intermediate_size
            for name, shape in ((f"{attention}.q_proj", (q_rows, hidden)), (f"{attention}.g_proj", (q_rows, hidden)),
                                (f"{attention}.k_proj", (kv_rows, hidden)), (f"{attention}.v_proj", (kv_rows, hidden)),
                                (f"{attention}.o_proj", (hidden, q_rows)), (f"{mlp}.gate_proj", (inner, hidden)),
                                (f"{mlp}.up_proj", (inner, hidden)), (f"{mlp}.down_proj", (hidden, inner))):
                self._mxfp8_source(name, shape)
            projections = dict(
                qgk=self.load_linear_group((f"{attention}.q_proj", f"{attention}.g_proj", f"{attention}.k_proj"),
                                           row_ranges=((q0, q1), (q0, q1), (k0, k1))),
                v_proj=self.load_linear(f"{attention}.v_proj", rows=(k0, k1)),
                o_proj=self.load_linear(f"{attention}.o_proj", columns=(q0, q1)),
                gate_up=self.load_linear_group((f"{mlp}.gate_proj", f"{mlp}.up_proj"),
                                               row_ranges=((r0, r1), (r0, r1))),
                down_proj=self.load_linear(f"{mlp}.down_proj", columns=(r0, r1)),
            )
            shapes = dict(qgk=(2 * (q1 - q0) + k1 - k0, hidden), v_proj=(k1 - k0, hidden), o_proj=(hidden, q1 - q0),
                          gate_up=(2 * (r1 - r0), hidden), down_proj=(hidden, r1 - r0))
            for name, weight in projections.items():
                if (weight.out_features, weight.in_features) != shapes[name]:
                    raise ValueError(f"{prefix}.{name}: MXFP8 projection shape disagrees with the model")
            return dict(norms, value_blend=blend, **projections)
        q = self._source(f"{prefix}.self_attn.q_proj.weight", (c.heads * dim, hidden))
        g = self._source(f"{prefix}.self_attn.g_proj.weight", (c.heads * dim, hidden))
        k = self._source(f"{prefix}.self_attn.k_proj.weight", (c.kv_heads * dim, hidden))
        v = self._source(f"{prefix}.self_attn.v_proj.weight", (c.kv_heads * dim, hidden))
        gate = self._source(f"{prefix}.mlp.gate_proj.weight", (c.intermediate_size, hidden))
        up = self._source(f"{prefix}.mlp.up_proj.weight", (c.intermediate_size, hidden))
        projections = dict(
            qgk=self._rows(((q, q0, q1), (g, q0, q1), (k, k0, k1))),
            v_proj=self._rows(((v, k0, k1),)),
            o_proj=self._columns(f"{prefix}.self_attn.o_proj.weight", (hidden, c.heads * dim), q0, q1),
            gate_up=self._rows(((gate, r0, r1), (up, r0, r1))),
            down_proj=self._columns(f"{prefix}.mlp.down_proj.weight", (hidden, c.intermediate_size), r0, r1),
        )
        if self.online_mxfp8:
            # Optional bandwidth trade: each layer's BF16 projections become b12x MXFP8 as they load.
            # A checkpoint that already stores MXFP8 projections returns above unchanged.
            projections = {name: quantize_bf16_linear_as_mxfp8(weight) for name, weight in projections.items()}
        # value_blend keeps the raw lambda; the decoder applies sigmoid in the parameter dtype.
        return dict(norms, value_blend=blend, **projections)
