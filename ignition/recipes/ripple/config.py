# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Structural checkpoint configuration for Ricursive Ripple dense transformers.

Ripple (`RicursiveForCausalLM`) is a dense decoder with PeriNorm blocks (pre/post
attention and pre/post FFN RMSNorm), per-head Q/K RMSNorm, a sigmoid output gate on
attention, a value residual blending the scaled token embeddings into every layer's
value projection input, sliding-window RoPE layers mixed with global NoPE layers,
and clamped SwiGLU activations. The checkpoint's `modeling_ricursive.py` is the
canonical reference for these semantics.
"""

from dataclasses import dataclass
import json
from pathlib import Path

ARCHITECTURE = "RicursiveForCausalLM"
PAGE_TOKENS = 128


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw: dict):
        if raw.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"Ripple preparation requires {ARCHITECTURE}")
        if raw.get("quantization_config") is not None:
            raise ValueError("Ripple preparation serves the unquantized BF16 checkpoint")
        required = {"model_type": "ricursive", "hidden_act": "silu", "head_dim": 128,
                    "use_gated_attention": True, "gate_full_precision": True, "use_valres": True,
                    "tie_word_embeddings": False}
        for name, expected in required.items():
            if raw.get(name) != expected:
                raise ValueError(f"unsupported Ripple {name}: {raw.get(name)!r}")
        for name in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "sliding_window",
                     "max_position_embeddings"):
            if type(raw.get(name)) is not int or raw[name] <= 0:
                raise ValueError(f"Ripple requires a positive integer {name}")
        layers = raw.get("layer_types", ())
        if (not isinstance(layers, list) or len(layers) != raw["num_hidden_layers"]
                or any(kind not in ("sliding_attention", "full_attention") for kind in layers)):
            raise ValueError("Ripple layer_types must describe every layer")
        if raw["num_attention_heads"] % raw["num_key_value_heads"]:
            raise ValueError("Ripple attention heads must be divisible by key/value heads")
        for name in ("rope_theta", "rms_norm_eps"):
            if not 0 < float(raw.get(name, 0)):
                raise ValueError(f"Ripple {name} must be positive")
        scale = raw.get("embed_scale")
        if scale is not None and not 0 < float(scale):
            raise ValueError("Ripple embed_scale must be positive when set")
        clamp = raw.get("ffn_clamp_threshold")
        if clamp is not None and float(clamp) < 0:
            raise ValueError("Ripple ffn_clamp_threshold must be nonnegative when set")
        return cls(raw)

    def validate_shard(self, shard) -> None:
        if shard.world_size not in (1, 2, 4):
            raise ValueError("Ripple supports tensor-parallel world_size in (1, 2, 4)")
        if self.heads % shard.world_size or self.kv_heads % shard.world_size:
            raise ValueError("Ripple attention and key/value heads must divide across ranks")
        if self.vocabulary % (shard.world_size * 64):
            raise ValueError("Ripple vocabulary must partition into 64-aligned rank shards")
        if self.intermediate_size % (shard.world_size * 128):
            raise ValueError("Ripple intermediate_size must divide into 128-aligned rank shards")

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.text["layer_types"])

    @property
    def num_layers(self) -> int:
        return self.text["num_hidden_layers"]

    @property
    def hidden(self) -> int:
        return self.text["hidden_size"]

    @property
    def vocabulary(self) -> int:
        return self.text["vocab_size"]

    @property
    def heads(self) -> int:
        return self.text["num_attention_heads"]

    @property
    def kv_heads(self) -> int:
        return self.text["num_key_value_heads"]

    @property
    def head_dim(self) -> int:
        return self.text["head_dim"]

    @property
    def intermediate_size(self) -> int:
        return self.text["intermediate_size"]

    @property
    def sliding_window(self) -> int:
        return self.text["sliding_window"]

    @property
    def rms_norm_eps(self) -> float:
        return float(self.text["rms_norm_eps"])

    @property
    def embed_scale(self) -> float | None:
        value = self.text.get("embed_scale")
        return None if value is None else float(value)

    @property
    def clamp(self) -> float | None:
        value = self.text.get("ffn_clamp_threshold")
        return float(value) if value else None

    def is_sliding(self, layer: int) -> bool:
        return self.layers[layer] == "sliding_attention"

    def local_heads(self, shard) -> int:
        return self.heads // shard.world_size

    def local_kv_heads(self, shard) -> int:
        return self.kv_heads // shard.world_size

    def q_range(self, shard) -> tuple[int, int]:
        return shard.partition(self.heads * self.head_dim, alignment=self.head_dim)

    def kv_range(self, shard) -> tuple[int, int]:
        return shard.partition(self.kv_heads * self.head_dim, alignment=self.head_dim)

    def intermediate_range(self, shard) -> tuple[int, int]:
        return shard.partition(self.intermediate_size, alignment=128)

    def vocabulary_partition(self, shard) -> tuple[int, int, int]:
        start, end = shard.partition(self.vocabulary, alignment=64)
        return start, end, end - start

    def rotary_inverse(self, device):
        import torch

        dim = self.head_dim
        return 1.0 / (float(self.text["rope_theta"])
                      ** (torch.arange(0, dim, 2, device=device).float() / dim))
