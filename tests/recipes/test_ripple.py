# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ripple-65B Preview configuration, tensor-parallel geometry, storage costs, and DFlash2 draft."""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes import resolve

HUB = Path.home() / ".cache/huggingface/hub"
MODEL = resolve("ripple-65b")
LAYER_TYPES = ["full_attention" if index % 5 == 4 else "sliding_attention" for index in range(80)]
CONFIG = {
    "architectures": ["RicursiveForCausalLM"],
    "auto_map": {"AutoConfig": "configuration_ricursive.RicursiveConfig",
                 "AutoModel": "modeling_ricursive.RicursiveModel",
                 "AutoModelForCausalLM": "modeling_ricursive.RicursiveForCausalLM"},
    "bos_token_id": 100257, "embed_scale": 71.55417527999327,
    "eos_token_id": [100265, 100257], "ffn_clamp_threshold": 20.0, "gate_full_precision": True,
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120,
    "hybrid_layer_pattern": [0 if index % 5 == 4 else 1 for index in range(80)], "intermediate_size": 40960,
    "is_hybrid_swa": True, "layer_types": LAYER_TYPES, "max_position_embeddings": 2097152,
    "model_type": "ricursive", "num_attention_heads": 80, "num_hidden_layers": 80, "num_key_value_heads": 8,
    "pad_token_id": 100277, "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": 1024,
    "tie_word_embeddings": False, "transformers_version": "4.57.6", "use_cache": True,
    "use_gated_attention": True, "use_valres": True, "vocab_size": 100352,
}
DRAFT_CONFIG = {
    "architectures": ["DFlash2DraftModel"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 100257,
    "dflash_config": {"block_size": 16, "conv_group_size": 16, "conv_kernel_size": 4, "mask_token_id": 100278,
                      "selector_rank": 256, "selector_top_k": 32, "target_layer_ids": [1, 20, 39, 58, 77]},
    "draft_vocab_size": 100352, "dtype": "bfloat16", "eos_token_id": [100265, 100257], "head_dim": 128,
    "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02, "intermediate_size": 40960,
    "is_causal": False, "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
    "max_position_embeddings": 131072, "max_window_layers": 5, "model_type": "qwen3", "no_rope_layers": [4],
    "num_attention_heads": 80, "num_hidden_layers": 5, "num_key_value_heads": 8, "num_target_layers": 80,
    "pad_token_id": 100277, "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "rope_theta": 10000.0,
    "sliding_window": 4096, "tie_word_embeddings": False, "transformers_version": "5.12.1", "use_cache": True,
    "use_sliding_window": True, "vocab_size": 100352,
}


def checkpoint_snapshot(variable, repository, revision, required):
    """A local checkpoint directory from the environment, else the pinned hub-cache snapshot."""
    candidates = [Path(os.environ[variable])] if os.environ.get(variable) else []
    candidates.append(HUB / f"models--{repository.replace('/', '--')}" / "snapshots" / revision)
    for path in candidates:
        if all((path / name).is_file() for name in required):
            return path
    pytest.skip(f"set {variable} or cache {repository}@{revision[:8]} ({', '.join(required)})")


def target_snapshot(*required):
    checkpoint = MODEL.checkpoint
    return checkpoint_snapshot("FLARE_RIPPLE_CHECKPOINT", checkpoint.repository, checkpoint.revision,
                               required or ("config.json",))


def draft_snapshot(*required):
    from ignition.recipes.ripple.draft_config import REPOSITORY, REVISION

    return checkpoint_snapshot("FLARE_RIPPLE_DRAFT_CHECKPOINT", REPOSITORY, REVISION, required or ("config.json",))


@pytest.fixture
def target():
    from ignition.recipes.ripple.config import ModelConfig

    return ModelConfig.from_dict(copy.deepcopy(CONFIG))


@pytest.fixture
def drafted(target):
    from ignition.recipes.ripple import draft_config

    draft = draft_config.DFlash2Config.from_dict(copy.deepcopy(DRAFT_CONFIG), target_hidden=target.hidden,
        target_vocabulary=target.vocabulary, target_layers=target.num_layers, expected=draft_config.PROFILE)
    return replace(target, draft=draft)


def test_checkpoint_configs_match_the_pinned_profiles():
    assert json.loads((target_snapshot() / "config.json").read_text()) == CONFIG
    assert json.loads((draft_snapshot() / "config.json").read_text()) == DRAFT_CONFIG


def test_real_config_parses_into_the_declared_geometry():
    from ignition.recipes.ripple.config import ModelConfig

    config = ModelConfig.load(target_snapshot())
    assert (config.num_layers, config.hidden, config.heads, config.kv_heads, config.head_dim) == (80, 5120, 80, 8, 128)
    assert (config.intermediate_size, config.vocabulary, config.sliding_window) == (40960, 100352, 1024)
    assert config.embed_scale == pytest.approx(5120 ** .5) and config.clamp == 20.
    assert config.rms_norm_eps == 1e-6
    assert [config.is_sliding(index) for index in range(80)] == [index % 5 != 4 for index in range(80)]
    inverse = config.rotary_inverse("cpu")
    assert inverse.shape == (64,) and inverse[0] == 1. and inverse[-1] == pytest.approx(10000. ** (-126 / 128))


def test_optional_scale_and_clamp(target):
    from ignition.recipes.ripple.config import ModelConfig

    raw = copy.deepcopy(CONFIG)
    del raw["embed_scale"], raw["ffn_clamp_threshold"]
    config = ModelConfig.from_dict(raw)
    assert config.embed_scale is None and config.clamp is None
    raw["ffn_clamp_threshold"] = 0
    assert ModelConfig.from_dict(raw).clamp is None
    assert target.clamp == 20.


@pytest.mark.parametrize("key,value", [
    ("architectures", ["Olmo3ForCausalLM"]), ("architectures", None),
    ("quantization_config", {"quant_method": "modelopt", "quant_algo": "NVFP4"}),
    ("model_type", "olmo3"), ("hidden_act", "gelu"), ("head_dim", 256),
    ("use_gated_attention", False), ("gate_full_precision", False), ("use_valres", False),
    ("tie_word_embeddings", True), ("hidden_size", 0), ("hidden_size", 5120.), ("vocab_size", True),
    ("num_hidden_layers", 79), ("num_key_value_heads", 7), ("sliding_window", -1),
    ("max_position_embeddings", None), ("layer_types", LAYER_TYPES[:-1]),
    ("layer_types", LAYER_TYPES[:-1] + ["linear_attention"]), ("layer_types", tuple(LAYER_TYPES)),
    ("rope_theta", 0.), ("rms_norm_eps", -1e-6), ("embed_scale", 0.), ("ffn_clamp_threshold", -1.),
])
def test_unsupported_checkpoints_rejected(key, value):
    from ignition.recipes.ripple.config import ModelConfig

    raw = copy.deepcopy(CONFIG)
    raw[key] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("key", ["model_type", "use_gated_attention", "gate_full_precision", "use_valres",
                                 "hidden_size", "layer_types", "rope_theta", "rms_norm_eps"])
def test_missing_required_fields_rejected(key):
    from ignition.recipes.ripple.config import ModelConfig

    raw = copy.deepcopy(CONFIG)
    del raw[key]
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("world,q,kv,intermediate,vocabulary", [
    (1, 10240, 1024, 40960, 100352),
    (2, 5120, 512, 20480, 50176),
    (4, 2560, 256, 10240, 25088),
])
def test_tensor_parallel_partitions(target, world, q, kv, intermediate, vocabulary):
    ranges = []
    for rank in range(world):
        shard = TensorParallelShard(rank, world)
        target.validate_shard(shard)
        assert target.local_heads(shard) == 80 // world and target.local_kv_heads(shard) == 8 // world
        ranges.append((target.q_range(shard), target.kv_range(shard), target.intermediate_range(shard),
                       target.vocabulary_partition(shard)))
    assert [r[0] for r in ranges] == [(rank * q, (rank + 1) * q) for rank in range(world)]
    assert [r[1] for r in ranges] == [(rank * kv, (rank + 1) * kv) for rank in range(world)]
    assert [r[2] for r in ranges] == [(rank * intermediate, (rank + 1) * intermediate) for rank in range(world)]
    assert [r[3] for r in ranges] == [(rank * vocabulary, (rank + 1) * vocabulary, vocabulary)
                                      for rank in range(world)]
    # Every query-head slice holds whole heads, and its key/value heads cover the same GQA groups.
    assert q % 128 == 0 and kv % 128 == 0 and (q // 128) // (kv // 128) == 10
    assert ranges[-1][3][1] == target.vocabulary and ranges[-1][2][1] == target.intermediate_size


@pytest.mark.parametrize("world", [3, 8])
def test_unsupported_tensor_parallel_sizes_rejected(target, world):
    with pytest.raises(ValueError, match="world_size"):
        target.validate_shard(TensorParallelShard(0, world))


def test_shard_alignment_rejected():
    from ignition.recipes.ripple.config import ModelConfig

    for key, value, message in (("vocab_size", 100352 + 64, "vocabulary"),
                                ("intermediate_size", 40960 + 128, "intermediate_size"),
                                ("num_key_value_heads", 2, "key/value heads")):
        raw = copy.deepcopy(CONFIG) | {key: value}
        with pytest.raises(ValueError, match=message):
            ModelConfig.from_dict(raw).validate_shard(TensorParallelShard(0, 4))


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("drafting", [False, True])
def test_pages_and_state_rows_match_declared_costs(drafted, world, drafting):
    from ignition.recipes.ripple.pool import KV_DTYPE, ContinuationPool, continuation_layout, page_bytes

    recipe = MODEL.recipe
    shard = TensorParallelShard(world - 1, world)
    costs = recipe.costs(world, recipe.speculation if drafting else None)
    layout = continuation_layout(drafted, shard, draft=drafting)
    assert layout.row_bytes == costs.state_row_bytes
    assert page_bytes(drafted, shard) == costs.page_bytes == 16 * 2 * 128 * (8 // world) * 128 * 2
    pool = ContinuationPool.for_model(drafted, 3, 2, "meta", shard=shard, draft=drafting)
    assert pool.layout.row_bytes == costs.state_row_bytes
    assert len(pool.pages) == 32 and sum(page[0].nbytes for page in pool.pages.values()) == costs.page_bytes
    assert pool.kv_dtype == KV_DTYPE and pool.ring_pages == (10 if drafting else 9)
    for layer in (0, 4, 79):
        for cache in pool.attention(layer, drafted, shard):
            assert cache.dtype == KV_DTYPE and cache.shape[1:] == (128, 8 // world, 128)
    rows = pool.row_pages(drafted, shard)
    assert pool.attention(0, drafted, shard)[0].shape[0] == 2 * rows + pool.ring_pages
    assert pool.attention(4, drafted, shard)[0].shape[0] == 2
    rings = [region for region in layout.regions if region.name.startswith("draft")]
    assert len(rings) == (10 if drafting else 0)
    for region in rings:
        ring = pool.reserved_region(region.name)
        assert ring.shape == (3, 4096, 8 // world, 128) and ring.dtype == torch.float8_e4m3fn
    assert pool.reserved_region("last_hidden").shape == (3, 5120)
    assert pool.reserved_region("consumed_tokens").dtype == torch.int64


def test_draft_costs_add_rings_and_a_ring_page():
    recipe = MODEL.recipe
    for world in (1, 2, 4):
        plain, drafting = recipe.costs(world, None), recipe.costs(world, recipe.speculation)
        assert drafting.page_bytes == plain.page_bytes and drafting.graph_bytes == plain.graph_bytes
        record = 128 * (8 // world) * 128 * 2
        # Sixteen-token rounds keep one more ring page per sliding layer than single tokens.
        assert drafting.state_row_bytes - plain.state_row_bytes == 5 * 2 * 4096 * (8 // world) * 128 + 128 * record


def test_window_pages_bound_every_round(target):
    from ignition.recipes.ripple.pool import window_pages

    assert [window_pages(target, width) for width in (1, 16, 128, 2048, 4096)] == [9, 10, 10, 25, 41]
    for width in (1, 2, 15, 16, 17, 128, 1536, 2048):
        touched = max((first + width - 1) // 128 - max(first - 1023, 0) // 128 + 1 for first in range(0, 4096))
        assert touched <= window_pages(target, width)
        assert any((first + width - 1) // 128 - (first - 1023) // 128 + 1 == window_pages(target, width)
                   for first in range(1023, 4096))


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fp8_kv_costs_match_the_layout(drafted, kv_dtype):
    from ignition.recipes.ripple.pool import continuation_layout, page_bytes

    checkpoint = next(c for c in MODEL.recipe.checkpoints if c.alias == "ripple-65b-mxfp8-fp8kv")
    assert checkpoint.repository == mxfp8_checkpoint().repository and checkpoint.revision == mxfp8_checkpoint().revision
    for world in (1, 2, 4):
        shard = TensorParallelShard(0, world)
        for drafting in (False, True):
            costs = MODEL.recipe.costs(world, MODEL.recipe.speculation if drafting else None,
                                       checkpoint if kv_dtype == torch.float8_e4m3fn else None)
            assert continuation_layout(drafted, shard, draft=drafting, kv_dtype=kv_dtype).row_bytes == costs.state_row_bytes
            assert page_bytes(drafted, shard, kv_dtype) == costs.page_bytes


class _Buffers:
    def zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype)


def _ring_reads_match_history(window, metadata, history, *, key):
    """Every position each live query reads under the window holds its latest written value."""
    m = metadata
    for token in range(m.positions.numel()):
        query = int(m.positions[token])
        if query < 0:
            continue
        request = int(m.requests[token])
        for position in range(max(query - 1023, 0), query + 1):
            page = int(window.table[request, position // 128])
            assert key.view(torch.int16)[page, position % 128, 0, 0].item() == history[request][position], (
                query, position)


def _write(window, metadata, key, values):
    for token, slot in enumerate(window.slots.tolist()):
        if slot >= 0:
            key.view(torch.int16)[slot // 128, slot % 128] = values[token]


def test_rings_and_staged_prefill_read_exact_window_history(drafted):
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.ripple.pool import ContinuationPool
    from ignition.recipes.ripple.window import WindowMetadata, WindowStaging

    shard = TensorParallelShard(3, 4)
    pool = ContinuationPool.for_model(drafted, 4, 2, "cpu", shard=shard, draft=True)
    blocks, chunk, width = 64, 512, 16
    staging = WindowStaging.allocate(drafted, shard, chunk, "cpu", kv_dtype=torch.bfloat16)
    prefill = Metadata.allocate(chunk, 1, blocks, "cpu")
    verify = Metadata.allocate(2 * width, 2, blocks, "cpu", max_query_tokens=width)
    staged = WindowMetadata(prefill, pool, drafted, shard, mode="prefill", buffers=_Buffers(), staging=staging)
    rings = WindowMetadata(verify, pool, drafted, shard, mode="verify", buffers=_Buffers())
    layer = rings.layer(0)
    history = {0: {}, 1: {}}
    generator = torch.Generator().manual_seed(0)

    def value(position, version):
        # A distinct bit pattern per position and rewrite.
        return (position * 160 + version) % 32749 + 1

    # Two requests on rows 1 and 2 prefill different lengths, chunk by chunk, then verify sixteen
    # positions per round and accept a random prefix.
    ends = [0, 0]
    for request, (row, total) in enumerate(((1, 2300), (2, 1100))):
        while ends[request] < total:
            length = min(chunk - 64 * request, total - ends[request])
            tokens = torch.zeros(1, chunk, dtype=torch.int64)
            prefill.set(torch.tensor([ends[request]]), torch.tensor([length], dtype=torch.int32),
                        torch.tensor([row]), torch.zeros(1, blocks, dtype=torch.int32), tokens, torch.tensor([1]))
            staged()
            window = staged.layer(0)
            window.gather()
            values = torch.zeros(chunk, 2, 128, dtype=torch.int16)
            for token in range(length):
                position = ends[request] + token
                values[token] = value(position, 0)
                history[request][position] = value(position, 0)
            _write(staged, prefill, window.key, values)
            _ring_reads_match_history(staged, prefill, {0: history[request]}, key=window.key)
            window.commit()
            ends[request] += length
    for version in range(1, 160):
        lengths = torch.tensor([width, width], dtype=torch.int32)
        verify.set(torch.tensor(ends), lengths, torch.tensor([1, 2]), torch.zeros(2, blocks, dtype=torch.int32),
                   torch.zeros(2, width, dtype=torch.int64), torch.tensor([2]))
        rings()
        values = torch.zeros(2 * width, 2, 128, dtype=torch.int16)
        for token in range(2 * width):
            request, position = int(verify.requests[token]), int(verify.positions[token])
            values[token] = value(position, version)
            history[request][position] = value(position, version)
        _write(rings, verify, layer.key, values)
        _ring_reads_match_history(rings, verify, history, key=layer.key)
        for request in range(2):
            ends[request] += int(torch.randint(1, width + 1, (), generator=generator))


def test_real_draft_config_geometry(target):
    from ignition.recipes.ripple import draft_config

    draft = draft_config.load(draft_snapshot(), target)
    assert draft.rotary_layers == (True, True, True, True, False)
    assert draft.full_attention_layers == (4,)
    assert draft.convolution == (4, 16)
    assert (draft.layers, draft.block_size, draft.noise_token) == (5, 16, 100278)
    assert draft.target_layers == (1, 20, 39, 58, 77)
    a = draft.attention
    assert (a.heads, a.kv_heads, a.key_dim, a.value_dim, a.rotary_dim, a.theta, a.window) == (
        80, 8, 128, 128, 128, 10000., 4096)
    for world in (2, 4):
        draft.validate_shard(TensorParallelShard(world - 1, world))
        assert a.local(TensorParallelShard(0, world)) == (80 // world, 8 // world)


def test_draft_context_and_proposal_scratch_follow_the_checkpoint(drafted):
    from ignition.components.dflash2 import DraftScratch
    from ignition.components.dflash2_geometry import draft_context_specs
    from ignition.preparation.scratch import GraphBufferPool

    for world in (2, 4):
        shard = TensorParallelShard(0, world)
        specs = {name: (shape, dtype) for name, shape, dtype in draft_context_specs(drafted.draft, shard)}
        assert set(specs) == {f"draft{layer}.{part}" for layer in range(5) for part in ("key", "value")}
        assert set(specs.values()) == {((4096, 8 // world, 128), torch.float8_e4m3fn)}
        scratch = DraftScratch.allocate(drafted.draft, shard, 32, "meta", buffers=GraphBufferPool("meta").cursor())
        # Each block of four taps carries per-group coefficients for its two convolutions.
        assert scratch.convolution["coefficients"].shape == (32, 2 * 4 * 5120 // 16)
        assert scratch.query.shape == (32, 80 // world, 128)
        assert scratch.key.shape == scratch.value.shape == (32, 8 // world, 128)


def test_draft_target_mismatch_rejected(target):
    from ignition.recipes.ripple import draft_config

    for change in (dict(hidden_size=4096), dict(vocab_size=100353), dict(num_target_layers=79)):
        raw = copy.deepcopy(DRAFT_CONFIG) | change
        with pytest.raises(ValueError):
            draft_config.DFlash2Config.from_dict(raw, target_hidden=target.hidden,
                target_vocabulary=target.vocabulary, target_layers=target.num_layers, expected=draft_config.PROFILE)


def _geometry(raw, target, *, pinned):
    from ignition.recipes.ripple import draft_config

    # Without the pinned profile, only the shared geometry validator judges layer kinds and NoPE indices.
    expected = draft_config.PROFILE if pinned else {
        key: value for key, value in draft_config.PROFILE.items() if key not in ("layer_types", "no_rope_layers")}
    return draft_config.DFlash2Config.from_dict(raw, target_hidden=target.hidden,
        target_vocabulary=target.vocabulary, target_layers=target.num_layers, expected=expected)


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("value", [[5], [-1], [4, 4], [4, 2], [True], ["4"], 4, None, [4.]])
def test_invalid_nope_layers_rejected(target, value, pinned):
    raw = copy.deepcopy(DRAFT_CONFIG) | {"no_rope_layers": value}
    with pytest.raises(ValueError, match=None if pinned else "NoPE layers"):
        _geometry(raw, target, pinned=pinned)


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("layers", [
    ["sliding_attention"] * 4 + ["linear_attention"], ["sliding_attention"] * 4 + ["chunked_attention"],
    ["sliding_attention"] * 4, ["sliding_attention"] * 6, None, "full_attention",
])
def test_unknown_draft_layer_types_rejected(target, layers, pinned):
    raw = copy.deepcopy(DRAFT_CONFIG) | {"layer_types": layers}
    with pytest.raises(ValueError, match=None if pinned else "sliding or full attention"):
        _geometry(raw, target, pinned=pinned)


def test_geometry_validator_accepts_other_nope_selections(target):
    raw = copy.deepcopy(DRAFT_CONFIG) | {"no_rope_layers": [0, 4], "layer_types": ["full_attention"] * 5}
    draft = _geometry(raw, target, pinned=False)
    assert draft.rotary_layers == (False, True, True, True, False)
    assert draft.full_attention_layers == (0, 1, 2, 3, 4)
    del raw["no_rope_layers"]
    assert _geometry(raw, target, pinned=False).rotary_layers == (True,) * 5


@pytest.mark.parametrize("key,value", [
    ("block_size", 8), ("conv_kernel_size", 2), ("conv_group_size", 32), ("mask_token_id", 100277),
    ("target_layer_ids", [1, 20, 39, 58, 78]), ("selector_top_k", 16),
])
def test_incompatible_draft_projector_rejected(target, key, value):
    raw = copy.deepcopy(DRAFT_CONFIG)
    raw["dflash_config"][key] = value
    with pytest.raises(ValueError):
        _geometry(raw, target, pinned=True)


def test_qwen_dense_draft_profile_still_loads():
    from ignition.recipes.qwen38_dense import draft_config
    from ignition.recipes.qwen38_dense.config import ModelConfig
    from tests.recipes.test_qwen38_dense_dflash import DRAFT_CONFIG as QWEN_DRAFT

    qwen = ModelConfig(dict(hidden_size=5120, vocab_size=248320, num_hidden_layers=64, num_key_value_heads=4,
        head_dim=256, layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]), {})
    draft = draft_config.from_dict(copy.deepcopy(QWEN_DRAFT), qwen)
    assert draft.rotary_layers == (True,) * 5
    assert draft.full_attention_layers == ()
    assert draft.convolution == (2, 16)
    assert "no_rope_layers" not in draft_config.PROFILE
    raw = copy.deepcopy(QWEN_DRAFT) | {"no_rope_layers": [4]}
    with pytest.raises(ValueError, match="NoPE"):
        draft_config.from_dict(raw | {"no_rope_layers": [5]}, qwen)
    assert draft_config.from_dict(raw, qwen).rotary_layers == (True,) * 4 + (False,)


def test_recipe_declaration():
    from ignition.recipes.ripple.draft_config import REPOSITORY, REVISION, STORED_BYTES

    recipe, checkpoint = MODEL.recipe, MODEL.checkpoint
    assert recipe.name == "ripple" and recipe.served_model_name == "Ripple-65B-Preview"
    assert resolve("ricursiveAI/Ripple-65B-Preview").checkpoint is checkpoint
    assert (checkpoint.repository, checkpoint.revision) == (
        "ricursiveAI/Ripple-65B-Preview", "240d5966cbe00f0627c699e59ef58560b60303cf")
    assert (REPOSITORY, REVISION) == ("ricursiveAI/Ripple-65B-Preview-DFlash2",
                                      "35609b3be3dd08a822b3ba1257c1ea5dd6f87632")
    assert recipe.tensor_parallel == (1, 2, 4)
    assert (recipe.context, recipe.page_tokens, recipe.concurrency, recipe.max_concurrency, recipe.prefill_chunk) == (
        262144, 128, 4, 32, 4096)
    assert recipe.speculation.method == "dflash2" and recipe.speculation.draft_tokens == 15
    assert recipe.speculation.packed_verification and recipe.speculation.token_capacity == 64
    assert checkpoint.draft_bytes == STORED_BYTES + 100352 * 5120 * 2
    assert checkpoint.device_bytes == checkpoint.stored_bytes + 2 * 100352 * 5120 * 2 + STORED_BYTES
    for world in (2, 4):
        plain = recipe.weights_bytes(checkpoint, world, None)
        assert plain == -(-(checkpoint.stored_bytes + 100352 * 5120 * 2) // world)
        assert recipe.weights_bytes(checkpoint, world, recipe.speculation) == -(-checkpoint.device_bytes // world)


def test_declared_stored_bytes_match_the_target_tensors():
    # Stored bytes count tensor data, as the index's total_size does, not safetensors headers.
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    target = target_snapshot("config.json", "model.safetensors.index.json")
    tensors = inspect_indexed_checkpoint(target).tensors.values()
    assert {tensor.dtype for tensor in tensors} == {"BF16"}
    total = json.loads((target / "model.safetensors.index.json").read_text())["metadata"]["total_size"]
    assert sum(tensor.nbytes for tensor in tensors) == total == MODEL.checkpoint.stored_bytes


def test_declared_draft_bytes_match_the_draft_tensors():
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.ripple.draft_config import STORED_BYTES

    draft = inspect_indexed_checkpoint(draft_snapshot("config.json", "model.safetensors"))
    assert {tensor.dtype for tensor in draft.tensors.values()} == {"BF16"}
    assert sum(tensor.nbytes for tensor in draft.tensors.values()) == STORED_BYTES
    # --online-mxfp8 converts every drafter projection DraftWeights packs.
    from ignition.recipes.ripple.recipe import _DRAFT_PROJECTIONS

    packed = ("fc.weight", "_proj.weight")
    assert sum(tensor.nbytes // 2 for name, tensor in draft.tensors.items() if name.endswith(packed)) == (
        _DRAFT_PROJECTIONS)


def test_online_mxfp8_plans_the_drafter_at_mxfp8_size():
    from ignition import planning

    recipe, checkpoint = MODEL.recipe, mxfp8_checkpoint()
    saved = checkpoint.draft_bytes - checkpoint.online_mxfp8_draft_bytes
    assert 3_300 << 20 < saved < 3_400 << 20
    for world in (1, 2, 4):
        assert recipe.weights_bytes(checkpoint, world, None, online_mxfp8=True) == recipe.weights_bytes(
            checkpoint, world, None)
        assert recipe.weights_bytes(checkpoint, world, recipe.speculation, online_mxfp8=True) == -(
            -(checkpoint.device_bytes + checkpoint.expansion_bytes - saved) // world)
    # One Max-Q with the baseline export's free memory: BF16 rings then plan one full-context request.
    gpu = (planning.Gpu(0, "NVIDIA RTX PRO 6000 Blackwell Max-Q", "GPU-0", 96 << 30, 97_320 << 20, (12, 0)),)
    plans = [planning.fit(recipe, checkpoint, gpu, concurrency=4, reserve_bytes=0, speculation=recipe.speculation,
                          prefill_chunk=2048, online_mxfp8=online) for online in (False, True)]
    assert plans[0].context < recipe.context == plans[1].context


def test_eos_ids_merge_config_and_tokenizer():
    from ignition.recipes.ripple.bootstrap import _eos_ids
    from ignition.recipes.ripple.config import ModelConfig

    snapshot = target_snapshot("config.json", "tokenizer.json", "tokenizer_config.json")
    assert _eos_ids(snapshot, ModelConfig.load(snapshot)) == [100265, 100257]


def test_frontend_contract_declares_chatml_hermes_tools_and_think_tags():
    from ignition.preparation import runtime_package_pb2 as p
    from ignition.recipes.ripple.contract import FRONTEND

    assert FRONTEND["tools"]["kind"] == p.HERMES_JSON
    assert FRONTEND["reasoning"] == dict(kind=p.THINK_TAGS, open="<think>", close="</think>",
                                         wrapping_newlines=1, terminal_markers=["<tool_call>"])
    assert FRONTEND["assistant_marker"] == "<|im_start|>assistant\n"
    assert FRONTEND["message_end_markers"] == ["<|im_end|>"]
    assert FRONTEND["prefix_boundary"]["required_start"] == "<|im_start|>system\n"


def test_speculative_catalog_and_explicit_ordinary_mode():
    from ignition.recipes.ripple.bootstrap import PreparationConfig

    recipe = MODEL.recipe
    config = PreparationConfig(mtp=True, draft_snapshot=Path("draft"), max_active=recipe.concurrency,
        speculative_widths=recipe.speculation.widths, speculative_token_capacity=recipe.speculation.token_capacity,
        speculative_graph_budget=recipe.speculation.graph_budget)
    shapes = set(config.speculative_round_shapes)
    assert config.verification_width == 16
    assert {(batch, 1, batch) for batch in (1, 2, 4)} <= shapes
    assert (4, 16, 64) in shapes
    assert all(batch <= packed <= min(64, batch * width) for batch, width, packed in shapes)
    assert sum(width > 1 for _, width, _ in shapes) <= 24
    plain = PreparationConfig()
    assert plain.speculative_round_shapes == () and plain.verification_width == 1
    fixed = PreparationConfig(mtp=True, draft_snapshot=Path("draft"), speculative_widths=(1, 4, 8),
                              speculative_fixed_width=4)
    assert {width for _, width, _ in fixed.speculative_round_shapes} == {1, 4}


@pytest.mark.parametrize("options,error", [
    (dict(mtp=True), ValueError), (dict(mtp=True, draft_snapshot=Path("d"), speculative_widths=(1, 17)), ValueError),
    (dict(mtp=True, draft_snapshot=Path("d"), speculative_widths=(1, 4), speculative_fixed_width=8), ValueError),
    (dict(devices=(0, 1, 2)), ValueError), (dict(devices=(0, 0)), ValueError), (dict(context=1000), ValueError),
    (dict(prefill_chunk_size=100), ValueError), (dict(mtp=1), TypeError), (dict(online_mxfp8="yes"), TypeError),
])
def test_preparation_config_rejections(options, error):
    from ignition.recipes.ripple.bootstrap import PreparationConfig

    with pytest.raises(error):
        PreparationConfig(**options)


@pytest.mark.parametrize("drafting", [False, True])
def test_two_rtx_pro_6000_plan_fits(drafting):
    from ignition import planning

    recipe, checkpoint = MODEL.recipe, MODEL.checkpoint
    gpus = tuple(planning.Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i}", 96 << 30, 95 << 30, (12, 0))
                 for i in range(2))
    plan = planning.fit(recipe, checkpoint, gpus, concurrency=4, reserve_bytes=0,
                        speculation=recipe.speculation if drafting else None)
    assert plan.world_size == 2 and plan.concurrency == 4
    assert plan.costs == recipe.costs(2, recipe.speculation if drafting else None)
    assert plan.weights_bytes == recipe.weights_bytes(checkpoint, 2, plan.speculation)


# MXFP8 checkpoint: ModelOpt MIXED_PRECISION with every decoder projection, and only those, as MXFP8.

MXFP8_DIRECTORY = "/data/models/Ripple-65B-Preview-MXFP8-GPTQ"  # a local copy of the pinned revision


def mxfp8_config():
    from ignition.recipes.ripple.config import PROJECTIONS

    return copy.deepcopy(CONFIG) | {"quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
        "quantized_layers": {f"model.layers.{index}.{name}": {"quant_algo": "MXFP8", "weight_block_size": [1, 32]}
                             for index in range(80) for name in PROJECTIONS}}}


def mxfp8_checkpoint():
    return next(checkpoint for checkpoint in MODEL.recipe.checkpoints if checkpoint.alias == "ripple-65b-mxfp8")


def test_mxfp8_checkpoint_config_is_accepted():
    from ignition.recipes.ripple.config import ModelConfig

    assert ModelConfig.from_dict(mxfp8_config()).mxfp8_projections
    assert not ModelConfig.from_dict(copy.deepcopy(CONFIG)).mxfp8_projections


def _drop_one(layers):
    del layers["model.layers.79.mlp.down_proj"]


@pytest.mark.parametrize("mutate", [
    lambda quant: quant.update(kv_cache_quant_algo="FP8"),
    lambda quant: quant.update(quant_algo="MXFP8"),
    lambda quant: quant.update(quant_method="compressed-tensors"),
    lambda quant: quant.pop("quantized_layers"),
    lambda quant: _drop_one(quant["quantized_layers"]),
    lambda quant: quant["quantized_layers"].update({"lm_head": {"quant_algo": "MXFP8", "weight_block_size": [1, 32]}}),
    lambda quant: quant["quantized_layers"].update({"model.layers.0.self_attn.q_proj": {"quant_algo": "NVFP4",
                                                                                           "group_size": 16}}),
    lambda quant: quant["quantized_layers"].update({"model.layers.0.self_attn.v_proj": {
        "quant_algo": "MXFP8", "weight_block_size": [1, 16]}}),
])
def test_mxfp8_checkpoint_config_rejections(mutate):
    from ignition.recipes.ripple.config import ModelConfig

    raw = mxfp8_config()
    mutate(raw["quantization_config"])
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


MXFP8_SHAPES = {"self_attn.q_proj": (10240, 5120), "self_attn.g_proj": (10240, 5120),
                "self_attn.k_proj": (1024, 5120), "self_attn.v_proj": (1024, 5120),
                "self_attn.o_proj": (5120, 10240), "mlp.gate_proj": (40960, 5120),
                "mlp.up_proj": (40960, 5120), "mlp.down_proj": (5120, 40960)}


def _mxfp8_layer_loader(monkeypatch, world, *, index=7, shapes=MXFP8_SHAPES, scale_dtype=torch.uint8):
    from types import SimpleNamespace

    from ignition.recipes.ripple.config import ModelConfig
    from ignition.recipes.ripple.weights import RippleWeightLoader

    loader = object.__new__(RippleWeightLoader)
    # Stored MXFP8 is never requantized, even when online conversion is requested for the draft.
    loader.config, loader.shard, loader.online_mxfp8 = (ModelConfig.from_dict(mxfp8_config()),
                                                        TensorParallelShard(world - 1, world), True)
    loader._sources = {}
    for name, (rows, columns) in shapes.items():
        loader._sources[f"model.layers.{index}.{name}.weight"] = torch.empty(
            (rows, columns), dtype=torch.float8_e4m3fn, device="meta")
        loader._sources[f"model.layers.{index}.{name}.weight_scale"] = torch.empty(
            (rows, columns // 32), dtype=scale_dtype, device="meta")
    calls = {}

    def load_linear_group(prefixes, *, row_ranges=None, column_range=None):
        calls[tuple(name.rsplit(".", 1)[-1] for name in prefixes)] = (prefixes, tuple(row_ranges), column_range)
        out = sum((r[1] - r[0]) if r is not None else loader._sources[f"{p}.weight"].shape[0]
                  for p, r in zip(prefixes, row_ranges))
        width = loader._sources[f"{prefixes[0]}.weight"].shape[1]
        return SimpleNamespace(out_features=out,
                               in_features=width if column_range is None else column_range[1] - column_range[0])

    monkeypatch.setattr(loader, "_require_open", lambda: None)
    monkeypatch.setattr(loader, "load_linear_group", load_linear_group)
    monkeypatch.setattr(loader, "load_tensors", lambda names: [
        torch.ones(128 if ".self_attn." in name else 5120, dtype=torch.bfloat16) for name in names])
    monkeypatch.setattr(loader, "load_tensor", lambda name: torch.zeros(1, dtype=torch.bfloat16))
    return loader, calls


@pytest.mark.parametrize("world", [1, 2, 4])
def test_mxfp8_layer_loads_stored_projections_for_the_last_rank(monkeypatch, world):
    loader, calls = _mxfp8_layer_loader(monkeypatch, world)
    layer = loader.load_layer(7)
    rank = world - 1
    q = (rank * 10240 // world, (rank + 1) * 10240 // world)
    kv = (rank * 1024 // world, (rank + 1) * 1024 // world)
    ffn = (rank * 40960 // world, (rank + 1) * 40960 // world)
    a, m = "model.layers.7.self_attn", "model.layers.7.mlp"
    assert calls == {
        ("q_proj", "g_proj", "k_proj"): ((f"{a}.q_proj", f"{a}.g_proj", f"{a}.k_proj"), (q, q, kv), None),
        ("v_proj",): ((f"{a}.v_proj",), (kv,), None),
        ("o_proj",): ((f"{a}.o_proj",), (None,), q),
        ("gate_proj", "up_proj"): ((f"{m}.gate_proj", f"{m}.up_proj"), (ffn, ffn), None),
        ("down_proj",): ((f"{m}.down_proj",), (None,), ffn),
    }
    assert set(layer) == {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                          "post_feedforward_layernorm", "q_norm", "k_norm", "value_blend",
                          "qgk", "v_proj", "o_proj", "gate_up", "down_proj"}


@pytest.mark.parametrize("projection,shape", [
    ("self_attn.q_proj", (10368, 5120)), ("self_attn.k_proj", (2048, 5120)), ("self_attn.v_proj", (1024, 5152)),
    ("self_attn.o_proj", (5120, 10272)), ("mlp.up_proj", (41216, 5120)), ("mlp.down_proj", (5120, 41216)),
])
def test_mxfp8_stored_shape_mismatch_rejected_before_loading(monkeypatch, projection, shape):
    loader, calls = _mxfp8_layer_loader(monkeypatch, 2, shapes=MXFP8_SHAPES | {projection: shape})
    with pytest.raises(ValueError, match=projection):
        loader.load_layer(7)
    assert not calls


def test_mxfp8_scale_dtype_mismatch_rejected(monkeypatch):
    loader, _ = _mxfp8_layer_loader(monkeypatch, 1, scale_dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="weight_scale"):
        loader.load_layer(7)


def test_mxfp8_checkpoint_declaration():
    checkpoint = mxfp8_checkpoint()
    assert checkpoint.repository == "notSnix/Ripple-65B-Preview-MXFP8-GPTQ"
    assert len(checkpoint.revision) == 40 and int(checkpoint.revision, 16) >= 0
    # One E4M3 byte plus one UE8M0 scale byte per 32 inputs for every decoder projection; the
    # embedding, vocabulary head, norms and value-residual weights stay BF16.
    projections = 80 * (2 * 10240 * 5120 + 2 * 1024 * 5120 + 5120 * 10240 + 3 * 40960 * 5120)
    bf16 = MODEL.checkpoint.stored_bytes - 2 * projections
    assert checkpoint.stored_bytes == projections + projections // 32 + bf16
    assert checkpoint.device_bytes - checkpoint.stored_bytes == (MODEL.checkpoint.device_bytes
                                                                 - MODEL.checkpoint.stored_bytes)
    assert checkpoint.draft_bytes == MODEL.checkpoint.draft_bytes
    # Loaded scales stay resident beside their MMA-layout copy.
    assert checkpoint.expansion_bytes == projections // 32 and MODEL.checkpoint.expansion_bytes == 0


@pytest.mark.parametrize("drafting", [False, True])
def test_mxfp8_checkpoint_fits_one_rtx_pro_6000(drafting):
    from ignition import planning

    recipe = MODEL.recipe
    gpu = (planning.Gpu(0, "NVIDIA RTX PRO 6000 Blackwell", "GPU-0", 96 << 30, 95 << 30, (12, 0)),)
    speculation = recipe.speculation if drafting else None
    plan = planning.fit(recipe, mxfp8_checkpoint(), gpu, concurrency=4, reserve_bytes=0, speculation=speculation)
    assert plan.world_size == 1 and plan.costs == recipe.costs(1, speculation)
    assert plan.weights_bytes == recipe.weights_bytes(mxfp8_checkpoint(), 1, plan.speculation)
    assert plan.weights_bytes < 80 << 30


def test_mxfp8_checkpoint_tensors_match_the_declaration():
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.ripple.config import PROJECTIONS, ModelConfig

    directory = Path(os.environ.get("FLARE_RIPPLE_MXFP8_CHECKPOINT", MXFP8_DIRECTORY))
    if not (directory / "model.safetensors.index.json").is_file():
        pytest.skip(f"set FLARE_RIPPLE_MXFP8_CHECKPOINT or provide {MXFP8_DIRECTORY}")
    assert ModelConfig.load(directory).mxfp8_projections
    tensors = inspect_indexed_checkpoint(directory).tensors
    quantized = {f"model.layers.{index}.{name}" for index in range(80) for name in PROJECTIONS}
    for name, tensor in tensors.items():
        prefix, _, field = name.rpartition(".")
        expected = ({"weight": "F8_E4M3", "weight_scale": "U8"}[field] if prefix in quantized else "BF16")
        assert tensor.dtype == expected, name
    assert {f"{prefix}.{field}" for prefix in quantized for field in ("weight", "weight_scale")} <= set(tensors)
    total = json.loads((directory / "model.safetensors.index.json").read_text())["metadata"]["total_size"]
    assert sum(tensor.nbytes for tensor in tensors.values()) == total == mxfp8_checkpoint().stored_bytes
