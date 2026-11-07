# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ricursive Ripple-65B Preview BF16 text preparation across tensor-parallel ranks."""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from ignition.recipes.declaration import (
    Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation,
)

from .draft_config import STORED_BYTES as _DRAFT_BYTES

_GRAPH_BYTES = 4 << 30
_EMBEDDING_BYTES = 100_352 * 5_120 * 2
_BF16_BYTES = 129_565_378_720
# GPTQ MXFP8 build: 63.75B projection weights at one E4M3 byte plus one UE8M0 scale byte per 32,
# and the BF16 embedding, vocabulary head, norms and value-residual weights unchanged.
_MXFP8_BYTES = 65_745_715_200 + 2_058_537_120
# b12x keeps each projection's UE8M0 scales twice on the device: as loaded and in MMA layout.
_MXFP8_SCALE_BYTES = 63_753_420_800 // 32
# --online-mxfp8 also converts the drafter's fc, QKV, output and MLP projections (3,853,516,800 weights) to
# one E4M3 byte plus twice one UE8M0 byte per 32, and keeps a BF16 copy of its context K/V projections.
_DRAFT_PROJECTIONS = 3_853_516_800
_DRAFT_CONTEXT_KV_BYTES = 5 * 2 * 1_024 * 5_120 * 2
_ONLINE_MXFP8_DRAFT_BYTES = (_DRAFT_BYTES + _EMBEDDING_BYTES - _DRAFT_PROJECTIONS * 2
                             + _DRAFT_PROJECTIONS * 17 // 16 + _DRAFT_CONTEXT_KV_BYTES)
_FP8_KV_DTYPE = "float8_e4m3fn"


def _costs(world: int, *, draft: bool, kv_bytes: int = 2) -> RankCosts:
    # Paged KV for the 16 full-attention layers, and with DFlash2 the drafter's full-attention layer:
    # 2 x 128 tokens x (8 / world) heads x 128 x kv_bytes each. A state row holds the 64 sliding layers'
    # key/value rings (pool.window_pages: 10 pages of 128 tokens with 16-token DFlash2 rounds, 9 with one
    # token), one partial page per paged layer, the last hidden state and the consumed-token cursor, with
    # DFlash2 the drafter's four sliding layers' FP8 key/value rings of 4,096 tokens, all rounded up to
    # whole page records.
    record = 128 * (8 // world) * 128 * kv_bytes
    rings = 64 * 2 * (10 if draft else 9) * record
    draft_rings = 4 * 2 * 4096 * (8 // world) * 128 if draft else 0
    # The drafter's paged full-attention layer keeps FP8 E4M3 keys and values.
    draft_pages = 2 * 128 * (8 // world) * 128 if draft else 0
    page = 16 * 2 * record + draft_pages
    row = rings + page + draft_rings + 10_496
    return RankCosts(-(-row // record) * record, page, _GRAPH_BYTES)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare as prepare_model, prepare_rank

    if ranks is not None:
        raise ValueError("Ripple preparation runs on local GPUs")
    speculation = plan.speculation
    drafting = {}
    if speculation is not None:
        from huggingface_hub import constants, snapshot_download

        from .draft_config import REPOSITORY, REVISION

        drafting = dict(draft_snapshot=Path(snapshot_download(REPOSITORY, revision=REVISION,
            cache_dir=plan.hub_cache, allow_patterns=("config.json", "model.safetensors"),
            # A pinned commit is immutable; offline, the hub cache's snapshot is used without a tree listing.
            local_files_only=constants.HF_HUB_OFFLINE)),
            speculative_graph_budget=speculation.graph_budget or 24)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        native_dense_activations=plan.native_dense_activations, online_mxfp8=plan.online_mxfp8,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        tuning_replica=plan.tuning_replica, devices=tuple(range(plan.world_size)), max_active=plan.concurrency,
        kv_dtype=_FP8_KV_DTYPE if plan.checkpoint.alias in _FP8_KV_ALIASES else "bfloat16",
        minimum_pages=plan.minimum_pages, context=plan.context, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, mtp=speculation is not None,
        speculative_widths=(1,) if speculation is None else speculation.widths,
        speculative_fixed_width=None if speculation is None else speculation.fixed_width,
        speculative_token_capacity=None if speculation is None else plan.speculative_token_capacity, **drafting)
    if plan.world_size == 1:
        return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def _fp8_kv(checkpoint: Checkpoint) -> Checkpoint:
    """The same checkpoint serving E4M3 keys and values: about half the KV bytes per token, values rounded."""
    return replace(checkpoint, alias=checkpoint.alias + "-fp8kv",
        ranks=MappingProxyType({world: _costs(world, draft=True, kv_bytes=1) for world in (1, 2, 4)}),
        plain_ranks=MappingProxyType({world: _costs(world, draft=False, kv_bytes=1) for world in (1, 2, 4)}))


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


_BF16_CHECKPOINT = Checkpoint(alias="ripple-65b", repository="ricursiveAI/Ripple-65B-Preview",
    revision="240d5966cbe00f0627c699e59ef58560b60303cf", architecture="RicursiveForCausalLM",
    stored_bytes=_BF16_BYTES, device_bytes=_BF16_BYTES + 2 * _EMBEDDING_BYTES + _DRAFT_BYTES,
    decode_weight_bytes=_BF16_BYTES, draft_bytes=_DRAFT_BYTES + _EMBEDDING_BYTES,
    online_mxfp8_draft_bytes=_ONLINE_MXFP8_DRAFT_BYTES)
# A GPTQ MXFP8 build of the same checkpoint that fits one 96 GB GPU; see
# docs/model-preparation.md for how it is made and its measured KL divergence.
_MXFP8_CHECKPOINT = Checkpoint(alias="ripple-65b-mxfp8", repository="notSnix/Ripple-65B-Preview-MXFP8-GPTQ",
    revision="bfea8efd56c3df0e8b44b028e65ea83aec05753c", architecture="RicursiveForCausalLM",
    stored_bytes=_MXFP8_BYTES, device_bytes=_MXFP8_BYTES + 2 * _EMBEDDING_BYTES + _DRAFT_BYTES,
    decode_weight_bytes=_MXFP8_BYTES, draft_bytes=_DRAFT_BYTES + _EMBEDDING_BYTES,
    online_mxfp8_draft_bytes=_ONLINE_MXFP8_DRAFT_BYTES,
    expansion_bytes=_MXFP8_SCALE_BYTES)
# Opt-in E4M3 KV (aliases ending in -fp8kv): values change, so BF16 KV stays the default.
_CHECKPOINTS = (_BF16_CHECKPOINT, _MXFP8_CHECKPOINT, _fp8_kv(_BF16_CHECKPOINT), _fp8_kv(_MXFP8_CHECKPOINT))
_FP8_KV_ALIASES = frozenset(c.alias for c in _CHECKPOINTS[2:])

RECIPE = Recipe(
    name="ripple", served_model_name="Ripple-65B-Preview", prepare=prepare, usage=usage,
    # The model card (https://huggingface.co/ricursiveAI/Ripple-65B-Preview) recommends no sampling
    # parameters, so the distribution is unmodified, filters are disabled, penalties are neutral,
    # and no request range is declared.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1, min_p=0., presence_penalty=0.,
        repetition_penalty=1., frequency_penalty=0.),
    # Device bytes include the replicated embedding the vocabulary-sharded head leaves on every rank
    # and the DFlash2 drafter with its column-sharded copy of that embedding, which plans subtract
    # when speculation is off.
    checkpoints=_CHECKPOINTS,
    context=262144, page_tokens=128, concurrency=4, max_concurrency=32, prefill_chunk=4096,
    # DFlash2 proposes sixteen-token blocks; verification reads every weight once per round, so
    # wide rounds cost little more than one token on this bandwidth-bound dense target.
    speculation=Speculation(draft_tokens=15, token_capacity=64, packed_verification=True, graph_budget=24,
                            method="dflash2"),
    max_draft_tokens=15, verification_rows=0,
    # --online-mxfp8 stores target and draft projections as MXFP8 while loading: about half the
    # decode weight traffic, at the cost of checkpoint precision. BF16 is the default.
    online_mxfp8=True,
    ranks=MappingProxyType({world: _costs(world, draft=True) for world in (1, 2, 4)}),
    plain_ranks=MappingProxyType({world: _costs(world, draft=False) for world in (1, 2, 4)}),
)
