# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ricursive Ripple-65B Preview BF16 text preparation across tensor-parallel ranks."""

from pathlib import Path
from types import MappingProxyType

from ignition.recipes.declaration import (
    Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation,
)

from .draft_config import STORED_BYTES as _DRAFT_BYTES

_GRAPH_BYTES = 4 << 30
_EMBEDDING_BYTES = 100_352 * 5_120 * 2


def _costs(world: int, *, draft: bool) -> RankCosts:
    # Paged BF16 KV for all 80 layers: 2 x 128 tokens x (8 / world) heads x 128 x 2 bytes per layer;
    # a state row holds one partial page plus the last hidden state and the consumed-token cursor,
    # and with DFlash2 the drafter's five FP8 key/value rings of 4,096 tokens.
    page = 80 * 2 * 128 * (8 // world) * 128 * 2
    rings = 5 * 2 * 4096 * (8 // world) * 128 if draft else 0
    return RankCosts(page + 10_496 + rings, page, _GRAPH_BYTES)


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


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


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
    checkpoints=(Checkpoint(alias="ripple-65b", repository="ricursiveAI/Ripple-65B-Preview",
        revision="240d5966cbe00f0627c699e59ef58560b60303cf", architecture="RicursiveForCausalLM",
        stored_bytes=129_565_378_720, device_bytes=129_565_378_720 + 2 * _EMBEDDING_BYTES + _DRAFT_BYTES,
        decode_weight_bytes=129_565_378_720, draft_bytes=_DRAFT_BYTES + _EMBEDDING_BYTES),),
    context=262144, page_tokens=128, concurrency=4, max_concurrency=32, prefill_chunk=4096,
    # DFlash2 proposes sixteen-token blocks; verification reads every weight once per round, so
    # wide rounds cost little more than one token on this bandwidth-bound dense target.
    speculation=Speculation(draft_tokens=15, token_capacity=64, packed_verification=True, graph_budget=24,
                            method="dflash2"),
    max_draft_tokens=15, verification_rows=0,
    # --online-mxfp8 stores target and draft projections as MXFP8 while loading: about half the
    # decode weight traffic, at the cost of checkpoint precision. BF16 is the default.
    online_mxfp8=True,
    ranks=MappingProxyType({world: _costs(world, draft=True) for world in (2, 4)}),
    plain_ranks=MappingProxyType({world: _costs(world, draft=False) for world in (2, 4)}),
)
