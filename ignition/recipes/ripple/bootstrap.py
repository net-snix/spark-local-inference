# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a single- or multi-rank Ripple text graph catalog."""

import contextlib
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.capacity import (
    batch_capacities, fine_prefill_capacities, parallel_speculative_shapes, request_capacity, speculative_token_capacity,
)
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.dflash2 import PreparedContext
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel_draft import PreparedParallelDraftRound
from ignition.components.parallel import TensorParallelShard
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime, create_dma_all_reduce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .pool import KV_DTYPES, ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .window import WindowStaging


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    tuning_replica: TuningReplica | None = None
    device: int = 0
    devices: tuple[int, ...] | None = None
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    minimum_pages: int = 8
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    online_mxfp8: bool = False
    mtp: bool = False
    draft_snapshot: Path | None = None
    speculative_graph_budget: int = 64
    speculative_widths: tuple[int, ...] = (1,)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    kv_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.devices is not None and (not isinstance(self.devices, tuple) or len(self.devices) not in (1, 2, 4)
                                         or len(set(self.devices)) != len(self.devices)
                                         or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("Ripple requires one, two, or four distinct CUDA devices")
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 2097152 or self.context % PAGE_TOKENS:
            raise ValueError("Ripple context must be 128-aligned in [128, 2097152]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Ripple prefill chunk must be 128-aligned and fit the context")
        if type(self.minimum_pages) is not int or self.minimum_pages <= 0:
            raise ValueError("minimum page capacity must be a positive integer")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if self.kv_dtype not in {str(dtype).removeprefix("torch.") for dtype in KV_DTYPES}:
            raise ValueError("Ripple KV must be bfloat16 or float8_e4m3fn")
        if type(self.mtp) is not bool or type(self.online_mxfp8) is not bool:
            raise TypeError("DFlash2 and MXFP8 selections must be booleans")
        if self.mtp and self.draft_snapshot is None:
            raise ValueError("Ripple DFlash2 preparation requires its pinned draft checkpoint")
        if self.mtp:
            if max(self.speculative_widths) > 16:
                raise ValueError("Ripple DFlash2 verifies at most sixteen tokens")
            parallel_speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity,
                self.speculative_graph_budget, fixed_width=self.speculative_fixed_width)
            if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
                raise ValueError("fixed verification width must belong to the prepared catalog")

    @property
    def speculative_round_shapes(self):
        return (parallel_speculative_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget,
            fixed_width=self.speculative_fixed_width) if self.mtp else ())

    @property
    def verification_width(self):
        return max(self.speculative_widths) if self.mtp else 1


def _eos_ids(snapshot, config):
    raw = config.text.get("eos_token_id")
    ids = list(raw if isinstance(raw, list) else [] if raw is None else [raw])
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
    text = tokenizer.get("eos_token")
    if isinstance(text, dict):
        text = text.get("content")
    if text:
        from tokenizers import Tokenizer

        token = Tokenizer.from_file(str(snapshot / "tokenizer.json")).token_to_id(text)
        if token is not None:
            ids.append(token)
    return list(dict.fromkeys(ids))


@torch.no_grad()
def prepare_rank(snapshot, config, *, rank=0, group=None, progress=None, package_storage=None):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, 1 if config.devices is None else len(config.devices))
    if (shard.world_size > 1) != (group is not None):
        raise ValueError("Ripple tensor-parallel preparation requires an established control group")
    device_id = config.device if config.devices is None else config.devices[rank]
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    if package_storage is not None:
        transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress,
        draft_snapshot=config.draft_snapshot if config.mtp else None, online_mxfp8=config.online_mxfp8)
    release_unused()
    context = min(config.context, weights.config.text["max_position_embeddings"]) // PAGE_TOKENS * PAGE_TOKENS
    if config.prefill_chunk_size > context:
        raise ValueError("Ripple prefill chunk size exceeds the prepared context")
    kv_dtype = getattr(torch, config.kv_dtype)
    layout = continuation_layout(weights.config, shard, draft=config.mtp, kv_dtype=kv_dtype)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if package_storage is None:
        snapshots = min(snapshots, 256)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Ripple continuation")
    slots, blocks = request_capacity(config.max_active), context // PAGE_TOKENS
    tokens = max(config.prefill_chunk_size, slots)
    state_rows = slots + snapshots
    prep_pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    pages = max(prep_pages, config.minimum_pages)
    initial_pages = prep_pages if package_storage is not None else pages
    bytes_per_page = page_bytes(weights.config, shard, kv_dtype, draft=config.mtp)
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, state_rows + 1, initial_pages, device, shard=shard,
            draft=config.mtp, kv_dtype=kv_dtype,
            storage=package_storage, maximum_pages=max(pages, total_memory // bytes_per_page),
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots) if package_storage is not None else None)
        # Every prefill graph stages its chunk's sliding window in these pages, one layer at a time.
        staging = WindowStaging.allocate(weights.config, shard, config.prefill_chunk_size, device, kv_dtype=kv_dtype)
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        rounds = config.speculative_round_shapes
        verify_tokens = max((packed for _, _, packed in rounds), default=slots)
        verify_samples = max((batch * width for batch, width, _ in rounds), default=slots)
        draft_rows = slots * weights.config.draft.block_size if config.mtp else 0
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + config.verification_width),
                                                device=device)
        coordinator = CollectivePreparation(group) if group is not None else None
        ipc = None if package_storage is None else package_storage.peer_storage
        collective = prefill_collective = vocabulary_collective = gather_collective = None
        local_vocab = weights.config.vocabulary_partition(shard)[2]
        if group is not None:
            collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning,
                max_tokens=max(tokens, verify_tokens, draft_rows), hidden_size=weights.config.hidden, ipc=ipc)
            prefill_collective = create_dma_all_reduce_runtime(group, device, max_tokens=config.prefill_chunk_size,
                hidden_size=weights.config.hidden, ipc=ipc)
            vocabulary_collective = create_all_gather_runtime(group, device,
                max_rows=max(slots, verify_samples, draft_rows), columns=local_vocab * shard.world_size, ipc=ipc)
            if config.mtp:
                # Draft projections leave column shards of the hidden state: context rows up to one
                # retained window per prefill chunk, and every proposal row.
                window = weights.config.draft.attention.window
                gather_collective = create_all_gather_runtime(group, device,
                    max_rows=max(verify_tokens, draft_rows, min(config.prefill_chunk_size, window)),
                    columns=weights.config.hidden, ipc=ipc)
        # The identity names the tuning namespace, so the KV layout stays out of it: kernels whose shapes
        # are unchanged keep their tuned plans. The runtime fingerprints the cache geometry separately.
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": [s.header_sha256 for s in manifest.shards],
            "world_size": shard.world_size, "contract": "ripple.graph-inputs.v1",
            "draft": None if weights.config.draft is None else weights.config.draft.text,
            "draft_headers": ([s.header_sha256 for s in inspect_indexed_checkpoint(config.draft_snapshot).shards]
                              if config.mtp else []),
            "verification_width": config.verification_width, "online_mxfp8": config.online_mxfp8},
            sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder, compile_workers=config.compile_workers,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            tuning_replica=config.tuning_replica, device=device, checkpoint=identity, workload_schema="ripple",
            numerical_policy="default", coordinator=coordinator, headroom_bytes=config.headroom_bytes,
            benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch_pool = contextlib.nullcontext() if package_storage is None else package_storage.scratch_pool()
        scratch_buffers, graph_buffers, layer_buffers = (GraphBufferPool(device) for _ in range(3))
        target_buffers, context_buffers, draft_buffers = (GraphBufferPool(device) for _ in range(3))
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame, submission_sequence=sequence, **completion.buffers())
        speculation = None
        if config.mtp:
            from .dflash import SpeculativePreparation

            with scratch_pool:
                verify_scratch = Scratch.allocate(verify_tokens, verify_samples, weights.config,
                                                  scratch_buffers.cursor(), shard=shard)
            speculation = SpeculativePreparation(weights, pool, requests, inputs, verify_scratch,
                completion=completion, frame=frame, sequence=sequence, session=session, workspace=arena,
                buffers=draft_buffers.cursor(), decoder_buffers=target_buffers, layer_buffers=layer_buffers,
                context_buffers=context_buffers, verification_width=config.verification_width,
                runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                gather_runtime=gather_collective)
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group, ipc=ipc,
                                     runtime=collective, coordinator=coordinator)
    forwards, signatures, metadata = {}, {}, {}
    prefill_sizes, batch_sizes = fine_prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    modes = (True, False) if config.mtp else (False,)
    count = (len(modes) * (prefill_graph_count(prefill_sizes, scoring) + 4 * len(batch_sizes))
             + len(batch_sizes) + len(config.speculative_round_shapes))
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens_count, fields, enabled=False, packed=0):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens_count, tuple(fields), speculation_enabled=enabled,
                                              packed_token_capacity=packed)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            # Speculative serving decodes through verification rounds; its prefill feeds the drafter.
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, sizes in prefill_modes(prefill_sizes, () if enabled else reversed(batch_sizes), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    cursor = graph_buffers.cursor()
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=cursor)
                    with scratch_pool:
                        scratch = Scratch.allocate(capacity, rows, weights.config, scratch_buffers.cursor(),
                                                   shard=shard)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None, buffers=cursor, staging=staging,
                        runtime=prefill_collective if mode == "prefill" and capacity >= 1024 else collective,
                        vocabulary_runtime=vocabulary_collective, coordinator=coordinator, session=session,
                        name=name, workspace=arena, layer_buffers=layer_buffers)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool,
                                          buffers=cursor)
                    absorption = (partial(PreparedContext, runtime=gather_collective, coordinator=coordinator,
                        buffers=context_buffers.cursor(),
                        compact_prefill=mode == "prefill" and not pool.draft_full_context) if enabled else None)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        buffers=cursor, context_factory=absorption)
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, forward_inputs, enabled)
        if speculation is not None:
            for batch_rows, width, packed in sorted(config.speculative_round_shapes,
                    key=lambda shape: (shape[2], shape[0]), reverse=True):
                name = f"speculative{batch_rows}x{width}p{packed}"
                forward = PreparedParallelDraftRound(speculation, batch_rows, width, name,
                    target_token_capacity=packed, buffers=graph_buffers.cursor())
                forward.prepare()
                add(name, forward, "speculative", batch_rows, width, forward_inputs, True, packed)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, activation_mode="auto",
                    buffers=graph_buffers.cursor())
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame,
                        sequence, runtime=collective, coordinator=coordinator, session=session, name=name,
                        capacity=capacity, admission_buffers=admission, draft_state=enabled,
                        buffers=graph_buffers.cursor())
                    fields = (row_inputs if mode == "snapshot"
                              else ("request_state_rows", *admission.__dataclass_fields__))
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream = torch.cuda.Stream(device=device)
    graph_pool = torch.cuda.graph_pool_handle() if package_storage is None else package_storage.graph_pool.id
    if config.compile_decoder and package_storage is not None:
        scratch_pool.release_backing()

    def materialize():
        if package_storage is not None:
            scratch_pool.materialize()
            if pages > initial_pages:
                package_storage.grow("KV_PAGES", pages)
                pool.memory_capacity[1] = pages
        pool.clear_for_preparation()

    try:
        graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
            pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
            progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
            materialize=materialize, warmup_storage=package_storage,
            compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative"))
    finally:
        if package_storage is not None:
            scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    assets = {
        "tokenizer": {"path": str(snapshot / "tokenizer.json"),
                      "sha256": hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()},
        "chat_template": annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(),
                                                  Path(__file__).with_name("assets") / "chat_template.patch.json"),
    }
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, eos_token_ids=_eos_ids(snapshot, weights.config)),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary,
                                        world_size=shard.world_size, mtp=config.mtp))
    if config.mtp:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, config.verification_width, config.speculative_token_capacity))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=device_id, weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes, row_bytes=layout.row_bytes,
        pages=pages, page_bytes=bytes_per_page, page_tokens=PAGE_TOKENS, kv_bytes=pages * bytes_per_page,
        kv_dtype=config.kv_dtype, ring_pages=pool.ring_pages, window_staging_bytes=staging.key.nbytes * 2,
        completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder,
        speculative_round_shapes=config.speculative_round_shapes,
        draft_buffer_bytes=draft_buffers.nbytes, target_buffer_bytes=target_buffers.nbytes,
        context_buffer_bytes=context_buffers.nbytes)
    if scoring is not None:
        memory.update(scoring.memory_report())
    close_resources, seen = [], set()
    for resource in (scoring, prefill_collective, vocabulary_collective, gather_collective, collective):
        if resource is not None and id(resource) not in seen:
            seen.add(id(resource))
            close_resources.append(resource)
    return PreparedModel(
        rebuild_weights=declare_weights(weights, components=("embedding", "norm", "head",
            *(("draft", "draft_embedding") if weights.draft is not None else ()),
            *(("draft_head",) if weights.draft_head is not None else ()))),
        device=device_id, graphs=graphs, buffers=buffers, signatures=signatures, max_active=config.max_active,
        session=session, server_config=server_config, memory_report=memory, close_resources=tuple(close_resources),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
            *storage_ranges([arena.tensor, metadata, staging.key, staging.value, *[chunk for p in (scratch_buffers, graph_buffers, layer_buffers,
                                                                target_buffers, context_buffers, draft_buffers)
                                                     for chunks in p.chunks.values() for chunk in chunks],
                             () if speculation is None else speculation.draft_scratch.scratch_tensors(),
                             [f.scratch_tensors() for f in forwards.values() if hasattr(f, "scratch_tensors")]]),
            *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, scratch_buffers,
                   graph_buffers, layer_buffers, metadata, compiled, stream, graph_pool, session, coordinator,
                   collective, prefill_collective, vocabulary_collective, gather_collective, scratch_pool,
                   target_buffers, context_buffers, draft_buffers, speculation, staging))


def prepare(snapshot, config, *, progress=None, package_storage):
    return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, package_storage=package_storage)
