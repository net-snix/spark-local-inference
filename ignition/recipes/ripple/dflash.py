# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ripple target verification and shared DFlash2 block proposals across tensor-parallel ranks."""

from ignition.components.dflash2_preparation import DFlash2Preparation

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder


class SpeculativePreparation(DFlash2Preparation):
    """Verify DFlash2 blocks with Ripple's paged decoder.

    The drafter reads a contiguous column shard of the target's token embedding
    table, gathered across ranks, and its vocabulary-sharded output head. Ripple scales
    embeddings in its forward, not in the table, so the noise block sees the
    unscaled rows, as the embedding module returns them; scaling them measured
    the same acceptance (2026-11-05).
    """

    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence, session,
                 workspace, buffers, decoder_buffers, layer_buffers, context_buffers, verification_width,
                 runtime, vocabulary_runtime, coordinator, gather_runtime):
        def decoder(weights, tables, scratch, metadata, pool, *, mode, session, name, workspace,
                    runtime, vocabulary_runtime, coordinator):
            return PreparedDecoder(weights, scratch, metadata, pool, mode=mode, session=session, name=name,
                workspace=workspace, layer_buffers=layer_buffers, runtime=runtime, coordinator=coordinator,
                vocabulary_runtime=vocabulary_runtime, buffers=decoder_buffers.cursor())

        super().__init__(weights, pool, requests, inputs, scratch, decoder=decoder,
            embedding=weights.draft_embedding, head=weights.head, page_tokens=PAGE_TOKENS,
            completion=completion, frame=frame, sequence=sequence, session=session, workspace=workspace,
            buffers=buffers, context_buffers=context_buffers, verification_width=verification_width,
            runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator,
            block_size=weights.config.draft.block_size, gather_runtime=gather_runtime)
