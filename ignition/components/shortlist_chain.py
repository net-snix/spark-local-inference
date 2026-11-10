# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One launch for a shortlisted DFlash2 proposal chain.

The stock chain runs about eleven small kernels per draft position (selector correction,
dense -inf row, full-vocabulary scores, row maximum, candidate shortlist, filtering and the
draw), and each position waits for the previous draw. This kernel keeps one program per
request row and walks the positions itself. It applies the same request policy in the same
order (penalties with the committed histogram and earlier drafts, temperature, grammar,
top-k with boundary ties, softmax, top-p, min-p, Philox draw at base_offset + 4 * position)
and writes the round's dense proposal rows, validity and drafted tokens. Only the selector
correction's BF16 dot product may round differently, which can change drafts but never
verified output.
"""

import torch
import triton
import triton.language as tl

from .draft_sampling import (DRAFT_CANDIDATES, _candidate_positions, _kth_largest, _orderable,
                             _selected_row)

_EMPTY = tl.constexpr(-9223372036854775808)


@triton.jit
def _shortlist_chain(Unary, Ids, Projected, Predecessor, Successor, First,
                     Slots, Generations, LiveRows, Generation, Status, TopK, Params, Histogram,
                     GrammarMask, GrammarEnabled, Seed, BaseOffset, Active, QueryLengths, Pool, Selected,
                     Probabilities, Valid, Drafts, Sampled,
                     u_row, u_pos, i_row, i_pos, p_row, p_pos, hist_stride, mask_stride, q_row, q_pos,
                     valid_row, drafts_row,
                     V: tl.constexpr, CAPACITY: tl.constexpr, BATCH: tl.constexpr, DEPTH: tl.constexpr,
                     OFFSET: tl.constexpr, RANK: tl.constexpr, SUPPORT: tl.constexpr, POOL: tl.constexpr,
                     K: tl.constexpr, PREFIX: tl.constexpr, RUNNING: tl.constexpr,
                     CODED_BF16: tl.constexpr, CORRECTION_BF16: tl.constexpr):
    row = tl.program_id(0)
    live_rows = tl.load(LiveRows)
    selected, slot, safe = _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY, BATCH)
    sharing = 0
    for other in tl.static_range(BATCH):
        other_selected, other_slot, _other_epoch = _selected_row(Slots, Generations, Generation, live_rows, other,
                                                      CAPACITY, BATCH)
        sharing += (other_selected & (other_slot == slot)).to(tl.int32)
    live = selected & (sharing <= 1) & (tl.load(Status + safe) == RUNNING)
    temperature = tl.load(Params + safe * 6)
    top_p = tl.load(Params + safe * 6 + 1)
    min_p = tl.load(Params + safe * 6 + 2)
    repetition = tl.load(Params + safe * 6 + 3)
    presence = tl.load(Params + safe * 6 + 4)
    frequency = tl.load(Params + safe * 6 + 5)
    neutral = ((repetition == 1.) & (presence.to(tl.int32, bitcast=True) == 0)
        & (frequency.to(tl.int32, bitcast=True) == 0))
    raw_top_k = tl.load(TopK + safe)
    top_k = tl.minimum(tl.maximum(tl.where(raw_top_k > 0, raw_top_k, V), 1), K)
    grammar = tl.load(GrammarEnabled + safe) != 0
    divisor = tl.where(temperature > 0, temperature, 1.)
    seed = tl.load(Seed + row)
    base_offset = tl.load(BaseOffset + row)
    previous = tl.load(First + row)
    active = tl.load(Active + row) != 0
    limit = tl.where(active, tl.load(QueryLengths + row).to(tl.int64), 0)

    support = tl.arange(0, SUPPORT)
    dims = tl.arange(0, RANK)
    index = tl.arange(0, POOL)
    rank = tl.arange(0, K)
    earlier = tl.arange(0, PREFIX)
    prefix = tl.full((PREFIX,), -1, tl.int64)
    padding = (index - SUPPORT).to(tl.int64)
    for position in range(DEPTH):
        column = position + OFFSET
        ids = tl.load(Ids + row * i_row + column * i_pos + support)
        unary = tl.load(Unary + row * u_row + column * u_pos + support)
        # Selector correction: predecessor code times the projected row, then the successor rows.
        safe_previous = tl.minimum(tl.maximum(previous, 0), V - 1)
        coded = (tl.load(Predecessor + safe_previous * RANK + dims).to(tl.float32)
                 * tl.load(Projected + row * p_row + column * p_pos + dims).to(tl.float32))
        if CODED_BF16:
            coded = coded.to(tl.bfloat16).to(tl.float32)
        successors = tl.load(Successor + ids[:, None] * RANK + dims[None, :]).to(tl.float32)
        correction = tl.sum(successors * coded[None, :], 1)
        if CORRECTION_BF16:
            correction = correction.to(tl.bfloat16).to(tl.float32)
        score = unary + correction
        # Request policy in _draft_score_values order.
        if not neutral:
            count = tl.load(Histogram + safe * hist_stride + ids)
            count += tl.sum(((ids[:, None] == prefix[None, :]) & (prefix[None, :] >= 0)
                             & (prefix[None, :] < V)).to(tl.int32), 1)
            adjusted = tl.where(score > 0, tl.math.div_rn(score, repetition), score * repetition)
            score = tl.where(count > 0, adjusted, score)
            score = score - (count > 0).to(tl.float32) * presence - count.to(tl.float32) * frequency
        score = tl.math.div_rn(score, divisor)
        if grammar:
            word = tl.load(GrammarMask + safe * mask_stride + ids // 32)
            score = tl.where(((word >> (ids % 32)) & 1) != 0, score, -float("inf"))
        maximum = tl.max(score, 0)
        finite_row = (maximum == maximum) & (tl.abs(maximum) < float("inf"))  # noqa: PLR0124
        prefix_valid = tl.sum(((earlier < position) & ((prefix < 0) | (prefix >= V))).to(tl.int32), 0) == 0
        row_valid = live & prefix_valid & finite_row

        # Candidate pool as _shortlist_candidates: the shortlist, then distinct padding tokens at -inf.
        tl.store(Pool + row * POOL + support,
                 (_orderable(score).to(tl.int64) << 32) | (2147483647 - ids).to(tl.int64))
        duplicate = tl.sum((padding[:, None] == ids[None, :]).to(tl.int32), 1) > 0
        pad = ((index >= SUPPORT) & (padding < K + SUPPORT) & (padding < V) & ~duplicate)
        negative = _orderable(tl.full((POOL,), -float("inf"), tl.float32)).to(tl.int64)
        tl.store(Pool + row * POOL + index,
                 tl.where(pad, (negative << 32) | (2147483647 - padding), _EMPTY), index >= SUPPORT)
        tl.debug_barrier()
        pool = tl.load(Pool + row * POOL + index)

        # _select_and_sample.
        present = pool != _EMPTY
        ordered = (pool >> 32).to(tl.int32)
        slot_position, take, _taken = _candidate_positions(ordered, _kth_largest(ordered, top_k), present, K)
        tl.store(Selected + row * K + rank, tl.full((K,), _EMPTY, tl.int64))
        tl.debug_barrier()
        tl.store(Selected + row * K + slot_position, pool, take)
        tl.debug_barrier()
        keys = tl.sort(tl.load(Selected + row * K + rank), descending=True)
        ordered = (keys >> 32).to(tl.int32)
        scores = tl.where(ordered < 0, ordered ^ 2147483647, ordered).to(tl.float32, bitcast=True)
        tokens = 2147483647 - (keys & 2147483647).to(tl.int32)
        kept_present = keys != _EMPTY
        finite = kept_present & (scores == scores) & (scores != float("inf")) & (scores != -float("inf"))  # noqa: PLR0124
        pivot = tl.max(tl.where(rank == top_k - 1, scores, -float("inf")), 0)
        kept = finite & (scores >= pivot)
        peak = tl.max(tl.where(kept, scores, -float("inf")), 0)
        weights = tl.where(kept, tl.exp(scores - peak), 0.)
        probabilities = weights / tl.sum(weights, 0)
        before = tl.cumsum(probabilities, 0) - probabilities
        cutoff = tl.min(tl.where(kept & (before < top_p), probabilities, float("inf")), 0)
        kept &= probabilities >= cutoff
        probabilities = tl.where(kept, probabilities, 0.)
        probabilities = probabilities / tl.sum(probabilities, 0)
        kept &= (probabilities > 0.) & (probabilities >= tl.max(probabilities, 0) * min_p)
        probabilities = tl.where(kept, probabilities, 0.)
        mass = tl.sum(probabilities, 0)
        valid = row_valid & (mass > 0.) & (mass == mass) & (mass != float("inf"))  # noqa: PLR0124
        probabilities = probabilities / tl.where(mass > 0., mass, 1.)
        offset = ((base_offset + 4 * position) & 4294967295).to(tl.uint32)
        uniform = tl.rand(seed, offset)
        cumulative = tl.cumsum(probabilities, 0)
        chosen = tl.sum(((cumulative <= uniform) & kept).to(tl.int32), 0)
        last = tl.max(tl.where(kept, rank, 0), 0)
        chosen = tl.minimum(chosen, last)
        sampled = tl.sum(tl.where(rank == chosen, tokens, 0), 0)
        sampled = tl.where(valid, sampled, 0)

        # Dense q exactly as the per-position proposal row (zeroed by the caller), then round outputs.
        q = Probabilities + row * q_row + position * q_pos
        tl.store(q + tokens, probabilities, kept & valid)
        tl.store(q + sampled, 1., ~valid)
        tl.store(Valid + row * valid_row + position, valid & active & (position < limit - 1))
        tl.store(Drafts + row * drafts_row + position, sampled.to(tl.int64))
        previous = sampled.to(tl.int64)
        prefix = tl.where(earlier == position, previous, prefix)
        tl.debug_barrier()
    tl.store(Sampled + row, previous)


class PreparedShortlistChain:
    """All draft positions of a shortlisted proposal round in one launch per request row.

    Writes what the per-position loop leaves for verification: dense proposal rows (the
    caller zeroes them first), per-position validity masked by the round's active rows and
    limits, the drafted tokens and the last draw.
    """

    def __init__(self, selector, first, state, slots, generations, live_rows, seed, base_offset, active,
                 query_lengths, probabilities, valid, drafts, sampled, *, logit_offset):
        unary, ids, projected = selector.unary, selector.ids, selector.projected
        predecessor, successor = selector.weights["predecessor"], selector.weights["successor"]
        batch, depth = drafts.shape
        if (ids.shape != unary.shape or ids.shape[0] != batch or ids.shape[2] > DRAFT_CANDIDATES
                or ids.shape[2] & (ids.shape[2] - 1) or depth + logit_offset > ids.shape[1] or depth > 16
                or projected.shape[:2] != ids.shape[:2] or projected.shape[2] != predecessor.shape[1]
                or successor.shape != predecessor.shape or not predecessor.is_contiguous()
                or not successor.is_contiguous() or unary.stride(2) != 1 or ids.stride(2) != 1
                or projected.stride(2) != 1 or probabilities.shape != (batch, depth, state.vocab_size)
                or probabilities.stride(2) != 1 or valid.shape != (batch, depth) or sampled.shape != (batch,)
                or active.shape != (batch,) or query_lengths.shape != (batch,)):
            raise ValueError("shortlist chain requires selector rows covering the proposal round outputs")
        rank = predecessor.shape[1]
        if rank & (rank - 1):
            raise ValueError("shortlist chain requires a power-of-two selector rank")
        from .sampling import RUNNING

        coded = torch.promote_types(predecessor.dtype, projected.dtype)
        product = torch.promote_types(successor.dtype, coded)
        device = ids.device
        self.pool = torch.zeros((batch, 4 * DRAFT_CANDIDATES), dtype=torch.int64, device=device)
        self.selected = torch.zeros((batch, DRAFT_CANDIDATES), dtype=torch.int64, device=device)
        self.args = (unary, ids, projected, predecessor, successor, first, slots, generations, live_rows,
                     state.generation, state.status, state.top_k, state.parameters, state.histogram,
                     state.grammar_mask, state.grammar_enabled, seed, base_offset, active, query_lengths,
                     self.pool, self.selected, probabilities, valid, drafts, sampled,
                     unary.stride(0), unary.stride(1), ids.stride(0), ids.stride(1), projected.stride(0),
                     projected.stride(1), state.histogram.stride(0), state.grammar_mask.stride(0),
                     probabilities.stride(0), probabilities.stride(1), valid.stride(0), drafts.stride(0))
        self.options = dict(V=state.vocab_size, CAPACITY=state.capacity, BATCH=batch, DEPTH=depth,
                            OFFSET=logit_offset, RANK=rank, SUPPORT=ids.shape[2], POOL=4 * DRAFT_CANDIDATES,
                            K=DRAFT_CANDIDATES, PREFIX=16, RUNNING=RUNNING,
                            CODED_BF16=coded == torch.bfloat16, CORRECTION_BF16=product == torch.bfloat16)
        self.grid = (batch, 1, 1)
        self.probabilities = probabilities
        self.kernel = None

    def prepare(self):
        from ignition.preparation.triton import prepare_triton

        self.probabilities.zero_()
        self.kernel = prepare_triton(_shortlist_chain, self.args, self.grid, num_warps=4, enable_fp_fusion=False,
                                     **self.options)

    def __call__(self):
        if self.kernel is None:
            raise RuntimeError("shortlist chain requires prepare() before capture")
        self.probabilities.zero_()
        self.kernel()
