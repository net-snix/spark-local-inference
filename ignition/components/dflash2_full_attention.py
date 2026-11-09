# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Noncausal DFlash2 block attention over a request's whole paged context.

A DFlash2 full-attention draft layer attends from one proposal block (an anchor and its
mask positions) to every accepted context position, then to the block itself. The context
keys and values live in the target's page records (one extra key/value field per draft
full-attention layer, addressed by the request's page table), so the layer's reach is the
request's context instead of a fixed ring. The block's own keys and values stay in scratch:
nothing is written to pages for proposals, and rejected positions never become context.

The kernel is split-KV flash decoding. Each program owns one request, one KV head, a chunk
of its query heads and one split: splits 0 .. SPLITS-1 share the context pages evenly and
split SPLITS reads the block. Partials (normalized output and log-sum-exp) are merged by a
second kernel. Softmax runs in FP32; scores and P @ V multiply in the query dtype (BF16 when
serving), as in the paged kernels.
"""

import torch
import triton
import triton.language as tl

# Splits of the context pages; each program reads about context / SPLITS tokens.
SPLITS = 16
# Query heads one program serves for its KV head: 4 x 16 block rows keep the FP32 tile at 64 x 128.
HEAD_CHUNK = 4


@triton.jit
def _attend(Q, K, V, BlockK, BlockV, Pages, Starts, Lengths, QueryStart, Acc, Lse,
            q_t, q_h, k_page, k_tok, k_head, page_row, b_t, b_h, a_s, a_t, a_h, l_s, l_t,
            scale, GROUP: tl.constexpr, CHUNK: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr,
            PAGE: tl.constexpr, SPLITS: tl.constexpr):
    request, unit, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    chunks = (GROUP + CHUNK - 1) // CHUNK
    kv_head, chunk = unit // chunks, unit % chunks
    count = tl.load(Lengths + request)
    if count > 0:
        start = tl.load(Starts + request)
        first = tl.load(QueryStart + request)
        rows = tl.arange(0, CHUNK * ROWS)
        group, query = chunk * CHUNK + rows // ROWS, rows % ROWS
        live = (group < GROUP) & (query < count)
        head = kv_head * GROUP + group
        dims = tl.arange(0, D)
        token = (first + query).to(tl.int64)
        q = tl.load(Q + token[:, None] * q_t + head[:, None] * q_h + dims[None, :], live[:, None], other=0.)
        maximum = tl.full((CHUNK * ROWS,), -float("inf"), tl.float32)
        total = tl.full((CHUNK * ROWS,), 0., tl.float32)
        acc = tl.full((CHUNK * ROWS, D), 0., tl.float32)
        if split < SPLITS:
            pages = (start + PAGE - 1) // PAGE
            share = (pages + SPLITS - 1) // SPLITS
            begin = split * share
            end = tl.minimum(begin + share, pages)
            offsets = tl.arange(0, PAGE)
            for page in range(begin, end):
                index = tl.load(Pages + request * page_row + page).to(tl.int64)
                inside = page * PAGE + offsets < start
                base = index * k_page + offsets[:, None] * k_tok + kv_head * k_head + dims[None, :]
                k = tl.load(K + base, inside[:, None], other=0.).to(Q.dtype.element_ty)
                scores = tl.dot(q, tl.trans(k)) * scale
                scores = tl.where(inside[None, :], scores, -float("inf"))
                peak = tl.maximum(maximum, tl.max(scores, 1))
                weights = tl.exp(scores - peak[:, None])
                correction = tl.exp(maximum - peak)
                v = tl.load(V + base, inside[:, None], other=0.).to(Q.dtype.element_ty)
                acc = acc * correction[:, None] + tl.dot(weights.to(Q.dtype.element_ty), v)
                total = total * correction + tl.sum(weights, 1)
                maximum = peak
        else:
            keys = tl.arange(0, ROWS)
            present = keys < count
            base = (first + keys).to(tl.int64)[:, None] * b_t + kv_head * b_h + dims[None, :]
            k = tl.load(BlockK + base, present[:, None], other=0.)
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(present[None, :], scores, -float("inf"))
            maximum = tl.max(scores, 1)
            weights = tl.exp(scores - maximum[:, None])
            v = tl.load(BlockV + base, present[:, None], other=0.)
            acc = tl.dot(weights.to(Q.dtype.element_ty), v)
            total = tl.sum(weights, 1)
        empty = total == 0.
        out = acc / tl.where(empty, 1., total)[:, None]
        lse = tl.where(empty, -float("inf"), maximum + tl.log(tl.where(empty, 1., total)))
        tl.store(Acc + split * a_s + token[:, None] * a_t + head[:, None] * a_h + dims[None, :], out, live[:, None])
        tl.store(Lse + split * l_s + token * l_t + head, lse, live)


@triton.jit
def _merge(Acc, Lse, Lengths, QueryStart, Out, a_s, a_t, a_h, l_s, l_t, o_t, o_h,
           ROWS: tl.constexpr, D: tl.constexpr, PARTS: tl.constexpr):
    request, head = tl.program_id(0), tl.program_id(1)
    count = tl.load(Lengths + request)
    if count > 0:
        first = tl.load(QueryStart + request)
        query = tl.arange(0, ROWS)
        live = query < count
        token = (first + query).to(tl.int64)
        dims = tl.arange(0, D)
        peak = tl.full((ROWS,), -float("inf"), tl.float32)
        for part in tl.static_range(PARTS):
            peak = tl.maximum(peak, tl.load(Lse + part * l_s + token * l_t + head, live, other=-float("inf")))
        weight_sum = tl.full((ROWS,), 0., tl.float32)
        out = tl.full((ROWS, D), 0., tl.float32)
        for part in tl.static_range(PARTS):
            lse = tl.load(Lse + part * l_s + token * l_t + head, live, other=-float("inf"))
            weight = tl.where(lse == -float("inf"), 0., tl.exp(lse - peak))
            values = tl.load(Acc + part * a_s + token[:, None] * a_t + head * a_h + dims[None, :],
                             live[:, None], other=0.)
            out += weight[:, None] * values
            weight_sum += weight
        out = out / tl.where(weight_sum > 0., weight_sum, 1.)[:, None]
        tl.store(Out + token[:, None] * o_t + head * o_h + dims[None, :], out.to(Out.dtype.element_ty), live[:, None])


def attend(query, key_pages, value_pages, page_table, starts, lengths, query_start, block_key, block_value,
           output, accumulators, lse, *, rows):
    """Write `output` for each request's `lengths` block rows (0 skips it).

    query/output [tokens, heads, D] BF16; key/value_pages [pages, PAGE, kv_heads, D] BF16 or E4M3;
    page_table [requests, blocks]; starts = context tokens before the block; query_start = first
    query row per request; block_key/value [tokens, kv_heads, D] BF16; accumulators [SPLITS + 1,
    tokens, heads, D] FP32 and lse [SPLITS + 1, tokens, heads] FP32 are scratch.
    """
    requests = starts.shape[0]
    heads, kv_heads, dim = query.shape[1], key_pages.shape[2], query.shape[2]
    group = heads // kv_heads
    units = kv_heads * triton.cdiv(group, HEAD_CHUNK)
    wrap = torch.library.wrap_triton
    wrap(_attend)[(requests, units, SPLITS + 1)](
        query, key_pages, value_pages, block_key, block_value, page_table, starts, lengths, query_start,
        accumulators, lse, query.stride(0), query.stride(1), key_pages.stride(0), key_pages.stride(1),
        key_pages.stride(2), page_table.stride(0), block_key.stride(0), block_key.stride(1),
        accumulators.stride(0), accumulators.stride(1), accumulators.stride(2), lse.stride(0), lse.stride(1),
        dim ** -0.5, group, HEAD_CHUNK, rows, dim, key_pages.shape[1], SPLITS, num_warps=4, num_stages=2)
    wrap(_merge)[(requests, heads)](
        accumulators, lse, lengths, query_start, output, accumulators.stride(0), accumulators.stride(1),
        accumulators.stride(2), lse.stride(0), lse.stride(1), output.stride(0), output.stride(1),
        rows, dim, SPLITS + 1, num_warps=4)
    return output


def reference(query, key_pages, value_pages, page_table, starts, lengths, query_start, block_key, block_value):
    """FP32 eager reference: each block row attends to its whole context and the whole block."""
    out = torch.zeros_like(query, dtype=torch.float32)
    heads, kv_heads, page = query.shape[1], key_pages.shape[2], key_pages.shape[1]
    for request in range(starts.shape[0]):
        count, start, first = int(lengths[request]), int(starts[request]), int(query_start[request])
        if count == 0:
            continue
        positions = torch.arange(start, device=query.device)
        slots = page_table[request, positions // page].long()
        offsets = positions % page
        keys = torch.cat((key_pages[slots, offsets].float(), block_key[first:first + count].float()))
        values = torch.cat((value_pages[slots, offsets].float(), block_value[first:first + count].float()))
        keys = keys.repeat_interleave(heads // kv_heads, 1)
        values = values.repeat_interleave(heads // kv_heads, 1)
        q = query[first:first + count].float()
        weights = torch.einsum("qhd,khd->hqk", q, keys) * query.shape[2] ** -0.5
        out[first:first + count] = torch.einsum("hqk,khd->qhd", weights.softmax(-1), values)
    return out


class PreparedDraftFullAttention:
    """Graph-capturable full-context attention for one DFlash2 draft layer and one proposal shape.

    `metadata` is the proposal batch's (starts = context tokens, lengths = block rows, page tables);
    `accumulators` / `lse` are split scratch shared by every layer and shape of a drafter.
    """

    def __init__(self, query, key_pages, value_pages, metadata, block_key, block_value, output, *, rows,
                 accumulators, lse):
        tokens, heads, dim = query.shape
        if (key_pages.shape != value_pages.shape or key_pages.ndim != 4 or key_pages.shape[3] != dim
                or block_key.shape != block_value.shape or block_key.shape[0] != tokens
                or block_key.shape[1:] != key_pages.shape[2:] or heads % key_pages.shape[2]
                or output.shape != query.shape or dim & (dim - 1) or key_pages.shape[1] & (key_pages.shape[1] - 1)
                or query.dtype != torch.bfloat16 or block_key.dtype != torch.bfloat16
                or key_pages.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or rows & (rows - 1)):
            raise ValueError("DFlash2 full attention requires BF16 queries and blocks over BF16/E4M3 pages")
        if (accumulators.shape[0] != SPLITS + 1 or accumulators.shape[1] < tokens
                or accumulators.shape[2:] != (heads, dim) or lse.shape != accumulators.shape[:3]
                or accumulators.dtype != torch.float32 or lse.dtype != torch.float32):
            raise ValueError("DFlash2 full attention scratch must hold FP32 partials of every split")
        for tensor in (query, block_key, block_value, output):
            if tensor.stride(-1) != 1:
                raise ValueError("DFlash2 full attention requires unit-stride head dimensions")
        self.query, self.key_pages, self.value_pages = query, key_pages, value_pages
        self.metadata, self.block_key, self.block_value, self.output = metadata, block_key, block_value, output
        self.rows, self.accumulators, self.lse = rows, accumulators, lse

    def __call__(self):
        m = self.metadata
        return attend(self.query, self.key_pages, self.value_pages, m.page_table, m.starts, m.lengths,
                      m.query_start_loc, self.block_key, self.block_value, self.output, self.accumulators,
                      self.lse, rows=self.rows)
