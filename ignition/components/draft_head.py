# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact BF16 head logits for a shortlist of drafter candidates.

A DFlash2 proposal keeps only the selector's top candidates of the full-vocabulary unary
logits. An MXFP8 copy of the head ranks the vocabulary at half the weight traffic, and
this kernel recomputes the BF16 head logits of the RESCORE_CANDIDATES best approximate
candidates from the target's BF16 rows, rounded to BF16 like the GEMM output. The selector
then sees the same values for those candidates as with the BF16 head, so the drafts only
change when a true top candidate ranks below RESCORE_CANDIDATES under MXFP8. Verified output
never depends on drafts.
"""

import torch
import triton
import triton.language as tl

RESCORE_CANDIDATES = 128


@triton.jit
def _rescore(Hidden, Head, Ids, Out, hidden_stride, head_stride, CANDIDATES: tl.constexpr,
             WIDTH: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    slot = group * GROUP + tl.arange(0, GROUP)
    ids = tl.load(Ids + row * CANDIDATES + slot)
    acc = tl.zeros((GROUP,), tl.float32)
    for start in tl.static_range(0, WIDTH, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        hidden = tl.load(Hidden + row * hidden_stride + cols).to(tl.float32)
        weight = tl.load(Head + ids[:, None] * head_stride + cols[None, :]).to(tl.float32)
        acc += tl.sum(weight * hidden[None, :], 1)
    tl.store(Out + row * CANDIDATES + slot, acc.to(Out.dtype.element_ty))


@torch.library.custom_op("flare::draft_head_rescore", mutates_args=("out",))
def rescore(hidden: torch.Tensor, head: torch.Tensor, ids: torch.Tensor, out: torch.Tensor) -> None:
    rows, width = hidden.shape
    candidates = ids.shape[1]
    _rescore[(rows, candidates // 16)](hidden, head, ids, out, hidden.stride(0), head.stride(0), candidates, width,
                                        256, 16, num_warps=4)


@rescore.register_fake
def _fake_rescore(hidden, head, ids, out):
    return None


def check(hidden, head, ids, out):
    if (hidden.dim() != 2 or head.dim() != 2 or hidden.shape[1] != head.shape[1] or hidden.stride(1) != 1
            or not head.is_contiguous() or hidden.dtype != torch.bfloat16 or head.dtype != torch.bfloat16
            or ids.shape != out.shape or ids.shape[0] != hidden.shape[0] or ids.dtype != torch.int64
            or not ids.is_contiguous() or not out.is_contiguous() or ids.shape[1] % 16
            or hidden.shape[1] % 256):
        raise ValueError("draft head rescoring requires BF16 rows, a contiguous BF16 head, 16-aligned candidate "
                         "rows and a 256-aligned width")
