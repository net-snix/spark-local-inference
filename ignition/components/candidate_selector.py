# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predecessor-conditioned low-rank corrections to a vocabulary shortlist."""

import torch

from .linear import UnquantizedProjection


def proposal_scales(temperatures, block, device):
    """Reciprocal proposal temperatures per block position, or None when every temperature is 1.

    The sampler writes the proposal it draws from and verification reads that same proposal, so any
    positive temperature keeps speculative sampling exact. It only moves the acceptance rate.
    """
    if temperatures is None:
        return None
    values = torch.as_tensor(temperatures, dtype=torch.float32)
    if values.shape != (block,) or not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("proposal temperatures require one positive finite value per block position")
    return None if bool((values == 1).all()) else values.reciprocal().to(device)


class PreparedCandidateSelector:
    # Reciprocal proposal temperature per block position (FP32), or None.
    scales = None

    def __init__(self, hidden, weights, *, top_k, session, name, temperatures=None):
        self.weights, self.top_k = weights, top_k
        self.projected = torch.zeros((*hidden.shape[:-1], weights["projection"].shape[0]),
                                     dtype=hidden.dtype, device=hidden.device)
        self.unary = torch.empty((*hidden.shape[:-1], top_k), dtype=torch.float32, device=hidden.device)
        self.ids = torch.empty(self.unary.shape, dtype=torch.int64, device=hidden.device)
        self.projection = UnquantizedProjection(hidden.flatten(0, -2), weights["projection"],
            self.projected.flatten(0, -2), session=session, name=name + ".hidden")
        self.scales = proposal_scales(temperatures, hidden.shape[-2], hidden.device)

    def prepare(self, hidden):
        self.projection(hidden.flatten(0, -2))

    def prepare_shortlists(self, logits):
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        self.unary.copy_(unary)
        self.ids.copy_(ids)

    def apply_shortlist(self, logits, predecessor, position):
        self._correct(logits, predecessor, position, self.unary[:, position], self.ids[:, position])

    def __call__(self, logits, predecessor, position):
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        self._correct(logits, predecessor, position, unary, ids)

    def _correct(self, logits, predecessor, position, unary, ids):
        coded = self.weights["predecessor"][predecessor] * self.projected[:, position]
        successors = self.weights["successor"][ids]
        correction = torch.bmm(successors, coded.unsqueeze(-1)).squeeze(-1)
        logits.fill_(-torch.inf)
        scores = unary + correction.float()
        if self.scales is not None:
            scores = scores * self.scales[position]
        logits.scatter_(1, ids, scores)
