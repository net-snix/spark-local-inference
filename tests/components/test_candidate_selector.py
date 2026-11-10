# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Proposal temperatures scale the shortlist scores of the candidate selector."""
import pytest
import torch

from ignition.components.candidate_selector import PreparedCandidateSelector, proposal_scales


def test_proposal_scales_validate_and_invert():
    assert proposal_scales(None, 4, "cpu") is None
    assert proposal_scales((1., 1., 1., 1.), 4, "cpu") is None
    assert proposal_scales((1., 2., 4., .5), 4, "cpu").tolist() == [1., .5, .25, 2.]
    for bad in ((1., 2., 4.), (1., 0., 1., 1.), (1., -1., 1., 1.), (1., float("nan"), 1., 1.),
                (1., float("inf"), 1., 1.)):
        with pytest.raises(ValueError, match="proposal temperatures"):
            proposal_scales(bad, 4, "cpu")


def _selector(temperatures, *, batch=3, block=4, vocab=96, rank=8, top_k=5):
    generator = torch.Generator().manual_seed(77)
    selector = object.__new__(PreparedCandidateSelector)
    selector.top_k = top_k
    selector.weights = {name: torch.randn((vocab, rank), generator=generator) for name in ("predecessor", "successor")}
    hidden = torch.randn((batch, block, rank), generator=generator)
    selector.projected = torch.zeros((batch, block, rank))
    selector.unary = torch.empty((batch, block, top_k))
    selector.ids = torch.empty((batch, block, top_k), dtype=torch.int64)
    selector.projection = lambda rows: selector.projected.copy_(rows.view_as(selector.projected))
    selector.scales = proposal_scales(temperatures, block, "cpu")
    logits = torch.randn((batch, block, vocab), generator=generator)
    predecessor = torch.tensor([3, 50, 95])
    return selector, hidden, logits, predecessor


def _scores(temperatures, *, shortlisted):
    selector, hidden, logits, predecessor = _selector(temperatures)
    selector.prepare(hidden)
    output = torch.empty_like(logits)
    if shortlisted:
        selector.prepare_shortlists(logits)
    for position in range(logits.shape[1]):
        if shortlisted:
            selector.apply_shortlist(output[:, position], predecessor, position)
        else:
            output[:, position].copy_(logits[:, position])
            selector(output[:, position], predecessor, position)
    return output


@pytest.mark.parametrize("shortlisted", [True, False])
def test_proposal_temperatures_divide_shortlist_scores(shortlisted):
    # Powers of two scale FP32 values exactly.
    temperatures = (1., 2., 4., .5)
    stock, scaled = _scores(None, shortlisted=shortlisted), _scores(temperatures, shortlisted=shortlisted)
    finite = torch.isfinite(stock)
    assert torch.equal(finite, torch.isfinite(scaled)) and int(finite.sum()) == stock.shape[0] * stock.shape[1] * 5
    expected = stock / torch.tensor(temperatures)[None, :, None]
    assert torch.equal(scaled[finite], expected[finite])


def test_proposal_temperatures_agree_between_shortlist_and_sequential_paths():
    temperatures = (1., 1.15, .9, .75)
    assert torch.equal(_scores(temperatures, shortlisted=True), _scores(temperatures, shortlisted=False))
