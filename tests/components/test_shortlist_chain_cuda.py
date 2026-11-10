import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="CUDA test")


def _grid(shape, generator):
    # Small multiples of 1/8 keep the selector dot products exact in FP32 before BF16 rounding.
    return (torch.randint(-2, 3, shape, generator=generator, device="cuda").float() / 8).bfloat16()


@pytest.mark.parametrize("mode", ["neutral", "policy", "grammar", "invalid", "greedy"])
@torch.no_grad()
def test_chain_matches_per_position_proposals(mode):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.components.shortlist_chain import PreparedShortlistChain

    g = torch.Generator(device="cuda").manual_seed(11)
    batch, vocab, block, depth, rank, support = 2, 4096, 16, 15, 256, 32
    state = DeviceRequestState.allocate("cuda:0", vocab_size=vocab, capacity=4)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device="cuda"))
    if mode in ("policy", "grammar", "invalid"):
        state.parameters.copy_(torch.tensor([.7, .9, .05, 1.1, .2, .1], device="cuda"))
        state.top_k.fill_(8)
        state.histogram.copy_(torch.randint(0, 3, state.histogram.shape, generator=g, device="cuda").int())
    if mode == "greedy":
        state.top_k.fill_(1)
    if mode == "grammar":
        state.grammar_enabled.fill_(1)
        state.grammar_mask.copy_(torch.randint(-2**31, 2**31 - 1, state.grammar_mask.shape, generator=g,
                                               device="cuda", dtype=torch.int64).int())
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    epochs = torch.full((batch,), 7, dtype=torch.int64, device="cuda")
    live = torch.tensor([batch], dtype=torch.int32, device="cuda")
    if mode == "invalid":
        state.status[1] = 0
    ids = torch.stack([torch.stack([torch.randperm(vocab, generator=g, device="cuda")[:support]
                                    for _ in range(block)]) for _ in range(batch)]).contiguous()
    unary = torch.randn((batch, block, support), generator=g, device="cuda") * 2
    if mode == "invalid":
        unary[0, 5] = float("nan")
    projected = _grid((batch, block, rank), g)
    weights = dict(predecessor=_grid((vocab, rank), g), successor=_grid((vocab, rank), g))
    active = torch.tensor([True, True], device="cuda")
    query_lengths = torch.tensor([16, 9], dtype=torch.int32, device="cuda")
    limits = torch.where(active, query_lengths, 0)
    first = torch.where(active, torch.randint(0, vocab, (batch,), generator=g, device="cuda"), 0)
    seed = torch.tensor([5, 9], dtype=torch.int64, device="cuda")
    base_offset = torch.tensor([12, 40], dtype=torch.int64, device="cuda")

    # Reference: parallel_draft's per-position loop without block broadcast.
    logits = torch.empty((batch, 1, vocab), device="cuda")
    drafts = torch.zeros((batch, depth), dtype=torch.int64, device="cuda")
    sampled = torch.zeros(batch, dtype=torch.int64, device="cuda")
    offset = torch.zeros(batch, dtype=torch.int64, device="cuda")
    probabilities = torch.zeros((batch, depth, vocab), device="cuda")
    valid = torch.zeros((batch, depth), dtype=torch.bool, device="cuda")
    proposals = [PreparedDraftDistribution(logits, state, slots, epochs, live, drafts, seed=seed, offset=offset,
                                           sampled=sampled, prefix_offset=i) for i in range(depth)]
    for i, proposal in enumerate(proposals):
        proposal.bind_shortlist(ids[:, i + 1])
        proposal.prepare()
    drafts.zero_()
    markov = first.clone()
    for i, proposal in enumerate(proposals):
        coded = weights["predecessor"][markov] * projected[:, i + 1]
        correction = torch.bmm(weights["successor"][ids[:, i + 1]], coded.unsqueeze(-1)).squeeze(-1)
        logits[:, 0].fill_(-torch.inf)
        logits[:, 0].scatter_(1, ids[:, i + 1], unary[:, i + 1] + correction.float())
        offset.copy_(base_offset + 4 * i)
        proposal()
        drafts[:, i].copy_(sampled)
        probabilities[:, i].copy_(proposal.probabilities[:, 0])
        valid[:, i].copy_(proposal.valid[:, 0] & active & (i < limits - 1))
        markov = sampled.clone()

    selector = SimpleNamespace(unary=unary, ids=ids, projected=projected, weights=weights)
    chain_probabilities = torch.full_like(probabilities, 3.)
    chain_valid = torch.zeros_like(valid)
    chain_drafts = torch.full_like(drafts, -5)
    chain_sampled = torch.zeros_like(sampled)
    chain = PreparedShortlistChain(selector, first, state, slots, epochs, live, seed, base_offset, active,
                                   query_lengths, chain_probabilities, chain_valid, chain_drafts, chain_sampled,
                                   logit_offset=1)
    chain.prepare()
    chain()
    assert torch.equal(chain_drafts, drafts)
    assert torch.equal(chain_sampled, sampled)
    assert torch.equal(chain_valid, valid)
    assert torch.equal(chain_probabilities.view(torch.int32), probabilities.view(torch.int32))
    if mode == "invalid":
        assert not valid[1].any()
