import os, pytest, torch
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("FLARE_RUN_CUDA_TESTS"),
                                reason="CUDA test")


def test_rescore_matches_bf16_head_rows():
    from ignition.components.draft_head import RESCORE_CANDIDATES, check, rescore
    torch.manual_seed(0)
    rows, width, vocab = 16, 5120, 100352
    hidden = (torch.randn(rows, width, device="cuda") * 0.5).bfloat16()
    head = (torch.randn(vocab, width, device="cuda") * 0.02).bfloat16()
    ids = torch.stack([torch.randperm(vocab, device="cuda")[:RESCORE_CANDIDATES] for _ in range(rows)])
    out = torch.empty((rows, RESCORE_CANDIDATES), dtype=torch.bfloat16, device="cuda")
    check(hidden, head, ids, out)
    rescore(hidden, head, ids, out)
    full = (hidden.float() @ head.float().T)
    expected = full.gather(1, ids).bfloat16()
    diff = (out.float() - expected.float()).abs()
    assert (diff <= expected.float().abs() * 2 ** -7 + 1e-3).all(), diff.max()


def test_mxfp8_shortlist_keeps_the_bf16_top_candidates():
    from ignition.components.draft_head import RESCORE_CANDIDATES
    from ignition.components.vocabulary import mxfp8_draft_head
    from ignition.components.linear import QuantizedProjection  # noqa: F401  (import check only)
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch
    torch.manual_seed(1)
    rows, width, vocab = 16, 5120, 8192
    head = (torch.randn(vocab, width, device="cuda") * 0.02).bfloat16()
    # Peaked rows like real proposals: a few strong directions per row.
    hidden = (torch.randn(rows, width, device="cuda") * 0.1)
    hidden += 3 * head[torch.randint(0, vocab, (rows,), device="cuda")].float() / 0.02 * 0.05
    hidden = hidden.bfloat16()
    q = quantize_mxfp8_rows_torch(head)
    scales = torch.pow(2.0, q.scale_rows.view(torch.uint8).reshape(vocab, -1).float() - 127)
    approx_head = (q.values.float().view(vocab, -1, 32) * scales[..., None]).view(vocab, width)
    exact = hidden.float() @ head.float().T
    approx = hidden.float() @ approx_head.T
    top32 = exact.topk(32, dim=-1).indices
    short = approx.topk(RESCORE_CANDIDATES, dim=-1).indices
    for r in range(rows):
        assert set(top32[r].tolist()) <= set(short[r].tolist())
