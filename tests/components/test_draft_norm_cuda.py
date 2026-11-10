import os, pytest, torch
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("FLARE_RUN_CUDA_TESTS"),
                                reason="CUDA test")


@pytest.mark.parametrize("shape,view", [((16, 5120), None), ((16, 80, 128), None), ((16, 8, 128), None),
                                        ((16, 12288), (0, 10240, 80, 128)), ((1, 5120), None)])
def test_draft_rms_norm_close_to_module(shape, view):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from ignition.components.draft_norm import DraftRMSNorm
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda").bfloat16() * 3
    if view is not None:
        start, stop, heads, dim = view
        x = x[:, start:stop].view(shape[0], heads, dim)
    width = x.shape[-1]
    weight = (torch.randn(width, device="cuda") * 0.2 + 1).bfloat16()
    module = Qwen3RMSNorm(width, eps=1e-6).cuda().bfloat16()
    module.weight.data.copy_(weight)
    expected = module(x)
    actual = DraftRMSNorm(weight, 1e-6)(x)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    diff = (actual.float() - expected.float()).abs()
    assert (diff <= expected.float().abs() * 2 ** -7 + 1e-6).all(), diff.max()
    assert (actual == expected).float().mean() > 0.97
