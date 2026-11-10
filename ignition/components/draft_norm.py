# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One-launch RMSNorm for DFlash2 drafter rows.

Follows Qwen3RMSNorm's order of operations (FP32 mean of squares, rsqrt, cast to the
input dtype, then the weight product in that dtype) with one program per row instead of
ATen's generic reduction, which runs 16 drafter rows of 5120 in a single block. Only the
FP32 summation order differs, so drafts can change but verified output cannot.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_rows(X, W, Y, x_outer, x_inner, inner, eps, inv_width, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0).to(tl.int64)
    outer, middle = program // inner, program % inner
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    x = tl.load(X + outer * x_outer + middle * x_inner + cols, mask=mask, other=0.).to(tl.float32)
    variance = tl.sum(x * x, 0) * inv_width
    normed = (x * (1.0 / tl.sqrt_rn(variance + eps))).to(Y.dtype.element_ty)
    weight = tl.load(W + cols, mask=mask)
    tl.store(Y + program * WIDTH + cols, (weight.to(tl.float32) * normed.to(tl.float32)).to(Y.dtype.element_ty),
             mask=mask)


def _rows(x):
    """Outer/inner strides of a 2-D or 3-D view whose last dimension is contiguous."""
    if x.dim() == 2:
        return x.shape[0], 1, x.stride(0), 0
    outer, inner, _ = x.shape
    return outer, inner, x.stride(0), x.stride(1)


@torch.library.custom_op("flare::draft_rms_norm", mutates_args=())
def draft_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return out
    width = x.shape[-1]
    outer, inner, x_outer, x_inner = _rows(x)
    _rms_norm_rows[(outer * inner,)](x, weight, out, x_outer, x_inner, inner, eps, 1.0 / width, width,
                                     triton.next_power_of_2(width), num_warps=4 if width <= 1024 else 8)
    return out


@draft_rms_norm.register_fake
def _fake_draft_rms_norm(x, weight, eps):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


class DraftRMSNorm:
    """Callable drop-in for an eval Qwen3RMSNorm module on drafter activations."""

    def __init__(self, weight, eps):
        self.weight, self.variance_epsilon = weight, eps

    def reference(self, x):
        dtype = x.dtype
        hidden = x.to(torch.float32)
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * hidden.to(dtype)

    def __call__(self, x):
        if (x.device.type != "cuda" or x.dim() not in (2, 3) or x.stride(-1) != 1
                or x.dtype != self.weight.dtype or not self.weight.is_contiguous()):
            return self.reference(x)
        return draft_rms_norm(x, self.weight, self.variance_epsilon)
