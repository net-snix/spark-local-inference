# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Device-resident continuation state and paged BF16 KV storage for Ripple.

Every layer keeps paged keys and values in BF16, the precision the reference
server uses. Sliding-window layers share the page table with the global layers,
so they hold their full history; the attention kernels read only the window.
"""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.parallel import TensorParallelShard
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS

KV_DTYPE = torch.bfloat16


def page_specs(config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
    size = PAGE_TOKENS * config.local_kv_heads(shard) * config.head_dim * KV_DTYPE.itemsize
    for index in range(config.num_layers):
        for field in ("key", "value"):
            yield f"layer{index}.{field}", (size,)


def page_bytes(config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
    return sum(shape[0] for _, shape in page_specs(config, shard))


def continuation_layout(config, shard: TensorParallelShard = TensorParallelShard(0, 1), *, draft=False):
    regions = [(name, shape, torch.uint8) for name, shape in page_specs(config, shard)]
    if draft:
        from ignition.components.dflash2_geometry import draft_context_specs

        # Each continuation keeps the drafter's FP8 accepted-context rings beside its pages.
        regions.extend(draft_context_specs(config.draft, shard))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16),
                    ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, shard: TensorParallelShard = TensorParallelShard(0, 1),
                  draft=False, **kwargs):
        return cls.allocate(continuation_layout(config, shard, draft=draft), page_specs(config, shard), slots,
                            pages, device, **kwargs)

    def attention(self, layer, config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
        shape = (-1, PAGE_TOKENS, config.local_kv_heads(shard), config.head_dim)
        return tuple(self.pages[f"layer{layer}.{field}"].view(KV_DTYPE).view(shape) for field in ("key", "value"))
