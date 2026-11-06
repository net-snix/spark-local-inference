# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Device-resident continuation state, paged full-attention KV and sliding-window rings for Ripple.

Full-attention layers keep paged keys and values for their whole history. Sliding-window
layers only ever read the last `sliding_window` positions, so each continuation row keeps
their keys and values in a ring of 128-token pages: logical page j of a request lives in
ring page j mod R of its row. Attention reads a ring through a page table in logical page
order, so the kernels see the same pages, positions and values as with paged history.

R covers every page one round of `width` query positions reads or writes under the window
(see `window_pages`), so a round never overwrites a page it reads, and positions written
beyond the accepted prefix (rejected drafts) are rewritten by later rounds, as with paged
history. Prefill chunks are wider than a ring; they stage the window in scratch pages
(see `window.py`) and commit the last R pages back.

KV is BF16, the precision the reference server uses, unless the package selects E4M3.
"""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.parallel import TensorParallelShard
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS

KV_DTYPE = torch.bfloat16
KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


def window_pages(config, tokens: int) -> int:
    """Most logical pages `tokens` consecutive query positions read under the sliding window.

    Queries at p .. p + tokens - 1 read positions p - (window - 1) .. p + tokens - 1, a span of
    window + tokens - 2 positions after the first, which touches at most that many pages plus one.
    """
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("window page count requires a positive query width")
    return (config.sliding_window - 2 + tokens + PAGE_TOKENS - 1) // PAGE_TOKENS + 1


def ring_width(config, draft: bool) -> int:
    """Query positions one round writes: a DFlash2 block when drafting, otherwise one token."""
    return config.draft.block_size if draft else 1


def page_record(config, shard, kv_dtype=KV_DTYPE) -> int:
    return PAGE_TOKENS * config.local_kv_heads(shard) * config.head_dim * kv_dtype.itemsize


def page_specs(config, shard: TensorParallelShard = TensorParallelShard(0, 1), kv_dtype=KV_DTYPE):
    size = page_record(config, shard, kv_dtype)
    for index in range(config.num_layers):
        if not config.is_sliding(index):
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (size,)


def page_bytes(config, shard: TensorParallelShard = TensorParallelShard(0, 1), kv_dtype=KV_DTYPE):
    return sum(shape[0] for _, shape in page_specs(config, shard, kv_dtype))


def continuation_layout(config, shard: TensorParallelShard = TensorParallelShard(0, 1), *, draft=False,
                        kv_dtype=KV_DTYPE):
    if kv_dtype not in KV_DTYPES:
        raise ValueError("Ripple KV must be BF16 or E4M3")
    ring = window_pages(config, ring_width(config, draft)) * PAGE_TOKENS
    heads = config.local_kv_heads(shard)
    # Rings come first, so each starts on a page record of the row.
    regions = [(f"layer{index}.{field}", (ring, heads, config.head_dim), kv_dtype)
               for index in range(config.num_layers) if config.is_sliding(index) for field in ("key", "value")]
    # A full-attention layer's partial page travels with snapshots of the row.
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, shard, kv_dtype))
    if draft:
        from ignition.components.dflash2_geometry import draft_context_specs

        # Each continuation keeps the drafter's FP8 accepted-context rings beside its pages.
        regions.extend(draft_context_specs(config.draft, shard))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16),
                    ("consumed_tokens", (1,), torch.int64)))
    layout = StateLayout.from_specs(regions)
    # Whole-record rows put every row's ring pages on one stride, so all rows' rings form one page array.
    record = page_record(config, shard, kv_dtype)
    return StateLayout(layout.regions, -(-layout.row_bytes // record) * record)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, shard: TensorParallelShard = TensorParallelShard(0, 1),
                  draft=False, kv_dtype=KV_DTYPE, **kwargs):
        return cls.allocate(continuation_layout(config, shard, draft=draft, kv_dtype=kv_dtype),
                            page_specs(config, shard, kv_dtype), slots, pages, device, **kwargs)

    def _ring(self, layer):
        return self.layout.region(f"layer{layer}.key")

    @property
    def kv_dtype(self):
        return next(r.dtype for r in self.layout.regions if r.name.startswith("layer") and r.dtype != torch.uint8)

    @property
    def ring_pages(self):
        """Pages in each sliding layer's ring."""
        return next(r.shape[0] for r in self.layout.regions
                    if r.name.startswith("layer") and r.dtype != torch.uint8) // PAGE_TOKENS

    def row_pages(self, config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
        """Ring pages between the same ring of adjacent rows."""
        record = page_record(config, shard, self.kv_dtype)
        if self.layout.row_bytes % record:
            raise ValueError("continuation rows must hold whole KV page records")
        return self.layout.row_bytes // record

    def ring_page_count(self, config, shard: TensorParallelShard = TensorParallelShard(0, 1), rows=None):
        """Pages a ring view addresses through `rows` rows (all reserved rows by default)."""
        rows = self.reserved_backing.shape[0] if rows is None else rows
        return (rows - 1) * self.row_pages(config, shard) + self.ring_pages

    def attention(self, layer, config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
        """Key and value pages of one layer: history pages, or ring pages of every reserved row."""
        heads, dim, dtype = config.local_kv_heads(shard), config.head_dim, self.kv_dtype
        if not config.is_sliding(layer):
            shape = (-1, PAGE_TOKENS, heads, dim)
            return tuple(self.pages[f"layer{layer}.{field}"].view(dtype).view(shape) for field in ("key", "value"))
        elements = self.reserved_backing.view(dtype)
        pages = self.ring_page_count(config, shard)
        views = []
        for field in ("key", "value"):
            region = self.layout.region(f"layer{layer}.{field}")
            views.append(elements.as_strided((pages, PAGE_TOKENS, heads, dim),
                (PAGE_TOKENS * heads * dim, heads * dim, dim, 1),
                storage_offset=elements.storage_offset() + region.offset // dtype.itemsize))
        return tuple(views)
