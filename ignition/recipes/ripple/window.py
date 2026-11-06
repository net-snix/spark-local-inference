# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window rings for Ripple: ring page tables, write slots and prefill staging.

Decode and verification write and read a request's ring directly: logical page j maps to
ring page j mod R of the request's continuation row (`pool.py`). A prefill chunk spans more
pages than a ring holds, so it gathers the window's prefix pages into scratch pages at the
same logical offsets, writes and attends there, and commits the last R logical pages back.
Either way the attention kernels read identical logical pages, so a sliding layer computes
exactly what it computes over paged history.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .config import PAGE_TOKENS
from .pool import window_pages

_COPY_BLOCK = 4096


@triton.jit
def _copy_pages(Source, Destination, SourcePages, DestinationPages, SOURCE_STRIDE: tl.constexpr,
                DESTINATION_STRIDE: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    index, tile = tl.program_id(0), tl.program_id(1)
    source = tl.load(SourcePages + index).to(tl.int64)
    destination = tl.load(DestinationPages + index).to(tl.int64)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    mask = (source >= 0) & (destination >= 0) & (columns < PAGE)
    values = tl.load(Source + source * SOURCE_STRIDE + columns, mask, other=0)
    tl.store(Destination + destination * DESTINATION_STRIDE + columns, values, mask)


@torch.library.custom_op("ripple::copy_pages", mutates_args=("destination",))
def copy_pages(source: torch.Tensor, destination: torch.Tensor, source_pages: torch.Tensor,
               destination_pages: torch.Tensor) -> None:
    """Copy whole KV pages, skipping pairs with a negative page; pairs must not share a destination."""
    if not source.is_cuda:
        for read, write in zip(source_pages.tolist(), destination_pages.tolist(), strict=True):
            if read >= 0 and write >= 0:
                destination[write] = source[read]
        return
    words, targets = source.view(torch.int32), destination.view(torch.int32)
    page = words[0].numel()
    _copy_pages[(source_pages.numel(), triton.cdiv(page, _COPY_BLOCK))](words, targets, source_pages,
        destination_pages, words.stride(0), targets.stride(0), page, _COPY_BLOCK)


@copy_pages.register_fake
def _copy_pages_fake(source, destination, source_pages, destination_pages):
    pass


@dataclass(frozen=True)
class WindowStaging:
    """Scratch pages for one prefill chunk's window, shared by every sliding layer and prefill graph."""

    key: torch.Tensor
    value: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, chunk, device, *, kv_dtype):
        shape = (window_pages(config, chunk), PAGE_TOKENS, config.local_kv_heads(shard), config.head_dim)
        return cls(torch.zeros(shape, dtype=kv_dtype, device=device),
                   torch.zeros(shape, dtype=kv_dtype, device=device))

    def pages(self, chunk, config):
        count = window_pages(config, chunk)
        if count > self.key.shape[0]:
            raise ValueError("prefill chunk exceeds the window staging capacity")
        return self.key[:count], self.value[:count]


@dataclass(frozen=True)
class LayerWindow:
    """What one sliding layer writes and attends over, with the staging copies around it."""

    key: torch.Tensor
    value: torch.Tensor
    pages: torch.Tensor
    slots: torch.Tensor
    tuning_page_count: int
    ring: tuple | None = None
    owner: "WindowMetadata | None" = None

    def gather(self):
        if self.ring is not None:
            o = self.owner
            copy_pages(self.ring[0], self.key, o.gather_source, o.gather_destination)
            copy_pages(self.ring[1], self.value, o.gather_source, o.gather_destination)

    def commit(self):
        if self.ring is not None:
            o = self.owner
            copy_pages(self.key, self.ring[0], o.commit_source, o.commit_destination)
            copy_pages(self.value, self.ring[1], o.commit_source, o.commit_destination)


class WindowMetadata:
    """Per-forward ring page tables and slots shared by every sliding layer of one graph."""

    def __init__(self, metadata, pool, config, shard, *, mode, buffers, staging=None):
        self.metadata, self.pool, self.config, self.shard = metadata, pool, config, shard
        self.ring, self.stride = pool.ring_pages, pool.row_pages(config, shard)
        self.window = config.sliding_window
        self.blocks = metadata.page_table.shape[1]
        self.staged = mode == "prefill"
        self.table = buffers.zeros(metadata.page_table.shape, torch.int32)
        self.slots = buffers.zeros(metadata.positions.shape, torch.int64).fill_(-1)
        if self.staged:
            if staging is None or metadata.page_table.shape[0] != 1:
                raise ValueError("Ripple prefill stages one sequence's window in shared scratch pages")
            self.key, self.value = staging.pages(metadata.positions.numel(), config)
            # Prefix pages a chunk reads: the window before its first position.
            self.prefix = window_pages(config, 1)
            self.gather_source = buffers.zeros((self.prefix,), torch.int64)
            self.gather_destination = buffers.zeros((self.prefix,), torch.int64)
            self.commit_source = buffers.zeros((self.ring,), torch.int64)
            self.commit_destination = buffers.zeros((self.ring,), torch.int64)
        else:
            self.backed_pages = pool.ring_page_count(config, shard, int(pool.memory_capacity[0].item()))

    def layer(self, index):
        if self.staged:
            return LayerWindow(self.key, self.value, self.table, self.slots, self.key.shape[0],
                               self.pool.attention(index, self.config, self.shard), self)
        key, value = self.pool.attention(index, self.config, self.shard)
        return LayerWindow(key, value, self.table, self.slots, self.backed_pages)

    def __call__(self):
        m = self.metadata
        device = m.positions.device
        columns = torch.arange(self.blocks, device=device)
        valid = m.positions >= 0
        positions = m.positions.clamp_min(0)
        if not self.staged:
            base = m.state_rows * self.stride
            self.table.copy_((base[:, None] + columns[None, :].remainder(self.ring)).to(torch.int32))
            page = m.state_rows[m.requests] * self.stride + (positions // PAGE_TOKENS).remainder(self.ring)
            self.slots.copy_(torch.where(valid, page * PAGE_TOKENS + positions.remainder(PAGE_TOKENS), -1))
            return
        live = (m.lengths[0] > 0) & (m.state_rows[0] > 0)
        start, length = m.starts[0], m.lengths[0].long()
        first = (start - (self.window - 1)).clamp_min(0) // PAGE_TOKENS
        last = (start + length - 1).clamp_min(0) // PAGE_TOKENS
        base = m.state_rows[0] * self.stride
        staged = self.key.shape[0]
        self.table.copy_((columns - first).clamp(0, staged - 1).to(torch.int32)[None, :])
        page = positions // PAGE_TOKENS - first
        self.slots.copy_(torch.where(valid, page * PAGE_TOKENS + positions.remainder(PAGE_TOKENS), -1))
        prefix = torch.arange(self.prefix, device=device)
        self.gather_source.copy_(torch.where(live, base + (first + prefix).remainder(self.ring), -1))
        self.gather_destination.copy_(prefix)
        logical = last - (self.ring - 1) + torch.arange(self.ring, device=device)
        self.commit_source.copy_(torch.where(live & (logical >= first), logical - first, -1))
        self.commit_destination.copy_(base + logical.remainder(self.ring))

    def scratch_tensors(self):
        staged = (self.gather_source, self.gather_destination, self.commit_source, self.commit_destination)
        return (self.table, self.slots, *(staged if self.staged else ()))
