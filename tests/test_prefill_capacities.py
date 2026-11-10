# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prefill graph capacities shared by recipes and Helios."""

from ignition.components.capacity import prefill_capacities


def test_prefill_capacities_step_after_doubling_and_end_at_the_chunk():
    assert prefill_capacities(64) == (64,)
    assert prefill_capacities(1000) == (128, 256, 512, 1000)
    assert prefill_capacities(4096) == (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
    assert prefill_capacities(8192) == (128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)


def test_midpoints_fill_every_doubling_gap():
    assert prefill_capacities(64, midpoints=True) == (64,)
    assert prefill_capacities(128, midpoints=True) == (128,)
    assert prefill_capacities(256, midpoints=True) == (128, 192, 256)
    assert prefill_capacities(1000, midpoints=True) == (128, 192, 256, 384, 512, 1000)
    assert prefill_capacities(2048, midpoints=True) == (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
    assert prefill_capacities(8192, midpoints=True) == (
        128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
    for chunk in (64, 128, 192, 256, 1024, 2048, 4096, 8192):
        capacities = prefill_capacities(chunk, midpoints=True)
        assert set(prefill_capacities(chunk)) <= set(capacities) and capacities[-1] == chunk
        assert all(capacity % 64 == 0 for capacity in capacities) and list(capacities) == sorted(set(capacities))
