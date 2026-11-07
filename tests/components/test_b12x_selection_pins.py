# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""b12x selection pins replay a recorded blockscaled GEMM choice as the plan's override."""

import json

import pytest

pins = pytest.importorskip("flashinfer.experimental.b12x.preparation._pins")
blockscaled = pytest.importorskip("flashinfer.experimental.b12x.gemm.blockscaled._preparation")
from flashinfer.experimental.b12x.gemm.blockscaled._tuning import TUNING  # noqa: E402

QUERY = {"activation_mode": "a16", "activation_scale_available": False, "allow_bf16_atomic_reduction": False,
         "autotune_effort": "default", "codegen": {"fused_fp6_quant": False, "split_k_atomic": True},
         "expected_m": None, "global_scale_kind": "none", "in_features": 5120, "num_tokens": 16,
         "out_features": 81920, "output_mode": "provided", "padded_in_features": 5120, "recipe": "mxfp8",
         "source_activation": "identity", "source_aligned": True, "source_contiguous": True,
         "workspace_form": "provided", "workspace_nbytes": None}
FUSED = {"a16_backend": "fused", "dense_config": None, "fuse_swiglu": False, "mma_warps": 4, "mode": "a16",
         "split_k": 1, "tile_k": 128, "tile_m": 16, "tile_n": 128}


def _write(tmp_path, entries):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"version": 1, "pins": entries}))
    return str(path)


def _query(payload=QUERY):
    return blockscaled._decode_query(payload)


def test_pinned_query_plans_with_the_recorded_config(tmp_path, monkeypatch):
    path = _write(tmp_path, [{"component": TUNING.component_id, "query": QUERY, "assignment": FUSED}])
    monkeypatch.setenv(pins.PIN_ENV, path)
    plan = blockscaled.plan(_query())
    assert dict(TUNING.encode_config(plan.override)) == FUSED


def test_unpinned_query_races_unless_strict(tmp_path, monkeypatch):
    path = _write(tmp_path, [{"component": TUNING.component_id, "query": QUERY, "assignment": FUSED}])
    monkeypatch.setenv(pins.PIN_ENV, path)
    other = _query(dict(QUERY, num_tokens=32))
    assert blockscaled.plan(other).override is None
    monkeypatch.setenv(pins.STRICT_ENV, "1")
    with pytest.raises(ValueError, match="no pin"):
        blockscaled.plan(other)


def test_without_a_pin_file_nothing_changes(monkeypatch):
    monkeypatch.delenv(pins.PIN_ENV, raising=False)
    monkeypatch.setenv(pins.STRICT_ENV, "1")
    assert blockscaled.plan(_query()).override is None


def test_conflicting_pins_are_rejected(tmp_path):
    other = dict(FUSED, split_k=2)
    path = _write(tmp_path, [{"component": TUNING.component_id, "query": QUERY, "assignment": FUSED},
                             {"component": TUNING.component_id, "query": QUERY, "assignment": other}])
    with pytest.raises(ValueError, match="conflicting"):
        pins.load_pins(path)


def test_pin_files_need_version_one(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"pins": []}))
    with pytest.raises(ValueError, match="version 1"):
        pins.load_pins(str(path))
