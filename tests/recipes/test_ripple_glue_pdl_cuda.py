# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""glue.py's programmatic launches (PDL + pre-wait L2 warming) keep every output bit.

Each kernel is compared against its non-PDL launch, eagerly, inside a CUDA graph whose producer releases
the glue kernel at its start and only then writes the glue's inputs (a read before griddepcontrol.wait would
see stale rows), and through torch.compile (the launches stay opaque custom ops, so Inductor neither
clones their arguments nor drops the programmatic launch).
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="set FLARE_RUN_CUDA_TESTS=1 to run CUDA kernels")

triton = pytest.importorskip("triton")
tl = triton.language

HIDDEN, EPS, HEADS, KV_HEADS, DIM, ROWS = 5120, 1e-6, 80, 8, 128, 6


@triton.jit
def _late_copy(X, Y, NUMEL: tl.constexpr, BLOCK: tl.constexpr, NS: tl.constexpr):
    """Release the next programmatic launch at once, wait for this grid's producer, then copy X to Y late."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t0 = tl.extra.cuda.globaltimer()
    t = t0
    while t - t0 < NS:
        t = tl.extra.cuda.globaltimer()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < NUMEL), i < NUMEL)


def _inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = lambda width, scale: (torch.randn(ROWS, width, device="cuda", generator=g) * scale).bfloat16()  # noqa: E731
    gain = lambda width: (torch.rand(width, device="cuda", generator=g) * 3 - 0.5).bfloat16()  # noqa: E731
    angle = torch.rand(ROWS, DIM, device="cuda", generator=g) * 6.28
    return dict(hidden=rows(HIDDEN, 4.0), delta=rows(HIDDEN, 1.0), start=rows(HIDDEN, 1.0), post=gain(HIDDEN),
                weight=gain(HIDDEN), blend=torch.tensor(0.3, device="cuda").bfloat16(),
                qgk=rows(2 * HEADS * DIM + KV_HEADS * DIM, 2.0), v=rows(KV_HEADS * DIM, 1.0), q_norm=gain(DIM),
                k_norm=gain(DIM), cos=angle.cos().bfloat16().float(), sin=angle.sin().bfloat16().float(),
                attended=rows(HEADS * DIM, 1.0))


def _forward(x, before=lambda names: None):
    """Every glue kernel once; `before(names)` runs ahead of each kernel with the inputs it reads."""
    from ignition.recipes.ripple import glue

    q = torch.empty(ROWS, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    key = torch.empty(ROWS, KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    value = torch.empty_like(key)
    outs = []
    for rope in (True, False):
        before(("qgk", "v"))
        glue.qk_rope(x["qgk"], x["v"], x["q_norm"], x["k_norm"], x["cos"], x["sin"], q, key, value, EPS, rope=rope)
        outs += [q.clone(), key.clone(), value.clone()]
    before(("attended", "qgk"))
    outs.append(glue.gate(x["attended"], x["qgk"], HEADS * DIM))
    for arguments, keywords in (((x["delta"], x["hidden"], x["post"], x["weight"], EPS), {}),
                                ((x["delta"], x["hidden"], x["post"], x["weight"], EPS),
                                 dict(start=x["start"], blend=x["blend"])),
                                ((None, x["hidden"], None, x["weight"], EPS),
                                 dict(start=x["start"], blend=x["blend"]))):
        before(("hidden", "delta", "start"))
        outs += [t.clone() for t in glue.residual_norm(*arguments, **keywords) if t is not None]
    return outs


def _equal(got, want):
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def _reference(x, monkeypatch):
    from ignition.recipes.ripple import glue

    with monkeypatch.context() as m:
        m.setattr(glue, "_pdl", lambda tensor, rows: False)
        return _forward(x)


def test_glue_launches_programmatically():
    from ignition.recipes.ripple import glue

    probe = torch.empty(1, device="cuda")
    assert glue._pdl(probe, ROWS) and glue._pdl(probe, 64)
    assert not glue._pdl(probe, 128)  # prefill-size launches stay plain


def test_pdl_glue_is_bitwise_equal(monkeypatch):
    x = _inputs(3)
    _equal(_forward(x), _reference(x, monkeypatch))


def test_pdl_glue_graph_waits_for_a_late_producer(monkeypatch):
    x = _inputs(5)
    want = _reference(x, monkeypatch)
    staged = {k: v.clone() for k, v in x.items()}
    block = 1024

    def late(names):
        # Zero the inputs, then copy each back late; every copy releases the next launch at its start, so the
        # glue kernel starts while the copies (all but the last one transitively) still run.
        for name in names:
            staged[name].zero_()
        for name in names:
            _late_copy[(triton.cdiv(x[name].numel(), block),)](x[name], staged[name], x[name].numel(), block, 10000,
                                                               launch_pdl=True)

    def chain():
        return _forward(staged, late)

    chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain()
    for _ in range(3):
        for t in outs:
            t.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _equal(outs, want)


def test_pdl_glue_survives_inductor(monkeypatch):
    from torch._inductor.utils import run_and_get_code

    x = _inputs(7)
    want = _reference(x, monkeypatch)
    torch._dynamo.reset()
    got, code = run_and_get_code(torch.compile(_forward, fullgraph=True), x)
    _equal(got, want)
    text = "\n".join(code)
    assert all(f"flare.ripple_{name}" in text for name in ("residual_norm", "qk_rope", "gate"))
