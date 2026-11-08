# decode2: verify-round GPU time on the Max-Q (Ripple-65B MXFP8-GPTQ, DFlash2)

Agent decode2, 2026-11-07. Baseline package `gb202-tp1-202611071333` (C1 54.94 ms, C4 58.89 ms per round).
Receipts: `/LINUX/LocalStudio/flare-ripple/results-maxq/decode2/`. Nothing pushed or uploaded.

## Branches

- b12x `~/Code/oss/b12x-ripple-decode2`, branch `ripple-decode2` from `6a81268`
  - `7e0e58a` feat(preparation): replay recorded selections from a pin file (`preparation/_pins.py`, hook in
    `gemm/blockscaled/_preparation.py::_plan_bf16`)
  - `3e3bde5` perf(blockscaled): decode fused MXFP8 A16 weights with F2FP and BF16 multiply
    (`gemm/blockscaled/_fused.py::_decode_tile`)
- Flare `~/Code/oss/flare-ripple-decode2`, branch `ripple-decode2` from `c9e2760b`: tests only
  (`tests/components/test_b12x_selection_pins.py`, `tests/components/test_b12x_mxfp8_fused_decode_cuda.py`) and
  this file.

## Profile of 071333 (nsys node trace, one verify round)

`prof/opt-c{1,4}-node.cats.txt`, `.seq.txt`, `.smi.csv`. Analysis tools `tools/round_cats2.py`, `round_seq2.py`.

| category | C1 b1w16 us | C4 (p13) us |
|---|---:|---:|
| fused a16 GEMM (gate_up, down) | 35128 | 39200 |
| packed a16 GEMM (qgk, v, o) | 9641 | 6899 |
| draft, sampling, bookkeeping | 4833 | 5695 |
| a16 split-K reduce (mostly PDL overlap) | 2375 | 476 |
| ATen mean (`reduce_kernel`, grid 1-2) | 1718 | 2330 |
| Inductor Triton | 1212 | 1311 |
| attention + KV write | 983 | 1220 |
| lm_head (nvjet) | 707 | 832 |
| gaps (all) | 2453 | 2461 |
| span | 56891 | 60458 |

Findings:
- 85-90% of the round is weight streaming in the a16 GEMMs. Gaps total 2.5 ms, so a persistent megakernel
  is not justified (best case -4%, high risk).
- The card sits on the 300 W software power cap (`clocks_event_reasons` 0x4) during decode at C1 and C4;
  SM clock drops to 1845-1900 MHz, memory stays at 16365 MHz. In-situ GEMMs run ~25% slower than their isolated
  export races (gate_up 287-296 us vs 236 us, down 151-156 vs 121.6). Decode is energy bound, so cutting ALU
  work per weight byte is the lever, not tiling.
- The packed CuTe a16 kernel (qgk/v/o) already decodes with `cvt.rn.bf16x2.e4m3x2` + `cvt.rn.bf16x2.ue8m0x2` +
  `mul.bf16x2` (`_lib/intrinsics/packed.py::mxfp8_pair_to_bf16x2_sm120`). The fused Triton kernel (gate_up,
  down: 62-67% of the round) used FP32: e4m3x2->f16x2, 2x f16->f32, ex2.approx scale, f32 mul, f32->bf16x2.

## Profile of e1 on the WS (2026-11-08, tune GPU moved to the WS: 2750 MHz lock, 450 W)

`results-ws/decode2/prof/e1ws-c{1,4}-node.cats.txt`, `.seq.txt` (default flags: C1 ran bucket p6, C4 p13).

| category | C1 us | C4 us |
|---|---:|---:|
| fused a16 GEMM (gate_up, down) | 27113 | 29808 |
| packed a16 GEMM (qgk, v, o) | 7718 | 5349 |
| draft, sampling, bookkeeping | 4038 | 4356 |
| a16 split-K reduce (PDL, overlaps the GEMM) | 2332 | 344 |
| ATen mean (`reduce_kernel`) | 1083 | 1599 |
| Inductor Triton | 904 | 979 |
| attention + KV write | 694 | 889 |
| lm_head | 534 | 557 |
| gaps (all) | 2116 | 2065 |
| span | 44319 | 45980 |

One C1 layer: gate_up 226 us, down 113, qgk 62 + v 4.7, o_proj 30, attention 7.7, and 25 small kernels
(~33 us with their 0.5 us gaps, 4 of them ATen means at 2.7 us). The WS is not power bound (406-430 W).
Isolated DRAM-streaming bench at 16 rows on the WS (`results-ws/decode2/bench/a16-r16.log`): served
gate_up 225.6 us vs 207.2 us read floor, down 117.7 vs 104.3, qgk 62.6 vs 55.4, o 30.6 vs 27.0. The served
tiles are already the fastest class-A choice at 16 rows (every other tile/warp/persistent variant with the same
split-K is equal or slower), so a pin retune buys nothing at C1. At 64 rows (Max-Q, `bench/a16-r64.log`)
gate_up m64n128k64w4 is 13% faster than the served tile with the same bits, which is what the sweep is for.

## Kernel 1: fused MXFP8 A16 decode on F2FP + BF16 multiply (class A)

`_decode_tile` builds the UE8M0 scale as a BF16 bit pattern (`e << 7`) and decodes four codes per inline-asm
call: 2x `cvt.rn.bf16x2.e4m3x2` + 2x `mul.rn.bf16x2`. K-loop decode per pair goes from 5 instructions to 2
(PTX: 64 cvt + 64 mul vs 64 cvt.e4m3x2 + 128 cvt.f32.f16 + 64 mul.f32x2 + 64 cvt.bf16x2 per tile; `ptx/`).
Shared memory and tiles unchanged (41984 B).

Why it is exact: E4M3 values are exact in BF16; the scale is a power of two, so the product is exact unless it
underflows, and both paths then round the exact product once to BF16 (the FP32 path's product is exact in FP32
down to 2^-135). e = 0: FP32 path flushes `ex2.approx.ftz(-127)` to 0, the new scale bit pattern is 0. Same.

Evidence:
- GPU decode table, all 256 codes x 256 exponents: 65536 of 65536 bitwise equal (`bench/fused-r16.log`,
  `bench/fused-old-vs-new-r16.log`).
- Isolated speed, 16 rows, served tiles, 12 distinct layers back to back in a CUDA graph, old source
  (b12x `7e0e58a`) vs the new decode, interleaved, 3 reps (`bench/fused-old-vs-new-r16.log`):

  | op | old us | new us | DRAM-read floor us | outputs |
  |---|---:|---:|---:|---|
  | gate_up 81920x5120 | 307.2 [302.0-311.6] | 296.8 [292.5-300.4] (-3.4%) | 237.1 | bitwise equal |
  | down 5120x40960 split-K 8 | 160.4 [160.3-160.5] | 154.1 [154.0-154.2] (-3.9%) | 119.9 | bitwise equal |

  Every variant, including a pure read of the same bytes, runs at 300 W; the SM clock rises 1747 -> 1800 MHz
  with the cheaper decode. Expected per round: 80 x (10.4 + 6.3) us ~= 1.3 ms at C1. stages=4 does nothing.
- Export e1 = `gb202-tp1-202611072020` (`logs/e1.log`, `e1.src`, `e1.path`), built with the pins: zero b12x
  races of any component during the export (`tools/races_since.py "2026-11-07 20:12" pins-071333.json`), so
  every GEMM keeps 071333's tiles and split-K.
- NOT RUN (lock went to the tonight-chain): `FLARE_RUN_CUDA_TESTS=1` tests, in-model gate
  (`gate/classa.sh` vs `integrate/classa/opt-noad-r1.json`) and the decode probe. Class A in the served model is
  therefore argued (exact decode on every input, same tiles/order, same MMA inputs) but not yet proven.

## Selection pins (keeps a kernel change class A)

`B12X_SELECTION_PINS_STRICT=1` is too strict as written: it raises for every unpinned query of a pinned
component, including queries that never race (the 2048-token prefill shapes: `logs/e1-strict-fail.log`).
The working check is non-strict export plus `tools/races_since.py` showing zero blockscaled races.

Any b12x source change alters `candidate_program_identity`, so the next export re-races every query and timing
noise picks other split-K/tiles (60 of 172 blockscaled choices differed between two exports of the same
source), which changes rounding. `B12X_SELECTION_PINS=<json>` replays recorded winners as plan overrides;
`B12X_SELECTION_PINS_STRICT=1` raises on an unpinned query, which also hits queries that never race, so
it is a diagnostic only and not part of the recipe. Pins for 071333: `pins-071333.json` (172 pins from
ledger session f32b167fcd2f, round-trip checked by `tools/pins_check.py`). Only the blockscaled component imports
`_pins`, so other components keep their identity and cached selections.

## Results vs 071333

Class A (2026-11-07 23:50 Bergen, on the WS, `results-ws/decode2/`): e1 (`gb202-tp1-202611072020`) vs 071333,
both `--no-mtp-adaptive-verification`, 6 seeds, reference collected before and after: BIT-IDENTICAL (tokens,
chosen logprobs, top-20). Loaded 72.77 vs 72.80 GiB, KV 325,632 vs 324,608: the 32 MiB is the e1 build's
smaller footprint, which buys 1,024 KV tokens.

Not measured: the interleaved A/B (`tools/job3.sh`) never got the lock before the tonight-chain. Expected from
the isolated bench: about -1.3 ms per round at C1 (2.4%), a little more at C4 (more fused rows of work).

2026-11-08 (WS): the A/B is still not measured. The BF16 eval block (`runner/bf16_block.sh`, both cards, both
locks) has held the GPUs since 14:09 Bergen (gsm8k, ifbench done at 15:58; mmlupro, gpqa, aime2025/2026 left).
`tools/job-g.sh` is queued in tmux `d2-jobG` (log `results-ws/decode2/job-g.log`) and runs, each step under its
own short lock hold: (1) `gate/ab.sh base=071333 e1=e1 4` with `AB_LOCK_PER=2` ->
`results-ws/gate/ab/d2-e1-vs-1333/`; (2) the CUDA tests including the new tile-invariance test plus the ATen
alignment probe (`tests-cuda2.log`, `bench/aten-align.log`); (3) a 071333 WS profile (`prof/base1333ws-*`);
(4) the tile sweep (`bench/sweep.jsonl`, served buckets first); (5) `make_pins2.py` -> `pins-e2.json` and,
if any pin changes, the e2 export (`e2.path`, `logs/e2.races.txt`). e2 still needs `gate/classa.sh` and its
own A/B. Stop it with `tmux kill-session -t d2-jobG` (it holds no lock while waiting).

Expected value of e2: small at the served buckets. The WS profile ran C1 at p6 and C4 at p13 for ~95% of
rounds, and at 16 rows the served tiles are already the fastest class-A choice (bench above). The 13% gate_up
gain is at p64, which adaptive verification picks only on high-acceptance rounds.

## Recipe

```
# export (holds maxq.lock; tools/export.sh). Non-strict pins: strict mode raises on queries that never race.
env B12X_SELECTION_PINS=/work/results-maxq/decode2/pins-071333.json \
  .venv/bin/ignition export ripple-65b-mxfp8 --gpus 0 --online-mxfp8 --prefill-chunk-size 2048 --compile-workers 6
# with FLARE_SRC=~/Code/oss/flare-ripple-decode2 B12X_SRC=~/Code/oss/b12x-ripple-decode2 (prep-run.sh overlay)
# check the replay: zero blockscaled races since the export started, champions equal to the pins
python3 results-maxq/decode2/tools/races_since.py "<export start, UTC>" results-maxq/decode2/pins-071333.json
bash gate/classa.sh <pkg> d2-e1-noad --no-mtp-adaptive-verification
AB_LOCK_PER=2 AB_TAG=d2-e1-vs-1333 bash gate/ab.sh "base=<071333>" "e1=<pkg>" 4
```

## Tile invariance (why a pin retune stays class A)

`tests/components/test_b12x_mxfp8_fused_tiles_cuda.py`: the fused kernel's outputs are bitwise equal across
tile_m/tile_n/tile_k, warps and stages for six Ripple shapes (16-64 rows, split-K 1/4/8), and differ when only
split-K changes (negative control). The bench sweep checks the same thing per candidate on the real shapes
(`bitwise True` column). The CUDA run is queued in job G (step 2); not yet run on a GPU.
`tools/make_pins2.py` only switches a pin to a candidate that was bitwise equal, has the same split-K and
backend, a b12x-legal tile_k (64/128) and is faster by >1.5% with non-overlapping reps.

## Not done / next

- Fused q/k norm + RoPE + KV write, single-kernel proposal positions, grouped v_proj with qgk: not built. The
  profile puts all of them inside 1.2 ms Inductor + 1.2 ms attention/KV + 4.8-5.7 ms draft; each is <2%.
- ATen mean replacement (1.7-2.3 ms, grid 1-2 `reduce_kernel`): needs exact ATen order (dec's model: BW =
  512/min(pow2(rows),16), float4 accumulators, halving tree); dec's fused residual norm was not class A
  in-graph for an unknown reason. Worth an in-graph check variant first.
- Split-K reduce fused into the next op: only 0.5 ms visible at C4 (PDL already overlaps it).
- lm_head: `tools/bench_head.py` checks whether a K-sequential Triton head is bit-identical to nvjet (not run).
- Remaining fused-GEMM headroom: a pure read of the same bytes is 237/120 us vs 297/154 us, also at 300 W.
  Closing that needs a kernel that skips the shared-memory round trip for M <= 16 (CuTe, register-direct
  weights), not more Triton tuning.
- dec's fused norm: the kernel takes ATen's block width from the traced row count. If the staged
  `flare::mean` sees a different runtime row count (live rows) or a non-16-byte-aligned view, ATen picks
  another order. Check those two first.
- Small-op chain on the WS: ~25 kernels per layer outside the GEMMs and attention, ~33 us with launch gaps
  (~2.6 ms per C1 round, 6%). Every chain contains an ATen mean (`flare::mean`, kept for its rounding) or the
  ATen sigmoid (`flare::sigmoid`), so a fused kernel is class A only if it reproduces ATen's order in-graph,
  which is the open question from dec. `tools/aten_mean_align.py` (queued) tests one candidate cause: ATen's
  `input_vectorized_thread_reduce_impl` handles a row head that is not 16-byte aligned separately, so a
  misaligned square buffer would take another order. If alignment does not change the result, the next step
  stays dec's in-graph check variant (compute both, record the first mismatching layer/row).
