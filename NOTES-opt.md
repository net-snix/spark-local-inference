# Ripple-65B single-card optimized build (agent integrate, 2026-11-07)

One Max-Q TP1 build of `notSnix/Ripple-65B-Preview-MXFP8-GPTQ` (rev bfea8efd) that combines the verified
wins of the kv, dec and prefill agents. Receipts: `/LINUX/LocalStudio/flare-ripple/results-maxq/integrate/`
(R below). Baseline: `gb202-tp1-202611061449` (unchanged `ripple` 12948e93, stock b12x).

## What is in it

**Flare, branch `ripple-opt`** (this worktree, from `ripple` 12948e93; local commits, Snix):

| Commit | From | Change |
|---|---|---|
| 3609797a | kv 9be965ab | Sliding-window KV (64/80 layers) in bounded per-request rings; `-fp8kv` aliases |
| 243e4dd2 | kv 15e0c7fb | Planner sizes the online-MXFP8 drafter at MXFP8 bytes |
| 2c97f491 | kv 586a3919 | Row copies longer than one grid dimension loop over tiles |
| 8e2a5466 | kv 4ba2f17f | docs: rings and opt-in E4M3 KV |
| ec8c717b | dec 94d112cc | Projections read the forward's own temporaries |
| 2a5c09bb, 92a6c43a | dec 109c5a2d, c502c7b0 | Norm restructure (each norm at the end of the previous layer) + fused norm kernel |
| 0138d5c2 | dec 18907c76 | Fused norm unwired: every row count runs the staged chain (Class A) |
| dc845056 | new | Fix kv's row-copy loop for the Triton interpreter (`tl.cdiv` -> constexpr arithmetic); CPU state-transfer test failed on ripple-kv |
| e776c195 | new | Decoder test stand-in gets a ring window (dec test + kv rings) |

Only conflicts: the `decoder.py` imports (window + residual_norm), resolved by hand; the dec delta applied
otherwise verbatim (diff of +/- lines identical to ripple-dec). dec's NOTES.md commit not taken.

**b12x overlay `~/Code/oss/b12x-ripple-opt`, branch `ripple-opt`** (git, from cefb4a2):
2a74b86 (prefill 686f33f, sliced expansion), 382192f (b947603, packed vs expansion race), 9538ff2 (e6292f9,
fused MXFP8 A16 Triton GEMM), 6a81268 (dec 555c1c0, PDL for the packed a16 GEMM + split-K reduce).
Branch `ripple-opt-band` (d937ccf) = experiment that keeps fused out of the 33-127 row race; **not used**.

Dropped: e3c4096 (4096-row chunks; 2 reps, ungated, -6,144 KV), the fused residual norm (not Class A),
the Max-Q retune as a separate step (contract bump re-races every GEMM anyway), FP8 KV as a default.

## Fused a16 at 64-row verify: kept

The prefill agent saw b4w16 +1.7% with fused (spreads overlapping). The race ledger shows fused winning
34-64 rows clearly in isolation (e.g. 64-row qgk 79.6 vs 116.4 us). Decode-only interleaved A/B, fused
everywhere (`opt`) vs the band variant, 6 reps each (R/decode-ab/fused-band/summary.txt): b4w16 59.32
[58.04-60.37] vs 59.44 [58.71-60.27], C4 overall 57.77 vs 57.96, C1 54.36 vs 54.35 ms/round. No difference,
so the verified e3 race stays as is.

## Numerics

- **Prefill gate** (twin `gb202-tp1-202611071421`, 120 x 2,047 positions; R/gate-compare-*.txt):
  vs BF16 coarse KL 0.002174 -> 0.002176, diff +0.000001 CI [-0.000174, +0.000186], top-1 98.68 -> 98.73%,
  "no measurable difference". vs hfsim 0.000820 -> 0.000720, CI [-0.000225, +0.000001], no measurable
  difference. vs the prefill-e3 twin: 348/360 served arrays bit-identical (the rest = CONC=2 co-batching),
  so rings + dec + PDL add nothing on the scoring path. Caveat from the gate verifier: ~0.9% of positions
  hit the tail floor for served packages, and prompts are 2k (single chunk).
- **Decode, `--no-mtp-adaptive-verification`** (R/classa/): the new build is bit-identical to itself
  (r1 vs r2, 6 seeds incl. 4,096-token prompt). Against baseline-noad it is Class B, as expected from the
  b12x GEMM change: logprobs differ from token 0, shared-prefix |d logprob| mean 0.0007-0.0056, max 0.040;
  tokens diverge at 13-52 on 4/6 seeds. Reference spread: the baseline against itself with default
  adaptive verification diverges at tokens 17-76, mean 8e-6-0.0022, max 0.039 (R/drift-baseline-adaptive-r1-vs-r2.json).
- **Unit tests**: 136 passed / 52 skipped CPU (R/tests-cpu.log); 53 passed on the Max-Q with
  FLARE_RUN_CUDA_TESTS=1 (state transfer cpu+cuda, rows, residual norm; R/tests-cuda.log). The 6 Qwen/Kimi
  failures in test_declarations.py predate this branch.
- **Tools / long context** (default flags): tools_probe 15/15; needles 31.7k at 0.1/0.5/0.9 and 261,102
  tokens at 0.25 and 0.75 all correct, 263 s / 260 s per 261k prompt (R/opt-tools.log, R/opt-needle-*.json).

## Speed and KV (gate/ab.sh, base/opt interleaved, 4 reps each; R/ab/opt-vs-base/summary.txt)

| Metric | Baseline mean [min-max] | opt mean [min-max] | opt/base |
|---|---|---|---|
| KV tokens (startup log) | 68,608 | 324,608 | 4.73 |
| Tokens per request | 49,792 | 262,144 | 5.26 |
| C1 GPU ms per round | 56.25 [56.17-56.30] | 54.94 [54.82-55.01] | 0.977 |
| C4 GPU ms per round | 59.06 [58.77-59.46] | 58.89 [58.31-59.31] | 0.997 |
| C1 / C4 rounds/s | 17.75 / 16.91 | 18.18 / 16.97 | 1.024 / 1.004 |
| C1 tok/s (decode probe) | 49.0 [44.2-55.5] | 59.2 [57.9-61.4] | 1.21 (acceptance noise) |
| C4 tok/s (decode probe) | 121.5 [116.9-123.3] | 131.8 [129.0-133.8] | 1.09 (acceptance noise) |
| TTFT 2k / 8k / 16k (s) | 2.354 / 8.954 / 18.06 | 1.347 / 5.448 / 10.85 | 0.57 / 0.61 / 0.60 |
| Prefill tok/s 2k / 8k / 16k | 871 / 915 / 907 | 1,525 / 1,506 / 1,510 | 1.75 / 1.65 / 1.66 |
| inference-bench quick C1 / C4 ctx0 | 51.1 / 148.6 | 48.8 / 160.1 | 0.96 / 1.08 |
| inference-bench C1 / C4 ctx16k | 49.5 / 142.7 | 49.4 / 146.7 | 1.00 / 1.03 |
| speed.py single stream | 66.9 [64.2-70.0] | 66.5 [63.7-68.1] | 0.995 |

Per shape: b1w16 56.05 -> 54.82 (0.978), b2w16 0.998, b4w16 60.47 -> 60.80 (1.005). The C1 gain is
real (ranges do not overlap); C4 is flat. dec's -2.5% at b4w16 does not survive next to e3's GEMMs.

## FP8 KV opt-in alias (`ripple-65b-mxfp8-fp8kv`, never the default; Class C)

Package `gb202-tp1-202611071441`, gate twin `gb202-tp1-202611071606` (R/fp8kv-*, R/gate-fp8kv-compare-*):
675,840 KV tokens (2.08x the BF16 rings), 262,144 per request; gate vs BF16 0.002174 -> 0.004557
(ratio 2.10, CI 1.94-2.26, worse on 120/120), top-1 98.68 -> 98.29%; vs hfsim 0.00082 -> 0.00361; same as
kv's FP8 build (diff -0.0001, CI [-0.0004, +0.0002]). Tools 15/15; needles 31.7k x3 and 261k at 0.25/0.75
correct. One rep: C1 54.5 / C4 58.6 GPU ms per round, TTFT 1.34 / 5.44 / 10.79 s (same as BF16 rings).

## Recipe

```bash
W=/LINUX/LocalStudio/flare-ripple
# under the lock; default build (BF16 KV, rings); add --prompt-logprobs for the gate twin;
# use ripple-65b-mxfp8-fp8kv instead of ripple-65b-mxfp8 for the opt-in FP8 KV alias
flock $W/maxq.lock env FLARE_SRC=$HOME/Code/oss/flare-ripple-opt B12X_SRC=$HOME/Code/oss/b12x-ripple-opt \
  GPUS=$(cat $W/tune-gpu) NAME=flare-ripple-maxq-integrate-export \
  $W/prep-run.sh .venv/bin/ignition export ripple-65b-mxfp8 --gpus 0 --online-mxfp8 \
  --prefill-chunk-size 2048 --compile-workers 6
# serve: write the printed package path into $W/package.path.mxfp8-maxq (empty extra-maxq.args, no
# image-maxq), request model ripple-65b-mxfp8-maxq via :9292; or `flare run <package> --gpus 0`.
```
Scripts: R/scripts/ (export-int.sh, job1/3/4/5.sh, decode-ab.sh, decode_sum.py, drift.py, ledger.py).

| Package | Path (`/home/prep/.cache/flare/packages/models/notSnix_2FRipple-65B-Preview-MXFP8-GPTQ/`) |
|---|---|
| opt (default) | gb202-tp1-202611071333 |
| opt gate twin | gb202-tp1-202611071421 |
| opt FP8 KV | gb202-tp1-202611071441 |
| opt FP8 KV twin | gb202-tp1-202611071606 |
| band experiment | gb202-tp1-202611071345 |

## Risks / open

- TP2 not exported or checked from this branch (rings and dec's `_reduced` path are untested on two ranks).
- Snapshot slots drop ~5x with rings (each prefix snapshot is a 368 MiB row): less prefix-cache reuse.
- Decode is Class B (GEMM lowering), measured only by drift; there is no decode-path KL gate.
- PDL commit is in the build but not measured on its own; fused/expansion win most shapes now.
- The fused-kernel tests live in results-maxq/prefill/, not in the b12x tree (it ships no tests).
