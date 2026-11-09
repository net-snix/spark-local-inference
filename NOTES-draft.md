# DFlash2 drafter notes (agent draft, 2026-11-07)

Goal: more tokens per verify round for single-card Ripple-65B MXFP8-GPTQ on the Max-Q, without
touching the target distribution. Base package: gb202-tp1-202611071333.

## Audit (drafter vs checkpoint and NeMo AutoModel reference)

Matches the reference: anchor + 15 mask tokens, 5 layers (4 sliding, window 4096, layer 4 NoPE
full attention), grouped conv (kernel 4, group 16) around each sublayer, taps -> fc -> hidden_norm
-> per-layer k/v + k_norm + rope (none on the NoPE layer), candidate selector (top-32, rank 256,
S_t(a,b) = U_t(b) + <A(a) H(h_t), B(b)>, walked from the anchor, softmax(S/T) sampling).
Verification is FlashInfer chain_speculative_sampling (exact rejection sampling), and the
adaptive verifier only picks the verified width, so the target distribution cannot move.

Gaps that cost acceptance (none is a target-correctness bug):
1. Full-attention reach. Layer 4 read the same 4096-token FP8 ring as the sliding layers, so
   past 4k it saw only the recent window. The checkpoint trains it as whole-context attention.
2. Draft context KV is stored as FP8 E4M3 rings (target pages are FP8 too; the reference is BF16).
3. --online-mxfp8 quantizes the drafter's linear weights to MXFP8 (the checkpoint is BF16).
4. Harness: older acceptance numbers were taken at T=0.7; this suite runs T=1.0 (the lossless
   and acceptance-relevant setting for chat sampling).

## What was built

Branch ripple-draft (~/Code/oss/flare-ripple-draft), on top of c9e2760b (flare-ripple-opt):
- 0b1678df feat(dflash2): split-KV block attention over a request's whole paged context.
  ignition/components/dflash2_full_attention.py + tests/components/test_dflash2_full_attention.py.
  Triton split-KV flash decoding: grid (requests, kv_heads x ceil(10/4) head chunks, 16+1 splits);
  splits 0..15 share the context pages, split 16 reads the block K/V from scratch; FP32 online
  softmax, BF16 dots; LSE merge kernel. BF16 or E4M3 pages. Graph-capturable (wrap_triton).
- 1d4f83c2 feat(ripple): DFlash2 full-attention draft layer reads the whole context.
  Layer 4's context K/V moves into 2 extra fields of each page record (32 -> 34 KV records,
  about 6% fewer KV tokens), written at the target's page slots for every accepted row; the
  layer attends through the page tables. The sliding layers keep their rings. Compact prefill
  capture is off when a layer is paged. Qwen3.8 recipes keep the ring path.

Branch ripple-draft-rings (~/Code/oss/flare-ripple-draft2):
- d0d6f05d feat(ripple): DFlash2 sliding layers read BF16 rings in place (ring mode of the same
  kernel, no staging copy). Not measured (no lock time before the 20:30 chain). Known failing test:
  test_online_mxfp8_plans_the_drafter_at_mxfp8_size (BF16 rings cost 32 MiB per state row, so the
  plan drops to 261,248 tokens per request). Fix or revert before use.

## Kernel evidence

Numerics: draft side only (class C for the drafter's proposals: it attends to more keys), target
distribution unchanged by construction (exact rejection sampling).
- CPU interpreter (FP32) and CUDA tests vs an FP32 eager reference: 5 passed (k1), 7 passed with
  ring mode (k2). logs/gputest-k1.log, logs/gputest-k2.log.
- Microbench (bench-full-k1.json): BF16 pages 1 req 33 us @4k, 204 us @32k, 1.64 ms @262k;
  4 req 766 us @32k, 6.19 ms @262k (~11% of a 55 ms round at the worst case). FP8 pages run at
  230-280 GB/s (dequant bound), BF16 at 650-730 GB/s. Max error <= 0.008.

## Results

Receipts: /LINUX/LocalStudio/flare-ripple/results-maxq/draft/ (ab1/1-base, ab4/{1-full,2-bf16d,3-base},
summary-base-full.txt, summary-base-bf16d.txt, logs/, tools/). Suite (tools/arm.sh): T=1, 30 s per C1 kind,
C4 code/prose, long-context lc-code/lc-prose at 2k-32k (2 requests x 1024 tokens each), greedy C1 code/json
(added after hold 1, so only the last base arm has it). Arms ran base | full bf16d base (one lock hold).
Lock contention (6-8 waiters, 21 min exports) left time for 2 base arms and 1 arm each of full and bf16d,
so per-workload C1 numbers carry run-to-run noise of about +-10% (base vs base: c1-json 4.45 vs 5.26).

Tokens per round, base (2 runs, mean [min-max]) vs full (1 run):

| context | lc-prose base | lc-prose full | ratio | lc-code base | lc-code full |
|---|---|---|---|---|---|
| 2k | 4.86 [4.73-4.99] | 5.19 | 1.07 | 4.21 [4.15-4.26] | 5.50 |
| 4k | 5.19 [4.41-5.98] | 4.67 | 0.90 | 4.25 [4.12-4.37] | 4.33 |
| 8k | 2.84 [2.82-2.87] | 5.68 | 2.00 | 4.51 [4.10-4.91] | 4.37 |
| 16k | 3.42 [3.34-3.50] | 5.11 | 1.49 | 3.83 [3.56-4.10] | 3.86 |
| 32k | 3.30 [3.28-3.33] | 4.31 | 1.31 | 4.71 [4.42-4.99] | 4.04 |

- Base acceptance on prose falls from about 4.9-5.2 tokens/round below 4k to 2.8-3.4 past 8k, with tight
  base-vs-base agreement there; the full-context layer recovers most of it (+31% to +100% at 8k-32k).
  Code does not move: its useful context is local and already inside the 4k window.
- GPU ms per round is unchanged at long context (full 55.8-58.0 ms vs base 55.3-58.8 ms at 2k-32k), as
  the microbench predicts (0.2 ms at 32k). So tok/s at C1 follows tokens/round: lc-prose 16k about
  89 vs 60 tok/s, 8k about 100 vs 51 tok/s.
- Short context (<4k, C1 and C4): full is within noise of base. Mean tok/s ratios: code 0.94, json 0.91,
  math 0.93, norsk 1.01, prose 1.03, C4 code 1.02, C4 prose 0.97, from one full run against two base
  runs whose own spread is up to 17%. Code and math sit a little below base's range; this needs the
  greedy check (c1-*-t0) on both arms before it is called noise.
- KV capacity (startup log): base 324,608 tokens, full 306,176 (-5.7%), both 262,144 per request.

BF16 drafter weights (export without --online-mxfp8, bf16d, built from flare-ripple-opt c9e2760b):
tokens/round +3% to +9% on C1/C4 short context (c1-code 1.01, json 1.03, norsk 1.09, prose 1.05, c4-code
1.07), but GPU ms per round +4-7% (57-58 vs 54-55 ms; 67 vs 63 ms at C4) and KV capacity drops to 271,360
tokens with 226,432 per request (below the 262,144 contract). Net tok/s 0.90-1.05. Not adopted.

Lossless: NOT RE-RUN. The check (tools/lossless.sh full <pkg>; ON seeds 1.., ON seeds 10001.., OFF
reused from results-maxq/dflash-lossless/off.jsonl, copied into lossless-full/) was queued as hold 2 of
tools/job4.sh but never got maxq.lock before the 20:30 chain. Run it before adopting the build: the
draft change cannot move the target distribution (verification is unchanged exact rejection sampling),
but the page record layout changed (2 extra fields per record), so target KV integrity needs the
empirical check. The long-context acceptance gains are indirect evidence that target KV is intact.

## Recipe

Build the full-context package (Max-Q, single card), from the gate helpers:

    FLARE_SRC=~/Code/oss/flare-ripple-draft B12X_SRC=~/Code/oss/b12x-ripple-draft \
      /LINUX/LocalStudio/flare-ripple/results-maxq/draft/tools/export.sh full --online-mxfp8
    # = ignition export ripple-65b-mxfp8 --gpus 0 --prefill-chunk-size 2048 --compile-workers 6 --online-mxfp8
    # result: gb202-tp1-202611071832 (flare 1d4f83c2, b12x 6a812683)

Serve it through the lane as usual (no extra args). Measure: tools/arm.sh <outdir> <label> <pkg>; summarize
with tools/draftsum.py DIR... --arms base full. Kernel tests: FLARE_RUN_CUDA_TESTS=1 pytest
tests/components/test_dflash2_full_attention.py (CUDA_VISIBLE_DEVICES = the Max-Q only).

Next: lossless check, more full/base reps (tools/job4.sh holds 2-4; delete its hold 1 line first, or it rewrites ab4/1-3), greedy check on full, block 16 vs 8-12 not
tried, BF16 rings branch (draft2) unmeasured.
