# Ripple-65B notes: branch `ripple-65b-ws-final`

This branch serves
[notSnix/Ripple-65B-Preview-MXFP8-GPTQ](https://huggingface.co/notSnix/Ripple-65B-Preview-MXFP8-GPTQ)
on one NVIDIA RTX PRO 6000 Blackwell card with NVIDIA Flare. It was tuned on the RTX PRO 6000 Blackwell Workstation Edition (450 W).
The branch `ripple-65b-maxq-final` holds the same kind of work tuned on the Max-Q (300 W) card.

- Paired b12x branch: `ripple-65b-ws-final` in [net-snix/flashinfer](https://github.com/net-snix/flashinfer)
  (b12x lives at `flashinfer/experimental/b12x/`).
- Base: NVIDIA Flare at `0009b89` ("Keep streamed responses alive when clients pause reading
  (#207)"). The 37 commits after it are by Snix.
- Main changes: the Ripple recipe and single-card MXFP8-GPTQ serving, sliding-window KV rings, fused decode glue kernels, programmatic launches, `--prefix-checkpoint-spacing` for agent turns, a prefill launch-plan rule in Helios, DFlash2 proposal temperatures per draft depth, and finer prefill graph capacities.

## Modified files (Apache-2.0, section 4(b))

Snix changed or added these files in November 2026. The git history shows every change.

- added `NOTES-decode2.md`
- added `NOTES-draft.md`
- added `NOTES-opt.md`
- changed `docs/api.md`
- changed `docs/model-preparation.md`
- changed `flare/src/cli.rs`
- changed `flare/src/main.rs`
- changed `helios/model-contract.md`
- changed `helios/proto/runtime_package.proto`
- changed `helios/src/model/contract.rs`
- changed `helios/src/output.rs`
- added `helios/src/output/hermes_json.rs`
- changed `helios/src/output/pipeline.rs`
- changed `helios/src/output/tagged.rs`
- changed `helios/src/runtime_package/protobuf.rs`
- changed `helios/src/scheduler.rs`
- changed `helios/src/scheduler/policy.rs`
- changed `helios/src/scheduler/policy_tests.rs`
- changed `helios/src/scheduler/utility.rs`
- changed `helios/src/server/config.rs`
- changed `ignition/components/candidate_selector.py`
- changed `ignition/components/capacity.py`
- changed `ignition/components/dflash2.py`
- added `ignition/components/dflash2_full_attention.py`
- changed `ignition/components/dflash2_geometry.py`
- changed `ignition/components/dflash2_preparation.py`
- changed `ignition/components/state_transfer.py`
- changed `ignition/planning.py`
- changed `ignition/preparation/runtime_package_pb2.py`
- changed `ignition/recipes/__init__.py`
- changed `ignition/recipes/declaration.py`
- added `ignition/recipes/ripple/__init__.py`
- added `ignition/recipes/ripple/assets/chat_template.patch.json`
- added `ignition/recipes/ripple/bootstrap.py`
- added `ignition/recipes/ripple/config.py`
- added `ignition/recipes/ripple/contract.py`
- added `ignition/recipes/ripple/decoder.py`
- added `ignition/recipes/ripple/dflash.py`
- added `ignition/recipes/ripple/draft_config.py`
- added `ignition/recipes/ripple/glue.py`
- added `ignition/recipes/ripple/pool.py`
- added `ignition/recipes/ripple/recipe.py`
- added `ignition/recipes/ripple/residual_norm.py`
- added `ignition/recipes/ripple/weights.py`
- added `ignition/recipes/ripple/window.py`
- changed `pyproject.toml`
- changed `tests/cli/test_draft_checkpoints.py`
- changed `tests/cli/test_planning.py`
- added `tests/components/test_b12x_mxfp8_fused_decode_cuda.py`
- added `tests/components/test_b12x_mxfp8_fused_pdl_cuda.py`
- added `tests/components/test_b12x_mxfp8_fused_tiles_cuda.py`
- added `tests/components/test_b12x_selection_pins.py`
- added `tests/components/test_candidate_selector.py`
- changed `tests/components/test_candidate_selector_cuda.py`
- changed `tests/components/test_dflash2.py`
- added `tests/components/test_dflash2_full_attention.py`
- added `tests/components/test_state_transfer_rows_cuda.py`
- added `tests/frontend/test_ripple.py`
- changed `tests/preparation/test_frontend_metadata.py`
- changed `tests/preparation/test_native_recipe_contract.py`
- added `tests/recipes/test_ripple.py`
- added `tests/recipes/test_ripple_decoder.py`
- added `tests/recipes/test_ripple_glue_cuda.py`
- added `tests/recipes/test_ripple_glue_pdl_cuda.py`
- added `tests/recipes/test_ripple_residual_norm_cuda.py`
- changed `tests/test_prefill_capacities.py`

## Credits

- The NVIDIA Flare authors and contributors ([NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference)).
- Local Inference Lab for b12x and the FlashInfer fork Flare builds on
  ([local-inference-lab/flashinfer](https://github.com/local-inference-lab/flashinfer)).
- ricursiveAI for Ripple-65B-Preview and its DFlash2 draft model.

## License

Apache License 2.0, the same as upstream. `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` are
unchanged.
