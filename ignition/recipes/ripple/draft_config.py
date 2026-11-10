# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned DFlash2 checkpoint and target feature geometry for Ripple-65B Preview."""

from dataclasses import replace

from ignition.components.dflash2_geometry import DFlash2Config

REPOSITORY = "ricursiveAI/Ripple-65B-Preview-DFlash2"
REVISION = "35609b3be3dd08a822b3ba1257c1ea5dd6f87632"
STORED_BYTES = 8_075_504_128
PROFILE = dict(hidden_size=5120, intermediate_size=40960, num_hidden_layers=5,
    num_attention_heads=80, num_key_value_heads=8, head_dim=128, vocab_size=100352,
    num_target_layers=80, rms_norm_eps=1e-6, sliding_window=4096,
    layer_types=["sliding_attention"] * 4 + ["full_attention"], no_rope_layers=[4],
    rope_parameters=dict(rope_type="default", rope_theta=10000.0),
    dflash_config=dict(block_size=16, conv_kernel_size=4, conv_group_size=16, selector_rank=256,
        selector_top_k=32, mask_token_id=100278, target_layer_ids=[1, 20, 39, 58, 77]))
# Proposal temperature per block position: position 0 is the anchor, position d drafts depth d.
# Exact acceptance (1 - TV of target and proposal on identical contexts) over 1,414 rounds of coding-agent
# sessions at request temperature 1 (2026-11-10): the proposal is too sharp at depth 1 and too flat from
# depth 3 on. This schedule gives +0.97% tokens per round (95% CI +0.70..+1.27%) and about +0.4% at 0.7.
# Verification reads the scaled proposal, so sampling stays exact.
PROPOSAL_TEMPERATURES = (1., 1.15, 1., .95, .9, .85, .8, .8, .8, .75, .75, .75, .75, .75, .75, .75)


def load(snapshot, target):
    config = DFlash2Config.load(snapshot, target_hidden=target.hidden, target_vocabulary=target.vocabulary,
        target_layers=target.num_layers, expected=PROFILE)
    return replace(config, proposal_temperatures=PROPOSAL_TEMPERATURES)
