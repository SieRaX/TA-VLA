#!/usr/bin/env bash
# TA-VLA pi0 *full* finetune on Franka FR3 Peg_in_hole with torque (DePost + future-effort).
# Variant: EffortType.EXPERT_IN_FUT, effort_dim=7, current-frame effort input.
#
# Differences from the LoRA twin (train_fr3_peg_in_hole_effort.sh):
#   - model.paligemma_variant / action_expert_variant default to "gemma_2b" / "gemma_300m"
#     (no LoRA), freeze_filter=Nothing, ema_decay=0.99 (TrainConfig default).
#   - Full backbones at bf16 do not fit on a single 24 GB card; run on a multi-GPU host
#     so openpi.training.sharding can FSDP the model. The training loop enforces
#     batch_size % jax.device_count() == 0 -- with batch_size=32, use 1/2/4/8/16/32 GPUs.
#   - On this machine the local 4090+3090 mix is known to trip a JAX cross-device
#     dispatch error (see the LoRA script); run on a uniform-GPU host instead.
#
# Two phases as in the LoRA script: smoke (batch=1, 200 steps) -> real finetune.

export LEROBOT_HOME=/PublicSSD/cspark/TA-VLA/datasets/lerobot
# Do NOT hard-pin to a single GPU here; full FT needs FSDP across multiple devices.
# Caller is expected to export CUDA_VISIBLE_DEVICES with a homogeneous GPU set.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

# --- Phase 1: smoke (200 steps, batch=1, no wandb) -------------------------------------
SMOKE_NORM_STATS=assets/pi0_fr3_peg_in_hole_effort_full_smoke/fr3/peg_in_hole_tavla_train_100/norm_stats.json
if [ -f "$SMOKE_NORM_STATS" ]; then
    echo "[1/4] smoke norm stats already at $SMOKE_NORM_STATS, skipping recompute."
else
    echo "[1/4] computing smoke norm stats..."
    uv run -- python scripts/compute_norm_stats.py --config-name=pi0_fr3_peg_in_hole_effort_full_smoke
fi

echo "[2/4] smoke training (200 steps)..."
uv run -- python scripts/train.py \
    pi0_fr3_peg_in_hole_effort_full_smoke \
    --exp_name="smoke_$(date +%Y%m%d_%H%M%S)" \
    --overwrite

# --- Phase 2: real full finetune (batch=32, wandb on, FSDP across visible GPUs) ---------
REAL_NORM_STATS=assets/pi0_fr3_peg_in_hole_effort_full/fr3/peg_in_hole_tavla_train_100/norm_stats.json
if [ -f "$REAL_NORM_STATS" ]; then
    echo "[3/4] real norm stats already at $REAL_NORM_STATS, skipping recompute."
else
    echo "[3/4] computing real norm stats..."
    uv run -- python scripts/compute_norm_stats.py --config-name=pi0_fr3_peg_in_hole_effort_full
fi

echo "[4/4] real full-FT training..."
uv run -- python scripts/train.py \
    pi0_fr3_peg_in_hole_effort_full \
    --exp_name="run_$(date +%Y%m%d_%H%M%S)"
