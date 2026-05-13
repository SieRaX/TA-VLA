#!/usr/bin/env bash
# TA-VLA pi0 LoRA finetune on Franka FR3 Erase_whiteboard with torque (Dec-H + future-effort).
# Variant: EffortType.EXPERT_HIS_C_FUT, effort_dim=7, 10-frame past history at 10 fps.
#
# Runs in two phases: a 200-step smoke to validate the new FR3+effort+future-prediction code
# path, then the real 30k-step finetune. set -e aborts before the real run if smoke fails.

set -euo pipefail
cd "$(dirname "$0")/../.."

set -euo pipefail
# fr3/ symlinks live in both TA-VLA and CompVLA lerobot trees (same physical dir).
export LEROBOT_HOME=${LEROBOT_HOME:-/PublicSSD/cspark/CompVLA/datasets/lerobot}
# Pin to a single GPU; mixed 4090+3090 trips a JAX cross-device dispatch error.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# JAX preallocates 75% of the GPU by default; bump it so the LoRA pi0 fits on a 24 GB card.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

# --- Phase 1: smoke (200 steps, batch=1, no wandb) -------------------------------------
SMOKE_NORM_STATS=assets/pi0_fr3_erase_whiteboard_effort_smoke/fr3/erase_whiteboard_joint_train_100/norm_stats.json
if [ -f "$SMOKE_NORM_STATS" ]; then
    echo "[1/4] smoke norm stats already at $SMOKE_NORM_STATS, skipping recompute."
else
    echo "[1/4] computing smoke norm stats..."
    uv run -- python scripts/compute_norm_stats.py --config-name=pi0_fr3_erase_whiteboard_effort_smoke
fi

echo "[2/4] smoke training (200 steps)..."
uv run -- python scripts/train.py \
    pi0_fr3_erase_whiteboard_effort_smoke \
    --exp_name="smoke_$(date +%Y%m%d_%H%M%S)" \
    --overwrite

# --- Phase 2: real finetune (30k steps, batch=32, wandb on) ----------------------------
# Same dataset, but assets are keyed by config name so a separate norm_stats file is needed.
REAL_NORM_STATS=assets/pi0_fr3_erase_whiteboard_effort/fr3/erase_whiteboard_joint_train_100/norm_stats.json
if [ -f "$REAL_NORM_STATS" ]; then
    echo "[3/4] real norm stats already at $REAL_NORM_STATS, skipping recompute."
else
    echo "[3/4] computing real norm stats..."
    uv run -- python scripts/compute_norm_stats.py --config-name=pi0_fr3_erase_whiteboard_effort
fi

echo "[4/4] real training (30k steps)..."
uv run -- python scripts/train.py \
    pi0_fr3_erase_whiteboard_effort \
    --exp_name="run_$(date +%Y%m%d_%H%M%S)"
