#!/usr/bin/env bash
# Smoke train: TA-VLA pi0 on Franka FR3 Pivot dataset, no effort conditioning.
# Verifies the FR3 data pipeline end-to-end before adding torque variants.
#
# TA-VLA's data_loader.transform_dataset requires precomputed norm stats, so
# we compute them first then run a tiny training loop.

set -euo pipefail
cd "$(dirname "$0")/../.."

export LEROBOT_HOME=/PublicSSD/cspark/CompVLA/datasets/lerobot
# Pin to a single GPU; mixed 4090+3090 trips a JAX cross-device dispatch error.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
# JAX preallocates 75% of the GPU by default; bump it so the LoRA pi0 fits on a 24 GB card.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

NORM_STATS=assets/pi0_fr3_pivot_smoke/fr3/pivot_joint_train_100/norm_stats.json
if [ -f "$NORM_STATS" ]; then
    echo "[1/2] norm stats already at $NORM_STATS, skipping recompute."
else
    echo "[1/2] computing norm stats..."
    uv run -- python scripts/compute_norm_stats.py --config-name=pi0_fr3_pivot_smoke
fi

echo "[2/2] training..."
uv run -- python scripts/train.py \
    pi0_fr3_pivot_smoke \
    --exp_name="smoke_$(date +%Y%m%d_%H%M%S)" \
    --overwrite
