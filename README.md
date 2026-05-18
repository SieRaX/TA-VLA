# TA-VLA — Setup & Training

This README covers (1) setting up the environment from scratch and (2) training the `pi0_fr3_erase_whiteboard_effort` config on the `fr3/erase_whiteboard_tavla_train_100` dataset.

For project background, the torque input-type design space, dataset format, deployment notes, and citation, see [`README_original.md`](README_original.md).

---

## 1. Environment setup

The instructions below mirror the upstream openpi setup. Requirements:

- Linux with an NVIDIA GPU (CUDA 12; tested on a 24 GB RTX 4090)
- Python ≥ 3.11 (provisioned automatically by `uv`)
- `git`, `git-lfs`, and `curl`

### 1.1 Install `uv`

The project uses [`uv`](https://docs.astral.sh/uv/) as its package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reload your shell, or:
export PATH="$HOME/.local/bin:$PATH"
```

### 1.2 Clone the repository

This fork uses git submodules under `third_party/` (`aloha`, `libero`). Clone recursively:

```bash
git clone --recurse-submodules https://github.com/SieRaX/TA-VLA.git
cd TA-VLA
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 1.3 Install Python dependencies

`GIT_LFS_SKIP_SMUDGE=1` avoids pulling large LFS blobs from transitive deps during dependency resolution; the openpi base checkpoint is fetched separately at training time from `s3://openpi-assets/`.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

This creates a `.venv/` and installs JAX (with CUDA 12 wheels), Flax, the openpi package, and the workspace member `packages/openpi-client`. Verify the install:

```bash
uv run python -c "import jax; print(jax.devices())"
```

You should see at least one `CudaDevice`.

### 1.4 (Optional) pre-commit hooks for development

```bash
uv run pre-commit install
```

### 1.5 Prepare the dataset

Training expects the dataset in standard **lerobot** format with an additional `observation.effort` field (per-frame joint torque, parallel to `observation.state`). Place the dataset under your `LEROBOT_HOME` so the resolved path is `$LEROBOT_HOME/fr3/erase_whiteboard_tavla_train_100/`.

For example, the layout used in this repo:

```bash
export LEROBOT_HOME=/PublicSSD/cspark/TA-VLA/datasets/lerobot
ls "$LEROBOT_HOME/fr3/erase_whiteboard_tavla_train_100"
# meta/  data/  videos/   (standard lerobot layout)
```

The `fr3/` prefix is required because HuggingFace `repo_id` strings allow only one slash; the directory can be a symlink to the actual data location.

---

## 2. Train on `fr3/erase_whiteboard_tavla_train_100`

The config `pi0_fr3_erase_whiteboard_effort` (defined in `src/openpi/training/config.py`) finetunes π₀ with LoRA on the FR3 erase-whiteboard dataset using torque input + future-torque prediction (`EffortType.EXPERT_HIS_C_FUT`, `effort_dim=7`, 10-frame history). A smaller smoke config `pi0_fr3_erase_whiteboard_effort_smoke` exists to validate the pipeline in ~200 steps before launching the real run.

### 2.1 One-shot script

The convenience script `scripts/fr3/train_fr3_erase_whiteboard_effort.sh` runs the full two-phase pipeline (smoke → real):

```bash
cd /path/to/TA-VLA
bash scripts/fr3/train_fr3_erase_whiteboard_effort.sh
```

The script handles `LEROBOT_HOME`, GPU pinning, JAX memory fraction, and norm-stats caching automatically.

### 2.2 Manual step-by-step

If you'd rather run each step yourself, the four commands below are equivalent to the script. Adjust `LEROBOT_HOME` and `CUDA_VISIBLE_DEVICES` to your machine.

```bash
cd /path/to/TA-VLA

# Environment
export LEROBOT_HOME=/PublicSSD/cspark/TA-VLA/datasets/lerobot
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# [1/4] Smoke: compute norm stats (only needed once; cached under assets/)
uv run python scripts/compute_norm_stats.py \
    --config-name=pi0_fr3_erase_whiteboard_effort_smoke

# [2/4] Smoke: 200-step run, batch=1, wandb off
uv run python scripts/train.py \
    pi0_fr3_erase_whiteboard_effort_smoke \
    --exp_name="smoke_$(date +%Y%m%d_%H%M%S)" \
    --overwrite

# [3/4] Real: compute norm stats (separate file, keyed by config name)
uv run python scripts/compute_norm_stats.py \
    --config-name=pi0_fr3_erase_whiteboard_effort

# [4/4] Real: 250k-step finetune, batch=8, wandb on
uv run python scripts/train.py \
    pi0_fr3_erase_whiteboard_effort \
    --exp_name="run_$(date +%Y%m%d_%H%M%S)"
```

Notes:

- `scripts/train.py` requires `batch_size % jax.device_count() == 0`. The default `batch_size=8` runs on a single GPU; if you use multiple, ensure 8 is divisible by the device count or override `--batch_size`.
- Checkpoints land under `checkpoints/pi0_fr3_erase_whiteboard_effort/<exp_name>/`. The wandb run ID is written to `wandb_id.txt` inside that directory and reused on resume.
- The real config trains for 250,108 steps (~62 epochs over the 32,273-frame dataset) and saves every 5,000 steps.

### 2.3 Serve the trained policy

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fr3_erase_whiteboard_effort \
    --policy.dir=checkpoints/pi0_fr3_erase_whiteboard_effort/<exp_name>/<step>
```
