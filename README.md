# TA-VLA — Setup & Training

This README covers (1) setting up the environment from scratch and (2) training on the `fr3/erase_whiteboard_tavla_train_100` and `fr3/pivot_box_tavla_train_100` datasets.

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

Training expects the dataset in standard **lerobot** format with an additional `observation.effort` field (per-frame joint torque, parallel to `observation.state`). Place each dataset under `LEROBOT_HOME` so the resolved paths are:

- `$LEROBOT_HOME/fr3/erase_whiteboard_tavla_train_100/`
- `$LEROBOT_HOME/fr3/pivot_box_tavla_train_100/`

For example, the layout used in this repo:

```bash
export LEROBOT_HOME=/PublicSSD/cspark/TA-VLA/datasets/lerobot
ls "$LEROBOT_HOME/fr3/erase_whiteboard_tavla_train_100"
ls "$LEROBOT_HOME/fr3/pivot_box_tavla_train_100"
# meta/  data/  videos/   (standard lerobot layout)
```

The `fr3/` prefix is required because HuggingFace `repo_id` strings allow only one slash; each directory can be a symlink to the actual data location.

---

## 2. Training

Each task has a two-phase shell script under `scripts/fr3/` that runs a 200-step smoke validation followed by the real finetune. The scripts handle `LEROBOT_HOME`, GPU pinning, JAX memory fraction, and norm-stats caching automatically.

### 2.1 Erase whiteboard

```bash
cd /path/to/TA-VLA
bash scripts/fr3/train_fr3_erase_whiteboard_effort.sh
```

### 2.2 Pivot box

```bash
cd /path/to/TA-VLA
bash scripts/fr3/train_fr3_pivot_box_effort.sh
```

### 2.3 Serve a trained policy

```bash
# Erase whiteboard
uv run python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fr3_erase_whiteboard_effort \
    --policy.dir=checkpoints/pi0_fr3_erase_whiteboard_effort/<exp_name>/<step>

# Pivot box
uv run python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fr3_pivot_box_effort \
    --policy.dir=checkpoints/pi0_fr3_pivot_box_effort/<exp_name>/<step>
```
