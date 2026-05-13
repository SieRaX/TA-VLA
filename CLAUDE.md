# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

TA-VLA is a fork of Physical Intelligence's [openpi](https://github.com/Physical-Intelligence/openpi) (branched from upstream commit `cd82848`) implementing the CoRL 2025 paper "TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models". Most of the code is upstream openpi; the project's own contribution is wiring joint **torque/effort** signals into the π₀ model in several different ways. Upstream openpi documentation in `docs/`, `examples/`, and `README.md` (e.g. setup, dataset, deployment) still applies — consult upstream's README for general environment setup; this repo's `README.md` documents only the deltas.

## Common commands

This project uses [`uv`](https://docs.astral.sh/uv/) (workspace member: `packages/openpi-client`). Python ≥ 3.11.

- Train: `uv run scripts/train.py <config_name> --exp_name=<name>` (e.g. `pi0_lora_effort_history`). Config names are defined in `src/openpi/training/config.py` (`_CONFIGS`); `cli()` exposes them via tyro.
- Compute norm stats (required before training a new dataset): `uv run scripts/compute_norm_stats.py <config_name>`.
- Serve policy: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<path>` (or `--env=[DROID|ALOHA|LIBERO]` for built-in envs).
- Lint / format: `ruff check .` and `ruff format .` (config in `pyproject.toml`, line length 120, target py311).
- Pre-commit (runs `uv-lock` + `ruff`): `pre-commit install` then `pre-commit run --all-files`.
- Tests: `uv run pytest` (testpaths: `src`, `scripts`, `packages`). Run a single test with `uv run pytest src/openpi/models/pi0_test.py::<TestName>`. Tests marked `manual` are skipped by default.

`scripts/train.py` requires `config.batch_size % jax.device_count() == 0`. It writes `wandb_id.txt` into the checkpoint dir; resuming reads it back.

## Architecture overview

Top-level layout (only the parts that need cross-file context):

- `src/openpi/models/` — model implementations. `pi0.py` is the diffusion-based π₀ model (the one TA-VLA modifies); `pi0_fast.py` is the autoregressive variant. **TA-VLA's effort handling is only tested on `pi0.py`; `pi0_fast` may not be compatible.** `gemma.py` / `gemma_fast.py` / `siglip.py` / `vit.py` are the LLM and vision backbones; `lora.py` provides LoRA wrappers used by the `*_lora` variants.
- `src/openpi/policies/` — per-robot input/output transforms (image renaming, padding, action packing). `tavla_policy.py` is the TA-VLA-specific transform: it expects `cam_high` / `cam_left_wrist` / `cam_right_wrist`, `observation.state` (14-dim), and optionally `observation.effort` (history × 14), and pads state/actions to the model's `action_dim`. Output cuts back to the first 14 dims.
- `src/openpi/training/config.py` — single source of truth for run configurations. `_CONFIGS` is a list of `TrainConfig`s; `LeRobotTavlaDataConfig` is the dataset factory used by TA-VLA configs. The TA-VLA-specific configs are `pi0_lora_baseline`, `pi0_lora_effort`, `pi0_lora_effort_history` (around lines 817–871). Everything threads through `DataConfigFactory.create()` → `DataConfig` → data loader.
- `src/openpi/shared/effort_type.py` — defines `EffortType` enum (`NO`, `STATE`, `LLM`, `LLM_HIS_C`, `LLM_HIS_T`, `EXPERT`, `EXPERT_HIS_C`, `EXPERT_HIS_T`, `EXPERT_FUT`, `EXPERT_HIS_C_FUT`, `EXPERT_HIS_C_L_FUT`). The docstrings map each variant to the paper's method names (DePre / Enc / Enc-1 / Enc-H / DePost / Dec-1 / Dec-H, plus future-effort variants). The token-layout comment at the top of the file is the canonical reference for where each effort token is injected:
  ```
  |<------------prefix(w=2048)------------>|<-----------suffix(w=1024)------------->|
  |<-images->|<-language->|<-effort(llm)->|<-effort(expert)->|<-state->|<-actions->|
  ```
- `scripts/train.py` — top-level training loop (FSDP sharding via `openpi.training.sharding`, optimizer, checkpointing, wandb logging).

### How effort flows through the system

1. Dataset must be standard **lerobot** format with an extra `observation.effort` field (per-frame joint torque, parallel to `observation.state`).
2. `LeRobotTavlaDataConfig.effort_history` is a sequence of *relative frame offsets* (e.g. `(-36, -32, ..., 0)` for 10 frames spanning ~2 s). When non-empty, the repack transform pulls `observation.effort` into the batch under the `effort` key.
3. `TavlaInputs` (in `policies/tavla_policy.py`) passes `effort` through unchanged when present; `TavlaOutputs` slices output back to 14 dims.
4. `Pi0Config.effort_type` (an `EffortType`) selects how the effort tensor is consumed inside `models/pi0.py`. For `EXPERT_*_FUT` types, the model concatenates predicted future effort to the action target and computes a weighted loss `action_loss + 0.1 * effort_loss` (see `pi0.py` ~line 364–366).

When adding a new config that uses effort, you must set both `Pi0Config(effort_type=...)` and `LeRobotTavlaDataConfig(effort_history=...)` consistently — the data side controls what gets loaded; the model side controls what gets used.

### Norm stats

Norm stats live next to each checkpoint and are loaded via `DataConfigFactory._load_norm_stats`. `LeRobotTavlaDataConfig` supports a list of `repo_id`s — when given a list, stats are combined across datasets weighted by `total_frames` (see the merging logic ~lines 422–490). `padding_stat=True` pads stats to dim 32 so π₀ can reuse π₀-FAST's stats. See `docs/norm_stats.md` for which pre-training stat IDs are available (`trossen`, `droid`, `franka`, `ur5e`, `arx`, etc.).

### Deployment

Real-robot data collection / inference uses a modified AgileX example under `examples/aloha_real/`. The modification is that the client maintains a rolling history buffer of joint torques from the ROS topic so it can feed the same `effort_history` window the model was trained on.

## Conventions

- Ruff is the only linter/formatter; rules in `pyproject.toml`. Notable: `force-single-line` imports, `force-sort-within-sections`, line length 120.
- `print` statements are allowed (`T201` ignored).
- Submodules under `third_party/` (`aloha`, `libero`) are excluded from lint and pre-commit.
