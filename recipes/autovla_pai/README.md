# AutoVLA-PAI Evaluation Recipe

Evaluate **AutoVLA / SKIPlan** trajectory-prediction models on the
[Physical AI AV (PAI) dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
using the **exact same** `alpamayo` `MetricRunner` pipeline as
`alpamayo1_sft/evaluate_hf.py`, so numbers are directly comparable.

## Recipe files

```
recipes/autovla_pai/
├── README.md
├── pyproject.toml
└── autovla_pai/
    ├── __init__.py
    ├── evaluate.py          ← CLI entry point (argparse + torchrun-ready DDP)
    ├── autovla_wrapper.py   ← AutoVLAWrapper: implements sample_trajectories_from_data
    └── pai_adapter.py       ← PAI batch → AutoVLA input format (Qwen3-VL video + ego state)
```

## How it works

```
PAI batch (ego_history_xyz, image_frames, …)
      │
      ▼
  pai_adapter.build_autovla_batch()
      │  front-wide camera frames → Qwen3-VL video
      │  ego_history_xyz / ego_history_rot → swin_ego_state
      ▼
  AutoVLAWrapper.sample_trajectories_from_data()
      │  forward_proposal() → stage3_xy [B, Tf, 2]
      │  returns pred_xyz [B, N=1, K=1, Tf, 3]  (Z=0)
      │          pred_rot [B, N=1, K=1, Tf, 3, 3]  (yaw-only from motion direction)
      ▼
  alpamayo MetricRunner
    ├── ReasoningSampler   (stores pred_xyz/rot in output_batch)
    └── DistanceMetrics(time_step=0.5)
          → min_ade, min_ade/by_t={0.5,1.0,3.0}, ade, ade/by_t=3.0, corner_distance
```

> **Horizon note:** AutoVLA predicts **4 s** (8 × 0.5 s); Alpamayo SFT predicts **6.4 s**
> (64 × 0.1 s). `DistanceMetrics` requests `by_t={0.5, 1.0, 3.0, 5.0}` s — the 5.0 s step
> is automatically dropped when `Tf < 10`.  Metrics at `t=0.5`, `1.0`, `3.0 s` and
> `ade/by_t=3.0` are **directly comparable** across all models. The overall `min_ade`
> (full horizon) is **not** directly comparable (AutoVLA = 4 s, Alpamayo = 6.4 s).

---

## Results — chunks 186–205 (1 896 samples)

All units in **metres**. Evaluated on the same 20 validation chunks.
Alpamayo Stage 1: single sample (`K=1`). Stage 2: 6 diffusion draws (`K=6`).
AutoVLA: deterministic `forward_proposal` (`K=1`, no map polylines).

| Metric | 2B Stage 1 | 2B Stage 2 | 10B Stage 1 | 10B Stage 2 | **AutoVLA (nuPlan)** | **AutoVLA (plan-token)** |
|--------|:----------:|:----------:|:-----------:|:-----------:|:--------------------:|:------------------------:|
| `min_ade` (full horizon †) | 4.735 | 3.543 | 0.952 | **0.874** | 4.945 | 9.247 |
| `min_ade @ t=0.5 s` | 0.046 | 0.043 | **0.009** | 0.011 | 0.827 | 1.908 |
| `min_ade @ t=1.0 s` | 0.157 | 0.141 | **0.030** | 0.036 | 1.289 | 2.847 |
| `min_ade @ t=3.0 s` ★ | 1.172 | 0.931 | **0.242** | 0.260 | 3.560 | 7.011 |
| `ade` | 5.943 | 6.014 | 1.742 | 2.131 | 4.945 | 9.247 |
| `ade @ t=3.0 s` ★ | 1.410 | 1.490 | 0.361 | 0.442 | 3.560 | 7.011 |
| `corner_distance` | 4.699 | 3.437 | 0.990 | **0.865** | 5.006 | 9.360 |

† full horizon: Alpamayo = 6.4 s, AutoVLA = 4 s — **not directly comparable**
★ directly comparable across all models

**Key observations:**
- AutoVLA nuPlan (transformer head, no map) matches the undertrained 2B Stage 1 on overall
  `min_ade` but is **18× worse** at `t=0.5 s` (0.827 vs 0.046 m), indicating poor short-horizon
  trajectory accuracy despite comparable total displacement.
- AutoVLA plan-token performs significantly worse than the transformer-head variant on PAI
  (`min_ade @ t=3.0 s`: 7.01 vs 3.56 m), suggesting the plan-token discretisation is even
  less suited to the PAI motion distribution.
- Without HD-map polylines (PAI has no NuPlan-format map), both AutoVLA variants lose their
  primary conditioning signal. `min_ade @ t=3.0 s = 3.56 m` (best AutoVLA) vs **0.24 m** for
  Alpamayo-R1-10B Stage 1 captures the domain gap between nuPlan-trained planners and PAI.
- `ade == min_ade` for both AutoVLA variants because `K=1` (deterministic). For Alpamayo
  Stage 2, `ade > min_ade` because 6 stochastic draws explore diverse trajectories.

---

## Available checkpoints

| Config | Checkpoint | Notes |
|--------|------------|-------|
| `/data/sungyeonpark/autovla/ckpts/paper/nuplan/config.yaml` | `final.ckpt` | NuPlan Qwen3-VL-8B, transformer proposal head — **results above** |
| `/data/sungyeonpark/autovla/ckpts/paper/plan_token/plan-0-config.yaml` | `plan-0.ckpt` | NuPlan plan-token variant — **results above** |
| `/data/sungyeonpark/autovla/ckpts/paper/plan_token/plan-0-config.yaml` | `plan-1.ckpt` | plan-token seed 1 (not evaluated) |
| `/data/sungyeonpark/autovla/ckpts/paper/plan_token/plan-0-config.yaml` | `plan-2.ckpt` | plan-token seed 2 (not evaluated) |
| `/data/sungyeonpark/autovla/ckpts/paper/plan_token/plan-0-config.yaml` | `plan-8.ckpt` | plan-token seed 8 (not evaluated) |

---

## Setup

### 1. Create venv

```bash
cd /home/achahe/alpamayo-recipes/recipes/autovla_pai
uv venv autovla_env --python 3.12
source autovla_env/bin/activate
uv sync --active   # resolves 85 packages; uses cached flash-attn wheel (~10 s)
```

### 2. Install SKIPlan / AutoVLA (no heavy deps)

SKIPlan's `requirements.txt` lists `nuplan-devkit`, which is not needed for inference.
Install the package code only:

```bash
uv pip install --no-deps -e /home/achahe/SKIPlan
```

### 3. Verify

```bash
python -c "
from autovla_pai.autovla_wrapper import AutoVLAWrapper
from alpamayo.metrics.metric_api import ReasoningSampler, DistanceMetrics
print('All imports OK')
"
```

---

## Run evaluation

All commands assume `cd /home/achahe/alpamayo-recipes/recipes/autovla_pai` with venv active.

### Multi-GPU (4 × H100, recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
    -m autovla_pai.evaluate \
    --config  /data/sungyeonpark/autovla/ckpts/paper/nuplan/config.yaml \
    --ckpt    /data/sungyeonpark/autovla/ckpts/paper/nuplan/final.ckpt \
    --pai_dir /temp/achahe/physical_ai_av \
    --chunk_ids 186-205 \
    --batch_size 4
# ~8 min on 4 × H100
```

### Single GPU

```bash
python -m autovla_pai.evaluate \
    --config  /data/sungyeonpark/autovla/ckpts/paper/nuplan/config.yaml \
    --ckpt    /data/sungyeonpark/autovla/ckpts/paper/nuplan/final.ckpt \
    --pai_dir /temp/achahe/physical_ai_av \
    --chunk_ids 186-205 \
    --batch_size 4
```

### plan-token variant

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
    -m autovla_pai.evaluate \
    --config  /data/sungyeonpark/autovla/ckpts/paper/plan_token/plan-0-config.yaml \
    --ckpt    /data/sungyeonpark/autovla/ckpts/paper/plan_token/plan-0.ckpt \
    --pai_dir /temp/achahe/physical_ai_av \
    --chunk_ids 186-205
```

### CLI options

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | *(required)* | Path to AutoVLA `config.yaml` |
| `--ckpt` | *(required)* | Path to `.ckpt` checkpoint |
| `--pai_dir` | *(required)* | PAI dataset root directory |
| `--chunk_ids` | `186-205` | Chunk range, e.g. `"186-205"` or `"0-9"` |
| `--batch_size` | `4` | Per-GPU batch size |
| `--max_samples` | `-1` | Cap total samples (−1 = all) |
| `--num_workers` | `4` | DataLoader workers |

---

## Design notes

### Deterministic inference
`forward_proposal` is fully deterministic (proposal head → action tokens → `action_to_traj`,
or FM-refiner `refined_mean` when a refiner is configured). Calling with `num_traj_samples > 1`
simply tiles the same trajectory; `K=1` is correct for AutoVLA.

### Map polylines
Both nuPlan configs specify `map_polylines.enabled: true` pointing to nuPlan HD-map files
that do not exist for PAI clips. `forward_proposal` uses `batch.get("map_polylines")` which
returns `None`; the proposal head runs without map conditioning. This is the primary source
of AutoVLA's degraded accuracy on PAI.

### Coordinate frame
`UnicycleAccelCurvatureActionSpace.action_to_traj` outputs ego-local XY with
`initial_x = initial_y = initial_yaw = 0` (origin at t0 ego position, +X = forward, +Y = left).
PAI's `ego_future_xyz` is also ego-local (`t0_rot_inv @ (world_xyz − t0_xyz)`), same convention.
No coordinate transform is required.

### Temporal resolution
PAI stores at 0.1 s natively, but `PAIDataset` accepts an arbitrary `time_step`.
We load at `time_step=0.5` (4 history + 8 future steps) to match AutoVLA's training resolution.

### `ego_future_rot` in batch
`DistanceMetrics` reads `data_batch["ego_future_rot"]` to compute `corner_distance`.
`load_physical_aiavdataset` (from `alpamayo_r1`) populates this as `(1, 1, Tf, 3, 3)`.
After `PAIDataset.squeeze(0)` and DataLoader batching the shape is `(B, 1, Tf, 3, 3)`.
`DistanceMetrics` indexes `[:, -1]` → `(B, Tf, 3, 3)`. ✓
