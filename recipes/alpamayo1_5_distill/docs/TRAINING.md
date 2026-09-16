# Training: KD -> Cotrain EoS -> Consistency

[Back to the recipe](../README.md)

Run the stages in order for either model size. These are the **two-camera,
all-anchor, full-expert** runs, not the older pruned or four-camera variants.

## Environment and Data

Use Python 3.12 and the shared SFT environment. From the repository root:

```bash
cd recipes/alpamayo1_5_sft
uv venv .venv
source .venv/bin/activate
uv sync --active
cd ../alpamayo1_5_distill
export PYTHONPATH="$(cd .. && pwd)"
```

If the environment already exists, activate it without recreating it. See the
[SFT setup](../../alpamayo1_5_sft/README.md) for model preparation, including
conversion of the released 10B checkpoint to the A1-compatible layout. These
distillation stages are separate from the SFT recipe's own stage numbering.

Before launching, prepare:

- Alpamayo-1.5-10B teacher weights/config in A1-compatible layout, plus
  Qwen3-VL-4B-Instruct or Cosmos-Reason2-2B base weights.
- PhysicalAI/LCDrive data and the navigation manifests: training
  `nav_lcdrive_train_anchors_all.json` (109,997 event anchors), validation
  `nav_lcdrive_val_mysubset_1k.json`, and their clip-UUID filters.
- The two-camera training frame cache, including its completed `_index.json`.
  This is a **decoded-frame cache**, not a teacher KV or trajectory cache.
- Slurm access and suitable GPU/host memory. The commands below use four GPUs;
  use H200-class memory for the all-layer 4B cotraining run. Effective batch is
  32. Authenticate W&B locally or set `WANDB_MODE=disabled`.

**Port the site paths first.** The launchers and YAML configs contain absolute
`/home/achahe/...` and `/temp/achahe/...` paths, Slurm node/partition names, log
destinations, and W&B settings. Update these for your machine, including base
model snapshots, teacher/config paths, dataset paths, and output paths. The
consistency launchers enable `HF_HUB_OFFLINE=1`, so download models beforehand.
Creating an `OUT` variable below does not change hard-coded launcher paths.

If the frame cache is missing, adapt and run
[slurm_build_frame_cache.sh](../slurm_build_frame_cache.sh). Its interpreter path
currently uses `a1_5_sft/bin/python3`; point it at the environment above.

```bash
CAMERAS=1,3 sbatch slurm_build_frame_cache.sh
```

Wait for its verification, build, and finalization to succeed before KD.
All stages must agree on cameras `[1,3]`, four frames per camera, event anchors,
camera/frame labels, and `strip_nav_turn_distance=true`.

## 1. KD: Block/Span Matching

The teacher runs online. The student learns to supply a cache that reproduces
the frozen teacher expert's block/span outputs. Spans grow **1 -> 9 -> 18 -> 36**,
one per epoch; CE, logit KD, and elementwise KV losses are disabled. The 2B also
learns its 28-to-36 mixer. No offline teacher-feature cache is needed.

From the recipe directory, preview the normalized navigation input first:

```bash
unset PRUNE_EXPERT_LAYERS INIT RESUME EXTRA_ARGS
ARM=nav4bspan2camallfc bash slurm_train_kd.sh
ARM=mixspan2bnavfc bash slurm_train_kd.sh
```

These previews do not train. After reviewing the displayed input, submit the
desired run (`APPROVED=1` acknowledges the launcher's input-format gate):

```bash
APPROVED=1 ARM=nav4bspan2camallfc BS=8 ACCUM=1 sbatch slurm_train_kd.sh
APPROVED=1 ARM=mixspan2bnavfc BS=2 ACCUM=4 sbatch slurm_train_kd.sh
```

First add `SMOKE=1` to run 20 steps in the launcher's separate smoke directory.
Check finite loss, gradients, camera counts, and the loaded teacher before a full
run. The full run is four epochs, normally ending at `checkpoint-13752`.

The launcher supplies the curriculum overrides on top of the base
[4B KD config](../configs/sft_kd_qwen3_4b_2cam_nav_lcdrive.yaml) or
[2B layer-mix KD config](../configs/sft_kd_cosmos2b_2cam_nav_layermix_lcdrive.yaml).
Launching either base YAML alone does **not** select this curriculum.

The commands below use the existing site layout; substitute your completed runs:

```bash
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
KD4="$OUT/output_kd_4b_nav4bspan2camallfc_m1-9-18-36_framecache1080p_lcdrive/checkpoint-13752"
KD2="$OUT/output_kd_2b_mixspan2bnavfc_m1-9-18-36_framecache1080p_lcdrive/checkpoint-13752"
EOS4="$OUT/output_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache"
EOS2="$OUT/output_eos_cotrain_2bmix_all28_lr1x_nav_framecache"
```

Use fresh output directories for new experiments. KD automatically resumes from
its existing run directory by default; do not reuse another experiment's path.

## 2. Cotrain EoS: Adapt the VLM and Expert

Initialize from KD plus the released action expert. Train all **text layers**
(4B: `0-35`; 2B: `0-27`) and the expert with GT flow matching. The vision encoder,
embeddings, LM head, and 2B mixer remain frozen; gradients still pass through the
mixer into the text layers. Trajectory-token CE stays off.

Use the all-layer [4B cotraining config](../configs/sft_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache_lcdrive.yaml)
or [2B cotraining config](../configs/sft_eos_cotrain_2bmix_all28_lr1x_nav_framecache_lcdrive.yaml).
Despite `cotrain_vlm: false`, their `cotrain_vlm_layers` fields explicitly enable
text-layer training. Verify that the startup log reports the requested layers.

```bash
PRUNE_EXPERT_LAYERS=none MIX=0 BS=2 ACCUM=4 \
  CONFIG=sft_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache_lcdrive \
  EXTRA_ARGS="model.checkpoint_path=$KD4 paths.output_dir=$EOS4" \
  sbatch --gpus=4 --cpus-per-task=48 --mem=340G slurm_train_eos.sh

MIX=1 BS=2 ACCUM=4 \
  CONFIG=sft_eos_cotrain_2bmix_all28_lr1x_nav_framecache_lcdrive \
  EXTRA_ARGS="model.checkpoint_path=$KD2 paths.output_dir=$EOS2" \
  sbatch --gpus=4 --cpus-per-task=48 --mem=340G slurm_train_eos.sh
```

Use `SMOKE=1` first, changing `paths.output_dir` to a separate smoke directory:
**this launcher does not separate smoke outputs automatically**. Keep the same
checkpoint input and verify nonzero VLM/expert gradients and finite losses.
Then run the full two epochs at LR `2e-5`, warmup 430, effective batch 32.
The expected final checkpoint is `checkpoint-6876`.

## 3. Consistency: Two Steps to One

Initialize each run from **its own cotrained EoS checkpoint**. Freeze the VLM and
2B mixer. A frozen copy of the EoS expert generates the actual two-step Euler path
online; the trainable expert learns consistency along its two edges with an EMA
target. There is **no GT loss or cached 10B endpoint loss** in this stage.

Use the [4B consistency config](../configs/sft_cd_eos_4b_consistency2to1_2cam_nav_lcdrive.yaml)
or [2B consistency config](../configs/sft_cd_eos_2b_consistency2to1_2cam_nav_lcdrive.yaml).
The final precision is **fp16 AMP with fp32 trainable weights, Adam moments, and
EMA**. The 4B base config predates that choice, so its overrides below are required;
the 2B config already includes them.

```bash
CM4="$OUT/output_cd_eos4b_consistency2to1_fp16_master32_2cam_nav"
CM2="$OUT/output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav"
unset RESUME PRUNE_EXPERT_LAYERS

EOS_CKPT="$EOS4/checkpoint-6876" OUTPUT_DIR="$CM4" \
  EXTRA_ARGS='++model.expert_precision=fp16 trainer.bf16=false ++trainer.fp16=true' \
  sbatch slurm_train_cd_eos_4b_consistency2to1.sh

EOS_CKPT="$EOS2/checkpoint-6876" OUTPUT_DIR="$CM2" EXTRA_ARGS='' \
  sbatch slurm_train_cd_eos_2b_consistency2to1.sh
```

First run each command with `SMOKE=1` and `OUTPUT_DIR` set to a fresh smoke path.
The two-step smoke saves a checkpoint: check finite loss, nonzero expert gradients,
EMA liveness, and successful EMA-save verification. Full training uses two epochs,
LR `2e-5`, warmup 430, effective batch 32, and EMA decay 0.99.

The launchers require a **sharded EoS checkpoint with config**, reject nonempty
outputs, and do not support exact online/EMA resume. Do not bypass that guard with
`trainer.resume_from_checkpoint`. Deployment uses the saved EMA
`checkpoint-6876`, not the final root-level model save. Checkpoint step numbers
assume the full manifest and effective batch 32; verify them for your run.

Next: [evaluate at NFE=1 and compare saved metrics](EVALUATION.md). A one-step
expert does not imply that total VLM-plus-expert latency halves or meets 10 Hz on
every device.