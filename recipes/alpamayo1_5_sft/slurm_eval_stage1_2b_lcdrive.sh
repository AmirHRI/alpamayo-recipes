#!/bin/bash
#SBATCH --job-name=a1_5_eval_2b
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/eval_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Evaluate the finished Stage-1 2B LCDrive checkpoint (minADE) on the official
# LCDrive val split. Uses the same sft_stage1_cosmos2b_lcdrive config (so the
# val_dataset UUID filter points at lcdrive_val_clip_uuids.txt) and the base
# `evaluate` block (ReasoningSampler + DistanceMetrics).
#
# The checkpoint and an optional max_eval_steps can be overridden at submit time:
#   sbatch slurm_eval_stage1_2b_lcdrive.sh                       # full val, checkpoint-3597
#   CKPT=.../checkpoint-3500 MAX_EVAL_STEPS=200 sbatch slurm_eval_stage1_2b_lcdrive.sh

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft
OUT_DIR=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training
mkdir -p "$OUT_DIR"

cd "$RECIPE_DIR"

# ── Checkpoint + eval extent (overridable via env) ──────────────────
CKPT="${CKPT:-$OUT_DIR/output_stage1_cosmos2b_lcdrive/checkpoint-3597}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"   # -1 = full val split
EVAL_BS="${EVAL_BS:-2}"                  # per-device eval batch size
echo "[slurm] Evaluating checkpoint: $CKPT"
echo "[slurm] max_eval_steps=$MAX_EVAL_STEPS (-1 = full LCDrive val split)"
echo "[slurm] per_device_eval_batch_size=$EVAL_BS"

# ── Weights & Biases ────────────────────────────────────────────────
# Auth resolves via WANDB_API_KEY if exported, else ~/.netrc. Metrics here are
# printed to the log regardless, so W&B is optional for eval.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
fi

# Reduce allocator fragmentation (large fp32 logits during rollout sampling).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single-GPU: SLURM binds the GPU and sets CUDA_VISIBLE_DEVICES for us.
MASTER_PORT=$((29540 + SLURM_JOB_ID % 20000))

srun "$RECIPE_DIR/a1_5_sft/bin/torchrun" \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage1_cosmos2b_lcdrive \
    evaluate.eval_ckpt="$CKPT" \
    evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    trainer.per_device_eval_batch_size="$EVAL_BS" \
    wandb.team=zrb20 \
    wandb.project=alpamayo1_5-sft-cosmos2b
