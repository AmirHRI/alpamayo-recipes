#!/bin/bash
#SBATCH --job-name=a1_5_eval_10b_tok
#SBATCH --partition=debug
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/eval10b_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/eval10b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Token-trajectory evaluation of the released Alpamayo-1.5-10B on the LCDrive val
# split (VLM discrete-token path only, no action expert / diffusion), for an
# apples-to-apples minADE comparison against our 2B Stage-1 model.
#
#   sbatch slurm_eval_10b_token_lcdrive.sh                    # full val, batch 8
#   EVAL_BS=8 MAX_EVAL_STEPS=5 sbatch slurm_eval_10b_token_lcdrive.sh   # smoke

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft
OUT_DIR=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training
mkdir -p "$OUT_DIR"

cd "$RECIPE_DIR"

MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"   # -1 = full val split
EVAL_BS="${EVAL_BS:-8}"                  # 8B VLM: smaller eval batch than the 2B
echo "[slurm] max_eval_steps=$MAX_EVAL_STEPS (-1 = full LCDrive val split)"
echo "[slurm] per_device_eval_batch_size=$EVAL_BS"

# ── Weights & Biases ────────────────────────────────────────────────
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$((29540 + SLURM_JOB_ID % 20000))

srun "$RECIPE_DIR/a1_5_sft/bin/torchrun" \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_eval_10b_token_lcdrive \
    evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    trainer.per_device_eval_batch_size="$EVAL_BS" \
    wandb.team=zrb20 \
    wandb.project=alpamayo1_5-sft-cosmos2b
