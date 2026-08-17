#!/bin/bash
#SBATCH --job-name=a1_5_eval_s2_2b
#SBATCH --partition=debug
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/eval_s2_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/eval_s2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Evaluate the finished Stage-2 action-expert model (2B Cosmos-Reason2 + expert) on
# the official LCDrive val split (minADE via the diffusion trajectory rollout).
#
# Unlike the Stage-1 eval, the checkpoint is loaded via the model's
# `stage2_checkpoint_path` (from_pretrained_vlm), NOT evaluate.eval_ckpt.
#
# Env overrides:
#   CKPT             Stage-2 checkpoint dir (default checkpoint-11502)
#   MAX_EVAL_STEPS   cap eval steps for a smoke test (default -1 = full val split)
#   EVAL_BS          per_device_eval_batch_size (default 4)
#
#   sbatch slurm_eval_stage2_2b_lcdrive.sh                      # full val, checkpoint-11502
#   MAX_EVAL_STEPS=5 sbatch slurm_eval_stage2_2b_lcdrive.sh     # 5-step smoke test

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft
OUT_DIR=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training
mkdir -p "$OUT_DIR"

cd "$RECIPE_DIR"

CKPT="${CKPT:-$OUT_DIR/output_stage2_cosmos2b_lcdrive/checkpoint-11502}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"
EVAL_BS="${EVAL_BS:-10}"
echo "[slurm] Stage-2 eval checkpoint: $CKPT"
echo "[slurm] max_eval_steps=$MAX_EVAL_STEPS (-1 = full LCDrive val split)"
echo "[slurm] per_device_eval_batch_size=$EVAL_BS"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$((29580 + SLURM_JOB_ID % 20000))

srun "$RECIPE_DIR/a1_5_sft/bin/torchrun" \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_eval_stage2_cosmos2b_lcdrive \
    model.stage2_checkpoint_path="$CKPT" \
    evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    trainer.per_device_eval_batch_size="$EVAL_BS"
