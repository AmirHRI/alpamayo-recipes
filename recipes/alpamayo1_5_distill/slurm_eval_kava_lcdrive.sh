#!/bin/bash
#SBATCH --job-name=a1_5_kava_eval
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kavaeval_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kavaeval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# minADE for the trained KAVA Stage-1 student on the official LCDrive VAL split
# (23,331 clips), via the same ReasoningSampler + DistanceMetrics the Stage-1 and
# Stage-2 evals use — so the number is directly comparable to those baselines.
#
#   MAX_EVAL_STEPS=20 sbatch slurm_eval_kava_lcdrive.sh    # smoke, ~2 min
#   sbatch slurm_eval_kava_lcdrive.sh                      # full val split
#
#   # §7a DEAD-SLOT ABLATION — run this second, it is the decisive test.
#   ZERO_SLOTS=true sbatch slurm_eval_kava_lcdrive.sh
#
# Interpreting the pair:
#   zero_slots=false vs true differ  -> the slots carry real signal
#   they do NOT differ               -> the slots are decorative and L_KV achieved
#                                       nothing, however cleanly it converged
#                                       (kv_loss fell 11x during training, which on
#                                       its own proves only that the objective is
#                                       optimisable — not that it helps driving)
#
# ⚠️ The checkpoint MUST go to model.kava_checkpoint_path, not evaluate.eval_ckpt.
# evaluate_hf routes eval_ckpt into model.checkpoint_path, which loads ONLY vlm.* —
# the trained slots would be silently replaced by fresh vocab-init.

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin
OUT_DIR=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training

CKPT="${CKPT:-$OUT_DIR/output_stage1_kava_cosmos2b_lcdrive/checkpoint-7191}"
CACHE_ROOT="${CACHE_ROOT:-$OUT_DIR/teacher_kv_lcdrive}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"    # -1 = full LCDrive val split
EVAL_BS="${EVAL_BS:-2}"
ZERO_SLOTS="${ZERO_SLOTS:-false}"

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
# Rollout sampling materialises large fp32 logits; expandable segments reclaim the
# reserved-but-unallocated blocks that otherwise trigger OOM mid-eval.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$((29540 + SLURM_JOB_ID % 20000))

echo "[slurm] job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
echo "[slurm] checkpoint : $CKPT"
echo "[slurm] zero_slots : $ZERO_SLOTS   $([ "$ZERO_SLOTS" = "true" ] && echo '(DEAD-SLOT ABLATION)')"
echo "[slurm] max_eval_steps=$MAX_EVAL_STEPS  eval_bs=$EVAL_BS"
[ -d "$CKPT" ] || { echo "[slurm] FATAL: checkpoint dir not found"; exit 1; }

srun "$VENV/torchrun" \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_eval_kava_cosmos2b_lcdrive \
    model.kava_checkpoint_path="$CKPT" \
    model.zero_slots="$ZERO_SLOTS" \
    evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    trainer.per_device_eval_batch_size="$EVAL_BS"
