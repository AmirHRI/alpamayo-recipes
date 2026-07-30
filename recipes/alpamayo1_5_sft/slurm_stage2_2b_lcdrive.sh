#!/bin/bash
#SBATCH --job-name=a1_5_stage2_2b
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/stage2_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/stage2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Stage 2 action-expert training for Alpamayo-1.5 (2B Cosmos-Reason2 backbone) on
# the official LCDrive train split. The VLM is FROZEN (loaded from the Stage-1
# LCDrive checkpoint); only the ~0.47B action expert + diffusion head are trained,
# so this is far lighter than Stage 1 and fits comfortably on multiple H100s.
#
# Env overrides:
#   GPUS       number of GPUs / processes (default 4)
#   TRAIN_BS   per_device_train_batch_size (default 4)
#   GRAD_ACC   gradient_accumulation_steps (default 1)
#   MAX_STEPS  cap training steps for a smoke test (default: unset = full run)
#   RESUME     resume from checkpoint: "true" (latest in output_dir) or a path

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft
OUT_DIR=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training
mkdir -p "$OUT_DIR"

cd "$RECIPE_DIR"

GPUS="${GPUS:-4}"
TRAIN_BS="${TRAIN_BS:-10}"
GRAD_ACC="${GRAD_ACC:-1}"

# ── Weights & Biases ────────────────────────────────────────────────
# Auth resolves via (1) WANDB_API_KEY if exported, else (2) ~/.netrc from
# `wandb login`. Don't hard-fail if the env var is unset.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
else
    echo "[slurm] WANDB_API_KEY not set; falling back to ~/.netrc for W&B auth."
fi

# Unique master port per job to avoid collisions on shared nodes.
MASTER_PORT=$((29560 + SLURM_JOB_ID % 20000))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional smoke-test cap.
EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
    # `max_steps` isn't in the base trainer struct, so append it with `+`.
    EXTRA_ARGS+=("+trainer.max_steps=${MAX_STEPS}")
fi

# Optional resume after an interrupted run. RESUME=true auto-detects the latest
# checkpoint in output_dir; RESUME=/path/to/checkpoint-N resumes from a specific one.
if [[ -n "${RESUME:-}" ]]; then
    EXTRA_ARGS+=("+trainer.resume_from_checkpoint=${RESUME}")
fi

echo "[slurm] Stage-2 LCDrive: GPUS=${GPUS} TRAIN_BS=${TRAIN_BS} GRAD_ACC=${GRAD_ACC} " \
     "effective_batch=$((TRAIN_BS * GRAD_ACC * GPUS)) MAX_STEPS=${MAX_STEPS:-<full>}"

# Effective batch = per_device (TRAIN_BS) x grad_accum (GRAD_ACC) x gpus (GPUS).
srun "$RECIPE_DIR/a1_5_sft/bin/torchrun" \
    --nproc_per_node "$GPUS" \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage2_cosmos2b_lcdrive \
    trainer.per_device_train_batch_size="$TRAIN_BS" \
    trainer.gradient_accumulation_steps="$GRAD_ACC" \
    wandb.team=zrb20 \
    wandb.project=alpamayo1_5-sft-cosmos2b \
    run_name=lcdrive_2b_a1_5_stage2 \
    "${EXTRA_ARGS[@]}"
