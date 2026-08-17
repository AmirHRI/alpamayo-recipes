#!/bin/bash
#SBATCH --job-name=a1_5_stage1_2b
#SBATCH --partition=debug
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/slurm_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Stage 1 VLM fine-tune of the 2B Cosmos-Reason2 backbone for Alpamayo-1.5 on the
# official LCDrive train split. Single-GPU run (1 process, one H100/80GB).

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft
OUT_DIR=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training
mkdir -p "$OUT_DIR"

cd "$RECIPE_DIR"

# ── Weights & Biases ────────────────────────────────────────────────
# Auth resolves via (1) WANDB_API_KEY if exported in the submit env, else
# (2) ~/.netrc created by `wandb login`. Don't hard-fail if the env var is
# unset, so the job still runs from a shell that didn't export it.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
else
    echo "[slurm] WANDB_API_KEY not set; falling back to ~/.netrc for W&B auth."
fi

# Single-GPU: SLURM allocates one GPU and sets CUDA_VISIBLE_DEVICES for us.
# Unique master port per job to avoid collisions on shared nodes.
MASTER_PORT=$((29540 + SLURM_JOB_ID % 20000))

# Reduce allocator fragmentation. The causal-LM loss upcasts logits to fp32
# ([batch * seq_len, vocab~151k]); expandable segments help reclaim the large
# reserved-but-unallocated blocks that otherwise trigger OOM at the loss step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Effective batch = per_device (2) x grad_accum (16) x gpus (1) = 32.
# per_device stays small so the fp32 logits tensor for the LM loss fits in 80 GB.
srun "$RECIPE_DIR/a1_5_sft/bin/torchrun" \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage1_cosmos2b_lcdrive \
    trainer.per_device_train_batch_size=2 \
    trainer.gradient_accumulation_steps=16 \
    wandb.team=zrb20 \
    wandb.project=alpamayo1_5-sft-cosmos2b \
    run_name=lcdrive_2b_a1_5_stage1_1gpu
