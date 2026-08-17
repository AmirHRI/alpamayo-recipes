#!/bin/bash
#SBATCH --job-name=a1_5_pruned_expert
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
# ⚠️ --mem is REQUIRED. Without it this job is given the node's entire 515 GB, and every
# other job then queues on (Resources) even though GPUs are free -- that is what blocked
# the blockrandt continuation. 120G matches slurm_train_kd.sh, which coexists fine.
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/prunedexp_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/prunedexp_%j.err
#
# Train the PRUNED action expert, teacher VLM frozen. See
# configs/sft_prunedexpert_10b_lcdrive.yaml for what this measures and why.
#
#   PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34 sbatch slurm_train_pruned_expert.sh
#   SMOKE=1 PRUNE_EXPERT_LAYERS=... sbatch slurm_train_pruned_expert.sh   # 20 steps
set -euo pipefail
: "${PRUNE_EXPERT_LAYERS:?set PRUNE_EXPERT_LAYERS=comma,separated,indices}"
export PRUNE_EXPERT_LAYERS
SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$((29860 + SLURM_JOB_ID % 20000))
TAG="prunedexpert_$(echo "$PRUNE_EXPERT_LAYERS" | tr ',' '-')"
EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=2 ++trainer.save_strategy=no
            ++trainer.eval_strategy=no ++trainer.warmup_steps=0
            ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids=0-120 ++data.val_dataset.chunk_ids=0-120
            ++trainer.dataloader_num_workers=2)
    echo "[slurm] SMOKE mode: 20 steps"
fi
echo "[slurm] PRUNE_EXPERT_LAYERS=$PRUNE_EXPERT_LAYERS -> $OUT/output_prunedexpert_10b_lcdrive"
srun "$VENV/torchrun" --nproc_per_node 2 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_prunedexpert_10b_lcdrive \
    run_name="${TAG}_$(date +%m%d-%H%M)" \
    "${EXTRA[@]}"
