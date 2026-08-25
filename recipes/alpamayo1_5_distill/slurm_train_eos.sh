#!/bin/bash
#SBATCH --job-name=a1_5_eos_train
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/eostrain_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/eostrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=56
#SBATCH --mem=170G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# EXPERT-ON-STUDENT: train the pruned action expert on a frozen student's cache.
# `from_student` sets requires_grad_(cotrain_vlm) on every VLM parameter and prints
# "VLM FROZEN, trainable X B" -- read that line, do not assume it.
#
#   CONFIG=sft_expert_on_student_2b_nav_lcdrive sbatch slurm_train_eos.sh
#   SMOKE=1 CONFIG=... sbatch slurm_train_eos.sh          # 20 steps
#
# ⚠️ NO PIN_GPUS here, deliberately. Pinning to a global index requires owning the whole node,
# and this arm is meant to coexist with another 2-GPU job. slurm's own assignment is correct
# WHEN the other occupant is a real slurm job -- which it is not always on this node, so the
# resolved uuid per rank is printed below. Read it against nvidia-smi -L before trusting a
# long run, and check no foreign process already sits on those cards.
#
# ⚠️ --mem is REQUIRED. Without it slurm hands over the node's entire RAM and every other job
# queues on (Resources) with GPUs idle. 170G is half the node, matching the 2-of-4 GPU share.
set -euo pipefail
: "${PRUNE_EXPERT_LAYERS:?set PRUNE_EXPERT_LAYERS=comma,separated,indices}"
export PRUNE_EXPERT_LAYERS
CONFIG="${CONFIG:?set CONFIG=<config name under configs/>}"
SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$((29880 + SLURM_JOB_ID % 20000))
GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-0}")

EXTRA=()
[[ -n "${BS:-}"     ]] && EXTRA+=(trainer.per_device_train_batch_size="$BS")
[[ -n "${ACCUM:-}"  ]] && EXTRA+=(trainer.gradient_accumulation_steps="$ACCUM")
[[ -n "${WARMUP:-}" ]] && EXTRA+=(trainer.warmup_steps="$WARMUP")
[[ -n "${EPOCHS:-}" ]] && EXTRA+=(++trainer.num_train_epochs="$EPOCHS")
# The cgroup exposes roughly half of --cpus-per-task, so derive workers from the allocation
# rather than trusting the config's default (sized for a bigger one).
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} / (3 * GPUS) ))}"
[[ "$WORKERS" -lt 2 ]] && WORKERS=2
EXTRA+=(++trainer.dataloader_num_workers="$WORKERS")
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=2
            ++trainer.save_strategy=no ++trainer.eval_strategy=no
            ++trainer.warmup_steps=0 ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids="0-600" ++data.val_dataset.chunk_ids="0-600")
    echo "[slurm] SMOKE: 20 steps"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

echo "[slurm] job=$SLURM_JOB_ID config=$CONFIG gpus=$CUDA_VISIBLE_DEVICES nproc=$GPUS workers=$WORKERS"
# Which PHYSICAL cards did slurm actually give us? Printed, not assumed.
"$VENV/python" - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    print(f"[slurm] rank{i} -> cuda:{i} uuid "
          f"{torch.cuda.get_device_properties(i).uuid}", flush=True)
PY
nvidia-smi -L

srun "$VENV/torchrun" --nproc_per_node "$GPUS" --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "$CONFIG" \
    "${EXTRA[@]}"
