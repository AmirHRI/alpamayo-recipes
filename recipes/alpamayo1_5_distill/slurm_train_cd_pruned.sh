#!/bin/bash
#SBATCH --job-name=a1_5_cd_p28
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdpruned_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdpruned_%j.err
#
# Two-epoch consistency distillation of the 28-active-layer action student against the
# original full 36-layer teacher. The frozen 8B VLM and all data/conditioning settings are
# identical to sft_cd_expert_2cam_nav_lcdrive.yaml.
#
# Global effective batch = 4 samples/rank x 2 ranks x 4 accumulation = 32.
#
#   sbatch slurm_train_cd_pruned.sh
#   SMOKE=1 sbatch slurm_train_cd_pruned.sh
set -euo pipefail

SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
PRUNE_MAP=4,10,13,15,19,25,27,34
PRUNE_CONFIG='[4,10,13,15,19,25,27,34]'
RUN_OUT="${OUTPUT_DIR:-$OUT/output_cd_expert_pruned28_2cam_nav_lcdrive_bs32_e2_20260826}"

if [[ -n "${PRUNE_EXPERT_LAYERS:-}" && "${PRUNE_EXPERT_LAYERS:-}" != "$PRUNE_MAP" ]]; then
    echo "[slurm] REFUSING mismatched PRUNE_EXPERT_LAYERS='$PRUNE_EXPERT_LAYERS'" >&2
    echo "[slurm] Expected '$PRUNE_MAP'." >&2
    exit 1
fi
export PRUNE_EXPERT_LAYERS="$PRUNE_MAP"

if [[ -d "$RUN_OUT" && -n "$(find "$RUN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[slurm] REFUSING non-empty output directory: $RUN_OUT" >&2
    echo "[slurm] Set OUTPUT_DIR to a new path for a fresh run." >&2
    exit 1
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAVA_GRAD_PROBE_STEPS="${KAVA_GRAD_PROBE_STEPS:-0}"
MASTER_PORT=$((30120 + SLURM_JOB_ID % 20000))

EXTRA=(
    "model.cd.student_prune_layers=$PRUNE_CONFIG"
    model.cd.teacher_num_layers=36
    trainer.gradient_accumulation_steps=4
    trainer.num_train_epochs=2
    "paths.output_dir=$RUN_OUT"
)
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=2 ++trainer.logging_steps=1 ++trainer.save_strategy=no
            ++trainer.eval_strategy=no ++trainer.warmup_steps=0
            ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids=0-120 ++data.val_dataset.chunk_ids=0-120
            ++trainer.dataloader_num_workers=2
            ++callbacks.ema.liveness_check_at=2 ++callbacks.ema.warmup_steps=1)
    echo "[slurm] SMOKE: 2 optimizer steps"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

echo "[slurm] consistency student: 28/36 active; skipped $PRUNE_MAP"
echo "[slurm] frozen teacher: full 36 layers (runtime asserted)"
echo "[slurm] effective batch: 4 x 2 GPUs x 4 accumulation = 32"
echo "[slurm] epochs: 2; output: $RUN_OUT"
nvidia-smi -L

# train_kd installs KaVaTrainer, which logs the CD bands and saves EMA weights.
srun "$VENV/torchrun" --nproc_per_node 2 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_cd_expert_2cam_nav_lcdrive \
    run_name="cd_pruned28_$(date +%m%d-%H%M)" \
    "${EXTRA[@]}"
