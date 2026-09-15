#!/bin/bash
#SBATCH --job-name=a1_5_cm4b_2to1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cm4b_2to1_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cm4b_2to1_%j.err
set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
CONFIG=sft_cd_eos_4b_consistency2to1_2cam_nav_lcdrive
EOS_CKPT="${EOS_CKPT:-$OUT/output_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache/checkpoint-6876}"
SMOKE="${SMOKE:-0}"
RUN_OUT="${OUTPUT_DIR:-$OUT/output_cd_eos4b_consistency2to1_2cam_nav}"
if [[ "$SMOKE" == "1" ]]; then
    RUN_OUT="${OUTPUT_DIR:-${RUN_OUT}_smoke_${SLURM_JOB_ID:-$$}}"
fi
if [[ -n "${RESUME:-}" ]]; then
    echo "[slurm] exact EMA/online resume is not supported; use a fresh output directory" >&2
    exit 1
fi
if [[ ! -f "$EOS_CKPT/model.safetensors.index.json" || ! -f "$EOS_CKPT/config.json" ]]; then
    echo "[slurm] frozen teacher requires a sharded EoS checkpoint and config: $EOS_CKPT" >&2
    exit 1
fi
if [[ -d "$RUN_OUT" && -n "$(find "$RUN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[slurm] refusing non-empty output directory: $RUN_OUT" >&2
    exit 1
fi
unset PRUNE_EXPERT_LAYERS
cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export HF_HOME=/temp/achahe/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CD_VERIFY_SAVE=1
export KAVA_GRAD_PROBE_STEPS=0
MASTER_PORT=$((30420 + ${SLURM_JOB_ID:-$$} % 20000))
EXTRA=(
    "model.eos_checkpoint_path=$EOS_CKPT"
    "paths.output_dir=$RUN_OUT"
    "trainer.dataloader_num_workers=${WORKERS:-4}"
)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a USER_EXTRA <<< "$EXTRA_ARGS"
    EXTRA+=("${USER_EXTRA[@]}")
fi
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=2 trainer.logging_steps=1 trainer.save_strategy=steps
            trainer.save_steps=2 trainer.eval_strategy=no trainer.warmup_steps=0
            trainer.report_to=none callbacks.ema.liveness_check_at=2
            data.train_dataset.chunk_ids=0-120 trainer.dataloader_num_workers=2)
fi
echo "[slurm] frozen two-step EoS teacher + online/EMA student <- $EOS_CKPT"
echo "[slurm] pure two-rung consistency; no GT loss or trajectory cache; deploy NFE=1"
echo "[slurm] default effective batch: 4 x 4 GPUs x 2 accumulation = 32; output: $RUN_OUT"
nvidia-smi -L
srun "$VENV/torchrun" --nproc_per_node 4 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "$CONFIG" "${EXTRA[@]}"