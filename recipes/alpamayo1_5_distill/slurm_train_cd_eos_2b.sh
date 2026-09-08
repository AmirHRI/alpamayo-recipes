#!/bin/bash
#SBATCH --job-name=a1_5_cd_eos2b
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeos2b_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeos2b_%j.err
#
# Two-epoch CD initialized from the 28-layer EoS checkpoint. The student VLM is frozen.
# Effective batch = 4 samples/rank x 2 ranks x 4 accumulation = 32.
#
#   sbatch slurm_train_cd_eos_2b.sh
#   SMOKE=1 sbatch slurm_train_cd_eos_2b.sh
set -euo pipefail

SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
EOS_CKPT="${EOS_CKPT:-$OUT/output_eos_2b_nav_e3_clean_maskfix_e2_lcdrive/checkpoint-4168}"
RUN_OUT="${OUTPUT_DIR:-$OUT/output_cd_eos2b_nav_e2_bs32_20260827}"

# This is already a compact, cache-aligned 28-layer expert. Identity-slot pruning is wrong.
unset PRUNE_EXPERT_LAYERS

if [[ ! -f "$EOS_CKPT/model.safetensors.index.json" && ! -f "$EOS_CKPT/model.safetensors" ]]; then
    echo "[slurm] missing EoS checkpoint weights: $EOS_CKPT" >&2
    exit 1
fi
if [[ -d "$RUN_OUT" && -n "$(find "$RUN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[slurm] refusing non-empty output directory: $RUN_OUT" >&2
    echo "[slurm] set OUTPUT_DIR to a fresh path" >&2
    exit 1
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAVA_GRAD_PROBE_STEPS="${KAVA_GRAD_PROBE_STEPS:-0}"
MASTER_PORT=$((30420 + SLURM_JOB_ID % 20000))

EXTRA=(
    "model.eos_checkpoint_path=$EOS_CKPT"
    "paths.output_dir=$RUN_OUT"
)
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=2 ++trainer.logging_steps=1 ++trainer.save_strategy=no
            ++trainer.eval_strategy=no ++trainer.warmup_steps=0
            ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids=0-120
            ++trainer.dataloader_num_workers=2)
    # ⚠️ The ema overrides apply ONLY if the chosen config actually declares that callback.
    # sft_base ships `callbacks: {}` and train_hf.py instantiates whatever is under it, so a
    # `++callbacks.ema.*` force-add on a config WITHOUT an ema block creates a bare dict with
    # no _target_ -- hydra returns the dict unchanged and the trainer dies on
    # "'dict' object has no attribute 'on_init_end'", ~2 minutes in and nowhere near the real
    # cause. sft_cd_eos_2b_nav_lcdrive declares one; the gt* endpoint family does not.
    if grep -qE "^[[:space:]]+ema:" "$RECIPE_DIR/configs/${CONFIG:-sft_cd_eos_2b_nav_lcdrive}.yaml" 2>/dev/null; then
        EXTRA+=(++callbacks.ema.liveness_check_at=2 ++callbacks.ema.warmup_steps=1)
    else
        echo "[slurm] SMOKE: config declares no callbacks.ema, skipping its overrides"
    fi
    echo "[slurm] SMOKE: 2 optimizer steps"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=(${EXTRA_ARGS})

echo "[slurm] online initialization: $EOS_CKPT"
echo "[slurm] frozen teacher:       $EOS_CKPT"
echo "[slurm] frozen VLM + online/teacher expert: 2B student cache + 28/28 layers"
echo "[slurm] effective batch: 4 x 2 GPUs x 4 accumulation = 32"
echo "[slurm] epochs: 2; output: $RUN_OUT"
nvidia-smi -L

srun "$VENV/torchrun" --nproc_per_node 2 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "${CONFIG:-sft_cd_eos_2b_nav_lcdrive}" \
    "run_name=cd_eos2b_$(date +%m%d-%H%M)" \
    "${EXTRA[@]}"
