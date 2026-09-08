#!/bin/bash
#SBATCH --job-name=a1_5_cd_ftcache
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdftcache_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdftcache_%j.err
#
# Two-epoch CD of the epoch-3 2B VLM + 28-layer EoS expert against cached
# full Alpamayo-1.5 rollouts. Effective batch = 4 x 2 GPUs x 4 accum = 32.
#
#   sbatch slurm_train_cd_eos2b_fullteacher_cache.sh
#   SMOKE=1 sbatch slurm_train_cd_eos2b_fullteacher_cache.sh
set -euo pipefail

SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
EOS_CKPT="${EOS_CKPT:-$OUT/output_eos_2b_nav_e3_clean_maskfix_e2_lcdrive/checkpoint-4168}"
CACHE_ROOT="${CACHE_ROOT:-$OUT/teacher_action_rollouts_full10b_2cam_nav_k6_m10}"
RUN_OUT="${OUTPUT_DIR:-$OUT/output_cd_eos2b_fullteacher_cache_x0gtw0.1_nav_e2_bs32_20260828}"
EXPECTED_CACHE_ENTRIES="${EXPECTED_CACHE_ENTRIES:-50000}"

unset PRUNE_EXPERT_LAYERS
if [[ ! -f "$EOS_CKPT/model.safetensors.index.json" && ! -f "$EOS_CKPT/model.safetensors" ]]; then
    echo "[slurm] missing EoS checkpoint weights: $EOS_CKPT" >&2
    exit 1
fi
if [[ ! -d "$CACHE_ROOT/action_rollouts" ]]; then
    echo "[slurm] missing full-teacher trajectory cache: $CACHE_ROOT/action_rollouts" >&2
    exit 1
fi
CACHE_COUNT=$(find "$CACHE_ROOT/action_rollouts" -type f -name '*.safetensors' | wc -l)
if [[ "$CACHE_COUNT" -lt 1 ]]; then
    echo "[slurm] full-teacher trajectory cache is empty: $CACHE_ROOT" >&2
    exit 1
fi
if [[ "$SMOKE" != "1" && "$CACHE_COUNT" -ne "$EXPECTED_CACHE_ENTRIES" ]]; then
    echo "[slurm] incomplete full-teacher trajectory cache: " \
         "$CACHE_COUNT/$EXPECTED_CACHE_ENTRIES entries" >&2
    echo "[slurm] refusing to start training; resume failed cache shards first" >&2
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
MASTER_PORT=$((30620 + SLURM_JOB_ID % 20000))

EXTRA=(
    "model.eos_checkpoint_path=$EOS_CKPT"
    "data.train_dataset.teacher_trajectory_cache_root=$CACHE_ROOT"
    "paths.output_dir=$RUN_OUT"
)
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=2 ++trainer.logging_steps=1 ++trainer.save_strategy=no
            ++trainer.eval_strategy=no ++trainer.warmup_steps=0
            ++trainer.gradient_accumulation_steps=1
            ++trainer.dataloader_num_workers=2
            ++data.train_dataset.teacher_trajectory_cached_only=true
            ++callbacks.ema.liveness_check_at=2 ++callbacks.ema.warmup_steps=1)
    echo "[slurm] SMOKE: train only over entries present in the partial cache"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=(${EXTRA_ARGS})

echo "[slurm] student init: $EOS_CKPT"
echo "[slurm] teacher: cached full Alpamayo 1.5 rollouts at $CACHE_ROOT"
echo "[slurm] no online teacher model; effective batch=32 for the full run"
nvidia-smi -L

srun "$VENV/torchrun" --nproc_per_node 2 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_cd_eos_2b_fullteacher_cache_nav_lcdrive \
    "run_name=cd_eos2b_fullteacher_$(date +%m%d-%H%M)" \
    "${EXTRA[@]}"
