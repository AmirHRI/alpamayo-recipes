#!/bin/bash
# ARM A: endpoint distillation of the EoS-2B expert against the cached full-10B rollouts.
#
#   L = || x_hi + tau_hi * v_theta(x_hi, tau_hi) - states[k, M] ||^2
#
# Direct launch on GPUs 0 and 1 (no SLURM). Effective batch = 4 x 2 GPUs x 4 accum = 32.
#
#   ./run_cd_eos2b_endpoint.sh              full 2-epoch run
#   SMOKE=1 ./run_cd_eos2b_endpoint.sh      2 optimizer steps, no save
set -euo pipefail

SMOKE="${SMOKE:-0}"
GPUS="${GPUS:-0,1}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
EOS_CKPT="${EOS_CKPT:-$OUT/output_eos_2b_nav_e3_clean_maskfix_e2_lcdrive/checkpoint-4168}"
CACHE_ROOT="${CACHE_ROOT:-$OUT/teacher_action_rollouts_full10b_2cam_nav_k6_m10}"
RUN_OUT="${OUTPUT_DIR:-$OUT/output_cd_eos2b_fullteacher_endpoint_nav_e2_bs32_20260829}"
EXPECTED_CACHE_ENTRIES="${EXPECTED_CACHE_ENTRIES:-50000}"

# ⚠️ A stray export would bypass 8 of the student's 28 expert layers and train the
# ablation under this run's name -- and `student_prune_layers: null` would then abort.
unset PRUNE_EXPERT_LAYERS
if [[ ! -f "$EOS_CKPT/model.safetensors.index.json" && ! -f "$EOS_CKPT/model.safetensors" ]]; then
    echo "[run] missing EoS checkpoint weights: $EOS_CKPT" >&2
    exit 1
fi
if [[ ! -d "$CACHE_ROOT/action_rollouts" ]]; then
    echo "[run] missing full-teacher trajectory cache: $CACHE_ROOT/action_rollouts" >&2
    exit 1
fi
CACHE_COUNT=$(find "$CACHE_ROOT/action_rollouts" -type f -name '*.safetensors' | wc -l)
if [[ "$SMOKE" != "1" && "$CACHE_COUNT" -ne "$EXPECTED_CACHE_ENTRIES" ]]; then
    echo "[run] incomplete cache: $CACHE_COUNT/$EXPECTED_CACHE_ENTRIES entries" >&2
    exit 1
fi
if [[ "$SMOKE" != "1" && -d "$RUN_OUT" && -n "$(find "$RUN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[run] refusing non-empty output directory: $RUN_OUT" >&2
    exit 1
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAVA_GRAD_PROBE_STEPS="${KAVA_GRAD_PROBE_STEPS:-0}"
export CUDA_VISIBLE_DEVICES="$GPUS"
MASTER_PORT="${MASTER_PORT:-30717}"

EXTRA=(
    "model.eos_checkpoint_path=$EOS_CKPT"
    "data.train_dataset.teacher_trajectory_cache_root=$CACHE_ROOT"
    "paths.output_dir=$RUN_OUT"
)
if [[ "$SMOKE" == "1" ]]; then
    RUN_OUT="$OUT/smoke_cd_eos2b_endpoint"
    rm -rf "$RUN_OUT"
    EXTRA=("model.eos_checkpoint_path=$EOS_CKPT"
           "data.train_dataset.teacher_trajectory_cache_root=$CACHE_ROOT"
           "paths.output_dir=$RUN_OUT"
           ++trainer.max_steps=2 ++trainer.logging_steps=1 ++trainer.save_strategy=no
           ++trainer.eval_strategy=no ++trainer.warmup_steps=0
           ++trainer.gradient_accumulation_steps=1
           ++trainer.dataloader_num_workers=2
           ++trainer.report_to=none
           ++data.train_dataset.teacher_trajectory_cached_only=true)
    echo "[run] SMOKE: 2 steps over entries present in the cache"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=(${EXTRA_ARGS})

echo "[run] student init : $EOS_CKPT"
echo "[run] target       : cached full-10B rollout ENDPOINTS at $CACHE_ROOT"
echo "[run] objective    : cd_weight=0, x0_gt_weight=1, x0_source=teacher (no EMA)"
echo "[run] GPUs         : $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

exec "$VENV/torchrun" --nproc_per_node 2 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_cd_eos_2b_fullteacher_endpoint_nav_lcdrive \
    "run_name=cd_eos2b_endpoint_$(date +%m%d-%H%M)" \
    "${EXTRA[@]}"
