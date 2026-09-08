#!/bin/bash
# ARM A' -- endpoint distillation + a GT damper, one run per weight.
#   GT_W=0.1 GPUS=0,1 PORT=31801 ./run_cd_eos2b_endpoint_gt.sh
set -euo pipefail
GT_W="${GT_W:?set GT_W (e.g. 0.1)}"
TAG=$(echo "$GT_W" | tr -d '.')
GPUS="${GPUS:-0,1}"; PORT="${PORT:-31801}"
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
EOS_CKPT=$OUT/output_eos_2b_nav_e3_clean_maskfix_e2_lcdrive/checkpoint-4168
CACHE_ROOT=$OUT/teacher_action_rollouts_full10b_2cam_nav_k6_m10
RUN_OUT=$OUT/output_cd_eos2b_endpoint_gt${TAG}_nav_e2_bs32_20260830
unset PRUNE_EXPERT_LAYERS
[[ -d "$CACHE_ROOT/action_rollouts" ]] || { echo "[run] missing cache" >&2; exit 1; }
[[ $(find "$CACHE_ROOT/action_rollouts" -name '*.safetensors' | wc -l) -eq 50000 ]] || { echo "[run] incomplete cache" >&2; exit 1; }
[[ -d "$RUN_OUT" && -n "$(find "$RUN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]] && { echo "[run] non-empty $RUN_OUT" >&2; exit 1; }
cd /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KAVA_GRAD_PROBE_STEPS=0
export CUDA_VISIBLE_DEVICES="$GPUS"
echo "[run] GT_W=$GT_W gpus=$GPUS -> $RUN_OUT"
exec "$VENV/torchrun" --nproc_per_node 2 --master_port "$PORT" \
  -m alpamayo1_5_distill.train_kd \
  --config-path pkg://alpamayo1_5_distill/configs \
  --config-name sft_cd_eos_2b_endpoint_gt${TAG}_nav_lcdrive \
  "run_name=cd_eos2b_endpoint_gt${TAG}_$(date +%m%d-%H%M)"
