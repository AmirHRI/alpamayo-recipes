#!/bin/bash
#SBATCH --job-name=a1_5_layerimp
#SBATCH --partition=debug
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/layerimp_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/layerimp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com
# Causal per-layer K/V swap: which VLM layers does the action expert actually depend on?
# See scripts/layer_importance.py. ~74 expert rollouts per clip, reusing one VLM forward.
set -euo pipefail
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
CKPT="${CKPT:-$OUT/output_kd_4b_kvonly_e3_lcdrive/checkpoint-3196}"
N="${N:-100}"
# fix-only by default: fix_gain is the quantity you weight L_KV/L_block by; break_cost was
# only ever the cross-check, and dropping it halves the per-clip cost.
DIRECTIONS="${DIRECTIONS:-[fix]}"
cd /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin/python \
  -m alpamayo1_5_distill.scripts.layer_importance \
  --config-path pkg://alpamayo1_5_distill/configs --config-name sft_eval_stitched_4b_lcdrive \
  ++model.attn_implementation=sdpa ++evaluate.eval_ckpt="$CKPT" \
  ++probe.n_clips="$N" ++probe.directions="$DIRECTIONS" ++probe.out="$OUT/layer_importance_fix_n${N}.json" \
  paths.output_dir="$OUT/layerprobe_n${N}"
