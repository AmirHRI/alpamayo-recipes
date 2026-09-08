#!/bin/bash
#SBATCH --job-name=a1_5_fulltraj
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/fulltraj_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/fulltraj_%j.err
#
# Exact full-Alpamayo trajectory cache.
#
# Smoke (8 scenes):
#   sbatch slurm_cache_full_teacher_trajectories.sh
#
# Full run over four independent GPU jobs:
#   for s in 0 1 2 3; do LIMIT=0 NUM_SHARDS=4 SHARD=$s \
#       sbatch slurm_cache_full_teacher_trajectories.sh; done
#
# Every scene is an atomic file and its seed depends only on (clip,t0), so jobs
# may be resumed or re-sharded without changing cached noise or overwriting data.
set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
CACHE_ROOT="${CACHE_ROOT:-$OUT/teacher_action_rollouts_full10b_2cam_nav_k6_m10}"
CONFIG="${CONFIG:-cache_full_teacher_trajectories_2cam_nav_lcdrive}"
LIMIT="${LIMIT:-8}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD="${SHARD:-0}"
NUM_NOISE="${NUM_NOISE:-6}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1234}"

mkdir -p "$CACHE_ROOT"
cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PRUNE_EXPERT_LAYERS

echo "[slurm] full Alpamayo 1.5: 8B VLM + 36-layer action expert"
echo "[slurm] shard=$SHARD/$NUM_SHARDS limit=$LIMIT K=$NUM_NOISE M=$NUM_STEPS"
echo "[slurm] cache_root=$CACHE_ROOT"
nvidia-smi -L

"$VENV/python" -u -m alpamayo1_5_distill.scripts.generate_teacher_trajectories \
    config="$CONFIG" \
    cache_root="$CACHE_ROOT" \
    num_steps="$NUM_STEPS" num_noise="$NUM_NOISE" seed="$SEED" \
    num_shards="$NUM_SHARDS" shard="$SHARD" \
    num_workers="$NUM_WORKERS" limit="$LIMIT" log_every=25

find "$CACHE_ROOT/action_rollouts" -type f -name '*.safetensors' | wc -l
du -sh "$CACHE_ROOT"
