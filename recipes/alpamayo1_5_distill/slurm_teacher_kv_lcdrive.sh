#!/bin/bash
#SBATCH --job-name=a1_5_kava_kv
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kv_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Build the teacher KV cache for KAVA distillation over the LCDrive train split:
# runs the frozen Alpamayo-1.5-10B once per clip and records the pre-RoPE per-layer
# CoT K/V, both R-KV scores, the <traj_future_start> hidden, and the CoT text.
#
#   # smoke test — 8 clips, ~2 min, verifies the whole pipeline end to end
#   sbatch slurm_teacher_kv_lcdrive.sh
#
#   # the real pilot from the README (step K2): fixes the N_C histogram and the disk bill
#   LIMIT=200 sbatch slurm_teacher_kv_lcdrive.sh
#
#   # full run, sharded across 4 GPUs (each shard writes its own files, no merge step)
#   LIMIT=0 NUM_SHARDS=4 SHARD=0 sbatch slurm_teacher_kv_lcdrive.sh   # ... and 1,2,3
#
#   # score CoT tokens by the ACTION EXPERT's cross-attention instead of the
#   # post-CoT text (the faithful reader; needs the expert-carrying teacher)
#   IMPORTANCE=expert TEACHER=teacher_ar1_5_10b_expert \
#       CACHE_ROOT=.../teacher_kv_lcdrive_expert sbatch slurm_teacher_kv_lcdrive.sh
#
# ⚠️ CACHE_ROOT must live on /data. /home is quota-capped at 100 GB and the full
# cache is O(100s of GB); writes there fail with EDQUOT mid-run.

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
PYTHON=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/python
OUT_DIR=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training

LIMIT="${LIMIT:-8}"                  # 0 = the whole shard
CACHE_ROOT="${CACHE_ROOT:-$OUT_DIR/teacher_kv_lcdrive}"
TEACHER="${TEACHER:-teacher_ar1_5_10b}"
IMPORTANCE="${IMPORTANCE:-vlm_post_cot}"
MODE="${MODE:-generate}"             # generate | teacher_force (LCDrive has no GT CoT)
M="${M:-16}"
LAM="${LAM:-0.1}"
EVICTION="${EVICTION:-rkv}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD="${SHARD:-0}"
# Frame decode runs in workers, overlapping the GPU. Keep this under the cgroup's
# visible CPU count or torch warns and the workers contend: --cpus-per-task=10 leaves
# ~5 usable, so 5 is the ceiling that does not thrash.
NUM_WORKERS="${NUM_WORKERS:-5}"
WRITE_FULL="${WRITE_FULL:-true}"

mkdir -p "$OUT_DIR" "$CACHE_ROOT"
cd "$RECIPE_DIR"

# Both packages must import: alpamayo1_5_distill and the alpamayo1_5_sft it subclasses.
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
# The teacher prefills ~3k tokens and generates; expandable segments keep the
# allocator from fragmenting across the variable-length generate calls.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[slurm] job=$SLURM_JOB_ID node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
echo "[slurm] teacher=$TEACHER importance=$IMPORTANCE mode=$MODE"
echo "[slurm] tier=M${M}_${EVICTION}${LAM} shard=$SHARD/$NUM_SHARDS limit=$LIMIT"
echo "[slurm] cache_root=$CACHE_ROOT"
df -h "$CACHE_ROOT" | tail -1

srun "$PYTHON" -m alpamayo1_5_distill.scripts.generate_teacher_kv \
    config=cache_teacher_kv_lcdrive \
    teacher="$TEACHER" \
    cache_root="$CACHE_ROOT" \
    mode="$MODE" \
    importance_source="$IMPORTANCE" \
    m="$M" lam="$LAM" eviction="$EVICTION" \
    write_full="$WRITE_FULL" \
    num_shards="$NUM_SHARDS" shard="$SHARD" \
    num_workers="$NUM_WORKERS" \
    limit="$LIMIT" \
    log_texts=5

echo "[slurm] --- cache contents ---"
du -sh "$CACHE_ROOT"/* 2>/dev/null || true
echo "[slurm] --- first CoTs recorded ---"
head -3 "$CACHE_ROOT/cot_text.shard${SHARD}.jsonl" 2>/dev/null || echo "(no cot_text yet)"
