#!/bin/bash
#SBATCH --job-name=a1_5_kava_compress
#SBATCH --partition=debug
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/compress_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/compress_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Derive a new compressed tier from an existing `full/` tier. **No teacher, no GPU** —
# every (M, lam, eviction) point is a different eviction of the same teacher forward,
# so re-tuning any of them costs minutes of CPU here instead of a ~3 h rebuild.
#
#   M=8  sbatch slurm_compress_teacher_kv.sh          # the arm where R-KV engages
#   M=16 EVICTION=cosine sbatch slurm_compress_teacher_kv.sh   # lambda = 0 ablation
#   M=16 EVICTION=attn   sbatch slurm_compress_teacher_kv.sh   # lambda = 1 ablation
#   M=16 EVICTION=crop   sbatch slurm_compress_teacher_kv.sh   # naive first-M baseline
#
# ⚠️ The IMPORTANCE score is fixed at cache-build time and cannot be re-derived here:
# redundancy is a pure function of the stored keys, but importance needs the answer-side
# attention from a live forward. A tier built from a `vlm_post_cot` cache stays
# vlm_post_cot no matter what M is. Switching to `expert` means rebuilding `full/`.

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
PYTHON=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/python
OUT_DIR=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training

CACHE_ROOT="${CACHE_ROOT:-$OUT_DIR/teacher_kv_lcdrive}"
M="${M:-8}"
LAM="${LAM:-0.1}"
EVICTION="${EVICTION:-rkv}"
# I/O-bound (each entry is a ~1.8 MB read plus a ~1.2 MB write), so more workers than
# the eviction math alone would need. Keep under the cgroup's usable CPUs.
NPROC="${NPROC:-8}"

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes

echo "[slurm] job=$SLURM_JOB_ID compress -> M=$M lam=$LAM eviction=$EVICTION"
echo "[slurm] cache_root=$CACHE_ROOT  nproc=$NPROC"

pids=()
for i in $(seq 0 $((NPROC - 1))); do
    "$PYTHON" -u -m alpamayo1_5_distill.scripts.compress_teacher_kv \
        cache_root="$CACHE_ROOT" \
        m="$M" lam="$LAM" eviction="$EVICTION" \
        num_shards="$NPROC" shard="$i" \
        > "$OUT_DIR/compress_${SLURM_JOB_ID}_shard${i}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done

echo "[slurm] done (status=$status)"
du -sh "$CACHE_ROOT"/* 2>/dev/null || true
