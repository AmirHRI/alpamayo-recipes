#!/bin/bash
#SBATCH --job-name=a1_5_kava_kv
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kv_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# 24 CPUs: PROCS_PER_GPU=3 processes x NUM_WORKERS=3 frame-decode workers, plus the
# three main processes. Two such jobs use 48 of the node's 128.
#SBATCH --cpus-per-task=24
# Host RAM is for the frame-decoding dataloader workers, not the model (that is 19 GiB
# of VRAM). 3 co-located processes need proportionally more. Two jobs x 180G = 360G of
# the node's 503 GiB; note SLURM holds a 4th job PENDING on (Resources) if the total
# would exceed that, which is a memory limit, not a GPU one.
#SBATCH --mem=180G
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
#   # full run on TWO GPUs, 3 co-located shards each (see GPU packing below).
#   # Each shard writes its own files, so there is no merge step.
#   for g in 0 1; do LIMIT=0 TOTAL_GPUS=2 GPU_SLOT=$g PROCS_PER_GPU=3 \
#       sbatch slurm_teacher_kv_lcdrive.sh; done
#
#   # one shard per GPU (the unpacked mode)
#   LIMIT=0 PROCS_PER_GPU=1 NUM_SHARDS=4 SHARD=0 sbatch slurm_teacher_kv_lcdrive.sh
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
WRITE_FULL="${WRITE_FULL:-true}"

# ── GPU packing ─────────────────────────────────────────────────────
# One capture process needs only ~19 GiB of an 80 GiB H100 (almost all of it the 8B
# teacher's weights; batch is 1 and activations are negligible), so reserving a whole
# GPU per shard wastes ~76% of its VRAM. Worse, the bottleneck is *not* saturation:
# 68% of the ~790 ms/sample is autoregressive CoT generation at 35 ms/token, which at
# batch 1 is dominated by per-step launch overhead — measured GPU utilisation swings
# between 7% and 92%. Co-locating several processes on one GPU fills those gaps.
#
# So each SLURM job takes ONE GPU and runs PROCS_PER_GPU shards on it in parallel.
# Shard ids are laid out so co-scheduled jobs never overlap:
#     shard = GPU_SLOT * PROCS_PER_GPU + i,   num_shards = TOTAL_GPUS * PROCS_PER_GPU
#
#   # 2 GPUs x 3 processes = 6 shards, freeing 2 GPUs for other work
#   for g in 0 1; do LIMIT=0 TOTAL_GPUS=2 GPU_SLOT=$g PROCS_PER_GPU=3 \
#       sbatch slurm_teacher_kv_lcdrive.sh; done
#
# PROCS_PER_GPU=4 would be ~76 GiB and risks OOM on the prefill spike; 3 (~57 GiB)
# leaves real headroom. Set PROCS_PER_GPU=1 for the plain one-shard-per-GPU mode, in
# which case NUM_SHARDS/SHARD are honoured directly.
PROCS_PER_GPU="${PROCS_PER_GPU:-3}"
GPU_SLOT="${GPU_SLOT:-0}"
TOTAL_GPUS="${TOTAL_GPUS:-2}"

if [[ "$PROCS_PER_GPU" -gt 1 ]]; then
    NUM_SHARDS=$((TOTAL_GPUS * PROCS_PER_GPU))
else
    NUM_SHARDS="${NUM_SHARDS:-1}"
    SHARD="${SHARD:-0}"
fi
# Frame decode runs in workers, overlapping the GPU. Budget across the co-located
# processes: the cgroup exposes roughly half of --cpus-per-task, so keep
# PROCS_PER_GPU x NUM_WORKERS under that or the workers thrash each other.
NUM_WORKERS="${NUM_WORKERS:-3}"

mkdir -p "$OUT_DIR" "$CACHE_ROOT"
cd "$RECIPE_DIR"

# Both packages must import: alpamayo1_5_distill and the alpamayo1_5_sft it subclasses.
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
# The teacher prefills ~3k tokens and generates; expandable segments keep the
# allocator from fragmenting across the variable-length generate calls, which matters
# more once several processes share one GPU's memory pool.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[slurm] job=$SLURM_JOB_ID node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
echo "[slurm] teacher=$TEACHER importance=$IMPORTANCE mode=$MODE"
echo "[slurm] tier=M${M}_${EVICTION}${LAM} limit=$LIMIT"
echo "[slurm] packing: gpu_slot=$GPU_SLOT/$TOTAL_GPUS procs_per_gpu=$PROCS_PER_GPU -> num_shards=$NUM_SHARDS"
echo "[slurm] cache_root=$CACHE_ROOT"
df -h "$CACHE_ROOT" | tail -1

# Each shard gets its OWN log file rather than being prefixed into the job's stdout.
# Piping through `sed` looked tidier but block-buffers when stdout is a file, so a
# multi-hour run showed zero progress until the buffer filled — and three shards
# interleaving into one file would garble lines anyway.
run_shard() {
    local shard="$1"
    local log="$OUT_DIR/kv_${SLURM_JOB_ID}_shard${shard}.log"
    echo "[slurm]   shard $shard -> $log"
    "$PYTHON" -u -m alpamayo1_5_distill.scripts.generate_teacher_kv \
        config=cache_teacher_kv_lcdrive \
        teacher="$TEACHER" \
        cache_root="$CACHE_ROOT" \
        mode="$MODE" \
        importance_source="$IMPORTANCE" \
        m="$M" lam="$LAM" eviction="$EVICTION" \
        write_full="$WRITE_FULL" \
        num_shards="$NUM_SHARDS" shard="$shard" \
        num_workers="$NUM_WORKERS" \
        limit="$LIMIT" \
        log_texts=2 > "$log" 2>&1
}

pids=()
for i in $(seq 0 $((PROCS_PER_GPU - 1))); do
    if [[ "$PROCS_PER_GPU" -gt 1 ]]; then
        shard=$((GPU_SLOT * PROCS_PER_GPU + i))
    else
        shard="$SHARD"
    fi
    echo "[slurm] launching shard $shard/$NUM_SHARDS"
    run_shard "$shard" &
    pids+=("$!")
    # Stagger: three simultaneous checkpoint loads would spike host RAM and contend
    # on the same safetensors shards.
    sleep 20
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

echo "[slurm] --- cache contents ---"
du -sh "$CACHE_ROOT"/* 2>/dev/null || true
echo "[slurm] all shards on this GPU exited (status=$status)"
exit "$status"
