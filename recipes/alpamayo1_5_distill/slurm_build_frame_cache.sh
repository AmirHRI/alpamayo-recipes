#!/bin/bash
#SBATCH --job-name=a1_5_framecache
#SBATCH --partition=gpu
#SBATCH --nodelist=amhrisvh100b
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/framecache_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/framecache_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# One-time materialisation of the frames the nav 2-camera arm actually consumes.
#
#   sbatch slurm_build_frame_cache.sh              # full build, 24 shards
#   SHARDS=8 sbatch slurm_build_frame_cache.sh     # fewer processes
#   VERIFY=8 sbatch slurm_build_frame_cache.sh     # verify only, builds nothing
#
# WHY A CPU JOB ON A GPU PARTITION: `gpu` is the only partition on this cluster, so the job
# asks for --gpus=0 and takes CPUs only. amhrisvh100b is the idle 48-core node; 200b is left
# free for the training run this cache feeds.
#
# SIZING. Measured on job 20710: the live loader streams 1.64 TiB/epoch to deliver 879,976
# images. This pass reads only the in-manifest clip members -- 832 GiB -- once. At the ~120
# MB/s /temp delivers (already nconnect=8, so this is the server, not the mount) that is ~2 h,
# and it is NFS-bound, not CPU-bound: one shard sustains ~7.7 MB/s, so ~18-24 shards saturate
# the wire and more only adds contention. Output is ~100-130 GiB.
#
# RESTARTABLE. Each clip ZIP is written atomically and skipped when already complete, so a
# resubmit after a timeout or a node failure resumes rather than restarting.

set -euo pipefail

REPO=/home/achahe/alpamayo-recipes
VENV="$REPO/recipes/alpamayo1_5_sft/a1_5_sft/bin/python3"
SCRIPT="$REPO/recipes/alpamayo1_5_distill/scripts/build_frame_cache.py"
OUT="${OUT:-/temp/achahe/physical_ai_av/framecache_nav2cam_1080p}"
SHARDS="${SHARDS:-24}"
CRF="${CRF:-18}"
# ⚠️ Must be passed to EVERY build_frame_cache.py call below. The script defaults to "1,3",
# so omitting it here would quietly write a 2-camera cache into a directory named for 4 --
# and the reader only rejects a mismatch against what the INDEX records, which would then
# also say 2. The camera set is part of the cache identity; keep it in the OUT name too.
CAMERAS="${CAMERAS:-1,3}"
VERIFY="${VERIFY:-0}"

[[ -x "$VENV" ]] || { echo "[cache] no interpreter at $VENV" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "[cache] no builder at $SCRIPT" >&2; exit 1; }

mkdir -p "$OUT"
echo "[cache] job=${SLURM_JOB_ID:-none} node=$(hostname) shards=$SHARDS crf=$CRF out=$OUT"
df -h /temp | tail -1 | sed 's/^/[cache] /'

# The memo transparency check and the codec round trip are cheap and gate everything else:
# a frame-selection or codec regression would be invisible in the trained model, so it is
# proven on real anchors before 100+ GiB is written.
echo "[cache] --- verify ---"
"$VENV" "$SCRIPT" --out "$OUT" --crf "$CRF" --cameras "$CAMERAS" --verify "${VERIFY_N:-8}" --verify-only \
    2>&1 | grep -vE "FutureWarning|import pynvml"

if [[ "$VERIFY" != "0" ]]; then
    echo "[cache] VERIFY=1 requested; not building"
    exit 0
fi

echo "[cache] --- build: $SHARDS shards ---"
pids=()
for ((s = 0; s < SHARDS; s++)); do
    "$VENV" "$SCRIPT" \
        --out "$OUT" \
        --crf "$CRF" \
        --cameras "$CAMERAS" \
        --shard "$s" \
        --num-shards "$SHARDS" \
        --log-every 50 \
        > "$OUT/.shard_${s}.log" 2>&1 &
    pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[cache] shard $i FAILED (see $OUT/.shard_${i}.log)" >&2
        status=1
    fi
done

echo "[cache] --- shard tails ---"
for ((s = 0; s < SHARDS; s++)); do
    tail -1 "$OUT/.shard_${s}.log" 2>/dev/null | sed "s/^/[shard $s] /"
done

echo "[cache] total size: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
echo "[cache] clip zips:  $(find "$OUT" -name '*.zip' | wc -l)"

# The reader refuses to start without _index.json, and --finalize only writes one when every
# manifest anchor is present -- so this doubles as the completeness gate. A shard that died
# leaves the index unwritten rather than letting training start on a partial cache.
if [[ "$status" == "0" ]]; then
    echo "[cache] --- finalize ---"
    if "$VENV" "$SCRIPT" --out "$OUT" --crf "$CRF" --cameras "$CAMERAS" --finalize 2>&1 \
        | grep -vE "FutureWarning|import pynvml"; then
        echo "[cache] cache is complete and indexed; train with"
        echo "        ARM=nav4bspan2camallfc sbatch slurm_train_kd.sh"
    else
        echo "[cache] finalize reported missing clips; resubmit this script to fill the gaps" >&2
        status=1
    fi
fi
exit "$status"
