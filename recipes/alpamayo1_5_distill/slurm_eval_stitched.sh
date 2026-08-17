#!/bin/bash
#SBATCH --job-name=a1_5_stitch_eval
#SBATCH --partition=debug
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stitcheval_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stitcheval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Student VLM -> TEACHER's action expert. THE eval the KV objective exists for.
#
#   ARM=teacher sbatch slurm_eval_stitched.sh   # ceiling -- RUN THIS, nothing else has scale
#   ARM=ce      sbatch slurm_eval_stitched.sh   # floor: cache never trained toward teacher
#   ARM=kd      sbatch slurm_eval_stitched.sh   # floor: logit-KD only, still no K/V objective
#   ARM=kv      sbatch slurm_eval_stitched.sh   # the arm L_KV was supposed to make work
#
# ⚠️ These numbers are NOT comparable to the token-head baselines (10B no-CoT ade 1.2111 /
# min_ade 0.6413). Different head. The only valid ceiling is ARM=teacher through THIS script.
# Smoke at n=12 for orientation: teacher min_ade 0.738, kv student min_ade 5.92.
#
# ⚠️ attn_implementation=sdpa is REQUIRED, not a preference. Under flash_attention_2 the
# expert dies with a device-side assert -- an out-of-bounds gather in flash-attn's unpadding
# path, which is the exact fragility alpamayo_r1.py:248-262 documents: that path hard-requires
# a 2D mask and does nonzero()/sum(-1) on whatever it is handed. sdpa takes the same mask
# safely. Do not "clean this up" back to flash_attention_2.

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT_DIR=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
TEACHER=/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format
COSMOS=/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9

ARM="${ARM:?set ARM=teacher|ce|kd|kv|cekv|kvonly|<arm>_eN}"
CKPT="${CKPT:-}"   # e.g. checkpoint-3196; default is the newest
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"
BS="${BS:-4}"

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$((29840 + SLURM_JOB_ID % 20000))

EXTRA=()
if [[ "$ARM" == "teacher" ]]; then
    # The teacher's own tower is present locally as CONFIG ONLY -- its weights live in the
    # Alpamayo checkpoint under vlm.*. from_teacher builds the skeleton from that config and
    # loads vlm.* from the checkpoint; from_stitch would try to pull real weights from the
    # Cosmos snapshot and die with "no file named model.safetensors".
    CKPT="$TEACHER"
    EXTRA+=(++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_teacher
            ++model.vlm_name_or_path="$COSMOS")
else
    if [[ -n "${CKPT:-}" ]]; then
        # Explicit checkpoint. Needed whenever a run is STILL TRAINING: the newest-checkpoint
        # default would silently pick up a later epoch mid-sweep, so two "epoch 2" numbers
        # could come from different weights.
        CKPT="$OUT_DIR/output_kd_4b_${ARM}_lcdrive/$CKPT"
        [[ -d "$CKPT" ]] || { echo "[slurm] no such checkpoint: $CKPT" >&2; exit 1; }
    else
        CKPT=$(ls -d "$OUT_DIR/output_kd_4b_${ARM}_lcdrive"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        [[ -z "$CKPT" ]] && { echo "[slurm] no checkpoint for ARM=$ARM" >&2; exit 1; }
    fi
fi
TAG="stitch_4b_${ARM}_$(basename "$CKPT")"
echo "[slurm] ARM=$ARM ckpt=$CKPT -> $OUT_DIR/$TAG.json"

srun "$VENV/torchrun" --nproc_per_node 1 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_eval_stitched_4b_lcdrive \
    ++model.attn_implementation=sdpa \
    ++evaluate.eval_ckpt="$CKPT" \
    ++evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    ++evaluate.per_clip_output="$OUT_DIR/$TAG.json" \
    ++trainer.per_device_eval_batch_size="$BS" \
    paths.output_dir="$OUT_DIR/$TAG" \
    "${EXTRA[@]}" \
    2>&1 | tee "$OUT_DIR/$TAG.log"

echo; echo "================ TRIPWIRES ================"
LOG="$OUT_DIR/$TAG.log"
grep -a "stitch\] loaded" "$LOG" | tail -1 || echo "!! expert never loaded"
grep -a "Loaded .* VLM tensors" "$LOG" | tail -1
grep -aE "val/count|kept [0-9]+/[0-9]+ clips" "$LOG" | tail -2
BAD=$(grep -ac "Invalid token ids\|not equal to the expected" "$LOG" || true)
"$VENV/python" - "$OUT_DIR/$TAG.json" "$BAD" <<'PY'
import json, sys, statistics as st
recs = json.load(open(sys.argv[1])); n=len(recs)
print(f"per-clip records: {n}" + ("" if n==1000 else "   <-- ⚠️ EXPECTED 1000"))
bad=int(sys.argv[2])
print(f"malformed warnings: {bad}" + (f" = {100*bad/n:.1f}% of clips" if n else ""))
for m in ("ade","min_ade"):
    v=[r[m] for r in recs if m in r]
    if v: print(f"  {m:<8} {st.mean(v):.4f}  (se {st.stdev(v)/len(v)**0.5:.4f})")
eq=sum(1 for r in recs if abs(r['ade']-r['min_ade'])<1e-9)
print(f"  all-6-samples-identical: {eq}/{n} = {100*eq/n:.1f}%   (mode-collapse tripwire)")
print("  compare ONLY against ARM=teacher from this same script -- token-head numbers do not apply")
PY
