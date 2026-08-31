#!/bin/bash
#SBATCH --job-name=a1_5_eos_eval
#SBATCH --partition=gpu
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/eoseval_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/eoseval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# EXPERT-ON-STUDENT read-out: nav-conditioned 2B student VLM + the expert fine-tuned on its
# OWN cache, scored on the event-anchored 1k nav val subset.
#
#   MAX_EVAL_STEPS=5 sbatch slurm_eval_eos.sh        # smoke -- ALWAYS run this first
#   sbatch slurm_eval_eos.sh                         # full 1k, 10 denoising steps
#   NFE=2 sbatch slurm_eval_eos.sh                   # full 1k, 2 denoising steps
#   CKPT=checkpoint-4689 sbatch slurm_eval_eos.sh    # an earlier epoch
#   ARM=control sbatch slurm_eval_eos.sh             # SAME student, TEACHER's expert
#
# ⚠️ RUN ARM=control TOO, or the number has no scale. `control` is the identical student and
# identical data with the untouched 10B expert -- i.e. the arm the EOS training was supposed
# to improve on (min_ade 2.6216 when it was measured). The 4B precedent for this intervention
# was -0.196 min_ade; anything much larger, in either direction, is a bug before it is a
# result.
#
# ⚠️ These numbers are event-anchored + nav-conditioned. They are NOT comparable to the
# default-keyframe rows (that distribution is ~0.044 min_ade easier) and NOT comparable to
# trajectory-TOKEN-head numbers at all -- different head.

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT_DIR=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
# MODEL selects which EOS run to score. The 2B arm stays the default so every previously
# recorded invocation of this script keeps its exact meaning; 4b is opt-in.
# ⚠️ EOS_DIR, CONFIG and the TAG prefix must move TOGETHER. Mixing a 4B checkpoint into the
# 2B config loads a 28-layer expert config against 36-layer weights, and reusing the 2B TAG
# would overwrite the 2B per-clip JSON with 4B numbers under a 2B name. That is why MODEL is
# a single switch rather than three independent env vars.
MODEL="${MODEL:-2b}"
case "$MODEL" in
    2b) EOS_DIR="$OUT_DIR/output_eos_2b_nav_lcdrive"
        CONFIG=sft_eval_eos_2b_nav_lcdrive
        TAG_PREFIX=eos_2b_nav ;;
    4b) EOS_DIR="$OUT_DIR/output_eos_4b_2cam_nav_lcdrive"
        CONFIG=sft_eval_eos_4b_2cam_nav_lcdrive
        TAG_PREFIX=eos_4b_2cam_nav ;;
    *)  echo "[slurm] unknown MODEL=$MODEL (want 2b|4b)" >&2; exit 1 ;;
esac
# Escape hatches for one-off run dirs; they default to whatever MODEL selected. Overriding
# EOS_DIR alone is fine (same architecture, different run); overriding it across
# architectures without also setting CONFIG/TAG_BASE is the failure described above.
EOS_DIR="${EOS_DIR_OVERRIDE:-$EOS_DIR}"
TEACHER=/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format

ARM="${ARM:-eos}"                       # eos | control
CKPT="${CKPT:-}"                        # e.g. checkpoint-4689; default = newest
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"
BS="${BS:-4}"
# NFE = euler denoising steps for the action expert (FlowMatching `inference_step`).
# 10 is the default the model was trained and previously scored at; NFE=2 is the cheap-
# inference question. ⚠️ It goes in the TAG below, because a 2-step and a 10-step run of the
# SAME checkpoint otherwise write the SAME per-clip JSON and the second silently overwrites
# the first -- destroying exactly the paired comparison the run exists to produce.
NFE="${NFE:-10}"
TAG_BASE="${TAG_BASE:-$TAG_PREFIX}"

# ⚠️ REFUSE if the pin is set. The expert in the EOS checkpoint is ALREADY 28 layers, so
# n_ckpt == n_have and the DEPTH REMAP must not run. A stray export from a stitched-eval
# shell would not crash here -- the branch is simply skipped -- but leaving it visible in the
# environment invites someone to conclude the remap was applied. Fail loudly instead.
if [[ -n "${PRUNE_EXPERT_LAYERS:-}" ]]; then
    echo "[slurm] REFUSING: PRUNE_EXPERT_LAYERS is set; this checkpoint's expert is already 28 layers." >&2
    exit 1
fi

# Newest-checkpoint default, sorted on the STEP NUMBER ALONE.
# ⚠️ NOT `sort -t- -k2 -n` on the full path, which is the idiom used elsewhere in this tree
# and is BROKEN: the path itself contains hyphens ("alpamayo-recipes"), so field 2 is
# "recipes/..." -- a non-numeric string that sorts as 0 for EVERY candidate. The comparison
# then degenerates to a stable no-op and `tail -1` returns whatever `ls` listed last, i.e.
# LEXICAL order. It happens to give the right answer for checkpoints 1563..7815 (all four
# digits) and the WRONG one the moment the widths differ: on the CD run it returned
# checkpoint-900 while checkpoint-2084 existed. Strip to the integer and sort that.
if [[ -z "$CKPT" ]]; then
    CKPT=$(ls -d "$EOS_DIR"/checkpoint-* 2>/dev/null \
           | sed 's/.*checkpoint-//' | sort -n | tail -1)
    CKPT="$EOS_DIR/checkpoint-$CKPT"
else
    CKPT="$EOS_DIR/$CKPT"
fi
[[ -d "$CKPT" ]] || { echo "[slurm] no such checkpoint: $CKPT" >&2; exit 1; }

# The student tower is the SAME in both arms -- only where `expert.*` comes from changes.
EXPERT_SRC="$CKPT"
[[ "$ARM" == "control" ]] && EXPERT_SRC="$TEACHER"
# ...and the 2B control DOES need the remap, because the 10B expert is 36 layers deep while
# the 2B student's is 28. ⚠️ NOT for 4b: that student's expert is already 36, so n_ckpt ==
# n_have, the remap branch is skipped, and setting the pin would only mislead a later reader
# into thinking a remap happened.
if [[ "$ARM" == "control" && "$MODEL" == "2b" ]]; then
    export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ${SLURM_JOB_ID:-$$}: this is also run OUTSIDE slurm when a long job holds the node.
MASTER_PORT=$((29900 + ${SLURM_JOB_ID:-$$} % 20000))

# ⚠️ nproc_per_node MUST stay 1. evaluate_hf collects per-clip records on the main process
# only, so the per-clip JSON is complete only in a single-process run.
TAG="${TAG_BASE}_${ARM}_$(basename "$CKPT")"
[[ "$NFE" != "10" ]] && TAG="${TAG}_nfe${NFE}"
[[ "$MAX_EVAL_STEPS" != "-1" ]] && TAG="${TAG}_smoke${MAX_EVAL_STEPS}"
echo "[slurm] ARM=$ARM student<-$CKPT expert<-$EXPERT_SRC bs=$BS nfe=$NFE steps=$MAX_EVAL_STEPS"
echo "[slurm] -> $OUT_DIR/$TAG.json"
echo "[slurm] -> $OUT_DIR/$TAG.npz  (raw pred_xyz/gt_xyz for offline re-scoring)"
nvidia-smi -L

srun "$VENV/torchrun" --nproc_per_node 1 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "$CONFIG" \
    ++evaluate.eval_ckpt="$CKPT" \
    ++model.teacher_checkpoint_path="$EXPERT_SRC" \
    ++evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    ++evaluate.per_clip_output="$OUT_DIR/$TAG.json" \
    ++evaluate.trajectory_output="$OUT_DIR/$TAG.npz" \
    ++evaluate.metric_runner.metrics.0.diffusion_kwargs.inference_step="$NFE" \
    ++trainer.per_device_eval_batch_size="$BS" \
    paths.output_dir="$OUT_DIR/$TAG" \
    ${EXTRA_ARGS:-} \
    2>&1 | tee "$OUT_DIR/$TAG.log"

echo; echo "================ TRIPWIRES ================"
LOG="$OUT_DIR/$TAG.log"
# "loaded N teacher tensors (309 expert)" -- if this line is absent the expert never loaded
# and the model rolled out with randomly-initialised action weights.
grep -a "stitch\] loaded" "$LOG" | tail -1 || echo "!! expert never loaded"
grep -a "Loaded .* VLM tensors" "$LOG" | tail -1
grep -aE "val/count|kept [0-9]+/[0-9]+ clips" "$LOG" | tail -2
BAD=$(grep -ac "Invalid token ids\|not equal to the expected" "$LOG" || true)
"$VENV/python" - "$OUT_DIR/$TAG.json" "$BAD" <<'PY'
import json, sys, statistics as st
recs = json.load(open(sys.argv[1])); n = len(recs)
print(f"per-clip records: {n}" + ("" if n == 1000 else "   <-- expected 1000 on a full run"))
bad = int(sys.argv[2])
print(f"malformed warnings: {bad}" + (f" = {100*bad/n:.1f}% of clips" if n else ""))
for m in ("ade", "min_ade", "max_ade"):
    v = [r[m] for r in recs if m in r]
    if v:
        print(f"  {m:<8} {st.mean(v):.4f}  (se {st.stdev(v)/len(v)**0.5:.4f})")
# If all 6 samples coincide the sampler collapsed and min_ade is really ade -- the
# -0.196-style deltas this arm is looking for are smaller than that artefact.
eq = sum(1 for r in recs if abs(r["ade"] - r["min_ade"]) < 1e-9)
if n:
    print(f"  all-6-samples-identical: {eq}/{n} = {100*eq/n:.1f}%   (mode-collapse tripwire)")
print("  compare ONLY against ARM=control from this same script")
PY
