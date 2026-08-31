#!/bin/bash
#SBATCH --job-name=a1_5_stitch_eval
#SBATCH --partition=gpu
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

ARM="${ARM:?set ARM=teacher|ce|kd|kv|cekv|kvonly|block2b|<arm>_eN}"
# ARM=block2b is the 2B student on the 28-layer PRUNED expert: a different student tower, a
# different eval config, and a different output_dir prefix. Its expert is built 28 deep (depth
# follows the VLM's text config) and initialised by the load-time remap in
# _load_teacher_non_vlm, which REQUIRES PRUNE_EXPERT_LAYERS -- exported here rather than left
# to the caller, because a missing pin would raise mid-load after ~5 min of weight loading.
MODEL_TAG=4b
CONFIG_NAME=sft_eval_stitched_4b_lcdrive
# ⚠️ SUBSTRING, not a `block2b*` prefix: the tag is derived from the ARM NAME, so a new 2B arm
# whose name does not start with "block2b" silently resolved MODEL_TAG=4b and died looking for
# output_kd_4b_<arm>_lcdrive (job 512, ARM=field2b). Any arm carrying "2b" is the 2B stack.
if [[ "$ARM" == *2b* ]]; then
    MODEL_TAG=2b
    CONFIG_NAME=sft_eval_stitched_2b_prunedexpert_lcdrive
    export PRUNE_EXPERT_LAYERS="${PRUNE_EXPERT_LAYERS:-4,10,13,15,19,25,27,34}"
fi
# PIN_GPU=3 -> run on that PHYSICAL card. ⚠️ Must NOT go through srun: slurm re-derives
# CUDA_VISIBLE_DEVICES from the step's GPU binding after --export is processed, so the pin is
# discarded and the job silently takes cuda:0 (this cost a co-tenant's card once -- see
# slurm_train_kd.sh). Launching torchrun directly keeps the allocation and the pin.
LAUNCH=(srun)
if [[ -n "${PIN_GPU:-}" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$PIN_GPU"
    LAUNCH=()
    echo "[slurm] PIN_GPU=$PIN_GPU -> CUDA_VISIBLE_DEVICES=$PIN_GPU, no srun"
fi
CKPT="${CKPT:-}"   # e.g. checkpoint-3196; default is the newest
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"
BS="${BS:-4}"

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ${SLURM_JOB_ID:-$$}: this script is also run OUTSIDE slurm (with PIN_GPU) whenever a
# long training job holds the whole node and a fresh sbatch would only pend. Under `set -u`
# the bare SLURM_JOB_ID would abort immediately there.
MASTER_PORT=$((29840 + ${SLURM_JOB_ID:-$$} % 20000))

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
        CKPT="$OUT_DIR/output_kd_${MODEL_TAG}_${ARM}_lcdrive/$CKPT"
        [[ -d "$CKPT" ]] || { echo "[slurm] no such checkpoint: $CKPT" >&2; exit 1; }
    else
        CKPT=$(ls -d "$OUT_DIR/output_kd_${MODEL_TAG}_${ARM}_lcdrive"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        [[ -z "$CKPT" ]] && { echo "[slurm] no checkpoint for ARM=$ARM" >&2; exit 1; }
    fi
fi
# CAMERAS=[1,3] -> evaluate on that camera SUBSET, with the prompt rebuilt from it.
# ⚠️ REQUIRED for any arm trained on a subset. block2bspan / block2b2cam train on cameras
# [1,3] (sft_kd_cosmos2b_2cam_lcdrive), but the val_dataset shipped here is the 4-camera
# PAIDataset -- so scoring one of those students without this is the SAME off-distribution
# error as dropping the camera-id flags, which moved min_ade 1.116 -> 0.626. It desynchronises
# both halves of the stitch at once: the student never saw 4 cameras, and the expert reads a
# cache built from a prompt the student was not trained on.
# evaluate_hf.py:79 instantiates val_dataset with model_config=model.config, which is the
# argument CameraSubsetPAIDataset needs; the base _target_ comes from sft_base.
TAG_SUF=""
if [[ -n "${CAMERAS:-}" ]]; then
    EXTRA+=(++data.val_dataset._target_=alpamayo1_5_distill.data.camera_subset.CameraSubsetPAIDataset
            ++data.val_dataset.cameras="$CAMERAS")
    TAG_SUF="_cam$(tr -d '[], ' <<< "$CAMERAS")"
    echo "[slurm] CAMERAS=$CAMERAS -> CameraSubsetPAIDataset, prompt rebuilt"
fi
# NAV=<annotations.json> -> evaluate through PAIDatasetWithNav: one sample per ANNOTATION, its
# own t0_relative, and the route in the prompt. Requires CAMERAS (it routes through
# CameraSubsetPAIDataset, which raises if "route" is missing from components_order).
# ⚠️ NOT comparable to the default-keyframe evals: the annotations carry 116 distinct t0 values
# and event-anchored sampling is +0.0440 min_ade harder than the 5.1 s keyframe. Report which
# one a number came from -- the _nav suffix in the tag is there so the files cannot be confused.
if [[ -n "${NAV:-}" ]]; then
    [[ -n "${CAMERAS:-}" ]] || { echo "[slurm] NAV needs CAMERAS (e.g. CAMERAS='[1,3]')" >&2; exit 1; }
    [[ -f "$NAV" ]] || { echo "[slurm] no such annotations file: $NAV" >&2; exit 1; }
    EXTRA+=(++data.val_dataset.annotations_path="$NAV"
            ++data.val_dataset.vla_preprocess_args.components_order="[image,traj_history,route,prompt,traj_future]")
    # ⚠️ the annotations BASENAME goes in the tag, not a bare "_nav": two NAV evals of the same
    # checkpoint (e.g. nav vs its no-nav control) otherwise write the SAME per_clip_output path
    # and the second silently overwrites the first, destroying the paired comparison that is
    # the whole reason for running a control.
    TAG_SUF="${TAG_SUF}_$(basename "$NAV" .json | sed 's/^nav_lcdrive_val_mysubset_1k//; s/^_//; s/^$/nav/')"
    echo "[slurm] NAV=$NAV -> PAIDatasetWithNav, route in components_order"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)
TAG="stitch_${MODEL_TAG}_${ARM}_$(basename "$CKPT")${TAG_SUF}${TAG_SUFFIX:-}"
echo "[slurm] ARM=$ARM ckpt=$CKPT -> $OUT_DIR/$TAG.json + $OUT_DIR/$TAG.npz"

"${LAUNCH[@]}" "$VENV/torchrun" --nproc_per_node 1 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "$CONFIG_NAME" \
    ++model.attn_implementation=sdpa \
    ++evaluate.eval_ckpt="$CKPT" \
    ++evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    ++evaluate.per_clip_output="$OUT_DIR/$TAG.json" \
    ++evaluate.trajectory_output="$OUT_DIR/$TAG.npz" \
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
"$VENV/python" - "$OUT_DIR/$TAG.json" "$BAD" "$OUT_DIR/$TAG.npz" <<'PY'
import json, sys, statistics as st
import numpy as np
recs = json.load(open(sys.argv[1])); n=len(recs)
print(f"per-clip records: {n}" + ("" if n==1000 else "   <-- ⚠️ EXPECTED 1000"))
bad=int(sys.argv[2])
print(f"malformed warnings: {bad}" + (f" = {100*bad/n:.1f}% of clips" if n else ""))
for m in ("ade","min_ade","max_ade"):
    v=[r[m] for r in recs if m in r]
    if v: print(f"  {m:<8} {st.mean(v):.4f}  (se {st.stdev(v)/len(v)**0.5:.4f})")
eq=sum(1 for r in recs if abs(r['ade']-r['min_ade'])<1e-9)
print(f"  all-6-samples-identical: {eq}/{n} = {100*eq/n:.1f}%   (mode-collapse tripwire)")
z=np.load(sys.argv[3], allow_pickle=False)
assert z["pred_xyz"].shape[:3] == (n, 1, 6), z["pred_xyz"].shape
assert z["gt_xyz"].shape[0] == n and len(z["clip_ids"]) == n
print(f"  trajectory archive: pred {z['pred_xyz'].shape} {z['pred_xyz'].dtype}, "
      f"gt {z['gt_xyz'].shape} {z['gt_xyz'].dtype}")
print("  compare ONLY against ARM=teacher from this same script -- token-head numbers do not apply")
PY
