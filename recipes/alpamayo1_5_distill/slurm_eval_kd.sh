#!/bin/bash
#SBATCH --job-name=a1_5_kd_eval
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdeval_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdeval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Score a trained Qwen3-VL-4B student on the 1k LCDrive val subset, through the SAME
# evaluate_hf + ReasoningSampler + DistanceMetrics path that produced the surviving 10B
# baselines -- which is the only reason the numbers are comparable.
#
#   ARM=ce sbatch slurm_eval_kd.sh                     # the CE-only control
#   ARM=kd sbatch slurm_eval_kd.sh                     # +logit-KD
#   ARM=kv sbatch slurm_eval_kd.sh                     # +KV alignment
#   ARM=ce CKPT=checkpoint-1000 sbatch slurm_eval_kd.sh
#   ARM=ce MAX_EVAL_STEPS=10 sbatch slurm_eval_kd.sh   # ~2 min smoke
#
# THE BASELINES THIS IS MEASURED AGAINST (n=1000, recomputed from the per-clip JSONs,
# both with camera ids + frame numbers ON and scored on the VLM TOKEN head -- head-matched
# and format-matched to the student):
#     Alpamayo-1.5-10B, no CoT :  ade 1.2111   min_ade 0.6413   <- the comparison
#     Alpamayo-1.5-10B, CoT    :  ade 1.2417   min_ade 0.6259
# Older numbers in this tree (ade ~2.02-2.21, min_ade ~0.99-1.12) are RETRACTED: they ran
# the action-expert/diffusion head AND without camera ids. Never quote them here.
#
# ⚠️ COMPARE ARMS PAIRED, PER CLIP -- not as a difference of aggregates. Measured on the
# two 10B arms: min_ade clip-difficulty correlation r = 0.812, so the standard error falls
# 0.0410 unpaired -> 0.0180 paired. A real 0.05 effect is INVISIBLE unpaired and clear
# paired. Every run writes bare-UUID per-clip records for exactly this reason; join on
# clip_id. Report BOTH ade and min_ade: ade is sample 0 of 6 (metric_api.py:224 fills
# logprob with zeros, so the argmax at :229 always returns index 0) while min_ade is
# oracle best-of-6, and they have already told opposite stories once.

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT_DIR=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
SUBSET=/data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests/lcdrive_val_mysubset_1k_clip_uuids.txt

ARM="${ARM:?set ARM=ce|kd|kv}"
CKPT="${CKPT:-}"
# Batch size shifts the RNG stream (generation is do_sample=True, hardcoded at
# sft_base_model.py:612; the only seeding is Trainer.__init__'s global set_seed(42)).
# 4 is what the surviving 10B baselines used -- keep it identical across every arm.
BS="${BS:-4}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"

RUN_DIR="$OUT_DIR/output_kd_4b_${ARM}_lcdrive"
if [[ -z "$CKPT" ]]; then
    CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [[ -z "$CKPT" ]] && { echo "[slurm] no checkpoint-* under $RUN_DIR" >&2; exit 1; }
else
    CKPT="$RUN_DIR/$CKPT"
fi
TAG="evalhf_4b_${ARM}_$(basename "$CKPT")"
echo "[slurm] ARM=$ARM ckpt=$CKPT -> $OUT_DIR/$TAG.json"

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$((29700 + SLURM_JOB_ID % 20000))

# --nproc_per_node 1 is NOT arbitrary: the per-clip dump (evaluate_hf.py:129-147) is inside
# `if is_main_process`, so at world_size>1 you silently get ~1/N of the clips with only a
# warning at :162-167.
srun "$VENV/torchrun" --nproc_per_node 1 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_kd_qwen3_4b_lcdrive \
    data.val_dataset.clip_uuid_filter="$SUBSET" \
    ++evaluate.eval_ckpt="$CKPT" \
    ++evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    ++evaluate.per_clip_output="$OUT_DIR/$TAG.json" \
    ++model.teacher_checkpoint_path=null \
    ++trainer.per_device_eval_batch_size="$BS" \
    ++trainer.deepspeed=null \
    ++trainer.gradient_checkpointing=false \
    ++trainer.report_to=none \
    ~wandb ~wandb.team ~wandb.project ~wandb.group \
    paths.output_dir="$OUT_DIR/$TAG" \
    2>&1 | tee "$OUT_DIR/$TAG.log"

# ---------------------------------------------------------------- tripwires
# Each of these failure modes produces a PLAUSIBLE NUMBER rather than a crash, which is
# how a wrong result gets published. Checked here so the failure is loud and attached to
# the run that produced it, instead of being noticed months later or not at all.
LOG="$OUT_DIR/$TAG.log"
echo; echo "================ TRIPWIRES ================"

# 1. Malformed generations. token_utils.py:58-117 ZERO-FILLS missing trajectory slots and
#    clamps out-of-range ids, logging only a WARNING -- a student that fails to emit
#    exactly 128 tokens still yields a believable min_ade. The 10B CoT run tripped this on
#    3 clips and excluding them moved ade 1.2417 -> 1.2021. A freshly distilled 4B is far
#    likelier to hit it, so this is the first thing to read.
#    ⚠️ Judge the RATE, not the count. An absolute threshold silently passes short runs:
#    the first smoke (MAX_EVAL_STEPS=10, 40 clips) logged 2 warnings and printed "ok" at a
#    5% malformation rate -- twelve times the 10B's 0.4%, i.e. exactly the condition the
#    tripwire exists to catch, reported as healthy.
BAD=$(grep -ac "Invalid token ids\|not equal to the expected" "$LOG" || true)
NREC=$("$VENV/python" -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$OUT_DIR/$TAG.json" 2>/dev/null || echo 0)
if [[ "$NREC" -gt 0 ]]; then
    PCT=$(awk -v b="$BAD" -v n="$NREC" 'BEGIN{printf "%.1f", 100*b/n}')
    VERDICT=$(awk -v b="$BAD" -v n="$NREC" 'BEGIN{print (100*b/n > 2.0) ? "<-- ⚠️ METRICS SUSPECT (10B reference: 0.4%)" : "ok"}')
    echo "malformed generations : $BAD / $NREC = ${PCT}%  $VERDICT"
else
    echo "malformed generations : $BAD (no per-clip json -- rate unknown)"
fi

# 2. Wrong weights, silently. load_alpamayo1_vlm takes only `vlm.*`; checkpoint-1598 holds
#    714 such tensors. missing>0 means part of the model stayed at its base init.
grep -a "Loaded .* VLM tensors" "$LOG" | tail -1 || echo "!! no 'Loaded N VLM tensors' line -- weights may not have loaded"

# 3. Prompt format -- the failure that already cost two retracted findings (min_ade
#    1.116 -> 0.626 from format alone). Both flags must be true, and `cot` must be absent.
echo "include_camera_ids true : $(grep -ac "'include_camera_ids': True" "$LOG" || true)"
echo "include_frame_nums true : $(grep -ac "'include_frame_nums': True" "$LOG" || true)"

# 4. Whole subset actually scored. max_eval_steps counts BATCHES and truncates from the
#    start, with no resume -- a partial run looks exactly like a complete one.
grep -aE "val/count|kept [0-9]+/[0-9]+ clips" "$LOG" | tail -2

# 5. Aggregates are never serialised (only logger.info at evaluate_hf.py:184-191), and two
#    jobs launched in the same second SHARE one hydra log file -- so recompute from the
#    per-clip JSON, which reproduces the logged value exactly, rather than scraping text.
"$VENV/python" - "$OUT_DIR/$TAG.json" <<'PY'
import json, sys, statistics as st
recs = json.load(open(sys.argv[1]))
n = len(recs)
print(f"per-clip records: {n}" + ("" if n == 1000 else "   <-- ⚠️ EXPECTED 1000"))
for m in ("ade", "min_ade"):
    v = [r[m] for r in recs if m in r]
    if v:
        print(f"  {m:<8} {st.mean(v):.4f}  (se {st.stdev(v)/len(v)**0.5:.4f})")
print("  baselines @n=1000: 10B no-CoT ade 1.2111 min_ade 0.6413 | 10B CoT ade 1.2417 min_ade 0.6259")
PY
