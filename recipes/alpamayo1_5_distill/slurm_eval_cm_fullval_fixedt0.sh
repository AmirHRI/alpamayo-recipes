#!/bin/bash
#SBATCH --job-name=cm_fullval_nfe1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cm_fullval_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cm_fullval_%j.err
set -euo pipefail
REPO=/home/achahe/alpamayo-recipes
VENV="$REPO/recipes/alpamayo1_5_sft/.venv/bin"
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
MANIFEST_DIR=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests
ANNOTATIONS="$MANIFEST_DIR/nav_lcdrive_val_available_fixedt0_23331_stripped.json"
CLIP_LIST="$MANIFEST_DIR/lcdrive_val_available_23331_clip_uuids.txt"
MODEL_ARGS=()
case "${MODEL:?Set MODEL=4b, MODEL=2b or MODEL=teacher}" in
    4b) RUN=output_cd_eos4b_consistency2to1_fp16_master32_2cam_nav
        CONFIG=sft_eval_eos_4b_2cam_nav_lcdrive
        CKPT="$OUT/$RUN/checkpoint-6876" ;;
    2b) RUN=output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav
        CONFIG=sft_eval_eos_2b_mix_nav_lcdrive
        CKPT="$OUT/$RUN/checkpoint-6876" ;;
    teacher) CONFIG=sft_eval_eos_4b_2cam_nav_lcdrive
        CKPT=/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format
        MODEL_ARGS=(
            ++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_teacher
            ++model.vlm_name_or_path=/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9
        ) ;;
    *) echo "MODEL must be 4b, 2b or teacher" >&2; exit 1 ;;
esac
TAG="cm${MODEL}_availableval23331_fixedt0_stripped_ep2_nfe1_${SLURM_JOB_ID}"
if [[ "$MODEL" == "teacher" ]]; then
    TAG="teacher15_availableval23331_fixedt0_stripped_nfe1_${SLURM_JOB_ID}"
fi
[[ -f "$CKPT/model.safetensors.index.json" && -f "$ANNOTATIONS" ]] || {
    echo "Missing checkpoint or prepared full validation manifest" >&2; exit 1;
}
export PYTHONPATH="$REPO/recipes:$REPO/src"
export HF_HOME=/temp/achahe/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PRUNE_EXPERT_LAYERS STITCH_ROLLOUT STITCH_PREFILL_SELFTEST
"$VENV/python" - "$ANNOTATIONS" "$CLIP_LIST" <<'PY'
import json
import sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())
expected = set(Path(sys.argv[2]).read_text().split())
assert len(rows) == len(expected) == 23331
assert {row["clip_id"] for row in rows} == expected
assert all(row["t0_relative"] == 5100000 for row in rows)
assert all(row["nav_text"] in {"Continue straight", "Turn left", "Turn right", "Reverse"} for row in rows)
print("Preflight: 23,331 available validation clips; 427 missing-index clips excluded; t0=5.1s, distance-free navigation")
PY
cd "$REPO/recipes/alpamayo1_5_distill"
echo "[fullval] MODEL=$MODEL VLM/expert/mixer checkpoint=$CKPT NFE=1 cameras=[1,3]"
echo "[fullval] output=$OUT/$TAG.npz"
srun "$VENV/torchrun" --nproc_per_node=1 --master_port="$((29900 + SLURM_JOB_ID % 20000))" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_distill/configs --config-name "$CONFIG" \
    "++evaluate.eval_ckpt=$CKPT" "++model.teacher_checkpoint_path=$CKPT" \
    "++data.val_dataset.annotations_path=$ANNOTATIONS" \
    "++data.val_dataset.clip_uuid_filter=$CLIP_LIST" \
    ++data.val_dataset.chunk_ids=0-3146 ++data.val_dataset.strip_nav_turn_distance=true \
    ++evaluate.max_eval_steps=-1 ++evaluate.metric_runner.metrics.0.diffusion_kwargs.inference_step=1 \
    ++trainer.per_device_eval_batch_size=4 \
    "++evaluate.per_clip_output=$OUT/$TAG.json" "++evaluate.trajectory_output=$OUT/$TAG.npz" \
    "paths.output_dir=$OUT/$TAG" "${MODEL_ARGS[@]}"
"$VENV/python" - "$OUT/$TAG" "$CLIP_LIST" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
stem = Path(sys.argv[1])
expected = set(Path(sys.argv[2]).read_text().split())
records = json.loads(stem.with_suffix(".json").read_text())
assert len(records) == len(expected) == 23331
assert {str(row["clip_id"]) for row in records} == expected
with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
    ids = archive["clip_ids"].astype(str)
    assert len(ids) == len(set(ids)) == 23331 and set(ids) == expected
    assert archive["pred_xyz"].shape == (23331, 1, 6, 64, 3)
    assert archive["pred_xyz"].dtype == np.float32
    assert np.isfinite(archive["pred_xyz"]).all() and np.isfinite(archive["gt_xyz"]).all()
print("Verified complete 23,331-clip available-validation JSON and float32 NPZ output")
PY