#!/bin/bash
#SBATCH --job-name=a1_5_cd_eos4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeos4b_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeos4b_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com
#
# Two-epoch endpoint distillation from the 4B 2-camera EoS checkpoint-3126.
# Student VLM frozen; only the 36-layer action expert trains.
# Effective batch = 4 samples/rank x 4 ranks x 2 accumulation = 32.
#
#   SMOKE=1 sbatch slurm_train_cd_eos_4b.sh    # 2 optimizer steps -- ALWAYS run this first
#   sbatch slurm_train_cd_eos_4b.sh            # full 2 epochs, ~3126 steps
#   RESUME=auto WORKERS=8 sbatch --nodelist=amhrisvh100b slurm_train_cd_eos_4b.sh
#
# ⚠️ 48 cores is sized for h100b, the SMALLEST node that runs this. h100c has 128 and h200b
# 384; ask for more there. This job is data-bound, not compute-bound -- the EoS run measured
# 12.2 s/it at 2 workers/rank against 4.7 s/it at 10 -- so cores, not the GPU model, set the
# wall clock. Keep nproc_per_node at 4 regardless: the 4 x bs4 x accum2 = 32 effective batch
# is part of the optimization recipe, and changing GPU count mid-run silently rescales it.
#
# ⚠️ No --exclusive. h100c is often already holding an interactive allocation and exclusive
# would queue behind it indefinitely.
#   sbatch --nodelist=amhrisvh200b slurm_train_cd_eos_4b.sh   # 384 cores, use WORKERS=10
set -euo pipefail

SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
CONFIG="${CONFIG:-sft_cd_eos_4b_endpoint_gt05_2cam_nav_lcdrive}"
EOS_CKPT="${EOS_CKPT:-$OUT/output_eos_4b_2cam_nav_lcdrive/checkpoint-3126}"
RUN_OUT="${OUTPUT_DIR:-$OUT/output_cd_eos4b_endpoint_gt05_2cam_nav_e2_bs32}"
CACHE="$OUT/teacher_action_rollouts_full10b_2cam_nav_k6_m10"

# The EoS expert is already the full 36 layers the 4B student was trained with, so the
# identity-slot depth remap must not run. Unset rather than export empty: the variable is
# read with a bare presence test elsewhere in this tree.
unset PRUNE_EXPERT_LAYERS

if [[ ! -f "$EOS_CKPT/model.safetensors.index.json" && ! -f "$EOS_CKPT/model.safetensors" ]]; then
    echo "[slurm] missing EoS checkpoint weights: $EOS_CKPT" >&2
    exit 1
fi
# The endpoint objective regresses onto cached 10B rollouts. If the cache is missing the
# run does NOT fail -- `teacher_trajectory_cached_only: false` lets every clip fall through
# uncached, and two epochs later you have a checkpoint trained on almost no target signal.
# Check it here, where it costs a second.
if [[ ! -d "$CACHE/action_rollouts" ]]; then
    echo "[slurm] missing teacher trajectory cache: $CACHE" >&2
    exit 1
fi
# RESUME: `auto` for the newest checkpoint, or a bare name like `checkpoint-1563`.
# ⚠️ A bare name is resolved against RUN_OUT here, NOT passed through. HF resolves
# `resume_from_checkpoint` as a filesystem path relative to the CWD, so handing it
# "checkpoint-1563" makes it look in the recipe dir and die with "Can't find a valid
# checkpoint" -- three minutes in, after the model has already loaded.
#
# ⚠️ Resuming REBUILDS the LR schedule from the config, it does not restore the old curve
# unless num_train_epochs is unchanged. Keep epochs at 2 when finishing this run, or the
# resumed steps get a warm restart at a higher LR than they ended on.
RESUME="${RESUME:-}"
if [[ -n "$RESUME" ]]; then
    if [[ "$RESUME" == "auto" ]]; then
        RESUME=$(ls -d "$RUN_OUT"/checkpoint-* 2>/dev/null \
                 | sed 's/.*checkpoint-//' | sort -n | tail -1)
        RESUME="$RUN_OUT/checkpoint-$RESUME"
    elif [[ "$RESUME" != /* ]]; then
        RESUME="$RUN_OUT/$RESUME"
    fi
    [[ -d "$RESUME" ]] || { echo "[slurm] no such checkpoint to resume: $RESUME" >&2; exit 1; }
    [[ -f "$RESUME/trainer_state.json" ]] || {
        echo "[slurm] $RESUME has no trainer_state.json; it is not resumable" >&2; exit 1; }
    echo "[slurm] RESUME <- $RESUME"
fi

# The freshness guard exists so a rerun never silently interleaves new checkpoints with an
# old run's. Resuming is the one legitimate way to write into a populated directory, so it
# is exempt -- but ONLY when RESUME is set.
if [[ -z "$RESUME" && -d "$RUN_OUT" && -n "$(find "$RUN_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[slurm] refusing non-empty output directory: $RUN_OUT" >&2
    echo "[slurm] set OUTPUT_DIR to a fresh path, or RESUME=auto to continue it" >&2
    exit 1
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAVA_GRAD_PROBE_STEPS="${KAVA_GRAD_PROBE_STEPS:-0}"
# ⚠️ REQUIRED, and not obvious. `sft_alpamayo_r1.from_pretrained_vlm` builds
# `AlpamayoR1Config.from_pretrained(<10B snapshot>)` BEFORE it overrides
# `vlm_name_or_path`, and that constructor eagerly calls
# `AutoProcessor.from_pretrained(self.vlm_name_or_path)`. At that instant the value is
# still whatever the 10B config.json stores -- the bare repo id `nvidia/Cosmos-Reason2-8B`,
# which is GATED. Without these two the job dies in ~60 s with a 401 that names a model
# this run does not otherwise touch. The 8B is already in /temp/achahe/hf_cache; HF_HOME
# points at it and OFFLINE stops the revision HEAD request that would 401 anyway.
# The stitched-model training/eval path never hits this because it rewrites the config
# before instantiation, which is why the EoS runs worked with no HF env at all.
export HF_HOME=/temp/achahe/hf_cache
export HF_HUB_OFFLINE=1
MASTER_PORT=$((30420 + SLURM_JOB_ID % 20000))

EXTRA=(
    "model.eos_checkpoint_path=$EOS_CKPT"
    "paths.output_dir=$RUN_OUT"
)
[[ -n "$RESUME" ]] && EXTRA+=("++trainer.resume_from_checkpoint=$RESUME")
# ⚠️ Sized to the NODE, not copied. The config's 10 workers/rank suits h100c (128 cores);
# h100b has only 48 across 8 GPUs. At 4 ranks, 8 workers/rank is 36 processes on 48 cores
# -- the most this node takes before the loaders start fighting each other. Overriding
# here rather than in the config keeps the config node-agnostic.
[[ -n "${WORKERS:-}" ]] && EXTRA+=("++trainer.dataloader_num_workers=$WORKERS")
if [[ "$SMOKE" == "1" ]]; then
    # No `callbacks.ema` overrides here, unlike the 2B launcher: cd_weight=0 builds no
    # target network, so those keys would create an unused config node.
    EXTRA+=(++trainer.max_steps=2 ++trainer.logging_steps=1 ++trainer.save_strategy=no
            ++trainer.eval_strategy=no ++trainer.warmup_steps=0
            ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids=0-120
            ++trainer.dataloader_num_workers=2)
    echo "[slurm] SMOKE: 2 optimizer steps"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=(${EXTRA_ARGS})

echo "[slurm] config:        $CONFIG"
echo "[slurm] init from EoS: $EOS_CKPT  (frozen 4B VLM + 36-layer expert)"
echo "[slurm] targets:       cached 10B rollouts <- $CACHE"
echo "[slurm] objective:     endpoint (cd_weight=0) + gt damper 0.5"
echo "[slurm] effective batch: 4 x 4 GPUs x 2 accumulation = 32"
echo "[slurm] epochs: 2; output: $RUN_OUT"
nvidia-smi -L

srun "$VENV/torchrun" --nproc_per_node 4 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "$CONFIG" \
    "run_name=cd_eos4b_gt05_$(date +%m%d-%H%M)" \
    "${EXTRA[@]}"
