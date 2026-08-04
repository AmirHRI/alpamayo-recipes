#!/bin/bash
#SBATCH --job-name=a1_5_kava_train
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kavatrain_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kavatrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
# 120G: host RAM is only for the frame-decoding dataloader workers (the model lives on
# the GPU). Asking 160G left the job PENDING on (Resources) when co-tenants held 368G of
# the node's 503 GiB — a memory limit, not a GPU one.
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Stage-1 KAVA distillation: 2B student warm-started from the Stage-1 LCDrive
# checkpoint, K=8 latent slots supervised by the teacher's M=8 compressed KV cache.
#
#   SMOKE=1 sbatch slurm_train_kava.sh     # 20 steps, probe every 2 — verify, then run
#   CONTROL=1 sbatch slurm_train_kava.sh   # same schedule, BOTH aux losses off
#   sbatch slurm_train_kava.sh             # the real run
#   GPUS=2 sbatch --gpus=2 slurm_train_kava.sh
#
# Memory: a single-GPU step is ~60 GiB (30 GiB forward/backward at bs=1, plus AdamW
# moments for 2.67 B params), so `deepspeed: zero2` from sft_base is load-bearing —
# it shards optimizer state and gradients. Do not disable it without dropping bs.
#
# ⚠️ gradient_checkpointing MUST stay false: it forces use_cache=False and detaches
# hook-captured K/V, so L_KV would train nothing while still looking finite.

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin
OUT_DIR=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
CACHE_ROOT="${CACHE_ROOT:-/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/teacher_kv_lcdrive}"
GPUS="${GPUS:-1}"
SMOKE="${SMOKE:-0}"
# Sweep knobs. Defaults reproduce the committed config; any override also gets its own
# output_dir and run_name so arms cannot overwrite each other's checkpoints.
JACOBI="${JACOBI:-}"      # T (PCCoT iterations); 2+ costs ~+10 GiB/GPU and needs >=2 ranks at bs=2
BS="${BS:-}"              # per-device batch; bs=2 needs >=2 ranks (ZeRO-2 shards only across ranks)
ACCUM="${ACCUM:-}"        # effective batch = GPUS x BS x ACCUM; the Stage-1 baseline used 48
WARMUP="${WARMUP:-}"      # scale with steps/epoch: the baseline ran ~21% of total steps

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
# The CE loss upcasts logits to fp32 over a 155,697 vocab; expandable segments reclaim
# the large reserved-but-unallocated blocks that otherwise trigger OOM at the loss step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$((29540 + SLURM_JOB_ID % 20000))

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
    # Short, self-contained proof that the whole path runs: warm start loads, slots
    # splice, L_KV is non-zero, and the gradient probe reports a live kv share.
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=1
            ++trainer.save_strategy=no ++trainer.eval_strategy=no
            ++trainer.warmup_steps=0 ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids="0-120" ++data.val_dataset.chunk_ids="0-120")
    export KAVA_GRAD_PROBE_STEPS=2
    echo "[slurm] SMOKE mode: 20 steps, gradient probe every 2"
fi

# CONTROL arm: identical warm start, data, schedule and slots, distillation OFF. Its
# only job is to attribute the first run's +0.158 min_ade regression, which accrued over
# 7,191 steps of which only ~300 carried real KAVA gradient — so over-training is a live
# explanation that has nothing to do with the method.
#   control regresses ~as much  -> the damage is over-training; KAVA is neutral here
#   control regresses less      -> the distillation terms genuinely hurt
#   control does not regress    -> something specific to this configuration
# Done as CLI overrides rather than a second config: hydra forbids inheriting a config
# that declares hydra.searchpath, and a copied config would drift from this one.
# NOTE: this must NOT set paths.output_dir itself. The sweep block below sets the same
# key, hydra keeps the LAST occurrence, and a CONTROL+sweep combination therefore wrote
# into the KAVA arm's directory — nearly overwriting the checkpoint it was meant to be
# compared against. CONTROL contributes to SWEEP_TAG instead, so one place owns the path.
CONTROL_TAG=""
if [[ "${CONTROL:-0}" == "1" ]]; then
    EXTRA+=(model.latent_loss_weight=0.0 model.kava.kv_loss_weight=0.0)
    CONTROL_TAG="_noaux"
    echo "[slurm] CONTROL arm: lambda_1 = lambda_2 = 0 (gradshare_kv should log 0.0)"
fi

# Sweep overrides. Effective batch MUST be compared against the Stage-1 baseline's 48:
# the first runs used 16 at the same lr 1e-5, and the CE-only control regressed
# +0.203 min_ade against the baseline on that schedule alone.
SWEEP_TAG="$CONTROL_TAG"
[[ -n "$JACOBI" ]] && { EXTRA+=(model.kava.jacobi_iters="$JACOBI"); SWEEP_TAG="${SWEEP_TAG}_T$JACOBI"; }
[[ -n "$BS"     ]] && { EXTRA+=(trainer.per_device_train_batch_size="$BS"); }
[[ -n "$ACCUM"  ]] && { EXTRA+=(trainer.gradient_accumulation_steps="$ACCUM"); }
[[ -n "$WARMUP" ]] && { EXTRA+=(trainer.warmup_steps="$WARMUP"); }
if [[ -n "$BS$ACCUM" ]]; then
    EFF=$(( GPUS * ${BS:-1} * ${ACCUM:-16} )); SWEEP_TAG="${SWEEP_TAG}_bs$EFF"
    echo "[slurm] effective batch = $GPUS gpus x ${BS:-1} x ${ACCUM:-16} = $EFF  (baseline: 48)"
fi
if [[ -n "$SWEEP_TAG" ]]; then
    # single owner of output_dir / run_name -- see the CONTROL note above
    EXTRA+=(paths.output_dir="$OUT_DIR/output_kava${SWEEP_TAG}_lcdrive"
            "run_name=kava_M8${SWEEP_TAG}_$(date +%m%d-%H%M)")
    echo "[slurm] sweep arm${SWEEP_TAG} -> $OUT_DIR/output_kava${SWEEP_TAG}_lcdrive"
fi

# The cgroup exposes roughly half of --cpus-per-task, so 12 workers (the config default,
# sized for a multi-GPU run) thrash here. Derive it from the actual allocation.
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} / (3 * GPUS) ))}"
[[ "$WORKERS" -lt 2 ]] && WORKERS=2
EXTRA+=(++trainer.dataloader_num_workers="$WORKERS")

# Free-form hydra overrides, word-split. For one-off knobs that do not deserve a named
# variable -- e.g. EXTRA_ARGS='++trainer.ddp_timeout=5400' after a NCCL collective
# timeout. Deliberately unquoted expansion so multiple overrides can be passed.
# shellcheck disable=SC2206,SC2086
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

echo "[slurm] job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES cache_root=$CACHE_ROOT workers=$WORKERS"

srun "$VENV/torchrun" \
    --nproc_per_node "$GPUS" \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kava \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_stage1_kava_cosmos2b_lcdrive \
    data.train_dataset.kv_cache_root="$CACHE_ROOT" \
    "${EXTRA[@]}"
