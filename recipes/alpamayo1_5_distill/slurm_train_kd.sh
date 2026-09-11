#!/bin/bash
#SBATCH --job-name=a1_5_kd_train
#SBATCH --partition=gpu
#SBATCH --nodelist=amhrisvh200b
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdtrain_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdtrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=96
# Host RAM covers parallel decoding, prefetch, and the production arm's 96 GiB ZIP cache.
# The model weights stay on GPUs; 640G leaves reclaim headroom after every 320G run saturated
# its Slurm memory cgroup (see nav4bspan2camall below).
#SBATCH --mem=640G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Qwen3-VL-4B student distilled from a co-resident, frozen Alpamayo-1.5-10B teacher.
#
#   ARM=ce  sbatch slurm_train_kd.sh     # control: CE only, no teacher loaded at all
#   ARM=kd  sbatch slurm_train_kd.sh     # + logit-KD over the 4000-token traj slice
#   ARM=kv  sbatch slurm_train_kd.sh     # + all-token/all-layer KV alignment (full recipe)
#   ARM=cekv   sbatch slurm_train_kd.sh  # CE + KV, no logit-KD
#   ARM=kvonly sbatch slurm_train_kd.sh  # KV alone, no CE and no KD
#   ARM=kvband sbatch slurm_train_kd.sh  # KV alone with depth-banded layer weights
#   ARM=blockonly sbatch slurm_train_kd.sh  # L_block alone (teacher-forced block match)
#   ARM=blockrandt sbatch slurm_train_kd.sh # L_block with t sampled, not pinned to 0
#   ARM=nav2bmix sbatch slurm_train_kd.sh   # 2B, nav-conditioned, block(m=1) + span(m=7)
#   ARM=nav4bmix sbatch slurm_train_kd.sh   # 4B, same objective, span m=9 (36-layer expert)
#   ARM=nav4bmix2cam sbatch slurm_train_kd.sh # 4B, span m=9, TWO front cameras
#   bash slurm_train_kd.sh                    # print requested sample; does NOT train
#   APPROVED=1 sbatch slurm_train_kd.sh       # 4B, 2cam, all-data, m=1/9/18/36
#
#   SMOKE=1 ARM=kv sbatch slurm_train_kd.sh
#   RESUME=<ckpt> EPOCHS=3 ARM=kvonly sbatch slurm_train_kd.sh   # continue for more epochs
#
# The three arms exist to attribute the result. Each writes its own output_dir and
# run_name, so they cannot overwrite one another — the failure mode the KAVA sweep hit
# when CONTROL and the sweep block both set paths.output_dir and hydra kept the last one.
#
# Weights are MEASURED, not guessed: scripts/calibrate_kd_weights.py put kd at 14.1% and
# kv at 0.22% of the CE gradient at weight 1.0, so the shipped kv_weight is 45.4. The
# in-training gradient probe that would normally re-check this CANNOT run here — DeepSpeed
# ZeRO-2's per-parameter grad hooks fire during its extra backwards (see
# trainer._probe_gradient_shares). KAVA_GRAD_PROBE_STEPS is therefore left at 0; re-measure
# mid-run by pointing the calibrator at a checkpoint instead.
#
# ⚠️ Unlike slurm_train_kava.sh, gradient_checkpointing here is TRUE and must stay true.
# That script had to forbid it because hook-captured K/V come back detached; this recipe
# recomputes the student's K/V from output_hidden_states instead, which is exactly what
# makes checkpointing available — and it is what fits a 4B student plus a frozen 8B
# teacher on one 80 GB card (measured 69.6 -> 38.6 GB, losses identical to 5 decimals).

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT_DIR=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
GPUS="${GPUS:-4}"
# PIN_GPUS=1,2 -> run on those PHYSICAL devices while still under slurm.
# ⚠️ Needs `--gpus=4` on the sbatch line (or `sbatch --gpus=4`): slurm sets
# CUDA_VISIBLE_DEVICES to the devices it allocated, so pinning to a global index is only
# possible when the job holds the whole node. Costs the idle GPUs, and buys the thing that
# actually matters -- slurm KNOWS these devices are busy. Running outside slurm to pin them
# is what let a later sbatch land on an occupied GPU and OOM two jobs.
LAUNCH=(srun)   # how torchrun is started; PIN_GPUS drops srun, see below
# ⚠️ PIN_GPUS CANNOT GO THROUGH srun. MEASURED on job 456: with `--gpus=4` and
# `srun --export=ALL,CUDA_VISIBLE_DEVICES=1,2`, slurm re-derives CUDA_VISIBLE_DEVICES from
# the step's GPU binding AFTER the export, so the pin was discarded and the two ranks took
# cuda:0/cuda:1 = PHYSICAL 0 and 1 -- landing on a co-tenant's card at 80.0/80 GB. An
# earlier version merely exported the variable in the batch script, which srun overwrote the
# same way: it printed the pin and did nothing.
# So when pinning, torchrun is launched DIRECTLY from the batch script. The job still holds
# the allocation (slurm bookkeeping intact, which is the reason to stay under sbatch at all),
# but nothing re-derives the device list, so the pin is real.
# ⚠️ Indices are CUDA's, and CUDA's default order is FASTEST_FIRST, not nvidia-smi's PCI
# order. They coincide on this node (verified: pin 1,2 -> uuid a5fa/d988 = nvidia-smi 1,2),
# but CUDA_DEVICE_ORDER=PCI_BUS_ID is set so the two can never drift apart. The resolved
# uuid per visible device is printed below -- read it against `nvidia-smi -L` before
# trusting a long run to the pin.
if [[ -n "${PIN_GPUS:-}" ]]; then
    GPUS=$(awk -F, '{print NF}' <<< "$PIN_GPUS")
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        N_ALLOC=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-}")
        N_NODE=$(nvidia-smi -L | wc -l)
        [[ "$N_ALLOC" -eq "$N_NODE" ]] || {
            echo "[slurm] PIN_GPUS=$PIN_GPUS needs the WHOLE node (--gpus=$N_NODE); this job" \
                 "holds $N_ALLOC of $N_NODE, so the pinned indices are not even allocated." >&2
            exit 1; }
    fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$PIN_GPUS"
    LAUNCH=()
    echo "[slurm] PIN_GPUS=$PIN_GPUS -> CUDA_VISIBLE_DEVICES=$PIN_GPUS, nproc=$GPUS, no srun"
fi
SMOKE="${SMOKE:-0}"
ARM="${ARM:-nav4bspan2camall}"
MODEL_TAG=4b          # overridden per-arm below; part of output_dir and run_name
BS="${BS:-}"              # per-device batch; ZeRO-2 shards only ACROSS ranks, so bs>1 needs >=2
ACCUM="${ACCUM:-}"        # effective batch = GPUS x per-device BS x ACCUM
WARMUP="${WARMUP:-}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-}"
MAX_RESTARTS="${MAX_RESTARTS:-}"
# Continue a finished run for more epochs, preserving Adam moments and the dataloader
# position. RESUME is a checkpoint-* dir; EPOCHS is the TOTAL including epochs already
# trained (so a 1-epoch run continued by 2 needs EPOCHS=3).
RESUME="${RESUME:-}"
EPOCHS="${EPOCHS:-}"

cd "$RECIPE_DIR"
# sbatch inherits the submit shell's env, so a shell that never sourced .env produced a run
# that loaded both models, then died in wandb.init with "No API key configured" -- and each
# of the 3 restarts repeated the ~6 min load before failing the same way. Read the key from
# the repo .env when the submit env did not carry one; ~/.netrc still works as the fallback.
if [[ -z "${WANDB_API_KEY:-}" && -r /home/achahe/alpamayo-recipes/.env ]]; then
    WANDB_API_KEY=$(sed -n 's/^[[:space:]]*WANDB_API_KEY[[:space:]]*=[[:space:]]*//p' \
                    /home/achahe/alpamayo-recipes/.env | tail -1 | tr -d '"'\''[:space:]')
    [[ -n "$WANDB_API_KEY" ]] && export WANDB_API_KEY
fi
[[ -n "${WANDB_API_KEY:-}" ]] \
    && echo "[slurm] WANDB_API_KEY set (len ${#WANDB_API_KEY})" \
    || echo "[slurm] WANDB_API_KEY not set; falling back to ~/.netrc for W&B auth."
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
# ⚠️ the venv's bin on PATH: DeepSpeed's CPUAdam is built by torch.utils.cpp_extension,
# which shells out to the `ninja` BINARY. Installing the python package is not enough --
# without this the offload arm dies with "Ninja is required to load C++ extensions"
# followed by "'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'".
export PATH="$VENV:$PATH"
# The CE loss upcasts logits to fp32 over a 155,697 vocab; expandable segments reclaim the
# large reserved-but-unallocated blocks that otherwise trigger OOM at the loss step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Off by design, not by oversight — see the header. A nonzero value costs a crash at
# step 0 (best case) or gradients silently double-reduced into the step (worst case).
export KAVA_GRAD_PROBE_STEPS=0

EXTRA=()
REQUIRE_APPROVAL=0
PREFLIGHT_MANIFEST=""
ZIP_CACHE_GB=0
case "$ARM" in
    ce)
        # ⚠️ teacher_checkpoint_path=null, not "weights at 0". Zeroing the weights would
        # still load and run the 8B teacher every step: same wall-clock and 16 GB/GPU for
        # a term multiplied by zero. Null skips construction entirely, so this arm also
        # tells us what the student costs on its own.
        EXTRA+=(++model.teacher_checkpoint_path=null
                ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0) ;;
    kd)
        EXTRA+=(++model.kd.kv_weight=0.0) ;;
    kv)
        : ;;  # config defaults are the full recipe
    cekv)
        # CE + KV, no logit-KD. The arm the stitched result implies but that was never run:
        # both existing KV numbers INCLUDE logit-KD, and logit-KD measurably HURTS the expert
        # head (+4.38 min_ade vs control, z=+13.7). Removing it should let KV do better than
        # the 2.9554 the kv arm reached.
        EXTRA+=(++model.kd.kd_weight=0.0) ;;
    block2b)
        # 2B student (28 text layers) + PRUNED expert (set C -> 28 active), L_block with t
        # sampled from the teacher's own schedule. The first arm where student and expert
        # depths match by construction, which is the whole point of pruning to 28.
        # ⚠️ PRUNE_EXPERT_LAYERS is exported here, not left to the caller: without it the
        # expert keeps 36 layers and the forward raises on the depth mismatch.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_prunedexpert_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta) ;;
    mixpin2bnav)
        # PINNED-ENDS layer mix. Same 36-layer expert and same objective as mix2bnav; the only
        # difference is WHICH cache slots are synthesised. Student 0..3 -> slots 0..3 and
        # 26..27 -> 34..35 are wired straight through (deepstack/ViT-injection at the head,
        # and the deep layers the causal ladder measured at ~95% of the recoverable gap at the
        # tail); only slots 4..33 are mixed, from student 4..25, in 2 blocks of 11 -> 15.
        # ⚠️ Read this against the mix2bnav arm at the SAME PLR, not against a different one --
        # the mapping is the variable, the multiplier must be held fixed.
        unset PRUNE_EXPERT_LAYERS
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_nav_layermix_pinned_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.layer_mix=true
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix="${MIXM:-9}"
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}")
        ARM="${ARM}_m${MIXM:-9}w${MIXW:-1.0}"
        if [[ -n "${PLR:-}" ]]; then
            EXTRA+=(++trainer.lr_multiplier.layer_mixer="$PLR")
            ARM="${ARM}_plr${PLR}"
        fi ;;
    mix2bnav)
        # THE UNPRUNED EXPERT. Same 2-camera nav stack as nav2bmix, but the teacher's action
        # expert keeps all 36 layers and its cache slots are SYNTHESISED from the student's 28
        # VLM layers by learned block-convex matrices, 4 blocks of 7 -> 9.
        # WHY: pruning to 28 costs 0.2117 min_ade of CEILING before the student is involved --
        # teacher through its full expert 0.5776, through the set-C ablation 0.7893
        # (PRUNING.md). Every 2B block/span arm so far has been optimising toward the lower one.
        # ⚠️ PRUNE_EXPERT_LAYERS IS DELIBERATELY NOT EXPORTED, and must not be inherited from a
        # previous shell: kd_model._init_layer_mix raises if it is set, because pruning and
        # mixing are alternatives. Unset it here so an interactive re-launch cannot leak it in.
        unset PRUNE_EXPERT_LAYERS
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_nav_layermix_lcdrive
        # ⚠️ MIXM defaults to 9, not nav2bmix's 7: 9 is one block's worth of expert layers, so
        # _span_sweep's disjoint spans land on [0,9,18,27] and each one grades exactly one
        # block's 9-from-7 reconstruction. Other values straddle blocks (kd_model warns).
        # SPEED: m=9 < SPAN_CKPT_MIN=14 so the span chain is UNCHECKPOINTED; set SPAN_CKPT_MIN=9
        # if the +29% from 28 -> 36 expert layers pushes it into OOM, and expect ~4x slower.
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.layer_mix=true
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix="${MIXM:-9}"
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}")
        # RUN_TAG derives from ARM alone, so the mix parameters must be in the name or two
        # configurations would share an output_dir and a wandb id (job 493).
        ARM="${ARM}_m${MIXM:-9}w${MIXW:-1.0}"
        # PLR overrides the mixing matrices' LR multiplier (config ships 10.0).
        # ⚠️ MEASURED: at 10.0 P is effectively FROZEN -- over a full 5-epoch run the most
        # active weight moved 3.3 percentage points (0.871 -> 0.838) and the two banks stayed
        # identical, so job 602's -27.1% min_ade came from a FIXED sharpened tent, not from a
        # learned mixture. Adam steps ~lr per parameter regardless of gradient magnitude, and
        # the oracle's logit-travel estimate put the multiplier needed to actually move P at
        # 80-110x. Nothing else may change alongside it, or the comparison against 602 stops
        # isolating what learning P is worth.
        # ⚠️ The tag is appended ONLY when PLR is set, so ARM=mix2bnav keeps resolving to
        # job 602's output_dir and its checkpoints stay evaluable.
        if [[ -n "${PLR:-}" ]]; then
            EXTRA+=(++trainer.lr_multiplier.layer_mixer="$PLR")
            ARM="${ARM}_plr${PLR}"
        fi ;;
    nav2bmix)
        # 2B student, 2 front cameras, NAV-CONDITIONED, on the teacher-forced block loss (m=1)
        # AND the m=7 span loss at EQUAL weight (block_span_mix=7, weight 1.0 -> weighted mean).
        # ⚠️ t0 is per-ANNOTATION (116 distinct values, event-anchored), NOT the 5.1 s keyframe,
        # so NOTHING here is comparable to the arms that use the default -- this run needs its
        # own no-nav control on the same annotations before its number means anything.
        # ⚠️ The route reaches BOTH towers: they read one input_ids tensor, and the config puts
        # "route" in components_order (CameraSubsetPAIDataset raises if it is missing).
        # SPEED: m=7 < SPAN_CKPT_MIN=14, so the span chain is UNCHECKPOINTED (4.3x measured);
        # the mix still costs ~2x block-only because it runs both sweeps.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_nav_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix="${MIXM:-7}"
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}")
        ARM="${ARM}_m${MIXM:-7}w${MIXW:-1.0}" ;;
    nav4bmix)
        # The 4B counterpart of nav2bmix: SAME objective (teacher-forced block loss m=1 mixed
        # with the span loss at equal weight), SAME nav-conditioned annotations, SAME effective
        # batch -- the ONLY intended difference is the student and the span length.
        # ⚠️ m=9, not 7. The 2B student runs a PRUNED 28-layer expert, so its span of 7 covers
        # a quarter of the depth; the 4B student keeps all 36 expert layers, and 36/4 = 9 is
        # the span that holds that same fraction. Copying 7 across would silently change the
        # objective's reach, which is the thing being held fixed.
        # ⚠️ NO PRUNE_EXPERT_LAYERS here. The 4B student's text tower already matches the
        # expert's 36 layers, so pruning would create the depth mismatch the 2B arm uses it
        # to avoid.
        # SPEED: m=9 is still < SPAN_CKPT_MIN=14, so the span chain is UNCHECKPOINTED, same
        # regime as the 2B run -- but on 36 layers rather than 28, so budget ~1.3x its cost
        # per step on top of the larger student.
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_nav_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix="${MIXM:-9}"
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}")
        # RUN_TAG derives from ARM alone, so the mix parameters must be in the name or two
        # configurations share an output_dir and a wandb id (job 493).
        ARM="${ARM}_m${MIXM:-9}w${MIXW:-1.0}" ;;
    nav4bmix2cam)
        # nav4bmix restricted to the TWO FRONT cameras -- the corrected counterpart of the
        # 4-camera run (job 20550), and the only 4B arm directly comparable to nav2bmix.
        # Identical objective, annotations, warmup, epochs and EFFECTIVE batch; the camera
        # set is the single difference from nav4bmix, and the student+span are the single
        # difference from nav2bmix.
        # ⚠️ The camera subsetting lives in the CONFIG (CameraSubsetPAIDataset), not here.
        # nav4bmix's config targets PAIDatasetWithNav, which is a plain PAIDataset plus nav
        # and therefore yields all four cameras -- that is exactly how job 20550 trained on
        # 16 images while being described as 2-camera. Verify from the log: this arm MUST
        # print "[camsubset] ... cameras [1, 3] (2 x 4 frames = 8 images)".
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_2cam_nav_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix="${MIXM:-9}"
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}")
        ARM="${ARM}_m${MIXM:-9}w${MIXW:-1.0}" ;;
    nav4bspan2camall)
        # Requested production run: Qwen3-VL-4B, front-wide + front-telephoto, every anchor,
        # direction-only route text, and one span objective per epoch. This is ONE four-epoch
        # Trainer run so Adam and the cosine LR schedule remain continuous across stage changes.
        # Random t is one Beta-schedule draw per sample; the loss evaluates one noisy state,
        # never a multi-step denoising rollout. The VLM prompt uses inference/generation mode:
        # it ends at <traj_future_start> and never contains ground-truth future tokens. The raw
        # future trajectory remains in the batch only to construct the noisy denoising state.
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_2cam_nav_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        ZIP_CACHE_GB=96
        [[ -n "$BS" ]] || BS=8        # proven by completed H200 job 20600 on this exact stack
        [[ -n "$ACCUM" ]] || ACCUM=1 # global/effective batch = 4 x 8 = 32
        [[ -n "$WARMUP" ]] || WARMUP=733  # 21.3% of 3438 steps/epoch for 109,997 rows
        # /temp is NFS. Job 20687 used 8 workers/rank with prefetch_factor=4: 128 batches
        # were eligible to be fetched concurrently and 7-8 workers/rank sat in D-state.
        # Reducing job 20688 to 4x2 removed that NFS storm, but ordered delivery exposed one
        # slow worker every fourth step: three ~3 s iterations followed by a 33-42 s stall.
        # Grouping chunk -> clip -> anchor lets one worker copy each immutable ZIP sequentially
        # into the job-scoped RAM cache; every rank then decodes its anchors locally. Per-file
        # locks prevent the 16 workers from duplicating a cache miss, while a 96 GiB LRU bound
        # keeps enough prefetched chunks hot without filling /dev/shm. Four workers/rank remains
        # the measured non-thrashing point; the cache removes NFS reads from their hot path.
        [[ -n "${WORKERS:-}" ]] || WORKERS=4
        export KAVA_DATALOADER_IN_ORDER=0
        # A transient NFS read wedged one rank in jobs 20700 and 20706. Fail the empty
        # DataLoader queue after two minutes, then let the bounded launcher loop restore the
        # newest complete checkpoint. Neither setting changes the successful-batch hot path.
        export KAVA_DATALOADER_TIMEOUT_SECONDS="${KAVA_DATALOADER_TIMEOUT_SECONDS:-120}"
        export KAVA_IO_GROUPED_SAMPLER=1
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix=0
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                # Eight evenly spaced saves over four epochs: one at each midpoint and one
                # at each epoch boundary. A ratio keeps this cadence correct if batch size
                # changes; for the default 13,752-step run it resolves to every 1,719 steps.
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=2
                ++trainer.dataloader_persistent_workers=true)
        # The I/O policy is part of the run identity. Cancelled jobs 20687/20688 left W&B
        # state in the older directories; a new namespace guarantees a clean run at step 0.
        ARM="${ARM}_m1-9-18-36_io4x2ooo_locality_ramcache96" ;;
    nav4bspan2camallfc)
        # nav4bspan2camall again, reading the PRE-MATERIALISED frame cache instead of the
        # dataset ZIPs. Same student, cameras, manifest, objective, batch and schedule -- ONLY
        # where the pixels come from changes, which is what keeps the two comparable.
        #
        # WHY. Job 20710 measured the live loader at 1.64 TiB of NFS per EPOCH to deliver
        # 879,976 images: whole 1.2 GiB chunk ZIPs for the ~49% of clips the manifest wants,
        # whole 604-frame clips for the ~27 frames it uses, at 1920x1080 for a ViT that sees
        # 576x320. /temp gives ~120 MB/s (already nconnect=8), so that IS ~4 h/epoch of wire
        # time -- a 3.0 s median step against a 2.0 s GPU floor, and three crashes when a 1 GiB
        # copy outran the 120 s dataloader timeout. scripts/build_frame_cache.py wrote those
        # 879,976 images once (~84 GiB, x264 crf18); a step now reads ~2.7 MB.
        # tests/test_frame_cache_equivalence.py holds the substitution honest: prompt text and
        # image_grid_thw identical to the live loader, ego tensors identical, pixels within
        # 0.011 mean |delta| of the source after the processor's downscale.
        #
        # So everything the old arm needed to SURVIVE NFS is gone, not retuned: no ZIP cache,
        # no locality sampler, no out-of-order delivery, no dataloader timeout. Shuffling is
        # free now, and mixes batches strictly better than chunk-grouped sampling did.
        #
        # Deliberately NOT staged into /dev/shm: at ~84 GiB the page cache holds the whole
        # working set inside the 640 GiB cgroup, reclaimably, and warms itself during epoch 1.
        # Staging would spend the same bytes on UNRECLAIMABLE tmpfs -- the accounting that made
        # the old arm fragile in the first place.
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_2cam_nav_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        FRAME_CACHE="${FRAME_CACHE:-/temp/achahe/physical_ai_av/framecache_nav2cam_1080p}"
        [[ -f "$FRAME_CACHE/_index.json" ]] || {
            echo "[slurm] no frame-cache index at $FRAME_CACHE/_index.json." >&2
            echo "        Build it:  sbatch slurm_build_frame_cache.sh" >&2
            echo "        Then:      build_frame_cache.py --out $FRAME_CACHE --finalize" >&2
            exit 1; }
        [[ -n "$BS" ]] || BS=8          # unchanged from nav4bspan2camall, so steps line up
        [[ -n "$ACCUM" ]] || ACCUM=1    # global/effective batch = 4 x 8 = 32
        [[ -n "$WARMUP" ]] || WARMUP=733
        # Decoding two 4-frame mini-clips is far cheaper than seeking 1080p video inside a
        # multi-GB ZIP, and there is no NFS storm left to provoke, so workers are cheap.
        [[ -n "${WORKERS:-}" ]] || WORKERS=8
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.frame_cache_root="$FRAME_CACHE"
                ++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix=0
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                # ⚠️ 0.125 of the run = 1,719 steps = HALF AN EPOCH, and that alignment is
                # the point, not the interval. 3,438/6,876/10,314/13,752 are the epoch
                # boundaries, so these 8 saves land on all four of them plus the four
                # midpoints -- and save_total_limit=8 then keeps exactly that set.
                # A "cheaper restart" interval of 500 divides none of those boundaries and
                # prunes to the last 8 (10,500..13,752), which is how job 20724 finished with
                # NO epoch-1/2/3 checkpoint at all. Per-epoch comparison needs them; a longer
                # rollback on the rare crash is the cheaper of the two costs.
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=4
                ++trainer.dataloader_persistent_workers=true)
        ARM="${ARM}_m1-9-18-36_framecache1080p" ;;
    nav4bspan4camallfc)
        # nav4bspan2camall again, reading the PRE-MATERIALISED frame cache instead of the
        # dataset ZIPs. Same student, cameras, manifest, objective, batch and schedule -- ONLY
        # where the pixels come from changes, which is what keeps the two comparable.
        #
        # WHY. Job 20710 measured the live loader at 1.64 TiB of NFS per EPOCH to deliver
        # 879,976 images: whole 1.2 GiB chunk ZIPs for the ~49% of clips the manifest wants,
        # whole 604-frame clips for the ~27 frames it uses, at 1920x1080 for a ViT that sees
        # 576x320. /temp gives ~120 MB/s (already nconnect=8), so that IS ~4 h/epoch of wire
        # time -- a 3.0 s median step against a 2.0 s GPU floor, and three crashes when a 1 GiB
        # copy outran the 120 s dataloader timeout. scripts/build_frame_cache.py wrote those
        # 879,976 images once (~84 GiB, x264 crf18); a step now reads ~2.7 MB.
        # tests/test_frame_cache_equivalence.py holds the substitution honest: prompt text and
        # image_grid_thw identical to the live loader, ego tensors identical, pixels within
        # 0.011 mean |delta| of the source after the processor's downscale.
        #
        # So everything the old arm needed to SURVIVE NFS is gone, not retuned: no ZIP cache,
        # no locality sampler, no out-of-order delivery, no dataloader timeout. Shuffling is
        # free now, and mixes batches strictly better than chunk-grouped sampling did.
        #
        # Deliberately NOT staged into /dev/shm: at ~84 GiB the page cache holds the whole
        # working set inside the 640 GiB cgroup, reclaimably, and warms itself during epoch 1.
        # Staging would spend the same bytes on UNRECLAIMABLE tmpfs -- the accounting that made
        # the old arm fragile in the first place.
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_4cam_nav_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        FRAME_CACHE="${FRAME_CACHE:-/temp/achahe/physical_ai_av/framecache_nav4cam_1080p}"
        [[ -f "$FRAME_CACHE/_index.json" ]] || {
            echo "[slurm] no frame-cache index at $FRAME_CACHE/_index.json." >&2
            echo "        Build it:  sbatch slurm_build_frame_cache.sh" >&2
            echo "        Then:      build_frame_cache.py --out $FRAME_CACHE --finalize" >&2
            exit 1; }
        # ⚠️ BS=4/ACCUM=2, NOT the 2-camera arm's 8/1. Four cameras is 16 images per sample
        # instead of 8, so the vision activations roughly double. The effective batch is
        # held at 4 x 4 x 2 = 32, which is what keeps warmup 733 and 3,438 steps/epoch
        # identical to every other run -- gradient accumulation is exactly equivalent for
        # a mean-reduced loss, so only memory and step timing change.
        [[ -n "$BS" ]] || BS=4
        [[ -n "$ACCUM" ]] || ACCUM=2    # effective batch = 4 GPUs x 4 x 2 = 32
        [[ -n "$WARMUP" ]] || WARMUP=733
        # Decoding two 4-frame mini-clips is far cheaper than seeking 1080p video inside a
        # multi-GB ZIP, and there is no NFS storm left to provoke, so workers are cheap.
        [[ -n "${WORKERS:-}" ]] || WORKERS=8
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.frame_cache_root="$FRAME_CACHE"
                ++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix=0
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                # ⚠️ 0.125 of the run = 1,719 steps = HALF AN EPOCH, and that alignment is
                # the point, not the interval. 3,438/6,876/10,314/13,752 are the epoch
                # boundaries, so these 8 saves land on all four of them plus the four
                # midpoints -- and save_total_limit=8 then keeps exactly that set.
                # A "cheaper restart" interval of 500 divides none of those boundaries and
                # prunes to the last 8 (10,500..13,752), which is how job 20724 finished with
                # NO epoch-1/2/3 checkpoint at all. Per-epoch comparison needs them; a longer
                # rollback on the rare crash is the cheaper of the two costs.
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=4
                ++trainer.dataloader_persistent_workers=true)
        ARM="${ARM}_m1-9-18-36_framecache4cam1080p" ;;
    nav4bspan2camallfc_cachenorm)
        # nav4bspan2camallfc with EXACTLY ONE variable changed: block_norm teacher -> cache.
        # Its own output_dir and run_name, so epoch 1 (checkpoint-3438) is paired against the
        # parent's checkpoint-3438 with only the normaliser differing.
        #
        # WHY. Measured in block_losses.py: driving a block with a ZERO cache scores 0.0104,
        # so 99% of ||y_teacher||^2 is cache-INDEPENDENT residual stream -- a component the
        # student cannot get wrong. The informative band is 0..0.0104 and a trained student
        # sits at 0.0012, i.e. 88% already captured with the whole remaining gap in the last
        # 12%. That is why "500 steps moved this loss within noise while ade moved -13% at
        # z=-3.44". block_norm=cache divides by ||y_teacher - y_zero||^2 instead: zero-cache
        # scores exactly 1.0, the model ~0.115, and layers are weighted by how much the cache
        # actually controls them rather than by residual magnitude.
        #
        # ⚠️ Adam is per-parameter scale-invariant, so this changes CROSS-LAYER weighting and
        # the readability of the curve, NOT the gradient direction within a layer. COMPARE_EVAL
        # §4 found three hand-designed cross-layer reweightings all NEGATIVE, so the honest
        # prior is "better diagnostic, unproven accuracy". Read the curve before the eval.
        #
        # ⚠️ COSTS AN EXTRA ZERO-CACHE FORWARD per step (y_zero, kd_model.py:879-887) on top of
        # the teacher-forced pass. Budget above the parent's ~3.4 h/epoch.
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_2cam_nav_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        FRAME_CACHE="${FRAME_CACHE:-/temp/achahe/physical_ai_av/framecache_nav2cam_1080p}"
        [[ -f "$FRAME_CACHE/_index.json" ]] || {
            echo "[slurm] no frame-cache index at $FRAME_CACHE/_index.json." >&2
            echo "        Build it:  sbatch slurm_build_frame_cache.sh" >&2
            exit 1; }
        [[ -n "$BS" ]] || BS=8
        [[ -n "$ACCUM" ]] || ACCUM=1    # 4 GPUs x 8 x 1 = 32, identical to the parent
        [[ -n "$WARMUP" ]] || WARMUP=733
        [[ -n "${WORKERS:-}" ]] || WORKERS=8
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.frame_cache_root="$FRAME_CACHE"
                ++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=cache ++model.kd.block_span=1
                ++model.kd.block_span_mix=0
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=4
                ++trainer.dataloader_persistent_workers=true)
        ARM="${ARM}_m1-9-18-36_framecache1080p" ;;

    nav4bspan2camallfc_freerun)
        # nav4bspan2camallfc with EXACTLY ONE variable changed: block_freerun_weight 0 -> 2e-4 (see SMOKE RESULT below).
        # block_norm stays 'teacher' so this is NOT confounded with the cachenorm twin; the two
        # run in parallel on separate output_dirs and are each paired against the same parent.
        #
        # WHY. freerun_probe.py (n=32): the teacher-forced error L_block trains on is FLAT at
        # ~4.1e-4 across all 36 layers, while the free-running error -- the student expert
        # consuming its OWN chain, which is what inference does -- grows to 1.3e-2. That is 32x
        # overall and 72x at the deepest layers. L_block has no gradient path to it; this term
        # supplies one. The span schedule is NOT a substitute: COMPARE_EVAL §3 measured the
        # objective m-INVARIANT (loss FALLS 0.00170 -> 0.00093 as spans deepen, peak memory flat
        # at 65.8 GiB from m=1 to m=28) because the expert layer map is contractive.
        #
        # ⚠️ BLOCK_FR_FP32=1 IS BROKEN -- DO NOT SET IT. kd_model.py:1472-1483 casts the
        # activations to fp32 while DISABLING autocast, but the frozen expert's weights stay
        # bf16, so the first q_proj raises
        #     RuntimeError: expected mat1 and mat2 to have the same dtype, float != BFloat16
        # at modeling_qwen3_vl.py:428, on every rank, at step 0. Job 20829 died on it four
        # times through the restart loop. The flag appears in no other script or doc, so the
        # fp32 opt-in has never run end-to-end on this stack.
        # Casting the whole expert to fp32 is NOT the fix: L_block runs OUTSIDE autocast
        # against bf16 weights, so that would silently change the parent-comparable term too.
        # A correct fix needs a SEPARATE fp32 copy of the expert for this chain only.
        #
        # So this arm runs the free-running chain in bf16, which kd_model.py:1464-1471 warns
        # produced a non-finite gradient on the first backward under plain zero2 (grad_norm
        # pinned at the 2.0 sentinel at step 0, nan at step 1). That warning is now UNVERIFIED
        # on this stack -- SMOKE=1 is the 20-step check, and is cheap. Run it before the full
        # 4-epoch job:
        #     SMOKE=1 APPROVED=1 ARM=nav4bspan2camallfc_freerun sbatch slurm_train_kd.sh
        #
        # SMOKE RESULT (job 20830, 20 steps, 158 s): bf16 is FINE -- no nan, no 2.0 sentinel,
        # so the kd_model.py warning does not hold on this stack and no fp32 path is needed.
        # But it also showed weight=1.0 is WRONG BY ~3 ORDERS OF MAGNITUDE here:
        #     freerun_loss ~89-190   block_loss ~0.083   ->  freerun is 99.94% of the total
        # and over 20 steps freerun was FLAT (89,143,106,130,114,143,190,130) while block_loss
        # ROSE 0.0789 -> 0.0893, with grad_norm swinging 2.5k-9.8k (cachenorm arm: ~1.2-1.6k).
        # The docstring's "L_block keeps ~4% of the gradient at weight 1.0" is not true here.
        #
        # ⚠️ WHY THAT MATTERS: `tgt = captured[n_layers-1]["y_out"]` -- freerun matches ONLY the
        # FINAL layer's output after chaining all 36 blocks, i.e. it is an ENDPOINT-ONLY
        # objective, structurally the same as L_field. COMPARE_EVAL §4a measured L_field alone
        # at 7.8x WORSE than L_block and worse than the CE-only and KD-only floors, because
        # intermediates drift freely while block_loss rises 25.6x. Rising block_loss under a
        # dominant endpoint term is that exact signature, already visible by step 16.
        #
        # So weight=0.0002 keeps L_BLOCK DOMINANT (~75%) and uses freerun as the auxiliary that
        # supplies the compounding gradient L_block cannot see. That is the `blockfield`
        # combination COMPARE_EVAL §9 lists as never having run past checkpoint-500.
        # ⚠️ COST: 7.9 s/step vs the parent's 3.5 s (job 20830) -> budget ~7.5 h/epoch.
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_2cam_nav_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        FRAME_CACHE="${FRAME_CACHE:-/temp/achahe/physical_ai_av/framecache_nav2cam_1080p}"
        [[ -f "$FRAME_CACHE/_index.json" ]] || {
            echo "[slurm] no frame-cache index at $FRAME_CACHE/_index.json." >&2
            echo "        Build it:  sbatch slurm_build_frame_cache.sh" >&2
            exit 1; }
        [[ -n "$BS" ]] || BS=8
        [[ -n "$ACCUM" ]] || ACCUM=1    # 4 GPUs x 8 x 1 = 32, identical to the parent
        [[ -n "$WARMUP" ]] || WARMUP=733
        [[ -n "${WORKERS:-}" ]] || WORKERS=8
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.frame_cache_root="$FRAME_CACHE"
                ++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix=0
                ++model.kd.block_freerun_weight=0.0002
                ++model.kd.block_freerun_layers=0
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=4
                ++trainer.dataloader_persistent_workers=true)
        ARM="${ARM}_m1-9-18-36_framecache1080p" ;;
    nav4bspanmix2camallfc)
        # nav4bspan2camallfc's twin, ADDING the teacher-forced term back instead of replacing
        # it. The plain-span arm swaps objectives at each boundary -- m=1, then m=9, then 18,
        # then 36 -- so after epoch 1 the per-layer signal is gone. This arm keeps m=1 in every
        # epoch and rides the scheduled span alongside it (weighted MEAN, block_span_mix_weight):
        #
        #   epoch 1  block_span_mix=1   mix degenerate -> teacher-forced m=1 alone
        #   epoch 2  block_span_mix=9   teacher-forced m=1 + span m=9
        #   epoch 3  block_span_mix=18  teacher-forced m=1 + span m=18
        #   epoch 4  block_span_mix=36  teacher-forced m=1 + span m=36
        #
        # Epoch 1 is therefore IDENTICAL to nav4bspan2camallfc's epoch 1, which is what makes
        # the two arms a clean A/B on "does the span add anything ON TOP of m=1, or is it
        # better as a replacement?" -- the question kd_model's mix branch was written to ask.
        # ⚠️ block_span stays 1: under a mix schedule it is what the teacher-forced term reads,
        # and the callback rejects any other value rather than silently changing that meaning.
        # Everything else is held to nav4bspan2camallfc exactly -- manifest, nav format, frame
        # cache, effective batch 32, warmup 733, epoch-aligned saves -- so the only difference
        # between the two runs is whether m=1 survives past epoch 1.
        MODEL_TAG=4b
        CONFIG_NAME=sft_kd_qwen3_4b_2cam_nav_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        FRAME_CACHE="${FRAME_CACHE:-/temp/achahe/physical_ai_av/framecache_nav2cam_1080p}"
        [[ -f "$FRAME_CACHE/_index.json" ]] || {
            echo "[slurm] no frame-cache index at $FRAME_CACHE/_index.json." >&2
            echo "        Build it:  sbatch slurm_build_frame_cache.sh" >&2
            exit 1; }
        [[ -n "$BS" ]] || BS=8
        [[ -n "$ACCUM" ]] || ACCUM=1    # effective batch = 4 x 8 = 32, as the twin arm
        [[ -n "$WARMUP" ]] || WARMUP=733
        [[ -n "${WORKERS:-}" ]] || WORKERS=8
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.frame_cache_root="$FRAME_CACHE"
                ++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix=1
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}"
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.target=block_span_mix
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                # 0.125 = 1,719 = half an epoch, so the 8 saves land on all four epoch
                # boundaries plus the midpoints and save_total_limit=8 keeps the lot.
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=4
                ++trainer.dataloader_persistent_workers=true)
        ARM="${ARM}_m1-9-18-36mix_w${MIXW:-1.0}_framecache1080p" ;;
    mixspanmix2bnavfc)
        # The FOURTH cell of the 2x2: {4B, 2B} x {span REPLACES m=1, span rides ALONGSIDE m=1}.
        # This is mixspan2bnavfc's twin -- same 2B layer-mix student, same everything -- with
        # the schedule driving block_span_mix instead of block_span, exactly as
        # nav4bspanmix2camallfc is nav4bspan2camallfc's twin:
        #
        #   epoch 1  block_span_mix=1   mix degenerate -> teacher-forced m=1 alone
        #   epoch 2  block_span_mix=9   teacher-forced m=1 + span m=9
        #   epoch 3  block_span_mix=18  teacher-forced m=1 + span m=18
        #   epoch 4  block_span_mix=36  teacher-forced m=1 + span m=36
        #
        # Epoch 1 is therefore identical to mixspan2bnavfc's epoch 1, so the pair isolates the
        # objective change at the 2B scale the same way the 4B pair does at 4B.
        # ⚠️ block_span stays 1: under a mix schedule it is what the teacher-forced term reads.
        # ⚠️ m=36 is legal here ONLY because the span is bounded by the MIXER's 36 synthesised
        # slots, not the 28-layer student or the 28-layer expert module. Getting that wrong is
        # what killed job 20777 at its epoch-4 boundary; see BlockSpanScheduleCallback.
        # ⚠️ Spans of 18 and 36 straddle the mixer's 4 blocks of 9, so from epoch 3 the span
        # term stops grading one block's 9-from-7 reconstruction at a time. That is inherent to
        # putting this schedule on a mixed student and is equally true of mixspan2bnavfc -- the
        # two arms remain comparable to each other, which is the point.
        unset PRUNE_EXPERT_LAYERS
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_nav_layermix_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        FRAME_CACHE="${FRAME_CACHE:-/temp/achahe/physical_ai_av/framecache_nav2cam_1080p}"
        [[ -f "$FRAME_CACHE/_index.json" ]] || {
            echo "[slurm] no frame-cache index at $FRAME_CACHE/_index.json." >&2
            echo "        Build it:  sbatch slurm_build_frame_cache.sh" >&2
            exit 1; }
        # BS=8/ACCUM=1 measured at ~65 GB/GPU on the H200 for this exact stack (job 20776), so
        # the twin's conservative BS=2/ACCUM=4 is unnecessary. Effective batch is 32 either way.
        [[ -n "$BS" ]] || BS=8
        [[ -n "$ACCUM" ]] || ACCUM=1    # 4 GPUs x 8 x 1 = 32
        [[ -n "$WARMUP" ]] || WARMUP=733
        [[ -n "${WORKERS:-}" ]] || WORKERS=8
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.frame_cache_root="$FRAME_CACHE"
                ++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.layer_mix=true
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix=1
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}"
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.target=block_span_mix
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                # 0.125 = 1,719 = half an epoch: the 8 saves land on all four epoch boundaries
                # plus the midpoints, and save_total_limit=8 keeps exactly that set.
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=4
                ++trainer.dataloader_persistent_workers=true)
        ARM="${ARM}_m1-9-18-36mix_w${MIXW:-1.0}_framecache1080p" ;;
    mixspan2bnavfc)
        # The 4B's CURRICULUM objective on the 2B LAYER-MIX student. Deliberately NOT mix2bnav:
        # that arm runs the simultaneous pair (teacher-forced m=1 PLUS span m=9, weighted mean)
        # every step, while this one runs ONE span per epoch, [1, 9, 18, 36], exactly as
        # nav4bspan2camallfc did. block_span_mix=0 is what selects the schedule over the mix;
        # BlockSpanScheduleCallback RAISES if it is left >1, so the two cannot be run by accident.
        #
        # Held identical to the 4B run so the 2B and 4B students are comparable end to end:
        # same manifest (anchors_all, 109,997), same nav format (distance stripped, so the
        # model-visible route is "<|route_start|>Turn left<|route_end|>"), same generation-mode
        # prompt ending at <traj_future_start>, same effective batch 32 -> 3,438 steps/epoch,
        # 13,752 total, warmup 733 = 21.3% of one epoch. ⚠️ That warmup is only correct BECAUSE
        # the effective batch is 32 on this manifest; the config's shipped 333 is for the 50k
        # manifest and must not be reused here.
        #
        # ⚠️ BS=2/ACCUM=4, not the 4B's BS=8/ACCUM=1, for the same effective 32. The config
        # measured 38.6 GB at bs=1 and projects ~66 GB at bs=2 for THIS stack -- a 2B student
        # carrying a 36-layer expert whose m=9 span chain is UNCHECKPOINTED (SPAN_CKPT_MIN=14).
        # Extrapolating that line puts bs=8 near 230 GB, well past even an H200. Gradient
        # accumulation is mathematically equivalent here, so only memory and step timing change.
        # Peak is epoch 2 (m=9, uncheckpointed); epochs 3-4 checkpoint the chain and cost less.
        unset PRUNE_EXPERT_LAYERS
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_nav_layermix_lcdrive
        PREFLIGHT_MANIFEST=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_all.json
        REQUIRE_APPROVAL=1
        # The 4B's cache serves this run unchanged: cameras [1,3] match, and the 50k manifest
        # this config ships is a strict subset of anchors_all (verified: 0 anchors and 0 clips
        # outside it), so every (clip, t0) the loader asks for is already materialised.
        FRAME_CACHE="${FRAME_CACHE:-/temp/achahe/physical_ai_av/framecache_nav2cam_1080p}"
        [[ -f "$FRAME_CACHE/_index.json" ]] || {
            echo "[slurm] no frame-cache index at $FRAME_CACHE/_index.json." >&2
            echo "        Build it:  sbatch slurm_build_frame_cache.sh" >&2
            exit 1; }
        [[ -n "$BS" ]] || BS=2
        [[ -n "$ACCUM" ]] || ACCUM=4    # 4 GPUs x 2 x 4 = 32, matching the 4B run
        [[ -n "$WARMUP" ]] || WARMUP=733
        [[ -n "${WORKERS:-}" ]] || WORKERS=8
        [[ -n "$AUTO_RESUME_LATEST" ]] || AUTO_RESUME_LATEST=1
        [[ -n "$MAX_RESTARTS" ]] || MAX_RESTARTS=3
        EXTRA+=(++data.train_dataset.frame_cache_root="$FRAME_CACHE"
                ++data.train_dataset.annotations_path="$PREFLIGHT_MANIFEST"
                ++data.train_dataset.strip_nav_turn_distance=true
                ++data.train_dataset.vla_preprocess_args.generation_mode=true
                ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.layer_mix=true
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix=0
                ++callbacks.block_span_schedule._target_=alpamayo1_5_distill.callbacks.BlockSpanScheduleCallback
                "++callbacks.block_span_schedule.spans=[1,9,18,36]"
                ++callbacks.block_span_schedule.strict_num_train_epochs=true
                ++trainer.num_train_epochs=4
                # ⚠️ 0.125 = 1,719 steps = HALF AN EPOCH. The alignment is the point: these 8
                # saves land on all four epoch boundaries (3,438/6,876/10,314/13,752) plus the
                # four midpoints, and save_total_limit=8 keeps exactly that set. An interval of
                # 500 divides none of the boundaries and prunes to the last 8, which is how the
                # 4B run finished with only its epoch-4 checkpoint -- useless for the per-epoch
                # 2B-vs-4B comparison this arm exists to enable.
                ++trainer.save_strategy=steps ++trainer.save_steps=0.125
                ++trainer.save_total_limit=8
                ++trainer.dataloader_prefetch_factor=4
                ++trainer.dataloader_persistent_workers=true)
        ARM="${ARM}_m1-9-18-36_framecache1080p" ;;
    field2b)
        # L_FIELD ALONE on the 2B/pruned-expert 2-camera stack: chain all 28 layers on the
        # STUDENT's own cache (no teacher forcing anywhere), take expert.norm +
        # action_out_proj, and MSE the VELOCITY against the teacher's, at a random t.
        # Never run before -- L_field only ever ran ALONGSIDE L_block (arm blockfield, which
        # itself never got past checkpoint-500 and was never evaluated).
        # WHY: BLOCK_ODE measured the velocity at 30% RMS error while the hidden states
        # L_block matches are only 3.2% off per layer (PRUNING.md:353) -- the quantity the
        # trajectory integrates is 10x more wrong than the one four epochs optimised.
        # AGAINST: L_field's gradient reaches shallow layers only through the contractive deep
        # half, so it may under-train them -- and ladder_add showed under-training early layers
        # hurts even though the ladder says the EXPERT ignores them (the VLM compounds forward).
        # The two arguments point opposite ways, which is why this is measured not argued.
        # Matched control: the block-only epoch 1 from scratch, min_ade 2.8367 / ade 5.8744.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=0.0 ++model.kd.block_timestep=beta
                ++model.kd.field_weight="${FIELD_W:-1.0}") ;;
    block2bmix)
        # BOTH block objectives: teacher-forced per-layer (m=1) + span(m=7), weighted mean.
        # Rationale: alone, the span term is nearly m-INVARIANT (0.0017 at m=1 -> 0.00093 at
        # m=28 on the weights it inherited) and the four-stage curriculum was a null result
        # (min_ade 2.8367 -> 2.3760, last stage z=-1.90 / +0.63). Layer REWEIGHTING was worse
        # still (+0.1487 and +0.1787 vs uniform). This asks the remaining question about the
        # span term: does chaining add anything ON TOP of per-layer teacher forcing, rather
        # than instead of it. Same 2-camera stack; block_span stays 1 so _sweep is the m=1
        # path bit-identical to every prior block number.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_span_mix="${MIXM:-7}"
                ++model.kd.block_span_mix_weight="${MIXW:-1.0}")
        # the mix parameters go in the ARM name -- RUN_TAG derives from ARM alone, so two
        # configurations under one ARM would share an output_dir and a wandb id (job 493).
        ARM="${ARM}_m${MIXM:-7}w${MIXW:-1.0}" ;;
    block2bdepth)
        # DEPTH-WEIGHTED block loss, from the cache ladder rather than from a guess.
        # The span curriculum was a null result (min_ade 2.8367 -> 2.3760 over four epochs,
        # the last stage z=-1.90 on min_ade and +0.63 on ade) because spans probe compounding
        # ACROSS LAYERS, and the ladder showed that compounding is damped by construction:
        # substituting the teacher's cache into layers 0-9 moves min_ade +0.0120 against a
        # 0.034 noise floor, while layers 19-27 alone recover 95% of the -1.3452 gap. A
        # uniform per-layer mean therefore spends the student on layers the expert ignores.
        # Same 2-camera stack and span=1 as the curriculum's first stage, so the ONE
        # difference from the m=1 epoch (min_ade 2.8367 / ade 5.8744) is the weighting.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher ++model.kd.block_span=1
                ++model.kd.block_layer_weights="${BLW:-ladder}")
        # ⚠️ The WEIGHTING GOES IN THE ARM NAME. RUN_TAG is derived from ARM alone, so two
        # profiles under one ARM share an output_dir: the second run would overwrite the
        # first's checkpoint AND be handed its .wandb_id, appending a different objective's
        # curve onto a finished run. Caught mid-flight once (job 493 vs 489's jxtq7mk7).
        ARM="${ARM}_${BLW:-ladder}" ;;
    block2bspan)
        # The SPAN CURRICULUM: same 2-camera stack as block2b2cam, but its own output_dir.
        # ⚠️ That separation is not cosmetic. Reusing block2b2cam's dir meant (a) save_total_limit
        # would have pruned job 470's checkpoints -- including the 3597 that was already
        # evaluated -- and (b) wandb_utils found the previous run's state in the dir and RESUMED
        # run evdhdhs1, appending a from-scratch curriculum onto a finished 3-epoch run's curves.
        # A new ARM gives a clean checkpoint namespace and a fresh wandb run.
        # block_span is passed per stage via EXTRA_ARGS: 1 -> 7 -> 14 -> 28, one epoch each.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_norm=teacher) ;;
    block2b2cam)
        # 2B student + pruned expert, L_block with sampled t, but TWO cameras (front-wide +
        # telephoto, ~1577 tokens instead of 3073) and the ViT at the FULL learning rate.
        # See configs/sft_kd_cosmos2b_2cam_lcdrive.yaml for why both.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_2cam_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta) ;;
    blockfield)
        # L_block + L_field on the 2B/pruned-expert stack. L_field matches the VELOCITY
        # (action_out_proj output), which the BLOCK_ODE probe measured at 30% RMS error while
        # the hidden states L_block matches are only 3.2% off per layer -- the head reads one
        # narrow projection that the uniform hidden-state loss under-weights.
        # ⚠️ Runs ALONGSIDE L_block, not instead of it: L_field's gradient reaches shallow
        # layers only through the deep half, which is contractive (0.79 across layers 14-27),
        # so on its own it would under-train exactly where L_block is strongest.
        # FIELD_W calibrates the mix; 1.0 is a starting point, not a measured optimum.
        export PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34
        MODEL_TAG=2b
        CONFIG_NAME=sft_kd_cosmos2b_prunedexpert_lcdrive
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.field_weight="${FIELD_W:-1.0}") ;;
    blockfr)
        # L_block + L_freerun: keep the teacher-forced per-layer term AND add a term on the
        # student's OWN chain at the final layer, which is the only place compounding shows.
        # Measured (scripts/freerun_probe.py, n=32 train clips, student = blockrandt e3):
        #   teacher-forced error  flat ~4.1e-04 across all 36 layers
        #   free-running error    grows to 1.3e-02  -- 32x the mean, 72x at the deepest layers
        # That is why both block arms plateau at epoch 2: cutting the teacher-forced term 33%
        # over epochs 2-3 bought only -0.06 min_ade, because L_block has NO gradient path to
        # the compounded error. L_freerun supplies one.
        # ⚠️ Additive, not a replacement: L_block still shapes every layer, and dropping it
        # would leave a single 36-layer-deep gradient path with very diffuse credit.
        # Control: blockrandt at the SAME total epochs, same init (its e3 checkpoint).
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta
                ++model.kd.block_freerun_weight="${FR_W:-1.0}"
                # ⚠️ zero2_OFFLOAD, not the usual zero2. The free-running chain adds a
                # SECOND backward path into the student's cache and the plain zero2 step
                # already sat at ~75 GB of 79 -- it OOMed on a 7.5 GB allocation with 6.2 GB
                # free. Offloading the 4 B student's Adam states (~24 GB/GPU) to host RAM
                # buys the room; costs step time, not correctness.
                # ⚠️ NOT zero3: it builds the model under `zero.Init`, so parameters are
                # partitioned before `from_pretrained_vlm` loads its state_dict and the load
                # dies on "size mismatch for vlm.model.visual.patch_embed.proj.weight".
                # ⚠️ Offload needs DeepSpeed's CPUAdam C++ extension -> `ninja` must be in
                # the venv (installed 2026-08-15); without it: "DeepSpeedCPUAdam object has
                # no attribute ds_opt_adam".
                ++trainer.deepspeed=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/configs/deepspeed/zero2_offload.json) ;;
    blockrandt)
        # L_block with t SAMPLED from the teacher's own training schedule (Beta(1.5,1.0)
        # rescaled by 0.999), instead of pinned to t=0. Same cost per step -- one expert
        # forward either way -- so any difference is attributable to WHERE on the flow the
        # cache is supervised, not to extra compute.
        # Motivation: CKA on the frozen expert shows it transforms its representation most
        # around t~0.2-0.4 and least at t=1, while the original L_block supervised only t=0.
        # Control: `blockonly` at 1 epoch (stitched min_ade 2.0293), same init, schedule and
        # budget.
        # ⚠️ model.kd.* -- NOT model.*. Hydra's `++` CREATES a missing key instead of
        # erroring, so the wrong prefix silently leaves every default weight in place and
        # trains the all-objectives config under this arm's name.
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta) ;;
    blockonly)
        # L_block ALONE -- teacher-forced block-output matching, no CE, no KD, no L_KV.
        # Alone by design: every arm so far showed that adding objectives to a cache-matching
        # term HURTS the expert head (logit-KD +4.38, and dropping CE from cekv bought -0.13).
        # block_weight is nominally 1.0 and is NOT a tuning knob here: with a single loss and
        # max_grad_norm=1.0 clipping active on every step (observed pre-clip norms 5.6-126),
        # the gradient is renormalised to unit norm, so any uniform scaling of the sole loss
        # is erased. The magnitude lever is `learning_rate`.
        EXTRA+=(++model.kd.kd_weight=0.0 ++model.kd.ce_weight=0.0
                ++model.kd.kv_weight=0.0 ++model.kd.block_weight=1.0) ;;
    kvband)
        # kvonly with DEPTH-BANDED layer weights instead of uniform. Bands measured causally
        # by scripts/layer_importance.py (n=100): swapping one layer of the student's cache
        # for the teacher's and re-driving the frozen expert put 0.1% of the recoverable gain
        # in layers 0-11, 49% in 12-23, 105% in 24-35 (marginal, so they overcount).
        # Weights renormalised to mean 1.0 so the total loss scale is unchanged -- this is a
        # DIRECTION change, not a magnitude one.
        # ⚠️ Start from BASE, not from a kvonly checkpoint: the matched control is uniform
        # kvonly at 1 epoch (stitched min_ade 2.6313), same init, same schedule, same budget.
        EXTRA+=(++model.kd.kd_weight=0.0 ++model.kd.ce_weight=0.0
                "++model.kd.kv_layer_bands=[0.01,0.96,2.03]") ;;
    kvonly)
        # KV alone -- no CE, no KD. Asks whether the student needs token supervision at all
        # when the target is a cache read by the teacher's expert.
        # Safe for the stitched eval: `<|traj_future_start|>` is part of the PROMPT under
        # `components_prompt: [traj_future]`, not something the student must learn to emit,
        # which is why all four stitched arms logged zero "No <traj_future_start>" warnings.
        # The student's own TOKEN head will be destroyed -- that is the point of the arm, so
        # do not evaluate this one with slurm_eval_kd.sh, only slurm_eval_stitched.sh.
        # kv_weight stays 45.409 rather than being rescaled: AdamW normalises by the second
        # moment, so a uniform scaling of the only remaining loss barely moves the step size,
        # and holding it fixed keeps this arm comparable to the others.
        EXTRA+=(++model.kd.kd_weight=0.0 ++model.kd.ce_weight=0.0) ;;
    *)
        echo "[slurm] unknown ARM=$ARM (expected ce|kd|kv|cekv|kvonly|kvband|blockonly|blockrandt|blockfr|block2b|nav2bmix|nav4bmix|nav4bmix2cam|nav4bspan2camall|nav4bspan2camallfc|nav4bspan4camallfc|nav4bspan2camallfc_cachenorm|nav4bspan2camallfc_freerun|mixspan2bnavfc|nav4bspanmix2camallfc|mixspanmix2bnavfc)" >&2; exit 1 ;;
esac

# The requested run has a deliberate human gate. Running this file directly prints a real turn
# from the manifest after normalization and exits before touching CUDA. Submit only after the
# user has approved that exact model-visible route component.
if [[ "$REQUIRE_APPROVAL" == "1" ]]; then
    "$VENV/python" -m alpamayo1_5_distill.scripts.preview_nav_instruction "$PREFLIGHT_MANIFEST"
    if [[ "${APPROVED:-0}" != "1" ]]; then
        echo "[preflight] NOT STARTING TRAINING. After approval: APPROVED=1 sbatch $0"
        exit 0
    fi
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "[slurm] training must be launched with sbatch" >&2
    exit 1
}
MASTER_PORT=$((29560 + SLURM_JOB_ID % 20000))

ZIP_CACHE_DIR=""
if [[ "$ZIP_CACHE_GB" != "0" ]]; then
    ZIP_CACHE_DIR="/dev/shm/alpamayo-kd-${SLURM_JOB_ID}"
    mkdir -p "$ZIP_CACHE_DIR"
    cleanup_zip_cache() {
        # Only remove the job-scoped path constructed immediately above.
        if [[ "${ZIP_CACHE_DIR:-}" == "/dev/shm/alpamayo-kd-${SLURM_JOB_ID}" ]]; then
            rm -rf -- "$ZIP_CACHE_DIR"
        else
            echo "[cache] refusing to remove unexpected path: ${ZIP_CACHE_DIR:-<unset>}" >&2
        fi
    }
    trap cleanup_zip_cache EXIT
    EXTRA+=(++data.train_dataset.zip_cache_dir="$ZIP_CACHE_DIR"
            ++data.train_dataset.zip_cache_max_gb="$ZIP_CACHE_GB")
    echo "[cache] shared ZIP cache=$ZIP_CACHE_DIR max=${ZIP_CACHE_GB}GiB"
fi

# INIT=<ckpt>: start from an existing student instead of the base VLM. Distinct from
# RESUME, which also restores optimizer + scheduler state; INIT takes the WEIGHTS only,
# so a new objective gets a fresh schedule with warmup rather than a decayed LR.
if [[ -n "${INIT:-}" ]]; then
    [[ -d "$INIT" ]] || { echo "[slurm] no such INIT checkpoint: $INIT" >&2; exit 1; }
    EXTRA+=(++model.checkpoint_path="$INIT")
    echo "[slurm] INIT weights from $INIT (fresh optimizer + schedule)"
fi
RUN_TAG="$ARM"
if [[ -n "$RESUME" ]]; then
    [[ -z "$EPOCHS" ]] && { echo "[slurm] RESUME needs EPOCHS (total, including epochs already done)" >&2; exit 1; }
    [[ -d "$RESUME" ]] || { echo "[slurm] no such checkpoint: $RESUME" >&2; exit 1; }
    # ⚠️ A DIFFERENT output_dir, deliberately. sft_base ships `save_total_limit: 2`, so
    # continuing in place would delete the very checkpoint being resumed from once two new
    # saves land -- and for kvonly that is the checkpoint behind the published 2.6313.
    RUN_TAG="${ARM}_e${EPOCHS}"
    # save_strategy=epoch (not the inherited save_steps=500) so there is exactly one
    # checkpoint per epoch, and save_total_limit high enough to keep all of them.
    EXTRA+=(++trainer.resume_from_checkpoint="$RESUME"
            ++trainer.num_train_epochs="$EPOCHS"
            ++trainer.save_strategy=epoch
            ++trainer.save_total_limit=10)
    echo "[slurm] RESUME from $RESUME -> total $EPOCHS epochs, one checkpoint each"
fi
# ⚠️ MODEL_TAG, not a hard-coded 4b: block2b trains the 2B student, and naming its output
# output_kd_4b_* would file it with the 4B arms it must never be compared against directly
# (different student, different expert depth).
if [[ "$SMOKE" == "1" ]]; then
    # Short, self-contained proof the path runs: teacher loads, sequences match, and all
    # three terms are finite AND falling. A term that is finite but flat is the failure
    # this recipe has to catch.
    RUN_TAG="${RUN_TAG}_smoke"
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=2
            ++trainer.save_strategy=no ++trainer.eval_strategy=no
            ++trainer.warmup_steps=0 ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids="0-120" ++data.val_dataset.chunk_ids="0-120")
    echo "[slurm] SMOKE mode: 20 steps"
fi

RUN_OUTPUT_DIR="$OUT_DIR/output_kd_${MODEL_TAG}_${RUN_TAG}_lcdrive"
EXTRA+=(paths.output_dir="$RUN_OUTPUT_DIR"
        "run_name=kd_${MODEL_TAG}_${RUN_TAG}_$(date +%m%d-%H%M)")
echo "[slurm] ARM=$ARM -> $OUT_DIR/output_kd_${MODEL_TAG}_${RUN_TAG}_lcdrive"

[[ -n "$BS"     ]] && EXTRA+=(trainer.per_device_train_batch_size="$BS")
[[ -n "$ACCUM"  ]] && EXTRA+=(trainer.gradient_accumulation_steps="$ACCUM")
[[ -n "$WARMUP" ]] && EXTRA+=(trainer.warmup_steps="$WARMUP")

# The cgroup exposes roughly half of --cpus-per-task, so the config's default worker count
# (sized for a bigger allocation) thrashes here. Derive it from the actual allocation.
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} / (3 * GPUS) ))}"
[[ "$WORKERS" -lt 2 ]] && WORKERS=2
EXTRA+=(++trainer.dataloader_num_workers="$WORKERS")

# Free-form hydra overrides, word-split. For one-off knobs that do not deserve a named
# variable -- e.g. EXTRA_ARGS='++trainer.ddp_timeout=5400' after a NCCL collective timeout.
# shellcheck disable=SC2206,SC2086
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

echo "[slurm] job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES workers=$WORKERS"
# Which PHYSICAL cards will the ranks actually use? Printed, not assumed -- this is the check
# that caught job 456 pointing at GPU 0.
"$VENV/python" -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'[slurm] rank{i} -> cuda:{i} uuid {getattr(torch.cuda.get_device_properties(i), \"uuid\", \"?\")}')
" 2>/dev/null || true
nvidia-smi -L | sed 's/^/[slurm] /' 

latest_complete_checkpoint() {
    local checkpoint name step
    local latest=""
    local latest_step=-1

    shopt -s nullglob
    for checkpoint in "$RUN_OUTPUT_DIR"/checkpoint-*; do
        [[ -d "$checkpoint" ]] || continue
        name="${checkpoint##*/}"
        step="${name#checkpoint-}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        [[ -f "$checkpoint/trainer_state.json" ]] || continue
        [[ -d "$checkpoint/global_step${step}" ]] || continue
        if (( step > latest_step )); then
            latest="$checkpoint"
            latest_step=$step
        fi
    done
    shopt -u nullglob
    printf '%s\n' "$latest"
}

run_training() {
    local resume_checkpoint="$1"
    local -a run_extra=("${EXTRA[@]}")
    if [[ -n "$resume_checkpoint" ]]; then
        run_extra+=(++trainer.resume_from_checkpoint="$resume_checkpoint")
        echo "[restart] resuming latest complete checkpoint: $resume_checkpoint"
    elif [[ "$AUTO_RESUME_LATEST" == "1" ]]; then
        echo "[restart] no complete checkpoint found; starting from configured initialization"
    fi
    "${LAUNCH[@]}" "$VENV/torchrun" \
        --nproc_per_node "$GPUS" \
        --master_port "$MASTER_PORT" \
        -m alpamayo1_5_distill.train_kd \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name "${CONFIG_NAME:-sft_kd_qwen3_4b_lcdrive}" \
        "${run_extra[@]}"
}

AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
[[ "$AUTO_RESUME_LATEST" == "0" || "$AUTO_RESUME_LATEST" == "1" ]] || { echo "[restart] AUTO_RESUME_LATEST must be 0 or 1" >&2; exit 2; }
[[ "$MAX_RESTARTS" =~ ^[0-9]+$ ]] || { echo "[restart] MAX_RESTARTS must be a non-negative integer" >&2; exit 2; }

restart_count=0
while true; do
    resume_checkpoint=""
    if [[ "$AUTO_RESUME_LATEST" == "1" ]]; then
        resume_checkpoint="$(latest_complete_checkpoint)"
    fi
    if run_training "$resume_checkpoint"; then
        exit 0
    else
        status=$?
    fi
    if (( restart_count >= MAX_RESTARTS )); then
        echo "[restart] training failed with status $status; retry budget exhausted" >&2
        exit "$status"
    fi
    restart_count=$((restart_count + 1))
    echo "[restart] training failed with status $status; restarting $restart_count/$MAX_RESTARTS in 5s" >&2
    sleep 5
done
