#!/bin/bash
#SBATCH --job-name=a1_5_kd_train
#SBATCH --partition=debug
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdtrain_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdtrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
# 120G: host RAM is only for the frame-decoding dataloader workers (the models live on
# the GPUs). Asking 160G left the KAVA job PENDING on (Resources) when co-tenants held
# 368G of the node's 503 GiB — a memory limit, not a GPU one.
#SBATCH --mem=120G
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
GPUS="${GPUS:-2}"
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
ARM="${ARM:-kv}"
MODEL_TAG=4b          # overridden per-arm below; part of output_dir and run_name
BS="${BS:-}"              # per-device batch; ZeRO-2 shards only ACROSS ranks, so bs>1 needs >=2
ACCUM="${ACCUM:-}"        # effective batch = GPUS x BS x ACCUM; the config ships 2 x 1 x 12 = 24
WARMUP="${WARMUP:-}"
# Continue a finished run for more epochs, preserving Adam moments and the dataloader
# position. RESUME is a checkpoint-* dir; EPOCHS is the TOTAL including epochs already
# trained (so a 1-epoch run continued by 2 needs EPOCHS=3).
RESUME="${RESUME:-}"
EPOCHS="${EPOCHS:-}"

cd "$RECIPE_DIR"
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
MASTER_PORT=$((29560 + SLURM_JOB_ID % 20000))

EXTRA=()
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
        echo "[slurm] unknown ARM=$ARM (expected ce|kd|kv|cekv|kvonly|kvband|blockonly|blockrandt|blockfr|block2b)" >&2; exit 1 ;;
esac
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
EXTRA+=(paths.output_dir="$OUT_DIR/output_kd_${MODEL_TAG}_${RUN_TAG}_lcdrive"
        "run_name=kd_${MODEL_TAG}_${RUN_TAG}_$(date +%m%d-%H%M)")
echo "[slurm] ARM=$ARM -> $OUT_DIR/output_kd_${MODEL_TAG}_${RUN_TAG}_lcdrive"

if [[ "$SMOKE" == "1" ]]; then
    # Short, self-contained proof the path runs: teacher loads, sequences match, and all
    # three terms are finite AND falling. A term that is finite but flat is the failure
    # this recipe has to catch.
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=2
            ++trainer.save_strategy=no ++trainer.eval_strategy=no
            ++trainer.warmup_steps=0 ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids="0-120" ++data.val_dataset.chunk_ids="0-120")
    echo "[slurm] SMOKE mode: 20 steps"
fi

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

"${LAUNCH[@]}" "$VENV/torchrun" \
    --nproc_per_node "$GPUS" \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "${CONFIG_NAME:-sft_kd_qwen3_4b_lcdrive}" \
    "${EXTRA[@]}"
