# Alpamayo 1.5 — latent-reasoning distillation (10B → 2B)

Distils the released **Alpamayo-1.5-10B** teacher into a **Cosmos-Reason2-2B**
student, targeting **10 Hz (100 ms)** closed-loop latency. This first cut
distils the **VLM backbone only** (the action expert is a documented
follow-up).

Built on top of [`alpamayo1_5_sft`](../alpamayo1_5_sft/): it imports that
recipe's student/teacher model classes, trainer, and shared config groups, and
adds only the distillation-specific model subclass, dataset wrappers, offline
teacher-feature cache script, and configs.

## The idea: distil reasoning into the KV, not into tokens

Profiling (see `alpamayo1_5_sft_qwen3_5/README.md`) shows the trained-2B's
~311 ms median is dominated by the **VLM**, not the expert:

| Phase | Cost (trained 2B) | Driver |
|---|---|---|
| VLM prefill | ~85 ms | 16 images (4 cams × 4 frames) × ~180 tok ≈ 3 k tokens |
| VLM autoregressive rollout | ~95–250 ms (high variance) | # reasoning tokens × 11.5 ms/token |
| Diffusion loop | ~41 ms | 10 Euler steps × 4.1 ms |

Autoregressive **reasoning tokens are the single biggest latency cost** — yet
reasoning is where the teacher's driving competence lives. Latent-reasoning
distillation resolves the tension:

- The **teacher** runs *with* chain-of-thought in context (a single forward, CoT
  teacher-forced from PAI's reasoning parquet / nav annotations).
- The **student** stays **text-silent** (no `cot` in its prompt), so its rollout
  collapses to ~1 token.
- We distil the teacher's **CoT-conditioned hidden state at `<traj_future_start>`**
  — the exact vector the action expert conditions on — into the student's hidden
  state at that token.

At inference the action expert consumes the VLM KV cache cropped at
`future_start_idx + 1` (i.e. up to and including `<traj_future_start>`; see
[`sft_alpamayo_r1.py`](../alpamayo1_5_sft/models/sft_alpamayo_r1.py) around the
`kv_cache.crop(...)` call). The last-layer hidden at that position is the single
best summary of the context handed to the expert. Matching it **compiles the
teacher's reasoning into the KV the expert reads**, with no reasoning tokens
emitted at run time.

## Two verified enablers (checked against the real configs)

- **Tokenizers align.** `Cosmos-Reason2-2B` base vocab 151936; teacher extends to
  155697 via the *same* Alpamayo vocab-extension code path. (Relevant if you add
  logit-KD later; the latent objective here needs no vocab alignment.)
- **The VLM KV width is identical** across sizes (both `num_key_value_heads=8 ×
  head_dim=128 = 1024`/layer), so the only teacher↔student mismatch is the
  last-hidden width — handled by a single learned `2048 → teacher_hidden`
  projector that is **discarded at inference**.

The loss is `L = CE_next_token + λ · (smooth_l1(P(h_S), h_T) + 0.1·(1 − cos))`,
where `h_S` / `h_T` are the student/teacher hidden states at `<traj_future_start>`
and `P` is the projector. The CE term is the unchanged Stage-1 objective, so
eval / generation are unaffected when no teacher feature is supplied.

## What's in this recipe

- **`models/distill_base_model.py`** — `DistillReasoningVLA(TrainableReasoningVLA)`:
  adds `output_hidden_states=True`, locates the `<traj_future_start>` column
  per row (by value, robust to left-padding), computes the latent loss, and
  holds the projector. Also serves as the *teacher* via `extract_tfs_hidden`
  (no projector, no CE — just the raw hidden).
- **`data/distill_dataset.py`** — `DistillPAIDataset` / `DistillNavDataset`:
  attach the cached `teacher_tfs_hidden` to each sample (keyed by
  `f"{clip_id}::{t0_us}"`, so the offline teacher pass and the training pass line
  up sample-for-sample) and, for the nav path, inject the annotation `cot` before
  preprocessing so a teacher-side processor can teacher-force it.
- **`scripts/generate_teacher_features.py`** — offline cache builder: runs the
  frozen teacher once, caches the `<traj_future_start>` hidden as safetensors +
  a `.meta.json` sidecar (records `teacher_hidden_dim`).
- **`configs/`**:
  - `cache_teacher_features.yaml` — teacher cache config (teacher runs *with* CoT
    via `vla_processor/distill_teacher_nav`).
  - `sft_stage1_distill_cosmos2b.yaml` — student Stage-1 (CoT-free) + latent KD.
  - `models/{teacher_ar1_5_10b,teacher_cosmos2b,cosmos_reason2_2b_distill}.yaml`.
  - `vla_processor/{distill_teacher,distill_teacher_nav}.yaml` — teacher-side
    processors that add `cot` to `components_order`.
  - Shared groups (`sft_base`, `deepspeed`, existing `models`/`vla_processor`)
    resolve from `alpamayo1_5_sft/configs` via `hydra.searchpath`.

## How to run

The recipe reuses `alpamayo1_5_sft`'s environment; run with its venv and
`PYTHONPATH` pointed at `recipes/` (so both packages import):

```bash
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
```

**1. Generate the teacher feature cache** (real 10B teacher, CoT in context):

```bash
cd recipes/alpamayo1_5_distill
CUDA_VISIBLE_DEVICES=0 $VENV/python -m alpamayo1_5_distill.scripts.generate_teacher_features \
    config=cache_teacher_features \
    teacher=teacher_ar1_5_10b \
    out=/path/to/teacher_features.safetensors
# prints teacher_hidden_dim (=4096 for the 10B) -> set it in the student config
```

For a plain-PAI / LCDrive corpus (no nav annotations), point
`data.cache_dataset` at `DistillPAIDataset` with
`reasoning_metadata=reasoning/ood_reasoning.parquet` and the
`vla_processor/distill_teacher` (non-nav) processor.

**2. Train the CoT-free student** with the latent loss:

```bash
CUDA_VISIBLE_DEVICES=0 $VENV/torchrun --nproc_per_node 1 \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_stage1_distill_cosmos2b \
    data.train_dataset.teacher_cache_path=/path/to/teacher_features.safetensors \
    model.teacher_hidden_dim=4096
```

**3. Stage-2 + deploy.** Freeze the distilled VLM and train the action expert
exactly as `alpamayo1_5_sft` Stage-2 does (`stage1_vlm_checkpoint_path` → the
distilled Stage-1 checkpoint) — no code change; it inherits the reasoning-rich
KV. At eval, stack the latency levers distillation pays for: fewer camera frames
(`num_frames=8`), a capped/silent rollout, and fewer diffusion steps
(`diffusion_kwargs={"inference_step": 2}`, already plumbed).

## Status — verified end-to-end (real H100, real PAI data)

- **Cache builder** runs with both a 2B stand-in teacher (dim 2048) and the real
  10B teacher; writes safetensors + `.meta.json`, keyed by `(clip_id, t0_us)`.
- **Distill forward + training** validated on the nav demo: the projector
  registers in the optimizer, `teacher_tfs_hidden` flows dataset → collator →
  `forward`, and the KD term is active and finite end-to-end. With a 2B stand-in
  teacher (matched 2048-d, near-identity projector) both terms drop cleanly over
  a short overfit (CE 29.3 → 14.4, latent 30.8 → 24.1). With the **real 10B
  teacher** (4096-d, projector 2048 → 4096 learned from scratch) the CE term
  drops the same, while the scale-sensitive smooth-L1 latent term decreases
  slowly per sample — hence the higher `latent_proj` LR multiplier in the
  config; a full run (warmup + many steps) is needed for it to converge.
- **The real teacher path** loads `nvidia/Alpamayo-1.5-10B` (~21 GB) via
  `from_alpamayo_checkpoint`, teacher-forces CoT, and emits a
  `teacher_hidden_dim=4096` cache — matching the student config default.
- Configs compose via `hydra.searchpath` onto `alpamayo1_5_sft/configs` (both
  the cache builder and the real `torchrun -m alpamayo1_5_sft.train_hf
  --config-path pkg://alpamayo1_5_distill/configs` launch); the student trains
  CoT-free while the teacher cache is built with CoT.

## Scope & follow-ups

- **This cut = VLM backbone only.** Stage-2 (the action expert) is unchanged and
  simply trains on the distilled, reasoning-infused backbone.
- **Offline-first.** No teacher in the training loop; the teacher runs once. This
  also manufactures reasoning supervision the released nav recipe never had.
- **Documented follow-ups** (not built here): expert-side distillation
  (trajectory relabel + shortcut/reflow diffusion step-reduction), an online
  co-resident teacher, and logit-KD (the natural next VLM term if the latent-only
  objective underperforms). Attention-map KD is deliberately skipped (it would
  force `eager`/`sdpa` attention, hurting both training and deploy latency).

See the design menu (all VLM/expert/orchestration options with pros/cons) that
this recipe was distilled from in the planning notes.
