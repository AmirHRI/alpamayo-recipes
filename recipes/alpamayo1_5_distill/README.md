# Alpamayo 1.5 — latent-reasoning distillation (10B → 2B)

Distils the released **Alpamayo-1.5-10B** teacher into a **Cosmos-Reason2-2B**
student, targeting **10 Hz (100 ms)** closed-loop latency. This first cut
distils the **VLM backbone only** (the action expert is a documented
follow-up).

Built on top of [`alpamayo1_5_sft`](../alpamayo1_5_sft/): it imports that
recipe's student/teacher model classes, trainer, and shared config groups, and
adds only the distillation-specific model subclass, dataset wrappers, offline
teacher-feature cache script, and configs.

---

## Background: how Alpamayo works

### Two models joined by a KV cache

Alpamayo is a vision-language-action model in two parts:

1. a **VLM backbone** that reads the cameras, the ego history and a text prompt,
   and — if the prompt asks for it — *writes out chain-of-thought reasoning*,
   stopping at the special token `<traj_future_start>`; and
2. an **action expert**, a separate transformer that produces the trajectory by
   iterative denoising (flow matching) and never emits text.

They are not joined by passing a summary vector. The expert **reads the VLM's KV
cache directly** (the π0 / Pi-0 pattern):

```
  4 cams x 4 frames        ┌───────────────── VLM backbone ─────────────────┐
  ego history        ────► │ [~2880 image tok][48 history][prompt]          │
  text prompt              │             ↓ generates reasoning ↓            │
                           │ [CoT tokens ....] <traj_future_start>          │
                           └────────────────────┬───────────────────────────┘
                                                │  KV cache, cropped at <tfs>
                                                │  (K,V for every layer x position)
                                                ▼
  noise ──────────────────►┌───────────────── action expert ────────────────┐
                           │ 64 action tokens attend NON-CAUSALLY over the  │
                           │ VLM's cached K/V        x 10 Euler steps       │
                           └────────────────────┬───────────────────────────┘
                                                ▼
                                     trajectory: 64 waypoints
```

The cache is cropped at `future_start_idx + 1` — everything **up to and
including** `<traj_future_start>` (see the `kv_cache.crop(...)` call in
[`sft_alpamayo_r1.py`](../alpamayo1_5_sft/models/sft_alpamayo_r1.py)). So that
token is the handoff point: everything before it is the VLM's job, everything
after it is the expert's.

### The expert reads the cache layer-by-layer

The handoff is not a single tensor — HF indexes the cache by `layer_idx`, so
**expert layer *i* attends to VLM cache layer *i***:

```
        VLM KV cache                      action expert
   layer  0  ──── K,V ──────────────────►  layer  0
   layer  1  ──── K,V ──────────────────►  layer  1
     ...                                     ...
   layer 35  ──── K,V ──────────────────►  layer 35
```

This is why the expert must be **as deep as the VLM** (see the section below) —
and why "which layer do we distil?" is a real question rather than a detail.

### What one forward pass produces

A transformer is a *stack* of layers, so a forward pass doesn't yield one output
per token — it yields a **grid** of vectors, one at every (position, layer) pair:

```
                       token position  →
             [images .....][history][prompt][CoT tokens][<tfs>]
  layer 36 ┌────·─────·────────·───────·──────·───·───·────●──   ← "last layer"
  layer 35 │    ·     ·        ·       ·      ·   ·   ·    ·
    ...    │    ·     ·        ·       ·      ·   ·   ·    ·
  layer  1 │    ·     ·        ·       ·      ·   ·   ·    ·
  embedding└────·─────·────────·───────·──────·───·───·────·──
```

"Last layer" = the top row: the most processed representation, the one the model
uses to predict its next token. Because attention is causal, the vector at a
position has absorbed every token **before** it — so the cell at
(`<traj_future_start>`, top layer), marked ●, is the model's most complete
summary of the scene at the exact moment it hands off to the expert. For a
teacher that just reasoned, that summary **includes its own chain-of-thought**.

### Teacher vs. student

Same structure, different size. Both were verified against the real checkpoints:

| | teacher (Alpamayo-1.5-10B) | student (Cosmos-Reason2-2B) |
|---|---|---|
| VLM layers / hidden | 36 / 4096 | 28 / 2048 |
| expert layers | 36 (= VLM depth) | 28 (= VLM depth) |
| expert hidden / q-heads | 2048 / 16 | 1024 / 8 |
| **KV per layer** | **8 × 128 = 1024** | **8 × 128 = 1024** |
| expert params | 2.279 B | 0.473 B |

The KV width being *identical* is what makes the two models' handoffs directly
comparable — only the last-hidden width differs (4096 vs 2048).

### Why latent-reasoning distillation makes sense

The teacher's driving skill comes substantially from **reasoning out loud**: CoT
gives it extra serial compute, letting it work through "the light is red, there's
a lead vehicle" across successive token positions before committing to a
trajectory. But those tokens are generated autoregressively, and generation is
the single biggest latency cost (see the table below).

Three ways to give a small student that competence:

| approach | reasoning transferred? | latency cost |
|---|---|---|
| copy the teacher's trajectory only | ✗ — just the answer, not the thinking | none |
| teach the student to emit CoT too | ✓ | ✗ ~11.5 ms **per token** |
| **distil the reasoning-conditioned representation** | ✓ | **none** |

The third works because of where the reasoning ends up. By the time the teacher
reaches `<traj_future_start>`, its CoT has already been absorbed into the
representation at that position — and that representation, not the text, is what
the expert consumes. So we can train a **text-silent** student to reproduce it
directly:

```
  TEACHER  (reasons)                        STUDENT  (text-silent)
  ...[prompt][CoT tokens]<tfs>              ...[prompt]<tfs>
  layer 36 ──────────────── ● 4096          layer 28 ────── ● 2048
                            │                               │
                            │                     Linear(2048 → 4096)
                            │                               │
                            └────── smooth-L1 + cosine ─────┘
```

The student must compress into one forward pass what the teacher spread across
generated tokens. If it succeeds, the reasoning is **compiled into the KV** the
expert reads, and the rollout collapses to ~1 token — the teacher's competence at
the student's latency. That is the whole bet of this recipe.

---

## The latency budget, and the three decisions it forces

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

- The **teacher generates its own chain-of-thought** (it was trained to) and
  stops at `<traj_future_start>`, exactly as at deployment — so its hidden there
  is reasoning-conditioned with **no ground-truth CoT required** (works on every
  clip). GT CoT teacher-forcing is available as a fallback where it exists.
- The **student** stays **text-silent** (no `cot` in its prompt), so its rollout
  collapses to ~1 token.
- We distil the teacher's **CoT-conditioned hidden state at `<traj_future_start>`**
  — the exact vector the action expert conditions on — into the student's hidden
  state at that token.

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

## Expert depth must equal VLM depth (fixed in `alpamayo1_5_sft`)

HF indexes the KV cache by `layer_idx`, so **expert layer *i* attends to VLM cache
layer *i***. Probing the Stage-2 forward confirmed the consequence empirically:

```
7-layer expert   → reads cache layers [0..6]   → 21 of 28 VLM layers STRANDED
28-layer expert  → reads cache layers [0..27]  → 0 stranded
real 10B teacher → reads cache layers [0..35]  → 0 stranded   (36 expert = 36 VLM)
```

(All three measured by instrumenting `Cache.update` during a real Stage-2
forward. The 10B's geometry was additionally confirmed against its released
tensors: `expert.layers.0..35`, no gaps; `q_proj [2048,2048]`; `k_proj
[1024,2048]` — 8 kv-heads × 128, identical to its VLM; 2.279 B params.)

The 2B recipe had sized its expert by copying the 10B's *parameter ratio* (~20%)
via depth (`expert_num_layers: 7`), which silently cut the action head off from
the VLM's 21 deepest layers. The 10B never does this — its `expert_cfg` omits
`num_hidden_layers` (expert 36 == VLM 36) and shrinks **width** instead. The 2B
expert now mirrors that rule (depth 28, hidden 1024, 8 query heads, `kv_heads` /
`head_dim` inherited), at essentially the same budget: **0.473 B vs 0.443 B**.

Two things follow for distillation:

- The single-vector objective above targets the **last layer**, which is *not* in
  the KV cache at all (K/V at layer ℓ are projected from layer ℓ's *input*, so the
  cache is built from layers 0..L−1; the final output feeds only the LM head). It
  shapes the expert's input indirectly, via gradients through the stack.
- The direct alternative is now well-defined across the whole stack: match teacher
  **K/V at the `<traj_future_start>` column**, layer-mapped 36 → 28. Widths already
  agree (8 × 128 = 1024 both sides), so it needs **no projector** — and with the
  mirrored expert every one of those layers actually reaches the trajectory.

## What's in this recipe

- **`models/distill_base_model.py`** — `DistillReasoningVLA(TrainableReasoningVLA)`:
  adds `output_hidden_states=True`, locates the `<traj_future_start>` column
  per row (by value, robust to left-padding), computes the latent loss, and
  holds the projector. Also serves as the *teacher* (no projector, no CE): either
  `extract_tfs_hidden_generated` (the teacher generates its own CoT, then a clean
  forward reads the hidden at the *generated* `<traj_future_start>`) or
  `extract_tfs_hidden` (GT-CoT teacher-forced, single forward).
- **`data/distill_dataset.py`** — `DistillPAIDataset` / `DistillNavDataset`:
  attach the cached `teacher_tfs_hidden` to each sample (keyed by
  `f"{clip_id}::{t0_us}"`, so the offline teacher pass and the training pass line
  up sample-for-sample) and, for the nav path, inject the annotation `cot` before
  preprocessing so a teacher-side processor can teacher-force it.
- **`scripts/generate_teacher_features.py`** — offline cache builder. Caches the
  `<traj_future_start>` hidden as safetensors + a `.meta.json` sidecar
  (`teacher_hidden_dim`, `mode`). **Resumable** (checkpoints every `save_every`,
  skips cached keys), **shardable** (`num_shards`/`shard`), prefetches frames with
  a DataLoader (`num_workers`), and skips the teacher's throwaway 8B init with
  `no_init_weights()` (load ~200s → ~12s).
- **`configs/`**:
  - `cache_teacher_features_lcdrive.yaml` — LCDrive-train cache (`mode=generate`,
    `vla_processor/distill_teacher_generate`); `cache_teacher_features.yaml` —
    nav-demo cache (`mode=teacher_force`, GT CoT).
  - `sft_stage1_distill_cosmos2b.yaml` — student Stage-1 (CoT-free) + latent KD.
  - `models/{teacher_ar1_5_10b,teacher_cosmos2b,cosmos_reason2_2b_distill}.yaml`.
  - `vla_processor/distill_teacher_generate.yaml` — `cot`-**last** processor that
    makes the teacher generate reasoning before the handoff; `distill_teacher{,_nav}.yaml`
    — GT-CoT (`cot`-in-order) processors for the teacher-force path.
  - Shared groups (`sft_base`, `deepspeed`, existing `models`/`vla_processor`)
    resolve from `alpamayo1_5_sft/configs` via `hydra.searchpath`.

## How to run

The recipe reuses `alpamayo1_5_sft`'s environment; run with its venv and
`PYTHONPATH` pointed at `recipes/` (so both packages import):

```bash
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
```

**1. Generate the teacher feature cache** — the teacher **generates its own CoT**
(no GT CoT needed) and we cache the reasoning-conditioned hidden. For the LCDrive
train split:

```bash
cd recipes/alpamayo1_5_distill
CUDA_VISIBLE_DEVICES=0 $VENV/python -m alpamayo1_5_distill.scripts.generate_teacher_features \
    config=cache_teacher_features_lcdrive \
    teacher=teacher_ar1_5_10b \
    mode=generate num_workers=8 save_every=500 \
    out=/data/.../alpamayo1_5_distill/training/teacher_lcdrive_train.safetensors
# prints teacher_hidden_dim (=4096 for the 10B) -> set it in the student config
```

- **Scale.** LCDrive train ≈ **38,340 clips** at ~**1.1 samples/s** (10B, 1 GPU) ⇒
  **~10 h**. The run is **resumable** (checkpoints every `save_every`, skips
  already-cached keys on restart) and **shardable** across GPUs — launch N copies
  with `num_shards=N shard=k` (`k=0..N-1`) to different `out=` files, then merge.
  Only a 10B fits on a free 80 GB GPU, so a single card can't be split; use
  separate GPUs for shards.
- **Why `mode=generate` + the `distill_teacher_generate` processor.** The teacher
  only reasons if the prompt asks for CoT **and** the assistant turn ends with
  `<|cot_start|>` — so that processor puts `cot` **last** in `components_order`.
  A `traj_future`-last prompt (the shipped `default`/`nav` processors) pre-fills
  `<|traj_future_start|>` and the teacher skips reasoning entirely (empty CoT).
  Example greedy CoTs: *"Stop for the red traffic light since the signal is red"*,
  *"Adapt speed for the right curve since the lane bends right ahead"*.
- **Greedy by default** (`do_sample=false`) for a reproducible, deterministic
  cache target. The `.meta.json` sidecar records `teacher_hidden_dim` and `mode`.

For the small **nav-demo** smoke path (annotations carry GT CoT), the
`cache_teacher_features.yaml` config with `mode=teacher_force` instead
teacher-forces the GT CoT via `vla_processor/distill_teacher_nav`.

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
  `from_alpamayo_checkpoint` (in ~12 s with `no_init_weights`) and emits a
  `teacher_hidden_dim=4096` cache — matching the student config default.
- **Generation-based extraction validated** on real nav *and* LCDrive clips: with
  the `cot`-last processor the 10B generates coherent, scene-specific reasoning
  before `<traj_future_start>` (e.g. *"Turn left due to green left-turn signal"*,
  *"Keep distance to the lead vehicle since it is directly ahead in our lane"*),
  and the hidden is captured there. Throughput ~1.1 samples/s (10B, 1 GPU) with
  8 prefetch workers ⇒ the full 38,340-clip LCDrive-train cache ≈ ~10 h.
  ⚠️ A `traj_future`-last prompt yields **empty** CoT — the `cot`-last processor
  is required (see step 1).
- Configs compose via `hydra.searchpath` onto `alpamayo1_5_sft/configs` (both
  the cache builder and the real `torchrun -m alpamayo1_5_sft.train_hf
  --config-path pkg://alpamayo1_5_distill/configs` launch); the student trains
  CoT-free while the teacher cache is built from the teacher's own CoT.

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
