# Alpamayo 1.5 — latent-reasoning distillation (10B → 2B)

Distils the released **Alpamayo-1.5-10B** teacher into a **Cosmos-Reason2-2B**
student, targeting **10 Hz (100 ms)** closed-loop latency. This first cut
distils the **VLM backbone only** (the action expert is a documented
follow-up).

Built on top of [`alpamayo1_5_sft`](../alpamayo1_5_sft/): it imports that
recipe's student/teacher model classes, trainer, and shared config groups, and
adds only the distillation-specific model subclass, dataset wrappers, offline
teacher-feature cache script, and configs.

> **Published artifacts** (both 🔒 **private, Honda-internal** — do not make public):
> | | |
> |---|---|
> | 🤗 `ac4462/alpamayo-kava-2b-m8-lcdrive` | the trained M=8 student, 4.70 GB — see **K5-alt** |
> | 🤗 `ac4462/alpamayo-kava-cache` | the teacher KV cache, 34 GB — see **K3-alt** |
>
> Together these let a second machine evaluate or continue from this work without the
> 10B teacher, the 22 GPU-hour cache build, or the 22-hour training run.

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

## Related architecture: MolmoAct2's action expert

MolmoAct2 solves the same problem — conditioning a flow-matching action expert on
a VLM — and independently reaches several of the same answers. Useful both as
corroboration and as a list of things Alpamayo does differently.

### What is identical

| | MolmoAct2 | Alpamayo |
|---|---|---|
| interpolation | `x_t = (1−t)·ε + t·a` | `noisy_x = t·x + (1−t)·noise` |
| velocity target | `u* = a − ε` | `target = x − noise` |
| loss | MSE on predicted velocity | `mse_loss(target, pred)` |
| **expert depth** | **L = 36 = VLM depth** | 36 = VLM depth (measured) |
| conditioning granularity | expert block ℓ ← VLM layer ℓ's K/V | expert layer ℓ ← cache layer ℓ |
| gradient flow | conditioning path detached from the VLM | `stop_grad_from_vlm=True` → `.detach()` |

The flow-matching formulation is the *same equation*. And MolmoAct2 states the
expert-depth-equals-VLM-depth rule explicitly (L = 36 for both) — precisely the
invariant the 2B recipe was violating before the fix documented above.

### What differs

**1. Separate self- + cross-attention vs. one fused attention.**

```
  Alpamayo block ℓ                     MolmoAct2 block ℓ
  ────────────────                     ─────────────────
  ONE self-attention over              SA  (actions ↔ actions)   + gate
    [ VLM prefix ‖ action tokens ]     CA  (actions → K̃_ℓ, Ṽ_ℓ)  + gate
  MLP                                  MLP                        + gate
```

Alpamayo's single softmax normalises over the VLM prefix *and* the action tokens
together, so the two **compete for attention mass**, and one set of Q/K/V
projections serves both roles. MolmoAct2 gives each its own softmax, its own
parameters, and a learned gate — it can attend fully to context *and* fully to
action structure independently.

**2. Per-layer DiT time conditioning vs. input-only.** Alpamayo injects the
diffusion timestep **once**, at the input: Fourier features concatenated onto the
action features inside `action_in_proj`. The expert itself is built by
`AutoModel.from_config(...)` — a stock Qwen3 decoder with **no time awareness at
all**, so `t` must propagate implicitly through all 36 layers. MolmoAct2 derives
AdaRMS shift/scale/gate parameters from `t` and applies them to **all three
residual branches of every block**. This is the clearest quality-relevant gap;
DiT-style adaptive-norm conditioning is well established to beat input-only
conditioning in diffusion transformers.

**3. Learned KV adapters vs. a shape constraint.** MolmoAct2 maps the VLM's keys
and values into the expert's cross-attention width with learned linear adapters
`P_K`, `P_V` (separate from the VLM's own attention projections). Alpamayo has no
adapter — the expert consumes the cached K/V **directly**, which is only possible
because `num_key_value_heads` (8) and `head_dim` (128) are constrained to match
the VLM. Alpamayo therefore pays zero parameters and zero extra compute, but
cannot freely choose the expert's attention geometry; MolmoAct2 pays a small
adapter cost (amortisable — the VLM K/V are fixed per observation, so the
projection can be computed once and reused across all denoising steps) and buys
complete freedom over expert width, head count, and depth mapping.

**4. Action masking.** MolmoAct2 masks padded action steps and dimensions (`m` in
its loss) for variable-length chunks across embodiments. Alpamayo's action space
is a fixed `(64, 2)`, so no mask is needed — a domain difference, not a gap.

### Implications for this recipe

- **Two independent votes for per-layer KV as the KD target.** MolmoAct2's stated
  rationale for rejecting "a shallow projection of the final hidden state" is
  exactly the argument for matching teacher **K/V per layer** rather than the
  single last-layer vector this recipe currently uses (see the caveat under
  *Expert depth must equal VLM depth*).
- **Both detach the conditioning path**, so in neither architecture does the
  action loss train the VLM. The VLM's contribution to driving quality can only
  be bought by training it *directly* — which is what this recipe does.
- **Port candidates, ranked** (none implemented here):
  1. *DiT/AdaRMS time conditioning* — best value/risk; contained (wrap the expert
     blocks with modulation driven by the existing Fourier time embedding),
     doesn't touch the KV interface. Would invalidate existing expert checkpoints.
  2. *KV adapters* — decouple expert width **and** depth from the VLM, which would
     make the prune plan below trivial and let the expert shrink past the 8 × 128
     floor.
  3. *Separate SA/CA* — largest change, least certain payoff; try last.

---

## The latency budget, and the three decisions it forces

Profiling (see `alpamayo1_5_sft_qwen3_5/README.md`) shows the trained-2B's
~311 ms median is dominated by the **VLM**, not the expert:

| Phase | Cost (trained 2B) | Driver |
|---|---|---|
| VLM prefill | ~85 ms | 16 images (4 cams × 4 frames) × ~180 tok ≈ 3 k tokens |
| VLM autoregressive rollout | ~95–250 ms (high variance) | # reasoning tokens × 11.5 ms/token |
| Diffusion loop | ~41 ms → **~136 ms** | 10 Euler steps × 4.1 ms → **13.6 ms** (see below) |

> **The diffusion row is superseded.** The 4.1 ms/step figure was measured with the
> old 7-layer expert, which read only 7 of 28 VLM cache layers. Making the expert
> full-depth (the fix below) costs **~2.9× per denoising step** — same FLOPs and
> parameter count, but 4× the sequential layers and 4× the cross-attention KV
> traffic, and at batch-1 over 64 action tokens the expert is launch-latency-bound
> rather than FLOP-bound:
>
> | expert | params | KV read/step | ms/step | ×10 steps | cache layers read |
> |---|---|---|---|---|---|
> | 7 × 2048 (old) | 443 M | 86 MB | 4.68 | 47 ms | 7/28 ✗ |
> | 28 × 1024 (current) | 473 M | 344 MB | 13.59 | 136 ms | 28/28 ✓ |
>
> (Isolated microbenchmark of `expert.forward` against a synthetic 3000-position
> cache; the 4.68 ms reproduces the 4.1 ms measured end-to-end, validating it. The
> full profiler on the current arch reports 19.3 ms/step under its own conditions —
> a random-init VLM that rolls out to the 128-token cap, so a longer cache. Both
> runs shared the GPU, so treat the ~2.9× **ratio** as the result, not the absolutes.)
>
> **This makes diffusion step-reduction a prerequisite, not a nice-to-have**: at
> 136 ms the loop alone exceeds the whole 100 ms budget. Cutting 10 → 2 steps
> (`diffusion_kwargs={"inference_step": 2}`, already plumbed) brings it to ~27 ms.
> A cheaper alternative that needs no retrain is noted under *Scope & follow-ups*.

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
  **K/V per layer**, layer-mapped 36 → 28. Widths already agree (8 × 128 = 1024 both
  sides) — and with the mirrored expert every one of those layers actually reaches
  the trajectory. This is what the **KAVA** section below implements.

  > Earlier drafts of this README concluded from the matching widths that K/V
  > matching "needs **no projector**". Matching *widths* is not matching *bases*:
  > teacher and student are separately trained checkpoints, so their `W_k`/`W_v`
  > need not agree even at identical geometry. Both variants are therefore wired
  > (`kv_align: direct | projector`), with the projectors identity-initialised so
  > `projector` starts at exactly the `direct` objective and the two form a clean
  > ablation pair rather than two unrelated runs.

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

### Added for KAVA

- **`models/kv_distill.py`** — the pure math: R-KV scoring (`redundancy_score`,
  `importance_score`, `combine_scores`), `select_top_m` / `evict_teacher_cache`,
  `build_layer_map` (28 → 36), `KVProjectorBank`, `kv_matching_loss`.
- **`models/kava_model.py`** — `KaVaReasoningVLA(DistillReasoningVLA)`: slot
  embeddings, the `embed_tokens` injection hook, per-layer pre-RoPE K/V capture,
  the PCCoT Jacobi loop, and `L_KV`.
- **`models/teacher_kv.py`** — teacher-side capture (one forward split into two
  segments; see below). Used only by the cache builder.
- **`models/expert_teacher.py`** — `KaVaExpertTeacher` (a teacher carrying both the
  action expert and CoT generation) plus `expert_cot_importance` / `score_from_qk` /
  `build_noisy_action` for `importance_source=expert`.
- **`data/kv_cache_io.py`** — the tiered per-sample store (`full/`,
  `compressed_<tag>/`, `index.*.json`, `cot_text.*.jsonl`), mmap reads.
- **`data/kava_dataset.py`** — `KaVaPAIDataset` and `KaVaCollator`
  (`splice_slot_placeholders`, `pad_kv_to_budget`).
- **`scripts/{cache_common,generate_teacher_kv,compress_teacher_kv,validate_kv_distill,profile_kava_latency}.py`**
  — shared builder plumbing (also now used by `generate_teacher_features.py`), the
  KV cache builder, offline recompression, and the KV1–KV7 harness.
- **`trainer.py` / `train_kava.py`** — `KaVaTrainer` logs the loss terms separately
  and keeps weight decay off the soft prompt; `train_kava` rebinds the trainer in
  `alpamayo1_5_sft.train_hf` rather than forking it.
- **`configs/{cache_teacher_kv_lcdrive,sft_stage1_kava_cosmos2b_lcdrive}.yaml`**,
  **`configs/models/cosmos_reason2_2b_kava.yaml`** and
  **`configs/models/teacher_ar1_5_10b_expert.yaml`**.

## KAVA: compressed KV-cache distillation

`KAVA` (arXiv:2510.02312, ICLR 2026) supplies the supervision the single-vector
objective cannot express. `K` continuous **latent slots** stand where the teacher's
text CoT stood, and their per-layer, per-head **K and V** are matched against the
teacher's CoT cache after **redundancy/importance-aware eviction** down to `K`
entries:

```
S_{i,h,l} = λ · I_{i,h,l}  +  (1 − λ) · R_{i,h,l}          λ = 0.1
L_KV      = mean over valid slots of  |sg[K̃_t] − A_l(K_s)|_p  +  |sg[Ṽ_t] − B_l(V_s)|_p
L         = L_CE(traj)  +  λ₁ · L_latent(tfs hidden)  +  λ₂ · L_KV
```

`I` is the attention the *reader* pays each CoT token (MaxPooled over each GQA query
group first — several queries share one cached pair, and a pair matters if any of
them needs it); `R` is softmax-normalised negated mean pairwise key cosine. The
paper's central claim is what makes this usable across two different models: a
compressed cache has **lost token correspondence**, and that is fine — continuous
latents can absorb structure that token- or hidden-level matching cannot express.

Two readers can supply `I`, selected by `importance_source`:

| | `vlm_post_cot` | `expert` |
|---|---|---|
| Queries | the `[<\|cot_end\|>, <\|traj_future_start\|>]` tokens (N_A ≈ 2) | the action expert's 64 noisy action tokens |
| Teacher needed | VLM only (`teacher_ar1_5_10b`) | VLM + expert (`teacher_ar1_5_10b_expert`) |
| Faithfulness | a text proxy for the reader | **the actual reader** — these queries consume this cache at inference |
| Cost / sample | negligible | ~1 s (3 timesteps), 22.7 GiB peak |

`expert` is the more faithful signal and is why this recipe can improve on KAVA's
text-answer formulation: in a driving VLA the answer is not text at all.

### Four places this departs from the paper

1. **Cross-model, not self-distillation.** KAVA runs one model in two modes. Here
   10B → 2B means depth differs (36 vs 28, hence the layer map) and the learned K/V
   bases need not agree, hence `kv_align`.
2. **The "answer" is not text.** The expert consumes the KV cache cropped at
   `future_start_idx + 1`, so the KV target *is* the expert's input rather than a
   proxy — and the slots must be spliced **before** `<|traj_future_start|>`, not
   appended at the end as in `reasoning-setup-2b.md` §9's harness, or they fall
   outside that crop. It also means R-KV's importance term has a *better* source here
   than in the paper: `importance_source=expert` scores CoT tokens by the expert's own
   cross-attention (see below).
3. **`M` can exceed `N_C`.** Real driving CoT is ~40 tokens and the teacher is capped
   at 128, so at `M=32` eviction is sometimes a near-no-op and short traces leave
   slots with no target. Those are masked out of `L_KV` (`valid_mask`); KAVA never
   meets this case. Mild compression is the paper's *good* regime, so this is
   favourable, not a problem.
4. **Keys are matched pre-RoPE.** `DynamicCache` stores post-RoPE keys; eviction
   reorders which CoT token each slot targets, so any position-dependent component of
   the target is arbitrary by construction. The hooks read `k_norm` instead.
   *Measured honestly:* on this model the phase effect is **modest** — 6.8% rel-L2
   pre- vs post-RoPE, 12.9% for the same keys rotated 3000 positions later, because
   `rope_theta = 5e6` leaves the low-frequency dimensions nearly unrotated. So
   pre-RoPE is a well-founded default that costs nothing, not a large measured win.

### Why the teacher runs offline first

Decisively cheaper, and it is the question worth answering before building anything:

| | Offline (this recipe) | Online teacher |
|---|---|---|
| Teacher forwards | 38,340 (once) | 38,340 × epochs × sweep points |
| CoT generation in the step loop | never | autoregressive, dominates step time |
| Resident weights | student only | +21 GB (10B VLM + 2.279B expert) |
| Sweeping `M`/`λ`/eviction | recompress from `full/`, CPU only | full re-run each time |

It is *valid* because the PAI path is fully deterministic — no augmentation, no random
frame sampling anywhere (the only `random` call in the data path is LingoQA's retry),
`t0_us` is the constant `DEFAULT_T0_US = 5_100_000`, camera order is force-sorted, and
generation is pinned greedy. The cost is disk and one ~10 h pass (~1.5 h over 8 shards).

### What the teacher pass records

One full forward, **split into two segments**, so the marginal cost over the existing
single-vector cache run is negligible:

- **segment A** `[context + CoT]` with `use_cache=True`; forward hooks on each layer's
  `k_norm` / `v_proj` collect the CoT span's pre-RoPE K/V (sliced inside the hook, so
  it costs kilobytes per layer rather than 36 full-sequence copies).
- **segment B** the `[<|cot_end|>, <|traj_future_start|>]` tail against that cache —
  yielding the `tfs` hidden, so one cache run feeds both objectives, and (for
  `importance_source=vlm_post_cot`) `I` via **eager** attention and
  `output_attentions=True`.

After segment B the cache is exactly `[prompt + CoT + <tfs>]`, i.e. the same object
`TrainableAlpamayoR1.forward` hands the expert after cropping at
`future_start_idx + 1`. So `importance_source=expert` needs **no third prefill** — it
reads that cache directly.

#### Scoring by the expert's cross-attention

`models/expert_teacher.py`. Two problems had to be solved rather than assumed away:

- **The expert runs non-causal.** `expert_non_causal_attention: true` arrives as an
  `is_causal=False` forward kwarg that only the *sdpa* path honours —
  `eager_attention_forward` **ignores `is_causal` entirely** and applies whatever mask
  `create_causal_mask` built. Capturing weights with `output_attentions=True` (which
  requires eager) would therefore have returned **causal** attention the expert never
  computes. Instead the attention is reconstructed directly: post-RoPE queries from a
  `q_norm` hook against the post-RoPE keys the forward leaves in the cache, softmaxed
  over the full key axis with no mask. `test_score_from_qk_matches_the_eager_reference`
  pins that to eager's arithmetic; it is also cheaper, one layer's logits at a time
  instead of 36 attention matrices at once.
- **The expert's input is noisy.** `noisy_x` and `timesteps` are randomly sampled in
  training, which an offline cache cannot tolerate. The timesteps are a fixed grid
  (default `(0.0, 0.5, 1.0)`) and the noise comes from a seeded CPU generator, so a
  clip always scores the same. The grid spans the flow: `t=1` is the clean action —
  KAVA's "answer" end — and `t=0` is the pure noise the expert actually starts from at
  inference. `build_noisy_action` asserts the diffusion is `FlowMatching` rather than
  duck-typing the interpolation.

`KaVaExpertTeacher` exists because `TrainableAlpamayoR1` (expert, action space,
diffusion) and `DistillReasoningVLA` (CoT generation) are **siblings** under
`ReasoningVLA`, not a chain — neither alone can build this cache. It borrows
`generate_cot_prefix` directly rather than duplicating it, since that method only
touches the shared `ReasoningVLA` surface.

| Tier | Contents | Measured size |
|---|---|---|
| `full/` | `k_pre`, `v` `[36, 8, N_C, 128]` bf16 + `imp`, `red` + `tfs_hidden` | **1.79 MB**/sample |
| `compressed_M16_rkv0.1/` | `k_pre`, `v` `[36, 8, M', 128]` + `sel_idx` + `tfs_hidden` | **1.72 MB**/sample |
| `cot_text.shard*.jsonl` | the teacher's reasoning as text | ~180 B |

Training reads only the compressed tier. `full/` exists so every `(M, λ, method)` point
is re-derivable by `compress_teacher_kv.py` with **no teacher and no GPU**. Measured
totals for LCDrive-train (38,340 clips): **67 GB** `full/` + **65 GB** `M=16` ≈ **132 GB**.
**Put `cache_root` on `/data`** — `/home` is quota-capped at 100 GB.

#### ⚠️ Measured CoT length changes the M guidance

The 200-clip pilot (below) measured the teacher's actual CoT on LCDrive:

```
N_C: min 6  p25 11  median 13  p75 14  p95 17  max 46  mean 12.4
```

This recipe originally assumed ~40 tokens, extrapolated from `reasoning-setup-2b.md` §0
which concerns a different setting. The real traces are **~13 tokens** — the teacher
emits one crisp sentence ("Stop for the red traffic light since the signal is red"), not
a paragraph. That has a direct consequence for `M`, since `M` is simultaneously the
latent budget *and* the number of KV pairs eviction keeps:

| M | eviction engages on | mean slots with a target |
|---|---|---|
| 4 | 100% of clips | 4.0 / 4 (100% of budget) |
| **8** | **82%** | 7.9 / 8 (99%) |
| 12 | 55% | 10.9 / 12 (91%) |
| **16** (config default) | **6.5%** | 12.0 / 16 (75%) |
| 32 | 1.0% | 12.4 / 32 (39%) |

So at the shipped `M=16`, **R-KV eviction is inactive on 93.5% of clips** — the
compressed tier is very nearly a 1:1 copy of the teacher's CoT cache (1.72 vs 1.79
MB/sample), and a quarter of the slots have no target and are masked out of `L_KV`.

That is not a bug, and per KAVA it may well be the *better* regime — its strongest
results are on GSM8k-AUG, where the cache "retains all of its content after eviction".
But it means two different things are worth running and they must not be conflated:

* `M=16` — "match the teacher's whole CoT cache, one slot per token". Simplest
  objective, most expected quality, **eviction untested**.
* `M=8` — eviction genuinely engages on 82% of clips, so this is the arm that
  exercises R-KV and the `lam` / `eviction` ablations. Derive it from `full/` with
  `compress_teacher_kv.py m=8`; no teacher re-run.

`M=32` is not useful here: 61% of the budget would never receive a target.

### Two things that fail silently, and how they are prevented

- **`gradient_checkpointing: true`** forces `use_cache=False` *and* detaches
  hook-captured tensors, so `L_KV` would be finite, look like it was descending, and
  supervise nothing. `KaVaReasoningVLA._assert_capture_possible` raises instead; the
  student config sets it `false`.
- **`cot` not last in `components_order`** pre-fills `<|traj_future_start|>` and the
  teacher emits an *empty* CoT. `find_cot_span` raises on an empty span, and
  `cot_text.*.jsonl` is where you would notice first.

### Validated on the real 2B — `scripts/validate_kv_distill.py`, 7/7

```
python -m alpamayo1_5_distill.scripts.validate_kv_distill 8 float32
```

| Check | Result |
|---|---|
| **KV1** rotate(hooked pre-RoPE K) == `DynamicCache` K; hooked V == cache V | rel-L2 **0.0** and **0.0** |
| **KV2** eviction is per-(layer, head); ablations differ; order is temporal | **220/224** distinct index sets; `rkv`≠`attn`≠`cosine`; `crop` = first M; ascending |
| **KV3** importance is a real attention distribution + GQA MaxPool | rows sum to 1.000000; 16 q-heads → 8 kv-heads (group 2) |
| **KV4** `slot_pos` ↔ placeholders; `<tfs>` two columns after the last slot | slots provably inside the expert's crop |
| **KV5** gradient reaches slots, backbone and projector through `L_KV` | ‖slots.grad‖ 2.6e1, non-zero 8/8, per-slot cosine max **0.58** (≈1.0 would be a broadcast bug) |
| **KV6** self-consistency floor: `L_KV` driven toward 0 on a solvable target | 2.767 → **0.393** (14.2%) in 200 steps |
| **KV7** cache round-trip; offline recompression reproduces the inline result | identical indices, rel-L2 0.0 |

### Latency measured on real LCDrive data — `scripts/profile_kava_latency.py`

The deployable Stage-2 student (2.609 B = frozen 2B VLM + 28-layer/1024-hidden expert)
on a real LCDrive **val** clip, batch 1, bf16, H100. Phases are timed separately and
the arms composed from measured parts, so nothing inherits another arm's overhead:

```
CUDA_VISIBLE_DEVICES=3 python -m alpamayo1_5_distill.scripts.profile_kava_latency \
    slots=0,8,16,32 n_timed=25
```

**The slots are free.** This is the load-bearing claim, and at a *realistic* prefill it
holds with room to spare — `reasoning-setup-2b.md` §8 warned that measuring at 64
vision tokens would badly understate it, so this runs at 2880:

| K | seq | prefill min (ms) | overhead vs K=0 |
|---|---|---|---|
| 0 | 2993 | 113.83 | — |
| 8 | 3003 | 113.86 | **+0.03** |
| 16 | 3011 | 113.84 | **+0.01** |
| 32 | 3027 | 114.38 | **+0.55** |

≤0.55 ms in every configuration measured, i.e. indistinguishable from zero — as the
causal mask predicts, since the slots only add ~1% to the sequence and attend to a
prefix that was going to be computed anyway.

**Per-phase costs** (min of 25 interleaved reps): ViT **59.0 ms** — *52% of prefill* —
language prefill ~54.8 ms, text decode **10.5 ms/token**, expert Euler step
**12.5 ms**. The decode and expert figures corroborate the ~11.5 ms/token and
13.59 ms/step recorded earlier in this README from a different script.

**Totals, and the frames lever.** Vision dominates, so `num_frames` is the knob that
decides whether 10 Hz is reachable; the expert is weight-bound (12.1–12.7 ms/step
regardless of cache length), so only `inference_step` moves it:

| frames/cam | images | vision tok | prefill | Full CoT (13 tok) | KAVA K=16 T=1 @10 steps | **@2 steps** | T=3 @2 |
|---|---|---|---|---|---|---|---|
| 4 (today) | 16 | 2880 | 113.8 | 381 ms (2.6 Hz) | 238 ms (4.2 Hz) | 139 ms (7.2 Hz) | 167 ms |
| **2** | 8 | 1440 | 62.0 | 321 ms (3.1 Hz) | 182 ms (5.5 Hz) | **85.7 ms (11.7 Hz)** | 113 ms |
| 1 | 4 | 720 | 38.0 | — | 165 ms (6.1 Hz) | 63.4 ms (15.8 Hz) | 91.6 ms |

`Full CoT` uses the **measured** 13-token trace length (see the pilot above), not the
40 tokens an earlier draft of this section assumed.

**The 100 ms budget is met** at `K=16, T=1, 8 camera images, inference_step=2`:
**86.6 ms / 11.6 Hz**. Three things that table says plainly:

- KAVA T=1 costs **the same as today's text-silent 2B** (138.8 vs 138.7 ms at 16
  images) — latent reasoning is added for free, and whether it *helps* is now purely a
  quality question, not a latency trade.
- The CoT it replaces costs **~143 ms** of rollout (13 measured tokens x 11.0 ms).
  Earlier drafts of this section said ~420 ms, from assuming a 40-token trace; the
  pilot measured 13. Still the single largest saving available — it is 1.7x the entire
  KAVA VLM cost at 8 images — but it is 3x smaller than first claimed, and the ViT
  (59 ms, 52% of prefill) is now the bigger target.
- `T>1` is not free: each Jacobi iteration is ~15 ms — *more* than a decode token,
  because it runs K tokens against the full cache and re-crops. `T=3` at 8 images lands
  at 114 ms, just over budget. Sweep T for quality knowing T=1 is the only free point.

⚠️ Read these as **relative**: H100, not Thor, and with untrained weights (latency is
shape-bound, so untrained is fine, but the device is not the deploy target).
`reasoning-setup-2b.md` §0 carries the Thor per-token figure. The first run of this
profile also reported the slot overhead as *negative* (−5%) — pure GPU contention from a
co-tenant job, which is why the script now times arms round-robin and reports `min`; a
strictly-more-work prefill cannot be faster, and any such result should be thrown out
rather than published.

Two findings worth carrying forward:

- **KV6's step budget is load-bearing.** At 30 steps it bottoms out near 44% of the
  starting loss and reads as a failure; 200 steps with a decaying LR reaches 14%. The
  slot → per-layer-K/V map is 28 layers deep and the slots attend to each other, so
  this is slow optimisation, not a broken objective. Do not shorten it and conclude
  the wiring is wrong.
- **Low `L_KV` does not pin the slot embedding.** The recovered slots sit 17× away
  from the ones that generated the target. That is the representational slack KAVA
  relies on — and a reminder that `L_KV` constrains the *cache*, not the latent.

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

### The KAVA path

**K1. Validate the machinery first** — it needs no data and no teacher, and it is
where the two silent failure modes get caught:

```bash
CUDA_VISIBLE_DEVICES=0 $VENV/python -m alpamayo1_5_distill.scripts.validate_kv_distill 8 float32
$VENV/python -m pytest tests/test_kava.py -q     # 35 GPU-free tests
```

**K2. Pilot the KV cache on 200 samples** before committing to the full run — `N_C`
is what decides the disk bill and it cannot be predicted from configs:

```bash
ROOT=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/teacher_kv_lcdrive
CUDA_VISIBLE_DEVICES=0 $VENV/python -m alpamayo1_5_distill.scripts.generate_teacher_kv \
    config=cache_teacher_kv_lcdrive teacher=teacher_ar1_5_10b \
    cache_root=$ROOT m=16 lam=0.1 eviction=rkv mode=generate limit=200
# prints per-tier MB/entry, the projected GB for 38,340 clips, and the N_C histogram
```

Check `cot_text.shard0.jsonl` is non-empty — an empty CoT means the
`components_order` gotcha, and it shows up here before anything else.

To score by the **action expert** instead of the post-CoT text (the faithful reader),
swap in the expert-carrying teacher:

```bash
CUDA_VISIBLE_DEVICES=0 $VENV/python -m alpamayo1_5_distill.scripts.generate_teacher_kv \
    config=cache_teacher_kv_lcdrive teacher=teacher_ar1_5_10b_expert \
    importance_source=expert expert_timesteps=0.0,0.5,1.0 expert_noise_seed=0 \
    cache_root=$ROOT-expert m=16 lam=0.1 eviction=rkv limit=200
```

Keep the two caches in separate `cache_root`s: the eviction they imply differs
substantially (agreement ~0.44 at `M < N_C`), so they are a real ablation pair, not
interchangeable. `importance_source=none` gives diversity-only eviction and needs no
answer queries at all.

**K3. Build the full cache**, sharded across GPUs (each shard writes its own files,
so no merge step):

```bash
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s $VENV/python -m alpamayo1_5_distill.scripts.generate_teacher_kv \
      config=cache_teacher_kv_lcdrive teacher=teacher_ar1_5_10b cache_root=$ROOT \
      m=16 lam=0.1 eviction=rkv num_shards=8 shard=$s &
done; wait
```

**K3-alt. Or restore the prebuilt cache from HuggingFace** — on a second machine,
this is much faster than rebuilding, and far faster than copying over a slow site link
(measured: 2.85 MB/s between our two sites, vs 19.2 MB/s pulling from HF).

The M=8 tier, the sidecars and the Stage-1 warm-start checkpoint are mirrored to a
**private** dataset repo, tarred into 12 zstd shards because 38,336 small files is
hostile to the Hub (43 GB → 34 GB; bf16 K/V compresses ~21%):

> 🔒 `ac4462/alpamayo-kava-cache` — **private, Honda-internal.** Teacher KV activations
> derived from the PAI driving data plus generated CoT text. Do not make it public or
> re-share it.

```bash
DEST=/path/on/target
hf download ac4462/alpamayo-kava-cache --repo-type dataset --local-dir $DEST/dl

mkdir -p $DEST/teacher_kv_lcdrive/compressed_M8_rkv0.1 $DEST/checkpoint-3597
for f in $DEST/dl/kv_M8_shard*.tar.zst; do
  zstd -dc "$f" | tar -C $DEST/teacher_kv_lcdrive/compressed_M8_rkv0.1 -xf -
done
zstd -dc $DEST/dl/kv_M8_sidecars.tar.zst  | tar -C $DEST/teacher_kv_lcdrive -xf -
zstd -dc $DEST/dl/stage1_ckpt3597.tar.zst | tar -C $DEST/checkpoint-3597 -xf -

# MUST be 38336 — a short cache fails at TRAINING time as a KeyError, not at unpack
find $DEST/teacher_kv_lcdrive/compressed_M8_rkv0.1 -name '*.safetensors' | wc -l
```

| in the repo | |
|---|---|
| `kv_M8_shard{00..11}.tar.zst` | the `compressed_M8_rkv0.1` tier, 38,336 entries |
| `kv_M8_sidecars.tar.zst` | `index.rebuilt.json` + `cot_text.shard0.jsonl` — **required**, `cot_text` drives the slot vocab-init |
| `stage1_ckpt3597.tar.zst` | the Stage-1 warm start (`model.safetensors` + `config.json` only; the 28 GB of DeepSpeed state is not needed) |

**Not** in the repo, because they are public — pull them directly rather than copying:
`nvidia/Cosmos-Reason2-2B` (student backbone) and the `Alpamayo-1.5-10B` config (only
its tokenizer/trajectory settings are read). The `full/` tier (69 GB) is also omitted;
it is only needed to derive *other* `(M, λ, method)` tiers, which is better done on the
machine that already has it.

Then point the training config at it — remember these must all agree:

```bash
data.train_dataset.kv_cache_root=$DEST/teacher_kv_lcdrive
data.train_dataset.kv_tier=compressed_M8_rkv0.1
data.train_dataset.num_slots=8  data.collate_fn.num_slots=8  model.kava.num_slots=8
model.checkpoint_path=$DEST/checkpoint-3597
```

**K4. Train.** Use `train_kava`, not `train_hf` — it swaps in `KaVaTrainer`, and
without the per-term logging "total went down" cannot tell you whether `L_KV` did
anything:

```bash
$VENV/torchrun --nproc_per_node 8 -m alpamayo1_5_distill.train_kava \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_stage1_kava_cosmos2b_lcdrive \
    data.train_dataset.kv_cache_root=$ROOT
```

**K5. Sweep without re-running the teacher.** Every `(M, λ, method)` point is a
different eviction of the same forward:

```bash
for m in 8 32; do $VENV/python -m alpamayo1_5_distill.scripts.compress_teacher_kv \
    cache_root=$ROOT m=$m lam=0.1 eviction=rkv; done
for e in cosine attn crop; do $VENV/python -m alpamayo1_5_distill.scripts.compress_teacher_kv \
    cache_root=$ROOT m=16 eviction=$e; done
# then: data.train_dataset.kv_tier=compressed_M32_rkv0.1 model.kava.num_slots=32 \
#       data.collate_fn.num_slots=32   (all three must agree)
```

Also sweep `model.kava.{jacobi_iters,kv_loss_type,kv_loss_weight,kv_align,kv_layerwise_std}`.
`jacobi_iters` is the one with a latency cost: `T=1` rides the single prefill pass for
free, each further iteration adds a pass against ~85 ms of prefill in a 100 ms budget.

**K5-alt. Or skip training and pull the trained model.** The finished M=8 run is
mirrored, so evaluation or a Stage-2 handoff needs neither the cache nor 22 GPU-hours:

> 🔒 `ac4462/alpamayo-kava-2b-m8-lcdrive` — **private, Honda-internal.** 4.70 GB.
> Cosmos-Reason2-2B distilled from Alpamayo-1.5-10B on Honda's LCDrive data.

```bash
hf download ac4462/alpamayo-kava-2b-m8-lcdrive --local-dir $DEST/kava-2b-m8
```

Plain safetensors, not tarred — 6 files, so `hf download` and `from_pretrained` work
directly. The 33 GB `global_step7191/` DeepSpeed optimizer state is deliberately **not**
mirrored: inference never reads it, and it is 87% of the checkpoint directory.

⚠️ **Point it at `kava_checkpoint_path`, never `checkpoint_path`.** The checkpoint holds
59 KAVA tensors (`slot_embeddings [8, 2048]`, 56 projectors, `latent_proj`) beside 626
VLM tensors. `checkpoint_path` loads **only `vlm.*`** — the trained slots would be
silently replaced by fresh vocab-init and you would score a model that never existed.
`kava_checkpoint_path` restores the full state dict and asserts the KAVA tensors were
found:

```yaml
model:
  kava_checkpoint_path: $DEST/kava-2b-m8   # ✅  logs "restored 685 tensors (59 KAVA ...)"
  checkpoint_path: null                    # ❌  vlm.* only
  kava: {num_slots: 8, ...}                #     must match the trained M
```

What it was trained with — warm start from Stage-1 `checkpoint-3597`, 3 epochs over
38,336 LCDrive clips, `CE + 0.01·L_latent + 1.0·L_KV`, M=8 with the identity-init
projector, T=1, 21 h 46 m on one H100:

```
              start    end
ce_loss        2.89 -> 2.06     (improved — trajectory quality not sacrificed)
latent_loss   59.7  -> 0.150
kv_loss        2.16 -> 0.193
```

⚠️ **Do not summarise `gradshare_kv` as "89% → 21%".** Two endpoints hide a collapse;
the full series is what matters:

```
step        0    latent 111%   kv 89%
steps 300-600    latent   7%   kv 10%
steps  >6000     latent   3%   kv 17%
```

Both terms collapse within ~300 of 7,191 steps and spend the bulk of training at 5–11%.
The late rise in `kv%` is CE's own gradient shrinking (2.17 → 0.63), not `L_KV`
strengthening. So this run was effectively ~300 steps of KAVA followed by ~6,900 steps
of plain CE continuation — which is why the 1-epoch configs below replaced it, and why
over-training was a live explanation for its regression that had nothing to do with the
distillation terms. An earlier version of this README claimed "`L_KV` never went inert"
from the endpoint alone; that was cherry-picking and is retracted.

**K6. The dead-slot gate — RUN, and it is the most informative result here.** Zero the
slots at inference: if quality does not drop, the slots are decorative and `L_KV`
achieved nothing, whatever the loss curve said (`reasoning-setup-2b.md` §7a). See
[Trained and evaluated](#trained-and-evaluated-lcdrive-val-n500-paired) — at `T=1` the
slots are decorative, and at `T=2` they are not.

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

### KAVA status

- **Machinery validated, 7/7** on the real Cosmos-Reason2-2B — see the KV1–KV7 table
  above. 35 GPU-free tests in `tests/test_kava.py` (they caught a mask-broadcast bug
  in `kv_matching_loss` that would have crashed the first masked batch).
- **Model wiring checked** on the real student: 28 layers → layer map
  `[0, 1, 3, …, 32, 34, 35]`, `slot_embeddings [16, 2048]`, projector bank **58.7 M**
  params, all four `lr_multiplier` prefixes matching real parameters, the projector
  exactly the identity at init (rel-L2 0.0), and the gradient-checkpointing guard
  firing.
- **End-to-end forward + backward on real LCDrive data**, through the real config
  (dataset → collator → `forward` → `backward`), with fabricated KV targets for two
  clips — one at `N_C = 16` and one at `N_C = 7`, so both the full and the short-CoT
  masking path run:

  ```
  batch: input_ids (2, 3142)  slot_pos row0 2993..3008  teacher_kv_k (2, 36, 8, 16, 128)
  valid/row [16, 7]   slots precede <tfs>: True   labels off slots: True
  loss=52.47  ce=29.16  latent=20.75  kv=2.56  n_valid_slots=11.5   (= (16+7)/2 ✓)
  grads: slot_embeddings 43.2 · latent_proj 31.1 · kv_projector.k_proj.0 0.154
         kv_projector.v_proj.27 0.115 · backbone k_proj.0 104.0
  per-slot grads non-zero 16/16, pairwise cosine max 0.73  (≈1.0 would be broadcast)
  ```

  This caught a config bug worth remembering: `data.collate_fn` inherits
  **`_partial_: true`** from `sft_base`, so a class-target collator comes back as
  `partial(KaVaCollator, …)` and HF Trainer calls it as `collate_fn(features)` — the
  batch lands in `model_config` and a fresh collator is built per batch. The KAVA
  config sets `_partial_: false` explicitly.
- **✅ The full LCDrive-train cache is BUILT** (`slurm_teacher_kv_lcdrive.sh`,
  2 GPUs x 3 co-located shards, `COMPLETED` in **3 h 12 m**):

  | | |
  |---|---|
  | entries | **38,336 / 38,340** in both `full/` and `compressed_M16_rkv0.1/` |
  | disk | **135 GB** (256 buckets) |
  | `N_C` over all 38,336 | min 6 · p25 11 · **median 13** · p95 17 · max 101 · mean 12.70 |
  | throughput | 2.5 samples/s aggregate (0.42/s per shard) |
  | failures | 4 clips (0.01%) — see below |

  Validated after the fact, not just from the logs: 0 orphan `.tmp`, shard indexes
  pairwise-disjoint, every file's key present in the index, `red` sums to 1 per
  (layer, head), and a spot-checked entry loads finite `[36, 8, N_C, 128]` bf16.

  **4 clips have no entry**: the teacher hit `max_new_tokens` without ever emitting
  `<|traj_future_start|>`, so there is no handoff point to cache. `KaVaPAIDataset`
  now handles this with `allow_missing=true`, which attaches an all-masked target so
  the clip trains on CE alone — returning `None` would have taken down the whole step,
  since the shared collator stacks tensors and cannot drop a sample.

  **`M=16` engages eviction on only 7.8% of clips** across the full set (`M=8`: 79.2%),
  confirming the pilot. See the boxed note above.

- Earlier: an 8-clip smoke test and a 200-clip pilot, both green.

  ```
  [kv-cache] 200 new (200/200 scanned)  1.04/s
  [kv-cache] full: 200 entries, 1.79 MB/entry -> 67.2 GB for 38340 clips
  [kv-cache] compressed_M16_rkv0.1: 1.72 MB/entry -> 64.5 GB
  [kv-cache] N_C over 200 samples: min=6 median=13 max=46 (mean=12.4)
  ```

  **1.04 samples/s** ⇒ the full 38,340-clip build is **~10.2 h on one GPU**, ~2.6 h over
  4 shards — matching the ~1.1 samples/s this README predicted. Total disk **~132 GB**.
  Artifacts verified by loading them back: `k_pre`/`v` `[36, 8, N_C, 128]` bf16 all
  finite, `red` sums to 1 per (layer, head), `tfs_hidden` 4096-d, and
  `compress_teacher_kv.py` re-deriving the stored `sel_idx` bit-identically (rel-L2 0.0).
  The CoT text is real and scene-specific — *"Stop for the red traffic light since the
  signal is red"*, *"Keep distance to the lead vehicle since it is directly ahead in our
  lane"* — so the `components_order` gotcha is confirmed absent on this path.

  The pilot's one substantive finding is the **13-token CoT** and what it does to the
  `M` guidance; see the boxed note above. It also corrected this README's latency
  saving from ~420 ms to ~143 ms.
- **`importance_source="expert"` built and verified against the real 10B**
  (`teacher_ar1_5_10b_expert`, expert loaded, 22.7 GiB peak):

  | | Measured |
  |---|---|
  | expert depth == VLM depth (the per-layer correspondence this relies on) | 36 == 36 ✅ |
  | score shape / non-negativity | `[36, 8, N_C]`, min 5.4e-5, max 3.7e-2 ✅ |
  | reproducible at a fixed seed | bit-identical across runs ✅ |
  | seed / timestep grid actually matter | rel 0.031 / 0.291 ✅ |
  | genuinely a different signal from `vlm_post_cot` | rel **0.982** |
  | changes *which* tokens survive (at `M=4 < N_C=11`) | agreement only **0.438** (rkv), **0.390** (attn-only) |
  | per-head variety | **125** distinct index sets across 288 (layer, head) pairs |
  | cost | ~**1.0 s/sample** for 3 timesteps |

  ⚠️ Two things worth knowing. First, HF warns that `action_space.{accel,curvature}_{mean,std}`
  and the Fourier `freqs` were "newly initialized" — that is **cosmetic**: they are
  config-derived registered buffers, and the loaded values match `config.json`
  (`accel_mean` 0.029053 vs 0.02902694, the difference being bf16 rounding under
  `dtype: auto`). Second, at `M >= N_C` eviction is a no-op and *every* scoring method
  agrees perfectly — so a short-CoT sample cannot demonstrate that the score works.
  Compare methods at `M < N_C`.

### Trained and evaluated (LCDrive val, n=500, paired)

Four arms trained, all warm-started from Stage-1 `checkpoint-3597`, `M=8`,
`kv_align: projector`, `kv_loss_type: l1`, `latent_loss_weight: 0.0` (so `CE + L_KV`
only — the logged `latent_loss` is raw and unweighted). Evaluated against the Stage-1
baseline's own per-clip file on a **bit-identical 500-clip set** (verified: union ==
intersection == 500, every id present in the 23,331-clip baseline dump).

| arm | eff. batch | `min_ade` | `ade` | slots load-bearing? |
|---|---|---|---|---|
| Stage-1 baseline, no KD | 48 | **4.094** | **4.949** | — |
| CE-only control, `T=1` | 16 | 4.297 | 5.127 | — |
| KAVA `T=1` | 16 | 4.021 | 5.764 | **no** (−0.024 ± 0.053, n.s.) |
| KAVA `T=2` | 48 | 4.165 | 9.570 | **yes** (−3.78 ± 0.59, 6.4σ) |

Three findings, and the second corrects a reading of the first table column:

1. **`T=1` slots are decorative.** Zeroing them changes nothing at any horizon
   (−0.0008 to +0.0037, all |z| < 0.5). `T=1` is PCCoT with one iteration ≡ pause
   tokens, so the slots add width but never re-read their own output. `L_KV` still
   *helped* — it recovered the CE-only control's +0.203 regression — but as a
   regulariser on the weights, not through the slots. That is not the mechanism KAVA
   claims.
2. **`T=2` makes the slots load-bearing, and this survives scrutiny.** Zeroing costs
   −0.221 at 0.5 s and −1.850 at 3 s (5.0–6.4σ), and it holds on the 247 clips where
   neither arm is degenerate (−0.021 to −0.327, 3.2–4.0σ) — so it is not a tail
   artifact. **But quality did not improve.** Full-horizon `min_ade` is +0.071 ± 0.046
   (1.5σ) vs baseline, which reads as parity and is *underpowered*; the
   horizon-resolved columns are unambiguous and all significant: +0.010 (3.3σ) at
   0.5 s, +0.034 (3.9σ) at 1 s, +0.157 (5.4σ) at 3 s. Prefer the per-horizon numbers —
   the full-horizon column is dominated by long-horizon variance.
3. **`T=2` destabilised the trajectory *distribution*.** `ade` mean 4.95 → 9.57, with
   48/500 clips above 20 versus 8 for the baseline — and 37 of those 48 are clips the
   baseline handles fine (< 10). On the worst, `ade` ≈ 102 while `min_ade` ≈ 0.9: a
   near-perfect mode still exists but a typical draw is 100× off. This is real, not a
   metric artifact — [`metric_api.py:225`](../../src/alpamayo/metrics/metric_api.py#L225)
   sets `logprob = torch.zeros_like(...)` ("dummy logprob for now"), so `argmax` always
   returns 0 and **`ade` is sample 0's error, not the best or the mean over modes**.
   `min_ade` is best-of-K. Deployment gets one trajectory, so `ade` is arguably the
   number that matters more.

**`L_KV` has a floor at ~0.60.** It falls 3.13 → 0.62 by epoch 0.30 and then moves
0.02 over the remaining 70% of the epoch. Doubling Jacobi depth does not move it either
(`T=2`: 0.589 vs `T=1`: 0.574). So the limit is not compute or steps — it is alignment
or capacity. The two untested hypotheses are the 59 M-param per-layer projector
absorbing the loss instead of forcing the backbone to match (`kv_align: direct` tests
this with zero params) and an intrinsic 2B-vs-10B basis gap.

⚠️ **Per-clip metric dumps are per-rank, not gathered.** A multi-GPU eval prints a
correct aggregate to the log but writes only rank 0's shard to the JSON. A 2-rank,
1000-clip eval left a 500-row file strided `0, 2, 4, …, 998`, which silently pairs
against nothing. Run each eval arm as an independent single-rank job.

**In flight:** a CE-only control at `T=2` and effective batch 48, differing from the
KAVA `T=2` arm in exactly one variable (λ₂ 1.0 → 0.0). It decides whether the `T=2`
slot dependence and the `ade` tail come from `L_KV` or merely from Jacobi refinement
making the slots input-dependent. Until it lands, neither is attributed.


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
- **Train full-depth, then prune — the plan for the expert's latency.** Rather
  than trading coverage for speed upfront, keep the faithful 28-layer expert for
  training and recover latency afterwards by dropping layers. This works because
  **`layer_idx` travels with the module**: slicing the expert's `ModuleList`
  leaves each surviving layer reading the exact cache layer it was trained
  against. Verified on a toy model — keeping positions `[0,3,5,7]` of an 8-layer
  expert yields `layer_idx == [0,3,5,7]`, and the forward reads cache indices
  `[0,3,5,7]`. No patch required.

  Pruning therefore *produces* a strided expert (e.g. 7 layers reading
  `[0,4,9,13,18,22,27]`) with two advantages over choosing that map upfront: the
  kept layers are selected by **measurement** rather than a uniform-stride guess,
  and their weights were trained in full-depth context. Suggested procedure:
  rank layers by ablation (drop one, measure minADE/corner-distance on val) or by
  residual-contribution norm `‖out − in‖ / ‖in‖`; keep the top-K; then briefly
  re-run Stage 2 to let them adapt (cheap — the VLM is frozen, only ~0.5 B trains).

  Note the two levers multiply, so pruning may not even be needed:

  | expert | 10 steps | 2 steps |
  |---|---|---|
  | 28 layers (current) | 136 ms | **27 ms** |
  | 14 layers | ~70 ms | ~14 ms |
  | 7 layers | ~40–77 ms | ~8–15 ms |

  Caveat: 28 → 7 is a 4× depth cut, aggressive enough that recovery fine-tuning
  is expected to be necessary rather than optional. If quality doesn't recover,
  the pruned expert can itself be distilled from the full-depth one
  (expert → expert), which is a cheaper problem than the VLM distillation above.

See the design menu (all VLM/expert/orchestration options with pros/cons) that
this recipe was distilled from in the planning notes.
