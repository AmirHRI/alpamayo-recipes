# Setting up reasoning slots on Cosmos-Reason2-2B
2026-07-28 · concrete implementation spec · verified config: 28 layers, hidden 2048, 16 heads, 8 KV heads, head_dim 128, ViT depth 24 / patch 16

---

## §0 Why this saves latency at all

The comparison is **not** "slots vs nothing." It is **"slots vs autoregressive text CoT."** Against
nothing, slots cost a little more. Against text CoT, they cost ~700× less.

### The one-sentence version

> **Text CoT's cost is not the thinking — it is the round-trip through the vocabulary.**

Generating one CoT token means: compute hidden state → project to the ~150k-token vocabulary → sample →
embed the sampled token back → **run the whole model again** for the next token. That re-embedding step
is what forces the passes to be *sequential*. Latent slots skip the projection/sample/re-embed round
trip entirely, so all K "thoughts" can be computed in the **same** forward pass.

### Why sequential passes are so expensive: decode is memory-bound

At batch size 1, each autoregressive forward pass must read **every model weight** from memory, while
performing only ~`2 × N_params` FLOPs. The arithmetic is trivial; the memory traffic is everything.

For the 2.44B student at FP8 on Thor (~273 GB/s, assume 66% achieved ≈ 180 GB/s):

| | Cost |
|---|---|
| weights to read per decode step | 2.44 GB |
| **time per autoregressive token** | 2.44 GB / 180 GB/s ≈ **13.6 ms** |
| arithmetic actually done per token | 2 × 2.44e9 ≈ 4.9 GFLOPs → **microseconds** |

So during decode the GPU is idle >99% of the time, waiting on memory. **That idle compute is what
parallel slots use, for free.**

### The comparison, concretely

| | Sequential passes | Extra FLOPs | Time on Thor |
|---|---|---|---|
| **40 text CoT tokens** | 40 | — | **≈ 542 ms** |
| **32 parallel slots** | **0** (folded into the existing prefill) | 32 × 4.9 GFLOPs = 156 GFLOPs | **≈ 0.8 ms** |

Context: the vision prefill is already ~1000 tokens ≈ 4.9 TFLOPs ≈ **25 ms**. Adding 32 slots makes it
~25.8 ms. **The slots are ~3% of a prefill you are already paying for.** The weights get read once
either way.

That is the entire latency argument, and it is why the roofline in
`backbone-and-latency-budget.md` matters: on Thor, AR decode costs 222–890 ms at *every* model size in
this family, so eliminating sequential decoding is a feasibility requirement, not an optimization.

### So what do the blank embeddings actually buy?

Transformer computation is organized **per position**. Each position gets its own residual stream — a
full pass through all 28 layers, attending to everything before it. K extra positions = **K extra
28-layer computations** that can each read the whole scene and cross-talk with one another.

Think of it as a scratchpad with K cells. The learned embedding is just the cell's *address* ("this is
cell 3") — blank by design. The model fills the cells by attending to the scene; the action head then
reads the filled scratchpad. Text CoT was doing the same thing, except every cell had to be decoded
into a word and re-encoded, which is what costs 13.6 ms per cell.

### The honest limitation — and why recurrence is required

Parallel slots buy computational **width**, not **depth**.

In one forward pass, slot 5 at layer *L* can attend to slots 1–4 at layers *< L*. So information flows
between slots, but the total serial composition is bounded by the **28 layers** — adding more slots does
not add depth. Autoregressively, token 5's *input* is token 4's fully-computed 28-layer output, so N
tokens give ≈ 28 × N layers of serial composition.

That is precisely the filler-token TC⁰ result: parallel tokens add power *within* TC⁰ but cannot escape
it, no matter how many you add. Serial CoT can.

**Recurrence recovers the depth across time:** carrying the slot state over T frames gives ≈ 28 × T
serial composition, while each individual frame still pays only one parallel pass. This is why
recurrence is load-bearing rather than a nice extra — without it, the architecture is strictly less
expressive than text CoT, not merely cheaper.

| | Serial depth | Latency per frame |
|---|---|---|
| Text CoT, N tokens | 28 × N | N × 13.6 ms |
| K parallel slots | 28 | ~0.8 ms |
| K parallel slots + recurrence over T frames | 28 × T | ~0.8 ms |

⚠️ Thor's 273 GB/s should be confirmed on your board; the conclusions hold under ±40% error because the
gap is ~700×.

---

## 1. The parameters you add — almost nothing

```python
# reasoning slots: K learnable embeddings at LM input dim
Q = nn.Parameter(torch.randn(K, 2048) * 0.02)     # K=32 → 65,536 params
```

**65k parameters — 0.003% of the 2.44B backbone.** The capacity does not come from these weights; it
comes from the K extra hidden-state positions computed over them. Worth stating in the paper, because
it makes the "extra computation, not extra parameters" argument concrete.

**Role typing** (from `architecture-spec.md`): interleave attribution / outcome by index parity —
`Q[0::2]` = attribution slots, `Q[1::2]` = outcome slots. Same tensor, different supervision. Interleaved
so every nested prefix contains both roles.

**Initialize K ≈ teacher CoT length (~40)**, then compress via the nested structure. Rationale in §6:
Stage 1 supervises the slots by *replacing real CoT token positions*, so they should start at
comparable count.

---

## 2. Where they go — and why no custom mask is needed

Append after everything, at the end of the sequence:

```
[system][<|vision_start|> V_t <|vision_end|>][ego][route][ Q_1 … Q_K ]
                                                            ↑ appended
```

Cosmos-Reason2-2B is a **decoder-only causal LM**, so the standard causal mask already gives exactly
the pattern you want, for free:

- slots attend to all vision / ego / route tokens ✓ (they precede)
- slot *i* attends to slots *j < i* ✓ (causal among slots → the ordering the nested budget needs)
- slots do **not** contaminate the vision/text representations ✓

**No mask surgery for the base version.** Just append embeddings and run one forward pass.

### 🔑 Causal masking makes the nested budget *exactly* free

Because slot *i* only sees *j < i*, the hidden states `h(Q_1…Q_N)` are **bit-identical** whether you
compute K slots or only N. So:

```
one backbone forward pass over all K slots
    → |S| cheap action-head passes, each cross-attending to prefix 1..N
```

The nested loss over `S = {0,1,2,4,8,16,32}` costs **one** backbone pass plus 7 small head passes — not
7× a training step. Prefix semantics are exact, not approximate. This is the single best reason to use
causal rather than bidirectional attention among slots.

### ⚠️ M-RoPE is the real implementation gotcha

Qwen3-VL uses multimodal RoPE with separate temporal/height/width position axes (vision tokens get
spatial indices; text tokens get identical t=h=w). **Check `get_rope_index` / the M-RoPE implementation
in the modeling code before touching anything.** Default for slots: treat them as text tokens
continuing the sequence — identical t/h/w equal to their sequence position.

This becomes load-bearing under recurrence (§4): as cached frames age, their relative position to the
current frame changes every step. Use **pre-RoPE key caching with rotary applied on the fly** at the
recomputed position ids — precisely Kamera's mechanism (arXiv 2606.23581). Getting this wrong produces
a silently degraded model rather than a crash, so validate with the null test in §7.

> ✅ **Measured — see §9.** Appending slots as ordinary text tokens gives correct M-RoPE for free
> (`t=h=w` continuing from the prefix), and vision keeps its 2-D spatial ids. The gotcha is
> confirmed real and silent: feeding slots naive `arange` positions against a cache is **~27 % off
> with no error** (cos 0.733), because the 78-token prefix has a max position id of only **21** —
> 64 vision tokens span 64 sequence slots but ~8 position steps. §9 also gives the injection
> pattern that keeps vision-merge and deepstack intact.

---

## 3. Action-head interface

AR1's trajectory decoder already *"cross-attends to both vision features and the CoT's key-value (KV)
cache"* (VLADriveBench). Keep that channel; change only what fills it.

```
flow-matching expert  ──cross-attn──►  [ KV(vision) , KV(Q_1..Q_N) ]
```

**Convenient:** KV is `8 heads × 128 = 1024`-dim. Build the expert at **hidden 1024** and it
cross-attends to the backbone KV **with no projection at all**.

**Size target ~0.2–0.25B** (not the teacher's 2.3B):

| | |
|---|---|
| 12 DiT-style blocks, hidden 1024 | self-attn 4·1024² + cross-attn 4·1024² + MLP 2·1024·4096 ≈ 17M/layer |
| total | **≈ 200M** |
| student total | 2.44B + 0.2B ≈ **2.65B** → ~2.7 GB at FP8, prefill-bound on Thor |

Precedent for shrinking it: DriveVLA-W0's *"lightweight action expert to address the inference latency
for real-time deployment."*

---

## 4. Recurrence

Retain the slots' KV across frames (all 28 layers), prepend to the current frame's attention:

```
KV_cache(Q, t-1)  ──►  prefix in attention for frame t  ──►  Q_t hidden states
```

**Memory is trivial:** `28 layers × 2 (K,V) × 32 slots × 1024 dim × 2 bytes ≈ 3.7 MB`.

> ✅ **Measured — see §9.** Slicing the stock `DynamicCache` down to slot positions works (86 → 8)
> and the carried size comes out at **exactly the 3.7 MB predicted** for K=32. Frame *t+1* runs
> against it and the carry demonstrably influences the output (rel diff 1.51 vs no-carry, 1.16 vs a
> zero-carry control). **Caveat:** RoPE phase is frozen at cache-write time, so *monotonic*
> positions work out of the box while *fixed* relative positioning needs the pre-RoPE cache — over
> a 4–8-frame BPTT window, monotonic is sufficient.

Add the two safety mechanisms from `closed-loop-causal-recurrent.md` §III:

```python
h_carried = g * KV_cache(t-1)          # g: learned gate, scalar or per-layer
if t % N_refresh == 0: drop cache      # bounded recompute — hard lock-in guard
```

Train with **truncated BPTT** over short windows (4–8 frames) before attempting longer.

### ⚠️ Both mechanisms are load-bearing — measurement in §9 says so

§9 measured carry influence as a function of slot age: it decays **~4 % over 8 frames**, i.e. it is
flat. RoPE attenuation provides *no meaningful natural forgetting*. Consequences:

- **`N_refresh` is the primary staleness bound, not a backstop.** With no natural decay, a stale
  belief retains essentially full influence right up until the hard reset. That makes the
  VLADriveBench lock-in failure mode (Alpamayo hallucinating speed bumps; 4/4 route failures) *more*
  concerning here, not less — budget `N_refresh` accordingly rather than treating it as a rare
  safety valve.
- **A scalar `g` is probably insufficient.** Any age-dependence must now be *learned explicitly*,
  since the geometry supplies none: prefer **per-slot or per-layer gating**, or feed an explicit
  age/staleness feature, over a single scalar.

### 🔑 Sliding window vs. accumulating — decide before Stage 4

The §9 trace (`86 → 8 → 94`) is a **sliding window of 1**: frame *t+1* carries only frame *t*'s
slots. State that explicitly, because the alternative — accumulating `t, t+1, …` — has a very
different risk profile:

| | Carried length | Staleness | Lock-in risk |
|---|---|---|---|
| **Window of 1** (recommended default) | constant `K` | structurally bounded to one frame | lower |
| Accumulating | grows `K` per frame | unbounded until `N_refresh` | **higher — and the 4 % decay result makes it worse**, since old slots never fade on their own |

Recommend the **window of 1** as the default: it bounds staleness structurally rather than relying
on a decay that was measured not to exist. Revisit only if a window of 1 proves too forgetful.

---

## 5. Losses

| Loss | What | Notes |
|---|---|---|
| `L_traj` | flow matching on trajectory | the base objective |
| `L_cot` | CE on teacher text CoT at slot positions | Stage 1 only, **annealed to 0** |
| `L_attn` | Drive-KD attention distillation | student layers (1, ~8–20, 27) ↔ teacher (1, group-matched, 35); **head-mean** so 16-vs-32 heads is free |
| `L_kv` | direct KV matching at slot positions | **no projection needed** — identical 8×128 geometry |
| `L_nested` | `Σ_{N∈S} L_traj(a \| Q_1..Q_N)` | one backbone pass, |S| head passes |
| `L_causal` | `d(Δ_S, Δ_T)` | meta-action space (primary) + slot-hidden space (auxiliary) |
| `L_wm` | droppable future-frame head | DriveVLA-W0; deleted at deployment |
| `L_vl` | VL co-training mix | prevents Qwen-RobotNav's reactive-mapper collapse |

Width projection for any hidden-state loss: RT-VLA's learnable `φ: 2048 → 4096`.

---

## 6. Training stages

| Stage | What | Why |
|---|---|---|
| **0** | slots present, **unsupervised**; measure vs no-slot baseline | this is **gate G3's control arm** — expect ≈ no gain, reproducing the pause-token finetuning-only result |
| **1** | student generates the teacher's **text CoT** (standard SFT), slots occupy those positions | gives the slots something real to internalize; you cannot supervise continuous slots with discrete text any other way |
| **2** | **LaRA curriculum**: progressively replace CoT token embeddings with learned slots on a decay schedule, keeping `L_cot` on the remaining text positions until it reaches zero. Add EMA target encoder on any latent target. Switch `L_nested` on partway. | this is the text→latent internalization; the annealing is what makes it converge |
| **3** | add `L_causal` on the offline counterfactual dataset | the dense causal signal; also compress K downward via nesting |
| **4** | add recurrence, truncated BPTT | temporal consistency + TC⁰ escape |
| **5** | closed-loop RL, World Engine recipe (KL-to-prior, dense reward, hard-frame mining, experience mixing) | |

Stage 0 vs Stage 2 **is** the G3 experiment: if 0 ≈ baseline and 2 > baseline, the design is sound and
you have explained the pause-token negative result. If both fail, the finetuning-only regime is the
blocker → fall back to serial latents, or continued pretraining with slots present.

---

## 7. Diagnostics — do not skip these

The failure mode is **dead slots**: training converges, quality is fine, and the slots contribute
nothing. Check explicitly.

**(a) Ablation health check — the primary test.** Zero out all slots at inference. If quality does not
drop, the slots are decorative. Run this every checkpoint.

**(b) Attention mass.** Fraction of the action head's cross-attention going to slot KV vs vision KV.
Should be non-trivial and should *grow* through Stage 2.

**(c) Linear probe.** Train a linear head on slot hidden states to predict the CoC-named critical
agent's identity and position. If it decodes, the slots carry causal content.

**(d) 🔑 Latent slot-splice — the latent analogue of VLADriveBench's CoT splice.** This is how you run
their causal-intervention protocol on a latent model, and you need it for evaluation:

| Condition | Expected |
|---|---|
| **self-splice**: substitute the same scene's own slot hiddens | **exactly 0.000** — the correctness gate, matching VLADriveBench's control |
| **cross-splice**: substitute slot hiddens from a *different* scene (e.g. one with a pedestrian, into one without) | large, directionally correct action change if reasoning is causal |
| **shuffle**: permute slot order | should matter (ordering is meaningful under causal masking) |
| **noise**: replace with Gaussian noise at matched scale | should degrade, not silently no-op |

If cross-splice does nothing, the slots are epiphenomenal — the same verdict VLADriveBench delivered on
ORION, and the thing `L_causal` is designed to prevent.

---

## 8. Order of work

1. ~~Append slots, verify a forward pass runs and **M-RoPE position ids are correct**~~ ✅ **done —
   see §9.**
2. Measure **latency vs K on Thor**. Expect nearly flat. Cheap, and a reportable result.
3. Stage 0 (unsupervised slots) → confirms the pause-token baseline.
4. Stage 1 + 2 (text CoT → latent curriculum) → **this is G3's answer**.
5. Shrink the action expert to ~0.2B, retrain the head.
6. Then Stages 3–5.

Steps 1–4 are the minimum viable experiment for the whole architecture, on a single node.

---

## §9 Validation on the real model — 2026-07-29

Step 1 executed against **Cosmos-Reason2-2B** (fp32 for clean numerics, sdpa attention), real
vision+text batch: **78-token prefix (64 vision tokens) + K=8 slots** appended.

### Results

Reproduce: `python -m alpamayo1_5_distill.scripts.validate_reasoning_slots [K] [dtype]`

| Check | | Evidence |
|---|---|---|
| Placeholder token collides with nothing | ✅ | id 13 `'.'` vs `image/video/vision_start/vision_end` = 151655/151656/151652/151653; not in `all_special_ids` |
| M-RoPE — vision keeps 2-D spatial ids | ✅ | over 64 vision tokens: `t` uniq=1, `h` uniq=8, `w` uniq=8 |
| M-RoPE — slots are text-like, contiguous, correctly placed | ✅ | `t=h=w`, +1 each, first slot at `prefix_max + 1` |
| **Nested-prefix invariance** (§2's load-bearing claim) | ✅ | `h(Q_1..Q_N)` for N=1,2,4 vs K=8: **cos = 1.000000000**, rel-L2 ≈ 2e-6 |
| Slots don't contaminate the prefix | ✅ | unchanged to round-off: **exactly 0** in bf16, rel-L2 ≈ 2.5e-6 in fp32 |
| KV-cache equivalence | ✅ | prefill prefix → feed slots, vs one pass: cos = 0.99999994, rel-L2 ≈ 3e-6 |
| **Splice instrument** (§7d gate) | ✅ | self-splice **exactly 0.000**; cross-splice rel-L2 = 0.166 |
| Slot-only carry across frames (§4) | ✅ | see below |

**The nested budget in §2 is exact, not approximate.** One backbone pass genuinely serves every
prefix in `S` — verified to fp32 round-off, so `L_nested` really does cost 1 backbone pass + |S|
head passes.

### ⚠️ The M-RoPE gotcha is real — and it is silent

Negative control: feed the slots **naive `arange` positions** instead of the true M-RoPE ids when
running against a cache.

```
naive arange positions vs correct M-RoPE:   cos = 0.733,  rel-L2 = 0.711
```

**~27 % off, no crash** — exactly the "silently degraded model" failure mode. The cause is concrete:

> the prefix is **78 tokens long but its maximum position id is only 21**, because 64 vision tokens
> occupy 64 *sequence* slots but only ~8 *position* steps per axis.

So `arange` puts slot 0 at position 78 when it belongs at **22**. Any code path that feeds slots
against a KV cache must carry the real M-RoPE ids.

### The injection pattern that works

Keep `input_ids` intact — Qwen3-VL needs it for **two** separate things: `get_rope_index` (spatial
M-RoPE) *and* `get_placeholder_mask` (where to `masked_scatter` vision embeddings, plus the
**deepstack** features injected at several layers). Passing only `inputs_embeds` silently loses both.

```python
# 1. append K benign placeholder tokens (NOT vision tokens) to input_ids
ids = torch.cat([ids, torch.full((1, K), PLACEHOLDER)], dim=1)

# 2. swap the slot rows in via a forward hook on embed_tokens. This runs BEFORE
#    the vision masked_scatter, so vision merge + deepstack are untouched.
#    Index by an EXPLICIT position tensor, never `[-K:]` -- see below.
def hook(mod, args, out):
    n = slot_pos.numel()
    out = out.clone(); out[:, slot_pos, :] = slots[:n]; return out
emb.register_forward_hook(hook)

# 3. positions need no custom code: slots are ordinary text tokens, so
#    get_rope_index gives them t=h=w continuing from the prefix automatically.
```

**Do not use `out[:, -K:, :]`.** The trailing-K assumption holds only for a single prefill pass and
breaks the moment a forward covers just the new tokens against a cache — i.e. exactly the Stage-4
recurrence path.

> **Provenance of the nested-invariance number.** While switching to explicit positions I briefly
> omitted the `[:n]` slice, which **crashed loudly** (`shape mismatch: value tensor of shape
> [8, 2048] cannot be broadcast to indexing result of shape [1, 1, 2048]`) — it never produced a
> number, so there is no false-positive risk. The `cos = 1.000000000` figure comes from the earlier
> fp32 script, whose hook (`out[:, -n:, :] = slots[:n]`) was correct for that test; the current
> script re-measures it independently via explicit positions and agrees at rel-L2 ≈ 2e-6. The N set
> shrank from {1,2,3,4,6} to {1,2,4} only to keep runtime down — nothing failed. Net: the
> load-bearing claim is confirmed by **two independent injection paths**.

Also verify the placeholder id collides with nothing `get_placeholder_mask` or the deepstack path
keys on — a token that is special for some *other* reason would fail in the same silent way M-RoPE
does. (Checked above: `'.'` = 13 is clean.)

### §4 recurrence: carrying slot KV across frames — works

Carrying **only** the slots' KV (dropping the frame's vision/text KV) is a three-liner on the stock
`DynamicCache`:

```python
for lyr in cache.layers:                      # keep just the slot positions
    lyr.keys   = lyr.keys[:,   :, P:P+K, :].contiguous()
    lyr.values = lyr.values[:, :, P:P+K, :].contiguous()
```

| | Measured |
|---|---|
| cache length after slicing | 86 → 8 ✅ |
| carried size | 0.92 MB at bf16 for K=8 → **3.7 MB at K=32** — matches §4's estimate exactly |
| frame *t+1* runs against the K-length cache | ✅ cache 8 → 94, finite |
| carry is **connected**, not silently dropped | ✅ rel diff **1.51** vs no-carry, **1.16** vs zero-carry |

The zero-carry control matters: it shows the effect comes from the carried *content*, not merely
from the presence of K extra attendable positions.

> ⚠️ **Read this as plumbing, not as a positive result.** "Carry influences the output" and "carry
> helps" are different claims and only training separates them. On an **untrained** model — which has
> never seen carried slot KV — a perturbation this large (rel > 1) most plausibly means the carried
> state is **off-manifold and disrupting** the forward pass, not informing it. What §6 establishes is
> the necessary precondition: the wiring is connected and fails loudly rather than silently. Nothing
> more.

### 🔑 RoPE phase is frozen at cache-write time — two regimes

HF's `DynamicCache` stores **post-RoPE** keys, so a carried slot's rotation is baked in at the
position it was written. Changing the offset assigned to the next frame changed the result by
**0.92 relative** — this is not a free choice.

| Regime | Works today? |
|---|---|
| **Monotonic positions** — frame *t+1* continues after frame *t* | ✅ stock HF, nothing to build. Relative distances stay meaningful; it is long context with holes. |
| **Fixed relative position** — slots always sit immediately before the current frame | ❌ requires **re-rotating** the cached keys ⇒ store them *pre*-RoPE and apply rotary on the fly (the Kamera mechanism named in §2). Custom cache class. |

**Practical read:** over the 4–8-frame truncated-BPTT windows §4 specifies, monotonic positions
drift only ~30 positions/frame — comfortably inside the trained range. Combined with the
`if t % N_refresh == 0: drop cache` guard already in §4, which bounds drift for long rollouts, the
pre-RoPE cache is likely **deferrable** until a long-horizon run actually shows degradation.

### ❗ RoPE attenuation does *not* give you free forgetting

If relative distance grows every frame under monotonic positions, does the carry's influence decay
on its own? Measured directly — hold a frame-1 carry fixed, advance frame-2's positions by
`age × 30`:

| slot age | 1 frame | 2 frames | 4 frames | 8 frames |
|---|---|---|---|---|
| influence (rel vs no-carry) | 1.5080 | 1.4728 | 1.4703 | 1.4455 |

Decay is **monotonic but negligible — ~4 % over 8 frames.** Two consequences:

- **Good:** there is no sharp cliff, so nothing forces the pre-RoPE cache for short windows.
- **Important:** monotonic positions are **not** doing any of the retention gate's job. Do not
  count on RoPE attenuation as a forgetting mechanism — the learned gate `g` and `N_refresh` in §4
  are carrying that load entirely, and remain load-bearing.

⚠️ Measured on an **untrained** model whose carried state is plausibly off-manifold; the decay
profile could differ once the slots are trained. Worth re-running at Stage 4.

### The §7(d) splice instrument is validated

The evaluation protocol for the whole causal-intervention line of work depends on extract→re-inject
being lossless. It is:

| Condition | Measured | Required |
|---|---|---|
| **self-splice** — re-inject the same scene's own slot KV | **0.000e+00 exactly** | exactly 0 ✅ |
| **cross-splice** — inject a different scene's slot KV | rel-L2 = 0.166 | non-trivial ✅ |

Self-splice hitting *exact* zero means the instrument itself introduces no drift, so any effect seen
in a cross-splice experiment is attributable to the substituted content rather than to the splicing
machinery. That gate is now green **before** training, which is when you want it — the same class of
silent-failure risk as the M-RoPE bug, caught the same way.

### Measurement note — do not use absolute tolerances

A first pass in bf16 appeared to fail the nested-prefix and cache checks with errors of exactly
`8.0` and `32.0`. Those are **1 ulp**: hidden states reach magnitudes ~1350 ("massive activations"),
where bf16 ulp is 8, and ~4096 where it is 32. In fp32 the same checks give rel-L2 ≈ 2e-6. Use
relative L2 / cosine, never an absolute threshold, when validating these.

### Gradient actually reaches Q — the check that decides whether Stage 0/1/2 mean anything

Forward mechanics can all be right while the *training* path is silently broken: if gradient never
reaches `Q`, Stages 0/1/2 still run, converge, and show no gain — and that reads as "reproduced the
pause-token negative result" when it is a plumbing bug. Reproduce with
`python -m alpamayo1_5_distill.scripts.validate_slot_gradients`.

| Check | | Result |
|---|---|---|
| grad reaches `Q` at all | ✅ | `‖Q.grad‖ = 1.16e6`, non-zero at **8/8** slot indices |
| every slot gets a **distinct** grad (no broadcast bug) | ✅ | per-slot norms span 1.9e5–9.4e5; pairwise cosine max **0.469**, mean 0.095 (≈1.0 would mean broadcast) |
| grad flows through a **carried** cache (§4 BPTT) | ✅ | sliced carried keys stay attached to the graph |

**`Q` is *not* in `model.parameters()`** — it is an `nn.Parameter` living outside the model, so it
will be silently skipped by any optimizer built the usual way:

```python
optim = AdamW([{"params": model.parameters()},
               {"params": [Q], "lr": lr_Q}])      # Q needs its own entry
```

#### ⚠️ Q's gradients are 40–1100× larger per element than the backbone's

| | per-element grad norm | ratio vs `Q` |
|---|---|---|
| `Q` (16 k params) | 9.10e3 | — |
| lm layer 0 `q_proj` | 2.25e2 | 40× |
| lm layer 27 `mlp.down` | 3.15e1 | 289× |
| lm layer 27 `q_proj` | 8.07e0 | 1128× |

Partly structural: `Q` sits at the input, so its gradient accumulates through all 28 layers of
backprop (visible in the layer-0 vs layer-27 spread too). Three practical consequences:

1. **This likely inverts the usual "new params need a higher LR" intuition.** The signal reaching
   `Q` is already very strong; a higher LR on top may be actively wrong. Start at parity or below
   and tune from measurement.
2. **Global grad-norm clipping needs watching.** A 16 k-parameter tensor contributing `1.16e6` of
   norm can eat a disproportionate share of the clip budget and quietly scale down the backbone's
   effective step. Consider clipping `Q` in its own group.
3. **Adam largely normalises raw magnitude away** per-parameter, so this matters most for clipping,
   weight decay, and any SGD-flavoured optimiser — less for the bare LR.

⚠️ Measured against **3 representative weight matrices**, not the full-backbone gradient norm, and
on an untrained model with a random-projection loss. Treat the ratios as order-of-magnitude.

#### Truncated BPTT: you must detach at the window boundary

G5 confirms the carried keys **remain attached to the autograd graph** after slicing. That is what
makes BPTT *within* a window work — and it also means that without an explicit
`cache.detach()`-equivalent at the truncation boundary, the graph will extend across every frame
ever carried and memory will grow without bound. Standard truncated-BPTT hygiene, but the slicing
operation gives no hint that it preserves the graph, so it is easy to miss.

### Downstream consequence: massive activations ⇒ plan for outlier-aware quantization

The ~1350-magnitude hidden states above are the known **massive-activations** phenomenon, and they
are exactly what breaks naive low-precision quantization — the reason FlashDrive needed ParoQuant to
"suppress outliers and prevent compounding errors" for W4A8. Since the Thor path assumes FP8/NVFP4,
budget for **outlier-aware** quantization rather than vanilla per-tensor scaling.

Second-order: massive activations frequently coincide with **attention-sink** dimensions. If sinks
dominate attention mass over the prefix, diagnostic §7(b) — attention mass to slot KV vs vision KV —
needs a **sink-excluded variant**, or slot attention will look misleadingly small.

### Not covered by these tests

Forward-pass mechanics only. Untested: whether the carried state stays **stable** over long rollouts
rather than blowing up or locking in (that is what the gate `g` and `N_refresh` are for), and
everything in §7 that requires a trained model — the ablation health check, attention mass, and the
linear probe. Note the splice *instrument* is validated above, but the splice *experiment* (does
cross-splice change actions in a directionally correct way?) still needs a trained model and an
action head.

### Next step, with a correction to §8 step 2

Measure **latency vs K** — but not at this test's token count. 64 vision tokens is a far harsher
case than reality: 32 slots on 64 vision tokens is **+50 %** sequence length, whereas on a realistic
~1000-token multi-camera prefill it is **+3 %**. Measuring at 64 would badly understate how cheap
slots are.
