# Alpamayo 1.5 SFT — Qwen 3.5 backbone (experimental)

Experimental variant of [`alpamayo1_5_sft`](../alpamayo1_5_sft/) using Alibaba's
[Qwen 3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B) (0.8B and 2B) as the VLM
backbone instead of Qwen3-VL/Cosmos-Reason2, following the same "swap the VLM,
keep the ~20% expert:VLM ratio" pattern as `alpamayo1_5_sft`'s Cosmos-Reason2-2B
variant.

## Why a separate recipe

Qwen 3.5 requires `transformers>=5.2` (a different major version than every
other recipe's pinned `transformers==4.57.1`) and a patched copy of the
upstream `alpamayo_r1` package (see below). Rather than touching the shared
pin, this recipe is a fully isolated `uv` project with its own venv, so every
other recipe is completely unaffected.

## What's different from `alpamayo1_5_sft`

- **`pyproject.toml`**: `transformers==5.14.1` (via `override-dependencies`,
  since `alpamayo_r1`'s own `pyproject.toml` still declares `4.57.1`).
- **`[tool.uv.sources]`**: `alpamayo_r1` points at
  [`AmirHRI/alpamayo`](https://github.com/AmirHRI/alpamayo), branch
  `qwen3_5-integration` (forked from the exact commit every other recipe's
  `uv.lock` pins) with a real patch on top — see "Upstream patch" below.
- **`configs/models/qwen3_5_{0_8b,2b}{,_expert}.yaml`**: new model configs,
  analogous to `cosmos_reason2_2b{,_expert}.yaml`.
- **`models/sft_alpamayo_r1.py`**: `expert_cfg` only overrides
  `num_hidden_layers` (not `hidden_size`/`intermediate_size`/`num_attention_heads`/
  `head_dim` the way the Cosmos-Reason2 variant does) — Qwen 3.5's `head_dim`
  (256) differs from the value hardcoded in the 10B's inherited `expert_cfg`
  (128), so reusing it verbatim would build an expert whose full-attention
  layers can't consume the VLM's own KV-cache shape. Also recomputes
  `vocab_size` after overriding `vlm_name_or_path` (see "Bugs found" below),
  and adds `mm_token_type_ids` before every VLM forward/generate call.
- **`hydra_compat.py`** (new): works around a `transformers>=5.2` /
  `hydra-core<=1.3` incompatibility — see docstring.

## Upstream patch (`alpamayo_r1`)

The local clone adds:
- `models/hybrid_cache.py`: makes the KV-cache crop/detach logic
  (`AlpamayoR1`'s "expert continues the VLM's own cache" mechanism) work for
  hybrid linear-attention/full-attention backbones like Qwen 3.5's Gated
  DeltaNet layers, which use a different cache convention than plain KV
  layers. Confirmed via direct testing (construct a smaller "expert" clone,
  feed it a larger model's cache non-causally, across repeated steps) that
  this generalizes correctly, including for linear-attention layers.
- `models/multimodal_inputs.py`: computes `mm_token_type_ids` (required by
  Qwen 3.5, not by Qwen3-VL/Cosmos-Reason2) from `input_ids`, since
  `alpamayo`'s processor pipeline never calls the high-level API path that
  would produce it automatically.
- `models/base_model.py`: real `vlm_backend` dispatch (auto-detected via
  `AutoConfig`, so existing Qwen3-VL/Cosmos-Reason2 callers need no changes),
  new `_initialize_qwen3_5_vlm`, generic `get_input_embeddings`, and a
  `tie_weights` signature fix for a `transformers>=5.2` API addition
  (`recompute_mapping` kwarg) that would otherwise break here regardless of
  VLM backend.
- `models/alpamayo_r1.py`: wires the above into
  `sample_trajectories_from_data_with_vlm_rollout` (the closed-loop inference
  rollout), and rebuilds its `attention_mask` as a standard 2D boolean
  key-padding mask instead of a dense 4D additive-bias tensor —
  `flash_attention_2`'s padding fast path (`_get_unpad_data`/`_upad_input`)
  hard-requires the 2D form and silently computes out-of-bounds indices when
  given the 4D one (the masked key range was already identical across every
  query row, so this is a lossless, more portable representation, not a
  behavior change).

## Status

**Validated (real GPU, real Qwen/Qwen3.5-{0.8B,2B} checkpoints, real PAI
camera data):**
- `scripts/build_base_checkpoint.py` for all four model configs.
- `TrainableReasoningVLA.forward()` (Stage 1) and `TrainableAlpamayoR1.forward()`
  (Stage 2) — the training path — for both the 0.8B and 2B tiers.
- `profile_qwen3_5_inference.py` — the closed-loop inference rollout — for
  both tiers, single-sample and batched (`num_traj_samples>1`).

## Performance vs. Cosmos-Reason2-2B

Stage-2, untrained checkpoints, same H100, same real PAI sample,
`num_traj_samples=1`, `flash-linear-attention`/`causal-conv1d` fast-path
kernels installed (see "Fast-path kernels" below), `torch.inference_mode()`
(both profiling scripts use this — see "torch.inference_mode()" below):

| | Cosmos-Reason2-2B | Qwen3.5-0.8B | Qwen3.5-2B |
|---|---|---|---|
| Params | 2.580B | 1.024B | 2.680B |
| Weights VRAM | 5.56 GiB | 2.13 GiB | 5.61 GiB |
| Peak VRAM (allocated) | 6.19 GiB | 2.57 GiB | 6.17 GiB |
| **Latency (best)** | **1628 ms** | **2307 ms** (+42%) | **2381 ms** (+46%) |
| Planning rate | 0.61 Hz | 0.43 Hz | 0.42 Hz |
| VLM prefill (16-img encode) | 84.8 ms | 36.4 ms | 64.6 ms |
| Per generated token | 11.48 ms | 17.21 ms | 17.00 ms |
| Expert, per denoising step | 4.1 ms † | 10.4 ms | 11.7 ms |
| Projected, trained model (1-tok rollout) | 144.6 ms (6.92 Hz) | 166.5 ms (6.01 Hz) | 207.3 ms (4.82 Hz) |

† Measured with the Cosmos-2B expert as it was then configured: 7 layers × hidden
2048. That expert read only 7 of the VLM's 28 KV-cache layers (HF indexes the
cache by `layer_idx`, so expert layer *i* reads cache layer *i*), and
`alpamayo1_5_sft` has since been changed to a full-depth 28 × 1024 expert, which
costs **~13.6 ms/step** — ~2.9× more for the same parameter count, because depth,
not FLOPs, dominates at batch-1. See
[`alpamayo1_5_distill/README.md`](../alpamayo1_5_distill/README.md). The two Qwen
3.5 columns are unaffected: that recipe still uses its own 8-layer expert and has
the same latent depth-vs-coverage tradeoff unresolved.

**VRAM tracks params (0.8B is meaningfully lighter; 2B is a wash vs. Cosmos);
latency doesn't.** Both Qwen 3.5 tiers are slower than Cosmos-Reason2-2B by
roughly the same ~42-46% — and, more strikingly, **0.8B is barely faster than
2B** (2307ms vs. 2381ms, ~3% apart) despite having <40% of the parameters.
At batch=1, single-sequence autoregressive decode (95% of this workload) is
memory-bandwidth-bound, not FLOPs-bound — reading a smaller weight matrix
per step doesn't buy much when the bottleneck is per-step overhead and
memory traffic, not compute. Don't expect the 0.8B tier to be proportionally
cheaper to *serve* at batch=1 just because it's proportionally cheaper to
*hold in VRAM*. The gap over Cosmos isn't in vision encoding either — Qwen
3.5's native-multimodal encoder prefills faster at both sizes. It's
concentrated in per-token decode and the action expert's diffusion loop.

### Effect of training (Cosmos-Reason2-2B, real Stage-1 checkpoint)

Everything above uses an **untrained** VLM, which never learns to predict
`<traj_future_start>` and so always runs to the `max_new_tokens` ceiling
(128 tokens) — the README already flagged this as unrealistic and gave a
linear (prefill + tokens × per-token-cost) *projection* for a trained model.
A real Stage-1 checkpoint (`alpamayo1_5_sft`'s
`output_stage1_cosmos2b_lcdrive/checkpoint-1500`, LCDrive-trained, Cosmos-
Reason2-2B only — no trained Qwen 3.5 checkpoint exists yet) lets that
projection be checked against reality instead of just trusted:

| | Untrained | Trained (checkpoint-1500) |
|---|---|---|
| Generated tokens (mean) | 128.0 (ceiling, every run) | 8.3–21.7 across samples (see below) |
| Latency | 1628 ms (low run-to-run variance) | 155–775 ms (**high** run-to-run variance) |
| Weights / peak VRAM | 5.56 / 6.19 GiB | 5.56 / 6.19 GiB (unchanged, expected) |

The token count *and* the latency swing by a wide margin between the three
sample runs above (n=1, n=10, n=20 timed iterations gave mean generated
tokens of 8.3, 21.7, and 16.6 respectively) — because
`sample_trajectories_from_data_with_vlm_rollout` samples with
`do_sample=True`, and training didn't collapse rollout length to a fixed
short value, it made *when to stop* a genuine, sample-dependent random
variable. So "best of a few runs" is cherry-picking, not a representative
number. Honest statistics from **n=20** timed iterations:

| | |
|---|---|
| Mean | 330.2 ms (3.03 Hz) |
| Median | 310.6 ms (3.22 Hz) |
| Std. dev | 139.3 ms (~42% of the mean) |
| Min / Max | 155.1 ms / 774.8 ms |
| p25 / p75 | 234.8 ms / 381.3 ms |

**Training gets you roughly a 5x speedup on the median (1628ms → 311ms), not
the ~10x an optimistic best-of-3 would suggest, and it trades a
low-variance-but-slow untrained model for a fast-on-average-but-highly-
variable trained one.** That variance is itself an operationally relevant
finding, not noise to average away: a real-time closed-loop planner cares
about worst-case latency, and this one's worst observed case (775ms) is
~2.6x its median. If bounded latency matters more than average latency for
deployment, that's a reason to consider constraining the stopping
decision (e.g. `do_sample=False` for it specifically, or a hard cap tighter
than 128 tokens) rather than assuming training alone fixes tail latency.

Reproduce with (any dotted `key=value` is forwarded as a Hydra override):
```bash
CUDA_VISIBLE_DEVICES=0 python profile_2b_inference.py \
  config=sft_stage2_cosmos2b num_traj_samples=1 n_warmup=2 n_timed=20 instrument=1 \
  model.stage1_vlm_checkpoint_path=/path/to/output_stage1_cosmos2b_lcdrive/checkpoint-1500
```

### Does downsizing actually pay off? (real 10B vs. real 2B, both trained)

The comparisons above are all *architecture-only* (untrained checkpoints) —
useful for isolating backbone efficiency, but not for the practical question
of whether a 2B backbone is actually worth deploying. Since a real trained
checkpoint exists for Cosmos-Reason2-2B, the released, fully-trained
**Alpamayo-1.5-10B** (`nvidia/Alpamayo-1.5-10B`, 11.079B params, loaded via
`config=sft_stage2_nav`, its native config/checkpoint format — the
`alpamayo1_5.*` release-package config namespace, not `alpamayo_r1.*`, so
point `model.pretrained_model_name_or_path` at a converted "A1-format"
snapshot rather than the raw release download) makes a like-for-like
comparison possible — same n=20 methodology, same real PAI sample:

| | Alpamayo-1.5-10B (real) | Cosmos-Reason2-2B (real, checkpoint-1500) |
|---|---|---|
| Params | 11.079B | 2.580B |
| Weights / peak VRAM | 20.65 / 21.56 GiB | 5.56 / 6.19 GiB |
| Mean latency | 1365.4 ms (0.73 Hz) | 330.2 ms (3.03 Hz) |
| **Median latency** | **1046.8 ms (0.96 Hz)** | **310.6 ms (3.22 Hz)** |
| Std. dev | 780.5 ms (~57% of mean) | 139.3 ms (~42% of mean) |
| Min / Max | 455.0 / 2374.1 ms | 155.1 / 774.8 ms |
| Generated tokens (mean) | 63.5 | 8.3–21.7 across samples |
| Diffusion loop, per step | 19.6 ms | 4.1 ms |

**Downsizing to 2B is a real, roughly 3.4x median-latency win once both
models are actually trained** (1047ms → 311ms) — not just an artifact of
comparing a trained model against an untrained one. VRAM drops by a similar
~3.5x. The 10B also generates a longer rollout on average before stopping
(63.5 vs. single digits to ~20 tokens) and shows *even higher* relative
variance (~57% of its mean vs. ~42%) — its run-to-run spread is wide enough
that the 25th/75th percentiles (631ms / 2289ms) span most of the full
min/max range, suggesting a bimodal-ish "stops early" vs. "rambles" split
rather than a smooth distribution. The same tail-latency caveat from the
Cosmos-Reason2-2B section applies here, more so: a 2.3x median-to-max ratio
is a bigger deal for a bigger, already-slower model.

**Methodological note:** this table reports the directly-measured overall
latency and phase split (VLM-generate vs. diffusion-loop time, both
instrumented on the real timed calls) rather than the script's
prefill/per-token/"projected trained-model latency" decomposition. That
decomposition calibrates by forcing generation caps of 1 and 64 tokens and
assumes the 64-cap run actually produces 64 tokens — true for an untrained
model (which never learns to stop), not guaranteed for an already-trained
one that may stop early even under a synthetic cap. Since a real trained
checkpoint's actual latency is being measured directly here, that
projection is unnecessary for this comparison anyway.

Reproduce (needs the A1-format-converted checkpoint, not the raw
`nvidia/Alpamayo-1.5-10B` download, to match `alpamayo_r1`'s config
namespace):
```bash
CUDA_VISIBLE_DEVICES=0 python profile_2b_inference.py \
  config=sft_stage2_nav num_traj_samples=1 n_warmup=1 n_timed=20 instrument=1 \
  model.pretrained_model_name_or_path=/path/to/models--nvidia--Alpamayo-1.5-10B-A1-format \
  data.val_dataset.local_dir=/path/to/physical_ai_av/ \
  data.val_dataset.annotations_path=/path/to/nav_demo_samples.json \
  data.val_dataset.chunk_ids=[2368]
```

### Why the expert's denoising step is slower

`torch.profiler` on a single `expert.forward()` call (8 layers: 6
Gated-DeltaNet + 2 full-attention, matching Qwen 3.5's native 3:1 ratio):

| | Calls | CPU time/call | CUDA time/call | CPU:GPU ratio |
|---|---|---|---|---|
| `ChunkGatedDeltaRuleFunction` (6 DeltaNet layers) | 6 | 798 µs | 46.5 µs | **~17x** |
| `FlashAttnFunc` (2 attention layers) | 2 | 130 µs | 22.3 µs | ~6x |

Whole-call totals: **12.25ms CPU vs. 1.77ms actual CUDA** — overwhelmingly
dispatch-bound, not compute-bound. The GPU kernel itself isn't slow (46.5µs
for conv+gating+delta-rule-update+norm vs. attention's 22.3µs for a single
matmul-softmax-matmul is a reasonable ~2x for doing more work); the cost is
`flash-linear-attention`'s (`fla`) Python-side dispatch overhead around that
kernel, paid 6x per step (one per DeltaNet layer) in a workload (64 tokens,
10 short steps, batch 1) that's exactly the regime where fixed per-call
overhead dominates instead of amortizing.

**Tested, didn't fix it:** `torch.compile(mode="reduce-overhead")` (CUDA
graphs) produced 21 graph breaks across 22 sub-graphs and then failed
outright (`CUDAGraphs ... overwritten by a subsequent run`). Root cause via
`torch._dynamo.explain()`: `fla`'s `chunk_gated_delta_rule` is explicitly
decorated `@torch.compiler.disable` by its own maintainers, and its fused
RMSNormGated backward calls `cuda_utils.get_device_properties` (an untraceable
C-extension builtin) — both break the graph right at the boundary of the
expensive op, so CUDA-graph capture can't span across it. This is a `fla`
library-maturity gap (deliberately opted out, not just unsupported), not a
config flag away from fixed.

### `torch.inference_mode()`

Applied: both `profile_qwen3_5_inference.py` and (the sibling
`alpamayo1_5_sft` recipe's) `profile_2b_inference.py` now use
`torch.inference_mode()` instead of `torch.no_grad()` for the timed calls.
Zero downside for a pure-inference path, and its benefit propagates through
nested `@torch.no_grad()`-decorated calls it wraps (e.g. the upstream
`FlowMatching.sample()`) without needing to touch those decorators —
`inference_mode`'s view-tracking/version-counter skip is a separate, broader
guard than `no_grad`'s, and stays active for the whole enclosed call whether
or not something inside redundantly re-enters `no_grad()`.

In isolation (a single `expert.forward()` call, no VLM involved), this cut
~9% of total CPU dispatch time (12.25ms → 11.18ms) with CUDA time unchanged,
as expected — it's a fixed tax reduction on every op's dispatch, not
specific to why `fla`'s wrapper in particular is heavy. At the full-pipeline
level (the table above already reflects `inference_mode` throughout), the
before/after:

| | `no_grad()` | `inference_mode()` | Δ |
|---|---|---|---|
| Cosmos-Reason2-2B (best) | 1705 ms | 1628 ms | -4.5% |
| Qwen3.5-2B (best) | 2547 ms | 2381 ms | -6.5% |

Consistent with the isolated-call measurement, and it doesn't move the core
`fla` gap (`chunk_gated_delta_rule` is still ~731µs CPU for ~46.5µs GPU,
~15.7x, barely down from ~17x) — but it's a real, free win worth having
regardless, so it's applied for real rather than just documented.

**Practical mitigations that don't require touching `fla`:** the fixed
per-call overhead is paid once per `expert.forward()` regardless of batch
size, and `sample_trajectories_from_data_with_vlm_rollout` already batches
`num_traj_samples` into a single call — so a larger `num_traj_samples`, or
fewer `num_inference_steps` (default 10) if trajectory quality tolerates it,
amortizes/reduces how many times that tax gets paid.

### Fast-path kernels

`flash-linear-attention`/`causal-conv1d` are real dependencies here (not
optional) — without them, Qwen 3.5's own code silently falls back to a
pure-PyTorch path for every Gated-DeltaNet layer, which is substantially
slower still (confirmed while benchmarking: ~2x on the expert's
per-denoising-step cost, ~4x on VLM prefill).

## Installation

```bash
cd recipes/alpamayo1_5_sft_qwen3_5
uv venv .venv
uv sync
source .venv/bin/activate
```

## Training

Same Hydra entry points as `alpamayo1_5_sft`, pointed at the new configs:

```bash
# Stage 1 (VLM SFT), 0.8B or 2B:
python train_hf.py --config-name sft_stage1_qwen3_5_0_8b
python train_hf.py --config-name sft_stage1_qwen3_5_2b

# Stage 2 (action expert), once Stage 1 has produced a checkpoint:
python train_hf.py --config-name sft_stage2_qwen3_5_0_8b \
  model.stage1_vlm_checkpoint_path=training/output_stage1_qwen3_5_0_8b/checkpoint-N
```

Both default to the 20-sample nav demo annotations (same ones
`alpamayo1_5_sft` uses) for a smoke test; swap `data.train_dataset.chunk_ids`/
`annotations_path` for real training data.
