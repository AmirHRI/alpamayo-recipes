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
