# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Profile the KAVA latent-reasoning student's inference latency on real LCDrive data.

Profiles the **deployable** architecture — the Stage-2 2B student (frozen VLM +
28-layer action expert) — with ``K`` latent slots spliced into the prefill exactly as
``KaVaCollator`` does, on a real LCDrive val clip.

The point is to separate the phases KAVA actually changes, rather than time one
end-to-end number that hides them:

======================  ====================================================
``prefill(K)``          one VLM forward over ``[images, history, prompt,
                        <cot_start> Q_1..Q_K <cot_end>, <tfs>]``.  ``K=0`` is
                        today's text-silent 2B; the difference is the entire
                        cost of latent reasoning at ``T=1``.
``jacobi_step(K)``      one slot-only forward against the cached prefix — what
                        each PCCoT iteration beyond the first costs.
``decode_token``        one text token against the cache — the per-token cost a
                        CoT rollout pays and KAVA removes.
``expert_step``         one action-expert forward over the cropped cache — one
                        Euler denoising step.
======================  ====================================================

Totals are then composed from measured parts, so each arm is comparable and no arm
inherits another's overhead:

* **Full CoT** ``prefill(0) + N_cot x decode_token + steps x expert_step``
* **Text-silent (today's 2B)** ``prefill(0) + steps x expert_step``
* **KAVA T** ``prefill(K) + (T-1) x jacobi_step(K) + steps x expert_step``

Note what this does and does not tell you. Latency is a function of shapes, not of
trained values, so untrained weights are fine here — but it is measured on an **H100**,
not on Thor, so treat the numbers as relative (does a slot cost anything? does T=3 fit?)
rather than as the deploy budget. ``reasoning-setup-2b.md`` §0 has the Thor per-token
figure for that.

Usage::

    CUDA_VISIBLE_DEVICES=0 python -m alpamayo1_5_distill.scripts.profile_kava_latency \
        slots=0,8,16,32 n_cot=40 inference_steps=10,2 n_timed=20
"""

import statistics
import sys
import time

import hydra.utils as hyu
import torch

from alpamayo1_5_distill.data.kava_dataset import (
    DEFAULT_SLOT_PLACEHOLDER_ID,
    splice_slot_placeholders,
)
from alpamayo1_5_distill.scripts.cache_common import parse_argv


def _time(fn, n_warmup: int, n_timed: int) -> tuple[float, float]:
    """Return (median, min) wall-clock ms, synchronising around every call."""
    return _time_many({"x": fn}, n_warmup, n_timed)["x"]


def _time_many(fns: dict, n_warmup: int, n_timed: int) -> dict:
    """Time several callables **round-robin**. -> ``{label: (median, min)}``.

    Interleaving matters when the quantity of interest is a *difference* between
    arms. Timed back-to-back, a few ms of clock drift or a co-tenant process on the
    same GPU lands entirely on whichever arm ran during it — enough, measured here, to
    make a strictly-more-work prefill look 5% *faster* than its baseline. Round-robin
    spreads any such disturbance across every arm instead.

    ``min`` is the more meaningful statistic for a marginal cost: it is the run least
    perturbed by whatever else the GPU was doing.
    """
    for _ in range(n_warmup):
        for fn in fns.values():
            fn()
    torch.cuda.synchronize()
    samples: dict = {label: [] for label in fns}
    for _ in range(n_timed):
        for label, fn in fns.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples[label].append((time.perf_counter() - started) * 1000.0)
    return {label: (statistics.median(v), min(v)) for label, v in samples.items()}


def main() -> None:
    argv = parse_argv(sys.argv[1:])
    config_name = argv.get("config", "sft_stage2_cosmos2b_lcdrive")
    slot_counts = [int(v) for v in argv.get("slots", "0,8,16,32").split(",")]
    jacobi_iters = [int(v) for v in argv.get("jacobi", "1,2,3").split(",")]
    inference_steps = [int(v) for v in argv.get("inference_steps", "10,2").split(",")]
    n_cot = int(argv.get("n_cot", 40))  # a realistic CoT length for this teacher
    n_warmup = int(argv.get("n_warmup", 3))
    n_timed = int(argv.get("n_timed", 20))
    num_frames = int(argv["num_frames"]) if "num_frames" in argv else None
    placeholder = int(argv.get("placeholder_id", DEFAULT_SLOT_PLACEHOLDER_ID))

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Fewer frames per camera => fewer vision tokens => cheaper prefill. Patch the
    # loader symbol the datasets import directly, so slicing happens before the VLA
    # preprocessor tokenizes (same approach as alpamayo1_5_sft/profile_2b_inference.py).
    if num_frames is not None:
        import alpamayo.data.pai as pai_mod

        original_loader = pai_mod.load_physical_aiavdataset

        def sliced_loader(*a, **kw):
            sample = original_loader(*a, **kw)
            if sample["image_frames"].shape[1] > num_frames:
                sample["image_frames"] = sample["image_frames"][:, -num_frames:]
                for key in ("relative_timestamps", "absolute_timestamps"):
                    if sample.get(key) is not None and sample[key].shape[-1] >= num_frames:
                        sample[key] = sample[key][..., -num_frames:]
            return sample

        pai_mod.load_physical_aiavdataset = sliced_loader

    from hydra import compose, initialize_config_dir

    cfg_dir = "/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/configs"
    overrides = [
        "data.val_dataset.vla_preprocess_args.generation_mode=true",
        "model.stage1_vlm_checkpoint_path=null",  # latency is shape-bound, not value-bound
    ]
    overrides += [f"{k}={v}" for k, v in argv.items() if "." in k]
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    print(f"[profile] building {config_name} (VLM + action expert) ...", flush=True)
    model = hyu.instantiate(cfg.model, _convert_="partial").to(device).eval()
    model.requires_grad_(False)
    text_model = model.vlm.model.language_model
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[profile] {torch.cuda.get_device_name(0)} | params {n_params / 1e9:.3f} B | "
        f"VLM layers {len(text_model.layers)} | expert layers {len(model.expert.layers)} | "
        f"expert hidden {model.expert.config.hidden_size}",
        flush=True,
    )

    dataset = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=model.config)
    collate_fn = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    batch = collate_fn([dataset[0]])
    tokenized = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch["tokenized_data"].items()
    }
    base_ids = model.fuse_traj_tokens(
        tokenized.pop("input_ids"),
        {
            "ego_history_xyz": batch["ego_history_xyz"].to(device),
            "ego_history_rot": batch["ego_history_rot"].to(device),
        },
    )
    tokenized.pop("attention_mask", None)
    tfs_id = int(model.special_token_ids["traj_future_start"])
    cot_start = int(model.special_token_ids["cot_start"])
    cot_end = int(model.special_token_ids["cot_end"])
    n_vision = int((base_ids == model.vlm.config.image_token_id).sum())
    print(
        f"[profile] LCDrive val clip {dataset.clip_ids[0]} | prefill {base_ids.shape[1]} tokens "
        f"({n_vision} vision, {num_frames or 4} frames/cam)",
        flush=True,
    )

    hidden = model.vlm.config.text_config.hidden_size
    n_action = model.action_space.get_action_space_dims()[0]

    def prefill_inputs(n_slots: int):
        """input_ids + slot columns for a prefill carrying ``n_slots`` latent slots."""
        if n_slots == 0:
            return base_ids, None
        spliced = splice_slot_placeholders(
            base_ids,
            tfs_id=tfs_id,
            cot_start_id=cot_start,
            cot_end_id=cot_end,
            placeholder_id=placeholder,
            num_slots=n_slots,
        )
        return spliced["input_ids"], spliced["slot_pos"]

    def run_prefill(ids, slot_pos, slots):
        """One VLM prefill, injecting slot embeddings exactly as the model does."""
        handle = None
        if slot_pos is not None:
            rows = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1)

            def inject(_m, _a, out):
                out = out.clone()
                out[rows, slot_pos] = slots.to(out.dtype)
                return out

            handle = model.vlm.get_input_embeddings().register_forward_hook(inject)
        try:
            return model.vlm(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
                **tokenized,
            )
        finally:
            if handle is not None:
                handle.remove()

    results: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------- vision encoder
    # 2880 of ~2993 prefill tokens are vision, so the ViT is the thing `num_frames`
    # actually trades against. Timed via a hook so it can be reported separately from
    # the language stack rather than inferred.
    vision_ms = {"total": 0.0, "calls": 0}
    visual = model.vlm.model.visual
    original_visual_forward = visual.forward

    def timed_visual(*a, **kw):
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = original_visual_forward(*a, **kw)
        torch.cuda.synchronize()
        vision_ms["total"] += (time.perf_counter() - started) * 1000.0
        vision_ms["calls"] += 1
        return out

    visual.forward = timed_visual

    # ---------------------------------------------------------------- prefill(K)
    print("\n[profile] prefill vs number of latent slots (round-robin) ...", flush=True)
    seq_len: dict[int, int] = {}
    prefill_fns = {}
    for n_slots in slot_counts:
        ids, slot_pos = prefill_inputs(n_slots)
        slots = torch.randn(max(n_slots, 1), hidden, device=device, dtype=dtype) * 0.02
        seq_len[n_slots] = ids.shape[1]
        prefill_fns[n_slots] = (
            lambda ids=ids, slot_pos=slot_pos, slots=slots: run_prefill(ids, slot_pos, slots)
        )
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        prefill_ms = _time_many(prefill_fns, n_warmup, n_timed)
    per_prefill_vision = vision_ms["total"] / max(vision_ms["calls"], 1)
    for n_slots in slot_counts:
        print(
            f"  K={n_slots:<3d} seq={seq_len[n_slots]:>5d}  median {prefill_ms[n_slots][0]:7.2f} ms  "
            f"min {prefill_ms[n_slots][1]:7.2f} ms",
            flush=True,
        )
    print(
        f"  vision encoder (ViT, inside every prefill above): {per_prefill_vision:.2f} ms",
        flush=True,
    )
    visual.forward = original_visual_forward

    # ------------------------------------------------- jacobi step / decode token
    # Both run against a cached prefix, so build the cache once per K and measure the
    # marginal forward. The cache is cropped back after every call, exactly as the
    # inference loop does between Euler steps.
    print("\n[profile] marginal passes against the cached prefix ...", flush=True)
    jacobi_ms: dict[int, tuple[float, float]] = {}
    decode_ms: tuple[float, float] | None = None
    expert_ms: tuple[float, float] | None = None

    for n_slots in slot_counts:
        ids, slot_pos = prefill_inputs(n_slots)
        slots = torch.randn(max(n_slots, 1), hidden, device=device, dtype=dtype) * 0.02
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            out = run_prefill(ids, slot_pos, slots)
            cache = out.past_key_values
            crop_len = cache.get_seq_length()
            position_ids, _ = model.vlm.model.get_rope_index(
                ids, tokenized.get("image_grid_thw"), None, torch.ones_like(ids)
            )

            if n_slots > 0:
                # A PCCoT iteration: re-run ONLY the slot columns against the prefix.
                start = int(slot_pos[0, 0])
                slot_positions = position_ids[:, :, start : start + n_slots]
                cache_position = torch.arange(start, start + n_slots, device=device)
                embeds = torch.randn(1, n_slots, hidden, device=device, dtype=dtype) * 0.02
                mask = torch.ones((1, start + n_slots), dtype=torch.long, device=device)

                def jacobi_step():
                    text_model(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        position_ids=slot_positions,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    cache.crop(crop_len)

                jacobi_ms[n_slots] = _time(jacobi_step, n_warmup, n_timed)
                print(
                    f"  K={n_slots:<3d} jacobi step   median {jacobi_ms[n_slots][0]:7.2f} ms  "
                    f"min {jacobi_ms[n_slots][1]:7.2f} ms",
                    flush=True,
                )

            if n_slots == slot_counts[0]:
                # One text token against the cache — the cost a CoT rollout repeats.
                one = torch.full((1, 1), placeholder, device=device, dtype=torch.long)
                one_pos = position_ids[:, :, -1:] + 1
                one_cache_pos = torch.tensor([crop_len], device=device)
                one_mask = torch.ones((1, crop_len + 1), dtype=torch.long, device=device)

                def decode_step():
                    text_model(
                        inputs_embeds=model.vlm.get_input_embeddings()(one),
                        attention_mask=one_mask,
                        position_ids=one_pos,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=one_cache_pos,
                    )
                    cache.crop(crop_len)

                decode_ms = _time(decode_step, n_warmup, n_timed)
                print(
                    f"  decode 1 text token   median {decode_ms[0]:7.2f} ms  "
                    f"min {decode_ms[1]:7.2f} ms",
                    flush=True,
                )

                # One Euler step of the action expert over the cropped cache.
                noisy = torch.randn(1, n_action, 2, device=device, dtype=dtype)
                timesteps = torch.full((1, 1, 1), 0.5, device=device, dtype=dtype)
                action_embeds = model.action_in_proj(noisy, timesteps)
                expert_pos = model._process_position_ids_qwen2_5_vl(
                    out, 1, action_embeds.shape[1], device
                )
                expert_kwargs = (
                    {"is_causal": False} if model.config.expert_non_causal_attention else {}
                )

                def expert_step():
                    model.expert(
                        inputs_embeds=action_embeds,
                        position_ids=expert_pos,
                        past_key_values=cache,
                        attention_mask=None,
                        use_cache=True,
                        **expert_kwargs,
                    )
                    cache.crop(crop_len)

                expert_ms = _time(expert_step, n_warmup, n_timed)
                print(
                    f"  expert 1 Euler step   median {expert_ms[0]:7.2f} ms  min {expert_ms[1]:7.2f} ms  "
                    f"({n_action} action tokens over a {crop_len}-token cache)",
                    flush=True,
                )
        del out, cache
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------- totals
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print("\n" + "=" * 78)
    print("KAVA LATENCY — composed from measured parts (min ms, least-perturbed run)")
    print("=" * 78)
    print(f"  GPU {torch.cuda.get_device_name(0)} | dtype {dtype} | batch 1 | peak {peak:.1f} GiB")
    print(f"  prefill {seq_len[slot_counts[0]]} tokens ({n_vision} vision, ViT {per_prefill_vision:.1f} ms)")
    print(f"  decode/token {decode_ms[1]:.2f} | expert/step {expert_ms[1]:.2f}")
    print("-" * 78)
    header = f"  {'arm':<34}" + "".join(f"{f'{s} steps':>13}" for s in inference_steps)
    print(header)
    print("-" * 78)

    def emit(label: str, vlm_ms: float) -> None:
        cells = ""
        for steps in inference_steps:
            total = vlm_ms + steps * expert_ms[1]
            cells += f"{total:>8.1f} ({1000 / total:>3.1f}Hz)"
        print(f"  {label:<34}{cells}")
        results[label] = {"vlm_ms": vlm_ms}

    base = prefill_ms[slot_counts[0]][1]
    emit(f"Full CoT ({n_cot} tok, no KAVA)", base + n_cot * decode_ms[1])
    emit("text-silent 2B (today, K=0)", base)
    for n_slots in slot_counts:
        if n_slots == 0:
            continue
        for t in jacobi_iters:
            extra = (t - 1) * jacobi_ms[n_slots][1]
            emit(f"KAVA K={n_slots} T={t}", prefill_ms[n_slots][1] + extra)
    print("=" * 78)

    # What the slots actually cost, which is the claim under test.
    print("\nSlot overhead vs the text-silent baseline (T=1, prefill only):")
    for n_slots in slot_counts:
        if n_slots == 0:
            continue
        delta = prefill_ms[n_slots][1] - base
        print(
            f"  K={n_slots:<3d} +{delta:6.2f} ms  ({delta / base * 100:+5.1f}% of prefill)  "
            f"seq +{seq_len[n_slots] - seq_len[slot_counts[0]]} tokens"
        )
    saving = n_cot * decode_ms[1]
    print(
        f"\nCoT tokens removed: {n_cot} x {decode_ms[1]:.2f} ms = {saving:.1f} ms of rollout "
        f"that latent reasoning does not pay."
    )


if __name__ == "__main__":
    main()
