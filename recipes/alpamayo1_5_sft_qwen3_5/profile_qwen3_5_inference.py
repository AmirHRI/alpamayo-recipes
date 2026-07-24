# Profile the 2B Alpamayo-1.5 model on a single PAI data point for CLOSED-LOOP
# planning: batch=1, ONE trajectory rollout. Measures peak VRAM and end-to-end
# latency, and prints the decoded future trajectory shape.
#
# Two models can be profiled:
#   * Stage 1 (config=sft_stage1_cosmos2b_lcdrive): base VLM only
#     (TrainableReasoningVLA, 2.13B). Trajectory = 128 autoregressive traj tokens.
#   * Stage 2 (config=sft_stage2_cosmos2b): FULL model with action expert
#     (TrainableAlpamayoR1, 2.58B = frozen VLM + 7-layer expert + diffusion head).
#     Trajectory = short VLM rollout to <traj_future_start> + diffusion denoising.
#     THIS is the deployable closed-loop planner.
#
# Runs on a single GPU (set CUDA_VISIBLE_DEVICES). Uses base 2B weights (no
# fine-tuned checkpoint needed) which is sufficient for VRAM/latency profiling.
#
# Usage:
#   # Stage 2 full planner, closed-loop (batch=1, single rollout):
#   CUDA_VISIBLE_DEVICES=0 a1_5_sft/bin/python profile_2b_inference.py \
#       config=sft_stage2_cosmos2b num_traj_samples=1 n_warmup=2 n_timed=5
#
#   # Stage 1 base VLM:
#   CUDA_VISIBLE_DEVICES=0 a1_5_sft/bin/python profile_2b_inference.py \
#       config=sft_stage1_cosmos2b_lcdrive num_traj_samples=1

import os
import sys
import time

import hydra
import hydra.utils as hyu
import torch
from hydra import compose, initialize_config_dir

from alpamayo1_5_sft_qwen3_5.hydra_compat import opaque_model_config


def gib(x_bytes: float) -> float:
    return x_bytes / (1024**3)


def main() -> None:
    # ---- simple kwargs from argv (key=value) ----
    argv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    config_name = argv.get("config", "sft_stage1_qwen3_5_0_8b")
    num_traj_samples = int(argv.get("num_traj_samples", 1))
    n_warmup = int(argv.get("n_warmup", 2))
    n_timed = int(argv.get("n_timed", 5))
    is_stage2 = "expert" in config_name or "stage2" in config_name

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    # Stage 1 (LCDrive) selects clips by UUID filter → override chunk + drop the
    # filter so a single clip loads. Stage 2 uses annotation-driven nav data whose
    # val chunk is already set in the config, so we leave its dataset untouched.
    overrides = ["data.val_dataset.vla_preprocess_args.generation_mode=true"]
    if not is_stage2:
        chunk = argv.get("chunk", "1519")
        overrides += [
            f"data.val_dataset.chunk_ids=[{chunk}]",
            "data.val_dataset.clip_uuid_filter=null",
        ]

    cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    label = "Stage 2 (full model + action expert)" if is_stage2 else "Stage 1 (base VLM only)"
    print(f"[profile] config={config_name}  ->  {label}", flush=True)

    # ---- build model ----
    print("[profile] instantiating model ...", flush=True)
    t0 = time.time()
    model = hyu.instantiate(cfg.model, _convert_="partial")
    model = model.to(device).eval()
    torch.cuda.synchronize()
    build_s = time.time() - t0
    mem_after_load = torch.cuda.memory_allocated()
    print(f"[profile] model built in {build_s:.1f}s", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[profile] params: {n_params/1e9:.3f} B", flush=True)
    print(f"[profile] weights VRAM (allocated): {gib(mem_after_load):.2f} GiB", flush=True)

    # ---- build one data sample ----
    print("[profile] loading one data sample ...", flush=True)

    # Optional: reduce #frames per camera fed to the model WITHOUT touching the
    # dataset code. We monkey-patch the loader symbol used inside the dataset so
    # each sample's image_frames (and matching timestamps) are sliced to the last
    # `num_frames` frames *before* the VLA preprocessor tokenizes them. Fewer
    # frames => fewer image tokens (~180/frame) => cheaper prefill.
    num_frames = argv.get("num_frames", None)
    if num_frames is not None:
        num_frames = int(num_frames)
        import alpamayo.data.pai as _pai_mod
        import alpamayo.data.pai_nav as _pai_nav_mod

        _orig_loader = _pai_mod.load_physical_aiavdataset

        def _sliced_loader(*a, **kw):
            s = _orig_loader(*a, **kw)
            # image_frames: (N_cam, T, 3, H, W); keep the most recent `num_frames`
            if "image_frames" in s and s["image_frames"].shape[1] > num_frames:
                s["image_frames"] = s["image_frames"][:, -num_frames:]
                for tk in ("relative_timestamps", "absolute_timestamps"):
                    if tk in s and s[tk] is not None and s[tk].shape[-1] >= num_frames:
                        s[tk] = s[tk][..., -num_frames:]
            return s

        # patch both modules (PAIDataset and PAIDatasetWithNav import it directly)
        _pai_mod.load_physical_aiavdataset = _sliced_loader
        _pai_nav_mod.load_physical_aiavdataset = _sliced_loader
        print(f"[profile] frames per camera reduced to last {num_frames}", flush=True)

    model_config = opaque_model_config(model.config)
    val_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model_config
    )
    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model_config
    )
    sample = val_dataset[0]
    batch = collate_fn([sample])
    # move tensors to GPU
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = {
                kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()
            }

    dtype = torch.bfloat16

    # Stage 1 ONLY: force full-length generation so latency reflects a *trained*
    # model's real workload. The base weights are untrained and would emit EOS
    # immediately (unrealistically low latency); pinning min_new_tokens == the
    # future-trajectory length makes every run generate the full token budget.
    # Stage 2 gets its trajectory from the diffusion expert (not VLM tokens): the
    # VLM only needs to roll out to <traj_future_start>, so we do NOT force length.
    full_len = model.config.tokens_per_future_traj
    if not is_stage2:
        model.vlm.generation_config.min_new_tokens = full_len

    # ---- optional phase instrumentation (Stage 2) -------------------------
    # Monkey-patch vlm.generate and expert.forward to bucket latency into:
    #   (a) VLM generate  = image prefill + autoregressive rollout to <traj_future_start>
    #   (b) diffusion loop = sum of expert.forward calls (one per denoising step)
    # We also record #generated tokens (to expose untrained-ramble inflation) and
    # #expert calls (= #denoising steps).
    instrument = argv.get("instrument", "0") == "1"
    stats = {"vlm_generate_ms": 0.0, "expert_ms": 0.0, "expert_calls": 0, "gen_tokens": 0}

    if instrument:
        _orig_generate = model.vlm.generate
        _orig_expert_fwd = model.expert.forward

        def timed_generate(*a, **kw):
            torch.cuda.synchronize()
            t = time.time()
            out = _orig_generate(*a, **kw)
            torch.cuda.synchronize()
            stats["vlm_generate_ms"] += (time.time() - t) * 1000
            try:
                seq = out.sequences if hasattr(out, "sequences") else out
                in_ids = kw.get("input_ids")
                if in_ids is not None:
                    stats["gen_tokens"] += int(seq.shape[1] - in_ids.shape[1])
            except Exception:
                pass
            return out

        def timed_expert_fwd(*a, **kw):
            torch.cuda.synchronize()
            t = time.time()
            out = _orig_expert_fwd(*a, **kw)
            torch.cuda.synchronize()
            stats["expert_ms"] += (time.time() - t) * 1000
            stats["expert_calls"] += 1
            return out

        model.vlm.generate = timed_generate
        model.expert.forward = timed_expert_fwd

    def run_once(max_gen=None):
        # sample_trajectories_from_data pops input_ids out of tokenized_data in
        # place, so hand it a fresh shallow copy each call.
        call_batch = dict(batch)
        if isinstance(batch.get("tokenized_data"), dict):
            call_batch["tokenized_data"] = dict(batch["tokenized_data"])
        kw = dict(
            num_traj_samples=num_traj_samples,
            num_traj_sets=1,
            top_p=0.98,
            temperature=0.6,
            return_extra=False,
        )
        kw["max_generation_length"] = max_gen if max_gen is not None else full_len
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            out = model.sample_trajectories_from_data(data=call_batch, **kw)
        return out

    # ---- warmup ----
    print(f"[profile] warmup x{n_warmup} ...", flush=True)
    for _ in range(n_warmup):
        out = run_once()
    torch.cuda.synchronize()

    # ---- timed runs ----
    torch.cuda.reset_peak_memory_stats()
    for k in stats:
        stats[k] = 0.0 if k.endswith("_ms") else 0
    lat = []
    for i in range(n_timed):
        torch.cuda.synchronize()
        t = time.time()
        out = run_once()
        torch.cuda.synchronize()
        dt = time.time() - t
        lat.append(dt)
        print(f"[profile]   run {i+1}/{n_timed}: {dt*1000:.1f} ms", flush=True)

    pred_xyz, pred_rot = out[0], out[1]

    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    mean_ms = sum(lat) / len(lat) * 1000
    best_ms = min(lat) * 1000

    print("\n==================== PROFILE RESULTS ====================")
    print(f"model                  : {label}")
    print(f"config                 : {config_name}")
    print(f"GPU                    : {torch.cuda.get_device_name(0)}")
    print(f"dtype                  : {dtype}")
    print(f"params                 : {n_params/1e9:.3f} B")
    print(f"batch size             : 1  (closed-loop: one clip)")
    print(f"num_traj_samples       : {num_traj_samples}")
    print(f"future traj tokens     : {model.config.tokens_per_future_traj}")
    print(f"pred_xyz shape         : {tuple(pred_xyz.shape)}")
    print(f"pred_rot shape         : {tuple(pred_rot.shape)}")
    print(f"weights VRAM           : {gib(mem_after_load):.2f} GiB")
    print(f"peak VRAM (allocated)  : {gib(peak):.2f} GiB")
    print(f"peak VRAM (reserved)   : {gib(reserved):.2f} GiB")
    print(f"latency  mean          : {mean_ms:.1f} ms")
    print(f"latency  best          : {best_ms:.1f} ms  <-- closed-loop planning latency")
    print(f"planning rate          : {1000.0/best_ms:.2f} Hz  (1 / best latency)")
    print("========================================================")
    if instrument and n_timed > 0:
        vlm_ms = stats["vlm_generate_ms"] / n_timed
        exp_ms = stats["expert_ms"] / n_timed
        calls = stats["expert_calls"] / n_timed
        gtok = stats["gen_tokens"] / n_timed
        other = mean_ms - vlm_ms - exp_ms
        print("------------------ PHASE BREAKDOWN (avg) ----------------")
        print(f"VLM generate (prefill+rollout) : {vlm_ms:8.1f} ms  ({vlm_ms/mean_ms*100:4.1f}%)")
        print(f"  -> generated tokens          : {gtok:8.1f}  (rollout length to <traj_future_start>)")
        print(f"diffusion loop (expert fwd)    : {exp_ms:8.1f} ms  ({exp_ms/mean_ms*100:4.1f}%)")
        print(f"  -> expert.forward calls      : {calls:8.1f}  (= #denoising steps)")
        if calls:
            print(f"  -> per denoising step        : {exp_ms/calls:8.1f} ms")
        print(f"other (decode/fuse/overhead)   : {other:8.1f} ms  ({other/mean_ms*100:4.1f}%)")
        print("========================================================")
        print("NOTE: untrained VLM may over-generate tokens before hitting")
        print("      <traj_future_start>; a trained no-CoT nav model rolls out")
        print("      only a few tokens, so real VLM-generate time is a floor +")
        print("      (#gen_tokens x per-token cost). Diffusion cost is fixed.")

        # --- decompose VLM generate into prefill vs per-token decode ---------
        # Measure vlm_generate time at two caps; slope = per-token, intercept =
        # prefill (16-image encode + first forward). Then project a TRAINED model
        # that rolls out only a few tokens to <traj_future_start>.
        if is_stage2:
            def measure_vlm(cap, reps=3):
                best = float("inf")
                toks = 0
                for _ in range(reps):
                    stats["vlm_generate_ms"] = 0.0
                    stats["gen_tokens"] = 0
                    run_once(max_gen=cap)
                    best = min(best, stats["vlm_generate_ms"])
                    toks = stats["gen_tokens"]
                return best, toks

            (t_lo, n_lo), (t_hi, n_hi) = measure_vlm(1), measure_vlm(64)
            if n_hi > n_lo:
                per_tok = (t_hi - t_lo) / (n_hi - n_lo)
                prefill = t_lo - per_tok * n_lo  # extrapolate to 0 generated tokens
            else:
                per_tok, prefill = 0.0, t_lo
            fixed = exp_ms + other  # diffusion + overhead (independent of rollout len)
            print("--------------- VLM DECODE DECOMPOSITION ----------------")
            print(f"prefill (16-img encode + 1st fwd) : {prefill:8.1f} ms")
            print(f"per generated token              : {per_tok:8.2f} ms/token")
            print(f"fixed tail (diffusion + overhead) : {fixed:8.1f} ms")
            print("  projected closed-loop latency for a TRAINED model")
            for k in (1, 3, 8):
                est = prefill + per_tok * k + fixed
                print(f"    rollout={k:>2} tok -> {est:8.1f} ms  ({1000.0/est:4.2f} Hz)")
            print("========================================================")
    xyz0 = pred_xyz.reshape(-1, pred_xyz.shape[-2], 3)[0]
    print(f"\nFirst predicted trajectory (xyz), {xyz0.shape[0]} steps @10Hz:")
    print(xyz0[:: max(1, xyz0.shape[0] // 8)].float().cpu().numpy())


if __name__ == "__main__":
    main()
