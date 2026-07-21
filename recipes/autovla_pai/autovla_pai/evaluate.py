"""Evaluate AutoVLA on the Physical AI AV (PAI) dataset.

Metrics are computed using the same ``alpamayo`` MetricRunner as
``alpamayo1_sft/evaluate_hf.py``, so numbers are directly comparable:
  - ``ReasoningSampler``  calls ``wrapper.sample_trajectories_from_data``
  - ``DistanceMetrics``   computes minADE / corner_distance at time_step=0.5 s

Horizon note: AutoVLA outputs 8 waypoints x 0.5 s = 4 s total.
``DistanceMetrics`` requests by_t={0.5, 1.0, 3.0, 5.0} s; the 5.0 s step
(step 10) exceeds Tf=8, so it is automatically skipped by ``compute_minade``.

Usage (single-GPU)
------------------
    cd /home/achahe/alpamayo-recipes/recipes/autovla_pai
    python -m autovla_pai.evaluate \
        --config  /data/sungyeonpark/autovla/ckpts/paper/nuplan/config.yaml \
        --ckpt    /data/sungyeonpark/autovla/ckpts/paper/nuplan/final.ckpt \
        --pai_dir /data/datasets/physical_ai_av \
        --chunk_ids 186-205

Multi-GPU (4 GPUs) via torchrun
---------------------------------
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
        -m autovla_pai.evaluate \
        --config  /data/sungyeonpark/autovla/ckpts/paper/nuplan/config.yaml \
        --ckpt    /data/sungyeonpark/autovla/ckpts/paper/nuplan/final.ckpt \
        --pai_dir /data/datasets/physical_ai_av \
        --chunk_ids 186-205

Interpreter: use the shared a1_sft venv from alpamayo1_sft
    (see README for setup instructions).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Add SKIPlan to sys.path when not installed editable
SKIPPLAN_ROOT = Path("/home/achahe/SKIPlan")
for _p in [SKIPPLAN_ROOT, SKIPPLAN_ROOT / "navsim"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ── Distributed helpers ───────────────────────────────────────────────────────

def _init_distributed() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    import torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ.get("RANK",       local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _all_reduce(d_sums: dict, d_counts: dict, n: int, device: torch.device, world_size: int) -> int:
    if world_size <= 1:
        return n
    import torch.distributed as dist
    for k in d_sums:
        t = torch.tensor([d_sums[k], float(d_counts[k])], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        d_sums[k], d_counts[k] = t[0].item(), int(t[1].item())
    n_t = torch.tensor(n, dtype=torch.long, device=device)
    dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
    return int(n_t.item())


# ── Collate ───────────────────────────────────────────────────────────────────

def _collate(batch: list) -> dict | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out: dict = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            try:
                out[key] = torch.stack(vals, dim=0)
            except Exception:
                out[key] = vals
        elif isinstance(vals[0], (int, float)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AutoVLA on PAI with alpamayo metrics")
    p.add_argument("--config",      required=True, help="AutoVLA config.yaml path")
    p.add_argument("--ckpt",        required=True, help="AutoVLA checkpoint (.ckpt) path")
    p.add_argument("--pai_dir",     required=True, help="PAI local_dir root")
    p.add_argument("--chunk_ids",   default="186-205", help="Chunk range, e.g. '186-205'")
    p.add_argument("--batch_size",  type=int, default=4, help="Per-GPU batch size")
    p.add_argument("--max_samples", type=int, default=-1, help="-1 = all samples")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = _init_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load AutoVLA config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data/sungyeonpark/nuplan/maps")

    # Load model
    from models.structured_stage3 import StructuredStage3Module
    from autovla_pai.autovla_wrapper import AutoVLAWrapper

    if is_main:
        print("[AutoVLA] Loading model...")

    model = StructuredStage3Module(config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_main:
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    # Load FM EMA weights if present
    if "fm_ema_shadow" in ckpt and hasattr(model.autovla, "fm_net"):
        fm_sd = model.autovla.fm_net.state_dict()
        for k, v in ckpt["fm_ema_shadow"].items():
            if k in fm_sd:
                fm_sd[k].copy_(v)
        model.autovla.fm_net.load_state_dict(fm_sd)
        if is_main:
            print("  FM EMA weights loaded")

    model.to(device).eval()
    model.autovla.device = str(device)
    wrapper = AutoVLAWrapper(model, config)
    if is_main:
        print(f"  Model on {device}")

    # PAI dataset — AutoVLA resolution: 4 history + 8 future @ 0.5 s
    if is_main:
        print(f"[PAI] chunks={args.chunk_ids}  dir={args.pai_dir}")

    from alpamayo.data.pai import PAIDataset
    dataset = PAIDataset(
        local_dir=args.pai_dir,
        chunk_ids=args.chunk_ids,
        use_default_keyframe=True,
        num_history_steps=4,    # 4 x 0.5 s = 2 s
        num_future_steps=8,     # 8 x 0.5 s = 4 s
        time_step=0.5,
        vla_preprocess_args=None,   # pai_adapter does its own image processing
    )
    if is_main:
        print(f"  Dataset size: {len(dataset)}")

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=False,
    )

    # MetricRunner -- same infrastructure as alpamayo evaluate_hf.py
    from alpamayo.metrics.metric_runner import MetricRunner
    from alpamayo.metrics.metric_api import ReasoningSampler, DistanceMetrics

    metric_runner = MetricRunner([
        ReasoningSampler(
            num_traj_sets=1,
            num_traj_samples=1,   # forward_proposal is deterministic
            top_p=0.98,
            temperature=0.6,
            traj_only_generation=False,
            max_generation_length=256,
        ),
        DistanceMetrics(time_step=0.5),   # by_t=5.0 auto-filtered (Tf=8 < step 10)
    ])

    # Evaluation loop
    metric_sums:   dict = defaultdict(float)
    metric_counts: dict = defaultdict(int)
    n_samples = 0

    if is_main:
        print("[Eval] Running inference...")

    bar = tqdm(loader, desc="Batches", disable=not is_main)
    for pai_batch in bar:
        if pai_batch is None:
            continue
        if args.max_samples > 0 and n_samples >= args.max_samples:
            break

        pai_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in pai_batch.items()
        }

        output_batch: dict = {}
        try:
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
            ):
                metric_runner.run(wrapper, pai_batch, output_batch)
        except Exception as exc:
            import traceback
            if is_main:
                print(f"  batch error: {exc}")
                traceback.print_exc()
            continue

        # ego_history_xyz is always [B, 1, T, 3] — reliable batch-size source
        n_samples += pai_batch["ego_history_xyz"].shape[0]

        for k, v in output_batch.items():
            if not k.startswith("metric/"):
                continue
            if isinstance(v, torch.Tensor):
                metric_sums[k]   += v.float().sum().item()
                metric_counts[k] += v.numel()

    # All-reduce across GPUs
    n_samples = _all_reduce(metric_sums, metric_counts, n_samples, device, world_size)

    # Report — same format as evaluate_hf.py for easy copy-paste comparison
    if is_main:
        final = {k: metric_sums[k] / max(metric_counts[k], 1) for k in metric_sums}
        pad = max(30, *(len(k) for k in final)) if final else 30
        print(f"\n{'='*60}")
        print(f"AutoVLA on Physical AI AV  |  n={n_samples}")
        print(f"Chunks  : {args.chunk_ids}")
        print(f"Config  : {args.config}")
        print(f"{'='*60}")
        print(f"{'val/count':<{pad}}  {n_samples}")
        for k in sorted(final):
            print(f"{'val/' + k.removeprefix('metric/'):<{pad}}  {final[k]:.4f}")
        print(f"{'='*60}")

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
