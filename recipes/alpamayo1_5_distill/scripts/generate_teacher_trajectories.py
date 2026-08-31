# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache exact full-Alpamayo action rollouts for offline consistency training.

For each ``(clip_id, t0)`` the full 8B-VLM + 36-layer action expert runs once
with ``K`` independent noises.  The cache records the input to every Euler step
and the final action, giving ``[K, M+1, 64, 2]`` states in native sampler order
(``s=0`` noise -> ``s=1`` data).  Training can then use adjacent full-teacher
states while the 2B student builds its own, architecture-compatible VLM cache.
"""

from __future__ import annotations

import hashlib

import hydra.utils as hyu
import torch

from alpamayo1_5_distill.data import teacher_trajectory_io
from alpamayo1_5_distill.scripts import cache_common


def seed_for_key(key: str, base_seed: int) -> int:
    """Stable seed independent of sharding, resume order, and Python hash salt."""
    digest = hashlib.sha256(f"{int(base_seed)}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def assemble_rollout_states(
    pre_step_states: list[torch.Tensor],
    final_action: torch.Tensor,
    *,
    num_steps: int,
    num_noise: int,
) -> torch.Tensor:
    """Turn ``M`` probed inputs plus the sampler result into ``[K,M+1,T,C]``."""
    if len(pre_step_states) != num_steps:
        raise ValueError(
            f"teacher probe captured {len(pre_step_states)} states, expected {num_steps}"
        )
    if final_action.ndim != 3 or final_action.shape[0] != num_noise:
        raise ValueError(
            f"final action must be [K,T,C] with K={num_noise}, got "
            f"{tuple(final_action.shape)}"
        )
    for i, state in enumerate(pre_step_states):
        if state.shape != final_action.shape:
            raise ValueError(
                f"teacher state {i} has {tuple(state.shape)}, final has "
                f"{tuple(final_action.shape)}"
            )
    return torch.stack([*pre_step_states, final_action], dim=1).float()


def main() -> None:
    argv = cache_common.main_argv()
    config_name = argv.get("config", "cache_full_teacher_trajectories_2cam_nav_lcdrive")
    cache_root = argv.get("cache_root")
    if not cache_root:
        raise SystemExit("cache_root=<dir> is required (put it on /data)")

    num_steps = int(argv.get("num_steps", 10))
    num_noise = int(argv.get("num_noise", 6))
    base_seed = int(argv.get("seed", 1234))
    limit = int(argv.get("limit", 0))
    num_shards = int(argv.get("num_shards", 1))
    shard = int(argv.get("shard", 0))
    num_workers = int(argv.get("num_workers", 6))
    log_every = int(argv.get("log_every", 25))
    if num_steps < 1 or num_noise < 1:
        raise SystemExit("num_steps and num_noise must both be positive")
    if not 0 <= shard < num_shards:
        raise SystemExit(f"shard must satisfy 0 <= shard < num_shards, got {shard}/{num_shards}")

    cfg = cache_common.compose_config(config_name, argv)
    device = torch.device("cuda")
    print(
        f"[teacher-rollout] full teacher, K={num_noise}, M={num_steps}, "
        f"shard={shard}/{num_shards}, root={cache_root}",
        flush=True,
    )
    model = cache_common.build_teacher(cfg, device)
    if not hasattr(model, "sample_trajectories_prefill_only"):
        raise TypeError(
            "cache model must expose sample_trajectories_prefill_only; use "
            "StitchedAlpamayoR1.from_teacher"
        )
    dataset = hyu.instantiate(
        cfg.data.cache_dataset, _convert_="partial", model_config=model.config
    )
    processor = cache_common.build_processor(model)

    owned = cache_common.shard_indices(len(dataset), num_shards, shard)
    todo = [
        i for i in owned
        if not teacher_trajectory_io.has_entry(cache_root, dataset._sample_key(i))
    ]
    if limit:
        todo = todo[:limit]
    print(
        f"[teacher-rollout] dataset={len(dataset)} assigned={len(owned)} "
        f"to-do={len(todo)} workers={num_workers}",
        flush=True,
    )

    provenance = {
        "format": teacher_trajectory_io.FORMAT,
        "config_name": config_name,
        "teacher_target": cfg.model.get("_target_"),
        "teacher_checkpoint": cfg.model.get("checkpoint_path"),
        "teacher_vlm": cfg.model.get("vlm_name_or_path"),
        "num_steps": num_steps,
        "num_noise": num_noise,
        "base_seed": base_seed,
        "cameras": list(cfg.data.cache_dataset.get("cameras", [])),
        "annotations_path": cfg.data.cache_dataset.get("annotations_path"),
    }
    entries: dict[str, int] = {}
    reporter = cache_common.RateReporter(len(todo), label="teacher-rollout")
    loader = cache_common.sample_loader(dataset, todo, processor, num_workers=num_workers)
    done = 0

    for scanned, (idx, batch) in enumerate(loader, start=1):
        if batch is None:
            continue
        key = dataset._sample_key(idx)
        sample_seed = seed_for_key(key, base_seed)
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        batch = cache_common.to_device(batch, device)
        pre_step_states: list[torch.Tensor] = []
        step_times: list[torch.Tensor] = []

        def probe(i, t, x, _last_hidden, _velocity) -> None:
            if i != len(pre_step_states):
                raise RuntimeError(f"non-sequential teacher probe: i={i}")
            pre_step_states.append(x.detach().float().clone())
            step_times.append(t[0, 0, 0].detach().float().clone())

        try:
            _xyz, _rot, action = model.sample_trajectories_prefill_only(
                data=batch,
                num_traj_samples=num_noise,
                num_traj_sets=1,
                diffusion_kwargs={"inference_step": num_steps},
                return_action=True,
                step_probe=probe,
            )
            states = assemble_rollout_states(
                pre_step_states,
                action[0, 0].detach(),
                num_steps=num_steps,
                num_noise=num_noise,
            )
            actual_times = torch.stack(step_times).cpu()
            expected_times = torch.arange(num_steps, dtype=torch.float32) / num_steps
            torch.testing.assert_close(actual_times, expected_times, atol=1e-6, rtol=0)
        except (AssertionError, ValueError, RuntimeError) as ex:
            print(
                f"[teacher-rollout] idx={idx} key={key} capture failed: {ex}",
                flush=True,
            )
            continue

        teacher_trajectory_io.save_states(
            cache_root,
            key,
            states,
            metadata={**provenance, "sample_seed": sample_seed},
        )
        entries[key] = sample_seed
        done += 1
        if done % log_every == 0:
            teacher_trajectory_io.write_index(cache_root, entries, provenance, shard)
            reporter.report(done, scanned)

    teacher_trajectory_io.write_index(cache_root, entries, provenance, shard)
    stats = teacher_trajectory_io.tier_stats(cache_root)
    projected_gb = float(stats["kb_per_entry"]) * len(dataset) / 1024**2
    print(
        f"[teacher-rollout] complete: wrote={done}; cache has "
        f"{stats['n_entries']} entries at {stats['kb_per_entry']:.1f} KiB/entry; "
        f"projected full size={projected_gb:.2f} GiB",
        flush=True,
    )
    failed = len(todo) - done
    if failed:
        raise RuntimeError(
            f"teacher trajectory cache shard {shard}/{num_shards} missed "
            f"{failed}/{len(todo)} assigned entries; resume this shard before training"
        )


if __name__ == "__main__":
    main()
