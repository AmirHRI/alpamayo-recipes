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

"""Replay a finished run's ``trainer_state.json`` into a Weights & Biases run.

The ``ce`` control arm (job 392, 1,598 steps) ran with ``report_to: none`` -- wandb was
switched off earlier in the project because ``init_wandb(**cfg.wandb)`` expands every key
and the shipped group leaves ``team: ???``, which raises during hydra resolution.  That is
fixed in the config now, but the control had already finished.

Leaving it un-backfilled would make the control the one arm with no curve, which defeats
the point of having a control: ce / kd / kv only mean anything read against each other on
the same axes.  ``trainer_state.json`` retains the complete ``log_history`` (every scalar
HF logged, at the right ``step``), so the replay is lossless for everything except
wall-clock and system metrics.

The replayed run is tagged ``backfilled`` so nobody later mistakes it for a live run whose
system panels are simply empty.

Usage::

    python -m alpamayo1_5_distill.scripts.backfill_wandb \\
        --run-dir /data/.../output_kd_4b_ce_lcdrive \\
        --name kd_4b_ce_0806-1935 --arm ce
"""

from __future__ import annotations

import argparse
import json
import os

import wandb

#: Must match `wandb:` in configs/sft_kd_qwen3_4b_lcdrive.yaml, or the backfilled control
#: lands somewhere the live arms are not and the comparison silently does not exist.
TEAM = "zrb20"
PROJECT = "alpamayo1_5-kd-qwen3-4b"
GROUP = "kd_4b_lcdrive_arms"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="output_dir containing checkpoint-*/")
    ap.add_argument("--name", required=True, help="wandb run name; use the original run_name")
    ap.add_argument("--arm", required=True, choices=["ce", "kd", "kv"])
    ap.add_argument("--team", default=TEAM)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--group", default=GROUP)
    args = ap.parse_args()

    # Prefer the LAST checkpoint: log_history is cumulative, so the final one holds every
    # entry and the earlier ones are strict prefixes.
    ckpts = sorted(
        (d for d in os.listdir(args.run_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    )
    if not ckpts:
        raise SystemExit(f"no checkpoint-* under {args.run_dir}")
    state_path = os.path.join(args.run_dir, ckpts[-1], "trainer_state.json")
    with open(state_path) as fh:
        state = json.load(fh)
    history = state["log_history"]
    print(f"[backfill] {state_path}: {len(history)} entries, final step {state['global_step']}")

    config = {}
    cfg_path = os.path.join(args.run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml

        with open(cfg_path) as fh:
            config = yaml.safe_load(fh) or {}
    config["arm"] = args.arm
    config["backfilled_from"] = state_path

    run = wandb.init(
        entity=args.team,
        project=args.project,
        group=args.group,
        name=args.name,
        job_type=args.arm,
        tags=["backfilled", f"arm-{args.arm}"],
        config=config,
        force=True,
    )
    for entry in history:
        # `step` is HF's own optimizer step. Passing it explicitly keeps this run's x-axis
        # identical to the live arms' -- without it wandb would number by call order, and
        # the control would be offset against everything it is meant to be compared to.
        step = entry.get("step")
        scalars = {k: v for k, v in entry.items() if k != "step" and isinstance(v, (int, float))}
        if scalars:
            run.log(scalars, step=step)
    print(f"[backfill] done -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
