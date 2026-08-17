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

r"""Drive the KD model's ``BLOCK_FREERUN`` diagnostic under ``no_grad``.

The diagnostic itself lives in ``kd_model._block_loss`` so it runs under EXACTLY the training
conditions -- one prefill per tower over the same token sequence, no CoT generation, so both
caches have identical length and differ only in values.  This file only supplies the loop.

⚠️ Why not just run ``train_kd`` with ``max_steps=32``: that builds the training graph for the
4 B student VLM (gradients + retained activations, ~56 GB) even at ``learning_rate=1e-12``,
and OOMs on a shared card.  Measurement needs no gradients; ``no_grad`` brings it down to
roughly the weights alone.

Usage::

    BLOCK_FREERUN=1 CUDA_VISIBLE_DEVICES=2 python -m alpamayo1_5_distill.scripts.freerun_probe \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_kd_qwen3_4b_lcdrive \
        ++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0 \
        ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta \
        ++model.checkpoint_path=<student ckpt> ++probe.n_clips=32
"""

from __future__ import annotations

import hydra
import hydra.utils as hyu
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    n_clips = int(cfg.get("probe", {}).get("n_clips", 32))
    dev = torch.device("cuda")
    model = hyu.instantiate(cfg.model, _convert_="partial").to(dev).eval()
    for p in model.parameters():                  # belt and braces: no graph anywhere
        p.requires_grad_(False)
    print(f"[probe] model resident, {n_clips} clips", flush=True)

    ds = hyu.instantiate(cfg.data.train_dataset, _convert_="partial",
                         model_config=model.config)
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial",
                              model_config=model.config)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=2, shuffle=False)

    done = 0
    for batch in loader:
        if done >= n_clips:
            break
        gpu = send_to_device(dict(batch), dev)
        gpu.pop("clip_id", None)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**gpu)
        done += 1
        if done % 8 == 0:
            print(f"[probe] {done}/{n_clips}", flush=True)
    print("DONE_PROBE", flush=True)


if __name__ == "__main__":
    main()
