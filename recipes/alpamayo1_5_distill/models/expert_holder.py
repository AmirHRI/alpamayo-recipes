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

"""Just the teacher's frozen action expert, without dragging its 8B tower along.

``L_block`` needs the expert inside the TRAINING loop, which no previous arm did.  The
obvious way to get it -- build a ``StitchedAlpamayoR1.from_teacher`` and use its ``.expert``
-- also materialises the teacher's 8B VLM, ~16 GB that would sit idle on every rank for the
entire run.  This builds the expert (~2 B) and the action projections alone, ~4 GB.

The expert config is derived exactly as ``AlpamayoR1.__init__`` derives it: start from the
VLM's ``text_config``, then apply ``expert_cfg``.  Doing it from the STUDENT's text config is
safe and was verified against the teacher's: every field the expert inherits matches
(36 layers, 8 kv-heads, head_dim 128, rope_theta 5e6, mrope_section [24,20,20]) and the two
that differ (hidden_size, intermediate_size) are both overridden by ``expert_cfg``.  So the
expert comes out identical either way and the teacher's 397 tensors load unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import os

import torch
import torch.nn as nn
from hydra.utils import instantiate
from safetensors.torch import load_file
from transformers import AutoModel

logger = logging.getLogger(__name__)

#: Teacher tensors this module owns. `vlm.*` is excluded -- that is the whole point --
#: and so is `action_out_proj`: L_block compares BLOCK OUTPUTS, never the predicted
#: velocity, so the output head is never called. `diffusion` has no parameters at all.
PREFIXES = ("expert.", "action_in_proj.")


def _read_config(checkpoint_path: str) -> dict:
    path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(path):
        snaps = os.path.join(checkpoint_path, "snapshots")
        if os.path.isdir(snaps):
            cands = sorted(os.listdir(snaps))
            if cands:
                path = os.path.join(snaps, cands[-1], "config.json")
    with open(path) as fh:
        return json.load(fh)


class FrozenExpert(nn.Module):
    """The teacher's action expert + action projections, frozen, eval-mode, no grad.

    ⚠️ Held by the KD student OUTSIDE ``nn.Module`` registration (see ``kd_model``), so it
    never enters the optimizer, the ZeRO shard, or a checkpoint. Registering it would add
    ~2 B params to every saved checkpoint and to the optimizer state.
    """

    def __init__(self, checkpoint_path: str, student_text_config) -> None:
        super().__init__()
        cfg = _read_config(checkpoint_path)

        expert_config = copy.deepcopy(student_text_config)
        for key, value in (cfg.get("expert_cfg") or {}).items():
            setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # The expert never embeds tokens -- it consumes projected actions. AlpamayoR1 deletes
        # this too (alpamayo_r1.py:98); keeping it would waste a 155k x 2048 table.
        if hasattr(self.expert, "embed_tokens"):
            del self.expert.embed_tokens

        self.action_space = instantiate(cfg["action_space_cfg"])
        self.action_in_proj = instantiate(
            cfg["action_in_proj_cfg"],
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )

        self._load(checkpoint_path)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        self.n_action_tokens = self.action_space.get_action_space_dims()[0]
        self.x_dims = self.action_space.get_action_space_dims()
        self.n_layers = len(self.expert.layers)

    def _load(self, checkpoint_path: str) -> None:
        index = os.path.join(checkpoint_path, "model.safetensors.index.json")
        with open(index) as fh:
            weight_map = json.load(fh)["weight_map"]
        wanted = [k for k in weight_map if k.startswith(PREFIXES)]
        state: dict[str, torch.Tensor] = {}
        for shard in sorted({weight_map[k] for k in wanted}):
            sd = load_file(os.path.join(checkpoint_path, shard))
            state.update({k: v for k, v in sd.items() if k.startswith(PREFIXES)})
        missing, unexpected = self.load_state_dict(state, strict=False)
        params = dict(self.named_parameters())
        # Only a missing PARAMETER is a fault; deterministic buffers (action_in_proj.*.freqs)
        # are recomputed at init and are absent from the checkpoint by design.
        bad = [k for k in missing if k.startswith(PREFIXES) and k in params]
        if unexpected or bad:
            raise RuntimeError(f"frozen expert load failed: unexpected={unexpected[:5]} missing={bad[:5]}")
        logger.info(f"[block] frozen expert: {len(state)} tensors, {len(self.expert.layers)} layers")

    def initial_action_embeds(self, batch: int, device, dtype) -> torch.Tensor:
        """Action embeddings for the sampler's FIRST denoising step.

        ``flow_matching._euler`` starts at ``x ~ N(0, I)`` and ``t = 0.0``, so this is exactly
        on-distribution for that step rather than an arbitrary (x, t). Averaging the loss over
        several points along the sampling trajectory is the obvious refinement if the arm
        pays off; one point keeps the cost at a single expert forward.
        """
        x = torch.randn(batch, *self.x_dims, device=device, dtype=torch.float32)
        t = torch.zeros(batch, *([1] * len(self.x_dims)), device=device, dtype=torch.float32)
        # ⚠️ autocast, not a manual cast. action_in_proj mixes precisions internally (its
        # Fourier/sinus encoders build fp32 frequency tables and RMSNorm upcasts), so feeding
        # it bf16 against cast weights raises "mat1 and mat2 must have the same dtype".
        # evaluate_hf.py:122 runs the whole rollout under autocast for exactly this reason;
        # matching it keeps the queries numerically identical to inference.
        with torch.autocast(device.type, dtype=dtype):
            embeds = self.action_in_proj(x, t)
        if embeds.dim() == 2:
            embeds = embeds.view(batch, self.n_action_tokens, -1)
        return embeds.to(dtype)
