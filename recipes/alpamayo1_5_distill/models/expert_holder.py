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
import re

import torch
import torch.nn as nn
from hydra.utils import instantiate
from safetensors.torch import load_file
from transformers import AutoModel

logger = logging.getLogger(__name__)

#: Teacher tensors this module owns. `vlm.*` is excluded -- that is the whole point --
#: and so is `action_out_proj`: L_block compares BLOCK OUTPUTS, never the predicted
#: velocity, so the output head is never called. `diffusion` has no parameters at all.
#: ``action_out_proj`` is loaded ALWAYS since L_field: it is the velocity head, ~12
#: tensors, and the term that matters most is measured through it. Leaving it out
#: would let it sit at its random init and produce a healthy-looking loss on noise.
PREFIXES = ("expert.", "action_in_proj.", "action_out_proj.")


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

        # ⚠️ action_out_proj + diffusion are loaded ONLY for BLOCK_ODE=1. L_block compares block
        # OUTPUTS and never the predicted velocity, so the training path genuinely does not need
        # them (see the note above); the ODE diagnostic integrates the real sampler and does.
        # ⚠️ kwargs mirror AlpamayoR1.__init__ exactly (in_features/out_features, and
            # x_dims on the sampler). Guessing the names here builds a differently-shaped head
            # that then fails to load, or worse, loads partially.
        self.action_out_proj = instantiate(
            cfg["action_out_proj_cfg"],
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )
        # The sampler holds NO parameters and is needed by both the ODE probe and the rollout
        # objective (for its t grid and step count), so it is always built.
        self.diffusion = instantiate(
            cfg["diffusion_cfg"], x_dims=self.action_space.get_action_space_dims()
        )
        self._ode = os.environ.get("BLOCK_ODE") == "1"

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
        prefixes = PREFIXES
        wanted = [k for k in weight_map if k.startswith(prefixes)]
        state: dict[str, torch.Tensor] = {}
        for shard in sorted({weight_map[k] for k in wanted}):
            sd = load_file(os.path.join(checkpoint_path, shard))
            state.update({k: v for k, v in sd.items() if k.startswith(prefixes)})
        # ⚠️ DEPTH REMAP. The expert's layer count is derived from the VLM's text config, so a
        # 28-layer student (Cosmos-Reason2-2B) yields a 28-layer expert while the teacher
        # checkpoint holds 36. Pruning is therefore a LOAD-TIME REMAP -- teacher expert layer
        # pi(j) becomes student expert layer j -- not identity stand-ins in a 36-slot list.
        # PRUNE_EXPERT_LAYERS names the teacher layers to DROP; the survivors, in order, are
        # pi. Without this the load dies on unexpected=['expert.layers.28...'].
        n_have = len(self.expert.layers)
        n_ckpt = 1 + max((int(m.group(1)) for m in
                          (re.match(r"expert\.layers\.(\d+)\.", k) for k in state) if m),
                         default=-1)
        if n_ckpt > n_have:
            drop = {int(x) for x in os.environ.get("PRUNE_EXPERT_LAYERS", "").split(",")
                    if x.strip()}
            surv = [i for i in range(n_ckpt) if i not in drop]
            if len(surv) != n_have:
                raise RuntimeError(
                    f"expert depth {n_have} but checkpoint has {n_ckpt}; "
                    f"PRUNE_EXPERT_LAYERS must drop exactly {n_ckpt - n_have} layers "
                    f"(currently drops {len(drop)})")
            remap = {pi: j for j, pi in enumerate(surv)}
            out: dict[str, torch.Tensor] = {}
            for k, v in state.items():
                m = re.match(r"(expert\.layers\.)(\d+)(\..*)", k)
                if not m:
                    out[k] = v
                    continue
                old_i = int(m.group(2))
                if old_i in remap:                      # dropped layers are simply not loaded
                    out[f"{m.group(1)}{remap[old_i]}{m.group(3)}"] = v
            state = out
            self._pi = surv
            logger.warning("[block] expert DEPTH REMAP %d -> %d layers; dropped %s; "
                           "student expert j <- teacher expert pi(j)",
                           n_ckpt, n_have, sorted(drop))
            print(f"[block] expert REMAP {n_ckpt}->{n_have}, dropped {sorted(drop)}", flush=True)
        else:
            self._pi = None
        missing, unexpected = self.load_state_dict(state, strict=False)
        params = dict(self.named_parameters())
        # Only a missing PARAMETER is a fault; deterministic buffers (action_in_proj.*.freqs)
        # are recomputed at init and are absent from the checkpoint by design.
        bad = [k for k in missing if k.startswith(prefixes) and k in params]
        if unexpected or bad:
            raise RuntimeError(f"frozen expert load failed: unexpected={unexpected[:5]} missing={bad[:5]}")
        logger.info(f"[block] frozen expert: {len(state)} tensors, {len(self.expert.layers)} layers")

    def velocity(self, h_last: torch.Tensor) -> torch.Tensor:
        """Final block output -> predicted velocity, the quantity the trajectory depends on.

        ⚠️ The expert applies its FINAL NORM before ``last_hidden_state``
        (``modeling_qwen3.py:421``), so the head must be fed ``norm(h)``, not ``h``. Skipping
        it is the same class of bug that pinned ``freerun_loss`` at 0.999: it runs, it is
        finite, and it measures the wrong thing. Owned here so L_field and the ODE probe
        cannot drift apart.
        """
        h = self.expert.norm(h_last)
        return self.action_out_proj(h).view(-1, *self.x_dims)

    def noisy_x(self, traj_data: dict, t: torch.Tensor, device):
        """The raw noisy ACTION at ``t`` -- the sampler's state, before ``action_in_proj``.

        ``noisy_action_embeds`` returns the projected embeddings; a rollout needs the state
        itself so it can be advanced by ``x + dt * v``. Same interpolation, same convention:
        ``noisy_x = t * x + (1 - t) * noise``.
        """
        action = self.action_space.traj_to_action(
            traj_history_xyz=traj_data["ego_history_xyz"],
            traj_history_rot=traj_data["ego_history_rot"],
            traj_future_xyz=traj_data["ego_future_xyz"],
            traj_future_rot=traj_data["ego_future_rot"],
        ).reshape(-1, *self.x_dims).to(device=device, dtype=torch.float32)
        noise = torch.randn(action.shape, device=device, dtype=torch.float32)
        tt = t.to(device=device, dtype=torch.float32)
        while tt.dim() < action.dim():
            tt = tt.unsqueeze(-1)
        return tt * action + (1.0 - tt) * noise

    def noisy_action_embeds(self, traj_data: dict, t: torch.Tensor, device, dtype):
        """Action embeddings at a SAMPLED point on the flow, built from the GT trajectory.

        Reproduces ``FlowMatching.construct_training_data``'s interpolation exactly --
        ``noisy_x = t * x + (1 - t) * noise`` -- with ``x`` the ground-truth action. At t=0
        this reduces to :meth:`initial_action_embeds` (pure noise); at t=1 it is the true
        action. Supervising only t=0, as the original L_block did, covers the single noisiest
        point on the flow; CKA showed the expert transforms its representation MOST around
        t~0.2-0.4, which is why sampling t is worth testing.

        ⚠️ The same (x_t, t) drives BOTH sides of the block loss, so the teacher-forcing
        argument is untouched: Q, K_a and V_a stay identical by construction and the VLM
        cache remains the only difference.
        """
        action = self.action_space.traj_to_action(
            traj_history_xyz=traj_data["ego_history_xyz"],
            traj_history_rot=traj_data["ego_history_rot"],
            traj_future_xyz=traj_data["ego_future_xyz"],
            traj_future_rot=traj_data["ego_future_rot"],
        ).reshape(-1, *self.x_dims).to(device=device, dtype=torch.float32)
        noise = torch.randn(action.shape, device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32)
        while t.dim() < action.dim():
            t = t.unsqueeze(-1)
        x_t = t * action + (1.0 - t) * noise
        # autocast, not a manual cast -- action_in_proj mixes precisions internally; see
        # initial_action_embeds for the full reason.
        with torch.autocast(device.type, dtype=dtype):
            embeds = self.action_in_proj(x_t, t)
        if embeds.dim() == 2:
            embeds = embeds.view(x_t.shape[0], self.n_action_tokens, -1)
        # ⚠️ .to(dtype) like initial_action_embeds: action_in_proj ends in a LayerNorm, which
        # autocast keeps in fp32, so the raw output meets the expert's bf16 weights and
        # raises "expected mat1 and mat2 to have the same dtype".
        return embeds.to(dtype)

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
