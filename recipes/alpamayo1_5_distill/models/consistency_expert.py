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

r"""Consistency distillation of an action expert against online or cached teacher steps.

Replaces ``TrainableAlpamayoR1.forward``'s flow-matching loss with the consistency objective
in :mod:`alpamayo1_5_distill.models.consistency_losses`.

``teacher_source="online"`` runs three expert forwards per step:

    v_teacher(x_hi, tau_hi)   frozen pretrained expert, no grad   <- the 1 teacher NFE
    v_target(x_lo, tau_lo)    EMA of the trainable expert, no grad   (skipped on the anchor)
    v_online(x_hi, tau_hi)    trainable expert, WITH grad

plus one frozen VLM prefill, whose KV cache all three share.

``teacher_source="cached_full"`` replaces the first line with an exact adjacent
transition cached from the full Alpamayo-1.5 8B-VLM + 36-layer expert rollout.  The
teacher velocity is recovered as ``(x_lo - x_hi) / delta_tau``, so the existing
stable endpoint CD algebra is unchanged. Only the student's frozen 2B VLM is
prefilled online, and its cache is consumed only by its own online/EMA expert.
This is output-level policy distillation across architectures; it neither assumes
nor attempts position-wise compatibility between the full and student VLM caches.

⚠️ TIME CONVENTION. This file speaks ``tau`` (0 = data, 1 = noise) because that is how the
objective is written; the expert speaks ``s = 1 - tau``. Every call into ``action_in_proj``
goes through :func:`~alpamayo1_5_distill.models.consistency_losses.tau_to_s`. See that
module's docstring for the two mirror errors this prevents.

⚠️ THE CACHE IS MUTATED BY EVERY EXPERT CALL. ``self.expert(..., use_cache=True)`` appends
the 64 action K/V to the ``DynamicCache``. With three calls per step the second one would
attend to the first one's action tokens unless the cache is cropped back each time -- the
same reason ``sample_trajectories_prefill_only`` calls ``prompt_cache.crop(prefill_seq_len)``
inside its ``step_fn``.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from transformers.modeling_outputs import ModelOutput

from alpamayo1_5_distill.models.consistency_losses import (
    DEFAULT_M,
    cached_teacher_endpoint,
    cd_loss,
    consistency_fn,
    interpolate,
    needs_target,
    sample_cached_teacher_transition,
    sample_rungs,
    tau_to_s,
    teacher_step,
    transition_velocity,
    x0_reconstruction_loss,
)
from alpamayo1_5_distill.models.expert_conditioning import (
    ExpertConditioning,
    build_expert_conditioning,
)
from alpamayo1_5_distill.models.expert_holder import FrozenExpert
# ⚠️ Reuse kd_model's timer rather than writing a second one: it already gets the
# CUDA sync right, and without the sync every phase but the last reports ~0 because
# the kernels are still queued. `CD_TIMERS=1` maps onto the same `KD_TIMERS` switch.
from alpamayo1_5_distill.models.kd_model import _Phase
from alpamayo1_5_distill.models.expert_teacher import KaVaExpertTeacher

logger = logging.getLogger(__name__)


def _layer_indices(value, *, name: str, n_layers: int) -> tuple[int, ...]:
    """Parse and validate a layer-index list from Hydra or the environment."""
    if value is None:
        raw = []
    elif isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
    else:
        raw = list(value)
    try:
        indices = [int(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer layer indices, got {value!r}") from exc
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} contains duplicate layer indices: {indices}")
    indices = sorted(indices)
    bad = [i for i in indices if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"{name} out of range for {n_layers} layers: {bad}")
    return tuple(indices)


def _skipped_expert_layer_indices(layers) -> tuple[int, ...]:
    """Return identity-pruned slots without importing the stitched model (avoids a cycle)."""
    return tuple(
        i for i, layer in enumerate(layers)
        if type(layer).__name__ == "_SkippedExpertLayer"
    )


def _validate_student_pruning(layers, configured, env_spec: str) -> tuple[int, ...]:
    """Require config, environment, and the materialized student to name the same cuts."""
    n_layers = len(layers)
    requested = _layer_indices(
        configured, name="model.cd.student_prune_layers", n_layers=n_layers
    )
    exported = _layer_indices(
        env_spec, name="PRUNE_EXPERT_LAYERS", n_layers=n_layers
    )
    actual = _skipped_expert_layer_indices(layers)
    if not requested:
        if exported or actual:
            raise RuntimeError(
                "PRUNE_EXPERT_LAYERS is set or the expert is already pruned, but "
                "model.cd.student_prune_layers is empty. Refusing an accidental ablation."
            )
        return ()
    if exported != requested:
        raise RuntimeError(
            "pruned consistency student mismatch: "
            f"config={list(requested)} PRUNE_EXPERT_LAYERS={list(exported)}"
        )
    if actual != requested:
        raise RuntimeError(
            "pruned consistency student was not materialized as configured: "
            f"config={list(requested)} actual_skipped={list(actual)}"
        )
    return requested


def _validate_full_teacher(layers, expected_layers: int) -> None:
    """Fail before training if the frozen target has reduced depth or identity slots."""
    skipped = _skipped_expert_layer_indices(layers)
    if len(layers) != expected_layers or skipped:
        raise RuntimeError(
            "consistency teacher must be full depth: "
            f"expected={expected_layers} slots={len(layers)} skipped={list(skipped)}"
        )


@dataclass
class CDVLAOutput(ModelOutput):
    """Everything but ``loss`` is detached, for ``KaVaTrainer`` to log.

    The per-rung splits are the early-warning instrument: a healthy bootstrap drops the
    ANCHOR band first and propagates toward noise, because that is the only band with a
    teacher-pure target. If the NOISE band falls fastest the model is collapsing to the
    conditional mean, which is the dominant failure mode here and is invisible in the total.
    """

    loss: torch.FloatTensor | None = None
    cd_loss: torch.FloatTensor | None = None
    x0_gt_loss: torch.FloatTensor | None = None
    x0_teacher_loss: torch.FloatTensor | None = None
    cd_loss_anchor: torch.FloatTensor | None = None
    cd_loss_mid: torch.FloatTensor | None = None
    cd_loss_noise: torch.FloatTensor | None = None
    x0_loss_anchor: torch.FloatTensor | None = None
    x0_loss_mid: torch.FloatTensor | None = None
    x0_loss_noise: torch.FloatTensor | None = None


class ConsistencyExpertVLA(KaVaExpertTeacher):
    """Frozen teacher VLM + optionally pruned expert trained by consistency distillation."""

    # ------------------------------------------------------------------ setup
    def init_cd(
        self,
        *,
        teacher_checkpoint_path: str,
        teacher_source: str = "online",
        m_rungs: int = DEFAULT_M,
        cd_weight: float = 1.0,
        x0_gt_weight: float = 0.0,
        x0_teacher_weight: float = 0.0,
        x0_gt_normalizer: float = 1.0,
        x0_source: str = "gt",
        metric: str = "mse",
        huber_c: float | None = None,
        normalizer: float = 1.0,
        seed: int | None = None,
        student_prune_layers: list[int] | tuple[int, ...] | str | None = None,
        teacher_num_layers: int = 36,
    ) -> None:
        self.m_rungs = int(m_rungs)
        self.cd_weight = float(cd_weight)
        self.x0_gt_weight = float(x0_gt_weight)
        self.x0_teacher_weight = float(x0_teacher_weight)
        self.x0_gt_normalizer = float(x0_gt_normalizer)
        self.x0_source = x0_source
        self.cd_metric = metric
        self.cd_huber_c = huber_c
        self.cd_normalizer = float(normalizer)
        self.teacher_source = str(teacher_source)
        self._cd_ckpt = teacher_checkpoint_path
        self.keep_loss_terms = False
        self.last_loss_terms: dict[str, torch.Tensor] | None = None
        # ⚠️ Held OUTSIDE nn.Module registration, exactly like kd_model's frozen expert and
        # teacher: a registered 2.28 B frozen copy would enter the optimizer, the ZeRO shard
        # and every 14 GB checkpoint. The cost is that nothing moves it -- see _ensure_teacher.
        self._teacher_expert_holder: list = []
        self._cd_seed = seed
        self._teacher_num_layers = int(teacher_num_layers)
        if self.x0_gt_weight < 0:
            raise ValueError(f"x0_gt_weight must be non-negative, got {self.x0_gt_weight}")
        if self.x0_teacher_weight < 0:
            raise ValueError(
                f"x0_teacher_weight must be non-negative, got {self.x0_teacher_weight}"
            )
        if self.x0_gt_normalizer <= 0:
            raise ValueError(
                f"x0_gt_normalizer must be positive, got {self.x0_gt_normalizer}"
            )
        if self.teacher_source not in ("online", "cached_full"):
            raise ValueError(
                "teacher_source must be 'online' or 'cached_full', got "
                f"{self.teacher_source!r}"
            )
        if x0_source not in ("gt", "teacher"):
            raise ValueError(f"x0_source must be 'gt' or 'teacher', got {x0_source!r}")
        # ⚠️ LEGACY SHIM, kept so `sft_cd_eos_2b_fullteacher_endpoint_nav_lcdrive`
        # (the Arm A run) still reproduces bit-for-bit. That config expresses the endpoint
        # objective as `x0_source: teacher` + `x0_gt_weight: 1.0`, from before the two
        # targets became independently weightable. Route it, loudly -- a silent
        # reinterpretation of a weight is exactly the class of bug this file is full of
        # warnings about. New configs should set `x0_teacher_weight` directly and leave
        # `x0_source` at its default.
        if x0_source == "teacher" and self.x0_teacher_weight == 0.0:
            self.x0_teacher_weight = self.x0_gt_weight
            self.x0_gt_weight = 0.0
            logger.warning(
                "[cd] legacy x0_source=teacher: moved x0_gt_weight=%.3g to "
                "x0_teacher_weight; set x0_teacher_weight explicitly instead",
                self.x0_teacher_weight,
            )
            print(f"[cd] LEGACY x0_source=teacher -> x0_teacher_weight="
                  f"{self.x0_teacher_weight}, x0_gt_weight=0", flush=True)
        # The noise-PAIRED endpoint states[k, M] only exists in cached_full mode; an online
        # teacher would have to be rolled out for M steps per iteration to produce one,
        # which is the cost this arm exists to avoid.
        if self.x0_teacher_weight > 0.0 and self.teacher_source != "cached_full":
            raise ValueError(
                "x0_teacher_weight needs the cached rollout endpoint; set "
                "teacher_source='cached_full'"
            )
        if (self.cd_weight == 0.0 and self.x0_gt_weight == 0.0
                and self.x0_teacher_weight == 0.0):
            raise ValueError("every loss weight is zero -- nothing to train")
        if len(self.expert.layers) != self._teacher_num_layers:
            raise RuntimeError(
                "the cache-aligned pruning path requires the student to retain the teacher's "
                f"{self._teacher_num_layers} layer slots; got {len(self.expert.layers)}"
            )
        self._student_prune_layers = _validate_student_pruning(
            self.expert.layers,
            student_prune_layers,
            os.environ.get("PRUNE_EXPERT_LAYERS", ""),
        )
        n_active = len(self.expert.layers) - len(self._student_prune_layers)
        teacher_label = (
            "offline full Alpamayo rollout"
            if self.teacher_source == "cached_full"
            else f"{self._teacher_num_layers}-layer online expert"
        )
        logger.warning(
            "[cd] M=%d cd_w=%.3g x0_teacher_w=%.3g x0_gt_w=%.3g metric=%s "
            "student=%d/%d active teacher=%s",
            self.m_rungs, self.cd_weight, self.x0_teacher_weight, self.x0_gt_weight,
            self.cd_metric, n_active, len(self.expert.layers), teacher_label,
        )
        print(f"[cd] M={self.m_rungs} cd_w={self.cd_weight} "
              f"x0_teacher_w={self.x0_teacher_weight} x0_gt_w={self.x0_gt_weight} "
              f"metric={self.cd_metric} student={n_active}/{len(self.expert.layers)} active "
              f"skipped={list(self._student_prune_layers)}; "
              f"teacher={teacher_label}", flush=True)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        cd_cfg = dict(kwargs.pop("cd", None) or {})
        teacher_ckpt = kwargs.pop("teacher_checkpoint_path", None) or (
            args[0] if args else kwargs.get("pretrained_model_name_or_path")
        )
        model = super().from_pretrained(*args, **kwargs)
        model.init_cd(teacher_checkpoint_path=str(teacher_ckpt), **cd_cfg)
        return model

    @classmethod
    def from_eos_checkpoint(
        cls,
        *,
        eos_checkpoint_path: str,
        vlm_name_or_path: str,
        alpamayo_config_path: str,
        cd: dict[str, Any] | None = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
    ):
        """Start CD from an expert already trained on the student VLM (EoS).

        The Stage-2/EoS checkpoint contains both the frozen student VLM and its
        cache-adapted expert and always initializes the online model. In the default
        ``teacher_source="online"`` mode, the same checkpoint also supplies the
        unregistered frozen CD teacher, so all three branches consume one shared
        student-VLM cache. In ``teacher_source="cached_full"`` mode, no frozen
        expert is constructed; the target transitions come from offline full-model
        rollouts while the online and EMA branches use the student-VLM cache.

        ``cotrain_vlm`` is intentionally forbidden here. Updating the cache producer
        while either teacher reference remains fixed would make the student expert's
        conditioning distribution move during CD.
        """
        if cotrain_vlm:
            raise ValueError(
                "EoS consistency training requires a frozen student VLM; "
                "set model.cotrain_vlm=false"
            )
        cd_cfg = dict(cd or {})
        cd_cfg.setdefault("teacher_num_layers", 28)
        teacher_layers = int(cd_cfg["teacher_num_layers"])
        model = super().from_pretrained_vlm(
            vlm_name_or_path=vlm_name_or_path,
            alpamayo_config_path=alpamayo_config_path,
            stage2_checkpoint_path=eos_checkpoint_path,
            cotrain_vlm=False,
            stop_grad_from_vlm=stop_grad_from_vlm,
            expert_num_layers=teacher_layers,
        )
        # Be explicit after assign=True checkpoint loading: the cache producer must
        # remain fixed for the whole CD run.
        model.cotrain_vlm = False
        model.stop_grad_from_vlm = True
        model._set_vlm_trainability()

        # ⚠️ LAYER MIX. The EoS checkpoint of a layer-mix student carries `layer_mixer.*`
        # alongside its 36-layer expert, but `_load_modules_from_checkpoint` above filters to
        # ("vlm.", "expert.", "action_in_proj.", "action_out_proj.", "diffusion.",
        # "action_space.") -- `layer_mixer.` is NOT in that list, so the trained matrices
        # would be dropped without a word. Load them explicitly, and RAISE if absent: a
        # 28-layer cache reaching a 36-layer expert does not fail, it makes DynamicCache
        # auto-extend slots 28..35 holding action tokens only.
        if cd_cfg.pop("layer_mix", False):
            from alpamayo1_5_distill.models.layer_mix import (
                LAYER_MIX_SHARPEN, LayerMixer,
            )
            from alpamayo1_5_distill.models.stitched_model import _load_layer_mix

            ref = next(model.vlm.parameters())
            model.layer_mixer = LayerMixer(
                n_student=len(model.vlm.model.language_model.layers),
                n_expert=teacher_layers,
                n_blocks=int(cd_cfg.pop("layer_mix_blocks", 4)),
                gain=bool(cd_cfg.pop("layer_mix_gain", False)),
                sharpen=float(cd_cfg.pop("layer_mix_sharpen", LAYER_MIX_SHARPEN)),
                pin_head=int(cd_cfg.pop("layer_mix_pin_head", 0)),
                pin_tail=int(cd_cfg.pop("layer_mix_pin_tail", 0)),
            ).to(device=ref.device, dtype=ref.dtype)
            _load_layer_mix(str(eos_checkpoint_path), model)
            # The cache producer is frozen for the whole CD run, and P is part of it.
            for prm in model.layer_mixer.parameters():
                prm.requires_grad_(False)
            print(f"[cd] layer mix ACTIVE ({model.layer_mixer.n_student} -> "
                  f"{model.layer_mixer.n_expert}), P frozen", flush=True)
        model.init_cd(
            teacher_checkpoint_path=str(eos_checkpoint_path),
            **cd_cfg,
        )
        return model

    @property
    def teacher_expert(self):
        return self._teacher_expert_holder[0] if self._teacher_expert_holder else None

    def _ensure_teacher_expert(self, device, dtype):
        """Build and place the frozen teacher expert. Lazy, because at construction time the
        model is still on CPU and the device is only known once a batch arrives.

        ⚠️ Unregistered means ``Trainer``, ``.to()`` and accelerate all skip it. Left on CPU
        it dies at the first matmul with "Expected all tensors to be on the same device".
        """
        if self.teacher_source != "online":
            raise RuntimeError(
                "_ensure_teacher_expert called in cached_full mode; the full teacher "
                "must remain offline"
            )
        if not self._teacher_expert_holder:
            self._teacher_expert_holder.append(
                FrozenExpert(self._cd_ckpt, self.vlm.config.text_config)
            )
        e = self._teacher_expert_holder[0]
        if next(e.parameters()).device != device:
            self._teacher_expert_holder[0] = e.to(device=device, dtype=dtype)
        e = self._teacher_expert_holder[0]
        _validate_full_teacher(e.expert.layers, self._teacher_num_layers)
        return e

    # ------------------------------------------------------------------ velocity
    def _velocity(
        self,
        module,
        x,
        tau,
        cache,
        conditioning: ExpertConditioning,
        dtype,
    ):
        """``v_repo`` at ``(x, tau)`` from ``module`` (self, EMA-loaded self, or the teacher).

        Action states stay fp32 through Fourier encoding, matching inference; only the
        projected embeddings are cast to the expert dtype.

        ⚠️ ``tau_to_s(tau)`` -- the expert's timestep encoder was trained on the repo's ``s``.
        ⚠️ autocast around ``action_in_proj``/``action_out_proj``, not at the call site:
        ``PerWaypointActionInProjV2`` forces ``x.float()`` internally, so its fp32 activations
        meet bf16 weights and raise "mat1 and mat2 must have the same dtype" otherwise.
        ⚠️ ``crop`` afterwards: the forward appended 64 action K/V to the shared cache.
        """
        n_act = self.action_space.get_action_space_dims()[0]
        s = tau_to_s(tau)
        device_type = x.device.type
        with torch.autocast(
            device_type, dtype=dtype, enabled=device_type == "cuda"
        ):
            emb = module.action_in_proj(x, s)
        emb = emb.to(dtype)
        if emb.dim() == 2:
            emb = emb.view(x.shape[0], n_act, -1)
        out = module.expert(
            inputs_embeds=emb,
            position_ids=conditioning.position_ids,
            past_key_values=cache,
            attention_mask=conditioning.attention_mask,
            use_cache=True,
            **({"is_causal": False} if self.config.expert_non_causal_attention else {}),
        )
        cache.crop(conditioning.prefix_len)
        h = out.last_hidden_state[:, -n_act:]
        with torch.autocast(
            device_type, dtype=dtype, enabled=device_type == "cuda"
        ):
            # Qwen3VLTextModel.forward applies `expert.norm` before returning
            # `last_hidden_state`. Applying it again here changes the vector field relative
            # to the ordinary training and deployment sampler paths.
            return module.action_out_proj(h).view(-1, n_act, 2)

    # ------------------------------------------------------------------ Gate A
    @torch.no_grad()
    def _cd_selftest(self, x0, cache, conditioning, dtype, device):
        """Gate A on REAL data and the REAL field. Runs once, on `CD_SELFTEST=1`.

        The unit tests pin the algebra on synthetic fields; this pins the three things only
        the real expert can answer, and prints `L@init` -- the number a trained student has
        to beat, which every result should be quoted against.

        ⚠️ Meaningful only at initialisation, where the trainable expert and the frozen
        teacher are the same weights. Resuming from a checkpoint makes `L@init` mean
        something else, so the header says which case it is.
        """
        b, n_act = x0.shape[0], x0.shape[1]
        teacher = self._ensure_teacher_expert(device, dtype)
        same = torch.allclose(
            next(self.expert.parameters()).float(),
            next(teacher.expert.parameters()).float(), atol=0, rtol=0)
        if self._student_prune_layers:
            print(
                "[cd-selftest] first surviving student weight == teacher at init: "
                f"{same}; architectures differ "
                f"({len(self.expert.layers) - len(self._student_prune_layers)} active vs "
                f"{len(teacher.expert.layers)} full layers)",
                flush=True,
            )
        else:
            print(f"[cd-selftest] online expert == frozen teacher at init: {same}", flush=True)

        eps = torch.randn(x0.shape, device=device, dtype=torch.float32)
        v = lambda mod, x, tau: self._velocity(
            mod, x.float(), tau, cache, conditioning, dtype
        ).float()

        # 1) boundary. f(x, 0) = x exactly, for the real head at real scale.
        xr = torch.randn(x0.shape, device=device, dtype=torch.float32)
        tau0 = torch.zeros(b, 1, 1, device=device)
        f0 = consistency_fn(xr, tau0, v(self, xr, tau0))
        print(f"[cd-selftest] boundary |f(x,0) - x|_max = {float((f0 - xr).abs().max()):.3e} "
              f"(must be 0)", flush=True)

        # 2) tau vs s. Feeding the mirrored time must visibly change the field; if it does
        #    not, the timestep is not reaching the encoder and every rung is the same query.
        t_a = torch.full((b, 1, 1), 0.9, device=device)
        d_mirror = float((v(self, x0, t_a) - v(self, x0, 1.0 - t_a)).abs().mean())
        d_self = float(v(self, x0, t_a).abs().mean())
        print(f"[cd-selftest] |v(tau) - v(1-tau)| = {d_mirror:.4e} vs |v| = {d_self:.4e} "
              f"-> ratio {d_mirror / max(d_self, 1e-12):.3f} (must be >> 0)", flush=True)

        # 3) L@init per rung, and the 1/M^2 law.
        print("[cd-selftest]   M   L@init      per-rung (anchor -> noise)", flush=True)
        prev = None
        for m in (10, 20, 40):
            per_rung, tot = [], 0.0
            for n in range(m):
                lo = torch.full((b, 1, 1), n / m, device=device)
                hi = torch.full((b, 1, 1), (n + 1) / m, device=device)
                x_hi = interpolate(x0, eps, hi)
                v_t = v(teacher, x_hi, hi)
                x_lo = teacher_step(x_hi, v_t, hi - lo)
                v_g = None if n == 0 else v(teacher, x_lo, lo)
                li, _ = cd_loss(v(self, x_hi, hi), v_t, v_g, lo, hi,
                                metric=self.cd_metric, huber_c=self.cd_huber_c)
                per_rung.append(float(li))
                tot += float(li)
            l_init = tot / m
            # ⚠️ The anchor rung is ~0 at init and is 1/M of the rungs, so it dilutes the
            # mean MORE at small M -- which flatters the ratio downward and is not part of
            # the scaling law. Report both; `ex_anchor` is the one to read.
            l_ex = sum(per_rung[1:]) / max(m - 1, 1)
            shown = " ".join(f"{q:.1e}" for q in per_rung[:6])
            ratio = ""
            if prev is not None:
                r_all, r_ex = prev[0] / max(l_init, 1e-30), prev[1] / max(l_ex, 1e-30)
                # L ~ M^-p; a smooth field gives p = 2 (ratio 4 per doubling).
                ratio = (f"  ratio {r_all:.2f} (all) {r_ex:.2f} (ex_anchor)"
                         f"  p={__import__('math').log(r_ex) / __import__('math').log(2):.2f}")
            print(f"[cd-selftest] {m:>4}  {l_init:.4e}  ex_anchor {l_ex:.4e}  "
                  f"{shown} ...{ratio}", flush=True)
            prev = (l_init, l_ex)
        print("[cd-selftest] anchor rung must be ~0 at init (target is the teacher itself)",
              flush=True)
        print("DONE_CD_SELFTEST", flush=True)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        teacher_trajectory_states: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CDVLAOutput:
        input_ids = tokenized_data.pop("input_ids")
        b = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.expert.parameters()).dtype
        traj_data = {
            "ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz, "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 1) frozen VLM prefill. no_grad because cotrain_vlm is false in this arm; the cache
        #    is also detached below so no graph survives into the expert.
        ctx = nullcontext() if self.cotrain_vlm else torch.no_grad()
        with _Phase.t("vlm_prefill"), ctx:
            vlm_outputs = self.vlm(input_ids=input_ids, use_cache=True, **tokenized_data)

        n_act = self.action_space.get_action_space_dims()[0]
        fs_id = self.config.traj_token_ids["future_start"]
        conditioning = build_expert_conditioning(
            traj_future_start_mask=input_ids == fs_id,
            tokenizer_attention_mask=tokenized_data.get("attention_mask"),
            rope_deltas=getattr(vlm_outputs, "rope_deltas", None),
            n_action_tokens=n_act,
            dtype=dtype,
            attention_implementation=getattr(
                self.expert.config, "_attn_implementation", None
            ),
        )
        cache = vlm_outputs.past_key_values
        cache.crop(conditioning.prefix_len)
        # ⚠️ 28 VLM layers -> 36 expert slots, on the ONE cache all three CD branches share
        # (v_teacher, v_target/EMA, v_online). Mixing here rather than per-branch is what
        # keeps them reading identical conditioning -- the property the consistency
        # constraint is defined against.
        mixer = getattr(self, "layer_mixer", None)
        if mixer is not None:
            cache = mixer.mix_cache(cache)
        for layer in cache.layers:
            layer.keys = layer.keys.detach()
            layer.values = layer.values.detach()

        # 2) x0. Pure CD uses x0 only to choose where the constraint is evaluated. When
        #    x0_gt_weight > 0, the same dataset action also directly supervises the online
        #    one-step endpoint. This aligns the objective with one-step ADE, at the cost of
        #    encouraging the conditional mean and potentially reducing sample diversity.
        with _Phase.t("traj_to_action"):
            x0 = self.action_space.traj_to_action(
                traj_history_xyz=ego_history_xyz, traj_history_rot=ego_history_rot,
                traj_future_xyz=ego_future_xyz, traj_future_rot=ego_future_rot,
            ).reshape(b, n_act, 2).to(device=device, dtype=torch.float32)

        gen = None
        x0_teacher = None
        if self._cd_seed is not None:
            gen = torch.Generator(device=device).manual_seed(self._cd_seed)
        if self.teacher_source == "cached_full":
            if teacher_trajectory_states is None:
                raise ValueError(
                    "teacher_source='cached_full' requires teacher_trajectory_states in "
                    "every batch; set data.train_dataset.teacher_trajectory_cache_root"
                )
            with torch.no_grad(), _Phase.t("teacher_cache"):
                states = teacher_trajectory_states.to(device=device, dtype=torch.float32)
                _, noise_index, x_hi_fp32, x_lo_fp32, tau_lo, tau_hi = (
                    sample_cached_teacher_transition(
                        states,
                        self.m_rungs,
                        generator=gen,
                    )
                )
                # Same rollout as x_hi/x_lo, so (eps, x0_teacher) stays a matched pair.
                x0_teacher = cached_teacher_endpoint(states, noise_index).float()
                if x_hi_fp32.shape[1:] != (n_act, 2):
                    raise ValueError(
                        f"cached action shape must be [B,{n_act},2], got "
                        f"{tuple(x_hi_fp32.shape)}"
                    )
                v_teacher = transition_velocity(
                    x_hi_fp32, x_lo_fp32, tau_lo, tau_hi
                )
                x_hi = x_hi_fp32
                x_lo = x_lo_fp32
            if (
                os.environ.get("CD_SELFTEST") == "1"
                and not getattr(self, "_cd_selftested", False)
            ):
                self._cd_selftested = True
                recovered = teacher_step(
                    x_hi_fp32, v_teacher, tau_hi - tau_lo
                )
                err = float((recovered - x_lo_fp32).abs().max())
                print(
                    f"[cd-selftest] cached full-teacher Euler endpoint error={err:.3e} "
                    "(must be ~0)",
                    flush=True,
                )
                print("DONE_CD_SELFTEST", flush=True)
        else:
            if (
                os.environ.get("CD_SELFTEST") == "1"
                and not getattr(self, "_cd_selftested", False)
            ):
                self._cd_selftested = True
                self._cd_selftest(x0, cache, conditioning, dtype, device)
            _, tau_lo, tau_hi = sample_rungs(b, self.m_rungs, device, generator=gen)
            eps = torch.randn(x0.shape, device=device, dtype=torch.float32, generator=gen)
            x_hi = interpolate(x0, eps, tau_hi)
            teacher = self._ensure_teacher_expert(device, dtype)
            # One online teacher NFE and the backward-in-tau Euler step it drives.
            with torch.no_grad(), _Phase.t("expert_teacher"):
                v_teacher = self._velocity(
                    teacher, x_hi, tau_hi, cache, conditioning, dtype
                )
                x_lo = teacher_step(
                    x_hi.float(), v_teacher.float(), tau_hi - tau_lo
                )

        # 4) the EMA target, theta^- = stopgrad(EMA(theta)). Skipped entirely when every
        #    sample is on the anchor rung, where its coefficient (1 - lam) is exactly zero.
        #
        # ⚠️ The EMA weights are SWAPPED INTO the live module for this forward and swapped
        # back immediately. Forwarding `self` directly here would silently make the target
        # stopgrad(theta) -- the mu=0 self-target -- which is a DIFFERENT objective with a
        # different fixed point, and nothing about the loss curve would look wrong.
        # try/finally because an exception between swap_in and swap_out would leave EMA
        # weights in the model and corrupt training from that step on.
        # ⚠️ `cd_weight == 0` (the endpoint-only arm) must skip this ENTIRELY, not just
        # multiply it by zero: with no `callbacks.ema` block there is no `_ema_ref`, and the
        # forward below would then silently build the mu=0 self-target -- a different
        # objective, at the cost of a second expert forward, for a term weighted zero.
        v_target = None
        if self.cd_weight != 0.0 and bool(needs_target(tau_lo).any()):
            ema = getattr(self, "_ema_ref", None)
            with torch.no_grad(), _Phase.t("expert_ema_target"):
                if ema is not None:
                    ema.swap_in(self)
                try:
                    v_target = self._velocity(
                        self, x_lo, tau_lo, cache, conditioning, dtype
                    )
                finally:
                    if ema is not None:
                        ema.swap_out(self)

        # 5) the online forward -- the only one that carries gradient.
        with _Phase.t("expert_online_fwd"):
            v_online = self._velocity(
                self, x_hi, tau_hi, cache, conditioning, dtype
            )

        if self.cd_weight == 0.0:
            loss_cd = v_online.new_zeros(())
            per_sample = v_online.new_zeros(b)
        else:
            loss_cd, per_sample = cd_loss(
                v_online, v_teacher, v_target, tau_lo, tau_hi,
                metric=self.cd_metric, huber_c=self.cd_huber_c,
                normalizer=self.cd_normalizer,
            )
        # THE TWO ENDPOINT TARGETS PULL IN OPPOSITE DIRECTIONS ON DIVERSITY, which is the
        # whole point of blending them. `teacher` is the cached endpoint of THIS noise draw,
        # so the K draws keep K distinct targets -- alone it overshot the teacher's spread
        # 2.8x (diversity 2.51 vs 0.89) and cost 0.36 of ade. `gt` is the dataset action,
        # shared by all K draws, so its minimiser at tau=1 is the conditional mean -- alone
        # it collapses diversity. Measured brackets: teacher-only 2.51, cd+0.1*gt 1.34,
        # teacher 0.89. A small gt weight on top of the endpoint term should land between.
        zero = v_online.new_zeros(())
        loss_x0_teacher, per_x0 = zero, v_online.new_zeros(b)
        if self.x0_teacher_weight > 0.0:
            if x0_teacher is None:
                raise RuntimeError(
                    "x0_teacher_weight > 0 but no cached endpoint was lifted this step"
                )
            loss_x0_teacher, per_x0 = x0_reconstruction_loss(
                x_hi, tau_hi, v_online, x0_teacher, normalizer=self.x0_gt_normalizer,
            )
        loss_x0_gt, per_gt = zero, v_online.new_zeros(b)
        if self.x0_gt_weight > 0.0:
            loss_x0_gt, per_gt = x0_reconstruction_loss(
                x_hi, tau_hi, v_online, x0, normalizer=self.x0_gt_normalizer,
            )
        if self.x0_teacher_weight == 0.0:
            per_x0 = per_gt          # bands always describe the dominant endpoint term
        total = (self.cd_weight * loss_cd
                 + self.x0_teacher_weight * loss_x0_teacher
                 + self.x0_gt_weight * loss_x0_gt)

        attached = {"cd": loss_cd, "x0_gt": loss_x0_gt}
        self.last_loss_terms = attached if self.keep_loss_terms else None

        t = tau_hi.flatten()
        selectors = (("anchor", t <= 1.5 / self.m_rungs),
                     ("mid", (t > 1.5 / self.m_rungs) & (t < 0.8)),
                     ("noise", t >= 0.8))

        def split(per: torch.Tensor) -> dict[str, torch.Tensor | None]:
            return {name: (per[sel].mean().detach() if bool(sel.any()) else None)
                    for name, sel in selectors}

        # The endpoint arm optimises x0, so ITS bands are the instrument there: `noise`
        # (tau >= 0.8) is what a 1-NFE sampler evaluates and is the number to watch.
        cd_bands, x0_bands = split(per_sample), split(per_x0)

        _Phase.report()
        return CDVLAOutput(
            loss=total,
            cd_loss=loss_cd.detach(),
            x0_gt_loss=loss_x0_gt.detach(),
            x0_teacher_loss=loss_x0_teacher.detach(),
            cd_loss_anchor=cd_bands["anchor"],
            cd_loss_mid=cd_bands["mid"],
            cd_loss_noise=cd_bands["noise"],
            x0_loss_anchor=x0_bands["anchor"],
            x0_loss_mid=x0_bands["mid"],
            x0_loss_noise=x0_bands["noise"],
        )
