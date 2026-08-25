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

r"""EMA of the trainable action expert -- the consistency objective's target network.

Nothing in this tree has ever kept a shadow copy of weights, so this is written from
scratch. Three properties are load-bearing and each one is a silent failure if missed.

**1. THE MASTER MUST BE fp32.**  Measured, not assumed::

    dtype  mu       movement after 2000 steps      exact
    bf16   0.999    0.000e+00                      1.730e-02   <- FROZEN
    bf16   0.9999   0.000e+00                      3.626e-03   <- FROZEN
    fp16   0.999    0.000e+00                      1.730e-02   <- FROZEN
    fp32   0.999    1.727e-02                      1.730e-02   ok

At mu=0.999 the increment is ``1e-3 * |theta - ema|``, while one bf16 ULP on an O(1) value
is ~8e-3. The update rounds to nothing and the EMA never moves *at all*. The failure is
invisible: the loss curve looks healthy, no warning fires, and the target silently stays
pinned at initialisation -- i.e. at the frozen teacher -- which quietly converts the run
into the no-bootstrap variant whose fixed point is a 2-step estimator. ``EMACallback``
asserts liveness on the first few updates rather than trusting this comment.

**2. UPDATE ONCE PER OPTIMIZER STEP, NOT PER MICRO-BATCH.**  ``on_step_end`` is the right
hook. Both expert arms run ``gradient_accumulation_steps: 12``, so a per-micro-batch update
would apply the decay 12x too often and shorten the effective horizon 12-fold.
``KaVaTrainer._should_probe`` documents exactly this distinction for the gradient probe.

**3. THE SAVED CHECKPOINT MUST CONTAIN THE EMA WEIGHTS.**  Saving is 100% stock
``Trainer._save_checkpoint`` -- nothing in this tree overrides ``_save``/``save_model`` --
and the eval path reads a trained expert back through the *teacher* slot
(``++model.teacher_checkpoint_path=<ckpt>``), where ``_load_teacher_non_vlm`` looks for
canonical ``expert.*`` names inside a sharded ``model.safetensors`` + index. A side-car
``ema.safetensors`` would be invisible to both that and ``FrozenExpert._load``. So
``on_save`` swaps the EMA into the live parameters, lets HF serialise, and swaps back --
which keeps the existing eval command working with no changes at all.

Wiring, via the ``callbacks:`` block that ``train_hf.py`` already instantiates and that no
config in this tree has ever used::

    callbacks:
      ema:
        _target_: alpamayo1_5_distill.models.ema.EMACallback
        decay: 0.999
        warmup_steps: 0
"""

from __future__ import annotations

import logging

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

#: Parameter-name prefixes the EMA tracks. Only the action expert and its projections are
#: trainable in this arm; the 8B VLM is frozen, and shadowing it would waste ~16 GB.
EMA_PREFIXES = ("expert.", "action_in_proj.", "action_out_proj.")


class ExpertEMA:
    """fp32 shadow of the trainable parameters, with in-place swap into the live model.

    Held by the callback, NOT registered on the model: a registered copy would enter the
    optimizer, the ZeRO partition and every saved checkpoint -- the same reason
    ``kd_model`` keeps its frozen expert and teacher in plain lists. The cost of that
    choice is that nothing moves it for us, so ``update`` places the shadow on first use.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999,
                 prefixes: tuple[str, ...] = EMA_PREFIXES) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self._names = [
            n for n, p in model.named_parameters()
            if p.requires_grad and n.startswith(prefixes)
        ]
        if not self._names:
            raise RuntimeError(
                "EMA found no trainable parameters under "
                f"{prefixes}. Is the expert frozen, or renamed?"
            )
        # ⚠️ fp32, always -- see the module docstring. `.detach().clone()` and NOT
        # `torch.zeros_like`: the shadow must start AT the weights, so that a target forward
        # before the first update is the teacher rather than noise.
        self.shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in set(self._names):
                    self.shadow[n] = p.detach().to(torch.float32).clone()
        self._backup: dict[str, torch.Tensor] = {}
        self.n_updates = 0
        n_par = sum(v.numel() for v in self.shadow.values())
        logger.warning("[ema] tracking %d tensors, %.2f B params, fp32 master (%.1f GB), decay %.4f",
                       len(self.shadow), n_par / 1e9, n_par * 4 / 1e9, self.decay)
        print(f"[ema] {len(self.shadow)} tensors, {n_par / 1e9:.2f} B params, "
              f"fp32 master {n_par * 4 / 1e9:.1f} GB, decay {self.decay}", flush=True)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: float | None = None) -> None:
        """``shadow = d*shadow + (1-d)*theta``, in fp32."""
        d = self.decay if decay is None else decay
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is None:
                continue
            if s.device != p.device:  # nothing else will move it; see the class docstring
                s = s.to(p.device)
                self.shadow[n] = s
            s.mul_(d).add_(p.detach().to(torch.float32), alpha=1.0 - d)
        self.n_updates += 1

    @torch.no_grad()
    def distance_from(self, other: dict[str, torch.Tensor]) -> float:
        """Global L2 between the shadow and a reference snapshot. The liveness probe."""
        tot = 0.0
        for n, s in self.shadow.items():
            ref = other.get(n)
            if ref is not None:
                tot += float((s.float() - ref.to(s.device).float()).pow(2).sum())
        return tot ** 0.5

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {n: s.detach().to("cpu").clone() for n, s in self.shadow.items()}

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> None:
        """Put EMA weights into the live model, stashing the originals."""
        if self._backup:
            raise RuntimeError("swap_in called twice without swap_out -- weights would be lost")
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is None:
                continue
            self._backup[n] = p.detach().clone()
            p.copy_(s.to(dtype=p.dtype, device=p.device))

    @torch.no_grad()
    def swap_out(self, model: torch.nn.Module) -> None:
        """Restore the training weights saved by :meth:`swap_in`."""
        if not self._backup:
            return
        for n, p in model.named_parameters():
            b = self._backup.pop(n, None)
            if b is not None:
                p.copy_(b)
        self._backup.clear()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "n_updates": self.n_updates,
                "shadow": {n: s.cpu() for n, s in self.shadow.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        self.n_updates = int(sd["n_updates"])
        for n, s in sd["shadow"].items():
            if n in self.shadow:
                self.shadow[n].copy_(s.to(self.shadow[n].device))


class EMACallback(TrainerCallback):
    """Maintains an :class:`ExpertEMA` and makes it the thing that gets saved.

    ``callbacks:`` is instantiated by ``train_hf.py:59-62`` and has never been used by any
    config in this tree, so this needs a config block and no entry-point edit.
    """

    def __init__(self, decay: float = 0.999, warmup_steps: int = 0,
                 save_ema_weights: bool = True, liveness_check_at: int = 20) -> None:
        self.decay = decay
        self.warmup_steps = int(warmup_steps)
        self.save_ema_weights = bool(save_ema_weights)
        self.liveness_check_at = int(liveness_check_at)
        self.ema: ExpertEMA | None = None
        self._init_snapshot: dict[str, torch.Tensor] | None = None
        self._liveness_ok = False

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _unwrap(model):
        return getattr(model, "module", model)

    def _ensure(self, model) -> ExpertEMA:
        if self.ema is None:
            self.ema = ExpertEMA(self._unwrap(model), decay=self.decay)
            self._init_snapshot = self.ema.snapshot()
        return self.ema

    # -- hooks -----------------------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, **kw):
        ema = self._ensure(model)
        # The model's forward needs the EMA to build theta^-. Handing it over here rather
        # than having the model construct its own keeps ONE shadow -- two would drift apart
        # and the saved checkpoint would not be the network that produced the targets.
        base = self._unwrap(model)
        base._ema_ref = ema
        print(f"[ema] bound to {type(base).__name__}._ema_ref for the target forward",
              flush=True)

    def on_step_end(self, args, state, control, model=None, **kw):
        """Once per OPTIMIZER step -- not per micro-batch. See the module docstring."""
        ema = self._ensure(model)
        step = int(getattr(state, "global_step", 0))
        # A short warmup lets the model move before the average starts tracking it; during it
        # the shadow follows the weights exactly, so the target is the live net.
        decay = 0.0 if step <= self.warmup_steps else None
        ema.update(self._unwrap(model), decay=decay)

        if not self._liveness_ok and step >= self.liveness_check_at:
            moved = ema.distance_from(self._init_snapshot or {})
            self._liveness_ok = True
            print(f"[ema] liveness @ step {step}: ||EMA - init|| = {moved:.6e}", flush=True)
            if moved == 0.0:
                raise RuntimeError(
                    f"EMA has not moved after {ema.n_updates} updates (||EMA - init|| == 0). "
                    "The classic cause is a non-fp32 master: at decay=0.999 the increment is "
                    "below one bf16 ULP and rounds away, leaving the target pinned at the "
                    "teacher for the whole run. Check ExpertEMA.shadow dtype."
                )

    # ⚠️ There is deliberately NO ``on_save`` here. In transformers 4.57.1
    # ``_maybe_log_save_evaluate`` runs ``self._save_checkpoint(model, trial)`` and only THEN
    # ``callback_handler.on_save(...)`` (trainer.py:3227-3229), so a swap performed in
    # ``on_save`` lands after the bytes are already written -- it would silently save the
    # training weights and then leave the EMA in the live model. The swap therefore lives in
    # ``KaVaTrainer._save_checkpoint``, wrapped in try/finally so an exception mid-write
    # cannot leave EMA weights in the model and corrupt training.
