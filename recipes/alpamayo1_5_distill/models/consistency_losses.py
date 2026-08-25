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

r"""Consistency distillation of the action expert's flow-matching sampler.

Cuts the sampler from 10 Euler steps to 1--2 by distilling the frozen post-trained expert
into a consistency model.  Denoising is 51-77% of end-to-end latency and 180.7 ms of the
teacher's 279.4 ms at 2 cameras, so this is the largest single latency term in the system.

⚠️⚠️ **TIME RUNS BACKWARDS BETWEEN THE TWO CONVENTIONS IN PLAY.** ⚠️⚠️

This module is written in the **consistency-model convention**, ``tau``, because that is how
the objective is specified.  The rest of the repo uses the opposite one, ``s``:

    ================  ==============================  ==============================
                      tau  (this module, CM/spec)     s  (flow_matching.py, the repo)
    ================  ==============================  ==============================
    interpolant       x = (1-tau)*x0 + tau*eps        x = s*x0 + (1-s)*noise
    value 0 means     DATA                            NOISE
    value 1 means     NOISE                           DATA
    velocity          v_phi   = dx/dtau = eps - x0    v_repo = dx/ds = x0 - noise
    ================  ==============================  ==============================

    => s = 1 - tau      and      v_phi = -v_repo

Both halves of that are load-bearing, and each is a *separate* way to get a run that trains
cleanly and learns the wrong map:

1. ``action_in_proj(x, t)`` MUST be fed ``s = 1 - tau``.  Its timestep Fourier encoder
   (``FourierEncoderV2``, log-spaced frequencies to 100 over the unit interval) was trained
   on the repo's ``s``; handing it ``tau`` mirrors the conditioning.  Use :func:`tau_to_s`.
2. The backward solver step is ``x - dtau*v_phi`` = ``x + dtau*v_repo``.  Written with the
   head's native output as ``x - dtau*v_repo`` it walks AWAY from the data.  Use
   :func:`teacher_step`, which takes ``v_repo`` and gets the sign right once.

**The one place the sign does NOT matter**, and it is worth knowing so nobody "fixes" it:
the residual is *linear* in the three velocities and then squared, so as long as all three
are in the SAME convention, ``v_repo`` throughout gives a bit-identical loss to ``v_phi``
throughout.  Everything here therefore takes the head's native ``v_repo`` and never negates.
Mixing the two conventions is what breaks.

The objective (uniform grid ``0 = tau_0 < ... < tau_M = 1``, rung ``n ~ U{0..M-1}``)::

    x_hi   = (1 - tau_hi)*x0 + tau_hi*eps                    tau_hi = tau_{n+1}
    x_lo   = x_hi + (tau_hi - tau_lo) * v_teacher(x_hi)      ONE teacher NFE, no grad
    L      = d( f_theta(x_hi, tau_hi),  f_theta_ema(x_lo, tau_lo) )

with the consistency parameterisation ``f(x, tau) = x - tau*F(x, tau) = x + tau*v_repo``,
which satisfies the boundary ``f(x, 0) = x`` identically, for any weights, for free.

**Stable evaluation of the f-space objective.**  Substituting the parameterisation, the
``x`` terms cancel exactly.  With ``sigma = tau_hi``, ``delta = tau_hi - tau_lo`` and
``lam = delta/sigma``::

    f(x_hi, tau_hi) - f(x_lo, tau_lo)
        = sigma * [ v_online - ( lam*v_teacher + (1 - lam)*v_target ) ]

The loss computes the right-hand side in fp32. This is algebraically the same residual as
the literal difference of the two consistency outputs, but avoids subtracting two O(1)
quantities in bf16. The ``sigma`` factor is load-bearing: omitting it is not merely a
numerical rewrite, but silently changes a uniform f-space loss into one weighted by
``1 / sigma**2``. On this 10-rung grid that overweights the data-end rung by 100x relative
to the noise-end rung.

At the anchor rung the residual simplifies to
``delta * (v_online - v_teacher)``. Forming that product directly preserves the small signal;
constructing and differencing the two bf16 consistency outputs can round it away.
``test_velocity_form_beats_literal_in_bf16`` demonstrates the precision benefit while
``test_loss_matches_uniform_f_space_mse`` pins the required weighting.

``lam`` also makes the anchor automatic: at ``n = 0`` we have ``tau_lo = 0``, so
``lam = 1`` and the target's coefficient is exactly zero -- the caller can skip that forward
entirely (:func:`needs_target`), and the rung reduces to pure velocity matching against the
teacher, which is what anchors the whole bootstrap chain.
"""

from __future__ import annotations

import torch

#: Default rungs. 10 makes one rung exactly one step of the deployment sampler
#: (``flow_matching._euler`` at ``inference_step=10``), so the fixed point of the objective
#: is the teacher's own 10-step output -- the min_ade 0.7185 / diversity 1.588 object that
#: Gate C measured, rather than the true ODE solution which was never the reference.
DEFAULT_M = 10


def tau_to_s(tau: torch.Tensor | float) -> torch.Tensor | float:
    """CM time -> repo time. ``s = 1 - tau``.

    ⚠️ Every call into ``action_in_proj`` must go through this. See the module docstring.
    """
    return 1.0 - tau


def uniform_tau_grid(m: int = DEFAULT_M, device=None, dtype=torch.float32) -> torch.Tensor:
    """``[0, 1/m, ..., 1]``, length ``m + 1``. Index n is ``tau_n``; 0 is data, 1 is noise."""
    if m < 1:
        raise ValueError(f"grid needs at least one rung, got m={m}")
    return torch.linspace(0.0, 1.0, m + 1, device=device, dtype=dtype)


def sample_rungs(batch: int, m: int, device, generator: torch.Generator | None = None):
    """Draw ``n ~ U{0..m-1}`` per sample and return ``(n, tau_lo, tau_hi)``.

    ``tau_lo``/``tau_hi`` come back shaped ``[B, 1, 1]``.

    ⚠️ THE SHAPE IS NOT COSMETIC. ``PerWaypointActionInProjV2.forward`` does
    ``timesteps[..., -1]`` then ``.repeat(1, T, 1)``; a ``[B]`` timestep collapses to a 0-dim
    scalar, the timestep features come out batch-1, and that broadcasts SILENTLY at B=1 and
    only dies at B>1 ("Expected size 8 but got size 1"). Every bs=1 smoke in this tree has
    passed over that bug at least once -- see ``kd_model.field_at``.
    """
    n = torch.randint(0, m, (batch,), device=device, generator=generator)
    tau_lo = (n.to(torch.float32) / m).view(batch, 1, 1)
    tau_hi = ((n + 1).to(torch.float32) / m).view(batch, 1, 1)
    return n, tau_lo, tau_hi


def needs_target(tau_lo: torch.Tensor) -> torch.Tensor:
    """``[B]`` bool: does this sample's target term have a nonzero coefficient?

    False exactly on the anchor rung (``tau_lo == 0``), where ``lam == 1``. Skipping the
    target forward there is free -- it saves one expert forward on 1/M of samples and the
    value it would return is multiplied by zero anyway.
    """
    return (tau_lo > 0).flatten()


def interpolate(x0: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """``x_tau = (1 - tau)*x0 + tau*eps``. tau=0 is data, tau=1 is noise."""
    return (1.0 - tau) * x0 + tau * eps


def teacher_step(x: torch.Tensor, v_repo: torch.Tensor, dtau: torch.Tensor) -> torch.Tensor:
    """One backward-in-tau solver step: ``x_lo = x_hi - dtau*v_phi``.

    ⚠️ Takes the head's NATIVE ``v_repo`` and applies ``+dtau*v_repo``, because
    ``v_phi = -v_repo``. Writing ``x - dtau*v_repo`` here integrates away from the data and
    is the single easiest way to get a plausible-looking, entirely wrong run.
    """
    return x + dtau * v_repo


def consistency_fn(x: torch.Tensor, tau: torch.Tensor, v_repo: torch.Tensor) -> torch.Tensor:
    """``f(x, tau) = x - tau*F(x, tau) = x + tau*v_repo``, an estimate of ``x0``.

    Exact for the linear interpolant: ``x0 = x_s + (1-s)*v_repo`` identically, and
    ``1 - s == tau``. So ``f(x, 0) = x`` holds for ANY weights -- the boundary condition is
    structural, not learned, and costs nothing.

    ⚠️ Used by the SAMPLER and by the self-tests. It is deliberately NOT used to build the
    training loss: see the module docstring on why the loss is computed in velocity space.
    """
    return x + tau * v_repo


def blend_weight(tau_lo: torch.Tensor, tau_hi: torch.Tensor) -> torch.Tensor:
    """``lam = (tau_hi - tau_lo) / tau_hi`` -- the teacher's share of the target.

    On a uniform grid ``tau_hi = (n+1)/M`` and ``delta = 1/M``, so ``lam = 1/(n+1)``:
    exactly 1 on the anchor rung (n=0, where the target term vanishes) and 1/M at the noise
    end (n=M-1). So the teacher dominates near the data and the bootstrapped target
    dominates near the noise -- which is the direction information has to travel.
    """
    return (tau_hi - tau_lo) / tau_hi.clamp_min(torch.finfo(tau_hi.dtype).tiny)


def cd_target_velocity(
    v_teacher: torch.Tensor, v_target: torch.Tensor | None,
    tau_lo: torch.Tensor, tau_hi: torch.Tensor,
) -> torch.Tensor:
    """``lam*v_teacher + (1 - lam)*v_target`` -- what the online velocity must match.

    A convex blend (the weights sum to 1), so the student's velocity is being pushed toward
    the *average* velocity over the remaining path rather than the instantaneous one. That is
    the same object MeanFlow learns; this is its finite-difference form with a teacher.

    ``v_target`` may be ``None`` only if every sample is on the anchor rung.
    """
    lam = blend_weight(tau_lo, tau_hi)
    if v_target is None:
        if bool(needs_target(tau_lo).any()):
            raise ValueError("v_target is None but some samples have tau_lo > 0")
        return v_teacher
    return lam * v_teacher + (1.0 - lam) * v_target


def cd_loss(
    v_online: torch.Tensor,
    v_teacher: torch.Tensor,
    v_target: torch.Tensor | None,
    tau_lo: torch.Tensor,
    tau_hi: torch.Tensor,
    *,
    metric: str = "mse",
    huber_c: float | None = None,
    normalizer: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The consistency-distillation loss, uniformly weighted in consistency-output space.

    Args:
        v_online:  ``v_repo`` from the TRAINABLE expert at ``(x_hi, tau_hi)``, WITH grad.
        v_teacher: ``v_repo`` from the FROZEN teacher at ``(x_hi, tau_hi)``, no grad.
                   This is the one teacher NFE per training iteration.
        v_target:  ``v_repo`` from the EMA target at ``(x_lo, tau_lo)``, no grad. May be
                   ``None`` when every sample is on the anchor rung.
        tau_lo, tau_hi: ``[B, 1, 1]``.
        metric: ``mse`` or ``pseudo_huber``. iCT found pseudo-Huber materially better than
            MSE for consistency training; it is offered but not the default because its
            scale constant is calibrated for pixels in [-1, 1] and this action space is
            normalised (accel, curvature), so ``huber_c`` needs measuring here before it
            means anything.
        huber_c: pseudo-Huber scale. Defaults to iCT's ``0.00054*sqrt(d)`` heuristic, which
            is a STARTING POINT for this data, not a calibrated value.
        normalizer: fixed scalar to divide by, so runs are comparable. ⚠️ Must be a constant,
            never a per-sample norm -- a per-sample denominator that can approach zero turns
            a few easy samples into an unbounded gradient.

    Returns:
        ``(loss, per_sample)`` -- scalar, and ``[B]`` detached for per-rung logging.

    ⚠️ fp32 throughout. The inputs arrive from bf16 forwards, but the subtraction and the
    reduction happen in fp32; see the module docstring on cancellation.
    """
    v_online = v_online.float()
    tau_hi = tau_hi.float()
    tgt = cd_target_velocity(
        v_teacher.float(), None if v_target is None else v_target.float(),
        tau_lo.float(), tau_hi,
    ).detach()
    # Evaluate f_online - f_target through the algebraically equivalent velocity residual.
    # Keeping tau_hi here preserves the standard uniform consistency-output objective.
    # Dropping it implicitly multiplies the f-space MSE by 1/tau_hi**2.
    residual = tau_hi * (v_online - tgt)

    if metric == "mse":
        per_sample = residual.pow(2).flatten(1).mean(1)
    elif metric == "pseudo_huber":
        d = residual[0].numel()
        c = huber_c if huber_c is not None else 0.00054 * (d ** 0.5)
        per_sample = (residual.pow(2).flatten(1).sum(1) + c * c).sqrt() - c
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'mse' or 'pseudo_huber'")

    return per_sample.mean() / normalizer, per_sample.detach()
