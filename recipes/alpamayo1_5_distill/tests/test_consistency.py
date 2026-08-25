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

"""Consistency-distillation unit tests -- the ones that catch a silently-wrong run.

Everything here is CPU-only and takes a second: the value is that each test corresponds to a
failure mode that would otherwise produce a healthy-looking loss curve and a wrong model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from alpamayo1_5_distill.models.consistency_losses import (
    blend_weight,
    cd_loss,
    cd_target_velocity,
    consistency_fn,
    interpolate,
    needs_target,
    sample_rungs,
    tau_to_s,
    teacher_step,
    uniform_tau_grid,
)
from alpamayo1_5_distill.models.ema import ExpertEMA

B, T, C = 8, 64, 2


def _tau(v: float, b: int = B) -> torch.Tensor:
    return torch.full((b, 1, 1), v, dtype=torch.float32)


# ---------------------------------------------------------------- convention


def test_tau_zero_is_data_and_tau_one_is_noise():
    """The whole objective hinges on this orientation; assert it rather than trust a comment."""
    x0, eps = torch.randn(B, T, C), torch.randn(B, T, C)
    torch.testing.assert_close(interpolate(x0, eps, _tau(0.0)), x0)
    torch.testing.assert_close(interpolate(x0, eps, _tau(1.0)), eps)


def test_tau_to_s_matches_the_repo_interpolant():
    """``x_tau`` under the CM convention must equal ``x_s`` under flow_matching's, at s=1-tau.

    This is the mapping the whole module rests on. flow_matching.py:156 is
    ``noisy_x = t*x + (1-t)*noise``.
    """
    x0, eps = torch.randn(B, T, C), torch.randn(B, T, C)
    for tv in (0.0, 0.1, 0.37, 0.9, 1.0):
        tau = _tau(tv)
        s = tau_to_s(tau)
        repo = s * x0 + (1.0 - s) * eps
        torch.testing.assert_close(interpolate(x0, eps, tau), repo)


def test_teacher_step_moves_toward_data_not_away():
    """``x - dtau*v_phi == x + dtau*v_repo``. The sign trap, pinned.

    With the optimal field at the noise end, one full step must land on x0. Applying the
    head's output with the wrong sign lands on ``2*eps - x0`` instead.
    """
    x0, eps = torch.randn(B, T, C), torch.randn(B, T, C)
    v_repo = x0 - eps                       # what action_out_proj emits
    landed = teacher_step(eps, v_repo, _tau(1.0))
    torch.testing.assert_close(landed, x0)
    wrong = eps - 1.0 * v_repo
    torch.testing.assert_close(wrong, 2 * eps - x0)
    assert (landed - wrong).abs().mean() > 0.1


# ---------------------------------------------------------------- parameterisation


def test_boundary_is_exact_for_any_weights():
    """``f(x, 0) = x`` must hold identically -- it is structural, not learned."""
    x = torch.randn(B, T, C)
    for _ in range(5):
        v = torch.randn(B, T, C) * 100.0     # arbitrary, deliberately huge
        torch.testing.assert_close(consistency_fn(x, _tau(0.0), v), x)


def test_consistency_fn_recovers_x0_under_the_optimal_field():
    """``f(x_tau, tau) = x0`` exactly, at every tau, when v is the true velocity."""
    x0, eps = torch.randn(B, T, C), torch.randn(B, T, C)
    v_repo = x0 - eps
    for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
        tau = _tau(tv)
        torch.testing.assert_close(consistency_fn(interpolate(x0, eps, tau), tau, v_repo), x0)


# ---------------------------------------------------------------- grid / blend


def test_grid_and_rungs_have_the_right_shapes_and_range():
    g = uniform_tau_grid(10)
    assert g.shape == (11,) and g[0] == 0.0 and g[-1] == 1.0
    n, lo, hi = sample_rungs(B, 10, torch.device("cpu"))
    # [B,1,1] is load-bearing: a [B] timestep broadcasts silently at B=1 in action_in_proj.
    assert lo.shape == (B, 1, 1) and hi.shape == (B, 1, 1)
    assert torch.all(hi > lo) and torch.all(lo >= 0.0) and torch.all(hi <= 1.0)
    assert int(n.min()) >= 0 and int(n.max()) <= 9


def test_anchor_rung_is_teacher_pure_and_needs_no_target():
    """At tau_lo=0 the blend weight is exactly 1, so the target term drops out."""
    lo, hi = _tau(0.0), _tau(0.1)
    torch.testing.assert_close(blend_weight(lo, hi), torch.ones_like(lo))
    assert not needs_target(lo).any()
    v_t = torch.randn(B, T, C)
    torch.testing.assert_close(cd_target_velocity(v_t, None, lo, hi), v_t)


def test_blend_weight_is_one_over_n_plus_one():
    """lam = delta/tau_hi = 1/(n+1): 1 at the anchor, 1/M at the noise end.

    So the teacher dominates the target near the data and the bootstrapped EMA dominates
    near the noise -- the direction information has to travel for the chain to converge.
    """
    m = 10
    for n in range(m):
        lo, hi = _tau(n / m), _tau((n + 1) / m)
        lam = blend_weight(lo, hi)
        torch.testing.assert_close(lam, torch.full_like(lam, 1.0 / (n + 1)), rtol=1e-5, atol=1e-6)
        assert 0.0 < float(lam[0]) <= 1.0
    assert float(blend_weight(_tau(0.0), _tau(1 / m))[0]) == 1.0
    torch.testing.assert_close(
        blend_weight(_tau((m - 1) / m), _tau(1.0)), torch.full((B, 1, 1), 1.0 / m))


# ---------------------------------------------------------------- the loss


def test_loss_is_zero_when_the_student_is_the_blend():
    v_t, v_g = torch.randn(B, T, C), torch.randn(B, T, C)
    lo, hi = _tau(0.3), _tau(0.4)
    tgt = cd_target_velocity(v_t, v_g, lo, hi)
    loss, per = cd_loss(tgt.clone(), v_t, v_g, lo, hi)
    assert float(loss) < 1e-12 and per.shape == (B,)


def test_loss_matches_uniform_f_space_mse():
    """The stable velocity algebra must not change the standard CD weighting.

    ``f_hi - f_lo = tau_hi * (v_online - v_target_blend)``. Omitting ``tau_hi``
    silently weights the f-space objective by ``1/tau_hi**2``.
    """
    v_o, v_t, v_g = (torch.randn(B, T, C) for _ in range(3))
    lo = torch.arange(B, dtype=torch.float32).view(B, 1, 1) / 10.0
    hi = lo + 0.1
    target = cd_target_velocity(v_t, v_g, lo, hi)
    expected = (hi * (v_o - target)).pow(2).flatten(1).mean(1)

    loss, per = cd_loss(v_o, v_t, v_g, lo, hi)

    torch.testing.assert_close(per, expected)
    torch.testing.assert_close(loss, expected.mean())


def test_uniform_f_space_loss_keeps_the_tau_squared_factor():
    """Equal velocity errors at tau=.1 and tau=1 have f-space MSE ratio .01."""
    v_o = torch.ones(2, T, C)
    zeros = torch.zeros_like(v_o)
    lo = torch.tensor([0.0, 0.9]).view(2, 1, 1)
    hi = torch.tensor([0.1, 1.0]).view(2, 1, 1)
    _, per = cd_loss(v_o, zeros, zeros, lo, hi)
    torch.testing.assert_close(per, torch.tensor([0.01, 1.0]))


def test_loss_gradient_reaches_only_the_online_velocity():
    v_o = torch.randn(B, T, C, requires_grad=True)
    v_t = torch.randn(B, T, C, requires_grad=True)
    v_g = torch.randn(B, T, C, requires_grad=True)
    loss, _ = cd_loss(v_o, v_t, v_g, _tau(0.3), _tau(0.4))
    loss.backward()
    assert v_o.grad is not None and v_o.grad.abs().sum() > 0
    # The target side is stop-grad by construction; a leak here would let the student
    # trivially minimise the loss by dragging the target toward itself.
    assert v_t.grad is None and v_g.grad is None


def test_sign_convention_cancels_when_applied_consistently():
    """v_repo throughout and v_phi throughout must give the SAME loss.

    Documents why the module never negates: the residual is linear in the velocities and
    then squared. Mixing conventions is the bug; using either one consistently is not.
    """
    v_o, v_t, v_g = (torch.randn(B, T, C) for _ in range(3))
    lo, hi = _tau(0.2), _tau(0.3)
    a, _ = cd_loss(v_o, v_t, v_g, lo, hi)
    b, _ = cd_loss(-v_o, -v_t, -v_g, lo, hi)
    torch.testing.assert_close(a, b)


def test_loss_at_init_falls_as_one_over_M_squared():
    """Gate A's sharp test, on a field with known curvature.

    At theta == phi the residual is the teacher's path curvature over one rung, which is
    O(delta); the loss is therefore O(delta^2) = O(1/M^2). Quadrupling M must drop it 16x.
    A wrong Euler step or a mirrored interpolant breaks this scaling, which is what makes
    it a useful test rather than a tautology.
    """
    torch.manual_seed(0)
    x0, eps = torch.randn(B, T, C), torch.randn(B, T, C)

    # A deliberately CURVED field: v depends on tau, so consecutive rungs disagree.
    def v_field(x, tau):
        return (x0 - eps) + 0.5 * torch.sin(3.0 * tau) * x

    losses = {}
    for m in (10, 20, 40):
        d = 1.0 / m
        tot = 0.0
        for n in range(m):
            lo, hi = _tau(n / m), _tau((n + 1) / m)
            x_hi = interpolate(x0, eps, hi)
            v_teach = v_field(x_hi, hi)
            x_lo = teacher_step(x_hi, v_teach, hi - lo)
            v_tgt = v_field(x_lo, lo)
            # theta == phi: the online net is the same field, evaluated at the same point.
            v_on = v_teach
            # f-space residual, which is what the 1/M^2 law is about.
            resid = consistency_fn(x_hi, hi, v_on) - consistency_fn(x_lo, lo, v_tgt)
            tot += float(resid.pow(2).mean())
        losses[m] = tot / m
        assert abs(d - 1.0 / m) < 1e-12
    r1 = losses[10] / losses[20]
    r2 = losses[20] / losses[40]
    assert 3.0 < r1 < 5.5, f"M 10->20 ratio {r1:.2f}, expected ~4"
    assert 3.0 < r2 < 5.5, f"M 20->40 ratio {r2:.2f}, expected ~4"


def test_velocity_form_beats_the_literal_form_in_bf16():
    """The cancellation, demonstrated rather than asserted.

    The realistic failure is building ``f = x + tau*v`` INSIDE the model's bf16 autocast
    region and differencing the two ``f``s there. At the anchor rung the true residual is
    ``delta*(v_online - v_teacher)`` -- roughly 1/10 of an already-small velocity gap -- but
    the literal form carries it inside two O(1) sums, where one bf16 ULP is ~8e-3. The
    velocity form keeps the small quantity small and does the combination in fp32.

    Note what this does NOT claim: if every term is upcast to fp32 before combining, the two
    forms are algebraically identical and agree exactly (asserted below). And neither form
    can recover a velocity gap that bf16 already destroyed in the forward -- the gap here is
    0.05, comfortably above bf16's resolution on an O(1) value.
    """
    torch.manual_seed(0)
    x0, eps = torch.randn(B, T, C), torch.randn(B, T, C)
    lo, hi = _tau(0.0), _tau(0.1)            # anchor rung: lam == 1, target drops out
    x_hi = interpolate(x0, eps, hi)
    v_teach = x0 - eps
    v_on = v_teach + 0.05 * torch.randn(B, T, C)
    x_lo = teacher_step(x_hi, v_teach, hi - lo)

    exact = float(((v_on - v_teach) * hi).pow(2).mean())

    def literal(dtype):
        """f built and differenced in `dtype` -- what happens inside an autocast forward."""
        a = consistency_fn(x_hi.to(dtype), hi.to(dtype), v_on.to(dtype))
        b = consistency_fn(x_lo.to(dtype), lo.to(dtype), v_teach.to(dtype))
        return float((a - b).float().pow(2).mean())

    def velocity(dtype):
        """Velocities in `dtype`, combination in fp32 -- what cd_loss does."""
        d = (v_on.to(dtype).float() - v_teach.to(dtype).float()) * hi
        return float(d.pow(2).mean())

    # In fp32 the two forms are the same computation, and agree.
    assert abs(literal(torch.float32) - exact) / exact < 1e-4
    assert abs(velocity(torch.float32) - exact) / exact < 1e-4

    lit_err = abs(literal(torch.bfloat16) - exact) / exact
    vel_err = abs(velocity(torch.bfloat16) - exact) / exact
    assert vel_err < 0.05, f"velocity form should survive bf16 inputs, got {vel_err:.4f}"
    assert lit_err > 4 * vel_err, f"literal {lit_err:.3f} vs velocity {vel_err:.4f}"


def test_literal_form_degrades_as_the_grid_gets_finer():
    """The cancellation is worst exactly where we might want to go: more rungs.

    Measured relative error of the two forms in bf16, over the gap between the online and
    teacher velocities. The literal form's error grows as the rung shrinks (the true
    residual scales with delta, the O(1) terms it hides inside do not); the velocity form
    is flat. This is why M is a free ablation knob with the velocity form and a precision
    cliff with the literal one.

        gap    M   literal   velocity
        0.05  10     0.389     0.0086
        0.05  40     5.712     0.0026
        0.01  40   149.732     0.1137
    """
    torch.manual_seed(0)
    x0, eps = torch.randn(B, T, C), torch.randn(B, T, C)
    v_teach = x0 - eps
    for gap, m, min_ratio in ((0.05, 10, 10.0), (0.05, 40, 100.0), (0.01, 40, 100.0)):
        lo, hi = _tau(0.0), _tau(1.0 / m)
        x_hi = interpolate(x0, eps, hi)
        v_on = v_teach + gap * torch.randn(B, T, C)
        x_lo = teacher_step(x_hi, v_teach, hi - lo)
        exact = float(((v_on - v_teach) * hi).pow(2).mean())
        d = torch.bfloat16
        lit = float((consistency_fn(x_hi.to(d), hi.to(d), v_on.to(d))
                     - consistency_fn(x_lo.to(d), lo.to(d), v_teach.to(d))).float().pow(2).mean())
        vel = float(((v_on.to(d).float() - v_teach.to(d).float()) * hi).pow(2).mean())
        le, ve = abs(lit - exact) / exact, abs(vel - exact) / exact
        assert le / max(ve, 1e-9) > min_ratio, f"gap={gap} M={m}: literal {le:.3f} velocity {ve:.4f}"


def test_loss_shapes_at_batch_one_and_eight():
    """B=1 is where the (B,1,1) timestep bug hides; B=8 is where it surfaces."""
    for b in (1, 8):
        lo, hi = _tau(0.2, b), _tau(0.3, b)
        loss, per = cd_loss(torch.randn(b, T, C), torch.randn(b, T, C),
                            torch.randn(b, T, C), lo, hi)
        assert loss.ndim == 0 and per.shape == (b,)


# ---------------------------------------------------------------- expert field wiring


def test_velocity_uses_shared_conditioning_and_does_not_normalize_twice():
    """Teacher/student calls must see the deployment mask and post-norm hidden state."""
    from alpamayo1_5_distill.models.consistency_expert import ConsistencyExpertVLA
    from alpamayo1_5_distill.models.expert_conditioning import build_expert_conditioning

    class Cache:
        def __init__(self):
            self.crops = []

        def crop(self, length):
            self.crops.append(length)

    class ActionIn(torch.nn.Module):
        def forward(self, x, timestep):
            self.timestep = timestep.detach().clone()
            return torch.cat((x, x), dim=-1)

    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = self._normalizing_twice_is_a_bug

        @staticmethod
        def _normalizing_twice_is_a_bug(_hidden):
            raise AssertionError("last_hidden_state is already final-normalized")

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])

    class ActionOut(torch.nn.Module):
        def forward(self, hidden):
            return hidden[..., :2]

    class Arm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = ActionIn()
            self.expert = Expert()
            self.action_out_proj = ActionOut()

    batch, n_action = 2, 2
    conditioning = build_expert_conditioning(
        traj_future_start_mask=torch.tensor([[False, True], [False, True]]),
        tokenizer_attention_mask=torch.ones(batch, 2),
        rope_deltas=torch.zeros(batch, 1, dtype=torch.long),
        n_action_tokens=n_action,
        dtype=torch.float32,
        attention_implementation="sdpa",
    )
    owner = SimpleNamespace(
        action_space=SimpleNamespace(get_action_space_dims=lambda: (n_action, 2)),
        config=SimpleNamespace(expert_non_causal_attention=True),
    )
    x = torch.randn(batch, n_action, 2)
    tau = torch.full((batch, 1, 1), 0.25)

    for arm in (Arm(), Arm()):  # frozen teacher and online/EMA share this exact path
        cache = Cache()
        velocity = ConsistencyExpertVLA._velocity(
            owner, arm, x, tau, cache, conditioning, torch.float32
        )
        torch.testing.assert_close(velocity, x)
        assert arm.expert.kwargs["attention_mask"] is conditioning.attention_mask
        assert arm.expert.kwargs["position_ids"] is conditioning.position_ids
        assert arm.expert.kwargs["is_causal"] is False
        assert cache.crops == [conditioning.prefix_len]


# ---------------------------------------------------------------- EMA


class _Tiny(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.expert = torch.nn.Linear(64, 64, dtype=dtype)
        self.vlm = torch.nn.Linear(8, 8, dtype=dtype)
        self.vlm.requires_grad_(False)


def test_ema_tracks_only_trainable_expert_params():
    m = _Tiny()
    ema = ExpertEMA(m, decay=0.9)
    assert all(n.startswith("expert.") for n in ema.shadow)
    assert not any(n.startswith("vlm.") for n in ema.shadow)


def test_ema_master_is_fp32_even_when_the_model_is_bf16():
    """The single most important line in ema.py.

    A bf16 master at decay=0.999 does not move AT ALL -- the increment is below one ULP --
    and the failure is silent: the target stays pinned at the teacher for the whole run.
    """
    m = _Tiny(dtype=torch.bfloat16)
    ema = ExpertEMA(m, decay=0.999)
    assert all(v.dtype == torch.float32 for v in ema.shadow.values())

    init = ema.snapshot()
    with torch.no_grad():
        for p in m.expert.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    for _ in range(200):
        ema.update(m)
    assert ema.distance_from(init) > 0.0, "fp32 EMA failed to move"

    # And the counterfactual: the same arithmetic carried in bf16 is frozen solid.
    shadow_bf16 = torch.ones(4096, dtype=torch.bfloat16)
    theta = torch.full((4096,), 1.02, dtype=torch.bfloat16)
    for _ in range(200):
        shadow_bf16.mul_(0.999).add_(theta, alpha=0.001)
    assert float((shadow_bf16.float() - 1.0).abs().max()) == 0.0


def test_ema_swap_in_and_out_round_trips_exactly():
    m = _Tiny()
    ema = ExpertEMA(m, decay=0.0)          # decay 0 => shadow follows theta exactly
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for p in m.expert.parameters():
            p.add_(1.0)
    ema.update(m)                          # shadow == the NEW weights
    with torch.no_grad():
        for p in m.expert.parameters():
            p.add_(1.0)                    # move again, so shadow != live
    live = {n: p.detach().clone() for n, p in m.named_parameters()}

    ema.swap_in(m)
    for n, p in m.named_parameters():
        if n in ema.shadow:
            torch.testing.assert_close(p, ema.shadow[n].to(p.dtype))
    ema.swap_out(m)
    for n, p in m.named_parameters():
        torch.testing.assert_close(p, live[n])
    assert before  # keep the reference; the point is `live`, not `before`


def test_ema_double_swap_in_raises_rather_than_losing_weights():
    """swap_in twice would overwrite the backup with EMA values and destroy training state."""
    m = _Tiny()
    ema = ExpertEMA(m, decay=0.9)
    ema.swap_in(m)
    try:
        ema.swap_in(m)
    except RuntimeError:
        ema.swap_out(m)
        return
    raise AssertionError("expected RuntimeError on double swap_in")
