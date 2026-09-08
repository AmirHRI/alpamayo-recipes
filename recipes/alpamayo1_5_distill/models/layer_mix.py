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

r"""Drive a DEEPER action expert than the student VLM, by mixing cache layers.

HF indexes ``past_key_values`` by ``layer_idx``, so expert layer *l* reads VLM cache layer
*l* and the expert's depth is pinned to the VLM's.  A 28-layer 2B student therefore forced
the teacher's 36-layer expert down to 28 (``expert_holder._load``'s depth remap, driven by
``PRUNE_EXPERT_LAYERS``).  That cut is expensive before the student is even involved:
``PRUNING.md`` measures the teacher at min_ade **0.5776** through its full 36-layer expert
and **0.7893** through the set-C 28-layer ablation.

This module removes the cut.  Keep all 36 expert layers and SYNTHESISE the 36 cache slots
from the student's 28 layers, block-structured ``n_blocks x (g_in -> g_out)``:

    student VLM layers   0..6      7..13     14..20    21..27      (4 blocks of 7)
                           |  P_K/P_V |         |         |         (each 7x9)
    expert cache slots   0..8      9..17     18..26    27..35      (4 blocks of 9)

Each output slot is a CONVEX combination of the 7 student layers in its block:
``P = softmax(logit, dim=g_in)``, so every column is non-negative and sums to 1.  Block
locality is structural -- block *b*'s outputs are a function of block *b*'s inputs and
nothing else -- which is what makes the ``m = g_out`` span loss meaningful: one span is
exactly one block's 9 expert layers driven by exactly one block's 7 student layers.

**Why this cannot hide inside the loss.**  ``KVProjectorBank`` carries the warning at
``kv_distill.py:294-304``: a learned map that exists only inside ``L_KV`` can absorb the
whole cross-model mismatch while the tensor the expert actually reads never moves (measured:
``kv_loss`` floors at ~0.60 by epoch 0.30 and stops).  P is not that.  P is ON THE INFERENCE
PATH by construction -- it *is* the cache the expert consumes, in training and at eval alike.
It has nowhere to hide, and any capacity it has is spent on the tensor being graded.

**Why P stays weak on purpose.**  63 convex weights per block per K/V can only REWEIGHT
layers the student already produced; they cannot synthesise content the student never had.
That structural cap -- not a small learning rate -- is what keeps the mix from absorbing the
training signal.  ⚠️ Under Adam the step size is ~``lr`` per parameter regardless of gradient
magnitude, so a *small* multiplier on 504 scalars freezes them: at 0.1x of a 1e-5 base a
logit moves ~0.0075 across a 5-epoch run.  The shipped arm uses ``layer_mix: 10.0``.

⚠️ **A layer-mix commutes with RoPE, and the whole train/eval agreement rests on it.**
``Qwen3VLTextModel.forward`` computes ``position_embeddings`` once and hands the SAME
``cos``/``sin`` to every layer, and rotation is linear in K:

.. math::
    \mathrm{RoPE}_p\Big(\sum_l a_l k^{(l)}\Big) = \sum_l a_l \,\mathrm{RoPE}_p\big(k^{(l)}\big)

So mixing pre-RoPE K (training, from ``recompute_kv``) and mixing the post-RoPE
``DynamicCache`` (eval, from the VLM prefill) give the same answer, and the two paths cannot
drift apart.  V is never rotated, so it is unconstrained either way.  This is asserted by
``tests/test_layer_mix.py``, not assumed.  Note the contrast with the per-layer ``D x D``
basis maps ``scripts/kv_linear_probe.py`` warns about -- those do NOT commute with RoPE,
because they mix the head dimension the rotation acts on.  This mixes the LAYER axis, which
the rotation does not touch.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

#: Probability floor applied to the off-tent entries at init.  Two jobs: it keeps
#: ``log(w)`` finite, and it leaves the unused entries with a small but NON-ZERO softmax
#: mass so they still receive gradient (the derivative w.r.t. a logit scales with its own
#: probability, so an exactly-zero entry is an entry that can never come back).  1e-3 costs
#: ~0.5% of each column's mass, i.e. the init is the depth-matched tent to within half a
#: percent.  1e-2 would cost 5% and visibly perturb the zero-shot starting point.
LAYER_MIX_FLOOR = 1e-3

_WHICH = ("k", "v")


def tent_weights(g_in: int, g_out: int) -> torch.Tensor:
    """Depth-proportional linear-interpolation weights, ``[g_in, g_out]``, columns sum to 1.

    Output *q* of a block draws from source position ``p = q * (g_in - 1) / (g_out - 1)``,
    splitting its mass between ``floor(p)`` and ``ceil(p)``.  For 7 -> 9 that is
    ``p = 0.75 q``: slot 0 is exactly student layer 0, slot 8 is exactly student layer 6, and
    the seven in between interpolate.  Block endpoints are therefore EXACT, and stacking the
    four blocks reproduces the tree's existing depth-proportional convention -- expert slot 9
    reads student layer 7, 18 -> 14, 27 -> 21, which is ``build_layer_map``'s inverse
    (``round(j * 27/35)``) to within a layer.

    This is the ``nn.init.eye_`` of this parameterisation (``kv_distill.py:336`` does the
    same for ``KVProjectorBank``): near-identity at init, so step 0 is a KNOWN quantity that
    can be evaluated on its own before a single gradient step, and everything the run learns
    is a departure from a documented baseline rather than from noise.
    """
    if g_in < 2 or g_out < 2:
        raise ValueError(f"tent_weights needs g_in, g_out >= 2; got {g_in}, {g_out}")
    w = torch.zeros(g_in, g_out, dtype=torch.float32)
    for q in range(g_out):
        p = q * (g_in - 1) / (g_out - 1)
        lo = int(p)
        frac = p - lo
        hi = min(lo + 1, g_in - 1)
        w[lo, q] += 1.0 - frac
        w[hi, q] += frac
    return w


def gather_weights(g_in: int, g_out: int) -> torch.Tensor:
    """One-hot NEAREST source per target -- the tent's sharp endpoint, ``[g_in, g_out]``.

    No blending at all: output *q* copies source ``round(q * (g_in-1) / (g_out-1))``.
    """
    w = torch.zeros(g_in, g_out, dtype=torch.float32)
    for q in range(g_out):
        w[round(q * (g_in - 1) / (g_out - 1)), q] = 1.0
    return w


#: How far to sharpen the depth-matched tent toward one-hot at init: 0.0 is the pure tent,
#: 1.0 is pure :func:`gather_weights`.
#:
#: ⚠️ MEASURED, and only weakly. The single relevant datum is behavioural: in the validated
#: teacher-mode oracle (n=500) one-hot ``gather`` scored min_ade 0.7396 against the tent's
#: 0.8059, so blending adjacent layers everywhere HURT by 0.066. That says "sharper than the
#: tent"; it does not pin a number, and 1.0 is the endpoint that actually won.
#:
#: ⚠️ Do NOT try to justify a value from the shape of the fitted P. Per column its effective
#: sources are 2.222 against the tent's 1.489 -- it is maximally SHARP on the 7 of 9 columns
#: where a pass-through source exists and maximally BROAD on the 2 where none does. That
#: bimodality is the probe's degeneracy showing through, not a property of a good mix.
#: (Aggregating as ``1/mean(sum w^2)`` reports 1.23 vs 1.38 and hides it entirely.)
LAYER_MIX_SHARPEN = 0.75


def init_weights(
    g_in: int, g_out: int, sharpen: float = LAYER_MIX_SHARPEN
) -> torch.Tensor:
    """The init: a convex blend of the depth-matched tent and its one-hot endpoint.

    A CONVEX blend rather than a softmax temperature, because temperature cannot sharpen a
    tied column: for 7 -> 9 two of every nine outputs land exactly halfway between two
    sources, and ``softmax(log w / T)`` leaves ``(0.5, 0.5)`` at ``(0.5, 0.5)`` for every T.
    Blending toward ``gather_weights`` moves those columns like all the others, and reaches
    the measured-best endpoint exactly at ``sharpen = 1``.

    Both endpoints have unit columns, so the blend does too -- no renormalisation.
    """
    if not 0.0 <= sharpen <= 1.0:
        raise ValueError(f"sharpen must be in [0, 1], got {sharpen}")
    t = tent_weights(g_in, g_out)
    return t if sharpen == 0.0 else (1.0 - sharpen) * t + sharpen * gather_weights(g_in, g_out)


def init_logits(
    g_in: int,
    g_out: int,
    sharpen: float = LAYER_MIX_SHARPEN,
    floor: float = LAYER_MIX_FLOOR,
) -> torch.Tensor:
    """``log`` of the floored init, so ``softmax`` over ``dim=0`` reproduces it."""
    return torch.log(init_weights(g_in, g_out, sharpen).clamp_min(floor))


class LayerMixer(nn.Module):
    """Synthesise an ``n_expert``-slot expert cache from an ``n_student``-layer VLM cache.

    Args:
        n_student: the VLM's text depth (28 for Cosmos-Reason2-2B).
        n_expert: the expert's depth, i.e. how many cache slots must be produced (36).
        n_blocks: how many independent ``g_in -> g_out`` mixes.  Must divide both depths;
            block locality is what makes ``block_span_mix = g_out`` grade one block at a time.
        gain: add a per-output-slot scalar (init 1.0), one bank for K and one for V.  OFF by
            default.  A mix drifting toward uniform shrinks ``||K||`` by up to ``sqrt(g_in)``
            for uncorrelated layers; ``kv_ratio_k``/``kv_ratio_v`` in ``kd_model._block_loss``
            report that every step, so turn this on in response to a measurement rather than
            in anticipation of one.
        sharpen: how far the init is pulled from the depth-matched tent toward one-hot.
            See :data:`LAYER_MIX_SHARPEN`.
    """

    def __init__(
        self,
        n_student: int = 28,
        n_expert: int = 36,
        n_blocks: int = 4,
        gain: bool = False,
        sharpen: float = LAYER_MIX_SHARPEN,
        pin_head: int = 0,
        pin_tail: int = 0,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        pin_head, pin_tail = int(pin_head), int(pin_tail)
        if pin_head < 0 or pin_tail < 0:
            raise ValueError(f"pins must be >= 0, got {pin_head}, {pin_tail}")
        # A pinned layer is wired STRAIGHT THROUGH: student layer i becomes expert slot i at
        # the head, and student layer n_student-1-k becomes expert slot n_expert-1-k at the
        # tail. No parameters, no mixing, and the slot count consumed is identical on both
        # sides -- so every extra expert slot is absorbed by the learned middle.
        mid_src = n_student - pin_head - pin_tail
        mid_dst = n_expert - pin_head - pin_tail
        if mid_src < n_blocks or mid_dst < n_blocks:
            raise ValueError(
                f"pins leave {mid_src} student / {mid_dst} expert layers for {n_blocks} "
                "blocks; nothing to mix"
            )
        if mid_src % n_blocks or mid_dst % n_blocks:
            raise ValueError(
                f"n_blocks={n_blocks} must divide the UNPINNED middle, but {mid_src} % "
                f"{n_blocks} = {mid_src % n_blocks} and {mid_dst} % {n_blocks} = "
                f"{mid_dst % n_blocks}. Ragged blocks would make the span loss compare across "
                "block boundaries."
            )
        self.n_student = int(n_student)
        self.n_expert = int(n_expert)
        self.n_blocks = int(n_blocks)
        self.pin_head = pin_head
        self.pin_tail = pin_tail
        self.g_in = mid_src // n_blocks
        self.g_out = mid_dst // n_blocks

        self.sharpen = float(sharpen)
        init = (init_logits(self.g_in, self.g_out, self.sharpen)
                .expand(self.n_blocks, -1, -1).clone())
        self.logit_k = nn.Parameter(init.clone())
        self.logit_v = nn.Parameter(init.clone())
        if gain:
            self.gain_k = nn.Parameter(torch.ones(self.n_expert))
            self.gain_v = nn.Parameter(torch.ones(self.n_expert))
        else:
            self.gain_k = self.gain_v = None

    # ------------------------------------------------------------------ weights
    def _logit(self, which: Literal["k", "v"]) -> torch.Tensor:
        if which not in _WHICH:
            raise ValueError(f"which must be 'k' or 'v', got {which!r}")
        return self.logit_k if which == "k" else self.logit_v

    def _gain(self, which: Literal["k", "v"]) -> torch.Tensor | None:
        return self.gain_k if which == "k" else self.gain_v

    def weights(self, which: Literal["k", "v"]) -> torch.Tensor:
        """``[n_blocks, g_in, g_out]``, each column on the simplex.

        The softmax is over ``dim=1`` -- the ``g_in`` STUDENT layers -- so column ``(b, :, q)``
        is how expert slot ``b * g_out + q`` weights its block's student layers.  Softmaxing
        the other axis would normalise over expert slots instead and mean nothing.
        """
        return torch.softmax(self._logit(which).float(), dim=1)

    @torch.no_grad()
    def entropy(self, which: Literal["k", "v"]) -> torch.Tensor:
        """Mean column entropy, normalised to ``[0, 1]``: 0 = one-hot, 1 = uniform.

        The single number that says whether P moved and WHICH WAY.  Flat at the init value
        means the learning rate is too low to matter; racing to 1.0 means the mix has
        collapsed to a plain block average, which also shrinks ``||K||`` -- read it next to
        ``kv_ratio_k``/``kv_ratio_v``.

        ⚠️ The tent init does NOT read ~0.  Seven of the nine slots in a block interpolate
        between two source layers, so the init sits at **~0.228** for the shipped 7 -> 9
        geometry (three one-hot slots, four at (0.75, 0.25), two at (0.5, 0.5)).  That is the
        baseline to read departures against; 0 would mean a pure nearest-neighbour map, which
        this deliberately is not.

        Diagnostic only, hence ``no_grad``: nothing should ever backprop through it.
        """
        p = self.weights(which)
        h = -(p.clamp_min(1e-12).log() * p).sum(dim=1)          # [n_blocks, g_out]
        return h.mean() / torch.log(torch.tensor(float(self.g_in)))

    # ------------------------------------------------------------------ apply
    def mix_stacked(self, x: torch.Tensor, which: Literal["k", "v"]) -> torch.Tensor:
        """``[B, n_student, H, T, D] -> [B, n_expert, H, T, D]``, block-convexly.

        Kept as one ``einsum`` per block rather than a single dense ``n_student x n_expert``
        contraction: the dense form would need a masked/zero-padded P and would let a bug
        leak mass across block boundaries silently, which is exactly the structural property
        the span loss depends on.
        """
        if x.shape[1] != self.n_student:
            raise ValueError(
                f"expected {self.n_student} source layers on dim 1, got {x.shape[1]}"
            )
        p = self.weights(which).to(x.dtype)
        out = []
        if self.pin_head:
            out.append(x[:, : self.pin_head])            # straight through, no parameters
        for b in range(self.n_blocks):
            lo = self.pin_head + b * self.g_in
            out.append(torch.einsum("bihtd,io->bohtd", x[:, lo : lo + self.g_in], p[b]))
        if self.pin_tail:
            out.append(x[:, self.n_student - self.pin_tail :])
        mixed = torch.cat(out, dim=1)
        g = self._gain(which)
        if g is not None:
            # ⚠️ NOT applied to pinned slots: a pin means "this cache layer reaches the expert
            # unaltered", and a learned scale would quietly make that false.
            gg = g.to(mixed.dtype).clone()
            if self.pin_head:
                gg[: self.pin_head] = 1.0
            if self.pin_tail:
                gg[self.n_expert - self.pin_tail :] = 1.0
            mixed = mixed * gg.view(1, -1, 1, 1, 1)
        return mixed

    def mix_dict(
        self, kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """TRAINING path: ``recompute_kv``'s ``{layer: (K, V)}`` at 28 -> the same at 36.

        Grad-carrying, and the returned dict is keyed 0..n_expert-1 so ``_block_loss`` indexes
        it exactly as it indexes an unmixed one.  K here is PRE-RoPE; see the module docstring
        for why that is equivalent to mixing after rotation.
        """
        missing = [i for i in range(self.n_student) if i not in kv]
        if missing:
            raise ValueError(
                f"layer mix needs all {self.n_student} student layers; missing {missing[:5]}"
            )
        k = torch.stack([kv[i][0] for i in range(self.n_student)], dim=1)
        v = torch.stack([kv[i][1] for i in range(self.n_student)], dim=1)
        k_m = self.mix_stacked(k, "k")
        v_m = self.mix_stacked(v, "v")
        return {j: (k_m[:, j], v_m[:, j]) for j in range(self.n_expert)}

    def mix_cache(self, cache: DynamicCache) -> DynamicCache:
        """EVAL path: a 28-layer prefill ``DynamicCache`` -> a fresh 36-slot one.

        ⚠️ The keys in a ``DynamicCache`` are POST-RoPE (``modeling_qwen3_vl.py:433`` rotates
        before ``update``), unlike the pre-RoPE tensors ``mix_dict`` receives.  Mixing either
        gives the same result -- see the module docstring -- which is what lets training and
        deployment share one P.

        A NEW cache is returned rather than the input mutated: the caller may still hold the
        VLM's own 28-layer cache, and ``DynamicCache`` slot count is not resizable in place.

        ⚠️ **Skipping this call does not raise.**  MEASURED against a real 36-layer Qwen3-VL
        decoder: handing it a 28-layer ``DynamicCache`` runs clean and returns finite hidden
        states, because ``Cache.update`` AUTO-EXTENDS -- slots 28..35 are created on demand
        and hold only the 6-64 action tokens, with no VLM prefix at all.  So the deepest eight
        expert layers, the ones ``scripts/layer_importance.py`` measured as carrying ~105% of
        the recoverable gap, would attend to nothing but the action queries and the number
        would still look like a number.  That is why every caller-side guard around this
        (``_load_layer_mix``'s hard raise, the ``STITCH_ROLLOUT`` refusal, and
        ``kd_model._init_layer_mix``) is load-bearing rather than defensive.
        """
        n_src = len(cache.layers)
        if n_src != self.n_student:
            raise ValueError(
                f"prefill cache has {n_src} layers, mixer expects {self.n_student}"
            )
        k = torch.stack([lyr.keys for lyr in cache.layers], dim=1)
        v = torch.stack([lyr.values for lyr in cache.layers], dim=1)
        k_m = self.mix_stacked(k, "k")
        v_m = self.mix_stacked(v, "v")
        out = DynamicCache()
        for j in range(self.n_expert):
            out.update(k_m[:, j].contiguous(), v_m[:, j].contiguous(), j, {})
        return out

    # ------------------------------------------------------------------ reporting
    def describe(self) -> str:
        """The full P, for the log.  What actually drove a number must be recoverable."""
        pins = (f", pinned head {self.pin_head} tail {self.pin_tail} (identity)"
                if (self.pin_head or self.pin_tail) else "")
        lines = [
            f"[layer-mix] {self.n_student} -> {self.n_expert} in {self.n_blocks} learned blocks "
            f"({self.g_in} -> {self.g_out} each){pins}, sharpen={self.sharpen}, "
            f"gain={'on' if self.gain_k is not None else 'off'}",
            f"[layer-mix] entropy k={float(self.entropy('k')):.4f} "
            f"v={float(self.entropy('v')):.4f}  (0=one-hot, 1=uniform)",
        ]
        if self.pin_head:
            lines.append(f"[layer-mix] PINNED head: student 0..{self.pin_head - 1} -> "
                         f"slots 0..{self.pin_head - 1}, identity")
        for which in _WHICH:
            p = self.weights(which).detach()
            for b in range(self.n_blocks):
                s0 = self.pin_head + b * self.g_in
                d0 = self.pin_head + b * self.g_out
                src = f"{s0}..{s0 + self.g_in - 1}"
                dst = f"{d0}..{d0 + self.g_out - 1}"
                for q in range(self.g_out):
                    row = " ".join(f"{float(x):5.3f}" for x in p[b, :, q])
                    lines.append(
                        f"[layer-mix] P_{which.upper()} block{b} slot{d0 + q:>2} "
                        f"<- [{src}] {row}   (block {src} -> {dst})"
                    )
        if self.pin_tail:
            lines.append(f"[layer-mix] PINNED tail: student "
                         f"{self.n_student - self.pin_tail}..{self.n_student - 1} -> slots "
                         f"{self.n_expert - self.pin_tail}..{self.n_expert - 1}, identity")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return (
            f"n_student={self.n_student}, n_expert={self.n_expert}, "
            f"n_blocks={self.n_blocks}, g_in={self.g_in}, g_out={self.g_out}, "
            f"pin_head={self.pin_head}, pin_tail={self.pin_tail}, "
            f"sharpen={self.sharpen}, gain={self.gain_k is not None}"
        )
