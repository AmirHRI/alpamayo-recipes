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

"""KV-cache distillation primitives (KAVA, arXiv:2510.02312).

Pure tensor math with no model dependencies, so everything here is testable
without a GPU or a checkpoint.  Three groups:

1. **R-KV eviction** — score each of the teacher's ``N_C`` CoT key/value pairs by
   a redundancy/importance mix ``S = lam * I + (1 - lam) * R`` and keep the top
   ``M``, independently per (layer, head).  This is what compresses a CoT cache
   down to the student's latent budget.
2. **Layer mapping** — the teacher has 36 decoder layers and the student 28, so
   the per-layer loss needs an explicit student->teacher correspondence.
3. **The matching loss** — ``L_KV`` of Eq. 7, plus the per-layer projector bank
   that absorbs the cross-model basis mismatch (KAVA is self-distillation; here
   teacher and student are separate checkpoints whose ``W_k``/``W_v`` bases need
   not agree).

Conventions used throughout:

* ``K``/``V`` tensors are ``[..., n_heads, n_tokens, head_dim]`` — the layout
  ``DynamicCache`` and the attention modules use.
* Stored and matched keys are **pre-RoPE** (post ``k_norm``).  ``DynamicCache``
  holds *post*-RoPE keys, and matching those would force the student's latents to
  reproduce the teacher's rotation phase at positions that differ and that
  eviction has scrambled.  Scores, in contrast, come from the *real* (post-RoPE)
  attention, so both halves stay faithful to what the model actually did.
* Selected indices are returned in **ascending order**, i.e. the surviving CoT
  tokens keep their temporal order.  The student's latent slots are causally
  ordered, so slot *i* should target the *i*-th surviving token in time, not the
  *i*-th highest-scoring one.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

EvictionMethod = Literal["rkv", "cosine", "attn", "crop"]
KVLossType = Literal["l1", "mse", "smooth_l1"]


# --------------------------------------------------------------------- scoring
def redundancy_score(keys: torch.Tensor, n_valid: int | None = None) -> torch.Tensor:
    """R-KV redundancy score: how unlike every other key each key is.

    Mirrors the paper's Listing 1 — cosine-normalise, take the full pairwise
    similarity matrix with a zeroed diagonal, sum the *negated* similarities (so a
    key that resembles nothing else scores high), then softmax so the result is a
    distribution over the ``N`` tokens.  That last step matters: it puts ``R`` on
    the same scale as :func:`importance_score`, which is also a distribution, so
    ``lam`` in :func:`combine_scores` mixes comparable quantities.

    Args:
        keys: ``[..., N, D]`` key vectors for one (layer, head) or a stack of them.
        n_valid: number of non-padding tokens, used as the averaging denominator.
            Defaults to ``N``.

    Returns:
        ``[..., N]`` scores summing to 1 along the last axis.
    """
    n = keys.shape[-2]
    denom = float(n_valid if n_valid is not None else n)
    keys = keys.float()
    key_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    cos = torch.einsum("...id,...jd->...ij", key_norm, key_norm)
    eye = torch.eye(n, dtype=torch.bool, device=keys.device)
    cos = cos.masked_fill(eye, 0.0)
    cos_score = (-cos).sum(dim=-2) / denom
    return cos_score.softmax(dim=-1)


def importance_score(attn_cot: torch.Tensor) -> torch.Tensor:
    """R-KV importance score: attention mass the *answer* puts on each CoT token.

    ``attn_cot`` holds attention **probabilities** — already softmaxed over the
    full key set by the model — restricted to the CoT columns and averaged over
    the answer query positions.  Renormalising to sum to 1 makes it comparable to
    :func:`redundancy_score`; it leaves the ordering untouched, so it only changes
    how ``I`` and ``R`` trade off, never how ``I`` alone ranks tokens.

    Args:
        attn_cot: ``[..., N]`` non-negative attention mass per CoT token.

    Returns:
        ``[..., N]`` scores summing to 1 along the last axis.
    """
    attn_cot = attn_cot.float().clamp_min(0.0)
    total = attn_cot.sum(dim=-1, keepdim=True)
    return attn_cot / total.clamp_min(1e-12)


def combine_scores(
    imp: torch.Tensor | None,
    red: torch.Tensor | None,
    lam: float,
    method: EvictionMethod = "rkv",
) -> torch.Tensor:
    """``S = lam * I + (1 - lam) * R``, with the paper's two ablation extremes.

    ``method="attn"`` forces ``lam=1`` (importance only) and ``method="cosine"``
    forces ``lam=0`` (diversity only); ``"rkv"`` uses the supplied ``lam`` (0.1
    throughout KAVA).  ``"crop"`` needs no score — :func:`select_top_m` handles it.
    """
    if method == "attn":
        lam = 1.0
    elif method == "cosine":
        lam = 0.0
    elif method not in ("rkv", "crop"):
        raise ValueError(f"unknown eviction method {method!r}")

    if lam > 0.0 and imp is None:
        raise ValueError(f"method={method!r} lam={lam} needs importance scores, got None")
    if lam < 1.0 and red is None:
        raise ValueError(f"method={method!r} lam={lam} needs redundancy scores, got None")

    if lam >= 1.0:
        return importance_score(imp)
    if lam <= 0.0:
        return red.float()
    return lam * importance_score(imp) + (1.0 - lam) * red.float()


def select_top_m(
    scores: torch.Tensor,
    m: int,
    n_valid: int,
    method: EvictionMethod = "rkv",
) -> torch.Tensor:
    """Keep the ``min(m, n_valid)`` best tokens per (layer, head), in time order.

    Args:
        scores: ``[..., N]`` from :func:`combine_scores`.  For ``"crop"`` only its
            shape is read.
        m: the latent budget.
        n_valid: real CoT length; tokens at ``>= n_valid`` are padding and are
            scored ``-inf`` so they can never be selected.
        method: ``"crop"`` keeps the first ``m`` tokens (the paper's naive
            baseline); anything else takes the top-``m`` scores.

    Returns:
        ``[..., min(m, n_valid)]`` long indices, ascending along the last axis.
    """
    m_eff = min(int(m), int(n_valid))
    if m_eff <= 0:
        raise ValueError(f"nothing to select: m={m}, n_valid={n_valid}")

    if method == "crop":
        idx = torch.arange(m_eff, device=scores.device)
        return idx.expand(*scores.shape[:-1], m_eff).contiguous()

    scores = scores.float().clone()
    if n_valid < scores.shape[-1]:
        scores[..., n_valid:] = float("-inf")
    idx = scores.topk(m_eff, dim=-1).indices
    return idx.sort(dim=-1).values


def gather_selected(kv: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``[..., N, D]`` down to ``[..., M, D]`` with per-row indices ``[..., M]``."""
    expanded = idx.unsqueeze(-1).expand(*idx.shape, kv.shape[-1])
    return kv.gather(dim=-2, index=expanded)


def evict_teacher_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    m: int,
    n_valid: int,
    imp: torch.Tensor | None = None,
    red: torch.Tensor | None = None,
    lam: float = 0.1,
    method: EvictionMethod = "rkv",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress a teacher CoT cache to the latent budget.

    Args:
        keys, values: ``[L, H, N, D]`` pre-RoPE keys / values over the CoT span.
        m: latent budget.
        n_valid: real CoT length (``keys`` may be padded past it).
        imp: ``[L, H, N]`` attention mass per CoT token, or None for
            diversity-only / crop.
        red: ``[L, H, N]`` precomputed redundancy, or None to compute it here.
        lam, method: see :func:`combine_scores`.

    Returns:
        ``(k_sel, v_sel, idx)`` shaped ``[L, H, M', D]``, ``[L, H, M', D]``,
        ``[L, H, M']`` where ``M' = min(m, n_valid)``.
    """
    if method != "crop" and lam < 1.0 and red is None:
        red = redundancy_score(keys[..., :n_valid, :], n_valid=n_valid)
        if n_valid < keys.shape[-2]:
            red = F.pad(red, (0, keys.shape[-2] - n_valid))

    scores = keys[..., 0] if method == "crop" else combine_scores(imp, red, lam, method)
    idx = select_top_m(scores, m, n_valid, method)
    return gather_selected(keys, idx), gather_selected(values, idx), idx


# ------------------------------------------------------------------- layer map
def build_layer_map(
    n_student_layers: int, n_teacher_layers: int, explicit: list[int] | None = None
) -> list[int]:
    """Student layer index -> teacher layer index.

    Default is a uniform stride with both endpoints preserved (student 0 -> teacher
    0, student ``L_s-1`` -> teacher ``L_t-1``), i.e. ``round(i * (L_t-1)/(L_s-1))``.

    Every student layer appears exactly once, deliberately: the action expert reads
    *all* of them (expert layer *i* cross-attends VLM cache layer *i*), so leaving
    any student layer unsupervised leaves part of the expert's input untouched.
    """
    if explicit is not None:
        if len(explicit) != n_student_layers:
            raise ValueError(
                f"explicit layer map has {len(explicit)} entries, expected {n_student_layers}"
            )
        if not all(0 <= t < n_teacher_layers for t in explicit):
            raise ValueError(f"explicit layer map out of range for {n_teacher_layers} layers")
        return [int(t) for t in explicit]

    if n_student_layers == 1:
        return [n_teacher_layers - 1]
    span = (n_teacher_layers - 1) / (n_student_layers - 1)
    return [int(round(i * span)) for i in range(n_student_layers)]


# -------------------------------------------------------------------- projector
class KVProjectorBank(nn.Module):
    """Per-layer ``1024 -> 1024`` maps carrying the student's K/V into teacher space.

    KAVA is self-distillation, so its K/V live in one basis.  Here teacher and
    student are separately trained checkpoints: per-layer geometry agrees exactly
    (``8 kv-heads x 128 = 1024`` on both sides) but the learned ``W_k``/``W_v``
    bases need not.  ``align="projector"`` learns that change of basis;
    ``align="direct"`` asserts there is nothing to learn and costs no parameters.

    The projectors are **identity-initialised**, so training starts at exactly the
    ``direct`` objective and can only move away from it if that helps — which makes
    the two settings a clean ablation pair rather than two unrelated runs.  They
    are discarded at inference; nothing outside the loss reads them.
    """

    def __init__(
        self,
        n_student_layers: int,
        kv_width: int = 1024,
        align: Literal["direct", "projector"] = "projector",
    ) -> None:
        super().__init__()
        if align not in ("direct", "projector"):
            raise ValueError(f"unknown kv_align {align!r}")
        self.align = align
        self.kv_width = int(kv_width)
        if align == "direct":
            return
        self.k_proj = nn.ModuleList(
            [nn.Linear(kv_width, kv_width, bias=False) for _ in range(n_student_layers)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(kv_width, kv_width, bias=False) for _ in range(n_student_layers)]
        )
        for mod in list(self.k_proj) + list(self.v_proj):
            nn.init.eye_(mod.weight)

    def forward(self, kv: torch.Tensor, layer: int, which: Literal["k", "v"]) -> torch.Tensor:
        """Project ``[B, H, M, D]`` for one student layer; identity when ``direct``.

        The map is over the flattened ``H*D`` width, so it can mix across heads —
        which is the point: eviction already destroyed head-wise token
        correspondence, so there is nothing to preserve per head.
        """
        if self.align == "direct":
            return kv
        proj = (self.k_proj if which == "k" else self.v_proj)[layer]
        b, h, m, d = kv.shape
        flat = kv.permute(0, 2, 1, 3).reshape(b, m, h * d)
        out = proj(flat.to(proj.weight.dtype))
        return out.view(b, m, h, d).permute(0, 2, 1, 3)


# ------------------------------------------------------------------------ loss
def _elementwise_loss(a: torch.Tensor, b: torch.Tensor, kind: KVLossType) -> torch.Tensor:
    if kind == "l1":
        return (a - b).abs()
    if kind == "mse":
        return (a - b).pow(2)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(a, b, reduction="none")
    raise ValueError(f"unknown kv loss {kind!r}")


def kv_matching_loss(
    student_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    teacher_k: torch.Tensor,
    teacher_v: torch.Tensor,
    layer_map: list[int],
    valid_mask: torch.Tensor | None = None,
    projector: "KVProjectorBank | None" = None,
    kind: KVLossType = "smooth_l1",
    layerwise_std: bool = False,
) -> torch.Tensor:
    """``L_KV`` of KAVA Eq. 7, layer-mapped and length-masked.

    Keys and values are weighted equally and averaged over every valid element,
    which keeps the loss scale independent of ``M``, of how many layers are mapped,
    and of the head count — so one ``kv_loss_weight`` transfers across the
    ``M``/layer-map sweep instead of needing a re-tune per point.

    Args:
        student_kv: ``{student_layer_idx: (K, V)}``, each ``[B, H, M, D]`` and
            gradient-carrying.
        teacher_k, teacher_v: ``[B, L_t, H, M, D]`` compressed teacher cache,
            zero-padded past each sample's ``M' = min(M, N_C)``.
        layer_map: from :func:`build_layer_map`, indexed by student layer.
        valid_mask: ``[B, M]`` bool, False on padded slots.  Real CoT traces run
            ~40 tokens and can be shorter than ``M`` — a case KAVA never faces —
            and those slots have no target, so they must not contribute.
        projector: applied to the student side; None means ``direct``.
        kind: ``"l1"`` / ``"mse"`` / ``"smooth_l1"`` (the paper sweeps all three).
        layerwise_std: divide each layer's residual by the std of that layer's
            teacher target (detached) — KAVA's "Layer-wise std" row.  Qwen3-VL K/V
            magnitudes differ by orders of magnitude across depth ("massive
            activations"), so without it the largest layers dominate the gradient.

    Returns:
        Scalar loss; a gradient-free zero if nothing valid is left.
    """
    weight = None
    if valid_mask is not None:
        # [B, M] -> [B, 1, M, 1] to broadcast over the per-layer [B, H, M, D] tensors.
        weight = valid_mask[:, None, :, None].to(torch.float32)

    total: torch.Tensor | float = 0.0
    denom = 0.0
    for student_layer, (k_s, v_s) in sorted(student_kv.items()):
        t_layer = layer_map[student_layer]
        k_t = teacher_k[:, t_layer].to(k_s.device).float()
        v_t = teacher_v[:, t_layer].to(v_s.device).float()

        if projector is not None:
            k_s = projector(k_s, student_layer, "k")
            v_s = projector(v_s, student_layer, "v")
        k_s = k_s.float()
        v_s = v_s.float()

        k_err = _elementwise_loss(k_s, k_t, kind)
        v_err = _elementwise_loss(v_s, v_t, kind)

        if layerwise_std:
            if weight is not None:
                sel = weight.expand_as(k_t) > 0
                k_scale, v_scale = k_t[sel].std(), v_t[sel].std()
            else:
                k_scale, v_scale = k_t.std(), v_t.std()
            k_err = k_err / k_scale.detach().clamp_min(1e-6)
            v_err = v_err / v_scale.detach().clamp_min(1e-6)

        if weight is not None:
            w = weight.expand_as(k_err)
            total = total + (k_err * w).sum() + (v_err * w).sum()
            denom += 2.0 * float(w.sum())
        else:
            total = total + k_err.sum() + v_err.sum()
            denom += 2.0 * k_err.numel()

    if not torch.is_tensor(total) or denom == 0:
        ref = next(iter(student_kv.values()))[0] if student_kv else teacher_k
        return torch.zeros((), device=ref.device, dtype=torch.float32)
    return total / denom
