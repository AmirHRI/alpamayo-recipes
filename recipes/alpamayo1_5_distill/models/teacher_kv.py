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

"""Teacher-side capture: per-layer CoT K/V plus the R-KV eviction scores.

Used only by the offline cache builder, never at training time.  Everything is
extracted from **one** full forward split into two segments, so the cost over the
existing single-vector cache run is negligible:

* **segment A** — ``[context + CoT]`` with ``use_cache=True``.  Forward hooks on
  each layer's ``k_norm`` / ``v_proj`` collect the CoT span's **pre-RoPE** keys and
  values, and the resulting cache is kept.
* **segment B** — the ``[<|cot_end|>, <|traj_future_start|>]`` tail against that
  cache with eager attention and ``output_attentions=True``.  This yields both the
  ``<traj_future_start>`` hidden (the existing single-vector target, so one cache
  run feeds both objectives) and the attention the *answer* pays to each CoT token,
  which is R-KV's importance signal.

Why pre-RoPE keys: ``DynamicCache`` stores keys *after* rotation, and matching those
would make the student reproduce the teacher's rotation phase at positions that
differ and that eviction has scrambled.  The scores, by contrast, come from the real
post-RoPE attention — so the choice of *what* to keep reflects what the model
actually attended to, while the *values* kept are position-free.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from alpamayo1_5_distill.models.kv_distill import redundancy_score

#: Query-side sources for R-KV's importance score.
IMPORTANCE_SOURCES = ("vlm_post_cot", "none", "expert")


@dataclass
class TeacherKVSample:
    """One sample's teacher capture, ready for :mod:`..data.kv_cache_io`."""

    k_pre: torch.Tensor  # [L, H, N_C, D] pre-RoPE keys over the CoT span
    v: torch.Tensor  # [L, H, N_C, D] values over the CoT span
    imp: torch.Tensor | None  # [L, H, N_C] answer-attention mass, or None
    red: torch.Tensor | None  # [L, H, N_C] redundancy, or None
    tfs_hidden: torch.Tensor  # [H_teacher] hidden at <traj_future_start>
    #: ``[L+1, H_teacher]`` hidden at ``<traj_future_start>`` for EVERY layer, index 0
    #: being the embedding output.  CoDI distils all layers and averages
    #: (``distill_loss /= len(outputs.hidden_states)``); ``tfs_hidden`` above is just
    #: the last entry of this, kept separate so old caches stay readable.
    tfs_hidden_all: torch.Tensor | None
    cot_text: str
    n_cot: int


def find_cot_span(
    input_ids: torch.Tensor, cot_start_id: int, cot_end_id: int, tfs_id: int
) -> tuple[int, int]:
    """Locate the CoT token span ``[lo, hi)`` in a single-row sequence.

    Starts after the last ``<|cot_start|>`` and ends at whichever of
    ``<|cot_end|>`` / ``<|traj_future_start|>`` comes first after it, so both
    ``components_order`` variants work unchanged: ``distill_teacher`` emits
    ``<|cot_start|> ... <|cot_end|> <|tfs|>`` while ``distill_teacher_generate``
    ends the prompt at ``<|cot_start|>`` and the teacher generates the rest.

    Raises:
        ValueError: if there is no ``<|cot_start|>``, or the span is empty.  An
            empty span is the signature of the ``components_order`` gotcha — a
            ``traj_future``-last prompt pre-fills ``<|traj_future_start|>`` and the
            teacher emits no reasoning at all — so it fails here rather than
            silently caching a zero-length target.
    """
    row = input_ids[0] if input_ids.dim() == 2 else input_ids
    starts = (row == cot_start_id).nonzero(as_tuple=True)[0]
    if len(starts) == 0:
        raise ValueError(
            "no <|cot_start|> in the sequence: the teacher-side processor must have "
            "`cot` in components_order (e.g. vla_processor=distill_teacher_generate)."
        )
    lo = int(starts[-1]) + 1

    tail = row[lo:]
    stops = ((tail == cot_end_id) | (tail == tfs_id)).nonzero(as_tuple=True)[0]
    hi = lo + int(stops[0]) if len(stops) else int(row.shape[0])
    if hi <= lo:
        raise ValueError(
            "empty CoT span. With components_order ending in `cot` the teacher "
            "should reason before emitting <|traj_future_start|>; an empty span "
            "means the prompt pre-filled it (put `cot` LAST in components_order)."
        )
    return lo, hi


@contextmanager
def eager_attention(vlm: Any) -> Iterator[None]:
    """Temporarily switch the text tower to eager attention.

    ``output_attentions=True`` is only honoured by ``eager`` / ``eager_paged`` /
    ``flex_attention``; the teacher runs ``flash_attention_2`` by default, under
    which the flag is silently useless (a ``UserWarning`` and no attentions).  Both
    the top-level and the text sub-config are switched, because
    ``check_model_inputs`` inspects both before allowing the capture.
    """
    configs = [vlm.config]
    text_config = getattr(vlm.config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    previous = [getattr(cfg, "_attn_implementation", None) for cfg in configs]
    try:
        for cfg in configs:
            cfg._attn_implementation = "eager"
        yield
    finally:
        for cfg, was in zip(configs, previous):
            if was is not None:
                cfg._attn_implementation = was


@contextmanager
def capture_pre_rope_kv(
    text_model: Any, lo: int, hi: int, n_kv_heads: int, head_dim: int
) -> Iterator[dict[int, dict[str, torch.Tensor]]]:
    """Collect pre-RoPE K and V over columns ``[lo, hi)`` for every layer.

    ``k_norm``'s output is ``[B, T, n_kv_heads, head_dim]`` — the transpose to
    ``[B, H, T, D]`` and the rotation both happen after it — and ``v_proj``'s output
    is still flat ``[B, T, n_kv_heads * head_dim]``.  Slicing inside the hook keeps
    only the CoT span, so the capture costs kilobytes per layer instead of holding
    36 full-sequence copies.
    """
    store: dict[int, dict[str, torch.Tensor]] = {}
    handles = []

    def make_k_hook(idx: int):
        def hook(_m: Any, _a: Any, out: torch.Tensor) -> None:
            store.setdefault(idx, {})["k"] = out[:, lo:hi].detach().permute(0, 2, 1, 3)

        return hook

    def make_v_hook(idx: int):
        def hook(_m: Any, _a: Any, out: torch.Tensor) -> None:
            shaped = out.view(out.shape[0], out.shape[1], n_kv_heads, head_dim)
            store.setdefault(idx, {})["v"] = shaped[:, lo:hi].detach().permute(0, 2, 1, 3)

        return hook

    for idx, layer in enumerate(text_model.layers):
        handles.append(layer.self_attn.k_norm.register_forward_hook(make_k_hook(idx)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(idx)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def _stack_capture(store: dict[int, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """``{layer: {"k","v"}}`` (each ``[1, H, N, D]``) -> two ``[L, H, N, D]`` tensors."""
    layers = sorted(store)
    if not layers:
        raise RuntimeError("no K/V captured: the k_norm / v_proj hooks never fired")
    k = torch.cat([store[i]["k"] for i in layers], dim=0)
    v = torch.cat([store[i]["v"] for i in layers], dim=0)
    return k, v


def importance_from_attentions(
    attentions: tuple[torch.Tensor, ...],
    lo: int,
    hi: int,
    n_kv_heads: int,
) -> torch.Tensor:
    """Per-(layer, kv-head) attention mass on each CoT token. -> ``[L, H, N_C]``.

    Each entry of ``attentions`` is ``[B, n_q_heads, N_A, N_keys]``, i.e. *after*
    ``repeat_kv``, so query head ``i`` belongs to kv head ``i // n_rep``.  KAVA's GQA
    footnote calls for a MaxPool over each query group before scoring — several
    queries share one cached KV pair, and a pair matters if *any* of them needs it —
    so the group max comes first and the mean over answer tokens second.
    """
    per_layer = []
    for attn in attentions:
        n_q = attn.shape[1]
        n_rep = n_q // n_kv_heads
        grouped = attn.view(attn.shape[0], n_kv_heads, n_rep, attn.shape[2], attn.shape[3])
        pooled = grouped.max(dim=2).values  # [B, H, N_A, N_keys]
        per_layer.append(pooled[..., lo:hi].mean(dim=-2).float())  # [1, H, N_C]
    return torch.cat(per_layer, dim=0)  # [L, H, N_C]


@torch.no_grad()
def extract_teacher_kv(
    model: Any,
    tokenized_data: dict[str, Any],
    ego_history_xyz: torch.Tensor | None = None,
    ego_history_rot: torch.Tensor | None = None,
    ego_future_xyz: torch.Tensor | None = None,
    ego_future_rot: torch.Tensor | None = None,
    mode: str = "generate",
    importance_source: str = "vlm_post_cot",
    store_redundancy: bool = True,
    max_new_tokens: int | None = None,
    do_sample: bool = False,
    expert_timesteps: tuple[float, ...] | None = None,
    expert_noise_seed: int = 0,
    **kwargs: Any,
) -> TeacherKVSample:
    """Capture one sample's CoT KV cache and eviction scores. Batch size 1.

    Args:
        model: a ``DistillReasoningVLA`` (or subclass) holding the teacher VLM.
        tokenized_data: the collated batch's ``tokenized_data``.
        mode: ``"generate"`` lets the teacher produce its own CoT (works on any clip
            and is what LCDrive needs, since it ships no ground-truth reasoning);
            ``"teacher_force"`` uses the CoT already in the prompt.
        importance_source: ``"vlm_post_cot"`` scores CoT tokens by the attention the
            post-CoT tokens pay them; ``"expert"`` scores them by the action expert's
            cross-attention — the queries that actually read this cache at inference,
            and the most faithful signal, but it needs an expert-carrying teacher
            (``KaVaExpertTeacher``, config ``teacher_ar1_5_10b_expert``);
            ``"none"`` skips importance entirely, leaving diversity-only eviction.
        store_redundancy: also compute and return the redundancy score.  It is a
            pure function of the keys, so it can be recomputed later; storing it
            makes recompression deterministic for ~150 KiB per sample.
        expert_timesteps: flow grid to average the expert score over; None uses
            :data:`~alpamayo1_5_distill.models.expert_teacher.DEFAULT_EXPERT_TIMESTEPS`.
        expert_noise_seed: seed for the expert's noise draw, so the score is
            reproducible.
    """
    if importance_source not in IMPORTANCE_SOURCES:
        raise ValueError(f"unknown importance_source {importance_source!r}")
    if importance_source == "expert" and not hasattr(model, "expert"):
        raise ValueError(
            "importance_source='expert' needs a teacher carrying the action expert. "
            "Use teacher=teacher_ar1_5_10b_expert (KaVaExpertTeacher); the plain "
            "teacher_ar1_5_10b config loads a VLM-only class with no expert weights, "
            "action space or diffusion."
        )

    tokenized_data = dict(tokenized_data)
    special = model.special_token_ids
    cot_start_id = int(special["cot_start"])
    cot_end_id = int(special["cot_end"])
    tfs_id = int(special["traj_future_start"])

    if mode == "generate":
        sequence, prompt_len = model.generate_cot_prefix(
            tokenized_data,
            ego_history_xyz=ego_history_xyz,
            ego_history_rot=ego_history_rot,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        cot_text_ids = sequence[0, prompt_len:]
    else:
        raw_ids = tokenized_data.pop("input_ids")
        assert raw_ids.shape[0] == 1, "extract_teacher_kv expects batch size 1"
        sequence = model.fuse_traj_tokens(
            raw_ids,
            {
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
                "ego_future_xyz": ego_future_xyz,
                "ego_future_rot": ego_future_rot,
            },
        )
        cot_text_ids = None

    lo, hi = find_cot_span(sequence, cot_start_id, cot_end_id, tfs_id)
    if cot_text_ids is None:
        cot_text_ids = sequence[0, lo:hi]
    cot_text = model.tokenizer.decode(cot_text_ids, skip_special_tokens=False)

    # The answer segment: everything from the end of the CoT to <traj_future_start>
    # inclusive. In `generate` mode `sequence` was truncated right after the tfs
    # token, so this is 1-2 tokens; in `teacher_force` mode the trajectory follows,
    # and cropping there keeps the answer queries to the handoff point the expert
    # actually starts from.
    tfs_positions = (sequence[0] == tfs_id).nonzero(as_tuple=True)[0]
    if len(tfs_positions) == 0:
        raise ValueError("no <traj_future_start> in the sequence; cannot locate the handoff")
    tfs_col = int(tfs_positions[-1] if mode != "generate" else tfs_positions[0])
    answer_end = tfs_col + 1

    n_kv_heads, head_dim = _kv_geometry(model)
    text_model = model.vlm.model.language_model
    fwd_kwargs = {
        k: v for k, v in tokenized_data.items() if k not in ("attention_mask", "input_ids")
    }

    # -- segment A: [0, hi) with the pre-RoPE hooks; keep the cache ---------------
    prefix_ids = sequence[:, :hi]
    with capture_pre_rope_kv(text_model, lo, hi, n_kv_heads, head_dim) as store:
        prefix_out = model.vlm(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            **fwd_kwargs,
        )
    k_pre, v = _stack_capture(store)

    # M-RoPE ids for the answer segment, taken from the FULL sequence — a naive
    # `arange` here is silently ~27% wrong, because vision tokens occupy many
    # sequence slots but few position steps (validate_reasoning_slots check S5).
    position_ids, _ = model.vlm.model.get_rope_index(
        sequence[:, :answer_end], tokenized_data.get("image_grid_thw"), None, None
    )

    # -- segment B: [hi, tfs] against that cache, eager, with attentions ---------
    answer_ids = sequence[:, hi:answer_end]
    answer_embeds = model.vlm.get_input_embeddings()(answer_ids)
    want_attn = importance_source == "vlm_post_cot"
    attn_ctx = eager_attention(model.vlm) if want_attn else nullcontext()
    with attn_ctx:
        answer_out = text_model(
            inputs_embeds=answer_embeds,
            attention_mask=torch.ones((1, answer_end), dtype=torch.long, device=sequence.device),
            position_ids=position_ids[:, :, hi:answer_end],
            past_key_values=prefix_out.past_key_values,
            use_cache=True,
            cache_position=torch.arange(hi, answer_end, device=sequence.device),
            output_attentions=want_attn,
            output_hidden_states=True,
        )

    # last_hidden_state is post-final-norm, i.e. the same quantity as
    # `hidden_states[-1]` in the existing single-vector cache. It is not bit-identical
    # to that cache when importance is on, because this segment runs eager attention
    # while the prefix ran flash — irrelevant for a distillation target, but worth
    # knowing before comparing two cache runs element-wise.
    tfs_hidden = answer_out.last_hidden_state[0, -1].detach().float().cpu()

    # Every layer at the same column, for the CoDI-style all-layer objective. HF returns
    # L+1 entries with index 0 the embedding output, which is what CoDI iterates over
    # (`zip(outputs.hidden_states, ref_outputs.hidden_states)`) and divides by, so keep
    # the embedding row rather than trimming it.
    #
    # ⚠️ hidden_states[-1] is NOT last_hidden_state for every architecture — some apply
    # the final norm after collecting the tuple. Assert instead of assuming, since a
    # silent half-layer offset would corrupt every target in the cache.
    tfs_hidden_all = None
    if answer_out.hidden_states:
        stacked = torch.stack([h[0, -1] for h in answer_out.hidden_states], dim=0)
        if not torch.allclose(stacked[-1].float(), answer_out.last_hidden_state[0, -1].float()):
            raise RuntimeError(
                "hidden_states[-1] != last_hidden_state at the tfs column, so this model "
                "collects hidden states before the final norm. The all-layer CoDI target "
                "would then be off by one normalisation; fix the indexing before caching."
            )
        tfs_hidden_all = stacked.detach().float().cpu()

    imp = None
    if want_attn:
        attentions = answer_out.attentions
        if not attentions:
            raise RuntimeError(
                "output_attentions returned nothing; the eager switch did not take "
                "effect. Check transformers' attention dispatch for this model."
            )
        imp = importance_from_attentions(attentions, lo, hi, n_kv_heads)
    elif importance_source == "expert":
        # The cache now holds exactly [prompt + CoT + <tfs>] — the same object
        # TrainableAlpamayoR1.forward hands the expert after cropping at
        # future_start_idx + 1 — so the expert can read it as-is.
        from alpamayo1_5_distill.models.expert_teacher import (
            DEFAULT_EXPERT_TIMESTEPS,
            expert_cot_importance,
        )

        imp = expert_cot_importance(
            model,
            prefix_out.past_key_values,
            lo,
            hi,
            crop_len=answer_end,
            traj_data={
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
                "ego_future_xyz": ego_future_xyz,
                "ego_future_rot": ego_future_rot,
            },
            rope_deltas=prefix_out.rope_deltas,
            timesteps=tuple(expert_timesteps or DEFAULT_EXPERT_TIMESTEPS),
            seed=expert_noise_seed,
        )

    red = redundancy_score(k_pre.float()) if store_redundancy else None

    return TeacherKVSample(
        k_pre=k_pre.cpu(),
        v=v.cpu(),
        imp=None if imp is None else imp.cpu(),
        red=None if red is None else red.cpu(),
        tfs_hidden=tfs_hidden,
        tfs_hidden_all=tfs_hidden_all,
        cot_text=cot_text,
        n_cot=hi - lo,
    )


def _kv_geometry(model: Any) -> tuple[int, int]:
    cfg = getattr(model.vlm.config, "text_config", model.vlm.config)
    return int(cfg.num_key_value_heads), int(cfg.head_dim)
