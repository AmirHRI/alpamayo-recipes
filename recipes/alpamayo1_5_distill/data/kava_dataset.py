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

"""Dataset and collator for KAVA KV-cache distillation.

Two pieces:

:class:`KaVaPAIDataset`
    ``DistillPAIDataset`` plus the compressed teacher KV target, read one sample at
    a time from the tiered store in :mod:`.kv_cache_io` and zero-padded to the
    latent budget so the shared ``basic_collation_fn`` can stack it.

:class:`KaVaCollator`
    Splices the latent slots into the tokenized batch.  The slots must land
    **before** ``<|traj_future_start|>``, because the action expert is handed the
    VLM KV cache cropped at ``future_start_idx + 1`` — slots appended at the end of
    the sequence (as in ``reasoning-setup-2b.md`` §9's harness) would fall outside
    that crop and the expert would never see them.

The resulting student sequence mirrors the teacher's ``components_order``
(``[image, traj_history, prompt, cot, traj_future]``) with the text CoT replaced by
``K`` continuous slots::

    [image][traj_history][prompt][<|cot_start|> Q_1 .. Q_K <|cot_end|>][<|tfs|> traj ..]

Only ``input_ids`` placeholders are inserted here; the *embeddings* are swapped in
by a forward hook on ``embed_tokens`` inside the model (see
:mod:`alpamayo1_5_distill.models.kava_model`).  Keeping ``input_ids`` intact is
required — Qwen3-VL reads it twice, once for ``get_rope_index`` (so the slots get
correct M-RoPE for free) and once for ``get_placeholder_mask`` (vision
``masked_scatter`` plus the deepstack injections).  Passing only ``inputs_embeds``
silently loses both.
"""

from typing import Any

import torch

from alpamayo.processor.qwen_processor import QwenProcessor
from alpamayo1_5_distill.data import kv_cache_io
from alpamayo1_5_distill.data.distill_dataset import DistillPAIDataset

#: Default latent-slot placeholder: ``'.'``.  Validated collision-free in
#: ``validate_reasoning_slots.py`` check S0 — it is not in ``all_special_ids`` and
#: not one of the image/video/vision_start/vision_end ids (151655/151656/151652/
#: 151653) that ``get_placeholder_mask`` and the deepstack path key on.  A token
#: that is special for some *other* reason would fail the same silent way the
#: M-RoPE bug does.
DEFAULT_SLOT_PLACEHOLDER_ID = 13


def pad_kv_to_budget(
    k: torch.Tensor, v: torch.Tensor, num_slots: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-pad ``[L, H, M', D]`` up to ``num_slots`` and return the validity mask.

    ``M' = min(num_slots, N_C)`` can be **less** than the budget: real driving CoT
    traces run ~40 tokens and the teacher is capped at 128, so at ``M=32`` a short
    trace leaves slots with no target.  KAVA never meets this case (its traces are
    always longer than the budget); here the surplus slots must be masked out of the
    loss rather than trained towards zero.
    """
    m_eff = k.shape[-2]
    if m_eff > num_slots:
        raise ValueError(f"cached KV has {m_eff} slots, more than the budget {num_slots}")
    valid = torch.zeros(num_slots, dtype=torch.bool)
    valid[:m_eff] = True
    if m_eff == num_slots:
        return k, v, valid
    pad = (0, 0, 0, num_slots - m_eff)  # pad the token axis (second-to-last)
    return (
        torch.nn.functional.pad(k, pad),
        torch.nn.functional.pad(v, pad),
        valid,
    )


class KaVaPAIDataset(DistillPAIDataset):
    """LCDrive dataset that attaches the compressed teacher KV cache per sample.

    Reuses ``DistillPAIDataset._sample_key`` verbatim, so the offline teacher run
    and this loader agree sample-for-sample as long as they iterate the same
    dataset config — which the PAI path makes deterministic (no augmentation, no
    random frame sampling, constant ``DEFAULT_T0_US``).

    Args:
        kv_cache_root: the ``cache_root`` written by ``generate_teacher_kv.py``.
            None disables the KV target entirely, degrading to plain
            ``DistillPAIDataset`` behaviour (useful for the no-KD control arm).
        kv_tier: compressed tier directory name, e.g. ``compressed_M16_rkv0.1``.
            Build it with :func:`kv_cache_io.compressed_tag`.
        num_slots: the latent budget ``M``; cached entries are padded up to it.
        attach_tfs_hidden: also surface the ``<traj_future_start>`` hidden that the
            same cache run recorded, so the existing single-vector loss can be kept
            alongside ``L_KV`` (KAVA's ``lambda_1`` term).
        allow_missing: give clips with no cache entry an all-masked target (CE only)
            instead of raising.  Off by default — a silent hole in the cache would
            train part of an epoch with no KV supervision and look like a mysteriously
            weak result.  Turn it ON for the LCDrive build, where 4 of 38,340 clips
            have no handoff token and therefore no cache entry.
    """

    def __init__(
        self,
        *args: Any,
        kv_cache_root: str | None = None,
        kv_tier: str | None = None,
        num_slots: int = 16,
        attach_tfs_hidden: bool = True,
        allow_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kv_cache_root = kv_cache_root
        self.kv_tier = kv_tier
        self.num_slots = int(num_slots)
        self.attach_tfs_hidden = attach_tfs_hidden
        self.allow_missing = allow_missing
        # Teacher KV geometry, refreshed from every successful load. Seeded with the
        # Alpamayo-1.5-10B values so a miss on the very first sample still produces a
        # correctly-shaped empty target.
        self._teacher_layers, self._teacher_kv_heads, self._head_dim = 36, 8, 128
        if kv_cache_root is not None and kv_tier is None:
            raise ValueError("kv_cache_root given without kv_tier; pass the compressed tier name")

    def _attach_empty_kv(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Attach an all-masked KV target, so an uncached clip trains on CE alone.

        Returning ``None`` instead would be worse than useless: the shared collator
        stacks tensors and has no notion of a dropped sample, so one uncached clip
        would take down the step. A zero target behind an all-False ``valid`` mask is
        already handled by ``kv_matching_loss`` (it contributes nothing to either the
        numerator or the denominator), so the sample simply trains without ``L_KV``.

        On the LCDrive build this affects **4 of 38,340 clips** (0.01%) — ones where
        the teacher hit ``max_new_tokens`` without ever emitting
        ``<|traj_future_start|>``, so there was no handoff point to cache.
        """
        shape = (self._teacher_layers, self._teacher_kv_heads, self.num_slots, self._head_dim)
        sample["teacher_kv_k"] = torch.zeros(shape, dtype=torch.bfloat16)
        sample["teacher_kv_v"] = torch.zeros(shape, dtype=torch.bfloat16)
        sample["teacher_kv_valid"] = torch.zeros(self.num_slots, dtype=torch.bool)
        return sample

    def _attach_teacher_kv(self, sample: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
        if sample is None or self.kv_cache_root is None:
            return sample

        names = ("k_pre", "v", "tfs_hidden") if self.attach_tfs_hidden else ("k_pre", "v")
        try:
            entry = kv_cache_io.load_entry(self.kv_cache_root, self.kv_tier, key, names=names)
        except KeyError:
            if self.allow_missing:
                return self._attach_empty_kv(sample)
            raise

        layers, heads, _, head_dim = entry["k_pre"].shape
        self._teacher_layers, self._teacher_kv_heads, self._head_dim = layers, heads, head_dim
        k, v, valid = pad_kv_to_budget(entry["k_pre"], entry["v"], self.num_slots)
        sample["teacher_kv_k"] = k
        sample["teacher_kv_v"] = v
        sample["teacher_kv_valid"] = valid
        if self.attach_tfs_hidden and "tfs_hidden" in entry:
            sample["teacher_tfs_hidden"] = entry["tfs_hidden"]
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        key = self._sample_key(idx)
        sample = super().__getitem__(idx)
        return self._attach_teacher_kv(sample, key)


def splice_slot_placeholders(
    input_ids: torch.Tensor,
    tfs_id: int,
    cot_start_id: int,
    cot_end_id: int,
    placeholder_id: int,
    num_slots: int,
    attention_mask: torch.Tensor | None = None,
    labels_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Insert ``<|cot_start|> [placeholder * K] <|cot_end|>`` before each row's tfs token.

    Batches are **left**-padded, so the ``<|traj_future_start|>`` column differs per
    row and the insert has to be located by value per row, not at a fixed index.
    The loop over the batch is deliberate: batch sizes here are single digits, and a
    vectorised gather would be harder to verify against the index arithmetic.

    Returns a dict with the widened ``input_ids`` / ``attention_mask`` /
    ``labels_mask`` (each ``K + 2`` columns longer) and ``slot_pos`` ``[B, K]``, the
    columns the slot embeddings must be written to.  ``slot_pos`` is returned rather
    than recomputed downstream because the model must index by an **explicit**
    position tensor — ``out[:, -K:, :]`` holds only for a single prefill pass and
    breaks the moment a forward covers just the new tokens against a cache.
    """
    b, length = input_ids.shape
    if num_slots <= 0:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_mask": labels_mask,
            "slot_pos": input_ids.new_zeros((b, 0)),
        }

    mask = input_ids == tfs_id
    cols = torch.arange(length, device=input_ids.device)
    masked = torch.where(mask, cols.unsqueeze(0), torch.full_like(input_ids, -1))
    tfs_col = masked.max(dim=1).values
    if (tfs_col < 0).any():
        missing = int((tfs_col < 0).sum())
        raise ValueError(
            f"{missing} sample(s) in the batch have no <traj_future_start> token; "
            "cannot place the latent slots. Check that the student's vla_processor "
            "has `traj_future` in components_order."
        )

    extra = num_slots + 2
    width = length + extra
    new_ids = input_ids.new_zeros((b, width))
    new_attn = None if attention_mask is None else attention_mask.new_zeros((b, width))
    new_labels = None if labels_mask is None else labels_mask.new_zeros((b, width))
    slot_pos = input_ids.new_zeros((b, num_slots))

    for row in range(b):
        c = int(tfs_col[row])
        new_ids[row, :c] = input_ids[row, :c]
        new_ids[row, c] = cot_start_id
        new_ids[row, c + 1 : c + 1 + num_slots] = placeholder_id
        new_ids[row, c + 1 + num_slots] = cot_end_id
        new_ids[row, c + extra :] = input_ids[row, c:]
        slot_pos[row] = torch.arange(c + 1, c + 1 + num_slots, device=input_ids.device)

        if new_attn is not None:
            new_attn[row, :c] = attention_mask[row, :c]
            new_attn[row, c : c + extra] = 1  # slots and their delimiters are real tokens
            new_attn[row, c + extra :] = attention_mask[row, c:]

        if new_labels is not None:
            new_labels[row, :c] = labels_mask[row, :c]
            # No CE on the slots or their delimiters: the student is text-silent and
            # the slot embeddings are injected, never generated.
            new_labels[row, c : c + extra] = False
            new_labels[row, c + extra :] = labels_mask[row, c:]

    return {
        "input_ids": new_ids,
        "attention_mask": new_attn,
        "labels_mask": new_labels,
        "slot_pos": slot_pos,
    }


class KaVaCollator:
    """Collate a batch, then splice the latent slots into it.

    Unlike ``collate_fn_from_model_config``, which rebuilds a ``QwenProcessor``
    (``AutoProcessor.from_pretrained`` + 4000 added tokens) on **every call**, this
    builds it once in ``__init__``.  Hydra instantiates it as an object rather than
    a ``_partial_``, so the cost is paid once per dataloader worker.

    Args:
        model_config: the Alpamayo config, supplied by ``train_hf``'s
            ``hyu.instantiate(..., model_config=model.config)``.
        num_slots: ``K`` latent slots to splice in; 0 disables splicing, which is
            the Stage-0 / no-slot control arm.
        placeholder_id: the filler token id occupying the slot positions.
    """

    def __init__(
        self,
        model_config: Any = None,
        num_slots: int = 16,
        placeholder_id: int = DEFAULT_SLOT_PLACEHOLDER_ID,
        padding_side: str = "left",
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
        chat_template_version: str = "r1_5",
    ) -> None:
        self.num_slots = int(num_slots)
        self.placeholder_id = int(placeholder_id)
        self.padding_side = padding_side
        self._proc = QwenProcessor(
            vlm_name_or_path=model_config.vlm_name_or_path,
            traj_vocab_size=model_config.traj_vocab_size,
            min_pixels=model_config.min_pixels,
            max_pixels=model_config.max_pixels,
            include_camera_ids=include_camera_ids,
            include_frame_nums=include_frame_nums,
            chat_template_version=chat_template_version,
        )
        self._proc.build_processor()
        tokenizer = self._proc.processor.tokenizer
        self.tfs_id = int(tokenizer.convert_tokens_to_ids("<|traj_future_start|>"))
        self.cot_start_id = int(tokenizer.convert_tokens_to_ids("<|cot_start|>"))
        self.cot_end_id = int(tokenizer.convert_tokens_to_ids("<|cot_end|>"))
        self._check_placeholder(tokenizer)

    def _check_placeholder(self, tokenizer: Any) -> None:
        """Fail loudly if the placeholder is special to something else.

        A token that ``get_placeholder_mask`` or the deepstack path keys on would be
        silently reinterpreted as vision, degrading the model without an error —
        the same failure class as the M-RoPE position bug.
        """
        if self.num_slots <= 0:
            return
        reserved = set(getattr(tokenizer, "all_special_ids", []) or [])
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            value = getattr(tokenizer, name, None)
            if value is not None:
                reserved.add(int(value))
        reserved |= {151652, 151653, 151655, 151656}
        if self.placeholder_id in reserved:
            raise ValueError(
                f"slot placeholder id {self.placeholder_id} collides with a special / "
                "vision token; pick a plain text token (default 13 = '.')"
            )

    def __call__(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._proc.collate_fn(data, padding_side=self.padding_side)
        if self.num_slots <= 0:
            return batch

        tokenized = batch["tokenized_data"]
        spliced = splice_slot_placeholders(
            tokenized["input_ids"],
            tfs_id=self.tfs_id,
            cot_start_id=self.cot_start_id,
            cot_end_id=self.cot_end_id,
            placeholder_id=self.placeholder_id,
            num_slots=self.num_slots,
            attention_mask=tokenized.get("attention_mask"),
            labels_mask=batch.get("labels_mask"),
        )
        tokenized["input_ids"] = spliced["input_ids"]
        if spliced["attention_mask"] is not None:
            tokenized["attention_mask"] = spliced["attention_mask"]
        if spliced["labels_mask"] is not None:
            batch["labels_mask"] = spliced["labels_mask"]
        batch["slot_pos"] = spliced["slot_pos"]
        return batch
