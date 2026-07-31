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

"""KAVA: latent reasoning supervised by the teacher's compressed KV cache.

``KaVaReasoningVLA`` extends :class:`DistillReasoningVLA` with what the existing
single-vector objective could not reach.  That objective matches
``hidden_states[-1]`` at ``<traj_future_start>`` — one post-final-RMSNorm vector,
which is *not in the KV cache at all* (K/V at layer l are projected from layer l's
*input*).  The action expert, meanwhile, is handed the whole KV cache cropped at
``future_start_idx + 1``, layer for layer.  KAVA supervises exactly that object:

* ``K`` continuous **latent slots** stand where the teacher's text CoT stood,
* their per-layer, per-head **K and V** are matched against the teacher's CoT cache
  after redundancy/importance-aware eviction down to ``K`` entries,
* so the reasoning is compiled into the cache the expert reads, and the student
  never emits a reasoning token.

The paper's central finding is what makes this work across two different models: a
compressed cache has lost token correspondence, and that is fine — continuous
latents can absorb structure that token- or hidden-level matching cannot express.

Three implementation facts worth knowing before editing this file:

**Keys are matched pre-RoPE.** The capture hooks read ``k_norm`` (pre-rotation)
rather than the ``DynamicCache`` (post-rotation).  Matching post-RoPE keys would
force the student's slots to reproduce the teacher's rotation phase at positions
that differ and that eviction has scrambled.

**Gradient checkpointing must be off.** It silently sets ``use_cache=False``
(``transformers/utils/generic.py``) and, worse, detaches anything a forward hook
captured on the first pass, so ``L_KV`` would compute a finite number and train
nothing.  :meth:`_assert_capture_possible` refuses to run instead.

**Slot positions come from the collator, per row.** Never index the injected
embeddings with ``[-K:]`` — that holds only for a single prefill pass and breaks
the moment a forward covers just the new tokens against a cache, which is exactly
what the Jacobi refinement path does.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo1_5_distill.models.distill_base_model import DistillReasoningVLA
from alpamayo1_5_distill.models.kv_distill import (
    KVProjectorBank,
    build_layer_map,
    kv_matching_loss,
)


@dataclass
class KaVaVLAOutput(ModelOutput):
    """Output of :class:`KaVaReasoningVLA`.

    Everything but ``loss`` is detached and exists for logging — ``KaVaTrainer``
    picks these up so the three objectives are separable in the run history rather
    than hidden inside one total.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    ce_loss: torch.FloatTensor | None = None
    latent_loss: torch.FloatTensor | None = None
    kv_loss: torch.FloatTensor | None = None
    n_valid_slots: torch.FloatTensor | None = None


class KaVaReasoningVLA(DistillReasoningVLA):
    """Stage-1 student with latent slots supervised by a compressed teacher KV cache.

    All KAVA state is created by :meth:`init_kava`; instances built through
    inherited factories that skip it (notably the offline teacher loaded via
    ``from_alpamayo_checkpoint``) stay in a well-defined inert state where
    ``forward`` is exactly the parent's.

    As with ``latent_proj`` in the parent, the submodules here are deliberately not
    class attributes — a class attribute would shadow ``nn.Module.__getattr__`` and
    always read ``None``.  Reach them with ``getattr(self, name, None)``.
    """

    num_slots: int = 0
    jacobi_iters: int = 1
    kv_loss_weight: float = 1.0
    kv_loss_type: str = "smooth_l1"
    kv_layerwise_std: bool = False
    kv_align: str = "projector"
    kv_layer_map: list[int] | None = None

    #: When True, ``forward`` also stashes the loss terms **with their graph attached**
    #: in :attr:`last_loss_terms`.  ``KaVaTrainer`` switches this on periodically so it
    #: can take ``autograd.grad`` of each term separately and report how much of the
    #: backbone gradient each objective actually contributes — the check that catches a
    #: silently inert term.  Off by default: keeping the references alive would pin the
    #: graph for longer than the training step needs.
    keep_loss_terms: bool = False

    # ------------------------------------------------------------------ setup
    def init_kava(
        self,
        num_slots: int = 16,
        teacher_layers: int = 36,
        jacobi_iters: int = 1,
        kv_loss_weight: float = 1.0,
        kv_loss_type: str = "smooth_l1",
        kv_layerwise_std: bool = False,
        kv_align: str = "projector",
        kv_layer_map: list[int] | None = None,
        slot_init: str = "vocab",
        slot_init_cot_text: str | None = None,
        slot_init_std: float = 0.02,
    ) -> None:
        """Create the slot embeddings, the projector bank and the Jacobi projection.

        Args:
            num_slots: ``K`` latent slots replacing the text CoT.  Must match the
                collator's ``num_slots`` and the cached tier's ``M``.
            teacher_layers: depth of the teacher whose cache was recorded (36 for
                Alpamayo-1.5-10B), used to build the student->teacher layer map.
            jacobi_iters: ``T`` passes producing the latents (PCCoT).  ``T=1`` means
                the slots ride the single prefill pass — free, given the causal mask.
            kv_loss_weight: ``lambda_2`` on ``L_KV``.
            kv_loss_type / kv_layerwise_std / kv_align: see
                :func:`~alpamayo1_5_distill.models.kv_distill.kv_matching_loss` and
                :class:`~alpamayo1_5_distill.models.kv_distill.KVProjectorBank`.
            kv_layer_map: explicit student->teacher layer list; None uses a uniform
                stride with endpoints preserved.
            slot_init: ``"vocab"`` seeds the slots from real token embeddings,
                ``"randn"`` from ``N(0, slot_init_std^2)``.  Vocab-init is measured
                to win on this exact model (``reasoning-setup-2b.md`` §9.1 C3: 3.71
                vs 4.98 starting loss, ~3x faster convergence), reproducing Lester
                et al. (2021) below ~10B.
            slot_init_cot_text: a ``cot_text.*.jsonl`` path (or its directory) from
                the teacher cache.  With ``slot_init="vocab"`` the slots are seeded
                from the most frequent tokens in the teacher's own reasoning, so they
                start in the region of embedding space the CoT they must internalise
                actually occupies.  Falls back to the most frequent tokens overall
                (by id order) with a printed warning if absent.
        """
        hidden = self._student_hidden_size()
        self.num_slots = int(num_slots)
        self.jacobi_iters = max(1, int(jacobi_iters))
        self.kv_loss_weight = float(kv_loss_weight)
        self.kv_loss_type = str(kv_loss_type)
        self.kv_layerwise_std = bool(kv_layerwise_std)
        self.kv_align = str(kv_align)

        n_student = self._n_student_layers()
        self.kv_layer_map = build_layer_map(n_student, int(teacher_layers), kv_layer_map)

        ref = next(self.vlm.parameters())
        self.slot_embeddings = nn.Parameter(
            torch.empty(self.num_slots, hidden, device=ref.device, dtype=ref.dtype)
        )
        self._init_slot_values(slot_init, slot_init_cot_text, slot_init_std)

        kv_width = self._kv_width()
        self.kv_projector = KVProjectorBank(n_student, kv_width=kv_width, align=self.kv_align)
        self.kv_projector.to(device=ref.device, dtype=ref.dtype)

        if self.jacobi_iters > 1:
            self.jacobi_proj = self._build_jacobi_proj(hidden)
            self.jacobi_proj.to(device=ref.device, dtype=ref.dtype)

        self._capture: dict[int, dict[str, torch.Tensor]] = {}

    def _build_jacobi_proj(self, hidden: int) -> nn.Module:
        """The PCCoT projection: slot output hidden -> next-iteration input embedding.

        Normalise first.  Qwen3-VL hidden states reach magnitudes ~1350 ("massive
        activations") while token embeddings sit at ~3.19e-2 RMS, so feeding hidden
        states straight back in — Coconut-style, i.e. an identity init — starts the
        refinement four orders of magnitude off-manifold.  The RMSNorm removes the
        scale entirely and the linear is initialised so its output RMS matches the
        model's *measured* embedding RMS, rather than a guessed constant.
        """
        emb_weight = self.vlm.get_input_embeddings().weight
        target_rms = float(emb_weight.detach().float().pow(2).mean().sqrt())
        proj = nn.Sequential(
            nn.RMSNorm(hidden, eps=1e-6),
            nn.Linear(hidden, hidden, bias=False),
        )
        nn.init.normal_(proj[1].weight, std=target_rms / (hidden**0.5))
        return proj

    def _init_slot_values(self, slot_init: str, cot_text: str | None, std: float) -> None:
        """Fill ``slot_embeddings`` in place, from the vocabulary or from noise."""
        emb = self.vlm.get_input_embeddings().weight
        if slot_init == "randn":
            with torch.no_grad():
                self.slot_embeddings.normal_(0.0, std)
            return
        if slot_init != "vocab":
            raise ValueError(f"unknown slot_init {slot_init!r}")

        token_ids = self._vocab_init_token_ids(cot_text)
        with torch.no_grad():
            self.slot_embeddings.copy_(emb[token_ids].to(self.slot_embeddings.dtype))

    def _vocab_init_token_ids(self, cot_text: str | None) -> torch.Tensor:
        """Most frequent token ids in the teacher's CoT, one per slot."""
        from collections import Counter
        from pathlib import Path
        import json

        counter: Counter = Counter()
        if cot_text is not None:
            path = Path(cot_text)
            files = sorted(path.glob("cot_text.*.jsonl")) if path.is_dir() else [path]
            for file in files:
                if not file.exists():
                    continue
                with file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        text = json.loads(line).get("cot_text", "")
                        if text:
                            counter.update(
                                self.tokenizer(text, add_special_tokens=False).input_ids
                            )

        if not counter:
            print(
                "[kava] no CoT text available for vocab-init "
                f"(slot_init_cot_text={cot_text!r}); seeding slots from the first "
                f"{self.num_slots} vocabulary entries instead. Point "
                "slot_init_cot_text at the cache's cot_text.*.jsonl for the "
                "measured init (reasoning-setup-2b.md 9.1 C3).",
                flush=True,
            )
            return torch.arange(self.num_slots)

        ranked = [tid for tid, _ in counter.most_common()]
        if len(ranked) < self.num_slots:  # cycle rather than pad with a constant
            ranked = (ranked * (self.num_slots // len(ranked) + 1))[: self.num_slots]
        return torch.tensor(ranked[: self.num_slots], dtype=torch.long)

    @classmethod
    def from_pretrained_vlm(
        cls,
        vlm_name_or_path: str,
        alpamayo_config_path: str | None = None,
        checkpoint_path: str | None = None,
        teacher_hidden_dim: int | None = None,
        latent_loss_weight: float = 1.0,
        latent_cosine_weight: float = 0.1,
        kava: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "KaVaReasoningVLA":
        """Build the student, the parent's latent projector, then the KAVA state.

        ``kava`` is the kwarg block forwarded to :meth:`init_kava`; omit it to get
        exactly a :class:`DistillReasoningVLA` (the no-slot control arm).
        """
        model = super().from_pretrained_vlm(
            vlm_name_or_path,
            alpamayo_config_path=alpamayo_config_path,
            checkpoint_path=checkpoint_path,
            teacher_hidden_dim=teacher_hidden_dim,
            latent_loss_weight=latent_loss_weight,
            latent_cosine_weight=latent_cosine_weight,
            **kwargs,
        )
        if kava is not None:
            model.init_kava(**kava)
        return model

    # ------------------------------------------------------------- accessors
    def _text_model(self) -> nn.Module:
        return self.vlm.model.language_model

    def _n_student_layers(self) -> int:
        return len(self._text_model().layers)

    def _kv_width(self) -> int:
        """``num_key_value_heads * head_dim`` — 1024 on both teacher and student."""
        cfg = getattr(self.vlm.config, "text_config", self.vlm.config)
        return int(cfg.num_key_value_heads) * int(cfg.head_dim)

    def _kv_shape(self) -> tuple[int, int]:
        cfg = getattr(self.vlm.config, "text_config", self.vlm.config)
        return int(cfg.num_key_value_heads), int(cfg.head_dim)

    def _assert_capture_possible(self) -> None:
        """Refuse to train the KV loss through gradient checkpointing.

        Under checkpointing the first pass runs with grad disabled, so tensors a
        hook captures then are graph-disconnected: ``L_KV`` would be finite,
        decreasing-looking, and supervise nothing.  This is the same silent-failure
        class as the M-RoPE bug, so it fails loudly instead.
        """
        text_model = self._text_model()
        if getattr(text_model, "gradient_checkpointing", False) and self.training:
            raise RuntimeError(
                "KAVA needs gradient_checkpointing=False: checkpointing detaches "
                "hook-captured K/V, so L_KV would train nothing while still looking "
                "finite. Set `trainer.gradient_checkpointing: false` in the student "
                "config (Stage-2 already does), or reduce per-device batch size."
            )

    # ------------------------------------------------------------------ hooks
    @contextmanager
    def _slot_hooks(
        self, slot_pos: torch.Tensor, slots: torch.Tensor, capture: bool
    ) -> Iterator[None]:
        """Inject slot embeddings and (optionally) capture per-layer pre-RoPE K/V.

        Injection runs as a forward hook on ``embed_tokens``, which fires *before*
        the vision ``masked_scatter``, so vision merge and the deepstack injections
        are untouched.  ``input_ids`` therefore stays intact and Qwen3-VL keeps both
        of the things it reads from it: M-RoPE positions and the vision placeholder
        mask.

        Capture slices to the slot columns *inside* the hook: the full-sequence
        activation is retained by autograd anyway (attention consumes it), so the
        slice costs ~4 MB rather than the ~700 MB a full-length copy would.
        """
        handles = []
        rows = torch.arange(slot_pos.shape[0], device=slot_pos.device).unsqueeze(1)

        def inject(_module: nn.Module, _args: Any, out: torch.Tensor) -> torch.Tensor:
            out = out.clone()
            out[rows, slot_pos] = slots.to(out.dtype)
            return out

        handles.append(self.vlm.get_input_embeddings().register_forward_hook(inject))

        if capture:
            self._capture = {}
            n_kv_heads, head_dim = self._kv_shape()

            def make_k_hook(layer_idx: int):
                def hook(_m: nn.Module, _a: Any, out: torch.Tensor) -> None:
                    # k_norm sees [B, T, n_kv_heads, head_dim] (the transpose to
                    # [B, H, T, D] happens after it, in the attention module).
                    sel = out[rows, slot_pos]  # [B, K, H, D]
                    self._capture.setdefault(layer_idx, {})["k"] = sel.permute(0, 2, 1, 3)

                return hook

            def make_v_hook(layer_idx: int):
                def hook(_m: nn.Module, _a: Any, out: torch.Tensor) -> None:
                    # v_proj output is still flat [B, T, n_kv_heads * head_dim].
                    shaped = out.view(out.shape[0], out.shape[1], n_kv_heads, head_dim)
                    sel = shaped[rows, slot_pos]
                    self._capture.setdefault(layer_idx, {})["v"] = sel.permute(0, 2, 1, 3)

                return hook

            for idx, layer in enumerate(self._text_model().layers):
                handles.append(layer.self_attn.k_norm.register_forward_hook(make_k_hook(idx)))
                handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(idx)))

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    # ----------------------------------------------------------------- jacobi
    def _run_jacobi(
        self,
        input_ids: torch.Tensor,
        tokenized_data: dict[str, Any],
        slot_pos: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """PCCoT refinement: ``T-1`` parallel updates of all slots against the prefix.

        Instead of ``K`` sequential decode steps, all slots are updated together for
        ``T-1`` iterations and only then does the caller run the single full forward
        that produces CE, the tfs hidden and the K/V used by ``L_KV`` — the paper's
        "generate the whole student sequence with Jacobi iterations and *then*
        distil".

        Two documented deviations:

        * the prefix cache is built under ``no_grad``, so during refinement the
          gradient reaches the slots but not the backbone through the prefix.  The
          backbone still gets full gradient from the final forward.  This keeps the
          cost at one extra prefill instead of ``T`` differentiable ones.
        * refinement needs one crop point for the whole batch, so slot columns must
          be uniform across the batch.  They are, for every LCDrive config: batches
          are right-aligned by left padding and the span after
          ``<|traj_future_start|>`` is a fixed length, so the tfs column is
          identical in every row.  The check is explicit rather than assumed.

        Returns:
            ``[B, K, H]`` refined slot embeddings.
        """
        cols = slot_pos[0]
        if not torch.equal(slot_pos, cols.unsqueeze(0).expand_as(slot_pos)):
            raise ValueError(
                "jacobi_iters > 1 needs the same slot columns in every row of the "
                "batch (it crops one shared prefix cache). Got per-row differences; "
                "use jacobi_iters=1 or a batch whose rows share a tfs column."
            )
        start = int(cols[0])
        n_slots = int(cols.numel())
        attention_mask = tokenized_data.get("attention_mask")

        # M-RoPE ids for the slots, taken from the FULL spliced sequence. Naive
        # `arange` here is ~27% wrong with no error raised (validate_reasoning_slots
        # check S5): a 78-token prefix can have a max position id of only 21,
        # because vision tokens span many sequence slots but few position steps.
        position_ids, _ = self.vlm.model.get_rope_index(
            input_ids,
            tokenized_data.get("image_grid_thw"),
            None,
            attention_mask,
        )
        slot_positions = position_ids[:, :, start : start + n_slots]
        cache_position = torch.arange(start, start + n_slots, device=input_ids.device)

        with torch.no_grad():
            prefix_kwargs = {k: v for k, v in tokenized_data.items() if k != "attention_mask"}
            prefix = self.vlm(
                input_ids=input_ids[:, :start],
                attention_mask=None if attention_mask is None else attention_mask[:, :start],
                use_cache=True,
                **prefix_kwargs,
            )
        cache = prefix.past_key_values

        if attention_mask is None:
            refine_mask = None
        else:
            ones = attention_mask.new_ones((attention_mask.shape[0], n_slots))
            refine_mask = torch.cat([attention_mask[:, :start], ones], dim=1)

        current = slots.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        text_model = self._text_model()
        for _ in range(self.jacobi_iters - 1):
            out = text_model(
                inputs_embeds=current,
                attention_mask=refine_mask,
                position_ids=slot_positions,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            cache.crop(start)  # roll the cache back so the next iteration re-reads the prefix
            current = self.jacobi_proj(out.last_hidden_state)
        return current

    # ------------------------------------------------------------------- loss
    def _kv_loss(
        self,
        teacher_kv_k: torch.Tensor,
        teacher_kv_v: torch.Tensor,
        teacher_kv_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        student_kv = {
            layer: (captured["k"], captured["v"])
            for layer, captured in self._capture.items()
            if "k" in captured and "v" in captured
        }
        if not student_kv:
            raise RuntimeError(
                "no student K/V captured; the k_norm/v_proj hooks did not fire. "
                "Check that slots are active and gradient checkpointing is off."
            )
        projector = getattr(self, "kv_projector", None)
        return kv_matching_loss(
            student_kv,
            teacher_kv_k,
            teacher_kv_v,
            self.kv_layer_map,
            valid_mask=teacher_kv_valid,
            projector=projector,
            kind=self.kv_loss_type,
            layerwise_std=self.kv_layerwise_std,
        )

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        teacher_tfs_hidden: torch.Tensor | None = None,
        slot_pos: torch.Tensor | None = None,
        teacher_kv_k: torch.Tensor | None = None,
        teacher_kv_v: torch.Tensor | None = None,
        teacher_kv_valid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> KaVaVLAOutput:
        """Stage-1 forward with CE + the endpoint latent loss + ``L_KV``.

        Degrades cleanly: with no slots it is the parent's objective, and with slots
        but no cached KV it is latent slots trained by CE alone (the Stage-0 control
        arm from ``reasoning-setup-2b.md`` §6).
        """
        # Copy before popping: the parent mutates the caller's dict, which makes a
        # second forward over the same batch fail with a bare KeyError. HF Trainer
        # never reuses a batch, but eval loops and diagnostics do.
        tokenized_data = dict(tokenized_data)
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        slots_active = (
            self.num_slots > 0
            and slot_pos is not None
            and slot_pos.numel() > 0
            and getattr(self, "slot_embeddings", None) is not None
        )
        want_kv = slots_active and teacher_kv_k is not None and teacher_kv_v is not None
        if want_kv:
            self._assert_capture_possible()

        if slots_active:
            slot_pos = slot_pos.to(input_ids.device)
            slots = self.slot_embeddings
            if self.jacobi_iters > 1:
                slots = self._run_jacobi(input_ids, tokenized_data, slot_pos, slots)
            else:
                slots = slots.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            with self._slot_hooks(slot_pos, slots, capture=want_kv):
                outputs = self.vlm(
                    input_ids=input_ids,
                    labels=labels,
                    output_hidden_states=True,
                    **tokenized_data,
                )
        else:
            outputs = self.vlm(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                **tokenized_data,
            )

        # Next-token CE, identical split to the parent / TrainableReasoningVLA.
        losses: dict[str, torch.Tensor] = {}
        traj_mask = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        losses["future_traj"] = self._compute_next_token_loss(outputs, labels, traj_mask)
        labels[traj_mask] = IGNORE_INDEX
        losses["others"] = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)
        ce_loss = sum(losses.values())
        total_loss = ce_loss
        attached: dict[str, torch.Tensor] = {"ce": ce_loss}

        # lambda_1: the endpoint hidden match (KAVA's L_CODI analogue).
        latent_loss = None
        latent_proj = getattr(self, "latent_proj", None)
        if teacher_tfs_hidden is not None and latent_proj is not None:
            h_student = latent_proj(self._gather_tfs_hidden(input_ids, outputs.hidden_states[-1]))
            latent_loss = self._latent_loss(h_student, teacher_tfs_hidden.to(h_student.device))
            total_loss = total_loss + self.latent_loss_weight * latent_loss
            attached["latent"] = latent_loss

        # lambda_2: the compressed-KV match.
        kv_loss = None
        n_valid = None
        if want_kv:
            valid = None if teacher_kv_valid is None else teacher_kv_valid.to(input_ids.device)
            kv_loss = self._kv_loss(
                teacher_kv_k.to(input_ids.device), teacher_kv_v.to(input_ids.device), valid
            )
            total_loss = total_loss + self.kv_loss_weight * kv_loss
            attached["kv"] = kv_loss
            n_valid = (
                valid.float().sum(dim=1).mean()
                if valid is not None
                else torch.tensor(float(self.num_slots))
            )
            self._capture = {}  # drop references so the graph is freed with the step

        self.last_loss_terms = attached if self.keep_loss_terms else None

        return KaVaVLAOutput(
            loss=total_loss,
            logits=outputs.logits,
            ce_loss=ce_loss.detach(),
            latent_loss=None if latent_loss is None else latent_loss.detach(),
            kv_loss=None if kv_loss is None else kv_loss.detach(),
            n_valid_slots=None if n_valid is None else n_valid.detach(),
        )
