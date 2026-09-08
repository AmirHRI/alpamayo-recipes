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

"""Trainer for KAVA runs: logs the loss terms, and keeps decay off the soft prompt.

``DistillVLAOutput`` has exposed detached ``ce_loss`` / ``latent_loss`` for logging
since the recipe was written, but nothing ever read them — HF ``Trainer`` only
consumes ``outputs["loss"]``, so the split was invisible in every run so far.  With
a third term (``L_KV``) that stops being a cosmetic gap: "total went down" cannot
tell you whether the KV objective did anything, and a dead ``L_KV`` is precisely the
failure this recipe has to detect.
"""

import os
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler
from transformers.trainer import seed_worker
from transformers.utils import is_datasets_available

if is_datasets_available():
    import datasets

from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer

#: Detached scalars on the model output that get averaged into the training logs.
#: A key absent from a given output type is skipped (``_stash`` ignores ``None``), so this
#: covers ``KaVaVLAOutput`` and ``KDVLAOutput`` without either needing to know about the
#: other. The ``kv_loss_*`` region splits matter because ~93% of positions are vision: a
#: single ``kv_loss`` scalar cannot show whether the vision region is converging at a
#: different rate from text and trajectory, or swamping them. Same reasoning for
#: ``block_loss_mse``/``block_loss_cosine``: the combined ``block_loss`` can fall while
#: only one of magnitude or direction is actually improving.
AUX_LOSS_KEYS = (
    "ce_loss",
    "latent_loss",
    "kv_loss",
    "n_valid_slots",
    "kd_loss",
    "kv_loss_vision",
    "kv_loss_text",
    "kv_loss_traj",
    "block_loss",
    "block_loss_mse",
    "block_loss_cosine",
    "block_loss_tf",
    "block_loss_span",
    "block_loss_early",
    "block_loss_mid",
    "block_loss_deep",
    "freerun_loss",
    "field_loss",
    "roll_loss",
    "kv_ratio_k",
    "kv_ratio_v",
    # Consistency distillation. `cd_loss` is the total; the per-rung terms are what
    # distinguish a healthy bootstrap (the anchor rung falls first, then propagates toward
    # noise) from a collapse to the conditional mean (the noise end falls fastest).
    "cd_loss",
    "x0_gt_loss",
    "x0_teacher_loss",
    "cd_loss_anchor",
    "cd_loss_mid",
    "cd_loss_noise",
    # Same three bands for the ENDPOINT term. In the `x0_source=teacher` arm the CD term is
    # off and these are the only live diagnostic; `x0_loss_noise` is the 1-NFE number.
    "x0_loss_anchor",
    "x0_loss_mid",
    "x0_loss_noise",
    "ema_dist",
)

#: Parameters excluded from weight decay on top of HF's own bias/norm exclusions.
#: ``slot_embeddings`` is a soft prompt: decaying it pulls the slots back toward the
#: origin and directly fights the measured vocabulary initialisation
#: (``reasoning-setup-2b.md`` §9.1 C3), which is the whole reason those values start
#: where they do.
NO_DECAY_PARAMS = ("slot_embeddings",)

#: How often to re-measure each loss term's share of the backbone gradient. 0 disables.
#: Cheap at this interval (three partial backwards over one layer, on the graph the
#: step already built), and it is the standing check that no term has gone inert.
#: Settable via ``KAVA_GRAD_PROBE_STEPS`` — an env var rather than a config key so it
#: needs no field on the shared ``TrainingArguments`` owned by the sft recipe.
GRAD_PROBE_STEPS = int(os.environ.get("KAVA_GRAD_PROBE_STEPS", 200))


class LocalityGroupedSampler(Sampler[int]):
    """Shuffle shards and clips while keeping their samples adjacent.

    The default random sampler turns every worker into a random reader over thousands of
    NFS-hosted ZIPs. This sampler still visits every dataset index exactly once per epoch and
    reshuffles at every epoch, but uses a hierarchy of ``chunk -> clip -> anchor``. Consecutive
    global batches therefore reuse a small working set in the node page cache, and repeated
    anchors avoid bouncing between ZIP members.
    """

    def __init__(self, dataset: Any, seed: int) -> None:
        if not hasattr(dataset, "io_locality_keys"):
            raise TypeError(
                "KAVA_IO_GROUPED_SAMPLER=1 requires dataset.io_locality_keys()"
            )
        keys = dataset.io_locality_keys()
        if len(keys) != len(dataset):
            raise ValueError(
                f"dataset returned {len(keys)} locality keys for {len(dataset)} samples"
            )
        groups: dict[int, dict[str, list[int]]] = {}
        for index, (chunk, clip_id) in enumerate(keys):
            groups.setdefault(int(chunk), {}).setdefault(str(clip_id), []).append(index)
        self._groups = groups
        self._size = len(dataset)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self._size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _shuffled(values: list[Any], generator: torch.Generator) -> list[Any]:
        if len(values) < 2:
            return values
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[i] for i in order]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        chunks = self._shuffled(list(self._groups), generator)
        for chunk in chunks:
            clips = self._shuffled(list(self._groups[chunk]), generator)
            for clip_id in clips:
                indices = self._shuffled(self._groups[chunk][clip_id].copy(), generator)
                yield from indices

#: The slice of the backbone the probe differentiates. One mid-stack layer is enough to
#: rank the terms and keeps the probe to a few ms.
#:
#: Derived from the loaded student rather than hard-coded. It used to be a constant
#: ``layers.14.`` — mid-stack for the 28-layer 2B, but silently 39% depth on a 36-layer
#: Qwen3-VL-4B, which would have made gradient shares incomparable between students
#: without anything looking wrong.
#:
#: ⚠️ On an arm where the VLM is FROZEN and only the expert trains (the consistency and
#: expert-on-student arms), no ``vlm.*`` parameter has ``requires_grad``, so the probe's
#: parameter list came back empty and ``_probe_gradient_shares`` set ``_probe_failed`` and
#: disabled itself for the rest of the run -- silently, since an empty list is not an error.
#: Falling back to the expert's own mid-stack layer keeps the probe meaningful there; it is
#: still "one mid-depth block of whatever is being trained".
def probe_layer_prefix(base: Any) -> str:
    try:
        vlm_trainable = any(p.requires_grad for p in base.vlm.parameters())
    except AttributeError:
        vlm_trainable = False
    if vlm_trainable:
        try:
            n_layers = len(base.vlm.model.language_model.layers)
        except AttributeError:
            n_layers = 28
        return f"vlm.model.language_model.layers.{n_layers // 2}."
    try:
        n_layers = len(base.expert.layers)
    except AttributeError:
        n_layers = 36
    return f"expert.layers.{n_layers // 2}."


class KaVaTrainer(ReasoningVLA_Trainer):
    """``ReasoningVLA_Trainer`` plus per-term loss logging.

    Terms are summed across ``compute_loss`` calls (so gradient accumulation is
    handled) and averaged on each ``log``, matching how HF averages ``loss`` itself.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._aux_sums: dict[str, float] = {}
        self._aux_count: int = 0
        # Gradient shares are RATIOS measured once per probe, not per-micro-batch
        # values. They must not go through _aux_sums, which log() divides by
        # _aux_count -- with logging_steps=5 x grad_accum=16 that is 80 compute_loss
        # calls, so the logged curve came out 80x too small while the printed probe
        # line was right. Kept in their own dict and logged verbatim.
        self._grad_shares: dict[str, float] = {}
        self._last_probe_step: int = -1
        self._probe_failed: bool = False

    def _get_train_sampler(self, train_dataset=None):
        raw = os.environ.get("KAVA_IO_GROUPED_SAMPLER", "0").strip().lower()
        if raw not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                "KAVA_IO_GROUPED_SAMPLER must be 0/1, false/true, or no/yes; "
                f"got {raw!r}"
            )
        if raw in {"0", "false", "no"}:
            return super()._get_train_sampler(train_dataset)
        dataset = self.train_dataset if train_dataset is None else train_dataset
        sampler = LocalityGroupedSampler(dataset, seed=self.args.seed)
        if self.args.process_index == 0:
            clip_count = sum(len(clips) for clips in sampler._groups.values())
            print(
                "[dataloader] locality sampler "
                f"samples={len(sampler)} chunks={len(sampler._groups)} clips={clip_count}",
                flush=True,
            )
        return sampler

    def _get_dataloader(
        self,
        dataset,
        description: str,
        batch_size: int,
        sampler_fn=None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> DataLoader:
        """Build the training loader with optional out-of-order worker delivery.

        PyTorch's ordered delivery creates head-of-line blocking when one video sample is
        slow to read or decode: all later completed batches wait behind it. On the NFS-backed
        navigation run this appeared exactly every ``num_workers`` steps (three ~3 s steps,
        then one 33--42 s step with one rank idle). ``in_order=False`` lets a ready worker
        feed the rank while the slow worker finishes. Every sampler index is still consumed
        exactly once; only its within-epoch arrival order changes.

        Keep evaluation ordered and make the optimization opt-in because out-of-order
        delivery changes exact replay order after a mid-epoch restart. The installed HF
        Trainer does not expose PyTorch's ``in_order`` argument, so this mirrors its 4.57.1
        loader construction with that single additional keyword.
        """
        raw = os.environ.get("KAVA_DATALOADER_IN_ORDER", "1").strip().lower()
        if raw not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                "KAVA_DATALOADER_IN_ORDER must be 0/1, false/true, or no/yes; "
                f"got {raw!r}"
            )
        in_order = raw in {"1", "true", "yes"}
        if not is_training or in_order:
            return super()._get_dataloader(
                dataset,
                description,
                batch_size,
                sampler_fn=sampler_fn,
                is_training=is_training,
                dataloader_key=dataloader_key,
            )

        timeout_raw = os.environ.get("KAVA_DATALOADER_TIMEOUT_SECONDS", "0").strip()
        try:
            dataloader_timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                "KAVA_DATALOADER_TIMEOUT_SECONDS must be a non-negative number; "
                f"got {timeout_raw!r}"
            ) from exc
        if dataloader_timeout < 0:
            raise ValueError(
                "KAVA_DATALOADER_TIMEOUT_SECONDS must be non-negative; "
                f"got {dataloader_timeout}"
            )

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(
                self.data_collator, description=description
            )

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "in_order": False,
            # A worker blocked in an NFS read otherwise leaves the other DDP ranks inside
            # a collective until NCCL's much longer watchdog fires. This wait is only on an
            # empty result queue, so it has no cost while prefetched batches are available.
            "timeout": dataloader_timeout,
        }
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )

        if self.args.process_index == 0:
            print(
                "[dataloader] training in_order=False "
                f"workers={self.args.dataloader_num_workers} "
                f"prefetch_factor={self.args.dataloader_prefetch_factor} "
                f"timeout={dataloader_timeout:g}s",
                flush=True,
            )
        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        return dataloader

    def get_decay_parameter_names(self, model: Any) -> list[str]:
        decay = super().get_decay_parameter_names(model)
        return [name for name in decay if not any(part in name for part in NO_DECAY_PARAMS)]

    # ------------------------------------------------------------------ EMA save
    def _ema(self):
        """The ``ExpertEMA`` owned by an ``EMACallback``, if one is configured."""
        for cb in getattr(self.callback_handler, "callbacks", []):
            ema = getattr(cb, "ema", None)
            if ema is not None and hasattr(ema, "swap_in"):
                return ema
        return None

    def _save_checkpoint(self, model: Any, trial: Any, **kwargs: Any):
        """Serialise the EMA weights instead of the training weights, when an EMA exists.

        ⚠️ This MUST wrap the write, not follow it. ``_maybe_log_save_evaluate`` calls
        ``_save_checkpoint`` and only then ``callback_handler.on_save`` (transformers 4.57.1,
        ``trainer.py:3227-3229``), so doing the swap in an ``on_save`` callback saves the
        training weights and then leaves the EMA in the live model -- wrong on both counts.

        Why swap at all rather than write a side-car: the eval path loads a trained expert
        back through the TEACHER slot (``++model.teacher_checkpoint_path=<ckpt>``), and
        ``stitched_model._load_teacher_non_vlm`` looks for canonical ``expert.*`` names in a
        sharded ``model.safetensors`` + index. An ``ema.safetensors`` beside it would be
        invisible to that loader and to ``FrozenExpert._load``. Swapping keeps every existing
        eval command working unchanged.

        try/finally is not decoration: an exception during serialisation (a full disk, an
        interrupted NFS write) would otherwise leave EMA weights in the live model and
        training would silently continue from the average.
        """
        ema = self._ema()
        if ema is None:
            return super()._save_checkpoint(model, trial, **kwargs)
        base = self._unwrapped(model)
        verify = os.environ.get("CD_VERIFY_SAVE") == "1"
        probe = next(iter(ema.shadow))
        live_before = dict(base.named_parameters())[probe].detach().float().clone()
        if verify:
            d = float((live_before - ema.shadow[probe].to(live_before.device).float()).norm())
            print(f"[ema-save] ||live - ema|| on {probe} = {d:.6e}", flush=True)
        ema.swap_in(base)
        try:
            out = super()._save_checkpoint(model, trial, **kwargs)
        finally:
            ema.swap_out(base)
        if verify:
            restored = float(
                (dict(base.named_parameters())[probe].detach().float() - live_before).norm())
            print(f"[ema-save] restore error after swap_out = {restored:.3e} (must be 0)",
                  flush=True)
            self._verify_saved_is_ema(probe, ema)
        return out

    def _verify_saved_is_ema(self, probe: str, ema: Any) -> None:
        """Read the tensor back off disk and confirm it is the EMA, not the live weights.

        End-to-end rather than by inspection: this is the one path where getting it wrong is
        silent -- you would ship a checkpoint of the training weights and never know.
        """
        import glob
        import json as _json

        from safetensors.torch import load_file as _load
        ck = sorted(glob.glob(os.path.join(self.args.output_dir, "checkpoint-*")),
                    key=lambda q: int(q.rsplit("-", 1)[1]))
        if not ck:
            print("[ema-save] no checkpoint dir found to verify", flush=True)
            return
        idx = os.path.join(ck[-1], "model.safetensors.index.json")
        single = os.path.join(ck[-1], "model.safetensors")
        if os.path.exists(idx):
            with open(idx) as fh:
                shard = _json.load(fh)["weight_map"][probe]
            got = _load(os.path.join(ck[-1], shard))[probe]
        elif os.path.exists(single):
            got = _load(single)[probe]
        else:
            print(f"[ema-save] {ck[-1]} has no safetensors to verify", flush=True)
            return
        err = float((got.float() - ema.shadow[probe].cpu().float()).norm())
        print(f"[ema-save] saved-vs-EMA on {probe} = {err:.3e} (must be ~0)  <- {ck[-1]}",
              flush=True)

    def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False, **kwargs: Any):
        base = self._unwrapped(model)
        probing = self._should_probe()
        if probing:
            base.keep_loss_terms = True
        try:
            result = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        finally:
            if probing:
                base.keep_loss_terms = False
        loss, outputs = result if isinstance(result, tuple) else (result, None)
        self._stash(outputs)
        if probing:
            self._probe_gradient_shares(base)
        return (loss, outputs) if return_outputs else loss

    # -------------------------------------------------------- gradient probe
    def _unwrapped(self, model: Any) -> Any:
        return getattr(self, "accelerator", None) and self.accelerator.unwrap_model(model) or model

    def _should_probe(self) -> bool:
        if self._probe_failed or GRAD_PROBE_STEPS <= 0:
            return False
        step = int(getattr(self.state, "global_step", 0))
        if step == self._last_probe_step:
            return False  # once per optimizer step, not once per accumulation micro-batch
        return step % GRAD_PROBE_STEPS == 0

    def _probe_gradient_shares(self, base: Any) -> None:
        """Log each loss term's share of the *backbone* gradient.

        A term can be finite, decreasing, and still steer nothing — which is exactly
        what happened to ``L_KV`` at its first ``lambda_2``: it normalises over every
        (layer, head, slot, dim) element, so the shipped weight left it contributing
        0.0% of the backbone gradient while looking perfectly healthy in the loss log.
        A weight calibrated once by measurement is only safe if something keeps
        checking it, so this re-measures on one batch every ``GRAD_PROBE_STEPS``.

        Uses ``autograd.grad`` on the *existing* graph (before HF's backward), so the
        cost is three partial backwards over one mid-stack layer rather than three
        extra forwards. Nothing is written to ``.grad``, so training is unaffected.

        The probe is best-effort, and on some stacks it cannot run at all:

        * ZeRO-3 shards the parameters, so there is nothing local to differentiate.
        * ZeRO-2 (deepspeed 0.19) registers a post-accumulate hook on every parameter
          (``stage_1_and_2.py:1075``, ``self._grad_acc_hooks``).  Those fire during the
          probe's extra backwards too, and reduce into an IPG bucket the real backward has
          not filled -- observed as ``IndexError`` at ``stage_1_and_2.py:1575``.  Probing
          an activation instead of parameters does not avoid it.  Note the crash is the
          *good* outcome; the bad one is those hooks quietly folding probe gradients into
          the step.

        A failure therefore disables the probe for the rest of the run rather than taking
        the job down.  When it is unavailable, measure the weights with
        ``scripts/calibrate_kd_weights.py``, which does the same thing on a plain
        single-GPU model with no ZeRO engine attached.
        """
        terms = getattr(base, "last_loss_terms", None)
        if not terms:
            return
        probe = [
            p
            for name, p in base.named_parameters()
            if name.startswith(probe_layer_prefix(base)) and p.requires_grad
        ]
        if not probe:
            self._probe_failed = True
            return

        # Weight per term, so the reported share reflects what actually enters the total.
        # `kv_weight` / `kd_weight` are the KD student's fields; `kv_loss_weight` /
        # `latent_loss_weight` are KAVA's. Both spellings are read so one probe serves both.
        weights = {
            "ce": 1.0,
            "latent": float(getattr(base, "latent_loss_weight", 0.0)),
            "kv": float(getattr(base, "kv_loss_weight", getattr(base, "kv_weight", 0.0))),
            "kd": float(getattr(base, "kd_weight", 0.0)),
            "block": float(getattr(base, "block_weight", 0.0)),
            # ⚠️ `freerun`, `field` and `roll` are in `attached` but were never listed here,
            # so `weights.get(name, 1.0)` silently reported them at weight 1.0 regardless of
            # their configured value. Added along with `cd` so the probe reports what
            # actually enters the total.
            "freerun": float(getattr(base, "block_freerun_weight", 0.0)),
            "field": float(getattr(base, "field_weight", 0.0)),
            "roll": float(getattr(base, "roll_weight", 0.0)),
            "cd": float(getattr(base, "cd_weight", 0.0)),
            "x0_gt": float(getattr(base, "x0_gt_weight", 0.0)),
        }
        try:
            norms = {}
            for name, term in terms.items():
                grads = torch.autograd.grad(
                    term, probe, retain_graph=True, allow_unused=True, materialize_grads=True
                )
                norms[name] = weights.get(name, 1.0) * float(
                    torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
                )
        except Exception as ex:  # ZeRO-3 sharding, a freed graph, ...
            # Include the traceback. A bare message here ("list index out of range") names
            # neither the frame nor the library, and sent a debugging session chasing the
            # wrong hypothesis -- the probe is disabled for the rest of the run, so this is
            # the only chance to record why.
            import traceback

            print(
                f"[kava] gradient probe disabled: {type(ex).__name__}: {ex}\n"
                + "".join(traceback.format_exc().splitlines(keepends=True)[-8:]),
                flush=True,
            )
            self._probe_failed = True
            return

        self._last_probe_step = int(getattr(self.state, "global_step", 0))
        reference = norms.get("ce", 0.0)
        summary = "  ".join(
            f"{k}={v:.3e}" + (f" ({v / reference:.0%} of ce)" if reference and k != "ce" else "")
            for k, v in norms.items()
        )
        print(f"[kava] step {self._last_probe_step} weighted backbone grad: {summary}", flush=True)
        for name, value in norms.items():
            if name != "ce" and reference:
                self._grad_shares[f"gradshare_{name}"] = value / reference
        self._grad_shares["gradshare_ce_absnorm"] = reference

    def _stash(self, outputs: Any) -> None:
        if outputs is None:
            return
        seen = False
        for key in AUX_LOSS_KEYS:
            value = outputs.get(key) if hasattr(outputs, "get") else getattr(outputs, key, None)
            if value is None:
                continue
            self._aux_sums[key] = self._aux_sums.get(key, 0.0) + float(value.detach().item())
            seen = True
        if seen:
            self._aux_count += 1

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._aux_count:
            for key, total in self._aux_sums.items():
                logs[key] = round(total / self._aux_count, 6)
            self._aux_sums = {}
            self._aux_count = 0
        if self._grad_shares:  # ratios: logged as-is, never averaged
            logs.update({k: round(v, 6) for k, v in self._grad_shares.items()})
            self._grad_shares = {}
        super().log(logs, *args, **kwargs)
