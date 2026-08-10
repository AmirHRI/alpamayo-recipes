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

"""Measure each loss term's raw backbone gradient, and solve for ``kd_weight`` / ``kv_weight``.

**Why this is a standalone script and not the in-training probe.**  ``KaVaTrainer`` has a
gradient probe that does the same measurement on the live training graph, and it worked
throughout the KAVA runs.  It cannot work here.  DeepSpeed ZeRO-2 registers a
post-accumulate hook on every parameter (``stage_1_and_2.py:1075``,
``self._grad_acc_hooks``); those hooks fire during *any* backward, including the probe's
extra ``autograd.grad`` calls.  The visible symptom is a crash inside DeepSpeed's own
reduction epilogue::

    stage_1_and_2.py:1575 reduce_ipg_grads
        self.average_tensor(bucket.buffer[bucket.index]...)   -> IndexError

but the crash is the lucky outcome.  The unlucky one is the hooks reducing and
partitioning gradients from the probe's backward into the same buffers the real backward
is about to fill -- silent double counting, on the calibration measurement that every
later arm depends on.  So the probe stays best-effort in-training (it self-disables), and
the number that actually sets the weights is measured here, on a plain single-GPU model
with no ZeRO engine attached.

**What it reports.**  For each term ``t`` in {ce, kd, kv}, the L2 norm of
``d t / d theta`` over one mid-stack decoder layer, with ``kd_weight = kv_weight = 1`` so
the norms are the *raw* per-term scales.  Then, for each requested share ``s``::

    weight_t = s * |grad ce| / |grad t|

i.e. the weight at which term ``t`` contributes ``s`` of CE's gradient into the backbone.

**Why a mid-stack layer and not the whole model.**  Ranking three terms needs a common
yardstick, not a complete one, and one layer keeps each of the three partial backwards
cheap.  It is the same slice ``trainer.probe_layer_prefix`` uses, so the numbers here and
the ``gradshare_*`` logs (where they survive) are directly comparable.

Usage::

    CUDA_VISIBLE_DEVICES=0 python -m alpamayo1_5_distill.scripts.calibrate_kd_weights \\
        --config-path pkg://alpamayo1_5_distill/configs \\
        --config-name sft_kd_qwen3_4b_lcdrive \\
        ++calib.batches=4
"""

from __future__ import annotations

import hydra
import hydra.utils as hyu
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from alpamayo1_5_distill.trainer import probe_layer_prefix

#: Shares of the CE gradient to solve weights for. 1% / 5% / 10% brackets the range the
#: KAVA runs found usable: below ~1% a term is decorative, above ~30% it starts competing
#: with the objective that actually defines the task.
DEFAULT_SHARES = (0.01, 0.05, 0.10, 0.30)


def _term_grad_norms(model: torch.nn.Module, batch: dict) -> dict[str, float]:
    """Raw ``|d term / d theta_midstack|`` for every term the model exposes."""
    model.keep_loss_terms = True
    try:
        model(**batch)
    finally:
        model.keep_loss_terms = False

    terms = model.last_loss_terms
    if not terms:
        raise RuntimeError(
            "model exposed no loss terms; is this a KDReasoningVLA with a teacher loaded?"
        )

    prefix = probe_layer_prefix(model)
    params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
    if not params:
        raise RuntimeError(f"no trainable parameters under {prefix!r}")

    norms = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(
            term, params, retain_graph=True, allow_unused=True, materialize_grads=True
        )
        norms[name] = float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads])))
    return norms


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    calib = cfg.get("calib", {})
    n_batches = int(calib.get("batches", 4))
    shares = tuple(calib.get("shares", DEFAULT_SHARES))

    # ⚠️ Measure the RAW term scales. If the config's weights leaked into the measurement
    # the solve below would be self-referential -- and both ship at 0.0, which would make
    # every gradient identically zero and the answer look like "no signal".
    OmegaConf.update(cfg, "model.kd.kd_weight", 1.0, force_add=True)
    OmegaConf.update(cfg, "model.kd.kv_weight", 1.0, force_add=True)

    model = hyu.instantiate(cfg.model, _convert_="partial")
    dataset = hyu.instantiate(cfg.data.train_dataset, _convert_="partial", model_config=model.config)
    collate_fn = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)

    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16)
    model.train()
    # Same memory strategy as training: the student's K/V come from output_hidden_states,
    # which survives checkpointing, so the 4B student and the frozen 8B teacher fit on one
    # card. Without it this script OOMs where the training job does not.
    if getattr(cfg.trainer, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable({"use_reentrant": False})

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.trainer.per_device_train_batch_size),
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=False,
    )

    totals: dict[str, list[float]] = {}
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Recursive, not a top-level dict comprehension: `traj_data` is a nested container,
        # and moving only the outer tensors leaves `fuse_traj_tokens` scattering a cuda
        # source into a cpu `input_ids`. In training the HF Trainer does this for us.
        batch = send_to_device(batch, device)
        norms = _term_grad_norms(model, batch)
        for k, v in norms.items():
            totals.setdefault(k, []).append(v)
        print(f"[calib] batch {i}: " + "  ".join(f"{k}={v:.4e}" for k, v in norms.items()), flush=True)

    if "ce" not in totals:
        raise RuntimeError(f"no ce term measured; got {sorted(totals)}")

    mean = {k: sum(v) / len(v) for k, v in totals.items()}
    ce = mean["ce"]
    print("\n[calib] mean raw |grad| over "
          f"{n_batches} batches at {probe_layer_prefix(model)}")
    for k, v in mean.items():
        print(f"  {k:<8} {v:.4e}" + (f"   ({v / ce:.1%} of ce at weight 1.0)" if k != "ce" else ""))

    print("\n[calib] weight that buys each share of the CE gradient")
    header = "  share   " + "".join(f"{k:>14}" for k in mean if k != "ce")
    print(header)
    for s in shares:
        row = f"  {s:>5.0%}   "
        for k, v in mean.items():
            if k == "ce":
                continue
            row += f"{(s * ce / v if v > 0 else float('nan')):>14.4e}"
        print(row)
    print("\n[calib] copy the chosen row into configs/models/qwen3_vl_4b_kd.yaml")


if __name__ == "__main__":
    main()
