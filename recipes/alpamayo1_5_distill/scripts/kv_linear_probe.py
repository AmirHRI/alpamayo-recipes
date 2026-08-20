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

r"""Is the student's cache MIS-ROTATED or MISSING INFORMATION?  A closed-form answer.

Everything measured on the 2B says the cache error cannot be reduced by shaping the loss:
spans amplify only 1.2x (deep half CONTRACTS, 0.79), the 10-step ODE damps rather than
compounds (0.69), ``L_field`` moved nothing at the plateau, and LR 5e-5 / 1e-4 both DAMAGED
the converged solution.  Meanwhile a single-clip overfit reached 1.85e-4, **5x below** the
8.8e-4 training floor -- so the architecture can represent the mapping for one clip and the
floor is about serving 38k at once.

That leaves one cheap, decisive question.  The student's layer *j* must serve an expert slot
whose weights were trained on teacher layer ``pi(j)``.  Is the needed information PRESENT in
the student's K/V but in a different basis, or absent?

Fit, per layer, the best LINEAR map and compare against doing nothing:

    identity   ||K^S - K^T||^2 / ||K^T||^2         <- what L_block currently has to fix
    fitted     ||K^S A  - K^T||^2 / ||K^T||^2      <- A = argmin, ridge, closed form

  * fitted << identity  -> the information is there, mis-rotated.  A ~59 M-parameter adapter
    (28 layers x 1024x1024 x {K,V}) recovers it with NO change to the 28-layer student, and
    ``KVProjectorBank`` already implements exactly this.
  * fitted ~ identity   -> the student's features genuinely lack what the teacher's encode,
    and no interface adapter can invent it.

⚠️ Fitted on PRE-RoPE K/V, which is where an adapter would actually live (``k_proj`` output,
before the rotation).  A position-independent matrix cannot be fitted on post-RoPE keys: RoPE
mixes a position-dependent rotation into every vector, so one A would have to undo and redo a
different rotation per position.  V has no RoPE and is unaffected either way.

⚠️ Grams are STREAMED (X^T X, X^T Y are 1024x1024 regardless of token count) and accumulated
in float64.  Holding the tokens instead would be ~100 MB per layer per tensor per clip, and
float32 accumulation over ~10^5 tokens loses the third digit -- the same failure that put a
0.9972 on the span-BI diagonal.

⚠️ A is fitted on FIT clips and scored on HELD-OUT clips.  A 1024x1024 map has 1.05 M
parameters per output block; scoring it on its own fit data would report the optimism, not the
generalisation, and the whole point is whether a deployed adapter would help.

Usage::

    PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34 CUDA_VISIBLE_DEVICES=3 \
      python -m alpamayo1_5_distill.scripts.kv_linear_probe \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_kd_cosmos2b_prunedexpert_lcdrive \
        ++model.checkpoint_path=<student ckpt> ++probe.n_fit=24 ++probe.n_eval=8
"""

from __future__ import annotations

import os

import hydra
import hydra.utils as hyu
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from alpamayo1_5_distill.models.kd_model import recompute_kv


def _pi(n_ckpt: int, n_student: int) -> list[int]:
    """Survivors of PRUNE_EXPERT_LAYERS, i.e. teacher expert layer pi(j) -> student slot j."""
    drop = {int(x) for x in os.environ.get("PRUNE_EXPERT_LAYERS", "").split(",") if x.strip()}
    surv = [i for i in range(n_ckpt) if i not in drop]
    if len(surv) != n_student:
        raise RuntimeError(
            f"PRUNE_EXPERT_LAYERS leaves {len(surv)} layers, student has {n_student}; "
            f"set it so exactly {n_ckpt - n_student} are dropped")
    return surv


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    p = cfg.get("probe", {})
    n_fit, n_eval = int(p.get("n_fit", 24)), int(p.get("n_eval", 8))
    lam = float(p.get("ridge", 1e-3))
    dev = torch.device("cuda")

    model = hyu.instantiate(cfg.model, _convert_="partial").to(dev).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    # ⚠️ the teacher is placed on GPU LAZILY, inside KDReasoningVLA.forward. This script drives
    # the two towers directly and never calls forward, so without this the teacher's weights sit
    # on CPU and the embedding lookup dies on a device mismatch.
    model._place_teacher(dev, next(model.vlm.parameters()).dtype)
    s_text, t_text = model._text_model(), model._teacher_text_model()
    n_s, n_t = len(s_text.layers), len(t_text.layers)
    pi = _pi(n_t, n_s)
    h, d = model._kv_shape()
    w = h * d
    print(f"[probe] student {n_s} layers, teacher {n_t}; pi = {pi}", flush=True)
    print(f"[probe] fitting {w}x{w} maps, ridge {lam}, {n_fit} fit + {n_eval} eval clips",
          flush=True)

    ds = hyu.instantiate(cfg.data.train_dataset, _convert_="partial", model_config=model.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    loader = DataLoader(ds, batch_size=1, collate_fn=coll, num_workers=4, shuffle=False)

    # {(layer, 'k'|'v'): [XtX, XtY, YtY_trace, n]} accumulated in float64
    G: dict = {}

    def acc(store, X, Y):
        st = G.setdefault(store, [torch.zeros(w, w, dtype=torch.float64, device=dev),
                                 torch.zeros(w, w, dtype=torch.float64, device=dev),
                                 torch.zeros((), dtype=torch.float64, device=dev),
                                 0])
        Xd, Yd = X.double(), Y.double()
        st[0] += Xd.T @ Xd
        st[1] += Xd.T @ Yd
        st[2] += Yd.pow(2).sum()
        st[3] += Xd.shape[0]

    def flat(kv, layer):
        """[B, H, T, D] -> [B*T, H*D], the per-token vector an adapter would transform."""
        k, v = kv[layer]
        f = lambda x: x.permute(0, 2, 1, 3).reshape(-1, w)
        return f(k), f(v)

    done = 0
    for batch in loader:
        if done >= n_fit + n_eval:
            break
        gpu = send_to_device(dict(batch), dev)
        gpu.pop("clip_id", None)
        td = dict(gpu["tokenized_data"])
        ids = td.pop("input_ids")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s_out = model.vlm(input_ids=ids, output_hidden_states=True, use_cache=False,
                              logits_to_keep=1, **td)
            t_out = model.teacher.vlm(input_ids=ids, output_hidden_states=True, use_cache=False,
                                      logits_to_keep=1, **td)
        s_kv = recompute_kv(s_text, s_out.hidden_states, h, d)
        t_kv = recompute_kv(t_text, t_out.hidden_states, h, d)
        tag = "fit" if done < n_fit else "eval"
        for j in range(n_s):
            sk, sv = flat(s_kv, j)
            tk, tv = flat(t_kv, pi[j])
            acc((j, "k", tag), sk, tk)
            acc((j, "v", tag), sv, tv)
        del s_kv, t_kv, s_out, t_out
        done += 1
        if done % 8 == 0:
            print(f"[probe] {done}/{n_fit + n_eval} clips", flush=True)

    eye = torch.eye(w, dtype=torch.float64, device=dev)
    print(f"\n{'layer':>5}{'':3}{'K identity':>12}{'K fitted':>11}{'K gain':>8}"
          f"{'':3}{'V identity':>12}{'V fitted':>11}{'V gain':>8}", flush=True)
    tot = {"k": [0.0, 0.0], "v": [0.0, 0.0]}
    for j in range(n_s):
        row = [f"{j:>5}   "]
        for which in ("k", "v"):
            XtX, XtY, _, _ = G[(j, which, "fit")]
            A = torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * eye, XtY)
            eXtX, eXtY, eYtY, _ = G[(j, which, "eval")]
            # identity: ||X - Y||^2 = tr(XtX) - 2 tr(XtY) + ||Y||^2, all on HELD-OUT clips
            ident = (eXtX.diagonal().sum() - 2 * eXtY.diagonal().sum() + eYtY) / eYtY
            fit = (torch.einsum("ij,ji->", A.T, eXtX @ A) - 2 * torch.einsum("ij,ij->", A, eXtY)
                   + eYtY) / eYtY
            tot[which][0] += float(ident); tot[which][1] += float(fit)
            row.append(f"{float(ident):>12.4f}{float(fit):>11.4f}"
                       f"{float(ident) / max(float(fit), 1e-12):>8.1f}x")
            if which == "k":
                row.append("   ")
        print("".join(row), flush=True)
    for which in ("k", "v"):
        a, b = tot[which][0] / n_s, tot[which][1] / n_s
        print(f"[probe] MEAN {which.upper()}: identity {a:.4f}  fitted {b:.4f}  "
              f"gain {a / max(b, 1e-12):.1f}x", flush=True)
    print("DONE_PROBE", flush=True)


if __name__ == "__main__":
    main()
