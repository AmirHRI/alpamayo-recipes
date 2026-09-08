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

r"""Evaluate one model across CAMERA SUBSETS, saving the trajectories, not just the metrics.

Answers "what does dropping cameras cost in accuracy", the question the latency tables in
``LATENCY_PROFILE.md`` deliberately could not.  Saves per clip: all sampled trajectories, the
ground truth, and the clip UUID, so downstream work (plots, corner distance, per-category
breakdowns, mode analysis) needs no re-run.

**Prefill only.**  Drives ``sample_trajectories_prefill_only``: one VLM forward over the prompt
-> KV cache -> the action expert denoises against it.  No CoT is generated.  That matches
training (which only ever prefills) and is verified equivalent to the old rollout path -- the
caches are bit-identical (max |diff| 0.000e+00) and the metrics agree end to end
(min_ade delta -0.0086, z = -0.11 on 240 clips).

**Metrics come from the repo's own ``DistanceMetrics``**, not a reimplementation, so the numbers
land on the same scale as every other arm in this tree (teacher 0.5776 at 4 cameras).

⚠️ Cameras are sliced BEFORE preprocessing so the prompt is REBUILT.  The prompt text is
adaptive -- verified by decoding it -- so a dropped camera loses its label, its four ``frame N``
tags and its four vision blocks.  Slicing after tokenisation would leave the token count
disagreeing with the pixel values.

⚠️ Camera indices are the loader's ``camera_features`` order: 0 cross_left ("Front left
camera"), 1 front_wide ("Front camera"), 2 cross_right ("Front right camera"), 3 front_tele
("Front telephoto camera").  NOT the global camera-index table, where front_tele is 6.

⚠️ Reduced-camera prompts are OUT OF DISTRIBUTION: the model was trained with all four cameras
present and named in a fixed order, so these numbers measure "what the trained model does when
starved", not "what a model trained at this camera count would do".

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m alpamayo1_5_distill.scripts.eval_camera_sweep \
      --config-path pkg://alpamayo1_5_distill/configs \
      --config-name sft_eval_stitched_4b_lcdrive \
      ++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_teacher \
      ++model.vlm_name_or_path=<cosmos 8b> ++model.checkpoint_path=<10b> \
      ++model.attn_implementation=sdpa ++sweep.tag=teacher
"""

from __future__ import annotations

import json
import os

import hydra
import hydra.utils as hyu
import numpy as np
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

SETTINGS = [
    ("4cam", [0, 1, 2, 3], "front-left, front-wide, front-right, front-tele"),
    ("3cam", [0, 1, 2], "front-left, front-wide, front-right"),
    ("2cam", [1, 3], "front-wide, front-tele"),
    ("1cam", [1], "front-wide"),
]


class _CameraSubset(torch.utils.data.Dataset):
    """PAIDataset with cameras sliced before the prompt is built.

    Composition rather than a subclass: the base class runs its preprocess inside
    ``__getitem__``, so the only way to slice first is to build it WITHOUT a preprocess and
    apply ours afterwards.
    """

    def __init__(self, base, pre, cams: list[int]):
        self.base, self.pre, self.cams = base, pre, cams

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        s = dict(self.base[i])
        for k in ("image_frames", "camera_indices", "absolute_timestamps",
                  "relative_timestamps"):
            if k in s and torch.is_tensor(s[k]):
                s[k] = s[k][self.cams]
        s["tokenized_data"] = self.pre(data=s)
        return s


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    sw = cfg.get("sweep", {})
    tag = str(sw.get("tag", "model"))
    bs = int(sw.get("batch_size", 4))
    n_samples = int(sw.get("num_traj_samples", 6))
    limit = int(sw.get("limit", 0))
    only = str(sw.get("only", ""))
    out_dir = str(sw.get("out_dir",
                         "/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/camsweep"))
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda")

    model = hyu.instantiate(cfg.model, _convert_="partial").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[sweep] {tag}: expert {len(model.expert.layers)} layers, bs {bs}, "
          f"{n_samples} traj samples", flush=True)

    ds_cfg = OmegaConf.to_container(cfg.data.val_dataset, resolve=True)
    pre_args = ds_cfg.pop("vla_preprocess_args")
    base = hyu.instantiate(ds_cfg, _convert_="partial", model_config=model.config)
    pre = hyu.instantiate(pre_args, _convert_="partial", model_config=model.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    metric = hyu.instantiate(cfg.evaluate.metric_runner.metrics[-1], _convert_="partial")
    print(f"[sweep] {len(base)} clips; metric {type(metric).__name__}", flush=True)

    for name, cams, desc in SETTINGS:
        if only and only != name:
            continue
        loader = DataLoader(_CameraSubset(base, pre, cams), batch_size=bs, collate_fn=coll,
                            num_workers=8, shuffle=False)
        ids, preds, gts, per = [], [], [], []
        seen = 0
        for batch in loader:
            if limit and seen >= limit:
                break
            clip_ids = batch.get("clip_id")
            gpu = send_to_device({k: v for k, v in batch.items() if k != "clip_id"}, dev)
            with torch.no_grad():
                pred_xyz, pred_rot = model.sample_trajectories_prefill_only(
                    data=gpu, num_traj_samples=n_samples, num_traj_sets=1)
            out = {"pred_xyz": pred_xyz, "pred_rot": pred_rot}
            m = metric.evaluate(model, gpu, out)
            b = pred_xyz.shape[0]
            for i in range(b):
                ids.append(str(clip_ids[i]))
                # [ns, nj, Tf, 3] -> keep every sample; float16 keeps 4 GB of sweep under 20 MB
                preds.append(pred_xyz[i].float().cpu().numpy().astype(np.float16))
                gts.append(gpu["ego_future_xyz"][i, -1].float().cpu().numpy().astype(np.float32))
                per.append({"clip_id": str(clip_ids[i]),
                            **{k: float(v[i]) for k, v in m.items()
                               if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == b}})
            seen += b
            if seen % 100 < bs:
                done = [p for p in per if "min_ade" in p]
                avg = np.mean([p["min_ade"] for p in done]) if done else float("nan")
                print(f"[sweep] {name} {seen}/{len(base)}  running min_ade {avg:.4f}", flush=True)

        npz = os.path.join(out_dir, f"{tag}_{name}.npz")
        np.savez_compressed(
            npz, clip_ids=np.array(ids), pred_xyz=np.stack(preds), gt_xyz=np.stack(gts),
            cameras=np.array(cams), description=np.array(desc),
            min_ade=np.array([p.get("min_ade", np.nan) for p in per], dtype=np.float32),
            ade=np.array([p.get("ade", np.nan) for p in per], dtype=np.float32))
        with open(os.path.join(out_dir, f"{tag}_{name}.json"), "w") as fh:
            json.dump(per, fh)
        ma = float(np.nanmean([p.get("min_ade", np.nan) for p in per]))
        ad = float(np.nanmean([p.get("ade", np.nan) for p in per]))
        print(f"[sweep] === {tag} {name} ({desc}): n={len(per)}  "
              f"min_ade {ma:.4f}  ade {ad:.4f}  -> {npz}", flush=True)
    print("DONE_SWEEP", flush=True)


if __name__ == "__main__":
    main()
