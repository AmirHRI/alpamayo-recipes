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

r"""Deployment latency of the 2B student + its tuned action expert, vs camera count.

Splits the closed-loop cost into the two parts that scale differently:

  prefill      one VLM forward over the whole prompt -> the KV cache.  Scales with TOKENS,
               and ~93% of them are vision, so this is what the camera count moves.
  expert step  action_in_proj -> 28 expert blocks over 64 action tokens attending the cached
               prefix -> action_out_proj.  ONE Euler step; a trajectory needs
               ``num_inference_steps`` (10) of them.

⚠️ Cameras are sliced BEFORE preprocessing and the prompt is rebuilt, not truncated after the
fact: the text carries one placeholder run per image, so dropping images without re-running
the processor would leave the token count disagreeing with the pixel values.

⚠️ Camera order is the loader's ``camera_features`` order, NOT the global camera-index table:
  0 cross_left, 1 front_wide, 2 cross_right, 3 front_tele
so front-wide + front-tele is [1, 3], and front-wide alone is [1].

⚠️ Every timing is CUDA-synced and preceded by warmup iterations. Without the sync the first
phase reports near-zero while its kernels are still queued behind the launch.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m alpamayo1_5_distill.scripts.latency_profile \
      --config-path pkg://alpamayo1_5_distill/configs \
      --config-name sft_eval_stitched_2b_prunedexpert_lcdrive \
      ++model.attn_implementation=sdpa \
      ++model.checkpoint_path=<eos ckpt> ++model.teacher_checkpoint_path=<eos ckpt> \
      ++lat.n_warmup=2 ++lat.n_timed=8
"""

from __future__ import annotations

import time

import hydra
import hydra.utils as hyu
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from transformers.cache_utils import DynamicCache

SETTINGS = [
    ("4 cam x 4 frames (front-left, front-wide, front-right, front-tele)", [0, 1, 2, 3]),
    ("3 cam x 4 frames (front-left, front-wide, front-right)", [0, 1, 2]),
    ("2 cam x 4 frames (front-wide, front-tele)", [1, 3]),
    ("1 cam x 4 frames (front-wide)", [1]),
]


def _sync_time(fn, n_warmup: int, n_timed: int):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_timed * 1e3      # ms


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    c = cfg.get("lat", {})
    n_warmup, n_timed = int(c.get("n_warmup", 2)), int(c.get("n_timed", 8))
    n_clips = int(c.get("n_clips", 3))
    dev = torch.device("cuda")

    model = hyu.instantiate(cfg.model, _convert_="partial").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_act = model.action_space.get_action_space_dims()[0]
    n_steps = int(getattr(model.diffusion, "num_inference_steps", 10))
    print(f"[lat] expert {len(model.expert.layers)} layers, {n_act} action tokens, "
          f"{n_steps} denoising steps per trajectory", flush=True)

    # ⚠️ vla_preprocess_args removed so the dataset hands back the RAW sample: the cameras must
    # be sliced before the prompt is built, and the built-in path would tokenise all 4 first.
    ds_cfg = OmegaConf.to_container(cfg.data.val_dataset, resolve=True)
    pre_args = ds_cfg.pop("vla_preprocess_args")
    ds = hyu.instantiate(ds_cfg, _convert_="partial", model_config=model.config)
    pre = hyu.instantiate(pre_args, _convert_="partial", model_config=model.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)

    raw = [ds[i] for i in range(n_clips)]
    print(f"[lat] {len(raw)} clips; image_frames {tuple(raw[0]['image_frames'].shape)}", flush=True)

    print(f"\n{'setting':<62}{'imgs':>5}{'tokens':>8}{'prefill':>10}{'1 step':>9}"
          f"{f'{n_steps} steps':>10}{'total':>9}")
    for name, cams in SETTINGS:
        batches = []
        for s in raw:
            d = dict(s)
            for k in ("image_frames", "camera_indices", "absolute_timestamps",
                      "relative_timestamps"):
                if k in d and torch.is_tensor(d[k]):
                    d[k] = d[k][cams]
            d["tokenized_data"] = pre(data=d)
            batches.append(d)
        batch = send_to_device(coll(batches[:1]), dev)
        td = dict(batch["tokenized_data"])
        ids = td.pop("input_ids")
        ids = model.fuse_traj_tokens(
            ids, {"ego_history_xyz": batch["ego_history_xyz"],
                  "ego_history_rot": batch["ego_history_rot"]})
        n_tok = ids.shape[1]

        def prefill():
            with torch.no_grad():
                return model.vlm(input_ids=ids, use_cache=True, logits_to_keep=1, **td)

        out = prefill()
        cache = out.past_key_values
        plen = cache.get_seq_length()
        pos = torch.arange(n_act, device=dev).repeat(3, 1, 1).clone()
        mask = torch.zeros((1, 1, n_act, plen + n_act), dtype=model.dtype, device=dev)
        x = torch.randn(1, *model.action_space.get_action_space_dims(), device=dev)
        t = torch.zeros(1, device=dev)

        def step():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                emb = model.action_in_proj(x, t)
                if emb.dim() == 2:
                    emb = emb.view(1, n_act, -1)
                o = model.expert(inputs_embeds=emb, position_ids=pos, past_key_values=cache,
                                 attention_mask=mask, use_cache=True)
                cache.crop(plen)      # roll the action K/V back off, as the sampler does
                return model.action_out_proj(o.last_hidden_state[:, -n_act:])

        ms_pre = _sync_time(prefill, n_warmup, n_timed)
        ms_step = _sync_time(step, n_warmup, n_timed)
        print(f"{name:<62}{len(cams) * 4:>5}{n_tok:>8}{ms_pre:>9.1f}m{ms_step:>8.1f}m"
              f"{ms_step * n_steps:>9.1f}m{ms_pre + ms_step * n_steps:>8.1f}m", flush=True)
        del cache, out
        torch.cuda.empty_cache()
    print("DONE_LATENCY", flush=True)


if __name__ == "__main__":
    main()
