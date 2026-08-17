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

r"""Does teacher-forcing hide compounding?  Two error curves through the SAME expert.

``L_block`` drives each expert block with the TEACHER's own action state ``h^T_l``, so every
layer's error is measured in isolation.  At inference nothing is teacher-forced: layer *l*
receives whatever the preceding layers produced from the student's cache, so errors
accumulate.  If that accumulation dominates, driving the isolated per-layer error lower --
which is exactly what epochs 2-3 did, -33% with only -0.06 min_ade to show for it -- cannot
help, and the objective has saturated as a proxy for reasons of DESIGN rather than capacity.

Two curves, same clips, same frozen teacher expert, same action embeds (same seed, same t,
so ``h_0`` is identical by construction and any divergence is the cache):

  teacher-forced   e_tf(l)  = || B_l(h^T_l ; K^S,V^S) - h^T_{l+1} ||^2 / || h^T_{l+1} ||^2
                              -- this IS the L_block term, per layer
  free-running     e_fr(l)  = || h^S_l - h^T_l ||^2 / || h^T_l ||^2
                              -- the student's own chain, what inference incurs

Reading it:
  * e_fr >> e_tf and growing with depth  -> compounding dominates. Fix the OBJECTIVE
    (unrolled / scheduled-sampling variant), not the capacity.
  * e_fr ~ e_tf, flat                    -> no compounding to speak of; the residual is a
    capacity floor and no reweighting will move it.

⚠️ Both models must be resident at once (teacher 10 B + student 6.4 B) so a single clip goes
through both without staging a ~450 MB cache per clip to disk.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m alpamayo1_5_distill.scripts.cache_error \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_eval_stitched_4b_lcdrive \
        ++model.attn_implementation=sdpa ++cerr.n_clips=32
"""

from __future__ import annotations

import os

import hydra
import hydra.utils as hyu
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from accelerate.utils import send_to_device  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from alpamayo1_5_distill.models.stitched_model import StitchedAlpamayoR1  # noqa: E402

TRAIN = "/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training"
TEACHER_CKPT = "/data/achahe/alpasim/huggingface/hub/models--nvidia--Alpamayo-1.5-10B-A1-format"
COSMOS = (
    "/data/achahe/alpasim/huggingface/hub/models--nvidia--Cosmos-Reason2-8B/"
    "snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9"
)
STUDENT = f"{TRAIN}/output_kd_4b_blockrandt_e3_lcdrive/checkpoint-4794"
TRAIN_CLIPS = ("/data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests/"
               "lcdrive_train_clip_uuids.txt")
SEED = 1234


def hook_expert(model, store, want_step=0):
    """Capture, at ONE denoising step: the VLM prefix cache, each layer's input, and the
    kwargs the enclosing forward passed. Kwargs are captured rather than rebuilt -- a
    hand-built mask/rope attends differently, which silently broke the first L_block."""
    layers = model.expert.layers
    n = len(layers)
    step = {"i": -1}

    def pre(_m, _a, kw):
        step["i"] += 1
        if step["i"] == want_step and kw.get("past_key_values") is not None:
            c = kw["past_key_values"]
            store["prefix"] = [
                (c.layers[i].keys.clone(), c.layers[i].values.clone())
                if hasattr(c, "layers") else
                (c.key_cache[i].clone(), c.value_cache[i].clone())
                for i in range(n)
            ]
            store["kwargs"] = {k: v for k, v in kw.items() if k in
                               ("attention_mask", "position_ids")}
        return None

    def mk(idx):
        def f(_m, args, kw, out):
            if step["i"] != want_step:
                return
            h_in = (args[0] if args else kw["hidden_states"]).detach()
            store.setdefault("h", {})[idx] = h_in
            store.setdefault("lkw", {})[idx] = {k: v for k, v in kw.items()
                                                if k != "past_key_values"}
            if idx == n - 1:
                y = out[0] if isinstance(out, tuple) else out
                store["h"][n] = y.detach()
        return f

    hs = [model.expert.register_forward_pre_hook(pre, with_kwargs=True)]
    hs += [layers[i].register_forward_hook(mk(i), with_kwargs=True) for i in range(n)]
    return hs


def rel(a, b):
    """||a-b||^2 / ||b||^2 -- the same normalisation block_output_loss uses."""
    return float((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean().clamp_min(1e-9))


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    c = cfg.get("cerr", {})
    n_clips = int(c.get("n_clips", 32))
    out_dir = c.get("out_dir", f"{TRAIN}/cache_error")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda")

    pool = sorted({l.strip() for l in open(TRAIN_CLIPS) if l.strip()})
    rng = np.random.default_rng(0)
    chosen = sorted(rng.choice(pool, size=min(n_clips, len(pool)), replace=False).tolist())
    uuid_file = os.path.join(out_dir, "_clips.txt")
    open(uuid_file, "w").write("\n".join(chosen) + "\n")
    print(f"[cerr] {len(chosen)} TRAIN clips (seed 0)", flush=True)

    tea = StitchedAlpamayoR1.from_teacher(
        checkpoint_path=TEACHER_CKPT, vlm_name_or_path=COSMOS, attn_implementation="sdpa"
    ).to(dev).eval()
    stu = hyu.instantiate({**cfg.model, "checkpoint_path": STUDENT},
                          _convert_="partial").to(dev).eval()
    n_layers = len(tea.expert.layers)
    print(f"[cerr] teacher + student resident, {n_layers} expert layers", flush=True)

    ds = hyu.instantiate({**cfg.data.val_dataset, "clip_uuid_filter": uuid_file},
                         _convert_="partial", model_config=tea.config)
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=tea.config)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=2, shuffle=False)

    e_fr = np.zeros(n_layers + 1)
    e_tf = np.zeros(n_layers)
    seen = 0
    for i, batch in enumerate(loader):
        if i >= n_clips:
            break
        st_t, st_s = {}, {}
        for model, store in ((tea, st_t), (stu, st_s)):
            # ⚠️ A FRESH dict per model. `sample_trajectories_from_data_with_vlm_rollout`
            # does `tokenized_data.pop("input_ids")`, mutating the batch, so driving a second
            # model with the same dict dies on KeyError: 'input_ids'. dict(batch) alone is
            # not enough -- the nested tokenized_data is the object being popped.
            gpu = send_to_device(dict(batch), dev)
            gpu["tokenized_data"] = dict(gpu["tokenized_data"])
            hs = hook_expert(model, store)
            try:
                torch.manual_seed(SEED)           # identical action noise for both
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=gpu, num_traj_samples=1, num_traj_sets=1,
                        top_p=0.98, temperature=0.6, max_generation_length=256)
            finally:
                for h in hs:
                    h.remove()
        if seen == 0 and "prefix" in st_s and "prefix" in st_t:
            print(f"[cerr] prefix len teacher {st_t['prefix'][0][0].shape[2]} "
                  f"student {st_s['prefix'][0][0].shape[2]}  (differ = different CoT)",
                  flush=True)
        if "prefix" not in st_s or "h" not in st_t:
            print(f"[cerr] clip {i}: capture failed, skipped", flush=True)
            continue

        # FREE-RUNNING: the student's own chain vs the teacher's, layer by layer.
        for l in range(n_layers + 1):
            e_fr[l] += rel(st_s["h"][l], st_t["h"][l])

        # TEACHER-FORCED: the L_block term -- teacher's h_l in, student's cache, compare to
        # the teacher's h_{l+1}. Fresh cache per layer: the block appends its own action K/V.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for l in range(n_layers):
                cache = DynamicCache()
                k, v = st_s["prefix"][l]
                cache.update(k, v, 0, {})
                blk = tea.expert.layers[l]
                orig = blk.self_attn.layer_idx
                blk.self_attn.layer_idx = 0
                try:
                    # ⚠️ the STUDENT's kwargs, not the teacher's. Each model generates its
                    # OWN CoT during the rollout, so the two prefixes differ in LENGTH
                    # (e.g. 3392 vs 3160) and a mask sized for one cannot index the other.
                    # Pair every cache with the kwargs captured from the run that built it.
                    y = blk(st_t["h"][l], past_key_values=cache, use_cache=True,
                            **st_s["lkw"][l])
                finally:
                    blk.self_attn.layer_idx = orig
                y = y[0] if isinstance(y, tuple) else y
                e_tf[l] += rel(y, st_t["h"][l + 1])
        seen += 1
        if seen % 8 == 0:
            print(f"[cerr] {seen}/{n_clips}", flush=True)

    e_fr /= max(seen, 1)
    e_tf /= max(seen, 1)
    np.savez(os.path.join(out_dir, "cache_error.npz"), free_running=e_fr,
             teacher_forced=e_tf, n_clips=seen, clips=np.array(chosen))
    print(f"\n[cerr] n={seen} clips\n{'layer':>6}{'teacher-forced':>17}{'free-running':>15}{'ratio':>9}")
    for l in range(0, n_layers, 4):
        r = e_fr[l] / max(e_tf[l], 1e-12)
        print(f"{l:>6}{e_tf[l]:>17.3e}{e_fr[l]:>15.3e}{r:>9.1f}")
    print(f"{'final':>6}{'':>17}{e_fr[n_layers]:>15.3e}")
    print(f"\n  mean teacher-forced {e_tf.mean():.3e}   final free-running {e_fr[n_layers]:.3e}"
          f"   ratio {e_fr[n_layers] / max(e_tf.mean(), 1e-12):.0f}x")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(range(n_layers), e_tf, "-o", ms=4, label="teacher-forced  (what $L_{block}$ trains)")
    ax.semilogy(range(n_layers + 1), e_fr, "-s", ms=4,
                label="free-running  (what inference incurs)")
    ax.set_xlabel("expert layer"); ax.set_ylabel("relative squared error")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
    ax.set_title(f"Does teacher-forcing hide compounding?  student=blockrandt e3, "
                 f"n={seen} train clips", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "cache_error.png"), dpi=140)
    print("[cerr] wrote cache_error.png", flush=True)


if __name__ == "__main__":
    main()
