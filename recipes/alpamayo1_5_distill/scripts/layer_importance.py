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

"""Which VLM layers does the action expert actually depend on?

`L_KV` currently weights all 36 layers equally.  Nothing has ever checked whether the expert
cares equally about them.  This measures it **causally**: swap ONE layer of the K/V cache
between student and teacher, run the expert, and see how much the driven trajectory moves.

    fix-one  : student cache, layer l replaced by the TEACHER's  -> how much aligning l BUYS
    break-one: teacher cache, layer l replaced by the STUDENT's  -> how much l MATTERS

Both are cheap for the same reason the stitched eval is: the expensive part is the VLM
forward that builds the cache, and it is done ONCE per clip and reused across all 74
expert rollouts.  The expert is 2048-dim and only runs ~128 action-token queries.

**Why the swap is well-posed.**  The expert masks out everything after
``<|traj_future_start|>`` (`alpamayo_r1.py:248-262` builds `attention_mask` with
``offset:-n_diffusion_tokens`` set False), so it reads only the PROMPT region of the cache.
The prompt is byte-identical between teacher and student -- verified earlier as element-wise
equal ``input_ids`` -- so the two caches correspond position-for-position over exactly the
span the expert consumes.  Generated tokens differ between the models but are never read.

⚠️ **Diffusion noise is seeded per rollout.**  ``diffusion.sample`` draws fresh noise, and an
unseeded comparison is precisely what produced a retracted finding earlier in this project:
two runs of an IDENTICAL arm differed by more than every effect being measured.  Every
variant here re-seeds to the same value immediately before the rollout, so the ONLY
difference between variants is the swapped layer.

⚠️ **Single-layer swaps are marginal, not additive.**  Each number is the effect of layer l
with all other layers held at the baseline model's values.  Layers interact, so the 36
values will not sum to the full student-teacher gap.  That is fine for the relative
weighting this exists to produce, but they cannot be read as a decomposition.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m alpamayo1_5_distill.scripts.layer_importance \\
        --config-path pkg://alpamayo1_5_distill/configs \\
        --config-name sft_eval_stitched_4b_lcdrive \\
        ++model.attn_implementation=sdpa \\
        ++evaluate.eval_ckpt=<student ckpt> \\
        ++probe.n_clips=100 ++probe.out=/path/layer_importance.json
"""

from __future__ import annotations

import contextlib
import json

import hydra
import hydra.utils as hyu
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from alpamayo1_5_distill.models.stitched_model import StitchedAlpamayoR1

TEACHER_CKPT = "/data/achahe/alpasim/huggingface/hub/models--nvidia--Alpamayo-1.5-10B-A1-format"
COSMOS = (
    "/data/achahe/alpasim/huggingface/hub/models--nvidia--Cosmos-Reason2-8B/"
    "snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9"
)
#: Same seed for every variant of a clip. The value is arbitrary; using the SAME one is not.
ROLLOUT_SEED = 1234


def _layers(cache):
    """(keys, values) per layer, across the transformers versions this repo has seen."""
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def _set_layer(cache, idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        cache.layers[idx].keys, cache.layers[idx].values = k, v
    else:
        cache.key_cache[idx], cache.value_cache[idx] = k, v


@contextlib.contextmanager
def cache_hook(model, fn):
    """Post-process the cache that ``vlm.generate`` returns, without touching the rollout.

    Wrapping ``generate`` rather than reimplementing
    ``sample_trajectories_from_data_with_vlm_rollout`` keeps the position-id, attention-mask
    and diffusion logic exactly as the eval harness runs it -- a copy would drift.
    """
    orig = model.vlm.generate

    def wrapped(*a, **kw):
        # The PROMPT length, read off the call rather than the batch: the rollout builds
        # `input_ids` internally from tokenized_data, so it never appears in the collated
        # batch. This is also the exact span the expert reads (everything after
        # <|traj_future_start|> is masked out), which is what makes the swap well-posed.
        prompt_len = int(kw["input_ids"].shape[1]) if "input_ids" in kw else int(a[0].shape[1])
        out = orig(*a, **kw)
        fn(out, prompt_len)
        return out

    model.vlm.generate = wrapped
    try:
        yield
    finally:
        model.vlm.generate = orig


def _fresh(batch: dict) -> dict:
    """A batch the rollout can safely consume.

    ``sample_trajectories_from_data_with_vlm_rollout`` does
    ``data["tokenized_data"].pop("input_ids")`` (`alpamayo_r1.py:159`) -- a DESTRUCTIVE read
    of a nested dict. The first rollout therefore leaves every later one with a KeyError, and
    this probe runs ~74 rollouts on the same clip. Copying both levels is enough; the tensors
    themselves are only read.
    """
    out = dict(batch)
    if isinstance(out.get("tokenized_data"), dict):
        out["tokenized_data"] = dict(out["tokenized_data"])
    return out


def rollout(model, batch, hook=None, n_samples: int = 6):
    batch = _fresh(batch)
    torch.manual_seed(ROLLOUT_SEED)  # see the warning in the module docstring
    ctx = cache_hook(model, hook) if hook is not None else contextlib.nullcontext()
    with ctx, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=batch, num_traj_samples=n_samples, num_traj_sets=1,
            top_p=0.98, temperature=0.6, max_generation_length=256,
        )
    return xyz


def traj_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean best-of-K XY displacement between two trajectory sets.

    XY only, matching `distance_metrics.py:28 only_xy=True`; best-of-K so this is comparable
    in spirit to `min_ade` rather than to one arbitrary sample.
    """
    a, b = a.float().reshape(-1, *a.shape[-2:]), b.float().reshape(-1, *b.shape[-2:])
    d = torch.linalg.vector_norm(a[..., :2] - b[..., :2], dim=-1).mean(dim=-1)
    return float(d.min())


def _dump(path, done, base_gap, fix, brk):
    """Write the profile so far. Called every few clips so a kill keeps what it earned."""
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    result = {
        "n_clips": done,
        "seed": ROLLOUT_SEED,
        "baseline_student_teacher_gap": mean(base_gap),
        # Normalised by the baseline gap: the absolute values drift while the denominator
        # settles (4.79 at n=5 -> 3.46 at n=50 on the first run), but the RANKING is what
        # this exists to produce and it is scale-free.
        "fix_gain_frac": {l: (mean(v) / mean(base_gap) if base_gap else float("nan"))
                          for l, v in fix.items()},
        "fix_gain": {l: mean(v) for l, v in fix.items()},
        "break_cost": {l: mean(v) for l, v in brk.items()},
    }
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
    return result


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    probe = cfg.get("probe", {})
    n_clips = int(probe.get("n_clips", 100))
    out_path = probe.get("out", "layer_importance.json")
    directions = list(probe.get("directions", ["fix", "break"]))

    device = torch.device("cuda")
    # Same injection evaluate_hf.py:70 does -- the config ships checkpoint_path: null and
    # expects the arm's checkpoint to arrive via evaluate.eval_ckpt.
    if cfg.evaluate.get("eval_ckpt"):
        cfg.model.checkpoint_path = cfg.evaluate.eval_ckpt
    student = hyu.instantiate(cfg.model, _convert_="partial")
    # ⚠️ device only, NOT .to(dtype=bfloat16). The model already carries its configured
    # dtype, and a blanket cast also hits fp32 buffers (action_in_proj.*.freqs), giving
    # 'mat1 and mat2 must have the same dtype'. evaluate_hf.py:122 instead runs the
    # rollout under torch.autocast, which is what `rollout()` below does.
    student.to(device=device).eval()

    teacher = StitchedAlpamayoR1.from_teacher(
        checkpoint_path=TEACHER_CKPT, vlm_name_or_path=COSMOS,
        attn_implementation="sdpa",
    ).to(device=device).eval()

    n_layers = len(student.vlm.model.language_model.layers)
    print(f"[probe] {n_layers} layers, directions={directions}, n_clips={n_clips}", flush=True)

    dataset = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=student.config)
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=student.config)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=2, shuffle=False)

    fix = {l: [] for l in range(n_layers)}
    brk = {l: [] for l in range(n_layers)}
    base_gap, done = [], 0

    for batch in loader:
        if done >= n_clips:
            break
        batch = send_to_device(batch, device)
        plen: dict[str, int] = {}

        # --- teacher pass: capture the prefix of every layer -------------------------
        cap: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def grab(out, prompt_len):
            plen["n"] = prompt_len
            for i, (k, v) in enumerate(_layers(out.past_key_values)):
                cap[i] = (k[:, :, :prompt_len].clone(), v[:, :, :prompt_len].clone())

        traj_T = rollout(teacher, batch, hook=grab)
        if len(cap) != n_layers:
            raise RuntimeError(f"captured {len(cap)} layers, expected {n_layers}")

        # --- student baseline, and its own captured prefix ---------------------------
        cap_s: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def grab_s(out, prompt_len):
            if prompt_len != plen["n"]:
                raise RuntimeError(
                    f"prompt length differs: teacher {plen['n']} vs student {prompt_len}. "
                    "The swap is only well-posed when the prompts are token-identical."
                )
            for i, (k, v) in enumerate(_layers(out.past_key_values)):
                cap_s[i] = (k[:, :, :prompt_len].clone(), v[:, :, :prompt_len].clone())

        traj_S = rollout(student, batch, hook=grab_s)
        gap = traj_distance(traj_S, traj_T)
        base_gap.append(gap)

        # ⚠️ Shapes must match over the read span or the swap is meaningless. Asserted per
        # clip rather than assumed from the earlier token-id parity check.
        for i in range(n_layers):
            if cap[i][0].shape != cap_s[i][0].shape:
                raise RuntimeError(f"layer {i} prefix shape mismatch {cap[i][0].shape} vs {cap_s[i][0].shape}")

        prompt_len = plen["n"]
        for l in range(n_layers):
            if "fix" in directions:
                def hook_fix(out, prompt_len, l=l):
                    for i, (k, v) in enumerate(_layers(out.past_key_values)):
                        if i == l:
                            k = k.clone(); v = v.clone()
                            k[:, :, :prompt_len] = cap[l][0]
                            v[:, :, :prompt_len] = cap[l][1]
                            _set_layer(out.past_key_values, i, k, v)
                # student cache with layer l repaired -> how much closer to the teacher
                fix[l].append(gap - traj_distance(rollout(student, batch, hook=hook_fix), traj_T))
            if "break" in directions:
                def hook_brk(out, prompt_len, l=l):
                    for i, (k, v) in enumerate(_layers(out.past_key_values)):
                        if i == l:
                            k = k.clone(); v = v.clone()
                            k[:, :, :prompt_len] = cap_s[l][0]
                            v[:, :, :prompt_len] = cap_s[l][1]
                            _set_layer(out.past_key_values, i, k, v)
                # teacher cache with layer l corrupted -> how far it drags the teacher off
                brk[l].append(traj_distance(rollout(teacher, batch, hook=hook_brk), traj_T))

        done += 1
        if done % 5 == 0:
            # ⚠️ DUMP INCREMENTALLY. Writing only at completion means a cancelled run loses
            # everything: this probe was once killed at 50/100 clips and six GPU-hours of
            # per-layer data went with it, because the progress lines carry only the running
            # mean. A partial profile is the whole point of a long sweep.
            _dump(out_path, done, base_gap, fix, brk)
            print(f"[probe] {done}/{n_clips} clips  mean student-teacher gap "
                  f"{sum(base_gap)/len(base_gap):.4f}  (partial dumped)", flush=True)

    result = _dump(out_path, done, base_gap, fix, brk)

    print(f"\n[probe] {done} clips, baseline student-teacher trajectory gap "
          f"{result['baseline_student_teacher_gap']:.4f}")
    print(f"{'layer':>5} {'fix_gain':>10} {'break_cost':>11}")
    for l in range(n_layers):
        print(f"{l:>5} {result['fix_gain'].get(l, float('nan')):>10.4f} "
              f"{result['break_cost'].get(l, float('nan')):>11.4f}")
    print(f"\n[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
