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

r"""Linear CKA between the teacher action expert's layers (Kornblith et al., 2019).

Layer-to-layer representational similarity for the frozen action expert, following the
recipe in CLP_VLA (``notebooks/cka.ipynb``): Gram matrix -> double-centering -> HSIC ->
normalisation, computed **per sample over the action tokens** and averaged across clips.

.. math::
    \mathrm{CKA}(X, Y) = \frac{\mathrm{HSIC}(X, Y)}
                              {\sqrt{\mathrm{HSIC}(X,X)}\,\sqrt{\mathrm{HSIC}(Y,Y)}},
    \quad \mathrm{HSIC}(X,Y) = \langle HXX^\top H,\; HYY^\top H \rangle_F

Rows are the 64 action tokens, columns the 2048 expert channels, so this asks whether two
layers induce the same similarity structure *over waypoints*.

**What this measures, and what it does not.** CKA is a *representational* statement: high
similarity between adjacent layers means they encode the waypoint set alike, which is
evidence of redundancy and hence a pruning/weighting candidate. It is NOT a causal claim
about the trajectory. This project already learned that distinction the expensive way --
the marginal cache-swap sweep found layers 0-11 contributing 0.1% of the outcome, and
down-weighting them on that basis (``ARM=kvband``) made min_ade WORSE (3.1763 vs 2.6313
uniform, z=+11.22). Treat a CKA block as a hypothesis to test causally, never as a
license to reweight.

⚠️ Note this is the ACTION EXPERT's own hidden states, a different object from the earlier
sweep, which perturbed the *VLM cache* the expert reads.

Usage::

    CUDA_VISIBLE_DEVICES=3 python -m alpamayo1_5_distill.scripts.expert_cka \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_eval_stitched_4b_lcdrive \
        ++model.attn_implementation=sdpa ++cka.n_clips=64 ++cka.step=0
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
from transformers.cache_utils import DynamicCache  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from alpamayo1_5_distill.models.expert_teacher import build_noisy_action  # noqa: E402
from alpamayo1_5_distill.models.stitched_model import StitchedAlpamayoR1  # noqa: E402

TRAIN = "/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training"
TEACHER_CKPT = "/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format"
COSMOS = (
    "/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/"
    "snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9"
)


# ---------------------------------------------------------------------------------------
# Reference implementation, verbatim from CLP_VLA notebooks/cka.ipynb (cair-vinuni/CLP_VLA),
# itself the standard Kornblith et al. formulation. Kept EXACTLY as published so the fast
# path below can be checked against it rather than trusted.
# ---------------------------------------------------------------------------------------
def _ref_centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return np.dot(np.dot(H, K), H)


def _ref_linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(_ref_centering(L_X) * _ref_centering(L_Y))


def _ref_linear_CKA(X, Y):
    hsic = _ref_linear_HSIC(X, Y)
    var1 = np.sqrt(_ref_linear_HSIC(X, X))
    var2 = np.sqrt(_ref_linear_HSIC(Y, Y))
    return hsic / (var1 * var2)


# ---------------------------------------------------------------------------------------
# Fast path: algebraically identical, but centres each layer's Gram ONCE instead of
# recomputing it inside every one of the L^2 pairs. For 36 layers that is 36 centerings
# rather than ~2600, and the (64 x 64) Grams are 16 KB each so everything stays in memory.
# ---------------------------------------------------------------------------------------
def centered_grams(acts: np.ndarray) -> np.ndarray:
    """(B, T, D) activations -> (B, T, T) double-centred Gram matrices."""
    g = acts @ acts.transpose(0, 2, 1)
    g -= g.mean(axis=1, keepdims=True)
    g -= g.mean(axis=2, keepdims=True)
    g += g.mean(axis=(1, 2), keepdims=True)
    return g


def cka_matrix_from_grams(grams: list[np.ndarray]) -> np.ndarray:
    """Per-sample linear CKA averaged over samples, for every layer pair."""
    n = len(grams)
    norms = [np.sqrt(np.einsum("bij,bij->b", g, g)) for g in grams]
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            hsic = np.einsum("bij,bij->b", grams[i], grams[j])
            out[i, j] = out[j, i] = float(np.mean(hsic / (norms[i] * norms[j])))
    return out


def _selftest(acts: list[np.ndarray]) -> None:
    """⚠️ The fast path must reproduce the published reference EXACTLY, and CKA(X,X) must
    be 1. An unverified reimplementation is how the first L_block came out silently wrong."""
    i, j, b = 0, min(3, len(acts) - 1), 0
    ref = _ref_linear_CKA(acts[i][b].astype(np.float64), acts[j][b].astype(np.float64))
    g = [centered_grams(a[b : b + 1].astype(np.float64)) for a in (acts[i], acts[j])]
    fast = cka_matrix_from_grams(g)[0, 1]
    ident = cka_matrix_from_grams([g[0], g[0]])[0, 1]
    print(f"[cka] selftest  reference {ref:.10f}  fast {fast:.10f}  "
          f"|diff| {abs(ref - fast):.2e}   CKA(X,X) {ident:.10f}", flush=True)
    assert abs(ref - fast) < 1e-8, "fast path disagrees with the published reference"
    assert abs(ident - 1.0) < 1e-8, "CKA(X,X) != 1"


def cache_kv(cache, n_layers):
    """(K, V) per layer out of a Cache, across transformers layouts. Cloned: the expert
    forward appends the action tokens' own K/V in place, so an uncloned view would grow."""
    out = []
    for i in range(n_layers):
        if hasattr(cache, "layers"):
            k, v = cache.layers[i].keys, cache.layers[i].values
        else:
            k, v = cache.key_cache[i], cache.value_cache[i]
        out.append((k.clone(), v.clone()))
    return out


def expert_layers(model):
    """The frozen expert's decoder layers, however this build nests them."""
    for path in ("expert.layers", "expert.expert.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    raise AttributeError("could not locate the expert decoder layers")


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    c = cfg.get("cka", {})
    n_clips = int(c.get("n_clips", 64))
    step = int(c.get("step", 0))          # which denoising step to read (0 = t=0, sampler's first)
    out_dir = c.get("out_dir", f"{TRAIN}/cka")
    os.makedirs(out_dir, exist_ok=True)

    module = str(c.get("module", "expert"))       # "expert" | "vlm"
    # ⚠️ front_only changes the MODEL'S INPUT, so it measures a different operating point --
    # the teacher never runs with one camera. It is a comparison between two states, not a
    # cheaper measurement of the deployed one. Filtering happens BEFORE preprocessing because
    # the image placeholders in input_ids are emitted per image; dropping images afterwards
    # would desync tokens from pixels.
    if bool(c.get("front_only", False)):
        import alpamayo.data.pai as _pai
        from alpamayo.common.constants import CAMERA_NAMES_TO_INDICES, FRONT_WIDE_CAMERA_NAME
        _orig = _pai.load_physical_aiavdataset
        _want = CAMERA_NAMES_TO_INDICES[FRONT_WIDE_CAMERA_NAME]
        _nf = int(c.get("n_frames", 4))

        def _front_only(*a, **kw):
            d = _orig(*a, **kw)
            ci = d["camera_indices"]
            keep = (ci == _want).nonzero().flatten()
            n_chunk = int(ci.shape[0])
            d["camera_indices"] = ci[keep]
            for k, v in list(d.items()):
                if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n_chunk and k != "camera_indices":
                    d[k] = v[keep]
            for k in ("image_frames", "absolute_timestamps", "relative_timestamps"):
                if k in d and torch.is_tensor(d[k]) and d[k].dim() >= 2:
                    d[k] = d[k][:, :_nf]
            return d

        _pai.load_physical_aiavdataset = _front_only
        print(f"[cka] FRONT-WIDE ONLY, {_nf} frames -> {_nf} images "
              f"(vs 7 cameras normally)", flush=True)
    n_tokens = int(c.get("n_tokens", 256))
    device = torch.device("cuda")
    model = StitchedAlpamayoR1.from_teacher(
        checkpoint_path=TEACHER_CKPT, vlm_name_or_path=COSMOS, attn_implementation="sdpa"
    ).to(device=device).eval()
    if module == "vlm":
        layers = model.vlm.model.language_model.layers
        path = "vlm.model.language_model.layers"
    else:
        layers, path = expert_layers(model)
    print(f"[cka] {module} layers: {len(layers)} via model.{path}", flush=True)

    # ⚠️ SEEDED RANDOM sample, and the chosen ids are written out. Earlier runs took the
    # dataset's first n clips (shuffle=False + break), which is a prefix, not a sample: on the
    # 39k-clip train list that would be one corner of the dataset, and no run recorded WHICH
    # clips it used, so nothing was auditable after the fact.
    pool_file = c.get("clip_filter") or cfg.data.val_dataset.clip_uuid_filter
    pool = sorted({ln.strip() for ln in open(pool_file) if ln.strip()})
    rng = np.random.default_rng(int(c.get("sample_seed", 0)))
    chosen = sorted(rng.choice(pool, size=min(n_clips, len(pool)), replace=False).tolist())
    uuid_file = os.path.join(out_dir, "_clips_used.txt")
    with open(uuid_file, "w") as fh:
        fh.write("\n".join(chosen) + "\n")
    print(f"[cka] pool {len(pool)} clips from {os.path.basename(pool_file)} "
          f"-> sampled {len(chosen)} (seed {int(c.get('sample_seed', 0))})", flush=True)

    ds = hyu.instantiate({**cfg.data.val_dataset, "clip_uuid_filter": uuid_file},
                         _convert_="partial", model_config=model.config)
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=2, shuffle=False)

    if module == "vlm":
        # ⚠️ NO t axis here, deliberately. The VLM runs ONCE, before denoising, and its cache
        # is consumed unchanged at every Euler step -- there is no x_t to condition on. A
        # "VLM CKA at t=0.5" would be the same numbers relabelled.
        # ⚠️ Only the PREFILL is recorded (seq_len > 1). The AR decode steps that follow have
        # seq_len == 1, whose 1x1 Gram makes CKA degenerate (0/0 after centering).
        vstore: dict[int, list[np.ndarray]] = {i: [] for i in range(len(layers))}
        done = {"n": -1}

        def vhook(idx):
            def f(_m, _args, output):
                y = output[0] if isinstance(output, tuple) else output
                if y.shape[1] <= 1 or done["n"] == len(vstore[idx]) - 1:
                    return                          # decode step, or this clip already taken
                if idx == 0:
                    print(f"[cka] PREFILL seq_len = {y.shape[1]}", flush=True)
                a = y.detach().float().cpu().numpy()[0]          # (S, D)
                # Evenly spaced positions: sequences differ in length across clips, and the
                # prefix mixes image patches with text, so an even stride keeps both.
                idxs = np.linspace(0, a.shape[0] - 1, min(n_tokens, a.shape[0])).astype(int)
                vstore[idx].append(a[idxs])
            return f

        hs = [layers[i].register_forward_hook(vhook(i)) for i in range(len(layers))]
        try:
            for n, batch in enumerate(loader):
                if n >= n_clips:
                    break
                cid = batch["clip_id"]
                seen = cid[0] if isinstance(cid, list) else str(cid)
                gpu = send_to_device(dict(batch), device)
                torch.manual_seed(1234)
                done["n"] = n - 1
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=gpu, num_traj_samples=1, num_traj_sets=1,
                        top_p=0.98, temperature=0.6, max_generation_length=256,
                    )
                if (n + 1) % 10 == 0:
                    print(f"[cka] {n + 1}/{n_clips} clips", flush=True)
        finally:
            for h in hs:
                h.remove()

        acts = [np.stack(vstore[i]) for i in range(len(layers))]
        print(f"[cka] activations per layer: {acts[0].shape} (clips, tokens, channels)",
              flush=True)
        _selftest(acts)
        grams = [centered_grams(a.astype(np.float64)) for a in acts]
        Mv = cka_matrix_from_grams(grams)
        np.savez(os.path.join(out_dir, "cka_vlm_teacher.npz"), cka=Mv,
                 n_clips=len(acts[0]), n_tokens=acts[0].shape[1], clips=np.array(chosen))
        np.save(os.path.join(out_dir, "reps_vlm.npy"), np.stack(acts).astype(np.float16))
        fig, ax = plt.subplots(figsize=(7.2, 6.1))
        im = ax.imshow(Mv, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("linear CKA", fontsize=9)
        tk = list(range(0, len(layers), 4))
        ax.set_xticks(tk); ax.set_yticks(tk)
        ax.set_xticklabels(tk, fontsize=8); ax.set_yticklabels(tk, fontsize=8)
        ax.set_xlabel("VLM layer"); ax.set_ylabel("VLM layer")
        ax.set_title(f"Teacher VLM — layer CKA\nn={len(acts[0])} clips, "
                     f"{acts[0].shape[1]} prefill tokens", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cka_vlm_teacher.png"), dpi=130); plt.close(fig)
        adj = [Mv[i, i + 1] for i in range(len(layers) - 1)]
        print(f"[cka] VLM adjacent: mean {np.mean(adj):.4f}  min {min(adj):.4f} "
              f"(L{int(np.argmin(adj))}->L{int(np.argmin(adj)) + 1})  "
              f"corner CKA(L0,L{len(layers) - 1})={Mv[0, -1]:.4f}", flush=True)
        return

    if str(c.get("source", "gt_interp")) == "rollout":
        # ⚠️ ON-POLICY: x_t is whatever the model's OWN Euler loop visits, not
        # t*x_GT + (1-t)*eps. No GT is used at all. The expert is invoked once per denoising
        # step, so invocation index == step, and `_euler` uses t_i = i/inference_step ->
        # steps land on t = 0.0 .. 0.9 (there is no forward pass AT t=1.0; the last step
        # integrates 0.9 -> 1.0). Comparing this against the GT-interpolation run measures
        # the train/inference distribution gap in representation space.
        n_steps = int(c.get("n_steps", 10))
        rstore: dict[tuple[int, int], list[np.ndarray]] = {}
        cur_step = {"i": -1}

        def step_hook(_m, _args, _kw):
            cur_step["i"] += 1
            return None

        def rhook(idx):
            def f(_m, _args, output):
                i = cur_step["i"]
                if 0 <= i < n_steps:
                    y = output[0] if isinstance(output, tuple) else output
                    rstore.setdefault((i, idx), []).append(
                        y.detach().float().cpu().numpy()[0])
            return f

        ph = model.expert.register_forward_pre_hook(step_hook, with_kwargs=True)
        hs = [layers[i].register_forward_hook(rhook(i)) for i in range(len(layers))]
        seen_clips = []
        try:
            for n, batch in enumerate(loader):
                if n >= n_clips:
                    break
                cid = batch["clip_id"]
                seen_clips.append(cid[0] if isinstance(cid, list) else str(cid))
                cur_step["i"] = -1
                gpu = send_to_device(dict(batch), device)
                torch.manual_seed(1234)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=gpu, num_traj_samples=1, num_traj_sets=1,
                        top_p=0.98, temperature=0.6, max_generation_length=256,
                    )
                if (n + 1) % 10 == 0:
                    print(f"[cka] {n + 1}/{n_clips} clips", flush=True)
        finally:
            ph.remove()
            for h in hs:
                h.remove()

        steps = sorted({k[0] for k in rstore})
        print(f"[cka] captured denoising steps: {steps}", flush=True)
        mats = {}
        for si in steps:
            acts = [np.stack(rstore[(si, i)]) for i in range(len(layers))]
            if si == steps[0]:
                print(f"[cka] activations per layer: {acts[0].shape}", flush=True)
                _selftest(acts)
            mats[si] = cka_matrix_from_grams(
                [centered_grams(a.astype(np.float64)) for a in acts])
            adj = [mats[si][l - 1, l] for l in range(1, len(layers))]
            print(f"[cka] step {si} (t={si / n_steps:.2f})  corner "
                  f"{mats[si][0, -1]:.4f}  mean adj {np.mean(adj):.4f}  min adj "
                  f"{min(adj):.4f} (L{int(np.argmin(adj))}->L{int(np.argmin(adj)) + 1})",
                  flush=True)
        C = np.stack([[mats[si][l - 1, l] for l in range(1, len(layers))] for si in steps])
        np.save(os.path.join(out_dir, "C_adjacent_cka.npy"), C)
        np.savez(os.path.join(out_dir, "cka_expert_rollout.npz"),
                 steps=np.array(steps), clip_ids=np.array(seen_clips),
                 **{f"cka_s{si}": mats[si] for si in steps})
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(steps) + 2.0))
        im = ax.imshow(C, aspect="auto", cmap="viridis", vmin=float(C.min()), vmax=1.0,
                       interpolation="nearest")
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([f"step {si} (t={si / n_steps:.1f})" for si in steps], fontsize=8)
        ax.set_xticks(range(0, len(layers) - 1, 2))
        ax.set_xticklabels(list(range(1, len(layers), 2)), fontsize=7)
        ax.set_xlabel("layer transition  l-1 -> l", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015).set_label("adjacent CKA", fontsize=8)
        ax.set_title(f"Action expert, ON-POLICY x_t from the model's own sampler\n"
                     f"dark = the layer acts   [scale {C.min():.3f}-1.000]", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "C_adjacent_cka.png"), dpi=140); plt.close(fig)
        print("[cka] wrote C_adjacent_cka.png", flush=True)
        return

    # ⚠️ t is swept over the flow, not fixed at 0. `build_noisy_action` reproduces
    # FlowMatching's own `noisy_x = t*x + (1-t)*noise` from the clip's GROUND-TRUTH
    # trajectory, so every (x_t, t) pair is on-distribution -- an arbitrary t with x~N(0,I)
    # would be valid only at t=0. One noise draw is shared across the whole grid (seeded CPU
    # generator), so the ONLY thing varying along it is t.
    ts = [float(x) for x in str(c.get("timesteps", "0.0,0.25,0.5,0.75,1.0")).split(",")]
    print(f"[cka] timestep grid: {ts}", flush=True)

    store: dict[tuple[float, int], list[np.ndarray]] = {(t, i): [] for t in ts
                                                        for i in range(len(layers))}
    grab: dict[str, object] = {}
    seen_clips: list[str] = []

    def pre_hook(_m, _args, kwargs):
        # Fires on the rollout's FIRST expert call, before it appends action K/V, so this is
        # the VLM prefix exactly as the expert receives it at the handoff.
        if "prefix" not in grab and kwargs.get("past_key_values") is not None:
            grab["prefix"] = cache_kv(kwargs["past_key_values"], len(layers))
            grab["kwargs"] = {k: v for k, v in kwargs.items()
                              if k in ("attention_mask", "position_ids")}
        return None

    cur = {"t": None}

    def hook(idx):
        def f(_m, _args, output):
            if cur["t"] is not None:
                y = output[0] if isinstance(output, tuple) else output
                store[(cur["t"], idx)].append(y.detach().float().cpu().numpy()[0])
        return f

    ph = model.expert.register_forward_pre_hook(pre_hook, with_kwargs=True)
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(len(layers))]
    try:
        for n, batch in enumerate(loader):
            if n >= n_clips:
                break
            grab.clear()
            cid = batch["clip_id"]
            seen_clips.append(cid[0] if isinstance(cid, list) else str(cid))
            gpu = send_to_device(dict(batch), device)
            torch.manual_seed(1234)
            cur["t"] = None                     # rollout itself is not recorded
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model.sample_trajectories_from_data_with_vlm_rollout(
                    data=gpu, num_traj_samples=1, num_traj_sets=1,
                    top_p=0.98, temperature=0.6, max_generation_length=256,
                )
            if "prefix" not in grab:
                raise RuntimeError("never saw the expert's prefix cache")

            traj = {k: gpu[k] for k in ("ego_history_xyz", "ego_history_rot",
                                        "ego_future_xyz", "ego_future_rot")}
            shared_noise = None
            for t in ts:
                x_t, t_vec, shared_noise = build_noisy_action(
                    model, traj, t, noise=shared_noise, seed=1234
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    embeds = model.action_in_proj(x_t, t_vec)
                    if embeds.dim() == 2:
                        embeds = embeds.view(x_t.shape[0], -1, embeds.shape[-1])
                    # Fresh cache per t, re-seeded from the SAME prefix -- mirrors the
                    # inference loop's crop_cache between Euler steps. Without this the
                    # action K/V from the previous t stay appended.
                    fresh = DynamicCache()
                    for i, (k, v) in enumerate(grab["prefix"]):
                        fresh.update(k, v, i, {})
                    cur["t"] = t
                    model.expert(inputs_embeds=embeds, past_key_values=fresh,
                                 use_cache=True, **grab["kwargs"])
                    cur["t"] = None
            if (n + 1) % 10 == 0:
                print(f"[cka] {n + 1}/{n_clips} clips", flush=True)
    finally:
        ph.remove()
        for h in handles:
            h.remove()

    # ⚠️ Persist the REPRESENTATIONS, not only the derived matrices. Every re-analysis
    # (a different similarity measure, a finer band split, probing) otherwise costs another
    # GPU pass. Two artefacts, because they answer different questions:
    #   reps   fp16 (n_t, L, B, T, D) -- everything, ~0.6 GB per timestep
    #   grams  fp32 (n_t, L, B, T, T) -- the SUFFICIENT STATISTIC for any linear CKA, 16x
    #          smaller, since centred Gram matrices are all the estimator ever reads.
    save_reps = bool(c.get("save_reps", True))
    mats, all_grams = {}, {}
    for ti, t in enumerate(ts):
        acts = [np.stack(store[(t, i)]) for i in range(len(layers))]
        if ti == 0:
            print(f"[cka] activations per layer: {acts[0].shape} "
                  f"(clips, action tokens, channels)", flush=True)
            _selftest(acts)
        grams = [centered_grams(a.astype(np.float64)) for a in acts]
        all_grams[t] = np.stack(grams).astype(np.float32)
        if save_reps:
            np.save(os.path.join(out_dir, f"reps_t{t:.2f}.npy"),
                    np.stack(acts).astype(np.float16))
        mats[t] = cka_matrix_from_grams(grams)
        adj = [mats[t][i, i + 1] for i in range(len(layers) - 1)]
        print(f"[cka] t={t:.2f}  corner CKA(L0,L35)={mats[t][0, -1]:.4f}  "
              f"mean adjacent={np.mean(adj):.4f}  min adjacent={min(adj):.4f} "
              f"(L{int(np.argmin(adj))}->L{int(np.argmin(adj)) + 1})", flush=True)
    np.savez(os.path.join(out_dir, "cka_expert_teacher_tsweep.npz"),
             timesteps=np.array(ts), n_clips=len(store[(ts[0], 0)]),
             clip_ids=np.array(seen_clips), pool=str(pool_file),
             **{f"cka_t{t:.2f}": mats[t] for t in ts})
    np.savez_compressed(os.path.join(out_dir, "grams_tsweep.npz"),
                        timesteps=np.array(ts), **{f"g_t{t:.2f}": all_grams[t] for t in ts})

    # C[n, l] -- rows are denoising steps, columns layer transitions l-1 -> l. High means
    # the layer barely changes the representation at that point on the flow.
    C = np.stack([[mats[t][l - 1, l] for l in range(1, len(layers))] for t in ts])
    np.save(os.path.join(out_dir, "C_adjacent_cka.npy"), C)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(ts) + 2.0))
    im = ax.imshow(C, aspect="auto", cmap="viridis", vmin=float(C.min()), vmax=1.0,
                   interpolation="nearest")
    ax.set_yticks(range(len(ts))); ax.set_yticklabels([f"t={t:.2f}" for t in ts], fontsize=8)
    ax.set_xticks(range(0, len(layers) - 1, 2))
    ax.set_xticklabels(list(range(1, len(layers), 2)), fontsize=7)
    ax.set_xlabel("layer transition  l-1 -> l", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015).set_label("adjacent-layer CKA",
                                                               fontsize=8)
    # ⚠️ Scaled to the data range, NOT [0,1]: adjacent CKA sits in ~[0.96, 1.0] and a full
    # [0,1] scale renders the whole panel one flat colour. The range is in the title so the
    # compression is not hidden by the stretch.
    ax.set_title(f"Where the expert transforms its representation, per denoising step\n"
                 f"dark = the layer acts; bright = near-identity   "
                 f"[scale {C.min():.3f}-1.000]", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "C_adjacent_cka.png"), dpi=140); plt.close(fig)
    M = mats[ts[0]]

    # CKA is a similarity in [0, 1] -- magnitude with no meaningful midpoint -- so this is a
    # SEQUENTIAL single hue, light->dark. (The source paper uses a diverging RdBu_r; a
    # diverging map implies a neutral centre this quantity does not have, and red/green-family
    # ramps fail colour-vision deficiency.)
    fig, axes = plt.subplots(1, len(ts), figsize=(3.3 * len(ts) + 1.2, 3.9))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, ts):
        im = ax.imshow(mats[t], vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ticks = list(range(0, len(layers), 6))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=7); ax.set_yticklabels(ticks, fontsize=7)
        ax.set_title(f"t = {t:.2f}", fontsize=10)
        ax.set_xlabel("expert layer", fontsize=8)
    axes[0].set_ylabel("expert layer", fontsize=8)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02).set_label("linear CKA",
                                                                         fontsize=8)
    fig.suptitle(f"Teacher action expert — layer CKA across the flow "
                 f"(n={len(store[(ts[0], 0)])} clips, GT-derived x_t)", fontsize=11)
    p = os.path.join(out_dir, "cka_expert_teacher_tsweep.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[cka] wrote {p}", flush=True)




if __name__ == "__main__":
    main()
