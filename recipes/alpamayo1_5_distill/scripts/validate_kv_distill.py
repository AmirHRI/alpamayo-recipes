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

"""Validate the KAVA KV-distillation machinery on the real Cosmos-Reason2-2B.

Same shape as ``validate_reasoning_slots.py``: named checks, a PASS/FAIL summary,
and **relative** tolerances only — hidden states here reach magnitudes ~1350
("massive activations") where one bf16 ulp is 8, so an absolute threshold reports
1-ulp agreement as a failure.

The checks target things that fail *silently* — every one of them would otherwise
leave a model that trains, converges, and supervises nothing:

* **KV1** the pre-RoPE hook captures the right tensor: rotate what the hook saw and
  it must reproduce ``DynamicCache``'s post-RoPE keys.
* **KV2** eviction actually varies per (layer, head), and the paper's ablation
  variants (``lam=1`` / ``lam=0`` / ``crop``) really differ.
* **KV3** the importance score comes from a genuine attention distribution, with the
  GQA group MaxPool applied.
* **KV4** slot columns line up: ``slot_pos`` matches the placeholders, and the
  ``<traj_future_start>`` handoff column still follows the slots after splicing.
* **KV5** gradient reaches the slot embeddings *and* the backbone through ``L_KV``,
  with per-slot-distinct gradients (a broadcast bug gives cosine ~1.0).
* **KV6** the self-consistency floor: with teacher ≡ student geometry, ``L_KV`` must
  be drivable to ~0 by the slots alone.  This separates "the objective is mis-wired"
  from "the two models' K/V bases do not align".
* **KV7** cache round-trip, and offline recompression reproducing the inline result.

Runs on the raw ``Qwen3VLForConditionalGeneration`` rather than the Alpamayo wrapper
so it needs no PAI data and no teacher checkpoint — what is under test is the
capture/eviction/loss mechanism, which is shared.

Reproduce::

    python -m alpamayo1_5_distill.scripts.validate_kv_distill [K] [dtype]
"""

import sys
import tempfile

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from alpamayo1_5_distill.data import kv_cache_io
from alpamayo1_5_distill.data.kava_dataset import pad_kv_to_budget, splice_slot_placeholders
from alpamayo1_5_distill.models.kv_distill import (
    KVProjectorBank,
    build_layer_map,
    combine_scores,
    evict_teacher_cache,
    importance_score,
    kv_matching_loss,
    redundancy_score,
    select_top_m,
)

MODEL = "/temp/achahe/hf_cache/hub/Cosmos-Reason2-2B"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DTYPE = getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.float32
dev = torch.device("cuda")
results: dict[str, bool] = {}


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 — never an absolute tolerance (see the module docstring)."""
    return float((a.float() - b.float()).norm() / max(b.float().norm().item(), 1e-12))


torch.manual_seed(0)
print(f"=== loading {MODEL} (K={K}, dtype={DTYPE}) ===", flush=True)
proc = AutoProcessor.from_pretrained(MODEL)
model = (
    Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=DTYPE, attn_implementation="sdpa")
    .to(dev)
    .eval()
)
tok = proc.tokenizer
text_model = model.model.language_model
cfg = model.config.text_config
N_LAYERS, N_KV, HEAD_DIM = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
HIDDEN = cfg.hidden_size
print(
    f"  layers={N_LAYERS} hidden={HIDDEN} kv_heads={N_KV} head_dim={HEAD_DIM} "
    f"kv_width={N_KV * HEAD_DIM}",
    flush=True,
)

# A real vision+text batch, as in validate_reasoning_slots.py.
g = np.random.RandomState(1)
img = g.randint(0, 255, (224, 224, 3), dtype=np.uint8)
msgs = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene."}],
    }
]
chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
batch = proc(text=[chat], images=[img], return_tensors="pt").to(dev)
batch["pixel_values"] = batch["pixel_values"].to(DTYPE)
PREFIX = batch["input_ids"].shape[1]
PLACEHOLDER = tok(".", add_special_tokens=False).input_ids[0]

ids = torch.cat(
    [batch["input_ids"], torch.full((1, K), PLACEHOLDER, device=dev, dtype=torch.long)], dim=1
)
attn = torch.ones_like(ids)
slot_pos = torch.arange(PREFIX, PREFIX + K, device=dev).unsqueeze(0)
rows = torch.zeros((1, 1), dtype=torch.long, device=dev)

vision_kwargs = {"pixel_values": batch["pixel_values"], "image_grid_thw": batch["image_grid_thw"]}


# --------------------------------------------------------------------- KV1
print("\n=== KV1: the pre-RoPE hook captures the tensor we think it does ===", flush=True)
store: dict[int, dict[str, torch.Tensor]] = {}


def make_k_hook(idx: int):
    def hook(_m, _a, out):
        store.setdefault(idx, {})["k"] = out.detach().clone()

    return hook


def make_v_hook(idx: int):
    def hook(_m, _a, out):
        store.setdefault(idx, {})["v"] = out.detach().clone()

    return hook


handles = []
for i, layer in enumerate(text_model.layers):
    handles.append(layer.self_attn.k_norm.register_forward_hook(make_k_hook(i)))
    handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))
with torch.no_grad():
    out = model(input_ids=ids, attention_mask=attn, use_cache=True, **vision_kwargs)
for h in handles:
    h.remove()

cache = out.past_key_values
position_ids, _ = model.model.get_rope_index(ids, batch["image_grid_thw"], None, attn)
cos, sin = text_model.rotary_emb(store[0]["k"], position_ids)
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb  # noqa: E402

# hook output is [B, T, H, D]; the model transposes to [B, H, T, D] before rotating.
k_hooked = store[0]["k"].permute(0, 2, 1, 3)
_, k_rotated = apply_rotary_pos_emb(k_hooked, k_hooked, cos, sin)
k_cached = cache.layers[0].keys
r_k = rel(k_rotated, k_cached)
# V is never rotated, so the hook output should equal the cache exactly.
v_hooked = store[0]["v"].view(1, -1, N_KV, HEAD_DIM).permute(0, 2, 1, 3)
r_v = rel(v_hooked, cache.layers[0].values)
print(f"  rotate(hooked pre-RoPE K) vs cache K : rel-L2 = {r_k:.3e}")
print(f"  hooked V              vs cache V     : rel-L2 = {r_v:.3e}")
print(f"  hook K shape {tuple(store[0]['k'].shape)}  cache K shape {tuple(k_cached.shape)}")
results["KV1 pre-RoPE hook is correct"] = r_k < 1e-4 and r_v < 1e-5

# Why pre-RoPE is the right target, measured rather than asserted: rotate the SAME
# keys at shifted positions and the post-RoPE tensor moves. A post-RoPE objective
# would therefore force the student's slots to reproduce a phase that depends on where
# the teacher's CoT token happened to sit — and eviction has already scrambled that
# ordering, so the phase it would be chasing is meaningless.
#
# MEASUREMENT, not a gate. The effect turns out to be modest on this model, and the
# number is reported rather than thresholded so nobody has to re-derive it: Qwen3-VL
# uses rope_theta = 5e6, which leaves the low-frequency dimensions almost unrotated
# even over thousands of positions, so only the top frequencies move.
#
# Read it as: pre-RoPE matching is a well-founded default that costs nothing (the hook
# is no harder than reading the cache), NOT as a large measured win. The stronger part
# of the argument stays structural — eviction reorders which CoT token each slot
# targets, so any position-dependent component of the target is arbitrary by
# construction, however small it is.
print(f"  pre-RoPE vs post-RoPE                : rel-L2 = {rel(k_hooked, k_cached):.3f}")
for shift in (30, 300, 3000):
    _, k_shifted = apply_rotary_pos_emb(
        k_hooked, k_hooked, *text_model.rotary_emb(store[0]["k"], position_ids + shift)
    )
    print(f"  same keys rotated +{shift:<5d} positions   : rel-L2 = {rel(k_shifted, k_cached):.3f}")


# --------------------------------------------------------------------- KV2
print("\n=== KV2: eviction varies per (layer, head) and the ablations differ ===", flush=True)
N_C = 24
k_cot = torch.stack([store[i]["k"][0, :N_C].permute(1, 0, 2) for i in range(N_LAYERS)]).float()
v_cot = torch.stack(
    [store[i]["v"][0, :N_C].view(N_C, N_KV, HEAD_DIM).permute(1, 0, 2) for i in range(N_LAYERS)]
).float()
print(f"  synthetic CoT span: k={tuple(k_cot.shape)}  (L, H, N_C, D)")

red = redundancy_score(k_cot)
imp = importance_score(torch.rand(N_LAYERS, N_KV, N_C, device=dev))
idx_rkv = select_top_m(combine_scores(imp, red, 0.1, "rkv"), K, N_C)
idx_attn = select_top_m(combine_scores(imp, red, 0.1, "attn"), K, N_C)
idx_cos = select_top_m(combine_scores(imp, red, 0.1, "cosine"), K, N_C)
idx_crop = select_top_m(red, K, N_C, method="crop")

unique_sets = {tuple(r.tolist()) for r in idx_rkv.reshape(-1, K)}
print(f"  distinct index sets across {N_LAYERS * N_KV} (layer, head) pairs: {len(unique_sets)}")
print(f"  rkv vs attn differ: {not torch.equal(idx_rkv, idx_attn)}")
print(f"  rkv vs cosine differ: {not torch.equal(idx_rkv, idx_cos)}")
print(f"  crop == first {K}: {bool((idx_crop == torch.arange(K, device=dev)).all())}")
print(f"  rkv indices ascending (temporal order kept): {bool((idx_rkv.diff(dim=-1) > 0).all())}")
results["KV2 eviction is per-head and ablations differ"] = (
    len(unique_sets) > 1
    and not torch.equal(idx_rkv, idx_attn)
    and not torch.equal(idx_rkv, idx_cos)
    and bool((idx_crop == torch.arange(K, device=dev)).all())
    and bool((idx_rkv.diff(dim=-1) > 0).all())
)


# --------------------------------------------------------------------- KV3
print("\n=== KV3: importance comes from a real attention distribution ===", flush=True)
answer_lo = PREFIX + K - 2  # stand-in for [<|cot_end|>, <|traj_future_start|>]
for c in (model.config, model.config.text_config):
    c._attn_implementation = "eager"
with torch.no_grad():
    eager_out = text_model(
        inputs_embeds=model.get_input_embeddings()(ids),
        attention_mask=attn,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=True,
    )
for c in (model.config, model.config.text_config):
    c._attn_implementation = "sdpa"

attns = eager_out.attentions
print(f"  attentions: {len(attns)} layers, shape {tuple(attns[0].shape)}")
row_sums = attns[0][0, :, -1, :].sum(-1)
print(f"  rows sum to 1: min={row_sums.min():.6f} max={row_sums.max():.6f}")

n_q = attns[0].shape[1]
n_rep = n_q // N_KV
grouped = attns[0].view(1, N_KV, n_rep, attns[0].shape[2], attns[0].shape[3])
pooled = grouped.max(dim=2).values
print(f"  GQA MaxPool: {n_q} q-heads -> {N_KV} kv-heads (group of {n_rep})")
print(f"  pooled >= mean over the group: {bool((pooled >= grouped.mean(dim=2)).all())}")
imp_real = pooled[..., answer_lo:].mean(dim=-2)[..., :N_C].float()
imp_norm = importance_score(imp_real)
print(f"  normalised importance sums to 1: {float(imp_norm.sum(-1).min()):.6f}")
results["KV3 importance is a real attention distribution"] = (
    len(attns) == N_LAYERS
    and bool((row_sums - 1.0).abs().max() < 1e-3)
    and bool((pooled >= grouped.mean(dim=2)).all())
    and bool((imp_norm.sum(-1) - 1.0).abs().max() < 1e-4)
)


# --------------------------------------------------------------------- KV4
print("\n=== KV4: spliced slot columns line up with the expert's crop point ===", flush=True)
TFS, COT_S, COT_E = 900, 901, 902
raw = torch.tensor([[7, 7, 7, TFS, 5, 6], [0, 0, 7, TFS, 5, 6]], device=dev)
spliced = splice_slot_placeholders(
    raw,
    tfs_id=TFS,
    cot_start_id=COT_S,
    cot_end_id=COT_E,
    placeholder_id=PLACEHOLDER,
    num_slots=K,
    attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]], device=dev),
    labels_mask=torch.zeros((2, 6), dtype=torch.bool, device=dev),
)
new_ids, sp = spliced["input_ids"], spliced["slot_pos"]
ok_placeholder = bool((new_ids.gather(1, sp) == PLACEHOLDER).all())
tfs_after = (new_ids == TFS).float().argmax(dim=1)
ok_order = bool((tfs_after == sp[:, -1] + 2).all())  # slots, <|cot_end|>, then <|tfs|>
ok_width = new_ids.shape[1] == raw.shape[1] + K + 2
print(f"  slot_pos points at placeholders: {ok_placeholder}")
print(f"  <tfs> is 2 columns after the last slot (cot_end between): {ok_order}")
print(f"  width {raw.shape[1]} -> {new_ids.shape[1]} (+K+2): {ok_width}")
print(f"  slots are INSIDE the crop at future_start_idx+1: {bool((sp < tfs_after[:, None]).all())}")
results["KV4 slot columns and crop point agree"] = ok_placeholder and ok_order and ok_width


# --------------------------------------------------------------------- KV5
print("\n=== KV5: gradient reaches the slots AND the backbone through L_KV ===", flush=True)
model.requires_grad_(False)
probe = text_model.layers[-1].self_attn.k_proj.weight
probe.requires_grad_(True)
slots = torch.nn.Parameter(torch.randn(K, HIDDEN, device=dev, dtype=DTYPE) * 0.02)

capture: dict[int, dict[str, torch.Tensor]] = {}
handles = []


def inject(_m, _a, out):
    out = out.clone()
    out[rows, slot_pos] = slots.to(out.dtype)
    return out


def make_k_cap(idx: int):
    def hook(_m, _a, out):
        capture.setdefault(idx, {})["k"] = out[rows, slot_pos].permute(0, 2, 1, 3)

    return hook


def make_v_cap(idx: int):
    def hook(_m, _a, out):
        shaped = out.view(out.shape[0], out.shape[1], N_KV, HEAD_DIM)
        capture.setdefault(idx, {})["v"] = shaped[rows, slot_pos].permute(0, 2, 1, 3)

    return hook


handles.append(model.get_input_embeddings().register_forward_hook(inject))
for i, layer in enumerate(text_model.layers):
    handles.append(layer.self_attn.k_norm.register_forward_hook(make_k_cap(i)))
    handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_cap(i)))
model(input_ids=ids, attention_mask=attn, use_cache=False, **vision_kwargs)
for h in handles:
    h.remove()

teacher_layers = N_LAYERS + 8  # pretend a deeper teacher, as 28 -> 36 really is
mapping = build_layer_map(N_LAYERS, teacher_layers)
t_k = torch.randn(1, teacher_layers, N_KV, K, HEAD_DIM, device=dev)
t_v = torch.randn(1, teacher_layers, N_KV, K, HEAD_DIM, device=dev)
bank = KVProjectorBank(N_LAYERS, kv_width=N_KV * HEAD_DIM, align="projector").to(dev, DTYPE)
student_kv = {i: (c["k"], c["v"]) for i, c in capture.items()}
loss = kv_matching_loss(student_kv, t_k, t_v, mapping, projector=bank, kind="smooth_l1")
loss.backward()

per_slot = slots.grad.norm(dim=1)
gn = torch.nn.functional.normalize(slots.grad.float(), dim=1)
off = (gn @ gn.T)[~torch.eye(K, dtype=torch.bool, device=dev)]
print(f"  L_KV = {loss.item():.6f}  over {len(student_kv)} captured layers")
print(f"  ||slots.grad|| = {slots.grad.norm():.4e}, non-zero at {int((per_slot > 0).sum())}/{K}")
print(f"  per-slot grad cosine: max={off.max():.4f} mean={off.mean():.4f} (~1.0 = broadcast bug)")
print(f"  backbone k_proj grad: {probe.grad.norm():.4e}")
print(f"  projector grad: {bank.k_proj[0].weight.grad.norm():.4e}")
results["KV5 gradient reaches slots, backbone and projector"] = (
    len(student_kv) == N_LAYERS
    and float(slots.grad.abs().sum()) > 0
    and int((per_slot > 0).sum()) == K
    and float(off.max()) < 0.99
    and probe.grad is not None
    and float(probe.grad.abs().sum()) > 0
    and float(bank.k_proj[0].weight.grad.abs().sum()) > 0
)
probe.requires_grad_(False)


# --------------------------------------------------------------------- KV6
print("\n=== KV6: self-consistency floor — can L_KV be driven to ~0? ===", flush=True)
# Teacher targets taken from the model's OWN K/V at the slot positions, so a perfect
# solution provably exists. If this cannot descend, the objective is mis-wired; that
# is a different diagnosis from "the 10B's and 2B's K/V bases do not align".
#
# The step budget is load-bearing: at 30 steps this bottoms out around 44% of the
# starting loss and looks like a failure, while 200 steps with a decaying LR reaches
# ~15%. The slot -> per-layer-K/V map is 28 layers deep and the slots attend to each
# other, so this is a slow optimisation, not a broken one.
STEPS = 200
target_k = torch.stack([capture[i]["k"].detach() for i in range(N_LAYERS)], dim=1)
target_v = torch.stack([capture[i]["v"].detach() for i in range(N_LAYERS)], dim=1)
identity_map = list(range(N_LAYERS))

fresh = torch.nn.Parameter(torch.randn(K, HIDDEN, device=dev, dtype=DTYPE) * 0.02)
opt = torch.optim.Adam([fresh], lr=5e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
trace = []
for step in range(STEPS):
    capture.clear()
    handles = []

    def inject_fresh(_m, _a, out):
        out = out.clone()
        out[rows, slot_pos] = fresh.to(out.dtype)
        return out

    handles.append(model.get_input_embeddings().register_forward_hook(inject_fresh))
    for i, layer in enumerate(text_model.layers):
        handles.append(layer.self_attn.k_norm.register_forward_hook(make_k_cap(i)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_cap(i)))
    model(input_ids=ids, attention_mask=attn, use_cache=False, **vision_kwargs)
    for h in handles:
        h.remove()

    step_loss = kv_matching_loss(
        {i: (c["k"], c["v"]) for i, c in capture.items()},
        target_k,
        target_v,
        identity_map,
        kind="l1",
    )
    opt.zero_grad()
    step_loss.backward()
    opt.step()
    sched.step()
    trace.append(step_loss.item())

print(f"  L_KV trace: {' · '.join(f'{trace[i]:.4f}' for i in range(0, STEPS, STEPS // 8))}")
print(f"  start {trace[0]:.4f} -> end {trace[-1]:.5f}  ({trace[-1] / trace[0]:.1%} of start)")
# The recovered slots need NOT resemble the ones that generated the target: the
# slot -> K/V map has enough slack that matching the cache does not pin the embedding.
# That is the representational slack KAVA relies on, and it is why token-level
# correspondence is not required — worth reporting, not asserting.
print(
    f"  ||recovered - target slots|| / ||target|| = "
    f"{float((fresh - slots).detach().norm() / slots.detach().norm()):.2f}"
    "  (low L_KV does not pin the embedding)"
)
results["KV6 L_KV descends toward zero on a solvable target"] = trace[-1] < 0.25 * trace[0]


# --------------------------------------------------------------------- KV7
print("\n=== KV7: cache round-trip and offline recompression agree ===", flush=True)
with tempfile.TemporaryDirectory() as tmp:
    key = "0badc0de-0000-4000-8000-000000000001::5100000"
    imp_cpu, red_cpu = imp.cpu(), red.cpu()
    kv_cache_io.save_full_entry(
        tmp, key, k_cot.cpu(), v_cot.cpu(), imp=imp_cpu, red=red_cpu, tfs_hidden=torch.randn(4096)
    )
    kv_cache_io.write_index(tmp, {key: N_C}, {"mode": "test"})

    inline_k, inline_v, inline_idx = evict_teacher_cache(
        k_cot.cpu().float(), v_cot.cpu().float(), m=K, n_valid=N_C, imp=imp_cpu, red=red_cpu
    )
    tag = kv_cache_io.compressed_tag(K, 0.1, "rkv")
    kv_cache_io.save_compressed_entry(tmp, tag, key, inline_k, inline_v, sel_idx=inline_idx)

    full = kv_cache_io.load_entry(tmp, "full", key)
    offline_k, offline_v, offline_idx = evict_teacher_cache(
        full["k_pre"].float(),
        full["v"].float(),
        m=K,
        n_valid=int(kv_cache_io.read_index(tmp)["n_cot"][key]),
        imp=full["imp"],
        red=full["red"],
    )
    comp = kv_cache_io.load_entry(tmp, tag, key)
    same_idx = torch.equal(offline_idx, inline_idx)
    r_round = rel(comp["k_pre"].float(), inline_k.to(torch.bfloat16).float())
    print(f"  full entry tensors: {sorted(full)}")
    print(f"  offline recompression picks the same indices: {same_idx}")
    print(f"  compressed round-trip rel-L2: {r_round:.3e} (bf16 storage)")
    padded_k, _, valid = pad_kv_to_budget(comp["k_pre"], comp["v"], num_slots=K + 4)
    print(f"  pad to a larger budget: {tuple(padded_k.shape)}, valid={int(valid.sum())}/{K + 4}")
    results["KV7 cache round-trip and recompression agree"] = (
        same_idx and r_round == 0.0 and int(valid.sum()) == K
    )


# ------------------------------------------------------------------- summary
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
for name, ok in results.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print("=" * 72)
print(f"{sum(results.values())}/{len(results)} checks passed", flush=True)
sys.exit(0 if all(results.values()) else 1)
