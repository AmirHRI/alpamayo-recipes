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

"""Validation harness for reasoning slots on Cosmos-Reason2-2B.

Implements the mechanical checks behind `reasoning-setup-2b.md` §9. Forward-pass
only -- nothing here needs a trained model, and nothing here shows the design
*works*, only that the plumbing is correct and not silently wrong.

Checks
------
S0  placeholder-token safety : the id used for slot positions collides with no
                               vision / special token that get_placeholder_mask
                               or the deepstack path keys on
S1  M-RoPE                   : vision keeps 2-D spatial ids; slots get text-like
                               contiguous ids at prefix_max+1
S2  nested-prefix invariance : h(Q_1..Q_N) identical for N slots vs K slots
S3  no contamination         : prefix hiddens unchanged by appending slots
S4  KV-cache equivalence     : prefill prefix -> feed slots == one pass
S5  M-RoPE negative control  : naive arange positions against a cache are WRONG
S6  slot-only carry (§4)     : slice DynamicCache to slot positions and reuse it
S7  splice instrument (§7d)  : self-splice must be EXACTLY 0; cross-splice must not
S8  carry influence vs age   : does RoPE attenuation decay the carry on its own?

Usage:  python -m alpamayo1_5_distill.scripts.validate_reasoning_slots [K] [dtype]
"""

import sys

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "/temp/achahe/hf_cache/hub/Cosmos-Reason2-2B"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DTYPE = getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.float32

dev = torch.device("cuda")
torch.manual_seed(0)
results: dict[str, bool] = {}


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2. Never use an absolute tolerance here -- see §9's bf16 note."""
    return ((a.float() - b.float()).norm() / a.float().norm()).item()


print(f"[slots] loading {MODEL.split('/')[-1]}  K={K}  dtype={DTYPE}", flush=True)
proc = AutoProcessor.from_pretrained(MODEL)
model = (
    Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=DTYPE, attn_implementation="sdpa"
    )
    .to(dev)
    .eval()
)
tok = proc.tokenizer
H = model.config.text_config.hidden_size


def make_frame(seed: int) -> dict:
    g = np.random.RandomState(seed)
    img = g.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene."}]}
    ]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    b = proc(text=[text], images=[img], return_tensors="pt").to(dev)
    b["pixel_values"] = b["pixel_values"].to(DTYPE)
    return b


# ---------------------------------------------------------------- S0
print("\n=== S0: placeholder-token safety ===", flush=True)
PLACEHOLDER = tok(".", add_special_tokens=False).input_ids[0]
cfg = model.config
reserved = {
    "image_token_id": getattr(cfg, "image_token_id", None),
    "video_token_id": getattr(cfg, "video_token_id", None),
    "vision_start_token_id": getattr(cfg, "vision_start_token_id", None),
    "vision_end_token_id": getattr(cfg, "vision_end_token_id", None),
}
clashes = [n for n, v in reserved.items() if v is not None and v == PLACEHOLDER]
in_special = PLACEHOLDER in set(tok.all_special_ids)
print(f"  placeholder id={PLACEHOLDER} ({tok.convert_ids_to_tokens([PLACEHOLDER])[0]!r})")
print(f"  reserved vision ids: {reserved}")
print(f"  collides with reserved: {clashes or 'none'};  is a special token: {in_special}")
results["S0 placeholder safe"] = not clashes and not in_special

# ---------------------------------------------------------------- setup
f1, f2 = make_frame(1), make_frame(2)
slots = torch.randn(K, H, device=dev, dtype=DTYPE) * 0.02
emb = model.model.language_model.embed_tokens


def with_slots(b):
    ids = torch.cat([b["input_ids"], torch.full((1, K), PLACEHOLDER, device=dev, dtype=torch.long)], 1)
    return ids, torch.ones_like(ids)


def make_hook(slot_pos: torch.Tensor | None):
    """Inject slot embeddings at EXPLICIT positions.

    Indexing by position rather than ``[-K:]`` is deliberate: the trailing-K
    assumption holds only for a single prefill pass and breaks as soon as a
    forward covers just the new tokens against a cache. ``slot_pos`` is indexed
    within the CURRENT forward's window; None disables injection.
    """

    def hook(mod, args, out):
        if slot_pos is None or slot_pos.numel() == 0:
            return out
        n = slot_pos.numel()          # may be < K (nested-prefix runs)
        out = out.clone()
        out[:, slot_pos, :] = slots[:n].to(out.dtype)
        return out

    return hook


def fwd(b, ids=None, attn=None, slot_pos=None, past=None, position_ids=None,
        inputs_embeds=None, pixel=True):
    h = emb.register_forward_hook(make_hook(slot_pos)) if slot_pos is not None else None
    kw = dict(attention_mask=attn, use_cache=True, output_hidden_states=True, return_dict=True)
    if pixel:
        kw.update(pixel_values=b["pixel_values"], image_grid_thw=b["image_grid_thw"])
    if past is not None:
        kw["past_key_values"] = past
    if position_ids is not None:
        kw["position_ids"] = position_ids
    if inputs_embeds is not None:
        kw["inputs_embeds"] = inputs_embeds
    else:
        kw["input_ids"] = ids
    try:
        with torch.inference_mode():
            return model(**kw)
    finally:
        if h:
            h.remove()


ids1, attn1 = with_slots(f1)
P1, L1 = f1["input_ids"].shape[1], ids1.shape[1]
SLOT_POS = torch.arange(P1, P1 + K, device=dev)          # explicit, not [-K:]
pos1, _ = model.model.get_rope_index(ids1, f1["image_grid_thw"], None, attn1)
o1 = fwd(f1, ids1, attn1, SLOT_POS)
hK = o1.hidden_states[-1][0, P1 : P1 + K]
print(f"\n[slots] frame1: prefix={P1} (+{K} slots) = {L1} tokens", flush=True)

# ---------------------------------------------------------------- S1
print("\n=== S1: M-RoPE position ids ===", flush=True)
vis = ids1[0] == cfg.image_token_id
vt, vh, vw = (pos1[i, 0][vis] for i in range(3))
spatial = vh.unique().numel() > 1 and vw.unique().numel() > 1
sp = pos1[:, 0, P1:]
same_axes = bool((sp[0] == sp[1]).all() and (sp[1] == sp[2]).all())
contig = bool((sp[0].diff() == 1).all())
prefix_max = int(pos1[:, 0, :P1].max())
adjacent = int(sp[0, 0]) == prefix_max + 1
print(f"  vision n={int(vis.sum())}: t uniq={vt.unique().numel()} h uniq={vh.unique().numel()} w uniq={vw.unique().numel()} -> spatial={spatial}")
print(f"  slots: t=h=w {same_axes}, contiguous {contig}, start at prefix_max+1 {adjacent}")
print(f"  NOTE prefix is {P1} tokens but its max position id is only {prefix_max} "
      f"({P1 / (prefix_max + 1):.1f}x decoupling) -- {int(vis.sum())} vision tokens span "
      f"{int(vt.max()) - int(vt.min()) + 1}/{vh.unique().numel()}/{vw.unique().numel()} position "
      f"steps on t/h/w, not one step each. This is why naive arange is wrong (S5).")
results["S1 M-RoPE"] = spatial and same_axes and contig and adjacent

# ---------------------------------------------------------------- S2 / S3
print("\n=== S2: nested-prefix invariance ===", flush=True)
ok = True
for N in (1, 2, 4):
    idsN = torch.cat([f1["input_ids"], torch.full((1, N), PLACEHOLDER, device=dev, dtype=torch.long)], 1)
    hN = fwd(f1, idsN, torch.ones_like(idsN), torch.arange(P1, P1 + N, device=dev)).hidden_states[-1][0, P1:]
    r = rel(hK[:N], hN)
    ok &= r < 1e-3
    print(f"  N={N}: rel_L2={r:.3e}")
results["S2 nested-prefix"] = ok

print("\n=== S3: no contamination of the prefix ===", flush=True)
pre = fwd(f1, f1["input_ids"], torch.ones_like(f1["input_ids"]), None)
r3 = rel(pre.hidden_states[-1][0, :P1], o1.hidden_states[-1][0, :P1])
print(f"  rel_L2={r3:.3e}")
results["S3 no contamination"] = r3 < 1e-5

# ---------------------------------------------------------------- S4 / S5
print("\n=== S4: KV-cache equivalence ===", flush=True)
h_inc = fwd(f1, None, attn1, None, past=pre.past_key_values,
            position_ids=pos1[:, :, P1:].contiguous(),
            inputs_embeds=slots.unsqueeze(0), pixel=False).hidden_states[-1][0]
r4 = rel(hK, h_inc)
print(f"  rel_L2={r4:.3e}")
results["S4 KV-cache"] = r4 < 1e-3

print("\n=== S5: negative control -- naive arange positions ===", flush=True)
pre2 = fwd(f1, f1["input_ids"], torch.ones_like(f1["input_ids"]), None)
bad = torch.arange(P1, P1 + K, device=dev).view(1, 1, -1).expand(3, 1, -1).contiguous()
h_bad = fwd(f1, None, attn1, None, past=pre2.past_key_values, position_ids=bad,
            inputs_embeds=slots.unsqueeze(0), pixel=False).hidden_states[-1][0]
r5 = rel(hK, h_bad)
print(f"  rel_L2={r5:.3e}  -> arange is {'WRONG (silently)' if r5 > 1e-2 else 'harmless'}")
results["S5 arange is caught"] = r5 > 1e-2

# ---------------------------------------------------------------- S6
print("\n=== S6: slot-only carry across frames (§4) ===", flush=True)


def slot_only_cache(frame, ids, attn, slot_pos):
    out = fwd(frame, ids, attn, slot_pos)
    c = out.past_key_values
    nb = 0
    for lyr in c.layers:
        lyr.keys = lyr.keys[:, :, P1 : P1 + K, :].contiguous()
        lyr.values = lyr.values[:, :, P1 : P1 + K, :].contiguous()
        nb += lyr.keys.numel() * lyr.keys.element_size() * 2
    return c, nb


cache_A, nbytes = slot_only_cache(f1, ids1, attn1, SLOT_POS)
bf16_mb = nbytes / 2e6 if DTYPE == torch.float32 else nbytes / 1e6
print(f"  cache len -> {cache_A.get_seq_length()} (expect {K});  "
      f"{bf16_mb:.2f} MB at bf16 for K={K}  -> {bf16_mb * 32 / K:.1f} MB at K=32")

ids2, attn2_nc = with_slots(f2)
L2 = ids2.shape[1]
pos2_base, _ = model.model.get_rope_index(ids2, f2["image_grid_thw"], None, attn2_nc)
off = int(pos1.max()) + 1
attn2 = torch.ones((1, K + L2), device=dev, dtype=torch.long)
h_carry = fwd(f2, ids2, attn2, SLOT_POS, past=cache_A,
              position_ids=pos2_base + off).hidden_states[-1][0, P1 : P1 + K]
h_nocarry = fwd(f2, ids2, attn2_nc, SLOT_POS, position_ids=pos2_base).hidden_states[-1][0, P1 : P1 + K]
d_nc = rel(h_nocarry, h_carry)
print(f"  frame2 runs on carry: yes;  rel diff vs no-carry = {d_nc:.3e}")
print("  NOTE: 'influences' != 'helps' -- untrained, so a large delta most likely means the")
print("        carried state is OFF-MANIFOLD, not informative. Only training separates these.")
results["S6 carry connected"] = d_nc > 1e-3

# ---------------------------------------------------------------- S7
print("\n=== S7: splice instrument (§7d correctness gate) ===", flush=True)
cache_self, _ = slot_only_cache(f1, ids1, attn1, SLOT_POS)      # same scene, re-extracted
cache_B, _ = slot_only_cache(f2, ids2, attn2_nc, SLOT_POS)      # different scene
h_self = fwd(f2, ids2, attn2, SLOT_POS, past=cache_self,
             position_ids=pos2_base + off).hidden_states[-1][0, P1 : P1 + K]
h_cross = fwd(f2, ids2, attn2, SLOT_POS, past=cache_B,
              position_ids=pos2_base + off).hidden_states[-1][0, P1 : P1 + K]
d_self = (h_carry.float() - h_self.float()).abs().max().item()
d_cross = rel(h_carry, h_cross)
print(f"  self-splice  max|d| = {d_self:.3e}   (gate: must be EXACTLY 0)")
print(f"  cross-splice rel_L2 = {d_cross:.3e}  (must be non-trivial)")
results["S7 self-splice == 0"] = d_self == 0.0
results["S7 cross-splice acts"] = d_cross > 1e-3

# ---------------------------------------------------------------- S8
print("\n=== S8: does carry influence decay with frame age? (RoPE attenuation) ===", flush=True)
span = int(pos1.max()) + 1
print(f"  holding frame-1 carry fixed, advancing frame-2 positions by age x {span}")
prev = None
for age in (1, 2, 4, 8):
    c_age, _ = slot_only_cache(f1, ids1, attn1, SLOT_POS)
    h_age = fwd(f2, ids2, attn2, SLOT_POS, past=c_age,
                position_ids=pos2_base + age * span).hidden_states[-1][0, P1 : P1 + K]
    d = rel(h_nocarry, h_age)
    arrow = "" if prev is None else ("  (decaying)" if d < prev else "  (growing)")
    print(f"  age={age:2d} frames (offset {age*span:4d}): influence rel={d:.4f}{arrow}")
    prev = d
results["S8 measured"] = True

# ---------------------------------------------------------------- summary
print("\n================= SUMMARY =================")
for k, v in results.items():
    print(f"  {k:26s} {'PASS' if v else 'FAIL'}")
print("\nAll checks are forward-pass mechanics. None of them show the design works;")
print("they show it is wired correctly and fails loudly rather than silently.")
