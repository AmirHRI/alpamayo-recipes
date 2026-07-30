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

"""Is the slot gradient asymmetry real, and can 16 k params actually move a loss?

C1  LIKE-FOR-LIKE gradient control. The earlier Q-vs-weight-matrix comparison was
    not apples-to-apples: dL/dW is an outer product accumulated over ~86 positions
    and normalised by a 4.2 M-element tensor, while dL/dQ_i is one backprop path
    through one position normalised by 2048. The fair control compares dL/dQ_i
    against dL/d(embed output) at ORDINARY TEXT POSITIONS in the same forward --
    same tensor, same units, different positions.

C2  Init scale. `randn * 0.02` vs the real embedding table. If real token
    embeddings are much larger, slots start off-manifold, which would both explain
    the gradient magnitude and motivate vocab-initialisation (Lester et al. 2021
    found real-vocab init substantially beats random below ~10 B).

C3  Capacity probe. Gradient flow proves the channel is CONNECTED; it does not
    prove it has enough capacity to matter. Frozen backbone, train only Q on a
    handful of samples, and see whether the loss moves at all -- an early read on
    G3, and a measurement of how many steps prompt-tuning needs here.
"""

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "/data/achahe/alpasim/huggingface/hub/Cosmos-Reason2-2B"
K = 8
dev = torch.device("cuda")
torch.manual_seed(0)

proc = AutoProcessor.from_pretrained(MODEL)
model = (
    Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="sdpa"
    ).to(dev).eval()
)
model.requires_grad_(False)
tok = proc.tokenizer
H = model.config.text_config.hidden_size
emb = model.model.language_model.embed_tokens
PLACEHOLDER = tok(".", add_special_tokens=False).input_ids[0]


def batch_for(seed):
    g = np.random.RandomState(seed)
    img = g.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene."}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    b = proc(text=[t], images=[img], return_tensors="pt").to(dev)
    b["pixel_values"] = b["pixel_values"].to(torch.float32)
    return b


b = batch_for(1)
P = b["input_ids"].shape[1]
ids = torch.cat([b["input_ids"], torch.full((1, K), PLACEHOLDER, device=dev, dtype=torch.long)], 1)
attn = torch.ones_like(ids)
SLOT_POS = torch.arange(P, P + K, device=dev)

# ================================================================= C1
print("=== C1: like-for-like -- dL/dQ vs dL/d(embed out) at TEXT positions ===", flush=True)
Q = torch.nn.Parameter(torch.randn(K, H, device=dev, dtype=torch.float32) * 0.02)
captured = {}


def hook(mod, args, out):
    out = out.clone()
    out[:, SLOT_POS, :] = Q.to(out.dtype)
    out.retain_grad()                    # grab dL/d(embed out) for ALL positions
    captured["e"] = out
    return out


vis_mask = ids[0] == model.config.image_token_id
text_pos = torch.tensor(
    [i for i in range(P) if not bool(vis_mask[i])], device=dev
)  # ordinary text tokens in the prefix

for name, slot_only in (("loss on ALL positions (fair)", False), ("loss on SLOT positions only", True)):
    Q.grad = None
    h = emb.register_forward_hook(hook)
    out = model(input_ids=ids, attention_mask=attn, pixel_values=b["pixel_values"],
                image_grid_thw=b["image_grid_thw"], output_hidden_states=True,
                use_cache=False, return_dict=True)
    h.remove()
    hs = out.hidden_states[-1][0]
    torch.manual_seed(7)
    tgt = hs[SLOT_POS] if slot_only else hs
    (tgt * torch.randn_like(tgt)).sum().backward()

    ge = captured["e"].grad[0]                       # [L, H] -- same tensor for all positions
    q_pe = ge[SLOT_POS].norm(dim=1).mean().item() / (H ** 0.5)
    t_pe = ge[text_pos].norm(dim=1).mean().item() / (H ** 0.5)
    v_pe = ge[vis_mask].norm(dim=1).mean().item() / (H ** 0.5)
    consistent = torch.allclose(Q.grad, ge[SLOT_POS], atol=1e-6)
    print(f"  [{name}]")
    print(f"    per-element grad @ SLOT positions : {q_pe:.4e}")
    print(f"    per-element grad @ TEXT positions : {t_pe:.4e}   ratio slot/text = {q_pe/max(t_pe,1e-30):.2f}")
    print(f"    per-element grad @ VISION positions: {v_pe:.4e}   ratio slot/vis  = {q_pe/max(v_pe,1e-30):.2f}")
    print(f"    dL/dQ == dL/d(embed out)[slots] : {consistent}")

# ================================================================= C2
print("\n=== C2: init scale -- randn*0.02 vs the real embedding table ===", flush=True)
W = emb.weight.detach()
row = W.norm(dim=1)
real_pe = (row.mean() / (H ** 0.5)).item()
init_pe = 0.02
print(f"  real token embeddings : mean row-norm {row.mean():.4f}  -> per-element RMS {real_pe:.4e}")
print(f"  randn*0.02            :                            per-element RMS {init_pe:.4e}")
print(f"  ratio real/init = {real_pe/init_pe:.2f}x")
if real_pe > 2 * init_pe:
    print("  -> randn*0.02 starts the slots WELL BELOW the real-embedding manifold.")
    print("     Supports vocab-initialisation (Lester et al. 2021) and plausibly explains")
    print("     part of the large slot gradient: they start far from where tokens live.")
elif real_pe < 0.5 * init_pe:
    print("  -> randn*0.02 starts the slots ABOVE the real-embedding scale.")
else:
    print("  -> scales are comparable; init magnitude is not the story.")

# ================================================================= C3
print("\n=== C3: capacity probe -- frozen backbone, train ONLY Q ===", flush=True)
N_SAMPLES, STEPS = 16, 120
data = [batch_for(100 + i) for i in range(N_SAMPLES)]
TARGET = tok(" a busy street scene", add_special_tokens=False).input_ids


def build(bi):
    """prompt + K slots + target text; CE on the target only."""
    base = bi["input_ids"]
    p = base.shape[1]
    tgt = torch.tensor(TARGET, device=dev).view(1, -1)
    full = torch.cat([base, torch.full((1, K), PLACEHOLDER, device=dev, dtype=torch.long), tgt], 1)
    labels = full.clone()
    labels[:, : p + K] = -100
    return full, torch.ones_like(full), torch.arange(p, p + K, device=dev), labels


def run_probe(init_kind: str) -> list[float]:
    torch.manual_seed(0)
    if init_kind == "random":
        q = torch.nn.Parameter(torch.randn(K, H, device=dev) * 0.02)
    else:  # vocab: real token embeddings of frequent words
        seed_ids = tok(" the a of to and in is on at it", add_special_tokens=False).input_ids[:K]
        while len(seed_ids) < K:
            seed_ids += seed_ids
        q = torch.nn.Parameter(W[torch.tensor(seed_ids[:K], device=dev)].clone())
    opt = torch.optim.AdamW([q], lr=1e-2)
    losses = []
    for step in range(STEPS):
        bi = data[step % N_SAMPLES]
        full, a, sp, labels = build(bi)

        def hk(mod, args, out):
            out = out.clone()
            out[:, sp, :] = q.to(out.dtype)
            return out

        hh = emb.register_forward_hook(hk)
        o = model(input_ids=full, attention_mask=a, labels=labels,
                  pixel_values=bi["pixel_values"], image_grid_thw=bi["image_grid_thw"],
                  use_cache=False, return_dict=True)
        hh.remove()
        o.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(o.loss.item())
    return losses


# baseline: identical setup, Q frozen at its init (isolates "does TRAINING Q help")
def run_frozen(init_kind: str) -> float:
    torch.manual_seed(0)
    if init_kind == "random":
        q = (torch.randn(K, H, device=dev) * 0.02)
    else:
        seed_ids = tok(" the a of to and in is on at it", add_special_tokens=False).input_ids[:K]
        q = W[torch.tensor(seed_ids[:K], device=dev)].clone()
    tot = 0.0
    with torch.no_grad():
        for bi in data:
            full, a, sp, labels = build(bi)

            def hk(mod, args, out):
                out = out.clone(); out[:, sp, :] = q.to(out.dtype); return out

            hh = emb.register_forward_hook(hk)
            o = model(input_ids=full, attention_mask=a, labels=labels,
                      pixel_values=bi["pixel_values"], image_grid_thw=bi["image_grid_thw"],
                      use_cache=False, return_dict=True)
            hh.remove()
            tot += o.loss.item()
    return tot / len(data)


for kind in ("random", "vocab"):
    base = run_frozen(kind)
    L = run_probe(kind)
    first, last = float(np.mean(L[:10])), float(np.mean(L[-10:]))
    print(f"  [{kind:6s} init] frozen-Q baseline {base:.4f} | trained Q: "
          f"first10 {first:.4f} -> last10 {last:.4f}  (drop {first-last:+.4f}, "
          f"{100*(first-last)/max(first,1e-9):.1f}%)")
    print(f"                 loss trace: {' '.join(f'{v:.3f}' for v in L[::max(1,STEPS//10)])}")

print("\nNOTE C3 is a capacity FLOOR test on a synthetic target with a frozen backbone.")
print("It answers 'can 16k input params move a loss at all', not 'do slots help driving'.")
