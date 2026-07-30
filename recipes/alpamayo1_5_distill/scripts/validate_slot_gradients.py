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

"""Does gradient actually reach the reasoning slots Q?

`validate_reasoning_slots.py` covers forward mechanics. This covers the training
path, which can fail *silently*: if gradient never reaches Q, Stages 0/1/2 all
run, converge, and show no gain -- and that reads as "reproduced the pause-token
negative result" when it is really a plumbing bug.

G1  grad reaches Q at all
G2  every slot index gets a DISTINCT non-zero grad (a broadcast bug shows up as
    zeros at some indices, or identical rows)
G3  Q is NOT in model.parameters() -- it is an nn.Parameter living outside the
    model, so it must be added to the optimizer explicitly
G4  ||Q.grad|| vs backbone grad norms -> the LR ratio to expect
G5  grad flows back THROUGH a carried slot cache (needed for §4 truncated BPTT
    inside a window)
"""

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "/data/achahe/alpasim/huggingface/hub/Cosmos-Reason2-2B"
K = 8
dev = torch.device("cuda")
torch.manual_seed(0)
results: dict[str, bool] = {}

proc = AutoProcessor.from_pretrained(MODEL)
model = (
    Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="sdpa"
    )
    .to(dev)
    .eval()
)
tok = proc.tokenizer
H = model.config.text_config.hidden_size
model.requires_grad_(False)          # freeze backbone; we only need grads to Q

g = np.random.RandomState(1)
img = g.randint(0, 255, (224, 224, 3), dtype=np.uint8)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene."}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
b = proc(text=[text], images=[img], return_tensors="pt").to(dev)
b["pixel_values"] = b["pixel_values"].to(torch.float32)
P = b["input_ids"].shape[1]
PLACEHOLDER = tok(".", add_special_tokens=False).input_ids[0]
ids = torch.cat([b["input_ids"], torch.full((1, K), PLACEHOLDER, device=dev, dtype=torch.long)], 1)
attn = torch.ones_like(ids)
SLOT_POS = torch.arange(P, P + K, device=dev)

# THE thing under test: a real nn.Parameter, as it would be in training
Q = torch.nn.Parameter(torch.randn(K, H, device=dev, dtype=torch.float32) * 0.02)
emb = model.model.language_model.embed_tokens


def hook(mod, args, out):
    n = SLOT_POS.numel()
    out = out.clone()
    out[:, SLOT_POS, :] = Q[:n].to(out.dtype)
    return out


# ---------------------------------------------------------------- G1 / G2
print("=== G1/G2: gradient reaches Q, per-slot and distinct ===", flush=True)
h = emb.register_forward_hook(hook)
out = model(input_ids=ids, attention_mask=attn, pixel_values=b["pixel_values"],
            image_grid_thw=b["image_grid_thw"], output_hidden_states=True,
            use_cache=False, return_dict=True)
h.remove()
hs = out.hidden_states[-1][0, P : P + K]
# random projection so each slot gets a different signal (uniform sum could mask a bug)
torch.manual_seed(7)
loss = (hs * torch.randn_like(hs)).sum()
loss.backward()

if Q.grad is None:
    print("  Q.grad is None -- GRADIENT DOES NOT REACH THE SLOTS")
    results["G1 grad reaches Q"] = False
    results["G2 per-slot distinct"] = False
else:
    per = Q.grad.norm(dim=1)
    nonzero = int((per > 0).sum())
    # distinctness: pairwise cosine between slot grads (a broadcast bug -> ~1.0)
    gn = torch.nn.functional.normalize(Q.grad, dim=1)
    cos = (gn @ gn.T)
    off = cos[~torch.eye(K, dtype=torch.bool, device=dev)]
    print(f"  Q.grad present; ||grad|| total = {Q.grad.norm():.4e}")
    print(f"  per-slot norms: {[f'{v:.3e}' for v in per.tolist()]}")
    print(f"  non-zero at {nonzero}/{K} slot indices")
    print(f"  pairwise cosine between slot grads: max={off.max():.4f} mean={off.mean():.4f} "
          f"(≈1.0 everywhere would mean broadcast)")
    results["G1 grad reaches Q"] = bool(Q.grad.abs().sum() > 0)
    results["G2 per-slot distinct"] = nonzero == K and off.max().item() < 0.99

# ---------------------------------------------------------------- G3
print("\n=== G3: is Q picked up by model.parameters()? ===", flush=True)
in_model = any(p is Q for p in model.parameters())
print(f"  Q in model.parameters(): {in_model}")
print("  -> Q lives OUTSIDE the model, so it must be added to the optimizer explicitly:")
print("     optim = AdamW([{'params': model.parameters()}, {'params': [Q], 'lr': lr_Q}])")
results["G3 Q needs explicit optim entry"] = not in_model

# ---------------------------------------------------------------- G4
print("\n=== G4: ||Q.grad|| vs backbone grad norms ===", flush=True)
probe = {
    "lm last layer q_proj": model.model.language_model.layers[-1].self_attn.q_proj.weight,
    "lm last layer mlp.down": model.model.language_model.layers[-1].mlp.down_proj.weight,
    "lm first layer q_proj": model.model.language_model.layers[0].self_attn.q_proj.weight,
}
for p in probe.values():
    p.requires_grad_(True)
Q.grad = None
h = emb.register_forward_hook(hook)
out2 = model(input_ids=ids, attention_mask=attn, pixel_values=b["pixel_values"],
             image_grid_thw=b["image_grid_thw"], output_hidden_states=True,
             use_cache=False, return_dict=True)
h.remove()
hs2 = out2.hidden_states[-1][0, P : P + K]
torch.manual_seed(7)
(hs2 * torch.randn_like(hs2)).sum().backward()
qn = Q.grad.norm().item()
print(f"  ||Q.grad||          = {qn:.4e}   (numel {Q.numel()})")
for n, p in probe.items():
    if p.grad is not None:
        pn = p.grad.norm().item()
        print(f"  ||{n:22s}.grad|| = {pn:.4e}   (numel {p.numel():>10,})  ratio Q/backbone = {qn/max(pn,1e-30):.2f}")
print("  NOTE norms scale with numel; the per-element view is the fairer comparison:")
qe = qn / (Q.numel() ** 0.5)
for n, p in probe.items():
    if p.grad is not None:
        pe = p.grad.norm().item() / (p.numel() ** 0.5)
        print(f"    per-elem  Q {qe:.3e}  vs  {n} {pe:.3e}   ratio {qe/max(pe,1e-30):.2f}")
results["G4 measured"] = True

# ---------------------------------------------------------------- G5
print("\n=== G5: does grad flow back through a CARRIED slot cache? (§4 BPTT) ===", flush=True)
for p in probe.values():
    p.requires_grad_(False)
Q.grad = None
try:
    h = emb.register_forward_hook(hook)
    f1 = model(input_ids=ids, attention_mask=attn, pixel_values=b["pixel_values"],
               image_grid_thw=b["image_grid_thw"], use_cache=True, return_dict=True)
    h.remove()
    cache = f1.past_key_values
    for lyr in cache.layers:                       # slice to slot positions only
        lyr.keys = lyr.keys[:, :, P : P + K, :]
        lyr.values = lyr.values[:, :, P : P + K, :]
    grad_alive = cache.layers[0].keys.requires_grad
    print(f"  carried keys still attached to the graph: {grad_alive}")

    pos, _ = model.model.get_rope_index(ids, b["image_grid_thw"], None, attn)
    off = int(pos.max()) + 1
    attn2 = torch.ones((1, K + ids.shape[1]), device=dev, dtype=torch.long)
    h = emb.register_forward_hook(hook)
    f2 = model(input_ids=ids, attention_mask=attn2, pixel_values=b["pixel_values"],
               image_grid_thw=b["image_grid_thw"], past_key_values=cache,
               position_ids=pos + off, output_hidden_states=True, return_dict=True)
    h.remove()
    hs3 = f2.hidden_states[-1][0, P : P + K]
    torch.manual_seed(7)
    (hs3 * torch.randn_like(hs3)).sum().backward()
    print(f"  ||Q.grad|| after frame-2 backward = {Q.grad.norm():.4e}")
    print("  (includes BOTH the direct frame-2 path and, if attached, the carry path)")
    results["G5 grad through carry"] = bool(grad_alive and Q.grad.abs().sum() > 0)
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
    results["G5 grad through carry"] = False

print("\n================= SUMMARY =================")
for k, v in results.items():
    print(f"  {k:32s} {'PASS' if v else 'FAIL'}")
