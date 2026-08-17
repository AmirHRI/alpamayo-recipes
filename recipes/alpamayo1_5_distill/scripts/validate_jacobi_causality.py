"""Is our Jacobi refinement causally masked among the slots, and is the update rule
the same one PCCoT uses?

Two questions, both answered by measuring the K x K influence matrix
    A[i, j] = || d(output slot i) / d(input slot j) ||
on the REAL Cosmos-Reason2-2B text stack, through the same call `_run_jacobi` makes.

  A strictly lower-triangular  -> causal: output i sees inputs 0..i only
  A dense                      -> bidirectional among slots

Then the update rule. Ours is  input_i <- proj(output_i)  (no shift). PCCoT's is
  input_0 <- unchanged,  input_{i+1} <- proj(output_i)
i.e. shifted by one, which is the autoregressive-consistent Jacobi fixed point for a
causal model: in a causal LM the output at position i predicts position i+1, so feeding
output_i back into position i has no next-token semantics.
"""

import sys

import torch

sys.path.insert(0, "/home/achahe/alpamayo-recipes/recipes")

MODEL = "/temp/achahe/hf_cache/hub/Cosmos-Reason2-2B"
K = 8  # num_slots, matching the trained config
PREFIX = 64  # a short synthetic prefix; the question is about masking, not content


def main() -> None:
    from transformers import Qwen3VLForConditionalGeneration

    dev = "cuda:0"
    print(f"loading {MODEL} ...", flush=True)
    full = (
        Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
        .to(dev)
        .eval()
    )
    # Exactly what _run_jacobi drives: KaVaReasoningVLA._text_model() is
    # `self.vlm.model.language_model`.
    text = full.model.language_model
    cfg = getattr(full.config, "text_config", full.config)
    hidden = int(cfg.hidden_size)
    print(f"  text stack: {type(text).__name__}  hidden={hidden}  layers={len(text.layers)}", flush=True)

    torch.manual_seed(0)
    ids = torch.randint(100, 1000, (1, PREFIX), device=dev)

    with torch.no_grad():
        pre = text(input_ids=ids, use_cache=True)
    cache = pre.past_key_values

    def refine(slot_in: torch.Tensor) -> torch.Tensor:
        """One refinement pass, mirroring _run_jacobi's call exactly."""
        import copy

        c = copy.deepcopy(cache)
        mask = torch.ones((1, PREFIX + K), dtype=torch.long, device=dev)
        pos = torch.arange(PREFIX, PREFIX + K, device=dev).view(1, -1)
        out = text(
            inputs_embeds=slot_in,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=c,
            use_cache=True,
            cache_position=torch.arange(PREFIX, PREFIX + K, device=dev),
        )
        return out.last_hidden_state

    base_in = torch.randn(1, K, hidden, device=dev) * 0.03  # ~embedding RMS
    with torch.no_grad():
        base_out = refine(base_in)

    print(f"\ninfluence matrix  A[i,j] = ||d out_i / d in_j||   (rows=output, cols=input)")
    A = torch.zeros(K, K)
    eps = 0.05
    for j in range(K):
        pert = base_in.clone()
        pert[0, j] += eps
        with torch.no_grad():
            out = refine(pert)
        d = (out - base_out)[0].norm(dim=-1)  # [K]
        A[:, j] = d.float().cpu()

    hdr = "        " + "".join(f"in{j:<7d}" for j in range(K))
    print(hdr)
    for i in range(K):
        row = "".join(f"{A[i, j]:<8.4f}" for j in range(K))
        print(f"  out{i}  {row}")

    upper = sum(A[i, j].item() for i in range(K) for j in range(K) if j > i)
    lower = sum(A[i, j].item() for i in range(K) for j in range(K) if j <= i)
    print(f"\n  sum of strictly-upper (out_i influenced by LATER in_j): {upper:.6f}")
    print(f"  sum of lower+diagonal                                  : {lower:.6f}")
    verdict = "CAUSAL (lower-triangular)" if upper < 1e-4 else "BIDIRECTIONAL"
    print(f"  => refinement among slots is {verdict}")
    if upper < 1e-4:
        print("\n  Consequence: out_0 depends on the prefix ONLY -- no other slot can ever")
        print("  reach it. Information flows forward along slot order, never backward.")
        print("  PCCoT's shift (input_{i+1} <- out_i) is what makes that AR-consistent;")
        print("  our unshifted input_i <- out_i converges to a different fixed point.")


if __name__ == "__main__":
    main()
