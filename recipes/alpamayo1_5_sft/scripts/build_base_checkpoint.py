#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build an (untrained) Alpamayo base model from a model YAML and save it as a
released-format HuggingFace checkpoint (config.json + safetensors).

The saved checkpoint is in the ``ReasoningVLA`` "released" format, which is what
AlpaGym's ``convert_release_to_alpagym_checkpoint.py`` consumes. Run this inside
the SFT recipe environment (where ``alpamayo1_5_sft`` is importable):

    cd /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft
    uv run --active python scripts/build_base_checkpoint.py \
      --model-config configs/models/cosmos_reason2_2b.yaml \
      --output /temp/achahe/checkpoints/alpamayo-2b_base_released
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True,
                        help="Path to the model YAML (e.g. configs/models/cosmos_reason2_2b.yaml).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination directory for the released-format checkpoint.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty output directory.")
    args = parser.parse_args()

    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir {output_dir} is not empty; pass --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.model_config)
    print(f"Building model from {args.model_config} ...")
    model = instantiate(cfg)
    model = model.to(torch.bfloat16)

    print(f"Saving released-format checkpoint to {output_dir} ...")
    model.save_pretrained(output_dir, safe_serialization=True)
    print("Done.\n")
    print("Next steps (in the AlpaGym venv):")
    print("  cd /home/achahe/alpagym")
    print("  uv run --no-sync --package alpagym-alpamayo-r1 python \\")
    print("    packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py \\")
    print(f"    --input {output_dir} \\")
    print("    --output /temp/achahe/checkpoints/alpamayo-2b_alpagym_ckpt \\")
    print("    --vlm-name-or-path /temp/achahe/hf_cache/hub/Cosmos-Reason2-2B \\")
    print("    --overwrite")


if __name__ == "__main__":
    main()
