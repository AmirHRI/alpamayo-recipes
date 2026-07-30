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

"""Roll Alpamayo over LCDrive clips frame-by-frame at 10 Hz (open-loop replay).

Captures per frame: the generated chain-of-thought, the meta-action, and the
predicted trajectory -- the raw material for scoring reasoning consistency
across consecutive frames (see score_reasoning_consistency.py).

Two facts drive the setup:

* The LCDrive-trained checkpoints CANNOT produce CoT. Their processor
  (`vla_processor/default.yaml`) has no `cot` in `components_order`, so they were
  never supervised on reasoning and emit an empty span. We therefore roll the
  RELEASED nvidia/Alpamayo-1.5-10B, with a cot-eliciting processor (`cot` LAST,
  so the assistant turn ends at `<|cot_start|>` and the model reasons before
  emitting `<|traj_future_start|>` -- see reasoning-setup-2b.md §9).
* Valid t0 range is [1.6e6, 13.6e6] us (history 1.6 s back, future 6.4 s forward
  inside a 20 s clip), so a 10 Hz sweep gives <=121 frames per clip.

Usage:
  python -m alpamayo1_5_distill.scripts.rollout_frame_sweep \
      --n_clips 6 --out sweep.jsonl [--temperature 0.01] [--max_frames 121]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PAI = "/data/datasets/physical_ai_av/"
MANIFEST_DIR = PAI + "lcdrive_physicalai_av_manifests/"
SCENARIO_CSV = MANIFEST_DIR + "lcdrive_val_primary_scenario_for_table2.csv"
TEACHER = "/data/achahe/alpasim/huggingface/hub/models--nvidia--Alpamayo-1.5-10B-A1-format"

T0_MIN, T0_MAX, T0_STEP = 1_700_000, 13_600_000, 100_000   # 10 Hz, inside the safe margins


def pick_clips(n_clips: int, seed: int = 0) -> list[tuple[str, str]]:
    """Stratified pick across LCDrive val scenario categories (skipping General Driving)."""
    df = pd.read_csv(SCENARIO_CSV)
    df = df[df["scenario_category_paper"] != "General Driving"]
    cats = sorted(df["scenario_category_paper"].unique())
    out: list[tuple[str, str]] = []
    r = 0
    while len(out) < n_clips:
        cat = cats[r % len(cats)]
        sub = df[df["scenario_category_paper"] == cat]
        k = r // len(cats)
        if k < len(sub):
            row = sub.iloc[k]
            out.append((str(row["clip_uuid"]), str(row["scenario_category_paper"])))
        r += 1
        if r > 10_000:
            break
    return out[:n_clips]


def chunks_for(clip_ids: list[str]) -> list[int]:
    idx = pd.read_parquet(PAI + "clip_index.parquet")
    present = [c for c in clip_ids if c in idx.index]
    return sorted({int(idx.at[c, "chunk"]) for c in present})


def build_dataset(pairs: list[tuple[str, int]], model_config, chunk_ids: list[int]):
    """PAIDataset variant driven by an explicit (clip_id, t0_us) list."""
    from alpamayo.data.pai import PAIDataset
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    class FrameSweep(PAIDataset):
        def __init__(self, sweep, **kw):
            super().__init__(**kw)
            self._sweep = sweep

        def __len__(self):
            return len(self._sweep)

        def __getitem__(self, i):
            clip_id, t0_us = self._sweep[i]
            s = load_physical_aiavdataset(
                clip_id, t0_us=int(t0_us), avdi=self.avdi,
                num_history_steps=self.num_history_steps,
                num_future_steps=self.num_future_steps,
                time_step=self.time_step,
            )
            for k in list(s.keys()):
                if k.startswith("ego_"):
                    s[k] = s[k].squeeze(0)
            s["clip_id"] = str(clip_id)
            s["t0_us"] = int(t0_us)
            if self.vla_preprocess_func is not None:
                s["tokenized_data"] = self.vla_preprocess_func(data=s)
            return s

    return FrameSweep(
        pairs,
        local_dir=PAI,
        chunk_ids=chunk_ids,
        use_default_keyframe=True,
        model_config=model_config,
        vla_preprocess_args={
            "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
            "chat_template_version": "r1_5",
            # `cot` LAST => assistant turn ends at <|cot_start|> => the model
            # generates reasoning, then <|traj_future_start|>.
            "components_order": ["image", "traj_history", "prompt", "cot"],
            "components_prompt": ["cot", "traj_future"],
            "label_components": ["traj_future"],
            "generation_mode": True,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clips", type=int, default=6)
    ap.add_argument("--max_frames", type=int, default=121)
    ap.add_argument("--temperature", type=float, default=0.01,
                    help="0.01 ~ greedy (isolates model flips from sampler noise); "
                         "0.6 reproduces the deployment sampler")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    import hydra.utils as hyu  # noqa: F401  (kept for parity with other scripts)
    from transformers.modeling_utils import no_init_weights
    from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1

    dev = torch.device("cuda")
    print("[sweep] loading Alpamayo-1.5-10B ...", flush=True)
    t0 = time.time()
    with no_init_weights():
        model = TrainableAlpamayoR1.from_pretrained(TEACHER, dtype="auto")
    model = model.to(dev).eval()
    model.requires_grad_(False)
    print(f"[sweep] loaded in {time.time()-t0:.0f}s", flush=True)

    clips = pick_clips(args.n_clips)
    ids = [c for c, _ in clips]
    cats = dict(clips)
    ch = chunks_for(ids)
    print(f"[sweep] {len(clips)} clips over chunks {ch}", flush=True)
    for c, cat in clips:
        print(f"          {c}  [{cat}]", flush=True)

    t0s = list(range(T0_MIN, T0_MAX + 1, T0_STEP))[: args.max_frames]
    pairs = [(c, t) for c in ids for t in t0s]
    ds = build_dataset(pairs, model.config, ch)
    from alpamayo.processor.qwen_processor import QwenProcessor

    qp = QwenProcessor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        chat_template_version="r1_5",
    )
    qp.build_processor()

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    t_start = time.time()
    with out_path.open("w") as fh:
        for i in range(len(ds)):
            clip_id, t0_us = pairs[i]
            try:
                sample = ds[i]
                batch = to_dev(qp.collate_fn([sample]))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    res = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=batch,
                        num_traj_samples=1,
                        num_traj_sets=1,        # reshape in the rollout assumes 1
                        top_p=args.top_p,
                        temperature=args.temperature,
                        max_generation_length=args.max_new_tokens,
                        return_extra=True,
                    )
                pred_xyz, _pred_rot, extra = res
                rec: dict[str, Any] = {
                    "clip_id": clip_id,
                    "t0_us": int(t0_us),
                    "scenario": cats.get(clip_id, ""),
                    "cot": str(extra["cot"].reshape(-1)[0]),
                    "meta_action": str(extra["meta_action"].reshape(-1)[0]),
                    "answer": str(extra["answer"].reshape(-1)[0]),
                    # first 20 waypoints are enough for smoothness/divergence
                    "traj_xyz": pred_xyz.reshape(-1, pred_xyz.shape[-2], 3)[0, :20].float().cpu().tolist(),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_ok += 1
            except Exception as e:  # a bad frame must not kill the sweep
                n_err += 1
                fh.write(json.dumps({"clip_id": clip_id, "t0_us": int(t0_us),
                                     "error": f"{type(e).__name__}: {str(e)[:160]}"}) + "\n")
                fh.flush()
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t_start)
                eta = (len(ds) - i - 1) / max(rate, 1e-9) / 60
                print(f"[sweep] {i+1}/{len(ds)}  ok={n_ok} err={n_err}  "
                      f"{rate:.2f} f/s  eta {eta:.1f} min", flush=True)

    print(f"[sweep] DONE ok={n_ok} err={n_err} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
