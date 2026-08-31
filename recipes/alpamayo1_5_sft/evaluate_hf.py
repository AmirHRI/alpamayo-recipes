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

import os
from collections import defaultdict
from itertools import islice

import hydra
import hydra.utils as hyu
import json
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer
from alpamayo1_5_sft.trainer import TrainingArguments
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1

from alpamayo.common import (
    distributed,
    misc,
    wandb_utils,
)
from alpamayo_r1.common import logging

from alpamayo_r1.common.logging import setup_logging

setup_logging()

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _add_max_ade_metrics(output_batch: dict) -> None:
    """Add worst-of-K ADE from DistanceMetrics' per-candidate ADE tensor.

    sample_ade is [B, N, K]: K sampled candidates in each of N trajectory
    sets. This mirrors min_ADE's reduction, except it takes the worst candidate.
    """
    for key, value in list(output_batch.items()):
        if not key.endswith("sample_ade") or not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 3:
            raise ValueError(f"{key} must have shape [B,N,K], got {tuple(value.shape)}")
        prefix = key[: -len("sample_ade")]
        output_batch[f"metric/{prefix}max_ade"] = value.amax(dim=2).mean(dim=1)


def _save_trajectory_archive(
    path: str,
    clip_ids: list[str],
    pred_batches: list[np.ndarray],
    gt_batches: list[np.ndarray],
    per_clip_records: list[dict],
    eval_ckpt: str,
) -> int:
    """Write a complete, self-describing trajectory archive and return its clip count."""
    if not pred_batches or not gt_batches:
        raise RuntimeError("trajectory_output was requested but no trajectories were collected")
    pred_xyz = np.concatenate(pred_batches, axis=0).astype(np.float32, copy=False)
    gt_xyz = np.concatenate(gt_batches, axis=0).astype(np.float32, copy=False)
    n = len(clip_ids)
    if pred_xyz.shape[0] != n or gt_xyz.shape[0] != n or len(per_clip_records) != n:
        raise RuntimeError(
            "trajectory archive count mismatch: "
            f"clip_ids={n} pred={pred_xyz.shape[0]} gt={gt_xyz.shape[0]} "
            f"metrics={len(per_clip_records)}"
        )
    metrics = {}
    for name in ("ade", "min_ade", "max_ade"):
        values = [r.get(name) for r in per_clip_records]
        if all(v is not None for v in values):
            metrics[name] = np.asarray(values, dtype=np.float32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        clip_ids=np.asarray(clip_ids),
        pred_xyz=pred_xyz,
        gt_xyz=gt_xyz,
        checkpoint=np.asarray(str(eval_ckpt)),
        num_traj_sets=np.asarray(pred_xyz.shape[1], dtype=np.int64),
        num_traj_samples=np.asarray(pred_xyz.shape[2], dtype=np.int64),
        description=np.asarray("ReasoningSampler xyz trajectories in the ego frame"),
        **metrics,
    )
    return n


@hydra.main(version_base=None, config_path=None, config_name="config")
def evaluate(cfg: DictConfig) -> None:
    distributed.initialize_distributed_simple()

    logger.info(
        "Dataset Configs:\n"
        + misc.pformat(OmegaConf.to_container(cfg.data.val_dataset, resolve=True))
    )
    logger.info(
        "Evaluate Configs:\n" + misc.pformat(OmegaConf.to_container(cfg.evaluate, resolve=True))
    )
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))

    if cfg.evaluate.get("eval_ckpt", None) is not None:
        logger.info(f"Loading model from {cfg.evaluate.eval_ckpt}")
        model_cls = hyu.get_class(cfg.model._target_.rsplit(".", 1)[0])
        if issubclass(model_cls, TrainableReasoningVLA):
            cfg.model.checkpoint_path = cfg.evaluate.eval_ckpt
        elif issubclass(model_cls, TrainableAlpamayoR1):
            cfg.model.pretrained_model_name_or_path = cfg.evaluate.eval_ckpt
            cfg.model.stage1_vlm_checkpoint_path = None
        else:
            raise ValueError(f"Unsupported model class: {model_cls}")
    model = hyu.instantiate(cfg.model, _convert_="partial")

    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )
    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model.config
    )

    trainer = ReasoningVLA_Trainer(
        model=model, args=training_args, eval_dataset=eval_dataset, data_collator=collate_fn
    )
    model = trainer.accelerator.prepare_model(model, evaluation_mode=True)
    model.eval()
    accelerator = trainer.accelerator
    is_main_process = accelerator.is_main_process

    if cfg.get("wandb", None) and is_main_process:
        os.makedirs(cfg.wandb.output_dir, exist_ok=True)
        wandb_utils.init_wandb(**cfg.wandb)

    val_dataloader = trainer.get_eval_dataloader()

    metric_runner = hydra.utils.instantiate(cfg.evaluate.metric_runner)

    max_eval_steps = cfg.evaluate.get("max_eval_steps", -1)
    if max_eval_steps == -1 or max_eval_steps is None:
        dataloader_iter = val_dataloader
        total = len(val_dataloader)
    else:
        dataloader_iter = islice(val_dataloader, max_eval_steps)
        total = max_eval_steps

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    val_count = 0

    # Per-clip metric records for post-hoc grouping (e.g. LCDrive scenario
    # categories). Only collected on the main process; correct as-is for a
    # single-process (nproc_per_node=1) run where each clip is seen exactly once.
    per_clip_records: list[dict] = []
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    trajectory_path = cfg.evaluate.get("trajectory_output", None)
    trajectory_clip_ids: list[str] = []
    trajectory_pred_batches: list[np.ndarray] = []
    trajectory_gt_batches: list[np.ndarray] = []
    if trajectory_path and world_size > 1:
        raise RuntimeError(
            "trajectory_output requires WORLD_SIZE=1; distributed ranks are not gathered "
            "into one ordered clip archive"
        )

    for data in tqdm(dataloader_iter, total=total, disable=not is_main_process):
        output_batch = {}
        with torch.autocast("cuda", dtype=dtype_map[cfg.evaluate.torch_dtype]):
            metric_runner.run(model, data, output_batch)
        _add_max_ade_metrics(output_batch)

        if is_main_process and trajectory_path:
            clip_ids = data.get("clip_id", None)
            pred_xyz = output_batch.get("pred_xyz", None)
            gt_all = data.get("ego_future_xyz", None)
            if clip_ids is None or pred_xyz is None or gt_all is None:
                raise RuntimeError(
                    "trajectory_output needs clip_id, pred_xyz, and ego_future_xyz"
                )
            if pred_xyz.shape[0] != len(clip_ids) or gt_all.shape[0] != len(clip_ids):
                raise RuntimeError(
                    f"trajectory batch mismatch: ids={len(clip_ids)} "
                    f"pred={pred_xyz.shape[0]} gt={gt_all.shape[0]}"
                )
            trajectory_clip_ids.extend(str(cid) for cid in clip_ids)
            trajectory_pred_batches.append(pred_xyz.detach().float().cpu().numpy())
            trajectory_gt_batches.append(gt_all[:, -1].detach().float().cpu().numpy())

        batch_size = len(data["image_frames"])
        gathered_batch_size = accelerator.gather_for_metrics(
            torch.tensor([batch_size], device=accelerator.device, dtype=torch.long)
        )
        if is_main_process:
            val_count += int(gathered_batch_size.sum().item())

        # Collect per-clip scalar metrics (shape [B]) keyed by clip_id.
        if is_main_process:
            clip_ids = data.get("clip_id", None)
            if clip_ids is not None:
                bsz = len(clip_ids)
                per_metric = {}
                for k, v in output_batch.items():
                    if not k.startswith("metric/"):
                        continue
                    if isinstance(v, torch.Tensor) and v.ndim == 1 and v.shape[0] == bsz:
                        per_metric[k[len("metric/") :]] = v.detach().float().cpu().tolist()
                for i, cid in enumerate(clip_ids):
                    rec = {"clip_id": cid}
                    for mk, vals in per_metric.items():
                        rec[mk] = vals[i]
                    per_clip_records.append(rec)

        for k, v in output_batch.items():
            if not k.startswith("metric/"):
                continue
            gathered_metric = accelerator.gather_for_metrics(v)
            if is_main_process:
                metric_sums[k] += gathered_metric.float().sum().item()
                metric_counts[k] += gathered_metric.numel()

    if not is_main_process:
        return

    # Write per-clip metrics so results can be grouped by scenario category etc.
    if per_clip_records:
        if world_size > 1:
            logger.warning(
                f"WORLD_SIZE={world_size} > 1: per-clip metrics reflect the main "
                "process shard only. Run eval with nproc_per_node=1 for a complete "
                "per-clip dump."
            )
        per_clip_path = cfg.evaluate.get("per_clip_output", None)
        if per_clip_path is None:
            per_clip_path = os.path.join(cfg.paths.output_dir, "lcdrive_val_per_clip_metrics.json")
        os.makedirs(os.path.dirname(per_clip_path), exist_ok=True)
        with open(per_clip_path, "w", encoding="utf-8") as f:
            json.dump(per_clip_records, f)
        logger.info(f"Wrote {len(per_clip_records)} per-clip metric records to {per_clip_path}")

    if trajectory_path:
        n_archive = _save_trajectory_archive(
            str(trajectory_path),
            trajectory_clip_ids,
            trajectory_pred_batches,
            trajectory_gt_batches,
            per_clip_records,
            str(cfg.evaluate.get("eval_ckpt", "")),
        )
        logger.info(f"Wrote {n_archive} generated-trajectory records to {trajectory_path}")

    final_metrics_dict = {}
    for key in metric_sums.keys():
        if metric_counts[key] > 0:
            final_metrics_dict["val/" + key] = metric_sums[key] / metric_counts[key]

    padding = 15
    if len(final_metrics_dict) > 0:
        padding = max(padding, *(len(k) for k in final_metrics_dict.keys()))
    logger.info(
        f"Validation @ iteration\n"
        f"{'val/count':<{padding}} {val_count:.4f}\n"
        + "\n".join(
            f"{k:<{padding}} {final_metrics_dict[k]:.4f}" for k in sorted(final_metrics_dict.keys())
        )
        + "\n"
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    evaluate()
