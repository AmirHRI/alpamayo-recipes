"""PAI Stage-1 SFT training entry point for AutoVLA.

Mirrors the logic of SKIPlan's ``tools/run_sft.py`` but replaces
``SFTDataset`` with ``PAISFTDataset`` and removes all nuPlan-specific
infrastructure (PDMS callback, codebook, metric cache).

Usage
-----
    cd /home/achahe/alpamayo-recipes/recipes/autovla_pai
    source autovla_env/bin/activate

    # Single-node 4-GPU
    [CUDA]_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \\
        -m autovla_pai.train \\
        --config configs/pai_sft_stage1.yaml

    # Inspect first batch only (no training)
    python -m autovla_pai.train --config configs/pai_sft_stage1.yaml --print_sample
"""
from __future__ import annotations

import argparse
import datetime
import functools
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# ── project paths ─────────────────────────────────────────────────────────────
RECIPES_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # alpamayo-recipes
AUTOVLA_PAI_ROOT = Path(__file__).resolve().parent.parent             # recipes/autovla_pai
SKIPPLAN_ROOT = Path("/home/achahe/SKIPlan")

for p in [SKIPPLAN_ROOT, AUTOVLA_PAI_ROOT]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor

from models.structured_stage3 import StructuredStage3Module

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextDecoderLayer,
        Qwen3VLVisionBlock,
    )
    _HAS_QWEN3 = True
except ImportError:
    Qwen3VLTextDecoderLayer = Qwen3VLVisionBlock = None
    _HAS_QWEN3 = False

from autovla_pai.pai_sft_dataset import PAISFTDataset, _build_action_space

torch.set_float32_matmul_precision("high")


# ── helpers ───────────────────────────────────────────────────────────────────

def _env_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except (TypeError, ValueError):
        return 0


def _is_rank0() -> bool:
    return _env_rank() == 0


def _resolve_save_dir(base: str, cfg_path: str) -> str:
    """Agree on one save dir across all torchrun ranks before dist init."""
    base_p = Path(base)
    base_p.mkdir(parents=True, exist_ok=True)
    launch_id = os.environ.get("TORCHELASTIC_RUN_ID") or f"ppid{os.getppid()}"
    stamp = base_p / f".train_pai_{launch_id}_{Path(cfg_path).stem}.txt"
    if _is_rank0():
        sd = str(base_p / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        stamp.write_text(sd)
        return sd
    for _ in range(300):
        if stamp.exists():
            return stamp.read_text().strip()
        time.sleep(0.1)
    return str(base_p / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


class _FixCheckpointDir(pl.Callback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def state_dict(self): return {}
    def load_state_dict(self, sd): pass

    def on_train_start(self, trainer, pl_module):
        for cb in trainer.checkpoint_callbacks:
            if cb.dirpath != self.save_dir:
                cb.dirpath = self.save_dir


# ── collator ─────────────────────────────────────────────────────────────────
# We cannot import DataCollator from sft_dataset.py because that module does a
# top-level `from navsim.agents.autovla_agent import AutoVLAAgent` which pulls
# in nuplan-devkit. This self-contained collator replicates the exact logic
# needed for PAI SFT (continuous action targets, no discrete codebook tokens).

class PAIDataCollator:
    """Tokenise PAI SFT samples and build the batch that StructuredStage3Module expects."""

    IGNORE_INDEX = -100
    ASSISTANT_ID = [151644, 77091]   # <|im_start|> + "assistant" for Qwen3-VL

    def __init__(self, processor):
        self.processor = processor
        self._asst_id = torch.tensor(self.ASSISTANT_ID)

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            return None

        texts = [f["text"] for f in features]
        video_inputs = [f["video_inputs"][0] for f in features]   # list of PIL-frame lists
        all_fps = [f["video_kwargs"]["fps"][0] for f in features]

        # Build processor kwargs so video fps is passed correctly
        from transformers.video_utils import VideoMetadata
        video_metadata = [
            VideoMetadata(
                total_num_frames=len(vi),
                fps=fps,
                frames_indices=list(range(len(vi))),
            )
            for vi, fps in zip(video_inputs, all_fps)
        ]

        batch = self.processor(
            text=texts,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            video_metadata=video_metadata,
            do_sample_frames=False,
        )

        # Build labels: mask everything before the assistant response
        labels = batch["input_ids"].clone()
        asst = self._asst_id
        for i in range(labels.shape[0]):
            start_idx = labels.shape[1]  # default: mask all
            for j in range(len(labels[i]) - len(asst) + 1):
                if torch.equal(labels[i, j:j + len(asst)], asst):
                    start_idx = j
                    break
            labels[i, :start_idx] = self.IGNORE_INDEX
            pad_mask = labels[i] == self.processor.tokenizer.pad_token_id
            labels[i, pad_mask] = self.IGNORE_INDEX
        batch["labels"] = labels

        # Trajectory GT
        batch["gt_trajectory"]      = torch.stack([f["gt_trajectory"] for f in features])
        batch["gt_action"]          = torch.stack([f["gt_action"] for f in features])
        batch["gt_action_alpamayo"] = torch.cat( [f["gt_action_alpamayo"] for f in features], dim=0)
        batch["has_cot"]            = torch.tensor([f["has_cot"] for f in features])

        # History
        batch["ego_history_xyz"] = torch.cat([f["ego_history_xyz"] for f in features], dim=0)
        batch["ego_history_rot"] = torch.cat([f["ego_history_rot"] for f in features], dim=0)

        # Ego conditioning
        batch["swin_ego_state"]   = torch.cat([f["swin_ego_state"]   for f in features], dim=0)
        batch["vehicle_velocity"] = torch.cat([f["vehicle_velocity"] for f in features], dim=0)

        return batch


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pai_sft_stage1.yaml")
    parser.add_argument("--print_sample", action="store_true",
                        help="Print first batch and exit without training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = AUTOVLA_PAI_ROOT / cfg_path
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    pai_cfg = config["pai"]
    model_cfg = config["model"]

    processor = AutoProcessor.from_pretrained(
        model_cfg["pretrained_model_path"], use_fast=True
    )
    action_space = _build_action_space(
        n_waypoints=model_cfg["alpamayo"]["n_waypoints"],
        dt=model_cfg["trajectory"]["interval_length"],
    )

    if _is_rank0():
        print(f"[PAI SFT] train chunks: {pai_cfg['train_chunk_ids']}")
        print(f"[PAI SFT]   val chunks: {pai_cfg['val_chunk_ids']}")

    train_ds = PAISFTDataset(
        pai_dir=pai_cfg["pai_dir"],
        chunk_ids=pai_cfg["train_chunk_ids"],
        processor=processor,
        action_space=action_space,
        img_max_pixels=model_cfg["video"]["max_pixels"],
    )
    val_ds = PAISFTDataset(
        pai_dir=pai_cfg["pai_dir"],
        chunk_ids=pai_cfg["val_chunk_ids"],
        processor=processor,
        action_space=action_space,
        img_max_pixels=model_cfg["video"]["max_pixels"],
    )
    if _is_rank0():
        print(f"[PAI SFT] train={len(train_ds)}  val={len(val_ds)}")

    collator = PAIDataCollator(processor)

    if args.print_sample:
        sample = train_ds[0]
        print("=== sample keys:", list(sample.keys()))
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            elif isinstance(v, str):
                print(f"  {k}: (str len={len(v)})\n{v[:300]}")
            else:
                print(f"  {k}: {type(v).__name__}")
        batch = collator([train_ds[i] for i in range(min(4, len(train_ds)))])
        print("\n=== collated batch keys:", list(batch.keys()))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}")
        return

    # ── model ─────────────────────────────────────────────────────────────────
    model = StructuredStage3Module(config)
    model.autovla.vlm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ── checkpoint loading ────────────────────────────────────────────────────
    resume_path = config["training"].get("resume_from_checkpoint")
    resume_mode = config["training"].get("resume_training_mode", "full")

    if resume_path and not Path(resume_path).exists():
        print(f"WARNING: checkpoint not found: {resume_path}")
        resume_path = None

    if resume_path and resume_mode == "weights_only":
        ckpt = torch.load(resume_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if _is_rank0():
            print(f"Loaded weights from {resume_path}")
            if missing:
                print(f"  missing  ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
            if unexpected:
                print(f"  unexpected ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
        resume_path = None   # don't pass to trainer.fit — optimizer starts fresh

    # ── data loaders ──────────────────────────────────────────────────────────
    use_dist = dist.is_initialized()
    train_sampler = DistributedSampler(train_ds, shuffle=True)  if use_dist else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if use_dist else None

    train_cfg = config["training"]
    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        collate_fn=collator,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=train_cfg["num_workers"],
        collate_fn=collator,
        drop_last=False,
    )

    # Expose train dataset size for auto lr-schedule computation
    config["training"]["_train_dataset_size"] = len(train_ds)

    # ── trainer setup ─────────────────────────────────────────────────────────
    devices = train_cfg.get("devices", "auto")
    n_dev = len(devices) if isinstance(devices, (list, tuple)) else devices
    use_fsdp = n_dev != 1

    save_dir = _resolve_save_dir(train_cfg.get("checkpoint_dir", "runs/sft"), str(cfg_path))

    if _is_rank0():
        import shutil
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_path, Path(save_dir) / "config.yaml")
        print(f"[PAI SFT] saving to {save_dir}")

    loggers = [CSVLogger(save_dir=save_dir)]
    wandb_conf = config.get("wandb", {})
    if wandb_conf.get("enable", False) and _is_rank0():
        loggers.append(WandbLogger(
            project=wandb_conf.get("project", "AutoVLA-PAI"),
            name=wandb_conf.get("name", "pai-sft-stage1") + f"-{Path(save_dir).name}",
            save_dir=save_dir,
        ))

    ckpt_every = train_cfg.get("checkpoint_every_n_epochs", 1)
    monitor = "val_loss"
    callbacks = [
        ModelCheckpoint(
            monitor=monitor, mode="min", save_top_k=3,
            dirpath=save_dir,
            filename="epoch={epoch}-loss={val_loss:.4f}",
            auto_insert_metric_name=False,
            every_n_epochs=ckpt_every,
        ),
        ModelCheckpoint(
            dirpath=save_dir,
            filename="epoch={epoch}-periodic",
            auto_insert_metric_name=False,
            save_top_k=-1,
            every_n_epochs=ckpt_every,
        ),
        EarlyStopping(
            monitor=monitor,
            patience=train_cfg.get("early_stopping_patience", 20),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
        _FixCheckpointDir(save_dir),
    ]

    if use_fsdp:
        if _HAS_QWEN3:
            decoder_cls = {Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock}
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
            decoder_cls = {Qwen2_5_VLDecoderLayer}
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=decoder_cls
        )
        strategy = FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=False,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy="FULL_SHARD",
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            state_dict_type="full",
            limit_all_gathers=True,
        )
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=train_cfg["epochs"],
        accelerator="gpu",
        devices=devices,
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        strategy=strategy,
        precision="bf16-mixed" if not use_fsdp else None,
        callbacks=callbacks,
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        logger=loggers,
        enable_model_summary=True,
    )

    torch.cuda.empty_cache()
    fit_kwargs = {
        "model": model,
        "train_dataloaders": train_dl,
        "val_dataloaders": val_dl,
    }
    if resume_path:
        fit_kwargs["ckpt_path"] = resume_path
    trainer.fit(**fit_kwargs)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
