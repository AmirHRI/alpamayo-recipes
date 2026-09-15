"""Diagnostics for short consistency optimizer-precision experiments."""

import torch
from transformers import TrainerCallback


class PrecisionProbeCallback(TrainerCallback):
    def __init__(self):
        self.successful_steps = 0
        self.skipped_steps = 0
        self.before = None
        self.probe_name = "expert.layers.0.self_attn.q_proj.weight"

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
            raise RuntimeError("precision probe requires all trainable weights in fp32")
        scaler = getattr(optimizer, "scaler", None)
        if args.fp16 and scaler is None:
            raise RuntimeError("fp16 precision probe requires GradScaler")
        if args.bf16:
            raise RuntimeError("precision probe expects bf16=False in Trainer")
        print(f"[precision-probe] trainable=fp32 fp16={args.fp16} "
              f"scaler={type(scaler).__name__ if scaler else 'none'}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        base = getattr(model, "module", model)
        self.before = dict(base.named_parameters())[self.probe_name].detach().clone()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        base = getattr(model, "module", model)
        parameter = dict(base.named_parameters())[self.probe_name]
        if parameter.grad is None:
            raise RuntimeError("precision probe expert gradient is missing before optimizer step")

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        skipped = bool(getattr(optimizer, "step_was_skipped", False))
        self.skipped_steps += int(skipped)
        self.successful_steps += int(not skipped)
        base = getattr(model, "module", model)
        parameter = dict(base.named_parameters())[self.probe_name].detach()
        delta = parameter - self.before
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError("non-finite online probe weights")
        moments = {
            str(value.dtype)
            for item in optimizer.state.values()
            for key, value in item.items()
            if key in ("exp_avg", "exp_avg_sq")
        }
        if not skipped and moments != {"torch.float32"}:
            raise RuntimeError(f"optimizer moments must be fp32, got {moments}")
        scaler = getattr(optimizer, "scaler", None)
        scale = scaler.get_scale() if scaler is not None else 1.0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(f"[precision-probe] step={state.global_step} skipped={skipped} "
              f"scale={scale:g} changed={float((delta != 0).float().mean()):.6f} "
              f"update_rms={float(delta.square().mean().sqrt()):.6e} "
              f"moments={sorted(moments)} peak_allocated_GiB={peak:.2f}", flush=True)
        self.before = None

    def on_train_end(self, args, state, control, **kwargs):
        if self.successful_steps == 0:
            raise RuntimeError("precision probe completed no successful optimizer steps")
        print(f"[precision-probe] COMPLETE successful={self.successful_steps} "
              f"skipped={self.skipped_steps}", flush=True)