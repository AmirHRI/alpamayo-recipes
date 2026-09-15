from types import SimpleNamespace

import pytest
import torch

from alpamayo1_5_distill.models.precision_probe import PrecisionProbeCallback


def test_precision_probe_checks_real_updates_and_moment_dtypes():
    model = torch.nn.Module()
    model.expert = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    callback = PrecisionProbeCallback()
    callback.probe_name = "expert.weight"
    args = SimpleNamespace(fp16=False, bf16=False)
    state = SimpleNamespace(global_step=1)
    callback.on_train_begin(args, state, None, model, optimizer)
    callback.on_step_begin(args, state, None, model)
    model.expert(torch.ones(1, 2)).sum().backward()
    callback.on_pre_optimizer_step(args, state, None, model)
    optimizer.step()
    callback.on_step_end(args, state, None, model, optimizer)
    callback.on_train_end(args, state, None)
    assert callback.successful_steps == 1


def test_precision_probe_rejects_missing_expert_gradient():
    model = torch.nn.Module()
    model.expert = torch.nn.Linear(2, 2, bias=False)
    callback = PrecisionProbeCallback()
    callback.probe_name = "expert.weight"
    with pytest.raises(RuntimeError, match="gradient is missing"):
        callback.on_pre_optimizer_step(None, None, None, model)


def test_precision_probe_requires_scaler_for_fp16():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="GradScaler"):
        PrecisionProbeCallback().on_train_begin(
            SimpleNamespace(fp16=True, bf16=False), None, None, model,
            torch.optim.AdamW(model.parameters()),
        )