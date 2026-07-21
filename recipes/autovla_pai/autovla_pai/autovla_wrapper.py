"""AutoVLA model wrapper implementing the alpamayo MetricRunner interface.

``AutoVLAWrapper`` exposes the single method ``sample_trajectories_from_data``
required by ``alpamayo.metrics.metric_api.ReasoningSampler``.  All trajectory
metric maths is then handled by alpamayo's ``DistanceMetrics``, making results
directly comparable across models.

Architecture note
-----------------
``StructuredStage3Module.autovla.forward_proposal`` is **deterministic** at
inference time (VLM + proposal transformer + optional FM-refiner ``refined_mean``
— no stochastic noise sampled on the forward path).  Calling it N times returns
the same trajectory, so ``num_traj_samples > 1`` is meaningless unless you
explicitly inject noise.  The default is therefore ``num_traj_samples=1``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# ── SKIPlan path (only needed when not installed editable) ────────────────────
SKIPPLAN_ROOT = Path("/home/achahe/SKIPlan")
for p in [SKIPPLAN_ROOT, SKIPPLAN_ROOT / "navsim"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from autovla_pai.pai_adapter import build_autovla_batch


def _rotation_from_xyz(xyz: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation matrices from consecutive XY positions.

    Works for any leading shape:  ``[..., T, 3]  →  [..., T, 3, 3]``

    The yaw at step t is inferred from the direction ``pos[t] - pos[t-1]``
    (the first step is padded by repeating step 1's direction).
    Z is ignored.
    """
    *leading, T, _ = xyz.shape
    xy = xyz[..., :2]                                  # [..., T, 2]
    d = xy[..., 1:, :] - xy[..., :-1, :]              # [..., T-1, 2]
    yaw = torch.atan2(d[..., 1], d[..., 0])           # [..., T-1]
    yaw = torch.cat([yaw[..., :1], yaw], dim=-1)       # [..., T]

    cos_y = yaw.cos()
    sin_y = yaw.sin()
    z     = torch.zeros_like(cos_y)
    o     = torch.ones_like(cos_y)
    # Rows: [cos, -sin, 0], [sin, cos, 0], [0, 0, 1]
    rot = torch.stack(
        [cos_y, -sin_y, z,
         sin_y,  cos_y, z,
         z,      z,     o],
        dim=-1,
    ).view(*leading, T, 3, 3)
    return rot


class AutoVLAWrapper:
    """Wrap ``StructuredStage3Module`` with the alpamayo ``MetricRunner`` API.

    Usage
    -----
    ::

        from models.structured_stage3 import StructuredStage3Module
        model = StructuredStage3Module(config)
        model.load_state_dict(...)
        model.to(device).eval()

        wrapper = AutoVLAWrapper(model, config)
        pred_xyz, pred_rot = wrapper.sample_trajectories_from_data(pai_batch)
    """

    def __init__(self, model: Any, config: dict) -> None:
        """Args:
        model: an initialised ``StructuredStage3Module`` (on CPU or any device).
        config: the raw dict loaded from the AutoVLA ``config.yaml``.
        """
        self.model = model
        self.config = config
        self.processor = model.autovla.processor

    # ── nn.Module-like helpers so the MetricRunner wrapper stays thin ─────────

    def to(self, device: str | torch.device) -> "AutoVLAWrapper":
        self.model.to(device)
        return self

    def eval(self) -> "AutoVLAWrapper":
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    # ── Core inference ────────────────────────────────────────────────────────

    def sample_trajectories_from_data(
        self,
        data: dict,
        num_traj_samples: int = 1,
        num_traj_sets: int = 1,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a PAI data batch to AutoVLA format, run inference, return trajectory tensors.

        Because ``forward_proposal`` is deterministic, ``num_traj_samples > 1``
        simply tiles the same prediction.  Pass ``num_traj_samples=1`` (default)
        for a fair single-sample comparison against Alpamayo Stage-1.

        Args:
            data: PAI batch dict (from ``PAIDataset`` with ``time_step=0.5,
                num_future_steps=8``).
            num_traj_samples: trajectory candidates per set (K dim).
            num_traj_sets:    independent trajectory sets    (N dim).
            **kwargs:         ignored (accepted for API compatibility).

        Returns:
            pred_xyz: ``[B, N, K, Tf, 3]``  ego-local XY, Z=0
            pred_rot: ``[B, N, K, Tf, 3, 3]``  yaw-only rotation matrices
        """
        device = next(iter(self.model.parameters())).device

        # Build AutoVLA-format input from PAI batch
        batch, _pil_frames = build_autovla_batch(
            data, self.config, self.processor, device
        )

        with torch.no_grad():
            out = self.model.autovla.forward_proposal(batch, compute_ce=False)

        stage3_xy: torch.Tensor = out["stage3_xy"]     # [B, Tf, 2]
        B, Tf = stage3_xy.shape[:2]

        # Append Z=0  →  [B, Tf, 3]
        pred_xy3 = torch.cat(
            [stage3_xy, torch.zeros(B, Tf, 1, device=stage3_xy.device, dtype=stage3_xy.dtype)],
            dim=-1,
        )

        # Expand to [B, N, K, Tf, 3]
        N, K = num_traj_sets, num_traj_samples
        pred_xyz = pred_xy3[:, None, None].expand(B, N, K, Tf, 3).contiguous()

        # Rotation from direction of travel
        pred_rot = _rotation_from_xyz(pred_xyz)         # [B, N, K, Tf, 3, 3]

        return pred_xyz, pred_rot
