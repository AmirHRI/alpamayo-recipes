"""PAI SFT Dataset — items identical in format to SKIPlan's SFTDataset.

Each ``__getitem__`` returns the exact dict that SKIPlan's ``DataCollator``
expects, so we can plug directly into ``tools/run_sft.py``'s training loop
without modifying any SKIPlan code.

Item shapes
-----------
text               str             apply_chat_template output (contains <plan>)
video_inputs       [[PIL×4]]       front-wide 4-frame clip at 2 Hz
image_inputs       None
has_cot            bool            always False
video_kwargs       {"fps":[2.0]}
gt_trajectory      Tensor[8, 3]    ego_future_xyz in ego-local frame
gt_action          Tensor[1]       dummy (zeros; DataCollator stacks it)
gt_action_alpamayo Tensor[1,8,2]   from action_space.traj_to_action()
ego_history_xyz    Tensor[1,4,3]   from PAIDataset
ego_history_rot    Tensor[1,4,3,3] from PAIDataset
swin_ego_state     Tensor[1,12]    velocity + command one-hot (matches nuPlan fmt)
vehicle_velocity   Tensor[1]       speed in m/s
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import Dataset

# ── SKIPlan path ──────────────────────────────────────────────────────────────
SKIPPLAN_ROOT = Path("/home/achahe/SKIPlan")
if SKIPPLAN_ROOT.exists() and str(SKIPPLAN_ROOT) not in sys.path:
    sys.path.insert(0, str(SKIPPLAN_ROOT))

from alpamayo.data.pai import PAIDataset

# PAI canonical camera order: 0=front_tele_30fov, 1=front_wide_120fov, …
FRONT_WIDE_IDX = 1
N_VIDEO_FRAMES = 4
VIDEO_FPS = 2.0


def _build_action_space(n_waypoints: int = 8, dt: float = 0.5):
    """Build the same DeltaActionSpace that StructuredStage3Module uses for nuPlan Stage 1.

    ``hybrid`` normalisation → ``robust_zscore`` with the nuPlan stats.
    These stats are only used for the FM-style loss (weight=0 in Stage 1),
    so reusing nuPlan stats is safe for PAI fine-tuning.
    """
    from models.alpamayo.action_space.delta_action_space import DeltaActionSpace
    return DeltaActionSpace(
        n_waypoints=n_waypoints,
        predict_z=False,
        normalization="robust_zscore",
        delta_median=(1.9273672103881836, 0.00505890604108572),
        delta_mad=(1.4997866960338442, 0.7348284578457333),
        soft_clip_C=[24.0, 6.0],
    )


def _resize_to_pil(cam_tensor: torch.Tensor, max_pixels: int, n_out: int, patch: int = 28) -> List:
    """Subsample + resize a [T, C, H, W] uint8 tensor → n_out PIL images."""
    T, _, H, W = cam_tensor.shape
    scale = min(1.0, (max_pixels / (H * W)) ** 0.5)
    tgt_h = max(patch, int(H * scale / patch) * patch)
    tgt_w = max(patch, int(W * scale / patch) * patch)

    step = max(1, (T - 1) // max(n_out - 1, 1))
    idxs = list(range(0, T, step))[:n_out]
    idxs += [T - 1] * (n_out - len(idxs))
    idxs = idxs[:n_out]

    sel = cam_tensor[idxs].float()
    res = F.interpolate(sel, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    return [PILImage.fromarray(res[j].byte().permute(1, 2, 0).cpu().numpy()) for j in range(n_out)]


def _ego_velocity_and_cmd(hist_xyz: torch.Tensor, dt: float):
    """Estimate speed (m/s) and coarse instruction from last two history steps.

    hist_xyz: [T, 3] in ego-local frame.
    """
    d = hist_xyz[-1, :2] - hist_xyz[-2, :2]         # [2]  displacement in ego frame
    vel = float(d.norm().item() / dt)
    heading = float(np.arctan2(float(d[1].item()), float(d[0].item())))
    if abs(heading) < 0.15:
        instruction = "keep forward"
    elif heading > 0:
        instruction = "turn left"
    else:
        instruction = "turn right"
    return vel, instruction


def _build_swin_ego_state(vel: float, instruction: str) -> torch.Tensor:
    """Build a [1, 12] ego-state vector matching the nuPlan Stage 1 format."""
    s = torch.zeros(1, 12)
    s[0, 0] = vel   # vx (longitudinal approx)
    s[0, 5] = vel   # speed
    s[0, 9]  = 1.0 if "left" in instruction else 0.0
    s[0, 10] = 1.0 if "forward" in instruction else 0.0
    s[0, 11] = 1.0 if "right" in instruction else 0.0
    return s


def _build_text(processor, pil_frames: List, fut_xyz: torch.Tensor,
                vel: float, instruction: str,
                time_horizon: float = 4.0,
                max_pixels: int = 109760) -> str:
    """Build the chat-template text string matching nuPlan Stage 1 prompt format.

    Uses ``simplified_prompt=True`` format:
      system: "You are an autonomous driving system. Predict the driving trajectory …"
      user:   "Velocity: … m/s. Instruction: …. Predict the X-second trajectory."
      asst:   "<plan>\\n<|traj_num_start|>\\n(+dx, +dy);\\n…\\n<|traj_num_end|>"
    """
    th = int(time_horizon) if time_horizon == int(time_horizon) else time_horizon
    sys_text = (
        f"You are an autonomous driving system. "
        f"Predict the driving trajectory for the next {th} seconds."
    )
    user_text = (
        f"Velocity: {vel:.3f} m/s. "
        f"Instruction: {instruction}. "
        f"Predict the {th}-second trajectory."
    )

    # Delta trajectory string (same format used in nuPlan Stage 1, trajectory_tokenizer='raw')
    fut_xy = fut_xyz[:, :2].cpu().numpy()               # [8, 2]
    coords = np.diff(fut_xy, axis=0, prepend=np.zeros((1, 2)))
    pts = [f"({c[0]:+.2f}, {c[1]:+.2f});" for c in coords]
    asst_text = "<plan>\n<|traj_num_start|>\n" + "\n".join(pts) + "\n<|traj_num_end|>"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Front view. "},
                {
                    "type": "video",
                    "video": pil_frames,
                    "fps": VIDEO_FPS,
                    "min_pixels": max_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": asst_text}]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_vision_id=True
    )


class PAISFTDataset(Dataset):
    """PAI-backed SFT dataset, drop-in replacement for SKIPlan's ``SFTDataset``.

    Parameters
    ----------
    pai_dir:            PAI local_dir root (same as used in evaluate.py)
    chunk_ids:          chunk range string, e.g. "0-185"
    processor:          Qwen3-VL AutoProcessor (for apply_chat_template)
    action_space:       UnicycleAccelCurvatureActionSpace instance (or None → auto-build)
    time_step:          0.5 s (must match AutoVLA's training resolution)
    num_history_steps:  4
    num_future_steps:   8
    time_horizon:       4.0 s
    img_max_pixels:     109760 (matches nuPlan Stage 1 config)
    """

    def __init__(
        self,
        pai_dir: str,
        chunk_ids: str,
        processor,
        action_space=None,
        time_step: float = 0.5,
        num_history_steps: int = 4,
        num_future_steps: int = 8,
        time_horizon: float = 4.0,
        img_max_pixels: int = 109760,
    ):
        self.processor = processor
        self.time_step = time_step
        self.time_horizon = time_horizon
        self.img_max_pixels = img_max_pixels

        # Register trajectory special tokens exactly as StructuredStage3Module does.
        # This must happen before any apply_chat_template call so <plan>,
        # <|traj_num_start|>, <|traj_num_end|> are single tokens, not split by BPE.
        _TRAJ_TOKENS = ["<|traj_num_start|>", "<|traj_num_end|>", "<plan>"]
        existing = set(processor.tokenizer.additional_special_tokens)
        new_toks = [t for t in _TRAJ_TOKENS if t not in existing]
        if new_toks:
            processor.tokenizer.add_special_tokens({"additional_special_tokens": new_toks})

        self.pai = PAIDataset(
            local_dir=pai_dir,
            chunk_ids=chunk_ids,
            use_default_keyframe=True,
            num_history_steps=num_history_steps,
            num_future_steps=num_future_steps,
            time_step=time_step,
            vla_preprocess_args=None,
        )

        if action_space is None:
            action_space = _build_action_space(n_waypoints=num_future_steps, dt=time_step)
        self.action_space = action_space

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pai)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.pai[idx]
        if sample is None:
            return None

        # ── Front-wide camera frames ──────────────────────────────────
        cam_idx = sample["camera_indices"]
        if isinstance(cam_idx, torch.Tensor):
            hits = (cam_idx == FRONT_WIDE_IDX).nonzero(as_tuple=True)[0]
        else:
            hits = [i for i, c in enumerate(cam_idx) if int(c) == FRONT_WIDE_IDX]
        if not len(hits):
            return None   # skip samples without front-wide camera
        front_pos = int(hits[0])

        cam_tensor = sample["image_frames"][front_pos]       # [T, C, H, W]
        pil_frames = _resize_to_pil(cam_tensor, self.img_max_pixels, N_VIDEO_FRAMES)

        # ── Ego tensors (already in ego-local frame from PAIDataset) ──
        hist_xyz = sample["ego_history_xyz"]   # [1, Th, 3]
        hist_rot = sample["ego_history_rot"]   # [1, Th, 3, 3]
        fut_xyz  = sample["ego_future_xyz"]    # [1, Tf, 3]
        fut_rot  = sample["ego_future_rot"]    # [1, Tf, 3, 3]

        # ── Velocity + instruction ────────────────────────────────────
        vel, instruction = _ego_velocity_and_cmd(hist_xyz[0], self.time_step)

        # ── GT action via traj_to_action ──────────────────────────────
        # Add batch dim: [1, group=1, T, …]
        with torch.no_grad():
            result = self.action_space.traj_to_action(
                hist_xyz.unsqueeze(0),   # [1, 1, Th, 3]
                hist_rot.unsqueeze(0),   # [1, 1, Th, 3, 3]
                fut_xyz.unsqueeze(0),    # [1, 1, Tf, 3]
                fut_rot.unsqueeze(0),    # [1, 1, Tf, 3, 3]
            )
        gt_action_alpamayo = result[0] if isinstance(result, tuple) else result
        gt_action_alpamayo = gt_action_alpamayo.squeeze(1)  # [1, Tf, 2]

        # ── Text prompt ───────────────────────────────────────────────
        text = _build_text(
            self.processor,
            pil_frames,
            fut_xyz[0],       # [Tf, 3]
            vel,
            instruction,
            time_horizon=self.time_horizon,
            max_pixels=self.img_max_pixels,
        )

        return {
            # ── VLM text input ─────────────────────────────────────
            "text":               text,
            "video_inputs":       [pil_frames],
            "image_inputs":       None,
            "has_cot":            False,
            "video_kwargs":       {"fps": [VIDEO_FPS]},
            # ── Trajectory GT ──────────────────────────────────────
            # gt_trajectory: DataCollator uses torch.stack → [B, Tf, 3]
            "gt_trajectory":      fut_xyz[0].clone(),           # [Tf, 3]
            "gt_action":          torch.zeros(1),               # dummy
            "gt_action_alpamayo": gt_action_alpamayo.clone(),   # [1, Tf, 2]
            # ── History (DataCollator uses torch.cat → [B, Th, …]) ─
            "ego_history_xyz":    hist_xyz.clone(),             # [1, Th, 3]
            "ego_history_rot":    hist_rot.clone(),             # [1, Th, 3, 3]
            # ── Ego conditioning (proposal head use_ego_token=True) ─
            "swin_ego_state":     _build_swin_ego_state(vel, instruction),  # [1, 12]
            "vehicle_velocity":   torch.tensor([vel]),          # [1]
        }
