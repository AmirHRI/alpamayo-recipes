"""Convert a PAIDataset batch into an AutoVLA-compatible batch.

Key responsibilities
--------------------
* Extract the front-wide-120fov camera frames and build Qwen3-VL video inputs.
* Sub-sample PAI history (0.5 s steps) → AutoVLA format [B, 1, 4, 3/3×3].
* Estimate ego velocity and coarse driving command from history.
* Build ``swin_ego_state`` [B, 12] velocity/command state vector.
* Leave ``map_polylines`` absent — ``forward_proposal`` safely handles ``None``.

The script does NOT require nuplan-devkit; any path-related config keys are
simply unused during inference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ── Camera index for front-wide-120fov in PAI canonical ordering ──────────────
# PAI canonical camera order: 0=front_tele_30fov, 1=front_wide_120fov,
# 2=cross_left_120fov, 3=cross_right_120fov.
FRONT_WIDE_CANONICAL_IDX = 1
AUTOVLA_VIDEO_FPS = 2.0    # 2 Hz → 4 frames over 2 s


def _find_front_camera_pos(camera_indices_batch: Any, batch_size: int) -> int:
    """Return the position of the front-wide camera in the N_cam dimension.

    ``camera_indices_batch`` may be:
    * A stacked tensor  [B, N_cam]
    * A list of tensors [N_cam] (one per sample, all identical)
    * A list of lists
    """
    if isinstance(camera_indices_batch, torch.Tensor):
        indices_0 = camera_indices_batch[0]  # [N_cam]
    elif isinstance(camera_indices_batch, (list, tuple)):
        elem = camera_indices_batch[0]
        if isinstance(elem, torch.Tensor):
            indices_0 = elem
        else:
            indices_0 = torch.tensor(elem)
    else:
        raise ValueError(f"Unexpected camera_indices type: {type(camera_indices_batch)}")

    hits = (indices_0 == FRONT_WIDE_CANONICAL_IDX).nonzero(as_tuple=True)[0]
    if len(hits) == 0:
        raise ValueError(
            f"Front-wide camera (canonical idx {FRONT_WIDE_CANONICAL_IDX}) "
            f"not found. Available indices: {indices_0.tolist()}"
        )
    return int(hits[0].item())


def _subsample_pil_frames(
    cam_tensor: torch.Tensor,
    n_frames_out: int,
    target_h: int,
    target_w: int,
) -> List[Any]:
    """Resize and subsample a [T, C, H, W] uint8 tensor to n_frames_out PIL images."""
    from PIL import Image as PILImage

    T = cam_tensor.shape[0]
    step = max(1, (T - 1) // max(n_frames_out - 1, 1))
    # Evenly spaced indices, always include last frame
    indices = list(range(0, T, step))[:n_frames_out]
    if len(indices) < n_frames_out:
        indices += [T - 1] * (n_frames_out - len(indices))
    indices = indices[:n_frames_out]

    selected = cam_tensor[indices].float()  # [n_frames_out, C, H, W]
    resized = F.interpolate(
        selected, size=(target_h, target_w), mode="bilinear", align_corners=False
    )  # [n_frames_out, C, H, W]
    return [
        PILImage.fromarray(resized[j].byte().permute(1, 2, 0).cpu().numpy())
        for j in range(n_frames_out)
    ]


def _compute_resize_target(raw_h: int, raw_w: int, max_pixels: int, patch: int = 28):
    """Return (target_h, target_w) with max_pixels budget, aligned to patch size."""
    scale = min(1.0, (max_pixels / (raw_h * raw_w)) ** 0.5)
    th = max(patch, int(raw_h * scale / patch) * patch)
    tw = max(patch, int(raw_w * scale / patch) * patch)
    return th, tw


def _infer_driving_command(hist_xy: torch.Tensor) -> List[str]:
    """Infer coarse command from last two history frames.

    hist_xy: [B, Th, 2]   ego-local XY at 0.5 s steps
    """
    if hist_xy.shape[1] < 2:
        return ["keep forward"] * hist_xy.shape[0]
    d = (hist_xy[:, -1] - hist_xy[:, -2]).cpu().numpy()  # [B, 2]
    headings = np.arctan2(d[:, 1], d[:, 0])
    cmds = []
    for h in headings:
        if abs(h) < 0.15:
            cmds.append("keep forward")
        elif h > 0:
            cmds.append("turn left")
        else:
            cmds.append("turn right")
    return cmds


def build_autovla_batch(
    pai_batch: Dict[str, Any],
    autovla_config: Dict[str, Any],
    processor: Any,
    device: torch.device,
    n_video_frames: int = 4,
) -> Tuple[Dict[str, Any], List[List[Any]]]:
    """Convert a PAIDataset batch (loaded at time_step=0.5 s) to an AutoVLA batch.

    Args:
        pai_batch:       Output of PAIDataset collate, batch of B samples.
                         Expected keys: image_frames [B, N_cam, T, C, H, W],
                         camera_indices, ego_history_xyz [B, 1, 4, 3],
                         ego_history_rot [B, 1, 4, 3, 3].
        autovla_config:  Loaded config.yaml dict.
        processor:       model.autovla.processor (Qwen3-VL processor).
        device:          Target device.
        n_video_frames:  Number of video frames to pass to the VLM (default 4).

    Returns:
        batch:           Dict ready for model.autovla.forward_proposal(batch).
        pil_frames_all:  [B] list of PIL-image lists — for visualization.
    """
    from qwen_vl_utils import process_vision_info
    from transformers.video_utils import VideoMetadata

    B = pai_batch["ego_history_xyz"].shape[0]
    m_cfg = autovla_config["model"]
    video_cfg = m_cfg.get("video", {})
    traj_cfg = m_cfg["trajectory"]
    time_horizon = float(traj_cfg["time_horizon"])
    time_horizon_str = str(int(time_horizon)) if time_horizon == int(time_horizon) else str(time_horizon)
    simplified = bool(m_cfg.get("simplified_prompt", True))

    max_pixels: int = int(video_cfg.get("max_pixels", 28 * 28 * 128))
    min_pixels: int = int(video_cfg.get("min_pixels", 28 * 28 * 4))

    # ── Locate front camera ───────────────────────────────────────────────────
    front_pos = _find_front_camera_pos(pai_batch["camera_indices"], B)

    # ── Compute target resize dimensions from first frame ────────────────────
    sample0_cam = pai_batch["image_frames"][0, front_pos]  # [T, C, H, W]
    _, _, H_raw, W_raw = sample0_cam.shape
    tgt_h, tgt_w = _compute_resize_target(H_raw, W_raw, max_pixels)

    # ── History tensors (PAI loaded at 0.5 s, shape [B, 1, 4, 3/3×3]) ────────
    hist_xyz = pai_batch["ego_history_xyz"].float()  # [B, 1, 4, 3]
    hist_rot = pai_batch["ego_history_rot"].float()  # [B, 1, 4, 3, 3]

    # Velocity from last two history frames (0.5 s apart)
    hist_xy = hist_xyz[:, 0, :, :2]  # [B, 4, 2]
    if hist_xy.shape[1] >= 2:
        d = (hist_xy[:, -1] - hist_xy[:, -2]).cpu().numpy()  # [B, 2]
        velocities = np.linalg.norm(d, axis=-1) / 0.5  # m/s
    else:
        velocities = np.zeros(B)

    driving_cmds = _infer_driving_command(hist_xy)

    # ── Build per-sample content lists ───────────────────────────────────────
    all_pil_frames: List[List[Any]] = []
    user_contents: List[list] = []
    texts: List[str] = []

    for i in range(B):
        cam_tensor = pai_batch["image_frames"][i, front_pos]  # [T, C, H, W]
        pil_frames = _subsample_pil_frames(cam_tensor, n_video_frames, tgt_h, tgt_w)
        all_pil_frames.append(pil_frames)

        vel = float(velocities[i])
        cmd = driving_cmds[i]

        if simplified:
            user_text = (
                f"Velocity: {vel:.3f} m/s. "
                f"Instruction: {cmd}. "
                f"Predict the driving trajectory for the next {time_horizon_str} seconds."
            )
            sys_text = (
                f"You are an autonomous driving system. "
                f"Predict the driving trajectory for the next {time_horizon_str} seconds."
            )
            content = [
                {"type": "text", "text": "Front view. "},
                {
                    "type": "video",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "fps": AUTOVLA_VIDEO_FPS,
                    "video": pil_frames,
                },
                {"type": "text", "text": user_text},
            ]
        else:
            sys_text = (
                f"You are an autonomous driving system. "
                f"Predict the driving trajectory for the next {time_horizon_str} seconds."
            )
            user_text = (
                f"The autonomous vehicle is equipped with 1 camera (front). "
                f"The video presents the front view, comprising {n_video_frames} frames "
                f"sampled at {AUTOVLA_VIDEO_FPS} Hz.\n"
                f"Velocity: {vel:.3f} m/s. "
                f"Instruction: {cmd}. "
                f"Predict the driving trajectory for the next {time_horizon_str} seconds."
            )
            content = [
                {
                    "type": "video",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "fps": AUTOVLA_VIDEO_FPS,
                    "video": pil_frames,
                },
                {"type": "text", "text": user_text},
            ]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<plan>"}]},
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=True,
        )
        texts.append(text)
        user_contents.append(content)

    # ── Batched vision processing (one call, not B sequential calls) ──────────
    _, video_inputs, video_kwargs = process_vision_info(
        [{"role": "user", "content": uc} for uc in user_contents],
        return_video_kwargs=True,
    )
    all_fps = (video_kwargs.get("fps") or [AUTOVLA_VIDEO_FPS] * B) if video_kwargs else [AUTOVLA_VIDEO_FPS] * B

    proc_kwargs: Dict[str, Any] = {}
    if video_inputs:
        metadata = [
            VideoMetadata(
                total_num_frames=len(vi) if isinstance(vi, (list, tuple)) else vi.shape[0],
                fps=float(all_fps[idx]) if idx < len(all_fps) else AUTOVLA_VIDEO_FPS,
                frames_indices=list(range(len(vi) if isinstance(vi, (list, tuple)) else vi.shape[0])),
            )
            for idx, vi in enumerate(video_inputs)
            if vi is not None
        ]
        if metadata:
            proc_kwargs["video_metadata"] = metadata
            proc_kwargs["do_sample_frames"] = False

    batch = processor(
        text=texts,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
        **proc_kwargs,
    )

    # ── Trajectory history ────────────────────────────────────────────────────
    batch["ego_history_xyz"] = hist_xyz   # [B, 1, 4, 3]
    batch["ego_history_rot"] = hist_rot   # [B, 1, 4, 3, 3]

    # ── Ego-state vector [B, 12] — velocity + command one-hot ─────────────────
    # StructuredStage3AutoVLA._history_embed uses swin_ego_state[..., :6]
    # (dx, dy, dz, roll, pitch, speed). We fill speed (index 5) only.
    # The command indices (9-11) are used by SwinTrajectory variant, which
    # is not present in the nuplan / plan_token checkpoints.
    ego_state = torch.zeros(B, 12)
    for i in range(B):
        vel = float(velocities[i])
        cmd = driving_cmds[i]
        ego_state[i, 5] = vel          # speed
        ego_state[i, 0] = vel          # vx (longitudinal approx)
        ego_state[i, 9]  = 1.0 if "left" in cmd else 0.0
        ego_state[i, 10] = 1.0 if "forward" in cmd else 0.0
        ego_state[i, 11] = 1.0 if "right" in cmd else 0.0
    batch["swin_ego_state"] = ego_state

    # Map polylines intentionally absent — forward_proposal uses .get() so
    # it receives None and the proposal head runs without map conditioning.

    # ── Move to device ────────────────────────────────────────────────────────
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    return batch, all_pil_frames
