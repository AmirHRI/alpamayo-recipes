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

r"""``PAIDataset`` restricted to a subset of cameras, with the prompt rebuilt to match.

The loader hands back 4 cameras x 4 frames = 16 images.  Training on fewer means the PROMPT has
to change too, not just the pixels: it carries one ``frame N`` tag and one
``<|vision_start|>...<|vision_end|>`` pair per image, plus one display-name label per camera
("Front camera:", "Front telephoto camera:").  Verified by decoding the real prompt -- drop a
camera and its label, its four frame tags and its four vision blocks all disappear.

⚠️ So the cameras must be sliced BEFORE preprocessing.  ``PAIDataset`` runs its preprocess
inside ``__getitem__``, so this composes rather than subclasses: it builds the base dataset with
NO preprocess and applies the processor itself afterwards.  Slicing after tokenisation would
leave the token count disagreeing with the pixel values, which fails deep inside the vision
merge with an opaque shape error.

⚠️ Camera indices are the loader's ``camera_features`` order:

===  ======================  =========================
idx  feature                 prompt label
===  ======================  =========================
0    cross_left_120fov       "Front left camera"
1    front_wide_120fov       "Front camera"
2    cross_right_120fov      "Front right camera"
3    front_tele_30fov        "Front telephoto camera"
===  ======================  =========================

NOT the global ``CAMERA_NAMES_TO_INDICES`` table, where front_tele is 6.  So front-wide plus
telephoto -- the pair that measured best on the teacher, min_ade 0.6981 at 26% less latency --
is ``[1, 3]``.

⚠️ In KD both towers read ONE ``input_ids`` tensor from the same batch, so subsetting here
applies to teacher and student identically and their caches stay position-for-position
comparable.  There is nothing to keep in sync.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from hydra.utils import instantiate

from alpamayo1_5_distill.data import teacher_trajectory_io
from alpamayo1_5_distill.data.distill_dataset import sample_key

#: Keys whose leading dimension is the camera axis, as ``load_physical_aiavdataset`` returns
#: them: ``image_frames`` is [N_cam, num_frames, 3, H, W] and the rest are [N_cam, num_frames].
#: ⚠️ ``reshape_tensors_for_rl`` would flatten that axis away; it defaults False and must stay
#: False for this class, which is why it is not exposed.
_CAMERA_AXIS_KEYS = ("image_frames", "camera_indices", "absolute_timestamps",
                     "relative_timestamps")

#: Positional order used by ``load_physical_aiavdataset`` when ``camera_features=None``.
#: ``CameraSubsetPAIDataset.cameras`` intentionally keeps this public, backward-compatible
#: numbering while passing the corresponding feature names down to the physical reader.
_DEFAULT_CAMERA_FEATURES = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)

_TURN_WITH_DISTANCE = re.compile(r"^(Turn (?:left|right)) in [0-9]+(?:\.[0-9]+)?m$")


def strip_turn_distance(nav_text: str) -> str:
    """Collapse a distance-qualified turn to the teacher's direction-only instruction.

    Non-turn instructions are preserved. A turn-like string outside the supported manifest
    grammar raises instead of silently leaking a distance (or inventing a rewrite) into a long
    training run.
    """
    text = str(nav_text).strip()
    if text in {"Turn left", "Turn right"}:
        return text
    match = _TURN_WITH_DISTANCE.fullmatch(text)
    if match is not None:
        return match.group(1)
    if text.startswith("Turn "):
        raise ValueError(
            f"unsupported nav turn instruction {nav_text!r}; expected 'Turn left/right in Nm'"
        )
    return text


class CameraSubsetPAIDataset(torch.utils.data.Dataset):
    """``PAIDataset`` yielding only ``cameras``, with the prompt built from that subset.

    Args:
        cameras: indices into the loader's camera order (see the module docstring).
        vla_preprocess_args: the processor config the base dataset would have used. Taken here
            instead so it runs AFTER slicing.
        model_config: forwarded to the processor; supplies the trajectory vocabulary and the
            pixel budget.
        annotations_path: when given, the base becomes ``PAIDatasetWithNav`` instead of
            ``PAIDataset``, so one sample per ANNOTATION (its own ``t0_relative`` and
            ``nav_text``) rather than one per clip. ⚠️ The route only reaches the prompt if
            ``"route"`` is in the processor's ``components_order`` -- ``r1_5.py`` emits it via
            `case "route":`, so otherwise nav_text is silently discarded and the run trains on
            nav data with the instruction invisible. Both towers read ONE input_ids tensor, so
            adding it here conditions the teacher and the student identically.
        **pai_kwargs: everything else ``PAIDataset`` takes (``local_dir``, ``chunk_ids``,
            ``clip_uuid_filter``, ``use_default_keyframe``, ...).
    """

    def __init__(
        self,
        cameras,
        vla_preprocess_args=None,
        model_config=None,
        annotations_path: str | None = None,
        strip_nav_turn_distance: bool = False,
        teacher_trajectory_cache_root: str | None = None,
        teacher_trajectory_cached_only: bool = False,
        frame_cache_root: str | None = None,
        frame_cache_require_all: bool = True,
        **pai_kwargs: Any,
    ) -> None:
        from alpamayo.data.pai import PAIDataset
        from alpamayo.data.pai_nav import PAIDatasetWithNav

        if vla_preprocess_args is None:
            raise ValueError(
                "CameraSubsetPAIDataset needs vla_preprocess_args: it exists to run the "
                "processor AFTER slicing, so there is nothing to do without one.")
        self.cameras = [int(c) for c in cameras]
        bad = [c for c in self.cameras if not 0 <= c < len(_DEFAULT_CAMERA_FEATURES)]
        if bad:
            raise IndexError(
                f"cameras {bad} out of range for the loader's four-camera default"
            )
        # ⚠️ model_config=None / no preprocess on the base: we want the RAW sample. With a nav
        # base that also means PAIDatasetWithNav skips its own preprocess, leaving nav_text in
        # the raw dict for our processor to pick up after slicing.
        # ⚠️ frame_cache_root swaps ONLY where the pixels come from. The cache was built through
        # this very class's base dataset, so the raw dict it returns is key-for-key identical
        # and everything below -- nav stripping, the camera guard, the processor -- is unchanged.
        # What goes away is 1.64 TiB of NFS per epoch (measured, job 20710) in favour of ~2.7 MB
        # of tmpfs per step. The cache holds exactly the requested cameras, so the slicing branch
        # in __getitem__ is a no-op and `camera_features` never needs setting.
        self.frame_cache_root = frame_cache_root
        if frame_cache_root is not None:
            from alpamayo1_5_distill.data.frame_cache import FrameCacheNavDataset

            if annotations_path is None:
                raise ValueError(
                    "frame_cache_root needs annotations_path: the cache is keyed by the "
                    "manifest's (clip_id, t0_relative) anchors."
                )
            self.base = FrameCacheNavDataset(
                annotations_path=annotations_path,
                cache_root=frame_cache_root,
                cameras=self.cameras,
                require_all=frame_cache_require_all,
            )
        elif annotations_path is not None:
            self.base = PAIDatasetWithNav(annotations_path=annotations_path, **pai_kwargs,
                                          model_config=None, vla_preprocess_args=None)
        else:
            self.base = PAIDataset(**pai_kwargs, model_config=None, vla_preprocess_args=None)

        # Applies to either nav base: the frame cache stores pixels only, and reads nav_text
        # from the same annotation JSON, so the stripping still has to happen here.
        if annotations_path is not None and strip_nav_turn_distance:
            changed = 0
            for entry in self.base._samples:
                original = entry["nav_text"]
                normalized = strip_turn_distance(original)
                entry["nav_text"] = normalized
                changed += normalized != original
            remaining = [
                entry["nav_text"] for entry in self.base._samples
                if _TURN_WITH_DISTANCE.fullmatch(entry["nav_text"])
            ]
            if remaining:
                raise RuntimeError(
                    f"distance stripping left {len(remaining)} qualified turns; "
                    f"first={remaining[0]!r}"
                )
            print(
                f"[nav] stripped distance from {changed}/{len(self.base._samples)} "
                "turn instructions",
                flush=True,
            )

        # Read only the requested cameras. Previously the base decoded all four cameras and
        # __getitem__ discarded two of them, doubling camera ZIP traffic for a 2-camera run.
        # The upstream loader sorts by the cameras' global IDs, producing exactly the same
        # [front-wide, front-tele] tensors/IDs as slicing [1, 3] from its default output.
        # The frame cache was built from that same subset, so it has no camera axis to narrow.
        if frame_cache_root is None:
            self.base.camera_features = [_DEFAULT_CAMERA_FEATURES[c] for c in self.cameras]
        self.pre = instantiate(vla_preprocess_args, model_config=model_config)
        self.teacher_trajectory_cache_root = teacher_trajectory_cache_root
        if teacher_trajectory_cached_only and teacher_trajectory_cache_root is None:
            raise ValueError(
                "teacher_trajectory_cached_only=true requires "
                "teacher_trajectory_cache_root"
            )
        self._indices = list(range(len(self.base)))
        if teacher_trajectory_cached_only:
            self._indices = [
                i for i in self._indices
                if teacher_trajectory_io.has_entry(
                    teacher_trajectory_cache_root, self._base_sample_key(i)
                )
            ]
            if not self._indices:
                raise RuntimeError(
                    f"no cached teacher trajectories found under "
                    f"{teacher_trajectory_cache_root}"
                )
        kind = "nav samples" if annotations_path is not None else "clips"
        cached_note = (
            f", cached-only={len(self._indices)}/{len(self.base)}"
            if teacher_trajectory_cached_only else ""
        )
        print(f"[camsubset] {len(self.base)} {kind}{cached_note}, cameras {self.cameras} "
              f"({len(self.cameras)} x 4 frames = {len(self.cameras) * 4} images)", flush=True)
        if annotations_path is not None:
            order = (vla_preprocess_args or {}).get("components_order") or []
            if "route" not in list(order):
                raise ValueError(
                    "annotations_path is set but 'route' is NOT in components_order="
                    f"{list(order)}. r1_5.py emits the route only on `case \"route\":`, so the "
                    "nav instruction would be silently dropped and this run would train on nav "
                    "data with the instruction invisible. Canonical order: "
                    "[image, traj_history, route, prompt, traj_future].")

    def __len__(self) -> int:
        return len(self._indices)

    def _base_sample_key(self, i: int) -> str:
        """Canonical ``clip::t0`` key shared with the offline cache builder."""
        if hasattr(self.base, "_samples"):
            entry = self.base._samples[i]
            return sample_key(entry["clip_id"], entry["t0_relative"])
        clip_id = self.base.clip_ids[i]
        t0_us = (
            self.base.DEFAULT_T0_US
            if self.base.use_default_keyframe
            else self.base.avdi.get_clip_key_frame(clip_id)
        )
        return sample_key(clip_id, t0_us)

    def _sample_key(self, i: int) -> str:
        """Public-index key; honors the cached-only subset used by smoke runs."""
        return self._base_sample_key(self._indices[i])

    def io_locality_keys(self) -> list[tuple[int, str]]:
        """Return ``(chunk, clip)`` per public index for locality-aware sampling.

        Anchors from the same clip reuse the same ZIP members, and clips from the same chunk
        reuse the same multi-GB ZIP shard through the node's page cache. The method exposes
        metadata only; it never reads image/video payloads.
        """
        if hasattr(self.base, "_samples"):
            clip_ids = [str(self.base._samples[i]["clip_id"]) for i in self._indices]
        else:
            clip_ids = [str(self.base.clip_ids[i]) for i in self._indices]
        if self.frame_cache_root is not None:
            # Kept answerable so a stale KAVA_IO_GROUPED_SAMPLER=1 does not crash, but the
            # grouping buys nothing here: the cache is in tmpfs, so random access is free and
            # plain shuffling gives strictly better-mixed batches.
            chunk_by_clip = self.base._chunk_by_clip
            return [(int(chunk_by_clip[c]), c) for c in clip_ids]
        chunks = self.base.avdi.clip_index.loc[clip_ids, "chunk"].to_numpy()
        if len(chunks) != len(clip_ids):
            raise RuntimeError(
                f"locality lookup returned {len(chunks)} chunks for {len(clip_ids)} samples"
            )
        return [(int(chunk), clip_id) for chunk, clip_id in zip(chunks, clip_ids)]

    def __getitem__(self, i: int) -> dict[str, Any] | None:
        base_i = self._indices[i]
        s = self.base[base_i]
        if s is None:
            return None
        s = dict(s)
        n_cam = s["image_frames"].shape[0]
        if n_cam != len(self.cameras):
            # Compatibility fallback for a custom/older PAIDataset that ignores the
            # ``camera_features`` attribute. Production reaches neither this branch nor the
            # two unused camera files.
            bad = [c for c in self.cameras if not 0 <= c < n_cam]
            if bad:
                raise IndexError(f"cameras {bad} out of range; the loader returned {n_cam}")
            for k in _CAMERA_AXIS_KEYS:
                if k in s and torch.is_tensor(s[k]):
                    s[k] = s[k][self.cameras]
        s["tokenized_data"] = self.pre(data=s)
        if self.teacher_trajectory_cache_root is not None:
            s["teacher_trajectory_states"] = teacher_trajectory_io.load_states(
                self.teacher_trajectory_cache_root, self._base_sample_key(base_i)
            )
        return s
