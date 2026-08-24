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

from typing import Any

import torch
from hydra.utils import instantiate

#: Keys whose leading dimension is the camera axis, as ``load_physical_aiavdataset`` returns
#: them: ``image_frames`` is [N_cam, num_frames, 3, H, W] and the rest are [N_cam, num_frames].
#: ⚠️ ``reshape_tensors_for_rl`` would flatten that axis away; it defaults False and must stay
#: False for this class, which is why it is not exposed.
_CAMERA_AXIS_KEYS = ("image_frames", "camera_indices", "absolute_timestamps",
                     "relative_timestamps")


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

    def __init__(self, cameras, vla_preprocess_args=None, model_config=None,
                 annotations_path: str | None = None, **pai_kwargs: Any) -> None:
        from alpamayo.data.pai import PAIDataset
        from alpamayo.data.pai_nav import PAIDatasetWithNav

        if vla_preprocess_args is None:
            raise ValueError(
                "CameraSubsetPAIDataset needs vla_preprocess_args: it exists to run the "
                "processor AFTER slicing, so there is nothing to do without one.")
        self.cameras = [int(c) for c in cameras]
        # ⚠️ model_config=None / no preprocess on the base: we want the RAW sample. With a nav
        # base that also means PAIDatasetWithNav skips its own preprocess, leaving nav_text in
        # the raw dict for our processor to pick up after slicing.
        if annotations_path is not None:
            self.base = PAIDatasetWithNav(annotations_path=annotations_path, **pai_kwargs,
                                          model_config=None, vla_preprocess_args=None)
        else:
            self.base = PAIDataset(**pai_kwargs, model_config=None, vla_preprocess_args=None)
        self.pre = instantiate(vla_preprocess_args, model_config=model_config)
        kind = "nav samples" if annotations_path is not None else "clips"
        print(f"[camsubset] {len(self.base)} {kind}, cameras {self.cameras} "
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
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, Any] | None:
        s = self.base[i]
        if s is None:
            return None
        s = dict(s)
        n_cam = s["image_frames"].shape[0]
        bad = [c for c in self.cameras if not 0 <= c < n_cam]
        if bad:
            raise IndexError(f"cameras {bad} out of range; the loader returned {n_cam}")
        for k in _CAMERA_AXIS_KEYS:
            if k in s and torch.is_tensor(s[k]):
                s[k] = s[k][self.cameras]
        s["tokenized_data"] = self.pre(data=s)
        return s
