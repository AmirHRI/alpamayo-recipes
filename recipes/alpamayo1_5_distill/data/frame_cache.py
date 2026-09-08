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

r"""Read the pre-materialised frame cache in place of ``PAIDatasetWithNav``.

Drop-in for the raw half of the loader: same keys, same shapes, same dtypes, so
``CameraSubsetPAIDataset`` slices and preprocesses it without knowing the difference.  What
changes is where the pixels come from -- ``scripts/build_frame_cache.py`` wrote exactly the
879,976 images the manifest asks for, so a step reads ~2.7 MB from tmpfs instead of pulling
multi-GB chunk ZIPs across NFS.

WHY.  Measured on job 20710: the live loader streamed **1.64 TiB per epoch** at the ~120 MB/s
``/temp`` delivers, giving a 3.0 s median step against a 2.0 s GPU floor and three crashes when
a 1 GiB copy outran the 120 s dataloader timeout.  The cache is ~100-130 GiB, so it stages into
tmpfs once and every epoch after that reads nothing from NFS at all.

``nav_text`` is deliberately NOT cached -- it is read from the annotation JSON here, exactly as
``PAIDatasetWithNav`` does, so ``strip_nav_turn_distance`` and any future nav-text edit stay
free.  Only the pixels and the ego tensors, which are a pure function of ``(clip_id, t0_us)``,
come out of the cache.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Any

import numpy as np
import torch


class FrameCacheNavDataset(torch.utils.data.Dataset):
    """Anchors from an annotation JSON, pixels and ego tensors from the frame cache.

    Args:
        annotations_path: the same JSON ``PAIDatasetWithNav`` reads; each entry supplies
            ``clip_id``, ``t0_relative`` and ``nav_text``.
        cache_root: directory written by ``scripts/build_frame_cache.py``, containing
            ``_index.json`` and ``<chunk>/<clip_id>.zip``.
        cameras: loader camera indices the cache was built for. Checked against the index
            rather than trusted: training on a cache built for a different camera subset would
            change the prompt and silently misalign every KD target.
    """

    #: Keys the builder stored per anchor. They are written post-squeeze, i.e. exactly as
    #: ``PAIDatasetWithNav.__getitem__`` returned them, so no reshaping happens on the way back.
    _EGO_KEYS = (
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
        "ego_future_rot",
        "camera_indices",
        "relative_timestamps",
        "absolute_timestamps",
    )

    def __init__(
        self,
        annotations_path: str,
        cache_root: str,
        cameras: list[int] | tuple[int, ...] = (1, 3),
        require_all: bool = True,
        **ignored: Any,
    ) -> None:
        self.cache_root = os.path.abspath(cache_root)
        self.cameras = [int(c) for c in cameras]

        index_path = os.path.join(self.cache_root, "_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"no frame-cache index at {index_path}; run "
                "`build_frame_cache.py --finalize` after the build completes"
            )
        with open(index_path) as handle:
            index = json.load(handle)
        if list(index.get("cameras", [])) != self.cameras:
            raise ValueError(
                f"frame cache at {self.cache_root} was built for cameras "
                f"{index.get('cameras')}, but this run asks for {self.cameras}. The prompt and "
                "the vision token count depend on the camera set, so this would train on "
                "pixels that do not match the sequence."
            )
        self._chunk_by_clip: dict[str, int] = {k: int(v) for k, v in index["clips"].items()}
        self.crf = index.get("crf")

        with open(annotations_path) as handle:
            samples: list[dict[str, Any]] = json.load(handle)

        # The cache holds precisely the anchors the live loader kept after its chunk_ids and
        # clip_uuid_filter pass, because it was built through that same dataset object. Anything
        # the manifest names that the cache lacks is a build that did not finish, so say so
        # rather than quietly training on a subset.
        kept = [s for s in samples if s["clip_id"] in self._chunk_by_clip]
        missing = len(samples) - len(kept)
        if missing and require_all:
            example = next(s["clip_id"] for s in samples if s["clip_id"] not in self._chunk_by_clip)
            raise RuntimeError(
                f"{missing}/{len(samples)} annotated anchors are absent from the frame cache "
                f"at {self.cache_root} (e.g. clip {example}). Re-run the build -- it skips "
                "clips that are already complete -- then --finalize. Pass require_all=false "
                "only for a deliberate smoke run on a partial cache."
            )
        self._samples = kept
        self.clip_ids = [s["clip_id"] for s in self._samples]
        print(
            f"[framecache] {len(self._samples)} anchors over {len(self._chunk_by_clip)} clips, "
            f"cameras {self.cameras}, crf={self.crf}, root={self.cache_root}"
            + (f" (dropped {missing} not in cache)" if missing else ""),
            flush=True,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _zip_path(self, clip_id: str) -> str:
        return os.path.join(self.cache_root, f"{self._chunk_by_clip[clip_id]:04d}", f"{clip_id}.zip")

    @staticmethod
    def _decode(payload: bytes) -> torch.Tensor:
        """Decode a 4-frame mini-clip to ``(T, 3, H, W)`` uint8."""
        import av

        with av.open(io.BytesIO(payload)) as container:
            frames = [
                torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                for frame in container.decode(video=0)
            ]
        if not frames:
            raise RuntimeError("frame-cache mini-clip decoded to zero frames")
        return torch.stack(frames)

    def __getitem__(self, i: int) -> dict[str, Any]:
        entry = self._samples[i]
        clip_id = entry["clip_id"]
        t0_us = int(entry["t0_relative"])
        path = self._zip_path(clip_id)

        # Opened per item rather than held: the ZIP is ~2.7 MB in tmpfs with a 13-entry central
        # directory, so the open costs microseconds, while a cross-worker handle cache would
        # have to survive fork and eviction for no measurable gain.
        with zipfile.ZipFile(path) as zf:
            images = torch.stack(
                [self._decode(zf.read(f"{t0_us}.cam{c}.mp4")) for c in self.cameras]
            )
            with np.load(io.BytesIO(zf.read(f"{t0_us}.ego.npz"))) as ego:
                tensors = {k: torch.from_numpy(ego[k]) for k in self._EGO_KEYS if k in ego}

        sample: dict[str, Any] = {"image_frames": images, **tensors}
        sample["t0_us"] = t0_us
        sample["clip_id"] = clip_id
        sample["nav_text"] = entry["nav_text"]
        return sample
