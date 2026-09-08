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

r"""The frame cache must be substitutable for the live loader, token for token.

The KD objective reads ONE ``input_ids`` tensor into both towers and aligns their KV caches
position by position, so a cache that shifted the sequence by even one token would misalign
every target while leaving the loss curve looking healthy. This asserts the substitution on
real anchors:

* prompt ``text`` and ``image_grid_thw`` EXACTLY equal -- the processor emits those and the
  collate turns them into ``input_ids``, so equality here is the same guarantee one step
  earlier, and it says whether a regression is in the prompt or in the image geometry.
* ego tensors EXACTLY equal -- they are cached verbatim, so this is a round-trip check.
* ``pixel_values`` CLOSE -- crf18 is a re-encode, so these differ by design. The bound is on
  how much, measured after the processor's downscale to 576x320, where the resampling washes
  out most of the codec's high-frequency error.

Run against a partial cache (a smoke build is enough)::

    pytest -s recipes/alpamayo1_5_distill/tests/test_frame_cache_equivalence.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "recipes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE_ROOT = os.environ.get(
    "FRAME_CACHE_ROOT", "/temp/achahe/physical_ai_av/framecache_nav2cam_1080p"
)
MANIFEST = (
    "/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/"
    "nav_lcdrive_train_anchors_all.json"
)
LOCAL_DIR = "/temp/achahe/physical_ai_av/"
CLIP_FILTER = (
    "/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/lcdrive_train_clip_uuids.txt"
)
CAMERAS = [1, 3]
#: Cap on mean |Δ| in normalised pixel units after the processor. crf18 round-trips the source
#: frames at ~42 dB PSNR before downscaling; anything near this bound means the codec setting
#: regressed, not that the pipeline broke.
PIXEL_TOLERANCE = 0.02

VLA_PREPROCESS_ARGS = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "traj_history", "route", "prompt", "traj_future"],
    "components_prompt": ["traj_future"],
    "label_components": ["traj_future"],
    "include_camera_ids": True,
    "include_frame_nums": True,
    "generation_mode": True,
}


def _built_clips() -> dict[str, int]:
    """Map clip -> chunk for every clip ZIP present, so a partial cache is testable."""
    clips: dict[str, int] = {}
    for path in glob.glob(os.path.join(CACHE_ROOT, "[0-9]" * 4, "*.zip")):
        clips[os.path.basename(path)[: -len(".zip")]] = int(os.path.basename(os.path.dirname(path)))
    return clips


@pytest.fixture(scope="module")
def partial_index(tmp_path_factory):
    """A cache root exposing only the clips that are actually built.

    Symlinked into a temp dir rather than writing ``_index.json`` into the real cache, so this
    never races the build job that may still be filling it.
    """
    # Once the build has been finalized, test the real thing: same root, same index, same code
    # path training will take. The symlink scaffolding below exists only so a half-built cache
    # is still testable during a build.
    if os.path.exists(os.path.join(CACHE_ROOT, "_index.json")):
        return CACHE_ROOT

    clips = _built_clips()
    if not clips:
        pytest.skip(f"no clip ZIPs under {CACHE_ROOT}; run the build first")
    root = tmp_path_factory.mktemp("framecache")
    for chunk in sorted(set(clips.values())):
        os.symlink(os.path.join(CACHE_ROOT, f"{chunk:04d}"), root / f"{chunk:04d}")
    (root / "_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "cameras": CAMERAS,
                "crf": 18,
                "manifest": MANIFEST,
                "clips": clips,
                "anchors": -1,
            }
        )
    )
    return str(root)


@pytest.fixture(scope="module")
def model_config():
    """The six fields the processor factory reads, taken from the shipped config.

    ``AutoConfig`` would need the ``alpamayo_r1`` architecture registered, which drags the whole
    model in for a data test. What matters here is that BOTH datasets are preprocessed by the
    same processor, and these are the real values that processor is built from.
    """
    import types

    path = "/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format/config.json"
    with open(path) as handle:
        raw = json.load(handle)
    fields = (
        "vlm_name_or_path",
        "traj_vocab_size",
        "min_pixels",
        "max_pixels",
        "tokens_per_history_traj",
        "tokens_per_future_traj",
    )
    missing = [f for f in fields if f not in raw]
    assert not missing, f"{path} is missing {missing}"
    cfg = types.SimpleNamespace(**{f: raw[f] for f in fields})
    # The local snapshot, not the hub id: the test box has no network.
    cfg.vlm_name_or_path = (
        "/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/"
        "a9fae2cf89dc64db96b12860417f0eb403013bb9"
    )
    return cfg


def _live_dataset(model_config):
    from alpamayo1_5_distill.data.camera_subset import CameraSubsetPAIDataset

    return CameraSubsetPAIDataset(
        cameras=CAMERAS,
        annotations_path=MANIFEST,
        local_dir=LOCAL_DIR,
        chunk_ids="0-3146",
        clip_uuid_filter=CLIP_FILTER,
        vla_preprocess_args=VLA_PREPROCESS_ARGS,
        model_config=model_config,
    )


def _cached_dataset(model_config, root):
    from alpamayo1_5_distill.data.camera_subset import CameraSubsetPAIDataset

    return CameraSubsetPAIDataset(
        cameras=CAMERAS,
        annotations_path=MANIFEST,
        frame_cache_root=root,
        frame_cache_require_all=False,
        vla_preprocess_args=VLA_PREPROCESS_ARGS,
        model_config=model_config,
    )


def test_cached_samples_match_live_loader(partial_index, model_config):
    live = _live_dataset(model_config)
    cached = _cached_dataset(model_config, partial_index)
    assert len(cached) > 0, "cached dataset is empty; the partial index found no anchors"

    live_by_key = {live._sample_key(i): i for i in range(len(live))}
    checked = 0
    worst_pixels = 0.0
    # Spread across the cache rather than taking the first N: consecutive anchors share a clip
    # (3.44 of them on average), so the head of the dataset would only exercise a handful of
    # ZIPs and one corner of the chunk range.
    n_probe = min(len(cached), 12)
    stride = max(1, len(cached) // n_probe)
    for j in range(0, stride * n_probe, stride):
        key = cached._sample_key(j)
        assert key in live_by_key, f"cached anchor {key} is absent from the live dataset"
        a = live[live_by_key[key]]
        b = cached[j]

        # The processor emits the prompt as text plus the vision grid; the collate turns those
        # into input_ids. Equal text and equal grid is therefore the same guarantee one step
        # earlier -- and it localises a regression to the prompt or to the image geometry.
        assert a["tokenized_data"]["text"] == b["tokenized_data"]["text"], (
            f"prompt text differs for {key}: the cache changed the sequence, "
            "KD targets would misalign"
        )
        assert torch.equal(
            a["tokenized_data"]["image_grid_thw"], b["tokenized_data"]["image_grid_thw"]
        ), f"image_grid_thw differs for {key}: the vision token count changed"

        for k in (
            "ego_history_xyz",
            "ego_history_rot",
            "ego_future_xyz",
            "ego_future_rot",
            "camera_indices",
            "absolute_timestamps",
        ):
            assert torch.equal(a[k], b[k]), f"{k} differs for {key}"
        assert a["nav_text"] == b["nav_text"]

        # Camera and frame ORDER, checked against the pixels themselves. Everything above is
        # blind to a transposed camera axis or a reversed frame axis: the prompt labels come
        # from camera_indices, which is cached, so a swap would carry through consistently and
        # look correct. The cross-difference matrix is what actually pins it -- the matching
        # pair differs only by the codec (~1.5/255), any mismatched pair by far more.
        live_f, cached_f = a["image_frames"].float(), b["image_frames"].float()
        for axis, size in (("camera", live_f.shape[0]), ("frame", live_f.shape[1])):
            for i in range(size):
                if axis == "camera":
                    diffs = [(live_f[i] - cached_f[j]).abs().mean().item() for j in range(size)]
                else:
                    diffs = [
                        (live_f[0, i] - cached_f[0, j]).abs().mean().item() for j in range(size)
                    ]
                best = min(range(size), key=lambda j: diffs[j])
                assert best == i, (
                    f"{axis} axis is permuted for {key}: live {axis} {i} matches cached "
                    f"{axis} {best} (diffs {['%.2f' % d for d in diffs]})"
                )

        pa = a["tokenized_data"]["pixel_values"].float()
        pb = b["tokenized_data"]["pixel_values"].float()
        assert pa.shape == pb.shape, f"pixel_values shape differs for {key}"
        delta = (pa - pb).abs().mean().item()
        worst_pixels = max(worst_pixels, delta)
        assert delta < PIXEL_TOLERANCE, (
            f"pixel_values drifted by {delta:.4f} for {key} (bound {PIXEL_TOLERANCE}); "
            "check the crf the cache was built with"
        )
        checked += 1

    print(
        f"\n[frame-cache] {checked} anchors: prompt text + image_grid_thw identical, "
        f"ego identical, worst mean |Δpixel| {worst_pixels:.4f} (bound {PIXEL_TOLERANCE})"
    )
