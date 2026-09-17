# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Pull the GT ego trajectories for a nav-anchor manifest, WITHOUT decoding any video.

``PAIDatasetWithNav.__getitem__`` calls ``load_physical_aiavdataset``, which reads egomotion
AND decodes 4 frames x N cameras. For a distribution plot the pixels are pure cost -- 50k
samples of video decode is hours; egomotion alone is minutes.

⚠️ The local-frame math is copied verbatim from ``load_physical_aiavdataset`` (the history/
future timestamp offsets, ``t0_xyz = ego_history_xyz[-1]``, and
``xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)``). If those diverge this stops matching the
trajectories the model is actually trained on, so the offsets are asserted against the
loader's documented shapes rather than re-derived.

⚠️ Egomotion is fetched ONCE PER CLIP and re-evaluated at each anchor's t0. The 50k manifest
has far fewer unique clips than anchors, so this is the difference between minutes and hours.

Usage::

    python -m alpamayo1_5_distill.scripts.extract_train_trajectories \
        manifest=/data/.../nav_lcdrive_train_anchors_50k_turnpreserved.json \
        out=/data/.../train50k_trajectories.npz [limit=0] [workers=12]
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.spatial.transform as spt

NUM_HISTORY = 16
NUM_FUTURE = 64
TIME_STEP = 0.1


def _offsets():
    """Byte-for-byte the loader's timestamp offsets, in microseconds."""
    us = TIME_STEP * 1_000_000
    hist = np.arange(-(NUM_HISTORY - 1) * us, us / 2, us).astype(np.int64)
    fut = np.arange(us, (NUM_FUTURE + 0.5) * us, us).astype(np.int64)
    assert hist.shape == (NUM_HISTORY,), hist.shape
    assert fut.shape == (NUM_FUTURE,), fut.shape
    return hist, fut


def main() -> None:
    argv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    manifest = argv["manifest"]
    out = argv["out"]
    limit = int(argv.get("limit", 0))

    # ⚠️ The LOCAL interface, matching PAIDataset. The streaming
    # ``PhysicalAIAVDatasetInterface()`` also works but costs ~1.8 s/clip -- 25 h over this
    # manifest, which has ONE anchor per clip and so nothing to amortise.
    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface

    avdi = PhysicalAIAVDatasetLocalInterface(
        local_dir=argv.get("local_dir", "/data/datasets/physical_ai_av/"),
        chunk_ids=argv.get("chunk_ids", "0-3146"),
    )
    hist_off, fut_off = _offsets()

    entries = json.load(open(manifest))
    if limit:
        entries = entries[:limit]
    by_clip = defaultdict(list)
    for i, e in enumerate(entries):
        by_clip[e["clip_id"]].append(i)
    print(f"[traj] {len(entries)} anchors over {len(by_clip)} unique clips", flush=True)

    fut = np.full((len(entries), NUM_FUTURE, 3), np.nan, dtype=np.float32)
    hst = np.full((len(entries), NUM_HISTORY, 3), np.nan, dtype=np.float32)
    yaw = np.full(len(entries), np.nan, dtype=np.float32)
    ok = np.zeros(len(entries), dtype=bool)

    t_start = time.time()
    for n, (clip_id, idxs) in enumerate(by_clip.items(), start=1):
        try:
            ego = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION)
        except Exception as ex:                                  # noqa: BLE001
            print(f"[traj] clip {clip_id} egomotion failed: {type(ex).__name__}: {ex}",
                  flush=True)
            continue
        for i in idxs:
            t0 = int(entries[i]["t0_relative"])
            try:
                h = ego(t0 + hist_off)
                f = ego(t0 + fut_off)
                h_xyz = h.pose.translation
                t0_xyz = h_xyz[-1].copy()
                t0_inv = spt.Rotation.from_quat(h.pose.rotation.as_quat()[-1].copy()).inv()
                fut[i] = t0_inv.apply(f.pose.translation - t0_xyz).astype(np.float32)
                hst[i] = t0_inv.apply(h_xyz - t0_xyz).astype(np.float32)
                # net heading change over the 6.4 s future, from the final rotation
                r = (t0_inv * spt.Rotation.from_quat(f.pose.rotation.as_quat()[-1])).as_matrix()
                yaw[i] = float(np.arctan2(r[1, 0], r[0, 0]))
                ok[i] = True
            except Exception as ex:                              # noqa: BLE001
                print(f"[traj] anchor {i} ({clip_id} @{t0}) failed: {type(ex).__name__}",
                      flush=True)
        if n % 50 == 0 or n == len(by_clip):
            el = time.time() - t_start
            print(f"[traj] {n}/{len(by_clip)} clips  {int(ok.sum())} anchors  "
                  f"{el:.0f}s  eta {el / n * (len(by_clip) - n):.0f}s", flush=True)

    np.savez_compressed(
        out,
        future_xyz=fut, history_xyz=hst, final_yaw=yaw, ok=ok,
        clip_ids=np.array([e["clip_id"] for e in entries]),
        t0_relative=np.array([int(e["t0_relative"]) for e in entries], dtype=np.int64),
        nav_text=np.array([e.get("nav_text", "") for e in entries]),
        meta_action=np.array([",".join(e.get("meta_action", [])) for e in entries]),
    )
    print(f"[traj] wrote {out}: {int(ok.sum())}/{len(entries)} anchors resolved", flush=True)
    print("DONE_EXTRACT", flush=True)


if __name__ == "__main__":
    main()
