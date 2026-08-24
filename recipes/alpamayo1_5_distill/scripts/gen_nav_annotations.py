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

r"""Generate the nav-text annotations JSON that ``PAIDatasetWithNav`` reads.

Alpamayo 1.5 takes navigation intent as TEXT wrapped in ``<|route_start|>...<|route_end|>``
(``chat_template/components.py::construct_route``), and the ``route`` component returns an empty
list when ``nav_text`` is absent -- which is how the same model runs with and without
navigation. The component and its special tokens already exist; the only missing piece was a
producer for ``nav_text`` on PAI, which this script is.

⚠️ SCHEMA. ``PAIDatasetWithNav.__getitem__`` reads ``entry["t0_relative"]``, NOT ``entry["t0"]``
as its own docstring example shows, and merges ONLY ``nav_text`` (the docstring's ``cot`` field
is ignored). Following the documented example crashes with KeyError: 't0_relative'. Emitted here:

    [{"clip_id": "0000e4f1-...", "t0_relative": 5100000, "nav_text": "Turn left in 11m"}, ...]

⚠️ t0_relative is written as DEFAULT_T0_US (5,100,000) to match ``use_default_keyframe: true``,
which every existing arm in this tree trains and evaluates at. PAIDatasetWithNav ignores
``use_default_keyframe`` and uses this field, so a different value here would silently move the
sampling point and make the numbers incomparable.

⚠️ NO VIDEO IS DECODED. The route comes from ``LABELS.EGOMOTION`` evaluated at the future
timestamps and rotated into the t0 frame -- the same transform ``load_physical_aiavdataset``
applies (``t0_rot.inv().apply(xyz - t0_xyz)``), which is what makes the result a rig-frame
polyline with +x forward. Going through the dataset's ``__getitem__`` instead would decode 16
video frames per clip and take hours for 38k clips.

⚠️ THE ROUTE HERE IS THE GT FUTURE. In alpasim the route is planned and independent of what the
ego does; on PAI the only available polyline is the trajectory the ego actually drove. So nav
text derived this way RESTATES the label's direction. That is legitimate conditioning for
training, but an eval that also sees it is being handed the answer's direction -- report which
way each arm was scored. ``--horizon-start`` exists for this: it drops the near waypoints so the
instruction describes intent BEYOND the prediction horizon rather than inside it.

Usage::

    python -m alpamayo1_5_distill.scripts.gen_nav_annotations \
        --clip-list /data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests/lcdrive_train_clip_uuids.txt \
        --out /data/.../nav_train.json --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_T0_US = 5_100_000
_MIN_SEGMENT_LENGTH_M = 1e-3


def route_to_nav_text(
    route_in_rig: np.ndarray,
    *,
    min_lookahead_m: float = 5.0,
    turn_angle_deg: float = 30.0,
    distance_lookahead_m: float = 40.0,
) -> str | None:
    """Classify the route ahead as a left turn, a right turn, or straight.

    Verbatim from the alpasim classifier except for typing, so the wording and thresholds -- the
    wording the model was trained on -- are unchanged. Turns are classified by HEADING change,
    not lateral offset: a fixed lateral threshold is ~24 deg of heading at 5 m but only ~3 deg at
    40 m, so it reads parallel lane offsets as turns up close and misses real turns further out.
    """
    if route_in_rig.size == 0:
        return None
    xy = route_in_rig[:, :2]
    points = xy[~np.isnan(xy).any(axis=1)]
    if len(points) == 0:
        return None
    if len(points) >= 2:
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > _MIN_SEGMENT_LENGTH_M
        points = points[keep]
    # Two segments are needed for a heading change; with one, a straight road at an angle to the
    # ego is indistinguishable from a turn.
    if len(points) < 3:
        return "Continue straight"
    distances = np.linalg.norm(points, axis=1)
    # Initial heading from the chord across the near zone, not the first segment, so a constant
    # yaw offset between ego and road does not read as a turn.
    near = np.flatnonzero(distances <= min_lookahead_m)
    start, end = (near[0], near[-1]) if len(near) >= 2 else (0, 1)
    base = points[end] - points[start]
    if np.linalg.norm(base) < _MIN_SEGMENT_LENGTH_M:
        base = points[1] - points[0]
    initial_heading = np.arctan2(base[1], base[0])
    segments = np.diff(points, axis=0)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    deviations = np.remainder(headings - initial_heading + np.pi, 2 * np.pi) - np.pi
    endpoint_distances = distances[1:]
    turning = (
        (endpoint_distances >= min_lookahead_m)
        & (endpoint_distances <= distance_lookahead_m)
        & (np.abs(deviations) > np.radians(turn_angle_deg))
    )
    if not turning.any():
        return "Continue straight"
    index = int(np.argmax(turning))
    direction = "left" if deviations[index] > 0 else "right"
    return f"Turn {direction} in {max(1, round(endpoint_distances[index]))}m"


_AVDI_CACHE: dict[tuple[str, str], object] = {}


def _avdi(local_dir: str, chunk_ids: str):
    """One interface per worker process, not per clip.

    Constructing PhysicalAIAVDatasetLocalInterface parses the chunk metadata, which at
    chunk_ids=0-3146 costs far more than the egomotion query it enables. Building it per
    anchor made the 110k-anchor run interface-construction-bound.
    """
    key = (local_dir, chunk_ids)
    if key not in _AVDI_CACHE:
        from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface

        _AVDI_CACHE[key] = PhysicalAIAVDatasetLocalInterface(
            local_dir=local_dir, chunk_ids=chunk_ids,
            features_metadata="features.csv", clip_index_metadata="clip_index.parquet")
    return _AVDI_CACHE[key]


def _routes_and_yaws(clip_id: str, local_dir: str, chunk_ids: str, t0_list: list[int],
                     n_future: int, time_step: float, horizon_start: int):
    """Future polyline in the t0 rig frame, plus total yaw change, for EACH t0 of one clip.

    Batched over t0 because ``get_clip_feature`` streams the clip's egomotion once and the
    anchors average 3.44 per clip; per-anchor calls refetched the same feature.

    Yaw is returned only so the caller can CHECK the y-sign convention: in a right-handed,
    z-up ego frame a left turn has both positive lateral offset and positive yaw change. If
    the two disagreed, left/right would be silently swapped for every sample.

    Also returns the REVERSE FRACTION: the share of moving steps whose velocity is
    anti-aligned with the car's own forward axis. ``route_to_nav_text`` cannot detect
    reversing -- it takes its reference heading from the route's own first chord, so driving
    straight backwards has near-zero deviation from it and classifies as "Continue straight".
    Net backward displacement in the t0 frame does NOT identify these: a U-turn also ends up
    behind the start while driving forward throughout (measured: 179 anchors have net
    backward displacement, but only 42 actually reverse; the rest are U-turns whose
    "Turn left/right" label is correct). Projecting each step's velocity onto that step's
    body x-axis separates the two.
    """
    from scipy.spatial.transform import Rotation

    avdi = _avdi(local_dir, chunk_ids)
    ego = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    # ⚠️ t0's pose comes from the HISTORY sample at t0, exactly as load_physical_aiavdataset
    # does (`t0_xyz = ego_history_xyz[-1]`), not from a separate query -- and `pose.rotation`
    # is already a scipy Rotation, so `.as_quat()` is the accessor (there is no `.matrix`).
    hist_off = (np.arange(-(16 - 1), 1) * time_step * 1e6).astype(np.int64)
    fut_off = (np.arange(1, n_future + 1) * time_step * 1e6).astype(np.int64)
    results = []
    for t0_us in t0_list:
        hist = ego(t0_us + hist_off)
        fut = ego(t0_us + fut_off)
        h_xyz = np.asarray(hist.pose.translation, dtype=np.float64)
        f_xyz = np.asarray(fut.pose.translation, dtype=np.float64)
        t0_xyz = h_xyz[-1].copy()
        r0_inv = Rotation.from_quat(hist.pose.rotation.as_quat()[-1].copy()).inv()
        local = r0_inv.apply(f_xyz - t0_xyz)
        f_rot = r0_inv * Rotation.from_quat(fut.pose.rotation.as_quat())
        yaw = f_rot.as_euler("zyx")[-1][0]
        # reverse detection, on the FULL future (a property of the manoeuvre, not of the
        # horizon-cropped view the classifier sees)
        fwd = f_rot.as_matrix()[:, :, 0]                  # body +x axis in the t0 frame
        vel = np.diff(np.vstack([np.zeros(3), local]), axis=0)
        along = (vel * fwd).sum(1)
        step = np.linalg.norm(vel, axis=1)
        moving = step > 0.02                              # a parked car has no direction
        rev = (float((along[moving] < 0).mean()) if moving.sum() >= 5 else 0.0)
        travelled = float(step[moving].sum())
        results.append((local[horizon_start:], float(yaw),
                        rev if travelled > 1.0 else 0.0))
    return results


def _route_and_yaw(clip_id: str, local_dir: str, chunk_ids: str, t0_us: int,
                   n_future: int, time_step: float, horizon_start: int):
    """Single-t0 form, kept so the fixed-keyframe path and probes are unchanged."""
    route, yaw, rev = _routes_and_yaws(clip_id, local_dir, chunk_ids, [t0_us],
                                       n_future, time_step, horizon_start)[0]
    return clip_id, route, yaw, rev


def _one(args):
    """Single-anchor worker: (clip_id, nav_text, yaw, y_end, reverse_fraction)."""
    clip_id, local_dir, chunk_ids, t0_us, n_future, time_step, horizon_start = args
    try:
        cid, route, yaw, rev = _route_and_yaw(clip_id, local_dir, chunk_ids, t0_us,
                                             n_future, time_step, horizon_start)
        return (cid, route_to_nav_text(route), yaw,
                float(route[-1][1]) if len(route) else 0.0, rev)
    except Exception:                             # a missing chunk must not kill 38k clips
        if os.environ.get("NAV_DEBUG") == "1":
            import traceback; traceback.print_exc()
        return clip_id, None, None, None, 0.0


def _one_clip(args):
    """Multi-anchor worker: all t0 of one clip -> [(t0_us, nav_text, yaw, y_end), ...].

    A per-clip failure (missing chunk, unreadable egomotion) drops that clip's anchors and
    is counted, rather than killing the pool.
    """
    clip_id, local_dir, chunk_ids, t0_list, n_future, time_step, horizon_start = args
    try:
        rs = _routes_and_yaws(clip_id, local_dir, chunk_ids, t0_list,
                              n_future, time_step, horizon_start)
    except Exception:
        if os.environ.get("NAV_DEBUG") == "1":
            import traceback; traceback.print_exc()
        return clip_id, None
    out = []
    for t0_us, (route, yaw, rev) in zip(t0_list, rs):
        out.append((t0_us, route_to_nav_text(route), yaw,
                    float(route[-1][1]) if len(route) else 0.0, rev))
    return clip_id, out


def load_anchors(path: str, hz: float, categories: set[str] | None,
                 lo_us: int, hi_us: int):
    """Autolabeler segments -> {clip_id: {t0_us: sorted[meta_action]}}, window-filtered.

    ⚠️ FRAME RATE. ``event_start_frame`` is a 10 Hz index (t0_us = frame * 100_000), not the
    30 Hz camera rate. Measured three ways: stop-event onsets recover 10.1-10.7 Hz; clips are
    605 frames / 30 fps = 20.1 s and only 10 Hz maps frames 20-135 onto the whole clip; and
    the labeler's own frame range lands exactly inside the loader's valid t0 window, which it
    only does at 10 Hz. A 30 Hz reading would shift every t0 by ~2/3.
    """
    with open(path) as fh:
        segs = json.load(fh)
    anchors: dict[str, dict[int, set[str]]] = {}
    n_seg = n_win = 0
    for cat, lst in segs.items():
        if categories is not None and cat not in categories:
            continue
        for s in lst:
            n_seg += 1
            t0_us = int(round(s["event_start_frame"] * 1e6 / hz))
            # the loader asserts t0 > num_history*time_step, and the future must fit the clip
            if not (lo_us <= t0_us <= hi_us):
                n_win += 1
                continue
            anchors.setdefault(s["clip_id"], {}).setdefault(t0_us, set()).add(cat)
    n_anchor = sum(len(v) for v in anchors.values())
    print(f"[nav] anchors: {n_seg} segments -> {n_anchor} distinct (clip,t0) over "
          f"{len(anchors)} clips  ({n_seg - n_win - n_anchor} duplicate t0, "
          f"{n_win} outside window)", flush=True)
    return anchors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-list", default=None,
                    help="one clip uuid per line; with --anchors it FILTERS the anchors")
    ap.add_argument("--anchors", default=None,
                    help="autolabeler segments JSON; emits one entry per distinct (clip,t0) "
                         "event anchor instead of one per clip at a fixed t0")
    ap.add_argument("--anchor-hz", type=float, default=10.0,
                    help="frame rate of event_start_frame (see load_anchors)")
    ap.add_argument("--anchor-categories", default=None,
                    help="comma-separated meta_action allowlist (default: all)")
    ap.add_argument("--fallback-default-t0", action="store_true",
                    help="with --anchors and --clip-list: for clips in the list that the "
                         "autolabeler never labelled, emit one entry at --t0-us instead of "
                         "skipping them. Needed for EVAL, where dropping unlabelled clips "
                         "would silently shrink the scored set and break comparability with "
                         "runs measured on the full list. Tagged meta_action=['no_event'].")
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-dir", default="/data/datasets/physical_ai_av/")
    ap.add_argument("--chunk-ids", default="0-3146")
    ap.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    ap.add_argument("--n-future", type=int, default=64)
    ap.add_argument("--num-history-steps", type=int, default=16)
    ap.add_argument("--clip-duration-s", type=float, default=20.1)
    ap.add_argument("--time-step", type=float, default=0.1)
    ap.add_argument("--horizon-start", type=int, default=0,
                    help="drop this many leading future waypoints, so the instruction can "
                         "describe intent BEYOND the prediction horizon instead of restating it")
    ap.add_argument("--reverse-policy", choices=("label", "drop", "keep"), default="label",
                    help="what to do with anchors whose future is driven in REVERSE. "
                         "route_to_nav_text cannot see reversing (it references the route's "
                         "own heading), so it calls a straight reverse 'Continue straight' -- "
                         "a wrong instruction. Default 'label' rewrites the text to "
                         "--reverse-text, keeping the anchor; 'drop' removes it; 'keep' "
                         "leaves the incorrect forward text. Measured: 42 of 109,997.")
    ap.add_argument("--reverse-text", default="Reverse",
                    help="nav_text for reversing anchors. No canonical phrasing exists in "
                         "this tree -- nav_text is free text inside the route tokens -- so "
                         "this mirrors the style of 'Continue straight'. NO direction is "
                         "asserted: for a reversing car 'left' is ambiguous (nose vs rear), "
                         "and the autolabeler's own reverse_left/right tags match the "
                         "geometry on only 18 of 28 anchors, so there is no reliable "
                         "convention to follow.")
    ap.add_argument("--reverse-thresh", type=float, default=0.5,
                    help="fraction of moving steps that must be reversing to count")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not a.clip_list and not a.anchors:
        ap.error("need --clip-list or --anchors")

    out: list[dict] = []
    counts: dict[str, int] = {}
    agree = checked = failed = dropped = reversed_out = 0

    def record(nav, yaw, y_end, rev=0.0):
        """Final nav_text for one anchor, or None if it must not be emitted."""
        nonlocal agree, checked, reversed_out
        if nav is None:
            return None
        is_rev = rev > a.reverse_thresh
        if is_rev:
            reversed_out += 1
            if a.reverse_policy == "drop":
                return None
            if a.reverse_policy == "label":
                nav = a.reverse_text
        counts[nav.split(" in ")[0]] = counts.get(nav.split(" in ")[0], 0) + 1
        # convention check on strong turns only, and never on a rewritten reverse -- its
        # yaw/lateral relationship describes a reversing manoeuvre, not a turn
        if not is_rev and yaw is not None and abs(yaw) > 0.3:
            checked += 1
            agree += int(np.sign(y_end) == np.sign(yaw))
        return nav

    if a.anchors:
        # t0 must leave room for the history window behind it and the future ahead of it
        lo_us = int(a.num_history_steps * a.time_step * 1e6) + 100_000
        hi_us = int((a.clip_duration_s - a.n_future * a.time_step) * 1e6)
        cats = set(a.anchor_categories.split(",")) if a.anchor_categories else None
        anchors = load_anchors(a.anchors, a.anchor_hz, cats, lo_us, hi_us)
        if a.clip_list:
            allowed = {l.strip() for l in open(a.clip_list) if l.strip()}
            before = len(anchors)
            anchors = {c: v for c, v in anchors.items() if c in allowed}
            print(f"[nav] --clip-list filter: {before} -> {len(anchors)} clips "
                  f"(of {len(allowed)} listed)", flush=True)
            if a.fallback_default_t0:
                missing = sorted(allowed - set(anchors))
                for c in missing:
                    anchors[c] = {a.t0_us: {"no_event"}}
                print(f"[nav] fallback: {len(missing)} listed clips have no labelled event "
                      f"-> one entry each at t0={a.t0_us / 1e6:.1f}s", flush=True)
        clips = sorted(anchors)
        if a.limit:
            clips = clips[: a.limit]
        total = sum(len(anchors[c]) for c in clips)
        print(f"[nav] {total} anchors over {len(clips)} clips, valid t0 window "
              f"[{lo_us / 1e6:.1f},{hi_us / 1e6:.1f}]s, horizon_start={a.horizon_start}",
              flush=True)
        jobs = [(c, a.local_dir, a.chunk_ids, sorted(anchors[c]), a.n_future, a.time_step,
                 a.horizon_start) for c in clips]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(_one_clip, j) for j in jobs]), 1):
                cid, res = fut.result()
                if res is None:
                    failed += len(anchors[cid])
                else:
                    for t0_us, nav, yaw, y_end, rev in res:
                        text = record(nav, yaw, y_end, rev)
                        if text is not None:
                            out.append({"clip_id": cid, "t0_relative": t0_us,
                                        "nav_text": text,
                                        "meta_action": sorted(anchors[cid][t0_us])})
                        else:
                            # PAIDatasetWithNav.__getitem__ does entry["nav_text"] unguarded,
                            # so an entry without it would KeyError mid-epoch.
                            dropped += 1
                if i % 2000 == 0:
                    print(f"[nav] {i}/{len(clips)} clips  kept {len(out)}  "
                          f"failed {failed}  unclassified {dropped}", flush=True)
    else:
        clips = [l.strip() for l in open(a.clip_list) if l.strip()]
        if a.limit:
            clips = clips[: a.limit]
        print(f"[nav] {len(clips)} clips, t0_relative={a.t0_us}, "
              f"horizon_start={a.horizon_start}", flush=True)
        jobs = [(c, a.local_dir, a.chunk_ids, a.t0_us, a.n_future, a.time_step,
                 a.horizon_start) for c in clips]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(_one, j) for j in jobs]), 1):
                cid, nav, yaw, y_end, rev = fut.result()
                text = record(nav, yaw, y_end, rev)
                if text is not None:
                    out.append({"clip_id": cid, "t0_relative": a.t0_us, "nav_text": text})
                else:
                    failed += 1
                if i % 2000 == 0:
                    print(f"[nav] {i}/{len(clips)}  kept {len(out)}  failed {failed}",
                          flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"\n[nav] wrote {len(out)} entries to {a.out}  "
          f"(clip failures {failed}, unclassified {dropped}, "
          f"reversing {reversed_out} -> {a.reverse_policy})")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"[nav]   {k:<16} {v:>7}  {100 * v / max(len(out), 1):.1f}%")
    turns = sum(v for k, v in counts.items() if k.startswith("Turn"))
    print(f"[nav]   {'TURN total':<16} {turns:>7}  {100 * turns / max(len(out), 1):.1f}%")
    if checked:
        print(f"[nav] y-sign convention check: sign(y_end) == sign(yaw) on "
              f"{100 * agree / checked:.1f}% of {checked} strongly-turning anchors "
              f"(expect ~100%; ~0% means left/right are SWAPPED)")


if __name__ == "__main__":
    main()
