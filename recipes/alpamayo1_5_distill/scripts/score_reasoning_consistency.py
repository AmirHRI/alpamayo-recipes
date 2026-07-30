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

"""Score frame-to-frame reasoning consistency from a 10 Hz sweep.

Three metrics over consecutive frame pairs within a clip:

1. CONFLICTING-INTENT RATE (headline). Map each frame's CoT to a lateral intent
   {left, keep, right} and a longitudinal intent {accel, maintain, decel}, then
   count pairs that CONFLICT (left<->right, accel<->decel) -- not pairs that
   merely differ (left<->keep is a refinement, not a contradiction).

2. CRITICAL-OBJECT FLIP RATE. Does the CoT name the same critical agent at t and
   t+1? "pedestrian on the right" -> "vehicle ahead" with no scene change is
   attribution flicker.

3. REASONING-VS-TRAJECTORY DIVERGENCE. Does the trajectory stay smooth while the
   reasoning flips? Compares trajectory delta on flip vs non-flip pairs.

NOTE on the intent mapping: VLADriveBench used LLM labelers. This is a
transparent lexicon+negation mapper instead -- reproducible and auditable, but
it will mislabel unusual phrasings. `--dump_unmapped` prints CoTs that fell
through to `none` so the lexicon can be audited against real outputs.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- lexicons
LAT = {
    "left":  [r"\bturn(?:ing)? left\b", r"\bsteer(?:ing)? left\b", r"\bmerg\w* left\b",
              r"\bbear(?:ing)? left\b", r"\bveer\w* left\b", r"\bleft[- ]turn\b",
              r"\bmove\w* (?:in)?to the left\b", r"\bchange lanes? to the left\b",
              r"\bleft lane\b", r"\bnudge left\b"],
    "right": [r"\bturn(?:ing)? right\b", r"\bsteer(?:ing)? right\b", r"\bmerg\w* right\b",
              r"\bbear(?:ing)? right\b", r"\bveer\w* right\b", r"\bright[- ]turn\b",
              r"\bmove\w* (?:in)?to the right\b", r"\bchange lanes? to the right\b",
              r"\bright lane\b", r"\bnudge right\b"],
    "keep":  [r"\bkeep\w* (?:the )?lane\b", r"\bmaintain\w* (?:the )?lane\b",
              r"\bstay\w* in (?:the|our|its) lane\b", r"\bcontinue\w* straight\b",
              r"\bgo(?:ing)? straight\b", r"\bproceed\w* straight\b", r"\blane[- ]keep\w*\b",
              r"\bfollow\w* the (?:lane|road|curve)\b",
              # weaker, but common in Alpamayo's phrasing ("... directly ahead in our lane"):
              # asserts the ego stays in-lane without naming a manoeuvre.
              r"\bin (?:our|the|its|this) lane\b", r"\bahead of us\b", r"\bsame lane\b"],
}
LON = {
    "decel": [r"\bslow\w*\b", r"\bbrak\w+\b", r"\bstop\w*\b", r"\byield\w*\b",
              r"\bdecelerat\w+\b", r"\breduc\w+ speed\b", r"\bcome to a (?:complete )?stop\b",
              r"\bhalt\w*\b", r"\bwait\w*\b", r"\bease off\b"],
    "accel": [r"\baccelerat\w+\b", r"\bspeed(?:ing)? up\b", r"\bincreas\w+ speed\b",
              r"\bresum\w+ speed\b", r"\bproceed\w*\b", r"\bcontinu\w+ forward\b",
              r"\bpull\w* (?:away|forward)\b"],
    "maintain": [r"\bmaintain\w* (?:a |the |current )?speed\b", r"\bkeep\w* (?:a |the )?speed\b",
                 r"\bsteady speed\b", r"\bconstant speed\b", r"\bcruis\w+\b",
                 r"\bkeep\w* (?:a |the )?(?:safe )?distance\b"],
}
# named critical agents (order matters: more specific first)
AGENTS = [
    ("traffic_light", [r"\btraffic (?:light|signal)\b", r"\bred light\b", r"\bgreen light\b",
                       r"\byellow light\b", r"\bsignal\b"]),
    ("stop_sign",     [r"\bstop sign\b"]),
    ("pedestrian",    [r"\bpedestrian\w*\b", r"\bperson\b", r"\bpeople\b", r"\bwalker\b"]),
    ("cyclist",       [r"\bcyclist\w*\b", r"\bbicycl\w+\b", r"\bbike\w*\b", r"\bmotorcycl\w+\b"]),
    ("lead_vehicle",  [r"\blead\w* vehicle\b", r"\bvehicle ahead\b", r"\bcar ahead\b",
                       r"\bvehicle in front\b", r"\bcar in front\b"]),
    ("work_zone",     [r"\bconstruction\b", r"\bwork zone\b", r"\bcone\w*\b", r"\bbarrier\w*\b",
                       r"\broadwork\b"]),
    ("bus_truck",     [r"\bbus\b", r"\btruck\b", r"\btrailer\b"]),
    ("vehicle",       [r"\bvehicle\w*\b", r"\bcar\w*\b", r"\btraffic\b"]),
    ("crosswalk",     [r"\bcrosswalk\b", r"\bcrossing\b", r"\bintersection\b"]),
]
NEG = re.compile(r"\b(?:no|not|without|never|clear of|absent|nothing)\b")


def _hit(text: str, pats: list[str]) -> bool:
    for p in pats:
        m = re.search(p, text)
        if m:
            # crude negation guard: ignore a hit negated within the preceding ~30 chars
            if NEG.search(text[max(0, m.start() - 30): m.start()]):
                continue
            return True
    return False


def classify(text: str) -> tuple[str, str, str]:
    t = (text or "").lower()
    lat = next((k for k in ("left", "right", "keep") if _hit(t, LAT[k])), "none")
    lon = next((k for k in ("decel", "accel", "maintain") if _hit(t, LON[k])), "none")
    agent = next((name for name, pats in AGENTS if _hit(t, pats)), "none")
    return lat, lon, agent


LAT_CONFLICT = {frozenset(("left", "right"))}
LON_CONFLICT = {frozenset(("accel", "decel"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=str)
    ap.add_argument("--dump_unmapped", type=int, default=0)
    ap.add_argument("--dump_conflicts", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    good = [r for r in rows if "error" not in r]
    errs = len(rows) - len(good)
    by_clip: dict[str, list[dict]] = {}
    for r in good:
        by_clip.setdefault(r["clip_id"], []).append(r)
    for v in by_clip.values():
        v.sort(key=lambda r: r["t0_us"])

    for r in good:
        r["lat"], r["lon"], r["agent"] = classify(r.get("cot", ""))

    n_pairs = 0
    lat_conf = lon_conf = any_conf = 0
    lat_diff = lon_diff = 0
    agent_flip = agent_pairs = 0
    traj_d_flip: list[float] = []
    traj_d_same: list[float] = []
    traj_d_objflip: list[float] = []
    traj_d_objsame: list[float] = []
    flip_pairs: Counter = Counter()
    examples: list[str] = []
    empty_cot = sum(1 for r in good if not (r.get("cot") or "").strip())

    for clip, seq in by_clip.items():
        for a, b in zip(seq, seq[1:]):
            if b["t0_us"] - a["t0_us"] != 100_000:
                continue                       # only true consecutive 10 Hz pairs
            n_pairs += 1
            lc = frozenset((a["lat"], b["lat"])) in LAT_CONFLICT
            oc = frozenset((a["lon"], b["lon"])) in LON_CONFLICT
            lat_conf += lc
            lon_conf += oc
            conflict = lc or oc
            any_conf += conflict
            lat_diff += a["lat"] != b["lat"]
            lon_diff += a["lon"] != b["lon"]
            obj_flip = None
            if a["agent"] != "none" and b["agent"] != "none":
                agent_pairs += 1
                obj_flip = a["agent"] != b["agent"]
                agent_flip += obj_flip
                if obj_flip:
                    flip_pairs[tuple(sorted((a["agent"], b["agent"])))] += 1
            if "traj_xyz" in a and "traj_xyz" in b:
                ta, tb = np.array(a["traj_xyz"]), np.array(b["traj_xyz"])
                n = min(len(ta), len(tb))
                d = float(np.linalg.norm(ta[:n] - tb[:n], axis=-1).mean())
                (traj_d_flip if conflict else traj_d_same).append(d)
                if obj_flip is not None:
                    (traj_d_objflip if obj_flip else traj_d_objsame).append(d)
            if conflict and len(examples) < args.dump_conflicts:
                examples.append(
                    f"    {clip[:8]} t={a['t0_us']/1e6:.1f}->{b['t0_us']/1e6:.1f}s  "
                    f"[{a['lat']}/{a['lon']}] -> [{b['lat']}/{b['lon']}]\n"
                    f"      A: {a['cot'][:120]}\n      B: {b['cot'][:120]}")

    pct = lambda x, d: (100.0 * x / d) if d else float("nan")
    print(f"frames: {len(good)} ok, {errs} errors, {len(by_clip)} clips, "
          f"{n_pairs} consecutive pairs;  empty CoT: {empty_cot}")
    print(f"  intent coverage: lat none={sum(1 for r in good if r['lat']=='none')}/{len(good)}, "
          f"lon none={sum(1 for r in good if r['lon']=='none')}/{len(good)}, "
          f"agent none={sum(1 for r in good if r['agent']=='none')}/{len(good)}")
    print("\n1) CONFLICTING-INTENT RATE  (conflict = left<->right / accel<->decel)")
    print(f"   lateral   conflicts : {lat_conf:5d} / {n_pairs}  = {pct(lat_conf,n_pairs):.2f}%")
    print(f"   longitud. conflicts : {lon_conf:5d} / {n_pairs}  = {pct(lon_conf,n_pairs):.2f}%")
    print(f"   EITHER (headline)   : {any_conf:5d} / {n_pairs}  = {pct(any_conf,n_pairs):.2f}%")
    print(f"   [context] merely-differs: lateral {pct(lat_diff,n_pairs):.2f}%, "
          f"longitudinal {pct(lon_diff,n_pairs):.2f}%")
    print("\n2) CRITICAL-OBJECT FLIP RATE  (both frames name an agent)")
    print(f"   flips: {agent_flip} / {agent_pairs} = {pct(agent_flip,agent_pairs):.2f}%")
    if flip_pairs:
        print("   most common flips (audit these -- some may be lexicon granularity,")
        print("   e.g. lead_vehicle<->vehicle is the SAME object described differently):")
        for (x, y), n in flip_pairs.most_common(8):
            print(f"     {n:4d}  {x} <-> {y}")
        coarse = sum(n for (x, y), n in flip_pairs.items()
                     if {x, y} <= {"vehicle", "lead_vehicle", "bus_truck"})
        print(f"   of which vehicle-family relabels: {coarse} "
              f"({pct(coarse, agent_flip):.1f}% of flips) -> genuine flips "
              f"{pct(agent_flip - coarse, agent_pairs):.2f}% of pairs")
    print("\n3) REASONING-VS-TRAJECTORY DIVERGENCE  (mean L2 over first 20 waypoints)")
    if traj_d_same:
        s = float(np.mean(traj_d_same))
        print(f"   traj delta on stable   pairs : {s:.3f} m  (n={len(traj_d_same)})")
    if traj_d_flip:
        f = float(np.mean(traj_d_flip))
        print(f"   traj delta on CONFLICT pairs : {f:.3f} m  (n={len(traj_d_flip)})")
    if traj_d_flip and traj_d_same:
        f, s = float(np.mean(traj_d_flip)), float(np.mean(traj_d_same))
        print(f"   ratio conflict/stable        : {f/max(s,1e-9):.2f}x")
        print("   -> ratio ~1 means the trajectory stayed smooth while the reasoning flipped,")
        print("      i.e. the reasoning is decoupled from the action (the concerning case).")
    elif not traj_d_flip:
        print("   (no conflicting pairs, so no divergence comparison is possible --")
        print("    the stable-pair delta above is the frame-to-frame trajectory jitter floor.)")
    if len(traj_d_flip) < 10:
        print(f"   !! only {len(traj_d_flip)} conflict pairs -- the ratio above is NOT significant.")
    # the object-flip split usually has enough n to be meaningful
    if traj_d_objflip and traj_d_objsame:
        f2, s2 = float(np.mean(traj_d_objflip)), float(np.mean(traj_d_objsame))
        print(f"\n   [better-powered split] conditioned on CRITICAL-OBJECT flip:")
        print(f"     traj delta, object flipped : {f2:.3f} m  (n={len(traj_d_objflip)})")
        print(f"     traj delta, object stable  : {s2:.3f} m  (n={len(traj_d_objsame)})")
        print(f"     ratio                       : {f2/max(s2,1e-9):.2f}x")
        print("     -> ~1.0 means the planner ignored the attribution change entirely.")
    if examples:
        print("\n   conflict examples:")
        print("\n".join(examples))
    if args.dump_unmapped:
        un = [r["cot"] for r in good if r["lat"] == "none" and r["lon"] == "none"]
        print(f"\n   unmapped CoTs ({len(un)}), first {args.dump_unmapped}:")
        for c in un[: args.dump_unmapped]:
            print(f"     - {c[:150]}")
        print("   agent distribution:", Counter(r["agent"] for r in good).most_common())


if __name__ == "__main__":
    main()
