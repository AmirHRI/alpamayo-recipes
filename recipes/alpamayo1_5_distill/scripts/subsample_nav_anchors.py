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

r"""Subsample a nav-annotation file, preserving meta_action ratios AND covering every clip.

Two constraints that pull against each other:

  * **ratio preservation** -- each ``meta_action`` keeps its share of the full file, so the
    subsample is distributionally the same training set, just smaller.
  * **full clip coverage** -- every clip contributes at least one anchor, so no clip's video
    is dropped from the epoch. At 32,022 clips this fixes 64% of a 50k budget, and the
    covering anchor's category is *not* free: 5,742 clips have exactly one anchor, so their
    category is forced.

Those forced picks are what makes naive stratified sampling fail: sampling each category
independently to quota ignores coverage, and picking one-per-clip first can blow a small
category's quota before the free budget is even allocated. So coverage runs FIRST, choosing
among each clip's anchors the one whose category is furthest below quota, with the least
flexible clips (fewest distinct categories) handled earliest. The remaining budget then
fills whatever deficits are left.

Feasibility is checked and reported rather than assumed: if a category's forced picks exceed
its quota the ratio cannot hold, and the script says so instead of silently skewing.

⚠️ Anchors may carry TWO meta_actions (3.3% do). Quotas are accounted per TAG, so selecting
a two-tag anchor advances both -- which is why the achieved shares can miss target by a few
tenths of a point. The deviation is printed; it is not hidden.

Usage::

    python -m alpamayo1_5_distill.scripts.subsample_nav_anchors \
        --in  .../nav_lcdrive_train_anchors_all.json \
        --out .../nav_lcdrive_train_anchors_50k.json --target 50000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cover", action="store_true",
                    help="skip the one-per-clip guarantee (pure stratified sample)")
    ap.add_argument("--preserve-turn-rate", action="store_true",
                    help="stratify on (meta_action, is_turn) jointly. Full clip coverage "
                         "otherwise DILUTES turns: turn rate rises monotonically with a "
                         "clip's event count (0.7%% at 1 event, 17.2%% at 6+), so forcing "
                         "one anchor per clip up-weights quiet clips 2.2x and drops the "
                         "turn rate ~1.4pp. This holds meta_action ratios AND turn rate.")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    with open(a.inp) as fh:
        data = json.load(fh)
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for e in data:
        by_clip[e["clip_id"]].append(e)
    def keys_of(e):
        """Stratification cells for one anchor (may be two, one per meta_action)."""
        if a.preserve_turn_rate:
            # only "Turn ..." counts as a turn; "Reverse" is neither turn nor straight
            suf = "|turn" if e["nav_text"].startswith("Turn") else "|straight"
            return [t + suf for t in e["meta_action"]]
        return list(e["meta_action"])

    tags = Counter(k for e in data for k in keys_of(e))
    scale = a.target / len(data)
    quota = {t: c * scale for t, c in tags.items()}

    print(f"[sub] {len(data)} anchors, {len(by_clip)} clips -> target {a.target} "
          f"(scale {scale:.4f})")
    if not a.no_cover and len(by_clip) > a.target:
        raise SystemExit(f"[sub] target {a.target} < {len(by_clip)} clips: coverage impossible")

    # feasibility: a clip with one anchor has no choice of category
    forced: Counter = Counter()
    for v in by_clip.values():
        if len(v) == 1:
            for t in keys_of(v[0]):
                forced[t] += 1
    infeasible = {t: (forced[t], quota[t]) for t in tags if forced[t] > quota[t]}
    if infeasible:
        print(f"[sub] WARNING ratio cannot hold; forced picks exceed quota: {infeasible}")

    got: Counter = Counter()
    chosen: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def key(e):
        return (e["clip_id"], e["t0_relative"])

    def take(e):
        chosen.append(e)
        seen.add(key(e))
        for t in keys_of(e):
            got[t] += 1

    if not a.no_cover:
        # least flexible clips first: their forced categories must be absorbed before the
        # flexible ones spend the quota those categories need
        order = sorted(by_clip, key=lambda c: (len({tuple(x["meta_action"]) for x in by_clip[c]}),
                                               len(by_clip[c]), c))
        for clip in order:
            cands = by_clip[clip][:]
            rng.shuffle(cands)
            # pick the anchor whose scarcest tag is furthest below quota
            best = max(cands, key=lambda e: min((quota[t] - got[t]) / quota[t]
                                                for t in keys_of(e)))
            take(best)
        print(f"[sub] coverage phase: {len(chosen)} anchors, one per clip")

    # fill the rest against the largest remaining deficit
    pools: dict[str, list[dict]] = {t: [] for t in tags}
    for e in data:
        if key(e) in seen:
            continue
        for t in keys_of(e):
            pools[t].append(e)
    for t in pools:
        rng.shuffle(pools[t])
    ptr = dict.fromkeys(pools, 0)

    stalled = 0
    while len(chosen) < a.target and stalled < len(tags):
        live = [t for t in tags if ptr[t] < len(pools[t])]
        if not live:
            break
        t = max(live, key=lambda t: quota[t] - got[t])
        if quota[t] - got[t] <= 0:
            # every quota met but budget remains: keep the ratio by drawing from the
            # largest category still holding anchors, rather than leaving the target short
            t = max(live, key=lambda t: len(pools[t]) - ptr[t])
        e = None
        while ptr[t] < len(pools[t]):
            cand = pools[t][ptr[t]]
            ptr[t] += 1
            if key(cand) not in seen:
                e = cand
                break
        if e is None:
            stalled += 1
            continue
        stalled = 0
        take(e)

    rng.shuffle(chosen)
    with open(a.out, "w") as fh:
        json.dump(chosen, fh)

    n = len(chosen)
    clips_out = {e["clip_id"] for e in chosen}
    print(f"\n[sub] wrote {n} anchors to {a.out}")
    print(f"[sub] clips covered {len(clips_out)}/{len(by_clip)}"
          f"{'  ALL' if len(clips_out) == len(by_clip) else '  *** MISSING ***'}")
    print(f"\n{'meta_action':<22}{'full %':>9}{'sample %':>10}{'delta':>8}"
          f"{'count':>8}{'quota':>8}")
    print("-" * 65)
    base_t = Counter(t for e in data for t in e["meta_action"])
    base_s = Counter(t for e in chosen for t in e["meta_action"])
    tags, got = base_t, base_s
    quota = {t: c * scale for t, c in base_t.items()}
    tot_t = sum(tags.values())
    tot_s = sum(got.values())
    worst = 0.0
    for t, c in tags.most_common():
        pf, ps = 100 * c / tot_t, 100 * got[t] / tot_s
        worst = max(worst, abs(pf - ps))
        print(f"{t:<22}{pf:>8.2f}%{ps:>9.2f}%{ps - pf:>+8.2f}{got[t]:>8}{quota[t]:>8.0f}")
    print("-" * 65)
    print(f"[sub] max share deviation {worst:.2f} pp")
    turn_f = sum(1 for e in data if e["nav_text"].startswith("Turn")) / len(data)
    turn_s = sum(1 for e in chosen if e["nav_text"].startswith("Turn")) / n
    print(f"[sub] turn rate: full {100 * turn_f:.2f}%  sample {100 * turn_s:.2f}%")
    per = Counter(e["clip_id"] for e in chosen)
    print(f"[sub] anchors/clip: min {min(per.values())} mean {n / len(clips_out):.2f} "
          f"max {max(per.values())}")


if __name__ == "__main__":
    main()
