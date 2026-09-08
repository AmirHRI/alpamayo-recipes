#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
r"""Plot the nav_text / meta_action composition of a nav anchor manifest.

    python scripts/plot_manifest_nav_dist.py \
        --manifest /temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_50k_turnpreserved.json \
        --compare  /temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/nav_lcdrive_train_anchors_50k.json \
        --out training/viz_manifest

⚠️ `nav_text` is NOT a categorical field. It carries a distance -- "Turn right in 9m" -- so the
raw value has 74 levels of which 72 are the same two manoeuvres at different ranges, each under
0.5%. Counting raw strings produces a chart that looks like 74 rare classes and hides that the
split is really 91/4.7/4.5. The distance is stripped into its own panel instead.
"""

import argparse
import collections
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: "Turn left in 12m" -> ("Turn left", 12)
DIST_RE = re.compile(r"^(.*?) in (\d+)m$")

ORDER = ["Continue straight", "Turn left", "Turn right", "Reverse"]
COLOURS = {"Continue straight": "#90a4ae", "Turn left": "#1e88e5",
           "Turn right": "#d81b60", "Reverse": "#6a1b9a"}


def load(path):
    with open(path) as fh:
        rows = json.load(fh)
    manoeuvre, dists = [], []
    for r in rows:
        m = DIST_RE.match(r["nav_text"])
        manoeuvre.append(m.group(1) if m else r["nav_text"])
        if m:
            dists.append((m.group(1), int(m.group(2))))
    return rows, manoeuvre, dists


def plot_straight_meta(rows, manoeuvre, name, out):
    """meta_action composition of the 'Continue straight' anchors.

    'Continue straight' is a ROUTING instruction, not a description of the motion: it only says
    the route does not branch. The meta_action labels underneath it still contain braking,
    stopping and lane-level steering, so this panel is what the 90.7% majority actually
    asks the model to do.
    """
    straight = [r for r, m in zip(rows, manoeuvre) if m == "Continue straight"]
    other = [r for r, m in zip(rows, manoeuvre) if m != "Continue straight"]
    ns, no = len(straight), len(other)
    if not ns:
        return

    cs = collections.Counter(a for r in straight for a in r.get("meta_action", []))
    co = collections.Counter(a for r in other for a in r.get("meta_action", []))
    keys = [k for k, _ in cs.most_common()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(f"{name}\nmeta_action within nav_text == 'Continue straight' "
                 f"({ns:,} of {len(rows):,} anchors, {100 * ns / len(rows):.1f}%)", fontsize=13)

    # -- absolute composition
    ax = axes[0]
    ordered = keys[::-1]
    vals = [cs[k] for k in ordered]
    ax.barh(ordered, vals, color="#00897b")
    for i, v in enumerate(vals):
        ax.text(v * 1.01, i, f" {v:,} ({100 * v / ns:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel(f"anchors carrying the label (of {ns:,}; labels overlap)")
    ax.set_title("multi-label -- does not sum to 100%", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    # -- straight vs turn, as share WITHIN each subset. Absolute counts would be useless here:
    # the straight subset is ~10x larger, so every bar would be taller by construction.
    ax = axes[1]
    y = np.arange(len(ordered))
    sh_s = [100 * cs[k] / ns for k in ordered]
    sh_o = [100 * co[k] / no for k in ordered] if no else [0] * len(ordered)
    ax.barh(y + 0.2, sh_s, height=0.4, color="#90a4ae", label=f"Continue straight (n={ns:,})")
    ax.barh(y - 0.2, sh_o, height=0.4, color="#d81b60", label=f"Turn / Reverse (n={no:,})")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered)
    ax.set_xlabel("share of that subset (%)")
    ax.set_title("same label, share within each nav_text group", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out, "manifest_straight_meta_action.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {path}")

    nlab = collections.Counter(len(r.get("meta_action", [])) for r in straight)
    print(f"[viz] 'Continue straight': {ns:,} anchors, "
          f"{sum(cs.values()) / ns:.2f} labels/anchor, "
          f"labels-per-anchor {dict(sorted(nlab.items()))}")
    for k in keys:
        s, o = 100 * cs[k] / ns, (100 * co[k] / no if no else 0.0)
        print(f"[viz]   {k:22s} straight {cs[k]:6,} ({s:5.2f}%)   turn {co[k]:5,} ({o:5.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--compare", default=None, help="optional second manifest for the turn-rate bar")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, manoeuvre, dists = load(args.manifest)
    n = len(rows)
    counts = collections.Counter(manoeuvre)
    keys = [k for k in ORDER if k in counts] + sorted(set(counts) - set(ORDER))

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.22)
    name = os.path.basename(args.manifest)
    clips = len({r["clip_id"] for r in rows})
    fig.suptitle(f"{name}\n{n:,} anchors over {clips:,} clips "
                 f"({n / clips:.1f} anchors per clip)", fontsize=13)

    # -- manoeuvre mix. LOG X: 'Continue straight' is 90.7% and 'Reverse' is 0.04%, a 2000x
    # span that a linear axis renders as one bar and three invisible slivers.
    ax = fig.add_subplot(gs[0, :])
    vals = [counts[k] for k in keys]
    bars = ax.barh(keys[::-1], vals[::-1], color=[COLOURS.get(k, "#777") for k in keys[::-1]])
    ax.set_xscale("log")
    ax.set_xlim(1, max(vals) * 3)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:,}  ({100 * v / n:.2f}%)",
                va="center", fontsize=10)
    ax.set_xlabel("anchors (log scale)")
    ax.set_title("nav_text manoeuvre, distance stripped", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    # -- turn distance
    ax = fig.add_subplot(gs[1, 0])
    for lab in ("Turn left", "Turn right"):
        d = [v for k, v in dists if k == lab]
        if d:
            ax.hist(d, bins=range(min(d), max(d) + 2), alpha=0.6,
                    color=COLOURS[lab], label=f"{lab}  (n={len(d):,})")
    ax.set_xlabel("distance to the turn (m)")
    ax.set_ylabel("anchors")
    ax.set_title("how far ahead the turn is announced", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # -- meta_action. ⚠️ A LIST per anchor, so these sum to more than n and are NOT a
    # partition; plotted as "share of anchors carrying the label", not as a pie.
    ax = fig.add_subplot(gs[1, 1])
    ma = collections.Counter(a for r in rows for a in r.get("meta_action", []))
    top = ma.most_common(12)[::-1]
    ax.barh([k for k, _ in top], [v for _, v in top], color="#00897b")
    for i, (_, v) in enumerate(top):
        ax.text(v * 1.01, i, f" {100 * v / n:.1f}%", va="center", fontsize=8)
    ax.set_xlabel(f"anchors carrying the label (of {n:,}; labels overlap)")
    ax.set_title("meta_action (multi-label, does not sum to 100%)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(args.out, "manifest_nav_distribution.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {path}")

    turns = sum(v for k, v in counts.items() if k.startswith("Turn"))
    print(f"[viz] {name}: {n:,} anchors, turns {turns:,} ({100 * turns / n:.2f}%)")

    plot_straight_meta(rows, manoeuvre, name, args.out)

    if args.compare:
        rows2, man2, _ = load(args.compare)
        c2 = collections.Counter(man2)
        t2 = sum(v for k, v in c2.items() if k.startswith("Turn"))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labs = [os.path.basename(args.compare), name]
        rates = [100 * t2 / len(rows2), 100 * turns / n]
        b = ax.barh(labs, rates, color=["#90a4ae", "#d81b60"])
        for bb, r_, t_, tot in zip(b, rates, [t2, turns], [len(rows2), n]):
            ax.text(r_ + 0.1, bb.get_y() + bb.get_height() / 2,
                    f"{r_:.2f}%  ({t_:,}/{tot:,})", va="center", fontsize=10)
        ax.set_xlim(0, max(rates) * 1.35)
        ax.set_xlabel("turn anchors (%)")
        ax.set_title("what 'turn-preserved' bought", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        p2 = os.path.join(args.out, "manifest_turn_rate.png")
        fig.savefig(p2, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] -> {p2}")
        print(f"[viz] turn rate {rates[0]:.2f}% -> {rates[1]:.2f}% "
              f"({rates[1] / rates[0]:.2f}x)")


if __name__ == "__main__":
    main()
