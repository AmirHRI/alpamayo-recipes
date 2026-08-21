# Navigation-text annotations: choosing `t0`, and what the labels are worth

Alpamayo 1.5 takes navigation intent as **text** wrapped in `<|route_start|>…<|route_end|>`
(`chat_template/components.py::construct_route`), and the `route` component returns an empty
list when `nav_text` is absent — which is how one model runs both with and without navigation.
The component and its special tokens already existed; the missing piece was a *producer* for
`nav_text` on PAI. This documents that producer, and the sampling decisions around it.

The classifier itself (`route_to_nav_text`) is **verbatim from alpasim**, so the wording and
thresholds — the wording a model would be trained on — are unchanged. Everything below is about
*where in the clip* to ask it, and what its answers are actually worth.

---

## 1. The problem: the default keyframe is 95% straight

Every arm in this tree trains and evaluates at `DEFAULT_T0_US = 5_100_000` (5.1 s). Measured on
1,500 clips, the nav text at that keyframe is:

| | turn rate |
|---|---|
| fixed `t0` = 5.1 s, train | **5.2%** ±1.1 (95% CI, n=1500) |
| fixed `t0` = 5.1 s, val | **5.4%** ±1.1 |

A conditioning signal that says "Continue straight" 95% of the time teaches the model to ignore
it. The fix is to stop sampling at a fixed clock time and instead anchor `t0` on **labelled
behaviour events** from the CoC autolabeler
(`/data/achahe/alpamayo-coc-autolabeler/experiments/lcdrive/keyframes_{train,val}/`).

`PAIDatasetWithNav` makes this free: it reads `entry["t0_relative"]` per sample, so `t0` can be
chosen **per anchor** rather than fixed globally.

---

## 2. ⚠️ The autolabeler's frames are 10 Hz, not 30 Hz

`event_start_frame` is a **10 Hz** index, so `t0_relative = event_start_frame × 100_000` µs.
This is the single most load-bearing fact here — a 30 Hz reading would shift every `t0` by
~2/3 — so it was confirmed three independent ways:

1. **Stop-event calibration.** For `stop` segments, `event_start_frame ÷ (time the ego actually
   reaches zero speed)` gives 10.1, 10.1, 10.3, 10.7 Hz on the clips where a speed threshold
   cleanly finds the stop.
2. **Clip length.** Clips are **605 camera frames at 30.0 fps = 20.1 s** (identical across
   clips; the egomotion feature spans 20–140 s and varies, so it is *not* the clock here). The
   labeler's frames run 20–200, and only at 10 Hz does that map onto the whole clip
   (2.0–20.0 s). At 20 Hz it would cover half the clip, at 30 Hz a third.
3. **The valid-`t0` window.** At 10 Hz every labelled `event_start_frame` lands inside the
   window the loader permits (below) — 113,671 of 113,671, none clipped. The labeler
   independently enforced the same constraint, which only lines up at 10 Hz.

## 3. The valid `t0` window

`load_physical_aiavdataset` needs history *behind* `t0` and future *ahead* of it:

* `num_history_steps=16` × 0.1 s ⇒ `t0` > 1.6 s
* `num_future_steps=64` × 0.1 s ⇒ `t0` + 6.4 s ≤ 20.1 s ⇒ `t0` ≤ 13.7 s

Labelled `event_start_frame` runs **20 → 135 = 2.0 → 13.5 s**, entirely inside it. Nothing is
dropped for the window. `event_end_frame` does reach frame 200 (20.0 s), so events that *end*
late begin inside the window while their manoeuvre extends past the 6.4 s prediction horizon —
that affects what the instruction should say, not whether `t0` is valid.

---

## 4. Anchor inventory

| | train | val |
|---|---|---|
| segments in the file | 113,671 | 68,965 |
| duplicates (same clip *and* `t0`, different `meta_action`) | 3,674 | 2,224 |
| **distinct `(clip, t0)` anchors** | **109,997** | **66,741** |
| clips | 32,022 | 19,489 |
| anchors per clip | 3.44 (max 13) | 3.42 |

Train and val share **zero clips**.

Train's 32,022 clips all lie inside the 39,072-clip LCDrive train filter. The two
`segments_relative_timestamp_{all,sampled}.json` files are **byte-identical** in both splits.

## 5. What each `meta_action` is worth

Turn rate = share of that category's anchors whose `nav_text` is `Turn left/right`, i.e.
agreement between the autolabeler's `meta_action` and the independent geometric classifier.
Measured at n=1000 per category per split; **every category agrees across splits**:

| meta_action | train turn% | val turn% |
|---|---|---|
| `sharp_steer_right` | 92.1 ±1.7 | 91.8 ±2.1 |
| `sharp_steer_left` | 90.7 ±1.8 | 90.3 ±2.3 |
| `strong_acceleration` | 16.2 ±2.3 | 15.6 ±2.2 |
| `strong_deceleration` | 13.2 ±2.1 | 13.7 ±2.1 |
| `steer_left` | 11.1 ±1.9 | 12.4 ±2.0 |
| `steer_right` | 11.4 ±2.0 | 11.5 ±2.0 |
| `gentle_acceleration` | 9.0 ±1.8 | 9.0 ±1.8 |
| `maintain_speed` | 6.7 ±1.5 | 6.2 ±1.5 |
| `gentle_deceleration` | 4.3 ±1.3 | 4.4 ±1.3 |
| `go_straight` | 3.8 ±1.2 | 3.6 ±1.2 |
| `stop` | 1.4 ±0.7 | 1.4 ±0.7 |

⚠️ **n=200 per category is not enough to rank the middle of this table.** At n=200 the CI is
±4.7 pp, and two readings inverted between splits (`gentle_deceleration` measured 1.0% on train
and 7.0% on val; the truth is 4.3/4.4%). Do not act on a per-category yield measured at n=200 —
the axis of disagreement was sample size, never train-vs-val.

**Direction agreement.** For `sharp_steer_*` the wrong-direction rate is **0.2–0.5%** (8 of
2,113 anchors) — two independent methods agree on the existence *and* the sign of the turn,
which is also an independent confirmation of the 10 Hz conversion. For `steer_*`, ~a fifth of
its turn calls are the *opposite* direction, consistent with weak curvature where the sign is
genuinely ambiguous.

**Consequence.** `sharp_steer_*` is 1.9% of anchors but ~19% of the turn instructions — 10×
over-represented in useful signal. It is the only category with a large, replicated effect, and
the right thing to over-weight. `steer_*` (24.5k anchors) yields barely above the 5.2% baseline
and is not the lever it looks like.

## 6. `meta_action` distribution (train, exact)

109,997 anchors carrying 113,671 tags — 3,674 anchors have a second tag, so shares sum to
103.3%. Val matches train **within 0.5 pp on every category**, so a sampling policy set on
train transfers to val unchanged.

| meta_action | count | % of tags | clips |
|---|---|---|---|
| `maintain_speed` | 19,414 | 17.1% | 17,018 |
| `go_straight` | 17,440 | 15.3% | 14,688 |
| `gentle_deceleration` | 16,955 | 14.9% | 15,173 |
| `gentle_acceleration` | 16,174 | 14.2% | 14,503 |
| `steer_right` | 12,630 | 11.1% | 10,538 |
| `steer_left` | 11,889 | 10.5% | 10,011 |
| `strong_deceleration` | 6,932 | 6.1% | 6,602 |
| `stop` | 5,416 | 4.8% | 5,303 |
| `strong_acceleration` | 4,676 | 4.1% | 4,422 |
| `sharp_steer_right` | 1,128 | 1.0% | 1,116 |
| `sharp_steer_left` | 985 | 0.9% | 978 |
| `reverse*` | 32 | 0.0% | 31 |

Grouped: longitudinal 44.1%, steady/straight 32.4%, mild lateral 21.6%, **sharp lateral 1.9%**.
Tags are nearly disjoint (3.3% multi-tag; `stop` never is), so treating `meta_action` as a
single categorical label for sampling is safe. `strong_acceleration` is the exception at 30%
co-tagged, mostly with a steer.

---

## 7. ⚠️ The classifier sees 40 m, the horizon is 6.4 s

`route_to_nav_text` filters waypoints by **distance** (5 m ≤ d ≤ 40 m) over a trajectory
sampled in **time** (6.4 s). How much of the horizon it inspects therefore depends on speed:

| ego speed | reach in 6.4 s | % of horizon inside 40 m | turn% (40 m cap) | turn% (uncapped) |
|---|---|---|---|---|
| 0–3 m/s | 7 m | 100% | 7.5% | 7.5% |
| 3–8 m/s | 36 m | 95% | 19.6% | 20.5% |
| 8–13 m/s | 67 m | 60% | 6.2% | 9.3% |
| 13–18 m/s | 98 m | 40% | 0.5% | 2.6% |
| 18+ m/s | 161 m | 25% | 0.0% | 0.6% |

**The cap hides 1.5 pp of turns overall (9.5% → 11.0%)**, concentrated in the 8–18 m/s band.
It is *not* a large effect at highway speed, because even uncapped that band turns only 0.6% of
the time — fast roads are genuinely straight. The default 40 m is kept: the thresholds are
alpasim's (changing them puts the wording off-distribution), and "Turn left in 120 m" is not
actionable conditioning for a 6.4 s prediction. `route_to_nav_text(..., distance_lookahead_m=)`
is a keyword if that ever needs revisiting.

This also explains `stop` at 1.4%: at 0–3 m/s the ego reaches only 7 m in the whole 6.4 s,
barely clearing `min_lookahead_m=5`, so there is almost no geometry left to classify.

## 8. ⚠️ Reversing: a real classifier bug, guarded not fixed

`route_to_nav_text` takes its reference heading from the **route's own first chord**:

```python
base = points[end] - points[start]
initial_heading = np.arctan2(base[1], base[0])
deviations = headings - initial_heading      # relative to the PATH, not the vehicle
```

That is deliberate for forward driving — it stops a constant ego/road yaw offset reading as a
turn. But it means a **straight reverse has near-zero deviation from its own backward heading
and classifies as "Continue straight"**, which is a wrong instruction.

**Detection must be geometric, not tag-based.** Two wrong ways to find these:

* *The `reverse*` tags.* Only 18 of 28 tagged anchors actually reverse, while **24 truly
  reversing anchors carry other tags** (`stop` ×10, `steer_right` ×4, `go_straight` ×4).
* *Net backward displacement in the t0 frame.* 179 anchors qualify, but ~137 of them are
  **U-turns** — driven forward throughout, ending up behind the start, and correctly labelled
  `Turn left/right` (131 of the 179 are "Turn left"; 55 are tagged `sharp_steer_left`).

The test that separates them is the sign of velocity in the car's **own instantaneous frame**:
project each step's displacement onto that step's body +x axis and count the anti-aligned share
of moving steps (`_routes_and_yaws` returns this). Result: **42 of 109,997 anchors (0.038%)**
truly reverse, 37 of which had been labelled "Continue straight".

**Policy:** `--reverse-policy label` (default) rewrites their text to `"Reverse"` and keeps the
anchor. `drop` removes it, `keep` restores the old wrong text. No direction is asserted — for a
reversing car "left" is ambiguous (nose vs rear), and the autolabeler's own
`reverse_left`/`reverse_right` tags agree with the geometry on only 18 of 28 anchors, so there
is no reliable convention to copy. `"Reverse"` is wording no model in this tree has seen
(nav_text is free text; no reverse phrasing exists anywhere here) — at 42 anchors its value is
*not mis-instructing*, not teaching the concept.

The classifier itself is left alone: changing `initial_heading` would shift labels on all 110k
anchors to fix 42.

---

## 9. Files

All under `/data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests/`.

### Train

| file | anchors | clips | turn rate |
|---|---|---|---|
| `nav_lcdrive_train_anchors_all.json` | 109,997 | 32,022 | 9.26% (10,185) |
| `nav_lcdrive_train_anchors_50k.json` | 50,000 | 32,022 (all) | 8.04% |
| `nav_lcdrive_train_anchors_50k_turnpreserved.json` | 50,000 | 32,022 (all) | **9.22%** |

**Use `_turnpreserved`.** Both 50k files hold every `meta_action` share within 0.7 pp and cover
every clip, but see §10 — the plain one is a systematically quieter dataset than its parent.

### Val — the 1k eval subset

Scoped to the 1,000 clips of `lcdrive_val_mysubset_1k_clip_uuids.txt`, which is what every
existing arm is scored on.

| file | entries | clips | turn rate |
|---|---|---|---|
| `nav_lcdrive_val_mysubset_1k.json` | 1,000 (one per clip) | 1,000 | **7.90%** (79) |
| `nav_lcdrive_val_mysubset_1k_fixedt0.json` | 1,000 | 1,000 | 4.00% (40) |
| `nav_lcdrive_val_mysubset_anchors_all.json` | 2,994 | 1,000 | 8.32% (249) |

Event-anchoring **doubles the eval turn rate, 4.00% → 7.90%** (44 clips flip straight→turn, 5
flip back), consistent with the 1.81× measured on train. 835 of 1,000 anchors moved off the
default keyframe, over 116 distinct `t0` values; left/right stay balanced (43/36). The
`_fixedt0` file is the **paired control** — same 1,000 clips at 5.1 s.

The full val file (66,741 anchors, 19,489 clips) is not generated; one command if needed.

### Schema

```json
{"clip_id": "004b7998-…", "t0_relative": 2300000,
 "nav_text": "Continue straight", "meta_action": ["steer_right"]}
```

`PAIDatasetWithNav.__getitem__` reads `t0_relative` (**not** `t0`, as its own docstring example
shows) and `nav_text`, and indexes `self._samples` **positionally** — so multiple entries per
clip are fine, each becoming its own sample. It does `entry["nav_text"]` **unguarded**, so every
entry must carry a non-empty value or training dies mid-epoch with `KeyError`. `meta_action` is
ignored by the loader and carried purely so the mix stays a training-time knob without
regenerating.

---

## 10. ⚠️ Full clip coverage fights the turn rate

Requiring ≥1 anchor per clip is not free, because **turn rate rises monotonically with a clip's
event count**:

| anchors in clip | turn rate | share of full | share of a covered 50k |
|---|---|---|---|
| 1 | **0.7%** | 5.2% | **11.5%** |
| 2 | 1.6% | 12.8% | 17.3% |
| 3 | 4.1% | 16.4% | 17.8% |
| 5 | 10.3% | 15.5% | 13.1% |
| 6+ | **17.2%** | 33.6% | **25.0%** |

A clip with many labelled manoeuvres is busy urban driving; a clip with one is quiet cruising.
Coverage up-weights the quiet clips 2.2× and **costs 1.4 pp of turn rate** (9.26% → 7.87%), and
it shows up *within* every category (`strong_acceleration` −3.5 pp, `steer_right` −2.3 pp) — so
the plain subsample is not just smaller, it is quieter.

`subsample_nav_anchors.py --preserve-turn-rate` fixes this by stratifying on
**`(meta_action, is_turn)`** jointly instead of `meta_action` alone. Result: identical
`meta_action` ratios (within 0.01 pp of the plain version), *and* per-category turn rates that
match the parent exactly (`strong_acceleration` 16.7% vs 16.7%, `steer_right` 11.1% vs 11.1%).
Cost: 660 extra turn instructions and nothing else.

**Sampler design.** Coverage runs **first**, choosing among each clip's anchors the one whose
scarcest category is furthest below quota, least-flexible clips (fewest distinct categories)
first. 5,742 train clips have exactly one anchor, so their category is *forced* — sampling each
category independently to quota would ignore coverage, and covering naively would blow a small
category's quota before the free budget is allocated. Feasibility is checked and reported, not
assumed.

## 11. ⚠️ Val at one-anchor-per-clip cannot hold the ratios

For the 1k eval file the `meta_action` deviation is **10.3 pp**, and it is structural rather
than a sampler failure:

* Only **840 of the 1,000** eval clips are labelled. The other 160 get one entry at the default
  5.1 s keyframe, tagged `meta_action: ["no_event"]` (`--fallback-default-t0`). Dropping them
  would shrink the scored set to 840 and break comparability with every existing `min_ade`.
* Those 160 are then necessarily **16%** of a 1,000-entry one-per-clip budget, against 5.2% of
  the 2,994-anchor pool. Every real category is within 1.8 pp, shifted down uniformly to make
  room.

At exactly one anchor per clip the category mix is determined by *which clips have which
events*, not by choice. If matching train's ratios matters more than one-per-clip coverage, a
1,000-anchor draw without the coverage constraint hits them closely.

## 12. ⚠️ Comparability and the label leak

* **`t0` moves, so the data distribution moves.** Every existing arm trains and evals at
  `t0_relative = 5_100_000`. A nav-conditioned run on event anchors is **not** comparable to the
  existing block/span numbers; it needs its own no-nav control on the *same* anchors.
* **The route here is the GT future.** In alpasim the route is planned and independent of what
  the ego does; on PAI the only available polyline is the trajectory the ego actually drove, so
  nav text derived this way **restates the label's direction**. Legitimate conditioning for
  training; an eval that also sees it is handed the answer's direction. Report which way each
  arm was scored. `--horizon-start N` drops the near waypoints so the instruction describes
  intent *beyond* the prediction horizon — it matters more now that turns are 2× denser.
* **No video is decoded** during generation: the route comes from `LABELS.EGOMOTION` evaluated
  at the future timestamps and rotated into the `t0` frame, the same transform the loader
  applies. Going through `__getitem__` would decode 16 frames per clip and take hours.

---

## 13. Reproducing

```bash
export PYTHONPATH=$PWD/src:$PWD/recipes
M=/data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests
K=/data/achahe/alpamayo-coc-autolabeler/experiments/lcdrive

# train: all anchors  (<1 min for 110k anchors; see the note below)
python -m alpamayo1_5_distill.scripts.gen_nav_annotations \
    --anchors $K/keyframes_train/segments_relative_timestamp_sampled.json \
    --clip-list $M/lcdrive_train_clip_uuids.txt \
    --out $M/nav_lcdrive_train_anchors_all.json --workers 28

# train: 50k, ratio- AND turn-rate-preserving, every clip covered
python -m alpamayo1_5_distill.scripts.subsample_nav_anchors --preserve-turn-rate \
    --in  $M/nav_lcdrive_train_anchors_all.json \
    --out $M/nav_lcdrive_train_anchors_50k_turnpreserved.json --target 50000

# val: the 1k eval subset, one anchor per clip, + its fixed-t0 control
python -m alpamayo1_5_distill.scripts.gen_nav_annotations \
    --anchors $K/keyframes_val/segments_relative_timestamp_sampled.json \
    --clip-list $M/lcdrive_val_mysubset_1k_clip_uuids.txt --fallback-default-t0 \
    --out $M/nav_lcdrive_val_mysubset_anchors_all.json --workers 28
python -m alpamayo1_5_distill.scripts.subsample_nav_anchors --preserve-turn-rate \
    --in  $M/nav_lcdrive_val_mysubset_anchors_all.json \
    --out $M/nav_lcdrive_val_mysubset_1k.json --target 1000
python -m alpamayo1_5_distill.scripts.gen_nav_annotations \
    --clip-list $M/lcdrive_val_mysubset_1k_clip_uuids.txt \
    --out $M/nav_lcdrive_val_mysubset_1k_fixedt0.json --workers 28

# one example per meta_action: front camera at t0 + BEV + label
python -m alpamayo1_5_distill.scripts.plot_nav_examples --fixed-span 45 \
    --annotations $M/nav_lcdrive_train_anchors_all.json \
    --out .../figures/nav_category_examples_fixedscale.png
```

**Why it is fast.** Anchors are batched **per clip** (32,022 jobs, not 109,997) and the
`PhysicalAIAVDatasetLocalInterface` is cached **per worker process**. Constructing that
interface parses the chunk metadata at `chunk_ids=0-3146` and costs far more than the egomotion
query it enables; building it per anchor made the run interface-construction-bound (~35 min →
<1 min).

Figures live outside the repo at
`/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/figures/`.

## 14. Open

* Nothing has been **trained** on these annotations yet — every number here is about the data.
* `--horizon-start` is unset (0) in all generated files, so nav text restates the label's
  direction. An eval-safe variant needs a value chosen and justified.
* The full val file (66,741 anchors) is ungenerated.
* Whether to over-weight `sharp_steer_*` at train time, and by how much, is untested; §5 says
  it is the only lever with a replicated effect, not how big the weight should be.
