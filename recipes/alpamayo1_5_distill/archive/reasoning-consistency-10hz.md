# Frame-to-frame reasoning consistency of Alpamayo-1.5-10B at 10 Hz
2026-07-30 · open-loop replay, 5 LCDrive val clips × 120 frames × 2 decoding arms · 1200 rollouts, 0 errors

Reproduce:
```bash
python -m alpamayo1_5_distill.scripts.rollout_frame_sweep \
    --n_clips 5 --max_frames 121 --temperature 0.6  --top_p 0.98 --out sweep_t06.jsonl
python -m alpamayo1_5_distill.scripts.rollout_frame_sweep \
    --n_clips 5 --max_frames 121 --temperature 0.01 --top_p 1.0  --out sweep_t001.jsonl
python -m alpamayo1_5_distill.scripts.score_reasoning_consistency sweep_t06.jsonl
```

## Setup

`t0` stepped in 100 ms increments over `[1.7 s, 13.6 s]` — the valid window given 1.6 s of history
and a 6.4 s future horizon inside a 20 s clip. Clips stratified across LCDrive val scenario
categories (Cut-In, Intersection Navigation, Lane Change, …), skipping General Driving.

**Two arms, and the contrast is the point.** `temp 0.6` is the deployment sampler; `temp 0.01` is
near-greedy, so any change there is attributable to the *scene advancing* rather than the sampler.

⚠️ **This has to be the released 10B.** The LCDrive-trained checkpoints use
`vla_processor/default.yaml`, which has no `cot` in `components_order` — they were never supervised
on reasoning and emit an empty span. The distilled 2B cannot be scored on reasoning consistency
until it is trained with CoT.

## Results

| metric (over 595 consecutive pairs) | deployment `t=0.6` | near-greedy `t=0.01` |
|---|---|---|
| **conflicting-intent rate (headline)** | **0.34 %** (2) | **0.00 %** (0) |
|  · lateral left↔right | 0.00 % | 0.00 % |
|  · longitudinal accel↔decel | 0.34 % | 0.00 % |
| *context:* lateral merely-differs | 13.61 % | 3.19 % |
| *context:* longitudinal merely-differs | 18.66 % | 2.69 % |
| **critical-object flip, raw** | **19.89 %** (112/563) | **4.36 %** (25/573) |
| **critical-object flip, genuine** ¹ | **12.61 %** | **1.22 %** |
| traj Δ, object flipped | 0.219 m (n=112) | 0.188 m (n=25) |
| traj Δ, object stable | 0.183 m (n=451) | 0.175 m (n=548) |
| **ratio (divergence)** | **1.20×** | **1.07×** |

¹ excluding `lead_vehicle ↔ vehicle`, which is the same object described two ways — a granularity
artifact of the lexicon, and 36.6 % / 72.0 % of raw flips respectively.

## Three findings

**1. The reasoning is volatile but almost never self-contradictory.** Distinguishing *conflict* from
*difference*, as VLADriveBench does, is what makes this visible: intent changes on 14–19 % of frames
at deployment temperature, yet outright contradictions occur on 0.34 %. Both conflicts were red→green
traffic-light transitions, which may be **correct** responses to a real scene change — so treat
0.34 % as an upper bound.

**2. ~90 % of the flicker is the sampler, not the model.** Genuine attribution flips fall
**12.61 % → 1.22 %** and conflicts fall to exactly zero when temperature drops. At 10 Hz the scene
barely changes in 100 ms, so a well-behaved model *should* be stable — the deployment sampler is
injecting instability the scene does not warrant. **This makes frame-to-frame flicker a decoding-policy
problem first**, and a far cheaper fix than any training change.

**3. …but the reasoning↔trajectory decoupling is *not* a sampler artifact, and that is the real
concern.** When the model switches which agent it calls critical, the trajectory moves only
**1.20×** the jitter floor at deployment — and **1.07×** near-greedy, i.e. essentially not at all.
The planner is largely ignoring the attribution change in both arms. Lowering temperature suppresses
the *visible* flicker without coupling the reasoning to the action. This is exactly the
epiphenomenality VLADriveBench reported for ORION, and it is a direct empirical argument for
`L_causal` — the metric that would catch it is §7(d) cross-splice, whose instrument is already
validated (`reasoning-setup-2b.md` §9).

## Limitations

- **Lateral coverage is weak**: 333–348 / 600 frames map to `lat=none`, so the 0.00 % lateral
  conflict rate is partly "could not tell". Longitudinal (36–45 unmapped) and agent (22–25) are solid.
- **Lexicon, not LLM labelers.** VLADriveBench used LLM labelers; this uses a transparent
  regex+negation mapper. The main failure mode is subject attribution — *"since it is slowing ahead"*
  describes the lead vehicle, not the ego — which is how a false conflict could be manufactured.
- **No ground truth for metric 2.** There are **no critical-object annotations** in PAI or the repo.
  The `coc` field in `ood_reasoning.parquet` is a free-text action rationale
  (*"Slow down for the one-way traffic control…"*), not an object label. We measure whether the
  model's *own* named agent is stable, which cannot be validated against truth.
- **5 clips.** A first read, not per-scenario claims.
- The 1.07× near-greedy divergence ratio rests on n=25.

## Bug found and fixed

[`src/alpamayo/data/pai_utils.py`](../../src/alpamayo/data/pai_utils.py) read `ev.get("cot")` while
the reasoning parquet's key is **`coc`**, so **all 1740 clips silently yielded empty reasoning text**
and any config with `cot` in `components_order` embedded an empty CoT. Fixed to prefer `coc` with a
`cot` fallback.
