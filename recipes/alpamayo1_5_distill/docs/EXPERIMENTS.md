# Experiment History and Design Decisions

[Back to the recipe](../README.md)

## From Reasoning Tokens to the Expert's Input

The initial goal was to transfer the 10B teacher's reasoning into a smaller,
text-silent model. We first matched the final hidden state at the trajectory
handoff, then tried **KAVA**: continuous slots supervised by compressed teacher
CoT keys/values. One-pass slots were not load-bearing in the measured ablation;
two-pass slots became useful relative to a matched control but introduced a large
trajectory-error tail. Neither became the final pipeline.

The next experiments compared CE, logit KD, and direct KV matching. Their ordering
reversed depending on whether evaluation used the **VLM trajectory-token head** or
the **action expert**. Cache alignment helped the expert even when the student's
token head was unusable. That established the endpoint for this work: the expert
must drive from the student's cache; generating good trajectory tokens is a
different task.

## Match What the Expert Does

Elementwise KV loss treats every cache direction equally. **Block-output matching**
instead runs the frozen expert with teacher versus student conditioning and
matches the resulting hidden states. It beat direct KV matching in the reported
stitched-head experiments. Sampling the flow timestep broadened supervision;
longer spans then exposed errors that compound across expert layers.

The final KD curriculum grows spans **1, 9, 18, 36**, one per epoch, on the full
navigation-anchor training set. This is depth within a single expert forward,
**not** 36 diffusion sampling steps. The teacher remains frozen and online;
student prefixes contain no generated CoT or ground-truth future tokens.

Depth-banded loss weights regressed despite promising post-hoc layer-importance
measurements. CKA-guided expert pruning also reduced quality. The final models
therefore keep all **36 expert layers**: native correspondence for the 4B, and
learned blockwise **28-to-36 KV mixing** for the 2B. Equal KV width alone does not
establish equal representations or solve the depth mismatch.

## Adapt, Then Reduce Sampling Steps

**EoS (Expert-on-Student)** first adapted a frozen teacher expert to the student's
cache while holding the VLM fixed. Cotraining sweeps then varied which text layers
could adapt and their learning rates. The selected pipeline trains every text
layer together with the expert at the base LR, while leaving vision, embeddings,
LM head, and the 2B mixer frozen. This uses GT flow matching, not token CE.

Step-reduction work included cached teacher-endpoint targets and GT mixtures.
Those remain documented experiments, but they are **not the final CD objective**.
The final consistency stage uses each cotrained model's **own two-step solver**
as a fixed online teacher, with an EMA consistency target and no GT/endpoint-cache
loss. Only the expert and action projections train; deployment is one step.
The selected runs use corrected fp16 AMP with fp32 optimizer weights and EMA.

## Measurement Lessons

- Match camera/frame labels to the model's training format. Earlier CoT and
  cross-generation conclusions were retracted after prompt-format errors.
- Keep cameras, event anchors, navigation-distance stripping, and the action
  head fixed across comparisons. Historical tables do not all share this setup.
- Compare paired clips and GT, and inspect typical-draw error, best-of-six error,
  center, and spread. A better best-of-six score alone can hide a worse tail.
- Reduced expert NFE is a compute saving, not a device-independent latency claim.

## Archived Reading

All original notes are preserved **byte-for-byte**. Some contain superseded
instructions, job statuses, and relative code links written for their former
location at the recipe root. Use the [current training guide](TRAINING.md) for
commands; resolve those historical code paths from the recipe root.

| Topic | Historical notes |
|---|---|
| Original narrative, latent KD, KAVA, KV/block results and retractions | [Original README](../archive/README.md) |
| Initial 2B reasoning plan | [Reasoning setup](../archive/reasoning-setup-2b.md) |
| Loss variants | [KD loss ablations](../archive/KD_LOSS_ABLATION.md) |
| Pruning and depth mapping | [Pruning](../archive/PRUNING.md), [mixing/pruning](../archive/mix_pruning.md) |
| Navigation and sampling | [Navigation sampling](../archive/NAVTEXT_SAMPLING.md) |
| EoS comparisons, cotraining and trajectory distribution | [Comparison notes](../archive/COMPARE_EVAL.md), [center gap](../archive/CENTER_GAP.md) |
| Earlier consistency and endpoint approaches | [4B CD](../archive/student4B_CD.md), [10 Hz reasoning/consistency](../archive/reasoning-consistency-10hz.md) |
| Two-step-to-one-step equations and precision investigations | [Two-step consistency](../archive/TWO_STEP_CONSISTENCY.md) |
| Device-specific timing | [Latency profile](../archive/LATENCY_PROFILE.md) |