# Alpamayo 1.5 Distillation

Train a **4B or 2B VLM paired with Alpamayo's full 36-layer action expert**.
The final pipeline uses two front cameras, navigation instructions, no generated
chain-of-thought (CoT), and a one-step trajectory head. The size labels refer to
the VLM, not the combined model.

## Final Approach

```text
Alpamayo-1.5-10B teacher
  -> 1. KD: teach the smaller VLM to condition the frozen action expert
  -> 2. Cotrain EoS: adapt the VLM text layers and expert together
  -> 3. Consistency (CD): distil the cotrained two-step sampler into one step
```

| Stage | What trains | Objective |
|---|---|---|
| KD | Student VLM; also the 2B layer mixer | Expert block/span-output matching, spans 1, 9, 18, 36 over four epochs |
| Cotrain EoS (Expert-on-Student) | All VLM text layers and the action expert | Ground-truth flow matching; no trajectory-token CE |
| Consistency / CD | Action expert and action projections; VLM frozen | Online two-step EoS teacher, EMA consistency target, one-step deployment |

The 4B uses **Qwen3-VL-4B-Instruct** with 36 text layers. The 2B uses
**Cosmos-Reason2-2B** with a learned **28-to-36 KV layer mixer**, retaining the
full expert rather than pruning it. Both use cameras `[1, 3]` (front-wide and
front-telephoto), four frames per camera, and navigation text with turn distances
removed.

## Start Here

1. **Train:** [Training guide](docs/TRAINING.md), including setup, data requirements,
   and separate 4B/2B commands for all three stages.
2. **Evaluate:** [Evaluation guide](docs/EVALUATION.md), including checkpoint loading
   and one-step evaluation commands.
3. **Compare saved metrics:** [NPZ metrics notebook](notebooks/npz_metrics_comparison.ipynb).
   It loads saved trajectories without loading models or using a GPU.
4. **Resources and methods:** [Resources and Training](docs/RESOURCES_AND_TRAINING.md),
   covering data provenance, licenses, final-run hyperparameters, and loss equations.

## Final Models

Both models completed **KD -> Cotrain EoS -> Consistency**. Their final model labels are:

| VLM | Model label |
|---|---|
| 4B | `cd_eos4b_consistency2to1_fp16_master32_2cam_nav` |
| 2B | `cd_eos_2bmixall28_consistency2to1_fp16_master32_2cam_nav` |

Use the **EMA epoch checkpoint `checkpoint-6876`**, not the output-directory root.
[Actual run directories and notebook labels](docs/EVALUATION.md#final-checkpoints)
are listed separately because their spellings differ from these model labels.

## How We Got Here

We started with last-hidden-state reasoning distillation and KAVA latent slots,
then tested token/logit KD, direct KV matching, expert block/span matching,
pruning, layer mixing, EoS adaptation and cotraining, and diffusion step reduction.
The key shift was to optimize what the **action expert reads and produces**, not
the VLM's text output. [Experiment history](docs/EXPERIMENTS.md) explains the
decisions and links to the detailed notes, including negative results and retractions.

The [original README](archive/README.md) and all previous root-level Markdown notes
are preserved unchanged in [archive/](archive/). They are historical records, not
the current launch instructions. Private Honda-internal checkpoints, caches, and
data-derived artifacts must not be made public or redistributed.