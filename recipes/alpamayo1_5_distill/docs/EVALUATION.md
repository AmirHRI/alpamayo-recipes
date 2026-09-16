# Evaluation and Saved Metrics

[Back to the recipe](../README.md)

## Final Checkpoints

Both final models completed KD, all-text-layer EoS cotraining, and consistency.
The model labels and on-disk run names are not identical:

| VLM | Final model label | Actual run directory under `training/` |
|---|---|---|
| 4B | `cd_eos4b_consistency2to1_fp16_master32_2cam_nav` | `output_cd_eos4b_consistency2to1_fp16_master32_2cam_nav` |
| 2B | `cd_eos_2bmixall28_consistency2to1_fp16_master32_2cam_nav` | `output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav` |

Select **`checkpoint-6876`** inside these directories for the final epoch-2 EMA
weights. The root-level final save is not the EMA-swapped checkpoint-save path.
The 2B must restore its trained layer mixer along with the VLM and full expert.
Local artifacts are not included in this Git checkout; obtain authorized access
or produce them with the [training guide](TRAINING.md).

## Load Existing Metrics

Open [npz_metrics_comparison.ipynb](../notebooks/npz_metrics_comparison.ipynb)
with a Python kernel containing NumPy, pandas, Matplotlib, and IPython. No GPU or
model loading is needed.

1. In **Cell 2**, set `TRAINING_ROOT` to the directory containing the saved NPZs.
   Update `runs` when adding your own evaluation outputs.
2. Cell 2 currently filters to `run[1] <= 1`, so only **NFE=1** rows are shown.
   Remove or adjust this filter to include the recorded EoS/teacher step budgets.
   The existing final CM rows are measured only at NFE=1.
3. Run the notebook in order. It validates 1,000 unique paired clip IDs,
   identical saved GT, six draws, 64 timesteps, and finite arrays, then computes
   the comparison table and provenance checks.

The final runs appear as **CM 4B** and **CM 2B**, using these archive stems:

```text
cm4b_fp16_master32_eos_checkpoint-6876_nfe1.npz
cm2b_all28_fp16_master32_h100c_eos_checkpoint-6876_nfe1.npz
```

All notebook metrics use XY distances in metres and equal weight per clip:

| Metric | Meaning |
|---|---|
| `ADE` | Time-averaged error of draw 0, not the mean of all six draws |
| `min_ADE` | Lowest trajectory ADE among the six draws |
| `max_ADE` | Highest trajectory ADE among the six draws |
| `center` | ADE of the mean predicted XY trajectory |

Read the provenance table: navigation stripping is not verified for every
historical row. Matching clips and GT alone does not prove identical model inputs.

## Generate New Evaluation Archives

First adapt site-specific paths as described in the training guide. From the
recipe directory, evaluate both the VLM and expert from the selected checkpoint:

```bash
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
CM4="$OUT/output_cd_eos4b_consistency2to1_fp16_master32_2cam_nav"
CM2="$OUT/output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav"
unset PRUNE_EXPERT_LAYERS

MODEL=4b MIX=0 ARM=eos NFE=1 CKPT=checkpoint-6876 \
  EOS_DIR_OVERRIDE="$CM4" TAG_BASE=cm4b_recheck \
  EXTRA_ARGS='++data.val_dataset.strip_nav_turn_distance=true' \
  sbatch slurm_eval_eos.sh

MODEL=2b MIX=1 ARM=eos NFE=1 CKPT=checkpoint-6876 \
  EOS_DIR_OVERRIDE="$CM2" TAG_BASE=cm2b_recheck \
  EXTRA_ARGS='++data.val_dataset.strip_nav_turn_distance=true' \
  sbatch slurm_eval_eos.sh
```

Add `MAX_EVAL_STEPS=5` first for a loading/shape smoke. Full runs use 1,000
validation clips, six trajectories per clip, cameras `[1,3]`, and stripped nav.
Keep the launcher's **single-rank evaluation**: multi-rank per-clip outputs are
not gathered into a complete archive. Use unique `TAG_BASE` values to avoid
overwriting earlier comparisons. The script writes JSON, NPZ, and a log; for
example, the full 4B command writes `cm4b_recheck_eos_checkpoint-6876_nfe1.npz`.

For the matched reference, run the same commands with `NFE=2`, each model's
**own cotrained EoS directory** from the training guide, and a distinct tag.
Do not use `ARM=control`: that substitutes the released teacher expert instead
of the cotrained two-step reference. A CM model is specialized for one step;
running it for two or ten steps does not restore the original EoS solver.

Score these models through the expert with
[slurm_eval_eos.sh](../slurm_eval_eos.sh), **not** the token-head
[slurm_eval_kd.sh](../slurm_eval_kd.sh). Use raw NPZ trajectories for diversity
checks: the launcher's `ade == min_ade` counter only says draw 0 was best, not
that all six trajectories were identical.