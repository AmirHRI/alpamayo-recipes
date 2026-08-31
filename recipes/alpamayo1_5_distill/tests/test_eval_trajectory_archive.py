"""Tests for exact worst-of-K ADE and generated-trajectory archives."""

import numpy as np
import torch

from alpamayo1_5_sft.evaluate_hf import (
    _add_max_ade_metrics,
    _save_trajectory_archive,
)


def test_add_max_ade_reduces_worst_candidate_then_averages_sets():
    sample_ade = torch.tensor(
        [
            [[1.0, 3.0, 2.0], [4.0, 2.0, 1.0]],
            [[0.5, 0.2, 0.4], [2.0, 5.0, 1.0]],
        ]
    )
    output = {"sample_ade": sample_ade}
    _add_max_ade_metrics(output)
    torch.testing.assert_close(output["metric/max_ade"], torch.tensor([3.5, 2.75]))


def test_step_sweep_max_ade_reduces_worst_candidate_then_averages_sets():
    from alpamayo1_5_distill.scripts.eval_step_sweep import _max_ade

    # One clip, two trajectory sets, three candidates, one waypoint. Candidate
    # distances are [1, 3, 2] and [4, 2, 1], so max-K then mean-N is 3.5.
    pred = torch.zeros(1, 2, 3, 1, 3)
    pred[0, 0, :, 0, 0] = torch.tensor([1.0, 3.0, 2.0])
    pred[0, 1, :, 0, 0] = torch.tensor([4.0, 2.0, 1.0])
    gt = torch.zeros(1, 1, 3)

    torch.testing.assert_close(_max_ade(pred, gt), torch.tensor([3.5]))


def test_trajectory_archive_preserves_float32_samples_and_metrics(tmp_path):
    pred = np.arange(2 * 1 * 6 * 4 * 3, dtype=np.float32).reshape(2, 1, 6, 4, 3)
    gt = np.zeros((2, 4, 3), dtype=np.float32)
    records = [
        {"clip_id": "a", "ade": 1.0, "min_ade": 0.5, "max_ade": 2.0},
        {"clip_id": "b", "ade": 2.0, "min_ade": 1.5, "max_ade": 3.0},
    ]
    path = tmp_path / "trajectories.npz"
    n = _save_trajectory_archive(
        str(path), ["a", "b"], [pred], [gt], records, "checkpoint-1"
    )

    assert n == 2
    with np.load(path, allow_pickle=False) as archive:
        assert archive["pred_xyz"].shape == (2, 1, 6, 4, 3)
        assert archive["pred_xyz"].dtype == np.float32
        assert archive["gt_xyz"].dtype == np.float32
        assert archive["clip_ids"].tolist() == ["a", "b"]
        np.testing.assert_allclose(archive["max_ade"], [2.0, 3.0])
        assert archive["checkpoint"].item() == "checkpoint-1"
