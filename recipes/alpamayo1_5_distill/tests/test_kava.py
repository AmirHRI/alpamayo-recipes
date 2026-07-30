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

"""GPU-free tests for the KAVA pieces that are easy to get subtly wrong.

Following ``test_distill.py``'s approach: never build a model, exercise the pure
functions and the index arithmetic directly.  What is covered here is chosen for
*silent* failure modes — a broadcast bug in eviction, an off-by-one in the slot
splice, or a layer map that quietly drops a layer would all still train and still
produce a decreasing loss.

Run::

    cd recipes/alpamayo1_5_distill && PYTHONPATH=.. \
        ../alpamayo1_5_sft/.venv/bin/python -m pytest tests/test_kava.py -q
"""

import torch

from alpamayo1_5_distill.data import kv_cache_io
from alpamayo1_5_distill.data.kava_dataset import (
    pad_kv_to_budget,
    splice_slot_placeholders,
)
from alpamayo1_5_distill.models.kv_distill import (
    KVProjectorBank,
    build_layer_map,
    combine_scores,
    evict_teacher_cache,
    gather_selected,
    importance_score,
    kv_matching_loss,
    redundancy_score,
    select_top_m,
)

L, H, N, D = 4, 3, 12, 8


# ------------------------------------------------------------------- scoring
def test_scores_are_distributions() -> None:
    keys = torch.randn(L, H, N, D)
    red = redundancy_score(keys)
    imp = importance_score(torch.rand(L, H, N))
    assert red.shape == (L, H, N) and imp.shape == (L, H, N)
    torch.testing.assert_close(red.sum(-1), torch.ones(L, H))
    torch.testing.assert_close(imp.sum(-1), torch.ones(L, H))


def test_redundancy_prefers_the_odd_key_out() -> None:
    """A key unlike all the others must score highest — that is the whole point."""
    keys = torch.ones(1, 1, 4, D)
    keys[0, 0, 2] = -1.0  # the only one pointing the other way
    red = redundancy_score(keys)
    assert int(red[0, 0].argmax()) == 2


def test_combine_scores_honours_the_ablation_extremes() -> None:
    imp = torch.rand(L, H, N)
    red = redundancy_score(torch.randn(L, H, N, D))
    torch.testing.assert_close(combine_scores(imp, red, 0.5, "attn"), importance_score(imp))
    torch.testing.assert_close(combine_scores(imp, red, 0.5, "cosine"), red)
    mixed = combine_scores(imp, red, 0.1, "rkv")
    torch.testing.assert_close(mixed, 0.1 * importance_score(imp) + 0.9 * red)


# ------------------------------------------------------------------ selection
def test_select_top_m_is_ascending_and_per_head() -> None:
    torch.manual_seed(0)
    scores = torch.rand(L, H, N)
    idx = select_top_m(scores, 5, N)
    assert idx.shape == (L, H, 5)
    # temporal order preserved: the student's slots are causally ordered, so slot i
    # must target the i-th SURVIVING token in time, not the i-th best-scoring one.
    assert bool((idx.diff(dim=-1) > 0).all())
    # and the choice must genuinely vary per (layer, head) — an accidental broadcast
    # would give every head the same index set and look perfectly healthy.
    flat = idx.reshape(-1, 5)
    assert len({tuple(row.tolist()) for row in flat}) > 1


def test_select_top_m_picks_the_highest_scores() -> None:
    scores = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3]]])
    idx = select_top_m(scores, 2, 5)
    assert idx.flatten().tolist() == [1, 3]


def test_select_top_m_never_picks_padding() -> None:
    scores = torch.zeros(1, 1, N)
    scores[..., N - 1] = 10.0  # highest score, but past n_valid
    idx = select_top_m(scores, 3, n_valid=5)
    assert idx.max().item() < 5


def test_crop_keeps_the_first_m() -> None:
    idx = select_top_m(torch.rand(L, H, N), 4, N, method="crop")
    assert idx.shape == (L, H, 4)
    assert bool((idx == torch.arange(4)).all())


def test_short_cot_shrinks_the_selection_not_the_indices() -> None:
    """N_C < M happens on real driving traces; KAVA never hits it."""
    keys = torch.randn(L, H, N, D)
    k_sel, v_sel, idx = evict_teacher_cache(keys, keys.clone(), m=10, n_valid=3, red=None, lam=0.0)
    assert k_sel.shape == (L, H, 3, D) and idx.shape == (L, H, 3)
    assert idx.max().item() < 3


def test_gather_selected_matches_manual_indexing() -> None:
    keys = torch.randn(2, 2, N, D)
    idx = torch.tensor([[[1, 4], [0, 7]], [[2, 3], [5, 6]]])
    out = gather_selected(keys, idx)
    for i in range(2):
        for j in range(2):
            for s in range(2):
                torch.testing.assert_close(out[i, j, s], keys[i, j, idx[i, j, s]])


# ------------------------------------------------------------------ layer map
def test_layer_map_preserves_endpoints_and_covers_every_student_layer() -> None:
    mapping = build_layer_map(28, 36)
    assert len(mapping) == 28
    assert mapping[0] == 0 and mapping[-1] == 35
    assert mapping == sorted(mapping)
    assert all(0 <= t < 36 for t in mapping)


def test_layer_map_identity_when_depths_match() -> None:
    assert build_layer_map(6, 6) == list(range(6))


def test_explicit_layer_map_is_validated() -> None:
    assert build_layer_map(3, 36, explicit=[0, 17, 35]) == [0, 17, 35]
    for bad in ([0, 1], [0, 1, 36]):
        try:
            build_layer_map(3, 36, explicit=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# ------------------------------------------------------------------ projector
def test_projector_starts_as_the_identity() -> None:
    """Identity init is what makes `projector` and `direct` a clean ablation pair."""
    bank = KVProjectorBank(4, kv_width=H * D, align="projector")
    kv = torch.randn(2, H, 5, D)
    torch.testing.assert_close(bank(kv, 0, "k"), kv, atol=1e-5, rtol=1e-5)


def test_direct_align_has_no_parameters() -> None:
    bank = KVProjectorBank(4, kv_width=H * D, align="direct")
    assert list(bank.parameters()) == []
    kv = torch.randn(2, H, 5, D)
    assert bank(kv, 2, "v") is kv


def test_projector_round_trips_head_layout() -> None:
    """The flatten/unflatten around the H*D map must not scramble heads."""
    bank = KVProjectorBank(1, kv_width=H * D, align="projector")
    with torch.no_grad():  # a permutation-free but non-identity map
        bank.k_proj[0].weight.mul_(2.0)
    kv = torch.randn(2, H, 5, D)
    torch.testing.assert_close(bank(kv, 0, "k"), kv * 2.0, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------- loss
def _student_kv(n_layers: int, b: int = 2, m: int = 5) -> dict:
    return {i: (torch.randn(b, H, m, D), torch.randn(b, H, m, D)) for i in range(n_layers)}


def test_kv_loss_is_zero_on_a_perfect_match() -> None:
    b, m, n_student = 2, 5, 4
    teacher_k = torch.randn(b, L, H, m, D)
    teacher_v = torch.randn(b, L, H, m, D)
    mapping = build_layer_map(n_student, L)
    student = {i: (teacher_k[:, mapping[i]].clone(), teacher_v[:, mapping[i]].clone()) for i in range(n_student)}
    loss = kv_matching_loss(student, teacher_k, teacher_v, mapping, kind="l1")
    assert loss.item() == 0.0


def test_kv_loss_ignores_masked_slots() -> None:
    """Padded slots have no target; letting them in would train the slots to zero."""
    b, m, n_student = 2, 6, 4
    teacher_k = torch.zeros(b, L, H, m, D)
    teacher_v = torch.zeros(b, L, H, m, D)
    mapping = build_layer_map(n_student, L)
    student = {i: (torch.zeros(b, H, m, D), torch.zeros(b, H, m, D)) for i in range(n_student)}
    for k, _ in student.values():
        k[:, :, 3:] = 99.0  # garbage, but only where the mask says invalid

    valid = torch.zeros(b, m, dtype=torch.bool)
    valid[:, :3] = True
    assert kv_matching_loss(student, teacher_k, teacher_v, mapping, valid_mask=valid).item() == 0.0
    assert kv_matching_loss(student, teacher_k, teacher_v, mapping).item() > 0.0


def test_kv_loss_scale_is_independent_of_m_and_depth() -> None:
    """One kv_loss_weight has to transfer across the M / layer-map sweep."""
    torch.manual_seed(0)
    values = []
    for m, n_student in ((4, 2), (16, 4)):
        teacher_k = torch.full((1, L, H, m, D), 0.5)
        teacher_v = torch.full((1, L, H, m, D), 0.5)
        mapping = build_layer_map(n_student, L)
        student = {
            i: (torch.zeros(1, H, m, D), torch.zeros(1, H, m, D)) for i in range(n_student)
        }
        values.append(kv_matching_loss(student, teacher_k, teacher_v, mapping, kind="l1").item())
    torch.testing.assert_close(values[0], values[1])


def test_kv_loss_backprops_to_the_student_and_the_projector() -> None:
    b, m, n_student = 1, 4, 3
    teacher_k = torch.randn(b, L, H, m, D)
    teacher_v = torch.randn(b, L, H, m, D)
    mapping = build_layer_map(n_student, L)
    student = {
        i: (
            torch.randn(b, H, m, D, requires_grad=True),
            torch.randn(b, H, m, D, requires_grad=True),
        )
        for i in range(n_student)
    }
    bank = KVProjectorBank(n_student, kv_width=H * D, align="projector")
    kv_matching_loss(student, teacher_k, teacher_v, mapping, projector=bank).backward()
    for k, v in student.values():
        assert k.grad is not None and k.grad.abs().sum() > 0
        assert v.grad is not None and v.grad.abs().sum() > 0
    assert bank.k_proj[0].weight.grad.abs().sum() > 0


def test_layerwise_std_rescales_a_dominant_layer() -> None:
    """Without it, layers with huge K/V magnitudes swamp the gradient."""
    b, m, n_student = 1, 4, 2
    teacher_k = torch.randn(b, L, H, m, D)
    teacher_v = torch.randn(b, L, H, m, D)
    teacher_k[:, -1] *= 1000.0  # the "massive activation" layer
    mapping = build_layer_map(n_student, L)
    student = {i: (torch.zeros(b, H, m, D), torch.zeros(b, H, m, D)) for i in range(n_student)}
    plain = kv_matching_loss(student, teacher_k, teacher_v, mapping, kind="l1").item()
    scaled = kv_matching_loss(
        student, teacher_k, teacher_v, mapping, kind="l1", layerwise_std=True
    ).item()
    assert scaled < plain / 10.0


# ---------------------------------------------------------------- slot splice
SPLICE_IDS = dict(tfs_id=900, cot_start_id=901, cot_end_id=902, placeholder_id=13)


def test_splice_places_slots_immediately_before_tfs() -> None:
    ids = torch.tensor([[5, 6, 7, 900, 8, 9]])
    out = splice_slot_placeholders(ids, num_slots=3, **SPLICE_IDS)
    assert out["input_ids"].tolist() == [[5, 6, 7, 901, 13, 13, 13, 902, 900, 8, 9]]
    assert out["slot_pos"].tolist() == [[4, 5, 6]]


def test_splice_handles_left_padding_per_row() -> None:
    """Left padding puts the tfs column in a different place per row."""
    ids = torch.tensor([[0, 0, 5, 900, 8], [5, 6, 7, 900, 8]])
    attn = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    labels = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool)
    out = splice_slot_placeholders(
        ids, num_slots=2, attention_mask=attn, labels_mask=labels, **SPLICE_IDS
    )
    assert out["slot_pos"].tolist() == [[4, 5], [4, 5]]
    for row in range(2):
        new_ids = out["input_ids"][row]
        pos = out["slot_pos"][row]
        assert bool((new_ids[pos] == 13).all())
        assert int(new_ids[pos[-1] + 2]) == 900  # cot_end then tfs
    # padding stays masked; the inserted span is attended
    assert out["attention_mask"][0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1]
    # no CE on slots or their delimiters — the student is text-silent
    assert not bool(out["labels_mask"][0][out["slot_pos"][0]].any())


def test_splice_preserves_the_label_mask_after_the_insert() -> None:
    ids = torch.tensor([[5, 900, 8, 9]])
    labels = torch.tensor([[False, True, True, False]])
    out = splice_slot_placeholders(ids, num_slots=2, labels_mask=labels, **SPLICE_IDS)
    # the True span must still sit on <tfs> and the token after it
    kept = out["labels_mask"][0]
    tfs_col = int((out["input_ids"][0] == 900).nonzero()[0])
    assert kept[tfs_col] and kept[tfs_col + 1] and not kept[tfs_col + 2]


def test_splice_is_a_noop_with_zero_slots() -> None:
    ids = torch.tensor([[5, 900, 8]])
    out = splice_slot_placeholders(ids, num_slots=0, **SPLICE_IDS)
    assert out["input_ids"] is ids and out["slot_pos"].shape == (1, 0)


def test_splice_raises_without_tfs() -> None:
    try:
        splice_slot_placeholders(torch.tensor([[1, 2, 3]]), num_slots=2, **SPLICE_IDS)
    except ValueError as ex:
        assert "traj_future_start" in str(ex)
        return
    raise AssertionError("expected ValueError when <traj_future_start> is missing")


# ------------------------------------------------------------------- padding
def test_pad_kv_to_budget_marks_only_real_slots() -> None:
    k = torch.randn(L, H, 3, D)
    padded_k, padded_v, valid = pad_kv_to_budget(k, k.clone(), num_slots=5)
    assert padded_k.shape == (L, H, 5, D)
    assert valid.tolist() == [True, True, True, False, False]
    assert float(padded_k[:, :, 3:].abs().sum()) == 0.0
    torch.testing.assert_close(padded_k[:, :, :3], k)


def test_pad_kv_to_budget_rejects_oversized_entries() -> None:
    try:
        pad_kv_to_budget(torch.randn(L, H, 9, D), torch.randn(L, H, 9, D), num_slots=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError when the cached tier exceeds the budget")


# ------------------------------------------------------------------- cache IO
def test_compressed_tag_omits_lambda_where_it_is_meaningless() -> None:
    assert kv_cache_io.compressed_tag(16, 0.1, "rkv") == "compressed_M16_rkv0.1"
    assert kv_cache_io.compressed_tag(32, 0.1, "crop") == "compressed_M32_crop"
    assert kv_cache_io.compressed_tag(8, 0.0, "cosine") == "compressed_M8_cosine"


def test_entry_path_buckets_and_sanitises_the_key() -> None:
    key = "25cd4769-5dcf-4b53-a351-bf2c5deb6124::5100000"
    path = kv_cache_io.entry_path("/tmp/root", "full", key)
    assert path.parent.name == "25"
    assert ":" not in path.name
    assert path.name == "25cd4769-5dcf-4b53-a351-bf2c5deb6124__5100000.safetensors"


def test_cache_roundtrip_and_key_recovery(tmp_path) -> None:
    key = "abcd1234-0000-0000-0000-00000000ffff::5100000"
    k = torch.randn(L, H, 4, D)
    v = torch.randn(L, H, 4, D)
    kv_cache_io.save_compressed_entry(
        tmp_path, "compressed_M4_rkv0.1", key, k, v, n_valid=4, metadata={"m": 4}
    )
    assert kv_cache_io.has_entry(tmp_path, "compressed_M4_rkv0.1", key)
    loaded = kv_cache_io.load_entry(tmp_path, "compressed_M4_rkv0.1", key, names=("k_pre", "v"))
    torch.testing.assert_close(loaded["k_pre"], k.to(torch.bfloat16))
    torch.testing.assert_close(loaded["v"], v.to(torch.bfloat16))
    assert list(kv_cache_io.iter_keys(tmp_path, "compressed_M4_rkv0.1")) == [key]
    meta = kv_cache_io.load_entry_metadata(tmp_path, "compressed_M4_rkv0.1", key)
    assert meta["key"] == key and meta["n_valid"] == "4"


def test_load_entry_names_filter_skips_absent_tensors(tmp_path) -> None:
    key = "ab::1"
    kv_cache_io.save_full_entry(tmp_path, key, torch.randn(L, H, 4, D), torch.randn(L, H, 4, D))
    loaded = kv_cache_io.load_entry(tmp_path, "full", key, names=("k_pre", "imp"))
    assert set(loaded) == {"k_pre"}  # imp was never written; no KeyError


def test_missing_entry_error_names_the_fix(tmp_path) -> None:
    try:
        kv_cache_io.load_entry(tmp_path, "full", "ff::1")
    except KeyError as ex:
        assert "generate_teacher_kv" in str(ex)
        return
    raise AssertionError("expected KeyError for a missing cache entry")


def test_index_merges_across_shards(tmp_path) -> None:
    kv_cache_io.write_index(tmp_path, {"a::1": 40}, {"mode": "generate"}, shard=0)
    kv_cache_io.write_index(tmp_path, {"b::2": 55}, {"m": 16}, shard=1)
    merged = kv_cache_io.read_index(tmp_path)
    assert merged["n_cot"] == {"a::1": 40, "b::2": 55}
    assert merged["metadata"]["mode"] == "generate" and merged["metadata"]["m"] == 16


def test_read_index_tolerates_an_absent_index(tmp_path) -> None:
    assert kv_cache_io.read_index(tmp_path) == {"metadata": {}, "n_cot": {}}


# -------------------------------------------------- expert-scored importance
def test_score_from_qk_matches_the_eager_reference() -> None:
    """Must equal eager attention with NO mask — what sdpa does for is_causal=False.

    The expert runs non-causal, and `eager_attention_forward` ignores `is_causal`, so
    capturing weights via output_attentions would have returned *causal* attention the
    expert never computes. This pins the reconstruction to the real semantics.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    from alpamayo1_5_distill.models.expert_teacher import score_from_qk

    b, n_kv, n_rep, t_q, t_k, d = 2, 4, 2, 6, 20, 16
    lo, hi = 5, 11
    q = torch.randn(b, n_kv * n_rep, t_q, d)
    keys = torch.randn(b, n_kv, t_k, d)
    scaling = d**-0.5

    got = score_from_qk(q, keys, n_rep, scaling, lo, hi)

    # reference: exactly eager_attention_forward's arithmetic, mask-free
    ref_probs = (torch.matmul(q, repeat_kv(keys, n_rep).transpose(2, 3)) * scaling).softmax(-1)
    ref = ref_probs.view(b, n_kv, n_rep, t_q, t_k).max(dim=2).values[..., lo:hi].mean(dim=-2)
    assert got.shape == (b, n_kv, hi - lo)
    torch.testing.assert_close(got, ref)


def test_score_from_qk_normalises_over_the_full_key_axis() -> None:
    """Slicing the CoT columns must not renormalise: mass on other keys is real."""
    from alpamayo1_5_distill.models.expert_teacher import score_from_qk

    q = torch.randn(1, 2, 3, 8)
    keys = torch.randn(1, 2, 30, 8)
    full = score_from_qk(q, keys, 1, 8**-0.5, 0, 30)
    part = score_from_qk(q, keys, 1, 8**-0.5, 5, 10)
    torch.testing.assert_close(full.sum(-1), torch.ones(1, 2))  # whole axis sums to 1
    assert float(part.sum(-1).max()) < 1.0  # a slice does not


def test_score_from_qk_maxpools_before_averaging_over_queries() -> None:
    """Order matters and the paper fixes it: MaxPool the GQA group, *then* average.

    KAVA's GQA footnote puts the MaxPool "before computing the importance score", and
    the score is the mean over answer tokens — so it is ``mean_q(max_group(p))``, not
    ``max_group(mean_q(p))``. The two differ (Jensen: the former is ≥ the latter), and
    picking the wrong one silently changes which KV pairs survive eviction.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    from alpamayo1_5_distill.models.expert_teacher import score_from_qk

    n_kv, n_rep, t_q, t_k, d = 2, 2, 3, 12, 8
    q = torch.randn(1, n_kv * n_rep, t_q, d)
    keys = torch.randn(1, n_kv, t_k, d)
    pooled = score_from_qk(q, keys, n_rep, d**-0.5, 0, t_k)

    # per-q-head probabilities, keys repeated so the grouping collapses
    probs = (
        torch.matmul(q, repeat_kv(keys, n_rep).transpose(2, 3)) * d**-0.5
    ).softmax(-1)
    for kv in range(n_kv):
        group = probs[0, n_rep * kv : n_rep * (kv + 1)]  # [n_rep, T_q, T_k]
        max_then_mean = group.max(dim=0).values.mean(dim=0)
        mean_then_max = group.mean(dim=1).max(dim=0).values
        torch.testing.assert_close(pooled[0, kv], max_then_mean)
        assert bool((max_then_mean >= mean_then_max - 1e-6).all())
        assert not torch.allclose(max_then_mean, mean_then_max)  # they really differ


class _StubActionSpace:
    """Minimal action space: 4 waypoints x 2 dims, ignoring the trajectory content."""

    def get_action_space_dims(self):
        return (4, 2)

    def traj_to_action(self, **kwargs):
        return torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)


def _stub_expert_model():
    import types

    from alpamayo_r1.diffusion.flow_matching import FlowMatching

    return types.SimpleNamespace(
        action_space=_StubActionSpace(), diffusion=FlowMatching(x_dims=(4, 2))
    )


#: The four trajectory keys the real caller always supplies; the stub ignores them.
STUB_TRAJ = dict.fromkeys(
    ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
)


def test_build_noisy_action_hits_both_flow_endpoints() -> None:
    """t=1 is the clean action (KAVA's 'answer'), t=0 is what inference starts from."""
    from alpamayo1_5_distill.models.expert_teacher import build_noisy_action

    model = _stub_expert_model()
    clean, t1, noise = build_noisy_action(model, STUB_TRAJ, timestep=1.0, seed=0)
    pure, t0, _ = build_noisy_action(model, STUB_TRAJ, timestep=0.0, noise=noise, seed=0)
    torch.testing.assert_close(clean, torch.arange(8, dtype=torch.float32).reshape(1, 4, 2))
    torch.testing.assert_close(pure, noise)
    assert t1.shape == (1, 1, 1) and float(t1) == 1.0 and float(t0) == 0.0


def test_build_noisy_action_is_seed_deterministic() -> None:
    """An offline cache cannot tolerate the random t / noise the training path uses."""
    from alpamayo1_5_distill.models.expert_teacher import build_noisy_action

    model = _stub_expert_model()
    a, _, na = build_noisy_action(model, STUB_TRAJ, 0.5, seed=7)
    b, _, nb = build_noisy_action(model, STUB_TRAJ, 0.5, seed=7)
    c, _, nc = build_noisy_action(model, STUB_TRAJ, 0.5, seed=8)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(na, nb)
    assert not torch.allclose(na, nc)


def test_build_noisy_action_rejects_a_foreign_diffusion() -> None:
    """A different scheme has a different interpolation; fail rather than mis-score."""
    import types

    from alpamayo1_5_distill.models.expert_teacher import build_noisy_action

    model = types.SimpleNamespace(action_space=_StubActionSpace(), diffusion=object())
    try:
        build_noisy_action(model, STUB_TRAJ, 0.5)
    except NotImplementedError as ex:
        assert "FlowMatching" in str(ex)
        return
    raise AssertionError("expected NotImplementedError for a non-FlowMatching diffusion")


def test_tier_stats_reports_per_entry_size(tmp_path) -> None:
    for i in range(3):
        kv_cache_io.save_compressed_entry(
            tmp_path, "t", f"a{i}::1", torch.randn(L, H, 2, D), torch.randn(L, H, 2, D)
        )
    stats = kv_cache_io.tier_stats(tmp_path, "t")
    assert stats["n_entries"] == 3 and stats["mb_per_entry"] > 0
    assert kv_cache_io.tier_stats(tmp_path, "absent")["n_entries"] == 0
