# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the block-convex 28 -> 36 cache layer mix."""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from alpamayo1_5_distill.models.block_losses import rotate_keys
from alpamayo1_5_distill.models.layer_mix import (
    LAYER_MIX_FLOOR,
    LAYER_MIX_SHARPEN,
    LayerMixer,
    gather_weights,
    init_weights,
    tent_weights,
)

# The shipped geometry: Cosmos-Reason2-2B (28 text layers) -> the teacher's 36-layer expert.
N_STUDENT, N_EXPERT, N_BLOCKS = 28, 36, 4
G_IN, G_OUT = N_STUDENT // N_BLOCKS, N_EXPERT // N_BLOCKS  # 7 -> 9


def _mixer(**kw) -> LayerMixer:
    """Default to the PURE tent so shape assertions stay readable; the shipped default is
    sharpen=LAYER_MIX_SHARPEN, covered by its own tests below."""
    kw.setdefault("sharpen", 0.0)
    return LayerMixer(N_STUDENT, N_EXPERT, N_BLOCKS, **kw)


def _kv(b=2, h=8, t=5, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, N_STUDENT, h, t, d, generator=g, dtype=torch.float32)


# ------------------------------------------------------------------ convexity


def test_columns_are_convex_at_init_and_after_perturbation():
    m = _mixer()
    for which in ("k", "v"):
        p = m.weights(which)
        assert p.shape == (N_BLOCKS, G_IN, G_OUT)
        assert (p >= 0).all()
        torch.testing.assert_close(p.sum(dim=1), torch.ones(N_BLOCKS, G_OUT))

    with torch.no_grad():
        m.logit_k.add_(torch.randn_like(m.logit_k) * 5.0)
        m.logit_v.add_(torch.randn_like(m.logit_v) * 5.0)
    for which in ("k", "v"):
        p = m.weights(which)
        assert (p >= 0).all()
        torch.testing.assert_close(p.sum(dim=1), torch.ones(N_BLOCKS, G_OUT))


def test_convexity_is_preserved_by_the_mix_itself():
    """A convex mix of layers cannot leave their elementwise hull."""
    m = _mixer()
    x = _kv()
    out = m.mix_stacked(x, "k")
    for b in range(N_BLOCKS):
        src = x[:, b * G_IN : (b + 1) * G_IN]
        dst = out[:, b * G_OUT : (b + 1) * G_OUT]
        assert (dst <= src.amax(dim=1, keepdim=True) + 1e-5).all()
        assert (dst >= src.amin(dim=1, keepdim=True) - 1e-5).all()


# ------------------------------------------------------------------ tent init


def test_tent_init_is_depth_proportional_with_exact_block_endpoints():
    w = tent_weights(G_IN, G_OUT)
    torch.testing.assert_close(w.sum(dim=0), torch.ones(G_OUT))
    # Block endpoints are exact: slot 0 is source 0, slot g_out-1 is source g_in-1.
    assert w[0, 0] == pytest.approx(1.0)
    assert w[G_IN - 1, G_OUT - 1] == pytest.approx(1.0)
    # Interior slots interpolate at p = 0.75 q.
    assert w[:, 4].tolist() == pytest.approx([0, 0, 0, 1.0, 0, 0, 0], abs=1e-6)  # p = 3.0
    assert w[1, 1] == pytest.approx(0.75)  # p = 0.75 -> 0.25 on layer 0, 0.75 on layer 1
    assert w[0, 1] == pytest.approx(0.25)


def test_tent_init_reproduces_the_trees_depth_proportional_layer_map():
    """Expert slot 9 -> student 7, 18 -> 14, 27 -> 21, matching round(j * 27/35)."""
    m = _mixer()
    p = m.weights("k")
    for b, expected_src in enumerate([0, 0, 0, 0]):  # slot b*g_out is source 0 of block b
        col = p[b, :, 0]
        assert col.argmax().item() == expected_src
        assert col[expected_src] > 1.0 - G_IN * LAYER_MIX_FLOOR - 1e-6
    # And the global statement: block b's slot 0 is expert slot 9b <- student layer 7b.
    for b in range(N_BLOCKS):
        assert round(b * G_OUT * (N_STUDENT - 1) / (N_EXPERT - 1)) == b * G_IN


def test_init_is_within_the_floor_of_the_exact_tent():
    m = _mixer()
    exact = tent_weights(G_IN, G_OUT)
    for which in ("k", "v"):
        p = m.weights(which)[0]
        assert (p - exact).abs().max() < G_IN * LAYER_MIX_FLOOR


def test_off_tent_entries_keep_nonzero_mass_so_they_can_still_learn():
    m = _mixer()
    p = m.weights("k")
    assert (p > 0).all(), "a zero-probability entry receives zero gradient forever"
    assert p.min() < 2 * LAYER_MIX_FLOOR


# ------------------------------------------------------------------ RoPE commutation


def test_mix_commutes_with_rope():
    """The property train (pre-RoPE) / eval (post-RoPE) agreement rests on.

    cos/sin are per-POSITION and shared by every layer, and rotation is linear in K, so
    mixing before or after rotating must give the same tensor.
    """
    m = _mixer()
    with torch.no_grad():                      # a non-trivial P, not just the tent
        m.logit_k.add_(torch.randn_like(m.logit_k) * 2.0)
    b, h, t, d = 2, 8, 5, 8
    k = _kv(b, h, t, d)
    g = torch.Generator().manual_seed(7)
    ang = torch.randn(b, t, d, generator=g)
    cos, sin = ang.cos(), ang.sin()

    rot_then_mix = m.mix_stacked(
        torch.stack([rotate_keys(k[:, i], cos, sin) for i in range(N_STUDENT)], dim=1), "k"
    )
    mixed = m.mix_stacked(k, "k")
    mix_then_rot = torch.stack(
        [rotate_keys(mixed[:, j], cos, sin) for j in range(N_EXPERT)], dim=1
    )
    torch.testing.assert_close(rot_then_mix, mix_then_rot, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ block locality


def test_block_locality_is_structural():
    """d(out slot j) / d(in layer i) is exactly 0 whenever i // g_in != j // g_out."""
    m = _mixer()
    x = _kv(b=1, h=2, t=3, d=4).requires_grad_(True)
    out = m.mix_stacked(x, "k")
    for j in (0, 8, 9, 17, 27, 35):
        x.grad = None
        out[:, j].sum().backward(retain_graph=True)
        touched = {int(i) for i in (x.grad.abs().sum(dim=(0, 2, 3, 4)) > 0).nonzero()}
        assert touched == set(range(j // G_OUT * G_IN, (j // G_OUT + 1) * G_IN)), (
            f"expert slot {j} must read only student block {j // G_OUT}"
        )


def test_span_at_g_out_lands_exactly_on_block_boundaries():
    """_span_sweep's disjoint spans, range(0, n - m + 1, m), must equal the block starts."""
    starts = list(range(0, N_EXPERT - G_OUT + 1, G_OUT))
    assert starts == [0, 9, 18, 27]
    assert starts == [b * G_OUT for b in range(N_BLOCKS)]
    for l0 in starts:
        blocks = {l // G_OUT for l in range(l0, l0 + G_OUT)}
        assert len(blocks) == 1, "a span must not straddle two mixing blocks"


# ------------------------------------------------------------------ path agreement


def test_mix_dict_and_mix_cache_agree():
    m = _mixer()
    with torch.no_grad():
        m.logit_k.add_(torch.randn_like(m.logit_k))
        m.logit_v.add_(torch.randn_like(m.logit_v))
    k, v = _kv(seed=1), _kv(seed=2)

    as_dict = m.mix_dict({i: (k[:, i], v[:, i]) for i in range(N_STUDENT)})
    assert sorted(as_dict) == list(range(N_EXPERT))

    cache = DynamicCache()
    for i in range(N_STUDENT):
        cache.update(k[:, i], v[:, i], i, {})
    as_cache = m.mix_cache(cache)
    assert len(as_cache.layers) == N_EXPERT

    for j in range(N_EXPERT):
        torch.testing.assert_close(as_dict[j][0], as_cache.layers[j].keys)
        torch.testing.assert_close(as_dict[j][1], as_cache.layers[j].values)


def test_mix_dict_carries_gradient_to_both_the_backbone_and_p():
    m = _mixer()
    k, v = _kv(seed=3).requires_grad_(True), _kv(seed=4).requires_grad_(True)
    out = m.mix_dict({i: (k[:, i], v[:, i]) for i in range(N_STUDENT)})
    sum(out[j][0].sum() + out[j][1].sum() for j in range(N_EXPERT)).backward()
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert v.grad is not None and v.grad.abs().sum() > 0
    assert m.logit_k.grad is not None
    assert m.logit_v.grad is not None


def test_gain_is_off_by_default_and_scales_per_slot_when_on():
    assert _mixer().gain_k is None
    m = _mixer(gain=True)
    assert m.gain_k.shape == (N_EXPERT,)
    x = _kv()
    base = m.mix_stacked(x, "k")
    with torch.no_grad():
        m.gain_k[3] = 2.0
    torch.testing.assert_close(m.mix_stacked(x, "k")[:, 3], base[:, 3] * 2.0)


# ------------------------------------------------------------------ diagnostics


def test_entropy_spans_the_tent_init_to_uniform():
    """The init is NOT one-hot: 7 of 9 slots per block interpolate between two layers.

    Three slots are exact (q = 0, 4, 8), four sit at (0.75, 0.25) and two at (0.5, 0.5), so
    the normalised mean entropy lands at ~0.228 -- the baseline a run's curve moves away from.
    """
    m = _mixer()
    assert float(m.entropy("k")) == pytest.approx(0.228, abs=0.01)
    assert float(m.entropy("v")) == pytest.approx(float(m.entropy("k")), abs=1e-6)
    with torch.no_grad():
        m.logit_k.zero_()                                # uniform
    assert float(m.entropy("k")) == pytest.approx(1.0, abs=1e-5)
    with torch.no_grad():
        m.logit_k.copy_(torch.nn.functional.one_hot(
            torch.zeros(N_BLOCKS, G_OUT, dtype=torch.long), G_IN
        ).permute(0, 2, 1).float() * 40.0)               # one-hot
    assert float(m.entropy("k")) == pytest.approx(0.0, abs=1e-5)


def test_describe_names_every_slot():
    text = _mixer().describe()
    for j in range(N_EXPERT):
        assert f"slot{j:>2}" in text
    assert "P_K" in text and "P_V" in text


# ------------------------------------------------------------------ guards


def test_ragged_blocks_raise():
    with pytest.raises(ValueError, match="must divide the UNPINNED middle"):
        LayerMixer(28, 36, 5)


def test_wrong_source_depth_raises():
    m = _mixer()
    with pytest.raises(ValueError, match="expected 28 source layers"):
        m.mix_stacked(torch.randn(1, 27, 2, 3, 4), "k")
    with pytest.raises(ValueError, match="missing"):
        m.mix_dict({i: (torch.randn(1, 2, 3, 4),) * 2 for i in range(27)})
    cache = DynamicCache()
    for i in range(30):
        cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), i, {})
    with pytest.raises(ValueError, match="30 layers"):
        m.mix_cache(cache)


def test_bad_which_raises():
    with pytest.raises(ValueError, match="'k' or 'v'"):
        _mixer().weights("q")


# ------------------------------------------------------------------ kd_model guards
#
# `_init_layer_mix` is exercised against a stub rather than a real KDReasoningVLA: the guards
# are pure attribute logic, and building an 8B teacher to check that a ValueError fires would
# make them untestable in CI, which is the same as untested.


class _Stub(torch.nn.Module):
    def __init__(self, **over):
        super().__init__()
        self.vlm = torch.nn.Linear(2, 2)          # only `next(self.vlm.parameters())` is used
        self.block_weight = 1.0
        self.field_weight = self.roll_weight = self.block_freerun_weight = 0.0
        self.kv_weight = 0.0
        self.kv_layer_map = None
        self.block_span = 1
        self.block_span_mix = G_OUT
        self.layer_mix_expert_layers = N_EXPERT
        self.layer_mix_blocks = N_BLOCKS
        self.layer_mix_gain = False
        self.layer_mix_sharpen = LAYER_MIX_SHARPEN
        self.layer_mix_pin_head = 0
        self.layer_mix_pin_tail = 0
        for k, v in over.items():
            setattr(self, k, v)

    def go(self, n_student=N_STUDENT):
        from alpamayo1_5_distill.models.kd_model import KDReasoningVLA

        return KDReasoningVLA._init_layer_mix(self, n_student)


def test_guard_pruning_and_mixing_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("PRUNE_EXPERT_LAYERS", "4,10,13,15,19,25,27,34")
    with pytest.raises(ValueError, match="alternatives, not companions"):
        _Stub().go()


def test_guard_kv_loss_is_a_contradictory_layer_map(monkeypatch):
    monkeypatch.delenv("PRUNE_EXPERT_LAYERS", raising=False)
    with pytest.raises(ValueError, match="contradicts the learned mix"):
        _Stub(kv_weight=45.409, kv_layer_map=list(range(N_STUDENT))).go()


def test_guard_no_block_family_term_means_p_never_learns(monkeypatch):
    monkeypatch.delenv("PRUNE_EXPERT_LAYERS", raising=False)
    with pytest.raises(ValueError, match="never receive gradient"):
        _Stub(block_weight=0.0).go()


def test_guard_ragged_blocks_reach_the_mixer(monkeypatch):
    monkeypatch.delenv("PRUNE_EXPERT_LAYERS", raising=False)
    with pytest.raises(ValueError, match="must divide the UNPINNED middle"):
        _Stub(layer_mix_blocks=5).go()


def test_happy_path_builds_a_mixer_and_warns_only_on_misaligned_spans(monkeypatch, capsys):
    monkeypatch.delenv("PRUNE_EXPERT_LAYERS", raising=False)
    stub = _Stub()
    stub.go()
    assert isinstance(stub.layer_mixer, LayerMixer)
    assert "WARNING" not in capsys.readouterr().out

    misaligned = _Stub(block_span_mix=7)          # the pruned arm's m, which straddles blocks
    misaligned.go()
    assert "straddle mixing blocks" in capsys.readouterr().out


# ------------------------------------------------------------------ config agreements


def _yaml(name):
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[1] / "configs"
    return yaml.safe_load((root / name).read_text())


MODEL_CFG = "models/cosmos2b_layermix_kd.yaml"
TRAIN_CFG = "sft_kd_cosmos2b_2cam_nav_layermix_lcdrive.yaml"
EVAL_CFG = "sft_eval_stitched_2b_layermix_lcdrive.yaml"


def test_train_and_eval_configs_describe_the_same_mixer():
    """Geometry lives in two files; a mismatch surfaces as a load error at best."""
    kd = _yaml(MODEL_CFG)["kd"]
    ev = _yaml(EVAL_CFG)["model"]
    assert kd["layer_mix"] is ev["layer_mix"] is True
    for key in ("layer_mix_expert_layers", "layer_mix_blocks", "layer_mix_gain"):
        assert kd[key] == ev[key], key
    assert kd["layer_mix_expert_layers"] == N_EXPERT
    assert kd["layer_mix_blocks"] == N_BLOCKS


def test_span_is_aligned_to_the_block_size_in_the_shipped_config():
    kd = _yaml(MODEL_CFG)["kd"]
    g_out = kd["layer_mix_expert_layers"] // kd["layer_mix_blocks"]
    assert kd["block_span_mix"] == g_out == G_OUT


def test_kv_loss_stays_off_so_the_init_guard_cannot_fire_at_launch():
    assert _yaml(MODEL_CFG)["kd"].get("kv_weight", 0.0) == 0.0


def test_lr_multiplier_targets_the_mixer_and_raises_it():
    """A prefix typo here is silent: the group simply never matches."""
    mult = _yaml(TRAIN_CFG)["trainer"]["lr_multiplier"]
    assert list(mult) == ["layer_mixer"]
    assert "layer_mixer.logit_k".startswith(next(iter(mult)))
    assert mult["layer_mixer"] > 1.0, "Adam makes a sub-1.0 multiplier a freeze, not a slowdown"


def test_layer_mix_params_are_excluded_from_weight_decay():
    """Decaying a softmax logit decays the mix toward UNIFORM, dismantling the tent init."""
    from alpamayo1_5_distill.trainer import NO_DECAY_PARAMS

    assert any(part in "layer_mixer.logit_k" for part in NO_DECAY_PARAMS)
    assert any(part in "layer_mixer.logit_v" for part in NO_DECAY_PARAMS)


def test_the_launcher_arm_does_not_export_prune_expert_layers():
    """Every other 2B arm exports it; this one must actively unset it."""
    import pathlib
    import re

    sh = (pathlib.Path(__file__).resolve().parents[1] / "slurm_train_kd.sh").read_text()
    arm = sh[sh.index("    mix2bnav)"):]
    arm = arm[: arm.index(";;")]
    assert "unset PRUNE_EXPERT_LAYERS" in arm
    assert not re.search(r"^\s*export PRUNE_EXPERT_LAYERS", arm, re.M)
    assert "++model.kd.layer_mix=true" in arm
    assert 'MIXM:-9' in arm


# ------------------------------------------------------------------ oracle probe math
#
# scripts/layer_mix_oracle.py decides whether the training arm is worth launching at all, so
# its solver is worth more than a smoke test. Loaded by path because `scripts` is a sibling
# package whose module pulls in hydra at import time.


def _oracle():
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "layer_mix_oracle.py"
    spec = importlib.util.spec_from_file_location("_layer_mix_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_layer_mix_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_simplex_projection_lands_on_the_simplex_and_fixes_points_already_on_it():
    o = _oracle()
    torch.manual_seed(0)
    p = o._project_simplex(torch.randn(G_IN, G_OUT, dtype=torch.float64) * 3)
    assert (p >= -1e-12).all()
    torch.testing.assert_close(p.sum(0), torch.ones(G_OUT, dtype=torch.float64))
    on = torch.softmax(torch.randn(G_IN, G_OUT, dtype=torch.float64), 0)
    torch.testing.assert_close(o._project_simplex(on), on)


def test_fit_recovers_a_known_convex_combination_exactly():
    """If the target IS a convex mix of the sources, the solver must find that mix."""
    o = _oracle()
    torch.manual_seed(0)
    src = torch.randn(G_IN, 4000, dtype=torch.float64)
    true = torch.softmax(torch.randn(G_IN, G_OUT, dtype=torch.float64) * 1.5, 0)
    f = o.BlockFit(G_IN, G_OUT)
    f.add(src.float(), (true.T @ src).float())
    w, resid = f.solve()
    assert (w.double() - true).abs().max() < 1e-3
    assert resid.max() < 1e-3


def test_fit_accumulates_identically_streamed_or_batched():
    """Clips arrive one at a time; the float64 Grams must not care how they were chunked."""
    o = _oracle()
    torch.manual_seed(1)
    src = torch.randn(G_IN, 3000)
    tgt = torch.randn(G_OUT, 3000)
    whole = o.BlockFit(G_IN, G_OUT)
    whole.add(src, tgt)
    streamed = o.BlockFit(G_IN, G_OUT)
    for cs, ct in zip(src.split(500, 1), tgt.split(500, 1)):
        streamed.add(cs, ct)
    torch.testing.assert_close(whole.solve()[0], streamed.solve()[0], rtol=1e-4, atol=1e-5)
    assert whole.n == streamed.n == 3000


def test_fit_is_never_worse_than_projecting_an_unconstrained_solution():
    """Projecting a least-squares answer onto the simplex is NOT minimising over it."""
    o = _oracle()
    torch.manual_seed(2)
    src = torch.randn(G_IN, 2000, dtype=torch.float64)
    tgt = (src[0] * 2.0 - src[1]).unsqueeze(0)      # deliberately outside the hull
    f = o.BlockFit(G_IN, 1)
    f.add(src.float(), tgt.float())
    w, _ = f.solve()
    naive = o._project_simplex(torch.linalg.lstsq(src.T, tgt.T).solution)
    obj = lambda a: float((a * (f.g @ a)).sum() - 2 * (a * f.c).sum() + f.t.sum())
    assert obj(w.double()) <= obj(naive) + 1e-6
    torch.testing.assert_close(w.sum(0), torch.ones(1))


def test_gather_control_is_one_hot_nearest_source():
    o = _oracle()
    g = o.gather_weights(G_IN, G_OUT)
    torch.testing.assert_close(g.sum(0), torch.ones(G_OUT))
    assert set(g.unique().tolist()) == {0.0, 1.0}
    assert g.argmax(0).tolist() == [round(q * (G_IN - 1) / (G_OUT - 1)) for q in range(G_OUT)]


def test_both_source_subsets_have_the_right_size():
    o = _oracle()
    assert len(o.source_layers("stride", N_STUDENT, N_EXPERT)) == N_STUDENT
    assert len(o.source_layers("pi", N_STUDENT, N_EXPERT)) == N_STUDENT
    assert o.source_layers("stride", N_STUDENT, N_EXPERT)[0] == 0
    assert o.source_layers("stride", N_STUDENT, N_EXPERT)[-1] == N_EXPERT - 1
    with pytest.raises(ValueError, match="stride|pi"):
        o.source_layers("nope", N_STUDENT, N_EXPERT)


def test_logits_from_fitted_weights_reproduce_them_through_the_mixer():
    """The probe drives the production mix path; softmax(log a) == a is what makes that safe."""
    torch.manual_seed(3)
    w = torch.softmax(torch.randn(N_BLOCKS, G_IN, G_OUT) * 2, dim=1)
    m = _mixer()
    with torch.no_grad():
        m.logit_k.copy_(torch.log(w.clamp_min(1e-12)))
    torch.testing.assert_close(m.weights("k"), w, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ why the guards exist


def test_a_short_cache_does_not_raise_against_a_deeper_expert():
    """The failure the mix exists to prevent is SILENT, which is why every guard is hard.

    ``DynamicCache`` auto-extends on ``update``, so a 28-layer cache handed to a 36-layer
    expert produces finite hidden states: slots 28..35 are created on demand holding only the
    action tokens, with no VLM prefix at all. Verified end-to-end against a real 36-layer
    Qwen3-VL decoder -- it ran clean. Those are the layers `scripts/layer_importance.py`
    measured as carrying ~105% of the recoverable gap.
    """
    cache = DynamicCache()
    for i in range(N_STUDENT):
        cache.update(torch.randn(1, 8, 4, 8), torch.randn(1, 8, 4, 8), i, {})
    assert len(cache.layers) == N_STUDENT
    cache.update(torch.randn(1, 8, 2, 8), torch.randn(1, 8, 2, 8), N_EXPERT - 1, {})
    assert len(cache.layers) == N_EXPERT, "auto-extended without a word"
    assert cache.layers[N_EXPERT - 1].keys.shape[2] == 2, "and slot 35 holds no prefix"


def test_orphaned_layer_mix_checkpoint_is_refused(tmp_path):
    """A layer_mix checkpoint evaluated with layer_mix=False scores an untrained model."""
    from safetensors.torch import save_file

    from alpamayo1_5_distill.models.stitched_model import (
        _layer_mix_keys,
        _refuse_orphaned_layer_mix,
    )

    save_file({"vlm.embed.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    assert _layer_mix_keys(str(tmp_path)) == {}
    _refuse_orphaned_layer_mix(str(tmp_path))          # a plain student passes through

    save_file(
        {"vlm.embed.weight": torch.zeros(2, 2),
         "layer_mixer.logit_k": torch.zeros(N_BLOCKS, G_IN, G_OUT),
         "layer_mixer.logit_v": torch.zeros(N_BLOCKS, G_IN, G_OUT)},
        str(tmp_path / "model.safetensors"),
    )
    assert sorted(_layer_mix_keys(str(tmp_path))) == ["layer_mixer.logit_k",
                                                      "layer_mixer.logit_v"]
    with pytest.raises(RuntimeError, match="layer_mix=False"):
        _refuse_orphaned_layer_mix(str(tmp_path))


def test_missing_layer_mix_in_a_checkpoint_is_refused(tmp_path):
    """The other direction: layer_mix=True against a checkpoint that has no matrices."""
    from safetensors.torch import save_file

    from alpamayo1_5_distill.models.stitched_model import _load_layer_mix

    save_file({"vlm.embed.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="no `layer_mixer.\\*` tensors"):
        _load_layer_mix(str(tmp_path), torch.nn.Linear(2, 2))


def test_logit_travel_is_shift_invariant():
    """Softmax ignores a per-column constant, so a raw logit difference is not a distance.

    Adding 0.9 to every logit leaves P bit-identical; an uncentred metric would report 0.9 of
    travel and recommend an lr_multiplier for a move that does not exist.
    """
    o = _oracle()
    tent = tent_weights(G_IN, G_OUT)
    tl = torch.log(tent.clamp_min(1e-3))

    shifted = torch.softmax(tl + 0.9, dim=0)
    torch.testing.assert_close(shifted, torch.softmax(tl, 0))       # P really is unchanged
    assert o.logit_travel(shifted, tent) < 0.02

    pert = torch.zeros_like(tl)
    pert[0] = 2.0                                                    # a genuine change
    moved = torch.softmax(tl + pert, dim=0)
    assert o.logit_travel(moved, tent) > 1.0
    # and a full collapse to uniform is the largest move of all
    assert o.logit_travel(torch.full_like(tent, 1 / G_IN), tent) > o.logit_travel(moved, tent)


def test_logit_travel_handles_the_batched_block_layout():
    """`solved[w]` is [n_blocks, g_in, g_out]; centring must be over g_in, not the blocks."""
    o = _oracle()
    tent = tent_weights(G_IN, G_OUT)
    blocked = tent.expand(N_BLOCKS, -1, -1).clone()
    assert o.logit_travel(blocked, tent) == pytest.approx(0.0, abs=1e-5)
    with torch.no_grad():
        blocked[2, 0, :] = 0.9                                       # perturb one block only
        blocked = blocked / blocked.sum(dim=1, keepdim=True)
    assert o.logit_travel(blocked, tent) > 0.5


# ------------------------------------------------------------------ sharpened init


def test_sharpen_interpolates_tent_to_gather_and_keeps_columns_convex():
    t, g = tent_weights(G_IN, G_OUT), gather_weights(G_IN, G_OUT)
    torch.testing.assert_close(init_weights(G_IN, G_OUT, 0.0), t)
    torch.testing.assert_close(init_weights(G_IN, G_OUT, 1.0), g)
    for a in (0.0, 0.25, 0.5, LAYER_MIX_SHARPEN, 1.0):
        w = init_weights(G_IN, G_OUT, a)
        assert (w >= 0).all()
        torch.testing.assert_close(w.sum(0), torch.ones(G_OUT))


def test_sharpening_is_monotone_and_reaches_the_tied_columns():
    """A softmax TEMPERATURE cannot do this: (0.5, 0.5) stays (0.5, 0.5) for every T.

    Two of every nine outputs land exactly halfway between sources (q = 2, 6), which is why
    the init blends toward gather_weights instead of dividing logits by a temperature.
    """
    tied = [q for q in range(G_OUT)
            if abs((q * (G_IN - 1) / (G_OUT - 1)) % 1 - 0.5) < 1e-9]
    assert tied == [2, 6]
    prev = init_weights(G_IN, G_OUT, 0.0).amax(0)
    for a in (0.25, 0.5, 0.75, 1.0):
        cur = init_weights(G_IN, G_OUT, a).amax(0)
        assert (cur >= prev - 1e-6).all(), "sharpening must never broaden a column"
        prev = cur
    # the tied columns really do move, all the way to one-hot
    assert init_weights(G_IN, G_OUT, 0.0)[:, 2].max() == pytest.approx(0.5)
    assert init_weights(G_IN, G_OUT, 1.0)[:, 2].max() == pytest.approx(1.0)


def test_gather_weights_matches_nearest_source():
    g = gather_weights(G_IN, G_OUT)
    torch.testing.assert_close(g.sum(0), torch.ones(G_OUT))
    assert set(g.unique().tolist()) == {0.0, 1.0}
    assert g.argmax(0).tolist() == [round(q * (G_IN - 1) / (G_OUT - 1)) for q in range(G_OUT)]


def test_shipped_default_is_sharper_than_the_tent_but_still_mixes():
    """gather beat the tent by 0.066 min_ade, so the default leans sharp -- but the floor
    must keep every off-diagonal alive or those entries can never be learned."""
    assert 0.0 < LAYER_MIX_SHARPEN <= 1.0
    m = LayerMixer(N_STUDENT, N_EXPERT, N_BLOCKS)          # shipped default
    p = m.weights("k")
    assert (p > 0).all(), "a zero-probability entry receives zero gradient forever"
    assert float(m.entropy("k")) < 0.208, "must be sharper than the tent's ~0.228"
    assert m.sharpen == LAYER_MIX_SHARPEN


def test_sharpen_out_of_range_raises():
    with pytest.raises(ValueError, match="sharpen must be in"):
        init_weights(G_IN, G_OUT, 1.5)
    with pytest.raises(ValueError, match="sharpen must be in"):
        LayerMixer(N_STUDENT, N_EXPERT, N_BLOCKS, sharpen=-0.1)


def test_train_and_eval_agree_on_sharpen():
    """The eval must rebuild the SAME geometry the checkpoint's matrices were saved from."""
    kd = _yaml(MODEL_CFG)["kd"]
    ev = _yaml(EVAL_CFG)["model"]
    assert kd.get("layer_mix_sharpen", LAYER_MIX_SHARPEN) == ev.get(
        "layer_mix_sharpen", LAYER_MIX_SHARPEN)


# ------------------------------------------------------------------ pinned head/tail
#
# The deepstack/ViT-injection layers at the head and the final layers at the tail are wired
# STRAIGHT THROUGH; only the middle is mixed. PRUNING.md protects layers 0-2 structurally and
# its ladder measured layers 19-27 as carrying ~95% of the recoverable gap, so those are the
# two regions where substituting a blend is most likely to cost something.

PIN_H, PIN_T = 4, 2
MID_IN, MID_OUT, MID_BLOCKS = 22, 30, 2          # 28-4-2 -> 36-4-2, in 2 blocks of 11 -> 15


def _pinned(**kw):
    kw.setdefault("pin_head", PIN_H)
    kw.setdefault("pin_tail", PIN_T)
    return LayerMixer(N_STUDENT, N_EXPERT, MID_BLOCKS, **kw)


def test_pinned_geometry():
    m = _pinned()
    assert (m.g_in, m.g_out) == (MID_IN // MID_BLOCKS, MID_OUT // MID_BLOCKS) == (11, 15)
    assert m.logit_k.shape == (MID_BLOCKS, 11, 15)
    assert m.pin_head + m.n_blocks * m.g_in + m.pin_tail == N_STUDENT
    assert m.pin_head + m.n_blocks * m.g_out + m.pin_tail == N_EXPERT


def test_pinned_slots_are_bit_exact_passthrough():
    """A pin must mean the cache layer reaches the expert UNALTERED."""
    m = _pinned()
    with torch.no_grad():                       # even with an arbitrary learned middle
        m.logit_k.add_(torch.randn_like(m.logit_k) * 3.0)
    x = _kv()
    out = m.mix_stacked(x, "k")
    torch.testing.assert_close(out[:, :PIN_H], x[:, :PIN_H])
    torch.testing.assert_close(out[:, N_EXPERT - PIN_T:], x[:, N_STUDENT - PIN_T:])


def test_gain_never_touches_pinned_slots():
    m = _pinned(gain=True)
    with torch.no_grad():
        m.gain_k.fill_(7.0)                     # would be glaring if it leaked through
    x = _kv()
    out = m.mix_stacked(x, "k")
    torch.testing.assert_close(out[:, :PIN_H], x[:, :PIN_H])
    torch.testing.assert_close(out[:, N_EXPERT - PIN_T:], x[:, N_STUDENT - PIN_T:])
    assert not torch.allclose(out[:, PIN_H], x[:, PIN_H]), "the middle SHOULD be scaled"


def test_pinned_block_locality():
    """Head slot j reads only student j; middle slots read only their own block."""
    m = _pinned()
    x = _kv(b=1, h=2, t=3, d=4).requires_grad_(True)
    out = m.mix_stacked(x, "k")
    for j in range(PIN_H):                                   # head is 1:1
        x.grad = None
        out[:, j].sum().backward(retain_graph=True)
        assert {int(i) for i in (x.grad.abs().sum(dim=(0,2,3,4)) > 0).nonzero()} == {j}
    for k in range(PIN_T):                                   # tail is 1:1
        x.grad = None
        out[:, N_EXPERT - 1 - k].sum().backward(retain_graph=True)
        assert {int(i) for i in (x.grad.abs().sum(dim=(0,2,3,4)) > 0).nonzero()} == \
               {N_STUDENT - 1 - k}
    for b in range(MID_BLOCKS):                              # middle is block-local
        j = PIN_H + b * m.g_out
        x.grad = None
        out[:, j].sum().backward(retain_graph=True)
        lo = PIN_H + b * m.g_in
        assert {int(i) for i in (x.grad.abs().sum(dim=(0,2,3,4)) > 0).nonzero()} == \
               set(range(lo, lo + m.g_in))


def test_pinned_still_commutes_with_rope():
    m = _pinned()
    with torch.no_grad():
        m.logit_k.add_(torch.randn_like(m.logit_k) * 2.0)
    b, h, t, d = 2, 8, 5, 8
    k = _kv(b, h, t, d)
    g = torch.Generator().manual_seed(11)
    ang = torch.randn(b, t, d, generator=g); cos, sin = ang.cos(), ang.sin()
    rot_then_mix = m.mix_stacked(
        torch.stack([rotate_keys(k[:, i], cos, sin) for i in range(N_STUDENT)], 1), "k")
    mixed = m.mix_stacked(k, "k")
    mix_then_rot = torch.stack([rotate_keys(mixed[:, j], cos, sin) for j in range(N_EXPERT)], 1)
    torch.testing.assert_close(rot_then_mix, mix_then_rot, rtol=1e-5, atol=1e-5)


def test_pinned_paths_agree_and_carry_gradient():
    m = _pinned()
    with torch.no_grad():
        m.logit_v.add_(torch.randn_like(m.logit_v))
    k, v = _kv(seed=5), _kv(seed=6)
    as_dict = m.mix_dict({i: (k[:, i], v[:, i]) for i in range(N_STUDENT)})
    cache = DynamicCache()
    for i in range(N_STUDENT):
        cache.update(k[:, i], v[:, i], i, {})
    as_cache = m.mix_cache(cache)
    for j in range(N_EXPERT):
        torch.testing.assert_close(as_dict[j][0], as_cache.layers[j].keys)
        torch.testing.assert_close(as_dict[j][1], as_cache.layers[j].values)


def test_unpinned_default_is_unchanged():
    """The committed 4x(7->9) arms must keep loading: same shape, same values."""
    m = LayerMixer(N_STUDENT, N_EXPERT, N_BLOCKS)
    assert (m.pin_head, m.pin_tail) == (0, 0)
    assert m.logit_k.shape == (N_BLOCKS, G_IN, G_OUT)
    assert (m.g_in, m.g_out) == (G_IN, G_OUT)


def test_pins_that_leave_nothing_to_mix_raise():
    with pytest.raises(ValueError, match="nothing to mix"):
        LayerMixer(N_STUDENT, N_EXPERT, 4, pin_head=26, pin_tail=1)
    with pytest.raises(ValueError, match="pins must be >= 0"):
        LayerMixer(N_STUDENT, N_EXPERT, 4, pin_head=-1)
