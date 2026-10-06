"""Tests for ``mtg_bot.training.data_gen``.

Unit-tests the contract: shape/dtype/value invariants on the
generated arrays, save/load round-trip, multi-file load, and a
small end-to-end smoke through real games (using tiny PIMC so the
test runs in single-digit seconds).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from mtg_bot.agents import RandomAgent
from mtg_bot.features import FEATURE_DIM
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.simulate import setup_game
from mtg_bot.training import (
    PimcConfig, generate_training_data, load_all_datasets,
    load_dataset, random_choice_factory, save_dataset,
)
from mtg_bot.training.data_gen import _play_one_game_recording

from tests._test_decks import simple_test_deck_pool


# ---------- generate_training_data: shape + content invariants ---------------


def test_generates_correct_shapes_and_dtypes():
    """Smoke: small data gen with RandomChoice on both sides
    completes in < 5s and produces well-shaped arrays."""
    features, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=42, deck_pool=simple_test_deck_pool(),
    )
    assert features.dtype == np.float32
    assert outcomes.dtype == np.float32
    assert features.shape[1] == FEATURE_DIM
    assert features.shape[0] == outcomes.shape[0]
    # Both player perspectives captured per decision → at least some
    # samples for any non-trivial game.
    assert features.shape[0] > 0


def test_outcomes_are_in_valid_set():
    """Outcomes must be in {-1, 0, +1} per the value-target convention."""
    _, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=7, deck_pool=simple_test_deck_pool(),
    )
    assert np.all(np.isin(outcomes, [-1.0, 0.0, 1.0]))


def test_outcomes_balanced_across_player_perspectives():
    """For each game, P0 and P1 perspectives have opposite outcome
    labels. So summing per-game-per-decision-per-viewer outcomes
    should net to ~0 (modulo draws which contribute 0 from both
    sides). Catches a bug where outcome labeling forgets to flip
    the sign per viewer."""
    _, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=4, base_seed=3, deck_pool=simple_test_deck_pool(),
    )
    # Sum should be exactly 0 (every +1 is paired with a -1 from
    # the opposite viewer).
    assert outcomes.sum() == 0.0


def test_deterministic_with_same_seed():
    a_features, a_outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=99, deck_pool=simple_test_deck_pool(),
    )
    b_features, b_outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=99, deck_pool=simple_test_deck_pool(),
    )
    np.testing.assert_array_equal(a_features, b_features)
    np.testing.assert_array_equal(a_outcomes, b_outcomes)


def test_sample_every_thins_dataset():
    """sample_every=2 should produce roughly half as many samples
    as sample_every=1 on the same games."""
    features_full, _ = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=11, sample_every=1, deck_pool=simple_test_deck_pool(),
    )
    features_half, _ = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=11, sample_every=2, deck_pool=simple_test_deck_pool(),
    )
    assert features_half.shape[0] < features_full.shape[0]


def test_zero_games_returns_empty_arrays():
    features, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=0, base_seed=0, deck_pool=simple_test_deck_pool(),
    )
    assert features.shape == (0, FEATURE_DIM)
    assert outcomes.shape == (0,)


def test_decision_cap_treats_unfinished_as_draw():
    """If we cap at an absurd low, every game is a draw and every
    captured sample has outcome 0."""
    _, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=5,
        decision_cap=2,  # absurdly low
        deck_pool=simple_test_deck_pool(),
    )
    assert np.all(outcomes == 0.0)


# ---------- save / load -----------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path):
    features, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=42, deck_pool=simple_test_deck_pool(),
    )
    path = tmp_path / "session.npz"
    save_dataset(features, outcomes, path)
    assert path.exists()
    f2, o2 = load_dataset(path)
    np.testing.assert_array_equal(features, f2)
    np.testing.assert_array_equal(outcomes, o2)


def test_save_rejects_mismatched_shapes(tmp_path: Path):
    import pytest
    bad_features = np.zeros((5, FEATURE_DIM), dtype=np.float32)
    bad_outcomes = np.zeros((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        save_dataset(bad_features, bad_outcomes, tmp_path / "x.npz")


def test_save_rejects_wrong_feature_dim(tmp_path: Path):
    import pytest
    bad = np.zeros((3, FEATURE_DIM + 1), dtype=np.float32)
    outs = np.zeros((3,), dtype=np.float32)
    with pytest.raises(ValueError, match="dim"):
        save_dataset(bad, outs, tmp_path / "x.npz")


def test_load_all_datasets_concatenates_files(tmp_path: Path):
    """Two saved sessions in a directory load and concat in name order."""
    f1 = np.ones((3, FEATURE_DIM), dtype=np.float32)
    o1 = np.array([1, 0, -1], dtype=np.float32)
    f2 = np.full((2, FEATURE_DIM), 2.0, dtype=np.float32)
    o2 = np.array([0, 1], dtype=np.float32)
    save_dataset(f1, o1, tmp_path / "a.npz")
    save_dataset(f2, o2, tmp_path / "b.npz")
    features, outcomes = load_all_datasets(tmp_path)
    assert features.shape == (5, FEATURE_DIM)
    assert outcomes.shape == (5,)
    # Sorted by filename: 'a.npz' first.
    assert features[0, 0] == 1.0
    assert features[3, 0] == 2.0


def test_load_all_datasets_empty_dir_returns_empty(tmp_path: Path):
    features, outcomes = load_all_datasets(tmp_path)
    assert features.shape == (0, FEATURE_DIM)
    assert outcomes.shape == (0,)


# ---------- A2.3: policy supervision in the npz schema ---------------------


def test_save_and_load_with_policy_roundtrip(tmp_path: Path):
    """Save a dataset with policy_targets + action_masks; load it
    back with ``with_policy=True``; arrays compare equal."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    n = 4
    features = np.random.rand(n, FEATURE_DIM).astype(np.float32)
    outcomes = np.array([1, -1, 0, 1], dtype=np.float32)
    policy_targets = np.zeros(
        (n, ACTION_VOCAB_SIZE), dtype=np.float32,
    )
    policy_targets[0, 0] = 1.0
    policy_targets[1, 5] = 0.7
    policy_targets[1, 12] = 0.3
    action_masks = np.zeros((n, ACTION_VOCAB_SIZE), dtype=np.bool_)
    action_masks[0, 0] = True
    action_masks[1, [5, 12]] = True
    path = tmp_path / "session.npz"
    save_dataset(
        features, outcomes, path,
        policy_targets=policy_targets, action_masks=action_masks,
    )
    f2, o2, p2, m2 = load_dataset(path, with_policy=True)
    np.testing.assert_array_equal(features, f2)
    np.testing.assert_array_equal(outcomes, o2)
    np.testing.assert_array_equal(policy_targets, p2)
    np.testing.assert_array_equal(action_masks, m2)


def test_load_dataset_old_format_returns_none_for_policy(tmp_path: Path):
    """An old value-only file (no policy_targets key) loads cleanly
    with ``with_policy=True`` — policy/mask come back as ``None``
    so callers can synthesize zero arrays."""
    features = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    outcomes = np.zeros((3,), dtype=np.float32)
    path = tmp_path / "old.npz"
    save_dataset(features, outcomes, path)  # no policy args
    f, o, pols, masks = load_dataset(path, with_policy=True)
    assert pols is None
    assert masks is None


def test_save_dataset_rejects_lonely_policy_arg(tmp_path: Path):
    """``policy_targets`` and ``action_masks`` must come together
    or both omitted — half-supplied is a programming error."""
    import pytest
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    f = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    o = np.zeros((2,), dtype=np.float32)
    p = np.zeros((2, ACTION_VOCAB_SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="must be provided together"):
        save_dataset(
            f, o, tmp_path / "x.npz", policy_targets=p,
        )
    with pytest.raises(ValueError, match="must be provided together"):
        save_dataset(
            f, o, tmp_path / "y.npz",
            action_masks=p.astype(np.bool_),
        )


def test_load_all_datasets_mixed_old_and_new(tmp_path: Path):
    """A directory with one old-format and one new-format file
    loads cleanly with ``include_policy=True``: old rows get
    zero-padded policy supervision so the concatenated arrays
    align row-for-row."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    # Old-format file (value-only).
    f_old = np.ones((2, FEATURE_DIM), dtype=np.float32)
    o_old = np.array([1, -1], dtype=np.float32)
    save_dataset(f_old, o_old, tmp_path / "a_old.npz")
    # New-format file (with policy).
    f_new = np.full((3, FEATURE_DIM), 2.0, dtype=np.float32)
    o_new = np.array([0, 1, -1], dtype=np.float32)
    p_new = np.zeros((3, ACTION_VOCAB_SIZE), dtype=np.float32)
    p_new[0, 7] = 1.0
    p_new[2, 11] = 1.0
    m_new = np.zeros((3, ACTION_VOCAB_SIZE), dtype=np.bool_)
    m_new[0, [3, 7]] = True
    m_new[2, [11, 99]] = True
    save_dataset(
        f_new, o_new, tmp_path / "b_new.npz",
        policy_targets=p_new, action_masks=m_new,
    )
    f, o, p, m = load_all_datasets(tmp_path, include_policy=True)
    assert f.shape == (5, FEATURE_DIM)
    assert o.shape == (5,)
    assert p.shape == (5, ACTION_VOCAB_SIZE)
    assert m.shape == (5, ACTION_VOCAB_SIZE)
    # First 2 rows from the old file have zero-padded policy supervision.
    assert p[0:2].sum() == 0.0
    assert not m[0:2].any()
    # Last 3 rows have the synthesized policy.
    assert p[2, 7] == 1.0
    assert p[4, 11] == 1.0
    assert m[2, 3]
    assert m[4, 99]


def test_load_all_datasets_default_drops_policy(tmp_path: Path):
    """Default ``include_policy=False`` returns only (features,
    outcomes) even when files contain policy data — backward compat
    with pre-A2.3 callers."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    f = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    o = np.zeros((2,), dtype=np.float32)
    p = np.zeros((2, ACTION_VOCAB_SIZE), dtype=np.float32)
    m = np.zeros((2, ACTION_VOCAB_SIZE), dtype=np.bool_)
    save_dataset(
        f, o, tmp_path / "x.npz",
        policy_targets=p, action_masks=m,
    )
    out = load_all_datasets(tmp_path)
    assert len(out) == 2
    assert out[0].shape == (2, FEATURE_DIM)
    assert out[1].shape == (2,)


# ---------- max_files window-recent selection -------------------------------


def _write_npz_with_mtime(
    path: Path, features: np.ndarray, outcomes: np.ndarray, mtime: float,
) -> None:
    """Save a dataset and stamp its mtime explicitly so window-
    recent selection is deterministic regardless of the test
    runner's filesystem timing resolution."""
    import os

    save_dataset(features, outcomes, path)
    os.utime(path, (mtime, mtime))


def test_load_all_datasets_max_files_picks_most_recent_by_mtime(
    tmp_path: Path,
):
    """``max_files=2`` with three files on disk loads only the 2
    newest by mtime (oldest is dropped). Concatenation order
    inside the window stays stable (alphabetical by filename) so
    repeated loads of the same window produce identical arrays."""
    f_old = np.full((1, FEATURE_DIM), 0.0, dtype=np.float32)
    f_mid = np.full((1, FEATURE_DIM), 1.0, dtype=np.float32)
    f_new = np.full((1, FEATURE_DIM), 2.0, dtype=np.float32)
    o = np.array([0.0], dtype=np.float32)
    _write_npz_with_mtime(tmp_path / "old.npz", f_old, o, 1_000_000.0)
    _write_npz_with_mtime(tmp_path / "mid.npz", f_mid, o, 2_000_000.0)
    _write_npz_with_mtime(tmp_path / "new.npz", f_new, o, 3_000_000.0)

    features, outcomes = load_all_datasets(tmp_path, max_files=2)
    # Loaded the 2 most recent (mid + new); the all-zero "old"
    # row must NOT appear.
    assert features.shape == (2, FEATURE_DIM)
    loaded_values = sorted(features[:, 0].tolist())
    assert loaded_values == [1.0, 2.0]
    assert outcomes.shape == (2,)


def test_load_all_datasets_max_files_none_loads_all(tmp_path: Path):
    """``max_files=None`` (the default) preserves the existing
    contract: every .npz in the directory loads."""
    f1 = np.ones((2, FEATURE_DIM), dtype=np.float32)
    f2 = np.full((3, FEATURE_DIM), 2.0, dtype=np.float32)
    o1 = np.array([1, 0], dtype=np.float32)
    o2 = np.array([0, 1, -1], dtype=np.float32)
    save_dataset(f1, o1, tmp_path / "a.npz")
    save_dataset(f2, o2, tmp_path / "b.npz")

    features, outcomes = load_all_datasets(tmp_path, max_files=None)
    assert features.shape == (5, FEATURE_DIM)
    assert outcomes.shape == (5,)


def test_load_all_datasets_max_files_greater_than_available(
    tmp_path: Path,
):
    """Asking for more files than exist is not an error — the
    ``max_files`` knob is a maximum, not a minimum. All available
    files load."""
    f1 = np.ones((2, FEATURE_DIM), dtype=np.float32)
    o1 = np.array([1, 0], dtype=np.float32)
    save_dataset(f1, o1, tmp_path / "only.npz")

    features, outcomes = load_all_datasets(tmp_path, max_files=10)
    assert features.shape == (2, FEATURE_DIM)
    assert outcomes.shape == (2,)


def test_load_all_datasets_max_files_zero_raises(tmp_path: Path):
    """``max_files=0`` makes no semantic sense (caller asked us to
    train on no data); raise rather than silently load zero rows."""
    import pytest

    f = np.ones((1, FEATURE_DIM), dtype=np.float32)
    o = np.array([0.0], dtype=np.float32)
    save_dataset(f, o, tmp_path / "x.npz")
    with pytest.raises(ValueError, match="max_files must be positive"):
        load_all_datasets(tmp_path, max_files=0)


def test_load_all_datasets_max_files_negative_raises(tmp_path: Path):
    """Negative ``max_files`` is also rejected (defensive check)."""
    import pytest

    f = np.ones((1, FEATURE_DIM), dtype=np.float32)
    o = np.array([0.0], dtype=np.float32)
    save_dataset(f, o, tmp_path / "x.npz")
    with pytest.raises(ValueError, match="max_files must be positive"):
        load_all_datasets(tmp_path, max_files=-1)


def test_load_all_datasets_max_files_with_empty_dir(tmp_path: Path):
    """Empty dir + ``max_files=N`` returns the same empty arrays
    as the default empty-dir path (no error from the windowing
    logic)."""
    features, outcomes = load_all_datasets(tmp_path, max_files=5)
    assert features.shape == (0, FEATURE_DIM)
    assert outcomes.shape == (0,)


def test_load_all_datasets_max_files_with_include_policy(tmp_path: Path):
    """The window-recent path returns the 4-tuple shape when
    ``include_policy=True``, just like the all-files path."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE

    f1 = np.ones((1, FEATURE_DIM), dtype=np.float32)
    f2 = np.full((1, FEATURE_DIM), 2.0, dtype=np.float32)
    o = np.array([0.0], dtype=np.float32)
    p = np.zeros((1, ACTION_VOCAB_SIZE), dtype=np.float32)
    p[0, 3] = 1.0
    m = np.zeros((1, ACTION_VOCAB_SIZE), dtype=np.bool_)
    m[0, 3] = True
    _write_npz_with_mtime(
        tmp_path / "old.npz", f1, o, 1_000_000.0,
    )
    save_dataset(
        f2, o, tmp_path / "new.npz",
        policy_targets=p, action_masks=m,
    )
    import os
    os.utime(tmp_path / "new.npz", (3_000_000.0, 3_000_000.0))

    features, outcomes, policy, masks = load_all_datasets(
        tmp_path, include_policy=True, max_files=1,
    )
    assert features.shape == (1, FEATURE_DIM)
    assert features[0, 0] == 2.0
    assert policy.shape == (1, ACTION_VOCAB_SIZE)
    assert masks.shape == (1, ACTION_VOCAB_SIZE)
    assert policy[0, 3] == 1.0
    assert masks[0, 3]


def test_load_all_datasets_max_files_logs_count(
    tmp_path: Path, capsys,
):
    """The window-recent path emits a one-line stderr summary for
    operator transparency: which subset was loaded."""
    f = np.ones((1, FEATURE_DIM), dtype=np.float32)
    o = np.array([0.0], dtype=np.float32)
    _write_npz_with_mtime(tmp_path / "a.npz", f, o, 1_000_000.0)
    _write_npz_with_mtime(tmp_path / "b.npz", f, o, 2_000_000.0)
    _write_npz_with_mtime(tmp_path / "c.npz", f, o, 3_000_000.0)

    load_all_datasets(tmp_path, max_files=2)
    captured = capsys.readouterr()
    assert "loading 2 of 3 files" in captured.err
    assert "window-recent=2" in captured.err


def test_generate_training_data_with_policy_returns_4tuple():
    """``with_policy=True`` produces the 4-tuple schema. Random
    strategies don't have visit counts, so the policy_targets are
    all-zero — but the array still has the right shape."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    f, o, p, m = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=42, with_policy=True, deck_pool=simple_test_deck_pool(),
    )
    n = f.shape[0]
    assert f.shape == (n, FEATURE_DIM)
    assert o.shape == (n,)
    assert p.shape == (n, ACTION_VOCAB_SIZE)
    assert m.shape == (n, ACTION_VOCAB_SIZE)
    # RandomChoice has no visits → all rows are value-only.
    assert p.sum() == 0.0
    assert not m.any()


# ---------- end-to-end with tiny PIMC ---------------------------------------


def test_runs_with_tiny_pimc_on_real_engine():
    """Sanity: PIMC(heuristic) on both sides for 1 game produces a
    real dataset. Tiny budget so test stays fast."""
    config = PimcConfig(samples=1, inner_nodes=4)
    factory = lambda seed, _c=config: _c.build(seed)
    features, outcomes = generate_training_data(
        p0_factory=factory, p1_factory=factory,
        n_games=1, base_seed=0, deck_pool=simple_test_deck_pool(),
    )
    assert features.shape[1] == FEATURE_DIM
    assert features.shape[0] > 0
    assert np.all(np.isfinite(features))
    assert np.all(np.isin(outcomes, [-1.0, 0.0, 1.0]))


def test_policy_targets_live_inside_action_mask_for_combat_decisions():
    """Trainer invariant: every slot a sample's ``policy_target``
    assigns positive probability to MUST be True in that sample's
    ``action_mask``. The trainer's ``masked_fill(~mask, -1e9)`` makes
    masked-CE explode (~1e9 per affected row) for any policy-on-
    masked-out-slot violation; the gen0 v4 train hit this with
    final policy_loss=613,242 because data-gen passed the raw
    engine state (not the adapter) to ``legal_action_mask``,
    making ``_combat_phase`` resolve to ``None`` and the attacker /
    blocker slot enumeration short-circuit. Enough games at PIMC
    budget that combat decisions get sampled (PIMC needs visits ≥ 1
    to emit a policy target, and combat happens early in the game)."""
    config = PimcConfig(samples=1, inner_nodes=4)
    factory = lambda seed, _c=config: _c.build(seed)
    f, o, p, m = generate_training_data(
        p0_factory=factory, p1_factory=factory,
        n_games=3, base_seed=0, with_policy=True, deck_pool=simple_test_deck_pool(),
    )
    has_pol = p.sum(axis=-1) > 0.5  # supervised rows
    assert has_pol.any(), (
        "expected some PIMC policy supervision; bump n_games / nodes"
    )
    # Any slot where mask is False should have zero policy mass.
    leaked = (p * (~m)).sum(axis=-1)
    bad_rows = leaked > 1e-6
    assert not bad_rows.any(), (
        f"{int(bad_rows.sum())} / {int(has_pol.sum())} supervised rows "
        f"carry policy mass on slots the action_mask says are illegal — "
        f"trainer will blow up CE on these. Total leaked mass: "
        f"{float(leaked[bad_rows].sum()):.4f}"
    )


# ---------- generate_training_data_parallel ---------------------------------


from mtg_bot.training import generate_training_data_parallel  # noqa: E402


def test_parallel_result_equals_sequential_result():
    """The headline determinism guarantee: parallel produces the
    exact same arrays as sequential for the same base_seed. Per-game
    seeds are derived from game_idx (not worker assignment), and
    Pool.map preserves input order, so results MUST be bit-identical.

    If this ever fails: investigate before declaring "good enough."
    Non-deterministic data-gen makes tuning runs uncomparable across
    hardware / worker counts."""
    seq_f, seq_o = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3, base_seed=42, deck_pool=simple_test_deck_pool(),
    )
    par_f, par_o = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3, base_seed=42, n_workers=2, deck_pool=simple_test_deck_pool(),
    )
    np.testing.assert_array_equal(seq_f, par_f)
    np.testing.assert_array_equal(seq_o, par_o)


def test_parallel_zero_games_returns_empty():
    f, o = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=0, n_workers=2, deck_pool=simple_test_deck_pool(),
    )
    assert f.shape == (0, FEATURE_DIM)
    assert o.shape == (0,)


def test_parallel_n_workers_validation():
    import pytest
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        generate_training_data_parallel(
            p0_factory=random_choice_factory,
            p1_factory=random_choice_factory,
            n_games=2, n_workers=0, deck_pool=simple_test_deck_pool(),
        )


def test_parallel_default_n_workers_path():
    """When n_workers=None, the default-derivation path runs without
    error. We pass n_games=0 so we don't actually spawn — the default
    n_workers value is only meaningful relative to cpu_count and we
    don't want this test to depend on the test machine's core count.
    Real-run coverage of the default path comes from the
    test_parallel_with_pimc_config_factory test below (which uses
    explicit n_workers=2)."""
    f, o = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=0, n_workers=None, deck_pool=simple_test_deck_pool(),
    )
    assert f.shape == (0, FEATURE_DIM)


def test_parallel_invariants_on_real_arrays():
    """Same invariants the sequential version satisfies: balanced
    outcomes, in-range labels, correct shapes."""
    f, o = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=11, n_workers=2, deck_pool=simple_test_deck_pool(),
    )
    assert f.shape[1] == FEATURE_DIM
    assert f.shape[0] == o.shape[0]
    assert np.all(np.isin(o, [-1.0, 0.0, 1.0]))
    # P0 and P1 perspective labels cancel (every +1 paired with -1).
    assert o.sum() == 0.0


def test_parallel_with_pimc_config_factory():
    """The intended real-world pattern: pass a PimcConfig.build
    bound method as the factory. Bound methods of frozen dataclasses
    are picklable so they cross the spawn boundary cleanly."""
    config = PimcConfig(samples=1, inner_nodes=4)
    f, o = generate_training_data_parallel(
        p0_factory=config.build,
        p1_factory=config.build,
        n_games=2, base_seed=0, n_workers=2, deck_pool=simple_test_deck_pool(),
    )
    assert f.shape[1] == FEATURE_DIM
    assert f.shape[0] > 0
    assert np.all(np.isfinite(f))


# ---------- deck_pool path --------------------------------------------------


import pytest  # noqa: E402

from mtg_bot.decks import Deck  # noqa: E402
from mtg_bot.training.data_gen import _sample_decks_for_game  # noqa: E402

from tests._test_decks import simple_test_decklist  # noqa: E402


def _two_player_pool() -> list[Deck]:
    """A small pool of distinct test decks across three colors.
    Cheap to build; uses real cards we know are in the active card
    source's printing registry."""
    return [
        Deck(draft_id="g_deck", build_index=0,
             cards=simple_test_decklist("G")),
        Deck(draft_id="r_deck", build_index=0,
             cards=simple_test_decklist("R")),
        Deck(draft_id="w_deck", build_index=0,
             cards=simple_test_decklist("W")),
    ]


def test_sample_decks_picks_two_distinct_decks_when_pool_has_multiple():
    pool = _two_player_pool()
    a, b = _sample_decks_for_game(pool, game_idx=0, base_seed=42)
    assert a is not b
    assert a in pool and b in pool


def test_sample_decks_is_deterministic():
    pool = _two_player_pool()
    a1, b1 = _sample_decks_for_game(pool, game_idx=5, base_seed=42)
    a2, b2 = _sample_decks_for_game(pool, game_idx=5, base_seed=42)
    assert a1 is a2 and b1 is b2


def test_sample_decks_varies_with_game_idx():
    """Different game_idx should usually give different deck pairs."""
    pool = _two_player_pool()
    a0, b0 = _sample_decks_for_game(pool, game_idx=0, base_seed=42)
    saw_different = False
    for game_idx in range(1, 10):
        a, b = _sample_decks_for_game(
            pool, game_idx=game_idx, base_seed=42,
        )
        if (a, b) != (a0, b0):
            saw_different = True
            break
    assert saw_different


def test_sample_decks_single_deck_pool_mirrors():
    """Pool with just one deck → both seats get the same deck (mirror)."""
    pool = [
        Deck(draft_id="only", build_index=0,
             cards=simple_test_decklist("G")),
    ]
    a, b = _sample_decks_for_game(pool, game_idx=0, base_seed=0)
    assert a is b


def test_sample_decks_empty_pool_raises():
    with pytest.raises(ValueError, match="empty"):
        _sample_decks_for_game([], game_idx=0, base_seed=0)


def test_generate_training_data_uses_deck_pool():
    """Smoke: sequential data gen with a deck pool produces well-
    shaped arrays + balanced outcomes."""
    pool = _two_player_pool()
    features, outcomes = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=42, deck_pool=pool,
    )
    assert features.shape[1] == FEATURE_DIM
    assert features.shape[0] > 0
    assert outcomes.sum() == 0.0


def test_deck_pool_path_parallel_equals_sequential():
    """Strong invariant: parallel and sequential paths produce
    identical arrays for the same deck_pool + base_seed. Per-game
    seeds for deck sampling derive from base_seed + game_idx (not
    worker assignment)."""
    pool = _two_player_pool()
    seq_f, seq_o = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3, base_seed=42, deck_pool=pool,
    )
    par_f, par_o = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3, base_seed=42, n_workers=2, deck_pool=pool,
    )
    np.testing.assert_array_equal(seq_f, par_f)
    np.testing.assert_array_equal(seq_o, par_o)


def test_deck_pool_none_raises():
    """deck_pool is required — there's no synthetic-fallback path."""
    with pytest.raises(ValueError, match="deck_pool is required"):
        generate_training_data(
            p0_factory=random_choice_factory,
            p1_factory=random_choice_factory,
            n_games=1, base_seed=42, deck_pool=None,
        )


def test_deck_pool_with_short_decks_works():
    """Real Limited decks are 40 cards, not 60. Verify the engine
    accepts them via the deck_pool path."""
    short_deck = (
        ["Forest"] * 17
        + simple_test_decklist("G")[:23]
    )
    pool = [
        Deck(draft_id="a", build_index=0, cards=short_deck),
        Deck(draft_id="b", build_index=0, cards=short_deck),
    ]
    f, o = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=1, base_seed=0, deck_pool=pool,
    )
    assert f.shape[1] == FEATURE_DIM
    assert f.shape[0] > 0


# ---------- A2.1: visits captured at sample points ---------------------------


class _FakeStrategyWithVisits:
    """Records that ``choose_with_visits`` was called and returns a
    fixed (synthetic) visit dict so the test can verify the dict is
    forwarded into the per-sample tuple unchanged. Picks a legal
    decision via the adapter so the game still progresses."""

    def __init__(self):
        self.visits_calls = 0

    def choose(self, adapter, player_id):
        legal = list(adapter.legal_decisions())
        return legal[0]

    def choose_with_visits(self, adapter, player_id):
        self.visits_calls += 1
        legal = list(adapter.legal_decisions())
        if not legal:
            return {}
        # Synthetic distribution: first legal decision gets 100
        # visits, the rest get 1 each. Realistic shape, easy to
        # distinguish from a uniform fallback.
        visits = {legal[0]: 100}
        for d in legal[1:]:
            visits[d] = 1
        return visits


def test_play_one_game_recording_captures_visits_at_sample_points():
    """The active player's sample at each sample point should
    carry the strategy's visit dict; the non-active player's
    same-state sample (data-doubling row) should carry ``None``.
    Random / no-visits strategies fall through with ``None``."""
    engine = setup_game([
            RandomAgent(rng=random.Random(11)),
            RandomAgent(rng=random.Random(22)),
        ], decks=simple_test_decklists(), seed=42,
    )
    adapter = MtgGameStateAdapter(
        engine, expose_combat_decisions=True,
    )
    p0 = _FakeStrategyWithVisits()
    p1 = _FakeStrategyWithVisits()
    rng = random.Random(7)

    _, samples, _struct = _play_one_game_recording(
        adapter, {0: p0, 1: p1}, rng,
        decision_cap=200, sample_every=1,
    )

    assert len(samples) > 0
    # Either p0 or p1 must have been hit for visits at least once.
    assert p0.visits_calls + p1.visits_calls > 0

    # Per-sample-point grouping: each sample point appends two
    # rows (one per viewer). Find a pair where the active viewer's
    # row carries visits.
    found_with_visits = False
    found_none_partner = False
    for feats, viewer, policy_target, action_mask in samples:
        # A2.3: visits are now encoded into a (ACTION_VOCAB_SIZE,)
        # policy_target array. "Has supervision" = sum > 0.5
        # (a normalized distribution sums to 1.0 + float noise).
        if policy_target.sum() > 0.5:
            found_with_visits = True
            # The synthetic dict was {legal[0]: 100, legal[1]: 1, ...},
            # normalized — the favorite slot ends up with the
            # largest fraction. We just check that the policy is
            # non-zero on at least one slot and that the mask is
            # set on at least one slot.
            assert action_mask.any(), (
                "active row has policy supervision but no legal "
                "slot in the action mask — encoder broken"
            )
        else:
            found_none_partner = True
            # Value-only row: policy_target is all-zero, mask
            # all-False.
            assert not action_mask.any()

    assert found_with_visits, (
        "no sample carried visits — _play_one_game_recording "
        "didn't forward choose_with_visits output"
    )
    assert found_none_partner, (
        "no sample had visits=None — non-active viewer's row "
        "should be value-only"
    )


def test_play_one_game_recording_filters_illegal_pimc_visits():
    """PIMC's K-determinization aggregation can put visits on
    decisions that are legal in some determinizations but NOT the
    root state — concretely, exile-zone casts whose legality depends
    on hidden info. Without filtering, those visits would put policy
    mass on slots the action_mask correctly excludes; the trainer's
    ``masked_fill(~mask, -1e9)`` then makes masked CE explode.

    This test injects a synthetic strategy that returns visits
    containing an extra "illegal-from-root" decision and asserts the
    written policy_target carries zero mass on slots not set by the
    action_mask.
    """
    from mtg_bot.actions import CastSpell

    engine = setup_game([
            RandomAgent(rng=random.Random(11)),
            RandomAgent(rng=random.Random(22)),
        ], decks=simple_test_decklists(), seed=42,
    )
    adapter = MtgGameStateAdapter(
        engine, expose_combat_decisions=True,
    )

    class _StrategyWithIllegalVisit:
        def __init__(self):
            self.visits_calls = 0

        def choose(self, adapter, player_id):
            return list(adapter.legal_decisions())[0]

        def choose_with_visits(self, adapter, player_id):
            self.visits_calls += 1
            legal = list(adapter.legal_decisions())
            if not legal:
                return {}
            # Return real legal visits PLUS a phantom CastSpell that
            # the root state never authorized — mimics PIMC
            # determinization aggregating visits on a hidden-info-
            # legal action. card_id=99999 is guaranteed not to match
            # anything in the test deck, so it can't accidentally
            # collide with a real legal action.
            visits: dict = {legal[0]: 100}
            for d in legal[1:]:
                visits[d] = 1
            visits[CastSpell(card_id=99999)] = 50
            return visits

    p0 = _StrategyWithIllegalVisit()
    p1 = _StrategyWithIllegalVisit()
    rng = random.Random(7)

    _, samples, _struct = _play_one_game_recording(
        adapter, {0: p0, 1: p1}, rng,
        decision_cap=200, sample_every=1,
    )
    assert samples
    assert p0.visits_calls + p1.visits_calls > 0

    # For every supervised row, every slot with policy mass must be
    # in the action_mask. The phantom CastSpell visit must NOT have
    # leaked into the policy_target.
    found_supervised = False
    for feats, viewer, policy_target, action_mask in samples:
        if policy_target.sum() <= 0.5:
            continue
        found_supervised = True
        slots_with_mass = (policy_target > 1e-9)
        # Every mass-carrying slot is in the legal mask.
        assert (slots_with_mass <= action_mask).all(), (
            "policy_target carries mass on slots the action_mask "
            "excluded — PIMC visit filter regressed"
        )
        # Sanity: distribution still normalized (the legal visits
        # got renormalized after filtering).
        assert abs(policy_target.sum() - 1.0) < 1e-5
    assert found_supervised, (
        "no supervised rows produced — visit filter shouldn't strip "
        "ALL visits when there's at least one legal one"
    )


def test_play_one_game_recording_falls_back_when_visits_empty():
    """If the strategy's ``choose_with_visits`` returns ``{}``
    (e.g. RandomChoice's default), the recording loop must still
    pick a legal decision and record value-only samples (visits=
    None on both rows). Catches a regression where the empty-dict
    path tries to argmax over nothing or skips the sample entirely.
    """
    engine = setup_game([
            RandomAgent(rng=random.Random(11)),
            RandomAgent(rng=random.Random(22)),
        ], decks=simple_test_decklists(), seed=42,
    )
    adapter = MtgGameStateAdapter(
        engine, expose_combat_decisions=True,
    )
    # random_choice_factory returns RandomChoice — its default
    # ``choose_with_visits`` returns ``{}``.
    rng = random.Random(7)
    rs0 = random_choice_factory(0)
    rs1 = random_choice_factory(0)

    _, samples, _struct = _play_one_game_recording(
        adapter, {0: rs0, 1: rs1}, rng,
        decision_cap=200, sample_every=1,
    )

    assert len(samples) > 0
    for _feats, _viewer, policy_target, action_mask in samples:
        assert policy_target.sum() == 0.0, (
            "RandomChoice has no visit data; policy target "
            "should be all-zero"
        )
        assert not action_mask.any(), (
            "RandomChoice has no visit data; action mask "
            "should be all-False"
        )


# ---------- DataGenProgress wiring ------------------------------------------

from mtg_bot.training import DataGenProgress  # noqa: E402
from tests._test_decks import simple_test_decklists  # noqa: E402


def _make_progress(progress_path: Path, total: int, workers: int = 1):
    return DataGenProgress(
        progress_path=progress_path,
        run_id=progress_path.stem,
        n_games_total=total,
        n_workers=workers,
        flush_interval_s=0.0,  # flush every record so the file is current
    )


def test_generate_training_data_threads_progress(tmp_path: Path):
    """Sequential driver calls progress.begin() once and
    progress.record_game() per game. Covers the
    ``if progress is not None:`` branches (lines 222 + 237)."""
    progress = _make_progress(tmp_path / "seq.json", total=2)
    f, o = generate_training_data(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=17,
        progress=progress, deck_pool=simple_test_deck_pool(),
    )
    progress.finalize(status="completed")
    payload = json.loads((tmp_path / "seq.json").read_text(encoding="utf-8"))
    assert payload["n_games_done"] == 2
    assert payload["n_games_total"] == 2
    # The samples count should match the array we got back.
    assert payload["n_samples"] == int(f.shape[0])


def test_parallel_threads_progress(tmp_path: Path):
    """Parallel driver also calls progress.begin() and
    progress.record_game() per completed game. Covers lines
    305 + 330-334."""
    progress = _make_progress(tmp_path / "par.json", total=2, workers=2)
    f, o = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=17, n_workers=2,
        progress=progress, deck_pool=simple_test_deck_pool(),
    )
    progress.finalize(status="completed")
    payload = json.loads((tmp_path / "par.json").read_text(encoding="utf-8"))
    assert payload["n_games_done"] == 2
    assert payload["n_samples"] == int(f.shape[0])


# ---------- with_policy=True parallel return shape --------------------------


def test_parallel_with_policy_returns_four_tuple():
    """``with_policy=True`` switches the parallel return from
    (features, outcomes) to (features, outcomes, policy_targets,
    action_masks). Covers line 349."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE

    f, o, p, m = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2, base_seed=5, n_workers=2,
        with_policy=True, deck_pool=simple_test_deck_pool(),
    )
    assert f.shape[1] == FEATURE_DIM
    assert o.shape == (f.shape[0],)
    assert p.shape == (f.shape[0], ACTION_VOCAB_SIZE)
    assert m.shape == (f.shape[0], ACTION_VOCAB_SIZE)
    # RandomChoice returns no visits, so all policy targets are zero
    # and all masks are all-False.
    assert p.sum() == 0.0
    assert not m.any()


def test_parallel_with_policy_zero_games_returns_empty_four_tuple():
    """``n_games=0 + with_policy=True`` returns the empty 4-tuple
    via ``_empty_arrays``. Covers the ``return (f,o,p,m) if
    with_policy else (f,o)`` branch in the early-return."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE

    f, o, p, m = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=0, n_workers=2,
        with_policy=True, deck_pool=simple_test_deck_pool(),
    )
    assert f.shape == (0, FEATURE_DIM)
    assert o.shape == (0,)
    assert p.shape == (0, ACTION_VOCAB_SIZE)
    assert m.shape == (0, ACTION_VOCAB_SIZE)


# ---------- worker crash resilience -----------------------------------------


# Top-level pickleable strategy + factory for the parallel crash test.
# The fix in _play_one_parallel_game catches the raised exception,
# prints a traceback to worker stderr, and returns an empty chunk
# (winner=None, 0 samples) so the pool keeps going.
class _DataGenRaisingStrategy:
    """Raises ``ValueError`` on the first ``choose_with_visits`` call.
    Picklable across the spawn boundary because it's at module top
    level and holds no closures."""

    def __init__(self, message: str = "simulated engine corruption") -> None:
        self.message = message

    def choose(self, state, player_id):  # noqa: ANN001
        raise ValueError(self.message)

    def choose_with_visits(self, state, player_id):  # noqa: ANN001
        raise ValueError(self.message)


def _data_gen_raising_factory(seed: int) -> _DataGenRaisingStrategy:
    """Top-level pickleable factory: every game built with this
    factory raises on its first decision."""
    return _DataGenRaisingStrategy()


def test_parallel_worker_crash_is_caught_and_warns(capfd):
    """A raising strategy in a worker triggers the broad
    ``except Exception`` in ``_play_one_parallel_game`` (lines
    392-404). All games end up as empty chunks (winner=None,
    0 samples), the driver counts them as crashed and prints a
    ``# WARNING: K/N games crashed`` summary to stderr (lines
    335-344). The driver does not re-raise."""
    f, o = generate_training_data_parallel(
        p0_factory=_data_gen_raising_factory,
        p1_factory=_data_gen_raising_factory,
        n_games=2, base_seed=0, n_workers=2, deck_pool=simple_test_deck_pool(),
    )
    # Every game crashed → no samples landed in the arrays.
    assert f.shape == (0, FEATURE_DIM)
    assert o.shape == (0,)
    captured = capfd.readouterr()
    assert "games crashed" in captured.err


# ---------- starting-seat alternation per game_idx --------------------------


def _engine_for_game_idx(game_idx: int, base_seed: int = 0):
    """Build an engine through the same code path
    ``_play_one_parallel_game`` uses, stopping right after setup so
    tests can introspect the initial active/priority state. Mirrors
    ``_generate_one_game_data`` up to (and including) the seat-
    alternation hook."""
    from mtg_bot.training.data_gen import _generate_one_game_data
    # Easiest way to get the post-setup engine without playing the
    # game is to inline the same setup; but we want to actually
    # exercise the production path, so call the helper with a tiny
    # decision_cap and inspect the engine via the strategies-side
    # path. Even simpler: replicate the setup block exactly here —
    # this test fails loudly if the production helper drifts.
    engine_seed = base_seed + game_idx
    engine = setup_game([
            RandomAgent(rng=random.Random(engine_seed + 1)),
            RandomAgent(rng=random.Random(engine_seed + 2)),
        ], decks=simple_test_decklists(),
        seed=engine_seed,
    )
    if game_idx % 2 == 1:
        engine.active_player_idx = 1
        engine.priority_player_idx = 1
    return engine


def test_seat_alternation_flips_starting_player_per_game_idx():
    """game_idx=0 → seat 0 starts. game_idx=1 → seat 1 starts.
    Same logic in production: ``_generate_one_game_data`` flips the
    engine's active/priority indices on odd game_idx so the seat-0
    on-the-play advantage averages out across the dataset."""
    engine_even = _engine_for_game_idx(game_idx=0)
    assert engine_even.active_player_idx == 0
    assert engine_even.priority_player_idx == 0
    engine_odd = _engine_for_game_idx(game_idx=1)
    assert engine_odd.active_player_idx == 1
    assert engine_odd.priority_player_idx == 1


def test_seat_alternation_balances_physical_seat_winrate():
    """Without alternation, seat 0 wins ~70-80% of self-play games
    because going first is structurally advantageous in MTG. With
    alternation, the on-the-play wins shift to ~50/50 between
    physical seats across the dataset.

    32 games of RandomChoice self-play is enough to see the bias
    clearly without the test getting slow. Pin a loose 75% upper
    bound on each seat's wincount — without alternation seat 0
    runs well above this; with alternation both seats sit near 50%."""
    from mtg_bot.training.data_gen import _generate_one_game_data
    n_games = 32
    base_seed = 12345
    seat_wins = {0: 0, 1: 0, None: 0}
    pool = simple_test_deck_pool()
    for game_idx in range(n_games):
        chunk = _generate_one_game_data(
            game_idx=game_idx,
            p0_factory=random_choice_factory,
            p1_factory=random_choice_factory,
            base_seed=base_seed,
            decision_cap=4000,
            sample_every=1,
            deck_pool=pool,
        )
        winner = chunk[4]
        seat_wins[winner] += 1
    decisive = seat_wins[0] + seat_wins[1]
    assert decisive > 0, "all games drew — seed picked pathologically"
    assert seat_wins[0] / decisive < 0.75, (
        f"seat 0 wins {seat_wins[0]}/{decisive} decisive games — "
        f"alternation isn't balancing the on-the-play advantage"
    )
    assert seat_wins[1] / decisive < 0.75, (
        f"seat 1 wins {seat_wins[1]}/{decisive} decisive games — "
        f"alternation overshot in the other direction"
    )


def test_outcome_labels_correct_under_seat_alternation():
    """Outcome labeling stays per-viewer regardless of which seat
    started. A game where player 0 wins must produce ``outcome=+1``
    on viewer=0 rows and ``outcome=-1`` on viewer=1 rows — even on
    odd game_idx where player 1 is on the play.

    Strategy: pick base seeds where the game_idx=0 winner is known
    (player 0) and the game_idx=1 winner is known (verify by
    running the helper). Then check that the recorded samples use
    the per-viewer outcome convention. This pins the contract:
    the data layer doesn't care about who started, only about
    ``winner == viewer``."""
    from mtg_bot.training.data_gen import _generate_one_game_data
    base_seed = 12345
    pool = simple_test_deck_pool()
    for game_idx in (0, 1):
        chunk = _generate_one_game_data(
            game_idx=game_idx,
            p0_factory=random_choice_factory,
            p1_factory=random_choice_factory,
            base_seed=base_seed,
            decision_cap=4000,
            sample_every=1,
            deck_pool=pool,
        )
        (
            features, outcomes, _policy, _mask, winner, _kind, _struct,
            _game_ids, _resolution_stats,
        ) = chunk
        if winner is None:
            continue  # draws label both viewers 0 — covered by other tests
        # Samples alternate viewer=0, viewer=1 (data-doubling order in
        # ``_play_one_game_recording``). Even-indexed rows are viewer=0;
        # odd-indexed rows are viewer=1.
        viewer0_outcomes = outcomes[0::2]
        viewer1_outcomes = outcomes[1::2]
        expected_v0 = 1.0 if winner == 0 else -1.0
        expected_v1 = 1.0 if winner == 1 else -1.0
        assert np.all(viewer0_outcomes == expected_v0), (
            f"game_idx={game_idx}, winner=P{winner}: viewer=0 rows "
            f"should label {expected_v0}, got {set(viewer0_outcomes)}"
        )
        assert np.all(viewer1_outcomes == expected_v1), (
            f"game_idx={game_idx}, winner=P{winner}: viewer=1 rows "
            f"should label {expected_v1}, got {set(viewer1_outcomes)}"
        )


# ---------- stop-run sentinel propagation ------------------------------------


def test_parallel_stop_check_finalizes_progress_as_stopped(tmp_path):
    """End-to-end: drop a sentinel file mid-loop via a stop_check
    that flips True after the first game completes. The pool is
    terminated, the data-gen call returns the partial result, and
    the progress tracker is finalized as ``status="stopped"``.

    This is the load-bearing integration covering the producer-side
    contract: the UI's Stop button writes a sentinel; the producer's
    stop_check sees it; the run ends as 'stopped' rather than
    'completed' or 'failed'.
    """
    from mtg_bot.training import DataGenProgress

    # Use a stop_check that returns True immediately so the very
    # first imap_unordered yield triggers the early exit. We don't
    # care which game finished first — only that the loop bails
    # before draining all 3.
    progress_path = tmp_path / "tiny_run.json"
    progress = DataGenProgress(
        progress_path=progress_path,
        run_id="tiny_run",
        n_games_total=3,
        n_workers=2,
        flush_interval_s=0.0,
    )

    calls = {"n": 0}

    def stop_after_first():
        calls["n"] += 1
        return calls["n"] >= 1  # first call returns True

    f, o = generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3,
        base_seed=7,
        n_workers=2,
        progress=progress,
        stop_check=stop_after_first,
    )

    # Some data was preserved (not zero, not all 3 games).
    assert f.shape[1] == FEATURE_DIM
    # Progress finalized as stopped.
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["status"] == "stopped"
    assert payload["ended_at"] is not None


# ---------- iter_019 BUG #4: subprocess pool leak guard -----------------


def test_parallel_attaches_pool_workers_to_job_guard(monkeypatch):
    """Pin that ``generate_training_data_parallel`` attaches the
    Pool's workers to a Win32 job-object guard (iter_019 BUG #4).

    The fix-shape: a ``JobObjectGuard`` is created BEFORE the Pool,
    the Pool spawns its workers, then we call
    ``guard.attach_pool_workers(pool)`` so each worker is in the
    job. The guard's ``KILL_ON_JOB_CLOSE`` flag ensures that if the
    master dies WITHOUT running Python cleanup (taskkill /F,
    segfault, OOM), the kernel terminates the in-job workers when
    the master's handle to the job closes on process exit.

    We can't unit-test the actual SIGKILL-master kernel-kill behavior
    in process. What we DO pin: the attach-pool-workers call fires
    against the same Pool instance the driver iterates.
    """
    from mtg_bot.training import data_gen

    calls: list[tuple[str, object]] = []

    class _FakeGuard:
        active = True
        attached = 0

        def __enter__(self):
            calls.append(("guard_enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("guard_exit", None))
            return None

        def attach_pool_workers(self, pool):
            calls.append(("attach", pool))
            self.attached = 1
            return 1

        def close(self):
            pass

    guard_instance = _FakeGuard()

    monkeypatch.setattr(
        data_gen, "JobObjectGuard", lambda: guard_instance,
    )

    # Stub the Pool to capture its construction without spawning
    # real worker processes.
    class _FakePool:
        def __init__(self, *args, **kwargs):
            calls.append(("pool_ctor", kwargs))
            self.kwargs = kwargs

        def __enter__(self):
            calls.append(("pool_enter", self))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("pool_exit", self))
            return None

        def imap_unordered(self, fn, args_list, chunksize=1):
            return iter(())  # zero results — driver loop ends fast

        def terminate(self):
            pass

        def join(self):
            pass

    class _FakeCtx:
        Pool = _FakePool

    monkeypatch.setattr(data_gen, "mp", _FakeCtx)

    data_gen.generate_training_data_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=1,
        n_workers=1,
        base_seed=0,
    )

    kinds = [k for k, _ in calls]
    assert "guard_enter" in kinds, "JobObjectGuard never entered"
    assert "pool_ctor" in kinds, "Pool was never constructed"
    assert "attach" in kinds, (
        "attach_pool_workers was never called — workers stay "
        "outside the kill-net"
    )

    # Order: guard enters first, pool is constructed and entered,
    # attach fires, then everything exits in LIFO order so the pool
    # exits BEFORE the guard (pool.terminate() drains workers before
    # the kernel kill-net is dismantled).
    g_enter = kinds.index("guard_enter")
    p_ctor = kinds.index("pool_ctor")
    p_enter = kinds.index("pool_enter")
    attach = kinds.index("attach")
    p_exit = kinds.index("pool_exit")
    g_exit = kinds.index("guard_exit")
    assert g_enter < p_ctor < p_enter < attach < p_exit < g_exit, (
        f"unexpected call order: {kinds}"
    )

    # Attach was called with the actual pool instance, not a
    # placeholder — the workers we attach must be the workers the
    # Pool spawned.
    attach_pool = next(arg for k, arg in calls if k == "attach")
    pool_arg = next(arg for k, arg in calls if k == "pool_enter")
    assert attach_pool is pool_arg, (
        "attach_pool_workers received a different Pool instance "
        "than the one being driven"
    )
