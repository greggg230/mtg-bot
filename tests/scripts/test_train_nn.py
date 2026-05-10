"""Tests for ``mtg_bot.cli.train`` (the runner formerly known as
``scripts/train_nn.py``; the script is now a thin shim).

Pins the ``--one-legal-keep-frac`` flag plumbing and the
``_downsample_one_legal_rows`` helper.

The downsample helper is the policy-head-investigation fix #2:
~84% of supervised rows in iter_002/iter_003 training data are
one-legal-action (CE=0 trivially), pinning the policy head's
plateau at ~0.157. Dropping 90% of those one-legal rows
rebalances the gradient toward branchy multi-legal rows where
the MCTS visit-count distribution actually carries signal.

The CLI-level argparse validation is exercised via subprocess
through the legacy shim path (``scripts/train_nn.py``), which
``iterate_alphazero`` still spawns. The helper is tested directly
with synthetic arrays via the cli module.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "train_nn.py"


def _load_train_nn():
    """Fresh-import ``mtg_bot.cli.train`` — the live module the shim
    routes into. ``_downsample_one_legal_rows`` and friends live here."""
    src_dir = _REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if "mtg_bot.cli.train" in sys.modules:
        del sys.modules["mtg_bot.cli.train"]
    return importlib.import_module("mtg_bot.cli.train")


def _build_synthetic(
    n_one_legal_supervised: int,
    n_multi_legal_supervised: int,
    n_value_only: int,
    *,
    seed: int = 0,
):
    """Build a synthetic dataset with three row classes:

    - ``n_one_legal_supervised``: action_mask has exactly one slot,
      policy_target sums to 1.0 (branch-trivial).
    - ``n_multi_legal_supervised``: action_mask has multiple slots,
      policy_target distribution sums to 1.0 (branchy — real signal).
    - ``n_value_only``: action_mask all-False, policy_target all-zero
      (the non-active-viewer copy or a worker that didn't record visits).

    Returns ``(features, outcomes, policy_targets, action_masks)``.
    """
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    from mtg_bot.features import FEATURE_DIM

    n_total = (
        n_one_legal_supervised
        + n_multi_legal_supervised
        + n_value_only
    )
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_total, FEATURE_DIM)).astype(np.float32)
    outcomes = rng.standard_normal(n_total).astype(np.float32)
    policy_targets = np.zeros(
        (n_total, ACTION_VOCAB_SIZE), dtype=np.float32,
    )
    action_masks = np.zeros((n_total, ACTION_VOCAB_SIZE), dtype=np.bool_)

    row = 0
    for _ in range(n_one_legal_supervised):
        slot = row % ACTION_VOCAB_SIZE
        action_masks[row, slot] = True
        policy_targets[row, slot] = 1.0
        row += 1
    for _ in range(n_multi_legal_supervised):
        # 3 legal slots; distribute visits 0.5 / 0.3 / 0.2.
        s0 = row % ACTION_VOCAB_SIZE
        s1 = (row + 1) % ACTION_VOCAB_SIZE
        s2 = (row + 2) % ACTION_VOCAB_SIZE
        action_masks[row, [s0, s1, s2]] = True
        policy_targets[row, s0] = 0.5
        policy_targets[row, s1] = 0.3
        policy_targets[row, s2] = 0.2
        row += 1
    for _ in range(n_value_only):
        # Value-only: no policy supervision, no action mask.
        row += 1

    return features, outcomes, policy_targets, action_masks


def test_downsample_keeps_expected_fraction_of_one_legal_rows():
    """80% one-legal supervised + 20% multi-legal: at keep_frac=0.1,
    the post-downsample row count should land at ~28% of original
    (within ±5%): ``20 + 0.1*80 = 28``."""
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=800,
        n_multi_legal_supervised=200,
        n_value_only=0,
        seed=0,
    )
    assert f.shape[0] == 1000

    fout, oout, pout, aout, total, kept = m._downsample_one_legal_rows(
        f, o, p, a, keep_frac=0.1, seed=42,
    )
    assert total == 800
    assert kept == 80  # round(800 * 0.1)
    # Expected final rows: 200 multi-legal kept + 80 one-legal kept = 280.
    expected = 280
    assert abs(fout.shape[0] - expected) <= int(0.05 * expected)
    # Outcomes/policy/mask all agree on the row count.
    assert oout.shape[0] == fout.shape[0]
    assert pout.shape[0] == fout.shape[0]
    assert aout.shape[0] == fout.shape[0]


def test_downsample_keeps_value_only_rows_intact():
    """Value-only rows (mask all-False, policy_target all-zero) must
    NEVER be downsampled — they don't contribute to the policy
    gradient anyway and they feed the value head."""
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=100,
        n_multi_legal_supervised=20,
        n_value_only=50,
        seed=0,
    )
    fout, _oout, pout, aout, total, kept = m._downsample_one_legal_rows(
        f, o, p, a, keep_frac=0.2, seed=7,
    )
    assert total == 100
    assert kept == 20

    # All 50 value-only rows survive: count rows with no policy supervision.
    has_policy = pout.sum(axis=-1) > 0.5
    n_value_only_kept = int((~has_policy).sum())
    assert n_value_only_kept == 50
    # Final row count = 20 (kept one-legal) + 20 (multi-legal) + 50 (value).
    assert fout.shape[0] == 90


def test_downsample_reproducible_with_same_seed():
    """Same seed + same input → identical kept-row indices."""
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=200,
        n_multi_legal_supervised=50,
        n_value_only=10,
        seed=0,
    )
    out_a = m._downsample_one_legal_rows(
        f, o, p, a, keep_frac=0.25, seed=123,
    )
    out_b = m._downsample_one_legal_rows(
        f, o, p, a, keep_frac=0.25, seed=123,
    )
    np.testing.assert_array_equal(out_a[0], out_b[0])
    np.testing.assert_array_equal(out_a[1], out_b[1])
    np.testing.assert_array_equal(out_a[2], out_b[2])
    np.testing.assert_array_equal(out_a[3], out_b[3])


def test_downsample_different_seeds_pick_different_rows():
    """Different seeds → different kept subsets (they may overlap by
    chance but the kept-row index sets shouldn't be identical)."""
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=200,
        n_multi_legal_supervised=20,
        n_value_only=0,
        seed=0,
    )
    out_a = m._downsample_one_legal_rows(
        f, o, p, a, keep_frac=0.1, seed=1,
    )
    out_b = m._downsample_one_legal_rows(
        f, o, p, a, keep_frac=0.1, seed=2,
    )
    # Rows are addressed by feature contents — first feature distinguishes
    # them. The two outputs should differ on at least one row.
    assert not np.array_equal(out_a[0], out_b[0])


def test_downsample_keep_frac_one_is_passthrough():
    """``keep_frac=1.0`` should leave the dataset untouched."""
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=50,
        n_multi_legal_supervised=10,
        n_value_only=5,
        seed=0,
    )
    fout, oout, pout, aout, total, kept = m._downsample_one_legal_rows(
        f, o, p, a, keep_frac=1.0, seed=0,
    )
    np.testing.assert_array_equal(fout, f)
    np.testing.assert_array_equal(oout, o)
    np.testing.assert_array_equal(pout, p)
    np.testing.assert_array_equal(aout, a)
    assert total == 50
    assert kept == 50


def test_downsample_zero_one_legal_rows_is_passthrough():
    """No one-legal supervised rows → nothing to downsample."""
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=0,
        n_multi_legal_supervised=20,
        n_value_only=10,
        seed=0,
    )
    fout, _oout, _pout, _aout, total, kept = (
        m._downsample_one_legal_rows(
            f, o, p, a, keep_frac=0.1, seed=0,
        )
    )
    assert total == 0
    assert kept == 0
    assert fout.shape[0] == 30


def test_downsample_rejects_zero_keep_frac():
    """``keep_frac=0.0`` is invalid (would erase all one-legal rows
    instead of just downsampling — use a different code path if you
    truly want that)."""
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=10,
        n_multi_legal_supervised=5,
        n_value_only=0,
        seed=0,
    )
    with pytest.raises(ValueError, match=r"keep_frac"):
        m._downsample_one_legal_rows(
            f, o, p, a, keep_frac=0.0, seed=0,
        )


def test_downsample_rejects_negative_keep_frac():
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=5,
        n_multi_legal_supervised=5,
        n_value_only=0,
        seed=0,
    )
    with pytest.raises(ValueError, match=r"keep_frac"):
        m._downsample_one_legal_rows(
            f, o, p, a, keep_frac=-0.1, seed=0,
        )


def test_downsample_rejects_keep_frac_above_one():
    m = _load_train_nn()
    f, o, p, a = _build_synthetic(
        n_one_legal_supervised=5,
        n_multi_legal_supervised=5,
        n_value_only=0,
        seed=0,
    )
    with pytest.raises(ValueError, match=r"keep_frac"):
        m._downsample_one_legal_rows(
            f, o, p, a, keep_frac=1.5, seed=0,
        )


def test_cli_rejects_zero_keep_frac():
    """CLI-level: ``--one-legal-keep-frac 0.0`` must surface as a
    parser error (exit code 2) before any data loading happens."""
    env_setup = {"PYTHONPATH": str(_REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--data-dir", str(_REPO_ROOT),  # any dir, won't be reached
            "--one-legal-keep-frac", "0.0",
        ],
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, **env_setup},
    )
    assert result.returncode != 0
    assert "one-legal-keep-frac" in result.stderr


def test_cli_rejects_keep_frac_above_one():
    env_setup = {"PYTHONPATH": str(_REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--data-dir", str(_REPO_ROOT),
            "--one-legal-keep-frac", "2.0",
        ],
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, **env_setup},
    )
    assert result.returncode != 0
    assert "one-legal-keep-frac" in result.stderr
