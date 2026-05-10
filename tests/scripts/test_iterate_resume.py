"""Tests for the partial-iter recovery path in ``mtg_bot.cli.iterate``
(formerly ``scripts/iterate_alphazero.py``; the script is a thin
shim now).

Covers ``_reconcile_iterate_state`` against the three on-disk shapes
the iterate loop can leave behind when a process is killed mid-cycle:

A. State matches the registry (clean — recovery is a no-op).
B. Trainer wrote v{N}/ but the loop crashed before persisting the
   promote/keep decision. The eval card under the progress dir is
   the source of truth — we read it and rewrite state.
C. State references a version whose dir is missing (registry
   corruption — refuse to touch state).

Plus the no-eval-data sub-case of B (orphan exists but eval card
never landed): refuse rather than silently guess.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    src_dir = _REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if "mtg_bot.cli.iterate" in sys.modules:
        del sys.modules["mtg_bot.cli.iterate"]
    return importlib.import_module("mtg_bot.cli.iterate")


def _make_version_dir(
    registry_root: Path, version: str, *, complete: bool = True,
) -> Path:
    d = registry_root / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "weights.pt").write_bytes(b"\x00")
    if complete:
        (d / "metadata.json").write_text(
            json.dumps({
                "version": version,
                "created_at": "2026-05-03T00:00:00Z",
                "feature_dim": 1,
                "hidden_sizes": [1],
            }),
            encoding="utf-8",
        )
    return d


def _write_eval_card(
    progress_dir: Path,
    iter_idx: int,
    new_version: str,
    old_version: str,
    *,
    wins_new: int,
    wins_old: int,
    draws: int = 0,
) -> Path:
    progress_dir.mkdir(parents=True, exist_ok=True)
    decided = wins_new + wins_old
    win_rate = (wins_new / decided) if decided > 0 else 0.0
    payload = {
        "kind": "eval",
        "status": "completed",
        "wins_new": wins_new,
        "wins_old": wins_old,
        "draws": draws,
        "win_rate": win_rate,
        "elapsed_seconds": 12.5,
        "opponent_label": old_version,
    }
    p = (
        progress_dir
        / f"iter_{iter_idx:03d}_eval_{new_version}_vs_{old_version}.json"
    )
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_case_a_clean_state_is_noop(tmp_path: Path):
    """State pointer matches the highest-numbered registry dir.
    Nothing to reconcile; helper returns the same state object."""
    iz = _load_module()
    registry_root = tmp_path / "trained_weights"
    progress_dir = tmp_path / "training_progress"
    _make_version_dir(registry_root, "v4")
    _make_version_dir(registry_root, "v5")
    state = {
        "current_best": "v5",
        "iterations_completed": 1,
        "history": [
            {"iteration": 1, "trained_version": "v5", "promoted": True},
        ],
    }
    out = iz._reconcile_iterate_state(
        state, registry_root, progress_dir,
        bootstrap_version="v4", acceptance_threshold=0.55,
    )
    assert out is state


def test_case_b_promotes_when_win_rate_meets_threshold(tmp_path: Path):
    """v6 orphan dir + an eval card showing the new model won 60%
    of decided games promotes v6 to current_best and bumps
    iterations_completed."""
    iz = _load_module()
    registry_root = tmp_path / "trained_weights"
    progress_dir = tmp_path / "training_progress"
    _make_version_dir(registry_root, "v4")
    _make_version_dir(registry_root, "v5")
    _make_version_dir(registry_root, "v6")  # orphan
    _write_eval_card(
        progress_dir, iter_idx=2, new_version="v6", old_version="v5",
        wins_new=30, wins_old=20, draws=0,
    )
    state = {
        "current_best": "v5",
        "iterations_completed": 1,
        "history": [
            {"iteration": 1, "trained_version": "v5", "promoted": True},
        ],
    }
    out = iz._reconcile_iterate_state(
        state, registry_root, progress_dir,
        bootstrap_version="v4", acceptance_threshold=0.55,
    )
    assert out["current_best"] == "v6"
    assert out["iterations_completed"] == 2
    last = out["history"][-1]
    assert last["iteration"] == 2
    assert last["trained_version"] == "v6"
    assert last["promoted"] is True
    assert last["reconciled"] is True
    assert last["eval"]["wins_new"] == 30
    assert last["eval"]["wins_old"] == 20
    assert last["eval"]["win_rate"] == pytest.approx(0.6)


def test_case_b_keeps_when_win_rate_below_threshold(tmp_path: Path):
    """Same shape, but win_rate=40% < 55% threshold → record a
    rejected iter and leave current_best alone."""
    iz = _load_module()
    registry_root = tmp_path / "trained_weights"
    progress_dir = tmp_path / "training_progress"
    _make_version_dir(registry_root, "v4")
    _make_version_dir(registry_root, "v5")
    _make_version_dir(registry_root, "v6")
    _write_eval_card(
        progress_dir, iter_idx=2, new_version="v6", old_version="v5",
        wins_new=20, wins_old=30, draws=0,
    )
    state = {
        "current_best": "v5",
        "iterations_completed": 1,
        "history": [
            {"iteration": 1, "trained_version": "v5", "promoted": True},
        ],
    }
    out = iz._reconcile_iterate_state(
        state, registry_root, progress_dir,
        bootstrap_version="v4", acceptance_threshold=0.55,
    )
    assert out["current_best"] == "v5"
    assert out["iterations_completed"] == 2
    last = out["history"][-1]
    assert last["promoted"] is False
    assert last["trained_version"] == "v6"
    assert last["eval"]["win_rate"] == pytest.approx(0.4)


def test_case_b_without_eval_data_raises(tmp_path: Path):
    """Orphan v6 dir but no eval card under progress_dir — the
    crash fired before / during eval. Reconcile refuses."""
    iz = _load_module()
    registry_root = tmp_path / "trained_weights"
    progress_dir = tmp_path / "training_progress"
    progress_dir.mkdir()
    _make_version_dir(registry_root, "v4")
    _make_version_dir(registry_root, "v5")
    _make_version_dir(registry_root, "v6")
    state = {
        "current_best": "v5",
        "iterations_completed": 1,
        "history": [
            {"iteration": 1, "trained_version": "v5", "promoted": True},
        ],
    }
    with pytest.raises(iz.ReconcileError, match="no eval progress card"):
        iz._reconcile_iterate_state(
            state, registry_root, progress_dir,
            bootstrap_version="v4", acceptance_threshold=0.55,
        )


def test_case_c_missing_current_best_dir_raises(tmp_path: Path):
    """State references v5 but only v4 exists — registry corruption.
    Raise rather than touch state."""
    iz = _load_module()
    registry_root = tmp_path / "trained_weights"
    progress_dir = tmp_path / "training_progress"
    _make_version_dir(registry_root, "v4")
    state = {
        "current_best": "v5",
        "iterations_completed": 1,
        "history": [
            {"iteration": 1, "trained_version": "v5", "promoted": True},
        ],
    }
    with pytest.raises(iz.ReconcileError, match="Case C"):
        iz._reconcile_iterate_state(
            state, registry_root, progress_dir,
            bootstrap_version="v4", acceptance_threshold=0.55,
        )


def test_multi_orphan_refuses_to_auto_reconcile(tmp_path: Path):
    """Two orphans is more than one partial-iter crash can produce —
    refuse rather than guess which iter ran which."""
    iz = _load_module()
    registry_root = tmp_path / "trained_weights"
    progress_dir = tmp_path / "training_progress"
    _make_version_dir(registry_root, "v4")
    _make_version_dir(registry_root, "v5")
    _make_version_dir(registry_root, "v6")
    _make_version_dir(registry_root, "v7")
    state = {
        "current_best": "v5",
        "iterations_completed": 1,
        "history": [
            {"iteration": 1, "trained_version": "v5", "promoted": True},
        ],
    }
    with pytest.raises(iz.ReconcileError, match="2 orphans"):
        iz._reconcile_iterate_state(
            state, registry_root, progress_dir,
            bootstrap_version="v4", acceptance_threshold=0.55,
        )


def test_orphan_with_no_decided_games_raises(tmp_path: Path):
    """Eval card exists but every game was a draw (or crashed) →
    win_rate is undefined; refuse rather than treat 0.0 as a real
    decision."""
    iz = _load_module()
    registry_root = tmp_path / "trained_weights"
    progress_dir = tmp_path / "training_progress"
    _make_version_dir(registry_root, "v4")
    _make_version_dir(registry_root, "v5")
    _make_version_dir(registry_root, "v6")
    _write_eval_card(
        progress_dir, iter_idx=2, new_version="v6", old_version="v5",
        wins_new=0, wins_old=0, draws=10,
    )
    state = {
        "current_best": "v5",
        "iterations_completed": 1,
        "history": [
            {"iteration": 1, "trained_version": "v5", "promoted": True},
        ],
    }
    with pytest.raises(iz.ReconcileError, match="no decided games"):
        iz._reconcile_iterate_state(
            state, registry_root, progress_dir,
            bootstrap_version="v4", acceptance_threshold=0.55,
        )


# ---- _compute_slack_extra ---------------------------------------


def test_compute_slack_extra_default_10pct():
    """Default --queue-slack-pct=0.1 → 10% over-queue. 500 → 50."""
    iz = _load_module()
    assert iz._compute_slack_extra(500, 0.1) == 50


def test_compute_slack_extra_zero_pct_disables_overqueue():
    """slack_pct=0 disables over-queue (legacy "drain everything"
    behavior). The orchestrator skips poller startup when this
    returns 0."""
    iz = _load_module()
    assert iz._compute_slack_extra(500, 0.0) == 0


def test_compute_slack_extra_rounds_up():
    """5% of 33 = 1.65 → ceil = 2. Never silently round down or we'd
    miss "I want some slack" for small target sizes."""
    iz = _load_module()
    assert iz._compute_slack_extra(33, 0.05) == 2


def test_compute_slack_extra_negative_clamps_to_zero():
    """Negative percentages don't make sense — clamp to 0 instead
    of negative-sizing the queue."""
    iz = _load_module()
    assert iz._compute_slack_extra(500, -0.5) == 0


def test_compute_slack_extra_large_pct():
    """50% over-queue: 500 → 250. The user should be able to dial
    slack arbitrarily; no implicit upper cap."""
    iz = _load_module()
    assert iz._compute_slack_extra(500, 0.5) == 250
