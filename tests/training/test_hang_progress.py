"""Tests for ``mtg_bot.training._hang_progress.HangProgress``.

The progress sentinel is shared between the inner-game thread and the
parent timeout handler. We test:

  - ``record`` updates fields and bumps ``decisions_completed``.
  - ``snapshot`` returns a JSON-serializable dict with the derived
    ``seconds_since_last_update`` and ``total_elapsed_sec`` fields.
  - ``record`` truncates oversized decision repr's to keep dumps small.
  - ``record`` never raises when ``repr()`` of the decision throws.
  - Integration: a TimeoutError raised inside the data-gen / self-play
    timeout handler dumps a hang JSON containing ``progress`` populated
    from the sentinel.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mtg_bot.training._hang_progress import HangProgress


def test_record_updates_fields_and_bumps_decisions():
    p = HangProgress()
    assert p.decisions_completed == 0
    assert p.last_active_player is None

    p.record(active=0, decision="PassPriority()", n_legal=2)
    assert p.decisions_completed == 1
    assert p.last_active_player == 0
    assert p.last_decision_repr == "'PassPriority()'"  # repr of a str
    assert p.last_n_legal == 2

    p.record(active=1, decision=42, n_legal=5)
    assert p.decisions_completed == 2
    assert p.last_active_player == 1
    assert p.last_decision_repr == "42"
    assert p.last_n_legal == 5


def test_snapshot_returns_serializable_dict_with_derived_fields():
    p = HangProgress()
    p.record(active=0, decision="X", n_legal=3)
    snap = p.snapshot()
    # Must be json-serializable (dump_hang_diagnostics jsons it).
    json.dumps(snap)
    assert snap["decisions_completed"] == 1
    assert snap["last_active_player"] == 0
    assert snap["last_decision_repr"] == "'X'"
    assert snap["last_n_legal"] == 3
    assert snap["seconds_since_last_update"] >= 0.0
    assert snap["total_elapsed_sec"] >= 0.0


def test_record_truncates_long_repr():
    p = HangProgress()
    huge = "a" * 1000
    p.record(active=0, decision=huge, n_legal=1)
    # repr adds the surrounding quotes; we cap at 160 chars total.
    assert len(p.last_decision_repr) == 160


def test_record_swallows_repr_exceptions():
    class _Exploding:
        def __repr__(self):  # pragma: no cover - tested via record
            raise RuntimeError("boom")

    p = HangProgress()
    # Must not raise.
    p.record(active=0, decision=_Exploding(), n_legal=0)
    assert p.last_decision_repr == "<unrepr-able>"
    assert p.decisions_completed == 1


def test_snapshot_seconds_since_last_update_grows():
    p = HangProgress()
    p.record(active=0, decision="X", n_legal=1)
    snap1 = p.snapshot()
    time.sleep(0.05)
    snap2 = p.snapshot()
    # Allow tiny clock noise; snap2 should be later than snap1.
    assert snap2["seconds_since_last_update"] >= snap1[
        "seconds_since_last_update"
    ]


# -----------------------------------------------------------------
# Integration: confirm the timeout handler dumps include the
# ``progress`` field. We monkey-patch ``run_with_timeout`` to (a) call
# the inner function for a few decisions then raise TimeoutError, so we
# can verify the sentinel was actually updated, and (b) avoid spinning
# up a full game.
# -----------------------------------------------------------------


def _drive_progress(progress: HangProgress, n: int) -> None:
    """Call ``progress.record`` ``n`` times to simulate decisions before
    a hang fires."""
    for i in range(n):
        progress.record(active=i % 2, decision=f"D{i}", n_legal=3)


def test_data_gen_timeout_dump_contains_progress(
    tmp_path: Path, monkeypatch,
):
    from mtg_bot.training import _timeout
    from mtg_bot.training import random_choice_factory
    from mtg_bot.training.data_gen import _generate_one_game_data
    from tests._test_decks import simple_test_deck_pool

    # ``run_with_timeout(func, args, timeout)`` — drive the progress
    # sentinel that's been threaded into ``args`` for a few iterations,
    # then raise. Progress is identified by isinstance check rather
    # than positional index because the diagnostic-deep-capture feature
    # added trailing ``tree_depth`` / ``tree_top_k`` ints after the
    # progress slot — index lookup is brittle to that.
    def fake_run_with_timeout(func, args, timeout_sec):
        progress = next(a for a in args if isinstance(a, HangProgress))
        _drive_progress(progress, 7)
        raise TimeoutError("simulated hang at decision 7")

    monkeypatch.setattr(
        _timeout, "run_with_timeout", fake_run_with_timeout,
    )

    dump_dir = tmp_path / "hangs"
    with pytest.raises(TimeoutError):
        _generate_one_game_data(
            game_idx=11,
            p0_factory=random_choice_factory,
            p1_factory=random_choice_factory,
            base_seed=99,
            decision_cap=4000,
            sample_every=1,
            per_game_timeout_sec=300.0,
            hang_dump_dir=str(dump_dir),
            deck_pool=simple_test_deck_pool(),
        )
    files = sorted(dump_dir.iterdir())
    assert len(files) == 2
    meta = next(f for f in files if f.suffix == ".json")
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert "progress" in payload
    assert payload["progress"]["decisions_completed"] == 7
    assert payload["progress"]["last_decision_repr"] == "'D6'"
    assert payload["progress"]["last_active_player"] == 0  # 6 % 2 == 0


def test_self_play_timeout_dump_contains_progress(
    tmp_path: Path, monkeypatch,
):
    from mtg_bot.training import _timeout
    from mtg_bot.training import random_choice_factory
    from mtg_bot.training.self_play import _play_one_match_game
    from tests._test_decks import simple_test_deck_pool

    def fake_run_with_timeout(func, args, timeout_sec):
        # Locate the progress sentinel by isinstance — the args tuple
        # gained trailing tree_depth / tree_top_k ints with the
        # diagnostic-deep-capture feature, so positional indexing is
        # brittle.
        progress = next(a for a in args if isinstance(a, HangProgress))
        _drive_progress(progress, 4)
        raise TimeoutError("simulated hang at decision 4")

    monkeypatch.setattr(
        _timeout, "run_with_timeout", fake_run_with_timeout,
    )

    dump_dir = tmp_path / "hangs"
    with pytest.raises(TimeoutError):
        _play_one_match_game(
            game_idx=2,
            p0_factory=random_choice_factory,
            p1_factory=random_choice_factory,
            base_seed=42,
            decision_cap=4000,
            per_game_timeout_sec=300.0,
            hang_dump_dir=str(dump_dir),
            deck_pool=simple_test_deck_pool(),
        )
    files = sorted(dump_dir.iterdir())
    assert len(files) == 2
    meta = next(f for f in files if f.suffix == ".json")
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert "progress" in payload
    assert payload["progress"]["decisions_completed"] == 4
    assert payload["progress"]["last_decision_repr"] == "'D3'"


def test_hang_dump_progress_classifies_real_vs_slow():
    """The whole point of the progress sentinel: a triage reader can
    classify a hang dump as
      (a) `decisions_completed` near 0 + large
          `seconds_since_last_update` -> real engine pathology, or
      (b) `decisions_completed` in the hundreds + recent
          `seconds_since_last_update` -> slow but progressing.
    This test pins the snapshot semantics so future readers can rely
    on them.
    """
    # Case (a): few decisions, long stall.
    stuck = HangProgress()
    stuck.record(active=0, decision="STUCK", n_legal=2)
    # Force the last_update_at backwards to simulate a long stall
    # without sleeping in the test.
    stuck.last_update_at -= 250.0
    snap_a = stuck.snapshot()
    assert snap_a["decisions_completed"] == 1
    assert snap_a["seconds_since_last_update"] >= 250.0

    # Case (b): many decisions, recent update.
    busy = HangProgress()
    for i in range(600):
        busy.record(active=i % 2, decision=f"D{i}", n_legal=3)
    snap_b = busy.snapshot()
    assert snap_b["decisions_completed"] == 600
    assert snap_b["seconds_since_last_update"] < 1.0
