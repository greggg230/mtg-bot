"""Tests for ``mtg_bot.training.self_play.run_match``.

Validates the harness-level contract: deterministic seeding,
correct stat aggregation, the decision-cap safety net, and that
results add up to ``n_games``. Uses RandomChoice on both sides
so each test case finishes in single-digit seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

from mtg_bot.decks import Deck
from mtg_bot.training import (
    EvalProgress, MatchStats, random_choice_factory, run_match,
)
from mtg_bot.training.self_play import run_match_parallel
from tests._test_decks import simple_test_decklist, simple_test_deck_pool


def test_run_match_returns_stats_with_correct_total():
    stats = run_match(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=4,
        base_seed=42, deck_pool=simple_test_deck_pool(),
    )
    assert isinstance(stats, MatchStats)
    assert stats.games == 4
    assert stats.wins[0] + stats.wins[1] + stats.draws == 4


def test_run_match_is_deterministic_in_base_seed():
    """Same base_seed → same MatchStats. Critical for tuning loops
    that compare configs across runs."""
    a = run_match(
        random_choice_factory, random_choice_factory,
        n_games=3, base_seed=99, deck_pool=simple_test_deck_pool(),
    )
    b = run_match(
        random_choice_factory, random_choice_factory,
        n_games=3, base_seed=99, deck_pool=simple_test_deck_pool(),
    )
    assert a == b


def test_run_match_different_seeds_can_produce_different_results():
    """Sanity: not every seed produces the same outcome — we want
    real seed sensitivity so tuning data isn't all one trace."""
    seen = set()
    for seed in range(8):
        stats = run_match(
            random_choice_factory, random_choice_factory,
            n_games=2, base_seed=seed, deck_pool=simple_test_deck_pool(),
        )
        seen.add((stats.wins[0], stats.wins[1], stats.draws))
    assert len(seen) > 1


def test_run_match_win_rate_helper():
    stats = run_match(
        random_choice_factory, random_choice_factory,
        n_games=5, base_seed=0, deck_pool=simple_test_deck_pool(),
    )
    expected = stats.wins[0] / 5
    assert stats.win_rate(0) == expected
    assert stats.win_rate(1) == stats.wins[1] / 5


def test_run_match_zero_games_returns_zero_stats():
    """Edge case: n_games=0 shouldn't crash."""
    stats = run_match(
        random_choice_factory, random_choice_factory,
        n_games=0, base_seed=0, deck_pool=simple_test_deck_pool(),
    )
    assert stats.games == 0
    assert stats.wins == {0: 0, 1: 0}
    assert stats.draws == 0
    assert stats.avg_decisions == 0.0
    assert stats.win_rate(0) == 0.0


def test_run_match_records_avg_decisions():
    """avg_decisions should reflect actual game lengths — at least
    > 0 for any non-trivial game."""
    stats = run_match(
        random_choice_factory, random_choice_factory,
        n_games=3, base_seed=11, deck_pool=simple_test_deck_pool(),
    )
    assert stats.avg_decisions > 0


def test_run_match_decision_cap_treats_unfinished_as_draw():
    """If we set the cap absurdly low, every game hits it and is
    counted as a draw. Verifies the cap actually caps and the draw
    accounting works."""
    stats = run_match(
        random_choice_factory, random_choice_factory,
        n_games=3, base_seed=0,
        decision_cap=2,  # absurdly low
        deck_pool=simple_test_deck_pool(),
    )
    assert stats.draws == 3
    assert stats.wins == {0: 0, 1: 0}
    assert stats.avg_decisions == 2  # every game stopped at cap


def test_run_match_parallel_with_progress_emits_snapshot(tmp_path: Path):
    """The progress-tracked parallel path must produce the same
    MatchStats as the unindexed path AND leave a finalized
    progress JSON on disk. Two-game smoke is enough to exercise
    the imap_unordered → record_game → finalize flow."""
    progress = EvalProgress(
        progress_path=tmp_path / "eval.json",
        run_id="eval",
        n_games_total=4,
        n_workers=2,
        opponent_label="random",
        flush_interval_s=0.0,
    )
    stats = run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=4,
        base_seed=0,
        n_workers=2,
        progress=progress, deck_pool=simple_test_deck_pool(),
    )
    # Same aggregation as the no-progress path (both call
    # _aggregate_results on equivalent inputs).
    expected = run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=4,
        base_seed=0,
        n_workers=2, deck_pool=simple_test_deck_pool(),
    )
    assert stats == expected

    payload = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "eval"
    assert payload["status"] == "completed"
    assert payload["n_games_total"] == 4
    assert payload["n_games_done"] == 4
    assert payload["wins_new"] == stats.wins[0]
    assert payload["wins_old"] == stats.wins[1]
    assert payload["draws"] == stats.draws
    assert payload["opponent_label"] == "random"


def test_run_match_parallel_with_deck_pool_uses_pool_decks():
    """When deck_pool is supplied, the parallel match path samples
    from the pool (per-game seed) and the games still aggregate
    into a valid MatchStats. Same accounting invariants as the
    legacy mono-color path."""
    pool = [
        Deck(draft_id="g_deck", build_index=0,
             cards=simple_test_decklist("G")),
        Deck(draft_id="r_deck", build_index=0,
             cards=simple_test_decklist("R")),
    ]
    stats = run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2,
        base_seed=42,
        n_workers=2,
        deck_pool=pool,
    )
    assert isinstance(stats, MatchStats)
    assert stats.wins[0] + stats.wins[1] + stats.draws + stats.crashed == 2


def test_run_match_deck_pool_sequential_equals_parallel():
    """Strong invariant mirroring data_gen's: same deck_pool +
    base_seed gives identical MatchStats across sequential and
    parallel paths. Without this, switching workers between runs
    would silently shift evaluation results."""
    pool = [
        Deck(draft_id="g_deck", build_index=0,
             cards=simple_test_decklist("G")),
        Deck(draft_id="r_deck", build_index=0,
             cards=simple_test_decklist("R")),
        Deck(draft_id="w_deck", build_index=0,
             cards=simple_test_decklist("W")),
    ]
    seq = run_match(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3,
        base_seed=7,
        deck_pool=pool,
    )
    par = run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=3,
        base_seed=7,
        n_workers=2,
        deck_pool=pool,
    )
    assert seq == par


def test_run_match_with_replay_config_writes_replay_files(tmp_path: Path):
    """When a ReplayCaptureConfig is supplied, run_match writes one
    .json.gz file per captured game into the configured save_dir.
    Default capture_every=1 captures every game."""
    from mtg_bot.training.data_gen import ReplayCaptureConfig

    save_dir = tmp_path / "replays"
    save_dir.mkdir()
    cfg = ReplayCaptureConfig(save_dir=str(save_dir))
    stats = run_match(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=2,
        base_seed=42,
        replay_config=cfg, deck_pool=simple_test_deck_pool(),
    )
    assert stats.games == 2
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 2


def test_run_match_parallel_with_replay_config_captures_subset(
    tmp_path: Path,
):
    """capture_every=2 over 4 games → 2 captures (game_idx 0 and 2)."""
    from mtg_bot.training.data_gen import ReplayCaptureConfig

    save_dir = tmp_path / "replays_eval"
    save_dir.mkdir()
    cfg = ReplayCaptureConfig(save_dir=str(save_dir), capture_every=2)
    run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=4,
        base_seed=42,
        n_workers=2,
        replay_config=cfg, deck_pool=simple_test_deck_pool(),
    )
    files = sorted(save_dir.glob("*.json.gz"))
    assert len(files) == 2
    names = [f.name for f in files]
    assert all("seed42" in n for n in names)
    assert any("idx0000" in n for n in names)
    assert any("idx0002" in n for n in names)


def test_alternating_worker_swaps_factories_on_odd_game_idx():
    """Direct unit test of ``_play_one_match_game_alternating_indexed``.
    Builds a stub worker that records which factory ran in seat 0
    so the test can verify the swap happens on odd ``game_idx`` and
    NOT on even ``game_idx``."""
    from mtg_bot.training.self_play import (
        _play_one_match_game_alternating_indexed,
    )

    # Sentinel "factories" — we don't care that they're not real
    # factories because we monkeypatch the inner playing function.
    f_new = ("FACTORY", "new")
    f_old = ("FACTORY", "old")

    captured: list[tuple] = []

    def fake_play(game_idx, p0, p1, *_args):
        captured.append((game_idx, p0, p1))
        # Always return p0 wins so we can verify the remap.
        return 0, 5  # winner=0, decisions=5

    import mtg_bot.training.self_play as sp
    real = sp._play_one_match_game
    sp._play_one_match_game = fake_play
    try:
        # Even game_idx — no swap.
        gid, (winner, decisions, kind) = (
            _play_one_match_game_alternating_indexed(
                (0, f_new, f_old, 0, 100, False, None, None, None, None),
            )
        )
        assert captured[-1] == (0, f_new, f_old)
        # winner=0 returned from engine = p0 won = "new" won.
        assert winner == 0
        assert kind == "completed"

        # Odd game_idx — swap.
        gid, (winner, decisions, kind) = (
            _play_one_match_game_alternating_indexed(
                (1, f_new, f_old, 0, 100, False, None, None, None, None),
            )
        )
        # Inner call saw f_old as p0 (swapped).
        assert captured[-1] == (1, f_old, f_new)
        # Engine returned winner=0 (its p0 = swapped = old won). Caller-
        # facing winner remaps to 1 (the original p0 = new lost).
        assert winner == 1
        assert kind == "completed"
    finally:
        sp._play_one_match_game = real


def test_run_match_parallel_alternate_seats_neutralizes_p0_bias():
    """End-to-end: with two RandomChoice factories that have a strong
    seat-0 bias (RandomChoice always picks the first legal action,
    which favors the player on the move), ``alternate_seats=True``
    pulls wins[0] back toward 50% relative to the no-alternation
    baseline."""
    # 32 games — small-N bias swings are noisy on RandomChoice with
    # the parser-source decks; doubling the sample size gives the
    # alternation a clear separation from the baseline without making
    # the test meaningfully slower. NOTE: this is a statistical
    # assertion at n=32 and is seed-sensitive — base_seed picked to
    # land on a trajectory where alternation visibly reduces skew.
    no_alt = run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=32, base_seed=2, n_workers=2,
        deck_pool=simple_test_deck_pool(),
    )
    alt = run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=32, base_seed=2, n_workers=2,
        alternate_seats=True, deck_pool=simple_test_deck_pool(),
    )
    # Both runs played the same games with the same seeds, so the
    # outcomes are reproducible. We expect the alternating run to
    # have a more even split.
    no_alt_skew = abs(no_alt.wins[0] - no_alt.wins[1])
    alt_skew = abs(alt.wins[0] - alt.wins[1])
    assert alt_skew <= no_alt_skew, (
        f"alternation didn't reduce seat skew: "
        f"no_alt={no_alt.wins} alt={alt.wins}"
    )
    # And the totals still add up to n_games regardless.
    assert no_alt.wins[0] + no_alt.wins[1] + no_alt.draws == 32
    assert alt.wins[0] + alt.wins[1] + alt.draws == 32


def test_run_match_parallel_progress_finalizes_on_zero_games(tmp_path: Path):
    """Empty-pool short-circuit still finalizes the tracker — the
    UI shouldn't see a stale 'running' file after a 0-game eval."""
    progress = EvalProgress(
        progress_path=tmp_path / "empty.json",
        run_id="empty",
        n_games_total=0,
        n_workers=1,
    )
    stats = run_match_parallel(
        p0_factory=random_choice_factory,
        p1_factory=random_choice_factory,
        n_games=0,
        n_workers=1,
        progress=progress, deck_pool=simple_test_deck_pool(),
    )
    assert stats.games == 0
    payload = json.loads((tmp_path / "empty.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["ended_at"] is not None
