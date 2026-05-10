"""Self-play harness: run N games at given configs, aggregate stats.

The existing ``mtg_bot.search.harness`` is generic and protocol-
shaped (works on toy games too). This module wraps it for the MTG-
specific tuning use case: deterministic seeding from a base seed,
hard decision cap so a stuck game can't hang the loop, and a
``MatchStats`` that tracks average decisions per game (useful for
spotting "this config plays infinite-loop games" without time-
profiling each one).

All state is built fresh per game — no shared engine state across
games. Both player slots get independent RNGs derived from the base
seed so the same call always produces the same result (essential
for comparing tuning runs).

Parallel mode: ``run_match_parallel`` distributes the N games
across a ``multiprocessing.Pool``. Per-game seeds are derived from
``game_idx``, not worker assignment, so the result is identical to
the sequential ``run_match`` for the same ``base_seed``.

Worker-crash resilience: the parallel path catches engine-level
exceptions inside each worker (state-corruption bugs that surface
~1-in-1000 games in PIMC self-play) and converts them to a
``(None, 0)`` "crashed" sentinel so a single bad game can't tank
the whole eval. The driver tracks crashed-game indices and emits a
loud stderr summary at the end of the run; the crashes are also
counted toward ``MatchStats.draws`` (consistent with how decision-
cap timeouts are counted — both represent "game didn't reach a
clean conclusion") so the total still satisfies
``wins[0] + wins[1] + draws == games``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mtg_bot.agents import RandomAgent
from mtg_bot.decks import Deck
from mtg_bot.replay import save_replay
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.heuristics_config import HeuristicsConfig
from mtg_bot.search.protocol import CHANCE_PLAYER, PlayerID
from mtg_bot.simulate import setup_game
from mtg_bot.training._worker_threads import pin_worker_torch_threads
from mtg_bot.training.config import StrategyFactory
from mtg_bot.training.data_gen import (
    ReplayCaptureConfig, _now_iso, _sample_decks_for_game,
)
from mtg_bot.training.progress import EvalProgress
from mtg_bot.training.replay_capture import (
    ReplayRecorder, build_search_annotation,
)


# Hard cap on decisions per game. A pathological infinite priority
# loop (which has shown up before — see PIMC step-reentry bug) would
# otherwise hang the tuning loop. Real games are well under this; if
# we hit the cap, that's a bug.
_DEFAULT_DECISION_CAP = 4000


@dataclass(frozen=True)
class MatchStats:
    """Aggregate result of ``run_match``.

    ``wins[0]`` / ``wins[1]``: games player 0 / player 1 won.
    ``draws``: games that ended in a draw — i.e. the engine hit
    the ``decision_cap`` without a terminal state. Real MTG draws
    in our model are rare; most "draws" are decision-cap timeouts.
    ``games``: number of games that COMPLETED (wins + draws).
    Worker-crashed and per-game-timeout games are dropped from
    the dataset entirely — they contribute to ``crashed`` but
    NOT to ``wins`` / ``draws`` / ``games``. So the invariant is
    ``games == wins[0] + wins[1] + draws``, NOT ``== n_games``;
    the input ``n_games`` is the requested count, ``games`` is
    what actually contributed signal.
    ``crashed``: games dropped because a worker raised an engine-
    level exception (state-corruption bug — todo e3b7d9c1).
    Reported separately from ``timed_out`` so engine bugs are
    visible distinct from CPU-contention slow games.
    ``timed_out``: games dropped because the worker blew through
    the per-game wall-clock timeout (TimeoutError from
    run_with_timeout). Different root cause from ``crashed``;
    same handling — excluded from win-rate denominator entirely.
    ``avg_decisions``: mean decisions across non-crashed/non-timed-
    out games. Spots pathological game-length blowups even when
    the win rate looks fine.
    """

    wins: dict[PlayerID, int]
    draws: int
    games: int
    avg_decisions: float
    crashed: int = 0
    timed_out: int = 0

    def win_rate(self, player_id: PlayerID) -> float:
        """Fraction of completed games ``player_id`` won. Draws
        count in the denominator (someone tied), but crashed /
        timed-out games are excluded entirely — they have no
        signal to contribute."""
        return self.wins.get(player_id, 0) / self.games if self.games else 0.0


def run_match(
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    n_games: int,
    base_seed: int = 0,
    decision_cap: int = _DEFAULT_DECISION_CAP,
    enable_mulligans: bool = False,
    per_game_timeout_sec: Optional[float] = None,
    deck_pool: Optional[list[Deck]] = None,
    replay_config: Optional[ReplayCaptureConfig] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
) -> MatchStats:
    """Play ``n_games`` of ``p0_factory`` vs ``p1_factory`` and
    aggregate stats.

    Each game gets independent seeds derived from ``base_seed``:

    - Game ``i`` uses engine seed ``base_seed + i``.
    - P0's strategy is built with seed ``base_seed + 100*i``.
    - P1's strategy is built with seed ``base_seed + 100*i + 50``.
    - Chance-node sampling RNG uses seed ``base_seed + 100*i + 99``.

    The deterministic seeding means the same call produces the same
    result — essential for "does heuristic config A beat config B?"
    comparisons across tuning runs.
    """
    if heuristics is None:
        heuristics = HeuristicsConfig()
    results = [
        _play_one_match_game(
            game_idx, p0_factory, p1_factory, base_seed, decision_cap,
            enable_mulligans, per_game_timeout_sec, deck_pool,
            replay_config, hang_dump_dir, heuristics,
        )
        for game_idx in range(n_games)
    ]
    return _aggregate_results(results, n_games)


def run_match_parallel(
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    n_games: int,
    base_seed: int = 0,
    decision_cap: int = _DEFAULT_DECISION_CAP,
    n_workers: Optional[int] = None,
    progress: Optional[EvalProgress] = None,
    enable_mulligans: bool = False,
    per_game_timeout_sec: Optional[float] = None,
    deck_pool: Optional[list[Deck]] = None,
    replay_config: Optional[ReplayCaptureConfig] = None,
    hang_dump_dir: Optional[str] = None,
    alternate_seats: bool = False,
    heuristics: Optional[HeuristicsConfig] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> MatchStats:
    """Same contract as ``run_match`` but distributes games across a
    ``multiprocessing.Pool``.

    Identical output to the sequential function for the same
    ``base_seed`` — per-game seeds are derived from ``game_idx``,
    not worker assignment, so each game produces the same result
    no matter which worker plays it.

    ``n_workers`` defaults to ``cpu_count() - 2`` (leaves a few
    cores for OS / Python main / other work). Pass an explicit int
    to override.

    Caller note: factories must be picklable. Module-level functions
    and bound methods of picklable dataclasses work; lambdas and
    nested closures don't. The ``evaluate_nn`` CLI builds factories
    that pickle.

    Worker spawn cost on Windows is ~1-2s per worker, paid once per
    pool. Dominates wall-clock at very small ``n_games``; amortizes
    away past a few dozen games.

    When ``progress`` is supplied, results stream back via
    ``imap_unordered`` and the tracker is fed per-game so the UI
    sees live win-rate updates. The tracker is finalized
    ('completed' or 'failed') by this function before returning;
    the caller doesn't need to call ``finalize`` itself. With no
    progress tracker the legacy ``pool.map`` blocking path is used
    so existing callers keep their exact behavior (no per-game
    Python overhead).
    """
    if heuristics is None:
        heuristics = HeuristicsConfig()
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 2)
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if n_games == 0:
        if progress is not None:
            progress.finalize(status="completed")
        return MatchStats(wins={0: 0, 1: 0}, draws=0, games=0, avg_decisions=0.0)

    args_list = [
        (i, p0_factory, p1_factory, base_seed, decision_cap,
         enable_mulligans, per_game_timeout_sec, deck_pool,
         replay_config, hang_dump_dir, heuristics)
        for i in range(n_games)
    ]
    packed_worker = (
        _play_one_match_game_alternating_packed if alternate_seats
        else _play_one_match_game_packed
    )
    indexed_worker = (
        _play_one_match_game_alternating_indexed if alternate_seats
        else _play_one_match_game_indexed
    )
    if progress is None:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(packed_worker, args_list)
        # ``_play_one_match_game_packed`` tags each result with an
        # outcome_kind ("completed" / "crashed" / "timed_out") in
        # the trailing slot. Bucket the affected indices into
        # crashed vs timed-out lists so the driver can print
        # separate loud-stderr summaries for engine bugs vs
        # CPU-contention timeouts.
        crashed_game_indices: list[int] = []
        timed_out_game_indices: list[int] = []
        for i, entry in enumerate(results):
            if len(entry) >= 3:
                outcome_kind = entry[2]
            else:
                winner, decisions = entry[0], entry[1]
                outcome_kind = (
                    "crashed"
                    if winner is None and decisions == 0
                    else "completed"
                )
            if outcome_kind == "crashed":
                crashed_game_indices.append(i)
            elif outcome_kind == "timed_out":
                timed_out_game_indices.append(i)
        _log_crashed_games(crashed_game_indices, n_games)
        _log_timed_out_games(timed_out_game_indices, n_games)
        return _aggregate_results(results, n_games)

    # Progress-tracked path: stream completions via imap_unordered.
    # Wrap args with the game_idx so we can re-sort for deterministic
    # aggregation (imap_unordered preserves identity but not order).
    progress.begin()
    indexed: list[tuple[int, tuple]] = []
    crashed_game_indices: list[int] = []
    timed_out_game_indices: list[int] = []
    stopped_early = False
    try:
        with mp.Pool(processes=n_workers) as pool:
            for game_idx, result in pool.imap_unordered(
                indexed_worker, args_list, chunksize=1,
            ):
                indexed.append((game_idx, result))
                # Workers tag each result with an outcome_kind in
                # the trailing slot ("completed" / "crashed" /
                # "timed_out") so the driver can distinguish engine
                # bugs from per-game wall-clock timeouts. Surface
                # both through the progress tracker as dedicated
                # counters so the live UI tells them apart.
                if len(result) >= 3:
                    winner, decisions, outcome_kind = (
                        result[0], result[1], result[2],
                    )
                else:
                    winner, decisions = result[0], result[1]
                    outcome_kind = (
                        "crashed"
                        if winner is None and decisions == 0
                        else "completed"
                    )
                crashed = outcome_kind == "crashed"
                timed_out = outcome_kind == "timed_out"
                if crashed:
                    crashed_game_indices.append(game_idx)
                elif timed_out:
                    timed_out_game_indices.append(game_idx)
                progress.record_game(
                    winner=winner, decisions=decisions,
                    crashed=crashed, timed_out=timed_out,
                )
                # User-requested stop via the UI sentinel — terminate
                # pending dispatches and exit the loop. Same trade-off
                # as data_gen: in-flight games on workers are
                # killed; previously-completed results are preserved.
                # See ``docs/SEARCH_LIMITATIONS.md``.
                if stop_check is not None and stop_check():
                    stopped_early = True
                    pool.terminate()
                    pool.join()
                    break
    except BaseException:
        progress.finalize(status="failed")
        raise
    progress.finalize(
        status="stopped" if stopped_early else "completed",
    )
    _log_crashed_games(crashed_game_indices, n_games)
    _log_timed_out_games(timed_out_game_indices, n_games)
    indexed.sort(key=lambda t: t[0])
    return _aggregate_results([r for _, r in indexed], n_games)


def _log_crashed_games(crashed: list[int], n_games: int) -> None:
    """Print a loud stderr summary if any worker games crashed.

    Mirrors ``data_gen``'s post-pool warning: a silent partial-skip
    would mask the underlying engine bug (today's residual ~1/1000
    PIMC state-corruption bug). Crashed games are still counted as
    draws in ``MatchStats`` so the totals satisfy
    ``wins+draws == n_games``; this print is purely operator-facing.
    """
    if not crashed:
        return
    print(
        f"# WARNING: {len(crashed)}/{n_games} games crashed in the "
        f"eval pool and were dropped (engine state corruption — see "
        f"worker stderr above for tracebacks). Crashed games are "
        f"EXCLUDED from MatchStats — they don't pollute the win-rate "
        f"denominator; ``MatchStats.crashed`` carries the count "
        f"separately. Affected game indices: {crashed[:20]}"
        f"{' ...' if len(crashed) > 20 else ''}",
        file=sys.stderr,
    )


def _log_timed_out_games(timed_out: list[int], n_games: int) -> None:
    """Print a loud stderr summary if any worker games hit the
    per-game wall-clock timeout.

    Companion to ``_log_crashed_games``: same shape, different root
    cause. Timed-out games are usually slow-but-progressing games
    under MKL/CPU contention rather than engine bugs; tracking them
    separately lets the operator tell the two failure modes apart.
    """
    if not timed_out:
        return
    print(
        f"# WARNING: {len(timed_out)}/{n_games} games hit the per-"
        f"game wall-clock timeout in the eval pool and were dropped "
        f"(slow games under CPU contention or genuine engine hangs). "
        f"Timed-out games are EXCLUDED from MatchStats; "
        f"``MatchStats.timed_out`` carries the count separately. "
        f"Affected game indices: {timed_out[:20]}"
        f"{' ...' if len(timed_out) > 20 else ''}",
        file=sys.stderr,
    )


def _play_one_match_game_packed(
    args: tuple,
) -> tuple[Optional[PlayerID], int, str]:
    """Top-level pickleable worker. Unpacks the args tuple and
    delegates to ``_play_one_match_game``. Lives at module top
    level (not nested) so multiprocessing can pickle it across
    the spawn boundary.

    Engine-level errors (state-corruption bugs that surface at scale —
    ~1 in 1000 PIMC games as of 2026-05-02) are caught here and
    converted to a ``(None, 0, "crashed")`` sentinel so a single bad
    game doesn't take down the entire eval pool. Without this catch,
    ``pool.map`` re-raises the worker exception and aborts the whole
    run, losing the ~99 games of useful data already played. The
    traceback is logged to worker stderr so failures stay visible
    via the pool's stderr relay; the driver reads the trailing
    outcome-kind string to bucket each empty result.

    Per-game wall-clock ``TimeoutError`` (out of
    ``run_with_timeout``) is caught separately and tagged
    ``"timed_out"`` so the driver can report it as a distinct
    counter rather than lumping it under ``crashed``.

    Mirrors the worker-crash handling in
    ``mtg_bot.training.data_gen._play_one_parallel_game``.
    """
    game_idx = args[0]
    try:
        winner, decisions = _play_one_match_game(*args)
        return winner, decisions, "completed"
    except TimeoutError as exc:
        print(
            f"[worker] eval game {game_idx} timed out "
            f"({exc}); dropping this game, pool continues.",
            file=sys.stderr,
        )
        return None, 0, "timed_out"
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        print(
            f"[worker] eval game {game_idx} crashed "
            f"({type(exc).__name__}: {exc}); dropping this game, "
            f"pool continues.",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return None, 0, "crashed"


def _play_one_match_game_indexed(
    args: tuple,
) -> tuple[int, tuple[Optional[PlayerID], int, str]]:
    """Same as ``_play_one_match_game_packed`` but tags each result
    with its ``game_idx`` so the progress-tracked driver can re-sort
    ``imap_unordered`` results into deterministic input order before
    aggregating. The unindexed variant uses ``pool.map`` which
    preserves order on its own and doesn't need this tagging.

    Same worker-crash / timeout handling as the packed variant:
    engine-level exceptions become ``(None, 0, "crashed")`` and
    per-game timeouts become ``(None, 0, "timed_out")``; the driver
    keys off the trailing string to bucket the result.
    """
    game_idx = args[0]
    return game_idx, _play_one_match_game_packed(args)


def _play_one_match_game_alternating_packed(
    args: tuple,
) -> tuple[Optional[PlayerID], int, str]:
    """``pool.map`` counterpart to
    ``_play_one_match_game_alternating_indexed`` — same swap-on-odd-
    game_idx logic, no game_idx tagging in the return."""
    game_idx = args[0]
    if game_idx % 2 == 0:
        return _play_one_match_game_packed(args)
    swapped = (args[0], args[2], args[1], *args[3:])
    winner, decisions, outcome_kind = _play_one_match_game_packed(swapped)
    remapped = None if winner is None else 1 - winner
    return remapped, decisions, outcome_kind


def _play_one_match_game_alternating_indexed(
    args: tuple,
) -> tuple[int, tuple[Optional[PlayerID], int, str]]:
    """Like ``_play_one_match_game_indexed`` but neutralizes seat-
    order bias by playing odd-indexed games with the two factories
    swapped. The returned winner is remapped so seat 0 in the result
    always means "the original ``p0_factory`` won" — the caller
    aggregates wins[0]/wins[1] without needing to know about the
    swap.

    Use case: eval-gate matches where seat 0 has a structural engine
    advantage. v6-vs-v6 (identical NN) showed wins[0] = 77% on a
    500-game match without alternation; with alternation it should
    converge to 50% within stat noise.
    """
    game_idx = args[0]
    if game_idx % 2 == 0:
        return _play_one_match_game_indexed(args)
    swapped = (args[0], args[2], args[1], *args[3:])
    game_idx_returned, (winner, decisions, outcome_kind) = (
        _play_one_match_game_indexed(swapped)
    )
    remapped = None if winner is None else 1 - winner
    return game_idx_returned, (remapped, decisions, outcome_kind)


def _play_one_match_game(
    game_idx: int,
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    base_seed: int,
    decision_cap: int,
    enable_mulligans: bool = False,
    per_game_timeout_sec: Optional[float] = None,
    deck_pool: Optional[list[Deck]] = None,
    replay_config: Optional[ReplayCaptureConfig] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
) -> tuple[Optional[PlayerID], int]:
    """Play one game and return ``(winner_or_None, decisions)``.
    Used by both the sequential and parallel match drivers — same
    seed derivation logic in both paths so the results match.

    Two distinct decks are sampled per game from ``deck_pool``
    (deterministic per ``(base_seed, game_idx)``); both seats use
    real archetypes from the pool. ``deck_pool`` is required — the
    synthetic mono-color G/R fallback was removed; every eval game
    now comes from a real deck distribution by construction.

    When ``replay_config`` is supplied AND ``should_capture(game_idx)``
    returns True, snapshots are recorded after each priority moment
    and saved to ``replay_path(base_seed, game_idx)`` after the game
    completes. Mirrors data_gen's capture pattern so the same replay
    files / format are produced for self-play and eval games.
    """
    if deck_pool is None:
        raise ValueError(
            "deck_pool is required; the synthetic mono-color G/R "
            "fallback was removed. Pass a real deck pool (typically "
            "loaded via ``mtg_bot.decks.decks_from_json`` from "
            "``data/17lands/eoe_decks.json.gz``)."
        )
    engine_seed = base_seed + game_idx
    seat_agents = [
        RandomAgent(rng=random.Random(engine_seed + 1)),
        RandomAgent(rng=random.Random(engine_seed + 2)),
    ]
    deck_p0, deck_p1 = _sample_decks_for_game(
        deck_pool, game_idx, base_seed,
    )
    engine = setup_game(
        seat_agents,
        decks=[deck_p0.cards, deck_p1.cards],
        seed=engine_seed,
        enable_mulligans=enable_mulligans,
    )
    adapter = MtgGameStateAdapter(
        engine, expose_combat_decisions=True,
        heuristics=heuristics,
    )
    strategies = {
        0: p0_factory(base_seed + 100 * game_idx),
        1: p1_factory(base_seed + 100 * game_idx + 50),
    }
    # Pin torch CPU threads to 1 in this worker process. Strategy
    # factories that use the NN evaluator have just imported torch;
    # without this pin, every worker inherits the default of "host
    # CPU count" threads and 30 workers x 32 MKL threads thrash the
    # 32 host cores (~30-40% effective utilization). The trainer
    # has the analogous pin (commit 3ad808f); this is the worker-
    # side equivalent. No-op when torch isn't loaded (random /
    # heuristic strategies); idempotent across games.
    pin_worker_torch_threads()
    rng = random.Random(base_seed + 100 * game_idx + 99)

    recorder: Optional[ReplayRecorder] = None
    if replay_config is not None and replay_config.should_capture(game_idx):
        recorder = replay_config.build_recorder()
        # Capture the initial state too so the first scrubber frame
        # shows "before any actions taken." Mirrors data_gen.
        recorder.record(adapter.engine_state)

    tree_depth = (
        replay_config.tree_depth if replay_config is not None else 1
    )
    tree_top_k = (
        replay_config.tree_top_k if replay_config is not None else None
    )
    if per_game_timeout_sec is not None and per_game_timeout_sec > 0:
        from mtg_bot.training._hang_progress import HangProgress
        from mtg_bot.training._timeout import run_with_timeout
        progress = HangProgress()
        try:
            winner, decisions = run_with_timeout(
                _play_capped_game,
                (adapter, strategies, rng, decision_cap, recorder,
                 progress, tree_depth, tree_top_k),
                per_game_timeout_sec,
            )
        except TimeoutError as exc:
            # Drop a structured hang dump before the outer worker-
            # crash sentinel handler converts this to a (None, 0)
            # tuple. ``hang_dump_dir`` is opt-in; without it the
            # timeout still works exactly as before.
            if hang_dump_dir is not None:
                from mtg_bot.training._hang_dump import (
                    dump_hang_diagnostics,
                )
                dump_hang_diagnostics(
                    hang_dump_dir,
                    base_seed=base_seed,
                    game_idx=game_idx,
                    engine_seed=engine_seed,
                    error_type="TimeoutError",
                    error_message=str(exc),
                    extra={
                        "phase": "eval",
                        "decision_cap": decision_cap,
                        "enable_mulligans": enable_mulligans,
                        "per_game_timeout_sec": per_game_timeout_sec,
                        "progress": progress.snapshot(),
                    },
                )
            raise
    else:
        winner, decisions = _play_capped_game(
            adapter, strategies, rng, decision_cap, recorder,
            tree_depth=tree_depth, tree_top_k=tree_top_k,
        )

    if recorder is not None and replay_config is not None:
        replay = recorder.to_replay(metadata={
            "game_idx": game_idx,
            "base_seed": base_seed,
            "engine_seed": engine_seed,
            "winner": winner,
            "n_decisions": len(recorder.snapshots),
            "captured_at": _now_iso(),
            "annotated_with_nn_version": replay_config.nn_version_label,
        })
        save_replay(replay, replay_config.replay_path(base_seed, game_idx))

    return winner, decisions


def _aggregate_results(
    results: list[tuple],
    n_games: int,
) -> MatchStats:
    """Aggregate per-game results into a ``MatchStats``.

    Accepts either the legacy 2-tuple ``(winner, decisions)`` (from
    the synchronous ``_play_one_match_game`` path which doesn't
    catch worker exceptions) or the 3-tuple ``(winner, decisions,
    outcome_kind)`` produced by the pool workers. The trailing
    ``outcome_kind`` is one of ``"completed"`` / ``"crashed"`` /
    ``"timed_out"`` — both crashed and timed_out are EXCLUDED
    entirely from ``wins`` / ``draws`` / ``games`` /
    ``avg_decisions`` (they contribute zero signal so they
    shouldn't pollute the win rate) and counted separately under
    ``MatchStats.crashed`` / ``MatchStats.timed_out``.

    Real draws (``winner=None``, ``decisions=decision_cap > 0``,
    outcome_kind=``"completed"``) keep the existing behavior —
    counted in ``draws`` and ``games``.
    """
    wins: dict[PlayerID, int] = {0: 0, 1: 0}
    draws = 0
    crashed = 0
    timed_out = 0
    total_decisions = 0
    for entry in results:
        if len(entry) >= 3:
            winner, decisions, outcome_kind = entry[0], entry[1], entry[2]
        else:
            winner, decisions = entry[0], entry[1]
            outcome_kind = (
                "crashed"
                if winner is None and decisions == 0
                else "completed"
            )
        if outcome_kind == "crashed":
            crashed += 1
            continue
        if outcome_kind == "timed_out":
            timed_out += 1
            continue
        total_decisions += decisions
        if winner == 0:
            wins[0] += 1
        elif winner == 1:
            wins[1] += 1
        else:
            # Real draw: decision-cap timeout (winner=None,
            # decisions=decision_cap > 0).
            draws += 1
    completed = wins[0] + wins[1] + draws
    return MatchStats(
        wins=wins,
        draws=draws,
        games=completed,
        avg_decisions=total_decisions / completed if completed else 0.0,
        crashed=crashed,
        timed_out=timed_out,
    )


def _play_capped_game(
    adapter: MtgGameStateAdapter,
    strategies: dict[PlayerID, "object"],
    rng: random.Random,
    decision_cap: int,
    replay_recorder: Optional[ReplayRecorder] = None,
    progress: Optional["object"] = None,
    tree_depth: int = 1,
    tree_top_k: Optional[int] = None,
) -> tuple[PlayerID | None, int]:
    """Play one game with a hard decision cap. Returns
    ``(winner_id_or_None, decisions_made)``. If the cap is hit
    without termination, returns ``(None, decision_cap)`` — caller
    counts that as a draw.

    When ``replay_recorder`` is supplied, captures a snapshot of
    the engine state after each applied decision (the initial
    pre-action state is captured by the caller). Snapshot cost is
    ~5 ms per priority moment, dwarfed by typical PIMC decision
    time, so capture-on means a small wall-clock hit but no skew.
    """
    decisions = 0
    # Skip the per-decision adapter clone for observe_action wiring
    # when no strategy implements it (the common case for PIMC /
    # heuristic / random eval). Cloning is ~milliseconds per priority
    # moment but compounds across thousands of decisions — only pay
    # it for belief-tracking ISMCTS strategies. Mirrors the same
    # guard in data_gen._play_one_game_recording.
    observe_action_active = any(
        hasattr(s, "observe_action") for s in strategies.values()
    )
    while not adapter.is_terminal() and decisions < decision_cap:
        active = adapter.active_player()
        # Search-tree annotation captured this iteration; threaded into
        # the recorder right after the snapshot is appended. Stays None
        # for chance moves and for non-recording runs so we don't pay
        # the wallclock-time bookkeeping when nothing reads it.
        search_annotation: Optional[dict[str, Any]] = None
        if active == CHANCE_PLAYER:
            outcomes = adapter.chance_outcomes()
            choices = [d for d, _ in outcomes]
            weights = [w for _, w in outcomes]
            decision = rng.choices(choices, weights=weights, k=1)[0]
        elif replay_recorder is not None:
            # When recording, pull rich stats out of the search in a
            # single call so the replay carries enough context for
            # the viewer's decision-detail panel. Strategies that
            # don't expose ``choose_with_full_stats`` (duck-typed test
            # baselines, RandomChoice subclasses) fall through to the
            # plain ``choose`` path — the snapshot's ``search`` field
            # stays None, matching the viewer's hide-when-absent
            # contract.
            strategy = strategies[active]
            deep_capture = tree_depth > 1
            subtree_fn = (
                getattr(strategy, "choose_with_subtree", None)
                if deep_capture else None
            )
            full_stats_fn = getattr(strategy, "choose_with_full_stats", None)
            if callable(subtree_fn):
                _t0 = time.perf_counter()
                chosen, candidates = subtree_fn(
                    adapter, active,
                    tree_depth=tree_depth, tree_top_k=tree_top_k,
                )
                if chosen is not None:
                    decision = chosen
                else:
                    decision = strategy.choose(adapter, active)
                if candidates:
                    elapsed_ms = int((time.perf_counter() - _t0) * 1000)
                    search_annotation = build_search_annotation(
                        chosen=decision,
                        candidates=candidates,
                        strategy=strategy,
                        elapsed_ms=elapsed_ms,
                        deep_capture=True,
                    )
            elif callable(full_stats_fn):
                _t0 = time.perf_counter()
                chosen, candidates = full_stats_fn(adapter, active)
                if chosen is not None:
                    decision = chosen
                else:
                    decision = strategy.choose(adapter, active)
                if candidates:
                    elapsed_ms = int((time.perf_counter() - _t0) * 1000)
                    search_annotation = build_search_annotation(
                        chosen=decision,
                        candidates=candidates,
                        strategy=strategy,
                        elapsed_ms=elapsed_ms,
                    )
            else:
                decision = strategy.choose(adapter, active)
        else:
            decision = strategies[active].choose(adapter, active)
        if observe_action_active and active != CHANCE_PLAYER:
            pre_apply_state = adapter.clone()
        else:
            pre_apply_state = None
        adapter.apply(decision)
        if pre_apply_state is not None:
            for seat in (0, 1):
                strategy = strategies.get(seat)
                if strategy is None:
                    continue
                if hasattr(strategy, "observe_action"):
                    strategy.observe_action(
                        pre_apply_state, decision, active,
                    )
        decisions += 1
        if replay_recorder is not None:
            replay_recorder.record(adapter.engine_state, decision=decision)
            if search_annotation is not None:
                # Attach to the PRE-apply snapshot (state the search
                # ran against), not the just-recorded post-apply snap.
                # See the matching block in ``data_gen`` for the full
                # rationale.
                replay_recorder.record_search_annotation(
                    len(replay_recorder.snapshots) - 2,
                    search_annotation,
                )
                # Soft size sanity check on deep-capture annotations.
                # No-op for production root-only captures; only fires
                # when ``children`` is populated.
                replay_recorder.maybe_warn_deep_annotation_size(
                    search_annotation,
                )
        # Update the optional shared progress sentinel — see
        # mtg_bot.training._hang_progress for rationale. Pure attribute
        # writes; no allocations on the steady-state path.
        if progress is not None:
            try:
                progress.record(active, decision, -1)
            except Exception:  # noqa: BLE001
                # Diagnostic capture must never break the game loop.
                pass
    if adapter.is_terminal():
        return adapter.winner(), decisions
    return None, decisions
