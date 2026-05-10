"""Self-play data generation: play games, record (features, outcome).

Produces the training data for Phase D's value-network training. For
every non-chance decision point in every game, captures features from
**both** player perspectives and labels them with the eventual game
outcome from that player's perspective. Same state, two viewers,
two labels — doubles the data with no extra game-play cost.

Storage is ``np.savez_compressed`` (one .npz per generation run).
Multiple .npz files in a directory can be loaded and concatenated
via ``load_all_datasets`` so the trainer can combine sessions.

The strategies used to play the games matter — random play produces
garbage labels (outcome doesn't reflect position quality). Phase D
typically uses PIMC + heuristic on both sides; later phases bootstrap
by using the previously-trained NN to play the next round of games
(the AlphaZero loop).

Two driver functions:

- ``generate_training_data`` runs games sequentially. Used by tests
  (small N, want simple code path) and one-off uses where setup
  overhead matters more than wall-clock parallelism.
- ``generate_training_data_parallel`` distributes games across a
  ``multiprocessing.Pool`` of workers. Both produce identical
  arrays for the same ``base_seed`` because per-game seeds are
  derived from ``game_idx``, not from worker assignment — both
  paths call the same per-game helper. This determinism is locked
  in by a test.

Optional replay capture: pass a ``ReplayCaptureConfig`` to either
driver to write a per-game replay file alongside the training-data
arrays. Useful for debugging engine gaps — see
``mtg_bot.training.replay_capture``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from mtg_bot.agents import RandomAgent
from mtg_bot.decks import Deck
from mtg_bot.features import FEATURE_DIM, extract_features
from mtg_bot.replay import save_replay
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.heuristics_config import HeuristicsConfig
from mtg_bot.search.protocol import CHANCE_PLAYER, PlayerID
from mtg_bot.simulate import setup_game
from mtg_bot.training._worker_threads import pin_worker_torch_threads
from mtg_bot.training.config import StrategyFactory
from mtg_bot.training.progress import DataGenProgress
from mtg_bot.training.replay_capture import (
    ReplayRecorder, build_search_annotation, heuristic_eval_annotator,
    nn_eval_annotator,
)


_DEFAULT_DECISION_CAP = 4000

# Type alias: per-game chunk + winner + outcome-kind label. Winner
# is the PlayerID of the game's victor or None for a draw / crash /
# timeout; the tracker uses it to compute the live winrate without
# having to re-parse the outcomes array (which holds per-viewer
# signs that look symmetric). The trailing string is one of
# "completed" / "crashed" / "timed_out" so the driver can
# distinguish engine-level exceptions from per-game wall-clock
# timeouts (different root causes that both produce a winner=None
# empty chunk — the UI now reports them as separate counters).
_GameChunk = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[PlayerID], str,
]

_DTYPE_FEATURES = np.float32
_DTYPE_OUTCOMES = np.float32
# Policy targets: per-slot probabilities. float32 keeps storage
# matched to features; precision well above what cross-entropy
# needs. Must match the dtype the AlphaZero trainer feeds into
# the loss function — float64 would be a wasteful upcast.
_DTYPE_POLICY = np.float32


@dataclass(frozen=True)
class ReplayCaptureConfig:
    """Picklable config for capturing replays during data gen.

    ``save_dir``: where to write replay files (one .json.gz per
    captured game).
    ``capture_every``: capture every Nth game starting at
    game_idx 0 (so capture_every=10 captures games 0, 10, 20, ...).
    Default 1 = capture all games.
    ``nn_weights_path``: optional path to NN weights — when set,
    each snapshot also gets an ``nn_eval`` annotation alongside the
    always-on ``heuristic_eval``.
    ``nn_version_label``: optional human-readable label (typically
    a registry version like "v3") stored in each ``nn_eval`` so
    consumers know which network produced the scores.
    ``filename_prefix``: prepended to each replay filename. Useful
    for tagging a batch of replays from one experiment.

    Frozen + primitive types so it crosses the multiprocessing spawn
    boundary cleanly.
    """

    save_dir: str
    capture_every: int = 1
    nn_weights_path: Optional[str] = None
    nn_version_label: Optional[str] = None
    filename_prefix: str = "game"
    # Architecture overrides for the NN annotator. Required when the
    # captured run uses anything other than the legacy default trunk
    # shape (modern iterate runs all do — v6 is 1024-1024-512). Stored
    # as tuple so the dataclass stays frozen + picklable across the
    # multiprocessing boundary.
    nn_hidden_sizes: Optional[tuple[int, ...]] = None
    nn_with_policy: bool = False
    # Dual-head architecture for the NN annotator. ``None`` means
    # "use NNEvaluator's default" (``"shared"``); a separate-trunks
    # checkpoint requires ``"separate"``. Threaded through from the
    # version's metadata.architecture by iterate / evaluate scripts.
    nn_architecture: Optional[str] = None
    nn_policy_hidden_sizes: Optional[tuple[int, ...]] = None
    # Diagnostic deep-tree capture. Defaults reproduce production
    # root-only behavior (depth=1, top_k irrelevant). The diagnostic
    # script ``capture_annotated_replays.py`` flips these via the
    # ``--capture-tree-depth`` / ``--capture-top-k`` flags so the
    # replay viewer can drill into the search subtree under any
    # candidate. Deep mode skips the production size budget — these
    # captures are expected to be larger and aren't shipped to the
    # trainer's replay buffer.
    tree_depth: int = 1
    tree_top_k: Optional[int] = None

    def should_capture(self, game_idx: int) -> bool:
        return game_idx % self.capture_every == 0

    def replay_path(self, base_seed: int, game_idx: int) -> Path:
        return Path(self.save_dir) / (
            f"{self.filename_prefix}_seed{base_seed}_idx{game_idx:04d}.json.gz"
        )

    def build_recorder(self) -> ReplayRecorder:
        annotators = [heuristic_eval_annotator()]
        if self.nn_weights_path:
            annotators.append(nn_eval_annotator(
                Path(self.nn_weights_path),
                version_label=self.nn_version_label,
                hidden_sizes=self.nn_hidden_sizes,
                with_policy=self.nn_with_policy,
                architecture=self.nn_architecture,
                policy_hidden_sizes=self.nn_policy_hidden_sizes,
            ))
        return ReplayRecorder(annotators=annotators)


def _empty_arrays() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Empty arrays matching the post-A2.3 4-tuple schema. Use for
    the no-games / no-samples paths in the data-gen functions.
    Lazy-imports ``ACTION_VOCAB_SIZE`` to avoid the ``training`` →
    ``training.action_encoding`` dep at module-load time."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    return (
        np.zeros((0, FEATURE_DIM), dtype=_DTYPE_FEATURES),
        np.zeros((0,), dtype=_DTYPE_OUTCOMES),
        np.zeros((0, ACTION_VOCAB_SIZE), dtype=_DTYPE_POLICY),
        np.zeros((0, ACTION_VOCAB_SIZE), dtype=np.bool_),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample_decks_for_game(
    deck_pool: list[Deck], game_idx: int, base_seed: int,
) -> tuple[Deck, Deck]:
    """Deterministically pick 2 decks (one per seat) from the pool
    for game ``game_idx``. Uses a per-game RNG seeded from
    ``base_seed`` + a fixed offset so the choice is reproducible
    AND independent of ``game_idx``'s usage in strategy/engine seeds.

    Single-deck pool degenerates into a mirror match (same deck on
    both seats) — explicit so callers know it's not an error. Empty
    pool raises.
    """
    if not deck_pool:
        raise ValueError("deck_pool is empty")
    rng = random.Random(base_seed + 100 * game_idx + 12345)
    if len(deck_pool) >= 2:
        a, b = rng.sample(range(len(deck_pool)), 2)
    else:
        a = b = 0
    return deck_pool[a], deck_pool[b]


def generate_training_data(
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    n_games: int,
    base_seed: int = 0,
    decision_cap: int = _DEFAULT_DECISION_CAP,
    sample_every: int = 1,
    replay_config: Optional[ReplayCaptureConfig] = None,
    deck_pool: Optional[list[Deck]] = None,
    with_policy: bool = False,
    progress: Optional[DataGenProgress] = None,
    enable_mulligans: bool = False,
    per_game_timeout_sec: Optional[float] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
) -> tuple[np.ndarray, ...]:
    """Play ``n_games`` and return training arrays.

    For each non-chance decision (every Nth, ``sample_every``),
    captures ``extract_features(state, viewer)`` for both viewers
    (0 and 1) and labels each with the eventual game outcome from
    that viewer's perspective (+1/-1/0).

    Default returns ``(features, outcomes)`` — the value-only
    schema that pre-A2.3 callers expect. Pass ``with_policy=True``
    for the AlphaZero schema:

    Returns a 2-tuple (default) or 4-tuple (with_policy=True):
        features:       ``(N, FEATURE_DIM)`` float32.
        outcomes:       ``(N,)`` float32 in ``{-1, 0, +1}``.
        policy_targets: ``(N, ACTION_VOCAB_SIZE)`` float32 — visit
            distribution per row, all-zero for rows without policy
            supervision (data-doubled non-active-viewer rows;
            baseline-strategy rows).
        action_masks:   ``(N, ACTION_VOCAB_SIZE)`` bool — legal-
            slot mask per row, all-False for rows without policy
            supervision.

    Deterministic in ``base_seed`` — same call produces the same
    arrays. Trainer code should detect "no policy supervision" via
    ``policy_targets.sum(-1) > 0.5`` per row.
    """
    if heuristics is None:
        heuristics = HeuristicsConfig()
    if progress is not None:
        progress.begin()
    chunks: list[_GameChunk] = []
    for i in range(n_games):
        chunk = _generate_one_game_data(
            game_idx=i,
            p0_factory=p0_factory, p1_factory=p1_factory,
            base_seed=base_seed,
            decision_cap=decision_cap, sample_every=sample_every,
            replay_config=replay_config,
            deck_pool=deck_pool,
            enable_mulligans=enable_mulligans,
            per_game_timeout_sec=per_game_timeout_sec,
            hang_dump_dir=hang_dump_dir,
            heuristics=heuristics,
        )
        chunks.append(chunk)
        if progress is not None:
            progress.record_game(
                n_samples=int(chunk[0].shape[0]),
                outcome_p0=_winner_to_p0_outcome(chunk[4]),
            )
    f, o, p, m = _concat_chunks([c[:4] for c in chunks])
    if with_policy:
        return f, o, p, m
    return f, o


def generate_training_data_parallel(
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    n_games: int,
    base_seed: int = 0,
    n_workers: Optional[int] = None,
    decision_cap: int = _DEFAULT_DECISION_CAP,
    sample_every: int = 1,
    replay_config: Optional[ReplayCaptureConfig] = None,
    deck_pool: Optional[list[Deck]] = None,
    with_policy: bool = False,
    progress: Optional[DataGenProgress] = None,
    enable_mulligans: bool = False,
    per_game_timeout_sec: Optional[float] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    action_encoding_version: str = "positional_v1",
) -> tuple[np.ndarray, ...]:
    """Same contract as ``generate_training_data`` but distributes
    games across a ``multiprocessing.Pool``. Same ``with_policy``
    flag controlling 2-tuple vs 4-tuple return.

    ``stop_check`` is an optional zero-arg callable polled between
    completed-game results (one per imap_unordered yield). When it
    returns ``True``, the driver stops draining the pool, terminates
    the workers, and returns whatever results have already landed
    plus an empty-chunk tail for the un-played game indices. The
    progress tracker (if supplied) is finalized with status
    ``"stopped"`` so the UI sees the run as user-aborted rather than
    completed/failed. In-flight games run to natural completion
    (``pool.terminate()`` with timeout would lose their data); we
    simply stop dispatching new ones. v1 design — not mid-game
    responsive; see ``docs/SEARCH_LIMITATIONS.md``.

    Identical output to the sequential function for the same
    ``base_seed`` — per-game seeds are derived from ``game_idx``, not
    worker assignment, so the same game produces the same samples
    no matter which worker plays it. We use ``imap_unordered`` so
    the live progress tracker can report completions as they land,
    then sort the per-game chunks by ``game_idx`` before concatenation
    so the output arrays remain deterministic in ``base_seed``
    (locked in by tests).

    ``n_workers`` defaults to ``cpu_count() - 2`` (leaves a few cores
    for OS / Python main / the user's other work). Pass an explicit
    int to override.

    Caller note: factories must be picklable. Module-level functions
    (e.g. ``random_choice_factory``) and bound methods of picklable
    dataclasses (e.g. ``PimcConfig().build``) work; lambdas and
    nested closures don't. The CLI builds factories the right way.

    Worker spawn cost on Windows is ~1-2s per worker, paid once per
    pool. Dominates wall clock at very small ``n_games``; amortizes
    away past a few dozen games.
    """
    if heuristics is None:
        heuristics = HeuristicsConfig()
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 2)
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if n_games == 0:
        f, o, p, m = _empty_arrays()
        return (f, o, p, m) if with_policy else (f, o)

    args_list = [
        (
            i, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            enable_mulligans, per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version,
        )
        for i in range(n_games)
    ]
    if progress is not None:
        progress.begin()
    # Pool() with no start method falls back to the platform default —
    # spawn on Windows, fork on Unix. Workers persist for pool
    # lifetime so import cost is paid once per worker.
    #
    # ``imap_unordered`` instead of ``map`` so completions stream
    # back as workers finish, letting the progress tracker emit
    # snapshots throughout the run rather than only at the end.
    # Results arrive out of order; we tag each per-game result with
    # its ``game_idx`` and sort at the end so the concatenated arrays
    # are deterministic in ``base_seed`` (locked in by tests).
    indexed: list[tuple[int, _GameChunk]] = []
    crashed_games: list[int] = []
    timed_out_games: list[int] = []
    stopped_early = False
    with mp.Pool(processes=n_workers) as pool:
        for game_idx, chunk in pool.imap_unordered(
            _play_one_parallel_game, args_list, chunksize=1,
        ):
            indexed.append((game_idx, chunk))
            # ``_play_one_parallel_game`` converts both engine-level
            # crashes AND per-game timeouts into empty chunks
            # (winner=None, 0 samples) but tags each with an outcome
            # kind ("crashed" / "timed_out" / "completed") in
            # ``chunk[5]``. We track them separately so the live
            # progress UI can show real exceptions vs slow-game
            # timeouts as distinct counters — same root cause for
            # both is "no signal", different diagnostic implications.
            outcome_kind = chunk[5] if len(chunk) > 5 else "completed"
            crashed = outcome_kind == "crashed"
            timed_out = outcome_kind == "timed_out"
            if crashed:
                crashed_games.append(game_idx)
            elif timed_out:
                timed_out_games.append(game_idx)
            if progress is not None:
                if crashed:
                    progress.record_game(crashed=True)
                elif timed_out:
                    progress.record_game(timed_out=True)
                else:
                    progress.record_game(
                        n_samples=int(chunk[0].shape[0]),
                        outcome_p0=_winner_to_p0_outcome(chunk[4]),
                    )
            # User-requested stop: terminate the pool's pending work
            # and break out, preserving the data we already have. We
            # call ``pool.terminate()`` rather than letting the pool
            # drain naturally because the queued args could be
            # hundreds of games (~hours of wall-time on a slow eval
            # bench) and the user wants the run to end now. In-flight
            # games on the workers when terminate() lands DO get
            # killed mid-game — their partial progress is lost, but
            # everything that's already arrived in ``indexed`` is
            # safe. Documented divergence: stops fire at imap-yield
            # boundaries, not mid-game decision points.
            if stop_check is not None and stop_check():
                stopped_early = True
                pool.terminate()
                pool.join()
                break
    if crashed_games:
        import sys
        print(
            f"# WARNING: {len(crashed_games)}/{n_games} games crashed "
            f"in the worker pool and were dropped (engine state "
            f"corruption — see worker stderr above for tracebacks). "
            f"Affected game indices: {crashed_games[:20]}"
            f"{' ...' if len(crashed_games) > 20 else ''}",
            file=sys.stderr,
        )
    if timed_out_games:
        import sys
        print(
            f"# WARNING: {len(timed_out_games)}/{n_games} games hit "
            f"the per-game wall-clock timeout and were dropped (slow "
            f"games under CPU contention or genuine engine hangs). "
            f"Affected game indices: {timed_out_games[:20]}"
            f"{' ...' if len(timed_out_games) > 20 else ''}",
            file=sys.stderr,
        )
    indexed.sort(key=lambda t: t[0])
    chunks = [c for _, c in indexed]
    f, o, p, m = _concat_chunks([c[:4] for c in chunks])
    if stopped_early and progress is not None:
        # Caller will see the early return; finalize the tracker so
        # the UI labels the run as user-aborted rather than leaving
        # it 'running' forever. Caller-driven finalize is also
        # supported (caller may want extra metadata) — that's a
        # no-op idempotent overwrite.
        progress.finalize(status="stopped")
    if with_policy:
        return f, o, p, m
    return f, o


def _play_one_parallel_game(
    args: tuple,
) -> tuple[int, _GameChunk]:
    """Top-level pickleable worker. Plays one game, returns the
    ``(game_idx, chunk)`` pair where ``chunk`` is the 5-tuple
    ``(features, outcomes, policy_targets, action_masks, winner)``.
    The driver uses ``game_idx`` to re-sort results from
    ``imap_unordered`` so the concatenated arrays stay deterministic.
    Lives at module top level (not nested) so multiprocessing can
    pickle it across the spawn boundary.

    Engine-level errors (state corruption bugs that surface at
    scale — ~1 in 1000 games for heuristic-vs-heuristic PIMC as of
    2026-05-03) are caught here and converted to an empty chunk
    so a single bad game doesn't take down the entire pool.
    Without this, ``pool.imap_unordered`` re-raises the worker
    exception and aborts the whole run, losing all in-flight
    sample data (``save_dataset`` only writes after every game
    completes). The traceback is logged to worker stderr so
    failures are visible in the pool's stderr relay; the
    parent's progress tracker still records the game as
    "completed" with 0 samples / no winner.

    Per-game wall-clock ``TimeoutError`` (out of
    ``run_with_timeout``) is caught separately so the driver can
    bucket it under the dedicated ``timed_out`` counter rather than
    ``crashed``. Both produce an empty chunk with ``winner=None``;
    the trailing ``outcome_kind`` element of the chunk tuple is
    what the driver reads to tell them apart.
    """
    # ``action_encoding_version`` was appended to the args tuple in
    # ``generate_training_data_parallel`` (feature/iterate-v2-wiring) so
    # iterate's bootstrap-metadata read flows all the way through to the
    # per-game policy-target encoder. Older callers that pre-date the
    # field (none in tree, but for forward-compat with hand-built tuples
    # in tests) fall through to the v1 default.
    if len(args) == 13:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            enable_mulligans, per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version,
        ) = args
    else:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            enable_mulligans, per_game_timeout_sec, hang_dump_dir,
            heuristics,
        ) = args
        action_encoding_version = "positional_v1"
    try:
        chunk = _generate_one_game_data(
            game_idx=game_idx,
            p0_factory=p0_factory, p1_factory=p1_factory,
            base_seed=base_seed,
            enable_mulligans=enable_mulligans,
            per_game_timeout_sec=per_game_timeout_sec,
            decision_cap=decision_cap, sample_every=sample_every,
            replay_config=replay_config,
            deck_pool=deck_pool,
            hang_dump_dir=hang_dump_dir,
            heuristics=heuristics,
            action_encoding_version=action_encoding_version,
        )
    except TimeoutError as exc:
        import sys
        print(
            f"[worker] game {game_idx} timed out "
            f"({exc}); dropping this game, pool continues.",
            file=sys.stderr,
        )
        f, o, p, m = _empty_arrays()
        chunk = (f, o, p, m, None, "timed_out")
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        import sys
        import traceback
        print(
            f"[worker] game {game_idx} crashed "
            f"({type(exc).__name__}: {exc}); dropping this game, "
            f"pool continues.",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        f, o, p, m = _empty_arrays()
        chunk = (f, o, p, m, None, "crashed")
    return game_idx, chunk


def _winner_to_p0_outcome(winner: Optional[PlayerID]) -> float:
    """Convert a per-game winner (0, 1, or None) into the +1 / -1 / 0
    outcome from p0's perspective. Used by the progress tracker to
    keep a live winrate without re-parsing ``outcomes`` (which holds
    per-viewer signs that look symmetric in aggregate)."""
    if winner is None:
        return 0.0
    return 1.0 if winner == 0 else -1.0


def _generate_one_game_data(
    *,
    game_idx: int,
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    base_seed: int,
    decision_cap: int,
    sample_every: int,
    replay_config: Optional[ReplayCaptureConfig] = None,
    deck_pool: Optional[list[Deck]] = None,
    enable_mulligans: bool = False,
    per_game_timeout_sec: Optional[float] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
    action_encoding_version: str = "positional_v1",
) -> _GameChunk:
    """Play one game and return its 4-tuple training arrays.

    The single source of truth for per-game data extraction; both
    sequential and parallel drivers call this so they produce
    bit-identical results for the same ``(game_idx, base_seed,
    factories)``.

    Two distinct decks are sampled from ``deck_pool`` deterministically
    (per-game seed) and used as the seats' libraries. ``deck_pool`` is
    required — the synthetic mono-color G/R fallback was removed; every
    self-play / eval data point now comes from a real deck distribution
    by construction.

    Starting-seat alternation: for ``game_idx % 2 == 1``, player 1
    starts on the play instead of player 0. Going first is
    structurally advantageous in MTG, so without alternation the
    seat-0 wins-on-the-play bias bleeds into the value labels (v6
    self-play hit 80.8% seat-0 wins, biasing every viewer=0 turn-1
    feature row toward +1 and every viewer=1 turn-1 row toward -1
    — the model learns "I'm seated at seat 0 → I win" rather than
    evaluating position quality). The features are seat-symmetric
    (see ``mtg_bot.features``) so flipping who starts averages the
    advantage out across game_idx without remapping winners or
    swapping strategies; the data is per-viewer-perspective and the
    label is ``winner == viewer``, which is correct regardless of
    who started.

    When ``replay_config`` is set and ``should_capture(game_idx)``
    returns True, also writes a per-game replay file to disk before
    returning. The annotators (heuristic + optional NN eval) are
    built fresh per game in the worker — NN model load happens
    once per captured game (a few ms; negligible vs game length).
    """
    if deck_pool is None:
        raise ValueError(
            "deck_pool is required; the synthetic mono-color G/R "
            "fallback was removed. Pass a real deck pool (typically "
            "loaded via ``mtg_bot.decks.decks_from_json`` from "
            "``data/17lands/eoe_decks.json.gz``)."
        )
    engine_seed = base_seed + game_idx
    deck_p0, deck_p1 = _sample_decks_for_game(
        deck_pool, game_idx, base_seed,
    )
    engine = setup_game(
        [
            RandomAgent(rng=random.Random(engine_seed + 1)),
            RandomAgent(rng=random.Random(engine_seed + 2)),
        ],
        decks=[deck_p0.cards, deck_p1.cards],
        seed=engine_seed,
        enable_mulligans=enable_mulligans,
    )
    if game_idx % 2 == 1:
        engine.active_player_idx = 1
        engine.priority_player_idx = 1
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
    # without this pin every worker inherits the default of "host
    # CPU count" threads — 30 workers x 32 MKL threads on a 32-core
    # box means ~960 threads thrashing 32 cores (~30-40% effective
    # utilization). The trainer has the same pin on CPU (commit
    # 3ad808f); this is the data-gen worker equivalent. No-op when
    # torch isn't loaded (heuristic-only runs); idempotent across
    # games (set_num_threads is process-global).
    pin_worker_torch_threads()
    rng = random.Random(base_seed + 100 * game_idx + 99)

    recorder: Optional[ReplayRecorder] = None
    if replay_config is not None and replay_config.should_capture(game_idx):
        recorder = replay_config.build_recorder()
        # Capture the initial state too so the replay viewer's first
        # snapshot reflects "game 0, before any actions taken."
        recorder.record(adapter.engine_state)

    tree_depth = (
        replay_config.tree_depth if replay_config is not None else 1
    )
    tree_top_k = (
        replay_config.tree_top_k if replay_config is not None else None
    )
    if per_game_timeout_sec is not None and per_game_timeout_sec > 0:
        from mtg_bot.training._timeout import run_with_timeout
        from mtg_bot.training._hang_progress import HangProgress
        progress = HangProgress()
        try:
            winner, samples = run_with_timeout(
                _play_one_game_recording,
                (adapter, strategies, rng, decision_cap, sample_every,
                 recorder, progress, tree_depth, tree_top_k,
                 action_encoding_version),
                per_game_timeout_sec,
            )
        except TimeoutError as exc:
            # Drop a structured hang dump before letting the worker-
            # crash sentinel handler convert this to an empty chunk.
            # ``hang_dump_dir`` is opt-in (default off) — without it
            # the timeout still works exactly as before.
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
                        "phase": "data_gen",
                        "decision_cap": decision_cap,
                        "sample_every": sample_every,
                        "enable_mulligans": enable_mulligans,
                        "per_game_timeout_sec": per_game_timeout_sec,
                        "progress": progress.snapshot(),
                    },
                )
            raise
    else:
        winner, samples = _play_one_game_recording(
            adapter, strategies, rng, decision_cap, sample_every,
            replay_recorder=recorder,
            tree_depth=tree_depth, tree_top_k=tree_top_k,
            action_encoding_version=action_encoding_version,
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

    if not samples:
        f, o, p, m = _empty_arrays()
        return f, o, p, m, winner, "completed"
    outcome_label = _outcome_label(winner)
    features = np.stack(
        [f for f, _, _, _ in samples], axis=0,
    ).astype(_DTYPE_FEATURES)
    outcomes = np.asarray(
        [outcome_label[v] for _, v, _, _ in samples],
        dtype=_DTYPE_OUTCOMES,
    )
    policy_targets = np.stack(
        [p for _, _, p, _ in samples], axis=0,
    ).astype(_DTYPE_POLICY)
    action_masks = np.stack(
        [m for _, _, _, m in samples], axis=0,
    ).astype(np.bool_)
    return (
        features, outcomes, policy_targets, action_masks, winner,
        "completed",
    )


def _concat_chunks(
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concat per-game 4-tuple chunks into the final arrays. Skips
    empty chunks (zero-decision games hit the cap)."""
    nonempty = [c for c in chunks if c[0].size > 0]
    if not nonempty:
        return _empty_arrays()
    features = np.concatenate(
        [c[0] for c in nonempty], axis=0,
    ).astype(_DTYPE_FEATURES)
    outcomes = np.concatenate(
        [c[1] for c in nonempty], axis=0,
    ).astype(_DTYPE_OUTCOMES)
    policy_targets = np.concatenate(
        [c[2] for c in nonempty], axis=0,
    ).astype(_DTYPE_POLICY)
    action_masks = np.concatenate(
        [c[3] for c in nonempty], axis=0,
    ).astype(np.bool_)
    return features, outcomes, policy_targets, action_masks


def _outcome_label(
    winner: PlayerID | None,
) -> dict[PlayerID, float]:
    """Per-viewer outcome label. Draw → 0 for both."""
    if winner is None:
        return {0: 0.0, 1: 0.0}
    return {0: 1.0 if winner == 0 else -1.0,
            1: 1.0 if winner == 1 else -1.0}


def _play_one_game_recording(
    adapter: MtgGameStateAdapter,
    strategies: dict[PlayerID, "object"],
    rng: random.Random,
    decision_cap: int,
    sample_every: int,
    replay_recorder: Optional[ReplayRecorder] = None,
    progress: Optional["object"] = None,
    tree_depth: int = 1,
    tree_top_k: Optional[int] = None,
    action_encoding_version: str = "positional_v1",
) -> tuple[
    PlayerID | None,
    list[tuple[np.ndarray, PlayerID, np.ndarray, np.ndarray]],
]:
    """Play one game with the decision cap, recording features at
    every Nth non-chance decision from both player perspectives.

    When ``replay_recorder`` is set, also captures a snapshot of
    the engine state after each ``adapter.apply(decision)`` — both
    chance and player decisions. The recorder's annotators (eval
    values, etc.) run inline.

    Returns ``(winner_or_None, samples)`` where each sample is a
    4-tuple ``(features, viewer_id, policy_target, action_mask)``:

    - ``features``      : ``(FEATURE_DIM,)`` float32, viewer-side.
    - ``viewer_id``     : the player whose perspective this is.
    - ``policy_target`` : ``(ACTION_VOCAB_SIZE,)`` float32. For
      the active player's row at a sample point where
      ``choose_with_visits`` returned non-empty visits, this is
      the normalized visit distribution encoded into slot indices
      via ``action_encoding.encode``. All zeros for the data-
      doubling non-active-viewer row, and for any row from a
      strategy with no visit data (RandomChoice etc.). The
      trainer detects "no policy supervision" by checking
      ``policy_target.sum() < 0.5``.
    - ``action_mask``   : ``(ACTION_VOCAB_SIZE,)`` bool. True at
      slots that were legal at the sample point. All-False for
      rows without policy supervision (mask is irrelevant to
      value-only loss).
    """
    # Local import — action_encoding pulls in legal-action
    # enumeration, which is heavier than this file's outer surface.
    from mtg_bot.training.action_encoding import (
        ACTION_VOCAB_SIZE, EncodingOverflowError, encode, legal_action_mask,
    )

    samples: list[
        tuple[np.ndarray, PlayerID, np.ndarray, np.ndarray]
    ] = []
    zero_policy = np.zeros(ACTION_VOCAB_SIZE, dtype=_DTYPE_POLICY)
    zero_mask = np.zeros(ACTION_VOCAB_SIZE, dtype=np.bool_)
    # Skip the per-decision adapter clone for observe_action wiring
    # when no strategy implements it (the common case for PIMC /
    # heuristic / random self-play). Cloning is ~milliseconds per
    # priority moment but compounds to seconds across ~3000-decision
    # games — only pay it for belief-tracking ISMCTS strategies.
    observe_action_active = any(
        hasattr(s, "observe_action") for s in strategies.values()
    )
    decision_idx = 0
    while not adapter.is_terminal() and decision_idx < decision_cap:
        active = adapter.active_player()
        # Search-tree annotation captured this iteration; threaded into
        # the recorder right after the snapshot is appended. Stays None
        # for chance moves and non-sample-point regular moves (the
        # cheap ``choose`` path doesn't expose tree stats; recording it
        # would require a second search).
        search_annotation: Optional[dict[str, Any]] = None
        if active == CHANCE_PLAYER:
            outcomes = adapter.chance_outcomes()
            choices = [d for d, _ in outcomes]
            weights = [w for _, w in outcomes]
            decision = rng.choices(choices, weights=weights, k=1)[0]
        else:
            sampling = decision_idx % sample_every == 0
            if sampling:
                # Capture visits, decision, AND (when supported) the
                # full search-tree stats in a SINGLE search.
                # ``choose_with_full_stats`` returns both the chosen
                # decision and per-candidate stats — we derive the
                # visit dict from candidates for policy supervision
                # and feed the candidates into the replay recorder's
                # search annotation. Strategies that don't expose the
                # richer interface (duck-typed test mocks, baseline
                # implementations) fall through to ``choose_with_
                # visits`` — the snapshot then carries ``search=None``
                # for those decisions, matching the contract on the
                # viewer side.
                _t0 = (
                    time.perf_counter() if replay_recorder is not None
                    else None
                )
                strategy = strategies[active]
                # Deep-tree capture: when the replay config asks for
                # depth > 1, prefer ``choose_with_subtree`` so the
                # annotation carries nested ``children`` per
                # candidate. Production captures (default depth=1)
                # take the same ``choose_with_full_stats`` fast path
                # they always have — zero extra work, no shape change.
                deep_capture = tree_depth > 1
                subtree_fn = (
                    getattr(strategy, "choose_with_subtree", None)
                    if deep_capture else None
                )
                full_stats_fn = getattr(strategy, "choose_with_full_stats", None)
                candidates: list[dict[str, Any]] = []
                if callable(subtree_fn):
                    chosen, candidates = subtree_fn(
                        adapter, active,
                        tree_depth=tree_depth, tree_top_k=tree_top_k,
                    )
                    if chosen is not None:
                        decision = chosen
                    else:
                        decision = list(adapter.legal_decisions())[0]
                    visits = {c["decision"]: c["visits"] for c in candidates}
                elif callable(full_stats_fn):
                    chosen, candidates = full_stats_fn(adapter, active)
                    if chosen is not None:
                        decision = chosen
                    else:
                        # No candidates (random/baseline strategy or
                        # search bottomed out). Pick a legal default
                        # without re-running the search.
                        decision = list(adapter.legal_decisions())[0]
                    visits = {c["decision"]: c["visits"] for c in candidates}
                else:
                    visits = strategy.choose_with_visits(adapter, active)
                    if visits:
                        decision = max(
                            visits.items(), key=lambda kv: kv[1],
                        )[0]
                    else:
                        decision = list(adapter.legal_decisions())[0]
                if replay_recorder is not None and candidates:
                    elapsed_ms = (
                        int((time.perf_counter() - _t0) * 1000)
                        if _t0 is not None else None
                    )
                    search_annotation = build_search_annotation(
                        chosen=decision,
                        candidates=candidates,
                        strategy=strategy,
                        elapsed_ms=elapsed_ms,
                        deep_capture=deep_capture,
                    )
                # Encode visits → slot-indexed normalized dist +
                # legal mask once per sample point. Shared across
                # the two viewer rows below — only the active
                # row gets the policy supervision; the data-
                # doubling non-active row gets zeros (its
                # "policy" would be the active player's
                # distribution viewed from the wrong seat).
                engine = adapter.engine_state
                if visits:
                    active_policy = np.zeros(
                        ACTION_VOCAB_SIZE, dtype=_DTYPE_POLICY,
                    )
                    # Filter visits down to root-state legal decisions
                    # before encoding. PIMC aggregates visits across K
                    # determinized worlds; some decisions (notably
                    # exile-zone casts whose legality depends on
                    # hidden info — battlefield-granted warp filters,
                    # etc.) are legal in some determinizations but not
                    # the actual root. Without filtering, those visits
                    # put policy mass on slots the action_mask
                    # correctly excludes; the trainer's
                    # masked_fill(~mask, -1e9) then makes masked CE
                    # explode (~10k per affected row). The principled
                    # fix is ISMCTS — see docs/ISMCTS_MIGRATION.md —
                    # but this filter cleans the training data
                    # immediately. Decisions that overflow an encoding
                    # cap also get dropped here; if everything's gone,
                    # fall through to no-policy-supervision for this
                    # row.
                    legal_now = set(adapter.legal_decisions())
                    counted = 0.0
                    for d, count in visits.items():
                        if d not in legal_now:
                            continue
                        try:
                            slot = encode(
                                d, engine, active,
                                version=action_encoding_version,
                            )
                        except EncodingOverflowError:
                            continue
                        active_policy[slot] += count
                        counted += count
                    if counted > 0:
                        active_policy /= counted
                        # Pass the adapter (not the raw engine) so
                        # ``_combat_phase`` resolves correctly via
                        # ``adapter.combat_phase`` and the attacker /
                        # blocker slots get enumerated when the
                        # sample point IS a combat sub-decision. Raw
                        # engine state has no ``combat_phase`` attr,
                        # so the combat branches in ``_enumerate_legal``
                        # short-circuit and combat-decision samples
                        # would carry policy mass on slots the mask
                        # said were illegal. The trainer's
                        # ``masked_fill(~mask, -1e9)`` then makes
                        # masked CE explode (~1e9 per affected row);
                        # gen0's v4 final policy_loss=613,242 was
                        # entirely this — see commit message.
                        active_mask = legal_action_mask(
                            adapter, active,
                            version=action_encoding_version,
                        )
                    else:
                        active_policy = zero_policy
                        active_mask = zero_mask
                else:
                    active_policy = zero_policy
                    active_mask = zero_mask
                for viewer in (0, 1):
                    feats = extract_features(engine, viewer)
                    if viewer == active and visits:
                        samples.append(
                            (feats, viewer, active_policy, active_mask)
                        )
                    else:
                        samples.append(
                            (feats, viewer, zero_policy, zero_mask)
                        )
            else:
                decision = strategies[active].choose(adapter, active)
        # Snapshot the pre-apply state so observe_action can be fed
        # the priority-moment state, not the post-apply state. Skip
        # the clone (and the observe_action loop) entirely when no
        # strategy implements observe_action — the clone cost is
        # nontrivial across thousands of decisions.
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
        if replay_recorder is not None:
            replay_recorder.record(adapter.engine_state, decision=decision)
            if search_annotation is not None:
                # Attach the search annotation to the PRE-apply snapshot —
                # i.e. the snapshot whose state was the search root. The
                # viewer's decision-detail panel renders the annotation
                # alongside the snapshot's state, so candidates must be
                # legal in THAT state. Attaching to the post-apply snap
                # (the one we just recorded) used to mismatch state vs.
                # candidates: e.g. a sorcery-window search at
                # PRECOMBAT_MAIN with active=priority showing 4
                # land-play candidates would land on the next snap whose
                # state has already advanced to the opp's instant-timing
                # priority window (1 candidate), making the displayed
                # annotation look like a no-op.
                #
                # Pre-apply snap index = ``len(snapshots) - 2`` (the snap
                # recorded BEFORE the just-appended post-apply one).
                # ``record`` always appends at least the initial snap
                # before this loop fires, so ``len - 2 >= 0`` for every
                # priority-action iteration that produces a search
                # annotation.
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
        decision_idx += 1
        # Update the optional shared progress sentinel so the timeout
        # handler (in the parent thread) can read decisions_completed
        # + last decision repr if this game ends up dumping. Pure
        # attribute writes — no lock, no allocation on the steady state.
        # n_legal=-1 means "not measured" — we'd have to enumerate
        # legal actions twice per decision otherwise (the search did it
        # internally, but the result isn't exposed back to us).
        if progress is not None:
            try:
                progress.record(active, decision, -1)
            except Exception:  # noqa: BLE001
                # Diagnostic capture must never break the game loop.
                pass
    winner = adapter.winner() if adapter.is_terminal() else None
    return winner, samples


def save_dataset(
    features: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
    *,
    policy_targets: Optional[np.ndarray] = None,
    action_masks: Optional[np.ndarray] = None,
) -> None:
    """Save ``(features, outcomes)`` plus optional policy
    supervision arrays to a single ``.npz`` file.

    Schema (the policy keys exist only when supplied):

    - ``features``       : ``(N, FEATURE_DIM)`` float32
    - ``outcomes``       : ``(N,)`` float32 in ``{-1, 0, +1}``
    - ``policy_targets`` : ``(N, ACTION_VOCAB_SIZE)`` float32 —
      each row is a normalized visit distribution over the legal
      action slots. Rows with no policy supervision (data-doubled
      non-active-viewer rows, baseline-strategy rows) are emitted
      as all-zeros — the trainer detects ``policy_targets.sum(-1)
      < threshold`` to skip the policy loss for those rows.
    - ``action_masks``   : ``(N, ACTION_VOCAB_SIZE)`` bool — True
      where the slot was legal at the sample point. Rows with no
      policy supervision get all-False (trainer doesn't need the
      mask there).

    Compressed — datasets are small even with policy supervision
    (zeros + bool masks compress aggressively).
    """
    n = features.shape[0]
    if outcomes.shape[0] != n:
        raise ValueError(
            f"features rows {n} != outcomes len {outcomes.shape[0]}"
        )
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"features have dim {features.shape[1]}, expected "
            f"FEATURE_DIM={FEATURE_DIM}"
        )
    if (policy_targets is None) != (action_masks is None):
        raise ValueError(
            "policy_targets and action_masks must be provided "
            "together (or both omitted)"
        )
    arrays: dict[str, np.ndarray] = {
        "features": features,
        "outcomes": outcomes,
    }
    if policy_targets is not None and action_masks is not None:
        if policy_targets.shape[0] != n or action_masks.shape[0] != n:
            raise ValueError(
                f"policy_targets / action_masks rows must equal "
                f"features rows ({n}); got "
                f"{policy_targets.shape[0]} / {action_masks.shape[0]}"
            )
        if policy_targets.shape[1] != action_masks.shape[1]:
            raise ValueError(
                f"policy_targets cols {policy_targets.shape[1]} != "
                f"action_masks cols {action_masks.shape[1]}"
            )
        arrays["policy_targets"] = policy_targets
        arrays["action_masks"] = action_masks
    np.savez_compressed(path, **arrays)


def load_dataset(
    path: Path,
    *,
    with_policy: bool = False,
) -> tuple[np.ndarray, ...]:
    """Load a single ``.npz`` saved by ``save_dataset``.

    Default returns ``(features, outcomes)`` for backward compat
    with pre-A2.3 callers. Pass ``with_policy=True`` to also pull
    policy supervision arrays — ``(features, outcomes,
    policy_targets, action_masks)``. For value-only files (no
    ``policy_targets`` key in the npz), the policy arrays come
    back as ``None``; ``load_all_datasets`` is the helper that
    handles the value-only ↔ value+policy mixing transparently.
    """
    with np.load(path) as data:
        features = data["features"]
        outcomes = data["outcomes"]
        if not with_policy:
            return features, outcomes
        if "policy_targets" in data.files:
            return (
                features,
                outcomes,
                data["policy_targets"],
                data["action_masks"],
            )
        return features, outcomes, None, None


def load_all_datasets(
    directory: Path,
    *,
    include_policy: bool = False,
    max_files: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Load and concatenate every ``*.npz`` in ``directory``.

    By default returns ``(features, outcomes)`` to preserve the
    pre-A2.3 value-only callers' contract — files that contain
    policy supervision have those arrays silently dropped. Pass
    ``include_policy=True`` to also pull policy_targets and
    action_masks back; value-only files are padded out with zero
    policy_targets / all-False action_masks so the concatenated
    arrays align row-for-row.

    The 4-tuple shape (when ``include_policy=True``) is what the
    A4-wiring trainer reads: it inspects ``policy_targets.sum(-1)
    > 0`` per row to decide which rows contribute to the policy
    loss.

    ``max_files`` (default ``None`` = load every match) caps the
    set of npz files to the ``N`` most recent by mtime. Used by
    the ``--window-recent`` training knob: as ``iterate_data/``
    accumulates over many iterations, training on the entire
    history slows linearly with size and biases the model toward
    early-iteration data. Picking the top-N most-recently-written
    files preserves the latest self-play distribution while
    capping load time. ``max_files=0`` raises (caller asked us to
    train on nothing); ``max_files`` greater than the available
    file count loads them all (no error — the caller's window is
    a maximum, not a minimum). Mtime is a fine proxy for
    generation time — the orchestrator writes each shard at
    self-play completion, so creation order matches mtime order.

    Empty arrays if the directory has no .npz files.
    """
    import sys

    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE

    if max_files is not None and max_files <= 0:
        raise ValueError(
            f"max_files must be positive or None; got {max_files}"
        )

    all_paths = list(directory.glob("*.npz"))
    if max_files is not None and all_paths:
        # Sort by mtime descending so the freshest files come
        # first, take the top ``max_files``. Then re-sort the
        # kept subset by filename so concatenation order matches
        # the all-files contract (alphabetical within the loaded
        # subset). Stable concat order matters for reproducibility
        # when the same window is re-trained.
        all_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        kept = all_paths[:max_files]
        print(
            f"loading {len(kept)} of {len(all_paths)} files "
            f"(window-recent={max_files})",
            file=sys.stderr,
        )
        paths = sorted(kept)
    else:
        paths = sorted(all_paths)
    if not paths:
        empty_features = np.zeros((0, FEATURE_DIM), dtype=_DTYPE_FEATURES)
        empty_outcomes = np.zeros((0,), dtype=_DTYPE_OUTCOMES)
        if not include_policy:
            return empty_features, empty_outcomes
        return (
            empty_features,
            empty_outcomes,
            np.zeros((0, ACTION_VOCAB_SIZE), dtype=_DTYPE_POLICY),
            np.zeros((0, ACTION_VOCAB_SIZE), dtype=np.bool_),
        )
    feature_chunks: list[np.ndarray] = []
    outcome_chunks: list[np.ndarray] = []
    policy_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    for path in paths:
        if include_policy:
            feats, outs, pols, masks = load_dataset(
                path, with_policy=True,
            )
            n_rows = feats.shape[0]
            if pols is None:
                # Old-format file — synthesize zero supervision so
                # the trainer treats every row as value-only.
                pols = np.zeros(
                    (n_rows, ACTION_VOCAB_SIZE), dtype=_DTYPE_POLICY,
                )
                masks = np.zeros(
                    (n_rows, ACTION_VOCAB_SIZE), dtype=np.bool_,
                )
            policy_chunks.append(pols)
            mask_chunks.append(masks)
        else:
            feats, outs = load_dataset(path)
        feature_chunks.append(feats)
        outcome_chunks.append(outs)
    features_all = np.concatenate(feature_chunks, axis=0)
    outcomes_all = np.concatenate(outcome_chunks, axis=0)
    if not include_policy:
        return features_all, outcomes_all
    return (
        features_all,
        outcomes_all,
        np.concatenate(policy_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0),
    )
