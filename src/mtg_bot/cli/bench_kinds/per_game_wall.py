"""Benchmark per-game wall-clock time distribution for NN-PUCT self-play.

One-off characterization tool — run a moderately-sized pool with v6
weights matching iter_001's config, time every game, print percentiles.
Goal: pick a sensible ``--per-game-timeout-sec`` for iterate.

Usage::

    PYTHONPATH=src .venv-py313/Scripts/python.exe scripts/bench_per_game_wall.py \\
        --n-games 100 --workers 24 --bootstrap-version v6

Optionally, pass ``--progress-dir`` so the run shows up in the live
training UI alongside data-gen / training / eval cards. The progress
JSON lands at ``<progress-dir>/<run-id>.json`` and follows the same
contract as the other BenchProgress emitters — start_pairing /
record_game / finish_pairing / finalize.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mtg_bot.decks import decks_from_json
from mtg_bot.training import BenchProgress, NNPimcConfig, Registry
from mtg_bot.training.data_gen import (
    _generate_one_game_data,
    _sample_decks_for_game,
)
from mtg_bot.training.trainer import auto_device


# Shared via initializer/global so each worker only loads decks once.
_DECK_POOL = None
_WEIGHTS_PATH = None
_HIDDEN_SIZES = None
_DEVICE = None


def _init_worker(
    deck_pool_path: str, weights_path: str, hidden_sizes_tuple, device: str,
):
    global _DECK_POOL, _WEIGHTS_PATH, _HIDDEN_SIZES, _DEVICE
    _DECK_POOL = decks_from_json(Path(deck_pool_path))
    _WEIGHTS_PATH = weights_path
    _HIDDEN_SIZES = hidden_sizes_tuple
    _DEVICE = device


def _run_one(args):
    """Time a single NN-PUCT v6-vs-v6 game; return (game_idx, wall_sec, n_samples, status)."""
    game_idx, base_seed = args
    cfg = NNPimcConfig(
        weights_path=_WEIGHTS_PATH,
        samples=4,
        inner_nodes=200,
        batch_size=32,
        with_policy=True,
        puct=True,
        hidden_sizes=_HIDDEN_SIZES,
        device=_DEVICE,
    )
    start = time.perf_counter()
    try:
        chunk = _generate_one_game_data(
            game_idx=game_idx,
            p0_factory=cfg.build,
            p1_factory=cfg.build,
            base_seed=base_seed,
            decision_cap=4000,
            sample_every=1,
            deck_pool=_DECK_POOL,
            enable_mulligans=True,
            per_game_timeout_sec=None,  # let everything run to completion
        )
        wall = time.perf_counter() - start
        return (game_idx, wall, int(chunk[0].shape[0]), "ok")
    except Exception as exc:  # noqa: BLE001
        wall = time.perf_counter() - start
        return (game_idx, wall, 0, f"err:{type(exc).__name__}")


def main(argv=None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=8000000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--bootstrap-version", default="v6",
        help="Registry version to use for both seats (default v6).",
    )
    parser.add_argument(
        "--registry-root", type=Path, default=Path("trained_weights"),
    )
    parser.add_argument(
        "--deck-pool", type=Path,
        default=Path("data/17lands/eoe_decks.json.gz"),
    )
    parser.add_argument(
        "--progress-dir", type=str, default=None,
        help=(
            "Optional directory for live progress JSON so the training "
            "UI can render this bench alongside data-gen / training / "
            "eval cards. Writes ``<progress-dir>/<run-id>.json``. When "
            "unset, the bench runs silently (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help=(
            "Run id used as the progress filename stem and the "
            "``run_id`` field in the progress JSON. Default: "
            "auto-generated like ``bench_<timestamp>``."
        ),
    )
    parser.add_argument(
        "--label", type=str, default="mac-standalone-bench",
        help=(
            "Human-readable label that lands in the progress JSON's "
            "``notes`` field so the run is identifiable in the UI."
        ),
    )
    args = parser.parse_args(argv)

    registry = Registry(args.registry_root)
    weights_path = str(registry.weights_path(args.bootstrap_version))
    meta = registry.load_metadata(args.bootstrap_version)
    hidden_sizes = (
        tuple(meta.hidden_sizes)
        if getattr(meta, "hidden_sizes", None) else None
    )
    device = auto_device()

    # Stage the BenchProgress tracker if --progress-dir was set. This
    # is purely additive — stdout output below is unchanged.
    progress: Optional[BenchProgress] = None
    if args.progress_dir:
        run_id = args.run_id or (
            "bench_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        progress_dir = Path(args.progress_dir)
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_path = progress_dir / f"{run_id}.json"
        progress = BenchProgress(
            progress_path=progress_path,
            run_id=run_id,
            weights_path=weights_path,
            n_games_per_pairing=args.n_games,
            budgets=(),  # single-pairing self-play, no AB budget grid
            include_belief=False,
            belief_m_candidates=None,
            nn_device=device,
            command=" ".join(sys.argv),
            notes=args.label,
        )
        progress.begin()
        # The legacy stdout stays unchanged; we add a tiny stderr line
        # so the operator can see where the progress JSON landed.
        print(
            f"# progress: {progress_path}", file=sys.stderr, flush=True,
        )

    print(
        f"# Benching {args.n_games} games at workers={args.workers}, "
        f"version={args.bootstrap_version}, hidden_sizes={hidden_sizes}, "
        f"base_seed={args.base_seed}"
    )

    args_list = [
        (i, args.base_seed) for i in range(args.n_games)
    ]
    overall_start = time.perf_counter()
    if progress is not None:
        # Single self-play pairing: challenger and defender are the
        # same version. The --label flows through to the UI via notes.
        pairing_name = f"{args.bootstrap_version}-self-play"
        progress.start_pairing(
            challenger=pairing_name,
            defender=pairing_name,
            n_games=args.n_games,
        )
    try:
        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(
                str(args.deck_pool), weights_path, hidden_sizes, device,
            ),
        ) as pool:
            results = []
            for r in pool.imap_unordered(_run_one, args_list, chunksize=1):
                results.append(r)
                # Stream into BenchProgress so the UI ticks live. This
                # bench has no notion of "challenger vs defender" —
                # both seats are the same version — so we treat any
                # 'ok' game as a draw (winner=None) and crashed games
                # as crashes. That keeps n_games_done accurate without
                # inventing a win column the bench can't actually
                # report.
                if progress is not None:
                    crashed = r[3] != "ok"
                    progress.record_game(
                        winner=None, crashed=crashed,
                    )
                walls = [x[1] for x in results]
                walls.sort()
                n = len(walls)
                if n % 10 == 0:
                    done_msg = f"# {n}/{args.n_games}"
                    if n >= 5:
                        done_msg += (
                            f"  median={walls[n // 2]:.1f}s"
                            f"  max={walls[-1]:.1f}s"
                        )
                    print(done_msg, flush=True)
    except Exception:
        if progress is not None:
            progress.finish_pairing(status="failed")
            progress.finalize(status="failed")
        raise
    overall = time.perf_counter() - overall_start

    walls = sorted(x[1] for x in results)
    n = len(walls)
    def pct(p):
        # walls[idx] where idx = ceil(n * p / 100) - 1, clamped
        idx = max(0, min(n - 1, int(round(n * p / 100)) - 1))
        return walls[idx]

    mean_per_game_wall = sum(walls) / n if n > 0 else 0.0
    # Throughput numbers — pinned to the same definitions used by
    # the BenchProgress pairings entry so the stdout block and the
    # JSON never disagree:
    #   sec/game/worker = mean per-game wall (seconds)
    #   games/hour      = 3600 / mean_per_game_wall * n_workers
    sec_per_game_per_worker = mean_per_game_wall
    games_per_hour = (
        3600.0 / mean_per_game_wall * args.workers
        if mean_per_game_wall > 0 else 0.0
    )

    if progress is not None:
        progress.set_throughput_stats(
            games_per_hour=games_per_hour,
            sec_per_game_per_worker=sec_per_game_per_worker,
        )
        progress.finish_pairing(status="completed")

    print(f"\n# Done in {overall/60:.1f} min wall (overall pool time)")
    print(f"# Per-game wall (n={n}):")
    print(f"#   min  = {walls[0]:.1f}s")
    print(f"#   p50  = {pct(50):.1f}s")
    print(f"#   p75  = {pct(75):.1f}s")
    print(f"#   p90  = {pct(90):.1f}s")
    print(f"#   p95  = {pct(95):.1f}s")
    print(f"#   p99  = {pct(99):.1f}s")
    print(f"#   max  = {walls[-1]:.1f}s")
    print(f"#   mean = {mean_per_game_wall:.1f}s")
    print(f"# Throughput (workers={args.workers}):")
    print(f"#   sec/game/worker = {sec_per_game_per_worker:.1f}s")
    print(f"#   games/hour      = {games_per_hour:.1f}")
    crashed = [r for r in results if r[3] != "ok"]
    if crashed:
        print(f"# crashed: {len(crashed)}")
        for r in crashed[:5]:
            print(f"#   game_idx={r[0]} wall={r[1]:.1f}s status={r[3]}")
    # Timeout-cost breakdown.
    print(f"\n# Game count over candidate timeouts:")
    for budget in (60, 120, 180, 240, 300, 450, 600, 900, 1200):
        n_over = sum(1 for w in walls if w > budget)
        print(f"#   timeout={budget:>4}s  -> {n_over:>3}/{n} ({n_over/n*100:.1f}%) lost")

    if progress is not None:
        progress.finalize(status="completed")


if __name__ == "__main__":
    main()
