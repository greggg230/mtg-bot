"""Shared argparse parents + helpers for the ``mtg_bot`` CLI.

Every subcommand consumes these via ``parents=[...]`` so cross-cutting
concerns — depth (samples + nodes), the queue/bundle URLs, the deck
pool, the progress dir, the per-game timeout — are defined exactly
once. Defaults match the most recent iteration's self-play config so
eval-vs-self-play depth never drifts again (see ``CLAUDE.md`` "Eval
depth must match self-play depth").

If you change a default here, every subcommand picks it up; that is
the whole point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Tailnet hostname of the master that runs the queue + bundle server.
# Hard-coded here (not auto-detected) so a remote shard pulling the
# bundle gets the right master URL out of the box. If the master
# changes, edit this constant.
DEFAULT_MASTER_TAILNET = "desktop-j05412i"

DEFAULT_QUEUE_URL = f"http://{DEFAULT_MASTER_TAILNET}:5099"
DEFAULT_BUNDLE_URL = f"{DEFAULT_QUEUE_URL}/repo.bundle"

# iter_005 self-play depth. Eval/capture defaults track this value by
# convention (CLAUDE.md "Eval depth"). Update both here AND in
# ``iterate_alphazero``'s self-play call when the convention shifts.
DEFAULT_NODES = 200
DEFAULT_SAMPLES = 4

DEFAULT_DECK_POOL = Path("data/17lands/eoe_decks.json.gz")
DEFAULT_PROGRESS_DIR = Path("training_progress")
DEFAULT_SHARDS_CONFIG = Path("shards_config.json")
# 450s was the original budget; 2026-05-08 v16-vs-v13 eval analysis
# showed completed-game p99 wall = 390s and max = 449s. With the
# observed master shard slowdown over long runs (+14% sec/decision
# +30% wall/game by hour 8, separate todo), the right tail of slow
# games regularly clipped the 450 cap. Bumping to 600 leaves margin
# for the right tail without throughput cost (theoretical only — most
# games finish well under 400s). Bias-check from the same investigation:
# v_X win rate is flat across decision-count bins, so longer games are
# not systematically biased toward either side, making this safe.
DEFAULT_PER_GAME_TIMEOUT_SEC = 600.0


def depth_parser() -> argparse.ArgumentParser:
    """``--samples`` + ``--nodes`` (the search-depth knobs).

    Eval, bench, capture, iterate all consume this. Defaults are pinned
    to the most-recent self-play config so a freshly-templated eval
    doesn't accidentally measure a different network.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=(
            f"PIMC samples K (default {DEFAULT_SAMPLES}). Match "
            "self-play depth for eval (CLAUDE.md 'Eval depth')."
        ),
    )
    p.add_argument(
        "--nodes", type=int, default=DEFAULT_NODES,
        help=(
            f"MCTS max_nodes per sub-search (default {DEFAULT_NODES}; "
            "the iter_005 self-play depth)."
        ),
    )
    return p


def cluster_parser() -> argparse.ArgumentParser:
    """``--shards-config`` + ``--queue-url`` + ``--bundle-url``.

    Distributed self-play, distributed eval, the bundle/queue server
    itself all need these. Defaults bake in the current master's
    tailnet hostname so a fresh-clone invocation works without flags.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--shards-config", type=Path, default=DEFAULT_SHARDS_CONFIG,
        help=f"ShardsConfig JSON path (default {DEFAULT_SHARDS_CONFIG}).",
    )
    p.add_argument(
        "--queue-url", default=DEFAULT_QUEUE_URL,
        help=(
            "Master's queue endpoint base URL (default "
            f"{DEFAULT_QUEUE_URL}). Remotes pull jobs from here."
        ),
    )
    p.add_argument(
        "--bundle-url", default=DEFAULT_BUNDLE_URL,
        help=(
            "Master's git bundle URL (default "
            f"{DEFAULT_BUNDLE_URL}). Remotes refresh their checkout "
            "from this on each shard."
        ),
    )
    return p


def runtime_parser() -> argparse.ArgumentParser:
    """``--deck-pool`` + ``--progress-dir`` + ``--per-game-timeout-sec``.

    Anything that runs games needs these.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--deck-pool", type=Path, default=DEFAULT_DECK_POOL,
        help=f"17lands deck pool path (default {DEFAULT_DECK_POOL}).",
    )
    p.add_argument(
        "--progress-dir", type=Path, default=DEFAULT_PROGRESS_DIR,
        help=(
            "Directory the training UI watches for progress JSON "
            f"(default {DEFAULT_PROGRESS_DIR})."
        ),
    )
    p.add_argument(
        "--per-game-timeout-sec", type=float,
        default=DEFAULT_PER_GAME_TIMEOUT_SEC,
        help=(
            "Wall-clock budget per game before the worker abandons it "
            f"(default {DEFAULT_PER_GAME_TIMEOUT_SEC}s)."
        ),
    )
    return p


def load_shards_config(path: Path):
    """Re-export of ``mtg_bot.training.distributed.load_shards_config``.

    Subcommands import this from ``_common`` to avoid each one carrying
    the ``from mtg_bot.training.distributed import ...`` boilerplate.
    """
    from mtg_bot.training.distributed import load_shards_config as _impl
    return _impl(path)


__all__ = [
    "DEFAULT_BUNDLE_URL",
    "DEFAULT_DECK_POOL",
    "DEFAULT_MASTER_TAILNET",
    "DEFAULT_NODES",
    "DEFAULT_PER_GAME_TIMEOUT_SEC",
    "DEFAULT_PROGRESS_DIR",
    "DEFAULT_QUEUE_URL",
    "DEFAULT_SAMPLES",
    "DEFAULT_SHARDS_CONFIG",
    "cluster_parser",
    "depth_parser",
    "load_shards_config",
    "runtime_parser",
]
