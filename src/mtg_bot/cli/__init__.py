"""``mtg_bot`` CLI dispatcher.

One entrypoint, one place to discover what the project can do. Run
``mtg_bot --help`` to list subcommands.

Subcommands live in ``mtg_bot.cli.<name>``; each exposes a ``main()``
that parses its own ``argv``. The dispatcher here just routes the
first positional to the right module and forwards the rest.

The four production subcommands (``iterate``, ``train``, ``shard``,
``serve``) also ship as thin shims under ``scripts/`` so existing
SSH invocations and the bundle/queue server keep working without
client-side changes; see ``scripts/run_self_play_shard.py`` for an
example shim.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


_SUBCOMMANDS = {
    "iterate": "mtg_bot.cli.iterate",
    "train": "mtg_bot.cli.train",
    "shard": "mtg_bot.cli.shard",
    "serve": "mtg_bot.cli.serve",
    "capture": "mtg_bot.cli.capture",
    "eval": "mtg_bot.cli.eval",
    "bench": "mtg_bot.cli.bench",
    "db": "mtg_bot.cli.db",
}

_DESCRIPTIONS = {
    "iterate": "Run the AlphaZero training loop (self-play -> train -> eval).",
    "train": "Train the value+policy network from saved self-play data.",
    "shard": "Run one self-play or eval shard (SSH-invoked from master).",
    "serve": "Bundle/queue HTTP server (port 5099) for the distributed flow.",
    "capture": "Capture annotated NN-driven self-play replays for replay-viewer review.",
    "eval": "Run a head-to-head eval between two registered NN versions.",
    "bench": "Run one of the perf benchmarks (clone, per-game wall, NN-eval-server, ...).",
    "db": "Run database admin tooling (backup, dump, query).",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Top-level dispatch. ``argv`` defaults to ``sys.argv[1:]``."""
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    parser = argparse.ArgumentParser(
        prog="mtg_bot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # We dispatch manually to subcommand modules so each one stays
        # a self-contained ``main(argv)`` — easier to import-and-test
        # in isolation than nested argparse subparsers.
        add_help=True,
    )
    parser.add_argument(
        "subcommand", choices=sorted(_SUBCOMMANDS.keys()),
        help="Which CLI subcommand to run. See per-subcommand --help.",
    )
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="Forwarded to the subcommand verbatim.",
    )

    if not argv or argv[0] in {"-h", "--help"}:
        # Print our own usage AND list the subcommand descriptions —
        # otherwise the bare ``mtg_bot`` command is opaque about what
        # exists.
        parser.print_help()
        print("\nSubcommands:")
        for name in sorted(_SUBCOMMANDS):
            print(f"  {name:<10} {_DESCRIPTIONS[name]}")
        return

    args = parser.parse_args(argv)
    import importlib
    mod = importlib.import_module(_SUBCOMMANDS[args.subcommand])
    mod.main(args.rest)


if __name__ == "__main__":
    main()
