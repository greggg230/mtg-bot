"""``mtg_bot`` CLI dispatcher.

One entrypoint, one place to discover what the project can do. Run
``mtg_bot --help`` to list subcommands.

Subcommands live in ``mtg_bot.cli.<name>``; each exposes a ``main()``
that parses its own ``argv``. The dispatcher here just routes the
first positional to the right module and forwards the rest.

Post v2-cutover the ``shard`` subcommand is gone — distributed work
goes through ``mtg_bot.worker_daemon`` (long-lived REST service)
instead of one-shot SSH-launched shard processes.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


_SUBCOMMANDS = {
    "iterate": "mtg_bot.cli.iterate",
    "iterate-distributed": "mtg_bot.cli.iterate_distributed",
    "train": "mtg_bot.cli.train",
    "calibrate-value": "mtg_bot.cli.calibrate_value",
    "card-vocab": "mtg_bot.cli.card_vocab",
    "serve": "mtg_bot.cli.serve",
    "capture": "mtg_bot.cli.capture",
    "eval": "mtg_bot.cli.eval",
    "bench": "mtg_bot.cli.bench",
    "db": "mtg_bot.cli.db",
    "ladder": "mtg_bot.cli.ladder",
}

_DESCRIPTIONS = {
    "iterate": "Run the AlphaZero training loop (self-play -> train -> eval).",
    "iterate-distributed": "Unified distributed-iter launcher (Phase A: local-only); single TOML config, runs.db pre-registration, lifecycle-spanned iterate child.",
    "train": "Train the value+policy network from saved self-play data.",
    "calibrate-value": "Re-fit a registered version's value-head temperature on a held-out split (metadata-only; weights untouched).",
    "card-vocab": "Inspect / export / stamp the card vocabulary structured checkpoints were trained on.",
    "serve": "Bundle/queue HTTP server (port 5099) for the distributed flow.",
    "capture": "Capture annotated NN-driven self-play replays for replay-viewer review.",
    "eval": "Run a head-to-head eval between two registered NN versions.",
    "bench": "Run one of the perf benchmarks (clone, per-game wall, NN-eval-server, ...).",
    "db": "Run database admin tooling (backup, dump, query).",
    "ladder": "Local fixed-reference strength ladder (random / heuristic / NN versions) with Elo.",
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
    rc = mod.main(args.rest)
    # Propagate the subcommand's exit code. Dropping it made
    # ``mtg_bot iterate-distributed`` exit 0 even when its ``_launch``
    # returned the iterate child's non-zero rc -- so on 2026-09-23 the
    # wrapper script relaunched the failing cycle four more times
    # (2026-09-23 postmortem). Subcommands that return None (most of
    # them) are unaffected.
    if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
