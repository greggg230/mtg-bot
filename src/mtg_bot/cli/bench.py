"""``mtg_bot bench <kind>`` — perf benchmarks.

Sub-subcommands route into ``mtg_bot.cli.bench_kinds.<kind>``:

- ``per-game-wall`` — per-game wall-clock distribution for NN-PUCT
  self-play (timeout-budget characterization)
- ``clone`` — ``GameState.clone()`` vs ``copy.deepcopy(state)``
  (perf-history hook)
- ``clone-zones`` — per-zone breakdown of clone() cost
- ``nn-eval-server`` — NNEvalServer vs in-process NN throughput

Each sub-subcommand exposes a ``main(argv)`` consumed by the
dispatcher here. Adding a new bench kind: drop a module under
``mtg_bot.cli.bench_kinds/`` and register it in ``_KINDS`` below.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


_KINDS = {
    "per-game-wall": "mtg_bot.cli.bench_kinds.per_game_wall",
    "clone": "mtg_bot.cli.bench_kinds.clone",
    "clone-zones": "mtg_bot.cli.bench_kinds.clone_zones",
    "nn-eval-server": "mtg_bot.cli.bench_kinds.nn_eval_server",
}

_DESCRIPTIONS = {
    "per-game-wall": (
        "Per-game wall distribution for NN-PUCT self-play "
        "(timeout-budget characterization)."
    ),
    "clone": "GameState.clone() vs copy.deepcopy() (perf-history hook).",
    "clone-zones": "Per-zone breakdown of clone() cost.",
    "nn-eval-server": "NNEvalServer vs in-process NN throughput.",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch ``mtg_bot bench <kind> <args>`` to the right module."""
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    parser = argparse.ArgumentParser(
        prog="mtg_bot bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "kind", choices=sorted(_KINDS.keys()),
        help="Which bench to run. See per-bench --help.",
    )
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="Forwarded to the bench module verbatim.",
    )

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        print("\nBench kinds:")
        for k in sorted(_KINDS):
            print(f"  {k:<16} {_DESCRIPTIONS[k]}")
        return

    args = parser.parse_args(argv)
    import importlib
    mod = importlib.import_module(_KINDS[args.kind])
    mod.main(args.rest)


if __name__ == "__main__":
    main()
