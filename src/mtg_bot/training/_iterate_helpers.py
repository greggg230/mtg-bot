"""Small utilities used only by ``scripts/iterate_alphazero.py``.

Lives under ``mtg_bot.training`` so it's importable from tests
without the sys.path-manipulating script preamble. Today this
module hosts:

- ``find_foreign_npzs`` — the foreign-npz guard that prevents the
  trainer from co-globbing stale debug npzs into the iterate loop's
  training mix.
- ``refresh_shards_config_for_iter`` — re-read shards_config.json
  from disk at the top of each iter so dynamic worker-scale
  changes made via the training UI between iters take effect on
  the upcoming pool init. Phase 1: between-iter only; mid-iter
  scaling is Phase 2 (deferred — would require switching the
  local self-play pool from multiprocessing.Pool to a
  queue + spawn/kill worker pattern).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from mtg_bot.training.distributed import ShardsConfig, load_shards_config
from mtg_bot.training.shards_config_edit import summarize_shards_for_log


def find_foreign_npzs(data_dir: Path) -> list[str]:
    """Return ``.npz`` filenames under ``data_dir`` that don't match
    the ``iter_*.npz`` pattern. iterate writes its self-play files
    as ``iter_NNN_<bootstrap>_<utc>.npz``, and ``train_nn.py`` globs
    every ``.npz`` in the dir it's pointed at. Anything else is a
    landmine: foreign npzs may have been written by older code paths
    where the policy/action-mask invariants didn't hold, and even a
    few hundred mask-mismatched rows are enough to blow CE through
    the roof and corrupt the next-iteration model.

    Returns sorted basenames; empty list when the dir is clean. The
    caller decides whether to fail (recommended) or just warn.
    """
    if not data_dir.is_dir():
        return []
    return sorted(
        p.name for p in data_dir.glob("*.npz")
        if not p.name.startswith("iter_")
    )


def refresh_shards_config_for_iter(
    *,
    config_path: Optional[Path],
    previous: Optional[ShardsConfig],
    iter_label: str,
    print_fn=print,
) -> Optional[ShardsConfig]:
    """Re-read ``config_path`` from disk and return a fresh
    :class:`ShardsConfig`. Logs a one-line summary so the operator
    can confirm a UI-driven worker-scale change took effect at the
    expected iter boundary.

    ``previous`` is the config from the last iter (or the initial
    load); used to detect *shard ENTRIES* coming or going (vs
    n_workers changes within stable entries). Mid-run shard set
    changes aren't supported in Phase 1 — log a warning if we see
    one but still honor the new config (the operator presumably
    knows what they're doing).

    Returns ``None`` when ``config_path`` is ``None`` (single-host
    iterate; no distributed pipeline). Returns the freshly-loaded
    config otherwise.

    On read errors (file missing, JSON corruption, validation
    failure) the function logs the error and returns ``previous``
    unchanged — better to keep running with the last-good config
    than to crash mid-iter on a transient disk hiccup.
    """
    if config_path is None:
        return None
    try:
        fresh = load_shards_config(config_path)
    except (OSError, ValueError) as exc:
        print_fn(
            f"[iterate] WARNING: failed to refresh shards_config "
            f"for {iter_label}: {exc!r}; "
            f"keeping previous config.",
            file=sys.stderr,
        )
        return previous

    # One-line summary so the user can grep the iter log to confirm
    # their UI bump took effect at the expected boundary.
    summary_parts = [
        f"{s.host}={s.n_workers}" if s.n_workers is not None
        else f"{s.host}=null (auto)"
        for s in fresh.shards
    ]
    print_fn(
        f"[iterate] shards_config refreshed for {iter_label}: "
        + ", ".join(summary_parts)
    )

    if previous is not None:
        prev_hosts = {s.host for s in previous.shards}
        new_hosts = {s.host for s in fresh.shards}
        added = new_hosts - prev_hosts
        removed = prev_hosts - new_hosts
        if added or removed:
            print_fn(
                f"[iterate] WARNING: shards_config entry set changed "
                f"between iters (added={sorted(added)}, "
                f"removed={sorted(removed)}). Honoring new set, but "
                f"shard *entries* are not designed to change mid-run "
                f"— only n_workers per shard. Subsequent runs may see "
                f"orphaned per-shard progress JSONs from the dropped "
                f"hosts; clean them out manually if needed.",
                file=sys.stderr,
            )

    return fresh


__all__ = [
    "find_foreign_npzs",
    "refresh_shards_config_for_iter",
    "summarize_shards_for_log",
]
