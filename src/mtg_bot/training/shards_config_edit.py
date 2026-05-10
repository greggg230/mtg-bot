"""Shared helpers for safely editing the on-disk ``shards_config.json``.

Used by:

- ``mtg_bot serve`` — the ``POST /api/shards/<host>/workers`` endpoint
  exposed for the training UI.
- The iterate per-iter refresh path (planned consumers) — read the
  current state of the file at the top of each iter.

All file mutations go through :func:`update_shard_n_workers`, which
holds an in-process lock and uses ``tempfile`` + ``os.replace`` for
atomic write-through. The lock is module-level so concurrent Flask
threads (multiple UI clients clicking Apply at once) serialize
through it.

**Phase 1 scope**: between-iter only. The mutation lands on disk
immediately, but iterate's currently-running pool is unaffected — it
re-reads the file at the start of the next iter cycle and picks up
the change there. Mid-iter scaling (Phase 2) requires switching the
local self-play pool from ``multiprocessing.Pool`` to a
queue + spawn/kill worker pattern; deferred.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

# Bounds on n_workers. The lower bound is 1 (a 0-worker shard is
# pointless); the upper bound is a sanity guard against fat-fingered
# values from the UI — no current machine has 64 cores worth of
# usable workers, but the bound is generous enough not to clip a
# legitimate config.
MIN_N_WORKERS = 1
MAX_N_WORKERS = 64

# Module-level lock so concurrent edits from multiple Flask request
# threads serialize. The lock guards both the read-modify-write and
# the disk swap so two simultaneous POSTs can't lose one another's
# change. Exposed so callers (e.g. tests) can introspect.
_EDIT_LOCK = threading.Lock()


class ShardsConfigEditError(Exception):
    """Raised for shards-config edit failures the caller should
    surface as a user-visible error (HTTP 4xx, etc.)."""


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON to ``path`` via a tempfile + os.replace.

    Ensures readers (the iterate loop, other UIs) never see a
    partially-written file. The tempfile lands in the same directory
    so ``os.replace`` is a same-filesystem rename.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the tempfile on failure so we don't litter the dir.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _validate_n_workers(value: Any) -> int:
    """Coerce + range-check an n_workers value from a JSON request
    body. Raises :class:`ShardsConfigEditError` for anything outside
    ``[MIN_N_WORKERS, MAX_N_WORKERS]`` or non-integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        # bool is an int subclass; explicit reject.
        raise ShardsConfigEditError(
            f"n_workers must be an integer, got {type(value).__name__}",
        )
    if value < MIN_N_WORKERS or value > MAX_N_WORKERS:
        raise ShardsConfigEditError(
            f"n_workers must be in [{MIN_N_WORKERS}, {MAX_N_WORKERS}], "
            f"got {value}",
        )
    return value


def update_shard_n_workers(
    config_path: Path, host: str, n_workers: Any,
) -> dict[str, Any]:
    """Update the ``n_workers`` field for the shard with ``host==host``
    in ``config_path``.

    Returns the mutated config dict (the same shape that was on disk).
    Raises :class:`ShardsConfigEditError` if the host is unknown or
    the value is invalid; caller maps to the appropriate HTTP status.

    Atomic: read + edit + write happens under :data:`_EDIT_LOCK` so
    concurrent Flask threads serialize cleanly.
    """
    validated = _validate_n_workers(n_workers)
    with _EDIT_LOCK:
        if not config_path.is_file():
            raise ShardsConfigEditError(
                f"shards-config not found: {config_path}",
            )
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ShardsConfigEditError(
                f"shards-config {config_path} is not valid JSON: {exc}",
            ) from exc
        if not isinstance(raw, dict) or "shards" not in raw:
            raise ShardsConfigEditError(
                f"shards-config {config_path} missing top-level 'shards'",
            )
        shards = raw["shards"]
        if not isinstance(shards, list):
            raise ShardsConfigEditError(
                f"shards-config {config_path}: 'shards' must be a list",
            )
        target = None
        for entry in shards:
            if isinstance(entry, dict) and entry.get("host") == host:
                target = entry
                break
        if target is None:
            raise ShardsConfigEditError(
                f"unknown host {host!r}",
            )
        target["n_workers"] = validated
        _atomic_write_json(config_path, raw)
        return raw


def read_shards_config_raw(config_path: Path) -> dict[str, Any]:
    """Read ``config_path`` as raw JSON (no parsing into ShardConfig
    dataclasses). Used by iterate's per-iter refresh + tests that
    want to inspect the freshly-written contents.
    """
    return json.loads(config_path.read_text(encoding="utf-8"))


def summarize_shards_for_log(raw: dict[str, Any]) -> str:
    """Produce a one-line summary of shard host -> n_workers mappings,
    suitable for logging from iterate's per-iter refresh.

    Format: ``host1=N, host2=null (auto), host3=M``. ``null (auto)``
    marks a shard with no explicit n_workers (the runner falls back
    to ``cpu_count() - 2``).
    """
    shards = raw.get("shards", []) if isinstance(raw, dict) else []
    parts = []
    for entry in shards:
        if not isinstance(entry, dict):
            continue
        host = entry.get("host", "?")
        n = entry.get("n_workers")
        parts.append(f"{host}={n}" if n is not None else f"{host}=null (auto)")
    return ", ".join(parts)
