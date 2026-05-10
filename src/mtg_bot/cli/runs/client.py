"""Cross-process / in-process client for the run database.

Subcommands call this module's functions to register a run, push
progress updates, append events, and finish the run. The client
auto-detects which transport to use:

- ``MTG_BOT_RUN_DB_URL`` set (e.g. ``http://desktop-j05412i:5099``)
  -> POST to the master's REST endpoints. Used by SSH-launched remote
  shards: they don't have direct file access to the master's runs.db.
- ``MTG_BOT_RUN_DB_PATH`` set (e.g.
  ``training_progress/runs.db``) -> open the SQLite file directly.
  Used by the master's own subcommands (``iterate``, ``train``,
  standalone ``eval``, ...).
- Neither set -> all calls are no-ops. Phase 1 invariant: callers
  keep their existing JSON writes; the DB layer is purely additive,
  so a missing config means "DB not deployed yet, just keep doing
  what you've always done."

Resolution happens **on every call**, not at import. That way a test
can set the env var partway through a fixture, or a subcommand can
override it via ``configure(...)`` after argparse runs.

All client functions swallow exceptions and emit a single warning
line to stderr instead of crashing the caller. The Phase 1 invariant
is "DB is shadow; never break the existing path." Once we trust the
DB enough to swap the UI (Phase 2), we can revisit and let
exceptions propagate.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from mtg_bot.cli.runs import db as _db
from mtg_bot.cli.runs.models import (
    Artifact, Event, ProgressUpdate, Run,
)


_ENV_DB_PATH = "MTG_BOT_RUN_DB_PATH"
_ENV_DB_URL = "MTG_BOT_RUN_DB_URL"


def _resolved_db_path() -> Optional[Path]:
    """Return the path the direct-SQLite transport should use, or
    ``None`` if not configured. Reads the env var fresh every call so
    test fixtures + late ``configure(...)`` writes propagate."""
    raw = os.environ.get(_ENV_DB_PATH)
    if not raw:
        return None
    return Path(raw)


def _resolved_db_url() -> Optional[str]:
    """Master REST base URL (e.g. ``http://desktop-j05412i:5099``), or
    ``None``. Trailing slash stripped so callers can append paths
    naively."""
    raw = os.environ.get(_ENV_DB_URL)
    if not raw:
        return None
    return raw.rstrip("/")


def configure(
    *,
    db_path: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> None:
    """Override the env-var-derived defaults at runtime. Either or
    both. Pass ``None`` to clear an override (falls back to env var
    or no-op)."""
    if db_path is None:
        os.environ.pop(_ENV_DB_PATH, None)
    else:
        os.environ[_ENV_DB_PATH] = str(db_path)
    if db_url is None:
        os.environ.pop(_ENV_DB_URL, None)
    else:
        os.environ[_ENV_DB_URL] = str(db_url)


def is_configured() -> bool:
    """``True`` if either transport is configured. Subcommands may
    skip the start_run / push_progress overhead entirely when this
    returns ``False``, but the client does the right thing
    (no-ops) anyway."""
    return _resolved_db_url() is not None or _resolved_db_path() is not None


# ---- transport helpers -----------------------------------------------------


def _post_json(
    url: str, payload: dict[str, Any], *, timeout: float = 5.0,
) -> dict[str, Any]:
    """POST ``payload`` as JSON to ``url`` and return the parsed
    response. On any HTTP / network error, raises so the caller's
    swallow-and-warn wrapper can log it."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8") or "{}"
    return json.loads(raw)


def _warn(action: str, exc: BaseException) -> None:
    """One-line stderr warning. Phase 1: never break the caller. The
    JSON write that pairs with this DB call already succeeded (or
    will), so the progress UI keeps working and the DB row is the
    only thing missed."""
    print(
        f"[runs.client] WARNING: {action} failed: "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
    )


def _direct(fn):
    """Run ``fn(conn)`` against a freshly-opened DB connection,
    serialized through the module-level write lock so concurrent
    pushes from different threads don't race at COMMIT.
    """
    db_path = _resolved_db_path()
    assert db_path is not None  # caller checked
    with _db.WRITE_LOCK:
        conn = _db.connect(db_path)
        try:
            return fn(conn)
        finally:
            conn.close()


# ---- public API ------------------------------------------------------------


def start_run(
    run_id: str,
    *,
    kind: str,
    config_json: str,
    git_commit: str,
    parent_run_id: Optional[str] = None,
    label: Optional[str] = None,
    started_at: Optional[str] = None,
) -> str:
    """Register a new run. Returns the (possibly server-issued, but
    in practice always-equal-to-input) run_id. Idempotent for
    re-registration of the same run_id is **not** supported — the
    underlying INSERT would fail; callers should not call start_run
    twice for the same id."""
    started = started_at or _db.now_iso()
    run = Run(
        run_id=run_id,
        kind=kind,
        status="running",
        started_at=started,
        git_commit=git_commit,
        config_json=config_json,
        parent_run_id=parent_run_id,
        label=label,
    )
    url = _resolved_db_url()
    if url is not None:
        try:
            _post_json(
                f"{url}/run/start",
                {
                    "run_id": run.run_id,
                    "kind": run.kind,
                    "config_json": run.config_json,
                    "git_commit": run.git_commit,
                    "parent_run_id": run.parent_run_id,
                    "label": run.label,
                    "started_at": run.started_at,
                },
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            _warn(f"start_run({run_id})", exc)
        return run_id
    if _resolved_db_path() is not None:
        try:
            _direct(lambda c: _db.insert_run(c, run))
        except Exception as exc:  # noqa: BLE001
            _warn(f"start_run({run_id})", exc)
    return run_id


def push_progress(
    run_id: str,
    *,
    shard_label: Optional[str] = None,
    n_games_done: Optional[int] = None,
    n_games_total: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
    crashed: Optional[int] = None,
    timed_out: Optional[int] = None,
    wins_new: Optional[int] = None,
    wins_old: Optional[int] = None,
    draws: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Upsert a progress row keyed by ``(run_id, shard_label)``.
    Same shard_label = same row, repeatable. Aggregate row uses
    ``shard_label=None``."""
    extra_json = json.dumps(extra, sort_keys=True) if extra else None
    update = ProgressUpdate(
        run_id=run_id,
        shard_label=shard_label,
        n_games_done=n_games_done,
        n_games_total=n_games_total,
        elapsed_seconds=elapsed_seconds,
        crashed=crashed,
        timed_out=timed_out,
        wins_new=wins_new,
        wins_old=wins_old,
        draws=draws,
        extra_json=extra_json,
    )
    url = _resolved_db_url()
    if url is not None:
        try:
            _post_json(
                f"{url}/run/{run_id}/progress",
                {
                    "shard_label": update.shard_label,
                    "n_games_done": update.n_games_done,
                    "n_games_total": update.n_games_total,
                    "elapsed_seconds": update.elapsed_seconds,
                    "crashed": update.crashed,
                    "timed_out": update.timed_out,
                    "wins_new": update.wins_new,
                    "wins_old": update.wins_old,
                    "draws": update.draws,
                    "extra_json": update.extra_json,
                },
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            _warn(f"push_progress({run_id})", exc)
        return
    if _resolved_db_path() is not None:
        try:
            _direct(lambda c: _db.upsert_progress(c, update))
        except Exception as exc:  # noqa: BLE001
            _warn(f"push_progress({run_id})", exc)


def push_event(
    run_id: str,
    event_type: str,
    *,
    shard_label: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    """Append an event to the run's log."""
    payload_json = json.dumps(payload, sort_keys=True) if payload else None
    event = Event(
        run_id=run_id,
        event_type=event_type,
        shard_label=shard_label,
        payload_json=payload_json,
    )
    url = _resolved_db_url()
    if url is not None:
        try:
            _post_json(
                f"{url}/run/{run_id}/event",
                {
                    "event_type": event.event_type,
                    "shard_label": event.shard_label,
                    "payload_json": event.payload_json,
                },
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            _warn(f"push_event({run_id}, {event_type})", exc)
        return
    if _resolved_db_path() is not None:
        try:
            _direct(lambda c: _db.insert_event(c, event))
        except Exception as exc:  # noqa: BLE001
            _warn(f"push_event({run_id}, {event_type})", exc)


def push_artifact(
    run_id: str,
    *,
    artifact_type: str,
    path: str,
    bytes_: Optional[int] = None,
) -> None:
    """Register an on-disk artifact against the run."""
    art = Artifact(
        run_id=run_id,
        artifact_type=artifact_type,
        path=path,
        bytes=bytes_,
    )
    url = _resolved_db_url()
    if url is not None:
        try:
            _post_json(
                f"{url}/run/{run_id}/artifact",
                {
                    "artifact_type": art.artifact_type,
                    "path": art.path,
                    "bytes": art.bytes,
                },
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            _warn(f"push_artifact({run_id})", exc)
        return
    if _resolved_db_path() is not None:
        try:
            _direct(lambda c: _db.upsert_artifact(c, art))
        except Exception as exc:  # noqa: BLE001
            _warn(f"push_artifact({run_id})", exc)


def finish_run(
    run_id: str,
    status: str,
    *,
    notes: Optional[str] = None,
) -> None:
    """Mark a run terminal. ``status`` is one of completed / failed /
    killed."""
    url = _resolved_db_url()
    if url is not None:
        try:
            _post_json(
                f"{url}/run/{run_id}/finish",
                {"status": status, "notes": notes},
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            _warn(f"finish_run({run_id})", exc)
        return
    if _resolved_db_path() is not None:
        try:
            _direct(
                lambda c: _db.finish_run(
                    c, run_id, status, notes=notes,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _warn(f"finish_run({run_id})", exc)


__all__ = [
    "configure",
    "finish_run",
    "is_configured",
    "push_artifact",
    "push_event",
    "push_progress",
    "start_run",
]
