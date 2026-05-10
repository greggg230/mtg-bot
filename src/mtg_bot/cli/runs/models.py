"""Typed record dataclasses for the run database.

Mirror the rows in ``runs.db`` — see :mod:`mtg_bot.cli.runs.db` for
the schema. These exist so callers (and tests) can ferry around row
data with attribute access instead of dict-key juggling, and so the
REST endpoint payloads have one canonical shape they round-trip
through.

Every dataclass is plain — no validators, no slots. The DB layer
serializes them to/from sqlite rows; the REST layer (in
:mod:`mtg_bot.cli.serve`) serializes them to/from JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Allowed enum-like strings. Mirror the CHECK constraints in db.py.
KINDS = ("iterate", "self_play", "eval", "bench", "capture", "training")
STATUSES = ("running", "completed", "failed", "killed")


@dataclass
class Run:
    """A row from the ``runs`` table."""

    run_id: str
    kind: str
    status: str
    started_at: str  # ISO 8601 UTC
    git_commit: str
    config_json: str  # already JSON-serialized; the DB stores TEXT
    ended_at: Optional[str] = None
    parent_run_id: Optional[str] = None
    label: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ProgressUpdate:
    """A row from the ``run_progress`` table.

    ``shard_label`` is ``None`` for the aggregate row (the rolled-up
    cluster total). Per-shard rows carry the shard's display label.
    The ``(run_id, shard_label)`` pair is the upsert key — repeated
    pushes update the same row in place.
    """

    run_id: str
    shard_label: Optional[str] = None
    n_games_done: Optional[int] = None
    n_games_total: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    crashed: Optional[int] = None
    timed_out: Optional[int] = None
    wins_new: Optional[int] = None
    wins_old: Optional[int] = None
    draws: Optional[int] = None
    extra_json: Optional[str] = None  # JSON-serialized free-form fields
    updated_at: Optional[str] = None  # filled in by the DB layer if None


@dataclass
class Event:
    """A row from the ``run_events`` table.

    Event log: append-only, used for stage transitions, exceptions,
    promotion decisions, eval results, and any other one-shot fact
    we want timestamped against a run.
    """

    run_id: str
    event_type: str
    shard_label: Optional[str] = None
    payload_json: Optional[str] = None  # JSON-serialized
    timestamp: Optional[str] = None  # filled in by the DB layer if None


@dataclass
class Artifact:
    """A row from the ``run_artifacts`` table.

    Tracks the on-disk byproducts of a run — weights, replays, npz,
    log, and (Phase 1 specifically) the legacy progress JSON files
    so the DB knows where to find them.
    """

    run_id: str
    artifact_type: str  # 'weights', 'replay', 'npz', 'log', 'progress_json'
    path: str
    bytes: Optional[int] = None
    created_at: Optional[str] = None


__all__ = [
    "Artifact",
    "Event",
    "KINDS",
    "ProgressUpdate",
    "Run",
    "STATUSES",
    "Status",
    "Kind",
]


# Aliases callers can use for clarity. Strings, since SQLite stores
# them as TEXT and Python's typing here is informal.
Kind = str
Status = str
