"""Tests for ``mtg_bot.cli.runs.db`` — the SQLite layer.

Per-test DB lives under ``tmp_path`` so test isolation is automatic
(no global state to clear). Coverage targets:

- schema apply is idempotent
- run / progress / event / artifact round-trips
- WAL mode + multiple readers + one writer
- finish_run rejects 'running'
- list_runs filters
- get_run returns None on miss
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from mtg_bot.cli.runs import db as runs_db
from mtg_bot.cli.runs.models import Artifact, Event, ProgressUpdate, Run


def _fresh_db(tmp_path: Path) -> Path:
    return tmp_path / "runs.db"


def _make_run(run_id: str = "r1", **overrides) -> Run:
    base = dict(
        run_id=run_id,
        kind="eval",
        status="running",
        started_at="2026-05-06T12:00:00+00:00",
        git_commit="6d7b9c4",
        config_json=json.dumps({"games": 100}),
    )
    base.update(overrides)
    return Run(**base)


# ---- schema apply ---------------------------------------------------------


def test_connect_creates_db_and_applies_schema(tmp_path):
    db_path = _fresh_db(tmp_path)
    assert not db_path.exists()
    conn = runs_db.connect(db_path)
    try:
        # Tables exist
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in rows}
        assert {"runs", "run_progress", "run_events", "run_artifacts"} <= names
        # WAL mode active
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        # Foreign keys enabled
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
    finally:
        conn.close()
    assert db_path.exists()


def test_apply_schema_is_idempotent(tmp_path):
    """Calling apply_schema twice on the same DB must not error."""
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.apply_schema(conn)
        runs_db.apply_schema(conn)  # no-op
        # And the indices still exist (no duplicate-create error).
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        names = {r["name"] for r in rows}
        assert {"idx_runs_status", "idx_runs_kind",
                "idx_runs_started_at", "idx_events_run"} <= names
    finally:
        conn.close()


def test_apply_schema_on_existing_db_preserves_rows(tmp_path):
    """Re-applying schema (the migration path on an existing DB)
    must not blow away existing rows."""
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        runs_db.apply_schema(conn)  # second apply
        rows = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()
        assert rows["c"] == 1
    finally:
        conn.close()


# ---- run round-trip -------------------------------------------------------


def test_insert_and_get_run(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1", label="my-eval"))
        record = runs_db.get_run(conn, "r1")
        assert record is not None
        assert record["run"]["run_id"] == "r1"
        assert record["run"]["label"] == "my-eval"
        assert record["run"]["status"] == "running"
        assert record["progress"] == []
        assert record["events"] == []
        assert record["artifacts"] == []
    finally:
        conn.close()


def test_get_run_returns_none_when_missing(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        assert runs_db.get_run(conn, "nope") is None
    finally:
        conn.close()


def test_insert_run_rejects_invalid_kind(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        with pytest.raises(ValueError):
            runs_db.insert_run(
                conn, _make_run("r1", kind="bogus_kind"),
            )
    finally:
        conn.close()


def test_insert_run_rejects_invalid_status(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        with pytest.raises(ValueError):
            runs_db.insert_run(
                conn, _make_run("r1", status="bogus"),
            )
    finally:
        conn.close()


def test_insert_run_duplicate_id_raises(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        with pytest.raises(sqlite3.IntegrityError):
            runs_db.insert_run(conn, _make_run("r1"))
    finally:
        conn.close()


# ---- progress upserts -----------------------------------------------------


def test_upsert_progress_aggregate_and_per_shard(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        # Aggregate row (NULL shard_label).
        runs_db.upsert_progress(
            conn,
            ProgressUpdate(
                run_id="r1",
                n_games_done=0, n_games_total=100,
                elapsed_seconds=1.0,
            ),
        )
        # Per-shard row.
        runs_db.upsert_progress(
            conn,
            ProgressUpdate(
                run_id="r1", shard_label="desktop",
                n_games_done=10, n_games_total=50,
                elapsed_seconds=1.5,
            ),
        )
        record = runs_db.get_run(conn, "r1")
        assert len(record["progress"]) == 2
        # Aggregate row sorts first.
        assert record["progress"][0]["shard_label"] is None
        assert record["progress"][0]["n_games_done"] == 0
        assert record["progress"][1]["shard_label"] == "desktop"
        assert record["progress"][1]["n_games_done"] == 10

        # Now upsert the aggregate again — same row, updated counts.
        runs_db.upsert_progress(
            conn,
            ProgressUpdate(
                run_id="r1",
                n_games_done=42, n_games_total=100,
                elapsed_seconds=10.0,
            ),
        )
        record = runs_db.get_run(conn, "r1")
        # Still 2 rows (no duplicate insert).
        assert len(record["progress"]) == 2
        agg = next(p for p in record["progress"] if p["shard_label"] is None)
        assert agg["n_games_done"] == 42
        assert agg["elapsed_seconds"] == 10.0
    finally:
        conn.close()


def test_upsert_progress_extra_json(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        runs_db.upsert_progress(
            conn,
            ProgressUpdate(
                run_id="r1",
                extra_json=json.dumps({"foo": "bar"}),
            ),
        )
        record = runs_db.get_run(conn, "r1")
        assert json.loads(record["progress"][0]["extra_json"]) == {
            "foo": "bar",
        }
    finally:
        conn.close()


# ---- events ---------------------------------------------------------------


def test_insert_event_appends(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        runs_db.insert_event(
            conn,
            Event(run_id="r1", event_type="started"),
        )
        runs_db.insert_event(
            conn,
            Event(
                run_id="r1", event_type="game_done",
                payload_json=json.dumps({"winner": 0}),
            ),
        )
        record = runs_db.get_run(conn, "r1")
        # Events ORDER BY id DESC, so newest first.
        assert len(record["events"]) == 2
        assert record["events"][0]["event_type"] == "game_done"
        assert record["events"][1]["event_type"] == "started"
        assert json.loads(record["events"][0]["payload_json"]) == {
            "winner": 0,
        }
    finally:
        conn.close()


# ---- artifacts ------------------------------------------------------------


def test_upsert_artifact_idempotent(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        runs_db.upsert_artifact(
            conn,
            Artifact(
                run_id="r1", artifact_type="progress_json",
                path="training_progress/r1.json", bytes=100,
            ),
        )
        runs_db.upsert_artifact(
            conn,
            Artifact(
                run_id="r1", artifact_type="progress_json",
                path="training_progress/r1.json", bytes=200,
            ),
        )
        record = runs_db.get_run(conn, "r1")
        assert len(record["artifacts"]) == 1
        assert record["artifacts"][0]["bytes"] == 200
    finally:
        conn.close()


# ---- finish_run -----------------------------------------------------------


def test_finish_run_completed(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        runs_db.finish_run(conn, "r1", "completed", notes="went fine")
        record = runs_db.get_run(conn, "r1")
        assert record["run"]["status"] == "completed"
        assert record["run"]["ended_at"] is not None
        assert record["run"]["notes"] == "went fine"
    finally:
        conn.close()


def test_finish_run_rejects_running(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        with pytest.raises(ValueError):
            runs_db.finish_run(conn, "r1", "running")
    finally:
        conn.close()


def test_finish_run_rejects_unknown_status(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        with pytest.raises(ValueError):
            runs_db.finish_run(conn, "r1", "weird")
    finally:
        conn.close()


def test_finish_run_without_notes(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1", notes="orig"))
        runs_db.finish_run(conn, "r1", "killed")
        record = runs_db.get_run(conn, "r1")
        assert record["run"]["status"] == "killed"
        # notes preserved when not passed
        assert record["run"]["notes"] == "orig"
    finally:
        conn.close()


# ---- list_runs ------------------------------------------------------------


def test_list_runs_filters(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run(
            "r1", kind="eval",
            started_at="2026-05-06T10:00:00+00:00",
        ))
        runs_db.insert_run(conn, _make_run(
            "r2", kind="bench",
            started_at="2026-05-06T11:00:00+00:00",
        ))
        runs_db.insert_run(conn, _make_run(
            "r3", kind="eval",
            started_at="2026-05-06T12:00:00+00:00",
        ))
        runs_db.finish_run(conn, "r2", "completed")
        runs_db.finish_run(conn, "r3", "failed")

        # All
        rows = runs_db.list_runs(conn)
        assert [r["run_id"] for r in rows] == ["r3", "r2", "r1"]

        # By status
        running = runs_db.list_runs(conn, status="running")
        assert [r["run_id"] for r in running] == ["r1"]

        # By kind
        evals = runs_db.list_runs(conn, kind="eval")
        assert {r["run_id"] for r in evals} == {"r1", "r3"}

        # By since
        recent = runs_db.list_runs(
            conn, since="2026-05-06T11:30:00+00:00",
        )
        assert [r["run_id"] for r in recent] == ["r3"]

        # Limit
        one = runs_db.list_runs(conn, limit=1)
        assert len(one) == 1
        assert one[0]["run_id"] == "r3"

        # Combined
        eval_running = runs_db.list_runs(
            conn, kind="eval", status="running",
        )
        assert [r["run_id"] for r in eval_running] == ["r1"]
    finally:
        conn.close()


# ---- WAL concurrency ------------------------------------------------------


def test_wal_allows_concurrent_readers_during_write(tmp_path):
    """Open three connections: one writer, two readers. The writer
    holds an open BEGIN IMMEDIATE; both readers must still be able to
    SELECT under WAL. (Without WAL — i.e. rollback-journal mode —
    readers would block on the writer's lock.)"""
    db_path = _fresh_db(tmp_path)
    setup = runs_db.connect(db_path)
    try:
        runs_db.insert_run(setup, _make_run("r1"))
    finally:
        setup.close()

    writer = runs_db.connect(db_path)
    reader_a = runs_db.connect(db_path)
    reader_b = runs_db.connect(db_path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO run_events(run_id, event_type, timestamp) "
            "VALUES ('r1', 'mid_write', ?)",
            (runs_db.now_iso(),),
        )
        # Readers should not see uncommitted row, but should not block.
        cnt_a = reader_a.execute(
            "SELECT COUNT(*) AS c FROM run_events"
        ).fetchone()["c"]
        cnt_b = reader_b.execute(
            "SELECT COUNT(*) AS c FROM runs"
        ).fetchone()["c"]
        assert cnt_a == 0  # write not yet committed
        assert cnt_b == 1  # the run row was committed up front
        writer.execute("COMMIT")
        # Now committed.
        cnt_after = reader_a.execute(
            "SELECT COUNT(*) AS c FROM run_events"
        ).fetchone()["c"]
        assert cnt_after == 1
    finally:
        writer.close()
        reader_a.close()
        reader_b.close()


def test_concurrent_progress_pushes_serialize(tmp_path):
    """Hammer ``upsert_progress`` from N threads. The module-level
    WRITE_LOCK + WAL mode should produce a final state with one row
    per shard_label and no SQL errors."""
    db_path = _fresh_db(tmp_path)
    setup = runs_db.connect(db_path)
    try:
        runs_db.insert_run(setup, _make_run("r1"))
    finally:
        setup.close()

    n_threads = 8
    n_pushes_per_thread = 25
    errors: list[BaseException] = []

    def worker(label: str) -> None:
        try:
            for i in range(n_pushes_per_thread):
                conn = runs_db.connect(db_path)
                try:
                    with runs_db.WRITE_LOCK:
                        runs_db.upsert_progress(
                            conn,
                            ProgressUpdate(
                                run_id="r1",
                                shard_label=label,
                                n_games_done=i,
                                elapsed_seconds=float(i),
                            ),
                        )
                finally:
                    conn.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(f"shard{i}",))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors

    # Each shard has exactly one row, n_games_done = last write
    # (n_pushes_per_thread - 1).
    conn = runs_db.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT shard_label, n_games_done FROM run_progress "
            "WHERE run_id='r1' ORDER BY shard_label"
        ).fetchall()
        assert len(rows) == n_threads
        for r in rows:
            assert r["n_games_done"] == n_pushes_per_thread - 1
    finally:
        conn.close()


# ---- transaction / rollback ----------------------------------------------


def test_transaction_rollback_on_exception(tmp_path):
    db_path = _fresh_db(tmp_path)
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(conn, _make_run("r1"))
        with pytest.raises(RuntimeError):
            with runs_db.transaction(conn):
                conn.execute(
                    "INSERT INTO run_events(run_id, event_type, timestamp) "
                    "VALUES ('r1', 'oops', ?)",
                    (runs_db.now_iso(),),
                )
                raise RuntimeError("explode")
        # Row should NOT be present because the txn rolled back.
        cnt = conn.execute(
            "SELECT COUNT(*) AS c FROM run_events"
        ).fetchone()["c"]
        assert cnt == 0
    finally:
        conn.close()


def test_now_iso_is_iso8601(tmp_path):
    s = runs_db.now_iso()
    # Loose check — parses + carries timezone offset.
    assert "T" in s
    assert "+" in s or s.endswith("Z")
