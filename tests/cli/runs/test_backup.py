"""Tests for ``mtg_bot.cli.runs.backup`` — the admin tooling.

Covers:
- ``backup_db`` produces a valid SQLite file with same row counts.
- Backup refuses to overwrite an existing file (not destructive).
- ``dump_run`` returns the same shape as ``GET /run/<id>``.
- CLI dispatch (``backup``, ``dump``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mtg_bot.cli.runs import backup as runs_backup, db as runs_db
from mtg_bot.cli.runs.models import Event, ProgressUpdate, Run


def _seed_db(db_path: Path) -> None:
    conn = runs_db.connect(db_path)
    try:
        runs_db.insert_run(
            conn,
            Run(
                run_id="r1", kind="eval", status="running",
                started_at="2026-05-06T12:00:00+00:00",
                git_commit="abc1234", config_json="{}",
            ),
        )
        runs_db.insert_run(
            conn,
            Run(
                run_id="r2", kind="bench", status="running",
                started_at="2026-05-06T12:01:00+00:00",
                git_commit="abc1234", config_json="{}",
            ),
        )
        runs_db.upsert_progress(
            conn,
            ProgressUpdate(run_id="r1", n_games_done=10),
        )
        runs_db.insert_event(
            conn, Event(run_id="r1", event_type="started"),
        )
    finally:
        conn.close()


def test_backup_creates_valid_sqlite_file(tmp_path):
    src = tmp_path / "runs.db"
    _seed_db(src)
    dest = tmp_path / "runs.db.bak"
    out = runs_backup.backup_db(src, dest)
    assert out == dest
    assert dest.is_file()
    # Open the backup directly — should be a real SQLite file.
    conn = sqlite3.connect(str(dest))
    try:
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_progress = conn.execute(
            "SELECT COUNT(*) FROM run_progress"
        ).fetchone()[0]
        n_events = conn.execute(
            "SELECT COUNT(*) FROM run_events"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_runs == 2
    assert n_progress == 1
    assert n_events == 1


def test_backup_refuses_to_overwrite(tmp_path):
    src = tmp_path / "runs.db"
    _seed_db(src)
    dest = tmp_path / "runs.db.bak"
    runs_backup.backup_db(src, dest)
    with pytest.raises(FileExistsError):
        runs_backup.backup_db(src, dest)


def test_backup_missing_source(tmp_path):
    src = tmp_path / "missing.db"
    dest = tmp_path / "runs.db.bak"
    with pytest.raises(FileNotFoundError):
        runs_backup.backup_db(src, dest)


def test_backup_creates_dest_parent(tmp_path):
    src = tmp_path / "runs.db"
    _seed_db(src)
    dest = tmp_path / "nested" / "subdir" / "runs.db.bak"
    runs_backup.backup_db(src, dest)
    assert dest.is_file()


def test_dump_run_returns_full_record(tmp_path):
    src = tmp_path / "runs.db"
    _seed_db(src)
    record = runs_backup.dump_run(src, "r1")
    assert record["run"]["run_id"] == "r1"
    assert len(record["progress"]) == 1
    assert len(record["events"]) == 1


def test_dump_run_unknown_id_raises(tmp_path):
    src = tmp_path / "runs.db"
    _seed_db(src)
    with pytest.raises(KeyError):
        runs_backup.dump_run(src, "nope")


# ---- CLI dispatch ---------------------------------------------------------


def test_cli_backup_default_dest(tmp_path, capsys, monkeypatch):
    src = tmp_path / "runs.db"
    _seed_db(src)
    out_dir = tmp_path / "backups"
    monkeypatch.chdir(tmp_path)
    runs_backup.main([
        "backup",
        "--db", str(src),
        "--out", str(out_dir / "runs.db.test.bak"),
    ])
    captured = capsys.readouterr().out
    assert "backup ok" in captured
    assert (out_dir / "runs.db.test.bak").is_file()


def test_cli_dump(tmp_path, capsys):
    src = tmp_path / "runs.db"
    _seed_db(src)
    runs_backup.main([
        "dump",
        "--db", str(src),
        "--run-id", "r1",
    ])
    out = capsys.readouterr().out
    record = json.loads(out)
    assert record["run"]["run_id"] == "r1"


def test_cli_backup_default_out_uses_timestamp(tmp_path, monkeypatch, capsys):
    """Without --out, default lands under
    training_progress/backups/runs.db.<UTC>.bak."""
    src = tmp_path / "runs.db"
    _seed_db(src)
    monkeypatch.chdir(tmp_path)
    runs_backup.main([
        "backup",
        "--db", str(src),
    ])
    out = capsys.readouterr().out
    # Output line names a file under training_progress/backups
    assert "training_progress" in out
    assert "backups" in out
    # Find the file
    backups_dir = tmp_path / "training_progress" / "backups"
    files = list(backups_dir.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("runs.db.")
    assert files[0].name.endswith(".bak")
