"""Tests for ``mtg_bot.cli.runs.lifecycle`` — the high-level
context-manager subcommands wrap themselves in.

The module is the seam between ``main()`` (which is # pragma: no cover
because it's argparse + orchestrator plumbing) and the
DB-write client. By pinning the lifecycle helpers' behavior here,
we get test signal on the start_run/finish_run contract every
subcommand depends on without having to test every main() body.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mtg_bot.cli.runs import client, db as runs_db, lifecycle


# ---- git_commit_short ------------------------------------------------------


def test_git_commit_short_returns_unknown_when_git_missing(monkeypatch):
    """Simulate ``git`` not on PATH (FileNotFoundError) — function
    must return ``unknown`` rather than raise."""
    import subprocess as _subprocess

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(_subprocess, "run", fake_run)
    assert lifecycle.git_commit_short() == "unknown"


def test_git_commit_short_returns_unknown_on_nonzero_exit(monkeypatch):
    import subprocess as _subprocess

    class _FakeResult:
        def __init__(self):
            self.returncode = 128
            self.stdout = ""

    monkeypatch.setattr(
        _subprocess, "run",
        lambda *a, **kw: _FakeResult(),
    )
    assert lifecycle.git_commit_short() == "unknown"


def test_git_commit_short_returns_sha_on_success(monkeypatch):
    import subprocess as _subprocess

    class _FakeResult:
        def __init__(self):
            self.returncode = 0
            self.stdout = "abc1234\n"

    monkeypatch.setattr(
        _subprocess, "run",
        lambda *a, **kw: _FakeResult(),
    )
    assert lifecycle.git_commit_short() == "abc1234"


# ---- serialize_config ------------------------------------------------------


def test_serialize_config_handles_path():
    """Path objects don't json.dumps natively; lifecycle must coerce
    them via the default= callback."""
    out = lifecycle.serialize_config({
        "data_dir": Path("training_data"),
        "games": 100,
    })
    parsed = json.loads(out)
    assert parsed["data_dir"] == "training_data"
    assert parsed["games"] == 100


def test_serialize_config_sorts_keys():
    out = lifecycle.serialize_config({"b": 1, "a": 2})
    # Sorted: 'a' before 'b'.
    assert out.index('"a"') < out.index('"b"')


# ---- run_lifecycle context manager -----------------------------------------


def _setup_db(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))
    return db_path


def test_run_lifecycle_completed_on_normal_exit(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc1234")
    with lifecycle.run_lifecycle(
        run_id="r1", kind="eval",
        config={"games": 5},
        label="my-eval",
    ):
        pass

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert record["run"]["status"] == "completed"
    assert record["run"]["label"] == "my-eval"
    assert record["run"]["git_commit"] == "abc1234"
    assert json.loads(record["run"]["config_json"]) == {"games": 5}


def test_run_lifecycle_failed_on_exception(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc1234")
    with pytest.raises(RuntimeError):
        with lifecycle.run_lifecycle(
            run_id="r1", kind="bench",
            config={},
        ):
            raise RuntimeError("boom")

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert record["run"]["status"] == "failed"
    assert "RuntimeError" in record["run"]["notes"]
    assert "boom" in record["run"]["notes"]


def test_run_lifecycle_killed_on_keyboard_interrupt(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")
    with pytest.raises(KeyboardInterrupt):
        with lifecycle.run_lifecycle(
            run_id="r1", kind="iterate", config={},
        ):
            raise KeyboardInterrupt()

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert record["run"]["status"] == "killed"


def test_run_lifecycle_killed_on_system_exit(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")
    with pytest.raises(SystemExit):
        with lifecycle.run_lifecycle(
            run_id="r1", kind="iterate", config={},
        ):
            raise SystemExit(1)

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert record["run"]["status"] == "killed"


def test_run_lifecycle_registers_progress_artifact(
    monkeypatch, tmp_path,
):
    db_path = _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")
    progress_json = tmp_path / "training_progress" / "r1.json"
    progress_json.parent.mkdir(parents=True, exist_ok=True)
    progress_json.write_text('{"x": 1}', encoding="utf-8")
    with lifecycle.run_lifecycle(
        run_id="r1", kind="eval", config={},
        progress_json_path=progress_json,
    ):
        pass

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert len(record["artifacts"]) == 1
    assert record["artifacts"][0]["artifact_type"] == "progress_json"
    assert record["artifacts"][0]["path"] == str(progress_json)
    assert record["artifacts"][0]["bytes"] == len('{"x": 1}')


def test_run_lifecycle_registers_artifact_path_not_yet_existing(
    monkeypatch, tmp_path,
):
    """The progress JSON typically gets created INSIDE the
    lifecycle's body (during run_distributed_*). Registering a
    not-yet-existing path should record bytes=None rather than
    raise."""
    db_path = _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")
    progress_json = tmp_path / "later.json"
    with lifecycle.run_lifecycle(
        run_id="r1", kind="eval", config={},
        progress_json_path=progress_json,
    ):
        pass

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert record["artifacts"][0]["bytes"] is None


def test_run_lifecycle_noop_when_db_unconfigured(monkeypatch, tmp_path):
    """Phase 1 invariant: with neither env var set, the lifecycle
    helpers must be no-ops — no DB file created, no exception."""
    monkeypatch.delenv("MTG_BOT_RUN_DB_PATH", raising=False)
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")
    with lifecycle.run_lifecycle(
        run_id="r1", kind="eval", config={},
    ):
        pass
    assert not (tmp_path / "runs.db").exists()


# ---- push_eval_summary -----------------------------------------------------


def test_push_eval_summary_writes_aggregate_progress(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")
    client.start_run(
        "r1", kind="eval",
        config_json="{}", git_commit="abc",
    )
    lifecycle.push_eval_summary(
        "r1",
        wins_new=320, wins_old=290, draws=10, crashed=2,
        elapsed_seconds=1234.5,
        extra={"win_rate": 0.5246},
    )

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    progress = record["progress"][0]
    assert progress["wins_new"] == 320
    assert progress["wins_old"] == 290
    assert progress["draws"] == 10
    assert progress["crashed"] == 2
    assert progress["n_games_done"] == 622
    assert progress["n_games_total"] == 622
    assert progress["elapsed_seconds"] == 1234.5
    assert json.loads(progress["extra_json"]) == {"win_rate": 0.5246}
