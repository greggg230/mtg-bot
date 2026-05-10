"""End-to-end test that ``mtg_bot eval``'s ``main()`` body
correctly issues the start_run -> push_eval_summary -> finish_run
trio against the SQLite run DB.

The body is ``# pragma: no cover`` (argparse + orchestrator
plumbing); we test it here by stubbing out ``run_distributed_eval``
with a fake that returns canned counts, pointing
``MTG_BOT_RUN_DB_PATH`` at a tmp DB, and verifying the rows
landed correctly. This pins the contract between
``mtg_bot.cli.eval.main`` and ``mtg_bot.cli.runs.lifecycle``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mtg_bot.cli import eval as cli_eval
from mtg_bot.cli.runs import db as runs_db, lifecycle


def test_eval_main_round_trips_start_summary_finish(
    monkeypatch, tmp_path,
):
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc1234")

    progress_dir = tmp_path / "training_progress"
    deck_pool = tmp_path / "decks.json.gz"
    deck_pool.write_bytes(b"")  # touch — eval doesn't open it in this test
    shards_config = tmp_path / "shards.json"
    shards_config.write_text(json.dumps({"shards": []}), encoding="utf-8")

    fake_counts = {
        "wins_new": 320, "wins_old": 290,
        "draws": 10, "crashed": 2,
    }

    captured_call = {}

    def fake_run_distributed_eval(*args, **kwargs):
        captured_call["args"] = args
        captured_call["kwargs"] = kwargs
        # Touch the progress JSON like the real impl would, so the
        # artifact registration sees a real file size.
        progress_json = (
            kwargs["progress_dir"]
            / f"{kwargs['run_id_prefix']}.json"
        )
        progress_json.parent.mkdir(parents=True, exist_ok=True)
        progress_json.write_text("{}", encoding="utf-8")
        return fake_counts

    fake_load_shards = lambda p: {"shards": []}  # noqa: E731

    with patch.object(
        cli_eval, "run_distributed_eval", fake_run_distributed_eval,
    ), patch.object(
        cli_eval, "load_shards_config", fake_load_shards,
    ):
        cli_eval.main([
            "--new", "v16", "--old", "v13",
            "--games", "622",
            "--shards-config", str(shards_config),
            "--progress-dir", str(progress_dir),
            "--deck-pool", str(deck_pool),
            "--queue-url", "http://master:5099",
            "--bundle-url", "http://master:5099/repo.bundle",
            "--seed", "42",
            "--nodes", "200",
            "--samples", "4",
        ])

    # Subcommand wired its body around the lifecycle CM.
    assert "args" in captured_call
    assert captured_call["kwargs"]["run_id_prefix"].startswith("eval_")

    # Verify DB rows landed.
    conn = runs_db.connect(db_path)
    try:
        run_rows = list(conn.execute("SELECT * FROM runs"))
        progress_rows = list(conn.execute("SELECT * FROM run_progress"))
        artifact_rows = list(conn.execute("SELECT * FROM run_artifacts"))
    finally:
        conn.close()

    assert len(run_rows) == 1
    run_id = run_rows[0]["run_id"]
    assert run_id == "eval_v16_vs_v13_622g"
    assert run_rows[0]["status"] == "completed"
    assert run_rows[0]["kind"] == "eval"
    assert run_rows[0]["git_commit"] == "abc1234"
    config = json.loads(run_rows[0]["config_json"])
    assert config["new"] == "v16"
    assert config["old"] == "v13"
    assert config["games"] == 622

    # The aggregate summary row records the eval counts.
    assert len(progress_rows) == 1
    p = progress_rows[0]
    assert p["wins_new"] == 320
    assert p["wins_old"] == 290
    assert p["draws"] == 10
    assert p["crashed"] == 2
    assert p["n_games_total"] == 622
    extra = json.loads(p["extra_json"])
    assert extra["win_rate"] == pytest.approx(320 / 610)

    # The progress JSON is registered as an artifact.
    assert len(artifact_rows) == 1
    assert artifact_rows[0]["artifact_type"] == "progress_json"


def test_eval_main_marks_failed_on_exception(monkeypatch, tmp_path):
    """If run_distributed_eval raises, the lifecycle CM should
    catch -> push finish_run('failed') -> re-raise. The DB row
    must reflect failed status."""
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")

    progress_dir = tmp_path / "training_progress"
    deck_pool = tmp_path / "decks.json.gz"
    deck_pool.write_bytes(b"")
    shards_config = tmp_path / "shards.json"
    shards_config.write_text("{}", encoding="utf-8")

    def boom(*a, **kw):
        raise RuntimeError("self-play crashed")

    with patch.object(
        cli_eval, "run_distributed_eval", boom,
    ), patch.object(
        cli_eval, "load_shards_config", lambda p: {"shards": []},
    ):
        with pytest.raises(RuntimeError, match="self-play crashed"):
            cli_eval.main([
                "--new", "v16", "--old", "v13",
                "--games", "10",
                "--shards-config", str(shards_config),
                "--progress-dir", str(progress_dir),
                "--deck-pool", str(deck_pool),
                "--queue-url", "http://master:5099",
                "--bundle-url", "http://master:5099/repo.bundle",
            ])

    conn = runs_db.connect(db_path)
    try:
        rows = list(conn.execute("SELECT status, notes FROM runs"))
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "RuntimeError" in (rows[0]["notes"] or "")
    assert "self-play crashed" in (rows[0]["notes"] or "")


def test_eval_main_marks_killed_on_sigint(monkeypatch, tmp_path):
    """KeyboardInterrupt mid-run should land in the DB as
    status=killed (not failed)."""
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc")

    progress_dir = tmp_path / "training_progress"
    deck_pool = tmp_path / "decks.json.gz"
    deck_pool.write_bytes(b"")
    shards_config = tmp_path / "shards.json"
    shards_config.write_text("{}", encoding="utf-8")

    def interrupted(*a, **kw):
        raise KeyboardInterrupt()

    with patch.object(
        cli_eval, "run_distributed_eval", interrupted,
    ), patch.object(
        cli_eval, "load_shards_config", lambda p: {"shards": []},
    ):
        with pytest.raises(KeyboardInterrupt):
            cli_eval.main([
                "--new", "v16", "--old", "v13",
                "--games", "10",
                "--shards-config", str(shards_config),
                "--progress-dir", str(progress_dir),
                "--deck-pool", str(deck_pool),
                "--queue-url", "http://master:5099",
                "--bundle-url", "http://master:5099/repo.bundle",
            ])

    conn = runs_db.connect(db_path)
    try:
        rows = list(conn.execute("SELECT status FROM runs"))
    finally:
        conn.close()
    assert rows[0]["status"] == "killed"
