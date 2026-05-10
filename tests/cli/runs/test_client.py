"""Tests for ``mtg_bot.cli.runs.client`` — the cross-process façade.

Direct-SQLite path is exercised end-to-end against a tmp DB. The REST
path is exercised with a stubbed ``urlopen``: we patch
``urllib.request.urlopen`` to capture POST requests and return canned
responses, since spinning up a real Flask server here would tangle
with ``test_rest.py``'s job (that's where we test the server side).

The "no transport configured -> no-op" path is covered to confirm
Phase 1's "purely additive" invariant: callers are safe to call
client.* unconditionally.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from mtg_bot.cli.runs import client, db as runs_db


# ---- env / configure helpers ----------------------------------------------


def _clear_env(monkeypatch):
    monkeypatch.delenv("MTG_BOT_RUN_DB_PATH", raising=False)
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)


def test_is_configured_default_false(monkeypatch):
    _clear_env(monkeypatch)
    assert client.is_configured() is False


def test_configure_sets_env_path(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    db_path = tmp_path / "runs.db"
    client.configure(db_path=db_path)
    assert client.is_configured() is True
    client.configure(db_path=None)
    assert client.is_configured() is False


def test_configure_sets_env_url(monkeypatch):
    _clear_env(monkeypatch)
    client.configure(db_url="http://master:5099")
    assert client.is_configured() is True
    client.configure(db_url=None)
    assert client.is_configured() is False


# ---- direct-SQLite path ---------------------------------------------------


def test_direct_lifecycle_round_trip(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))

    client.start_run(
        "r1", kind="eval",
        config_json=json.dumps({"games": 100}),
        git_commit="abc1234",
        label="my-eval",
    )
    client.push_progress(
        "r1", n_games_done=5, n_games_total=100,
        elapsed_seconds=1.5,
    )
    client.push_progress(
        "r1", shard_label="desktop",
        n_games_done=3, elapsed_seconds=1.5,
    )
    client.push_event("r1", "started", payload={"foo": 1})
    client.push_artifact(
        "r1", artifact_type="progress_json",
        path="training_progress/r1.json", bytes_=1024,
    )
    client.finish_run("r1", "completed", notes="all good")

    # Verify via a direct read.
    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert record is not None
    assert record["run"]["status"] == "completed"
    assert record["run"]["label"] == "my-eval"
    assert record["run"]["notes"] == "all good"
    assert len(record["progress"]) == 2
    assert len(record["events"]) == 1
    assert record["events"][0]["event_type"] == "started"
    assert len(record["artifacts"]) == 1


def test_direct_extra_json_serialized(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))

    client.start_run(
        "r1", kind="bench",
        config_json="{}", git_commit="abc",
    )
    client.push_progress("r1", extra={"win_rate": 0.51})

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    extra = json.loads(record["progress"][0]["extra_json"])
    assert extra == {"win_rate": 0.51}


def test_no_transport_calls_are_noops(monkeypatch, tmp_path):
    """With neither env var set, every client call is a no-op (no
    files created, no exceptions raised)."""
    _clear_env(monkeypatch)
    # These must not raise even without configuration.
    client.start_run(
        "r1", kind="eval", config_json="{}", git_commit="abc",
    )
    client.push_progress("r1", n_games_done=1)
    client.push_event("r1", "test")
    client.push_artifact(
        "r1", artifact_type="log", path="x.log",
    )
    client.finish_run("r1", "completed")
    # No DB files created anywhere obvious.
    assert not (tmp_path / "runs.db").exists()


def test_direct_path_swallows_db_errors(monkeypatch, tmp_path, capsys):
    """If the DB can't be written for some reason (e.g. an invalid
    kind sneaks in), the client must warn but not raise — Phase 1
    invariant."""
    _clear_env(monkeypatch)
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))

    # Bad kind — direct insert raises ValueError. The client must
    # swallow it.
    client.start_run(
        "r1", kind="not_a_kind",
        config_json="{}", git_commit="abc",
    )
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "start_run" in err


# ---- REST path ------------------------------------------------------------


class _StubResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_rest_start_run_posts(monkeypatch):
    """When MTG_BOT_RUN_DB_URL is set, start_run POSTs to the master
    endpoint instead of touching SQLite."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://master:5099/")

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        captured["method"] = req.get_method()
        return _StubResponse(b'{"run_id": "r1"}')

    with patch("urllib.request.urlopen", fake_urlopen):
        client.start_run(
            "r1", kind="eval",
            config_json="{}", git_commit="abc",
            label="lbl",
        )

    assert captured["method"] == "POST"
    assert captured["url"] == "http://master:5099/run/start"
    assert captured["data"]["run_id"] == "r1"
    assert captured["data"]["kind"] == "eval"
    assert captured["data"]["label"] == "lbl"


def test_rest_push_progress_posts(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://master:5099")

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return _StubResponse(b"{}")

    with patch("urllib.request.urlopen", fake_urlopen):
        client.push_progress(
            "r1", shard_label="desktop",
            n_games_done=5, elapsed_seconds=1.0,
            extra={"foo": "bar"},
        )

    assert captured["url"] == "http://master:5099/run/r1/progress"
    assert captured["data"]["shard_label"] == "desktop"
    assert captured["data"]["n_games_done"] == 5
    assert json.loads(captured["data"]["extra_json"]) == {"foo": "bar"}


def test_rest_push_event_posts(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://master:5099")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return _StubResponse(b"{}")

    with patch("urllib.request.urlopen", fake_urlopen):
        client.push_event(
            "r1", "promoted",
            payload={"version": "v17"},
        )

    assert captured["url"] == "http://master:5099/run/r1/event"
    assert captured["data"]["event_type"] == "promoted"


def test_rest_push_artifact_posts(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://master:5099")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return _StubResponse(b"{}")

    with patch("urllib.request.urlopen", fake_urlopen):
        client.push_artifact(
            "r1", artifact_type="progress_json",
            path="x.json", bytes_=42,
        )

    assert captured["url"] == "http://master:5099/run/r1/artifact"
    assert captured["data"]["artifact_type"] == "progress_json"


def test_rest_finish_run_posts(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://master:5099")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        return _StubResponse(b"{}")

    with patch("urllib.request.urlopen", fake_urlopen):
        client.finish_run("r1", "failed", notes="boom")

    assert captured["url"] == "http://master:5099/run/r1/finish"
    assert captured["data"]["status"] == "failed"
    assert captured["data"]["notes"] == "boom"


def test_rest_swallows_network_errors(monkeypatch, capsys):
    """If the REST POST raises (master down, timeout, ...), the
    client warns but does not propagate. Phase 1 invariant."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://nosuchhost:9999")

    import urllib.error

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("nope")

    with patch("urllib.request.urlopen", fake_urlopen):
        client.start_run(
            "r1", kind="eval",
            config_json="{}", git_commit="abc",
        )
        client.push_progress("r1", n_games_done=1)
        client.push_event("r1", "started")
        client.push_artifact(
            "r1", artifact_type="log", path="x.log",
        )
        client.finish_run("r1", "completed")

    err = capsys.readouterr().err
    # All five calls should have produced warnings.
    assert err.count("WARNING") == 5


def test_url_strips_trailing_slash(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://master:5099/")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return _StubResponse(b"{}")

    with patch("urllib.request.urlopen", fake_urlopen):
        client.push_event("r1", "x")

    # No double-slash.
    assert captured["url"] == "http://master:5099/run/r1/event"
