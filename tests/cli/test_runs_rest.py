"""Tests for the run-database REST endpoints in
``mtg_bot.cli.serve.create_app``.

The endpoints are the network mirror of ``mtg_bot.cli.runs.client``'s
direct-SQLite path: SSH-launched remotes can't open the master's
SQLite file, so they POST to these endpoints which serialize through
the same module-level write lock.

Cover:
- Each endpoint's happy path (start / progress / event / finish /
  artifact / get / list).
- Validation: missing fields, bad enum values, non-existent run_id.
- ``runs_db_path=None`` -> 404 disabled response.
- Concurrent writes from multiple threads through the single write
  lock are serialized (no SQLite-level lock errors).
- The end-to-end flow exposed via the
  :mod:`mtg_bot.cli.runs.client` REST transport pointed at this
  Flask test client round-trips correctly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from mtg_bot.cli import serve as serve_mod
from mtg_bot.cli.runs import client as runs_client, db as runs_db


def _make_app(
    *, runs_db_path: Path | None,
) -> "serve_mod.Flask":
    """Build a minimal Flask app with the run-DB endpoints wired."""
    return serve_mod.create_app(
        repo_url="https://example.test/mtg-bot.git",
        master_pubkey="ssh-ed25519 KEY",
        master_tailnet_name="myhost",
        port=5099,
        runs_db_path=runs_db_path,
    )


# ---- /run/start ------------------------------------------------------------


def test_run_start_creates_row(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    res = client.post(
        "/run/start",
        json={
            "run_id": "r1",
            "kind": "eval",
            "config_json": json.dumps({"games": 100}),
            "git_commit": "abc1234",
            "label": "lbl",
        },
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["run_id"] == "r1"

    conn = runs_db.connect(db_path)
    try:
        record = runs_db.get_run(conn, "r1")
    finally:
        conn.close()
    assert record is not None
    assert record["run"]["status"] == "running"
    assert record["run"]["label"] == "lbl"


def test_run_start_missing_field_400(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    res = client.post(
        "/run/start",
        json={"run_id": "r1"},  # missing kind, config_json, git_commit
    )
    assert res.status_code == 400
    assert "missing" in res.get_json()["error"]


def test_run_start_invalid_kind_400(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    res = client.post(
        "/run/start",
        json={
            "run_id": "r1",
            "kind": "not_a_real_kind",
            "config_json": "{}",
            "git_commit": "abc",
        },
    )
    assert res.status_code == 400


def test_run_start_duplicate_400(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    payload = {
        "run_id": "r1",
        "kind": "eval",
        "config_json": "{}",
        "git_commit": "abc",
    }
    assert client.post("/run/start", json=payload).status_code == 200
    res = client.post("/run/start", json=payload)
    assert res.status_code == 400


# ---- /run/<id>/progress ----------------------------------------------------


def _seed_run(client, run_id: str = "r1", kind: str = "eval") -> None:
    res = client.post(
        "/run/start",
        json={
            "run_id": run_id, "kind": kind,
            "config_json": "{}", "git_commit": "abc",
        },
    )
    assert res.status_code == 200


def test_run_progress_aggregate_then_per_shard(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)

    res = client.post(
        "/run/r1/progress",
        json={"n_games_done": 5, "n_games_total": 100,
              "elapsed_seconds": 1.0},
    )
    assert res.status_code == 200
    res = client.post(
        "/run/r1/progress",
        json={"shard_label": "desktop", "n_games_done": 3,
              "elapsed_seconds": 0.7},
    )
    assert res.status_code == 200
    res = client.post(
        "/run/r1/progress",
        json={"shard_label": "desktop", "n_games_done": 10,
              "elapsed_seconds": 2.0},
    )
    assert res.status_code == 200

    res = client.get("/run/r1")
    assert res.status_code == 200
    body = res.get_json()
    progress = body["progress"]
    # Aggregate (shard_label NULL) + 1 per-shard = 2 rows.
    assert len(progress) == 2
    by_label = {p["shard_label"]: p for p in progress}
    assert by_label[None]["n_games_done"] == 5
    assert by_label["desktop"]["n_games_done"] == 10


def test_run_progress_extra_json_passthrough(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client, kind="bench")
    extra = json.dumps({"win_rate": 0.51})
    res = client.post(
        "/run/r1/progress",
        json={"extra_json": extra},
    )
    assert res.status_code == 200
    res = client.get("/run/r1")
    assert res.get_json()["progress"][0]["extra_json"] == extra


# ---- /run/<id>/event -------------------------------------------------------


def test_run_event_appends(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    for et in ("started", "stage_change", "finished_subphase"):
        res = client.post(
            f"/run/r1/event",
            json={"event_type": et, "payload_json": "{}"},
        )
        assert res.status_code == 200
    res = client.get("/run/r1")
    body = res.get_json()
    assert len(body["events"]) == 3
    assert {e["event_type"] for e in body["events"]} == {
        "started", "stage_change", "finished_subphase"
    }


def test_run_event_missing_type_400(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    res = client.post("/run/r1/event", json={})
    assert res.status_code == 400


# ---- /run/<id>/finish ------------------------------------------------------


def test_run_finish_marks_terminal(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    res = client.post(
        "/run/r1/finish",
        json={"status": "completed", "notes": "all good"},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "completed"
    rec = client.get("/run/r1").get_json()
    assert rec["run"]["status"] == "completed"
    assert rec["run"]["notes"] == "all good"
    assert rec["run"]["ended_at"] is not None


def test_run_finish_rejects_running(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    res = client.post(
        "/run/r1/finish",
        json={"status": "running"},
    )
    assert res.status_code == 400


def test_run_finish_rejects_unknown_status(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    res = client.post(
        "/run/r1/finish",
        json={"status": "completed_with_warnings"},
    )
    assert res.status_code == 400


# ---- /run/<id>/artifact ----------------------------------------------------


def test_run_artifact_inserts_then_idempotent_update(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    payload = {
        "artifact_type": "progress_json",
        "path": "training_progress/r1.json",
        "bytes": 1024,
    }
    assert client.post(
        "/run/r1/artifact", json=payload,
    ).status_code == 200
    payload["bytes"] = 2048
    assert client.post(
        "/run/r1/artifact", json=payload,
    ).status_code == 200
    rec = client.get("/run/r1").get_json()
    # UPSERT keyed on (run_id, artifact_type, path) -> single row.
    assert len(rec["artifacts"]) == 1
    assert rec["artifacts"][0]["bytes"] == 2048


def test_run_artifact_missing_fields_400(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    res = client.post("/run/r1/artifact", json={})
    assert res.status_code == 400


# ---- GET /run/<id> ---------------------------------------------------------


def test_get_run_returns_full_record(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client)
    client.post(
        "/run/r1/progress",
        json={"n_games_done": 1, "elapsed_seconds": 0.1},
    )
    client.post(
        "/run/r1/event",
        json={"event_type": "started"},
    )
    client.post(
        "/run/r1/artifact",
        json={"artifact_type": "log", "path": "x.log"},
    )
    res = client.get("/run/r1")
    assert res.status_code == 200
    body = res.get_json()
    assert body["run"]["run_id"] == "r1"
    assert len(body["progress"]) == 1
    assert len(body["events"]) == 1
    assert len(body["artifacts"]) == 1


def test_get_run_404_when_missing(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    res = client.get("/run/no-such")
    assert res.status_code == 404


# ---- GET /runs -------------------------------------------------------------


def test_list_runs_filters_and_limits(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    _seed_run(client, run_id="ev1", kind="eval")
    _seed_run(client, run_id="ev2", kind="eval")
    _seed_run(client, run_id="bn1", kind="bench")
    client.post("/run/ev1/finish", json={"status": "completed"})

    body = client.get("/runs").get_json()
    assert {r["run_id"] for r in body["runs"]} == {"ev1", "ev2", "bn1"}

    body = client.get("/runs?kind=eval").get_json()
    assert {r["run_id"] for r in body["runs"]} == {"ev1", "ev2"}

    body = client.get("/runs?status=running").get_json()
    assert {r["run_id"] for r in body["runs"]} == {"ev2", "bn1"}

    body = client.get("/runs?limit=1").get_json()
    assert len(body["runs"]) == 1


def test_list_runs_invalid_limit_400(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    client = app.test_client()
    res = client.get("/runs?limit=not-an-int")
    assert res.status_code == 400


# ---- runs_db_path=None disables endpoints ----------------------------------


@pytest.mark.parametrize("method,path,kwargs", [
    ("post", "/run/start", {"json": {}}),
    ("post", "/run/r1/progress", {"json": {}}),
    ("post", "/run/r1/event", {"json": {"event_type": "x"}}),
    ("post", "/run/r1/finish", {"json": {"status": "completed"}}),
    ("post", "/run/r1/artifact",
     {"json": {"artifact_type": "log", "path": "x"}}),
    ("get", "/run/r1", {}),
    ("get", "/runs", {}),
])
def test_runs_db_disabled_returns_404(method, path, kwargs):
    app = _make_app(runs_db_path=None)
    client = app.test_client()
    fn = getattr(client, method)
    res = fn(path, **kwargs)
    assert res.status_code == 404
    assert "not configured" in res.get_json()["error"]


# ---- concurrent writes serialize -------------------------------------------


def test_concurrent_event_pushes_serialize(tmp_path: Path):
    """Multiple threads POSTing to /run/<id>/event concurrently should
    all succeed (the WRITE_LOCK serializes them). Mirrors the
    runs/db.py:test_concurrent_progress_pushes_serialize coverage but
    via the network layer."""
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    test_client = app.test_client()
    _seed_run(test_client)

    n_threads = 8
    per_thread = 5
    errors: list[str] = []

    def push_some(thread_idx: int):
        # Each thread builds its own test client so concurrent
        # request handling actually exercises threading.
        c = app.test_client()
        for j in range(per_thread):
            res = c.post(
                "/run/r1/event",
                json={
                    "event_type": "tick",
                    "payload_json": json.dumps(
                        {"t": thread_idx, "j": j},
                    ),
                },
            )
            if res.status_code != 200:
                errors.append(f"{thread_idx}/{j}: {res.status_code}")

    threads = [
        threading.Thread(target=push_some, args=(i,))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], errors

    rec = test_client.get("/run/r1").get_json()
    # We may not see all 40 events in the latest-100 window, but the
    # row count should match: query the DB directly.
    conn = runs_db.connect(db_path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = 'r1'",
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == n_threads * per_thread


# ---- end-to-end via runs.client REST transport -----------------------------


def test_client_rest_round_trip_through_test_client(
    tmp_path: Path, monkeypatch,
):
    """Wire ``runs.client`` to POST through the Flask test_client (via
    a stubbed ``urlopen``) and confirm a full lifecycle round-trips
    correctly. Pinning that the JSON shape expected by the server
    matches what the client sends."""
    db_path = tmp_path / "runs.db"
    app = _make_app(runs_db_path=db_path)
    test_client = app.test_client()

    monkeypatch.delenv("MTG_BOT_RUN_DB_PATH", raising=False)
    monkeypatch.setenv("MTG_BOT_RUN_DB_URL", "http://master:5099")

    class _StubResp:
        def __init__(self, body: bytes, status: int = 200):
            self._body, self.status = body, status

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        # Strip the "http://master:5099" base; route to test_client.
        path = req.full_url.replace("http://master:5099", "")
        body_bytes = req.data
        body = json.loads(body_bytes.decode("utf-8")) if body_bytes else None
        if req.get_method() == "POST":
            res = test_client.post(path, json=body)
        else:
            res = test_client.get(path)
        if res.status_code >= 400:
            import urllib.error
            raise urllib.error.HTTPError(
                req.full_url, res.status_code,
                "err", {}, None,
            )
        return _StubResp(res.get_data())

    with patch("urllib.request.urlopen", fake_urlopen):
        runs_client.start_run(
            "rxx", kind="eval",
            config_json="{}", git_commit="abcd",
            label="round-trip",
        )
        runs_client.push_progress(
            "rxx", n_games_done=10, elapsed_seconds=2.0,
        )
        runs_client.push_event("rxx", "started")
        runs_client.push_artifact(
            "rxx", artifact_type="log", path="x.log",
        )
        runs_client.finish_run("rxx", "completed", notes="done")

    res = test_client.get("/run/rxx")
    body = res.get_json()
    assert body["run"]["status"] == "completed"
    assert body["run"]["label"] == "round-trip"
    assert body["run"]["notes"] == "done"
    assert len(body["progress"]) == 1
    assert len(body["events"]) == 1
    assert len(body["artifacts"]) == 1
