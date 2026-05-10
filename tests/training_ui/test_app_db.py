"""Phase 2 DB-backed read path for ``mtg_bot.training_ui.app``.

These tests cover the new code paths added in
``feature/ui-from-db``:

- ``/api/runs`` reads the SQLite ``runs.db`` and merges with
  standalone JSON files (the JSON-only fallback for runs that
  haven't been wired to the DB yet — currently iterate / train /
  bench / capture).
- ``/api/runs/<run_id>`` returns the merged DB+JSON record plus the
  per-shard progress rows / events / artifacts pulled from the DB.
- ``/api/runs/recent`` surfaces the last N completed runs across
  all kinds for the cross-run summary panel.
- A DB row's ``kind = 'self_play'`` is translated to the UI's
  ``'data_gen'`` discriminator on the read path so the front-end
  renderer doesn't need to know about the DB enum.
- A JSON file with no DB row gets a synthetic
  ``_synthetic_from_json: true`` marker so backfill tooling and
  tests can tell apart real vs synthesized rows.
- A DB row with no JSON file still surfaces (covers Phase 3
  approach: JSON writes drop, DB becomes sole source).

The existing JSON-fixture tests in ``test_app.py`` exercise the
JSON-only fallback path with no DB present — they still pass under
the new merging logic.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mtg_bot.cli.runs import db as runs_db
from mtg_bot.cli.runs.models import (
    Artifact, Event, ProgressUpdate, Run,
)
from mtg_bot.training_ui.app import create_app


def _open_db(progress_dir: Path) -> sqlite3.Connection:
    """Create + open the colocated runs.db, applying schema. The UI
    queries this same path on every request via
    ``runs_db.connect``."""
    return runs_db.connect(progress_dir / "runs.db")


def _seed_run(
    conn: sqlite3.Connection, run_id: str, *,
    kind: str = "eval",
    status: str = "running",
    started_at: str = "2026-05-04T10:00:00+00:00",
    ended_at: str | None = None,
    git_commit: str = "deadbeef",
    config_json: str = "{}",
    parent_run_id: str | None = None,
    label: str | None = None,
    notes: str | None = None,
) -> None:
    runs_db.insert_run(
        conn,
        Run(
            run_id=run_id, kind=kind, status=status,
            started_at=started_at, ended_at=ended_at,
            git_commit=git_commit, config_json=config_json,
            parent_run_id=parent_run_id, label=label, notes=notes,
        ),
    )


def _seed_progress(
    conn: sqlite3.Connection, run_id: str, *,
    shard_label: str | None = None,
    n_games_done: int | None = None,
    n_games_total: int | None = None,
    elapsed_seconds: float | None = None,
    crashed: int | None = None,
    timed_out: int | None = None,
    wins_new: int | None = None,
    wins_old: int | None = None,
    draws: int | None = None,
    extra: dict | None = None,
) -> None:
    extra_json = json.dumps(extra, sort_keys=True) if extra else None
    runs_db.upsert_progress(
        conn,
        ProgressUpdate(
            run_id=run_id, shard_label=shard_label,
            n_games_done=n_games_done, n_games_total=n_games_total,
            elapsed_seconds=elapsed_seconds,
            crashed=crashed, timed_out=timed_out,
            wins_new=wins_new, wins_old=wins_old, draws=draws,
            extra_json=extra_json,
        ),
    )


def _seed_event(
    conn: sqlite3.Connection, run_id: str, event_type: str,
    *, payload: dict | None = None,
    timestamp: str = "2026-05-04T10:01:00+00:00",
) -> None:
    runs_db.insert_event(
        conn,
        Event(
            run_id=run_id, event_type=event_type,
            payload_json=json.dumps(payload, sort_keys=True) if payload else None,
            timestamp=timestamp,
        ),
    )


def _seed_artifact(
    conn: sqlite3.Connection, run_id: str, *,
    artifact_type: str = "progress_json",
    path: str = "training_progress/eval_v9.json",
    bytes_: int | None = 1024,
) -> None:
    runs_db.upsert_artifact(
        conn,
        Artifact(
            run_id=run_id, artifact_type=artifact_type,
            path=path, bytes=bytes_,
        ),
    )


def _write_json(progress_dir: Path, run_id: str, **fields) -> Path:
    payload = {
        "run_id": run_id,
        "started_at": "2026-05-04T10:00:00+00:00",
        "status": "running",
    }
    payload.update(fields)
    path = progress_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def progress_dir(tmp_path: Path) -> Path:
    d = tmp_path / "training_progress"
    d.mkdir()
    return d


@pytest.fixture()
def client(progress_dir: Path):
    app = create_app(progress_dir)
    app.config["TESTING"] = True
    return app.test_client()


# ---- /api/runs : DB + JSON merging ----------------------------------------


def test_api_runs_reads_run_row_from_db_when_no_json(
    client, progress_dir: Path,
):
    """A run that exists ONLY in the DB (no JSON file) shows up in
    /api/runs. This is the Phase 3 endgame — at that point JSON
    writes are gone and the UI must surface the DB row directly."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "eval_v9_vs_v8", kind="eval", status="running",
            started_at="2026-05-04T10:00:00+00:00",
        )
        _seed_progress(
            conn, "eval_v9_vs_v8",
            n_games_done=42, n_games_total=200,
            elapsed_seconds=600.0,
            wins_new=20, wins_old=18, draws=4, crashed=0, timed_out=0,
            extra={"win_rate": 0.526, "opponent_label": "v8"},
        )
    finally:
        conn.close()
    res = client.get("/api/runs")
    assert res.status_code == 200
    runs = res.get_json()["runs"]
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == "eval_v9_vs_v8"
    assert r["kind"] == "eval"
    assert r["status"] == "running"
    assert r["n_games_done"] == 42
    assert r["n_games_total"] == 200
    assert r["wins_new"] == 20
    # extra_json fan-out: opponent_label + win_rate available at the
    # top level so the UI's eval renderer reads them naturally.
    assert r["opponent_label"] == "v8"
    assert r["win_rate"] == pytest.approx(0.526)
    assert r["_source"] == "db"
    assert r["_db_kind"] == "eval"


def test_api_runs_translates_self_play_kind_to_data_gen(
    client, progress_dir: Path,
):
    """The DB's ``kind = 'self_play'`` is translated to the UI's
    ``'data_gen'`` so the front-end discriminator (which still
    matches on the JSON-era ``data_gen`` literal) lights up the
    blue games-progress card. ``_db_kind`` retains the canonical
    DB value for downstream tooling."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "iter_001_self_play_v6", kind="self_play",
            started_at="2026-05-04T11:00:00+00:00",
        )
    finally:
        conn.close()
    runs = client.get("/api/runs").get_json()["runs"]
    assert len(runs) == 1
    assert runs[0]["kind"] == "data_gen"
    assert runs[0]["_db_kind"] == "self_play"


def test_api_runs_synthesizes_db_row_for_json_only(
    client, progress_dir: Path,
):
    """The common Phase-2 case today: most subcommands are NOT yet
    wired into the DB so they only have a JSON file. The UI must
    still render them; we synthesize a DB-shape row from the JSON
    payload and stamp ``_synthetic_from_json: true`` so a future
    backfill knows the row was reconstructed."""
    _write_json(
        progress_dir, "iter_002_train_v7",
        kind="training", status="running",
        total_epochs=30, current_epoch=10,
        train_losses=[0.5, 0.4, 0.3], val_losses=[0.55, 0.45, 0.35],
        started_at="2026-05-04T12:00:00+00:00",
    )
    runs = client.get("/api/runs").get_json()["runs"]
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == "iter_002_train_v7"
    assert r["_synthetic_from_json"] is True
    assert r["_source"] == "json"
    # extra_json on a synthesized row carries the same marker so a
    # backfill that mirrors run_progress.extra_json can detect it.
    assert json.loads(r["extra_json"])["_synthetic_from_json"] is True
    # JSON-shape fields remain at the top level for the renderer.
    assert r["kind"] == "training"
    assert r["current_epoch"] == 10
    assert r["train_losses"] == [0.5, 0.4, 0.3]


def test_api_runs_merges_db_and_json_when_both_exist(
    client, progress_dir: Path,
):
    """When BOTH the DB and a JSON file exist for the same run_id
    (the eval case today), merge: JSON wins for the rich UI fields
    (machines, win_rate, ...) but DB wins for status / ended_at when
    JSON is missing them. The merged row is stamped ``_source:
    "merged"``."""
    _write_json(
        progress_dir, "eval_v9_vs_v8",
        kind="eval", status="running",
        n_games_done=100, n_games_total=200,
        wins_new=55, wins_old=42, draws=3,
        opponent_label="v8",
        machines=[
            {"label": "macbook", "n_games_done": 60, "n_workers": 14},
            {"label": "desktop", "n_games_done": 40, "n_workers": 16},
        ],
    )
    conn = _open_db(progress_dir)
    try:
        # DB lags slightly: progress hasn't been pushed but the run
        # row exists.
        _seed_run(
            conn, "eval_v9_vs_v8", kind="eval", status="running",
            git_commit="dbsha01",
            label="v9_vs_v8",
        )
    finally:
        conn.close()
    runs = client.get("/api/runs").get_json()["runs"]
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == "eval_v9_vs_v8"
    assert r["_source"] == "merged"
    # JSON-side fields preserved.
    assert r["machines"][0]["label"] == "macbook"
    assert r["wins_new"] == 55
    # DB-side fields fill in (label was JSON-absent).
    assert r["label"] == "v9_vs_v8"
    assert r["git_commit"] == "dbsha01"


def test_api_runs_sorts_by_started_at_desc_across_sources(
    client, progress_dir: Path,
):
    """A mixed pile of DB-only + JSON-only + merged runs sorts as
    one stream by ``started_at`` desc — same invariant the JSON-only
    listing held."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "db_only_old", kind="eval",
            started_at="2026-05-01T08:00:00+00:00",
        )
        _seed_run(
            conn, "merged_mid", kind="eval",
            started_at="2026-05-02T08:00:00+00:00",
        )
    finally:
        conn.close()
    _write_json(
        progress_dir, "merged_mid",
        kind="eval", status="running",
        started_at="2026-05-02T08:00:00+00:00",
    )
    _write_json(
        progress_dir, "json_only_new",
        kind="training", status="running",
        started_at="2026-05-03T08:00:00+00:00",
    )
    runs = client.get("/api/runs").get_json()["runs"]
    assert [r["run_id"] for r in runs] == [
        "json_only_new", "merged_mid", "db_only_old",
    ]


def test_api_runs_skips_per_shard_json_files(
    client, progress_dir: Path,
):
    """Per-shard JSON files (``*_shard_*.json``) are still skipped
    even after the Phase 2 rewrite — the aggregate row in the DB or
    the rolled-up JSON is what surfaces, not N per-shard cards."""
    _write_json(
        progress_dir, "iter_001_self_play_v6",
        kind="data_gen", status="running",
        started_at="2026-05-04T10:00:00+00:00",
    )
    _write_json(
        progress_dir, "iter_001_self_play_v6_shard_local",
        kind="data_gen", status="running",
        started_at="2026-05-04T10:00:00+00:00",
    )
    _write_json(
        progress_dir, "iter_001_self_play_v6_shard_macbook",
        kind="data_gen", status="running",
        started_at="2026-05-04T10:00:00+00:00",
    )
    runs = client.get("/api/runs").get_json()["runs"]
    assert [r["run_id"] for r in runs] == ["iter_001_self_play_v6"]


def test_api_runs_no_db_file_falls_back_to_json_only(
    client, progress_dir: Path,
):
    """If runs.db doesn't exist (a fresh checkout that hasn't run
    the dual-write path yet), the UI falls through to JSON-only
    listing. Every JSON-only run is still synthesized."""
    _write_json(
        progress_dir, "v1",
        started_at="2026-05-01T10:00:00+00:00",
        status="completed",
    )
    runs = client.get("/api/runs").get_json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "v1"
    assert runs[0]["_synthetic_from_json"] is True
    # And the runs.db file was NOT created by the read path — the UI
    # is read-only with respect to the DB.
    assert not (progress_dir / "runs.db").is_file()


def test_api_runs_corrupt_db_does_not_crash_ui(
    client, progress_dir: Path,
):
    """Defensive: a half-written / unreadable runs.db must not 500
    the UI. We fall back to JSON-only listing instead."""
    (progress_dir / "runs.db").write_bytes(b"this is not a sqlite database")
    _write_json(
        progress_dir, "v1",
        started_at="2026-05-01T10:00:00+00:00",
        status="completed",
    )
    res = client.get("/api/runs")
    assert res.status_code == 200
    runs = res.get_json()["runs"]
    assert [r["run_id"] for r in runs] == ["v1"]


# ---- /api/runs/<run_id> : per-run detail ----------------------------------


def test_api_run_detail_surfaces_db_events_and_artifacts(
    client, progress_dir: Path,
):
    """Click-through on a run card hits this endpoint. It must
    surface the per-shard progress rows, last 100 events, and
    registered artifacts from the DB so the detail panel can show
    them without an extra round-trip."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "eval_v9", kind="eval", status="running",
            started_at="2026-05-04T10:00:00+00:00",
        )
        _seed_progress(
            conn, "eval_v9", n_games_done=50, n_games_total=200,
        )
        _seed_progress(
            conn, "eval_v9", shard_label="macbook",
            n_games_done=30, n_games_total=100,
        )
        _seed_progress(
            conn, "eval_v9", shard_label="desktop",
            n_games_done=20, n_games_total=100,
        )
        _seed_event(
            conn, "eval_v9", "stage_change",
            payload={"stage": "self_play"},
            timestamp="2026-05-04T10:01:00+00:00",
        )
        _seed_event(
            conn, "eval_v9", "promoted",
            payload={"version": "v9"},
            timestamp="2026-05-04T10:02:00+00:00",
        )
        _seed_artifact(
            conn, "eval_v9",
            artifact_type="progress_json",
            path="training_progress/eval_v9.json",
            bytes_=2048,
        )
    finally:
        conn.close()
    res = client.get("/api/runs/eval_v9")
    assert res.status_code == 200
    body = res.get_json()
    assert body["run_id"] == "eval_v9"
    # Aggregate counters surfaced from the (shard_label IS NULL) row.
    assert body["n_games_done"] == 50
    # Per-shard rows in their own bucket.
    per_shard = body["_db_progress_per_shard"]
    labels = sorted(p["shard_label"] for p in per_shard)
    assert labels == ["desktop", "macbook"]
    # Events: latest first by id (insert_event uses autoincrement, so
    # the SECOND event seeded comes back first).
    events = body["_db_events"]
    assert len(events) == 2
    assert events[0]["event_type"] == "promoted"
    assert events[1]["event_type"] == "stage_change"
    # Artifacts.
    arts = body["_db_artifacts"]
    assert len(arts) == 1
    assert arts[0]["artifact_type"] == "progress_json"
    assert arts[0]["bytes"] == 2048


def test_api_run_detail_caps_events_at_100(
    client, progress_dir: Path,
):
    """The DB's get_run() helper already caps events at 100; assert
    the UI passes that cap through unchanged."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(conn, "noisy_run", kind="eval")
        for i in range(150):
            _seed_event(
                conn, "noisy_run", f"event_{i}",
                timestamp=f"2026-05-04T10:{(i // 60) :02d}:{(i % 60) :02d}+00:00",
            )
    finally:
        conn.close()
    body = client.get("/api/runs/noisy_run").get_json()
    assert len(body["_db_events"]) == 100


def test_api_run_detail_synthesizes_for_json_only_run(
    client, progress_dir: Path,
):
    """A click-through on a JSON-only run still resolves; the
    detail surfaces the JSON's rich fields with empty DB-side
    arrays."""
    _write_json(
        progress_dir, "iter_002_train_v7",
        kind="training", status="completed",
        total_epochs=30, current_epoch=30,
    )
    body = client.get("/api/runs/iter_002_train_v7").get_json()
    assert body["run_id"] == "iter_002_train_v7"
    assert body["_synthetic_from_json"] is True
    assert body["_db_events"] == []
    assert body["_db_artifacts"] == []
    assert body["_db_progress_per_shard"] == []


def test_api_run_detail_404_for_unknown(client):
    res = client.get("/api/runs/no_such_run")
    assert res.status_code == 404


# ---- /api/runs/recent : cross-run aggregations ----------------------------


def test_api_runs_recent_returns_completed_runs_with_summary_stats(
    client, progress_dir: Path,
):
    """The cross-run summary panel asks for "last 20 completed runs
    across all kinds with summary stats". We surface kind / counters
    / wins+draws so the panel can tabulate without per-run details
    fetches."""
    conn = _open_db(progress_dir)
    try:
        # Three runs: 2 completed, 1 running. Recent endpoint returns
        # only the completed pair.
        _seed_run(
            conn, "eval_done_a", kind="eval", status="completed",
            started_at="2026-05-04T10:00:00+00:00",
            ended_at="2026-05-04T11:00:00+00:00",
        )
        _seed_progress(
            conn, "eval_done_a",
            n_games_done=200, n_games_total=200,
            elapsed_seconds=3600.0,
            wins_new=110, wins_old=80, draws=10,
            crashed=0, timed_out=0,
        )
        _seed_run(
            conn, "self_play_done", kind="self_play", status="completed",
            started_at="2026-05-03T10:00:00+00:00",
            ended_at="2026-05-03T13:00:00+00:00",
        )
        _seed_progress(
            conn, "self_play_done",
            n_games_done=500, n_games_total=500,
            elapsed_seconds=10800.0,
            crashed=2, timed_out=4,
        )
        _seed_run(
            conn, "still_running", kind="eval", status="running",
            started_at="2026-05-04T12:00:00+00:00",
        )
    finally:
        conn.close()
    body = client.get("/api/runs/recent").get_json()
    assert body["limit"] == 20
    runs = body["runs"]
    assert len(runs) == 2
    # Newest-first.
    assert [r["run_id"] for r in runs] == [
        "eval_done_a", "self_play_done",
    ]
    # Summary stats from the aggregate progress row.
    a = runs[0]
    assert a["kind"] == "eval"
    assert a["wins_new"] == 110
    assert a["wins_old"] == 80
    assert a["draws"] == 10
    sp = runs[1]
    # self_play translates to data_gen for the UI.
    assert sp["kind"] == "data_gen"
    assert sp["crashed"] == 2


def test_api_runs_recent_respects_limit(client, progress_dir: Path):
    conn = _open_db(progress_dir)
    try:
        for i in range(25):
            _seed_run(
                conn, f"run_{i:02d}", kind="eval", status="completed",
                started_at=f"2026-05-04T10:{i:02d}:00+00:00",
            )
    finally:
        conn.close()
    body = client.get("/api/runs/recent?limit=5").get_json()
    assert len(body["runs"]) == 5
    assert body["limit"] == 5


def test_api_runs_recent_clamps_bad_limit(client, progress_dir: Path):
    """A bogus / out-of-range limit clamps to the default 20 (or
    the [1, 200] range) rather than 500'ing."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "r1", kind="eval", status="completed",
            started_at="2026-05-04T10:00:00+00:00",
        )
    finally:
        conn.close()
    body = client.get("/api/runs/recent?limit=not_a_number").get_json()
    assert body["limit"] == 20
    body = client.get("/api/runs/recent?limit=99999").get_json()
    assert body["limit"] == 200
    body = client.get("/api/runs/recent?limit=-1").get_json()
    # Negative clamps up to 1.
    assert body["limit"] == 1


def test_api_runs_recent_with_no_db_file_returns_empty(client):
    """Recent is explicitly DB-driven; with no DB file it returns an
    empty list rather than walking JSON (the JSON-only fallback is
    a /api/runs concern, not /api/runs/recent)."""
    body = client.get("/api/runs/recent").get_json()
    assert body["runs"] == []


# ---- Stop endpoint over the merged read path ------------------------------


def test_stop_endpoint_finds_db_only_run(client, progress_dir: Path):
    """Even without a JSON file, a DB-known run_id can be stopped.
    The sentinel still lands in progress_dir."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "eval_db_only", kind="eval", status="running",
            started_at="2026-05-04T10:00:00+00:00",
        )
    finally:
        conn.close()
    res = client.post("/api/runs/eval_db_only/stop", json={"reason": "test"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["stopped_target"] == "eval_db_only"
    assert (progress_dir / ".STOP_eval_db_only").is_file()


def test_stop_endpoint_uses_db_parent_run_id_as_cycle(
    client, progress_dir: Path,
):
    """When JSON has no ``cycle_id`` but the DB has a
    ``parent_run_id`` set, the stop sentinel targets the parent so
    a stop on a child cancels the whole parent (mirrors the JSON
    cycle_id behavior)."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "iter_001_cycle", kind="iterate", status="running",
            started_at="2026-05-04T09:00:00+00:00",
        )
        _seed_run(
            conn, "iter_001_self_play_v6", kind="self_play",
            status="running", parent_run_id="iter_001_cycle",
            started_at="2026-05-04T10:00:00+00:00",
        )
    finally:
        conn.close()
    res = client.post("/api/runs/iter_001_self_play_v6/stop")
    body = res.get_json()
    assert body["stopped_target"] == "iter_001_cycle"
    assert body["is_cycle_stop"] is True
    assert (progress_dir / ".STOP_iter_001_cycle").is_file()
    assert not (progress_dir / ".STOP_iter_001_self_play_v6").is_file()


def test_stop_status_endpoint_works_for_db_only_run(
    client, progress_dir: Path,
):
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "eval_db_only", kind="eval", status="running",
            started_at="2026-05-04T10:00:00+00:00",
        )
    finally:
        conn.close()
    body = client.get("/api/runs/eval_db_only/stop").get_json()
    assert body["stop_requested"] is False
    client.post("/api/runs/eval_db_only/stop")
    body = client.get("/api/runs/eval_db_only/stop").get_json()
    assert body["stop_requested"] is True
    assert body["own_sentinel"] is True


# ---- /api/runs/<id>/replays over the merged read path ---------------------


def test_replays_endpoint_resolves_for_db_only_run(
    client, progress_dir: Path, tmp_path: Path,
):
    """Replays_dir is a JSON-only field today; a DB-only run with no
    JSON correctly returns a "this run did not capture replays"
    response rather than 404. (The endpoint just falls through the
    unset replays_dir branch.)"""
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "iter_001_self_play_v6", kind="self_play",
            status="running",
            started_at="2026-05-04T10:00:00+00:00",
        )
    finally:
        conn.close()
    res = client.get("/api/runs/iter_001_self_play_v6/replays")
    assert res.status_code == 200
    body = res.get_json()
    assert body["count"] == 0
    assert "did not capture" in body["note"]


# ---- Defensive branches ---------------------------------------------------


def test_ui_kind_from_db_handles_none():
    """Defensive: ``_ui_kind_from_db(None)`` returns None even
    though every real DB row has a NOT-NULL kind. Cheap helper, but
    a future call site (e.g. an aggregation query that selects
    optional columns) shouldn't have to check first."""
    from mtg_bot.training_ui.app import _ui_kind_from_db
    assert _ui_kind_from_db(None) is None
    assert _ui_kind_from_db("eval") == "eval"
    assert _ui_kind_from_db("self_play") == "data_gen"


def test_read_run_file_skips_non_dict_payload(
    client, progress_dir: Path,
):
    """A JSON file that parses to e.g. a list rather than a dict is
    silently skipped — the trainer always writes objects, but a
    typo'd dump shouldn't break the listing."""
    (progress_dir / "list.json").write_text(
        "[1, 2, 3]", encoding="utf-8",
    )
    runs = client.get("/api/runs").get_json()["runs"]
    assert runs == []


def test_extra_json_parse_failure_does_not_crash(
    client, progress_dir: Path,
):
    """If a writer ever stores malformed JSON in run_progress.extra_json
    (e.g. a partially-quoted blob from a buggy serializer), the UI
    must surface the run anyway, just without the fan-out fields."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(conn, "eval_bad_extra", kind="eval", status="running")
        # Bypass upsert_progress so we can write deliberately-broken
        # JSON; ProgressUpdate.extra_json is a TEXT column with no
        # CHECK constraint.
        conn.execute(
            """
            INSERT INTO run_progress(
                run_id, shard_label, n_games_done, extra_json, updated_at
            ) VALUES (?, NULL, ?, ?, ?)
            """,
            ("eval_bad_extra", 5, "{not_json", "2026-05-04T10:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    res = client.get("/api/runs")
    assert res.status_code == 200
    runs = res.get_json()["runs"]
    assert len(runs) == 1
    assert runs[0]["n_games_done"] == 5
    # No fan-out fields landed because the parse failed; the run still
    # surfaces with its core counter.
    assert "win_rate" not in runs[0]


def test_extra_json_non_dict_parse_does_not_crash(
    client, progress_dir: Path,
):
    """An extra_json that parses to a list rather than a dict (the
    fan-out branch only iterates dict items) likewise doesn't break
    the listing."""
    conn = _open_db(progress_dir)
    try:
        _seed_run(conn, "eval_list_extra", kind="eval", status="running")
        conn.execute(
            """
            INSERT INTO run_progress(
                run_id, shard_label, n_games_done, extra_json, updated_at
            ) VALUES (?, NULL, ?, ?, ?)
            """,
            ("eval_list_extra", 5, "[1, 2, 3]",
             "2026-05-04T10:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    res = client.get("/api/runs")
    assert res.status_code == 200
    runs = res.get_json()["runs"]
    assert len(runs) == 1


def test_run_detail_merges_db_and_json(client, progress_dir: Path):
    """The detail endpoint merges DB run + JSON content when both
    exist, layering the JSON-side rich fields onto the DB row's
    canonical status / git_commit / etc."""
    _write_json(
        progress_dir, "eval_merge",
        kind="eval", status="completed",
        started_at="2026-05-04T10:00:00+00:00",
        ended_at="2026-05-04T11:00:00+00:00",
        wins_new=110, wins_old=80, draws=10,
        opponent_label="v8",
        machines=[{"label": "macbook", "n_games_done": 100}],
    )
    conn = _open_db(progress_dir)
    try:
        _seed_run(
            conn, "eval_merge", kind="eval", status="completed",
            started_at="2026-05-04T10:00:00+00:00",
            ended_at="2026-05-04T11:00:00+00:00",
            git_commit="dbsha02",
            label="v9_vs_v8",
        )
        _seed_progress(
            conn, "eval_merge",
            n_games_done=200, n_games_total=200,
            wins_new=110, wins_old=80, draws=10,
        )
        _seed_event(
            conn, "eval_merge", "completed",
            payload={"wins_new": 110},
        )
    finally:
        conn.close()
    body = client.get("/api/runs/eval_merge").get_json()
    assert body["_source"] == "merged"
    # JSON-side rich fields preserved.
    assert body["machines"][0]["label"] == "macbook"
    assert body["opponent_label"] == "v8"
    # DB-side fields fill JSON gaps.
    assert body["git_commit"] == "dbsha02"
    assert body["label"] == "v9_vs_v8"
    # Events flow from the DB.
    assert len(body["_db_events"]) == 1
    assert body["_db_events"][0]["event_type"] == "completed"


def test_stop_status_endpoint_404_for_unknown_run(client):
    """GET /api/runs/<unknown>/stop returns 404 — same as the POST
    side and the detail endpoint."""
    res = client.get("/api/runs/no_such_run/stop")
    assert res.status_code == 404
