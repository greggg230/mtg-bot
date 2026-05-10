"""Flask app showing live progress of NN training runs.

Phase 2 of the runs-db rollout: this UI now reads run records from
the SQLite ``runs.db`` (Phase 1 dual-write store). The legacy
JSON-walking path is retained as a fallback so runs that haven't
been wired into the DB yet (Phase 1 only wired ``mtg_bot eval`` —
``iterate``, ``train``, ``shard``, ``capture``, ``bench`` still
write JSON only) still render. Mixed mode is the steady state until
Phase 3 drops the JSON writes entirely.

DB query shape per request:

- ``/api/runs``: ``SELECT * FROM runs ORDER BY started_at DESC``,
  joined with the aggregate (``shard_label IS NULL``) row from
  ``run_progress`` and the artifact rows from ``run_artifacts``.
  Then unioned with the JSON-only runs scanned from ``progress_dir``
  (any file whose ``run_id`` doesn't already appear in the DB result),
  whose payload is treated as the synthesized run row.
- ``/api/runs/<run_id>``: the full DB record (run row + all progress
  rows + last 100 events + artifacts) merged with the JSON file's
  contents. JSON wins for UI-render fields it provides; DB wins for
  ``events`` / ``artifacts`` / ``progress`` (rich) / ``_db_*`` meta.
- ``/api/runs/recent``: last 20 completed runs across all kinds with
  summary stats from ``run_progress``.

JSON fallback / backfill mechanism:

When ``progress_dir`` contains ``<run_id>.json`` but the DB has no
matching ``runs`` row, the JSON content is treated as the source of
truth and a synthetic in-memory row is stamped with
``extra_json={"_synthetic_from_json": true}`` so callers (and tests)
can tell the row was reconstructed. When the DB has a ``runs`` row
but the matching JSON is missing, we serve the DB row directly with
the UI-render keys filled in best-effort (``kind`` translated from
db enum to UI enum; counters surfaced from the aggregate progress
row).

Connection lifecycle: each request opens its own short-lived
``sqlite3.Connection`` via :func:`mtg_bot.cli.runs.db.connect` and
closes it before returning. WAL mode (set on connect) lets these
reads run concurrently with the writer (the cli subcommand pushing
progress) without lock contention. SQLite connections are also
cheap (~hundreds of microseconds) so we don't bother with pooling.

Write surface: ``POST /api/runs/<run_id>/stop``, which drops a
``.STOP_<run_id>`` sentinel file in the progress directory — long-
running producers (iterate cycle, self-play pool, eval pool, queue
server) check for the sentinel at stage boundaries and exit
cleanly when they see it. Stop semantics still flow through the
JSON since cycle_id back-pointers are written there; we read the
JSON for the stop endpoint when present, and fall back to the DB
row's ``parent_run_id`` if not.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from flask import Flask, abort, jsonify, render_template, request

from mtg_bot.cli.runs import db as runs_db
from mtg_bot.training.stop_signal import (
    sentinel_path, stop_requested, write_stop,
)


# ---- DB <-> UI kind translation -------------------------------------------

# The DB schema's ``kind`` enum (see ``cli.runs.db.SCHEMA_SQL``) uses
# canonical names; the UI front-end (the inline JS in index.html)
# discriminates cards on JSON-shape ``kind`` strings. The two diverge
# only on self-play: the DB calls it ``self_play``, the UI calls it
# ``data_gen`` (for historical reasons — the JSON predates the DB).
# Map DB->UI on the read path so the renderer doesn't need to know
# about the DB enum.
_DB_TO_UI_KIND = {"self_play": "data_gen"}


def _ui_kind_from_db(db_kind: Optional[str]) -> Optional[str]:
    if db_kind is None:
        return None
    return _DB_TO_UI_KIND.get(db_kind, db_kind)


# ---- JSON read helpers (legacy fallback) ----------------------------------


def _read_run_file(path: Path) -> Optional[dict[str, Any]]:
    """Read one progress JSON file. Returns ``None`` if the file is
    missing or unparseable — the trainer's atomic write should
    prevent half-files in normal operation, but the file COULD be
    deleted out from under us between listdir and read, and a future
    trainer-side bug shouldn't take down the whole UI."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    # Stamp the source filename so the UI has a stable handle even
    # when run_id is missing or duplicated.
    data.setdefault("run_id", path.stem)
    data["_filename"] = path.name
    return data


def _scan_json_dir(progress_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``{run_id: payload}`` for every parseable ``*.json``
    file in ``progress_dir`` that is NOT a per-shard file. Per-shard
    files (``*_shard_*.json``) are skipped — see the original
    ``_list_runs`` doc for why."""
    if not progress_dir.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in progress_dir.glob("*.json"):
        if "_shard_" in path.name:
            continue
        run = _read_run_file(path)
        if run is None:
            continue
        out[run["run_id"]] = run
    return out


# ---- DB read helpers ------------------------------------------------------


def _open_conn(progress_dir: Path) -> Optional[sqlite3.Connection]:
    """Open a fresh read connection to the runs DB colocated with
    ``progress_dir``. Returns ``None`` if the DB file doesn't exist
    yet (Phase 1 invariant: an unwired master with no DB still has
    a working UI via JSON fallback)."""
    db_path = progress_dir / "runs.db"
    if not db_path.is_file():
        return None
    try:
        return runs_db.connect(db_path)
    except sqlite3.Error:
        return None


def _aggregate_progress_row(
    conn: sqlite3.Connection, run_id: str,
) -> Optional[dict[str, Any]]:
    """The ``shard_label IS NULL`` aggregate row for ``run_id``, or
    ``None`` if no progress has been pushed yet."""
    row = conn.execute(
        """
        SELECT * FROM run_progress
        WHERE run_id = ? AND shard_label IS NULL
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def _scan_db(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Return ``{run_id: ui_run_dict}`` for every row in ``runs``,
    each enriched with the aggregate progress row. Newest first by
    ``started_at`` (the dict insertion order is preserved)."""
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC"
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_dict = dict(row)
        agg = _aggregate_progress_row(conn, run_dict["run_id"])
        out[run_dict["run_id"]] = _db_row_to_ui_dict(run_dict, agg)
    return out


def _db_row_to_ui_dict(
    run_row: dict[str, Any],
    progress_row: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Project a ``runs`` + aggregate ``run_progress`` pair into the
    JSON-shape dict the UI front-end expects. Missing fields are left
    absent (the JS uses ``|| 0`` / ``|| '—'`` everywhere) rather than
    stamped with placeholders."""
    ui: dict[str, Any] = {
        "run_id": run_row["run_id"],
        "kind": _ui_kind_from_db(run_row.get("kind")),
        "status": run_row.get("status"),
        "started_at": run_row.get("started_at"),
        "ended_at": run_row.get("ended_at"),
        "git_commit": run_row.get("git_commit"),
        "label": run_row.get("label"),
        "notes": run_row.get("notes"),
        "parent_run_id": run_row.get("parent_run_id"),
        "_db_kind": run_row.get("kind"),
        "_source": "db",
    }
    if progress_row is not None:
        for key in (
            "n_games_done", "n_games_total", "elapsed_seconds",
            "crashed", "timed_out", "wins_new", "wins_old", "draws",
        ):
            val = progress_row.get(key)
            if val is not None:
                ui[key] = val
        # Aggregate-row extra_json may carry win_rate, opponent_label,
        # etc. — fan it out so the UI doesn't have to dig through
        # extra_json in a separate pass.
        extra_raw = progress_row.get("extra_json")
        if extra_raw:
            try:
                extra = json.loads(extra_raw)
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        ui.setdefault(k, v)
            except (json.JSONDecodeError, TypeError):
                pass
    return ui


def _synthesize_db_row_from_json(
    json_payload: dict[str, Any],
) -> dict[str, Any]:
    """Stamp a JSON-only run with the marker the brief calls for so
    callers / tests can tell apart a real DB-backed row from a
    JSON-fallback row, AND so a future Phase-3 backfill knows which
    rows it can safely DELETE-then-INSERT once the DB catches up.

    The returned dict is the JSON payload itself (mutated in place),
    not a separate object — the front-end expects the JSON-shape
    fields, and we don't want to fan them out twice. The marker
    lives under ``_synthetic_from_json``.
    """
    json_payload.setdefault("_source", "json")
    json_payload["_synthetic_from_json"] = True
    # Preserve a parsed-shape ``extra_json`` so consumers who DO want
    # the structured marker (e.g., backfill tooling) get the same
    # shape they'd find on a real DB-backed row.
    extra = {"_synthetic_from_json": True}
    json_payload.setdefault("extra_json", json.dumps(extra, sort_keys=True))
    return json_payload


def _merge_db_and_json(
    db_dict: dict[str, Any], json_dict: dict[str, Any],
) -> dict[str, Any]:
    """Both sources have a row for the same ``run_id``. JSON has
    the rich UI-render fields (``machines``, ``pairings``, replay
    hints, per-epoch losses, ...); the DB has authoritative
    started_at / status / kind. Strategy: start from JSON, layer the
    DB row's status + ended_at + git_commit + label + notes on top
    when they're set in the DB but absent in JSON, and stamp
    ``_source: "merged"`` so the test fixtures can pin behavior.
    """
    merged = dict(json_dict)
    # DB-side authoritative fields fill JSON gaps but don't overwrite
    # JSON when JSON has a value. Today that's almost everything since
    # JSON is the legacy source; this matters as Phase 3 approaches
    # when JSON keys go absent.
    for key in (
        "status", "ended_at", "started_at", "git_commit",
        "kind", "label", "notes",
    ):
        if merged.get(key) in (None, "") and db_dict.get(key) is not None:
            merged[key] = db_dict[key]
    merged["_db_kind"] = db_dict.get("_db_kind")
    merged["_source"] = "merged"
    return merged


# ---- Top-level "list runs" entry point ------------------------------------


def _list_runs(progress_dir: Path) -> list[dict[str, Any]]:
    """All runs visible to the UI, newest-first by ``started_at``.

    Combines the DB's ``runs`` rows with any standalone JSON files
    that the DB doesn't yet know about. JSON files for which a DB
    row exists merge into the DB row (DB authoritative for status /
    ended_at; JSON authoritative for the rich render fields).

    Per-shard JSON files (``*_shard_*.json``) are skipped — the
    aggregator writes a rolled-up ``<run_id>.json`` alongside them
    and the UI shows the aggregate, not N per-shard cards.
    """
    json_runs = _scan_json_dir(progress_dir)
    db_runs: dict[str, dict[str, Any]] = {}
    conn = _open_conn(progress_dir)
    try:
        if conn is not None:
            db_runs = _scan_db(conn)
    finally:
        if conn is not None:
            conn.close()

    merged: dict[str, dict[str, Any]] = {}
    # Start with DB-known runs; fold JSON into them when the JSON
    # exists alongside.
    for run_id, db_row in db_runs.items():
        if run_id in json_runs:
            merged[run_id] = _merge_db_and_json(
                db_row, json_runs[run_id],
            )
        else:
            merged[run_id] = db_row
    # Add JSON-only runs as synthesized rows.
    for run_id, json_row in json_runs.items():
        if run_id in merged:
            continue
        merged[run_id] = _synthesize_db_row_from_json(json_row)

    runs = list(merged.values())
    runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return runs


def _get_run_detail(
    progress_dir: Path, run_id: str,
) -> Optional[dict[str, Any]]:
    """Per-run detail: merged DB + JSON view PLUS the events list,
    artifacts list, and per-shard progress rows from the DB. Returns
    ``None`` when the run is unknown to both stores."""
    json_runs = _scan_json_dir(progress_dir)
    json_row = json_runs.get(run_id)

    conn = _open_conn(progress_dir)
    db_record: Optional[dict[str, Any]] = None
    try:
        if conn is not None:
            db_record = runs_db.get_run(conn, run_id)
    finally:
        if conn is not None:
            conn.close()

    if db_record is None and json_row is None:
        return None

    if db_record is not None:
        run_row = db_record["run"]
        agg = None
        per_shard = []
        for p in db_record.get("progress", []):
            if p.get("shard_label") is None:
                agg = p
            else:
                per_shard.append(p)
        ui = _db_row_to_ui_dict(run_row, agg)
        if json_row is not None:
            ui = _merge_db_and_json(ui, json_row)
        ui["_db_progress_per_shard"] = per_shard
        ui["_db_events"] = db_record.get("events", [])
        ui["_db_artifacts"] = db_record.get("artifacts", [])
        return ui

    # DB has no row but JSON does — synthesize.
    assert json_row is not None
    ui = _synthesize_db_row_from_json(dict(json_row))
    ui["_db_progress_per_shard"] = []
    ui["_db_events"] = []
    ui["_db_artifacts"] = []
    return ui


def _list_recent_runs(
    progress_dir: Path, limit: int = 20,
) -> list[dict[str, Any]]:
    """Last N completed runs across all kinds, with summary stats.
    Source is the DB (JSON-only completed runs surface via the
    ``/api/runs`` listing; this endpoint is explicitly DB-driven so
    the cross-run aggregations stay cheap).
    """
    conn = _open_conn(progress_dir)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            SELECT r.*,
                   p.n_games_done   AS p_n_games_done,
                   p.n_games_total  AS p_n_games_total,
                   p.elapsed_seconds AS p_elapsed_seconds,
                   p.wins_new       AS p_wins_new,
                   p.wins_old       AS p_wins_old,
                   p.draws          AS p_draws,
                   p.crashed        AS p_crashed,
                   p.timed_out      AS p_timed_out
            FROM runs r
            LEFT JOIN run_progress p
              ON p.run_id = r.run_id AND p.shard_label IS NULL
            WHERE r.status = 'completed'
            ORDER BY r.started_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        out.append({
            "run_id": d["run_id"],
            "kind": _ui_kind_from_db(d.get("kind")),
            "status": d.get("status"),
            "started_at": d.get("started_at"),
            "ended_at": d.get("ended_at"),
            "git_commit": d.get("git_commit"),
            "label": d.get("label"),
            "n_games_done": d.get("p_n_games_done"),
            "n_games_total": d.get("p_n_games_total"),
            "elapsed_seconds": d.get("p_elapsed_seconds"),
            "wins_new": d.get("p_wins_new"),
            "wins_old": d.get("p_wins_old"),
            "draws": d.get("p_draws"),
            "crashed": d.get("p_crashed"),
            "timed_out": d.get("p_timed_out"),
        })
    return out


# ---- Flask app factory ----------------------------------------------------


def create_app(progress_dir: Path) -> Flask:
    """Build the Flask app. ``progress_dir`` is the directory that
    holds both the per-run progress JSON files (legacy + fallback)
    AND the ``runs.db`` SQLite file (Phase 2 primary). The directory
    does NOT have to exist at app-creation time — it's read lazily
    on every request, so a UI started before the first training run
    still works."""
    progress_dir = Path(progress_dir).resolve()

    # Templates live next to this file (mirroring how mtg_bot.ui.py
    # uses src/mtg_bot/templates). Static assets — favicon, etc. —
    # live in the sibling ``static/`` and are served at ``/static/``
    # by Flask's default static_url_path.
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config["PROGRESS_DIR"] = progress_dir

    @app.route("/")
    def index():  # noqa: D401 - Flask view
        return render_template("index.html")

    @app.route("/api/runs")
    def api_runs():
        runs = _list_runs(app.config["PROGRESS_DIR"])
        return jsonify({"runs": runs})

    @app.route("/api/runs/recent")
    def api_runs_recent():
        """Last 20 completed runs across all kinds. Light-weight
        cross-run summary panel."""
        limit_raw = request.args.get("limit", "20")
        try:
            limit = max(1, min(int(limit_raw), 200))
        except (TypeError, ValueError):
            limit = 20
        runs = _list_recent_runs(app.config["PROGRESS_DIR"], limit=limit)
        return jsonify({"runs": runs, "limit": limit})

    @app.route("/api/runs/<run_id>")
    def api_run(run_id: str):
        run = _get_run_detail(app.config["PROGRESS_DIR"], run_id)
        if run is None:
            abort(404)
        return jsonify(run)

    @app.route("/api/runs/<run_id>/replays")
    def api_run_replays(run_id: str):
        """List captured replay files for a given run.

        The producer (iterate_alphazero) writes ``replays_dir`` into
        the run's progress JSON pointing at the directory it captured
        into. This endpoint reads that hint and returns the
        ``.json.gz`` files alongside the subpath the main mtg_bot UI
        needs to load them via ``/api/load_replay/<subpath>``. Subpath
        = ``<replays_dir.name>/<filename>``, since the main UI's
        ``--replays-root`` is meant to point at the parent directory.

        ``replays_dir`` is currently a JSON-only field — Phase 1's
        run-DB schema doesn't store it. We resolve via JSON when
        present, and fall through to a "no replays" response otherwise.
        """
        run = _get_run_detail(app.config["PROGRESS_DIR"], run_id)
        if run is None:
            abort(404)
        replays_dir_str = run.get("replays_dir")
        if not replays_dir_str:
            return jsonify({
                "replays_dir": None,
                "replays": [],
                "count": 0,
                "note": "this run did not capture replays",
            })
        replays_dir = Path(replays_dir_str)
        if not replays_dir.is_dir():
            return jsonify({
                "replays_dir": str(replays_dir),
                "replays": [],
                "count": 0,
                "note": "replays_dir does not exist (yet?)",
            })
        files = sorted(p.name for p in replays_dir.glob("*.json.gz"))
        subdir = replays_dir.name
        return jsonify({
            "replays_dir": str(replays_dir),
            "replays": [
                {"filename": name, "subpath": f"{subdir}/{name}"}
                for name in files
            ],
            "count": len(files),
        })

    @app.route("/api/runs/<run_id>/stop", methods=["POST"])
    def api_run_stop(run_id: str):
        """Request a stop for ``run_id``.

        Writes a sentinel file ``.STOP_<target_id>`` in the progress
        directory. The producer (iterate cycle / self-play /
        eval pool / queue server) polls for that sentinel at stage
        boundaries and exits cleanly, finalizing its progress JSON
        with ``status="stopped"``.

        Cycle-vs-single-run targeting: when the run's progress JSON
        has a ``cycle_id`` field (set by ``iterate_alphazero`` for
        all child stages — self-play, train, eval), the sentinel is
        written for the cycle, which propagates to the in-flight
        child via the dual-id check in
        ``stop_signal.stop_requested``. Otherwise the sentinel is
        written for the single run.

        Idempotent — calling twice does NOT error; the second call
        refreshes the sentinel's timestamp + reason. Body is an
        optional JSON ``{"reason": str}`` for log/audit metadata.
        """
        progress_dir: Path = app.config["PROGRESS_DIR"]
        run = _get_run_detail(progress_dir, run_id)
        if run is None:
            abort(404)
        body = request.get_json(silent=True) or {}
        reason = body.get("reason") if isinstance(body, dict) else None
        # Cycle-aware targeting: a child stage's progress JSON carries
        # a ``cycle_id`` back-pointer when iterate_alphazero owns it.
        # Fall back to the DB ``parent_run_id`` if JSON's missing.
        cycle_id = run.get("cycle_id") or run.get("parent_run_id")
        target_id = cycle_id if cycle_id else run_id
        path = write_stop(progress_dir, target_id, reason=reason)
        return jsonify({
            "ok": True,
            "run_id": run_id,
            "stopped_target": target_id,
            "is_cycle_stop": bool(cycle_id),
            "sentinel_path": str(path),
        })

    @app.route("/api/runs/<run_id>/stop", methods=["GET"])
    def api_run_stop_status(run_id: str):
        """Diagnostic: report whether a stop has been requested for
        ``run_id`` (or its parent cycle, if any). Useful for the UI
        to show "stop pending" between the click and the next
        producer-side stage boundary, and for tests."""
        progress_dir: Path = app.config["PROGRESS_DIR"]
        run = _get_run_detail(progress_dir, run_id)
        if run is None:
            abort(404)
        cycle_id = run.get("cycle_id") or run.get("parent_run_id")
        own = sentinel_path(progress_dir, run_id).is_file()
        cycle = bool(cycle_id) and sentinel_path(
            progress_dir, cycle_id,
        ).is_file()
        return jsonify({
            "run_id": run_id,
            "cycle_id": cycle_id,
            "stop_requested": own or cycle or stop_requested(
                progress_dir, run_id, cycle_id=cycle_id,
            ),
            "own_sentinel": own,
            "cycle_sentinel": cycle,
        })

    return app
