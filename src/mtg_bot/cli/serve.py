"""Tiny master-side HTTP server that hands per-OS setup scripts to
remote machines over Tailscale.

Run this on the master::

    PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli serve

(or, via the legacy shim that's still used in some user workflows:
``python scripts/serve_setup_scripts.py``).

Endpoints:
- ``GET /setup_remote.sh``   — macOS / Linux setup script
- ``GET /setup_remote.ps1``  — Windows setup script
- ``GET /``                  — landing page with the curl + iex one-liners
- ``POST /queue/init``, ``POST /queue/take``, ``POST /queue/close/<id>``,
  ``GET /queue/status``  — the pull-mode self-play queue
- ``GET /repo.bundle``  — fresh ``git bundle --all`` of the master repo

Each request reads the corresponding template file, fills in the
master's repo URL + ssh-ed25519 public key + Tailscale hostname, and
returns the rendered text. The remote can then::

    # macOS / Linux:
    curl -sSL http://<master-tailnet-name>:5099/setup_remote.sh | bash

    # Windows (run as Administrator):
    iex (iwr http://<master-tailnet-name>:5099/setup_remote.ps1).Content

Path / pubkey / repo URL defaults are auto-detected; override via
CLI flags if the master's setup is non-standard.
"""

from __future__ import annotations

import argparse
import socket
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from flask import (
    Flask, Response, jsonify, render_template_string, request,
)

# Templates live in ``scripts/`` alongside the legacy shim — both for
# historical reasons (they shipped there before this refactor) and
# because they're shell-script templates, not Python source. Locate
# them relative to the repo root via this module's path.
_REPO_ROOT_FROM_HERE = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT_FROM_HERE / "scripts"


_LANDING_HTML = """<!doctype html>
<html><head><title>mtg-bot remote setup</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 820px;
         margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
  h1 { font-size: 1.6rem; }
  pre { background: #f4f4f6; padding: 12px 14px; border-radius: 6px;
        overflow-x: auto; font-size: 0.85rem; }
  code { font-family: ui-monospace, monospace; }
  .meta { color: #666; font-size: 0.9rem; }
</style></head>
<body>
<h1>mtg-bot remote setup</h1>
<p class="meta">
  Master: <code>{{ master_tailnet_name }}</code> &nbsp;·&nbsp;
  Repo: <code>{{ repo_url }}</code> &nbsp;·&nbsp;
  Pubkey: <code>{{ pubkey_short }}</code>
</p>
<h2>macOS / Linux</h2>
<p>Run on the remote (any user):</p>
<pre><code>curl -sSL http://{{ master_tailnet_name }}:{{ port }}/setup_remote.sh | bash</code></pre>
<h2>Windows</h2>
<p>Run on the remote in <strong>elevated PowerShell</strong>:</p>
<pre><code>iex (iwr http://{{ master_tailnet_name }}:{{ port }}/setup_remote.ps1).Content</code></pre>
<h2>What the scripts do</h2>
<ul>
  <li>Install Python 3.13 (via Homebrew on Mac, winget on Windows) if missing.</li>
  <li>Install the GitHub CLI (<code>gh</code>) and check authentication.
      If not logged in, the script exits with instructions to run
      <code>gh auth login</code> + <code>gh auth setup-git</code>
      interactively, then re-run the one-liner.</li>
  <li>Clone <code>{{ repo_url }}</code> into a default path under the user's home.</li>
  <li>Create <code>.venv-py313</code> + install the package + PyTorch.</li>
  <li>Append the master's public key to <code>~/.ssh/authorized_keys</code>.</li>
  <li>Windows: enable OpenSSH Server, open the firewall, set the right ACL.</li>
  <li>Run the distributed-self-play tests as a smoke check.</li>
</ul>
<p>Tailscale install + login still needs a browser, so it stays manual on each remote.</p>
<h2>One-time GitHub auth (per remote)</h2>
<p>Each remote needs to be logged in to GitHub via <code>gh</code>
before the scripts can clone. After running the one-liner once and
seeing the auth prompt:</p>
<pre><code>gh auth login    # GitHub.com → HTTPS → "Login with a web browser"
gh auth setup-git</code></pre>
<p>Then re-run the curl/iex one-liner above — the clone will succeed
and the rest of setup proceeds.</p>
</body></html>
"""


def _detect_master_tailnet_name() -> str:  # pragma: no cover
    """Best-effort: ``tailscale status`` lists the master's name in
    the row marked with ``-``. Fall back to the host's gethostname()
    if Tailscale isn't installed or running. (Subprocess-driven
    helper; exercised only on operator machines, not in tests.)"""
    try:
        result = subprocess.run(
            ["tailscale", "status"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                # Format: ``<ip>  <name>  <user>  <os>  -``
                parts = line.split()
                if parts and parts[-1] == "-":
                    return parts[1]
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return socket.gethostname()


def _detect_repo_url(repo_root: Path) -> str:  # pragma: no cover
    """``git remote get-url origin`` from the repo root.
    (Subprocess-driven; exercised only at runtime.)"""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def build_repo_bundle(repo_root: Path) -> bytes:
    """Run ``git bundle create -`` in ``repo_root`` and return the
    bundle bytes.

    ``--all`` packs every ref (every branch, every tag) so remotes
    can checkout any branch the master has — they'll typically only
    need ``main``, but we don't want to silently exclude branches.

    The bundle is binary, ~5-50 MB depending on repo size. Master's
    serve subcommand builds one fresh on each request so remote
    ``git pull``-equivalents always see the latest commits.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), "bundle", "create", "-", "--all"],
        capture_output=True, check=True, timeout=120,
    )
    return result.stdout


def _read_pubkey(pubkey_path: Path) -> str:
    if not pubkey_path.is_file():
        return ""
    return pubkey_path.read_text(encoding="utf-8").strip()


def _render_template(
    template_path: Path,
    *,
    repo_url: str,
    repo_path: str,
    master_pubkey: str,
    master_tailnet_name: str,
) -> str:
    """Substitute the ``{{KEY}}`` placeholders in the template. We
    don't use Jinja for this because the templates are ALSO valid
    standalone scripts that the user can read / hand-edit / sanity-
    check — Jinja would obscure the syntax."""
    body = template_path.read_text(encoding="utf-8")
    body = body.replace("{{REPO_URL}}", repo_url)
    body = body.replace("{{REPO_PATH}}", repo_path)
    body = body.replace("{{MASTER_PUBKEY}}", master_pubkey)
    body = body.replace("{{MASTER_TAILNET_NAME}}", master_tailnet_name)
    return body


def create_app(
    *,
    repo_url: str,
    master_pubkey: str,
    master_tailnet_name: str,
    port: int,
    posix_repo_path: str = "$HOME/projects/mtg-bot",
    windows_repo_path: str = "$env:USERPROFILE\\Projects\\mtg-bot",
    template_dir: Path = _SCRIPTS_DIR,
    repo_root: Optional[Path] = None,
    progress_dir: Optional[Path] = None,
    runs_db_path: Optional[Path] = None,
    shards_config_path: Optional[Path] = None,
) -> Flask:
    """Build the master HTTP server. ``progress_dir`` (when set)
    enables Stop-button propagation to remotes: each ``/queue/take``
    request first checks for a ``.STOP_<run_id>`` sentinel, and on
    hit returns 410 Gone with a ``"stopped"`` body so remotes drain
    cleanly without playing extra games. Default ``None`` falls back
    to the old behavior (no stop check) for callers that haven't
    plumbed a progress_dir through yet.

    ``runs_db_path`` (when set) enables the run-database REST
    endpoints (``/run/start``, ``/run/<id>/progress``, ...). All
    write endpoints serialize through ``mtg_bot.cli.runs.db.WRITE_LOCK``
    so concurrent POSTs from many remotes don't race the SQLite
    writer. Default ``None`` disables these endpoints (returns 404).

    ``shards_config_path`` (when set) enables the dynamic worker
    scaling endpoints (``GET /api/shards``, ``POST /api/shards/
    <host>/workers``). The training UI uses these to bump
    ``n_workers`` per shard between iters. **Phase 1 only**: the
    write lands on disk immediately and the iterate loop's per-iter
    refresh picks it up at the next iter boundary; the in-flight
    iter is unchanged. Mid-iter scaling is Phase 2 (deferred).
    """
    # Local import to avoid scripts/ depending on the package at module
    # import time during early bootstrap.
    from mtg_bot.training.stop_signal import (  # noqa: WPS433
        stop_requested as _stop_requested,
    )
    from mtg_bot.cli.runs import db as _runs_db  # noqa: WPS433
    from mtg_bot.cli.runs.models import (  # noqa: WPS433
        Artifact, Event, ProgressUpdate, Run,
    )
    from mtg_bot.training import shards_config_edit  # noqa: WPS433

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            _LANDING_HTML,
            master_tailnet_name=master_tailnet_name,
            repo_url=repo_url,
            pubkey_short=(
                master_pubkey[:30] + "..." if master_pubkey
                else "(missing)"
            ),
            port=port,
        )

    @app.route("/setup_remote.sh")
    def setup_macos():
        body = _render_template(
            template_dir / "setup_remote_macos.sh",
            repo_url=repo_url,
            repo_path=posix_repo_path,
            master_pubkey=master_pubkey,
            master_tailnet_name=master_tailnet_name,
        )
        return Response(body, mimetype="text/x-shellscript")

    @app.route("/setup_remote.ps1")
    def setup_windows():
        body = _render_template(
            template_dir / "setup_remote_windows.ps1",
            repo_url=repo_url,
            repo_path=windows_repo_path,
            master_pubkey=master_pubkey,
            master_tailnet_name=master_tailnet_name,
        )
        return Response(body, mimetype="text/x-powershell")

    # ---- pull-model self-play queue ------------------------------
    # In-memory queue state, keyed by run_id (e.g.
    # "iter_001_self_play_v6"). Each iter creates a fresh queue;
    # there's no persistence — if the master restarts mid-iter,
    # workers will see "queue not initialized" and bail. That's
    # the right outcome (iter is dead anyway).
    _queues: dict[str, dict] = {}
    _queues_lock = threading.Lock()

    @app.route("/queue/init", methods=["POST"])
    def queue_init():
        """Initialize the queue for a fresh run. Body:
        ``{"run_id": str, "total_games": int}``. Idempotent — re-init
        with the same total_games is a no-op; with a different total
        replaces the queue (intentional — caller knows what they're
        doing, e.g. extending an iter)."""
        data = request.get_json(silent=True) or {}
        run_id = data.get("run_id")
        total = data.get("total_games")
        if not run_id or not isinstance(total, int) or total <= 0:
            return jsonify({"error": "run_id + total_games required"}), 400
        with _queues_lock:
            _queues[run_id] = {"next": 0, "total": total, "closed": False}
        return jsonify({"ok": True, "run_id": run_id, "total_games": total})

    @app.route("/queue/close/<run_id>", methods=["POST"])
    def queue_close(run_id: str):
        """Mark the queue for ``run_id`` as closed: subsequent
        ``/queue/take`` calls return an empty list with
        ``remaining=0`` regardless of how many indices remain. Used
        by the orchestrator to early-stop a run once enough games
        have completed successfully — the over-queue slack is
        abandoned, in-flight stragglers are no longer blocking.

        Idempotent: closing an already-closed queue is a no-op.
        Returns 404 if the run_id was never initialized."""
        with _queues_lock:
            q = _queues.get(run_id)
            if q is None:
                return jsonify({"error": f"unknown run_id {run_id!r}"}), 404
            q["closed"] = True
            return jsonify({
                "ok": True,
                "run_id": run_id,
                "closed": True,
                "next": q["next"],
                "total": q["total"],
            })

    @app.route("/queue/take", methods=["POST"])
    def queue_take():
        """Hand out the next batch of game indices for ``run_id``.
        Body: ``{"run_id": str, "batch_size": int}``. Returns
        ``{"games": [int, ...]}`` — empty list means queue is
        drained (worker should exit its loop).

        Atomic: the lock-protected slice + advance ensures two
        workers calling concurrently can't get overlapping ranges.

        Stop-button propagation: when ``progress_dir`` is configured
        and a ``.STOP_<run_id>`` sentinel exists, returns 410 Gone
        with body ``{"stopped": True}``. Remotes treat 410 as a
        clean drain signal — they exit their take-loop after
        finishing their current game, the same way they handle 404
        on a master shutdown.
        """
        data = request.get_json(silent=True) or {}
        run_id = data.get("run_id")
        batch_size = int(data.get("batch_size", 25))
        if batch_size < 1:
            return jsonify({"error": "batch_size must be >= 1"}), 400
        if progress_dir is not None and run_id and _stop_requested(
            progress_dir, run_id,
        ):
            return jsonify({
                "stopped": True,
                "run_id": run_id,
                "games": [],
                "remaining": 0,
            }), 410
        with _queues_lock:
            q = _queues.get(run_id)
            if q is None:
                return jsonify({"error": f"unknown run_id {run_id!r}"}), 404
            # Early-stop signal from the orchestrator: refuse to
            # hand out new games even though indices remain. The
            # in-flight games are still in flight; the abandoned
            # slack is the over-queue safety margin we don't need.
            if q.get("closed"):
                return jsonify({
                    "games": [],
                    "remaining": 0,
                    "closed": True,
                })
            start = q["next"]
            end = min(start + batch_size, q["total"])
            q["next"] = end
        return jsonify({
            "games": list(range(start, end)),
            "remaining": q["total"] - end,
        })

    @app.route("/queue/status", methods=["GET"])
    def queue_status():
        """Diagnostic: report current state of all queues. Useful to
        eyeball "is the master making progress" without polling each
        shard's progress JSON."""
        with _queues_lock:
            return jsonify({
                "queues": {
                    rid: {
                        "next": q["next"],
                        "total": q["total"],
                        "remaining": q["total"] - q["next"],
                        "closed": bool(q.get("closed")),
                    }
                    for rid, q in _queues.items()
                },
            })

    # ---- run database endpoints ----------------------------------
    # Enabled iff ``runs_db_path`` is set. All writes serialize
    # through the module-level WRITE_LOCK so concurrent POSTs from
    # many remotes can't race the single SQLite writer. Each handler
    # opens a fresh connection (cheap; pragmas are per-conn for
    # foreign_keys but journal_mode=WAL is DB-global so the cost is
    # one syscall) — keeping connections short-lived avoids the
    # check_same_thread / Flask-threading hazards a long-lived shared
    # conn would invite.

    def _runs_db_disabled():
        return jsonify({
            "error": "run database not configured on this server",
        }), 404

    @app.route("/run/start", methods=["POST"])
    def run_start():
        if runs_db_path is None:
            return _runs_db_disabled()
        data = request.get_json(silent=True) or {}
        try:
            run = Run(
                run_id=data["run_id"],
                kind=data["kind"],
                status="running",
                started_at=data.get("started_at") or _runs_db.now_iso(),
                git_commit=data["git_commit"],
                config_json=data["config_json"],
                parent_run_id=data.get("parent_run_id"),
                label=data.get("label"),
            )
        except KeyError as exc:
            return jsonify({"error": f"missing field: {exc}"}), 400
        try:
            with _runs_db.WRITE_LOCK:
                conn = _runs_db.connect(runs_db_path)
                try:
                    _runs_db.insert_run(conn, run)
                finally:
                    conn.close()
        except (ValueError, sqlite3.IntegrityError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "run_id": run.run_id})

    @app.route("/run/<run_id>/progress", methods=["POST"])
    def run_progress(run_id: str):
        if runs_db_path is None:
            return _runs_db_disabled()
        data = request.get_json(silent=True) or {}
        update = ProgressUpdate(
            run_id=run_id,
            shard_label=data.get("shard_label"),
            n_games_done=data.get("n_games_done"),
            n_games_total=data.get("n_games_total"),
            elapsed_seconds=data.get("elapsed_seconds"),
            crashed=data.get("crashed"),
            timed_out=data.get("timed_out"),
            wins_new=data.get("wins_new"),
            wins_old=data.get("wins_old"),
            draws=data.get("draws"),
            extra_json=data.get("extra_json"),
        )
        with _runs_db.WRITE_LOCK:
            conn = _runs_db.connect(runs_db_path)
            try:
                _runs_db.upsert_progress(conn, update)
            finally:
                conn.close()
        return jsonify({"ok": True, "run_id": run_id})

    @app.route("/run/<run_id>/event", methods=["POST"])
    def run_event(run_id: str):
        if runs_db_path is None:
            return _runs_db_disabled()
        data = request.get_json(silent=True) or {}
        event_type = data.get("event_type")
        if not event_type:
            return jsonify({"error": "event_type required"}), 400
        event = Event(
            run_id=run_id,
            event_type=event_type,
            shard_label=data.get("shard_label"),
            payload_json=data.get("payload_json"),
        )
        with _runs_db.WRITE_LOCK:
            conn = _runs_db.connect(runs_db_path)
            try:
                _runs_db.insert_event(conn, event)
            finally:
                conn.close()
        return jsonify({"ok": True, "run_id": run_id})

    @app.route("/run/<run_id>/finish", methods=["POST"])
    def run_finish(run_id: str):
        if runs_db_path is None:
            return _runs_db_disabled()
        data = request.get_json(silent=True) or {}
        status = data.get("status")
        if status not in ("completed", "failed", "killed"):
            return jsonify({
                "error": "status must be completed/failed/killed",
            }), 400
        try:
            with _runs_db.WRITE_LOCK:
                conn = _runs_db.connect(runs_db_path)
                try:
                    _runs_db.finish_run(
                        conn, run_id, status, notes=data.get("notes"),
                    )
                finally:
                    conn.close()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "run_id": run_id, "status": status})

    @app.route("/run/<run_id>/artifact", methods=["POST"])
    def run_artifact(run_id: str):
        if runs_db_path is None:
            return _runs_db_disabled()
        data = request.get_json(silent=True) or {}
        artifact_type = data.get("artifact_type")
        path = data.get("path")
        if not artifact_type or not path:
            return jsonify({
                "error": "artifact_type and path required",
            }), 400
        artifact = Artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            path=path,
            bytes=data.get("bytes"),
        )
        with _runs_db.WRITE_LOCK:
            conn = _runs_db.connect(runs_db_path)
            try:
                _runs_db.upsert_artifact(conn, artifact)
            finally:
                conn.close()
        return jsonify({"ok": True, "run_id": run_id})

    @app.route("/run/<run_id>", methods=["GET"])
    def run_get(run_id: str):
        if runs_db_path is None:
            return _runs_db_disabled()
        # Reads don't need WRITE_LOCK (WAL gives concurrent readers).
        conn = _runs_db.connect(runs_db_path)
        try:
            record = _runs_db.get_run(conn, run_id)
        finally:
            conn.close()
        if record is None:
            return jsonify({"error": "run not found"}), 404
        return jsonify(record)

    @app.route("/runs", methods=["GET"])
    def run_list():
        if runs_db_path is None:
            return _runs_db_disabled()
        status = request.args.get("status")
        kind = request.args.get("kind")
        since = request.args.get("since")
        try:
            limit = int(request.args.get("limit", 100))
        except ValueError:
            return jsonify({"error": "limit must be an int"}), 400
        conn = _runs_db.connect(runs_db_path)
        try:
            rows = _runs_db.list_runs(
                conn, status=status, kind=kind, since=since, limit=limit,
            )
        finally:
            conn.close()
        return jsonify({"runs": rows})

    # ---- dynamic worker scaling (Phase 1) -----------------------
    # Enabled iff ``shards_config_path`` is set. Phase 1 semantics:
    # the POST mutates the on-disk shards-config file immediately,
    # and iterate's per-iter refresh picks the change up at the next
    # iter boundary. The CURRENTLY RUNNING iter is unaffected (its
    # multiprocessing.Pool is already sized). Phase 2 will support
    # mid-iter rescale by switching to a queue + spawn/kill worker
    # pattern; explicitly out of scope here.

    def _shards_disabled():
        return jsonify({
            "error": "shards-config not configured on this server",
        }), 404

    @app.route("/api/shards", methods=["GET"])
    def api_shards_get():
        """Return the current contents of the shards-config file
        (the ``{"shards": [...]}`` JSON dict). Used by the training
        UI to know which hosts exist and what their current
        n_workers values are independent of any in-flight iter's
        per-shard progress JSON."""
        if shards_config_path is None:
            return _shards_disabled()
        if not shards_config_path.is_file():
            return jsonify({
                "error": f"shards-config not found: {shards_config_path}",
            }), 404
        try:
            raw = shards_config_edit.read_shards_config_raw(
                shards_config_path,
            )
        except (OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify(raw)

    @app.route("/api/shards/<host>/workers", methods=["POST"])
    def api_shards_set_workers(host: str):
        """Update the n_workers field for the shard with
        ``host == <host>``. Body: ``{"n_workers": N}`` with N in
        ``[1, 64]``. Returns 400 on invalid value, 404 on unknown
        host. The mutation persists across server / iterate
        restarts because it writes through to the on-disk JSON."""
        if shards_config_path is None:
            return _shards_disabled()
        data = request.get_json(silent=True) or {}
        if "n_workers" not in data:
            return jsonify({"error": "n_workers required"}), 400
        try:
            updated = shards_config_edit.update_shard_n_workers(
                shards_config_path, host, data["n_workers"],
            )
        except shards_config_edit.ShardsConfigEditError as exc:
            msg = str(exc)
            if msg.startswith("unknown host"):
                return jsonify({"error": msg}), 404
            return jsonify({"error": msg}), 400
        return jsonify({
            "ok": True,
            "host": host,
            "n_workers": int(data["n_workers"]),
            "shards": updated.get("shards", []),
        })

    @app.route("/repo.bundle")
    def repo_bundle():
        """Stream the master's repo as a git bundle — every ref,
        current commits. Remotes use this in lieu of ``git pull
        origin`` so they don't need GitHub auth (only Tailscale
        access). The orchestrator's ``run_shard_remote`` curls this
        URL before each iter to refresh the remote's working tree.

        Re-builds on every request — the latest committed state is
        always served. ~few hundred ms on this repo size; only fires
        at iter boundaries so the cost is negligible.
        """
        if repo_root is None:
            return Response(
                "repo_root not configured", status=500,
                mimetype="text/plain",
            )
        try:
            data = build_repo_bundle(repo_root)
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            return Response(
                f"git bundle failed: {exc}", status=500,
                mimetype="text/plain",
            )
        return Response(data, mimetype="application/x-git-bundle")

    return app


def main(argv=None) -> None:  # pragma: no cover
    # Argparse + Flask-app boot body. The ``create_app`` factory and
    # the helpers are unit-tested directly via ``tests/scripts/
    # test_serve_setup_scripts.py``; this body just wires defaults
    # and calls ``app.run`` (which would block the test runner).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=5099,
        help="Port to listen on. Default 5099.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help=(
            "Bind address. Default 0.0.0.0 so Tailscale peers can "
            "reach the server."
        ),
    )
    parser.add_argument(
        "--pubkey-path", type=Path,
        default=Path.home() / ".ssh" / "id_ed25519.pub",
        help=(
            "Master's SSH public key file. Default "
            "~/.ssh/id_ed25519.pub. The contents are inlined into "
            "every served setup script so the remote can authorize "
            "the master without a separate key-copy step."
        ),
    )
    parser.add_argument(
        "--repo-root", type=Path,
        default=_REPO_ROOT_FROM_HERE,
        help=(
            "Path to the local mtg-bot repo. Used to detect "
            "``origin``'s URL via ``git remote get-url``."
        ),
    )
    parser.add_argument(
        "--repo-url", default=None,
        help=(
            "Override the auto-detected git remote URL (handy if you "
            "want remotes to clone over Tailscale instead of public "
            "GitHub)."
        ),
    )
    parser.add_argument(
        "--progress-dir", type=Path, default=None,
        help=(
            "Path to the master's training_progress directory. When "
            "set, /queue/take checks for a .STOP_<run_id> sentinel "
            "before handing out games and returns 410 Gone if "
            "present, so the UI's Stop button propagates cleanly to "
            "remotes."
        ),
    )
    parser.add_argument(
        "--runs-db-path", type=Path,
        default=Path("training_progress") / "runs.db",
        help=(
            "Path to the master's run database (SQLite). When set, "
            "the server exposes /run/start, /run/<id>/progress, "
            "/run/<id>/event, /run/<id>/finish, /run/<id>/artifact, "
            "GET /run/<id>, and GET /runs. Default "
            "training_progress/runs.db. Pass an empty string to "
            "disable the run-DB endpoints entirely."
        ),
    )
    parser.add_argument(
        "--shards-config", type=Path,
        default=Path("shards_config.json"),
        help=(
            "Path to the shards-config JSON file iterate reads at "
            "the top of each iter. When set, the server exposes "
            "GET /api/shards and POST /api/shards/<host>/workers "
            "for the training UI's dynamic worker-scale controls. "
            "Phase 1 only: writes take effect on the NEXT iter (the "
            "in-flight iter is unchanged). Pass an empty string to "
            "disable these endpoints. Default: shards_config.json."
        ),
    )
    args = parser.parse_args(argv)

    repo_url = args.repo_url or _detect_repo_url(args.repo_root)
    if not repo_url:
        sys.exit(
            f"could not detect git remote URL from {args.repo_root!r}; "
            f"pass --repo-url manually."
        )
    pubkey = _read_pubkey(args.pubkey_path)
    if not pubkey:
        print(
            f"[serve_setup] WARNING: no public key at "
            f"{args.pubkey_path}; setup scripts will print a manual "
            f"step instead.",
            file=sys.stderr,
        )
    master_tailnet_name = _detect_master_tailnet_name()

    runs_db_path: Optional[Path] = args.runs_db_path
    if runs_db_path is not None and str(runs_db_path) == ".":
        # ``--runs-db-path ''`` collapses to ``Path('.')``; treat as
        # explicit-disable.
        runs_db_path = None
    shards_config_path: Optional[Path] = args.shards_config
    if shards_config_path is not None and str(shards_config_path) == ".":
        # ``--shards-config ''`` collapses to ``Path('.')``; treat as
        # explicit-disable.
        shards_config_path = None
    app = create_app(
        repo_url=repo_url,
        master_pubkey=pubkey,
        master_tailnet_name=master_tailnet_name,
        port=args.port,
        repo_root=args.repo_root,
        progress_dir=args.progress_dir,
        runs_db_path=runs_db_path,
        shards_config_path=shards_config_path,
    )
    print(
        f"[serve_setup] master={master_tailnet_name} repo={repo_url} "
        f"pubkey={'yes' if pubkey else 'NO'}\n"
        f"[serve_setup] mac one-liner:  curl -sSL "
        f"http://{master_tailnet_name}:{args.port}/setup_remote.sh | bash\n"
        f"[serve_setup] win one-liner:  iex (iwr "
        f"http://{master_tailnet_name}:{args.port}/setup_remote.ps1).Content",
        file=sys.stderr,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
