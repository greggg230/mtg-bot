"""Tests for ``mtg_bot.cli.serve`` (the runner formerly known as
``scripts/serve_setup_scripts.py``; the script is now a thin shim
that just calls into this module).

Cover the template-substitution + endpoint behavior. The real-
remote flow (curl from a Mac, iex from a Win box) needs the actual
machines and is validated manually after both are up.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    """Fresh-import ``mtg_bot.cli.serve``. Force-reload so each test
    gets independent module state."""
    src_dir = _REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if "mtg_bot.cli.serve" in sys.modules:
        del sys.modules["mtg_bot.cli.serve"]
    return importlib.import_module("mtg_bot.cli.serve")


def test_render_template_replaces_all_placeholders(tmp_path: Path):
    serve = _load_module()
    template = tmp_path / "tpl.sh"
    template.write_text(
        "REPO_URL=`{{REPO_URL}}`\n"
        "REPO_PATH=`{{REPO_PATH}}`\n"
        "MASTER_PUBKEY=`{{MASTER_PUBKEY}}`\n"
        "MASTER_TAILNET_NAME=`{{MASTER_TAILNET_NAME}}`\n",
        encoding="utf-8",
    )
    out = serve._render_template(
        template,
        repo_url="git@github.com:user/mtg-bot.git",
        repo_path="$HOME/projects/mtg-bot",
        master_pubkey="ssh-ed25519 AAAA... master-id",
        master_tailnet_name="desktop-j05412i",
    )
    assert "git@github.com:user/mtg-bot.git" in out
    assert "$HOME/projects/mtg-bot" in out
    assert "ssh-ed25519 AAAA... master-id" in out
    assert "desktop-j05412i" in out
    # No template markers should be left behind.
    assert "{{" not in out
    assert "}}" not in out


def test_setup_remote_sh_endpoint_renders_full_script(tmp_path: Path):
    """Smoke that hitting /setup_remote.sh returns a body containing
    both the template's static content AND the substituted values."""
    serve = _load_module()
    app = serve.create_app(
        repo_url="https://example.test/mtg-bot.git",
        master_pubkey="ssh-ed25519 KEY",
        master_tailnet_name="myhost",
        port=5099,
        template_dir=_REPO_ROOT / "scripts",
    )
    client = app.test_client()
    res = client.get("/setup_remote.sh")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    # Substituted values present.
    assert "https://example.test/mtg-bot.git" in body
    assert "ssh-ed25519 KEY" in body
    # Static skeleton lines from the template are still present.
    assert "set -euo pipefail" in body
    assert "python3.13" in body
    # No leftover template markers.
    assert "{{REPO_URL}}" not in body
    assert "{{MASTER_PUBKEY}}" not in body


def test_setup_remote_ps1_endpoint_renders_full_script(tmp_path: Path):
    serve = _load_module()
    app = serve.create_app(
        repo_url="https://example.test/mtg-bot.git",
        master_pubkey="ssh-ed25519 WINKEY",
        master_tailnet_name="myhost",
        port=5099,
        template_dir=_REPO_ROOT / "scripts",
    )
    client = app.test_client()
    res = client.get("/setup_remote.ps1")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "https://example.test/mtg-bot.git" in body
    assert "ssh-ed25519 WINKEY" in body
    # Static markers from the Windows template.
    assert "OpenSSH.Server" in body
    assert "icacls" in body
    assert "{{REPO_URL}}" not in body


def test_landing_page_lists_one_liners():
    serve = _load_module()
    app = serve.create_app(
        repo_url="https://example.test/mtg-bot.git",
        master_pubkey="ssh-ed25519 SHORTKEY",
        master_tailnet_name="myhost",
        port=5099,
        template_dir=_REPO_ROOT / "scripts",
    )
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "curl -sSL http://myhost:5099/setup_remote.sh | bash" in body
    assert "iwr http://myhost:5099/setup_remote.ps1" in body


def test_no_pubkey_warns_in_rendered_script(tmp_path: Path):
    """If the master's pubkey isn't set up, the served script should
    still work — it'll print a warning instead of substituting a
    real key. Doesn't break the rest of the install."""
    serve = _load_module()
    app = serve.create_app(
        repo_url="https://example.test/mtg-bot.git",
        master_pubkey="",  # missing
        master_tailnet_name="myhost",
        port=5099,
        template_dir=_REPO_ROOT / "scripts",
    )
    client = app.test_client()
    res = client.get("/setup_remote.sh")
    assert res.status_code == 200
    # The script's runtime check on MASTER_PUBKEY=="" path emits a
    # warning. Just confirm the rendered output is non-empty and
    # didn't break.
    assert "set -euo pipefail" in res.get_data(as_text=True)


# ---- /repo.bundle endpoint --------------------------------------


def _init_tiny_repo(tmp_path: Path) -> Path:
    import subprocess as sp
    repo = tmp_path / "fake-repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "--initial-branch=main"],
           cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@example"],
           cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README.md").write_text("hi", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def test_repo_bundle_endpoint_returns_valid_git_bundle(tmp_path: Path):
    """Hits /repo.bundle on a real git repo + verifies that git can
    clone from the returned bytes. Round-trips the full contract:
    master serves bundle → remote consumes bundle."""
    import subprocess as sp

    serve = _load_module()
    repo = _init_tiny_repo(tmp_path)

    app = serve.create_app(
        repo_url="(unused)",
        master_pubkey="",
        master_tailnet_name="m",
        port=5099,
        template_dir=_REPO_ROOT / "scripts",
        repo_root=repo,
    )
    client = app.test_client()
    res = client.get("/repo.bundle")
    assert res.status_code == 200
    assert res.mimetype == "application/x-git-bundle"
    # Persist + clone from it like a remote would.
    bundle = tmp_path / "repo.bundle"
    bundle.write_bytes(res.get_data())
    clone = tmp_path / "clone"
    sp.run(["git", "clone", "-q", str(bundle), str(clone)], check=True)
    assert (clone / "README.md").read_text(encoding="utf-8") == "hi"


def test_repo_bundle_endpoint_500s_when_repo_root_unconfigured():
    """If serve_setup_scripts is started without a repo_root, the
    bundle endpoint should return 500 rather than raise / silently
    misbehave."""
    serve = _load_module()
    app = serve.create_app(
        repo_url="(unused)",
        master_pubkey="",
        master_tailnet_name="m",
        port=5099,
        template_dir=_REPO_ROOT / "scripts",
        repo_root=None,
    )
    client = app.test_client()
    res = client.get("/repo.bundle")
    assert res.status_code == 500


def test_build_repo_bundle_helper_returns_nonempty_bytes(tmp_path: Path):
    """Direct unit-test of the helper used by the endpoint. Bundles
    are binary; no specific size to pin, but they have a stable
    magic prefix."""
    serve = _load_module()
    repo = _init_tiny_repo(tmp_path)
    data = serve.build_repo_bundle(repo)
    assert isinstance(data, bytes)
    assert len(data) > 0
    # Git bundle files start with one of these signatures.
    assert (
        data.startswith(b"# v2 git bundle\n")
        or data.startswith(b"# v3 git bundle\n")
    )


# ---- pull-model queue endpoints ---------------------------------


def _queue_app():
    serve = _load_module()
    app = serve.create_app(
        repo_url="(unused)", master_pubkey="",
        master_tailnet_name="m", port=5099,
        template_dir=_REPO_ROOT / "scripts",
        repo_root=None,
    )
    return app.test_client()


def test_queue_take_returns_410_when_stop_sentinel_present(tmp_path: Path):
    """Stop-button propagation to remotes: a master-side
    .STOP_<run_id> sentinel turns subsequent /queue/take into a 410
    Gone with ``stopped: True``. Remotes treat 410 the same as 404
    (drain + exit)."""
    serve = _load_module()
    app = serve.create_app(
        repo_url="(unused)", master_pubkey="",
        master_tailnet_name="m", port=5099,
        template_dir=_REPO_ROOT / "scripts",
        repo_root=None,
        progress_dir=tmp_path,
    )
    c = app.test_client()
    c.post("/queue/init", json={
        "run_id": "iter_001", "total_games": 10,
    })
    # Drop the sentinel.
    (tmp_path / ".STOP_iter_001").write_text("{}", encoding="utf-8")
    res = c.post("/queue/take", json={
        "run_id": "iter_001", "batch_size": 5,
    })
    assert res.status_code == 410
    body = res.get_json()
    assert body["stopped"] is True
    assert body["games"] == []


def test_queue_take_no_progress_dir_keeps_legacy_behavior():
    """Default ``progress_dir=None`` keeps the old behavior: stops
    are not even considered. Backward-compat for any caller that
    builds the app without plumbing a progress dir."""
    c = _queue_app()  # no progress_dir
    c.post("/queue/init", json={"run_id": "iter_001", "total_games": 5})
    res = c.post("/queue/take", json={
        "run_id": "iter_001", "batch_size": 5,
    })
    assert res.status_code == 200
    assert res.get_json()["games"] == [0, 1, 2, 3, 4]


def test_queue_init_then_take_advances_through_all_games():
    c = _queue_app()
    res = c.post("/queue/init", json={
        "run_id": "iter_001", "total_games": 10,
    })
    assert res.status_code == 200
    res = c.post("/queue/take", json={
        "run_id": "iter_001", "batch_size": 5,
    })
    body = res.get_json()
    assert body["games"] == [0, 1, 2, 3, 4]
    assert body["remaining"] == 5
    res = c.post("/queue/take", json={
        "run_id": "iter_001", "batch_size": 5,
    })
    body = res.get_json()
    assert body["games"] == [5, 6, 7, 8, 9]
    assert body["remaining"] == 0
    res = c.post("/queue/take", json={
        "run_id": "iter_001", "batch_size": 5,
    })
    body = res.get_json()
    assert body["games"] == []
    assert body["remaining"] == 0


def test_queue_take_with_partial_batch_at_end():
    c = _queue_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 7})
    c.post("/queue/take", json={"run_id": "r", "batch_size": 5})
    res = c.post("/queue/take", json={"run_id": "r", "batch_size": 5})
    body = res.get_json()
    assert body["games"] == [5, 6]
    assert body["remaining"] == 0


def test_queue_take_unknown_run_id_404():
    c = _queue_app()
    res = c.post("/queue/take", json={
        "run_id": "never-init", "batch_size": 5,
    })
    assert res.status_code == 404


def test_queue_init_validates_inputs():
    c = _queue_app()
    res = c.post("/queue/init", json={"total_games": 10})
    assert res.status_code == 400
    res = c.post("/queue/init", json={"run_id": "r"})
    assert res.status_code == 400
    res = c.post("/queue/init", json={"run_id": "r", "total_games": 0})
    assert res.status_code == 400


def test_queue_isolated_per_run_id():
    c = _queue_app()
    c.post("/queue/init", json={"run_id": "A", "total_games": 5})
    c.post("/queue/init", json={"run_id": "B", "total_games": 7})
    res = c.post("/queue/take", json={"run_id": "A", "batch_size": 3})
    assert res.get_json()["games"] == [0, 1, 2]
    res = c.post("/queue/take", json={"run_id": "B", "batch_size": 3})
    assert res.get_json()["games"] == [0, 1, 2]


def test_queue_status_reflects_state():
    c = _queue_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 10})
    c.post("/queue/take", json={"run_id": "r", "batch_size": 4})
    res = c.get("/queue/status")
    body = res.get_json()
    # Legacy take (no worker_id) leaves no inflight bookkeeping; the
    # core ``next/total/remaining/closed`` fields keep their pre-
    # lease semantics. The lease-related fields default to 0.
    assert body["queues"]["r"] == {
        "next": 4, "total": 10, "remaining": 6, "closed": False,
        "inflight": 0, "ready_reclaimed": 0, "failed": 0,
    }


def test_queue_close_makes_take_return_empty():
    """Early-stop: once /queue/close/<run_id> is POSTed, subsequent
    /queue/take calls for that run_id return an empty list with
    remaining=0 even though indices remain. Slack is abandoned —
    the orchestrator already has enough successful games."""
    c = _queue_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 10})
    res = c.post("/queue/take", json={"run_id": "r", "batch_size": 3})
    assert res.get_json()["games"] == [0, 1, 2]
    res = c.post("/queue/close/r")
    assert res.status_code == 200
    body = res.get_json()
    assert body["closed"] is True
    assert body["next"] == 3
    assert body["total"] == 10
    res = c.post("/queue/take", json={"run_id": "r", "batch_size": 3})
    assert res.status_code == 200
    body = res.get_json()
    assert body["games"] == []
    assert body["remaining"] == 0
    assert body["closed"] is True


def test_queue_close_unknown_run_id_404():
    """Closing a run_id that was never initialized returns 404 so
    the orchestrator detects a master/queue mismatch instead of
    silently no-op'ing."""
    c = _queue_app()
    res = c.post("/queue/close/never-init")
    assert res.status_code == 404


def test_queue_close_is_idempotent():
    """Closing twice is a no-op (no error). The orchestrator's
    poller may race with itself or retry on transient HTTP failures;
    both calls should succeed."""
    c = _queue_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    res1 = c.post("/queue/close/r")
    assert res1.status_code == 200
    assert res1.get_json()["closed"] is True
    res2 = c.post("/queue/close/r")
    assert res2.status_code == 200
    assert res2.get_json()["closed"] is True


def test_queue_close_isolated_per_run_id():
    """Closing run A must not close run B. Different iters share the
    same server; a self-play run-id finishing must not break a
    concurrent eval run-id."""
    c = _queue_app()
    c.post("/queue/init", json={"run_id": "A", "total_games": 5})
    c.post("/queue/init", json={"run_id": "B", "total_games": 5})
    c.post("/queue/close/A")
    res = c.post("/queue/take", json={"run_id": "A", "batch_size": 3})
    assert res.get_json()["games"] == []
    res = c.post("/queue/take", json={"run_id": "B", "batch_size": 3})
    assert res.get_json()["games"] == [0, 1, 2]


def test_queue_status_reports_closed_flag():
    """Diagnostic endpoint surfaces the closed flag so an operator
    can eyeball whether the master has signaled early-stop."""
    c = _queue_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 10})
    res = c.get("/queue/status")
    assert res.get_json()["queues"]["r"]["closed"] is False
    c.post("/queue/close/r")
    res = c.get("/queue/status")
    assert res.get_json()["queues"]["r"]["closed"] is True


def test_queue_concurrent_takes_no_overlap():
    """Threading correctness: 50 threads each grabbing batch=1 from
    a 50-total queue should get exactly 50 unique game-idxes
    (0..49, in some order). Pinned because the bug class is "two
    workers return the same game" which is silently wrong and
    doubles the policy gradient on those rows."""
    import concurrent.futures

    c = _queue_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 50})

    def take():
        return c.post(
            "/queue/take", json={"run_id": "r", "batch_size": 1},
        ).get_json()["games"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        results = list(ex.map(lambda _: take(), range(50)))

    flat = [g for batch in results for g in batch]
    assert sorted(flat) == list(range(50))


# ---- lease + heartbeat semantics --------------------------------
#
# Phase 1 of distributed-run robustness: a `/queue/take` request
# that includes a ``worker_id`` is tracked in-flight with a lease.
# Workers heartbeat to extend; if they vanish the lease expires and
# the next take call lazily reclaims the job. After ``max_attempts``
# reclaims a job lands in the per-queue ``failed`` set rather than
# looping forever.
#
# Tests below use a `FakeClock` so we can step time forward
# precisely; the real `time.monotonic` clock is injected via the
# ``monotonic`` kwarg on `create_app`.


class _FakeClock:
    """Step-controllable monotonic source. The clock value is in
    seconds (float); call ``advance(n)`` between requests to push
    leases over their TTL deterministically."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _lease_app(
    *,
    lease_ttl_sec: int = 60,
    max_attempts: int = 3,
    clock: _FakeClock = None,
):
    """Build a queue-only test app with the lease knobs exposed.
    Returns ``(client, clock)`` so each test can advance time."""
    serve = _load_module()
    if clock is None:
        clock = _FakeClock()
    app = serve.create_app(
        repo_url="(unused)", master_pubkey="",
        master_tailnet_name="m", port=5099,
        template_dir=_REPO_ROOT / "scripts",
        repo_root=None,
        lease_ttl_sec=lease_ttl_sec,
        max_attempts=max_attempts,
        monotonic=clock,
    )
    return app.test_client(), clock


def test_queue_take_with_worker_id_records_inflight_lease():
    """Including ``worker_id`` in the take request adds the handed-
    out idxes to the queue's in-flight tracker. The response carries
    ``lease_ttl_sec`` so the worker knows how often to heartbeat."""
    c, _ = _lease_app(lease_ttl_sec=60)
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 2, "worker_id": "shard-mac",
    })
    body = res.get_json()
    assert body["games"] == [0, 1]
    assert body["lease_ttl_sec"] == 60
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["inflight"] == 2


def test_queue_heartbeat_extends_lease_so_take_does_not_reclaim():
    """A worker that heartbeats inside the TTL keeps its job. The
    next `/queue/take` from a different worker must NOT see the
    job in the reclaimed pool."""
    clock = _FakeClock()
    c, _ = _lease_app(lease_ttl_sec=60, clock=clock)
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })
    # Half-TTL elapses; alice heartbeats.
    clock.advance(30)
    hb = c.post("/queue/heartbeat", json={
        "run_id": "r", "worker_id": "alice", "job_ids": [0],
    })
    assert hb.status_code == 200
    body = hb.get_json()
    assert body["extended"] == [0]
    assert body["rejected"] == []
    # Another half-TTL elapses (total 60s, but lease was extended
    # at 30s so it now expires at 90s — still 30s left).
    clock.advance(30)
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 5, "worker_id": "bob",
    })
    body = res.get_json()
    # Job 0 still owned by alice; bob gets fresh idxes only.
    assert body["games"] == [1, 2, 3, 4]


def test_queue_take_reclaims_expired_lease_on_next_take():
    """A worker disappears (no heartbeat). After TTL expires, the
    next take call re-hands-out the abandoned idxes — the bob client
    receives idx 0 even though alice was 'first'."""
    clock = _FakeClock()
    c, _ = _lease_app(lease_ttl_sec=60, clock=clock)
    c.post("/queue/init", json={"run_id": "r", "total_games": 3})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })
    # alice silently dies. TTL elapses.
    clock.advance(61)
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 5, "worker_id": "bob",
    })
    body = res.get_json()
    # Reclaimed idx 0 first, then fresh idxes 1, 2.
    assert body["games"] == [0, 1, 2]
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["inflight"] == 3
    assert status["queues"]["r"]["ready_reclaimed"] == 0


def test_queue_take_max_attempts_marks_job_failed():
    """A job that gets reclaimed ``max_attempts`` times lands in the
    ``failed`` bucket and is never retried. With max_attempts=2 the
    first take counts as attempt=1, the reclaim/take cycle pushes
    attempt=2, and the next reclaim pushes it to failed."""
    clock = _FakeClock()
    c, _ = _lease_app(lease_ttl_sec=10, max_attempts=2, clock=clock)
    c.post("/queue/init", json={"run_id": "r", "total_games": 1})
    # Attempt 1: alice gets idx 0, vanishes.
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })
    assert res.get_json()["games"] == [0]
    clock.advance(11)
    # Attempt 2: bob picks up idx 0, vanishes.
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "bob",
    })
    assert res.get_json()["games"] == [0]
    clock.advance(11)
    # Now: cap reached. Reclaim sweeps idx 0 into failed instead of
    # re-handing it out. Carol's take returns empty.
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "carol",
    })
    assert res.get_json()["games"] == []
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["inflight"] == 0
    assert status["queues"]["r"]["failed"] == 1
    assert status["queues"]["r"]["ready_reclaimed"] == 0


def test_queue_finish_removes_inflight_entry():
    """Calling /queue/finish with a held job removes it from the
    in-flight tracker. The lease bookkeeping doesn't grow forever."""
    c, _ = _lease_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 2, "worker_id": "alice",
    })
    res = c.post("/queue/finish", json={
        "run_id": "r", "worker_id": "alice", "job_ids": [0, 1],
    })
    assert res.status_code == 200
    body = res.get_json()
    assert body["finished"] == [0, 1]
    assert body["rejected"] == []
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["inflight"] == 0


def test_queue_finish_rejects_jobs_not_held_by_worker():
    """A worker can only finish jobs it actually holds. If alice's
    lease expired and bob now holds idx 0, alice's late
    `/queue/finish` for idx 0 must be rejected — otherwise alice
    could clobber a job bob is actively working on."""
    clock = _FakeClock()
    c, _ = _lease_app(lease_ttl_sec=10, clock=clock)
    c.post("/queue/init", json={"run_id": "r", "total_games": 2})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })
    clock.advance(11)
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "bob",
    })  # bob now holds idx 0
    res = c.post("/queue/finish", json={
        "run_id": "r", "worker_id": "alice", "job_ids": [0],
    })
    body = res.get_json()
    assert body["finished"] == []
    assert body["rejected"] == [0]


def test_queue_fail_releases_job_back_to_ready():
    """Voluntary fail re-queues a job *before* its lease expires so
    a clean-shutdown worker doesn't make the rest of the cluster
    wait for the TTL."""
    c, _ = _lease_app(max_attempts=3)
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })  # idx 0 → alice
    res = c.post("/queue/fail", json={
        "run_id": "r", "worker_id": "alice", "job_ids": [0],
    })
    body = res.get_json()
    assert body["requeued"] == [0]
    assert body["failed"] == []
    # Bob's next take pulls the freshly-requeued idx 0 ahead of new
    # idxes — reclaimed-priority semantics apply to voluntary fail
    # too.
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 2, "worker_id": "bob",
    })
    assert res.get_json()["games"] == [0, 1]


def test_queue_fail_at_max_attempts_marks_job_failed():
    """Voluntary fail when attempts are already at max should NOT
    re-queue the job (otherwise a misbehaving worker could
    indefinitely cycle a poison-pill job)."""
    c, _ = _lease_app(max_attempts=1)
    c.post("/queue/init", json={"run_id": "r", "total_games": 1})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })  # attempts=1 (== max_attempts)
    res = c.post("/queue/fail", json={
        "run_id": "r", "worker_id": "alice", "job_ids": [0],
    })
    body = res.get_json()
    assert body["requeued"] == []
    assert body["failed"] == [0]
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["failed"] == 1


def test_queue_heartbeat_rejects_when_lease_already_reclaimed():
    """A worker whose lease has already expired and been reclaimed
    receives a 'rejected' response when it heartbeats — the worker
    can use this to drop the job from its local tracking and avoid
    double-finishing."""
    clock = _FakeClock()
    c, _ = _lease_app(lease_ttl_sec=10, clock=clock)
    c.post("/queue/init", json={"run_id": "r", "total_games": 2})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })
    clock.advance(11)
    # bob takes — alice's idx 0 gets reclaimed and handed to bob.
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "bob",
    })
    # alice belatedly heartbeats: she lost the lease.
    res = c.post("/queue/heartbeat", json={
        "run_id": "r", "worker_id": "alice", "job_ids": [0],
    })
    body = res.get_json()
    assert body["extended"] == []
    assert body["rejected"] == [0]


def test_queue_heartbeat_validation():
    """Missing fields / unknown run_id are validated."""
    c, _ = _lease_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    res = c.post("/queue/heartbeat", json={"run_id": "r"})
    assert res.status_code == 400
    res = c.post("/queue/heartbeat", json={
        "run_id": "r", "worker_id": "x",
    })
    assert res.status_code == 400  # job_ids missing
    res = c.post("/queue/heartbeat", json={
        "run_id": "nope", "worker_id": "x", "job_ids": [],
    })
    assert res.status_code == 404


def test_queue_finish_validation():
    c, _ = _lease_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    res = c.post("/queue/finish", json={"run_id": "r"})
    assert res.status_code == 400
    res = c.post("/queue/finish", json={
        "run_id": "nope", "worker_id": "x", "job_ids": [],
    })
    assert res.status_code == 404


def test_queue_fail_validation():
    c, _ = _lease_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 5})
    res = c.post("/queue/fail", json={"run_id": "r"})
    assert res.status_code == 400
    res = c.post("/queue/fail", json={
        "run_id": "nope", "worker_id": "x", "job_ids": [],
    })
    assert res.status_code == 404


def test_queue_fail_rejects_jobs_not_held_by_worker():
    """Same protection as ``/queue/finish``: a worker can only fail
    jobs it actually holds. Fail-on-someone-else's-job is rejected so
    a stale worker can't cause its successor's job to bounce around."""
    c, _ = _lease_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 2})
    c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "alice",
    })  # alice holds idx 0
    res = c.post("/queue/fail", json={
        "run_id": "r", "worker_id": "ghost", "job_ids": [0, 99],
    })
    body = res.get_json()
    assert body["requeued"] == []
    assert body["failed"] == []
    assert body["rejected"] == [0, 99]


def test_queue_take_legacy_no_worker_id_unchanged():
    """The pre-PR shape (no ``worker_id``) keeps working: response
    has ``games`` + ``remaining`` + (now also) ``lease_ttl_sec``,
    and the queue's inflight tracker stays empty so the legacy
    take-and-forget contract is preserved bit-for-bit for callers
    that haven't been ported."""
    c, _ = _lease_app()
    c.post("/queue/init", json={"run_id": "r", "total_games": 3})
    res = c.post("/queue/take", json={"run_id": "r", "batch_size": 5})
    body = res.get_json()
    assert body["games"] == [0, 1, 2]
    assert body["remaining"] == 0
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["inflight"] == 0


def test_queue_concurrent_claim_and_heartbeat_safe():
    """Threading correctness for the lease path: many workers
    concurrently take + heartbeat + finish should never produce
    overlapping claims, and the in-flight count should converge to
    zero after all finishes complete."""
    import concurrent.futures

    c, clock = _lease_app(lease_ttl_sec=600)  # large TTL to avoid
    # spurious reclaims under the executor's scheduling jitter.
    c.post("/queue/init", json={"run_id": "r", "total_games": 100})

    def worker(wid: int) -> list[int]:
        my_id = f"w{wid}"
        got: list[int] = []
        for _ in range(5):
            res = c.post("/queue/take", json={
                "run_id": "r", "batch_size": 1, "worker_id": my_id,
            }).get_json()
            if not res["games"]:
                break
            idx = res["games"][0]
            got.append(idx)
            # Simulate a heartbeat mid-job.
            c.post("/queue/heartbeat", json={
                "run_id": "r", "worker_id": my_id, "job_ids": [idx],
            })
            c.post("/queue/finish", json={
                "run_id": "r", "worker_id": my_id, "job_ids": [idx],
            })
        return got

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(worker, range(20)))
    flat = [g for r in results for g in r]
    assert sorted(flat) == list(range(100))
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["inflight"] == 0
    assert status["queues"]["r"]["failed"] == 0


def test_queue_reclaim_increments_attempts_counter():
    """An idx that gets reclaimed twice carries attempts=3 on its
    third claim. Verified by leveraging the max_attempts cap as a
    counter probe: with max_attempts=3, idx 0 should still be
    recoverable on its 3rd claim and only fail on the 4th sweep."""
    clock = _FakeClock()
    c, _ = _lease_app(lease_ttl_sec=10, max_attempts=3, clock=clock)
    c.post("/queue/init", json={"run_id": "r", "total_games": 1})
    for wid in ("a", "b", "c"):
        res = c.post("/queue/take", json={
            "run_id": "r", "batch_size": 1, "worker_id": wid,
        })
        assert res.get_json()["games"] == [0]
        clock.advance(11)
    # 3 takes done, 3 reclaim sweeps will have happened on each take
    # past the first; at this point idx 0 is in-flight to 'c' with
    # attempts=3 (== max). Next reclaim moves it to failed.
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "d",
    })
    assert res.get_json()["games"] == []
    status = c.get("/queue/status").get_json()
    assert status["queues"]["r"]["failed"] == 1


def test_queue_take_default_lease_ttl():
    """``create_app`` without explicit ``lease_ttl_sec`` falls back
    to the module default. Pin the value so a default change is
    visible in test output."""
    serve = _load_module()
    app = serve.create_app(
        repo_url="(unused)", master_pubkey="",
        master_tailnet_name="m", port=5099,
        template_dir=_REPO_ROOT / "scripts",
        repo_root=None,
    )
    c = app.test_client()
    c.post("/queue/init", json={"run_id": "r", "total_games": 1})
    res = c.post("/queue/take", json={
        "run_id": "r", "batch_size": 1, "worker_id": "x",
    })
    assert res.get_json()["lease_ttl_sec"] == serve.DEFAULT_LEASE_TTL_SEC
    assert serve.DEFAULT_LEASE_TTL_SEC == 120
    assert serve.DEFAULT_MAX_ATTEMPTS == 3
