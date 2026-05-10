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
    assert body["queues"]["r"] == {
        "next": 4, "total": 10, "remaining": 6, "closed": False,
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
