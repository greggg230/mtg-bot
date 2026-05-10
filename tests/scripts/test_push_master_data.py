"""Tests for ``scripts/push_master_data.py``.

The actual scp/ssh invocations need a real remote, so we test:
- shard resolution: explicit flags vs --shards-config + --shard
- OS detection from venv_python
- Missing-file detection (sys.exit before any network call)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "push_master_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "push_master_data", _SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_shards(tmp_path: Path, shards: list[dict]) -> Path:
    p = tmp_path / "shards.json"
    p.write_text(json.dumps({"shards": shards}), encoding="utf-8")
    return p


def test_resolve_shard_from_config_picks_correct_remote(
    tmp_path: Path,
):
    """--shards-config + --shard pulls host/user/repo_path from the
    matching shard. Windows is detected via .exe suffix on
    venv_python."""
    push = _load_module()
    p = _write_shards(tmp_path, [
        {"host": "local"},
        {
            "host": "win-1080ti",
            "user": "gregg",
            "repo_path": "C:\\Users\\gregg\\Projects\\mtg-bot",
            "venv_python": ".venv-py313\\Scripts\\python.exe",
            "ssh_port": 2222,
        },
        {
            "host": "macbook",
            "user": "gregg",
            "repo_path": "/Users/gregg/projects/mtg-bot",
            "venv_python": ".venv-py313/bin/python",
        },
    ])

    host, user, repo_path, is_windows, ssh_port = push._resolve_shard(
        p, "win-1080ti",
        cli_host=None, cli_user=None, cli_repo_path=None,
    )
    assert host == "win-1080ti"
    assert user == "gregg"
    assert repo_path == "C:\\Users\\gregg\\Projects\\mtg-bot"
    assert is_windows is True
    assert ssh_port == 2222

    host, user, repo_path, is_windows, ssh_port = push._resolve_shard(
        p, "macbook",
        cli_host=None, cli_user=None, cli_repo_path=None,
    )
    assert host == "macbook"
    assert is_windows is False
    assert ssh_port == 22  # default


def test_resolve_shard_local_rejected(tmp_path: Path):
    """Pushing to a local shard makes no sense — files are already
    on the master. Surface that explicitly."""
    push = _load_module()
    p = _write_shards(tmp_path, [
        {"host": "local"},
    ])
    with pytest.raises(SystemExit, match="local"):
        push._resolve_shard(
            p, "local",
            cli_host=None, cli_user=None, cli_repo_path=None,
        )


def test_resolve_shard_unknown_name_rejected(tmp_path: Path):
    push = _load_module()
    p = _write_shards(tmp_path, [
        {
            "host": "macbook",
            "user": "u", "repo_path": "/r",
            "venv_python": ".venv/bin/python",
        },
    ])
    with pytest.raises(SystemExit, match="no shard"):
        push._resolve_shard(
            p, "win-1080ti",
            cli_host=None, cli_user=None, cli_repo_path=None,
        )


def test_resolve_shard_explicit_flags_path():
    push = _load_module()
    host, user, repo_path, is_windows, ssh_port = push._resolve_shard(
        None, None,
        cli_host="macbook", cli_user="u",
        cli_repo_path="/r",
    )
    assert host == "macbook"
    assert is_windows is False  # POSIX-style path
    assert ssh_port == 22


def test_resolve_shard_explicit_flags_windows_path():
    """Heuristic OS-detection from explicit --repo-path: backslashes
    → Windows."""
    push = _load_module()
    _, _, _, is_windows, _ = push._resolve_shard(
        None, None,
        cli_host="winbox", cli_user="u",
        cli_repo_path="C:\\Users\\u\\repo",
    )
    assert is_windows is True


def test_resolve_shard_missing_flags_rejected():
    push = _load_module()
    with pytest.raises(SystemExit, match="missing required flags"):
        push._resolve_shard(
            None, None,
            cli_host=None, cli_user=None, cli_repo_path=None,
        )
    with pytest.raises(SystemExit, match="missing required flags"):
        push._resolve_shard(
            None, None,
            cli_host="x", cli_user=None, cli_repo_path="/r",
        )


def test_push_to_remote_aborts_on_missing_local_file(
    tmp_path: Path, monkeypatch,
):
    """If the local weights or deck pool isn't on disk, abort
    BEFORE any ssh/scp call. Don't surprise the user with a
    half-finished push."""
    push = _load_module()
    # Capture ssh/scp invocations to make sure none fire.
    calls = []
    monkeypatch.setattr(push.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(SystemExit, match="missing local file"):
        push.push_to_remote(
            user="u", host="h", repo_path="/r",
            is_windows=False,
            registry_root=tmp_path / "no_such_registry",
        )
    assert calls == []
