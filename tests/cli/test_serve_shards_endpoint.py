"""Tests for the dynamic worker scaling REST endpoints in
``mtg_bot.cli.serve.create_app``:

- ``GET /api/shards`` returns the current shards-config JSON.
- ``POST /api/shards/<host>/workers`` mutates ``n_workers`` for one
  shard, write-through to the on-disk JSON.

These endpoints are the network mirror of
``mtg_bot.training.shards_config_edit``; cover the HTTP surface
(status codes + body shape + write-through to disk) here. The
underlying lock + validation invariants are covered by
``tests/training/test_shards_config_edit.py``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from mtg_bot.cli import serve as serve_mod


def _seed_config(tmp_path: Path) -> Path:
    p = tmp_path / "shards_config.json"
    payload = {
        "shards": [
            {"host": "local", "n_workers": 8},
            {
                "host": "desktop-g72aptg",
                "user": "gregg",
                "repo_path": "C:/X",
                "venv_python": ".venv-py313/Scripts/python.exe",
                "n_workers": 4,
            },
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _make_app(*, shards_config_path):
    return serve_mod.create_app(
        repo_url="https://example.test/mtg-bot.git",
        master_pubkey="ssh-ed25519 KEY",
        master_tailnet_name="myhost",
        port=5099,
        shards_config_path=shards_config_path,
    )


def test_get_shards_returns_current_config(tmp_path: Path):
    p = _seed_config(tmp_path)
    app = _make_app(shards_config_path=p)
    client = app.test_client()
    res = client.get("/api/shards")
    assert res.status_code == 200
    body = res.get_json()
    assert body["shards"][0]["host"] == "local"
    assert body["shards"][0]["n_workers"] == 8


def test_get_shards_404_when_disabled(tmp_path: Path):
    app = _make_app(shards_config_path=None)
    client = app.test_client()
    res = client.get("/api/shards")
    assert res.status_code == 404
    assert "not configured" in res.get_json()["error"]


def test_get_shards_404_when_file_missing(tmp_path: Path):
    app = _make_app(shards_config_path=tmp_path / "nope.json")
    client = app.test_client()
    res = client.get("/api/shards")
    assert res.status_code == 404
    assert "not found" in res.get_json()["error"]


def test_get_shards_500_on_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json {", encoding="utf-8")
    app = _make_app(shards_config_path=p)
    client = app.test_client()
    res = client.get("/api/shards")
    assert res.status_code == 500


def test_post_workers_happy_path(tmp_path: Path):
    p = _seed_config(tmp_path)
    app = _make_app(shards_config_path=p)
    client = app.test_client()
    res = client.post(
        "/api/shards/local/workers",
        json={"n_workers": 16},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body == {
        "ok": True,
        "host": "local",
        "n_workers": 16,
        "shards": json.loads(p.read_text(encoding="utf-8"))["shards"],
    }
    # Write-through: the file on disk reflects the new value.
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["shards"][0]["n_workers"] == 16


def test_post_workers_404_when_disabled(tmp_path: Path):
    app = _make_app(shards_config_path=None)
    client = app.test_client()
    res = client.post(
        "/api/shards/local/workers", json={"n_workers": 8},
    )
    assert res.status_code == 404


def test_post_workers_400_missing_field(tmp_path: Path):
    p = _seed_config(tmp_path)
    app = _make_app(shards_config_path=p)
    client = app.test_client()
    res = client.post("/api/shards/local/workers", json={})
    assert res.status_code == 400
    assert "n_workers required" in res.get_json()["error"]


@pytest.mark.parametrize("bad_value", [0, -1, 65, 1000])
def test_post_workers_400_out_of_range(tmp_path: Path, bad_value):
    p = _seed_config(tmp_path)
    app = _make_app(shards_config_path=p)
    client = app.test_client()
    res = client.post(
        "/api/shards/local/workers", json={"n_workers": bad_value},
    )
    assert res.status_code == 400
    assert "must be in" in res.get_json()["error"]


@pytest.mark.parametrize("bad_value", ["8", 8.5, None, [8], {}])
def test_post_workers_400_non_int(tmp_path: Path, bad_value):
    p = _seed_config(tmp_path)
    app = _make_app(shards_config_path=p)
    client = app.test_client()
    res = client.post(
        "/api/shards/local/workers", json={"n_workers": bad_value},
    )
    assert res.status_code == 400
    assert "integer" in res.get_json()["error"]


def test_post_workers_404_unknown_host(tmp_path: Path):
    p = _seed_config(tmp_path)
    app = _make_app(shards_config_path=p)
    client = app.test_client()
    res = client.post(
        "/api/shards/no-such-host/workers", json={"n_workers": 8},
    )
    assert res.status_code == 404
    assert "unknown host" in res.get_json()["error"]


def test_post_workers_concurrent_writes(tmp_path: Path):
    """Hit the endpoint from many threads at once; all responses
    should be 200 (the lock prevents interleaving) and the on-disk
    file should end up at one of the values that was POSTed."""
    p = _seed_config(tmp_path)
    app = _make_app(shards_config_path=p)
    statuses: list[int] = []
    lock = threading.Lock()

    def worker(value: int):
        client = app.test_client()
        res = client.post(
            "/api/shards/local/workers", json={"n_workers": value},
        )
        with lock:
            statuses.append(res.status_code)

    threads = [
        threading.Thread(target=worker, args=(v,))
        for v in range(2, 10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(s == 200 for s in statuses), statuses
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["shards"][0]["n_workers"] in range(2, 10)
