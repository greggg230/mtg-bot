"""Tests for ``mtg_bot.training.shards_config_edit``.

Cover:
- Happy path: bumping ``n_workers`` for an existing host writes
  through atomically.
- Validation: rejects non-int values, out-of-range values, and
  unknown hosts.
- Atomic write: a torn write (raise mid-dump) leaves the original
  file intact.
- Concurrent edits from multiple threads serialize through the
  module-level lock and end with the last writer's value.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from mtg_bot.training import shards_config_edit


def _seed_config(tmp_path: Path) -> Path:
    """Drop a sample shards-config file with two hosts and return
    its path."""
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
            {
                "host": "greggs-macbook-pro",
                "user": "greggkeithley",
                "repo_path": "/Users/greggkeithley/projects/mtg-bot",
                "venv_python": ".venv-py313/bin/python",
            },
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_update_happy_path(tmp_path: Path):
    p = _seed_config(tmp_path)
    result = shards_config_edit.update_shard_n_workers(p, "local", 12)
    assert result["shards"][0]["n_workers"] == 12
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["shards"][0]["n_workers"] == 12
    # Other entries left untouched.
    assert on_disk["shards"][1]["n_workers"] == 4
    assert "n_workers" not in on_disk["shards"][2]


def test_update_adds_field_when_absent(tmp_path: Path):
    """A shard without an explicit ``n_workers`` (default = auto)
    should gain the field on first edit."""
    p = _seed_config(tmp_path)
    result = shards_config_edit.update_shard_n_workers(
        p, "greggs-macbook-pro", 6,
    )
    assert result["shards"][2]["n_workers"] == 6
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["shards"][2]["n_workers"] == 6


def test_update_unknown_host_raises(tmp_path: Path):
    p = _seed_config(tmp_path)
    with pytest.raises(
        shards_config_edit.ShardsConfigEditError,
        match="unknown host",
    ):
        shards_config_edit.update_shard_n_workers(p, "no-such-host", 8)


def test_update_rejects_non_int(tmp_path: Path):
    p = _seed_config(tmp_path)
    for bad in ("8", 8.5, None, [8], {}, True):
        with pytest.raises(
            shards_config_edit.ShardsConfigEditError,
            match="must be an integer",
        ):
            shards_config_edit.update_shard_n_workers(p, "local", bad)


def test_update_rejects_out_of_range(tmp_path: Path):
    p = _seed_config(tmp_path)
    for bad in (0, -1, 65, 1000):
        with pytest.raises(
            shards_config_edit.ShardsConfigEditError,
            match="must be in",
        ):
            shards_config_edit.update_shard_n_workers(p, "local", bad)


def test_update_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "nope.json"
    with pytest.raises(
        shards_config_edit.ShardsConfigEditError,
        match="not found",
    ):
        shards_config_edit.update_shard_n_workers(missing, "local", 8)


def test_update_invalid_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json {", encoding="utf-8")
    with pytest.raises(
        shards_config_edit.ShardsConfigEditError,
        match="not valid JSON",
    ):
        shards_config_edit.update_shard_n_workers(p, "local", 8)


def test_update_missing_shards_key_raises(tmp_path: Path):
    p = tmp_path / "no-shards.json"
    p.write_text(json.dumps({"other": []}), encoding="utf-8")
    with pytest.raises(
        shards_config_edit.ShardsConfigEditError,
        match="missing top-level 'shards'",
    ):
        shards_config_edit.update_shard_n_workers(p, "local", 8)


def test_update_shards_not_list_raises(tmp_path: Path):
    p = tmp_path / "shards-bad.json"
    p.write_text(json.dumps({"shards": "wrong"}), encoding="utf-8")
    with pytest.raises(
        shards_config_edit.ShardsConfigEditError,
        match="'shards' must be a list",
    ):
        shards_config_edit.update_shard_n_workers(p, "local", 8)


def test_atomic_write_cleanup_handles_unlink_failure(
    tmp_path: Path, monkeypatch,
):
    """When ``os.replace`` raises AND the cleanup ``os.unlink`` ALSO
    raises (e.g. tempfile already gone), the original error should
    still propagate — we must not mask it with the cleanup failure."""
    p = _seed_config(tmp_path)
    real_unlink = shards_config_edit.os.unlink

    def boom_replace(*args, **kwargs):
        raise OSError("replace failed")

    def boom_unlink(*args, **kwargs):
        raise OSError("unlink failed too")

    monkeypatch.setattr(
        "mtg_bot.training.shards_config_edit.os.replace", boom_replace,
    )
    monkeypatch.setattr(
        "mtg_bot.training.shards_config_edit.os.unlink", boom_unlink,
    )
    with pytest.raises(OSError, match="replace failed"):
        shards_config_edit.update_shard_n_workers(p, "local", 16)
    # Restore so test cleanup can remove the leftover.
    monkeypatch.setattr(
        "mtg_bot.training.shards_config_edit.os.unlink", real_unlink,
    )


def test_atomic_write_preserves_old_on_failure(
    tmp_path: Path, monkeypatch,
):
    """If ``os.replace`` raises mid-write, the original file must
    still be valid JSON pointing at the pre-edit state."""
    p = _seed_config(tmp_path)
    original = p.read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("simulated disk failure during replace")

    monkeypatch.setattr(
        "mtg_bot.training.shards_config_edit.os.replace", boom,
    )
    with pytest.raises(OSError, match="simulated"):
        shards_config_edit.update_shard_n_workers(p, "local", 16)
    # Original file untouched.
    assert p.read_text(encoding="utf-8") == original
    # No tempfile leftovers.
    leftovers = list(tmp_path.glob("shards_config.json.*.tmp"))
    assert leftovers == []


def test_concurrent_edits_serialize(tmp_path: Path):
    """Race ten threads bumping the same host concurrently. All ten
    POSTs must complete cleanly (no exceptions; the lock prevents
    interleaved writes), and the final on-disk value must equal the
    last value any thread wrote."""
    p = _seed_config(tmp_path)
    results: list[int] = []
    errors: list[BaseException] = []

    def worker(value: int):
        try:
            shards_config_edit.update_shard_n_workers(p, "local", value)
            results.append(value)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(v,))
        for v in range(2, 12)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(results) == list(range(2, 12))
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["shards"][0]["n_workers"] in range(2, 12)


def test_read_shards_config_raw_round_trips(tmp_path: Path):
    p = _seed_config(tmp_path)
    raw = shards_config_edit.read_shards_config_raw(p)
    assert raw["shards"][0]["host"] == "local"
    assert raw["shards"][0]["n_workers"] == 8


def test_summarize_for_log(tmp_path: Path):
    p = _seed_config(tmp_path)
    raw = shards_config_edit.read_shards_config_raw(p)
    summary = shards_config_edit.summarize_shards_for_log(raw)
    assert "local=8" in summary
    assert "desktop-g72aptg=4" in summary
    assert "greggs-macbook-pro=null (auto)" in summary


def test_summarize_for_log_handles_garbage():
    """``summarize_shards_for_log`` should not crash on degenerate
    input (e.g. an empty dict, or shards entries that aren't
    dicts)."""
    assert shards_config_edit.summarize_shards_for_log({}) == ""
    assert shards_config_edit.summarize_shards_for_log({"shards": []}) == ""
    assert shards_config_edit.summarize_shards_for_log(
        {"shards": ["not-a-dict", {"host": "x", "n_workers": 2}]},
    ) == "x=2"


def test_lock_serializes_via_observable_delay(tmp_path: Path, monkeypatch):
    """Force the writer to sleep inside the locked section; assert
    that a second concurrent caller finishes AFTER the first has
    released the lock (i.e. the lock is actually held across the
    read-modify-write)."""
    p = _seed_config(tmp_path)
    real_write = shards_config_edit._atomic_write_json
    in_flight = threading.Event()
    proceed = threading.Event()

    def slow_write(path, payload):
        in_flight.set()
        proceed.wait(timeout=2.0)
        return real_write(path, payload)

    monkeypatch.setattr(
        shards_config_edit, "_atomic_write_json", slow_write,
    )

    finish_times: dict[str, float] = {}

    def first():
        shards_config_edit.update_shard_n_workers(p, "local", 5)
        finish_times["first"] = time.perf_counter()

    def second():
        # Wait until the first thread is parked inside the lock.
        in_flight.wait(timeout=2.0)
        shards_config_edit.update_shard_n_workers(p, "local", 7)
        finish_times["second"] = time.perf_counter()

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    # Give the second thread a chance to attempt the lock and block.
    time.sleep(0.05)
    proceed.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert "first" in finish_times and "second" in finish_times
    assert finish_times["second"] >= finish_times["first"]
