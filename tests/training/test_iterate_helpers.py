"""Tests for ``mtg_bot.training._iterate_helpers``.

Today the module hosts:

- ``find_foreign_npzs`` — safety check
  ``scripts/iterate_alphazero.py`` runs at startup to refuse to
  feed non-iterate npzs into the training step. The bug it
  prevents took out v7 in the first deck-pool iterate run:
  ``gen0_smoke.npz`` had been sitting in ``training_data/`` from
  a much older debug run, and its 603 mask-mismatched rows blew
  up policy CE when it got co-globbed with the iter_001 npz.
- ``refresh_shards_config_for_iter`` — re-read shards_config.json
  at the top of each iter so dynamic worker-scale changes from
  the training UI take effect on the upcoming pool init.
"""

from __future__ import annotations

import json
from pathlib import Path

from mtg_bot.training._iterate_helpers import (
    find_foreign_npzs, refresh_shards_config_for_iter,
)


def test_finds_foreign_npzs(tmp_path: Path):
    (tmp_path / "iter_001_v6_20260503T213931Z.npz").write_bytes(b"")
    (tmp_path / "iter_002_v7_20260503T230000Z.npz").write_bytes(b"")
    (tmp_path / "gen0_smoke.npz").write_bytes(b"")  # ← foreign
    (tmp_path / "run01.npz").write_bytes(b"")        # ← foreign
    (tmp_path / "notes.txt").write_text("not a foreign npz")
    found = find_foreign_npzs(tmp_path)
    assert found == ["gen0_smoke.npz", "run01.npz"]


def test_clean_dir_returns_empty(tmp_path: Path):
    (tmp_path / "iter_001_v6_20260503T213931Z.npz").write_bytes(b"")
    (tmp_path / "iter_002_v7_20260503T230000Z.npz").write_bytes(b"")
    (tmp_path / "metadata.json").write_text("{}")
    assert find_foreign_npzs(tmp_path) == []


def test_empty_dir_returns_empty(tmp_path: Path):
    """No npzs at all (typical first-iter setup) → no false positives."""
    assert find_foreign_npzs(tmp_path) == []


def test_missing_dir_returns_empty(tmp_path: Path):
    """A dir that doesn't exist (e.g. iterate's first run, before
    mkdir) returns empty rather than raising — caller handles the
    later mkdir."""
    nonexistent = tmp_path / "no_such_dir"
    assert find_foreign_npzs(nonexistent) == []


def test_pattern_match_is_strict_prefix_only(tmp_path: Path):
    """Only ``iter_`` prefix qualifies as iterate-owned. Files that
    contain 'iter' elsewhere in the name are still foreign."""
    (tmp_path / "iter_001_v6.npz").write_bytes(b"")
    (tmp_path / "my_iter_run.npz").write_bytes(b"")  # ← foreign
    (tmp_path / "ITER_005_capitalized.npz").write_bytes(b"")  # ← foreign
    found = find_foreign_npzs(tmp_path)
    assert "my_iter_run.npz" in found
    assert "ITER_005_capitalized.npz" in found
    assert "iter_001_v6.npz" not in found


# ---- refresh_shards_config_for_iter ---------------------------------


def _seed_shards_config(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_refresh_returns_none_when_path_is_none():
    """No --shards-config flag → no distributed pipeline → refresh
    is a no-op returning None."""
    result = refresh_shards_config_for_iter(
        config_path=None,
        previous=None,
        iter_label="iter_001",
        print_fn=lambda *a, **k: None,
    )
    assert result is None


def test_refresh_picks_up_disk_change(tmp_path: Path):
    """The core invariant: a value mutated on disk between iters
    shows up in the freshly-loaded config."""
    p = tmp_path / "shards_config.json"
    _seed_shards_config(p, {
        "shards": [
            {"host": "local", "n_workers": 8},
        ],
    })
    first = refresh_shards_config_for_iter(
        config_path=p, previous=None, iter_label="iter_001",
        print_fn=lambda *a, **k: None,
    )
    assert first is not None
    assert first.shards[0].n_workers == 8

    # Simulate a UI-driven POST that bumped the value to 12.
    _seed_shards_config(p, {
        "shards": [
            {"host": "local", "n_workers": 12},
        ],
    })
    second = refresh_shards_config_for_iter(
        config_path=p, previous=first, iter_label="iter_002",
        print_fn=lambda *a, **k: None,
    )
    assert second is not None
    assert second.shards[0].n_workers == 12


def test_refresh_logs_summary(tmp_path: Path):
    """The one-liner the operator greps for must include each shard's
    host=value form."""
    p = tmp_path / "shards_config.json"
    _seed_shards_config(p, {
        "shards": [
            {"host": "local", "n_workers": 8},
            {
                "host": "macbook", "user": "u", "repo_path": "/r",
                "venv_python": ".venv/bin/python",
            },
            {
                "host": "win10", "user": "u", "repo_path": "C:/r",
                "venv_python": ".venv/Scripts/python.exe",
                "n_workers": 4,
            },
        ],
    })
    captured = []

    def capture(msg, *a, **k):
        captured.append(msg)

    refresh_shards_config_for_iter(
        config_path=p, previous=None,
        iter_label="iter_007", print_fn=capture,
    )
    assert any("iter_007" in m for m in captured)
    summary_line = next(m for m in captured if "iter_007" in m)
    assert "local=8" in summary_line
    assert "macbook=null (auto)" in summary_line
    assert "win10=4" in summary_line


def test_refresh_warns_on_added_or_removed_shard(tmp_path: Path):
    """Shard ENTRIES coming or going should produce a stderr
    warning but the refresh should still honor the new config."""
    p = tmp_path / "shards_config.json"
    _seed_shards_config(p, {
        "shards": [
            {"host": "local", "n_workers": 8},
            {
                "host": "win10", "user": "u", "repo_path": "C:/r",
                "venv_python": ".venv/Scripts/python.exe",
                "n_workers": 4,
            },
        ],
    })
    first = refresh_shards_config_for_iter(
        config_path=p, previous=None, iter_label="iter_001",
        print_fn=lambda *a, **k: None,
    )

    # Replace win10 with a brand-new mac shard.
    _seed_shards_config(p, {
        "shards": [
            {"host": "local", "n_workers": 8},
            {
                "host": "macbook", "user": "u", "repo_path": "/r",
                "venv_python": ".venv/bin/python",
                "n_workers": 6,
            },
        ],
    })
    captured = []

    def capture(msg, *a, **k):
        captured.append(msg)

    second = refresh_shards_config_for_iter(
        config_path=p, previous=first, iter_label="iter_002",
        print_fn=capture,
    )
    # Honored the new config.
    new_hosts = {s.host for s in second.shards}
    assert new_hosts == {"local", "macbook"}
    # And warned.
    warning = next(m for m in captured if "WARNING" in m)
    assert "added=['macbook']" in warning
    assert "removed=['win10']" in warning


def test_refresh_keeps_previous_on_load_error(tmp_path: Path):
    """A torn JSON write or transient file-missing must not crash
    the iter — the function logs and returns the previous config."""
    p = tmp_path / "shards_config.json"
    _seed_shards_config(p, {
        "shards": [{"host": "local", "n_workers": 8}],
    })
    first = refresh_shards_config_for_iter(
        config_path=p, previous=None, iter_label="iter_001",
        print_fn=lambda *a, **k: None,
    )
    # Now break the file mid-iter.
    p.write_text("{ partially-written ...", encoding="utf-8")
    captured = []

    def capture(msg, *a, **k):
        captured.append(msg)

    second = refresh_shards_config_for_iter(
        config_path=p, previous=first, iter_label="iter_002",
        print_fn=capture,
    )
    # Returned previous unchanged.
    assert second is first
    # And logged the failure.
    assert any("failed to refresh" in m for m in captured)


def test_refresh_keeps_previous_on_missing_file(tmp_path: Path):
    """Same fall-through path for FileNotFoundError specifically."""
    p = tmp_path / "shards_config.json"
    _seed_shards_config(p, {
        "shards": [{"host": "local", "n_workers": 8}],
    })
    first = refresh_shards_config_for_iter(
        config_path=p, previous=None, iter_label="iter_001",
        print_fn=lambda *a, **k: None,
    )
    p.unlink()
    captured = []

    def capture(msg, *a, **k):
        captured.append(msg)

    second = refresh_shards_config_for_iter(
        config_path=p, previous=first, iter_label="iter_002",
        print_fn=capture,
    )
    assert second is first
    assert any("failed to refresh" in m for m in captured)


def test_refresh_no_warning_when_only_n_workers_changes(tmp_path: Path):
    """The expected hot-path: bump n_workers without touching the
    shard set should NOT emit the entry-set-changed warning."""
    p = tmp_path / "shards_config.json"
    _seed_shards_config(p, {
        "shards": [{"host": "local", "n_workers": 8}],
    })
    first = refresh_shards_config_for_iter(
        config_path=p, previous=None, iter_label="iter_001",
        print_fn=lambda *a, **k: None,
    )
    _seed_shards_config(p, {
        "shards": [{"host": "local", "n_workers": 12}],
    })
    captured = []

    def capture(msg, *a, **k):
        captured.append(msg)

    refresh_shards_config_for_iter(
        config_path=p, previous=first, iter_label="iter_002",
        print_fn=capture,
    )
    assert not any("WARNING" in m for m in captured)
