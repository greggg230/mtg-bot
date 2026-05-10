"""Tests for ``mtg_bot.cli._common`` — the shared argparse parents
and default constants every subcommand consumes.

The whole point of ``_common`` is "exactly one place to update the
default eval depth", so the value of these tests is mostly
defending against accidental drift in the defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from mtg_bot.cli import _common


def test_default_constants_match_iter_005_self_play_depth():
    """The whole point of pulling these defaults out: there is one
    place to update them. ``DEFAULT_NODES`` MUST match whatever
    self-play depth iterate is running at — currently 200 (iter_005).
    """
    assert _common.DEFAULT_NODES == 200
    assert _common.DEFAULT_SAMPLES == 4


def test_default_urls_use_master_tailnet():
    """Queue + bundle URLs are derived from the master's tailnet
    hostname so a fresh-clone invocation works with no flags."""
    assert _common.DEFAULT_MASTER_TAILNET == "desktop-j05412i"
    assert _common.DEFAULT_QUEUE_URL == (
        f"http://{_common.DEFAULT_MASTER_TAILNET}:5099"
    )
    assert _common.DEFAULT_BUNDLE_URL == (
        f"{_common.DEFAULT_QUEUE_URL}/repo.bundle"
    )


def test_default_paths_are_relative_to_repo_root():
    """The path defaults assume cwd is the repo root (which is how
    operators run things). They're ``Path`` objects, not strings —
    the rest of the codebase expects ``Path``."""
    assert isinstance(_common.DEFAULT_DECK_POOL, Path)
    assert isinstance(_common.DEFAULT_PROGRESS_DIR, Path)
    assert isinstance(_common.DEFAULT_SHARDS_CONFIG, Path)
    assert _common.DEFAULT_DECK_POOL.parts[-1] == "eoe_decks.json.gz"
    assert _common.DEFAULT_PROGRESS_DIR.parts[-1] == "training_progress"
    assert _common.DEFAULT_SHARDS_CONFIG.parts[-1] == "shards_config.json"


def test_default_per_game_timeout_is_finite_positive():
    """Sanity — a 0 or negative default would mean every game times
    out instantly. 600s matches the production iter timeout post the
    2026-05-08 bump (was 450; observed completed-game p99=390s and
    master-slowdown right-tail clipped the old cap)."""
    assert _common.DEFAULT_PER_GAME_TIMEOUT_SEC > 0.0
    assert _common.DEFAULT_PER_GAME_TIMEOUT_SEC == 600.0


def test_depth_parser_yields_defaults_when_no_flags():
    """``depth_parser`` is consumed via ``parents=[depth_parser()]``
    by other parsers; calling it with no overrides should return
    the documented defaults."""
    parent = _common.depth_parser()
    p = argparse.ArgumentParser(parents=[parent])
    args = p.parse_args([])
    assert args.samples == _common.DEFAULT_SAMPLES
    assert args.nodes == _common.DEFAULT_NODES


def test_depth_parser_accepts_overrides():
    parent = _common.depth_parser()
    p = argparse.ArgumentParser(parents=[parent])
    args = p.parse_args(["--samples", "8", "--nodes", "1000"])
    assert args.samples == 8
    assert args.nodes == 1000


def test_cluster_parser_yields_defaults():
    parent = _common.cluster_parser()
    p = argparse.ArgumentParser(parents=[parent])
    args = p.parse_args([])
    assert args.shards_config == _common.DEFAULT_SHARDS_CONFIG
    assert args.queue_url == _common.DEFAULT_QUEUE_URL
    assert args.bundle_url == _common.DEFAULT_BUNDLE_URL


def test_cluster_parser_accepts_overrides():
    parent = _common.cluster_parser()
    p = argparse.ArgumentParser(parents=[parent])
    args = p.parse_args([
        "--shards-config", "alt_shards.json",
        "--queue-url", "http://other-host:5099",
        "--bundle-url", "http://other-host:5099/repo.bundle",
    ])
    assert args.shards_config == Path("alt_shards.json")
    assert args.queue_url == "http://other-host:5099"
    assert args.bundle_url == "http://other-host:5099/repo.bundle"


def test_runtime_parser_yields_defaults():
    parent = _common.runtime_parser()
    p = argparse.ArgumentParser(parents=[parent])
    args = p.parse_args([])
    assert args.deck_pool == _common.DEFAULT_DECK_POOL
    assert args.progress_dir == _common.DEFAULT_PROGRESS_DIR
    assert args.per_game_timeout_sec == _common.DEFAULT_PER_GAME_TIMEOUT_SEC


def test_runtime_parser_accepts_overrides():
    parent = _common.runtime_parser()
    p = argparse.ArgumentParser(parents=[parent])
    args = p.parse_args([
        "--deck-pool", "data/other.json.gz",
        "--progress-dir", "alt_progress",
        "--per-game-timeout-sec", "300",
    ])
    assert args.deck_pool == Path("data/other.json.gz")
    assert args.progress_dir == Path("alt_progress")
    assert args.per_game_timeout_sec == 300.0


def test_parents_compose_in_a_single_parser():
    """The cross-cutting use case: a subcommand inherits all three
    parents. Verifies they don't collide on flag names."""
    p = argparse.ArgumentParser(parents=[
        _common.depth_parser(),
        _common.cluster_parser(),
        _common.runtime_parser(),
    ])
    args = p.parse_args([])
    # All defaults present, no conflicts.
    assert args.samples == _common.DEFAULT_SAMPLES
    assert args.nodes == _common.DEFAULT_NODES
    assert args.queue_url == _common.DEFAULT_QUEUE_URL
    assert args.deck_pool == _common.DEFAULT_DECK_POOL


def test_load_shards_config_delegates_to_distributed_module():
    """``_common.load_shards_config`` is a thin re-export so
    subcommands don't have to reach into ``mtg_bot.training.distributed``
    themselves. Verifies the wiring."""
    sentinel = object()
    with patch(
        "mtg_bot.training.distributed.load_shards_config",
        return_value=sentinel,
    ) as mock_impl:
        result = _common.load_shards_config(Path("any.json"))
    assert result is sentinel
    mock_impl.assert_called_once_with(Path("any.json"))
