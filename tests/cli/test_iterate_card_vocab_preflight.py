"""``mtg_bot iterate`` checks the structured bootstrap's card vocabulary
up front, on the master, before any self-play stage is launched — so an
unstamped checkpoint fails in seconds with an actionable message instead
of crashing every worker's first game.

Re-uses the stubbed-downstream harness from
``test_iterate_run_db_wiring``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtg_bot.cli import iterate as cli_iterate
from mtg_bot.cli.runs import lifecycle
from mtg_bot.training.card_vocab import CardVocab, write_sidecar

from tests.cli.test_iterate_run_db_wiring import (
    _baseline_argv,
    _make_version_dir,
    _setup_mocks_for_iterate,
)


def _structured_bootstrap(registry_root: Path) -> Path:
    _make_version_dir(registry_root, "v1")
    meta_path = registry_root / "v1" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["architecture"] = "structured"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return registry_root / "v1" / "weights.pt"


def _argv(tmp_path: Path) -> list[str]:
    deck_pool = tmp_path / "deck_pool.json.gz"
    deck_pool.write_bytes(b"")
    return _baseline_argv(
        tmp_path=tmp_path, deck_pool=deck_pool, bootstrap="v1",
        n_iterations=1, eval_games=0, acceptance_threshold=0.0,
    )


@pytest.fixture
def _harness(monkeypatch, tmp_path):
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc1234")
    _setup_mocks_for_iterate(monkeypatch)


def test_unstamped_structured_bootstrap_is_refused(_harness, tmp_path):
    _structured_bootstrap(tmp_path / "trained_weights")
    with pytest.raises(SystemExit) as exc:
        cli_iterate.main(_argv(tmp_path))
    msg = str(exc.value)
    assert "Bootstrap version 'v1'" in msg
    assert "card vocabulary" in msg and "card-vocab stamp" in msg


def test_stamped_structured_bootstrap_passes_the_preflight(
    _harness, tmp_path, capsys,
):
    weights = _structured_bootstrap(tmp_path / "trained_weights")
    vocab = CardVocab.from_names(["Alpha", "Beta", "Gamma"])
    write_sidecar(vocab, weights)
    try:
        cli_iterate.main(_argv(tmp_path))
    except SystemExit as exc:  # later checks reject the fake harness
        assert "card vocabulary" not in str(exc)
    out = capsys.readouterr().out
    assert f"v1 card vocabulary {vocab.short_sha} (3 names, from sidecar)" in out
