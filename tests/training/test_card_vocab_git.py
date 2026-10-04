"""Tests for ``mtg_bot.training.card_vocab_git`` — reconstructing the
card vocabulary a past commit's tokenizer used, from the commit itself.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from mtg_bot.training import card_vocab_git as cvg
from mtg_bot.training.card_vocab import current_card_vocab
from mtg_bot.training.card_vocab_git import (
    CardVocabReconstructionError,
    reconstruct_card_vocab,
    resolve_commit,
)


def _git_available() -> bool:
    try:
        resolve_commit("HEAD")
    except Exception:  # noqa: BLE001 - not a git checkout
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _git_available(), reason="needs a git checkout of the repo",
)


@pytest.fixture(scope="module")
def head_reconstruction():
    return reconstruct_card_vocab("HEAD")


def test_head_reconstructs_to_the_current_vocabulary(head_reconstruction):
    """HEAD's committed card data + tokenizer give exactly the
    vocabulary this checkout's active source produces."""
    rec = head_reconstruction
    assert rec.vocab == current_card_vocab()
    assert rec.git_commit == resolve_commit("HEAD")
    assert len(rec.git_commit) == 40
    assert rec.pool == "default"
    assert rec.default_n_printings == 320
    prov = rec.provenance()
    assert prov["method"] == "reconstructed_from_git"
    assert prov["git_commit"] == rec.git_commit


def test_unknown_commit_is_an_error():
    with pytest.raises(CardVocabReconstructionError, match="unknown git commit"):
        resolve_commit("0" * 40)
    with pytest.raises(CardVocabReconstructionError):
        reconstruct_card_vocab("no-such-ref-xyz")


def test_pool_selection_is_never_silently_ignored():
    """A ``--pool`` on a commit without a pool registry must fail, not
    quietly reconstruct the default pool; an unknown pool name on a
    commit WITH one must fail too."""
    try:
        import mtg_bot.cards.pools  # noqa: F401
        has_pools = True
    except ImportError:
        has_pools = False
    if has_pools:  # pragma: no cover - only once pools reach this branch
        with pytest.raises(CardVocabReconstructionError):
            reconstruct_card_vocab("HEAD", pool="definitely-not-a-pool")
    else:
        with pytest.raises(
            CardVocabReconstructionError, match="no card-pool registry",
        ):
            reconstruct_card_vocab("HEAD", pool="cube")


def _fake_run(stdout: str, returncode: int = 0):
    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "git":
            return real_run(cmd, *args, **kwargs)
        return types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr="boom",
        )

    return run


def test_garbled_subprocess_output_is_an_error(monkeypatch):
    monkeypatch.setattr(cvg.subprocess, "run", _fake_run("not json"))
    with pytest.raises(CardVocabReconstructionError, match="could not"):
        reconstruct_card_vocab("HEAD")


def test_reported_error_is_surfaced(monkeypatch):
    monkeypatch.setattr(
        cvg.subprocess, "run",
        _fake_run('{"error": "printing ids are not 1..N"}', returncode=3),
    )
    with pytest.raises(CardVocabReconstructionError, match="1..N"):
        reconstruct_card_vocab("HEAD")


def test_non_sorted_name_map_is_refused(monkeypatch):
    monkeypatch.setattr(
        cvg.subprocess, "run",
        _fake_run('{"names": ["b", "a"], "default_n_printings": 320}'),
    )
    with pytest.raises(CardVocabReconstructionError, match="sorted_names_v1"):
        reconstruct_card_vocab("HEAD")


def test_archive_failure_is_an_error(monkeypatch, tmp_path):
    def fail(repo, args):
        return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"nope")

    sha = resolve_commit("HEAD")
    monkeypatch.setattr(cvg, "_git", fail)
    with pytest.raises(CardVocabReconstructionError, match="git archive"):
        cvg._extract(cvg._REPO_DEFAULT, sha, tmp_path)
