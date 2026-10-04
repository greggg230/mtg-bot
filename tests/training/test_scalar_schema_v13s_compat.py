"""Back-compat pin: the registered ``v13s`` structured checkpoint
(metadata.json predates ``scalar_schema_version``) must still resolve
to scalar schema 1 and build a model whose ``scalar_proj`` takes the
legacy 3-slot row — i.e. this feature does not change how any
pre-existing weights load.

``trained_weights/`` is not tracked by git, so the test looks for it at
the repo root and, for worktrees, at the sibling main checkout
(``../../mtg-bot/trained_weights``); it skips when neither exists.
Nothing under ``trained_weights/`` is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtg_bot.training.card_vocab import (
    MissingCardVocabError,
    current_card_vocab,
)
from mtg_bot.training.registry import ModelMetadata

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = [
    _REPO_ROOT / "trained_weights" / "v13s",
    _REPO_ROOT.parent.parent / "mtg-bot" / "trained_weights" / "v13s",
]


def _v13s_dir() -> Path:
    for d in _CANDIDATES:
        if (d / "metadata.json").is_file():
            return d
    pytest.skip("trained_weights/v13s/metadata.json not available")


def test_v13s_metadata_resolves_to_scalar_schema_1() -> None:
    d = _v13s_dir()
    with (d / "metadata.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    assert "scalar_schema_version" not in raw  # predates the field
    meta = ModelMetadata.from_dict(raw)
    assert meta.architecture == "structured"
    assert meta.scalar_schema_version == 1


def test_v13s_builds_with_three_wide_scalar_proj() -> None:
    """Instantiate the model exactly the way ``NNEvaluator.from_metadata``
    does (no weights load — 13 MB of v13s weights are not needed to
    pin the projection width) and check ``scalar_proj.in_features``."""
    from mtg_bot.search.nn_evaluator import NNEvaluator

    d = _v13s_dir()
    with (d / "metadata.json").open("r", encoding="utf-8") as f:
        meta = ModelMetadata.from_dict(json.load(f))
    ev = NNEvaluator.from_metadata(None, meta)
    assert ev.is_structured
    assert ev.model.config.scalar_schema_version == 1
    assert ev.model.scalar_proj.in_features == 3
    assert ev._structured_scalar_schema_version == 1
    # The trunk dims come from the (nested + flat) v13s metadata.
    assert ev.model.config.emb_dim == 256
    assert ev.model.config.n_layers == 4


def test_v13s_weights_load_into_schema_1_model() -> None:
    """If the weights file is present, the real state_dict must load
    into the schema-1 model bit-for-bit (a wrong width would raise a
    size-mismatch error here)."""
    from mtg_bot.search.nn_evaluator import NNEvaluator

    d = _v13s_dir()
    weights = d / "weights.pt"
    if not weights.is_file():
        pytest.skip("v13s weights.pt not available")
    with (d / "metadata.json").open("r", encoding="utf-8") as f:
        meta = ModelMetadata.from_dict(json.load(f))
    if meta.card_vocab or (d / "weights.pt.card_vocab.json").is_file():
        pytest.skip("v13s has been stamped; the unstamped pin no longer applies")
    # Unstamped (predates the card-vocab fingerprint): refused, never
    # guessed.
    with pytest.raises(MissingCardVocabError):
        NNEvaluator.from_metadata(weights, meta)
    # With its vocabulary supplied explicitly it loads bit-for-bit.
    # v13s was trained at 3af7edea, whose card vocabulary is the
    # 475-name list main still uses (``mtg_bot card-vocab export
    # --git-commit 3af7edea``); this pin is about the scalar width.
    meta.card_vocab = current_card_vocab().to_dict()
    ev = NNEvaluator.from_metadata(weights, meta)
    assert ev.model.scalar_proj.in_features == 3
    assert ev.card_vocab_report is not None
