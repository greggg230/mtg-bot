"""Test helpers for the card-vocabulary fingerprint.

Structured checkpoints must record the card vocabulary their printing
embeddings were trained on (``mtg_bot.training.card_vocab``); loaders
refuse one without it. Tests that write a bare ``StructuredModel``
state_dict with ``torch.save`` stamp it with these helpers, exactly as
production code would (``NNEvaluator.save`` stamps automatically;
``mtg_bot train`` writes ``metadata.card_vocab``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mtg_bot.training.card_vocab import current_card_vocab, write_sidecar


def stamp_current_vocab(weights_path: Path) -> Path:
    """Stamp ``weights_path`` (already written) with the current card
    vocabulary via its ``<weights>.card_vocab.json`` sidecar."""
    return write_sidecar(
        current_card_vocab(), Path(weights_path),
        provenance={"source": "test"},
    )


def current_vocab_dict() -> dict[str, Any]:
    """``metadata.card_vocab`` value for a checkpoint trained on the
    current card vocabulary."""
    return current_card_vocab().to_dict(provenance={"source": "test"})
