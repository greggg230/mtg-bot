"""Worker-side coverage for HEURISTIC (warm-start corpus) self-play.

The v2 worker must build the PIMC-over-heuristic agent from the
claim's ``evaluator`` field, never touch the weights registry, and
emit npz columns identical to the NN path's so ``mtg_bot train``
consumes the shard unchanged.

Companion to ``test_worker_daemon_production_runner.py``, which
covers the NN path.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mtg_bot import worker_daemon_production_runner as prr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_tiny_deck_pool(tmp_path: Path) -> Path:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from _test_decks import simple_test_deck_pool
    finally:
        sys.path.pop(0)

    decks = simple_test_deck_pool()
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "pool.json.gz"
    serialized = json.dumps([
        {"draft_id": d.draft_id, "build_index": d.build_index,
         "cards": d.cards}
        for d in decks
    ])
    with gzip.open(path, "wb") as f:
        f.write(serialized.encode("utf-8"))
    return path


def _heuristic_claim(**overrides) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "claim_id": "s2_iter_011_self_play_heuristic:000000",
        "game_idx": 0,
        "deck_indices": [0, 1],
        "seed": 11_000_000,
        "kind": "self_play",
        # The interlock: falsy weights, so an old worker raises.
        "weights": {"p0": "", "p1": ""},
        "evaluator": "heuristic",
        "action_encoding_version": "canonical_v2",
        "structured_tokens": True,
        "structured_scalar_schema_version": 2,
        "nodes": 6,
        "samples": 1,
        "per_game_timeout_sec": 120.0,
        "capture_every": 0,
        "move_temperature": 1.0,
        "temperature_cutoff": 30,
        "expires_at": "2099-01-01T00:00:00Z",
    }
    claim.update(overrides)
    return claim


def _empty_chunk():
    empty = (
        np.zeros((0, 51), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0, 517), dtype=np.float32),
        np.zeros((0, 517), dtype=np.bool_),
    )
    return empty + (0, "completed", [], np.zeros((0,), dtype=np.int32))


def _build_runner(tmp_path: Path, events: list | None = None):
    def _exploding_http_get(url, timeout=30.0):  # pragma: no cover
        raise AssertionError(
            f"heuristic self-play must not fetch weights (got {url})"
        )

    return prr.build_self_play_runner(
        coordinator_url="http://test:5099",
        stage_id="s2_iter_011_self_play_heuristic",
        n_workers=1,
        weights_cache_root=tmp_path / "cache",
        worker_id="test-worker",
        log_event=(
            (lambda name, payload: events.append((name, payload)))
            if events is not None else None
        ),
        deck_pool_path=str(_save_tiny_deck_pool(tmp_path / "pool")),
        http_get=_exploding_http_get,
    )


# ---------------------------------------------------------------------------
# _heuristic_self_play_plan
# ---------------------------------------------------------------------------


def test_plan_builds_pimc_over_the_heuristic_evaluator():
    from mtg_bot.search.heuristic_evaluator import MtgHeuristicEvaluator
    from mtg_bot.search.pimc import PIMC

    plan = prr._heuristic_self_play_plan(_heuristic_claim())
    assert plan["version"] == "heuristic"
    assert plan["nodes"] == 6
    assert plan["samples"] == 1
    assert plan["action_encoding_version"] == "canonical_v2"
    assert plan["structured_tokens"] is True
    assert plan["structured_scalar_schema_version"] == 2

    strategy = plan["factory"](0)
    assert isinstance(strategy, PIMC)
    assert isinstance(
        strategy.inner_factory().evaluator, MtgHeuristicEvaluator,
    )


def test_plan_rejects_claim_without_action_encoding_version():
    """A heuristic stage has no metadata.json to fall back to, and a
    silently-wrong encoding would mislabel every policy target."""
    claim = _heuristic_claim()
    claim.pop("action_encoding_version")
    with pytest.raises(RuntimeError, match="action_encoding_version"):
        prr._heuristic_self_play_plan(claim)


def test_plan_defaults_structured_flags_off_when_absent():
    claim = _heuristic_claim()
    claim.pop("structured_tokens")
    claim.pop("structured_scalar_schema_version")
    plan = prr._heuristic_self_play_plan(claim)
    assert plan["structured_tokens"] is False
    assert plan["structured_scalar_schema_version"] is None


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def test_runner_plays_heuristic_claim_without_touching_weights(
    tmp_path, monkeypatch,
):
    """The whole point: no weights fetch, no NNPimcConfig, no torch
    state_dict load -- just the heuristic PIMC factory."""
    captured: dict[str, Any] = {}

    def fake_play(**kwargs):
        captured.update(kwargs)
        chunk = _empty_chunk()
        return chunk, "completed", 0, 0

    monkeypatch.setattr(prr, "_play_self_play_game", fake_play)
    events: list = []
    runner = _build_runner(tmp_path, events)
    runner.setup("http://test:5099", tmp_path / "cache", "self_play", 1)

    result = runner.play_one_game(_heuristic_claim(), "test")

    assert result.outcome == "completed"
    assert result.npz_bytes
    # The factory came from the heuristic plan, not NNPimcConfig.
    from mtg_bot.search.pimc import PIMC
    assert isinstance(captured["factory"](0), PIMC)
    assert captured["action_encoding_version"] == "canonical_v2"
    assert captured["write_structured_tokens"] is True
    assert captured["structured_scalar_schema_version"] == 2
    assert captured["move_temperature"] == 1.0
    assert captured["temperature_cutoff"] == 30
    # No weights cache directory was ever created.
    assert not (tmp_path / "cache").exists()
    kinds = [name for name, _ in events]
    assert "weights_loaded" not in kinds
    starts = [p for n, p in events if n == "game_start"]
    assert starts and starts[0]["evaluator"] == "heuristic"


def test_runner_honours_claim_search_knobs(tmp_path, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_play(**kwargs):
        captured.update(kwargs)
        return _empty_chunk(), "completed", 0, 0

    monkeypatch.setattr(prr, "_play_self_play_game", fake_play)
    runner = _build_runner(tmp_path)
    runner.setup("http://test:5099", tmp_path / "cache", "self_play", 1)
    runner.play_one_game(
        _heuristic_claim(
            nodes=17, samples=3, move_temperature=0.5,
            temperature_cutoff=8,
        ),
        "test",
    )
    strategy = captured["factory"](0)
    assert strategy.samples == 3
    assert captured["move_temperature"] == 0.5
    assert captured["temperature_cutoff"] == 8


def test_runner_still_requires_weights_for_nn_claims(tmp_path):
    """The heuristic branch must not weaken the NN path's guard."""
    runner = _build_runner(tmp_path)
    runner.setup("http://test:5099", tmp_path / "cache", "self_play", 1)
    claim = _heuristic_claim(evaluator="nn", weights={})
    with pytest.raises(RuntimeError, match="missing weights"):
        runner.play_one_game(claim, "test")


def test_runner_rejects_unknown_evaluator(tmp_path):
    runner = _build_runner(tmp_path)
    runner.setup("http://test:5099", tmp_path / "cache", "self_play", 1)
    with pytest.raises(ValueError, match="unknown self-play evaluator"):
        runner.play_one_game(_heuristic_claim(evaluator="random"), "test")


# ---------------------------------------------------------------------------
# End-to-end: one real heuristic game through the worker body
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_heuristic_game_emits_nn_compatible_npz(tmp_path):
    """One real (tiny) heuristic game: the npz the worker ships back
    must carry exactly the column set an NN self-play shard carries,
    so the trainer consumes it unchanged."""
    runner = _build_runner(tmp_path)
    runner.setup("http://test:5099", tmp_path / "cache", "self_play", 1)
    result = runner.play_one_game(
        _heuristic_claim(nodes=4, samples=1), "test",
    )
    assert result.outcome == "completed", result.outcome

    import io
    with np.load(io.BytesIO(result.npz_bytes)) as data:
        assert set(data.files) == {
            "features", "outcomes", "policy_targets", "action_masks",
            "structured_token_type", "structured_token_emb_ids",
            "structured_token_scalars", "structured_attention_mask",
            "root_values", "game_ids", "structured_card_vocab",
            # Policy-column encoding + resolution-supervision counters.
            "action_encoding_version", "resolution_pause_samples",
            "resolution_policy_rows", "resolution_overflow_visits",
        }
        from mtg_bot.training.card_vocab import (
            current_card_vocab,
            decode_shard_card_vocab,
        )
        from mtg_bot.training.data_gen import shard_n_printings_cap

        assert decode_shard_card_vocab(data["structured_card_vocab"]) == (
            current_card_vocab(), shard_n_printings_cap(),
        )
        assert data["features"].shape[0] > 0
        # Dense value target (2026-09-26): the heuristic PIMC reports a
        # root Q at every searched sample point, so a real game lands
        # finite root values on (at least) those rows, in [-1, 1].
        rv = data["root_values"]
        assert rv.shape == (data["features"].shape[0],)
        assert rv.dtype == np.float32
        finite = np.isfinite(rv)
        # Searched sample points carry the root Q; forced-move sample
        # points carry the heuristic's static value -- so with a
        # heuristic PIMC every recorded row has a value.
        assert finite.all(), f"{(~finite).sum()} rows without a value"
        assert np.all(np.abs(rv) <= 1.0 + 1e-6)
        # Scalar schema 2 width, as the claim asked for.
        assert data["structured_token_scalars"].shape[-1] == 29
        # Visit-count policy supervision actually landed.
        assert float(
            (data["policy_targets"].sum(axis=-1) > 0.5).mean()
        ) > 0.0
