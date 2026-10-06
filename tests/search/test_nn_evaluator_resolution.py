"""NN evaluation + search at resolution pauses: policy-slot
subsetting, per-version vocab width, PUCT priors for resolution
children (``docs/RESOLUTION_DECISIONS.md`` §7 / §10)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mtg_bot.search import MCTS, PIMC, Budget
from mtg_bot.search.nn_evaluator import NNEvaluator
from mtg_bot.training import action_encoding as ae
from mtg_bot.training.structured_model import (
    StructuredModel, StructuredModelConfig, make_dummy_batch,
)
from tests.search._resolution_helpers import (
    chaos_adapter, paused_bombard_adapter,
)

_TINY = dict(
    structured_emb_dim=16, structured_n_heads=2, structured_n_layers=1,
    structured_hidden_dim=32,
)


def _ev(version="canonical_v3"):
    torch.manual_seed(0)
    return NNEvaluator(
        architecture="structured", action_encoding_version=version, **_TINY,
    )


def test_forward_slot_range_matches_the_full_forward_columns():
    torch.manual_seed(1)
    model = StructuredModel(StructuredModelConfig(
        emb_dim=16, n_heads=2, n_layers=1, hidden_dim=32, action_vocab_size=805,
    )).eval()
    batch = make_dummy_batch(batch_size=3, seq_len=9, config=model.config)
    with torch.no_grad():
        full = model(batch)
        lo = model(batch, policy_slot_range=(0, 517))
        hi = model(batch, policy_slot_range=(517, 805))
    assert lo["policy_logits"].shape == (3, 517)
    assert hi["policy_logits"].shape == (3, 288)
    assert torch.equal(full["value"], lo["value"])
    assert torch.allclose(full["policy_logits"][:, :517], lo["policy_logits"], atol=1e-5)
    assert torch.allclose(full["policy_logits"][:, 517:], hi["policy_logits"], atol=1e-5)
    for bad in ((0, 0), (5, 3), (-1, 4), (0, 806)):
        with pytest.raises(ValueError, match="policy_slot_range"):
            model(batch, policy_slot_range=bad)


def test_evaluator_width_and_constructor_checks():
    assert _ev().action_vocab_size == 805
    assert _ev("canonical_v2").action_vocab_size == 517
    with pytest.raises(ValueError, match="disagrees"):
        NNEvaluator(
            architecture="structured", action_encoding_version="canonical_v3",
            structured_action_vocab_size=517, **_TINY,
        )
    with pytest.raises(ValueError, match="only supported by the structured"):
        NNEvaluator(
            architecture="shared", with_policy=True,
            action_encoding_version="canonical_v3",
        )


def test_policy_slot_range_selection():
    ev = _ev()
    paused = paused_bombard_adapter()
    plain = chaos_adapter(1, expose_resolution_decisions=False)
    m_p = ae.legal_action_mask(paused, 0, version="canonical_v3")
    m_n = ae.legal_action_mask(plain, plain.active_player(), version="canonical_v3")
    assert ev._policy_slot_range([plain], [m_n]) == (0, 517)
    assert ev._policy_slot_range([paused], [m_p]) == (517, 805)
    assert ev._policy_slot_range([paused, plain], [m_p, m_n]) == (0, 805)
    odd = m_n.copy()
    odd[700] = True
    assert ev._policy_slot_range([plain], [odd]) == (0, 805)
    assert _ev("canonical_v2")._policy_slot_range([paused], [m_p[:517]]) == (0, 517)


def test_evaluate_with_policy_at_a_pause_and_batched():
    ev = _ev()
    paused = paused_bombard_adapter()
    v, pol = ev.evaluate_with_policy(paused, 0)
    assert pol.shape == (805,)
    assert not pol[:517].any()
    assert pol.sum() == pytest.approx(1.0, abs=1e-5)
    plain = chaos_adapter(1, expose_resolution_decisions=False)
    who = plain.active_player()
    v2, pol2 = ev.evaluate_with_policy(plain, who)
    assert not pol2[517:].any()
    # A mixed batch (full-width forward) agrees with the single calls.
    ev_p0 = ev.batch_evaluate_with_policy([paused], 0)[0]
    assert np.allclose(ev_p0[1], pol, atol=1e-5)
    mixed = ev.batch_evaluate_with_policy([paused, paused.clone()], 0)
    assert all(np.allclose(p, pol, atol=1e-5) for _, p in mixed)
    # Non-decider at a pause: no decision, all-zero policy.
    _, none = ev.evaluate_with_policy(paused, 1)
    assert not none.any()
    _, logits = ev.evaluate_logits(paused, 0)
    assert logits.shape == (805,)
    assert all(l.shape == (805,) for _, l in ev.batch_evaluate_logits([paused], 0))


def test_mcts_priors_resolution_children_from_the_policy_head():
    ev = _ev()
    paused = paused_bombard_adapter()
    mcts = MCTS(
        evaluator=ev, budget_factory=lambda: Budget(max_nodes=12),
        rng=random.Random(0), puct=True,
    )
    _, cands = mcts.choose_with_full_stats(paused, 0)
    legal = set(paused.legal_decisions())
    assert {c["decision"] for c in cands} <= legal
    priors = [c["prior"] for c in cands if c["prior"] is not None]
    assert priors and all(p > 0 for p in priors)
    assert sum(priors) == pytest.approx(1.0, abs=1e-4)
    assert mcts.prior_overflow_count == 0


def test_pimc_answers_a_pause_with_a_legal_resolution_decision():
    ev = _ev()
    paused = paused_bombard_adapter()

    def inner():
        return MCTS(
            evaluator=ev, budget_factory=lambda: Budget(max_nodes=8),
            rng=random.Random(1), puct=True,
        )

    choice = PIMC(inner_factory=inner, samples=2, rng=random.Random(2)).choose(
        paused, 0,
    )
    assert choice in set(paused.legal_decisions())
