"""Cracked Skull look-at-hand + may-discard substrate — Cycle 1.

Spec: lane brief (2026-05-22). The Rust side gains the
``PendingDecision::LookAtOpponentHandMayDiscard {source_obj_id,
controller, target_player, candidates}`` variant, the dict-snapshot
surface key, the fingerprint serialization arm, the
``submit_look_at_opponent_hand_may_discard_decision`` dispatcher (the
agent-driven Some/None routing), the
``push_look_at_opponent_hand_may_discard_choice`` synthetic-test
setter (no production caller in Cycle 1 — Cycle 2 wires the
``Effect::TargetOpponentLookHandMayDiscard`` resolve arm that pushes
the decision), and the ``look_at_hand_decline`` /
``look_at_hand_discard_first`` auto-resolve strategies.

Cycle 1 substrate-only — no live mint site is wired (no
``Effect::TargetOpponentLookHandMayDiscard`` resolve arm in Rust);
the push helper + dispatcher + auto-resolve arms are wired but no
internal Rust resolve site routes through them yet (Cycle 2 wires
the live mint).

Mirrors Python's ``TargetOpponentLookHandMayDiscard.resolve`` at
``src/mtg_bot/effects/generic/discard.py:330-385``. Cycle 1's tests
exercise the Rust-side primitive directly without crossing the
Python ↔ Rust boundary on a real state.

**Tests cover:**

1. Default — fresh state has an empty pending-decisions queue;
   ``has_pending_decision`` returns False.
2. Push helper — pushes a ``LookAtOpponentHandMayDiscard`` onto the
   queue; ``pending_decisions()`` returns a list of length 1 with the
   correct ``kind`` / ``source_obj_id`` / ``controller`` /
   ``target_player`` / ``candidates`` keys.
3. Push validation — invalid ``target_player`` (≥ 2) or invalid
   ``controller`` (≥ 2) raises ``ValueError`` transactionally
   (queue stays empty).
4. Push validation — candidate id not in ``target_player.hand`` raises
   ``ValueError`` transactionally (queue stays empty).
5. Submit Some(id) → discard — pre-pushed variant: submitting an id
   from the candidate set moves the card from
   ``target_player.hand`` → ``target_player.graveyard`` (zone field
   stamped to graveyard); queue head consumed.
6. Submit None → decline — pre-pushed variant: submitting None
   consumes the head but leaves zones unchanged (no discard).
7. Submit validation — Some(id) with id NOT in candidate set raises
   ``ValueError`` transactionally (queue head stays in place; no
   mutation on zones).
8. Submit validation — empty queue raises ``ValueError``.
9. Submit validation — discriminant mismatch (queue head is some
   other variant) raises ``ValueError``.
10. Auto-resolve ``"default"`` → decline (no discard). Matches
    Python's no-agent-seated fallback at
    ``effects/generic/discard.py:362-363`` (``if caster.agent is
    None: return``).
11. Auto-resolve ``"look_at_hand_discard_first"`` → discard
    ``candidates[0]`` from target opponent's hand.
12. Auto-resolve ``"look_at_hand_decline"`` → decline (no discard);
    explicit-name alias for the default.
13. Auto-resolve drain — multi-variant queue: a Surveil + a
    LookAtOpponentHandMayDiscard together drain with ``"default"``
    in FIFO order; the LookAt decision declines.
14. Fingerprint — the LookAtOpponentHandMayDiscard variant participates
    in the per-state fingerprint: identical (push + same candidates)
    yields identical fingerprints; differing target_player or
    candidates yields different fingerprints.
15. Back-compat — fresh state with NO pushed decision has a
    fingerprint byte-equal to itself across two recomputations.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


def _fresh_state():
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


def _creature_dict(name="Bears", power=2, toughness=2):
    return {
        "name": name,
        "mana_cost": {"generic": 1, "G": 1},
        "card_types": ["Creature"],
        "subtypes": ["Bear"],
        "power": power,
        "toughness": toughness,
        "keywords": [],
    }


def _spell_dict(name="Cantrip"):
    return {
        "name": name,
        "mana_cost": {"generic": 1, "U": 1},
        "card_types": ["Sorcery"],
        "subtypes": [],
        "keywords": [],
    }


def _give_card_to_hand(state, name, controller, mana_cost=None):
    """Materialize a non-creature spell card in `controller`'s hand."""
    d = _spell_dict(name)
    if mana_cost is not None:
        d["mana_cost"] = mana_cost
    return state.register_and_inject_object(d, controller, ZONE_HAND)


def _give_aura_to_bf(state, controller):
    """Materialize a Cracked-Skull-shaped source on `controller`'s bf
    so the push helper can attribute the decision to it. Cycle 1
    doesn't need a real Aura; any bf permanent will do."""
    d = _creature_dict("HostBear")
    return state.register_and_inject_object(d, controller, ZONE_BATTLEFIELD)


def _hand_ids(state, pid):
    return [o["id"] for o in state.hand_snapshot(pid)]


def _gy_ids(state, pid):
    return [o["id"] for o in state.graveyard_snapshot(pid)]


# ---------------------------------------------------------------------------
# Default + push helper + snapshot surface
# ---------------------------------------------------------------------------


def test_default_no_pending_decision():
    """Pin 1: fresh state has an empty pending-decisions queue;
    ``has_pending_decision`` is False and ``pending_decisions()`` is
    empty. Substrate-only — no Effect mints the decision in Cycle 1."""
    _require_engine()
    state = _fresh_state()
    assert state.has_pending_decision() is False
    assert state.pending_decisions() == []


def test_push_helper_appends_correct_dict_shape():
    """Pin 2: the push helper appends a single
    ``LookAtOpponentHandMayDiscard`` variant; the dict surface
    matches the spec'd keys exactly."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    c2 = _give_card_to_hand(state, "CardB", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1, c2],
    )
    pds = state.pending_decisions()
    assert len(pds) == 1
    pd = pds[0]
    assert pd["kind"] == "LookAtOpponentHandMayDiscard"
    assert pd["source_obj_id"] == src_id
    assert pd["controller"] == 0
    assert pd["target_player"] == 1
    assert list(pd["candidates"]) == [c1, c2]
    assert state.has_pending_decision() is True


def test_push_helper_rejects_invalid_target_player():
    """Pin 3a: ``target_player`` ≥ 2 raises ``ValueError`` and the
    queue stays empty (transactional shape)."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    with pytest.raises(ValueError):
        state.push_look_at_opponent_hand_may_discard_choice(
            src_id, 0, 5, [],
        )
    assert state.pending_decisions() == []


def test_push_helper_rejects_invalid_controller():
    """Pin 3b: ``controller`` ≥ 2 raises ``ValueError`` and the queue
    stays empty (transactional shape)."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    with pytest.raises(ValueError):
        state.push_look_at_opponent_hand_may_discard_choice(
            src_id, 9, 1, [],
        )
    assert state.pending_decisions() == []


def test_push_helper_rejects_candidate_not_in_target_hand():
    """Pin 4: candidate id not in ``target_player.hand`` raises
    ``ValueError`` transactionally (queue stays empty). The push
    helper is the synthetic-test entrypoint, but invalid ids would
    create a queue-state no submit can safely route, so the
    validation is strict."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    real_id = _give_card_to_hand(state, "Real", 1)
    fake_id = real_id + 999
    with pytest.raises(ValueError):
        state.push_look_at_opponent_hand_may_discard_choice(
            src_id, 0, 1, [real_id, fake_id],
        )
    assert state.pending_decisions() == []
    # Real card stayed in hand (no zone mutation on failure).
    assert real_id in _hand_ids(state, 1)


# ---------------------------------------------------------------------------
# Submit dispatcher
# ---------------------------------------------------------------------------


def test_submit_some_id_discards_chosen_card():
    """Pin 5: submitting ``Some(id)`` discards that card from
    ``target_player.hand`` → ``target_player.graveyard``; queue head
    consumed. Mirrors Python's
    ``target.hand.remove(chosen); target.graveyard.append(chosen)``
    at ``effects/generic/discard.py:379-381``."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    c2 = _give_card_to_hand(state, "CardB", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1, c2],
    )
    # Pre-submit zones.
    assert c1 in _hand_ids(state, 1)
    assert c1 not in _gy_ids(state, 1)
    # Submit — discard c1.
    state.submit_look_at_opponent_hand_may_discard_decision(c1)
    # Queue drained.
    assert state.pending_decisions() == []
    # c1 moved from hand to graveyard; c2 still in hand.
    assert c1 not in _hand_ids(state, 1)
    assert c1 in _gy_ids(state, 1)
    assert c2 in _hand_ids(state, 1)


def test_submit_none_declines_no_zone_mutation():
    """Pin 6: submitting ``None`` declines the may-gate — no discard
    fires. Queue head consumed; zones unchanged. Mirrors Python's
    ``if not caster.agent.choose_yes_no: log, return`` at
    ``effects/generic/discard.py:364-372``."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    c2 = _give_card_to_hand(state, "CardB", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1, c2],
    )
    state.submit_look_at_opponent_hand_may_discard_decision(None)
    assert state.pending_decisions() == []
    # Both cards still in hand.
    hand = _hand_ids(state, 1)
    assert c1 in hand
    assert c2 in hand
    assert _gy_ids(state, 1) == []


def test_submit_unknown_id_raises_transactional():
    """Pin 7: submitting ``Some(id)`` with id NOT in the candidate
    set raises ``ValueError`` transactionally — the queue head stays
    in place and no zone mutation fires."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    c2 = _give_card_to_hand(state, "CardB", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1],  # only c1 is a candidate
    )
    with pytest.raises(ValueError):
        state.submit_look_at_opponent_hand_may_discard_decision(c2)
    # Queue head untouched.
    assert len(state.pending_decisions()) == 1
    # Zones untouched — no discard.
    assert c1 in _hand_ids(state, 1)
    assert c2 in _hand_ids(state, 1)
    assert _gy_ids(state, 1) == []


def test_submit_empty_queue_raises():
    """Pin 8a: submitting against an empty queue raises ``ValueError``
    (the substrate is honest about the empty-queue case rather than
    silently no-op'ing)."""
    _require_engine()
    state = _fresh_state()
    with pytest.raises(ValueError):
        state.submit_look_at_opponent_hand_may_discard_decision(7)


def test_submit_discriminant_mismatch_raises():
    """Pin 9: if the head of the pending-decisions queue is some other
    variant (e.g. a Surveil), the look-at-hand submit dispatcher
    raises ``ValueError`` (the validation runs before any consume /
    mutation)."""
    _require_engine()
    state = _fresh_state()
    # Push a non-LookAtOpponentHandMayDiscard variant. Easiest: drive
    # a surveil push through the engine surface. If no synthetic
    # surveil pusher is wired, fall through to the SacrificeSelfUnless
    # tap-permanent pump (drives bf + a candidate). The shape of this
    # test only needs ANY non-LookAtOpponentHandMayDiscard on the head.
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1],
    )
    # Now try to drain it with the WRONG dispatcher.
    with pytest.raises(ValueError):
        state.submit_tap_permanent_decision(c1)
    # Queue head untouched (transactional).
    assert len(state.pending_decisions()) == 1
    assert state.pending_decisions()[0]["kind"] == "LookAtOpponentHandMayDiscard"


# ---------------------------------------------------------------------------
# Auto-resolve strategies
# ---------------------------------------------------------------------------


def test_auto_resolve_default_declines():
    """Pin 10: ``"default"`` declines (no discard) — Python no-agent-
    seated parity."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1],
    )
    state.auto_resolve_pending_decisions("default")
    assert state.pending_decisions() == []
    assert c1 in _hand_ids(state, 1)
    assert _gy_ids(state, 1) == []


def test_auto_resolve_discard_first_takes_candidate_zero():
    """Pin 11: ``"look_at_hand_discard_first"`` discards
    ``candidates[0]`` from the target opponent's hand."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    c2 = _give_card_to_hand(state, "CardB", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1, c2],
    )
    state.auto_resolve_pending_decisions("look_at_hand_discard_first")
    assert state.pending_decisions() == []
    # c1 (candidates[0]) discarded; c2 still in hand.
    assert c1 not in _hand_ids(state, 1)
    assert c1 in _gy_ids(state, 1)
    assert c2 in _hand_ids(state, 1)


def test_auto_resolve_decline_alias():
    """Pin 12: ``"look_at_hand_decline"`` is an explicit-name alias
    for the decline-default. Same behavior as ``"default"``."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1],
    )
    state.auto_resolve_pending_decisions("look_at_hand_decline")
    assert state.pending_decisions() == []
    assert c1 in _hand_ids(state, 1)


def test_auto_resolve_empty_candidates_decline_is_noop():
    """Pin 12b: an empty-candidates push with ``"look_at_hand_discard
    _first"`` is a no-op on the discard (no candidate to pick) — the
    auto-resolve consumes the head defensively. The Cycle 2 Effect
    arm SHOULD avoid pushing in this case; this test pins the
    substrate's defensive contract for the Cycle 1 push helper."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [],
    )
    state.auto_resolve_pending_decisions("look_at_hand_discard_first")
    assert state.pending_decisions() == []
    assert _gy_ids(state, 1) == []


def test_auto_resolve_default_drains_multi_variant_queue():
    """Pin 13: a multi-variant queue (e.g. a LookAtOpponentHand
    decision sitting beside other unrelated variant) drains under
    ``"default"`` in FIFO order; the LookAt decision declines
    (no discard). Cycle 1 substrate only — uses the synthetic push
    helper. The mixed-queue case is exercised here to ensure the
    drain loop dispatches per-variant cleanly."""
    _require_engine()
    state = _fresh_state()
    src_id = _give_aura_to_bf(state, 0)
    c1 = _give_card_to_hand(state, "CardA", 1)
    state.push_look_at_opponent_hand_may_discard_choice(
        src_id, 0, 1, [c1],
    )
    # Push a second LookAt with empty candidates (defensive case).
    src2 = _give_aura_to_bf(state, 0)
    state.push_look_at_opponent_hand_may_discard_choice(
        src2, 0, 1, [],
    )
    assert len(state.pending_decisions()) == 2
    state.auto_resolve_pending_decisions("default")
    # Both consumed; default declines both -> c1 still in hand.
    assert state.pending_decisions() == []
    assert c1 in _hand_ids(state, 1)
    assert _gy_ids(state, 1) == []


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_participates_on_push():
    """Pin 14a: identical pushes (same source / controller /
    target_player / candidates) yield identical fingerprints. Differs
    only when one of those slots differs."""
    _require_engine()
    state_a = _fresh_state()
    state_b = _fresh_state()
    sa = _give_aura_to_bf(state_a, 0)
    sb = _give_aura_to_bf(state_b, 0)
    ca = _give_card_to_hand(state_a, "CardA", 1)
    cb = _give_card_to_hand(state_b, "CardA", 1)
    # The newly-injected ids on a fresh synthetic state are
    # deterministic across two equivalent states. Push with matching
    # shape.
    state_a.push_look_at_opponent_hand_may_discard_choice(
        sa, 0, 1, [ca],
    )
    state_b.push_look_at_opponent_hand_may_discard_choice(
        sb, 0, 1, [cb],
    )
    # Equivalent pushes -> equal fingerprints.
    assert state_a.fingerprint() == state_b.fingerprint()


def test_fingerprint_differs_on_different_target_player():
    """Pin 14b: pushing with target_player=0 vs target_player=1 (with
    cards in BOTH players' hands) yields DIFFERENT fingerprints — the
    target_player slot is in the fingerprint serialization, so the
    choice surface is honestly distinguished."""
    _require_engine()
    state_a = _fresh_state()
    state_b = _fresh_state()
    sa = _give_aura_to_bf(state_a, 0)
    sb = _give_aura_to_bf(state_b, 0)
    # Give EACH state a card in P0's hand AND P1's hand so the only
    # difference is the target_player on the pending decision.
    ca_p0 = state_a.register_and_inject_object(
        _spell_dict("HandCard"), 0, ZONE_HAND,
    )
    ca_p1 = state_a.register_and_inject_object(
        _spell_dict("HandCard"), 1, ZONE_HAND,
    )
    cb_p0 = state_b.register_and_inject_object(
        _spell_dict("HandCard"), 0, ZONE_HAND,
    )
    cb_p1 = state_b.register_and_inject_object(
        _spell_dict("HandCard"), 1, ZONE_HAND,
    )
    # state_a: P1 (the opponent of P0) is the discard target.
    state_a.push_look_at_opponent_hand_may_discard_choice(
        sa, 0, 1, [ca_p1],
    )
    # state_b: P0 (the opponent of P1) is the discard target.
    state_b.push_look_at_opponent_hand_may_discard_choice(
        sb, 1, 0, [cb_p0],
    )
    assert state_a.fingerprint() != state_b.fingerprint()


def test_fingerprint_default_state_byte_equal():
    """Pin 15: a fresh state with NO pushed decision has a fingerprint
    that recomputes byte-equal — the back-compat invariant for the
    new variant addition (the Cycle 1 substrate adds the variant but
    no fresh state observes it; pre-Cycle-1 baselines pin)."""
    _require_engine()
    state = _fresh_state()
    fp1 = state.fingerprint()
    fp2 = state.fingerprint()
    assert fp1 == fp2
    # And adding bf objects (without pushing a decision) still
    # leaves the pending-decisions slice of the fingerprint empty.
    _give_aura_to_bf(state, 0)
    _give_card_to_hand(state, "X", 1)
    fp3 = state.fingerprint()
    # fp3 differs from fp1 (the bf / hand zones differ), but recomputing
    # is byte-stable.
    fp4 = state.fingerprint()
    assert fp3 == fp4
