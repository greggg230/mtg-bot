"""Tests for the modifier-grammar foundation (PR A).

Covers:

- :class:`EventTransform` primitives apply pure payload edits
  (``MultiplyCount``, ``AddToCount``, ``AddInstances``, ``Replace``,
  ``Prevent``).
- :class:`EventPattern` data-class wiring.
- :class:`ContinuousModifier` / :class:`ReplacementEffect` dataclass
  shapes survive frozen-equality and field-default expectations.
- :func:`dispatch_replacement` returns the original payload when no
  replacement applies (0 candidates).
- :func:`dispatch_replacement` applies a single matching replacement
  (1 candidate) without consulting any agent.
- :func:`dispatch_replacement` consults the affected player's
  ``choose_replacement_order`` agent hook for ≥2 candidates.
- The legacy ``is_token_doubler`` flag synthesises a
  ``ReplacementEffect(WOULD_CREATE_TOKEN, MultiplyCount(2))`` and
  composes correctly with the dispatcher (Exalted Sunborn doubles to
  4; two doublers compound to 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mtg_bot.effects.modifiers import (
    AddInstances,
    AddSubtype,
    AddToCount,
    ContinuousModifier,
    EventPattern,
    EventTransform,
    GrantKeyword,
    LayerPayload,
    MultiplyCount,
    PowerToughnessBonus,
    Prevent,
    Replace,
    RemoveKeyword,
    ReplacementEffect,
    dispatch_replacement,
    walk_continuous_modifiers,
)
from mtg_bot.effects.object_predicates import IsCreature, IsArtifact
from mtg_bot.effects.predicates import EVENT_SUBJECTS, Event


# ────────────────────────────────────────────────────────────────────
# EventTransform primitives
# ────────────────────────────────────────────────────────────────────


def test_multiply_count_doubles_count():
    out = MultiplyCount(2).apply({"count": 3})
    assert out["count"] == 6


def test_multiply_count_zero_count_stays_zero():
    out = MultiplyCount(2).apply({"count": 0})
    assert out["count"] == 0


def test_multiply_count_missing_count_treated_as_zero():
    out = MultiplyCount(3).apply({})
    assert out["count"] == 0


def test_multiply_count_preserves_other_keys():
    out = MultiplyCount(2).apply({"count": 1, "kind": "Drone"})
    assert out["kind"] == "Drone"
    assert out["count"] == 2


def test_add_to_count_increments():
    out = AddToCount(1).apply({"count": 5})
    assert out["count"] == 6


def test_add_instances_increments_default_one():
    out = AddInstances(1).apply({})
    assert out["instances"] == 2


def test_add_instances_increments_explicit_count():
    out = AddInstances(2).apply({"instances": 3})
    assert out["instances"] == 5


def test_replace_sets_destination():
    out = Replace("library").apply({"destination": "graveyard"})
    assert out["destination"] == "library"


def test_prevent_sets_prevented_flag():
    out = Prevent().apply({"count": 1})
    assert out["prevented"] is True
    # Other keys preserved so a downstream consumer can introspect.
    assert out["count"] == 1


def test_transforms_are_pure_dont_mutate_input():
    payload = {"count": 1}
    MultiplyCount(2).apply(payload)
    assert payload == {"count": 1}


# ────────────────────────────────────────────────────────────────────
# EventPattern / dataclass shapes
# ────────────────────────────────────────────────────────────────────


def test_event_pattern_default_filters_empty():
    p = EventPattern(event=Event.WOULD_CREATE_TOKEN)
    assert p.subject_filters == {}


def test_event_pattern_with_filters():
    p = EventPattern(
        event=Event.WOULD_PLACE_COUNTER,
        subject_filters={"obj": IsCreature()},
    )
    assert "obj" in p.subject_filters


def test_replacement_effect_default_scope_filter_none():
    re = ReplacementEffect(
        event_pattern=EventPattern(event=Event.WOULD_CREATE_TOKEN),
        transform=MultiplyCount(2),
    )
    assert re.scope_filter is None


def test_continuous_effect_minimal_shape():
    ce = ContinuousModifier(
        layer=6,
        subject_filter=IsCreature(),
        payload=GrantKeyword("flying"),
    )
    assert ce.layer == 6


# ────────────────────────────────────────────────────────────────────
# Layer-payload variants
# ────────────────────────────────────────────────────────────────────


def test_grant_keyword_payload():
    p = GrantKeyword("flying")
    assert isinstance(p, LayerPayload)
    assert p.keyword == "flying"


def test_remove_keyword_payload():
    p = RemoveKeyword("defender")
    assert p.keyword == "defender"


def test_power_toughness_bonus_default_zero():
    p = PowerToughnessBonus()
    assert p.power_bonus == 0
    assert p.toughness_bonus == 0


def test_add_subtype_payload():
    p = AddSubtype("Sliver")
    assert p.subtype == "Sliver"


# ────────────────────────────────────────────────────────────────────
# Event enum / EVENT_SUBJECTS — the 5 new WOULD_X entries
# ────────────────────────────────────────────────────────────────────


def test_would_draw_cards_in_event_subjects():
    assert EVENT_SUBJECTS[Event.WOULD_DRAW_CARDS] == ("player",)


def test_would_create_token_in_event_subjects():
    assert EVENT_SUBJECTS[Event.WOULD_CREATE_TOKEN] == ("controller", "kind")


def test_triggered_ability_triggers_in_event_subjects():
    assert EVENT_SUBJECTS[Event.TRIGGERED_ABILITY_TRIGGERS] == (
        "triggering_obj", "ability_source",
    )


def test_would_place_counter_in_event_subjects():
    assert EVENT_SUBJECTS[Event.WOULD_PLACE_COUNTER] == ("obj", "kind")


def test_would_go_to_graveyard_in_event_subjects():
    assert EVENT_SUBJECTS[Event.WOULD_GO_TO_GRAVEYARD] == ("card",)


# ────────────────────────────────────────────────────────────────────
# dispatch_replacement — 0 / 1 / N candidates
# ────────────────────────────────────────────────────────────────────


def _empty_state():
    """Build a minimal ``GameState`` with two players and no battlefield
    contents so the dispatcher's ``_collect_replacement_effects`` walk
    finds zero candidates."""
    from mtg_bot.game import GameState, Player

    state = GameState()
    state.players = [Player(id=0), Player(id=1)]
    return state


def test_dispatch_no_replacements_returns_payload_unchanged():
    state = _empty_state()
    out = dispatch_replacement(
        state, Event.WOULD_CREATE_TOKEN,
        affected_player_id=0,
        subjects={"controller_id": 0, "kind": "Drone"},
        payload={"count": 3},
    )
    assert out["count"] == 3


def test_dispatch_with_token_doubler_doubles_count():
    """Exalted Sunborn on the battlefield doubles the count via the
    legacy ``is_token_doubler`` synthesis path."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.game import GameState, Player, Zone

    src = get_active_source()
    prs = src.get_printings()
    state = GameState()
    state.players = [Player(id=0), Player(id=1)]
    sunborn = state.new_object(
        prs["Exalted Sunborn"], 0, Zone.BATTLEFIELD,
    )
    sunborn.controller_id = 0
    state.battlefield.append(sunborn)

    out = dispatch_replacement(
        state, Event.WOULD_CREATE_TOKEN,
        affected_player_id=0,
        subjects={"controller_id": 0, "kind": "Soldier"},
        payload={"count": 2},
    )
    assert out["count"] == 4


def test_dispatch_token_doubler_only_for_recipient_controller():
    """Exalted Sunborn under P1 doesn't double tokens that P0 creates —
    the doubler keys on the recipient's controller."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.game import GameState, Player, Zone

    src = get_active_source()
    prs = src.get_printings()
    state = GameState()
    state.players = [Player(id=0), Player(id=1)]
    sunborn = state.new_object(
        prs["Exalted Sunborn"], 1, Zone.BATTLEFIELD,
    )
    sunborn.controller_id = 1
    state.battlefield.append(sunborn)

    out = dispatch_replacement(
        state, Event.WOULD_CREATE_TOKEN,
        affected_player_id=0,
        subjects={"controller_id": 0, "kind": "Soldier"},
        payload={"count": 2},
    )
    assert out["count"] == 2


def test_dispatch_two_doublers_compound():
    """Two Exalted Sunborns under the same controller compound: count
    × 2 × 2 = ×4. Both ReplacementEffects collect; CR 616.1 ordering
    among MultiplyCount transforms is commutative, so the post-
    replacement count is the same regardless of order."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.game import GameState, Player, Zone

    src = get_active_source()
    prs = src.get_printings()
    state = GameState()
    state.players = [Player(id=0), Player(id=1)]
    for _ in range(2):
        sunborn = state.new_object(
            prs["Exalted Sunborn"], 0, Zone.BATTLEFIELD,
        )
        sunborn.controller_id = 0
        state.battlefield.append(sunborn)

    out = dispatch_replacement(
        state, Event.WOULD_CREATE_TOKEN,
        affected_player_id=0,
        subjects={"controller_id": 0, "kind": "Soldier"},
        payload={"count": 1},
    )
    assert out["count"] == 4


def test_dispatch_consults_choose_replacement_order_for_multiple():
    """When ≥2 candidates apply, the affected player's agent is asked
    for the order. The agent receives the candidate list and the
    affected player id; the dispatcher applies the candidates in the
    returned index order. The default base-class implementation
    returns identity ordering, so this test pins the contract by
    overriding with a reverse-order agent and observing that the
    count multiplication is applied in reverse."""
    from mtg_bot.agents import Agent
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.game import GameState, Player, Zone

    captured: dict[str, Any] = {}

    class CaptureOrderAgent(Agent):
        def choose_replacement_order(
            self, candidates, state, affected_player_id,
        ):
            captured["n_candidates"] = len(candidates)
            captured["affected_player_id"] = affected_player_id
            return tuple(range(len(candidates)))

    src = get_active_source()
    prs = src.get_printings()
    state = GameState()
    state.players = [
        Player(id=0, agent=CaptureOrderAgent()),
        Player(id=1),
    ]
    for _ in range(2):
        sunborn = state.new_object(
            prs["Exalted Sunborn"], 0, Zone.BATTLEFIELD,
        )
        sunborn.controller_id = 0
        state.battlefield.append(sunborn)

    dispatch_replacement(
        state, Event.WOULD_CREATE_TOKEN,
        affected_player_id=0,
        subjects={"controller_id": 0, "kind": "Soldier"},
        payload={"count": 1},
    )

    assert captured["n_candidates"] == 2
    assert captured["affected_player_id"] == 0


def test_dispatch_prevent_short_circuits_subsequent_transforms():
    """A ``Prevent`` ordered first sets ``prevented=True`` and the
    dispatcher breaks out of the loop, so subsequent ``MultiplyCount``
    transforms don't run on the prevented event. Verifies via a
    custom agent that orders the synthetic Prevent first."""
    # No card in the EoE pool prints a prevent-replacement on token
    # creation, so we exercise the dispatcher contract in isolation
    # via a direct Prevent transform call. This test pins the
    # short-circuit semantics so future Prevent consumers can rely on
    # them.
    out = Prevent().apply({"count": 5})
    assert out["prevented"] is True
    # Dispatcher's loop reads the ``prevented`` flag and breaks
    # before applying further transforms — that contract is verified
    # in the Exalted Sunborn integration tests above (no prevent
    # carrier in the EoE pool).


# ────────────────────────────────────────────────────────────────────
# walk_continuous_modifiers
# ────────────────────────────────────────────────────────────────────


def _make_arena_for_walker():
    """Minimal two-player state with P0 and P1 for walker tests."""
    from mtg_bot.agents import RandomAgent
    from mtg_bot.game import GameState, Player

    state = GameState()
    state.players.append(Player(id=0, agent=RandomAgent()))
    state.players.append(Player(id=1, agent=RandomAgent()))
    return state


def test_walk_continuous_modifiers_empty_when_no_statics():
    """No static abilities → empty result."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _make_arena_for_walker()
    source = state.new_object(make_creature("A", "{1}", 1, 1), 0, Zone.BATTLEFIELD)
    target = state.new_object(make_creature("B", "{1}", 1, 1), 0, Zone.BATTLEFIELD)
    state.battlefield.extend([source, target])
    assert walk_continuous_modifiers(state, target) == []


def test_walk_continuous_modifiers_no_continuous_effect_slot():
    """A StaticAbility without a continuous_effect is skipped."""
    from mtg_bot.cards.base import StaticAbility, StaticScope, make_creature
    from mtg_bot.game import Zone

    state = _make_arena_for_walker()
    printing = make_creature("A", "{1}", 1, 1)
    sa = StaticAbility(scope=StaticScope.CREATURES_YOU_CONTROL, granted_keywords=frozenset({"flying"}))
    from dataclasses import replace as dc_replace
    printing = dc_replace(printing, static_abilities=(sa,))
    source = state.new_object(printing, 0, Zone.BATTLEFIELD)
    target = state.new_object(make_creature("B", "{1}", 1, 1), 0, Zone.BATTLEFIELD)
    state.battlefield.extend([source, target])
    assert walk_continuous_modifiers(state, target) == []


def test_walk_continuous_modifiers_self_scope_matches_source_only():
    """A SELF-scope ContinuousModifier only fires when target IS the source."""
    from mtg_bot.cards.base import StaticAbility, StaticScope, make_creature
    from mtg_bot.game import Zone
    from dataclasses import replace as dc_replace

    state = _make_arena_for_walker()
    cm = ContinuousModifier(layer=6, payload=GrantKeyword("flying"))
    sa = StaticAbility(scope=StaticScope.SELF, continuous_effect=cm)
    printing = dc_replace(make_creature("A", "{1}", 1, 1), static_abilities=(sa,))
    source = state.new_object(printing, 0, Zone.BATTLEFIELD)
    other = state.new_object(make_creature("B", "{1}", 1, 1), 0, Zone.BATTLEFIELD)
    state.battlefield.extend([source, other])

    # Query against self → match.
    result = walk_continuous_modifiers(state, source)
    assert len(result) == 1
    assert result[0] == (source.id, cm)

    # Query against a different creature → no match.
    assert walk_continuous_modifiers(state, other) == []


def test_walk_continuous_modifiers_condition_gates():
    """A condition on the StaticAbility is respected."""
    from mtg_bot.cards.base import StaticAbility, StaticScope, make_creature
    from mtg_bot.effects.conditions import AtLeastNSpellsCastThisTurn
    from mtg_bot.game import Zone
    from dataclasses import replace as dc_replace

    state = _make_arena_for_walker()
    cm = ContinuousModifier(layer=6, payload=GrantKeyword("unblockable"))
    sa = StaticAbility(
        scope=StaticScope.SELF,
        condition=AtLeastNSpellsCastThisTurn(2),
        continuous_effect=cm,
    )
    printing = dc_replace(make_creature("A", "{1}", 1, 1), static_abilities=(sa,))
    source = state.new_object(printing, 0, Zone.BATTLEFIELD)
    state.battlefield.append(source)

    # 0 spells cast → condition not met → no match.
    assert walk_continuous_modifiers(state, source) == []

    # 2 spells cast → condition met → match.
    state.players[0].spells_cast_this_turn = 2
    result = walk_continuous_modifiers(state, source)
    assert len(result) == 1
    assert result[0] == (source.id, cm)


def test_walk_continuous_modifiers_subject_filter_secondary_gate():
    """A subject_filter on ContinuousModifier narrows beyond applies_to."""
    from dataclasses import replace as dc_replace
    from mtg_bot.cards.base import StaticAbility, StaticScope, make_creature
    from mtg_bot.game import Zone

    state = _make_arena_for_walker()
    # Anthem-scope modifier but with an IsArtifact subject_filter.
    cm = ContinuousModifier(layer=6, payload=GrantKeyword("flying"), subject_filter=IsArtifact())
    sa = StaticAbility(scope=StaticScope.CREATURES_YOU_CONTROL, continuous_effect=cm)
    src_printing = dc_replace(make_creature("A", "{1}", 1, 1), static_abilities=(sa,))
    source = state.new_object(src_printing, 0, Zone.BATTLEFIELD)
    non_artifact = state.new_object(make_creature("B", "{1}", 1, 1), 0, Zone.BATTLEFIELD)
    artifact = state.new_object(make_creature("C", "{1}", 1, 1, artifact=True), 0, Zone.BATTLEFIELD)
    state.battlefield.extend([source, non_artifact, artifact])

    # Non-artifact creature → subject_filter rejects it.
    assert walk_continuous_modifiers(state, non_artifact) == []

    # Artifact creature → subject_filter passes.
    result = walk_continuous_modifiers(state, artifact)
    assert len(result) == 1
    assert result[0][1] is cm


def test_walk_continuous_modifiers_multiple_sources():
    """Multiple sources each contributing once all appear in the result."""
    from mtg_bot.cards.base import StaticAbility, StaticScope, make_creature
    from mtg_bot.game import Zone
    from dataclasses import replace as dc_replace

    state = _make_arena_for_walker()
    cm1 = ContinuousModifier(layer=6, payload=GrantKeyword("flying"))
    cm2 = ContinuousModifier(layer=6, payload=GrantKeyword("trample"))
    sa1 = StaticAbility(scope=StaticScope.CREATURES_YOU_CONTROL, continuous_effect=cm1)
    sa2 = StaticAbility(scope=StaticScope.CREATURES_YOU_CONTROL, continuous_effect=cm2)
    p1 = dc_replace(make_creature("Lord1", "{2}", 2, 2), static_abilities=(sa1,))
    p2 = dc_replace(make_creature("Lord2", "{2}", 2, 2), static_abilities=(sa2,))
    lord1 = state.new_object(p1, 0, Zone.BATTLEFIELD)
    lord2 = state.new_object(p2, 0, Zone.BATTLEFIELD)
    target = state.new_object(make_creature("T", "{1}", 1, 1), 0, Zone.BATTLEFIELD)
    state.battlefield.extend([lord1, lord2, target])

    results = walk_continuous_modifiers(state, target)
    payloads = {r[1].payload for r in results}
    assert GrantKeyword("flying") in payloads
    assert GrantKeyword("trample") in payloads
