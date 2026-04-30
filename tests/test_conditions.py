"""Tests for the ``Condition`` grammar that drives cost-reduction
rules. Each variant is exercised in isolation with light-weight fake
objects where the engine surface area is small (no GameState-internal
helpers) and against a real ``GameState`` where the variant reads
through ``state.player_by_id`` / ``state.battlefield`` /
``state.active_player`` and the boolean / multiplier behavior needs to
be observed end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mtg_bot.effects.conditions import (
    AndCondition,
    AnotherSpellCastThisTurn,
    AtLeastNSpellsCastThisTurn,
    AttackedWithThisTurn,
    Condition,
    ControlledArtifactEnteredThisTurn,
    ControlPermanent,
    DuringYourTurn,
    NotCondition,
    NthSpellThisTurn,
    OpponentsControlAtLeast,
    OrCondition,
    PerControlled,
    PerDifferentlyNamedLand,
    TargetMatches,
    VoidActiveCondition,
)
from mtg_bot.effects.object_predicates import (
    HasCounter,
    HasSubtype,
    IsArtifact,
    IsBlocking,
    IsCreature,
    IsLand,
    IsTapped,
)


# ---------- a stub Condition for combinator tests ----------


@dataclass(frozen=True)
class _Static(Condition):
    """A constant-value condition for exercising the combinator algebra
    independently of any state shape. ``value`` drives ``matches``;
    ``factor`` drives ``multiplier`` (defaults to 1, matching the
    Condition base default)."""

    value: bool
    factor: int = 1

    def matches(self, state, viewer_id, chosen_targets=None):
        return self.value

    def multiplier(self, state, viewer_id, chosen_targets=None):
        return self.factor


# ---------- combinator algebra ----------


def test_and_condition_true_iff_both_true():
    state = object()
    assert AndCondition(_Static(True), _Static(True)).matches(state, 0) is True
    assert AndCondition(_Static(True), _Static(False)).matches(state, 0) is False
    assert AndCondition(_Static(False), _Static(True)).matches(state, 0) is False
    assert AndCondition(_Static(False), _Static(False)).matches(state, 0) is False


def test_or_condition_true_iff_either_true():
    state = object()
    assert OrCondition(_Static(True), _Static(True)).matches(state, 0) is True
    assert OrCondition(_Static(True), _Static(False)).matches(state, 0) is True
    assert OrCondition(_Static(False), _Static(True)).matches(state, 0) is True
    assert OrCondition(_Static(False), _Static(False)).matches(state, 0) is False


def test_not_condition_inverts():
    state = object()
    assert NotCondition(_Static(True)).matches(state, 0) is False
    assert NotCondition(_Static(False)).matches(state, 0) is True


def test_operator_overloads_build_combinators():
    """``&`` / ``|`` / ``~`` produce And/Or/Not nodes equivalent to
    explicit construction. Frozen dataclass equality checks the tree
    is value-equal."""
    a = _Static(True)
    b = _Static(False)
    assert (a & b) == AndCondition(a, b)
    assert (a | b) == OrCondition(a, b)
    assert (~a) == NotCondition(a)


def test_combinator_tree_is_value_equal():
    a = ControlPermanent(IsCreature() & HasCounter("+1/+1"))
    b = ControlPermanent(IsCreature() & HasCounter("+1/+1"))
    assert a == b


def test_and_multiplier_is_product():
    state = object()
    cond = AndCondition(_Static(True, factor=2), _Static(True, factor=3))
    assert cond.multiplier(state, 0) == 6


def test_default_multiplier_is_one():
    """A bare boolean Condition reports multiplier=1 — the engine
    multiplies amount * 1 = amount, matching the historic flat-amount
    semantics for non-variable rules."""
    cond = _Static(True)
    assert cond.multiplier(object(), 0) == 1


def test_or_condition_default_multiplier_is_one():
    """OR doesn't override ``multiplier`` (only AND does, multiplying
    children); inheriting the base Condition default of 1 means the
    OR of two boolean conditions still yields a flat-amount
    reduction at the engine level."""
    cond = OrCondition(_Static(True), _Static(False))
    assert cond.multiplier(object(), 0) == 1


def test_not_condition_default_multiplier_is_one():
    cond = NotCondition(_Static(False))
    assert cond.multiplier(object(), 0) == 1


def test_condition_subclass_must_implement_matches():
    """Concrete ``Condition`` subclasses must implement ``matches`` —
    the ABC will refuse to instantiate a subclass that doesn't."""
    class _Bare(Condition):
        pass
    with pytest.raises(TypeError):
        _Bare()


# ---------- ControlPermanent ----------


def _new_state():
    """Two-player ``GameState`` with sentinel agents — small enough to
    be fast, real enough that ``state.battlefield`` / ``player_by_id``
    work as the conditions expect."""
    from mtg_bot.agents import RandomAgent
    from mtg_bot.game import GameState, Player

    state = GameState()
    state.players.append(Player(id=0, agent=RandomAgent()))
    state.players.append(Player(id=1, agent=RandomAgent()))
    return state


def test_control_permanent_matches_when_filter_satisfied():
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    creature = state.new_object(
        make_creature("Beater", "{1}{G}", 2, 2), 0, Zone.BATTLEFIELD,
    )
    creature.plus_one_counters = 1
    state.battlefield.append(creature)
    cond = ControlPermanent(IsCreature() & HasCounter("+1/+1"))
    assert cond.matches(state, 0) is True


def test_control_permanent_rejects_no_match():
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    plain = state.new_object(
        make_creature("Plain", "{1}{G}", 2, 2), 0, Zone.BATTLEFIELD,
    )
    state.battlefield.append(plain)
    cond = ControlPermanent(IsCreature() & HasCounter("+1/+1"))
    assert cond.matches(state, 0) is False


def test_control_permanent_scopes_to_viewer():
    """Permanents the OPPONENT controls don't satisfy a viewer-scoped
    ``ControlPermanent``."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    creature = state.new_object(
        make_creature("Foreigner", "{1}{G}", 2, 2), 1, Zone.BATTLEFIELD,
    )
    creature.plus_one_counters = 1
    state.battlefield.append(creature)
    cond = ControlPermanent(IsCreature() & HasCounter("+1/+1"))
    assert cond.matches(state, 0) is False
    assert cond.matches(state, 1) is True


# ---------- OpponentsControlAtLeast ----------


def test_opponents_control_at_least_threshold_met():
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    for i in range(3):
        c = state.new_object(
            make_creature(f"Foe{i}", "{1}{R}", 1, 1), 1, Zone.BATTLEFIELD,
        )
        state.battlefield.append(c)
    cond = OpponentsControlAtLeast(IsCreature(), 3)
    assert cond.matches(state, 0) is True


def test_opponents_control_at_least_threshold_not_met():
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    for i in range(2):
        c = state.new_object(
            make_creature(f"Foe{i}", "{1}{R}", 1, 1), 1, Zone.BATTLEFIELD,
        )
        state.battlefield.append(c)
    cond = OpponentsControlAtLeast(IsCreature(), 3)
    assert cond.matches(state, 0) is False


def test_opponents_control_at_least_skips_viewer_permanents():
    """The viewer's own creatures don't count toward the opponent
    threshold even when the filter would match them."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    for i in range(3):
        c = state.new_object(
            make_creature(f"Mine{i}", "{1}{G}", 1, 1), 0, Zone.BATTLEFIELD,
        )
        state.battlefield.append(c)
    cond = OpponentsControlAtLeast(IsCreature(), 3)
    assert cond.matches(state, 0) is False


# ---------- AnotherSpellCastThisTurn ----------


def test_another_spell_cast_this_turn_true_after_first_spell():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 1
    assert AnotherSpellCastThisTurn().matches(state, 0) is True


def test_another_spell_cast_this_turn_false_at_zero():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 0
    assert AnotherSpellCastThisTurn().matches(state, 0) is False


# ---------- NthSpellThisTurn ----------


def test_nth_spell_this_turn_2_matches_when_about_to_cast_second():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 1
    assert NthSpellThisTurn(2).matches(state, 0) is True


def test_nth_spell_this_turn_2_rejects_first_spell():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 0
    assert NthSpellThisTurn(2).matches(state, 0) is False


def test_nth_spell_this_turn_2_rejects_third_spell():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 2
    assert NthSpellThisTurn(2).matches(state, 0) is False


# ---------- AttackedWithThisTurn ----------


def test_attacked_with_this_turn_matches_filter():
    """If the latch contains a Spacecraft, the Spacecraft filter
    matches."""
    from mtg_bot.cards.base import make_artifact
    from mtg_bot.game import Zone

    state = _new_state()
    ship = state.new_object(
        make_artifact("Cruiser", "{4}", subtype="Spacecraft"),
        0, Zone.BATTLEFIELD,
    )
    state.players[0].attacked_this_turn = [ship]
    cond = AttackedWithThisTurn(HasSubtype("Spacecraft"))
    assert cond.matches(state, 0) is True


def test_attacked_with_this_turn_rejects_when_filter_unsatisfied():
    """Latch holds a permanent of the wrong subtype → no match."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    grunt = state.new_object(
        make_creature("Grunt", "{1}{G}", 2, 2), 0, Zone.BATTLEFIELD,
    )
    state.players[0].attacked_this_turn = [grunt]
    cond = AttackedWithThisTurn(HasSubtype("Spacecraft"))
    assert cond.matches(state, 0) is False


def test_attacked_with_this_turn_rejects_empty_latch():
    state = _new_state()
    state.players[0].attacked_this_turn = []
    cond = AttackedWithThisTurn(HasSubtype("Spacecraft"))
    assert cond.matches(state, 0) is False


# ---------- DuringYourTurn ----------


def test_during_your_turn_matches_active_player():
    state = _new_state()
    state.active_player_idx = 0
    assert DuringYourTurn().matches(state, 0) is True
    assert DuringYourTurn().matches(state, 1) is False


def test_during_your_turn_rejects_off_turn():
    state = _new_state()
    state.active_player_idx = 1
    assert DuringYourTurn().matches(state, 0) is False
    assert DuringYourTurn().matches(state, 1) is True


# ---------- VoidActiveCondition ----------


def test_void_active_condition_reads_state_helper(monkeypatch):
    """The condition delegates to ``mtg_bot.game.is_void_active`` —
    mock that and observe both branches."""
    import mtg_bot.game as game_module

    state = _new_state()
    monkeypatch.setattr(game_module, "is_void_active", lambda s: True)
    assert VoidActiveCondition().matches(state, 0) is True
    monkeypatch.setattr(game_module, "is_void_active", lambda s: False)
    assert VoidActiveCondition().matches(state, 0) is False


# ---------- TargetMatches ----------


def test_target_matches_optimistic_probe_matches_when_blocker_exists():
    """``chosen_targets is None`` (the legal_actions affordability
    probe) — the condition matches if ANY battlefield creature
    qualifies."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    blocker = state.new_object(
        make_creature("Wall", "{1}{W}", 0, 4), 0, Zone.BATTLEFIELD,
    )
    blocker.blocking = 99  # any non-None
    state.battlefield.append(blocker)
    cond = TargetMatches(IsCreature() & IsBlocking())
    assert cond.matches(state, 0, chosen_targets=None) is True


def test_target_matches_optimistic_probe_rejects_when_no_blocker():
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    plain = state.new_object(
        make_creature("Plain", "{1}{G}", 2, 2), 0, Zone.BATTLEFIELD,
    )
    state.battlefield.append(plain)
    cond = TargetMatches(IsCreature() & IsBlocking())
    assert cond.matches(state, 0, chosen_targets=None) is False


def test_target_matches_with_chosen_target_qualifies():
    """When the cast pipeline passes the committed target, the
    condition tests the picked creature directly."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    blocker = state.new_object(
        make_creature("Wall", "{1}{W}", 0, 4), 1, Zone.BATTLEFIELD,
    )
    blocker.blocking = 7
    state.battlefield.append(blocker)
    cond = TargetMatches(IsCreature() & IsBlocking())
    assert cond.matches(
        state, 0, chosen_targets=(("creature", blocker.id),),
    ) is True


def test_target_matches_with_chosen_non_qualifying_target():
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    plain = state.new_object(
        make_creature("Plain", "{1}{G}", 2, 2), 1, Zone.BATTLEFIELD,
    )
    state.battlefield.append(plain)
    cond = TargetMatches(IsCreature() & IsBlocking())
    assert cond.matches(
        state, 0, chosen_targets=(("creature", plain.id),),
    ) is False


def test_target_matches_skips_non_creature_target_kinds():
    """Player target kind doesn't trip the creature-only
    ``TargetMatches`` check."""
    state = _new_state()
    cond = TargetMatches(IsCreature() & IsBlocking())
    # No creatures at all — pure player target → no match.
    assert cond.matches(
        state, 0, chosen_targets=(("player", 1),),
    ) is False


def test_target_matches_skips_missing_target_id():
    """A target whose id no longer exists on the battlefield (the
    permanent left between cost evaluation and now) doesn't crash —
    the condition just fails to match for that entry."""
    state = _new_state()
    cond = TargetMatches(IsCreature() & IsBlocking())
    assert cond.matches(
        state, 0, chosen_targets=(("creature", 9999),),
    ) is False


def test_target_matches_tapped_creature_variant():
    """Same shape as the blocking variant but on the tapped
    qualifier — Seized from Slumber decomposition."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    napper = state.new_object(
        make_creature("Napper", "{2}{U}", 1, 4), 1, Zone.BATTLEFIELD,
    )
    napper.tapped = True
    state.battlefield.append(napper)
    cond = TargetMatches(IsCreature() & IsTapped())
    assert cond.matches(
        state, 0, chosen_targets=(("creature", napper.id),),
    ) is True


# ---------- PerDifferentlyNamedLand ----------


def test_per_differently_named_land_counts_distinct_names():
    from mtg_bot.cards.base import make_basic_land
    from mtg_bot.game import Zone

    state = _new_state()
    for name, color in (("Plains", "W"), ("Island", "U"), ("Swamp", "B")):
        land = state.new_object(
            make_basic_land(name, color), 0, Zone.BATTLEFIELD,
        )
        state.battlefield.append(land)
    cond = PerDifferentlyNamedLand()
    assert cond.matches(state, 0) is True
    assert cond.multiplier(state, 0) == 3


def test_per_differently_named_land_dedupes_duplicates():
    """Three Plains → one distinct name → multiplier 1."""
    from mtg_bot.cards.base import make_basic_land
    from mtg_bot.game import Zone

    state = _new_state()
    for _ in range(3):
        land = state.new_object(
            make_basic_land("Plains", "W"), 0, Zone.BATTLEFIELD,
        )
        state.battlefield.append(land)
    cond = PerDifferentlyNamedLand()
    assert cond.multiplier(state, 0) == 1


def test_per_differently_named_land_zero_when_no_lands():
    state = _new_state()
    cond = PerDifferentlyNamedLand()
    assert cond.matches(state, 0) is False
    assert cond.multiplier(state, 0) == 0


def test_per_differently_named_land_scopes_to_viewer():
    """Lands the OPPONENT controls don't count for a viewer-scoped
    multiplier."""
    from mtg_bot.cards.base import make_basic_land
    from mtg_bot.game import Zone

    state = _new_state()
    for name, color in (("Plains", "W"), ("Island", "U")):
        land = state.new_object(
            make_basic_land(name, color), 1, Zone.BATTLEFIELD,
        )
        state.battlefield.append(land)
    cond = PerDifferentlyNamedLand()
    assert cond.matches(state, 0) is False
    assert cond.multiplier(state, 0) == 0


# ---------- PerControlled ----------


def test_per_controlled_counts_filter_matches():
    from mtg_bot.cards.base import make_artifact
    from mtg_bot.game import Zone

    state = _new_state()
    for i in range(4):
        a = state.new_object(
            make_artifact(f"Trinket{i}", "{1}"), 0, Zone.BATTLEFIELD,
        )
        state.battlefield.append(a)
    cond = PerControlled(IsArtifact())
    assert cond.matches(state, 0) is True
    assert cond.multiplier(state, 0) == 4


def test_per_controlled_zero_when_no_match():
    state = _new_state()
    cond = PerControlled(IsArtifact())
    assert cond.matches(state, 0) is False
    assert cond.multiplier(state, 0) == 0


def test_per_controlled_subtype_filter():
    """``PerControlled(HasSubtype("Sliver"))`` covers Affinity for
    Slivers."""
    from mtg_bot.cards.base import make_creature
    from mtg_bot.game import Zone

    state = _new_state()
    sliver = state.new_object(
        make_creature(
            "Hivespine Sliver", "{1}{G}", 2, 2, subtypes=("Sliver",),
        ),
        0, Zone.BATTLEFIELD,
    )
    state.battlefield.append(sliver)
    cond = PerControlled(HasSubtype("Sliver"))
    assert cond.multiplier(state, 0) == 1


# ---------- IsLand sanity ----------


def test_is_land_compatible_with_per_differently_named_land():
    """``IsLand`` predicate evaluates true on a make_basic_land
    permanent (sanity check that the filter atoms compose with the
    condition grammar where wired)."""
    from mtg_bot.cards.base import make_basic_land
    from mtg_bot.game import Zone

    state = _new_state()
    land = state.new_object(
        make_basic_land("Plains", "W"), 0, Zone.BATTLEFIELD,
    )
    state.battlefield.append(land)
    assert IsLand().test(land, state) is True


# ---------- ControlledArtifactEnteredThisTurn ----------


def test_controlled_artifact_entered_this_turn_matches_when_positive():
    state = _new_state()
    state.players[0].artifacts_entered_under_your_control_this_turn = 1
    assert ControlledArtifactEnteredThisTurn().matches(state, 0) is True


def test_controlled_artifact_entered_this_turn_matches_more_than_one():
    state = _new_state()
    state.players[0].artifacts_entered_under_your_control_this_turn = 3
    assert ControlledArtifactEnteredThisTurn().matches(state, 0) is True


def test_controlled_artifact_entered_this_turn_rejects_when_zero():
    state = _new_state()
    assert state.players[0].artifacts_entered_under_your_control_this_turn == 0
    assert ControlledArtifactEnteredThisTurn().matches(state, 0) is False


def test_controlled_artifact_entered_this_turn_scopes_to_viewer():
    """P1's artifact-entered counter must not satisfy P0's condition."""
    state = _new_state()
    state.players[1].artifacts_entered_under_your_control_this_turn = 5
    assert ControlledArtifactEnteredThisTurn().matches(state, 0) is False
    assert ControlledArtifactEnteredThisTurn().matches(state, 1) is True


# ---------- AtLeastNSpellsCastThisTurn ----------


def test_at_least_n_spells_matches_exactly_n():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 2
    assert AtLeastNSpellsCastThisTurn(2).matches(state, 0) is True


def test_at_least_n_spells_matches_more_than_n():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 5
    assert AtLeastNSpellsCastThisTurn(2).matches(state, 0) is True


def test_at_least_n_spells_rejects_fewer_than_n():
    state = _new_state()
    state.players[0].spells_cast_this_turn = 1
    assert AtLeastNSpellsCastThisTurn(2).matches(state, 0) is False


def test_at_least_n_spells_rejects_zero():
    state = _new_state()
    assert state.players[0].spells_cast_this_turn == 0
    assert AtLeastNSpellsCastThisTurn(2).matches(state, 0) is False


def test_at_least_n_spells_scopes_to_viewer():
    """P1's cast counter must not satisfy P0's condition."""
    state = _new_state()
    state.players[1].spells_cast_this_turn = 3
    assert AtLeastNSpellsCastThisTurn(2).matches(state, 0) is False
    assert AtLeastNSpellsCastThisTurn(2).matches(state, 1) is True


def test_at_least_n_spells_n_equals_one():
    """n=1 matches AnotherSpellCastThisTurn semantics (>= 1)."""
    state = _new_state()
    state.players[0].spells_cast_this_turn = 1
    assert AtLeastNSpellsCastThisTurn(1).matches(state, 0) is True
    state.players[0].spells_cast_this_turn = 0
    assert AtLeastNSpellsCastThisTurn(1).matches(state, 0) is False
