"""Rule-verification tests — keywords.

Part of the split of the old monolithic tests/test_rules.py.
"""

from __future__ import annotations

from mtg_bot import apply, legal_moves
from mtg_bot.actions import (
    CastSpell,
    DeclareAttackers,
    DeclareBlockers,
    PassPriority,
    PlayLand,
)
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES, make_creature
from mtg_bot.effects import (
    DealDamageAnyTarget,
    DrawCards,
    EachPlayerDraws,
    GainLife,
    TargetOpponentDiscards,
    TriggeredAbility,
)
from mtg_bot.game import GameState, Player, StackItem, Zone
from mtg_bot.turn import (
    Step,
    draw_card,
    run_cleanup_step,
    run_untap_step,
    state_based_actions,
)


# --------------------------------------------------------------------------
# Scenario-building helpers (all tests go through legal_moves / apply)
# --------------------------------------------------------------------------


from tests._rules_helpers import *  # noqa: F401,F403

from mtg_bot.cards.source import get_active_source

_PRINTINGS = get_active_source().get_printings()


# --------------------------------------------------------------------------
# Rule: FLYING — can only be blocked by creatures with flying or reach
# --------------------------------------------------------------------------


def _flier(name="Test Flier", power=2, toughness=2):
    return make_creature(name, "{1}{U}", power, toughness, keywords={"flying"})


def _ground(name="Test Ground", power=2, toughness=2):
    return make_creature(name, "{1}{G}", power, toughness)


def _reach(name="Test Reacher", power=1, toughness=3):
    return make_creature(name, "{2}{G}", power, toughness, keywords={"reach"})


def test_ground_creature_cannot_block_a_flier():
    state = _arena(
        p0_battlefield=[_flier()],
        p1_battlefield=[_ground()],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # Block was filtered → flier got through unblocked
    assert state.players[1].life == 20 - 2
    assert blocker.blocking is None
    assert blocker in state.battlefield  # untouched


def test_flier_can_block_a_flier():
    state = _arena(
        p0_battlefield=[_flier("A", 2, 2)],
        p1_battlefield=[_flier("B", 2, 2)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # Both 2/2 fliers trade
    assert attacker not in state.battlefield
    assert blocker not in state.battlefield
    assert state.players[1].life == 20


def test_reach_creature_can_block_a_flier():
    state = _arena(
        p0_battlefield=[_flier("A", 2, 2)],
        p1_battlefield=[_reach("Spider", 1, 3)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # 2/2 flier vs 1/3 reacher: both survive (attacker takes 1, blocker takes 2)
    assert attacker in state.battlefield
    assert blocker in state.battlefield
    assert attacker.damage == 1
    assert blocker.damage == 2
    # The reach blocker successfully blocked the flier — no damage to player
    assert state.players[1].life == 20


def test_non_flying_attacker_can_be_blocked_normally():
    state = _arena(
        p0_battlefield=[_ground("A", 2, 2)],
        p1_battlefield=[_ground("B", 2, 2)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    assert attacker not in state.battlefield
    assert blocker not in state.battlefield


def test_flier_can_block_non_flying_attacker():
    # A flier CAN still choose to block a non-flier (not restricted from below).
    state = _arena(
        p0_battlefield=[_ground("A", 2, 2)],
        p1_battlefield=[_flier("B", 2, 2)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    assert attacker not in state.battlefield
    assert blocker not in state.battlefield


# --------------------------------------------------------------------------
# Rule: VIGILANCE — attacking doesn't tap
# --------------------------------------------------------------------------


def _vigilant(name="Test Vigilant", power=2, toughness=3):
    return make_creature(name, "{1}{W}", power, toughness, keywords={"vigilance"})


def test_vigilance_attacker_does_not_tap():
    state = _arena(p0_battlefield=[_vigilant()])
    guard = state.battlefield[0]
    state.step = Step.DECLARE_ATTACKERS
    apply(state, DeclareAttackers(attacker_ids=frozenset({guard.id})))
    assert guard.attacking is True
    assert guard.tapped is False


def test_non_vigilance_attacker_still_taps():
    state = _arena(p0_battlefield=[SYNTHETIC_CREATURES["Wild Bear"]])
    bear = state.battlefield[0]
    state.step = Step.DECLARE_ATTACKERS
    apply(state, DeclareAttackers(attacker_ids=frozenset({bear.id})))
    assert bear.attacking is True
    assert bear.tapped is True


def test_vigilance_attacker_still_eligible_blocker_this_turn():
    # Not strictly a keyword test but the natural consequence — since the
    # attacker didn't tap, it remains an eligible blocker if we needed to block.
    from mtg_bot.combat import eligible_blockers
    state = _arena(p0_battlefield=[_vigilant()])
    guard = state.battlefield[0]
    state.step = Step.DECLARE_ATTACKERS
    apply(state, DeclareAttackers(attacker_ids=frozenset({guard.id})))
    assert guard in eligible_blockers(state, player_id=0)


# --------------------------------------------------------------------------
# Rule: DEFENDER — can't attack
# --------------------------------------------------------------------------


def _defender(name="Test Wall", power=0, toughness=5):
    return make_creature(name, "{1}{W}", power, toughness, keywords={"defender"})


def test_defender_cannot_be_declared_as_attacker():
    from mtg_bot.combat import eligible_attackers

    state = _arena(p0_battlefield=[_defender()])
    wall = state.battlefield[0]
    assert wall not in eligible_attackers(state, player_id=0)


def test_defender_filtered_out_when_forced_as_attacker():
    state = _arena(p0_battlefield=[_defender()])
    wall = state.battlefield[0]
    state.step = Step.DECLARE_ATTACKERS
    apply(state, DeclareAttackers(attacker_ids=frozenset({wall.id})))
    assert wall.attacking is False
    assert wall.tapped is False


def test_defender_can_still_block():
    from mtg_bot.combat import eligible_blockers

    state = _arena(
        p0_battlefield=[SYNTHETIC_CREATURES["Wild Bear"]],
        p1_battlefield=[_defender("Wall", 0, 5)],
    )
    wall = state.battlefield[1]
    assert wall in eligible_blockers(state, player_id=1)


# --------------------------------------------------------------------------
# Rule: MECHAN SHIELDMATE — defender that can attack the turn an artifact
# entered the battlefield under your control. Uses a SELF-scope
# StaticAbility with ControlledArtifactEnteredThisTurn condition and a
# ContinuousModifier(RemoveKeyword("defender")); effective_keywords drops
# "defender" while the condition holds, so eligible_attackers passes.
# --------------------------------------------------------------------------


def test_mechan_shieldmate_cannot_attack_with_no_artifact_entering_this_turn():
    """Baseline: with no artifact entering this turn, Defender still
    blocks attack-eligibility — Mechan Shieldmate is a normal wall."""
    from mtg_bot.combat import eligible_attackers

    state = _arena(p0_battlefield=[_PRINTINGS["Mechan Shieldmate"]])
    shieldmate = state.battlefield[0]
    # _arena sets up battlefield directly, so the per-turn counter is 0.
    assert state.players[0].artifacts_entered_under_your_control_this_turn == 0
    assert shieldmate not in eligible_attackers(state, player_id=0)


def test_mechan_shieldmate_can_attack_when_artifact_entered_this_turn():
    """When ``put_into_battlefield`` runs for any artifact under the
    same controller, Mechan Shieldmate becomes a legal attacker."""
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.combat import eligible_attackers

    state = _arena(p0_battlefield=[_PRINTINGS["Mechan Shieldmate"]])
    shieldmate = state.battlefield[0]
    assert shieldmate not in eligible_attackers(state, player_id=0)
    # Now resolve a fresh artifact entering under P0's control. We use
    # Mechan Shieldmate itself as the entering printing — it doesn't
    # matter that there are now two on the battlefield for this assertion;
    # what matters is that put_into_battlefield bumps the per-turn counter.
    new_obj = state.new_object(
        _PRINTINGS["Mechan Shieldmate"], owner_id=0, zone=Zone.HAND,
    )
    put_into_battlefield(state, new_obj, controller_id=0)
    assert state.players[0].artifacts_entered_under_your_control_this_turn == 1
    # The original (non-summoning-sick) Shieldmate can now attack.
    assert shieldmate in eligible_attackers(state, player_id=0)


def test_mechan_shieldmate_attack_bypass_resets_at_cleanup():
    """The per-turn artifact-entered counter resets at cleanup, so the
    bypass only applies on the turn the artifact entered."""
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.combat import eligible_attackers

    state = _arena(p0_battlefield=[_PRINTINGS["Mechan Shieldmate"]])
    shieldmate = state.battlefield[0]
    new_obj = state.new_object(
        _PRINTINGS["Mechan Shieldmate"], owner_id=0, zone=Zone.HAND,
    )
    put_into_battlefield(state, new_obj, controller_id=0)
    assert shieldmate in eligible_attackers(state, player_id=0)
    # Now cleanup runs — counter should reset and Defender re-asserts.
    run_cleanup_step(state)
    assert state.players[0].artifacts_entered_under_your_control_this_turn == 0
    assert shieldmate not in eligible_attackers(state, player_id=0)


def test_mechan_shieldmate_attack_bypass_does_not_leak_across_controllers():
    """The bypass keys on the *controller's* per-turn counter — an
    artifact entering under the OPPONENT's control must not unlock
    Mechan Shieldmate for its controller."""
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.combat import eligible_attackers

    state = _arena(p0_battlefield=[_PRINTINGS["Mechan Shieldmate"]])
    shieldmate = state.battlefield[0]
    # P1 (the opponent) gets an artifact entering under their control.
    other_artifact = state.new_object(
        _PRINTINGS["Mechan Shieldmate"], owner_id=1, zone=Zone.HAND,
    )
    put_into_battlefield(state, other_artifact, controller_id=1)
    assert state.players[0].artifacts_entered_under_your_control_this_turn == 0
    assert state.players[1].artifacts_entered_under_your_control_this_turn == 1
    # P0's Shieldmate is still walled — no bypass.
    assert shieldmate not in eligible_attackers(state, player_id=0)


# --------------------------------------------------------------------------
# Rule: MENACE — can't be blocked by fewer than two creatures
# --------------------------------------------------------------------------


def _menacer(name="Test Menacer", power=3, toughness=2):
    return make_creature(name, "{2}{R}", power, toughness, keywords={"menace"})


def test_menace_attacker_cannot_be_blocked_by_one():
    state = _arena(
        p0_battlefield=[_menacer()],
        p1_battlefield=[_ground("Blocker", 2, 2)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # Single block filtered → menacer goes through unblocked
    assert state.players[1].life == 20 - 3
    assert attacker in state.battlefield
    assert blocker in state.battlefield
    assert blocker.blocking is None


def test_menace_attacker_can_be_blocked_by_two():
    state = _arena(
        p0_battlefield=[_menacer("Brute", 3, 3)],
        p1_battlefield=[_ground("B1", 2, 2), _ground("B2", 2, 2)],
    )
    attacker, b1, b2 = state.battlefield
    _simulate_combat_step(
        state, [attacker.id], [(b1.id, attacker.id), (b2.id, attacker.id)]
    )
    # Legal double-block stands: 3 damage goes to b1 (dies), then b2 gets
    # leftover; b1+b2 deal 2+2=4 back, attacker dies
    assert state.players[1].life == 20
    assert attacker not in state.battlefield


def test_menace_attacker_unblocked_is_unchanged():
    state = _arena(p0_battlefield=[_menacer()])
    attacker = state.battlefield[0]
    _simulate_combat_step(state, [attacker.id])
    # No blockers at all is fine — unblocked
    assert state.players[1].life == 20 - 3


# --------------------------------------------------------------------------
# Rule: TRAMPLE — excess combat damage spills to defending player
# --------------------------------------------------------------------------


def _trampler(name="Test Trampler", power=4, toughness=4):
    return make_creature(name, "{2}{G}", power, toughness, keywords={"trample"})


def test_trample_attacker_spills_excess_damage_to_player():
    state = _arena(
        p0_battlefield=[_trampler("Rhino", 4, 4)],
        p1_battlefield=[_ground("Chump", 1, 2)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # 4 power, blocker needs 2 to kill → 2 trample through
    assert state.players[1].life == 20 - 2
    assert blocker not in state.battlefield
    assert attacker in state.battlefield
    assert attacker.damage == 1  # blocker dealt 1 back


def test_trample_with_exactly_lethal_spills_nothing():
    state = _arena(
        p0_battlefield=[_trampler("Rhino", 3, 3)],
        p1_battlefield=[_ground("Blocker", 3, 3)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    assert state.players[1].life == 20
    # 3/3 vs 3/3 → both die, no trample spillover
    assert attacker not in state.battlefield
    assert blocker not in state.battlefield


def test_trample_against_multiple_blockers_spills_after_all_lethal():
    state = _arena(
        p0_battlefield=[_trampler("Giant Rhino", 6, 6)],
        p1_battlefield=[_ground("B1", 1, 2), _ground("B2", 1, 2)],
    )
    attacker, b1, b2 = state.battlefield
    _simulate_combat_step(
        state, [attacker.id], [(b1.id, attacker.id), (b2.id, attacker.id)]
    )
    # Lethal 2 to each blocker (4 total), 2 tramples through
    assert state.players[1].life == 20 - 2


def test_non_trample_excess_damage_stays_on_last_blocker():
    # Sanity check: without trample, over-assigned damage doesn't leak.
    state = _arena(
        p0_battlefield=[SYNTHETIC_CREATURES["Great Wurm"]],  # 6/6, no trample
        p1_battlefield=[SYNTHETIC_CREATURES["Sapling"]],  # 1/2
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    assert state.players[1].life == 20


# --------------------------------------------------------------------------
# Rule: DEATHTOUCH — any nonzero damage destroys a creature
# --------------------------------------------------------------------------


def _deathtoucher(name="Test Deathtoucher", power=1, toughness=1):
    return make_creature(name, "{1}{B}", power, toughness, keywords={"deathtouch"})


def test_deathtouch_kills_large_blocker_with_one_damage():
    state = _arena(
        p0_battlefield=[_deathtoucher("Serpent", 1, 1)],
        p1_battlefield=[_ground("Giant", 5, 5)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # Giant took 1 damage; without deathtouch it survives. With deathtouch, dies.
    assert blocker not in state.battlefield
    # Attacker took 5, dies too
    assert attacker not in state.battlefield


def test_deathtouch_splits_damage_across_multiple_blockers():
    # 2-power deathtoucher with 2 blockers: 1 damage each is lethal.
    state = _arena(
        p0_battlefield=[_deathtoucher("Hydra", 2, 2)],
        p1_battlefield=[_ground("B1", 3, 3), _ground("B2", 3, 3)],
    )
    attacker, b1, b2 = state.battlefield
    _simulate_combat_step(
        state, [attacker.id], [(b1.id, attacker.id), (b2.id, attacker.id)]
    )
    # Both blockers take 1 → marked lethal; attacker takes 6, dies
    assert b1 not in state.battlefield
    assert b2 not in state.battlefield
    assert attacker not in state.battlefield


def test_non_deathtouch_does_not_mark_lethal():
    # Control case: 1 damage to a 5/5 from a non-deathtouch source survives.
    state = _arena(
        p0_battlefield=[_ground("Tiny", 1, 1)],
        p1_battlefield=[_ground("Giant", 5, 5)],
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    assert blocker in state.battlefield
    assert blocker.damage == 1
    assert blocker.marked_lethal is False


def test_deathtouch_plus_trample_spills_correctly():
    # 3-power deathtouch+trample attacker vs 2 blockers (2 toughness each):
    # 1 damage per blocker (lethal via deathtouch) → 1 damage trample-through.
    dt_trample = make_creature(
        "Wurm", "{3}{G}", 3, 3, keywords={"deathtouch", "trample"}
    )
    state = _arena(
        p0_battlefield=[dt_trample],
        p1_battlefield=[_ground("B1", 2, 2), _ground("B2", 2, 2)],
    )
    attacker, b1, b2 = state.battlefield
    _simulate_combat_step(
        state, [attacker.id], [(b1.id, attacker.id), (b2.id, attacker.id)]
    )
    assert b1 not in state.battlefield
    assert b2 not in state.battlefield
    assert state.players[1].life == 20 - 1


def test_marked_lethal_resets_on_cleanup():
    state = _arena(p0_battlefield=[_ground("Survivor", 3, 3)])
    obj = state.battlefield[0]
    obj.marked_lethal = True
    run_cleanup_step(state)
    assert obj.marked_lethal is False


# --------------------------------------------------------------------------
# Rule: LIFELINK — damage dealt also gains the source's controller life
# --------------------------------------------------------------------------


def _lifelinker(name="Test Lifelinker", power=2, toughness=2):
    return make_creature(name, "{2}{W}", power, toughness, keywords={"lifelink"})


def test_lifelink_attacker_unblocked_gains_life_equal_to_damage():
    state = _arena(p0_battlefield=[_lifelinker("Cleric", 3, 3)], p0_life=15)
    attacker = state.battlefield[0]
    _simulate_combat_step(state, [attacker.id])
    assert state.players[1].life == 20 - 3
    assert state.players[0].life == 15 + 3


def test_lifelink_blocker_gains_life_from_damage_to_attacker():
    state = _arena(
        p0_battlefield=[_ground("Bear", 2, 2)],
        p1_battlefield=[_lifelinker("Cleric", 2, 2)],
        p1_life=15,
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # Blocker deals 2 to attacker → gains 2 life for P1
    assert state.players[1].life == 15 + 2


def test_lifelink_and_damage_both_happen_in_same_combat():
    state = _arena(
        p0_battlefield=[_lifelinker("Cleric", 3, 3)],
        p1_battlefield=[_ground("Bear", 2, 2)],
        p0_life=10,
    )
    attacker, blocker = state.battlefield
    _simulate_combat_step(state, [attacker.id], [(blocker.id, attacker.id)])
    # Attacker deals 3 → gains 3 life. Attacker takes 2 → damaged, alive.
    assert state.players[0].life == 10 + 3
    assert blocker not in state.battlefield
    assert attacker in state.battlefield
    assert attacker.damage == 2


def test_no_lifelink_does_not_gain_life():
    state = _arena(
        p0_battlefield=[SYNTHETIC_CREATURES["Wild Bear"]],  # no lifelink
        p0_life=10,
    )
    bear = state.battlefield[0]
    _simulate_combat_step(state, [bear.id])
    assert state.players[0].life == 10  # no change


# --------------------------------------------------------------------------
# Rule: HASTE — ignores summoning sickness
# --------------------------------------------------------------------------


def _haster(name="Test Haster", power=2, toughness=1):
    return make_creature(name, "{R}", power, toughness, keywords={"haste"})


def test_haste_creature_can_attack_while_summoning_sick():
    from mtg_bot.combat import eligible_attackers

    state = _arena(p0_battlefield=[_haster()])
    runner = state.battlefield[0]
    runner.summoning_sick = True  # just entered this turn
    assert runner in eligible_attackers(state, player_id=0)


def test_haste_creature_actually_attacks_and_deals_damage():
    state = _arena(p0_battlefield=[_haster()])
    runner = state.battlefield[0]
    runner.summoning_sick = True
    _simulate_combat_step(state, [runner.id])
    assert state.players[1].life == 20 - 2


def test_non_haste_sick_creature_is_filtered_out_of_attackers():
    state = _arena(p0_battlefield=[SYNTHETIC_CREATURES["Wild Bear"]])
    bear = state.battlefield[0]
    bear.summoning_sick = True
    state.step = Step.DECLARE_ATTACKERS
    apply(state, DeclareAttackers(attacker_ids=frozenset({bear.id})))
    assert bear.attacking is False


def test_flier_blocked_by_multiple_eligible_blockers():
    # Flier attacker, two reach blockers — both can block.
    state = _arena(
        p0_battlefield=[_flier("Dragon", 4, 4)],
        p1_battlefield=[_reach("Spider A", 2, 2), _reach("Spider B", 2, 2)],
    )
    attacker, b1, b2 = state.battlefield
    _simulate_combat_step(
        state, [attacker.id], [(b1.id, attacker.id), (b2.id, attacker.id)]
    )
    # Dragon 4/4 takes 2+2=4 → dies; blockers take 2 each (assignment order)
    # and also die since both are 2-tough
    assert attacker not in state.battlefield
    assert b1 not in state.battlefield
    assert b2 not in state.battlefield

