"""``mtg_bot.search.blunder_census`` detectors on hand-built states."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtg_bot.actions import CastSpell, PassPriority
from mtg_bot.agents import RandomAgent
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
from mtg_bot.cards.source import get_active_source
from mtg_bot.game import StackItem, Zone
from mtg_bot.search.blunder_census import (
    DETECTORS, BlunderCensus, effect_harm_classes, has_benign_return,
)
from mtg_bot.simulate import setup_game
from mtg_bot.turn import Step
from tests._test_decks import simple_test_decklists


def _bare_state():
    eng = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=3,
    )
    eng.battlefield = []
    for p in eng.players:
        p.hand = []
        p.graveyard = []
        p.exile = []
    eng.stack = []
    eng.active_player_idx = 0
    eng.priority_player_idx = 0
    eng.step = Step.PRECOMBAT_MAIN
    return eng


def _creature(eng, pid, name="Wild Bear"):
    o = eng.new_object(SYNTHETIC_CREATURES[name], pid, Zone.BATTLEFIELD)
    o.controller_id = pid
    eng.battlefield.append(o)
    return o


def _view(eng):
    return SimpleNamespace(engine_state=eng)


def _bombard():
    return get_active_source().get_printings()["Bombard"]


def _push_bombard(eng, controller, target):
    card = eng.new_object(_bombard(), controller, Zone.STACK)
    item = StackItem(source=card, controller_id=controller, targets=(target,))
    eng.stack.append(item)
    return item


def test_effect_classification():
    harm = effect_harm_classes(_bombard().spell_effect)
    assert harm == {"DealDamageToTargetCreature"}
    assert effect_harm_classes(None) == set()
    assert not has_benign_return(_bombard().spell_effect)
    assert not has_benign_return(None)


def test_self_target_avoidable_vs_forced_vs_fine():
    eng = _bare_state()
    own = _creature(eng, 0)
    opp = _creature(eng, 1)
    c = BlunderCensus(seat=0)
    _push_bombard(eng, 0, ("creature", own.id))
    c.observe_moment(_view(eng))
    assert c.counts["self_target_avoidable"] == 1
    assert c.events[-1]["target"] == "Wild Bear"
    assert (c.targeted_items, c.harmful_targeted_items) == (1, 1)

    # Aimed at the opponent's creature: fine.
    eng.stack = []
    c2 = BlunderCensus(seat=0)
    _push_bombard(eng, 0, ("creature", opp.id))
    c2.after_decision(_view(eng))
    assert sum(c2.counts.values()) == 0

    # Only own creatures exist: forced.
    eng.stack = []
    eng.battlefield = [own]
    c3 = BlunderCensus(seat=0)
    _push_bombard(eng, 0, ("creature", own.id))
    c3.observe_moment(_view(eng))
    assert c3.counts["self_target_forced"] == 1

    # The OTHER seat's self-target is not ours to count.
    eng.stack = []
    c4 = BlunderCensus(seat=1)
    _push_bombard(eng, 0, ("creature", own.id))
    c4.observe_moment(_view(eng))
    assert sum(c4.counts.values()) == 0


def test_fizzle_when_every_target_is_gone_and_not_countered():
    eng = _bare_state()
    opp = _creature(eng, 1)
    c = BlunderCensus(seat=0)
    item = _push_bombard(eng, 0, ("creature", opp.id))
    c.observe_moment(_view(eng))
    eng.battlefield.remove(opp)          # target dies in response
    c.observe_moment(_view(eng))         # still on the stack, target gone
    eng.stack.remove(item)               # resolves -> does nothing
    c.after_decision(_view(eng))
    assert c.counts["fizzle"] == 1

    # Same, but countered: not a fizzle.
    eng2 = _bare_state()
    opp2 = _creature(eng2, 1)
    c2 = BlunderCensus(seat=0)
    item2 = _push_bombard(eng2, 0, ("creature", opp2.id))
    c2.observe_moment(_view(eng2))
    eng2.battlefield.remove(opp2)
    item2.countered = True
    c2.observe_moment(_view(eng2))
    eng2.stack.remove(item2)
    c2.finish(_view(eng2))
    assert c2.counts["fizzle"] == 0


def _attack_setup(blocker_dies: bool, defender_takes_damage: bool):
    eng = _bare_state()
    atk = _creature(eng, 0)
    blk = _creature(eng, 1, "Wall of Stone")
    atk.attacking = True
    atk.blocked_by = [blk.id]
    eng.step = Step.DECLARE_BLOCKERS
    c = BlunderCensus(seat=0)
    c.observe_moment(_view(eng))
    # Combat damage: attacker dies; maybe the blocker too.
    eng.battlefield.remove(atk)
    if blocker_dies:
        eng.battlefield.remove(blk)
    if defender_takes_damage:
        eng.players[1].life -= 3
    eng.step = Step.END_OF_COMBAT
    c.observe_moment(_view(eng))
    return c


def test_chump_attack_detected():
    c = _attack_setup(blocker_dies=False, defender_takes_damage=False)
    assert c.counts["chump_attack"] == 1
    assert c.counts["chump_attack_no_other_damage"] == 1


def test_chump_attack_with_other_damage_only_counts_raw():
    c = _attack_setup(blocker_dies=False, defender_takes_damage=True)
    assert c.counts["chump_attack"] == 1
    assert c.counts["chump_attack_no_other_damage"] == 0


def test_trade_is_not_a_chump_attack():
    c = _attack_setup(blocker_dies=True, defender_takes_damage=False)
    assert c.counts["chump_attack"] == 0


def test_unblocked_or_surviving_attacker_is_fine_and_finish_closes_combat():
    eng = _bare_state()
    atk = _creature(eng, 0)
    atk.attacking = True
    eng.step = Step.DECLARE_BLOCKERS
    c = BlunderCensus(seat=0)
    c.observe_moment(_view(eng))
    eng.battlefield.remove(atk)
    c.finish(_view(eng))
    assert c.counts["chump_attack"] == 0


def _main2_with_creature_in_hand(castable: bool):
    eng = _bare_state()
    eng.step = Step.POSTCOMBAT_MAIN
    p0 = eng.players[0]
    p0.hand.append(eng.new_object(SYNTHETIC_CREATURES["Trained Soldier"], 0, Zone.HAND))
    if castable:
        for _ in range(2):
            land = eng.new_object(BASIC_LANDS["Plains"], 0, Zone.BATTLEFIELD)
            land.controller_id = 0
            eng.battlefield.append(land)
    return eng


def test_missed_play_once_per_turn():
    eng = _main2_with_creature_in_hand(castable=True)
    c = BlunderCensus(seat=0)
    c.before_decision(_view(eng), 0, PassPriority())
    c.before_decision(_view(eng), 0, PassPriority())
    assert c.counts["missed_play_creature"] == 1
    assert c.events[0]["castable"] == ["Trained Soldier"]


@pytest.mark.parametrize("mutate", [
    "uncastable", "not_pass", "other_actor", "main1", "stack",
])
def test_missed_play_negative_cases(mutate):
    eng = _main2_with_creature_in_hand(castable=mutate != "uncastable")
    decision = PassPriority()
    actor = 0
    if mutate == "not_pass":
        decision = CastSpell(card_id=eng.players[0].hand[0].id)
    elif mutate == "other_actor":
        actor = 1
    elif mutate == "main1":
        eng.step = Step.PRECOMBAT_MAIN
    elif mutate == "stack":
        _push_bombard(eng, 1, ("player", 0))
    c = BlunderCensus(seat=0)
    c.before_decision(_view(eng), actor, decision)
    assert c.counts["missed_play_creature"] == 0
    assert c.counts["missed_play_other"] == 0


def test_detector_names_are_stable():
    assert set(BlunderCensus(seat=0).counts) == set(DETECTORS)


def test_counterspell_stack_target_is_not_a_fizzle_or_self_target():
    """Regression (2026-10-05 horizon run): a counterspell's
    ``("stack", index)`` target was read as an object id that is not
    on the battlefield -> false fizzle."""
    from mtg_bot.search import horizon_probe as hp
    eng = _bare_state()
    bear = eng.new_object(SYNTHETIC_CREATURES["Wild Bear"], 1, Zone.STACK)
    eng.stack.append(StackItem(source=bear, controller_id=1))
    counter = eng.new_object(
        get_active_source().get_printings()["Divert Disaster"], 0, Zone.STACK,
    )
    item = StackItem(source=counter, controller_id=0, targets=(("stack", 0),))
    eng.stack.append(item)
    c = BlunderCensus(seat=0)
    c.observe_moment(_view(eng))
    assert hp._target_label(eng, ("stack", 0), 0) == "spell Wild Bear(opp)"
    assert hp._target_label(eng, ("stack", 7), 0) == "stack[7]"
    eng.stack.remove(item)
    c.after_decision(_view(eng))
    assert sum(c.counts.values()) == 0
    assert c.targeted_items == 1 and c.harmful_targeted_items == 0
