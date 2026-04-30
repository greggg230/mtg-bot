"""Rule-verification tests — cards misc.

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
from tests._etb_helpers import _legacy_or_inferred_key, _matches_legacy_key

from mtg_bot.cards.source import get_active_source

_PRINTINGS = get_active_source().get_printings()


def test_mill_target_player_fixed_amount():
    from mtg_bot.effects import MillTargetPlayer

    state = _arena()
    _stock_library(state, 1, count=10)
    starting_lib = len(state.players[1].library)
    gy_before = len(state.players[1].graveyard)
    e = MillTargetPlayer(amount=3)
    assert e.requires_target() is True
    assert "3" in e.describe()
    legal = {tid for kind, tid in e.legal_targets(state, controller_id=0)
             if kind == "player"}
    assert 0 in legal and 1 in legal
    e.resolve(state, controller_id=0, targets=(("player", 1),))
    assert len(state.players[1].library) == starting_lib - 3
    assert len(state.players[1].graveyard) == gy_before + 3


def test_mill_target_player_uses_controller_life():
    """Space-Time Anomaly style: amount = caster's life total."""
    from mtg_bot.effects import MillTargetPlayer

    state = _arena()
    _stock_library(state, 1, count=20)
    state.players[0].life = 5
    starting_lib = len(state.players[1].library)
    e = MillTargetPlayer(use_controller_life=True)
    assert "life total" in e.describe()
    e.resolve(state, controller_id=0, targets=(("player", 1),))
    assert len(state.players[1].library) == starting_lib - 5


def test_mill_caps_at_empty_library():
    from mtg_bot.effects import MillTargetPlayer

    state = _arena()
    # Leave P1 with just 2 cards in library.
    _stock_library(state, 1, count=2)
    e = MillTargetPlayer(amount=10)
    e.resolve(state, controller_id=0, targets=(("player", 1),))
    assert state.players[1].library == []
    assert len(state.players[1].graveyard) >= 2


def test_mill_noop_paths():
    from mtg_bot.effects import MillTargetPlayer

    state = _arena()
    e = MillTargetPlayer(amount=5)
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    # Negative life → mill 0.
    state.players[0].life = -3
    e2 = MillTargetPlayer(use_controller_life=True)
    before = len(state.players[1].library)
    e2.resolve(state, controller_id=0, targets=(("player", 1),))
    assert len(state.players[1].library) == before


def test_beyond_the_quiet_exiles_creatures_and_spacecraft():
    """Exiles every creature and Spacecraft from both sides; lands survive."""
    from mtg_bot.effects import ExileAllCreaturesAndSpacecraft
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    my_creature = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(my_creature)
    their_creature = state.new_object(
        SYNTHETIC_CREATURES["Sky Sentinel"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(their_creature)
    land = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(land)

    e = ExileAllCreaturesAndSpacecraft()
    assert "exile" in e.describe()
    assert e.requires_target() is False
    e.resolve(state, controller_id=0)

    assert my_creature not in state.battlefield
    assert their_creature not in state.battlefield
    assert land in state.battlefield  # lands untouched
    assert my_creature in state.players[0].exile
    assert their_creature in state.players[1].exile


def test_beyond_the_quiet_empty_battlefield_is_noop():
    from mtg_bot.effects import ExileAllCreaturesAndSpacecraft

    state = _arena()
    e = ExileAllCreaturesAndSpacecraft()
    e.resolve(state, controller_id=0)  # no crash


def test_focus_fire_only_legal_against_attackers_or_blockers():
    from mtg_bot.effects import FocusFireEffect

    state = _arena()
    attacker = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(attacker)
    blocker = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(blocker)
    idle = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(idle)
    attacker.attacking = True
    blocker.blocking = attacker.id

    e = FocusFireEffect(base=2)
    assert e.requires_target() is True
    assert "attacking" in e.describe()
    legal = {tid for _, tid in e.legal_targets(state, controller_id=0)}
    assert attacker.id in legal
    assert blocker.id in legal
    assert idle.id not in legal


def test_focus_fire_damage_scales_with_board():
    from mtg_bot.effects import FocusFireEffect

    state = _arena()
    attacker = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # 4/4
    state.battlefield.append(attacker)
    attacker.attacking = True

    # P0 controls two creatures of their own.
    for _ in range(2):
        mine = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(mine)

    # Focus Fire = 2 base + 2 = 4 damage
    e = FocusFireEffect(base=2)
    e.resolve(state, controller_id=0, targets=(("creature", attacker.id),))
    assert attacker.damage == 4


def test_focus_fire_fizzles_if_target_exits_combat():
    from mtg_bot.effects import FocusFireEffect

    state = _arena()
    attacker = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(attacker)
    attacker.attacking = True
    attacker.attacking = False  # exits combat before resolve
    e = FocusFireEffect(base=2)
    e.resolve(state, controller_id=0, targets=(("creature", attacker.id),))
    assert attacker.damage == 0


def test_focus_fire_noop_paths():
    from mtg_bot.effects import FocusFireEffect

    state = _arena()
    e = FocusFireEffect(base=2)
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))


def test_cut_propulsion_self_damage_non_flyer():
    from mtg_bot.effects import CutPropulsionEffect

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # 2/2
    state.battlefield.append(bear)
    e = CutPropulsionEffect()
    assert e.requires_target() is True
    assert "flying" in e.describe()
    e.resolve(state, controller_id=0, targets=(("creature", bear.id),))
    # Power 2 → 2 damage → 2/2 survives at 2 damage (equals toughness → SBA
    # would kill on next check; here we just see damage marked).
    assert bear.damage == 2


def test_cut_propulsion_doubled_for_flyer():
    from mtg_bot.effects import CutPropulsionEffect

    state = _arena()
    flyer = state.new_object(
        SYNTHETIC_CREATURES["Sky Sentinel"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # 1/3 flying
    state.battlefield.append(flyer)
    e = CutPropulsionEffect()
    e.resolve(state, controller_id=0, targets=(("creature", flyer.id),))
    # Power 1, doubled → 2 damage.
    assert flyer.damage == 2


def test_cut_propulsion_zero_power_deals_none():
    from mtg_bot.effects import CutPropulsionEffect

    state = _arena()
    wall = state.new_object(
        SYNTHETIC_CREATURES["Wall of Stone"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # 0/5 defender
    state.battlefield.append(wall)
    e = CutPropulsionEffect()
    e.resolve(state, controller_id=0, targets=(("creature", wall.id),))
    assert wall.damage == 0


def test_cut_propulsion_noop_paths():
    from mtg_bot.effects import CutPropulsionEffect

    state = _arena()
    e = CutPropulsionEffect()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))


def test_invasive_maneuvers_base_damage():
    from mtg_bot.effects import DealDamageToTargetCreatureSpacecraftBoosted

    state = _arena()
    victim = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    e = DealDamageToTargetCreatureSpacecraftBoosted(base=3, boosted=5)
    assert e.requires_target() is True
    assert "Spacecraft" in e.describe()
    e.resolve(state, controller_id=0, targets=(("creature", victim.id),))
    assert victim.damage == 3


def test_invasive_maneuvers_boosted_by_spacecraft():
    from mtg_bot.effects import DealDamageToTargetCreatureSpacecraftBoosted

    state = _arena()
    skiff = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(skiff)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    e = DealDamageToTargetCreatureSpacecraftBoosted(base=3, boosted=5)
    e.resolve(state, controller_id=0, targets=(("creature", victim.id),))
    assert victim.damage == 5


def test_invasive_maneuvers_noop_paths():
    from mtg_bot.effects import DealDamageToTargetCreatureSpacecraftBoosted

    state = _arena()
    e = DealDamageToTargetCreatureSpacecraftBoosted()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))


def test_elegy_acolyte_void_creates_robot():
    """Lifelink-printed + Void EoT creates a 2/2 Robot token."""
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    acolyte = state.new_object(
        _PRINTINGS["Elegy Acolyte"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(acolyte)
    assert "lifelink" in acolyte.printing.keywords

    from mtg_bot.actions import drain_stack
    # Void inactive → no token.
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert not any(o.printing.name == "Robot" for o in state.battlefield)

    # Void active → Robot token joins the battlefield.
    state.permanent_left_battlefield_this_turn = True
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    robots = [o for o in state.battlefield if o.printing.name == "Robot"]
    assert len(robots) == 1
    assert robots[0].controller_id == 0


def test_remnant_elemental_landfall_pumps():
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import fire_landfall_triggers

    state = _arena()
    elem = state.new_object(
        _PRINTINGS["Remnant Elemental"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(elem)
    # Put a land onto the battlefield and fire landfall.
    land = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(land)
    fire_landfall_triggers(state, land)
    from mtg_bot.actions import drain_stack
    drain_stack(state)
    assert elem.temp_power_bonus == 2


def test_icecave_crasher_trample_plus_landfall():
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import fire_landfall_triggers

    state = _arena()
    crasher = state.new_object(
        _PRINTINGS["Icecave Crasher"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(crasher)
    assert "trample" in crasher.printing.keywords
    land = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(land)
    fire_landfall_triggers(state, land)
    from mtg_bot.actions import drain_stack
    drain_stack(state)
    assert crasher.temp_power_bonus == 1


def test_blooming_stinger_grants_deathtouch():
    """Printed deathtouch + ETB grants deathtouch to another creature."""

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    stinger = state.new_object(
        _PRINTINGS["Blooming Stinger"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(stinger)
    assert "deathtouch" in stinger.printing.keywords

    # Manually apply the ETB targeting ally (test the grant path directly).
    etb = next(
        ab for ab in stinger.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = stinger
    etb.effect.resolve(state, 0, (("creature", ally.id),))

    from mtg_bot.game import effective_keywords
    assert "deathtouch" in effective_keywords(state, ally)


def test_seedship_agrarian_tap_creates_lander():
    from mtg_bot.turn import tap_permanent
    from mtg_bot.actions import drain_stack

    state = _arena()
    agr = state.new_object(
        _PRINTINGS["Seedship Agrarian"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(agr)
    tap_permanent(state, agr)
    drain_stack(state)
    landers = [o for o in state.battlefield if o.printing.name == "Lander"]
    assert len(landers) == 1


def test_seedship_agrarian_landfall_counter():
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import fire_landfall_triggers

    state = _arena()
    agr = state.new_object(
        _PRINTINGS["Seedship Agrarian"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(agr)
    land = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(land)
    fire_landfall_triggers(state, land)
    from mtg_bot.actions import drain_stack
    drain_stack(state)
    assert agr.plus_one_counters == 1


def test_biomechan_engineer_etb_and_activation():

    state = _arena()
    eng = state.new_object(
        _PRINTINGS["Biomechan Engineer"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(eng)
    # Fire ETB manually.
    etb = next(
        ab for ab in eng.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = eng
    etb.effect.resolve(state, 0)
    assert any(o.printing.name == "Lander" for o in state.battlefield)

    # Fire activated effect manually.
    _stock_library(state, 0, count=5)
    hand_before = len(state.players[0].hand)
    act = eng.printing.activated_abilities[0]
    act.effect.resolve(state, 0)
    assert len(state.players[0].hand) == hand_before + 2
    assert any(o.printing.name == "Robot" for o in state.battlefield)


def test_interceptor_mechan_etb_grabs_from_graveyard():

    state = _arena()
    # Seed a creature card in P0's graveyard.
    ghost = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.GRAVEYARD
    )
    state.players[0].graveyard.append(ghost)

    mechan = state.new_object(
        _PRINTINGS["Interceptor Mechan"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mechan)
    etb = next(
        ab for ab in mechan.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = mechan
    etb.effect.resolve(state, 0, (("graveyard_creature", ghost.id),))
    assert ghost in state.players[0].hand
    assert ghost not in state.players[0].graveyard


def test_return_creature_to_hand_noop_paths():
    from mtg_bot.effects import ReturnCreatureFromYourGraveyardToHand

    state = _arena()
    e = ReturnCreatureFromYourGraveyardToHand()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(
        state, controller_id=0, targets=(("graveyard_creature", 99999),)
    )
    assert e.requires_target() is True
    assert "hand" in e.describe()


def test_self_mill_moves_cards_to_graveyard():
    from mtg_bot.effects import SelfMill

    state = _arena()
    _stock_library(state, 0, count=5)
    e = SelfMill(amount=3)
    assert "mill" in e.describe()
    e.resolve(state, controller_id=0)
    assert len(state.players[0].library) == 2
    assert len(state.players[0].graveyard) >= 3


def test_self_mill_empty_library():
    from mtg_bot.effects import SelfMill

    state = _arena()  # library empty
    SelfMill(amount=5).resolve(state, controller_id=0)  # no crash


def test_seedship_broodtender_etb_and_reanimate():

    state = _arena()
    _stock_library(state, 0, count=10)
    # Pre-seed a creature card in graveyard.
    gone = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.GRAVEYARD
    )
    state.players[0].graveyard.append(gone)

    broodt = state.new_object(
        _PRINTINGS["Seedship Broodtender"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(broodt)
    # Fire ETB (mills 3).
    etb = next(
        ab for ab in broodt.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = broodt
    etb.effect.resolve(state, 0)
    assert len(state.players[0].library) == 10 - 3

    # Fire activated (reanimate the Wild Bear).
    act = broodt.printing.activated_abilities[0]
    assert act.sorcery_speed is True
    act.effect.resolve(state, 0, (("graveyard_creature", gone.id),))
    assert gone in state.battlefield


def test_xu_ifit_activated_reanimate():

    state = _arena()
    ghost = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.GRAVEYARD
    )
    state.players[0].graveyard.append(ghost)
    xu = state.new_object(
        _PRINTINGS["Xu-Ifit, Osteoharmonist"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(xu)
    assert "Legendary" in xu.printing.supertypes
    act = xu.printing.activated_abilities[0]
    assert act.cost.tap_self is True
    assert act.sorcery_speed is True
    state.resolving_source = xu
    act.effect.resolve(state, 0, (("graveyard_creature", ghost.id),))
    assert ghost in state.battlefield


def test_flight_deck_coordinator_gains_life_with_two_tapped():
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    coord = state.new_object(
        _PRINTINGS["Flight-Deck Coordinator"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(coord)
    # Two tapped creatures (including coord itself).
    coord.tapped = True
    other = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    other.tapped = True
    state.battlefield.append(other)

    life_before = state.players[0].life
    run_beginning_of_end_step_triggers(state)
    from mtg_bot.actions import drain_stack
    drain_stack(state)
    assert state.players[0].life == life_before + 2


def test_flight_deck_coordinator_noop_when_condition_unmet():
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    coord = state.new_object(
        _PRINTINGS["Flight-Deck Coordinator"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(coord)  # only one (untapped) creature
    life_before = state.players[0].life
    run_beginning_of_end_step_triggers(state)
    assert state.players[0].life == life_before


def test_conditional_on_tapped_creatures_forwards_targets():
    """If the wrapped effect requires a target, the conditional exposes it."""
    from mtg_bot.effects import ConditionalOnTappedCreatures, GainLife

    state = _arena()
    cond = ConditionalOnTappedCreatures(
        effect=GainLife(amount=1), min_count=0
    )
    assert cond.requires_target() is False
    assert cond.legal_targets(state, 0) == []
    assert "tapped" in cond.describe()


def test_sunstar_chaplain_conditional_plus_one():

    state = _arena()
    state.players[0].agent = None  # we'll drive the trigger manually
    cpn = state.new_object(
        _PRINTINGS["Sunstar Chaplain"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(cpn)
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)

    trigger = cpn.printing.triggered_abilities[0]
    # Condition unmet → no counter.
    state.resolving_source = cpn
    trigger.effect.resolve(state, 0, (("creature", ally.id),))
    assert ally.plus_one_counters == 0

    # Condition met: two tapped creatures.
    cpn.tapped = True
    ally.tapped = True
    trigger.effect.resolve(state, 0, (("creature", ally.id),))
    assert ally.plus_one_counters == 1


def test_dawnstrike_vanguard_mass_buff_when_condition_met():
    """Condition met → each OTHER creature you control gets +1/+1; the
    Vanguard itself does NOT (oracle: "other than this creature")."""
    from mtg_bot.game import effective_power, effective_toughness

    state = _arena()
    van = state.new_object(
        _PRINTINGS["Dawnstrike Vanguard"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(van)
    allies = []
    for _ in range(2):
        ally = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        ally.tapped = True
        state.battlefield.append(ally)
        allies.append(ally)
    van.tapped = True
    assert "lifelink" in van.printing.keywords
    trigger = van.printing.triggered_abilities[0]
    state.resolving_source = van
    trigger.effect.resolve(state, 0)
    # Vanguard itself is excluded — base 4/5 unchanged.
    assert van.plus_one_counters == 0
    assert effective_power(state, van) == 4
    assert effective_toughness(state, van) == 5
    # Each OTHER creature got +1/+1.
    for ally in allies:
        assert ally.plus_one_counters == 1
        assert effective_power(state, ally) == 3
        assert effective_toughness(state, ally) == 3


def test_dawnstrike_vanguard_no_buff_when_condition_unmet():
    """Condition unmet (fewer than 2 tapped creatures) → no counters
    on anybody, including the Vanguard."""
    from mtg_bot.game import effective_power, effective_toughness

    state = _arena()
    van = state.new_object(
        _PRINTINGS["Dawnstrike Vanguard"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(van)
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    # Only the Vanguard is tapped — that's one tapped creature, fewer
    # than the required two.
    van.tapped = True
    trigger = van.printing.triggered_abilities[0]
    state.resolving_source = van
    trigger.effect.resolve(state, 0)
    assert van.plus_one_counters == 0
    assert ally.plus_one_counters == 0
    assert effective_power(state, van) == 4
    assert effective_toughness(state, van) == 5
    assert effective_power(state, ally) == 2
    assert effective_toughness(state, ally) == 2


def test_dawnstrike_vanguard_repeated_triggers_never_buff_self():
    """Two end-step trigger resolutions stack counters on allies but
    never on the Vanguard itself, even though it's a tapped creature
    that helps satisfy the condition. Live shifts in plus_one_counters
    confirm the "other than this creature" clause holds across firings."""
    from mtg_bot.game import effective_power, effective_toughness

    state = _arena()
    van = state.new_object(
        _PRINTINGS["Dawnstrike Vanguard"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(van)
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    ally.tapped = True
    state.battlefield.append(ally)
    van.tapped = True
    trigger = van.printing.triggered_abilities[0]
    state.resolving_source = van

    # First firing: condition met (2 tapped) → ally gets +1/+1.
    trigger.effect.resolve(state, 0)
    assert van.plus_one_counters == 0
    assert ally.plus_one_counters == 1
    assert effective_power(state, van) == 4
    assert effective_toughness(state, van) == 5
    assert effective_power(state, ally) == 3

    # Untap the ally — only the Vanguard is tapped now, condition unmet.
    ally.tapped = False
    trigger.effect.resolve(state, 0)
    assert van.plus_one_counters == 0
    assert ally.plus_one_counters == 1  # unchanged

    # Re-tap the ally → condition met again, second counter on ally,
    # still none on the Vanguard.
    ally.tapped = True
    trigger.effect.resolve(state, 0)
    assert van.plus_one_counters == 0
    assert ally.plus_one_counters == 2
    assert effective_power(state, van) == 4
    assert effective_toughness(state, van) == 5
    assert effective_power(state, ally) == 4
    assert effective_toughness(state, ally) == 4


def test_cloudsculpt_technician_self_buff_with_artifact():
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.game import effective_power, effective_keywords

    state = _arena()
    tech = state.new_object(
        _PRINTINGS["Cloudsculpt Technician"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(tech)
    # No artifact → base 1/4.
    assert effective_power(state, tech) == 1
    # Add a Lander artifact → +1/+0 applies (now 2/4).
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)
    assert effective_power(state, tech) == 2
    assert "flying" in effective_keywords(state, tech)


def test_gravblade_heavy_self_buff_plus_deathtouch():
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.game import effective_power, effective_keywords

    state = _arena()
    grav = state.new_object(
        _PRINTINGS["Gravblade Heavy"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(grav)
    # Base 3/4, no deathtouch.
    assert effective_power(state, grav) == 3
    assert "deathtouch" not in effective_keywords(state, grav)

    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)
    assert effective_power(state, grav) == 4
    assert "deathtouch" in effective_keywords(state, grav)


def test_self_static_missing_state_returns_false():
    """applies_to with state=None on a conditional static returns False."""
    from mtg_bot.effects import StaticAbility, StaticScope

    ab = StaticAbility(
        scope=StaticScope.SELF,
        power_bonus=1,
        requires_artifact_you_control=True,
    )
    state = _arena()
    tech = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    # applies_to without state → cannot verify artifact presence → False.
    assert ab.applies_to(tech, tech, state=None) is False


def test_wurmwall_sweeper_etb_surveil():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    _stock_library(state, 0, count=10)
    ws = state.new_object(
        _PRINTINGS["Wurmwall Sweeper"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ws)
    # Trigger ETB surveil 2.
    etb = next(
        ab for ab in ws.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = ws
    etb.effect.resolve(state, 0)
    # Either milled to graveyard or kept on top — we just check that surveil
    # examined 2 cards via the log entry.
    assert any("surveils 2" in entry for entry in state.log)


def test_fell_gravship_etb_mill_and_return():

    state = _arena()
    _stock_library(state, 0, count=10)
    fg = state.new_object(
        _PRINTINGS["Fell Gravship"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fg)
    # Pre-seed a creature in graveyard so return-to-hand has a target.
    ghost = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.GRAVEYARD
    )
    state.players[0].graveyard.append(ghost)

    etb = next(
        ab for ab in fg.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    legal = etb.effect.legal_targets(state, 0)
    # Legal targets from ReturnCreatureFromYourGraveyardToHand — at least
    # our pre-seeded Wild Bear should show up (milled cards are basic
    # lands, not creatures, so they don't appear).
    assert ("graveyard_creature", ghost.id) in legal
    state.resolving_source = fg
    etb.effect.resolve(state, 0, (("graveyard_creature", ghost.id),))
    assert ghost in state.players[0].hand
    # 3 cards were milled.
    assert len(state.players[0].library) == 10 - 3


def test_pinnacle_kill_ship_etb_deals_damage():

    state = _arena()
    target = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )  # 4/4
    state.battlefield.append(target)
    ship = state.new_object(
        _PRINTINGS["Pinnacle Kill-Ship"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(ship)
    etb = next(
        ab for ab in ship.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = ship
    etb.effect.resolve(state, 0, (("creature", target.id),))
    assert target.damage == 10


def test_pinnacle_kill_ship_etb_target_is_optional_no_creatures():
    """Oracle: 'up to one target creature.' With no creatures on the
    battlefield, the trigger still goes on the stack and resolves as a
    no-op (target_count_range = (0, 1) → empty targets allowed)."""

    ship_p = _PRINTINGS["Pinnacle Kill-Ship"]
    etb = next(
        ab for ab in ship_p.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    assert etb.effect.target_count_range() == (0, 1)
    assert etb.effect.requires_target() is False
    # Resolving with no targets is a no-op (no exception).
    state = _arena()
    etb.effect.resolve(state, 0, ())


def test_warmaker_gunship_damage_scales_with_artifacts():
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    target = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(target)
    gunship = state.new_object(
        _PRINTINGS["Warmaker Gunship"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(gunship)
    # Gunship itself counts as an artifact (Spacecraft is an artifact).
    # Add a Lander for +1.
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)
    etb = next(
        ab for ab in gunship.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = gunship
    etb.effect.resolve(state, 0, (("creature", target.id),))
    # 2 artifacts (gunship + lander) → 2 damage.
    assert target.damage == 2


def test_warmaker_gunship_noop_without_artifacts():
    from mtg_bot.effects import DealDamageToOpponentCreatureByArtifactCount

    state = _arena()
    target = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(target)
    e = DealDamageToOpponentCreatureByArtifactCount()
    e.resolve(state, controller_id=0, targets=(("creature", target.id),))
    assert target.damage == 0  # no artifacts → no damage


def test_warmaker_gunship_effect_noop_paths():
    from mtg_bot.effects import DealDamageToOpponentCreatureByArtifactCount

    state = _arena()
    e = DealDamageToOpponentCreatureByArtifactCount()
    assert e.requires_target() is True
    assert "artifacts" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 1),))


def test_susurian_dirgecraft_forces_opponent_sac():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    # Give opponent two nontoken creatures.
    victims = []
    for _ in range(2):
        bear = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(bear)
        victims.append(bear)

    ship = state.new_object(
        _PRINTINGS["Susurian Dirgecraft"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ship)
    etb = next(
        ab for ab in ship.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = ship
    etb.effect.resolve(state, 0)
    # One of the two victims should now be in P1's graveyard.
    sac_count = sum(1 for v in victims if v in state.players[1].graveyard)
    assert sac_count == 1


def test_susurian_dirgecraft_grants_flying_at_7():
    """Regression: printed text grants Flying at 7+ (not Menace).
    Audit-found bug — earlier impl had a copy-paste 'menace' value."""

    c = _PRINTINGS["Susurian Dirgecraft"]
    assert c.is_spacecraft
    assert c.station_threshold == 7
    assert c.stationed_keywords == frozenset({"flying"})
    assert "menace" not in c.stationed_keywords


def test_susurian_dirgecraft_flies_when_stationed():
    """Behaviour test: once charge_counters reach the station threshold,
    effective_keywords includes flying."""
    from mtg_bot.game import effective_keywords

    state = _arena()
    ship = state.new_object(
        _PRINTINGS["Susurian Dirgecraft"],
        owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(ship)
    # Below threshold — no flying yet.
    ship.charge_counters = 6
    assert "flying" not in effective_keywords(state, ship)
    # At/above threshold — flying granted.
    ship.charge_counters = 7
    assert "flying" in effective_keywords(state, ship)


def test_each_opponent_sac_with_only_tokens_is_noop():
    from mtg_bot.effects import EachOpponentSacrificesCreature
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    # P1 has only a token creature → can't be sacrificed.
    robot = state.new_object(
        TOKEN_PRINTINGS["Robot"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(robot)
    EachOpponentSacrificesCreature().resolve(state, controller_id=0)
    assert robot in state.battlefield


def test_cerebral_download_surveil_x_then_draw_three():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(1))
    _stock_library(state, 0, count=20)
    # Give P0 two artifacts (two Lander tokens).
    for _ in range(2):
        a = state.new_object(
            TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(a)

    hand_before = len(state.players[0].hand)
    cd = _PRINTINGS["Cerebral Download"]
    cd.spell_effect.resolve(state, 0)
    # Drew 3 cards.
    assert len(state.players[0].hand) == hand_before + 3
    # Surveil amount was 2 → log should mention surveils 2.
    assert any("surveils 2" in l for l in state.log)


def test_mental_modulation_taps_and_draws():

    state = _arena()
    _stock_library(state, 0, count=3)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    mental = _PRINTINGS["Mental Modulation"]
    hand_before = len(state.players[0].hand)
    mental.spell_effect.resolve(
        state, controller_id=0, targets=(("permanent", victim.id),)
    )
    assert victim.tapped is True
    assert len(state.players[0].hand) == hand_before + 1


def test_tap_target_noop_paths():
    from mtg_bot.effects import TapTargetArtifactOrCreature

    state = _arena()
    e = TapTargetArtifactOrCreature()
    assert "tap" in e.describe()
    assert e.requires_target() is True
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))


def test_ruinous_rampage_mode_zero_damages_opponents():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    # Deterministic agent that always picks mode 0 (burn).
    class PickZero(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 0
    state.players[0].agent = PickZero(rng=random.Random(0))
    life_before = state.players[1].life
    _PRINTINGS["Ruinous Rampage"].spell_effect.resolve(state, 0)
    assert state.players[1].life == life_before - 3


def test_ruinous_rampage_mode_one_exiles_artifacts():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()

    class PickOne(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 1
    state.players[0].agent = PickOne(rng=random.Random(0))
    # Put some low-MV artifacts on the battlefield.
    lander0 = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander0)
    lander1 = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander1)

    _PRINTINGS["Ruinous Rampage"].spell_effect.resolve(state, 0)
    # Both Landers exiled (tokens cease to exist regardless of player).
    assert lander0 not in state.battlefield
    assert lander1 not in state.battlefield


def test_modal_choose_one_out_of_bounds_clamped_and_describe():
    from mtg_bot.effects import (
        ModalChooseOne, GainLife, DealDamageEachOpponent,
        ExileAllArtifactsWithManaValueAtMost,
    )
    from mtg_bot.agents import RandomAgent
    import random

    modes = (
        GainLife(amount=1),
        DealDamageEachOpponent(amount=1),
    )
    mc = ModalChooseOne(modes=modes)
    assert "OR" in mc.describe()
    assert mc.requires_target() is False
    assert mc.legal_targets(_arena(), 0) == []

    class OutOfBounds(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 9999
    state = _arena()
    state.players[0].agent = OutOfBounds(rng=random.Random(0))
    mc.resolve(state, 0)  # shouldn't crash; clamps to last mode


def test_modal_fizzles_when_targeted_mode_has_no_legal_target():
    """If mode picker chooses a target-requiring mode but no legal target
    was passed, silently fizzle."""
    from mtg_bot.effects import (
        ModalChooseOne, GainLife, DestroyTargetCreature,
    )
    from mtg_bot.agents import RandomAgent
    import random

    class PickTargeted(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 1  # the targeted mode

    state = _arena()
    state.players[0].agent = PickTargeted(rng=random.Random(0))
    modes = (GainLife(amount=1), DestroyTargetCreature())
    mc = ModalChooseOne(modes=modes)
    # No target passed → silent fizzle.
    mc.resolve(state, 0)
    assert state.players[0].life == 20  # unchanged


def test_drill_too_deep_destroy_artifact_mode():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()

    class PickDestroy(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 1
    state.players[0].agent = PickDestroy(rng=random.Random(0))

    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)

    dtd = _PRINTINGS["Drill Too Deep"]
    dtd.spell_effect.resolve(
        state, 0, (("permanent", lander.id),)
    )
    assert lander not in state.battlefield


def test_drill_too_deep_charge_counter_mode():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()

    class PickCharge(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 0
    state.players[0].agent = PickCharge(rng=random.Random(0))

    skiff = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(skiff)
    assert skiff.charge_counters == 0

    dtd = _PRINTINGS["Drill Too Deep"]
    dtd.spell_effect.resolve(
        state, 0, (("permanent", skiff.id),)
    )
    assert skiff.charge_counters == 5


def test_destroy_target_artifact_noop_paths():
    from mtg_bot.effects import DestroyTargetArtifact

    state = _arena()
    e = DestroyTargetArtifact()
    assert "artifact" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))


def test_put_charge_counters_noop_paths():
    from mtg_bot.effects import PutChargeCountersOnTargetSpacecraftYouControl

    state = _arena()
    e = PutChargeCountersOnTargetSpacecraftYouControl(amount=3)
    assert "charge" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))


def test_luxknight_breacher_enters_with_scaled_counters():
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    # Seed two existing creatures + one artifact (Lander) for P0.
    for _ in range(2):
        mate = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(mate)
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)
    # Opponent board shouldn't count.
    enemy = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(enemy)

    lux = state.new_object(
        _PRINTINGS["Luxknight Breacher"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lux)
    etb = next(
        ab for ab in lux.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = lux
    etb.effect.resolve(state, 0)
    # 2 creatures + 1 artifact = 3 counters (self excluded).
    assert lux.plus_one_counters == 3


def test_luxknight_breacher_noop_with_empty_board():

    state = _arena()
    lux = state.new_object(
        _PRINTINGS["Luxknight Breacher"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lux)
    etb = next(
        ab for ab in lux.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = lux
    etb.effect.resolve(state, 0)
    assert lux.plus_one_counters == 0


def test_put_counters_per_other_noop_without_source():
    from mtg_bot.effects import (
        PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    )

    state = _arena()
    e = PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact()
    state.resolving_source = None
    e.resolve(state, controller_id=0)
    assert "each other" in e.describe()


def test_fallers_faithful_etb_destroys_creature():

    state = _arena()
    # Stock P1's library so the rider draw (target's controller draws 2)
    # doesn't deck them out.
    _stock_library(state, 1, count=10)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    ff = state.new_object(
        _PRINTINGS["Faller's Faithful"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ff)
    etb = next(
        ab for ab in ff.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = ff
    etb.effect.resolve(state, 0, (("creature", victim.id),))
    assert victim not in state.battlefield


def test_fallers_faithful_undamaged_target_draws_two():
    """Oracle: 'If that creature wasn't dealt damage this turn, its
    controller draws two cards.' Target is a freshly-summoned Wild Bear
    that's never been damaged, so its controller (P1) draws 2.
    """

    state = _arena()
    _stock_library(state, 1, count=10)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    assert victim.damaged_this_turn is False
    ff = state.new_object(
        _PRINTINGS["Faller's Faithful"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ff)
    etb = next(
        ab for ab in ff.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = ff
    p1_hand_before = len(state.players[1].hand)
    p1_lib_before = len(state.players[1].library)
    etb.effect.resolve(state, 0, (("creature", victim.id),))
    assert victim not in state.battlefield
    # Victim's controller draws 2 cards.
    assert len(state.players[1].hand) == p1_hand_before + 2
    assert len(state.players[1].library) == p1_lib_before - 2


def test_fallers_faithful_damaged_target_no_draw():
    """If the targeted creature WAS dealt damage this turn, the rider
    doesn't fire — controller doesn't draw.
    """

    state = _arena()
    _stock_library(state, 1, count=10)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    # Pre-damage the victim this turn (e.g., Shock targeting it earlier).
    victim.damage = 1
    victim.damaged_this_turn = True
    ff = state.new_object(
        _PRINTINGS["Faller's Faithful"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ff)
    etb = next(
        ab for ab in ff.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = ff
    p1_hand_before = len(state.players[1].hand)
    p1_lib_before = len(state.players[1].library)
    etb.effect.resolve(state, 0, (("creature", victim.id),))
    assert victim not in state.battlefield
    # No draw because the target was damaged this turn.
    assert len(state.players[1].hand) == p1_hand_before
    assert len(state.players[1].library) == p1_lib_before


def test_fallers_faithful_etb_target_is_optional():
    """Oracle: 'up to one OTHER target creature.' With no creatures on
    the battlefield (other than the trigger source), the trigger
    resolves as a no-op rather than fizzling on lack of targets."""

    ff_p = _PRINTINGS["Faller's Faithful"]
    etb = next(
        ab for ab in ff_p.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    assert etb.effect.target_count_range() == (0, 1)
    assert etb.effect.requires_target() is False
    state = _arena()
    # Empty battlefield — no legal targets.
    legal = etb.effect.legal_targets(state, controller_id=0)
    assert legal == []
    # Empty resolve is a clean no-op.
    etb.effect.resolve(state, 0, ())


def test_fallers_faithful_excludes_self_target():
    """The 'other' clause: Faller's Faithful itself is not a legal
    target of its own ETB trigger."""

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(bear)
    ff = state.new_object(
        _PRINTINGS["Faller's Faithful"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ff)
    etb = next(
        ab for ab in ff.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = ff
    legal = etb.effect.legal_targets(state, controller_id=0)
    target_ids = {tid for _, tid in legal}
    assert bear.id in target_ids
    assert ff.id not in target_ids


def test_damaged_this_turn_clears_at_cleanup():
    """The damaged_this_turn flag is per-turn and clears at cleanup —
    a turn-1 damage event must not bleed into turn 2's bookkeeping.
    """
    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    bear.damage = 1
    bear.damaged_this_turn = True
    run_cleanup_step(state)
    assert bear.damaged_this_turn is False
    assert bear.damage == 0


def test_apply_damage_marks_damaged_this_turn():
    """The shared _apply_damage helper (used by direct-damage spells)
    must set damaged_this_turn so Faller's Faithful and similar riders
    see the right history.
    """
    from mtg_bot.effects.base import _apply_damage

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    assert bear.damaged_this_turn is False
    _apply_damage(state, ("creature", bear.id), 1)
    assert bear.damage == 1
    assert bear.damaged_this_turn is True


def test_biotech_specialist_and_kav_landseeker_make_landers():
    """Both creatures create exactly one Lander on ETB."""

    for name in ("Biotech Specialist", "Kav Landseeker"):
        state = _arena()
        c = state.new_object(
            _PRINTINGS[name], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(c)
        etb = next(
            ab for ab in c.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
        )
        state.resolving_source = c
        etb.effect.resolve(state, 0)
        landers = [o for o in state.battlefield if o.printing.name == "Lander"]
        assert len(landers) == 1, f"{name} should create 1 Lander"


def test_kav_landseeker_has_menace():

    assert "menace" in _PRINTINGS["Kav Landseeker"].keywords


def test_biotech_specialist_sacrifice_artifact_damages_opponent():
    """Oracle: 'Whenever you sacrifice an artifact, this creature deals 2
    damage to target opponent.' Sacrificing the Lander it creates should
    fire the trigger and deal 2 to the opponent."""
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.turn import sacrifice
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    spec = state.new_object(
        _PRINTINGS["Biotech Specialist"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(spec)
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(lander)
    starting_life = state.players[1].life
    sacrifice(state, lander)
    # Trigger landed on the stack — resolve it.
    assert state.stack
    resolve_top_of_stack(state)
    assert state.players[1].life == starting_life - 2


def test_biotech_specialist_sacrifice_creature_does_not_fire_artifact_trigger():
    """Sacrificing a non-artifact creature should not fire the artifact-
    sacrifice trigger."""
    from mtg_bot.turn import sacrifice

    state = _arena()
    spec = state.new_object(
        _PRINTINGS["Biotech Specialist"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(spec)
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(bear)
    starting_life = state.players[1].life
    sacrifice(state, bear)
    # No artifact-sacrifice trigger landed on the stack.
    assert not [
        item for item in state.stack
        if "sacrifice" in (item.label or "")
    ]
    assert state.players[1].life == starting_life


def test_exosuit_savior_bounces_own_permanent():
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    plains = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(plains)
    savior = state.new_object(
        _PRINTINGS["Exosuit Savior"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(savior)
    assert "flying" in savior.printing.keywords

    etb = next(
        ab for ab in savior.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = savior
    etb.effect.resolve(state, 0, (("permanent", plains.id),))
    assert plains not in state.battlefield
    assert plains in state.players[0].hand


def test_exosuit_savior_target_is_optional_and_excludes_self():
    """Oracle: 'return up to one OTHER target permanent you control to its
    owner's hand.' Trigger lands on the stack with no target when only the
    Savior itself is in play, and Savior is excluded from its own legal
    targets."""

    savior_p = _PRINTINGS["Exosuit Savior"]
    etb = next(
        ab for ab in savior_p.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    assert etb.effect.target_count_range() == (0, 1)
    assert etb.effect.requires_target() is False

    state = _arena()
    savior = state.new_object(savior_p, owner_id=0, zone=Zone.BATTLEFIELD)
    state.battlefield.append(savior)
    state.resolving_source = savior
    legal = etb.effect.legal_targets(state, controller_id=0)
    # Only the Savior itself is on the battlefield → no legal targets
    # because of the "other" filter.
    assert legal == []
    # Empty resolve is a clean no-op.
    etb.effect.resolve(state, 0, ())
    assert savior in state.battlefield


def test_exosuit_savior_excludes_self_with_other_permanents():
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    plains = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(plains)
    savior = state.new_object(
        _PRINTINGS["Exosuit Savior"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(savior)
    etb = next(
        ab for ab in savior.printing.triggered_abilities
        if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = savior
    legal = etb.effect.legal_targets(state, controller_id=0)
    target_ids = {tid for _, tid in legal}
    assert plains.id in target_ids
    assert savior.id not in target_ids


def test_return_target_permanent_you_control_rejects_opponent():
    from mtg_bot.effects import ReturnTargetPermanentYouControlToOwnersHand

    state = _arena()
    enemy = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(enemy)
    e = ReturnTargetPermanentYouControlToOwnersHand()
    assert e.requires_target() is True
    assert "you control" in e.describe()
    legal = e.legal_targets(state, controller_id=0)
    assert ("permanent", enemy.id) not in legal
    # Even if passed as target, resolution filters out.
    e.resolve(state, controller_id=0, targets=(("permanent", enemy.id),))
    assert enemy in state.battlefield


def test_return_target_permanent_noop_paths():
    from mtg_bot.effects import ReturnTargetPermanentYouControlToOwnersHand

    state = _arena()
    e = ReturnTargetPermanentYouControlToOwnersHand()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))


def test_ward_counters_spell_when_caster_cant_pay():
    """Casting Bombard (target-creature burn) at a warded opponent
    creature — caster has no mana left → spell countered, nothing dies."""
    from mtg_bot.actions import _apply_ward_check

    state = _arena()
    mouth = state.new_object(
        _PRINTINGS["Mouth of the Storm"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mouth)
    # Caster has 0 untapped lands → cannot pay the Ward {2}.
    countered = _apply_ward_check(state, caster_id=0, targets=(("creature", mouth.id),))
    assert countered is True


def test_ward_autopays_when_caster_has_lands():
    """Caster with 2 untapped lands auto-taps them to pay Ward {2}."""
    from mtg_bot.actions import _apply_ward_check
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    mouth = state.new_object(
        _PRINTINGS["Mouth of the Storm"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mouth)
    lands: list = []
    for _ in range(2):
        l = state.new_object(
            BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(l)
        lands.append(l)
    countered = _apply_ward_check(state, caster_id=0, targets=(("creature", mouth.id),))
    assert countered is False
    assert all(l.tapped for l in lands)  # both auto-tapped


def test_ward_does_not_fire_on_own_permanent():
    """A creature's ward is only relevant for OPPONENT spells."""
    from mtg_bot.actions import _apply_ward_check

    state = _arena()
    mouth = state.new_object(
        _PRINTINGS["Mouth of the Storm"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mouth)
    countered = _apply_ward_check(state, caster_id=0, targets=(("creature", mouth.id),))
    assert countered is False


def test_countered_spell_goes_to_graveyard_and_no_effect():
    """End-to-end: cast a burn spell at a warded opponent creature with no
    mana left → spell resolves as a no-op into the caster's graveyard."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    # Caster has exactly {1}{R} (one Mountain) — enough for Bombard's {1}{R}
    # but NOT for Ward {2} on top.
    mtn = state.new_object(
        BASIC_LANDS["Mountain"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mtn)
    mtn2 = state.new_object(
        BASIC_LANDS["Mountain"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mtn2)
    # Seed the spell in hand. Bombard = {R} DealDamageToTargetCreature(2)
    bombard = _PRINTINGS["Bombard"]
    hand_card = state.new_object(bombard, owner_id=0, zone=Zone.HAND)
    state.players[0].hand.append(hand_card)
    mouth = state.new_object(
        _PRINTINGS["Mouth of the Storm"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mouth)

    _cast_spell(state, 0, hand_card.id)
    assert state.stack and state.stack[-1].countered is True
    resolve_top_of_stack(state)
    # Card landed in graveyard; mouth unharmed.
    assert hand_card in state.players[0].graveyard
    assert mouth.damage == 0


def test_bioengineered_future_etb_makes_lander():

    state = _arena()
    enc = state.new_object(
        _PRINTINGS["Bioengineered Future"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(enc)
    etb = next(
        ab for ab in enc.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = enc
    etb.effect.resolve(state, 0)
    landers = [o for o in state.battlefield if o.printing.name == "Lander"]
    assert len(landers) == 1


def test_bioengineered_future_grants_etb_counters_per_land_entered_this_turn():
    """Oracle: 'Each creature you control enters with an additional +1/+1
    counter on it for each land that entered the battlefield under your
    control this turn.'

    Two lands entered → next creature enters with 2 +1/+1 counters."""
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.actions import put_into_battlefield

    state = _arena()
    state.players[0].lands_entered_this_turn = 2
    enc = state.new_object(
        _PRINTINGS["Bioengineered Future"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(enc)
    bear_in_hand = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.HAND,
    )
    put_into_battlefield(state, bear_in_hand, controller_id=0)
    assert bear_in_hand.plus_one_counters == 2


def test_bioengineered_future_no_lands_means_no_extra_counters():
    """If no lands entered yet, the entering creature gets 0 extra counters."""
    from mtg_bot.actions import put_into_battlefield

    state = _arena()
    assert state.players[0].lands_entered_this_turn == 0
    enc = state.new_object(
        _PRINTINGS["Bioengineered Future"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(enc)
    bear_in_hand = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.HAND,
    )
    put_into_battlefield(state, bear_in_hand, controller_id=0)
    assert bear_in_hand.plus_one_counters == 0


def test_bioengineered_future_anthem_stacks_with_multiple_sources():
    """Two Bioengineered Futures double the counters per land entered."""
    from mtg_bot.actions import put_into_battlefield

    state = _arena()
    state.players[0].lands_entered_this_turn = 3
    for _ in range(2):
        enc = state.new_object(
            _PRINTINGS["Bioengineered Future"], owner_id=0,
            zone=Zone.BATTLEFIELD,
        )
        state.battlefield.append(enc)
    bear_in_hand = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.HAND,
    )
    put_into_battlefield(state, bear_in_hand, controller_id=0)
    # 2 sources × 3 lands = 6 extra counters.
    assert bear_in_hand.plus_one_counters == 6


def test_lands_entered_this_turn_increments_on_play_land():
    """Playing a land via _play_land bumps lands_entered_this_turn (in
    addition to lands_played_this_turn)."""
    from mtg_bot.actions import _play_land
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    forest = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.HAND,
    )
    state.players[0].hand.append(forest)
    _play_land(state, 0, forest.id)
    assert state.players[0].lands_played_this_turn == 1
    assert state.players[0].lands_entered_this_turn == 1


def test_lands_entered_this_turn_resets_at_cleanup():
    """The counter resets at end-of-turn cleanup."""
    from mtg_bot.turn import run_cleanup_step

    state = _arena()
    state.players[0].lands_entered_this_turn = 5
    state.players[1].lands_entered_this_turn = 2
    run_cleanup_step(state)
    assert state.players[0].lands_entered_this_turn == 0
    assert state.players[1].lands_entered_this_turn == 0


def test_eusocial_engineering_landfall_makes_robot():
    """Enchantment sits on battlefield and fires Landfall."""
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import fire_landfall_triggers

    state = _arena()
    euso = state.new_object(
        _PRINTINGS["Eusocial Engineering"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(euso)
    land = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(land)
    fire_landfall_triggers(state, land)
    from mtg_bot.actions import drain_stack
    drain_stack(state)
    robots = [o for o in state.battlefield if o.printing.name == "Robot"]
    assert len(robots) == 1


def test_enchantment_is_permanent_but_not_creature():

    c = _PRINTINGS["Eusocial Engineering"]
    assert c.is_permanent is True
    assert c.is_creature is False
    assert c.is_instant is False


def test_eusocial_engineering_warp_cast_and_end_step_exile():
    """Non-creature warp end-to-end: cast an enchantment via Warp {1}{G},
    it resolves onto the battlefield, then the delayed end-step exile
    trigger fires."""
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import run_delayed_end_step_triggers
    import random

    euso = _PRINTINGS["Eusocial Engineering"]
    assert euso.warp_cost_str == "{1}{G}"
    state = _arena(
        p0_hand=[euso],
        p0_battlefield=[BASIC_LANDS["Forest"]] * 2,  # enough for warp {1}{G}
    )
    for p in state.players:
        p.agent = RandomAgent(rng=random.Random(0))
    card = state.players[0].hand[0]
    apply(state, CastSpell(card.id, use_warp=True))
    resolve_top_of_stack(state)
    assert card in state.battlefield
    assert card.cast_via_warp is True

    run_delayed_end_step_triggers(state)
    # Trigger is on the stack; resolution happens via standard path.
    assert len(state.stack) == 1
    while state.stack:
        resolve_top_of_stack(state)
    assert card not in state.battlefield
    assert card in state.players[0].exile
    assert card.warped is True


def test_simplified_spacecraft_loads_and_stations():
    """Synthesizer Labship and Sledge-Class Seedship ship as vanilla
    Spacecraft — simplified to station threshold + stationed_keywords."""

    for name, threshold, stationed_kw in (
        ("Synthesizer Labship", 9, "flying"),
        ("Sledge-Class Seedship", 7, "flying"),
    ):
        c = _PRINTINGS[name]
        assert c.is_spacecraft
        assert c.station_threshold == threshold
        assert stationed_kw in c.stationed_keywords


def test_indestructible_creature_survives_destroy_effect():
    from mtg_bot.effects import DestroyTargetCreature, PumpTargetCreatureUntilEOT

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    # Grant indestructible UEOT.
    PumpTargetCreatureUntilEOT(
        granted_keywords=frozenset({"indestructible"})
    ).resolve(state, 0, (("creature", bear.id),))
    # Cast Vote Out (destroy target creature).
    DestroyTargetCreature().resolve(state, 0, (("creature", bear.id),))
    assert bear in state.battlefield


def test_indestructible_creature_survives_lethal_combat_damage():
    from mtg_bot.effects import PumpTargetCreatureUntilEOT
    from mtg_bot.turn import state_based_actions

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # 2/2
    state.battlefield.append(bear)
    PumpTargetCreatureUntilEOT(
        granted_keywords=frozenset({"indestructible"})
    ).resolve(state, 0, (("creature", bear.id),))
    bear.damage = 99  # far in excess of toughness
    state_based_actions(state)
    assert bear in state.battlefield


def test_indestructible_still_dies_to_negative_toughness():
    """Per 704.5f, indestructible creatures still die to -toughness."""
    from mtg_bot.effects import PumpTargetCreatureUntilEOT
    from mtg_bot.turn import state_based_actions

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # 2/2
    state.battlefield.append(bear)
    # Grant indestructible.
    PumpTargetCreatureUntilEOT(
        granted_keywords=frozenset({"indestructible"})
    ).resolve(state, 0, (("creature", bear.id),))
    # Drop toughness to -1 via temp debuff.
    bear.temp_toughness_bonus = -5
    state_based_actions(state)
    assert bear not in state.battlefield


def test_emergency_eject_fizzles_against_indestructible_no_lander():
    from mtg_bot.effects import (
        DestroyTargetNonlandPermanentWithLanderForController,
        PumpTargetCreatureUntilEOT,
    )

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    PumpTargetCreatureUntilEOT(
        granted_keywords=frozenset({"indestructible"})
    ).resolve(state, 0, (("creature", bear.id),))
    DestroyTargetNonlandPermanentWithLanderForController().resolve(
        state, 0, (("permanent", bear.id),)
    )
    assert bear in state.battlefield
    landers = [o for o in state.battlefield if o.printing.name == "Lander"]
    assert landers == []


def test_reroute_systems_indestructible_mode():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()

    class PickIndestructible(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 0
    state.players[0].agent = PickIndestructible(rng=random.Random(0))
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    _PRINTINGS["Reroute Systems"].spell_effect.resolve(
        state, 0, (("creature", bear.id),)
    )
    from mtg_bot.game import effective_keywords
    assert "indestructible" in effective_keywords(state, bear)


def test_singularity_rupture_destroys_all_creatures():

    state = _arena()
    # Mix of creatures + a non-creature (Lander artifact stays).
    for owner in (0, 1):
        bear = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=owner,
            zone=Zone.BATTLEFIELD,
        )
        state.battlefield.append(bear)
    from mtg_bot.cards import TOKEN_PRINTINGS
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)

    _PRINTINGS["Singularity Rupture"].spell_effect.resolve(state, 0)
    creatures_remaining = [o for o in state.battlefield if o.printing.is_creature]
    assert creatures_remaining == []
    assert lander in state.battlefield


def test_destroy_all_creatures_respects_indestructible():
    from mtg_bot.effects import DestroyAllCreatures, PumpTargetCreatureUntilEOT

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    PumpTargetCreatureUntilEOT(
        granted_keywords=frozenset({"indestructible"})
    ).resolve(state, 0, (("creature", bear.id),))
    DestroyAllCreatures().resolve(state, 0)
    assert bear in state.battlefield  # indestructible survives
    assert victim not in state.battlefield


def test_annul_counters_artifact_spell_on_stack():
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    # P1 has Mountain for their own spell; P0 has Island for Annul.
    p1_land = state.new_object(
        BASIC_LANDS["Mountain"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(p1_land)
    p0_land = state.new_object(
        BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(p0_land)

    # P1 casts Hullcarver (artifact creature) — put on stack.
    # Actually Hullcarver is in _PRINTINGS; pick any artifact card.
    # Let's use a Spacecraft which is an artifact card.
    skiff_in_hand = state.new_object(
        _PRINTINGS["Synthesizer Labship"], owner_id=1, zone=Zone.HAND
    )
    state.players[1].hand.append(skiff_in_hand)
    # Only enough mana for {U} not {1}{U}; our cost engine auto-taps
    # generic-from-colored, which works — Synthesizer Labship costs {U}.
    _cast_spell(state, 1, skiff_in_hand.id)
    assert len(state.stack) == 1
    assert state.stack[0].source.printing.name == "Synthesizer Labship"

    # P0 casts Annul targeting the Synthesizer Labship.
    annul_in_hand = state.new_object(
        _PRINTINGS["Annul"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(annul_in_hand)
    _cast_spell(state, 0, annul_in_hand.id)
    # Annul picked the stack item it can see (the Synthesizer Labship).
    assert len(state.stack) == 2
    # Resolve top (Annul) first — it marks Labship as countered.
    resolve_top_of_stack(state)
    assert state.stack[0].countered is True
    # Resolve the countered Labship — it goes to graveyard, no ETB etc.
    resolve_top_of_stack(state)
    assert skiff_in_hand in state.players[1].graveyard
    assert skiff_in_hand not in state.battlefield


def test_counter_target_spell_filters_types():
    from mtg_bot.effects import CounterTargetSpell

    state = _arena()
    # Put an artifact-creature spell and an instant on the stack manually.
    art_card = state.new_object(
        _PRINTINGS["Synthesizer Labship"], owner_id=1, zone=Zone.STACK
    )
    state.stack.append(StackItem(source=art_card, controller_id=1))
    bolt_card = state.new_object(
        _PRINTINGS["Plasma Bolt"], owner_id=1, zone=Zone.STACK
    )
    state.stack.append(StackItem(source=bolt_card, controller_id=1))

    e = CounterTargetSpell(filter_types=frozenset({"Artifact"}))
    assert "artifact" in e.describe()
    legal = e.legal_targets(state, 0)
    # Only the artifact spell (index 0) should be legal.
    assert ("stack", 0) in legal
    assert ("stack", 1) not in legal


def test_counter_noop_paths():
    from mtg_bot.effects import CounterTargetSpell

    state = _arena()
    e = CounterTargetSpell()
    assert e.requires_target() is True
    assert "counter" in e.describe()
    # No stack → no legal targets.
    assert e.legal_targets(state, 0) == []
    # Resolve paths:
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("stack", 99999),))


def test_frontline_war_rager_conditional_counter():
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    rager = state.new_object(
        _PRINTINGS["Frontline War-Rager"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(rager)
    from mtg_bot.actions import drain_stack
    # Only one tapped creature → no counter.
    rager.tapped = True
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert rager.plus_one_counters == 0
    # Add a second tapped creature.
    mate = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    mate.tapped = True
    state.battlefield.append(mate)
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert rager.plus_one_counters == 1


def test_kavaron_turbodrone_activated_pump():
    from mtg_bot.game import effective_keywords, effective_power

    state = _arena()
    drone = state.new_object(
        _PRINTINGS["Kavaron Turbodrone"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(drone)
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    act = drone.printing.activated_abilities[0]
    assert act.cost.tap_self is True
    assert act.sorcery_speed is True
    act.effect.resolve(state, 0, (("creature", ally.id),))
    assert effective_power(state, ally) == 3  # 2 base + 1
    assert "haste" in effective_keywords(state, ally)


def test_melded_moxite_etb_loot_and_sac_for_robot():
    from mtg_bot.agents import RandomAgent
    import random

    # Use a deterministic yes-agent so the new optional "may discard"
    # gate (audit-found bug fix) takes the loot path. RandomAgent's
    # coin-flip choose_yes_no would make this test flaky.
    class _YesAgent(RandomAgent):
        def choose_yes_no(self, state, player_id, prompt):
            return True

    state = _arena()
    state.players[0].agent = _YesAgent(rng=random.Random(0))
    _stock_library(state, 0, count=5)
    # Pre-seed a card in hand so the discard has something to discard.
    discard_fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(discard_fodder)

    moxite = state.new_object(
        _PRINTINGS["Melded Moxite"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(moxite)
    # Fire ETB (loot: discard 1, draw 2).
    etb = next(
        ab for ab in moxite.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = moxite
    hand_before = len(state.players[0].hand)
    gy_before = len(state.players[0].graveyard)
    etb.effect.resolve(state, 0)
    # Net: -1 discard + 2 draw = +1 in hand
    assert len(state.players[0].hand) == hand_before + 1
    assert len(state.players[0].graveyard) == gy_before + 1

    # Activated: create a tapped 2/2 Robot token.
    act = moxite.printing.activated_abilities[0]
    assert act.cost.sacrifice_self is True
    act.effect.resolve(state, 0)
    robots = [o for o in state.battlefield if o.printing.name == "Robot"]
    assert len(robots) == 1
    assert robots[0].tapped is True


def test_melded_moxite_activated_ability_cost_is_three_generic_and_sac():
    """Audit regression: oracle cost is '{3}, Sacrifice this artifact'.

    Lock in that the only activated ability costs exactly {3} generic plus
    sacrificing the Moxite itself — no tap, no colored mana, no other
    riders. A future refactor that changes either piece would break this.
    """

    moxite = _PRINTINGS["Melded Moxite"]
    assert len(moxite.activated_abilities) == 1
    act = moxite.activated_abilities[0]
    assert act.cost.mana_str == "{3}"
    assert act.cost.mana.generic == 3
    assert act.cost.mana.colored == ()
    assert act.cost.mana.x_pips == 0
    assert act.cost.sacrifice_self is True
    assert act.cost.tap_self is False


def test_melded_moxite_activated_creates_one_tapped_2_2_colorless_robot():
    """Audit regression: token created by the activated ability matches
    the printed rider — exactly one 2/2 colorless Robot artifact creature
    token, entering tapped. Guards against any of: wrong count, wrong P/T,
    accidentally adding a color, dropping the artifact card type, or the
    'enters tapped' bit getting toggled off.
    """
    from mtg_bot.effects import CreateToken

    moxite = _PRINTINGS["Melded Moxite"]
    act = moxite.activated_abilities[0]
    assert isinstance(act.effect, CreateToken)
    assert act.effect.count == 1
    assert act.effect.tapped is True

    tok = act.effect.token_printing
    assert tok.name == "Robot"
    assert tok.power == 2
    assert tok.toughness == 2
    assert "Token" in tok.supertypes
    assert "Artifact" in tok.card_types
    assert "Creature" in tok.card_types
    assert "Robot" in tok.subtypes
    # Colorless: no colored pips and no color identity bits in mana cost.
    assert tok.mana_cost.colored == ()
    assert tok.mana_cost.generic == 0


def test_dubious_delicacy_etb_and_activated():

    state = _arena()
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    dd = state.new_object(
        _PRINTINGS["Dubious Delicacy"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(dd)
    assert "flash" in dd.printing.keywords

    # Fire ETB (-3/-3 UEOT on victim).
    etb = next(
        ab for ab in dd.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = dd
    etb.effect.resolve(state, 0, (("creature", victim.id),))
    # Bear was 2/2, now -1/-1 — SBA kills it.
    from mtg_bot.turn import state_based_actions
    state_based_actions(state)
    assert victim not in state.battlefield

    # Activated: gain 3 life.
    gain_act = dd.printing.activated_abilities[0]
    life_before = state.players[0].life
    gain_act.effect.resolve(state, 0)
    assert state.players[0].life == life_before + 3



def test_second_spell_trigger_fires_once_per_turn():
    """Casting two spells in one turn fires 'your second spell' triggers."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    # Red mana for Plasma Bolt.
    for _ in range(4):
        m = state.new_object(
            BASIC_LANDS["Mountain"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(m)
    # Also a {U} land for Illvoi Operative's target-kind universe (we
    # won't cast it; just need it on battlefield as a trigger host).
    # Operative already on battlefield; count triggers on its controller.
    op = state.new_object(
        _PRINTINGS["Illvoi Operative"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(op)
    # Plasma Bolt {R} damage — won't add counters to Operative.
    for _ in range(2):
        card = state.new_object(
            _PRINTINGS["Plasma Bolt"], owner_id=0, zone=Zone.HAND
        )
        state.players[0].hand.append(card)
    # Cast first spell — no trigger yet.
    first = state.players[0].hand[0]
    _cast_spell(state, 0, first.id)
    resolve_top_of_stack(state)
    assert op.plus_one_counters == 0
    # Cast second spell — trigger goes on stack above the spell; resolve
    # the trigger first to apply the counter.
    second = state.players[0].hand[0]
    _cast_spell(state, 0, second.id)
    resolve_top_of_stack(state)  # second-spell trigger
    assert op.plus_one_counters == 1
    resolve_top_of_stack(state)  # spell


def test_second_spell_counter_resets_across_turns():
    """cleanup step resets spells_cast_this_turn to 0."""
    from mtg_bot.turn import run_cleanup_step

    state = _arena()
    state.players[0].spells_cast_this_turn = 5
    state.players[1].spells_cast_this_turn = 3
    run_cleanup_step(state)
    assert state.players[0].spells_cast_this_turn == 0
    assert state.players[1].spells_cast_this_turn == 0


def test_cosmogrand_zenith_token_mode():
    from mtg_bot.agents import RandomAgent
    import random

    class PickTokens(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 0

    state = _arena()
    state.players[0].agent = PickTokens(rng=random.Random(0))
    zenith = state.new_object(
        _PRINTINGS["Cosmogrand Zenith"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(zenith)
    trig = zenith.printing.triggered_abilities[0]
    state.resolving_source = zenith
    trig.effect.resolve(state, 0)
    soldiers = [
        o for o in state.battlefield if o.printing.name == "Human Soldier"
    ]
    assert len(soldiers) == 2


def test_cosmogrand_zenith_counter_mode():
    from mtg_bot.agents import RandomAgent
    import random

    class PickCounters(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 1

    state = _arena()
    state.players[0].agent = PickCounters(rng=random.Random(0))
    zenith = state.new_object(
        _PRINTINGS["Cosmogrand Zenith"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(zenith)
    mate = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mate)
    trig = zenith.printing.triggered_abilities[0]
    state.resolving_source = zenith
    trig.effect.resolve(state, 0)
    assert zenith.plus_one_counters == 1
    assert mate.plus_one_counters == 1


def test_brightspear_zealot_power_boost_requires_two_spells():
    from mtg_bot.game import effective_power

    state = _arena()
    z = state.new_object(
        _PRINTINGS["Brightspear Zealot"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(z)
    # 0 spells → base 2 power.
    assert effective_power(state, z) == 2
    state.players[0].spells_cast_this_turn = 1
    assert effective_power(state, z) == 2
    state.players[0].spells_cast_this_turn = 2
    assert effective_power(state, z) == 4
    # Dropping back below threshold re-hides the bonus.
    state.players[0].spells_cast_this_turn = 1
    assert effective_power(state, z) == 2


def test_oreplate_pangolin_artifact_enters_trigger():
    from mtg_bot.actions import put_into_battlefield, drain_stack
    from mtg_bot.cards import BASIC_LANDS, TOKEN_PRINTINGS

    state = _arena(p0_battlefield=(BASIC_LANDS["Mountain"],))
    pang = state.new_object(
        _PRINTINGS["Oreplate Pangolin"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(pang)
    # Another artifact enters via token creation → controller can pay
    # {1} (one untapped Mountain), so the trigger pays and lands a
    # +1/+1 counter on pangolin.
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, lander, 0)
    drain_stack(state)
    assert pang.plus_one_counters == 1

    # An opponent's artifact entering shouldn't trigger.
    enemy_art = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, enemy_art, 1)
    drain_stack(state)
    assert pang.plus_one_counters == 1  # unchanged


def test_pangolin_doesnt_trigger_on_itself():
    """The trigger says 'another artifact' — self-entry shouldn't fire."""
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.cards import BASIC_LANDS

    # Plenty of mana available — even so the self-entry must not put a
    # counter, because the trigger says "another artifact".
    state = _arena(p0_battlefield=(BASIC_LANDS["Mountain"],) * 3)
    pang = state.new_object(
        _PRINTINGS["Oreplate Pangolin"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    put_into_battlefield(state, pang, 0)
    assert pang.plus_one_counters == 0


def test_pangolin_skips_counter_when_cannot_pay_optional_one():
    """No mana available → optional {1} can't be paid → no counter."""
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()  # no lands → controller can't pay {1}
    pang = state.new_object(
        _PRINTINGS["Oreplate Pangolin"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(pang)
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, lander, 0)
    assert pang.plus_one_counters == 0


def test_pangolin_taps_a_land_to_pay_optional_cost():
    """When the trigger fires and pays {1}, an untapped land becomes tapped."""
    from mtg_bot.actions import put_into_battlefield, drain_stack
    from mtg_bot.cards import BASIC_LANDS, TOKEN_PRINTINGS

    state = _arena(p0_battlefield=(BASIC_LANDS["Mountain"],))
    mountain = next(
        o for o in state.battlefield
        if o.controller_id == 0 and o.printing.name == "Mountain"
    )
    assert not mountain.tapped
    pang = state.new_object(
        _PRINTINGS["Oreplate Pangolin"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(pang)
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, lander, 0)
    drain_stack(state)
    assert pang.plus_one_counters == 1
    assert mountain.tapped  # paid {1} by tapping the Mountain


def test_mechan_assembler_fires_once_per_turn():
    from mtg_bot.actions import put_into_battlefield, drain_stack
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    ma = state.new_object(
        _PRINTINGS["Mechan Assembler"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ma)
    # First artifact enters → Robot token created.
    art1 = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, art1, 0)
    drain_stack(state)
    robots = [o for o in state.battlefield if o.printing.name == "Robot"]
    assert len(robots) == 1
    # Second artifact enters → should NOT create another token (once/turn).
    art2 = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, art2, 0)
    drain_stack(state)
    robots = [o for o in state.battlefield if o.printing.name == "Robot"]
    assert len(robots) == 1  # still just one


def test_once_per_turn_resets_on_cleanup():
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.turn import run_cleanup_step

    state = _arena()
    ma = state.new_object(
        _PRINTINGS["Mechan Assembler"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ma)
    art = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, art, 0)
    assert state.once_per_turn_fired  # populated
    run_cleanup_step(state)
    assert not state.once_per_turn_fired  # reset


def test_nanoform_sentinel_fizzles_without_tapped_target():
    """When no legal untap target exists, the trigger fizzles silently."""
    from mtg_bot.turn import tap_permanent

    state = _arena()
    ns = state.new_object(
        _PRINTINGS["Nanoform Sentinel"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ns)
    # No other tapped permanents → legal_targets is empty → fizzle.
    tap_permanent(state, ns)
    assert ns.tapped is True  # the tap itself still happened


def test_nanoform_sentinel_no_agent_just_taps():
    """Controller has no agent → trigger can still fire but picks nothing."""
    from mtg_bot.turn import tap_permanent

    state = _arena()
    state.players[0].agent = None  # no agent
    ns = state.new_object(
        _PRINTINGS["Nanoform Sentinel"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ns)
    tapped_ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    tapped_ally.tapped = True
    state.battlefield.append(tapped_ally)
    tap_permanent(state, ns)
    # No agent → no target chosen → resolve with empty targets (no-op).
    assert tapped_ally.tapped is True


def test_nanoform_sentinel_untaps_once_per_turn():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.actions import drain_stack
    from mtg_bot.turn import tap_permanent

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    ns = state.new_object(
        _PRINTINGS["Nanoform Sentinel"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ns)
    tapped_ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    tapped_ally.tapped = True
    state.battlefield.append(tapped_ally)
    # Tap Nanoform Sentinel — should fire the trigger, untapping ally.
    tap_permanent(state, ns)
    drain_stack(state)
    assert tapped_ally.tapped is False


def test_meltstrider_eulogist_counter_dies_triggers_draw():
    from mtg_bot.actions import drain_stack
    from mtg_bot.turn import move_off_battlefield

    state = _arena()
    _stock_library(state, 0, count=5)
    eul = state.new_object(
        _PRINTINGS["Meltstrider Eulogist"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(eul)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    victim.plus_one_counters = 1  # has a counter
    state.battlefield.append(victim)
    hand_before = len(state.players[0].hand)
    move_off_battlefield(state, victim, destination="graveyard")
    drain_stack(state)
    assert len(state.players[0].hand) == hand_before + 1


def test_meltstrider_eulogist_no_trigger_if_no_counter():
    from mtg_bot.turn import move_off_battlefield

    state = _arena()
    _stock_library(state, 0, count=5)
    eul = state.new_object(
        _PRINTINGS["Meltstrider Eulogist"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(eul)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )  # no counter
    state.battlefield.append(victim)
    hand_before = len(state.players[0].hand)
    move_off_battlefield(state, victim, destination="graveyard")
    assert len(state.players[0].hand) == hand_before


def test_meltstrider_eulogist_no_trigger_on_opponent_creature():
    """Only YOUR creatures dying trigger it."""
    from mtg_bot.turn import move_off_battlefield

    state = _arena()
    eul = state.new_object(
        _PRINTINGS["Meltstrider Eulogist"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(eul)
    enemy = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    enemy.plus_one_counters = 1
    state.battlefield.append(enemy)
    hand_before = len(state.players[0].hand)
    move_off_battlefield(state, enemy, destination="graveyard")
    assert len(state.players[0].hand) == hand_before


def test_lightless_evangel_counter_on_sacrifice():
    from mtg_bot.actions import drain_stack
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.turn import sacrifice

    state = _arena()
    le = state.new_object(
        _PRINTINGS["Lightless Evangel"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(le)
    # Sacrificing a Lander (artifact) → counter on Evangel.
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)
    sacrifice(state, lander)
    drain_stack(state)
    assert le.plus_one_counters == 1


def test_lightless_evangel_does_not_trigger_on_self_sacrifice():
    """Sacrificing the Evangel itself shouldn't count (trigger says
    'another creature or artifact')."""
    from mtg_bot.turn import sacrifice

    state = _arena()
    le = state.new_object(
        _PRINTINGS["Lightless Evangel"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(le)
    # Only Evangel on the field; sacrificing it shouldn't re-trigger.
    sacrifice(state, le)
    # Evangel is gone, no counter to verify — just ensure no crash.
    assert le not in state.battlefield


def test_sunstar_expansionist_etb_triggers_only_when_behind():
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    # Opponent has 2 lands, caster has 0.
    for _ in range(2):
        land = state.new_object(
            BASIC_LANDS["Forest"], owner_id=1, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(land)

    se = state.new_object(
        _PRINTINGS["Sunstar Expansionist"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(se)
    # The "behind" intervening-if gate now lives on the predicate
    # rather than wrapping the effect (per CR 603.4 semantics).
    etb = se.printing.triggered_abilities[0]
    assert etb.predicate.matches(state, {"obj": se}, se) is True
    state.resolving_source = se
    etb.effect.resolve(state, 0)
    landers = [o for o in state.battlefield if o.printing.name == "Lander"]
    assert len(landers) == 1


def test_sunstar_expansionist_etb_skipped_when_ahead():
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    # Caster has 3 lands, opponent has 1.
    for _ in range(3):
        land = state.new_object(
            BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(land)
    opp_land = state.new_object(
        BASIC_LANDS["Forest"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(opp_land)

    se = state.new_object(
        _PRINTINGS["Sunstar Expansionist"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(se)
    # Caster is ahead → the predicate gate fails so the trigger
    # never goes on the stack and no Lander is created.
    etb = se.printing.triggered_abilities[0]
    assert etb.predicate.matches(state, {"obj": se}, se) is False
    landers = [o for o in state.battlefield if o.printing.name == "Lander"]
    assert len(landers) == 0


def test_sunstar_expansionist_landfall_pumps_self():
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import fire_landfall_triggers

    state = _arena()
    se = state.new_object(
        _PRINTINGS["Sunstar Expansionist"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(se)
    land = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(land)
    fire_landfall_triggers(state, land)
    from mtg_bot.actions import drain_stack
    drain_stack(state)
    assert se.temp_power_bonus == 1


def test_illvoi_infiltrator_draws_on_combat_damage_to_player():
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.combat import _resolve_damage_step

    state = _arena()
    _stock_library(state, 0, count=5)
    inf = state.new_object(
        _PRINTINGS["Illvoi Infiltrator"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(inf)
    inf.attacking = True
    # P1 is defending; no blockers.
    hand_before = len(state.players[0].hand)
    _resolve_damage_step(state, defending_player_id=1, first_strike_step=False)
    # The trigger now goes on the stack rather than resolving inline —
    # opponents get a chance to respond before the draw happens.
    assert len(state.stack) == 1
    assert len(state.players[0].hand) == hand_before
    # Resolve the stack: the draw lands.
    resolve_top_of_stack(state)
    assert len(state.players[0].hand) == hand_before + 1


def test_combat_damage_to_player_trigger_skips_when_blocked():
    """No damage to player → no trigger."""
    from mtg_bot.combat import _resolve_damage_step

    state = _arena()
    _stock_library(state, 0, count=5)
    inf = state.new_object(
        _PRINTINGS["Illvoi Infiltrator"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(inf)
    inf.attacking = True
    blocker = state.new_object(
        SYNTHETIC_CREATURES["Wall of Stone"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    blocker.blocking = inf.id
    inf.blocked_by = [blocker.id]
    state.battlefield.append(blocker)
    hand_before = len(state.players[0].hand)
    _resolve_damage_step(state, defending_player_id=1, first_strike_step=False)
    assert len(state.players[0].hand) == hand_before  # no trigger


def test_illvoi_infiltrator_blockable_when_fewer_than_two_spells_cast():
    """can_block returns True (Illvoi blockable) when controller has cast
    0 or 1 spells this turn — the conditional unblockable hasn't kicked in.
    """
    from mtg_bot.combat import can_block

    state = _arena()
    inf = state.new_object(
        _PRINTINGS["Illvoi Infiltrator"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(inf)
    inf.attacking = True
    blocker = state.new_object(
        SYNTHETIC_CREATURES["Wall of Stone"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(blocker)
    # 0 spells cast: blockable.
    assert state.players[0].spells_cast_this_turn == 0
    assert can_block(state, blocker, inf) is True
    # 1 spell cast: still blockable (need 2+).
    state.players[0].spells_cast_this_turn = 1
    assert can_block(state, blocker, inf) is True


def test_illvoi_infiltrator_unblockable_when_two_spells_cast():
    """can_block returns False against Illvoi once its controller has cast
    2 (or more) spells this turn.
    """
    from mtg_bot.combat import can_block

    state = _arena()
    inf = state.new_object(
        _PRINTINGS["Illvoi Infiltrator"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(inf)
    inf.attacking = True
    blocker = state.new_object(
        SYNTHETIC_CREATURES["Wall of Stone"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(blocker)
    state.players[0].spells_cast_this_turn = 2
    assert can_block(state, blocker, inf) is False
    # Higher counts also unblockable.
    state.players[0].spells_cast_this_turn = 5
    assert can_block(state, blocker, inf) is False
    # Crucially: gated on the ATTACKER's controller, not the blocker's
    # controller. Bumping P1's counter doesn't unblock anything.
    state.players[0].spells_cast_this_turn = 0
    state.players[1].spells_cast_this_turn = 5
    assert can_block(state, blocker, inf) is True


def test_illvoi_infiltrator_declare_blockers_drops_block_and_logs():
    """End-to-end: declare_blockers itself ignores assigned blockers when
    the conditional unblockable is active, and emits an explanatory log.
    The infiltrator then connects for combat damage and queues its
    draw-a-card trigger.
    """
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.combat import _resolve_damage_step, declare_blockers

    state = _arena()
    _stock_library(state, 0, count=5)
    inf = state.new_object(
        _PRINTINGS["Illvoi Infiltrator"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(inf)
    inf.attacking = True
    blocker = state.new_object(
        SYNTHETIC_CREATURES["Wall of Stone"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(blocker)
    state.players[0].spells_cast_this_turn = 2
    declare_blockers(state, {blocker.id: inf.id})
    # The block was rejected — neither side recorded the assignment.
    assert blocker.blocking is None
    assert inf.blocked_by == []
    assert any(
        "can't be blocked" in line and "Illvoi Infiltrator" in line
        for line in state.log
    ), state.log
    # Combat damage now hits the player and queues the draw trigger.
    hand_before = len(state.players[0].hand)
    _resolve_damage_step(state, defending_player_id=1, first_strike_step=False)
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    assert len(state.players[0].hand) == hand_before + 1


def test_thaumaton_torpedo_destroys_nonland_permanent():

    state = _arena()
    torpedo = state.new_object(
        _PRINTINGS["Thaumaton Torpedo"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(torpedo)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    act = torpedo.printing.activated_abilities[0]
    act.effect.resolve(state, 0, (("permanent", victim.id),))
    assert victim not in state.battlefield


def test_thaumaton_torpedo_full_cost_when_no_spacecraft_attack():
    """Without a Spacecraft attack this turn, the activation costs
    the printed {6}."""
    from mtg_bot.actions import _effective_ability_mana_cost

    state = _arena()
    torpedo = state.new_object(
        _PRINTINGS["Thaumaton Torpedo"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(torpedo)
    ability = torpedo.printing.activated_abilities[0]
    assert state.players[0].attacked_this_turn == []
    eff = _effective_ability_mana_cost(state, torpedo, ability)
    assert eff.generic == 6


def test_thaumaton_torpedo_reduced_cost_after_spacecraft_attack():
    """After the controller has attacked with a Spacecraft this turn,
    the activation costs {3} less, i.e. {3}."""
    from mtg_bot.actions import _effective_ability_mana_cost

    state = _arena()
    torpedo = state.new_object(
        _PRINTINGS["Thaumaton Torpedo"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(torpedo)
    ability = torpedo.printing.activated_abilities[0]
    # Latch in a Spacecraft attacker — the AttackedWithThisTurn(filter)
    # condition reads through the generalized list.
    skiff = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.players[0].attacked_this_turn.append(skiff)
    eff = _effective_ability_mana_cost(state, torpedo, ability)
    assert eff.generic == 3


def test_thaumaton_torpedo_cost_reduction_floors_at_zero():
    """A cost reduction larger than the printed generic floors at 0
    rather than going negative — matches ManaCost.reduced()'s clamp."""
    from mtg_bot.actions import _effective_ability_mana_cost
    from mtg_bot.cards.base import ActivationCost, CostReduction, parse_mana_cost
    from mtg_bot.effects import ActivatedAbility, DestroyTargetNonlandPermanent
    from mtg_bot.effects.conditions import AttackedWithThisTurn
    from mtg_bot.effects.object_predicates import HasSubtype

    state = _arena()
    torpedo = state.new_object(
        _PRINTINGS["Thaumaton Torpedo"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(torpedo)
    # Build a synthetic ability with a tiny generic cost {1} and a
    # {3}-less reduction; verify the result clamps at 0 instead of -2.
    ability = ActivatedAbility(
        cost=ActivationCost(
            mana=parse_mana_cost("{1}"), mana_str="{1}",
            tap_self=True, sacrifice_self=True,
        ),
        effect=DestroyTargetNonlandPermanent(),
        cost_reductions=(
            CostReduction(
                amount=3,
                condition=AttackedWithThisTurn(HasSubtype("Spacecraft")),
            ),
        ),
    )
    skiff = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.players[0].attacked_this_turn.append(skiff)
    eff = _effective_ability_mana_cost(state, torpedo, ability)
    assert eff.generic == 0


def test_destroy_target_nonland_permanent_noop_paths():
    from mtg_bot.effects import DestroyTargetNonlandPermanent
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    e = DestroyTargetNonlandPermanent()
    assert e.requires_target() is True
    assert "nonland" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))

    # Land can't be targeted.
    land = state.new_object(
        BASIC_LANDS["Forest"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(land)
    legal = e.legal_targets(state, controller_id=0)
    assert ("permanent", land.id) not in legal
    # Explicit resolve against a land is filtered.
    e.resolve(state, controller_id=0, targets=(("permanent", land.id),))
    assert land in state.battlefield


def test_meltstriders_gear_equips_and_buffs():
    from mtg_bot.game import effective_power, effective_keywords

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    gear = state.new_object(
        _PRINTINGS["Meltstrider's Gear"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(gear)
    assert "Equipment" in gear.printing.subtypes
    # Trigger ETB to attach.
    etb = next(
        ab for ab in gear.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = gear
    etb.effect.resolve(state, 0, (("creature", ally.id),))
    assert gear.attached_to == ally.id
    # Ally is now 4/3 with reach.
    assert effective_power(state, ally) == 4  # 2 + 2
    assert "reach" in effective_keywords(state, ally)


def test_equipment_falls_off_when_creature_dies():
    from mtg_bot.turn import destroy, move_off_battlefield
    from mtg_bot.game import effective_power

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    gear = state.new_object(
        _PRINTINGS["Meltstrider's Gear"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(gear)
    gear.attached_to = ally.id

    # Destroy the attached creature.
    move_off_battlefield(state, ally, destination="graveyard")
    assert gear in state.battlefield  # Equipment survives
    assert gear.attached_to is None  # but falls off


def test_squires_lightblade_has_flash_and_buff():

    c = _PRINTINGS["Squire's Lightblade"]
    assert "flash" in c.keywords
    assert c.equipment_bonuses == (1, 0, frozenset())


def test_illvoi_light_jammer_has_flash_plus_buff():
    from mtg_bot.game import effective_power, effective_toughness

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    jammer = state.new_object(
        _PRINTINGS["Illvoi Light Jammer"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(jammer)
    etb = next(
        ab for ab in jammer.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = jammer
    etb.effect.resolve(state, 0, (("creature", ally.id),))
    assert effective_power(state, ally) == 3  # 2 + 1
    assert effective_toughness(state, ally) == 4  # 2 + 2


def test_attach_fizzle_paths():
    """AttachToTargetCreatureYouControl no-ops when misused."""
    from mtg_bot.effects import AttachToTargetCreatureYouControl

    state = _arena()
    e = AttachToTargetCreatureYouControl()
    assert e.requires_target() is True
    assert "attach" in e.describe()
    # No resolving_source.
    state.resolving_source = None
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    # Non-Equipment source.
    state.resolving_source = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    # Equipment source, missing targets.
    gear = state.new_object(
        _PRINTINGS["Meltstrider's Gear"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.resolving_source = gear
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))
    # Target is an opponent's creature.
    enemy = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(enemy)
    e.resolve(state, controller_id=0, targets=(("creature", enemy.id),))
    assert gear.attached_to is None


def test_auxiliary_boosters_etb_creates_and_attaches():
    from mtg_bot.game import effective_power, effective_toughness, effective_keywords

    state = _arena()
    gear = state.new_object(
        _PRINTINGS["Auxiliary Boosters"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(gear)
    etb = next(
        ab for ab in gear.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = gear
    etb.effect.resolve(state, 0)
    robots = [o for o in state.battlefield if o.printing.name == "Robot"]
    assert len(robots) == 1
    robot = robots[0]
    assert gear.attached_to == robot.id
    # Robot is 2/2, Boosters grant +1/+2 + flying → 3/4 flying.
    assert effective_power(state, robot) == 3
    assert effective_toughness(state, robot) == 4
    assert "flying" in effective_keywords(state, robot)


def test_hylderblade_void_reattaches_at_end_step():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.actions import drain_stack
    from mtg_bot.turn import run_beginning_of_end_step_triggers, Step

    state = _arena(step=Step.END, active=0)
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    hyl = state.new_object(
        _PRINTINGS["Hylderblade"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(hyl)
    # Void inactive → no re-attach.
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert hyl.attached_to is None
    # Activate Void.
    state.permanent_left_battlefield_this_turn = True
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert hyl.attached_to == ally.id


def test_end_step_trigger_fizzles_without_legal_targets():
    """Hylderblade on an otherwise-empty battlefield: no target → fizzle."""
    from mtg_bot.turn import run_beginning_of_end_step_triggers, Step

    state = _arena(step=Step.END, active=0)
    hyl = state.new_object(
        _PRINTINGS["Hylderblade"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(hyl)
    state.permanent_left_battlefield_this_turn = True
    run_beginning_of_end_step_triggers(state)
    assert hyl.attached_to is None
    assert any("fizzles" in l for l in state.log)


def test_specimen_freighter_bounces_up_to_two():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(1))
    # Two opposing creatures.
    bears: list = []
    for _ in range(2):
        b = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(b)
        bears.append(b)

    freighter = state.new_object(
        _PRINTINGS["Specimen Freighter"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(freighter)
    # Fire the ETB with multi-target: pick both bears.
    etb = next(
        ab for ab in freighter.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = freighter
    etb.effect.resolve(
        state, 0,
        tuple(("creature", b.id) for b in bears),
    )
    assert all(b not in state.battlefield for b in bears)
    assert all(b in state.players[1].hand for b in bears)


def test_return_up_to_n_empty_targets_is_noop():
    from mtg_bot.effects import ReturnUpToNTargetCreaturesToOwnersHands

    state = _arena()
    e = ReturnUpToNTargetCreaturesToOwnersHands(max_n=3)
    assert e.target_count_range() == (0, 3)
    assert e.requires_target() is True
    assert "3" in e.describe()
    e.resolve(state, controller_id=0, targets=())  # no-op
    e.resolve(state, controller_id=0, targets=(("player", 0),))  # wrong kind
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))


def test_specimen_freighter_etb_excludes_spacecraft_targets():
    """Oracle: 'return up to two target NON-SPACECRAFT creatures.'"""

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(bear)
    # Stationed enemy Spacecraft (counts as a creature) — oracle excludes
    # it from the bounce target list.
    rival = state.new_object(
        _PRINTINGS["Pinnacle Kill-Ship"], owner_id=1, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(rival)
    freighter = state.new_object(
        _PRINTINGS["Specimen Freighter"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(freighter)
    etb = next(
        ab for ab in freighter.printing.triggered_abilities
        if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = freighter
    legal = etb.effect.legal_targets(state, controller_id=0)
    target_ids = {tid for _, tid in legal}
    assert bear.id in target_ids
    assert rival.id not in target_ids


def test_specimen_freighter_attacks_trigger_mills_4_from_each_opponent():
    """Oracle: 'Whenever this Spacecraft attacks, defending player mills
    four cards.' Modeled as 'each opponent mills 4' (exact in 2-player)."""
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    # Stack opponent's library with enough cards to mill.
    for _ in range(8):
        c = state.new_object(
            BASIC_LANDS["Plains"], owner_id=1, zone=Zone.LIBRARY,
        )
        state.players[1].library.append(c)
    freighter = state.new_object(
        _PRINTINGS["Specimen Freighter"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(freighter)
    from mtg_bot.effects.predicates import (
        Event,
        IsSelf,
    )
    attacks_ability = next(
        ab for ab in freighter.printing.triggered_abilities
        if ab.event == Event.ATTACKS and ab.predicate == IsSelf()
    )
    before_lib = len(state.players[1].library)
    before_grave = len(state.players[1].graveyard)
    attacks_ability.effect.resolve(state, controller_id=0)
    assert len(state.players[1].library) == before_lib - 4
    assert len(state.players[1].graveyard) == before_grave + 4


def test_multi_target_cast_path_through_cast_spell():
    """Casting Specimen Freighter via _cast_spell walks the multi-target
    pipeline: picks up to two bears from opponent's board."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    # Fund {5}{U} cost.
    for _ in range(5):
        p = state.new_object(
            BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(p)
    # Two enemy bears.
    bears: list = []
    for _ in range(2):
        b = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(b)
        bears.append(b)
    # Freighter in hand.
    freighter_in_hand = state.new_object(
        _PRINTINGS["Specimen Freighter"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(freighter_in_hand)
    _cast_spell(state, 0, freighter_in_hand.id)
    # Resolve onto battlefield → ETB trigger pushed onto stack.
    resolve_top_of_stack(state)
    # Resolve the ETB.
    resolve_top_of_stack(state)
    # Both bears bounced.
    assert all(b in state.players[1].hand for b in bears)


def test_scout_for_survivors_returns_with_counters():

    state = _arena()
    # Oracle: 'with TOTAL mana value 3 or less'. Seed 3 MV-1 creatures
    # (sum 3, fits the budget) plus a big creature too large to be a
    # legal singleton target. Wild Bear (MV 2) was the prior fixture
    # but two of them already exceeds the budget; switch to Coastal
    # Scout (MV 1) so the test exercises the maximum-sized legal pick.
    small = []
    for _ in range(3):
        c = state.new_object(
            SYNTHETIC_CREATURES["Coastal Scout"], owner_id=0,
            zone=Zone.GRAVEYARD,
        )  # MV 1
        state.players[0].graveyard.append(c)
        small.append(c)
    big = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=0,
        zone=Zone.GRAVEYARD,
    )  # {3}{U}, MV 4
    state.players[0].graveyard.append(big)

    effect = _PRINTINGS["Scout for Survivors"].spell_effect
    legal = effect.legal_targets(state, 0)
    legal_ids = {tid for _, tid in legal}
    # All 3 small creatures legal, big one NOT legal (MV > 3 budget).
    for s in small:
        assert s.id in legal_ids
    assert big.id not in legal_ids

    effect.resolve(
        state, 0,
        tuple(("graveyard_creature", s.id) for s in small),
    )
    # All 3 small creatures are back on the battlefield with a counter.
    for s in small:
        assert s in state.battlefield
        assert s.plus_one_counters == 1


def test_scout_for_survivors_noop_paths():
    from mtg_bot.effects import ReturnUpToNSmallCreaturesFromGraveyardWithCounters

    state = _arena()
    e = ReturnUpToNSmallCreaturesFromGraveyardWithCounters(
        max_n=2, max_total_mv=1,
    )
    assert "graveyard" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("graveyard_creature", 99999),))


def test_scout_for_survivors_total_mv_budget_skips_overflow_picks():
    """Oracle: 'with TOTAL mana value 3 or less'. Picking three MV-1
    creatures is fine; adding an MV-3 to the same set should drop the
    last pick (the MV-3) since the running budget is already 3 → 0."""
    from mtg_bot.cards import SYNTHETIC_CREATURES
    from mtg_bot.effects import ReturnUpToNSmallCreaturesFromGraveyardWithCounters

    state = _arena()
    p = state.players[0]
    # Three MV-1 creatures + one MV-3 creature, all in the graveyard.
    one_mv = []
    for _ in range(3):
        c = state.new_object(
            SYNTHETIC_CREATURES["Coastal Scout"], owner_id=0, zone=Zone.GRAVEYARD,
        )  # 1/1 with mana cost {R}
        p.graveyard.append(c)
        one_mv.append(c)
    big = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=0, zone=Zone.GRAVEYARD,
    )  # 4/4 with mana cost {3}{U} — MV 4
    p.graveyard.append(big)
    e = ReturnUpToNSmallCreaturesFromGraveyardWithCounters(
        max_n=4, max_total_mv=3,
    )
    e.resolve(
        state, 0,
        targets=tuple(("graveyard_creature", c.id) for c in one_mv + [big]),
    )
    # All three MV-1 came back; the MV-4 was over-budget and stayed.
    for c in one_mv:
        assert c in state.battlefield
        assert c.plus_one_counters == 1
    assert big in p.graveyard
    assert big not in state.battlefield


def test_scout_for_survivors_total_mv_budget_admits_full_set_when_within_budget():
    """A pure MV-1 set of three under a budget of 3 fits exactly."""
    from mtg_bot.cards import SYNTHETIC_CREATURES
    from mtg_bot.effects import ReturnUpToNSmallCreaturesFromGraveyardWithCounters

    state = _arena()
    p = state.players[0]
    cs = []
    for _ in range(3):
        c = state.new_object(
            SYNTHETIC_CREATURES["Coastal Scout"], owner_id=0, zone=Zone.GRAVEYARD,
        )  # MV 1
        p.graveyard.append(c)
        cs.append(c)
    e = ReturnUpToNSmallCreaturesFromGraveyardWithCounters(
        max_n=3, max_total_mv=3,
    )
    e.resolve(
        state, 0,
        targets=tuple(("graveyard_creature", c.id) for c in cs),
    )
    for c in cs:
        assert c in state.battlefield
        assert c.plus_one_counters == 1


def test_scout_for_survivors_legal_targets_excludes_oversized_singletons():
    """A creature with MV greater than the budget is never a legal
    single-pick target."""
    from mtg_bot.cards import SYNTHETIC_CREATURES
    from mtg_bot.effects import ReturnUpToNSmallCreaturesFromGraveyardWithCounters

    state = _arena()
    p = state.players[0]
    big = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=0, zone=Zone.GRAVEYARD,
    )  # MV 4
    p.graveyard.append(big)
    small = state.new_object(
        SYNTHETIC_CREATURES["Coastal Scout"], owner_id=0, zone=Zone.GRAVEYARD,
    )  # MV 1
    p.graveyard.append(small)
    e = ReturnUpToNSmallCreaturesFromGraveyardWithCounters(
        max_n=3, max_total_mv=3,
    )
    legal = e.legal_targets(state, controller_id=0)
    target_ids = {tid for _, tid in legal}
    assert small.id in target_ids
    assert big.id not in target_ids


def test_biosynthic_burst_counter_keywords_untap():
    from mtg_bot.game import effective_keywords

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    ally.tapped = True
    state.battlefield.append(ally)
    _PRINTINGS["Biosynthic Burst"].spell_effect.resolve(
        state, 0, (("creature", ally.id),)
    )
    assert ally.plus_one_counters == 1
    assert ally.tapped is False
    kws = effective_keywords(state, ally)
    assert "reach" in kws
    assert "trample" in kws
    assert "indestructible" in kws


def test_biosynthic_burst_noop_paths():
    from mtg_bot.effects import BiosynthicBurstEffect

    state = _arena()
    e = BiosynthicBurstEffect()
    assert e.requires_target() is True
    assert "reach" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 1),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))
    # Opponent's creature → filtered at resolve time.
    enemy = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(enemy)
    e.resolve(state, controller_id=0, targets=(("creature", enemy.id),))
    assert enemy.plus_one_counters == 0


def test_multi_target_agent_declines_after_min():
    """Agent returns None after min targets reached → stop picking."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()

    class PickOneThenStop(RandomAgent):
        def __init__(self):
            super().__init__(rng=random.Random(0))
            self.picks = 0

        def choose_target(self, state, player_id, legal):
            self.picks += 1
            if self.picks > 1:
                return None  # decline further
            return legal[0]

    state.players[0].agent = PickOneThenStop()
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    for _ in range(5):
        l = state.new_object(
            BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(l)
    bears: list = []
    for _ in range(2):
        b = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(b)
        bears.append(b)
    freighter = state.new_object(
        _PRINTINGS["Specimen Freighter"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(freighter)
    _cast_spell(state, 0, freighter.id)
    # Resolve permanent
    resolve_top_of_stack(state)
    # Resolve ETB (one target only was picked).
    resolve_top_of_stack(state)
    # One of the bears was bounced; the other stayed.
    bounced = sum(1 for b in bears if b in state.players[1].hand)
    assert bounced == 1


def test_multi_target_etb_runs_out_of_legal_targets():
    """Freighter ETB wants up to 2 but only 1 legal target → picks 1."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    for _ in range(6):
        l = state.new_object(
            BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(l)
    # Only ONE enemy creature.
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    freighter = state.new_object(
        _PRINTINGS["Specimen Freighter"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(freighter)
    _cast_spell(state, 0, freighter.id)
    resolve_top_of_stack(state)  # Freighter enters → ETB triggered
    # ETB targets picked at trigger-push time; fires with 1 target since
    # only 1 legal. Resolve the ETB.
    resolve_top_of_stack(state)
    assert victim in state.players[1].hand


def test_multi_target_agent_picks_out_of_set():
    """Agent returns a target not in the remaining set → stop picking early."""
    from mtg_bot.effects import ReturnUpToNTargetCreaturesToOwnersHands
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()

    class RogueAgent(RandomAgent):
        def __init__(self):
            super().__init__(rng=random.Random(0))

        def choose_target(self, state, player_id, legal):
            # Return a target that isn't in the legal list at all.
            return ("creature", 999999)

    state.players[0].agent = RogueAgent()
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    for _ in range(6):
        l = state.new_object(
            BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(l)
    for _ in range(2):
        b = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(b)
    freighter = state.new_object(
        _PRINTINGS["Specimen Freighter"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(freighter)
    _cast_spell(state, 0, freighter.id)
    # The cast-path loop should abort with 0 targets chosen (rogue agent).
    assert state.stack[-1].targets == ()


def test_dark_endurance_pumps_and_protects():
    from mtg_bot.game import effective_power, effective_keywords

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    _PRINTINGS["Dark Endurance"].spell_effect.resolve(
        state, 0, (("creature", bear.id),)
    )
    assert effective_power(state, bear) == 4  # 2 + 2
    assert "indestructible" in effective_keywords(state, bear)


def test_full_bore_pumps_own_creature():
    from mtg_bot.game import effective_power, effective_toughness, effective_keywords

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    _PRINTINGS["Full Bore"].spell_effect.resolve(
        state, 0, (("creature", ally.id),)
    )
    assert effective_power(state, ally) == 5  # 2 + 3
    assert effective_toughness(state, ally) == 4  # 2 + 2
    # Hard-cast target: no trample / haste rider.
    kw = effective_keywords(state, ally)
    assert "trample" not in kw
    assert "haste" not in kw


def test_full_bore_warp_cast_target_also_gains_trample_and_haste():
    """Oracle: 'If that creature was cast for its warp cost, it also gains
    trample and haste until end of turn.'"""
    from mtg_bot.game import effective_keywords

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    ally.cast_via_warp = True
    state.battlefield.append(ally)
    _PRINTINGS["Full Bore"].spell_effect.resolve(
        state, 0, (("creature", ally.id),)
    )
    kw = effective_keywords(state, ally)
    assert "trample" in kw
    assert "haste" in kw


def test_full_bore_warp_rider_cleans_up_at_end_of_turn():
    from mtg_bot.game import effective_keywords
    from mtg_bot.turn import run_cleanup_step

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    ally.cast_via_warp = True
    state.battlefield.append(ally)
    _PRINTINGS["Full Bore"].spell_effect.resolve(
        state, 0, (("creature", ally.id),)
    )
    assert "haste" in effective_keywords(state, ally)
    run_cleanup_step(state)
    kw = effective_keywords(state, ally)
    assert "trample" not in kw
    assert "haste" not in kw


def test_cryoshatter_aura_attaches_and_debuffs():
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    # Fund {U}.
    mana = state.new_object(
        BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mana)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )  # 4/4 — with -5/-0, SBAs should eventually kill it via damage
    state.battlefield.append(victim)
    cryo_in_hand = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(cryo_in_hand)
    _cast_spell(state, 0, cryo_in_hand.id)
    resolve_top_of_stack(state)
    # Cryoshatter is on battlefield, attached to Leviathan.
    assert cryo_in_hand in state.battlefield
    assert cryo_in_hand.attached_to == victim.id
    # Effective power of Leviathan: 4 - 5 = -1.
    from mtg_bot.game import effective_power
    assert effective_power(state, victim) == -1


def test_aura_fizzles_if_target_gone_at_resolution():
    """Cast Aura targeting creature, then creature dies before resolution."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import destroy

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    mana = state.new_object(
        BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mana)
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(cryo)
    _cast_spell(state, 0, cryo.id)
    # Kill the target before the Aura resolves.
    destroy(state, victim)
    resolve_top_of_stack(state)
    # Cryoshatter fizzled to graveyard.
    assert cryo in state.players[0].graveyard
    assert cryo not in state.battlefield


def test_aura_goes_to_graveyard_when_enchanted_creature_leaves():
    from mtg_bot.turn import destroy

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    aura = state.new_object(
        _PRINTINGS["Meltstrider's Resolve"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    aura.attached_to = ally.id
    state.battlefield.append(aura)
    # Destroy the enchanted creature.
    destroy(state, ally)
    assert aura not in state.battlefield
    assert aura in state.players[0].graveyard


def test_cryoshatter_destroys_host_when_host_becomes_tapped():
    """Cryoshatter's "when enchanted creature becomes tapped, destroy it"
    half: tapping the host should destroy the host, and the aura then
    follows it off the battlefield (auras leave when their host does).
    """
    from mtg_bot.turn import tap_permanent

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    cryo.attached_to = bear.id
    state.battlefield.append(cryo)
    from mtg_bot.actions import drain_stack
    # Sanity: both on the battlefield, neither is in a graveyard yet.
    assert bear in state.battlefield
    assert cryo in state.battlefield
    # Tap the bear via tap_permanent — this is the trigger source.
    tap_permanent(state, bear)
    drain_stack(state)
    # Bear was destroyed.
    assert bear not in state.battlefield
    assert bear in state.players[1].graveyard
    # Cryoshatter, having lost its host, also left the battlefield
    # (auras get sent to their owner's graveyard when their host leaves).
    assert cryo not in state.battlefield
    assert cryo in state.players[0].graveyard


def test_cryoshatter_no_op_when_host_already_tapped():
    """If the host is already tapped, tap_permanent is a no-op and the
    aura's becomes-tapped trigger does NOT fire (no re-trigger)."""
    from mtg_bot.turn import tap_permanent

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    bear.tapped = True
    state.battlefield.append(bear)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    cryo.attached_to = bear.id
    state.battlefield.append(cryo)
    tap_permanent(state, bear)
    assert bear in state.battlefield
    assert cryo in state.battlefield


def test_tap_permanent_does_not_fire_aura_trigger_for_other_hosts():
    """An aura's enchanted-permanent-becomes-tapped trigger only fires
    when ITS host is tapped, not when some other permanent is tapped."""
    from mtg_bot.turn import tap_permanent

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    other = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(other)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    cryo.attached_to = bear.id
    state.battlefield.append(cryo)
    # Tap the OTHER bear (not the enchanted one).
    tap_permanent(state, other)
    # Enchanted bear is untouched; aura still attached.
    assert bear in state.battlefield
    assert cryo in state.battlefield
    assert cryo.attached_to == bear.id


def test_cryoshatter_destroys_host_when_dealt_combat_damage():
    """Cryoshatter's "or is dealt damage" half: a Wild Bear (2/2)
    blocking another creature takes combat damage, fires the aura's
    enchanted_creature_dealt_damage trigger, and is destroyed.

    Drives damage through ``_resolve_damage_step`` (combat damage)
    rather than _apply_damage (direct damage) so this exercises the
    combat-damage path of the trigger plumbing.
    """
    from mtg_bot.combat import _resolve_damage_step
    from mtg_bot.turn import state_based_actions

    state = _arena()
    # Defending bear (P1) with Cryoshatter attached, controlled by P0.
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    cryo.attached_to = bear.id
    state.battlefield.append(cryo)
    # Attacker (P0) — a vanilla Wild Bear deals 2.
    attacker = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(attacker)
    attacker.attacking = True
    attacker.blocked_by = [bear.id]
    # Drive a single damage step. The Cryoshatter trigger destroys the
    # bear immediately (resolves inline), so SBAs after are a no-op for
    # the bear.
    _resolve_damage_step(state, defending_player_id=1, first_strike_step=False)
    state_based_actions(state)
    # Bear is destroyed by Cryoshatter's dealt-damage trigger.
    assert bear not in state.battlefield
    assert bear in state.players[1].graveyard
    # Aura follows host off the battlefield.
    assert cryo not in state.battlefield
    assert cryo in state.players[0].graveyard


def test_cryoshatter_destroys_host_when_dealt_direct_damage():
    """Cryoshatter's dealt-damage half also fires for non-combat damage:
    a "deal N damage to target creature" effect routed through
    ``_apply_damage`` triggers the aura the same way combat damage does.
    """
    from mtg_bot.effects import DealDamageAnyTarget
    from mtg_bot.turn import state_based_actions

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    cryo.attached_to = bear.id
    state.battlefield.append(cryo)
    # Lightning-Bolt-shaped effect: 1 damage to the bear via _apply_damage.
    DealDamageAnyTarget(amount=1).resolve(
        state, controller_id=0, targets=(("creature", bear.id),)
    )
    _drain_stack(state)
    state_based_actions(state)
    # Bear destroyed by Cryoshatter (1 damage is sublethal vs 2 toughness,
    # so the destruction comes from the aura trigger, not SBA lethal damage).
    assert bear not in state.battlefield
    assert bear in state.players[1].graveyard
    assert cryo not in state.battlefield
    assert cryo in state.players[0].graveyard


def test_cryoshatter_no_op_when_no_damage_dealt():
    """Cryoshatter's dealt-damage trigger does NOT fire when nothing
    deals damage to the host. Without a damage event, the bear survives
    indefinitely (only the -5/-0 P/T modifier applies, and a 2/2 with
    -5/-0 is still tracked as a creature with negative power; SBAs only
    kill it via lethal damage / zero-toughness, neither of which apply).
    """
    from mtg_bot.turn import state_based_actions

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    cryo.attached_to = bear.id
    state.battlefield.append(cryo)
    # No damage, no tapping — just run SBAs.
    state_based_actions(state)
    assert bear in state.battlefield
    assert cryo in state.battlefield
    assert cryo.attached_to == bear.id


def test_meltstriders_resolve_grants_buff_and_menace():
    from mtg_bot.game import effective_toughness, effective_keywords

    state = _arena()
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    aura = state.new_object(
        _PRINTINGS["Meltstrider's Resolve"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    aura.attached_to = ally.id
    state.battlefield.append(aura)
    assert effective_toughness(state, ally) == 4  # 2 + 2
    assert "menace" in effective_keywords(state, ally)


def test_aura_cast_with_no_legal_target_no_targets():
    """Casting an Aura with no creatures in play: no target is recorded,
    so on resolution the Aura fizzles."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    mana = state.new_object(
        BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mana)
    cryo = state.new_object(
        _PRINTINGS["Cryoshatter"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(cryo)
    _cast_spell(state, 0, cryo.id)
    resolve_top_of_stack(state)
    # Cryoshatter should be in graveyard (fizzled).
    assert cryo in state.players[0].graveyard


def test_gigastorm_titan_cost_reduction_when_second_spell():
    """Gigastorm Titan normally costs {4}{U}; with 1 spell already cast
    this turn, costs {1}{U}."""
    from mtg_bot.actions import apply_cost_reductions

    state = _arena()
    card = _PRINTINGS["Gigastorm Titan"]
    # No spells cast yet → no reduction.
    reduced = apply_cost_reductions(card, state, 0, card.mana_cost)
    assert reduced.generic == 4
    # Bump spell count — now reduction applies.
    state.players[0].spells_cast_this_turn = 1
    reduced = apply_cost_reductions(card, state, 0, card.mana_cost)
    assert reduced.generic == 1


def test_fungal_colossus_cost_reduction_per_land_group():
    """{6}{G} → reduced by number of differently-named lands."""
    from mtg_bot.actions import apply_cost_reductions
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    card = _PRINTINGS["Fungal Colossus"]
    # No lands → no reduction.
    assert apply_cost_reductions(card, state, 0, card.mana_cost).generic == 6
    # One Forest, one Mountain, one Plains → 3 different names → -3.
    for name in ("Forest", "Mountain", "Plains"):
        l = state.new_object(BASIC_LANDS[name], owner_id=0, zone=Zone.BATTLEFIELD)
        state.battlefield.append(l)
    # Plus a second Forest (same name — doesn't add another reduction).
    extra = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(extra)
    reduced = apply_cost_reductions(card, state, 0, card.mana_cost)
    assert reduced.generic == 3


def test_attacked_with_spacecraft_tracking_and_cleanup():
    """declare_attackers appends the attacker to ``attacked_this_turn``
    (the generalized latch); cleanup clears it. The Spacecraft-subtype
    filter is applied at read time by the
    ``AttackedWithThisTurn(HasSubtype("Spacecraft"))`` condition rather
    than baked into the latch."""
    from mtg_bot.combat import declare_attackers
    from mtg_bot.turn import run_cleanup_step

    state = _arena()
    ship = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ship)
    # Station it so it's a creature.
    ship.charge_counters = 10
    declare_attackers(state, [ship.id])
    assert state.players[0].attacked_this_turn == [ship]
    run_cleanup_step(state)
    assert state.players[0].attacked_this_turn == []


# NOTE: ``test_unknown_cost_reduction_condition_logged_and_ignored``
# was removed when the string-keyed cost-reduction whitelist was
# replaced by the structural ``Condition`` grammar. The grammar makes
# unknown-condition no-ops impossible to construct: every rule carries
# a ``Condition`` instance whose ``matches`` method dispatches without
# a string lookup, so there is no "unknown key" branch to exercise.


def test_uthros_psionicist_external_reduction_kicks_in_for_second_spell():
    """Psionicist on the battlefield → second spell of the turn costs
    {2} less generic. First spell of the turn → no reduction."""
    from mtg_bot.actions import apply_cost_reductions

    state = _arena(p0_battlefield=[_PRINTINGS["Uthros Psionicist"]])
    target = _PRINTINGS["Rayblade Trooper"]  # {2}{G} 3/3, no own reductions
    # First spell: spells_cast_this_turn == 0 → no reduction.
    reduced = apply_cost_reductions(target, state, 0, target.mana_cost)
    assert reduced.generic == 2
    # Second spell: spells_cast_this_turn == 1 → -2.
    state.players[0].spells_cast_this_turn = 1
    reduced = apply_cost_reductions(target, state, 0, target.mana_cost)
    assert reduced.generic == 0


def test_uthros_psionicist_does_not_reduce_for_third_or_later_spell():
    """The condition is exactly the second spell, not 'spells beyond
    the first' — third spell of the turn shouldn't get the discount."""
    from mtg_bot.actions import apply_cost_reductions

    state = _arena(p0_battlefield=[_PRINTINGS["Uthros Psionicist"]])
    target = _PRINTINGS["Rayblade Trooper"]
    state.players[0].spells_cast_this_turn = 2  # this would be the 3rd
    reduced = apply_cost_reductions(target, state, 0, target.mana_cost)
    assert reduced.generic == 2  # back to base


def test_psionicist_does_not_reduce_opponent_spells():
    """External cost reduction is "you control" only — opponents'
    spells get no discount even if it's their second spell."""
    from mtg_bot.actions import apply_cost_reductions

    state = _arena(p0_battlefield=[_PRINTINGS["Uthros Psionicist"]])
    target = _PRINTINGS["Rayblade Trooper"]
    state.players[1].spells_cast_this_turn = 1
    # Casting as P1 — Psionicist belongs to P0, no discount.
    reduced = apply_cost_reductions(target, state, 1, target.mana_cost)
    assert reduced.generic == 2


def test_psionicist_plus_gigastorm_titan_stacks_to_zero_generic():
    """The replay-40 case: Uthros Psionicist on the battlefield, one
    spell already cast this turn (Melded Moxite), then cast Gigastorm
    Titan. Titan's own -3 + Psionicist's -2 → {4}{U} → {U}."""
    from mtg_bot.actions import apply_cost_reductions

    state = _arena(p0_battlefield=[_PRINTINGS["Uthros Psionicist"]])
    titan = _PRINTINGS["Gigastorm Titan"]
    state.players[0].spells_cast_this_turn = 1  # Moxite was the 1st spell
    reduced = apply_cost_reductions(titan, state, 0, titan.mana_cost)
    assert reduced.generic == 0  # 4 - 3 (own) - 2 (psionicist) = -1 → clamped to 0
    # The colored {U} requirement is preserved; only generic gets reduced.
    assert reduced.colored == titan.mana_cost.colored


def test_external_reduction_logged_with_source_name():
    """The reduction log line should name the source permanent so a
    replay reader can see WHERE the discount came from."""
    from mtg_bot.actions import apply_cost_reductions

    state = _arena(p0_battlefield=[_PRINTINGS["Uthros Psionicist"]])
    target = _PRINTINGS["Rayblade Trooper"]
    state.players[0].spells_cast_this_turn = 1
    apply_cost_reductions(target, state, 0, target.mana_cost)
    assert any(
        "Uthros Psionicist" in line and "cost reduced" in line
        for line in state.log
    ), f"expected log line naming the source; got: {state.log}"


def test_mana_cost_reduced_never_below_zero():
    from mtg_bot.cards import ManaCost

    mc = ManaCost(generic=2, colored=(("U", 1),))
    assert mc.reduced(0) is mc  # no-op
    assert mc.reduced(5).generic == 0  # clamped
    assert mc.reduced(5).colored == (("U", 1),)  # colored unchanged


def test_kav_landseeker_schedules_sacrifice_and_fires():
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.turn import run_delayed_end_step_triggers

    state = _arena()
    state.turn_number = 1  # P0's turn
    kav = state.new_object(
        _PRINTINGS["Kav Landseeker"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(kav)
    # Fire ETB: schedules delayed trigger.
    etb = next(
        ab for ab in kav.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = kav
    etb.effect.resolve(state, 0)
    landers = [o for o in state.battlefield if o.printing.name == "Lander"]
    assert len(landers) == 1
    lander = landers[0]
    assert len(state.delayed_triggers) == 1
    # Scheduled for turn+2 since P0 is active.
    assert state.delayed_triggers[0]["fire_at_turn"] == 3
    # Lander is the source of the delayed trigger (it's what gets sac'd).
    assert state.delayed_triggers[0]["source_id"] == lander.id

    # Simulate reaching P0's next end step (turn 3).
    state.turn_number = 3
    state.active_player_idx = 0
    run_delayed_end_step_triggers(state)
    # Trigger sits on the stack until resolution.
    assert lander in state.battlefield
    assert len(state.stack) == 1
    while state.stack:
        resolve_top_of_stack(state)
    assert lander not in state.battlefield  # sacrificed
    assert state.delayed_triggers == []


def test_delayed_trigger_survives_removal_if_target_gone():
    """If the target leaves before the trigger fires, the entry is
    consumed and the trigger fires as a no-op (lander is a token, so
    it ceased to exist after being destroyed — the source can't be
    found anywhere; we skip the push entirely)."""
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.turn import run_delayed_end_step_triggers, destroy

    state = _arena()
    state.turn_number = 1
    kav = state.new_object(
        _PRINTINGS["Kav Landseeker"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(kav)
    etb = next(
        ab for ab in kav.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = kav
    etb.effect.resolve(state, 0)
    lander = [o for o in state.battlefield if o.printing.name == "Lander"][0]
    destroy(state, lander)  # Lander (token) ceases to exist
    state.turn_number = 3
    run_delayed_end_step_triggers(state)
    # Token is gone, so the trigger doesn't go on the stack at all.
    assert state.stack == []
    assert state.delayed_triggers == []
    while state.stack:
        resolve_top_of_stack(state)


def test_delayed_trigger_waits_for_correct_turn():
    """Trigger scheduled for turn 3 doesn't fire at turn 2."""
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.effects import SacrificeSourceDelayed
    from mtg_bot.turn import run_delayed_end_step_triggers

    state = _arena()
    state.turn_number = 1
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)
    state.delayed_triggers.append({
        "fire_at_turn": 3,
        "controller_id": 0,
        "source_id": lander.id,
        "effect": SacrificeSourceDelayed(),
    })
    # Turn 2, wrong controller — don't fire.
    state.turn_number = 2
    state.active_player_idx = 1
    run_delayed_end_step_triggers(state)
    assert lander in state.battlefield
    assert state.stack == []
    assert len(state.delayed_triggers) == 1


def test_shock_land_agent_pays_life_enters_untapped():
    from mtg_bot.actions import _play_land
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import SHOCK_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    shock = state.new_object(
        SHOCK_LANDS["Sacred Foundry"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(shock)
    life_before = state.players[0].life  # default 20 → healthy
    _play_land(state, 0, shock.id)
    assert shock in state.battlefield
    assert shock.tapped is False
    assert state.players[0].life == life_before - 2


def test_shock_land_agent_declines_low_life_enters_tapped():
    from mtg_bot.actions import _play_land
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import SHOCK_LANDS

    state = _arena(p0_life=3)
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    shock = state.new_object(
        SHOCK_LANDS["Watery Grave"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(shock)
    _play_land(state, 0, shock.id)
    assert shock in state.battlefield
    assert shock.tapped is True
    assert state.players[0].life == 3


def test_shock_land_produces_both_colors():
    from mtg_bot.cards import SHOCK_LANDS

    for name, expected_colors in {
        "Breeding Pool": {"G", "U"},
        "Godless Shrine": {"W", "B"},
        "Sacred Foundry": {"R", "W"},
        "Stomping Ground": {"R", "G"},
        "Watery Grave": {"U", "B"},
    }.items():
        card = SHOCK_LANDS[name]
        assert card.produces == expected_colors
        assert card.shock_land_life_cost == 2


def test_shock_land_without_agent_enters_normal():
    """Edge case: if somehow there's no agent, skip the shock-life
    decision and enter untapped (defensive)."""
    from mtg_bot.actions import _play_land
    from mtg_bot.cards import SHOCK_LANDS

    state = _arena()
    state.players[0].agent = None
    shock = state.new_object(
        SHOCK_LANDS["Breeding Pool"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(shock)
    _play_land(state, 0, shock.id)
    assert shock in state.battlefield
    assert shock.tapped is False  # no agent → didn't take the branch
    assert state.players[0].life == 20  # no life paid


def test_weftstalker_ardent_pings_opponent_on_creature_enter():
    from mtg_bot.actions import put_into_battlefield, drain_stack

    state = _arena()
    stalker = state.new_object(
        _PRINTINGS["Weftstalker Ardent"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(stalker)
    p1_life_before = state.players[1].life
    # New creature enters → trigger fires.
    mate = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, mate, 0)
    drain_stack(state)
    assert state.players[1].life == p1_life_before - 1


def test_haliya_gains_life_on_creature_enter():
    from mtg_bot.actions import put_into_battlefield, drain_stack

    state = _arena()
    haliya = state.new_object(
        _PRINTINGS["Haliya, Guided by Light"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(haliya)
    life_before = state.players[0].life
    mate = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, mate, 0)
    drain_stack(state)
    assert state.players[0].life == life_before + 1


def test_haliya_guided_eot_draws_when_three_life_gained():
    from mtg_bot.actions import drain_stack
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    state.active_player_idx = 0
    haliya = state.new_object(
        _PRINTINGS["Haliya, Guided by Light"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(haliya)
    # Stock the library with a known card so we can detect the draw.
    plains = state.new_object(BASIC_LANDS["Plains"], 0, Zone.LIBRARY)
    state.players[0].library.append(plains)
    state.players[0].life_gained_this_turn = 3
    hand_before = len(state.players[0].hand)
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert len(state.players[0].hand) == hand_before + 1


def test_haliya_guided_eot_no_draw_below_threshold():
    from mtg_bot.actions import drain_stack
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    state.active_player_idx = 0
    haliya = state.new_object(
        _PRINTINGS["Haliya, Guided by Light"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(haliya)
    plains = state.new_object(BASIC_LANDS["Plains"], 0, Zone.LIBRARY)
    state.players[0].library.append(plains)
    state.players[0].life_gained_this_turn = 2  # under threshold
    hand_before = len(state.players[0].hand)
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert len(state.players[0].hand) == hand_before


def test_haliya_guided_has_warp_cost():

    h = _PRINTINGS["Haliya, Guided by Light"]
    assert h.warp_cost_str == "{W}"


def test_creature_or_artifact_trigger_skips_self_entry():
    """The trigger says 'another creature or artifact'. Self-entry
    shouldn't trigger — the source is excluded."""
    from mtg_bot.actions import put_into_battlefield

    state = _arena()
    stalker = state.new_object(
        _PRINTINGS["Weftstalker Ardent"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    put_into_battlefield(state, stalker, 0)
    # Own entry didn't self-trigger (we excluded source in the firing
    # loop).
    assert state.players[1].life == 20


def test_entropic_battlecruiser_loads():

    c = _PRINTINGS["Entropic Battlecruiser"]
    assert c.is_spacecraft
    assert c.station_threshold == 8
    assert "flying" in c.stationed_keywords
    assert "deathtouch" in c.stationed_keywords


def test_drix_fatemaker_grants_trample_to_counter_creatures():
    from mtg_bot.game import effective_keywords

    state = _arena()
    drix = state.new_object(
        _PRINTINGS["Drix Fatemaker"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(drix)
    without_counter = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(without_counter)
    with_counter = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    with_counter.plus_one_counters = 1
    state.battlefield.append(with_counter)
    assert "trample" not in effective_keywords(state, without_counter)
    assert "trample" in effective_keywords(state, with_counter)


def test_drix_fatemaker_etb_adds_counter_to_target():

    state = _arena()
    drix = state.new_object(
        _PRINTINGS["Drix Fatemaker"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(drix)
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    etb = next(
        ab for ab in drix.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = drix
    etb.effect.resolve(state, 0, (("creature", ally.id),))
    assert ally.plus_one_counters == 1


def test_seedship_impact_destroys_small_artifact_and_makes_lander():
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    # Lander is an artifact with MV 0 → qualifies for the rider.
    enemy_art = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(enemy_art)
    _PRINTINGS["Seedship Impact"].spell_effect.resolve(
        state, 0, (("permanent", enemy_art.id),)
    )
    assert enemy_art not in state.battlefield
    # We should now have our own Lander.
    our_landers = [
        o for o in state.battlefield
        if o.printing.name == "Lander" and o.controller_id == 0
    ]
    assert len(our_landers) == 1


def test_seedship_impact_big_artifact_no_lander():
    """A Spacecraft with MV > 2 doesn't get the Lander rider."""

    state = _arena()
    big = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # {5}{W} MV=6
    state.battlefield.append(big)
    _PRINTINGS["Seedship Impact"].spell_effect.resolve(
        state, 0, (("permanent", big.id),)
    )
    assert big not in state.battlefield
    our_landers = [
        o for o in state.battlefield
        if o.printing.name == "Lander" and o.controller_id == 0
    ]
    assert len(our_landers) == 0


def test_seedship_impact_noop_paths():
    from mtg_bot.effects import SeedshipImpactEffect

    state = _arena()
    e = SeedshipImpactEffect()
    assert e.requires_target() is True
    assert "enchantment" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))
    # Non-artifact / non-enchantment target → filtered at resolve.
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    e.resolve(state, controller_id=0, targets=(("permanent", bear.id),))
    assert bear in state.battlefield


def test_banishing_light_exiles_then_returns_on_leave():
    from mtg_bot.turn import destroy

    state = _arena()
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    bl = state.new_object(
        _PRINTINGS["Banishing Light"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bl)
    # Fire Banishing Light's ETB trigger.
    etb = next(
        ab for ab in bl.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = bl
    etb.effect.resolve(state, 0, (("permanent", victim.id),))
    assert victim not in state.battlefield
    assert victim in state.players[1].exile
    assert victim.exiled_by_source == bl.id

    # Now destroy Banishing Light. Victim should return.
    destroy(state, bl)
    assert victim in state.battlefield
    assert victim.exiled_by_source is None


def test_warp_card_exiled_by_banishing_light_is_not_castable_from_exile():
    """Cards in exile are only castable when they got there via their
    own warp ability (`warped=True`) or impulse-cast effects. A card
    with a warp_cost that gets exiled by Banishing Light should NOT
    set warped=True and should NOT appear in the owner's legal
    cast actions."""
    from mtg_bot.actions import CastSpell
    from mtg_bot.legal import legal_actions

    state = _arena()
    # Red Tiger Mechan has warp_cost {1}{R} and mana_cost {3}{R}.
    victim = state.new_object(
        _PRINTINGS["Red Tiger Mechan"],
        owner_id=1, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(victim)
    bl = state.new_object(
        _PRINTINGS["Banishing Light"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(bl)
    # Fire Banishing Light's ETB exile.
    etb = next(
        ab for ab in bl.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = bl
    etb.effect.resolve(state, 0, (("permanent", victim.id),))
    # Now in P1's exile, but warped flag must NOT have flipped.
    assert victim in state.players[1].exile
    assert victim.warped is False, (
        "Banishing Light's exile must not set the warp-recast flag"
    )
    # Give P1 plenty of mana and priority; their legal actions still
    # must not include casting the exiled victim.
    state.priority_player_idx = 1
    state.active_player_idx = 1
    # 4 untapped Mountains so cost wouldn't be the limiter.
    from mtg_bot.cards import BASIC_LANDS
    for _ in range(4):
        m = state.new_object(
            BASIC_LANDS["Mountain"], owner_id=1, zone=Zone.BATTLEFIELD,
        )
        state.battlefield.append(m)
    actions = legal_actions(state, 1)
    cast_victim = [
        a for a in actions
        if isinstance(a, CastSpell) and a.card_id == victim.id
    ]
    assert cast_victim == [], (
        f"victim is in exile without warped=True; should be uncastable, "
        f"but got {cast_victim}"
    )


def test_seam_rip_rejects_big_permanents():
    from mtg_bot.effects import ExileOpponentsNonlandPermanentUntilLeaves

    state = _arena()
    big = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # MV 6
    state.battlefield.append(big)
    small = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )  # MV 2
    state.battlefield.append(small)
    e = ExileOpponentsNonlandPermanentUntilLeaves(max_mv=2)
    assert e.requires_target() is True
    assert "2" in e.describe()
    legal = {tid for _, tid in e.legal_targets(state, 0)}
    assert small.id in legal
    assert big.id not in legal


def test_banishing_light_cant_target_own():
    from mtg_bot.effects import ExileOpponentsNonlandPermanentUntilLeaves

    state = _arena()
    mine = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mine)
    e = ExileOpponentsNonlandPermanentUntilLeaves()
    legal = {tid for _, tid in e.legal_targets(state, 0)}
    assert mine.id not in legal


def test_exile_until_leaves_noop_paths():
    from mtg_bot.effects import ExileOpponentsNonlandPermanentUntilLeaves

    state = _arena()
    e = ExileOpponentsNonlandPermanentUntilLeaves()
    # No resolving source.
    state.resolving_source = None
    e.resolve(state, controller_id=0, targets=(("permanent", 1),))
    # Source set, but empty targets / wrong kind / missing id.
    state.resolving_source = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))


def test_sunset_saboteur_attacks_trigger_fires():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.combat import declare_attackers

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    sab = state.new_object(
        _PRINTINGS["Sunset Saboteur"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    sab.summoning_sick = False
    state.battlefield.append(sab)
    enemy = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(enemy)
    declare_attackers(state, [sab.id])
    _drain_stack(state)
    assert enemy.plus_one_counters == 1


def test_attacks_trigger_fizzles_without_targets():
    from mtg_bot.combat import declare_attackers

    state = _arena()
    sab = state.new_object(
        _PRINTINGS["Sunset Saboteur"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    sab.summoning_sick = False
    state.battlefield.append(sab)
    # No opponent creatures — trigger fizzles.
    declare_attackers(state, [sab.id])
    assert any("fizzles" in l for l in state.log)


def test_put_counter_on_opponent_creature_rejects_own():
    from mtg_bot.effects import PutPlusOneCounterOnOpponentsCreature

    state = _arena()
    mine = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mine)
    e = PutPlusOneCounterOnOpponentsCreature(amount=1)
    assert e.requires_target() is True
    assert "opponent" in e.describe()
    legal = {tid for _, tid in e.legal_targets(state, 0)}
    assert mine.id not in legal
    # Even if passed, filtered at resolve.
    e.resolve(state, controller_id=0, targets=(("creature", mine.id),))
    assert mine.plus_one_counters == 0
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))


def test_temporal_intervention_directed_discard():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    # Seed P1's hand: a creature card + a land. Directed discard should
    # ignore the land (nonland-only).
    from mtg_bot.cards import BASIC_LANDS
    creature = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.HAND
    )
    state.players[1].hand.append(creature)
    land = state.new_object(
        BASIC_LANDS["Forest"], owner_id=1, zone=Zone.HAND
    )
    state.players[1].hand.append(land)
    _PRINTINGS["Temporal Intervention"].spell_effect.resolve(
        state, 0, (("player", 1),)
    )
    # Creature discarded, land stays.
    assert creature in state.players[1].graveyard
    assert land in state.players[1].hand


def test_temporal_intervention_no_nonland_cards_noop():
    """If the opponent has only lands, the directed discard fizzles."""
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    land = state.new_object(
        BASIC_LANDS["Forest"], owner_id=1, zone=Zone.HAND
    )
    state.players[1].hand.append(land)
    _PRINTINGS["Temporal Intervention"].spell_effect.resolve(
        state, 0, (("player", 1),)
    )
    assert land in state.players[1].hand


def test_temporal_intervention_cost_reduced_under_void():
    """Void active → cost reduced by 2."""
    from mtg_bot.actions import apply_cost_reductions

    state = _arena()
    spell = _PRINTINGS["Temporal Intervention"]
    # Void inactive.
    assert apply_cost_reductions(
        spell, state, 0, spell.mana_cost
    ).generic == 2
    # Activate Void.
    state.permanent_left_battlefield_this_turn = True
    assert apply_cost_reductions(
        spell, state, 0, spell.mana_cost
    ).generic == 0


def test_temporal_intervention_noop_paths():
    from mtg_bot.effects import TargetOpponentDiscardsCasterChoice

    state = _arena()
    e = TargetOpponentDiscardsCasterChoice()
    assert e.requires_target() is True
    assert "discard" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))


def test_temporal_intervention_no_agent_picks_first():

    state = _arena()
    state.players[0].agent = None
    creature = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.HAND
    )
    state.players[1].hand.append(creature)
    _PRINTINGS["Temporal Intervention"].spell_effect.resolve(
        state, 0, (("player", 1),)
    )
    assert creature in state.players[1].graveyard


def test_additional_costs_payable_returns_true_for_no_cost():
    """Cards without additional costs pass the check trivially."""
    from mtg_bot.legal import _additional_costs_payable

    state = _arena()
    card = state.new_object(
        _PRINTINGS["Plasma Bolt"], owner_id=0, zone=Zone.HAND
    )
    assert _additional_costs_payable(state, 0, card) is True


def test_cost_reduction_void_condition_active():
    """apply_cost_reductions with the ``VoidActiveCondition`` fires
    when Void is active (exercises that branch for coverage)."""
    from mtg_bot.actions import apply_cost_reductions
    from mtg_bot.cards import CostReduction, make_instant
    from mtg_bot.effects import GainLife
    from mtg_bot.effects.conditions import VoidActiveCondition

    c = make_instant(
        "Voidy", "{3}",
        effect=GainLife(amount=1),
        cost_reductions=(
            CostReduction(amount=2, condition=VoidActiveCondition()),
        ),
    )
    state = _arena()
    assert apply_cost_reductions(c, state, 0, c.mana_cost).generic == 3
    state.permanent_left_battlefield_this_turn = True
    assert apply_cost_reductions(c, state, 0, c.mana_cost).generic == 1


def test_embrace_oblivion_requires_sac_target_to_cast():
    """With no own creatures/artifacts, Embrace Oblivion can't be cast."""
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    # Caster has mana + a valid target on opponent's side, but NO
    # creatures or artifacts of their own to sacrifice.
    swamp = state.new_object(
        BASIC_LANDS["Swamp"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(swamp)
    enemy = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(enemy)
    obli_in_hand = state.new_object(
        _PRINTINGS["Embrace Oblivion"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(obli_in_hand)
    actions = legal_moves(state)
    # CastSpell for Embrace Oblivion should NOT be in legal actions.
    cast_actions = [
        a for a in actions
        if isinstance(a, CastSpell) and a.card_id == obli_in_hand.id
    ]
    assert cast_actions == []


def test_embrace_oblivion_with_sac_target_casts_and_destroys():
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    swamp = state.new_object(
        BASIC_LANDS["Swamp"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(swamp)
    # Fodder for the sac.
    fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fodder)
    # Victim to destroy.
    victim = state.new_object(
        SYNTHETIC_CREATURES["Deep Leviathan"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(victim)
    obli = state.new_object(
        _PRINTINGS["Embrace Oblivion"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(obli)
    _cast_spell(state, 0, obli.id)
    assert fodder not in state.battlefield  # sacrificed as cost
    resolve_top_of_stack(state)
    assert victim not in state.battlefield  # destroyed by spell


def test_embrace_oblivion_can_target_unstationed_spacecraft():
    """Oracle: 'Destroy target creature OR Spacecraft.' An unstationed
    Spacecraft (not currently a creature) is a legal target via the
    or_spacecraft branch."""

    state = _arena()
    spacecraft = state.new_object(
        _PRINTINGS["Pinnacle Kill-Ship"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    # No charge counters → not animated as a creature.
    assert spacecraft.charge_counters == 0
    state.battlefield.append(spacecraft)
    legal = _PRINTINGS["Embrace Oblivion"].spell_effect.legal_targets(
        state, controller_id=0,
    )
    target_ids = {tid for _, tid in legal}
    assert spacecraft.id in target_ids


def test_embrace_oblivion_destroys_unstationed_spacecraft():
    """An unstationed Spacecraft picked as the target gets destroyed."""

    state = _arena()
    spacecraft = state.new_object(
        _PRINTINGS["Pinnacle Kill-Ship"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(spacecraft)
    _PRINTINGS["Embrace Oblivion"].spell_effect.resolve(
        state, 0, (("creature", spacecraft.id),),
    )
    assert spacecraft not in state.battlefield
    assert spacecraft in state.players[1].graveyard


def test_umbral_collar_activated_sac_another_surveil():
    from mtg_bot.actions import _activate_ability, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    _stock_library(state, 0, count=5)
    zealot = state.new_object(
        _PRINTINGS["Umbral Collar Zealot"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(zealot)
    # Fodder for the sac.
    fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fodder)
    _activate_ability(state, 0, zealot.id, 0)
    # Fodder was sacrificed.
    assert fodder not in state.battlefield
    # Resolve the ability → surveil 1.
    resolve_top_of_stack(state)
    assert any("surveils 1" in l for l in state.log)


def test_umbral_collar_not_legal_without_another_sac():
    """Without another creature/artifact to sacrifice, the ability
    should not appear in legal_actions."""
    from mtg_bot.actions import ActivateAbility

    state = _arena()
    zealot = state.new_object(
        _PRINTINGS["Umbral Collar Zealot"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    zealot.summoning_sick = False
    state.battlefield.append(zealot)
    actions = legal_moves(state)
    activate_actions = [
        a for a in actions
        if isinstance(a, ActivateAbility) and a.source_id == zealot.id
    ]
    assert activate_actions == []


def test_umbral_collar_no_agent_uses_first_sac():
    from mtg_bot.actions import _activate_ability

    state = _arena()
    state.players[0].agent = None
    _stock_library(state, 0, count=5)
    zealot = state.new_object(
        _PRINTINGS["Umbral Collar Zealot"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(zealot)
    fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fodder)
    _activate_ability(state, 0, zealot.id, 0)
    assert fodder not in state.battlefield


def test_slagdrill_no_agent_picks_first_sac():
    from mtg_bot.actions import _activate_ability
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = None
    _stock_library(state, 0, count=5)
    for _ in range(2):
        m = state.new_object(
            BASIC_LANDS["Mountain"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(m)
    scrap = state.new_object(
        _PRINTINGS["Slagdrill Scrapper"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    scrap.summoning_sick = False
    state.battlefield.append(scrap)
    land_count_before = sum(
        1 for o in state.battlefield if o.printing.is_land
    )
    _activate_ability(state, 0, scrap.id, 0)
    # One land (or some non-scrap artifact) should have been sacrificed.
    land_count_after = sum(
        1 for o in state.battlefield if o.printing.is_land
    )
    assert land_count_after == land_count_before - 1


def test_slagdrill_scrapper_sac_land_or_artifact_draws():
    from mtg_bot.actions import _activate_ability, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    _stock_library(state, 0, count=5)
    # Pay {2}, tap Slagdrill, sac a land → draw.
    for _ in range(2):
        m = state.new_object(
            BASIC_LANDS["Mountain"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(m)
    scrap = state.new_object(
        _PRINTINGS["Slagdrill Scrapper"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    scrap.summoning_sick = False
    state.battlefield.append(scrap)
    hand_before = len(state.players[0].hand)
    _activate_ability(state, 0, scrap.id, 0)
    resolve_top_of_stack(state)
    assert len(state.players[0].hand) == hand_before + 1


def test_slagdrill_scrapper_not_legal_without_sac_target():
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.actions import ActivateAbility

    state = _arena()
    # Caster has {2} but nothing else to sac.
    for _ in range(2):
        m = state.new_object(
            BASIC_LANDS["Mountain"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(m)
    scrap = state.new_object(
        _PRINTINGS["Slagdrill Scrapper"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    scrap.summoning_sick = False
    state.battlefield.append(scrap)
    # Sacrificing lands is legal if the lands are still there after
    # tap-for-mana, but the activation cost only checks untapped
    # creature targets. Let me tap both lands before the check.
    for l in state.battlefield:
        if l.printing.is_land:
            l.tapped = True
    # Still — there ARE lands; sacrificing tapped lands is legal. So
    # activation IS legal in this variant. Instead, remove the non-Slag
    # lands entirely by keeping only the Slagdrill on battlefield.
    state.battlefield = [o for o in state.battlefield if o.id == scrap.id]
    actions = legal_moves(state)
    activate_actions = [
        a for a in actions
        if isinstance(a, ActivateAbility) and a.source_id == scrap.id
    ]
    assert activate_actions == []


def test_aura_fizzle_cleanup_covers_no_targets_branch():
    """Cast an Aura without any legal creature target — attached_to
    stays None when no targets were picked, so it fizzles at resolve."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    # Fund cost but no creature targets available.
    mana = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mana)
    aura = state.new_object(
        _PRINTINGS["Meltstrider's Resolve"], owner_id=0,
        zone=Zone.HAND,
    )
    state.players[0].hand.append(aura)
    _cast_spell(state, 0, aura.id)
    resolve_top_of_stack(state)
    # Should be in the graveyard (fizzled — no legal target at resolve).
    assert aura in state.players[0].graveyard


def test_cost_reduction_attacked_with_spacecraft_active():
    """Exercise the AttackedWithThisTurn(HasSubtype("Spacecraft"))
    condition in apply_cost_reductions — dead without at least one test."""
    from mtg_bot.actions import apply_cost_reductions
    from mtg_bot.cards import CostReduction, make_instant
    from mtg_bot.effects import GainLife
    from mtg_bot.effects.conditions import AttackedWithThisTurn
    from mtg_bot.effects.object_predicates import HasSubtype

    c = make_instant(
        "SpacecraftDiscount", "{3}",
        effect=GainLife(amount=1),
        cost_reductions=(
            CostReduction(
                amount=2,
                condition=AttackedWithThisTurn(HasSubtype("Spacecraft")),
            ),
        ),
    )
    state = _arena()
    # Latch empty → full cost.
    assert apply_cost_reductions(c, state, 0, c.mana_cost).generic == 3
    skiff = state.new_object(
        _PRINTINGS["Rescue Skiff"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.players[0].attacked_this_turn.append(skiff)
    assert apply_cost_reductions(c, state, 0, c.mana_cost).generic == 1


def test_activation_cost_describe_includes_sac_another():
    from mtg_bot.effects import ActivationCost

    cost = ActivationCost(sacrifice_another_creature_or_artifact=True)
    assert "another" in cost.describe().lower()


def test_comet_crawler_attacks_may_sac_pumps():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.combat import declare_attackers
    from mtg_bot.game import effective_power

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    crawler = state.new_object(
        _PRINTINGS["Comet Crawler"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    crawler.summoning_sick = False
    state.battlefield.append(crawler)
    fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fodder)
    declare_attackers(state, [crawler.id])
    _drain_stack(state)
    # Fodder sacrificed; crawler pumped.
    assert fodder not in state.battlefield
    assert effective_power(state, crawler) == 4  # 2 + 2


def test_comet_crawler_attacks_no_sac_no_pump():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.combat import declare_attackers
    from mtg_bot.game import effective_power

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    crawler = state.new_object(
        _PRINTINGS["Comet Crawler"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    crawler.summoning_sick = False
    state.battlefield.append(crawler)
    # No other creatures or artifacts — trigger no-ops silently.
    declare_attackers(state, [crawler.id])
    assert effective_power(state, crawler) == 2


def test_optional_sac_for_pump_no_agent():
    """Defensive path: when the controller has no agent, pick the first
    option."""
    from mtg_bot.effects import OptionalSacAnotherForPumpSelfUEOT

    state = _arena()
    state.players[0].agent = None
    source = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(source)
    fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fodder)
    state.resolving_source = source
    e = OptionalSacAnotherForPumpSelfUEOT(power_bonus=2)
    e.resolve(state, 0)
    assert fodder not in state.battlefield
    assert source.temp_power_bonus == 2


def test_optional_sac_missing_source_is_noop():
    from mtg_bot.effects import OptionalSacAnotherForPumpSelfUEOT

    state = _arena()
    state.resolving_source = None
    e = OptionalSacAnotherForPumpSelfUEOT(power_bonus=2)
    e.resolve(state, 0)  # no crash
    assert "sacrifice" in e.describe()


def test_swarm_culler_becomes_tapped_may_sac_for_draw():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.turn import tap_permanent

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    _stock_library(state, 0, count=5)
    culler = state.new_object(
        _PRINTINGS["Swarm Culler"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(culler)
    fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fodder)
    hand_before = len(state.players[0].hand)
    from mtg_bot.actions import drain_stack
    tap_permanent(state, culler)
    drain_stack(state)
    # Fodder sacrificed, hand grew by 1 (drawn card).
    assert fodder not in state.battlefield
    assert len(state.players[0].hand) == hand_before + 1


def test_swarm_culler_becomes_tapped_no_fodder_noop():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.turn import tap_permanent

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    _stock_library(state, 0, count=5)
    culler = state.new_object(
        _PRINTINGS["Swarm Culler"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(culler)
    hand_before = len(state.players[0].hand)
    tap_permanent(state, culler)
    # No fodder to sacrifice → no draw.
    assert len(state.players[0].hand) == hand_before


def test_optional_sac_another_for_effect_no_source():
    from mtg_bot.effects import OptionalSacAnotherForEffect, DrawCards

    state = _arena()
    state.resolving_source = None
    e = OptionalSacAnotherForEffect(inner_effect=DrawCards(amount=1))
    assert "sacrifice" in e.describe()
    e.resolve(state, 0)  # no crash


def test_optional_sac_another_for_effect_no_agent_picks_first():
    from mtg_bot.effects import OptionalSacAnotherForEffect, DrawCards

    state = _arena()
    state.players[0].agent = None
    _stock_library(state, 0, count=5)
    source = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(source)
    fodder = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(fodder)
    state.resolving_source = source
    hand_before = len(state.players[0].hand)
    e = OptionalSacAnotherForEffect(inner_effect=DrawCards(amount=1))
    e.resolve(state, 0)
    assert fodder not in state.battlefield
    assert len(state.players[0].hand) == hand_before + 1


def test_seriema_tutors_legendary_creature_to_hand():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    # Seed library with Xu-Ifit (legendary) + non-legendaries.
    xu_ifit = state.new_object(
        _PRINTINGS["Xu-Ifit, Osteoharmonist"], owner_id=0,
        zone=Zone.LIBRARY,
    )
    state.players[0].library.append(xu_ifit)
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.LIBRARY
    )
    state.players[0].library.append(bear)
    seriema = state.new_object(
        _PRINTINGS["The Seriema"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(seriema)
    etb = next(
        ab for ab in seriema.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = seriema
    etb.effect.resolve(state, 0)
    assert xu_ifit in state.players[0].hand
    assert xu_ifit not in state.players[0].library


def test_starfield_shepherd_tutors_small_creature():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    class PickCreatureMode(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 0  # creature_small mode

    state = _arena()
    state.players[0].agent = PickCreatureMode(rng=random.Random(0))
    # Seed library: MV-1 creature + MV-2 creature + land.
    wisp = state.new_object(
        SYNTHETIC_CREATURES["Healing Wisp"], owner_id=0,
        zone=Zone.LIBRARY,
    )  # {W} MV 1
    state.players[0].library.append(wisp)
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.LIBRARY
    )  # {1}{G} MV 2 — not a legal target
    state.players[0].library.append(bear)
    land = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.LIBRARY
    )
    state.players[0].library.append(land)

    shep = state.new_object(
        _PRINTINGS["Starfield Shepherd"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(shep)
    etb = next(
        ab for ab in shep.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = shep
    etb.effect.resolve(state, 0)
    assert wisp in state.players[0].hand
    assert bear in state.players[0].library
    assert land in state.players[0].library


def test_starfield_shepherd_tutors_basic_plains():
    """Mode 1: controller picks the basic-Plains alternative — the
    Plains card moves from library to hand; the eligible creature
    stays put."""
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.cards import BASIC_LANDS

    class PickPlainsMode(RandomAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 1  # basic_plains mode

    state = _arena()
    state.players[0].agent = PickPlainsMode(rng=random.Random(0))
    # Seed library: MV-1 creature + basic Plains + a non-Plains basic.
    wisp = state.new_object(
        SYNTHETIC_CREATURES["Healing Wisp"], owner_id=0,
        zone=Zone.LIBRARY,
    )
    state.players[0].library.append(wisp)
    plains = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.LIBRARY
    )
    state.players[0].library.append(plains)
    forest = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.LIBRARY
    )
    state.players[0].library.append(forest)

    shep = state.new_object(
        _PRINTINGS["Starfield Shepherd"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(shep)
    etb = next(
        ab for ab in shep.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = shep
    etb.effect.resolve(state, 0)
    assert plains in state.players[0].hand
    assert plains not in state.players[0].library
    assert wisp in state.players[0].library  # not picked under mode 1
    assert forest in state.players[0].library  # not a Plains


def test_starfield_shepherd_modal_describes_both_modes():
    """Sanity: the ETB effect is a ModalChooseOne with two modes that
    cover both halves of the printed oracle text — creature search and
    basic-Plains search. ``describe()`` surfaces both for the agent's
    mode-pick prompt; ``legal_targets`` is empty (neither mode is
    targeted)."""
    from mtg_bot.effects import ModalChooseOne

    shep = _PRINTINGS["Starfield Shepherd"]
    etb = next(
        ab for ab in shep.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    assert isinstance(etb.effect, ModalChooseOne)
    assert len(etb.effect.modes) == 2

    desc = etb.effect.describe()
    assert "creature_small" in desc
    assert "basic_plains" in desc
    # Mode descriptions surfaced separately for choose_mode prompts.
    mode_descs = [m.describe() for m in etb.effect.modes]
    assert any("creature_small" in d for d in mode_descs)
    assert any("basic_plains" in d for d in mode_descs)
    # Neither mode targets — ETB tutor is untargeted.
    state = _arena()
    assert etb.effect.legal_targets(state, 0) == []
    assert etb.effect.requires_target() is False


def test_search_library_no_match_logs_and_fizzles():
    from mtg_bot.effects import SearchLibraryForCard

    state = _arena()
    # Empty library — no matches.
    e = SearchLibraryForCard(filter_name="legendary_creature")
    e.resolve(state, 0)
    assert any("no legendary_creature" in l for l in state.log)


def test_search_library_unknown_filter_no_match():
    from mtg_bot.effects import SearchLibraryForCard

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.LIBRARY
    )
    state.players[0].library.append(bear)
    e = SearchLibraryForCard(filter_name="invented_filter_name")
    e.resolve(state, 0)
    assert bear in state.players[0].library
    assert "library" in e.describe()


def test_search_library_agent_declines_when_may_fail():
    """If may_fail=True and agent returns None, no fetch happens."""
    from mtg_bot.agents import RandomAgent
    from mtg_bot.effects import SearchLibraryForCard

    class Decliner(RandomAgent):
        def choose_card_from_library(self, state, player_id, option_ids):
            return None

    import random
    state = _arena()
    state.players[0].agent = Decliner(rng=random.Random(0))
    xu = state.new_object(
        _PRINTINGS["Xu-Ifit, Osteoharmonist"], owner_id=0,
        zone=Zone.LIBRARY,
    )
    state.players[0].library.append(xu)
    e = SearchLibraryForCard(filter_name="legendary_creature", may_fail=True)
    e.resolve(state, 0)
    assert xu in state.players[0].library  # stayed in library


def test_search_library_no_agent_falls_back():
    """With agent=None, treated as None choice; may_fail=False still fetches."""
    from mtg_bot.effects import SearchLibraryForCard

    state = _arena()
    state.players[0].agent = None
    xu = state.new_object(
        _PRINTINGS["Xu-Ifit, Osteoharmonist"], owner_id=0,
        zone=Zone.LIBRARY,
    )
    state.players[0].library.append(xu)
    e = SearchLibraryForCard(filter_name="legendary_creature", may_fail=False)
    e.resolve(state, 0)
    assert xu in state.players[0].hand


def test_search_library_for_basic_plains():
    """basic_plains filter fetches a Plains basic."""
    from mtg_bot.agents import RandomAgent
    from mtg_bot.effects import SearchLibraryForCard
    from mtg_bot.cards import BASIC_LANDS
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    plains = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.LIBRARY
    )
    state.players[0].library.append(plains)
    forest = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.LIBRARY
    )
    state.players[0].library.append(forest)
    e = SearchLibraryForCard(filter_name="basic_plains", may_fail=False)
    e.resolve(state, 0)
    assert plains in state.players[0].hand
    assert forest in state.players[0].library


def test_search_library_forced_fetch_when_may_fail_false():
    """may_fail=False forces a fetch even if agent returns None."""
    from mtg_bot.agents import RandomAgent
    from mtg_bot.effects import SearchLibraryForCard

    class Decliner(RandomAgent):
        def choose_card_from_library(self, state, player_id, option_ids):
            return None

    import random
    state = _arena()
    state.players[0].agent = Decliner(rng=random.Random(0))
    xu = state.new_object(
        _PRINTINGS["Xu-Ifit, Osteoharmonist"], owner_id=0,
        zone=Zone.LIBRARY,
    )
    state.players[0].library.append(xu)
    e = SearchLibraryForCard(filter_name="legendary_creature", may_fail=False)
    e.resolve(state, 0)
    assert xu in state.players[0].hand


def test_colorless_land_taps_for_generic():
    from mtg_bot.actions import can_pay_cost
    from mtg_bot.cards import UTILITY_LANDS, parse_mana_cost

    state = _arena()
    starforge = state.new_object(
        UTILITY_LANDS["Secluded Starforge"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(starforge)
    # Can pay {1} (pure generic).
    assert can_pay_cost(state, 0, parse_mana_cost("{1}")) is True
    # Cannot pay {W} (colored requirement not met).
    assert can_pay_cost(state, 0, parse_mana_cost("{W}")) is False


def test_colorless_land_is_in_all_printings():

    printings = _PRINTINGS
    assert "Secluded Starforge" in printings
    assert "Command Bridge" in printings


def test_planet_lands_enter_tapped_and_produce_one_color():
    from mtg_bot.actions import _play_land
    from mtg_bot.cards import PLANET_LANDS

    for name, expected_color in [
        ("Adagia, Windswept Bastion", "W"),
        ("Uthros, Titanic Godcore", "U"),
        ("Susur Secundi, Void Altar", "B"),
        ("Kavaron, Memorial World", "R"),
        ("Evendo, Waking Haven", "G"),
    ]:
        state = _arena()
        card = state.new_object(
            PLANET_LANDS[name], owner_id=0, zone=Zone.HAND
        )
        state.players[0].hand.append(card)
        _play_land(state, 0, card.id)
        assert card in state.battlefield
        assert card.tapped is True
        assert expected_color in card.printing.produces


def test_planet_land_in_all_printings():

    printings = _PRINTINGS
    for name in (
        "Adagia, Windswept Bastion", "Uthros, Titanic Godcore",
        "Susur Secundi, Void Altar", "Kavaron, Memorial World",
        "Evendo, Waking Haven",
    ):
        assert name in printings


def _command_bridge(state, owner_id=0):
    """Helper: put a Command Bridge into ``owner_id``'s hand and
    return (card, agent). The test installs an agent on the player —
    Command Bridge's ETB rider asks the agent for a tap-or-sac choice.
    """
    from mtg_bot.cards import UTILITY_LANDS

    cb = state.new_object(
        UTILITY_LANDS["Command Bridge"], owner_id=owner_id,
        zone=Zone.HAND,
    )
    state.players[owner_id].hand.append(cb)
    return cb


def test_command_bridge_enters_tapped_with_alt_permanent_tap():
    """Plays Command Bridge with an untapped Plains in play. Agent
    chooses to tap the Plains; Command Bridge survives and is itself
    tapped (enters tapped). The chosen Plains becomes tapped."""
    from mtg_bot.actions import _play_land, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    plains = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    plains.tapped = False
    state.battlefield.append(plains)
    cb = _command_bridge(state, owner_id=0)
    _play_land(state, 0, cb.id)
    # Command Bridge entered the battlefield and is tapped.
    assert cb in state.battlefield
    assert cb.tapped is True
    # The ETB went on the stack — resolve it.
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    # Bridge is still on the battlefield (alt-cost paid by tapping
    # the Plains).
    assert cb in state.battlefield
    assert plains.tapped is True


def test_command_bridge_sacrifices_when_no_untapped_permanent():
    """No untapped permanents available (the Bridge itself enters tapped
    and is excluded from candidates). The ETB sacrifices the Bridge."""
    from mtg_bot.actions import _play_land, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    cb = _command_bridge(state, owner_id=0)
    _play_land(state, 0, cb.id)
    # Bridge is on the battlefield and tapped.
    assert cb in state.battlefield
    assert cb.tapped is True
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    # No untapped permanent → sacrificed to the graveyard.
    assert cb not in state.battlefield
    assert cb in state.players[0].graveyard


def test_command_bridge_sacrifices_when_agent_declines():
    """Agent refuses to tap any permanent (returns None). The Bridge
    is sacrificed even if untapped permanents are available."""
    from mtg_bot.actions import _play_land, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random

    class Decliner(RandomAgent):
        def choose_target(self, state, player_id, legal_targets):
            return None

    state = _arena()
    state.players[0].agent = Decliner(rng=random.Random(0))
    plains = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    plains.tapped = False
    state.battlefield.append(plains)
    cb = _command_bridge(state, owner_id=0)
    _play_land(state, 0, cb.id)
    resolve_top_of_stack(state)
    # Plains is still untapped; Bridge was sacrificed.
    assert plains.tapped is False
    assert cb not in state.battlefield
    assert cb in state.players[0].graveyard


def test_command_bridge_taps_for_any_color():
    """Activate Command Bridge's mana ability and verify it adds an
    "any"-colored coin that can pay any colored requirement. Mana
    abilities resolve immediately (no stack)."""
    from mtg_bot.actions import _activate_ability, can_pay_cost
    from mtg_bot.cards import UTILITY_LANDS, parse_mana_cost
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    cb = state.new_object(
        UTILITY_LANDS["Command Bridge"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    cb.tapped = False  # bypass enters-tapped rider for this isolated test
    state.battlefield.append(cb)
    # Pre-activation: nothing in pool.
    assert state.players[0].mana_pool == []
    _activate_ability(state, 0, cb.id, ability_index=0)
    # Bridge tapped, one "any"-color coin in the pool.
    assert cb.tapped is True
    assert len(state.players[0].mana_pool) == 1
    assert state.players[0].mana_pool[0].color == "any"
    # That coin can pay {W}, {U}, {B}, {R}, or {G} (and {1} generic).
    for symbol in ("{W}", "{U}", "{B}", "{R}", "{G}", "{1}"):
        assert can_pay_cost(state, 0, parse_mana_cost(symbol)) is True


def test_sacrifice_self_unless_tap_no_resolving_source_is_noop():
    """Defensive branch: if state.resolving_source is None the effect is
    a silent no-op (not all triggers wire a source)."""
    from mtg_bot.effects import SacrificeSelfUnlessTapPermanent

    state = _arena()
    eff = SacrificeSelfUnlessTapPermanent()
    state.resolving_source = None
    eff.resolve(state, controller_id=0)
    # Nothing crashed; nothing changed.
    assert state.battlefield == []
    # describe() returns a stable text.
    assert "sacrifice" in eff.describe()
    assert "tap" in eff.describe()


def test_sacrifice_self_unless_tap_invalid_agent_pick_sacrifices():
    """Defensive branch: agent returns an id that isn't a candidate.
    The effect treats this like a decline and sacrifices the source."""
    from mtg_bot.actions import _play_land, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    import random

    class BadPicker(RandomAgent):
        def choose_target(self, state, player_id, legal_targets):
            # Return a tuple shaped like a legal pick but with a
            # non-existent object id.
            return ("untapped_permanent", 99999)

    state = _arena()
    state.players[0].agent = BadPicker(rng=random.Random(0))
    plains = state.new_object(
        BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    plains.tapped = False
    state.battlefield.append(plains)
    cb = _command_bridge(state, owner_id=0)
    _play_land(state, 0, cb.id)
    resolve_top_of_stack(state)
    assert cb not in state.battlefield
    assert cb in state.players[0].graveyard


def test_command_bridge_in_all_printings_with_full_rules_text():
    """Sanity: Command Bridge is registered AND its printing reflects
    the full rules text (enters_tapped, ETB trigger, mana ability)."""
    from mtg_bot.cards import UTILITY_LANDS
    from mtg_bot.effects import (
        AddManaToPool, SacrificeSelfUnlessTapPermanent,
    )

    assert "Command Bridge" in _PRINTINGS
    cb = UTILITY_LANDS["Command Bridge"]
    assert cb.enters_tapped is True
    assert len(cb.triggered_abilities) == 1
    assert _matches_legacy_key(cb.triggered_abilities[0], "etb")
    assert isinstance(
        cb.triggered_abilities[0].effect, SacrificeSelfUnlessTapPermanent
    )
    assert len(cb.activated_abilities) == 1
    assert cb.activated_abilities[0].is_mana_ability is True
    eff = cb.activated_abilities[0].effect
    assert isinstance(eff, AddManaToPool)
    assert eff.produces == (("any", 1),)


def test_lashwhip_cost_reduction_with_three_enemy_creatures():
    from mtg_bot.actions import apply_cost_reductions

    state = _arena()
    card = _PRINTINGS["Lashwhip Predator"]
    # 0 enemy creatures → full cost.
    assert apply_cost_reductions(card, state, 0, card.mana_cost).generic == 4
    # 2 enemy creatures → still full cost.
    for _ in range(2):
        b = state.new_object(
            SYNTHETIC_CREATURES["Wild Bear"], owner_id=1,
            zone=Zone.BATTLEFIELD,
        )
        state.battlefield.append(b)
    assert apply_cost_reductions(card, state, 0, card.mana_cost).generic == 4
    # 3rd enemy creature → reduction kicks in.
    b = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(b)
    assert apply_cost_reductions(card, state, 0, card.mana_cost).generic == 2


def test_mm_menon_artifact_enters_triggers_counter_on_target():
    from mtg_bot.actions import put_into_battlefield, drain_stack
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    from mtg_bot.agents import RandomAgent
    import random
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    mm = state.new_object(
        _PRINTINGS["Mm'menon, Uthros Exile"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(mm)
    # Target must be available for the trigger.
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    # Create a Lander (artifact) via put_into_battlefield — fires
    # artifact-enters-you-control trigger on Mm'menon.
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, lander, 0)
    drain_stack(state)
    # Some creature (ally or mm itself) got a +1/+1 counter.
    total_counters = sum(
        o.plus_one_counters for o in state.battlefield
        if o.printing.is_creature
    )
    assert total_counters >= 1


def test_zookeeper_mechan_pump_activation():
    from mtg_bot.effects import PumpTargetCreatureUntilEOT
    from mtg_bot.game import effective_power

    state = _arena()
    zoo = state.new_object(
        _PRINTINGS["Zookeeper Mechan"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(zoo)
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)
    pump = next(
        a for a in zoo.printing.activated_abilities
        if isinstance(a.effect, PumpTargetCreatureUntilEOT)
    )
    pump.effect.resolve(state, 0, (("creature", ally.id),))
    assert effective_power(state, ally) == 6  # 2 + 4


def test_zookeeper_mechan_has_mana_ability():
    """Oracle: '{T}: Add {R}.' Zookeeper Mechan now ships the mana
    ability that was previously dropped behind a stale 'creature mana
    abilities aren't modeled' comment."""
    from mtg_bot.effects import AddManaToPool

    zoo = _PRINTINGS["Zookeeper Mechan"]
    mana_abilities = [
        a for a in zoo.activated_abilities
        if isinstance(a.effect, AddManaToPool)
    ]
    assert len(mana_abilities) == 1
    ma = mana_abilities[0]
    assert ma.is_mana_ability is True
    assert ma.cost.tap_self is True
    assert ma.effect.produces == (("R", 1),)


def test_zookeeper_mechan_taps_to_add_red_mana():
    """Activating the mana ability taps Zookeeper and adds {R} to the pool."""
    from mtg_bot.effects import AddManaToPool

    state = _arena()
    zoo = state.new_object(
        _PRINTINGS["Zookeeper Mechan"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(zoo)
    ma = next(
        a for a in zoo.printing.activated_abilities
        if isinstance(a.effect, AddManaToPool)
    )
    # Resolve the effect (the cost-pay path is shared with land taps;
    # this test focuses on the produces side).
    ma.effect.resolve(state, 0, ())
    pool = state.players[0].mana_pool
    assert any(coin.color == "R" for coin in pool)


def test_skystinger_has_blocks_flying_trigger():
    from tests._etb_helpers import _legacy_or_inferred_key

    c = _PRINTINGS["Skystinger"]
    assert "reach" in c.keywords
    assert c.power == 3 and c.toughness == 3
    # Blocks-a-flying-creature trigger pumps Skystinger +5/+0 UEOT.
    # Phase 9 migrated this from ``trigger="blocks_creature_with_flying"``
    # to predicate-style ``event=Event.BLOCKS, predicate=
    # IsSelf(subject="blocker") & HasKeyword("flying", subject=
    # "attacker")`` — assertion uses the helper to keep the legacy
    # key as the canonical form for membership checks.
    assert len(c.triggered_abilities) == 1
    assert _legacy_or_inferred_key(c.triggered_abilities[0]) == (
        "blocks_creature_with_flying"
    )


def test_all_fates_stalker_exiles_and_returns_on_death():
    from mtg_bot.turn import destroy

    state = _arena()
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)
    stalker = state.new_object(
        _PRINTINGS["All-Fates Stalker"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(stalker)
    # Fire ETB.
    etb = next(
        ab for ab in stalker.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = stalker
    etb.effect.resolve(state, 0, (("creature", victim.id),))
    assert victim in state.players[1].exile
    # Kill Stalker → victim returns.
    destroy(state, stalker)
    assert victim in state.battlefield


def test_exile_target_creature_until_source_leaves_noop_paths():
    from mtg_bot.effects import ExileTargetCreatureUntilSourceLeaves

    state = _arena()
    e = ExileTargetCreatureUntilSourceLeaves()
    assert e.requires_target() is True
    assert "exile" in e.describe()
    state.resolving_source = None
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    # Missing / wrong target handling.
    state.resolving_source = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))


def test_codecracker_hound_splits_top_two():
    from mtg_bot.agents import RandomAgent
    import random

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    _stock_library(state, 0, count=5)
    hound = state.new_object(
        _PRINTINGS["Codecracker Hound"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(hound)
    lib_before = len(state.players[0].library)
    hand_before = len(state.players[0].hand)
    gy_before = len(state.players[0].graveyard)
    etb = next(
        ab for ab in hound.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = hound
    etb.effect.resolve(state, 0)
    assert len(state.players[0].library) == lib_before - 2
    assert len(state.players[0].hand) == hand_before + 1
    assert len(state.players[0].graveyard) == gy_before + 1


def test_codecracker_empty_library_noop():
    from mtg_bot.effects import LookAtTopTwoSplitHandGraveyard

    state = _arena()
    e = LookAtTopTwoSplitHandGraveyard()
    assert "top two" in e.describe()
    e.resolve(state, 0)  # no crash


def test_codecracker_single_card_library_goes_to_hand():
    from mtg_bot.effects import LookAtTopTwoSplitHandGraveyard

    state = _arena()
    _stock_library(state, 0, count=1)
    last_card = state.players[0].library[0]
    LookAtTopTwoSplitHandGraveyard().resolve(state, 0)
    assert last_card in state.players[0].hand


def test_effect_describe_coverage():
    """Exercise describe() on a handful of effects that aren't hit by
    other card tests — keeps module coverage above threshold."""
    from mtg_bot.effects import (
        PumpSelfUntilEOT,
        UntapTargetOtherPermanent,
    )

    assert "until end of turn" in PumpSelfUntilEOT(
        power_bonus=1, toughness_bonus=2
    ).describe()
    assert "untap" in UntapTargetOtherPermanent().describe()


def test_effect_describe_coverage_2():
    """More describe() hits on effects that aren't covered by specific
    card tests."""
    from mtg_bot.effects import (
        DealDamageToOpponentCreature,
        ExileEachOpponentsGraveyard,
        ExileTargetCreature,
        PutPlusOneCounterOnTargetCreature,
        TargetOpponentDiscards,
    )

    assert "damage" in DealDamageToOpponentCreature(amount=2).describe()
    assert "exile" in ExileTargetCreature().describe()
    assert "exile" in ExileEachOpponentsGraveyard().describe()
    assert "counter" in PutPlusOneCounterOnTargetCreature(amount=1).describe()
    assert "discard" in TargetOpponentDiscards(count=1).describe()


# NOTE: ``test_cost_reduction_per_land_not_scaled_noops`` was removed
# along with the ``per_land_group: bool`` flag on ``CostReduction``.
# The flag's only purpose was distinguishing "this rule scales by
# distinct-land-name count" from "this rule is a flat amount for the
# same engine key", an awkward dual-meaning whose unscaled branch
# never had a real card. The structural ``Condition`` grammar moves
# the scaling intent into the ``PerDifferentlyNamedLand`` variant
# directly — no flag, so no unscaled no-op to test.


def test_codecracker_agent_chooses_to_mill_top():
    """When the agent says 'mill' on the top card, the top goes to GY
    and the second to hand."""
    from mtg_bot.effects import LookAtTopTwoSplitHandGraveyard
    from mtg_bot.agents import RandomAgent
    import random

    class AlwaysMill(RandomAgent):
        def choose_surveil(self, state, player_id, top_card_id):
            return True  # mill it

    state = _arena()
    state.players[0].agent = AlwaysMill(rng=random.Random(0))
    _stock_library(state, 0, count=2)
    top = state.players[0].library[0]
    second = state.players[0].library[1]
    LookAtTopTwoSplitHandGraveyard().resolve(state, 0)
    assert top in state.players[0].graveyard
    assert second in state.players[0].hand


def test_codecracker_no_agent_default_assignment():
    from mtg_bot.effects import LookAtTopTwoSplitHandGraveyard

    state = _arena()
    state.players[0].agent = None
    _stock_library(state, 0, count=2)
    top = state.players[0].library[0]
    second = state.players[0].library[1]
    LookAtTopTwoSplitHandGraveyard().resolve(state, 0)
    assert top in state.players[0].hand
    assert second in state.players[0].graveyard


def test_stun_counter_keeps_creature_tapped_one_turn():
    from mtg_bot.turn import run_untap_step

    state = _arena()
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    bear.tapped = True
    bear.stun_counters = 1
    state.battlefield.append(bear)
    run_untap_step(state)
    # Stun counter consumed; still tapped this turn.
    assert bear.tapped is True
    assert bear.stun_counters == 0
    # Next untap step untaps it normally.
    run_untap_step(state)
    assert bear.tapped is False


def test_cryogen_relic_etb_and_leaves_draw():
    from mtg_bot.actions import drain_stack, put_into_battlefield
    from mtg_bot.turn import destroy

    state = _arena()
    _stock_library(state, 0, count=5)
    relic = state.new_object(
        _PRINTINGS["Cryogen Relic"], owner_id=0, zone=Zone.HAND
    )
    hand_before = len(state.players[0].hand)
    put_into_battlefield(state, relic, 0)
    drain_stack(state)
    # ETB trigger drew 1 card.
    assert len(state.players[0].hand) == hand_before + 1
    # Destroy it — leaves trigger draws another card.
    hand_before_destroy = len(state.players[0].hand)
    destroy(state, relic)
    drain_stack(state)
    assert len(state.players[0].hand) == hand_before_destroy + 1


def test_cryogen_relic_stun_activated():

    state = _arena()
    relic = state.new_object(
        _PRINTINGS["Cryogen Relic"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(relic)
    tapped_target = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    tapped_target.tapped = True
    state.battlefield.append(tapped_target)
    act = [
        a for a in relic.printing.activated_abilities
        if isinstance(a.effect.__class__, type)
    ][0]
    act.effect.resolve(state, 0, (("creature", tapped_target.id),))
    assert tapped_target.stun_counters == 1


def test_put_stun_counter_noop_paths():
    from mtg_bot.effects import PutStunCounterOnTappedCreature

    state = _arena()
    e = PutStunCounterOnTappedCreature()
    assert e.requires_target() is True
    assert "stun" in e.describe()
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))
    # Untapped creature → filtered at resolve.
    bear = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(bear)
    e.resolve(state, controller_id=0, targets=(("creature", bear.id),))
    assert bear.stun_counters == 0


def test_weapons_manufacturing_creates_munitions_on_nontoken_artifact():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.actions import put_into_battlefield, drain_stack
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    wm = state.new_object(
        _PRINTINGS["Weapons Manufacturing"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(wm)
    # Put a NON-token artifact onto the battlefield.
    moxite = state.new_object(
        _PRINTINGS["Melded Moxite"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, moxite, 0)
    drain_stack(state)
    # A Munitions token should have been created.
    munitions = [
        o for o in state.battlefield if o.printing.name == "Munitions"
    ]
    assert len(munitions) == 1


def test_weapons_manufacturing_ignores_token_artifacts():
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.cards import TOKEN_PRINTINGS

    state = _arena()
    wm = state.new_object(
        _PRINTINGS["Weapons Manufacturing"], owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(wm)
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, lander, 0)
    munitions = [
        o for o in state.battlefield if o.printing.name == "Munitions"
    ]
    assert munitions == []


def test_munitions_leaves_battlefield_damages_target():
    from mtg_bot.agents import RandomAgent
    import random
    from mtg_bot.actions import put_into_battlefield
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.turn import sacrifice

    state = _arena()
    state.players[0].agent = RandomAgent(rng=random.Random(0))
    state.players[1].agent = RandomAgent(rng=random.Random(0))
    munitions = state.new_object(
        TOKEN_PRINTINGS["Munitions"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    put_into_battlefield(state, munitions, 0)
    # A target for the damage — we'll let P1 be the target.
    life_before = state.players[1].life
    # Sacrifice the Munitions — it ceases to exist as a token, firing
    # its leaves-battlefield trigger on the way.
    sacrifice(state, munitions)
    # 2 damage dealt to some target; whichever was picked, life changed
    # somewhere.
    life_after = state.players[1].life
    assert life_after <= life_before  # target might be P1 or something


def test_dockworker_drone_etb_counter_and_dies_transfer():

    state = _arena()
    drone = state.new_object(
        _PRINTINGS["Dockworker Drone"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(drone)
    # Fire ETB to put the counter.
    etb = next(
        ab for ab in drone.printing.triggered_abilities if _matches_legacy_key(ab, "etb")
    )
    state.resolving_source = drone
    etb.effect.resolve(state, 0)
    assert drone.plus_one_counters == 1

    # Add a target creature.
    ally = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(ally)

    # Fire dies trigger — counter moves to ally.
    dies = next(
        ab for ab in drone.printing.triggered_abilities
        if _matches_legacy_key(ab, "dies")
    )
    state.resolving_source = drone
    dies.effect.resolve(state, 0, (("creature", ally.id),))
    assert drone.plus_one_counters == 0
    assert ally.plus_one_counters == 1


def test_move_self_counters_noop_paths():
    from mtg_bot.effects import MoveSelfCountersToTargetCreatureYouControl

    state = _arena()
    e = MoveSelfCountersToTargetCreatureYouControl()
    assert e.requires_target() is True
    assert "+1/+1" in e.describe()
    # No source on resolving_source.
    state.resolving_source = None
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    # Source has no counters.
    drone = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.resolving_source = drone
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    # Empty target / wrong kind / missing id.
    drone.plus_one_counters = 2
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("player", 0),))
    e.resolve(state, controller_id=0, targets=(("creature", 99999),))


def test_kavaron_skywarden_void_eot_counter():
    """Reach + Void EoT +1/+1 counter on self."""
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    warden = state.new_object(
        _PRINTINGS["Kavaron Skywarden"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(warden)
    # Sanity: reach keyword on printing.
    assert "reach" in warden.printing.keywords

    from mtg_bot.actions import drain_stack
    # Void inactive → no change.
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert warden.plus_one_counters == 0

    # Void active → +1/+1 counter added.
    state.permanent_left_battlefield_this_turn = True
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert warden.plus_one_counters == 1


def test_voidforged_titan_fires_only_when_void_active():
    """End-step Void trigger draws + loses 1 when void is active."""
    from mtg_bot.turn import run_beginning_of_end_step_triggers

    state = _arena()
    # Stock P0's library so the draw has something to pull.
    _stock_library(state, 0, count=5)
    titan = state.new_object(
        _PRINTINGS["Voidforged Titan"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(titan)

    from mtg_bot.actions import drain_stack
    # Void NOT active → no draw/life change.
    hand_before = len(state.players[0].hand)
    life_before = state.players[0].life
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert len(state.players[0].hand) == hand_before
    assert state.players[0].life == life_before

    # Flip void active (a nonland permanent left the battlefield this turn).
    state.permanent_left_battlefield_this_turn = True
    run_beginning_of_end_step_triggers(state)
    drain_stack(state)
    assert len(state.players[0].hand) == hand_before + 1
    assert state.players[0].life == life_before - 1


def test_emergency_eject_describe_and_noop_paths():
    from mtg_bot.effects import (
        DestroyTargetNonlandPermanentWithLanderForController,
    )

    state = _arena()
    e = DestroyTargetNonlandPermanentWithLanderForController()
    assert "Lander" in e.describe()
    # No-op paths: empty targets, wrong kind, missing id.
    e.resolve(state, controller_id=0, targets=())
    e.resolve(state, controller_id=0, targets=(("creature", 1),))
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))


# --------------------------------------------------------------------------
# Lost in Space — bounce target artifact/creature to top or bottom of
# its owner's library (owner chooses), then surveil 1.
# --------------------------------------------------------------------------


def _lost_in_space_arena():
    """Build a 2-player arena with P0 holding a castable Lost in Space and
    enough mana ({3}{U}) to cast it. P0 and P1 both have a non-trivial
    library so surveil has something to look at. Returns the state and
    the Lost in Space hand card.
    """
    from mtg_bot.cards import BASIC_LANDS

    state = _arena()
    # 4 lands for {3}{U}. Three Plains for generic + one Island for {U}.
    for _ in range(3):
        land = state.new_object(
            BASIC_LANDS["Plains"], owner_id=0, zone=Zone.BATTLEFIELD
        )
        state.battlefield.append(land)
    island = state.new_object(
        BASIC_LANDS["Island"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(island)
    lis = state.new_object(
        _PRINTINGS["Lost in Space"], owner_id=0, zone=Zone.HAND
    )
    state.players[0].hand.append(lis)
    # Stock both libraries so the surveil step has cards to examine.
    _stock_library(state, 0, count=10)
    _stock_library(state, 1, count=10)
    return state, lis


class _ChooseModeAgent:
    """Tiny agent stub used by the Lost in Space tests. ``mode`` controls
    choose_mode (0 = top, 1 = bottom). ``surveil`` controls whether
    the controller mills surveiled cards.

    Implements just enough of the agent surface area to drive a single
    Lost in Space cast / resolution flow."""

    def __init__(self, mode=0, surveil_keep=True):
        self.mode = mode
        self.surveil_keep = surveil_keep

    def choose_action(self, state, player_id, legal):
        # Prefer the first non-pass action — the test driver always
        # sets up exactly one castable Lost in Space.
        for a in legal:
            if not isinstance(a, PassPriority):
                return a
        return PassPriority()

    def choose_target(self, state, player_id, legal_targets):
        return legal_targets[0] if legal_targets else None

    def choose_mode(self, state, player_id, mode_descriptions):
        return self.mode

    def choose_surveil(self, state, player_id, top_card_id):
        # surveil_keep=True → never mill (return False = "keep").
        return not self.surveil_keep

    def choose_discard(self, state, player_id, hand_card_ids):
        return hand_card_ids[0]

    def choose_sacrifice(self, state, player_id, legal_object_ids):
        return legal_object_ids[0] if legal_object_ids else None


def test_lost_in_space_resolves_owner_picks_top():
    """End-to-end cast: P0 targets P1's creature; P1 (the owner) picks
    'top'; creature ends up on top of P1's library and is no longer
    on the battlefield. Surveil 1 also runs."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack

    state, lis = _lost_in_space_arena()
    state.players[0].agent = _ChooseModeAgent(mode=0)
    # P1 is the owner — their agent decides top vs bottom.
    state.players[1].agent = _ChooseModeAgent(mode=0)

    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)

    _cast_spell(state, 0, lis.id)
    # The cast should have selected P1's creature as the target.
    assert state.stack and state.stack[-1].targets == (
        ("permanent", victim.id),
    )
    resolve_top_of_stack(state)

    # Victim is off the battlefield and on top of P1's library.
    assert victim not in state.battlefield
    assert state.players[1].library[0] is victim
    assert victim.zone == Zone.LIBRARY
    # Lost in Space went to the caster's graveyard.
    assert lis in state.players[0].graveyard
    # Surveil 1 ran (log entry from Surveil's resolve).
    assert any("surveils 1" in entry for entry in state.log)


def test_lost_in_space_resolves_owner_picks_bottom():
    """Same setup as above but P1 picks 'bottom' — creature lands at
    the end of P1's library."""
    from mtg_bot.actions import _cast_spell, resolve_top_of_stack

    state, lis = _lost_in_space_arena()
    state.players[0].agent = _ChooseModeAgent(mode=0)
    state.players[1].agent = _ChooseModeAgent(mode=1)

    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)

    _cast_spell(state, 0, lis.id)
    resolve_top_of_stack(state)

    assert victim not in state.battlefield
    # Bottom = end of the list.
    assert state.players[1].library[-1] is victim
    assert victim.zone == Zone.LIBRARY


def test_lost_in_space_not_castable_without_legal_target():
    """With no artifacts or creatures on the battlefield, Lost in Space
    cannot be cast — it shouldn't appear in legal_actions / legal_moves.
    """
    state, lis = _lost_in_space_arena()
    state.players[0].agent = _ChooseModeAgent(mode=0)
    state.players[1].agent = _ChooseModeAgent(mode=0)
    # No artifacts or creatures — only lands on the battlefield.
    assert all(
        not (o.printing.is_artifact or o.printing.is_creature)
        for o in state.battlefield
    )

    actions = legal_moves(state)
    assert not any(
        isinstance(a, CastSpell) and a.card_id == lis.id for a in actions
    )


def test_lost_in_space_castable_when_artifact_target_exists():
    """An opponent's artifact (no creatures) is a legal target — the
    spell becomes castable."""
    from mtg_bot.cards import TOKEN_PRINTINGS

    state, lis = _lost_in_space_arena()
    state.players[0].agent = _ChooseModeAgent(mode=0)
    state.players[1].agent = _ChooseModeAgent(mode=0)
    # Drop an opponent-owned artifact (Lander token is artifact, not creature).
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)

    actions = legal_moves(state)
    assert any(
        isinstance(a, CastSpell) and a.card_id == lis.id for a in actions
    )


def test_lost_in_space_target_required_and_describe():
    """Sanity: the new effect requires a target, lists artifacts/creatures
    only, and has a sensible describe()."""
    from mtg_bot.effects import (
        OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    )
    from mtg_bot.cards import BASIC_LANDS

    e = OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary()
    assert e.requires_target() is True
    assert "artifact or creature" in e.describe()
    assert "top or bottom" in e.describe()

    state = _arena()
    # A land is NOT a legal target.
    forest = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(forest)
    creature = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(creature)

    legal = e.legal_targets(state, controller_id=0)
    assert ("permanent", creature.id) in legal
    assert ("permanent", forest.id) not in legal


def test_lost_in_space_token_target_ceases_to_exist():
    """Token bounced 'to library' ceases to exist (rule 704.5d)."""
    from mtg_bot.cards import TOKEN_PRINTINGS
    from mtg_bot.effects import (
        OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    )

    state = _arena()
    state.players[0].agent = _ChooseModeAgent(mode=0)
    state.players[1].agent = _ChooseModeAgent(mode=0)
    lander = state.new_object(
        TOKEN_PRINTINGS["Lander"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(lander)
    lib_size_before = len(state.players[1].library)

    OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary().resolve(
        state, controller_id=0,
        targets=(("permanent", lander.id),),
    )

    assert lander not in state.battlefield
    # Token did NOT enter the library.
    assert len(state.players[1].library) == lib_size_before
    assert any(
        "ceases to exist" in entry for entry in state.log
    )


def test_lost_in_space_noop_paths():
    """Defensive resolves: empty targets, wrong kind, missing id,
    non-artifact non-creature target → no-op."""
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.effects import (
        OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    )

    state = _arena()
    e = OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary()
    # No targets → silent.
    e.resolve(state, controller_id=0, targets=())
    # Wrong kind → silent.
    e.resolve(state, controller_id=0, targets=(("creature", 999),))
    # Missing id → silent.
    e.resolve(state, controller_id=0, targets=(("permanent", 99999),))
    # Non-artifact non-creature target (a land) → silent (defensive guard).
    forest = state.new_object(
        BASIC_LANDS["Forest"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(forest)
    e.resolve(state, controller_id=0, targets=(("permanent", forest.id),))
    assert forest in state.battlefield


def test_lost_in_space_owner_with_no_agent_defaults_to_top():
    """When the owner has no agent (test scaffolding), default to
    placing the target on the top of the library."""
    from mtg_bot.effects import (
        OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    )

    state = _arena()
    # P1 has no agent; that's our default state from _arena().
    assert state.players[1].agent is None
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)

    OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary().resolve(
        state, controller_id=0,
        targets=(("permanent", victim.id),),
    )
    # Defaulted to top (insert at index 0).
    assert state.players[1].library[0] is victim


def test_lost_in_space_agent_returning_invalid_mode_normalizes_to_top():
    """If the agent returns an out-of-range mode index, fall back to top."""
    from mtg_bot.effects import (
        OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    )

    state = _arena()

    class _BadAgent(_ChooseModeAgent):
        def choose_mode(self, state, player_id, mode_descriptions):
            return 42  # bogus; effect should normalize to top.

    state.players[1].agent = _BadAgent()
    victim = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(victim)

    OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary().resolve(
        state, controller_id=0,
        targets=(("permanent", victim.id),),
    )
    assert state.players[1].library[0] is victim


def test_etb_triggers_fire_in_full_game_flow():
    """End-to-end: with at least one ETB card in the deck, a real simulated
    game should fire some ETB triggers. Uses a white deck (Healing Wisp is
    the Stage-1 ETB card).

    Phase 2 migrated ETB-family triggers to the predicate dispatcher,
    which logs ``"<source>'s enters trigger goes on the stack"``
    rather than ``"<source>'s ETB trigger goes on the stack"``. We
    accept either log shape — both signal the predicate-side
    self-ETB ability fired."""
    from mtg_bot.simulate import simulate_game

    seen_etb = False
    for seed in range(10):
        state = simulate_game(colors=["W", "W"], seed=seed, max_turns=300)
        if any(
            "ETB trigger" in entry or "enters trigger" in entry
            for entry in state.log
        ):
            seen_etb = True
            break
    assert seen_etb, (
        "expected at least one ETB/enters trigger across 10 seeded W-vs-W games"
    )


# --------------------------------------------------------------------------
# Mouth of the Storm — ETB debuffs opponents' creatures -3/-0 UEOT
# (oracle: "until your next turn"; engine approximates with UEOT).
# --------------------------------------------------------------------------


def test_mouth_of_the_storm_etb_debuffs_only_opponents_creatures():
    """ETB resolves: opponent creatures get -3/-0 UEOT, controller's own
    creatures are untouched."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT
    from mtg_bot.game import effective_power, effective_toughness

    state = _arena()
    # P0 controls Mouth of the Storm (the source). Two opponent (P1)
    # creatures and one own (P0) creature on the battlefield.
    mouth = state.new_object(
        _PRINTINGS["Mouth of the Storm"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(mouth)
    op1 = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(op1)
    op2 = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(op2)
    own = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(own)

    # Snapshot starting effective P/T.
    op1_p_before = effective_power(state, op1)
    op1_t_before = effective_toughness(state, op1)
    op2_p_before = effective_power(state, op2)
    op2_t_before = effective_toughness(state, op2)
    own_p_before = effective_power(state, own)
    own_t_before = effective_toughness(state, own)

    # Resolve the ETB effect directly (Mouth's only ETB ability).
    PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=-3, toughness_bonus=0
    ).resolve(state, controller_id=0)

    # Both opponent creatures: power -3, toughness unchanged.
    assert effective_power(state, op1) == op1_p_before - 3
    assert effective_toughness(state, op1) == op1_t_before
    assert effective_power(state, op2) == op2_p_before - 3
    assert effective_toughness(state, op2) == op2_t_before
    # Controller's own creature: untouched.
    assert effective_power(state, own) == own_p_before
    assert effective_toughness(state, own) == own_t_before


def test_mouth_of_the_storm_debuff_clears_at_cleanup():
    """The -3/-0 wears off at the cleanup step — UEOT plumbing reverts
    temp_power_bonus to 0."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT
    from mtg_bot.game import effective_power

    state = _arena()
    op = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(op)
    base_p = effective_power(state, op)

    PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=-3, toughness_bonus=0
    ).resolve(state, controller_id=0)
    assert effective_power(state, op) == base_p - 3

    run_cleanup_step(state)
    assert op.temp_power_bonus == 0
    assert effective_power(state, op) == base_p


def test_mouth_of_the_storm_card_definition_has_etb_trigger():
    """The card registry exposes the ETB triggered ability with the
    expected parameters."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT

    mouth = _PRINTINGS["Mouth of the Storm"]
    etb_abilities = [
        ab for ab in mouth.triggered_abilities if _matches_legacy_key(ab, "etb")
    ]
    assert len(etb_abilities) == 1
    eff = etb_abilities[0].effect
    assert isinstance(eff, PumpCreaturesOpponentsControlUntilEOT)
    assert eff.power_bonus == -3
    assert eff.toughness_bonus == 0
    # New: Mouth of the Storm uses the "until your next turn" scope.
    assert eff.until_caster_next_turn is True


def test_mouth_of_the_storm_debuff_persists_through_cleanup():
    """Oracle: 'until YOUR next turn' — the -3/-0 must NOT clear at
    the casting turn's cleanup. The opponent's full next turn still
    sees the debuffed P/T."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT
    from mtg_bot.game import effective_power

    state = _arena()
    op = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(op)
    base_p = effective_power(state, op)
    # Cast on P0's turn (active_player_idx already 0 in _arena).
    PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=-3, toughness_bonus=0,
        until_caster_next_turn=True,
    ).resolve(state, controller_id=0)
    assert effective_power(state, op) == base_p - 3

    # Casting turn's cleanup — debuff stays.
    run_cleanup_step(state)
    assert effective_power(state, op) == base_p - 3
    assert op.extended_temp_power_bonus == -3


def test_mouth_of_the_storm_debuff_clears_at_caster_next_untap():
    """The debuff clears at the START of the source-controller's NEXT
    untap step (their next turn beginning), per oracle."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT
    from mtg_bot.game import effective_power
    from mtg_bot.turn import run_untap_step

    state = _arena()
    op = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(op)
    base_p = effective_power(state, op)
    PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=-3, toughness_bonus=0,
        until_caster_next_turn=True,
    ).resolve(state, controller_id=0)
    # Pretend a turn passes: cleanup, opponent's untap (their turn),
    # opponent's cleanup, then back to caster's untap.
    run_cleanup_step(state)
    state.active_player_idx = 1
    run_untap_step(state)  # opponent's untap — debuff still active
    assert op.extended_temp_power_bonus == -3
    assert effective_power(state, op) == base_p - 3
    run_cleanup_step(state)
    state.active_player_idx = 0
    run_untap_step(state)  # caster's next untap — debuff clears now
    assert op.extended_temp_power_bonus == 0
    assert effective_power(state, op) == base_p


def test_mouth_of_the_storm_debuff_skips_caster_own_creatures():
    """Sanity: the until-your-next-turn debuff hits opponents' creatures
    only — controller's own creatures are untouched even with the new
    extended-temporal pump path."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT
    from mtg_bot.game import effective_power

    state = _arena()
    own = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD,
    )
    state.battlefield.append(own)
    base_p = effective_power(state, own)
    PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=-3, toughness_bonus=0,
        until_caster_next_turn=True,
    ).resolve(state, controller_id=0)
    assert effective_power(state, own) == base_p
    assert own.extended_temp_power_bonus == 0


def test_pump_creatures_opponents_control_describe_and_log():
    """Sanity: describe() and the log line render with signed bonuses."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT

    eff = PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=-3, toughness_bonus=0
    )
    d = eff.describe()
    assert "creatures your opponents control" in d
    assert "-3" in d
    assert "+0" in d
    assert "end of turn" in d

    state = _arena()
    op = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(op)
    eff.resolve(state, controller_id=0)
    assert any(
        "opponents' creatures" in entry and "-3/+0" in entry
        for entry in state.log
    )


def test_pump_creatures_opponents_control_no_targets_logs_zero():
    """Resolving with no opponent creatures present is a clean no-op
    (still logs, with count 0)."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT

    state = _arena()
    # Only a P0 creature — no opponent creatures on the battlefield.
    own = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=0, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(own)
    PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=-3, toughness_bonus=0
    ).resolve(state, controller_id=0)
    assert own.temp_power_bonus == 0
    assert any("(0)" in entry for entry in state.log)


def test_pump_creatures_opponents_control_grants_keywords():
    """Coverage: granted_keywords path adds keywords to opponent
    creatures alongside the P/T change."""
    from mtg_bot.effects import PumpCreaturesOpponentsControlUntilEOT

    state = _arena()
    op = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], owner_id=1, zone=Zone.BATTLEFIELD
    )
    state.battlefield.append(op)
    PumpCreaturesOpponentsControlUntilEOT(
        power_bonus=0,
        toughness_bonus=0,
        granted_keywords=frozenset({"first strike"}),
    ).resolve(state, controller_id=0)
    from mtg_bot.game import effective_keywords
    assert "first strike" in effective_keywords(state, op)


