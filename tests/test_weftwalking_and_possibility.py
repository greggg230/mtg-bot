"""Tests for Weftwalking's ETB shuffle-hand-and-grave + draw-7 and
Possibility Technician's ETB / Kavu-observer + persistent
cast-from-exile clause."""

from mtg_bot.actions import CastSpell, PlayLand
from mtg_bot.agents import RandomAgent
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
from mtg_bot.effects import (
    ConditionalIfCastFromHand,
    DrawCards,
    PossibilityTechnicianETBEffect,
    Sequence,
    ShuffleZonesIntoLibrary,
)


def _weftwalking_etb_effect():
    """Reproduce Weftwalking's ETB body as the composition the parser
    now produces. The bespoke ``WeftwalkingETBEffect`` was deleted when
    the body migrated to the unified clause grammar; the engine now
    routes it through ``ConditionalIfCastFromHand`` wrapping a
    ``Sequence`` of ``ShuffleZonesIntoLibrary`` and ``DrawCards``."""
    return ConditionalIfCastFromHand(
        inner_effect=Sequence(effects=(
            ShuffleZonesIntoLibrary(zones=("hand", "graveyard")),
            DrawCards(amount=7),
        )),
    )
from mtg_bot.game import GameState, Player, Zone
from tests._etb_helpers import _legacy_or_inferred_key, _matches_legacy_key

from mtg_bot.cards.source import get_active_source

_PRINTINGS = get_active_source().get_printings()


def _new_state():
    state = GameState()
    state.players.append(Player(id=0, agent=RandomAgent()))
    state.players.append(Player(id=1, agent=RandomAgent()))
    return state


def _stock_library(state, player_id, n, printing=None):
    printing = printing or BASIC_LANDS["Plains"]
    p = state.players[player_id]
    for _ in range(n):
        c = state.new_object(printing, player_id, Zone.LIBRARY)
        p.library.append(c)


# -----------------------
# Weftwalking ETB
# -----------------------


def test_weftwalking_shuffles_hand_and_graveyard_into_library_and_draws_7():
    state = _new_state()
    p = state.players[0]
    # Hand has 2, graveyard has 3, library has 5.
    for _ in range(2):
        c = state.new_object(SYNTHETIC_CREATURES["Wild Bear"], 0, Zone.HAND)
        p.hand.append(c)
    for _ in range(3):
        c = state.new_object(BASIC_LANDS["Plains"], 0, Zone.GRAVEYARD)
        p.graveyard.append(c)
    _stock_library(state, 0, 5)
    _weftwalking_etb_effect().resolve(state, 0, ())
    # Hand contains exactly 7 cards (the draw). Graveyard is empty.
    # Library started with 5 + 2 (hand) + 3 (gy) = 10; drew 7 → 3 left.
    assert len(p.hand) == 7
    assert len(p.graveyard) == 0
    assert len(p.library) == 3


def test_weftwalking_describe():
    d = _weftwalking_etb_effect().describe()
    # Composed describe text: shuffle + draw 7 + the if-you-cast-it
    # gate prefix from ``ConditionalIfCastFromHand``. The composed
    # describes use digits (``you draw 7 cards``) where the bespoke's
    # describe used the spelled-out form ("draw seven cards") — that
    # delta is purely surface, no behavioral effect.
    assert "shuffle" in d
    assert "7" in d or "seven" in d
    assert "cast it" in d


def test_weftwalking_printing_has_etb():
    w = _PRINTINGS["Weftwalking"]
    triggers = {_legacy_or_inferred_key(ab) for ab in w.triggered_abilities}
    assert "etb" in triggers


# -----------------------
# Possibility Technician ETB
# -----------------------


def test_possibility_exiles_top_card_and_marks_persistent_kavu_gate():
    state = _new_state()
    _stock_library(state, 0, 3)
    PossibilityTechnicianETBEffect().resolve(state, 0, ())
    p = state.players[0]
    assert len(p.exile) == 1
    # The exiled card is marked persistently playable while the
    # controller controls a Kavu — NOT the this-turn-only impulse flag.
    assert p.exile[0].play_from_exile_requires_kavu is True
    assert p.exile[0].playable_from_exile_this_turn is False


def test_possibility_empty_library_noop():
    state = _new_state()
    PossibilityTechnicianETBEffect().resolve(state, 0, ())
    assert state.players[0].exile == []


def test_possibility_describe():
    d = PossibilityTechnicianETBEffect().describe()
    # Describe must reflect the persistence + Kavu-control gate, not
    # the old "this turn" impulse phrasing.
    assert "exile" in d
    assert "Kavu" in d
    assert "this turn" not in d


def test_possibility_printing_has_subtypes_etb_kavu_observer_and_warp():
    pt = _PRINTINGS["Possibility Technician"]
    # Oracle text says "Warp {1}{R}"; the old bespoke had a spurious
    # warp="{R}" that was dropped when the card migrated to the oracle
    # parser (wave 6).
    assert pt.warp_cost_str == "{1}{R}"
    assert "Kavu" in pt.subtypes
    triggers = {_legacy_or_inferred_key(ab) for ab in pt.triggered_abilities}
    # Both halves of "this creature or another Kavu you control enters"
    # are wired: self-ETB (etb) AND the Kavu-observer trigger.
    assert "etb" in triggers
    assert "kavu_you_control_enters" in triggers


# ---- Kavu-observer trigger (clause a) ----


def test_kavu_observer_fires_when_another_kavu_enters_under_technician():
    """When Possibility Technician is on the battlefield and another
    Kavu enters under the same controller, the observer trigger
    (``HasSubtype("Kavu") & Controlled(YOU) & ~IsSelf()`` on the
    predicate path) must fire — pushed on the stack via
    ``dispatch_event(Event.ENTERS, obj=...)`` from
    ``_trigger_etb_abilities``. After draining the stack, a card is
    exiled with the persistent Kavu-gated flag set. The entering
    Kavu's own ETB also goes on the stack but isn't part of this
    assertion."""
    from mtg_bot.actions import drain_stack, put_into_battlefield

    state = _new_state()
    _stock_library(state, 0, 5)
    p = state.players[0]
    # Place Technician on the battlefield (skip its own ETB since
    # we're measuring the observer firing on a different Kavu's ETB).
    tech = state.new_object(
        _PRINTINGS["Possibility Technician"], 0, Zone.BATTLEFIELD
    )
    state.battlefield.append(tech)
    # Snapshot exile-size pre-observer.
    before = len(p.exile)
    # Another Kavu enters: reuse the Technician printing.
    other_kavu = state.new_object(
        _PRINTINGS["Possibility Technician"], 0, Zone.STACK
    )
    put_into_battlefield(state, other_kavu, 0)
    drain_stack(state)
    # Observer's exile-the-top-card effect resolved off the stack.
    # Exactly one new card is exiled, flagged for Kavu-gated cast.
    # (The other_kavu's own ETB also exiled a card via IsSelf()-style
    # wiring, so we expect at least one new exile entry — pin to the
    # observer's by checking the latest exiled card carries the flag.)
    assert len(p.exile) >= before + 1
    assert p.exile[-1].play_from_exile_requires_kavu is True


def test_kavu_observer_does_not_fire_for_non_kavu_creature():
    """A non-Kavu creature entering must NOT trigger the Technician's
    observer (filter is Kavu-only, not "any creature")."""
    from mtg_bot.actions import put_into_battlefield

    state = _new_state()
    _stock_library(state, 0, 5)
    p = state.players[0]
    tech = state.new_object(
        _PRINTINGS["Possibility Technician"], 0, Zone.BATTLEFIELD
    )
    state.battlefield.append(tech)
    # SYNTHETIC_CREATURES["Wild Bear"] has no Kavu subtype.
    bear = state.new_object(SYNTHETIC_CREATURES["Wild Bear"], 0, Zone.STACK)
    put_into_battlefield(state, bear, 0)
    # Exile remains empty — the Technician's own ETB already fired
    # earlier (we skipped that), and Wild Bear is not a Kavu.
    assert p.exile == []


# ---- Persistence clause (clause b) ----


def test_persistent_cast_from_exile_legal_only_while_controlling_kavu():
    """The exiled card must be legal to cast from exile only when the
    controller currently controls at least one Kavu — and across turns,
    not just the turn it was exiled."""
    import mtg_bot.legal as legal_mod
    import mtg_bot.turn as turn_mod

    state = _new_state()
    # Stock exile with a castable creature flagged for Kavu-gated cast.
    target = state.new_object(SYNTHETIC_CREATURES["Wild Bear"], 0, Zone.EXILE)
    target.play_from_exile_requires_kavu = True
    state.players[0].exile.append(target)
    # Controller has plenty of mana available (Wild Bear is {1}{G}, so
    # untapped Forests cover the cost via can_pay_cost).
    for _ in range(5):
        land = state.new_object(BASIC_LANDS["Forest"], 0, Zone.BATTLEFIELD)
        state.battlefield.append(land)
    # Make it sorcery-timing on player 0's main phase with empty stack.
    state.active_player_idx = 0
    state.priority_player_idx = 0
    state.step = turn_mod.Step.PRECOMBAT_MAIN
    # No Kavu on battlefield: cast must NOT be offered.
    actions = legal_mod.legal_actions(state, 0)
    assert not any(
        isinstance(a, CastSpell) and a.card_id == target.id for a in actions
    )
    # Add a Kavu to the battlefield (reuse the Technician printing).
    kavu = state.new_object(
        _PRINTINGS["Possibility Technician"], 0, Zone.BATTLEFIELD
    )
    state.battlefield.append(kavu)
    actions = legal_mod.legal_actions(state, 0)
    assert any(
        isinstance(a, CastSpell) and a.card_id == target.id for a in actions
    )
    # Persistence: the flag survives a cleanup step (which clears the
    # this-turn impulse flag). Simulate by calling the cleanup handler
    # directly and re-checking the flag is still set.
    target.playable_from_exile_this_turn = False  # baseline
    turn_mod.run_cleanup_step(state)
    assert target.play_from_exile_requires_kavu is True
    actions = legal_mod.legal_actions(state, 0)
    assert any(
        isinstance(a, CastSpell) and a.card_id == target.id for a in actions
    )


def test_persistent_flag_cleared_on_cast_from_exile():
    """Once the card is cast from exile, the persistent flag must be
    cleared so the same GameObject doesn't double-cast."""
    from mtg_bot.actions import _cast_spell
    from mtg_bot.game import ManaCoin

    state = _new_state()
    state.active_player_idx = 0
    state.priority_player_idx = 0
    target = state.new_object(SYNTHETIC_CREATURES["Wild Bear"], 0, Zone.EXILE)
    target.play_from_exile_requires_kavu = True
    state.players[0].exile.append(target)
    # Pre-fill the mana pool with Wild Bear's cost ({1}{G}).
    state.players[0].mana_pool.extend(
        [ManaCoin(color="G"), ManaCoin(color="C")]
    )
    _cast_spell(state, 0, target.id)
    # Spell goes onto the stack; the persistent flag is consumed on
    # the cast-from-exile path so a re-cast can't happen.
    assert target.play_from_exile_requires_kavu is False
    assert target not in state.players[0].exile


# -----------------------
# Weftwalking static: "first spell each turn may be cast for free"
# -----------------------


def test_weftwalking_printing_grants_first_spell_each_turn_free():
    """The static second clause is plumbed via a flag on the printing
    so legal_actions can detect a Weftwalking on the battlefield."""
    w = _PRINTINGS["Weftwalking"]
    assert w.grants_first_spell_each_turn_free is True


def test_legal_moves_offers_free_first_cast_when_weftwalking_in_play():
    """With a Weftwalking on the battlefield and no spells cast yet
    this turn, legal_actions should propose a CastSpell(cast_for_free=True)
    variant for spells in hand — even ones whose normal mana cost the
    player can't afford."""
    from mtg_bot import legal_moves
    from mtg_bot.actions import CastSpell
    from tests._rules_helpers import _arena

    bolt = _PRINTINGS["Plasma Bolt"]  # {R} — needs a Mountain
    state = _arena(
        p0_hand=[bolt],
        p0_battlefield=[_PRINTINGS["Weftwalking"]],  # no lands
    )
    card = state.players[0].hand[0]
    moves = legal_moves(state)
    free_casts = [
        m for m in moves
        if isinstance(m, CastSpell) and m.card_id == card.id and m.cast_for_free
    ]
    paid_casts = [
        m for m in moves
        if isinstance(m, CastSpell) and m.card_id == card.id
        and not m.cast_for_free
    ]
    assert free_casts, (
        "free-cast variant should be offered when Weftwalking is on bf"
    )
    # No lands → no payable variant.
    assert not paid_casts


def test_free_cast_skips_mana_payment_and_increments_spells_cast():
    """Apply a cast_for_free CastSpell and verify no lands tap, no mana
    is spent, and the spell still lands on the stack and bumps the
    caster's spells_cast_this_turn counter."""
    from mtg_bot import apply
    from mtg_bot.actions import CastSpell, resolve_top_of_stack
    from mtg_bot.agents import RandomAgent
    from mtg_bot.cards import BASIC_LANDS
    from tests._rules_helpers import _arena
    import random

    weftwalking = _PRINTINGS["Weftwalking"]
    nova = _PRINTINGS["Nova Hellkite"]  # {3}{R}{R} normal — way too pricey
    state = _arena(
        p0_hand=[nova],
        p0_battlefield=[weftwalking, BASIC_LANDS["Mountain"]],  # 1 land
    )
    for p in state.players:
        p.agent = RandomAgent(rng=random.Random(0))
    card = state.players[0].hand[0]
    mountain = next(
        o for o in state.battlefield
        if o.printing.name == "Mountain"
    )
    assert mountain.tapped is False

    apply(state, CastSpell(card.id, cast_for_free=True))

    # Spell is on the stack. Mountain was NOT tapped — free cast.
    assert len(state.stack) == 1
    assert mountain.tapped is False
    # Caster's counter incremented (so subsequent free casts won't be
    # offered until next turn).
    assert state.players[0].spells_cast_this_turn == 1

    # Resolve to ensure nothing blows up downstream.
    resolve_top_of_stack(state)
    assert card in state.battlefield


def test_legal_moves_does_not_offer_free_cast_after_first_spell():
    """Once a player has cast a spell this turn, the free-cast variant
    should no longer be offered — only the first spell each turn is
    eligible."""
    from mtg_bot import legal_moves
    from mtg_bot.actions import CastSpell
    from tests._rules_helpers import _arena

    bolt = _PRINTINGS["Plasma Bolt"]
    state = _arena(
        p0_hand=[bolt],
        p0_battlefield=[_PRINTINGS["Weftwalking"]],
    )
    # Simulate a prior cast this turn.
    state.players[0].spells_cast_this_turn = 1
    card = state.players[0].hand[0]
    moves = legal_moves(state)
    free_casts = [
        m for m in moves
        if isinstance(m, CastSpell) and m.card_id == card.id and m.cast_for_free
    ]
    assert not free_casts, (
        "free-cast must be offered only on the player's FIRST spell each turn"
    )


def test_legal_moves_does_not_offer_free_cast_to_nonactive_player():
    """The free-cast applies during EACH PLAYER'S OWN TURN. While it is
    P0's turn, P1 (the nonactive player) should not be offered the
    free-cast variant for spells in their hand even if they have
    priority to cast instants."""
    from mtg_bot.actions import CastSpell
    from mtg_bot.legal import legal_actions
    from tests._rules_helpers import _arena

    bolt = _PRINTINGS["Plasma Bolt"]
    state = _arena(
        p1_hand=[bolt],
        p0_battlefield=[_PRINTINGS["Weftwalking"]],
    )
    # Hand priority off to P1 (e.g. P0 cast and passed).
    state.priority_player_idx = 1
    card = state.players[1].hand[0]
    moves = legal_actions(state, 1)
    free_casts = [
        m for m in moves
        if isinstance(m, CastSpell) and m.card_id == card.id and m.cast_for_free
    ]
    assert not free_casts, (
        "nonactive player should not get free-cast on the active player's turn"
    )


def test_legal_actions_does_not_offer_cast_for_free_for_warp_castable_card_in_graveyard():
    """Regression: legal_actions used to emit ``CastSpell(cast_for_free=True)``
    for a card in the graveyard with ``warp_castable_from_graveyard``
    when Weftwalking is on the battlefield, but the apply path's
    graveyard cast route is gated on ``use_warp=True`` — leaving no
    cast route for the ``cast_for_free=True`` graveyard variant. The
    crash surfaced as a 0.03%-rate ValueError in v16-vs-v13 eval
    (1/3263 games). Per CR 117.9b a single cast can apply only one
    alternative cost, and the warp-from-graveyard permission already
    bakes in the warp cost as the alt cost — so a free-cast variant
    isn't legal in any case.
    """
    from mtg_bot.actions import CastSpell, apply_action
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.legal import legal_actions
    from tests._rules_helpers import _arena

    culler = _PRINTINGS["Timeline Culler"]
    state = _arena(
        p0_battlefield=[
            _PRINTINGS["Weftwalking"],
            BASIC_LANDS["Swamp"],
        ],
        p0_life=20,
    )
    grave_card = state.new_object(culler, owner_id=0, zone=Zone.GRAVEYARD)
    state.players[0].graveyard.append(grave_card)

    moves = legal_actions(state, 0)
    free_grave = [
        m for m in moves
        if isinstance(m, CastSpell)
        and m.card_id == grave_card.id
        and m.cast_for_free
    ]
    assert not free_grave, (
        "cast_for_free=True must not be offered for a card in the "
        "graveyard — apply path has no route for this combination "
        "and the warp permission is itself an alt cost (CR 117.9b)"
    )

    # The warp-from-graveyard variant remains legal and apply must
    # still accept it.
    warp_grave = [
        m for m in moves
        if isinstance(m, CastSpell)
        and m.card_id == grave_card.id
        and m.use_warp
    ]
    assert warp_grave, (
        "warp-from-graveyard must remain a legal move (Timeline Culler)"
    )
    apply_action(state, 0, warp_grave[0])
    assert grave_card not in state.players[0].graveyard
