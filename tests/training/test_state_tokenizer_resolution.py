"""History tokens for resolution answers: distinct answers give
distinct tokens (``docs/RESOLUTION_DECISIONS.md`` §6)."""

from __future__ import annotations

from mtg_bot.agents import RandomAgent
from mtg_bot.cards.source import get_active_source
from mtg_bot.game import StackItem, Zone
from mtg_bot.search.resolution_decisions import (
    ResolutionDevourLandsDecision, ResolutionModeDecision,
    ResolutionModesDecision, ResolutionShockLifeDecision,
    ResolutionTargetDecision,
)
from mtg_bot.simulate import setup_game
from mtg_bot.training import state_tokenizer as st
from mtg_bot.training.structured_model import (
    EMB_FIELD_ACTION_KIND, EMB_FIELD_PRINTING, EMB_FIELD_TARGET_KIND,
    EMB_FIELD_TARGET_PRINTING, EMB_FIELD_VARIANT, SCALAR_FIELD_MANA_COST_PAID,
)
from tests._test_decks import simple_test_decklists


def _real_creature():
    """A creature printing that is in the card vocabulary (synthetic
    test creatures tokenise to the NA printing id)."""
    printings = get_active_source().get_printings()
    return printings[next(
        n for n in sorted(printings) if printings[n].is_creature
    )]


def _state():
    s = setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=4,
    )
    s.battlefield.clear()
    return s


def _tok(action, state, viewer=0, actor=0):
    _tt, emb, sc = st._make_action_token(
        action=action, state_at_action=state, action_turn=1, current_turn=1,
        viewer_id=viewer, actor_player_id=actor, state_step=state.step,
        n_printings_cap=320,
    )
    return emb, sc


def test_player_target_never_looks_up_an_object_with_that_id():
    s = _state()
    # Plant an object whose id collides with player id 1.
    c = s.new_object(_real_creature(), 0, Zone.BATTLEFIELD)
    c.id = 1
    s.battlefield.append(c)
    emb, _ = _tok(ResolutionTargetDecision(("player", 1)), s)
    assert emb[EMB_FIELD_TARGET_PRINTING] == 0
    assert emb[EMB_FIELD_TARGET_KIND] == st.TARGET_KIND_PLAYER
    emb2, _ = _tok(ResolutionTargetDecision(("creature", 1)), s)
    assert emb2[EMB_FIELD_TARGET_PRINTING] != 0
    assert emb2[EMB_FIELD_TARGET_KIND] == st.TARGET_KIND_PERMANENT


def test_stack_graveyard_exile_and_player_fallback_targets():
    s = _state()
    src = s.new_object(_real_creature(), 1, Zone.STACK)
    s.stack.append(StackItem(source=src, controller_id=1))
    emb, _ = _tok(ResolutionTargetDecision(("stack", 0)), s)
    assert emb[EMB_FIELD_TARGET_KIND] == st.TARGET_KIND_STACK_ITEM
    assert emb[EMB_FIELD_TARGET_PRINTING] != 0
    g = s.new_object(_real_creature(), 1, Zone.GRAVEYARD)
    s.players[1].graveyard.append(g)
    x = s.new_object(_real_creature(), 0, Zone.EXILE)
    s.players[0].exile.append(x)
    eg, _ = _tok(ResolutionTargetDecision(("graveyard_creature", g.id)), s)
    ex, _ = _tok(ResolutionTargetDecision(("exile_card", x.id)), s)
    assert eg[EMB_FIELD_TARGET_PRINTING] != 0
    assert ex[EMB_FIELD_TARGET_KIND] == st.TARGET_KIND_EXILE_CARD
    ep, _ = _tok(ResolutionTargetDecision(("any_damage_target", 0)), s)
    assert ep[EMB_FIELD_TARGET_KIND] == st.TARGET_KIND_PLAYER
    none_emb, _ = _tok(ResolutionTargetDecision(None), s)
    assert none_emb[EMB_FIELD_TARGET_KIND] == 0


def test_mode_shock_and_devour_answers_are_distinct():
    s = _state()
    m0 = _tok(ResolutionModeDecision(0), s)[1][SCALAR_FIELD_MANA_COST_PAID]
    m1 = _tok(ResolutionModeDecision(1), s)[1][SCALAR_FIELD_MANA_COST_PAID]
    assert m0 != m1
    both_emb, both_sc = _tok(ResolutionModesDecision((0, 1)), s)
    one_emb, one_sc = _tok(ResolutionModesDecision((1,)), s)
    assert both_emb[EMB_FIELD_ACTION_KIND] == st.ACTION_KIND_VOCAB[
        "ResolutionModeDecision"
    ]
    assert both_sc[SCALAR_FIELD_MANA_COST_PAID] != one_sc[SCALAR_FIELD_MANA_COST_PAID]
    pay = _tok(ResolutionShockLifeDecision(True), s)[0][EMB_FIELD_VARIANT]
    dont = _tok(ResolutionShockLifeDecision(False), s)[0][EMB_FIELD_VARIANT]
    assert pay != dont
    none = _tok(ResolutionDevourLandsDecision(()), s)[1][SCALAR_FIELD_MANA_COST_PAID]
    two = _tok(ResolutionDevourLandsDecision((3, 4)), s)[1][SCALAR_FIELD_MANA_COST_PAID]
    assert none != two
    assert _tok(ResolutionModeDecision(0), s)[0][EMB_FIELD_PRINTING] == 0
