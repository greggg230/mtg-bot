"""Board-only opponent model (``opponent_hidden_model="board_only"``).

Pins:

* the hidden-card placeholder is inert (no types, no costs, never a
  legal cast / land play, no sorcery-speed capability);
* ``determinize(..., opponent_hidden_model="board_only")`` turns every
  opponent hand / library card the viewer can't identify into the
  placeholder, keeps revealed opponent cards, the viewer's hand and
  every public zone untouched, still samples the viewer's OWN library,
  preserves zone counts, and never leaks into the source state;
* ``"sampled"`` is the unchanged default (``make_determinize_fn``
  returns the very same function);
* the engine survives a whole game in a board-only world (draws,
  cleanup discards, graveyard placeholders) and the tokenizer maps
  placeholders to the redacted printing id from BOTH perspectives;
* a search whose opponent holds a real removal spell: the sampled
  tree contains the opponent casting it, the board-only tree contains
  no opponent cast at all.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from mtg_bot.actions import CastSpell, PassPriority, PlayLand
from mtg_bot.agents import RandomAgent
from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
from mtg_bot.cards.hidden import (
    HIDDEN_CARD_NAME, HIDDEN_CARD_PRINTING, is_hidden_placeholder,
)
from mtg_bot.cards.source import get_active_source
from mtg_bot.game import Zone, reveal_to
from mtg_bot.legal import legal_actions
from mtg_bot.search import MtgGameStateAdapter
from mtg_bot.search._player_capabilities import (
    has_player_capability_to_act,
)
from mtg_bot.search.determinize import (
    OPPONENT_HIDDEN_MODELS, determinize, make_determinize_fn,
)
from mtg_bot.simulate import setup_game
from mtg_bot.training import state_tokenizer as tok
from mtg_bot.training.structured_model import (
    EMB_FIELD_PRINTING, SCALAR_FIELD_PUBLIC_MASK,
    TOKEN_TYPE_GRAVEYARD_OPPONENT, TOKEN_TYPE_GRAVEYARD_VIEWER,
    TOKEN_TYPE_HAND_OPPONENT, TOKEN_TYPE_HAND_VIEWER,
)
from tests._test_decks import simple_test_decklists


def _game(seed: int = 7):
    return setup_game(
        [RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=seed,
    )


# ---------------------------------------------------------------------
# Placeholder printing
# ---------------------------------------------------------------------


def test_placeholder_is_inert():
    p = HIDDEN_CARD_PRINTING
    assert p.name == HIDDEN_CARD_NAME
    assert is_hidden_placeholder(p)
    assert not is_hidden_placeholder(None)
    assert not is_hidden_placeholder(BASIC_LANDS["Forest"])
    for prop in (
        "is_land", "is_creature", "is_instant", "is_sorcery",
        "is_permanent", "is_artifact", "is_enchantment", "is_room",
        "is_aura", "is_planeswalker",
    ):
        assert getattr(p, prop) is False, prop
    assert p.keywords == frozenset()
    assert p.spell_effect is None
    assert p.mana_cost.total == 0 and p.mana_cost.colors == frozenset()
    assert p.warp_cost is None and p.kicker_cost is None
    assert p.impending_cost is None
    assert p.triggered_abilities == () and p.activated_abilities == ()


def test_placeholder_name_never_in_card_pool_or_vocab():
    assert HIDDEN_CARD_NAME not in get_active_source().get_printings()
    assert tok.printing_id(HIDDEN_CARD_PRINTING) == tok.PRINTING_NA


# ---------------------------------------------------------------------
# determinize(board_only)
# ---------------------------------------------------------------------


def test_board_only_blanks_unknown_opponent_cards_only():
    state = _game()
    viewer, opp = state.players[0], state.players[1]
    revealed = opp.library[3]
    reveal_to(revealed, viewer.id)
    real_revealed = revealed.printing
    own_hand = [c.printing for c in viewer.hand]
    own_lib = sorted(c.printing.name for c in viewer.library)
    counts = (len(opp.hand), len(opp.library), len(viewer.library))

    out = determinize(
        state, viewer.id, random.Random(1),
        opponent_hidden_model="board_only",
    )
    v2, o2 = out.players[0], out.players[1]
    assert (len(o2.hand), len(o2.library), len(v2.library)) == counts
    for card in o2.hand:
        assert is_hidden_placeholder(card.printing)
    for card in o2.library:
        if card.id == revealed.id:
            assert card.printing is real_revealed
        else:
            assert is_hidden_placeholder(card.printing)
    # Viewer's hand untouched, own library still a permutation of
    # the same multiset (normal sampling).
    assert [c.printing for c in v2.hand] == own_hand
    assert sorted(c.printing.name for c in v2.library) == own_lib
    # Source state untouched (clone, not in place).
    assert not any(is_hidden_placeholder(c.printing) for c in opp.hand)
    assert not any(is_hidden_placeholder(c.printing) for c in opp.library)


def test_board_only_in_place_on_adapter_clone_does_not_leak():
    state = _game(seed=11)
    adapter = MtgGameStateAdapter(state)
    sub = adapter.clone()
    make_determinize_fn("board_only")(
        sub.engine_state, 0, random.Random(3), in_place=True,
    )
    assert all(
        is_hidden_placeholder(c.printing)
        for c in sub.engine_state.players[1].library
    )
    # CoW share-group fork: the source library objects are pristine.
    assert not any(
        is_hidden_placeholder(c.printing)
        for c in adapter.engine_state.players[1].library
    )


def test_board_only_single_hidden_card_is_still_blanked():
    state = _game()
    opp = state.players[1]
    opp.hand = opp.hand[:1]
    opp.library = []
    out = determinize(state, 0, random.Random(0), opponent_hidden_model="board_only")
    assert is_hidden_placeholder(out.players[1].hand[0].printing)


def test_sampled_is_default_and_bit_identical():
    assert OPPONENT_HIDDEN_MODELS == ("sampled", "board_only")
    assert make_determinize_fn("sampled") is determinize
    s1 = determinize(_game(), 0, random.Random(5))
    s2 = determinize(
        _game(), 0, random.Random(5), opponent_hidden_model="sampled",
    )
    for p1, p2 in zip(s1.players, s2.players):
        assert [c.printing.name for c in p1.hand + list(p1.library)] == [
            c.printing.name for c in p2.hand + list(p2.library)
        ]


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="opponent_hidden_model"):
        determinize(_game(), 0, opponent_hidden_model="nope")
    with pytest.raises(ValueError, match="opponent_hidden_model"):
        make_determinize_fn("nope")


# ---------------------------------------------------------------------
# Engine + capability behaviour with placeholders
# ---------------------------------------------------------------------


def _main_phase_adapter(seed: int = 3):
    """Adapter at P0's first main-phase priority with real cards."""
    adapter = MtgGameStateAdapter(_game(seed))
    for _ in range(200):
        eng = adapter.engine_state
        if (
            eng.step.value == "precombat_main"
            and eng.priority_player_idx == eng.active_player_idx
        ):
            return adapter
        adapter.apply(PassPriority())
    raise AssertionError("never reached a main phase")


def test_placeholders_are_never_castable_or_playable():
    adapter = _main_phase_adapter()
    eng = adapter.engine_state
    active = eng.players[eng.active_player_idx]
    # Give the active player plenty of mana and only placeholders.
    for _ in range(6):
        land = eng.new_object(BASIC_LANDS["Forest"], active.id, Zone.BATTLEFIELD)
        land.controller_id = active.id
        eng.battlefield.append(land)
    for card in active.hand:
        card.printing = HIDDEN_CARD_PRINTING
    acts = legal_actions(eng, active.id)
    assert not any(isinstance(a, (CastSpell, PlayLand)) for a in acts)
    # No sorcery-speed capability from a placeholder-only hand (the
    # untapped Forests have only mana abilities).
    eng.battlefield = [
        o for o in eng.battlefield
        if o.controller_id != active.id or o.printing.is_land
    ]
    assert has_player_capability_to_act(
        eng, active.id, is_active_player=True,
    ) is False


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_whole_game_in_a_board_only_world(seed):
    """Determinize at game start from P0's view, then play the world
    out with random decisions: P1 holds and draws only placeholders,
    discards them at cleanup, and the engine + tokenizer cope from both
    perspectives until the game ends."""
    adapter = MtgGameStateAdapter(_game(seed))
    make_determinize_fn("board_only")(
        adapter.engine_state, 0, random.Random(seed), in_place=True,
    )
    rng = random.Random(seed)
    saw_grave_placeholder = False
    for step in range(3000):
        if adapter.is_terminal():
            break
        eng = adapter.engine_state
        legal = adapter.legal_decisions()
        if adapter.active_player() == 1:
            assert not any(isinstance(d, (CastSpell, PlayLand)) for d in legal)
        if step % 25 == 0:
            for viewer in (0, 1):
                out = tok.tokenize_state(eng, viewer)
                ids = out["token_emb_ids"][:, EMB_FIELD_PRINTING]
                assert ids.min() >= 0
                assert ids.max() < tok.DEFAULT_N_PRINTINGS
        if any(is_hidden_placeholder(c.printing) for c in eng.players[1].graveyard):
            saw_grave_placeholder = True
        adapter.apply(rng.choice(legal))
    assert adapter.is_terminal()
    assert saw_grave_placeholder or adapter.winner() is not None


def test_tokenizer_redacts_placeholders_from_both_perspectives():
    state = _game()
    opp = state.players[1]
    out_state = determinize(state, 0, random.Random(0), opponent_hidden_model="board_only")
    o = out_state.players[1]
    # Push one placeholder into the opponent's graveyard.
    card = o.hand.pop()
    card.zone = Zone.GRAVEYARD
    o.graveyard.append(card)
    del opp

    # Opponent's own perspective: their hand is "viewer" hand.
    t1 = tok.tokenize_state(out_state, 1)
    tt, emb, sc = t1["token_type"], t1["token_emb_ids"], t1["token_scalars"]
    hand_rows = np.where(tt == TOKEN_TYPE_HAND_VIEWER)[0]
    assert len(hand_rows) == len(o.hand) > 0
    assert (emb[hand_rows, EMB_FIELD_PRINTING] == tok.PRINTING_NA).all()
    assert (sc[hand_rows, SCALAR_FIELD_PUBLIC_MASK] == 0.0).all()
    grave_rows = np.where(tt == TOKEN_TYPE_GRAVEYARD_VIEWER)[0]
    assert (emb[grave_rows, EMB_FIELD_PRINTING] == tok.PRINTING_NA).sum() >= 1

    # Root (viewer 0) perspective: opp hand redacted exactly as always.
    t0 = tok.tokenize_state(out_state, 0)
    rows = np.where(t0["token_type"] == TOKEN_TYPE_HAND_OPPONENT)[0]
    assert len(rows) == len(o.hand)
    assert (t0["token_emb_ids"][rows, EMB_FIELD_PRINTING] == tok.PRINTING_NA).all()
    g = np.where(t0["token_type"] == TOKEN_TYPE_GRAVEYARD_OPPONENT)[0]
    assert (t0["token_emb_ids"][g, EMB_FIELD_PRINTING] == tok.PRINTING_NA).sum() >= 1


def test_tokenizer_real_cards_unchanged():
    """A state with no placeholders tokenizes exactly as before (the
    placeholder branch only fires on the placeholder name)."""
    state = _game()
    out = tok.tokenize_state(state, 0)
    rows = np.where(out["token_type"] == TOKEN_TYPE_HAND_VIEWER)[0]
    assert (out["token_emb_ids"][rows, EMB_FIELD_PRINTING] != tok.PRINTING_NA).all()
    assert (out["token_scalars"][rows, SCALAR_FIELD_PUBLIC_MASK] == 1.0).all()


# ---------------------------------------------------------------------
# Search: opponent removal visible only under "sampled"
# ---------------------------------------------------------------------


def _removal_position():
    """P0 (searcher) at its main phase with a creature in play; P1
    holds Bombard (instant, 4 damage to target creature) with three
    untapped Mountains and a library made only of Bombards, so every
    sampled world gives P1 the removal in hand."""
    adapter = _main_phase_adapter(seed=5)
    eng = adapter.engine_state
    p0, p1 = eng.players[0], eng.players[1]
    assert eng.players[eng.active_player_idx].id == p0.id
    bombard = get_active_source().get_printings()["Bombard"]
    eng.battlefield = []
    p0.hand = []
    p1.hand = []
    bear = eng.new_object(SYNTHETIC_CREATURES["Wild Bear"], p0.id, Zone.BATTLEFIELD)
    bear.controller_id = p0.id
    bear.summoning_sick = False
    eng.battlefield.append(bear)
    # Give P0 a real choice (cast a creature or pass) so the search
    # builds a tree instead of taking the forced-move shortcut.
    p0.hand.append(eng.new_object(
        SYNTHETIC_CREATURES["Trained Soldier"], p0.id, Zone.HAND,
    ))
    for _ in range(2):
        plains = eng.new_object(BASIC_LANDS["Plains"], p0.id, Zone.BATTLEFIELD)
        plains.controller_id = p0.id
        eng.battlefield.append(plains)
    for _ in range(3):
        m = eng.new_object(BASIC_LANDS["Mountain"], p1.id, Zone.BATTLEFIELD)
        m.controller_id = p1.id
        eng.battlefield.append(m)
    p1.hand.append(eng.new_object(bombard, p1.id, Zone.HAND))
    for card in p1.library:
        card.printing = bombard
        card.original_printing = None
    adapter._invalidate_legal_actions_cache()
    return adapter


def _opp_casts_in_tree(model: str) -> int:
    from mtg_bot.training.config import HeuristicConfig, PimcConfig

    cfg = PimcConfig(
        samples=1, inner_nodes=150, heuristic=HeuristicConfig(),
        opponent_hidden_model=model,
    )
    pimc = cfg.build(seed=0)
    found = {"casts": 0, "nodes": 0}

    def observer(root, root_player):
        stack = [root]
        while stack:
            n = stack.pop()
            for c in n.children:
                found["nodes"] += 1
                if (
                    n.state.active_player() != root_player
                    and isinstance(c.decision, CastSpell)
                ):
                    found["casts"] += 1
                stack.append(c)

    inner = pimc.inner_factory

    def wrapped():
        m = inner()
        m.tree_observer = observer
        return m

    pimc.inner_factory = wrapped
    pimc.choose(_removal_position(), 0)
    assert found["nodes"] > 0
    return found["casts"]


def test_sampled_search_sees_opponent_removal_board_only_does_not():
    assert _opp_casts_in_tree("sampled") > 0
    assert _opp_casts_in_tree("board_only") == 0
