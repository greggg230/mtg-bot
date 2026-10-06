"""``mtg_bot.search.horizon_probe`` + the opt-in ``MCTS.tree_observer``
hook it relies on."""

from __future__ import annotations

import random

import pytest

from mtg_bot.actions import CastSpell, PassPriority, PlayLand
from mtg_bot.agents import RandomAgent
from mtg_bot.search import MCTS, MtgGameStateAdapter
from mtg_bot.search.budget import Budget
from mtg_bot.search.combat_decisions import AttackerDecision, BlockerDecision
from mtg_bot.search.heuristic_evaluator import MtgHeuristicEvaluator
from mtg_bot.search import horizon_probe as hp
from mtg_bot.simulate import setup_game
from mtg_bot.turn import Step
from tests._test_decks import simple_test_decklists


def _mcts(nodes: int = 120) -> MCTS:
    return MCTS(
        evaluator=MtgHeuristicEvaluator(),
        budget_factory=lambda: Budget(max_nodes=nodes),
        rng=random.Random(0),
    )


def _decision_point(seed: int = 4, min_options: int = 2, want=None):
    """Play random moves until player 0 has a real choice (and, when
    ``want`` is given, ``want(legal)`` holds)."""
    adapter = MtgGameStateAdapter(setup_game(
        [RandomAgent(), RandomAgent()], decks=simple_test_decklists(),
        seed=seed,
    ))
    rng = random.Random(seed)
    for _ in range(3000):
        if adapter.is_terminal():
            break
        legal = adapter.legal_decisions()
        if (
            adapter.active_player() == 0 and len(legal) >= min_options
            and (want is None or want(legal))
        ):
            return adapter
        adapter.apply(rng.choice(legal))
    raise AssertionError("no decision point found")


def _tree(seed: int = 4, nodes: int = 120, want=None):
    adapter = _decision_point(seed, want=want)
    got = []
    m = _mcts(nodes)
    m.tree_observer = lambda root, pid: got.append((root, pid))
    m.choose(adapter, 0)
    assert len(got) == 1
    return got[0]


# ---------------------------------------------------------------------
# MCTS hook
# ---------------------------------------------------------------------


def test_tree_observer_defaults_to_none():
    assert _mcts().tree_observer is None


@pytest.mark.parametrize("entry", [
    "choose_with_visits", "choose_with_full_stats", "choose_with_subtree",
])
def test_tree_observer_called_once_per_search_entry_point(entry):
    adapter = _decision_point()
    calls = []
    m = _mcts(30)
    m.tree_observer = lambda root, pid: calls.append((root, pid))
    getattr(m, entry)(adapter, 0)
    assert len(calls) == 1
    root, pid = calls[0]
    assert pid == 0 and root.parent is None and root.children


def test_tree_observer_not_called_on_forced_move():
    adapter = _decision_point()
    while len(adapter.legal_decisions()) != 1:
        adapter.apply(adapter.legal_decisions()[0])
    calls = []
    m = _mcts(30)
    m.tree_observer = lambda root, pid: calls.append(1)
    m.choose_with_visits(adapter, adapter.active_player())
    assert calls == []


def test_observer_does_not_change_the_search():
    adapter = _decision_point()
    a = _mcts(60).choose_with_visits(adapter, 0)
    m = _mcts(60)
    m.tree_observer = lambda root, pid: hp.summarize_tree(root, pid)
    assert m.choose_with_visits(adapter, 0) == a


# ---------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------


def test_decision_kind_and_object_ids():
    assert hp.decision_kind(PassPriority()) == "pass"
    assert hp.decision_kind(CastSpell(card_id=3)) == "cast"
    assert hp.decision_kind(PlayLand(card_id=4)) == "land"
    assert hp.decision_kind(AttackerDecision(attacker_id=5, attack=True)) == "attack"
    assert hp.decision_kind(AttackerDecision(attacker_id=5, attack=False)) == "no_attack"
    assert hp.decision_kind(BlockerDecision(blocker_id=6, attacker_id=5)) == "block"
    assert hp.decision_kind(BlockerDecision(blocker_id=6, attacker_id=None)) == "no_block"
    from mtg_bot.actions import ActivateAbility
    from mtg_bot.search.resolution_decisions import ResolutionTargetDecision
    assert hp.decision_kind(ActivateAbility(source_id=1, ability_index=0)) == "activate"
    assert hp.decision_kind(ResolutionTargetDecision(target=("player", 1))) == "resolution"
    assert hp.decision_kind(object()) == "object"
    assert hp.decision_object_ids(CastSpell(card_id=3)) == (3,)
    assert hp.decision_object_ids(BlockerDecision(blocker_id=6, attacker_id=5)) == (5, 6)
    assert hp.decision_object_ids(BlockerDecision(blocker_id=6, attacker_id=None)) == (6,)
    assert hp.decision_object_ids(PassPriority()) == ()


@pytest.mark.parametrize("args,expected", [
    ((5, Step.UPKEEP, 5, Step.UPKEEP), "same_step"),
    ((5, Step.UPKEEP, 5, Step.DRAW), "later_step"),
    ((5, Step.DECLARE_ATTACKERS, 5, Step.COMBAT_DAMAGE), "later_step"),
    ((5, Step.PRECOMBAT_MAIN, 5, Step.DECLARE_ATTACKERS), "later_phase"),
    ((5, Step.END, 6, Step.UPKEEP), "next_turn"),
    ((5, Step.END, 7, Step.UPKEEP), "two_plus_turns"),
    ((5, None, 5, Step.UPKEEP), "same_step"),
])
def test_reach_category(args, expected):
    assert hp.reach_category(*args) == expected


# ---------------------------------------------------------------------
# summarize_tree / render_tree on a real search tree
# ---------------------------------------------------------------------


def test_summarize_tree_is_internally_consistent():
    root, pid = _tree()
    s = hp.summarize_tree(root, pid)
    n = s["n_nodes"]
    assert 0 < n <= 120
    assert sum(s["reach_nodes"].values()) == n
    assert sum(s["edge_kinds"].values()) == n
    assert 0 <= s["nodes_forced"] <= n
    assert 0 <= s["nodes_opp_point"] <= n
    assert s["nodes_imagined_edge"] == 0 and s["nodes_under_imagined"] == 0
    assert s["max_choice_depth"] <= s["max_depth"]
    assert s["pv_choice_len"] <= s["pv_len"] <= s["max_depth"]
    assert s["max_reach"] in hp.REACH_CATEGORIES
    assert len(s["root_children"]) == s["root_options"] or s["root_options"] >= len(s["root_children"])
    assert sum(c["subtree_nodes"] for c in s["root_children"]) == n
    chosen = s["root_children"][s["chosen_index"]]
    assert chosen["visits"] == max(c["visits"] for c in s["root_children"])
    for c in s["root_children"]:
        assert c["resolution"] in ("reached", "not_reached", "immediate")


def test_imagined_card_accounting_marks_opponent_hidden_cards():
    """Mark every opponent card id as 'hidden': each opponent cast /
    land in the tree is an imagined-card edge and everything below it
    counts as under-imagined."""
    root, pid = _tree(seed=6, nodes=200)
    eng = root.state.engine_state
    opp = next(p for p in eng.players if p.id != pid)
    hidden = {c.id for c in list(opp.hand) + list(opp.library)}
    s = hp.summarize_tree(root, pid, hidden)
    direct = 0
    stack = [root]
    while stack:
        node = stack.pop()
        for c in node.children:
            if any(i in hidden for i in hp.decision_object_ids(c.decision)):
                direct += 1
            stack.append(c)
    assert s["nodes_imagined_edge"] == direct
    assert s["nodes_under_imagined"] >= direct


def test_cast_resolution_detected_through_stack_height():
    """Find a tree whose root has a cast child: its resolution is
    'reached' once a descendant's stack is back to the root height."""
    def has_cast(legal):
        return any(isinstance(d, CastSpell) for d in legal)

    for seed in range(1, 30):
        root, pid = _tree(seed=seed, nodes=150, want=has_cast)
        s = hp.summarize_tree(root, pid)
        casts = [c for c in s["root_children"] if c["kind"] == "cast"]
        if casts and any(c["resolution"] == "reached" for c in casts):
            return
    raise AssertionError("no root cast resolved in 29 positions")


def test_render_tree_is_readable():
    root, pid = _tree()
    lines = hp.render_tree(root, pid, max_depth=3, top_k=2)
    assert lines[0].startswith("root: T")
    assert len(lines) > len(root.children)
    body = "\n".join(lines)
    assert "N=" in body and "Q=" in body and "P=" in body
    uncollapsed = hp.render_tree(root, pid, max_depth=2, collapse_forced=False)
    assert all("forced" not in ln for ln in uncollapsed)


def test_label_decision_names_cards_and_sides():
    adapter = _decision_point()
    eng = adapter.engine_state
    legal = adapter.legal_decisions()
    for d in legal:
        child = adapter.clone()
        child.apply(d)
        label = hp.label_decision(d, adapter, child, 0)
        if isinstance(d, (CastSpell, PlayLand)):
            name = hp.find_object(eng, d.card_id).printing.name
            assert name in label and "(own)" in label
        if isinstance(d, PassPriority):
            assert label == "Pass"
    assert hp.find_object(eng, 10**9) is None
    assert hp._obj_label(eng, None, 0) == "-"
    assert hp._obj_label(eng, 10**9, 0) == f"#{10**9}"
    assert hp._target_label(eng, ("player", 0), 0) == "own face"
    assert hp._target_label(eng, ("player", 1), 0) == "opp face"


def test_label_decision_covers_every_kind():
    """Attack / block / land / activate / cast-with-target labels on a
    hand-built board."""
    from mtg_bot.actions import ActivateAbility
    from mtg_bot.cards import BASIC_LANDS, SYNTHETIC_CREATURES
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.game import StackItem, Zone

    adapter = _decision_point()
    eng = adapter.engine_state
    own = eng.new_object(SYNTHETIC_CREATURES["Wild Bear"], 0, Zone.BATTLEFIELD)
    own.controller_id = 0
    opp = eng.new_object(SYNTHETIC_CREATURES["Wild Bear"], 1, Zone.BATTLEFIELD)
    opp.controller_id = 1
    eng.battlefield.extend([own, opp])
    land = eng.new_object(BASIC_LANDS["Forest"], 0, Zone.HAND)
    eng.players[0].hand.append(land)
    assert hp.label_decision(
        AttackerDecision(attacker_id=own.id, attack=True), adapter, None, 0,
    ).startswith("Attack with Wild Bear")
    assert hp.label_decision(
        AttackerDecision(attacker_id=own.id, attack=False), adapter, None, 0,
    ).startswith("Hold back")
    assert hp.label_decision(
        BlockerDecision(blocker_id=opp.id, attacker_id=own.id), adapter, None, 1,
    ).startswith("Block Wild Bear")
    assert hp.label_decision(
        BlockerDecision(blocker_id=opp.id, attacker_id=None), adapter, None, 1,
    ).startswith("No block")
    assert hp.label_decision(PlayLand(card_id=land.id), adapter, None, 0).startswith(
        "Play land Forest"
    )
    assert hp.label_decision(
        ActivateAbility(source_id=own.id, ability_index=0), adapter, None, 0,
    ) == f"Activate Wild Bear#{own.id}(own)[0]"
    # Cast whose child state has a new targeted stack item.
    bombard = get_active_source().get_printings()["Bombard"]
    card = eng.new_object(bombard, 0, Zone.HAND)
    eng.players[0].hand.append(card)
    child = adapter.clone()
    src = child.engine_state.new_object(bombard, 0, Zone.STACK)
    child.engine_state.stack.append(StackItem(
        source=src, controller_id=0, targets=(("creature", own.id),),
    ))
    lab = hp.label_decision(
        CastSpell(card_id=card.id, kicked=True), adapter, child, 0,
    )
    assert lab.startswith("Cast Bombard") and "[kicked]" in lab
    assert lab.endswith(f"-> Wild Bear#{own.id}(own)")


def test_past_combat_damage_and_terminal_flags():
    root, pid = _tree()
    turn = root.state.engine_state.turn_number
    assert hp._past_combat_damage(turn - 1, root.state) is True
    eng = root.state.engine_state
    saved = eng.step
    eng.step = Step.UPKEEP
    assert hp._past_combat_damage(turn, root.state) is False
    eng.step = Step.END
    assert hp._past_combat_damage(turn, root.state) is True
    eng.step = saved
