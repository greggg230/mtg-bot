"""Search-horizon probe: post-hoc statistics over one MCTS tree.

Opt-in diagnostics for "how far does the search actually see?". The
hook is ``MCTS.tree_observer`` (default ``None`` -> one attribute
check per search, nothing else): when set, every ``choose_*`` call
hands its finished root ``_Node`` and the root player to the observer
before the tree is dropped. :func:`summarize_tree` turns that tree
into plain-dict statistics; :func:`render_tree` prints a readable
dump. Nothing here runs during the search itself.

Definitions (also in ``docs/SEARCH_HORIZON.md``):

* **node** — every ``_Node`` below the root. Under the production
  ``FullEnumeration`` expansion each search iteration creates at most
  one node, so the node count is the budget actually turned into tree.
* **decision point** of a node — its *parent*: the state where the
  mover chose the decision that leads to the node. A node is "spent
  at" its parent's decision point.
* **forced** decision point — the parent had exactly one legal
  decision (``len(legal_decisions()) == 1``; nearly always a
  ``PassPriority`` the adapter short-circuited).
* **opponent** decision point — the parent's mover is not the root
  player.
* **imagined-card** decision — the decision references an object id
  that was a *hidden* opponent card at the real root (opponent hand /
  library card the viewer could not identify), i.e. an identity the
  determinizer sampled: a cast / land play of it, an activation /
  attack / block by the permanent it became.
* **game-time reach** of a node relative to the root, from the engine
  ``turn_number`` and ``step``: ``same_step`` / ``later_step``
  (same phase) / ``later_phase`` (same turn) / ``next_turn`` /
  ``two_plus_turns``. Phases: beginning (untap, upkeep, draw), main 1,
  combat (beginning of combat .. end of combat), main 2, ending.
* **resolution reached** for a root child: an attack / block decision
  is resolved once some node in that child's subtree sits at or past
  the root turn's combat-damage step (or a later turn, or the game
  ended); a decision that put an object on the stack (cast / non-mana
  activation) is resolved once some node in its subtree has a stack no
  taller than before the decision; a decision that did not grow the
  stack (land, mana ability, special action) resolved immediately.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Optional

from mtg_bot.turn import STEP_ORDER, Step

REACH_CATEGORIES: tuple[str, ...] = (
    "same_step", "later_step", "later_phase", "next_turn", "two_plus_turns",
)

_PHASE_OF_STEP: dict[Step, int] = {
    Step.UNTAP: 0, Step.UPKEEP: 0, Step.DRAW: 0,
    Step.PRECOMBAT_MAIN: 1,
    Step.BEGINNING_OF_COMBAT: 2, Step.DECLARE_ATTACKERS: 2,
    Step.DECLARE_BLOCKERS: 2, Step.COMBAT_DAMAGE: 2,
    Step.END_OF_COMBAT: 2,
    Step.POSTCOMBAT_MAIN: 3,
    Step.END: 4, Step.CLEANUP: 4,
}
_STEP_INDEX: dict[Step, int] = {s: i for i, s in enumerate(STEP_ORDER)}
_COMBAT_DAMAGE_INDEX = _STEP_INDEX[Step.COMBAT_DAMAGE]


# ---------------------------------------------------------------------
# Decision classification
# ---------------------------------------------------------------------


def decision_kind(decision: Any) -> str:
    """Coarse kind label for a search decision (engine action, factored
    combat decision, or resolution decision)."""
    name = type(decision).__name__
    if name == "PassPriority":
        return "pass"
    if name == "CastSpell":
        return "cast"
    if name == "PlayLand":
        return "land"
    if name in (
        "ActivateAbility", "ActivateStation", "UnlockRoomHalf",
        "TurnFaceUp",
    ):
        return "activate"
    if name == "AttackerDecision":
        return "attack" if decision.attack else "no_attack"
    if name == "BlockerDecision":
        return "block" if decision.attacker_id is not None else "no_block"
    if name.startswith("Resolution") and name.endswith("Decision"):
        return "resolution"
    return name


def decision_object_ids(decision: Any) -> tuple[int, ...]:
    """Object ids a decision references (the card cast / land played /
    permanent activated / creature attacking or blocking)."""
    ids: list[int] = []
    for attr in (
        "card_id", "source_id", "spacecraft_id", "crew_id",
        "attacker_id", "blocker_id",
    ):
        v = getattr(decision, attr, None)
        if isinstance(v, int) and not isinstance(v, bool):
            ids.append(v)
    return tuple(ids)


def reach_category(
    root_turn: int, root_step: Optional[Step], turn: int,
    step: Optional[Step],
) -> str:
    """Game-time distance of ``(turn, step)`` from ``(root_turn,
    root_step)`` as one of :data:`REACH_CATEGORIES`."""
    dt = turn - root_turn
    if dt >= 2:
        return "two_plus_turns"
    if dt == 1:
        return "next_turn"
    if dt < 0 or step is None or root_step is None:
        # Never happens in a real tree (time only moves forward);
        # classify defensively as no progress.
        return "same_step"
    if step == root_step:
        return "same_step"
    if _PHASE_OF_STEP[step] == _PHASE_OF_STEP[root_step]:
        return "later_step"
    return "later_phase"


def _engine(node_state: Any) -> Any:
    return getattr(node_state, "engine_state", node_state)


def _turn_step(node_state: Any) -> tuple[int, Optional[Step]]:
    eng = _engine(node_state)
    return int(getattr(eng, "turn_number", 0)), getattr(eng, "step", None)


def _stack_height(node_state: Any) -> int:
    return len(getattr(_engine(node_state), "stack", ()) or ())


def _past_combat_damage(
    root_turn: int, node_state: Any,
) -> bool:
    if node_state.is_terminal():
        return True
    turn, step = _turn_step(node_state)
    if turn > root_turn:
        return True
    return step is not None and _STEP_INDEX[step] >= _COMBAT_DAMAGE_INDEX


# ---------------------------------------------------------------------
# Tree summary
# ---------------------------------------------------------------------


def _n_options(node: Any, cache: dict[int, int]) -> int:
    key = id(node)
    n = cache.get(key)
    if n is None:
        n = 0 if node.state.is_terminal() else len(
            list(node.state.legal_decisions())
        )
        cache[key] = n
    return n


def summarize_tree(
    root: Any,
    root_player: int,
    hidden_opp_ids: Iterable[int] = (),
) -> dict[str, Any]:
    """Statistics for one finished MCTS tree (see module docstring for
    the definitions). ``root`` is an ``mcts._Node`` whose ``state`` is
    an ``MtgGameStateAdapter``; ``hidden_opp_ids`` are the object ids
    of the opponent's hidden cards at the real decision (pass an empty
    set to skip the imagined-card accounting)."""
    hidden = frozenset(hidden_opp_ids)
    root_turn, root_step = _turn_step(root.state)
    opt_cache: dict[int, int] = {}

    n_nodes = 0
    max_depth = 0
    max_choice_depth = 0
    depth_sum = 0
    reach_nodes: Counter = Counter()
    leaf_reach: Counter = Counter()
    nodes_forced = 0
    nodes_pass_edge = 0
    nodes_opp_point = 0
    nodes_imagined_edge = 0
    nodes_under_imagined = 0
    nodes_terminal = 0
    max_reach_rank = 0
    edge_kinds: Counter = Counter()

    # Iterative DFS: (node, depth, choice_depth, under_imagined).
    stack: list[tuple[Any, int, int, bool]] = [(root, 0, 0, False)]
    while stack:
        node, depth, choice_depth, under = stack.pop()
        if node is not root:
            parent = node.parent
            n_nodes += 1
            depth_sum += depth
            max_depth = max(max_depth, depth)
            max_choice_depth = max(max_choice_depth, choice_depth)
            turn, step = _turn_step(node.state)
            cat = reach_category(root_turn, root_step, turn, step)
            reach_nodes[cat] += 1
            max_reach_rank = max(
                max_reach_rank, REACH_CATEGORIES.index(cat),
            )
            if not node.children:
                leaf_reach[cat] += 1
            if node.state.is_terminal():
                nodes_terminal += 1
            n_opt = _n_options(parent, opt_cache)
            if n_opt == 1:
                nodes_forced += 1
            kind = decision_kind(node.decision)
            edge_kinds[kind] += 1
            if kind == "pass":
                nodes_pass_edge += 1
            if parent.state.active_player() != root_player:
                nodes_opp_point += 1
            if under:
                nodes_under_imagined += 1
        for child in node.children:
            child_imagined = bool(hidden) and any(
                oid in hidden for oid in decision_object_ids(child.decision)
            )
            if child_imagined:
                nodes_imagined_edge += 1
            is_choice = _n_options(node, opt_cache) > 1
            stack.append((
                child, depth + 1,
                choice_depth + (1 if is_choice else 0),
                under or child_imagined,
            ))

    # Principal variation: most-visited path from the root.
    pv: list[dict[str, Any]] = []
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.visits)
        turn, step = _turn_step(node.state)
        pv.append({
            "kind": decision_kind(node.decision),
            "mover_is_root": (
                node.parent.state.active_player() == root_player
            ),
            "visits": int(node.visits),
            "reach": reach_category(root_turn, root_step, turn, step),
        })

    root_children = [
        _root_child_stats(c, root_turn) for c in root.children
    ]
    chosen = (
        max(root.children, key=lambda c: c.visits)
        if root.children else None
    )
    return {
        "n_nodes": n_nodes,
        "root_visits": int(root.visits),
        "root_options": _n_options(root, opt_cache),
        "max_depth": max_depth,
        "max_choice_depth": max_choice_depth,
        "mean_depth": (depth_sum / n_nodes) if n_nodes else 0.0,
        "pv_len": len(pv),
        "pv_choice_len": _pv_choice_len(root, opt_cache),
        "pv_reach": pv[-1]["reach"] if pv else "same_step",
        "pv": pv,
        "max_reach": REACH_CATEGORIES[max_reach_rank] if n_nodes else "same_step",
        "reach_nodes": {c: int(reach_nodes.get(c, 0)) for c in REACH_CATEGORIES},
        "leaf_reach": {c: int(leaf_reach.get(c, 0)) for c in REACH_CATEGORIES},
        "nodes_forced": nodes_forced,
        "nodes_pass_edge": nodes_pass_edge,
        "nodes_opp_point": nodes_opp_point,
        "nodes_imagined_edge": nodes_imagined_edge,
        "nodes_under_imagined": nodes_under_imagined,
        "nodes_terminal": nodes_terminal,
        "edge_kinds": dict(edge_kinds),
        "root_children": root_children,
        "chosen_index": (
            root.children.index(chosen) if chosen is not None else None
        ),
    }


def _pv_choice_len(root: Any, opt_cache: dict[int, int]) -> int:
    """Number of real choices (decision points with >1 option) along
    the most-visited path."""
    n = 0
    node = root
    while node.children:
        if _n_options(node, opt_cache) > 1:
            n += 1
        node = max(node.children, key=lambda c: c.visits)
    return n


def _root_child_stats(child: Any, root_turn: int) -> dict[str, Any]:
    """Per-root-child stats incl. whether its subtree reaches the
    decision's resolution (module docstring)."""
    kind = decision_kind(child.decision)
    parent = child.parent
    h_pre = _stack_height(parent.state)
    h_post = _stack_height(child.state)
    combat = kind in ("attack", "no_attack", "block", "no_block")
    grew_stack = h_post > h_pre
    resolved = False
    subtree_nodes = 0
    max_rank = 0
    root_turn_, root_step = _turn_step(parent.state)
    stack = [child]
    while stack:
        n = stack.pop()
        subtree_nodes += 1
        turn, step = _turn_step(n.state)
        max_rank = max(max_rank, REACH_CATEGORIES.index(
            reach_category(root_turn_, root_step, turn, step),
        ))
        if not resolved:
            if combat:
                resolved = _past_combat_damage(root_turn, n.state)
            elif grew_stack:
                resolved = (
                    n.state.is_terminal()
                    or _stack_height(n.state) <= h_pre
                )
        stack.extend(n.children)
    if combat:
        resolution = "reached" if resolved else "not_reached"
    elif grew_stack:
        resolution = "reached" if resolved else "not_reached"
    else:
        resolution = "immediate"
    return {
        "kind": kind,
        "visits": int(child.visits),
        "q": child.mean_value() if child.visits > 0 else None,
        "prior": float(child.prior),
        "subtree_nodes": subtree_nodes,
        "max_reach": REACH_CATEGORIES[max_rank],
        "resolution": resolution,
    }


# ---------------------------------------------------------------------
# Readable dumps
# ---------------------------------------------------------------------


def find_object(eng: Any, obj_id: int) -> Optional[Any]:
    """Locate a ``GameObject`` by id in any zone of ``eng`` (battlefield,
    stack sources, every player's hand / graveyard / exile / library)."""
    for o in getattr(eng, "battlefield", ()):
        if o.id == obj_id:
            return o
    for item in getattr(eng, "stack", ()):
        src = getattr(item, "source", None)
        if src is not None and src.id == obj_id:
            return src
    for p in getattr(eng, "players", ()):
        for zone in ("hand", "graveyard", "exile", "library"):
            for o in getattr(p, zone, ()):
                if o.id == obj_id:
                    return o
    return None


def _obj_label(eng: Any, obj_id: Optional[int], viewer: int) -> str:
    if obj_id is None:
        return "-"
    o = find_object(eng, obj_id)
    if o is None:
        return f"#{obj_id}"
    side = "own" if o.controller_id == viewer else "opp"
    return f"{o.printing.name}#{obj_id}({side})"


def _target_label(eng: Any, target: tuple, viewer: int) -> str:
    """Label one ``(kind, id)`` engine target. ``("stack", i)`` is a
    stack INDEX (counterspells), every other non-player kind carries an
    object id."""
    kind, tid = target
    if kind == "player":
        return "own face" if tid == viewer else "opp face"
    if kind == "stack":
        stack = getattr(eng, "stack", ())
        if isinstance(tid, int) and 0 <= tid < len(stack):
            item = stack[tid]
            side = "own" if item.controller_id == viewer else "opp"
            return f"spell {item.source.printing.name}({side})"
        return f"stack[{tid}]"
    return _obj_label(eng, tid, viewer)


def label_decision(
    decision: Any, parent_state: Any, child_state: Any, viewer: int,
) -> str:
    """Human-readable label for ``decision`` taken at ``parent_state``
    (card names, own/opp side relative to ``viewer``). For a cast /
    activation that put an object on the stack, the auto-chosen
    targets are read off the new stack top in ``child_state``."""
    eng = _engine(parent_state)
    kind = decision_kind(decision)
    if kind == "pass":
        return "Pass"
    if kind == "attack":
        return f"Attack with {_obj_label(eng, decision.attacker_id, viewer)}"
    if kind == "no_attack":
        return f"Hold back {_obj_label(eng, decision.attacker_id, viewer)}"
    if kind in ("block", "no_block"):
        b = _obj_label(eng, decision.blocker_id, viewer)
        if decision.attacker_id is None:
            return f"No block with {b}"
        return f"Block {_obj_label(eng, decision.attacker_id, viewer)} with {b}"
    if kind == "land":
        return f"Play land {_obj_label(eng, decision.card_id, viewer)}"
    label = repr(decision)
    if kind == "cast":
        label = f"Cast {_obj_label(eng, decision.card_id, viewer)}"
        if getattr(decision, "use_warp", False):
            label += " [warp]"
        if getattr(decision, "kicked", False):
            label += " [kicked]"
    elif kind == "activate":
        # Explicit None checks: object id 0 is a real id.
        src = next(
            (
                v for v in (
                    getattr(decision, "source_id", None),
                    getattr(decision, "card_id", None),
                    getattr(decision, "spacecraft_id", None),
                ) if v is not None
            ),
            None,
        )
        label = f"Activate {_obj_label(eng, src, viewer)}"
        idx = getattr(decision, "ability_index", None)
        if idx is not None:
            label += f"[{idx}]"
    if child_state is not None:
        ceng = _engine(child_state)
        if _stack_height(child_state) > _stack_height(parent_state):
            top = ceng.stack[-1]
            tg = getattr(top, "targets", ()) or ()
            if tg:
                label += " -> " + ", ".join(
                    _target_label(ceng, t, viewer) for t in tg
                )
    return label


def _when(node_state: Any) -> str:
    """``T12/P1 upkeep``: turn number, whose turn it is, step."""
    eng = _engine(node_state)
    step = getattr(eng, "step", None)
    try:
        turn_player = eng.players[eng.active_player_idx].id
    except (AttributeError, IndexError):  # pragma: no cover - defensive
        turn_player = "?"
    return (
        f"T{getattr(eng, 'turn_number', '?')}/P{turn_player} "
        f"{step.value if step else '?'}"
    )


def render_tree(
    root: Any,
    root_player: int,
    *,
    max_depth: int = 4,
    top_k: int = 3,
    collapse_forced: bool = True,
    actor_name: Callable[[int], str] = lambda pid: f"P{pid}",
) -> list[str]:
    """Render ``root``'s tree as indented text lines.

    Each line: game time of the decision point, actor, action label,
    visits ``N``, ``Q`` (mean value from the ACTOR's perspective, the
    MCTS convention), PUCT prior ``P``. All root children are shown;
    below the root only the ``top_k`` most-visited children per node,
    ``max_depth`` levels deep. With ``collapse_forced`` a chain of
    forced single-option nodes is folded into one ``(+k forced)`` note
    so the dump shows decisions, not priority bookkeeping."""
    hdr = (
        f"root: {_when(root.state)}, actor {actor_name(root.state.active_player())}"
        f", searcher {actor_name(root_player)}, N={root.visits}, "
        f"options={len(root.children)}"
    )
    lines = [hdr]

    def walk(node: Any, depth: int, indent: str, is_root: bool) -> None:
        if depth >= max_depth:
            return
        kids = sorted(node.children, key=lambda c: c.visits, reverse=True)
        if not is_root:
            kids = kids[:top_k]
        for child in kids:
            cur = child
            skipped = 0
            while (
                collapse_forced
                and len(cur.children) == 1
                and _single_option(cur)
                and not cur.state.is_terminal()
            ):
                cur = cur.children[0]
                skipped += 1
            q = child.mean_value() if child.visits else None
            qs = f"{q:+.2f}" if q is not None else "  n/a"
            actor = actor_name(node.state.active_player())
            lab = label_decision(
                child.decision, node.state, child.state, root_player,
            )
            note = f"  (+{skipped} forced -> {_when(cur.state)})" if skipped else ""
            term = ""
            if cur.state.is_terminal():
                w = cur.state.winner()
                term = f"  [GAME OVER winner={actor_name(w) if w is not None else 'draw'}]"
            lines.append(
                f"{indent}[{_when(node.state)}] {actor}: {lab}"
                f"   N={child.visits} Q={qs} P={child.prior:.2f}{note}{term}"
            )
            walk(cur, depth + 1, indent + "    ", False)

    walk(root, 0, "  ", True)
    return lines


def _single_option(node: Any) -> bool:
    try:
        return len(list(node.state.legal_decisions())) == 1
    except Exception:  # pragma: no cover - defensive for exotic states
        return False
