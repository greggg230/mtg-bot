"""Monte Carlo Tree Search with UCB1/PUCT selection, chance-node sampling,
and evaluator-at-leaf.

Scope: two-player zero-sum games. Chance nodes are expanded but not
optimized over — MCTS samples an outcome by its declared probability
and keeps descending. This matches the canonical "imperfect but
fast" handling of stochastic games in the MCTS literature.

Final-move policy: pick the root's most-visited child. Standard
MCTS — visits track how much confidence we have, which empirically
outperforms picking the highest-value child (prefers robustness
over lucky-eval).

Injectable pieces:

- ``evaluator``        — state → scalar value at leaves.
- ``expansion_policy`` — controls when/what to expand (full
  enumeration by default; swap in progressive widening or a
  prior-driven policy for large action spaces).
- ``budget_factory``   — time or node cap per ``choose()`` call.

Selection: UCB1 by default (``puct=False``), or AlphaZero-style PUCT
with a network-supplied policy prior (``puct=True``). PUCT requires an
evaluator that implements ``evaluate_with_policy(state, player_id) ->
(value, policy_array)``; ``NNEvaluator(with_policy=True)`` is the
intended source. UCB1 stays bit-exact with the pre-PUCT MCTS so old
benchmarks and tests don't drift.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from mtg_bot.cancellation import raise_if_cancelled
from mtg_bot.search.budget import Budget
from mtg_bot.search.canonical_state import canonical_state_key
from mtg_bot.search.evaluator import Evaluator
from mtg_bot.search.expansion import ExpansionPolicy, FullEnumeration
from mtg_bot.search.protocol import (
    CHANCE_PLAYER, Decision, GameState, PlayerID,
)
from mtg_bot.search.strategy import SearchStrategy


@dataclass(slots=True)
class _CachedEval:
    """A cached leaf-evaluation result.

    Holds the value (always populated) and the policy logits
    (populated in PUCT mode; ``None`` under UCB1). Lifetime is one
    ``choose_with_visits`` call — see ``MCTS._eval_cache``.

    ``viewer`` records whose perspective ``value`` (and the policy
    mask) were computed from. Leaf evaluations are always from the
    root player's perspective (that's the value the backup needs);
    prior stamping evaluates from the *mover's* perspective at the
    parent, which differs from the root player at opponent-to-move
    nodes. The cache key already embeds the viewer so the two kinds
    of entry never collide; the field lets every consumer assert it
    is reading a value in the perspective it expects instead of
    silently relying on the zero-sum sign flip. ``None`` means
    "unspecified" (legacy constructor calls in tests).
    """

    value: float
    policy_logits: Optional[np.ndarray] = None
    viewer: Optional[PlayerID] = None


class _Node:
    """Internal MCTS tree node.

    ``state`` is an independent clone held for expansion/rollouts.
    ``decision`` is the move taken at ``parent`` to reach this node
    (None at the root). ``total_value`` sums values from the
    perspective of the *parent's* active player, which is the player
    who chose to come here — matches UCB1's expected interpretation.

    ``pending_decisions`` is a lazy iterator of decisions that haven't
    been expanded yet at this node. The iterator is built by the
    ``ExpansionPolicy`` and may be truncated — progressive widening
    only pulls it when the policy says a new child is warranted.

    Implemented as a plain ``__slots__`` class (not ``@dataclass``)
    because CPython 3.13's specializing interpreter had a
    reproducible bug against the dataclass-mutation pattern in our
    backprop hot loop (intermittent ``NameError``/``AttributeError``
    crashes). Plain class instances with fixed slots avoid the
    specializer's buggy fast path.
    """

    __slots__ = (
        "state", "parent", "decision", "children",
        "pending_decisions", "visits", "total_value", "virtual_loss",
        "prior", "_policy_cache", "leaf_nn_value", "_canonical_key",
    )

    def __init__(
        self,
        state: GameState,
        parent: Optional["_Node"] = None,
        decision: Optional[Decision] = None,
        pending_decisions: Optional[Iterator[Decision]] = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.decision = decision
        self.children: list["_Node"] = []
        self.pending_decisions: Iterator[Decision] = (
            pending_decisions if pending_decisions is not None
            else iter(())
        )
        self.visits: int = 0
        self.total_value: float = 0.0
        # Virtual loss for batched leaf evaluation (B1). Zero when no
        # batched evaluation is in flight through this node.
        # ``_apply_virtual_loss`` increments along the leaf→root path
        # before queuing a leaf for the NN batch; ``_revert_virtual_loss``
        # decrements when the real value comes back. ``ucb1`` and
        # ``mean_value`` consult the *augmented* stats (visits +
        # virtual_loss, total_value - virtual_loss), treating each
        # in-flight evaluation as a worst-case -1 contribution. This
        # discourages other selections in the same batch from
        # converging on the same path. With ``batch_size=1`` (the
        # default), no virtual loss is ever applied — augmented stats
        # equal real stats, and behavior is bit-exact with the
        # pre-batched MCTS.
        self.virtual_loss: int = 0
        # PUCT prior P(s, a): the network's predicted probability that
        # ``self.decision`` is the best move at the parent. Used by
        # ``puct_score``; unused under UCB1. Populated by
        # ``MCTS._make_child`` when ``puct=True`` from the parent's
        # cached policy distribution. 0.0 is the safe default — under
        # UCB1 it's never read, and under PUCT a child whose decision
        # somehow can't be encoded falls through to the small constant
        # exploration the formula provides at large parent-visit
        # counts (better than crashing mid-search).
        self.prior: float = 0.0
        # Cached policy distribution from
        # ``evaluator.evaluate_with_policy(state, mover)`` where
        # ``mover == state.active_player()`` — i.e. ALWAYS the
        # perspective of the player who acts at this node, never the
        # root player's. Stays ``None`` under UCB1; populated lazily
        # in PUCT mode the first time the node either gets expanded
        # or evaluated as a leaf (a leaf eval only stashes it when
        # the leaf's mover IS the root player, since the leaf eval is
        # root-perspective). Reused across all subsequent child
        # expansions from this node so the network sees one forward
        # pass per node per viewer, not one per child.
        self._policy_cache: Optional[np.ndarray] = None
        # Raw NN value at *this state* the first time it was evaluated
        # as a leaf. Stays ``None`` until the first leaf evaluation
        # backs a value through here. Captured for the search-annotation
        # hook (replay viewer's decision-detail panel) so the user
        # can see what the network thought of each root-child
        # position before MCTS averaged in the rollouts. Always-on:
        # populated regardless of UCB1 vs PUCT; the field is just
        # ignored when the annotator isn't asking. PUCT mode passes
        # through ``evaluate_with_policy`` (value head + cached
        # policy); UCB1 mode passes through ``evaluate`` (value
        # only). Either way the per-node "first leaf value" is the
        # same scalar.
        self.leaf_nn_value: Optional[float] = None
        # Memoized canonical-state-key for the per-search NN-eval
        # cache. ``None`` until first computed by
        # ``MCTS._cache_key_for_node`` (or set to a sentinel by paths
        # that know the state isn't an MTG adapter). The bench's
        # ``serial`` MCTS path computes the key exactly once per leaf,
        # so memoizing on the node is mostly a defensive win for the
        # batched paths in ``_search_batched`` (which used to
        # recompute the key once for the cache lookup at queueing time
        # and again at store time after the batched NN call returned)
        # and for any future code path that re-visits the same node.
        # Memory cost: one extra slot per node (~8 bytes) holding a
        # 32-byte digest when populated, dropped with the search tree
        # at the end of ``choose_with_visits``.
        # Type is the same as ``MCTS._cache_key``'s return value:
        # ``Optional[tuple[bytes, tuple]]`` post-drift-fix
        # (canonical-state hash + viewer slot fingerprint). Was
        # ``Optional[bytes]`` pre-drift-fix.
        self._canonical_key: Optional[Any] = None

    def is_leaf(self) -> bool:
        return self.state.is_terminal()

    def mean_value(self) -> float:
        n = self.visits + self.virtual_loss
        return (self.total_value - self.virtual_loss) / n if n else 0.0

    def ucb1(self, parent_visits: int, c: float) -> float:
        n = self.visits + self.virtual_loss
        if n == 0:
            return math.inf
        exploit = (self.total_value - self.virtual_loss) / n
        explore = c * math.sqrt(math.log(parent_visits) / n)
        return exploit + explore

    def puct_score(self, parent_visits: int, c: float) -> float:
        """AlphaZero-style PUCT score:

            Q + c · P(s,a) · √N_parent / (1 + N_child)

        where ``Q`` is the running mean value (from the parent's
        perspective, matching ``ucb1``), ``P(s,a)`` is this child's
        prior from the parent's cached policy, and the +1 in the
        denominator keeps the formula finite when ``N_child == 0``.

        At ``N_child == 0`` the exploit term is undefined (no value
        backed up yet); we fall through to pure-prior exploration:
        ``c · P · √N_parent`` (which is exactly the explore term with
        ``1 + N_child = 1``). Differs from UCB1 here, where unvisited
        children return ``+inf`` to force-pick first. PUCT *wants* to
        weight unvisited children by prior — that's the whole point.
        """
        n = self.visits + self.virtual_loss
        explore = c * self.prior * math.sqrt(parent_visits) / (1 + n)
        if n == 0:
            return explore
        exploit = (self.total_value - self.virtual_loss) / n
        return exploit + explore


class MCTS(SearchStrategy):
    """Budget-bounded MCTS with pluggable expansion.

    Call ``choose(state, player_id)`` once per decision. A fresh tree
    is built each call — we don't reuse across decisions yet (would
    need tree pruning and state-identity equality, both deferred).
    """

    def __init__(
        self,
        evaluator: Evaluator,
        budget_factory,
        rng: random.Random | None = None,
        exploration_c: float = math.sqrt(2),
        expansion_policy: Optional[ExpansionPolicy] = None,
        batch_size: int = 1,
        puct: bool = False,
        root_dirichlet_alpha: Optional[float] = None,
        root_dirichlet_eps: float = 0.25,
    ) -> None:
        """
        ``evaluator`` scores leaves from ``player_id``'s perspective
        (in ``[-1, 1]``; terminals short-circuit to ±1/0).

        ``root_dirichlet_alpha`` / ``root_dirichlet_eps`` add
        AlphaZero-style exploration noise to the ROOT priors under
        PUCT: once every root child has been expanded and its prior
        stamped, each child's prior becomes ``(1 - eps) * prior +
        eps * noise_i`` with ``noise ~ Dirichlet(alpha)`` over the
        root's children. ``alpha=None`` (default) disables the noise
        entirely and leaves the search bit-identical to before (no
        RNG draws are made). The noise is drawn from ``self.rng`` via
        ``gammavariate`` so a seeded search stays deterministic.
        Ignored unless ``puct=True`` (UCB1 has no priors). Only the
        root, only once per ``choose_*`` call — see
        ``_apply_root_dirichlet_noise`` for the timing rationale.

        ``budget_factory`` is a zero-arg callable that returns a new
        ``Budget`` — one per ``choose`` call.

        ``exploration_c`` is UCB1's exploration constant. ``√2`` is
        the theoretical default for values in ``[-1, 1]``. Under PUCT
        the same constant scales the prior-weighted explore term;
        AlphaZero's published value is ~1.25–4. ``√2`` is fine as a
        starting point and lets a single MCTS share the constant
        across the two selection rules.

        ``expansion_policy`` controls tree growth. Defaults to
        ``FullEnumeration`` — preserves classic MCTS behavior.
        Swap in ``ProgressiveWidening`` (optionally with a prior)
        for large action spaces.

        ``batch_size`` controls the batched-leaf-evaluation path
        (B1). With ``batch_size=1`` (default), the search iterates
        leaves serially — bit-exact with the pre-batched MCTS, no
        virtual loss applied. With ``batch_size > 1``, the search
        collects up to ``batch_size`` leaves at a time, applies
        virtual loss along each leaf→root path so subsequent
        selections fan out, calls ``evaluator.batch_evaluate`` once
        per batch, then reverts virtual loss and backprops each
        real value. The win shows up when the evaluator has real
        batched compute (NN inference): one ``(B, D)`` forward pass
        is ~5–25× faster than ``B`` single-state forward passes
        because Python↔C overhead amortizes across the batch.
        Heuristic and rollout evaluators see no win from batching
        and should leave ``batch_size=1``.

        ``puct`` switches selection from UCB1 to AlphaZero PUCT:
        ``Q + c · P(s,a) · √N_parent / (1 + N_child)`` where
        ``P(s,a)`` is the network's policy at the parent. Requires
        an evaluator that exposes ``evaluate_with_policy(state,
        player_id) -> (value, np.ndarray)``; we validate that here
        rather than waiting for the first search call to fail.
        ``puct=False`` (default) preserves the classic UCB1 path
        bit-exact — same selection, same backprop, same
        deterministic outcome under a seeded RNG.

        Combination ``puct=True`` + ``batch_size > 1`` is supported:
        ``_search_batched`` routes through
        ``evaluator.batch_evaluate_with_policy`` and stashes each
        leaf's policy on its ``_policy_cache`` so subsequent
        expansions from that node reuse the cached distribution.
        Requires the evaluator to also expose
        ``batch_evaluate_with_policy``; we validate that here too.
        """
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {batch_size!r}"
            )
        # Per-search NN-eval cache. Allocated fresh at the top of every
        # ``choose_with_visits`` / ``choose_with_full_stats`` call,
        # consulted before each leaf-eval, populated on miss, dropped
        # at end-of-search. Not shared across searches: this is a
        # transposition cache *within* one MCTS tree, not DAG-MCTS
        # (which would also share visit counts). Keys are 32-byte
        # ``canonical_state_key`` digests; cache is bypassed when the
        # leaf state isn't an MTG adapter (no ``engine_state`` attr —
        # toy/test games like tic-tac-toe). See
        # ``mtg_bot.search.canonical_state`` for the equivalence
        # rules. Scout-pass measured ~40% transposition hit rate at
        # PUCT samples=1, inner_nodes=1000 across representative game
        # states; the cache captures that as a wall-clock speedup
        # because the NN forward pass dominates leaf-evaluation cost
        # at PUCT batch_size=1 (the production AlphaZero-training
        # configuration).
        self._eval_cache: dict[Any, _CachedEval] = {}
        # Viewer (root player) for the in-flight search. Set at the
        # top of ``choose_with_*`` and consumed by ``_cache_key`` to
        # build the slot-mapping fingerprint that pins the cache to
        # entries whose action-encoding slots match — see ``_cache_key``
        # docstring for the rationale. ``None`` outside an active
        # search call; ``_cache_key`` defensively returns ``None`` in
        # that case rather than risk a wrong-slot cache hit.
        self._cache_root_player: Optional[PlayerID] = None
        # Debug counter: number of child expansions whose decision
        # could not be encoded into the action vocab (``prior`` left
        # at 0.0). Cumulative across searches on this object (never
        # auto-reset) so tests / benches can assert it stays ~0 in
        # ordinary states. Before the mover-perspective prior fix
        # this fired on essentially every opponent CastSpell /
        # PlayLand / Activate child because the decision was looked
        # up in the ROOT player's zones.
        self.prior_overflow_count: int = 0
        if puct:
            # Construction-time validation: PUCT needs a policy head.
            # ``NNEvaluator(with_policy=True)`` is the canonical
            # source; anything else must duck-type the same method.
            policy_fn = getattr(evaluator, "evaluate_with_policy", None)
            if not callable(policy_fn):
                raise ValueError(
                    "puct=True requires an evaluator with "
                    "evaluate_with_policy(state, player_id); got "
                    f"{type(evaluator).__name__} which does not."
                )
            if batch_size > 1:
                # The batched PUCT path also needs the batched-
                # with-policy entry point. ``NNEvaluator(with_policy
                # =True)`` provides it; anything else must duck-type
                # it. We don't fall back to N single-state calls
                # because the whole point of batch_size>1 is one
                # forward pass per batch — silent fallback would
                # erase the perf win without telling the caller.
                batched_policy_fn = getattr(
                    evaluator, "batch_evaluate_with_policy", None,
                )
                if not callable(batched_policy_fn):
                    raise ValueError(
                        "puct=True with batch_size>1 requires an "
                        "evaluator with batch_evaluate_with_policy("
                        "states, player_id); got "
                        f"{type(evaluator).__name__} which does not."
                    )
        self.evaluator = evaluator
        self.budget_factory = budget_factory
        self.rng = rng or random.Random()
        self.exploration_c = exploration_c
        self.expansion_policy: ExpansionPolicy = (
            expansion_policy if expansion_policy is not None
            else FullEnumeration()
        )
        self.batch_size = batch_size
        self.puct = puct
        if root_dirichlet_alpha is not None and root_dirichlet_alpha <= 0:
            raise ValueError(
                "root_dirichlet_alpha must be > 0 (or None to disable); "
                f"got {root_dirichlet_alpha!r}"
            )
        if not (0.0 <= root_dirichlet_eps <= 1.0):
            raise ValueError(
                "root_dirichlet_eps must be in [0, 1]; "
                f"got {root_dirichlet_eps!r}"
            )
        self.root_dirichlet_alpha: Optional[float] = root_dirichlet_alpha
        self.root_dirichlet_eps: float = root_dirichlet_eps
        # Per-search latch: True once the root's priors have been
        # noised for the in-flight ``choose_*`` call. Reset by
        # ``_make_root``.
        self._root_noise_applied: bool = False
        # Pull the action-encoding version off the evaluator if it
        # advertises one (NNEvaluator does), else default to v1.
        # Threaded into the single ``ae.encode`` call in
        # ``_stamp_child_prior`` below so PUCT prior lookups use the
        # same slot mapping the policy head was trained against.
        self._action_encoding_version = getattr(
            evaluator, "action_encoding_version", "positional_v1",
        )
        # Opt-in search diagnostics (``mtg_bot.search.horizon_probe``):
        # when set, every ``choose_*`` call that actually builds a tree
        # hands the finished root ``_Node`` and the root player to
        # ``tree_observer(root, player_id)`` before the tree is
        # dropped. ``None`` (default) costs one attribute check per
        # search and changes nothing. Forced moves build no tree and
        # are not reported.
        self.tree_observer: Optional[Any] = None

    def choose(
        self, state: GameState, player_id: PlayerID
    ) -> Decision:
        visits = self.choose_with_visits(state, player_id)
        if not visits:
            # Budget too small to expand even once. Fall back to a
            # random legal decision so the caller still gets an answer.
            return self.rng.choice(list(state.legal_decisions()))
        # Standard MCTS final-move policy: pick the most-visited child.
        return max(visits.items(), key=lambda kv: kv[1])[0]

    def choose_with_full_stats(
        self, state: GameState, player_id: PlayerID,
    ) -> tuple[
        Optional[Decision],
        list[dict[str, Any]],
    ]:
        """Run a search and return the chosen decision plus per-root-
        child stats for the search-annotation hook.

        Returns ``(chosen_decision_or_None, candidates)``. Each
        candidate dict carries ``decision`` (the action), ``visits``
        (int), ``q`` (Q from the root's perspective — the parent's
        active-player view, matching ``mean_value`` semantics — None
        when ``visits == 0``), ``prior`` (PUCT prior; None outside
        PUCT mode), and ``leaf_value`` (the raw NN value at the
        child's state on its first leaf evaluation; None when the
        child was never expanded as a leaf, e.g. budget too small,
        or when the evaluator doesn't expose a usable per-state
        scalar).

        ``chosen_decision`` is the most-visited root child (matches
        ``choose``); ``None`` when no children were expanded (budget
        too small) so callers can fall back to a legal default.

        Forced moves: when only one legal decision exists, returns
        ``(decision, [{"decision": d, "visits": 1, "q": None,
        "prior": None, "leaf_value": None}])``. The synthetic shape
        keeps the annotation present (so the viewer panel still
        renders) even though no tree was built.
        """
        legal = list(state.legal_decisions())
        if len(legal) == 1:
            decision = legal[0]
            return (
                decision,
                [
                    {
                        "decision": decision,
                        "visits": 1,
                        "q": None,
                        "prior": None,
                        "leaf_value": None,
                    }
                ],
            )

        # Per-search NN-eval cache lifetime: allocated fresh here, dropped
        # in the ``finally`` clause below. Two consecutive ``choose_with_*``
        # calls must NOT share cache entries — a fresh tree is built each
        # call and cross-call sharing would risk stale entries when the
        # caller mutates ``state`` between calls.
        self._eval_cache = {}
        # Pin the viewer for the slot-mapping fingerprint used by
        # ``_cache_key``. Cleared in the ``finally``.
        self._cache_root_player = player_id
        try:
            root = self._make_root(state)

            budget = self.budget_factory()
            budget.start()
            if self.batch_size == 1:
                self._search_serial(root, player_id, budget)
            else:
                self._search_batched(
                    root, player_id, budget, self.batch_size,
                )

            if self.tree_observer is not None:
                self.tree_observer(root, player_id)
            candidates = [
                {
                    "decision": child.decision,
                    "visits": child.visits,
                    "q": child.mean_value() if child.visits > 0 else None,
                    "prior": child.prior if self.puct else None,
                    "leaf_value": child.leaf_nn_value,
                }
                for child in root.children
            ]
            if not candidates:
                return (None, [])
            chosen = max(
                root.children, key=lambda c: c.visits,
            ).decision
            return (chosen, candidates)
        finally:
            self._eval_cache = {}
            self._cache_root_player = None

    def choose_with_subtree(
        self,
        state: GameState,
        player_id: PlayerID,
        tree_depth: int = 1,
        tree_top_k: Optional[int] = None,
    ) -> tuple[Optional[Decision], list[dict[str, Any]]]:
        """Run a search and return per-root-child stats *plus* the
        nested top-K sub-tree under each candidate, up to
        ``tree_depth`` levels deep.

        Returns ``(chosen_decision_or_None, candidates)``. Each
        candidate dict carries the same fields as
        ``choose_with_full_stats`` plus an extra ``"subtree"`` key:
        a list of grandchild dicts (same shape, recursive). The list
        is empty when ``tree_depth <= 1`` or when the corresponding
        ``_Node`` has no expanded children. At depth ≥ 1 we sort by
        visits desc and slice to ``tree_top_k`` (None = keep all).
        At depth 0 (the root) we keep ALL candidates — that matches
        today's ``choose_with_full_stats`` shape and the data-gen
        loop's "encode every visited slot" expectation.
        ``tree_depth=1`` reproduces ``choose_with_full_stats``
        exactly with ``subtree=[]`` on every candidate.

        ``"subtree"`` is the in-memory key used by callers walking
        the search tree directly. ``build_search_annotation``
        translates this to the on-disk ``children`` field (with
        ``None`` meaning "no deep capture"). Keeping a different
        in-memory name avoids accidentally setting ``children=[]``
        on root-only captures, which would change the on-disk
        shape and break the v3-back-compat invariant.

        Forced moves: same single-element shape as
        ``choose_with_full_stats``, with ``subtree=[]``.
        """
        if tree_depth < 1:
            raise ValueError(
                f"tree_depth must be >= 1, got {tree_depth!r}"
            )
        legal = list(state.legal_decisions())
        if len(legal) == 1:
            decision = legal[0]
            return (
                decision,
                [
                    {
                        "decision": decision,
                        "visits": 1,
                        "q": None,
                        "prior": None,
                        "leaf_value": None,
                        "subtree": [],
                    }
                ],
            )

        root = self._make_root(state)

        budget = self.budget_factory()
        budget.start()
        if self.batch_size == 1:
            self._search_serial(root, player_id, budget)
        else:
            self._search_batched(
                root, player_id, budget, self.batch_size,
            )

        if self.tree_observer is not None:
            self.tree_observer(root, player_id)
        candidates = self._walk_subtree(
            root,
            depth_remaining=tree_depth,
            top_k=tree_top_k,
            is_root=True,
        )
        if not candidates:
            return (None, [])
        chosen = max(
            root.children, key=lambda c: c.visits,
        ).decision
        return (chosen, candidates)

    def _walk_subtree(
        self,
        node: "_Node",
        depth_remaining: int,
        top_k: Optional[int],
        is_root: bool,
    ) -> list[dict[str, Any]]:
        """Recursively translate a ``_Node``'s expanded children into
        the candidate-dict shape used by the search-annotation hook.

        ``depth_remaining`` is the number of additional levels to
        emit beneath ``node``. At ``depth_remaining == 0`` we return
        an empty list (terminate). At each level we sort children
        by visits desc; at non-root levels we slice to ``top_k``
        (keep all when ``top_k is None``). At the root, we keep
        every child regardless of ``top_k`` so the data-gen policy
        encoder still sees the full visit distribution.
        """
        if depth_remaining <= 0:
            return []
        children = list(node.children)
        children.sort(key=lambda c: c.visits, reverse=True)
        if not is_root and top_k is not None:
            children = children[:top_k]
        out: list[dict[str, Any]] = []
        for child in children:
            sub = self._walk_subtree(
                child,
                depth_remaining=depth_remaining - 1,
                top_k=top_k,
                is_root=False,
            )
            out.append({
                "decision": child.decision,
                "visits": child.visits,
                "q": child.mean_value() if child.visits > 0 else None,
                "prior": child.prior if self.puct else None,
                "leaf_value": child.leaf_nn_value,
                "subtree": sub,
            })
        return out

    def choose_with_visits(
        self, state: GameState, player_id: PlayerID
    ) -> dict[Decision, int]:
        """Run a search and return ``{decision: visit_count}`` over the
        root's expanded children.

        Used by ensemble wrappers (PIMC) that need to aggregate visits
        across multiple independent searches before picking a final
        move. Returns an empty dict when the budget was too small to
        expand any child — callers should fall back to a legal default.

        Forced moves (exactly one legal decision) short-circuit the
        search and report a synthetic ``{decision: 1}``. The "1" is a
        placeholder so caller aggregation doesn't need a special case;
        nothing should compare a forced-move count against a real
        search's count.
        """
        # Forced-move shortcut: if there's exactly one legal decision,
        # take it. Computed eagerly up front because it's cheap and
        # lets us skip the whole tree build for trivial priority
        # moments (MTG has lots of these — "only PassPriority legal"
        # is the common case during upkeep/draw/combat).
        legal = list(state.legal_decisions())
        if len(legal) == 1:
            return {legal[0]: 1}

        # Per-search NN-eval cache lifetime: allocated fresh here, dropped
        # in the ``finally`` clause below. See ``__init__`` for design.
        self._eval_cache = {}
        # Pin the viewer for the slot-mapping fingerprint used by
        # ``_cache_key``. Cleared in the ``finally``.
        self._cache_root_player = player_id
        try:
            root = self._make_root(state)

            budget = self.budget_factory()
            budget.start()
            if self.batch_size == 1:
                self._search_serial(root, player_id, budget)
            else:
                self._search_batched(
                    root, player_id, budget, self.batch_size,
                )

            if self.tree_observer is not None:
                self.tree_observer(root, player_id)
            return {c.decision: c.visits for c in root.children}
        finally:
            self._eval_cache = {}
            self._cache_root_player = None

    def _make_root(self, state: GameState) -> _Node:
        """Build the root node for a fresh search and reset the
        per-search root-noise latch. All three ``choose_*`` entry
        points go through here so root construction (and the
        Dirichlet bookkeeping) has exactly one definition."""
        self._root_noise_applied = False
        return _Node(
            state=state.clone(),
            pending_decisions=self.expansion_policy.make_decision_iterator(
                state
            ),
        )

    def _apply_root_dirichlet_noise(self, root: _Node) -> None:
        """Mix Dirichlet(alpha) noise into the root children's priors.

        ``prior_i <- (1 - eps) * prior_i + eps * noise_i`` where
        ``noise`` is one Dirichlet(alpha) draw over the root's
        expanded children, sampled as normalised
        ``rng.gammavariate(alpha, 1.0)`` variates (no numpy in the
        search hot path; deterministic under a seeded ``self.rng``).

        Timing — applied LAZILY, not by pre-expanding the root:
        under ``FullEnumeration`` (the production expansion policy)
        ``_select_and_expand`` expands one new root child per
        iteration until ``pending_decisions`` is exhausted, and the
        priors are never consulted during that phase (an unexpanded
        child can't be PUCT-selected). So the first iteration at
        which ``_try_expand(root)`` returns ``None`` is exactly the
        first iteration whose selection reads root priors — noising
        them there is observationally identical to having noised
        them up front, without perturbing the expansion order or
        the batched-search leaf schedule. Pre-expanding the root
        instead would skip the per-child leaf evaluation + backprop
        that the normal expansion path performs (the children would
        be created without ever being scored as leaves), changing
        the search itself, which is why that route was rejected.

        Caveat: under a ``ProgressiveWidening`` root the latch fires
        the first time the policy declines to widen, so children
        added later get un-noised priors. Root noise is designed for
        (and only wired from) the full-enumeration self-play path.
        """
        self._root_noise_applied = True
        children = root.children
        if not children:
            return
        alpha = self.root_dirichlet_alpha
        eps = self.root_dirichlet_eps
        assert alpha is not None
        gammas = [self.rng.gammavariate(alpha, 1.0) for _ in children]
        total = sum(gammas)
        if total > 0.0:
            noise = [g / total for g in gammas]
        else:  # pragma: no cover - astronomically unlikely underflow
            noise = [1.0 / len(children)] * len(children)
        for child, n_i in zip(children, noise):
            child.prior = (1.0 - eps) * child.prior + eps * n_i

    def _search_serial(
        self, root: _Node, player_id: PlayerID, budget,
    ) -> None:
        """Today's iteration loop, factored out for symmetry with
        ``_search_batched``. Bit-exact with the pre-batching MCTS
        — no virtual loss is applied along the way.

        ``player_id`` is also the *root player* in PUCT mode and gets
        threaded down to ``_select_and_expand`` / ``_evaluate_leaf``
        so the policy head sees the search from the same viewer
        perspective at every node."""
        while not budget.exhausted():
            # Cooperative stop for an abandoned (timed-out) game — see
            # mtg_bot.cancellation. No-op (~100ns) unless this thread
            # is running a game that run_with_timeout gave up on.
            raise_if_cancelled()
            leaf = self._select_and_expand(root, player_id)
            value = self._evaluate_leaf(leaf, player_id)
            self._backprop(leaf, value, player_id)
            budget.consume_node()

    def _search_batched(
        self, root: _Node, player_id: PlayerID, budget,
        batch_size: int,
    ) -> None:
        """Collect up to ``batch_size`` leaves with virtual loss,
        evaluate them in one ``batch_evaluate`` call, revert virtual
        loss, and backprop each real value. See ``__init__`` for
        the design rationale. Terminal leaves short-circuit to the
        evaluator's terminal handling and are backpropped
        immediately without entering the batch (no NN value to
        wait on).

        Under PUCT (``self.puct=True``), the batched evaluator is
        ``batch_evaluate_with_policy`` rather than ``batch_evaluate``
        — one forward pass yields both heads for every leaf. Each
        leaf's policy gets stashed on ``_Node._policy_cache`` so
        when the leaf later becomes an interior node and starts
        expanding children, ``_stamp_prior`` reads the cached
        distribution rather than triggering a second forward pass.
        That preserves the "exactly one policy query per node"
        invariant that the serial PUCT path enforces via
        ``_evaluate_leaf``.

        Cache invariant under batching: a pending leaf has no
        ``_policy_cache`` yet (it's queued for the upcoming batch
        call). If the next ``_select_and_expand`` would descend
        through a still-pending leaf and try to expand a grandchild,
        ``_stamp_prior`` would have to fire a synchronous single-
        state forward pass — defeating the cache. We avoid this by
        flushing the batch early: if the descent is about to enter
        a pending leaf, we break out of inner-loop collection,
        fire ``batch_evaluate_with_policy``, populate the leaves'
        caches, and then resume collecting. The effective batch
        size is bounded by both ``batch_size`` and the tree's
        breadth at the current frontier.
        """
        while not budget.exhausted():
            # Cooperative stop for an abandoned (timed-out) game — see
            # mtg_bot.cancellation.
            raise_if_cancelled()
            pending: list[_Node] = []
            pending_set: set[int] = set()
            terminal_results: list[tuple[_Node, float]] = []
            # Cached leaves collected during this batch — resolved
            # instantly from ``self._eval_cache`` without the NN call.
            # Treated like terminals: backpropped before the batch
            # call, no virtual loss applied.
            cached_results: list[tuple[_Node, float]] = []
            # PUCT only: non-terminal leaves whose MOVER is not the
            # root player (opponent-to-move) need a second,
            # mover-perspective forward pass to populate
            # ``_policy_cache`` — the root-perspective leaf eval's
            # policy is masked to the root player's (degenerate,
            # pass-only) legal set and is useless as a prior over the
            # opponent's options. Grouped by mover so each group is
            # one extra batched call rather than N synchronous
            # single-state calls from ``_stamp_prior`` later.
            policy_pending: dict[PlayerID, list[_Node]] = {}
            # Collect a batch worth of leaves. Each non-terminal
            # non-cached leaf gets virtual loss applied so the next
            # selection in this batch sees a degraded UCB1 score on the
            # path we're already evaluating.
            for _ in range(batch_size):
                if budget.exhausted():
                    break
                # The pending_blockers set is consulted only in the
                # PUCT path (see _select_and_expand) to keep the cache
                # invariant. Under UCB1 it's harmlessly ignored — the
                # set is cheap to build either way.
                leaf = self._select_and_expand(
                    root, player_id, pending_blockers=pending_set,
                )
                if leaf is None:
                    # ``_select_and_expand`` aborted because it would
                    # have descended into a still-pending leaf. Fire
                    # the batch we've collected so far, populate the
                    # caches, then restart the outer loop to continue
                    # collecting from a fresh state.
                    break
                budget.consume_node()
                if leaf.is_leaf():
                    # Terminal — evaluator short-circuits to ±1/0
                    # without touching the network. Backprop
                    # immediately; no virtual loss needed because
                    # nothing's pending on this leaf.
                    value = self._evaluate_leaf(leaf, player_id)
                    terminal_results.append((leaf, value))
                    continue
                # Per-search NN-eval cache check. On hit, skip both
                # the batch and the virtual-loss machinery — the cached
                # value is the same one a fresh forward pass would
                # produce, so we can backprop it immediately and free
                # up the batch slot for a leaf that actually needs the
                # network. Mirrors the cacheless ``_evaluate_leaf``
                # bookkeeping (leaf_nn_value, _policy_cache). The
                # node-memoizing helper saves the (otherwise duplicate)
                # hash recomputation that would happen at store time
                # below if this leaf turns out to be a cache miss.
                cache_key = self._cache_key_for_node(leaf)
                cached = (
                    self._eval_cache.get(cache_key)
                    if cache_key is not None else None
                )
                if cached is not None:
                    self._check_cached_viewer(cached, player_id)
                    self._maybe_stash_policy(
                        leaf, cached.policy_logits, player_id,
                    )
                    if leaf.leaf_nn_value is None:
                        leaf.leaf_nn_value = float(cached.value)
                    cached_results.append((leaf, cached.value))
                    if self._queue_mover_policy(
                        leaf, player_id, policy_pending,
                    ):
                        # The leaf still needs its mover-perspective
                        # policy; block descents through it until the
                        # side batch below has populated it.
                        pending_set.add(id(leaf))
                    continue
                self._apply_virtual_loss(leaf)
                pending.append(leaf)
                pending_set.add(id(leaf))
                self._queue_mover_policy(leaf, player_id, policy_pending)

            # Backprop terminal + cache-hit leaves first — they have
            # real values immediately and don't depend on the in-flight
            # NN batch. Order between the two doesn't affect
            # correctness.
            for leaf, value in terminal_results:
                self._backprop(leaf, value, player_id)
            for leaf, value in cached_results:
                self._backprop(leaf, value, player_id)

            # Batch-evaluate the non-terminal non-cached leaves and
            # backprop. Cache-misses go into the network call;
            # results get stored before backprop so subsequent leaves
            # at the same canonical state in later batches will hit.
            if pending:
                states = [leaf.state for leaf in pending]
                if self.puct:
                    # One forward pass, both heads. Each (value,
                    # policy) pair lines up with the matching leaf
                    # by index. Stash the policy on the leaf so
                    # subsequent expansions from that node read the
                    # cached distribution and don't trigger a
                    # second forward pass.
                    pairs = self.evaluator.batch_evaluate_with_policy(
                        states, player_id,
                    )
                    for leaf, (value, policy) in zip(pending, pairs):
                        # Root-perspective policy is only a valid
                        # prior source when the leaf's mover IS the
                        # root player; opponent-mover leaves get
                        # theirs from the side batch below.
                        self._maybe_stash_policy(leaf, policy, player_id)
                        if leaf.leaf_nn_value is None:
                            leaf.leaf_nn_value = float(value)
                        # Store in the per-search cache so a later
                        # transposition through the same canonical
                        # state can short-circuit. The hash was
                        # already computed at the cache-lookup site
                        # above and memoized on the node, so this
                        # call hits the slot instead of re-hashing.
                        cache_key = self._cache_key_for_node(leaf)
                        if cache_key is not None:
                            self._eval_cache[cache_key] = _CachedEval(
                                value=float(value), policy_logits=policy,
                                viewer=player_id,
                            )
                        self._revert_virtual_loss(leaf)
                        self._backprop(leaf, value, player_id)
                else:
                    values = self.evaluator.batch_evaluate(
                        states, player_id,
                    )
                    for leaf, value in zip(pending, values):
                        if leaf.leaf_nn_value is None:
                            leaf.leaf_nn_value = float(value)
                        cache_key = self._cache_key_for_node(leaf)
                        if cache_key is not None:
                            self._eval_cache[cache_key] = _CachedEval(
                                value=float(value), policy_logits=None,
                                viewer=player_id,
                            )
                        self._revert_virtual_loss(leaf)
                        self._backprop(leaf, value, player_id)

            # PUCT side batch: mover-perspective policies for the
            # opponent-to-move leaves collected this round (both the
            # NN-evaluated and the cache-hit ones). One batched call
            # per distinct mover (exactly one in a 2-player game).
            # The value that comes back is the MOVER's, not the root
            # player's — it is stored under the mover-keyed cache
            # entry only and never backpropped.
            for mover, leaves in policy_pending.items():
                self._flush_mover_policies(mover, leaves)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_and_expand(
        self, root: _Node, root_player: PlayerID,
        pending_blockers: Optional[set[int]] = None,
    ) -> Optional[_Node]:
        """Walk down the tree until we find a node to score.

        - At chance nodes, sample an outcome by its probability and
          descend (creating a new child if needed).
        - At real-player nodes, ask the ``ExpansionPolicy`` whether
          to add a new child. If yes and the pending-iterator has
          more decisions, expand and return the new child.
        - Otherwise, descend into the UCB1-best existing child
          (or PUCT-best when ``self.puct``).
        - At terminals, return as-is.

        ``root_player`` flows through to ``_make_child`` /
        ``_stamp_prior`` as the search viewer. Note the two
        perspectives in play: *values* are always from
        ``root_player``'s perspective (``_backprop`` negates at
        opponent-mover parents), while PUCT *priors* are computed
        from the perspective of the player who actually acts at the
        parent (``parent.state.active_player()``). Evaluating the
        policy head from the root player's view at an opponent node
        would mask it to the root player's legal set — ``[Pass]``
        when they don't hold priority — and model the opponent as
        "always passes".

        ``pending_blockers`` (PUCT-only) is the id-set of leaves
        currently in the in-flight batch. If a descent would land
        on (or pass through) one of these nodes, expansion of its
        children would require a not-yet-populated policy cache —
        breaking the "no double evaluation" invariant. Returning
        ``None`` signals the caller to flush the current batch
        before continuing. Under UCB1 the set is ignored.
        """
        node = root
        while not node.is_leaf():
            if node.state.active_player() == CHANCE_PLAYER:
                node = self._descend_chance(node, root_player)
                continue

            expanded = self._try_expand(node, root_player)
            if expanded is not None:
                return expanded

            # Root fully expanded (per the expansion policy) and
            # about to be PUCT-selected for the first time: this is
            # where the Dirichlet root noise goes. See
            # ``_apply_root_dirichlet_noise`` for why it's lazy.
            if (
                node is root
                and self.puct
                and self.root_dirichlet_alpha is not None
                and not self._root_noise_applied
            ):
                self._apply_root_dirichlet_noise(root)

            if not node.children:
                # No children to descend into and policy refused to
                # expand — dead end. Return this node as the leaf
                # so evaluation still happens. (Shouldn't arise
                # under FullEnumeration; a badly-tuned
                # ProgressiveWidening could get here if it caps
                # children at 0.)
                return node
            # Fully widened (per policy) — UCB1 / PUCT descent.
            # Parent visits is the augmented count (real +
            # virtual_loss) so the exploration term reflects
            # pending in-flight evaluations during batched search;
            # identical to today when virtual_loss is zero
            # everywhere.
            parent_n = node.visits + node.virtual_loss
            if self.puct:
                node = max(
                    node.children,
                    key=lambda c: c.puct_score(
                        parent_n, self.exploration_c,
                    ),
                )
                # Cache-invariant guard: if we're about to descend
                # into a leaf that's still queued for the in-flight
                # batch, its ``_policy_cache`` is empty. Expanding
                # one of its grandchildren would force ``_stamp_prior``
                # to fire a synchronous single-state forward pass
                # — exactly the duplicate eval the caller is trying
                # to prevent. Signal "flush the batch first".
                if (
                    pending_blockers is not None
                    and id(node) in pending_blockers
                ):
                    return None
            else:
                node = max(
                    node.children,
                    key=lambda c: c.ucb1(parent_n, self.exploration_c),
                )
        return node

    def _try_expand(
        self, node: _Node, root_player: PlayerID,
    ) -> Optional[_Node]:
        """If the policy permits and the pending iterator has more,
        expand a new child and return it. Otherwise return None."""
        if not self.expansion_policy.should_expand(node):
            return None
        try:
            decision = next(node.pending_decisions)
        except StopIteration:
            return None
        return self._make_child(node, decision, root_player)

    def _make_child(
        self,
        parent: _Node,
        decision: Decision,
        root_player: PlayerID,
    ) -> _Node:
        """Build and attach a new child for ``decision`` at
        ``parent``. The child's pending-decisions iterator is
        freshly built from the expansion policy.

        Under PUCT, also stamp the child's ``prior`` from the
        parent's cached policy distribution (computing the
        distribution on first call). Encoding errors are tolerated:
        an unencodable decision (an action-vocab gap or a decision
        that legality enumerated but the codec doesn't know how to
        place) leaves ``prior=0.0``, which means PUCT treats it as
        no-prior — explore by visit-count alone. That's a soft
        failure rather than a crash because legality and
        encodability can drift slightly under engine evolution and
        we'd rather lose a little selection signal than abort the
        search.
        """
        child_state = parent.state.clone()
        child_state.apply(decision)
        pending = (
            self.expansion_policy.make_decision_iterator(child_state)
            if not child_state.is_terminal() else iter(())
        )
        child = _Node(
            state=child_state,
            parent=parent,
            decision=decision,
            pending_decisions=pending,
        )
        if self.puct:
            self._stamp_prior(parent, child, decision, root_player)
        parent.children.append(child)
        return child

    def _stamp_prior(
        self,
        parent: _Node,
        child: _Node,
        decision: Decision,
        root_player: PlayerID,
    ) -> None:
        """Set ``child.prior`` from ``parent._policy_cache[encode(d)]``,
        populating the cache on first call.

        **Perspective.** The prior is computed from the perspective
        of the *mover* at ``parent`` — ``parent.state.active_player()``
        — not from ``root_player``'s. The adapter's ``active_player``
        already resolves the actor for factored-combat pauses (the
        active player for ``AttackerDecision``, the defender for
        ``BlockerDecision``) and mid-resolution pauses (the player
        the engine callback asked), so it is the actor of every
        decision ``parent.legal_decisions()`` yields. Both the
        network's legal-action mask (built inside
        ``evaluate_with_policy`` for the given player) and the
        ``encode`` zone lookup are keyed on that mover, so opponent
        casts / land plays / activations resolve to real slots
        instead of overflowing. ``root_player`` is kept in the
        signature as the search viewer; it only matters here to
        decide whether the mover-perspective policy is the same one
        the leaf evaluation already produced.

        Chance parents (``CHANCE_PLAYER``) have no mover; their
        children keep ``prior=0.0`` and no network call is made.

        Forced-move shortcut: when the mover is an opponent and the
        parent has exactly one legal decision (the common
        "opponent auto-passes" case), the prior is set to ``1.0``
        without a forward pass — a single child needs no prior for
        selection, and the masked softmax over one legal slot would
        return ``1.0`` anyway.

        Per-search NN-eval cache integration: before triggering a
        synchronous single-state forward pass to populate
        ``parent._policy_cache``, we check ``self._eval_cache`` keyed
        on (canonical state, mover, mover slot fingerprint). On
        cache hit we reuse the stored policy without a network
        call. On miss we run the forward pass and store the result
        (value AND policy from the mover's perspective, tagged with
        ``viewer=mover``) so subsequent expansions at the same
        canonical state share it. The root-perspective entry for
        the same state lives under a different key.

        Lazy-imported codec because ``mtg_bot.training`` pulls in
        ``mtg_bot.search`` at package init — importing at module load
        would create a cycle. By the time any search runs the cycle
        has resolved.
        """
        mover = parent.state.active_player()
        if mover == CHANCE_PLAYER:
            return
        if parent._policy_cache is None:
            if (
                mover != root_player
                and self._has_single_legal_decision(parent.state)
            ):
                child.prior = 1.0
                return
            cache_key = self._policy_cache_key_for_node(parent, mover)
            cached = (
                self._eval_cache.get(cache_key)
                if cache_key is not None else None
            )
            if (
                cached is not None
                and cached.policy_logits is not None
                and cached.viewer == mover
            ):
                parent._policy_cache = cached.policy_logits
            else:
                value, policy = self.evaluator.evaluate_with_policy(
                    parent.state, mover,
                )
                parent._policy_cache = policy
                if cache_key is not None:
                    self._eval_cache[cache_key] = _CachedEval(
                        value=float(value), policy_logits=policy,
                        viewer=mover,
                    )
        from mtg_bot.training import action_encoding as ae
        eng = getattr(parent.state, "engine_state", parent.state)
        try:
            slot = ae.encode(
                decision, eng, mover,
                version=self._action_encoding_version,
            )
        except ae.EncodingOverflowError:
            # Decision didn't fit in the fixed vocab. Leave prior 0;
            # PUCT will explore by visit-count alone for this child.
            self.prior_overflow_count += 1
            return
        child.prior = float(parent._policy_cache[slot])

    @staticmethod
    def _has_single_legal_decision(state: GameState) -> bool:
        """True iff ``state`` has exactly one legal decision. On the
        MTG adapter this re-reads the per-decision-moment legal
        cache that ``_make_child``'s iterator construction already
        populated, so it is cheap."""
        legal = state.legal_decisions()
        try:
            return len(legal) == 1
        except TypeError:
            return len(list(legal)) == 1

    def _maybe_stash_policy(
        self, node: _Node, policy: Optional[np.ndarray],
        viewer: PlayerID,
    ) -> None:
        """Stash ``policy`` on ``node._policy_cache`` iff it is the
        policy the node's MOVER would use — i.e. ``viewer`` is the
        player to act at ``node``. A root-perspective policy at an
        opponent-mover node is masked to the root player's legal set
        (``[Pass]``) and must NOT be used as that node's prior
        source; ``_stamp_prior`` will compute the mover's instead."""
        if (
            self.puct
            and policy is not None
            and node._policy_cache is None
            and node.state.active_player() == viewer
        ):
            node._policy_cache = policy

    @staticmethod
    def _check_cached_viewer(
        cached: _CachedEval, expected: PlayerID,
    ) -> None:
        """Assert a cache entry's value is in the perspective the
        consumer expects. The key already embeds the viewer, so a
        mismatch means a keying bug — fail loudly rather than
        backprop a sign-flipped value."""
        if cached.viewer is not None and cached.viewer != expected:
            raise RuntimeError(
                f"_eval_cache entry viewer={cached.viewer} read as "
                f"viewer={expected}; cache keying is broken"
            )

    def _queue_mover_policy(
        self, leaf: _Node, root_player: PlayerID,
        policy_pending: dict[PlayerID, list[_Node]],
    ) -> bool:
        """Batched-PUCT helper. If ``leaf`` is an opponent-to-move
        node that will need a mover-perspective policy when expanded,
        either satisfy it from the mover-keyed cache entry now or
        queue it for the side batch. Returns True iff the leaf was
        queued (caller adds it to the in-flight blocker set)."""
        if not self.puct or leaf._policy_cache is not None:
            return False
        mover = leaf.state.active_player()
        if mover == root_player or mover == CHANCE_PLAYER:
            return False
        if self._has_single_legal_decision(leaf.state):
            # ``_stamp_prior`` short-circuits to prior=1.0 here; no
            # forward pass needed.
            return False
        cache_key = self._policy_cache_key_for_node(leaf, mover)
        cached = (
            self._eval_cache.get(cache_key)
            if cache_key is not None else None
        )
        if (
            cached is not None
            and cached.policy_logits is not None
            and cached.viewer == mover
        ):
            leaf._policy_cache = cached.policy_logits
            return False
        policy_pending.setdefault(mover, []).append(leaf)
        return True

    def _flush_mover_policies(
        self, mover: PlayerID, leaves: list[_Node],
    ) -> None:
        """Batched-PUCT helper: one ``batch_evaluate_with_policy``
        call from ``mover``'s perspective over ``leaves``, stashing
        each policy on its leaf and in the mover-keyed cache. The
        returned values are the mover's and are never backpropped."""
        if not leaves:
            return
        pairs = self.evaluator.batch_evaluate_with_policy(
            [leaf.state for leaf in leaves], mover,
        )
        for leaf, (value, policy) in zip(leaves, pairs):
            if leaf._policy_cache is None:
                leaf._policy_cache = policy
            cache_key = self._policy_cache_key_for_node(leaf, mover)
            if cache_key is not None and cache_key not in self._eval_cache:
                self._eval_cache[cache_key] = _CachedEval(
                    value=float(value), policy_logits=policy,
                    viewer=mover,
                )

    def _descend_chance(
        self, node: _Node, root_player: PlayerID,
    ) -> _Node:
        """Sample a chance outcome, either picking the existing
        child for that outcome or creating one."""
        outcomes = node.state.chance_outcomes()
        decisions = [d for d, _ in outcomes]
        weights = [w for _, w in outcomes]
        decision = self.rng.choices(decisions, weights=weights, k=1)[0]
        for child in node.children:
            if child.decision == decision:
                return child
        # First time we sampled this outcome — create the child.
        # Chance nodes don't have a meaningful policy prior (they
        # aren't a real player's choice), so PUCT priors at chance
        # children stay 0.0. They never get UCB-selected anyway —
        # they're sampled by weight.
        return self._make_child(node, decision, root_player)

    def _cache_key(self, state: GameState) -> Optional[Any]:
        """Return a content-addressed hash of ``state`` suitable for
        the per-search NN-eval cache, or ``None`` when the state isn't
        an MTG adapter (toy/test games).

        Detection: an MTG adapter exposes ``engine_state``; toys
        don't. The latter fall through to the cacheless fast path —
        their evaluators (rollout, heuristic) are cheap enough that
        caching is pointless and the canonical-state hash is
        MTG-specific anyway.

        The key combines two pieces:

        1. ``canonical_state_key(state)`` — the forward-play-equivalent
           hash that drops raw ``GameObject.id`` ordering inside the
           hand / graveyard / exile so two states with permuted card
           ids hash identically.
        2. A **slot-mapping fingerprint** of the viewer's hand /
           graveyard / exile / library-top raw ``card.id`` ordering.

        Why both: the cached ``policy_logits`` from
        ``evaluate_with_policy`` are positional — slot
        ``LAND_FROM_HAND_BASE + i`` is the prior for "play the card
        at hand position ``i``". Two canonically-equivalent states
        with permuted card.ids in hand have IDENTICAL value AND
        IDENTICAL post-softmax-by-canonical-position policy, but
        different ``card.id`` -> position -> slot mappings. Reusing
        the cached logits at slot indices computed from the new
        state's positional encoding reads the WRONG prior for each
        decision (smoking-gun observed 2026-05-06: cache-on flipped
        the chosen action at T3-main when hand had two Mountains).

        Including the slot fingerprint scopes hits to states where
        the slot mapping is identical — same value AND same policy
        slots, so the cached pair is safe to reuse byte-for-byte.
        Slot-mismatched transpositions miss the cache and recompute
        (preserving correctness; the perf cost is the lost transposition
        hits when card.ids permute, but those hits would have been
        wrong-slot reads anyway).

        ``self._cache_root_player`` is the viewer (set at the top of
        ``choose_with_visits`` / ``choose_with_full_stats``) — slot
        encoding is positional in the viewer's zones, and the viewer
        is fixed for the entire search.

        Note: most internal call sites prefer
        :meth:`_cache_key_for_node`, which memoizes the
        ``canonical_state_key`` portion of the digest on the
        :class:`_Node` so repeated calls during the same search (e.g.
        ``_search_batched``'s cache-lookup-then-store pattern) only
        pay the hash cost once per node. This raw method stays as
        the public hook for tests that monkey-patch it (e.g. the
        bench script's ``cache_disabled=True`` mode) and for the
        rare paths that hash a state without a node attached.
        """
        if not hasattr(state, "engine_state"):
            return None
        eng = state.engine_state
        viewer = self._cache_root_player
        # Defensive: if the cache is consulted before
        # ``choose_with_*`` set the viewer (shouldn't happen via the
        # production paths), bypass the cache rather than risk a
        # wrong-slot hit.
        if viewer is None:
            return None
        return (
            canonical_state_key(state),
            self._slot_fingerprint(eng, viewer),
        )

    @staticmethod
    def _slot_fingerprint(eng: Any, viewer: PlayerID) -> tuple:
        """Viewer-tagged positional fingerprint of ``viewer``'s hand /
        graveyard / exile / library-top raw ``card.id`` ordering.
        Leading with ``viewer`` itself means the root-perspective and
        mover-perspective entries for one canonical state occupy
        distinct cache keys."""
        p = eng.player_by_id(viewer)
        return (
            viewer,
            tuple(c.id for c in p.hand),
            tuple(c.id for c in p.graveyard),
            tuple(c.id for c in p.exile),
            p.library[0].id if p.library else -1,
        )

    def _policy_cache_key_for_node(
        self, node: _Node, viewer: PlayerID,
    ) -> Optional[Any]:
        """Cache key for a ``viewer``-perspective policy at ``node``.

        Same shape as :meth:`_cache_key` — ``(canonical hash, slot
        fingerprint)`` — but the fingerprint is taken over
        ``viewer``'s zones so the positional policy logits line up
        with ``encode(decision, eng, viewer)``. For the root viewer
        this IS the memoized root key; for any other viewer we reuse
        the memoized canonical hash and swap the fingerprint.
        Returns ``None`` whenever the root key is ``None`` (toy state,
        or a test/bench that monkey-patched ``_cache_key`` to disable
        the cache) so the override is honored for every viewer.
        """
        root_key = self._cache_key_for_node(node)
        if root_key is None:
            return None
        if viewer == self._cache_root_player:
            return root_key
        eng = getattr(node.state, "engine_state", None)
        if eng is None:
            return None
        return (root_key[0], self._slot_fingerprint(eng, viewer))

    def _cache_key_for_node(self, node: _Node) -> Optional[bytes]:
        """Like :meth:`_cache_key` but memoized on ``node``.

        Computes ``canonical_state_key(node.state)`` on first call and
        stashes the result in ``_Node._canonical_key`` so subsequent
        calls return in O(1). For non-MTG states (toys) the slot stays
        ``None`` and every call falls through to a cheap ``hasattr``
        check; cycling through the slow path on toys isn't worth a
        second sentinel because the cache is a no-op for them anyway.
        """
        cached = node._canonical_key
        if cached is not None:
            return cached
        # Honor monkey-patched ``_cache_key`` (e.g. the bench's
        # cache-disabled mode swaps in a ``lambda state: None``).
        # Tests that override ``_cache_key`` to return ``None``
        # expect every call to be cacheless; respect that by going
        # through the override rather than the canonical-state hash
        # directly.
        key = self._cache_key(node.state)
        if key is not None:
            node._canonical_key = key
        return key

    def _evaluate_leaf(
        self, leaf: _Node, root_player: PlayerID
    ) -> float:
        """Return the leaf's value from ``root_player``'s perspective,
        in ``[-1, 1]``. The evaluator handles terminal vs non-terminal
        internally.

        Under PUCT we route through ``evaluate_with_policy`` so the
        single forward pass produces both heads at once; the policy
        gets cached on the leaf for any future expansion from this
        node, avoiding a second forward pass when the leaf later
        becomes an interior node. Terminal leaves still go through
        the value-only path because the policy head is undefined
        there (no legal action; ``evaluate_with_policy`` returns the
        all-zeros vector for terminals, which we don't bother
        caching — a terminal never gets expanded).

        Per-search NN-eval cache (``self._eval_cache``): on MTG
        adapter states we hash the state via ``canonical_state_key``
        before calling the evaluator. Cache hit returns the stored
        ``(value, policy)`` pair without touching the network; cache
        miss runs the evaluator and stores the result before
        returning. Terminal leaves bypass the cache: their evaluator
        path short-circuits to ±1/0 with no NN call, so caching them
        wouldn't save anything (and the canonical hash is correct
        but pointless). Toy/test states (no ``engine_state`` attr)
        also bypass the cache transparently — the framework
        operates as before for non-MTG inputs.
        """
        # Terminal leaves: always go through the value-only path; no
        # cache lookup (no NN call to skip).
        if leaf.is_leaf():
            value = self.evaluator.evaluate(leaf.state, root_player)
            if leaf.leaf_nn_value is None:
                leaf.leaf_nn_value = float(value)
            return value

        cache_key = self._cache_key_for_node(leaf)
        if cache_key is not None:
            cached = self._eval_cache.get(cache_key)
            if cached is not None:
                # Cache HIT — skip the NN. Mirror the bookkeeping the
                # cacheless paths do: stash the policy on the leaf
                # for future child-expansions (only when the leaf's
                # mover is the root player — see
                # ``_maybe_stash_policy``), set ``leaf_nn_value``
                # for the annotation hook.
                self._check_cached_viewer(cached, root_player)
                self._maybe_stash_policy(
                    leaf, cached.policy_logits, root_player,
                )
                if leaf.leaf_nn_value is None:
                    leaf.leaf_nn_value = float(cached.value)
                return cached.value

        # Cache MISS — run the evaluator and store.
        if self.puct:
            value, policy = self.evaluator.evaluate_with_policy(
                leaf.state, root_player,
            )
            self._maybe_stash_policy(leaf, policy, root_player)
            if leaf.leaf_nn_value is None:
                leaf.leaf_nn_value = float(value)
            if cache_key is not None:
                self._eval_cache[cache_key] = _CachedEval(
                    value=float(value), policy_logits=policy,
                    viewer=root_player,
                )
            return value
        value = self.evaluator.evaluate(leaf.state, root_player)
        if leaf.leaf_nn_value is None:
            leaf.leaf_nn_value = float(value)
        if cache_key is not None:
            self._eval_cache[cache_key] = _CachedEval(
                value=float(value), policy_logits=None,
                viewer=root_player,
            )
        return value

    def _backprop(
        self, leaf: _Node, value_for_root: float,
        root_player: PlayerID,
    ) -> None:
        """Walk from leaf to root, updating visits and total_value.

        ``total_value`` is stored from the perspective of the parent's
        active player (the one who chose to move to this child). In a
        two-player zero-sum game the perspectives differ only by sign.
        """
        node: _Node | None = leaf
        while node is not None:
            node.visits += 1
            if node.parent is not None:
                parent_player = node.parent.state.active_player()
                if parent_player == root_player:
                    node.total_value += value_for_root
                elif parent_player == CHANCE_PLAYER:
                    # Chance parents don't have a "mover perspective" —
                    # total_value is unused for their children (we
                    # sample rather than UCB-select). Leave it zero.
                    pass
                else:
                    node.total_value += -value_for_root
            node = node.parent

    def _apply_virtual_loss(self, leaf: _Node) -> None:
        """Add a worst-case -1 contribution along the leaf→root path
        so subsequent selections in the same batch see a degraded
        UCB1 score on this path and fan out to other branches.
        Reverted by ``_revert_virtual_loss`` when the real value
        comes back from ``batch_evaluate``."""
        node: _Node | None = leaf
        while node is not None:
            node.virtual_loss += 1
            node = node.parent

    def _revert_virtual_loss(self, leaf: _Node) -> None:
        """Undo a single ``_apply_virtual_loss`` along the leaf→root
        path. Called once per batched leaf right before its real
        value gets backpropped."""
        node: _Node | None = leaf
        while node is not None:
            node.virtual_loss -= 1
            node = node.parent
