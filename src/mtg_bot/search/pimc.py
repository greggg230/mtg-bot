"""Perfect Information Monte Carlo (PIMC) for MTG.

Wraps an inner ``MCTS`` (or any strategy that exposes a
``choose_with_visits`` method). Each ``choose()`` call:

1. Pulls the underlying engine ``GameState`` from the
   ``MtgGameStateAdapter`` it was handed.
2. Samples ``K`` determinized worlds — each one is a clone of the
   engine state with every card the viewer doesn't know re-permuted
   within its owner's hand+library pool. See
   ``mtg_bot.search.determinize`` for the sampler.
3. Wraps each world in a fresh adapter and runs the inner search,
   collecting per-decision visit counts.
4. Aggregates visits across worlds; returns the most-visited decision.

Why this is the integration shape: the determinizer operates on the
engine state (printings live there), but the search runs against the
search-protocol adapter. PIMC bridges them — pulling the engine state
out, determinizing, wrapping back up, then handing each world to a
separate inner search. The inner search itself stays MTG-ignorant.

Soundness vs. running raw MCTS on the god-view state:
- Each sub-search sees a viewer-consistent world. No more cheating on
  top-of-library or opponent hand contents.
- Visit-count aggregation (vs. ensemble vote on the chosen decision)
  weights heavily-explored decisions more, matching the standard PIMC
  literature. A move that wins 30 games out of 1,000 visits in one
  world doesn't outweigh a move that wins 200 out of 800 in another.

Known limitations (canonical PIMC issues; both fixable by upgrading
to Information-Set MCTS later):
- "Strategy fusion": each sub-search picks its strategy assuming it
  knows the world. PIMC doesn't model "I'd play differently if I
  learned X."
- "Non-locality": PIMC over-trusts moves that look great in the few
  worlds where opponent's hand happens to be cooperative. Larger K
  reduces (but doesn't eliminate) the bias.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional, Protocol

from mtg_bot.cancellation import raise_if_cancelled
from mtg_bot.search.determinize import (
    determinize as _default_determinize, determinized_clone,
)
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.protocol import Decision, GameState, PlayerID
from mtg_bot.search.strategy import SearchStrategy


class _VisitsProducer(Protocol):
    """The slice of MCTS PIMC actually needs: produce a per-decision
    visit-count distribution at the root. Anything implementing this
    can be plugged in as ``inner_factory``."""

    def choose_with_visits(
        self, state: GameState, player_id: PlayerID
    ) -> dict[Decision, int]:
        ...


class PIMC(SearchStrategy):
    """Determinize K times, run an inner search per world, aggregate
    visit counts, pick argmax.

    ``inner_factory`` is a zero-arg callable returning a fresh inner
    search per world. Producing fresh ones (rather than reusing one)
    matters because MCTS holds its tree across the call's lifetime —
    sharing would entangle worlds.

    ``samples`` is K. K=1 is the cheapest sound option (one
    determinized world, no aggregation). K=8–16 is the standard PIMC
    range; higher K trades sub-search depth for sample diversity.
    Pick K from harness numbers, not gut.
    """

    def __init__(
        self,
        inner_factory: Callable[[], _VisitsProducer],
        samples: int = 8,
        rng: Optional[random.Random] = None,
        determinize_fn: Callable = _default_determinize,
    ) -> None:
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")
        self.inner_factory = inner_factory
        self.samples = samples
        self.rng = rng or random.Random()
        self._determinize = determinize_fn

    def choose(
        self, state: GameState, player_id: PlayerID
    ) -> Decision:
        visits = self.choose_with_visits(state, player_id)
        if not visits:
            # Every sub-search bottomed out (likely budget too small).
            # Fall back to a legal default rather than raising.
            return list(state.legal_decisions())[0]
        return max(visits.items(), key=lambda kv: kv[1])[0]

    def choose_with_visits(
        self, state: GameState, player_id: PlayerID
    ) -> dict[Decision, int]:
        """Run PIMC and return aggregated visit counts per decision,
        summed across all K sub-searches.

        Used by the recommendation panel to produce a ranked list of
        actions with relative scores (each decision's share of total
        visits is its score). ``choose()`` is just the argmax wrapper
        around this. Returns an empty dict only if every sub-search
        bottomed out — callers should fall back to a legal default.

        Forced-move shortcut: when only one legal decision exists,
        returns ``{decision: 1}`` without spinning up sub-searches.
        The synthetic count is a placeholder so callers' aggregation
        logic doesn't need a special case.
        """
        if not isinstance(state, MtgGameStateAdapter):
            raise TypeError(
                "PIMC requires an MtgGameStateAdapter (state must wrap "
                f"a real engine GameState); got {type(state).__name__}. "
                "Use MCTS directly for non-MTG game states."
            )
        # Forced-move shortcut at the PIMC layer too. Avoids spinning
        # up K sub-searches when there's only one legal action.
        legal = list(state.legal_decisions())
        if len(legal) == 1:
            return {legal[0]: 1}

        visit_totals: dict[Decision, int] = {}
        for _ in range(self.samples):
            # Cooperative stop for an abandoned (timed-out) game —
            # see mtg_bot.cancellation.
            raise_if_cancelled()
            # Independent rng per sub-search so each sub-tree is
            # reproducible from the outer seed alone.
            sub_seed = self.rng.randrange(1 << 62)
            adapter = self._build_sub_adapter(state, player_id, sub_seed)
            inner = self.inner_factory()
            for decision, visits in inner.choose_with_visits(
                adapter, player_id
            ).items():
                visit_totals[decision] = (
                    visit_totals.get(decision, 0) + visits
                )
        return visit_totals

    def choose_with_full_stats(
        self, state: GameState, player_id: PlayerID,
    ) -> tuple[Optional[Decision], list[dict]]:
        """PIMC equivalent of ``MCTS.choose_with_full_stats``: run K
        determinized sub-searches and aggregate the per-root-child
        stats so the replay viewer can show what the search saw.

        Aggregation rules:

        - ``visits`` summed across sub-searches.
        - ``q`` is the visit-weighted mean of the per-sub-search Q
          values (sub-searches with visits==0 on a decision contribute
          nothing). None when the decision had no visits anywhere.
        - ``prior`` is the visit-weighted mean of the per-sub-search
          PUCT prior. None when no sub-search reported a prior (UCB1
          mode, every world). Different determinized worlds can hand
          the policy head slightly different states, so the means
          differ; the viewer just needs a single comparable scalar
          per candidate.
        - ``leaf_value`` is the visit-weighted mean of the per-sub-
          search ``leaf_nn_value`` (the NN value at the child's
          state on its first leaf evaluation). None when no
          sub-search recorded one.

        Returns ``(chosen_decision_or_None, candidates)``. Forced
        moves take the same single-element shape as
        ``MCTS.choose_with_full_stats``.
        """
        if not isinstance(state, MtgGameStateAdapter):
            raise TypeError(
                "PIMC requires an MtgGameStateAdapter; got "
                f"{type(state).__name__}."
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
                    }
                ],
            )

        # Per-decision running aggregates.
        agg_visits: dict[Decision, int] = {}
        agg_q_num: dict[Decision, float] = {}      # sum(q_i * visits_i)
        agg_q_den: dict[Decision, int] = {}        # sum(visits_i where q_i present)
        agg_prior_num: dict[Decision, float] = {}
        agg_prior_den: dict[Decision, int] = {}
        agg_leaf_num: dict[Decision, float] = {}
        agg_leaf_den: dict[Decision, int] = {}
        for _ in range(self.samples):
            # Cooperative stop for an abandoned (timed-out) game —
            # see mtg_bot.cancellation.
            raise_if_cancelled()
            sub_seed = self.rng.randrange(1 << 62)
            adapter = self._build_sub_adapter(state, player_id, sub_seed)
            inner = self.inner_factory()
            chosen_fn = getattr(inner, "choose_with_full_stats", None)
            if chosen_fn is None:
                # Fallback: inner doesn't expose the richer interface,
                # so synthesize candidates from visits alone (the only
                # universal contract). Used by user-supplied inners.
                visits = inner.choose_with_visits(adapter, player_id)
                sub_candidates = [
                    {
                        "decision": d, "visits": v, "q": None,
                        "prior": None, "leaf_value": None,
                    }
                    for d, v in visits.items()
                ]
            else:
                _, sub_candidates = chosen_fn(adapter, player_id)
            for c in sub_candidates:
                d = c["decision"]
                v = int(c["visits"])
                agg_visits[d] = agg_visits.get(d, 0) + v
                if c.get("q") is not None and v > 0:
                    agg_q_num[d] = (
                        agg_q_num.get(d, 0.0) + float(c["q"]) * v
                    )
                    agg_q_den[d] = agg_q_den.get(d, 0) + v
                if c.get("prior") is not None and v > 0:
                    agg_prior_num[d] = (
                        agg_prior_num.get(d, 0.0)
                        + float(c["prior"]) * v
                    )
                    agg_prior_den[d] = agg_prior_den.get(d, 0) + v
                if c.get("leaf_value") is not None and v > 0:
                    agg_leaf_num[d] = (
                        agg_leaf_num.get(d, 0.0)
                        + float(c["leaf_value"]) * v
                    )
                    agg_leaf_den[d] = agg_leaf_den.get(d, 0) + v
        if not agg_visits:
            return (None, [])
        candidates = []
        for d, total in agg_visits.items():
            candidates.append({
                "decision": d,
                "visits": total,
                "q": (
                    agg_q_num[d] / agg_q_den[d]
                    if agg_q_den.get(d) else None
                ),
                "prior": (
                    agg_prior_num[d] / agg_prior_den[d]
                    if agg_prior_den.get(d) else None
                ),
                "leaf_value": (
                    agg_leaf_num[d] / agg_leaf_den[d]
                    if agg_leaf_den.get(d) else None
                ),
            })
        chosen = max(candidates, key=lambda c: c["visits"])["decision"]
        return (chosen, candidates)

    def choose_with_subtree(
        self,
        state: GameState,
        player_id: PlayerID,
        tree_depth: int = 1,
        tree_top_k: Optional[int] = None,
    ) -> tuple[Optional[Decision], list[dict]]:
        """PIMC equivalent of ``MCTS.choose_with_subtree``: run K
        determinized sub-searches and aggregate the per-level
        candidate stats so the diagnostic replay viewer can drill
        into the search tree under any root candidate.

        Aggregation rules at every level mirror ``choose_with_full_stats``:
        visits sum, q/prior/leaf_value are visit-weighted means
        across the sub-searches that visited the same decision.
        Sub-trees are merged by recursively aggregating the per-decision
        children lists, then sorting by visits desc and slicing to
        ``tree_top_k`` (None = keep all). The root level keeps ALL
        candidates regardless of ``tree_top_k`` — the data-gen policy
        encoder needs the full visit distribution. Top-K pruning kicks
        in at depth ≥ 1.

        ``tree_depth=1`` reproduces ``choose_with_full_stats`` exactly
        with ``subtree=[]`` on every candidate.

        Forced moves: same single-element shape as
        ``choose_with_full_stats``, with ``subtree=[]``.
        """
        if tree_depth < 1:
            raise ValueError(
                f"tree_depth must be >= 1, got {tree_depth!r}"
            )
        if not isinstance(state, MtgGameStateAdapter):
            raise TypeError(
                "PIMC requires an MtgGameStateAdapter; got "
                f"{type(state).__name__}."
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

        per_sub_candidates: list[list[dict]] = []
        for _ in range(self.samples):
            # Cooperative stop for an abandoned (timed-out) game —
            # see mtg_bot.cancellation.
            raise_if_cancelled()
            sub_seed = self.rng.randrange(1 << 62)
            adapter = self._build_sub_adapter(state, player_id, sub_seed)
            inner = self.inner_factory()
            sub_fn = getattr(inner, "choose_with_subtree", None)
            if sub_fn is None:
                # Inner doesn't expose the subtree-aware interface —
                # fall back to ``choose_with_full_stats`` (or visits)
                # and synthesize an empty subtree per candidate.
                full_fn = getattr(inner, "choose_with_full_stats", None)
                if full_fn is not None:
                    _, sub_candidates = full_fn(adapter, player_id)
                else:
                    visits = inner.choose_with_visits(adapter, player_id)
                    sub_candidates = [
                        {
                            "decision": d, "visits": v, "q": None,
                            "prior": None, "leaf_value": None,
                        }
                        for d, v in visits.items()
                    ]
                # Stamp empty subtree so the aggregator's recursion
                # sees a uniform shape.
                sub_candidates = [
                    {**c, "subtree": []} for c in sub_candidates
                ]
            else:
                _, sub_candidates = sub_fn(
                    adapter, player_id,
                    tree_depth=tree_depth, tree_top_k=tree_top_k,
                )
            per_sub_candidates.append(sub_candidates)

        candidates = _aggregate_subtree_candidates(
            per_sub_candidates,
            top_k=tree_top_k,
            is_root=True,
        )
        if not candidates:
            return (None, [])
        chosen = max(candidates, key=lambda c: c["visits"])["decision"]
        return (chosen, candidates)

    def _build_sub_adapter(
        self, source: MtgGameStateAdapter, viewer_id: PlayerID, sub_seed: int,
    ) -> MtgGameStateAdapter:
        """Make a fresh sub-search adapter for one PIMC sample.

        Crucially: ``source.clone()`` (not ``MtgGameStateAdapter(...)``)
        because the latter's ``__init__`` calls ``_advance_to_next_decision``
        which re-runs the current step's pre-priority logic — re-firing
        upkeep triggers, re-drawing the active player's card at DRAW,
        re-firing beginning-of-end-step warp-exile, etc. Cloning
        preserves the source's ``_step_entered``, ``_pass_count``, and
        combat-phase bookkeeping so the sub-search starts from exactly
        the state the source was at (modulo determinization).

        After cloning, we determinize the cloned engine state in place
        — re-cloning would be wasteful and the clone() already paid
        for one.
        """
        # At a resolution pause this also pins the cards the decider can
        # see, re-takes the pre-action snapshot from the determinized
        # state (no rollback to the true state) and checks the world
        # replays to the same pause (``determinized_clone``).
        return determinized_clone(
            source, viewer_id, random.Random(sub_seed), self._determinize,
        )


def _aggregate_subtree_candidates(
    per_sub_candidates: list[list[dict[str, Any]]],
    top_k: Optional[int],
    is_root: bool,
) -> list[dict[str, Any]]:
    """Merge K parallel candidate lists (one per PIMC sub-search)
    into a single per-decision list, recursively aggregating each
    decision's subtree.

    For every distinct ``decision`` seen across the K input lists,
    we:

    - Sum visits.
    - Compute a visit-weighted mean of q / prior / leaf_value over
      sub-searches that observed the decision (visits > 0 and the
      field present). None when no sub-search recorded the field.
    - Recursively aggregate the per-sub-search ``subtree`` lists for
      that same decision into a single subtree.

    After merging, sort by visits desc. At ``is_root=True`` we keep
    every decision; at deeper levels we slice to ``top_k`` (None =
    keep all).
    """
    by_decision: dict[Any, dict[str, Any]] = {}
    sub_lists_by_decision: dict[Any, list[list[dict[str, Any]]]] = {}
    for sub_candidates in per_sub_candidates:
        for c in sub_candidates:
            d = c["decision"]
            v = int(c["visits"])
            entry = by_decision.setdefault(
                d,
                {
                    "decision": d,
                    "visits": 0,
                    "q_num": 0.0, "q_den": 0,
                    "prior_num": 0.0, "prior_den": 0,
                    "leaf_num": 0.0, "leaf_den": 0,
                },
            )
            entry["visits"] += v
            if c.get("q") is not None and v > 0:
                entry["q_num"] += float(c["q"]) * v
                entry["q_den"] += v
            if c.get("prior") is not None and v > 0:
                entry["prior_num"] += float(c["prior"]) * v
                entry["prior_den"] += v
            if c.get("leaf_value") is not None and v > 0:
                entry["leaf_num"] += float(c["leaf_value"]) * v
                entry["leaf_den"] += v
            sub_lists_by_decision.setdefault(d, []).append(
                list(c.get("subtree") or []),
            )

    candidates: list[dict[str, Any]] = []
    for d, entry in by_decision.items():
        merged_sub = _aggregate_subtree_candidates(
            sub_lists_by_decision.get(d, []),
            top_k=top_k,
            is_root=False,
        )
        candidates.append({
            "decision": d,
            "visits": entry["visits"],
            "q": (
                entry["q_num"] / entry["q_den"]
                if entry["q_den"] else None
            ),
            "prior": (
                entry["prior_num"] / entry["prior_den"]
                if entry["prior_den"] else None
            ),
            "leaf_value": (
                entry["leaf_num"] / entry["leaf_den"]
                if entry["leaf_den"] else None
            ),
            "subtree": merged_sub,
        })

    candidates.sort(key=lambda c: c["visits"], reverse=True)
    if not is_root and top_k is not None:
        candidates = candidates[:top_k]
    return candidates
