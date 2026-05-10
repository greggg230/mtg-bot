"""Search strategies: produce a ``Decision`` given a ``GameState``.

``SearchStrategy`` is the top-level interface. ``RandomChoice`` is the
trivial baseline that every real strategy must beat to justify its
existence.

MCTS lives in its own module (``mcts``) because it's substantially
bigger and depends on ``Evaluator`` + ``Budget``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from mtg_bot.search.protocol import Decision, GameState, PlayerID


class SearchStrategy(ABC):
    """Produce a decision for ``player_id`` at ``state``.

    The strategy is free to clone state, run rollouts, consult
    evaluators, etc. Must NOT mutate the caller's ``state``.
    """

    @abstractmethod
    def choose(
        self, state: GameState, player_id: PlayerID
    ) -> Decision:
        """Return one of ``state.legal_decisions()`` for
        ``player_id``. Caller's ``state`` must be unchanged on return.
        """

    def choose_with_visits(
        self, state: GameState, player_id: PlayerID
    ) -> dict[Decision, int]:
        """Return a per-decision visit-count distribution at the
        root.

        For MCTS-family strategies (``MCTS``, ``PIMC``) this is the
        actual search-tree visit count, and ``choose()`` is a thin
        argmax wrapper around this method's result. For strategies
        that don't have a notion of visit counts (``RandomChoice``,
        toy openers), the default returns an empty dict — the
        AlphaZero-style data-gen pipeline treats an empty dict as
        "no policy supervision available for this sample" and skips
        recording a policy target.

        Subclasses with real visit data SHOULD override this so the
        AlphaZero-style training loop can capture per-state policy
        targets. Visit-count distributions are the supervision
        signal for the policy head, much richer than the per-game
        outcome label the value head uses.

        Signature note: this method's return shape (just the visit
        dict, NOT a (decision, visits) tuple) matches the existing
        ``MCTS.choose_with_visits`` and ``PIMC.choose_with_visits``
        contracts. Callers that want the chosen decision too should
        call ``choose()`` separately or argmax over the dict.
        Sites that need both should call this method ONCE (to avoid
        re-running the search) and argmax themselves — see
        ``data_gen._play_one_game_recording`` for the canonical
        pattern.
        """
        return {}

    def choose_with_full_stats(
        self, state: GameState, player_id: PlayerID,
    ):
        """Return ``(chosen_decision_or_None, candidates)`` where
        ``candidates`` is a list of dicts shaped ``{"decision",
        "visits", "q", "prior", "leaf_value"}``. Used by the
        replay capture pipeline to attach a search-tree summary
        annotation to each captured snapshot, populating the replay
        viewer's decision-detail panel.

        Default implementation derives the candidates from
        ``choose_with_visits`` with ``q``, ``prior``, and
        ``leaf_value`` all None — strategies without real search
        machinery (RandomChoice, scripted openers) still produce a
        valid annotation, just without per-candidate diagnostics.
        Returns ``(None, [])`` when no decisions could be derived.

        MCTS / PIMC / ISMCTS override to populate the richer fields
        from their internal tree state without re-running the
        search.
        """
        visits = self.choose_with_visits(state, player_id)
        if not visits:
            return (None, [])
        candidates = [
            {
                "decision": d, "visits": v, "q": None,
                "prior": None, "leaf_value": None,
            }
            for d, v in visits.items()
        ]
        chosen = max(candidates, key=lambda c: c["visits"])["decision"]
        return (chosen, candidates)

    def choose_with_subtree(
        self,
        state: GameState,
        player_id: PlayerID,
        tree_depth: int = 1,
        tree_top_k=None,
    ):
        """Return ``(chosen, candidates)`` where each candidate has
        an extra ``"subtree"`` key — a recursive list of grandchild
        candidates the search would visit, up to ``tree_depth``
        levels deep. Used by the diagnostic deep-capture path.

        Default implementation reuses ``choose_with_full_stats`` and
        stamps an empty ``subtree`` on every candidate — strategies
        that don't expose a real search tree (RandomChoice etc.)
        get a valid shape with no nested data. ``tree_depth`` /
        ``tree_top_k`` are accepted but ignored at this fallback.
        MCTS / PIMC override with a real recursive walk.
        """
        chosen, candidates = self.choose_with_full_stats(state, player_id)
        return (
            chosen,
            [{**c, "subtree": []} for c in candidates],
        )


class RandomChoice(SearchStrategy):
    """Pick uniformly among legal decisions. Zero-heuristic baseline.

    The yardstick every smarter strategy must beat in head-to-head
    matches.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def choose(
        self, state: GameState, player_id: PlayerID
    ) -> Decision:
        legal = state.legal_decisions()
        if not legal:
            raise ValueError(
                "RandomChoice called on a state with no legal decisions"
            )
        return self.rng.choice(legal)
