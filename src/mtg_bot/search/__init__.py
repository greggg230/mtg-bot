"""Tree-search + evaluator decision-making framework.

Organized so that the search code is **ignorant of MTG** and talks only
to the abstract ``GameState`` protocol in ``protocol``. This lets us
validate search correctness on toy games (tic-tac-toe, a chance-toy
stand-in for card draw) before wiring it up against the real MTG
engine — per the build plan agreed with the user.

Module map:

- ``protocol``   — ``GameState`` Protocol, ``Decision`` alias, ``PlayerID``.
- ``evaluator``  — ``Evaluator`` ABC (state → scalar in ``[-1, 1]`` from
                   a player's perspective).
- ``budget``     — ``Budget`` dataclass (wall-clock or node-count cap
                   on a single search call).
- ``strategy``   — ``SearchStrategy`` ABC + ``RandomChoice`` baseline.
- ``mcts``       — ``MCTS`` implementation with UCB1 selection, chance-
                   node sampling, and evaluator-at-leaf. Also exposes
                   ``choose_with_visits`` for ensemble wrappers.
- ``ismcts``     — ``ISMCTS``: Single-Observer Information Set MCTS.
                   Single tree keyed by information-set hash; per-
                   iteration determinization during selection.
- ``determinize`` — ``determinize(state, viewer_id, rng)`` root sampler.
                   See ``docs/HIDDEN_INFO_MODEL.md``.
- ``expansion``  — ``ExpansionPolicy`` ABC + ``FullEnumeration`` (classic)
                   and ``ProgressiveWidening`` (budget-aware) — pluggable
                   into MCTS for tunable tree growth.
- ``harness``    — Agent-vs-agent match runner + win-rate stats.
- ``toys.*``     — Tiny self-contained ``GameState`` implementations
                   (tic-tac-toe, TTT-with-chance) used to exercise
                   search on known-good games.
- ``tactical``   — ``DefaultTacticalAgent``: deterministic policy
                   used by the real-engine adapter for mid-resolution
                   + combat callbacks.
- ``engine_adapter`` — ``MtgGameStateAdapter``: wraps a real
                   ``mtg_bot.game.GameState`` to implement the
                   search ``GameState`` protocol.
- ``combat_decisions`` — ``AttackerDecision`` / ``BlockerDecision``
                   search-side decision types for factored combat
                   when the adapter is opt-in for combat exposure.

Search code (protocol, evaluator, strategy, mcts, harness, toys)
remains ignorant of MTG — only ``engine_adapter`` and ``tactical``
import from the engine side.
"""

from __future__ import annotations

from mtg_bot.search.budget import Budget
from mtg_bot.search.combat_decisions import (
    AttackerDecision, BlockerDecision,
)
from mtg_bot.search.determinize import determinize, viewer_knows
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.resolution_channel import ResolutionChannelAgent
from mtg_bot.search.resolution_decisions import (
    ResolutionCardFromLibraryDecision,
    ResolutionCardFromOpponentHandDecision,
    ResolutionDevourLandsDecision,
    ResolutionDiscardDecision,
    ResolutionModeDecision,
    ResolutionSacrificeDecision,
    ResolutionShockLifeDecision,
    ResolutionSurveilDecision,
    ResolutionTargetDecision,
    ResolutionXDecision,
    ResolutionYesNoDecision,
)
from mtg_bot.search.evaluator import (
    Evaluator,
    RandomRolloutEvaluator,
)
from mtg_bot.search.expansion import (
    ExpansionPolicy,
    FullEnumeration,
    ProgressiveWidening,
)
from mtg_bot.search.harness import MatchResult, play_one_game, run_match
from mtg_bot.search.heuristic_evaluator import MtgHeuristicEvaluator
from mtg_bot.search.ismcts import ISMCTS
from mtg_bot.search.mcts import MCTS
from mtg_bot.search.nn_evaluator import NNEvaluator
from mtg_bot.search.mtg_priors import (
    attack_aggressive_prior,
    block_biggest_attacker_first_prior,
    mtg_default_prior,
    priority_action_prior,
)
from mtg_bot.search.protocol import (
    CHANCE_PLAYER,
    Decision,
    GameState,
    PlayerID,
)
from mtg_bot.search.strategy import RandomChoice, SearchStrategy
from mtg_bot.search.tactical import DefaultTacticalAgent

__all__ = [
    "AttackerDecision",
    "BlockerDecision",
    "Budget",
    "CHANCE_PLAYER",
    "Decision",
    "DefaultTacticalAgent",
    "Evaluator",
    "ExpansionPolicy",
    "FullEnumeration",
    "GameState",
    "ISMCTS",
    "MCTS",
    "MatchResult",
    "MtgGameStateAdapter",
    "MtgHeuristicEvaluator",
    "NNEvaluator",
    "PlayerID",
    "ProgressiveWidening",
    "RandomChoice",
    "RandomRolloutEvaluator",
    "ResolutionCardFromLibraryDecision",
    "ResolutionCardFromOpponentHandDecision",
    "ResolutionChannelAgent",
    "ResolutionDevourLandsDecision",
    "ResolutionDiscardDecision",
    "ResolutionModeDecision",
    "ResolutionSacrificeDecision",
    "ResolutionShockLifeDecision",
    "ResolutionSurveilDecision",
    "ResolutionTargetDecision",
    "ResolutionXDecision",
    "ResolutionYesNoDecision",
    "SearchStrategy",
    "attack_aggressive_prior",
    "block_biggest_attacker_first_prior",
    "determinize",
    "mtg_default_prior",
    "play_one_game",
    "priority_action_prior",
    "run_match",
    "viewer_knows",
]
