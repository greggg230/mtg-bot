"""Head-isolation evaluators for the strength ladder (2026-10-01).

Every s2 ladder game pits ``NNEvaluator`` (value head + policy-prior
PUCT) against ``MtgHeuristicEvaluator`` (static value, UCB1). When the
net loses we cannot tell *which* head is at fault. Two agents split it:

* ``value-only``: the NN's value head driving a plain UCB1 search --
  the heuristic's search with the net's evaluation. Built with the
  ordinary :class:`NNPimcConfig` (``puct=False``); nothing here.
* ``heur-value``: :class:`HeuristicValueNNPolicyEvaluator` -- the
  heuristic's static value with the net's policy priors steering PUCT.
  :class:`HybridPimcConfig` wires it into PIMC.

If ``value-only`` beats the heuristic while ``heur-value`` does not,
the policy priors are the weak head; if it is the reverse, the value
head is. Both tie the plain net -> the search dynamics, not the heads.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from mtg_bot.search.budget import Budget
from mtg_bot.search.evaluator import Evaluator
from mtg_bot.search.mcts import MCTS
from mtg_bot.search.pimc import PIMC
from mtg_bot.search.strategy import SearchStrategy


class HeuristicValueNNPolicyEvaluator(Evaluator):
    """Value from one evaluator, policy priors from another.

    ``value_evaluator`` is any :class:`Evaluator` (the heuristic in
    the ladder experiment). ``policy_evaluator`` must expose
    ``evaluate_with_policy`` / ``batch_evaluate_with_policy`` and an
    ``action_encoding_version`` (``NNEvaluator(with_policy=True)``).
    The MCTS reads ``action_encoding_version`` off the evaluator to
    map decisions onto policy slots, so it is forwarded here.
    """

    with_policy = True

    def __init__(self, value_evaluator: Evaluator, policy_evaluator: Any):
        if not callable(getattr(policy_evaluator, "evaluate_with_policy", None)):
            raise ValueError(
                "policy_evaluator must expose evaluate_with_policy(); got "
                f"{type(policy_evaluator).__name__}"
            )
        self.value_evaluator = value_evaluator
        self.policy_evaluator = policy_evaluator
        self.action_encoding_version = getattr(
            policy_evaluator, "action_encoding_version", "positional_v1",
        )

    def _evaluate_nonterminal(self, state, player_id) -> float:
        return self.value_evaluator.evaluate(state, player_id)

    def batch_evaluate(self, states: Sequence, player_id) -> list[float]:
        return self.value_evaluator.batch_evaluate(states, player_id)

    def evaluate_with_policy(self, state, player_id) -> tuple[float, np.ndarray]:
        # ``evaluate`` (base class) handles the terminal short-circuit;
        # the policy evaluator returns the all-zeros vector at terminals.
        _nn_value, policy = self.policy_evaluator.evaluate_with_policy(
            state, player_id,
        )
        return self.evaluate(state, player_id), policy

    def batch_evaluate_with_policy(
        self, states: Sequence, player_id,
    ) -> list[tuple[float, np.ndarray]]:
        values = self.batch_evaluate(states, player_id)
        pairs = self.policy_evaluator.batch_evaluate_with_policy(
            states, player_id,
        )
        return [(float(v), p) for v, (_nn_v, p) in zip(values, pairs)]


@dataclass
class HybridPimcConfig:
    """PIMC whose inner MCTS runs PUCT on the NN policy with the
    heuristic's value. ``nn`` must be a policy-capable
    :class:`NNPimcConfig` (``with_policy=True``); its ``weights_path``,
    trunk discovery and ``batch_size`` are reused verbatim by building
    it once and lifting the evaluator out of the inner MCTS, so the
    structured-metadata plumbing lives in exactly one place.
    Picklable (dataclass of dataclasses) so ``build`` ships across the
    ladder's ``mp.Pool``.
    """

    nn: Any  # NNPimcConfig (imported lazily to avoid a training import cycle)
    heuristic: Any = None  # HeuristicConfig; None -> defaults
    samples: int = 2
    inner_nodes: int = 100
    root_dirichlet_alpha: float | None = None
    root_dirichlet_eps: float = 0.25

    def __post_init__(self) -> None:
        if not getattr(self.nn, "with_policy", False):
            raise ValueError(
                "HybridPimcConfig.nn must have with_policy=True (the "
                "policy priors come from the NN)"
            )

    def build_evaluator(self) -> HeuristicValueNNPolicyEvaluator:
        from mtg_bot.training import HeuristicConfig
        nn_pimc = self.nn.build(0)
        nn_eval = nn_pimc.inner_factory().evaluator
        heur = (self.heuristic or HeuristicConfig()).build()
        return HeuristicValueNNPolicyEvaluator(heur, nn_eval)

    def build(self, seed: int = 0) -> SearchStrategy:
        evaluator = self.build_evaluator()
        inner_nodes = self.inner_nodes
        batch_size = int(getattr(self.nn, "batch_size", 1))
        alpha, eps = self.root_dirichlet_alpha, self.root_dirichlet_eps
        rng_seed_inner = seed + 100
        rng_seed_outer = seed + 200

        def inner_factory() -> MCTS:
            return MCTS(
                evaluator=evaluator,
                budget_factory=lambda: Budget(max_nodes=inner_nodes),
                rng=random.Random(rng_seed_inner),
                batch_size=batch_size,
                puct=True,
                root_dirichlet_alpha=alpha,
                root_dirichlet_eps=eps,
            )

        from mtg_bot.search.determinize import make_determinize_fn
        return PIMC(
            inner_factory=inner_factory,
            samples=self.samples,
            rng=random.Random(rng_seed_outer),
            # The opponent model rides on the NN config (``+boardonly``).
            determinize_fn=make_determinize_fn(
                getattr(self.nn, "opponent_hidden_model", "sampled"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "hybrid_heur_value_nn_policy",
            "nn": self.nn.to_dict(),
            "heuristic": (
                self.heuristic.to_dict() if self.heuristic is not None else None
            ),
            "samples": self.samples,
            "inner_nodes": self.inner_nodes,
            "root_dirichlet_alpha": self.root_dirichlet_alpha,
            "root_dirichlet_eps": self.root_dirichlet_eps,
        }
