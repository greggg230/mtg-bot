"""Search / evaluator configs as serializable dataclasses.

Each config is a pure data description that can be:
- Serialized to JSON (for storing tuning results, versioning
  evaluators, comparing runs).
- Built into a live ``SearchStrategy`` via ``build(seed)``.

Frozen so configs are hashable (usable as dict keys, set members)
— important for grid-search loops that want to deduplicate or
look up "have we measured this config yet?"

The build pattern keeps strategy construction in one place. A
tuning loop can produce hundreds of configs and just call
``cfg.build(seed)`` per game without the loop knowing anything
about MCTS / PIMC / evaluator wiring.

Picklability matters: ``run_match_parallel`` ships configs across
``multiprocessing.Pool`` worker boundaries (spawn on Windows,
fork on Unix). Frozen dataclasses + their bound ``.build``
methods pickle cleanly. Closures over local variables do NOT —
that's why every per-game configuration lives in a config class
here, not in an inline closure at the call site.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from mtg_bot.search import (
    Budget,
    ISMCTS,
    MCTS,
    PIMC,
    RandomChoice,
)
from mtg_bot.search.evaluator import Evaluator
from mtg_bot.search.heuristic_evaluator import MtgHeuristicEvaluator
from mtg_bot.search.nn_eval_server import NNEvalClient, NNServerHandle
from mtg_bot.search.nn_evaluator import NNEvaluator
from mtg_bot.search.strategy import SearchStrategy


@dataclass(frozen=True)
class HeuristicConfig:
    """Weights for ``MtgHeuristicEvaluator``. Defaults match the
    evaluator's own defaults (0.4 / 0.4 / 0.15 / 0.05)."""

    life_weight: float = 0.4
    board_weight: float = 0.4
    card_weight: float = 0.15
    mana_dev_weight: float = 0.05

    def build(self) -> Evaluator:
        return MtgHeuristicEvaluator(
            life_weight=self.life_weight,
            board_weight=self.board_weight,
            card_weight=self.card_weight,
            mana_dev_weight=self.mana_dev_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeuristicConfig":
        return cls(**d)


@dataclass(frozen=True)
class PimcConfig:
    """Full PIMC search config: K samples, MCTS budget per sub-search,
    and the heuristic-evaluator weights.

    Defaults match ``recommend.py``: samples=4, max_nodes=200, default
    heuristic weights. That's the UI's recommendation budget — slow
    enough to actually search, fast enough to stay responsive.
    Tuning loops typically dial samples + nodes down to keep games
    cheap (~3-6s instead of ~10-15s)."""

    samples: int = 4
    inner_nodes: int = 200
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    batch_size: int = 1

    def build(self, seed: int = 0) -> SearchStrategy:
        """Construct a fresh PIMC search strategy seeded for
        reproducibility. Different ``seed`` values give different
        sub-search RNGs."""
        evaluator = self.heuristic.build()
        inner_nodes = self.inner_nodes
        batch_size = self.batch_size
        rng_seed_inner = seed + 100
        rng_seed_outer = seed + 200

        def inner_factory() -> MCTS:
            return MCTS(
                evaluator=evaluator,
                budget_factory=lambda: Budget(max_nodes=inner_nodes),
                rng=random.Random(rng_seed_inner),
                batch_size=batch_size,
            )

        return PIMC(
            inner_factory=inner_factory,
            samples=self.samples,
            rng=random.Random(rng_seed_outer),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "inner_nodes": self.inner_nodes,
            "heuristic": self.heuristic.to_dict(),
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PimcConfig":
        return cls(
            samples=d["samples"],
            inner_nodes=d["inner_nodes"],
            heuristic=HeuristicConfig.from_dict(d["heuristic"]),
            batch_size=d.get("batch_size", 1),
        )


@dataclass(frozen=True)
class NNPimcConfig:
    """Full PIMC config backed by a saved NN evaluator. Carries a
    ``weights_path`` rather than an evaluator instance because
    ``NNEvaluator`` (and the underlying ``torch.nn.Module``) is not
    designed for cross-process pickling — the evaluator is built
    fresh inside ``build`` from the bytes on disk, which always
    works.

    ``batch_size`` defaults to 32 to capture the B1 batched-leaf-
    evaluation win on the value-only NN forward pass.

    AlphaZero opt-in:

    - ``with_policy=True``: load the saved weights as a dual-head
      ``_ValuePolicyMLP``. Required when the saved weights came
      from an A4-wiring training run (registry's
      ``metadata.json["with_policy"]`` is True for those).
    - ``puct=True``: use PUCT selection in MCTS, treating the
      policy head's softmax as a per-action prior. Pair with
      ``with_policy=True`` for AlphaZero-style search. After
      ``feature/batched-puct`` landed, PUCT supports
      ``batch_size>1`` — leaves are batched into one
      ``batch_evaluate_with_policy`` call, with each leaf's policy
      stashed on the per-node cache. ``puct=True`` without
      ``with_policy`` is rejected at MCTS construction (value-only
      model can't produce a policy prior).
    """

    weights_path: str
    samples: int = 4
    inner_nodes: int = 200
    batch_size: int = 32
    with_policy: bool = False
    puct: bool = False
    # ``hidden_sizes`` carries the trunk shape the saved weights were
    # trained at. ``None`` = use NNEvaluator's default (the registry-
    # baseline (128, 128, 64)); workers loading a non-default-arch
    # version (e.g. v7's wider 1024-1024-512 trunk) must set this
    # explicitly so the model the state_dict loads into has the
    # matching shape. ``evaluate_nn.py`` reads it from the version's
    # metadata.json and threads it through this field.
    hidden_sizes: Optional[tuple[int, ...]] = None
    # ``architecture`` carries which dual-head model class the saved
    # weights were trained under: ``"shared"`` (default, back-compat
    # with v1-v15) or ``"separate"`` (v16+ separate-trunks runs). The
    # workers' NNEvaluator needs to instantiate the matching class so
    # the state_dict load doesn't fail with a key-mismatch error.
    # Ignored when ``with_policy=False`` (the value-only ``_ValueMLP``
    # has no architectural choice). Iterate / evaluate scripts read
    # this from the version's metadata.json and thread it through.
    architecture: Optional[str] = None
    # ``policy_hidden_sizes`` carries the ``policy_trunk`` shape for
    # ``architecture="separate"`` runs trained with an asymmetric
    # ``--policy-hidden-sizes``. ``None`` mirrors ``hidden_sizes``,
    # which is what the symmetric default uses.
    policy_hidden_sizes: Optional[tuple[int, ...]] = None
    # ``action_encoding_version`` carries the slot-mapping convention
    # the saved weights were trained against. ``None`` mirrors
    # NNEvaluator's "positional_v1" default — back-compat with v13/v15/
    # v16 which trained against the positional-by-card.id mapping.
    # Iterate / evaluate scripts read this from the version's
    # metadata.json and thread it through.
    action_encoding_version: Optional[str] = None
    # When True, ``build`` returns ISMCTS instead of PIMC. The class
    # name is historical — kept for serialization compatibility with
    # in-flight specs and saved JSON. Matched-budget conversion:
    # ``total_iterations = samples * inner_nodes`` (same total search
    # work as PIMC's K independent searches of N nodes each).
    use_ismcts: bool = False
    # When True, the ISMCTS ``determinize_fn`` callback is wired to
    # ``BeliefWeightedDeterminizer`` (Phase 6) rather than the uniform
    # sampler. Requires ``use_ismcts=True``; rejected at __post_init__
    # otherwise — belief weighting is meaningless for PIMC's K-
    # independent-trees structure (no shared search state to bias).
    # The determinizer is constructed with an empty played_history;
    # the game loop populates it via ``ISMCTS.observe_action`` calls
    # as the game progresses (see Phase 6.5 wireup in
    # ``data_gen._play_one_game_recording`` and
    # ``self_play._play_capped_game``).
    use_belief_weighted: bool = False
    # Number of particles in the belief-weighted determinizer. Higher
    # = better posterior approximation, linearly higher cold-call
    # cost. Default 64. Ignored unless ``use_belief_weighted=True``.
    belief_m_candidates: int = 64
    # Conditional-activation gate. Skip the belief scoring pass and
    # fall back to uniform determinization until the played history
    # has at least this many opponent (non-viewer) priority moments.
    # Default 3. Set to 0 to disable.
    belief_min_opponent_moments: int = 3
    # Torch device for the underlying ``NNEvaluator``. ``None`` means
    # "use NNEvaluator's default" (currently CPU) — preserves the
    # production data_gen / self-play default of CPU-only workers
    # (per the CLAUDE.md note: multiprocessing workers stay CPU-only
    # to avoid creating one CUDA context per worker process). The
    # bench may opt into ``"cuda"`` for matched-budget runs where the
    # batched-leaf NN forward dominates the per-chunk wall-clock.
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.use_belief_weighted and not self.use_ismcts:
            raise ValueError(
                "use_belief_weighted=True requires use_ismcts=True; "
                "belief-weighted determinization only applies to "
                "ISMCTS's per-iteration determinize_fn callback. "
                "PIMC builds K independent search trees with their "
                "own root determinizations and has no equivalent "
                "seam."
            )

    def build(self, seed: int = 0) -> SearchStrategy:
        eval_kwargs: dict[str, Any] = {
            "weights_path": Path(self.weights_path),
            "with_policy": self.with_policy,
        }
        if self.hidden_sizes is not None:
            eval_kwargs["hidden_sizes"] = self.hidden_sizes
        if self.architecture is not None:
            eval_kwargs["architecture"] = self.architecture
        if self.policy_hidden_sizes is not None:
            eval_kwargs["policy_hidden_sizes"] = self.policy_hidden_sizes
        if self.device is not None:
            eval_kwargs["device"] = self.device
        if self.action_encoding_version is not None:
            eval_kwargs["action_encoding_version"] = (
                self.action_encoding_version
            )
        evaluator = NNEvaluator(**eval_kwargs)
        if self.use_ismcts:
            total_iters = max(1, self.samples * self.inner_nodes)
            ismcts_kwargs: dict[str, Any] = {
                "evaluator": evaluator,
                "budget_factory": lambda: Budget(max_nodes=total_iters),
                "rng": random.Random(seed + 300),
                "batch_size": self.batch_size,
                "puct": self.puct,
            }
            if self.use_belief_weighted:
                from mtg_bot.search.belief import (
                    BeliefWeightedDeterminizer,
                )
                ismcts_kwargs["determinize_fn"] = (
                    BeliefWeightedDeterminizer(
                        opponent_policy_evaluator=evaluator,
                        m_candidates=self.belief_m_candidates,
                        min_opponent_moments=(
                            self.belief_min_opponent_moments
                        ),
                    )
                )
            return ISMCTS(**ismcts_kwargs)
        inner_nodes = self.inner_nodes
        batch_size = self.batch_size
        puct = self.puct
        rng_seed_inner = seed + 100
        rng_seed_outer = seed + 200

        def inner_factory() -> MCTS:
            return MCTS(
                evaluator=evaluator,
                budget_factory=lambda: Budget(max_nodes=inner_nodes),
                rng=random.Random(rng_seed_inner),
                batch_size=batch_size,
                puct=puct,
            )

        return PIMC(
            inner_factory=inner_factory,
            samples=self.samples,
            rng=random.Random(rng_seed_outer),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_path": self.weights_path,
            "samples": self.samples,
            "inner_nodes": self.inner_nodes,
            "batch_size": self.batch_size,
            "with_policy": self.with_policy,
            "puct": self.puct,
            "hidden_sizes": (
                list(self.hidden_sizes)
                if self.hidden_sizes is not None else None
            ),
            "architecture": self.architecture,
            "policy_hidden_sizes": (
                list(self.policy_hidden_sizes)
                if self.policy_hidden_sizes is not None else None
            ),
            "use_ismcts": self.use_ismcts,
            "use_belief_weighted": self.use_belief_weighted,
            "belief_m_candidates": self.belief_m_candidates,
            "belief_min_opponent_moments": (
                self.belief_min_opponent_moments
            ),
            "device": self.device,
            "action_encoding_version": self.action_encoding_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NNPimcConfig":
        hidden_sizes = d.get("hidden_sizes")
        policy_hidden_sizes = d.get("policy_hidden_sizes")
        return cls(
            weights_path=d["weights_path"],
            samples=d["samples"],
            inner_nodes=d["inner_nodes"],
            batch_size=d.get("batch_size", 32),
            with_policy=d.get("with_policy", False),
            puct=d.get("puct", False),
            hidden_sizes=(
                tuple(hidden_sizes) if hidden_sizes is not None else None
            ),
            architecture=d.get("architecture"),
            policy_hidden_sizes=(
                tuple(policy_hidden_sizes)
                if policy_hidden_sizes is not None else None
            ),
            use_ismcts=d.get("use_ismcts", False),
            use_belief_weighted=d.get("use_belief_weighted", False),
            belief_m_candidates=d.get("belief_m_candidates", 64),
            belief_min_opponent_moments=d.get(
                "belief_min_opponent_moments", 3,
            ),
            device=d.get("device"),
            action_encoding_version=d.get("action_encoding_version"),
        )


@dataclass(frozen=True)
class NNServerPimcConfig:
    """Mirror of :class:`NNPimcConfig` but backed by a remote
    :class:`NNEvalClient` instead of a local :class:`NNEvaluator`.

    Carries an :class:`NNServerHandle` so workers can rebuild a
    client over the IPC boundary at ``build`` time. The handle is a
    frozen dataclass of pickle-friendly types (address, authkey
    bytes, with_policy bool), which means this whole config pickles
    cleanly across spawn-based multiprocessing pools — same
    pattern as :class:`NNPimcConfig` carrying ``weights_path``.

    The server is started OUT-OF-BAND by the launcher (CLI script)
    BEFORE the worker pool spins up; workers just need the handle.
    """

    server_handle: NNServerHandle
    samples: int = 4
    inner_nodes: int = 200
    batch_size: int = 32
    with_policy: bool = False
    puct: bool = False

    def build(self, seed: int = 0) -> SearchStrategy:
        evaluator = NNEvalClient(self.server_handle)
        inner_nodes = self.inner_nodes
        batch_size = self.batch_size
        puct = self.puct
        rng_seed_inner = seed + 100
        rng_seed_outer = seed + 200

        def inner_factory() -> MCTS:
            return MCTS(
                evaluator=evaluator,
                budget_factory=lambda: Budget(max_nodes=inner_nodes),
                rng=random.Random(rng_seed_inner),
                batch_size=batch_size,
                puct=puct,
            )

        return PIMC(
            inner_factory=inner_factory,
            samples=self.samples,
            rng=random.Random(rng_seed_outer),
        )


def random_choice_factory(seed: int) -> RandomChoice:
    """Strategy factory for ``RandomChoice``. The standard "is the
    PIMC actually doing something useful?" baseline. Has no config
    of its own; just needs a seed."""
    return RandomChoice(rng=random.Random(seed))


# Type alias for any callable that produces a search strategy from
# a seed. Both ``PimcConfig.build`` and ``random_choice_factory``
# satisfy this; tuning loops are written against this surface so
# they can pit any strategy type against any other.
StrategyFactory = Callable[[int], SearchStrategy]
