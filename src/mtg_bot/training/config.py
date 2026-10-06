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
from mtg_bot.search.determinize import (
    DEFAULT_OPPONENT_HIDDEN_MODEL,
    check_opponent_hidden_model,
    make_determinize_fn,
)
from mtg_bot.search.evaluator import Evaluator
from mtg_bot.search.heuristic_evaluator import MtgHeuristicEvaluator
from mtg_bot.search.nn_eval_server import NNEvalClient, NNServerHandle
from mtg_bot.search.nn_evaluator import (
    NNEvaluator,
    check_flat_metadata_value_options,
)
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
    # AlphaZero root exploration noise, carried for symmetry with
    # :class:`NNPimcConfig` so the heuristic self-play stage accepts
    # the same search-level knobs as the NN one. NOTE: the noise is
    # mixed into the root children's PUCT *priors*, and priors are
    # only consulted when ``puct=True`` -- which requires a
    # policy-head evaluator. The heuristic evaluator has none, so
    # this MCTS runs UCB1 and the noise is inert. The self-play
    # driver warns about that rather than pretending otherwise; see
    # ``mtg_bot.training.self_play_evaluator``.
    root_dirichlet_alpha: Optional[float] = None
    root_dirichlet_eps: float = 0.25
    # How each PIMC world models the opponent's hidden cards:
    # ``"sampled"`` (default, unchanged behaviour) or ``"board_only"``
    # (inert placeholders; see ``mtg_bot.search.determinize``).
    opponent_hidden_model: str = DEFAULT_OPPONENT_HIDDEN_MODEL
    # Whether this seat's mid-resolution choices are search decisions
    # (when the game-level switch ``HeuristicsConfig.
    # search_resolution_decisions`` is on). False = the old
    # ``DefaultTacticalAgent`` behaviour for this seat only (ladder
    # ``+autotarget``). UCB1 needs no priors, so the heuristic can
    # always search them.
    search_resolution_decisions: bool = True

    def __post_init__(self) -> None:
        check_opponent_hidden_model(self.opponent_hidden_model)

    def build(self, seed: int = 0) -> SearchStrategy:
        """Construct a fresh PIMC search strategy seeded for
        reproducibility. Different ``seed`` values give different
        sub-search RNGs."""
        evaluator = self.heuristic.build()
        inner_nodes = self.inner_nodes
        batch_size = self.batch_size
        root_dirichlet_alpha = self.root_dirichlet_alpha
        root_dirichlet_eps = self.root_dirichlet_eps
        rng_seed_inner = seed + 100
        rng_seed_outer = seed + 200

        def inner_factory() -> MCTS:
            return MCTS(
                evaluator=evaluator,
                budget_factory=lambda: Budget(max_nodes=inner_nodes),
                rng=random.Random(rng_seed_inner),
                batch_size=batch_size,
                root_dirichlet_alpha=root_dirichlet_alpha,
                root_dirichlet_eps=root_dirichlet_eps,
            )

        return _tag_resolution(
            PIMC(
                inner_factory=inner_factory,
                samples=self.samples,
                rng=random.Random(rng_seed_outer),
                determinize_fn=make_determinize_fn(
                    self.opponent_hidden_model,
                ),
            ),
            search=self.search_resolution_decisions,
            can_prior=True,
            note="heuristic PIMC",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "inner_nodes": self.inner_nodes,
            "heuristic": self.heuristic.to_dict(),
            "batch_size": self.batch_size,
            "root_dirichlet_alpha": self.root_dirichlet_alpha,
            "root_dirichlet_eps": self.root_dirichlet_eps,
            "opponent_hidden_model": self.opponent_hidden_model,
            "search_resolution_decisions": self.search_resolution_decisions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PimcConfig":
        return cls(
            samples=d["samples"],
            inner_nodes=d["inner_nodes"],
            heuristic=HeuristicConfig.from_dict(d["heuristic"]),
            batch_size=d.get("batch_size", 1),
            root_dirichlet_alpha=d.get("root_dirichlet_alpha"),
            root_dirichlet_eps=d.get("root_dirichlet_eps", 0.25),
            opponent_hidden_model=d.get(
                "opponent_hidden_model", DEFAULT_OPPONENT_HIDDEN_MODEL,
            ),
            search_resolution_decisions=bool(
                d.get("search_resolution_decisions", True),
            ),
        )


def _tag_resolution(
    strategy: SearchStrategy, *, search: bool, can_prior: bool, note: str,
) -> SearchStrategy:
    """Stamp the per-agent resolution-decision attributes the game
    constructor reads (``engine_adapter_factory.resolution_seats_for``):
    ``search_resolution_decisions`` (the agent's opt-in) and
    ``can_prior_resolution_decisions`` (False for a PUCT policy head
    without resolution slots -- refused when the seat is searched)."""
    strategy.search_resolution_decisions = bool(search)
    strategy.can_prior_resolution_decisions = bool(can_prior)
    strategy.resolution_prior_note = note
    return strategy


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
    # AlphaZero root exploration noise for the PUCT sub-searches.
    # ``root_dirichlet_alpha=None`` (default) = off, bit-identical to
    # the pre-noise search. When set, every PIMC sub-search root
    # mixes ``(1-eps) * prior + eps * Dirichlet(alpha)`` into its
    # children's priors once they are all expanded (see
    # ``MCTS._apply_root_dirichlet_noise``). Self-play data-gen sets
    # this (iterate default 0.5 / 0.25); eval configs leave it None so
    # the gate stays deterministic-argmax. Not applied on the ISMCTS
    # path (``use_ismcts=True``), which has no root-prior hook.
    root_dirichlet_alpha: Optional[float] = None
    root_dirichlet_eps: float = 0.25
    # Torch device for the underlying ``NNEvaluator``. ``None`` means
    # "use NNEvaluator's default" (currently CPU) — preserves the
    # production data_gen / self-play default of CPU-only workers
    # (per the CLAUDE.md note: multiprocessing workers stay CPU-only
    # to avoid creating one CUDA context per worker process). The
    # bench may opt into ``"cuda"`` for matched-budget runs where the
    # batched-leaf NN forward dominates the per-chunk wall-clock.
    #
    # ``"auto"`` is a sentinel resolved by ``build()`` (which runs in
    # the worker process, not the parent that pickles this spec into
    # the mp.Pool): a deferred ``auto_device()`` call picks
    # ``"cuda"`` / ``"mps"`` / ``"cpu"`` per-host. Distributed shard
    # runners pass ``"auto"`` so each host's workers resolve locally.
    # Eagerly calling ``auto_device()`` in the parent and storing
    # ``"cuda"`` here breaks Linux fork() — PyTorch raises "Cannot
    # re-initialize CUDA in forked subprocess" because the parent's
    # ``torch.cuda.is_available()`` already primed the CUDA context
    # before the pool fork()ed.
    device: Optional[str] = None
    # Apply the checkpoint's post-hoc value calibration
    # (``metadata.value_calibration``: ``tanh(logit / T + b)``). True
    # (default) is the correct behaviour and matches
    # ``NNEvaluator.from_metadata``; False reproduces the pre-2026-10-04
    # build path, which never applied it (``mtg_bot ladder`` label
    # modifier ``+nocal``). Only consulted when a ``metadata.json`` sits
    # next to ``weights_path``.
    apply_value_calibration: bool = True
    # How each search world models the opponent's hidden cards:
    # ``"sampled"`` (default, unchanged behaviour) or ``"board_only"``
    # (inert placeholders; see ``mtg_bot.search.determinize``). Each
    # seat's config carries its own, so the two seats of a match are
    # independent. ``mtg_bot ladder`` label modifier ``+boardonly``.
    opponent_hidden_model: str = DEFAULT_OPPONENT_HIDDEN_MODEL
    # This seat's mid-resolution choices are search decisions when the
    # game-level switch is on (see ``PimcConfig``); False = the old
    # auto-targeting for this seat only (ladder ``+autotarget``). A
    # PUCT seat needs a policy head with resolution slots
    # (``canonical_v3``) to search them -- otherwise the game refuses
    # to start (``ResolutionPriorUnavailableError``).
    search_resolution_decisions: bool = True

    def __post_init__(self) -> None:
        check_opponent_hidden_model(self.opponent_hidden_model)
        if (
            self.use_belief_weighted
            and self.opponent_hidden_model != DEFAULT_OPPONENT_HIDDEN_MODEL
        ):
            raise ValueError(
                "use_belief_weighted=True samples the opponent's hand "
                "from a belief model; it cannot be combined with "
                f"opponent_hidden_model={self.opponent_hidden_model!r}"
            )
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
        # Fork-inherited SHM server handle takes precedence over the
        # per-worker NNEvaluator path. Looked up by ``self.weights_path``
        # first (eval routes new vs old via this key — two SHM servers
        # in master, each registered under its weights path); falls
        # back to the singleton handle (self-play registers under
        # the singleton slot via ``run_shard_pull``'s Pool initializer).
        # Absent any handle, the legacy NNEvaluator construction runs
        # unchanged.
        from mtg_bot.search.shm_nn_eval_server import (
            maybe_get_worker_handle,
        )
        shm_handle = maybe_get_worker_handle(self.weights_path)
        if shm_handle is not None:
            from mtg_bot.search.shm_nn_eval_server import (
                SharedTensorNNEvalClient,
            )
            evaluator = SharedTensorNNEvalClient(shm_handle)
        else:
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
                if self.device == "auto":
                    # Deferred resolution: import + call auto_device()
                    # inside build(), which runs in the mp.Pool worker
                    # process. The parent that pickles this spec must
                    # NOT have initialized CUDA, otherwise Linux fork()
                    # crashes with "Cannot re-initialize CUDA in forked
                    # subprocess". See the ``device`` field comment
                    # above.
                    from mtg_bot.training.trainer import auto_device
                    eval_kwargs["device"] = auto_device()
                else:
                    eval_kwargs["device"] = self.device
            if self.action_encoding_version is not None:
                eval_kwargs["action_encoding_version"] = (
                    self.action_encoding_version
                )
            # The version's ``metadata.json`` next to the weights is
            # the source of truth for everything NNPimcConfig itself
            # doesn't carry: the structured trunk dims, scalar schema,
            # value-head input and post-hoc value calibration. It goes
            # through the SAME resolver as ``NNEvaluator.from_metadata``
            # (``nn_evaluator.structured_kwargs_from_metadata``), so a
            # worker / ladder / arena seat runs exactly the evaluator
            # the held-out gate graded. (Before 2026-10-04 this branch
            # read the trunk dims by hand and dropped the calibration
            # and value_head_input -- every played game ran the value
            # head uncalibrated.)
            meta = _sibling_metadata(Path(self.weights_path))
            if meta is not None:
                # Config fields fill what the sidecar omits and must
                # agree with what it records (raises otherwise).
                meta = _reconcile_metadata(
                    meta,
                    architecture=self.architecture,
                    action_encoding_version=self.action_encoding_version,
                    weights_path=self.weights_path,
                )
            if meta is not None and meta.get("architecture") == "structured":
                evaluator = NNEvaluator.from_metadata(
                    Path(self.weights_path),
                    meta,
                    device=eval_kwargs.get("device", "cpu"),
                    apply_value_calibration=self.apply_value_calibration,
                )
            else:
                if meta is not None:
                    # Flat MLP trunk: no logit hook, no alternative
                    # value-head inputs -- refuse metadata declaring
                    # either instead of silently dropping it.
                    check_flat_metadata_value_options(
                        meta,
                        apply_value_calibration=(
                            self.apply_value_calibration
                        ),
                    )
                # No sidecar metadata (hand-built / test weights):
                # nothing declared, nothing to apply -- the evaluator's
                # defaults (identity calibration, shared value input).
                evaluator = NNEvaluator(**eval_kwargs)
        return _tag_resolution(
            self._build_search(evaluator, seed),
            search=self.search_resolution_decisions,
            can_prior=_can_prior_resolution(evaluator, self.puct),
            note=f"NN seat {self.weights_path}",
        )

    def _build_search(self, evaluator: Any, seed: int) -> SearchStrategy:
        if self.use_ismcts:
            total_iters = max(1, self.samples * self.inner_nodes)
            ismcts_kwargs: dict[str, Any] = {
                "evaluator": evaluator,
                "budget_factory": lambda: Budget(max_nodes=total_iters),
                "rng": random.Random(seed + 300),
                "batch_size": self.batch_size,
                "puct": self.puct,
            }
            if self.opponent_hidden_model != DEFAULT_OPPONENT_HIDDEN_MODEL:
                ismcts_kwargs["determinize_fn"] = make_determinize_fn(
                    self.opponent_hidden_model,
                )
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
        root_dirichlet_alpha = self.root_dirichlet_alpha
        root_dirichlet_eps = self.root_dirichlet_eps
        rng_seed_inner = seed + 100
        rng_seed_outer = seed + 200

        def inner_factory() -> MCTS:
            return MCTS(
                evaluator=evaluator,
                budget_factory=lambda: Budget(max_nodes=inner_nodes),
                rng=random.Random(rng_seed_inner),
                batch_size=batch_size,
                puct=puct,
                root_dirichlet_alpha=root_dirichlet_alpha,
                root_dirichlet_eps=root_dirichlet_eps,
            )

        return PIMC(
            inner_factory=inner_factory,
            samples=self.samples,
            rng=random.Random(rng_seed_outer),
            determinize_fn=make_determinize_fn(self.opponent_hidden_model),
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
            "root_dirichlet_alpha": self.root_dirichlet_alpha,
            "root_dirichlet_eps": self.root_dirichlet_eps,
            "apply_value_calibration": self.apply_value_calibration,
            "opponent_hidden_model": self.opponent_hidden_model,
            "search_resolution_decisions": self.search_resolution_decisions,
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
            root_dirichlet_alpha=d.get("root_dirichlet_alpha"),
            root_dirichlet_eps=d.get("root_dirichlet_eps", 0.25),
            apply_value_calibration=bool(
                d.get("apply_value_calibration", True)
            ),
            opponent_hidden_model=d.get(
                "opponent_hidden_model", DEFAULT_OPPONENT_HIDDEN_MODEL,
            ),
            search_resolution_decisions=bool(
                d.get("search_resolution_decisions", True),
            ),
        )


def _can_prior_resolution(evaluator: Any, puct: bool) -> bool:
    """Whether a seat built on ``evaluator`` can search resolution
    pauses: UCB1 (``puct=False``) needs no priors; PUCT needs a policy
    head whose action encoding has resolution slots."""
    if not puct:
        return True
    from mtg_bot.training.action_encoding import has_resolution_slots

    version = getattr(evaluator, "action_encoding_version", None)
    if version is None:
        return False
    return has_resolution_slots(version)


def _sibling_metadata(weights_path: Path) -> Optional[dict[str, Any]]:
    """The raw ``metadata.json`` next to ``weights_path`` (the registry
    layout ``vN/{weights.pt,metadata.json}``; the worker weights cache
    mirrors it), or ``None`` when there is none.

    A present-but-unreadable file raises: it is the only record of the
    checkpoint's trunk shape and value calibration, and loading without
    it would silently run a different evaluator than the one trained.
    """
    import json

    meta_path = weights_path.parent / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"cannot read {meta_path}: {exc}. It records the "
            "checkpoint's trunk dims and value calibration; refusing "
            "to load the weights without it."
        ) from exc
    if not isinstance(meta, dict):
        raise ValueError(
            f"{meta_path} is not a JSON object (got "
            f"{type(meta).__name__})"
        )
    return meta


def _reconcile_metadata(
    meta: dict[str, Any],
    *,
    architecture: Optional[str],
    action_encoding_version: Optional[str],
    weights_path: str,
) -> dict[str, Any]:
    """Merge the explicit ``NNPimcConfig`` fields into the sidecar
    metadata for ``NNEvaluator.from_metadata``.

    A field the sidecar lacks takes the config's value (minimal
    hand-written sidecars omit ``architecture``); a field both carry
    must agree -- a mismatched action encoding scrambles the policy
    priors, a mismatched architecture loads the wrong model class --
    so a disagreement raises instead of picking one silently.
    """
    merged = dict(meta)
    for key, value in (
        ("architecture", architecture),
        ("action_encoding_version", action_encoding_version),
    ):
        if value is None:
            continue
        recorded = merged.get(key)
        if recorded is None:
            merged[key] = value
        elif recorded != value:
            raise ValueError(
                f"NNPimcConfig.{key}={value!r} disagrees with "
                f"{key}={recorded!r} in the metadata.json next to "
                f"{weights_path}"
            )
    return merged


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

        return _tag_resolution(
            PIMC(
                inner_factory=inner_factory,
                samples=self.samples,
                rng=random.Random(rng_seed_outer),
            ),
            search=True,
            # The NN eval server path is 517 slots wide (no resolution
            # block): fine under UCB1, refused when PUCT and searched.
            can_prior=not self.puct,
            note="NN-server seat",
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
