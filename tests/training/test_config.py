"""Tests for ``mtg_bot.training.config``.

Focused on the contract callers depend on: configs are frozen
(hashable), build() produces a working SearchStrategy, JSON round-
trips are exact, and equal configs hash to the same bucket so a
tuning loop can dedupe.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mtg_bot.search import PIMC, RandomChoice
from mtg_bot.search.heuristic_evaluator import MtgHeuristicEvaluator
from mtg_bot.search.nn_eval_server import NNEvalServer
from mtg_bot.search.nn_evaluator import NNEvaluator
from mtg_bot.training.card_vocab import MissingCardVocabError
from tests._card_vocab_helpers import current_vocab_dict, stamp_current_vocab
from mtg_bot.training.config import (
    HeuristicConfig,
    NNPimcConfig,
    NNServerPimcConfig,
    PimcConfig,
    random_choice_factory,
)


# ---------- HeuristicConfig --------------------------------------------------


def test_heuristic_config_defaults_match_evaluator_defaults():
    cfg = HeuristicConfig()
    assert cfg.life_weight == 0.4
    assert cfg.board_weight == 0.4
    assert cfg.card_weight == 0.15
    assert cfg.mana_dev_weight == 0.05


def test_heuristic_config_is_frozen_and_hashable():
    cfg = HeuristicConfig(life_weight=0.5, board_weight=0.3, card_weight=0.2)
    with pytest.raises(Exception):
        cfg.life_weight = 0.9  # type: ignore[misc]
    # Hashable → usable as dict key.
    bucket: dict = {cfg: "x"}
    assert bucket[
        HeuristicConfig(life_weight=0.5, board_weight=0.3, card_weight=0.2)
    ] == "x"


def test_heuristic_config_build_returns_evaluator_with_normalized_weights():
    cfg = HeuristicConfig(
        life_weight=2.0, board_weight=2.0, card_weight=1.0,
        mana_dev_weight=0.0,
    )
    evaluator = cfg.build()
    assert isinstance(evaluator, MtgHeuristicEvaluator)
    # Evaluator renormalizes — 2/2/1/0 → 0.4/0.4/0.2/0.0 internally.
    assert evaluator.life_weight == pytest.approx(0.4)
    assert evaluator.board_weight == pytest.approx(0.4)
    assert evaluator.card_weight == pytest.approx(0.2)
    assert evaluator.mana_dev_weight == pytest.approx(0.0)


def test_heuristic_config_json_round_trip():
    cfg = HeuristicConfig(life_weight=0.3, board_weight=0.5, card_weight=0.2)
    assert HeuristicConfig.from_dict(cfg.to_dict()) == cfg


# ---------- PimcConfig -------------------------------------------------------


def test_pimc_config_defaults_match_recommend_defaults():
    cfg = PimcConfig()
    assert cfg.samples == 4
    assert cfg.inner_nodes == 200
    assert cfg.heuristic == HeuristicConfig()


def test_pimc_config_build_returns_pimc_strategy():
    cfg = PimcConfig(samples=2, inner_nodes=20)
    strategy = cfg.build(seed=42)
    assert isinstance(strategy, PIMC)
    assert strategy.samples == 2


def test_pimc_config_build_is_deterministic_in_seed():
    """Same config + same seed → produces same RNG state. Two calls
    with seed=42 are independent objects (each new build) but the
    first decision they'd produce must match given the same input."""
    cfg = PimcConfig(samples=1, inner_nodes=8)
    s1 = cfg.build(seed=42)
    s2 = cfg.build(seed=42)
    # Different objects, but their internal RNGs were seeded
    # identically — first random draw must match.
    assert s1.rng.random() == s2.rng.random()


def test_pimc_config_is_frozen_and_hashable():
    cfg = PimcConfig(samples=2, inner_nodes=20)
    with pytest.raises(Exception):
        cfg.samples = 99  # type: ignore[misc]
    bucket: dict = {cfg: "x"}
    assert bucket[PimcConfig(samples=2, inner_nodes=20)] == "x"


def test_pimc_config_json_round_trip():
    cfg = PimcConfig(
        samples=3, inner_nodes=75,
        heuristic=HeuristicConfig(
            life_weight=0.5, board_weight=0.3, card_weight=0.2,
        ),
    )
    assert PimcConfig.from_dict(cfg.to_dict()) == cfg


# ---------- random_choice_factory -------------------------------------------


def test_random_choice_factory_returns_random_choice():
    strategy = random_choice_factory(seed=7)
    assert isinstance(strategy, RandomChoice)


def test_random_choice_factory_seeded():
    s1 = random_choice_factory(seed=7)
    s2 = random_choice_factory(seed=7)
    # Same seed → same RNG draw.
    assert s1.rng.random() == s2.rng.random()


# ---------- NNPimcConfig -----------------------------------------------------


def _save_value_weights(tmp_path: Path, name: str = "value.pt") -> Path:
    """Save a fresh value-only NNEvaluator's state_dict so build()
    has a real file to load. The contents don't matter — we don't
    invoke the resulting strategy beyond constructing it."""
    torch.manual_seed(0)
    ev = NNEvaluator(with_policy=False)
    p = tmp_path / name
    ev.save(p)
    return p


def _save_policy_weights(tmp_path: Path, name: str = "policy.pt") -> Path:
    torch.manual_seed(0)
    ev = NNEvaluator(with_policy=True)
    p = tmp_path / name
    ev.save(p)
    return p


def test_nn_pimc_config_build_loads_value_only_weights(tmp_path: Path) -> None:
    weights = _save_value_weights(tmp_path)
    cfg = NNPimcConfig(
        weights_path=str(weights), samples=2, inner_nodes=8, batch_size=1,
    )
    strategy = cfg.build(seed=3)
    assert isinstance(strategy, PIMC)
    assert strategy.samples == 2


def test_nn_pimc_config_build_with_policy_path(tmp_path: Path) -> None:
    """``with_policy=True`` exercises the dual-head loader branch
    inside ``NNEvaluator``. ``puct=True`` is paired with it because
    the value-only model would otherwise reject puct."""
    weights = _save_policy_weights(tmp_path)
    cfg = NNPimcConfig(
        weights_path=str(weights), samples=2, inner_nodes=4, batch_size=1,
        with_policy=True, puct=True,
    )
    strategy = cfg.build(seed=5)
    assert isinstance(strategy, PIMC)


def test_nn_pimc_config_build_with_explicit_hidden_sizes(
    tmp_path: Path,
) -> None:
    """Setting ``hidden_sizes`` threads the override into NNEvaluator
    construction; the saved weights must use the same shape so
    state_dict load doesn't reject."""
    torch.manual_seed(0)
    sizes = (32, 32)
    ev = NNEvaluator(with_policy=False, hidden_sizes=sizes)
    weights = tmp_path / "small.pt"
    ev.save(weights)
    cfg = NNPimcConfig(
        weights_path=str(weights), samples=1, inner_nodes=2, batch_size=1,
        hidden_sizes=sizes,
    )
    strategy = cfg.build(seed=0)
    assert isinstance(strategy, PIMC)


def test_nn_pimc_config_build_picks_up_structured_meta(
    tmp_path: Path,
) -> None:
    """When ``architecture='structured'``, the build path must
    discover the trunk hparams from the version's metadata.json
    sidecar (``training.structured_*``). Without this discovery, a
    checkpoint trained with non-default dims (emb!=128, layers!=2,
    hidden!=256) would load against ``StructuredModelConfig``'s
    defaults and crash with a state_dict shape mismatch on every
    worker — the v10 regression that motivated this fix."""
    import json as _json

    from mtg_bot.training.structured_model import (
        StructuredModel, StructuredModelConfig,
    )

    # Build a tiny non-default-dim structured model so the loader-
    # vs-saved-shape distinction is exercised.
    torch.manual_seed(0)
    cfg = StructuredModelConfig(
        emb_dim=64, n_heads=4, n_layers=1, hidden_dim=128,
        action_history_n=8,
    )
    model = StructuredModel(cfg)
    weights = tmp_path / "v_test" / "weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save(model.state_dict(), weights)

    # Write the metadata sidecar in the same shape the trainer would.
    meta_path = weights.parent / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        _json.dump({
            "version": "v_test",
            "architecture": "structured",
            "action_encoding_version": "canonical_v2",
            # Absent -> schema 1 (3-slot); the saved model is on the
            # config default, so record it.
            "scalar_schema_version": cfg.scalar_schema_version,
            "card_vocab": current_vocab_dict(),
            "training": {
                "structured_emb_dim": 64,
                "structured_n_heads": 4,
                "structured_n_layers": 1,
                "structured_hidden_dim": 128,
                "structured_action_history_n": 8,
            },
        }, f)

    # If the build path ignored the metadata sidecar, NNEvaluator
    # would construct a default-config 10M-param trunk and torch
    # would raise on the state_dict load.
    pimc_cfg = NNPimcConfig(
        weights_path=str(weights),
        samples=1, inner_nodes=2, batch_size=1, with_policy=True,
        architecture="structured",
        action_encoding_version="canonical_v2",
    )
    strategy = pimc_cfg.build(seed=0)
    assert isinstance(strategy, PIMC)


def test_nn_pimc_config_build_picks_up_structured_meta_no_sidecar(
    tmp_path: Path,
) -> None:
    """When ``architecture='structured'`` but no metadata.json sidecar
    exists, build still goes through the default-config path. The
    new discovery code must not raise on the missing-sidecar branch
    — defensive against a fresh registry checkout / hand-crafted
    weights / etc."""
    from mtg_bot.training.structured_model import (
        StructuredModel, StructuredModelConfig,
    )

    torch.manual_seed(0)
    # Use the DEFAULT structured config so the load works without a
    # sidecar telling NNEvaluator otherwise.
    cfg = StructuredModelConfig()
    model = StructuredModel(cfg)
    weights = tmp_path / "v_test" / "weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save(model.state_dict(), weights)
    # Intentionally NO metadata.json next to weights; the card
    # vocabulary comes from the weights file's own stamp.
    stamp_current_vocab(weights)

    pimc_cfg = NNPimcConfig(
        weights_path=str(weights),
        samples=1, inner_nodes=2, batch_size=1, with_policy=True,
        architecture="structured",
    )
    strategy = pimc_cfg.build(seed=0)
    assert isinstance(strategy, PIMC)


def test_nn_pimc_config_build_refuses_unstamped_structured_weights(
    tmp_path: Path,
) -> None:
    """The worker build path must never guess a structured
    checkpoint's card vocabulary: no metadata ``card_vocab`` and no
    stamp sidecar -> ``MissingCardVocabError``, not a silent load whose
    printing embeddings may be indexed by a different card list."""
    from mtg_bot.training.structured_model import (
        StructuredModel, StructuredModelConfig,
    )

    weights = tmp_path / "v_test" / "weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save(StructuredModel(StructuredModelConfig()).state_dict(), weights)
    pimc_cfg = NNPimcConfig(
        weights_path=str(weights),
        samples=1, inner_nodes=2, batch_size=1, with_policy=True,
        architecture="structured",
    )
    with pytest.raises(MissingCardVocabError, match="card vocabulary"):
        pimc_cfg.build(seed=0)


def test_nn_pimc_config_to_dict_with_explicit_hidden_sizes(
    tmp_path: Path,
) -> None:
    """``to_dict`` converts the tuple to a list (JSON-friendly) and
    ``from_dict`` converts back to a tuple — covers the
    ``hidden_sizes is not None`` branch in serialization."""
    cfg = NNPimcConfig(
        weights_path="ignored.pt",
        samples=4, inner_nodes=8, batch_size=2,
        with_policy=True, puct=False,
        hidden_sizes=(64, 32),
    )
    d = cfg.to_dict()
    assert d["hidden_sizes"] == [64, 32]
    rebuilt = NNPimcConfig.from_dict(d)
    assert rebuilt == cfg
    assert rebuilt.hidden_sizes == (64, 32)


def test_nn_pimc_config_to_dict_without_hidden_sizes() -> None:
    """``hidden_sizes=None`` round-trips as ``None`` through the
    JSON serializer — covers the ``else`` branch in ``to_dict``."""
    cfg = NNPimcConfig(weights_path="ignored.pt")
    d = cfg.to_dict()
    assert d["hidden_sizes"] is None
    rebuilt = NNPimcConfig.from_dict(d)
    assert rebuilt == cfg
    assert rebuilt.hidden_sizes is None


def test_nn_pimc_config_from_dict_with_missing_optional_fields() -> None:
    """``from_dict`` tolerates older serializations that don't carry
    ``batch_size`` / ``with_policy`` / ``puct`` / ``hidden_sizes``
    / ``use_ismcts`` — they default. Covers the ``.get(..., default)``
    branches."""
    minimal = {
        "weights_path": "old.pt", "samples": 4, "inner_nodes": 200,
    }
    cfg = NNPimcConfig.from_dict(minimal)
    assert cfg.weights_path == "old.pt"
    assert cfg.batch_size == 32
    assert cfg.with_policy is False
    assert cfg.puct is False
    assert cfg.hidden_sizes is None
    assert cfg.use_ismcts is False


def test_nn_pimc_config_use_ismcts_flag_switches_strategy(
    tmp_path: Path,
) -> None:
    """``use_ismcts=True`` makes ``build()`` return ISMCTS instead of
    PIMC. Matched-budget conversion: total_iterations = samples *
    inner_nodes (the bench script's convention; same total search
    work as PIMC's K independent searches of N nodes each)."""
    from mtg_bot.search.ismcts import ISMCTS
    weights = _save_value_weights(tmp_path)
    cfg_pimc = NNPimcConfig(
        weights_path=str(weights), samples=2, inner_nodes=4,
    )
    cfg_ismcts = NNPimcConfig(
        weights_path=str(weights), samples=2, inner_nodes=4,
        use_ismcts=True,
    )
    assert isinstance(cfg_pimc.build(seed=0), PIMC)
    assert isinstance(cfg_ismcts.build(seed=0), ISMCTS)


def test_nn_pimc_config_use_ismcts_round_trips_through_dict() -> None:
    """The flag survives JSON serialization so distributed shards
    receive the same search-algorithm choice the orchestrator made."""
    cfg = NNPimcConfig(
        weights_path="x.pt", samples=4, inner_nodes=200,
        use_ismcts=True,
    )
    assert NNPimcConfig.from_dict(cfg.to_dict()).use_ismcts is True


# ---------- NNPimcConfig.use_belief_weighted (Phase 6.5) --------------------


def test_nn_pimc_config_use_belief_weighted_requires_ismcts() -> None:
    """``use_belief_weighted=True`` without ``use_ismcts=True`` is
    rejected at construction. Belief-weighted determinization is a
    seam on ISMCTS's per-iteration determinize_fn callback; PIMC
    builds K independent search trees and has no equivalent hook.
    """
    with pytest.raises(ValueError, match="requires use_ismcts"):
        NNPimcConfig(
            weights_path="x.pt", samples=4, inner_nodes=200,
            use_ismcts=False,
            use_belief_weighted=True,
        )


def test_nn_pimc_config_use_belief_weighted_builds_belief_determinizer(
    tmp_path: Path,
) -> None:
    """``use_belief_weighted=True`` + ``use_ismcts=True`` constructs
    an ISMCTS whose ``_determinize`` is a ``BeliefWeightedDeterminizer``
    instance (not the module-level uniform determinize).
    """
    from mtg_bot.search.belief import BeliefWeightedDeterminizer
    from mtg_bot.search.ismcts import ISMCTS

    weights = _save_value_weights(tmp_path)
    cfg = NNPimcConfig(
        weights_path=str(weights), samples=2, inner_nodes=4,
        use_ismcts=True,
        use_belief_weighted=True,
    )
    strategy = cfg.build(seed=0)
    assert isinstance(strategy, ISMCTS)
    assert isinstance(
        strategy._determinize, BeliefWeightedDeterminizer,
    )
    # Empty played_history at construction; populated during play
    # via ISMCTS.observe_action calls in the game loop.
    assert strategy._determinize.played_history == ()


def test_nn_pimc_config_use_belief_weighted_round_trips_through_dict() -> None:
    """The Phase 6.5 flag survives JSON serialization so distributed
    shards build the same determinizer the orchestrator chose."""
    cfg = NNPimcConfig(
        weights_path="x.pt", samples=4, inner_nodes=200,
        use_ismcts=True,
        use_belief_weighted=True,
    )
    rebuilt = NNPimcConfig.from_dict(cfg.to_dict())
    assert rebuilt.use_belief_weighted is True
    assert rebuilt.use_ismcts is True


def test_nn_pimc_config_use_belief_weighted_default_is_false() -> None:
    """The new flag defaults to off and is omitted from older
    serializations cleanly via the ``.get(..., False)`` branch."""
    cfg = NNPimcConfig(weights_path="x.pt", samples=4, inner_nodes=200)
    assert cfg.use_belief_weighted is False
    minimal = {
        "weights_path": "old.pt", "samples": 4, "inner_nodes": 200,
    }
    assert NNPimcConfig.from_dict(minimal).use_belief_weighted is False


# ---------- NNPimcConfig.device (Tier 1.1 GPU NN) ---------------------------


def test_nn_pimc_config_device_default_is_none() -> None:
    """``device`` defaults to ``None`` — meaning "use NNEvaluator's
    default" (currently CPU). This preserves the production
    multiprocessing-data-gen contract that workers stay CPU-only."""
    cfg = NNPimcConfig(weights_path="x.pt", samples=4, inner_nodes=200)
    assert cfg.device is None


def test_nn_pimc_config_device_round_trips_through_dict() -> None:
    """Explicit ``device`` survives JSON serialization so distributed
    shards instantiate the evaluator on the same device the
    orchestrator chose."""
    cfg = NNPimcConfig(
        weights_path="x.pt", samples=4, inner_nodes=200,
        device="cpu",
    )
    rebuilt = NNPimcConfig.from_dict(cfg.to_dict())
    assert rebuilt.device == "cpu"
    assert rebuilt == cfg


def test_nn_pimc_config_device_missing_in_dict_defaults_to_none() -> None:
    """Older serializations without ``device`` decode cleanly with
    ``device=None`` via the ``.get(...)`` branch."""
    minimal = {
        "weights_path": "old.pt", "samples": 4, "inner_nodes": 200,
    }
    assert NNPimcConfig.from_dict(minimal).device is None


def test_nn_pimc_config_device_threaded_into_evaluator(
    tmp_path: Path,
) -> None:
    """``device="cpu"`` is passed through to ``NNEvaluator``; the
    constructed strategy's evaluator carries a CPU torch device."""
    weights = _save_value_weights(tmp_path)
    cfg = NNPimcConfig(
        weights_path=str(weights), samples=2, inner_nodes=4,
        batch_size=1, device="cpu",
    )
    strategy = cfg.build(seed=0)
    # PIMC's inner_factory closes over the evaluator; build one MCTS
    # to inspect.
    mcts = strategy.inner_factory()
    assert str(mcts.evaluator.device) == "cpu"


def test_nn_pimc_config_device_auto_does_not_resolve_at_construction() -> None:
    """``device="auto"`` is a sentinel — the literal string "auto" is
    stored on the spec until ``build()`` is called. This is the
    Linux-fork-safety contract: distributed shard parents pass "auto"
    so auto_device() (which calls torch.cuda.is_available() and primes
    the CUDA context) doesn't fire until the worker process picks it
    up via ``cfg.build()``. Parent-side resolution would crash
    fork()'d workers with "Cannot re-initialize CUDA in forked
    subprocess"."""
    cfg = NNPimcConfig(
        weights_path="x.pt", samples=4, inner_nodes=200, device="auto",
    )
    assert cfg.device == "auto"


def test_nn_pimc_config_device_auto_round_trips_through_dict() -> None:
    """The "auto" sentinel survives JSON serialization so distributed
    specs can ship it across the SSH boundary unchanged."""
    cfg = NNPimcConfig(
        weights_path="x.pt", samples=4, inner_nodes=200, device="auto",
    )
    rebuilt = NNPimcConfig.from_dict(cfg.to_dict())
    assert rebuilt.device == "auto"
    assert rebuilt == cfg


def test_nn_pimc_config_build_uses_shm_handle_when_set(
    tmp_path: Path,
) -> None:
    """When a worker-process SHM handle is installed (via
    ``set_worker_handle``), ``NNPimcConfig.build()`` returns a strategy
    that uses :class:`SharedTensorNNEvalClient` instead of constructing
    a per-worker NNEvaluator. The cfg's own ``weights_path`` / ``device``
    fields are IGNORED in this branch — proves the precedence ordering.
    Runs on Linux + Windows (the SHM substrate now works on both)."""
    from mtg_bot.search.shm_nn_eval_server import (
        SharedTensorNNEvalClient,
        SharedTensorNNEvalServer,
        set_worker_handle,
    )

    weights = _save_value_weights(tmp_path)
    server = SharedTensorNNEvalServer(
        weights_path=weights,
        device="cpu",
        with_policy=False,
        batch_size=1,
        batch_wait_ms=0.0,
        n_slots=4,
    )
    try:
        with server:
            set_worker_handle(server.handle())
            try:
                cfg = NNPimcConfig(
                    weights_path=str(weights), samples=2, inner_nodes=4,
                    batch_size=1, device="cpu",
                )
                strategy = cfg.build(seed=0)
                mcts = strategy.inner_factory()
                assert isinstance(mcts.evaluator, SharedTensorNNEvalClient)
            finally:
                # Restore the global to None so other tests aren't
                # affected by this one's state.
                set_worker_handle(None)  # type: ignore[arg-type]
    finally:
        server.stop(timeout=5.0)


def test_nn_pimc_config_device_auto_resolves_via_auto_device_in_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build()`` calls ``auto_device()`` exactly once and threads
    its return value into the underlying ``NNEvaluator`` — verifying
    the deferred-resolution wiring. Patches ``auto_device`` to "cpu"
    so the test doesn't init real CUDA (which would defeat the
    fork-safety the feature exists to protect)."""
    weights = _save_value_weights(tmp_path)
    call_count = {"n": 0}

    def fake_auto_device() -> str:
        call_count["n"] += 1
        return "cpu"

    monkeypatch.setattr(
        "mtg_bot.training.trainer.auto_device", fake_auto_device,
    )

    cfg = NNPimcConfig(
        weights_path=str(weights), samples=2, inner_nodes=4,
        batch_size=1, device="auto",
    )
    # Constructing the spec must NOT have called auto_device — that's
    # the entire point of the sentinel.
    assert call_count["n"] == 0

    strategy = cfg.build(seed=0)
    assert call_count["n"] == 1
    mcts = strategy.inner_factory()
    assert str(mcts.evaluator.device) == "cpu"


# ---------- NNPimcConfig.architecture (separate-trunks dispatch) -----------


def test_nn_pimc_config_architecture_default_is_none() -> None:
    """``architecture`` defaults to ``None`` — meaning "use
    NNEvaluator's default" (which is "shared"). Preserves back-compat
    with v1-v15 versions whose metadata predates the field."""
    cfg = NNPimcConfig(weights_path="x.pt", samples=4, inner_nodes=200)
    assert cfg.architecture is None


def test_nn_pimc_config_architecture_round_trips_through_dict() -> None:
    """Explicit ``architecture`` survives JSON serialization so
    remote shards instantiate the matching dual-head class."""
    cfg = NNPimcConfig(
        weights_path="x.pt", samples=4, inner_nodes=200,
        architecture="separate",
        policy_hidden_sizes=(64, 32),
    )
    rebuilt = NNPimcConfig.from_dict(cfg.to_dict())
    assert rebuilt.architecture == "separate"
    assert rebuilt.policy_hidden_sizes == (64, 32)
    assert rebuilt == cfg


def test_nn_pimc_config_architecture_missing_in_dict_defaults_to_none() -> None:
    """Older serializations without ``architecture`` (every saved
    config before v16) decode cleanly with ``architecture=None``
    via ``.get(...)``."""
    minimal = {
        "weights_path": "old.pt", "samples": 4, "inner_nodes": 200,
    }
    rebuilt = NNPimcConfig.from_dict(minimal)
    assert rebuilt.architecture is None
    assert rebuilt.policy_hidden_sizes is None


def test_nn_pimc_config_architecture_threaded_into_evaluator(
    tmp_path: Path,
) -> None:
    """``architecture="separate"`` flows through to ``NNEvaluator``,
    which in turn instantiates ``_ValuePolicyMLPSeparate``."""
    import torch
    from mtg_bot.search.nn_evaluator import (
        NNEvaluator, _ValuePolicyMLPSeparate,
    )
    # Save a separate-trunks state_dict so the build() path has
    # something to load.
    a = NNEvaluator(with_policy=True, architecture="separate")
    weights_path = tmp_path / "sep.pt"
    torch.save(a.model.state_dict(), weights_path)
    cfg = NNPimcConfig(
        weights_path=str(weights_path),
        samples=2, inner_nodes=4, batch_size=1,
        with_policy=True, puct=True,
        architecture="separate",
    )
    strategy = cfg.build(seed=0)
    mcts = strategy.inner_factory()
    assert isinstance(mcts.evaluator.model, _ValuePolicyMLPSeparate)


# ---------- NNServerPimcConfig ----------------------------------------------


def test_nn_server_pimc_config_build_uses_running_server(
    tmp_path: Path,
) -> None:
    """``NNServerPimcConfig.build`` constructs an ``NNEvalClient``
    against a live ``NNEvalServer``. We start a server in-process,
    pass its handle into the config, and verify build() produces a
    PIMC strategy. The strategy is constructed but not exercised —
    we only need to land lines 256-272."""
    weights = _save_value_weights(tmp_path)
    server = NNEvalServer(
        weights_path=weights, device="cpu", with_policy=False,
        batch_size=1, batch_wait_ms=0.0,
    )
    with server:
        handle = server.handle()
        cfg = NNServerPimcConfig(
            server_handle=handle,
            samples=2, inner_nodes=4, batch_size=1,
            with_policy=False, puct=False,
        )
        strategy = cfg.build(seed=1)
        try:
            assert isinstance(strategy, PIMC)
            assert strategy.samples == 2
        finally:
            # Close the inner_factory's evaluator client by building
            # an MCTS instance, then closing its connection. PIMC
            # builds the inner search lazily, so no client is open
            # yet from `cfg.build` itself; the inner_factory closure
            # is what would open one. We exercise that path here so
            # the server can shut down cleanly when the `with` block
            # exits.
            pass


def test_nn_server_pimc_config_build_with_puct_and_policy(
    tmp_path: Path,
) -> None:
    """Same shape as above but with ``with_policy=True`` and
    ``puct=True``. Covers the policy-server build path."""
    weights = _save_policy_weights(tmp_path)
    server = NNEvalServer(
        weights_path=weights, device="cpu", with_policy=True,
        batch_size=1, batch_wait_ms=0.0,
    )
    with server:
        handle = server.handle()
        cfg = NNServerPimcConfig(
            server_handle=handle,
            samples=1, inner_nodes=2, batch_size=1,
            with_policy=True, puct=True,
        )
        strategy = cfg.build(seed=0)
        assert isinstance(strategy, PIMC)


def test_nn_pimc_config_build_reads_scalar_schema_from_sidecar(
    tmp_path: Path,
) -> None:
    """A sidecar WITHOUT ``scalar_schema_version`` is a pre-v2
    checkpoint: the build path must size ``scalar_proj`` for schema 1
    so legacy structured weights load on every worker."""
    import json as _json

    from mtg_bot.training.structured_model import (
        StructuredModel, StructuredModelConfig,
    )

    torch.manual_seed(0)
    cfg = StructuredModelConfig(
        emb_dim=32, n_heads=2, n_layers=1, hidden_dim=64,
        action_history_n=4, scalar_schema_version=1,
    )
    weights = tmp_path / "v_legacy" / "weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save(StructuredModel(cfg).state_dict(), weights)
    with (weights.parent / "metadata.json").open("w", encoding="utf-8") as f:
        _json.dump({
            "version": "v_legacy",
            "architecture": "structured",
            "card_vocab": current_vocab_dict(),
            "training": {
                "structured_config": {
                    "emb_dim": 32, "n_heads": 2, "n_layers": 1,
                    "hidden_dim": 64, "action_history_n": 4,
                },
            },
        }, f)
    pimc_cfg = NNPimcConfig(
        weights_path=str(weights),
        samples=1, inner_nodes=2, batch_size=1, with_policy=True,
        architecture="structured",
    )
    strategy = pimc_cfg.build(seed=0)
    assert isinstance(strategy, PIMC)
