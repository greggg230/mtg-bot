"""Neural-network evaluator. Phase C of the learned-evaluator work.

Wraps a small PyTorch MLP in the ``Evaluator`` interface so MCTS can
call it like any other evaluator. Phase D adds the training loop that
fits the weights from self-play data; this PR ships the architecture,
the inference path, and load/save plumbing — no training code.

Architecture: 51-dim input (FEATURE_DIM from features.py) → 128 →
128 → 64 → 1, ReLU activations between, ``tanh`` on the output to
clamp into [-1, 1] (the Evaluator contract). ~25k parameters.
Tiny by NN standards; trains in seconds on CPU. Easy to scale up
later — change ``HIDDEN_SIZES`` and retrain.

Inference path:

1. ``extract_features(state, viewer_id)`` → np.float32[51]
2. Convert to torch tensor + add batch dimension
3. ``model.forward()`` → scalar
4. Return as Python float

CPU-only here. Phase D adds CUDA for *training* on the 5090; for
*inference* during MCTS, the per-call Python↔CUDA round-trip
overhead is much higher than running a tiny MLP on CPU.

Batched inference: ``batch_evaluate(states, player_id)`` stacks
the feature vectors into a single ``(B, FEATURE_DIM)`` tensor and
runs one forward pass — the path that lets MCTS amortize the
Python↔C overhead across the batch. MCTS opts in via its
``batch_size`` knob; ``batch_size=1`` falls through to the
per-state path. CUDA would also pay off here once batched
inference is the dominant rollout cost.

Save / load uses ``torch.save(model.state_dict(), path)`` — standard
PyTorch convention. Old saved weights become incompatible if
``FEATURE_DIM`` changes (new features) or ``HIDDEN_SIZES`` changes
(new architecture). That's the cost of fixed-shape networks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtg_bot.features import FEATURE_DIM, extract_features
from mtg_bot.game import GameState as EngineGameState
from mtg_bot.search import eval_counters
from mtg_bot.search.evaluator import Evaluator
from mtg_bot.search.protocol import GameState, PlayerID

# ``mtg_bot.training.action_encoding`` lookups are deferred to first
# call (the ``_action_encoding`` helper below). The naïve
# ``from mtg_bot.training.action_encoding import ...`` triggers
# ``mtg_bot.training.__init__``, which in turn imports
# ``mtg_bot.search.PIMC`` — circular while ``mtg_bot.search.__init__``
# is still in the middle of importing us. Lazy resolution side-steps
# the cycle: by the time anyone constructs a ``_ValuePolicyMLP`` or
# calls ``evaluate_with_policy``, both packages have finished loading.
_ACTION_ENCODING_MODULE: Optional[object] = None


def _action_encoding() -> Any:
    """Return the ``mtg_bot.training.action_encoding`` module,
    importing it on first call. Used by the dual-head paths to fetch
    ``ACTION_VOCAB_SIZE`` and ``legal_action_mask`` without inducing
    a circular import at this module's load time."""
    global _ACTION_ENCODING_MODULE
    if _ACTION_ENCODING_MODULE is None:
        import importlib
        _ACTION_ENCODING_MODULE = importlib.import_module(
            "mtg_bot.training.action_encoding",
        )
    return _ACTION_ENCODING_MODULE


# ``mtg_bot.training.structured_model`` and ``state_tokenizer`` are
# only needed for ``architecture='structured'`` evaluators. Importing
# them eagerly would (a) trigger ``mtg_bot.training.__init__`` (same
# cycle worry as action_encoding above) and (b) pull torch's
# transformer machinery into the import path of every NNEvaluator,
# even the legacy 51-dim MLP path that doesn't use them. Resolve
# lazily so the structured-model code path stays opt-in.
_STRUCTURED_MODEL_MODULE: Optional[object] = None
_STATE_TOKENIZER_MODULE: Optional[object] = None


def _structured_model() -> Any:
    """Return the ``mtg_bot.training.structured_model`` module,
    importing it on first call."""
    global _STRUCTURED_MODEL_MODULE
    if _STRUCTURED_MODEL_MODULE is None:
        import importlib
        _STRUCTURED_MODEL_MODULE = importlib.import_module(
            "mtg_bot.training.structured_model",
        )
    return _STRUCTURED_MODEL_MODULE


def _state_tokenizer() -> Any:
    """Return the ``mtg_bot.training.state_tokenizer`` module,
    importing it on first call. Slice E talks to the tokenizer via
    this helper so the evaluator stays decoupled from the eager
    import — and so tests can monkey-patch the module attribute
    on this module to inject a fake tokenizer.
    """
    global _STATE_TOKENIZER_MODULE
    if _STATE_TOKENIZER_MODULE is None:
        import importlib
        _STATE_TOKENIZER_MODULE = importlib.import_module(
            "mtg_bot.training.state_tokenizer",
        )
    return _STATE_TOKENIZER_MODULE

# Hidden-layer widths. 3 hidden layers, ReLU between. ~25k params at
# these sizes (51 → 128 → 128 → 64 → 1).
#
# History: the 2026-05-02 architecture-scaling experiment bumped
# this to (512, 512, 256) ≈ 420k params (commit fd095a1) testing
# the hypothesis "v2 didn't beat heuristic because the 25k network
# was capacity-bound." v3 (the 420k network trained on the same
# 398k samples) was strictly WORSE: 23-77 vs heuristic vs v2's
# 45-55 — capacity wasn't the constraint. The bigger network
# fit the heuristic-driven training distribution tighter and
# generalized worse to the MCTS-driven evaluation distribution,
# and value-only training without policy supervision can't
# direct search any more than the heuristic does.
#
# The actual gating items are A4 (policy head — the network
# learns to predict MCTS visit-count distributions per state, not
# just outcomes) and A5 (PUCT MCTS — uses the policy as a search
# prior). Once those land we can re-explore capacity with a
# richer training signal. Until then, the 25k arch is the
# known-OK baseline.
HIDDEN_SIZES: tuple[int, ...] = (128, 128, 64)


class _ValueMLP(nn.Module):
    """The actual network. FEATURE_DIM → HIDDEN_SIZES... → 1, with
    ReLU between hidden layers and tanh on output.

    Defined as a separate ``nn.Module`` so save/load uses the
    standard ``state_dict`` pattern (not pickling the whole
    NNEvaluator wrapper, which would make weights non-portable
    across small refactors).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        # Final value head: single scalar, tanh-squashed to [-1, 1].
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ValuePolicyMLP(nn.Module):
    """Two-headed network: shared trunk → (value scalar, policy logits).

    Trunk shape mirrors ``_ValueMLP`` (FEATURE_DIM → HIDDEN_SIZES with
    ReLU between) so we can ablate the policy head against the
    value-only baseline at the same trunk capacity. Heads diverge from
    the trunk's last hidden activation:

    - Value head: ``Linear(h_last, 1) → Tanh`` — same shape and range
      as the value-only network's output. Compatible with the
      ``Evaluator`` contract.
    - Policy head: ``Linear(h_last, ACTION_VOCAB_SIZE)`` returning raw
      logits. The caller (``evaluate_with_policy``) applies the legal-
      action mask and then softmax, so the network never has to learn
      to zero out illegal slots — the mask handles that exactly.

    Param count at HIDDEN_SIZES = (128, 128, 64) and ACTION_VOCAB_SIZE
    = 517: trunk ~25k + policy head 64*517 + 517 ≈ 33.6k → ~58k total.
    The policy head dominates; trunk depth is the lever to scale up
    later if the data demands it.

    Used by ``NNEvaluator(with_policy=True)``. The state_dict layout
    differs from ``_ValueMLP`` because the heads are separate modules
    — old v1/v2 weights cannot load into this network and that's fine
    (different architecture entirely; v4 trains fresh).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
        policy_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Resolve ACTION_VOCAB_SIZE lazily — see the import-cycle
        # note at the top of the module. When called at construction
        # time, both packages are fully loaded.
        if policy_dim is None:
            policy_dim = _action_encoding().ACTION_VOCAB_SIZE
        trunk_layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            trunk_layers.append(nn.Linear(prev, size))
            trunk_layers.append(nn.ReLU())
            prev = size
        self.trunk = nn.Sequential(*trunk_layers)
        # Value head: tanh-squashed scalar in [-1, 1].
        self.value_head = nn.Sequential(
            nn.Linear(prev, 1),
            nn.Tanh(),
        )
        # Policy head: raw logits. No softmax here — the masking +
        # softmax happens at the use site so illegal slots get -inf
        # before normalization (numerically stable, exact zero on
        # illegal slots after softmax).
        self.policy_head = nn.Linear(prev, policy_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (value, policy_logits) with shapes (B, 1) and
        (B, ACTION_VOCAB_SIZE)."""
        h = self.trunk(x)
        value = self.value_head(h)
        policy_logits = self.policy_head(h)
        return value, policy_logits


class _ValuePolicyMLPSeparate(nn.Module):
    """Two-headed network with SEPARATE trunks for value and policy.

    Where ``_ValuePolicyMLP`` ties both heads to a single shared trunk
    (and the louder value gradient saturates the trunk's bottleneck
    representation, crowding the policy head's branchy-row signal out),
    this variant builds two parallel MLPs — ``value_trunk`` and
    ``policy_trunk`` — that consume the raw input independently. Each
    head sits on top of its own trunk's last hidden activation, so the
    two heads no longer compete for shared capacity.

    Public API matches ``_ValuePolicyMLP``: ``forward(x) →
    (value, policy_logits)`` with shapes ``(B, 1)`` and
    ``(B, ACTION_VOCAB_SIZE)``. ``NNEvaluator`` and the trainer can
    use either class through identical call sites; the dispatch on
    architecture happens at construction time.

    State_dict layout differs from the shared-trunk variant — keys
    are ``value_trunk.*`` + ``policy_trunk.*`` + ``value_head.*`` +
    ``policy_head.*`` instead of the shared ``trunk.*``. Saved weights
    from the two architectures are NOT cross-loadable; the metadata's
    ``architecture`` field tells the loader which class to instantiate.

    Param count at HIDDEN_SIZES = (128, 128, 64) and ACTION_VOCAB_SIZE
    = 517: roughly 2x the shared-trunk variant (each trunk independently
    holds the same parameter count as the shared trunk; the heads are
    unchanged).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
        policy_hidden_sizes: Optional[tuple[int, ...]] = None,
        policy_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if policy_dim is None:
            policy_dim = _action_encoding().ACTION_VOCAB_SIZE
        if policy_hidden_sizes is None:
            policy_hidden_sizes = hidden_sizes

        def _build_trunk(sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
            layers: list[nn.Module] = []
            prev = input_dim
            for size in sizes:
                layers.append(nn.Linear(prev, size))
                layers.append(nn.ReLU())
                prev = size
            return nn.Sequential(*layers), prev

        self.value_trunk, v_last = _build_trunk(hidden_sizes)
        self.policy_trunk, p_last = _build_trunk(policy_hidden_sizes)
        self.value_head = nn.Sequential(
            nn.Linear(v_last, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(p_last, policy_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_features = self.value_trunk(x)
        value = self.value_head(v_features)
        p_features = self.policy_trunk(x)
        policy_logits = self.policy_head(p_features)
        return value, policy_logits


# Architecture identifier strings for ``NNEvaluator`` /
# ``train_value_network`` dispatch. Recorded into each version's
# ``metadata.json`` so loaders can route weights to the right class.
ARCHITECTURE_SHARED = "shared"
ARCHITECTURE_SEPARATE = "separate"
# ``structured`` selects the per-object-token transformer trunk
# (``mtg_bot.training.structured_model.StructuredModel``) — the Step 1
# NN-representation rewrite. This architecture does NOT consume the
# flat 51-dim feature vector; the inference path tokenizes the live
# engine state via ``state_tokenizer.tokenize_state`` and forwards a
# variable-length per-token batch through the transformer trunk.
# Training routes through ``train_structured_model`` (instead of the
# flat-feature ``train_value_network``) so the resulting weights get a
# ``metadata.json`` that dispatches the loader to ``StructuredModel``.
# See ``docs/NN_REPRESENTATION_ROADMAP.md`` for the full design.
ARCHITECTURE_STRUCTURED = "structured"
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    ARCHITECTURE_SHARED, ARCHITECTURE_SEPARATE, ARCHITECTURE_STRUCTURED,
)


def _build_dual_head_model(
    architecture: str,
    *,
    hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
    policy_hidden_sizes: Optional[tuple[int, ...]] = None,
) -> nn.Module:
    """Instantiate the dual-head model class matching ``architecture``.

    Centralized so trainer + NNEvaluator dispatch the same way. Raises
    ``ValueError`` on an unknown architecture string.

    NOTE: ``ARCHITECTURE_STRUCTURED`` is NOT handled here — the
    structured model has a fundamentally different input shape (per-
    token dict, not a flat ``(B, FEATURE_DIM)`` tensor) and is built
    by ``_build_structured_model`` instead. Mixing the dispatch would
    invite an MLP getting fed structured input (or vice versa) on a
    typo. Raising here for ``"structured"`` keeps the boundary loud.
    """
    if architecture == ARCHITECTURE_SHARED:
        return _ValuePolicyMLP(hidden_sizes=hidden_sizes)
    if architecture == ARCHITECTURE_SEPARATE:
        return _ValuePolicyMLPSeparate(
            hidden_sizes=hidden_sizes,
            policy_hidden_sizes=policy_hidden_sizes,
        )
    if architecture == ARCHITECTURE_STRUCTURED:
        # The structured trunk is the StructuredModel transformer; it
        # does NOT consume the flat 51-dim feature vector this builder
        # is meant for. The training dispatch routes around this
        # function via train_structured_model, and the loader-side
        # instantiation routes through _build_structured_model below.
        # Surfacing the wrong path as a hard error here keeps a future
        # caller from accidentally instantiating an MLP trunk for a
        # structured-architecture version.
        raise ValueError(
            "architecture='structured' uses StructuredModel, not the "
            "flat-feature MLP — route through _build_structured_model "
            "instead of _build_dual_head_model."
        )
    raise ValueError(
        f"unknown architecture {architecture!r}; "
        f"expected one of {SUPPORTED_ARCHITECTURES}"
    )


def _build_structured_model(
    *,
    emb_dim: Optional[int] = None,
    n_heads: Optional[int] = None,
    n_layers: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    action_history_n: Optional[int] = None,
    n_printings: Optional[int] = None,
    action_vocab_size: Optional[int] = None,
    enable_hand_belief_head: Optional[bool] = None,
    dropout: Optional[float] = None,
    scalar_schema_version: Optional[int] = None,
    value_head_input: Optional[str] = None,
    value_temperature: Optional[float] = None,
    value_bias: Optional[float] = None,
) -> nn.Module:
    """Instantiate a ``StructuredModel`` with optional config overrides.

    Any field left ``None`` falls back to the default in
    ``StructuredModelConfig``. The structured-model module is imported
    lazily (see ``_structured_model``) so this function is callable
    even from contexts where the training package isn't fully loaded.

    Used by ``NNEvaluator.__init__`` when ``architecture='structured'``
    and (eventually) by the matching trainer slice when constructing
    the model for training. Centralizing the config-building keeps
    the loader and trainer in lockstep on hyperparam handling — e.g.
    when ``ModelMetadata.training`` carries ``action_history_n=64``,
    both sides pick it up the same way.
    """
    sm = _structured_model()
    cfg_kwargs: dict[str, Any] = {}
    for key, value in (
        ("emb_dim", emb_dim),
        ("n_heads", n_heads),
        ("n_layers", n_layers),
        ("hidden_dim", hidden_dim),
        ("action_history_n", action_history_n),
        ("n_printings", n_printings),
        ("action_vocab_size", action_vocab_size),
        ("enable_hand_belief_head", enable_hand_belief_head),
        ("dropout", dropout),
        ("scalar_schema_version", scalar_schema_version),
        # Value-head routing + post-hoc calibration. Both are config,
        # not state_dict, so a checkpoint trained under either setting
        # loads either way; the loader passes what metadata recorded.
        ("value_head_input", value_head_input),
        ("value_temperature", value_temperature),
        ("value_bias", value_bias),
    ):
        if value is not None:
            cfg_kwargs[key] = value
    config = sm.StructuredModelConfig(**cfg_kwargs)
    return sm.StructuredModel(config)


def _resolve_structured_dim(
    training: dict,
    flat_key: str,
    nested_key: str,
) -> Any:
    """Read a structured-trunk dim from either of the two on-disk
    metadata layouts.

    Returns ``training[flat_key]`` if present (forward-compat with
    hand-patched / older metadata that promoted nested values to flat
    keys, including v13's iter_019 workaround). Otherwise returns
    ``training["structured_config"][nested_key]`` if the trainer's
    nested-dict layout is present. Falls through to ``None`` when
    neither layout carries the dim — the caller forwards ``None`` to
    ``NNEvaluator.__init__``, which in turn lets
    ``StructuredModelConfig`` apply its own field default.

    This is the iter_019 BUG #2 compat shim: ``mtg_bot.cli.train``
    writes the nested layout (``training["structured_config"] =
    {emb_dim, n_heads, ...}``) but the loader historically only read
    the flat keys, so any non-default-trunk bootstrap silently fell
    back to ``StructuredModelConfig`` defaults and produced a
    weight-shape mismatch at load time. v5 worked as a bootstrap only
    because its dims happened to match the defaults; v13 was the first
    non-default-trunk bootstrap and surfaced the bug. Reading both
    layouts (flat wins) keeps existing flat-key metadata loading
    unchanged while fixing the nested-only case.
    """
    if flat_key in training:
        return training[flat_key]
    nested = training.get("structured_config") or {}
    return nested.get(nested_key)


# --------------------------------------------------------------------------
# Metadata -> evaluator kwargs: the ONE place that decides how a
# checkpoint's recorded value-head options are applied.
#
# Every path that turns registered weights into an evaluator --
# ``NNEvaluator.from_metadata`` (calibrate-value, replay annotator),
# ``NNPimcConfig.build`` (every worker self-play / eval game, iterate's
# local arenas, ``mtg_bot ladder``, capture, bench) and the SHM eval
# server -- goes through ``structured_kwargs_from_metadata`` (structured
# trunks) or ``check_flat_metadata_value_options`` (MLP trunks). Before
# 2026-10-04 ``NNPimcConfig.build`` read the trunk dims itself and never
# passed ``value_calibration`` / ``value_head_input``, so every played
# game (self-play, arenas, ladders) ran the value head uncalibrated
# while the held-out gate graded the calibrated one.
# --------------------------------------------------------------------------


def _meta_field(metadata: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a ``ModelMetadata`` (or any attribute bag) or a
    raw ``metadata.json`` dict. Missing / ``None`` -> ``default``."""
    if isinstance(metadata, dict):
        value = metadata.get(key, default)
    else:
        value = getattr(metadata, key, default)
    return default if value is None else value


def resolve_value_calibration(
    metadata: Any,
    *,
    structured: bool = True,
    apply_value_calibration: bool = True,
) -> tuple[float, float]:
    """``(temperature, bias)`` to apply to the value head's pre-tanh
    logit, from ``metadata.value_calibration`` (falling back to
    ``metadata.training.value_calibration``).

    Absent / empty / ``identity`` -> ``(1.0, 0.0)``, i.e. bit-identical
    to an uncalibrated head. ``apply_value_calibration=False`` returns
    the identity unconditionally (the explicit ``+nocal`` opt-out; the
    recorded fit is still validated so a corrupt record is not hidden
    behind the flag).

    Fails loudly (``ValueError``) on anything this builder cannot apply
    exactly: the post-tanh ``output_shrink`` method, an unknown method,
    an ``identity`` record carrying non-identity parameters, a
    ``logit_temperature`` record with ``scale != 1``, or any non-identity
    calibration on a flat MLP trunk (which has no logit hook). A
    silently-dropped calibration changes the leaf values search backs
    up, so "can't apply" must never degrade to "ignore".
    """
    from mtg_bot.training.value_calibration import (
        CALIBRATION_IDENTITY,
        CALIBRATION_LOGIT_TEMPERATURE,
        CALIBRATION_OUTPUT_SHRINK,
        ValueCalibration,
    )

    training = _meta_field(metadata, "training", {}) or {}
    raw = (
        _meta_field(metadata, "value_calibration", None)
        or training.get("value_calibration")
    )
    calibration = ValueCalibration.from_dict(raw)
    method = calibration.method
    if method == CALIBRATION_OUTPUT_SHRINK:
        raise ValueError(
            "metadata.value_calibration uses the 'output_shrink' method, "
            "which the evaluator cannot apply (it calibrates the "
            "pre-tanh logit). Re-run `mtg_bot calibrate-value` for "
            "this version."
        )
    if method not in (CALIBRATION_IDENTITY, CALIBRATION_LOGIT_TEMPERATURE):
        raise ValueError(
            f"metadata.value_calibration has unknown method {method!r}; "
            f"expected {CALIBRATION_IDENTITY!r} or "
            f"{CALIBRATION_LOGIT_TEMPERATURE!r}."
        )
    if method == CALIBRATION_IDENTITY:
        if not (
            calibration.temperature == 1.0
            and calibration.bias == 0.0
            and calibration.scale == 1.0
        ):
            raise ValueError(
                "metadata.value_calibration says method 'identity' but "
                f"carries temperature={calibration.temperature} "
                f"bias={calibration.bias} scale={calibration.scale}; "
                "refusing to guess which one is meant."
            )
        return 1.0, 0.0
    # logit_temperature
    if calibration.scale != 1.0:
        raise ValueError(
            "metadata.value_calibration 'logit_temperature' record "
            f"carries scale={calibration.scale}; the evaluator applies "
            "temperature/bias on the logit only."
        )
    temperature = float(calibration.temperature)
    bias = float(calibration.bias)
    if not apply_value_calibration:
        return 1.0, 0.0
    if not structured and not (temperature == 1.0 and bias == 0.0):
        raise ValueError(
            "metadata.value_calibration is set on a flat MLP checkpoint "
            f"(temperature={temperature}, bias={bias}), but only the "
            "structured trunk exposes the pre-tanh logit the "
            "calibration applies to."
        )
    return temperature, bias


def structured_kwargs_from_metadata(
    metadata: Any,
    *,
    apply_value_calibration: bool = True,
) -> dict[str, Any]:
    """Every ``structured_*`` keyword of ``NNEvaluator`` /
    ``SharedTensorNNEvalServer`` resolved from a structured checkpoint's
    metadata (``ModelMetadata`` or the raw ``metadata.json`` dict):
    trunk dims (flat ``training.structured_*`` or nested
    ``training.structured_config``, see ``_resolve_structured_dim``),
    the scalar schema, the value-head input and the post-hoc value
    calibration (``resolve_value_calibration``).

    ``None`` values mean "the model's default"; the constructors skip
    them. An unsupported ``value_head_input`` is rejected by
    ``StructuredModelConfig`` at build time (loud ``ValueError``).
    """
    training = _meta_field(metadata, "training", {}) or {}
    _resolve = _resolve_structured_dim
    # Scalar schema: a real ``ModelMetadata`` always carries
    # ``scalar_schema_version`` (``from_dict`` defaults a missing key to
    # 1 -- the pre-v2 3-slot layout). A raw ``metadata.json`` without
    # the key, and with no ``training`` mirror of it, is a pre-v2
    # checkpoint too, so it resolves to 1 as well; duck-typed stand-ins
    # with neither fall through to ``StructuredModelConfig``'s default.
    scalar_schema_version = _meta_field(metadata, "scalar_schema_version")
    if scalar_schema_version is None:
        scalar_schema_version = _resolve(
            training,
            "structured_scalar_schema_version",
            "scalar_schema_version",
        )
    if scalar_schema_version is None and isinstance(metadata, dict):
        scalar_schema_version = 1
    temperature, bias = resolve_value_calibration(
        metadata,
        structured=True,
        apply_value_calibration=apply_value_calibration,
    )
    return {
        "structured_scalar_schema_version": (
            int(scalar_schema_version)
            if scalar_schema_version is not None else None
        ),
        "structured_value_head_input": _resolve(
            training, "structured_value_head_input", "value_head_input",
        ),
        "structured_value_temperature": temperature,
        "structured_value_bias": bias,
        "structured_emb_dim": _resolve(
            training, "structured_emb_dim", "emb_dim",
        ),
        "structured_n_heads": _resolve(
            training, "structured_n_heads", "n_heads",
        ),
        "structured_n_layers": _resolve(
            training, "structured_n_layers", "n_layers",
        ),
        "structured_hidden_dim": _resolve(
            training, "structured_hidden_dim", "hidden_dim",
        ),
        "structured_action_history_n": _resolve(
            training, "structured_action_history_n", "action_history_n",
        ),
        "structured_n_printings": _resolve(
            training, "structured_n_printings", "n_printings",
        ),
        "structured_action_vocab_size": _resolve(
            training, "structured_action_vocab_size", "action_vocab_size",
        ),
        "structured_enable_hand_belief_head": _resolve(
            training,
            "structured_enable_hand_belief_head",
            "enable_hand_belief_head",
        ),
        "structured_dropout": _resolve(
            training, "structured_dropout", "dropout",
        ),
    }


def check_flat_metadata_value_options(
    metadata: Any,
    *,
    apply_value_calibration: bool = True,
) -> None:
    """Refuse value-head options a flat MLP trunk cannot honour.

    The MLP architectures (``shared`` / ``separate``) have neither a
    pre-tanh logit hook nor alternative value-head inputs, so a
    checkpoint whose metadata declares a non-identity calibration
    (unless the caller explicitly opted out of applying it) or a
    ``value_head_input`` other than the shared trunk raises here
    instead of loading with those options silently dropped.
    """
    resolve_value_calibration(
        metadata,
        structured=False,
        apply_value_calibration=apply_value_calibration,
    )
    training = _meta_field(metadata, "training", {}) or {}
    head_input = _resolve_structured_dim(
        training, "structured_value_head_input", "value_head_input",
    )
    if head_input not in (None, "trunk"):
        raise ValueError(
            f"metadata declares value_head_input={head_input!r} on a "
            "flat MLP checkpoint; only the structured trunk supports "
            "alternative value-head inputs."
        )


class NNEvaluator(Evaluator):
    """Wraps a ``_ValueMLP`` in the ``Evaluator`` interface.

    Construction:
    - No args → fresh random-init network. Useful for tests and
      for the first iteration of training (which initializes from
      random and then trains via self-play).
    - ``weights_path`` → load a saved ``state_dict`` from disk.
      The architecture is fixed (FEATURE_DIM input, HIDDEN_SIZES
      hidden), so saved weights are tied to those constants;
      bumping either invalidates old saves.

    Inference is single-state per call (one tensor, one forward pass)
    on CPU. ``torch.no_grad()`` and ``model.eval()`` skip gradient
    tracking and any train-mode behaviors (dropout, batchnorm) for
    speed. Output is converted back to a Python float because the
    Evaluator contract returns ``float``, not tensor.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        *,
        device: str = "cpu",
        with_policy: bool = False,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
        architecture: str = ARCHITECTURE_SHARED,
        policy_hidden_sizes: Optional[tuple[int, ...]] = None,
        action_encoding_version: str = "positional_v1",
        # ----- structured-architecture hyperparameters --------------
        # All optional; ``None`` defers to ``StructuredModelConfig``
        # defaults. Only consulted when ``architecture='structured'``.
        # The matching ``ModelMetadata.training`` dict carries the
        # values the saved weights were trained against; loaders read
        # them out and pass them here so the constructed model's
        # parameter shapes match the saved state_dict (mismatched
        # shapes surface as a PyTorch load_state_dict error — the
        # same loud failure as the dual-head shape-mismatch case).
        structured_emb_dim: Optional[int] = None,
        structured_n_heads: Optional[int] = None,
        structured_n_layers: Optional[int] = None,
        structured_hidden_dim: Optional[int] = None,
        structured_action_history_n: Optional[int] = None,
        structured_n_printings: Optional[int] = None,
        structured_action_vocab_size: Optional[int] = None,
        structured_enable_hand_belief_head: Optional[bool] = None,
        structured_dropout: Optional[float] = None,
        # Which ``token_scalars`` row layout the structured trunk was
        # trained against (``ModelMetadata.scalar_schema_version``).
        # ``None`` defers to ``StructuredModelConfig``'s default (the
        # current schema) — fresh random-init evaluators get the new
        # layout; loaders ALWAYS pass the metadata value so pre-v2
        # checkpoints (schema 1, 3-wide ``scalar_proj``) keep loading.
        # Threaded into every ``tokenize_state`` call so the input
        # width matches the projection.
        structured_scalar_schema_version: Optional[int] = None,
        # Which representation the structured value head reads
        # (``StructuredModelConfig.value_head_input``) and the
        # post-hoc value calibration fitted on the held-out split.
        # ``from_metadata`` fills all three from metadata.json;
        # ``None`` keeps the model defaults (shared trunk, identity
        # calibration), which is exactly how every pre-2026-09-25
        # checkpoint behaved.
        structured_value_head_input: Optional[str] = None,
        structured_value_temperature: Optional[float] = None,
        structured_value_bias: Optional[float] = None,
        # The card vocabulary a STRUCTURED ``weights_path`` was trained
        # on (``ModelMetadata.card_vocab`` / a ``CardVocab``). ``None``
        # resolves it from the checkpoint's own ``metadata.json`` or
        # ``<weights>.card_vocab.json`` stamping sidecar; a checkpoint with
        # neither is REFUSED (``MissingCardVocabError``) — never
        # guessed. A vocabulary that differs from the current one is
        # remapped by card name at load (see
        # ``mtg_bot.training.card_vocab``). Ignored for the MLP
        # architectures and for random-init (no ``weights_path``).
        card_vocab: Any = None,
    ) -> None:
        # ``with_policy=False`` (default) preserves bit-exact backward
        # compat with v1/v2 weights — same _ValueMLP, same forward
        # path, same state_dict layout. ``with_policy=True`` switches
        # to the two-headed _ValuePolicyMLP and unlocks
        # ``evaluate_with_policy`` / ``batch_evaluate_with_policy``.
        # The two state_dict layouts are NOT cross-compatible; loading
        # a value-only checkpoint into a dual-head evaluator (or vice
        # versa) raises a shape-mismatch error from PyTorch — that's
        # the intended loud failure mode.
        #
        # ``hidden_sizes`` overrides the trunk shape so the same
        # NNEvaluator class can host networks of any capacity. The
        # caller is responsible for matching the shape to the saved
        # weights — load-time mismatches surface as PyTorch
        # state_dict shape errors, the same loud failure as for the
        # value-only-vs-dual-head mismatch above. Registry lookups
        # (``evaluate_nn.py``, ``NNPimcConfig.build``) read the
        # ``hidden_sizes`` field from each version's metadata and
        # pass it here, so per-version architectures coexist.
        # ``architecture`` selects WHICH dual-head class hosts the
        # policy / value heads when ``with_policy=True``:
        # - ``"shared"`` (default, back-compat): single shared trunk
        #   feeds both heads. The state_dict layout v1-v15 was trained
        #   under, so old weights load cleanly when this is left at
        #   the default.
        # - ``"separate"``: two parallel trunks, one per head. Each
        #   trunk independently consumes the raw input and feeds its
        #   own head. The two heads stop competing for shared trunk
        #   capacity, fixing the ~3:1 value-vs-policy gradient
        #   saturation observed in iter_002 v13. State_dict keys are
        #   ``value_trunk.*`` + ``policy_trunk.*`` (vs the shared
        #   variant's ``trunk.*``) — incompatible with shared-trunk
        #   weights, so the architecture choice is recorded in each
        #   version's metadata.json and threaded through here.
        # Ignored when ``with_policy=False`` (the value-only
        # ``_ValueMLP`` has no policy head and so no architectural
        # choice to make).
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of "
                f"{SUPPORTED_ARCHITECTURES}; got {architecture!r}"
            )
        # Validate the action-encoding version eagerly so a typo at
        # NNEvaluator-construction time surfaces with a clear message
        # rather than silently falling through to v1 behavior at the
        # first ``legal_action_mask`` call. Validation goes via the
        # action_encoding module's helper to keep the supported set
        # definition single-sourced.
        ae = _action_encoding()
        ae._check_version(action_encoding_version)
        # Policy-head width is a property of the action-encoding
        # version (517 for positional_v1 / canonical_v2, 797+ for
        # canonical_v3 -- ``action_encoding.action_vocab_size``).
        self.action_vocab_size = ae.action_vocab_size(action_encoding_version)
        if (
            structured_action_vocab_size is not None
            and int(structured_action_vocab_size) != self.action_vocab_size
        ):
            raise ValueError(
                f"structured_action_vocab_size={structured_action_vocab_size}"
                f" disagrees with action_encoding_version="
                f"{action_encoding_version!r} ({self.action_vocab_size} "
                f"slots); the checkpoint metadata is inconsistent"
            )
        if (
            ae.has_resolution_slots(action_encoding_version)
            and architecture != ARCHITECTURE_STRUCTURED
        ):
            raise ValueError(
                f"action_encoding_version={action_encoding_version!r} is "
                f"only supported by the structured architecture (the MLP "
                f"policy heads are {ae.ACTION_VOCAB_SIZE} wide)"
            )
        # The resolution block is scored only at resolution pauses
        # (``_policy_slot_range``); None for versions without one.
        self._resolution_slot_range = (
            (ae.SLOT_RESOLUTION_BASE, self.action_vocab_size)
            if ae.has_resolution_slots(action_encoding_version) else None
        )
        self.device = torch.device(device)
        # ``is_structured`` is the per-call branch the evaluate /
        # batch_evaluate / evaluate_with_policy / ... paths consult.
        # Set once at __init__ time per the slice's "detect by
        # inspecting ModelMetadata.architecture at load time, don't
        # dispatch per-call" rule.
        self.is_structured = architecture == ARCHITECTURE_STRUCTURED
        # The structured trunk always has a policy head (and a value
        # head). ``with_policy`` is the legacy MLP-evaluator's switch
        # between the value-only and dual-head builds; for the
        # structured trunk we force it on so the with-policy entry
        # points (``evaluate_with_policy``, etc.) are always
        # available. Callers that pass ``with_policy=False`` to a
        # structured evaluator are upgraded silently — the cost
        # of running the policy head is folded into the same
        # transformer forward pass, so there's nothing to opt out of.
        if self.is_structured:
            with_policy = True
        self.with_policy = with_policy
        self.architecture = architecture
        self.action_encoding_version = action_encoding_version
        if self.is_structured:
            self.model: nn.Module = _build_structured_model(
                emb_dim=structured_emb_dim,
                n_heads=structured_n_heads,
                n_layers=structured_n_layers,
                hidden_dim=structured_hidden_dim,
                action_history_n=structured_action_history_n,
                n_printings=structured_n_printings,
                action_vocab_size=self.action_vocab_size,
                enable_hand_belief_head=(
                    structured_enable_hand_belief_head
                ),
                dropout=structured_dropout,
                scalar_schema_version=structured_scalar_schema_version,
                value_head_input=structured_value_head_input,
                value_temperature=structured_value_temperature,
                value_bias=structured_value_bias,
            )
            # Scalar schema the tokenizer must emit — read back off
            # the built model's config so a ``None`` override and an
            # explicit value both land on the width ``scalar_proj``
            # actually has.
            self._structured_scalar_schema_version = (
                self.model.config.scalar_schema_version
            )
            # The action-history window the tokenizer should use.
            # Mirrors the model config so the input shape's
            # action-token slice matches the trunk's expectation. The
            # tokenizer accepts any ``n``; we read the loaded model's
            # config to stay in lockstep with the trained
            # hyperparameter.
            self._structured_action_history_n = (
                self.model.config.action_history_n
            )
            # Tokenizer cap so unknown / out-of-table printings
            # bucket deterministically into the model's embedding
            # table — see ``state_tokenizer.printing_id``.
            self._structured_n_printings_cap = (
                self.model.config.n_printings
            )
        elif with_policy:
            self.model = _build_dual_head_model(
                architecture,
                hidden_sizes=hidden_sizes,
                policy_hidden_sizes=policy_hidden_sizes,
            )
        else:
            self.model = _ValueMLP(hidden_sizes=hidden_sizes)
        # What happened to the checkpoint's printing rows at load
        # (structured + weights only); ``None`` otherwise.
        self.card_vocab_report = None
        # The card vocabulary this evaluator's printing rows are
        # indexed by: always the CURRENT one for a structured model
        # (random init is born in it; a loaded checkpoint is remapped
        # into it). ``save`` stamps it next to the weights. ``None``
        # for the MLP architectures (no printing identity).
        self.card_vocab = None
        if self.is_structured:
            from mtg_bot.training.card_vocab import current_card_vocab

            self.card_vocab = current_card_vocab()
        if weights_path is not None and self.is_structured:
            # The single structured-weights load path: resolve the
            # checkpoint's stored card vocabulary (refusing an
            # unstamped one), remap its per-printing rows to the
            # current vocabulary by name, and rebuild the model when
            # the remap grew the embedding table.
            from mtg_bot.training.card_vocab import (
                load_structured_checkpoint,
            )

            self.model, self.card_vocab_report = load_structured_checkpoint(
                self.model, Path(weights_path),
                map_location=self.device, card_vocab=card_vocab,
            )
            # The tokenizer must bucket against the LOADED table size.
            self._structured_n_printings_cap = (
                self.model.config.n_printings
            )
        elif weights_path is not None:
            state_dict = torch.load(
                weights_path, map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _evaluate_nonterminal(
        self, state: GameState, player_id: PlayerID,
    ) -> float:
        # Base ``Evaluator.evaluate`` already short-circuited terminals,
        # so reaching here is exactly one non-terminal NN forward
        # (whether structured or MLP — counted at the public-method
        # boundary, not the ``self.model``/``_structured_forward`` site,
        # so the architecture split can't double- or under-count).
        eval_counters.record_nn_eval(1)
        if self.is_structured:
            # Tokenize + forward through the StructuredModel. The
            # batched helper handles the structured input pipeline for
            # both per-state and per-batch entry points; here we
            # construct a 1-element batch and pull the single value
            # out.
            values, _ = self._structured_forward([state], player_id)
            return float(values[0])
        engine = self._engine_state(state)
        features = extract_features(engine, player_id)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            out = self.model(x)
            # Dual-head model returns (value, policy_logits); the
            # value-only path returns a single tensor. Same forward-
            # call site works for both because the value tensor is
            # always the first (or only) element.
            value_t = out[0] if isinstance(out, tuple) else out
            value = value_t.item()
        return value

    def batch_evaluate(
        self, states: Sequence[GameState], player_id: PlayerID,
    ) -> list[float]:
        """Single forward pass over a stacked batch of feature
        vectors. Captures the 5–20× win the per-state path forfeits
        to Python↔C round-trip overhead — at batch size 1 there's
        almost no compute being done; at batch 32 the same wall-
        clock buys 32 evaluations.

        Terminal states short-circuit to ±1/0 (matches the
        per-state path) without touching the network. Non-terminal
        states are extracted, stacked into a ``(B, FEATURE_DIM)``
        tensor, run through ``self.model``, and unpacked. The
        returned list preserves input order and includes the
        terminal short-circuits intermixed with NN-scored values.
        """
        if not states:
            return []
        # Partition into (index, kind, value-or-engine-state) so we
        # can run the NN only over the non-terminal subset and
        # stitch the terminals back in by index.
        terminal_results: list[tuple[int, float]] = []
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
        nonterminal_states: list[GameState] = []
        for i, s in enumerate(states):
            if s.is_terminal():
                winner = s.winner()
                if winner is None:
                    terminal_results.append((i, 0.0))
                else:
                    terminal_results.append(
                        (i, 1.0 if winner == player_id else -1.0)
                    )
                continue
            nonterminal_indices.append(i)
            if self.is_structured:
                nonterminal_states.append(s)
            else:
                engine = self._engine_state(s)
                nonterminal_features.append(
                    extract_features(engine, player_id)
                )

        out: list[float] = [0.0] * len(states)
        for i, v in terminal_results:
            out[i] = v
        # One batched forward over the non-terminal subset (structured
        # or MLP). Terminals were already split out above and never
        # touch the network, so they must not count.
        if nonterminal_indices:
            eval_counters.record_nn_eval(len(nonterminal_indices))
        if self.is_structured:
            if nonterminal_states:
                values_t, _ = self._structured_forward(
                    nonterminal_states, player_id,
                )
                values = values_t.tolist()
                for idx, v in zip(nonterminal_indices, values):
                    out[idx] = v
            return out
        if nonterminal_features:
            stacked = np.stack(nonterminal_features, axis=0)
            with torch.no_grad():
                x = torch.from_numpy(stacked).to(self.device)
                model_out = self.model(x)
                value_t = (
                    model_out[0] if isinstance(model_out, tuple)
                    else model_out
                )
                # ``value_t`` is ``(B, 1)``; squeeze the value dim and
                # pull values back to the host as a Python list.
                values = value_t.squeeze(-1).cpu().tolist()
            for idx, v in zip(nonterminal_indices, values):
                out[idx] = v
        return out

    def evaluate_with_policy(
        self, state: GameState, player_id: PlayerID,
    ) -> tuple[float, np.ndarray]:
        """Forward pass that returns both heads.

        Returns ``(value, policy)`` where ``policy`` is a length-
        ACTION_VOCAB_SIZE float32 array. Illegal slots are exactly
        ``0.0``; legal slots sum to ``1.0`` (within float epsilon).
        Masking is applied to the logits as ``-inf`` *before* softmax,
        which is the numerically-safe way to zero out illegal slots
        and renormalize the legal subset in one step.

        Terminal states short-circuit without calling the network:
        value comes from the engine's true result (+1/-1/0 same as
        ``Evaluator.evaluate``) and the policy is the all-zeros vector
        because no decision is taken at a terminal — there is no
        legal action to predict.

        Requires ``with_policy=True`` at construction. Raises
        ``RuntimeError`` otherwise so callers can't silently get a
        garbage uniform-over-junk distribution from the value-only
        network's missing policy head.
        """
        if not self.with_policy:
            raise RuntimeError(
                "evaluate_with_policy requires with_policy=True at "
                "NNEvaluator construction"
            )
        ae = _action_encoding()
        vocab_size = self.action_vocab_size
        # Terminal short-circuit. Mirrors Evaluator.evaluate: +1 if the
        # asked-for player won, -1 if they lost, 0 for a draw. Policy
        # is the all-zeros vector (no legal action at a terminal).
        if state.is_terminal():
            winner = state.winner()
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if winner == player_id else -1.0
            policy = np.zeros(vocab_size, dtype=np.float32)
            return value, policy

        # Past the terminal short-circuit: exactly one non-terminal
        # (value, policy) forward, structured or MLP.
        eval_counters.record_nn_eval(1)
        mask_np = ae.legal_action_mask(
            state, player_id, version=self.action_encoding_version,
        )
        if self.is_structured:
            lo, hi = self._policy_slot_range([state], [mask_np])
            values_t, logits_t = self._structured_forward(
                [state], player_id, policy_slot_range=(lo, hi),
            )
            value = float(values_t[0].item())
            row_logits = logits_t[0]
            sub_mask = mask_np[lo:hi]
            mask_t = torch.from_numpy(sub_mask).to(self.device)
            masked = row_logits.masked_fill(~mask_t, float("-inf"))
            if not sub_mask.any():
                policy = np.zeros(vocab_size, dtype=np.float32)
            else:
                policy_t = F.softmax(masked, dim=-1)
                sub_policy = policy_t.cpu().numpy().astype(
                    np.float32, copy=False,
                )
                if (lo, hi) == (0, vocab_size):
                    policy = sub_policy
                else:
                    policy = np.zeros(vocab_size, dtype=np.float32)
                    policy[lo:hi] = sub_policy
            return value, policy
        engine = self._engine_state(state)
        features = extract_features(engine, player_id)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            value_t, logits_t = self.model(x)
            value = value_t.item()
            # Mask illegal slots with -inf so post-softmax probability
            # is exactly zero. ``mask_np`` is bool[ACTION_VOCAB_SIZE]
            # — True == legal. Convert to a torch bool tensor on the
            # right device.
            mask_t = torch.from_numpy(mask_np).to(self.device)
            logits_t = logits_t.squeeze(0)
            # ``masked_fill`` is the canonical "set positions to -inf"
            # idiom; the ``~mask_t`` selects illegal slots.
            masked = logits_t.masked_fill(~mask_t, float("-inf"))
            # If no slot is legal (shouldn't happen on a non-terminal
            # state, but be defensive — softmax of all -inf is NaN),
            # return an all-zeros policy. Caller handling is the same
            # as a terminal in that case.
            if not mask_np.any():
                policy = np.zeros(vocab_size, dtype=np.float32)
            else:
                policy_t = F.softmax(masked, dim=-1)
                policy = policy_t.cpu().numpy().astype(
                    np.float32, copy=False,
                )
        return value, policy

    def batch_evaluate_with_policy(
        self,
        states: Sequence[GameState],
        player_id: PlayerID,
    ) -> list[tuple[float, np.ndarray]]:
        """Batched companion to ``evaluate_with_policy``.

        Single forward pass over the stacked non-terminal features;
        per-state masking + softmax happens on the host (the masks
        differ per state, so they can't be fused into a single matmul
        anyway, and CPU softmax over a 517-vector is a rounding
        error compared to the matmul cost).

        Terminal states short-circuit per-element to (±1/0,
        all-zeros policy) — same convention as the single-state path.
        Result list preserves input order.
        """
        if not self.with_policy:
            raise RuntimeError(
                "batch_evaluate_with_policy requires with_policy=True "
                "at NNEvaluator construction"
            )
        ae = _action_encoding()
        vocab_size = self.action_vocab_size
        out: list[tuple[float, np.ndarray]] = [
            (0.0, np.zeros(vocab_size, dtype=np.float32))
            for _ in states
        ]
        if not states:
            return out

        # Collect terminals + nonterminals into separate streams,
        # mirroring batch_evaluate's pattern.
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
        nonterminal_masks: list[np.ndarray] = []
        nonterminal_states: list[GameState] = []
        for i, s in enumerate(states):
            if s.is_terminal():
                winner = s.winner()
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if winner == player_id else -1.0
                out[i] = (
                    value, np.zeros(vocab_size, dtype=np.float32),
                )
                continue
            nonterminal_indices.append(i)
            nonterminal_masks.append(
                ae.legal_action_mask(
                    s, player_id,
                    version=self.action_encoding_version,
                ),
            )
            if self.is_structured:
                nonterminal_states.append(s)
            else:
                engine = self._engine_state(s)
                nonterminal_features.append(
                    extract_features(engine, player_id)
                )

        # One batched (value, policy) forward over the non-terminal
        # subset; terminals were split out above and never count.
        if nonterminal_indices:
            eval_counters.record_nn_eval(len(nonterminal_indices))
        if self.is_structured:
            if nonterminal_states:
                lo, hi = self._policy_slot_range(
                    nonterminal_states, nonterminal_masks,
                )
                values_t, logits_t = self._structured_forward(
                    nonterminal_states, player_id,
                    policy_slot_range=(lo, hi),
                )
                values = values_t.tolist()
                logits_np = logits_t.cpu().numpy()
                for k, idx in enumerate(nonterminal_indices):
                    mask_np = nonterminal_masks[k][lo:hi]
                    if not mask_np.any():
                        policy = np.zeros(
                            vocab_size, dtype=np.float32,
                        )
                    else:
                        row = logits_np[k].astype(np.float32, copy=True)
                        row[~mask_np] = -np.inf
                        row -= row.max()
                        exp = np.exp(row)
                        sub_policy = (exp / exp.sum()).astype(
                            np.float32, copy=False,
                        )
                        if (lo, hi) == (0, vocab_size):
                            policy = sub_policy
                        else:
                            policy = np.zeros(vocab_size, dtype=np.float32)
                            policy[lo:hi] = sub_policy
                    out[idx] = (values[k], policy)
            return out

        if nonterminal_features:
            stacked = np.stack(nonterminal_features, axis=0)
            with torch.no_grad():
                x = torch.from_numpy(stacked).to(self.device)
                value_t, logits_t = self.model(x)
                values = value_t.squeeze(-1).cpu().tolist()
                logits_np = logits_t.cpu().numpy()
            for k, idx in enumerate(nonterminal_indices):
                mask_np = nonterminal_masks[k]
                if not mask_np.any():
                    policy = np.zeros(
                        vocab_size, dtype=np.float32,
                    )
                else:
                    # Mask + softmax on the host. Subtract max for
                    # numerical stability before exp().
                    row = logits_np[k].astype(np.float32, copy=True)
                    row[~mask_np] = -np.inf
                    row -= row.max()  # safe: max is finite (legal slot exists)
                    exp = np.exp(row)
                    policy = (exp / exp.sum()).astype(
                        np.float32, copy=False,
                    )
                out[idx] = (values[k], policy)
        return out

    def evaluate_logits(
        self, state: GameState, player_id: PlayerID,
    ) -> tuple[float, np.ndarray]:
        """Forward pass returning value + RAW (unmasked, un-softmaxed)
        policy logits.

        Returns ``(value, logits)`` where ``logits`` is a length-
        ACTION_VOCAB_SIZE float32 array straight off the policy head.
        No legal-action mask is applied; no softmax is taken. Callers
        that want a probability distribution must mask + softmax
        themselves with whichever mask is appropriate to their use
        case.

        Motivation: ISMCTS wants the prior at an information-set node
        to span the *union* of legal actions across the
        determinizations seen at that node, not the per-iteration
        determinization's narrower mask. Re-querying the network on
        every union-mask growth is wasteful — instead the search
        caches raw logits once per node and re-masks cheaply on the
        host as the union grows.

        Terminal states short-circuit identically to
        ``evaluate_with_policy``: value is the engine's true result,
        logits is the all-zeros vector (no decision to predict).

        Requires ``with_policy=True`` at construction.
        """
        if not self.with_policy:
            raise RuntimeError(
                "evaluate_logits requires with_policy=True at "
                "NNEvaluator construction"
            )
        vocab_size = self.action_vocab_size
        if state.is_terminal():
            winner = state.winner()
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if winner == player_id else -1.0
            logits = np.zeros(vocab_size, dtype=np.float32)
            return value, logits
        # Non-terminal value+raw-logits forward (belief / ISMCTS prior
        # path). One forward per call, structured or MLP.
        eval_counters.record_nn_eval(1)
        if self.is_structured:
            values_t, logits_t = self._structured_forward(
                [state], player_id,
            )
            value = float(values_t[0].item())
            logits = logits_t[0].cpu().numpy().astype(
                np.float32, copy=False,
            )
            return value, logits
        engine = self._engine_state(state)
        features = extract_features(engine, player_id)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            value_t, logits_t = self.model(x)
            value = value_t.item()
            logits = logits_t.squeeze(0).cpu().numpy().astype(
                np.float32, copy=False,
            )
        return value, logits

    def batch_evaluate_logits(
        self,
        states: Sequence[GameState],
        player_id: PlayerID,
    ) -> list[tuple[float, np.ndarray]]:
        """Batched companion to ``evaluate_logits``. Returns one
        ``(value, logits)`` tuple per state, in input order, with
        terminal states short-circuited to (±1/0, all-zeros)."""
        if not self.with_policy:
            raise RuntimeError(
                "batch_evaluate_logits requires with_policy=True at "
                "NNEvaluator construction"
            )
        vocab_size = self.action_vocab_size
        out: list[tuple[float, np.ndarray]] = [
            (0.0, np.zeros(vocab_size, dtype=np.float32))
            for _ in states
        ]
        if not states:
            return out
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
        nonterminal_states: list[GameState] = []
        for i, s in enumerate(states):
            if s.is_terminal():
                winner = s.winner()
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if winner == player_id else -1.0
                out[i] = (
                    value, np.zeros(vocab_size, dtype=np.float32),
                )
                continue
            nonterminal_indices.append(i)
            if self.is_structured:
                nonterminal_states.append(s)
            else:
                engine = self._engine_state(s)
                nonterminal_features.append(
                    extract_features(engine, player_id)
                )
        # One batched value+raw-logits forward over the non-terminal
        # subset; terminals split out above never count.
        if nonterminal_indices:
            eval_counters.record_nn_eval(len(nonterminal_indices))
        if self.is_structured:
            if nonterminal_states:
                values_t, logits_t = self._structured_forward(
                    nonterminal_states, player_id,
                )
                values = values_t.tolist()
                logits_np = logits_t.cpu().numpy().astype(
                    np.float32, copy=False,
                )
                for k, idx in enumerate(nonterminal_indices):
                    out[idx] = (values[k], logits_np[k].copy())
            return out
        if nonterminal_features:
            stacked = np.stack(nonterminal_features, axis=0)
            with torch.no_grad():
                x = torch.from_numpy(stacked).to(self.device)
                value_t, logits_t = self.model(x)
                values = value_t.squeeze(-1).cpu().tolist()
                logits_np = logits_t.cpu().numpy().astype(
                    np.float32, copy=False,
                )
            for k, idx in enumerate(nonterminal_indices):
                out[idx] = (values[k], logits_np[k].copy())
        return out

    # ----------------------------------------------------------------
    # Structured-architecture forward pipeline
    # ----------------------------------------------------------------

    def _policy_slot_range(
        self, states: Sequence[GameState], masks: Sequence[np.ndarray],
    ) -> tuple[int, int]:
        """Slot range the policy decoder must score for a batch.

        Versions without a resolution block: the full width. With one
        (``canonical_v3``): the base 517 slots when no state in the
        batch is at a resolution pause, the resolution block when all
        are, the full width for a mixed batch. A mask with legal bits
        outside the chosen range (never produced by
        ``legal_action_mask``) falls back to the full width rather
        than dropping them. See ``docs/RESOLUTION_DECISIONS.md`` §7.
        """
        full = (0, self.action_vocab_size)
        res = self._resolution_slot_range
        if res is None:
            return full
        paused = [
            bool(getattr(s, "_resolution_paused", False)) for s in states
        ]
        if all(paused):
            rng = res
        elif not any(paused):
            rng = (0, res[0])
        else:
            return full
        lo, hi = rng
        for m in masks:
            if m[:lo].any() or m[hi:].any():
                return full
        return rng

    def _structured_forward(
        self,
        states: Sequence[GameState],
        viewer_id: PlayerID,
        policy_slot_range: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize each state, collate into a padded batch, and run
        the structured model.

        Returns ``(values, policy_logits)``:

        - ``values`` is a 1-D ``FloatTensor`` of length ``len(states)``
          in ``[-1, +1]`` (the trunk's value head is tanh-activated).
        - ``policy_logits`` is a ``FloatTensor`` of shape
          ``(B, action_vocab_size)`` — raw logits straight off the
          cross-attention policy decoder — or ``(B, hi - lo)`` for the
          slots ``policy_slot_range=(lo, hi)`` selects. Callers mask +
          softmax themselves (matching the pre-existing dual-head MLP
          path's contract).

        All terminal-state short-circuiting is the caller's
        responsibility — this helper assumes every state in ``states``
        is non-terminal and should be tokenized + forwarded. Returning
        tensors (not numpy) lets each public method do its
        per-state masking + post-processing in the existing place.

        Action-history strategy
        -----------------------
        Each state's ``recent_actions`` (the bounded tuple of
        ``ActionEvent``s ``MtgGameStateAdapter.apply`` records and
        ``clone()`` shares with every search-tree child / determinized
        world — see ``mtg_bot.search.action_history``) is passed to
        the tokenizer. This is the SAME attribute ``data_gen`` reads
        when it writes training rows, so the trunk sees at search
        time exactly the history signal it was trained on. States
        without the attribute (raw engine states, the Rust provider,
        duck-typed test fakes) fall back to an empty history — the
        tokenizer accepts an empty sequence (the action-token slice
        then contributes zero rows).
        """
        tokenizer = _state_tokenizer()
        token_dicts: list[dict[str, np.ndarray]] = []
        for s in states:
            engine = self._engine_state(s)
            token_dicts.append(
                tokenizer.tokenize_state(
                    engine,
                    int(viewer_id),
                    getattr(s, "recent_actions", ()),
                    n=self._structured_action_history_n,
                    n_printings_cap=self._structured_n_printings_cap,
                    scalar_schema_version=(
                        self._structured_scalar_schema_version
                    ),
                ),
            )
        batch = _collate_structured_token_dicts(
            token_dicts, device=self.device,
        )
        with torch.no_grad():
            if (
                policy_slot_range is None
                or tuple(policy_slot_range) == (0, self.action_vocab_size)
            ):
                out = self.model(batch)
            else:
                out = self.model(
                    batch, policy_slot_range=tuple(policy_slot_range),
                )
        values = out["value"]
        if values.dim() == 0:
            # Single-element batch on some PyTorch versions returns a
            # 0-D tensor when the model squeezes; normalize to 1-D so
            # callers always index it as values[i].
            values = values.unsqueeze(0)
        return values, out["policy_logits"]

    @classmethod
    def from_metadata(
        cls,
        weights_path: Path,
        metadata: Any,
        *,
        device: str = "cpu",
        apply_value_calibration: bool = True,
    ) -> "NNEvaluator":
        """Construct an evaluator dispatched on a
        ``ModelMetadata.architecture`` field.

        Centralizes the loader-side dispatch that today is replicated
        across iterate / capture / eval / bench call sites — each
        reads ``metadata.architecture``, ``metadata.hidden_sizes``,
        ``metadata.action_encoding_version``, etc. and forwards them
        into ``NNEvaluator(...)`` by hand. The structured architecture
        adds another set of fields the existing call sites would have
        to learn about; folding the dispatch here keeps the structured
        path's hyperparam wiring single-sourced.

        For the structured architecture, the additional hyperparams
        live under ``metadata.training`` (which is a free-form dict
        the trainer fills in) — same convention the existing dual-
        head architectures use for ``policy_hidden_sizes``. Two
        on-disk layouts exist in the wild and both are accepted (see
        ``_resolve_structured_dim``):

        - Flat keys: ``structured_emb_dim`` / ``structured_n_heads``
          / ``structured_n_layers`` / ``structured_hidden_dim`` /
          ``structured_action_history_n`` / ``structured_n_printings``
          / ``structured_action_vocab_size`` /
          ``structured_enable_hand_belief_head`` /
          ``structured_dropout``.
        - Nested under ``training.structured_config = {emb_dim,
          n_heads, n_layers, hidden_dim, action_history_n,
          n_printings}`` — this is what ``mtg_bot.cli.train``'s
          structured branch writes today (see the trainer's
          ``train_config["structured_config"]``).

        The resolver prefers the flat key when present and falls back
        to the nested-dict key otherwise; missing-from-both falls
        through to ``StructuredModelConfig``'s default. The dual-read
        is the iter_019 BUG #2 fix — the trainer was writing nested
        while the loader read flat, so any non-default-trunk bootstrap
        silently fell back to the defaults until the v13 attempt
        surfaced the resulting weight-shape mismatch.

        ``metadata`` may be a ``ModelMetadata``, any attribute bag, or
        the raw ``metadata.json`` dict (what ``NNPimcConfig.build``
        reads next to the weights). The structured kwargs -- trunk
        dims, scalar schema, ``value_head_input`` and the post-hoc
        ``value_calibration`` -- come from
        :func:`structured_kwargs_from_metadata`, the single resolver
        every evaluator-construction path shares.
        ``apply_value_calibration=False`` loads the head uncalibrated
        (the ladder's ``+nocal`` control); the default applies it.
        Value-head options the architecture cannot honour raise
        ``ValueError`` (see :func:`resolve_value_calibration` /
        :func:`check_flat_metadata_value_options`).
        """
        architecture = _meta_field(metadata, "architecture", "shared")
        action_encoding_version = _meta_field(
            metadata, "action_encoding_version", "positional_v1",
        )
        if architecture == ARCHITECTURE_STRUCTURED:
            return cls(
                weights_path=weights_path,
                device=device,
                architecture=architecture,
                action_encoding_version=action_encoding_version,
                # The checkpoint's stored card vocabulary (a sidecar
                # stamp is already merged in by Registry.load_metadata).
                # Empty -> NNEvaluator resolves it next to weights_path
                # and refuses an unstamped checkpoint.
                card_vocab=(_meta_field(metadata, "card_vocab") or None),
                **structured_kwargs_from_metadata(
                    metadata,
                    apply_value_calibration=apply_value_calibration,
                ),
            )
        # MLP architectures (shared / separate). Hidden sizes default
        # to HIDDEN_SIZES when metadata's hidden_sizes is empty or
        # absent (back-compat for v1-/v2-style metadata). No logit hook
        # and no alternative value-head inputs: refuse metadata that
        # declares either rather than dropping it.
        check_flat_metadata_value_options(
            metadata, apply_value_calibration=apply_value_calibration,
        )
        hidden_sizes_meta = _meta_field(metadata, "hidden_sizes", ()) or ()
        hidden_sizes = (
            tuple(hidden_sizes_meta) if hidden_sizes_meta else HIDDEN_SIZES
        )
        with_policy = bool(_meta_field(metadata, "with_policy", False))
        training = _meta_field(metadata, "training", {}) or {}
        policy_hidden_sizes_meta = (
            tuple(training.get("policy_hidden_sizes") or ()) or None
        )
        return cls(
            weights_path=weights_path,
            device=device,
            with_policy=with_policy,
            hidden_sizes=hidden_sizes,
            architecture=architecture,
            policy_hidden_sizes=policy_hidden_sizes_meta,
            action_encoding_version=action_encoding_version,
        )

    def save(self, path: Path) -> None:
        """Persist ``state_dict`` to disk. Standard PyTorch convention.

        A structured model is also stamped with the card vocabulary its
        printing rows are indexed by (``<path>.card_vocab.json``), so the
        file can be loaded back — under any later vocabulary — without
        guessing. MLP weights carry no printing identity and get none.
        """
        torch.save(self.model.state_dict(), path)
        if self.is_structured:
            from mtg_bot.training.card_vocab import write_sidecar

            write_sidecar(
                self.card_vocab, Path(path),
                provenance={"source": "NNEvaluator.save"},
            )

    @staticmethod
    def _engine_state(state: GameState) -> EngineGameState:
        """Reach through the adapter (or accept a raw engine state
        directly, mirroring MtgHeuristicEvaluator's helper)."""
        engine = getattr(state, "engine_state", None)
        if engine is not None:
            return engine
        return state  # assume already an EngineGameState


def _collate_structured_token_dicts(
    token_dicts: Sequence[dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pad a sequence of per-state token dicts into a single batched
    tensor dict suitable for ``StructuredModel.forward``.

    The tokenizer emits one dict per state with these keys:

    - ``token_type``     int64 ``(seq_len_i,)``
    - ``token_emb_ids``  int64 ``(seq_len_i, N_EMB_FIELDS)``
    - ``token_scalars``  float32 ``(seq_len_i, N_SCALAR_FIELDS)``
    - ``attention_mask`` bool  ``(seq_len_i,)`` — every emitted row True

    Different states have different ``seq_len_i`` because hand /
    graveyard / exile / battlefield sizes vary. The collator picks
    ``max_len = max(seq_len_i)`` and pads each row up to it. Padded
    rows have ``attention_mask=False``; their integer / scalar
    contents are zero so the model's NA-token (id 0 on every
    categorical embedding) absorbs them cleanly.

    The output tensors are placed on ``device`` so the subsequent
    ``model.forward`` doesn't pay a per-row host-to-device transfer.

    This is the inference-time collate. The structured-trainer slice
    is writing the matching training-time collator — both should
    produce identical shapes given identical inputs. Keeping the two
    consistent is a "tested via shape contract" rather than via
    shared code; the inputs to each collator come from the same
    tokenizer output format, so the inference path's collator
    matching the training path's is the unit-test invariant.
    """
    if not token_dicts:
        raise ValueError("token_dicts must be non-empty")
    n_emb_fields = token_dicts[0]["token_emb_ids"].shape[1]
    n_scalar_fields = token_dicts[0]["token_scalars"].shape[1]
    # Width is schema-dependent (3 for scalar schema v1, more for v2)
    # and inferred from the first dict; every row in one batch must
    # come from the same tokenizer schema.
    for i, d in enumerate(token_dicts):
        w = d["token_scalars"].shape[1]
        if w != n_scalar_fields:
            raise ValueError(
                f"token_dicts[{i}] has token_scalars width {w} but "
                f"token_dicts[0] has {n_scalar_fields}; all states in a "
                f"batch must be tokenized under the same "
                f"scalar_schema_version"
            )
    max_len = max(d["token_type"].shape[0] for d in token_dicts)
    batch_size = len(token_dicts)

    token_type = np.zeros((batch_size, max_len), dtype=np.int64)
    token_emb_ids = np.zeros(
        (batch_size, max_len, n_emb_fields), dtype=np.int64,
    )
    token_scalars = np.zeros(
        (batch_size, max_len, n_scalar_fields), dtype=np.float32,
    )
    attention_mask = np.zeros((batch_size, max_len), dtype=bool)
    for i, d in enumerate(token_dicts):
        n = d["token_type"].shape[0]
        token_type[i, :n] = d["token_type"]
        token_emb_ids[i, :n] = d["token_emb_ids"]
        token_scalars[i, :n] = d["token_scalars"]
        attention_mask[i, :n] = d["attention_mask"]

    return {
        "token_type": torch.from_numpy(token_type).to(device),
        "token_emb_ids": torch.from_numpy(token_emb_ids).to(device),
        "token_scalars": torch.from_numpy(token_scalars).to(device),
        "attention_mask": torch.from_numpy(attention_mask).to(device),
    }
