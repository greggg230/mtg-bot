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
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    ARCHITECTURE_SHARED, ARCHITECTURE_SEPARATE,
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
    """
    if architecture == ARCHITECTURE_SHARED:
        return _ValuePolicyMLP(hidden_sizes=hidden_sizes)
    if architecture == ARCHITECTURE_SEPARATE:
        return _ValuePolicyMLPSeparate(
            hidden_sizes=hidden_sizes,
            policy_hidden_sizes=policy_hidden_sizes,
        )
    raise ValueError(
        f"unknown architecture {architecture!r}; "
        f"expected one of {SUPPORTED_ARCHITECTURES}"
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
        _action_encoding()._check_version(action_encoding_version)
        self.device = torch.device(device)
        self.with_policy = with_policy
        self.architecture = architecture
        self.action_encoding_version = action_encoding_version
        if with_policy:
            self.model: nn.Module = _build_dual_head_model(
                architecture,
                hidden_sizes=hidden_sizes,
                policy_hidden_sizes=policy_hidden_sizes,
            )
        else:
            self.model = _ValueMLP(hidden_sizes=hidden_sizes)
        if weights_path is not None:
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
            engine = self._engine_state(s)
            nonterminal_indices.append(i)
            nonterminal_features.append(
                extract_features(engine, player_id)
            )

        out: list[float] = [0.0] * len(states)
        for i, v in terminal_results:
            out[i] = v
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
        vocab_size = ae.ACTION_VOCAB_SIZE
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

        engine = self._engine_state(state)
        features = extract_features(engine, player_id)
        mask_np = ae.legal_action_mask(
            state, player_id, version=self.action_encoding_version,
        )
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
        vocab_size = ae.ACTION_VOCAB_SIZE
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
            engine = self._engine_state(s)
            nonterminal_indices.append(i)
            nonterminal_features.append(
                extract_features(engine, player_id)
            )
            nonterminal_masks.append(
                ae.legal_action_mask(
                    s, player_id,
                    version=self.action_encoding_version,
                ),
            )

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
        ae = _action_encoding()
        vocab_size = ae.ACTION_VOCAB_SIZE
        if state.is_terminal():
            winner = state.winner()
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if winner == player_id else -1.0
            logits = np.zeros(vocab_size, dtype=np.float32)
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
        ae = _action_encoding()
        vocab_size = ae.ACTION_VOCAB_SIZE
        out: list[tuple[float, np.ndarray]] = [
            (0.0, np.zeros(vocab_size, dtype=np.float32))
            for _ in states
        ]
        if not states:
            return out
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
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
            engine = self._engine_state(s)
            nonterminal_indices.append(i)
            nonterminal_features.append(
                extract_features(engine, player_id)
            )
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

    def save(self, path: Path) -> None:
        """Persist ``state_dict`` to disk. Standard PyTorch convention."""
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def _engine_state(state: GameState) -> EngineGameState:
        """Reach through the adapter (or accept a raw engine state
        directly, mirroring MtgHeuristicEvaluator's helper)."""
        engine = getattr(state, "engine_state", None)
        if engine is not None:
            return engine
        return state  # assume already an EngineGameState
