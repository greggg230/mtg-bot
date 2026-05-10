"""PyTorch training loop for the value (and optional policy) network.

Loads (features, outcomes [, policy_targets, action_masks]) from
data_gen, fits ``_ValueMLP`` (value-only) or ``_ValuePolicyMLP``
(dual-head) from nn_evaluator.py, saves weights to disk in the
state_dict format NNEvaluator loads.

Loss:

- Value-only mode: MSE between value head output and the game-
  outcome target (-1 / 0 / +1).
- AlphaZero mode: same value-MSE PLUS cross-entropy between the
  policy head's masked log-softmax and the per-row visit
  distribution. Policy loss only contributes for rows where the
  visit distribution carries actual supervision (sum > 0.5);
  rows from baseline strategies or data-doubled non-active-viewer
  views go through value-only.

Mode selection is automatic: pass ``policy_targets`` +
``action_masks`` arrays to ``train_value_network`` for AlphaZero
training; omit them for value-only. Function name kept (rather than
renamed to ``train_actor_critic``) so the existing CLI / tests /
imports keep working — the AlphaZero supervision is opt-in.

Adam optimizer. Train/val split for monitoring overfitting. CUDA
when available; CPU pin to a single thread because OMP/MKL barrier
sync dominates on tiny matmuls (revisit when we scale past ~100k
params).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mtg_bot.search.nn_evaluator import (
    ARCHITECTURE_SEPARATE,
    ARCHITECTURE_SHARED,
    HIDDEN_SIZES,
    SUPPORTED_ARCHITECTURES,
    _build_dual_head_model,
    _ValueMLP,
)
from mtg_bot.training.progress import atomic_write_json, git_short_sha


@dataclass
class TrainingHistory:
    """One training-run summary.

    ``train_losses`` / ``val_losses`` are per-epoch mean TOTAL loss
    (value MSE + λ·policy CE in AlphaZero mode; value MSE alone in
    value-only mode).

    The two ``*_value_*`` and ``*_policy_*`` lists carry the
    per-component per-epoch averages so the trainer can report each
    head's progress separately. In value-only mode the policy
    components are NaN (no policy head to track).

    ``best_val_loss`` is the lowest TOTAL val loss seen — the epoch
    we'd checkpoint at if we did early stopping.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_value_losses: list[float] = field(default_factory=list)
    val_value_losses: list[float] = field(default_factory=list)
    train_policy_losses: list[float] = field(default_factory=list)
    val_policy_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_val_epoch: int = -1
    weights_path: Optional[Path] = None


def auto_device() -> str:
    """Return the best available torch device string for this host.

    Order of preference: ``"cuda"`` > ``"mps"`` > ``"cpu"``. CUDA wins
    on machines with NVIDIA GPUs (master / Win10 remote); MPS picks
    up Apple Silicon (Mac remote); CPU is the universal fallback.

    The ``hasattr`` check on ``torch.backends.mps`` keeps this safe
    on older torch builds where the submodule isn't present (current
    torch 2.x always exposes it, but the check costs nothing and
    documents the precondition).
    """
    if torch.cuda.is_available():
        return "cuda"
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def train_value_network(
    features: np.ndarray,
    outcomes: np.ndarray,
    *,
    weights_out_path: Path,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    val_split: float = 0.1,
    device: Optional[str] = None,
    rng_seed: int = 0,
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    notes: str = "",
    policy_targets: Optional[np.ndarray] = None,
    action_masks: Optional[np.ndarray] = None,
    policy_loss_weight: float = 1.0,
    compile_model: bool = False,
    hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
    architecture: str = ARCHITECTURE_SHARED,
    policy_hidden_sizes: Optional[tuple[int, ...]] = None,
    early_stop_patience: Optional[int] = None,
) -> TrainingHistory:
    """Fit a value (or value+policy) network to the supplied
    training data and write the final state_dict to
    ``weights_out_path``.

    ``features``: ``(N, FEATURE_DIM)`` float32. ``outcomes``: ``(N,)``
    float32 in ``{-1, 0, +1}``.

    AlphaZero mode (opt-in): pass both ``policy_targets`` (``(N,
    ACTION_VOCAB_SIZE)`` float32 — per-row normalized visit
    distribution from MCTS, all-zero rows = no policy supervision)
    and ``action_masks`` (``(N, ACTION_VOCAB_SIZE)`` bool — legal-
    slot mask per row, all-False for rows without supervision).
    With both provided, we instantiate the dual-head model
    (``_ValuePolicyMLP`` for ``architecture="shared"``,
    ``_ValuePolicyMLPSeparate`` for ``architecture="separate"``)
    instead of ``_ValueMLP``, the loss becomes
    ``MSE(v, z) + policy_loss_weight · masked_CE(p_pred, π_target)``,
    and the policy CE only sums over rows whose policy_target sum
    is > 0.5 (other rows contribute zero to the policy loss).

    ``architecture`` selects the dual-head model class:

    - ``"shared"`` (default, back-compat): single shared trunk
      feeds both heads. v1-v15 weights all use this.
    - ``"separate"``: two parallel trunks, one per head. Each trunk
      consumes the raw input independently; the heads stop competing
      for shared trunk capacity. State_dict layout differs from
      ``"shared"`` (keys are ``value_trunk.*`` + ``policy_trunk.*``
      + ``value_head.*`` + ``policy_head.*``); the choice is
      recorded into the trained version's metadata.json so loaders
      dispatch to the right class. Optional
      ``policy_hidden_sizes`` lets the policy_trunk diverge from
      ``hidden_sizes``; default mirrors ``hidden_sizes``.

    Value-only mode (default): both ``policy_targets`` and
    ``action_masks`` omitted. Trains ``_ValueMLP`` on MSE, exact
    pre-A2.3 behavior. The existing CLI / tests don't change.

    The split is deterministic in ``rng_seed`` so re-running with
    the same seed and dataset reproduces the same train/val
    partition (important for tuning loops that compare training
    runs at different hyperparameters).

    If ``progress_path`` is given, a JSON status file is written
    after each epoch (and once at startup) so an external viewer
    can poll it. Writes are atomic via ``os.replace`` — a polling
    reader never sees a half-written file. ``run_id`` is the stable
    identifier baked into that file; it defaults to the file stem.

    Returns a ``TrainingHistory`` with per-epoch losses (total +
    value + policy components separately). The weights file
    written at ``weights_out_path`` always reflects the FINAL
    epoch trained, not the best-val-loss epoch — best-checkpoint
    isn't implemented yet (but ``early_stop_patience`` will at
    least stop the loop once val loss has stopped improving, so
    the final-epoch weights are typically near the best).

    ``early_stop_patience`` (default ``None`` = disabled) caps the
    number of epochs trained AFTER the running-best val_loss epoch.
    With ``patience=5``, training breaks once the current epoch is
    5 or more epochs past ``best_val_epoch``. ``None`` preserves
    the original behavior (always run the full ``epochs`` budget);
    any int must be ``>= 1``. Requires non-empty val data
    (``val_split > 0`` and at least one row in val) — without val
    loss to track, early-stop is silently skipped. The progress-
    JSON ``total_epochs`` field reflects the ACTUAL number of
    epochs trained when early-stop fires (not the requested
    ceiling) so the UI's progress bar reads 100% at the real stop.

    Performance notes:

    - All training and validation tensors are moved to ``device_obj``
      ONCE before the epoch loop and sliced via index selection on
      device for each batch — no per-batch host→device copy. For
      the current 1.1 M × 51-feature dataset this is ~3 GB of GPU
      memory, well under a 32 GB Blackwell card.
    - ``compile_model=True`` (opt-in) wraps the model in
      ``torch.compile`` — eliminates per-op interpreter overhead by
      capturing the forward graph and fusing kernels. First batch
      pays a 10-30 s compile cost; per-epoch wall amortizes 2-3x
      faster on tiny models (Python overhead, not matmuls, was the
      bottleneck). Saved weights are unaffected — the underlying
      module's state_dict is what gets written.
    """
    if features.shape[0] != outcomes.shape[0]:
        raise ValueError(
            f"features rows {features.shape[0]} != outcomes len "
            f"{outcomes.shape[0]}"
        )
    if features.shape[0] == 0:
        raise ValueError("cannot train on an empty dataset")
    if not (0.0 <= val_split < 1.0):
        raise ValueError(
            f"val_split must be in [0, 1); got {val_split}"
        )
    if (policy_targets is None) != (action_masks is None):
        raise ValueError(
            "policy_targets and action_masks must be provided "
            "together (or both omitted for value-only training)"
        )
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {SUPPORTED_ARCHITECTURES}; "
            f"got {architecture!r}"
        )
    if early_stop_patience is not None and early_stop_patience < 1:
        raise ValueError(
            f"early_stop_patience must be >= 1 (or None to disable); "
            f"got {early_stop_patience}"
        )
    with_policy = policy_targets is not None
    if with_policy:
        if policy_targets.shape[0] != features.shape[0]:
            raise ValueError(
                f"policy_targets rows {policy_targets.shape[0]} != "
                f"features rows {features.shape[0]}"
            )
        if action_masks.shape != policy_targets.shape:
            raise ValueError(
                f"action_masks shape {action_masks.shape} != "
                f"policy_targets shape {policy_targets.shape}"
            )

    # Pin to a single torch thread on CPU. With our small batches
    # (256 rows × FEATURE_DIM=51 floats) and small model (~25k
    # params), per-batch OMP/MKL barrier sync across the host's
    # cores costs more than it saves — empirically observed an
    # ~85× slowdown going from 1 thread to 32 threads on a 32-core
    # box (248s/epoch → 2.93s/epoch). Once the model scales up to
    # ~100k+ parameters the matmul work per batch grows enough
    # that multi-threading wins; revisit this pin then.
    # GPU training is unaffected (set_num_threads governs CPU only).
    if device is None:
        device = auto_device()
    if device == "cpu":
        torch.set_num_threads(1)
    device_obj = torch.device(device)
    torch.manual_seed(rng_seed)

    # Train/val split. Shuffle indices once, slice deterministically.
    n = features.shape[0]
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Move every tensor to ``device_obj`` ONCE here, before the epoch
    # loop. The previous ``DataLoader(TensorDataset(...))`` shape did
    # numpy → host-tensor → device-tensor copy on every batch — which
    # at our model size dominated wall-clock (the matmul itself is
    # ~microseconds; the per-batch host→device hop is ~ms). For the
    # current 1.1 M × 51-feature dataset this is ~3 GB on the GPU,
    # well under the RTX 5090's 32 GB. Index-shuffled slicing of the
    # device tensors is essentially free per batch.
    x_train = torch.from_numpy(features[train_idx]).float().to(device_obj)
    y_train = torch.from_numpy(
        outcomes[train_idx]
    ).float().unsqueeze(1).to(device_obj)
    x_val = torch.from_numpy(features[val_idx]).float().to(device_obj)
    y_val = torch.from_numpy(
        outcomes[val_idx]
    ).float().unsqueeze(1).to(device_obj)

    if with_policy:
        # Per-row "has supervision" flag computed once on host and
        # moved with the rest. The flag gates the policy CE so rows
        # without MCTS visits contribute zero.
        p_train = torch.from_numpy(
            policy_targets[train_idx]
        ).float().to(device_obj)
        m_train = torch.from_numpy(
            action_masks[train_idx],
        ).bool().to(device_obj)
        has_pol_train = (
            p_train.sum(dim=-1, keepdim=True) > 0.5
        ).float()
        p_val = torch.from_numpy(
            policy_targets[val_idx]
        ).float().to(device_obj)
        m_val = torch.from_numpy(
            action_masks[val_idx]
        ).bool().to(device_obj)
        has_pol_val = (
            p_val.sum(dim=-1, keepdim=True) > 0.5
        ).float()
        # Dispatch on the architecture string. ``"shared"`` (default,
        # back-compat) builds the original single-trunk
        # ``_ValuePolicyMLP``; ``"separate"`` builds the parallel-
        # trunks ``_ValuePolicyMLPSeparate``. The trainer doesn't
        # care which beyond keeping the same forward signature
        # ``(value, policy_logits)``.
        model = _build_dual_head_model(
            architecture,
            hidden_sizes=hidden_sizes,
            policy_hidden_sizes=policy_hidden_sizes,
        ).to(device_obj)
    else:
        model = _ValueMLP(hidden_sizes=hidden_sizes).to(device_obj)

    if compile_model:
        # ``torch.compile`` fuses kernels and removes per-op Python
        # overhead — biggest win on tiny models like ours where
        # interpreter cost dominates compute. First forward pass
        # incurs ~10-30 s of compilation; amortizes across 100
        # epochs. Wrap AFTER ``.to(device_obj)`` so the compiled
        # graph captures the right device.
        #
        # ``torch.compile``'s default ``inductor`` backend depends
        # on Triton, which isn't packaged for Windows as of writing
        # (Triton is Linux-first). Probe the import once here so a
        # missing-Triton install on Windows degrades to a printed
        # warning + no compile, rather than crashing on the first
        # forward pass several seconds into training.
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
        except ImportError:
            import warnings
            warnings.warn(
                "torch.compile requested but Triton is not "
                "installed; falling back to eager execution. "
                "Triton isn't packaged for Windows yet — install "
                "on Linux to benefit, or omit --compile.",
                RuntimeWarning, stacklevel=2,
            )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    history = TrainingHistory()

    # Progress emission. Snapshot a few values up front so the writer
    # never has to recompute or worry about dataclass mutation
    # mid-write.
    progress_enabled = progress_path is not None
    started_at_iso = ""
    started_perf = 0.0
    git_commit: Optional[str] = None
    if progress_enabled:
        progress_path = Path(progress_path)
        if run_id is None:
            run_id = progress_path.stem
        started_at_iso = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        started_perf = time.perf_counter()
        git_commit = git_short_sha()

    # Effective epoch ceiling for progress reporting. Defaults to the
    # requested ``epochs`` budget; once early-stop fires we overwrite
    # this with the actual number of epochs trained so the progress
    # JSON's ``total_epochs`` matches what really ran (UI progress bar
    # then reads 100% at the real stop, not at the would-have-been
    # ceiling).
    effective_total_epochs = int(epochs)

    def _emit(status: str, current_epoch: int) -> None:
        if not progress_enabled:
            return
        elapsed = time.perf_counter() - started_perf
        done = max(current_epoch, 0)
        avg = (elapsed / done) if done > 0 else 0.0
        remaining_epochs = max(effective_total_epochs - done, 0)
        eta = avg * remaining_epochs if avg > 0 else 0.0
        payload = {
            "run_id": run_id,
            "kind": "training",
            "started_at": started_at_iso,
            "total_epochs": int(effective_total_epochs),
            "current_epoch": int(done),
            "elapsed_seconds": float(elapsed),
            "per_epoch_seconds_avg": float(avg),
            "estimated_remaining_seconds": float(eta),
            "train_losses": list(history.train_losses),
            "val_losses": list(history.val_losses),
            "train_value_losses": list(history.train_value_losses),
            "val_value_losses": list(history.val_value_losses),
            "train_policy_losses": list(history.train_policy_losses),
            "val_policy_losses": list(history.val_policy_losses),
            "policy_loss_weight": float(policy_loss_weight),
            "status": status,
            "git_commit": git_commit,
            "notes": notes,
        }
        atomic_write_json(progress_path, payload)

    # Initial "running" snapshot so the UI shows the run immediately,
    # before the first epoch finishes (epochs can be ~30s each at
    # 420k params on CPU).
    _emit("running", current_epoch=0)

    def _policy_loss(
        policy_logits: torch.Tensor,
        target_dist: torch.Tensor,
        action_mask: torch.Tensor,
        has_supervision: torch.Tensor,
    ) -> torch.Tensor:
        """Masked cross-entropy averaged over rows that have policy
        supervision. Rows with all-zero target / all-False mask
        contribute zero — the ``has_supervision`` gate scales them
        to 0 and they never reach the denominator.

        Implementation note: we use ``-1e9`` (not ``-inf``) for
        masked slots. log_softmax of an all-``-inf`` row is NaN
        (softmax denominator is 0, log of 0 is undefined), and
        ``NaN * 0`` is still NaN — would propagate through the
        gate. ``-1e9`` is finite enough to keep softmax stable
        (e^-1e9 ≈ 0 in float32 anyway) and finite enough that
        log_softmax of an all-``-1e9`` row is uniform-then-finite
        rather than NaN. The unsupervised-row contribution then
        cleanly multiplies to 0.

        Defensive: zero target mass on action_mask=False slots
        before computing CE. An inconsistent (target, mask) pair
        — target_i > 0 where mask_i = False — would otherwise
        multiply with the masked_fill ``-1e9`` to produce ~1e9
        per-row CE, which one row in 60k is enough to dominate
        an entire epoch's batch averages (observed: 7000+ loss
        with ~46 such rows in a 2.8M-row historical pool).
        """
        target_dist = target_dist * action_mask.float()
        masked_logits = policy_logits.masked_fill(~action_mask, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        per_row_ce = -(target_dist * log_probs).sum(dim=-1)
        weighted = per_row_ce * has_supervision.squeeze(-1)
        denom = has_supervision.sum().clamp(min=1.0)
        return weighted.sum() / denom

    # CPU-side generator drives the per-epoch shuffle. ``torch.randperm``
    # on CPU is reproducible with a seeded ``Generator``; the resulting
    # index tensor is moved to ``device_obj`` once per epoch and used
    # to slice the device-resident training tensors. This replaces the
    # pre-refactor ``DataLoader(TensorDataset(...))`` shape, which paid
    # a per-batch numpy → host-tensor → device-tensor copy cost.
    n_train = x_train.shape[0]
    shuffle_gen = torch.Generator(device="cpu").manual_seed(rng_seed)

    try:
        for epoch in range(epochs):
            # Train.
            model.train()
            train_total: list[float] = []
            train_value: list[float] = []
            train_policy: list[float] = []
            perm = torch.randperm(n_train, generator=shuffle_gen)
            perm = perm.to(device_obj, non_blocking=True)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                bx = x_train.index_select(0, idx)
                by = y_train.index_select(0, idx)
                optimizer.zero_grad()
                if with_policy:
                    bp = p_train.index_select(0, idx)
                    bm = m_train.index_select(0, idx)
                    bhas = has_pol_train.index_select(0, idx)
                    v_pred, p_logits = model(bx)
                    v_loss = mse(v_pred, by)
                    p_loss = _policy_loss(p_logits, bp, bm, bhas)
                    total = v_loss + policy_loss_weight * p_loss
                    train_value.append(v_loss.item())
                    train_policy.append(p_loss.item())
                else:
                    pred = model(bx)
                    total = mse(pred, by)
                    train_value.append(total.item())
                    train_policy.append(float("nan"))
                total.backward()
                optimizer.step()
                train_total.append(total.item())
            history.train_losses.append(float(np.mean(train_total)))
            history.train_value_losses.append(
                float(np.mean(train_value))
            )
            history.train_policy_losses.append(
                float(np.nanmean(train_policy))
                if any(not np.isnan(x) for x in train_policy)
                else float("nan")
            )

            # Val (skip if no val data). Tensors are already on
            # ``device_obj`` from the upfront move; no per-call copy.
            if n_val > 0:
                model.eval()
                with torch.no_grad():
                    if with_policy:
                        v_pred_val, p_logits_val = model(x_val)
                        v_loss_val = mse(v_pred_val, y_val).item()
                        p_loss_val = _policy_loss(
                            p_logits_val, p_val, m_val, has_pol_val,
                        ).item()
                        val_loss = (
                            v_loss_val
                            + policy_loss_weight * p_loss_val
                        )
                        history.val_value_losses.append(v_loss_val)
                        history.val_policy_losses.append(p_loss_val)
                    else:
                        val_pred = model(x_val)
                        val_loss = mse(val_pred, y_val).item()
                        history.val_value_losses.append(val_loss)
                        history.val_policy_losses.append(
                            float("nan")
                        )
            else:
                val_loss = float("nan")
                history.val_value_losses.append(float("nan"))
                history.val_policy_losses.append(float("nan"))
            history.val_losses.append(val_loss)

            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_val_epoch = epoch

            _emit("running", current_epoch=epoch + 1)

            # Patience-based early stopping. Only meaningful when we
            # have val data to track; with no val rows ``best_val_epoch``
            # stays -1 and val_loss is NaN, so nothing to compare.
            if (
                early_stop_patience is not None
                and n_val > 0
                and history.best_val_epoch >= 0
                and (epoch - history.best_val_epoch) >= early_stop_patience
            ):
                effective_total_epochs = epoch + 1
                print(
                    f"[trainer] early-stop fired at epoch "
                    f"{epoch + 1} (best was epoch "
                    f"{history.best_val_epoch + 1}, "
                    f"patience={early_stop_patience})",
                    file=sys.stderr,
                )
                # Re-emit so total_epochs in progress JSON reflects
                # the actual stop point before we exit the loop.
                _emit("running", current_epoch=epoch + 1)
                break
    except BaseException:
        # Mark the run failed before re-raising so a watcher knows
        # not to keep polling. BaseException so even KeyboardInterrupt
        # leaves a useful tombstone.
        try:
            _emit("failed", current_epoch=len(history.train_losses))
        except Exception:
            # Don't mask the original error with a logging error.
            pass
        raise

    # Save final weights. (Move back to CPU so the state_dict is
    # device-agnostic when loaded.) When the model is ``torch.compile``-
    # wrapped, save the underlying module — its state_dict is the
    # canonical form that ``NNEvaluator(weights_path=...)`` expects to
    # load (without any ``_orig_mod.`` key prefix).
    save_target = getattr(model, "_orig_mod", model)
    save_target.to("cpu")
    torch.save(save_target.state_dict(), weights_out_path)
    history.weights_path = weights_out_path

    # Use the count of actual training epochs run — equals ``epochs``
    # when no early-stop fired, or the stop epoch when it did.
    _emit("completed", current_epoch=len(history.train_losses))

    return history
