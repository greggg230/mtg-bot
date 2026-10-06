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

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

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
from mtg_bot.training.block_shuffle import DEFAULT_BLOCK_ROWS
from mtg_bot.training.progress import atomic_write_json, git_short_sha

_log = logging.getLogger(__name__)


def grouped_train_val_split(
    group_ids: np.ndarray,
    val_split: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into ``(train_idx, val_idx)`` by GROUP.

    ``group_ids`` is a ``(N,)`` int array; rows sharing a value came
    from the same self-play game. Groups (not rows) are shuffled and
    ``~val_split`` of them are assigned to val, so no group ever
    straddles the split. Splitting by row instead leaks near-
    duplicate consecutive game states into val, and val loss stops
    measuring generalisation and starts measuring memorisation.

    Rules:

    - ``val_split <= 0`` → everything is train (val empty).
    - ``n_val_groups = int(n_groups * val_split)``; when that rounds
      to 0 but there are at least 2 groups, one group still goes to
      val so early-stopping keeps a signal (e.g. a handful of legacy
      one-group-per-file NPZs at ``val_split=0.1``). A single group
      can't be split — val is empty and a warning is logged.
    - Row order within each side is shuffled with ``rng`` so the
      caller can slice mini-batches directly. Deterministic in
      ``rng``.
    """
    group_ids = np.asarray(group_ids)
    n = int(group_ids.shape[0])
    if group_ids.ndim != 1:
        raise ValueError(
            f"group_ids must be 1-D; got shape {group_ids.shape}"
        )
    if val_split <= 0.0 or n == 0:
        return rng.permutation(n), np.zeros((0,), dtype=np.int64)
    uniq = np.unique(group_ids)
    n_groups = int(uniq.shape[0])
    n_val_groups = int(n_groups * val_split)
    if n_val_groups == 0 and n_groups > 1:
        n_val_groups = 1
    if n_groups == 1:
        _log.warning(
            "grouped_train_val_split: only ONE group in %d rows; "
            "cannot hold out a game-level validation set (val empty). "
            "Provide game_ids (newer NPZ files) or more files.",
            n,
        )
    shuffled_groups = uniq[rng.permutation(n_groups)]
    val_groups = shuffled_groups[:n_val_groups]
    is_val = np.isin(group_ids, val_groups)
    val_idx = np.flatnonzero(is_val)
    train_idx = np.flatnonzero(~is_val)
    val_idx = val_idx[rng.permutation(val_idx.shape[0])]
    train_idx = train_idx[rng.permutation(train_idx.shape[0])]
    _log.info(
        "train/val split by group: %d groups -> %d train groups / "
        "%d val groups; %d train rows / %d val rows",
        n_groups, n_groups - n_val_groups, n_val_groups,
        int(train_idx.shape[0]), int(val_idx.shape[0]),
    )
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


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

    ``restored_best_epoch`` is the 0-based epoch whose weights were
    actually SAVED. It is ``best_val_epoch`` when restore-best fired
    and ``None`` when the final epoch's weights were kept (either
    ``--keep-last-epoch``, or no val data to pick a best from).

    ``holdout_value_metrics`` carries the post-training value-head
    quality measured on the held-out split (see
    :mod:`mtg_bot.training.value_gate`). ``None`` for the flat-feature
    trainer, which does not compute it.
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
    restored_best_epoch: Optional[int] = None
    holdout_value_metrics: Optional[Any] = None


def auto_device() -> str:
    """Return the best available torch device string for this host.

    Order of preference: ``"cuda"`` > ``"cpu"``. CUDA wins on machines
    with NVIDIA GPUs (master / Win10 remote); CPU is the universal
    fallback — including on Apple Silicon, see the MPS measurement
    below.

    NOTE: these probes (``torch.cuda.is_available``,
    ``torch.backends.mps.is_available``, ``torch.backends.mps.is_built``)
    MUST stay inside the function body. Calling them at module-import
    time primes the CUDA driver in the parent and breaks the Linux
    fork+mp.Pool path used by the production master (see PR #36 +
    the deferred ``device='auto'`` sentinel in ``NNPimcConfig.build``).

    **Why MPS does NOT win on Mac (measured 2026-05-11, v18 weights,
    1024-1024-512 trunk + 1+517 heads, batch_evaluate_with_policy):**

    +---------+--------------+--------------+--------------+
    | Batch   | MPS ms/state | CPU ms/state | MPS verdict  |
    +=========+==============+==============+==============+
    | 1       | 0.94         | 0.07         | 15x slower   |
    | 8       | 0.106        | 0.041        | 2.6x slower  |
    | 32      | 0.033        | 0.017        | 1.9x slower  |
    | 128     | 0.017        | 0.015        | 1.1x slower  |
    +---------+--------------+--------------+--------------+

    Apple Silicon's CPU (M-series with AMX matrix unit) is genuinely
    fast at small dense matmul. At our NN size (51 input dim, ~830k
    params) the MPS GPU-dispatch overhead never amortizes; even at
    batch=128 MPS still loses. Correctness is fine — values match
    CPU within 6e-8 and policies are bit-identical — but enabling
    MPS in production would slow the Mac shard ~2x at the MCTS
    batch=32 working size.

    If the NN grows substantially (e.g. >2048-trunk) or Apple ships
    further MPS perf improvements, revisit this measurement. Until
    then the MPS branch is intentionally absent; we keep the
    ``mps.is_available`` import-time guards out of this function so
    nothing primes the MPS driver speculatively either.
    """
    if torch.cuda.is_available():
        return "cuda"
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
    epoch_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    group_ids: Optional[np.ndarray] = None,
) -> TrainingHistory:
    """Fit a value (or value+policy) network to the supplied
    training data and write the final state_dict to
    ``weights_out_path``.

    ``group_ids`` (``(N,)`` int, optional) are per-row game group ids
    (``load_all_datasets(..., include_group_ids=True)``). The
    train/val split is done by GROUP via
    :func:`grouped_train_val_split` so no game straddles the split.
    ``None`` falls back to one group per row — i.e. a plain row
    split — and logs a warning; only appropriate for synthetic data
    with no game structure.

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

    ``epoch_callback`` (default ``None``) is invoked once at the end
    of every epoch with a dict snapshot of per-epoch metrics:
    ``{epoch, total_epochs, elapsed_seconds, train_loss, val_loss,
    train_value_loss, val_value_loss, train_policy_loss,
    val_policy_loss, status}``. Used by ``mtg_bot.cli.train`` to push
    progress to the runs DB alongside the existing JSON write — the
    callback fires after the trainer's own ``_emit`` so the in-memory
    history is already updated. Any exception inside the callback is
    swallowed so a flaky telemetry sink can't break training.

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

    # Train/val split BY GAME GROUP (deterministic in ``rng_seed``).
    n = features.shape[0]
    rng = np.random.default_rng(rng_seed)
    if group_ids is None:
        _log.warning(
            "train_value_network: no group_ids supplied; falling back "
            "to a per-row train/val split (rows from one game may land "
            "on both sides). Pass group_ids from "
            "load_all_datasets(include_group_ids=True)."
        )
        group_ids = np.arange(n, dtype=np.int64)
    elif np.asarray(group_ids).shape[0] != n:
        raise ValueError(
            f"group_ids rows {np.asarray(group_ids).shape[0]} != "
            f"features rows {n}"
        )
    train_idx, val_idx = grouped_train_val_split(
        group_ids, val_split, rng,
    )
    n_val = int(val_idx.shape[0])

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
            if epoch_callback is not None:
                # Snapshot per-epoch metrics for the optional external
                # sink (runs DB push lives in cli/train.py). Swallow any
                # exception so a flaky telemetry path can't kill an
                # otherwise-healthy training run.
                try:
                    epoch_callback({
                        "epoch": epoch + 1,
                        "total_epochs": int(effective_total_epochs),
                        "elapsed_seconds": (
                            time.perf_counter() - started_perf
                            if progress_enabled else 0.0
                        ),
                        "train_loss": history.train_losses[-1],
                        "val_loss": history.val_losses[-1],
                        "train_value_loss": (
                            history.train_value_losses[-1]
                        ),
                        "val_value_loss": history.val_value_losses[-1],
                        "train_policy_loss": (
                            history.train_policy_losses[-1]
                        ),
                        "val_policy_loss": (
                            history.val_policy_losses[-1]
                        ),
                        # Full per-epoch history. The UI is runs.db-only
                        # post-Phase 3 (no JSON-side backfill); the
                        # callback is the sole path that lands the
                        # arrays into the DB extra_json so the UI's
                        # loss chart has data points.
                        "train_losses": list(history.train_losses),
                        "val_losses": list(history.val_losses),
                        "train_value_losses": list(
                            history.train_value_losses,
                        ),
                        "val_value_losses": list(
                            history.val_value_losses,
                        ),
                        "train_policy_losses": list(
                            history.train_policy_losses,
                        ),
                        "val_policy_losses": list(
                            history.val_policy_losses,
                        ),
                        "status": "running",
                    })
                except Exception:  # noqa: BLE001
                    pass

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


def _structured_policy_loss(
    policy_logits: torch.Tensor,
    target_dist: torch.Tensor,
    action_mask: torch.Tensor,
    has_supervision: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy for the StructuredModel policy head.

    Identical shape contract to the inner ``_policy_loss`` defined in
    ``train_value_network`` — kept module-level so it can be reused
    across the structured + MLP paths if a future slice consolidates
    them. The two implementations are character-for-character the
    same; the duplication is intentional to avoid surfacing a private
    helper across function bodies that already share enough state.

    ``policy_logits``     ``(B, action_vocab_size)``
    ``target_dist``       ``(B, action_vocab_size)`` — non-negative,
                          rows sum to 1 where supervised
    ``action_mask``       ``(B, action_vocab_size)`` — True = legal
    ``has_supervision``   ``(B,)`` — float 0/1
    """
    target_dist = target_dist * action_mask.float()
    masked_logits = policy_logits.masked_fill(~action_mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    per_row_ce = -(target_dist * log_probs).sum(dim=-1)
    weighted = per_row_ce * has_supervision
    denom = has_supervision.sum().clamp(min=1.0)
    return weighted.sum() / denom


def _parse_value_target(spec: str) -> tuple[str, float]:
    """``"outcome"`` | ``"root"`` | ``"blend:<w>"`` -> ``(mode, w)``.

    - ``outcome``: value head regresses the +/-1 game result (the
      pre-2026-09-26 behaviour).
    - ``root``: regresses the search root value recorded in the shard
      (``root_values``; viewer perspective). Rows whose root value is
      NaN (legacy shards, forced moves) fall back to the outcome.
    - ``blend:<w>``: ``w * root + (1 - w) * outcome`` with the same
      NaN fallback; ``w`` in ``[0, 1]``.
    """
    spec = (spec or "outcome").strip().lower()
    if spec == "outcome":
        return "outcome", 0.0
    if spec == "root":
        return "root", 1.0
    if spec.startswith("blend:"):
        try:
            w = float(spec.split(":", 1)[1])
        except ValueError as e:
            raise ValueError(f"bad --value-target {spec!r}: {e}") from e
        if not (0.0 <= w <= 1.0):
            raise ValueError(
                f"--value-target blend weight must be in [0, 1]; got {w}"
            )
        return "blend", w
    raise ValueError(
        f"unknown value_target {spec!r}; expected 'outcome', 'root' "
        f"or 'blend:<w>'"
    )


def _value_target_tensor(
    batch: dict[str, torch.Tensor], mode: str, w: float,
) -> torch.Tensor:
    """Resolve the per-row regression target for the value head."""
    outcome = batch["outcome"]
    if mode == "outcome":
        return outcome
    rv = batch.get("root_value")
    if rv is None:
        return outcome
    rv = rv.to(outcome.dtype)
    rv = torch.where(torch.isfinite(rv), rv, outcome)
    if mode == "root":
        return rv
    return w * rv + (1.0 - w) * outcome


def train_structured_model(
    dataset,
    *,
    weights_out_path: Path,
    config: Optional[Any] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.1,
    device: Optional[str] = None,
    rng_seed: int = 0,
    progress_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    notes: str = "",
    policy_loss_weight: float = 1.0,
    hand_belief_loss_weight: float = 0.05,
    epoch_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    autocast_dtype: Any = "auto",
    compile_model: Optional[bool] = None,
    early_stop_patience: Optional[int] = None,
    num_workers: int = 0,
    loader_timeout_sec: Optional[float] = 600.0,
    weight_decay: float = 0.0,
    restore_best: bool = True,
    shuffle_mode: str = "auto",
    shuffle_block_rows: int = DEFAULT_BLOCK_ROWS,
    value_target: str = "outcome",
) -> TrainingHistory:
    """Train a :class:`StructuredModel` over a
    :class:`StructuredDataset`.

    Regularisation knobs (added 2026-09-24 after every s2-lineage
    version memorised game identity — see
    :mod:`mtg_bot.training.value_gate`):

    - ``weight_decay``: decoupled L2 through ``torch.optim.AdamW``.
      ``0.0`` (the signature default, for back-compat with existing
      callers/tests) makes AdamW numerically identical to the Adam
      this used to construct; the training CLI defaults it to 1e-4.
      Dropout is set on the model side via
      ``StructuredModelConfig.dropout``.
    - ``restore_best``: when True (default) and a val split exists,
      the state_dict from ``history.best_val_epoch`` is reloaded
      before the weights are written. Previously the FINAL epoch's
      weights were always saved, so early stopping tracked a best
      epoch it then threw away — every extra epoch past the optimum
      shipped straight into the registry. Pass False (the CLI's
      ``--keep-last-epoch``) for the old behaviour.

    After training, the held-out split is re-scored and
    ``history.holdout_value_metrics`` is populated with a
    :class:`~mtg_bot.training.value_gate.HoldoutValueMetrics` (MSE,
    corr(pred, outcome), the constant-0 predictor's MSE on the same
    rows, and the life-differential baseline correlation when the
    scalar schema carries life). The CALLER decides what to do with
    it — this function never refuses to save.

    GPU fast path knobs (CUDA-only, no-op elsewhere):

    - ``autocast_dtype``: ``"auto"`` (default) enables bf16 autocast
      on CUDA / disables on CPU/MPS. Pass ``torch.float16`` /
      ``torch.bfloat16`` to force a specific precision, or
      ``False`` / ``None`` to disable. bf16 is preferred for
      transformers (no GradScaler needed; wider dynamic range than
      fp16). fp16 requires a GradScaler to avoid underflow during
      gradient computation.
    - ``compile_model``: ``None`` (default) enables
      ``torch.compile(mode='reduce-overhead')`` on CUDA / disables
      elsewhere. Pass ``True`` / ``False`` to override. First epoch
      pays a one-time compile cost (~seconds); subsequent epochs
      amortize.

    Sibling of :func:`train_value_network` for the Step 1 NN-
    representation rewrite. The two functions intentionally share the
    progress-emit shape (so the existing training UI continues to
    show per-epoch loss series without modification) and the
    :class:`TrainingHistory` return type. They diverge on:

    - Model: :class:`StructuredModel` consuming a dict batch of
      per-object tokens rather than ``_ValuePolicyMLP`` on a flat
      51-dim feature vector.
    - Data loading: :class:`torch.utils.data.DataLoader` with the
      :func:`structured_collate` collate_fn for per-batch padding,
      rather than the upfront-tensors+index-shuffle path the
      MLP trainer uses (variable-length input requires per-batch
      padding).
    - Loss: value MSE + ``policy_loss_weight`` × policy CE +
      ``hand_belief_loss_weight`` × hand-belief BCE. The hand-belief
      term is currently unused (the current dataset does not include
      hand-belief targets — Step 4 will), but the head's gradients
      stay wired so checkpoint formats are stable when Step 4 turns
      it on. With no targets, the term is zero.

    Saves the final state_dict to ``weights_out_path``. Loading-side
    dispatch (``NNEvaluator``) lands separately; the saved weights
    are bytes-compatible with
    ``StructuredModel(config).load_state_dict(...)``.

    ``epoch_callback`` mirrors :func:`train_value_network` — fires
    after the in-memory history is updated, exceptions are swallowed.

    ``shuffle_mode`` picks how the train split is ordered:

    - ``"row"`` — the historical ``DataLoader(Subset, shuffle=True)``:
      a uniform shuffle over every train row. Each batch is
      ``batch_size`` independent random slices into the flat store's
      memmaps.
    - ``"block"`` — :class:`~mtg_bot.training.block_shuffle.BlockShuffleDataset`:
      the same rows per epoch, but read in shuffled contiguous blocks
      of ``shuffle_block_rows`` rows so the store is streamed
      sequentially instead of page-faulted at random. Also sorts the
      val split's indices, which are otherwise a random permutation
      read back with ``shuffle=False`` — the same random-I/O problem
      for 10% of the rows, every epoch.
    - ``"auto"`` (default) — ``block`` once the flat store is larger
      than half of available RAM (i.e. once the page cache can no
      longer hold it), ``row`` below that. See
      :func:`~mtg_bot.training.block_shuffle.resolve_shuffle_mode`.

    Block mode changes the ORDER rows are visited in, never the SET:
    one epoch is still exactly one pass over every train row.

    ``loader_timeout_sec`` (default 600 s) is the DataLoader's
    per-batch collection timeout. It only applies when
    ``num_workers > 0`` — torch asserts ``timeout == 0`` for the
    single-process iterator, so a non-zero value there would raise on
    the first batch. With worker processes and NO timeout, a wedged
    worker (observed 2026-09-24 on the 4M-row cycle-5 train) hangs the
    training run forever with no error and no progress; the timeout
    turns that into a loud ``RuntimeError: DataLoader timed out``.
    Pass ``None`` (or 0) to restore the block-forever behaviour.
    """
    # Lazy imports to avoid a hard dep on the structured-model module
    # at import time of the broader training package (the structured
    # trainer is a narrow slice; older callers shouldn't pay its
    # import cost).
    from torch.utils.data import DataLoader, Subset

    from mtg_bot.training.block_shuffle import (
        BlockShuffleDataset, resolve_shuffle_mode,
    )
    from mtg_bot.training.structured_dataset import (
        shutdown_dataloader_workers, structured_collate,
    )
    from mtg_bot.training.structured_model import (
        StructuredModel, StructuredModelConfig,
    )

    if len(dataset) == 0:
        raise ValueError("cannot train on an empty dataset")
    if not (0.0 <= val_split < 1.0):
        raise ValueError(
            f"val_split must be in [0, 1); got {val_split}"
        )
    if early_stop_patience is not None and early_stop_patience < 1:
        raise ValueError(
            f"early_stop_patience must be >= 1 (or None to disable); "
            f"got {early_stop_patience}"
        )
    if shuffle_block_rows < 1:
        raise ValueError(
            f"shuffle_block_rows must be >= 1; got {shuffle_block_rows}"
        )
    # Resolved BEFORE anything expensive so an invalid mode (or
    # 'block' on a dataset with no flat store) fails on the argument
    # rather than 20 minutes into epoch 1.
    resolved_shuffle = resolve_shuffle_mode(shuffle_mode, dataset)
    if config is None:
        config = StructuredModelConfig()
    # Scalar-schema width gate. The dataset's ``token_scalars`` row
    # width is fixed at data-gen time (``tokenize_state(...,
    # scalar_schema_version=...)``); the model's ``scalar_proj`` width
    # is fixed by ``config.scalar_schema_version``. A mismatch would
    # surface as a cryptic matmul shape error on the first batch (or,
    # worse, silently train on the wrong slot semantics if a future
    # schema kept the width) — refuse loudly up front instead.
    dataset_width = getattr(dataset, "scalar_width", None)
    if dataset_width is not None and dataset_width != config.scalar_width:
        raise ValueError(
            f"scalar schema mismatch: the dataset's token_scalars rows "
            f"are {dataset_width} wide but StructuredModelConfig"
            f"(scalar_schema_version={config.scalar_schema_version}) "
            f"expects {config.scalar_width}. Either regenerate the "
            f"training data with data_gen's "
            f"structured_scalar_schema_version={config.scalar_schema_version}"
            f", or train with a config whose scalar_schema_version "
            f"matches the data (``mtg_bot train --scalar-schema-version N``)."
        )

    # Policy-width gate: the dataset's policy columns must be the
    # model's slot count (517 for positional_v1 / canonical_v2, 805 for
    # canonical_v3). A mismatch would silently mis-align every slot (or
    # crash deep inside the loss); ``load_structured_dataset`` already
    # padded legacy shards explicitly when asked to.
    dataset_policy_width = getattr(dataset, "policy_width", None)
    if (
        dataset_policy_width is not None
        and dataset_policy_width != config.action_vocab_size
    ):
        raise ValueError(
            f"policy width mismatch: the dataset's policy_targets / "
            f"action_masks rows are {dataset_policy_width} wide but the "
            f"model's policy head has {config.action_vocab_size} slots. "
            f"Load the window with the model's action_encoding_version "
            f"(load_structured_dataset(action_encoding_version=...)) or "
            f"train with the matching --action-encoding-version."
        )

    if device is None:
        device = auto_device()
    if device == "cpu":
        torch.set_num_threads(1)
    device_obj = torch.device(device)
    torch.manual_seed(rng_seed)

    # Deterministic split BY GAME GROUP. ``StructuredDataset.group_ids``
    # carries one id per row (``load_structured_dataset`` derives it
    # from each file's ``game_ids``, or the whole file for legacy
    # NPZs); a dataset without the attribute falls back to per-row
    # groups (= plain row split) with a warning.
    n = len(dataset)
    rng = np.random.default_rng(rng_seed)
    group_ids = getattr(dataset, "group_ids", None)
    if group_ids is None:
        _log.warning(
            "train_structured_model: dataset exposes no group_ids; "
            "falling back to a per-row train/val split."
        )
        group_ids = np.arange(n, dtype=np.int64)
    train_idx_arr, val_idx_arr = grouped_train_val_split(
        np.asarray(group_ids), val_split, rng,
    )
    n_val = int(val_idx_arr.shape[0])
    val_idx = val_idx_arr.tolist()
    train_idx = train_idx_arr.tolist()
    train_subset = Subset(dataset, train_idx) if train_idx else dataset
    if resolved_shuffle == "block":
        # Sorting the val indices costs nothing and removes the other
        # half of the random-I/O problem: ``grouped_train_val_split``
        # returns a PERMUTATION, and the val loader reads it back with
        # ``shuffle=False``, so every eval pass was seeking too. The
        # val ROW SET is unchanged, and a row's model output doesn't
        # depend on which rows share its batch, so the holdout metrics
        # are the same.
        val_idx = sorted(val_idx)
    # ``num_workers=0`` keeps DataLoader on the main thread — fine for
    # smoke tests + tiny datasets but starves the GPU on real iter
    # data (7M+ samples ran at ~9% GPU util with single-process
    # loading, ~22 min/epoch instead of the expected ~30s). Production
    # callers pass ``num_workers >= 1`` to spawn worker processes that
    # prefetch batches; ``persistent_workers=True`` keeps them alive
    # across epochs so the dataset's flat-storage isn't re-pickled
    # every epoch.
    _pw = num_workers > 0
    # ``timeout`` is only legal with worker processes: torch's
    # ``_SingleProcessDataLoaderIter`` asserts ``timeout == 0``. With
    # workers it is the stall guard — without it a wedged worker hangs
    # the whole run silently (seen 2026-09-24).
    _timeout = (
        float(loader_timeout_sec)
        if (_pw and loader_timeout_sec) else 0
    )
    if resolved_shuffle == "block":
        # ``shuffle=True`` is illegal on an IterableDataset — the
        # wrapper IS the shuffle. Workers shard by block, so each one
        # gets a disjoint slice of the epoch with no sampler involved.
        train_loader = DataLoader(
            BlockShuffleDataset(
                train_subset,
                block_rows=shuffle_block_rows,
                seed=rng_seed,
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=structured_collate,
            num_workers=num_workers,
            persistent_workers=_pw,
            pin_memory=device_obj.type == "cuda",
            timeout=_timeout,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=structured_collate,
            num_workers=num_workers,
            persistent_workers=_pw,
            pin_memory=device_obj.type == "cuda",
            timeout=_timeout,
        )
    val_loader: Optional[DataLoader] = None
    if n_val > 0:
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=structured_collate,
            num_workers=num_workers,
            persistent_workers=_pw,
            pin_memory=device_obj.type == "cuda",
            timeout=_timeout,
        )

    model = StructuredModel(config).to(device_obj)

    # ----- GPU fast path: autocast + torch.compile ------------------
    # Resolve autocast dtype. ``"auto"`` -> bf16 on CUDA / off
    # elsewhere. Explicit ``False`` / ``None`` disables. Mirrors
    # ``SharedTensorNNEvalServer``'s knob shape so train + inference
    # use a consistent precision policy.
    is_cuda = device_obj.type == "cuda"
    resolved_autocast_dtype = autocast_dtype
    if resolved_autocast_dtype == "auto":
        resolved_autocast_dtype = torch.bfloat16 if is_cuda else None
    elif resolved_autocast_dtype is False:
        resolved_autocast_dtype = None
    autocast_enabled = is_cuda and resolved_autocast_dtype is not None
    # GradScaler is only needed for fp16; bf16's wider dynamic range
    # makes it unnecessary. Build one when the resolved dtype is
    # fp16 specifically.
    use_grad_scaler = (
        autocast_enabled and resolved_autocast_dtype == torch.float16
    )
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None
    # Compile gate: default True on CUDA, False elsewhere.
    if compile_model is None:
        compile_model = is_cuda
    uncompiled_model = model
    if compile_model and is_cuda:
        try:
            model = torch.compile(
                model, mode="reduce-overhead", dynamic=False,
            )
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"torch.compile failed ({exc}); falling back to "
                f"eager execution.",
                RuntimeWarning, stacklevel=2,
            )
            model = uncompiled_model

    # AdamW rather than Adam: decoupled weight decay is the correct
    # L2 formulation for adaptive optimizers, and at weight_decay=0
    # AdamW's update is identical to Adam's, so existing callers see
    # no numerical change.
    if weight_decay < 0.0:
        raise ValueError(
            f"weight_decay must be >= 0; got {weight_decay}"
        )
    optimizer = optim.AdamW(
        uncompiled_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    mse = nn.MSELoss()

    import contextlib

    def _autocast_ctx():
        if autocast_enabled:
            return torch.autocast(
                device_type="cuda", dtype=resolved_autocast_dtype,
            )
        return contextlib.nullcontext()

    history = TrainingHistory()
    # CPU snapshot of the best-val-loss epoch's weights. Stays None
    # when restore_best is off or no epoch ever produced a finite val
    # loss (e.g. val_split=0).
    best_state: Optional[dict[str, torch.Tensor]] = None
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
            "architecture": "structured",
            # Populated by the post-training held-out scoring pass, so
            # it's None for every mid-run emit and a metrics dict on
            # the terminal "completed" write. The training UI's
            # progress card renders it; the CLI's gate reads the same
            # numbers off the returned history.
            "holdout_value_metrics": (
                history.holdout_value_metrics.as_dict()
                if history.holdout_value_metrics is not None else None
            ),
            "restored_best_epoch": history.restored_best_epoch,
        }
        atomic_write_json(progress_path, payload)

    _emit("running", current_epoch=0)

    _vt_mode, _vt_w = _parse_value_target(value_target)

    def _run_one(
        batch: dict[str, torch.Tensor], *, train: bool,
    ) -> tuple[float, float, float]:
        """Run forward + (optional) backward on one batch. Returns
        ``(total_loss, value_loss, policy_loss)`` as Python floats.

        Forward pass is wrapped in ``torch.autocast`` when the
        precision policy enables it; the backward + optimizer step
        run in fp32 (autocast scope ends after the forward), with
        a ``GradScaler`` injected for the fp16 case to keep
        gradients from underflowing.
        """
        # Move dict batch to the training device. ``non_blocking=True``
        # is a no-op for CPU tensors and a small win when the
        # DataLoader pins memory; setting it unconditionally costs
        # nothing.
        on_device = {
            k: v.to(device_obj, non_blocking=True)
            for k, v in batch.items()
        }
        with _autocast_ctx():
            out = model(on_device)
            value_pred = out["value"]
            policy_logits = out["policy_logits"]
            v_loss = mse(
                value_pred, _value_target_tensor(on_device, _vt_mode, _vt_w),
            )
            has_pol = on_device["has_policy"].float()
            # If no row in this batch has supervision, the masked-CE
            # denominator is 1 (clamped) and per-row weights are zero,
            # so the loss collapses to zero. Keeping the call avoids a
            # branch and matches the MLP path's behavior.
            p_loss = _structured_policy_loss(
                policy_logits,
                on_device["policy_target"],
                on_device["action_mask"],
                has_pol,
            )
            # Hand-belief term: the dataset doesn't carry hand-belief
            # targets yet (Step 4 wires those), so this term contributes
            # zero. We still touch the head's parameters during the
            # forward pass when ``enable_hand_belief_head=True`` so the
            # autograd graph is stable; with the head disabled (default),
            # the contribution is identically zero.
            total = v_loss + policy_loss_weight * p_loss
        if train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                optimizer.step()
        return total.item(), v_loss.item(), p_loss.item()

    try:
        for epoch in range(epochs):
            model.train()
            train_total: list[float] = []
            train_value: list[float] = []
            train_policy: list[float] = []
            for batch in train_loader:
                t, v, p = _run_one(batch, train=True)
                train_total.append(t)
                train_value.append(v)
                train_policy.append(p)
            history.train_losses.append(
                float(np.mean(train_total)) if train_total else float("nan")
            )
            history.train_value_losses.append(
                float(np.mean(train_value)) if train_value
                else float("nan")
            )
            history.train_policy_losses.append(
                float(np.mean(train_policy)) if train_policy
                else float("nan")
            )

            if val_loader is not None:
                model.eval()
                val_total: list[float] = []
                val_value: list[float] = []
                val_policy: list[float] = []
                with torch.no_grad():
                    for batch in val_loader:
                        t, v, p = _run_one(batch, train=False)
                        val_total.append(t)
                        val_value.append(v)
                        val_policy.append(p)
                history.val_losses.append(
                    float(np.mean(val_total)) if val_total
                    else float("nan")
                )
                history.val_value_losses.append(
                    float(np.mean(val_value)) if val_value
                    else float("nan")
                )
                history.val_policy_losses.append(
                    float(np.mean(val_policy)) if val_policy
                    else float("nan")
                )
            else:
                history.val_losses.append(float("nan"))
                history.val_value_losses.append(float("nan"))
                history.val_policy_losses.append(float("nan"))

            if (
                history.val_losses[-1] == history.val_losses[-1]
                and history.val_losses[-1] < history.best_val_loss
            ):
                history.best_val_loss = history.val_losses[-1]
                history.best_val_epoch = epoch
                if restore_best:
                    # Snapshot on CPU so the checkpoint doesn't pin a
                    # second copy of the model in VRAM for the rest of
                    # the run. ``.clone()`` is required — ``.to("cpu")``
                    # on an already-CPU tensor is a no-op view and
                    # would alias the live parameter.
                    best_state = {
                        k: v.detach().to("cpu", copy=True)
                        for k, v in
                        uncompiled_model.state_dict().items()
                    }

            _emit("running", current_epoch=epoch + 1)
            if epoch_callback is not None:
                try:
                    epoch_callback({
                        "epoch": epoch + 1,
                        "total_epochs": int(effective_total_epochs),
                        "elapsed_seconds": (
                            time.perf_counter() - started_perf
                            if progress_enabled else 0.0
                        ),
                        "train_loss": history.train_losses[-1],
                        "val_loss": history.val_losses[-1],
                        "train_value_loss": (
                            history.train_value_losses[-1]
                        ),
                        "val_value_loss": history.val_value_losses[-1],
                        "train_policy_loss": (
                            history.train_policy_losses[-1]
                        ),
                        "val_policy_loss": (
                            history.val_policy_losses[-1]
                        ),
                        # Full per-epoch history. The UI is runs.db-only
                        # post-Phase 3 (no JSON-side backfill); the
                        # callback is the sole path that lands the
                        # arrays into the DB extra_json so the UI's
                        # loss chart has data points.
                        "train_losses": list(history.train_losses),
                        "val_losses": list(history.val_losses),
                        "train_value_losses": list(
                            history.train_value_losses,
                        ),
                        "val_value_losses": list(
                            history.val_value_losses,
                        ),
                        "train_policy_losses": list(
                            history.train_policy_losses,
                        ),
                        "val_policy_losses": list(
                            history.val_policy_losses,
                        ),
                        "status": "running",
                    })
                except Exception:  # noqa: BLE001
                    pass

            # Patience-based early stopping. Mirrors the
            # ``train_value_network`` semantics: only meaningful when we
            # have val data; with no val rows ``best_val_epoch`` stays
            # -1 and val_loss is NaN, so nothing to compare. Without
            # this, the structured CLI's --early-stop-patience flag is
            # captured into metadata but never honored — the lever
            # SCALING_DIAGNOSTICS.md D3 recommends.
            if (
                early_stop_patience is not None
                and n_val > 0
                and history.best_val_epoch >= 0
                and (epoch - history.best_val_epoch) >= early_stop_patience
            ):
                effective_total_epochs = epoch + 1
                print(
                    f"[structured-trainer] early-stop fired at epoch "
                    f"{epoch + 1} (best was epoch "
                    f"{history.best_val_epoch + 1}, "
                    f"patience={early_stop_patience})",
                    file=sys.stderr,
                )
                _emit("running", current_epoch=epoch + 1)
                break
    except BaseException:
        try:
            _emit("failed", current_epoch=len(history.train_losses))
        except Exception:
            pass
        raise
    finally:
        # ``persistent_workers=True`` keeps the DataLoader workers (and
        # their memmap handles on the dataset's flat store) alive until
        # the loader is garbage-collected. On Windows those handles
        # block deletion of the store directory — the leak that filled
        # %TEMP% with 20-50 GB per run. Shut them down explicitly the
        # moment the epoch loop is done.
        for _loader in (train_loader, val_loader):
            if _loader is not None:
                shutdown_dataloader_workers(_loader)

    # ----- restore-best ---------------------------------------------
    # Load the best-val-loss epoch's weights back into the live model
    # BEFORE scoring the held-out split and BEFORE saving. Without
    # this, early stopping identified a best epoch and then shipped
    # the (worse, more-overfit) final epoch's weights — patience
    # epochs' worth of pure memorisation straight into the registry.
    if restore_best and best_state is not None:
        uncompiled_model.load_state_dict(best_state)
        history.restored_best_epoch = history.best_val_epoch
        _log.info(
            "restore-best: loaded epoch %d weights (val loss %.4f) "
            "over the final epoch %d",
            history.best_val_epoch + 1, history.best_val_loss,
            len(history.train_losses),
        )
    elif restore_best:
        _log.info(
            "restore-best requested but no best-epoch snapshot exists "
            "(no finite val loss — e.g. val_split=0); keeping the "
            "final epoch's weights.",
        )

    # ----- held-out value quality ------------------------------------
    # Scored on the SAME rows early stopping used, with the weights
    # that are about to be written. The caller (mtg_bot.cli.train)
    # turns this into a promote/reject decision; the trainer only
    # measures.
    history.holdout_value_metrics = _score_holdout_value(
        model=model,
        val_loader=val_loader,
        device_obj=device_obj,
        autocast_ctx=_autocast_ctx,
        scalar_schema_version=getattr(
            config, "scalar_schema_version", None,
        ),
    )

    # When ``torch.compile`` wraps the model, save the underlying
    # module's state_dict so the on-disk format stays compatible with
    # ``NNEvaluator``'s loader (no ``_orig_mod.`` prefix).
    save_target = getattr(model, "_orig_mod", uncompiled_model)
    save_target.to("cpu")
    torch.save(save_target.state_dict(), weights_out_path)
    history.weights_path = weights_out_path

    _emit("completed", current_epoch=len(history.train_losses))

    return history


def _score_holdout_value(
    *,
    model,
    val_loader,
    device_obj,
    autocast_ctx,
    scalar_schema_version: Optional[int] = None,
):
    """Re-score the held-out split and return
    :class:`~mtg_bot.training.value_gate.HoldoutValueMetrics`.

    Collects the model's PRE-tanh ``value_logit`` alongside the
    squashed ``value`` so
    :func:`~mtg_bot.training.value_gate.compute_holdout_value_metrics`
    can fit the post-hoc temperature
    (:mod:`mtg_bot.training.value_calibration`) that the caller then
    writes into ``metadata.value_calibration``.

    Returns a zero-row metrics object when there is no val loader
    (``val_split=0`` or a single-game dataset) — the gate reads that
    as "could not evaluate" rather than as a failure.

    The life-differential baseline needs the viewer/opponent life
    scalars, which only exist in scalar schema v2+. Under schema v1
    the baseline correlation comes back ``None`` (documented as
    "skipped" rather than silently zero).
    """
    from mtg_bot.training.structured_model import (
        SCALAR_FIELD_OPP_LIFE,
        SCALAR_FIELD_VIEWER_LIFE,
        SCALAR_SCHEMA_V2,
        TOKEN_TYPE_GLOBAL,
    )
    from mtg_bot.training.value_gate import (
        compute_holdout_value_metrics,
    )

    if val_loader is None:
        return compute_holdout_value_metrics(
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    want_life = (
        scalar_schema_version is not None
        and int(scalar_schema_version) >= SCALAR_SCHEMA_V2
    )
    preds: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lifes: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in val_loader:
                on_device = {
                    k: v.to(device_obj, non_blocking=True)
                    for k, v in batch.items()
                }
                with autocast_ctx():
                    out = model(on_device)
                preds.append(
                    out["value"].detach().float().cpu().numpy()
                )
                # Pre-tanh logits feed the post-hoc temperature fit.
                # ``.get`` rather than ``[...]`` so a stand-in model
                # in a test (or a future head that squashes
                # internally) degrades to the post-tanh shrink fit
                # instead of raising.
                logit_t = out.get("value_logit")
                if logit_t is not None:
                    logits.append(
                        logit_t.detach().float().cpu().numpy()
                    )
                targets.append(
                    on_device["outcome"].detach().float().cpu().numpy()
                )
                if want_life:
                    lifes.append(
                        _global_life_diff(
                            batch["token_type"],
                            batch["token_scalars"],
                            batch["attention_mask"],
                            global_type=TOKEN_TYPE_GLOBAL,
                            viewer_idx=SCALAR_FIELD_VIEWER_LIFE,
                            opp_idx=SCALAR_FIELD_OPP_LIFE,
                        )
                    )
    finally:
        if was_training:
            model.train()
    life_arr = np.concatenate(lifes) if lifes else None
    logit_arr = (
        np.concatenate(logits)
        if logits and len(logits) == len(preds) else None
    )
    return compute_holdout_value_metrics(
        np.concatenate(preds) if preds
        else np.zeros((0,), dtype=np.float32),
        np.concatenate(targets) if targets
        else np.zeros((0,), dtype=np.float32),
        life_diffs=life_arr,
        logits=logit_arr,
    )


def _global_life_diff(
    token_type: torch.Tensor,
    token_scalars: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    global_type: int,
    viewer_idx: int,
    opp_idx: int,
) -> np.ndarray:
    """Per-row ``viewer_life - opponent_life`` from a collated batch.

    Life totals live on the row's GLOBAL token (schema v2). Rows with
    no global token — shouldn't happen for tokenizer output, but a
    hand-built test batch can lack one — get 0.0, which is the
    neutral value for the baseline correlation.
    """
    is_global = (token_type == global_type) & attention_mask.bool()
    # First global token per row; rows with none select index 0 and
    # are zeroed by the ``has_any`` mask below.
    idx = torch.argmax(is_global.to(torch.int8), dim=1)
    has_any = is_global.any(dim=1)
    rows = torch.arange(token_scalars.shape[0])
    viewer = token_scalars[rows, idx, viewer_idx]
    opp = token_scalars[rows, idx, opp_idx]
    diff = torch.where(
        has_any, viewer - opp, torch.zeros_like(viewer),
    )
    return diff.detach().float().cpu().numpy()
