"""Train the value+policy network from saved self-play data and
register it as a new versioned model in the registry.

Loads every ``*.npz`` in ``--data-dir``, fits the ``_ValuePolicyMLP``
(or value-only ``_ValueMLP`` if no policy supervision in the data),
registers as ``v{N+1}`` under ``--registry-root`` (default
``trained_weights/``) with a metadata sidecar capturing training
hyperparameters, loss history, and the data sources used. Uses CUDA
when available.

Usage::

    python -m mtg_bot.cli train --data-dir training_data/ --epochs 100

    # Custom registry location
    python -m mtg_bot.cli train --data-dir training_data/ \\
        --registry-root experiments/run3/weights --epochs 50

The version ID is auto-assigned (v1, v2, ...). The trained network
loads via ``NNEvaluator(weights_path=registry.weights_path("v3"))``
or via the eval / capture / iterate subcommands by version label.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from mtg_bot.features import FEATURE_DIM
from mtg_bot.search.nn_evaluator import (
    ARCHITECTURE_SHARED,
    HIDDEN_SIZES,
    SUPPORTED_ARCHITECTURES,
)
from mtg_bot.training import (
    auto_device, load_all_datasets, train_value_network,
)
from mtg_bot.training.registry import ModelMetadata, Registry


def _downsample_one_legal_rows(
    features: np.ndarray,
    outcomes: np.ndarray,
    policy_targets: np.ndarray,
    action_masks: np.ndarray,
    *,
    keep_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Drop a fraction of one-legal-action supervised rows.

    Why: when ~84% of supervised rows have exactly one legal action,
    their CE loss is trivially 0 and they dominate the policy-head
    gradient with zero learning signal. Downsampling them lets the
    sparse multi-legal rows (where MCTS visit-count distribution
    actually carries signal) drive the gradient.

    A row is "one-legal-supervised" iff
    ``action_mask.sum(-1) == 1`` AND ``policy_target.sum(-1) > 0.5``.
    Value-only rows (no policy supervision) are kept fully — they
    contribute nothing to the policy gradient anyway.

    Returns the downsampled (features, outcomes, policy_targets,
    action_masks) along with (n_one_legal_total, n_one_legal_kept)
    for logging.
    """
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(
            f"keep_frac must be in (0.0, 1.0]; got {keep_frac}"
        )

    has_policy = policy_targets.sum(axis=-1) > 0.5
    n_legal = action_masks.sum(axis=-1)
    one_legal_supervised = has_policy & (n_legal == 1)
    n_one_legal_total = int(one_legal_supervised.sum())

    if keep_frac >= 1.0 or n_one_legal_total == 0:
        return (
            features,
            outcomes,
            policy_targets,
            action_masks,
            n_one_legal_total,
            n_one_legal_total,
        )

    rng = np.random.default_rng(seed)
    # Build keep-mask: every non-one-legal-supervised row stays;
    # one-legal-supervised rows pass with probability ``keep_frac``.
    keep_mask = np.ones(features.shape[0], dtype=bool)
    one_legal_idx = np.flatnonzero(one_legal_supervised)
    n_keep = int(round(n_one_legal_total * keep_frac))
    if n_keep < n_one_legal_total:
        # rng.permutation is reproducible given the seed.
        shuffled = rng.permutation(one_legal_idx)
        drop = shuffled[n_keep:]
        keep_mask[drop] = False

    return (
        features[keep_mask],
        outcomes[keep_mask],
        policy_targets[keep_mask],
        action_masks[keep_mask],
        n_one_legal_total,
        n_keep,
    )


def _git_commit() -> str | None:
    """Best-effort current git commit short SHA; None if not a git
    repo or git isn't on PATH."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def main(argv=None) -> None:  # pragma: no cover
    # Argparse + training-driver body; previously script-level code
    # under ``scripts/train_nn.py``. The ``_downsample_one_legal_rows``
    # helper is unit-tested directly; this body is mostly argparse
    # plumbing + a single ``train_value_network(...)`` dispatch.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory of *.npz training files (data_gen output).",
    )
    parser.add_argument(
        "--registry-root", type=Path,
        default=Path("trained_weights"),
        help=(
            "Registry root directory. Trained weights register as "
            "v{N}/ subdirectories under here. Default trained_weights/."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs (default 30).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size (default 256).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate (default 1e-3).",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of data held out for validation (default 0.1).",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda/cpu (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for split + init (default 0).",
    )
    parser.add_argument(
        "--notes", default="",
        help="Optional free-form note saved into metadata.json.",
    )
    parser.add_argument(
        "--progress-path", type=Path, default=None,
        help=(
            "Where to write live per-epoch progress JSON for the "
            "training UI. Defaults to "
            "<cwd>/training_progress/<version>.json where <version> "
            "is the next registry version (peeked, not reserved). "
            "Pass an explicit path to override; pass an empty string "
            "to disable progress emission entirely."
        ),
    )
    parser.add_argument(
        "--compile", dest="compile_model", action="store_true",
        help=(
            "Wrap the model in ``torch.compile`` (PyTorch 2.x) — "
            "fuses kernels and removes per-op interpreter overhead. "
            "Pays a 10-30 s compile cost on the first batch; net "
            "win over 100 epochs is typically 1.5-3x. Off by "
            "default (compile path can be flaky on edge-case ops); "
            "recommended on for any real GPU training run."
        ),
    )
    parser.add_argument(
        "--hidden-sizes", default=None,
        help=(
            "Comma-separated trunk shape (e.g. '1024,1024,512') "
            "overriding the registry-baseline (128, 128, 64). "
            "Saved into the version's metadata.json so loading via "
            "--version (in evaluate_nn.py / iterate_alphazero.py) "
            "rebuilds an evaluator at the matching shape. Useful "
            "for capacity-experiment training runs without making "
            "old versions unloadable."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=list(SUPPORTED_ARCHITECTURES),
        default=ARCHITECTURE_SHARED,
        help=(
            "Dual-head model class: 'shared' (default, back-compat "
            "with v1-v15) uses a single trunk feeding both value and "
            "policy heads. 'separate' instantiates parallel value_"
            "trunk + policy_trunk MLPs, eliminating the shared-"
            "bottleneck competition between heads (the value head's "
            "louder gradient saturates a shared trunk's representation "
            "and crowds the policy head's branchy-row signal out — see "
            "iter_002 v13 training analysis). Switching from 'shared' "
            "to 'separate' (or vice versa) invalidates the existing "
            "v-chain — the new architecture must train fresh. The "
            "choice is recorded in metadata.json so loaders can "
            "dispatch."
        ),
    )
    parser.add_argument(
        "--policy-hidden-sizes", default=None,
        help=(
            "Optional comma-separated trunk shape for the policy_trunk "
            "when --architecture=separate. Defaults to mirror "
            "--hidden-sizes. Useful for asymmetric capacity allocation "
            "(e.g. trim the policy trunk if VRAM matters)."
        ),
    )
    parser.add_argument(
        "--policy-loss-weight", type=float, default=1.0,
        help=(
            "Coefficient on the policy-head CE in the combined loss "
            "(total = value_loss + policy_loss_weight * policy_loss). "
            "Default 1.0 lets value_loss (~0.5 at epoch 1) dominate "
            "policy_loss (~0.16 at epoch 1) ~3:1 — the policy head "
            "plateaus at the floor set by ~84%% one-legal-action rows "
            "(observed in iter_002 v13 training). Bump to 3.0-10.0 "
            "to give the policy head enough gradient on the ~2%% of "
            "branchy rows where it actually has signal to learn."
        ),
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help=(
            "Stop training after the val loss has failed to improve "
            "for this many consecutive epochs. Default None = disabled "
            "(run all --epochs). Typical iter runs hit best val loss "
            "around epoch 25-30 of a 50-epoch budget; the remaining "
            "~20 epochs are mostly overfitting. Set to 5 to ride "
            "through plateau noise but still cut the wasted tail. "
            "Must be a positive int if set."
        ),
    )
    parser.add_argument(
        "--window-recent", type=int, default=None,
        help=(
            "If set, load only the N most-recent (by mtime) .npz "
            "files from --data-dir instead of the entire directory. "
            "Default None = load every file. As iterate_data/ "
            "accumulates over many iterations, training on the "
            "entire history slows linearly with size and biases "
            "early data; capping to the latest N files keeps "
            "training fast and weighted toward fresh self-play. "
            "Files greater than the available count loads them "
            "all (no error)."
        ),
    )
    parser.add_argument(
        "--one-legal-keep-frac", type=float, default=1.0,
        help=(
            "Fraction of one-legal-action supervised rows to keep "
            "before training (default 1.0 = no downsample). When "
            "<1.0, only this fraction of rows whose action_mask has "
            "exactly one legal slot AND that carry policy supervision "
            "are retained; multi-legal supervised rows and value-only "
            "rows are kept fully. The companion fix to "
            "--policy-loss-weight: rather than reweighting the loss, "
            "rebalances the dataset so the policy head's gradient is "
            "dominated by branchy rows with real visit-count signal. "
            "At 0.1, the ~84%% one-legal share drops to ~8.4%% so "
            "branchy supervised rows become the majority. Seeded "
            "from --seed for reproducibility. Must be in (0.0, 1.0]."
        ),
    )
    parser.add_argument(
        "--action-encoding-version",
        choices=["positional_v1", "canonical_v2"],
        default="positional_v1",
        help=(
            "Action-encoding slot mapping convention. "
            "'positional_v1' (default, back-compat with v13/v15/v16) "
            "uses raw card.id-positional slots. 'canonical_v2' sorts "
            "hand/graveyard/exile by (printing.name, modifier_tuple, "
            "card.id) before slot assignment so two canonically-"
            "equivalent states share their slot mappings. The choice "
            "is recorded in metadata.json so loaders thread it through "
            "to NNEvaluator at inference. Switching versions is a "
            "v-chain break — old weights cannot be reused."
        ),
    )
    args = parser.parse_args(argv)
    if not (0.0 < args.one_legal_keep_frac <= 1.0):
        parser.error(
            "--one-legal-keep-frac must be in (0.0, 1.0]; got "
            f"{args.one_legal_keep_frac}"
        )
    if args.early_stop_patience is not None and args.early_stop_patience < 1:
        parser.error(
            "--early-stop-patience must be a positive int (>= 1); got "
            f"{args.early_stop_patience}"
        )
    if args.window_recent is not None and args.window_recent <= 0:
        parser.error(
            "--window-recent must be a positive integer; got "
            f"{args.window_recent}"
        )
    if args.hidden_sizes is not None:
        try:
            hidden_sizes = tuple(
                int(x.strip()) for x in args.hidden_sizes.split(",")
            )
        except ValueError as exc:
            sys.exit(f"--hidden-sizes parse error: {exc}")
        if not hidden_sizes:
            sys.exit("--hidden-sizes must list at least one layer")
    else:
        hidden_sizes = HIDDEN_SIZES

    policy_hidden_sizes: tuple[int, ...] | None = None
    if args.policy_hidden_sizes is not None:
        if args.architecture != "separate":
            sys.exit(
                "--policy-hidden-sizes only applies when "
                "--architecture=separate"
            )
        try:
            policy_hidden_sizes = tuple(
                int(x.strip()) for x in args.policy_hidden_sizes.split(",")
            )
        except ValueError as exc:
            sys.exit(f"--policy-hidden-sizes parse error: {exc}")
        if not policy_hidden_sizes:
            sys.exit(
                "--policy-hidden-sizes must list at least one layer"
            )

    if not args.data_dir.is_dir():
        sys.exit(f"--data-dir {args.data_dir} is not a directory")

    print(f"# Loading data from {args.data_dir}")
    if args.window_recent is not None:
        print(
            f"# Window-recent: loading at most {args.window_recent} "
            f"most-recent files by mtime"
        )
    # Always pull policy supervision arrays. ``include_policy=True``
    # zero-pads value-only files so the concatenated arrays align —
    # mixed dirs (some pre-A2.3 .npz, some post-A2.3) load cleanly.
    features, outcomes, policy_targets, action_masks = (
        load_all_datasets(
            args.data_dir,
            include_policy=True,
            max_files=args.window_recent,
        )
    )
    if features.shape[0] == 0:
        sys.exit(
            f"No .npz files found in {args.data_dir}. Generate data "
            f"with scripts/generate_training_data.py first."
        )
    # Reflect the actually-loaded subset in metadata (not every
    # file in the dir) when --window-recent is in play; otherwise
    # this is identical to the all-files glob.
    all_npz = sorted(args.data_dir.glob("*.npz"))
    if args.window_recent is not None and len(all_npz) > args.window_recent:
        loaded_paths = sorted(
            all_npz, key=lambda p: p.stat().st_mtime, reverse=True,
        )[:args.window_recent]
        data_files = sorted(p.name for p in loaded_paths)
    else:
        data_files = [p.name for p in all_npz]

    if args.one_legal_keep_frac < 1.0:
        n_before = features.shape[0]
        (
            features,
            outcomes,
            policy_targets,
            action_masks,
            n_one_legal_total,
            n_one_legal_kept,
        ) = _downsample_one_legal_rows(
            features, outcomes, policy_targets, action_masks,
            keep_frac=args.one_legal_keep_frac,
            seed=args.seed,
        )
        print(
            f"[train_nn] downsampled one-legal rows: kept "
            f"{n_one_legal_kept} of {n_one_legal_total} "
            f"(kept_frac={args.one_legal_keep_frac}); "
            f"final dataset: {features.shape[0]} rows "
            f"(was {n_before})",
            file=sys.stderr,
        )

    n_policy_rows = int(
        (policy_targets.sum(axis=-1) > 0.5).sum()
    )
    has_any_policy = n_policy_rows > 0
    print(
        f"# {features.shape[0]} samples from {len(data_files)} files, "
        f"FEATURE_DIM={features.shape[1]}"
    )
    print(
        f"# Policy supervision: {n_policy_rows} / "
        f"{features.shape[0]} rows "
        f"({n_policy_rows / features.shape[0]:.1%}). Mode: "
        f"{'AlphaZero (dual-head)' if has_any_policy else 'value-only'}"
    )

    device = args.device or auto_device()
    print(f"# Training on {device} for {args.epochs} epochs")

    # Resolve the progress-file path BEFORE training so the UI can
    # see the run from epoch 0. We peek at the next version ID
    # rather than reserve it — the registry assigns sequentially at
    # register time, so as long as nobody else trains concurrently
    # the peek value matches the eventual assignment. If a race
    # happens, the progress filename simply won't match the
    # registered version; nothing breaks (the UI just sees an extra
    # file with the original name).
    registry = Registry(args.registry_root)
    peeked_version = registry.next_version_id()
    if args.progress_path is None:
        progress_dir = Path("training_progress")
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_path: Path | None = progress_dir / f"{peeked_version}.json"
    elif str(args.progress_path) == "":
        progress_path = None
    else:
        progress_path = args.progress_path
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path is not None:
        print(f"# Progress JSON: {progress_path}")

    # Train into a temp file, then hand off to the registry which
    # copies the bytes into the versioned directory.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_weights = Path(tmp) / "weights.pt"
        started = time.perf_counter()
        # Dispatch on schema: if any row carries policy supervision,
        # forward the policy_targets / action_masks arrays so the
        # trainer trains the dual-head ``_ValuePolicyMLP``. Otherwise
        # train the value-only ``_ValueMLP`` (existing behavior; old
        # v1/v2 weights still load via ``NNEvaluator()`` default).
        train_kwargs: dict = {
            "features": features,
            "outcomes": outcomes,
            "weights_out_path": tmp_weights,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "val_split": args.val_split,
            "device": device,
            "rng_seed": args.seed,
            "progress_path": progress_path,
            "run_id": peeked_version,
            "notes": args.notes,
            "compile_model": args.compile_model,
            "hidden_sizes": hidden_sizes,
            "early_stop_patience": args.early_stop_patience,
        }
        if has_any_policy:
            train_kwargs["policy_targets"] = policy_targets
            train_kwargs["action_masks"] = action_masks
            train_kwargs["policy_loss_weight"] = args.policy_loss_weight
            train_kwargs["architecture"] = args.architecture
            if policy_hidden_sizes is not None:
                train_kwargs["policy_hidden_sizes"] = policy_hidden_sizes
        history = train_value_network(**train_kwargs)
        elapsed = time.perf_counter() - started

        metadata = ModelMetadata(
            version="",  # filled in by Registry.register
            created_at="",  # filled in by Registry.register
            feature_dim=FEATURE_DIM,
            hidden_sizes=list(hidden_sizes),
            with_policy=has_any_policy,
            # Architecture is meaningful only for dual-head training
            # (``has_any_policy=True``); for value-only runs we still
            # default to "shared" so the field is present and stable.
            architecture=(
                args.architecture if has_any_policy
                else ARCHITECTURE_SHARED
            ),
            action_encoding_version=args.action_encoding_version,
            training={
                "data_dir": str(args.data_dir),
                "data_files": data_files,
                "n_samples": int(features.shape[0]),
                "n_policy_rows": n_policy_rows,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "val_split": args.val_split,
                "device": device,
                "rng_seed": args.seed,
                "duration_seconds": elapsed,
                "one_legal_keep_frac": args.one_legal_keep_frac,
                "window_recent": args.window_recent,
                "policy_hidden_sizes": (
                    list(policy_hidden_sizes)
                    if policy_hidden_sizes is not None else None
                ),
                "early_stop_patience": args.early_stop_patience,
                "epochs_trained": len(history.train_losses),
            },
            training_history={
                "train_losses": history.train_losses,
                "val_losses": history.val_losses,
                "train_value_losses": history.train_value_losses,
                "val_value_losses": history.val_value_losses,
                "train_policy_losses": history.train_policy_losses,
                "val_policy_losses": history.val_policy_losses,
                "final_train_loss": history.train_losses[-1],
                "final_val_loss": history.val_losses[-1],
                "best_val_loss": history.best_val_loss,
                "best_val_epoch": history.best_val_epoch,
            },
            git_commit=_git_commit(),
            notes=args.notes,
        )
        version = registry.register(tmp_weights, metadata)

    print(f"\n# Done in {elapsed:.1f}s. Registered as {version}.")
    print(f"# Weights:  {registry.weights_path(version)}")
    print(f"# Final train loss: {history.train_losses[-1]:.4f}")
    print(f"# Final val loss:   {history.val_losses[-1]:.4f}")
    if has_any_policy:
        print(
            f"#   value:  train={history.train_value_losses[-1]:.4f} "
            f"val={history.val_value_losses[-1]:.4f}"
        )
        print(
            f"#   policy: train={history.train_policy_losses[-1]:.4f} "
            f"val={history.val_policy_losses[-1]:.4f}"
        )
    print(
        f"# Best val loss: {history.best_val_loss:.4f} "
        f"at epoch {history.best_val_epoch}"
    )


if __name__ == "__main__":
    main()
