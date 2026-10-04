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
import re
import subprocess
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from mtg_bot import run_labels as _run_labels
from mtg_bot.cli.runs import client as _runs_client
from mtg_bot.cli.runs import lifecycle as _lifecycle
from mtg_bot.features import FEATURE_DIM
from mtg_bot.search.nn_evaluator import (
    ARCHITECTURE_SHARED,
    ARCHITECTURE_STRUCTURED,
    HIDDEN_SIZES,
    SUPPORTED_ARCHITECTURES,
)
from mtg_bot.training import (
    auto_device,
    load_all_datasets,
    load_structured_dataset,
    train_structured_model,
    train_value_network,
)
from mtg_bot.training.block_shuffle import DEFAULT_BLOCK_ROWS
from mtg_bot.training.card_vocab import (
    SHARD_CARD_VOCAB_ENV,
    CardVocabError,
    legacy_shard_vocab_from_arg,
    resolve_shards_card_vocab,
)
from mtg_bot.training.structured_model import (
    SUPPORTED_VALUE_HEAD_INPUTS,
)
from mtg_bot.training.registry import ModelMetadata, Registry
from mtg_bot.training.value_gate import (
    DEFAULT_MIN_HOLDOUT_VALUE_QUALITY,
    evaluate_value_gate,
    format_gate_failure,
)


_GATE_REJECT_EXIT_CODE = 3
"""Exit code for "training finished but the held-out value gate
rejected it". Distinct from 1 (generic failure / argparse error) so
``mtg_bot iterate`` can tell a rejected model apart from a crashed
trainer and keep the chain running. See
:mod:`mtg_bot.training.value_gate`."""

DEFAULT_STRUCTURED_DROPOUT = 0.1
"""``--dropout`` default for the structured trainer.

0.1 rather than a heavier 0.3: the production trunk is SMALL
(emb_dim 128 / 2 layers / hidden 256), and the memorisation route is
the printing-embedding path rather than raw trunk capacity, so a
light regulariser plus restore-best plus weight decay is the
proportionate first move. It is also the value the transformer
literature uses by default at this scale. Raise it if the gate keeps
rejecting."""

DEFAULT_STRUCTURED_WEIGHT_DECAY = 1e-4
"""``--weight-decay`` default. Standard AdamW starting point; enough
to keep the printing embeddings from growing unboundedly into
per-game lookup keys without visibly slowing the fit at 1e-3 LR."""


class _ValueGateRejected(RuntimeError):
    """Raised inside the run-lifecycle block when the held-out value
    gate rejects a freshly-trained model, so the run row is marked
    ``failed`` before the CLI exits ``_GATE_REJECT_EXIT_CODE``.
    Carries the already-formatted operator message."""


def _merge_into_progress_json(
    progress_path: Optional[Path], extra: dict,
) -> None:
    """Merge ``extra`` into an existing progress JSON in place.

    Best-effort: a missing path (progress disabled) or an unreadable /
    malformed file is a no-op. The trainer owns the file during the
    run; this runs after it has written its terminal payload, so a
    read-modify-write is safe.
    """
    if progress_path is None:
        return
    import json

    from mtg_bot.training.progress import atomic_write_json

    try:
        blob = json.loads(Path(progress_path).read_text())
        if not isinstance(blob, dict):
            return
    except (OSError, ValueError):
        return
    blob.update(extra)
    try:
        atomic_write_json(Path(progress_path), blob)
    except OSError:
        pass


# Regex matching the iterate cycle run_id format. When ``mtg_bot
# iterate`` invokes train via subprocess it threads the cycle_id
# into ``--notes`` so the resulting training run can be linked back
# to its parent iterate cycle row.
#
# Lineage-prefix tolerant: ``mtg_bot iterate --run-label-prefix s2_``
# emits ``s2_iter_001_cycle``. A leading ``\b`` would NOT match there
# (``_`` is a word character, so there is no word boundary between
# ``s2_`` and ``iter``) and the parent link would silently drop. The
# optional ``<token>_`` groups absorb the prefix; the lookbehind keeps
# the match anchored at a token start so ``xiter_001_cycle`` still
# doesn't match.
_ITER_CYCLE_RE = re.compile(
    r"(?<![\w.-])((?:[A-Za-z0-9][A-Za-z0-9-]*_)*iter_\d+_cycle)\b"
)


def _now_stamp() -> str:
    """Compact UTC ISO-8601 timestamp suitable for embedding in
    run_ids / filenames (no colons, no fractional seconds, trailing
    ``Z``). Matches the format ``mtg_bot iterate`` uses for its own
    launch IDs (:func:`mtg_bot.cli.iterate._now_stamp`)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_train_run_id(version_label: str, stamp: str) -> str:
    """Compose the standalone-train run_id from a registry version
    label + a UTC stamp.

    The leading ``train_`` namespace plus the trailing timestamp keeps
    each ``mtg_bot train`` invocation unique even when the same
    registry version label is recycled (fresh checkout reusing
    ``vN``, manual recovery train after a failed attempt, ...). See
    ``feature/unique-train-run-id`` for the bug this resolves.
    """
    return f"train_{version_label}_{stamp}"


def _parent_run_id_from_notes(notes: str) -> Optional[str]:
    """Best-effort extract an iterate cycle_id from a ``--notes``
    string. Returns ``None`` when nothing matches — standalone
    ``mtg_bot train`` invocations have no parent."""
    if not notes:
        return None
    match = _ITER_CYCLE_RE.search(notes)
    return match.group(1) if match else None


def _downsample_one_legal_rows(
    features: np.ndarray,
    outcomes: np.ndarray,
    policy_targets: np.ndarray,
    action_masks: np.ndarray,
    *,
    keep_frac: float,
    seed: int,
    group_ids: Optional[np.ndarray] = None,
) -> tuple:
    """Drop a fraction of one-legal-action supervised rows.

    ``group_ids`` (optional, per-row game group ids) is subset with
    the same keep-mask and returned as a SEVENTH element when
    supplied; without it the classic 6-tuple is returned unchanged.

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
        out = (
            features,
            outcomes,
            policy_targets,
            action_masks,
            n_one_legal_total,
            n_one_legal_total,
        )
        return out + (group_ids,) if group_ids is not None else out

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

    out = (
        features[keep_mask],
        outcomes[keep_mask],
        policy_targets[keep_mask],
        action_masks[keep_mask],
        n_one_legal_total,
        n_keep,
    )
    if group_ids is not None:
        out = out + (np.asarray(group_ids)[keep_mask],)
    return out


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


class WindowedDataDir:
    """Duck-typed stand-in for the ``--data-dir`` ``Path`` that exposes
    only a pre-selected subset of the directory's ``*.npz`` files.

    Both dataset loaders (``data_gen.load_all_datasets`` and
    ``structured_dataset.load_structured_dataset``) enumerate their
    inputs with ``directory.glob("*.npz")`` and otherwise only use the
    directory object for error messages. Wrapping it here lets the CLI
    apply an ITERATION-level training window (see
    :func:`select_window_files`) without teaching either loader about
    run labels and without a second copy of the npz concatenation
    logic.

    Every attribute other than the enumeration methods delegates to the
    real ``Path``, so ``str()`` / ``os.fspath()`` / ``is_dir()`` behave
    exactly as before. ``glob`` and ``iterdir`` flip
    :attr:`enumerated`; :meth:`check_enumerated` asserts it afterwards
    so a loader that ever stops enumerating through the ``Path`` API
    fails LOUDLY instead of silently training on the whole directory.
    """

    def __init__(self, directory: Path, paths) -> None:
        self._directory = Path(directory)
        self._paths = [Path(p) for p in paths]
        self.enumerated = False

    # -- enumeration surface the loaders use ------------------------
    def glob(self, pattern: str):
        import fnmatch

        self.enumerated = True
        return iter([
            p for p in self._paths
            if fnmatch.fnmatch(p.name, pattern)
        ])

    def iterdir(self):
        self.enumerated = True
        return iter(list(self._paths))

    # -- Path pass-through -----------------------------------------
    def __getattr__(self, name):
        return getattr(self._directory, name)

    def __truediv__(self, other):
        return self._directory / other

    def __fspath__(self) -> str:
        return str(self._directory)

    def __str__(self) -> str:
        return str(self._directory)

    def __repr__(self) -> str:
        return (
            f"WindowedDataDir({self._directory!r}, "
            f"{len(self._paths)} files)"
        )

    @property
    def paths(self) -> "list[Path]":
        return list(self._paths)

    def check_enumerated(self) -> None:
        """Raise when the loader never asked us for the file list.

        A silently-ignored window would train on the ENTIRE data dir —
        exactly the failure this change exists to remove — so treat it
        as a hard error rather than a warning.
        """
        if not self.enumerated:
            raise RuntimeError(
                "training window was not applied: the dataset loader "
                "did not enumerate {} via .glob()/.iterdir(), so the "
                "--window-recent* selection would have been silently "
                "ignored. Teach the loader an explicit file-list "
                "argument and pass the window through it.".format(
                    self._directory
                )
            )


def _iter_group_key(path: Path) -> "tuple[str, int] | None":
    """``(lineage_prefix, iteration_index)`` parsed out of an npz
    filename, or ``None`` when the name carries no iteration label."""
    parsed = _run_labels.parse_iter_label(path.name)
    if parsed is None:
        return None
    prefix, idx, _rest = parsed
    return prefix, idx


def select_window_files(
    data_dir: Path,
    *,
    window_iters: "int | None",
    window_files: "int | None",
    log=print,
) -> "list[Path] | None":
    """Resolve the training window to a concrete list of npz paths.

    Returns ``None`` when no window is configured (load the whole
    directory — the historical default).

    ``window_files`` reproduces the pre-2026-09-24 behaviour: the N
    most recently modified files.

    ``window_iters`` is the iteration-aware window: files are grouped
    by the ``<prefix>iter_<NNN>`` label parsed out of their filename
    (:func:`mtg_bot.run_labels.parse_iter_label`) and the N highest
    iteration indices are kept IN FULL. This matters because the two
    data pipelines write wildly different file counts per iteration —
    the local loop writes ONE npz per iteration
    (``s2_iter_001_v1_<stamp>.npz``) while the distributed loop
    rehydrates ONE npz per GAME
    (``s2_iter_003_self_play_v1_<host>_..._000437.npz``). Under the old
    file-count semantics ``--window-recent 3`` trained a distributed
    cycle on THREE GAMES (observed 2026-09-23, 7,278 rows).

    Grouping is per lineage prefix: a directory holding both
    ``iter_00N_*`` and ``s2_iter_00N_*`` keeps each lineage's last N
    iterations rather than letting the lineage with the higher indices
    evict the other entirely. Mixed prefixes are warned about — a
    single data dir should belong to a single lineage.

    Files whose name carries no iteration label (hand-copied data,
    ad-hoc ``generate_training_data.py`` output, ...) are ALWAYS kept
    and listed in a loud warning: dropping unlabelled data silently
    would be the same class of bug this function fixes.
    """
    all_paths = sorted(Path(data_dir).glob("*.npz"))
    if window_files is not None:
        if not all_paths or len(all_paths) <= window_files:
            return list(all_paths)
        kept = sorted(
            sorted(all_paths, key=lambda p: p.stat().st_mtime,
                   reverse=True)[:window_files]
        )
        log(
            f"# Window-recent-files: keeping {len(kept)} of "
            f"{len(all_paths)} .npz files (most recent by mtime)"
        )
        return kept
    if window_iters is None:
        return None
    if not all_paths:
        return []

    groups: dict[tuple[str, int], list[Path]] = {}
    unlabelled: list[Path] = []
    for path in all_paths:
        key = _iter_group_key(path)
        if key is None:
            unlabelled.append(path)
        else:
            groups.setdefault(key, []).append(path)

    prefixes = sorted({prefix for prefix, _idx in groups})
    if len(prefixes) > 1:
        log(
            "# WARNING: --window-recent found MULTIPLE lineage "
            "prefixes in {}: {}. Keeping the last {} iterations of "
            "EACH lineage — a data dir should normally hold one "
            "lineage.".format(
                data_dir,
                ", ".join(repr(p) for p in prefixes),
                window_iters,
            )
        )
    kept_keys: set[tuple[str, int]] = set()
    for prefix in prefixes:
        indices = sorted(
            (idx for pfx, idx in groups if pfx == prefix), reverse=True,
        )
        for idx in indices[:window_iters]:
            kept_keys.add((prefix, idx))

    kept = [p for key in kept_keys for p in groups[key]]
    if unlabelled:
        log(
            "# WARNING: --window-recent: {} .npz file(s) in {} carry "
            "no <prefix>iter_NNN label and were KEPT unconditionally "
            "(they cannot be placed in an iteration window): {}".format(
                len(unlabelled), data_dir,
                ", ".join(p.name for p in sorted(unlabelled)),
            )
        )
        kept.extend(unlabelled)
    kept = sorted(kept)
    log(
        f"# Window-recent: keeping {len(kept)} of {len(all_paths)} "
        f".npz files from the last {window_iters} iteration(s) "
        f"({len(kept_keys)} iteration group(s) kept of "
        f"{len(groups)} found)"
    )
    return kept


def _resolve_data_window(args, log=print):
    """Return ``(load_dir, data_files)`` for the configured window.

    ``load_dir`` is either ``args.data_dir`` itself (no window) or a
    :class:`WindowedDataDir` exposing only the windowed subset;
    ``data_files`` is the basename list to record in metadata.
    """
    window_paths = select_window_files(
        args.data_dir,
        window_iters=args.window_recent,
        window_files=args.window_recent_files,
        log=log,
    )
    if window_paths is None:
        return args.data_dir, sorted(
            p.name for p in args.data_dir.glob("*.npz")
        )
    return (
        WindowedDataDir(args.data_dir, window_paths),
        sorted(p.name for p in window_paths),
    )


def _validate_window_args(args, parser) -> None:
    """Shared ``--window-recent*`` validation for both trainer paths."""
    if args.window_recent is not None and args.window_recent <= 0:
        parser.error(
            "--window-recent must be a positive integer; got "
            f"{args.window_recent}"
        )
    if args.window_recent_files is not None and args.window_recent_files <= 0:
        parser.error(
            "--window-recent-files must be a positive integer; got "
            f"{args.window_recent_files}"
        )
    if args.window_recent is not None and args.window_recent_files is not None:
        parser.error(
            "--window-recent (iterations) and --window-recent-files "
            "(raw file count) are mutually exclusive; pass one."
        )


def main(argv=None) -> None:
    # Argparse + training-driver body; previously script-level code
    # under ``scripts/train_nn.py``. The ``_downsample_one_legal_rows``
    # helper is unit-tested directly; this body is mostly argparse
    # plumbing + a single ``train_value_network(...)`` dispatch. The
    # lifecycle wiring (start_run / push_progress / finish_run) is
    # pinned by ``tests/cli/test_train_runs_db.py``.
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
            "<cwd>/training_progress/train_<version>_<stamp>.json — "
            "<version> is the next registry version (peeked, not "
            "reserved) and <stamp> is a UTC compact-ISO start "
            "timestamp (YYYYMMDDThhmmssZ) so two standalone trains "
            "of the same version land in separate files. Pass an "
            "explicit path to override; pass an empty string to "
            "disable progress emission entirely. (Iterate threads its "
            "own iter-namespaced --progress-path through, so this "
            "default only governs standalone callers.)"
        ),
    )
    parser.add_argument(
        "--progress-dir", type=Path, default=Path("training_progress"),
        help=(
            "Directory the default --progress-path is built under "
            "(default training_progress/). ``mtg_bot iterate`` "
            "forwards its own --progress-dir here (which the unified "
            "launcher in turn forwards from the launch TOML) so a "
            "second lineage launched with e.g. "
            "--progress-dir training_progress_s2 keeps its training "
            "cards out of the first lineage's directory. Ignored when "
            "an explicit --progress-path is given."
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
        "--early-stop-patience", type=int, default=3,
        help=(
            "Stop training after the val loss has failed to improve "
            "for this many consecutive epochs. Default 3 — the D3 "
            "epochs-ablation (docs/SCALING_DIAGNOSTICS.md) showed v5 "
            "was undertrained at 5 epochs / patience=None; re-training "
            "with epochs=20 + patience=3 cleared the eval gate against "
            "v5 (Wilson-LB 0.546). To run every epoch unconditionally, "
            "pass a patience >= --epochs. Must be a positive int."
        ),
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help=(
            "DataLoader worker process count. Default 0 (main thread) "
            "is fine for smoke tests but starves the GPU on real iter "
            "data (~9%% GPU util at 7M+ samples). Production callers "
            "should pass >= 4 so DataLoader prefetches batches in "
            "parallel; persistent_workers=True is enabled automatically "
            "when num_workers > 0 so the dataset is pickled once at "
            "epoch 0 instead of every epoch."
        ),
    )
    parser.add_argument(
        "--loader-timeout-sec", type=float, default=600.0,
        help=(
            "Per-batch DataLoader collection timeout in seconds "
            "(default 600). Only applies when --num-workers > 0 — "
            "torch asserts timeout==0 for the single-process loader. "
            "With worker processes and no timeout, a wedged worker "
            "hangs training FOREVER with no error and no progress "
            "(observed 2026-09-24 on the 4M-row cycle-5 train); the "
            "timeout turns that stall into a loud RuntimeError. Pass "
            "0 to block forever (the pre-2026-09-24 behaviour)."
        ),
    )
    parser.add_argument(
        "--shuffle-mode", choices=("auto", "row", "block"),
        default="auto",
        help=(
            "Structured-trainer train-split ordering. 'row' is the "
            "historical uniform shuffle: every batch is batch_size "
            "random slices into the flat store's memmaps, which is "
            "thousands of 4 KB page faults per batch once the store "
            "no longer fits the page cache (measured 2026-09-25: "
            "~60 GB store on a 64 GB box ran the disk at 111%% busy / "
            "~6,800 reads/s and left the GPU at 10-20%%). 'block' "
            "reads shuffled contiguous blocks of --shuffle-block-rows "
            "rows, one sequential slice per array, and shuffles "
            "within a 2-block window; same rows per epoch, different "
            "order. 'auto' (default) picks block once the store "
            "exceeds half of available RAM."
        ),
    )
    parser.add_argument(
        "--shuffle-block-rows", type=int, default=DEFAULT_BLOCK_ROWS,
        help=(
            "Rows per contiguous block under --shuffle-mode block "
            "(default %(default)s). Bigger = more sequential I/O and "
            "better within-block mixing, at ~4 KB of worker RAM per "
            "row per block in the 2-block window. Ignored under "
            "--shuffle-mode row."
        ),
    )
    parser.add_argument(
        "--window-recent", type=int, default=None,
        help=(
            "If set, train only on the last N ITERATIONS of data in "
            "--data-dir instead of the entire directory. Files are "
            "grouped by the <prefix>iter_NNN label in their filename "
            "and every file of the N highest iteration indices is "
            "kept (per lineage prefix). Default None = load every "
            "file. NOTE this counts ITERATIONS, not files: the "
            "distributed pipeline rehydrates one .npz per GAME, so "
            "the old file-count semantics made --window-recent 3 "
            "train on three games. Files whose name carries no "
            "iteration label are always kept (with a warning). N "
            "larger than the available iteration count loads them "
            "all (no error). For the raw file-count behaviour use "
            "--window-recent-files."
        ),
    )
    parser.add_argument(
        "--window-recent-files", type=int, default=None,
        help=(
            "If set, load only the N most-recent (by mtime) .npz "
            "files from --data-dir. This is the pre-2026-09-24 "
            "--window-recent behaviour, kept for callers that want a "
            "hard cap on file count (e.g. bounding RAM on the "
            "one-npz-per-game distributed path). Mutually exclusive "
            "with --window-recent. N greater than the available file "
            "count loads them all (no error)."
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
    # Structured-architecture (Step 1) hyperparameters. Only consumed
    # when ``--architecture structured``; the flat-feature path
    # ignores them. Defaults mirror
    # :class:`StructuredModelConfig`'s defaults so the structured
    # trunk built here matches the skeleton called out in the
    # roadmap.
    parser.add_argument(
        "--emb-dim", type=int, default=None,
        help=(
            "Structured-trunk: per-token embedding dimension. "
            "Default None defers to StructuredModelConfig (128)."
        ),
    )
    parser.add_argument(
        "--n-heads", type=int, default=None,
        help=(
            "Structured-trunk: transformer attention heads. "
            "Default None defers to StructuredModelConfig (4)."
        ),
    )
    parser.add_argument(
        "--n-layers", type=int, default=None,
        help=(
            "Structured-trunk: encoder layer count. "
            "Default None defers to StructuredModelConfig (2)."
        ),
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help=(
            "Structured-trunk: MLP hidden dim inside the encoder. "
            "Default None defers to StructuredModelConfig (256)."
        ),
    )
    parser.add_argument(
        "--action-history-n", type=int, default=None,
        help=(
            "Structured-trunk: last-N action-history window. "
            "Default None defers to StructuredModelConfig (32)."
        ),
    )
    parser.add_argument(
        "--scalar-schema-version", type=int, default=None,
        help=(
            "Structured-trunk: token_scalars row layout the model "
            "consumes (1 = legacy 3-slot row; 2 = adds board / life "
            "state). Default None defers to StructuredModelConfig "
            "(currently 2). Must match the schema the npz files were "
            "generated under — training refuses a width mismatch. "
            "Recorded in metadata.json as scalar_schema_version."
        ),
    )
    parser.add_argument(
        "--value-head-input", default=None,
        choices=SUPPORTED_VALUE_HEAD_INPUTS,
        help=(
            "Structured-trunk: which representation the VALUE head "
            "reads. 'trunk' (default) shares one trunk pass with the "
            "policy head — the value head sees per-printing card "
            "identity and, on the s2 lineage, memorised it. "
            "'trunk_no_printing' runs the value head off a second "
            "pass of the SAME trunk weights with the printing "
            "embeddings dropped, so it can only use board shape / "
            "life / counts / action kinds; the policy head keeps "
            "full information. Adds no parameters; costs one extra "
            "trunk forward per evaluation. Recorded in metadata.json "
            "as training.structured_value_head_input."
        ),
    )
    parser.add_argument(
        "--value-target", default="outcome",
        help=(
            "Structured-trunk only. What the value head regresses: "
            "'outcome' (+/-1 game result; default), 'root' (the "
            "search root value recorded per row in shards written "
            "after 2026-09-26 -- the teacher's own estimate at the "
            "decision; rows without one fall back to the outcome), or "
            "'blend:<w>' (w*root + (1-w)*outcome). The held-out gate "
            "and calibration always score against the OUTCOME so a "
            "root-trained head is judged on what matters at play time. "
            "Recorded in metadata.json as training.value_target."
        ),
    )
    # --- Regularisation + held-out value gate (structured only) -----
    # Added 2026-09-24. Every s2-lineage version trained to value-MSE
    # ~0.005 while scoring 1.42-1.78 on a game-level held-out split
    # (worse than the constant-0 predictor's 1.0): the value head
    # memorised game identity. Defaults live here rather than on
    # StructuredModelConfig / train_structured_model so that library
    # callers and old checkpoints keep their exact previous behaviour
    # and only new TRAINING RUNS pick the regularisation up.
    parser.add_argument(
        "--dropout", type=float, default=None,
        help=(
            "Structured-trunk: dropout probability applied to the "
            "transformer encoder layers, the policy / hand-belief "
            "cross-attention AND the value / policy head MLPs. "
            "Default (when omitted) 0.1. Pass 0 to reproduce the "
            "pre-2026-09-24 unregularised runs. Ignored by the "
            "flat-feature architectures."
        ),
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help=(
            "Structured-trunk: AdamW decoupled weight decay. Default "
            "(when omitted) 1e-4; 0 reproduces plain Adam. Ignored by "
            "the flat-feature architectures."
        ),
    )
    parser.add_argument(
        "--keep-last-epoch", action="store_true",
        help=(
            "Save the FINAL epoch's weights instead of the best-"
            "validation epoch's. Default (flag absent) is restore-"
            "best: after early stopping (or the epoch cap) the "
            "state_dict from the best val epoch is reloaded before "
            "the weights are written. Pre-2026-09-24 behaviour was "
            "always keep-last, which threw away the very epoch early "
            "stopping had identified."
        ),
    )
    parser.add_argument(
        "--min-holdout-value-quality", type=float,
        default=DEFAULT_MIN_HOLDOUT_VALUE_QUALITY,
        help=(
            "Structured-trunk only (the flat-feature trainer has no "
            "held-out scoring pass and ignores this). "
            "Held-out value-MSE CEILING. After training, the value "
            "head is re-scored on the game-level held-out split; if "
            "its MSE is at or above this number the run is REJECTED "
            "-- nothing is registered and the process exits "
            f"{_GATE_REJECT_EXIT_CODE}. Quality is measured as MSE, "
            "so a SMALLER value is a STRICTER gate. Default "
            f"{DEFAULT_MIN_HOLDOUT_VALUE_QUALITY} is exactly the "
            "constant-0 predictor's score against +/-1 outcome "
            "labels: a value head that has learned nothing useful "
            "cannot be promoted. Pass 0 (or a negative number) to "
            "disable the gate. The metrics are written to "
            "metadata.json and the progress JSON either way."
        ),
    )
    parser.add_argument(
        "--shard-card-vocab", type=str, default=None,
        help=(
            "Structured-trunk only. Card-vocabulary file (from `mtg_bot "
            "card-vocab export`, a <weights>.card_vocab.json sidecar, or a "
            "stamped metadata.json) ASSERTING the vocabulary that "
            "pre-fingerprint training shards were generated under. "
            "Shards written since the fingerprint landed record their "
            "own vocabulary and need nothing; unstamped shards are "
            "refused without this (never guessed). Default: "
            f"${'{'}{SHARD_CARD_VOCAB_ENV}{'}'} if set. The trained model "
            "is stamped with the shards' vocabulary."
        ),
    )
    args = parser.parse_args(argv)
    if args.loader_timeout_sec is not None and args.loader_timeout_sec < 0:
        parser.error(
            "--loader-timeout-sec must be >= 0 (0 disables the "
            f"stall guard); got {args.loader_timeout_sec}"
        )
    # Structured architecture has a different data pipeline + model
    # + loss shape from the flat-feature MLP trainers; dispatch
    # before any of the flat-feature-specific argument validation.
    if args.architecture == ARCHITECTURE_STRUCTURED:
        _main_structured(args, parser)
        return
    # The regularisation + value-gate flags are structured-only (the
    # flat MLP has no dropout module and its trainer has no held-out
    # scoring pass). Refuse rather than silently ignoring an
    # explicitly-passed knob — a flag that does nothing is worse than
    # an error.
    _flat_only_rejects = [
        name for name, val in (
            ("--dropout", args.dropout),
            ("--weight-decay", args.weight_decay),
        ) if val is not None
    ]
    if args.keep_last_epoch:
        _flat_only_rejects.append("--keep-last-epoch")
    if args.value_target != "outcome":
        _flat_only_rejects.append("--value-target")
    if _flat_only_rejects:
        parser.error(
            f"{', '.join(_flat_only_rejects)} only apply to "
            f"--architecture structured; got "
            f"--architecture {args.architecture}."
        )
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
    _validate_window_args(args, parser)
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
    # Resolve the training window (iterations or raw file count) to an
    # explicit file list BEFORE loading, and hand the loader a
    # directory view restricted to that list — the loaders' own
    # ``max_files`` knob can only express "N most recent by mtime",
    # which is the semantics this window replaces.
    load_dir, data_files = _resolve_data_window(args)
    # Always pull policy supervision arrays. ``include_policy=True``
    # zero-pads value-only files so the concatenated arrays align —
    # mixed dirs (some pre-A2.3 .npz, some post-A2.3) load cleanly.
    # ``include_group_ids=True`` also returns per-row game group ids
    # (from each file's ``game_ids``, or whole-file groups for legacy
    # NPZs) so the trainer can split train/val by game, not by row.
    features, outcomes, policy_targets, action_masks, group_ids = (
        load_all_datasets(
            load_dir,
            include_policy=True,
            include_group_ids=True,
        )
    )
    if isinstance(load_dir, WindowedDataDir):
        load_dir.check_enumerated()
    if features.shape[0] == 0:
        sys.exit(
            f"No .npz files found in {args.data_dir}. Generate data "
            f"with scripts/generate_training_data.py first."
        )

    if args.one_legal_keep_frac < 1.0:
        n_before = features.shape[0]
        (
            features,
            outcomes,
            policy_targets,
            action_masks,
            n_one_legal_total,
            n_one_legal_kept,
            group_ids,
        ) = _downsample_one_legal_rows(
            features, outcomes, policy_targets, action_masks,
            keep_frac=args.one_legal_keep_frac,
            seed=args.seed,
            group_ids=group_ids,
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

    # Resolve the progress-file path BEFORE training so we can
    # register it as a ``progress_json`` artifact on the run row.
    # We peek at the next version ID rather than reserve it — the
    # registry assigns sequentially at register time, so as long as
    # nobody else trains concurrently the peek value matches the
    # eventual assignment.
    #
    # Phase 3 cutover: the JSON file itself is no longer written
    # (the UI reads runs.db directly). The path string survives as
    # an artifact pointer for historical-compatibility reasons —
    # downstream tooling that walks ``run_artifacts`` keeps seeing
    # a ``progress_json`` entry per training run, even though that
    # file no longer lands on disk for new runs.
    registry = Registry(args.registry_root)
    peeked_version = registry.next_version_id()
    # The run_id salts the registry version label with the start
    # timestamp so re-running ``mtg_bot train`` against the same
    # version label (fresh checkout, manual recovery train, ...) lands
    # a NEW ``runs.run_id`` row rather than colliding with the prior
    # row's UNIQUE constraint. Iterate's subprocess-invoked train
    # supplies its own iter-namespaced run_id via the parent lifecycle
    # CM; this build only governs the standalone-train row.
    train_run_id = _build_train_run_id(peeked_version, _now_stamp())
    if args.progress_path is None:
        # The parent dir comes from --progress-dir so an iterate /
        # launcher lineage that runs with --progress-dir
        # training_progress_s2 keeps its training cards (and the
        # .STOP sentinels the UI writes next to them) in ITS
        # directory. Before this was threaded, the trainer's default
        # hardcoded ``training_progress`` and cycle-5 of the s2
        # lineage printed
        # ``# Progress JSON: training_progress\s2_iter_005_train_v6.json``.
        progress_dir = Path(args.progress_dir)
        progress_dir.mkdir(parents=True, exist_ok=True)
        # Default progress JSON tracks the run_id (not just the
        # version label) so two concurrent / consecutive standalone
        # trains of the same version land in separate files. Iterate's
        # explicit ``--progress-path`` still wins via the elif/else
        # branches below.
        progress_path: Path | None = progress_dir / f"{train_run_id}.json"
    elif str(args.progress_path) == "":
        progress_path = None
    else:
        progress_path = args.progress_path
        progress_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1 dual-write: alongside the existing JSON progress
    # emission, push start_run / per-epoch progress + events /
    # finish_run into the SQLite run DB. Lifecycle helpers are no-ops
    # when neither MTG_BOT_RUN_DB_PATH nor MTG_BOT_RUN_DB_URL is set,
    # so this stays additive on a fresh clone.
    train_config = {
        "kind": "training",
        "data_dir": str(args.data_dir),
        "registry_root": str(args.registry_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": args.val_split,
        "device": device,
        "seed": args.seed,
        "notes": args.notes,
        "compile_model": args.compile_model,
        "hidden_sizes": list(hidden_sizes),
        "policy_hidden_sizes": (
            list(policy_hidden_sizes)
            if policy_hidden_sizes is not None else None
        ),
        "architecture": args.architecture,
        "policy_loss_weight": args.policy_loss_weight,
        "early_stop_patience": args.early_stop_patience,
        "window_recent": args.window_recent,
        "window_recent_files": args.window_recent_files,
        "one_legal_keep_frac": args.one_legal_keep_frac,
        "action_encoding_version": args.action_encoding_version,
        "progress_path": (
            str(progress_path) if progress_path is not None else None
        ),
        "has_any_policy": has_any_policy,
        "n_samples": int(features.shape[0]),
        "n_policy_rows": n_policy_rows,
    }
    parent_run_id = _parent_run_id_from_notes(args.notes)

    def _on_epoch(payload: dict) -> None:
        """Per-epoch sink: push a ``run_progress`` upsert keyed by
        the run_id (no shard_label = aggregate row) and append one
        ``epoch_completed`` event carrying the per-epoch loss
        components. ``n_games_done`` reuses the existing field as
        ``current_epoch`` (mirrors how the JSON producer overloads
        the same column for training kind)."""
        _runs_client.push_progress(
            train_run_id,
            n_games_done=int(payload["epoch"]),
            n_games_total=int(payload["total_epochs"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            extra={
                "current_epoch": int(payload["epoch"]),
                "total_epochs": int(payload["total_epochs"]),
                "train_loss": payload["train_loss"],
                "val_loss": payload["val_loss"],
                "train_value_loss": payload["train_value_loss"],
                "val_value_loss": payload["val_value_loss"],
                "train_policy_loss": payload["train_policy_loss"],
                "val_policy_loss": payload["val_policy_loss"],
                # Full per-epoch arrays for the UI's loss chart (see
                # the structured _on_epoch below for the long version
                # of why this matters now that the UI is runs.db-only).
                "train_losses": list(payload.get("train_losses", [])),
                "val_losses": list(payload.get("val_losses", [])),
                "train_value_losses": list(
                    payload.get("train_value_losses", []),
                ),
                "val_value_losses": list(
                    payload.get("val_value_losses", []),
                ),
                "train_policy_losses": list(
                    payload.get("train_policy_losses", []),
                ),
                "val_policy_losses": list(
                    payload.get("val_policy_losses", []),
                ),
                "status": payload["status"],
            },
        )
        _runs_client.push_event(
            train_run_id,
            event_type="epoch_completed",
            payload={
                "epoch": int(payload["epoch"]),
                "train_loss": payload["train_loss"],
                "val_loss": payload["val_loss"],
                "train_value_loss": payload["train_value_loss"],
                "val_value_loss": payload["val_value_loss"],
                "train_policy_loss": payload["train_policy_loss"],
                "val_policy_loss": payload["val_policy_loss"],
            },
        )

    # Train into a temp file, then hand off to the registry which
    # copies the bytes into the versioned directory. The DB
    # ``runs.label`` column carries the registry version label
    # (e.g. ``v10``) so the UI can still group successive train
    # attempts of the same version together; ``run_id`` is the unique
    # timestamped form so the UNIQUE constraint never collides.
    with tempfile.TemporaryDirectory() as tmp, _lifecycle.run_lifecycle(
        run_id=train_run_id,
        kind="training",
        config=train_config,
        progress_json_path=progress_path,
        parent_run_id=parent_run_id,
        label=peeked_version,
    ):
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
            # Phase 3 cutover: the per-epoch JSON dual-write is dead I/O
            # now that the UI reads runs.db directly. ``_on_epoch``
            # pushes the same per-epoch progress + ``epoch_completed``
            # event into the DB. Pass ``progress_path=None`` so
            # train_value_network skips its ``atomic_write_json`` calls;
            # ``progress_path`` (the path string) is still registered as
            # a ``progress_json`` artifact via run_lifecycle for
            # historical compatibility.
            "progress_path": None,
            "run_id": train_run_id,
            "notes": args.notes,
            "compile_model": args.compile_model,
            "hidden_sizes": hidden_sizes,
            "early_stop_patience": args.early_stop_patience,
            "epoch_callback": _on_epoch,
            "group_ids": group_ids,
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
                "window_recent_files": args.window_recent_files,
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
                # ``--epochs 0`` (smoke runs that just want a fresh
                # registered version with no training) produces empty
                # loss lists; guard the ``[-1]`` accesses so the post-
                # train summary doesn't IndexError.
                "final_train_loss": (
                    history.train_losses[-1]
                    if history.train_losses else None
                ),
                "final_val_loss": (
                    history.val_losses[-1]
                    if history.val_losses else None
                ),
                "best_val_loss": history.best_val_loss,
                "best_val_epoch": history.best_val_epoch,
            },
            git_commit=_git_commit(),
            notes=args.notes,
        )
        version = registry.register(tmp_weights, metadata)

        # Final post-train DB writes (still inside the lifecycle
        # context so any failure propagates as status=failed):
        #   1. Aggregate row carries the full training_history blob
        #      as ``extra_json["training_history"]``. A sibling slice
        #      may later add a dedicated ``training_history_json``
        #      column to ``runs``; until then this is the agreed
        #      stash location.
        #   2. Register the registered weights file as an artifact
        #      so the UI / downstream tooling can locate it without
        #      walking the registry tree.
        final_history = {
            "train_losses": list(history.train_losses),
            "val_losses": list(history.val_losses),
            "train_value_losses": list(history.train_value_losses),
            "val_value_losses": list(history.val_value_losses),
            "train_policy_losses": list(history.train_policy_losses),
            "val_policy_losses": list(history.val_policy_losses),
            "best_val_loss": history.best_val_loss,
            "best_val_epoch": history.best_val_epoch,
            "epochs_trained": len(history.train_losses),
        }
        if history.train_losses:
            final_history["final_train_loss"] = history.train_losses[-1]
        if history.val_losses:
            final_history["final_val_loss"] = history.val_losses[-1]
        _runs_client.push_progress(
            train_run_id,
            n_games_done=len(history.train_losses),
            n_games_total=args.epochs,
            elapsed_seconds=elapsed,
            extra={
                "training_history": final_history,
                "registered_version": version,
            },
        )
        weights_path = registry.weights_path(version)
        try:
            weights_bytes = weights_path.stat().st_size
        except OSError:
            weights_bytes = None
        _runs_client.push_artifact(
            train_run_id,
            artifact_type="weights",
            path=str(weights_path),
            bytes_=weights_bytes,
        )

    print(f"\n# Done in {elapsed:.1f}s. Registered as {version}.")
    print(f"# Weights:  {registry.weights_path(version)}")
    # ``--epochs 0`` short-circuits the post-train summary block: the
    # registered version exists, but there were no epochs to report
    # losses for. Don't IndexError on the empty history lists.
    if history.train_losses:
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
    else:
        print("# epochs_trained=0; no per-epoch losses to report")


def _fmt_opt(val) -> str:
    """Format an ``Optional[float]`` for the operator-facing lines."""
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def _print_calibration_summary(metrics) -> None:
    """Print the raw-vs-calibrated held-out numbers.

    The gate judges the calibrated MSE, so an operator reading only
    the gate line would not see how much of the pass (or fail) came
    from the temperature. Print both, always.
    """
    cal = getattr(metrics, "calibration", None) or {}
    method = cal.get("method")
    if not method or method == "identity":
        return
    print(
        f"# Value calibration: {method} "
        f"T={_fmt_opt(cal.get('temperature'))} "
        f"bias={_fmt_opt(cal.get('bias'))}"
    )
    print(
        f"#   held-out MSE  raw {_fmt_opt(metrics.value_mse)} | "
        f"calibrated {_fmt_opt(metrics.calibrated_value_mse)} | "
        f"cross-fitted {_fmt_opt(cal.get('cv_mse'))} | "
        f"constant-0 {_fmt_opt(metrics.constant_zero_mse)}"
    )
    print(
        f"#   corr(pred, outcome)  raw {_fmt_opt(metrics.value_corr)} "
        f"| calibrated {_fmt_opt(metrics.calibrated_value_corr)} | "
        f"life-diff baseline "
        f"{_fmt_opt(metrics.baseline_life_diff_corr)}"
    )
    print(
        f"#   mean |pred|  raw {_fmt_opt(metrics.pred_abs_mean)} | "
        f"calibrated {_fmt_opt(metrics.calibrated_pred_abs_mean)}"
    )
    # Advisory, NOT a gate: calibration fixes the SCALE, never the
    # information content. A head whose correlation is still under
    # the one-line life-differential baseline has learned nothing
    # that baseline doesn't already know, and passing the MSE
    # ceiling on a shrunken output must not read as "solved".
    head_corr = metrics.calibrated_value_corr
    if head_corr is None:
        head_corr = metrics.value_corr
    baseline = metrics.baseline_life_diff_corr
    if (
        head_corr is not None and baseline is not None
        and abs(head_corr) <= abs(baseline)
    ):
        print(
            f"#   NOTE: corr(pred, outcome) {_fmt_opt(head_corr)} does "
            f"not beat the trivial life-differential baseline "
            f"{_fmt_opt(baseline)} on the same rows. Calibration "
            f"fixes the SCALE, not the information — this head is "
            f"still no more informative than 'who is ahead on life'."
        )


def _main_structured(args, parser) -> None:
    """Train + register a :class:`StructuredModel` checkpoint.

    Reached when ``--architecture structured`` is passed. Loads the
    structured-token npz files from ``--data-dir``, runs
    :func:`train_structured_model`, and registers the result with
    ``metadata.architecture='structured'`` so loaders dispatch to the
    transformer trunk at inference time.

    The flat-feature flags (``--one-legal-keep-frac``,
    ``--window-recent``, ``--policy-hidden-sizes``,
    ``--compile``, ``--hidden-sizes``) DO NOT apply to the
    structured trainer — its capacity is set by
    :class:`StructuredModelConfig`'s defaults today, not by the
    flat MLP's trunk shape. We accept them on the namespace but
    ignore (with a stderr note when they were set non-default) rather
    than erroring so iterate's shared CLI shim isn't forced to fork
    its arg-construction path.
    """
    from mtg_bot.training.structured_model import StructuredModelConfig

    if not args.data_dir.is_dir():
        sys.exit(f"--data-dir {args.data_dir} is not a directory")
    if args.early_stop_patience is not None and args.early_stop_patience < 1:
        parser.error(
            "--early-stop-patience must be a positive int (>= 1); got "
            f"{args.early_stop_patience}"
        )
    _validate_window_args(args, parser)

    print(
        f"# Structured-trainer: loading data from {args.data_dir}"
    )
    # Resolve the iteration (or file-count) window to an explicit file
    # list and load through a directory view restricted to it — see
    # :func:`select_window_files` / :class:`WindowedDataDir`.
    load_dir, data_files = _resolve_data_window(args)
    # The shards' printing ids live in ONE card-vocabulary id space
    # (or training refuses); the trained model is stamped with it.
    # Resolved before the expensive flat-store build so a mixed or
    # unstamped pool fails in seconds, not after the load.
    shard_vocab = None
    legacy_source = args.shard_card_vocab or os.environ.get(
        SHARD_CARD_VOCAB_ENV,
    )
    if data_files:
        try:
            shard_vocab = resolve_shards_card_vocab(
                [Path(args.data_dir) / name for name in data_files],
                legacy_vocab=legacy_shard_vocab_from_arg(
                    args.shard_card_vocab,
                ),
            )
        except CardVocabError as exc:
            sys.exit(f"card vocabulary: {exc}")
    if shard_vocab is not None:
        print(
            f"# Structured-trainer: shards use card vocabulary "
            f"{shard_vocab.vocab.short_sha} ({shard_vocab.vocab.size} "
            f"names, n_printings cap {shard_vocab.n_printings_cap}; "
            f"{shard_vocab.n_stamped} stamped / {shard_vocab.n_legacy} "
            f"legacy files"
            + (
                f", legacy vocabulary asserted by {legacy_source}"
                if shard_vocab.legacy_assumed else ""
            )
            + ")"
        )
    try:
        dataset = load_structured_dataset(load_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    except ValueError as exc:
        # Surface the data-pipeline issue verbatim — most likely "npz
        # missing structured arrays" which means the data_gen run
        # wasn't invoked with structured_tokens=True.
        sys.exit(
            f"--architecture structured requires npz files written "
            f"with structured_tokens=True. {exc}"
        )

    if isinstance(load_dir, WindowedDataDir):
        load_dir.check_enumerated()
    n_samples = len(dataset)
    if n_samples == 0:
        sys.exit(f"No samples found in {args.data_dir}")
    if shard_vocab is None:  # pragma: no cover - rows imply a resolution
        sys.exit(
            "card vocabulary: the loaded shards carry rows but no card "
            "vocabulary could be resolved for them"
        )

    device = args.device or auto_device()
    print(
        f"# Structured-trainer: {n_samples} samples from "
        f"{len(data_files)} files; device={device}; "
        f"epochs={args.epochs} batch={args.batch_size}"
    )

    registry = Registry(args.registry_root)
    peeked_version = registry.next_version_id()
    # Salt the run_id with the start timestamp so the standalone-train
    # row is unique per invocation — see :func:`_build_train_run_id`'s
    # docstring + the flat-feature ``main()`` block above for the
    # ``run_id == label`` UNIQUE-collision bug this resolves.
    train_run_id = _build_train_run_id(peeked_version, _now_stamp())
    if args.progress_path is None:
        # The parent dir comes from --progress-dir so an iterate /
        # launcher lineage that runs with --progress-dir
        # training_progress_s2 keeps its training cards (and the
        # .STOP sentinels the UI writes next to them) in ITS
        # directory. Before this was threaded, the trainer's default
        # hardcoded ``training_progress`` and cycle-5 of the s2
        # lineage printed
        # ``# Progress JSON: training_progress\s2_iter_005_train_v6.json``.
        progress_dir = Path(args.progress_dir)
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_path: Path | None = progress_dir / f"{train_run_id}.json"
    elif str(args.progress_path) == "":
        progress_path = None
    else:
        progress_path = args.progress_path
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path is not None:
        print(f"# Progress JSON: {progress_path}")

    # Apply any explicit hyperparam overrides from the CLI; defaults
    # of ``None`` defer to StructuredModelConfig's own defaults (the
    # roadmap-pinned 128/4/2/256/32 set).
    structured_overrides: dict = {}
    if args.emb_dim is not None:
        structured_overrides["emb_dim"] = args.emb_dim
    if args.n_heads is not None:
        structured_overrides["n_heads"] = args.n_heads
    if args.n_layers is not None:
        structured_overrides["n_layers"] = args.n_layers
    if args.hidden_dim is not None:
        structured_overrides["hidden_dim"] = args.hidden_dim
    if args.action_history_n is not None:
        structured_overrides["action_history_n"] = args.action_history_n
    if args.scalar_schema_version is not None:
        structured_overrides["scalar_schema_version"] = (
            args.scalar_schema_version
        )
    if args.value_head_input is not None:
        structured_overrides["value_head_input"] = args.value_head_input
    dropout = (
        DEFAULT_STRUCTURED_DROPOUT if args.dropout is None
        else float(args.dropout)
    )
    if not (0.0 <= dropout < 1.0):
        parser.error(
            f"--dropout must be in [0.0, 1.0); got {dropout}"
        )
    structured_overrides["dropout"] = dropout
    weight_decay = (
        DEFAULT_STRUCTURED_WEIGHT_DECAY if args.weight_decay is None
        else float(args.weight_decay)
    )
    if weight_decay < 0.0:
        parser.error(
            f"--weight-decay must be >= 0; got {weight_decay}"
        )
    restore_best = not args.keep_last_epoch
    try:
        structured_config = StructuredModelConfig(**structured_overrides)
    except ValueError as exc:
        parser.error(str(exc))
    # Refuse a scalar-schema width mismatch up front with a CLI-shaped
    # message; ``train_structured_model`` re-checks as the backstop.
    dataset_scalar_width = getattr(dataset, "scalar_width", None)
    if (
        dataset_scalar_width is not None
        and dataset_scalar_width != structured_config.scalar_width
    ):
        parser.error(
            f"scalar schema mismatch: npz files in {args.data_dir} carry "
            f"token_scalars rows of width {dataset_scalar_width}, but "
            f"--scalar-schema-version "
            f"{structured_config.scalar_schema_version} expects width "
            f"{structured_config.scalar_width}. Regenerate the data "
            f"under the model's schema or pass a matching "
            f"--scalar-schema-version."
        )
    # The shards' printing ids were bucketed at their recorded cap; a
    # model with a different table size would index rows the data never
    # trained (or collide differently than the data assumes).
    if (
        shard_vocab is not None
        and structured_config.n_printings != shard_vocab.n_printings_cap
    ):
        parser.error(
            f"n_printings mismatch: the training shards' printing ids "
            f"were tokenised with n_printings cap "
            f"{shard_vocab.n_printings_cap}, but the model would have "
            f"n_printings={structured_config.n_printings}."
        )

    train_config = {
        "kind": "training",
        "architecture": ARCHITECTURE_STRUCTURED,
        "data_dir": str(args.data_dir),
        "registry_root": str(args.registry_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": args.val_split,
        "device": device,
        "seed": args.seed,
        "notes": args.notes,
        "n_samples": n_samples,
        "window_recent": args.window_recent,
        "window_recent_files": args.window_recent_files,
        "action_encoding_version": args.action_encoding_version,
        "policy_loss_weight": args.policy_loss_weight,
        "early_stop_patience": args.early_stop_patience,
        "num_workers": args.num_workers,
        "loader_timeout_sec": args.loader_timeout_sec,
        "shuffle_mode": args.shuffle_mode,
        "shuffle_block_rows": args.shuffle_block_rows,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "restore_best": restore_best,
        "min_holdout_value_quality": args.min_holdout_value_quality,
        "progress_path": (
            str(progress_path) if progress_path is not None else None
        ),
        "structured_config": {
            "emb_dim": structured_config.emb_dim,
            "n_heads": structured_config.n_heads,
            "n_layers": structured_config.n_layers,
            "hidden_dim": structured_config.hidden_dim,
            "dropout": structured_config.dropout,
            "action_history_n": structured_config.action_history_n,
            "n_printings": structured_config.n_printings,
            "scalar_schema_version": (
                structured_config.scalar_schema_version
            ),
            "value_head_input": structured_config.value_head_input,
        },
    }
    parent_run_id = _parent_run_id_from_notes(args.notes)

    def _on_epoch(payload: dict) -> None:
        _runs_client.push_progress(
            train_run_id,
            n_games_done=int(payload["epoch"]),
            n_games_total=int(payload["total_epochs"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            extra={
                "current_epoch": int(payload["epoch"]),
                "total_epochs": int(payload["total_epochs"]),
                "train_loss": payload["train_loss"],
                "val_loss": payload["val_loss"],
                "train_value_loss": payload["train_value_loss"],
                "val_value_loss": payload["val_value_loss"],
                "train_policy_loss": payload["train_policy_loss"],
                "val_policy_loss": payload["val_policy_loss"],
                # Full per-epoch arrays so the UI's per-loss-component
                # chart has data points across the whole run, not just
                # the latest scalar. Older trainer payloads (pre-
                # 2026-05-19) omitted these; ``.get(..., [])`` keeps
                # the structured path back-compat with the flat-
                # feature trainer's callback shape.
                "train_losses": list(payload.get("train_losses", [])),
                "val_losses": list(payload.get("val_losses", [])),
                "train_value_losses": list(
                    payload.get("train_value_losses", []),
                ),
                "val_value_losses": list(
                    payload.get("val_value_losses", []),
                ),
                "train_policy_losses": list(
                    payload.get("train_policy_losses", []),
                ),
                "val_policy_losses": list(
                    payload.get("val_policy_losses", []),
                ),
                "status": payload["status"],
                "architecture": ARCHITECTURE_STRUCTURED,
            },
        )
        _runs_client.push_event(
            train_run_id,
            event_type="epoch_completed",
            payload={
                "epoch": int(payload["epoch"]),
                "train_loss": payload["train_loss"],
                "val_loss": payload["val_loss"],
                "train_value_loss": payload["train_value_loss"],
                "val_value_loss": payload["val_value_loss"],
                "train_policy_loss": payload["train_policy_loss"],
                "val_policy_loss": payload["val_policy_loss"],
            },
        )

    try:
      with tempfile.TemporaryDirectory() as tmp, _lifecycle.run_lifecycle(
        run_id=train_run_id,
        kind="training",
        config=train_config,
        progress_json_path=progress_path,
        parent_run_id=parent_run_id,
        label=peeked_version,
      ):
        tmp_weights = Path(tmp) / "weights.pt"
        started = time.perf_counter()
        history = train_structured_model(
            dataset,
            weights_out_path=tmp_weights,
            config=structured_config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split,
            device=device,
            rng_seed=args.seed,
            progress_path=progress_path,
            run_id=train_run_id,
            notes=args.notes,
            policy_loss_weight=args.policy_loss_weight,
            epoch_callback=_on_epoch,
            # ``--compile`` is opt-in (action="store_true"). Without
            # this passthrough the structured trainer's internal
            # default fires (``compile_model=None`` → True on CUDA),
            # which crashes on Windows-cu128 (Triton not packaged)
            # because the failure surfaces lazily on the first forward,
            # AFTER the try/except around ``torch.compile()`` itself.
            # The flat-feature train path already passes this through
            # (lines 549, 646 above); structured was the outlier.
            compile_model=args.compile_model,
            early_stop_patience=args.early_stop_patience,
            num_workers=args.num_workers,
            loader_timeout_sec=args.loader_timeout_sec,
            weight_decay=weight_decay,
            restore_best=restore_best,
            shuffle_mode=args.shuffle_mode,
            shuffle_block_rows=args.shuffle_block_rows,
            value_target=args.value_target,
        )
        elapsed = time.perf_counter() - started

        # --- held-out value-quality gate ------------------------------
        # Decided BEFORE ``registry.register`` so a rejected run leaves
        # no promotable version behind. The metrics are surfaced either
        # way (progress JSON above, metadata.json below, run-DB event
        # here) so a rejection is diagnosable without a re-run.
        holdout_metrics = history.holdout_value_metrics
        gate = evaluate_value_gate(
            holdout_metrics,
            min_holdout_value_quality=args.min_holdout_value_quality,
        )
        print(f"# Held-out value gate: {gate.reason}")
        _print_calibration_summary(holdout_metrics)
        _merge_into_progress_json(
            progress_path,
            {
                "holdout_value_metrics": holdout_metrics.as_dict(),
                "value_gate": gate.as_dict(),
            },
        )
        _runs_client.push_event(
            train_run_id,
            event_type=(
                "value_gate_passed" if gate.passed
                else "value_gate_rejected"
            ),
            payload={
                "gate": gate.as_dict(),
                "metrics": holdout_metrics.as_dict(),
                "peeked_version": peeked_version,
            },
        )
        if not gate.passed:
            raise _ValueGateRejected(
                format_gate_failure(
                    gate, holdout_metrics,
                    version_label=peeked_version,
                )
            )

        metadata = ModelMetadata(
            version="",
            created_at="",
            feature_dim=FEATURE_DIM,  # legacy field; ignored for structured
            hidden_sizes=[],
            with_policy=True,
            architecture=ARCHITECTURE_STRUCTURED,
            action_encoding_version=args.action_encoding_version,
            # Canonical scalar-schema record; loaders size
            # ``scalar_proj`` + the tokenizer's row width from it.
            scalar_schema_version=structured_config.scalar_schema_version,
            # Post-hoc value calibration fitted on the held-out split
            # (see mtg_bot.training.value_calibration). Loaders push
            # temperature/bias into StructuredModelConfig, so search
            # and both eval servers read CALIBRATED values. Empty
            # dict when nothing could be fit (no held-out rows).
            value_calibration=dict(holdout_metrics.calibration or {}),
            # The card vocabulary the printing embeddings were trained
            # on = the id space of the training shards (NOT this
            # process's current pool: shards are pre-tokenised). Loaders
            # remap it to whatever vocabulary is current at load time.
            card_vocab=shard_vocab.vocab.to_dict(
                provenance={
                    **shard_vocab.provenance(legacy_source),
                    "git_commit": _git_commit(),
                    "stamped_at": _now_stamp(),
                },
            ),
            training={
                "data_dir": str(args.data_dir),
                "data_files": data_files,
                "n_samples": n_samples,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "val_split": args.val_split,
                "device": device,
                "rng_seed": args.seed,
                "duration_seconds": elapsed,
                "window_recent": args.window_recent,
                "window_recent_files": args.window_recent_files,
                "early_stop_patience": args.early_stop_patience,
                "epochs_trained": len(history.train_losses),
                # Regularisation knobs, recorded so a version's
                # metadata fully describes how it was fit (and so a
                # later lineage comparison can attribute a held-out
                # improvement to the settings that produced it).
                "dropout": dropout,
                "weight_decay": weight_decay,
                "restore_best": restore_best,
                "restored_best_epoch": history.restored_best_epoch,
                "value_target": args.value_target,
                "min_holdout_value_quality": (
                    args.min_holdout_value_quality
                ),
                "structured_config": train_config["structured_config"],
                # Flat ``structured_*`` mirror of the nested dims
                # above. The loader (``NNEvaluator.from_metadata``)
                # reads either layout via ``_resolve_structured_dim``
                # (flat wins); writing both means a metadata file
                # round-trips through any historical loader version
                # — including pre-iter_019 loaders that ONLY read the
                # flat keys, so a fresh-trained model is still loadable
                # under an old worker that hasn't picked up the
                # loader-side compat shim yet. iter_019 BUG #2 fix.
                "structured_emb_dim": structured_config.emb_dim,
                "structured_n_heads": structured_config.n_heads,
                "structured_n_layers": structured_config.n_layers,
                "structured_hidden_dim": structured_config.hidden_dim,
                "structured_action_history_n": (
                    structured_config.action_history_n
                ),
                "structured_n_printings": structured_config.n_printings,
                "structured_scalar_schema_version": (
                    structured_config.scalar_schema_version
                ),
                "structured_value_head_input": (
                    structured_config.value_head_input
                ),
            },
            training_history={
                "train_losses": history.train_losses,
                "val_losses": history.val_losses,
                "train_value_losses": history.train_value_losses,
                "val_value_losses": history.val_value_losses,
                "train_policy_losses": history.train_policy_losses,
                "val_policy_losses": history.val_policy_losses,
                "final_train_loss": (
                    history.train_losses[-1]
                    if history.train_losses else None
                ),
                "final_val_loss": (
                    history.val_losses[-1]
                    if history.val_losses else None
                ),
                "best_val_loss": history.best_val_loss,
                "best_val_epoch": history.best_val_epoch,
                "restored_best_epoch": history.restored_best_epoch,
                # Value-head generalisation on the game-level held-out
                # split, plus the gate's verdict. Present on every
                # registered version so the lineage can be audited
                # without re-scoring.
                "holdout_value_metrics": holdout_metrics.as_dict(),
                "value_gate": gate.as_dict(),
            },
            git_commit=_git_commit(),
            notes=args.notes,
        )
        version = registry.register(tmp_weights, metadata)

        final_history = {
            "train_losses": list(history.train_losses),
            "val_losses": list(history.val_losses),
            "train_value_losses": list(history.train_value_losses),
            "val_value_losses": list(history.val_value_losses),
            "train_policy_losses": list(history.train_policy_losses),
            "val_policy_losses": list(history.val_policy_losses),
            "best_val_loss": history.best_val_loss,
            "best_val_epoch": history.best_val_epoch,
            "restored_best_epoch": history.restored_best_epoch,
            "holdout_value_metrics": holdout_metrics.as_dict(),
            "value_gate": gate.as_dict(),
            "epochs_trained": len(history.train_losses),
        }
        if history.train_losses:
            final_history["final_train_loss"] = history.train_losses[-1]
        if history.val_losses:
            final_history["final_val_loss"] = history.val_losses[-1]
        _runs_client.push_progress(
            train_run_id,
            n_games_done=len(history.train_losses),
            n_games_total=args.epochs,
            elapsed_seconds=elapsed,
            extra={
                "training_history": final_history,
                "registered_version": version,
                "architecture": ARCHITECTURE_STRUCTURED,
            },
        )
        weights_path = registry.weights_path(version)
        try:
            weights_bytes = weights_path.stat().st_size
        except OSError:
            weights_bytes = None
        _runs_client.push_artifact(
            train_run_id,
            artifact_type="weights",
            path=str(weights_path),
            bytes_=weights_bytes,
        )
    except _ValueGateRejected as exc:
        # The run row is already ``failed`` (the lifecycle CM saw the
        # exception) and nothing was registered. Print the operator
        # message and exit with the dedicated code so ``mtg_bot
        # iterate`` can distinguish "model rejected" from "trainer
        # crashed" and keep the cycle chain going.
        print(str(exc), file=sys.stderr)
        sys.exit(_GATE_REJECT_EXIT_CODE)
    finally:
        # Release the flat store (tens of GB in temp) the moment
        # training is over instead of waiting for the interpreter's
        # atexit hook — the window between "training done" and
        # "process exits" is exactly where an operator abort / kill
        # leaves the store stranded on disk.
        dataset.close()

    print(f"\n# Done in {elapsed:.1f}s. Registered as {version}.")
    print(f"# Weights: {registry.weights_path(version)}")
    if history.train_losses:
        print(
            f"# Final train loss: {history.train_losses[-1]:.4f}; "
            f"val loss: {history.val_losses[-1]:.4f}"
        )
    else:
        print("# epochs_trained=0; no per-epoch losses to report")


if __name__ == "__main__":
    main()
