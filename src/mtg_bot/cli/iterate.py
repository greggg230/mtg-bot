"""AlphaZero-style training-loop driver.

Each iteration:

1. **Self-play data gen** — the current "best" network plays itself
   (same weights both seats), driving PUCT-MCTS via its policy
   prior + value head. The visit-count distributions from those
   games become the policy targets for the next training pass;
   the eventual outcomes become the value targets.
2. **Train next gen** — fits a fresh ``_ValuePolicyMLP`` against
   the accumulated training data. Registers as ``v{N+1}`` in the
   registry.
3. **Evaluate (optional acceptance gate)** — head-to-head match
   between the new and current-best networks. If the new one
   wins at least ``acceptance_threshold`` of the eval games,
   promote it to "best" for the next iteration. Otherwise keep
   the current best and let the next iteration generate fresh
   data with the same network.

Bootstrap: the very first iteration needs a network to drive
self-play, so pass ``--bootstrap-version`` pointing at a
pre-existing registered version (typically v4 — the first
dual-head + AlphaZero-trained model). If no such version exists,
train one manually first (gen0 + train_nn.py).

Per-iteration files:

  training_data/iter_{N}_az_{timestamp}.npz   — self-play data
  trained_weights/v{M}/                       — trained weights
  trained_weights/iterate_state.json          — current "best" pointer

The data directory accumulates — every iteration trains on the
full history. Pass ``--window-recent N`` to train on only the last
N ITERATIONS of data (every .npz whose filename carries one of the
N highest ``<prefix>iter_NNN`` labels), or ``--window-recent-files
N`` for a hard cap on FILE count; useful once the dir gets large or
when you want to anneal toward more-recent data. The iteration form
is the one you almost always want: the distributed pipeline writes
one .npz per GAME, so a file-count window of 3 means three games.

Usage::

    # Bootstrap from v4, run 5 iterations, no acceptance gate
    # (always promote — simpler, faster early on):
    python scripts/iterate_alphazero.py --bootstrap-version v4 \\
        --n-iterations 5 --games-per-iteration 500 \\
        --epochs 50 --acceptance-threshold 0.0

    # With a 55% acceptance gate (AlphaGo Zero default):
    python scripts/iterate_alphazero.py --bootstrap-version v4 \\
        --n-iterations 10 --games-per-iteration 1000 \\
        --epochs 100 --acceptance-threshold 0.55 --eval-games 40

The script is incremental — each iteration's outputs land on
disk before the next starts, so a kill mid-run is safe (resume
by re-invoking; the current-best pointer + data files persist).
"""

from __future__ import annotations

from mtg_bot.search.leaf_batch import SEARCH_LEAF_BATCH_SIZE

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Repo root, three levels up from ``src/mtg_bot/cli/iterate.py``. Used
# to locate the ``scripts/train_nn.py`` shim that this loop spawns
# via subprocess (the shim then routes into ``mtg_bot.cli.train``)
# and to root the orphan-sweep matcher.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
# Default location for shards_config.json (auto-detected when
# ``--shards-config`` is not passed). Matches the convention
# ``mtg_bot serve`` already uses for its own default. Auto-detection
# is anchored at the repo root specifically (NOT cwd) so test
# isolation holds — a test running from an arbitrary tmpdir or a
# user running ``mtg_bot iterate`` from a sibling directory both
# see the same canonical file.
_DEFAULT_SHARDS_CONFIG_PATH = _REPO_ROOT / "shards_config.json"
# Sentinel ``Path`` returned by ``_shards_config_type`` when the
# operator explicitly passed ``--shards-config ""`` to force local
# Pool mode (i.e. opt out of auto-detection). This is preferable to
# a magic string because it round-trips cleanly through
# ``argparse.Namespace`` and keeps the rest of the dispatch a
# ``Path`` vs ``Path`` comparison.
_LOCAL_POOL_SENTINEL = Path("__iterate_local_pool__")


def _shards_config_type(s: str) -> Path:
    """Argparse type for ``--shards-config``.

    * Non-empty string → ``Path(s)`` (normal path).
    * Empty string (``--shards-config ""``) → ``_LOCAL_POOL_SENTINEL``,
      explicit opt-out of auto-detection so the loop runs local Pool.

    Mapping the empty string to a sentinel Path instance (rather than
    letting ``Path("")`` resolve to ``WindowsPath(".")``) keeps the
    explicit-opt-out signal distinguishable from any real on-disk
    path.
    """
    if s == "":
        return _LOCAL_POOL_SENTINEL
    return Path(s)

import os

from mtg_bot.cli import _common
from mtg_bot.cli.runs import client as _runs_client
from mtg_bot.cli.runs import lifecycle as _lifecycle
from mtg_bot.decks import Deck, decks_from_json
from mtg_bot import run_labels as _run_labels
from mtg_bot.training import (
    DataGenProgress,
    EvalProgress,
    HeuristicConfig,
    HeuristicsConfig,
    NNPimcConfig,
    PimcConfig,
    Registry,
    generate_training_data_parallel,
    save_dataset,
)
from mtg_bot.training._iterate_helpers import (
    find_foreign_npzs,
)
from mtg_bot.training.self_play_evaluator import (
    EVALUATOR_HEURISTIC as _EVALUATOR_HEURISTIC,
    SELF_PLAY_EVALUATORS as _SELF_PLAY_EVALUATORS,
)
from mtg_bot.training.data_gen import ReplayCaptureConfig
from mtg_bot.training.block_shuffle import DEFAULT_BLOCK_ROWS
from mtg_bot.training.gate import EVAL_GATE_THRESHOLD
from mtg_bot.training.structured_model import (
    SUPPORTED_VALUE_HEAD_INPUTS,
)
from mtg_bot import distributed_rest_orchestrator as _rest_orch
from mtg_bot.training.orphan_sweep import sweep_local_orphans
from mtg_bot.training.self_play import run_match_parallel
from mtg_bot.training.stop_signal import (
    cleanup_orphans, stop_requested, sentinel_path,
)

_STATE_FILENAME = "iterate_state.json"
_VERSION_DIR_RE = re.compile(r"^v(\d+)$")


def _push_cycle_stage_progress(
    *,
    cycle_id: str,
    iter_idx_one_based: int,
    n_iterations: int,
    started_perf: float,
    stage: str,
    notes: str = "",
) -> None:
    """Push cycle-level progress to the runs DB. Carries the iter
    index / stage / elapsed in ``extra_json`` for the UI's compact
    "iter N of M, stage=<self_play|train|eval>" card.

    Replaces the pre-Phase-3 ``_write_cycle_progress`` JSON writer:
    the UI now reads pure DB, so the cycle's state lives in
    ``run_progress`` (counters + extra_json) alongside the
    structured ``runs`` row's ``status`` (managed via
    ``client.finish_run`` at cycle end).
    """
    elapsed = max(time.perf_counter() - started_perf, 0.0)
    extra = {
        "iter_idx": iter_idx_one_based,
        "n_iterations": n_iterations,
        "stage": stage,
    }
    if notes:
        extra["notes"] = notes
    _runs_client.push_progress(
        cycle_id,
        elapsed_seconds=float(elapsed),
        extra=extra,
    )


def _cycle_config_dict(
    *,
    args: argparse.Namespace,
    iter_idx_one_based: int,
    best_version: str,
) -> dict:
    """Build the cycle-level config dict that lands in
    ``runs.config_json``. Captures the iteration's identity (iter #,
    bootstrap) plus the salient hyper-parameters (search depth, game
    count, eval gate, deck pool) so an offline reader can reconstruct
    the run's parameters from the DB alone.

    Replay-capture fields (``replays_dir`` + the two
    ``capture_every_*`` rates) are included so the training UI can
    surface "replay capture: on / off" and the directory captures
    land in, without reading iterate's argparse defaults out-of-band.
    Without these the UI's
    ``if (!run.replays_dir) → 'replay capture: off'`` branch fires
    on every cycle even when capture IS configured.
    """
    return {
        "kind": "cycle",
        "iter_idx": iter_idx_one_based,
        # Which training lineage this cycle belongs to. "" is the
        # original (un-prefixed) lineage. ``getattr`` keeps older
        # callers that build a bare Namespace working.
        "label_prefix": getattr(args, "run_label_prefix", ""),
        "bootstrap_version": args.bootstrap_version,
        "best_version_at_start": best_version,
        "games_per_iteration": args.games_per_iteration,
        "samples": args.samples,
        "nodes": args.nodes,
        "epochs": args.epochs,
        "eval_games": args.eval_games,
        "acceptance_threshold": args.acceptance_threshold,
        "deck_pool": str(args.deck_pool),
        "seed": args.seed,
        "n_iterations": args.n_iterations,
        "replays_dir": str(args.replays_dir),
        "capture_every_self_play": args.capture_every_self_play,
        "capture_every_eval": args.capture_every_eval,
    }


def _self_play_config_dict(
    *,
    args: argparse.Namespace,
    best_version: str,
    cycle_seed: int,
    distributed: bool,
) -> dict:
    """Stage config dict for a self-play stage run row."""
    return {
        "kind": "self_play",
        "best_version": best_version,
        "n_games": args.games_per_iteration,
        "samples": args.samples,
        "nodes": args.nodes,
        "seed": cycle_seed,
        "deck_pool": str(args.deck_pool),
        "distributed": distributed,
        "root_dirichlet_alpha": _dirichlet_alpha_or_none(
            getattr(args, "root_dirichlet_alpha", None)
        ),
        "root_dirichlet_eps": getattr(args, "root_dirichlet_eps", 0.25),
        "move_temperature": getattr(args, "move_temperature", 1.0),
        "temperature_cutoff": getattr(args, "temperature_cutoff", 30),
        # Provenance: a corpus generated by the heuristic evaluator is
        # a fundamentally different dataset from an NN-driven one, and
        # the run row is the only durable record of which it was.
        "self_play_evaluator": getattr(
            args, "self_play_evaluator", "nn",
        ),
        "corpus_only": bool(getattr(args, "corpus_only", False)),
    }


def _iteration_index(
    *,
    iterations_completed: int,
    iter_idx: int,
    corpus_only: bool,
    corpus_batch: "int | None",
) -> int:
    """Return ``n_done`` -- the 0-based iteration index the cycle's
    labels and seeds are derived from (``iter_label(n_done + 1)``).

    Normal runs: the lineage counter from ``iterate_state.json``. The
    loop body advances it by promoting/training, so the in-process
    ``iter_idx`` must NOT be added -- it is re-read each pass.

    ``--corpus-only`` runs never advance that counter (no train, no
    promotion), so the offset has to come from somewhere else, or
    every pass derives the SAME iter_label / cycle_seed / stage_id:
    the coordinator would see an identical, already-drained stage
    spec, answer 204 to every claim, and the run would silently
    produce zero extra games.

    - Within one invocation, ``iter_idx`` supplies the offset.
    - Across invocations (the distributed launcher always spawns
      ``iterate --n-iterations 1``), the caller must pass an explicit
      1-based ``--corpus-batch``.
    """
    if corpus_only and corpus_batch is not None:
        return int(corpus_batch) - 1 + iter_idx
    return int(iterations_completed) + (iter_idx if corpus_only else 0)


def _dirichlet_alpha_or_none(value: "float | None") -> "float | None":
    """CLI convention: ``--root-dirichlet-alpha <= 0`` (or unset)
    disables root noise; ``MCTS`` wants ``None`` for "off"."""
    if value is None:
        return None
    return float(value) if float(value) > 0 else None


def _train_config_dict(
    *,
    args: argparse.Namespace,
    peeked_train_version: str,
    cycle_seed: int,
) -> dict:
    """Stage config dict for a train stage run row."""
    return {
        "kind": "training",
        "peeked_version": peeked_train_version,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_split": args.val_split,
        "seed": cycle_seed,
        "data_dir": str(args.data_dir),
        "window_recent": args.window_recent,
        "window_recent_files": args.window_recent_files,
        "early_stop_patience": args.early_stop_patience,
        "architecture": args.architecture,
        # Regularisation + gate knobs (None = defer to the trainer's
        # own default; the flags are only emitted for the structured
        # architecture).
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "value_head_input": args.value_head_input,
        "value_target": getattr(args, "value_target", None),
        "keep_last_epoch": args.keep_last_epoch,
        "min_holdout_value_quality": args.min_holdout_value_quality,
    }


def _eval_config_dict(
    *,
    args: argparse.Namespace,
    new_version: str,
    best_version: str,
    cycle_seed: int,
    distributed: bool,
) -> dict:
    """Stage config dict for an eval-gate stage run row."""
    return {
        "kind": "eval",
        "new": new_version,
        "old": best_version,
        "games": args.eval_games,
        "samples": args.samples,
        "nodes": args.nodes,
        "seed": cycle_seed + 7777,
        "acceptance_threshold": args.acceptance_threshold,
        "deck_pool": str(args.deck_pool),
        "distributed": distributed,
    }


class ReconcileError(RuntimeError):
    """Raised when iterate_state.json drifts from the registry in a
    way the auto-reconciler refuses to fix on its own. The user must
    inspect and either patch the state file by hand or rerun without
    ``--reconcile-state`` and let the loop crash loudly."""


def _load_iterate_state(registry_root: Path) -> dict:
    """Load the iterate-loop bookkeeping file. Returns an empty
    dict (no current-best yet) if the file doesn't exist."""
    path = registry_root / _STATE_FILENAME
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_iterate_state(registry_root: Path, state: dict) -> None:
    """Write the iterate-loop bookkeeping atomically (tmp + replace)."""
    path = registry_root / _STATE_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _scan_version_dirs(registry_root: Path) -> list[int]:
    """Return sorted version numbers from ``v\\d+`` subdirs under
    ``registry_root``. Mirrors ``Registry.list_versions`` semantics
    (including the complete-files filter) but goes through the
    filesystem directly so the reconciler can detect partially-written
    orphans where weights.pt landed but metadata.json did not."""
    if not registry_root.is_dir():
        return []
    nums: list[int] = []
    for child in registry_root.iterdir():
        if not child.is_dir():
            continue
        match = _VERSION_DIR_RE.match(child.name)
        if match is None:
            continue
        nums.append(int(match.group(1)))
    nums.sort()
    return nums


def _highest_known_version(state: dict, bootstrap_version: str) -> int:
    """Largest version number the state file is aware of: max of
    bootstrap, current_best, and every recorded trained_version in
    history. Used to detect orphan version dirs the loop never wrote
    a history entry for."""
    candidates = [bootstrap_version, state.get("current_best", "")]
    for entry in state.get("history", []):
        v = entry.get("trained_version", "")
        if v:
            candidates.append(v)
    max_n = 0
    for v in candidates:
        m = _VERSION_DIR_RE.match(v) if v else None
        if m is not None:
            max_n = max(max_n, int(m.group(1)))
    return max_n


def _find_eval_progress(
    progress_dir: Path,
    iter_idx_one_based: int,
    new_version: str,
    label_prefix: str = "",
) -> Path | None:
    """Locate the per-iter eval card written by ``EvalProgress``.
    Filename pattern from main():
    ``<prefix>iter_NNN_eval_<new>_vs_<old>.json``.
    Returns the first match (iter+new uniquely identify the file)
    or None if none exists.

    ``label_prefix`` namespaces the lineage (see
    :mod:`mtg_bot.run_labels`); the default ``""`` reproduces the
    historical un-prefixed glob exactly."""
    if not progress_dir.is_dir():
        return None
    label = _run_labels.eval_run_id_prefix(
        iter_idx_one_based, new_version, label_prefix,
    )
    for p in progress_dir.glob(f"{label}*.json"):
        return p
    return None


def _reconcile_iterate_state(
    state: dict,
    registry_root: Path,
    progress_dir: Path,
    bootstrap_version: str,
    acceptance_threshold: float,
    label_prefix: str = "",
) -> dict:
    """Reconcile ``iterate_state.json`` with what's actually on disk.

    Crash-recovery cases handled (see module docstring + the
    ``--reconcile-state`` flag in main()):

    * Case A — state matches reality. No-op; returns the input state.
    * Case B — exactly one orphan version dir whose number is one
      above the highest version the state knows about. The trainer
      wrote ``v{N}/`` but the iter loop crashed before persisting
      the promotion decision. Read the eval progress card written
      by ``EvalProgress`` for that iter; use its ``win_rate`` to
      decide promotion; rewrite state. If no eval card exists (eval
      never ran, or crashed before the first flush) raise so the
      user reconciles by hand.
    * Case C — ``current_best`` references a version whose dir is
      missing. Registry corruption; raise.

    Anything more elaborate (multi-orphan, version gaps, etc.) is
    out of scope and raises ``ReconcileError``.
    """
    current_best = state.get("current_best", bootstrap_version)
    iterations_completed = int(state.get("iterations_completed", 0))

    cb_match = _VERSION_DIR_RE.match(current_best)
    if cb_match is None:
        raise ReconcileError(
            f"current_best={current_best!r} does not match v\\d+ pattern"
        )
    if not (registry_root / current_best).is_dir():
        raise ReconcileError(
            f"current_best={current_best} but {registry_root / current_best} "
            f"does not exist (Case C: registry corruption). Refusing to "
            f"touch state. Inspect the registry by hand."
        )

    version_nums = _scan_version_dirs(registry_root)
    if not version_nums:
        raise ReconcileError(
            f"no v\\d+ directories under {registry_root}; "
            f"can't reconcile against an empty registry"
        )
    max_dir_n = version_nums[-1]
    max_known_n = _highest_known_version(state, bootstrap_version)

    if max_dir_n == max_known_n:
        print(
            f"[reconcile] state matches reality "
            f"(max version v{max_dir_n}, "
            f"iterations_completed={iterations_completed}); no-op"
        )
        return state

    if max_dir_n < max_known_n:
        raise ReconcileError(
            f"state knows about v{max_known_n} but registry max is "
            f"v{max_dir_n} (Case C: missing version dir). Refusing "
            f"to touch state."
        )

    if max_dir_n - max_known_n > 1:
        raise ReconcileError(
            f"registry max is v{max_dir_n} but state's highest known "
            f"version is v{max_known_n}: {max_dir_n - max_known_n} "
            f"orphans is more than one partial-iter crash can produce. "
            f"Refusing to auto-reconcile."
        )

    orphan_n = max_dir_n
    orphan_version = f"v{orphan_n}"
    orphan_iter_idx = iterations_completed + 1

    eval_path = _find_eval_progress(
        progress_dir, orphan_iter_idx, orphan_version, label_prefix,
    )
    if eval_path is None:
        eval_glob = _run_labels.eval_run_id_prefix(
            orphan_iter_idx, orphan_version, label_prefix,
        )
        raise ReconcileError(
            f"orphan {orphan_version} for iter {orphan_iter_idx} but no "
            f"eval progress card found under {progress_dir} matching "
            f"{eval_glob}*.json. "
            f"The crash likely fired before / during eval; "
            f"can't infer promote/keep without a win-rate. Reconcile "
            f"iterate_state.json by hand."
        )

    with eval_path.open("r", encoding="utf-8") as f:
        eval_payload = json.load(f)

    n_decided = (
        int(eval_payload.get("wins_new", 0))
        + int(eval_payload.get("wins_old", 0))
    )
    if n_decided <= 0:
        raise ReconcileError(
            f"eval card {eval_path} has no decided games "
            f"(wins_new+wins_old=0); cannot infer a win-rate. "
            f"Reconcile iterate_state.json by hand."
        )
    win_rate = float(eval_payload.get("win_rate", 0.0))
    promoted = win_rate >= acceptance_threshold

    decision = "promoting" if promoted else "keeping"
    print(
        f"[reconcile] iter {orphan_iter_idx} produced {orphan_version}; "
        f"eval shows win_rate={win_rate:.1%}, "
        f"threshold={acceptance_threshold:.1%}; {decision}"
    )

    new_state = {
        "current_best": (
            orphan_version if promoted else current_best
        ),
        "iterations_completed": orphan_iter_idx,
        "history": list(state.get("history", [])) + [{
            "iteration": orphan_iter_idx,
            "trained_version": orphan_version,
            "promoted": promoted,
            "eval": {
                "vs": current_best,
                "wins_new": int(eval_payload.get("wins_new", 0)),
                "wins_old": int(eval_payload.get("wins_old", 0)),
                "draws": int(eval_payload.get("draws", 0)),
                "win_rate": win_rate,
                "elapsed_seconds": float(
                    eval_payload.get("elapsed_seconds", 0.0)
                ),
            },
            "reconciled": True,
        }],
    }
    return new_state


def _now_stamp() -> str:
    """ISO-like timestamp with no colons (filename-safe)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _compute_slack_extra(target_games: int, slack_pct: float) -> int:
    """Compute the number of EXTRA games to queue beyond
    ``target_games`` for straggler tolerance.

    Returns ``ceil(target_games * slack_pct)``. Negative slack_pct
    is clamped to 0 (no over-queue, legacy behavior).
    """
    import math
    if slack_pct is None or slack_pct <= 0:
        return 0
    return int(math.ceil(target_games * slack_pct))


def _self_play(  # pragma: no cover
    *,
    weights_path: Path,
    n_games: int,
    samples: int,
    nodes: int,
    seed: int,
    n_workers: int | None,
    output: Path,
    per_game_timeout_sec: float | None = None,
    hidden_sizes: tuple[int, ...] | None = None,
    architecture: str | None = None,
    policy_hidden_sizes: tuple[int, ...] | None = None,
    progress_path: Path | None = None,
    run_id: str | None = None,
    deck_pool: list[Deck] | None = None,
    replay_config: ReplayCaptureConfig | None = None,
    hang_dump_dir: str | None = None,
    heuristics: HeuristicsConfig | None = None,
    cycle_id: str | None = None,
    stop_check: "callable | None" = None,
    action_encoding_version: str = "positional_v1",
    write_structured_tokens: bool = False,
    structured_scalar_schema_version: int | None = None,
    root_dirichlet_alpha: float | None = None,
    root_dirichlet_eps: float = 0.25,
    move_temperature: float = 1.0,
    temperature_cutoff: int = 30,
    evaluator: str = "nn",
) -> tuple[int, int, float]:
    """Run one self-play data-gen pass. Returns (n_samples,
    n_policy_rows, elapsed_seconds). Both seats use the same
    config — that's what makes it self-play.

    ``evaluator`` picks what drives the PIMC search on both seats:

    - ``"nn"`` (default): the NN-driven PUCT config built from
      ``weights_path`` — the historical AlphaZero path.
    - ``"heuristic"``: ``MtgHeuristicEvaluator`` under the same PIMC
      search, ``weights_path`` unused. Generates the warm-start
      corpus (see ``mtg_bot.training.self_play_evaluator``). The
      emitted npz is byte-compatible with the NN path's: same
      columns, same action encoding, same structured-token schema,
      same ``game_ids``.

    Exploration knobs (self-play ONLY — ``_eval_match`` never sets
    them, so the promotion gate stays deterministic-argmax):
    ``root_dirichlet_alpha`` / ``root_dirichlet_eps`` add Dirichlet
    noise to every PIMC sub-search root (``None`` = off);
    ``move_temperature`` / ``temperature_cutoff`` sample the played
    move from the visit distribution for the first N non-forced
    decisions of each game. The saved npz also carries ``game_ids``
    so training splits train/val by game.

    ``hidden_sizes`` must match the saved weights' trunk shape,
    or every worker crashes at NNEvaluator load with state_dict
    size mismatch. Caller threads from bootstrap metadata.

    When ``progress_path`` is set, builds a ``DataGenProgress``
    so the training UI shows live games-done / win-rate / ETA.
    Useful in long iterate runs where the self-play phase
    consumes 80%+ of wall time and the UI would otherwise be
    silent for ~40 min per iteration.

    ``write_structured_tokens=True`` flips data_gen into the Step 1
    structured-token writer (``docs/NN_REPRESENTATION_ROADMAP.md``):
    each row's per-state token stream is captured alongside the
    legacy 51-dim featurizer output and packed into four extra
    arrays on the npz. Iterate sets this when the bootstrap
    architecture is ``"structured"``; flat-feature bootstraps leave
    it at False so the existing v* lineage stays bit-identical.
    """
    from mtg_bot.training.self_play_evaluator import (
        EVALUATOR_HEURISTIC, build_heuristic_pimc_config,
        normalize_evaluator,
    )
    evaluator = normalize_evaluator(evaluator)
    cfg: NNPimcConfig | PimcConfig
    if evaluator == EVALUATOR_HEURISTIC:
        cfg = build_heuristic_pimc_config(
            nodes=nodes,
            samples=samples,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_dirichlet_eps=root_dirichlet_eps,
        )
    else:
        cfg = NNPimcConfig(
            weights_path=str(weights_path),
            samples=samples,
            inner_nodes=nodes,
            # Leaf batch 1 on the CPU path: batching at 32 cost ~8
            # points of strength at nodes 100-200 for ~3 % wall (see
            # mtg_bot.search.leaf_batch).
            batch_size=SEARCH_LEAF_BATCH_SIZE,
            with_policy=True,
            puct=True,
            hidden_sizes=hidden_sizes,
            architecture=architecture,
            policy_hidden_sizes=policy_hidden_sizes,
            action_encoding_version=action_encoding_version,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_dirichlet_eps=root_dirichlet_eps,
        )
    progress: DataGenProgress | None = None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        worker_count = (
            n_workers if n_workers is not None
            else max(1, (os.cpu_count() or 1) - 2)
        )
        progress = DataGenProgress(
            progress_path=progress_path,
            run_id=run_id or progress_path.stem,
            n_games_total=n_games,
            n_workers=worker_count,
            command=f"iterate_alphazero self-play ({evaluator})",
            notes=f"iterate_alphazero self-play ({evaluator})",
            replays_dir=(
                replay_config.save_dir if replay_config is not None
                else None
            ),
            cycle_id=cycle_id,
        )
    started = time.perf_counter()
    try:
        gen_result = generate_training_data_parallel(
            p0_factory=cfg.build,
            p1_factory=cfg.build,
            n_games=n_games,
            base_seed=seed,
            n_workers=n_workers,
            with_policy=True,
            per_game_timeout_sec=per_game_timeout_sec,
            progress=progress,
            deck_pool=deck_pool,
            replay_config=replay_config,
            hang_dump_dir=hang_dump_dir,
            heuristics=heuristics,
            stop_check=stop_check,
            action_encoding_version=action_encoding_version,
            structured_tokens=write_structured_tokens,
            structured_scalar_schema_version=(
                structured_scalar_schema_version
            ),
            with_game_ids=True,
            move_temperature=move_temperature,
            temperature_cutoff=temperature_cutoff,
        )
    except BaseException:
        if progress is not None:
            progress.finalize(status="failed")
        raise
    # generate_training_data_parallel finalizes the progress tracker
    # itself with status="stopped" when the sentinel fires; only
    # finalize as completed on a clean run-to-end.
    if progress is not None and (
        stop_check is None or not stop_check()
    ):
        progress.finalize(status="completed")
    elapsed = time.perf_counter() - started
    if write_structured_tokens:
        # data_gen returns a 9-tuple when structured_tokens is on
        # (8 arrays + the trailing game_ids from with_game_ids=True).
        (
            features, outcomes, policy_targets, action_masks,
            structured_token_type, structured_token_emb_ids,
            structured_token_scalars, structured_attention_mask,
            game_ids,
        ) = gen_result
        save_dataset(
            features, outcomes, output,
            policy_targets=policy_targets, action_masks=action_masks,
            structured_token_type=structured_token_type,
            structured_token_emb_ids=structured_token_emb_ids,
            structured_token_scalars=structured_token_scalars,
            structured_attention_mask=structured_attention_mask,
            game_ids=game_ids,
            action_encoding_version=action_encoding_version,
        )
    else:
        (
            features, outcomes, policy_targets, action_masks, game_ids,
        ) = gen_result
        save_dataset(
            features, outcomes, output,
            policy_targets=policy_targets, action_masks=action_masks,
            game_ids=game_ids,
            action_encoding_version=action_encoding_version,
        )
    n_samples = features.shape[0]
    n_policy = int((policy_targets.sum(axis=-1) > 0.5).sum())
    return n_samples, n_policy, elapsed


def _aggregate_self_play_npzs(
    stage_id: str, iterate_data_dir: Path,
) -> tuple[int, int]:
    """Sum ``n_samples`` + ``n_policy_rows`` across every npz the
    coordinator wrote for ``stage_id``.

    The v2 coordinator's ``/v2/report_result`` handler writes per-game
    npzs to ``iterate_data_dir/<stage_id>_<shard_label>_<safe_claim>.npz``
    (see ``cli/runs/v2_endpoints.py``). We glob the prefix and
    aggregate.
    """
    import numpy as np
    n_samples = 0
    n_policy = 0
    pattern = f"{stage_id}_*.npz"
    for p in sorted(iterate_data_dir.glob(pattern)):
        try:
            with np.load(p) as data:
                if "features" in data.files:
                    n_samples += int(data["features"].shape[0])
                if "policy_targets" in data.files:
                    n_policy += int(
                        (data["policy_targets"].sum(axis=-1) > 0.5).sum()
                    )
        except (OSError, ValueError):
            # Half-written or stale file — skip. The coordinator
            # writes files atomically (NamedTemporaryFile + rename),
            # so this is defensive only.
            continue
    return n_samples, n_policy


def self_play_yield_error(
    *,
    stage_id: str,
    data_dir: Path,
    n_samples: int,
    upload_failed: int = 0,
    n_games_done: Optional[int] = None,
) -> Optional[str]:
    """Return an actionable error message if a self-play stage did
    not actually produce usable training data, else ``None``.

    Two failure modes, both of which used to be SILENT:

    1. ``n_samples == 0`` -- no ``.npz`` reached ``--data-dir``. The
       old code went straight into ``n_policy / n_samples`` and died
       with a bare ``ZeroDivisionError`` several hundred games of
       compute too late (2026-09-23).
    2. ``upload_failed > 0`` -- the coordinator recorded games whose
       artifact it could not fetch. Those games STILL count toward
       the stage's ``n_games_done`` and ``n_samples`` aggregate, so
       nothing upstream noticed the data loss.

    Pure: takes counters, returns a string. The callers do the
    exiting so this is trivially unit-testable.
    """
    if n_samples > 0 and upload_failed <= 0:
        return None
    lines: list[str] = []
    if n_samples <= 0:
        lines.append(
            f"[iterate] FATAL: self-play stage {stage_id!r} produced 0 "
            f"training samples."
        )
        lines.append(
            f"  No files matching {stage_id}_*.npz were found in "
            f"--data-dir {data_dir}."
        )
    else:
        lines.append(
            f"[iterate] FATAL: self-play stage {stage_id!r} recorded "
            f"{upload_failed} upload_failed result(s) -- those games' "
            f"artifacts never reached the coordinator."
        )
        lines.append(
            f"  Only {n_samples} samples landed in --data-dir "
            f"{data_dir}; the training set is incomplete."
        )
    if n_games_done is not None:
        lines.append(
            f"  Coordinator stage view: n_games_done={n_games_done}, "
            f"upload_failed={upload_failed}."
        )
    lines.extend([
        "  The two usual causes:",
        "    1. A worker is running --artifact-storage-uri null:// "
        "(the worker-daemon default).",
        "       Its games complete and count toward the stage "
        "totals, but every .npz is",
        "       dropped on the floor. Restart that worker with "
        "--artifact-storage-uri",
        "       file:///<shared-dir> (coordinator-local) or "
        "http://<coordinator>:5099.",
        "    2. The coordinator's --iterate-data-dir points somewhere "
        "other than this",
        f"       --data-dir ({data_dir}), so the artifacts were "
        "rehydrated where iterate",
        "       is not looking. The two MUST match.",
        "  Refusing to train on missing self-play data.",
    ])
    return "\n".join(lines)


def _self_play_distributed(  # pragma: no cover
    *,
    worker_urls: list[str],
    coordinator_url: str,
    nn_version_label: str,
    samples: int,
    nodes: int,
    seed: int,
    per_game_timeout_sec: float | None,
    capture_every: int,
    total_games: int,
    deck_pool_size: int,
    run_id_prefix: str,
    iterate_data_dir: Path,
    poll_interval_s: float = _rest_orch.DEFAULT_POLL_INTERVAL_S,
    stop_check: Optional[Callable[[], bool]] = None,
    claim_ttl_seconds: int = 1200,
    root_dirichlet_alpha: float | None = None,
    root_dirichlet_eps: float = 0.25,
    move_temperature: float = 1.0,
    temperature_cutoff: int = 30,
    evaluator: str = "nn",
    action_encoding_version: str = "positional_v1",
    structured_tokens: bool = False,
    structured_scalar_schema_version: int | None = None,
) -> tuple[int, int, float]:
    """Distributed self-play via the v2 REST contract.

    Orchestration:
    1. Build the per-game stage spec (1000 games with deck_indices +
       seed).
    2. POST ``/v2/stage/create`` on the coordinator.
    3. POST ``/v2/start_self_play`` to every worker URL in parallel.
    4. Poll ``GET /v2/stage/<id>`` every ``poll_interval_s`` until
       ``status == "complete"``.
    5. Glob the coordinator's ``iterate_data_dir`` for the npzs the
       workers streamed back via ``/v2/report_result``.

    Returns ``(n_samples, n_policy_rows, elapsed_seconds)``.

    Worker-owned config (deck_pool_path, n_workers, weights cache
    root, NN architecture / heuristics / shm / structured-token
    flags) is decided locally on each worker daemon via its CLI flags
    — NOT threaded across the wire. The cluster's bootstrap doc
    (deploy/README.md) covers per-machine setup.

    The one exception is ``evaluator="heuristic"``: a warm-start
    corpus stage has no weights version, so the worker has no
    ``metadata.json`` to discover the action encoding / structured
    token schema from and those three fields DO travel on the spec.
    ``nn_version_label`` is ignored in that mode and the spec's
    ``weights`` block is emptied so a pre-``evaluator`` worker fails
    loudly instead of silently running the NN.
    """
    started = time.perf_counter()
    spec = _rest_orch.make_self_play_stage_spec(
        stage_id=run_id_prefix,
        n_games=total_games,
        base_seed=seed,
        deck_pool_size=deck_pool_size,
        weights_version=nn_version_label,
        nodes=nodes,
        samples=samples,
        per_game_timeout_sec=per_game_timeout_sec or 600.0,
        capture_every=capture_every,
        claim_ttl_seconds=claim_ttl_seconds,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_eps=root_dirichlet_eps,
        move_temperature=move_temperature,
        temperature_cutoff=temperature_cutoff,
        evaluator=evaluator,
        action_encoding_version=action_encoding_version,
        structured_tokens=structured_tokens,
        structured_scalar_schema_version=structured_scalar_schema_version,
    )
    # Stream stage progress into runs.db so the training UI's per-
    # machine cards + "Samples" / "Workers" / "Games/hr" / "ETA" stats
    # populate live. The writer no-ops cleanly when neither
    # MTG_BOT_RUN_DB_PATH nor MTG_BOT_RUN_DB_URL is set.
    progress_writer = _rest_orch.make_runs_db_progress_writer(
        run_id_prefix, started_perf=started,
    )
    stage_result = _rest_orch.run_stage(
        coordinator_url=coordinator_url,
        worker_urls=worker_urls,
        spec=spec,
        poll_interval_s=poll_interval_s,
        stop_check=stop_check,
        progress_writer=progress_writer,
    )
    elapsed = time.perf_counter() - started
    n_samples, n_policy = _aggregate_self_play_npzs(
        stage_id=run_id_prefix,
        iterate_data_dir=iterate_data_dir,
    )
    # Stage completion is NOT success: a stage can report every game
    # done while every artifact was dropped (worker on null://) or
    # unfetchable (upload_failed rows). Fail here rather than a few
    # lines later on ZeroDivisionError -- or, worse, silently train
    # on a partial dataset.
    problem = self_play_yield_error(
        stage_id=run_id_prefix,
        data_dir=iterate_data_dir,
        n_samples=n_samples,
        upload_failed=stage_result.upload_failed,
        n_games_done=stage_result.n_games_done,
    )
    if problem is not None:
        sys.exit(problem)
    return n_samples, n_policy, elapsed


def _eval_match_distributed(  # pragma: no cover
    *,
    worker_urls: list[str],
    coordinator_url: str,
    new_version_label: str,
    old_version_label: str,
    n_games: int,
    samples: int,
    nodes: int,
    seed: int,
    per_game_timeout_sec: float | None,
    capture_every: int,
    deck_pool_size: int,
    run_id_prefix: str,
    poll_interval_s: float = _rest_orch.DEFAULT_POLL_INTERVAL_S,
    stop_check: Optional[Callable[[], bool]] = None,
    claim_ttl_seconds: int = 1200,
) -> tuple[int, int, int, float]:
    """Distributed eval-gate match via the v2 REST contract.

    Mirrors ``_eval_match``'s return contract
    ``(new_wins, old_wins, draws, elapsed_seconds)`` so iterate's
    caller doesn't change shape.

    The eval watcher's ``POST /v2/stage/<id>/close`` mechanism stays
    intact for Wilson-CI early-stop — this function just polls the
    stage to ``complete`` and reads the final wins / draws counters
    off ``GET /v2/stage/<id>``.
    """
    started = time.perf_counter()
    spec = _rest_orch.make_eval_stage_spec(
        stage_id=run_id_prefix,
        n_games=n_games,
        base_seed=seed,
        deck_pool_size=deck_pool_size,
        new_version=new_version_label,
        old_version=old_version_label,
        nodes=nodes,
        samples=samples,
        per_game_timeout_sec=per_game_timeout_sec or 600.0,
        capture_every=capture_every,
        claim_ttl_seconds=claim_ttl_seconds,
    )
    progress_writer = _rest_orch.make_runs_db_progress_writer(
        run_id_prefix, started_perf=started,
    )
    result = _rest_orch.run_stage(
        coordinator_url=coordinator_url,
        worker_urls=worker_urls,
        spec=spec,
        poll_interval_s=poll_interval_s,
        stop_check=stop_check,
        progress_writer=progress_writer,
    )
    elapsed = time.perf_counter() - started
    return (
        int(result.wins_new or 0),
        int(result.wins_old or 0),
        int(result.draws or 0),
        elapsed,
    )


TRAIN_GATE_REJECT_EXIT_CODE = 3
"""Exit code ``mtg_bot train`` uses for "held-out value gate rejected
this model". Mirrors ``mtg_bot.cli.train._GATE_REJECT_EXIT_CODE``;
kept as a literal here so importing iterate doesn't drag in the whole
training stack (the value is pinned by a test)."""


class TrainRejected(RuntimeError):
    """The train subprocess ran to completion but its held-out
    value-quality gate refused to register the model.

    Distinct from a generic training failure: the cycle keeps
    ``current_best``, is marked failed with the reason, and the loop
    CONTINUES to the next cycle rather than aborting the chain."""


def _train_next(
    *,
    data_dir: Path,
    registry_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    seed: int,
    notes: str,
    window_recent: int | None,
    window_recent_files: int | None = None,
    hidden_sizes: tuple[int, ...] | None = None,
    progress_path: Path | None = None,
    progress_dir: Path | None = None,
    policy_loss_weight: float = 1.0,
    one_legal_keep_frac: float = 1.0,
    architecture: str = "shared",
    policy_hidden_sizes: tuple[int, ...] | None = None,
    early_stop_patience: int | None = None,
    action_encoding_version: str = "positional_v1",
    structured_emb_dim: int | None = None,
    structured_n_heads: int | None = None,
    structured_n_layers: int | None = None,
    structured_hidden_dim: int | None = None,
    structured_action_history_n: int | None = None,
    structured_scalar_schema_version: int | None = None,
    num_workers: int = 0,
    shuffle_mode: str | None = None,
    shuffle_block_rows: int | None = None,
    dropout: float | None = None,
    weight_decay: float | None = None,
    value_head_input: str | None = None,
    value_target: str | None = None,
    keep_last_epoch: bool = False,
    min_holdout_value_quality: float | None = None,
) -> str:
    """Run train_nn.py via subprocess so its progress JSON / hub
    UI / registry plumbing all behave the same as a manual train.
    Returns the registered version ID (e.g. ``v5``).

    When ``hidden_sizes`` is set, passes ``--hidden-sizes
    "h1,h2,...,hn"`` to train_nn.py so the new model trains at
    the same trunk capacity as the bootstrap version. Without
    this, iterations of a non-default-arch bootstrap (e.g. v6's
    1024-1024-512 trunk) would silently regress to the
    registry-baseline 128-128-64; the new model would have ~5%
    of the bootstrap's parameters and the acceptance gate would
    always reject it.

    ``policy_loss_weight`` scales the policy CE in the combined loss
    (default 1.0). Iter_002 v13 training showed the policy head
    plateaus at ~0.157 because ~84% of supervised rows are
    one-legal-action (CE=0 trivially) and value loss (~0.5 at
    epoch 1) dominates the gradient ~3:1. Bumping this to 3-10
    rebalances toward the policy head's branchy-row signal.

    ``one_legal_keep_frac`` (default 1.0 = no downsample) drops the
    given fraction of one-legal-action supervised rows BEFORE training
    so the policy head's gradient is dominated by branchy rows. Set
    e.g. 0.1 to keep only 10%% of trivial one-legal rows. Orthogonal
    to ``policy_loss_weight`` — both can be combined.

    ``early_stop_patience`` (default ``None`` = run all epochs)
    forwards ``--early-stop-patience N`` to train_nn.py so training
    breaks once val loss has stopped improving for N consecutive
    epochs. Iterate's CLI defaults this to 5 to ship the
    speedup-by-default; the helper here keeps ``None`` so a direct
    caller (tests / one-offs) opts in explicitly.
    """
    import subprocess

    cmd = [
        sys.executable, "-u",
        str(_SCRIPTS_DIR / "train_nn.py"),
        "--data-dir", str(data_dir),
        "--registry-root", str(registry_root),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(learning_rate),
        "--val-split", str(val_split),
        "--seed", str(seed),
        "--notes", notes,
        "--policy-loss-weight", str(policy_loss_weight),
        "--one-legal-keep-frac", str(one_legal_keep_frac),
        "--architecture", architecture,
        # Inherit the bootstrap version's action-encoding convention so
        # the new model is trained against the same slot mapping its
        # self-play data was generated under. Without this, a v17 trained
        # via iterate from a v2 bootstrap would silently regress to v1
        # encoding because train_nn defaults to positional_v1.
        "--action-encoding-version", action_encoding_version,
    ]
    if hidden_sizes is not None:
        cmd.extend(["--hidden-sizes", ",".join(str(s) for s in hidden_sizes)])
    if policy_hidden_sizes is not None:
        cmd.extend([
            "--policy-hidden-sizes",
            ",".join(str(s) for s in policy_hidden_sizes),
        ])
    if early_stop_patience is not None:
        cmd.extend([
            "--early-stop-patience", str(early_stop_patience),
        ])
    if num_workers > 0:
        cmd.extend(["--num-workers", str(num_workers)])
    # Train-split ordering. Blank = inherit train_nn.py's own default
    # ("auto": block-shuffle once the flat store outgrows the page
    # cache).
    if shuffle_mode is not None:
        cmd.extend(["--shuffle-mode", str(shuffle_mode)])
    if shuffle_block_rows is not None:
        cmd.extend(["--shuffle-block-rows", str(shuffle_block_rows)])
    # Structured-architecture hyperparams (Slice G). Only emitted when
    # the architecture is "structured" — the flat-feature trainer
    # ignores them, and emitting them unconditionally would surface as
    # an argparse "unrecognized arguments" error on the structured
    # path that hasn't grown the flag yet. Defaults are picked at the
    # iterate CLI level so each emitted flag carries an explicit value
    # (matching :class:`StructuredModelConfig`'s defaults — see
    # ``--emb-dim`` / ``--n-heads`` / ... below).
    if architecture == "structured":
        if structured_emb_dim is not None:
            cmd.extend(["--emb-dim", str(structured_emb_dim)])
        if structured_n_heads is not None:
            cmd.extend(["--n-heads", str(structured_n_heads)])
        if structured_n_layers is not None:
            cmd.extend(["--n-layers", str(structured_n_layers)])
        if structured_hidden_dim is not None:
            cmd.extend(["--hidden-dim", str(structured_hidden_dim)])
        if structured_action_history_n is not None:
            cmd.extend([
                "--action-history-n", str(structured_action_history_n),
            ])
        if structured_scalar_schema_version is not None:
            cmd.extend([
                "--scalar-schema-version",
                str(structured_scalar_schema_version),
            ])
        # Regularisation + held-out value gate. Structured-only on the
        # train side too (the flat trainer rejects them), so they are
        # emitted inside this branch.
        if dropout is not None:
            cmd.extend(["--dropout", str(dropout)])
        if weight_decay is not None:
            cmd.extend(["--weight-decay", str(weight_decay)])
        if value_head_input is not None:
            cmd.extend(["--value-head-input", str(value_head_input)])
        if value_target is not None:
            cmd.extend(["--value-target", str(value_target)])
        if keep_last_epoch:
            cmd.append("--keep-last-epoch")
        if min_holdout_value_quality is not None:
            cmd.extend([
                "--min-holdout-value-quality",
                str(min_holdout_value_quality),
            ])
    if progress_dir is not None:
        # Thread the lineage's progress dir through so train_nn's OWN
        # default progress path lands there too. Without it the
        # trainer hardcoded ``training_progress/`` and an s2-lineage
        # cycle printed ``# Progress JSON: training_progress        # s2_iter_005_train_v6.json`` while every other artifact of
        # that cycle lived under ``training_progress_s2/``.
        cmd.extend(["--progress-dir", str(progress_dir)])
    if progress_path is not None:
        # Override train_nn.py's default ``training_progress/<peeked
        # _version>.json`` so the iterate-loop's training cards have
        # iter-N-tagged filenames (matches the self-play / eval cards
        # so all of one iteration's UI artifacts are visually grouped).
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--progress-path", str(progress_path)])
    # Pass --window-recent through to train_nn.py so the loader
    # caps its file set to the N most-recent .npz files in
    # ``data_dir`` rather than the entire accumulated history.
    # Once iterate_data/ holds hundreds of shards, training the
    # full dir is both slow and biased toward early-iteration
    # data; capping to the latest N keeps wall-time bounded and
    # weights training toward fresh self-play.
    if window_recent is not None:
        cmd.extend(["--window-recent", str(window_recent)])
    if window_recent_files is not None:
        cmd.extend(["--window-recent-files", str(window_recent_files)])

    print(f"\n[iterate] Training: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode == TRAIN_GATE_REJECT_EXIT_CODE:
        # Training finished but the held-out value gate refused to
        # register the model. That's a verdict, not a crash — a
        # dedicated exception so the cycle loop can keep current_best
        # and move on to the next cycle instead of tearing the whole
        # chain down.
        raise TrainRejected(
            "train_nn.py rejected the trained model: its held-out "
            "value MSE failed the --min-holdout-value-quality gate "
            "(no version was registered). See the training run's "
            "stderr / progress JSON for the measured metrics."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"train_nn.py exited with code {result.returncode}"
        )

    # Pick up the new version: the highest-numbered version in the
    # registry after training. train_nn.py always registers as the
    # next sequential ID, so this is a reliable post-hoc lookup.
    registry = Registry(registry_root)
    versions = registry.list_versions()
    if not versions:
        raise RuntimeError(
            f"train_nn.py succeeded but registry has no versions"
        )
    return versions[-1]


def _eval_match(  # pragma: no cover
    *,
    new_weights: Path,
    old_weights: Path | None,
    n_games: int,
    samples: int,
    nodes: int,
    n_workers: int | None,
    seed: int,
    per_game_timeout_sec: float | None = None,
    hidden_sizes: tuple[int, ...] | None = None,
    new_architecture: str | None = None,
    old_architecture: str | None = None,
    policy_hidden_sizes: tuple[int, ...] | None = None,
    progress_path: Path | None = None,
    run_id: str | None = None,
    opponent_label: str = "previous",
    deck_pool: list[Deck] | None = None,
    replay_config: ReplayCaptureConfig | None = None,
    hang_dump_dir: str | None = None,
    cycle_id: str | None = None,
    stop_check: "callable | None" = None,
    action_encoding_version: str = "positional_v1",
) -> tuple[int, int, int, float]:
    """Head-to-head match: new (seat 0) vs old (seat 1). Returns
    (new_wins, old_wins, draws, elapsed_seconds).

    ``old_weights=None`` falls back to a heuristic opponent (useful
    for the very first iteration where there's no "old NN" to
    compare against, OR for sanity-check eval against the
    heuristic baseline).

    ``hidden_sizes`` must match the saved weights' trunk shape for
    BOTH new and old — they're trained at the same capacity per the
    iterate loop's contract (see main()'s ``iter_hidden_sizes``).
    Without it the worker NNEvaluator load fails with state_dict
    size mismatch and every game crashes."""
    # Both NN seats use the production leaf batch size (1 on CPU --
    # see mtg_bot.search.leaf_batch; batch 32 was an ~8-point search
    # handicap at these node budgets). The heuristic seat is batch 1
    # as well, so new, old and heuristic search at equal quality.
    new_cfg = NNPimcConfig(
        weights_path=str(new_weights),
        samples=samples,
        inner_nodes=nodes,
        batch_size=SEARCH_LEAF_BATCH_SIZE,
        with_policy=True,
        puct=True,
        hidden_sizes=hidden_sizes,
        architecture=new_architecture,
        policy_hidden_sizes=policy_hidden_sizes,
        action_encoding_version=action_encoding_version,
    )
    if old_weights is not None:
        old_cfg = NNPimcConfig(
            weights_path=str(old_weights),
            samples=samples,
            inner_nodes=nodes,
            batch_size=SEARCH_LEAF_BATCH_SIZE,
            with_policy=True,
            puct=True,
            hidden_sizes=hidden_sizes,
            architecture=old_architecture,
            policy_hidden_sizes=policy_hidden_sizes,
            action_encoding_version=action_encoding_version,
        )
        p1_build = old_cfg.build
    else:
        p1_build = PimcConfig(
            samples=samples,
            inner_nodes=nodes,
            heuristic=HeuristicConfig(),
            batch_size=1,
        ).build

    progress: EvalProgress | None = None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        worker_count = (
            n_workers if n_workers is not None
            else max(1, (os.cpu_count() or 1) - 2)
        )
        progress = EvalProgress(
            progress_path=progress_path,
            run_id=run_id or progress_path.stem,
            n_games_total=n_games,
            n_workers=worker_count,
            opponent_label=opponent_label,
            command="iterate_alphazero acceptance-gate eval",
            notes=f"iterate eval vs {opponent_label}",
            replays_dir=(
                replay_config.save_dir if replay_config is not None
                else None
            ),
            cycle_id=cycle_id,
        )
    started = time.perf_counter()
    stats = run_match_parallel(
        p0_factory=new_cfg.build,
        p1_factory=p1_build,
        n_games=n_games,
        base_seed=seed,
        n_workers=n_workers,
        per_game_timeout_sec=per_game_timeout_sec,
        progress=progress,
        deck_pool=deck_pool,
        replay_config=replay_config,
        hang_dump_dir=hang_dump_dir,
        alternate_seats=True,
        stop_check=stop_check,
    )
    elapsed = time.perf_counter() - started
    return (
        int(stats.wins[0]),
        int(stats.wins[1]),
        int(stats.draws),
        elapsed,
    )


def main(argv=None) -> None:  # pragma: no cover
    # Argparse + orchestration body; previously script-level code
    # under ``scripts/iterate_alphazero.py``. Never had unit-test
    # coverage (testing it requires a real registry + 5+ minute
    # training runs); helpers like ``_reconcile_iterate_state`` and
    # ``_train_next`` are unit-tested directly. Leaving this body
    # un-pragma'd would drop coverage by ~5pp without surfacing real
    # risk — the body is mostly argument plumbing.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-version", required=True,
        help="Registered version to start from (e.g. v4).",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=5,
        help="How many iterations to run (default 5).",
    )
    parser.add_argument(
        "--games-per-iteration", type=int, default=500,
        help="Self-play games per iteration (default 500).",
    )
    parser.add_argument(
        "--samples", type=int, default=4,
        help="PIMC samples K (default 4).",
    )
    parser.add_argument(
        "--nodes", type=int, default=200,
        help="MCTS max_nodes per sub-search (default 200).",
    )
    parser.add_argument(
        "--self-play-evaluator",
        choices=list(_SELF_PLAY_EVALUATORS), default="nn",
        help=(
            "Which evaluator drives the SELF-PLAY search on both "
            "seats (default nn). 'heuristic' runs the same PIMC "
            "search over MtgHeuristicEvaluator with no weights at "
            "all -- the warm-start corpus path. The emitted npz is "
            "byte-compatible with the nn path's (same columns, "
            "action encoding, structured-token schema, game_ids), "
            "so `mtg_bot train` consumes it unchanged. The eval "
            "gate is ALWAYS nn-vs-nn; this flag never touches it."
        ),
    )
    parser.add_argument(
        "--corpus-only", action="store_true",
        help=(
            "Run ONLY the self-play stage of each iteration: no "
            "train, no eval, no promotion, and iterate_state.json is "
            "left untouched. Use with --self-play-evaluator "
            "heuristic to build a warm-start corpus. "
            "--n-iterations N produces N self-play shards with "
            "distinct seeds / stage ids."
        ),
    )
    parser.add_argument(
        "--corpus-batch", type=int, default=None,
        help=(
            "Corpus-only mode: the 1-based iteration index to label "
            "this batch with, instead of iterate_state.json's "
            "iterations_completed + 1. REQUIRED when generating a "
            "corpus across SEPARATE process invocations (e.g. the "
            "distributed launcher, which always spawns iterate with "
            "--n-iterations 1): --corpus-only never advances "
            "iterations_completed, so every invocation would "
            "otherwise derive the same stage_id, the coordinator "
            "would treat it as the already-drained stage and answer "
            "204 to every claim, and the run would silently produce "
            "no games. Within ONE invocation, --n-iterations N "
            "already offsets each pass. Requires --corpus-only."
        ),
    )
    parser.add_argument(
        "--root-dirichlet-alpha", type=float, default=0.5,
        help=(
            "Dirichlet alpha for AlphaZero root exploration noise in "
            "the SELF-PLAY PUCT sub-searches (default 0.5). Values "
            "<= 0 disable the noise. Eval-gate matches never use it."
        ),
    )
    parser.add_argument(
        "--root-dirichlet-eps", type=float, default=0.25,
        help=(
            "Mixing weight for the root Dirichlet noise: prior = "
            "(1-eps)*prior + eps*noise (default 0.25)."
        ),
    )
    parser.add_argument(
        "--move-temperature", type=float, default=1.0,
        help=(
            "Self-play move temperature T: the played move is sampled "
            "proportional to visits**(1/T) for the first "
            "--temperature-cutoff non-forced decisions of each game "
            "(default 1.0; <= 0 = always argmax). Policy targets stay "
            "the raw visit distribution. Eval matches stay argmax."
        ),
    )
    parser.add_argument(
        "--temperature-cutoff", type=int, default=30,
        help=(
            "Number of NON-forced decisions (>1 legal option) per "
            "self-play game after which move selection reverts to "
            "argmax (default 30)."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per iteration (default 50).",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=5,
        help=(
            "Stop the per-iteration training loop after this many "
            "consecutive epochs without val-loss improvement. Default 5 "
            "(conservative — small enough to save substantial wall, "
            "large enough to ride through plateau noise). Empirically "
            "v10/v13 hit best val loss around epoch 23-27 of a 50-epoch "
            "budget; the tail is mostly overfitting. Must be a positive "
            "int. Threaded through to train_nn.py."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size (default 256).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Adam learning rate (default 1e-3).",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Train/val split fraction (default 0.1).",
    )
    parser.add_argument(
        "--architecture",
        choices=["shared", "separate", "structured"],
        default="shared",
        help=(
            "Dual-head model class for trained iterations: 'shared' "
            "(default, back-compat) keeps the existing single-trunk "
            "_ValuePolicyMLP. 'separate' switches to the parallel "
            "value_trunk + policy_trunk _ValuePolicyMLPSeparate, "
            "fixing the value-vs-policy gradient saturation observed "
            "at the shared trunk's bottleneck. 'structured' switches "
            "to the Step 1 transformer trunk (per-object tokens + "
            "printing embeddings + last-N action tokens — see "
            "docs/NN_REPRESENTATION_ROADMAP.md); self-play also "
            "emits structured-token columns so the next train phase "
            "has the data it needs. The three architectures are NOT "
            "cross-loadable — switching forms a fresh v-chain. The "
            "choice is passed to train_nn.py and recorded in "
            "metadata.json so self-play / eval workers dispatch the "
            "right NNEvaluator class when loading weights."
        ),
    )
    parser.add_argument(
        "--emb-dim", type=int, default=128,
        help=(
            "Structured-trunk only (--architecture=structured): "
            "per-token embedding dimension. Default 128 matches "
            "StructuredModelConfig.emb_dim. Threaded through to "
            "train_nn.py as ``--emb-dim`` only when the iter's "
            "architecture is 'structured'; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--n-heads", type=int, default=4,
        help=(
            "Structured-trunk only: transformer attention heads. "
            "Default 4 matches StructuredModelConfig.n_heads. "
            "``emb_dim`` must divide evenly by this — torch's "
            "MultiheadAttention enforces it at construction time."
        ),
    )
    parser.add_argument(
        "--n-layers", type=int, default=2,
        help=(
            "Structured-trunk only: transformer encoder-layer count. "
            "Default 2 matches StructuredModelConfig.n_layers."
        ),
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help=(
            "Structured-trunk only: MLP hidden dimension inside the "
            "transformer encoder layers. Default 256 matches "
            "StructuredModelConfig.hidden_dim."
        ),
    )
    parser.add_argument(
        "--action-history-n", type=int, default=32,
        help=(
            "Structured-trunk only: last-N action-history window the "
            "state tokenizer emits per row. Default 32 matches "
            "StructuredModelConfig.action_history_n and the §3 "
            "default in the NN-representation roadmap."
        ),
    )
    parser.add_argument(
        "--scalar-schema-version", type=int, default=None,
        help=(
            "Structured-trunk only: token_scalars row layout (1 = "
            "legacy 3-slot row; 2 = adds board / life state) used "
            "BOTH by self-play data-gen (structured-token columns) "
            "and by the train phase's --scalar-schema-version. "
            "Default None = the tokenizer / StructuredModelConfig "
            "default (currently 2). Independent of the bootstrap's "
            "schema: the bootstrap is only used for self-play "
            "inference (its own metadata sizes its input), while the "
            "next version trains from scratch on this schema."
        ),
    )
    # --- Regularisation + held-out value gate (structured only) -----
    # Forwarded verbatim to ``mtg_bot train``; see that CLI for the
    # defaults and the rationale (the s2 lineage's value heads
    # memorised game identity). Left at None here so iterate does not
    # duplicate the trainer's default values -- omitting the flag lets
    # the trainer's own default win.
    parser.add_argument(
        "--dropout", type=float, default=None,
        help=(
            "Structured-trunk only: dropout for the encoder layers, "
            "cross-attention and head MLPs. Blank defers to the "
            "trainer default (0.1)."
        ),
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help=(
            "Structured-trunk only: AdamW decoupled weight decay. "
            "Blank defers to the trainer default (1e-4)."
        ),
    )
    parser.add_argument(
        "--value-head-input", default=None,
        choices=SUPPORTED_VALUE_HEAD_INPUTS,
        help=(
            "Structured-trunk only: which representation the VALUE "
            "head reads. 'trunk' (default) shares the policy head's "
            "pass and can memorise per-printing card identity; "
            "'trunk_no_printing' hides the printing embeddings from "
            "the value head only. Blank defers to the trainer "
            "default (trunk)."
        ),
    )
    parser.add_argument(
        "--value-target", default=None,
        help=(
            "Structured-trunk only: what the value head regresses -- "
            "'outcome' (+/-1 game result), 'root' (the search root value "
            "each row was recorded with; D6 2026-09-26: 43%% vs heuristic "
            "against 24.5%% for outcome on the same corpus), or "
            "'blend:<w>'. Forwarded to train_nn.py as --value-target. "
            "Blank defers to the trainer default (outcome)."
        ),
    )
    parser.add_argument(
        "--keep-last-epoch", action="store_true",
        help=(
            "Structured-trunk only: save the final epoch's weights "
            "instead of the best-validation epoch's. Off by default "
            "(restore-best)."
        ),
    )
    parser.add_argument(
        "--min-holdout-value-quality", type=float, default=None,
        help=(
            "Structured-trunk only: held-out value-MSE ceiling. A "
            "cycle whose trained value head scores at or above this "
            "on the game-level held-out split is REJECTED -- no "
            "version is registered, the cycle is marked failed with "
            "the reason, current_best is kept, and the NEXT cycle "
            "still runs. Blank defers to the trainer default (1.0 = "
            "the constant-0 predictor); pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--train-num-workers", type=int, default=4,
        help=(
            "Train-phase DataLoader worker process count. Default 4 — "
            "single-process loading (num_workers=0) starves the GPU "
            "at production-scale (~7M samples) because every batch's "
            "__getitem__ + collate runs on the main thread between "
            "GPU forward/backward calls. Set to 0 for smoke tests "
            "where multiprocessing fork is finicky."
        ),
    )
    parser.add_argument(
        "--train-shuffle-mode", choices=("auto", "row", "block"),
        default="auto",
        help=(
            "Train-phase ordering of the train split. 'row' is the "
            "historical uniform shuffle -- every batch is batch_size "
            "random slices into the flat memmap store, i.e. thousands "
            "of 4 KB page faults per batch once the store outgrows "
            "the page cache. 'block' streams shuffled contiguous "
            "blocks of --train-shuffle-block-rows rows instead (same "
            "rows per epoch, different order). 'auto' (default) picks "
            "block once the store exceeds half of available RAM."
        ),
    )
    parser.add_argument(
        "--train-shuffle-block-rows", type=int,
        default=DEFAULT_BLOCK_ROWS,
        help=(
            "Rows per contiguous block under --train-shuffle-mode "
            "block (default %(default)s)."
        ),
    )
    parser.add_argument(
        "--policy-hidden-sizes", default=None,
        help=(
            "Optional comma-separated trunk shape for the policy_trunk "
            "when --architecture=separate. Defaults to mirror the "
            "iterate run's hidden_sizes (inherited from the bootstrap "
            "version's metadata)."
        ),
    )
    parser.add_argument(
        "--policy-loss-weight", type=float, default=1.0,
        help=(
            "Coefficient on the policy-head CE in the combined "
            "training loss (default 1.0). Iter_002 v13 training "
            "showed the policy head plateaus at ~0.157 — ~84%% of "
            "supervised rows are one-legal-action (CE=0 trivially) "
            "and value loss dominates the gradient ~3:1. Bump to "
            "3-10 to rebalance toward branchy-row signal."
        ),
    )
    parser.add_argument(
        "--one-legal-keep-frac", type=float, default=1.0,
        help=(
            "Fraction of one-legal-action supervised rows to keep "
            "before training (default 1.0 = no downsample). The "
            "data-side companion to --policy-loss-weight: rather "
            "than reweighting the loss, drops trivial one-legal "
            "rows so branchy rows with real visit-count signal "
            "dominate the policy gradient. Threaded through to "
            "train_nn.py. Must be in (0.0, 1.0]."
        ),
    )
    parser.add_argument(
        "--eval-games", type=int, default=2000,
        help=(
            "Games per acceptance-gate eval (default 2000). The "
            "default is sized for the early-stop watcher "
            "(``scripts/eval_early_stop_watcher.py``) — when run "
            "alongside iterate, the watcher writes a stop sentinel "
            "as soon as the Wilson 95%% CI on win-rate clears the "
            "acceptance threshold in either direction, so lopsided "
            "results cost only ~50-100 games while close calls get "
            "the full statistical headroom of a 2000-game run "
            "(~+/- 2.2 pts of noise at p=50%%). Without the watcher "
            "the eval-gate phase runs to completion at 2000 games, "
            "which is ~4x the wall of the prior 500-game default — "
            "drop to 500 explicitly if you're running iterate "
            "without the watcher."
        ),
    )
    parser.add_argument(
        "--acceptance-threshold", type=float,
        default=EVAL_GATE_THRESHOLD,
        help=(
            "Win-rate threshold the new NN must beat (vs the "
            "current best, in head-to-head eval) to be promoted "
            "to 'best'. Default tracks "
            "``mtg_bot.training.gate.EVAL_GATE_THRESHOLD`` (the "
            "single source of truth for the gate everywhere in "
            "the project). 0.0 means always promote — pass it "
            "explicitly when bootstrapping or running smokes "
            "where any-improvement isn't the goal. The first "
            "iteration always promotes regardless (no previous "
            "NN to compare against)."
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("iterate_data"),
        help=(
            "Where self-play .npz files land AND the dir passed to "
            "train_nn.py for its glob (default iterate_data/). The "
            "default is dedicated to iterate so train_nn.py won't "
            "co-glob stale debug npzs that happen to be in "
            "training_data/ (today's bug — gen0_smoke.npz had 603 "
            "leaky rows that exploded v7's policy CE). Pass an "
            "explicit path if you want to share a dir; just confirm "
            "no foreign npzs are present."
        ),
    )
    parser.add_argument(
        "--registry-root", type=Path,
        default=Path("trained_weights"),
        help="Registry root (default trained_weights/).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Worker count for self-play and eval (default: "
            "cpu_count() - 2)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base seed (default 0). Per-iteration seeds derive from this.",
    )
    parser.add_argument(
        "--window-recent", type=int, default=None,
        help=(
            "If set, train_nn trains on only the last N ITERATIONS "
            "of data in --data-dir instead of the whole accumulated "
            "history (files are grouped by the <prefix>iter_NNN "
            "label in their name). Default None = train on the full "
            "dir. Useful once iterate_data/ has accumulated hundreds "
            "of shards and per-iteration train wall-time is "
            "dominated by data load. NOTE this counts ITERATIONS, "
            "not files — the distributed path writes one .npz per "
            "GAME. For a raw file cap use --window-recent-files."
        ),
    )
    parser.add_argument(
        "--window-recent-files", type=int, default=None,
        help=(
            "If set, train_nn loads only the N most-recent (by "
            "mtime) .npz files from --data-dir. The pre-2026-09-24 "
            "--window-recent behaviour, kept for callers that need a "
            "hard cap on file count / RAM on the one-npz-per-game "
            "distributed path. Mutually exclusive with "
            "--window-recent."
        ),
    )
    parser.add_argument(
        "--per-game-timeout-self-play-sec", type=float,
        default=None,
        help=(
            "Wall-clock cap per individual self-play game. Default "
            "is ``_common.default_per_game_timeout_sec(--nodes, "
            "--samples)`` (resolved after arg parse): the budget "
            "tracks the search WORK per decision, which is "
            "nodes*samples under PIMC, not nodes alone. The previous "
            "``3 * --nodes`` rule ignored --samples and landed below "
            "the measured mean game wall at nodes=100/samples=2 -- "
            "see the calibration data on "
            "``_common.PER_GAME_TIMEOUT_SEC_PER_SEARCH_UNIT``. Pass "
            "an explicit positive value to override; pass 0 or "
            "negative to disable."
        ),
    )
    parser.add_argument(
        "--per-game-timeout-eval-sec", type=float,
        default=None,
        help=(
            "Wall-clock cap per individual eval-match game. Default "
            "is ``_common.default_per_game_timeout_sec(--nodes, "
            "--samples)`` (resolved after arg parse) -- same "
            "work-relative formula as self-play. Eval is contended, "
            "which inflates per-decision wall further. Timed-out "
            "eval games are DROPPED (excluded from MatchStats, "
            "counted under ``timed_out``), and the games that time "
            "out are systematically the long ones, so a too-tight "
            "cap biases the measured win rate. Pass an "
            "explicit positive value to override; pass 0 or "
            "negative to disable."
        ),
    )
    parser.add_argument(
        "--per-game-timeout-sec", type=float, default=None,
        help=(
            "DEPRECATED back-compat alias. If set, overrides BOTH "
            "--per-game-timeout-self-play-sec and "
            "--per-game-timeout-eval-sec with this single value. "
            "Prefer the split flags so self-play (uncontended) and "
            "eval (14-worker contended) can have different budgets."
        ),
    )
    parser.add_argument(
        "--deck-pool", type=Path, required=True,
        help=(
            "Path to a deck-pool JSON (e.g. data/17lands/eoe_decks."
            "json.gz — the production EoE pool). Every self-play and "
            "eval game samples two distinct real archetypes from the "
            "pool deterministically per (base_seed, game_idx); both "
            "seats use them. Required: there is no synthetic deck "
            "fallback any more — every iteration loop trains on a "
            "real-deck distribution by construction."
        ),
    )
    parser.add_argument(
        "--replays-dir", type=Path, default=Path("replays"),
        help=(
            "Parent directory for captured replay files. Each iter "
            "phase writes its sample of replays to a subdirectory "
            "named after its run_id (e.g. replays/iter_001_self_play_v6/). "
            "The training UI surfaces these via 'View N games' links "
            "on each card, and the main mtg_bot UI loads them in its "
            "scrubber when started with --replays-root pointing here. "
            "Default 'replays' is what mtg_bot.ui defaults to, so the "
            "default-on-default case Just Works."
        ),
    )
    parser.add_argument(
        "--capture-every-self-play", type=int, default=25,
        help=(
            "Capture every Nth self-play game as a viewable replay "
            "(default 25 — at 500 games/iter that's 20 captures per "
            "iteration). Pass 0 to disable replay capture for self-play."
        ),
    )
    parser.add_argument(
        "--capture-every-eval", type=int, default=10,
        help=(
            "Capture every Nth eval game as a viewable replay "
            "(default 10 — at 200 games/iter that's 20 captures per "
            "iteration). Pass 0 to disable replay capture for eval."
        ),
    )
    parser.add_argument(
        "--hang-dump-dir", type=Path, default=Path("debug_hangs"),
        help=(
            "Parent directory for per-game hang diagnostics. When a "
            "self-play or eval game hits the per-game timeout, the "
            "worker writes meta.json + traceback.txt under "
            "<hang-dump-dir>/<run_id>/ before returning the sentinel. "
            "Reproducible inputs (seed + config) plus a "
            "faulthandler.dump_traceback of all threads. Default "
            "'debug_hangs' (gitignored). Pass empty string to disable."
        ),
    )
    parser.add_argument(
        "--shards-config", type=_shards_config_type, default=None,
        help=(
            "Path to ``shards_config.json``. Post-cutover shape is "
            "``{'worker_urls': ['http://host:5198', ...]}`` — a URL "
            "list only. Each worker URL is the daemon's tailnet "
            "endpoint. When set, every iter's self-play + eval stage "
            "runs via the v2 REST contract: the orchestrator POSTs "
            "/v2/stage/create on the coordinator (--queue-url), POSTs "
            "/v2/start_<kind> to each worker, then polls "
            "/v2/stage/<id> until complete. "
            "When the flag is omitted, iterate auto-detects "
            f"``{_DEFAULT_SHARDS_CONFIG_PATH}`` and dispatches "
            "distributed if that file is present, otherwise falls "
            "back to local Pool. Pass ``--shards-config \"\"`` to "
            "explicitly opt out of auto-detection and force local "
            "Pool. Pass ``--shards-config /path/to/file`` to point "
            "at a non-default file (it must exist; missing files "
            "fail loudly rather than silently dropping to local). "
            "See docs/DISTRIBUTED_API_v2.md for the wire contract "
            "and deploy/README.md for the per-machine setup."
        ),
    )
    parser.add_argument(
        "--queue-url", default=None,
        help=(
            "Base URL of the coordinator (master's ``mtg_bot serve`` "
            "on :5099 with the v2 blueprint enabled — typically "
            "``http://<master-tailnet-name>:5099``). Required when "
            "``--shards-config`` is set: the orchestrator POSTs "
            "/v2/stage/create + GETs /v2/stage/<id> here, and each "
            "worker POSTs /v2/claim / /v2/report_result. Name kept as "
            "``--queue-url`` for back-compat with existing scripts; "
            "it's the v2 coordinator URL."
        ),
    )
    parser.add_argument(
        "--queue-slack-pct", type=float, default=0.1,
        help=(
            "DEPRECATED on the REST path — natural-drain + auto-close "
            "handles stage completion without an over-queue+early-stop "
            "tail; the flag is kept for argparse back-compat only and "
            "is ignored. (Single-process local self-play still ignores "
            "it too.)"
        ),
    )
    parser.add_argument(
        "--use-ismcts", action="store_true",
        help=(
            "Use Information Set MCTS instead of PIMC for both self-"
            "play and eval-gate searches. Matched-budget conversion: "
            "ISMCTS's total_iterations = samples * inner_nodes (same "
            "total search work as PIMC's K independent searches of "
            "N nodes each). Off by default — flip on after running "
            "scripts/bench_ismcts_vs_pimc_strength.py to confirm "
            "ISMCTS is at least as strong on the current weights "
            "+ budget. See docs/ISMCTS_MIGRATION.md."
        ),
    )
    parser.add_argument(
        "--use-belief-weighted", action="store_true",
        help=(
            "Wire ISMCTS's per-iteration determinize_fn callback to "
            "BeliefWeightedDeterminizer (Phase 6) — particle-filter "
            "samples conditioned on the opponent's observed action "
            "history. Requires --use-ismcts (belief weighting has no "
            "PIMC analogue). The game loop populates the played-"
            "history log via ISMCTS.observe_action; the determinizer "
            "scores particles with the ISMCTS evaluator's policy "
            "head. See docs/ISMCTS_MIGRATION.md (Phase 6) and the "
            "Phase 6.5 wireup."
        ),
    )
    parser.add_argument(
        "--no-phase-priority-prune", action="store_true",
        help=(
            "Disable the phase-based active-priority prune in the "
            "self-play adapter. With the prune enabled (default), the "
            "active player auto-passes during their own UPKEEP / DRAW "
            "/ BEGINNING_OF_COMBAT / END step when the stack is empty "
            "-- a training-time heuristic that skips many priority "
            "moments at the cost of a small set of niche legitimate "
            "plays. Pass this flag for tournament / correctness-"
            "sensitive runs. See docs/SEARCH_LIMITATIONS.md entry #6 "
            "for the full list of acceptable-failure cases."
        ),
    )
    parser.add_argument(
        "--use-shm-self-play",
        action=argparse.BooleanOptionalAction,
        default=(sys.platform in ("linux", "win32")),
        help=(
            "Use the SharedTensorNNEvalServer (PR #40) for SELF-PLAY "
            "on the LOCAL master shard. Linux + Windows master only; "
            "remote shards (Mac) ignore the flag silently. Bench "
            "numbers from PR #40 show 3-10x NN throughput vs the per-"
            "worker NNEvaluator path; PR #41 smoke at 4w showed ~36 "
            "percent per-game improvement. Default ON on Linux + "
            "Windows, OFF on Mac (no MPS routing in the SHM server "
            "yet). Pass --no-use-shm-self-play to fall back to the "
            "legacy per-worker NNEvaluator path."
        ),
    )
    parser.add_argument(
        "--use-shm-eval",
        action=argparse.BooleanOptionalAction,
        default=(sys.platform in ("linux", "win32")),
        help=(
            "Use the SharedTensorNNEvalServer for EVAL on the LOCAL "
            "master shard. Master constructs two SHM servers (one per "
            "weights set) and workers route via cfg.weights_path. "
            "Linux + Windows master only. Default ON on Linux + "
            "Windows, OFF on Mac. Pass --no-use-shm-eval to fall back "
            "to the legacy per-worker NNEvaluator path. (Earlier "
            "wall-regression at 14 workers was traced to the shutdown "
            "deadlock fixed in PR #43; the substrate now hits 2,981 "
            "RPS @ 14w / 8,695 RPS @ 32w with p50 RTT 74us.)"
        ),
    )
    parser.add_argument(
        "--use-shm-server", action="store_true",
        help=(
            "DEPRECATED — back-compat alias for --use-shm-self-play "
            "AND --use-shm-eval. Both split flags now default ON on "
            "Linux; prefer the split flags (with --no-* to opt out) "
            "rather than this alias."
        ),
    )
    parser.add_argument(
        "--reconcile-state", action="store_true",
        help=(
            "Recover from a partial-iter crash where the trainer "
            "wrote v{N}/ but the iter loop crashed before persisting "
            "the promotion decision into iterate_state.json. At "
            "startup, scan the registry; if exactly one orphan version "
            "is present past iterations_completed, read the per-iter "
            "eval card under training_progress/ and rewrite the state "
            "with the implied promote/keep decision. Off by default — "
            "without this flag the script asserts state matches "
            "reality and crashes on mismatch (silent state mutation "
            "is dangerous). See ``_reconcile_iterate_state`` for the "
            "exact rules and which cases refuse to auto-fix."
        ),
    )
    parser.add_argument(
        "--progress-dir", type=Path, default=Path("training_progress"),
        help=(
            "Directory the iterate loop writes per-iter self-play / "
            "train / eval progress JSON cards to. Default "
            "training_progress/ matches the legacy hardcoded path. "
            "Used by ``--reconcile-state`` to find the orphan iter's "
            "eval card."
        ),
    )
    parser.add_argument(
        "--run-label-prefix", default="",
        help=_run_labels.LABEL_PREFIX_HELP,
    )
    args = parser.parse_args(argv)

    # Fail on a bad training window HERE rather than after a
    # multi-hour self-play phase has already run: the train subprocess
    # is the first thing that would reject it.
    if args.window_recent is not None and args.window_recent_files is not None:
        parser.error(
            "--window-recent (iterations) and --window-recent-files "
            "(raw file count) are mutually exclusive; pass one."
        )
    for _flag, _val in (
        ("--window-recent", args.window_recent),
        ("--window-recent-files", args.window_recent_files),
    ):
        if _val is not None and _val <= 0:
            parser.error(f"{_flag} must be a positive integer; got {_val}")

    # Normalise the lineage prefix once, up-front: every label the
    # loop generates flows through ``mtg_bot.run_labels`` with this
    # value. ``""`` (the default) reproduces the historical
    # ``iter_NNN_*`` ids byte-for-byte. A malformed prefix fails here
    # -- before any runs.db row / npz / replay dir is written.
    try:
        args.run_label_prefix = _run_labels.normalize_label_prefix(
            args.run_label_prefix,
        )
    except _run_labels.InvalidLabelPrefix as exc:
        parser.error(str(exc))
    label_prefix = args.run_label_prefix
    if label_prefix:
        print(
            f"[iterate] run-label prefix: {label_prefix!r} "
            f"(cycle ids look like "
            f"{_run_labels.cycle_id(1, label_prefix)})"
        )

    # Back-compat: --per-game-timeout-sec, if explicitly passed,
    # overrides BOTH phase-specific timeouts. Existing scripts that
    # pass the single legacy flag keep working with no behavior
    # change; new scripts should use the split flags so self-play
    # (uncontended) and eval (14-worker contended) can have distinct
    # budgets per the rationale in ``cli._common``.
    if args.per_game_timeout_sec is not None:
        args.per_game_timeout_self_play_sec = args.per_game_timeout_sec
        args.per_game_timeout_eval_sec = args.per_game_timeout_sec

    # Resolve per-phase timeout defaults from the search work per
    # decision (``nodes * samples``) — see
    # ``_common.PER_GAME_TIMEOUT_SEC_PER_SEARCH_UNIT`` for the
    # measured calibration and why the previous ``3 * --nodes``
    # formula landed below the median game at nodes=100/samples=2.
    # Done here (post-parse) rather than via argparse default= so the
    # formula picks up the user's actual --nodes / --samples values
    # instead of baking in a fixed default at module import.
    _default_timeout = _common.default_per_game_timeout_sec(
        args.nodes, args.samples,
    )
    if args.per_game_timeout_self_play_sec is None:
        args.per_game_timeout_self_play_sec = _default_timeout
    if args.per_game_timeout_eval_sec is None:
        args.per_game_timeout_eval_sec = _default_timeout

    if args.use_belief_weighted and not args.use_ismcts:
        parser.error(
            "--use-belief-weighted requires --use-ismcts; belief-"
            "weighted determinization is meaningless for PIMC's K-"
            "independent-trees structure (no shared determinize_fn "
            "callback). See docs/ISMCTS_MIGRATION.md (Phase 6)."
        )

    if args.corpus_batch is not None:
        if not args.corpus_only:
            parser.error(
                "--corpus-batch only applies to --corpus-only runs "
                "(outside corpus-only mode the iteration index comes "
                "from iterate_state.json's iterations_completed)."
            )
        if args.corpus_batch < 1:
            parser.error(
                f"--corpus-batch must be >= 1 (1-based iteration "
                f"index); got {args.corpus_batch}"
            )

    if not (0.0 < args.one_legal_keep_frac <= 1.0):
        parser.error(
            "--one-legal-keep-frac must be in (0.0, 1.0]; got "
            f"{args.one_legal_keep_frac}"
        )

    if args.early_stop_patience < 1:
        parser.error(
            "--early-stop-patience must be a positive int (>= 1); got "
            f"{args.early_stop_patience}"
        )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.registry_root.mkdir(parents=True, exist_ok=True)

    # Defense-in-depth: scan the data dir for foreign npzs that don't
    # match the <prefix>iter_*.npz pattern. train_nn.py globs
    # everything in the dir, so a stale debug npz here will get mixed
    # into training
    # — that's how today's policy-CE explosion happened (gen0_smoke.npz
    # in training_data/ leaked 603 mask-mismatched rows into v7).
    # With a lineage prefix the guard ALSO catches the other
    # lineage's npzs, so two lineages can't silently share one
    # --data-dir.
    foreign_npzs = find_foreign_npzs(args.data_dir, label_prefix)
    if foreign_npzs:
        sys.exit(
            f"[iterate] {args.data_dir} contains {len(foreign_npzs)} "
            f"non-iter .npz file(s): {foreign_npzs[:10]}"
            f"{' ...' if len(foreign_npzs) > 10 else ''}\n"
            f"train_nn.py would glob these into the training mix and "
            f"potentially corrupt the policy head. Move them out (or "
            f"point --data-dir at a dedicated iterate dir) and retry."
        )

    # v2-cutover: shards_config.json is now a worker URL list only;
    # the coordinator (master's `mtg_bot serve` on :5099) is the
    # entity the orchestrator and workers all talk to. --queue-url
    # is treated as the coordinator URL after the cutover.
    #
    # Dispatch resolution (three branches):
    #
    # 1. ``--shards-config`` omitted → auto-detect
    #    ``_DEFAULT_SHARDS_CONFIG_PATH`` (repo-root anchored). If the
    #    file exists, dispatch distributed. Otherwise fall through to
    #    local Pool with a clear log line so the operator can see the
    #    chosen mode in the iterate header. This is the UX fix for
    #    the iter_019 BUG #4 trap: previously a present
    #    ``shards_config.json`` was silently ignored unless the flag
    #    was passed explicitly.
    #
    # 2. ``--shards-config ""`` → explicit local Pool opt-out (the
    #    ``_LOCAL_POOL_SENTINEL`` returned by ``_shards_config_type``).
    #    Skips auto-detection entirely; used by tests that want to
    #    pin local mode regardless of what's on disk.
    #
    # 3. ``--shards-config /path/to/file`` → use that path. Missing
    #    file fails loudly (sys.exit) — never silently drops to local.
    worker_urls: list[str] | None = None
    shards_config_path: Path | None = None
    if args.shards_config is None:
        # Branch 1: auto-detect.
        if _DEFAULT_SHARDS_CONFIG_PATH.is_file():
            shards_config_path = _DEFAULT_SHARDS_CONFIG_PATH
            print(
                f"[iterate] dispatch mode: distributed "
                f"(auto-detected {shards_config_path})"
            )
        else:
            print(
                f"[iterate] dispatch mode: local Pool "
                f"(no shards_config.json found at "
                f"{_DEFAULT_SHARDS_CONFIG_PATH}; pass "
                f"--shards-config /path to override)"
            )
    elif args.shards_config == _LOCAL_POOL_SENTINEL:
        # Branch 2: explicit local-Pool opt-out.
        print(
            "[iterate] dispatch mode: local Pool "
            "(explicit --shards-config \"\" opt-out)"
        )
    else:
        # Branch 3: explicit path.
        shards_config_path = args.shards_config
        print(
            f"[iterate] dispatch mode: distributed "
            f"(explicit --shards-config {shards_config_path})"
        )

    if shards_config_path is not None:
        if not shards_config_path.is_file():
            sys.exit(f"--shards-config {shards_config_path} not found")
        if not args.queue_url:
            sys.exit(
                "--shards-config requires --queue-url (the master's "
                "coordinator endpoint, e.g. "
                "http://<master-tailnet>:5099)"
            )
        try:
            worker_urls = _rest_orch.load_worker_urls(shards_config_path)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"--shards-config: {e}")
        if not worker_urls:
            sys.exit(
                f"--shards-config {shards_config_path} parsed but "
                "contains no worker_urls entries"
            )
        print(
            f"[iterate] v2 REST distributed run across "
            f"{len(worker_urls)} workers via coordinator {args.queue_url}; "
            f"workers: {worker_urls}"
        )

    if not args.deck_pool.is_file():
        sys.exit(f"--deck-pool {args.deck_pool} not found")
    deck_pool: list[Deck] = decks_from_json(args.deck_pool)
    if not deck_pool:
        sys.exit(
            f"--deck-pool {args.deck_pool} loaded 0 decks"
        )
    avg_size = sum(d.card_count for d in deck_pool) / len(deck_pool)
    print(
        f"[iterate] loaded {len(deck_pool)} decks from "
        f"{args.deck_pool} (avg {avg_size:.1f} cards/deck)"
    )

    registry = Registry(args.registry_root)
    if not registry.has(args.bootstrap_version):
        sys.exit(
            f"Bootstrap version {args.bootstrap_version!r} not found "
            f"in {args.registry_root}. Train one first via "
            f"scripts/train_nn.py."
        )
    # Inherit the trunk shape from the bootstrap version so each
    # iteration trains at the same capacity. Without this, an
    # iterate run starting from v6 (1024-1024-512) would silently
    # train v7 / v8 / ... at the registry-baseline 128-128-64,
    # reducing param count by ~95% per iteration and making the
    # acceptance gate impossible to clear.
    bootstrap_meta = registry.load_metadata(args.bootstrap_version)
    if getattr(bootstrap_meta, "architecture", None) == "structured":
        # Fail on the master in seconds rather than on every worker's
        # first game: a structured bootstrap must record (or be
        # stamped with) the card vocabulary its embeddings index.
        from mtg_bot.training.card_vocab import (
            CardVocabError,
            resolve_checkpoint_card_vocab,
        )

        try:
            # Resolved from the version dir itself (metadata.json, or
            # the stamping sidecar) so the log names where it came from.
            resolved = resolve_checkpoint_card_vocab(
                registry.weights_path(args.bootstrap_version),
            )
        except CardVocabError as exc:
            sys.exit(
                f"Bootstrap version {args.bootstrap_version!r}: {exc}"
            )
        print(
            f"[iterate] {args.bootstrap_version} card vocabulary "
            f"{resolved.vocab.short_sha} ({resolved.vocab.size} names, "
            f"from {resolved.origin})"
        )
    iter_hidden_sizes = (
        tuple(bootstrap_meta.hidden_sizes)
        if getattr(bootstrap_meta, "hidden_sizes", None) else None
    )
    if iter_hidden_sizes:
        print(
            f"[iterate] hidden_sizes inherited from "
            f"{args.bootstrap_version}: {iter_hidden_sizes}"
        )
    # Action-encoding version: inherit from the bootstrap version's
    # metadata so self-play / data-gen / training all run under the
    # bootstrap's slot-mapping convention. Without this thread-through
    # a v17 trained via iterate from a canonical_v2 bootstrap would
    # silently regress to v1 encoding (NNPimcConfig defaults to None →
    # NNEvaluator's positional_v1 default; data_gen defaults to v1;
    # train_nn defaults to v1). Back-compat: v13/v15/v16 metadata
    # predates the field; ``ModelMetadata.from_dict`` defaults missing
    # entries to "positional_v1".
    iter_action_encoding_version = getattr(
        bootstrap_meta, "action_encoding_version", "positional_v1",
    ) or "positional_v1"
    print(
        f"[iterate] action_encoding_version inherited from "
        f"{args.bootstrap_version}: {iter_action_encoding_version}"
    )
    # Architecture: prefer the explicit CLI flag when given; else
    # inherit from the bootstrap version's metadata so an iterate run
    # picks up wherever the v-chain was trained. v1-v15 metadata has
    # no architecture key and back-compat-defaults to "shared".
    bootstrap_architecture = getattr(
        bootstrap_meta, "architecture", "shared",
    )
    iter_architecture = args.architecture
    # 'structured' is a hard v-chain break (the input shape is
    # different — per-token, not 51-dim flat). The shared / separate
    # path can mix bootstraps because both consume the same 51-dim
    # featurizer at NN-eval time; structured can't. Fail loud on a
    # cross-arch attempt rather than silently mis-routing self-play
    # through an NNEvaluator that would crash on first load.
    if (
        bootstrap_architecture == "structured"
        and iter_architecture != "structured"
    ):
        sys.exit(
            f"[iterate] bootstrap {args.bootstrap_version} is a "
            f"structured-architecture model but --architecture="
            f"{iter_architecture!r}; the structured trunk consumes "
            f"per-token input and is NOT cross-loadable with the "
            f"flat-feature MLP. Either pass --architecture structured "
            f"to stay on the structured v-chain, or pick a "
            f"flat-feature bootstrap."
        )
    if (
        iter_architecture == "structured"
        and bootstrap_architecture != "structured"
    ):
        sys.exit(
            f"[iterate] --architecture=structured requires a "
            f"structured-architecture bootstrap version, but "
            f"{args.bootstrap_version} has "
            f"architecture={bootstrap_architecture!r}. Train a "
            f"structured bootstrap first via ``mtg_bot train "
            f"--architecture structured`` and pass its version "
            f"here. (Structured is a fresh v-chain break: the "
            f"per-token input shape is incompatible with the "
            f"flat-feature MLP weights.)"
        )
    if iter_architecture != bootstrap_architecture:
        # The architectures aren't cross-loadable, so this combination
        # bootstrap-from-random-init is the only viable path. Self-play
        # MUST run against the bootstrap's matching architecture
        # (otherwise NNEvaluator load fails); training writes the new
        # architecture for the produced version, and on iter 2 the
        # promoted version will use that arch end-to-end.
        print(
            f"[iterate] WARNING: --architecture={iter_architecture!r} "
            f"differs from bootstrap {args.bootstrap_version}'s "
            f"architecture={bootstrap_architecture!r}. Self-play will "
            f"use {bootstrap_architecture!r} (the bootstrap's saved "
            f"weights' shape); the trained new version will use "
            f"{iter_architecture!r} from random init. The two "
            f"architectures are NOT cross-loadable — once promoted, "
            f"subsequent iterations run end-to-end on the new arch."
        )
    # Slice G: the iter's effective write_structured_tokens flag
    # follows ``iter_architecture == 'structured'``. Self-play emits
    # the four extra structured-token columns on every npz alongside
    # the legacy 51-dim featurizer; the next train phase
    # (subprocess-routed through ``--architecture structured`` in
    # ``_train_next``) consumes those columns via the structured
    # trainer. Flat-feature lineages leave it at False so existing v*
    # npzs stay bit-identical.
    iter_write_structured_tokens = iter_architecture == "structured"
    if iter_write_structured_tokens:
        print(
            "[iterate] structured architecture detected: "
            "self-play will emit structured-token columns "
            "(emb_dim={ed}, n_heads={nh}, n_layers={nl}, "
            "hidden_dim={hd}, action_history_n={ahn})".format(
                ed=args.emb_dim, nh=args.n_heads, nl=args.n_layers,
                hd=args.hidden_dim, ahn=args.action_history_n,
            )
        )
    iter_policy_hidden_sizes: tuple[int, ...] | None = None
    if args.policy_hidden_sizes is not None:
        if iter_architecture != "separate":
            sys.exit(
                "--policy-hidden-sizes only applies when "
                "--architecture=separate"
            )
        try:
            iter_policy_hidden_sizes = tuple(
                int(x.strip())
                for x in args.policy_hidden_sizes.split(",")
            )
        except ValueError as exc:
            sys.exit(f"--policy-hidden-sizes parse error: {exc}")
        if not iter_policy_hidden_sizes:
            sys.exit("--policy-hidden-sizes must list at least one layer")
    print(
        f"[iterate] architecture: bootstrap={bootstrap_architecture}, "
        f"new-iters will train as={iter_architecture}"
    )

    # Initialize the iterate state from the bootstrap version (or
    # resume from disk if the loop was interrupted earlier).
    state = _load_iterate_state(args.registry_root)
    if not state:
        state = {
            "current_best": args.bootstrap_version,
            "iterations_completed": 0,
            "history": [],
        }
        _save_iterate_state(args.registry_root, state)
        print(
            f"[iterate] Initialized state at "
            f"{args.registry_root / _STATE_FILENAME}: "
            f"current_best={state['current_best']}"
        )
    else:
        print(
            f"[iterate] Resuming from disk state: "
            f"current_best={state['current_best']}, "
            f"iterations_completed={state['iterations_completed']}"
        )
        # Slice G: detect a clearly-mismatched resume where
        # ``current_best`` in the state file is on a different
        # architecture lineage than the iter is being told to run.
        # The structured trunk is a hard v-chain break so the
        # cross-arch case isn't recoverable by warning — self-play
        # would fail at the NNEvaluator load. Fail loud with the
        # versions in scope so the user can either swap
        # --bootstrap-version or rewrite iterate_state.json.
        try:
            current_best_meta = registry.load_metadata(state["current_best"])
            current_best_arch = getattr(
                current_best_meta, "architecture", "shared",
            )
        except (FileNotFoundError, KeyError, OSError):
            current_best_arch = None
        if (
            current_best_arch is not None
            and current_best_arch != bootstrap_architecture
            and (
                current_best_arch == "structured"
                or bootstrap_architecture == "structured"
            )
        ):
            sys.exit(
                f"[iterate] resume mismatch: iterate_state.json "
                f"current_best={state['current_best']} has "
                f"architecture={current_best_arch!r} but "
                f"--bootstrap-version {args.bootstrap_version} has "
                f"architecture={bootstrap_architecture!r}. The "
                f"structured trunk is a hard v-chain break (per-token "
                f"input shape ≠ flat 51-dim featurizer); the resumed "
                f"iter cannot bridge the two. Either rerun with a "
                f"bootstrap that matches current_best's architecture, "
                f"or hand-edit iterate_state.json to clear the "
                f"current_best pointer."
            )
        if args.reconcile_state:
            try:
                reconciled = _reconcile_iterate_state(
                    state,
                    args.registry_root,
                    args.progress_dir,
                    args.bootstrap_version,
                    args.acceptance_threshold,
                    label_prefix,
                )
            except ReconcileError as e:
                sys.exit(f"[reconcile] FAILED: {e}")
            if reconciled is not state:
                state = reconciled
                _save_iterate_state(args.registry_root, state)
                print(
                    f"[reconcile] state rewritten: "
                    f"current_best={state['current_best']}, "
                    f"iterations_completed={state['iterations_completed']}"
                )

    # Orphan sweep — kill any prior-run multiprocessing/shard
    # workers that survived their parent. See
    # ``mtg_bot.training.orphan_sweep`` for the matcher rules. Runs
    # before iter 1 begins so the next iteration's fresh workers have
    # the machine to themselves; otherwise stale orphans (50-700 MB
    # each) compete for CPU and tank self-play throughput. Loud so
    # the user can see what's happening.
    repo_root = str(_REPO_ROOT.resolve())
    print(f"[iterate] orphan-sweep: scanning local processes (master "
          f"PID={os.getpid()}, repo_root={repo_root!r})")
    try:
        sweep_local_orphans(repo_root=repo_root, label="local")
    except Exception as e:  # noqa: BLE001
        # Sweep is a safety net, not a critical-path action — never
        # let it abort the run. Log loudly and continue.
        print(
            f"[iterate] orphan-sweep: local sweep raised "
            f"{type(e).__name__}: {e}; continuing.",
            file=sys.stderr,
        )
    # v2-cutover: remote-orphan SSH sweep is gone. Each worker is a
    # long-lived daemon under its own native service supervisor
    # (systemd / launchd / Task Scheduler); orphan workers (if any)
    # are the daemon's own multiprocessing pool's problem to clean
    # up, not the orchestrator's. The master-local sweep above stays
    # (it scans the local processes for stale workers).

    from datetime import datetime, timezone
    user_stopped = False
    # In-flight cycle id used by the outermost except handler to mark
    # an unfinished cycle as failed/killed in the run DB when an
    # exception punches through every nested try-block. The cycle
    # body sets it immediately after start_run and the inner
    # ``_finish_cycle`` helper clears it once finish_run lands; if
    # the helper short-circuits via ``cycle_finished``, the cleared
    # state already matches reality.
    _in_flight_cycle = {"id": None}  # mutable holder so closures can clear
    try:
      for iter_idx in range(args.n_iterations):
        if user_stopped:
            break
        n_done = _iteration_index(
            iterations_completed=state["iterations_completed"],
            iter_idx=iter_idx,
            corpus_only=args.corpus_only,
            corpus_batch=args.corpus_batch,
        )
        cycle_seed = args.seed + 1_000_000 * (n_done + 1)
        best_version = state["current_best"]
        best_weights = registry.weights_path(best_version)
        self_play_evaluator = args.self_play_evaluator
        heuristic_self_play = (
            self_play_evaluator == _EVALUATOR_HEURISTIC
        )
        # A heuristic corpus is not attributable to any network, so
        # the stage's run id / npz names say so rather than naming a
        # version that never played a card.
        self_play_agent_label = (
            _EVALUATOR_HEURISTIC if heuristic_self_play
            else best_version
        )
        # Look up the current best's architecture so the self-play
        # NNEvaluator instantiates the matching dual-head class. v1-v15
        # pre-architecture metadata round-trips to "shared" via
        # ModelMetadata.from_dict.
        best_meta = registry.load_metadata(best_version)
        best_architecture = getattr(best_meta, "architecture", "shared")
        best_policy_hidden_sizes = (
            tuple(
                best_meta.training.get("policy_hidden_sizes") or ()
            ) or None
        )
        best_action_encoding_version = getattr(
            best_meta, "action_encoding_version", "positional_v1",
        )

        print(
            f"\n[iterate] === Iteration {n_done + 1}/"
            f"{state['iterations_completed'] + args.n_iterations} "
            f"(global #{n_done + 1}) ==="
        )
        print(
            f"[iterate] Self-playing with {self_play_agent_label}"
            f"{' (heuristic evaluator, no weights)' if heuristic_self_play else ''}"
            f"..."
        )

        # 1. Self-play data gen
        #
        # Every id below is derived through ``mtg_bot.run_labels`` so
        # the unified launcher (which pre-registers the cycle row in
        # runs.db from ``[iter].target_iter``) and this loop agree
        # byte-for-byte. ``label_prefix`` namespaces the lineage; ""
        # reproduces the historical ``iter_NNN_*`` ids exactly.
        iter_label = _run_labels.iter_label(n_done + 1, label_prefix)
        progress_dir = args.progress_dir
        data_path = (
            args.data_dir
            / f"{iter_label}_{self_play_agent_label}_{_now_stamp()}.npz"
        )
        self_play_run_id = _run_labels.self_play_run_id(
            n_done + 1, self_play_agent_label, label_prefix,
        )
        self_play_progress_path = (
            progress_dir / f"{self_play_run_id}.json"
        )

        # Cycle-level progress JSON: a parent card the user can target
        # with a single Stop click to abort the whole iter (self-play
        # + train + eval). Children carry a back-pointer so the UI
        # stop endpoint resolves their click to this cycle_id.
        cycle_id = _run_labels.cycle_id(n_done + 1, label_prefix)
        cycle_started_iso = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        cycle_started_perf = time.perf_counter()
        # Sweep stale sentinels so a leftover .STOP_iter_001_cycle from
        # a previous session doesn't trigger an immediate abort. We
        # only keep sentinels for run_ids we're about to spawn this
        # cycle.
        active_ids = [
            cycle_id, self_play_run_id,
            f"{iter_label}_train",
            f"{iter_label}_eval",
        ]
        cleanup_orphans(progress_dir, active_ids)
        # If the cycle-level sentinel already exists at this point
        # (set between the cleanup and now? Or set externally?), skip
        # the iter entirely. Defensive — normal flow has
        # cleanup_orphans drop everything.
        if stop_requested(progress_dir, cycle_id):
            print(
                f"[iterate]   stop sentinel already present for "
                f"{cycle_id}; skipping iter."
            )
            user_stopped = True
            break
        # Register the cycle in the SQLite run DB. Multiple early-break
        # paths (stop-sentinel detected after self-play / train / eval)
        # need to mark the row ``killed`` rather than ``completed``,
        # which the ``run_lifecycle`` CM can't express (its else-clause
        # always overwrites mid-body finish_run calls with
        # ``completed``). So we use the lower-level ``start_run`` /
        # ``finish_run`` primitives directly and replicate lifecycle's
        # status mapping in the surrounding try/except. The run is a
        # no-op when neither ``MTG_BOT_RUN_DB_PATH`` nor
        # ``MTG_BOT_RUN_DB_URL`` is set.
        cycle_config = _cycle_config_dict(
            args=args,
            iter_idx_one_based=n_done + 1,
            best_version=best_version,
        )
        cycle_json_path = progress_dir / f"{cycle_id}.json"
        _runs_client.start_run(
            cycle_id,
            kind="iterate",
            config_json=_lifecycle.serialize_config(cycle_config),
            git_commit=_lifecycle.git_commit_short(),
            label=iter_label,
        )
        _in_flight_cycle["id"] = cycle_id
        # Register the (now-no-longer-written) cycle JSON path as an
        # artifact for historical compatibility — the artifact row
        # records "this is where the JSON used to live" even though
        # Phase 3 stops writing the file.
        _lifecycle.register_progress_artifact(cycle_id, cycle_json_path)
        _runs_client.push_event(
            cycle_id, "cycle_started",
            payload={
                "iter_idx": n_done + 1,
                "best_version": best_version,
            },
        )
        # Push the initial cycle-progress row carrying iter_idx + stage
        # so the UI's compact "iter N of M, stage=self_play" card has
        # something to render before any per-stage progress lands.
        _push_cycle_stage_progress(
            cycle_id=cycle_id,
            iter_idx_one_based=n_done + 1,
            n_iterations=state["iterations_completed"]
            + args.n_iterations,
            started_perf=cycle_started_perf,
            stage="self_play",
        )
        cycle_finished = False

        def _cycle_stop_check() -> bool:
            return stop_requested(progress_dir, cycle_id)

        def _finish_cycle(status: str, *, notes: str | None = None) -> None:
            """Mark the cycle terminal in the run DB. Idempotent — the
            sentinel ``cycle_finished`` guards against double-call
            (which would just be a no-op UPDATE but keeps logs clean).
            Also clears the outer ``_in_flight_cycle`` holder so the
            outermost except handler doesn't double-write.
            """
            nonlocal cycle_finished
            if cycle_finished:
                return
            cycle_finished = True
            _runs_client.finish_run(cycle_id, status, notes=notes)
            _in_flight_cycle["id"] = None

        self_play_replay_config: ReplayCaptureConfig | None = None
        if args.capture_every_self_play > 0:
            sp_replays_dir = args.replays_dir / self_play_run_id
            sp_replays_dir.mkdir(parents=True, exist_ok=True)
            # Heuristic stages capture replays with the always-on
            # heuristic_eval annotator only: there is no network in
            # the loop, and wiring one in purely to annotate would
            # load torch on every worker for no training signal.
            self_play_replay_config = ReplayCaptureConfig(
                save_dir=str(sp_replays_dir),
                capture_every=args.capture_every_self_play,
                nn_weights_path=(
                    None if heuristic_self_play else str(best_weights)
                ),
                nn_version_label=best_version,
                nn_hidden_sizes=iter_hidden_sizes,
                nn_with_policy=True,
                nn_architecture=best_architecture,
                nn_policy_hidden_sizes=best_policy_hidden_sizes,
                # Sidecar metadata.json sits next to weights.pt in the
                # registry; threading it lets the per-game annotator
                # construct via NNEvaluator.from_metadata so non-default
                # structured-trunk dims load (iter_019 BUG #3).
                nn_metadata_path=str(
                    best_weights.parent / "metadata.json"
                ),
            )
        sp_hang_dir = (
            str(args.hang_dump_dir / self_play_run_id)
            if str(args.hang_dump_dir) else None
        )
        # Phase 1 dual-write: register the self-play stage as its own
        # run row keyed under the cycle as parent. ``run_lifecycle``'s
        # try/finally maps a clean exit to ``completed``,
        # KeyboardInterrupt/SystemExit to ``killed``, and any other
        # exception to ``failed``. The wrapped body is the existing
        # self-play call site, unchanged.
        _runs_client.push_event(
            cycle_id, "stage_started",
            payload={"stage": "self_play", "run_id": self_play_run_id},
        )
        with _lifecycle.run_lifecycle(
            run_id=self_play_run_id,
            kind="self_play",
            config=_self_play_config_dict(
                args=args, best_version=best_version,
                cycle_seed=cycle_seed,
                distributed=worker_urls is not None,
            ),
            progress_json_path=self_play_progress_path,
            parent_run_id=cycle_id,
            label=self_play_run_id,
        ):
            if worker_urls is None:
                n_samples, n_policy, gen_elapsed = _self_play(
                    weights_path=best_weights,
                    n_games=args.games_per_iteration,
                    samples=args.samples,
                    nodes=args.nodes,
                    seed=cycle_seed,
                    n_workers=args.workers,
                    output=data_path,
                    per_game_timeout_sec=args.per_game_timeout_self_play_sec,
                    hidden_sizes=iter_hidden_sizes,
                    architecture=best_architecture,
                    policy_hidden_sizes=best_policy_hidden_sizes,
                    progress_path=self_play_progress_path,
                    run_id=self_play_run_id,
                    deck_pool=deck_pool,
                    replay_config=self_play_replay_config,
                    hang_dump_dir=sp_hang_dir,
                    heuristics=HeuristicsConfig(
                        phase_prune_enabled=not args.no_phase_priority_prune,
                    ),
                    cycle_id=cycle_id,
                    stop_check=_cycle_stop_check,
                    action_encoding_version=iter_action_encoding_version,
                    write_structured_tokens=iter_write_structured_tokens,
                    structured_scalar_schema_version=(
                        args.scalar_schema_version
                    ),
                    root_dirichlet_alpha=_dirichlet_alpha_or_none(
                        args.root_dirichlet_alpha
                    ),
                    root_dirichlet_eps=args.root_dirichlet_eps,
                    move_temperature=args.move_temperature,
                    temperature_cutoff=args.temperature_cutoff,
                    evaluator=self_play_evaluator,
                )
            else:
                # v2 distributed: the coordinator's /v2/report_result
                # handler streams each worker's per-game npz into
                # iterate_data/ on the master. train_nn.py globs the
                # dir at the next stage, so the same training mix is
                # picked up automatically.
                n_samples, n_policy, gen_elapsed = _self_play_distributed(
                    worker_urls=worker_urls,
                    coordinator_url=args.queue_url,
                    nn_version_label=best_version,
                    samples=args.samples,
                    nodes=args.nodes,
                    seed=cycle_seed,
                    per_game_timeout_sec=args.per_game_timeout_self_play_sec,
                    capture_every=args.capture_every_self_play,
                    total_games=args.games_per_iteration,
                    deck_pool_size=len(deck_pool),
                    run_id_prefix=self_play_run_id,
                    iterate_data_dir=args.data_dir,
                    stop_check=_cycle_stop_check,
                    root_dirichlet_alpha=_dirichlet_alpha_or_none(
                        args.root_dirichlet_alpha
                    ),
                    root_dirichlet_eps=args.root_dirichlet_eps,
                    move_temperature=args.move_temperature,
                    temperature_cutoff=args.temperature_cutoff,
                    evaluator=self_play_evaluator,
                    action_encoding_version=iter_action_encoding_version,
                    structured_tokens=iter_write_structured_tokens,
                    structured_scalar_schema_version=(
                        args.scalar_schema_version
                    ),
                )
        # Self-play stage finished — register the produced npz as an
        # artifact on the self_play stage run and emit a stage_completed
        # event on the cycle (the lifecycle CM already wrote the stage's
        # terminal status into the runs row).
        try:
            npz_bytes = data_path.stat().st_size if data_path.exists() else None
        except OSError:
            npz_bytes = None
        if npz_bytes is not None:
            _runs_client.push_artifact(
                self_play_run_id,
                artifact_type="npz", path=str(data_path),
                bytes_=npz_bytes,
            )
        _runs_client.push_progress(
            self_play_run_id,
            n_games_done=args.games_per_iteration,
            n_games_total=args.games_per_iteration,
            elapsed_seconds=gen_elapsed,
            extra={
                "n_samples": int(n_samples),
                "n_policy_rows": int(n_policy),
            },
        )
        _runs_client.push_event(
            cycle_id, "stage_completed",
            payload={
                "stage": "self_play", "run_id": self_play_run_id,
                "n_samples": int(n_samples),
                "n_policy_rows": int(n_policy),
                "elapsed_seconds": float(gen_elapsed),
            },
        )
        # Guard the division below: a zero-sample stage is a data-loss
        # bug, not a divide-by-zero bug. Distributed runs also check
        # ``upload_failed`` inside ``_self_play_distributed``; this
        # site covers the local path and any distributed path that
        # came back empty.
        yield_problem = self_play_yield_error(
            stage_id=self_play_run_id,
            data_dir=args.data_dir,
            n_samples=n_samples,
        )
        if yield_problem is not None:
            sys.exit(yield_problem)
        print(
            f"[iterate]   gen done in {gen_elapsed/60:.1f}m: "
            f"{n_samples} samples, {n_policy} with policy "
            f"({n_policy / n_samples:.1%})"
        )

        # Stage boundary: if a stop sentinel appeared during self-
        # play (or was set just now), short-circuit the rest of the
        # cycle. Self-play may have already finalized as 'stopped'
        # via its stop_check; the cycle JSON still needs to be
        # finalized so the parent card flips out of 'running'.
        if _cycle_stop_check():
            print(
                f"[iterate]   stop sentinel detected after "
                f"self-play; aborting cycle {cycle_id}"
            )
            _finish_cycle("killed", notes="stopped via UI sentinel (self_play)")
            user_stopped = True
            break

        # --corpus-only: the self-play shard IS the deliverable. No
        # train, no eval, no promotion, and iterate_state.json is
        # left exactly as we found it, so a later real iterate run
        # resumes its v-chain from where it was. The corpus lands in
        # --data-dir like any other self-play shard and is picked up
        # by the next `mtg_bot train` / iterate cycle that globs it.
        if args.corpus_only:
            print(
                f"[iterate]   --corpus-only: skipping train + eval; "
                f"corpus shard(s) for {self_play_run_id} are in "
                f"{args.data_dir}"
            )
            _finish_cycle(
                "completed",
                notes=f"corpus-only ({self_play_evaluator}) self-play",
            )
            continue

        # 2. Train next version
        peeked_train_version = registry.next_version_id()
        # Phase 1 dual-write: wrap the train subprocess call in a
        # ``training`` stage run row. ``_train_next`` raises
        # RuntimeError on a non-zero exit, which the lifecycle CM
        # maps to status=failed.
        train_run_id = _run_labels.train_run_id(
            n_done + 1, peeked_train_version, label_prefix,
        )
        train_progress_path = progress_dir / f"{train_run_id}.json"
        _runs_client.push_event(
            cycle_id, "stage_started",
            payload={"stage": "train", "run_id": train_run_id},
        )
        try:
          with _lifecycle.run_lifecycle(
            run_id=train_run_id,
            kind="training",
            config=_train_config_dict(
                args=args,
                peeked_train_version=peeked_train_version,
                cycle_seed=cycle_seed,
            ),
            progress_json_path=train_progress_path,
            parent_run_id=cycle_id,
            label=train_run_id,
          ):
            new_version = _train_next(
                data_dir=args.data_dir,
                registry_root=args.registry_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                val_split=args.val_split,
                seed=cycle_seed,
                notes=(
                    f"iterate_alphazero iteration #{n_done + 1}, "
                    f"self-play driven by {best_version}"
                ),
                window_recent=args.window_recent,
                window_recent_files=args.window_recent_files,
                hidden_sizes=iter_hidden_sizes,
                progress_path=train_progress_path,
                progress_dir=progress_dir,
                policy_loss_weight=args.policy_loss_weight,
                one_legal_keep_frac=args.one_legal_keep_frac,
                architecture=iter_architecture,
                policy_hidden_sizes=iter_policy_hidden_sizes,
                early_stop_patience=args.early_stop_patience,
                action_encoding_version=iter_action_encoding_version,
                # Structured-trunk hyperparameters; ``_train_next``
                # only forwards them as CLI flags when
                # ``architecture == 'structured'`` so the flat-feature
                # trainer doesn't see unknown flags.
                structured_emb_dim=args.emb_dim,
                structured_n_heads=args.n_heads,
                structured_n_layers=args.n_layers,
                structured_hidden_dim=args.hidden_dim,
                structured_action_history_n=args.action_history_n,
                structured_scalar_schema_version=(
                    args.scalar_schema_version
                ),
                num_workers=args.train_num_workers,
                shuffle_mode=args.train_shuffle_mode,
                shuffle_block_rows=args.train_shuffle_block_rows,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                value_head_input=args.value_head_input,
                value_target=args.value_target,
                keep_last_epoch=args.keep_last_epoch,
                min_holdout_value_quality=(
                    args.min_holdout_value_quality
                ),
            )
        except TrainRejected as exc:
            # The trained value head failed its held-out quality gate,
            # so no version was registered. Treat the CYCLE as failed
            # (with the reason on the row) but keep going: the next
            # cycle generates fresh self-play data off the SAME
            # current_best and gets another shot. Tearing the chain
            # down here would turn one bad train into a dead run.
            print(
                f"[iterate]   train REJECTED for cycle {cycle_id}: "
                f"{exc}",
                file=sys.stderr,
            )
            _runs_client.push_event(
                cycle_id, "stage_failed",
                payload={
                    "stage": "train",
                    "run_id": train_run_id,
                    "reason": "holdout_value_gate",
                    "detail": str(exc),
                },
            )
            state["history"].append({
                "iteration": n_done + 1,
                "data_file": str(data_path.name),
                "n_samples": n_samples,
                "n_policy_rows": n_policy,
                "gen_seconds": gen_elapsed,
                "trained_version": None,
                "promoted": False,
                "rejected": True,
                "rejected_reason": "holdout_value_gate",
                "eval": {},
            })
            state["iterations_completed"] = n_done + 1
            _save_iterate_state(args.registry_root, state)
            _push_cycle_stage_progress(
                cycle_id=cycle_id,
                iter_idx_one_based=n_done + 1,
                n_iterations=state["iterations_completed"]
                + args.n_iterations - 1,
                started_perf=cycle_started_perf,
                stage="rejected",
            )
            _finish_cycle(
                "failed",
                notes=(
                    "train rejected by the held-out value-quality "
                    f"gate; keeping current_best="
                    f"{state['current_best']}"
                ),
            )
            continue
        new_weights = registry.weights_path(new_version)
        # Register the produced weights file as an artifact on the
        # training run. Bytes lookup is best-effort — the weights write
        # may have raced (it shouldn't, since _train_next returned, but
        # the OSError fallback keeps the dual-write loud-fail-free).
        try:
            weights_bytes = (
                new_weights.stat().st_size
                if new_weights.exists() else None
            )
        except OSError:
            weights_bytes = None
        if weights_bytes is not None:
            _runs_client.push_artifact(
                train_run_id,
                artifact_type="weights",
                path=str(new_weights),
                bytes_=weights_bytes,
            )
        _runs_client.push_event(
            cycle_id, "stage_completed",
            payload={
                "stage": "train",
                "run_id": train_run_id,
                "trained_version": new_version,
            },
        )
        print(f"[iterate]   trained: {new_version}")

        if _cycle_stop_check():
            print(
                f"[iterate]   stop sentinel detected after train; "
                f"keeping trained weights {new_version} but skipping "
                f"eval/promote for cycle {cycle_id}"
            )
            _finish_cycle("killed", notes="stopped via UI sentinel (train)")
            user_stopped = True
            break

        # 3. Acceptance gate. Runs whenever the user opted in via
        # ``--acceptance-threshold > 0``: every freshly-trained
        # version IS apples-to-apples comparable to the current
        # best (they're two distinct registered weights). The
        # earlier ``best_version != bootstrap_version`` guard was
        # wrong — when iterate is RESUMED from a state where
        # current_best happens to equal --bootstrap-version (a
        # totally normal case), it silently skipped the gate and
        # promoted blind. Observed live on iter 2 of the 2026-05-06
        # run: v13 promoted without eval despite
        # --acceptance-threshold 0.55 --eval-games 100 on the CLI.
        promoted = False
        eval_summary = {}
        if args.acceptance_threshold > 0 and args.eval_games > 0:
            print(
                f"[iterate]   eval-gate: {new_version} vs "
                f"{best_version}, {args.eval_games} games..."
            )
            eval_run_id = _run_labels.eval_run_id(
                n_done + 1, new_version, best_version, label_prefix,
            )
            eval_progress_path = progress_dir / f"{eval_run_id}.json"
            eval_replay_config: ReplayCaptureConfig | None = None
            if args.capture_every_eval > 0:
                ev_replays_dir = args.replays_dir / eval_run_id
                ev_replays_dir.mkdir(parents=True, exist_ok=True)
                eval_replay_config = ReplayCaptureConfig(
                    save_dir=str(ev_replays_dir),
                    capture_every=args.capture_every_eval,
                    nn_weights_path=str(new_weights),
                    nn_version_label=new_version,
                    nn_hidden_sizes=iter_hidden_sizes,
                    nn_with_policy=True,
                    nn_architecture=iter_architecture,
                    nn_policy_hidden_sizes=iter_policy_hidden_sizes,
                    # Sidecar metadata.json sits next to weights.pt in
                    # the registry; threading it lets the per-game
                    # annotator construct via NNEvaluator.from_metadata
                    # so non-default structured-trunk dims load
                    # (iter_019 BUG #3).
                    nn_metadata_path=str(
                        new_weights.parent / "metadata.json"
                    ),
                )
            eval_hang_dir = (
                str(args.hang_dump_dir / eval_run_id)
                if str(args.hang_dump_dir) else None
            )
            _runs_client.push_event(
                cycle_id, "stage_started",
                payload={"stage": "eval", "run_id": eval_run_id},
            )
            with _lifecycle.run_lifecycle(
                run_id=eval_run_id,
                kind="eval",
                config=_eval_config_dict(
                    args=args, new_version=new_version,
                    best_version=best_version,
                    cycle_seed=cycle_seed,
                    distributed=worker_urls is not None,
                ),
                progress_json_path=eval_progress_path,
                parent_run_id=cycle_id,
                label=eval_run_id,
            ):
                if worker_urls is None:
                    wins_new, wins_old, draws, eval_elapsed = _eval_match(
                        new_weights=new_weights,
                        old_weights=best_weights,
                        n_games=args.eval_games,
                        samples=args.samples,
                        nodes=args.nodes,
                        n_workers=args.workers,
                        seed=cycle_seed + 7777,
                        per_game_timeout_sec=args.per_game_timeout_eval_sec,
                        hidden_sizes=iter_hidden_sizes,
                        new_architecture=iter_architecture,
                        old_architecture=best_architecture,
                        policy_hidden_sizes=iter_policy_hidden_sizes,
                        progress_path=eval_progress_path,
                        run_id=eval_run_id,
                        opponent_label=best_version,
                        deck_pool=deck_pool,
                        replay_config=eval_replay_config,
                        hang_dump_dir=eval_hang_dir,
                        cycle_id=cycle_id,
                        stop_check=_cycle_stop_check,
                        action_encoding_version=iter_action_encoding_version,
                    )
                else:
                    wins_new, wins_old, draws, eval_elapsed = (
                        _eval_match_distributed(
                            worker_urls=worker_urls,
                            coordinator_url=args.queue_url,
                            new_version_label=new_version,
                            old_version_label=best_version,
                            n_games=args.eval_games,
                            samples=args.samples,
                            nodes=args.nodes,
                            seed=cycle_seed + 7777,
                            per_game_timeout_sec=args.per_game_timeout_eval_sec,
                            capture_every=args.capture_every_eval,
                            deck_pool_size=len(deck_pool),
                            run_id_prefix=eval_run_id,
                            stop_check=_cycle_stop_check,
                        )
                    )
            n_decided = wins_new + wins_old
            win_rate = wins_new / n_decided if n_decided else 0.0
            eval_summary = {
                "vs": best_version,
                "wins_new": wins_new,
                "wins_old": wins_old,
                "draws": draws,
                "win_rate": win_rate,
                "elapsed_seconds": eval_elapsed,
            }
            # Push the eval summary as the aggregate progress row and
            # log the gate-decision event before promotion bookkeeping
            # so the DB reflects the same wins/draws/win_rate the
            # eval JSON does (and the UI's runs.db query agrees with
            # the per-iter eval card).
            _lifecycle.push_eval_summary(
                eval_run_id,
                wins_new=wins_new, wins_old=wins_old, draws=draws,
                elapsed_seconds=eval_elapsed,
                extra={
                    "win_rate": win_rate,
                    "opponent_label": best_version,
                },
            )
            gate_passed = win_rate >= args.acceptance_threshold
            _runs_client.push_event(
                cycle_id, "eval_gate_decision",
                payload={
                    "run_id": eval_run_id,
                    "new": new_version,
                    "old": best_version,
                    "win_rate": win_rate,
                    "threshold": args.acceptance_threshold,
                    "passed": gate_passed,
                    "wins_new": int(wins_new),
                    "wins_old": int(wins_old),
                    "draws": int(draws),
                },
            )
            _runs_client.push_event(
                cycle_id, "stage_completed",
                payload={
                    "stage": "eval", "run_id": eval_run_id,
                    "win_rate": win_rate,
                    "elapsed_seconds": float(eval_elapsed),
                },
            )
            print(
                f"[iterate]   eval result: {new_version} won "
                f"{wins_new}, {best_version} won {wins_old}, "
                f"draws {draws} (win_rate={win_rate:.1%})"
            )
            if gate_passed:
                promoted = True
                print(
                    f"[iterate]   promote {new_version} -> best "
                    f"(win_rate {win_rate:.1%} >= "
                    f"{args.acceptance_threshold:.1%})"
                )
            else:
                print(
                    f"[iterate]   keep {best_version} as best "
                    f"(win_rate {win_rate:.1%} < "
                    f"{args.acceptance_threshold:.1%})"
                )
        else:
            promoted = True
            print(f"[iterate]   no eval gate; promoting {new_version}")

        # 4. Update state
        state["history"].append({
            "iteration": n_done + 1,
            "data_file": str(data_path.name),
            "n_samples": n_samples,
            "n_policy_rows": n_policy,
            "gen_seconds": gen_elapsed,
            "trained_version": new_version,
            "promoted": promoted,
            "eval": eval_summary,
        })
        state["iterations_completed"] = n_done + 1
        if promoted:
            state["current_best"] = new_version
        _save_iterate_state(args.registry_root, state)
        # Mirror the promote/keep decision as a run-DB event so the UI
        # can pin a "promoted v17 over v16" annotation onto the cycle
        # row without re-parsing the JSON history. Fired for both
        # promoted and not-promoted cases; the payload distinguishes.
        _runs_client.push_event(
            cycle_id, "promoted",
            payload={
                "from_version": best_version,
                "to_version": new_version,
                "promoted": promoted,
                "eval_summary": eval_summary or None,
            },
        )

        # Cycle complete: stage-progress -> completed so the UI's
        # stage indicator advances, then flip the row terminal. Stage
        # boundaries above already short-circuit on a stop sentinel,
        # so reaching here means the cycle ran to completion.
        _push_cycle_stage_progress(
            cycle_id=cycle_id,
            iter_idx_one_based=n_done + 1,
            n_iterations=state["iterations_completed"]
            + args.n_iterations - 1,
            started_perf=cycle_started_perf,
            stage="completed",
        )
        # ``cycle_finished`` is the idempotency guard.
        _finish_cycle("completed")
    except (KeyboardInterrupt, SystemExit):
        # User-triggered abort (Ctrl-C, SystemExit from a stop signal
        # handler upstream). Mark any in-flight cycle ``killed`` in
        # the run DB before re-raising; the per-stage lifecycle CMs
        # already handled their own rows. Phase 1 invariant: this is
        # purely additive — the existing JSON-side bookkeeping is
        # unchanged.
        in_flight = _in_flight_cycle["id"]
        if in_flight is not None:
            _runs_client.finish_run(
                in_flight, "killed",
                notes="KeyboardInterrupt / SystemExit",
            )
        raise
    except BaseException as _exc:  # noqa: BLE001 — re-raised below
        # Unexpected exception. Mark the cycle ``failed`` with the
        # exception type + message in notes; the per-stage lifecycle
        # CMs already mapped their own rows. Re-raise so the caller
        # sees the same exception path as before.
        in_flight = _in_flight_cycle["id"]
        if in_flight is not None:
            _runs_client.finish_run(
                in_flight, "failed",
                notes=f"{type(_exc).__name__}: {_exc}",
            )
        raise

    if user_stopped:
        print(
            "\n[iterate] Loop aborted by user stop request "
            f"(after {state['iterations_completed']} completed "
            f"iterations). Current best: {state['current_best']}."
        )
    else:
        print(
            f"\n[iterate] Loop done. {args.n_iterations} iterations "
            f"completed. Current best: {state['current_best']}."
        )


if __name__ == "__main__":
    main()
