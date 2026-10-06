"""Production GameRunners for the worker daemon (Phase 2c).

Lifted from ``mtg_bot.cli.shard._run_pull_shard`` / ``_run_eval_shard``
and ``mtg_bot.training.distributed.run_shard_pull`` /
``run_shard_eval_pull``. Where those drive a pull-mode SQLite-backed
master queue, this module drives the v2 REST claim/play/report loop
one game at a time.

Key shape differences from the SSH-era shards:

- The deck pool is loaded once at setup time from a fixed path; the
  per-game ``deck_indices`` from the ``/v2/claim`` response select two
  archetypes from that pool (rather than re-running the legacy
  ``_sample_decks_for_game`` per-game-seed sampler). This is the
  authoritative deck selection — the coordinator's stage spec is the
  source of truth.
- Weights are fetched lazily on first claim that needs them, then
  cached in memory by version. A version change between claims (rare
  in self-play, normal in eval where p0=new and p1=old) re-runs
  ``fetch_weights_if_missing`` and rebuilds the NN evaluator.
- The runner's per-call body (``_play_one``) is synchronous in the
  calling thread. The :class:`ClaimLoopRunner` spawns ``n_workers``
  parallel claim-loop threads, all of which dispatch through this same
  runner instance — so the deck-pool load and the per-version weights
  cache are guarded by a single ``threading.Lock`` here. The actual
  game-execution body (``_generate_one_game_data`` / NN-PIMC) holds no
  shared mutable state across threads beyond the cached config dict,
  which is read-only after first build.
- No SHM NN eval server. At the small per-game volume this runner sees
  (1 in flight), a per-call ``NNEvaluator.forward`` is fine; the SHM
  fan-out was a benefit of the pool-with-workers shape that we now
  flatten. The SHM path stays available for the master's in-process
  iterate runs via ``run_shard_local``.

The npz file the SSH shard wrote to disk is here returned as raw bytes
in :attr:`GameResult.npz_bytes` and shipped over multipart in the
``/v2/report_result`` POST. The coordinator's ``record_report`` writes
it under ``iterate_data/<stage>_<shard>_<safe_claim>.npz`` — same
naming the SSH writer used, so downstream train / UI keep working
unchanged.
"""

from __future__ import annotations

from mtg_bot.search.leaf_batch import SEARCH_LEAF_BATCH_SIZE

import concurrent.futures
import io
import logging
import multiprocessing
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from mtg_bot.worker_daemon_claim_loop import (
    CallableGameRunner,
    GameResult,
    GameRunner,
    _http_get,
    fetch_weights_if_missing,
)


log = logging.getLogger(__name__)


# Default deck pool path. Matches the path that
# ``mtg_bot iterate`` / ``mtg_bot eval`` / the SSH shard runner have
# used since the synthetic mono-color G/R fallback was retired.
DEFAULT_DECK_POOL_PATH = "data/17lands/eoe_decks.json.gz"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_deck_pool_path(explicit: Optional[str]) -> Path:
    """Resolve the deck pool path. Honors explicit override; falls
    back to the system-wide default.
    """
    if explicit:
        return Path(explicit)
    return Path(DEFAULT_DECK_POOL_PATH)


def _read_metadata_json(weights_dir: Path) -> dict[str, Any]:
    """Read the version's ``metadata.json`` sidecar. Returns an empty
    dict on failure so callers fall back to ``NNEvaluator`` defaults.
    """
    import json

    meta_path = weights_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _build_nn_pimc_config(
    weights_path: Path,
    metadata: dict[str, Any],
    *,
    device: Optional[str] = None,
):
    """Construct an ``NNPimcConfig`` from the version's metadata
    sidecar. Mirrors the architecture / policy / action-encoding
    auto-discovery the iterate orchestrator does today.

    ``device`` is the worker-owned NN-eval device selector, threaded
    in from the daemon's ``--nn-device`` CLI flag. It is passed
    straight through to :attr:`NNPimcConfig.device` WITHOUT being
    resolved here — the daemon parent / claim-loop thread must never
    call ``auto_device()`` / ``torch.cuda.is_available()`` before the
    subprocess pool spawns, or per-worker CUDA-context init under
    Windows-spawn (deadlock) / Linux-fork ("Cannot re-initialize CUDA
    in forked subprocess") breaks. This is exactly the constraint the
    ``device`` field docstring in ``mtg_bot.training.config`` and the
    CLAUDE.md note describe; the resolution mechanism mirrors the
    proven one used by the distributed shard runners and the
    ``per_game_wall`` bench (thread an ``"auto"`` sentinel, resolve
    inside ``NNPimcConfig.build()`` which runs in the worker process).
    Accepted values mirror the ``NNPimcConfig.device`` field exactly:

    - ``None`` (default; also what ``--nn-device cpu`` resolves to) —
      ``NNEvaluator``'s CPU default. Byte-identical to the pre-flag
      behavior; the entire existing test suite and every non-GPU
      worker are unaffected.
    - ``"auto"`` — deferred sentinel. ``NNPimcConfig.build()`` (which
      runs INSIDE the PPE subprocess, not the daemon parent) calls
      ``auto_device()`` there to pick ``cuda`` / ``mps`` / ``cpu``
      per-host.
    - ``"cuda"`` — pin CUDA; still instantiated per-worker inside
      ``build()`` for the same fork/spawn-safety reason.
    """
    from mtg_bot.training import NNPimcConfig

    training = metadata.get("training") or {}
    # Schema reminder: `architecture`, `hidden_sizes`,
    # `policy_hidden_sizes`, `with_policy`, and `action_encoding_version`
    # are at the TOP LEVEL of metadata.json (where every registered
    # version v5+ writes them). Fall back to `training.*` for forward-
    # compat with any future trainer that nests them.
    hs = metadata.get("hidden_sizes") or training.get("hidden_sizes")
    phs = metadata.get("policy_hidden_sizes") or training.get(
        "policy_hidden_sizes"
    )
    architecture = (
        metadata.get("architecture") or training.get("architecture")
    )
    action_encoding_version = metadata.get("action_encoding_version")
    with_policy = bool(
        metadata.get("with_policy", training.get("with_policy", True))
    )

    return NNPimcConfig(
        weights_path=str(weights_path),
        samples=4,  # Default; overridden per-claim if claim carries `samples`.
        inner_nodes=200,  # Default; overridden per-claim.
        hidden_sizes=tuple(hs) if hs else None,
        architecture=architecture,
        policy_hidden_sizes=tuple(phs) if phs else None,
        action_encoding_version=action_encoding_version,
        with_policy=with_policy,
        puct=with_policy,  # PUCT requires a policy head.
        batch_size=SEARCH_LEAF_BATCH_SIZE,
        # ``device`` is threaded through verbatim from the daemon's
        # ``--nn-device`` flag (default ``None`` -> NNEvaluator's CPU
        # default, byte-identical to the pre-flag behavior). It is NOT
        # resolved here: an ``"auto"`` / ``"cuda"`` value is carried as
        # a string into ``NNPimcConfig.build()``, which runs in the PPE
        # subprocess and is the only place that imports
        # ``auto_device()`` / touches the CUDA driver — exactly the
        # established run_shard / per_game_wall-bench pattern. The
        # daemon parent never primes CUDA, so Windows-spawn /
        # Linux-fork per-worker context init stays safe.
        device=device,
    )


def _heuristic_self_play_plan(claim: dict[str, Any]) -> dict[str, Any]:
    """Build the per-claim plan for a HEURISTIC self-play game.

    Returns ``{factory, action_encoding_version, structured_tokens,
    structured_scalar_schema_version, nodes, samples, version}``.

    A heuristic stage carries no weights version, so there is no
    ``metadata.json`` on the worker to discover the corpus format
    from: the coordinator's stage spec is authoritative for the
    action encoding and the structured-token schema, and a spec that
    omits ``action_encoding_version`` is rejected here rather than
    silently defaulting (a mislabelled policy target is invisible
    downstream). See ``mtg_bot.training.self_play_evaluator``.
    """
    from mtg_bot.training.self_play_evaluator import (
        build_heuristic_pimc_config,
    )
    action_encoding_version = claim.get("action_encoding_version")
    if not action_encoding_version:
        raise RuntimeError(
            "heuristic self-play claim is missing "
            f"action_encoding_version: claim={claim}. The coordinator "
            "must set it on the stage spec -- a heuristic stage has "
            "no weights metadata.json to discover it from."
        )
    nodes = int(claim.get("nodes") or 200)
    samples = int(claim.get("samples") or 4)
    knobs = _self_play_exploration_knobs(claim)
    cfg = build_heuristic_pimc_config(
        nodes=nodes,
        samples=samples,
        root_dirichlet_alpha=knobs["root_dirichlet_alpha"],
        root_dirichlet_eps=knobs["root_dirichlet_eps"],
    )
    schema = claim.get("structured_scalar_schema_version")
    return {
        "factory": cfg.build,
        "action_encoding_version": str(action_encoding_version),
        "structured_tokens": bool(claim.get("structured_tokens")),
        "structured_scalar_schema_version": (
            int(schema) if schema is not None else None
        ),
        "nodes": nodes,
        "samples": samples,
        "version": "heuristic",
    }


def _run_heuristic_self_play_game(
    *,
    claim: dict[str, Any],
    deck_pool,
    t0: float,
) -> dict[str, Any]:
    """Play one heuristic self-play game and return the result payload.

    Shared by the subprocess body and the in-thread runner so the two
    paths can't drift. The payload shape matches
    :func:`_subprocess_play_self_play`'s NN return value exactly, so
    the claim loop / GameResult decoding is unchanged.
    """
    plan = _heuristic_self_play_plan(claim)
    knobs = _self_play_exploration_knobs(claim)
    per_game_timeout_sec = claim.get("per_game_timeout_sec")
    if per_game_timeout_sec is not None:
        per_game_timeout_sec = float(per_game_timeout_sec)
    chunk, outcome_kind, winner, n_samples = _play_self_play_game(
        factory=plan["factory"],
        base_seed=int(claim.get("seed") or 0),
        game_idx=int(claim.get("game_idx") or 0),
        deck_pool=deck_pool,
        deck_indices=list(claim.get("deck_indices") or [0, 0]),
        per_game_timeout_sec=per_game_timeout_sec,
        capture_every=int(claim.get("capture_every") or 0),
        action_encoding_version=plan["action_encoding_version"],
        write_structured_tokens=plan["structured_tokens"],
        structured_scalar_schema_version=(
            plan["structured_scalar_schema_version"]
        ),
        move_temperature=knobs["move_temperature"],
        temperature_cutoff=knobs["temperature_cutoff"],
    )
    npz_bytes = _encode_npz_bytes(
        chunk, write_structured_tokens=plan["structured_tokens"],
    )
    return {
        "outcome": outcome_kind,
        "winner": winner,
        "n_samples": n_samples,
        "npz_bytes": npz_bytes,
        "duration_s": time.perf_counter() - t0,
        "game_idx": int(claim.get("game_idx") or 0),
        "version": plan["version"],
        "nodes": plan["nodes"],
        "samples": plan["samples"],
    }


def _self_play_exploration_knobs(claim: dict[str, Any]) -> dict[str, Any]:
    """Pull the self-play exploration knobs off a coordinator claim.

    Keys: ``root_dirichlet_alpha`` (``None`` = off), ``root_dirichlet_eps``,
    ``move_temperature``, ``temperature_cutoff``. Claims from an older
    coordinator that never set them fall back to data_gen's defaults
    (no root noise; T=1.0 for the first 30 non-forced decisions).
    Only the SELF-PLAY handlers read these; eval claims are ignored
    so the promotion gate stays deterministic-argmax.
    """
    alpha = claim.get("root_dirichlet_alpha")
    alpha = float(alpha) if alpha is not None else None
    if alpha is not None and alpha <= 0:
        alpha = None
    eps = claim.get("root_dirichlet_eps")
    temperature = claim.get("move_temperature")
    cutoff = claim.get("temperature_cutoff")
    return {
        "root_dirichlet_alpha": alpha,
        "root_dirichlet_eps": float(eps) if eps is not None else 0.25,
        "move_temperature": (
            float(temperature) if temperature is not None else 1.0
        ),
        "temperature_cutoff": int(cutoff) if cutoff is not None else 30,
    }


def _encode_npz_bytes(
    chunk: tuple, write_structured_tokens: bool = False
) -> bytes:
    """Serialize one game's chunk to npz-format bytes. Mirrors the
    SSH shard's ``_flush`` writer, but to a BytesIO instead of an
    output_dir path.

    ``chunk`` is the 8-tuple returned by
    ``_play_one_parallel_game``'s success path:
    ``(features, outcomes, policy_targets, action_masks, winner,
       outcome_kind, struct_rows, game_ids)``. Empty arrays for crashed /
    timed-out games encode cleanly (zero-row npz). ``game_ids`` (one
    value per row = the claim's ``game_idx``) is written under the
    ``game_ids`` key so the trainer's game-level train/val split sees
    the same schema as the local data_gen path; a legacy 7-tuple
    chunk gets ``-1`` per row.
    """
    import numpy as np

    from mtg_bot.training import save_dataset
    from mtg_bot.training.data_gen import _chunk_game_ids

    features = chunk[0]
    outcomes = chunk[1]
    policy_targets = chunk[2]
    action_masks = chunk[3]
    game_ids = _chunk_game_ids(chunk)
    # Write to a temp file under a BytesIO since np.savez_compressed
    # wants a file-like or path. BytesIO works because savez_compressed
    # internally uses the zipfile module which only needs seek/write.
    buf = io.BytesIO()
    if write_structured_tokens:
        # Empty struct_rows (timed-out / crashed games whose chunk
        # came from ``_empty_arrays()``) used to fall through to the
        # flat-only ``else`` branch via the
        # ``... and chunk[6]`` truthiness check, leaving the npz
        # without the structured-token columns. train_nn.py then
        # rejected the ENTIRE batch on the first such file — even
        # though the empty game contributes no samples — because
        # the schema must be uniform across every npz when
        # ``--architecture structured``. Always emit the structured
        # columns when the flag is on; ``_pad_and_stack_structured_rows([])``
        # returns shape-(0,0) arrays that round-trip cleanly.
        from mtg_bot.training.data_gen import _pad_and_stack_structured_rows
        struct_rows = chunk[6] if len(chunk) > 6 else []
        struct = _pad_and_stack_structured_rows(struct_rows)
        arrays = {
            "features": features,
            "outcomes": outcomes,
            "policy_targets": policy_targets,
            "action_masks": action_masks,
            "structured_token_type": struct["structured_token_type"],
            "structured_token_emb_ids": struct["structured_token_emb_ids"],
            "structured_token_scalars": struct["structured_token_scalars"],
            "structured_attention_mask": struct["structured_attention_mask"],
            # Search root value per row (NaN when unknown) -- the
            # dense value target. Always present on structured shards
            # written after 2026-09-26; older shards lack the key.
            "root_values": struct["root_values"],
            "game_ids": game_ids,
        }
        # Stamp the id space this worker tokenised in (its card
        # vocabulary + bucketing cap). A worker running a different
        # card pool / parser than its peers produces a different stamp,
        # and the trainer refuses to mix the two instead of training on
        # scrambled printing ids.
        from mtg_bot.training.data_gen import shard_card_vocab_entries

        arrays.update(shard_card_vocab_entries())
    else:
        arrays = {
            "features": features,
            "outcomes": outcomes,
            "policy_targets": policy_targets,
            "action_masks": action_masks,
            "game_ids": game_ids,
        }
    # Resolution-supervision stamp (``chunk[8]``, 2026-10-06+): the
    # action-encoding version the policy columns were written under
    # and the game's resolution-row counters, so the trainer can tell
    # an 805-wide canonical_v3 shard from a legacy 517-wide one and
    # log how much resolution supervision the window carries.
    from mtg_bot.training.data_gen import resolution_stats_arrays

    stats = chunk[8] if len(chunk) > 8 else None
    arrays.update(resolution_stats_arrays(stats))
    if stats:
        log.info(
            "[self_play_runner] game resolution supervision: %d rows "
            "from %d sampled pauses, %d overflow visits (%s)",
            int(stats.get("resolution_policy_rows", 0)),
            int(stats.get("resolution_pause_samples", 0)),
            int(stats.get("resolution_overflow_visits", 0)),
            stats.get("action_encoding_version"),
        )
    # Drop empty rows (crashed / timed-out games): save_dataset would
    # raise on FEATURE_DIM mismatch when features is a (0, FEATURE_DIM)
    # array. The save_dataset path validates; for raw bytes we just
    # round-trip via numpy.
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _policy_width(action_encoding_version: Optional[str]) -> int:
    """Policy-column width of a self-play shard (517 for v1 / v2, 805
    for canonical_v3) -- also for the zero-row npz of a crashed /
    timed-out game, so a window's files agree on it."""
    from mtg_bot.training.action_encoding import action_vocab_size

    return action_vocab_size(action_encoding_version or "positional_v1")


def _play_self_play_game(
    *,
    factory,
    base_seed: int,
    game_idx: int,
    deck_pool,
    deck_indices: list[int],
    per_game_timeout_sec: Optional[float],
    capture_every: int,
    action_encoding_version: Optional[str],
    write_structured_tokens: bool,
    structured_scalar_schema_version: Optional[int] = None,
    move_temperature: float = 1.0,
    temperature_cutoff: int = 30,
) -> tuple[tuple, str, Optional[int], int]:
    """Run one self-play game synchronously in this thread.

    ``structured_scalar_schema_version`` picks the ``token_scalars``
    row layout for the structured-token columns; ``None`` = the
    tokenizer's current default schema (what a fresh ``mtg_bot train
    --architecture structured`` expects). The v2 stage spec does not
    carry this field yet, so production workers always emit the
    default.

    Returns ``(chunk, outcome_kind, winner, n_samples)``.

    The deck pool is indexed by the claim's ``deck_indices`` rather
    than re-running the legacy ``_sample_decks_for_game`` seed-based
    sampler — the coordinator's stage spec is authoritative for which
    archetypes each game runs.
    """
    from mtg_bot.training.data_gen import (
        _empty_arrays,
        _empty_game_ids,
        _generate_one_game_data,
    )

    # Honor the claim's deck_indices. If the pool is smaller than the
    # claim requested, fall back to modulo-wrap rather than crashing —
    # tiny test pools can still drive an end-to-end smoke.
    n = len(deck_pool)
    if n == 0:
        raise RuntimeError("deck pool is empty; cannot play game")
    a_idx = int(deck_indices[0]) % n
    b_idx = int(deck_indices[1]) % n if len(deck_indices) > 1 else a_idx
    selected_pool = [deck_pool[a_idx], deck_pool[b_idx]]

    try:
        chunk = _generate_one_game_data(
            game_idx=game_idx,
            p0_factory=factory,
            p1_factory=factory,
            base_seed=base_seed,
            decision_cap=4000,
            sample_every=1,
            replay_config=None,
            deck_pool=selected_pool,
            per_game_timeout_sec=per_game_timeout_sec,
            hang_dump_dir=None,
            heuristics=None,
            action_encoding_version=(
                action_encoding_version or "positional_v1"
            ),
            structured_tokens=write_structured_tokens,
            structured_scalar_schema_version=(
                structured_scalar_schema_version
            ),
            move_temperature=move_temperature,
            temperature_cutoff=temperature_cutoff,
        )
    except TimeoutError:
        f, o, p, m = _empty_arrays(_policy_width(action_encoding_version))
        chunk = (f, o, p, m, None, "timed_out", [], _empty_game_ids())
    except Exception:  # noqa: BLE001 - intentional broad catch
        log.exception(
            "[self_play_runner] game %d crashed; dropping", game_idx
        )
        f, o, p, m = _empty_arrays(_policy_width(action_encoding_version))
        chunk = (f, o, p, m, None, "crashed", [], _empty_game_ids())

    # ``chunk`` is now an 8-tuple ``(features, outcomes, policy_targets,
    # action_masks, winner, outcome_kind, struct_rows, game_ids)``.
    winner = chunk[4]
    outcome_kind = chunk[5] if len(chunk) > 5 else (
        "crashed" if winner is None and not bool(getattr(chunk[0], "size", 0))
        else "completed"
    )
    n_samples = int(chunk[0].shape[0]) if hasattr(chunk[0], "shape") else 0
    return chunk, outcome_kind, winner, n_samples


def _play_eval_game(
    *,
    p0_factory,
    p1_factory,
    base_seed: int,
    game_idx: int,
    deck_pool,
    deck_indices: list[int],
    per_game_timeout_sec: Optional[float],
) -> tuple[Optional[int], int, str]:
    """Run one eval game synchronously in this thread.

    Returns ``(winner_remapped, decisions, outcome_kind)`` where
    ``winner_remapped == 0`` always means "the original ``new`` seat
    won" and ``1`` means "the original ``old`` seat won" — the
    seat-alternation remap from
    ``_play_one_match_game_alternating_indexed``.
    """
    from mtg_bot.training.self_play import _play_one_match_game

    n = len(deck_pool)
    if n == 0:
        raise RuntimeError("deck pool is empty; cannot play eval game")
    a_idx = int(deck_indices[0]) % n
    b_idx = int(deck_indices[1]) % n if len(deck_indices) > 1 else a_idx
    selected_pool = [deck_pool[a_idx], deck_pool[b_idx]]

    # Alternate seat on odd game_idx (mirror
    # _play_one_match_game_alternating_indexed). Even game_idx: p0=new
    # plays seat 0. Odd: factories swap; the returned winner is
    # remapped so 0=>"new won" regardless of seating.
    if game_idx % 2 == 0:
        p0, p1 = p0_factory, p1_factory
    else:
        p0, p1 = p1_factory, p0_factory

    try:
        winner, decisions = _play_one_match_game(
            game_idx=game_idx,
            p0_factory=p0,
            p1_factory=p1,
            base_seed=base_seed,
            decision_cap=4000,
            per_game_timeout_sec=per_game_timeout_sec,
            deck_pool=selected_pool,
            replay_config=None,
            hang_dump_dir=None,
            heuristics=None,
            heartbeat_path=None,
        )
        outcome_kind = "completed"
    except TimeoutError as exc:
        partial = int(getattr(exc, "decisions_completed", 0) or 0)
        return None, partial, "timed_out"
    except Exception:  # noqa: BLE001 - intentional broad catch
        log.exception(
            "[eval_runner] game %d crashed; dropping", game_idx
        )
        return None, 0, "crashed"

    if game_idx % 2 == 1 and winner is not None:
        winner = 1 - winner
    return winner, decisions, outcome_kind


# ---------------------------------------------------------------------------
# Subprocess game executors (Phase 2 — true multi-core concurrency).
#
# These two functions are the body of a single game; they're top-level
# (not closures) so they pickle cleanly into a
# :class:`concurrent.futures.ProcessPoolExecutor` worker. The runner
# factory below submits one of these per claim when
# ``use_subprocess_pool=True``; each PPE worker has its own Python
# interpreter (and its own GIL), so N parallel games on a single daemon
# run truly concurrently instead of GIL-serialized.
#
# Module-level state caches the deck pool + per-version
# :class:`NNPimcConfig` so successive submits to the same persistent
# subprocess reuse the heavy startup work (deck pool decode ~1.5 GB,
# state_dict load ~10 MB, CUDA context ~1.5 GB). Each subprocess is its
# own interpreter, so these globals are isolated from the parent and
# from peer workers.
# ---------------------------------------------------------------------------


_SUBPROCESS_STATE: dict[str, Any] = {}


def _subprocess_play_self_play(
    *,
    claim: dict[str, Any],
    coordinator_url: str,
    weights_cache_root: str,
    deck_pool_path: Optional[str],
    write_structured_tokens: bool = False,
    nn_device: Optional[str] = None,
) -> dict[str, Any]:
    """Body of one self-play game, executed in a PPE subprocess.

    All args must be picklable. Returns a picklable dict the parent
    decodes into a :class:`GameResult`.

    ``nn_device`` is the picklable device-selector string forwarded
    from the daemon's ``--nn-device`` flag (``None`` / ``"auto"`` /
    ``"cuda"``). It is passed straight to ``_build_nn_pimc_config``;
    the actual ``auto_device()`` resolution / CUDA-context init
    happens HERE in the subprocess (via ``NNPimcConfig.build()``),
    never in the daemon parent.
    """
    t0 = time.perf_counter()

    # Lazy deck-pool load (per-subprocess; persists across submits).
    if _SUBPROCESS_STATE.get("deck_pool") is None:
        path = _resolve_deck_pool_path(deck_pool_path)
        if not path.is_file():
            raise RuntimeError(
                f"deck pool not found at {path} (subprocess); pass "
                f"deck_pool_path via worker daemon --deck-pool-path."
            )
        from mtg_bot.decks import decks_from_json
        _SUBPROCESS_STATE["deck_pool"] = decks_from_json(path)
    deck_pool = _SUBPROCESS_STATE["deck_pool"]

    # Heuristic (warm-start corpus) stages never touch the registry:
    # no weights fetch, no torch import, no metadata discovery. Handle
    # them before the NN path so a missing/empty weights block is a
    # non-event rather than the RuntimeError below.
    from mtg_bot.training.self_play_evaluator import (
        EVALUATOR_HEURISTIC, claim_evaluator,
    )
    if claim_evaluator(claim) == EVALUATOR_HEURISTIC:
        return _run_heuristic_self_play_game(
            claim=claim, deck_pool=deck_pool, t0=t0,
        )

    # Lazy per-version weights + NNPimcConfig (persists across submits).
    configs = _SUBPROCESS_STATE.setdefault("configs_by_version", {})
    weights = claim.get("weights") or {}
    version = weights.get("p0") or weights.get("new")
    if not version:
        raise RuntimeError(
            f"self-play claim missing weights.p0: claim={claim}"
        )
    cfg = configs.get(version)
    if cfg is None:
        weights_dir = fetch_weights_if_missing(
            coordinator_url=coordinator_url,
            version=version,
            weights_root=Path(weights_cache_root),
        )
        metadata = _read_metadata_json(weights_dir)
        cfg = _build_nn_pimc_config(
            weights_dir / "weights.pt", metadata, device=nn_device,
        )
        configs[version] = cfg

    nodes = int(claim.get("nodes") or cfg.inner_nodes)
    samples = int(claim.get("samples") or cfg.samples)
    per_game_timeout_sec = claim.get("per_game_timeout_sec")
    if per_game_timeout_sec is not None:
        per_game_timeout_sec = float(per_game_timeout_sec)
    capture_every = int(claim.get("capture_every") or 0)
    seed = int(claim.get("seed") or 0)
    game_idx = int(claim.get("game_idx") or 0)
    deck_indices = list(claim.get("deck_indices") or [0, 0])
    knobs = _self_play_exploration_knobs(claim)

    from mtg_bot.training import NNPimcConfig
    per_game_cfg = NNPimcConfig(
        weights_path=cfg.weights_path,
        samples=samples,
        inner_nodes=nodes,
        hidden_sizes=cfg.hidden_sizes,
        architecture=cfg.architecture,
        policy_hidden_sizes=cfg.policy_hidden_sizes,
        action_encoding_version=cfg.action_encoding_version,
        with_policy=cfg.with_policy,
        puct=cfg.puct,
        batch_size=cfg.batch_size,
        device=cfg.device,
        root_dirichlet_alpha=knobs["root_dirichlet_alpha"],
        root_dirichlet_eps=knobs["root_dirichlet_eps"],
    )
    factory = per_game_cfg.build

    # The structured trainer demands structured-token columns; the
    # flat-feature trainer ignores them. The caller's
    # ``write_structured_tokens`` flag was hardcoded ``False`` at the
    # ``_play_one`` call sites, which silently dropped the columns
    # for every structured-net self_play stage and broke the
    # downstream train_nn.py with a "missing structured-token
    # arrays" error. Derive from cfg.architecture (the network's
    # actual shape, read from its metadata.json on first claim) so
    # the npzs always match what the trainer expects, regardless of
    # what the caller passed.
    effective_structured = (
        write_structured_tokens or cfg.architecture == "structured"
    )
    chunk, outcome_kind, winner, n_samples = _play_self_play_game(
        factory=factory,
        base_seed=seed,
        game_idx=game_idx,
        deck_pool=deck_pool,
        deck_indices=deck_indices,
        per_game_timeout_sec=per_game_timeout_sec,
        capture_every=capture_every,
        action_encoding_version=cfg.action_encoding_version,
        write_structured_tokens=effective_structured,
        move_temperature=knobs["move_temperature"],
        temperature_cutoff=knobs["temperature_cutoff"],
    )
    npz_bytes = _encode_npz_bytes(
        chunk, write_structured_tokens=effective_structured,
    )
    duration_s = time.perf_counter() - t0
    return {
        "outcome": outcome_kind,
        "winner": winner,
        "n_samples": n_samples,
        "npz_bytes": npz_bytes,
        "duration_s": duration_s,
        "game_idx": game_idx,
        "version": version,
        "nodes": nodes,
        "samples": samples,
    }


def _subprocess_play_eval(
    *,
    claim: dict[str, Any],
    coordinator_url: str,
    weights_cache_root: str,
    deck_pool_path: Optional[str],
    nn_device: Optional[str] = None,
) -> dict[str, Any]:
    """Body of one eval game, executed in a PPE subprocess. Mirror of
    :func:`_subprocess_play_self_play` but loads two NN versions
    (``new`` and ``old``) and routes through
    :func:`_play_eval_game` for the alternation/remap.

    ``nn_device`` carries the daemon's ``--nn-device`` selector; same
    deferred-resolution contract as the self-play subprocess body.
    """
    t0 = time.perf_counter()

    if _SUBPROCESS_STATE.get("deck_pool") is None:
        path = _resolve_deck_pool_path(deck_pool_path)
        if not path.is_file():
            raise RuntimeError(
                f"deck pool not found at {path} (subprocess); pass "
                f"deck_pool_path via worker daemon --deck-pool-path."
            )
        from mtg_bot.decks import decks_from_json
        _SUBPROCESS_STATE["deck_pool"] = decks_from_json(path)
    deck_pool = _SUBPROCESS_STATE["deck_pool"]

    configs = _SUBPROCESS_STATE.setdefault("configs_by_version", {})

    def _ensure_cfg(version: str):
        cfg = configs.get(version)
        if cfg is not None:
            return cfg
        weights_dir = fetch_weights_if_missing(
            coordinator_url=coordinator_url,
            version=version,
            weights_root=Path(weights_cache_root),
        )
        metadata = _read_metadata_json(weights_dir)
        cfg = _build_nn_pimc_config(
            weights_dir / "weights.pt", metadata, device=nn_device,
        )
        configs[version] = cfg
        return cfg

    weights = claim.get("weights") or {}
    new_version = weights.get("new") or weights.get("p0")
    old_version = weights.get("old") or weights.get("p1")
    if not new_version or not old_version:
        raise RuntimeError(
            f"eval claim missing weights.new / weights.old: claim={claim}"
        )

    new_cfg_base = _ensure_cfg(new_version)
    old_cfg_base = _ensure_cfg(old_version)

    nodes = int(claim.get("nodes") or new_cfg_base.inner_nodes)
    samples = int(claim.get("samples") or new_cfg_base.samples)
    per_game_timeout_sec = claim.get("per_game_timeout_sec")
    if per_game_timeout_sec is not None:
        per_game_timeout_sec = float(per_game_timeout_sec)
    seed = int(claim.get("seed") or 0)
    game_idx = int(claim.get("game_idx") or 0)
    deck_indices = list(claim.get("deck_indices") or [0, 0])

    from mtg_bot.training import NNPimcConfig

    def _per_game(base):
        return NNPimcConfig(
            weights_path=base.weights_path,
            samples=samples,
            inner_nodes=nodes,
            hidden_sizes=base.hidden_sizes,
            architecture=base.architecture,
            policy_hidden_sizes=base.policy_hidden_sizes,
            action_encoding_version=base.action_encoding_version,
            with_policy=base.with_policy,
            puct=base.puct,
            batch_size=base.batch_size,
            device=base.device,
        )

    p0_factory = _per_game(new_cfg_base).build
    p1_factory = _per_game(old_cfg_base).build

    winner, decisions, outcome_kind = _play_eval_game(
        p0_factory=p0_factory,
        p1_factory=p1_factory,
        base_seed=seed,
        game_idx=game_idx,
        deck_pool=deck_pool,
        deck_indices=deck_indices,
        per_game_timeout_sec=per_game_timeout_sec,
    )
    duration_s = time.perf_counter() - t0
    return {
        "outcome": outcome_kind,
        "winner": winner,
        "n_samples": int(decisions),
        "duration_s": duration_s,
        "game_idx": game_idx,
        "new_version": new_version,
        "old_version": old_version,
        "nodes": nodes,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Self-play GameRunner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Process-pool lifecycle
# ---------------------------------------------------------------------------


class _PoolControls:
    """Owns one runner's :class:`ProcessPoolExecutor` across rebuilds.

    Shared verbatim by the self-play and eval factories.

    Why this exists: a ``ProcessPoolExecutor`` whose worker process
    dies hard (the documented CPython-3.13 interpreter-corruption
    class; seen twice on 2026-09-25 at roughly one death per
    1,200-1,500 games) is *permanently* broken -- every later
    ``submit()`` raises ``BrokenProcessPool`` immediately. The only
    recovery is a fresh executor.

    Concurrency: N claim-loop threads share one pool, so all N see the
    same death within milliseconds. Each reads :meth:`generation`
    BEFORE its game and passes that value to :meth:`reset`; the first
    caller through rebuilds and bumps the generation, and the peers'
    stale generations make their calls no-ops. Threads still blocked
    on a future from the old executor fail over on their own next
    iteration -- their future resolves (or raises) against the old
    object, which nothing else references once we swap.
    """

    def __init__(
        self,
        state: dict[str, Any],
        *,
        enabled: bool,
        emit: Callable[[str, dict], None],
        label: str,
    ) -> None:
        self._state = state
        self._enabled = bool(enabled)
        self._emit = emit
        self._label = label
        self._lock = threading.Lock()
        self._generation = 0
        self._max_workers = 1

    def _new_pool(self) -> "concurrent.futures.ProcessPoolExecutor":
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def ensure(self, n_workers: int) -> None:
        """Create the pool if this runner uses one and none is live."""
        if not self._enabled:
            return
        with self._lock:
            self._max_workers = max(1, int(n_workers))
            if self._state.get("pool") is not None:
                return
            self._state["pool"] = self._new_pool()
            self._generation += 1
            self._emit(
                "pool_created",
                {
                    "max_workers": self._max_workers,
                    "mode": "subprocess",
                    "generation": self._generation,
                },
            )

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def reset(self, generation: Optional[int] = None) -> int:
        """Rebuild the pool. Returns the live generation.

        A return value equal to the caller's ``generation`` means
        nothing was rebuilt (pool-less runner, a peer already rebuilt,
        or the runner is torn down).
        """
        with self._lock:
            if not self._enabled:
                return self._generation
            if (
                generation is not None
                and int(generation) != self._generation
            ):
                # A peer thread already rebuilt after the same death.
                return self._generation
            old = self._state.get("pool")
            if old is None:
                # Torn down (or never set up) -- do NOT resurrect a
                # pool behind teardown's back.
                return self._generation
            try:
                old.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001 - best-effort reaping
                log.exception(
                    "[%s] shutdown of the broken pool raised", self._label,
                )
            self._state["pool"] = self._new_pool()
            self._generation += 1
            self._emit(
                "pool_reset",
                {
                    "max_workers": self._max_workers,
                    "mode": "subprocess",
                    "generation": self._generation,
                },
            )
            log.warning(
                "[%s] rebuilt the subprocess pool (generation %d, "
                "max_workers=%d)",
                self._label, self._generation, self._max_workers,
            )
            return self._generation

    def shutdown(self, *, wait: bool) -> None:
        with self._lock:
            pool = self._state.get("pool")
            if pool is None:
                return
            pool.shutdown(wait=wait, cancel_futures=True)
            self._state["pool"] = None


def build_self_play_runner(
    coordinator_url: str,
    stage_id: str,
    n_workers: int,
    weights_cache_root: Path,
    worker_id: str,
    log_event: Optional[Callable[[str, dict], None]] = None,
    *,
    deck_pool_path: Optional[str] = None,
    nn_device: Optional[str] = None,
    http_get: Callable[..., tuple[int, bytes, dict[str, str]]] = _http_get,
    use_subprocess_pool: bool = False,
) -> GameRunner:
    """Production self-play :class:`GameRunner`.

    Per-claim flow:

    1. Resolve ``weights["p0"]`` from the claim payload. (Self-play
       stages always carry ``p0 == p1`` per the API contract, so we
       only build one evaluator.)
    2. Fetch the weights blob + metadata sidecar from
       ``<coordinator_url>/v2/weights/<version>``; cache by version
       under ``weights_cache_root``.
    3. Build (or reuse cached) ``NNPimcConfig`` from the metadata
       sidecar — architecture / hidden_sizes / policy_hidden_sizes /
       action_encoding_version all auto-discover.
    4. Override per-claim knobs (``nodes``, ``samples``) on the cached
       config.
    5. Run one game via ``_generate_one_game_data`` — either synchronously
       in the caller's thread (default) or dispatched to a
       :class:`concurrent.futures.ProcessPoolExecutor` of ``n_workers``
       subprocesses (when ``use_subprocess_pool=True``) so the Python
       GIL doesn't serialize parallel claim-loop threads.
    6. Encode the per-game chunk to npz bytes; return :class:`GameResult`.

    ``use_subprocess_pool``:
        When True, ``_setup`` spawns a :class:`ProcessPoolExecutor` with
        ``max_workers=n_workers`` (spawn context, Windows-safe). Each
        ``_play_one`` invocation submits the game body to the pool and
        blocks on the future. Subprocess persistence means the deck
        pool + NN weights are loaded once per subprocess (lazily on
        first task), then reused across that subprocess's lifetime.

        Default is ``False`` for back-compat with the existing tests
        that monkeypatch internals — those mocks don't propagate to
        subprocesses, so tests stay on the in-thread path.

    ``nn_device``:
        Worker-owned NN-eval device selector from the daemon's
        ``--nn-device`` flag. ``None`` (default; ``--nn-device cpu``)
        keeps the historical CPU path byte-identical. ``"auto"`` /
        ``"cuda"`` are carried unresolved into ``NNPimcConfig`` and
        only resolved per-worker in ``NNPimcConfig.build()`` — this
        builder, ``_setup``, and the daemon parent never call
        ``auto_device()`` / ``torch.cuda.is_available()``, so the
        subprocess pool's per-worker CUDA-context init stays
        Windows-spawn / Linux-fork-safe (the same proven pattern the
        distributed shard runners + ``per_game_wall`` bench use).
    """

    # In-memory caches. Multiple claim-loop threads dispatch through
    # this runner concurrently; ``state_lock`` guards the write paths
    # (deck-pool load + per-version config build). Reads after the
    # initial write are race-free because both values are effectively
    # immutable once built.
    state: dict[str, Any] = {
        "deck_pool": None,
        "configs_by_version": {},  # version -> NNPimcConfig
    }
    state_lock = threading.Lock()

    def _emit(name: str, payload: dict) -> None:
        if log_event is not None:
            try:
                log_event(name, payload)
            except Exception:  # noqa: BLE001 - swallow logger failures
                log.exception("[self_play_runner] log_event %s raised", name)

    # Pool lifecycle (create / rebuild-after-death / shut down) lives
    # in its own lock-guarded object so the claim loop can recover from
    # a BrokenProcessPool without racing peer threads.
    pool_controls = _PoolControls(
        state,
        enabled=use_subprocess_pool,
        emit=_emit,
        label="self_play_runner",
    )

    def _ensure_deck_pool() -> None:
        if state["deck_pool"] is not None:
            return
        with state_lock:
            if state["deck_pool"] is not None:  # double-checked
                return
            path = _resolve_deck_pool_path(deck_pool_path)
            if not path.is_file():
                raise RuntimeError(
                    f"deck pool not found at {path}; pass deck_pool_path "
                    f"explicitly via /v2/start_self_play body, or place "
                    f"the canonical pool at {DEFAULT_DECK_POOL_PATH}."
                )
            from mtg_bot.decks import decks_from_json
            state["deck_pool"] = decks_from_json(path)
            _emit(
                "deck_pool_loaded",
                {"path": str(path), "n_decks": len(state["deck_pool"])},
            )

    def _ensure_config(version: str):
        cfg = state["configs_by_version"].get(version)
        if cfg is not None:
            return cfg
        with state_lock:
            cfg = state["configs_by_version"].get(version)  # double-checked
            if cfg is not None:
                return cfg
            weights_dir = fetch_weights_if_missing(
                coordinator_url=coordinator_url,
                version=version,
                weights_root=Path(weights_cache_root),
                http_get=http_get,
            )
            metadata = _read_metadata_json(weights_dir)
            cfg = _build_nn_pimc_config(
                weights_dir / "weights.pt", metadata, device=nn_device,
            )
            state["configs_by_version"][version] = cfg
            _emit(
                "weights_loaded",
                {
                    "version": version,
                    "architecture": cfg.architecture,
                    "action_encoding_version": cfg.action_encoding_version,
                },
            )
            return cfg

    def _setup(coord_url_unused, weights_root_unused, kind, n_workers_at_setup):
        # All lazy — weight load happens on first claim, deck pool is
        # loaded on first game. ``setup`` just records the kind for
        # the assertion below; misuse (calling the self-play factory
        # for an eval stage) surfaces as a clear error rather than
        # silent wrong-runner behavior.
        state["kind"] = kind
        if kind != "self_play":
            log.warning(
                "[self_play_runner] setup called with kind=%s "
                "(expected self_play)", kind,
            )
        # Bring up the subprocess pool if requested. We pull the
        # active n_workers from the claim-loop setup call (which is
        # authoritative — it reflects the daemon's current --n-workers)
        # rather than the factory-time ``n_workers`` arg, which was
        # captured before the daemon had a chance to override.
        pool_controls.ensure(n_workers_at_setup)

    def _play_one(claim: dict, shard_label: str) -> GameResult:
        # Subprocess path: submit to the PPE and decode the result.
        pool = state.get("pool")
        if pool is not None:
            game_idx = int(claim.get("game_idx") or 0)
            _emit(
                "game_start",
                {
                    "claim_id": claim.get("claim_id"),
                    "game_idx": game_idx,
                    "version": (
                        (claim.get("weights") or {}).get("p0")
                        or claim.get("evaluator")
                    ),
                    "evaluator": claim.get("evaluator") or "nn",
                    "mode": "subprocess",
                },
            )
            future = pool.submit(
                _subprocess_play_self_play,
                claim=claim,
                coordinator_url=coordinator_url,
                weights_cache_root=str(weights_cache_root),
                deck_pool_path=deck_pool_path,
                write_structured_tokens=False,
                nn_device=nn_device,
            )
            payload = future.result()  # releases GIL while waiting
            _emit(
                "game_end",
                {
                    "claim_id": claim.get("claim_id"),
                    "game_idx": payload["game_idx"],
                    "outcome": payload["outcome"],
                    "winner": payload["winner"],
                    "n_samples": payload["n_samples"],
                    "duration_s": payload["duration_s"],
                    "mode": "subprocess",
                },
            )
            return GameResult(
                outcome=payload["outcome"],
                winner=payload["winner"],
                n_samples=payload["n_samples"],
                duration_s=payload["duration_s"],
                npz_bytes=payload["npz_bytes"],
            )

        # In-thread fallback (tests, n_workers=1, or
        # ``use_subprocess_pool=False`` builds).
        t0 = time.perf_counter()
        _ensure_deck_pool()

        # Heuristic (warm-start corpus) stage: no weights fetch, no
        # NN. Handled before the weights lookup below so an empty
        # weights block is expected rather than fatal.
        from mtg_bot.training.self_play_evaluator import (
            EVALUATOR_HEURISTIC, claim_evaluator,
        )
        if claim_evaluator(claim) == EVALUATOR_HEURISTIC:
            _emit(
                "game_start",
                {
                    "claim_id": claim.get("claim_id"),
                    "game_idx": int(claim.get("game_idx") or 0),
                    "version": "heuristic",
                    "nodes": claim.get("nodes"),
                    "samples": claim.get("samples"),
                    "evaluator": EVALUATOR_HEURISTIC,
                },
            )
            payload = _run_heuristic_self_play_game(
                claim=claim, deck_pool=state["deck_pool"], t0=t0,
            )
            _emit(
                "game_end",
                {
                    "claim_id": claim.get("claim_id"),
                    "game_idx": payload["game_idx"],
                    "outcome": payload["outcome"],
                    "winner": payload["winner"],
                    "n_samples": payload["n_samples"],
                    "duration_s": payload["duration_s"],
                    "evaluator": EVALUATOR_HEURISTIC,
                },
            )
            return GameResult(
                outcome=payload["outcome"],
                winner=payload["winner"],
                n_samples=payload["n_samples"],
                duration_s=payload["duration_s"],
                npz_bytes=payload["npz_bytes"],
            )

        weights = claim.get("weights") or {}
        version = weights.get("p0") or weights.get("new")
        if not version:
            raise RuntimeError(
                f"self-play claim missing weights.p0: claim={claim}"
            )
        cfg = _ensure_config(version)

        # Apply per-claim search knobs on top of the metadata-derived
        # config. The metadata defaults are sensible (200/4) but the
        # coordinator's stage spec is authoritative.
        nodes = int(claim.get("nodes") or cfg.inner_nodes)
        samples = int(claim.get("samples") or cfg.samples)
        per_game_timeout_sec = claim.get("per_game_timeout_sec")
        if per_game_timeout_sec is not None:
            per_game_timeout_sec = float(per_game_timeout_sec)
        capture_every = int(claim.get("capture_every") or 0)
        seed = int(claim.get("seed") or 0)
        game_idx = int(claim.get("game_idx") or 0)
        deck_indices = list(claim.get("deck_indices") or [0, 0])
        knobs = _self_play_exploration_knobs(claim)

        # Per-claim re-build with overridden samples/inner_nodes; the
        # underlying NNEvaluator is rebuilt fresh each time which costs
        # a (cheap) state_dict load. Reusing the evaluator across games
        # would require pulling it out of ``cfg.build`` — premature
        # optimization until profiling says so.
        from mtg_bot.training import NNPimcConfig
        per_game_cfg = NNPimcConfig(
            weights_path=cfg.weights_path,
            samples=samples,
            inner_nodes=nodes,
            hidden_sizes=cfg.hidden_sizes,
            architecture=cfg.architecture,
            policy_hidden_sizes=cfg.policy_hidden_sizes,
            action_encoding_version=cfg.action_encoding_version,
            with_policy=cfg.with_policy,
            puct=cfg.puct,
            batch_size=cfg.batch_size,
            device=cfg.device,
            root_dirichlet_alpha=knobs["root_dirichlet_alpha"],
            root_dirichlet_eps=knobs["root_dirichlet_eps"],
        )
        factory = per_game_cfg.build

        _emit(
            "game_start",
            {
                "claim_id": claim.get("claim_id"),
                "game_idx": game_idx,
                "version": version,
                "nodes": nodes,
                "samples": samples,
            },
        )

        # Derive from cfg.architecture — see _subprocess_play_self_play
        # for the rationale (the subprocess path's identical fix).
        effective_structured = cfg.architecture == "structured"
        chunk, outcome_kind, winner, n_samples = _play_self_play_game(
            factory=factory,
            base_seed=seed,
            game_idx=game_idx,
            deck_pool=state["deck_pool"],
            deck_indices=deck_indices,
            per_game_timeout_sec=per_game_timeout_sec,
            capture_every=capture_every,
            action_encoding_version=cfg.action_encoding_version,
            write_structured_tokens=effective_structured,
            move_temperature=knobs["move_temperature"],
            temperature_cutoff=knobs["temperature_cutoff"],
        )

        npz_bytes = _encode_npz_bytes(
            chunk, write_structured_tokens=effective_structured,
        )
        duration_s = time.perf_counter() - t0
        _emit(
            "game_end",
            {
                "claim_id": claim.get("claim_id"),
                "game_idx": game_idx,
                "outcome": outcome_kind,
                "winner": winner,
                "n_samples": n_samples,
                "duration_s": duration_s,
            },
        )
        return GameResult(
            outcome=outcome_kind,
            winner=winner,
            n_samples=n_samples,
            duration_s=duration_s,
            npz_bytes=npz_bytes,
        )

    def _teardown() -> None:
        pool = state.get("pool")
        if pool is not None:
            # ``wait=False`` signals the pool's worker subprocesses to
            # shut down but does NOT block teardown waiting for them
            # to exit. Combined with the bounded ``thread.join`` in
            # ``ClaimLoopRunner.run``, this keeps the daemon's drain
            # transition prompt even when one subprocess is hung — the
            # orphan subprocesses get reaped by their own
            # ``Process.join`` in the executor's manager thread (or
            # when the daemon process exits). ``cancel_futures=True``
            # drops any still-queued tasks; running tasks proceed
            # until they finish.
            pool_controls.shutdown(wait=False)
        state["configs_by_version"].clear()
        state["deck_pool"] = None

    return CallableGameRunner(
        _play_one, _setup, _teardown,
        reset_pool_fn=pool_controls.reset,
        pool_generation_fn=pool_controls.generation,
    )


# ---------------------------------------------------------------------------
# Eval GameRunner
# ---------------------------------------------------------------------------


def build_eval_runner(
    coordinator_url: str,
    stage_id: str,
    n_workers: int,
    weights_cache_root: Path,
    worker_id: str,
    log_event: Optional[Callable[[str, dict], None]] = None,
    *,
    deck_pool_path: Optional[str] = None,
    nn_device: Optional[str] = None,
    http_get: Callable[..., tuple[int, bytes, dict[str, str]]] = _http_get,
    use_subprocess_pool: bool = False,
) -> GameRunner:
    """Production eval :class:`GameRunner`.

    Per-claim flow:

    1. Read ``weights["new"]`` and ``weights["old"]`` from the claim
       payload.
    2. Fetch both weights+metadata blobs (cache-aware).
    3. Build (or reuse) one ``NNPimcConfig`` per version.
    4. Run one match game; alternate seats by ``game_idx`` parity and
       remap the winner so seat 0 always means "new won."
    5. Return :class:`GameResult` with ``wins_new`` / ``wins_old`` /
       ``draws``.

    ``use_subprocess_pool`` mirrors the self-play factory: True spawns
    a ``ProcessPoolExecutor`` and dispatches each game to a subprocess
    so the Python GIL doesn't serialize parallel claim-loop threads.
    Default ``False`` keeps the in-thread path for the
    monkeypatch-heavy test suite.

    ``nn_device`` mirrors the self-play factory exactly: the daemon's
    ``--nn-device`` selector, carried unresolved into ``NNPimcConfig``
    and only resolved per-worker in ``NNPimcConfig.build()`` so the
    daemon parent never primes CUDA. Default ``None`` (CPU) keeps the
    historical eval path byte-identical.
    """

    state: dict[str, Any] = {
        "deck_pool": None,
        "configs_by_version": {},
    }
    state_lock = threading.Lock()

    def _emit(name: str, payload: dict) -> None:
        if log_event is not None:
            try:
                log_event(name, payload)
            except Exception:  # noqa: BLE001
                log.exception("[eval_runner] log_event %s raised", name)

    # Same pool lifecycle owner the self-play factory uses -- see
    # :class:`_PoolControls` for the BrokenProcessPool recovery story.
    pool_controls = _PoolControls(
        state,
        enabled=use_subprocess_pool,
        emit=_emit,
        label="eval_runner",
    )

    def _ensure_deck_pool() -> None:
        if state["deck_pool"] is not None:
            return
        with state_lock:
            if state["deck_pool"] is not None:  # double-checked
                return
            path = _resolve_deck_pool_path(deck_pool_path)
            if not path.is_file():
                raise RuntimeError(
                    f"deck pool not found at {path}; pass deck_pool_path "
                    f"explicitly via /v2/start_eval body, or place the "
                    f"canonical pool at {DEFAULT_DECK_POOL_PATH}."
                )
            from mtg_bot.decks import decks_from_json
            state["deck_pool"] = decks_from_json(path)
            _emit(
                "deck_pool_loaded",
                {"path": str(path), "n_decks": len(state["deck_pool"])},
            )

    def _ensure_config(version: str):
        cfg = state["configs_by_version"].get(version)
        if cfg is not None:
            return cfg
        with state_lock:
            cfg = state["configs_by_version"].get(version)  # double-checked
            if cfg is not None:
                return cfg
            weights_dir = fetch_weights_if_missing(
                coordinator_url=coordinator_url,
                version=version,
                weights_root=Path(weights_cache_root),
                http_get=http_get,
            )
            metadata = _read_metadata_json(weights_dir)
            cfg = _build_nn_pimc_config(
                weights_dir / "weights.pt", metadata, device=nn_device,
            )
            state["configs_by_version"][version] = cfg
            _emit(
                "weights_loaded",
                {
                    "version": version,
                    "architecture": cfg.architecture,
                    "action_encoding_version": cfg.action_encoding_version,
                },
            )
            return cfg

    def _setup(coord_url_unused, weights_root_unused, kind, n_workers_at_setup):
        state["kind"] = kind
        if kind != "eval":
            log.warning(
                "[eval_runner] setup called with kind=%s "
                "(expected eval)", kind,
            )
        pool_controls.ensure(n_workers_at_setup)

    def _play_one(claim: dict, shard_label: str) -> GameResult:
        # Subprocess path: submit to the PPE and decode the result.
        pool = state.get("pool")
        if pool is not None:
            game_idx = int(claim.get("game_idx") or 0)
            weights = claim.get("weights") or {}
            _emit(
                "game_start",
                {
                    "claim_id": claim.get("claim_id"),
                    "game_idx": game_idx,
                    "new_version": weights.get("new"),
                    "old_version": weights.get("old"),
                    "mode": "subprocess",
                },
            )
            future = pool.submit(
                _subprocess_play_eval,
                claim=claim,
                coordinator_url=coordinator_url,
                weights_cache_root=str(weights_cache_root),
                deck_pool_path=deck_pool_path,
                nn_device=nn_device,
            )
            payload = future.result()
            winner_remapped = payload["winner"]
            decisions = payload["n_samples"]
            outcome_kind = payload["outcome"]
            wins_new = 1 if winner_remapped == 0 else 0
            wins_old = 1 if winner_remapped == 1 else 0
            draws = 1 if winner_remapped is None and outcome_kind == "completed" else 0
            _emit(
                "game_end",
                {
                    "claim_id": claim.get("claim_id"),
                    "game_idx": payload["game_idx"],
                    "outcome": outcome_kind,
                    "winner_remapped": winner_remapped,
                    "decisions": decisions,
                    "duration_s": payload["duration_s"],
                    "mode": "subprocess",
                },
            )
            return GameResult(
                outcome=outcome_kind,
                winner=winner_remapped,
                wins_new=wins_new,
                wins_old=wins_old,
                draws=draws,
                n_samples=decisions,
                duration_s=payload["duration_s"],
            )

        # In-thread fallback.
        t0 = time.perf_counter()
        _ensure_deck_pool()

        weights = claim.get("weights") or {}
        new_version = weights.get("new") or weights.get("p0")
        old_version = weights.get("old") or weights.get("p1")
        if not new_version or not old_version:
            raise RuntimeError(
                f"eval claim missing weights.new / weights.old: "
                f"claim={claim}"
            )

        new_cfg_base = _ensure_config(new_version)
        old_cfg_base = _ensure_config(old_version)

        nodes = int(claim.get("nodes") or new_cfg_base.inner_nodes)
        samples = int(claim.get("samples") or new_cfg_base.samples)
        per_game_timeout_sec = claim.get("per_game_timeout_sec")
        if per_game_timeout_sec is not None:
            per_game_timeout_sec = float(per_game_timeout_sec)
        seed = int(claim.get("seed") or 0)
        game_idx = int(claim.get("game_idx") or 0)
        deck_indices = list(claim.get("deck_indices") or [0, 0])

        from mtg_bot.training import NNPimcConfig

        def _per_game(cfg):
            return NNPimcConfig(
                weights_path=cfg.weights_path,
                samples=samples,
                inner_nodes=nodes,
                hidden_sizes=cfg.hidden_sizes,
                architecture=cfg.architecture,
                policy_hidden_sizes=cfg.policy_hidden_sizes,
                action_encoding_version=cfg.action_encoding_version,
                with_policy=cfg.with_policy,
                puct=cfg.puct,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )

        new_factory = _per_game(new_cfg_base).build
        old_factory = _per_game(old_cfg_base).build

        _emit(
            "game_start",
            {
                "claim_id": claim.get("claim_id"),
                "game_idx": game_idx,
                "new_version": new_version,
                "old_version": old_version,
                "nodes": nodes,
                "samples": samples,
            },
        )

        winner_remapped, decisions, outcome_kind = _play_eval_game(
            p0_factory=new_factory,
            p1_factory=old_factory,
            base_seed=seed,
            game_idx=game_idx,
            deck_pool=state["deck_pool"],
            deck_indices=deck_indices,
            per_game_timeout_sec=per_game_timeout_sec,
        )

        wins_new = 1 if winner_remapped == 0 else 0
        wins_old = 1 if winner_remapped == 1 else 0
        draws = 1 if (
            outcome_kind == "completed" and winner_remapped is None
        ) else 0
        duration_s = time.perf_counter() - t0
        _emit(
            "game_end",
            {
                "claim_id": claim.get("claim_id"),
                "game_idx": game_idx,
                "outcome": outcome_kind,
                "winner_remapped": winner_remapped,
                "decisions": decisions,
                "duration_s": duration_s,
            },
        )
        return GameResult(
            outcome=outcome_kind,
            winner=winner_remapped,
            n_samples=0,
            duration_s=duration_s,
            wins_new=wins_new,
            wins_old=wins_old,
            draws=draws,
            npz_bytes=None,
        )

    def _teardown() -> None:
        pool_controls.shutdown(wait=True)
        state["configs_by_version"].clear()
        state["deck_pool"] = None

    return CallableGameRunner(
        _play_one, _setup, _teardown,
        reset_pool_fn=pool_controls.reset,
        pool_generation_fn=pool_controls.generation,
    )
