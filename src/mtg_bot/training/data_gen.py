"""Self-play data generation: play games, record (features, outcome).

Produces the training data for Phase D's value-network training. For
every non-chance decision point in every game, captures features from
**both** player perspectives and labels them with the eventual game
outcome from that player's perspective. Same state, two viewers,
two labels — doubles the data with no extra game-play cost.

Storage is ``np.savez_compressed`` (one .npz per generation run).
Multiple .npz files in a directory can be loaded and concatenated
via ``load_all_datasets`` so the trainer can combine sessions.

The strategies used to play the games matter — random play produces
garbage labels (outcome doesn't reflect position quality). Phase D
typically uses PIMC + heuristic on both sides; later phases bootstrap
by using the previously-trained NN to play the next round of games
(the AlphaZero loop).

Two driver functions:

- ``generate_training_data`` runs games sequentially. Used by tests
  (small N, want simple code path) and one-off uses where setup
  overhead matters more than wall-clock parallelism.
- ``generate_training_data_parallel`` distributes games across a
  ``multiprocessing.Pool`` of workers. Both produce identical
  arrays for the same ``base_seed`` because per-game seeds are
  derived from ``game_idx``, not from worker assignment — both
  paths call the same per-game helper. This determinism is locked
  in by a test.

Optional replay capture: pass a ``ReplayCaptureConfig`` to either
driver to write a per-game replay file alongside the training-data
arrays. Useful for debugging engine gaps — see
``mtg_bot.training.replay_capture``.

Structured-token opt-in (Step 1 NN representation)
==================================================

The Step 1 roadmap (``docs/NN_REPRESENTATION_ROADMAP.md``) replaces
the 51-dim flat featurizer with a per-token structured input matching
the shape pinned in ``mtg_bot.training.structured_model``. The
trainer-side consumer for that input lives in a separate slice that
conflicts with the running runs-db cutover; this writer opts into
emitting the new shape alongside the legacy 51-dim features so the
dataset-format change can land independently.

When ``structured_tokens=True`` is passed to
``generate_training_data`` / ``generate_training_data_parallel``,
each sample row records the tokenized state via
``state_tokenizer.tokenize_state`` and the resulting arrays are
packed into the npz under four additional keys:

- ``structured_token_type``     int64   ``(N, max_seq_len)``
- ``structured_token_emb_ids``  int64   ``(N, max_seq_len, N_EMB_FIELDS=7)``
- ``structured_token_scalars``  float32 ``(N, max_seq_len, scalar_width)``
  where ``scalar_width`` follows ``structured_scalar_schema_version``
  (3 for schema 1, ``N_SCALAR_FIELDS_V2`` for schema 2)
- ``structured_attention_mask`` bool    ``(N, max_seq_len)``

``max_seq_len`` is computed per-file as the longest row's emitted
length — shorter rows are right-padded with zeros (and
``structured_attention_mask=False`` on the padded slots). The model's
``[CLS]`` is NOT emitted here — ``StructuredModel.forward`` prepends
it itself.

Default is OFF for backwards compatibility: existing self-play paths
that don't opt in produce the exact same files they always have.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import itertools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from mtg_bot.agents import RandomAgent
from mtg_bot.cancellation import raise_if_cancelled
from mtg_bot.decks import Deck
from mtg_bot.features import FEATURE_DIM, extract_features
from mtg_bot.replay import save_replay
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.engine_adapter_factory import make_game_state_adapter
from mtg_bot.search.heuristics_config import HeuristicsConfig
from mtg_bot.search.protocol import CHANCE_PLAYER, Decision, PlayerID
from mtg_bot.simulate import setup_game
from mtg_bot.training._win32_job import JobObjectGuard
from mtg_bot.training._worker_threads import pin_worker_torch_threads
from mtg_bot.training.config import StrategyFactory
from mtg_bot.training.progress import DataGenProgress
from mtg_bot.training.replay_capture import (
    ReplayRecorder, build_search_annotation, heuristic_eval_annotator,
    nn_eval_annotator,
)


_DEFAULT_DECISION_CAP = 4000

# Type alias: per-game chunk + winner + outcome-kind label. Winner
# is the PlayerID of the game's victor or None for a draw / crash /
# timeout; the tracker uses it to compute the live winrate without
# having to re-parse the outcomes array (which holds per-viewer
# signs that look symmetric). The trailing string is one of
# "completed" / "crashed" / "timed_out" so the driver can
# distinguish engine-level exceptions from per-game wall-clock
# timeouts (different root causes that both produce a winner=None
# empty chunk — the UI now reports them as separate counters).
_GameChunk = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[PlayerID], str,
]

_log = logging.getLogger(__name__)

_DTYPE_FEATURES = np.float32
_DTYPE_OUTCOMES = np.float32
# Policy targets: per-slot probabilities. float32 keeps storage
# matched to features; precision well above what cross-entropy
# needs. Must match the dtype the AlphaZero trainer feeds into
# the loss function — float64 would be a wasteful upcast.
_DTYPE_POLICY = np.float32

# Structured-token dtypes — must match what the StructuredModel
# forward path expects (LongTensor for ids / FloatTensor for scalars
# / BoolTensor for the mask). Stored at the dataset level so the
# pad/stack helper and the npz reader agree.
# Per-row game group id (``game_idx`` within the generating run).
# Lets the trainer split train/val by GAME rather than by row so
# rows from one game never straddle the split (row-level splits
# leak near-duplicate states into val and turn val loss into a
# memorization score).
_DTYPE_GAME_IDS = np.int32
_DTYPE_STRUCT_TOKEN_TYPE = np.int64
_DTYPE_STRUCT_EMB_IDS = np.int64
_DTYPE_STRUCT_SCALARS = np.float32
_DTYPE_STRUCT_MASK = np.bool_

# Keys used in the saved npz when structured-tokens are enabled.
_STRUCT_NPZ_KEYS = (
    "structured_token_type",
    "structured_token_emb_ids",
    "structured_token_scalars",
    "structured_attention_mask",
)


def _root_value_from_candidates(
    candidates: list[dict[str, Any]],
) -> Optional[float]:
    """Visit-weighted mean root Q over the search's root children, from
    the ACTING player's perspective -- the search's own estimate of the
    position. ``None`` when the search reported no usable Q (forced
    move, random/baseline strategy, or every candidate unvisited).

    This is the dense value target (2026-09-26): every net trained on
    +/-1 game outcomes alone plateaued ~200 Elo below the heuristic
    teacher with a value head that barely beat the life-differential
    baseline. The root value is what the teacher actually believed at
    the decision, not what happened 40 decisions later.
    """
    num = 0.0
    den = 0
    for c in candidates:
        q = c.get("q")
        v = int(c.get("visits") or 0)
        if q is None or v <= 0:
            continue
        num += float(q) * v
        den += v
    if den <= 0:
        return None
    return num / den


def _static_evaluator_for(strategy: Any) -> Optional[Any]:
    """The evaluator behind ``strategy``, for a no-search static value.

    Resolution order: an explicit ``static_evaluator`` attribute; the
    strategy's own ``evaluator`` (MCTS / ISMCTS); or, for PIMC, the
    evaluator of one inner search built from ``inner_factory``. The
    result is cached on the strategy so PIMC builds that inner search
    at most once per game. ``None`` when the strategy carries no
    evaluator (random / baseline strategies).
    """
    ev = getattr(strategy, "static_evaluator", None)
    if ev is None:
        ev = getattr(strategy, "evaluator", None)
    if ev is None:
        factory = getattr(strategy, "inner_factory", None)
        if callable(factory):
            ev = getattr(factory(), "evaluator", None)
    if ev is not None:
        try:
            strategy.static_evaluator = ev
        except AttributeError:  # __slots__ strategies: just don't cache
            pass
    return ev


def _static_value(strategy: Any, state: Any, player_id: PlayerID) -> Optional[float]:
    """``evaluator.evaluate(state, player_id)`` for the strategy's
    evaluator, or ``None`` when it has none. Used for sample points
    where no search ran (forced moves), so the dense value target
    covers every recorded row rather than the ~12 % that were real
    decisions (measured on batch 17, 2026-09-26)."""
    ev = _static_evaluator_for(strategy)
    if ev is None:
        return None
    return float(ev.evaluate(state, player_id))


def _pad_and_stack_structured_rows(
    rows: list[dict[str, np.ndarray]],
    *,
    scalar_schema_version: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Right-pad variable-length per-row token dicts to the longest
    row and stack into ``(N, max_seq_len, ...)`` arrays.

    Each input dict is the output of
    ``state_tokenizer.tokenize_state`` (or a sentinel with seq_len=0
    for rows that opted out of structured tokenization). Padded
    slots get zeros for ids / scalars and ``False`` for the
    attention mask, matching the StructuredModel's
    ``key_padding_mask`` semantics (True = real token; padding is
    suppressed).
    """
    if not rows:
        # The trailing dim must match the populated-rows path so empty
        # timeout/crash npzs can be stacked alongside real ones in the
        # dataset loader. Shape-(0,0,0) used to leak through here and
        # crash train_nn.py at "could not broadcast input array from
        # shape (0,0,0) into shape (0,0,N_EMB_FIELDS)" — the iter_004
        # workaround was to quarantine the empties before training.
        from mtg_bot.training.structured_model import (
            DEFAULT_SCALAR_SCHEMA_VERSION, N_EMB_FIELDS,
            scalar_width_for_schema,
        )
        # Scalar width follows the schema the rows WOULD have used
        # (the structured trainer normalizes zero-row files to the
        # real files' width, but keeping the placeholder honest is
        # cheap).
        empty_scalar_width = scalar_width_for_schema(
            DEFAULT_SCALAR_SCHEMA_VERSION
            if scalar_schema_version is None else scalar_schema_version
        )
        return {
            "structured_token_type": np.zeros(
                (0, 0), dtype=_DTYPE_STRUCT_TOKEN_TYPE,
            ),
            "structured_token_emb_ids": np.zeros(
                (0, 0, N_EMB_FIELDS), dtype=_DTYPE_STRUCT_EMB_IDS,
            ),
            "structured_token_scalars": np.zeros(
                (0, 0, empty_scalar_width), dtype=_DTYPE_STRUCT_SCALARS,
            ),
            "structured_attention_mask": np.zeros(
                (0, 0), dtype=_DTYPE_STRUCT_MASK,
            ),
            "root_values": np.zeros((0,), dtype=np.float32),
        }
    n_rows = len(rows)
    max_len = max(
        int(r["token_type"].shape[0]) for r in rows
    )
    # All rows share the same per-token field widths (defined by
    # structured_model). Pull from the first non-empty row; fall back
    # to the standard widths if every row is empty (rare — would
    # mean every state had zero tokens).
    from mtg_bot.training.structured_model import (
        DEFAULT_SCALAR_SCHEMA_VERSION, N_EMB_FIELDS,
        scalar_width_for_schema,
    )
    emb_width = N_EMB_FIELDS
    scalar_width = scalar_width_for_schema(
        DEFAULT_SCALAR_SCHEMA_VERSION
        if scalar_schema_version is None else scalar_schema_version
    )
    for r in rows:
        if r["token_emb_ids"].size:
            emb_width = int(r["token_emb_ids"].shape[1])
            scalar_width = int(r["token_scalars"].shape[1])
            break
    token_type = np.zeros(
        (n_rows, max_len), dtype=_DTYPE_STRUCT_TOKEN_TYPE,
    )
    emb_ids = np.zeros(
        (n_rows, max_len, emb_width), dtype=_DTYPE_STRUCT_EMB_IDS,
    )
    scalars = np.zeros(
        (n_rows, max_len, scalar_width), dtype=_DTYPE_STRUCT_SCALARS,
    )
    attention_mask = np.zeros(
        (n_rows, max_len), dtype=_DTYPE_STRUCT_MASK,
    )
    # Per-row search root value (viewer's perspective); NaN when the
    # row was recorded without one (pre-2026-09-26 shards, forced
    # moves, strategies without ``choose_with_full_stats``).
    root_values = np.full((n_rows,), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        rv = r.get("root_value")
        if rv is not None:
            root_values[i] = np.float32(rv)
        n = int(r["token_type"].shape[0])
        if n == 0:
            continue
        token_type[i, :n] = r["token_type"]
        emb_ids[i, :n, :] = r["token_emb_ids"]
        scalars[i, :n, :] = r["token_scalars"]
        attention_mask[i, :n] = r["attention_mask"]
    return {
        "structured_token_type": token_type,
        "structured_token_emb_ids": emb_ids,
        "structured_token_scalars": scalars,
        "structured_attention_mask": attention_mask,
        "root_values": root_values,
    }


def _empty_structured_row(
    scalar_schema_version: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Per-row sentinel for "no structured token captured at this
    sample point." Used so the parallel rows-array has stable length
    when structured_tokens is disabled but a downstream consumer
    asks for the arrays anyway. ``scalar_schema_version`` picks the
    (zero-row) scalar width; ``None`` = the tokenizer default."""
    from mtg_bot.training.structured_model import (
        DEFAULT_SCALAR_SCHEMA_VERSION, N_EMB_FIELDS,
        scalar_width_for_schema,
    )
    scalar_width = scalar_width_for_schema(
        DEFAULT_SCALAR_SCHEMA_VERSION
        if scalar_schema_version is None else scalar_schema_version
    )
    return {
        "token_type": np.zeros((0,), dtype=_DTYPE_STRUCT_TOKEN_TYPE),
        "token_emb_ids": np.zeros(
            (0, N_EMB_FIELDS), dtype=_DTYPE_STRUCT_EMB_IDS,
        ),
        "token_scalars": np.zeros(
            (0, scalar_width), dtype=_DTYPE_STRUCT_SCALARS,
        ),
        "attention_mask": np.zeros((0,), dtype=_DTYPE_STRUCT_MASK),
    }


@dataclass(frozen=True)
class ReplayCaptureConfig:
    """Picklable config for capturing replays during data gen.

    ``save_dir``: where to write replay files (one .json.gz per
    captured game).
    ``capture_every``: capture every Nth game starting at
    game_idx 0 (so capture_every=10 captures games 0, 10, 20, ...).
    Default 1 = capture all games.
    ``nn_weights_path``: optional path to NN weights — when set,
    each snapshot also gets an ``nn_eval`` annotation alongside the
    always-on ``heuristic_eval``.
    ``nn_version_label``: optional human-readable label (typically
    a registry version like "v3") stored in each ``nn_eval`` so
    consumers know which network produced the scores.
    ``filename_prefix``: prepended to each replay filename. Useful
    for tagging a batch of replays from one experiment.

    Frozen + primitive types so it crosses the multiprocessing spawn
    boundary cleanly.
    """

    save_dir: str
    capture_every: int = 1
    nn_weights_path: Optional[str] = None
    nn_version_label: Optional[str] = None
    filename_prefix: str = "game"
    # Architecture overrides for the NN annotator. Required when the
    # captured run uses anything other than the legacy default trunk
    # shape (modern iterate runs all do — v6 is 1024-1024-512). Stored
    # as tuple so the dataclass stays frozen + picklable across the
    # multiprocessing boundary.
    nn_hidden_sizes: Optional[tuple[int, ...]] = None
    nn_with_policy: bool = False
    # Dual-head architecture for the NN annotator. ``None`` means
    # "use NNEvaluator's default" (``"shared"``); a separate-trunks
    # checkpoint requires ``"separate"``. Threaded through from the
    # version's metadata.architecture by iterate / evaluate scripts.
    nn_architecture: Optional[str] = None
    nn_policy_hidden_sizes: Optional[tuple[int, ...]] = None
    # Sidecar metadata.json path for the NN annotator. When set, the
    # annotator builds its NNEvaluator via
    # ``NNEvaluator.from_metadata`` so every architecture knob — incl.
    # the structured-trunk ``emb_dim`` / ``n_layers`` / ``hidden_dim``
    # / ... that the flat ``nn_hidden_sizes`` / ``nn_architecture`` /
    # ``nn_policy_hidden_sizes`` kwargs can't express — is single-
    # sourced from the metadata file. The flat overrides above are
    # ignored when this is set. iter_019 BUG #3: without this path,
    # a structured-trunk run trained at non-default dims falls back
    # to ``StructuredModelConfig`` defaults at annotator-construction
    # time and the per-game capture loop crashes on a state_dict
    # shape mismatch. Stored as a string for the same picklable-
    # across-spawn reason ``nn_weights_path`` is.
    nn_metadata_path: Optional[str] = None
    # Diagnostic deep-tree capture. Defaults reproduce production
    # root-only behavior (depth=1, top_k irrelevant). The diagnostic
    # script ``capture_annotated_replays.py`` flips these via the
    # ``--capture-tree-depth`` / ``--capture-top-k`` flags so the
    # replay viewer can drill into the search subtree under any
    # candidate. Deep mode skips the production size budget — these
    # captures are expected to be larger and aren't shipped to the
    # trainer's replay buffer.
    tree_depth: int = 1
    tree_top_k: Optional[int] = None

    def should_capture(self, game_idx: int) -> bool:
        return game_idx % self.capture_every == 0

    def replay_path(self, base_seed: int, game_idx: int) -> Path:
        return Path(self.save_dir) / (
            f"{self.filename_prefix}_seed{base_seed}_idx{game_idx:04d}.json.gz"
        )

    def build_recorder(self) -> ReplayRecorder:
        annotators = [heuristic_eval_annotator()]
        if self.nn_weights_path:
            annotators.append(nn_eval_annotator(
                Path(self.nn_weights_path),
                version_label=self.nn_version_label,
                hidden_sizes=self.nn_hidden_sizes,
                with_policy=self.nn_with_policy,
                architecture=self.nn_architecture,
                policy_hidden_sizes=self.nn_policy_hidden_sizes,
                metadata_path=(
                    Path(self.nn_metadata_path)
                    if self.nn_metadata_path else None
                ),
            ))
        return ReplayRecorder(annotators=annotators)


def _empty_arrays() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Empty arrays matching the post-A2.3 4-tuple schema. Use for
    the no-games / no-samples paths in the data-gen functions.
    Lazy-imports ``ACTION_VOCAB_SIZE`` to avoid the ``training`` →
    ``training.action_encoding`` dep at module-load time."""
    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
    return (
        np.zeros((0, FEATURE_DIM), dtype=_DTYPE_FEATURES),
        np.zeros((0,), dtype=_DTYPE_OUTCOMES),
        np.zeros((0, ACTION_VOCAB_SIZE), dtype=_DTYPE_POLICY),
        np.zeros((0, ACTION_VOCAB_SIZE), dtype=np.bool_),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample_decks_for_game(
    deck_pool: list[Deck], game_idx: int, base_seed: int,
) -> tuple[Deck, Deck]:
    """Deterministically pick 2 decks (one per seat) from the pool
    for game ``game_idx``. Uses a per-game RNG seeded from
    ``base_seed`` + a fixed offset so the choice is reproducible
    AND independent of ``game_idx``'s usage in strategy/engine seeds.

    Single-deck pool degenerates into a mirror match (same deck on
    both seats) — explicit so callers know it's not an error. Empty
    pool raises.
    """
    if not deck_pool:
        raise ValueError("deck_pool is empty")
    rng = random.Random(base_seed + 100 * game_idx + 12345)
    if len(deck_pool) >= 2:
        a, b = rng.sample(range(len(deck_pool)), 2)
    else:
        a = b = 0
    return deck_pool[a], deck_pool[b]


def generate_training_data(
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    n_games: int,
    base_seed: int = 0,
    decision_cap: int = _DEFAULT_DECISION_CAP,
    sample_every: int = 1,
    replay_config: Optional[ReplayCaptureConfig] = None,
    deck_pool: Optional[list[Deck]] = None,
    with_policy: bool = False,
    progress: Optional[DataGenProgress] = None,
    per_game_timeout_sec: Optional[float] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
    structured_tokens: bool = False,
    structured_scalar_schema_version: Optional[int] = None,
    with_game_ids: bool = False,
    move_temperature: float = 1.0,
    temperature_cutoff: int = 30,
) -> tuple[np.ndarray, ...]:
    """Play ``n_games`` and return training arrays.

    ``structured_scalar_schema_version`` picks the ``token_scalars``
    row layout the structured tokens are emitted under (see
    ``structured_model.SCALAR_FIELD_*``); ``None`` = the tokenizer's
    current default. Must match the ``scalar_schema_version`` the
    downstream ``mtg_bot train --architecture structured`` run uses
    (the trainer refuses mismatched widths).

    For each non-chance decision (every Nth, ``sample_every``),
    captures ``extract_features(state, viewer)`` for both viewers
    (0 and 1) and labels each with the eventual game outcome from
    that viewer's perspective (+1/-1/0).

    Default returns ``(features, outcomes)`` — the value-only
    schema that pre-A2.3 callers expect. Pass ``with_policy=True``
    for the AlphaZero schema:

    Returns a 2-tuple (default) or 4-tuple (with_policy=True):
        features:       ``(N, FEATURE_DIM)`` float32.
        outcomes:       ``(N,)`` float32 in ``{-1, 0, +1}``.
        policy_targets: ``(N, ACTION_VOCAB_SIZE)`` float32 — visit
            distribution per row, all-zero for rows without policy
            supervision (data-doubled non-active-viewer rows;
            baseline-strategy rows).
        action_masks:   ``(N, ACTION_VOCAB_SIZE)`` bool — legal-
            slot mask per row, all-False for rows without policy
            supervision.

    Deterministic in ``base_seed`` — same call produces the same
    arrays. Trainer code should detect "no policy supervision" via
    ``policy_targets.sum(-1) > 0.5`` per row.

    ``structured_tokens=True`` records the per-state structured token
    representation alongside each row (see module docstring). When
    enabled, the return value grows to a 6-tuple
    ``(features, outcomes, policy_targets, action_masks,
    structured_token_type, structured_token_emb_ids,
    structured_token_scalars, structured_attention_mask)`` and
    ``with_policy=True`` is implied (callers asking for structured
    tokens always want policy supervision too — the Step 1 trainer
    is masked-CE + value-MSE + aux on the same row).

    ``with_game_ids=True`` appends one more array, ``game_ids``
    ``(N,)`` int32 — the ``game_idx`` each row came from — as the
    LAST element of whichever tuple shape the other flags selected.
    Save it via ``save_dataset(..., game_ids=...)`` so the trainer
    can split train/val by game.

    ``move_temperature`` / ``temperature_cutoff`` control self-play
    move sampling (see ``_play_one_game_recording``): for the first
    ``temperature_cutoff`` NON-forced decisions of each game the
    played move is sampled proportional to ``visits ** (1/T)``;
    afterwards (or when ``T <= 0``) the search argmax is played.
    The recorded policy TARGET is always the untempered visit
    distribution.
    """
    if heuristics is None:
        heuristics = HeuristicsConfig()
    if progress is not None:
        progress.begin()
    chunks: list[_GameChunk] = []
    struct_rows_chunks: list[list[dict[str, np.ndarray]]] = []
    for i in range(n_games):
        chunk = _generate_one_game_data(
            game_idx=i,
            p0_factory=p0_factory, p1_factory=p1_factory,
            base_seed=base_seed,
            decision_cap=decision_cap, sample_every=sample_every,
            replay_config=replay_config,
            deck_pool=deck_pool,
            per_game_timeout_sec=per_game_timeout_sec,
            hang_dump_dir=hang_dump_dir,
            heuristics=heuristics,
            structured_tokens=structured_tokens,
            structured_scalar_schema_version=(
                structured_scalar_schema_version
            ),
            move_temperature=move_temperature,
            temperature_cutoff=temperature_cutoff,
        )
        chunks.append(chunk)
        struct_rows_chunks.append(chunk[6] if len(chunk) > 6 else [])
        if progress is not None:
            progress.record_game(
                n_samples=int(chunk[0].shape[0]),
                outcome_p0=_winner_to_p0_outcome(chunk[4]),
            )
    f, o, p, m, g = _concat_chunks(chunks)
    if structured_tokens:
        flat_rows: list[dict[str, np.ndarray]] = []
        for rows in struct_rows_chunks:
            flat_rows.extend(rows)
        struct_arrays = _pad_and_stack_structured_rows(
            flat_rows,
            scalar_schema_version=structured_scalar_schema_version,
        )
        out: tuple[np.ndarray, ...] = (
            f, o, p, m,
            struct_arrays["structured_token_type"],
            struct_arrays["structured_token_emb_ids"],
            struct_arrays["structured_token_scalars"],
            struct_arrays["structured_attention_mask"],
        )
    elif with_policy:
        out = (f, o, p, m)
    else:
        out = (f, o)
    if with_game_ids:
        out = out + (g,)
    return out


def generate_training_data_parallel(
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    n_games: int,
    base_seed: int = 0,
    n_workers: Optional[int] = None,
    decision_cap: int = _DEFAULT_DECISION_CAP,
    sample_every: int = 1,
    replay_config: Optional[ReplayCaptureConfig] = None,
    deck_pool: Optional[list[Deck]] = None,
    with_policy: bool = False,
    progress: Optional[DataGenProgress] = None,
    per_game_timeout_sec: Optional[float] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    action_encoding_version: str = "positional_v1",
    pool_initializer: Optional[Callable[..., None]] = None,
    pool_initargs: tuple = (),
    pool_mp_context: Optional["mp.context.BaseContext"] = None,
    structured_tokens: bool = False,
    structured_scalar_schema_version: Optional[int] = None,
    with_game_ids: bool = False,
    move_temperature: float = 1.0,
    temperature_cutoff: int = 30,
) -> tuple[np.ndarray, ...]:
    """Same contract as ``generate_training_data`` but distributes
    games across a ``multiprocessing.Pool``. Same ``with_policy``
    flag controlling 2-tuple vs 4-tuple return.

    ``stop_check`` is an optional zero-arg callable polled between
    completed-game results (one per imap_unordered yield). When it
    returns ``True``, the driver stops draining the pool, terminates
    the workers, and returns whatever results have already landed
    plus an empty-chunk tail for the un-played game indices. The
    progress tracker (if supplied) is finalized with status
    ``"stopped"`` so the UI sees the run as user-aborted rather than
    completed/failed. In-flight games run to natural completion
    (``pool.terminate()`` with timeout would lose their data); we
    simply stop dispatching new ones. v1 design — not mid-game
    responsive; see ``docs/SEARCH_LIMITATIONS.md``.

    Identical output to the sequential function for the same
    ``base_seed`` — per-game seeds are derived from ``game_idx``, not
    worker assignment, so the same game produces the same samples
    no matter which worker plays it. We use ``imap_unordered`` so
    the live progress tracker can report completions as they land,
    then sort the per-game chunks by ``game_idx`` before concatenation
    so the output arrays remain deterministic in ``base_seed``
    (locked in by tests).

    ``n_workers`` defaults to ``cpu_count() - 2`` (leaves a few cores
    for OS / Python main / the user's other work). Pass an explicit
    int to override.

    Caller note: factories must be picklable. Module-level functions
    (e.g. ``random_choice_factory``) and bound methods of picklable
    dataclasses (e.g. ``PimcConfig().build``) work; lambdas and
    nested closures don't. The CLI builds factories the right way.

    Worker spawn cost on Windows is ~1-2s per worker, paid once per
    pool. Dominates wall clock at very small ``n_games``; amortizes
    away past a few dozen games.

    ``pool_initializer`` / ``pool_initargs`` / ``pool_mp_context``
    plumb the SharedTensorNNEvalServer activation path from
    ``run_shard_local`` (and any other in-process caller that wants
    SHM-routed NN eval). When all three are provided, the
    ``multiprocessing.Pool`` is created off the supplied context with
    the initializer + args; workers fork-inherit / pickle-receive the
    SHM handle before any game runs. Defaults preserve the legacy
    ``mp.Pool(processes=n_workers)`` (platform default context, no
    initializer) so direct test callers stay unchanged.
    """
    if heuristics is None:
        heuristics = HeuristicsConfig()
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 2)
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if n_games == 0:
        f, o, p, m = _empty_arrays()
        if structured_tokens:
            empty_struct = _pad_and_stack_structured_rows(
                [], scalar_schema_version=structured_scalar_schema_version,
            )
            out: tuple[np.ndarray, ...] = (
                f, o, p, m,
                empty_struct["structured_token_type"],
                empty_struct["structured_token_emb_ids"],
                empty_struct["structured_token_scalars"],
                empty_struct["structured_attention_mask"],
            )
        else:
            out = (f, o, p, m) if with_policy else (f, o)
        if with_game_ids:
            out = out + (_empty_game_ids(),)
        return out

    args_list = [
        (
            i, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version, structured_tokens,
            structured_scalar_schema_version,
            move_temperature, temperature_cutoff,
        )
        for i in range(n_games)
    ]
    if progress is not None:
        progress.begin()
    # Pool() with no start method falls back to the platform default —
    # spawn on Windows, fork on Unix. Workers persist for pool
    # lifetime so import cost is paid once per worker.
    #
    # ``imap_unordered`` instead of ``map`` so completions stream
    # back as workers finish, letting the progress tracker emit
    # snapshots throughout the run rather than only at the end.
    # Results arrive out of order; we tag each per-game result with
    # its ``game_idx`` and sort at the end so the concatenated arrays
    # are deterministic in ``base_seed`` (locked in by tests).
    indexed: list[tuple[int, _GameChunk]] = []
    crashed_games: list[int] = []
    timed_out_games: list[int] = []
    stopped_early = False
    # Pool construction: ``run_shard_local``'s SHM path supplies a
    # ctx + initializer so workers install the SharedTensorNNEval
    # handle before the first game runs. Legacy callers leave the
    # three kwargs at their defaults and get the platform-default
    # pool with no initializer (unchanged from pre-SHM behavior).
    if pool_mp_context is not None:
        pool_ctx = pool_mp_context
    else:
        pool_ctx = mp
    pool_kwargs: dict[str, Any] = {"processes": n_workers}
    if pool_initializer is not None:
        pool_kwargs["initializer"] = pool_initializer
        pool_kwargs["initargs"] = pool_initargs
    # iter_019 BUG #4 — Windows job-object guard for kernel-level
    # child cleanup if the master dies WITHOUT running Python-level
    # cleanup (taskkill /F, segfault, OOM, etc.). The ``with Pool()
    # as p:`` context manager already handles normal exit + Python
    # exceptions via ``__exit__`` → ``terminate()``; this guard adds
    # the safety net for harder master deaths. On POSIX, the guard
    # is a no-op.
    #
    # The guard ONLY contains the worker children, NOT the master
    # (see the module docstring on _win32_job for why — putting the
    # master in the job would arm KILL_ON_JOB_CLOSE on the master
    # too, killing it on guard close). We attach workers after Pool
    # construction (which eagerly spawns them) and close the guard
    # at function exit; close is safe because Pool's terminate()
    # has already drained the workers.
    # Context order matters: ``with A, B:`` exits B first, then A.
    # We want Pool.__exit__ (terminate workers) to run BEFORE
    # JobObjectGuard.__exit__ (close handle). If the guard closed
    # first while workers were still alive in the job, the kernel's
    # KILL_ON_JOB_CLOSE would race with Pool's orderly shutdown.
    # Put the guard FIRST in the with statement so it exits LAST.
    _job_guard = JobObjectGuard()
    with _job_guard, pool_ctx.Pool(**pool_kwargs) as pool:
        _job_guard.attach_pool_workers(pool)
        for game_idx, chunk in pool.imap_unordered(
            _play_one_parallel_game, args_list, chunksize=1,
        ):
            indexed.append((game_idx, chunk))
            # ``_play_one_parallel_game`` converts both engine-level
            # crashes AND per-game timeouts into empty chunks
            # (winner=None, 0 samples) but tags each with an outcome
            # kind ("crashed" / "timed_out" / "completed") in
            # ``chunk[5]``. We track them separately so the live
            # progress UI can show real exceptions vs slow-game
            # timeouts as distinct counters — same root cause for
            # both is "no signal", different diagnostic implications.
            outcome_kind = chunk[5] if len(chunk) > 5 else "completed"
            crashed = outcome_kind == "crashed"
            timed_out = outcome_kind == "timed_out"
            if crashed:
                crashed_games.append(game_idx)
            elif timed_out:
                timed_out_games.append(game_idx)
            if progress is not None:
                if crashed:
                    progress.record_game(crashed=True)
                elif timed_out:
                    progress.record_game(timed_out=True)
                else:
                    progress.record_game(
                        n_samples=int(chunk[0].shape[0]),
                        outcome_p0=_winner_to_p0_outcome(chunk[4]),
                    )
            # User-requested stop: terminate the pool's pending work
            # and break out, preserving the data we already have. We
            # call ``pool.terminate()`` rather than letting the pool
            # drain naturally because the queued args could be
            # hundreds of games (~hours of wall-time on a slow eval
            # bench) and the user wants the run to end now. In-flight
            # games on the workers when terminate() lands DO get
            # killed mid-game — their partial progress is lost, but
            # everything that's already arrived in ``indexed`` is
            # safe. Documented divergence: stops fire at imap-yield
            # boundaries, not mid-game decision points.
            if stop_check is not None and stop_check():
                stopped_early = True
                pool.terminate()
                pool.join()
                break
    if crashed_games:
        import sys
        print(
            f"# WARNING: {len(crashed_games)}/{n_games} games crashed "
            f"in the worker pool and were dropped (engine state "
            f"corruption — see worker stderr above for tracebacks). "
            f"Affected game indices: {crashed_games[:20]}"
            f"{' ...' if len(crashed_games) > 20 else ''}",
            file=sys.stderr,
        )
    if timed_out_games:
        import sys
        print(
            f"# WARNING: {len(timed_out_games)}/{n_games} games hit "
            f"the per-game wall-clock timeout and were dropped (slow "
            f"games under CPU contention or genuine engine hangs). "
            f"Affected game indices: {timed_out_games[:20]}"
            f"{' ...' if len(timed_out_games) > 20 else ''}",
            file=sys.stderr,
        )
    indexed.sort(key=lambda t: t[0])
    chunks = [c for _, c in indexed]
    f, o, p, m, g = _concat_chunks(chunks)
    if stopped_early and progress is not None:
        # Caller will see the early return; finalize the tracker so
        # the UI labels the run as user-aborted rather than leaving
        # it 'running' forever. Caller-driven finalize is also
        # supported (caller may want extra metadata) — that's a
        # no-op idempotent overwrite.
        progress.finalize(status="stopped")
    if structured_tokens:
        # Each chunk's 7th element (index 6) is the list of per-row
        # structured-token dicts. Crashed / timed-out chunks emit an
        # empty list — safe to extend.
        flat_rows: list[dict[str, np.ndarray]] = []
        for c in chunks:
            if len(c) > 6:
                flat_rows.extend(c[6])
        struct_arrays = _pad_and_stack_structured_rows(
            flat_rows,
            scalar_schema_version=structured_scalar_schema_version,
        )
        out = (
            f, o, p, m,
            struct_arrays["structured_token_type"],
            struct_arrays["structured_token_emb_ids"],
            struct_arrays["structured_token_scalars"],
            struct_arrays["structured_attention_mask"],
        )
    elif with_policy:
        out = (f, o, p, m)
    else:
        out = (f, o)
    if with_game_ids:
        out = out + (g,)
    return out


def _play_one_parallel_game(
    args: tuple,
) -> tuple[int, _GameChunk]:
    """Top-level pickleable worker. Plays one game, returns the
    ``(game_idx, chunk)`` pair where ``chunk`` is the 5-tuple
    ``(features, outcomes, policy_targets, action_masks, winner)``.
    The driver uses ``game_idx`` to re-sort results from
    ``imap_unordered`` so the concatenated arrays stay deterministic.
    Lives at module top level (not nested) so multiprocessing can
    pickle it across the spawn boundary.

    Engine-level errors (state corruption bugs that surface at
    scale — ~1 in 1000 games for heuristic-vs-heuristic PIMC as of
    2026-05-03) are caught here and converted to an empty chunk
    so a single bad game doesn't take down the entire pool.
    Without this, ``pool.imap_unordered`` re-raises the worker
    exception and aborts the whole run, losing all in-flight
    sample data (``save_dataset`` only writes after every game
    completes). The traceback is logged to worker stderr so
    failures are visible in the pool's stderr relay; the
    parent's progress tracker still records the game as
    "completed" with 0 samples / no winner.

    Per-game wall-clock ``TimeoutError`` (out of
    ``run_with_timeout``) is caught separately so the driver can
    bucket it under the dedicated ``timed_out`` counter rather than
    ``crashed``. Both produce an empty chunk with ``winner=None``;
    the trailing ``outcome_kind`` element of the chunk tuple is
    what the driver reads to tell them apart.
    """
    # ``action_encoding_version`` was appended to the args tuple in
    # ``generate_training_data_parallel`` (feature/iterate-v2-wiring) so
    # iterate's bootstrap-metadata read flows all the way through to the
    # per-game policy-target encoder. ``structured_tokens`` was appended
    # next (Step 1 NN-representation roadmap). The legacy
    # ``enable_mulligans`` field was removed by
    # feature/always-on-mulligans — mulligans are always on now and the
    # tuple shape lost an element.
    # ``structured_scalar_schema_version`` (the token_scalars row
    # layout) was appended as the 14th element (feature/lf-tokenizer);
    # ``move_temperature`` / ``temperature_cutoff`` (self-play move
    # sampling knobs, feature/lf-training) as the 15th/16th. Older
    # tuple shapes default them: schema ``None`` (= tokenizer default),
    # temperature 1.0 / cutoff 30. The 15-element form is the
    # lf-training layout (13 + temperature knobs, no schema).
    structured_scalar_schema_version: Optional[int] = None
    move_temperature = 1.0
    temperature_cutoff = 30
    if len(args) == 16:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version, structured_tokens,
            structured_scalar_schema_version,
            move_temperature, temperature_cutoff,
        ) = args
    elif len(args) == 15:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version, structured_tokens,
            move_temperature, temperature_cutoff,
        ) = args
    elif len(args) == 14:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version, structured_tokens,
            structured_scalar_schema_version,
        ) = args
    elif len(args) == 13:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version, structured_tokens,
        ) = args
    elif len(args) == 12:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            per_game_timeout_sec, hang_dump_dir,
            heuristics, action_encoding_version,
        ) = args
        structured_tokens = False
    else:
        (
            game_idx, p0_factory, p1_factory, base_seed,
            decision_cap, sample_every, replay_config, deck_pool,
            per_game_timeout_sec, hang_dump_dir,
            heuristics,
        ) = args
        action_encoding_version = "positional_v1"
        structured_tokens = False
    try:
        chunk = _generate_one_game_data(
            game_idx=game_idx,
            p0_factory=p0_factory, p1_factory=p1_factory,
            base_seed=base_seed,
            per_game_timeout_sec=per_game_timeout_sec,
            decision_cap=decision_cap, sample_every=sample_every,
            replay_config=replay_config,
            deck_pool=deck_pool,
            hang_dump_dir=hang_dump_dir,
            heuristics=heuristics,
            action_encoding_version=action_encoding_version,
            structured_tokens=structured_tokens,
            structured_scalar_schema_version=(
                structured_scalar_schema_version
            ),
            move_temperature=move_temperature,
            temperature_cutoff=temperature_cutoff,
        )
    except TimeoutError as exc:
        import sys
        # ``decisions_completed`` is attached by
        # ``_generate_one_game_data``'s timeout handler; the 0 fallback
        # keeps this working for a TimeoutError raised before the
        # attach step (e.g. a shutdown race).
        partial_decisions = int(
            getattr(exc, "decisions_completed", 0) or 0,
        )
        print(
            f"[worker] game {game_idx} timed out "
            f"({exc}) at decisions={partial_decisions}; "
            f"dropping this game, pool continues.",
            file=sys.stderr,
        )
        f, o, p, m = _empty_arrays()
        chunk = (f, o, p, m, None, "timed_out", [], _empty_game_ids())
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        import sys
        import traceback
        print(
            f"[worker] game {game_idx} crashed "
            f"({type(exc).__name__}: {exc}); dropping this game, "
            f"pool continues.",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        f, o, p, m = _empty_arrays()
        chunk = (f, o, p, m, None, "crashed", [], _empty_game_ids())
    return game_idx, chunk


def _winner_to_p0_outcome(winner: Optional[PlayerID]) -> float:
    """Convert a per-game winner (0, 1, or None) into the +1 / -1 / 0
    outcome from p0's perspective. Used by the progress tracker to
    keep a live winrate without re-parsing ``outcomes`` (which holds
    per-viewer signs that look symmetric in aggregate)."""
    if winner is None:
        return 0.0
    return 1.0 if winner == 0 else -1.0


def _generate_one_game_data(
    *,
    game_idx: int,
    p0_factory: StrategyFactory,
    p1_factory: StrategyFactory,
    base_seed: int,
    decision_cap: int,
    sample_every: int,
    replay_config: Optional[ReplayCaptureConfig] = None,
    deck_pool: Optional[list[Deck]] = None,
    per_game_timeout_sec: Optional[float] = None,
    hang_dump_dir: Optional[str] = None,
    heuristics: Optional[HeuristicsConfig] = None,
    action_encoding_version: str = "positional_v1",
    structured_tokens: bool = False,
    structured_scalar_schema_version: Optional[int] = None,
    move_temperature: float = 1.0,
    temperature_cutoff: int = 30,
) -> tuple:
    """Play one game and return its per-game chunk tuple
    ``(features, outcomes, policy_targets, action_masks, winner,
    outcome_kind, struct_rows, game_ids)``. ``game_ids`` is an
    ``(N,)`` int32 array filled with ``game_idx`` — the per-row
    group id the trainer uses for game-level train/val splitting.

    The single source of truth for per-game data extraction; both
    sequential and parallel drivers call this so they produce
    bit-identical results for the same ``(game_idx, base_seed,
    factories)``.

    Two distinct decks are sampled from ``deck_pool`` deterministically
    (per-game seed) and used as the seats' libraries. ``deck_pool`` is
    required — the synthetic mono-color G/R fallback was removed; every
    self-play / eval data point now comes from a real deck distribution
    by construction.

    Starting-seat alternation: for ``game_idx % 2 == 1``, player 1
    starts on the play instead of player 0. Going first is
    structurally advantageous in MTG, so without alternation the
    seat-0 wins-on-the-play bias bleeds into the value labels (v6
    self-play hit 80.8% seat-0 wins, biasing every viewer=0 turn-1
    feature row toward +1 and every viewer=1 turn-1 row toward -1
    — the model learns "I'm seated at seat 0 → I win" rather than
    evaluating position quality). The features are seat-symmetric
    (see ``mtg_bot.features``) so flipping who starts averages the
    advantage out across game_idx without remapping winners or
    swapping strategies; the data is per-viewer-perspective and the
    label is ``winner == viewer``, which is correct regardless of
    who started.

    When ``replay_config`` is set and ``should_capture(game_idx)``
    returns True, also writes a per-game replay file to disk before
    returning. The annotators (heuristic + optional NN eval) are
    built fresh per game in the worker — NN model load happens
    once per captured game (a few ms; negligible vs game length).
    """
    if deck_pool is None:
        raise ValueError(
            "deck_pool is required; the synthetic mono-color G/R "
            "fallback was removed. Pass a real deck pool (typically "
            "loaded via ``mtg_bot.decks.decks_from_json`` from "
            "``data/17lands/eoe_decks.json.gz``)."
        )
    engine_seed = base_seed + game_idx
    deck_p0, deck_p1 = _sample_decks_for_game(
        deck_pool, game_idx, base_seed,
    )
    engine = setup_game(
        [
            RandomAgent(rng=random.Random(engine_seed + 1)),
            RandomAgent(rng=random.Random(engine_seed + 2)),
        ],
        decks=[deck_p0.cards, deck_p1.cards],
        seed=engine_seed,
    )
    if game_idx % 2 == 1:
        engine.active_player_idx = 1
        engine.priority_player_idx = 1
    # B3-S1: route through the single decision-source factory. Under
    # decision_source="python" (the default, carried on `heuristics`)
    # this returns the identical MtgGameStateAdapter built with the
    # same args — byte-identical to pre-B3 behavior. The flag is read
    # once here at game construction (the single atomic-revert knob).
    adapter = make_game_state_adapter(
        engine, expose_combat_decisions=True,
        heuristics=heuristics,
    )
    strategies = {
        0: p0_factory(base_seed + 100 * game_idx),
        1: p1_factory(base_seed + 100 * game_idx + 50),
    }
    # Pin torch CPU threads to 1 in this worker process. Strategy
    # factories that use the NN evaluator have just imported torch;
    # without this pin every worker inherits the default of "host
    # CPU count" threads — 30 workers x 32 MKL threads on a 32-core
    # box means ~960 threads thrashing 32 cores (~30-40% effective
    # utilization). The trainer has the same pin on CPU (commit
    # 3ad808f); this is the data-gen worker equivalent. No-op when
    # torch isn't loaded (heuristic-only runs); idempotent across
    # games (set_num_threads is process-global).
    pin_worker_torch_threads()
    rng = random.Random(base_seed + 100 * game_idx + 99)

    recorder: Optional[ReplayRecorder] = None
    if replay_config is not None and replay_config.should_capture(game_idx):
        recorder = replay_config.build_recorder()
        # Capture the initial state too so the replay viewer's first
        # snapshot reflects "game 0, before any actions taken."
        recorder.record(adapter.engine_state)

    tree_depth = (
        replay_config.tree_depth if replay_config is not None else 1
    )
    tree_top_k = (
        replay_config.tree_top_k if replay_config is not None else None
    )
    if per_game_timeout_sec is not None and per_game_timeout_sec > 0:
        from mtg_bot.training._timeout import run_with_timeout
        from mtg_bot.training._hang_progress import HangProgress
        progress = HangProgress()
        try:
            winner, samples, struct_rows = run_with_timeout(
                _play_one_game_recording,
                (adapter, strategies, rng, decision_cap, sample_every,
                 recorder, progress, tree_depth, tree_top_k,
                 action_encoding_version, structured_tokens,
                 structured_scalar_schema_version,
                 move_temperature, temperature_cutoff),
                per_game_timeout_sec,
            )
        except TimeoutError as exc:
            # Drop a structured hang dump before letting the worker-
            # crash sentinel handler convert this to an empty chunk.
            # ``hang_dump_dir`` is opt-in (default off) — without it
            # the timeout still works exactly as before.
            if hang_dump_dir is not None:
                from mtg_bot.training._hang_dump import (
                    dump_hang_diagnostics,
                )
                dump_hang_diagnostics(
                    hang_dump_dir,
                    base_seed=base_seed,
                    game_idx=game_idx,
                    engine_seed=engine_seed,
                    error_type="TimeoutError",
                    error_message=str(exc),
                    extra={
                        "phase": "data_gen",
                        "decision_cap": decision_cap,
                        "sample_every": sample_every,
                        "per_game_timeout_sec": per_game_timeout_sec,
                        "progress": progress.snapshot(),
                    },
                )
            # Attach the partial decisions count so the worker-level
            # handler can report "wedged at decision 3" vs "ground
            # through 900 decisions and blew the wall budget" — the
            # single most useful bit for triaging a timeout, and the
            # one self_play._play_one_match_game has carried since
            # the hang-dump work. Without it the self-play phase's
            # stderr line is just "timed out", which is exactly the
            # ambiguity ``_hang_progress`` exists to resolve.
            exc.decisions_completed = progress.decisions_completed  # type: ignore[attr-defined]
            raise
    else:
        winner, samples, struct_rows = _play_one_game_recording(
            adapter, strategies, rng, decision_cap, sample_every,
            replay_recorder=recorder,
            tree_depth=tree_depth, tree_top_k=tree_top_k,
            action_encoding_version=action_encoding_version,
            structured_tokens=structured_tokens,
            structured_scalar_schema_version=(
                structured_scalar_schema_version
            ),
            move_temperature=move_temperature,
            temperature_cutoff=temperature_cutoff,
        )

    if recorder is not None and replay_config is not None:
        replay = recorder.to_replay(metadata={
            "game_idx": game_idx,
            "base_seed": base_seed,
            "engine_seed": engine_seed,
            "winner": winner,
            "n_decisions": len(recorder.snapshots),
            "captured_at": _now_iso(),
            "annotated_with_nn_version": replay_config.nn_version_label,
        })
        save_replay(replay, replay_config.replay_path(base_seed, game_idx))

    if not samples:
        f, o, p, m = _empty_arrays()
        return f, o, p, m, winner, "completed", [], _empty_game_ids()
    outcome_label = _outcome_label(winner)
    features = np.stack(
        [f for f, _, _, _ in samples], axis=0,
    ).astype(_DTYPE_FEATURES)
    outcomes = np.asarray(
        [outcome_label[v] for _, v, _, _ in samples],
        dtype=_DTYPE_OUTCOMES,
    )
    policy_targets = np.stack(
        [p for _, _, p, _ in samples], axis=0,
    ).astype(_DTYPE_POLICY)
    action_masks = np.stack(
        [m for _, _, _, m in samples], axis=0,
    ).astype(np.bool_)
    game_ids = np.full(
        features.shape[0], game_idx, dtype=_DTYPE_GAME_IDS,
    )
    return (
        features, outcomes, policy_targets, action_masks, winner,
        "completed", struct_rows, game_ids,
    )


def _empty_game_ids() -> np.ndarray:
    """Zero-length ``game_ids`` array (empty / crashed / timed-out
    game chunks)."""
    return np.zeros((0,), dtype=_DTYPE_GAME_IDS)


def _chunk_game_ids(chunk: tuple) -> np.ndarray:
    """Per-row game ids for one chunk. Chunks produced by
    ``_generate_one_game_data`` carry them at index 7; legacy /
    hand-built chunks without that element get ``-1`` per row so a
    downstream consumer can still tell "unknown game" apart from a
    real ``game_idx``."""
    if len(chunk) > 7 and chunk[7] is not None:
        return np.asarray(chunk[7], dtype=_DTYPE_GAME_IDS)
    return np.full(chunk[0].shape[0], -1, dtype=_DTYPE_GAME_IDS)


def _concat_chunks(
    chunks: list[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concat per-game chunks into the final arrays
    ``(features, outcomes, policy_targets, action_masks, game_ids)``.
    Skips empty chunks (zero-decision games hit the cap)."""
    nonempty = [c for c in chunks if c[0].size > 0]
    if not nonempty:
        return _empty_arrays() + (_empty_game_ids(),)
    features = np.concatenate(
        [c[0] for c in nonempty], axis=0,
    ).astype(_DTYPE_FEATURES)
    outcomes = np.concatenate(
        [c[1] for c in nonempty], axis=0,
    ).astype(_DTYPE_OUTCOMES)
    policy_targets = np.concatenate(
        [c[2] for c in nonempty], axis=0,
    ).astype(_DTYPE_POLICY)
    action_masks = np.concatenate(
        [c[3] for c in nonempty], axis=0,
    ).astype(np.bool_)
    game_ids = np.concatenate(
        [_chunk_game_ids(c) for c in nonempty], axis=0,
    ).astype(_DTYPE_GAME_IDS)
    return features, outcomes, policy_targets, action_masks, game_ids


def _outcome_label(
    winner: PlayerID | None,
) -> dict[PlayerID, float]:
    """Per-viewer outcome label. Draw → 0 for both."""
    if winner is None:
        return {0: 0.0, 1: 0.0}
    return {0: 1.0 if winner == 0 else -1.0,
            1: 1.0 if winner == 1 else -1.0}


def _play_one_game_recording(
    adapter: MtgGameStateAdapter,
    strategies: dict[PlayerID, "object"],
    rng: random.Random,
    decision_cap: int,
    sample_every: int,
    replay_recorder: Optional[ReplayRecorder] = None,
    progress: Optional["object"] = None,
    tree_depth: int = 1,
    tree_top_k: Optional[int] = None,
    action_encoding_version: str = "positional_v1",
    structured_tokens: bool = False,
    structured_scalar_schema_version: Optional[int] = None,
    move_temperature: float = 1.0,
    temperature_cutoff: int = 30,
) -> tuple[
    PlayerID | None,
    list[tuple[np.ndarray, PlayerID, np.ndarray, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
    """Play one game with the decision cap, recording features at
    every Nth non-chance decision from both player perspectives.

    Self-play move temperature (AlphaZero-style): for the first
    ``temperature_cutoff`` NON-forced decisions of the game (a
    decision counts as non-forced when the actor has more than one
    legal option, whether or not it is a sample point), the PLAYED
    move at a sample point is sampled from the search's root visit
    counts with probability proportional to ``visits ** (1 /
    move_temperature)`` using the game ``rng`` (so it stays
    deterministic per seed). After the cutoff — or whenever
    ``move_temperature <= 0`` — the search's argmax is played, as
    before. The recorded policy TARGET is never tempered: it is
    always the normalized raw visit distribution. Only decisions
    legal at the actual root are eligible for sampling (PIMC visits
    can include world-specific decisions the root can't apply).

    When ``replay_recorder`` is set, also captures a snapshot of
    the engine state after each ``adapter.apply(decision)`` — both
    chance and player decisions. The recorder's annotators (eval
    values, etc.) run inline.

    Returns ``(winner_or_None, samples)`` where each sample is a
    4-tuple ``(features, viewer_id, policy_target, action_mask)``:

    - ``features``      : ``(FEATURE_DIM,)`` float32, viewer-side.
    - ``viewer_id``     : the player whose perspective this is.
    - ``policy_target`` : ``(ACTION_VOCAB_SIZE,)`` float32. For
      the active player's row at a sample point where
      ``choose_with_visits`` returned non-empty visits, this is
      the normalized visit distribution encoded into slot indices
      via ``action_encoding.encode``. All zeros for the data-
      doubling non-active-viewer row, and for any row from a
      strategy with no visit data (RandomChoice etc.). The
      trainer detects "no policy supervision" by checking
      ``policy_target.sum() < 0.5``.
    - ``action_mask``   : ``(ACTION_VOCAB_SIZE,)`` bool. True at
      slots that were legal at the sample point. All-False for
      rows without policy supervision (mask is irrelevant to
      value-only loss).
    """
    # Local import — action_encoding pulls in legal-action
    # enumeration, which is heavier than this file's outer surface.
    from mtg_bot.training.action_encoding import (
        ACTION_VOCAB_SIZE, EncodingOverflowError, encode, legal_action_mask,
    )
    if structured_tokens:
        # Local import — state_tokenizer pulls in the entire model
        # constant surface and ndarray construction helpers; only worth
        # it when the structured-token flag is enabled.
        from mtg_bot.training.state_tokenizer import tokenize_state
        from mtg_bot.training.structured_model import (
            DEFAULT_SCALAR_SCHEMA_VERSION,
        )
        tokenize_kwargs: dict = {
            "scalar_schema_version": (
                DEFAULT_SCALAR_SCHEMA_VERSION
                if structured_scalar_schema_version is None
                else int(structured_scalar_schema_version)
            ),
            # Explicit (== the tokenizer's default) because every shard
            # records the cap its printing ids were bucketed at — see
            # ``shard_n_printings_cap`` / ``save_dataset``.
            "n_printings_cap": shard_n_printings_cap(),
        }

    samples: list[
        tuple[np.ndarray, PlayerID, np.ndarray, np.ndarray]
    ] = []
    # Parallel list of structured-token dicts; one row per sample row
    # (two viewers per sample point, in the order they were appended
    # to ``samples``). Empty when structured_tokens=False — the caller
    # checks the flag rather than the list's content.
    struct_rows: list[dict[str, np.ndarray]] = []
    # Action history for the structured tokens is NOT tracked here: the
    # adapter records it in ``apply`` (``adapter.recent_actions``, see
    # ``mtg_bot.search.action_history``) and the NN evaluator reads
    # the same attribute at search time, so training rows and
    # inference tokenize from one source of truth by construction.
    zero_policy = np.zeros(ACTION_VOCAB_SIZE, dtype=_DTYPE_POLICY)
    zero_mask = np.zeros(ACTION_VOCAB_SIZE, dtype=np.bool_)
    # Skip the per-decision adapter clone for observe_action wiring
    # when no strategy implements it (the common case for PIMC /
    # heuristic / random self-play). Cloning is ~milliseconds per
    # priority moment but compounds to seconds across ~3000-decision
    # games — only pay it for belief-tracking ISMCTS strategies.
    observe_action_active = any(
        hasattr(s, "observe_action") for s in strategies.values()
    )
    # Number of non-forced (>1 legal option) player decisions seen
    # so far. Drives the temperature cutoff. Only maintained while
    # temperature sampling is still live — once past the cutoff (or
    # with T<=0) we stop paying for the legal-count probe.
    n_nonforced = 0
    temperature_live = move_temperature > 0 and temperature_cutoff > 0
    decision_idx = 0
    while not adapter.is_terminal() and decision_idx < decision_cap:
        # Cooperative stop for an abandoned (timed-out) game — see
        # mtg_bot.cancellation.
        raise_if_cancelled()
        active = adapter.active_player()
        # Search-tree annotation captured this iteration; threaded into
        # the recorder right after the snapshot is appended. Stays None
        # for chance moves and non-sample-point regular moves (the
        # cheap ``choose`` path doesn't expose tree stats; recording it
        # would require a second search).
        search_annotation: Optional[dict[str, Any]] = None
        if active == CHANCE_PLAYER:
            outcomes = adapter.chance_outcomes()
            choices = [d for d, _ in outcomes]
            weights = [w for _, w in outcomes]
            decision = rng.choices(choices, weights=weights, k=1)[0]
        else:
            sampling = decision_idx % sample_every == 0
            if sampling:
                # Capture visits, decision, AND (when supported) the
                # full search-tree stats in a SINGLE search.
                # ``choose_with_full_stats`` returns both the chosen
                # decision and per-candidate stats — we derive the
                # visit dict from candidates for policy supervision
                # and feed the candidates into the replay recorder's
                # search annotation. Strategies that don't expose the
                # richer interface (duck-typed test mocks, baseline
                # implementations) fall through to ``choose_with_
                # visits`` — the snapshot then carries ``search=None``
                # for those decisions, matching the contract on the
                # viewer side.
                _t0 = (
                    time.perf_counter() if replay_recorder is not None
                    else None
                )
                strategy = strategies[active]
                # Deep-tree capture: when the replay config asks for
                # depth > 1, prefer ``choose_with_subtree`` so the
                # annotation carries nested ``children`` per
                # candidate. Production captures (default depth=1)
                # take the same ``choose_with_full_stats`` fast path
                # they always have — zero extra work, no shape change.
                deep_capture = tree_depth > 1
                subtree_fn = (
                    getattr(strategy, "choose_with_subtree", None)
                    if deep_capture else None
                )
                full_stats_fn = getattr(strategy, "choose_with_full_stats", None)
                candidates: list[dict[str, Any]] = []
                if callable(subtree_fn):
                    chosen, candidates = subtree_fn(
                        adapter, active,
                        tree_depth=tree_depth, tree_top_k=tree_top_k,
                    )
                    if chosen is not None:
                        decision = chosen
                    else:
                        decision = list(adapter.legal_decisions())[0]
                    visits = {c["decision"]: c["visits"] for c in candidates}
                elif callable(full_stats_fn):
                    chosen, candidates = full_stats_fn(adapter, active)
                    if chosen is not None:
                        decision = chosen
                    else:
                        # No candidates (random/baseline strategy or
                        # search bottomed out). Pick a legal default
                        # without re-running the search.
                        decision = list(adapter.legal_decisions())[0]
                    visits = {c["decision"]: c["visits"] for c in candidates}
                else:
                    visits = strategy.choose_with_visits(adapter, active)
                    if visits:
                        decision = max(
                            visits.items(), key=lambda kv: kv[1],
                        )[0]
                    else:
                        decision = list(adapter.legal_decisions())[0]
                # Root-legal decision set. Needed by the policy
                # encoder below whenever ``visits`` is non-empty, and
                # by the temperature sampler; computed at most once.
                legal_now: Optional[set] = None
                if temperature_live and n_nonforced < temperature_cutoff:
                    legal_now = set(adapter.legal_decisions())
                    if len(legal_now) > 1:
                        n_nonforced += 1
                        if visits:
                            decision = _sample_move_with_temperature(
                                visits, legal_now, move_temperature,
                                rng, fallback=decision,
                            )
                if replay_recorder is not None and candidates:
                    elapsed_ms = (
                        int((time.perf_counter() - _t0) * 1000)
                        if _t0 is not None else None
                    )
                    search_annotation = build_search_annotation(
                        chosen=decision,
                        candidates=candidates,
                        strategy=strategy,
                        elapsed_ms=elapsed_ms,
                        deep_capture=deep_capture,
                    )
                # Encode visits → slot-indexed normalized dist +
                # legal mask once per sample point. Shared across
                # the two viewer rows below — only the active
                # row gets the policy supervision; the data-
                # doubling non-active row gets zeros (its
                # "policy" would be the active player's
                # distribution viewed from the wrong seat).
                engine = adapter.engine_state
                if visits:
                    active_policy = np.zeros(
                        ACTION_VOCAB_SIZE, dtype=_DTYPE_POLICY,
                    )
                    # Filter visits down to root-state legal decisions
                    # before encoding. PIMC aggregates visits across K
                    # determinized worlds; some decisions (notably
                    # exile-zone casts whose legality depends on
                    # hidden info — battlefield-granted warp filters,
                    # etc.) are legal in some determinizations but not
                    # the actual root. Without filtering, those visits
                    # put policy mass on slots the action_mask
                    # correctly excludes; the trainer's
                    # masked_fill(~mask, -1e9) then makes masked CE
                    # explode (~10k per affected row). The principled
                    # fix is ISMCTS — see docs/ISMCTS_MIGRATION.md —
                    # but this filter cleans the training data
                    # immediately. Decisions that overflow an encoding
                    # cap also get dropped here; if everything's gone,
                    # fall through to no-policy-supervision for this
                    # row.
                    if legal_now is None:
                        legal_now = set(adapter.legal_decisions())
                    counted = 0.0
                    for d, count in visits.items():
                        if d not in legal_now:
                            continue
                        try:
                            slot = encode(
                                d, engine, active,
                                version=action_encoding_version,
                            )
                        except EncodingOverflowError:
                            continue
                        active_policy[slot] += count
                        counted += count
                    if counted > 0:
                        active_policy /= counted
                        # Pass the adapter (not the raw engine) so
                        # ``_combat_phase`` resolves correctly via
                        # ``adapter.combat_phase`` and the attacker /
                        # blocker slots get enumerated when the
                        # sample point IS a combat sub-decision. Raw
                        # engine state has no ``combat_phase`` attr,
                        # so the combat branches in ``_enumerate_legal``
                        # short-circuit and combat-decision samples
                        # would carry policy mass on slots the mask
                        # said were illegal. The trainer's
                        # ``masked_fill(~mask, -1e9)`` then makes
                        # masked CE explode (~1e9 per affected row);
                        # gen0's v4 final policy_loss=613,242 was
                        # entirely this — see commit message.
                        active_mask = legal_action_mask(
                            adapter, active,
                            version=action_encoding_version,
                        )
                    else:
                        active_policy = zero_policy
                        active_mask = zero_mask
                else:
                    active_policy = zero_policy
                    active_mask = zero_mask
                # Dense value target: the search's root estimate for
                # the actor (searched sample points), else the
                # evaluator's static value (forced moves -- the
                # majority of sample points). The non-active viewer
                # row gets its negation (zero-sum). None -> NaN.
                root_value = _root_value_from_candidates(candidates)
                if root_value is None:
                    root_value = _static_value(strategy, adapter, active)
                for viewer in (0, 1):
                    feats = extract_features(engine, viewer)
                    if viewer == active and visits:
                        samples.append(
                            (feats, viewer, active_policy, active_mask)
                        )
                    else:
                        samples.append(
                            (feats, viewer, zero_policy, zero_mask)
                        )
                    if structured_tokens:
                        # Tokenize per-viewer with the adapter-carried
                        # history (already §3-filtered and bounded —
                        # the tokenizer's own filter + truncation is a
                        # no-op on it). Snapshot is the current engine
                        # state (sample is taken pre-apply, so the
                        # caller's view is "state right now, with this
                        # decision about to be made").
                        struct_row = tokenize_state(
                            engine, viewer_id=viewer,
                            recent_actions=adapter.recent_actions,
                            **tokenize_kwargs,
                        )
                        struct_row["root_value"] = (
                            np.float32(
                                root_value if viewer == active
                                else -root_value
                            )
                            if root_value is not None
                            else np.float32(np.nan)
                        )
                        struct_rows.append(struct_row)
            else:
                if (
                    temperature_live
                    and n_nonforced < temperature_cutoff
                    and _has_multiple_legal(adapter)
                ):
                    n_nonforced += 1
                decision = strategies[active].choose(adapter, active)
        # Snapshot the pre-apply state so observe_action can be fed
        # the priority-moment state, not the post-apply state. Skip
        # the clone (and the observe_action loop) entirely when no
        # strategy implements observe_action — the clone cost is
        # nontrivial across thousands of decisions.
        if observe_action_active and active != CHANCE_PLAYER:
            pre_apply_state = adapter.clone()
        else:
            pre_apply_state = None
        # ``apply`` also records the decision into
        # ``adapter.recent_actions`` (pre-apply snapshot + turn / step
        # / actor) for the structured-token history.
        adapter.apply(decision)
        if pre_apply_state is not None:
            for seat in (0, 1):
                strategy = strategies.get(seat)
                if strategy is None:
                    continue
                if hasattr(strategy, "observe_action"):
                    strategy.observe_action(
                        pre_apply_state, decision, active,
                    )
        if replay_recorder is not None:
            replay_recorder.record(adapter.engine_state, decision=decision)
            if search_annotation is not None:
                # Attach the search annotation to the PRE-apply snapshot —
                # i.e. the snapshot whose state was the search root. The
                # viewer's decision-detail panel renders the annotation
                # alongside the snapshot's state, so candidates must be
                # legal in THAT state. Attaching to the post-apply snap
                # (the one we just recorded) used to mismatch state vs.
                # candidates: e.g. a sorcery-window search at
                # PRECOMBAT_MAIN with active=priority showing 4
                # land-play candidates would land on the next snap whose
                # state has already advanced to the opp's instant-timing
                # priority window (1 candidate), making the displayed
                # annotation look like a no-op.
                #
                # Pre-apply snap index = ``len(snapshots) - 2`` (the snap
                # recorded BEFORE the just-appended post-apply one).
                # ``record`` always appends at least the initial snap
                # before this loop fires, so ``len - 2 >= 0`` for every
                # priority-action iteration that produces a search
                # annotation.
                replay_recorder.record_search_annotation(
                    len(replay_recorder.snapshots) - 2,
                    search_annotation,
                )
                # Soft size sanity check on deep-capture annotations.
                # No-op for production root-only captures; only fires
                # when ``children`` is populated.
                replay_recorder.maybe_warn_deep_annotation_size(
                    search_annotation,
                )
        decision_idx += 1
        # Update the optional shared progress sentinel so the timeout
        # handler (in the parent thread) can read decisions_completed
        # + last decision repr if this game ends up dumping. Pure
        # attribute writes — no lock, no allocation on the steady state.
        # n_legal=-1 means "not measured" — we'd have to enumerate
        # legal actions twice per decision otherwise (the search did it
        # internally, but the result isn't exposed back to us).
        if progress is not None:
            try:
                progress.record(active, decision, -1)
            except Exception:  # noqa: BLE001
                # Diagnostic capture must never break the game loop.
                pass
    winner = adapter.winner() if adapter.is_terminal() else None
    return winner, samples, struct_rows


def _has_multiple_legal(adapter: MtgGameStateAdapter) -> bool:
    """True when the actor has more than one legal decision. Pulls
    at most two items from the legal-decision iterator so the probe
    stays cheap on the non-sample-point path."""
    return len(list(itertools.islice(adapter.legal_decisions(), 2))) > 1


def _sample_move_with_temperature(
    visits: dict[Decision, int],
    legal: set[Decision],
    temperature: float,
    rng: random.Random,
    fallback: Decision,
) -> Decision:
    """Sample a decision from ``visits`` with probability
    proportional to ``visits ** (1 / temperature)``, restricted to
    decisions in ``legal`` with a positive count.

    - ``temperature <= 0`` returns the argmax (first-max in
      ``visits`` insertion order, matching the search's own
      ``max(...)`` tie-break).
    - No eligible candidates (every visited decision is illegal at
      the root, or all counts are zero) returns ``fallback``.
    - Weights are computed as ``(v / v_max) ** (1/T)`` so very
      small temperatures don't overflow; as ``T -> 0`` the sample
      converges on the argmax.

    Uses ``rng.choices`` so results are deterministic per game seed.
    """
    cands = [(d, v) for d, v in visits.items() if d in legal and v > 0]
    if not cands:
        return fallback
    if temperature <= 0 or len(cands) == 1:
        return max(cands, key=lambda kv: kv[1])[0]
    v_max = max(v for _, v in cands)
    inv_t = 1.0 / temperature
    weights = [(v / v_max) ** inv_t for _, v in cands]
    return rng.choices([d for d, _ in cands], weights=weights, k=1)[0]


def shard_n_printings_cap() -> int:
    """The printing-id bucketing cap data-gen tokenises at.

    Training shards are tokenised independently of any model (a
    heuristic corpus has none), at the tokenizer's default cap; the
    trainer builds its model at the same ``n_printings``. Every
    structured shard records this cap next to its card vocabulary
    (``card_vocab.SHARD_KEY``) so a trainer can never silently read ids
    bucketed for a different table size.
    """
    from mtg_bot.training.structured_model import DEFAULT_N_PRINTINGS

    return int(DEFAULT_N_PRINTINGS)


def shard_card_vocab_entries() -> dict[str, np.ndarray]:
    """The npz entries stamping a structured shard written by THIS
    process with the card vocabulary + cap its printing ids use."""
    from mtg_bot.training.card_vocab import (
        current_card_vocab,
        shard_card_vocab_arrays,
    )

    return shard_card_vocab_arrays(
        current_card_vocab(), shard_n_printings_cap(),
    )


def save_dataset(
    features: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
    *,
    policy_targets: Optional[np.ndarray] = None,
    action_masks: Optional[np.ndarray] = None,
    structured_token_type: Optional[np.ndarray] = None,
    structured_token_emb_ids: Optional[np.ndarray] = None,
    structured_token_scalars: Optional[np.ndarray] = None,
    structured_attention_mask: Optional[np.ndarray] = None,
    game_ids: Optional[np.ndarray] = None,
    root_values: Optional[np.ndarray] = None,
) -> None:
    """Save ``(features, outcomes)`` plus optional policy
    supervision, structured-token, and game-id arrays to a single
    ``.npz`` file.

    Schema (the policy / structured / game_ids keys exist only when
    supplied):

    - ``game_ids``       : ``(N,)`` int32 — the ``game_idx`` each
      row was generated from (unique per game WITHIN this file).
      The trainer groups rows by ``(file, game_id)`` so a game never
      straddles the train/val split. Files written before this key
      existed load fine; the loader falls back to treating the whole
      file as one group.

    - ``features``       : ``(N, FEATURE_DIM)`` float32
    - ``outcomes``       : ``(N,)`` float32 in ``{-1, 0, +1}``
    - ``policy_targets`` : ``(N, ACTION_VOCAB_SIZE)`` float32 —
      each row is a normalized visit distribution over the legal
      action slots. Rows with no policy supervision (data-doubled
      non-active-viewer rows, baseline-strategy rows) are emitted
      as all-zeros — the trainer detects ``policy_targets.sum(-1)
      < threshold`` to skip the policy loss for those rows.
    - ``action_masks``   : ``(N, ACTION_VOCAB_SIZE)`` bool — True
      where the slot was legal at the sample point. Rows with no
      policy supervision get all-False (trainer doesn't need the
      mask there).
    - ``structured_token_type``     : ``(N, max_seq_len)`` int64
    - ``structured_token_emb_ids``  : ``(N, max_seq_len, N_EMB_FIELDS)`` int64
    - ``structured_token_scalars``  : ``(N, max_seq_len, N_SCALAR_FIELDS)`` float32
    - ``structured_attention_mask`` : ``(N, max_seq_len)`` bool
    - ``structured_card_vocab``     : ``uint8`` utf-8 JSON — the card
      vocabulary (names + sha256) and ``n_printings_cap`` the
      structured printing ids were tokenised under
      (``card_vocab.SHARD_KEY``; written whenever the structured
      arrays are).

    Compressed — datasets are small even with policy supervision
    (zeros + bool masks compress aggressively).
    """
    n = features.shape[0]
    if outcomes.shape[0] != n:
        raise ValueError(
            f"features rows {n} != outcomes len {outcomes.shape[0]}"
        )
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"features have dim {features.shape[1]}, expected "
            f"FEATURE_DIM={FEATURE_DIM}"
        )
    if (policy_targets is None) != (action_masks is None):
        raise ValueError(
            "policy_targets and action_masks must be provided "
            "together (or both omitted)"
        )
    arrays: dict[str, np.ndarray] = {
        "features": features,
        "outcomes": outcomes,
    }
    if policy_targets is not None and action_masks is not None:
        if policy_targets.shape[0] != n or action_masks.shape[0] != n:
            raise ValueError(
                f"policy_targets / action_masks rows must equal "
                f"features rows ({n}); got "
                f"{policy_targets.shape[0]} / {action_masks.shape[0]}"
            )
        if policy_targets.shape[1] != action_masks.shape[1]:
            raise ValueError(
                f"policy_targets cols {policy_targets.shape[1]} != "
                f"action_masks cols {action_masks.shape[1]}"
            )
        arrays["policy_targets"] = policy_targets
        arrays["action_masks"] = action_masks
    struct_provided = [
        structured_token_type, structured_token_emb_ids,
        structured_token_scalars, structured_attention_mask,
    ]
    n_struct_provided = sum(1 for a in struct_provided if a is not None)
    if n_struct_provided not in (0, 4):
        raise ValueError(
            "All four structured-token arrays (token_type, "
            "token_emb_ids, token_scalars, attention_mask) must be "
            f"provided together or omitted together; got "
            f"{n_struct_provided}/4."
        )
    if n_struct_provided == 4:
        for name, arr in zip(_STRUCT_NPZ_KEYS, struct_provided):
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} rows {arr.shape[0]} != features rows {n}"
                )
            arrays[name] = arr
        # Record the id space the structured printing ids live in (the
        # card vocabulary + bucketing cap). The trainer refuses to mix
        # shards from different id spaces and stamps the trained model
        # with this vocabulary.
        arrays.update(shard_card_vocab_entries())
    if root_values is not None:
        # ``(N,)`` float32 search root value from the row's viewer
        # perspective, NaN where unknown. Optional: files without it
        # load with all-NaN root values (the trainer falls back to
        # the outcome for those rows).
        root_values = np.ascontiguousarray(root_values, dtype=np.float32)
        if root_values.shape != (n,):
            raise ValueError(
                f"root_values shape {root_values.shape} != ({n},)"
            )
        arrays["root_values"] = root_values
    if game_ids is not None:
        game_ids = np.asarray(game_ids)
        if game_ids.ndim != 1 or game_ids.shape[0] != n:
            raise ValueError(
                f"game_ids must be shape ({n},); got {game_ids.shape}"
            )
        arrays["game_ids"] = game_ids.astype(_DTYPE_GAME_IDS)
    np.savez_compressed(path, **arrays)


def load_dataset(
    path: Path,
    *,
    with_policy: bool = False,
    with_structured: bool = False,
    with_game_ids: bool = False,
) -> tuple[np.ndarray, ...]:
    """Load a single ``.npz`` saved by ``save_dataset``.

    ``with_game_ids=True`` appends one more element — the file's
    ``game_ids`` array, or ``None`` for files written before that
    key existed — to whichever tuple shape the other flags select.

    Default returns ``(features, outcomes)`` for backward compat
    with pre-A2.3 callers. Pass ``with_policy=True`` to also pull
    policy supervision arrays — ``(features, outcomes,
    policy_targets, action_masks)``. For value-only files (no
    ``policy_targets`` key in the npz), the policy arrays come
    back as ``None``; ``load_all_datasets`` is the helper that
    handles the value-only ↔ value+policy mixing transparently.

    Pass ``with_structured=True`` to also pull the structured-token
    arrays. Returns 8-tuple in this case:
    ``(features, outcomes, policy_targets, action_masks,
    structured_token_type, structured_token_emb_ids,
    structured_token_scalars, structured_attention_mask)``. For
    files written without the structured-token flag the four
    structured arrays come back as ``None`` so the trainer can
    differentiate "not in this file" from "all-zeros for these
    rows."
    """
    with np.load(path) as data:
        features = data["features"]
        outcomes = data["outcomes"]
        game_ids: Optional[np.ndarray] = (
            data["game_ids"] if "game_ids" in data.files else None
        )
        gid_tail: tuple = (game_ids,) if with_game_ids else ()
        if not with_policy and not with_structured:
            return (features, outcomes) + gid_tail
        policy_targets: Optional[np.ndarray] = (
            data["policy_targets"]
            if "policy_targets" in data.files else None
        )
        action_masks: Optional[np.ndarray] = (
            data["action_masks"]
            if "action_masks" in data.files else None
        )
        if not with_structured:
            return (
                features, outcomes, policy_targets, action_masks,
            ) + gid_tail
        struct_arrays: list[Optional[np.ndarray]] = []
        for key in _STRUCT_NPZ_KEYS:
            struct_arrays.append(
                data[key] if key in data.files else None
            )
        return (
            features, outcomes, policy_targets, action_masks,
            struct_arrays[0], struct_arrays[1],
            struct_arrays[2], struct_arrays[3],
        ) + gid_tail


# Stride between per-file group-id blocks. A file's rows get group
# id ``file_index * _GROUP_ID_FILE_STRIDE + game_id`` (or just
# ``file_index * _GROUP_ID_FILE_STRIDE`` when the file predates the
# ``game_ids`` key and is treated as one group). ``game_idx`` within
# a file is always far below 1e6 (self-play runs are O(1e4) games).
_GROUP_ID_FILE_STRIDE = 1_000_000


def file_group_ids(
    game_ids: Optional[np.ndarray],
    n_rows: int,
    file_index: int,
    *,
    path: Optional[Path] = None,
) -> np.ndarray:
    """Per-row group ids for one loaded file, unique across files.

    ``game_ids`` present → ``file_index * 1_000_000 + game_id`` per
    row (one group per game). ``None`` (file written before the key
    existed) → ``file_index * 1_000_000`` for every row: the whole
    file is a single group, which is the only leak-free choice when
    game boundaries are unknown. Logs at INFO which mode was used
    and how many groups the file contributed.
    """
    base = int(file_index) * _GROUP_ID_FILE_STRIDE
    label = str(path) if path is not None else f"file#{file_index}"
    if game_ids is None:
        _log.info(
            "group ids for %s: no game_ids key (legacy file); "
            "treating all %d rows as ONE group",
            label, n_rows,
        )
        return np.full(n_rows, base, dtype=np.int64)
    game_ids = np.asarray(game_ids)
    if game_ids.shape[0] != n_rows:
        raise ValueError(
            f"{label}: game_ids rows {game_ids.shape[0]} != "
            f"features rows {n_rows}"
        )
    if game_ids.size and int(game_ids.max()) >= _GROUP_ID_FILE_STRIDE:
        raise ValueError(
            f"{label}: game_id {int(game_ids.max())} exceeds the "
            f"per-file stride {_GROUP_ID_FILE_STRIDE}"
        )
    out = base + game_ids.astype(np.int64)
    _log.info(
        "group ids for %s: game_ids present; %d rows in %d game groups",
        label, n_rows, int(np.unique(out).shape[0]) if n_rows else 0,
    )
    return out


def load_all_datasets(
    directory: Path,
    *,
    include_policy: bool = False,
    max_files: int | None = None,
    include_group_ids: bool = False,
) -> tuple[np.ndarray, ...]:
    """Load and concatenate every ``*.npz`` in ``directory``.

    ``include_group_ids=True`` appends a ``(N,)`` int64 ``group_ids``
    array as the LAST tuple element: per-row game group ids unique
    across the loaded files (see :func:`file_group_ids`). Files
    without a ``game_ids`` key contribute one group each. Pass it to
    ``train_value_network(group_ids=...)`` for a game-level split.

    By default returns ``(features, outcomes)`` to preserve the
    pre-A2.3 value-only callers' contract — files that contain
    policy supervision have those arrays silently dropped. Pass
    ``include_policy=True`` to also pull policy_targets and
    action_masks back; value-only files are padded out with zero
    policy_targets / all-False action_masks so the concatenated
    arrays align row-for-row.

    The 4-tuple shape (when ``include_policy=True``) is what the
    A4-wiring trainer reads: it inspects ``policy_targets.sum(-1)
    > 0`` per row to decide which rows contribute to the policy
    loss.

    ``max_files`` (default ``None`` = load every match) caps the
    set of npz files to the ``N`` most recent by mtime. Used by
    the ``--window-recent`` training knob: as ``iterate_data/``
    accumulates over many iterations, training on the entire
    history slows linearly with size and biases the model toward
    early-iteration data. Picking the top-N most-recently-written
    files preserves the latest self-play distribution while
    capping load time. ``max_files=0`` raises (caller asked us to
    train on nothing); ``max_files`` greater than the available
    file count loads them all (no error — the caller's window is
    a maximum, not a minimum). Mtime is a fine proxy for
    generation time — the orchestrator writes each shard at
    self-play completion, so creation order matches mtime order.

    Empty arrays if the directory has no .npz files.
    """
    import sys

    from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE

    if max_files is not None and max_files <= 0:
        raise ValueError(
            f"max_files must be positive or None; got {max_files}"
        )

    all_paths = list(directory.glob("*.npz"))
    if max_files is not None and all_paths:
        # Sort by mtime descending so the freshest files come
        # first, take the top ``max_files``. Then re-sort the
        # kept subset by filename so concatenation order matches
        # the all-files contract (alphabetical within the loaded
        # subset). Stable concat order matters for reproducibility
        # when the same window is re-trained.
        all_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        kept = all_paths[:max_files]
        print(
            f"loading {len(kept)} of {len(all_paths)} files "
            f"(window-recent={max_files})",
            file=sys.stderr,
        )
        paths = sorted(kept)
    else:
        paths = sorted(all_paths)
    empty_gid_tail: tuple = (
        (np.zeros((0,), dtype=np.int64),) if include_group_ids else ()
    )
    if not paths:
        empty_features = np.zeros((0, FEATURE_DIM), dtype=_DTYPE_FEATURES)
        empty_outcomes = np.zeros((0,), dtype=_DTYPE_OUTCOMES)
        if not include_policy:
            return (empty_features, empty_outcomes) + empty_gid_tail
        return (
            empty_features,
            empty_outcomes,
            np.zeros((0, ACTION_VOCAB_SIZE), dtype=_DTYPE_POLICY),
            np.zeros((0, ACTION_VOCAB_SIZE), dtype=np.bool_),
        ) + empty_gid_tail
    feature_chunks: list[np.ndarray] = []
    outcome_chunks: list[np.ndarray] = []
    policy_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    for file_index, path in enumerate(paths):
        if include_policy:
            feats, outs, pols, masks, gids = load_dataset(
                path, with_policy=True, with_game_ids=True,
            )
            n_rows = feats.shape[0]
            if pols is None:
                # Old-format file — synthesize zero supervision so
                # the trainer treats every row as value-only.
                pols = np.zeros(
                    (n_rows, ACTION_VOCAB_SIZE), dtype=_DTYPE_POLICY,
                )
                masks = np.zeros(
                    (n_rows, ACTION_VOCAB_SIZE), dtype=np.bool_,
                )
            policy_chunks.append(pols)
            mask_chunks.append(masks)
        else:
            feats, outs, gids = load_dataset(path, with_game_ids=True)
        feature_chunks.append(feats)
        outcome_chunks.append(outs)
        if include_group_ids:
            group_chunks.append(
                file_group_ids(
                    gids, int(feats.shape[0]), file_index, path=path,
                )
            )
    features_all = np.concatenate(feature_chunks, axis=0)
    outcomes_all = np.concatenate(outcome_chunks, axis=0)
    gid_tail: tuple = (
        (np.concatenate(group_chunks, axis=0).astype(np.int64),)
        if include_group_ids else ()
    )
    if not include_policy:
        return (features_all, outcomes_all) + gid_tail
    return (
        features_all,
        outcomes_all,
        np.concatenate(policy_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0),
    ) + gid_tail
