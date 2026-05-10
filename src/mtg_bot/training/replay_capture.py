"""Replay capture during data generation.

``ReplayRecorder`` accumulates serialized snapshots as a game plays,
optionally annotating each snapshot with evaluator scores so the
viewer can spot eval anomalies (sudden swings, persistent disagreement
with intuition, eval saturating long before the game actually ends).

Two annotators ship today:

- ``heuristic_eval_annotator`` — scores each snapshot with the
  hand-crafted ``MtgHeuristicEvaluator`` from each player's
  perspective. Always cheap, always available.
- ``nn_eval_annotator`` — same shape, using ``NNEvaluator`` loaded
  from a registered version's weights. Optional — only attached
  when capture was launched with ``--annotate-with-nn-version``.

Annotations are merged into the snapshot dict under reserved keys
(``heuristic_eval``, ``nn_eval``). The replay file format documents
these as optional; readers that don't know about them ignore them.

Capture interacts with ``generate_training_data`` via the existing
per-game loop: at each ``adapter.apply(decision)`` the recorder is
told to ``record(state)``. Cost: one ``serialize_state`` call per
priority moment plus the annotators' work — for a typical PIMC
game (~270 decisions), captures ~5 ms of work and ~1 MB
compressed disk per game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from mtg_bot.game import GameState as EngineGameState
from mtg_bot.replay import REPLAY_FORMAT_VERSION, Replay
from mtg_bot.serialize import serialize_state
from mtg_bot.training.replay_decisions import decision_to_json

# An "annotator" is a function that takes the current engine state
# and returns a dict to merge into the snapshot. Returning an empty
# dict (or None) means "don't annotate this snapshot."
SnapshotAnnotator = Callable[[EngineGameState], dict[str, Any]]


@dataclass
class ReplayRecorder:
    """Accumulates snapshots through a game.

    ``annotators`` are applied in order; later annotators can
    overwrite earlier ones' keys (so put the most authoritative
    annotators last if they overlap).

    Not thread-safe (one recorder per game; the parallel data gen
    builds a fresh recorder per worker per game).
    """

    annotators: list[SnapshotAnnotator] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        state: EngineGameState,
        decision: Any = None,
    ) -> None:
        """Capture a snapshot of ``state``. Called once per priority
        moment by the data-gen loop (after each applied decision).

        ``decision`` is the action that was just applied to produce
        ``state`` — encoded into JSON via ``decision_to_json`` and
        stored under the ``decision`` key on the snapshot. The very
        first record (the pre-game initial state) is captured with
        ``decision=None`` because no action preceded it; the JSON
        ``decision`` field is then ``None`` on that snapshot.
        """
        snap = serialize_state(state)
        snap["decision"] = (
            None if decision is None else decision_to_json(decision)
        )
        # ``search`` defaults to None — populated retroactively by
        # ``record_search_annotation`` when a real search produced
        # this decision (the data-gen / self-play loops fire that
        # hook right before / right after the decision is applied,
        # threading per-candidate visits, Q-values, NN priors, and
        # leaf NN values into the snapshot for the viewer's
        # decision-detail panel). v3+ schema field; absent on v1/v2.
        snap["search"] = None
        for annotator in self.annotators:
            extra = annotator(state)
            if extra:
                snap.update(extra)
        self.snapshots.append(snap)

    def maybe_warn_deep_annotation_size(
        self, annotation: dict[str, Any], context: str = "",
    ) -> Optional[int]:
        """If ``annotation`` carries deep-capture data (any candidate
        has ``children`` populated as a non-empty list), measure its
        raw-JSON size and emit a soft warning when it exceeds
        ``_DEEP_ANNOTATION_SIZE_WARN_BYTES`` (50 KB). Returns the
        measured raw-byte size when the check ran, None when the
        annotation isn't deep (so callers can skip logging).

        Soft-only: never aborts the capture, never raises. Diagnostic
        captures are expected to be larger than production replays;
        the threshold catches pathological tree-explosion cases (bad
        top-k clamping, runaway expansion) without blocking the user."""
        candidates = annotation.get("candidates") or []
        deep = any(
            (c.get("children") or [])
            for c in candidates
        )
        if not deep:
            return None
        import json as _json
        raw = len(_json.dumps(annotation).encode("utf-8"))
        if raw > _DEEP_ANNOTATION_SIZE_WARN_BYTES:
            import warnings as _warnings
            _warnings.warn(
                f"deep search annotation is {raw} bytes "
                f"(>{_DEEP_ANNOTATION_SIZE_WARN_BYTES} threshold)"
                + (f" at {context}" if context else "")
                + " — consider lowering --capture-tree-depth or "
                "--capture-top-k",
                stacklevel=3,
            )
        return raw

    def record_search_annotation(
        self,
        snapshot_index: int,
        annotation: dict[str, Any],
    ) -> None:
        """Attach a search-tree summary to the snapshot at
        ``snapshot_index``. Called by the data-gen / self-play loop
        the moment a search-driven decision is picked, so the
        viewer can later show *why* the bot took that action.

        ``annotation`` is the dict shape documented at the
        ``Replay`` schema docstring (v3): ``chosen`` (json action),
        ``candidates`` (list of {action, visits, q, prior,
        leaf_value}), ``samples`` (PIMC fan-out), ``inner_nodes``
        (per-subsearch MCTS budget), ``elapsed_ms`` (wallclock for
        this decision's search; optional). The recorder stores it
        verbatim — JSON-safe shape is the caller's contract; we
        validate the snapshot index but not the inner shape so a
        future writer can carry extra fields without ceremony.

        Raises ``IndexError`` when the snapshot index doesn't exist
        yet — fail loud rather than silently dropping annotations,
        which would mask off-by-one bugs in the caller's bookkeeping.
        """
        if snapshot_index < 0 or snapshot_index >= len(self.snapshots):
            raise IndexError(
                f"snapshot_index {snapshot_index} out of range "
                f"(have {len(self.snapshots)} snapshots)"
            )
        self.snapshots[snapshot_index]["search"] = annotation

    def to_replay(self, metadata: dict[str, Any]) -> Replay:
        return Replay(
            format_version=REPLAY_FORMAT_VERSION,
            metadata=dict(metadata),
            snapshots=list(self.snapshots),
        )


def build_search_annotation(
    chosen: Any,
    candidates: list[dict[str, Any]],
    strategy: Any,
    elapsed_ms: Optional[int] = None,
    deep_capture: bool = False,
) -> dict[str, Any]:
    """Build the v3 ``search`` snapshot annotation from a strategy's
    ``choose_with_full_stats`` (or ``choose_with_subtree``) output.

    Encodes ``chosen`` and each candidate's ``decision`` via
    ``decision_to_json`` so the JSON payload is the same shape the
    viewer reads. ``samples`` and ``inner_nodes`` are introspected
    from the strategy when present (PIMC's ``samples`` for fan-out;
    PIMC's inner_factory result has ``budget_factory`` if it's an
    MCTS — best-effort, ``inner_nodes=None`` when the strategy isn't
    a PIMC-of-MCTS). ``elapsed_ms`` is the wallclock for this
    decision's whole search; passed in so callers can time the
    single search call without round-tripping the strategy.

    Candidates are returned sorted by visit count descending so the
    viewer can iterate in display order without re-sorting on the
    JS side.

    Schema decision (deep capture, v3 back-compat): each candidate
    gains an optional ``children: Optional[List[Candidate]]`` field.
    Production self-play (the default) leaves ``children=None`` on
    every candidate — readers ignore the field, the on-disk shape is
    bit-identical to pre-feature replays, and ``format_version``
    stays at v3. When the diagnostic capture script is invoked with
    ``--capture-tree-depth N`` (N > 1), the strategy returns
    candidates with a ``"subtree"`` key whose value is a recursive
    list of the same shape — we translate that here into the on-disk
    ``children`` field. ``deep_capture=True`` flips the default for
    the ``children`` field from None to ``[]`` so even leaf
    candidates with no expanded children render with a hidden
    chevron in the viewer (``[]`` is truthy-equivalent to "no
    children" in the renderer's ``c.children?.length > 0`` check).
    """
    sorted_candidates = sorted(
        candidates, key=lambda c: int(c["visits"]), reverse=True,
    )
    encoded = []
    for c in sorted_candidates:
        encoded.append({
            "action": decision_to_json(c["decision"]),
            "visits": int(c["visits"]),
            "q": (
                float(c["q"]) if c.get("q") is not None else None
            ),
            "prior": (
                float(c["prior"]) if c.get("prior") is not None
                else None
            ),
            "leaf_value": (
                float(c["leaf_value"])
                if c.get("leaf_value") is not None else None
            ),
            "children": _encode_children(c, deep_capture=deep_capture),
        })
    samples = getattr(strategy, "samples", 1)
    # PIMC.inner_factory is a zero-arg callable that returns an MCTS-
    # shaped strategy. MCTS exposes a budget_factory, also zero-arg,
    # returning a Budget that has a ``max_nodes`` attribute. We
    # introspect best-effort: if anything raises or is missing, leave
    # ``inner_nodes`` as None.
    inner_nodes: Optional[int] = None
    inner_factory = getattr(strategy, "inner_factory", None)
    if inner_factory is not None:
        try:
            inner_strategy = inner_factory()
            budget_factory = getattr(inner_strategy, "budget_factory", None)
            if budget_factory is not None:
                budget = budget_factory()
                if hasattr(budget, "max_nodes"):
                    inner_nodes = int(budget.max_nodes)
        except Exception:  # noqa: BLE001
            inner_nodes = None
    else:
        # Direct MCTS (no PIMC wrapper) — sample count = 1, budget on
        # the strategy itself.
        budget_factory = getattr(strategy, "budget_factory", None)
        if budget_factory is not None:
            try:
                budget = budget_factory()
                if hasattr(budget, "max_nodes"):
                    inner_nodes = int(budget.max_nodes)
            except Exception:  # noqa: BLE001
                inner_nodes = None
    return {
        "chosen": (
            None if chosen is None else decision_to_json(chosen)
        ),
        "candidates": encoded,
        "samples": int(samples),
        "inner_nodes": inner_nodes,
        "elapsed_ms": elapsed_ms,
    }


# Soft warning threshold for a single deep-mode snapshot's annotation
# raw-JSON size. The shallow (root-only) annotation budget is ~2 KB raw;
# deep captures (depth=3 top-k=5) are expected to land in the 10-30 KB
# range. 50 KB is a sanity ceiling — anything past that suggests the
# tree exploded (e.g., bad top-k clamping) and the user should re-check
# their flags rather than ship a multi-GB replay dir. Soft warning, not
# an abort.
_DEEP_ANNOTATION_SIZE_WARN_BYTES = 50_000


def _encode_children(
    src_candidate: dict[str, Any], deep_capture: bool,
) -> Optional[list[dict[str, Any]]]:
    """Translate the in-memory ``subtree`` key on a candidate into the
    on-disk ``children`` field. Returns ``None`` (no deep capture)
    when the candidate has no ``subtree`` *and* deep capture wasn't
    requested. Returns a recursively-encoded list otherwise.

    The return-None default is what makes deep capture purely additive:
    pre-feature replays and production captures emit
    ``children=None`` on every candidate, viewer renders identically
    to before. The empty-list case (``deep_capture=True`` but the
    candidate had no expanded grandchildren) is distinct from None
    and lets the viewer logic stay uniform.
    """
    sub = src_candidate.get("subtree")
    if sub is None:
        return [] if deep_capture else None
    return [
        {
            "action": decision_to_json(c["decision"]),
            "visits": int(c["visits"]),
            "q": (
                float(c["q"]) if c.get("q") is not None else None
            ),
            "prior": (
                float(c["prior"]) if c.get("prior") is not None
                else None
            ),
            "leaf_value": (
                float(c["leaf_value"])
                if c.get("leaf_value") is not None else None
            ),
            "children": _encode_children(c, deep_capture=True),
        }
        for c in sub
    ]


def _terminal_value(state: EngineGameState, player_id: int) -> float:
    """Return ±1/0 from ``player_id``'s perspective for a terminal
    engine state. Mirrors the protocol-side Evaluator's terminal
    short-circuit. Engine state uses ``is_over`` / ``winner_id`` /
    ``is_draw`` (different names than the protocol's
    ``is_terminal`` / ``winner``)."""
    if state.is_draw or state.winner_id is None:
        return 0.0
    return 1.0 if state.winner_id == player_id else -1.0


def heuristic_eval_annotator() -> SnapshotAnnotator:
    """Returns an annotator that adds ``heuristic_eval``: ``{p0,
    p1}`` to each snapshot. Uses the default heuristic weights;
    if you want to compare different weight settings, build the
    annotator with a custom evaluator yourself.

    Engine state is passed directly (not via an adapter), so this
    bypasses ``Evaluator.evaluate``'s protocol-side terminal check
    and uses the engine's ``is_over`` / ``winner_id`` directly."""
    # Lazy import to keep the search/evaluator import out of any code
    # path that doesn't actually capture replays.
    from mtg_bot.search.heuristic_evaluator import MtgHeuristicEvaluator
    evaluator = MtgHeuristicEvaluator()

    def annotate(state: EngineGameState) -> dict[str, Any]:
        if state.is_over():
            return {
                "heuristic_eval": {
                    "p0": _terminal_value(state, 0),
                    "p1": _terminal_value(state, 1),
                }
            }
        return {
            "heuristic_eval": {
                "p0": float(evaluator._evaluate_nonterminal(state, 0)),
                "p1": float(evaluator._evaluate_nonterminal(state, 1)),
            }
        }

    return annotate


def nn_eval_annotator(
    weights_path: Path,
    version_label: Optional[str] = None,
    hidden_sizes: Optional[tuple[int, ...]] = None,
    with_policy: bool = False,
    architecture: Optional[str] = None,
    policy_hidden_sizes: Optional[tuple[int, ...]] = None,
) -> SnapshotAnnotator:
    """Returns an annotator that adds ``nn_eval``: ``{p0, p1,
    version}`` to each snapshot. ``version`` is the registry version
    id (or the raw weights path) so consumers can tell which network
    produced the scores when comparing replays across training
    iterations.

    ``hidden_sizes`` and ``with_policy`` must match the saved weights'
    architecture (trunk shape and presence of a policy head). Without
    them, NNEvaluator builds the legacy default _ValueMLP and the
    state_dict load fails on any modern (dual-head, big-trunk)
    checkpoint. iterate_alphazero passes both through.

    Same engine-state-direct treatment as the heuristic annotator
    (bypasses the protocol's terminal short-circuit; uses engine
    fields directly)."""
    from mtg_bot.search.nn_evaluator import NNEvaluator
    nn_kwargs: dict[str, Any] = {"weights_path": weights_path}
    if hidden_sizes is not None:
        nn_kwargs["hidden_sizes"] = hidden_sizes
    if with_policy:
        nn_kwargs["with_policy"] = True
    if architecture is not None:
        nn_kwargs["architecture"] = architecture
    if policy_hidden_sizes is not None:
        nn_kwargs["policy_hidden_sizes"] = policy_hidden_sizes
    evaluator = NNEvaluator(**nn_kwargs)
    label = version_label if version_label is not None else str(weights_path)

    def annotate(state: EngineGameState) -> dict[str, Any]:
        if state.is_over():
            return {
                "nn_eval": {
                    "p0": _terminal_value(state, 0),
                    "p1": _terminal_value(state, 1),
                    "version": label,
                }
            }
        return {
            "nn_eval": {
                "p0": float(evaluator._evaluate_nonterminal(state, 0)),
                "p1": float(evaluator._evaluate_nonterminal(state, 1)),
                "version": label,
            }
        }

    return annotate
