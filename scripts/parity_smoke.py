"""Parity smoke: run one game per engine, compare trajectories.

Records every agent decision (choose_action / declare_attackers /
declare_blockers / choose_target / choose_discard / choose_legend_keep)
along with a state-hash before each decision. Two engines running the
same seed and same decks should produce byte-identical trajectories.

Usage:
    PYTHONPATH=src python scripts/parity_smoke.py --seed 42
    PYTHONPATH=src python scripts/parity_smoke.py --mode initial-state --seed 42
    PYTHONPATH=src python scripts/parity_smoke.py --mode trajectory --seed 42 --num-decisions 20
    PYTHONPATH=src python scripts/parity_smoke.py --mode multi-seed --num-seeds 10 --num-decisions 20

Modes:
    game           — Run a full game with RandomAgent on each engine.
                     Rust side stubs out until ``legal_decisions`` /
                     full ``apply_action`` coverage land. Default.
    initial-state  — Build a Rust synthetic state and compare its
                     native ``fingerprint()`` against a Python-side
                     re-implementation of ``hash_state`` that reads
                     the state via the round-21/22 snapshot getters.
                     Fast regression gate on the parity primitive: if
                     either engine ever drifts on the hash recipe,
                     this trips immediately.
    trajectory     — Rust-only. Build a synthetic state, walk N
                     decisions picked deterministically from
                     ``legal_decisions``, then clone the original
                     state and replay the same decision sequence —
                     assert the fingerprint trajectory is
                     byte-identical. Catches non-deterministic
                     mutation and clone-divergence bugs. Cross-engine
                     trajectory compare is deferred until Python's
                     ``legal.py:legal_actions`` and Rust's
                     ``legal_decisions`` have full coverage parity.
    multi-seed     — Run trajectory mode across K seeds and print an
                     aggregate summary table. Catches non-determinism
                     bugs that only surface at certain seeds — the
                     single-seed trajectory mode is necessary but not
                     sufficient since a fixed seed only exercises one
                     path through the legal space.
    field-diff     — Build a Rust synthetic state and print a
                     per-field OK/DIFFER diff between two parallel
                     views of it. Parity-debug primitive: when a
                     future trajectory smoke says "digests diverge at
                     decision 47", run this mode at that state to see
                     "life=20/19 — DIVERGE" instead of just two
                     mismatched digests. Supports
                     ``--inject-divergence FIELD`` to artificially
                     mutate one field on the right-hand view so the
                     diff machinery itself can be smoke-tested.
    cross-engine-game — Phase-5 prep. Drive Python AND Rust through
                     the SAME deterministic decision sequence one
                     step at a time on a Forest-only deck, comparing
                     per-step state via the field-diff machinery, and
                     report the first decision index where the two
                     engines diverge on any field. The smoke is
                     expected to trip on the known G1-G24 gaps in
                     ``docs/RUST_PARITY_GAPS.md`` — the harness's
                     deliverable is the measurement (first-divergence
                     index + which field), NOT a parity-clean run.
                     Robust to "Python can't apply Rust's chosen
                     action" (logged as a STOP row, not a crash).

Exit codes:
    0 — both engines agree (or only one engine ran successfully and
        printed its trajectory).
    1 — engines diverged. First-divergence index + a context window
        printed.
    2 — scaffold error (e.g., import failure).

Phase 1 status: Rust side is a stub that raises NotImplementedError
in ``--mode game``. The ``--mode initial-state`` path is fully wired
end-to-end as of round 23 (make_state + fingerprint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any


# ---------- state hashing -------------------------------------------------

def _zone_signature(zone: list[Any]) -> tuple:
    """Hashable signature of a zone (battlefield / hand / library / ...).

    Sensitive to order (zones are lists in Python — battlefield order is
    insertion timestamp, hand/library order matters for top-of-deck and
    "first match" tie-breaking). Uses (printing_id, tapped, summoning_sick,
    damage, counters, attached_to, controller_id) so per-permanent state
    that affects gameplay is captured.
    """
    sig = []
    for obj in zone:
        sig.append(
            (
                getattr(obj, "printing", None) and obj.printing.name,
                int(getattr(obj, "tapped", False)),
                int(getattr(obj, "summoning_sick", False)),
                getattr(obj, "damage", 0),
                getattr(obj, "counters", 0),
                getattr(obj, "attached_to", None),
                getattr(obj, "controller_id", None),
                getattr(obj, "id", None),
            )
        )
    return tuple(sig)


def _hash_parts(parts: list) -> str:
    """SHA1 of a JSON-serialized ``parts`` list, truncated to 12 hex chars.

    Factored out so the dict-shim variant for the Rust state and the
    classic attribute-walking variant for the Python state hit the same
    serializer + truncation. The Rust ``fingerprint()`` mirrors the
    Python ``json.dumps(parts, default=repr, sort_keys=False)`` exactly;
    keeping the JSON kwargs in one place protects against accidental
    drift (e.g. someone flipping `sort_keys=True` in a refactor would
    invalidate every pinned baseline).
    """
    blob = json.dumps(parts, default=repr, sort_keys=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def hash_state(state) -> str:
    """SHA1 of the gameplay-relevant invariants of a Python GameState.

    Attribute-walking variant — reads ``state.players[i].life`` /
    ``.hand`` / ... directly. The Rust mirror is :func:`_hash_rust_state`
    which reads the same shape through the round-21 / round-22 snapshot
    getters (``hand_snapshot`` / ``battlefield_snapshot`` / ...) instead
    of Python attributes.
    """
    parts: list = [
        state.turn_number,
        getattr(state, "step", None) and state.step.name if hasattr(getattr(state, "step", None), "name") else getattr(state, "step", None),
        state.active_player_idx,
        getattr(state, "priority_player_idx", None),
        state.is_over(),
        getattr(state, "is_draw", False),
        getattr(state, "winner_id", None),
    ]
    for p in state.players:
        parts.append((
            p.id,
            p.life,
            getattr(p, "has_lost", False),
            _zone_signature(p.hand),
            _zone_signature(p.library),
            _zone_signature(p.graveyard),
            _zone_signature(getattr(p, "exile", [])),
            tuple(sorted((p.mana_pool or {}).items())) if isinstance(getattr(p, "mana_pool", None), dict) else getattr(p, "mana_pool", None),
        ))
    # Battlefield is centralized on GameState in Python; objects carry
    # controller_id so per-player ownership is captured in _zone_signature.
    parts.append(_zone_signature(getattr(state, "battlefield", [])))
    parts.append(_zone_signature(getattr(state, "stack", [])))
    return _hash_parts(parts)


# ---------- Rust-state mirror of hash_state -------------------------------

def _rust_zone_obj_sig(d: dict) -> tuple:
    """Mirror ``_zone_signature``'s per-object 8-tuple from a snapshot dict.

    Rust ``hand_snapshot`` / ``library_snapshot`` / ``graveyard_snapshot``
    / ``exile_snapshot`` / ``battlefield_snapshot`` all return lists of
    dicts with the same key set:

        {id, printing_id, printing_name, tapped, summoning_sick,
         damage, counters, controller_id, zone}

    The Python ``_zone_signature`` produces an 8-tuple
    ``(printing_name | None, int(tapped), int(summoning_sick), damage,
       counters, attached_to, controller_id, id)``.

    Rust doesn't track ``attached_to`` yet (auras are out of scope for
    round 22/23), so we emit ``None`` for that slot — matching the
    ``Value::Null`` the Rust ``fingerprint()`` emits in the same position.
    """
    return (
        d["printing_name"],
        int(d["tapped"]),
        int(d["summoning_sick"]),
        d["damage"],
        d["counters"],
        None,  # attached_to — not modeled on the Rust side
        d["controller_id"],
        d["id"],
    )


def _rust_stack_obj_sig(d: dict) -> tuple:
    """Stack items have a different snapshot shape than zone permanents.

    The Rust ``stack_snapshot`` returns dicts of shape
    ``{source_obj_id, controller}`` (a stack item carries only an obj
    reference + its controller; per-permanent fields like tapped/damage
    are nonsensical for a card-on-the-stack).

    The Rust ``fingerprint()`` still emits stack items as the same
    8-tuple shape as zone objects — with the per-permanent slots zeroed
    or nulled — so the hash recipe stays uniform. We mirror that exactly:

        (None, 0, 0, 0, 0, None, controller, source_obj_id)
    """
    return (None, 0, 0, 0, 0, None, d["controller"], d["source_obj_id"])


def _hash_rust_state(state) -> str:
    """SHA1-12 fingerprint of a Rust GameState, reading via snapshot getters.

    Two ways to fingerprint a Rust state exist in this codebase:

    * ``state.fingerprint()`` — the native pymethod, builds the hash
      blob in Rust (see ``rust/src/state_ops_fingerprint.rs``).
    * ``_hash_rust_state(state)`` — this function, builds the hash
      blob in Python by walking the dict-form snapshots.

    They MUST produce byte-equal output. ``--mode initial-state``
    asserts that on a freshly built synthetic state — if either side
    ever drifts on the hash recipe, the harness trips immediately.
    """
    # Top-level scalars.
    step = state.step  # Rust pymethod already returns the Step.<NAME> string.
    parts: list = [
        state.turn_number,
        step,
        state.active_player_idx,
        state.priority_player_idx,
        state.is_over(),
        state.is_draw,
        state.winner_id,
    ]
    # Per-player tuples — life via get_life, has_lost derived from life<=0,
    # zone signatures via snapshot getters, mana pool as sorted dict items.
    for player_id in (0, 1):
        life = state.get_life(player_id)
        mana = state.mana_pool_snapshot(player_id)
        parts.append((
            player_id,
            life,
            life <= 0,
            tuple(_rust_zone_obj_sig(d) for d in state.hand_snapshot(player_id)),
            tuple(_rust_zone_obj_sig(d) for d in state.library_snapshot(player_id)),
            tuple(_rust_zone_obj_sig(d) for d in state.graveyard_snapshot(player_id)),
            tuple(_rust_zone_obj_sig(d) for d in state.exile_snapshot(player_id)),
            tuple(sorted(mana.items())),
        ))
    # Battlefield is split across players in Rust but `battlefield_snapshot`
    # already concatenates p0's then p1's in that order, which is what the
    # Rust fingerprint emits.
    parts.append(tuple(_rust_zone_obj_sig(d) for d in state.battlefield_snapshot()))
    parts.append(tuple(_rust_stack_obj_sig(d) for d in state.stack_snapshot()))
    return _hash_parts(parts)


# ---------- recording agent -----------------------------------------------

@dataclass
class DecisionRecord:
    method: str          # which Agent method was called
    player_id: int
    state_hash: str      # state hash BEFORE the decision
    decision: str        # repr of the returned decision
    n_legal: int         # how many legal options were offered

    def as_tuple(self) -> tuple:
        # Used for trajectory hashing. State hash + decision is enough.
        return (self.method, self.player_id, self.state_hash, self.decision, self.n_legal)


class RecordingAgent:
    """Wraps another Agent and records every method call it receives.

    Forwards ALL Agent methods via ``__getattr__`` so we don't need to
    enumerate ``choose_action`` / ``choose_permanent_to_tap`` / ... by
    hand — there are 25+ of them on the Agent base class and missing
    one silently breaks games. Every call gets a DecisionRecord with
    the state-hash before the call and the repr of the chosen result.
    """

    name = "RecordingAgent"

    def __init__(self, inner, log: list[DecisionRecord]):
        # Use object.__setattr__ to avoid triggering our own
        # ``__getattr__`` proxy during construction.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str):
        # ``__getattr__`` only fires when normal lookup fails — so every
        # Agent method (which we don't define here) lands here.
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            # Convention across every Agent method: arg0 = state, arg1 =
            # player_id. Some methods take a 3rd "legal options" arg —
            # peek for an int-able length when available.
            state = args[0] if args else None
            player_id = args[1] if len(args) > 1 and isinstance(args[1], int) else -1
            n_legal = -1
            if len(args) >= 3:
                try:
                    n_legal = len(args[2])
                except TypeError:
                    pass
            result = attr(*args, **kwargs)
            self._log.append(
                DecisionRecord(
                    method=name,
                    player_id=player_id,
                    state_hash=hash_state(state) if state is not None else "",
                    decision=repr(result),
                    n_legal=n_legal,
                )
            )
            return result

        return wrapper


# ---------- engine runners ------------------------------------------------

@dataclass
class TrajectoryResult:
    engine: str
    log: list[DecisionRecord] = field(default_factory=list)
    final_hash: str | None = None
    decisions: int = 0
    turn: int = 0
    winner: int | None = None
    is_draw: bool = False
    error: str | None = None

    def trajectory_hash(self) -> str | None:
        if self.error is not None:
            return None
        blob = json.dumps([r.as_tuple() for r in self.log] + [self.final_hash], default=repr).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:12]


def run_python_engine(seed: int, colors: list[str], max_turns: int) -> TrajectoryResult:
    res = TrajectoryResult(engine="python")
    try:
        from mtg_bot.simulate import simulate_game
        from mtg_bot.agents import RandomAgent

        sub_rng = random.Random(seed)
        inner_agents = [RandomAgent(rng=random.Random(sub_rng.randint(0, 10**9))) for _ in colors]
        wrapped = [RecordingAgent(a, res.log) for a in inner_agents]

        state = simulate_game(
            agents=wrapped,
            colors=colors,
            seed=seed,
            max_turns=max_turns,
            verbose=False,
        )
        res.final_hash = hash_state(state)
        res.decisions = len(res.log)
        res.turn = state.turn_number
        res.winner = getattr(state, "winner_id", None)
        res.is_draw = getattr(state, "is_draw", False)
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return res


def run_rust_engine(seed: int, colors: list[str], max_turns: int) -> TrajectoryResult:
    """Phase 1 stub: raises NotImplementedError.

    Phase 2 wires this to a Rust GameState exposed via PyO3 — a drop-in
    for the Python GameState that emits Action objects identical to the
    Python ones for the same input. Until then the harness reports
    `error=NotImplementedError(setup)` and the divergence index is
    `setup` (i.e., we never even started).
    """
    res = TrajectoryResult(engine="rust")
    res.error = "NotImplementedError: Rust GameState bridge not built yet (Phase 2)"
    return res


# ---------- comparison ----------------------------------------------------

def find_first_divergence(a: TrajectoryResult, b: TrajectoryResult) -> int | str | None:
    """Return the index of the first divergent record, or 'setup' / None."""
    if a.error and b.error:
        return "both_errored"
    if a.error:
        return f"setup ({a.engine})"
    if b.error:
        return f"setup ({b.engine})"
    n = min(len(a.log), len(b.log))
    for i in range(n):
        if a.log[i].as_tuple() != b.log[i].as_tuple():
            return i
    if len(a.log) != len(b.log):
        return n
    if a.final_hash != b.final_hash:
        return "final_hash"
    return None


def print_result(res: TrajectoryResult) -> None:
    if res.error:
        first_line = res.error.splitlines()[0]
        print(f"[{res.engine:>6}] ERROR: {first_line}")
    else:
        outcome = (
            "draw" if res.is_draw
            else f"p{res.winner} wins" if res.winner is not None
            else "max_turns_hit"
        )
        print(
            f"[{res.engine:>6}] {res.decisions} decisions, "
            f"turn {res.turn}, {outcome}, "
            f"trajectory_hash={res.trajectory_hash()}"
        )


# ---------- initial-state mode --------------------------------------------

def run_initial_state_compare(
    seed: int,
    num_battlefield: int = 10,
    num_hand: int = 7,
    num_library: int = 50,
) -> int:
    """Build a Rust synthetic state and compare its native fingerprint to
    a Python-side reimplementation of the same hash recipe.

    The ``seed`` argument is currently informational — ``make_synthetic_state``
    is deterministic (no shuffling), so the seed doesn't shape the state.
    It's accepted so the CLI surface matches the game mode and to leave
    room for a future variant that calls ``shuffle_library_with_seed``
    or otherwise threads randomness into the test state.

    Returns 0 on PASS, 1 on FAIL, 2 on import error.
    """
    try:
        import mtg_bot_engine
    except ImportError as e:
        print(f"[initial-state] ERROR: mtg_bot_engine not built ({e})")
        print("  -> run `maturin develop --release -m rust/Cargo.toml`")
        return 2

    state = mtg_bot_engine.make_synthetic_state(num_battlefield, num_hand, num_library)
    rust_fp = state.fingerprint()
    shim_fp = _hash_rust_state(state)

    if rust_fp == shim_fp:
        print(
            f"PARITY: PASS — fingerprint match ({rust_fp}) "
            f"[bf={num_battlefield}, hand={num_hand}, lib={num_library}, seed={seed}]"
        )
        return 0

    print(
        f"PARITY: FAIL — fingerprint mismatch on synthetic state "
        f"[bf={num_battlefield}, hand={num_hand}, lib={num_library}, seed={seed}]"
    )
    print(f"  rust.fingerprint()       = {rust_fp}")
    print(f"  _hash_rust_state(state)  = {shim_fp}")
    print("  -> the Rust fingerprint() recipe and the Python shim have drifted.")
    print("     Inspect rust/src/state_ops_fingerprint.rs against")
    print("     scripts/parity_smoke.py:_hash_rust_state and reconcile.")
    return 1


# ---------- trajectory mode -----------------------------------------------


def _decision_summary(action: dict) -> str:
    """Render an Action dict as a short human-readable label.

    Used in failure diagnostics — the diff between the original and
    clone-replay paths cites the decision kind + card_id (when present)
    so a reader can tell at a glance whether the divergence is on a
    play_land, cast_spell, or pass_priority.
    """
    kind = action.get("kind", "?")
    if "card_id" in action:
        return f"{kind}(card_id={action['card_id']})"
    return kind


def _walk_trajectory(state, seed: int, num_decisions: int) -> tuple[list[dict], list[str]]:
    """Walk a Rust state through ``num_decisions`` deterministic decisions.

    Returns ``(decisions_taken, fingerprint_trajectory)`` where:

    - ``decisions_taken[i]`` is the action dict applied at step i.
    - ``fingerprint_trajectory`` is the list of fingerprints in
      ``[fp_initial, fp_after_step_0, fp_after_step_1, ...]`` order.
      Length is ``num_decisions + 1``.

    Decision selection: a single ``random.Random(seed)`` instance is
    seeded once and ``rng.choice(legal)`` is called per step. This is
    bit-identical across runs with the same seed.

    Stubbed-dispatch handling: ``ActivateAbility`` currently raises
    ``ValueError("ActivateAbility dispatch not implemented yet ...")``
    when applied (round 24 ships the variant + bridge but not the
    primitive). Such failures are caught and the offending action is
    re-picked — the rng state advances on each re-pick so the choice
    is still deterministic. After 16 consecutive re-pick attempts at
    one step we abort with RuntimeError (defensive: every legal action
    being a stubbed dispatch is a real bug, not a transient skip).

    Empty ``legal_decisions`` is a contract violation — PassPriority
    is always supposed to be at index 0 — and aborts immediately.
    """
    rng = random.Random(seed)
    decisions_taken: list[dict] = []
    trajectory: list[str] = [state.fingerprint()]

    for step in range(num_decisions):
        # Inner pick-loop: re-roll on a stubbed dispatch failure.
        max_retries = 16
        applied = False
        last_error: Exception | None = None
        for _ in range(max_retries):
            legal = state.legal_decisions(state.get_priority_player())
            if not legal:
                raise RuntimeError(
                    f"step {step}: legal_decisions returned an empty list "
                    f"(contract violation — PassPriority should always emit)"
                )
            choice = rng.choice(legal)
            try:
                state.apply_action(choice)
                decisions_taken.append(choice)
                trajectory.append(state.fingerprint())
                applied = True
                break
            except ValueError as e:
                # Stubbed dispatch (ActivateAbility today) — re-pick.
                # rng has already advanced from the rng.choice call, so
                # the retry picks a different element from the same
                # legal list (or, on the same element, the loop tries
                # again with a fresh choice).
                last_error = e
                continue
        if not applied:
            raise RuntimeError(
                f"step {step}: gave up after {max_retries} re-pick attempts "
                f"(every legal action raised; last error: {last_error!r})"
            )

    return decisions_taken, trajectory


def _replay_trajectory(state, decisions: list[dict]) -> list[str]:
    """Replay a recorded decision sequence on a fresh state and return
    the resulting fingerprint trajectory.

    Used for the clone-replay determinism check: walk the original
    state, clone it, then apply the SAME decision sequence to the
    clone and verify the fingerprint trajectory matches byte-for-byte.
    """
    trajectory = [state.fingerprint()]
    for action in decisions:
        state.apply_action(action)
        trajectory.append(state.fingerprint())
    return trajectory


@dataclass
class TrajectorySeedResult:
    """One seed's outcome from the trajectory walk + clone-replay check.

    Used by both ``run_trajectory_mode`` (single-seed) and
    ``run_multi_seed_mode`` (K seeds). The single-seed runner prints
    one of these and translates to an exit code; the multi-seed
    runner collects a list of them and prints an aggregate table.

    Fields:
      seed              — the seed the walk was driven with.
      n_decisions       — number of decisions actually applied. Equals
                          ``num_decisions`` on success; on a walk-time
                          ``RuntimeError`` it's the count completed
                          before the abort.
      final_fingerprint — last fingerprint in the original trajectory,
                          or ``None`` if the walk failed before any
                          state was produced.
      clone_match       — True iff the clone-replay trajectory is
                          byte-equal to the original. False on either
                          trajectory mismatch OR a clone-side
                          exception that the original walk didn't hit.
      error             — short error string if anything went wrong
                          (walk RuntimeError, clone-replay exception),
                          else None. Used for the "errors" column in
                          the multi-seed table.
      orig_trajectory   — full original trajectory (initial + after
                          each step). Populated when the walk
                          succeeded; left as an empty list on walk
                          failure. Carried so the single-seed runner
                          can print the divergence-step diagnostic
                          without recomputing.
      clone_trajectory  — full clone-replay trajectory. Empty list if
                          the clone-replay raised.
      decisions         — the decisions actually applied (so the
                          divergence-step diagnostic can render the
                          triggering action).
    """

    seed: int
    n_decisions: int = 0
    final_fingerprint: str | None = None
    clone_match: bool = False
    error: str | None = None
    orig_trajectory: list[str] = field(default_factory=list)
    clone_trajectory: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)


def _run_single_seed_trajectory(
    seed: int,
    num_decisions: int,
    num_battlefield: int,
    num_hand: int,
    num_library: int,
) -> TrajectorySeedResult:
    """Run the trajectory walk + clone-replay check for one seed and
    return a result struct (no printing — the caller decides whether
    to print a single-seed line or aggregate into a multi-seed table).

    Assumes ``mtg_bot_engine`` is importable; the caller is responsible
    for the up-front import-error handling.
    """
    import mtg_bot_engine

    res = TrajectorySeedResult(seed=seed)
    state = mtg_bot_engine.make_synthetic_state(
        num_battlefield, num_hand, num_library
    )
    # Clone BEFORE any mutation so the clone is byte-identical to the
    # starting state. If we cloned after walking, we'd be cloning the
    # post-walk state — meaningless for replay.
    clone = state.clone_once()

    try:
        decisions, orig_trajectory = _walk_trajectory(state, seed, num_decisions)
    except RuntimeError as e:
        res.error = f"walk: {type(e).__name__}: {e}"
        return res

    res.decisions = decisions
    res.orig_trajectory = orig_trajectory
    res.n_decisions = len(decisions)
    res.final_fingerprint = orig_trajectory[-1]

    try:
        clone_trajectory = _replay_trajectory(clone, decisions)
    except Exception as e:
        res.error = f"clone-replay: {type(e).__name__}: {e}"
        res.clone_match = False
        return res

    res.clone_trajectory = clone_trajectory
    res.clone_match = orig_trajectory == clone_trajectory
    return res


def run_trajectory_mode(
    seed: int,
    num_decisions: int,
    num_battlefield: int = 2,
    num_hand: int = 3,
    num_library: int = 5,
) -> int:
    """Run a Rust-only trajectory walk + clone-replay determinism check.

    Steps:
      1. Build a synthetic Rust state.
      2. Clone it BEFORE walking so the clone is byte-identical to
         the pre-walk state.
      3. Walk ``num_decisions`` decisions deterministically on the
         original state, recording each decision + fingerprint.
      4. Replay the same decision sequence on the clone.
      5. Assert the two fingerprint trajectories match step-by-step.
         Report the first divergence on failure.

    Returns 0 on PASS, 1 on FAIL, 2 on import / contract error.

    Thin wrapper around :func:`_run_single_seed_trajectory` — the
    helper does the actual walk + replay and returns a result struct;
    this function prints the single-seed status line and translates to
    an exit code. ``run_multi_seed_mode`` calls the helper directly
    so it can aggregate K results without printing per-seed full
    diagnostics.
    """
    try:
        import mtg_bot_engine  # noqa: F401  (used by helper)
    except ImportError as e:
        print(f"[trajectory] ERROR: mtg_bot_engine not built ({e})")
        print("  -> run `maturin develop --release -m rust/Cargo.toml`")
        return 2

    print("[python] (skipped — trajectory mode runs Rust-only for now)")

    result = _run_single_seed_trajectory(
        seed=seed,
        num_decisions=num_decisions,
        num_battlefield=num_battlefield,
        num_hand=num_hand,
        num_library=num_library,
    )

    # Walk-time failure (RuntimeError from _walk_trajectory) — abort
    # with scaffold-error exit code, since this signals a contract
    # violation (e.g. empty legal_decisions) rather than a parity bug.
    if result.error is not None and not result.orig_trajectory:
        print(f"[  rust] ERROR during trajectory walk: {result.error}")
        return 2

    # Clone-replay exception (original walk succeeded but clone raised):
    # itself a divergence — report + return 1.
    if result.error is not None and not result.clone_trajectory:
        print(
            f"[  rust] trajectory of {num_decisions} decisions, "
            f"final_hash={result.final_fingerprint}, clone-replay-matches=False"
        )
        print(
            f"PARITY: FAIL — clone-replay raised "
            f"{result.error} where original did not"
        )
        return 1

    print(
        f"[  rust] trajectory of {num_decisions} decisions, "
        f"final_hash={result.final_fingerprint}, "
        f"clone-replay-matches={result.clone_match}"
    )

    if result.clone_match:
        print(f"PARITY: PASS — Rust trajectory deterministic across seed={seed}")
        return 0

    # Find the first divergence step.
    orig_trajectory = result.orig_trajectory
    clone_trajectory = result.clone_trajectory
    decisions = result.decisions
    div_step = next(
        (
            i
            for i, (a, b) in enumerate(zip(orig_trajectory, clone_trajectory))
            if a != b
        ),
        min(len(orig_trajectory), len(clone_trajectory)),
    )
    print(f"PARITY: FAIL — clone-replay diverged at step {div_step}")
    # Decision at index `div_step - 1` is the one that caused the
    # divergence (it's the apply between trajectory[div_step - 1] and
    # trajectory[div_step]). At step 0 (initial fingerprint mismatch
    # before any decision) there's no triggering decision.
    if div_step == 0:
        print(
            f"  step 0 original: fp={orig_trajectory[0]} (pre-decision)"
        )
        print(
            f"  step 0 clone:    fp={clone_trajectory[0]} (pre-decision)"
        )
    else:
        decision_idx = div_step - 1
        decision_repr = _decision_summary(decisions[decision_idx])
        print(
            f"  step {div_step} original: fp={orig_trajectory[div_step]}, "
            f"decision={decision_repr}"
        )
        print(
            f"  step {div_step} clone:    fp={clone_trajectory[div_step]}, "
            f"decision={decision_repr}"
        )
    return 1


# ---------- multi-seed mode -----------------------------------------------


def run_multi_seed_mode(
    num_seeds: int,
    seed_start: int,
    num_decisions: int,
    num_battlefield: int = 2,
    num_hand: int = 3,
    num_library: int = 5,
) -> int:
    """Run trajectory mode across K seeds and print an aggregate summary.

    For each seed in ``[seed_start, seed_start + num_seeds)``:
      1. Build a fresh synthetic Rust state.
      2. Walk ``num_decisions`` deterministic decisions.
      3. Clone-replay them on a pre-walk clone.
      4. Record (seed, n_decisions, final_fp, clone_match, error).

    Prints a per-seed table + aggregate stats, then a final PASS/FAIL
    summary line. Returns 0 if every seed clone-replay-matched with
    no errors; 1 if any seed had a divergence or error; 2 on import
    failure.

    Catches non-determinism bugs that only manifest at specific
    seeds — the single-seed mode is necessary but not sufficient
    because a fixed seed only exercises one path through the legal
    space. By sweeping K seeds we tile that space and the chance
    that a real nondeterminism bug stays hidden drops sharply with K.
    """
    try:
        import mtg_bot_engine  # noqa: F401  (used by helper)
    except ImportError as e:
        print(f"[multi-seed] ERROR: mtg_bot_engine not built ({e})")
        print("  -> run `maturin develop --release -m rust/Cargo.toml`")
        return 2

    seed_end = seed_start + num_seeds
    print(
        f"[multi-seed] running {num_seeds} seeds ({seed_start}..{seed_end - 1}), "
        f"{num_decisions} decisions each, "
        f"board=({num_battlefield},{num_hand},{num_library})"
    )
    print()
    # Header: keep column widths in sync with the data rows below so
    # the table reads cleanly in a terminal.
    print("seed |   decisions | fingerprint  | clone-match | errors")
    print("-----|-------------|--------------|-------------|-------")

    results: list[TrajectorySeedResult] = []
    for seed in range(seed_start, seed_end):
        result = _run_single_seed_trajectory(
            seed=seed,
            num_decisions=num_decisions,
            num_battlefield=num_battlefield,
            num_hand=num_hand,
            num_library=num_library,
        )
        results.append(result)
        fp_display = result.final_fingerprint if result.final_fingerprint else "-"
        err_display = result.error if result.error else "-"
        print(
            f"{seed:>4} | {result.n_decisions:>11} | {fp_display:<12} | "
            f"{str(result.clone_match):<11} | {err_display}"
        )

    # Aggregate stats.
    all_match = all(r.clone_match for r in results)
    n_errors = sum(1 for r in results if r.error is not None)
    fingerprints = [r.final_fingerprint for r in results if r.final_fingerprint]
    unique_fingerprints = len(set(fingerprints))
    failing_seeds = [r for r in results if not r.clone_match or r.error is not None]

    uniq_note = ""
    if fingerprints:
        if unique_fingerprints == len(fingerprints):
            uniq_note = "  (every seed produced a unique trajectory)"
        else:
            uniq_note = (
                f"  ({len(fingerprints) - unique_fingerprints} collisions across "
                f"{len(fingerprints)} runs)"
            )

    print()
    print("aggregate:")
    print(f"  total seeds:       {num_seeds}")
    print(f"  all clone-match:   {all_match}")
    print(f"  unique fingerprints: {unique_fingerprints}{uniq_note}")
    print(f"  total errors:      {n_errors}")
    print()

    if all_match and n_errors == 0:
        print(
            f"PARITY: PASS — all {num_seeds} seeds clone-replay-matched"
        )
        return 0

    n_failed = len(failing_seeds)
    print(f"PARITY: FAIL — {n_failed} of {num_seeds} seeds failed")
    for r in failing_seeds:
        if r.error is not None:
            print(f"  seed {r.seed}: {r.error}")
        else:
            # No exception but clone_match is False — a fingerprint
            # mismatch between original and clone-replay. Cite the
            # first-divergence step the way single-seed mode does.
            div_step = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(r.orig_trajectory, r.clone_trajectory)
                    )
                    if a != b
                ),
                min(len(r.orig_trajectory), len(r.clone_trajectory)),
            )
            print(f"  seed {r.seed}: clone-replay diverged at step {div_step}")
    return 1


# ---------- field-diff mode ----------------------------------------------


# Canonical ordering of fields the field-diff mode reports on. Mirrors the
# fields that ``_hash_rust_state`` folds into its hash recipe, plus the
# scalar/zone fields the task brief specifically names. Keeping this list
# in module scope makes it easy to assert "every field in _hash_parts shows
# up in the diff table" in tests + lets the inject-divergence flag accept
# the same field names.
FIELD_DIFF_FIELDS: tuple[str, ...] = (
    "turn",
    "step",
    "phase",
    "active_player",
    "priority_player",
    "is_over",
    "is_draw",
    "winner_id",
    "life_p0",
    "life_p1",
    "hand_p0",
    "hand_p1",
    "library_p0",
    "library_p1",
    "graveyard_p0",
    "graveyard_p1",
    "exile_p0",
    "exile_p1",
    "mana_pool_p0",
    "mana_pool_p1",
    "battlefield",
    "stack",
)


def _build_field_view(state) -> dict:
    """Build a field-keyed view of a Rust GameState from its snapshot getters.

    Returns an ordered dict (insertion-order = ``FIELD_DIFF_FIELDS`` order)
    keyed by field name, with values being the comparable signatures:

      * scalar fields (life, active_player, ...) -> the raw value
      * zone fields (hand_p0, battlefield, ...) -> tuple of per-object
        8-tuples produced by ``_rust_zone_obj_sig``
      * stack -> tuple of per-stack-item 8-tuples produced by
        ``_rust_stack_obj_sig``
      * mana_pool_pN -> tuple of sorted ``(symbol, count)`` items

    Both the "python-side recipe walk" and the "Rust-direct snapshot walk"
    feed off ``_build_field_view`` — by construction both views start
    field-equal on a vanilla state, which is the field-diff mode's
    null-hypothesis baseline. The mode's value comes from the
    inject-divergence flag (smoke-test the smoke) and from a future
    bridge implementation that walks the Python GameState through a
    parallel ``_build_python_field_view`` -- left as a TODO for the next
    slice once Python-side snapshot getters land.

    Step and phase are both surfaced — Rust's ``state.step`` returns the
    enum string (e.g. ``"UNTAP"``) which corresponds to a sub-step; the
    ``phase`` field uses the same string today (we don't yet have a
    separate phase enum on the Rust side) but is plumbed in as its own
    diff slot so a future "phase != step" expansion doesn't break the
    table layout.
    """
    view: dict = {}
    view["turn"] = state.turn_number
    view["step"] = state.step
    view["phase"] = state.step  # same as step today, see docstring
    view["active_player"] = state.active_player_idx
    view["priority_player"] = state.priority_player_idx
    view["is_over"] = state.is_over()
    view["is_draw"] = state.is_draw
    view["winner_id"] = state.winner_id
    view["life_p0"] = state.get_life(0)
    view["life_p1"] = state.get_life(1)
    view["hand_p0"] = tuple(_rust_zone_obj_sig(d) for d in state.hand_snapshot(0))
    view["hand_p1"] = tuple(_rust_zone_obj_sig(d) for d in state.hand_snapshot(1))
    view["library_p0"] = tuple(
        _rust_zone_obj_sig(d) for d in state.library_snapshot(0)
    )
    view["library_p1"] = tuple(
        _rust_zone_obj_sig(d) for d in state.library_snapshot(1)
    )
    view["graveyard_p0"] = tuple(
        _rust_zone_obj_sig(d) for d in state.graveyard_snapshot(0)
    )
    view["graveyard_p1"] = tuple(
        _rust_zone_obj_sig(d) for d in state.graveyard_snapshot(1)
    )
    view["exile_p0"] = tuple(_rust_zone_obj_sig(d) for d in state.exile_snapshot(0))
    view["exile_p1"] = tuple(_rust_zone_obj_sig(d) for d in state.exile_snapshot(1))
    view["mana_pool_p0"] = tuple(sorted(state.mana_pool_snapshot(0).items()))
    view["mana_pool_p1"] = tuple(sorted(state.mana_pool_snapshot(1).items()))
    view["battlefield"] = tuple(
        _rust_zone_obj_sig(d) for d in state.battlefield_snapshot()
    )
    view["stack"] = tuple(_rust_stack_obj_sig(d) for d in state.stack_snapshot())
    return view


# Mapping of inject-divergence field name -> a transform that returns a
# *different* but type-compatible value. Used by --inject-divergence to
# smoke-test that the diff machinery actually detects per-field changes.
# Each transform takes the current value and returns a guaranteed-different
# replacement.
def _mutate_field_value(field: str, value):
    """Return a value guaranteed to differ from ``value`` for ``field``.

    The mutation is type-aware so the post-mutation view still type-checks
    when compared element-by-element. For scalar int fields we add 1; for
    booleans we negate; for the step/phase string we tack on a suffix; for
    zone tuples we replace with the empty tuple (clearly different from
    any non-empty zone, and also distinct from an already-empty zone when
    we use a sentinel marker — see below).

    For zones that are already empty in the vanilla state (graveyard,
    exile, stack), we inject a fake sentinel tuple so the post-mutation
    value is meaningfully different from the empty baseline.
    """
    sentinel_obj = ("__INJECTED__", 1, 0, 0, 0, None, 0, -1)
    sentinel_stack = (None, 0, 0, 0, 0, None, 0, -1)
    if field in ("turn",):
        return (value or 0) + 1
    if field in ("active_player", "priority_player"):
        return 1 - (value or 0)  # toggle 0 <-> 1
    if field in ("life_p0", "life_p1"):
        return (value if value is not None else 20) - 1
    if field == "winner_id":
        return 0 if value is None else None
    if field in ("is_over", "is_draw"):
        return not value
    if field in ("step", "phase"):
        return f"{value}__INJECTED__"
    if field in ("mana_pool_p0", "mana_pool_p1"):
        # Replace with a clearly-different mana set; tuple of sorted items
        # so the comparison signature stays type-uniform.
        return tuple(sorted({"W": 99}.items()))
    if field in (
        "hand_p0", "hand_p1", "library_p0", "library_p1",
        "graveyard_p0", "graveyard_p1", "exile_p0", "exile_p1",
        "battlefield",
    ):
        # Append a sentinel so empty zones still get a non-trivially-
        # different post-injection signature.
        return tuple(value) + (sentinel_obj,)
    if field == "stack":
        return tuple(value) + (sentinel_stack,)
    raise ValueError(
        f"_mutate_field_value: unknown field {field!r} "
        f"(no mutation recipe)"
    )


def run_field_diff_mode(
    seed: int,
    num_battlefield: int = 10,
    num_hand: int = 7,
    num_library: int = 50,
    inject_divergence: str | None = None,
) -> int:
    """Build a Rust synthetic state and report a per-field diff between
    two parallel views of it.

    Today both views are walked from the same Rust state via the snapshot
    getters — one through ``_hash_rust_state``'s recipe (the
    "python-recipe" view) and one direct (the "rust-direct" view). On a
    vanilla state they're identical by construction; the value of this
    mode is twofold:

      1. ``--inject-divergence FIELD`` artificially mutates one field on
         the rust-direct view after building it, so we can verify the
         diff actually catches per-field mismatches. Smoke-tests the
         smoke before the actually-divergent bridge case exists.
      2. When the future cross-engine bridge lands (Python engine wired
         in), one side switches to a Python-walked view and any read-
         side drift surfaces as a localized "life=20/19 — DIVERGE" entry
         instead of "different digest, good luck".

    Returns 0 if every field matches, 1 if any field differs, 2 on
    import error.

    The ``seed`` argument is informational today (make_synthetic_state is
    deterministic), matching the contract on ``run_initial_state_compare``.
    """
    try:
        import mtg_bot_engine
    except ImportError as e:
        print(f"[field-diff] ERROR: mtg_bot_engine not built ({e})")
        print("  -> run `maturin develop --release -m rust/Cargo.toml`")
        return 2

    # Accept unqualified per-player field names (e.g. "life" instead of
    # "life_p0") as a convenience — defaults to player 0. Matches the
    # task brief's CLI surface: `--inject-divergence life` is shorter
    # and reads more naturally than `--inject-divergence life_p0`.
    if inject_divergence is not None and inject_divergence not in FIELD_DIFF_FIELDS:
        # Try suffixing _p0 for the per-player family fields.
        candidate = f"{inject_divergence}_p0"
        if candidate in FIELD_DIFF_FIELDS:
            inject_divergence = candidate
        else:
            print(
                f"[field-diff] ERROR: unknown --inject-divergence field "
                f"{inject_divergence!r}; must be one of: "
                f"{', '.join(FIELD_DIFF_FIELDS)}"
            )
            return 2

    state = mtg_bot_engine.make_synthetic_state(
        num_battlefield, num_hand, num_library
    )

    # Both views come from the same _build_field_view today (see docstring
    # — the bridge to a Python-walked view is the next slice). For the
    # inject-divergence path we mutate ONE field on the right-hand side
    # to prove the diff actually fires.
    left = _build_field_view(state)
    right = _build_field_view(state)

    if inject_divergence is not None:
        right[inject_divergence] = _mutate_field_value(
            inject_divergence, right[inject_divergence]
        )

    # Walk every field in canonical order. For each field, emit one row
    # of the table — OK if equal, DIFFER with the two values if not.
    print(
        f"[field-diff] synthetic state "
        f"[bf={num_battlefield}, hand={num_hand}, lib={num_library}, seed={seed}]"
        + (
            f" (--inject-divergence {inject_divergence})"
            if inject_divergence is not None
            else ""
        )
    )
    print()
    # Column widths chosen so a 20-char field name + a "DIFFER" verdict +
    # short value reprs fit on a typical terminal line; long zone tuples
    # are truncated in the rendering helper.
    print("field             | status | left                                  | right")
    print("------------------|--------|---------------------------------------|--------")

    n_differ = 0
    for field in FIELD_DIFF_FIELDS:
        l_val = left.get(field)
        r_val = right.get(field)
        if l_val == r_val:
            print(f"{field:<17} | OK     | {_field_repr(l_val):<37} | (same)")
        else:
            n_differ += 1
            print(
                f"{field:<17} | DIFFER | "
                f"{_field_repr(l_val):<37} | {_field_repr(r_val)}"
            )

    print()
    if n_differ == 0:
        print(
            f"PARITY: PASS — all {len(FIELD_DIFF_FIELDS)} fields match between views"
        )
        return 0

    print(
        f"PARITY: FAIL — {n_differ} of {len(FIELD_DIFF_FIELDS)} fields differ"
    )
    return 1


def _field_repr(value, max_len: int = 35) -> str:
    """Render a field value as a short, single-line string.

    Long zone tuples get truncated with ``...`` so the table reads
    cleanly. Truncation is purely cosmetic for the printed table — the
    actual equality comparison is done on the unrendered values.
    """
    rendered = repr(value)
    if len(rendered) > max_len:
        return rendered[: max_len - 3] + "..."
    return rendered


# ---------- known-gap masks ----------------------------------------------
#
# `docs/RUST_PARITY_GAPS.md` lists G1-G24 known divergences between
# Python and Rust. Several of those gaps fire reliably inside the
# cross-engine-game harness — most prominently G3 (`pass_priority`
# pure toggle), which trips at decision 0 on every seed and masks
# the more interesting *unknown* divergences that would surface
# further down the trajectory.
#
# ``--ignore-known-gaps`` tells the harness "I already know about
# these — skip them and walk further". The harness then prints
# ``MASKED (Gxx)`` for the divergence row and keeps walking; only
# the FIRST UNKNOWN divergence stops the walk.
#
# The mask schema below is intentionally simple: each entry is a
# gap_id -> dict mapping listing the fields the gap touches and the
# trigger context (action kind it was preceded by, step in which it
# applies, or simply "always" for context-free masks). A divergence
# is "known" if it lives on one of the gap's fields AND the trigger
# context matches. If a divergence on a masked field doesn't match
# the trigger context, the harness still flags it as a DIVERGE (it's
# a *new* signal on a documented-but-different surface).
#
# Coverage today: a representative subset (G3 / G14 / G15 / G18) —
# not every gap. The harness errs on the side of "stop loudly on
# anything unrecognized" so adding masks is an explicit decision.
# When a new gap surfaces in practice, document it in
# RUST_PARITY_GAPS.md first, then add an entry here.

# Module-level mapping of gap_id (matching the ``### Gn`` anchors in
# ``docs/RUST_PARITY_GAPS.md``) to mask metadata. Each value is a
# dict with:
#   - description: 1-line summary of the gap (mirrors the doc).
#   - fields: tuple of FIELD_DIFF_FIELDS names the gap can touch.
#   - trigger_action_kinds: optional tuple of action ``kind`` strings
#     that must have just been applied for the mask to apply. If
#     absent, the mask is action-context-free.
#   - trigger_steps: optional tuple of step-name strings ("UNTAP",
#     "DRAW", ...) the state must be in for the mask to apply. If
#     absent, the mask is step-context-free.
#
# A divergence matches a gap iff:
#   1. The diverging field is in ``gap["fields"]``, AND
#   2. (if ``trigger_action_kinds`` is present) the last action's
#      kind is in it, AND
#   3. (if ``trigger_steps`` is present) the current step is in it.
#
# The first matching gap wins; gaps are checked in dict-insertion
# order (which is gap-id order since the doc numbers them
# monotonically).
KNOWN_GAP_MASKS: dict[str, dict] = {
    # G3: pass_priority is a pure toggle in Rust — flips
    # `priority_player` without composing step-end / stack-resolution.
    # Python's pass_priority is a no-op on the field. See
    # docs/RUST_PARITY_GAPS.md#G3.
    "G3": {
        "description": (
            "pass_priority pure toggle — Rust flips priority_player, "
            "Python no-ops on the field"
        ),
        "fields": ("priority_player",),
        "trigger_action_kinds": ("pass_priority",),
    },
    # G14: advance_phase does NOT auto-fire draw_card on the DRAW
    # step in Rust. Python's turn machinery DOES draw. Visible as
    # an off-by-one on the active player's hand / library. See
    # docs/RUST_PARITY_GAPS.md#G14. Note: in the current harness
    # `advance_phase` is engine-internal and not directly emitted by
    # legal_decisions, so this mask is precautionary — it will fire
    # if a future Rust slice exposes step transitions to the action
    # dispatcher.
    "G14": {
        "description": (
            "no auto-draw on DRAW step — Rust hand/library stay "
            "frozen across the DRAW transition"
        ),
        "fields": ("hand_p0", "hand_p1", "library_p0", "library_p1"),
        "trigger_steps": ("DRAW",),
    },
    # G15: advance_phase's CLEANUP -> UNTAP wrap does NOT reset
    # priority to the new active player in Rust. Python's wrap does.
    # Distinct trigger context from G3 — fires on phase advance, not
    # on pass_priority. See docs/RUST_PARITY_GAPS.md#G15. Same
    # precautionary note as G14 re: advance_phase not being a
    # legal_decisions action today.
    "G15": {
        "description": (
            "no priority reset on CLEANUP->UNTAP wrap — Rust keeps "
            "the old priority_player after the turn boundary"
        ),
        "fields": ("priority_player",),
        "trigger_steps": ("UNTAP",),
    },
    # G18: ActivateAbility credits the SOURCE's controller, not
    # `active_player`. Documented as a parity invariant (I9) for the
    # behavior, BUT listed here because if a future change drifts
    # either side, the mana-pool delta will land on the "wrong"
    # player and surface as a mana_pool_pN divergence right after
    # the activate_ability action. See docs/RUST_PARITY_GAPS.md#G18
    # / #I9.
    "G18": {
        "description": (
            "ActivateAbility credits source.controller not "
            "active_player — mana lands on the source's controller"
        ),
        "fields": ("mana_pool_p0", "mana_pool_p1"),
        "trigger_action_kinds": ("activate_ability",),
    },
}


def _match_known_gap(
    field: str,
    last_action: dict | None,
    rust_view: dict,
    python_view: dict,
) -> str | None:
    """Return the gap_id whose mask matches the ``(field, action,
    step)`` divergence context, or ``None`` if no documented gap
    explains the divergence.

    Gaps are checked in ``KNOWN_GAP_MASKS`` insertion order; the
    first match wins. A gap's mask matches iff (a) the field is in
    its ``fields`` tuple, (b) if it specifies
    ``trigger_action_kinds``, the last action's ``kind`` is in that
    set, and (c) if it specifies ``trigger_steps``, the current
    step (read from the Rust view — Python view's step is canonicalized
    to the same string shape) is in that set.

    ``last_action`` may be ``None`` for the initial-state diff (no
    action has been applied yet); in that case action-kind-gated
    masks won't match.
    """
    last_kind = last_action.get("kind") if last_action else None
    # Both views encode the step as the same uppercase string, so
    # either works; prefer Rust's since the mask is about Rust's
    # documented behavior.
    current_step = rust_view.get("step")
    for gap_id, mask in KNOWN_GAP_MASKS.items():
        if field not in mask["fields"]:
            continue
        if "trigger_action_kinds" in mask:
            if last_kind not in mask["trigger_action_kinds"]:
                continue
        if "trigger_steps" in mask:
            if current_step not in mask["trigger_steps"]:
                continue
        return gap_id
    return None


def _diverging_fields(rust_view: dict, python_view: dict) -> list[str]:
    """Return every field in canonical ``FIELD_DIFF_FIELDS`` order
    whose value differs between the two views. Empty list when the
    views agree on every field.

    Sibling of ``_first_diverging_field``; used by the masking path
    which needs to inspect every divergent field, not just the
    first, so it can distinguish "the only divergence was a masked
    one" (continue walking) from "there's also an unmasked
    divergence" (stop).
    """
    out: list[str] = []
    for field_name in FIELD_DIFF_FIELDS:
        if rust_view.get(field_name) != python_view.get(field_name):
            out.append(field_name)
    return out


# ---------- cross-engine-game mode ---------------------------------------
#
# Phase-5 prep: drive Python AND Rust through the SAME deterministic
# decision sequence one step at a time, comparing per-step state via
# the field-diff machinery, and report the first divergence index.
#
# What this is for
# ----------------
#
# `docs/RUST_PARITY_GAPS.md` lists G1-G24 known divergences between
# Python and Rust. The smoke harness's job is NOT to make them match
# (that would require closing the gaps); it's to make a measurement
# loop that reports the first decision index where Python and Rust
# disagree per seed, so the operator can match the symptom against
# the gap table and either ignore (known) or escalate (new).
#
# Why Forest-only decks
# ---------------------
#
# The cross-engine bridge needs matching card data on both sides. The
# simplest deck shape that exercises non-trivial decision flow without
# requiring full card-text fidelity (Python's CardPrinting has ~50
# fields; the Rust bridge accepts a minimal dict) is "Forest x N".
# Lands exercise: pass_priority, play_land, advance_phase (turn step
# progression), and basic zone mutation. They DON'T exercise spell
# casting / stack resolution / combat — left to a richer-deck
# follow-up slice. Forest specifically because it's in the active
# Python card source AND maps cleanly to the Rust bridge's
# `{name, card_types=[Land], mana_cost={}, effects=[], mana_produced=[G]}`
# shape.
#
# Tolerance for asymmetry
# -----------------------
#
# Per G5: Rust `legal_decisions(non_priority)` returns `[pass_priority]`
# singleton. Per G3/G4: Rust `pass_priority`/`cast_creature` /
# `activate_ability` are pure toggles or no-ops on priority. Per G14:
# Rust's `advance_phase` does NOT auto-draw on DRAW step.
# Consequence: even on the first pass_priority, Rust will toggle
# `priority_player` while Python (which has no auto-step-progression
# inside `apply_action`) leaves it untouched. THAT is the expected
# first divergence per the gap doc — exactly the measurement signal
# we want, not a bug. The harness reports the divergence index and
# fields, doesn't try to mask it.


# Minimal printing dict accepted by the Rust bridge for a Forest.
# Mirrors the shape used in `tests_rust/test_setup.py::_forest()` — the
# bridge requires `name`, `card_types`, `mana_cost`, and `effects`;
# `mana_produced` is the optional mana-ability list.
_FOREST_RUST_DICT = {
    "name": "Forest",
    "card_types": ["Land"],
    "mana_cost": {},
    "effects": [],
    "mana_produced": ["G"],
}


def _build_python_state_for_cross_engine(seed: int, deck_size: int = 30):
    """Build a Python ``GameState`` with Forest-only decks for both seats.

    Calls ``simulate.setup_game`` which shuffles + deals 7 to each
    player (the production setup path). Forest is the only card so
    every hand and library entry is Forest, which makes the
    Python-vs-Rust deck-content mapping trivial: every card maps to
    the single Forest printing-id on the Rust side.

    ``deck_size`` defaults to 30 — large enough to draw 7 + a few
    extra without decking out in a short walk, small enough to keep
    the snapshot loops fast.
    """
    from mtg_bot.simulate import setup_game
    from mtg_bot.agents import RandomAgent

    decks = [["Forest"] * deck_size, ["Forest"] * deck_size]
    state = setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=decks,
        seed=seed,
    )
    return state


def _build_rust_state_for_cross_engine(python_state):
    """Build a Rust ``GameState`` whose zone contents byte-match
    ``python_state``'s. Walks Python's per-player library + hand in
    order and emits the matching Forest-printing-id list on the Rust
    side. ``shuffle_seed=None`` because the Python setup already
    permuted the library — we want byte-equal zone order, not
    independently-shuffled order.

    Only Forest is registered on the Rust side; if a non-Forest card
    sneaks into the Python state's zones (won't happen for the
    Forest-only deck shape but the assertion guards against future
    callers passing a richer deck), the function raises so the
    asymmetry surfaces immediately instead of a silent printing-id
    map miss.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    forest_pid = reg.register_printing(_FOREST_RUST_DICT)
    name_to_pid = {"Forest": forest_pid}

    def to_pids(zone):
        out = []
        for obj in zone:
            name = obj.printing.name
            if name not in name_to_pid:
                raise ValueError(
                    f"cross-engine bridge: unsupported card {name!r} in "
                    f"Python zone (only Forest is bridged in this slice)"
                )
            out.append(name_to_pid[name])
        return out

    p0_lib = to_pids(python_state.players[0].library)
    p0_hand = to_pids(python_state.players[0].hand)
    p1_lib = to_pids(python_state.players[1].library)
    p1_hand = to_pids(python_state.players[1].hand)

    rust_state = mtg_bot_engine.make_state(
        reg,
        p0_lib,
        p1_lib,
        p0_hand_printing_ids=p0_hand,
        p1_hand_printing_ids=p1_hand,
        shuffle_seed=None,
    )
    return rust_state


def _python_zone_obj_sig_for_view(obj, *, on_battlefield: bool = False) -> tuple:
    """Per-object 7-tuple for a Python ``GameObject``, canonicalized
    for the cross-engine compare. NOT byte-equal to the
    ``_rust_zone_obj_sig`` 8-tuple used in the field-diff mode —
    cross-engine intentionally drops the ``id`` slot, see below.

    Python and Rust allocate per-object integer IDs in different
    orders (Python: setup_game allocates p0 lib (0..N-1), p1 lib
    (N..2N-1), then shuffles per-player; Rust make_state: p0 lib
    (0..N-1), p0 hand (N..N+H-1), p1 lib, p1 hand). For a
    Forest-only deck where every card is interchangeable, the IDs
    are purely a labeling concern at the gameplay level — yet the
    raw 8-tuple difference would fire as an initial-state
    "DIVERGE: hand_p0" on every seed, masking the real gameplay
    divergences (priority shifts, life changes, ...) we want the
    measurement loop to surface. The cross-engine signature drops
    the ``id`` slot so the per-object key is identity-by-content:
    ``(printing_name, tapped, summoning_sick, damage, counters,
    attached_to, controller_id)``. The field-diff mode (which only
    compares Rust-to-Rust) keeps the ``id`` slot via
    ``_rust_zone_obj_sig``.

    Summoning-sick canonicalization: same rationale as the previous
    revision of this function — Rust defaults off-battlefield
    objects to False, Python defaults them to True; they agree at
    the gameplay level but differ byte-wise. ``on_battlefield``
    gates the read so the field still surfaces real
    summoning-sick-on-battlefield divergences.
    """
    sick = int(getattr(obj, "summoning_sick", False)) if on_battlefield else 0
    return (
        obj.printing.name,
        int(getattr(obj, "tapped", False)),
        sick,
        getattr(obj, "damage", 0),
        getattr(obj, "counters", 0),
        getattr(obj, "attached_to", None),
        getattr(obj, "controller_id", None),
    )


def _rust_zone_obj_sig_for_cross_engine(d: dict) -> tuple:
    """Cross-engine companion to ``_python_zone_obj_sig_for_view``:
    same 7-tuple shape (no ``id`` slot), reading from the dict-form
    Rust snapshot. See ``_python_zone_obj_sig_for_view``'s docstring
    for the rationale on dropping ``id`` from the cross-engine
    compare. The field-diff mode uses ``_rust_zone_obj_sig`` (with
    the id slot) since it only compares Rust-to-Rust.
    """
    return (
        d["printing_name"],
        int(d["tapped"]),
        int(d["summoning_sick"]),
        d["damage"],
        d["counters"],
        None,  # attached_to — not modeled in Rust
        d["controller_id"],
    )


def _rust_stack_obj_sig_for_cross_engine(d: dict) -> tuple:
    """Cross-engine companion to ``_python_stack_obj_sig_for_view``.
    Same shape as the field-diff version but with the id slot
    dropped — stack items reference a source-obj id that wouldn't
    match across engines for the same reason zone-object ids
    don't (see ``_python_zone_obj_sig_for_view`` docstring).
    """
    return (None, 0, 0, 0, 0, None, d["controller"])


def _python_stack_obj_sig_for_view(item) -> tuple:
    """Per-stack-item 7-tuple for a Python ``StackItem``, paired
    with ``_rust_stack_obj_sig_for_cross_engine``. The ``id`` slot
    that the field-diff version of this function carried is dropped
    for the cross-engine compare for the same per-engine ID-mismatch
    reason zone-object ids are (see
    ``_python_zone_obj_sig_for_view`` docstring).
    """
    return (
        None, 0, 0, 0, 0, None,
        getattr(item, "controller_id", None),
    )


def _build_python_field_view(state) -> dict:
    """Build a field-keyed view of a Python ``GameState`` in the same
    shape ``_build_field_view`` produces for the Rust side.

    The two views feed into the SAME ``FIELD_DIFF_FIELDS`` diff so the
    operator can answer "which field diverged" by name (e.g.
    "priority_player_idx: rs=1 / py=0 — the G3 pure-toggle gap").

    Step normalization: Python's ``state.step`` is a ``Step`` enum
    instance; Rust's ``state.step`` is the uppercase enum name as a
    string. The view canonicalizes to the string-name shape (matches
    Rust) so the field-diff doesn't constantly trip on the
    representation gap. ``state.step is None`` (a fresh state mid-
    setup) renders as ``None`` on both sides.
    """
    step_value = None
    raw_step = getattr(state, "step", None)
    if raw_step is not None:
        # Step enum has .name; Rust returns the same uppercase string.
        step_value = getattr(raw_step, "name", str(raw_step))

    view: dict = {}
    view["turn"] = state.turn_number
    view["step"] = step_value
    view["phase"] = step_value
    view["active_player"] = state.active_player_idx
    view["priority_player"] = state.priority_player_idx
    view["is_over"] = state.is_over()
    view["is_draw"] = getattr(state, "is_draw", False)
    view["winner_id"] = getattr(state, "winner_id", None)
    view["life_p0"] = state.players[0].life
    view["life_p1"] = state.players[1].life
    view["hand_p0"] = tuple(
        _python_zone_obj_sig_for_view(o) for o in state.players[0].hand
    )
    view["hand_p1"] = tuple(
        _python_zone_obj_sig_for_view(o) for o in state.players[1].hand
    )
    view["library_p0"] = tuple(
        _python_zone_obj_sig_for_view(o) for o in state.players[0].library
    )
    view["library_p1"] = tuple(
        _python_zone_obj_sig_for_view(o) for o in state.players[1].library
    )
    view["graveyard_p0"] = tuple(
        _python_zone_obj_sig_for_view(o) for o in state.players[0].graveyard
    )
    view["graveyard_p1"] = tuple(
        _python_zone_obj_sig_for_view(o) for o in state.players[1].graveyard
    )
    view["exile_p0"] = tuple(
        _python_zone_obj_sig_for_view(o)
        for o in getattr(state.players[0], "exile", [])
    )
    view["exile_p1"] = tuple(
        _python_zone_obj_sig_for_view(o)
        for o in getattr(state.players[1], "exile", [])
    )
    # Python mana pools are a flat ``list[ManaCoin]`` where each coin
    # has a ``.color`` attribute; collapse to a sorted
    # ``[(color, count), ...]`` tuple matching Rust's snapshot shape.
    # Rust's ``mana_pool_snapshot(p)`` returns a 6-key dict (one per
    # color: B, C, G, R, U, W) including zero entries — we emit the
    # same 6 keys here so an empty Python pool matches an empty Rust
    # pool byte-for-byte instead of producing ``()`` vs all-zeros.
    def _py_mana_signature(pool):
        counts = {c: 0 for c in ("B", "C", "G", "R", "U", "W")}
        for coin in pool or ():
            color = getattr(coin, "color", None)
            if color in counts:
                counts[color] += 1
            elif color == "any":
                # ``any`` coins (per CR 106.6, "any color" mana) don't
                # have a fixed bucket — they materialize their color
                # when spent. Rust doesn't model these yet; fold into
                # the colorless bucket here so the cross-engine diff
                # surfaces a "Python has C+1 while Rust has C+0" entry
                # rather than crashing on an unknown key.
                counts["C"] += 1
        return tuple(sorted(counts.items()))

    view["mana_pool_p0"] = _py_mana_signature(state.players[0].mana_pool)
    view["mana_pool_p1"] = _py_mana_signature(state.players[1].mana_pool)
    view["battlefield"] = tuple(
        _python_zone_obj_sig_for_view(o, on_battlefield=True)
        for o in getattr(state, "battlefield", [])
    )
    view["stack"] = tuple(
        _python_stack_obj_sig_for_view(it)
        for it in getattr(state, "stack", [])
    )
    return view


def _build_rust_cross_engine_view(state) -> dict:
    """Build the cross-engine field-keyed view of a Rust GameState.

    Same shape as ``_build_python_field_view`` produces for the
    Python side. NOT the same as ``_build_field_view`` (which is
    the field-diff mode's Rust-to-Rust view) — cross-engine drops
    the ``id`` slot from per-object signatures to avoid the
    Python/Rust ID-allocation order mismatch swamping the diff;
    see ``_python_zone_obj_sig_for_view`` docstring for the full
    rationale.
    """
    view: dict = {}
    view["turn"] = state.turn_number
    view["step"] = state.step
    view["phase"] = state.step
    view["active_player"] = state.active_player_idx
    view["priority_player"] = state.priority_player_idx
    view["is_over"] = state.is_over()
    view["is_draw"] = state.is_draw
    view["winner_id"] = state.winner_id
    view["life_p0"] = state.get_life(0)
    view["life_p1"] = state.get_life(1)
    view["hand_p0"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.hand_snapshot(0)
    )
    view["hand_p1"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.hand_snapshot(1)
    )
    view["library_p0"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.library_snapshot(0)
    )
    view["library_p1"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.library_snapshot(1)
    )
    view["graveyard_p0"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.graveyard_snapshot(0)
    )
    view["graveyard_p1"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.graveyard_snapshot(1)
    )
    view["exile_p0"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.exile_snapshot(0)
    )
    view["exile_p1"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.exile_snapshot(1)
    )
    view["mana_pool_p0"] = tuple(sorted(state.mana_pool_snapshot(0).items()))
    view["mana_pool_p1"] = tuple(sorted(state.mana_pool_snapshot(1).items()))
    view["battlefield"] = tuple(
        _rust_zone_obj_sig_for_cross_engine(d) for d in state.battlefield_snapshot()
    )
    view["stack"] = tuple(
        _rust_stack_obj_sig_for_cross_engine(d) for d in state.stack_snapshot()
    )
    return view


def _translate_rust_card_id(
    rust_card_id: int,
    rust_state,
    python_state,
) -> int | None:
    """Find a Python ``GameObject.id`` whose zone-position + zone +
    controller match Rust's ``rust_card_id``. Returns ``None`` if
    Rust's id isn't found in any zone we know how to scan.

    Necessary because Python and Rust allocate per-object IDs in
    different orders (see ``_python_zone_obj_sig_for_view`` docstring).
    The cross-engine field-view canonicalizes by dropping the ``id``
    slot from per-object signatures, but action dicts emitted by
    ``legal_decisions`` still cite a Rust ``card_id`` that the Python
    engine doesn't know about. This translator bridges by zone-
    position:

    1. Find ``rust_card_id`` in Rust's per-zone snapshots; record
       (zone, controller, position).
    2. Pull the card at the same (zone, controller, position) in
       Python; return its Python id.
    3. Return ``None`` if Rust's id isn't on any known zone — the
       caller treats that as a non-translatable action and reports
       it as a STOP row.

    Robust to Forest-only decks (the only deck shape this slice
    supports): every card is interchangeable, so position-matching
    is the same as identity-matching at the gameplay level. For a
    multi-card deck the position-match would still be the right
    correspondence as long as both engines mutate zones in the same
    order — which is itself a parity invariant the smoke would
    catch.
    """
    # Per-zone scan order: hand, battlefield, graveyard, exile,
    # library, stack. Stack is last because stack items have a
    # different shape and we punt on them — card_id translation
    # only makes sense for zone permanents in this slice.
    for player_id in (0, 1):
        for zone_name, py_zone, rs_snapshot in (
            (
                "hand",
                python_state.players[player_id].hand,
                rust_state.hand_snapshot(player_id),
            ),
            (
                "library",
                python_state.players[player_id].library,
                rust_state.library_snapshot(player_id),
            ),
            (
                "graveyard",
                python_state.players[player_id].graveyard,
                rust_state.graveyard_snapshot(player_id),
            ),
            (
                "exile",
                getattr(python_state.players[player_id], "exile", []),
                rust_state.exile_snapshot(player_id),
            ),
        ):
            for pos, d in enumerate(rs_snapshot):
                if d["id"] == rust_card_id:
                    if pos < len(py_zone):
                        return py_zone[pos].id
                    return None
        # Battlefield is shared across players in Python but split per
        # player in Rust; we walk Rust's per-player battlefield and
        # match against Python's centralized battlefield filtered by
        # controller.
        py_bf = [
            o
            for o in getattr(python_state, "battlefield", [])
            if getattr(o, "controller_id", None) == player_id
        ]
        rs_bf = [
            d
            for d in rust_state.battlefield_snapshot()
            if d["controller_id"] == player_id
        ]
        for pos, d in enumerate(rs_bf):
            if d["id"] == rust_card_id:
                if pos < len(py_bf):
                    return py_bf[pos].id
                return None
    return None


def _translate_rust_action_for_python(
    rust_action: dict,
    rust_state,
    python_state,
) -> dict:
    """Return a new action dict whose ``card_id`` / ``source_id`` /
    ``attacker_ids`` / ``assignments`` fields point at Python IDs
    instead of Rust IDs. The kind + non-id keys are preserved as-is.

    Translation failures (e.g. a Rust id that doesn't map to any
    Python card) yield a translated value of ``None`` so the caller
    can detect the asymmetry and report it as a STOP row instead of
    silently dropping the action. Wrapped in a separate dict (not a
    mutation of the input) so the original Rust action dict can
    still be used for ``rs_state.apply_action(rust_action)``.
    """
    out = dict(rust_action)
    if "card_id" in out:
        out["card_id"] = _translate_rust_card_id(
            out["card_id"], rust_state, python_state
        )
    if "source_id" in out:
        out["source_id"] = _translate_rust_card_id(
            out["source_id"], rust_state, python_state
        )
    if "attacker_ids" in out:
        out["attacker_ids"] = [
            _translate_rust_card_id(a, rust_state, python_state)
            for a in out["attacker_ids"]
        ]
    if "assignments" in out:
        out["assignments"] = [
            (
                _translate_rust_card_id(b, rust_state, python_state),
                _translate_rust_card_id(a, rust_state, python_state),
            )
            for (b, a) in out["assignments"]
        ]
    return out


def _rust_dict_to_python_action(action_dict: dict):
    """Translate a Rust-emitted action dict into a Python ``Action``.

    Mirrors the variant shapes in ``rust/src/state_ops_legal.rs``'s
    per-kind dict builders (`pass_priority_dict`, `play_land_dict`,
    `cast_spell_dict`, ...). Unknown kinds raise ``NotImplementedError``
    with the dict's full shape so a future Rust-side new kind surfaces
    loudly in the harness instead of silently being dropped.

    The harness's "if Python can't apply this, record asymmetry" path
    catches the ``NotImplementedError`` upstream and reports it as a
    divergence reason — we deliberately fail-loud here so the smoke
    table cites the exact unknown kind.
    """
    from mtg_bot.actions import (
        ActivateAbility,
        CastSpell,
        DeclareAttackers,
        DeclareBlockers,
        PassPriority,
        PlayLand,
        TurnFaceUp,
        UnlockRoomHalf,
    )

    kind = action_dict.get("kind")
    if kind == "pass_priority":
        return PassPriority()
    if kind == "play_land":
        cid = action_dict["card_id"]
        if cid is None:
            raise NotImplementedError(
                "play_land card_id failed Rust->Python translation"
            )
        return PlayLand(card_id=cid)
    if kind == "cast_spell":
        return CastSpell(
            card_id=action_dict["card_id"],
            use_warp=action_dict.get("use_warp", False),
            kicked=action_dict.get("kicked", False),
            cast_for_free=action_dict.get("cast_for_free", False),
            use_impending=action_dict.get("use_impending", False),
            cast_room_half=action_dict.get("cast_room_half"),
        )
    if kind == "activate_ability":
        return ActivateAbility(
            source_id=action_dict["source_id"],
            ability_index=action_dict["ability_index"],
        )
    if kind == "declare_attackers":
        return DeclareAttackers(
            attacker_ids=frozenset(action_dict.get("attacker_ids", []))
        )
    if kind == "declare_blockers":
        return DeclareBlockers(
            assignments=tuple(
                (b, a) for (b, a) in action_dict.get("assignments", [])
            )
        )
    if kind == "turn_face_up":
        return TurnFaceUp(card_id=action_dict["card_id"])
    if kind == "unlock_room_half":
        return UnlockRoomHalf(
            card_id=action_dict["card_id"],
            half_name=action_dict["half_name"],
        )
    raise NotImplementedError(
        f"_rust_dict_to_python_action: unknown kind {kind!r} "
        f"(full dict: {dict(action_dict)!r})"
    )


def _rust_action_short_label(action_dict: dict) -> str:
    """Compact label for a Rust action dict — used in the per-decision
    table. Mirrors ``_decision_summary``'s shape but reads directly
    from the dict (which is what the cross-engine mode walks).
    """
    kind = action_dict.get("kind", "?")
    if "card_id" in action_dict:
        return f"{kind}(card_id={action_dict['card_id']})"
    if "attacker_ids" in action_dict:
        ids = action_dict["attacker_ids"]
        return f"{kind}(n={len(ids)})"
    if "assignments" in action_dict:
        return f"{kind}(n={len(action_dict['assignments'])})"
    if "source_id" in action_dict:
        return f"{kind}(source_id={action_dict['source_id']})"
    return kind


def _python_action_short_label(action) -> str:
    """Compact label for a Python ``Action`` — paired with
    ``_rust_action_short_label`` in the per-decision table. ``None``
    when Python couldn't translate the Rust action (handled upstream).
    """
    if action is None:
        return "<not-translated>"
    name = type(action).__name__
    if hasattr(action, "card_id"):
        return f"{name}(card_id={action.card_id})"
    if hasattr(action, "attacker_ids"):
        return f"{name}(n={len(action.attacker_ids)})"
    if hasattr(action, "assignments"):
        return f"{name}(n={len(action.assignments)})"
    if hasattr(action, "source_id"):
        return f"{name}(source_id={action.source_id})"
    return name


@dataclass
class CrossEngineStep:
    """One row of the cross-engine-game decision table.

    Fields:
      index           — decision index (0-based).
      rust_label      — short label of the Rust action chosen.
      python_label    — short label of the Python action applied
                        (or `<not-translated>` if translation failed,
                        or `<apply-failed>` if Python raised).
      verdict         — "OK", "DIVERGE: <field>", "STOP: <reason>".
      diverging_field — the field name from ``FIELD_DIFF_FIELDS`` that
                        diverged at this step, or ``None``.
      rust_value      — short repr of the Rust value at the diverging
                        field, or ``None``.
      python_value    — short repr of the Python value at the
                        diverging field, or ``None``.
    """

    index: int
    rust_label: str
    python_label: str
    verdict: str
    diverging_field: str | None = None
    rust_value: str | None = None
    python_value: str | None = None


def _first_diverging_field(rust_view: dict, python_view: dict) -> str | None:
    """Return the first field name in canonical ``FIELD_DIFF_FIELDS``
    order whose value differs between the two views, or ``None`` if
    they agree on every field.

    Single-source-of-truth for "did the two states diverge", reused
    by initial-state check + every per-step check inside the loop.
    """
    for field_name in FIELD_DIFF_FIELDS:
        if rust_view.get(field_name) != python_view.get(field_name):
            return field_name
    return None


def _print_cross_engine_header(seed: int, num_decisions: int, deck_size: int) -> None:
    """Print the header line + table column headers for the per-step
    table. Factored out so tests can match exact column names without
    being coupled to the data-row format.
    """
    print(
        f"[cross-engine-game] seed={seed}, num_decisions={num_decisions}, "
        f"deck=Forest x {deck_size} per seat"
    )
    print()
    print(
        "step | rust_action                              | "
        "python_action                            | verdict"
    )
    print(
        "-----|------------------------------------------|"
        "------------------------------------------|--------"
    )


def _print_cross_engine_row(row: CrossEngineStep) -> None:
    """Render one ``CrossEngineStep`` as a table row matching the
    header layout in ``_print_cross_engine_header``."""
    print(
        f"{row.index:>4} | {row.rust_label:<40} | "
        f"{row.python_label:<40} | {row.verdict}"
    )


def run_cross_engine_game_mode(
    seed: int,
    num_decisions: int = 20,
    deck_size: int = 30,
    ignore_known_gaps: bool = False,
) -> int:
    """Phase-5 prep: drive Python AND Rust through the SAME deterministic
    decision sequence one step at a time and report the first
    divergence index per the field-diff machinery.

    Returns 0 if the trajectory completes ``num_decisions`` steps (or
    hits a natural stop — game-over / both stuck) WITHOUT any field
    diverging. Returns 1 the first time the field-diff catches a
    mismatch OR Python can't apply Rust's chosen action. Returns 2 on
    import / setup error.

    The harness deliberately reports divergence rather than masking
    it: see the module-level docstring on ``cross-engine-game mode``
    for the "this is a measurement loop, gaps are the signal" rationale.

    When ``ignore_known_gaps`` is True, the harness consults
    ``KNOWN_GAP_MASKS`` after each per-step diff. Divergences that
    match a documented gap render as ``MASKED (Gxx)`` and the walk
    continues; the walk only stops on a divergence that DOES NOT
    match any known gap — the more interesting "first unknown
    divergence" signal. The masking knob is intended for "I already
    know about G3, show me what's behind it" investigations; the
    default-False behavior preserves the original "stop at first
    divergence" measurement contract.
    """
    try:
        import mtg_bot_engine  # noqa: F401  (used by helpers below)
    except ImportError as e:
        print(f"[cross-engine-game] ERROR: mtg_bot_engine not built ({e})")
        print("  -> run `maturin develop --release -m rust/Cargo.toml`")
        return 2

    try:
        py_state = _build_python_state_for_cross_engine(seed, deck_size=deck_size)
        rs_state = _build_rust_state_for_cross_engine(py_state)
    except Exception as e:
        print(
            f"[cross-engine-game] ERROR: state setup raised "
            f"{type(e).__name__}: {e}"
        )
        return 2

    _print_cross_engine_header(seed, num_decisions, deck_size)

    # Initial-state field-diff — if this trips, the cross-engine setup
    # itself drifted (zone-mirror bug). Report it as the "step -1" row
    # and stop; it's a setup-side bug, not a per-decision divergence.
    rust_view = _build_rust_cross_engine_view(rs_state)
    python_view = _build_python_field_view(py_state)
    initial_field = _first_diverging_field(rust_view, python_view)
    if initial_field is not None:
        row = CrossEngineStep(
            index=-1,
            rust_label="<initial-state>",
            python_label="<initial-state>",
            verdict=f"DIVERGE: {initial_field}",
            diverging_field=initial_field,
            rust_value=_field_repr(rust_view.get(initial_field)),
            python_value=_field_repr(python_view.get(initial_field)),
        )
        _print_cross_engine_row(row)
        print()
        print(
            f"PARITY: FAIL — initial-state divergence on field "
            f"{initial_field!r}"
        )
        print(
            f"  rust  = {_field_repr(rust_view.get(initial_field))}"
        )
        print(
            f"  python= {_field_repr(python_view.get(initial_field))}"
        )
        return 1

    # Decision loop. Single RNG seeded once so the choice sequence is
    # deterministic across runs with the same `seed`. We pick from
    # Rust's `legal_decisions(priority_player)` (not Python's) per the
    # task brief — Rust's emission set is the authoritative one we
    # measure against; G5 documents that off-priority callers get the
    # `[pass_priority]` singleton, so this is always non-empty.
    from mtg_bot.actions import apply_action as py_apply_action

    rng = random.Random(seed)
    rows: list[CrossEngineStep] = []

    for step in range(num_decisions):
        priority_player = rs_state.get_priority_player()
        legal = rs_state.legal_decisions(priority_player)
        if not legal:
            row = CrossEngineStep(
                index=step,
                rust_label="<no-legal>",
                python_label="<no-legal>",
                verdict="STOP: rust legal_decisions is empty",
            )
            rows.append(row)
            _print_cross_engine_row(row)
            break

        rust_action = dict(rng.choice(legal))
        rust_label = _rust_action_short_label(rust_action)

        # Translate Rust ids to Python ids before building the
        # Python Action (Rust + Python allocate per-object ids in
        # different orders; see ``_translate_rust_card_id`` docstring
        # for the position-matching strategy).
        translated_dict = _translate_rust_action_for_python(
            rust_action, rs_state, py_state
        )

        # Build Python Action from the translated dict.
        try:
            py_action = _rust_dict_to_python_action(translated_dict)
        except NotImplementedError as e:
            row = CrossEngineStep(
                index=step,
                rust_label=rust_label,
                python_label="<not-translated>",
                verdict=f"STOP: untranslatable Rust action ({type(e).__name__})",
            )
            rows.append(row)
            _print_cross_engine_row(row)
            break

        python_label = _python_action_short_label(py_action)

        # Apply to Python first (best-effort). If Python raises, that's
        # an asymmetry: record + stop without crashing.
        python_applied = True
        python_error: Exception | None = None
        try:
            py_apply_action(py_state, priority_player, py_action)
        except Exception as e:
            python_applied = False
            python_error = e

        if not python_applied:
            row = CrossEngineStep(
                index=step,
                rust_label=rust_label,
                python_label=python_label,
                verdict=(
                    f"STOP: python apply raised "
                    f"{type(python_error).__name__}: {python_error}"
                )[:200],
            )
            rows.append(row)
            _print_cross_engine_row(row)
            break

        # Apply to Rust. We've translated successfully so the action
        # shape is valid; an exception here is a bug in apply_action
        # itself, not a parity gap — re-raise to surface it.
        try:
            rs_state.apply_action(rust_action)
        except Exception as e:
            row = CrossEngineStep(
                index=step,
                rust_label=rust_label,
                python_label=python_label,
                verdict=(
                    f"STOP: rust apply raised "
                    f"{type(e).__name__}: {e}"
                )[:200],
            )
            rows.append(row)
            _print_cross_engine_row(row)
            break

        # Field-diff after both applies succeeded. Use the
        # cross-engine view (7-tuple, no id slot) on both sides —
        # using ``_build_field_view`` here (8-tuple, with id) was a
        # pre-existing bug that the original "stop at G3 priority_player
        # before we even compare zones" measurement contract hid; once
        # ``--ignore-known-gaps`` masks the priority_player divergence
        # past, the zone fields become the next divergence and surface
        # the shape mismatch as a false-positive UNKNOWN. Matches the
        # initial-state check above which already used the cross-engine
        # view.
        rust_view = _build_rust_cross_engine_view(rs_state)
        python_view = _build_python_field_view(py_state)

        if not ignore_known_gaps:
            # Fast-path: original behavior — stop on first
            # divergence of any kind.
            diff_field = _first_diverging_field(rust_view, python_view)

            if diff_field is None:
                row = CrossEngineStep(
                    index=step,
                    rust_label=rust_label,
                    python_label=python_label,
                    verdict="OK",
                )
                rows.append(row)
                _print_cross_engine_row(row)
                continue

            row = CrossEngineStep(
                index=step,
                rust_label=rust_label,
                python_label=python_label,
                verdict=f"DIVERGE: {diff_field}",
                diverging_field=diff_field,
                rust_value=_field_repr(rust_view.get(diff_field)),
                python_value=_field_repr(python_view.get(diff_field)),
            )
            rows.append(row)
            _print_cross_engine_row(row)
            print()
            print(
                f"PARITY: FAIL — first divergence at decision {step} "
                f"on field {diff_field!r}"
            )
            print(
                f"  rust  = {_field_repr(rust_view.get(diff_field))}"
            )
            print(
                f"  python= {_field_repr(python_view.get(diff_field))}"
            )
            return 1

        # Masking path: walk EVERY divergent field, classify each
        # against KNOWN_GAP_MASKS. If all divergences are masked,
        # report MASKED (Gxx) and continue. Otherwise the first
        # UNMASKED divergence stops the walk (the interesting
        # signal).
        diverging = _diverging_fields(rust_view, python_view)
        if not diverging:
            row = CrossEngineStep(
                index=step,
                rust_label=rust_label,
                python_label=python_label,
                verdict="OK",
            )
            rows.append(row)
            _print_cross_engine_row(row)
            continue

        # Classify every diverging field.
        first_unmasked: str | None = None
        masked_gap_ids: list[str] = []
        for field_name in diverging:
            gap_id = _match_known_gap(
                field_name, rust_action, rust_view, python_view
            )
            if gap_id is None:
                first_unmasked = field_name
                break
            if gap_id not in masked_gap_ids:
                masked_gap_ids.append(gap_id)

        if first_unmasked is None:
            # All divergences were masked. Cite the gap(s) and walk on.
            gap_label = ",".join(masked_gap_ids)
            row = CrossEngineStep(
                index=step,
                rust_label=rust_label,
                python_label=python_label,
                verdict=f"MASKED ({gap_label})",
            )
            rows.append(row)
            _print_cross_engine_row(row)
            continue

        # First unmasked field — stop and surface as a real
        # divergence. This is the "the FIRST UNKNOWN divergence"
        # signal the masking flag exists to surface.
        row = CrossEngineStep(
            index=step,
            rust_label=rust_label,
            python_label=python_label,
            verdict=f"DIVERGE: {first_unmasked}",
            diverging_field=first_unmasked,
            rust_value=_field_repr(rust_view.get(first_unmasked)),
            python_value=_field_repr(python_view.get(first_unmasked)),
        )
        rows.append(row)
        _print_cross_engine_row(row)
        print()
        print(
            f"PARITY: FAIL — first UNMASKED divergence at decision "
            f"{step} on field {first_unmasked!r}"
        )
        print(
            f"  rust  = {_field_repr(rust_view.get(first_unmasked))}"
        )
        print(
            f"  python= {_field_repr(python_view.get(first_unmasked))}"
        )
        return 1

    # Loop completed without divergence (or stopped on STOP: rows).
    # Distinguish the OK-finish case (return 0) from the STOP case
    # (return 1 — Python couldn't follow Rust's emission, which is
    # itself an asymmetry signal).
    if rows and rows[-1].verdict.startswith("STOP"):
        print()
        print(
            f"PARITY: FAIL — Python couldn't follow Rust at decision "
            f"{rows[-1].index} ({rows[-1].verdict})"
        )
        return 1

    print()
    print(
        f"PARITY: PASS — {len(rows)} decisions applied without "
        f"field-level divergence (seed={seed})"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--colors", default="G,R", help="Comma-separated decklist colors (default G,R)")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument(
        "--engine",
        choices=("python", "rust", "both"),
        default="both",
        help="Which engine(s) to run (game mode only). 'both' runs both and compares.",
    )
    p.add_argument(
        "--mode",
        choices=(
            "game",
            "initial-state",
            "trajectory",
            "multi-seed",
            "field-diff",
            "cross-engine-game",
        ),
        default="game",
        help=(
            "Which parity check to run. 'game' (default) plays a full game on "
            "each engine and compares trajectories. 'initial-state' builds a "
            "Rust synthetic state and asserts its native fingerprint() matches "
            "a Python reimplementation of the hash recipe. 'trajectory' walks "
            "a Rust state through N decisions and asserts clone-replay "
            "determinism (Rust-only — cross-engine compare deferred). "
            "'multi-seed' runs 'trajectory' across K seeds and prints an "
            "aggregate summary. 'field-diff' builds a synthetic state, walks "
            "two parallel views of it, and prints a per-field OK/DIFFER table "
            "— the parity-debug primitive used to answer 'WHICH field "
            "diverged' once a trajectory smoke says digests don't match. "
            "'cross-engine-game' (Phase 5 prep) drives Python AND Rust in "
            "lockstep through the SAME deterministic decision sequence and "
            "reports the first decision index where they diverge per field; "
            "the smoke is expected to trip on the known G1-G24 gaps in "
            "docs/RUST_PARITY_GAPS.md and is the measurement loop that "
            "Phase 6 (1000-game smoke) will build on."
        ),
    )
    p.add_argument(
        "--inject-divergence",
        type=str,
        default=None,
        help=(
            "(field-diff) artificially mutate one field on the right-hand "
            "view post-snapshot so the diff machinery has something to "
            "detect. Smoke-test the smoke. Valid field names: "
            + ", ".join(FIELD_DIFF_FIELDS) + "."
        ),
    )
    p.add_argument(
        "--num-decisions",
        type=int,
        default=20,
        help="(trajectory / multi-seed) number of decisions to apply per walk.",
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="(multi-seed) number of seeds to run the trajectory walk on.",
    )
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help=(
            "(multi-seed) starting seed; sweep iterates "
            "[seed_start, seed_start + num_seeds)."
        ),
    )
    p.add_argument(
        "--num-battlefield",
        type=int,
        default=10,
        help=(
            "(initial-state / trajectory) per-player battlefield size for "
            "make_synthetic_state. Trajectory mode's default differs from "
            "initial-state (smaller state -> faster walk); pass explicitly "
            "when you want a specific size."
        ),
    )
    p.add_argument(
        "--num-hand",
        type=int,
        default=7,
        help="(initial-state / trajectory) per-player hand size for make_synthetic_state.",
    )
    p.add_argument(
        "--num-library",
        type=int,
        default=50,
        help="(initial-state / trajectory) per-player library size for make_synthetic_state.",
    )
    p.add_argument(
        "--deck-size",
        type=int,
        default=30,
        help=(
            "(cross-engine-game) per-seat Forest deck size. Large enough to "
            "draw 7 + walk N decisions without decking out; small enough to "
            "keep the snapshot loops fast."
        ),
    )
    p.add_argument(
        "--ignore-known-gaps",
        action="store_true",
        help=(
            "(cross-engine-game) when set, divergences that match a "
            "documented gap in docs/RUST_PARITY_GAPS.md (G1-G24, see "
            "KNOWN_GAP_MASKS) render as 'MASKED (Gxx)' instead of "
            "stopping the walk. The walk only stops on the FIRST "
            "UNKNOWN divergence — the more interesting 'what's behind "
            "G3?' signal. Without this flag the harness stops at the "
            "first divergence of any kind (the original measurement "
            "contract)."
        ),
    )
    args = p.parse_args()

    if args.mode == "initial-state":
        return run_initial_state_compare(
            seed=args.seed,
            num_battlefield=args.num_battlefield,
            num_hand=args.num_hand,
            num_library=args.num_library,
        )

    if args.mode == "trajectory":
        return run_trajectory_mode(
            seed=args.seed,
            num_decisions=args.num_decisions,
            num_battlefield=args.num_battlefield,
            num_hand=args.num_hand,
            num_library=args.num_library,
        )

    if args.mode == "field-diff":
        return run_field_diff_mode(
            seed=args.seed,
            num_battlefield=args.num_battlefield,
            num_hand=args.num_hand,
            num_library=args.num_library,
            inject_divergence=args.inject_divergence,
        )

    if args.mode == "multi-seed":
        return run_multi_seed_mode(
            num_seeds=args.num_seeds,
            seed_start=args.seed_start,
            num_decisions=args.num_decisions,
            num_battlefield=args.num_battlefield,
            num_hand=args.num_hand,
            num_library=args.num_library,
        )

    if args.mode == "cross-engine-game":
        return run_cross_engine_game_mode(
            seed=args.seed,
            num_decisions=args.num_decisions,
            deck_size=args.deck_size,
            ignore_known_gaps=args.ignore_known_gaps,
        )

    colors = args.colors.split(",")

    py_res = None
    rs_res = None
    if args.engine in ("python", "both"):
        py_res = run_python_engine(args.seed, colors, args.max_turns)
        print_result(py_res)
    if args.engine in ("rust", "both"):
        rs_res = run_rust_engine(args.seed, colors, args.max_turns)
        print_result(rs_res)

    if args.engine != "both":
        return 0 if (py_res or rs_res).error is None else 2

    div = find_first_divergence(py_res, rs_res)
    if div is None:
        print(f"PARITY: PASS — {py_res.decisions} decisions match")
        return 0

    print(f"PARITY: FAIL — first divergence at: {div}")
    if isinstance(div, int):
        lo = max(0, div - 2)
        hi = min(min(len(py_res.log), len(rs_res.log)), div + 3)
        print(f"  context [{lo}:{hi}]:")
        for i in range(lo, hi):
            marker = " <- " if i == div else "   "
            py = py_res.log[i] if i < len(py_res.log) else None
            rs = rs_res.log[i] if i < len(rs_res.log) else None
            print(f"  {marker}#{i}")
            print(f"      py: {py}")
            print(f"      rs: {rs}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
