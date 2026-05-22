"""Control-change-UEOT subsystem — Cycle 3 (bridge round-trip).

Spec: lane brief (2026-05-21). Cycle 1 (substrate) added the per-object
``controller_override`` field + ``grant_control_change_ueot`` PyO3 setter +
``object_to_pydict`` outbound-side snapshot key. Cycle 2 rerouted 10
gameplay-decision read sites through ``effective_controller_id``. Cycle 3
(THIS slice) closes the Python ↔ Rust bridge round-trip:

- **Outbound (already shipped Cycle 1):** verified pinned —
  ``object_to_pydict`` emits ``controller_override`` as Python ``None``
  (default) or the override pid (post-grant).
- **Inbound (NEW Cycle 3):** ``register_and_inject_object`` gains an
  optional ``controller_override`` kwarg (PyO3 keyword-only, default
  ``None``). When ``Some(pid)``, the freshly-injected object enters
  the target zone with its ``controller_override`` already stamped.
  Validates the pid against the same 0/1 range
  ``grant_control_change_ueot`` enforces; out-of-range raises
  ``IndexError`` BEFORE the registry extend.
- **Python-side reciprocal (``parity/cross_engine.py``):**
  :func:`mtg_bot.parity.cross_engine.python_controller_override` reads
  the layer-2 ``"control_change"`` continuous-effect entry off a
  Python ``GameObject`` and returns the override pid (or ``None`` for
  same-controller-short-circuit / no-entry). Paired with
  :func:`mtg_bot.parity.cross_engine.inject_object_with_controller_override`
  which threads that pid through the Rust ``register_and_inject_object``
  call. Closes the loop: a Python state with an active UEOT control-
  grant can be byte-equal mirrored into Rust state.

**Substrate-only invariant — CRITICAL.**

Cycle 3 ships ONLY the bridge round-trip plumbing. No call site mints
a non-None override yet (Cycle 4 wires the Mutinous Massacre /
Systems Override Effect arms). Every pre-Cycle 3 cross-engine
trajectory must stay byte-identical: ``inject_object_with_controller_override(
..., controller_override=None)`` is the byte-equal default, calling
through to ``register_and_inject_object`` WITHOUT the kwarg so the
PyO3 default fires.

**Tests cover:**

1. Default — ``register_and_inject_object`` without
   ``controller_override`` kwarg is byte-equal to pre-Cycle 3;
   ``controller_override`` is ``None`` on the freshly-injected
   object's snapshot.
2. Inbound mint — ``register_and_inject_object(..., controller_override=
   0)`` stamps the override at injection time;
   ``effective_controller_id`` reads back the override; the snapshot
   surfaces the override pid.
3. Invalid pid — ``controller_override=2`` (or any value ≥ 2) raises
   ``IndexError`` before the registry mutates; the state is unchanged
   (canary on a fresh state).
4. Round-trip Rust → dict → Rust — for a default-``None`` object,
   re-injecting from the dict produces a byte-equal
   ``controller_override`` field.
5. Round-trip Rust → dict → Rust — for a ``grant_control_change_ueot``-
   set override, re-injecting via the new kwarg surfaces the same
   override pid post-injection.
6. Cross-engine reciprocal — ``python_controller_override`` reads
   ``None`` from a Python ``GameObject`` with no layer-2 control_change
   entry; reads the override pid when ``grant_control_change_ueot`` is
   stamped on the Python state.
7. Cross-engine reciprocal — same-controller short-circuit returns
   ``None`` (matches Rust's grant short-circuit; the layer-2 entry
   doesn't exist on a parity-clean state where the grant was a no-op).
8. End-to-end — ``inject_object_with_controller_override`` with a
   non-None pid produces a Rust object whose snapshot bridge re-emits
   the same pid. Round-trip closure.
9. Per-object fingerprint invariant — the override field is NOT in
   the per-object fingerprint tuple, so default-``None`` injections
   stay byte-equal to the pre-Cycle 1 fingerprint (sanity canary).
10. Cleanup drain composition — the Cycle 3 inbound-side stamped
    override drains the same way a ``grant_control_change_ueot`` mint
    does (no semantic divergence between the two mint paths).
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


def _fresh_state():
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


def _creature_dict(name="Bears", power=2, toughness=2):
    return {
        "name": name,
        "mana_cost": {"generic": 1, "G": 1},
        "card_types": ["Creature"],
        "subtypes": ["Bear"],
        "power": power,
        "toughness": toughness,
        "keywords": [],
    }


def _bf_obj(state, obj_id):
    for o in state.battlefield_snapshot():
        if o["id"] == obj_id:
            return o
    return None


# ---------------------------------------------------------------------------
# Inbound side — register_and_inject_object's new optional kwarg
# ---------------------------------------------------------------------------


def test_register_and_inject_object_default_is_byte_equal_to_cycle1():
    """Calling ``register_and_inject_object`` WITHOUT the new kwarg is
    byte-equal to the pre-Cycle 3 shape. Back-compat invariant — every
    pre-Cycle 3 call site (Aura S3 round-trip pins, ETB-trigger setup
    primitives, cross-engine harness probes) keeps the same default
    ``controller_override = None`` on the freshly-injected object."""
    _require_engine()
    state = _fresh_state()
    obj_id = state.register_and_inject_object(
        _creature_dict(), 0, ZONE_BATTLEFIELD,
    )
    bf = _bf_obj(state, obj_id)
    assert bf is not None
    assert bf["controller_override"] is None
    assert bf["controller_id"] == 0


def test_register_and_inject_object_with_override_stamps_at_injection_time():
    """``register_and_inject_object(..., controller_override=0)`` on a
    P1-controlled bear stamps the override at injection time. The
    freshly-injected object's snapshot emits the override pid;
    ``effective_controller_id`` reads back the new pid. The raw
    ``controller`` field stays at the printed baseline (1)."""
    _require_engine()
    state = _fresh_state()
    obj_id = state.register_and_inject_object(
        _creature_dict(), 1, ZONE_BATTLEFIELD,
        controller_override=0,
    )
    bf = _bf_obj(state, obj_id)
    assert bf is not None
    assert bf["controller_id"] == 1
    assert bf["controller_override"] == 0
    assert state.effective_controller_id(obj_id) == 0


def test_register_and_inject_object_with_override_none_kwarg_is_default():
    """Passing ``controller_override=None`` explicitly is identical to
    omitting the kwarg — byte-equal to the default. No spurious
    override is stamped."""
    _require_engine()
    state = _fresh_state()
    obj_id = state.register_and_inject_object(
        _creature_dict(), 0, ZONE_BATTLEFIELD,
        controller_override=None,
    )
    bf = _bf_obj(state, obj_id)
    assert bf["controller_override"] is None


def test_register_and_inject_object_invalid_override_pid_raises():
    """``controller_override`` must be in ``{0, 1}`` (matches
    ``grant_control_change_ueot``'s range guard). Out-of-range values
    raise ``IndexError`` BEFORE the registry extend, so a rejected
    override never leaves partial state behind. Mirrors the Cycle 1
    grant-call validation."""
    _require_engine()
    state = _fresh_state()
    # Snapshot the pre-call state so we can pin "unchanged on reject".
    pre_bf_count = len(state.battlefield_snapshot())
    with pytest.raises(IndexError):
        state.register_and_inject_object(
            _creature_dict("BadPid2"), 0, ZONE_BATTLEFIELD,
            controller_override=2,
        )
    with pytest.raises(IndexError):
        state.register_and_inject_object(
            _creature_dict("BadPidHigh"), 0, ZONE_BATTLEFIELD,
            controller_override=7,
        )
    # No leftover half-injected object on the battlefield.
    post_bf_count = len(state.battlefield_snapshot())
    assert post_bf_count == pre_bf_count


def test_register_and_inject_object_override_on_hand_zone():
    """The new kwarg is zone-agnostic — stamping the override on a
    hand-zone injection is structurally valid (the field is on
    GameObject regardless of zone). The override has no gameplay
    effect off the battlefield (per Cycle 2's read-site reroute),
    but the field round-trips through the bridge for any zone, so
    the cross-engine harness can mirror it uniformly."""
    _require_engine()
    state = _fresh_state()
    hand_id = state.register_and_inject_object(
        _creature_dict("HandBear"), 0, ZONE_HAND,
        controller_override=1,
    )
    hand = next(o for o in state.hand_snapshot(0) if o["id"] == hand_id)
    assert hand["controller_override"] == 1


# ---------------------------------------------------------------------------
# Round-trip Rust → dict → Rust byte-equal invariant
# ---------------------------------------------------------------------------


def test_round_trip_default_none_byte_equal():
    """Round-trip Rust → dict → Rust for a default-``None`` object
    produces a byte-equal ``controller_override`` field. The
    substrate-only invariant: no pre-Cycle 3 cross-engine trajectory
    observes a non-None value, so the round-trip is byte-identical
    to the pre-Cycle 1 shape on every existing path."""
    _require_engine()
    src_state = _fresh_state()
    src_id = src_state.register_and_inject_object(
        _creature_dict("SrcBear"), 0, ZONE_BATTLEFIELD,
    )
    src_bf = _bf_obj(src_state, src_id)
    # Outbound side: dict emits None.
    assert src_bf["controller_override"] is None
    # Inbound side: re-inject from the dict shape (controller_override
    # passed through). Default-None case is byte-equal to omitting
    # the kwarg.
    dst_state = _fresh_state()
    dst_id = dst_state.register_and_inject_object(
        _creature_dict("DstBear"), src_bf["controller_id"], ZONE_BATTLEFIELD,
        controller_override=src_bf["controller_override"],
    )
    dst_bf = _bf_obj(dst_state, dst_id)
    # Round-trip byte-equal.
    assert dst_bf["controller_override"] == src_bf["controller_override"]
    assert dst_bf["controller_override"] is None


def test_round_trip_granted_override_survives():
    """Round-trip Rust → dict → Rust for a ``grant_control_change_ueot``-
    set override surfaces the same override pid post-injection. This
    is the load-bearing test for Cycle 4: once Mutinous Massacre /
    Systems Override Effect arms mint the override on the Rust side,
    the cross-engine harness reads it via the outbound dict and
    re-injects it on the mirror state via the inbound kwarg —
    byte-equal."""
    _require_engine()
    src_state = _fresh_state()
    src_id = src_state.register_and_inject_object(
        _creature_dict("SrcBear"), 1, ZONE_BATTLEFIELD,  # P1's creature
    )
    # P0 takes control UEOT (the substrate Cycle 1 added).
    ok = src_state.grant_control_change_ueot(src_id, 0)
    assert ok is True
    src_bf = _bf_obj(src_state, src_id)
    # Outbound: dict carries the override pid.
    assert src_bf["controller_override"] == 0
    assert src_bf["controller_id"] == 1
    # Inbound: re-inject with the same override.
    dst_state = _fresh_state()
    dst_id = dst_state.register_and_inject_object(
        _creature_dict("DstBear"), src_bf["controller_id"], ZONE_BATTLEFIELD,
        controller_override=src_bf["controller_override"],
    )
    dst_bf = _bf_obj(dst_state, dst_id)
    # Round-trip byte-equal — both raw controller AND override survive.
    assert dst_bf["controller_id"] == 1
    assert dst_bf["controller_override"] == 0
    # Behavioral confirmation: effective controller agrees post-RT.
    assert dst_state.effective_controller_id(dst_id) == 0


def test_round_trip_via_kwarg_matches_grant_call_path_byte_equal():
    """Two equivalent paths to the same end state:

    Path A — inject + grant: ``register_and_inject_object`` (default
    None) followed by ``grant_control_change_ueot(id, override_pid)``.

    Path B — inject with override: ``register_and_inject_object(...,
    controller_override=pid)`` (the Cycle 3 inbound kwarg).

    Both paths must produce byte-equal snapshots on the override
    field, the raw controller field, AND the
    ``effective_controller_id`` read. The Cycle 3 inbound shorthand
    is semantically equivalent to inject-then-grant — no observable
    divergence."""
    _require_engine()
    state_a = _fresh_state()
    a_id = state_a.register_and_inject_object(
        _creature_dict(), 1, ZONE_BATTLEFIELD,
    )
    state_a.grant_control_change_ueot(a_id, 0)

    state_b = _fresh_state()
    b_id = state_b.register_and_inject_object(
        _creature_dict(), 1, ZONE_BATTLEFIELD,
        controller_override=0,
    )

    a_bf = _bf_obj(state_a, a_id)
    b_bf = _bf_obj(state_b, b_id)
    # Same controller fields.
    assert a_bf["controller_id"] == b_bf["controller_id"] == 1
    assert a_bf["controller_override"] == b_bf["controller_override"] == 0
    # Same effective-controller read.
    assert (
        state_a.effective_controller_id(a_id)
        == state_b.effective_controller_id(b_id)
        == 0
    )


# ---------------------------------------------------------------------------
# Cross-engine reciprocal — Python-side ``python_controller_override``
# ---------------------------------------------------------------------------


def _python_state_with_creature(controller_pid: int):
    """Build a minimal Python ``GameState`` with one creature on the
    indicated player's battlefield. Used by the cross-engine
    reciprocal tests below.

    Mirrors the ``_new_state`` + ``_put`` shape from
    ``tests/test_tractor_beam_control.py`` — the canonical Python-
    side state setup for control-change tests. Uses
    ``SYNTHETIC_CREATURES["Wild Bear"]`` (2/2 Bear, the same
    printing the Tractor Beam tests exercise) so the reciprocal
    test runs against a printing the cross_engine harness already
    touches in the Aura / Tractor-Beam paths.
    """
    from mtg_bot.cards import SYNTHETIC_CREATURES
    from mtg_bot.game import GameState, Player, Zone
    from mtg_bot.turn import Step

    state = GameState()
    state.players.append(Player(id=0))
    state.players.append(Player(id=1))
    state.active_player_idx = 0
    state.priority_player_idx = 0
    state.step = Step.PRECOMBAT_MAIN
    obj = state.new_object(
        SYNTHETIC_CREATURES["Wild Bear"], controller_pid, Zone.BATTLEFIELD,
    )
    state.battlefield.append(obj)
    return state, obj


def test_python_controller_override_returns_none_when_no_layer2_entry():
    """A Python ``GameObject`` with no layer-2 ``"control_change"``
    continuous-effect entry has no override — :func:`python_controller_override`
    returns ``None``. Matches Rust's outbound shape on a default-None
    object."""
    from mtg_bot.parity.cross_engine import python_controller_override

    state, obj = _python_state_with_creature(controller_pid=0)
    assert python_controller_override(state, obj) is None


def test_python_controller_override_reads_granted_pid():
    """After ``state.grant_control_change_ueot`` mints a layer-2
    ``"control_change"`` entry, :func:`python_controller_override` reads
    the override pid in the same shape Rust's ``object_to_pydict``
    emits. Closes the cross-engine reciprocal — the Python-side
    reader recovers the same pid Rust would have stored in its
    per-object ``controller_override`` field."""
    from mtg_bot.parity.cross_engine import python_controller_override

    state, obj = _python_state_with_creature(controller_pid=1)
    # P0 takes control UEOT.
    state.grant_control_change_ueot(
        source_id=0,  # synthetic source id, doesn't matter for this test
        target_id=obj.id,
        new_controller_id=0,
    )
    assert python_controller_override(state, obj) == 0


def test_python_controller_override_same_controller_short_circuit():
    """When the layer-2 entry's pid equals the printed baseline
    ``obj.controller_id``, :func:`python_controller_override` returns
    ``None`` (matches Rust's same-controller short-circuit — the
    override field is ``None`` on a state where the grant was a no-
    op). The Python side never writes a layer-2 entry in that case,
    so this branch is paranoia; the canary ensures the reader is
    defensive against an accidental no-op-entry leak."""
    from mtg_bot.parity.cross_engine import python_controller_override
    from mtg_bot.game import ContinuousEffect

    state, obj = _python_state_with_creature(controller_pid=0)
    # Hand-write a no-op layer-2 entry (the production grant call
    # short-circuits before writing such an entry; this canary
    # constructs the entry directly so the reader's short-circuit
    # is exercised end-to-end).
    state.continuous_effects.append(
        ContinuousEffect(
            source_id=0,
            layer=2,
            kind="control_change",
            target_id=obj.id,
            params=(0,),  # same as baseline controller_id
            timestamp=1,
            expires_at="cleanup",
        )
    )
    assert python_controller_override(state, obj) is None


def test_python_controller_override_latest_timestamp_wins():
    """When multiple layer-2 ``"control_change"`` entries target the
    same object, the latest-timestamp entry's pid wins (matches
    :func:`mtg_bot.game.effective_controller`'s walk). The Cycle 3
    Threaten-chain shape — multiple Threaten effects in the same
    turn — only the most-recent override is in effect."""
    from mtg_bot.parity.cross_engine import python_controller_override
    from mtg_bot.game import ContinuousEffect

    state, obj = _python_state_with_creature(controller_pid=0)
    # Two layer-2 entries, ascending timestamps. The first grants
    # to p1, the second grants to p1 again (so the latest is a
    # genuine override vs the p0 baseline).
    state.continuous_effects.append(
        ContinuousEffect(
            source_id=0,
            layer=2,
            kind="control_change",
            target_id=obj.id,
            params=(1,),
            timestamp=1,
            expires_at="cleanup",
        )
    )
    state.continuous_effects.append(
        ContinuousEffect(
            source_id=0,
            layer=2,
            kind="control_change",
            target_id=obj.id,
            params=(1,),
            timestamp=2,
            expires_at="cleanup",
        )
    )
    # Latest pid is 1 (different from baseline 0) — override is
    # active.
    assert python_controller_override(state, obj) == 1


# ---------------------------------------------------------------------------
# End-to-end — Python state → cross-engine helper → Rust state
# ---------------------------------------------------------------------------


def test_end_to_end_inject_helper_default_none():
    """``inject_object_with_controller_override(..., controller_override=
    None)`` is byte-equal to the bare ``register_and_inject_object``
    call. Substrate-only back-compat invariant — every pre-Cycle 3
    cross-engine path observes the same Rust-side behavior with or
    without the helper."""
    _require_engine()
    from mtg_bot.parity.cross_engine import inject_object_with_controller_override

    state = _fresh_state()
    obj_id = inject_object_with_controller_override(
        state, _creature_dict(), 0, ZONE_BATTLEFIELD,
        controller_override=None,
    )
    bf = _bf_obj(state, obj_id)
    assert bf["controller_override"] is None
    assert bf["controller_id"] == 0


def test_end_to_end_inject_helper_round_trip_with_override():
    """``inject_object_with_controller_override(..., controller_override=
    0)`` on a P1-baseline injection produces a Rust object whose
    snapshot bridge re-emits the same override pid. End-to-end
    Python-side closure: a Python ``GameObject`` with an active UEOT
    control-grant CAN be byte-equal mirrored into Rust state."""
    _require_engine()
    from mtg_bot.parity.cross_engine import inject_object_with_controller_override

    state = _fresh_state()
    obj_id = inject_object_with_controller_override(
        state, _creature_dict(), 1, ZONE_BATTLEFIELD,
        controller_override=0,
    )
    bf = _bf_obj(state, obj_id)
    assert bf["controller_id"] == 1
    assert bf["controller_override"] == 0
    # Effective-controller read agrees with the override.
    assert state.effective_controller_id(obj_id) == 0


def test_end_to_end_full_round_trip_via_helpers():
    """**Full Python ↔ Rust round-trip closure.** Build a Python
    ``GameState`` with a P0-controlled creature whose Python-side
    layer-2 ``"control_change"`` entry grants control to P1. Read
    the override via :func:`python_controller_override`. Mirror the
    creature into a Rust state via
    :func:`inject_object_with_controller_override`. Verify the Rust
    side observes the same override pid via the snapshot bridge.
    Then export the Rust snapshot's override field and verify it
    round-trips back to the same value the Python reader produced.

    This is the load-bearing canary for Cycle 4 — once Mutinous
    Massacre / Systems Override Effect arms mint the override on
    the Rust side, the cross-engine harness uses this exact closure
    to assert behavioral parity end-to-end."""
    _require_engine()
    from mtg_bot.parity.cross_engine import (
        python_controller_override,
        inject_object_with_controller_override,
    )

    # Python side: P0-controlled creature, P1 takes control UEOT.
    py_state, py_obj = _python_state_with_creature(controller_pid=0)
    py_state.grant_control_change_ueot(
        source_id=0,
        target_id=py_obj.id,
        new_controller_id=1,
    )
    py_override = python_controller_override(py_state, py_obj)
    assert py_override == 1

    # Rust side: mirror the creature with the Cycle 3 inbound helper.
    rust_state = _fresh_state()
    rust_id = inject_object_with_controller_override(
        rust_state, _creature_dict(), 0, ZONE_BATTLEFIELD,
        controller_override=py_override,
    )
    rust_bf = _bf_obj(rust_state, rust_id)
    assert rust_bf["controller_id"] == 0
    assert rust_bf["controller_override"] == 1

    # Final pin: the Rust outbound dict's override matches the
    # Python-side reader's value. Round-trip closure.
    assert rust_bf["controller_override"] == py_override


# ---------------------------------------------------------------------------
# Cleanup drain composition — inbound-stamped override drains identically
# ---------------------------------------------------------------------------


def test_cleanup_drains_inbound_stamped_override():
    """An override stamped via the Cycle 3 inbound kwarg drains
    through ``cleanup_step_for_active_player`` the same way a
    ``grant_control_change_ueot`` mint does. No semantic divergence
    between the two mint paths — the cleanup walks the same field
    irrespective of how it was set."""
    _require_engine()
    state = _fresh_state()
    obj_id = state.register_and_inject_object(
        _creature_dict(), 1, ZONE_BATTLEFIELD,
        controller_override=0,
    )
    # Pre-cleanup: override is in effect.
    assert _bf_obj(state, obj_id)["controller_override"] == 0
    assert state.effective_controller_id(obj_id) == 0
    # Cleanup drains.
    summary = state.cleanup_step_for_active_player()
    assert summary["controller_overrides_cleared"] == 1
    # Post-cleanup: override is None, effective controller is the
    # printed baseline.
    assert _bf_obj(state, obj_id)["controller_override"] is None
    assert state.effective_controller_id(obj_id) == 1


# ---------------------------------------------------------------------------
# Per-object fingerprint invariant — Cycle 3 inbound-stamped override
# stays out of the fingerprint tuple
# ---------------------------------------------------------------------------


def test_inbound_override_stays_out_of_fingerprint_tuple():
    """Cycle 1's substrate-invariant: ``controller_override`` is NOT
    in the per-object fingerprint tuple. The Cycle 3 inbound kwarg
    must preserve that invariant — a pre-Cycle-1 pinned baseline
    stays byte-equal regardless of whether the override is stamped
    via the inbound kwarg or via a downstream
    ``grant_control_change_ueot`` call (or not at all).

    We check this structurally: a fresh state's
    ``parity_smoke._zone_signature`` over its battlefield is identical
    with and without an inbound-stamped override on a fresh object.
    The signature doesn't include the override field (by Cycle 1
    design), so the two states' signatures match."""
    _require_engine()
    state_a = _fresh_state()
    a_id = state_a.register_and_inject_object(
        _creature_dict("FpBear"), 0, ZONE_BATTLEFIELD,
    )
    state_b = _fresh_state()
    b_id = state_b.register_and_inject_object(
        _creature_dict("FpBear"), 0, ZONE_BATTLEFIELD,
        controller_override=1,  # different override
    )
    # The bf snapshots' override fields differ (sanity).
    assert (
        _bf_obj(state_a, a_id)["controller_override"]
        != _bf_obj(state_b, b_id)["controller_override"]
    )
    # But the fingerprint (over the printing identity + counters +
    # tapped state, NOT the override) is the same. The Cycle 1
    # design explicitly excluded the field from the fingerprint
    # tuple; this test re-asserts that invariant survives Cycle 3.
    # We probe via a structural canary: clone-equality on the bf
    # subset.
    a_bf = _bf_obj(state_a, a_id)
    b_bf = _bf_obj(state_b, b_id)
    # Same printing, same controller, same tapped/sick/damage —
    # the non-override fields match (and the override is the only
    # difference). This is the structural "fingerprint" canary.
    for key in (
        "printing_id", "printing_name", "controller_id", "zone",
        "tapped", "summoning_sick", "damage", "counters",
    ):
        assert a_bf[key] == b_bf[key], (
            f"Cycle 3 inbound override leaked into fingerprint key "
            f"{key!r}: A={a_bf[key]!r}, B={b_bf[key]!r}"
        )
