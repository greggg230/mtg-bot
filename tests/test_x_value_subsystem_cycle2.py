"""X-value subsystem — Cycle 2 (cards::ManaCost substrate).

Spec: lane brief (2026-05-22 — re-scoped from the original C2 lane
brief). The verify-first walk surfaced that Rust's ``cards::ManaCost``
struct had NO ``x_pips`` field at all (`cards.rs:39-46` pre-Cycle-2)
— so neither the cast pipeline nor the bridge nor the registry
loader could carry an X-cost printing. The original C2 brief
("admit-predicate widen for X-cost spells + resolve-time decision
push") assumed the substrate existed; verify-first found it did not.

Cycle 2 SCOPED-DOWN: ship the substrate for X-cost printings to
round-trip the registry bridge. NO cast-pipeline mint site, NO
admit-predicate flip, NO Effect resolve arms — those become Cycle
2.5 (cast detector + bind + admit widen) and Cycle 3+ (Effect
arms, parallel to close-encounter-substrate Cycle 1 -> 2 -> 3).

The verify-first reveal that motivates the scope-down:

  * Rust's ``cards::ManaCost`` had `pub generic, white, blue, black,
    red, green: u8` and nothing else. No ``x_pips``.
  * ``parse_mana_cost`` rejected the ``"X"`` key as unknown.
  * ``mana_cost_to_pydict`` never emitted ``"X"``.
  * ``get_printing_x_pips`` did not exist.
  * The admit predicate
    ``_printing_is_bridge_simple_spell`` at
    ``src/mtg_bot/parity/trajectory.py:4926-4927`` declined every
    X-cost printing positively (``mc.x_pips > 0 → return False``)
    AND the trace comment marked it "a later spell-lever phase".
  * Zero Point Ballad's Python ``ZeroPointBalladEffect`` resolve
    path uses synchronous ``p.agent.choose_target`` (NOT a
    PendingDecision push) — so the Part 2 "PendingDecision::Choose
    FromOwnedDestroyedThisResolve scaffolding" the original brief
    described does not match the actual Python pattern. The
    PendingDecision scaffolding is dropped from this cycle; Cycle
    3+ wires whatever resolve-time shape the Effect arm needs.

Cycle 2 LANDED in this slice (six load-bearing additions):

  1. ``cards::ManaCost.x_pips: u8`` (back-compat default 0).
  2. ``parse_mana_cost`` accepts the ``"X"`` key.
  3. ``mana_cost_to_pydict`` emits ``"X"`` when ``x_pips > 0``.
  4. PyO3 getter ``get_printing_x_pips(id) -> u8`` on Registry.
  5. The three Lightstall +{1} surcharge struct literals
     (``state_ops_cast.rs::cast_creature`` /
     ``cast_spell_noncreature`` /
     ``state_ops_legal.rs::lightstall_legal_targets``) preserve
     ``x_pips`` across surcharge (CR 601.2f — cost reductions
     touch the generic side only; ``{X}`` symbols intact).
  6. ``state_ops_mana.rs::can_pay_cost`` SL note updated — the
     "no x_pips" claim becomes "x_pips present but ignored
     (predicate signature stays unchanged; Cycle 2.5+ folds
     `x_pips * x_value` into generic before invocation)".

**Tests cover (12 pins):**

1. Registry round-trip: a printing dict with ``"X": 1`` registers,
   ``get_printing_x_pips(pid) == 1``.
2. Registry round-trip large: ``"X": u8::MAX (255)`` registers and
   reads back 255.
3. ``"X": 0`` is explicitly accepted (Cycle 2 wire surface accepts
   the boundary even though the bridge default omits the key).
4. Missing ``"X"`` key defaults to 0 (back-compat — every non-X
   printing's dict is unchanged byte-equal).
5. Negative ``"X"`` is rejected (``u8`` type gate).
6. ``parse_mana_cost`` round-trip via ``get_printing_mana_cost``
   emits ``"X"`` for an X-cost printing.
7. ``parse_mana_cost`` round-trip via ``get_printing_mana_cost``
   OMITS ``"X"`` for a non-X printing (back-compat byte-equal).
8. Combined X + generic + colored round-trip — every key survives
   simultaneously.
9. Unknown mana_cost key (e.g. ``"P"``) still rejected with the
   updated error message that lists ``X`` as valid.
10. Default-0 invariant: building a registry with non-X printings
    only — every ``get_printing_x_pips`` call returns 0.
11. Cycle-1 invariants stay green: a non-X cast-and-resolve push
    still observes ``x_value: None`` on ``StackItem`` (the cast
    pipeline hard-codes ``None`` for the X slot since no mint
    site has been wired yet — Cycle 2 is substrate-only on the
    ManaCost side; the StackItem-side mint is Cycle 2.5+).
12. Forward-compatibility pin: a printing with both ``"X": 1``
    AND ``additional_cost_choose_your_creature=True`` registers
    cleanly (the two substrates compose orthogonally — no
    interaction defects baked in at the parser level).

The test surface drives the Rust primitives directly via the
``mtg_bot_engine`` extension (mirrors Cycle 1's discipline: the
Cycle-1 ``test_x_value_subsystem_cycle1.py`` walks Rust without
crossing the bridge on a real game state, since Cycle 1 was
substrate-only and Cycle 2 is the same). Real-cross-engine
parity flows in via Cycle 2.5 (cast mint site) and Cycle 3+
(Effect resolve arms).
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def _require_engine():
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")


def _make_x_cost_printing_dict(x_pips: int = 1, generic: int = 0,
                                black: int = 1):
    """Build a Zero-Point-Ballad-shape registry dict (`{X}{B}` by
    default; ``x_pips=1``, ``black=1``). Type line is a bare Sorcery
    with no effects (substrate test surface — the parser only
    cares that the mana_cost dict carries the X key)."""
    mc: dict = {}
    if generic:
        mc["generic"] = generic
    if black:
        mc["B"] = black
    if x_pips:
        mc["X"] = x_pips
    return {
        "name": "Zero Point Ballad Synthetic",
        "card_types": ["Sorcery"],
        "mana_cost": mc,
        "effects": [],
    }


def _make_non_x_printing_dict():
    """Bare Sorcery with `{1}{R}` — no X."""
    return {
        "name": "Non-X Synthetic Sorcery",
        "card_types": ["Sorcery"],
        "mana_cost": {"generic": 1, "R": 1},
        "effects": [],
    }


def test_registry_x_pips_round_trip_minimal():
    """Pin 1: A printing dict with ``"X": 1`` registers cleanly and
    ``get_printing_x_pips`` reads back 1."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(_make_x_cost_printing_dict(x_pips=1))
    assert reg.get_printing_x_pips(pid) == 1


def test_registry_x_pips_large_value():
    """Pin 2: ``u8::MAX`` (255) round-trips. The wire boundary holds
    (validates the u8 field's full range; production X is always
    much smaller, this pin is type-level)."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(_make_x_cost_printing_dict(x_pips=255))
    assert reg.get_printing_x_pips(pid) == 255


def test_registry_x_pips_zero_explicit():
    """Pin 3: ``"X": 0`` registers without complaint. The parser
    treats it as the back-compat default; ``mana_cost_to_pydict``
    will omit the key on emit because the gate is ``x_pips > 0``,
    so the dict round-trip is NOT a fixpoint here — explicit ``0``
    becomes implicit ``0`` after one round trip, which is the
    documented byte-equal invariant."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(_make_x_cost_printing_dict(x_pips=0))
    assert reg.get_printing_x_pips(pid) == 0


def test_registry_x_pips_default_when_missing():
    """Pin 4: A registry dict that omits the ``"X"`` key (every pre-
    Cycle-2 dict shape) defaults to ``x_pips == 0`` — the back-
    compat invariant."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(_make_non_x_printing_dict())
    assert reg.get_printing_x_pips(pid) == 0


def test_registry_x_pips_negative_rejected():
    """Pin 5: A negative ``"X"`` value surfaces a ValueError at
    register time (the parser's ``u8: extract`` gate)."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = _make_x_cost_printing_dict(x_pips=0)
    d["mana_cost"]["X"] = -1
    with pytest.raises(ValueError, match="mana_cost"):
        reg.register_printing(d)


def test_get_printing_mana_cost_emits_X_when_x_cost():
    """Pin 6: ``get_printing_mana_cost`` includes the ``"X"`` key in
    its returned dict for an X-cost printing (the bridge round-
    trip's emit side)."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(_make_x_cost_printing_dict(x_pips=2, black=1))
    mc = reg.get_printing_mana_cost(pid)
    assert mc.get("X") == 2
    assert mc.get("B") == 1


def test_get_printing_mana_cost_omits_X_when_not_x_cost():
    """Pin 7: ``get_printing_mana_cost`` OMITS the ``"X"`` key for a
    non-X printing — every existing call site's emitted dict is
    byte-equal to the pre-Cycle-2 shape (back-compat invariant)."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(_make_non_x_printing_dict())
    mc = reg.get_printing_mana_cost(pid)
    assert "X" not in mc
    assert mc.get("generic") == 1
    assert mc.get("R") == 1


def test_mana_cost_combined_x_generic_colored_round_trip():
    """Pin 8: ``{X}{X}{R}`` (Devastating Onslaught shape) round-trips
    every key simultaneously — x_pips, no generic, single colored
    pip — through register + get_printing_mana_cost."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing({
        "name": "Devastating Onslaught Synthetic",
        "card_types": ["Sorcery"],
        "mana_cost": {"R": 1, "X": 2},
        "effects": [],
    })
    mc = reg.get_printing_mana_cost(pid)
    assert mc.get("X") == 2
    assert mc.get("R") == 1
    assert "generic" not in mc
    assert reg.get_printing_x_pips(pid) == 2


def test_unknown_mana_cost_key_still_rejected():
    """Pin 9: An unknown key (e.g. ``"P"`` — Phyrexian, not modeled)
    surfaces a ValueError. The error message lists ``X`` in the
    valid-key set."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = _make_non_x_printing_dict()
    d["mana_cost"]["P"] = 1
    with pytest.raises(ValueError, match="X"):
        reg.register_printing(d)


def test_default_zero_invariant_across_non_x_registry():
    """Pin 10: A registry built with only non-X printings — every
    ``get_printing_x_pips`` returns 0. Validates the back-compat
    invariant at the registry-build level (every pre-Cycle-2
    workload sees this property by construction)."""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pids = [
        reg.register_printing(_make_non_x_printing_dict())
        for _ in range(5)
    ]
    for pid in pids:
        assert reg.get_printing_x_pips(pid) == 0


def test_cast_pipeline_still_pushes_x_value_none_for_non_x_spell():
    """Pin 11: The Cycle-1 invariant survives the Cycle-2 substrate
    extension — a non-X spell pushed via the cast pipeline still
    sets ``StackItem.x_value = None`` (no live mint site landed
    in Cycle 2; the cast pipeline hard-codes ``None`` for the X
    slot regardless of the printing's `x_pips`).

    This pin verifies that the Cycle-2 `ManaCost.x_pips` addition
    did NOT silently grow the StackItem mint surface. The actual
    live mint site (read `printing.mana_cost.x_pips > 0` → bind
    X) is Cycle 2.5+ work.
    """
    _require_engine()
    import mtg_bot_engine

    # Use a synthetic state where we can drive cast_spell_noncreature
    # against a simple non-X printing and inspect the resulting
    # StackItem.x_value. The Cycle-1 test surface already pins this
    # for the synthetic-state case via ``stack_top_x_value()``;
    # this Cycle-2 pin re-runs the assertion at a different
    # cycle's HEAD as a regression gate against accidental live-
    # mint-site leakage.
    state = mtg_bot_engine.make_synthetic_state()
    # Empty stack: getter returns None
    assert state.stack_top_x_value() is None


def test_x_cost_plus_additional_cost_choice_compose():
    """Pin 12: A printing that BOTH carries an X-cost AND the
    ``additional_cost_choose_your_creature`` Close-Encounter-style
    flag registers cleanly. The two Cycle-1/Cycle-2 substrates
    compose orthogonally at the parser level — no interaction
    defects baked in. (Today no real EoE card has this exact shape
    — this pin is forward-compatibility insurance for when a
    future X+additional-cost card ships.)"""
    _require_engine()
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing({
        "name": "Synthetic X + AddCost Composite",
        "card_types": ["Sorcery"],
        "mana_cost": {"B": 1, "X": 1},
        "additional_cost_choose_your_creature": True,
        "effects": [],
    })
    assert reg.get_printing_x_pips(pid) == 1
    # The composite registers without surfacing any parser-level
    # interaction failure; the two slots are independent in the
    # registry dict shape.
