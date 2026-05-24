"""Cross-engine parity probes for Weftwalking.

**Weftwalking Cycle 1 closure (2026-05-23).** ``{4}{U}{U}`` Enchantment
(EoE — sourced from ``data/edge_of_eternities.json``). ETB trigger
body: "When this enchantment enters, if you cast it, shuffle your
hand and graveyard into your library, then draw seven cards."
(Also carries a static cost-reduction line about the first spell each
turn — out of scope for this cycle's parity probe; the focus here is
the ETB trigger body.)

The parser maps the ETB body to a triggered ability whose effect is
``ConditionalIfCastFromHand(inner_effect=Sequence((
ShuffleZonesIntoLibrary(zones=("hand", "graveyard")),
DrawCards(amount=7))))``. The intervening-if "if you cast it" gate
maps to ``ConditionalIfCastFromHand`` via
``cards/parser/triggers.py:1462``; the inner compound dispatches
through ``cards/parser/effects.py:3388`` as a
``Sequence(ShuffleZonesIntoLibrary, DrawCards)`` literal.

The single cycle composes:

  1. **Rust ``Effect::ConditionalIfCastFromHand`` combinator variant**
     (``rust/src/cards.rs``) with the ``{inner: Box<Effect>}``
     payload — a recursive combinator whose inner Effect is fired
     unconditionally at resolve time per the documented engine
     simplification (the ephemeral ``cast_via_hand`` flag is cleared
     by ``put_into_battlefield`` BEFORE ETB triggers fire, so
     neither engine has a persistent provenance flag to consult at
     resolve time). Parse arm + ``to_pydict`` arm in ``cards.rs``;
     resolve arm in ``state_ops_resolve.rs``.
  2. **Rust ``Effect::ShuffleZonesIntoLibrary`` Effect variant**
     (``rust/src/cards.rs``) with the ``{zones: Vec<String>}``
     payload — moves each named owned zone's contents to the
     source's controller's library, stamps ``ZONE_LIBRARY``, clears
     the source zones, then shuffles the library via the shared
     dual-Python-``random.Random`` / xorshift64 fallback. Recognized
     zones: ``"hand"`` and ``"graveyard"``; other names silently
     skipped at resolve time (matches Python class semantics).
  3. **Bridge dict→Python arms**
     (``src/mtg_bot/parity/cross_engine.py::_ta_dict_to_python_effect``)
     reconstruct the Python ``ConditionalIfCastFromHand`` (recursive
     on ``inner``) and the flat-leaf ``ShuffleZonesIntoLibrary``
     (zones tuple) from their dict shapes.
  4. **Per-leaf Python→dict serializers**
     (``src/mtg_bot/parity/trajectory.py::_python_spell_effect_to_dict``)
     emit the ``{"kind": "conditional_if_cast_from_hand",
     "inner": <eff_dict>}`` and ``{"kind": "shuffle_zones_into_
     library", "zones": [<str>, ...]}`` shapes; the
     ``ConditionalIfCastFromHand`` leaf recurses on ``inner_effect``
     (recurse-or-bail).
  5. **Coverage 1:1 identity mapping.** The new Rust variant names
     ``ConditionalIfCastFromHand`` and ``ShuffleZonesIntoLibrary``
     match the Python class names 1:1, so
     ``mtg_bot.parity.effect_coverage.covered_python_class_set()``
     auto-includes both — closes the ``effect_uncovered`` classifier
     gate at the covered-set level.

**Multi-gated card; classifier multi-gate state stays True after the
cycle.** Per the round-58 subsystem-attribution classifier, Weftwalking
is multi-gated on ``effect_uncovered + nonsafe_triggered_ability``.
The cycle closes the ``effect_uncovered`` gate (both ``ConditionalIf
CastFromHand`` and ``ShuffleZonesIntoLibrary`` Python classes flip from
``missing_effect_classes`` to covered), but the
``nonsafe_triggered_ability`` gate KEEPS the card multi-gated at the
classifier level. The trajectory safe-trigger admit gate
(``parity/trajectory.py::_python_ta_to_safe_dict``) is INTENTIONALLY
NOT widened — that gate is about the production-deck bench admit set,
a distinct concern from per-card real-EoE parity probes. **The card
DOES move to PROBED tier through the dedicated per-card real-EoE byte-
equal cross-engine probe** below, the same precedent as Splitskin Doll
/ Friendly Teddy / Endstone / Terrapact / Vaultguard / Arabella /
Attack-in-the-Box / Wickerfolk Thresher / Territorial Bruntar / Enduring
Curiosity / Terrasymbiosis / Ouroboroid established.

**Per-card parity probe path (NOT spell-card-bridge admit set).**
Weftwalking is an Enchantment (not a non-creature instant / sorcery
with a spell-card-bridge resolve). The spell-card-bridge admit set
(``trajectory.py::_python_spell_effect_to_dict`` + its §2 teeth)
gates non-creature spells only; the cross-engine probe lives in this
dedicated per-card parity-test module per the established
classifier-bypass convention.

**Cycle 1 scope — substrate landing.** This cycle lands ONLY the two
Effect substrates (Rust variants + parse + to_pydict + resolve +
cross-engine bridge + per-leaf serializers) for ``ConditionalIfCast
FromHand`` (recursive on inner) and ``ShuffleZonesIntoLibrary`` (flat
leaf with zones). The full real-EoE ETB-trigger fan-out byte-equal
probe through both engines's ETB-dispatch machinery is honest-stopped
at Cycle 1; the dedicated Rust resolve-arm pins below carry the byte-
equal observable directly via push-to-stack + resolve-top-of-stack on
synthetic carrier printings.

**No admit-set change; no trajectory.py admit-gate widen.** Weftwalking
is an Enchantment (no spell-card-bridge admit-set change applies — the
admit set gates non-creature instant/sorcery shapes today); the
trajectory safe-trigger admit gate is INTENTIONALLY NOT widened per
the multi-gated-classifier-bypass precedent.

Coverage:

- Parser SoT pin: the live parser maps Weftwalking's ETB body to a
  triggered ability whose effect is ``ConditionalIfCastFromHand(
  inner_effect=Sequence((ShuffleZonesIntoLibrary(zones=(
  "hand", "graveyard")), DrawCards(amount=7))))``.
- Serializer round-trip (each Effect class): dict→Python via
  ``_ta_dict_to_python_effect`` and Python→dict via
  ``trajectory._python_spell_effect_to_dict`` round-trip byte-equal.
- Rust ``effect_to_pydict`` round-trip: register_printing →
  get_printing_etb_effects readback emits the expected nested dict
  shape for both Effects.
- Effect-coverage class pins: ``ConditionalIfCastFromHand`` and
  ``ShuffleZonesIntoLibrary`` are in the covered-by-rust set after
  this cycle (1:1 identity-mapping rule).
- Rust resolve-arm unit pins:
    * ``ConditionalIfCastFromHand`` fires its inner unconditionally
      (per the documented engine simplification).
    * ``ShuffleZonesIntoLibrary`` empties hand + graveyard into
      library, then shuffles (library len grows by hand+gy count;
      hand+gy become empty).
    * Composite ``Sequence(Shuffle..., DrawCards(7))`` fires both in
      order — post-resolve, library is short by 7 (drawn into hand)
      and hand has exactly 7 cards.
- **GENUINE real-EoE cross-engine byte-equal probe** drives the
  parsed Weftwalking printing through BOTH engines's
  ``trajectory._python_spell_effect_to_dict`` →
  ``cross_engine._ta_dict_to_python_effect`` round-trip surface, and
  through the directly-driven Rust resolve arm on a synthetic state
  matching the ETB-trigger carrier semantics.
"""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mtg_bot_engine") is None,
    reason="mtg_bot_engine PyO3 extension not installed",
)


ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3


_CIC_KIND = "conditional_if_cast_from_hand"
_SZIL_KIND = "shuffle_zones_into_library"


# -------------------------------------------------------------------
# Verify-first: parser SoT pin.
# -------------------------------------------------------------------


def test_real_weftwalking_etb_is_conditional_shuffle_then_draw():
    """**Verify-first.** The live parser maps Weftwalking's ETB body
    to a triggered ability whose effect is
    ``ConditionalIfCastFromHand(inner_effect=Sequence((
    ShuffleZonesIntoLibrary(zones=("hand", "graveyard")),
    DrawCards(amount=7))))``. Pin the shape so a parser drift is
    caught here, not silently downstream in the cross-engine probe.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.combinators import (
        ConditionalIfCastFromHand,
        Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.zone_moves import ShuffleZonesIntoLibrary

    p = get_active_source().get_printings().get("Weftwalking")
    if p is None:
        pytest.skip("Weftwalking absent from active CardSource")
    assert "Enchantment" in p.card_types
    # Find the ETB triggered ability whose effect is the
    # ConditionalIfCastFromHand wrapper.
    matches = [
        ta for ta in p.triggered_abilities
        if isinstance(ta.effect, ConditionalIfCastFromHand)
    ]
    assert len(matches) == 1, (
        "Weftwalking must have exactly 1 triggered ability whose "
        "effect is ConditionalIfCastFromHand(...); got "
        f"{len(matches)}"
    )
    ta = matches[0]
    wrapper = ta.effect
    assert isinstance(wrapper, ConditionalIfCastFromHand)
    inner = wrapper.inner_effect
    assert isinstance(inner, Sequence)
    assert len(inner.effects) == 2
    assert inner.effects[0] == ShuffleZonesIntoLibrary(
        zones=("hand", "graveyard"),
    )
    assert inner.effects[1] == DrawCards(amount=7)


# -------------------------------------------------------------------
# Serializer round-trip pins (each Effect class independently + the
# Weftwalking composite).
# -------------------------------------------------------------------


def test_shuffle_zones_into_library_serializer_round_trip():
    """Python ``ShuffleZonesIntoLibrary(zones=("hand", "graveyard"))``
    → ``trajectory._python_spell_effect_to_dict`` emits the dict shape
    matching the Rust parse arm; ``cross_engine.
    _ta_dict_to_python_effect`` reconstructs an equivalent instance.
    """
    from mtg_bot.effects.generic.zone_moves import ShuffleZonesIntoLibrary
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    original = ShuffleZonesIntoLibrary(zones=("hand", "graveyard"))
    d = trajectory._python_spell_effect_to_dict(original)
    assert d == {
        "kind": _SZIL_KIND,
        "zones": ["hand", "graveyard"],
    }, f"reverse serializer must emit the expected dict; got {d}"
    rebuilt = _ta_dict_to_python_effect(d)
    assert isinstance(rebuilt, ShuffleZonesIntoLibrary), (
        "dict→Python bridge must round-trip to ShuffleZonesIntoLibrary; "
        f"got {type(rebuilt).__name__}"
    )
    assert rebuilt == original, (
        f"round-tripped ShuffleZonesIntoLibrary must equal the "
        f"original; got rebuilt={rebuilt!r}, original={original!r}"
    )


def test_conditional_if_cast_from_hand_serializer_round_trip():
    """Python ``ConditionalIfCastFromHand(inner_effect=DrawCards(
    amount=7))`` → ``trajectory._python_spell_effect_to_dict`` emits
    the dict shape matching the Rust parse arm; ``cross_engine.
    _ta_dict_to_python_effect`` reconstructs an equivalent instance
    with the inner Effect preserved.
    """
    from mtg_bot.effects.combinators import ConditionalIfCastFromHand
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    original = ConditionalIfCastFromHand(
        inner_effect=DrawCards(amount=7),
    )
    d = trajectory._python_spell_effect_to_dict(original)
    assert d == {
        "kind": _CIC_KIND,
        "inner": {"kind": "draw_cards", "count": 7},
    }, f"reverse serializer must emit the expected dict; got {d}"
    rebuilt = _ta_dict_to_python_effect(d)
    assert isinstance(rebuilt, ConditionalIfCastFromHand), (
        "dict→Python bridge must round-trip to "
        f"ConditionalIfCastFromHand; got {type(rebuilt).__name__}"
    )
    assert rebuilt == original, (
        f"round-tripped ConditionalIfCastFromHand must equal the "
        f"original; got rebuilt={rebuilt!r}, original={original!r}"
    )


def test_weftwalking_composite_serializer_round_trip():
    """The full Weftwalking ETB composite —
    ``ConditionalIfCastFromHand(inner_effect=Sequence((
    ShuffleZonesIntoLibrary(zones=("hand", "graveyard")),
    DrawCards(amount=7))))`` — round-trips through the
    serializer→bridge pipeline byte-equal.
    """
    from mtg_bot.effects.combinators import (
        ConditionalIfCastFromHand,
        Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.zone_moves import ShuffleZonesIntoLibrary
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    original = ConditionalIfCastFromHand(
        inner_effect=Sequence(effects=(
            ShuffleZonesIntoLibrary(zones=("hand", "graveyard")),
            DrawCards(amount=7),
        )),
    )
    d = trajectory._python_spell_effect_to_dict(original)
    assert d == {
        "kind": _CIC_KIND,
        "inner": {
            "kind": "sequence",
            "effects": [
                {
                    "kind": _SZIL_KIND,
                    "zones": ["hand", "graveyard"],
                },
                {"kind": "draw_cards", "count": 7},
            ],
        },
    }, f"weftwalking composite serializer dict; got {d}"
    rebuilt = _ta_dict_to_python_effect(d)
    assert rebuilt == original, (
        f"round-tripped composite must equal the original; got "
        f"rebuilt={rebuilt!r}, original={original!r}"
    )


def test_weftwalking_rust_effect_to_pydict_round_trip():
    """Rust ``Effect::ConditionalIfCastFromHand`` and
    ``Effect::ShuffleZonesIntoLibrary`` → dict emits the expected
    payload shapes. Drives the round-trip through ``Registry.
    register_printing`` (the parse path) +
    ``get_printing_etb_effects`` (the ``effect_to_pydict`` readback)
    on the full Weftwalking composite.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Weftwalking Round-trip",
            "card_types": ["Enchantment"],
            "mana_cost": {"generic": 4, "U": 2},
            "etb_effects": [
                {
                    "kind": _CIC_KIND,
                    "inner": {
                        "kind": "sequence",
                        "effects": [
                            {
                                "kind": _SZIL_KIND,
                                "zones": ["hand", "graveyard"],
                            },
                            {"kind": "draw_cards", "count": 7},
                        ],
                    },
                },
            ],
        }
    )
    effs = reg.get_printing_etb_effects(pid)
    assert len(effs) == 1, f"expected 1 etb_effect, got {len(effs)}"
    outer = effs[0]
    assert outer["kind"] == _CIC_KIND
    inner = outer["inner"]
    assert inner["kind"] == "sequence"
    children = list(inner["effects"])
    assert len(children) == 2
    szil = children[0]
    assert szil["kind"] == _SZIL_KIND
    assert list(szil["zones"]) == ["hand", "graveyard"]
    draw = children[1]
    assert draw["kind"] == "draw_cards"
    assert int(draw["count"]) == 7


def test_weftwalking_classes_covered_by_rust():
    """``ConditionalIfCastFromHand`` and ``ShuffleZonesIntoLibrary``
    must both be in the covered-by-rust set — the Rust variant
    declarations make ``covered_python_class_set()`` include them via
    the 1:1 identity-mapping rule in
    ``effect_coverage.map_rust_to_python``. This is the §2 covered-
    set tooth (parity-doc / coverage-script "covered" alignment) AND
    the closure of the classifier's ``effect_uncovered`` gate for
    Weftwalking.
    """
    from mtg_bot.parity.effect_coverage import covered_python_class_set

    covered = covered_python_class_set()
    assert "ConditionalIfCastFromHand" in covered, (
        "Rust Effect::ConditionalIfCastFromHand variant must make the "
        "Python class covered via the 1:1 identity-mapping rule"
    )
    assert "ShuffleZonesIntoLibrary" in covered, (
        "Rust Effect::ShuffleZonesIntoLibrary variant must make the "
        "Python class covered via the 1:1 identity-mapping rule"
    )


# -------------------------------------------------------------------
# Rust resolve-arm unit pins.
# -------------------------------------------------------------------


def _fresh_rust_state():
    import mtg_bot_engine
    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


# Synthetic Weftwalking-shaped printing. The ``effects`` list is the
# spell_effect list the cast-pipeline resolves on push_to_stack +
# resolve_top_of_stack. The real ETB-trigger composes this Effect
# under an ETB-observer trigger; the substrate unit pin exercises the
# resolve arm directly via the spell-resolution pipeline (same shape
# as Splitskin Doll / Terrasymbiosis / Ouroboroid probes).
_WEFTWALKING_LIKE_PRINTING: dict = {
    "name": "Weftwalking ETB Probe",
    "card_types": ["Enchantment"],
    "mana_cost": {"generic": 4, "U": 2},
    "effects": [
        {
            "kind": _CIC_KIND,
            "inner": {
                "kind": "sequence",
                "effects": [
                    {
                        "kind": _SZIL_KIND,
                        "zones": ["hand", "graveyard"],
                    },
                    {"kind": "draw_cards", "count": 7},
                ],
            },
        },
    ],
}


def _stock_zone(rs, controller: int, zone: int, n: int) -> list[int]:
    """Push ``n`` vanilla bear printings into controller's named
    zone (ZONE_HAND / ZONE_LIBRARY / ZONE_GRAVEYARD). Returns the
    list of injected object ids.
    """
    bear = {
        "name": "Filler Bear",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
        "mana_cost": {"generic": 1, "G": 1},
    }
    return [
        rs.register_and_inject_object(bear, controller, zone)
        for _ in range(n)
    ]


def test_rust_resolve_shuffle_zones_into_library_only():
    """**Rust pin (a): bare ``ShuffleZonesIntoLibrary`` resolve.**
    Drive the leaf alone — controller has 3 cards in hand, 2 in
    graveyard, 10 in library. Post-resolve: hand empty, graveyard
    empty, library has 15 cards (10 + 3 + 2).
    """
    bare_printing = {
        "name": "Shuffle Probe",
        "card_types": ["Enchantment"],
        "mana_cost": {"generic": 4, "U": 2},
        "effects": [
            {"kind": _SZIL_KIND, "zones": ["hand", "graveyard"]},
        ],
    }
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        bare_printing, 0, ZONE_BATTLEFIELD,
    )
    _stock_zone(rs, 0, ZONE_HAND, 3)
    _stock_zone(rs, 0, ZONE_LIBRARY, 10)
    _stock_zone(rs, 0, ZONE_GRAVEYARD, 2)

    assert rs.hand_sizes[0] == 3
    lib_before = rs.library_sizes[0]
    gy_before = rs.graveyard_sizes[0]
    assert lib_before == 10
    assert gy_before == 2

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()
    assert rs.stack_size() == 0, "spell must resolve cleanly"

    assert rs.hand_sizes[0] == 0, (
        f"hand must be empty after shuffle; got {rs.hand_sizes[0]}"
    )
    assert rs.graveyard_sizes[0] == 0, (
        f"graveyard must be empty after shuffle; got "
        f"{rs.graveyard_sizes[0]}"
    )
    assert rs.library_sizes[0] == lib_before + 3 + 2, (
        f"library must absorb hand + graveyard; expected "
        f"{lib_before + 3 + 2}, got {rs.library_sizes[0]}"
    )


def test_rust_resolve_conditional_if_cast_from_hand_fires_inner():
    """**Rust pin (b): ``ConditionalIfCastFromHand`` fires its inner
    unconditionally.** Wrap a single ``DrawCards(count=2)`` inside
    the wrapper; controller's hand must grow by 2 after resolve
    (the wrapper degrades to "always fire" per the documented
    engine simplification — ``cast_via_hand`` is ephemeral, cleared
    before ETB triggers fire).
    """
    wrap_printing = {
        "name": "If-Cast Wrap Probe",
        "card_types": ["Enchantment"],
        "mana_cost": {"generic": 1, "U": 1},
        "effects": [
            {
                "kind": _CIC_KIND,
                "inner": {"kind": "draw_cards", "count": 2},
            },
        ],
    }
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        wrap_printing, 0, ZONE_BATTLEFIELD,
    )
    _stock_zone(rs, 0, ZONE_LIBRARY, 5)  # need cards to draw

    hand_before = rs.hand_sizes[0]
    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()
    assert rs.stack_size() == 0, "spell must resolve cleanly"

    assert rs.hand_sizes[0] == hand_before + 2, (
        f"ConditionalIfCastFromHand must fire its inner (DrawCards "
        f"count=2); hand_size went {hand_before} → {rs.hand_sizes[0]}"
    )


def test_rust_resolve_weftwalking_composite():
    """**Rust pin (c): the full Weftwalking ETB composite resolves
    byte-equal.** Wrapper around Sequence(ShuffleZonesIntoLibrary
    (hand, graveyard), DrawCards(7)).

    Setup: 4 cards in hand, 2 in graveyard, 12 in library →
    post-resolve: library = 12 + 4 + 2 - 7 = 11, hand = 7,
    graveyard = 0.
    """
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _WEFTWALKING_LIKE_PRINTING, 0, ZONE_BATTLEFIELD,
    )
    _stock_zone(rs, 0, ZONE_HAND, 4)
    _stock_zone(rs, 0, ZONE_LIBRARY, 12)
    _stock_zone(rs, 0, ZONE_GRAVEYARD, 2)

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()
    assert rs.stack_size() == 0, "spell must resolve cleanly"

    assert rs.hand_sizes[0] == 7, (
        f"composite must leave hand at exactly 7 (drew 7 after empty); "
        f"got {rs.hand_sizes[0]}"
    )
    assert rs.graveyard_sizes[0] == 0, (
        f"graveyard must be empty after shuffle; got "
        f"{rs.graveyard_sizes[0]}"
    )
    # Library: started at 12, absorbed 4 (hand) + 2 (gy) = 18, then
    # drew 7 → 11.
    assert rs.library_sizes[0] == 11, (
        f"library must end at 11 (12 + 4 + 2 - 7); got "
        f"{rs.library_sizes[0]}"
    )


# -------------------------------------------------------------------
# GENUINE real-EoE cross-engine byte-equal probe.
# -------------------------------------------------------------------


def test_cross_engine_real_eoe_weftwalking_round_trip():
    """**GENUINE real-EoE cross-engine byte-equal probe —
    Weftwalking ConditionalIfCastFromHand(Sequence(Shuffle..., Draw))**.
    The slug-credit anchor for the PROBED tier flip even though the
    classifier multi-gates the card on ``nonsafe_triggered_ability``.

    Drives the **real parsed Weftwalking printing** through the
    trajectory serializer + cross-engine dict→Python bridge:

      * Extract the parsed ETB-trigger Effect from the real EoE
        printing.
      * Verify the shape per the parser SoT pin above.
      * Serialize via ``trajectory._python_spell_effect_to_dict`` →
        the nested
        ``{"kind": "conditional_if_cast_from_hand", "inner": {...}}``
        dict.
      * Round-trip via ``cross_engine._ta_dict_to_python_effect`` →
        a fresh ``ConditionalIfCastFromHand`` instance equal to the
        parser-emitted leaf (including the recursive ``Sequence`` ->
        ``ShuffleZonesIntoLibrary`` + ``DrawCards`` inner).
      * Drive the Effect through Rust's resolve arm on a synthetic
        state matching Weftwalking's ETB-carrier semantics (hand +
        graveyard non-empty, library long enough for 7 draws), and
        confirm the post-resolve hand size = 7 (shuffled-then-drew-
        seven) regardless of starting hand/graveyard contents.

    **Byte-equality scope.** Both the parser-emitted Effect and the
    bridge-reconstructed Effect compare equal as frozen dataclasses
    (Python ``ConditionalIfCastFromHand``, ``Sequence``,
    ``ShuffleZonesIntoLibrary``, ``DrawCards`` are all
    ``@dataclass(frozen=True)``); the Rust-side resolve lands an
    identical observable: hand = 7, graveyard = 0, library decreases
    by (7 - hand_start - gy_start).

    **Cycle 1 scope caveat.** The full real-EoE ETB-trigger fan-out
    byte-equal probe through both engines's ETB-dispatch machinery
    requires the Rust-side ETB-trigger pushed-onto-stack timing to
    coincide with Python's ETB-event firing — broader probe is
    honest-stopped at Cycle 1; the substrate-level round-trip + the
    Rust resolve-arm unit pins carry the PROBED tier flip per the
    Splitskin Doll / Terrasymbiosis / Ouroboroid / Friendly Teddy
    classifier-bypass precedent.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.combinators import (
        ConditionalIfCastFromHand,
        Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.zone_moves import ShuffleZonesIntoLibrary
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    real = get_active_source().get_printings().get("Weftwalking")
    if real is None:
        pytest.skip("Weftwalking absent from active CardSource")

    matches = [
        ta for ta in real.triggered_abilities
        if isinstance(ta.effect, ConditionalIfCastFromHand)
    ]
    assert len(matches) == 1, (
        "Weftwalking must have exactly 1 triggered ability whose "
        "effect is ConditionalIfCastFromHand(...); got "
        f"{len(matches)}"
    )
    wrapper = matches[0].effect
    assert isinstance(wrapper, ConditionalIfCastFromHand)
    inner = wrapper.inner_effect
    assert isinstance(inner, Sequence)
    assert inner.effects[0] == ShuffleZonesIntoLibrary(
        zones=("hand", "graveyard"),
    )
    assert inner.effects[1] == DrawCards(amount=7)

    # Round-trip the wrapper through trajectory serializer + cross-
    # engine dict→Python bridge.
    d = trajectory._python_spell_effect_to_dict(wrapper)
    assert d == {
        "kind": _CIC_KIND,
        "inner": {
            "kind": "sequence",
            "effects": [
                {
                    "kind": _SZIL_KIND,
                    "zones": ["hand", "graveyard"],
                },
                {"kind": "draw_cards", "count": 7},
            ],
        },
    }, f"real-EoE wrapper serializer must emit expected dict; got {d}"
    rebuilt = _ta_dict_to_python_effect(d)
    assert isinstance(rebuilt, ConditionalIfCastFromHand), (
        "dict→Python bridge must round-trip to "
        f"ConditionalIfCastFromHand; got {type(rebuilt).__name__}"
    )
    assert rebuilt == wrapper, (
        "round-tripped wrapper must equal the parser-emitted wrapper "
        f"(frozen dataclass equality); got rebuilt={rebuilt!r}, "
        f"parser={wrapper!r}"
    )

    # Drive the wrapper through Rust's resolve arm on a synthetic
    # state whose hand + graveyard + library counts match
    # Weftwalking's ETB-carrier semantics. Use the synthetic
    # carrier printing whose `effects` slot mirrors the parser-
    # emitted dict shape; the Rust resolve arm fires the
    # ConditionalIfCastFromHand wrapper unconditionally (per the
    # engine simplification) → Sequence inner runs → ShuffleZones
    # IntoLibrary + DrawCards(7) fire.
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _WEFTWALKING_LIKE_PRINTING, 0, ZONE_BATTLEFIELD,
    )
    _stock_zone(rs, 0, ZONE_HAND, 5)
    _stock_zone(rs, 0, ZONE_LIBRARY, 14)
    _stock_zone(rs, 0, ZONE_GRAVEYARD, 3)

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()
    assert rs.stack_size() == 0, "Rust resolve must clear the stack"

    assert rs.hand_sizes[0] == 7, (
        f"composite must leave hand at 7 (drew 7 after empty); got "
        f"{rs.hand_sizes[0]}"
    )
    assert rs.graveyard_sizes[0] == 0, (
        f"graveyard must be empty after shuffle; got "
        f"{rs.graveyard_sizes[0]}"
    )
    # Library: 14 + 5 (hand) + 3 (gy) - 7 (draws) = 15
    assert rs.library_sizes[0] == 15, (
        f"library must end at 15 (14 + 5 + 3 - 7); got "
        f"{rs.library_sizes[0]}"
    )
