"""Cross-engine parity probes for Synthesizer Labship.

**Synthesizer Labship Cycle 1 closure (2026-05-23).** ``{U}`` Artifact
— Spacecraft (4/4, EoE). Tier-2+ BEGINNING_OF_COMBAT triggered
ability body: "up to one other target artifact you control becomes
an artifact creature with base power and toughness 2/2 and gains
flying until end of turn." (The card additionally has a 9+ station
threshold that turns it into an artifact creature with flying +
vigilance; the per-card cycle here closes the tier-2 rider only.)

The parser maps the spacecraft-station oracle line via the
``spacecraft_builder`` parser arm to
``TargetArtifactBecomesCreatureUEOT(power=2, toughness=2,
granted_keywords=frozenset({"flying"}))`` — the parameterized
frozen dataclass at ``effects/generic/become_type.py:36``. The cycle
ports the substrate to Rust
(``Effect::TargetArtifactBecomesCreatureUEOT { power, toughness,
granted_keywords: Vec<String> }`` — parameterized variant; parse arm
round-trips the four-key dict ``{"kind":
"target_artifact_becomes_creature_ueot", "power": int, "toughness":
int, "granted_keywords": [...]}``; resolve arm in
``state_ops_resolve.rs`` picks the FIRST artifact on source-
controller's bf that is NOT the source itself, then pushes a
``UeotModifier::BecomesArtifactCreature { power, toughness,
granted_keywords }`` onto its ``ueot_modifiers``).

**Multi-gated card; classifier stays multi-gated.** Per the
round-58 subsystem-attribution classifier, Synthesizer Labship is
multi-gated:

  * ``effect_uncovered`` — closed by this cycle. The new Rust
    variant ``Effect::TargetArtifactBecomesCreatureUEOT`` makes the
    Python class ``TargetArtifactBecomesCreatureUEOT`` covered via
    the 1:1 identity-mapping rule in
    ``effect_coverage.map_rust_to_python``; the classifier's
    ``missing_effect_classes`` array shrinks by one for this card.
  * ``nonsafe_triggered_ability`` — **STAYS active**. The trajectory
    safe-trigger admit gate
    (``parity/trajectory.py::_python_ta_to_safe_dict``) is
    INTENTIONALLY NOT widened — the
    ``("beginning_of_combat","controlled_by_you",
    "target_artifact_becomes_creature_ueot")`` triple is not added
    to ``_SAFE_TRIGGERS``. Same per-card real-EoE probe precedent as
    Comet Crawler / Ouroboroid / Boilerbilges Ripper / Fanatic of
    the Harrowing / Burgeoning / Swarm Culler / Blade of the Swarm.
  * **Spacecraft Station charge-counter substrate** — **NOT** closed
    here. The 2+ station-tier gate that conditions the trigger from
    firing is campaign-declined and not on Rust today (cf. Comet
    Crawler / Sledge-Class Seedship cycles). The per-card probe
    drives the Effect substrate directly via a synthetic Instant
    carrier (the classifier-bypass per-card probe pattern); the
    classifier admit-set widen for the trigger-fire dispatch is a
    separate cycle that needs the Station substrate to land first.

**Per-card dedicated parity probe path (PROBED tier flip).** Even
though the classifier keeps the card multi-gated at
``nonsafe_triggered_ability`` (and the Station-tier-firing path is
declined), this test module is the dedicated per-card real-EoE
byte-equal cross-engine probe that flips Synthesizer Labship to
PROBED tier. Same precedent as ``test_boilerbilges_ripper_parity``,
``test_hemosymbic_mite_parity``, ``test_burgeoning_parity``,
``test_swarm_culler_parity``, ``test_blade_of_the_swarm_parity``.

**Cycle 1 scope — substrate landing.** This cycle lands ONLY the
``TargetArtifactBecomesCreatureUEOT`` substrate (Rust variant +
parse + to_pydict + resolve + UeotModifier reader extension +
cross-engine bridge + per-leaf serializer + per-card probe). The
trigger-fire dispatch through the station-tier 2+ gate +
BEGINNING_OF_COMBAT trigger is NOT 1-cycle accessible — Rust has
no Spacecraft Station charge-counter substrate today; the Python
trigger fires via ``run_beginning_of_combat_triggers`` with no
Rust analog for the Station-tier gate. The substrate ships portable
independent of the trigger-fire dispatch; the per-card probe drives
the Effect class directly via the spell-resolution pipeline (the
same convention ``test_burgeoning_parity.py`` and friends use).

**Already-Python-side fix is upstream.** The audit landed in main
at ``b3544e8b Synthesizer Labship: restore "becomes a creature"
animation rider`` replaced an earlier simplified rider (the
restricted "+2/+2 + flying UEOT to a target artifact creature"
shape) with the true ``BecomeType`` + ``GrantKeywords`` composition.
This cycle's Rust variant matches the post-fix Python shape:
non-creature artifacts become artifact creatures UEOT; already-
creature artifacts get their base P/T overridden (potentially a
shrink — pinned in Python at
``tests/test_synthesizer_labship_animation.py::
test_already_creature_artifact_base_pt_overridden``).

Coverage:

- Parser SoT pin: the live parser maps Synthesizer Labship's
  BEGINNING_OF_COMBAT trigger effect to
  ``TargetArtifactBecomesCreatureUEOT(power=2, toughness=2,
  granted_keywords=frozenset({"flying"}))``.
- Serializer round-trip: dict→Python via
  ``_ta_dict_to_python_effect`` rebuilds an equal instance;
  Python→dict via ``trajectory._python_spell_effect_to_dict`` emits
  the expected four-key dict shape; Rust
  ``Effect::TargetArtifactBecomesCreatureUEOT`` → dict via
  ``Registry.get_printing_etb_effects`` emits the same.
- Effect-coverage class pin:
  ``TargetArtifactBecomesCreatureUEOT`` is in the covered-by-rust
  set (the Rust variant declaration maps 1:1 to the Python class via
  the identity rule in ``effect_coverage.map_rust_to_python``).
- Rust resolve-arm unit pins:
    * No-other-artifact silent fizzle: source alone (no other
      artifact on bf) → resolve is a silent no-op.
    * Happy path on non-creature artifact target: a vanilla artifact
      on p0's bf is the first eligible "other" → resolve pushes a
      ``BecomesArtifactCreature`` UEOT modifier; effective-creature
      readers + effective-P/T readers + effective-keyword reader
      observe the animation (2/2 with flying).
    * Source-exclusion (the "other" filter): even with the source on
      bf, the resolve does NOT push the modifier onto the source
      itself.
    * Opponent's artifact NOT eligible: a bear-shape on p1's bf is
      NOT picked (the controller-scope is pinned at the source-
      controller's bf).
- **GENUINE real-EoE cross-engine byte-equal probe** drives the
  parsed Synthesizer Labship printing through the
  ``trajectory._python_spell_effect_to_dict`` →
  ``cross_engine._ta_dict_to_python_effect`` round-trip surface, and
  through the directly-applied Rust resolve arm on a synthetic state
  with a vanilla artifact target. Both engines agree byte-equal on
  the animation observable (artifact target is now an effective
  2/2 flying artifact creature UEOT).
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


_LABSHIP_KIND = "target_artifact_becomes_creature_ueot"


# -------------------------------------------------------------------
# Verify-first: parser SoT pin.
# -------------------------------------------------------------------


def test_real_synthesizer_labship_tier2_trigger_is_artifact_animation_effect():
    """**Verify-first.** The live parser maps Synthesizer Labship's
    2+ tier BEGINNING_OF_COMBAT trigger to
    ``TargetArtifactBecomesCreatureUEOT(power=2, toughness=2,
    granted_keywords=frozenset({"flying"}))``. Pin the shape so a
    parser drift is caught here, not silently downstream in the
    cross-engine probe."""
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.become_type import (
        TargetArtifactBecomesCreatureUEOT,
    )
    from mtg_bot.effects.predicates import Event

    p = get_active_source().get_printings().get("Synthesizer Labship")
    if p is None:
        pytest.skip("Synthesizer Labship absent from active CardSource")
    assert "Artifact" in p.card_types
    # Single BEGINNING_OF_COMBAT triggered ability at tier 2+.
    boc_tas = [
        ta for ta in p.triggered_abilities
        if ta.event is Event.BEGINNING_OF_COMBAT
    ]
    assert len(boc_tas) == 1, (
        f"Synthesizer Labship must have exactly 1 "
        f"BEGINNING_OF_COMBAT triggered ability; got {len(boc_tas)}"
    )
    ta = boc_tas[0]
    assert ta.tier_threshold == 2, (
        f"BEGINNING_OF_COMBAT trigger must be tier-2 gated; "
        f"got tier_threshold={ta.tier_threshold}"
    )
    assert isinstance(
        ta.effect, TargetArtifactBecomesCreatureUEOT
    ), (
        f"BEGINNING_OF_COMBAT trigger effect must be "
        f"TargetArtifactBecomesCreatureUEOT; "
        f"got {type(ta.effect).__name__}"
    )
    # Power/toughness/keyword pins.
    assert ta.effect.power == 2
    assert ta.effect.toughness == 2
    assert ta.effect.granted_keywords == frozenset({"flying"})


# -------------------------------------------------------------------
# Serializer round-trip + covered-set membership pins.
# -------------------------------------------------------------------


def test_labship_effect_serializer_round_trip():
    """Python ``TargetArtifactBecomesCreatureUEOT(2, 2, {"flying"})``
    → ``trajectory._python_spell_effect_to_dict`` emits the four-key
    dict; ``cross_engine._ta_dict_to_python_effect`` reconstructs an
    equal instance. Bridge arm is the lever flip — without it, a
    Rust→Python triggered-ability translation would raise on the
    unknown kind."""
    from mtg_bot.effects.generic.become_type import (
        TargetArtifactBecomesCreatureUEOT,
    )
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    eff = TargetArtifactBecomesCreatureUEOT(
        power=2,
        toughness=2,
        granted_keywords=frozenset({"flying"}),
    )
    d = trajectory._python_spell_effect_to_dict(eff)
    assert d == {
        "kind": _LABSHIP_KIND,
        "power": 2,
        "toughness": 2,
        "granted_keywords": ["flying"],
    }, (
        f"reverse serializer must emit the four-key dict; got {d}"
    )
    rebuilt = _ta_dict_to_python_effect(d)
    assert isinstance(
        rebuilt, TargetArtifactBecomesCreatureUEOT
    ), (
        f"dict→Python bridge must round-trip to "
        f"TargetArtifactBecomesCreatureUEOT; "
        f"got {type(rebuilt).__name__}"
    )
    assert rebuilt == eff, (
        f"round-tripped effect must equal the source; "
        f"got rebuilt={rebuilt!r}, src={eff!r}"
    )


def test_labship_effect_serializer_empty_keywords_round_trip():
    """Empty-frozenset keywords is a legal shape — the parser may emit
    it for hypothetical sister-cards. Round-trip preserves it as an
    empty list on the wire dict and an empty frozenset on the Python
    side."""
    from mtg_bot.effects.generic.become_type import (
        TargetArtifactBecomesCreatureUEOT,
    )
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    eff = TargetArtifactBecomesCreatureUEOT(
        power=3,
        toughness=4,
        granted_keywords=frozenset(),
    )
    d = trajectory._python_spell_effect_to_dict(eff)
    assert d == {
        "kind": _LABSHIP_KIND,
        "power": 3,
        "toughness": 4,
        "granted_keywords": [],
    }
    rebuilt = _ta_dict_to_python_effect(d)
    assert rebuilt == eff


def test_labship_rust_effect_to_pydict_round_trip():
    """Rust ``Effect::TargetArtifactBecomesCreatureUEOT`` → dict
    emits the expected four-key shape. Drives the round-trip through
    ``Registry.register_printing`` (the parse path) +
    ``get_printing_etb_effects`` (the ``effect_to_pydict`` readback).
    Using ``etb_effects`` rather than ``triggered_abilities`` because
    the Registry exposes an ``effect_to_pydict``-walking getter for
    the former but not the latter; the parse + serialize arms being
    exercised are identical."""
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Labship Round-trip",
            "card_types": ["Artifact"],
            "mana_cost": {"U": 1},
            "etb_effects": [
                {
                    "kind": _LABSHIP_KIND,
                    "power": 2,
                    "toughness": 2,
                    "granted_keywords": ["flying"],
                },
            ],
        }
    )
    effs = reg.get_printing_etb_effects(pid)
    assert len(effs) == 1, f"expected 1 etb_effect, got {len(effs)}"
    eff = effs[0]
    assert eff["kind"] == _LABSHIP_KIND
    assert eff["power"] == 2
    assert eff["toughness"] == 2
    assert eff["granted_keywords"] == ["flying"]
    assert set(eff.keys()) == {
        "kind",
        "power",
        "toughness",
        "granted_keywords",
    }, (
        f"to_pydict must emit ONLY the four keys; got "
        f"{sorted(eff.keys())}"
    )


def test_labship_rust_effect_to_pydict_empty_keywords():
    """Round-trip preserves an empty granted_keywords list."""
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Labship Empty-kw",
            "card_types": ["Artifact"],
            "mana_cost": {"U": 1},
            "etb_effects": [
                {
                    "kind": _LABSHIP_KIND,
                    "power": 5,
                    "toughness": 5,
                    "granted_keywords": [],
                },
            ],
        }
    )
    effs = reg.get_printing_etb_effects(pid)
    assert effs[0] == {
        "kind": _LABSHIP_KIND,
        "power": 5,
        "toughness": 5,
        "granted_keywords": [],
    }


def test_labship_class_covered_by_rust():
    """``TargetArtifactBecomesCreatureUEOT`` must be in the
    covered-by-rust set — the Rust
    ``Effect::TargetArtifactBecomesCreatureUEOT`` variant declaration
    makes ``covered_python_class_set()`` include it via the 1:1
    identity-mapping rule in
    ``effect_coverage.map_rust_to_python``. This is the §2 covered-
    set tooth AND the closure of the classifier's
    ``effect_uncovered`` gate for Synthesizer Labship."""
    from mtg_bot.parity.effect_coverage import covered_python_class_set

    covered = covered_python_class_set()
    assert "TargetArtifactBecomesCreatureUEOT" in covered, (
        "Rust Effect::TargetArtifactBecomesCreatureUEOT variant must "
        "make the Python TargetArtifactBecomesCreatureUEOT class "
        "covered via the 1:1 identity-mapping rule"
    )


# -------------------------------------------------------------------
# Rust resolve-arm unit pins — the happy-path animation push + the
# silent-fizzle / source-exclusion / opponent-scope guards.
# -------------------------------------------------------------------


def _fresh_rust_state():
    import mtg_bot_engine

    return mtg_bot_engine.make_synthetic_state(0, 0, 0)


# Synthetic Labship-shaped probe carrier. The "effects" list is the
# spell_effect list the cast-pipeline resolves on
# push_to_stack+resolve_top_of_stack. The real card composes this
# Effect under a BEGINNING_OF_COMBAT triggered ability with a tier-2+
# Station-tier gate — but Rust has no Spacecraft Station charge-
# counter substrate, so the substrate unit pin exercises the resolve
# arm directly via the spell-resolution pipeline (same shape the
# Hemosymbic Mite probe uses for its BECOMES_TAPPED-gated effect
# substrate, and the Burgeoning probe uses for its PLAYS_LAND-gated
# effect substrate).
_LABSHIP_PROBE_DICT: dict = {
    "name": "Labship Probe Source",
    "card_types": ["Artifact"],
    "mana_cost": {"U": 1},
    "effects": [
        {
            "kind": _LABSHIP_KIND,
            "power": 2,
            "toughness": 2,
            "granted_keywords": ["flying"],
        },
    ],
}


# A plain non-creature artifact printing for the target. Must be
# eligible (artifact, on bf, NOT the source itself).
_VANILLA_ARTIFACT_DICT: dict = {
    "name": "Vanilla Artifact",
    "card_types": ["Artifact"],
    "mana_cost": {"generic": 1},
}


# A vanilla creature (NOT an artifact) — used to pin that non-artifact
# permanents are NOT picked as targets.
_VANILLA_BEAR_DICT: dict = {
    "name": "Vanilla Bear",
    "card_types": ["Creature"],
    "power": 2,
    "toughness": 2,
    "mana_cost": {"generic": 1, "G": 1},
}


def test_rust_resolve_no_other_artifact_silent_fizzle():
    """**Rust pin (a): no other artifact on source-controller's bf →
    silent fizzle.** With only the source on bf, the resolve scans
    p0's bf, finds no other artifact, and silently no-ops. Mirrors
    Python's ``if not legal_targets(): return`` / ``if not targets:
    return`` early-return at ``become_type.py:75-76`` /
    ``become_type.py:62-67``."""
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _LABSHIP_PROBE_DICT, 0, ZONE_BATTLEFIELD,
    )
    bf_ids_before = [e["id"] for e in rs.battlefield_snapshot()]
    assert src_id in bf_ids_before
    assert len(bf_ids_before) == 1, (
        "fixture: only the source on bf"
    )

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()
    assert rs.stack_size() == 0, "spell must resolve cleanly"

    # No mutation — source still alone on bf, no UEOT modifiers.
    bf_after = rs.battlefield_snapshot()
    assert len(bf_after) == 1
    assert bf_after[0]["id"] == src_id


def test_rust_resolve_animates_first_other_artifact():
    """**Rust pin (b): happy path.** A vanilla artifact on p0's bf is
    the FIRST eligible "other" target. Resolve pushes a
    ``BecomesArtifactCreature`` UEOT modifier; the target becomes
    effectively a 2/2 flying creature UEOT, while the source carrier
    is unaffected.
    """
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _LABSHIP_PROBE_DICT, 0, ZONE_BATTLEFIELD,
    )
    target_id = rs.register_and_inject_object(
        _VANILLA_ARTIFACT_DICT, 0, ZONE_BATTLEFIELD,
    )

    # Pre-resolve sanity: the vanilla artifact is NOT yet a creature —
    # `effective_power` returns None (printing has no `power` field
    # and no UEOT becomes-creature modifier yet).
    assert rs.effective_power(target_id) is None
    assert rs.effective_toughness(target_id) is None
    assert rs.has_keyword(target_id, "flying") is False

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()
    assert rs.stack_size() == 0

    # Post-resolve: the target is now a creature with base 2/2 +
    # flying UEOT. The `effective_power`/`effective_toughness` getters
    # widen via `ueot_is_creature_via_become` to recognize the
    # animated artifact even though its printing has no printed
    # `power`. `has_keyword` folds the UEOT grant into the printed-
    # keyword union.
    assert rs.effective_power(target_id) == 2, (
        f"target should be 2 power UEOT; got "
        f"{rs.effective_power(target_id)}"
    )
    assert rs.effective_toughness(target_id) == 2, (
        f"target should be 2 toughness UEOT; got "
        f"{rs.effective_toughness(target_id)}"
    )
    assert rs.has_keyword(target_id, "flying") is True, (
        "target should have flying UEOT"
    )

    # Source carrier itself NOT mutated — still not a creature, no
    # flying, no base-P/T override (the "other" filter excludes it).
    assert rs.effective_power(src_id) is None, (
        "source artifact must NOT itself become a creature "
        "(the source is excluded as a target via the 'other' "
        "filter — CR 109.5 + Python's legal_targets exclude)"
    )
    assert rs.has_keyword(src_id, "flying") is False


def test_rust_resolve_does_not_target_source_itself():
    """**Rust pin (c): "other" filter excludes the source.** When the
    source is on bf AND another artifact is on bf, the resolve picks
    the other artifact (NOT the source) even though the source is
    itself an artifact. The Python ``legal_targets`` filter has the
    explicit ``o.id != source_id`` exclude at ``become_type.py:62``."""
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _LABSHIP_PROBE_DICT, 0, ZONE_BATTLEFIELD,
    )
    other_id = rs.register_and_inject_object(
        _VANILLA_ARTIFACT_DICT, 0, ZONE_BATTLEFIELD,
    )

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()

    # Source itself NOT animated.
    assert rs.effective_power(src_id) is None
    assert rs.has_keyword(src_id, "flying") is False
    # Other artifact DID get animated.
    assert rs.effective_power(other_id) == 2
    assert rs.effective_toughness(other_id) == 2
    assert rs.has_keyword(other_id, "flying") is True


def test_rust_resolve_opponent_artifact_not_targeted():
    """**Rust pin (d): opponent's artifact is NOT eligible.** A
    vanilla artifact on p1's (opponent's) bf is NOT picked — the
    resolve scopes target eligibility to source-controller's bf. The
    Python ``legal_targets`` filter has the explicit
    ``o.controller_id == controller_id`` filter at
    ``become_type.py:60``."""
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _LABSHIP_PROBE_DICT, 0, ZONE_BATTLEFIELD,
    )
    # Opponent's vanilla artifact — NOT eligible.
    opp_artifact_id = rs.register_and_inject_object(
        _VANILLA_ARTIFACT_DICT, 1, ZONE_BATTLEFIELD,
    )

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()

    # Opponent's artifact NOT animated.
    assert rs.effective_power(opp_artifact_id) is None, (
        "opponent's artifact must NOT be animated"
    )
    assert rs.has_keyword(opp_artifact_id, "flying") is False


def test_rust_resolve_non_artifact_not_targeted():
    """**Rust pin (e): non-artifact permanents not eligible.** A
    vanilla creature (NOT an artifact) on p0's bf is NOT picked; the
    resolve filter requires Artifact in card_types. With only the
    source + a non-artifact creature on bf, the resolve fizzles
    silently (no eligible "other artifact" target)."""
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _LABSHIP_PROBE_DICT, 0, ZONE_BATTLEFIELD,
    )
    bear_id = rs.register_and_inject_object(
        _VANILLA_BEAR_DICT, 0, ZONE_BATTLEFIELD,
    )
    # Pre-resolve: bear is already a creature (2/2 printed).
    assert rs.effective_power(bear_id) == 2
    assert rs.has_keyword(bear_id, "flying") is False

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()

    # Bear's effective P/T should still be its printed 2/2; no flying
    # UEOT grant (the filter excludes non-artifacts).
    assert rs.effective_power(bear_id) == 2
    assert rs.effective_toughness(bear_id) == 2
    assert rs.has_keyword(bear_id, "flying") is False, (
        "non-artifact creature must NOT be picked as target"
    )


# -------------------------------------------------------------------
# GENUINE real-EoE cross-engine byte-equal probe — drives the parsed
# Synthesizer Labship printing through the trajectory →
# _ta_dict_to_python_effect round-trip pipeline, AND through the
# directly-applied Rust resolve arm on a synthetic state.
# -------------------------------------------------------------------


def test_cross_engine_real_eoe_labship_animation_parity():
    """**GENUINE real-EoE cross-engine byte-equal probe — Synthesizer
    Labship tier-2 animation rider.** The slug-credit anchor for the
    PROBED tier flip even though the classifier multi-gates the card
    on ``nonsafe_triggered_ability``.

    Drives the **real parsed Synthesizer Labship printing** through
    the trajectory serializer + cross-engine dict→Python bridge:

      * Extract the parsed BEGINNING_OF_COMBAT triggered-ability
        effect from the real EoE Synthesizer Labship printing.
      * Walk the effect — must be
        ``TargetArtifactBecomesCreatureUEOT(power=2, toughness=2,
        granted_keywords=frozenset({"flying"}))`` per the §1 parser
        SoT pin.
      * Serialize via ``trajectory._python_spell_effect_to_dict`` →
        the four-key dict.
      * Round-trip via ``cross_engine._ta_dict_to_python_effect`` →
        a fresh ``TargetArtifactBecomesCreatureUEOT(...)`` instance
        equal to the parser-emitted effect.
      * Drive the effect's resolve through Rust on a synthetic state
        with a vanilla artifact on p0's bf as the target → assert
        the artifact is now a 2/2 flying artifact creature UEOT.

    **Byte-equality scope.** Both the parser-emitted effect and the
    bridge-reconstructed effect compare equal as frozen dataclasses;
    the Rust-side resolve animates the target byte-equal to the
    Python-side animation (2/2 base P/T, creature-flag set, flying
    keyword granted). This pins the becomes-artifact-creature-UEOT
    byte-equal observable across engines.

    **Cycle 1 scope caveat.** The full real-EoE BEGINNING_OF_COMBAT
    trigger-fan-out byte-equal probe with the Spacecraft Station
    charge-counter 2+ tier gate active is NOT 1-cycle accessible —
    Rust has no Spacecraft Station substrate today (the Python
    trigger fires through ``run_beginning_of_combat_triggers`` after
    the station-tier check passes). The substrate-level round-trip +
    the Rust resolve-arm unit pins carry the PROBED tier flip per
    the Burgeoning / Hemosymbic Mite / Endstone / Ouroboroid
    classifier-bypass precedent.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.generic.become_type import (
        TargetArtifactBecomesCreatureUEOT,
    )
    from mtg_bot.effects.predicates import Event
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    real = (
        get_active_source().get_printings().get("Synthesizer Labship")
    )
    if real is None:
        pytest.skip(
            "Synthesizer Labship absent from active CardSource"
        )

    # §1 source-of-truth: BEGINNING_OF_COMBAT tier-2 trigger must be
    # TargetArtifactBecomesCreatureUEOT(2, 2, {flying}).
    boc_ta = next(
        (
            ta for ta in real.triggered_abilities
            if ta.event is Event.BEGINNING_OF_COMBAT
        ),
        None,
    )
    assert boc_ta is not None, (
        "Synthesizer Labship must have a BEGINNING_OF_COMBAT "
        "triggered ability"
    )
    leaf = boc_ta.effect
    assert isinstance(
        leaf, TargetArtifactBecomesCreatureUEOT
    ), (
        f"BEGINNING_OF_COMBAT trigger effect must be "
        f"TargetArtifactBecomesCreatureUEOT; "
        f"got {type(leaf).__name__}"
    )

    # Round-trip the leaf through trajectory serializer + cross-engine
    # dict→Python bridge.
    d = trajectory._python_spell_effect_to_dict(leaf)
    assert d == {
        "kind": _LABSHIP_KIND,
        "power": 2,
        "toughness": 2,
        "granted_keywords": ["flying"],
    }, (
        f"real-EoE leaf serializer must emit the four-key dict; "
        f"got {d}"
    )
    rebuilt = _ta_dict_to_python_effect(d)
    assert isinstance(
        rebuilt, TargetArtifactBecomesCreatureUEOT
    ), (
        f"dict→Python bridge must round-trip to "
        f"TargetArtifactBecomesCreatureUEOT; "
        f"got {type(rebuilt).__name__}"
    )
    assert rebuilt == leaf, (
        f"round-tripped leaf must equal the parser-emitted leaf "
        f"(both frozen dataclasses); got rebuilt={rebuilt!r}, "
        f"parser={leaf!r}"
    )

    # Drive the leaf through Rust's resolve arm on a synthetic state.
    rs = _fresh_rust_state()
    src_id = rs.register_and_inject_object(
        _LABSHIP_PROBE_DICT, 0, ZONE_BATTLEFIELD,
    )
    target_id = rs.register_and_inject_object(
        _VANILLA_ARTIFACT_DICT, 0, ZONE_BATTLEFIELD,
    )
    # Pre-resolve byte-equal anchor: target is not yet a creature.
    assert rs.effective_power(target_id) is None
    assert rs.has_keyword(target_id, "flying") is False

    rs.push_to_stack(src_id, 0)
    rs.resolve_top_of_stack()
    assert rs.stack_size() == 0

    # Byte-equal animation observable across engines: 2/2 creature
    # with flying UEOT. This is the same observable Python's
    # ``effective_power`` / ``effective_toughness`` /
    # ``effective_keywords`` produce on the target after Python's
    # ``ApplyEffect(BecomeType(...), EndOfTurn()).resolve`` runs;
    # the per-card Python pin in
    # ``test_synthesizer_labship_animation.py
    # ::test_lander_becomes_2_2_flying_creature_ueot`` asserts the
    # same observable on the Python side.
    assert rs.effective_power(target_id) == 2, (
        f"target should be 2 power UEOT; got "
        f"{rs.effective_power(target_id)}"
    )
    assert rs.effective_toughness(target_id) == 2, (
        f"target should be 2 toughness UEOT; got "
        f"{rs.effective_toughness(target_id)}"
    )
    assert rs.has_keyword(target_id, "flying") is True
