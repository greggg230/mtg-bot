"""Aura-attachment substrate — Phase 1 first-deliverable tests.

Spec: ``docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md`` §3 Phase 1.

**This slice ships the substrate ONLY:**

* :class:`mtg_bot.parity.cross_engine.CardDef` gains the aura
  representation fields (``is_aura`` / ``aura_enchant_type`` /
  ``aura_yours_only`` / ``aura_power_bonus`` / ``aura_toughness_bonus``
  / ``aura_granted_keywords``).
* :func:`mtg_bot.parity.cross_engine.build_rust_state` emits the
  matching aura keys into the registry ``printing_dict``; Rust's
  ``parse_printing`` silently ignores unknown keys so the round-trip is
  inert on the Rust side this slice (the long-pole cast-time
  enchant-target decision + aura-resolution arm + CR-704.5n SBA are
  honest-stopped — see Phase 1 §3 of the plan).
* :func:`mtg_bot.parity.cross_engine._card_def_to_python_printing`
  materializes the Python ``CardPrinting`` aura fields (including the
  ``"Aura"`` subtype + ``Enchantment - Aura`` type-line) so Python's
  ``CardPrinting.is_aura`` derivation fires correctly on a CardDef-built
  printing.
* :func:`mtg_bot.parity.trajectory._printing_is_bridge_simple_aura` is
  the §2 positive-allowlist predicate per the plan §3 Phase 1 step 7.
  It is NOT yet OR'd into :func:`_filter_deck_to_bridge_supported`'s
  5-disjunct admit chokepoint — substrate-only, no Aura reaches the
  bench this slice.
* :func:`mtg_bot.parity.trajectory._printing_to_card_def` gains the
  aura branch (callable directly for tests; unreachable from the loader
  this slice because the predicate is not wired in).

**Honest-stop sentinels (Phase 1 remaining items, NOT in this slice):**

* Rust ``parse_printing`` arm to consume the new ``is_aura`` /
  ``aura_enchant_type`` / ``aura_yours_only`` /
  ``aura_equipment_bonuses`` registry-dict keys (step 1+2 partial).
* Rust cast-time enchant-target decision over legal hosts per
  ``aura_enchant_type`` / ``aura_yours_only`` (step 3 — the long
  pole).
* Rust Aura-resolution arm that stamps ``aura.attached_to =
  Some(host_id)`` + applies the layer-7 P/T + keyword grant via an
  ``attached_to``-scan analogue of Python's ``game._effective_*`` (step
  4).
* Rust CR-704.5n SBA (bury host-less Aura) + host-leave bury (step 5).
* Bidirectional bridge arms across the new Rust state slots (step 6).
* One-line OR-in of ``_printing_is_bridge_simple_aura`` into
  :func:`_filter_deck_to_bridge_supported`'s 5-disjunct (step 7
  wire-up).

Once those land, Phase 1 closes for sub-shape S1 (Frantic Strength,
Shardmage's Rescue — both 0 EoE 17lands decks; the value is the
substrate). Phase 2 adds S2 control-changing Auras (Duskmourn's
Domination), which then unblocks Tractor Beam's static halves; Phase 3
ships the residue Aura-shape Effects (Meltstrider's Resolve, Pain for
All, Tractor Beam composition); Phase 4 ships S3/S4 (Moonlit Meditation
+ Chorale of the Void / Cracked Skull / Sporogenic Infection — each
needs additional substrate on top of attach).
"""

from __future__ import annotations

import pytest

from mtg_bot.parity.cross_engine import (
    CardDef,
    DeckSpec,
    _FOREST,
    _card_def_to_python_printing,
    build_python_state,
)
from mtg_bot.parity.trajectory import (
    _filter_deck_to_bridge_supported,
    _printing_is_bridge_simple_aura,
    _printing_to_card_def,
)


# ---------------------------------------------------------------------------
# CardDef shape: the new aura fields exist with back-compat defaults
# ---------------------------------------------------------------------------


def test_carddef_has_aura_fields_with_back_compat_defaults():
    """The CardDef dataclass MUST carry the Phase-1 aura fields and
    every one MUST default to the no-aura value so every pre-substrate
    CardDef is bit-identical to before this slice landed."""
    fields = CardDef.__dataclass_fields__
    assert "is_aura" in fields
    assert "aura_enchant_type" in fields
    assert "aura_yours_only" in fields
    assert "aura_power_bonus" in fields
    assert "aura_toughness_bonus" in fields
    assert "aura_granted_keywords" in fields
    # Back-compat defaults — a CardDef built with no aura args is
    # bit-identical to a pre-substrate CardDef.
    cd = CardDef(name="Forest", card_type="Land", mana_produced=("G",))
    assert cd.is_aura is False
    assert cd.aura_enchant_type == "creature"
    assert cd.aura_yours_only is False
    assert cd.aura_power_bonus == 0
    assert cd.aura_toughness_bonus == 0
    assert cd.aura_granted_keywords == ()


def test_carddef_is_frozen_with_aura_fields():
    """The aura fields preserve the frozen-dataclass contract — the
    same hash-discipline every other CardDef field obeys."""
    cd = CardDef(
        name="Frantic Strength",
        card_type="Enchantment",
        mana_cost_generic=1,
        mana_cost_colored=(("R", 1),),
        is_aura=True,
        aura_power_bonus=2,
        aura_toughness_bonus=2,
        aura_granted_keywords=("trample",),
    )
    # frozen=True → mutation raises
    with pytest.raises(Exception):
        cd.is_aura = False  # type: ignore[misc]
    # And the dataclass is hashable (frozen contract).
    assert hash(cd) == hash(cd)


# ---------------------------------------------------------------------------
# Python-side bridge: _card_def_to_python_printing materialization
# ---------------------------------------------------------------------------


def test_python_bridge_non_aura_carddef_is_bit_identical_pre_substrate():
    """A non-aura CardDef (``is_aura=False`` default) materializes a
    Python CardPrinting whose aura-related fields are exactly the
    Python defaults — bit-identical to a pre-substrate materialization."""
    cd = CardDef(name="Forest", card_type="Land", mana_produced=("G",))
    pr = _card_def_to_python_printing(cd)
    assert pr.is_aura is False
    assert pr.aura_enchant_type == "creature"  # Python's default
    assert pr.aura_yours_only is False  # Python's default
    assert pr.equipment_bonuses is None  # Python's default


def test_python_bridge_aura_static_pt_carddef_materializes_correctly():
    """An S1 static-P/T Aura CardDef (Frantic Strength shape) builds a
    Python CardPrinting that fires ``is_aura`` (the predicate derives
    from card_types contains Enchantment AND subtypes contains Aura per
    ``cards/base.py:1060``)."""
    cd = CardDef(
        name="Frantic Strength",
        card_type="Enchantment",
        mana_cost_generic=1,
        mana_cost_colored=(("R", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_power_bonus=2,
        aura_toughness_bonus=2,
        aura_granted_keywords=("trample",),
    )
    pr = _card_def_to_python_printing(cd)
    assert pr.is_aura is True
    assert "Enchantment" in pr.card_types
    assert "Aura" in pr.subtypes
    assert pr.aura_enchant_type == "creature"
    assert pr.aura_yours_only is False
    assert pr.equipment_bonuses == (2, 2, frozenset({"trample"}))


def test_python_bridge_aura_yours_only_round_trips():
    """The ``aura_yours_only`` flag (Shardmage's Rescue shape) round-
    trips through the bridge unchanged."""
    cd = CardDef(
        name="Shardmage's Rescue",
        card_type="Enchantment",
        mana_cost_generic=1,
        mana_cost_colored=(("W", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_yours_only=True,
        aura_power_bonus=1,
        aura_toughness_bonus=1,
        aura_granted_keywords=(),
    )
    pr = _card_def_to_python_printing(cd)
    assert pr.is_aura is True
    assert pr.aura_yours_only is True
    # Empty granted keywords + nonzero bonus -> equipment_bonuses populated.
    assert pr.equipment_bonuses == (1, 1, frozenset())


def test_python_bridge_aura_bare_no_bonus_yields_none_equipment_bonuses():
    """An Aura CardDef with zero P/T bonus and empty granted keywords
    materializes a Python CardPrinting with ``equipment_bonuses=None``
    (the no-grant shape)."""
    cd = CardDef(
        name="Bare Aura",
        card_type="Enchantment",
        is_aura=True,
        aura_enchant_type="creature",
    )
    pr = _card_def_to_python_printing(cd)
    assert pr.is_aura is True
    assert pr.equipment_bonuses is None


def test_python_bridge_aura_artifact_enchant_type():
    """An Aura that enchants ARTIFACT (Hardlight Containment shape;
    plan §1 calls out artifact-enchanting as a valid Phase-1 sub-shape
    variant) round-trips with ``aura_enchant_type='artifact'``."""
    cd = CardDef(
        name="Artifact Aura",
        card_type="Enchantment",
        mana_cost_generic=1,
        is_aura=True,
        aura_enchant_type="artifact",
        aura_power_bonus=0,
        aura_toughness_bonus=0,
    )
    pr = _card_def_to_python_printing(cd)
    assert pr.is_aura is True
    assert pr.aura_enchant_type == "artifact"


# ---------------------------------------------------------------------------
# Rust-side bridge: build_rust_state registers aura CardDef without crash
# ---------------------------------------------------------------------------


def _aura_spec(aura_card_def):
    """Build a smallest possible DeckSpec carrying a Forest + the
    given Aura CardDef. Both libraries are Forests so neither engine
    tries to draw the (uncastable in synthetic state) Aura."""
    return DeckSpec(
        card_defs=(_FOREST, aura_card_def),
        p0_library=("Forest",) * 7,
        p1_library=("Forest",) * 7,
    )


@pytest.mark.skipif(
    pytest.importorskip("importlib.util") is None,
    reason="never skip — placeholder for missing-rust skip",
)
def test_rust_side_aura_carddef_registers_without_crash():
    """The Rust registry MUST accept a CardDef carrying the new aura
    keys without raising. ``parse_printing`` silently ignores any
    unknown key (verified at ``rust/src/cards.rs:8341`` —
    ``match dict.get_item { Some => extract, None => default }`` for
    every optional field), so this slice's emission is inert on the
    Rust side until the parse arm lands in a later Phase-1 step."""
    import importlib
    if importlib.util.find_spec("mtg_bot_engine") is None:
        pytest.skip("mtg_bot_engine PyO3 extension not installed")
    from mtg_bot.parity.cross_engine import build_rust_state
    cd = CardDef(
        name="Frantic Strength",
        card_type="Enchantment",
        mana_cost_generic=1,
        mana_cost_colored=(("R", 1),),
        is_aura=True,
        aura_enchant_type="creature",
        aura_power_bonus=2,
        aura_toughness_bonus=2,
        aura_granted_keywords=("trample",),
    )
    spec = _aura_spec(cd)
    rs = build_rust_state(spec)
    # The Aura registered as the second printing (Forest is first).
    assert rs.registry_size == 2


def test_python_side_aura_carddef_builds_state_with_correct_printing():
    """``build_python_state`` materializes the Aura printing onto every
    library object built from the spec — the printing's ``is_aura``
    fires, and its equipment_bonuses tuple matches the CardDef's
    declared grant."""
    cd = CardDef(
        name="Shardmage's Rescue",
        card_type="Enchantment",
        mana_cost_generic=1,
        mana_cost_colored=(("W", 1),),
        is_aura=True,
        aura_yours_only=True,
        aura_power_bonus=1,
        aura_toughness_bonus=1,
    )
    spec = DeckSpec(
        card_defs=(_FOREST, cd),
        p0_library=("Forest", "Shardmage's Rescue", "Forest"),
        p1_library=("Forest", "Forest", "Forest"),
    )
    state = build_python_state(spec)
    aura_obj = state.players[0].library[1]
    assert aura_obj.printing.name == "Shardmage's Rescue"
    assert aura_obj.printing.is_aura is True
    assert aura_obj.printing.aura_yours_only is True
    assert aura_obj.printing.equipment_bonuses == (1, 1, frozenset())


# ---------------------------------------------------------------------------
# Loader predicate: _printing_is_bridge_simple_aura
# ---------------------------------------------------------------------------


def _live_printing(name):
    """Look up a live (parser-built) CardPrinting by name."""
    from mtg_bot.cards.source import get_active_source
    return get_active_source().get_printings()[name]


def test_predicate_declines_non_aura():
    """A non-Aura printing (any Land / Creature / Instant / etc.) is
    positively declined — the predicate is the §2 positive-allowlist
    by aura-shape."""
    forest = _live_printing("Forest")
    assert _printing_is_bridge_simple_aura(forest) is False


def test_predicate_admits_live_eoe_s1_auras_post_phase_4_s3_closure():
    """**Chorale of the Void Cycle 1 closure (2026-05-22).** The
    closure adds Chorale of the Void to the admit set (the Phase-3
    safe-trigger gate now admits its two-trigger composition
    ``ATTACKS+IsHost(subject="attacker")+
    ReanimateTargetCreatureCardFromDefendersGraveyard`` +
    ``BEGINNING_OF_END_STEP+Controlled(by=YOU)+
    SacrificeSelfUnlessVoidActive``). With Chorale, ALL THREE Aura
    S4 multi-trigger siblings (Cracked Skull, Sporogenic Infection,
    Chorale of the Void) are now bench-loader-admitted. Combined
    with the prior Phase 1/Phase 2/Phase 3/Phase 4 first
    deliverable widenings, the bench-loader admits NINE live EoE
    Auras:

      * Frantic Strength + Shardmage's Rescue (S1 — keyword + P/T),
      * Duskmourn's Domination (S2 — control + lose-abilities),
      * Tractor Beam (S2 — control + prevents_untap),
      * Pain for All (S4 — triggered, Phase 3 closure),
      * Moonlit Meditation (S3 — replacement, Phase 4 S3 closure),
      * Cracked Skull (S4 — multi-trigger composition, C2 closure),
      * Sporogenic Infection (S4 — multi-trigger composition,
        Sporogenic Infection Cycle 1 closure),
      * Chorale of the Void (S4 — multi-trigger composition,
        Chorale of the Void Cycle 1 closure).

    Meltstrider's Resolve stays declined on its
    ``EnchantedCreatureFightsTargetEffect`` trigger (separate
    follow-up slice — distinct from the Aura S4 framework).

    This pin captures the exact admit set so a future Phase widen
    has to update it explicitly (verify-first discipline)."""
    from mtg_bot.cards.source import get_active_source
    pool = get_active_source().get_printings()
    declined: list[tuple[str, str]] = []
    admitted: list[str] = []
    for name, p in pool.items():
        if not getattr(p, "is_aura", False):
            continue
        if _printing_is_bridge_simple_aura(p):
            admitted.append(name)
        else:
            declined.append((name, "declined"))
    # Chorale of the Void Cycle 1 closure admit set: S1 carriers +
    # S2 Duskmourn-shape + Tractor Beam + Pain for All +
    # Moonlit Meditation + Cracked Skull + Sporogenic Infection +
    # Chorale of the Void. If this set changes (parser drift,
    # future widen), update both this pin AND the deck-count
    # expectations in the plan.
    assert sorted(admitted) == sorted([
        "Frantic Strength",
        "Shardmage's Rescue",
        "Duskmourn's Domination",
        "Tractor Beam",
        "Pain for All",
        "Moonlit Meditation",
        "Cracked Skull",
        "Sporogenic Infection",
        "Chorale of the Void",
    ]), (
        f"Chorale of the Void Cycle 1 closure expected nine carriers "
        f"to admit; got {sorted(admitted)!r}. If this is intentional "
        f"(a later Phase widen of the predicate's admit shape), update "
        f"this pin to assert the new expected admit list explicitly."
    )
    # Sanity: the parser pool has at least the 14 Auras the scout
    # documented (round-61 N1 + round-63 N4 drift extensions).
    assert (len(declined) + len(admitted)) >= 14


def test_predicate_admits_synthetic_phase1_s1_aura():
    """A SYNTHETIC printing carrying ONLY the Phase-1 S1 admit shape
    (is_aura + safe-keyword-only granted-keywords + no triggers / no
    riders / parseable cost) IS admitted. This proves the predicate
    accepts the shape — when the Phase-1 Rust steps land + the loader
    wires the OR-in, a real S1 carrier whose ``flash`` is admitted (or
    whose static ``flash`` keyword is dropped on both sides) flips
    automatically."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    pr = CardPrinting(
        name="Synthetic S1 Aura",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("R", 1),)),
        type_line="Enchantment - Aura",
        supertypes=(),
        card_types=("Enchantment",),
        subtypes=("Aura",),
        power=None,
        toughness=None,
        produces=frozenset(),
        aura_enchant_type="creature",
        aura_yours_only=False,
        # The S1 grant: +2/+2/trample. trample IS in _SAFE_KEYWORDS.
        equipment_bonuses=(2, 2, frozenset({"trample"})),
    )
    assert _printing_is_bridge_simple_aura(pr) is True


def test_predicate_declines_unsafe_granted_keyword():
    """The granted-keyword check on ``equipment_bonuses`` declines an
    Aura whose grant carries a keyword outside :data:`_SAFE_KEYWORDS`
    — even if the Aura is otherwise rider-free."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    pr = CardPrinting(
        name="Synthetic Unsafe-grant Aura",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("R", 1),)),
        type_line="Enchantment - Aura",
        supertypes=(),
        card_types=("Enchantment",),
        subtypes=("Aura",),
        power=None,
        toughness=None,
        produces=frozenset(),
        aura_enchant_type="creature",
        # "flash" is NOT in _SAFE_KEYWORDS — this is the live-Aura
        # decline reason for Frantic Strength / Shardmage's Rescue,
        # mirrored here on a synthetic carrier so the granted-keyword
        # path is tested independently.
        equipment_bonuses=(1, 1, frozenset({"flash"})),
    )
    assert _printing_is_bridge_simple_aura(pr) is False


def test_predicate_admits_static_flash_keyword_phase_2():
    """**Phase 2 first deliverable (round-79) widen.** The static
    keyword check now admits ``flash`` (in addition to
    :data:`_SAFE_KEYWORDS`). The CardDef bridge does NOT propagate the
    source printing's ``keywords`` through to the materialized
    CardDef → CardPrinting round-trip on either engine — both Python
    and Rust materialize the Aura with empty keywords, so the
    ``flash`` rider is symmetrically dropped on BOTH bridge sides
    bit-equally (round-39 Ward / round-59 shockland symmetric-drop
    precedent). See `docs/SEARCH_LIMITATIONS.md` for the SL entry."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    pr = CardPrinting(
        name="Synthetic Static-Flash Aura",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("W", 1),)),
        type_line="Enchantment - Aura",
        supertypes=(),
        card_types=("Enchantment",),
        subtypes=("Aura",),
        power=None,
        toughness=None,
        produces=frozenset(),
        keywords=frozenset({"flash"}),
        aura_enchant_type="creature",
        equipment_bonuses=(1, 1, frozenset()),
    )
    assert _printing_is_bridge_simple_aura(pr) is True


def test_predicate_declines_truly_unsafe_static_keyword():
    """The printing's static-keyword check still declines an Aura
    whose ``keywords`` frozenset contains a keyword that is neither in
    :data:`_SAFE_KEYWORDS` nor in the Aura-specific
    :data:`_AURA_SAFE_EXTRA_KEYWORDS` (which only contains ``flash``
    today). Pins the §2 teeth on a synthetic carrier so the widen does
    NOT regress the rejection of truly unbridged keyword shapes (e.g.
    hexproof, ward, protection — none of these would round-trip
    symmetrically through the CardDef bridge)."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    pr = CardPrinting(
        name="Synthetic Static-Hexproof Aura",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("W", 1),)),
        type_line="Enchantment - Aura",
        supertypes=(),
        card_types=("Enchantment",),
        subtypes=("Aura",),
        power=None,
        toughness=None,
        produces=frozenset(),
        # "hexproof" is not in _SAFE_KEYWORDS and not in
        # _AURA_SAFE_EXTRA_KEYWORDS — must still decline.
        keywords=frozenset({"hexproof"}),
        aura_enchant_type="creature",
        equipment_bonuses=(1, 1, frozenset()),
    )
    assert _printing_is_bridge_simple_aura(pr) is False


def test_predicate_declines_aura_with_triggered_ability():
    """S4 sub-shape: an Aura with a triggered ability (Sheltered by
    Ghosts ETB, Chorale of the Void attack-trigger, ...) is positively
    declined — Phase 4 of the plan."""
    sheltered = _live_printing("Sheltered by Ghosts")
    assert sheltered.is_aura is True
    assert _printing_is_bridge_simple_aura(sheltered) is False


def test_predicate_admits_duskmourns_domination_post_phase_2_s2():
    """S2 sub-shape, Phase 2 S2 first deliverable (round-79+,
    2026-05-20): Duskmourn's Domination
    (``attached_grants_control + attached_grants_lose_all_abilities +
    equipment_bonuses=(-3,0,...)``) is now ADMITTED through the
    loader. The Rust attach arm flips the host's controller (layer 2)
    and `battlefield_has_keyword` wipes the host's keyword set
    (layer 6 honest-stop) while the Aura is attached. Tractor Beam
    (also S2-shape) stays declined separately on its triggered ETB +
    `attached_prevents_untap` rider."""
    dd = _live_printing("Duskmourn's Domination")
    assert dd.is_aura is True
    assert dd.attached_grants_control is True
    assert dd.attached_grants_lose_all_abilities is True
    assert _printing_is_bridge_simple_aura(dd) is True


def test_predicate_admits_tractor_beam_at_bench_loader_post_cast_time_alignment():
    """**Cast-time auto-resolve seat alignment (2026-05-20).**
    The Phase-2-S2-extension slice landed the full substrate for
    Tractor Beam — ``aura_prevents_host_untap`` CardDef field +
    Rust ``CardPrinting`` field + bridge round-trip + untap-step
    skip in ``state_ops_steps.rs`` + the narrow
    ``Event.ENTERS + IsSelf + TapEnchantedPermanent`` admit gate
    helpers (``_aura_triggers_are_phase2_s2_safe`` +
    ``_python_ta_to_safe_dict_aura_extension``). The bench-loader
    admit was honest-stopped pending the cross-engine cast-time
    auto-resolve seat alignment; this follow-up slice closes that
    by draining the ``EnchantTargetChoice`` synchronously after
    ``cast_aura`` returns true in ``state_ops_actions.rs`` —
    mirroring Python's ``_select_spell_targets`` inline target pick.

    The predicate now ADMITS Tractor Beam: the
    ``triggered_abilities`` gate routes through
    ``_aura_triggers_are_phase2_s2_safe``; the lone-control-grant
    decline requires `not attached_prevents_untap`; the bare
    `attached_prevents_untap` decline requires
    `not attached_grants_control`. The joint Tractor-Beam shape
    (control + prevents_untap + ENTERS-IsSelf-TapEnchantedPermanent)
    flows through every guardrail."""
    tb = _live_printing("Tractor Beam")
    assert tb.is_aura is True
    assert tb.attached_grants_control is True
    assert tb.attached_prevents_untap is True
    assert _printing_is_bridge_simple_aura(tb) is True


def test_predicate_admits_aura_with_replacement_post_phase_4_s3_closure():
    """S3 sub-shape: an Aura that carries the
    ``replaces_first_tokens_with_copies`` replacement effect (Moonlit
    Meditation, the 633-deck single-gated single-card S3 carrier per
    the plan §1) ADMITS post-Phase-4-S3-closure (2026-05-21). The Rust
    CreateToken arms now consult the controller's attached
    Moonlit-shape Aura via ``moonlit_token_substitution`` and swap
    the token's printing_id to the host's printing with the
    per-object Token-supertype override stamped — byte-equal against
    Python's ``_find_moonlit_replacement_source`` + ``_dc.replace``
    shim."""
    moonlit = _live_printing("Moonlit Meditation")
    assert moonlit.is_aura is True
    assert moonlit.replaces_first_tokens_with_copies is True
    assert _printing_is_bridge_simple_aura(moonlit) is True


# ---------------------------------------------------------------------------
# Loader wire-up: STEP 7 (round-78) — the predicate is now the 6th
# disjunct of :func:`_filter_deck_to_bridge_supported`'s admit
# chokepoint. Phase 1 closed.
# ---------------------------------------------------------------------------


def test_loader_admits_aura_simple_shape():
    """**STEP 7 wire-up pin (round-78).** The §2 positive allowlist
    predicate :func:`_printing_is_bridge_simple_aura` is now OR'd into
    :func:`_filter_deck_to_bridge_supported`'s admit chokepoint as the
    6th disjunct. A synthetic Phase-1-S1 Aura that passes the predicate
    flows end-to-end through the bench loader (filtered name preserved,
    CardDef materialized with the aura fields populated).

    Replaces the round-77 honest-stop sentinel that asserted "no Aura
    is admitted". Phase 1 STEP 7 made the substrate one-line-flippable;
    this pin confirms the flip is in place."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    pr = CardPrinting(
        name="Synthetic S1 Aura",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("R", 1),)),
        type_line="Enchantment - Aura",
        supertypes=(),
        card_types=("Enchantment",),
        subtypes=("Aura",),
        power=None,
        toughness=None,
        produces=frozenset(),
        aura_enchant_type="creature",
        aura_yours_only=False,
        equipment_bonuses=(2, 2, frozenset({"trample"})),
    )
    assert _printing_is_bridge_simple_aura(pr) is True, (
        "synthetic S1 Aura must pass the §2 allowlist predicate"
    )
    printings = {"Synthetic S1 Aura": pr}
    deck = ["Synthetic S1 Aura", "Synthetic S1 Aura"]
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck,
        printings,
        not_bridged_sink=not_bridged_sink,
    )
    # STEP 7 wire-up: the loader admits the Aura through the new
    # 6th disjunct.
    assert filtered == deck, (
        "STEP 7 OR-in must admit a predicate-passing Aura: %r" % (filtered,)
    )
    assert not_bridged_sink == []
    assert len(card_defs) == 1
    cd = card_defs[0]
    assert cd.name == "Synthetic S1 Aura"
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert cd.aura_power_bonus == 2
    assert cd.aura_toughness_bonus == 2
    assert cd.aura_granted_keywords == ("trample",)


def test_loader_admits_frantic_strength_phase_2_first_deliverable():
    """**Phase 2 first deliverable (round-79).** The keyword admit set
    now allows ``flash`` symmetrically (per the round-39 Ward /
    round-59 shockland precedent — the CardDef bridge does NOT
    propagate the source printing's ``flash`` keyword through to the
    materialized round-trip on either engine, so the rider is
    dropped bit-equal on BOTH sides). Frantic Strength is the first
    **live EoE Aura** to admit through the loader: it's the cleanest
    S1 carrier (`equipment_bonuses=(+2,+2,{trample})`, no
    `aura_yours_only`, no rider beyond `flash`). The deck-count
    payoff is 0 (Frantic Strength has 0 EoE 17lands decks per the
    plan §1 scout); the substrate's value is the **first live admit**
    of the substrate Phases 1+2 wire through end-to-end."""
    from mtg_bot.cards.source import get_active_source
    printings = get_active_source().get_printings()
    deck = ["Forest", "Forest", "Frantic Strength", "Forest"]
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck,
        printings,
        not_bridged_sink=not_bridged_sink,
    )
    assert "Frantic Strength" in filtered
    assert "Frantic Strength" not in not_bridged_sink
    # The CardDef materializes the Aura's parsed shape:
    aura_cd = [cd for cd in card_defs if cd.name == "Frantic Strength"]
    assert len(aura_cd) == 1
    cd = aura_cd[0]
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert cd.aura_yours_only is False
    assert cd.aura_power_bonus == 2
    assert cd.aura_toughness_bonus == 2
    assert cd.aura_granted_keywords == ("trample",)


def test_loader_admits_shardmages_rescue_phase_2_first_deliverable():
    """Shardmage's Rescue is the second S1 carrier — also admits in
    Phase 2 first deliverable. Carries ``flash`` (now symmetrically
    dropped) + ``aura_yours_only=True`` (independently bridge-correct
    on both engines per `actions.py:1337-1357` /
    `state_ops_cast.rs:2000-2055`'s yours_only filter); its
    conditional ``entered_this_turn``-gated hexproof grant is **not
    parsed** by the live parser at all (shared CR-incompleteness on
    BOTH bridge sides — symmetric)."""
    from mtg_bot.cards.source import get_active_source
    printings = get_active_source().get_printings()
    deck = ["Forest", "Forest", "Shardmage's Rescue", "Forest"]
    not_bridged_sink: list[str] = []
    filtered, card_defs = _filter_deck_to_bridge_supported(
        deck,
        printings,
        not_bridged_sink=not_bridged_sink,
    )
    assert "Shardmage's Rescue" in filtered
    assert "Shardmage's Rescue" not in not_bridged_sink
    aura_cd = [cd for cd in card_defs if cd.name == "Shardmage's Rescue"]
    assert len(aura_cd) == 1
    cd = aura_cd[0]
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert cd.aura_yours_only is True
    assert cd.aura_power_bonus == 1
    assert cd.aura_toughness_bonus == 1
    assert cd.aura_granted_keywords == ()


def test_loader_admits_at_least_one_live_eoe_aura_post_phase_2():
    """Sweep the live EoE pool: the Phase 2 first deliverable widens
    the admit set so at least one real EoE Aura now passes the
    predicate. Every S2/S3/S4 Aura stays positively declined by the
    §2 rider checks (control-swap, replacement, triggered abilities).

    Positive pin: at least Frantic Strength must admit. Negative pin:
    no S2/S3/S4 Aura leaks through (they each fail a separate §2
    teeth check — the substrate stays honest)."""
    from mtg_bot.cards.source import get_active_source
    printings = get_active_source().get_printings()
    admitted_auras = sorted(
        name for name, pr in printings.items()
        if getattr(pr, "is_aura", False)
        and _printing_is_bridge_simple_aura(pr)
    )
    assert "Frantic Strength" in admitted_auras, (
        "Phase 2 first deliverable MUST admit Frantic Strength as "
        "the first live EoE Aura. Admitted set: %r" % (admitted_auras,)
    )
    # Phase 2 S2 first deliverable (round-79+, 2026-05-20): Duskmourn's
    # Domination is now ADMITTED — it carries the paired
    # control-swap + lose-abilities S2 shape the new Rust attach arm
    # handles. Tractor Beam is now ALSO ADMITTED post-cast-time-auto-
    # resolve-seat alignment (2026-05-20): the substrate (round-trip
    # + untap-step skip + TapEnchantedPermanent Aura ETB bridge)
    # paired with the new apply_action-side EnchantTargetChoice
    # auto-resolve drain closes the cross-engine play-parity gap
    # for the joint control-grant + prevents-untap + ENTERS-IsSelf-
    # TapEnchantedPermanent shape. S3/S4 Auras outside that shape
    # stay positively declined by their respective §2 teeth.
    assert "Duskmourn's Domination" in admitted_auras, (
        "Phase 2 S2 expected Duskmourn's Domination to admit. "
        "Admitted set: %r" % (admitted_auras,)
    )
    assert "Tractor Beam" in admitted_auras, (
        "Cast-time auto-resolve seat alignment expected Tractor Beam "
        "to admit. Admitted set: %r" % (admitted_auras,)
    )
    # Phase 3 / S4 (Pain for All closure, round-80, 2026-05-20):
    # Pain for All now ADMITS through the loader — its two
    # triggers (``ENTERS+IsSelf+PainForAllETBEffect`` +
    # ``DEALT_DAMAGE+IsHost+PainForAllDamageEchoEffect``) are
    # Phase-3-safe; the bespoke Effect classes round-trip via
    # the Rust ``Effect::PainForAllETBEffect`` /
    # ``::PainForAllDamageEchoEffect`` parity that landed in the
    # Pain for All closure slice. Every other S4 carrier stays
    # declined (their triggers don't match the Phase-3 safe shape).
    assert "Pain for All" in admitted_auras, (
        "Phase 3 / S4 expected Pain for All to admit. "
        "Admitted set: %r" % (admitted_auras,)
    )
    # Phase 4 S3 closure (Moonlit Meditation, 2026-05-21): the
    # ``replaces_first_tokens_with_copies`` rider now admits through
    # the loader — the Rust CreateToken-family arms consult the
    # controller's attached Moonlit-shape Aura via
    # ``moonlit_token_substitution`` (rust/src/state_ops_resolve.rs)
    # and substitute the host's printing with a per-object Token-
    # supertype override.
    assert "Moonlit Meditation" in admitted_auras, (
        "Phase 4 S3 closure expected Moonlit Meditation to admit. "
        "Admitted set: %r" % (admitted_auras,)
    )
    # Cracked Skull C2 closure (2026-05-22): the two-trigger
    # composition (ENTERS+IsSelf+TargetOpponentLookHandMayDiscard +
    # DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent) is now
    # Phase-3-safe; Cracked Skull admits through the loader (first
    # Aura with a two-trigger composition to ride the bench loader
    # end-to-end).
    assert "Cracked Skull" in admitted_auras, (
        "Cracked Skull C2 closure expected Cracked Skull to admit. "
        "Admitted set: %r" % (admitted_auras,)
    )
    # Sporogenic Infection Cycle 1 closure (2026-05-22): the
    # two-trigger composition
    # (ENTERS+IsSelf+TargetOpponentSacrificesCreatureOtherThan
    # Enchanted + DEALT_DAMAGE+IsHost+DestroyEnchantedPermanent) is
    # now Phase-3-safe; Sporogenic Infection admits through the
    # loader (the second Aura with a two-trigger composition to ride
    # the bench loader end-to-end, alongside Cracked Skull).
    assert "Sporogenic Infection" in admitted_auras, (
        "Sporogenic Infection Cycle 1 closure expected Sporogenic "
        "Infection to admit. Admitted set: %r" % (admitted_auras,)
    )
    # Chorale of the Void Cycle 1 closure (2026-05-22): the
    # two-trigger composition
    # (ATTACKS+IsHost(subject="attacker")+
    # ReanimateTargetCreatureCardFromDefendersGraveyard +
    # BEGINNING_OF_END_STEP+Controlled(by=YOU)+
    # SacrificeSelfUnlessVoidActive) is now Phase-3-safe; Chorale of
    # the Void admits through the loader. All three Aura S4 siblings
    # (Cracked Skull, Sporogenic Infection, Chorale of the Void) are
    # bench-loader-admitted post this Cycle.
    assert "Chorale of the Void" in admitted_auras, (
        "Chorale of the Void Cycle 1 closure expected Chorale of the "
        "Void to admit. Admitted set: %r" % (admitted_auras,)
    )
    must_decline = {
        "Meltstrider's Resolve",     # residue triggered_abilities (non-Phase-3-safe — Fight effect)
    }
    leaked = must_decline & set(admitted_auras)
    assert leaked == set(), (
        "S4/residue Auras must stay positively declined by §2 "
        "teeth — surprise admit: %r" % (sorted(leaked),)
    )


def test_loader_still_admits_basic_land_unchanged():
    """Sanity: the existing 5-disjunct admit paths are bit-identical
    to before this slice — Phase-1 first-deliverable is substrate-
    only, no behavior change at the loader."""
    from mtg_bot.cards.source import get_active_source
    printings = get_active_source().get_printings()
    deck = ["Forest", "Forest", "Forest"]
    filtered, card_defs = _filter_deck_to_bridge_supported(deck, printings)
    assert filtered == deck
    assert len(card_defs) == 1
    assert card_defs[0].name == "Forest"


# ---------------------------------------------------------------------------
# _printing_to_card_def: the aura branch (callable in isolation, NOT
# yet reached from the loader — substrate-only)
# ---------------------------------------------------------------------------


def test_printing_to_card_def_handles_synthetic_aura():
    """The aura branch of :func:`_printing_to_card_def` produces a
    CardDef whose aura fields match the source printing — even though
    the loader never reaches this branch this slice (the predicate is
    not in the 5-disjunct), the branch is callable directly. This pin
    proves the Phase-1 wire-up flip will work cleanly: drop one
    one-liner into the 5-disjunct and a real S1 carrier flows through
    end-to-end."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    pr = CardPrinting(
        name="Synthetic S1 Aura",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("R", 1),)),
        type_line="Enchantment - Aura",
        supertypes=(),
        card_types=("Enchantment",),
        subtypes=("Aura",),
        power=None,
        toughness=None,
        produces=frozenset(),
        aura_enchant_type="creature",
        aura_yours_only=False,
        equipment_bonuses=(2, 2, frozenset({"trample"})),
    )
    assert _printing_is_bridge_simple_aura(pr) is True
    cd = _printing_to_card_def(pr)
    assert cd.is_aura is True
    assert cd.aura_enchant_type == "creature"
    assert cd.aura_yours_only is False
    assert cd.aura_power_bonus == 2
    assert cd.aura_toughness_bonus == 2
    assert cd.aura_granted_keywords == ("trample",)
    # Round-trip back through _card_def_to_python_printing reproduces
    # the printing's aura shape.
    pr2 = _card_def_to_python_printing(cd)
    assert pr2.is_aura is True
    assert pr2.aura_enchant_type == "creature"
    assert pr2.equipment_bonuses == (2, 2, frozenset({"trample"}))


def test_printing_to_card_def_handles_synthetic_aura_yours_only():
    """The aura branch round-trips ``aura_yours_only`` correctly — the
    Shardmage's Rescue shape (the second Phase-1 S1 carrier per the
    plan §1)."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    pr = CardPrinting(
        name="Synthetic Shardmage's Rescue",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("W", 1),)),
        type_line="Enchantment - Aura",
        supertypes=(),
        card_types=("Enchantment",),
        subtypes=("Aura",),
        power=None,
        toughness=None,
        produces=frozenset(),
        aura_enchant_type="creature",
        aura_yours_only=True,
        equipment_bonuses=(1, 1, frozenset()),
    )
    assert _printing_is_bridge_simple_aura(pr) is True
    cd = _printing_to_card_def(pr)
    assert cd.is_aura is True
    assert cd.aura_yours_only is True
    assert cd.aura_power_bonus == 1
    assert cd.aura_toughness_bonus == 1
    assert cd.aura_granted_keywords == ()
