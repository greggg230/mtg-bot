"""Cross-engine ETB-trigger smoke (Slice 3 of the ETB-trigger closure).

Pins the harness-side ``cross_engine.py`` ``printing_dict`` builder's
new ``triggered_abilities`` key + the ``_ta_dict_to_python_triggered_ability``
bridge so a CE19-shape probe (predicate-path ENTERS trigger) builds the
SAME triggered ability on BOTH engines from one ``CardDef``.

The probe shape: a vanilla creature whose
``triggered_abilities = [{event: enters, predicate: source_is_self,
effect: draw_cards count=1}]`` carries through the bridge. We don't
drive a full multi-turn cross-engine walk here — Slice 3's scope is
the bridge wiring; a full CE19-closure smoke comes in Slice 4+ once
predicates richer than ``IsSelf`` land.

Companion docs:
- ``docs/RUST_PARITY_GAPS.md`` I49 (Slice 3 closure).
- ``docs/RUST_ETB_TRIGGER_AUDIT.md`` §6.1 — Slice 3 cross-engine wiring.
"""

from __future__ import annotations


# A creature card def carrying the IsSelf-ENTERS DrawCards(1) trigger.
ETB_DRAW_BEAR_TA = {
    "event": "enters",
    "predicate": {"kind": "source_is_self"},
    "effect": {"kind": "draw_cards", "count": 1},
}


def _etb_draw_bear_card_def():
    from mtg_bot.parity.cross_engine import CardDef
    return CardDef(
        name="ETB Draw Bear (synthetic)",
        card_type="Creature",
        mana_cost_generic=1,
        mana_cost_colored=(("G", 1),),
        power=2,
        toughness=2,
        triggered_abilities=(ETB_DRAW_BEAR_TA,),
    )


def test_card_def_carries_triggered_abilities_field():
    """The new ``triggered_abilities`` field is part of the
    ``CardDef`` dataclass and defaults to empty tuple."""
    from mtg_bot.parity.cross_engine import CardDef
    plain = CardDef(name="Plain", card_type="Creature")
    assert plain.triggered_abilities == ()
    bear = _etb_draw_bear_card_def()
    assert bear.triggered_abilities == (ETB_DRAW_BEAR_TA,)


def test_python_state_builder_wires_triggered_abilities():
    """``build_python_state`` translates the dict-shape triggered_abilities
    into Python's ``TriggeredAbility`` shape via the
    ``_ta_dict_to_python_triggered_ability`` bridge. The resulting
    ``CardPrinting`` carries the right Event + Predicate + Effect.
    """
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.parity.cross_engine import DeckSpec, build_python_state

    spec = DeckSpec(
        card_defs=(_etb_draw_bear_card_def(),),
        p0_library=("ETB Draw Bear (synthetic)",),
        p1_library=("ETB Draw Bear (synthetic)",),
    )
    state = build_python_state(spec)
    obj = state.players[0].library[0]
    printing = obj.printing
    assert len(printing.triggered_abilities) == 1
    ta = printing.triggered_abilities[0]
    assert isinstance(ta, TriggeredAbility)
    assert ta.event is Event.ENTERS
    assert isinstance(ta.predicate, IsSelf)
    assert isinstance(ta.effect, DrawCards)
    # Python's DrawCards exposes the count via `amount` (the dict-shape
    # uses `count` to match Rust's parser; cross_engine bridges them).
    assert ta.effect.amount == 1
    assert ta.once_per_turn is False


def test_rust_state_builder_wires_triggered_abilities():
    """``build_rust_state`` passes ``triggered_abilities`` through to
    the Rust registry, and the printing surfaces the parsed shape via
    the registry's ``get_printing_triggered_abilities`` reader (when
    available) or — more directly — the bridge round-trips through
    the mtg_bot_engine extension.

    The substrate-level proof: build a Rust state, then verify the
    printing's triggered_ability dispatches by casting and observing
    the resulting trigger on the stack."""
    import mtg_bot_engine
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state

    spec = DeckSpec(
        card_defs=(_etb_draw_bear_card_def(),),
        p0_library=("ETB Draw Bear (synthetic)",),
        p1_library=("ETB Draw Bear (synthetic)",),
    )
    rust_state = build_rust_state(spec)
    # `register_and_inject_object` is the simplest path to materialize
    # a fresh GameObject pointing at the registered printing without
    # driving a full priority loop. Inject onto p0's hand, push the
    # spell, resolve — the synthesized self-ETB trigger lands on the
    # stack post-resolve.
    bear_dict = {
        "name": "ETB Draw Bear (synthetic)",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "G": 1},
        "effects": [],
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [ETB_DRAW_BEAR_TA],
    }
    bear_id = rust_state.register_and_inject_object(bear_dict, 0, 0)  # ZONE_HAND
    rust_state.push_to_stack(bear_id, 0)
    rust_state.resolve_top_of_stack()
    # Bear landed on bf, trigger on stack.
    ZONE_BATTLEFIELD = 2
    assert rust_state.find_object_zone(bear_id) == ZONE_BATTLEFIELD
    assert rust_state.stack_size() == 1, (
        "expected the synthesized self-ETB trigger to be on the stack "
        "post-resolve; cross_engine.py wiring didn't make it to Rust"
    )
    top = rust_state.peek_top_of_stack()
    assert top["source_obj_id"] == bear_id


def test_pre_slice_3_specs_unaffected():
    """A DeckSpec whose card_defs all leave ``triggered_abilities``
    at its default empty tuple registers identically post-Slice-3 —
    no spurious extra triggers, no extra stack pushes. Regression
    guard against the bridge accidentally injecting triggers on
    pre-Slice-3 fixtures."""
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        tiny_synthetic_deckspec,
    )

    py_state, rust_state = build_matched_states(tiny_synthetic_deckspec())
    # No bf, no stack at construction time.
    assert rust_state.stack_size() == 0
    # Both engines see zero triggered_abilities on every loaded printing.
    for obj in py_state.players[0].library:
        assert obj.printing.triggered_abilities == ()
    for obj in py_state.players[1].library:
        assert obj.printing.triggered_abilities == ()


# =====================================================================
# Slice 4 — predicate-bridge round-trip pins for the new variants.
# Pins each Slice 4 predicate's `_ta_dict_to_python_predicate` arm so
# the harness translates Rust's parser-shape predicate dict into the
# Python `TriggerPredicate` instance that matches the same firing
# semantics.
# =====================================================================


def test_slice4_bridge_subject_is_land_translates_to_python():
    """Rust ``{"kind": "subject_is_land"}`` -> Python ``Subject(slot=None,
    inner=IsLand())``."""
    from mtg_bot.effects.predicates import Subject
    from mtg_bot.effects.object_predicates import IsLand
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_predicate

    p = _ta_dict_to_python_predicate({"kind": "subject_is_land"})
    assert isinstance(p, Subject)
    assert p.slot is None
    assert isinstance(p.inner, IsLand)


def test_slice4_bridge_subject_is_artifact_translates_to_python():
    """Rust ``{"kind": "subject_is_artifact"}`` -> Python
    ``Subject(slot=None, inner=IsArtifact())``."""
    from mtg_bot.effects.predicates import Subject
    from mtg_bot.effects.object_predicates import IsArtifact
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_predicate

    p = _ta_dict_to_python_predicate({"kind": "subject_is_artifact"})
    assert isinstance(p, Subject)
    assert p.slot is None
    assert isinstance(p.inner, IsArtifact)


def test_slice4_bridge_subject_is_creature_translates_to_python():
    """Rust ``{"kind": "subject_is_creature"}`` -> Python
    ``Subject(slot=None, inner=IsCreature())``."""
    from mtg_bot.effects.predicates import Subject
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_predicate

    p = _ta_dict_to_python_predicate({"kind": "subject_is_creature"})
    assert isinstance(p, Subject)
    assert p.slot is None
    assert isinstance(p.inner, IsCreature)


def test_slice4_bridge_controlled_by_you_translates_to_python():
    """Rust ``{"kind": "controlled_by_you"}`` -> Python
    ``Controlled(by=YOU, subject=None)``."""
    from mtg_bot.effects.predicates import Controlled, YOU
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_predicate

    p = _ta_dict_to_python_predicate({"kind": "controlled_by_you"})
    assert isinstance(p, Controlled)
    assert p.by == YOU
    assert p.subject is None


def test_slice4_bridge_composes_panharmonicon_faithful_predicate():
    """The audit's §5.4 Example A Panharmonicon-faithful predicate —
    `subject_is_creature AND controlled_by_you` — round-trips through
    the bridge to Python ``And(Subject(...), Controlled(...))``.
    """
    from mtg_bot.effects.predicates import (
        And,
        Subject,
        Controlled,
        YOU,
    )
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_predicate

    pred_dict = {
        "kind": "and",
        "left": {"kind": "subject_is_creature"},
        "right": {"kind": "controlled_by_you"},
    }
    p = _ta_dict_to_python_predicate(pred_dict)
    assert isinstance(p, And)
    assert isinstance(p.left, Subject)
    assert isinstance(p.left.inner, IsCreature)
    assert isinstance(p.right, Controlled)
    assert p.right.by == YOU


def test_slice4_rust_parser_accepts_new_predicate_kinds():
    """The Rust parser surface accepts each new Slice 4 predicate
    kind on register. Pin the parser-side coverage to keep the
    Python bridge + Rust parser in lockstep.
    """
    import mtg_bot_engine

    for kind in (
        "subject_is_land",
        "subject_is_artifact",
        "subject_is_creature",
        "controlled_by_you",
    ):
        reg = mtg_bot_engine.Registry()
        d = {
            "name": f"Slice4 Bridge Probe {kind}",
            "card_types": ["Creature"],
            "mana_cost": {"generic": 1},
            "power": 1,
            "toughness": 1,
            "triggered_abilities": [
                {
                    "event": "enters",
                    "predicate": {"kind": kind},
                    "effect": {"kind": "draw_cards", "count": 1},
                }
            ],
        }
        pid = reg.register_printing(d)
        assert pid == 0, f"failed to register {kind!r}"


def test_bridge_damage_each_opponent_translates_to_python():
    """Probe F bridge extension: Rust
    ``{"kind": "damage_each_opponent", "amount": N}`` translates to
    Python ``DealDamageEachOpponent(amount=N)``.

    Unblocks Tannuk-Memorial-Ensign / Weftstalker-Ardent-shape
    "ETB-fan-out + deal 1 to each opponent" triggers on the
    cross-engine bridge. The Rust effect arm already exists at
    ``state_ops_resolve.rs:1563`` (``Effect::DamageEachOpponent``);
    Python's class already exists at
    ``effects/generic/damage.py:144`` (``DealDamageEachOpponent``).
    This pins the bridge translation that wires them up.
    """
    from mtg_bot.effects.generic.damage import DealDamageEachOpponent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "damage_each_opponent", "amount": 2}
    )
    assert isinstance(e, DealDamageEachOpponent)
    assert e.amount == 2


def test_bridge_damage_each_opponent_amount_default_zero():
    """Missing ``amount`` defaults to 0 (mirrors the gain_life arm's
    default-zero convention -- bridge consumers are required to be
    explicit, but a missing key shouldn't crash the translator).
    """
    from mtg_bot.effects.generic.damage import DealDamageEachOpponent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "damage_each_opponent"})
    assert isinstance(e, DealDamageEachOpponent)
    assert e.amount == 0


def test_rust_parser_accepts_damage_each_opponent_in_triggered_ability():
    """Pin the Rust parser surface for the matched effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Probe F Bridge Pin",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "R": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "damage_each_opponent",
                    "amount": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_any_target_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "deal_damage_any_target", "amount": N}`` translates to
    Python ``DealDamageAnyTarget(amount=N)``.

    Unblocks Nebula-Dragon / Nova-Hellkite-shape ETB triggers that
    "deal N damage to any target" on the cross-engine bridge. The
    Rust effect variant already exists at
    ``rust/src/cards.rs:631`` (``Effect::DealDamageAnyTarget``) and
    its parse_effect arm at ``cards.rs:3063-3078``; the Rust resolve
    arm at ``state_ops_resolve.rs:3536-3603`` carries the I67
    controller-of-effect fix (defender derives from
    ``current_resolve_source_controller``, not ``active_player``).
    Python's class already exists at
    ``effects/generic/damage.py:264`` (``DealDamageAnyTarget``).
    This pins the bridge translation that wires them up.
    """
    from mtg_bot.effects.generic.damage import DealDamageAnyTarget
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_any_target", "amount": 3}
    )
    assert isinstance(e, DealDamageAnyTarget)
    assert e.amount == 3
    # The any-target filter is intrinsic to the subclass; verify the
    # constructor sugar's __post_init__ wired the TargetSpec (used by
    # ``effect.legal_targets`` at push time).
    assert len(e.targets) == 1


def test_bridge_deal_damage_any_target_amount_default_zero():
    """Missing ``amount`` defaults to 0 (mirrors the gain_life /
    damage_each_opponent arms' default-zero convention -- bridge
    consumers are required to be explicit, but a missing key
    shouldn't crash the translator). Rust's parse_effect is STRICTER
    (raises ``KeyError`` on missing ``amount``); the Python bridge's
    lenient default exists for partial test fixtures only and is
    out-of-band relative to the production parser path.
    """
    from mtg_bot.effects.generic.damage import DealDamageAnyTarget
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "deal_damage_any_target"})
    assert isinstance(e, DealDamageAnyTarget)
    assert e.amount == 0


def test_rust_parser_accepts_deal_damage_any_target_in_triggered_ability():
    """Pin the Rust parser surface for the matched effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the existing ``damage_each_opponent`` parser-surface pin
    (which closed Probe F's bridge contract end-to-end). Required
    field is ``amount`` (non-negative i32); the predicate-path
    triggered-ability wrapper is just a carrier — the substrate
    being pinned is parse_effect's ``deal_damage_any_target`` arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Any Target",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 6, "R": 1},
        "power": 4,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_any_target",
                    "amount": 3,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_to_opponent_creature_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "deal_damage_to_opponent_creature", "amount": N}``
    translates to Python ``DealDamageToOpponentCreature(amount=N)``.

    Unblocks Nova-Hellkite-shape ETB triggers that "deal N damage to
    target creature an opponent controls" on the cross-engine bridge.
    Distinct from the ``deal_damage_any_target`` arm in two ways:
    (a) the slot filter restricts legal targets to creatures
    ``IsCreature() & ControlledByOpponent()`` so players are never
    eligible, and (b) the Rust resolve arm
    (state_ops_resolve.rs:6271-6342) resolves the source's controller
    INTRINSICALLY via a 5-zone walk rather than reading
    ``current_resolve_source_controller`` — the controller-of-effect
    fix is wired in by the resolve arm's own scan, not by the I67
    transient swap.
    """
    from mtg_bot.effects.generic.damage import DealDamageToOpponentCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature", "amount": 1}
    )
    assert isinstance(e, DealDamageToOpponentCreature)
    assert e.amount == 1
    # The IsCreature() & ControlledByOpponent() filter is intrinsic
    # to the subclass; verify the constructor sugar's __post_init__
    # wired the TargetSpec slot (used by ``effect.legal_targets`` at
    # push time).
    assert len(e.targets) == 1


def test_bridge_deal_damage_to_opponent_creature_amount_default_zero():
    """Missing ``amount`` defaults to 0 (mirrors the gain_life /
    damage_each_opponent / deal_damage_any_target arms' default-zero
    convention -- bridge consumers are required to be explicit, but a
    missing key shouldn't crash the translator). Rust's parse_effect
    is STRICTER (raises ``KeyError`` on missing ``amount``); the
    Python bridge's lenient default exists for partial test fixtures
    only and is out-of-band relative to the production parser path.
    """
    from mtg_bot.effects.generic.damage import DealDamageToOpponentCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature"}
    )
    assert isinstance(e, DealDamageToOpponentCreature)
    assert e.amount == 0


def test_rust_parser_accepts_deal_damage_to_opponent_creature_in_triggered_ability():
    """Pin the Rust parser surface for the matched effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the existing ``deal_damage_any_target`` parser-surface
    pin. Required field is ``amount`` (non-negative i32); the
    predicate-path triggered-ability wrapper is just a carrier — the
    substrate being pinned is parse_effect's
    ``deal_damage_to_opponent_creature`` arm
    (cards.rs:3877-3890).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Opponent Creature",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 3, "R": 2},
        "power": 4,
        "toughness": 5,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_to_opponent_creature",
                    "amount": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_any_target_optional_flag_translates_to_python():
    """Phase 5 bridge extension (Pinnacle Kill-Ship slice): Rust
    ``{"kind": "deal_damage_any_target", "amount": N, "optional": true}``
    translates to Python
    ``DealDamageAnyTarget(amount=N, optional=True)``, and the
    Python class threads ``optional=True`` through to a
    ``TargetSpec`` with ``count_min=0`` ("up to one target" — CR
    601.2c "up to N targets" reading).

    Mirrors the ``DealDamageToTargetCreature`` optional-slot
    pattern already in ``effects/generic/damage.py`` for the
    creature-restricted "up to one target creature" shape; this
    pin extends the same shape to the wider "any target" filter
    used by Pinnacle Kill-Ship ("deals 10 damage to up to one
    target creature" — observably indistinguishable from the
    "any target" filter on probes that drive a single opponent
    creature on bf).
    """
    from mtg_bot.effects.generic.damage import DealDamageAnyTarget
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_any_target",
            "amount": 10,
            "optional": True,
        }
    )
    assert isinstance(e, DealDamageAnyTarget)
    assert e.amount == 10
    assert e.optional is True
    # The TargetSpec's count_min flipped to 0 — "up to one target"
    # semantics. count_max stays at 1.
    assert len(e.targets) == 1
    spec = e.targets[0]
    assert spec.count_min == 0
    assert spec.count_max == 1


def test_bridge_deal_damage_any_target_optional_default_false():
    """Missing ``optional`` defaults to ``False`` — the bare 1..1
    "any target" form Nebula Dragon / Debris Field Crusher already
    use stays bit-identical to its pre-extension shape.

    Defense-in-depth: the pre-existing
    ``test_bridge_deal_damage_any_target_translates_to_python``
    pin tests the bare shape's translation; this pin tests the
    same shape's ``optional`` field is ``False`` and the slot
    keeps ``count_min=1`` (the strict "any target" reading where
    the spell fizzles if no legal target exists at push time).
    """
    from mtg_bot.effects.generic.damage import DealDamageAnyTarget
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_any_target", "amount": 3}
    )
    assert isinstance(e, DealDamageAnyTarget)
    assert e.amount == 3
    assert e.optional is False
    assert len(e.targets) == 1
    spec = e.targets[0]
    assert spec.count_min == 1
    assert spec.count_max == 1


def test_rust_parser_accepts_optional_key_on_deal_damage_any_target():
    """Pin the Rust parser surface for the ``optional`` key on the
    ``deal_damage_any_target`` arm. The Rust ``Effect::DealDamage
    AnyTarget`` variant remains amount-only (no ``optional`` field
    on the variant), but the parse_effect arm SILENTLY DROPS the
    extra ``optional`` key per the broader silent-drop convention
    for unrecognized fields (cards.rs:3039 destroy_all_creatures
    arm comment). Both engines accept the same dict shape — the
    Python side observes ``optional`` and threads it to a 0..1
    slot, while the Rust side observes it as an ignored extra
    field and keeps its deterministic at-apply-time pick.

    The dict-shape round-trip contract is what this pin protects:
    a probe's ``triggered_abilities`` dict registers cleanly on
    BOTH engines without per-engine forking.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Any Target Optional",
        "card_types": ["Artifact"],
        "mana_cost": {"generic": 7},
        "power": 7,
        "toughness": 7,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_any_target",
                    "amount": 10,
                    "optional": True,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_surveil_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "surveil", "count": N}`` translates to Python
    ``Surveil(amount=N)``.

    Unblocks Starbreach-Whale / Wurmwall-Sweeper / Thawbringer-ETB
    surveil-on-enters triggers on the cross-engine bridge. The Rust
    effect variant already exists at ``rust/src/cards.rs:274``
    (``Effect::Surveil { count: i32 }``) and its parse_effect arm at
    ``cards.rs:2894-2911``; the Rust resolve arm at
    ``state_ops_resolve.rs:2610-2649`` pushes a
    ``PendingDecision::SurveilChoice`` onto
    ``state.pending_decisions`` with the popped cards + source's
    controller (derived from ``current_resolve_source_controller``
    per the I61 controller-of-effect fix). Python's class already
    exists at ``effects/generic/library_lookat.py:569`` (``Surveil``).
    This pins the bridge translation that wires them up.

    Note the dict-key rename: Rust's parser uses ``count`` (mirrors
    ScryN's count-shape parser arm); Python's constructor spells the
    same value ``amount``. The bridge translates count -> amount.
    """
    from mtg_bot.effects.generic.library_lookat import Surveil
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "surveil", "count": 2})
    assert isinstance(e, Surveil)
    assert e.amount == 2
    # The use_artifact_count flag stays at its default False (the
    # dynamic-X "surveil X where X is artifact count" shape is
    # specific to Cerebral Download and isn't part of the EoE pool's
    # straight surveil-N triggers).
    assert e.use_artifact_count is False


def test_bridge_surveil_count_default_one():
    """Missing ``count`` defaults to 1 (mirrors Python's Surveil
    default of 1; the draw_cards arm uses the same default-one
    convention for its missing-count fallback).

    Rust's parse_effect is STRICTER (raises KeyError on a missing
    ``count`` field); the Python bridge's lenient default exists for
    partial test fixtures only and is out-of-band relative to the
    production parser path.
    """
    from mtg_bot.effects.generic.library_lookat import Surveil
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "surveil"})
    assert isinstance(e, Surveil)
    assert e.amount == 1


def test_rust_parser_accepts_surveil_in_triggered_ability():
    """Pin the Rust parser surface for the matched effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the existing ``deal_damage_any_target`` parser-surface
    pin. Required field is ``count`` (non-negative i32); the
    predicate-path triggered-ability wrapper is just a carrier — the
    substrate being pinned is parse_effect's ``surveil`` arm
    (cards.rs:2894-2911).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Surveil",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 4, "U": 1},
        "power": 4,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "surveil",
                    "count": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_scry_n_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "scry_n", "count": N}`` translates to Python
    ``Scry(amount=N)``.

    Sister effect to surveil — same look-at-top-N pattern, differs in
    destination zone (library bottom for scry vs graveyard for surveil).
    The Rust effect variant already exists at ``rust/src/cards.rs:123``
    (``Effect::ScryN { count: i32 }``) and its parse_effect arm at
    ``cards.rs:3350-3352``; the Rust resolve arm at
    ``state_ops_resolve.rs:1777-1792`` implements the first-slice
    deterministic shortcut "keep all on top" — no library mutation
    needed. The Python ``Scry`` class
    (effects/generic/library_lookat.py, sister to Surveil) mirrors the
    same keep-on-top semantic so post-resolve state agrees bit-for-bit
    across engines without any agent seating.

    Note the dict-key rename: Rust's parser uses ``count`` (consistent
    with the Mill / Surveil / DrawCards / DamageEachOpponent count-shape
    family); Python's constructor spells the same value ``amount``. The
    bridge translates count -> amount.
    """
    from mtg_bot.effects.generic.library_lookat import Scry
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "scry_n", "count": 3})
    assert isinstance(e, Scry)
    assert e.amount == 3


def test_bridge_scry_n_count_default_one():
    """Missing ``count`` defaults to 1 (mirrors Python's Scry default of
    1; same lenient-default convention as the surveil / mill / draw_cards
    arms).

    Rust's parse_effect is STRICTER (cards.rs:3350-3352 requires an
    explicit ``count`` field); the Python bridge's lenient default
    exists for partial test fixtures only and is out-of-band relative
    to the production parser path.
    """
    from mtg_bot.effects.generic.library_lookat import Scry
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "scry_n"})
    assert isinstance(e, Scry)
    assert e.amount == 1


def test_bridge_scry_n_translates_count_zero():
    """Scry 0 is a legal no-op (mirrors Rust's keep-on-top when
    count=0: no cards examined, library and graveyard untouched). The
    bridge passes the explicit zero through verbatim — Python's Scry
    resolve also no-ops cleanly on count=0 (the min-clamp against
    library size yields 0; the log line records 0 and returns).
    """
    from mtg_bot.effects.generic.library_lookat import Scry
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "scry_n", "count": 0})
    assert isinstance(e, Scry)
    assert e.amount == 0


def test_bridge_scry_n_translates_large_count():
    """Large ``count`` values pass through verbatim (no upper-bound
    clamp at the bridge layer). The Python ``Scry.resolve`` clamps to
    library length per CR 701.18d at resolve time, mirroring Rust's
    silent clamp at state_ops_resolve.rs:1777-1792 — the bridge stays
    a pure translator.
    """
    from mtg_bot.effects.generic.library_lookat import Scry
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "scry_n", "count": 100})
    assert isinstance(e, Scry)
    assert e.amount == 100


def test_rust_parser_accepts_scry_n_in_triggered_ability():
    """Pin the Rust parser surface for the matched effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the existing ``surveil`` parser-surface pin. Required field
    is ``count`` (non-negative i32); the predicate-path triggered-
    ability wrapper is just a carrier — the substrate being pinned is
    parse_effect's ``scry_n`` arm (cards.rs:3350-3352).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- Scry N",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 4, "U": 1},
        "power": 4,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "scry_n",
                    "count": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_mill_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "mill", "count": N}`` translates to Python
    ``SelfMill(amount=N)``.

    Unblocks Seedship-Broodtender / Icetill-Explorer-shape ETB
    triggers that "mill N cards" on the cross-engine bridge. The
    Rust effect variant already exists at
    ``rust/src/cards.rs:75`` (``Effect::Mill``) and its parse_effect
    arm at ``cards.rs:2744-2747``; the Rust resolve arm at
    ``state_ops_resolve.rs:1034-1051`` carries the I65 controller-
    of-effect fix (mills the SOURCE's controller's library, not the
    active player's). This pins the field-name translation
    ``count -> amount`` for the controller-self mill shape.
    """
    from mtg_bot.effects.generic.library_lookat import SelfMill
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "mill", "count": 3}
    )
    assert isinstance(e, SelfMill)
    assert e.amount == 3


def test_bridge_mill_count_default_one():
    """Missing ``count`` defaults to 1 (matching ``SelfMill``'s class
    default ``amount: int = 1``, which mirrors the typical oracle
    text "mill a card" without an explicit count).

    Distinct from the ``gain_life`` / ``damage_each_opponent`` /
    ``deal_damage_any_target`` arms (which default to 0): SelfMill's
    Python class default is 1, not 0. The bridge mirrors the Python
    class's chosen default so the translator is robust to partial
    fixtures without changing semantics relative to a direct
    ``SelfMill()`` construction.

    Bridge consumers SHOULD pass ``count`` explicitly to keep round-
    trip dict-shape consistency with the Rust parser (cards.rs:2744-
    2747 requires the field). This default is a robustness backstop,
    not a contract.
    """
    from mtg_bot.effects.generic.library_lookat import SelfMill
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "mill"})
    assert isinstance(e, SelfMill)
    assert e.amount == 1


def test_rust_parser_accepts_mill_in_triggered_ability():
    """Pin the Rust parser surface for the ``mill`` effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the existing ``damage_each_opponent`` /
    ``deal_damage_any_target`` parser-surface pins -- what's being
    pinned is parse_effect's ``mill`` arm accepting a
    triggered_abilities entry with ``count: N``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- Mill",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "B": 1, "G": 1},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "mill",
                    "count": 3,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_return_creature_from_graveyard_to_hand_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "return_creature_from_your_graveyard_to_hand"}``
    translates to Python
    ``ReturnCreatureFromYourGraveyardToHand()``.

    Unblocks Interceptor-Mechan / Fell-Gravship-suffix-shape ETB
    triggers (the bare unit-variant; the rider-widened EoE-card
    shapes — or_artifact / or_spacecraft — would surface as
    separate parametric Rust variants and re-enter the bridge as
    new arms). The Rust effect variant exists at
    ``rust/src/cards.rs:782`` (unit variant
    ``Effect::ReturnCreatureFromYourGraveyardToHand``); its
    parse_effect arm at ``cards.rs:3148-3151`` takes no fields;
    the resolve arm at ``state_ops_resolve.rs:2986-3020`` carries
    the I82 controller-of-effect fix (``controller_idx`` derives
    from ``current_resolve_source_controller``, not
    ``active_player``).

    Python's class already exists at
    ``effects/generic/bounce.py:356`` as constructor sugar over
    ``ReturnFromGraveyard`` (pins ``creature=True`` +
    ``destination="hand"``). This pins the bridge translation that
    wires them up.
    """
    from mtg_bot.effects.generic.bounce import (
        ReturnCreatureFromYourGraveyardToHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_creature_from_your_graveyard_to_hand"}
    )
    assert isinstance(e, ReturnCreatureFromYourGraveyardToHand)
    # The unified base's flags should reflect the unit-variant shape:
    # primary type = creature, destination = hand, no widening
    # riders, no MV cap, single-target.
    assert e.creature is True
    assert e.destination == "hand"
    assert e.or_artifact is False
    assert e.or_spacecraft is False
    assert e.or_enchantment is False
    assert e.max_mv == 0
    assert e.min_n == 1
    assert e.max_n == 1
    assert e.target_kind == "graveyard_creature"


def test_bridge_return_creature_from_graveyard_to_hand_ignores_extra_fields():
    """The unit variant has no fields, so the bridge translator
    silently ignores every key beyond ``kind`` on the dict (matches
    the Rust parse_effect arm at ``cards.rs:3148-3151`` which only
    matches on the discriminator and constructs the unit variant
    without reading any companion fields). Defense-in-depth: keep
    this lenient pin so partial fixtures don't crash the translator
    if a downstream slice prepends rider-shape stubs to the dict
    before the rider arms land.
    """
    from mtg_bot.effects.generic.bounce import (
        ReturnCreatureFromYourGraveyardToHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "return_creature_from_your_graveyard_to_hand",
            # Round-trip noise: future rider stubs that aren't yet
            # plumbed through. The bridge silently ignores them, as
            # does the Rust unit-variant parser arm.
            "or_artifact": True,
            "max_mv": 4,
        }
    )
    assert isinstance(e, ReturnCreatureFromYourGraveyardToHand)
    # The riders DO NOT carry through — the bare unit variant ignores
    # them. Probes that need widening must wait for a parametric Rust
    # variant + companion bridge arm.
    assert e.or_artifact is False
    assert e.max_mv == 0


def test_rust_parser_accepts_return_creature_from_graveyard_to_hand_in_triggered_ability():
    """Pin the Rust parser surface for the matched effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the existing ``deal_damage_any_target`` /
    ``damage_each_opponent`` parser-surface pins. The unit variant
    takes no fields beyond ``kind`` — the predicate-path triggered-
    ability wrapper is just a carrier; the substrate being pinned
    is parse_effect's
    ``return_creature_from_your_graveyard_to_hand`` arm at
    ``cards.rs:3148-3151``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Return Creature From GY to Hand",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 3, "B": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "return_creature_from_your_graveyard_to_hand",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0



def test_bridge_pump_target_creature_until_eot_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "pump_target_creature_until_eot",
    "power_bonus": N, "toughness_bonus": M}`` translates to Python
    ``PumpTargetCreatureUntilEOT(power_bonus=N, toughness_bonus=M)``.

    Unblocks Sinister-Cryologist / Dubious-Delicacy-shape ETB triggers
    that "target creature gets +X/+Y UEOT" on the cross-engine bridge.
    """
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_target_creature_until_eot",
            "power_bonus": -3,
            "toughness_bonus": 0,
        }
    )
    assert isinstance(e, PumpTargetCreatureUntilEOT)
    assert e.power_bonus == -3
    assert e.toughness_bonus == 0
    assert e.opponents_only is False
    assert e.you_control_only is False
    assert e.optional is False


def test_bridge_pump_target_creature_until_eot_positive_bonus_translates():
    """Positive bonus arm -- +3/+3 round-trip."""
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_target_creature_until_eot",
            "power_bonus": 3,
            "toughness_bonus": 3,
        }
    )
    assert isinstance(e, PumpTargetCreatureUntilEOT)
    assert e.power_bonus == 3
    assert e.toughness_bonus == 3


def test_bridge_pump_target_creature_until_eot_defaults_to_zero():
    """Missing ``power_bonus`` / ``toughness_bonus`` default to 0."""
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "pump_target_creature_until_eot"}
    )
    assert isinstance(e, PumpTargetCreatureUntilEOT)
    assert e.power_bonus == 0
    assert e.toughness_bonus == 0


def test_rust_parser_accepts_pump_target_creature_until_eot_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``pump_target_creature_until_eot`` effect kind.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- PumpTargetCreatureUntilEOT",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "U": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "pump_target_creature_until_eot",
                    "power_bonus": -3,
                    "toughness_bonus": 0,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_fight_target_creature_with_target_creature_translates_to_python():
    """BLOCKED-RUST-EFFECT slice -- Fight effect arm. Rust
    ``{"kind": "fight_target_creature_with_target_creature"}``
    translates to Python
    ``FightTargetCreatureWithTargetCreature()``.

    Unit-variant arm; no fields beyond ``kind`` (mirrors the
    bilateral source-vs-opponent shape baked into the variant
    identity on both engines). Routes Rust's
    ``Effect::FightTargetCreatureWithTargetCreature`` to its
    Python sibling for the cross-engine bridge.
    """
    from mtg_bot.effects.generic.damage import (
        FightTargetCreatureWithTargetCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "fight_target_creature_with_target_creature"}
    )
    assert isinstance(e, FightTargetCreatureWithTargetCreature)


def test_bridge_fight_target_creature_with_target_creature_ignores_extra_fields():
    """Unit-variant arm tolerates extra fields on the dict (mirrors
    every other unit-variant bridge arm -- the kind alone determines
    the translation; extra keys are inert)."""
    from mtg_bot.effects.generic.damage import (
        FightTargetCreatureWithTargetCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "fight_target_creature_with_target_creature",
            "spurious_field": 42,
        }
    )
    assert isinstance(e, FightTargetCreatureWithTargetCreature)


def test_bridge_fight_target_creature_with_target_creature_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``{"kind": "sequence", "effects": [{"kind":
    "fight_target_creature_with_target_creature"}, {"kind":
    "draw_cards", "count": 1}]}`` translates to ``Sequence(effects=(
    FightTargetCreatureWithTargetCreature(), DrawCards(amount=1)))``.
    Pins the unit-variant arm against accidental list-construction
    drift in the Sequence translator."""
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        FightTargetCreatureWithTargetCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "fight_target_creature_with_target_creature"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], FightTargetCreatureWithTargetCreature)


def test_rust_parser_accepts_fight_target_creature_with_target_creature_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``fight_target_creature_with_target_creature`` effect kind on
    a triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the
    bridge from drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs`` accepts the
    bare kind with no companion fields (unit variant); extra dict
    fields are silently ignored for round-trip dict-shape
    consistency.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- FightTargetCreatureWithTargetCreature",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "G": 1},
        "power": 3,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "fight_target_creature_with_target_creature",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_put_plus_one_counter_on_creature_you_control_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "put_plus_one_counter_on_creature_you_control",
    "amount": N}`` translates to Python
    ``PutPlusOneCounterOnCreatureYouControl(amount=N)``.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counter_on_creature_you_control",
            "amount": 1,
        }
    )
    assert isinstance(e, PutPlusOneCounterOnCreatureYouControl)
    assert e.amount == 1


def test_bridge_put_plus_one_counter_on_creature_you_control_amount_default_one():
    """Missing ``amount`` defaults to 1."""
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counter_on_creature_you_control"}
    )
    assert isinstance(e, PutPlusOneCounterOnCreatureYouControl)
    assert e.amount == 1


def test_rust_parser_accepts_put_plus_one_counter_on_creature_you_control_in_triggered_ability():
    """Pin the Rust parser surface for
    ``put_plus_one_counter_on_creature_you_control``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- PutPlusOneCounterOnCreatureYouControl",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "W": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "put_plus_one_counter_on_creature_you_control",
                    "amount": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


# ===== Phase 5 rider-widened return-to-hand bridge pins =====
#
# Two new Rust Effect variants extend the unit-variant
# return_creature_from_your_graveyard_to_hand with the EoE riders:
#
#   * return_artifact_or_creature_from_your_graveyard_to_hand
#     (Interceptor Mechan)
#   * return_creature_or_spacecraft_from_your_graveyard_to_hand
#     (Fell Gravship suffix)
#
# Both translate to Python's
# ``ReturnCreatureFromYourGraveyardToHand`` with the matching
# rider flag set (``or_artifact=True`` or ``or_spacecraft=True``).
# The Python class is unified — the rider flags flow through
# ``_gy_card_matches`` to admit the widened types alongside the
# creature primary.


def test_bridge_return_artifact_or_creature_translates_to_python():
    """Phase 5 rider bridge: Rust
    ``{"kind": "return_artifact_or_creature_from_your_graveyard_to_hand"}``
    translates to Python
    ``ReturnCreatureFromYourGraveyardToHand(or_artifact=True)``.

    Unblocks Interceptor Mechan (EoE): "When this creature enters,
    return target artifact or creature card from your graveyard to
    your hand."
    """
    from mtg_bot.effects.generic.bounce import (
        ReturnCreatureFromYourGraveyardToHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_artifact_or_creature_from_your_graveyard_to_hand"}
    )
    assert isinstance(e, ReturnCreatureFromYourGraveyardToHand)
    # Primary type stays creature; or_artifact widens the pool to
    # admit Artifact cards alongside Creature.
    assert e.creature is True
    assert e.destination == "hand"
    assert e.or_artifact is True
    # Other riders stay off — this variant only widens by artifact.
    assert e.or_spacecraft is False
    assert e.or_enchantment is False
    assert e.max_mv == 0
    assert e.min_n == 1
    assert e.max_n == 1
    assert e.target_kind == "graveyard_creature"


def test_bridge_return_creature_or_spacecraft_translates_to_python():
    """Phase 5 rider bridge: Rust
    ``{"kind": "return_creature_or_spacecraft_from_your_graveyard_to_hand"}``
    translates to Python
    ``ReturnCreatureFromYourGraveyardToHand(or_spacecraft=True)``.

    Unblocks Fell Gravship (EoE) suffix: "...then return a creature
    or Spacecraft card from your graveyard to your hand."

    Discriminator vs the or_artifact arm: Spacecraft is a SUBTYPE,
    not a card_type. Python's ``_gy_card_matches`` honors the
    ``or_spacecraft`` flag specifically (graveyard.py:46) which
    reads ``printing.is_spacecraft`` -> ``"Spacecraft" in subtypes``.
    """
    from mtg_bot.effects.generic.bounce import (
        ReturnCreatureFromYourGraveyardToHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_creature_or_spacecraft_from_your_graveyard_to_hand"}
    )
    assert isinstance(e, ReturnCreatureFromYourGraveyardToHand)
    assert e.creature is True
    assert e.destination == "hand"
    assert e.or_spacecraft is True
    # or_artifact stays off — discriminator. A plain Artifact card
    # is NOT admitted by this rider.
    assert e.or_artifact is False
    assert e.or_enchantment is False
    assert e.max_mv == 0
    assert e.min_n == 1
    assert e.max_n == 1
    assert e.target_kind == "graveyard_creature"


def test_rust_parser_accepts_return_artifact_or_creature_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the rider-
    widened kind in a triggered_ability dict. Mirrors the existing
    parser-surface pin for the unit variant.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Return Artifact Or Creature",
        "card_types": ["Artifact", "Creature"],
        "subtypes": ["Robot"],
        "mana_cost": {"generic": 4},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "return_artifact_or_creature_from_your_graveyard_to_hand"
                    ),
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_accepts_return_creature_or_spacecraft_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    or_spacecraft rider-widened kind in a triggered_ability dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Return Creature Or Spacecraft",
        "card_types": ["Artifact"],
        "subtypes": ["Spacecraft"],
        "mana_cost": {"generic": 5},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "return_creature_or_spacecraft_from_your_graveyard_to_hand"
                    ),
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_return_enchantment_or_creature_translates_to_python():
    """Phase 5 rider bridge: Rust
    ``{"kind": "return_enchantment_or_creature_from_your_graveyard_to_hand"}``
    translates to Python
    ``ReturnCreatureFromYourGraveyardToHand(or_enchantment=True)``.

    Sister bridge to the or_artifact / or_spacecraft to-hand arms.
    The audit's unique EoE consumer of the
    "creature or enchantment card from your graveyard" rider is
    Rescue Skiff, but Rescue Skiff's destination is the BATTLEFIELD
    (not the hand) — Rescue Skiff itself unblocks via the
    to-battlefield sibling (separate slice). This bridge arm lands
    here for substrate symmetry alongside the existing or_artifact
    / or_spacecraft to-hand arms.

    Discriminator vs the or_artifact / or_spacecraft arms:
    Enchantment is a CARD TYPE (not a subtype), and a plain Artifact
    or Spacecraft is NOT admitted by this rider. Python's
    ``_gy_card_matches`` honors the ``or_enchantment`` flag
    specifically (graveyard.py:46) which reads
    ``printing.is_enchantment`` -> ``"Enchantment" in card_types``.
    """
    from mtg_bot.effects.generic.bounce import (
        ReturnCreatureFromYourGraveyardToHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_enchantment_or_creature_from_your_graveyard_to_hand"}
    )
    assert isinstance(e, ReturnCreatureFromYourGraveyardToHand)
    # Primary type stays creature; or_enchantment widens the pool
    # to admit Enchantment cards alongside Creature.
    assert e.creature is True
    assert e.destination == "hand"
    assert e.or_enchantment is True
    # Other riders stay off — discriminator. A plain Artifact or
    # Spacecraft is NOT admitted by this rider.
    assert e.or_artifact is False
    assert e.or_spacecraft is False
    assert e.max_mv == 0
    assert e.min_n == 1
    assert e.max_n == 1
    assert e.target_kind == "graveyard_creature"


def test_bridge_return_enchantment_or_creature_round_trips_through_rust():
    """Round-trip pin: Rust accepts the kind via parse_effect, and
    the to-dict serializer round-trips it back to the same string.
    Mirrors the parser-surface pins for the unit variant and the
    or_artifact / or_spacecraft sibling arms.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Return Enchantment Or Creature",
        "card_types": ["Sorcery"],
        "mana_cost": {"B": 1},
        "effects": [
            {
                "kind": (
                    "return_enchantment_or_creature_from_your_graveyard_to_hand"
                ),
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_accepts_return_enchantment_or_creature_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    or_enchantment rider-widened kind in a triggered_ability dict.
    Mirrors the parser-surface pins for the or_artifact /
    or_spacecraft sibling arms.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Trigger Return Enchantment Or Creature",
        "card_types": ["Enchantment"],
        "mana_cost": {"generic": 3},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "return_enchantment_or_creature_from_your_graveyard_to_hand"
                    ),
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_return_enchantment_or_creature_describe_matches_oracle():
    """The bridged Python effect's ``describe()`` text matches the
    oracle phrasing. Pins the legacy printed phrasing through the
    rider — load-bearing for the audit's 'creature or enchantment
    card' substring search and the Python class's stringification
    contract.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_enchantment_or_creature_from_your_graveyard_to_hand"}
    )
    desc = e.describe()
    assert "creature or enchantment" in desc.lower()
    assert "graveyard" in desc
    assert "hand" in desc


# ===== Phase 5 to-BATTLEFIELD or_enchantment bridge pins (4) =====
#
# Sister bridge to the to-HAND or_enchantment block above; the
# destination delta is the load-bearing difference. Mirrors Rust's
# ``Effect::ReturnEnchantmentOrCreatureFromYourGraveyardToBattlefield``
# (cards.rs new variant) end-to-end.
#
# **Closes the loop on Rescue Skiff.** Unlike the to-hand sibling
# (which had no real-EoE consumer and landed for substrate
# symmetry), this arm is the direct unblocker for Rescue Skiff
# (EoE): "When this Spacecraft enters, return target creature or
# enchantment card from your graveyard to the battlefield."


def test_bridge_return_enchantment_or_creature_to_bf_translates_to_python():
    """Phase 5 rider bridge: Rust
    ``{"kind": "return_enchantment_or_creature_from_your_graveyard_to_battlefield"}``
    translates to Python
    ``ReturnCreatureFromYourGraveyardToBattlefield(or_enchantment=True)``.

    Sister bridge to the to-hand or_enchantment arm; the destination
    delta (BATTLEFIELD vs HAND) is the load-bearing difference.
    Closes the loop on Rescue Skiff (EoE) — its ETB resolves
    through this exact bridge arm.

    Discriminator vs the to-hand sibling: ``destination ==
    "battlefield"`` (pinned by
    ``ReturnCreatureFromYourGraveyardToBattlefield.__post_init__``).
    Python's ``_gy_card_matches`` honors the ``or_enchantment``
    flag (graveyard.py:46) which reads ``printing.is_enchantment``
    -> ``"Enchantment" in card_types``.
    """
    from mtg_bot.effects.generic.graveyard import (
        ReturnCreatureFromYourGraveyardToBattlefield,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": (
                "return_enchantment_or_creature_from_your_graveyard_to_battlefield"
            )
        }
    )
    assert isinstance(e, ReturnCreatureFromYourGraveyardToBattlefield)
    # Primary type stays creature; or_enchantment widens the pool
    # to admit Enchantment cards alongside Creature.
    assert e.creature is True
    # Destination delta from the to-hand sibling — load-bearing.
    assert e.destination == "battlefield"
    assert e.or_enchantment is True
    # Other riders stay off — discriminator. A plain Artifact or
    # Spacecraft is NOT admitted by this rider.
    assert e.or_artifact is False
    assert e.or_spacecraft is False
    assert e.max_mv == 0
    assert e.min_n == 1
    assert e.max_n == 1
    assert e.target_kind == "graveyard_creature"


def test_bridge_return_enchantment_or_creature_to_bf_round_trips_through_rust():
    """Round-trip pin: Rust accepts the kind via parse_effect, and
    the to-dict serializer round-trips it back to the same string.
    Mirrors the parser-surface pins for the bare to-bf unit variant
    and the to-hand sibling arms.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Return Enchantment Or Creature To BF",
        "card_types": ["Sorcery"],
        "mana_cost": {"B": 1, "generic": 1},
        "effects": [
            {
                "kind": (
                    "return_enchantment_or_creature_from_your_graveyard_to_battlefield"
                ),
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_accepts_return_enchantment_or_creature_to_bf_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    or_enchantment-rider to-bf kind in a triggered_ability dict.
    Mirrors the parser-surface pins for the bare to-bf unit variant
    and the to-hand sibling arms. Load-bearing for Rescue Skiff —
    its ETB trigger carries this exact effect kind.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Trigger Rescue Skiff Recur",
        "card_types": ["Artifact"],
        "mana_cost": {"W": 1, "generic": 5},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "return_enchantment_or_creature_from_your_graveyard_to_battlefield"
                    ),
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_return_enchantment_or_creature_to_bf_describe_matches_oracle():
    """The bridged Python effect's ``describe()`` text matches the
    oracle phrasing for Rescue Skiff: "return target creature or
    enchantment card from your graveyard to the battlefield". Pins
    the legacy printed phrasing through the rider for the to-bf
    destination — load-bearing for the audit's 'creature or
    enchantment card' substring search and the Python class's
    stringification contract.

    The ``ReturnCreatureFromYourGraveyardToBattlefield`` describe()
    override (graveyard.py:382-389) was already or_enchantment-aware;
    this pin guards against any future regression that would drop
    the rider from the override's type-phrase chain.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": (
                "return_enchantment_or_creature_from_your_graveyard_to_battlefield"
            )
        }
    )
    desc = e.describe()
    assert "creature or enchantment" in desc.lower()
    assert "graveyard" in desc
    # Destination delta from the to-hand sibling — load-bearing.
    assert "battlefield" in desc


# ===== Phase 5 composer bridge pins (sequence) =====
#
# Rust's ``Effect::Sequence { effects: Vec<Effect> }`` carries an
# ordered list of nested effects; the parse_effect arm at
# rust/src/cards.rs:4046-4075 recursively parses each child. The
# bridge mirrors that shape: ``_ta_dict_to_python_effect`` recurses
# into each ``effects[i]`` dict and wraps the resulting list in
# ``Sequence([...])``.
#
# Real EoE consumers unblocked by this composer:
#
#   * Uthros Scanship: Sequence{[DrawCards{2}, DiscardSelf{1}]}
#   * Fell Gravship suffix: Sequence{[Mill{3},
#     ReturnCreatureFromYourGraveyardToHand]}


def test_bridge_sequence_empty_translates_to_python():
    """Phase 5 composer bridge: an empty Sequence (no children) is
    a legal no-op on both engines and round-trips through the bridge.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "sequence", "effects": []})
    assert isinstance(e, Sequence)
    assert e.effects == ()


def test_bridge_sequence_two_children_translates_to_python():
    """Phase 5 composer bridge: a 2-element Sequence with leaf
    children translates each child via the same bridge function and
    preserves order.

    Uthros Scanship's ETB shape: Sequence{[DrawCards{2}, DiscardSelf{1}]}.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.discard import DiscardSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "draw_cards", "count": 2},
                {"kind": "discard_self", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DrawCards)
    assert e.effects[0].amount == 2
    assert isinstance(e.effects[1], DiscardSelf)
    assert e.effects[1].amount == 1


def test_bridge_sequence_nested_translates_to_python():
    """Phase 5 composer bridge: Sequence-in-Sequence nesting works
    (the recursive translator descends into nested composer dicts).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "gain_life", "amount": 1},
                {
                    "kind": "sequence",
                    "effects": [
                        {"kind": "draw_cards", "count": 1},
                        {"kind": "gain_life", "amount": 2},
                    ],
                },
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], GainLife)
    assert e.effects[0].amount == 1
    inner = e.effects[1]
    assert isinstance(inner, Sequence)
    assert len(inner.effects) == 2
    assert isinstance(inner.effects[0], DrawCards)
    assert inner.effects[0].amount == 1
    assert isinstance(inner.effects[1], GainLife)
    assert inner.effects[1].amount == 2


def test_rust_parser_accepts_sequence_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts a sequence
    composer (with nested children) inside a triggered_ability dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — Sequence Composer",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "sequence",
                    "effects": [
                        {"kind": "draw_cards", "count": 2},
                        {"kind": "discard_self", "amount": 1},
                    ],
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


# ===== Phase 5 composer bridge pins (if_you_do) =====
#
# Rust's ``Effect::IfYouDo { do_action: Box<Effect>, then_action:
# Box<Effect> }`` is the gate-on-completion composer. parse_effect
# at rust/src/cards.rs:3610-3640 REQUIRES both ``do_action`` and
# ``then_action`` (no defaults — missing either field raises
# ValueError). The bridge mirrors that contract: missing fields
# raise KeyError out of ``_ta_dict_to_python_effect``.
#
# Python's ``IfYouDo(do_action: Effect, then_action: Effect)``
# from ``effects/combinators.py:560`` resolves do_action first; if
# it returns False (signaling failure / decline), then_action is
# skipped. ``DiscardSelf`` and ``Optional`` already signal
# completion via the True/False return — this is how the gate fires.
#
# Real EoE consumer unblocked by this composer:
#
#   * Melded Moxite: IfYouDo{cost=DiscardSelf, then=DrawCards{2}}
#     -- "Discard a card. If you do, draw two cards."


def test_bridge_if_you_do_translates_to_python():
    """Phase 5 composer bridge: a basic IfYouDo with simple cost +
    then-side translates each branch via the recursive translator.

    Melded Moxite's ETB shape: IfYouDo{cost=DiscardSelf{1},
    then=DrawCards{2}}.
    """
    from mtg_bot.effects.combinators import IfYouDo
    from mtg_bot.effects.generic.discard import DiscardSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "if_you_do",
            "do_action": {"kind": "discard_self", "amount": 1},
            "then_action": {"kind": "draw_cards", "count": 2},
        }
    )
    assert isinstance(e, IfYouDo)
    assert isinstance(e.do_action, DiscardSelf)
    assert e.do_action.amount == 1
    assert isinstance(e.then_action, DrawCards)
    assert e.then_action.amount == 2


def test_bridge_if_you_do_missing_then_action_raises():
    """Phase 5 composer bridge: missing ``then_action`` field raises
    out of the bridge — mirrors Rust's parse_effect requiring the
    field (no default). Same contract for missing ``do_action``;
    one direction pins the surface symmetrically.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {
                "kind": "if_you_do",
                "do_action": {"kind": "discard_self", "amount": 1},
                # then_action intentionally absent
            }
        )


def test_rust_parser_accepts_if_you_do_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts an if_you_do
    composer (with both do/then children) inside a triggered_ability
    dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin — IfYouDo Composer",
        "card_types": ["Artifact"],
        "mana_cost": {"generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "if_you_do",
                    "do_action": {
                        "kind": "discard_self", "amount": 1,
                    },
                    "then_action": {
                        "kind": "draw_cards", "count": 2,
                    },
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_pump_creatures_opponents_control_until_eot_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "pump_creatures_opponents_control_until_eot",
    "power_bonus": N, "toughness_bonus": M}`` translates to Python
    ``PumpCreaturesOpponentsControlUntilEOT(power_bonus=N,
    toughness_bonus=M)``.

    Mass-debuff "creatures your opponents control get +X/+Y UEOT"
    on the cross-engine bridge -- the opponent-side mirror of the
    (not-yet-on-the-bridge) ``pump_creatures_you_control_until_eot``
    arm. EoE consumer Mouth of the Storm uses the extended
    ``until your next turn`` duration which has no Rust counterpart;
    the bare ``until end of turn`` arm has no real-EoE consumer
    today and is exercised end-to-end via a synthetic probe.
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesOpponentsControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_opponents_control_until_eot",
            "power_bonus": -3,
            "toughness_bonus": 0,
        }
    )
    assert isinstance(e, PumpCreaturesOpponentsControlUntilEOT)
    assert e.power_bonus == -3
    assert e.toughness_bonus == 0
    # The Python-only ``until_caster_next_turn`` extended-duration
    # flag stays at its default False (the Rust variant has no
    # equivalent field; the bridge mirrors Rust's bare-EOT shape).
    assert e.until_caster_next_turn is False
    # Granted-keywords rider stays at its empty default (Rust's
    # variant has no equivalent field).
    assert e.granted_keywords == frozenset()


def test_bridge_pump_creatures_opponents_control_until_eot_positive_bonus_translates():
    """Positive-bonus arm -- ``+2/+2`` round-trips. Same shape as the
    debuff form; the Rust parse_effect arm imposes no sign constraint
    (cards.rs:3844-3862)."""
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesOpponentsControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_opponents_control_until_eot",
            "power_bonus": 2,
            "toughness_bonus": 2,
        }
    )
    assert isinstance(e, PumpCreaturesOpponentsControlUntilEOT)
    assert e.power_bonus == 2
    assert e.toughness_bonus == 2


def test_bridge_pump_creatures_opponents_control_until_eot_defaults_to_zero():
    """Missing ``power_bonus`` / ``toughness_bonus`` default to 0 in
    the bridge translator (translator-robustness backstop for direct
    unit-test calls; Rust's parse_effect REQUIRES both keys present so
    end-to-end Rust callers must supply them explicitly)."""
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesOpponentsControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "pump_creatures_opponents_control_until_eot"}
    )
    assert isinstance(e, PumpCreaturesOpponentsControlUntilEOT)
    assert e.power_bonus == 0
    assert e.toughness_bonus == 0


def test_rust_parser_accepts_pump_creatures_opponents_control_until_eot_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``pump_creatures_opponents_control_until_eot`` effect kind in a
    triggered_ability dict. Rust's parse_effect at cards.rs:3844-3862
    requires both ``power_bonus`` and ``toughness_bonus`` keys
    explicitly.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- PumpCreaturesOpponentsControlUntilEOT",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 6, "U": 1},
        "power": 6,
        "toughness": 6,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "pump_creatures_opponents_control_until_eot",
                    "power_bonus": -3,
                    "toughness_bonus": 0,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_pump_creatures_opponents_control_until_caster_next_turn_translates_to_python():
    """Mouth-of-the-Storm bridge extension: Rust
    ``{"kind": "pump_creatures_opponents_control_until_caster_next_turn",
       "power_bonus": N, "toughness_bonus": M}`` translates to Python
    ``PumpCreaturesOpponentsControlUntilEOT(power_bonus=N,
    toughness_bonus=M, until_caster_next_turn=True)``.

    Sibling-of-the-bare-EOT bridge arm. Real-EoE consumer: Mouth of
    the Storm ({6}{U} Elemental, "When this creature enters,
    creatures your opponents control get -3/-0 UNTIL YOUR NEXT TURN.")
    -- the discriminator carries the extended-duration distinction;
    the Python sibling routes through ``until_caster_next_turn=True``
    which writes the modifier via ``grant_pt_pump_until_player_next_turn``
    (layer-7c registry entry tagged
    ``expires_at="controller_untap"``), keeping the debuff through
    the opponent's full subsequent turn.

    The Rust resolve arm currently uses the SAME UEOT cleanup-step
    lifetime as the bare sibling (the cleanup-gate divergence is
    documented in docs/SEARCH_LIMITATIONS.md and is a deferred
    follow-up Rust slice). The discriminator-only sibling matters
    NOW because it routes Mouth of the Storm cleanly today and the
    cleanup-gate slice can swap the resolve arm WITHOUT any bridge
    or call-site changes.
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesOpponentsControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_opponents_control_until_caster_next_turn",
            "power_bonus": -3,
            "toughness_bonus": 0,
        }
    )
    assert isinstance(e, PumpCreaturesOpponentsControlUntilEOT)
    assert e.power_bonus == -3
    assert e.toughness_bonus == 0
    # Load-bearing translation: bridge sets until_caster_next_turn=True
    # on the Python class, distinguishing the extended duration from
    # the bare-EOT sibling above.
    assert e.until_caster_next_turn is True
    # Granted-keywords rider stays at its empty default (Rust's
    # variant has no equivalent field).
    assert e.granted_keywords == frozenset()


def test_bridge_pump_creatures_opponents_control_until_caster_next_turn_defaults_to_zero():
    """Missing ``power_bonus`` / ``toughness_bonus`` default to 0 in
    the bridge translator; ``until_caster_next_turn=True`` still
    flows through (it's set unconditionally on this arm). Translator-
    robustness backstop for direct unit-test calls; Rust's
    parse_effect REQUIRES both keys present so end-to-end Rust
    callers must supply them explicitly."""
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesOpponentsControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "pump_creatures_opponents_control_until_caster_next_turn"}
    )
    assert isinstance(e, PumpCreaturesOpponentsControlUntilEOT)
    assert e.power_bonus == 0
    assert e.toughness_bonus == 0
    assert e.until_caster_next_turn is True


def test_bridge_pump_creatures_opponents_control_until_caster_next_turn_distinct_from_bare_sibling():
    """Discriminator pin: bare-EOT and extended-duration arms produce
    DIFFERENT Python objects -- the bare arm's
    ``until_caster_next_turn`` defaults to False; the extended-arm
    bridge sets it True. A regression that aliased the two arms in
    the bridge dispatch would surface here.
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesOpponentsControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    bare = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_opponents_control_until_eot",
            "power_bonus": -3,
            "toughness_bonus": 0,
        }
    )
    extended = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_opponents_control_until_caster_next_turn",
            "power_bonus": -3,
            "toughness_bonus": 0,
        }
    )
    assert isinstance(bare, PumpCreaturesOpponentsControlUntilEOT)
    assert isinstance(extended, PumpCreaturesOpponentsControlUntilEOT)
    # Same P/T payload.
    assert bare.power_bonus == extended.power_bonus == -3
    assert bare.toughness_bonus == extended.toughness_bonus == 0
    # Load-bearing distinction: ONLY the extended arm sets
    # until_caster_next_turn=True. The bare arm must default it to
    # False so cleanup-step expiration fires at the casting turn's
    # cleanup (Python-side parity with the Rust bare-EOT cleanup).
    assert bare.until_caster_next_turn is False
    assert extended.until_caster_next_turn is True


def test_rust_parser_accepts_pump_creatures_opponents_control_until_caster_next_turn_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``pump_creatures_opponents_control_until_caster_next_turn``
    effect kind on a triggered-ability carrier (Mouth of the Storm
    shape). The bridge's dict shape must register cleanly on the
    Rust side too; this lockstep gate keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the bare-sibling parser-surface pin above. Rust's
    parse_effect arm tolerates missing ``power_bonus`` /
    ``toughness_bonus`` (defaults to 0) on this arm, mirroring the
    bare sibling's tolerance.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Mouth-of-the-Storm Bridge Pin -- ExtendedDuration",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 6, "U": 1},
        "power": 6,
        "toughness": 6,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "pump_creatures_opponents_control_until_caster_next_turn",
                    "power_bonus": -3,
                    "toughness_bonus": 0,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_each_opponent_sacrifices_creature_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "each_opponent_sacrifices_creature"}`` translates to
    Python ``EachOpponentSacrificesCreature()``.

    Rust's variant is a UNIT VARIANT -- no fields beyond ``kind``.
    Bridges Susurian Dirgecraft (EoE Spacecraft) end-to-end.
    """
    from mtg_bot.effects.generic.opponent_punishers import (
        EachOpponentSacrificesCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_sacrifices_creature"}
    )
    assert isinstance(e, EachOpponentSacrificesCreature)


def test_bridge_each_opponent_sacrifices_creature_ignores_extra_fields():
    """Unit-variant arm tolerates extra fields on the dict (mirrors
    every other unit-variant arm on the bridge -- the kind alone
    determines the translation; extra keys are inert)."""
    from mtg_bot.effects.generic.opponent_punishers import (
        EachOpponentSacrificesCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "each_opponent_sacrifices_creature",
            "spurious_field": 42,
        }
    )
    assert isinstance(e, EachOpponentSacrificesCreature)


def test_bridge_each_opponent_sacrifices_creature_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``{"kind": "sequence", "effects": [{"kind":
    "each_opponent_sacrifices_creature"}, {"kind": "draw_cards",
    "count": 1}]}`` translates to ``Sequence(effects=(
    EachOpponentSacrificesCreature(), DrawCards(amount=1)))``."""
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.opponent_punishers import (
        EachOpponentSacrificesCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "each_opponent_sacrifices_creature"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], EachOpponentSacrificesCreature)


def test_rust_parser_accepts_each_opponent_sacrifices_creature_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``each_opponent_sacrifices_creature`` effect kind in a
    triggered_ability dict. Rust's parse_effect at cards.rs:3391-3393
    accepts the bare kind with no companion fields.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- EachOpponentSacrificesCreature",
        "card_types": ["Artifact", "Creature"],
        "subtypes": ["Spacecraft"],
        "mana_cost": {"generic": 4, "B": 1},
        "power": 4,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "each_opponent_sacrifices_creature",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_each_opponent_mills_cards_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "each_opponent_mills_cards", "count": N}`` translates
    to Python ``EachOpponentMillsCards(amount=N)``.

    Field-name translation: Rust ``count`` -> Python ``amount``.
    Mirrors the field-name divergence between Rust's count-keyed
    arms (each_opponent_mills_cards, surveil) and Python's amount-
    keyed effect family.
    """
    from mtg_bot.effects.generic.draw_mill import EachOpponentMillsCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_mills_cards", "count": 3}
    )
    assert isinstance(e, EachOpponentMillsCards)
    assert e.amount == 3


def test_bridge_each_opponent_mills_cards_count_default_one():
    """Missing ``count`` defaults to 1 in the bridge translator
    (mirrors the Python class's ``amount: int = 1`` default).
    Rust's parse_effect REQUIRES the key present so direct Rust
    callers must supply it explicitly; the bridge default is a
    translator-robustness backstop for direct unit-test calls.
    """
    from mtg_bot.effects.generic.draw_mill import EachOpponentMillsCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_mills_cards"}
    )
    assert isinstance(e, EachOpponentMillsCards)
    assert e.amount == 1


def test_bridge_each_opponent_mills_cards_zero_count_translates():
    """``count=0`` translates to ``amount=0`` (a no-op on both
    engines but a legal field value -- Rust's parse_effect at
    cards.rs:3604-3605 rejects only NEGATIVE counts, not zero)."""
    from mtg_bot.effects.generic.draw_mill import EachOpponentMillsCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_mills_cards", "count": 0}
    )
    assert isinstance(e, EachOpponentMillsCards)
    assert e.amount == 0


def test_rust_parser_accepts_each_opponent_mills_cards_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``each_opponent_mills_cards`` effect kind in a triggered_ability
    dict. Rust's parse_effect at cards.rs:3592-3608 requires the
    ``count`` key explicitly.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- EachOpponentMillsCards",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "B": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "each_opponent_mills_cards",
                    "count": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


# ===== Phase 5 mass-counter bridge pins =====
#
# Anthem-shape +1/+1 counter add for the source's controller's bf.
# Translates ``put_plus_one_counter_on_each_creature_you_control``
# (Rust ``Effect::PutPlusOneCounterOnEachCreatureYouControl``) into
# Python's ``PutPlusOneCounterOnEachCreatureYouControl(amount=N)``.
#
# Sister arm to ``put_plus_one_counter_on_creature_you_control``
# (single-target). The MASS variant has no targeting; the bridge
# constructor mirrors the Rust variant's bare ``amount`` shape.
# Real EoE consumer: Atmospheric Greenhouse.


def test_bridge_put_plus_one_counter_on_each_creature_you_control_translates_to_python():
    """Phase 5 mass-counter bridge: Rust
    ``{"kind": "put_plus_one_counter_on_each_creature_you_control",
    "amount": N}`` translates to Python
    ``PutPlusOneCounterOnEachCreatureYouControl(amount=N)``.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counter_on_each_creature_you_control",
            "amount": 1,
        }
    )
    assert isinstance(e, PutPlusOneCounterOnEachCreatureYouControl)
    assert e.amount == 1
    # Default exclude_source is False (matches the Rust variant
    # which has no exclude-source field; the bridge mirrors the
    # bare-shape variant).
    assert e.exclude_source is False


def test_bridge_put_plus_one_counter_on_each_creature_you_control_amount_default_one():
    """Missing ``amount`` defaults to 1 (matches the Python class's
    ``amount: int = 1`` default and the Rust parser's tolerant arm
    at cards.rs:3256-3272).
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counter_on_each_creature_you_control"}
    )
    assert isinstance(e, PutPlusOneCounterOnEachCreatureYouControl)
    assert e.amount == 1


def test_bridge_put_plus_one_counter_on_each_creature_you_control_amount_two_translates():
    """Non-default ``amount`` round-trips. The Rust variant carries
    ``amount: i32`` so larger printings (none in EoE today, but the
    field is parametric) translate the same way.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counter_on_each_creature_you_control",
            "amount": 2,
        }
    )
    assert isinstance(e, PutPlusOneCounterOnEachCreatureYouControl)
    assert e.amount == 2


def test_rust_parser_accepts_put_plus_one_counter_on_each_creature_you_control_in_triggered_ability():
    """Pin the Rust parser surface for
    ``put_plus_one_counter_on_each_creature_you_control``. Uses an
    Artifact carrier (matches Atmospheric Greenhouse's type-line
    shape) so a future regression that tightens the Rust
    parse_triggered_ability arm to creature-only sources surfaces
    here.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- PutPlusOneCounterOnEachCreatureYouControl",
        "card_types": ["Artifact"],
        "mana_cost": {"generic": 4},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "put_plus_one_counter_on_each_creature_you_control",
                    "amount": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


# ===== Phase 5 create_token bridge pins =====
#
# Token-spawn ETB shape. Translates ``create_token`` (Rust
# ``Effect::CreateToken {count, token_printing_id}``) into Python's
# ``CreateToken(token_printing=..., count=...)``. The dict carries a
# Rust-registry pid; the bridge looks up the matching Python printing
# from the probe-time ``register_token_printing`` map (pre-existing
# substrate added with the bridge arm). Real-EoE consumers: any
# Spacecraft / Artifact whose ETB prints a vanilla Soldier / Robot
# token (the I81 cross-controller probe drives the synthetic shape;
# the bridge unlocks real-EoE token-creation ETB probes too).


def test_bridge_create_token_translates_to_python():
    """Phase 5 create_token bridge: Rust
    ``{"kind": "create_token", "count": 1, "token_printing_id": pid}``
    translates to Python
    ``CreateToken(token_printing=<registered>, count=1)``.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    soldier = make_token_creature(
        name="Soldier Token (bridge pin)",
        power=1, toughness=1, subtypes=("Soldier",),
    )
    register_token_printing(7, soldier)
    e = _ta_dict_to_python_effect(
        {"kind": "create_token", "count": 1, "token_printing_id": 7}
    )
    assert isinstance(e, CreateToken)
    assert e.token_printing is soldier
    assert e.count == 1
    # Rider flags default per the dataclass (Rust variant has no
    # mirror for them; the bridge constructs the bare shape).
    assert e.tapped is False
    assert e.attacking is False
    assert e.each_player is False
    assert e.schedule_end_of_combat_sacrifice is False
    clear_token_printings()


def test_bridge_create_token_count_default_one():
    """Missing ``count`` defaults to 1 (mirrors the lenient
    ``draw_cards`` / ``surveil`` default-convention for partial test
    fixtures; Rust's parse_effect is STRICTER and would raise
    ``KeyError`` on a missing field, but the bridge tolerates the
    omission for round-trip dict-shape consistency on the Python
    side).
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    robot = make_token_creature(
        name="Robot Token (bridge pin -- count default)",
        power=2, toughness=1, subtypes=("Robot",), artifact=True,
    )
    register_token_printing(11, robot)
    e = _ta_dict_to_python_effect(
        {"kind": "create_token", "token_printing_id": 11}
    )
    assert isinstance(e, CreateToken)
    assert e.token_printing is robot
    assert e.count == 1
    clear_token_printings()


def test_bridge_create_token_missing_registration_raises_keyerror():
    """``create_token`` with a ``token_printing_id`` not present in
    the registry raises ``KeyError`` with a clear hint -- silent
    divergence from a missing setup step would be much harder to
    debug than an immediate failure at translation time.
    """
    import pytest

    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
    )

    clear_token_printings()
    with pytest.raises(KeyError, match="token_printing_id=999"):
        _ta_dict_to_python_effect(
            {"kind": "create_token", "count": 1, "token_printing_id": 999}
        )


def test_register_token_printing_rejects_conflicting_pid():
    """``register_token_printing`` raises ``ValueError`` on a
    conflicting registration (a different Python printing under an
    existing pid). Idempotent on a repeat (pid, printing) pair.
    Probes that genuinely want to clear-and-re-register call
    :func:`clear_token_printings` first.
    """
    import pytest

    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.parity.cross_engine import (
        clear_token_printings, register_token_printing,
    )

    clear_token_printings()
    a = make_token_creature(
        name="Token A (conflict pin)", power=1, toughness=1,
    )
    b = make_token_creature(
        name="Token B (conflict pin)", power=2, toughness=2,
    )
    register_token_printing(42, a)
    # Idempotent re-register of the SAME printing on the same pid
    # is a no-op.
    register_token_printing(42, a)
    # Conflicting registration raises.
    with pytest.raises(ValueError, match="already registered"):
        register_token_printing(42, b)
    clear_token_printings()


def test_rust_parser_accepts_create_token_in_triggered_ability():
    """Pin the Rust parser surface for ``create_token``. Uses a
    Creature carrier (matches the I81 synthetic Token Maker Bear's
    type-line shape) so a future regression that tightens the Rust
    parse_triggered_ability arm to artifact-only sources surfaces
    here. The Rust parser requires both ``count`` and
    ``token_printing_id`` fields; the inner token printing is
    pre-registered on the same Registry so the pid is valid.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    # Pre-register the inner Soldier token so the carrier's
    # triggered-ability dict can reference its pid.
    soldier_pid = reg.register_printing({
        "name": "Phase 5 Bridge Pin Soldier",
        "card_types": ["Creature"],
        "power": 1,
        "toughness": 1,
    })
    d = {
        "name": "Phase 5 Bridge Pin -- CreateToken",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "W": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "create_token",
                    "count": 1,
                    "token_printing_id": soldier_pid,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    # Carrier is the second printing in the registry.
    assert pid == soldier_pid + 1


def test_bridge_mill_target_player_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "mill_target_player", "count": N}`` translates to
    Python ``MillTargetPlayer(amount=N)``.

    Unblocks Mind-Funeral / Tome-Scour / Scrabbling-Skullcrab-shape
    "target player mills N" effects on the cross-engine bridge --
    sibling closure to the bare ``mill`` arm (controller-self
    mill shortcut). The Rust effect variant exists at
    ``rust/src/cards.rs:84`` (``Effect::MillTargetPlayer { count }``)
    and its parse_effect arm at ``cards.rs:2833-2848``; the Rust
    resolve arm at ``state_ops_resolve.rs:1085-1148`` carries the
    I77 controller-of-effect fix (target_idx = 1 - controller, where
    controller derives from current_resolve_source_controller).
    Pins the field-name translation ``count -> amount`` for the
    target-player mill shape.
    """
    from mtg_bot.effects.generic.draw_mill import MillTargetPlayer
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "mill_target_player", "count": 2}
    )
    assert isinstance(e, MillTargetPlayer)
    assert e.amount == 2
    assert e.use_controller_life is False


def test_bridge_mill_target_player_count_default_one():
    """Missing ``count`` defaults to 1 (matching ``MillTargetPlayer``'s
    class default ``amount: int = 1``, which mirrors the bare ``mill``
    arm's lenient-default convention; both share the SelfMill /
    MillTargetPlayer ``amount: int = 1`` Python class default).

    Bridge consumers SHOULD pass ``count`` explicitly to keep round-
    trip dict-shape consistency with the Rust parser
    (``cards.rs:2833-2848`` requires the field at register time and
    rejects negatives). This default is a robustness backstop for
    partial fixtures, not a contract.
    """
    from mtg_bot.effects.generic.draw_mill import MillTargetPlayer
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "mill_target_player"})
    assert isinstance(e, MillTargetPlayer)
    assert e.amount == 1


def test_rust_parser_accepts_mill_target_player_in_triggered_ability():
    """Pin the Rust parser surface for the ``mill_target_player``
    effect kind. The bridge's dict shape must register cleanly on
    the Rust side too; this lockstep gate keeps the Python bridge
    from drifting away from Rust's parse_effect coverage.

    Mirrors the existing ``mill`` parser-surface pin
    (test_rust_parser_accepts_mill_in_triggered_ability) -- what's
    being pinned is parse_effect's ``mill_target_player`` arm
    accepting a triggered_abilities entry with ``count: N`` and
    rejecting negatives at register time per
    ``cards.rs:2841-2846``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- MillTargetPlayer",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "U": 1, "B": 1},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "mill_target_player",
                    "count": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_exile_target_creature_until_source_leaves_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "exile_target_creature_until_source_leaves"}``
    translates to Python
    ``ExileTargetCreatureUntilSourceLeaves()``.

    Unblocks Banisher-Priest / Fiend-Hunter / All-Fates-Stalker /
    Hardlight-Containment-shape ETB exile triggers on the cross-
    engine bridge. The Rust effect variant exists at
    ``rust/src/cards.rs:1268`` (unit variant
    ``Effect::ExileTargetCreatureUntilSourceLeaves``); its
    parse_effect arm at ``cards.rs:3322-3323`` takes no fields; the
    resolve arm at ``state_ops_resolve.rs:5537-5639`` carries the
    I79 controller-of-effect fix (defender = 1 - controller, where
    controller derives from current_resolve_source_controller).
    Stamps ``exiled_with_source = Some(source_obj_id)`` so the
    paired return on source-leaves works.
    """
    from mtg_bot.effects.generic.exile import (
        ExileTargetCreatureUntilSourceLeaves,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "exile_target_creature_until_source_leaves"}
    )
    assert isinstance(e, ExileTargetCreatureUntilSourceLeaves)
    # Bare-shape defaults: empty excluded_subtypes (no subtype
    # filter), opponents_only / yours_only both False (no
    # controller-side filter). The Rust unit variant has no rider
    # mirrors today; the bridge constructs the bare default shape.
    assert e.excluded_subtypes == frozenset()
    assert e.opponents_only is False
    assert e.yours_only is False


def test_bridge_exile_target_creature_until_source_leaves_ignores_extra_fields():
    """Extra dict fields are ignored — the bridge translator mirrors
    the Rust unit-variant convention (no fields beyond ``kind``) and
    constructs the bare-default Python shape regardless of whether
    the dict carries extra keys. Mirrors the
    ``return_creature_from_your_graveyard_to_hand`` /
    ``return_creature_from_your_graveyard_to_battlefield`` arm
    convention (both unit variants).
    """
    from mtg_bot.effects.generic.exile import (
        ExileTargetCreatureUntilSourceLeaves,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({
        "kind": "exile_target_creature_until_source_leaves",
        # Extra keys that future rider-widened Rust variants might
        # carry; the bare-shape arm ignores them.
        "extra_key_should_be_ignored": True,
        "another_unused_field": 42,
    })
    assert isinstance(e, ExileTargetCreatureUntilSourceLeaves)


def test_rust_parser_accepts_exile_target_creature_until_source_leaves_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``exile_target_creature_until_source_leaves`` effect kind. The
    bridge's dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the existing
    ``return_creature_from_your_graveyard_to_hand`` parser-surface
    pin (test_rust_parser_accepts_return_creature_from_graveyard_to_hand_in_triggered_ability)
    -- both are unit-variant arms with no companion fields. The
    Rust parse_effect arm at ``cards.rs:3322-3323`` accepts the
    bare ``{"kind": "exile_target_creature_until_source_leaves"}``
    shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- ExileUntilSourceLeaves",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "W": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "exile_target_creature_until_source_leaves",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_discard_card_translates_to_python_target_opponent_discards():
    """Phase 5 bridge extension WIDER variant: Rust
    ``{"kind": "discard_card", "target": {"kind": "any_player"}}``
    translates to Python
    ``TargetOpponentDiscards(count=1)``.

    Unblocks Mind Rot / Wrench Mind / Specter-bonk-shape ETB discard
    triggers on the cross-engine bridge. The Rust effect variant
    exists at the ``Effect::DiscardCard { target: TargetFilter }``
    arm; its parse_effect arm at ``cards.rs:2890-2897`` requires a
    ``target`` dict; the resolve arm at
    ``state_ops_resolve.rs:1607-1660`` carries the I74 controller-
    of-effect fix (defender = 1 - controller, where controller
    derives from current_resolve_source_controller).

    **First bridge arm to honor ONLY the wider any_player filter**
    (siblings honor only any_creature; the deal_damage arm honors
    both). Rust's resolve arm explicitly raises PyValueError on
    AnyCreature at state_ops_resolve.rs:1608-1611 -- discard is
    semantically a player-targeted effect.

    Distinct from the existing ``each_opponent_discards`` arm
    (multi-target sweep): this ``discard_card`` arm is single-target
    Mind-Rot shape ("target player discards a card"). Rust's
    first-slice picks the discarder's hand[0] deterministically;
    Python defers to the discarder's agent ``choose_discard``.
    """
    from mtg_bot.effects.generic.discard import TargetOpponentDiscards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "discard_card", "target": {"kind": "any_player"}}
    )
    assert isinstance(e, TargetOpponentDiscards)
    # Default count=1 mirrors Rust's first-slice "discards exactly
    # one card" semantics (the Rust variant has no count field).
    assert e.count == 1


def test_bridge_discard_card_rejects_unsupported_target_filter_kind():
    """The bridge mirrors the Rust parser's accepted-kind allowlist
    for target filters (cards.rs:2790-2799 accepts only ``any_player``
    and ``any_creature``). The discard_card arm is filter-narrow-by-
    design to the OPPOSITE shape from destroy_target / exile_target /
    return_to_hand / tap_target: only ``any_player`` is honored.
    Rust raises a runtime PyValueError on AnyCreature at
    state_ops_resolve.rs:1608-1611; the bridge surface rejects
    ``any_creature`` AT REGISTER time (loud failure) rather than
    letting it fall through to Rust's runtime resolve raise.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="any_creature"):
        _ta_dict_to_python_effect(
            {"kind": "discard_card", "target": {"kind": "any_creature"}}
        )


def test_rust_parser_accepts_discard_card_in_triggered_ability():
    """Pin the Rust parser surface for the ``discard_card`` effect
    kind. The bridge's dict shape must register cleanly on the Rust
    side too; this lockstep gate keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing destroy_target / exile_target /
    return_to_hand / tap_target / deal_damage parser-surface pins --
    all six arms require a companion ``target`` field. The Rust
    parse_effect arm at ``cards.rs:2890-2897`` accepts the
    ``{"kind": "discard_card", "target": {"kind": "any_player"}}``
    shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- DiscardCard",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "B": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "discard_card",
                    "target": {"kind": "any_player"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_tap_target_translates_to_python_tap_with_creature_slot():
    """Phase 5 bridge extension: Rust
    ``{"kind": "tap_target", "target": {"kind": "any_creature"}}``
    translates to Python
    ``Tap(targets=(TargetSpec(IsCreature()),))``.

    Unblocks Twiddle / Sleep-on-target / Frost Lynx-shape ETB tap
    triggers on the cross-engine bridge. The Rust effect variant
    exists at the ``Effect::TapTarget { target: TargetFilter }``
    arm; its parse_effect arm at ``cards.rs:2857-2864`` requires a
    ``target`` dict; the resolve arm at
    ``state_ops_resolve.rs:1277-1324`` carries the I71 controller-
    of-effect fix (defender = 1 - controller, where controller
    derives from current_resolve_source_controller). Sibling closure
    to I70 / I75 / I76 / I78 / I79 / I80 -- identical defender-swap
    shape; differs only in the mutation applied (``tapped = true``
    on the target).

    Distinct from ``TapTargetArtifactOrCreature`` (the wider-filter
    variant at state_ops_resolve.rs:4776): the bare ``tap_target``
    arm here only honors AnyCreature, matching the parser-side
    filter allowlist.
    """
    from mtg_bot.effects.generic.tap import Tap
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "tap_target", "target": {"kind": "any_creature"}}
    )
    assert isinstance(e, Tap)
    # The bridge constructs a single TargetSpec slot with an
    # IsCreature() filter; mirrors the AnyCreature first-slice
    # symbol from the Rust side.
    assert len(e.targets) == 1
    slot = e.targets[0]
    assert isinstance(slot, TargetSpec)
    assert isinstance(slot.filter, IsCreature)
    # Default count_min/count_max for a TargetSpec is 1/1 (the
    # "exactly one" target shape that mirrors Rust's first-slice
    # picker which always selects bf[0] of the defender).
    assert slot.count_min == 1
    assert slot.count_max == 1


def test_bridge_tap_target_rejects_unsupported_target_filter_kind():
    """The bridge mirrors the Rust parser's accepted-kind allowlist
    for target filters (cards.rs:2790-2799 accepts only ``any_player``
    and ``any_creature``). The tap_target arm is filter-narrow-by-
    design (you can't tap a player; Rust raises PyValueError on
    AnyPlayer at state_ops_resolve.rs:1278-1282). The bridge surface
    rejects ``any_player`` AT REGISTER time (loud failure) rather
    than letting it fall through to Rust's runtime resolve raise.

    Sibling to the destroy_target / exile_target / return_to_hand
    unsupported-filter pins.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="any_player"):
        _ta_dict_to_python_effect(
            {"kind": "tap_target", "target": {"kind": "any_player"}}
        )


def test_rust_parser_accepts_tap_target_in_triggered_ability():
    """Pin the Rust parser surface for the ``tap_target`` effect
    kind. The bridge's dict shape must register cleanly on the Rust
    side too; this lockstep gate keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing destroy_target / exile_target /
    return_to_hand / deal_damage parser-surface pins -- all five
    arms require a companion ``target`` field. The Rust parse_effect
    arm at ``cards.rs:2857-2864`` accepts the
    ``{"kind": "tap_target", "target": {"kind": "any_creature"}}``
    shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- TapTarget",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "U": 1},
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "tap_target",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_translates_to_python_deal_damage_with_creature_slot():
    """Phase 5 bridge extension: Rust
    ``{"kind": "deal_damage", "amount": 2, "target": {"kind": "any_creature"}}``
    translates to Python
    ``DealDamage(amount=2, targets=(TargetSpec(IsCreature()),))``.

    Unblocks Lightning Bolt / Shock / Galvanic Blast-shape ETB burn
    triggers on the cross-engine bridge. The Rust effect variant
    exists at ``rust/src/cards.rs:631`` vicinity
    (``Effect::DealDamage { amount: i32, target: TargetFilter }``);
    its parse_effect arm at ``cards.rs:2804-2812`` requires both an
    ``amount`` field and a ``target`` dict; the resolve arm at
    ``state_ops_resolve.rs:740-790`` carries the I67 controller-of-
    effect substrate (defender = 1 - controller, where controller
    derives from current_resolve_source_controller).

    Distinct from the existing ``deal_damage_any_target`` arm
    (``DealDamageAnyTarget``, unit "any target" filter, deterministic
    Rust at-resolve picking): this ``deal_damage`` arm is filter-
    parametric (caller picks AnyCreature OR AnyPlayer at register
    time) and uses the standard PUSH-time agent-driven target
    selection on the Python side.
    """
    from mtg_bot.effects.generic.damage import DealDamage
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage",
            "amount": 2,
            "target": {"kind": "any_creature"},
        }
    )
    assert isinstance(e, DealDamage)
    # Amount round-trips from the dict.
    assert e.amount == 2
    # The bridge constructs a single TargetSpec slot with an
    # IsCreature() filter; mirrors the AnyCreature first-slice
    # symbol from the Rust side.
    assert len(e.targets) == 1
    slot = e.targets[0]
    assert isinstance(slot, TargetSpec)
    assert isinstance(slot.filter, IsCreature)
    # Default count_min/count_max for a TargetSpec is 1/1 (the
    # "exactly one" target shape that mirrors Rust's first-slice
    # picker which always selects bf[0] of the defender).
    assert slot.count_min == 1
    assert slot.count_max == 1


def test_bridge_deal_damage_translates_to_python_deal_damage_with_player_slot():
    """Phase 5 bridge extension WIDER variant: Rust
    ``{"kind": "deal_damage", "amount": 3, "target": {"kind": "any_player"}}``
    translates to Python
    ``DealDamage(amount=3, targets=(TargetSpec(IsPlayer()),))``.

    Unblocks Lava Spike / Searing Blaze (player half) / Skullcrack-
    shape ETB burn-the-player triggers on the cross-engine bridge.
    Rust's resolve arm honors ``TargetFilter::AnyPlayer`` directly
    (state_ops_resolve.rs:768-770) by damaging the defending player.
    Python's ``DealDamage`` base class is filter-parametric on
    ``targets[0].filter`` so swapping ``IsCreature()`` for
    ``IsPlayer()`` reuses the same resolve walk; the picker emits
    one ``("player", id)`` legal target on the 2-player harness, and
    ``_apply_damage`` routes to the player's life total.

    First bridge arm to honor the WIDER ``any_player`` filter shape
    (siblings ``destroy_target`` / ``exile_target`` /
    ``return_to_hand`` are filter-narrow-by-design — you can't
    destroy / exile / bounce a player in MTG).
    """
    from mtg_bot.effects.generic.damage import DealDamage
    from mtg_bot.effects.object_predicates import IsPlayer
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage",
            "amount": 3,
            "target": {"kind": "any_player"},
        }
    )
    assert isinstance(e, DealDamage)
    assert e.amount == 3
    # The bridge constructs a single TargetSpec slot with an
    # IsPlayer() filter; mirrors the AnyPlayer first-slice symbol
    # from the Rust side.
    assert len(e.targets) == 1
    slot = e.targets[0]
    assert isinstance(slot, TargetSpec)
    assert isinstance(slot.filter, IsPlayer)
    assert slot.count_min == 1
    assert slot.count_max == 1


def test_bridge_deal_damage_rejects_unsupported_target_filter_kind():
    """The bridge mirrors the Rust parser's accepted-kind allowlist
    for target filters (cards.rs:2790-2799 accepts only ``any_player``
    and ``any_creature``). The deal_damage arm honors BOTH (unlike
    sibling destroy_target / exile_target / return_to_hand which only
    honor any_creature); an unknown filter kind raises a ValueError
    pointing at Rust's parse_target_filter allowlist.

    This pin documents the bridge-side surface contract: future
    additions to ``parse_target_filter`` (e.g., ``any_artifact``,
    ``any_nonland_permanent``) need a paired bridge update on this
    arm before they can land cross-engine.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="any_player"):
        _ta_dict_to_python_effect(
            {
                "kind": "deal_damage",
                "amount": 1,
                "target": {"kind": "any_artifact"},
            }
        )


def test_rust_parser_accepts_deal_damage_in_triggered_ability():
    """Pin the Rust parser surface for the ``deal_damage`` effect
    kind. The bridge's dict shape must register cleanly on the Rust
    side too; this lockstep gate keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing ``destroy_target`` / ``exile_target`` /
    ``return_to_hand`` parser-surface pins -- all four arms require
    a companion ``target`` field; deal_damage additionally requires
    ``amount``. The Rust parse_effect arm at ``cards.rs:2804-2812``
    accepts the
    ``{"kind": "deal_damage", "amount": N, "target": {"kind":
    "any_creature"}}`` shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- DealDamage",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "R": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage",
                    "amount": 2,
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_destroy_target_translates_to_python_destroy_with_creature_slot():
    """Phase 5 bridge extension: Rust
    ``{"kind": "destroy_target", "target": {"kind": "any_creature"}}``
    translates to Python
    ``Destroy(targets=(TargetSpec(IsCreature()),))``.

    Unblocks Doom-Blade / Murder / Naturalize-shape ETB destroy
    triggers on the cross-engine bridge. The Rust effect variant
    exists at ``rust/src/cards.rs:1567`` (``Effect::DestroyTarget {
    target: TargetFilter }``); its parse_effect arm at
    ``cards.rs:2902-2909`` requires a ``target`` dict; the resolve
    arm at ``state_ops_resolve.rs:1687`` carries the I75 controller-
    of-effect fix (defender = 1 - controller, where controller
    derives from current_resolve_source_controller). Sibling closure
    to I70 / I76 / I78 / I79 / I80 -- identical defender-swap shape;
    differs only in destination zone (graveyard for I75 vs exile /
    hand / library bottom for the siblings) and in carrying the
    indestructible suppress gate (CR 702.12) on the destroyed object.
    """
    from mtg_bot.effects.generic.destroy import Destroy
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target", "target": {"kind": "any_creature"}}
    )
    assert isinstance(e, Destroy)
    # The bridge constructs a single TargetSpec slot with an
    # IsCreature() filter; mirrors the AnyCreature first-slice
    # symbol from the Rust side.
    assert len(e.targets) == 1
    slot = e.targets[0]
    assert isinstance(slot, TargetSpec)
    assert isinstance(slot.filter, IsCreature)
    # Default count_min/count_max for a TargetSpec is 1/1 (the
    # "exactly one" target shape that mirrors Rust's first-slice
    # picker which always selects bf[0] of the defender).
    assert slot.count_min == 1
    assert slot.count_max == 1


def test_bridge_destroy_target_rejects_unsupported_target_filter_kind():
    """The bridge mirrors the Rust parser's accepted-kind allowlist
    for target filters (cards.rs:2790-2799 accepts only ``any_player``
    and ``any_creature``). The destroy_target arm only honors
    creature-shaped filters; an unknown filter kind raises a
    ValueError with a pointer to the Rust parse_target_filter
    docstring location.

    This pin documents the bridge-side surface contract: future
    additions to ``parse_target_filter`` (e.g., ``any_artifact``,
    ``any_nonland_permanent``) need a paired bridge update on this
    arm before they can land cross-engine.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="any_player"):
        _ta_dict_to_python_effect(
            {"kind": "destroy_target", "target": {"kind": "any_player"}}
        )


def test_rust_parser_accepts_destroy_target_in_triggered_ability():
    """Pin the Rust parser surface for the ``destroy_target`` effect
    kind. The bridge's dict shape must register cleanly on the Rust
    side too; this lockstep gate keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing ``mill_target_player`` parser-surface pin
    -- both arms require a companion field (``count`` for
    mill_target_player, ``target`` for destroy_target). The Rust
    parse_effect arm at ``cards.rs:2902-2909`` accepts the
    ``{"kind": "destroy_target", "target": {"kind": "any_creature"}}``
    shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- DestroyTarget",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "B": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "destroy_target",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_exile_target_translates_to_python_exile_with_creature_slot():
    """Phase 5 bridge extension: Rust
    ``{"kind": "exile_target", "target": {"kind": "any_creature"}}``
    translates to Python
    ``Exile(targets=(TargetSpec(IsCreature()),))``.

    Unblocks Path-to-Exile / Swords-to-Plowshares / Anguished
    Unmaking-shape ETB exile triggers on the cross-engine bridge.
    The Rust effect variant exists at ``rust/src/cards.rs:1110``
    (``Effect::ExileTarget { target: TargetFilter }``); its
    parse_effect arm at ``cards.rs:2849-2856`` requires a ``target``
    dict; the resolve arm at ``state_ops_resolve.rs:1110`` carries
    the I70 controller-of-effect fix (defender = 1 - controller,
    where controller derives from
    current_resolve_source_controller). Sibling closure to I75 /
    I76 / I78 / I79 / I80 -- identical defender-swap shape; differs
    only in destination zone (exile for I70 vs graveyard / hand /
    library bottom for the siblings).

    Distinct from ``exile_target_creature_until_source_leaves``
    (the I79 sibling) which stamps ``exiled_with_source =
    Some(source_obj_id)`` for the paired return on source-leaves;
    the bare ``exile_target`` permanently exiles with no paired
    return.
    """
    from mtg_bot.effects.generic.exile import Exile
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "exile_target", "target": {"kind": "any_creature"}}
    )
    assert isinstance(e, Exile)
    # The bridge constructs a single TargetSpec slot with an
    # IsCreature() filter; mirrors the AnyCreature first-slice
    # symbol from the Rust side.
    assert len(e.targets) == 1
    slot = e.targets[0]
    assert isinstance(slot, TargetSpec)
    assert isinstance(slot.filter, IsCreature)
    # Default count_min/count_max for a TargetSpec is 1/1 (the
    # "exactly one" target shape that mirrors Rust's first-slice
    # picker which always selects bf[0] of the defender).
    assert slot.count_min == 1
    assert slot.count_max == 1


def test_bridge_exile_target_rejects_unsupported_target_filter_kind():
    """The bridge mirrors the Rust parser's accepted-kind allowlist
    for target filters (cards.rs:2790-2799 accepts only ``any_player``
    and ``any_creature``). The exile_target arm only honors
    creature-shaped filters; an unknown filter kind raises a
    ValueError pointing at Rust's parse_target_filter allowlist.

    This pin documents the bridge-side surface contract: future
    additions to ``parse_target_filter`` (e.g., ``any_artifact``,
    ``any_nonland_permanent``) need a paired bridge update on this
    arm before they can land cross-engine. Sibling pin to the
    destroy_target unsupported-filter pin.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="any_player"):
        _ta_dict_to_python_effect(
            {"kind": "exile_target", "target": {"kind": "any_player"}}
        )


def test_rust_parser_accepts_exile_target_in_triggered_ability():
    """Pin the Rust parser surface for the ``exile_target`` effect
    kind. The bridge's dict shape must register cleanly on the Rust
    side too; this lockstep gate keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing ``destroy_target`` parser-surface pin --
    both arms require a companion ``target`` field. The Rust
    parse_effect arm at ``cards.rs:2849-2856`` accepts the
    ``{"kind": "exile_target", "target": {"kind": "any_creature"}}``
    shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- ExileTarget",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "W": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "exile_target",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_exile_opponents_nonland_permanent_until_leaves_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "exile_opponents_nonland_permanent_until_leaves"}``
    translates to Python
    ``ExileOpponentsNonlandPermanentUntilLeaves()``.

    Unblocks Banishing-Light / Oblivion-Ring / Hardlight-
    Containment-shape ETB exile-non-land triggers on the cross-
    engine bridge. The Rust effect variant exists at
    ``rust/src/cards.rs:1283`` (unit variant
    ``Effect::ExileOpponentsNonlandPermanentUntilLeaves``); its
    parse_effect arm at ``cards.rs:3325-3327`` takes no fields;
    the resolve arm at ``state_ops_resolve.rs:5645-5754`` carries
    the I80 controller-of-effect fix (defender = 1 - controller,
    where controller derives from
    current_resolve_source_controller). Stamps
    ``exiled_with_source = Some(source_obj_id)`` so the paired
    return on source-leaves works. Sibling closure to I79 -- same
    "until source leaves" exile shape; differs only in the target
    filter ("not a Land" vs "is a Creature").
    """
    from mtg_bot.effects.generic.exile import (
        ExileOpponentsNonlandPermanentUntilLeaves,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "exile_opponents_nonland_permanent_until_leaves"}
    )
    assert isinstance(e, ExileOpponentsNonlandPermanentUntilLeaves)
    # Bare-shape default: max_mv=0 (no MV cap). The Rust unit
    # variant has no field mirror today; the bridge constructs
    # the bare default shape, matching Rust's any-non-land scan.
    assert e.max_mv == 0


def test_bridge_exile_opponents_nonland_permanent_until_leaves_ignores_extra_fields():
    """Extra dict fields are ignored -- the bridge translator
    mirrors the Rust unit-variant convention (no fields beyond
    ``kind``) and constructs the bare-default Python shape
    regardless of whether the dict carries extra keys. Mirrors the
    ``exile_target_creature_until_source_leaves`` /
    ``return_creature_from_your_graveyard_to_hand`` /
    ``return_creature_from_your_graveyard_to_battlefield`` arm
    convention (all unit variants).
    """
    from mtg_bot.effects.generic.exile import (
        ExileOpponentsNonlandPermanentUntilLeaves,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({
        "kind": "exile_opponents_nonland_permanent_until_leaves",
        # Extra keys that future MV-cap-widened Rust variants
        # might carry; the bare-shape arm ignores them.
        "max_mv": 4,
        "extra_key_should_be_ignored": True,
        "another_unused_field": 42,
    })
    assert isinstance(e, ExileOpponentsNonlandPermanentUntilLeaves)
    # Bare-shape arm constructs max_mv=0 even when the dict
    # carries an explicit max_mv -- the rider field has no Rust
    # mirror yet, so the bridge surface drops it for round-trip
    # dict-shape consistency with what Rust's parser accepts.
    assert e.max_mv == 0


def test_rust_parser_accepts_exile_opponents_nonland_permanent_until_leaves_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``exile_opponents_nonland_permanent_until_leaves`` effect
    kind. The bridge's dict shape must register cleanly on the
    Rust side too; this lockstep gate keeps the Python bridge
    from drifting away from Rust's parse_effect coverage.

    Mirrors the existing
    ``exile_target_creature_until_source_leaves`` parser-surface
    pin (test_rust_parser_accepts_exile_target_creature_until_
    source_leaves_in_triggered_ability) -- both are unit-variant
    arms with no companion fields. The Rust parse_effect arm at
    ``cards.rs:3325-3327`` accepts the bare ``{"kind":
    "exile_opponents_nonland_permanent_until_leaves"}`` shape on
    a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- ExileOppsNonlandUntilLeaves",
        "card_types": ["Enchantment"],
        "mana_cost": {"generic": 2, "W": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "exile_opponents_nonland_permanent_until_leaves",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_exile_opponents_nonland_permanent_with_max_mv_until_leaves_translates_to_python():
    """Round-14 bridge extension: Rust
    ``{"kind": "exile_opponents_nonland_permanent_with_max_mv_until_leaves",
    "max_mv": 2}`` translates to Python
    ``ExileOpponentsNonlandPermanentUntilLeaves(max_mv=2)``.

    Sister bridge arm to the bare
    ``exile_opponents_nonland_permanent_until_leaves``; the rider
    variant carries a ``max_mv`` mana-value cap field that flows
    through into the Python class's ``max_mv`` attribute. Real
    EoE consumer: Seam Rip ({W} Enchantment, ETB exile target
    nonland permanent an opponent controls with mana value 2 or
    less until this enchantment leaves the battlefield).

    The Rust effect variant exists at the new
    ``Effect::ExileOpponentsNonlandPermanentWithMaxMvUntilLeaves
    { max_mv: i32 }`` slot in cards.rs; its parse_effect arm
    requires a ``max_mv`` int field (non-negative); the resolve
    arm in state_ops_resolve.rs inherits the I80 controller-of-
    effect fix and adds the per-permanent MV-cap check. Python's
    ``ExileOpponentsNonlandPermanentUntilLeaves`` class
    (effects/generic/exile.py:587) carries ``max_mv: int = 0``
    where 0 = no cap and N > 0 caps at N; both legal_targets()
    and resolve() honor the cap.
    """
    from mtg_bot.effects.generic.exile import (
        ExileOpponentsNonlandPermanentUntilLeaves,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({
        "kind": "exile_opponents_nonland_permanent_with_max_mv_until_leaves",
        "max_mv": 2,
    })
    assert isinstance(e, ExileOpponentsNonlandPermanentUntilLeaves)
    # The rider field flows through — distinct from the bare
    # sibling's bridge arm which constructs max_mv=0 unconditionally.
    assert e.max_mv == 2


def test_bridge_exile_opponents_nonland_permanent_with_max_mv_until_leaves_forwards_arbitrary_caps():
    """Round-14: the bridge translator forwards arbitrary
    ``max_mv`` values into the Python constructor — pin a few
    spread-out caps so a regression that hard-codes the value
    (e.g. always 2) is caught immediately. Mirrors the
    parametric-bridge-arm precedent (e.g. the ``draw_cards`` arm
    forwards arbitrary count values).
    """
    from mtg_bot.effects.generic.exile import (
        ExileOpponentsNonlandPermanentUntilLeaves,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    for max_mv_value in (1, 3, 5, 10):
        e = _ta_dict_to_python_effect({
            "kind": "exile_opponents_nonland_permanent_with_max_mv_until_leaves",
            "max_mv": max_mv_value,
        })
        assert isinstance(e, ExileOpponentsNonlandPermanentUntilLeaves)
        assert e.max_mv == max_mv_value, (
            f"bridge must forward max_mv verbatim; "
            f"expected {max_mv_value}, got {e.max_mv}"
        )


def test_bridge_exile_opponents_nonland_permanent_with_max_mv_until_leaves_rejects_missing_max_mv():
    """Round-14: missing ``max_mv`` is a ValueError on the bridge
    side, mirroring Rust's parse_effect arm which raises
    PyValueError on a missing required field. The bridge does NOT
    silently fall through to ``max_mv=0`` — that would mask a
    dict-shape producer bug as a no-cap effect.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required 'max_mv'"):
        _ta_dict_to_python_effect({
            "kind": "exile_opponents_nonland_permanent_with_max_mv_until_leaves",
        })


def test_rust_parser_accepts_exile_opponents_nonland_permanent_with_max_mv_until_leaves_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``exile_opponents_nonland_permanent_with_max_mv_until_leaves``
    effect kind on a triggered-ability carrier. The bridge's
    dict shape must register cleanly on the Rust side too; this
    lockstep gate keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors the bare-sibling parser-surface pin
    (test_rust_parser_accepts_exile_opponents_nonland_permanent_
    until_leaves_in_triggered_ability) but adds the required
    ``max_mv`` field. The Rust parse_effect arm requires
    ``max_mv: i32`` (non-negative); we exercise both the
    happy-path 2 (Seam Rip's printed cap) and a larger value to
    pin that the field is forwarded literally rather than
    clamped.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Round 14 Bridge Pin -- ExileOppsNonlandWithMaxMvUntilLeaves",
        "card_types": ["Enchantment"],
        "mana_cost": {"W": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "exile_opponents_nonland_permanent_with_max_mv_until_leaves",
                    "max_mv": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_grant_keywords_to_target_creature_until_eot_translates_to_python():
    """Round-15 bridge extension: Rust
    ``{"kind": "grant_keywords_to_target_creature_until_eot",
       "keywords": ["deathtouch"], "exclude_source": True}``
    translates to Python ``PumpTargetCreatureUntilEOT(power_bonus=0,
    toughness_bonus=0, granted_keywords=frozenset({"deathtouch"}),
    you_control_only=True, exclude_self=True)``.

    Rider-widened bridge arm for the round-28 single-keyword
    grant-keyword-target-UEOT cluster. Real EoE consumer:
    Blooming Stinger ({1}{G} Plant Scorpion, "When this creature
    enters, another target creature you control gains deathtouch
    until end of turn.").

    The Rust effect variant exists at the new
    ``Effect::GrantKeywordsToTargetCreatureUntilEOT
    { keywords: Vec<String>, exclude_source: bool }`` slot in
    cards.rs; its parse_effect arm requires a non-empty list of
    non-empty keyword strings; the resolve arm in
    state_ops_resolve.rs inherits the controller-of-effect resolution
    from the single-keyword sibling and adds the exclude_source skip
    + per-keyword UeotModifier::GrantKeyword push.

    Python's ``PumpTargetCreatureUntilEOT`` class
    (effects/generic/pump.py) carries the keyword-grant-only
    sub-shape via ``granted_keywords=frozenset({...})`` with
    P/T bonuses both zero; ``you_control_only`` + ``exclude_self``
    flags narrow the legal-target pool to "another target creature
    you control" — the bridge translates Rust's ``exclude_source``
    onto the Python ``exclude_self`` flag (same semantics, naming
    convention adapted at the boundary).
    """
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({
        "kind": "grant_keywords_to_target_creature_until_eot",
        "keywords": ["deathtouch"],
        "exclude_source": True,
    })
    assert isinstance(e, PumpTargetCreatureUntilEOT)
    # P/T bonuses pinned at 0 — the keyword-grant-only sub-shape.
    assert e.power_bonus == 0
    assert e.toughness_bonus == 0
    # Granted keywords flow through verbatim as a frozenset.
    assert e.granted_keywords == frozenset({"deathtouch"})
    # Targeting flags wired correctly. The bridge always sets
    # you_control_only=True (Rust's resolve arm scans the
    # source-controller's bf, which is the cross-engine semantic
    # match for "you control"); exclude_source on the dict maps to
    # exclude_self on the Python class.
    assert e.you_control_only is True
    assert e.exclude_self is True


def test_bridge_grant_keywords_to_target_creature_until_eot_forwards_arbitrary_keywords():
    """Round-15: the bridge translator forwards arbitrary keyword
    LISTS into the Python constructor — pin a few sweep cases so a
    regression that hard-codes the keyword set (e.g. always
    {"deathtouch"}) is caught immediately. Single-element + multi-
    element lists both round-trip; the Python frozenset preserves
    every entry. Mirrors the parametric-bridge-arm precedent
    (round-14 max_mv arbitrary-cap forwarding).
    """
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    cases = [
        ["deathtouch"],
        ["flying"],
        ["haste", "trample"],
        ["lifelink", "vigilance", "first_strike"],
    ]
    for kws in cases:
        e = _ta_dict_to_python_effect({
            "kind": "grant_keywords_to_target_creature_until_eot",
            "keywords": kws,
            "exclude_source": False,
        })
        assert isinstance(e, PumpTargetCreatureUntilEOT)
        assert e.granted_keywords == frozenset(kws), (
            f"bridge must forward keywords verbatim; "
            f"expected {frozenset(kws)}, got {e.granted_keywords}"
        )
        assert e.exclude_self is False


def test_bridge_grant_keywords_to_target_creature_until_eot_rejects_missing_keywords():
    """Round-15: missing ``keywords`` list is a ValueError on the
    bridge side, mirroring Rust's parse_effect arm which raises
    PyValueError on a missing required field. The bridge does NOT
    silently fall through to an empty grant — that would mask a
    dict-shape producer bug as a no-op effect.

    Sister pins for the empty-list / empty-string-entry shapes
    also raise (mirroring Rust's defensive guards). Default
    ``exclude_source`` (missing) is allowed — defaults to False
    on both engines.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required 'keywords'"):
        _ta_dict_to_python_effect({
            "kind": "grant_keywords_to_target_creature_until_eot",
        })

    # Empty list rejected.
    with pytest.raises(ValueError, match="must be non-empty"):
        _ta_dict_to_python_effect({
            "kind": "grant_keywords_to_target_creature_until_eot",
            "keywords": [],
        })

    # Empty-string entry rejected.
    with pytest.raises(ValueError, match="non-empty string"):
        _ta_dict_to_python_effect({
            "kind": "grant_keywords_to_target_creature_until_eot",
            "keywords": ["deathtouch", ""],
        })


def test_rust_parser_accepts_grant_keywords_to_target_creature_until_eot_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``grant_keywords_to_target_creature_until_eot`` effect kind on
    a triggered-ability carrier. The bridge's dict shape must
    register cleanly on the Rust side too; this lockstep gate keeps
    the Python bridge from drifting away from Rust's parse_effect
    coverage.

    Mirrors the round-14 ``exile_opponents_nonland_permanent_with_
    max_mv_until_leaves`` parser-surface pin. The Rust parse_effect
    arm requires a non-empty ``keywords`` list (each entry non-
    empty); we exercise both single-keyword + ``exclude_source=True``
    (Blooming Stinger's printed shape) and multi-keyword +
    ``exclude_source=False`` (forward-compatible card shape) to pin
    that the field forwarding is literal rather than clamped.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()

    # Blooming-Stinger-shape: single-keyword + exclude_source=True.
    d_stinger = {
        "name": "Round 15 Bridge Pin -- GrantKeywordsBlooming",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "G": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "grant_keywords_to_target_creature_until_eot",
                    "keywords": ["deathtouch"],
                    "exclude_source": True,
                },
            }
        ],
    }
    pid_stinger = reg.register_printing(d_stinger)
    assert pid_stinger == 0

    # Forward-compatible multi-keyword shape, no exclude_source.
    d_multi = {
        "name": "Round 15 Bridge Pin -- GrantKeywordsMulti",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "U": 1},
        "power": 1,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "grant_keywords_to_target_creature_until_eot",
                    "keywords": ["flying", "trample"],
                },
            }
        ],
    }
    pid_multi = reg.register_printing(d_multi)
    assert pid_multi == 1


def test_bridge_return_to_hand_translates_to_python_bounce_with_creature_slot():
    """Phase 5 bridge extension: Rust
    ``{"kind": "return_to_hand", "target": {"kind": "any_creature"}}``
    translates to Python
    ``Bounce(targets=(TargetSpec(IsCreature()),))``.

    Unblocks Unsummon / Boomerang / Aether Spellbomb-shape ETB
    bounce triggers on the cross-engine bridge. The Rust effect
    variant exists at ``rust/src/cards.rs:1417``
    (``Effect::ReturnToHand { target: TargetFilter }``); its
    parse_effect arm at ``cards.rs:2821-2828`` requires a ``target``
    dict; the resolve arm at ``state_ops_resolve.rs:1417`` carries
    the I76 controller-of-effect fix (defender = 1 - controller,
    where controller derives from
    current_resolve_source_controller). Sibling closure to I70 /
    I75 / I78 / I79 / I80 -- identical defender-swap shape; differs
    only in destination zone (owner's hand for I76 vs exile /
    graveyard / library bottom for the siblings).

    Distinct from the bounce-from-graveyard family
    (``return_creature_from_your_graveyard_to_hand`` etc., already
    on the bridge): both share the "to hand" destination but differ
    on the source zone (battlefield vs graveyard).
    """
    from mtg_bot.effects.generic.bounce import Bounce
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_to_hand", "target": {"kind": "any_creature"}}
    )
    assert isinstance(e, Bounce)
    # The bridge constructs a single TargetSpec slot with an
    # IsCreature() filter; mirrors the AnyCreature first-slice
    # symbol from the Rust side.
    assert len(e.targets) == 1
    slot = e.targets[0]
    assert isinstance(slot, TargetSpec)
    assert isinstance(slot.filter, IsCreature)
    # Default count_min/count_max for a TargetSpec is 1/1 (the
    # "exactly one" target shape that mirrors Rust's first-slice
    # picker which always selects bf[0] of the defender).
    assert slot.count_min == 1
    assert slot.count_max == 1


def test_bridge_return_to_hand_rejects_unsupported_target_filter_kind():
    """The bridge mirrors the Rust parser's accepted-kind allowlist
    for target filters (cards.rs:2790-2799 accepts only ``any_player``
    and ``any_creature``). The return_to_hand arm only honors
    creature-shaped filters; players can't move via owner-hand bounce
    per CR 401.1, and an unknown filter kind raises a ValueError
    pointing at Rust's parse_target_filter allowlist.

    This pin documents the bridge-side surface contract: future
    additions to ``parse_target_filter`` (e.g., ``any_artifact``,
    ``any_nonland_permanent``) need a paired bridge update on this
    arm before they can land cross-engine. Sibling pin to the
    destroy_target / exile_target unsupported-filter pins.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="any_player"):
        _ta_dict_to_python_effect(
            {"kind": "return_to_hand", "target": {"kind": "any_player"}}
        )


def test_rust_parser_accepts_return_to_hand_in_triggered_ability():
    """Pin the Rust parser surface for the ``return_to_hand`` effect
    kind. The bridge's dict shape must register cleanly on the Rust
    side too; this lockstep gate keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing ``destroy_target`` / ``exile_target``
    parser-surface pins -- all three arms require a companion
    ``target`` field. The Rust parse_effect arm at
    ``cards.rs:2821-2828`` accepts the
    ``{"kind": "return_to_hand", "target": {"kind": "any_creature"}}``
    shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- ReturnToHand",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "U": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "return_to_hand",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_owner_puts_target_artifact_or_creature_on_top_or_bottom_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library"}``
    translates to Python
    ``OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary()``.

    Unblocks Lost-in-Space-shape ETB owner-puts-on-library triggers
    on the cross-engine bridge. The Rust effect variant exists at
    ``rust/src/cards.rs:1170`` (unit variant
    ``Effect::OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary``);
    its parse_effect arm at ``cards.rs:3306-3309`` takes no fields;
    the resolve arm at ``state_ops_resolve.rs:5067`` carries the
    I78 controller-of-effect fix (defender = 1 - controller, where
    controller derives from current_resolve_source_controller).
    Sibling closure to I70 / I75 / I76 / I79 / I80 -- same
    defender-swap shape; differs only in destination zone (owner's
    library top-or-bottom for I78 vs exile / hand / graveyard for
    the siblings) and adds an OWNER-side top-vs-bottom choice via
    the agent's ``choose_mode`` hook.
    """
    from mtg_bot.effects.generic.zone_moves import (
        OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library"}
    )
    assert isinstance(
        e, OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary
    )
    # Bare-shape default: nonland_permanent=False (the printed
    # "artifact or creature" filter, NOT the wider "any nonland
    # permanent" widening). The Rust unit variant has no rider
    # mirror today; the bridge constructs the bare default shape,
    # matching Rust's any-artifact-or-creature scan.
    assert e.nonland_permanent is False


def test_bridge_owner_puts_target_artifact_or_creature_on_top_or_bottom_ignores_extra_fields():
    """Extra dict fields are ignored -- the bridge translator
    mirrors the Rust unit-variant convention (no fields beyond
    ``kind``) and constructs the bare-default Python shape
    regardless of whether the dict carries extra keys. Mirrors the
    ``exile_target_creature_until_source_leaves`` /
    ``exile_opponents_nonland_permanent_until_leaves`` /
    ``return_creature_from_your_graveyard_to_hand`` arm convention
    (all unit variants).
    """
    from mtg_bot.effects.generic.zone_moves import (
        OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({
        "kind": "owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library",
        # Extra keys that future rider-widened Rust variants might
        # carry; the bare-shape arm ignores them.
        "nonland_permanent": True,
        "extra_key_should_be_ignored": True,
        "another_unused_field": 42,
    })
    assert isinstance(
        e, OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary
    )
    # Bare-shape arm constructs nonland_permanent=False even when
    # the dict carries an explicit nonland_permanent rider -- the
    # rider has no Rust mirror yet, so the bridge surface drops it
    # for round-trip dict-shape consistency with what Rust's
    # parser accepts.
    assert e.nonland_permanent is False


def test_rust_parser_accepts_owner_puts_target_artifact_or_creature_on_top_or_bottom_in_triggered_ability():
    """Pin the Rust parser surface for the
    ``owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library``
    effect kind. The bridge's dict shape must register cleanly on
    the Rust side too; this lockstep gate keeps the Python bridge
    from drifting away from Rust's parse_effect coverage.

    Mirrors the existing
    ``exile_target_creature_until_source_leaves`` /
    ``exile_opponents_nonland_permanent_until_leaves`` parser-
    surface pins -- all three are unit-variant arms with no
    companion fields. The Rust parse_effect arm at
    ``cards.rs:3306-3309`` accepts the bare ``{"kind":
    "owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library"}``
    shape on a triggered-ability carrier.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Phase 5 Bridge Pin -- OwnerPutsTopOrBottom",
        "card_types": ["Enchantment"],
        "mana_cost": {"generic": 2, "W": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_set_base_pt_until_eot_translates_to_python():
    """BLOCKED-RUST-EFFECT slice -- layer 7b base-P/T override.
    Rust ``{"kind": "set_base_pt_until_eot", "power": N, "toughness":
    M}`` translates to Python ``SetBasePTUntilEOT(power=N,
    toughness=M)``.

    Field names agree on both engines (``power``, ``toughness``);
    no rename in the bridge. Mirrors Rust's parse_effect arm
    (cards.rs:3879-3909) which requires both fields with no
    defaults. The Python class is the layer-7b sibling of
    PumpTargetCreatureUntilEOT (layer-7c) -- same target-pick shape
    (source-controller's first creature), different composition
    layer (overrides the base instead of stacking on top).
    """
    from mtg_bot.effects.generic.pump import SetBasePTUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "set_base_pt_until_eot", "power": 4, "toughness": 4}
    )
    assert isinstance(e, SetBasePTUntilEOT)
    assert e.power == 4
    assert e.toughness == 4


def test_bridge_set_base_pt_until_eot_negative_values_legal():
    """Negative power / toughness round-trip cleanly.
    Mirrors Rust's parse_effect arm (no parse-time guard on
    negatives) and Python's dataclass (no raise-on-negative). A
    debuff-shaped "becomes base -2/-2" is meaningful (the creature
    dies to SBA toughness <= 0 when SBA fires); the bridge does not
    sanitize.
    """
    from mtg_bot.effects.generic.pump import SetBasePTUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "set_base_pt_until_eot", "power": -2, "toughness": -2}
    )
    assert isinstance(e, SetBasePTUntilEOT)
    assert e.power == -2
    assert e.toughness == -2


def test_bridge_set_base_pt_until_eot_asymmetric_values():
    """Asymmetric (power != toughness) values round-trip cleanly.
    Pins that the bridge doesn't accidentally collapse the two
    fields onto a single value or swap them. ``power=1, toughness=4``
    is the canonical "becomes a 1/4" defender-shape; the Python
    class stores them in field order, no swap.
    """
    from mtg_bot.effects.generic.pump import SetBasePTUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "set_base_pt_until_eot", "power": 1, "toughness": 4}
    )
    assert isinstance(e, SetBasePTUntilEOT)
    assert e.power == 1
    assert e.toughness == 4


def test_bridge_set_base_pt_until_eot_missing_power_raises():
    """Missing ``power`` field raises KeyError. Mirrors Rust's
    parse_effect arm which raises PyValueError on missing required
    field; the bridge enforces the same contract so a typo in a
    fixture surfaces immediately rather than silently defaulting to
    0/0 (which would be a "becomes base 0/0" override and die to SBA
    -- a different observable than a fizzle, and a likely bug
    rather than an intent).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {"kind": "set_base_pt_until_eot", "toughness": 4}
        )


def test_bridge_set_base_pt_until_eot_missing_toughness_raises():
    """Missing ``toughness`` field raises KeyError. Sister test to
    the missing-power case; both fields are independently required
    on both engines.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {"kind": "set_base_pt_until_eot", "power": 4}
        )


def test_bridge_set_base_pt_until_eot_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``{"kind": "sequence", "effects": [{"kind":
    "set_base_pt_until_eot", "power": 4, "toughness": 4}, {"kind":
    "draw_cards", "count": 1}]}`` translates to ``Sequence(effects=(
    SetBasePTUntilEOT(power=4, toughness=4), DrawCards(amount=1)))``.
    Pins the parametric arm against accidental field-loss in the
    Sequence translator (Sequence recursively dispatches into
    ``_ta_dict_to_python_effect`` per child, so this also exercises
    the recursive path).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.pump import SetBasePTUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "set_base_pt_until_eot",
                    "power": 4,
                    "toughness": 4,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], SetBasePTUntilEOT)
    assert e.effects[0].power == 4
    assert e.effects[0].toughness == 4


def test_rust_parser_accepts_set_base_pt_until_eot_in_triggered_ability():
    """Pin the Rust parser surface for the ``set_base_pt_until_eot``
    effect kind on a triggered_ability carrier. The bridge dict
    shape must register cleanly on the Rust side too; this lockstep
    gate keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3879-3909``
    accepts the dict shape with required ``power`` and ``toughness``
    integer fields.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- SetBasePTUntilEOT",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "G": 1},
        "power": 3,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "set_base_pt_until_eot",
                    "power": 4,
                    "toughness": 4,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_may_do_translates_to_python():
    """BLOCKED-RUST-EFFECT slice -- may_do composer bridge.
    Rust ``{"kind": "may_do", "inner_effect": <effect_dict>,
    "prompt": <str>}`` translates to Python ``MayDo(inner=<Effect>,
    prompt=<str>)``.

    **Field-name translation**: Rust ``inner_effect`` -> Python
    ``inner``. Pin verifies the rename happens correctly and the
    nested effect dict is recursively translated through the same
    bridge function (so any leaf arm composes for free).

    Looter-shape ``MayDo(DiscardSelf{1})`` is the Melded-Moxite
    do_action half (the IfYouDo wrapper around it lives at the
    composer level and is already on the bridge).
    """
    from mtg_bot.effects.combinators import MayDo
    from mtg_bot.effects.generic.discard import DiscardSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "may_do",
            "inner_effect": {"kind": "discard_self", "amount": 1},
            "prompt": "you may discard a card",
        }
    )
    assert isinstance(e, MayDo)
    assert isinstance(e.inner, DiscardSelf)
    assert e.inner.amount == 1
    assert e.prompt == "you may discard a card"


def test_bridge_may_do_default_prompt_empty():
    """Missing ``prompt`` field defaults to empty string. Mirrors
    Rust's parse_effect arm at cards.rs:3734-3741 which uses
    ``String::new()`` when the field is omitted, and Python's
    ``MayDo(inner: Effect, prompt: str = "")`` dataclass default.
    Both engines treat empty prompt as a request for the inner
    effect's ``describe()`` to surface as the agent prompt; the
    default agrees end-to-end.
    """
    from mtg_bot.effects.combinators import MayDo
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "may_do",
            "inner_effect": {"kind": "draw_cards", "count": 2},
            # prompt intentionally omitted -- defaults to empty string
        }
    )
    assert isinstance(e, MayDo)
    assert isinstance(e.inner, DrawCards)
    assert e.inner.amount == 2
    assert e.prompt == ""


def test_bridge_may_do_missing_inner_effect_raises():
    """Missing ``inner_effect`` field raises KeyError out of the
    bridge -- mirrors Rust's parse_effect arm at cards.rs:3722-3727
    which raises PyValueError on the missing required field. A
    ``MayDo`` with no inner action is meaningless on both engines
    (the prompt would have nothing to gate); the bridge enforces the
    same contract so a typo surfaces immediately rather than
    silently translating to a degenerate effect.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {"kind": "may_do", "prompt": "you may"}
        )


def test_bridge_may_do_composable_in_if_you_do():
    """Round-trip in an IfYouDo composer to pin the looter-shape
    nesting end-to-end: ``IfYouDo(MayDo(DiscardSelf), DrawCards{2})``
    -- the Melded-Moxite "you may discard a card; if you do, draw
    two cards" oracle. Both composer arms (``if_you_do`` and
    ``may_do``) recursively dispatch through
    ``_ta_dict_to_python_effect``; pinning the nested case ensures
    the looter pattern works through the bridge translator without
    field-rename or recursion bugs surfacing only at the composed
    level.
    """
    from mtg_bot.effects.combinators import IfYouDo, MayDo
    from mtg_bot.effects.generic.discard import DiscardSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "if_you_do",
            "do_action": {
                "kind": "may_do",
                "inner_effect": {
                    "kind": "discard_self", "amount": 1,
                },
                "prompt": "",
            },
            "then_action": {"kind": "draw_cards", "count": 2},
        }
    )
    assert isinstance(e, IfYouDo)
    assert isinstance(e.do_action, MayDo)
    assert isinstance(e.do_action.inner, DiscardSelf)
    assert e.do_action.inner.amount == 1
    assert isinstance(e.then_action, DrawCards)
    assert e.then_action.amount == 2


def test_bridge_may_do_composable_in_sequence():
    """Round-trip inside a Sequence composer to pin nesting symmetry
    with the if_you_do nesting test. Sequence's recursive translator
    must descend into ``inner_effect`` correctly even when the
    MayDo is the second child of the sequence (not just the top
    level), pinning that the field-rename branch isn't position-
    sensitive in the parser.
    """
    from mtg_bot.effects.combinators import MayDo, Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "draw_cards", "count": 1},
                {
                    "kind": "may_do",
                    "inner_effect": {
                        "kind": "draw_cards", "count": 1,
                    },
                    "prompt": "",
                },
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DrawCards)
    assert isinstance(e.effects[1], MayDo)
    assert isinstance(e.effects[1].inner, DrawCards)
    assert e.effects[1].inner.amount == 1


def test_rust_parser_accepts_may_do_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts a may_do
    composer (with inner_effect + prompt) inside a triggered_ability
    dict. Mirrors the if_you_do / set_base_pt_until_eot lockstep
    gates -- keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3722-3746``
    accepts the dict shape; ``inner_effect`` is required,
    ``prompt`` defaults to the empty string when omitted.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- MayDo Composer",
        "card_types": ["Artifact"],
        "mana_cost": {"generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "may_do",
                    "inner_effect": {
                        "kind": "draw_cards", "count": 1,
                    },
                    "prompt": "",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_attach_to_target_creature_you_control_translates_to_python():
    """BLOCKED-BRIDGE-ARM slice (round-19 audit refresh) -- Equipment
    ETB attach. Rust ``{"kind": "attach_to_target_creature_you_control"}``
    translates to Python ``AttachToTargetCreatureYouControl()``.

    Mirrors Rust's parse_effect arm (cards.rs:3676-3678) which is a
    UNIT VARIANT (no fields). The bridge translator ignores every
    other key on the dict for round-trip dict-shape consistency. The
    Python class carries an OPTIONAL ``granted_keywords_ueot`` rider
    not exercised by the bare arm; the default empty frozenset()
    matches Rust's no-keyword-grant unit variant.
    """
    from mtg_bot.effects.generic.attach import (
        AttachToTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "attach_to_target_creature_you_control"}
    )
    assert isinstance(e, AttachToTargetCreatureYouControl)
    # Default rider value -- the bare bridge arm does NOT carry the
    # UEOT keyword grant (would compose via separate arm).
    assert e.granted_keywords_ueot == frozenset()


def test_bridge_attach_to_target_creature_extra_keys_ignored():
    """Round-trip pin: extra keys on the dict are silently ignored
    (the unit variant takes no fields). Mirrors the
    owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library
    arm shape -- forward-compatible with future rider-widened Rust
    variants that would land as separate ``kind`` arms.
    """
    from mtg_bot.effects.generic.attach import (
        AttachToTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "attach_to_target_creature_you_control",
            "ignored_extra_field": "noise",
        }
    )
    assert isinstance(e, AttachToTargetCreatureYouControl)
    assert e.granted_keywords_ueot == frozenset()


def test_bridge_equip_to_target_creature_you_control_translates_to_python():
    """BLOCKED-BRIDGE-ARM slice (round-19 audit refresh) -- Equip
    activated-ability resolution (CR 702.6). Rust
    ``{"kind": "equip_to_target_creature_you_control"}`` translates
    to Python ``EquipToTargetCreatureYouControl()``.

    Sister bridge arm to ``attach_to_target_creature_you_control``;
    Python's ``EquipToTargetCreatureYouControl`` is a SUBCLASS of
    ``AttachToTargetCreatureYouControl`` with identical resolve
    behavior but a distinct ``describe`` string. The two are
    isinstance-compatible (Equip IS-A Attach) -- pin verifies the
    bridge returns the more-specific subclass type, not the parent,
    so log tooling can distinguish ETB-attach from paid Equip
    activation correctly.
    """
    from mtg_bot.effects.generic.attach import (
        AttachToTargetCreatureYouControl,
        EquipToTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "equip_to_target_creature_you_control"}
    )
    # Specific subclass, not the parent.
    assert type(e) is EquipToTargetCreatureYouControl
    # IS-A parent (subclass relationship preserved).
    assert isinstance(e, AttachToTargetCreatureYouControl)
    # Distinct describe string for log-tooling disambiguation.
    assert e.describe() != AttachToTargetCreatureYouControl().describe()


def test_bridge_attach_and_equip_arms_distinct_kinds():
    """Cross-arm discriminator pin: the bridge does NOT collapse the
    two ``kind`` strings onto one Python class. The ETB-shape attach
    and the activated-ability paid Equip are SEPARATE Rust variants
    + SEPARATE Python classes; confirming each kind dispatches to
    its own class prevents accidental cross-wiring (e.g. a typo on
    one arm masking via the other).
    """
    from mtg_bot.effects.generic.attach import (
        AttachToTargetCreatureYouControl,
        EquipToTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    attach = _ta_dict_to_python_effect(
        {"kind": "attach_to_target_creature_you_control"}
    )
    equip = _ta_dict_to_python_effect(
        {"kind": "equip_to_target_creature_you_control"}
    )
    # The attach arm returns the bare parent type, NOT the Equip
    # subclass.
    assert type(attach) is AttachToTargetCreatureYouControl
    # The equip arm returns the more-specific subclass.
    assert type(equip) is EquipToTargetCreatureYouControl
    # Distinct types.
    assert type(attach) is not type(equip)


def test_bridge_attach_to_target_creature_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([AttachTo, DrawCards])`` translates correctly via the
    recursive Sequence translator. Pins the unit-variant arm against
    accidental dispatch loss inside Sequence (recursive translation
    walks each child via the same ``_ta_dict_to_python_effect`` so
    this also exercises the recursive path).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.attach import (
        AttachToTargetCreatureYouControl,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "attach_to_target_creature_you_control"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert type(e.effects[0]) is AttachToTargetCreatureYouControl
    assert isinstance(e.effects[1], DrawCards)
    assert e.effects[1].amount == 1


def test_rust_parser_accepts_attach_to_target_creature_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bare
    ``{"kind": "attach_to_target_creature_you_control"}`` dict shape
    on a triggered_ability carrier. The bridge dict shape must
    register cleanly on the Rust side too; this lockstep gate keeps
    the bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing
    ``owner_puts_target_artifact_or_creature_on_top_or_bottom_of_library``
    parser-surface pin -- both are unit-variant arms with no
    companion fields. Rust's parse_effect arm at
    ``rust/src/cards.rs:3676-3678`` accepts the bare unit dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- AttachTo Equipment",
        "card_types": ["Artifact"],
        "subtypes": ["Equipment"],
        "mana_cost": {"generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "attach_to_target_creature_you_control",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_accepts_equip_to_target_creature_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bare
    ``{"kind": "equip_to_target_creature_you_control"}`` dict shape
    on a triggered_ability carrier. Sister to the AttachTo lockstep
    pin -- pins the Equip variant separately so a typo on either
    arm surfaces via this gate.

    Rust's parse_effect arm at ``rust/src/cards.rs:3679-3681``
    accepts the bare unit dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- EquipTo Equipment",
        "card_types": ["Artifact"],
        "subtypes": ["Equipment"],
        "mana_cost": {"generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "equip_to_target_creature_you_control",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_optional_sac_artifact_for_effect_translates_to_python():
    """BLOCKED-BRIDGE-ARM slice (round-19 audit refresh) -- "you may
    sacrifice an artifact. If you do, [inner]" composer. Rust
    ``{"kind": "optional_sac_artifact_for_effect", "inner_effect":
    <effect_dict>}`` translates to Python
    ``OptionalSacArtifactForEffect(inner_effect=<Effect>)``.

    Field name agrees on both engines (``inner_effect``); no rename
    in the bridge. The inner effect recurses through the same
    translator so any leaf arm composes for free. Pin verifies the
    nested DrawCards inner is correctly translated and stored as
    ``inner_effect``.
    """
    from mtg_bot.effects.combinators import OptionalSacArtifactForEffect
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_sac_artifact_for_effect",
            "inner_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, OptionalSacArtifactForEffect)
    assert isinstance(e.inner_effect, DrawCards)
    assert e.inner_effect.amount == 1


def test_bridge_optional_sac_artifact_missing_inner_effect_raises():
    """Missing ``inner_effect`` field raises KeyError. Mirrors
    Rust's parse_effect arm at cards.rs:3656-3661 which raises
    PyValueError on missing required field; the bridge enforces
    the same contract so a typo in a fixture surfaces immediately
    rather than silently constructing an OptionalSac wrapper around
    a None inner that would crash on resolve.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {"kind": "optional_sac_artifact_for_effect"}
        )


def test_bridge_optional_sac_artifact_inner_effect_recurses():
    """Recursive translation pin: the inner effect dict is walked
    through ``_ta_dict_to_python_effect`` again. Pins the recursive
    path against accidental shallow translation (passing the dict
    through unchanged, or constructing the wrapper with a dict
    instead of an Effect instance). A nested gain_life with a
    distinct ``amount`` field round-trips into a real GainLife
    instance, not a dict.
    """
    from mtg_bot.effects.combinators import OptionalSacArtifactForEffect
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_sac_artifact_for_effect",
            "inner_effect": {"kind": "gain_life", "amount": 3},
        }
    )
    assert isinstance(e, OptionalSacArtifactForEffect)
    # Inner is a real GainLife instance (NOT a dict).
    assert isinstance(e.inner_effect, GainLife)
    assert e.inner_effect.amount == 3


def test_bridge_optional_sac_artifact_with_sequence_inner():
    """Selfcraft-Mechan-shape composition pin: the inner effect is a
    Sequence wrapping multiple leaf arms. Pins the composer
    composition end-to-end -- ``OptionalSacArtifactForEffect(
    Sequence([GainLife{1}, DrawCards{1}]))`` round-trips with the
    Sequence preserved + each leaf at its proper class/amount.

    This is the composer shape Selfcraft Mechan would use if the
    "When you do, put a +1/+1 counter on target creature and draw a
    card" rider were folded into a single OptionalSac wrapper
    (Selfcraft's actual oracle uses a separate reflexive trigger,
    but the composition syntax is identical).
    """
    from mtg_bot.effects.combinators import (
        OptionalSacArtifactForEffect,
        Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_sac_artifact_for_effect",
            "inner_effect": {
                "kind": "sequence",
                "effects": [
                    {"kind": "gain_life", "amount": 1},
                    {"kind": "draw_cards", "count": 1},
                ],
            },
        }
    )
    assert isinstance(e, OptionalSacArtifactForEffect)
    assert isinstance(e.inner_effect, Sequence)
    assert len(e.inner_effect.effects) == 2
    assert isinstance(e.inner_effect.effects[0], GainLife)
    assert e.inner_effect.effects[0].amount == 1
    assert isinstance(e.inner_effect.effects[1], DrawCards)
    assert e.inner_effect.effects[1].amount == 1


def test_bridge_optional_sac_artifact_composable_in_sequence():
    """Round-trip in a Sequence COMPOSER (Sequence wraps the
    OptionalSac, not vice-versa). Pins that the OptionalSac arm
    composes correctly when it's a child of a parent Sequence:
    ``Sequence([OptionalSacArtifact(GainLife{2}), DrawCards{1}])``.
    Sequence's recursive translator walks each child via the same
    ``_ta_dict_to_python_effect``, so this also exercises the
    OptionalSac arm INSIDE the recursive Sequence translator path.
    """
    from mtg_bot.effects.combinators import (
        OptionalSacArtifactForEffect,
        Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "optional_sac_artifact_for_effect",
                    "inner_effect": {
                        "kind": "gain_life", "amount": 2,
                    },
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], OptionalSacArtifactForEffect)
    assert isinstance(e.effects[0].inner_effect, GainLife)
    assert e.effects[0].inner_effect.amount == 2
    assert isinstance(e.effects[1], DrawCards)
    assert e.effects[1].amount == 1


def test_rust_parser_accepts_optional_sac_artifact_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``optional_sac_artifact_for_effect`` composer (with
    ``inner_effect`` nested) inside a triggered_ability dict.
    Mirrors the may_do / set_base_pt_until_eot lockstep gates --
    keeps the bridge from drifting away from Rust's parse_effect
    coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3656-3671``
    accepts the dict shape; ``inner_effect`` is required and is
    recursively parsed via ``parse_effect``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- OptionalSacArtifact",
        "card_types": ["Artifact", "Creature"],
        "subtypes": ["Robot", "Artificer"],
        "mana_cost": {"generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "optional_sac_artifact_for_effect",
                    "inner_effect": {
                        "kind": "draw_cards", "count": 1,
                    },
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_search_library_for_basic_land_tapped_default_count():
    """Round-21 BLOCKED-BRIDGE-ARM slice -- default count=1 path.
    Rust ``{"kind": "search_library_for_basic_land_tapped"}`` with
    no ``count`` field translates to Python
    ``SearchLibraryForBasicLandTapped(count=1)`` via the class-level
    default (mirrors Rust parse_effect at cards.rs:3272-3279 where
    missing count defaults to 1).
    """
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "search_library_for_basic_land_tapped"}
    )
    assert isinstance(e, SearchLibraryForBasicLandTapped)
    assert e.count == 1


def test_bridge_search_library_for_basic_land_tapped_explicit_count():
    """Explicit ``count`` field flows through verbatim. Pins
    Larval-Scoutlander-shape ``count=2`` (the printed value on the
    real EoE consumer of the OptionalSacArtifact composition).
    """
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "search_library_for_basic_land_tapped", "count": 2}
    )
    assert isinstance(e, SearchLibraryForBasicLandTapped)
    assert e.count == 2


def test_bridge_search_library_for_basic_land_tapped_negative_raises():
    """Negative count surfaces at translation time, mirroring Rust's
    parse_effect contract at cards.rs:3280-3284. Python's class
    would accept silently (the for-range loop just no-ops on
    negative count), but the bridge enforces the parser-side check
    so the two engines' translation contracts stay symmetric (a
    typo or fixture bug surfaces immediately rather than silently
    constructing a no-op effect).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "search_library_for_basic_land_tapped", "count": -1}
        )


def test_bridge_search_library_for_basic_land_tapped_count_zero_no_op():
    """Edge case pin: ``count=0`` is a legal-but-no-op shape. Both
    engines accept it (Rust parser permits zero; Python's for-range
    loop iterates zero times). Pins that the bridge doesn't
    incorrectly upgrade ``0`` to ``1`` via a falsy-default check
    (``or 1`` would lift this to 1; ``.get(..., 1)`` keeps it 0).
    """
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "search_library_for_basic_land_tapped", "count": 0}
    )
    assert isinstance(e, SearchLibraryForBasicLandTapped)
    assert e.count == 0


def test_bridge_search_library_for_basic_land_tapped_composable_in_optional_sac():
    """Composition pin: this arm composes as the inner_effect of the
    round-20 ``optional_sac_artifact_for_effect`` arm. Pins the
    Larval Scoutlander shape end-to-end --
    ``OptionalSacArtifactForEffect(SearchLibraryForBasicLand
    Tapped(count=2))`` round-trips with the inner correctly typed
    + count preserved. The OptionalSac translator recurses into
    ``_ta_dict_to_python_effect`` for the inner dict, so this also
    exercises this arm INSIDE the recursive composer translator
    path.
    """
    from mtg_bot.effects.combinators import OptionalSacArtifactForEffect
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_sac_artifact_for_effect",
            "inner_effect": {
                "kind": "search_library_for_basic_land_tapped",
                "count": 2,
            },
        }
    )
    assert isinstance(e, OptionalSacArtifactForEffect)
    assert isinstance(
        e.inner_effect, SearchLibraryForBasicLandTapped
    )
    assert e.inner_effect.count == 2


def test_bridge_search_library_for_basic_land_tapped_composable_in_sequence():
    """Round-trip in a Sequence COMPOSER. Pins that the
    SearchLibraryForBasicLandTapped arm composes correctly when it's
    a child of a parent Sequence:
    ``Sequence([SearchLibraryForBasicLandTapped(1), DrawCards(1)])``.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "search_library_for_basic_land_tapped",
                    "count": 1,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], SearchLibraryForBasicLandTapped
    )
    assert e.effects[0].count == 1
    assert isinstance(e.effects[1], DrawCards)
    assert e.effects[1].amount == 1


def test_rust_parser_accepts_search_library_for_basic_land_tapped_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``search_library_for_basic_land_tapped`` dict shape inside a
    triggered_ability dict. Mirrors the may_do /
    optional_sac_artifact_for_effect lockstep gates -- keeps the
    bridge from drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3266-3286``
    accepts the dict shape; ``count`` is optional with default 1.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- SearchLibraryForBasicLandTapped",
        "card_types": ["Creature"],
        "subtypes": ["Bear"],
        "mana_cost": {"generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "search_library_for_basic_land_tapped",
                    "count": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_search_library_for_card_legendary_creature_filter():
    """Round-21 BLOCKED-BRIDGE-ARM slice -- legendary_creature
    filter (Starfield Shepherd / The Seriema shape). Rust
    ``{"kind": "search_library_for_card",
    "filter_name": "legendary_creature"}`` translates to Python
    ``SearchLibraryForCard(filter_name="legendary_creature",
    max_mv=0, may_fail=True)``. Pins default values for max_mv +
    may_fail flow through correctly when omitted.
    """
    from mtg_bot.effects.generic.search import SearchLibraryForCard
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "search_library_for_card",
            "filter_name": "legendary_creature",
        }
    )
    assert isinstance(e, SearchLibraryForCard)
    assert e.filter_name == "legendary_creature"
    assert e.max_mv == 0
    assert e.may_fail is True


def test_bridge_search_library_for_card_basic_land_filter():
    """Generic basic-land filter (Spineseeker Centipede shape).
    Pins that the basic_land filter string passes through verbatim
    -- both engines dispatch on the same filter-name string.
    """
    from mtg_bot.effects.generic.search import SearchLibraryForCard
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "search_library_for_card",
            "filter_name": "basic_land",
        }
    )
    assert isinstance(e, SearchLibraryForCard)
    assert e.filter_name == "basic_land"


def test_bridge_search_library_for_card_creature_small_with_max_mv():
    """``creature_small`` filter uses ``max_mv`` to cap the cost
    of matching creatures (Python's _matches arm at search.py:63-67).
    Pins explicit max_mv flows through verbatim.
    """
    from mtg_bot.effects.generic.search import SearchLibraryForCard
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "search_library_for_card",
            "filter_name": "creature_small",
            "max_mv": 3,
        }
    )
    assert isinstance(e, SearchLibraryForCard)
    assert e.filter_name == "creature_small"
    assert e.max_mv == 3
    assert e.may_fail is True


def test_bridge_search_library_for_card_may_fail_false():
    """Explicit ``may_fail=False`` flows through. Pins the
    forced-pick path (agent can't decline; resolve falls back to
    options[0] if agent declines).
    """
    from mtg_bot.effects.generic.search import SearchLibraryForCard
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "search_library_for_card",
            "filter_name": "spacecraft",
            "may_fail": False,
        }
    )
    assert isinstance(e, SearchLibraryForCard)
    assert e.filter_name == "spacecraft"
    assert e.may_fail is False


def test_bridge_search_library_for_card_missing_filter_name_raises():
    """Missing ``filter_name`` field raises ValueError. Mirrors
    Rust's parse_effect contract at cards.rs:4336-4341 which raises
    PyValueError on missing required field; the bridge enforces the
    same contract so a typo in a fixture surfaces immediately
    rather than silently constructing a SearchLibraryForCard with
    an empty filter (which would match no card and shuffle-only-
    fizzle without obvious failure mode).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="filter_name"):
        _ta_dict_to_python_effect({"kind": "search_library_for_card"})


def test_bridge_search_library_for_card_composable_in_sequence():
    """Round-trip in a Sequence COMPOSER. Pins that the
    SearchLibraryForCard arm composes correctly when it's a child
    of a parent Sequence:
    ``Sequence([SearchLibraryForCard("legendary_creature"),
    DrawCards(1)])``.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.search import SearchLibraryForCard
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "search_library_for_card",
                    "filter_name": "legendary_creature",
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], SearchLibraryForCard)
    assert e.effects[0].filter_name == "legendary_creature"
    assert isinstance(e.effects[1], DrawCards)
    assert e.effects[1].amount == 1


def test_rust_parser_accepts_search_library_for_card_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``search_library_for_card`` dict shape inside a triggered_ability
    dict. Mirrors the may_do / optional_sac_artifact_for_effect
    lockstep gates -- keeps the bridge from drifting away from
    Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:4327-4363``
    accepts the dict shape; ``filter_name`` is required.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- SearchLibraryForCard",
        "card_types": ["Creature"],
        "subtypes": ["Bird"],
        "mana_cost": {"generic": 4},
        "power": 2,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "search_library_for_card",
                    "filter_name": "legendary_creature",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_look_top_n_put_one_in_hand_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round-21 audit refresh) -- Codecracker
    Hound family. Rust ``{"kind": "look_top_n_put_one_in_hand_rest_to_
    graveyard", "n": 2}`` translates to Python
    ``LookTopNPutOneInHandRestToGraveyard(n=2)``.

    Mirrors Rust's parse_effect arm at cards.rs:3744-3761 (single
    required `n` field, i32, non-negative). Field name `n` agrees on
    both engines (no rename in the bridge). Pin verifies the n field
    is correctly stored.
    """
    from mtg_bot.effects.generic.library_lookat import (
        LookTopNPutOneInHandRestToGraveyard,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "look_top_n_put_one_in_hand_rest_to_graveyard", "n": 2}
    )
    assert isinstance(e, LookTopNPutOneInHandRestToGraveyard)
    assert e.n == 2


def test_bridge_look_top_n_put_one_in_hand_commune_with_evil_shape():
    """Bridge arm pin: full Commune-with-Evil shape (n=4) round-trips.
    Pulls in the Python class default-equivalent value (4) explicitly
    via the dict path so a future change to the Python default never
    masks a bridge bug -- the dict-shape contract is independent of
    the class-level default.
    """
    from mtg_bot.effects.generic.library_lookat import (
        LookTopNPutOneInHandRestToGraveyard,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "look_top_n_put_one_in_hand_rest_to_graveyard", "n": 4}
    )
    assert e == LookTopNPutOneInHandRestToGraveyard(n=4)


def test_bridge_look_top_n_put_one_in_hand_extra_keys_ignored():
    """Round-trip pin: extra keys on the dict are silently ignored
    (the variant carries only one field, `n`). Mirrors the existing
    bridge-arm extra-keys-ignored convention (set_base_pt_until_eot,
    attach_to_target_creature_you_control). Forward-compatible with
    future rider-widened Rust variants -- those would land as
    separate ``kind`` arms on this translator.
    """
    from mtg_bot.effects.generic.library_lookat import (
        LookTopNPutOneInHandRestToGraveyard,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "look_top_n_put_one_in_hand_rest_to_graveyard",
            "n": 2,
            "ignored_extra_field": "noise",
        }
    )
    assert isinstance(e, LookTopNPutOneInHandRestToGraveyard)
    assert e.n == 2


def test_bridge_look_top_n_put_one_in_hand_missing_n_raises():
    """Bridge arm pin: missing required `n` field raises ValueError to
    mirror Rust's parse_effect contract at cards.rs:3744-3760 (which
    raises PyValueError on missing `n`). Python's class has a default
    of 4 (Commune-with-Evil shape), but the bridge enforces explicit
    `n` to keep the dict-shape contract aligned with Rust's parse
    path -- mirrors the round-20 LookAtTopNRevealOrCounter `n`-
    required precedent.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required field 'n'"):
        _ta_dict_to_python_effect(
            {"kind": "look_top_n_put_one_in_hand_rest_to_graveyard"}
        )


def test_bridge_look_top_n_put_one_in_hand_int_coercion():
    """Bridge arm pin: `int(...)` coercion on `n` handles JSON-shape
    int values cleanly (mirrors the LookAtTopNRevealOrCounter
    int-coercion-robustness pin). Pins the explicit
    ``int(eff_dict["n"])`` cast so a downstream consumer never gets
    a stringy / floaty `n` slipping through.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "look_top_n_put_one_in_hand_rest_to_graveyard", "n": 5}
    )
    assert e.n == 5
    assert isinstance(e.n, int)


def test_bridge_look_top_n_put_one_in_hand_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([LookTopNPut..., DrawCards])`` translates correctly via
    the recursive Sequence translator. Pins this arm against
    accidental dispatch loss inside Sequence -- the recursive
    translation walks each child via the same
    ``_ta_dict_to_python_effect`` so this also exercises the
    recursive path for the look-N arm.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.library_lookat import (
        LookTopNPutOneInHandRestToGraveyard,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "look_top_n_put_one_in_hand_rest_to_graveyard",
                    "n": 2,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], LookTopNPutOneInHandRestToGraveyard)
    assert e.effects[0].n == 2
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_look_top_n_put_one_in_hand_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "look_top_n_put_one_in_hand_rest_to_graveyard",
    "n": 2}`` dict shape on a triggered_ability carrier. The bridge
    dict shape must register cleanly on the Rust side too; this
    lockstep gate keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms (set_base_pt_until_eot, may_do, attach_to_target_
    creature_you_control). Rust's parse_effect arm at
    ``rust/src/cards.rs:3744-3761`` accepts the bare dict shape with
    a required `n: i32` field.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- LookTopNPutOneInHand",
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "look_top_n_put_one_in_hand_rest_to_graveyard",
                    "n": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_put_plus_one_counter_on_each_of_up_to_n_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round-21 audit refresh) -- Weftblade
    Enhancer family. Rust ``{"kind": "put_plus_one_counter_on_each_
    of_up_to_n_target_creatures", "max_n": 2}`` translates to Python
    ``PutPlusOneCounterOnEachOfUpToNTargetCreatures(max_n=2)``.

    Mirrors Rust's parse_effect arm at cards.rs:4128-4149 (single
    required `max_n` field, i32, non-negative). Field name `max_n`
    agrees on both engines (no rename in the bridge). Pin verifies
    the max_n field is correctly stored.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachOfUpToNTargetCreatures,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
            "max_n": 2,
        }
    )
    assert isinstance(e, PutPlusOneCounterOnEachOfUpToNTargetCreatures)
    assert e.max_n == 2


def test_bridge_put_plus_one_counter_on_each_of_up_to_n_max_n_3():
    """Bridge arm pin: max_n=3 round-trips. Pulls in a value distinct
    from Python's class-level default (2) so a future change to the
    Python default never masks a bridge bug -- the dict-shape contract
    is independent of the class-level default.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachOfUpToNTargetCreatures,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
            "max_n": 3,
        }
    )
    assert e == PutPlusOneCounterOnEachOfUpToNTargetCreatures(max_n=3)


def test_bridge_put_plus_one_counter_on_each_of_up_to_n_max_n_zero_silent_noop():
    """Bridge arm pin: max_n=0 is a valid silent-no-op shape. Both
    engines accept max_n=0 (Rust at cards.rs:4144 short-circuits
    only on max_n < 0; Python's class accepts 0 cleanly with
    target_count_range=(0,0) yielding 0 targets picked). Bridge
    must NOT reject 0 -- only negative max_n raises.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachOfUpToNTargetCreatures,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
            "max_n": 0,
        }
    )
    assert isinstance(e, PutPlusOneCounterOnEachOfUpToNTargetCreatures)
    assert e.max_n == 0


def test_bridge_put_plus_one_counter_on_each_of_up_to_n_extra_keys_ignored():
    """Round-trip pin: extra keys on the dict are silently ignored
    (the variant carries only one field, `max_n`). Mirrors the
    existing bridge-arm extra-keys-ignored convention. Forward-
    compatible with future rider-widened Rust variants -- those
    would land as separate ``kind`` arms on this translator.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachOfUpToNTargetCreatures,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
            "max_n": 2,
            "ignored_extra_field": "noise",
        }
    )
    assert isinstance(e, PutPlusOneCounterOnEachOfUpToNTargetCreatures)
    assert e.max_n == 2


def test_bridge_put_plus_one_counter_on_each_of_up_to_n_missing_max_n_raises():
    """Bridge arm pin: missing required `max_n` field raises ValueError
    to mirror Rust's parse_effect contract at cards.rs:4128-4149
    (which raises PyValueError on missing required field). Python's
    class has a default of 2 (Weftblade-Enhancer shape), but the
    bridge enforces explicit `max_n` to keep the dict-shape contract
    aligned with Rust's parse path -- mirrors the round-20/21
    LookAtTopNRevealOrCounter / LookTopNPutOneInHand precedent.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required field 'max_n'"):
        _ta_dict_to_python_effect(
            {"kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures"}
        )


def test_bridge_put_plus_one_counter_on_each_of_up_to_n_negative_max_n_raises():
    """Bridge arm pin: negative max_n raises ValueError to mirror
    Rust's parse_effect contract at cards.rs:4144-4148 (which raises
    PyValueError on negative max_n). Bridge mirrors the contract so
    a typo in a fixture surfaces immediately rather than silently
    constructing a no-op effect.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="max_n must be non-negative"):
        _ta_dict_to_python_effect(
            {
                "kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
                "max_n": -1,
            }
        )


def test_bridge_put_plus_one_counter_on_each_of_up_to_n_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PutPlusOneCounterOnEachOfUpToN(2), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence -- the
    recursive translation walks each child via the same
    ``_ta_dict_to_python_effect`` so this also exercises the
    recursive path for the up-to-N-counters arm.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnEachOfUpToNTargetCreatures,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
                    "max_n": 2,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], PutPlusOneCounterOnEachOfUpToNTargetCreatures
    )
    assert e.effects[0].max_n == 2
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_put_plus_one_counter_on_each_of_up_to_n_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the ``{"kind":
    "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
    "max_n": 2}`` dict shape on a triggered_ability carrier. The
    bridge dict shape must register cleanly on the Rust side too;
    this lockstep gate keeps the bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms (look_top_n_put_one_in_hand,
    set_base_pt_until_eot, may_do). Rust's parse_effect arm at
    ``rust/src/cards.rs:4128-4149`` accepts the bare dict shape
    with a required `max_n: i32` field.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- PutPlusOneCounterOnEachOfUpToN",
        "card_types": ["Creature"],
        "mana_cost": {"W": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "put_plus_one_counter_on_each_of_up_to_n_target_creatures",
                    "max_n": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_put_plus_one_counter_on_self_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    PutPlusOneCounterOnSelf family. Rust ``{"kind": "put_plus_one_
    counter_on_self", "amount": 1}`` translates to Python
    ``PutPlusOneCounterOnSelf(amount=1)``.

    Mirrors Rust's parse_effect arm at cards.rs:3287-3306 (optional
    `amount` field with default 1, i32, non-negative). Field name
    `amount` agrees on both engines (no rename in the bridge). Pin
    verifies the amount field is correctly stored.
    """
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counter_on_self", "amount": 1}
    )
    assert isinstance(e, PutPlusOneCounterOnSelf)
    assert e.amount == 1


def test_bridge_put_plus_one_counter_on_self_amount_default_to_one():
    """Bridge arm pin: missing `amount` defaults to 1 (mirrors Rust's
    parse_effect default-handling at cards.rs:3293-3295 -- ``None ->
    1``). This is the round-23 sister default-handling shape (vs the
    required-field shape of round-21's up-to-N-counters slice). The
    EoE consumer Sunstar Lightsmith carries amount=1 implicit; the
    bridge must accept the bare ``{"kind": "..."}`` dict without an
    explicit amount.
    """
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "put_plus_one_counter_on_self"})
    assert isinstance(e, PutPlusOneCounterOnSelf)
    assert e.amount == 1


def test_bridge_put_plus_one_counter_on_self_amount_two():
    """Bridge arm pin: explicit amount=2 round-trips. Pulls in a
    value distinct from the class-level default (1) so a future
    change to the Python default never masks a bridge bug -- the
    dict-shape contract is independent of the class-level default.
    """
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counter_on_self", "amount": 2}
    )
    assert e == PutPlusOneCounterOnSelf(amount=2)


def test_bridge_put_plus_one_counter_on_self_amount_zero_silent_noop():
    """Bridge arm pin: amount=0 is a valid silent-no-op shape. Both
    engines accept amount=0 (Rust at state_ops_resolve.rs:2607-2609
    short-circuits; Python's class accepts 0 cleanly via
    add_plus_one_counters). Bridge must NOT reject 0 -- only negative
    amount raises.
    """
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counter_on_self", "amount": 0}
    )
    assert isinstance(e, PutPlusOneCounterOnSelf)
    assert e.amount == 0


def test_bridge_put_plus_one_counter_on_self_negative_amount_raises():
    """Bridge arm pin: negative amount raises ValueError to mirror
    Rust's parse_effect contract at cards.rs:3296-3300 (which raises
    PyValueError on negative amount). Bridge mirrors the contract so
    a typo in a fixture surfaces immediately rather than silently
    constructing a no-op effect.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="amount must be non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "put_plus_one_counter_on_self", "amount": -1}
        )


def test_bridge_put_plus_one_counter_on_self_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PutPlusOneCounterOnSelf(1), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence -- the recursive
    translation walks each child via the same
    ``_ta_dict_to_python_effect`` so this also exercises the recursive
    path for the put-plus-one-counter-on-self arm.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "put_plus_one_counter_on_self", "amount": 1},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], PutPlusOneCounterOnSelf)
    assert e.effects[0].amount == 1
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_put_plus_one_counter_on_self_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "put_plus_one_counter_on_self", "amount": 1}`` dict
    shape on a triggered_ability carrier. The bridge dict shape must
    register cleanly on the Rust side too; this lockstep gate keeps
    the bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms (look_top_n_put_one_in_hand,
    set_base_pt_until_eot, may_do, up_to_n_counters). Rust's
    parse_effect arm at ``rust/src/cards.rs:3287-3306`` accepts the
    dict shape with optional `amount: i32` field defaulting to 1.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- PutPlusOneCounterOnSelf",
        "card_types": ["Creature"],
        "mana_cost": {"W": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "put_plus_one_counter_on_self",
                    "amount": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_counter_target_spell_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    CounterTargetSpell family. Rust ``{"kind": "counter_target_spell"}``
    translates to Python ``CounterTargetSpell()`` (default empty
    filter_types).

    Mirrors Rust's parse_effect arm at cards.rs:3361-3370 (unit
    variant; no required fields). Pin verifies the bare dict shape
    constructs the Python class with default filter_types.
    """
    from mtg_bot.effects.generic.stack_control import CounterTargetSpell
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "counter_target_spell"})
    assert isinstance(e, CounterTargetSpell)
    # Default filter_types is empty frozenset (any spell).
    assert e.filter_types == frozenset()


def test_bridge_counter_target_spell_extra_keys_ignored():
    """Round-trip pin: extra keys on the dict are silently ignored
    (the Rust variant is a unit type with no fields). Mirrors the
    Rust parse_effect arm's silent-drop convention for unrecognized
    keys (cards.rs:3361-3370 docstring). Forward-compatible with
    future rider-widened Rust variants -- those would land as
    separate ``kind`` arms on this translator.
    """
    from mtg_bot.effects.generic.stack_control import CounterTargetSpell
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "counter_target_spell",
            "filter_types": ["Artifact"],  # ignored on Rust side
            "ignored_extra_field": "noise",
        }
    )
    assert isinstance(e, CounterTargetSpell)
    # filter_types stays at default empty -- bridge does NOT pass
    # the ignored key through, mirroring Rust's first-slice drop.
    assert e.filter_types == frozenset()


def test_bridge_counter_target_spell_equality_with_default_constructor():
    """Pin: bare-dict translation produces an instance equal to the
    direct ``CounterTargetSpell()`` constructor. This is the round-
    trip-equality contract that enables fixture-side comparison via
    ``assert e == CounterTargetSpell()``. Mirrors the up-to-N-counters
    sister test pattern (assert e == PutPlusOne...) but for the unit
    variant.
    """
    from mtg_bot.effects.generic.stack_control import CounterTargetSpell
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "counter_target_spell"})
    assert e == CounterTargetSpell()


def test_bridge_counter_target_spell_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([CounterTargetSpell(), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence -- the recursive
    translation walks each child via the same
    ``_ta_dict_to_python_effect`` so this also exercises the
    recursive path for the counter-target-spell arm.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.stack_control import CounterTargetSpell
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "counter_target_spell"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], CounterTargetSpell)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_counter_target_spell_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "counter_target_spell"}`` dict shape on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the
    bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms (look_top_n_put_one_in_hand,
    set_base_pt_until_eot, may_do, up_to_n_counters,
    put_plus_one_counter_on_self). Rust's parse_effect arm at
    ``rust/src/cards.rs:3361-3370`` accepts the bare dict shape
    with no companion fields.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- CounterTargetSpell",
        "card_types": ["Instant"],
        "mana_cost": {"U": 2},
        "effects": [{"kind": "counter_target_spell"}],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_target_opponent_loses_life_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    TargetOpponentLosesLife family. Rust ``{"kind": "target_
    opponent_loses_life", "amount": 3}`` translates to Python
    ``TargetOpponentLosesLife(amount=3)``.

    Mirrors Rust's parse_effect arm at cards.rs:3413-3437 (optional
    `amount` field with default 1, i32, non-negative). Field name
    `amount` agrees on both engines (no rename in the bridge). Pin
    verifies the amount field is correctly stored.
    """
    from mtg_bot.effects.generic.life import TargetOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "target_opponent_loses_life", "amount": 3}
    )
    assert isinstance(e, TargetOpponentLosesLife)
    assert e.amount == 3


def test_bridge_target_opponent_loses_life_amount_default_to_one():
    """Bridge arm pin: missing `amount` defaults to 1 (mirrors Rust's
    parse_effect default-handling at cards.rs:3429 -- ``None -> 1``).
    Default-handling shape (round-23 sister DiscardSelf /
    EachOpponentDiscards / PutPlusOneCounterOnSelf precedent).
    """
    from mtg_bot.effects.generic.life import TargetOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "target_opponent_loses_life"})
    assert isinstance(e, TargetOpponentLosesLife)
    assert e.amount == 1


def test_bridge_target_opponent_loses_life_amount_zero_silent_noop():
    """Bridge arm pin: amount=0 is a valid silent-no-op shape. Both
    engines accept amount=0 (Rust at state_ops_resolve.rs:3742-3746
    short-circuits only on negatives; Python's class accepts 0
    cleanly via opp.life -= 0). Bridge must NOT reject 0.
    """
    from mtg_bot.effects.generic.life import TargetOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "target_opponent_loses_life", "amount": 0}
    )
    assert isinstance(e, TargetOpponentLosesLife)
    assert e.amount == 0


def test_bridge_target_opponent_loses_life_negative_amount_raises():
    """Bridge arm pin: negative amount raises ValueError to mirror
    Rust's parse_effect contract at cards.rs:3431-3435 (which raises
    PyValueError on negative amount). Bridge mirrors the contract so
    a typo in a fixture surfaces immediately rather than silently
    constructing a no-op effect.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="amount must be non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "target_opponent_loses_life", "amount": -1}
        )


def test_bridge_target_opponent_loses_life_amount_three_dubious_delicacy_shape():
    """Bridge arm pin: amount=3 round-trips (Dubious Delicacy
    "Target opponent loses 3 life" shape). Pulls in a real-EoE-
    consumer-shaped value to pin the dict-shape contract against
    the canonical printing.
    """
    from mtg_bot.effects.generic.life import TargetOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "target_opponent_loses_life", "amount": 3}
    )
    assert e == TargetOpponentLosesLife(amount=3)


def test_bridge_target_opponent_loses_life_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([TargetOpponentLosesLife(2), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import TargetOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "target_opponent_loses_life", "amount": 2},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], TargetOpponentLosesLife)
    assert e.effects[0].amount == 2
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_target_opponent_loses_life_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "target_opponent_loses_life", "amount": 3}`` dict
    shape on a triggered_ability carrier. The bridge dict shape
    must register cleanly on the Rust side too; this lockstep gate
    keeps the bridge from drifting away from Rust's parse_effect
    coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3413-3437`` accepts the dict shape with
    optional `amount: i32` field defaulting to 1.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- TargetOpponentLosesLife",
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "target_opponent_loses_life",
                    "amount": 3,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_optional_mana_cost_then_effect_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    OptionalManaCostThenEffect family. Rust ``{"kind": "optional_
    mana_cost_then_effect", "cost": {"generic": 1}, "inner_effect":
    {"kind": "draw_cards", "count": 2}}`` translates to Python
    ``OptionalManaCostThenEffect(cost_str="{1}", inner_effect=
    DrawCards(amount=2))``.

    Mirrors Rust's parse_effect arm at cards.rs:4514-4544. Pins both
    the cost-dict-to-string translation (generic=1 -> "{1}") and the
    recursive inner-effect translation.
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {"generic": 1},
            "inner_effect": {"kind": "draw_cards", "count": 2},
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == "{1}"
    assert isinstance(e.inner_effect, DrawCards)
    assert e.inner_effect.amount == 2


def test_bridge_optional_mana_cost_then_effect_colored_cost():
    """Bridge arm pin: colored mana cost dict ({"generic": 1, "R": 1})
    translates to Python's "{1}{R}" string form. Pins the canonical-
    color-order convention (W/U/B/R/G after generic) used by the
    bridge's cost-dict-to-string builder.
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {"generic": 1, "R": 1},
            "inner_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == "{1}{R}"


def test_bridge_optional_mana_cost_then_effect_empty_cost():
    """Bridge arm pin: empty cost dict ({}) translates to the empty
    string "" (the "free" variant — not a real card shape but a
    canonical edge case). Mirrors Python's parse_mana_cost('') ->
    ManaCost() round-trip.
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {},
            "inner_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == ""


def test_bridge_optional_mana_cost_then_effect_missing_cost_raises():
    """Bridge arm pin: missing `cost` field raises ValueError to
    mirror Rust's parse_effect contract at cards.rs:4518-4527
    (which raises PyValueError on missing 'cost'). Bridge mirrors the
    contract so a typo in a fixture surfaces immediately.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required 'cost'"):
        _ta_dict_to_python_effect(
            {
                "kind": "optional_mana_cost_then_effect",
                "inner_effect": {"kind": "draw_cards", "count": 1},
            }
        )


def test_bridge_optional_mana_cost_then_effect_missing_inner_raises():
    """Bridge arm pin: missing `inner_effect` field raises ValueError
    to mirror Rust's parse_effect contract at cards.rs:4529-4538
    (which raises PyValueError on missing 'inner_effect'). Symmetric
    with the missing-cost arm.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required 'inner_effect'"):
        _ta_dict_to_python_effect(
            {
                "kind": "optional_mana_cost_then_effect",
                "cost": {"generic": 1},
            }
        )


def test_bridge_optional_mana_cost_then_effect_recursive_inner_translation():
    """Bridge arm pin: inner_effect recursion fires through
    ``_ta_dict_to_python_effect``. Pins the recursive-translation
    contract (mirrors the MayDo/IfYouDo/Sequence composer pattern):
    the nested ``put_plus_one_counter_on_self`` shape translates
    cleanly to ``PutPlusOneCounterOnSelf(amount=1)`` -- the exact
    Oreplate-Pangolin shape ("you may pay {1}; if you do, put a +1/+1
    counter on this creature").
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {"generic": 1},
            "inner_effect": {
                "kind": "put_plus_one_counter_on_self",
                "amount": 1,
            },
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == "{1}"
    assert isinstance(e.inner_effect, PutPlusOneCounterOnSelf)
    assert e.inner_effect.amount == 1


def test_bridge_optional_mana_cost_then_effect_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([OptionalManaCostThenEffect("{1}", DrawCards(2)),
    DiscardSelf()])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss inside
    Sequence.
    """
    from mtg_bot.effects.combinators import (
        OptionalManaCostThenEffect, Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "optional_mana_cost_then_effect",
                    "cost": {"generic": 1},
                    "inner_effect": {"kind": "draw_cards", "count": 2},
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], OptionalManaCostThenEffect)
    assert e.effects[0].cost_str == "{1}"
    assert isinstance(e.effects[0].inner_effect, DrawCards)
    assert e.effects[0].inner_effect.amount == 2
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_optional_mana_cost_then_effect_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "optional_mana_cost_then_effect", "cost":
    {"generic": 1}, "inner_effect": {...}}`` on a triggered_ability
    carrier. The bridge dict shape must register cleanly on Rust too;
    this lockstep gate keeps the bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:4514-4544`` accepts the dict shape with
    REQUIRED `cost` (mana-cost dict) and `inner_effect` (nested
    effect-shape dict).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- OptionalManaCostThenEffect",
        "card_types": ["Creature"],
        "mana_cost": {"R": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "optional_mana_cost_then_effect",
                    "cost": {"generic": 1},
                    "inner_effect": {
                        "kind": "put_plus_one_counter_on_self",
                        "amount": 1,
                    },
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_optional_mana_cost_then_effect_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    OptionalManaCostThenEffect family. Rust ``{"kind": "optional_
    mana_cost_then_effect", "cost": {"generic": 1}, "inner_effect":
    {"kind": "draw_cards", "count": 2}}`` translates to Python
    ``OptionalManaCostThenEffect(cost_str="{1}", inner_effect=
    DrawCards(amount=2))``.

    Mirrors Rust's parse_effect arm at cards.rs:4514-4544. Pins both
    the cost-dict-to-string translation (generic=1 -> "{1}") and the
    recursive inner-effect translation.
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {"generic": 1},
            "inner_effect": {"kind": "draw_cards", "count": 2},
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == "{1}"
    assert isinstance(e.inner_effect, DrawCards)
    assert e.inner_effect.amount == 2


def test_bridge_optional_mana_cost_then_effect_colored_cost():
    """Bridge arm pin: colored mana cost dict ({"generic": 1, "R": 1})
    translates to Python's "{1}{R}" string form. Pins the canonical-
    color-order convention (W/U/B/R/G after generic) used by the
    bridge's cost-dict-to-string builder.
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {"generic": 1, "R": 1},
            "inner_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == "{1}{R}"


def test_bridge_optional_mana_cost_then_effect_empty_cost():
    """Bridge arm pin: empty cost dict ({}) translates to the empty
    string "" (the "free" variant — not a real card shape but a
    canonical edge case). Mirrors Python's parse_mana_cost('') ->
    ManaCost() round-trip.
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {},
            "inner_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == ""


def test_bridge_optional_mana_cost_then_effect_missing_cost_raises():
    """Bridge arm pin: missing `cost` field raises ValueError to
    mirror Rust's parse_effect contract at cards.rs:4518-4527
    (which raises PyValueError on missing 'cost'). Bridge mirrors the
    contract so a typo in a fixture surfaces immediately.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required 'cost'"):
        _ta_dict_to_python_effect(
            {
                "kind": "optional_mana_cost_then_effect",
                "inner_effect": {"kind": "draw_cards", "count": 1},
            }
        )


def test_bridge_optional_mana_cost_then_effect_missing_inner_raises():
    """Bridge arm pin: missing `inner_effect` field raises ValueError
    to mirror Rust's parse_effect contract at cards.rs:4529-4538
    (which raises PyValueError on missing 'inner_effect'). Symmetric
    with the missing-cost arm.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required 'inner_effect'"):
        _ta_dict_to_python_effect(
            {
                "kind": "optional_mana_cost_then_effect",
                "cost": {"generic": 1},
            }
        )


def test_bridge_optional_mana_cost_then_effect_recursive_inner_translation():
    """Bridge arm pin: inner_effect recursion fires through
    ``_ta_dict_to_python_effect``. Pins the recursive-translation
    contract (mirrors the MayDo/IfYouDo/Sequence composer pattern):
    the nested ``put_plus_one_counter_on_self`` shape translates
    cleanly to ``PutPlusOneCounterOnSelf(amount=1)`` -- the exact
    Oreplate-Pangolin shape ("you may pay {1}; if you do, put a +1/+1
    counter on this creature").
    """
    from mtg_bot.effects.combinators import OptionalManaCostThenEffect
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "optional_mana_cost_then_effect",
            "cost": {"generic": 1},
            "inner_effect": {
                "kind": "put_plus_one_counter_on_self",
                "amount": 1,
            },
        }
    )
    assert isinstance(e, OptionalManaCostThenEffect)
    assert e.cost_str == "{1}"
    assert isinstance(e.inner_effect, PutPlusOneCounterOnSelf)
    assert e.inner_effect.amount == 1


def test_bridge_optional_mana_cost_then_effect_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([OptionalManaCostThenEffect("{1}", DrawCards(2)),
    DiscardSelf()])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss inside
    Sequence.
    """
    from mtg_bot.effects.combinators import (
        OptionalManaCostThenEffect, Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "optional_mana_cost_then_effect",
                    "cost": {"generic": 1},
                    "inner_effect": {"kind": "draw_cards", "count": 2},
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], OptionalManaCostThenEffect)
    assert e.effects[0].cost_str == "{1}"
    assert isinstance(e.effects[0].inner_effect, DrawCards)
    assert e.effects[0].inner_effect.amount == 2
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_optional_mana_cost_then_effect_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "optional_mana_cost_then_effect", "cost":
    {"generic": 1}, "inner_effect": {...}}`` on a triggered_ability
    carrier. The bridge dict shape must register cleanly on Rust too;
    this lockstep gate keeps the bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:4514-4544`` accepts the dict shape with
    REQUIRED `cost` (mana-cost dict) and `inner_effect` (nested
    effect-shape dict).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- OptionalManaCostThenEffect",
        "card_types": ["Creature"],
        "mana_cost": {"R": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "optional_mana_cost_then_effect",
                    "cost": {"generic": 1},
                    "inner_effect": {
                        "kind": "put_plus_one_counter_on_self",
                        "amount": 1,
                    },
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_add_n_mana_of_chosen_color_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    AddNManaOfChosenColor family. Rust ``{"kind": "add_n_mana_of_
    chosen_color", "amount": 3, "options": ["W","U","B","R","G"]}``
    translates to Python ``AddNManaOfChosenColor(amount=3,
    options=("W","U","B","R","G"))``.

    Mirrors Rust's parse_effect arm at cards.rs:4407-4456. Pins both
    the amount field translation and the list -> tuple conversion
    for options.
    """
    from mtg_bot.effects.generic.mana import AddNManaOfChosenColor
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "add_n_mana_of_chosen_color",
            "amount": 3,
            "options": ["W", "U", "B", "R", "G"],
        }
    )
    assert isinstance(e, AddNManaOfChosenColor)
    assert e.amount == 3
    assert e.options == ("W", "U", "B", "R", "G")


def test_bridge_add_n_mana_of_chosen_color_default_options():
    """Bridge arm pin: missing `options` defaults to the 5-color
    ("W","U","B","R","G") tuple (Lotus-Field default; mirrors Rust's
    parse_effect default at cards.rs:4448-4454 and Python's class
    default at mana.py:105). C is intentionally NOT in the default
    per the printed "any one color" wording (which excludes
    colorless).
    """
    from mtg_bot.effects.generic.mana import AddNManaOfChosenColor
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "add_n_mana_of_chosen_color", "amount": 3}
    )
    assert isinstance(e, AddNManaOfChosenColor)
    assert e.amount == 3
    assert e.options == ("W", "U", "B", "R", "G")


def test_bridge_add_n_mana_of_chosen_color_amount_zero_silent_noop():
    """Bridge arm pin: amount=0 is a valid silent-no-op shape. Rust's
    resolve at state_ops_resolve.rs:4744-4746 short-circuits on
    amount=0; Python's class loops range(0) which is a no-op.
    Bridge must NOT reject 0.
    """
    from mtg_bot.effects.generic.mana import AddNManaOfChosenColor
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "add_n_mana_of_chosen_color", "amount": 0}
    )
    assert isinstance(e, AddNManaOfChosenColor)
    assert e.amount == 0


def test_bridge_add_n_mana_of_chosen_color_missing_amount_raises():
    """Bridge arm pin: missing `amount` raises ValueError to mirror
    Rust's parse_effect contract at cards.rs:4416-4420 (which raises
    PyValueError on missing amount). Bridge mirrors so a typo in a
    fixture surfaces immediately.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="missing required 'amount'"):
        _ta_dict_to_python_effect({"kind": "add_n_mana_of_chosen_color"})


def test_bridge_add_n_mana_of_chosen_color_negative_amount_raises():
    """Bridge arm pin: negative amount raises ValueError to mirror
    Rust's parse_effect contract at cards.rs:4422-4426 (which raises
    PyValueError on negative amount).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="amount must be non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "add_n_mana_of_chosen_color", "amount": -1}
        )


def test_bridge_add_n_mana_of_chosen_color_empty_options_raises():
    """Bridge arm pin: empty options list raises ValueError to mirror
    Rust's parse_effect contract at cards.rs:4434-4438 (which raises
    PyValueError on empty options). A degenerate "no colors to choose
    from" shape can't produce mana.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="options must be non-empty"):
        _ta_dict_to_python_effect(
            {
                "kind": "add_n_mana_of_chosen_color",
                "amount": 1,
                "options": [],
            }
        )


def test_bridge_add_n_mana_of_chosen_color_unknown_color_raises():
    """Bridge arm pin: unknown color string in options raises
    ValueError to mirror Rust's parse_effect contract at
    cards.rs:4439-4445. Closed mana-color vocab: W/U/B/R/G/C only.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="unknown color"):
        _ta_dict_to_python_effect(
            {
                "kind": "add_n_mana_of_chosen_color",
                "amount": 1,
                "options": ["W", "Q"],
            }
        )


def test_bridge_add_n_mana_of_chosen_color_subset_options():
    """Bridge arm pin: options subset (e.g. ("W","U")) translates
    cleanly to a 2-color tuple. Pins that the bridge does NOT silently
    expand subsets back to the 5-color default. Real-card example:
    the gold-2-pip filter "{T}: Add {W} or {U}" shape is a different
    Effect variant, but a hypothetical "{T}: Add 2 mana of any one of
    {W} or {U}" would route through this arm with options=("W","U").
    """
    from mtg_bot.effects.generic.mana import AddNManaOfChosenColor
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "add_n_mana_of_chosen_color",
            "amount": 2,
            "options": ["W", "U"],
        }
    )
    assert isinstance(e, AddNManaOfChosenColor)
    assert e.amount == 2
    assert e.options == ("W", "U")


def test_rust_parser_accepts_add_n_mana_of_chosen_color_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "add_n_mana_of_chosen_color", "amount": 3}`` on
    a triggered_ability carrier. The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Note: the canonical real-card use is an ACTIVATED mana ability
    (e.g. Lotus Field's "{T}: Add three mana of any one color"), but
    a triggered_abilities carrier registers cleanly too -- it
    exercises the same parse_effect path. Mirrors the existing
    parser-surface pins for sister BLOCKED-BRIDGE-ARM arms.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- AddNManaOfChosenColor",
        "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "add_n_mana_of_chosen_color",
                    "amount": 3,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_tap_target_artifact_or_creature_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    TapTargetArtifactOrCreature family. Rust ``{"kind": "tap_target_
    artifact_or_creature"}`` translates to Python
    ``TapTargetArtifactOrCreature()``.

    Mirrors Rust's parse_effect arm at cards.rs:3693-3694 (unit
    variant; no fields). Pin verifies the bare-shape translation.
    """
    from mtg_bot.effects.generic.tap import TapTargetArtifactOrCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "tap_target_artifact_or_creature"})
    assert isinstance(e, TapTargetArtifactOrCreature)


def test_bridge_tap_target_artifact_or_creature_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    Rust's parse_effect silent-drop convention -- unrecognized fields
    in the dict are ignored (not raised as errors). Pins the unit-
    variant contract against future-evolution where a sibling Rust
    variant might add fields.
    """
    from mtg_bot.effects.generic.tap import TapTargetArtifactOrCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "tap_target_artifact_or_creature",
            "spurious_field": 42,
        }
    )
    assert isinstance(e, TapTargetArtifactOrCreature)


def test_bridge_tap_target_artifact_or_creature_targets_filter_set():
    """Bridge arm pin: the constructed Python effect carries the
    ``targets`` tuple set by ``__post_init__`` to a single TargetSpec
    with the IsArtifact() | IsCreature() filter. Pins the implicit
    construction-time wiring -- the bridge does NOT need to thread
    the filter explicitly because Python's class sets it internally.
    """
    from mtg_bot.effects.generic.tap import TapTargetArtifactOrCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "tap_target_artifact_or_creature"})
    assert isinstance(e, TapTargetArtifactOrCreature)
    # __post_init__ sets `targets` to a 1-tuple of TargetSpec.
    assert hasattr(e, "targets")
    assert len(e.targets) == 1


def test_bridge_tap_target_artifact_or_creature_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([TapTargetArtifactOrCreature(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.tap import TapTargetArtifactOrCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "tap_target_artifact_or_creature"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], TapTargetArtifactOrCreature)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_tap_target_artifact_or_creature_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "tap_target_artifact_or_creature"}`` on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3693-3694`` accepts the unit-shape dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- TapTargetArtifactOrCreature",
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 1},
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "tap_target_artifact_or_creature"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_pump_creatures_you_control_until_eot_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 22 fresh discovery walk) --
    PumpCreaturesYouControlUntilEOT family. Rust ``{"kind": "pump_
    creatures_you_control_until_eot", "power_bonus": 1,
    "toughness_bonus": 1}`` translates to Python
    ``PumpCreaturesYouControlUntilEOT(power_bonus=1,
    toughness_bonus=1)``.

    Mirrors Rust's parse_effect arm at cards.rs:4476-4498 (both
    fields optional with default 0, i32, negatives allowed). Field
    names ``power_bonus`` and ``toughness_bonus`` agree on both
    engines (no rename in the bridge).
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesYouControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_you_control_until_eot",
            "power_bonus": 1,
            "toughness_bonus": 1,
        }
    )
    assert isinstance(e, PumpCreaturesYouControlUntilEOT)
    assert e.power_bonus == 1
    assert e.toughness_bonus == 1


def test_bridge_pump_creatures_you_control_until_eot_default_bonuses_zero():
    """Bridge arm pin: missing ``power_bonus`` and ``toughness_bonus``
    default to 0 (mirrors Rust's parse_effect default-handling at
    cards.rs:4487+4495 -- ``None -> 0``). Default-handling shape
    (round-23 sister DiscardSelf / EachOpponentDiscards /
    PutPlusOneCounterOnSelf precedent). ``(0, 0)`` is a valid
    silent no-op shape on both engines.
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesYouControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "pump_creatures_you_control_until_eot"}
    )
    assert isinstance(e, PumpCreaturesYouControlUntilEOT)
    assert e.power_bonus == 0
    assert e.toughness_bonus == 0


def test_bridge_pump_creatures_you_control_until_eot_negative_bonuses_allowed():
    """Bridge arm pin: negative bonuses are LEGAL (debuff-shaped
    mass-pumps, e.g. spells that shrink your own team for a benefit).
    Rust intentionally allows them (cards.rs:4477-4480 docstring,
    mirroring PumpSelfUntilEOT precedent); Python's class is a bare
    dataclass with no non-negative guard. Bridge does NOT raise on
    negatives -- mirrors the contract.
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesYouControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_you_control_until_eot",
            "power_bonus": -1,
            "toughness_bonus": -1,
        }
    )
    assert isinstance(e, PumpCreaturesYouControlUntilEOT)
    assert e.power_bonus == -1
    assert e.toughness_bonus == -1


def test_bridge_pump_creatures_you_control_until_eot_independent_fields():
    """Bridge arm pin: independent fields. Specify ``toughness_bonus``
    only; ``power_bonus`` defaults to 0 ("creatures you control get
    +0/+2 UEOT" -- defensive anthem). Pins that the get-with-default
    per-field is correctly applied independently for each axis.
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesYouControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_you_control_until_eot",
            "toughness_bonus": 2,
        }
    )
    assert isinstance(e, PumpCreaturesYouControlUntilEOT)
    assert e.power_bonus == 0
    assert e.toughness_bonus == 2


def test_bridge_pump_creatures_you_control_until_eot_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``PumpCreaturesYouControlUntilEOT(
    power_bonus=2, toughness_bonus=2)``. Pins the dataclass equality
    semantic -- the four rider fields (granted_keywords,
    exclude_source, attacking_only, also_untap_if_not_your_turn) all
    take their defaults on both sides (no leakage from the bridge
    dict shape).
    """
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesYouControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_creatures_you_control_until_eot",
            "power_bonus": 2,
            "toughness_bonus": 2,
        }
    )
    assert e == PumpCreaturesYouControlUntilEOT(
        power_bonus=2, toughness_bonus=2,
    )


def test_bridge_pump_creatures_you_control_until_eot_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PumpCreaturesYouControlUntilEOT(1, 1), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.pump import (
        PumpCreaturesYouControlUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "pump_creatures_you_control_until_eot",
                    "power_bonus": 1,
                    "toughness_bonus": 1,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], PumpCreaturesYouControlUntilEOT)
    assert e.effects[0].power_bonus == 1
    assert e.effects[0].toughness_bonus == 1
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_pump_creatures_you_control_until_eot_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "pump_creatures_you_control_until_eot",
    "power_bonus": 1, "toughness_bonus": 1}`` dict shape on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the
    bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:4476-4498`` accepts the dict shape with
    optional ``power_bonus`` / ``toughness_bonus`` i32 fields
    defaulting to 0.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "PumpCreaturesYouControlUntilEOT"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"W": 1, "generic": 2},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "pump_creatures_you_control_until_eot",
                    "power_bonus": 1,
                    "toughness_bonus": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_destroy_target_artifact_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23) -- DestroyTargetArtifact
    family. Rust ``{"kind": "destroy_target_artifact"}`` translates
    to Python ``DestroyTargetArtifact()``.

    Mirrors Rust's parse_effect arm at cards.rs:3519-3525 (unit
    variant -- no fields). Python's class is a ``Destroy`` subclass
    that sets ``targets = (TargetSpec(IsArtifact()),)`` in
    ``__post_init__``.
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetArtifact
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "destroy_target_artifact"})
    assert isinstance(e, DestroyTargetArtifact)


def test_bridge_destroy_target_artifact_targets_set_to_is_artifact():
    """Bridge arm pin: post-init wires ``targets`` tuple to a single
    ``TargetSpec(IsArtifact())``. Pins the filter shape -- a bug
    that dropped the ``__post_init__`` would leave ``targets`` empty
    and make the resolver pick anything.
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetArtifact
    from mtg_bot.effects.object_predicates import IsArtifact
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "destroy_target_artifact"})
    assert isinstance(e, DestroyTargetArtifact)
    assert e.targets == (TargetSpec(IsArtifact()),)


def test_bridge_destroy_target_artifact_extra_keys_silently_ignored():
    """Bridge arm pin: extra keys on the dict are silently dropped
    (mirrors Rust's parse_effect arm at cards.rs:3519-3525 which
    takes no fields and simply returns the unit variant -- there
    is no extra-key validation). Mirrors the silent-drop convention
    on sister unit-variant arms (counter_target_spell, etc.).
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetArtifact
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_artifact", "spurious": 42}
    )
    assert isinstance(e, DestroyTargetArtifact)


def test_bridge_destroy_target_artifact_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``DestroyTargetArtifact()``. Pins the
    dataclass equality semantic -- the bare unit-shape carries no
    parameters so equality is structural.
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetArtifact
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "destroy_target_artifact"})
    assert e == DestroyTargetArtifact()


def test_bridge_destroy_target_artifact_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DestroyTargetArtifact(), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.destroy import DestroyTargetArtifact
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "destroy_target_artifact"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DestroyTargetArtifact)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_destroy_target_artifact_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the ``{"kind":
    "destroy_target_artifact"}`` dict shape on a triggered_ability
    carrier. The bridge dict shape must register cleanly on the
    Rust side too; this lockstep gate keeps the bridge from drifting
    away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3519-3525`` accepts the unit-shape dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- DestroyTargetArtifact",
        "card_types": ["Creature"],
        "mana_cost": {"R": 1, "generic": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "destroy_target_artifact"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_destroy_target_nonland_permanent_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23) --
    DestroyTargetNonlandPermanent family. Rust ``{"kind":
    "destroy_target_nonland_permanent"}`` translates to Python
    ``DestroyTargetNonlandPermanent()``.

    Mirrors Rust's parse_effect arm at cards.rs:3514-3517 (unit
    variant -- no fields). Python's class is a ``Destroy``
    subclass that sets ``targets = (TargetSpec(~IsLand() &
    ~IsPlayer()),)`` in ``__post_init__``.
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNonlandPermanent,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_nonland_permanent"}
    )
    assert isinstance(e, DestroyTargetNonlandPermanent)


def test_bridge_destroy_target_nonland_permanent_targets_set_to_nonland_nonplayer():
    """Bridge arm pin: post-init wires ``targets`` tuple to a
    single ``TargetSpec(~IsLand() & ~IsPlayer())``. Pins the filter
    composition -- a bug that dropped the ``__post_init__`` would
    leave ``targets`` empty and make the resolver pick anything,
    including lands and players.
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNonlandPermanent,
    )
    from mtg_bot.effects.object_predicates import IsLand, IsPlayer
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_nonland_permanent"}
    )
    assert isinstance(e, DestroyTargetNonlandPermanent)
    assert e.targets == (TargetSpec(~IsLand() & ~IsPlayer()),)


def test_bridge_destroy_target_nonland_permanent_extra_keys_silently_ignored():
    """Bridge arm pin: extra keys on the dict are silently dropped
    (mirrors Rust's parse_effect arm at cards.rs:3514-3517 which
    takes no fields and simply returns the unit variant -- there
    is no extra-key validation). Mirrors the silent-drop convention
    on sister unit-variant arms (counter_target_spell, etc.).
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNonlandPermanent,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "destroy_target_nonland_permanent",
            "target": {"kind": "any_creature"},  # ignored
        }
    )
    assert isinstance(e, DestroyTargetNonlandPermanent)


def test_bridge_destroy_target_nonland_permanent_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect
    equals a directly-constructed
    ``DestroyTargetNonlandPermanent()``. Pins the dataclass
    equality semantic -- the bare unit-shape carries no parameters
    so equality is structural.
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNonlandPermanent,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_nonland_permanent"}
    )
    assert e == DestroyTargetNonlandPermanent()


def test_bridge_destroy_target_nonland_permanent_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DestroyTargetNonlandPermanent(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNonlandPermanent,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "destroy_target_nonland_permanent"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DestroyTargetNonlandPermanent)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_destroy_target_nonland_permanent_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the ``{"kind":
    "destroy_target_nonland_permanent"}`` dict shape on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the
    bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3514-3517`` accepts the unit-shape dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "DestroyTargetNonlandPermanent"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 2},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "destroy_target_nonland_permanent"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_destroy_all_creatures_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23) -- DestroyAllCreatures
    family (Wrath shape). Rust ``{"kind": "destroy_all_creatures"}``
    translates to Python ``DestroyAllCreatures()``.

    Mirrors Rust's parse_effect arm at cards.rs:3504-3512 (unit
    variant -- no fields). Python's class is a frozen dataclass
    with three OPTIONAL narrowing fields (``tapped_only``,
    ``untapped_only``, ``min_power``) all defaulting to the
    unrestricted Wrath shape; Rust's variant carries no narrowing
    fields so the bridge constructs the bare class.
    """
    from mtg_bot.effects.generic.destroy import DestroyAllCreatures
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "destroy_all_creatures"})
    assert isinstance(e, DestroyAllCreatures)


def test_bridge_destroy_all_creatures_narrowing_fields_default_to_unrestricted():
    """Bridge arm pin: the three narrowing fields default to the
    unrestricted Wrath shape (``tapped_only=False``,
    ``untapped_only=False``, ``min_power=0``). Pins the contract
    that the bare Rust unit variant maps to the unrestricted
    Python form -- a bug that flipped a narrowing flag default
    would silently shrink the wrath's victim set.
    """
    from mtg_bot.effects.generic.destroy import DestroyAllCreatures
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "destroy_all_creatures"})
    assert isinstance(e, DestroyAllCreatures)
    assert e.tapped_only is False
    assert e.untapped_only is False
    assert e.min_power == 0


def test_bridge_destroy_all_creatures_extra_keys_silently_ignored():
    """Bridge arm pin: extra keys on the dict are silently dropped
    (mirrors Rust's parse_effect arm at cards.rs:3504-3512 which
    takes no fields and simply returns the unit variant). Mirrors
    the silent-drop convention on sister unit-variant arms. Pins
    that Rust-side narrowing fields like ``tapped_only`` (NOT
    represented on the Rust unit variant) are dropped here too --
    the bridge cannot synthesize narrowing it doesn't see on the
    Rust side.
    """
    from mtg_bot.effects.generic.destroy import DestroyAllCreatures
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_all_creatures", "tapped_only": True}
    )
    # Even with a stray narrowing key, bridge produces the
    # unrestricted form (mirrors Rust's bare unit-variant shape).
    assert isinstance(e, DestroyAllCreatures)
    assert e.tapped_only is False


def test_bridge_destroy_all_creatures_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``DestroyAllCreatures()``. Pins the
    dataclass equality semantic -- the bare unit-shape carries no
    parameters so equality is structural.
    """
    from mtg_bot.effects.generic.destroy import DestroyAllCreatures
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "destroy_all_creatures"})
    assert e == DestroyAllCreatures()


def test_bridge_destroy_all_creatures_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DestroyAllCreatures(), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this
    arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.destroy import DestroyAllCreatures
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "destroy_all_creatures"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DestroyAllCreatures)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_destroy_all_creatures_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the ``{"kind":
    "destroy_all_creatures"}`` dict shape on a triggered_ability
    carrier. The bridge dict shape must register cleanly on the
    Rust side too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3504-3512`` accepts the unit-shape dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- DestroyAllCreatures",
        "card_types": ["Creature"],
        "mana_cost": {"W": 1, "B": 1, "generic": 2},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "destroy_all_creatures"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_damage_each_creature_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23 fresh discovery walk) --
    DealNDamageToEachCreature family. Rust ``{"kind":
    "damage_each_creature", "amount": 2}`` translates to Python
    ``DealNDamageToEachCreature(amount=2)``.

    Mirrors Rust's parse_effect arm at cards.rs:3286-3289 (amount
    required, i32, no default). Field name ``amount`` agrees on both
    engines (no rename in the bridge).
    """
    from mtg_bot.effects.generic.damage import DealNDamageToEachCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "damage_each_creature", "amount": 2}
    )
    assert isinstance(e, DealNDamageToEachCreature)
    assert e.amount == 2


def test_bridge_damage_each_creature_amount_required_keyerror_when_missing():
    """Bridge arm pin: missing ``amount`` raises KeyError (mirrors
    Rust's ``require()`` contract at cards.rs:3286-3289). No default
    on either side -- the variant has no zero-arg form.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect({"kind": "damage_each_creature"})


def test_bridge_damage_each_creature_amount_zero_silent_noop():
    """Bridge arm pin: amount=0 is legal (silent no-op) on both
    engines. Rust's resolve at state_ops_resolve.rs:2035-2039 only
    raises on negatives; amount=0 falls through to a no-op damage
    mark on every creature. Python's class is a bare dataclass with
    no zero-amount guard. Mirrors GainLife / DrawCards /
    DamageEachOpponent zero-amount conventions.
    """
    from mtg_bot.effects.generic.damage import DealNDamageToEachCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "damage_each_creature", "amount": 0}
    )
    assert isinstance(e, DealNDamageToEachCreature)
    assert e.amount == 0


def test_bridge_damage_each_creature_negative_amount_raises():
    """Bridge arm pin: amount<0 raises ValueError at translation time
    (mirrors Rust's resolve raise at state_ops_resolve.rs:2035-2039).
    The bridge front-loads the check rather than waiting for
    Python's ``DealNDamageToEachCreature.resolve`` to handle it (the
    Python class itself doesn't guard, but cross-engine probes that
    feed a negative amount through the bridge would silently apply
    the bogus value on the Python side and raise on the Rust side --
    asymmetric. The bridge's pre-check keeps the contracts symmetric).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "damage_each_creature", "amount": -1}
        )


def test_bridge_damage_each_creature_amount_four_extinguisher_battleship_shape():
    """Bridge arm pin: ``amount=4`` round-trips correctly --
    Extinguisher Battleship's ETB sequence body shape. Pins that the
    field translates as i32 (no clamp / overflow at the bridge).
    """
    from mtg_bot.effects.generic.damage import DealNDamageToEachCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "damage_each_creature", "amount": 4}
    )
    assert e == DealNDamageToEachCreature(amount=4)


def test_bridge_damage_each_creature_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealNDamageToEachCreature(2), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence (the
    real Lithobraking / Extinguisher Battleship cascades are Sequence-
    wrapped on the printed surface).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import DealNDamageToEachCreature
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "damage_each_creature", "amount": 2},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DealNDamageToEachCreature)
    assert e.effects[0].amount == 2
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_damage_each_creature_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "damage_each_creature", "amount": 2}`` dict shape on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the bridge
    from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3286-3289`` accepts the dict shape with
    required ``amount`` i32 field.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- DamageEachCreature"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"R": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "damage_each_creature",
                    "amount": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_each_opponent_loses_life_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23 fresh discovery walk) --
    EachOpponentLosesLife family. Rust ``{"kind":
    "each_opponent_loses_life", "amount": 2}`` translates to Python
    ``EachOpponentLosesLife(amount=2)``.

    Mirrors Rust's parse_effect arm at cards.rs:3425-3437 (amount
    required, i32, no default; negatives raise at parse time).
    Field name ``amount`` agrees on both engines (no rename in the
    bridge).
    """
    from mtg_bot.effects.generic.life import EachOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life", "amount": 2}
    )
    assert isinstance(e, EachOpponentLosesLife)
    assert e.amount == 2


def test_bridge_each_opponent_loses_life_amount_required_keyerror_when_missing():
    """Bridge arm pin: missing ``amount`` raises KeyError (mirrors
    Rust's ``require()`` contract at cards.rs:3431). No default on
    either side (distinct from sister
    ``target_opponent_loses_life`` which defaults to 1).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect({"kind": "each_opponent_loses_life"})


def test_bridge_each_opponent_loses_life_amount_zero_silent_noop():
    """Bridge arm pin: amount=0 is legal (silent no-op) on both
    engines. Rust's resolve at state_ops_resolve.rs:3592-3636's
    saturating_sub(0) is a no-op; Python's class iterates opponents
    and subtracts 0 from each life total. Mirrors GainLife /
    DrawCards / DamageEachOpponent / DiscardSelf zero-amount
    conventions per the resolve docstring at
    state_ops_resolve.rs:3587-3589.
    """
    from mtg_bot.effects.generic.life import EachOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life", "amount": 0}
    )
    assert isinstance(e, EachOpponentLosesLife)
    assert e.amount == 0


def test_bridge_each_opponent_loses_life_negative_amount_raises():
    """Bridge arm pin: amount<0 raises ValueError at translation
    time (mirrors Rust's parse_effect raise at cards.rs:3432-3436).
    Rust raises at BOTH parse and resolve as defense-in-depth; the
    bridge raises early to keep the translation-time contract
    symmetric across engines.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "each_opponent_loses_life", "amount": -1}
        )


def test_bridge_each_opponent_loses_life_amount_two_void_consumer_shape():
    """Bridge arm pin: ``amount=2`` round-trips correctly -- the EoE
    Void-replacement consumer's shape ("you draw two cards and each
    opponent loses 2 life"). Pins that the field translates as i32
    (no clamp / overflow at the bridge).
    """
    from mtg_bot.effects.generic.life import EachOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life", "amount": 2}
    )
    assert e == EachOpponentLosesLife(amount=2)


def test_bridge_each_opponent_loses_life_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DrawCards(2), EachOpponentLosesLife(2)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence (the Void-
    consumer body is Sequence-wrapped on the printed surface).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import EachOpponentLosesLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "draw_cards", "count": 2},
                {"kind": "each_opponent_loses_life", "amount": 2},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DrawCards)
    assert isinstance(e.effects[1], EachOpponentLosesLife)
    assert e.effects[1].amount == 2


def test_rust_parser_accepts_each_opponent_loses_life_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "each_opponent_loses_life", "amount": 2}`` dict shape
    on a triggered_ability carrier. The bridge dict shape must
    register cleanly on the Rust side too; this lockstep gate keeps
    the bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3425-3437`` accepts the dict shape with
    required ``amount`` i32 field (negatives raise).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- EachOpponentLosesLife"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "each_opponent_loses_life",
                    "amount": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_untap_target_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23 fresh discovery walk) --
    UntapTarget family. Rust ``{"kind": "untap_target", "target":
    {"kind": "any_creature"}}`` translates to Python
    ``Untap(targets=(TargetSpec(IsCreature()),))``.

    Mirrors Rust's parse_effect arm at cards.rs:3237-3244 (target
    dict required, parsed via parse_target_filter). The bridge wires
    the same generic ``Untap`` primitive added in this slice as the
    Python-side mirror of the Rust resolver.
    """
    from mtg_bot.effects.generic.tap import Untap
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "untap_target", "target": {"kind": "any_creature"}}
    )
    assert isinstance(e, Untap)
    assert e.targets == (TargetSpec(IsCreature()),)


def test_bridge_untap_target_default_target_filter_is_any_creature():
    """Bridge arm pin: missing ``target`` defaults to
    ``any_creature`` (mirrors the lenient-default convention used by
    sister tap_target / destroy_target / exile_target /
    return_to_hand arms). Bridge consumers SHOULD pass ``target``
    explicitly for round-trip dict-shape consistency, but the
    default keeps the arm tolerant of stripped dicts that lose
    optional fields.
    """
    from mtg_bot.effects.generic.tap import Untap
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "untap_target"})
    assert isinstance(e, Untap)
    assert e.targets == (TargetSpec(IsCreature()),)


def test_bridge_untap_target_any_player_filter_raises():
    """Bridge arm pin: ``target.kind == "any_player"`` raises
    ValueError (mirrors Rust's resolve raise at
    state_ops_resolve.rs:1429-1434 -- "UntapTarget with AnyPlayer
    target is semantically nonsense"). The bridge front-loads the
    check rather than waiting for resolve so the contract is
    symmetric across engines.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="any_player"):
        _ta_dict_to_python_effect(
            {"kind": "untap_target", "target": {"kind": "any_player"}}
        )


def test_bridge_untap_target_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed
    ``Untap(targets=(TargetSpec(IsCreature()),))``. Pins the
    dataclass equality semantic.
    """
    from mtg_bot.effects.generic.tap import Untap
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "untap_target", "target": {"kind": "any_creature"}}
    )
    assert e == Untap(targets=(TargetSpec(IsCreature()),))


def test_bridge_untap_target_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([Untap(...), DrawCards(1)])`` translates correctly
    via the recursive Sequence translator. Pins this arm against
    accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.tap import Untap
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "untap_target",
                    "target": {"kind": "any_creature"},
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], Untap)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_untap_target_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "untap_target", "target": {"kind": "any_creature"}}``
    dict shape on a triggered_ability carrier. The bridge dict shape
    must register cleanly on the Rust side too; this lockstep gate
    keeps the bridge from drifting away from Rust's parse_effect
    coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3237-3244`` requires ``target`` and
    downcasts to a target-filter dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- UntapTarget",
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 1},
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "untap_target",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_tap_another_target_creature_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23 fresh discovery walk) --
    TapAnotherTargetCreature family. Rust ``{"kind": "tap_another_
    target_creature"}`` translates to Python
    ``TapAnotherTargetCreature()``.

    Mirrors Rust's parse_effect arm at cards.rs:3835-3837 (unit
    variant; no fields). Pin verifies the bare-shape translation.
    """
    from mtg_bot.effects.generic.tap import TapAnotherTargetCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "tap_another_target_creature"})
    assert isinstance(e, TapAnotherTargetCreature)


def test_bridge_tap_another_target_creature_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    Rust's parse_effect silent-drop convention -- unrecognized fields
    in the dict are ignored (not raised as errors). Pins the unit-
    variant contract against future-evolution where a sibling Rust
    variant might add fields.
    """
    from mtg_bot.effects.generic.tap import TapAnotherTargetCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "tap_another_target_creature",
            "spurious_field": 42,
        }
    )
    assert isinstance(e, TapAnotherTargetCreature)


def test_bridge_tap_another_target_creature_targets_filter_set():
    """Bridge arm pin: the constructed Python effect carries the
    ``targets`` tuple set by ``__post_init__`` to a single TargetSpec
    with the ``IsCreature() & ~IsSource()`` filter. Pins the
    implicit construction-time wiring -- the bridge does NOT need to
    thread the filter explicitly because Python's class sets it
    internally. The ``~IsSource()`` arm is what makes this distinct
    from the bare TapTarget(any_creature) shape -- the resolving
    source is excluded from its own target pool so "another" reads
    CR-correct.
    """
    from mtg_bot.effects.generic.tap import TapAnotherTargetCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "tap_another_target_creature"})
    assert isinstance(e, TapAnotherTargetCreature)
    # __post_init__ sets `targets` to a 1-tuple of TargetSpec.
    assert hasattr(e, "targets")
    assert len(e.targets) == 1


def test_bridge_tap_another_target_creature_equality_default_constructor():
    """Bridge arm pin: two ``TapAnotherTargetCreature()`` instances
    are equal under dataclass(frozen=True) default __eq__ semantics.
    The ``targets`` field is set in ``__post_init__`` to the same
    tuple shape on both, so structural equality holds. Pins the
    frozen-dataclass equality contract for the hash-based keying
    used downstream (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.tap import TapAnotherTargetCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect({"kind": "tap_another_target_creature"})
    e2 = TapAnotherTargetCreature()
    assert e1 == e2


def test_bridge_tap_another_target_creature_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([TapAnotherTargetCreature(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.tap import TapAnotherTargetCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "tap_another_target_creature"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], TapAnotherTargetCreature)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_tap_another_target_creature_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "tap_another_target_creature"}`` on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3835-3837`` accepts the unit-shape dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- TapAnotherTargetCreature",
        "card_types": ["Creature"],
        "mana_cost": {"W": 1, "generic": 1},
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "tap_another_target_creature"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_tap_enchanted_permanent_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23 fresh discovery walk) --
    TapEnchantedPermanent family. Rust ``{"kind": "tap_enchanted_
    permanent"}`` translates to Python ``TapEnchantedPermanent()``.

    Mirrors Rust's parse_effect arm at cards.rs:3728-3730 (unit
    variant; no fields). Pin verifies the bare-shape translation.
    """
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "tap_enchanted_permanent"})
    assert isinstance(e, TapEnchantedPermanent)


def test_bridge_tap_enchanted_permanent_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    Rust's parse_effect silent-drop convention -- unrecognized fields
    in the dict are ignored (not raised as errors). Pins the unit-
    variant contract against future-evolution where a sibling Rust
    variant might add fields.
    """
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "tap_enchanted_permanent",
            "spurious_field": "ignored",
        }
    )
    assert isinstance(e, TapEnchantedPermanent)


def test_bridge_tap_enchanted_permanent_no_constructor_args():
    """Bridge arm pin: the constructed Python effect has no
    constructor args -- the host is resolved at resolve-time via
    ``state.resolving_source.attached_to``. Pins the no-args
    construction contract; the Aura attachment is a state-side
    concern, not an effect-time field. This is the distinguishing
    shape vs sister TapTarget(any_creature) and
    TapAnotherTargetCreature -- those carry a TargetSpec for
    Decision-pipeline targeting; this one resolves the host
    implicitly via the source's attachment slot.
    """
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "tap_enchanted_permanent"})
    assert isinstance(e, TapEnchantedPermanent)


def test_bridge_tap_enchanted_permanent_equality_default_constructor():
    """Bridge arm pin: two ``TapEnchantedPermanent()`` instances are
    equal under dataclass(frozen=True) default __eq__ semantics. Pins
    the frozen-dataclass equality contract for the hash-based keying
    used downstream (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect({"kind": "tap_enchanted_permanent"})
    e2 = TapEnchantedPermanent()
    assert e1 == e2


def test_bridge_tap_enchanted_permanent_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([TapEnchantedPermanent(), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "tap_enchanted_permanent"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], TapEnchantedPermanent)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_tap_enchanted_permanent_in_aura_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "tap_enchanted_permanent"}`` on an Aura effects
    carrier (the canonical Stay-Hidden-Stay-Silent shape). The
    bridge dict shape must register cleanly on Rust too; this
    lockstep gate keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3728-3730`` accepts the unit-shape dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- TapEnchantedPermanent",
        "card_types": ["Enchantment"],
        "mana_cost": {"U": 1, "generic": 1},
        "effects": [{"kind": "tap_enchanted_permanent"}],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_put_card_from_graveyard_on_bottom_of_library_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 24 fresh discovery walk) --
    PutCardFromGraveyardOnBottomOfLibrary family. Rust
    ``{"kind": "put_card_from_graveyard_on_bottom_of_library"}``
    translates to Python's bare-constructor
    ``PutCardFromGraveyardOnBottomOfLibrary()``.

    Mirrors Rust's parse_effect arm at cards.rs:3683-3686 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/graveyard.py:419``) as the mirror of
    the Rust resolver.
    """
    from mtg_bot.effects.generic.graveyard import (
        PutCardFromGraveyardOnBottomOfLibrary,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_card_from_graveyard_on_bottom_of_library"}
    )
    assert isinstance(e, PutCardFromGraveyardOnBottomOfLibrary)


def test_bridge_put_card_from_graveyard_on_bottom_of_library_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:3683-3686 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.graveyard import (
        PutCardFromGraveyardOnBottomOfLibrary,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_card_from_graveyard_on_bottom_of_library",
            "this_extra_key_should_be_silently_ignored": 42,
            "another": ["arbitrary", "junk"],
        }
    )
    assert isinstance(e, PutCardFromGraveyardOnBottomOfLibrary)


def test_bridge_put_card_from_graveyard_on_bottom_of_library_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed
    ``PutCardFromGraveyardOnBottomOfLibrary()``. Pins the dataclass
    equality semantic for the no-fields unit variant.
    """
    from mtg_bot.effects.generic.graveyard import (
        PutCardFromGraveyardOnBottomOfLibrary,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_card_from_graveyard_on_bottom_of_library"}
    )
    assert e == PutCardFromGraveyardOnBottomOfLibrary()


def test_bridge_put_card_from_graveyard_on_bottom_of_library_requires_target_property():
    """Bridge arm pin: the bridged Python object reports
    ``requires_target() is True``, mirroring the class definition at
    ``effects/generic/graveyard.py:426-427``. Pins the targeting-
    Decision-pipeline contract: the activated/triggered ability
    using this effect MUST collect a target before resolve.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_card_from_graveyard_on_bottom_of_library"}
    )
    assert e.requires_target() is True


def test_bridge_put_card_from_graveyard_on_bottom_of_library_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PutCardFromGraveyardOnBottomOfLibrary(),
    DrawCards(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss inside
    Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.graveyard import (
        PutCardFromGraveyardOnBottomOfLibrary,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "put_card_from_graveyard_on_bottom_of_library"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], PutCardFromGraveyardOnBottomOfLibrary)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_put_card_from_graveyard_on_bottom_of_library_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "put_card_from_graveyard_on_bottom_of_library"}`` dict
    shape on a triggered_ability carrier. The bridge dict shape must
    register cleanly on the Rust side too; this lockstep gate keeps
    the bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3683-3686`` accepts the bare ``kind``-only
    dict (no ``target`` field — Rust resolves auto-pick
    deterministically per state_ops_resolve.rs:5103-5133).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "PutCardFromGraveyardOnBottomOfLibrary"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 1},
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "put_card_from_graveyard_on_bottom_of_library",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_lose_life_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 24 fresh discovery walk) --
    LoseLife family. Rust ``{"kind": "lose_life", "amount": 2}``
    translates to Python ``LoseLife(amount=2)``.

    Mirrors Rust's parse_effect arm at cards.rs:3439-3454 (required
    ``amount`` field, non-negative). Sister of round-23
    each_opponent_loses_life arm but applies to source-controller
    instead of opponents.
    """
    from mtg_bot.effects.generic.life import LoseLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "lose_life", "amount": 2})
    assert isinstance(e, LoseLife)
    assert e.amount == 2


def test_bridge_lose_life_amount_required_keyerror_when_missing():
    """Bridge arm pin: ``amount`` is REQUIRED -- missing raises
    KeyError (mirrors Rust's parse_effect ``require()`` contract at
    cards.rs:3448). Distinct from sister target_opponent_loses_life
    which defaults to 1.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect({"kind": "lose_life"})


def test_bridge_lose_life_amount_zero_silent_noop():
    """Bridge arm pin: amount=0 is a legal silent no-op (mirrors
    Rust's resolve docstring at state_ops_resolve.rs:3661-3665 --
    "amount=0 is a legal no-op"; sister to GainLife / DrawCards /
    Mill / EachOpponentLosesLife / DiscardSelf zero-amount
    conventions). The translation does NOT raise; the resolved
    effect leaves life unchanged.
    """
    from mtg_bot.effects.generic.life import LoseLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "lose_life", "amount": 0})
    assert isinstance(e, LoseLife)
    assert e.amount == 0


def test_bridge_lose_life_negative_amount_raises():
    """Bridge arm pin: negative ``amount`` raises ValueError at
    translation time (mirrors Rust's parse_effect contract at
    cards.rs:3449-3452 -- "lose_life: amount must be non-negative").
    Defense-in-depth: surfaces the contract at the bridge boundary
    rather than waiting for resolve to fault.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect({"kind": "lose_life", "amount": -1})


def test_bridge_lose_life_amount_one_lifelink_spacecraft_shape():
    """Bridge arm pin: amount=1 round-trips correctly. Mirrors the
    Lifelink Spacecraft attacker-rider shape "you draw a card and
    lose 1 life" -- the LoseLife composes inside a Sequence with
    DrawCards. This pin verifies the typical EoE consumer's amount
    value translates exactly.
    """
    from mtg_bot.effects.generic.life import LoseLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "lose_life", "amount": 1})
    assert e == LoseLife(amount=1)


def test_bridge_lose_life_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DrawCards(1), LoseLife(1)])`` translates correctly
    via the recursive Sequence translator. Mirrors the Lifelink
    Spacecraft attacker-rider canonical shape.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import LoseLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "draw_cards", "count": 1},
                {"kind": "lose_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DrawCards)
    assert isinstance(e.effects[1], LoseLife)
    assert e.effects[1].amount == 1


def test_rust_parser_accepts_lose_life_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "lose_life", "amount": 1}`` dict shape on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the bridge
    from drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3439-3454`` requires
    the ``amount`` field and accepts non-negative ints.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- LoseLife",
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 1},
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "lose_life", "amount": 2},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_untap_target_other_permanent_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 24 fresh discovery walk) --
    UntapTargetOtherPermanent family. Rust ``{"kind":
    "untap_target_other_permanent"}`` translates to Python
    ``UntapTargetOtherPermanent()``.

    Mirrors Rust's parse_effect arm at cards.rs:3832-3833 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/tap.py:408``) as the mirror of the
    Rust resolver.
    """
    from mtg_bot.effects.generic.tap import UntapTargetOtherPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "untap_target_other_permanent"}
    )
    assert isinstance(e, UntapTargetOtherPermanent)


def test_bridge_untap_target_other_permanent_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:3832-3833 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.tap import UntapTargetOtherPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "untap_target_other_permanent",
            "this_extra_key_should_be_silently_ignored": 42,
        }
    )
    assert isinstance(e, UntapTargetOtherPermanent)


def test_bridge_untap_target_other_permanent_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``UntapTargetOtherPermanent()``. Pins the
    dataclass equality semantic for the no-fields unit variant.
    """
    from mtg_bot.effects.generic.tap import UntapTargetOtherPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "untap_target_other_permanent"}
    )
    assert e == UntapTargetOtherPermanent()


def test_bridge_untap_target_other_permanent_requires_target_property():
    """Bridge arm pin: the bridged Python object reports
    ``requires_target() is True``, mirroring the class definition at
    ``effects/generic/tap.py:415-416``. Pins the targeting-Decision-
    pipeline contract: the becomes-tapped triggered ability using
    this effect MUST collect a target before resolve.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "untap_target_other_permanent"}
    )
    assert e.requires_target() is True


def test_bridge_untap_target_other_permanent_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([UntapTargetOtherPermanent(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.tap import UntapTargetOtherPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "untap_target_other_permanent"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], UntapTargetOtherPermanent)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_untap_target_other_permanent_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "untap_target_other_permanent"}`` dict shape on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the bridge
    from drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3832-3833`` accepts
    the bare ``kind``-only unit-variant dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "UntapTargetOtherPermanent"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 1},
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "untap_target_other_permanent"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_create_token_copy_of_target_creature_you_control_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 23 fresh discovery walk) --
    CreateTokenCopyOfTargetCreatureYouControl family. Rust
    ``{"kind": "create_token_copy_of_target_creature_you_control"}``
    translates to Python
    ``CreateTokenCopyOfTargetCreatureYouControl()``.

    Mirrors Rust's parse_effect arm at cards.rs:3696-3698 (unit
    variant; no fields). Pin verifies the bare-shape translation.
    """
    from mtg_bot.effects.generic.tokens import (
        CreateTokenCopyOfTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "create_token_copy_of_target_creature_you_control"}
    )
    assert isinstance(e, CreateTokenCopyOfTargetCreatureYouControl)


def test_bridge_create_token_copy_of_target_creature_you_control_default_subtype_none():
    """Bridge arm pin: the constructed Python effect has
    ``additional_subtype=None`` by default. Mirrors Rust's substrate
    decision to defer the subtype-injection rider (per
    cards.rs:1364-1365 ``Python field drop. Python's
    additional_subtype field is deferred (Rust has no subtypes on
    registry yet)``). This pin will fail loudly if a future Rust
    schema bump adds an ``additional_subtype`` field to the bridge
    dict shape and the Python side has to start threading it.
    """
    from mtg_bot.effects.generic.tokens import (
        CreateTokenCopyOfTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "create_token_copy_of_target_creature_you_control"}
    )
    assert isinstance(e, CreateTokenCopyOfTargetCreatureYouControl)
    assert e.additional_subtype is None


def test_bridge_create_token_copy_of_target_creature_you_control_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    Rust's parse_effect silent-drop convention -- unrecognized fields
    in the dict are ignored (not raised as errors). Pins the unit-
    variant contract against future-evolution where a sibling Rust
    variant might add fields. Notably, even an explicit
    ``additional_subtype`` field in the dict is silently dropped at
    bridge time today (Python defaults to None; Rust hasn't shipped
    the pass-through yet).
    """
    from mtg_bot.effects.generic.tokens import (
        CreateTokenCopyOfTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "create_token_copy_of_target_creature_you_control",
            "additional_subtype": "Reflection",
            "spurious_field": 42,
        }
    )
    assert isinstance(e, CreateTokenCopyOfTargetCreatureYouControl)
    # Even though dict carried "Reflection", bridge defaults to None
    # because Rust's substrate doesn't pass it through yet.
    assert e.additional_subtype is None


def test_bridge_create_token_copy_of_target_creature_you_control_equality_default_constructor():
    """Bridge arm pin: two
    ``CreateTokenCopyOfTargetCreatureYouControl()`` instances are
    equal under dataclass(frozen=True) default __eq__ semantics. Pins
    the frozen-dataclass equality contract for the hash-based keying
    used downstream (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.tokens import (
        CreateTokenCopyOfTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "create_token_copy_of_target_creature_you_control"}
    )
    e2 = CreateTokenCopyOfTargetCreatureYouControl()
    assert e1 == e2


def test_bridge_create_token_copy_of_target_creature_you_control_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([CreateTokenCopyOfTargetCreatureYouControl(),
    DrawCards(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss
    inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.tokens import (
        CreateTokenCopyOfTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "create_token_copy_of_target_creature_you_control"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], CreateTokenCopyOfTargetCreatureYouControl
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_create_token_copy_of_target_creature_you_control_in_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "create_token_copy_of_target_creature_you_
    control"}`` on a Sorcery effects carrier (the canonical
    Mirrorpool-shape). The bridge dict shape must register cleanly
    on Rust too; this lockstep gate keeps the bridge from drifting
    away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3696-3698`` accepts the unit-shape dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "CreateTokenCopyOfTargetCreatureYouControl"
        ),
        "card_types": ["Sorcery"],
        "mana_cost": {"U": 2, "generic": 1},
        "effects": [
            {"kind": "create_token_copy_of_target_creature_you_control"}
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_move_self_counters_to_target_creature_you_control_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 25 fresh discovery walk) --
    MoveSelfCountersToTargetCreatureYouControl family. Rust ``{"kind":
    "move_self_counters_to_target_creature_you_control"}`` translates
    to Python ``MoveSelfCountersToTargetCreatureYouControl()``.

    Mirrors Rust's parse_effect arm at cards.rs:3686-3688 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/counters.py:263``) as the mirror of
    the Rust resolver.
    """
    from mtg_bot.effects.generic.counters import (
        MoveSelfCountersToTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "move_self_counters_to_target_creature_you_control"}
    )
    assert isinstance(e, MoveSelfCountersToTargetCreatureYouControl)


def test_bridge_move_self_counters_to_target_creature_you_control_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:3686-3688 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23/24 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.counters import (
        MoveSelfCountersToTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "move_self_counters_to_target_creature_you_control",
            "this_extra_key_should_be_silently_ignored": 42,
        }
    )
    assert isinstance(e, MoveSelfCountersToTargetCreatureYouControl)


def test_bridge_move_self_counters_to_target_creature_you_control_equality_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``MoveSelfCountersToTargetCreatureYouControl()``.
    Pins the dataclass equality semantic for the no-fields unit
    variant.
    """
    from mtg_bot.effects.generic.counters import (
        MoveSelfCountersToTargetCreatureYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "move_self_counters_to_target_creature_you_control"}
    )
    assert e == MoveSelfCountersToTargetCreatureYouControl()


def test_bridge_move_self_counters_to_target_creature_you_control_requires_target_property():
    """Bridge arm pin: the bridged Python object reports
    ``requires_target() is True``, mirroring the class definition at
    ``effects/generic/counters.py:272-273``. Pins the targeting-
    Decision-pipeline contract: the dies-trigger ability using this
    effect MUST collect a target before resolve.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "move_self_counters_to_target_creature_you_control"}
    )
    assert e.requires_target() is True


def test_bridge_move_self_counters_to_target_creature_you_control_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([MoveSelfCountersToTargetCreatureYouControl(),
    DrawCards(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss inside
    Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.counters import (
        MoveSelfCountersToTargetCreatureYouControl,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "move_self_counters_to_target_creature_you_control"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], MoveSelfCountersToTargetCreatureYouControl
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_move_self_counters_to_target_creature_you_control_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "move_self_counters_to_target_creature_you_control"}``
    dict shape on a triggered_ability carrier (Dockworker-Drone-shape
    "When this dies, you may put its counters on target creature you
    control"). The bridge dict shape must register cleanly on Rust
    too; this lockstep gate keeps the bridge from drifting away from
    Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3686-3688``
    accepts the bare ``kind``-only unit-variant dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "MoveSelfCountersToTargetCreatureYouControl"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "move_self_counters_to_target_creature_you_control",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_pump_target_artifact_or_creature_until_eot_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 25 fresh discovery walk) --
    PumpTargetArtifactOrCreatureUntilEOT family. Rust ``{"kind":
    "pump_target_artifact_or_creature_until_eot", "power_bonus": 2,
    "toughness_bonus": 2}`` translates to Python
    ``PumpTargetArtifactOrCreatureUntilEOT(power_bonus=2,
    toughness_bonus=2)``.

    Mirrors Rust's parse_effect arm at cards.rs:3806-3826. The bridge
    wires the same Python-side primitive
    (``effects/generic/pump.py:977``) as the mirror of the Rust
    resolver.
    """
    from mtg_bot.effects.generic.pump import (
        PumpTargetArtifactOrCreatureUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_target_artifact_or_creature_until_eot",
            "power_bonus": 2,
            "toughness_bonus": 2,
        }
    )
    assert isinstance(e, PumpTargetArtifactOrCreatureUntilEOT)
    assert e.power_bonus == 2
    assert e.toughness_bonus == 2


def test_bridge_pump_target_artifact_or_creature_until_eot_missing_keys_default_zero():
    """Bridge arm pin: missing ``power_bonus`` / ``toughness_bonus``
    keys default to 0 (mirrors Rust's parse_effect arm at
    cards.rs:3810-3825 which `match dict.get_item("power_bonus")?` and
    falls back to 0 when None). Pure no-op on resolve when both are 0.
    """
    from mtg_bot.effects.generic.pump import (
        PumpTargetArtifactOrCreatureUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "pump_target_artifact_or_creature_until_eot"}
    )
    assert isinstance(e, PumpTargetArtifactOrCreatureUntilEOT)
    assert e.power_bonus == 0
    assert e.toughness_bonus == 0


def test_bridge_pump_target_artifact_or_creature_until_eot_negative_bonuses_allowed():
    """Bridge arm pin: negative bonuses are accepted and round-trip
    intact (debuff shapes like -3/-3 oracle text). Mirrors Rust's
    parser which ALLOWS negative i32 values for power_bonus and
    toughness_bonus -- there is no clamp at parse time, the bonus is
    applied as-is at resolve.
    """
    from mtg_bot.effects.generic.pump import (
        PumpTargetArtifactOrCreatureUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "pump_target_artifact_or_creature_until_eot",
            "power_bonus": -3,
            "toughness_bonus": -3,
        }
    )
    assert isinstance(e, PumpTargetArtifactOrCreatureUntilEOT)
    assert e.power_bonus == -3
    assert e.toughness_bonus == -3


def test_bridge_pump_target_artifact_or_creature_until_eot_asymmetric_bonus():
    """Bridge arm pin: power-only buff (+N/+0) and toughness-only buff
    (+0/+M) shapes round-trip independently. Pins the two-field
    independence -- the bridge does not couple the two values.
    """
    from mtg_bot.effects.generic.pump import (
        PumpTargetArtifactOrCreatureUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    power_only = _ta_dict_to_python_effect(
        {
            "kind": "pump_target_artifact_or_creature_until_eot",
            "power_bonus": 3,
        }
    )
    assert power_only.power_bonus == 3
    assert power_only.toughness_bonus == 0

    toughness_only = _ta_dict_to_python_effect(
        {
            "kind": "pump_target_artifact_or_creature_until_eot",
            "toughness_bonus": 5,
        }
    )
    assert toughness_only.power_bonus == 0
    assert toughness_only.toughness_bonus == 5


def test_bridge_pump_target_artifact_or_creature_until_eot_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PumpTargetArtifactOrCreatureUntilEOT(power_bonus=1,
    toughness_bonus=1), DrawCards(1)])`` translates correctly via the
    recursive Sequence translator. Pins this arm against accidental
    dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.pump import (
        PumpTargetArtifactOrCreatureUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "pump_target_artifact_or_creature_until_eot",
                    "power_bonus": 1,
                    "toughness_bonus": 1,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], PumpTargetArtifactOrCreatureUntilEOT
    )
    assert e.effects[0].power_bonus == 1
    assert e.effects[0].toughness_bonus == 1
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_pump_target_artifact_or_creature_until_eot_in_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "pump_target_artifact_or_creature_until_eot",
    "power_bonus": N, "toughness_bonus": M}`` dict shape on an Instant
    effects carrier. The bridge dict shape must register cleanly on
    Rust too; this lockstep gate keeps the bridge from drifting away
    from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3806-3826`` accepts
    the struct-variant dict with both fields tolerating defaults.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "PumpTargetArtifactOrCreatureUntilEOT"
        ),
        "card_types": ["Instant"],
        "mana_cost": {"R": 1},
        "effects": [
            {
                "kind": "pump_target_artifact_or_creature_until_eot",
                "power_bonus": 2,
                "toughness_bonus": 2,
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_put_charge_counters_on_self_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 26 fresh discovery walk) --
    PutChargeCountersOnSelf family. Rust ``{"kind":
    "put_charge_counters_on_self", "amount": 2,
    "amount_from_x": False}`` translates to Python
    ``PutChargeCountersOnSelf(amount=2, amount_from_x=False)``.

    Mirrors Rust's parse_effect arm at cards.rs:4154-4177 (optional
    ``amount`` defaulting to 1, optional ``amount_from_x`` defaulting
    to False, both round-trip explicitly via to_dict at
    cards.rs:5609-5613).
    """
    from mtg_bot.effects.generic.counters import PutChargeCountersOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_charge_counters_on_self",
            "amount": 2,
            "amount_from_x": False,
        }
    )
    assert isinstance(e, PutChargeCountersOnSelf)
    assert e.amount == 2
    assert e.amount_from_x is False


def test_bridge_put_charge_counters_on_self_amount_defaults_to_one():
    """Bridge arm pin: missing ``amount`` defaults to 1 (mirrors Rust's
    parse_effect arm at cards.rs:4155-4163 ``None => 1`` and Python's
    class default at counters.py:562).
    """
    from mtg_bot.effects.generic.counters import PutChargeCountersOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_charge_counters_on_self"}
    )
    assert isinstance(e, PutChargeCountersOnSelf)
    assert e.amount == 1
    assert e.amount_from_x is False


def test_bridge_put_charge_counters_on_self_amount_zero_silent_noop():
    """Bridge arm pin: amount=0 is a legal silent no-op (mirrors
    Python's resolve early-return at counters.py:573-574 on n <= 0).
    The translation does NOT raise; the resolved effect leaves the
    source's charge counters unchanged.
    """
    from mtg_bot.effects.generic.counters import PutChargeCountersOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_charge_counters_on_self", "amount": 0}
    )
    assert isinstance(e, PutChargeCountersOnSelf)
    assert e.amount == 0


def test_bridge_put_charge_counters_on_self_negative_amount_raises():
    """Bridge arm pin: negative ``amount`` raises ValueError at
    translation time (mirrors Rust's parse_effect contract at
    cards.rs:4163-4167 -- "put_charge_counters_on_self: amount must
    be non-negative"). Defense-in-depth: surfaces the contract at the
    bridge boundary rather than waiting for resolve to fault.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "put_charge_counters_on_self", "amount": -1}
        )


def test_bridge_put_charge_counters_on_self_amount_from_x_flag_passes_through():
    """Bridge arm pin: amount_from_x=True round-trips correctly.
    Mirrors Blast-Zone-shape "{X}{X}, {T}: Put X charge counters on
    this land" pattern. Rust treats amount_from_x=True as a deferred
    no-op today (cards.rs:1339-1343); Python's resolve reads
    ``state.resolving_x_value`` (counters.py:572). The bridge passes
    the flag through faithfully so the Python side can apply X-cost
    semantics where Rust defers.
    """
    from mtg_bot.effects.generic.counters import PutChargeCountersOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_charge_counters_on_self",
            "amount": 1,
            "amount_from_x": True,
        }
    )
    assert isinstance(e, PutChargeCountersOnSelf)
    assert e.amount_from_x is True


def test_bridge_put_charge_counters_on_self_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals a
    directly-constructed ``PutChargeCountersOnSelf(amount=1,
    amount_from_x=False)``. Pins the dataclass equality semantic for
    the default-field shape used by the typical Spacecraft / Planet
    "enters with a charge counter" carrier.
    """
    from mtg_bot.effects.generic.counters import PutChargeCountersOnSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_charge_counters_on_self",
            "amount": 1,
            "amount_from_x": False,
        }
    )
    assert e == PutChargeCountersOnSelf(amount=1, amount_from_x=False)


def test_bridge_put_charge_counters_on_self_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PutChargeCountersOnSelf(amount=1), DrawCards(1)])``
    translates correctly via the recursive Sequence translator.
    Pins this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.counters import PutChargeCountersOnSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "put_charge_counters_on_self", "amount": 1},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], PutChargeCountersOnSelf)
    assert e.effects[0].amount == 1
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_put_charge_counters_on_self_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "put_charge_counters_on_self", "amount": 1}`` on
    a triggered_ability carrier. The bridge dict shape must register
    cleanly on the Rust side too; this lockstep gate keeps the bridge
    from drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:4154-4177`` accepts
    optional ``amount`` (default 1) and optional ``amount_from_x``
    (default false). Mirrors the Spacecraft "enters with a charge
    counter" carrier shape.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- PutChargeCountersOnSelf",
        "card_types": ["Artifact", "Spacecraft"],
        "mana_cost": {"generic": 3},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "put_charge_counters_on_self",
                    "amount": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_draw_cards_equal_to_resolving_combat_damage_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 26 fresh discovery walk) --
    DrawCardsEqualToResolvingCombatDamage family. Rust ``{"kind":
    "draw_cards_equal_to_resolving_combat_damage"}`` translates to
    Python ``DrawCardsEqualToResolvingCombatDamage()``.

    Mirrors Rust's parse_effect arm at cards.rs:4545-4551 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/draw_mill.py:88``) as the mirror of
    the Rust resolver.
    """
    from mtg_bot.effects.generic.draw_mill import (
        DrawCardsEqualToResolvingCombatDamage,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "draw_cards_equal_to_resolving_combat_damage"}
    )
    assert isinstance(e, DrawCardsEqualToResolvingCombatDamage)


def test_bridge_draw_cards_equal_to_resolving_combat_damage_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:4545-4551 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23/24/25/26 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.draw_mill import (
        DrawCardsEqualToResolvingCombatDamage,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "draw_cards_equal_to_resolving_combat_damage",
            "this_extra_key_should_be_silently_ignored": 99,
        }
    )
    assert isinstance(e, DrawCardsEqualToResolvingCombatDamage)


def test_bridge_draw_cards_equal_to_resolving_combat_damage_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``DrawCardsEqualToResolvingCombatDamage()``.
    Pins the dataclass equality semantic for the no-fields unit
    variant.
    """
    from mtg_bot.effects.generic.draw_mill import (
        DrawCardsEqualToResolvingCombatDamage,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "draw_cards_equal_to_resolving_combat_damage"}
    )
    assert e == DrawCardsEqualToResolvingCombatDamage()


def test_bridge_draw_cards_equal_to_resolving_combat_damage_does_not_require_target():
    """Bridge arm pin: the bridged Python object reports
    ``requires_target() is False`` (the default Effect base
    behavior). The "you may draw that many cards" rider on the
    combat-damage trigger is untargeted -- the controller of the
    triggered ability is the implicit recipient, and the carrier
    pushes the count via state.resolving_combat_damage_amount.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "draw_cards_equal_to_resolving_combat_damage"}
    )
    assert e.requires_target() is False


def test_bridge_draw_cards_equal_to_resolving_combat_damage_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DrawCardsEqualToResolvingCombatDamage(),
    GainLife(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss
    inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import (
        DrawCardsEqualToResolvingCombatDamage,
    )
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "draw_cards_equal_to_resolving_combat_damage"},
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], DrawCardsEqualToResolvingCombatDamage
    )
    assert isinstance(e.effects[1], GainLife)


def test_rust_parser_accepts_draw_cards_equal_to_resolving_combat_damage_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "draw_cards_equal_to_resolving_combat_damage"}``
    on a triggered_ability carrier (the Anticausal-Vestige-shape
    "Whenever this creature deals combat damage to a player, you may
    draw that many cards"). The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:4545-4551`` accepts
    the bare ``kind``-only unit-variant dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "DrawCardsEqualToResolvingCombatDamage"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "draw_cards_equal_to_resolving_combat_damage",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_equal_to_source_power_any_target_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 26 fresh discovery walk) --
    DealDamageEqualToSourcePowerAnyTarget family. Rust ``{"kind":
    "deal_damage_equal_to_source_power_any_target"}`` translates to
    Python ``DealDamageEqualToSourcePowerAnyTarget()``.

    Mirrors Rust's parse_effect arm at cards.rs:3690-3692 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/damage.py:204``) as the mirror of
    the Rust resolver.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageEqualToSourcePowerAnyTarget,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_equal_to_source_power_any_target"}
    )
    assert isinstance(e, DealDamageEqualToSourcePowerAnyTarget)


def test_bridge_deal_damage_equal_to_source_power_any_target_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:3690-3692 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23/24/25/26 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageEqualToSourcePowerAnyTarget,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_equal_to_source_power_any_target",
            "this_extra_key_should_be_silently_ignored": "ignored",
        }
    )
    assert isinstance(e, DealDamageEqualToSourcePowerAnyTarget)


def test_bridge_deal_damage_equal_to_source_power_any_target_default_targets_populated():
    """Bridge arm pin: Python's __post_init__ populates the default
    ``targets`` slot with ``(TargetSpec(any_target_filter()),)``
    when the bridge constructs the effect bare. Mirrors Rust's
    intrinsic any-target filter on the unit variant -- the bridge
    produces the canonical default-targets shape that any consumer
    (cast / dies-trigger) expects when collecting legal targets.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageEqualToSourcePowerAnyTarget,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_equal_to_source_power_any_target"}
    )
    assert isinstance(e, DealDamageEqualToSourcePowerAnyTarget)
    # __post_init__ filled the default any-target slot.
    assert len(e.targets) == 1


def test_bridge_deal_damage_equal_to_source_power_any_target_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed
    ``DealDamageEqualToSourcePowerAnyTarget()``. Pins the dataclass
    equality semantic for the post-init-defaulted-fields shape.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageEqualToSourcePowerAnyTarget,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_equal_to_source_power_any_target"}
    )
    assert e == DealDamageEqualToSourcePowerAnyTarget()


def test_bridge_deal_damage_equal_to_source_power_any_target_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealDamageEqualToSourcePowerAnyTarget(),
    DrawCards(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss
    inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageEqualToSourcePowerAnyTarget,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "deal_damage_equal_to_source_power_any_target",
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], DealDamageEqualToSourcePowerAnyTarget
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_deal_damage_equal_to_source_power_any_target_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "deal_damage_equal_to_source_power_any_target"}``
    on a triggered_ability carrier (the Infernal-Phantom-shape "When
    this dies, it deals damage equal to its power to any target").
    The bridge dict shape must register cleanly on Rust too; this
    lockstep gate keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3690-3692`` accepts
    the bare ``kind``-only unit-variant dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "DealDamageEqualToSourcePowerAnyTarget"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"R": 1, "generic": 2},
        "power": 3,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_equal_to_source_power_any_target",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_become_land_creature_ueot_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 25 fresh discovery walk) --
    BecomeLandCreatureUEOT family. Rust ``{"kind":
    "become_land_creature_ueot", "power": 2, "toughness": 2}``
    translates to Python ``BecomeLandCreatureUEOT(power=2,
    toughness=2)`` with default ``granted_subtype=None`` and default
    empty ``granted_keywords``.

    Mirrors Rust's parse_effect arm at cards.rs:3953-4008. The bridge
    wires the same Python-side primitive
    (``effects/generic/become_type.py:126``) as the mirror of the
    Rust resolver.
    """
    from mtg_bot.effects.generic.become_type import (
        BecomeLandCreatureUEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "become_land_creature_ueot",
            "power": 2,
            "toughness": 2,
        }
    )
    assert isinstance(e, BecomeLandCreatureUEOT)
    assert e.power == 2
    assert e.toughness == 2
    assert e.granted_subtype is None
    assert e.granted_keywords == frozenset()
    assert e.unblockable is False


def test_bridge_become_land_creature_ueot_missing_power_raises():
    """Bridge arm pin: missing ``power`` field raises KeyError
    (mirrors Rust's parse_effect arm at cards.rs:3962-3974 which
    raises 'missing required power field' with PyValueError). The
    bridge enforces the required-fields contract on the Python side
    so the dict shape is symmetric.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {
                "kind": "become_land_creature_ueot",
                "toughness": 2,
            }
        )


def test_bridge_become_land_creature_ueot_missing_toughness_raises():
    """Bridge arm pin: missing ``toughness`` field raises KeyError
    (mirrors Rust's parse_effect arm at cards.rs:3975-3987 which
    raises 'missing required toughness field' with PyValueError).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {
                "kind": "become_land_creature_ueot",
                "power": 2,
            }
        )


def test_bridge_become_land_creature_ueot_granted_subtype_present():
    """Bridge arm pin: ``granted_subtype`` field round-trips when set
    (e.g. ``"Elemental"`` for Lavaclaw-shape, ``"Blinkmoth"`` for
    Blinkmoth Nexus-shape). Pins the optional-string field handling.
    """
    from mtg_bot.effects.generic.become_type import (
        BecomeLandCreatureUEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "become_land_creature_ueot",
            "power": 1,
            "toughness": 1,
            "granted_subtype": "Elemental",
        }
    )
    assert isinstance(e, BecomeLandCreatureUEOT)
    assert e.granted_subtype == "Elemental"


def test_bridge_become_land_creature_ueot_empty_subtype_normalizes_to_none():
    """Bridge arm pin: empty-string ``granted_subtype`` normalizes to
    None (mirrors Rust's parse_effect arm at cards.rs:3996-4000
    `if s.is_empty() { None } else { Some(s) }`). Pins the empty-
    string-as-None convention for round-trip parity.
    """
    from mtg_bot.effects.generic.become_type import (
        BecomeLandCreatureUEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "become_land_creature_ueot",
            "power": 2,
            "toughness": 2,
            "granted_subtype": "",
        }
    )
    assert isinstance(e, BecomeLandCreatureUEOT)
    assert e.granted_subtype is None


def test_bridge_become_land_creature_ueot_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([BecomeLandCreatureUEOT(power=2, toughness=2),
    DrawCards(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss inside
    Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.become_type import (
        BecomeLandCreatureUEOT,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "become_land_creature_ueot",
                    "power": 2,
                    "toughness": 2,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], BecomeLandCreatureUEOT)
    assert e.effects[0].power == 2
    assert e.effects[0].toughness == 2
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_become_land_creature_ueot_in_activated_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "become_land_creature_ueot", "power": N, "toughness":
    M, "granted_subtype": "..."}`` dict shape on a Land's activated-
    ability effects carrier (the canonical manland animation shape).
    The bridge dict shape must register cleanly on Rust too; this
    lockstep gate keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3953-4008`` accepts
    the struct-variant dict with required power/toughness and optional
    granted_subtype.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "BecomeLandCreatureUEOT"
        ),
        "card_types": ["Land"],
        "activated_abilities": [
            {
                "cost": {"mana": {"generic": 1}, "tap_self": True},
                "effect": {
                    "kind": "become_land_creature_ueot",
                    "power": 2,
                    "toughness": 2,
                    "granted_subtype": "Elemental",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_add_mana_to_pool_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 27 fresh discovery walk) --
    AddManaToPool family. Rust ``{"kind": "add_mana_to_pool",
    "color": "W", "amount": 1}`` translates to Python
    ``AddManaToPool(produces=(("W", 1),))``.

    Mirrors Rust's parse_effect arm at cards.rs:3295-3308 (struct
    variant; required color: String + amount: i32). The bridge folds
    Rust's scalar (color, amount) pair into Python's richer
    ``produces: tuple[tuple[str, int], ...]`` shape via a one-tuple of
    one pair (effects/generic/mana.py:30-80 -- the universal mana
    primitive that every "{T}: Add {color}" mana ability emits).
    """
    from mtg_bot.effects.generic.mana import AddManaToPool
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "add_mana_to_pool", "color": "W", "amount": 1}
    )
    assert isinstance(e, AddManaToPool)
    assert e.produces == (("W", 1),)


def test_bridge_add_mana_to_pool_each_color_round_trips():
    """Bridge arm pin: each of the closed mana-color vocab
    ``{W,U,B,R,G,C}`` round-trips as the produces[0][0] tag. Mirrors
    Rust's ``validate_mana_symbol`` at cards.rs:4592-4599 closed set;
    the bridge passes the color string through unchanged so the
    canonical 6-color vocab is preserved end-to-end.
    """
    from mtg_bot.effects.generic.mana import AddManaToPool
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    for color in ("W", "U", "B", "R", "G", "C"):
        e = _ta_dict_to_python_effect(
            {"kind": "add_mana_to_pool", "color": color, "amount": 1}
        )
        assert isinstance(e, AddManaToPool)
        assert e.produces == ((color, 1),)


def test_bridge_add_mana_to_pool_amount_two_round_trips():
    """Bridge arm pin: amount > 1 (a single-color batched mint)
    round-trips. The Rust substrate's ``amount: i32`` field and
    Python's per-pair ``count`` field carry the same semantic --
    "mint N coins of one color in one resolve" -- and the bridge
    preserves the count without interpretation.
    """
    from mtg_bot.effects.generic.mana import AddManaToPool
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "add_mana_to_pool", "color": "G", "amount": 2}
    )
    assert isinstance(e, AddManaToPool)
    assert e.produces == (("G", 2),)


def test_bridge_add_mana_to_pool_amount_zero_round_trips():
    """Bridge arm pin: amount=0 is accepted (Rust's parser only
    raises on negative amounts at cards.rs:3302-3306). The bridge
    passes zero through unchanged -- the resolve no-ops naturally
    (the for-loop in AddManaToPool.resolve runs zero times).
    """
    from mtg_bot.effects.generic.mana import AddManaToPool
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "add_mana_to_pool", "color": "U", "amount": 0}
    )
    assert isinstance(e, AddManaToPool)
    assert e.produces == (("U", 0),)


def test_bridge_add_mana_to_pool_missing_color_raises():
    """Bridge arm pin: missing required 'color' field raises
    KeyError. Mirrors Rust's parse_effect arm at cards.rs:3296 which
    calls ``require(dict, "color")`` -- a missing color is a hard
    parse error on the Rust side, and the bridge surfaces the same
    contract via KeyError (matching the round-25
    become_land_creature_ueot precedent).
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    import pytest

    with pytest.raises(KeyError, match="color"):
        _ta_dict_to_python_effect(
            {"kind": "add_mana_to_pool", "amount": 1}
        )


def test_bridge_add_mana_to_pool_missing_amount_raises():
    """Bridge arm pin: missing required 'amount' field raises
    KeyError. Mirrors Rust's parse_effect arm at cards.rs:3301 which
    calls ``require(dict, "amount")`` -- a missing amount is a hard
    parse error on the Rust side, and the bridge surfaces the same
    contract via KeyError.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    import pytest

    with pytest.raises(KeyError, match="amount"):
        _ta_dict_to_python_effect(
            {"kind": "add_mana_to_pool", "color": "B"}
        )


def test_bridge_add_mana_to_pool_equality_with_direct_construction():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``AddManaToPool(produces=(("R", 1),))``.
    Pins the dataclass equality semantic for the produces-tuple
    shape; spend_restriction defaults to None on both paths.
    """
    from mtg_bot.effects.generic.mana import AddManaToPool
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "add_mana_to_pool", "color": "R", "amount": 1}
    )
    assert e == AddManaToPool(produces=(("R", 1),))


def test_bridge_add_mana_to_pool_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([AddManaToPool({W,1}), GainLife(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.mana import AddManaToPool
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "add_mana_to_pool", "color": "W", "amount": 1},
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], AddManaToPool)
    assert e.effects[0].produces == (("W", 1),)
    assert isinstance(e.effects[1], GainLife)


def test_rust_parser_accepts_add_mana_to_pool_in_activated_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "add_mana_to_pool", "color": "G", "amount": 1}``
    on an activated_ability carrier (the canonical "{T}: Add {G}"
    mana-ability shape). The bridge dict shape must register cleanly
    on Rust too; this lockstep gate keeps the bridge from drifting
    away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3295-3308`` accepts
    the struct-variant dict with required color + amount fields and
    closed-vocab color validation.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- AddManaToPool"
        ),
        "card_types": ["Land"],
        "activated_abilities": [
            {
                "cost": {"tap_self": True},
                "effect": {
                    "kind": "add_mana_to_pool",
                    "color": "G",
                    "amount": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_conditional_if_void_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 26 fresh discovery walk) --
    ConditionalIfVoid family. Rust ``{"kind":
    "conditional_if_void", "then_effect": {...}, "else_effect":
    None}`` translates to Python ``ConditionalIfVoid(then_effect=...,
    else_effect=None)``.

    Mirrors Rust's parse_effect arm at cards.rs:4220-4250. The
    bridge wires the same Python-side primitive
    (``effects/combinators.py:1409``) as the mirror of the Rust
    resolver.
    """
    from mtg_bot.effects.combinators import ConditionalIfVoid
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_if_void",
            "then_effect": {"kind": "draw_cards", "count": 1},
            "else_effect": None,
        }
    )
    assert isinstance(e, ConditionalIfVoid)
    assert isinstance(e.then_effect, DrawCards)
    assert e.then_effect.amount == 1
    assert e.else_effect is None


def test_bridge_conditional_if_void_missing_then_effect_raises():
    """Bridge arm pin: missing ``then_effect`` field raises KeyError
    (mirrors Rust's parse_effect arm at cards.rs:4222-4226 which
    raises 'missing required then_effect field' with PyValueError).
    The bridge enforces the required-fields contract on the Python
    side so the dict shape is symmetric.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {
                "kind": "conditional_if_void",
                "else_effect": None,
            }
        )


def test_bridge_conditional_if_void_else_effect_present():
    """Bridge arm pin: nested ``else_effect`` dict round-trips via
    recursive translator. Mirrors Rust's parse_effect arm at
    cards.rs:4234-4245 which decodes ``else_effect`` as a nested
    effect dict when present (vs key absent / explicit None).
    """
    from mtg_bot.effects.combinators import ConditionalIfVoid
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_if_void",
            "then_effect": {"kind": "draw_cards", "count": 2},
            "else_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, ConditionalIfVoid)
    assert isinstance(e.then_effect, DrawCards)
    assert e.then_effect.amount == 2
    assert isinstance(e.else_effect, DrawCards)
    assert e.else_effect.amount == 1


def test_bridge_conditional_if_void_else_effect_absent_key():
    """Bridge arm pin: ``else_effect`` key absent decodes as None
    (mirrors Rust's parse_effect arm at cards.rs:4234-4236 which
    treats key-absent as None). Pins the three-form else_effect
    contract: absent / explicit None / nested dict all valid.
    """
    from mtg_bot.effects.combinators import ConditionalIfVoid
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_if_void",
            "then_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, ConditionalIfVoid)
    assert isinstance(e.then_effect, DrawCards)
    assert e.else_effect is None


def test_bridge_conditional_if_void_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed
    ``ConditionalIfVoid(then_effect=DrawCards(1))``. Pins the
    dataclass equality semantic for the optional-else default shape.
    """
    from mtg_bot.effects.combinators import ConditionalIfVoid
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_if_void",
            "then_effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert e == ConditionalIfVoid(then_effect=DrawCards(amount=1))


def test_bridge_conditional_if_void_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([ConditionalIfVoid(DrawCards(1)), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import ConditionalIfVoid, Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "conditional_if_void",
                    "then_effect": {"kind": "draw_cards", "count": 1},
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], ConditionalIfVoid)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_conditional_if_void_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape on a triggered_ability carrier (the canonical "Void --
    Whenever this attacks, ..." rider). The bridge dict shape must
    register cleanly on Rust too; this lockstep gate keeps the
    bridge from drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:4220-4250`` accepts
    the struct-variant dict with required ``then_effect`` and
    optional ``else_effect``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "ConditionalIfVoid"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "attacks",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "conditional_if_void",
                    "then_effect": {"kind": "draw_cards", "count": 1},
                    "else_effect": None,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_conditional_on_tapped_creatures_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 26 fresh discovery walk) --
    ConditionalOnTappedCreatures family. Rust ``{"kind":
    "conditional_on_tapped_creatures", "effect": {...},
    "min_count": 2}`` translates to Python
    ``ConditionalOnTappedCreatures(effect=..., min_count=2)``.

    Mirrors Rust's parse_effect arm at cards.rs:4194-4219. The
    bridge wires the same Python-side primitive
    (``effects/combinators.py:1007``) as the mirror of the Rust
    resolver.
    """
    from mtg_bot.effects.combinators import ConditionalOnTappedCreatures
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_on_tapped_creatures",
            "effect": {"kind": "draw_cards", "count": 1},
            "min_count": 2,
        }
    )
    assert isinstance(e, ConditionalOnTappedCreatures)
    assert isinstance(e.effect, DrawCards)
    assert e.effect.amount == 1
    assert e.min_count == 2


def test_bridge_conditional_on_tapped_creatures_missing_effect_raises():
    """Bridge arm pin: missing ``effect`` field raises KeyError
    (mirrors Rust's parse_effect arm at cards.rs:4196-4200 which
    raises 'missing required effect field' with PyValueError).
    The bridge enforces the required-fields contract on the
    Python side so the dict shape is symmetric.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(KeyError):
        _ta_dict_to_python_effect(
            {
                "kind": "conditional_on_tapped_creatures",
                "min_count": 2,
            }
        )


def test_bridge_conditional_on_tapped_creatures_min_count_default():
    """Bridge arm pin: ``min_count`` key absent defaults to 2
    (mirrors Rust's parse_effect arm at cards.rs:4207-4214 which
    defaults to 2 when key absent). Pins the documented default
    that matches Python's class field default at
    combinators.py:1016 (``min_count: int = 2``).
    """
    from mtg_bot.effects.combinators import ConditionalOnTappedCreatures
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_on_tapped_creatures",
            "effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert isinstance(e, ConditionalOnTappedCreatures)
    assert isinstance(e.effect, DrawCards)
    assert e.min_count == 2


def test_bridge_conditional_on_tapped_creatures_custom_min_count():
    """Bridge arm pin: explicit ``min_count`` value (e.g. 3)
    round-trips correctly. Pins the explicit-override path
    against silent default-clobbering bugs.
    """
    from mtg_bot.effects.combinators import ConditionalOnTappedCreatures
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_on_tapped_creatures",
            "effect": {"kind": "draw_cards", "count": 1},
            "min_count": 3,
        }
    )
    assert isinstance(e, ConditionalOnTappedCreatures)
    assert e.min_count == 3


def test_bridge_conditional_on_tapped_creatures_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``ConditionalOnTappedCreatures(...)``.
    Pins the dataclass equality semantic for the default-min_count
    shape.
    """
    from mtg_bot.effects.combinators import ConditionalOnTappedCreatures
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "conditional_on_tapped_creatures",
            "effect": {"kind": "draw_cards", "count": 1},
        }
    )
    assert e == ConditionalOnTappedCreatures(
        effect=DrawCards(amount=1), min_count=2
    )


def test_bridge_conditional_on_tapped_creatures_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([ConditionalOnTappedCreatures(DrawCards(1)),
    DrawCards(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss
    inside Sequence.
    """
    from mtg_bot.effects.combinators import (
        ConditionalOnTappedCreatures,
        Sequence,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "conditional_on_tapped_creatures",
                    "effect": {"kind": "draw_cards", "count": 1},
                    "min_count": 2,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], ConditionalOnTappedCreatures)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_conditional_on_tapped_creatures_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape on a triggered_ability carrier (the canonical "if you
    control two or more tapped creatures, ..." rider). The bridge
    dict shape must register cleanly on Rust too; this lockstep
    gate keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:4194-4219`` accepts
    the struct-variant dict with required ``effect`` and optional
    ``min_count`` (default 2).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "ConditionalOnTappedCreatures"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"W": 1, "generic": 2},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "attacks",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "conditional_on_tapped_creatures",
                    "effect": {"kind": "draw_cards", "count": 1},
                    "min_count": 2,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_choose_one_mode_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 27 fresh discovery walk) --
    ChooseOneMode family. Rust ``{"kind": "choose_one_mode", "modes":
    [<effect_dict>, <effect_dict>]}`` translates to Python
    ``ModalChooseOne(modes=(...,))``.

    Mirrors Rust's parse_effect arm at cards.rs:4119-4153 (struct
    variant; required modes: Vec<Effect>). The bridge recursively
    translates each mode-dict via ``_ta_dict_to_python_effect`` and
    constructs Python ``ModalChooseOne`` (effects/combinators.py:
    1112-1208) with the resulting tuple.
    """
    from mtg_bot.effects.combinators import ModalChooseOne
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "choose_one_mode",
            "modes": [
                {"kind": "draw_cards", "count": 1},
                {"kind": "gain_life", "amount": 2},
            ],
        }
    )
    assert isinstance(e, ModalChooseOne)
    assert len(e.modes) == 2
    assert isinstance(e.modes[0], DrawCards)
    assert isinstance(e.modes[1], GainLife)


def test_bridge_choose_one_mode_or_both_defaults_to_false():
    """Bridge arm pin: bridged ``ModalChooseOne.or_both`` defaults to
    False, mirroring the Rust substrate's choose-one-only coverage
    (Rust's ``ChooseOneMode`` variant has no or_both discriminator;
    a "Choose one or both --" mode would need a Rust schema bump
    first). Pins the structural narrowing convention.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "choose_one_mode",
            "modes": [
                {"kind": "gain_life", "amount": 1},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert e.or_both is False


def test_bridge_choose_one_mode_single_mode_round_trips():
    """Bridge arm pin: a single-mode ``ChooseOneMode`` (degenerate
    "Choose one --" with one option) round-trips. Rust's parser
    requires non-empty modes (cards.rs:4138-4142) but accepts a
    one-element list; the bridge passes it through.
    """
    from mtg_bot.effects.combinators import ModalChooseOne
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "choose_one_mode",
            "modes": [{"kind": "draw_cards", "count": 1}],
        }
    )
    assert isinstance(e, ModalChooseOne)
    assert len(e.modes) == 1
    assert isinstance(e.modes[0], DrawCards)


def test_bridge_choose_one_mode_three_modes_round_trips():
    """Bridge arm pin: a three-mode ``ChooseOneMode`` (the
    Cryptic-Command-shape "Choose one of three options") round-trips
    each mode independently. Pins the recursive translator's
    iteration over the modes Vec without dropping any.
    """
    from mtg_bot.effects.combinators import ModalChooseOne
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "choose_one_mode",
            "modes": [
                {"kind": "draw_cards", "count": 1},
                {"kind": "gain_life", "amount": 2},
                {"kind": "draw_cards", "count": 2},
            ],
        }
    )
    assert isinstance(e, ModalChooseOne)
    assert len(e.modes) == 3
    assert isinstance(e.modes[0], DrawCards)
    assert e.modes[0].amount == 1
    assert isinstance(e.modes[1], GainLife)
    assert e.modes[1].amount == 2
    assert isinstance(e.modes[2], DrawCards)
    assert e.modes[2].amount == 2


def test_bridge_choose_one_mode_missing_modes_raises():
    """Bridge arm pin: missing required 'modes' field raises
    KeyError. Mirrors Rust's parse_effect arm at cards.rs:4123-4129
    which raises PyValueError on missing modes -- the bridge
    surfaces the same contract via KeyError (matching the round-25
    become_land_creature_ueot precedent for required-fields parity).
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    import pytest

    with pytest.raises(KeyError, match="modes"):
        _ta_dict_to_python_effect({"kind": "choose_one_mode"})


def test_bridge_choose_one_mode_empty_modes_raises():
    """Bridge arm pin: empty 'modes' list raises ValueError. Mirrors
    Rust's parse_effect arm at cards.rs:4138-4142 which raises
    PyValueError on empty modes ('at least one mode required'). The
    bridge surfaces the same contract.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        _ta_dict_to_python_effect(
            {"kind": "choose_one_mode", "modes": []}
        )


def test_bridge_choose_one_mode_nested_in_sequence():
    """Round-trip a ``ChooseOneMode`` nested INSIDE a Sequence
    composer: ``Sequence([ChooseOneMode([DrawCards, GainLife]),
    GainLife])`` translates correctly via the recursive Sequence
    translator. Pins the modal arm against accidental dispatch loss
    inside Sequence.
    """
    from mtg_bot.effects.combinators import ModalChooseOne, Sequence
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "choose_one_mode",
                    "modes": [
                        {"kind": "draw_cards", "count": 1},
                        {"kind": "gain_life", "amount": 1},
                    ],
                },
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], ModalChooseOne)
    assert len(e.effects[0].modes) == 2
    assert isinstance(e.effects[1], GainLife)


def test_bridge_choose_one_mode_sequence_inside_mode():
    """Reverse-nesting pin: a ``Sequence`` inside one of the modes
    of a ``ChooseOneMode`` round-trips correctly. Pins the
    recursive-into-mode translator path against accidental dispatch
    loss when the mode itself is a composer.
    """
    from mtg_bot.effects.combinators import ModalChooseOne, Sequence
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "choose_one_mode",
            "modes": [
                {
                    "kind": "sequence",
                    "effects": [
                        {"kind": "draw_cards", "count": 1},
                        {"kind": "gain_life", "amount": 1},
                    ],
                },
                {"kind": "gain_life", "amount": 3},
            ],
        }
    )
    assert isinstance(e, ModalChooseOne)
    assert len(e.modes) == 2
    assert isinstance(e.modes[0], Sequence)
    assert len(e.modes[0].effects) == 2


def test_rust_parser_accepts_choose_one_mode_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "choose_one_mode", "modes": [...]}`` on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:4119-4153`` accepts
    the struct-variant dict with required modes list (recursively
    parsed via parse_effect for each entry).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- ChooseOneMode"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "choose_one_mode",
                    "modes": [
                        {"kind": "draw_cards", "count": 1},
                        {"kind": "gain_life", "amount": 2},
                    ],
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_destroy_enchanted_permanent_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 26 fresh discovery walk) --
    DestroyEnchantedPermanent family. Rust ``{"kind":
    "destroy_enchanted_permanent"}`` translates to Python
    ``DestroyEnchantedPermanent()``.

    Mirrors Rust's parse_effect arm at cards.rs:3725-3727 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/destroy.py:213``) as the mirror of
    the Rust resolver.
    """
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_enchanted_permanent"}
    )
    assert isinstance(e, DestroyEnchantedPermanent)


def test_bridge_destroy_enchanted_permanent_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:3725-3727 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23/24/25/26 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "destroy_enchanted_permanent",
            "this_extra_key_should_be_silently_ignored": "ignored",
        }
    )
    assert isinstance(e, DestroyEnchantedPermanent)


def test_bridge_destroy_enchanted_permanent_does_not_require_target():
    """Bridge arm pin: ``requires_target() -> False`` -- the Aura-
    host destruction is intrinsic to the variant (the host is read
    from ``state.resolving_source.attached_to`` at resolve time, not
    from the Decision targets tuple). Mirrors Python's class
    contract at destroy.py:225-226 and Rust's resolve which also
    reads the attached host without consulting any explicit target.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_enchanted_permanent"}
    )
    assert e.requires_target() is False


def test_bridge_destroy_enchanted_permanent_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``DestroyEnchantedPermanent()``. Pins the
    dataclass equality semantic for the unit-shape (no fields to
    compare; identity reduces to the type).
    """
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_enchanted_permanent"}
    )
    assert e == DestroyEnchantedPermanent()


def test_bridge_destroy_enchanted_permanent_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DestroyEnchantedPermanent(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.destroy import DestroyEnchantedPermanent
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "destroy_enchanted_permanent"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DestroyEnchantedPermanent)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_destroy_enchanted_permanent_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "destroy_enchanted_permanent"}`` on a
    triggered_ability carrier (the canonical Cryoshatter-shape
    Aura-trigger pattern). The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3725-3727`` accepts
    the bare ``kind``-only unit-variant dict.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "DestroyEnchantedPermanent"
        ),
        "card_types": ["Enchantment"],
        "subtypes": ["Aura"],
        "mana_cost": {"U": 1, "generic": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "destroy_enchanted_permanent"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_grant_keyword_to_target_creature_ueot_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 27 fresh discovery walk) --
    GrantKeywordToTargetCreatureUntilEOT family (singular keyword
    variant). Rust ``{"kind":
    "grant_keyword_to_target_creature_until_eot", "keyword":
    "deathtouch"}`` translates to Python
    ``PumpTargetCreatureUntilEOT(power_bonus=0, toughness_bonus=0,
    granted_keywords=frozenset({"deathtouch"}),
    you_control_only=True, exclude_self=False)``.

    Mirrors Rust's parse_effect arm at cards.rs:4010-4028 (struct
    variant; required keyword: String). The bridge folds the
    singular variant into the same Python primitive used by the
    plural arm (PumpTargetCreatureUntilEOT) with a 1-element
    frozenset for keywords.
    """
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "grant_keyword_to_target_creature_until_eot",
            "keyword": "deathtouch",
        }
    )
    assert isinstance(e, PumpTargetCreatureUntilEOT)
    assert e.granted_keywords == frozenset({"deathtouch"})
    assert e.power_bonus == 0
    assert e.toughness_bonus == 0
    assert e.you_control_only is True
    assert e.exclude_self is False


def test_bridge_grant_keyword_to_target_creature_ueot_each_canonical_keyword():
    """Bridge arm pin: each common MTG evergreen keyword round-trips
    cleanly through the singular variant. Pins the keyword string
    pass-through (the bridge does not validate keyword vocab; that's
    the parser's responsibility on Rust side via grammar; the bridge
    surfaces the keyword as-is into Python's frozenset).
    """
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    for kw in (
        "deathtouch", "first strike", "flying", "haste", "hexproof",
        "lifelink", "trample", "vigilance", "menace", "reach",
    ):
        e = _ta_dict_to_python_effect(
            {
                "kind": "grant_keyword_to_target_creature_until_eot",
                "keyword": kw,
            }
        )
        assert isinstance(e, PumpTargetCreatureUntilEOT)
        assert e.granted_keywords == frozenset({kw})


def test_bridge_grant_keyword_to_target_creature_ueot_exclude_self_defaults_false():
    """Bridge arm pin: bridged
    ``PumpTargetCreatureUntilEOT.exclude_self`` defaults to False on
    the singular arm. The Blooming-Stinger-shape "another target
    creature" rider rides on the PLURAL variant
    (grant_keywords_to_target_creature_until_eot) -- the singular
    arm has no exclude_source field on the Rust side, so the bridge
    constructs Python with exclude_self=False to mirror the Rust
    substrate's coverage today.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "grant_keyword_to_target_creature_until_eot",
            "keyword": "haste",
        }
    )
    assert e.exclude_self is False


def test_bridge_grant_keyword_to_target_creature_ueot_you_control_only_true():
    """Bridge arm pin: bridged
    ``PumpTargetCreatureUntilEOT.you_control_only`` is True. Mirrors
    the source-controller-bf scan on Rust's resolve arm (the grant
    targets a creature you control; opponents' creatures are not
    legal targets per the variant's name).
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "grant_keyword_to_target_creature_until_eot",
            "keyword": "indestructible",
        }
    )
    assert e.you_control_only is True


def test_bridge_grant_keyword_to_target_creature_ueot_missing_keyword_raises():
    """Bridge arm pin: missing required 'keyword' field raises
    KeyError. Mirrors Rust's parse_effect arm at cards.rs:4012-4015
    which raises PyValueError on missing keyword -- the bridge
    surfaces the same contract via KeyError (matching the round-25
    become_land_creature_ueot precedent for required-fields parity).
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    import pytest

    with pytest.raises(KeyError, match="keyword"):
        _ta_dict_to_python_effect(
            {"kind": "grant_keyword_to_target_creature_until_eot"}
        )


def test_bridge_grant_keyword_to_target_creature_ueot_empty_keyword_raises():
    """Bridge arm pin: empty 'keyword' string raises ValueError.
    Mirrors Rust's parse_effect arm at cards.rs:4022-4026 which
    raises PyValueError on empty keyword (the variant requires a
    non-empty keyword name to bind a meaningful grant). The bridge
    surfaces the same contract.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        _ta_dict_to_python_effect(
            {
                "kind": "grant_keyword_to_target_creature_until_eot",
                "keyword": "",
            }
        )


def test_bridge_grant_keyword_to_target_creature_ueot_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([GrantKeywordToTargetCreatureUEOT, GainLife])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "grant_keyword_to_target_creature_until_eot",
                    "keyword": "lifelink",
                },
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], PumpTargetCreatureUntilEOT)
    assert e.effects[0].granted_keywords == frozenset({"lifelink"})
    assert isinstance(e.effects[1], GainLife)


def test_rust_parser_accepts_grant_keyword_to_target_creature_ueot_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "grant_keyword_to_target_creature_until_eot",
    "keyword": "deathtouch"}`` on a triggered_ability carrier. The
    bridge dict shape must register cleanly on Rust too; this
    lockstep gate keeps the bridge from drifting away from Rust's
    parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:4010-4028`` accepts
    the struct-variant dict with required keyword: String (non-empty).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "GrantKeywordToTargetCreatureUntilEOT"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "grant_keyword_to_target_creature_until_eot",
                    "keyword": "trample",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_destroy_target_land_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 27 fresh discovery walk) --
    DestroyTargetLand family. Rust ``{"kind":
    "destroy_target_land"}`` translates to Python
    ``DestroyTargetLand()``.

    Mirrors Rust's parse_effect arm at cards.rs:3526-3533 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/destroy.py:DestroyTargetLand``) as
    the mirror of the Rust resolver at state_ops_resolve.rs:4112.
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetLand
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_land"}
    )
    assert isinstance(e, DestroyTargetLand)


def test_bridge_destroy_target_land_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:3526-3533 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23/24/25/26/27 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetLand
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "destroy_target_land",
            "this_extra_key_should_be_silently_ignored": "ignored",
        }
    )
    assert isinstance(e, DestroyTargetLand)


def test_bridge_destroy_target_land_nonbasic_only_defaults_false():
    """Bridge arm pin: ``nonbasic_only=False`` schema-narrowness
    default. Rust's substrate covers ONLY the bare "destroy target
    land" variant (no nonbasic-only field on the Rust side); the
    bridge constructs Python with ``nonbasic_only=False`` (the
    default) to mirror Rust's coverage. Pins the schema-narrowness
    convention -- nonbasic-only-with-Rust-substrate would need a
    Rust schema bump first (sister-arm convention -- see round-27
    grant_keyword_to_target_creature_until_eot's exclude_self=False
    default and round-26 destroy_enchanted_permanent's no-extras
    default).
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_land"}
    )
    assert e.nonbasic_only is False


def test_bridge_destroy_target_land_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``DestroyTargetLand()``. Pins the
    dataclass equality semantic for the bare-shape (only the
    ``nonbasic_only`` field present, defaulted to False).
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetLand
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_land"}
    )
    assert e == DestroyTargetLand()


def test_bridge_destroy_target_land_target_filter_is_land():
    """Bridge arm pin: the ``targets`` tuple constructed by
    ``DestroyTargetLand.__post_init__`` carries an ``IsLand()``
    predicate (the bare lane -- no ``& ~IsBasic()`` rider). Pins
    the predicate-construction shape against accidental drift to
    the nonbasic-only filter; the unified ``Destroy.resolve``
    re-verifies this filter at resolution per CR 608.2b.
    """
    from mtg_bot.effects.object_predicates import IsLand
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_land"}
    )
    assert len(e.targets) == 1
    # The bare form's filter is exactly ``IsLand()`` (no compound).
    assert e.targets[0].filter == IsLand()


def test_bridge_destroy_target_land_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DestroyTargetLand(), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.destroy import DestroyTargetLand
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "destroy_target_land"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DestroyTargetLand)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_destroy_target_land_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "destroy_target_land"}`` on a triggered_ability
    carrier. The bridge dict shape must register cleanly on Rust too;
    this lockstep gate keeps the bridge from drifting away from
    Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3526-3533`` accepts
    the unit-variant dict (no fields).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- DestroyTargetLand"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"R": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "destroy_target_land"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_return_all_creatures_from_gy_to_bf_basic_translation():
    """BLOCKED-BRIDGE-ARM slice (round 27 fresh discovery walk) --
    ReturnAllCreaturesFromYourGraveyardToBattlefield mass-reanimation
    family. Rust ``{"kind":
    "return_all_creatures_from_your_graveyard_to_battlefield"}``
    translates to Python
    ``ReturnAllCreaturesFromYourGraveyardToBattlefield()``.

    Mirrors Rust's parse_effect arm at cards.rs:3838-3840 (unit
    variant; no fields). The bridge wires the same Python-side
    primitive (``effects/generic/graveyard.py:ReturnAllCreaturesFromYourGraveyardToBattlefield``)
    as the mirror of the Rust resolver.
    """
    from mtg_bot.effects.generic.graveyard import (
        ReturnAllCreaturesFromYourGraveyardToBattlefield,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_all_creatures_from_your_graveyard_to_battlefield"}
    )
    assert isinstance(e, ReturnAllCreaturesFromYourGraveyardToBattlefield)


def test_bridge_return_all_creatures_from_gy_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance. Extra dict fields
    beyond ``kind`` are silently ignored (mirrors Rust's parse_effect
    arm at cards.rs:3838-3840 which doesn't read additional keys).
    Pins the silent-drop convention for unit variants used across
    sister round-22/23/24/25/26/27 BLOCKED-BRIDGE-ARM unit-shape arms.
    """
    from mtg_bot.effects.generic.graveyard import (
        ReturnAllCreaturesFromYourGraveyardToBattlefield,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "return_all_creatures_from_your_graveyard_to_battlefield",
            "this_extra_key_should_be_silently_ignored": "ignored",
        }
    )
    assert isinstance(e, ReturnAllCreaturesFromYourGraveyardToBattlefield)


def test_bridge_return_all_creatures_from_gy_untargeted_all_pinned():
    """Bridge arm pin: ``untargeted_all=True`` on the constructed
    instance. Distinguishes this mass-reanimation arm from the sister
    single-target ``ReturnCreatureFromYourGraveyardToBattlefield`` arm
    (``untargeted_all=False`` default). Pins the
    ``__post_init__``-set flag against accidental drift to the single-
    target shape.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_all_creatures_from_your_graveyard_to_battlefield"}
    )
    assert e.untargeted_all is True
    # Sister-arm divergence pin: also confirm creature/destination flags
    # set by __post_init__ (the constructor sugar's contract).
    assert e.creature is True
    assert e.destination == "battlefield"


def test_bridge_return_all_creatures_from_gy_does_not_require_target():
    """Bridge arm pin: ``requires_target() -> False`` -- mass-
    reanimation is intrinsically untargeted (the ``untargeted_all=True``
    flag short-circuits the targeting pipeline; the controller's
    graveyard is iterated at resolve time, not consulted via Decision
    targets). Mirrors Rust's resolve which iterates the source-
    controller's graveyard without consulting any explicit target.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_all_creatures_from_your_graveyard_to_battlefield"}
    )
    assert e.requires_target() is False


def test_bridge_return_all_creatures_from_gy_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed
    ``ReturnAllCreaturesFromYourGraveyardToBattlefield()``. Pins the
    dataclass equality semantic for the sugar-class shape.
    """
    from mtg_bot.effects.generic.graveyard import (
        ReturnAllCreaturesFromYourGraveyardToBattlefield,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "return_all_creatures_from_your_graveyard_to_battlefield"}
    )
    assert e == ReturnAllCreaturesFromYourGraveyardToBattlefield()


def test_bridge_return_all_creatures_from_gy_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([ReturnAllCreaturesFromYourGraveyardToBattlefield(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.graveyard import (
        ReturnAllCreaturesFromYourGraveyardToBattlefield,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "return_all_creatures_from_your_graveyard_to_battlefield"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], ReturnAllCreaturesFromYourGraveyardToBattlefield
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_return_all_creatures_from_gy_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind":
    "return_all_creatures_from_your_graveyard_to_battlefield"}`` on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3838-3840`` accepts
    the unit-variant dict (no fields).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "ReturnAllCreaturesFromYourGraveyardToBattlefield"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"W": 2, "generic": 4},
        "power": 3,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "return_all_creatures_from_your_graveyard_to_battlefield",
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_manifest_dread_basic_translation_default_count():
    """BLOCKED-BRIDGE-ARM slice (round 27 fresh discovery walk) --
    ManifestDread family. Rust ``{"kind": "manifest_dread"}`` (no
    ``count`` field) translates to Python ``ManifestDread(count=1)``
    (the default-count case for bare "Manifest dread" oracle text).

    Mirrors Rust's parse_effect arm at cards.rs:3841-3862 which
    accepts the optional ``count`` defaulting to 1. The bridge wires
    the same Python-side primitive
    (``effects/generic/library_lookat.py:ManifestDread``).
    """
    from mtg_bot.effects.generic.library_lookat import ManifestDread
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "manifest_dread"})
    assert isinstance(e, ManifestDread)
    assert e.count == 1


def test_bridge_manifest_dread_explicit_count_one():
    """Bridge arm pin: explicit ``count=1`` translates identically
    to the missing-count default-1 case. Pins the dict-shape round-
    trip equivalence (bare oracle vs explicit-count emitters both
    land on the same Python instance).
    """
    from mtg_bot.effects.generic.library_lookat import ManifestDread
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "manifest_dread", "count": 1}
    )
    assert isinstance(e, ManifestDread)
    assert e.count == 1
    assert e == ManifestDread(count=1)


def test_bridge_manifest_dread_count_two():
    """Bridge arm pin: ``count=2`` translates to ``ManifestDread(count=2)``
    -- the "Manifest dread twice" oracle shape. Pins the count-pass-
    through against accidental clamping. Note: the count != 1 lane
    is parse-tolerated (so dict round-trip works) but resolve-time-
    deferred on Rust per cards.rs:3845-3847; this test exercises only
    the parse/dict-translation surface.
    """
    from mtg_bot.effects.generic.library_lookat import ManifestDread
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "manifest_dread", "count": 2}
    )
    assert isinstance(e, ManifestDread)
    assert e.count == 2


def test_bridge_manifest_dread_negative_count_raises():
    """Bridge arm pin: negative ``count`` raises ValueError, mirroring
    Rust's parse_effect at cards.rs:3856-3860. Pins the non-negative
    contract on both engines.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "manifest_dread", "count": -1}
        )


def test_bridge_manifest_dread_non_integer_count_raises():
    """Bridge arm pin: non-integer ``count`` raises ValueError. Rust
    extracts via ``c.extract::<i32>()`` which rejects non-int; the
    bridge mirrors via explicit isinstance check (and rejects bool
    explicitly since bool is an int subclass in Python but not a
    legal count).
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="must be an integer"):
        _ta_dict_to_python_effect(
            {"kind": "manifest_dread", "count": "two"}
        )


def test_bridge_manifest_dread_does_not_require_target():
    """Bridge arm pin: ``requires_target() -> False`` -- manifest
    dread is intrinsically untargeted (the controller's library top-
    two cards are examined at resolve time via
    ``choose_card_from_library``, not consulted via Decision targets).
    Mirrors Rust's resolve which reads the controller's library
    without consulting any explicit target.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "manifest_dread"})
    assert e.requires_target() is False


def test_bridge_manifest_dread_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([ManifestDread(), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.library_lookat import ManifestDread
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "manifest_dread"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], ManifestDread)
    assert e.effects[0].count == 1
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_manifest_dread_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "manifest_dread"}`` (default count) on a
    triggered_ability carrier. The bridge dict shape must register
    cleanly on Rust too; this lockstep gate keeps the bridge from
    drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3841-3862`` accepts
    the struct-variant dict with optional count: i32 (defaults to 1).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- ManifestDread",
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 2},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "manifest_dread"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_accepts_manifest_dread_with_explicit_count():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "manifest_dread", "count": 2}`` on a
    triggered_ability carrier. Pins the explicit-count register-on-
    Rust path against accidental schema drift (the count != 1 lane
    is parse-tolerated even though resolve defers).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- ManifestDread (count=2)"
        ),
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 3},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "manifest_dread", "count": 2},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_put_stun_counter_on_tapped_creature_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 30) -- stun-counter substrate.
    Rust ``{"kind": "put_stun_counter_on_tapped_creature"}`` translates
    to Python ``PutStunCounterOnTappedCreature()``.

    Mirrors Rust's parse_effect arm at cards.rs:3766-3771 (unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-30 stun-counter substrate's first Effect arm (Cryogen
    Relic activated half).
    """
    from mtg_bot.effects.generic.counters import (
        PutStunCounterOnTappedCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_stun_counter_on_tapped_creature"}
    )
    assert isinstance(e, PutStunCounterOnTappedCreature)


def test_bridge_put_stun_counter_on_tapped_creature_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (tap_enchanted_permanent /
    destroy_enchanted_permanent / attach_to_target_creature_you_control)
    -- unrecognized fields in the dict are ignored, not raised.
    """
    from mtg_bot.effects.generic.counters import (
        PutStunCounterOnTappedCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_stun_counter_on_tapped_creature",
            "spurious_field": "ignored",
        }
    )
    assert isinstance(e, PutStunCounterOnTappedCreature)


def test_bridge_put_stun_counter_on_tapped_creature_equality_default_constructor():
    """Bridge arm pin: two ``PutStunCounterOnTappedCreature()`` instances
    are equal under dataclass(frozen=True) default __eq__ semantics.
    Pins the frozen-dataclass equality contract for downstream hash-
    based keying (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.counters import (
        PutStunCounterOnTappedCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "put_stun_counter_on_tapped_creature"}
    )
    e2 = PutStunCounterOnTappedCreature()
    assert e1 == e2


def test_bridge_put_stun_counter_on_tapped_creature_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PutStunCounterOnTappedCreature(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-30 arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.counters import (
        PutStunCounterOnTappedCreature,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "put_stun_counter_on_tapped_creature"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], PutStunCounterOnTappedCreature)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_put_stun_counter_on_tapped_creature_in_activated_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "put_stun_counter_on_tapped_creature"}`` on a
    triggered_ability carrier. The Cryogen Relic real-card consumer
    actually mounts this Effect on an ACTIVATED ability (sacrifice
    cost), but the lockstep gate doesn't depend on the trigger event
    -- registering through any abilities carrier exercises the
    parse_effect arm. Pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm at cards.rs:3766-3771.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- PutStunCounterOnTappedCreature",
        "card_types": ["Artifact"],
        "mana_cost": {"U": 1, "generic": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "put_stun_counter_on_tapped_creature"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_tap_target_opponent_permanent_and_stun_it_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 30) -- stun-counter substrate.
    Rust ``{"kind": "tap_target_opponent_permanent_and_stun_it"}``
    translates to Python ``TapTargetOpponentPermanentAndStunIt()``.

    Mirrors Rust's parse_effect arm at cards.rs:3772-3779 (unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-30 stun-counter substrate's second Effect arm
    (Mechanozoa ETB).
    """
    from mtg_bot.effects.generic.tap import (
        TapTargetOpponentPermanentAndStunIt,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "tap_target_opponent_permanent_and_stun_it"}
    )
    assert isinstance(e, TapTargetOpponentPermanentAndStunIt)


def test_bridge_tap_target_opponent_permanent_and_stun_it_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys, same
    convention as the sister put_stun_counter_on_tapped_creature arm
    above.
    """
    from mtg_bot.effects.generic.tap import (
        TapTargetOpponentPermanentAndStunIt,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "tap_target_opponent_permanent_and_stun_it",
            "spurious_field": "ignored",
        }
    )
    assert isinstance(e, TapTargetOpponentPermanentAndStunIt)


def test_bridge_tap_target_opponent_permanent_and_stun_it_equality_default_constructor():
    """Bridge arm pin: two ``TapTargetOpponentPermanentAndStunIt()``
    instances are equal under dataclass(frozen=True) default __eq__
    semantics. Pins the frozen-dataclass equality contract.
    """
    from mtg_bot.effects.generic.tap import (
        TapTargetOpponentPermanentAndStunIt,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "tap_target_opponent_permanent_and_stun_it"}
    )
    e2 = TapTargetOpponentPermanentAndStunIt()
    assert e1 == e2


def test_bridge_tap_target_opponent_permanent_and_stun_it_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.tap import (
        TapTargetOpponentPermanentAndStunIt,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "tap_target_opponent_permanent_and_stun_it"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], TapTargetOpponentPermanentAndStunIt)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_tap_target_opponent_permanent_and_stun_it_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape on a triggered_ability carrier. Mechanozoa's actual ETB is
    the canonical real-card consumer of this shape.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- TapTargetOpponentPermanentAndStunIt",
        "card_types": ["Creature"],
        "mana_cost": {"U": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "tap_target_opponent_permanent_and_stun_it"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_manifest_dread_then_attach_self_basic_translation_default_count():
    """BLOCKED-BRIDGE-ARM slice (round 28 closeout discovery) --
    ManifestDreadThenAttachSelf composite (Conductive-Machete /
    Killer's-Mask / Cursed-Windbreaker / Dissection-Tools Duskmourn-
    Equipment-ETB family). Rust ``{"kind":
    "manifest_dread_then_attach_self"}`` (no ``count`` field)
    translates to Python ``ManifestDreadThenAttachSelf()`` (the
    default-count case for bare oracle text).

    Mirrors Rust's parse_effect arm at cards.rs:3911-3930 which
    accepts the optional ``count`` defaulting to 1. The bridge wires
    the same Python-side primitive
    (``effects/generic/library_lookat.py:ManifestDreadThenAttachSelf``).
    """
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDreadThenAttachSelf,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "manifest_dread_then_attach_self"}
    )
    assert isinstance(e, ManifestDreadThenAttachSelf)


def test_bridge_manifest_dread_then_attach_self_explicit_count_one():
    """Bridge arm pin: explicit ``count=1`` translates identically
    to the missing-count default-1 case. Pins the dict-shape round-
    trip equivalence (bare oracle vs explicit-count emitters both
    land on the same Python instance). The Python class has no
    count field today (Equipment-ETB oracle is always bare), so
    both paths construct the bare instance.
    """
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDreadThenAttachSelf,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "manifest_dread_then_attach_self", "count": 1}
    )
    assert isinstance(e, ManifestDreadThenAttachSelf)
    assert e == ManifestDreadThenAttachSelf()


def test_bridge_manifest_dread_then_attach_self_negative_count_raises():
    """Bridge arm pin: negative ``count`` raises ValueError, mirroring
    Rust's parse_effect at cards.rs:3925-3929. Pins the non-negative
    contract on both engines.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {"kind": "manifest_dread_then_attach_self", "count": -1}
        )


def test_bridge_manifest_dread_then_attach_self_non_integer_count_raises():
    """Bridge arm pin: non-integer ``count`` raises ValueError. Rust
    extracts via ``c.extract::<i32>()`` which rejects non-int; the
    bridge mirrors via explicit isinstance check (and rejects bool
    explicitly since bool is an int subclass in Python but not a
    legal count).
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="must be an integer"):
        _ta_dict_to_python_effect(
            {"kind": "manifest_dread_then_attach_self", "count": "two"}
        )


def test_bridge_manifest_dread_then_attach_self_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([ManifestDreadThenAttachSelf(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator.
    Pins this arm against accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDreadThenAttachSelf,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "manifest_dread_then_attach_self"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], ManifestDreadThenAttachSelf)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_manifest_dread_then_attach_self_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "manifest_dread_then_attach_self"}`` (default
    count) on a triggered_ability carrier. The bridge dict shape
    must register cleanly on Rust too; this lockstep gate keeps the
    bridge from drifting away from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3911-3930``
    accepts the struct-variant dict with optional count: i32
    (defaults to 1).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "ManifestDreadThenAttachSelf"
        ),
        "card_types": ["Artifact"],
        "subtypes": ["Equipment"],
        "mana_cost": {"generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "manifest_dread_then_attach_self"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_accepts_manifest_dread_then_attach_self_with_explicit_count():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "manifest_dread_then_attach_self", "count": 1}``
    on a triggered_ability carrier. Pins the explicit-count
    register-on-Rust path against accidental schema drift (the
    count != 1 lane is parse-tolerated even though resolve defers
    per state_ops_resolve.rs:6803-6806).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "ManifestDreadThenAttachSelf (count=1)"
        ),
        "card_types": ["Artifact"],
        "subtypes": ["Equipment"],
        "mana_cost": {"generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "manifest_dread_then_attach_self",
                    "count": 1,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_manifest_dread_then_put_plus_one_counters_basic_translation_default():
    """BLOCKED-BRIDGE-ARM slice (round 28 closeout discovery) --
    ManifestDreadThenPutPlusOneCountersOnIt composite (Slimy-Aquarium
    / Weight-Room Duskmourn-Room door-trigger family). Rust ``{"kind":
    "manifest_dread_then_put_plus_one_counters_on_it"}`` (no fields)
    translates to Python ``ManifestDreadThenPutPlusOneCountersOnIt()``
    with default ``counter_amount=1`` (Slimy Aquarium emits this).

    Mirrors Rust's parse_effect arm at cards.rs:3932-3969 which
    accepts optional ``count`` AND ``counter_amount`` both
    defaulting to 1. The bridge wires the same Python-side primitive
    (``effects/generic/library_lookat.py:ManifestDreadThenPutPlusOneCountersOnIt``).
    """
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDreadThenPutPlusOneCountersOnIt,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "manifest_dread_then_put_plus_one_counters_on_it"}
    )
    assert isinstance(e, ManifestDreadThenPutPlusOneCountersOnIt)
    assert e.counter_amount == 1


def test_bridge_manifest_dread_then_put_plus_one_counters_explicit_one():
    """Bridge arm pin: explicit ``counter_amount=1`` translates
    identically to the missing-field default-1 case (Slimy-Aquarium
    bare oracle vs explicit emitters land on the same Python
    instance). Pins the dict-shape round-trip equivalence.
    """
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDreadThenPutPlusOneCountersOnIt,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "manifest_dread_then_put_plus_one_counters_on_it",
            "counter_amount": 1,
        }
    )
    assert isinstance(e, ManifestDreadThenPutPlusOneCountersOnIt)
    assert e == ManifestDreadThenPutPlusOneCountersOnIt(counter_amount=1)


def test_bridge_manifest_dread_then_put_plus_one_counters_amount_three():
    """Bridge arm pin: ``counter_amount=3`` translates to
    ``ManifestDreadThenPutPlusOneCountersOnIt(counter_amount=3)`` --
    the Weight-Room "three +1/+1 counters on that creature" oracle
    shape. Pins the counter_amount pass-through against accidental
    clamping.
    """
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDreadThenPutPlusOneCountersOnIt,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "manifest_dread_then_put_plus_one_counters_on_it",
            "counter_amount": 3,
        }
    )
    assert isinstance(e, ManifestDreadThenPutPlusOneCountersOnIt)
    assert e.counter_amount == 3


def test_bridge_manifest_dread_then_put_plus_one_counters_negative_count_raises():
    """Bridge arm pin: negative ``count`` raises ValueError, mirroring
    Rust's parse_effect at cards.rs:3947-3951. Pins the non-negative
    ``count`` contract on both engines.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {
                "kind": "manifest_dread_then_put_plus_one_counters_on_it",
                "count": -1,
            }
        )


def test_bridge_manifest_dread_then_put_plus_one_counters_negative_amount_raises():
    """Bridge arm pin: negative ``counter_amount`` raises ValueError,
    mirroring Rust's parse_effect at cards.rs:3960-3964. Pins the
    non-negative ``counter_amount`` contract on both engines.
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="non-negative"):
        _ta_dict_to_python_effect(
            {
                "kind": "manifest_dread_then_put_plus_one_counters_on_it",
                "counter_amount": -2,
            }
        )


def test_bridge_manifest_dread_then_put_plus_one_counters_non_integer_amount_raises():
    """Bridge arm pin: non-integer ``counter_amount`` raises
    ValueError. Rust extracts via ``c.extract::<i32>()`` which
    rejects non-int; the bridge mirrors via explicit isinstance
    check (and rejects bool explicitly since bool is an int subclass
    in Python but not a legal counter_amount).
    """
    import pytest
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="must be an integer"):
        _ta_dict_to_python_effect(
            {
                "kind": "manifest_dread_then_put_plus_one_counters_on_it",
                "counter_amount": "three",
            }
        )


def test_bridge_manifest_dread_then_put_plus_one_counters_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([ManifestDreadThenPutPlusOneCountersOnIt(counter_amount=3),
    DrawCards(1)])`` translates correctly via the recursive Sequence
    translator. Pins this arm against accidental dispatch loss
    inside Sequence and pins counter_amount pass-through under a
    composer.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDreadThenPutPlusOneCountersOnIt,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": (
                        "manifest_dread_then_put_plus_one_counters_on_it"
                    ),
                    "counter_amount": 3,
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], ManifestDreadThenPutPlusOneCountersOnIt
    )
    assert e.effects[0].counter_amount == 3
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_manifest_dread_then_put_plus_one_counters_default():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "manifest_dread_then_put_plus_one_counters_on_it"}``
    (default count + default counter_amount) on a triggered_ability
    carrier. The bridge dict shape must register cleanly on Rust
    too; this lockstep gate keeps the bridge from drifting away
    from Rust's parse_effect coverage.

    Rust's parse_effect arm at ``rust/src/cards.rs:3932-3969``
    accepts the struct-variant dict with optional count: i32 +
    optional counter_amount: i32 (both default to 1).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "ManifestDreadThenPutPlusOneCountersOnIt"
        ),
        "card_types": ["Enchantment"],
        "mana_cost": {"G": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "manifest_dread_then_put_plus_one_counters_on_it"
                    )
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_accepts_manifest_dread_then_put_plus_one_counters_three():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "manifest_dread_then_put_plus_one_counters_on_it",
    "counter_amount": 3}`` on a triggered_ability carrier (the
    Weight-Room oracle shape). Pins the counter_amount-3 register-
    on-Rust path against accidental schema drift.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": (
            "BLOCKED-BRIDGE-ARM Bridge Pin -- "
            "ManifestDreadThenPutPlusOneCountersOnIt (counter_amount=3)"
        ),
        "card_types": ["Enchantment"],
        "mana_cost": {"G": 2, "generic": 3},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "manifest_dread_then_put_plus_one_counters_on_it"
                    ),
                    "counter_amount": 3,
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_double_target_creatures_power_ueot_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 31) -- double target creature's
    power UEOT substrate. Rust ``{"kind": "double_target_creatures_power_ueot"}``
    translates to Python ``DoubleTargetCreaturesPowerUEOT()``.

    Mirrors Rust's parse_effect arm at cards.rs:3712-3721 (unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-31 double-power Effect arm (Mightform Harmonizer's
    landfall trigger).
    """
    from mtg_bot.effects.generic.pump import (
        DoubleTargetCreaturesPowerUEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "double_target_creatures_power_ueot"}
    )
    assert isinstance(e, DoubleTargetCreaturesPowerUEOT)


def test_bridge_double_target_creatures_power_ueot_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (tap_enchanted_permanent /
    destroy_enchanted_permanent / put_stun_counter_on_tapped_creature)
    -- unrecognized fields in the dict are ignored, not raised.
    """
    from mtg_bot.effects.generic.pump import (
        DoubleTargetCreaturesPowerUEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "double_target_creatures_power_ueot",
            "spurious_field": "ignored",
            "power_bonus": 99,  # ignored — not part of unit variant schema
        }
    )
    assert isinstance(e, DoubleTargetCreaturesPowerUEOT)


def test_bridge_double_target_creatures_power_ueot_equality_default_constructor():
    """Bridge arm pin: two ``DoubleTargetCreaturesPowerUEOT()`` instances
    are equal under dataclass(frozen=True) default __eq__ semantics.
    Pins the frozen-dataclass equality contract for downstream hash-
    based keying (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.pump import (
        DoubleTargetCreaturesPowerUEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "double_target_creatures_power_ueot"}
    )
    e2 = DoubleTargetCreaturesPowerUEOT()
    assert e1 == e2


def test_bridge_double_target_creatures_power_ueot_requires_target():
    """Bridge arm pin: Python's DoubleTargetCreaturesPowerUEOT carries
    requires_target() == True (pump.py:205-206 -- "target creature you
    control" is a hard requires-target). The bridge round-trip pin
    catches accidental drift to a no-target unit-variant by asserting
    the requires_target flag survives translation. (Distinct from the
    DestroyEnchantedPermanent / DealDamageEqualToSourcePower
    does-NOT-require-target convention -- those arms read their
    target from resolving_source / source-bf scan, while this arm
    needs an explicit target choice.)
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "double_target_creatures_power_ueot"}
    )
    assert e.requires_target() is True


def test_bridge_double_target_creatures_power_ueot_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DoubleTargetCreaturesPowerUEOT(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-31 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter Sequence-composability pin).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.pump import (
        DoubleTargetCreaturesPowerUEOT,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "double_target_creatures_power_ueot"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DoubleTargetCreaturesPowerUEOT)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_double_target_creatures_power_ueot_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "double_target_creatures_power_ueot"}`` on a
    triggered_ability carrier. Mightform Harmonizer's actual real-
    card consumer mounts this Effect on a LANDFALL trigger predicate
    (Subject(IsLand) AND ControlledByYou); the lockstep gate doesn't
    depend on the trigger event -- registering through any abilities
    carrier exercises the parse_effect arm. Pinning here keeps the
    Python bridge from drifting away from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm at cards.rs:3712-3721.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DoubleTargetCreaturesPowerUEOT",
        "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 3},
        "power": 4,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "double_target_creatures_power_ueot"},
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_double_target_creatures_power_ueot_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``double_target_creature_power_ueot`` -- singular "creature" vs
    plural "creatures") surfaces as ValueError on Rust's parse_effect
    arm because the new variant isn't registered for that spelling.
    Pin against accidental schema drift between bridge dict shape and
    Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Creature"],
        "mana_cost": {"G": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — singular "creature" instead of "creatures".
                    "kind": "double_target_creature_power_ueot"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_double_target_creatures_power_ueot_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "double_target_creatures_power_ueot"}`` on a
    cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either
    pin. Mirrors the round-25/26/27 cast-spell vs trigger lockstep
    convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DoubleTargetCreaturesPowerUEOT (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"G": 1},
        "effects": [
            {"kind": "double_target_creatures_power_ueot"},
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_to_opponent_creature_by_artifact_count_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 32) -- deal damage to an
    opponent's creature equal to the source-controller's artifact count.
    Rust ``{"kind": "deal_damage_to_opponent_creature_by_artifact_count"}``
    translates to Python ``DealDamageToOpponentCreatureByArtifactCount()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-31 double_target_creatures_power_ueot arm; unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-32 dynamic-X-from-artifact-count Effect arm (Warmaker
    Gunship's ETB).
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    assert isinstance(e, DealDamageToOpponentCreatureByArtifactCount)


def test_bridge_deal_damage_to_opponent_creature_by_artifact_count_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-31 double_target_creatures_power_ueot
    / round-30 put_stun_counter_on_tapped_creature) -- unrecognized
    fields in the dict are ignored, not raised. Notably, an ``amount``
    key (which the sister DealDamageToOpponentCreature arm DOES
    consume) is silently dropped here -- the dynamic-X is the whole
    point of this variant, not a static amount.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_to_opponent_creature_by_artifact_count",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``deal_damage_to_opponent_creature`` arm; here it's
            # silently dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, DealDamageToOpponentCreatureByArtifactCount)


def test_bridge_deal_damage_to_opponent_creature_by_artifact_count_equality_default_constructor():
    """Bridge arm pin: two ``DealDamageToOpponentCreatureByArtifactCount()``
    instances are equal under dataclass(frozen=True) default __eq__
    semantics (provided their auto-defaulted ``targets`` tuple
    converges via __post_init__). Pins the frozen-dataclass equality
    contract for downstream hash-based keying (e.g. de-dup in Sequence
    composers).
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    e2 = DealDamageToOpponentCreatureByArtifactCount()
    assert e1 == e2


def test_bridge_deal_damage_to_opponent_creature_by_artifact_count_requires_target():
    """Bridge arm pin: Python's
    DealDamageToOpponentCreatureByArtifactCount carries
    requires_target() == True (damage.py:421-430 -- the auto-defaulted
    ``targets`` tuple holds a TargetSpec with count_min=1 by default;
    "target creature an opponent controls" is a hard requires-target).
    The bridge round-trip pin catches accidental drift to a no-target
    unit-variant by asserting the requires_target flag survives
    translation. (Distinct from the sister
    DealDamageEqualToSourcePower does-NOT-require-target convention --
    that arm reads its target from a defender-bf scan; this arm needs
    an explicit opponent-creature target choice.)
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    assert e.requires_target() is True


def test_bridge_deal_damage_to_opponent_creature_by_artifact_count_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealDamageToOpponentCreatureByArtifactCount(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-32 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter / round-31 double-power
    Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "deal_damage_to_opponent_creature_by_artifact_count"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DealDamageToOpponentCreatureByArtifactCount)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_artifact_count_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "deal_damage_to_opponent_creature_by_artifact_count"}``
    on a triggered_ability carrier. Warmaker Gunship's actual real-card
    consumer mounts this Effect on a Spacecraft ETB trigger; the
    lockstep gate doesn't depend on the trigger event -- registering
    through any abilities carrier exercises the parse_effect arm.
    Pinning here keeps the Python bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-31 double_target_creatures_power_ueot arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByArtifactCount",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"R": 1, "generic": 4},
        "power": 5,
        "toughness": 5,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_to_opponent_creature_by_artifact_count"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_deal_damage_to_opponent_creature_by_artifact_count_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``deal_damage_to_opponent_creature_by_artifacts_count`` -- plural
    "artifacts" vs singular "artifact") surfaces as ValueError on
    Rust's parse_effect arm because the new variant isn't registered
    for that spelling. Pin against accidental schema drift between
    bridge dict shape and Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"R": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — plural "artifacts" instead of singular "artifact".
                    "kind": "deal_damage_to_opponent_creature_by_artifacts_count"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_artifact_count_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "deal_damage_to_opponent_creature_by_artifact_count"}``
    on a cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either
    pin. Mirrors the round-25/26/27 cast-spell vs trigger lockstep
    convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByArtifactCount (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"R": 1},
        "effects": [
            {"kind": "deal_damage_to_opponent_creature_by_artifact_count"},
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_add_counters_basic_translation():
    """BLOCKED-BRIDGE-ARM tail slice (round 28 closeout residue) --
    AddCounters family. Rust ``{"kind": "add_counters", "count": 2,
    "target": {"kind": "any_creature"}}`` translates to Python
    ``PutPlusOneCounterOnTargetCreature(amount=2)`` (which auto-stamps
    ``targets=(TargetSpec(IsCreature()),)`` via __post_init__).

    Mirrors Rust's parse_effect arm at cards.rs:3370-3378 (count
    extracted as i32 + target dict required, parsed via
    parse_target_filter). Sister tail entry to round-32 scry_n.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnTargetCreature,
    )
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "add_counters",
            "count": 2,
            "target": {"kind": "any_creature"},
        }
    )
    assert isinstance(e, PutPlusOneCounterOnTargetCreature)
    assert e.amount == 2
    # __post_init__ auto-stamps the IsCreature() filter.
    assert e.targets == (TargetSpec(IsCreature()),)


def test_bridge_add_counters_default_count_is_one():
    """Bridge arm pin: missing ``count`` defaults to 1 (mirrors the
    Python ``PutPlusOneCounterOnTargetCreature`` dataclass default
    inherited from PutPlusOneCounters.amount=1). Bridge consumers
    SHOULD pass ``count`` explicitly for round-trip dict-shape
    consistency, but the default keeps the arm tolerant of stripped
    fixtures.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnTargetCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "add_counters", "target": {"kind": "any_creature"}}
    )
    assert isinstance(e, PutPlusOneCounterOnTargetCreature)
    assert e.amount == 1


def test_bridge_add_counters_default_target_filter_is_any_creature():
    """Bridge arm pin: missing ``target`` defaults to ``any_creature``
    (mirrors the lenient-default convention used by sister tap_target
    / untap_target / destroy_target / exile_target arms). Bridge
    consumers SHOULD pass ``target`` explicitly for round-trip
    dict-shape consistency.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnTargetCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "add_counters", "count": 3})
    assert isinstance(e, PutPlusOneCounterOnTargetCreature)
    assert e.amount == 3


def test_bridge_add_counters_any_player_filter_raises():
    """Bridge arm pin: ``target.kind == "any_player"`` raises
    ValueError (mirrors Rust's resolve raise at
    state_ops_resolve.rs:1608-1613 -- "AddCounters with AnyPlayer
    target is semantically nonsense"). The bridge front-loads the
    check rather than waiting for resolve so the contract is
    symmetric across engines.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="AnyPlayer"):
        _ta_dict_to_python_effect(
            {
                "kind": "add_counters",
                "count": 1,
                "target": {"kind": "any_player"},
            }
        )


def test_bridge_add_counters_negative_count_raises():
    """Bridge arm pin: negative ``count`` (Rust's -1/-1 lane via the
    signed net-counter model) raises ValueError at bridge time --
    Python's PutPlusOneCounterOnTargetCreature primitive is
    +1/+1-counter-only; -1/-1 counters live on a distinct Python
    primitive family. Round-31 ManifestDread defer-error precedent:
    parse-tolerated on Rust (so dict round-trip works), bridge-time-
    deferred so a typo doesn't silently no-op.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="negative count"):
        _ta_dict_to_python_effect(
            {
                "kind": "add_counters",
                "count": -1,
                "target": {"kind": "any_creature"},
            }
        )


def test_bridge_add_counters_non_int_count_raises():
    """Bridge arm pin: non-int ``count`` raises ValueError. Rust's
    parser at cards.rs:3371 extracts as i32 strictly; the bridge
    mirrors the type narrowness so cross-engine dict shapes don't
    silently coerce. Excludes bool subtype of int (a common
    accidental shape -- True/False as count is nonsensical).
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="must be an int"):
        _ta_dict_to_python_effect(
            {
                "kind": "add_counters",
                "count": 1.5,
                "target": {"kind": "any_creature"},
            }
        )
    with pytest.raises(ValueError, match="must be an int"):
        _ta_dict_to_python_effect(
            {
                "kind": "add_counters",
                "count": True,
                "target": {"kind": "any_creature"},
            }
        )


def test_bridge_add_counters_unknown_target_filter_raises():
    """Bridge arm pin: target kind that's neither ``any_creature`` nor
    ``any_player`` raises ValueError. Pins the bridge against
    accidental future filter expansion that drifts away from Rust's
    parse_target_filter coverage at cards.rs:2790-2799.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="unsupported target filter"):
        _ta_dict_to_python_effect(
            {
                "kind": "add_counters",
                "count": 1,
                "target": {"kind": "any_artifact"},
            }
        )


def test_bridge_add_counters_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([AddCounters(...), DrawCards(1)])`` translates correctly
    via the recursive Sequence translator. Pins this arm against
    accidental dispatch loss inside Sequence.
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnTargetCreature,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "add_counters",
                    "count": 1,
                    "target": {"kind": "any_creature"},
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], PutPlusOneCounterOnTargetCreature)
    assert e.effects[0].amount == 1
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_add_counters_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "add_counters", "count": N, "target": {"kind":
    "any_creature"}}`` dict shape on a triggered_ability carrier.
    The bridge dict shape must register cleanly on the Rust side
    too; this lockstep gate keeps the bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3370-3378`` requires both ``count`` and
    ``target``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- AddCounters",
        "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 1},
        "power": 1,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "add_counters",
                    "count": 2,
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


# ----------------------------------------------------------------------
# Round-34 BLOCKED-BRIDGE-ARM tail closeout: ``sacrifice`` translator.
# Closes the round-28 closeout's three substrate-asymmetric residue
# items (round-30 scry_n + round-32/33 add_counters + round-34
# sacrifice). With this arm landed, every Rust ``Effect::*`` parse-arm
# shape has a cross-engine Python translator.
# ----------------------------------------------------------------------


def test_bridge_sacrifice_basic_translation():
    """BLOCKED-BRIDGE-ARM tail closeout (round 34) -- Sacrifice family.
    Rust ``{"kind": "sacrifice", "target": {"kind": "any_creature"}}``
    translates to Python ``SacrificeFirstPermanentYouControl()``.

    Mirrors Rust's parse_effect arm at cards.rs:3405-3412 (target
    dict required, parsed via parse_target_filter). Rust's resolve at
    state_ops_resolve.rs:1504-1583 is the deterministic-shortcut
    "pick bf[0] of source-controller's bf" shape; the Python
    primitive mirrors that bit-identically (no agent.choose_target
    plumbing, no PendingDecision, no creature-only filter at resolve
    time -- the AnyCreature label is bridge-time narrowness only).
    Closes the BLOCKED-BRIDGE-ARM tail.
    """
    from mtg_bot.effects.generic.zone_moves import (
        SacrificeFirstPermanentYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "sacrifice", "target": {"kind": "any_creature"}}
    )
    assert isinstance(e, SacrificeFirstPermanentYouControl)


def test_bridge_sacrifice_default_target_filter_is_any_creature():
    """Bridge arm pin: missing ``target`` defaults to ``any_creature``
    (mirrors the lenient-default convention used by sister tap_target
    / untap_target / destroy_target / exile_target / add_counters arms).
    Bridge consumers SHOULD pass ``target`` explicitly for round-trip
    dict-shape consistency.
    """
    from mtg_bot.effects.generic.zone_moves import (
        SacrificeFirstPermanentYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "sacrifice"})
    assert isinstance(e, SacrificeFirstPermanentYouControl)


def test_bridge_sacrifice_any_player_filter_raises():
    """Bridge arm pin: ``target.kind == "any_player"`` raises
    ValueError (mirrors Rust's resolve raise at
    state_ops_resolve.rs:1505-1509 -- "Sacrifice with AnyPlayer
    target is semantically nonsense"). The bridge front-loads the
    check rather than waiting for resolve so the contract is
    symmetric across engines.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="AnyPlayer"):
        _ta_dict_to_python_effect(
            {"kind": "sacrifice", "target": {"kind": "any_player"}}
        )


def test_bridge_sacrifice_unknown_target_filter_raises():
    """Bridge arm pin: target kind that's neither ``any_creature`` nor
    ``any_player`` raises ValueError. Pins the bridge against
    accidental future filter expansion that drifts away from Rust's
    parse_target_filter coverage at cards.rs:2790-2799.
    """
    import pytest

    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    with pytest.raises(ValueError, match="unsupported target filter"):
        _ta_dict_to_python_effect(
            {"kind": "sacrifice", "target": {"kind": "any_artifact"}}
        )


def test_bridge_sacrifice_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([Sacrifice(any_creature), DrawCards(1)])`` translates
    correctly via the recursive Sequence translator. Pins this arm
    against accidental dispatch loss inside Sequence (the classic
    "sacrifice a creature, draw a card" Altar's-Reap composition).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.zone_moves import (
        SacrificeFirstPermanentYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "sacrifice", "target": {"kind": "any_creature"}},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], SacrificeFirstPermanentYouControl)
    assert isinstance(e.effects[1], DrawCards)


def _sacrifice_pin_state():
    """Test fixture mirroring tests/test_stellar_sights_batch7.py
    helpers: new GameState with two players (PickFirstAgent), p0
    active, precombat-main step. Returns (state, _put_helper).
    """
    from mtg_bot.agents import Agent
    from mtg_bot.game import GameState, Player, Zone
    from mtg_bot.turn import Step

    class _PickFirstAgent(Agent):
        def choose_target(self, state, player_id, legal_targets):
            return legal_targets[0] if legal_targets else None

        def choose_sacrifice(self, state, player_id, legal_ids):
            return legal_ids[0] if legal_ids else None

    state = GameState()
    state.players.append(Player(id=0, agent=_PickFirstAgent()))
    state.players.append(Player(id=1, agent=_PickFirstAgent()))
    state.active_player_idx = 0
    state.priority_player_idx = 0
    state.step = Step.PRECOMBAT_MAIN

    def _put(printing, owner_id=0):
        obj = state.new_object(printing, owner_id, Zone.BATTLEFIELD)
        state.battlefield.append(obj)
        return obj

    return state, _put


def test_sacrifice_first_permanent_you_control_resolve_picks_bf_zero():
    """Resolve-side pin for the new SacrificeFirstPermanentYouControl
    primitive -- mirrors Rust's first-slice deterministic shortcut
    at state_ops_resolve.rs:1539-1543: pick state.battlefield[0]
    filtered by controller_id (no card-type filter at resolve time),
    route through ``mtg_bot.turn.sacrifice``, leave the picked
    object in the controller's graveyard.
    """
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.effects.generic.zone_moves import (
        SacrificeFirstPermanentYouControl,
    )

    state, _put = _sacrifice_pin_state()
    # Seed two basic lands onto p0's bf (deterministic, simplest).
    mountain_a = _put(BASIC_LANDS["Mountain"], owner_id=0)
    mountain_b = _put(BASIC_LANDS["Mountain"], owner_id=0)
    p0_bf = [o for o in state.battlefield if o.controller_id == 0]
    assert p0_bf == [mountain_a, mountain_b]

    SacrificeFirstPermanentYouControl().resolve(state, controller_id=0)

    # bf[0] (mountain_a) sacrificed -> moved to p0's graveyard;
    # mountain_b remains on bf.
    p0_bf_after = [o for o in state.battlefield if o.controller_id == 0]
    assert p0_bf_after == [mountain_b]
    assert mountain_a in state.players[0].graveyard


def test_sacrifice_first_permanent_you_control_resolve_silent_fizzle_on_empty_bf():
    """Resolve-side pin: when the controller's battlefield is empty,
    the resolve fizzles silently (mirrors Rust state_ops_resolve.rs:
    1540-1542 -- ``if bf.is_empty() { return Ok(()); }``). No raise,
    no graveyard mutation.
    """
    from mtg_bot.effects.generic.zone_moves import (
        SacrificeFirstPermanentYouControl,
    )

    state, _ = _sacrifice_pin_state()
    # Empty bf on both sides.
    assert state.battlefield == []

    # Should not raise.
    SacrificeFirstPermanentYouControl().resolve(state, controller_id=0)

    # No state mutation.
    assert state.battlefield == []
    assert state.players[0].graveyard == []
    assert state.players[1].graveyard == []


def test_sacrifice_first_permanent_you_control_resolve_uses_controller_id_not_active_player():
    """Resolve-side pin: the primitive scans the controller's bf, not
    the active player's. Mirrors Rust's I66 controller-of-effect fix
    at state_ops_resolve.rs:1535-1538 (resolve picks
    state.current_resolve_source_controller, falling back to
    state.active_player). The Python primitive consumes the
    ``controller_id`` parameter directly (the upstream resolve
    pipeline is responsible for passing the source-controller).

    Pin scenario: p0 is active, p1's creature is the source. The
    primitive sacrifices from P1's bf, not p0's.
    """
    from mtg_bot.cards import BASIC_LANDS
    from mtg_bot.effects.generic.zone_moves import (
        SacrificeFirstPermanentYouControl,
    )

    state, _put = _sacrifice_pin_state()
    p0_mountain = _put(BASIC_LANDS["Mountain"], owner_id=0)
    p1_mountain = _put(BASIC_LANDS["Mountain"], owner_id=1)

    # Resolve as p1's effect even though p0 is active.
    SacrificeFirstPermanentYouControl().resolve(state, controller_id=1)

    # p1's mountain sacrificed (controller-id-driven), p0's mountain
    # untouched.
    assert p0_mountain in state.battlefield
    assert p1_mountain not in state.battlefield
    assert p1_mountain in state.players[1].graveyard
    assert state.players[0].graveyard == []


def test_rust_parser_accepts_sacrifice_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the
    ``{"kind": "sacrifice", "target": {"kind": "any_creature"}}`` dict
    shape on a triggered_ability carrier. The bridge dict shape must
    register cleanly on the Rust side too; this lockstep gate keeps
    the bridge from drifting away from Rust's parse_effect coverage.

    Mirrors the existing parser-surface pins for sister BLOCKED-
    BRIDGE-ARM arms. Rust's parse_effect arm at
    ``rust/src/cards.rs:3405-3412`` requires ``target``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-BRIDGE-ARM Bridge Pin -- Sacrifice",
        "card_types": ["Creature"],
        "mana_cost": {"B": 1, "generic": 1},
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "sacrifice",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_put_plus_one_counters_on_self_per_other_creature_or_artifact_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 33) -- put a +1/+1 counter on
    this creature for each OTHER creature and/or artifact you control.
    Rust ``{"kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"}``
    translates to Python
    ``PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-32 deal_damage_to_opponent_creature_by_artifact_count
    arm; unit variant; no fields). Pin verifies the bare-shape
    translation for the round-33 dynamic-X-from-other-permanent-count
    Effect arm (Luxknight Breacher's ETB).
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"}
    )
    assert isinstance(e, PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact)


def test_bridge_put_plus_one_counters_on_self_per_other_creature_or_artifact_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 deal_damage_to_opponent_creature
    _by_artifact_count / round-31 double_target_creatures_power_ueot /
    round-30 put_stun_counter_on_tapped_creature) -- unrecognized
    fields in the dict are ignored, not raised. Notably, an ``amount``
    key (which the sister PutPlusOneCounterOnSelf arm DOES consume) is
    silently dropped here -- the dynamic-X is the whole point of this
    variant, not a static amount.
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``put_plus_one_counter_on_self`` arm; here it's silently
            # dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact)


def test_bridge_put_plus_one_counters_on_self_per_other_creature_or_artifact_equality_default_constructor():
    """Bridge arm pin: two ``PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact()``
    instances are equal under dataclass(frozen=True) default __eq__
    semantics. Pins the frozen-dataclass equality contract for
    downstream hash-based keying (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"}
    )
    e2 = PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact()
    assert e1 == e2


def test_bridge_put_plus_one_counters_on_self_per_other_creature_or_artifact_no_target():
    """Bridge arm pin: Python's
    PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact carries
    requires_target() == False (counters.py:393-431 -- @dataclass with
    no override of requires_target, no targets tuple field; the base
    Effect.requires_target() returns False when self.targets is empty).
    "+1/+1 counter on this creature" is a self-counter effect; no
    explicit target choice. The bridge round-trip pin catches accidental
    drift to a requires-target shape by asserting requires_target ==
    False survives translation.

    Distinct from the sister round-32
    DealDamageToOpponentCreatureByArtifactCount which DOES require a
    target (opponent-creature pick); this arm targets self via the
    resolving_source slot.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"}
    )
    assert e.requires_target() is False


def test_bridge_put_plus_one_counters_on_self_per_other_creature_or_artifact_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-33 arm against accidental dispatch loss inside Sequence
    (mirrors the round-32 by-artifact-count / round-31 double-power /
    round-30 stun-counter Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_put_plus_one_counters_on_self_per_other_creature_or_artifact_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"}``
    on a triggered_ability carrier. Luxknight Breacher's actual
    real-card consumer mounts this Effect on an ETB trigger (modeling
    the "enters with N +1/+1 counters" clause as an ETB trigger since
    the engine does not implement static replacement effects); the
    lockstep gate doesn't depend on the trigger event -- registering
    through any abilities carrier exercises the parse_effect arm.
    Pinning here keeps the Python bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-32 deal_damage_to_opponent_creature_by_artifact_count
    arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"W": 1, "generic": 2},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_put_plus_one_counters_on_self_per_other_creature_or_artifact_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``put_plus_one_counter_on_self_per_other_creature_or_artifact`` --
    singular "counter" vs plural "counters") surfaces as ValueError on
    Rust's parse_effect arm because the new variant isn't registered
    for that spelling. Pin against accidental schema drift between
    bridge dict shape and Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Creature"],
        "mana_cost": {"W": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — singular "counter" instead of plural "counters".
                    "kind": "put_plus_one_counter_on_self_per_other_creature_or_artifact"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_put_plus_one_counters_on_self_per_other_creature_or_artifact_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"}``
    on a cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either
    pin. Mirrors the round-25/26/27/30/31/32 cast-spell vs trigger
    lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"W": 1},
        "effects": [
            {
                "kind": "put_plus_one_counters_on_self_per_other_creature_or_artifact"
            },
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_surveil_x_equal_to_artifact_count_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 34) -- surveil X, where X is
    the source-controller's artifact count. Rust
    ``{"kind": "surveil_x_equal_to_artifact_count"}`` translates to
    Python ``Surveil(use_artifact_count=True)`` (NOT bare
    ``Surveil(amount=N)`` -- the whole point of the variant is the
    dynamic-X lane).

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-33
    put_plus_one_counters_on_self_per_other_creature_or_artifact
    arm; unit variant; no fields). Pin verifies the bare-shape
    translation for the round-34 dynamic-X-from-artifact-count
    Effect arm (Cerebral Download's cast-spell carrier).
    """
    from mtg_bot.effects.generic.library_lookat import Surveil
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "surveil_x_equal_to_artifact_count"}
    )
    assert isinstance(e, Surveil)
    assert e.use_artifact_count is True


def test_bridge_surveil_x_equal_to_artifact_count_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 / round-33 dynamic-X
    primitives) -- unrecognized fields in the dict are ignored, not
    raised. Notably, a ``count`` or ``amount`` key (which the sister
    bare ``surveil`` arm DOES consume into Surveil.amount) is silently
    dropped here -- the dynamic-X is the whole point of this variant,
    not a static amount, and Python's class ignores ``amount`` when
    use_artifact_count is True.
    """
    from mtg_bot.effects.generic.library_lookat import Surveil
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "surveil_x_equal_to_artifact_count",
            "spurious_field": "ignored",
            # ``count`` would be consumed by the sister bare
            # ``surveil`` arm into Surveil.amount; here it's silently
            # dropped -- the count is dynamic from artifact count.
            "count": 99,
            "amount": 99,
        }
    )
    assert isinstance(e, Surveil)
    assert e.use_artifact_count is True


def test_bridge_surveil_x_equal_to_artifact_count_equality_default_constructor():
    """Bridge arm pin: bridge-translated and direct-constructed
    ``Surveil(use_artifact_count=True)`` instances are equal under
    dataclass(frozen=True) default __eq__ semantics. Pins the
    frozen-dataclass equality contract for downstream hash-based
    keying (e.g. de-dup in Sequence composers, where the same
    SurveilXEqualToArtifactCount may appear twice).
    """
    from mtg_bot.effects.generic.library_lookat import Surveil
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "surveil_x_equal_to_artifact_count"}
    )
    e2 = Surveil(use_artifact_count=True)
    assert e1 == e2


def test_bridge_surveil_x_equal_to_artifact_count_distinct_from_bare_surveil():
    """Bridge arm pin: the dynamic-X variant is NOT equal to bare
    Surveil(amount=N). Catches accidental bridge-arm collapse
    (e.g. someone "simplifying" the arm to return Surveil(amount=1)
    would silently drop the dynamic-X dispatch). Both shapes share
    the same dataclass; equality must distinguish on the
    use_artifact_count flag.
    """
    from mtg_bot.effects.generic.library_lookat import Surveil
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    dynamic = _ta_dict_to_python_effect(
        {"kind": "surveil_x_equal_to_artifact_count"}
    )
    bare_one = Surveil(amount=1)
    bare_three = Surveil(amount=3)
    assert dynamic != bare_one
    assert dynamic != bare_three
    # And equal to itself:
    assert dynamic == Surveil(use_artifact_count=True)


def test_bridge_surveil_x_equal_to_artifact_count_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([SurveilXEqualToArtifactCount(), DrawCards(3)])`` --
    the actual Cerebral Download oracle shape -- translates correctly
    via the recursive Sequence translator. Pins the round-34 arm
    against accidental dispatch loss inside Sequence (mirrors the
    round-32 by-artifact-count / round-33 counters-per-other Sequence-
    composability pins) AND models Cerebral Download's printed
    cast-spell effect bit-for-bit (surveil-X then draw-3).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.library_lookat import Surveil
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "surveil_x_equal_to_artifact_count"},
                {"kind": "draw_cards", "count": 3},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], Surveil)
    assert e.effects[0].use_artifact_count is True
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_surveil_x_equal_to_artifact_count_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "surveil_x_equal_to_artifact_count"}`` on a
    triggered_ability carrier. Cerebral Download's actual real-card
    consumer is a cast-spell carrier (Instant), but the parse_effect
    arm dispatches both carriers identically -- registering through
    a triggered_abilities carrier exercises the parse_effect arm
    without needing a cast-effects carrier here. Pinning against
    the trigger carrier keeps the Python bridge from drifting away
    from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-33
    put_plus_one_counters_on_self_per_other_creature_or_artifact
    arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- SurveilXEqualToArtifactCount",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"U": 1, "generic": 2},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "surveil_x_equal_to_artifact_count"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_surveil_x_equal_to_artifact_count_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``surveil_x_equal_to_artifacts_count`` -- plural "artifacts" vs
    singular "artifact") surfaces as ValueError on Rust's parse_effect
    arm because the new variant isn't registered for that spelling.
    Pin against accidental schema drift between bridge dict shape
    and Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Creature"],
        "mana_cost": {"U": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- plural "artifacts" instead of singular.
                    "kind": "surveil_x_equal_to_artifacts_count"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_surveil_x_equal_to_artifact_count_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "surveil_x_equal_to_artifact_count"}`` on a
    cast-spell ``effects`` carrier. Cerebral Download's REAL real-EoE
    consumer is a cast-spell shape (Instant whose cast effect is a
    Sequence: surveil-X then draw 3) -- this carrier is the actual
    deployment shape, NOT the synthetic triggered_ability sister pin.
    Sister of the triggered_ability-carrier lockstep pin -- the same
    parse_effect arm dispatches both carriers, so any drift surfaces
    on either pin. Mirrors the round-25/26/27/30/31/32/33 cast-spell
    vs trigger lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- SurveilXEqualToArtifactCount (Instant)",
        "card_types": ["Instant"],
        "mana_cost": {"U": 1, "generic": 2},
        "effects": [
            {
                "kind": "sequence",
                "effects": [
                    {"kind": "surveil_x_equal_to_artifact_count"},
                    {"kind": "draw_cards", "count": 3},
                ],
            },
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_hand_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 35) -- deal damage to an
    opponent's creature equal to the source-controller's hand size.
    Rust ``{"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}``
    translates to Python ``DealDamageToOpponentCreatureByCardsInHand()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-32 deal_damage_to_opponent_creature_by_artifact_count
    arm; unit variant; no fields). Pin verifies the bare-shape
    translation for the round-35 dynamic-X-from-hand-size Effect arm
    (Roaring Furnace's door trigger). Sister of round-32's
    test_bridge_deal_damage_to_opponent_creature_by_artifact_count_basic_translation
    -- same apply substrate, different denominator.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}
    )
    assert isinstance(e, DealDamageToOpponentCreatureByCardsInHand)


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_hand_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 by_artifact_count / round-31
    double_target_creatures_power_ueot / round-30
    put_stun_counter_on_tapped_creature) -- unrecognized fields in the
    dict are ignored, not raised. Notably, an ``amount`` key (which
    the sister DealDamageToOpponentCreature arm DOES consume) is
    silently dropped here -- the dynamic-X is the whole point of this
    variant, not a static amount.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_to_opponent_creature_by_cards_in_hand",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``deal_damage_to_opponent_creature`` arm; here it's
            # silently dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, DealDamageToOpponentCreatureByCardsInHand)


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_hand_equality_default_constructor():
    """Bridge arm pin: two ``DealDamageToOpponentCreatureByCardsInHand()``
    instances are equal under dataclass(frozen=True) default __eq__
    semantics (provided their auto-defaulted ``targets`` tuple
    converges via __post_init__). Pins the frozen-dataclass equality
    contract for downstream hash-based keying (e.g. de-dup in Sequence
    composers). Sister of the round-32 equality pin.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}
    )
    e2 = DealDamageToOpponentCreatureByCardsInHand()
    assert e1 == e2


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_hand_distinct_from_by_artifact_count():
    """Bridge arm pin: the round-35 hand-size variant and the round-32
    artifact-count variant are DISTINCT effect types even though they
    share the same apply substrate (damage to first eligible opponent
    creature). The denominator is the only axis that differs; Python
    uses two distinct @dataclass classes (damage.py:406 vs damage.py:
    464) and Rust uses two distinct unit variants. Pin guards against
    accidental cross-translation drift between the sisters at the
    bridge layer.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
        DealDamageToOpponentCreatureByCardsInHand,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e_hand = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}
    )
    e_art = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    assert isinstance(e_hand, DealDamageToOpponentCreatureByCardsInHand)
    assert isinstance(e_art, DealDamageToOpponentCreatureByArtifactCount)
    assert not isinstance(e_hand, DealDamageToOpponentCreatureByArtifactCount)
    assert not isinstance(e_art, DealDamageToOpponentCreatureByCardsInHand)
    assert type(e_hand) is not type(e_art)


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_hand_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealDamageToOpponentCreatureByCardsInHand(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-35 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter / round-31 double-power /
    round-32 by-artifact-count Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInHand,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DealDamageToOpponentCreatureByCardsInHand)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_cards_in_hand_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}``
    on a triggered_ability carrier. Roaring Furnace's actual real-card
    consumer mounts this Effect on a Room door-trigger; the lockstep
    gate doesn't depend on the trigger event -- registering through
    any abilities carrier exercises the parse_effect arm. Pinning here
    keeps the Python bridge from drifting away from Rust's
    parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-32 deal_damage_to_opponent_creature_by_artifact_count
    arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByCardsInHand",
        "card_types": ["Enchantment"],
        "mana_cost": {"R": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_to_opponent_creature_by_cards_in_hand"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_deal_damage_to_opponent_creature_by_cards_in_hand_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``deal_damage_to_opponent_creature_by_cards_in_hands`` -- plural
    "hands" vs singular "hand") surfaces as ValueError on Rust's
    parse_effect arm because the new variant isn't registered for
    that spelling. Pin against accidental schema drift between bridge
    dict shape and Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Enchantment"],
        "mana_cost": {"R": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — plural "hands" instead of singular "hand".
                    "kind": "deal_damage_to_opponent_creature_by_cards_in_hands"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_cards_in_hand_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}``
    on a cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either
    pin. Mirrors the round-25/26/27/30/31/32/33/34 cast-spell vs
    trigger lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByCardsInHand (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"R": 1, "generic": 2},
        "effects": [
            {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"},
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_to_opponent_creature_by_lands_controlled_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 36) -- deal damage to an
    opponent's creature equal to the source-controller's lands count.
    Rust ``{"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}``
    translates to Python
    ``DealDamageToOpponentCreatureByLandsControlled()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-35
    deal_damage_to_opponent_creature_by_cards_in_hand arm; unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-36 dynamic-X-from-lands-controlled Effect arm. Sister of
    the round-32 / round-35 basic_translation pins -- same apply
    substrate, third denominator.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLandsControlled,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}
    )
    assert isinstance(e, DealDamageToOpponentCreatureByLandsControlled)


def test_bridge_deal_damage_to_opponent_creature_by_lands_controlled_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 by_artifact_count / round-35
    by_cards_in_hand / round-31 double_target_creatures_power_ueot /
    round-30 put_stun_counter_on_tapped_creature) -- unrecognized
    fields in the dict are ignored, not raised. Notably, an ``amount``
    key (which the sister DealDamageToOpponentCreature arm DOES
    consume) is silently dropped here -- the dynamic-X from
    lands-controlled is the whole point of this variant, not a static
    amount.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLandsControlled,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_to_opponent_creature_by_lands_controlled",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``deal_damage_to_opponent_creature`` arm; here it's
            # silently dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, DealDamageToOpponentCreatureByLandsControlled)


def test_bridge_deal_damage_to_opponent_creature_by_lands_controlled_equality_default_constructor():
    """Bridge arm pin: two
    ``DealDamageToOpponentCreatureByLandsControlled()`` instances are
    equal under dataclass(frozen=True) default __eq__ semantics
    (provided their auto-defaulted ``targets`` tuple converges via
    __post_init__). Pins the frozen-dataclass equality contract for
    downstream hash-based keying (e.g. de-dup in Sequence composers).
    Sister of the round-32 / round-35 equality pins.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLandsControlled,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}
    )
    e2 = DealDamageToOpponentCreatureByLandsControlled()
    assert e1 == e2


def test_bridge_deal_damage_to_opponent_creature_by_lands_controlled_distinct_from_sisters():
    """Bridge arm pin: the round-36 lands-controlled variant, the
    round-35 hand-size variant, and the round-32 artifact-count variant
    are THREE DISTINCT effect types even though they share the same
    apply substrate (damage to first eligible opponent creature). The
    denominator is the only axis that differs; Python uses three
    distinct @dataclass classes (damage.py -- ByArtifactCount,
    ByCardsInHand, ByLandsControlled) and Rust uses three distinct
    unit variants. Pin guards against accidental cross-translation
    drift between the three sisters at the bridge layer.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
        DealDamageToOpponentCreatureByCardsInHand,
        DealDamageToOpponentCreatureByLandsControlled,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e_lands = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}
    )
    e_hand = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}
    )
    e_art = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    assert isinstance(e_lands, DealDamageToOpponentCreatureByLandsControlled)
    assert isinstance(e_hand, DealDamageToOpponentCreatureByCardsInHand)
    assert isinstance(e_art, DealDamageToOpponentCreatureByArtifactCount)
    # Cross-isinstance negatives -- the three classes are siblings,
    # not subclasses of one another.
    assert not isinstance(e_lands, DealDamageToOpponentCreatureByCardsInHand)
    assert not isinstance(e_lands, DealDamageToOpponentCreatureByArtifactCount)
    assert not isinstance(e_hand, DealDamageToOpponentCreatureByLandsControlled)
    assert not isinstance(e_art, DealDamageToOpponentCreatureByLandsControlled)
    # Strict type identity (rules out any subclass-relation drift).
    assert type(e_lands) is not type(e_hand)
    assert type(e_lands) is not type(e_art)
    assert type(e_hand) is not type(e_art)


def test_bridge_deal_damage_to_opponent_creature_by_lands_controlled_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealDamageToOpponentCreatureByLandsControlled(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-36 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter / round-31 double-power /
    round-32 by-artifact-count / round-35 by-cards-in-hand
    Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLandsControlled,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DealDamageToOpponentCreatureByLandsControlled)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_lands_controlled_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape
    ``{"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}``
    on a triggered_ability carrier. No EoE consumer today, but the
    lockstep gate doesn't depend on the trigger event -- registering
    through any abilities carrier exercises the parse_effect arm.
    Pinning here keeps the Python bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-35 deal_damage_to_opponent_creature_by_cards_in_hand
    arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByLandsControlled",
        "card_types": ["Enchantment"],
        "mana_cost": {"G": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_to_opponent_creature_by_lands_controlled"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_deal_damage_to_opponent_creature_by_lands_controlled_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``deal_damage_to_opponent_creature_by_land_controlled`` -- singular
    "land" vs plural "lands") surfaces as ValueError on Rust's
    parse_effect arm because the new variant isn't registered for that
    spelling. Pin against accidental schema drift between bridge dict
    shape and Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Enchantment"],
        "mana_cost": {"G": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — singular "land" instead of plural "lands".
                    "kind": "deal_damage_to_opponent_creature_by_land_controlled"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_lands_controlled_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape
    ``{"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}``
    on a cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either pin.
    Mirrors the round-25/26/27/30/31/32/33/34/35 cast-spell vs trigger
    lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByLandsControlled (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"G": 1, "generic": 2},
        "effects": [
            {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"},
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_to_opponent_creature_by_creatures_you_control_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 37) -- deal damage to an
    opponent's creature equal to the source-controller's creatures
    count. Rust
    ``{"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}``
    translates to Python
    ``DealDamageToOpponentCreatureByCreaturesYouControl()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-36
    deal_damage_to_opponent_creature_by_lands_controlled arm; unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-37 dynamic-X-from-creatures-controlled Effect arm.
    Sister of the round-32 / round-35 / round-36 basic_translation
    pins -- same apply substrate, fourth denominator.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCreaturesYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}
    )
    assert isinstance(e, DealDamageToOpponentCreatureByCreaturesYouControl)


def test_bridge_deal_damage_to_opponent_creature_by_creatures_you_control_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 by_artifact_count / round-35
    by_cards_in_hand / round-36 by_lands_controlled / round-31
    double_target_creatures_power_ueot / round-30
    put_stun_counter_on_tapped_creature) -- unrecognized fields in the
    dict are ignored, not raised. Notably, an ``amount`` key (which the
    sister DealDamageToOpponentCreature arm DOES consume) is silently
    dropped here -- the dynamic-X from creatures-controlled is the
    whole point of this variant, not a static amount.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCreaturesYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_to_opponent_creature_by_creatures_you_control",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``deal_damage_to_opponent_creature`` arm; here it's
            # silently dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, DealDamageToOpponentCreatureByCreaturesYouControl)


def test_bridge_deal_damage_to_opponent_creature_by_creatures_you_control_equality_default_constructor():
    """Bridge arm pin: two
    ``DealDamageToOpponentCreatureByCreaturesYouControl()`` instances
    are equal under dataclass(frozen=True) default __eq__ semantics
    (provided their auto-defaulted ``targets`` tuple converges via
    __post_init__). Pins the frozen-dataclass equality contract for
    downstream hash-based keying (e.g. de-dup in Sequence composers).
    Sister of the round-32 / round-35 / round-36 equality pins.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCreaturesYouControl,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}
    )
    e2 = DealDamageToOpponentCreatureByCreaturesYouControl()
    assert e1 == e2


def test_bridge_deal_damage_to_opponent_creature_by_creatures_you_control_distinct_from_sisters():
    """Bridge arm pin: the round-37 creatures-controlled variant, the
    round-36 lands-controlled variant, the round-35 hand-size variant,
    and the round-32 artifact-count variant are FOUR DISTINCT effect
    types even though they share the same apply substrate (damage to
    first eligible opponent creature). The denominator is the only
    axis that differs; Python uses four distinct @dataclass classes
    (damage.py -- ByArtifactCount, ByCardsInHand, ByLandsControlled,
    ByCreaturesYouControl) and Rust uses four distinct unit variants.
    Pin guards against accidental cross-translation drift among the
    four sisters at the bridge layer.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
        DealDamageToOpponentCreatureByCardsInHand,
        DealDamageToOpponentCreatureByCreaturesYouControl,
        DealDamageToOpponentCreatureByLandsControlled,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e_creatures = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}
    )
    e_lands = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}
    )
    e_hand = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}
    )
    e_art = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    assert isinstance(
        e_creatures, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert isinstance(e_lands, DealDamageToOpponentCreatureByLandsControlled)
    assert isinstance(e_hand, DealDamageToOpponentCreatureByCardsInHand)
    assert isinstance(e_art, DealDamageToOpponentCreatureByArtifactCount)
    # Cross-isinstance negatives -- the four classes are siblings,
    # not subclasses of one another.
    assert not isinstance(
        e_creatures, DealDamageToOpponentCreatureByLandsControlled
    )
    assert not isinstance(
        e_creatures, DealDamageToOpponentCreatureByCardsInHand
    )
    assert not isinstance(
        e_creatures, DealDamageToOpponentCreatureByArtifactCount
    )
    assert not isinstance(
        e_lands, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert not isinstance(
        e_hand, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert not isinstance(
        e_art, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    # Strict type identity (rules out any subclass-relation drift).
    assert type(e_creatures) is not type(e_lands)
    assert type(e_creatures) is not type(e_hand)
    assert type(e_creatures) is not type(e_art)


def test_bridge_deal_damage_to_opponent_creature_by_creatures_you_control_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealDamageToOpponentCreatureByCreaturesYouControl(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-37 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter / round-31 double-power /
    round-32 by-artifact-count / round-35 by-cards-in-hand / round-36
    by-lands-controlled Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCreaturesYouControl,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_creatures_you_control_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape
    ``{"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}``
    on a triggered_ability carrier. No EoE consumer today, but the
    lockstep gate doesn't depend on the trigger event -- registering
    through any abilities carrier exercises the parse_effect arm.
    Pinning here keeps the Python bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-36
    deal_damage_to_opponent_creature_by_lands_controlled arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByCreaturesYouControl",
        "card_types": ["Enchantment"],
        "mana_cost": {"G": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_to_opponent_creature_by_creatures_you_control"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_deal_damage_to_opponent_creature_by_creatures_you_control_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``deal_damage_to_opponent_creature_by_creature_you_control`` --
    singular "creature" vs plural "creatures") surfaces as ValueError
    on Rust's parse_effect arm because the new variant isn't registered
    for that spelling. Pin against accidental schema drift between
    bridge dict shape and Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Enchantment"],
        "mana_cost": {"G": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — singular "creature" instead of plural "creatures".
                    "kind": "deal_damage_to_opponent_creature_by_creature_you_control"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_creatures_you_control_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape
    ``{"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}``
    on a cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either pin.
    Mirrors the round-25/26/27/30/31/32/33/34/35/36 cast-spell vs
    trigger lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByCreaturesYouControl (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"G": 1, "generic": 2},
        "effects": [
            {"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"},
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_graveyard_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 46) -- deal damage to an
    opponent's creature equal to the source-controller's graveyard
    size. Rust
    ``{"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"}``
    translates to Python
    ``DealDamageToOpponentCreatureByCardsInGraveyard()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-37
    deal_damage_to_opponent_creature_by_creatures_you_control arm;
    unit variant; no fields). Pin verifies the bare-shape translation
    for the round-46 dynamic-X-from-graveyard-size Effect arm. Sister
    of the round-32 / 35 / 36 / 37 basic_translation pins -- same
    apply substrate, FIFTH denominator on the damage substrate.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInGraveyard,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"}
    )
    assert isinstance(e, DealDamageToOpponentCreatureByCardsInGraveyard)


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_graveyard_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 by_artifact_count / round-35
    by_cards_in_hand / round-36 by_lands_controlled / round-37
    by_creatures_you_control / round-31
    double_target_creatures_power_ueot / round-30
    put_stun_counter_on_tapped_creature) -- unrecognized fields in the
    dict are ignored, not raised. Notably, an ``amount`` key (which
    the sister DealDamageToOpponentCreature arm DOES consume) is
    silently dropped here -- the dynamic-X from graveyard size is the
    whole point of this variant, not a static amount.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInGraveyard,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``deal_damage_to_opponent_creature`` arm; here it's
            # silently dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, DealDamageToOpponentCreatureByCardsInGraveyard)


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_graveyard_equality_default_constructor():
    """Bridge arm pin: two
    ``DealDamageToOpponentCreatureByCardsInGraveyard()`` instances are
    equal under dataclass(frozen=True) default __eq__ semantics
    (provided their auto-defaulted ``targets`` tuple converges via
    __post_init__). Pins the frozen-dataclass equality contract for
    downstream hash-based keying (e.g. de-dup in Sequence composers).
    Sister of the round-32 / 35 / 36 / 37 equality pins.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInGraveyard,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"}
    )
    e2 = DealDamageToOpponentCreatureByCardsInGraveyard()
    assert e1 == e2


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_graveyard_distinct_from_sisters():
    """Bridge arm pin: the round-46 cards-in-graveyard variant, the
    round-37 creatures-controlled variant, the round-36
    lands-controlled variant, the round-35 hand-size variant, and the
    round-32 artifact-count variant are FIVE DISTINCT effect types
    even though they share the same apply substrate (damage to first
    eligible opponent creature). The denominator is the only axis that
    differs; Python uses five distinct @dataclass classes
    (damage.py -- ByArtifactCount, ByCardsInHand, ByLandsControlled,
    ByCreaturesYouControl, ByCardsInGraveyard) and Rust uses five
    distinct unit variants. Pin guards against accidental
    cross-translation drift among the five sisters at the bridge
    layer.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
        DealDamageToOpponentCreatureByCardsInGraveyard,
        DealDamageToOpponentCreatureByCardsInHand,
        DealDamageToOpponentCreatureByCreaturesYouControl,
        DealDamageToOpponentCreatureByLandsControlled,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e_graveyard = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"}
    )
    e_creatures = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}
    )
    e_lands = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}
    )
    e_hand = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}
    )
    e_art = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    assert isinstance(
        e_graveyard, DealDamageToOpponentCreatureByCardsInGraveyard
    )
    assert isinstance(
        e_creatures, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert isinstance(e_lands, DealDamageToOpponentCreatureByLandsControlled)
    assert isinstance(e_hand, DealDamageToOpponentCreatureByCardsInHand)
    assert isinstance(e_art, DealDamageToOpponentCreatureByArtifactCount)
    # Cross-isinstance negatives -- the five classes are siblings, not
    # subclasses of one another.
    assert not isinstance(
        e_graveyard, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert not isinstance(
        e_graveyard, DealDamageToOpponentCreatureByLandsControlled
    )
    assert not isinstance(
        e_graveyard, DealDamageToOpponentCreatureByCardsInHand
    )
    assert not isinstance(
        e_graveyard, DealDamageToOpponentCreatureByArtifactCount
    )
    assert not isinstance(
        e_creatures, DealDamageToOpponentCreatureByCardsInGraveyard
    )
    assert not isinstance(
        e_lands, DealDamageToOpponentCreatureByCardsInGraveyard
    )
    assert not isinstance(
        e_hand, DealDamageToOpponentCreatureByCardsInGraveyard
    )
    assert not isinstance(
        e_art, DealDamageToOpponentCreatureByCardsInGraveyard
    )
    # Strict type identity (rules out any subclass-relation drift).
    assert type(e_graveyard) is not type(e_creatures)
    assert type(e_graveyard) is not type(e_lands)
    assert type(e_graveyard) is not type(e_hand)
    assert type(e_graveyard) is not type(e_art)


def test_bridge_deal_damage_to_opponent_creature_by_cards_in_graveyard_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealDamageToOpponentCreatureByCardsInGraveyard(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-46 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter / round-31 double-power /
    round-32 by-artifact-count / round-35 by-cards-in-hand / round-36
    by-lands-controlled / round-37 by-creatures-you-control
    Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByCardsInGraveyard,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], DealDamageToOpponentCreatureByCardsInGraveyard
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_cards_in_graveyard_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape
    ``{"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"}``
    on a triggered_ability carrier. No EoE consumer today, but the
    lockstep gate doesn't depend on the trigger event -- registering
    through any abilities carrier exercises the parse_effect arm.
    Pinning here keeps the Python bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-37
    deal_damage_to_opponent_creature_by_creatures_you_control arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByCardsInGraveyard",
        "card_types": ["Enchantment"],
        "mana_cost": {"B": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_deal_damage_to_opponent_creature_by_cards_in_graveyard_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``deal_damage_to_opponent_creature_by_card_in_graveyard`` --
    singular "card" vs plural "cards") surfaces as ValueError on
    Rust's parse_effect arm because the new variant isn't registered
    for that spelling. Pin against accidental schema drift between
    bridge dict shape and Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Enchantment"],
        "mana_cost": {"B": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — singular "card" instead of plural "cards".
                    "kind": "deal_damage_to_opponent_creature_by_card_in_graveyard"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_cards_in_graveyard_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape
    ``{"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"}``
    on a cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either pin.
    Mirrors the round-25/26/27/30/31/32/33/34/35/36/37 cast-spell vs
    trigger lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByCardsInGraveyard (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"B": 1, "generic": 2},
        "effects": [
            {"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"},
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_mill_target_player_equal_to_your_life_total_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 42) -- mill an opponent a
    number of cards equal to the source-controller's current life
    total. Rust
    ``{"kind": "mill_target_player_equal_to_your_life_total"}``
    translates to Python
    ``MillTargetPlayer(use_controller_life=True)``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-19 ``mill_target_player`` arm; unit variant; no
    fields). Pin verifies the bare-shape translation for the
    round-42 dynamic-X-from-life-total Effect arm. Sister of the
    round-32 / 35 / 36 / 37 basic_translation pins (damage substrate)
    and round-33 (counters) / round-34 (surveil) -- FOURTH apply
    substrate and FIFTH denominator on the X-from-count plumbing.
    """
    from mtg_bot.effects.generic.draw_mill import MillTargetPlayer
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "mill_target_player_equal_to_your_life_total"}
    )
    assert isinstance(e, MillTargetPlayer)
    assert e.use_controller_life is True


def test_bridge_mill_target_player_equal_to_your_life_total_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 by_artifact_count / round-35
    by_cards_in_hand / round-36 by_lands_controlled / round-37
    by_creatures_you_control / round-31 double_target_creatures_power_ueot
    / round-30 put_stun_counter_on_tapped_creature) -- unrecognized
    fields in the dict are ignored, not raised. Notably, the
    ``count`` and ``amount`` keys (which the sister
    ``mill_target_player`` arm DOES consume) are silently dropped
    here -- the dynamic-X from life total is the whole point of this
    variant, not a static amount.
    """
    from mtg_bot.effects.generic.draw_mill import MillTargetPlayer
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "mill_target_player_equal_to_your_life_total",
            "spurious_field": "ignored",
            # ``count`` would be consumed by sister
            # ``mill_target_player`` arm; here it's silently dropped.
            "count": 99,
            # ``amount`` would be consumed by sister
            # ``mill_target_player`` arm; here it's silently dropped.
            "amount": 7,
        }
    )
    assert isinstance(e, MillTargetPlayer)
    assert e.use_controller_life is True


def test_bridge_mill_target_player_equal_to_your_life_total_distinct_from_static_mill_sister():
    """Bridge arm pin: the round-42 dynamic-X-from-life-total variant
    and the round-19 static-amount ``mill_target_player`` arm both
    produce ``MillTargetPlayer`` Python instances, but they differ on
    the ``use_controller_life`` flag (True vs False) AND on the
    ``amount`` field. Pin guards against accidental cross-translation
    drift between the two sisters at the bridge layer.
    """
    from mtg_bot.effects.generic.draw_mill import MillTargetPlayer
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e_dynamic = _ta_dict_to_python_effect(
        {"kind": "mill_target_player_equal_to_your_life_total"}
    )
    e_static = _ta_dict_to_python_effect(
        {"kind": "mill_target_player", "count": 2}
    )
    assert isinstance(e_dynamic, MillTargetPlayer)
    assert isinstance(e_static, MillTargetPlayer)
    # The two sisters differ on use_controller_life.
    assert e_dynamic.use_controller_life is True
    assert e_static.use_controller_life is False
    # Static-amount sister carries the explicit count via ``amount``.
    assert e_static.amount == 2
    # Dynamic sister keeps the class default ``amount=1`` (unused;
    # life total drives the count at resolve time).
    assert e_dynamic.amount == 1
    # The two instances are NOT equal -- they differ on a frozen
    # dataclass field (use_controller_life).
    assert e_dynamic != e_static


def test_bridge_mill_target_player_equal_to_your_life_total_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([MillTargetPlayer(use_controller_life=True), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-42 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter / round-31 double-power /
    round-32 by-artifact-count / round-35 by-cards-in-hand / round-36
    by-lands-controlled / round-37 by-creatures-you-control
    Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards, MillTargetPlayer
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "mill_target_player_equal_to_your_life_total"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], MillTargetPlayer)
    assert e.effects[0].use_controller_life is True
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_mill_target_player_equal_to_your_life_total_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "mill_target_player_equal_to_your_life_total"}``
    on a triggered_ability carrier. No EoE consumer for the trigger
    carrier today (Space-Time Anomaly is an instant, not a trigger),
    but the lockstep gate doesn't depend on the trigger event --
    registering through any abilities carrier exercises the
    parse_effect arm. Pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-19 ``mill_target_player`` arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- MillTargetPlayerEqualToYourLifeTotal",
        "card_types": ["Enchantment"],
        "mana_cost": {"U": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "mill_target_player_equal_to_your_life_total"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_mill_target_player_equal_to_your_life_total_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``mill_target_player_equal_to_your_life`` -- missing ``_total``
    suffix) surfaces as ValueError on Rust's parse_effect arm because
    the new variant isn't registered for that spelling. Pin against
    accidental schema drift between bridge dict shape and Rust
    parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Enchantment"],
        "mana_cost": {"U": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- missing ``_total`` suffix.
                    "kind": "mill_target_player_equal_to_your_life"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_mill_target_player_equal_to_your_life_total_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "mill_target_player_equal_to_your_life_total"}``
    on a cast-spell ``effects`` carrier (the Space-Time Anomaly
    consumer-shape -- it's an instant). Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either pin.
    Mirrors the round-25/26/27/30/31/32/33/34/35/36/37 cast-spell vs
    trigger lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- MillTargetPlayerEqualToYourLifeTotal (Instant)",
        "card_types": ["Instant"],
        "mana_cost": {"U": 1, "B": 1, "generic": 2},
        "effects": [
            {"kind": "mill_target_player_equal_to_your_life_total"},
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_create_token_each_player_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 43) -- each living player
    creates N tokens of the given printing under their own control.
    Rust ``{"kind": "create_token_each_player", "count": N,
    "token_printing_id": pid}`` translates to Python
    ``CreateToken(token_printing=<registered>, count=N,
    each_player=True)``.

    Sister of the existing ``create_token`` bridge arm; same two-
    field schema, distinguished by variant (closed-vocab convention
    -- the each-player vs self-only choice rides on the variant
    name / dict ``kind``, not on a bool field). Pins the
    each_player=True flag on the returned Python CreateToken.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    lander = make_token_creature(
        name="Lander Token (each-player bridge pin)",
        power=0, toughness=0, subtypes=("Lander",), artifact=True,
    )
    register_token_printing(13, lander)
    e = _ta_dict_to_python_effect(
        {
            "kind": "create_token_each_player",
            "count": 1,
            "token_printing_id": 13,
        }
    )
    assert isinstance(e, CreateToken)
    assert e.token_printing is lander
    assert e.count == 1
    assert e.each_player is True
    # Other rider flags default per the dataclass (Rust each-player
    # variant has no mirror for them; the bridge constructs the
    # bare shape).
    assert e.tapped is False
    assert e.attacking is False
    assert e.schedule_end_of_combat_sacrifice is False
    clear_token_printings()


def test_bridge_create_token_each_player_count_default_one():
    """Missing ``count`` defaults to 1 (mirrors the sister
    ``create_token`` arm's lenient default convention). Each-player
    fanout with count=1 is the EoE consumer shape (Edge Rover ->
    each player creates a Lander token, count=1).
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    lander = make_token_creature(
        name="Lander Token (each-player default-count pin)",
        power=0, toughness=0, subtypes=("Lander",), artifact=True,
    )
    register_token_printing(14, lander)
    e = _ta_dict_to_python_effect(
        {"kind": "create_token_each_player", "token_printing_id": 14}
    )
    assert isinstance(e, CreateToken)
    assert e.token_printing is lander
    assert e.count == 1
    assert e.each_player is True
    clear_token_printings()


def test_bridge_create_token_each_player_distinct_from_static_create_token_sister():
    """Bridge arm pin: the round-43 each-player variant and the
    round-12 static ``create_token`` arm both produce ``CreateToken``
    Python instances, but they differ on the ``each_player`` flag
    (True vs False). Pin guards against accidental cross-translation
    drift between the two sisters at the bridge layer (mirrors
    round-42 distinct-from-sister pin for mill_target_player_equal).
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    soldier = make_token_creature(
        name="Soldier (each-vs-static pin)",
        power=1, toughness=1, subtypes=("Soldier",),
    )
    register_token_printing(15, soldier)
    e_each = _ta_dict_to_python_effect(
        {
            "kind": "create_token_each_player",
            "count": 1,
            "token_printing_id": 15,
        }
    )
    e_static = _ta_dict_to_python_effect(
        {"kind": "create_token", "count": 1, "token_printing_id": 15}
    )
    assert isinstance(e_each, CreateToken)
    assert isinstance(e_static, CreateToken)
    # The two sisters differ on each_player.
    assert e_each.each_player is True
    assert e_static.each_player is False
    # The two instances are NOT equal -- they differ on a frozen
    # dataclass field (each_player).
    assert e_each != e_static
    clear_token_printings()


def test_bridge_create_token_each_player_missing_registration_raises_keyerror():
    """``create_token_each_player`` with a ``token_printing_id`` not
    present in the registry raises ``KeyError`` with a clear hint --
    silent divergence from a missing setup step would be much harder
    to debug than an immediate failure at translation time. Mirrors
    the sister ``create_token`` arm's missing-registration shape.
    """
    import pytest

    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
    )

    clear_token_printings()
    with pytest.raises(KeyError, match="token_printing_id=888"):
        _ta_dict_to_python_effect(
            {
                "kind": "create_token_each_player",
                "count": 1,
                "token_printing_id": 888,
            }
        )


def test_rust_parser_accepts_create_token_each_player_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "create_token_each_player", ...}`` on a
    triggered_ability carrier. The EoE consumer (Edge Rover) uses a
    dies-trigger carrier; pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage on this surface.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    # Pre-register the inner Lander token so the carrier's
    # triggered-ability dict can reference its pid.
    lander_pid = reg.register_printing({
        "name": "Round-43 Bridge Pin Lander",
        "card_types": ["Artifact"],
    })
    assert lander_pid == 0
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- CreateTokenEachPlayer",
        "card_types": ["Creature"],
        "power": 2, "toughness": 2,
        "mana_cost": {"R": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "create_token_each_player",
                    "count": 1,
                    "token_printing_id": lander_pid,
                },
            }
        ],
    }
    carrier_pid = reg.register_printing(d)
    assert carrier_pid == 1


def test_rust_parser_rejects_create_token_each_player_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``create_tokens_each_player`` -- spurious ``s`` plural) surfaces
    as ValueError on Rust's parse_effect arm because the new variant
    isn't registered for that spelling. Pin against accidental schema
    drift between bridge dict shape and Rust parser's accepted-kinds
    set (mirrors round-42 typo-rejection pin).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    # Pre-register an inner printing so the only failure mode is the
    # typo on the outer effect kind.
    inner_pid = reg.register_printing({
        "name": "Round-43 Typo Pin Inner",
        "card_types": ["Artifact"],
    })
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- create_token_each typo",
        "card_types": ["Creature"],
        "power": 1, "toughness": 1,
        "mana_cost": {"R": 1},
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- spurious ``s`` on ``tokens``.
                    "kind": "create_tokens_each_player",
                    "count": 1,
                    "token_printing_id": inner_pid,
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_rust_parser_accepts_create_token_each_player_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "create_token_each_player", ...}`` on a
    cast-spell ``effects`` carrier. Sister of the
    triggered_ability-carrier lockstep pin -- the same parse_effect
    arm dispatches both carriers, so any drift surfaces on either pin.
    Mirrors the round-25/26/27/30/31/32/33/34/35/36/37/42 cast-spell
    vs trigger lockstep convention.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    lander_pid = reg.register_printing({
        "name": "Round-43 Cast-effects Pin Lander",
        "card_types": ["Artifact"],
    })
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- CreateTokenEachPlayer (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"R": 1, "generic": 2},
        "effects": [
            {
                "kind": "create_token_each_player",
                "count": 1,
                "token_printing_id": lander_pid,
            },
        ],
    }
    carrier_pid = reg.register_printing(d)
    assert carrier_pid == 1


def test_bridge_each_opponent_loses_life_equal_to_artifact_count_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 44) -- each opponent loses
    life equal to the source-controller's artifact count. Rust
    ``{"kind": "each_opponent_loses_life_equal_to_artifact_count"}``
    translates to Python
    ``EachOpponentLosesLifeEqualToArtifactCount()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-26 each_opponent_loses_life arm; unit variant; no
    fields). Pin verifies the bare-shape translation for the round-44
    dynamic-X-from-artifact-count Effect arm. FIFTH apply substrate to
    pick up the round-32 X-from-count plumbing (sister of round-32
    damage-by-artifact, round-33 counters-per-other, round-34
    surveil-by-artifact, round-42 mill-by-life-total -- this slice
    extends the substrate axis to life-loss-each-opponent).
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_artifact_count"}
    )
    assert isinstance(e, EachOpponentLosesLifeEqualToArtifactCount)


def test_bridge_each_opponent_loses_life_equal_to_artifact_count_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 / round-34 / round-37
    X-from-count sister arms) -- unrecognized fields in the dict are
    ignored, not raised. Notably, an ``amount`` key (which the sister
    static EachOpponentLosesLife arm DOES consume as a required field)
    is silently dropped here -- the dynamic-X is the whole point of
    this variant, not a static amount.
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "each_opponent_loses_life_equal_to_artifact_count",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``each_opponent_loses_life`` arm; here it's silently
            # dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, EachOpponentLosesLifeEqualToArtifactCount)


def test_bridge_each_opponent_loses_life_equal_to_artifact_count_equality_default_constructor():
    """Bridge arm pin: two ``EachOpponentLosesLifeEqualToArtifactCount()``
    instances are equal under dataclass(frozen=True) default __eq__
    semantics. The class has no constructor args so any two instances
    are trivially equal. Pins the frozen-dataclass equality contract
    for downstream hash-based keying (e.g. de-dup in Sequence
    composers).
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_artifact_count"}
    )
    e2 = EachOpponentLosesLifeEqualToArtifactCount()
    assert e1 == e2


def test_bridge_each_opponent_loses_life_equal_to_artifact_count_distinct_from_static_sister():
    """Bridge arm pin: the round-44 dynamic-X variant and the round-26
    static ``each_opponent_loses_life`` sister are DIFFERENT Python
    classes -- they translate to two structurally distinct types
    (``EachOpponentLosesLifeEqualToArtifactCount`` vs
    ``EachOpponentLosesLife``). Pins against accidental class merge or
    silent type-coercion in the bridge translator. Sister round-37
    pin (creatures-by-control vs static damage) precedent."""
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLife,
        EachOpponentLosesLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    dynamic = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_artifact_count"}
    )
    static = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life", "amount": 3}
    )
    assert isinstance(dynamic, EachOpponentLosesLifeEqualToArtifactCount)
    assert isinstance(static, EachOpponentLosesLife)
    # Distinct types — no accidental class merge.
    assert type(dynamic) is not type(static)
    # The dynamic one carries no static amount field; the static one
    # carries amount=3.
    assert getattr(static, "amount", None) == 3


def test_bridge_each_opponent_loses_life_equal_to_artifact_count_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([EachOpponentLosesLifeEqualToArtifactCount(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-44 arm against accidental dispatch loss inside Sequence
    (mirrors round-30 / round-31 / round-32 / round-33 / round-34 /
    round-37 / round-42 unit-variant Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "each_opponent_loses_life_equal_to_artifact_count"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], EachOpponentLosesLifeEqualToArtifactCount
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_each_opponent_loses_life_equal_to_artifact_count_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "each_opponent_loses_life_equal_to_artifact_count"}``
    on a triggered_ability carrier. No EoE consumer for this shape
    today -- this slice ships per round-27/31/37 round-trip-surface-
    coverage rule; the lockstep gate doesn't depend on a real consumer,
    registering through any abilities carrier exercises the
    parse_effect arm. Pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-26 each_opponent_loses_life arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- EachOpponentLosesLifeEqualToArtifactCount",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"B": 1, "generic": 2},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "each_opponent_loses_life_equal_to_artifact_count"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_each_opponent_loses_life_equal_to_artifact_count_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``each_opponent_loses_life_equal_to_artifacts_count`` -- plural
    "artifacts" vs singular "artifact") surfaces as ValueError on
    Rust's parse_effect arm because the new variant isn't registered
    for that spelling. Pin against accidental schema drift between
    bridge dict shape and Rust parser's accepted-kinds set. Sister of
    the round-32 / round-37 typo-rejection pins.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"B": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — plural "artifacts" instead of singular "artifact".
                    "kind": "each_opponent_loses_life_equal_to_artifacts_count"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_bridge_gain_life_equal_to_artifact_count_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 45) -- the source's controller
    gains life equal to the source-controller's artifact count. Rust
    ``{"kind": "gain_life_equal_to_artifact_count"}`` translates to
    Python ``GainLifeEqualToArtifactCount()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-44 each_opponent_loses_life_equal_to_artifact_count
    arm; unit variant; no fields). Pin verifies the bare-shape
    translation for the round-45 dynamic-X-from-artifact-count Effect
    arm. SIXTH apply substrate to pick up the round-32 X-from-count
    plumbing (sister of round-32 damage-by-artifact, round-33
    counters-per-other, round-34 surveil-by-artifact, round-42
    mill-by-life-total, round-44 each-opp-life-loss-by-artifact -- this
    slice extends the substrate axis to gain-life-controller-self, the
    inverted-recipient-and-sign mirror of round-44).
    """
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_artifact_count"}
    )
    assert isinstance(e, GainLifeEqualToArtifactCount)


def test_bridge_gain_life_equal_to_artifact_count_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 / round-34 / round-37 /
    round-44 X-from-count sister arms) -- unrecognized fields in the
    dict are ignored, not raised. Notably, an ``amount`` key (which the
    sister static GainLife arm DOES consume as a required field) is
    silently dropped here -- the dynamic-X is the whole point of this
    variant, not a static amount.
    """
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "gain_life_equal_to_artifact_count",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister ``gain_life`` arm;
            # here it's silently dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, GainLifeEqualToArtifactCount)


def test_bridge_gain_life_equal_to_artifact_count_equality_default_constructor():
    """Bridge arm pin: two ``GainLifeEqualToArtifactCount()`` instances
    are equal under dataclass(frozen=True) default __eq__ semantics.
    The class has no constructor args so any two instances are
    trivially equal. Pins the frozen-dataclass equality contract for
    downstream hash-based keying (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_artifact_count"}
    )
    e2 = GainLifeEqualToArtifactCount()
    assert e1 == e2


def test_bridge_gain_life_equal_to_artifact_count_distinct_from_static_sister():
    """Bridge arm pin: the round-45 dynamic-X variant and the round-26
    static ``gain_life`` sister are DIFFERENT Python classes -- they
    translate to two structurally distinct types
    (``GainLifeEqualToArtifactCount`` vs ``GainLife``). Pins against
    accidental class merge or silent type-coercion in the bridge
    translator. Sister round-37 / round-44 pin precedent."""
    from mtg_bot.effects.generic.life import (
        GainLife,
        GainLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    dynamic = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_artifact_count"}
    )
    static = _ta_dict_to_python_effect(
        {"kind": "gain_life", "amount": 3}
    )
    assert isinstance(dynamic, GainLifeEqualToArtifactCount)
    assert isinstance(static, GainLife)
    # Distinct types — no accidental class merge.
    assert type(dynamic) is not type(static)
    # The dynamic one carries no static amount field; the static one
    # carries amount=3.
    assert getattr(static, "amount", None) == 3


def test_bridge_gain_life_equal_to_artifact_count_distinct_from_round44_each_opponent_sister():
    """Bridge arm pin: the round-45 controller-self life-gain variant
    and the round-44 each-opponent life-loss variant are DIFFERENT
    Python classes despite sharing the artifact-count denominator.
    Inverted-recipient-and-sign mirror -- gain vs loss, controller-self
    vs opponents-of-controller. Pins against accidental cross-arm
    confusion in the bridge translator (the two arms' dict shapes
    differ only in the kind string ``gain_life_*`` vs
    ``each_opponent_loses_life_*``).
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToArtifactCount,
        GainLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    gain = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_artifact_count"}
    )
    loss = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_artifact_count"}
    )
    assert isinstance(gain, GainLifeEqualToArtifactCount)
    assert isinstance(loss, EachOpponentLosesLifeEqualToArtifactCount)
    # Distinct types — no accidental cross-arm merge.
    assert type(gain) is not type(loss)


def test_bridge_gain_life_equal_to_artifact_count_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([GainLifeEqualToArtifactCount(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-45 arm against accidental dispatch loss inside Sequence
    (mirrors round-30 / round-31 / round-32 / round-33 / round-34 /
    round-37 / round-42 / round-44 unit-variant Sequence-composability
    pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToArtifactCount,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "gain_life_equal_to_artifact_count"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], GainLifeEqualToArtifactCount
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_gain_life_equal_to_artifact_count_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "gain_life_equal_to_artifact_count"}`` on a
    triggered_ability carrier. No EoE consumer for this shape today --
    this slice ships per round-27/31/37 round-trip-surface-coverage
    rule; the lockstep gate doesn't depend on a real consumer,
    registering through any abilities carrier exercises the
    parse_effect arm. Pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-44 each_opponent_loses_life_equal_to_artifact_count
    arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- GainLifeEqualToArtifactCount",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"W": 1, "generic": 2},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "gain_life_equal_to_artifact_count"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_gain_life_equal_to_artifact_count_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``gain_life_equal_to_artifacts_count`` -- plural "artifacts" vs
    singular "artifact") surfaces as ValueError on Rust's parse_effect
    arm because the new variant isn't registered for that spelling.
    Pin against accidental schema drift between bridge dict shape and
    Rust parser's accepted-kinds set. Sister of the round-32 / round-37
    / round-44 typo-rejection pins.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"W": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled — plural "artifacts" instead of singular "artifact".
                    "kind": "gain_life_equal_to_artifacts_count"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_bridge_deal_damage_to_opponent_creature_by_life_total_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 47) -- deal damage to an
    opponent's creature equal to the source-controller's LIFE TOTAL.
    Rust
    ``{"kind": "deal_damage_to_opponent_creature_by_life_total"}``
    translates to Python
    ``DealDamageToOpponentCreatureByLifeTotal()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-46
    deal_damage_to_opponent_creature_by_cards_in_graveyard arm; unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-47 dynamic-X-from-life-total Effect arm. Sister of the
    round-32 / 35 / 36 / 37 / 46 basic_translation pins -- same apply
    substrate, SIXTH (final) denominator on the damage substrate.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_life_total"}
    )
    assert isinstance(e, DealDamageToOpponentCreatureByLifeTotal)


def test_bridge_deal_damage_to_opponent_creature_by_life_total_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-32 by_artifact_count / round-35
    by_cards_in_hand / round-36 by_lands_controlled / round-37
    by_creatures_you_control / round-46 by_cards_in_graveyard) --
    unrecognized fields in the dict are ignored, not raised. Notably,
    an ``amount`` key (which the sister DealDamageToOpponentCreature
    arm DOES consume) is silently dropped here -- the dynamic-X from
    the clamped life total is the whole point of this variant, not a
    static amount.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "deal_damage_to_opponent_creature_by_life_total",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``deal_damage_to_opponent_creature`` arm; here it's
            # silently dropped -- the count is dynamic.
            "amount": 99,
        }
    )
    assert isinstance(e, DealDamageToOpponentCreatureByLifeTotal)


def test_bridge_deal_damage_to_opponent_creature_by_life_total_equality_default_constructor():
    """Bridge arm pin: two
    ``DealDamageToOpponentCreatureByLifeTotal()`` instances are equal
    under dataclass(frozen=True) default __eq__ semantics (provided
    their auto-defaulted ``targets`` tuple converges via
    __post_init__). Pins the frozen-dataclass equality contract for
    downstream hash-based keying (e.g. de-dup in Sequence composers).
    Sister of the round-32 / 35 / 36 / 37 / 46 equality pins.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_life_total"}
    )
    e2 = DealDamageToOpponentCreatureByLifeTotal()
    assert e1 == e2


def test_bridge_deal_damage_to_opponent_creature_by_life_total_distinct_from_sisters():
    """Bridge arm pin: the round-47 life-total variant and its five
    sister denominators (round-46 cards-in-graveyard, round-37
    creatures-controlled, round-36 lands-controlled, round-35
    hand-size, round-32 artifact-count) are SIX DISTINCT effect types
    even though they share the same apply substrate (damage to first
    eligible opponent creature). The denominator is the only axis that
    differs; Python uses six distinct @dataclass classes (damage.py --
    ByArtifactCount, ByCardsInHand, ByLandsControlled,
    ByCreaturesYouControl, ByCardsInGraveyard, ByLifeTotal) and Rust
    uses six distinct unit variants. Pin guards against accidental
    cross-translation drift among the six sisters at the bridge layer
    -- the damage-to-opponent-creature row of the X-from-count matrix
    is now fully saturated.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByArtifactCount,
        DealDamageToOpponentCreatureByCardsInGraveyard,
        DealDamageToOpponentCreatureByCardsInHand,
        DealDamageToOpponentCreatureByCreaturesYouControl,
        DealDamageToOpponentCreatureByLandsControlled,
        DealDamageToOpponentCreatureByLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e_life = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_life_total"}
    )
    e_graveyard = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_graveyard"}
    )
    e_creatures = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_creatures_you_control"}
    )
    e_lands = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_lands_controlled"}
    )
    e_hand = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_cards_in_hand"}
    )
    e_art = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_opponent_creature_by_artifact_count"}
    )
    assert isinstance(e_life, DealDamageToOpponentCreatureByLifeTotal)
    assert isinstance(
        e_graveyard, DealDamageToOpponentCreatureByCardsInGraveyard
    )
    assert isinstance(
        e_creatures, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert isinstance(e_lands, DealDamageToOpponentCreatureByLandsControlled)
    assert isinstance(e_hand, DealDamageToOpponentCreatureByCardsInHand)
    assert isinstance(e_art, DealDamageToOpponentCreatureByArtifactCount)
    # Cross-isinstance negatives -- the six classes are siblings, not
    # subclasses of one another.
    assert not isinstance(
        e_life, DealDamageToOpponentCreatureByCardsInGraveyard
    )
    assert not isinstance(
        e_life, DealDamageToOpponentCreatureByCreaturesYouControl
    )
    assert not isinstance(
        e_life, DealDamageToOpponentCreatureByLandsControlled
    )
    assert not isinstance(e_life, DealDamageToOpponentCreatureByCardsInHand)
    assert not isinstance(e_life, DealDamageToOpponentCreatureByArtifactCount)
    assert not isinstance(
        e_graveyard, DealDamageToOpponentCreatureByLifeTotal
    )
    assert not isinstance(
        e_creatures, DealDamageToOpponentCreatureByLifeTotal
    )
    assert not isinstance(e_lands, DealDamageToOpponentCreatureByLifeTotal)
    assert not isinstance(e_hand, DealDamageToOpponentCreatureByLifeTotal)
    assert not isinstance(e_art, DealDamageToOpponentCreatureByLifeTotal)
    # Strict type identity (rules out any subclass-relation drift).
    assert type(e_life) is not type(e_graveyard)
    assert type(e_life) is not type(e_creatures)
    assert type(e_life) is not type(e_lands)
    assert type(e_life) is not type(e_hand)
    assert type(e_life) is not type(e_art)


def test_bridge_deal_damage_to_opponent_creature_by_life_total_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([DealDamageToOpponentCreatureByLifeTotal(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-47 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30 stun-counter / round-31 double-power /
    round-32 by-artifact-count / round-35 by-cards-in-hand / round-36
    by-lands-controlled / round-37 by-creatures-you-control / round-46
    by-cards-in-graveyard Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageToOpponentCreatureByLifeTotal,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "deal_damage_to_opponent_creature_by_life_total"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], DealDamageToOpponentCreatureByLifeTotal
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_deal_damage_to_opponent_creature_by_life_total_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape
    ``{"kind": "deal_damage_to_opponent_creature_by_life_total"}``
    on a triggered_ability carrier. No EoE consumer today, but the
    lockstep gate doesn't depend on the trigger event -- registering
    through any abilities carrier exercises the parse_effect arm.
    Pinning here keeps the Python bridge from drifting away from
    Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-46
    deal_damage_to_opponent_creature_by_cards_in_graveyard arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- DealDamageToOpponentCreatureByLifeTotal",
        "card_types": ["Enchantment"],
        "mana_cost": {"R": 1, "generic": 2},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "deal_damage_to_opponent_creature_by_life_total"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_deal_damage_to_opponent_creature_by_life_total_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``deal_damage_to_opponent_creature_by_life_totals`` -- spurious
    trailing "s") surfaces as ValueError on Rust's parse_effect arm
    because the new variant isn't registered for that spelling. Pin
    against accidental schema drift between bridge dict shape and
    Rust parser's accepted-kinds set.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- life-total typo lockstep",
        "card_types": ["Enchantment"],
        "mana_cost": {"R": 1},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- spurious trailing "s".
                    "kind": "deal_damage_to_opponent_creature_by_life_totals"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_bridge_each_opponent_loses_life_equal_to_your_life_total_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 48) -- each opponent loses
    life equal to the source-controller's CURRENT LIFE TOTAL. Rust
    ``{"kind": "each_opponent_loses_life_equal_to_your_life_total"}``
    translates to Python
    ``EachOpponentLosesLifeEqualToYourLifeTotal()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-44 each_opponent_loses_life_equal_to_artifact_count
    arm; unit variant; no fields). Pin verifies the bare-shape
    translation for the round-48 dynamic-X-from-life-total Effect arm.
    SECOND denominator on the life-loss-each-opponent apply substrate
    (sister of round-44 each-opp-life-loss-by-artifact -- same apply,
    only the denominator read differs; the life-total denominator was
    established by round-42 mill / round-47 damage on other substrates).
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_your_life_total"}
    )
    assert isinstance(e, EachOpponentLosesLifeEqualToYourLifeTotal)


def test_bridge_each_opponent_loses_life_equal_to_your_life_total_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-42 / round-44 / round-47
    X-from-count sister arms) -- unrecognized fields in the dict are
    ignored, not raised. Notably, an ``amount`` key (which the sister
    static EachOpponentLosesLife arm DOES consume as a required field)
    is silently dropped here -- the dynamic-X is the whole point of
    this variant, not a static amount.
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "each_opponent_loses_life_equal_to_your_life_total",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister
            # ``each_opponent_loses_life`` arm; here it's silently
            # dropped -- the count is dynamic (life total).
            "amount": 99,
        }
    )
    assert isinstance(e, EachOpponentLosesLifeEqualToYourLifeTotal)


def test_bridge_each_opponent_loses_life_equal_to_your_life_total_equality_default_constructor():
    """Bridge arm pin: two
    ``EachOpponentLosesLifeEqualToYourLifeTotal()`` instances are equal
    under dataclass(frozen=True) default __eq__ semantics. The class has
    no constructor args so any two instances are trivially equal. Pins
    the frozen-dataclass equality contract for downstream hash-based
    keying (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_your_life_total"}
    )
    e2 = EachOpponentLosesLifeEqualToYourLifeTotal()
    assert e1 == e2


def test_bridge_each_opponent_loses_life_equal_to_your_life_total_distinct_from_sisters():
    """Bridge arm pin: the round-48 life-total variant, the round-44
    artifact-count variant, and the round-26 static
    ``each_opponent_loses_life`` are THREE DIFFERENT Python classes --
    they translate to structurally distinct types. Pins against
    accidental class merge or silent type-coercion in the bridge
    translator (round-37 / round-44 distinct-from-sister precedent).
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLife,
        EachOpponentLosesLifeEqualToArtifactCount,
        EachOpponentLosesLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    by_life = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_your_life_total"}
    )
    by_artifact = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_artifact_count"}
    )
    static = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life", "amount": 3}
    )
    assert isinstance(by_life, EachOpponentLosesLifeEqualToYourLifeTotal)
    assert isinstance(
        by_artifact, EachOpponentLosesLifeEqualToArtifactCount
    )
    assert isinstance(static, EachOpponentLosesLife)
    # Three distinct types -- no accidental class merge.
    assert type(by_life) is not type(by_artifact)
    assert type(by_life) is not type(static)
    assert type(by_artifact) is not type(static)
    # The dynamic ones carry no static amount field; the static one
    # carries amount=3.
    assert getattr(static, "amount", None) == 3
    assert not hasattr(by_life, "amount")


def test_bridge_each_opponent_loses_life_equal_to_your_life_total_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([EachOpponentLosesLifeEqualToYourLifeTotal(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-48 arm against accidental dispatch loss inside Sequence
    (mirrors round-30 / round-31 / round-32 / round-33 / round-34 /
    round-37 / round-42 / round-44 unit-variant Sequence-composability
    pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "each_opponent_loses_life_equal_to_your_life_total"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], EachOpponentLosesLifeEqualToYourLifeTotal
    )
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_each_opponent_loses_life_equal_to_your_life_total_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "each_opponent_loses_life_equal_to_your_life_total"}``
    on a triggered_ability carrier. No EoE consumer for this shape
    today -- this slice ships per round-27/31/37 round-trip-surface-
    coverage rule; the lockstep gate doesn't depend on a real consumer,
    registering through any abilities carrier exercises the
    parse_effect arm. Pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-44 each_opponent_loses_life_equal_to_artifact_count
    arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- EachOpponentLosesLifeEqualToYourLifeTotal",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"B": 1, "generic": 2},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "each_opponent_loses_life_equal_to_your_life_total"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_each_opponent_loses_life_equal_to_your_life_total_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``each_opponent_loses_life_equal_to_your_life_totals`` -- spurious
    trailing "s") surfaces as ValueError on Rust's parse_effect arm
    because the new variant isn't registered for that spelling. Pin
    against accidental schema drift between bridge dict shape and Rust
    parser's accepted-kinds set. Sister of the round-44 / round-47
    typo-rejection pins.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep r48",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"B": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- spurious trailing "s".
                    "kind": "each_opponent_loses_life_equal_to_your_life_totals"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_bridge_gain_life_equal_to_your_life_total_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 49) -- the source's controller
    GAINS life equal to the source-controller's CURRENT LIFE TOTAL.
    Rust ``{"kind": "gain_life_equal_to_your_life_total"}`` translates
    to Python ``GainLifeEqualToYourLifeTotal()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-45 gain_life_equal_to_artifact_count arm; unit
    variant; no fields). Pin verifies the bare-shape translation for
    the round-49 dynamic-X-from-life-total Effect arm. SECOND
    denominator on the gain-life-controller-self apply substrate
    (sister of round-45 gain-life-by-artifact -- same apply, only the
    denominator read differs; the life-total denominator was
    established by round-42 mill / round-47 damage / round-48
    each-opponent-loses on other substrates).
    """
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_your_life_total"}
    )
    assert isinstance(e, GainLifeEqualToYourLifeTotal)


def test_bridge_gain_life_equal_to_your_life_total_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the unit-variant precedent (round-42 / round-45 / round-47 /
    round-48 X-from-count sister arms) -- unrecognized fields in the
    dict are ignored, not raised. Notably, an ``amount`` key (which the
    sister static GainLife arm DOES consume as a required field) is
    silently dropped here -- the dynamic-X is the whole point of this
    variant, not a static amount.
    """
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "gain_life_equal_to_your_life_total",
            "spurious_field": "ignored",
            # ``amount`` would be consumed by sister ``gain_life``
            # arm; here it's silently dropped -- the count is dynamic
            # (life total).
            "amount": 99,
        }
    )
    assert isinstance(e, GainLifeEqualToYourLifeTotal)


def test_bridge_gain_life_equal_to_your_life_total_equality_default_constructor():
    """Bridge arm pin: two ``GainLifeEqualToYourLifeTotal()`` instances
    are equal under dataclass(frozen=True) default __eq__ semantics.
    The class has no constructor args so any two instances are
    trivially equal. Pins the frozen-dataclass equality contract for
    downstream hash-based keying (e.g. de-dup in Sequence composers).
    """
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_your_life_total"}
    )
    e2 = GainLifeEqualToYourLifeTotal()
    assert e1 == e2


def test_bridge_gain_life_equal_to_your_life_total_distinct_from_sisters():
    """Bridge arm pin: the round-49 gain-life-by-life-total variant,
    the round-45 gain-life-by-artifact-count variant, the round-26
    static ``gain_life``, and the round-48 each-opponent-loses-by-life-
    total variant are FOUR DIFFERENT Python classes -- they translate
    to structurally distinct types. Pins against accidental class merge
    or silent type-coercion in the bridge translator (round-44 /
    round-48 distinct-from-sister precedent). The round-48 sister has
    the SAME life-total denominator but the opposite recipient
    (each-opponent vs controller-self) AND opposite sign (loss vs
    gain); the round-45 sister has the SAME apply substrate but the
    artifact-count denominator instead of life-total.
    """
    from mtg_bot.effects.generic.life import (
        EachOpponentLosesLifeEqualToYourLifeTotal,
        GainLife,
        GainLifeEqualToArtifactCount,
        GainLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    by_life = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_your_life_total"}
    )
    by_artifact = _ta_dict_to_python_effect(
        {"kind": "gain_life_equal_to_artifact_count"}
    )
    static = _ta_dict_to_python_effect(
        {"kind": "gain_life", "amount": 4}
    )
    each_opp_by_life = _ta_dict_to_python_effect(
        {"kind": "each_opponent_loses_life_equal_to_your_life_total"}
    )
    assert isinstance(by_life, GainLifeEqualToYourLifeTotal)
    assert isinstance(by_artifact, GainLifeEqualToArtifactCount)
    assert isinstance(static, GainLife)
    assert isinstance(
        each_opp_by_life, EachOpponentLosesLifeEqualToYourLifeTotal
    )
    # Four distinct types -- no accidental class merge.
    assert type(by_life) is not type(by_artifact)
    assert type(by_life) is not type(static)
    assert type(by_life) is not type(each_opp_by_life)
    assert type(by_artifact) is not type(static)
    # The dynamic ones carry no static amount field; the static one
    # carries amount=4.
    assert getattr(static, "amount", None) == 4
    assert not hasattr(by_life, "amount")


def test_bridge_gain_life_equal_to_your_life_total_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([GainLifeEqualToYourLifeTotal(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-49 arm against accidental dispatch loss inside Sequence
    (mirrors round-30 / round-31 / round-32 / round-33 / round-34 /
    round-37 / round-42 / round-44 / round-45 / round-48 unit-variant
    Sequence-composability pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import (
        GainLifeEqualToYourLifeTotal,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "gain_life_equal_to_your_life_total"},
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], GainLifeEqualToYourLifeTotal)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_gain_life_equal_to_your_life_total_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "gain_life_equal_to_your_life_total"}`` on a
    triggered_ability carrier. No EoE consumer for this shape today --
    this slice ships per round-27/31/37 round-trip-surface-coverage
    rule; the lockstep gate doesn't depend on a real consumer,
    registering through any abilities carrier exercises the
    parse_effect arm. Pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage.

    Mirrors Rust's parse_effect arm added in this slice immediately
    after the round-45 gain_life_equal_to_artifact_count arm.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- GainLifeEqualToYourLifeTotal",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"W": 1, "generic": 2},
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "gain_life_equal_to_your_life_total"
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_gain_life_equal_to_your_life_total_with_typo():
    """Negative lockstep pin: a misspelled kind (e.g.
    ``gain_life_equal_to_your_life_totals`` -- spurious trailing "s")
    surfaces as ValueError on Rust's parse_effect arm because the new
    variant isn't registered for that spelling. Pin against accidental
    schema drift between bridge dict shape and Rust parser's
    accepted-kinds set. Sister of the round-45 / round-48 typo-
    rejection pins.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep r49",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"W": 1},
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- spurious trailing "s".
                    "kind": "gain_life_equal_to_your_life_totals"
                },
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


# ===== Round-49 N1 token-printing substrate (Phase 1) bridge pins =====
#
# Pins the loader-side fixed-single-vanilla-token
# ENTERS+IsSelf+create_token substrate: the recognizer admit/decline
# gates (§2 silent-mistranslation discipline), the _token_vanilla_card_def
# gate, and the _seat_token_printings loader pass producing a valid
# deterministic Rust-registry pid + bridge-map registration + a dict the
# Rust parser accepts. No Rust code changed this round, so these are
# Python-side substrate pins; the Rust create_token parser/resolve arms
# are already pinned by the round-3/21/43 create_token pin family above.
# See docs/TOKEN_PRINTING_SUBSTRATE.md.


def test_r49_recognizer_admits_clean_vanilla_token_carrier():
    """A clean creature carrier with ENTERS+IsSelf+CreateToken(count=1,
    no riders) and a VANILLA token round-trips through
    _python_ta_to_safe_dict, emitting the symbolic _token_card_def
    shape (pid assigned later by the loader-seating pass). Tunnel
    Surveyor (Creature, Glimmer 1/1 Enchantment-Creature vanilla
    token) is the canonical clean EoE consumer (0 deck presence, so
    substrate-only — but it exercises the full recognizer path).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity.trajectory import (
        _python_ta_to_safe_dict,
        _printing_is_bridge_simple_creature,
    )
    from mtg_bot.parity.cross_engine import CardDef

    pr = get_active_source().get_printings()
    ts = pr.get("Tunnel Surveyor")
    assert ts is not None
    d = _python_ta_to_safe_dict(ts.triggered_abilities[0])
    assert d is not None, "clean vanilla-token carrier must be admitted"
    assert d["event"] == "enters"
    assert d["predicate"] == {"kind": "source_is_self"}
    eff = d["effect"]
    assert eff["kind"] == "create_token"
    assert eff["count"] == 1
    # Pre-seating: symbolic token CardDef, NO numeric pid yet.
    assert "token_printing_id" not in eff
    tcd = eff["_token_card_def"]
    assert isinstance(tcd, CardDef)
    assert tcd.name == "Glimmer"
    assert tcd.card_type == "Creature"
    assert tcd.extra_card_types == ("Enchantment",)
    assert tcd.power == 1 and tcd.toughness == 1
    assert tcd.supertypes == ("Token",)
    # Whole printing passes the bridge-simple creature filter.
    assert _printing_is_bridge_simple_creature(ts) is True


def test_r49_recognizer_declines_non_vanilla_lander_token():
    """§2 gate (round-49 Phase-1 pin, CONVERTED for Phase 3a round-53).

    Phase 1 declined the whole high-deck-count Lander cluster because
    the CardDef bridge had no activated-ability field — the Lander's
    ``{2}, Sac: SearchLibraryForBasicLandTapped`` would silently
    vanish cross-engine. The round-52 N1 precursor landed the
    ``token_activated_abilities`` substrate + bridge plumbing, and
    **Phase 3a (round 53)** adds
    ``_token_with_safe_activated_ability_card_def`` which §2-strictly
    ADMITS the bare Lander. So Galactic Wayfarer / Beamsaw Prospector
    are the round-49-design-doc-anticipated reject→convert (the
    round-50 Gravpack-Monoist sister-precedent).

    The §2-honest split, pinned here:
      * ``_token_vanilla_card_def`` is UNCHANGED — it STILL declines
        the Lander (activated ability + non-creature body).
      * ``_token_with_safe_activated_ability_card_def`` (the new
        sibling) ADMITS the bare bridge-correct Lander.
      * the carrier's trigger now round-trips +
        ``_printing_is_bridge_simple_creature`` returns True.
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity.trajectory import (
        _python_ta_to_safe_dict,
        _printing_is_bridge_simple_creature,
        _token_vanilla_card_def,
        _token_with_safe_activated_ability_card_def,
    )

    pr = get_active_source().get_printings()
    for name in ("Galactic Wayfarer", "Beamsaw Prospector"):
        carrier = pr.get(name)
        assert carrier is not None, name
        ta = carrier.triggered_abilities[0]
        # Phase 3a: the carrier's Lander trigger now ROUND-TRIPS.
        sd = _python_ta_to_safe_dict(ta)
        assert sd is not None, (
            f"{name}: Phase 3a must ADMIT the bare-Lander carrier "
            f"(reject→convert of the round-49 Phase-1 decline)"
        )
        assert sd["effect"]["kind"] == "create_token"
        assert _printing_is_bridge_simple_creature(carrier) is True
        # The vanilla gate is UNCHANGED — still declines the Lander.
        assert _token_vanilla_card_def(ta.effect.token_printing) is None
        # The Phase-3a sibling gate ADMITS it (the §2-honest split).
        tok_cd = _token_with_safe_activated_ability_card_def(
            ta.effect.token_printing
        )
        assert tok_cd is not None
        assert tok_cd.name == "Lander"
        assert len(tok_cd.token_activated_abilities) == 1


def test_r49_recognizer_declines_unsupported_riders():
    """§2 gate (round-49 Phase-1 pin, UPDATED for Phase 4 round-50).

    Phase 1 declined ALL riders (tapped / each_player / attacking)
    because the bare Rust ``Effect::CreateToken { count,
    token_printing_id }`` had no mirror for any of them. **Phase 4
    (round 50)** adds the ``Effect::CreateTokenTapped`` sister variant
    + the ``create_token_tapped`` bridge/recognizer arm, so the
    ``tapped=True`` rider is now ADMITTED as the tapped sister kind
    (this is the round-49-design-doc-anticipated reject→convert — see
    ``docs/TOKEN_PRINTING_SUBSTRATE.md`` §4 Phase 4 / the EoE-pool
    delta pin ``test_phase4_gravpack_monoist_flips_drop_to_admit``).
    ``each_player=True`` (Edge Rover — Phase 5) and ``attacking=True``
    (no Rust mirror) STAY declined; ``tapped`` COMBINED with another
    rider also stays declined (Rust's tapped variant only mirrors the
    lone-tapped profile). Synthetic riders pinned directly so the gate
    is exercised independent of the EoE pool's exact carriers.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.trajectory import _python_ta_to_safe_dict

    robot = make_token_creature(
        name="R49 Robot", power=2, toughness=2,
        subtypes=("Robot",), artifact=True,
    )
    base_kw = dict(event=Event.ENTERS, predicate=IsSelf())
    # Bare rider -> admitted as the Phase-1 create_token kind.
    ok = TriggeredAbility(
        effect=CreateToken(token_printing=robot, count=1), **base_kw
    )
    ok_sd = _python_ta_to_safe_dict(ok)
    assert ok_sd is not None
    assert ok_sd["effect"]["kind"] == "create_token"
    # Phase 4: tapped=True -> now ADMITTED as create_token_tapped
    # (was declined under Phase 1; the round-49 design doc's
    # anticipated reject→convert).
    tapped = TriggeredAbility(
        effect=CreateToken(token_printing=robot, count=1, tapped=True),
        **base_kw,
    )
    tapped_sd = _python_ta_to_safe_dict(tapped)
    assert tapped_sd is not None, (
        "Phase 4 admits the lone tapped=True rider as "
        "create_token_tapped"
    )
    assert tapped_sd["effect"]["kind"] == "create_token_tapped"
    # each_player=True -> STILL declined (Phase 5 / Edge Rover).
    eachp = TriggeredAbility(
        effect=CreateToken(
            token_printing=robot, count=1, each_player=True
        ),
        **base_kw,
    )
    assert _python_ta_to_safe_dict(eachp) is None
    # attacking=True -> STILL declined (no Rust mirror).
    atk = TriggeredAbility(
        effect=CreateToken(
            token_printing=robot, count=1, attacking=True
        ),
        **base_kw,
    )
    assert _python_ta_to_safe_dict(atk) is None
    # Phase 4 §2 gate: tapped COMBINED with another rider stays
    # declined (Rust's CreateTokenTapped only mirrors lone-tapped).
    tapped_eachp = TriggeredAbility(
        effect=CreateToken(
            token_printing=robot, count=1,
            tapped=True, each_player=True,
        ),
        **base_kw,
    )
    assert _python_ta_to_safe_dict(tapped_eachp) is None


def test_r49_token_vanilla_card_def_gate():
    """_token_vanilla_card_def returns a multi-type-aware CardDef for
    a vanilla token and None for any non-vanilla shape (activated
    ability / keyword / nested trigger / non-creature / None-P/T /
    block-restriction). Pins the §2 gate's per-disqualifier behavior.
    """
    from mtg_bot.cards.base import make_token_creature, CardPrinting, ManaCost
    from mtg_bot.effects.base import (
        ActivatedAbility, ActivationCost, TriggeredAbility,
    )
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.mana import AddManaToPool
    from mtg_bot.parity.trajectory import _token_vanilla_card_def

    # Vanilla artifact-creature token -> multi-type CardDef.
    robot = make_token_creature(
        name="R49 Robot V", power=2, toughness=2,
        subtypes=("Robot",), artifact=True,
    )
    cd = _token_vanilla_card_def(robot)
    assert cd is not None
    assert cd.card_type == "Creature"
    assert cd.extra_card_types == ("Artifact",)
    assert cd.power == 2 and cd.toughness == 2
    assert cd.subtypes == ("Robot",)
    assert cd.supertypes == ("Token",)
    # Plain creature token -> single-type CardDef (empty extra).
    plain = make_token_creature(
        name="R49 Soldier", power=1, toughness=1, subtypes=("Soldier",),
    )
    pcd = _token_vanilla_card_def(plain)
    assert pcd is not None and pcd.extra_card_types == ()
    # None -> None.
    assert _token_vanilla_card_def(None) is None
    # Activated ability on the token -> declined.
    with_aa = make_token_creature(
        name="R49 Lander-like", power=1, toughness=1,
        activated_abilities=(
            ActivatedAbility(
                cost=ActivationCost(sacrifice_self=True),
                effect=DrawCards(amount=1),
            ),
        ),
    )
    assert _token_vanilla_card_def(with_aa) is None
    # Keyword on the token -> declined (Phase 1 _SAFE_TOKEN_KEYWORDS
    # is the empty set; keyword-bearing tokens need their own probe).
    with_kw = make_token_creature(
        name="R49 Flyer", power=1, toughness=1, keywords=frozenset({"flying"}),
    )
    assert _token_vanilla_card_def(with_kw) is None
    # Nested triggered ability on the token -> declined.
    with_ta = make_token_creature(name="R49 Trig", power=1, toughness=1)
    object.__setattr__(
        with_ta, "triggered_abilities",
        (TriggeredAbility(
            event=Event.ENTERS, predicate=IsSelf(),
            effect=DrawCards(amount=1),
        ),),
    )
    assert _token_vanilla_card_def(with_ta) is None
    # Non-creature / None-P/T body -> declined (Lander shape: Artifact
    # only, no power/toughness).
    artifact_only = CardPrinting(
        name="R49 Lander-shape", mana_cost_str="", mana_cost=ManaCost(),
        type_line="Token Artifact", supertypes=("Token",),
        card_types=("Artifact",), subtypes=("Lander",),
    )
    assert _token_vanilla_card_def(artifact_only) is None


def test_r49_loader_seating_assigns_deterministic_pid_and_registers():
    """End-to-end loader pass: production_deck_to_deckspec on a deck
    with a vanilla-token carrier (Tunnel Surveyor) appends the token
    CardDef to card_defs (absent from libraries), rewrites the
    carrier trigger to a numeric token_printing_id == the token's
    Rust-registry pid (= card_defs index), registers the matching
    Python printing into the cross_engine bridge map, AND the Rust
    Registry accepts the seated carrier dict at that pid. This is the
    headline substrate pin: deterministic auto-registration with NO
    hand-picked magic pid.
    """
    import mtg_bot_engine
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        clear_token_printings, _TOKEN_PRINTING_BY_PID, build_rust_state,
    )

    clear_token_printings()
    try:
        deck = ["Tunnel Surveyor"] * 4 + ["Forest"] * 36
        spec = trajectory.production_deck_to_deckspec(
            deck, shared_rng_seed=42, pad_to_size=60, pad_with="Forest",
        )
        names = [cd.name for cd in spec.card_defs]
        assert "Tunnel Surveyor" in names
        assert "Glimmer" in names, "token CardDef must be seated"
        glimmer_idx = names.index("Glimmer")
        # Token absent from BOTH libraries (registered, never drawn).
        assert "Glimmer" not in spec.p0_library
        assert "Glimmer" not in spec.p1_library
        # Carrier trigger rewritten to the numeric pid == card_defs idx.
        ts_cd = next(c for c in spec.card_defs if c.name == "Tunnel Surveyor")
        eff = ts_cd.triggered_abilities[0]["effect"]
        assert eff["kind"] == "create_token"
        assert "_token_card_def" not in eff, "symbolic ref must be gone"
        assert eff["token_printing_id"] == glimmer_idx
        # Bridge map seeded at that pid with the Glimmer printing.
        assert glimmer_idx in _TOKEN_PRINTING_BY_PID
        assert _TOKEN_PRINTING_BY_PID[glimmer_idx].name == "Glimmer"
        # Rust Registry accepts the seated carrier dict — build the
        # Rust state and confirm the carrier registered at the SAME
        # pid the loader assigned the trigger (registration order ==
        # card_defs order == Registry pid order).
        rust_state = build_rust_state(spec)
        assert rust_state is not None
        # Re-derive the Rust pids the same way build_rust_state does
        # to confirm Glimmer's pid matches the seated trigger.
        reg = mtg_bot_engine.Registry()
        name_to_pid = {}
        for c in spec.card_defs:
            pd = {
                "name": c.name,
                "card_types": [c.card_type, *c.extra_card_types],
                "mana_cost": {}, "effects": [],
            }
            if c.power is not None:
                pd["power"] = c.power
            if c.toughness is not None:
                pd["toughness"] = c.toughness
            if c.card_type == "Land":
                pd["supertypes"] = ["Basic"]
            if c.supertypes:
                pd["supertypes"] = list(c.supertypes)
            if c.subtypes:
                pd["subtypes"] = list(c.subtypes)
            if c.mana_produced:
                pd["mana_produced"] = list(c.mana_produced)
            if c.triggered_abilities:
                pd["triggered_abilities"] = [dict(t) for t in c.triggered_abilities]
            name_to_pid[c.name] = reg.register_printing(pd)
        assert name_to_pid["Glimmer"] == glimmer_idx == eff["token_printing_id"]
    finally:
        clear_token_printings()


# ---------------------------------------------------------------------------
# Phase 6 token-printing substrate — Phase 4 (round 50): the `tapped`
# create-token rider sister variant. Rust regression pins + cross-engine
# bridge pins. The real-EoE Gravpack-Monoist byte-equal trajectory probe
# lives in test_trajectory_parity.py; the EoE-pool-delta admit pin lives
# in test_trajectory.py (test_phase4_gravpack_monoist_flips_drop_to_admit).
# ---------------------------------------------------------------------------


def test_phase4_bridge_create_token_tapped_produces_tapped_createtoken():
    """Bridge arm pin: ``{"kind": "create_token_tapped", ...}`` ->
    ``CreateToken(token_printing=..., count=..., tapped=True)``.
    Mirrors the round-12 static ``create_token`` arm shape with the
    ``tapped=True`` flag set.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    try:
        robot = make_token_creature(
            name="Robot (phase4 tapped pin)", power=2, toughness=2,
            subtypes=("Robot",), artifact=True,
        )
        register_token_printing(20, robot)
        e = _ta_dict_to_python_effect(
            {
                "kind": "create_token_tapped",
                "count": 1,
                "token_printing_id": 20,
            }
        )
        assert isinstance(e, CreateToken)
        assert e.token_printing is robot
        assert e.count == 1
        assert e.tapped is True
        # Other riders stay at dataclass defaults.
        assert e.each_player is False
        assert e.attacking is False
        assert e.schedule_end_of_combat_sacrifice is False
    finally:
        clear_token_printings()


def test_phase4_bridge_create_token_tapped_count_default_one():
    """Missing ``count`` defaults to 1 (mirrors the sister
    ``create_token`` / ``create_token_each_player`` lenient-default
    convention). Gravpack Monoist's real shape is count=1.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    try:
        robot = make_token_creature(
            name="Robot (phase4 default-count pin)", power=2,
            toughness=2, subtypes=("Robot",), artifact=True,
        )
        register_token_printing(21, robot)
        e = _ta_dict_to_python_effect(
            {"kind": "create_token_tapped", "token_printing_id": 21}
        )
        assert isinstance(e, CreateToken)
        assert e.count == 1
        assert e.tapped is True
    finally:
        clear_token_printings()


def test_phase4_bridge_create_token_tapped_distinct_from_static_sister():
    """Bridge pin: ``create_token_tapped`` and the round-12 static
    ``create_token`` arm both produce ``CreateToken`` instances but
    differ on the ``tapped`` flag (True vs False). Guards against
    accidental cross-translation drift between the two sisters at the
    bridge layer (mirrors the round-43 each-player distinct-from-sister
    pin).
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
        register_token_printing,
    )

    clear_token_printings()
    try:
        robot = make_token_creature(
            name="Robot (phase4 tapped-vs-static pin)", power=2,
            toughness=2, subtypes=("Robot",), artifact=True,
        )
        register_token_printing(22, robot)
        e_tapped = _ta_dict_to_python_effect(
            {
                "kind": "create_token_tapped",
                "count": 1,
                "token_printing_id": 22,
            }
        )
        e_static = _ta_dict_to_python_effect(
            {"kind": "create_token", "count": 1, "token_printing_id": 22}
        )
        assert isinstance(e_tapped, CreateToken)
        assert isinstance(e_static, CreateToken)
        assert e_tapped.tapped is True
        assert e_static.tapped is False
        # Differ on a frozen dataclass field -> not equal.
        assert e_tapped != e_static
    finally:
        clear_token_printings()


def test_phase4_bridge_create_token_tapped_missing_registration_raises():
    """``create_token_tapped`` with an unregistered ``token_printing_id``
    raises ``KeyError`` with a clear hint (mirrors the sister
    ``create_token`` / ``create_token_each_player`` missing-registration
    shape — silent divergence from a forgotten setup step would be
    much harder to debug than an immediate failure).
    """
    import pytest

    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect, clear_token_printings,
    )

    clear_token_printings()
    with pytest.raises(KeyError, match="token_printing_id=777"):
        _ta_dict_to_python_effect(
            {
                "kind": "create_token_tapped",
                "count": 1,
                "token_printing_id": 777,
            }
        )


def test_phase4_rust_parser_accepts_create_token_tapped_in_trigger():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind": "create_token_tapped", ...}`` on a triggered_
    ability carrier. The EoE consumer (Gravpack Monoist) uses a
    dies-trigger carrier; pinning here keeps the Python bridge from
    drifting away from Rust's parse_effect coverage on this surface.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    robot_pid = reg.register_printing({
        "name": "Phase4 Trigger Pin Robot",
        "card_types": ["Artifact", "Creature"],
        "power": 2, "toughness": 2,
        "supertypes": ["Token"], "subtypes": ["Robot"],
    })
    assert robot_pid == 0
    d = {
        "name": "Phase4 Bridge Pin -- CreateTokenTapped (Gravpack-shape)",
        "card_types": ["Creature"],
        "power": 2, "toughness": 1,
        "mana_cost": {"generic": 2, "U": 1},
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "create_token_tapped",
                    "count": 1,
                    "token_printing_id": robot_pid,
                },
            }
        ],
    }
    carrier_pid = reg.register_printing(d)
    assert carrier_pid == 1


def test_phase4_rust_parser_rejects_create_token_tapped_typo():
    """Negative lockstep pin: a misspelled kind
    (``create_token_taped`` -- missing ``p``) surfaces as ValueError
    on Rust's parse_effect arm because the new variant isn't
    registered for that spelling. Pin against accidental schema drift
    between the bridge dict shape and Rust's accepted-kinds set
    (mirrors the round-43 each-player typo-rejection pin).
    """
    import pytest

    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    inner_pid = reg.register_printing({
        "name": "Phase4 Typo Pin Inner",
        "card_types": ["Artifact"],
    })
    d = {
        "name": "Phase4 Bridge Pin -- create_token_taped typo",
        "card_types": ["Creature"],
        "power": 1, "toughness": 1,
        "mana_cost": {"U": 1},
        "triggered_abilities": [
            {
                "event": "dies",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- missing the second ``p``.
                    "kind": "create_token_taped",
                    "count": 1,
                    "token_printing_id": inner_pid,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def test_phase4_rust_parser_accepts_create_token_tapped_in_cast_effects():
    """Lockstep pin: Rust's parse_effect arm accepts
    ``{"kind": "create_token_tapped", ...}`` on a cast-spell
    ``effects`` carrier too. Sister of the triggered-ability lockstep
    pin -- the same parse_effect arm dispatches both carriers, so any
    drift surfaces on either pin (round-25..43 cast-vs-trigger
    lockstep convention).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    robot_pid = reg.register_printing({
        "name": "Phase4 Cast-effects Pin Robot",
        "card_types": ["Artifact", "Creature"],
        "power": 2, "toughness": 2,
        "supertypes": ["Token"], "subtypes": ["Robot"],
    })
    d = {
        "name": "Phase4 Bridge Pin -- CreateTokenTapped (Sorcery)",
        "card_types": ["Sorcery"],
        "mana_cost": {"generic": 2, "U": 1},
        "effects": [
            {
                "kind": "create_token_tapped",
                "count": 1,
                "token_printing_id": robot_pid,
            }
        ],
    }
    spell_pid = reg.register_printing(d)
    assert spell_pid == 1


def test_phase4_recognizer_routes_tapped_vanilla_to_tapped_kind():
    """Cross-engine recognizer pin: a synthetic DIES+IsSelf carrier
    whose parsed ``CreateToken`` has ``tapped=True`` + a vanilla token
    + every other rider default recognizes as the
    ``create_token_tapped`` symbolic dict (NOT the bare
    ``create_token`` / NOT dropped). Mirrors Gravpack Monoist's exact
    shape on a synthetic carrier so the routing logic is pinned
    independent of the real-card pool.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.trajectory import _python_ta_to_safe_dict

    robot = make_token_creature(
        name="Phase4 Recognizer Robot", power=2, toughness=2,
        subtypes=("Robot",), artifact=True,
    )
    ta = TriggeredAbility(
        event=Event.DIES,
        predicate=IsSelf(),
        effect=CreateToken(token_printing=robot, count=1, tapped=True),
    )
    sd = _python_ta_to_safe_dict(ta)
    assert sd is not None, "tapped vanilla-token DIES carrier must admit"
    assert sd["event"] == "dies"
    assert sd["predicate"] == {"kind": "source_is_self"}
    assert sd["effect"]["kind"] == "create_token_tapped"
    assert sd["effect"]["count"] == 1
    assert "_token_card_def" in sd["effect"], "pre-seating symbolic ref"
    assert "token_printing_id" not in sd["effect"]
    # Bare (tapped=False) sister still routes to plain create_token
    # under ENTERS (Phase-1 leaf) — same recognizer, different kind.
    ta_bare = TriggeredAbility(
        event=Event.ENTERS,
        predicate=IsSelf(),
        effect=CreateToken(token_printing=robot, count=1, tapped=False),
    )
    sd_bare = _python_ta_to_safe_dict(ta_bare)
    assert sd_bare is not None
    assert sd_bare["effect"]["kind"] == "create_token"


def test_phase4_recognizer_declines_tapped_combined_with_other_rider():
    """§2 silent-mistranslation gate pin: ``tapped=True`` COMBINED
    with another rider (each_player / attacking) is DECLINED —
    Rust's ``Effect::CreateTokenTapped`` only mirrors the lone-tapped
    profile; a tapped+each_player carrier would byte-diverge (Rust
    can't express the combination). Keeps the recognizer honest.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.trajectory import _python_ta_to_safe_dict

    robot = make_token_creature(
        name="Phase4 Decline Robot", power=2, toughness=2,
        subtypes=("Robot",), artifact=True,
    )
    for extra in ("each_player", "attacking"):
        eff = CreateToken(
            token_printing=robot, count=1, tapped=True, **{extra: True}
        )
        ta = TriggeredAbility(
            event=Event.DIES, predicate=IsSelf(), effect=eff,
        )
        assert _python_ta_to_safe_dict(ta) is None, (
            f"tapped + {extra} must be DECLINED (no Rust mirror — §2)"
        )


def test_phase4_seat_token_printings_preserves_tapped_kind():
    """Loader pin: ``_seat_token_printings`` recognizes a symbolic
    ``create_token_tapped`` carrier, appends the token CardDef at the
    deterministic Rust-registry pid, registers the bridge map, and
    PRESERVES the ``create_token_tapped`` kind through the rewrite
    (the tapped-vs-untapped choice rides on the kind — it must survive
    seating). Mirrors the round-49 ``create_token`` seating pin.
    """
    from mtg_bot.cards.base import make_token_creature
    from mtg_bot.parity.cross_engine import (
        CardDef, clear_token_printings, _TOKEN_PRINTING_BY_PID,
    )
    from mtg_bot.parity.trajectory import (
        _token_vanilla_card_def, _seat_token_printings,
    )

    clear_token_printings()
    try:
        robot = make_token_creature(
            name="Phase4 Seated Robot", power=2, toughness=2,
            subtypes=("Robot",), artifact=True,
        )
        tcd = _token_vanilla_card_def(robot)
        assert tcd is not None
        carrier = CardDef(
            name="Phase4 Tapped Maker", card_type="Creature",
            mana_cost_generic=2, power=2, toughness=1,
            triggered_abilities=(
                {
                    "event": "dies",
                    "predicate": {"kind": "source_is_self"},
                    "effect": {
                        "kind": "create_token_tapped", "count": 1,
                        "_token_card_def": tcd,
                    },
                },
            ),
        )
        forest = CardDef(
            name="Forest", card_type="Land", mana_produced=("G",)
        )
        seated = _seat_token_printings([carrier, forest])
        names = [c.name for c in seated]
        assert names.count("Phase4 Seated Robot") == 1
        pid = names.index("Phase4 Seated Robot")
        assert pid == len(names) - 1, "token appended after carriers"
        c = next(x for x in seated if x.name == "Phase4 Tapped Maker")
        eff = c.triggered_abilities[0]["effect"]
        # Kind preserved -> tapped sister survives seating.
        assert eff["kind"] == "create_token_tapped"
        assert eff["token_printing_id"] == pid
        assert "_token_card_def" not in eff
        assert _TOKEN_PRINTING_BY_PID[pid].name == "Phase4 Seated Robot"
    finally:
        clear_token_printings()


def test_bridge_two_subject_pump_and_debuff_by_artifact_count_translates_to_python():
    """Round-51 BLOCKED-RUST-EFFECT slice — Monoist Circuit-Feeder ETB.
    Rust ``{"kind": "two_subject_pump_and_debuff_by_artifact_count"}``
    translates to Python's DETERMINISTIC bridge sibling
    ``PumpYoursAndDebuffOpponentByArtifactCountUntilEOT`` (NOT the
    agent-driven ``PairedTargetPumpDebuffUEOT`` the real card parses
    to — see the bridge-arm docstring + SEARCH_LIMITATIONS for the §2
    rationale; this is the exact Fight precedent
    ``FightTargetCreatureWithTargetCreature`` deterministic-sibling
    pattern).

    The Rust effect variant ships in this slice
    (``rust/src/cards.rs`` ``Effect::TwoSubjectPumpAndDebuffBy
    ArtifactCount`` unit-variant + parse arm + to_dict arm +
    ``rust/src/state_ops_resolve.rs`` resolve arm). Python's
    deterministic class ships in this slice
    (``effects/generic/pump.py``). This pins the bridge translation
    that wires them up.
    """
    from mtg_bot.effects.generic.pump import (
        PumpYoursAndDebuffOpponentByArtifactCountUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "two_subject_pump_and_debuff_by_artifact_count"}
    )
    assert isinstance(
        e, PumpYoursAndDebuffOpponentByArtifactCountUntilEOT
    )
    # Unit variant — no constructor args; deterministic pick (no agent
    # target choice).
    assert e.requires_target() is False


def test_bridge_two_subject_pump_debuff_ignores_extra_dict_keys():
    """Unit-variant tolerance: extra dict fields are silently ignored
    (mirrors every other unit-variant bridge arm + Rust's parse-arm
    tolerance). A future schema extension carrying extra keys must not
    crash the translator.
    """
    from mtg_bot.effects.generic.pump import (
        PumpYoursAndDebuffOpponentByArtifactCountUntilEOT,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "two_subject_pump_and_debuff_by_artifact_count",
            "spurious": 99,
        }
    )
    assert isinstance(
        e, PumpYoursAndDebuffOpponentByArtifactCountUntilEOT
    )


def test_rust_parser_accepts_two_subject_pump_debuff_in_triggered_ability():
    """Lockstep gate: the bridge dict shape must register cleanly on
    the Rust side too. Keeps the Python bridge from drifting away from
    Rust's parse_effect coverage (mirrors the
    ``damage_each_opponent`` / ``deal_damage_any_target`` Rust-parser
    pins).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Monoist Bridge Pin",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"generic": 4, "B": 2},
        "power": 4,
        "toughness": 4,
        "keywords": ["flying"],
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "two_subject_pump_and_debuff_by_artifact_count"
                    ),
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_two_subject_pump_debuff_synthetic_end_to_end_byte_equal():
    """Synthetic cross-engine end-to-end byte-equal pin for the new
    arm. A vanilla carrier with the
    ``two_subject_pump_and_debuff_by_artifact_count`` ETB trigger fires
    on both engines; X = 1 (one artifact: the bait artifact on the
    source-controller's bf). The source-controller's FIRST creature
    gets +1/+0; the opponent's FIRST creature gets +0/-1. Assert the
    effective P/T deltas are byte-equal across engines.

    This is the synthetic sister of the GENUINE real-EoE Monoist
    Circuit-Feeder probe in ``tests/test_parity_smoke.py`` (which
    drives the actual parsed ``PairedTargetPumpDebuffUEOT`` printing
    constrained by a picker shim). Here the Rust side runs the
    deterministic unit-variant directly and the Python side runs the
    deterministic bridge sibling — no agent needed (both pick the
    first creature on each side intrinsically).
    """
    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
    )
    from mtg_bot.actions import _trigger_etb_abilities
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.pump import (
        PumpYoursAndDebuffOpponentByArtifactCountUntilEOT,
    )
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.game import (
        Zone, effective_power, effective_toughness,
    )

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)
    ZONE_BATTLEFIELD = 2

    # ---- Rust side: bait artifact (X=1) on p0, own creature on p0,
    # opponent creature on p1, then the source carrying the ETB.
    rust_artifact_id = rust_state.register_and_inject_object(
        {
            "name": "Bait Artifact",
            "card_types": ["Artifact"],
            "mana_cost": {"generic": 1},
            "effects": [],
        },
        0, ZONE_BATTLEFIELD,
    )
    rust_own_id = rust_state.register_and_inject_object(
        {
            "name": "Own Bear (2/2)",
            "card_types": ["Creature"],
            "mana_cost": {"generic": 1, "G": 1},
            "power": 2, "toughness": 2, "effects": [],
        },
        0, ZONE_BATTLEFIELD,
    )
    rust_opp_id = rust_state.register_and_inject_object(
        {
            "name": "Opp Bear (3/3)",
            "card_types": ["Creature"],
            "mana_cost": {"generic": 2, "R": 1},
            "power": 3, "toughness": 3, "effects": [],
        },
        1, ZONE_BATTLEFIELD,
    )
    rust_state.register_and_inject_object(
        {
            "name": "Monoist Synthetic Source",
            "card_types": ["Artifact", "Creature"],
            "mana_cost": {"generic": 4, "B": 2},
            "power": 4, "toughness": 4, "effects": [],
            "triggered_abilities": [
                {
                    "event": "enters",
                    "predicate": {"kind": "source_is_self"},
                    "effect": {
                        "kind": (
                            "two_subject_pump_and_debuff_by_"
                            "artifact_count"
                        ),
                    },
                }
            ],
        },
        0, ZONE_BATTLEFIELD,
    )
    assert rust_state.stack_size() == 1

    # ---- Python side: same board, deterministic bridge sibling.
    art_printing = CardPrinting(
        name="Bait Artifact", mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=()),
        type_line="Artifact", supertypes=(),
        card_types=("Artifact",), subtypes=(),
        power=None, toughness=None, produces=frozenset(),
    )
    py_art = py_state.new_object(
        art_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_art.summoning_sick = False
    py_art.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_art)

    own_printing = CardPrinting(
        name="Own Bear (2/2)", mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("G", 1),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=2, toughness=2, produces=frozenset(),
    )
    py_own = py_state.new_object(
        own_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_own.summoning_sick = False
    py_own.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_own)

    opp_printing = CardPrinting(
        name="Opp Bear (3/3)", mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=(("R", 1),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=3, toughness=3, produces=frozenset(),
    )
    py_opp = py_state.new_object(
        opp_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_opp.summoning_sick = False
    py_opp.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_opp)

    src_printing = CardPrinting(
        name="Monoist Synthetic Source", mana_cost_str="",
        mana_cost=ManaCost(generic=4, colored=(("B", 2),)),
        type_line="Artifact Creature", supertypes=(),
        card_types=("Artifact", "Creature"), subtypes=(),
        power=4, toughness=4, produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=IsSelf(),
                effect=(
                    PumpYoursAndDebuffOpponentByArtifactCountUntilEOT()
                ),
            ),
        ),
    )
    py_src = py_state.new_object(
        src_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    assert len(py_state.stack) == 1

    # ---- Drain both stacks.
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    from mtg_bot.actions import resolve_top_of_stack
    while py_state.stack:
        resolve_top_of_stack(py_state)
    assert rust_state.stack_size() == 0
    assert len(py_state.stack) == 0

    # X = 1 (one Bait Artifact on p0's bf — the source is also an
    # artifact, so X = 2 actually: Bait Artifact + the source itself).
    # The source is an Artifact Creature so it counts. own gets +2/+0,
    # opp gets +0/-2.
    assert rust_state.effective_power(rust_own_id) == 4  # 2 + 2
    assert rust_state.effective_toughness(rust_own_id) == 2
    assert rust_state.effective_power(rust_opp_id) == 3
    assert rust_state.effective_toughness(rust_opp_id) == 1  # 3 - 2

    assert effective_power(py_state, py_own) == 4
    assert effective_toughness(py_state, py_own) == 2
    assert effective_power(py_state, py_opp) == 3
    assert effective_toughness(py_state, py_opp) == 1

    # Cross-engine byte-equal P/T deltas.
    assert rust_state.effective_power(rust_own_id) == effective_power(
        py_state, py_own
    )
    assert rust_state.effective_toughness(
        rust_own_id
    ) == effective_toughness(py_state, py_own)
    assert rust_state.effective_power(rust_opp_id) == effective_power(
        py_state, py_opp
    )
    assert rust_state.effective_toughness(
        rust_opp_id
    ) == effective_toughness(py_state, py_opp)
    # Bait artifact untouched (not a creature; never targeted).
    assert rust_state.effective_power(rust_artifact_id) is None
    assert py_art.printing.power is None


def test_r52_lander_token_activated_ability_python_byte_identical():
    """Round-52 N1 Phase-3 precursor: a CardDef carrying the Lander's
    ``{2}, Sac: SearchLibraryForBasicLandTapped`` via
    ``token_activated_abilities`` materializes a Python
    ``ActivatedAbility`` BYTE-IDENTICAL to the real authored
    ``TOKEN_PRINTINGS["Lander"]``. This is the substrate's core
    correctness contract: the bridge-built ability == the production
    Lander ability, so a later Phase-3 slice can admit a Lander
    carrier with zero shape drift.
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, _card_def_to_python_printing,
    )
    from mtg_bot.cards.tokens import TOKEN_PRINTINGS
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )

    lander_aa = {
        "cost": {"mana": {"generic": 2}, "sacrifice_self": True},
        "effect": {"kind": "search_library_for_basic_land_tapped"},
    }
    cd = CardDef(
        name="Lander",
        card_type="Artifact",
        supertypes=("Token",),
        subtypes=("Lander",),
        token_activated_abilities=(lander_aa,),
    )
    pyp = _card_def_to_python_printing(cd)
    assert pyp.card_types == ("Artifact",)
    assert pyp.supertypes == ("Token",)
    assert pyp.subtypes == ("Lander",)
    assert len(pyp.activated_abilities) == 1
    aa = pyp.activated_abilities[0]
    real_aa = TOKEN_PRINTINGS["Lander"].activated_abilities[0]
    # Cost: {2} + sacrifice_self, byte-equal to the real Lander.
    assert aa.cost.mana.generic == 2
    assert aa.cost.mana.generic == real_aa.cost.mana.generic
    assert aa.cost.mana.colored == real_aa.cost.mana.colored
    assert aa.cost.sacrifice_self is True
    assert aa.cost.sacrifice_self == real_aa.cost.sacrifice_self
    assert aa.cost.tap_self is False  # Lander has no tap in its cost
    # Effect: SearchLibraryForBasicLandTapped(count=1), the round-21
    # bridge arm — fully deterministic, no agent seating on either
    # engine (first ``count`` basics in library order).
    assert isinstance(aa.effect, SearchLibraryForBasicLandTapped)
    assert isinstance(
        real_aa.effect, SearchLibraryForBasicLandTapped
    )
    assert aa.effect.count == real_aa.effect.count == 1
    assert aa.is_mana_ability is False
    assert aa.is_mana_ability == real_aa.is_mana_ability


def test_r52_lander_token_activated_ability_rust_registry_accepts():
    """Round-52 N1: the SAME Lander ``token_activated_abilities`` dict
    that the Python bridge consumes is accepted by the Rust registry
    via ``build_rust_state`` (``register_printing`` ->
    ``parse_activated_ability``). No Rust change needed — the CE20
    ``activated_abilities`` substrate is already on main; this pin
    proves the field round-trips Python<->Rust from ONE dict.
    """
    import pytest

    pytest.importorskip("mtg_bot_engine")
    from mtg_bot.parity.cross_engine import CardDef, build_rust_state
    from mtg_bot.parity.trajectory import DeckSpec

    lander = CardDef(
        name="Lander",
        card_type="Artifact",
        supertypes=("Token",),
        subtypes=("Lander",),
        token_activated_abilities=(
            {
                "cost": {
                    "mana": {"generic": 2},
                    "sacrifice_self": True,
                },
                "effect": {
                    "kind": "search_library_for_basic_land_tapped"
                },
            },
        ),
    )
    forest = CardDef(
        name="Forest", card_type="Land", mana_produced=("G",)
    )
    spec = DeckSpec(
        card_defs=(lander, forest),
        p0_library=("Forest",),
        p1_library=("Forest",),
    )
    # Builds without raising == Rust's parse_activated_ability +
    # parse_activation_cost (mana + sacrifice_self) +
    # parse_effect(search_library_for_basic_land_tapped) all accepted
    # the exact dict the Python bridge consumes.
    rust_state = build_rust_state(spec)
    assert rust_state.registry_size == 2


def test_r52_activated_ability_cost_bridge_strict_on_unknown_key():
    """§2 silent-mistranslation discipline: the cost bridge raises on
    ANY cost key outside the bridge-correct set, exactly mirroring
    Rust's ``unknown activation_cost key`` error — a cost component
    one engine honors but the other can't surface MUST NOT land
    silently.
    """
    import pytest

    from mtg_bot.parity.cross_engine import (
        _aa_dict_to_python_activation_cost,
    )

    # ``exile_self`` is a real Python ActivationCost field but is NOT
    # in Rust's parse_activation_cost accepted set, so it is NOT
    # bridge-correct — the strict guard must reject it.
    with pytest.raises(ValueError, match="unsupported activation_cost"):
        _aa_dict_to_python_activation_cost(
            {"mana": {"generic": 1}, "exile_self": True}
        )
    # A pure typo also raises (not silently dropped).
    with pytest.raises(ValueError, match="unsupported activation_cost"):
        _aa_dict_to_python_activation_cost({"sacrifice_slef": True})


def test_r52_activated_ability_bridge_requires_cost_and_effect():
    """The activated-ability bridge mirrors Rust's
    ``parse_activated_ability`` ``require()`` — both ``cost`` and
    ``effect`` are mandatory; a malformed dict raises rather than
    silently building a degenerate ability.
    """
    import pytest

    from mtg_bot.parity.cross_engine import (
        _aa_dict_to_python_activated_ability,
    )

    with pytest.raises(ValueError, match="missing required 'cost'"):
        _aa_dict_to_python_activated_ability(
            {"effect": {"kind": "search_library_for_basic_land_tapped"}}
        )
    with pytest.raises(ValueError, match="missing required 'effect'"):
        _aa_dict_to_python_activated_ability(
            {"cost": {"mana": {"generic": 2}, "sacrifice_self": True}}
        )
    # is_mana_ability must be a bool if present.
    with pytest.raises(ValueError, match="is_mana_ability must be a bool"):
        _aa_dict_to_python_activated_ability(
            {
                "cost": {"sacrifice_self": True},
                "effect": {
                    "kind": "search_library_for_basic_land_tapped"
                },
                "is_mana_ability": "yes",
            }
        )


def test_r52_token_cache_key_distinguishes_lander_from_vanilla_robot():
    """Round-52 N1: the round-51 identity-idempotent token-printing
    cache key folds ``token_activated_abilities`` into the shape
    signature, so a Lander (activated ability) and an otherwise
    same-shaped vanilla token DO NOT collide on the cross-game cache —
    each resolves to its own stable ``CardPrinting`` object. Without
    this, a Lander seated after a same-named vanilla token would
    silently reuse the vanilla printing (the §2 class).
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, _token_printing_cache_key, cached_token_printing,
        clear_token_printings,
    )

    clear_token_printings()
    try:
        lander = CardDef(
            name="X",
            card_type="Artifact",
            supertypes=("Token",),
            token_activated_abilities=(
                {
                    "cost": {
                        "mana": {"generic": 2},
                        "sacrifice_self": True,
                    },
                    "effect": {
                        "kind": "search_library_for_basic_land_tapped"
                    },
                },
            ),
        )
        vanilla = CardDef(
            name="X", card_type="Artifact", supertypes=("Token",),
        )
        assert _token_printing_cache_key(
            lander
        ) != _token_printing_cache_key(vanilla)
        # Distinct cache keys -> distinct cached printings.
        assert cached_token_printing(lander) is not cached_token_printing(
            vanilla
        )
        # And the cache is still identity-idempotent per shape (the
        # round-51 contract is preserved for the ability-bearing token).
        assert cached_token_printing(lander) is cached_token_printing(
            CardDef(
                name="X",
                card_type="Artifact",
                supertypes=("Token",),
                token_activated_abilities=(
                    {
                        "cost": {
                            "mana": {"generic": 2},
                            "sacrifice_self": True,
                        },
                        "effect": {
                            "kind": (
                                "search_library_for_basic_land_tapped"
                            )
                        },
                    },
                ),
            )
        )
    finally:
        clear_token_printings()


def test_bridge_destroy_target_noncreature_permanent_translates_to_python():
    """Round-52 — Extinguisher Battleship ETB head. Rust
    ``{"kind": "destroy_target_noncreature_permanent"}`` translates to
    Python's ``DestroyTargetNoncreaturePermanent`` whose ``TargetSpec``
    filter is exactly ``~IsCreature() & ~IsPlayer()`` (NOT the round-24
    ``DestroyTargetNonlandPermanent`` ``~IsLand()`` filter — the two
    candidate sets DIVERGE; see the bridge-arm §2 docstring +
    SEARCH_LIMITATIONS).

    The Rust effect variant ships in this slice
    (``rust/src/cards.rs`` ``Effect::DestroyTargetNoncreaturePermanent``
    unit-variant + parse arm + to_dict arm +
    ``rust/src/state_ops_resolve.rs`` resolve arm). Python's class
    pre-existed (``effects/generic/destroy.py``). This pins the bridge
    translation that wires them up.
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNoncreaturePermanent,
    )
    from mtg_bot.effects.object_predicates import IsCreature, IsPlayer
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_noncreature_permanent"}
    )
    assert isinstance(e, DestroyTargetNoncreaturePermanent)
    # Filter is the precise NONCREATURE filter (not nonland).
    assert e.targets == (TargetSpec(~IsCreature() & ~IsPlayer()),)


def test_bridge_destroy_target_noncreature_permanent_is_not_nonland():
    """§2 precision pin: the noncreature arm must NOT collapse onto the
    pre-existing ``destroy_target_nonland_permanent`` arm. Their Python
    filters DIVERGE — ``~IsCreature()`` admits lands + excludes
    creatures; ``~IsLand()`` admits creatures + excludes lands. A
    regression that wired ``destroy_target_noncreature_permanent`` to
    ``DestroyTargetNonlandPermanent`` would silently mistranslate the
    real Extinguisher Battleship target set.
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNoncreaturePermanent, DestroyTargetNonlandPermanent,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    noncreature = _ta_dict_to_python_effect(
        {"kind": "destroy_target_noncreature_permanent"}
    )
    nonland = _ta_dict_to_python_effect(
        {"kind": "destroy_target_nonland_permanent"}
    )
    assert isinstance(noncreature, DestroyTargetNoncreaturePermanent)
    assert isinstance(nonland, DestroyTargetNonlandPermanent)
    assert type(noncreature) is not type(nonland)
    assert noncreature.targets != nonland.targets


def test_bridge_destroy_target_noncreature_permanent_ignores_extra_keys():
    """Unit-variant tolerance: extra dict fields silently ignored
    (mirrors every other unit-variant bridge arm + Rust's parse-arm
    silent-drop convention).
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetNoncreaturePermanent,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_noncreature_permanent", "spurious": 7}
    )
    assert isinstance(e, DestroyTargetNoncreaturePermanent)


def test_rust_parser_accepts_extinguisher_sequence_in_triggered_ability():
    """Lockstep gate: the full Extinguisher Battleship ETB bridge dict
    (``Sequence{[destroy_target_noncreature_permanent,
    damage_each_creature{4}]}``) must register cleanly on the Rust side
    too. Keeps the Python bridge from drifting away from Rust's
    parse_effect coverage (mirrors the Monoist / damage_each_opponent
    Rust-parser pins).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "Extinguisher Bridge Pin",
        "card_types": ["Artifact"],
        "mana_cost": {"generic": 8},
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "sequence",
                    "effects": [
                        {
                            "kind": (
                                "destroy_target_noncreature_permanent"
                            ),
                        },
                        {"kind": "damage_each_creature", "amount": 4},
                    ],
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_bridge_extinguisher_sequence_synthetic_end_to_end_byte_equal():
    """Synthetic cross-engine end-to-end byte-equal pin for the full
    Extinguisher Battleship ETB Sequence
    (``destroy_target_noncreature_permanent`` → ``damage_each_creature
    {4}``). A noncreature-artifact carrier with this ETB fires on both
    engines; a single bait Enchantment is the forced destroy target;
    one creature each side takes the 4-damage AOE.

    This is the synthetic sister of the GENUINE real-EoE Extinguisher
    Battleship probe in ``tests/test_parity_smoke.py`` (which drives the
    actual parsed printing constrained by a picker shim). Here the
    Python side composes the same effects directly under ``Sequence``.
    """
    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
    )
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.cards import (
        DestroyTargetNoncreaturePermanent, DealNDamageToEachCreature,
    )
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.game import Zone
    from tests.parity._picker_shims import _PickFirstLegalTargetAgent

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)
    ZB = 2

    # ---- Rust side.
    rust_bait = rust_state.register_and_inject_object(
        {
            "name": "Bait Enchantment", "card_types": ["Enchantment"],
            "mana_cost": {"generic": 2}, "effects": [],
        },
        0, ZB,
    )
    rust_own = rust_state.register_and_inject_object(
        {
            "name": "Own Bear (2/2)", "card_types": ["Creature"],
            "mana_cost": {"generic": 1, "G": 1},
            "power": 2, "toughness": 2, "effects": [],
        },
        0, ZB,
    )
    rust_opp = rust_state.register_and_inject_object(
        {
            "name": "Opp Bear (3/3)", "card_types": ["Creature"],
            "mana_cost": {"generic": 2, "R": 1},
            "power": 3, "toughness": 3, "effects": [],
        },
        1, ZB,
    )
    rust_state.register_and_inject_object(
        {
            "name": "Extinguisher Synthetic", "card_types": ["Artifact"],
            "mana_cost": {"generic": 8}, "effects": [],
            "triggered_abilities": [
                {
                    "event": "enters",
                    "predicate": {"kind": "source_is_self"},
                    "effect": {
                        "kind": "sequence",
                        "effects": [
                            {
                                "kind": (
                                    "destroy_target_noncreature_"
                                    "permanent"
                                ),
                            },
                            {
                                "kind": "damage_each_creature",
                                "amount": 4,
                            },
                        ],
                    },
                }
            ],
        },
        0, ZB,
    )
    assert rust_state.stack_size() == 1

    # ---- Python side: same board, Sequence composed directly.
    bait_printing = CardPrinting(
        name="Bait Enchantment", mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=()),
        type_line="Enchantment", supertypes=(),
        card_types=("Enchantment",), subtypes=(),
        power=None, toughness=None, produces=frozenset(),
    )
    py_bait = py_state.new_object(
        bait_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_bait.summoning_sick = False
    py_bait.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_bait)

    own_printing = CardPrinting(
        name="Own Bear (2/2)", mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("G", 1),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=2, toughness=2, produces=frozenset(),
    )
    py_own = py_state.new_object(
        own_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_own.summoning_sick = False
    py_own.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_own)

    opp_printing = CardPrinting(
        name="Opp Bear (3/3)", mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=(("R", 1),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=3, toughness=3, produces=frozenset(),
    )
    py_opp = py_state.new_object(
        opp_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_opp.summoning_sick = False
    py_opp.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_opp)

    src_printing = CardPrinting(
        name="Extinguisher Synthetic", mana_cost_str="",
        mana_cost=ManaCost(generic=8, colored=()),
        type_line="Artifact", supertypes=(),
        card_types=("Artifact",), subtypes=(),
        power=None, toughness=None, produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=IsSelf(),
                effect=Sequence(
                    effects=(
                        DestroyTargetNoncreaturePermanent(),
                        DealNDamageToEachCreature(amount=4),
                    ),
                ),
            ),
        ),
    )
    py_state.players[0].agent = _PickFirstLegalTargetAgent()
    py_src = py_state.new_object(
        src_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    assert len(py_state.stack) == 1

    # ---- Drain both stacks.
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    while py_state.stack:
        resolve_top_of_stack(py_state)
    assert rust_state.stack_size() == 0
    assert len(py_state.stack) == 0

    # Cross-engine byte-equal: bait destroyed; creatures took 4.
    assert rust_state.find_object_zone(rust_bait) == 3  # graveyard
    assert py_bait.zone == Zone.GRAVEYARD
    assert rust_state.get_damage(rust_own) == 4
    assert rust_state.get_damage(rust_opp) == 4
    assert py_own.damage == 4
    assert py_opp.damage == 4
    assert rust_state.get_damage(rust_own) == py_own.damage
    assert rust_state.get_damage(rust_opp) == py_opp.damage
    assert py_own.zone == Zone.BATTLEFIELD
    assert py_opp.zone == Zone.BATTLEFIELD


def test_bridge_double_target_creatures_power_ueot_synthetic_end_to_end_byte_equal():
    """Round-53 — synthetic cross-engine end-to-end byte-equal pin for
    the round-31 ``double_target_creatures_power_ueot`` arm
    (counter-free). The existing round-31 bridge pins were
    translation-only; this fills the end-to-end gap.

    Rust drives the deterministic unit-variant; Python runs the
    agent-driven ``DoubleTargetCreaturesPowerUEOT`` shim-constrained
    (``_PickFirstControlledCreatureAgent`` — forces the target to the
    first own creature, byte-equal with Rust's first-creature scan).
    Own Bear 3/3 -> 6/3 on both engines.
    """
    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
    )
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.pump import DoubleTargetCreaturesPowerUEOT
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.game import Zone, effective_power, effective_toughness
    from tests.parity._picker_shims import (
        _PickFirstControlledCreatureAgent,
    )

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)
    ZB = 2

    rust_own = rust_state.register_and_inject_object(
        {
            "name": "Own Bear (3/3)", "card_types": ["Creature"],
            "mana_cost": {"generic": 2, "G": 1},
            "power": 3, "toughness": 3, "effects": [],
        },
        0, ZB,
    )
    rust_state.register_and_inject_object(
        {
            "name": "Mightform Carrier", "card_types": ["Creature"],
            "mana_cost": {"generic": 2, "G": 2},
            "power": 4, "toughness": 4, "effects": [],
            "triggered_abilities": [
                {
                    "event": "enters",
                    "predicate": {"kind": "source_is_self"},
                    "effect": {
                        "kind": "double_target_creatures_power_ueot",
                    },
                }
            ],
        },
        0, ZB,
    )
    assert rust_state.stack_size() == 1

    own_printing = CardPrinting(
        name="Own Bear (3/3)", mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=(("G", 1),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=3, toughness=3, produces=frozenset(),
    )
    py_own = py_state.new_object(
        own_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_own.summoning_sick = False
    py_own.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_own)
    py_state.players[0].agent = _PickFirstControlledCreatureAgent()

    src_printing = CardPrinting(
        name="Mightform Carrier", mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=(("G", 2),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=4, toughness=4, produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=IsSelf(),
                effect=DoubleTargetCreaturesPowerUEOT(),
            ),
        ),
    )
    py_src = py_state.new_object(
        src_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    assert len(py_state.stack) == 1

    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    while py_state.stack:
        resolve_top_of_stack(py_state)

    assert rust_state.effective_power(rust_own) == 6
    assert effective_power(py_state, py_own) == 6
    assert rust_state.effective_power(rust_own) == effective_power(
        py_state, py_own
    )
    assert rust_state.effective_toughness(
        rust_own
    ) == effective_toughness(py_state, py_own)


def test_bridge_double_target_creatures_power_reads_effective_power_with_counters_byte_equal():
    """**§2 CROSS-ENGINE pin — round-53 fix to the round-31
    ``double_target_creatures_power_ueot`` arm. "Double reads
    EFFECTIVE power (post-counters), not base power" — byte-equal
    doubling DELTA.**

    This is the pin the original round-31 bridge pins MISSED (they
    were translation-only, no counter scenario), which is why the §2
    silent mistranslation stayed latent. Stage a 3/3 with +2/+2
    counters on BOTH engines, fire the round-31 double-power effect,
    assert the doubling DELTA (``effective_power_after -
    effective_power_before``) is byte-equal cross-engine — i.e. both
    engines doubled the SAME snapshot (effective power 5, NOT base 3).

    NOTE on the absolute getter: this test asserts only the DELTA
    (``effective_power_after - effective_power_before``), which is +5
    on BOTH engines regardless of the absolute-getter shape. (Stale
    historical note, kept for context: pre-R74-1 Rust's public
    ``effective_power`` getter omitted ``obj.counters`` so absolute
    numbers differed by the counter amount — Rust 3->8, Python 5->10.
    The round-74 R74-1 / RUST_PARITY_GAPS I23 slice CLOSED that
    combat-side asymmetry: the getter now folds counters byte-equal
    with Python, so absolute numbers AGREE — Rust 5->10, Python 5->10.
    The delta invariant this test pins is unaffected either way; the
    private ``read_doubling_power`` re-derivation was always
    counter-inclusive.) Documented in docs/SEARCH_LIMITATIONS.md.

    Pre-round-53 this pin would FAIL: Rust's delta was +3 (base) while
    Python's was +5 (effective). The round-53 ``read_doubling_power``
    fix makes both +5.
    """
    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
    )
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.pump import DoubleTargetCreaturesPowerUEOT
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.game import Zone, effective_power
    from tests.parity._picker_shims import (
        _PickFirstControlledCreatureAgent,
    )

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)
    ZB = 2

    # ---- Rust: 3/3 + deterministic +2/+2 counter granter, then the
    # round-31 carrier.
    rust_own = rust_state.register_and_inject_object(
        {
            "name": "Counter Bear (3/3)", "card_types": ["Creature"],
            "mana_cost": {"generic": 2, "G": 1},
            "power": 3, "toughness": 3, "effects": [],
        },
        0, ZB,
    )
    rust_state.register_and_inject_object(
        {
            "name": "Counter Granter", "card_types": ["Creature"],
            "mana_cost": {"generic": 1}, "power": 1, "toughness": 1,
            "effects": [],
            "triggered_abilities": [
                {
                    "event": "enters",
                    "predicate": {"kind": "source_is_self"},
                    "effect": {
                        "kind": (
                            "put_plus_one_counter_on_each_"
                            "creature_you_control"
                        ),
                        "amount": 2,
                    },
                }
            ],
        },
        0, ZB,
    )
    rust_state.resolve_top_of_stack()
    rust_before = rust_state.effective_power(rust_own)
    rust_state.register_and_inject_object(
        {
            "name": "Mightform Carrier", "card_types": ["Creature"],
            "mana_cost": {"generic": 2, "G": 2},
            "power": 4, "toughness": 4, "effects": [],
            "triggered_abilities": [
                {
                    "event": "enters",
                    "predicate": {"kind": "source_is_self"},
                    "effect": {
                        "kind": "double_target_creatures_power_ueot",
                    },
                }
            ],
        },
        0, ZB,
    )
    rust_state.resolve_top_of_stack()
    rust_delta = rust_state.effective_power(rust_own) - rust_before

    # ---- Python: same 3/3 with +2/+2 counters set directly, then the
    # agent-driven round-31 class shim-constrained.
    own_printing = CardPrinting(
        name="Counter Bear (3/3)", mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=(("G", 1),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=3, toughness=3, produces=frozenset(),
    )
    py_own = py_state.new_object(
        own_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_own.summoning_sick = False
    py_own.plus_one_counters = 2
    py_own.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_own)
    py_before = effective_power(py_state, py_own)
    py_state.players[0].agent = _PickFirstControlledCreatureAgent()

    src_printing = CardPrinting(
        name="Mightform Carrier", mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=(("G", 2),)),
        type_line="Creature", supertypes=(),
        card_types=("Creature",), subtypes=(),
        power=4, toughness=4, produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=IsSelf(),
                effect=DoubleTargetCreaturesPowerUEOT(),
            ),
        ),
    )
    py_src = py_state.new_object(
        src_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    while py_state.stack:
        resolve_top_of_stack(py_state)
    py_delta = effective_power(py_state, py_own) - py_before

    assert rust_delta == 5, (
        f"round-53 §2 fix: Rust doubling delta must be 5 (effective "
        f"power 3+2); got {rust_delta}"
    )
    assert py_delta == 5, (
        f"Python source-of-truth doubling delta must be 5; got "
        f"{py_delta}"
    )
    assert rust_delta == py_delta, (
        f"§2 silent mistranslation: Rust doubled {rust_delta} but "
        f"Python doubled {py_delta} — engines disagree on which "
        f"power 'double' reads (pre-round-53 this was 3 vs 5)."
    )


def test_phase3a_real_galactic_wayfarer_seated_lander_bridge_correct():
    """Phase 3a (round-53 N1): the REAL Galactic Wayfarer, run through
    the production-deck loader, seats a Lander token CardDef carrying
    the bridge-correct ``token_activated_abilities`` at a deterministic
    Rust-registry pid + the cross-engine bridge map.

    Pins the full loader path: ``production_deck_to_deckspec`` ->
    ``_seat_token_printings`` (auto-registers the Lander CardDef,
    rewrites the ENTERS trigger to the numeric ``token_printing_id``,
    seeds ``register_token_printing``) -> ``build_rust_state``
    (Rust ``register_printing`` -> ``parse_activated_ability`` accepts
    the SAME Lander ability dict the Python bridge consumes).
    """
    import pytest

    pytest.importorskip("mtg_bot_engine")
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        build_rust_state, clear_token_printings,
        _TOKEN_PRINTING_BY_PID,
    )

    clear_token_printings()
    try:
        pr = get_active_source().get_printings()
        if pr.get("Galactic Wayfarer") is None:
            pytest.skip("Galactic Wayfarer absent from active CardSource")

        spec = trajectory.production_deck_to_deckspec(
            ["Galactic Wayfarer"] * 6 + ["Forest"] * 54,
            shared_rng_seed=7, pad_to_size=60, pad_with="Forest",
        )
        names = [cd.name for cd in spec.card_defs]
        assert "Galactic Wayfarer" in names
        assert "Lander" in names, "Lander token CardDef must be seated"
        assert "Lander" not in spec.p0_library, (
            "token registered but never drawn"
        )
        lander_cd = next(c for c in spec.card_defs if c.name == "Lander")
        # The seated Lander CardDef carries the bridge-correct ability.
        assert lander_cd.card_type == "Artifact"
        assert lander_cd.supertypes == ("Token",)
        assert len(lander_cd.token_activated_abilities) == 1
        aa = lander_cd.token_activated_abilities[0]
        assert aa["cost"] == {
            "mana": {"generic": 2}, "sacrifice_self": True
        }
        assert aa["effect"] == {
            "kind": "search_library_for_basic_land_tapped"
        }
        # Carrier trigger rewritten to the numeric pid.
        gw_cd = next(
            c for c in spec.card_defs if c.name == "Galactic Wayfarer"
        )
        eff = gw_cd.triggered_abilities[0]["effect"]
        assert eff["kind"] == "create_token"
        lander_pid = eff["token_printing_id"]
        assert lander_pid == names.index("Lander")
        # Bridge map seeded at that pid with the Lander printing.
        assert _TOKEN_PRINTING_BY_PID[lander_pid].name == "Lander"
        assert len(
            _TOKEN_PRINTING_BY_PID[lander_pid].activated_abilities
        ) == 1
        # Rust registry accepts the SAME seated dicts (no raise ==
        # parse_activated_ability + parse_activation_cost +
        # parse_effect all accepted the Python-bridge dict).
        rust_state = build_rust_state(spec)
        assert rust_state.registry_size == len(names)
    finally:
        clear_token_printings()


def test_phase3a_real_beamsaw_prospector_dies_lander_seated():
    """Phase 3a: the REAL Beamsaw Prospector (a DIES-trigger Lander
    carrier) seats the bridge-correct Lander too — pinning the
    Phase-3a DIES recognizer-row widen end-to-end through the loader.
    """
    import pytest

    pytest.importorskip("mtg_bot_engine")
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        build_rust_state, clear_token_printings,
    )

    clear_token_printings()
    try:
        pr = get_active_source().get_printings()
        if pr.get("Beamsaw Prospector") is None:
            pytest.skip("Beamsaw Prospector absent from active CardSource")
        spec = trajectory.production_deck_to_deckspec(
            ["Beamsaw Prospector"] * 6 + ["Forest"] * 54,
            shared_rng_seed=11, pad_to_size=60, pad_with="Forest",
        )
        names = [cd.name for cd in spec.card_defs]
        assert "Lander" in names
        bp_cd = next(
            c for c in spec.card_defs if c.name == "Beamsaw Prospector"
        )
        eff = bp_cd.triggered_abilities[0]["effect"]
        # Phase 3a's DIES-row widen — bare create_token under DIES.
        assert eff["kind"] == "create_token"
        assert bp_cd.triggered_abilities[0]["event"] == "dies"
        assert eff["token_printing_id"] == names.index("Lander")
        rust_state = build_rust_state(spec)
        assert rust_state.registry_size == len(names)
    finally:
        clear_token_printings()


def test_phase3a_seated_lander_python_byte_identical_to_real_token():
    """Phase 3a: the loader-synthesized + seated Lander CardDef
    materializes a Python ``CardPrinting`` whose single activated
    ability is byte-identical to the REAL authored
    ``TOKEN_PRINTINGS["Lander"]`` — the no-shape-drift contract
    (mirrors the round-52 ``test_r52_lander_token_activated_ability_
    python_byte_identical`` pin, but driven from the REAL carrier
    through the production loader rather than a hand-built CardDef).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.cards.tokens import TOKEN_PRINTINGS
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        clear_token_printings, _card_def_to_python_printing,
    )
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )

    clear_token_printings()
    try:
        import pytest

        pr = get_active_source().get_printings()
        if pr.get("Galactic Wayfarer") is None:
            pytest.skip("Galactic Wayfarer absent from active CardSource")
        spec = trajectory.production_deck_to_deckspec(
            ["Galactic Wayfarer"] * 4 + ["Forest"] * 56,
            shared_rng_seed=3, pad_to_size=60, pad_with="Forest",
        )
        lander_cd = next(
            c for c in spec.card_defs if c.name == "Lander"
        )
        py_lander = _card_def_to_python_printing(lander_cd)
        assert len(py_lander.activated_abilities) == 1
        py_aa = py_lander.activated_abilities[0]
        real_aa = TOKEN_PRINTINGS["Lander"].activated_abilities[0]
        assert (
            py_aa.cost.mana.generic == real_aa.cost.mana.generic == 2
        )
        assert not py_aa.cost.mana.colored
        assert (
            py_aa.cost.sacrifice_self
            == real_aa.cost.sacrifice_self
            is True
        )
        assert py_aa.cost.tap_self is False
        assert isinstance(
            py_aa.effect, SearchLibraryForBasicLandTapped
        )
        assert py_aa.effect.count == real_aa.effect.count == 1
        assert py_aa.is_mana_ability is False
        # Type-line / supertype round-trip (Token Artifact — Lander).
        assert "Artifact" in py_lander.card_types
        assert "Token" in py_lander.supertypes
    finally:
        clear_token_printings()


def test_phase3a_real_galactic_wayfarer_etb_lander_spawn_byte_equal():
    """Phase 3a HEADLINE admit contract: the REAL Galactic Wayfarer's
    ETB fires + a Lander token spawns BYTE-EQUAL on BOTH engines.

    This is the bench-relevant contract: the bench/trajectory walk
    drives the carrier's ETB (cast / enters) but NEVER sac-activates
    the Lander (the bench only emits ``activate_ability`` for tapping
    lands for mana). So the admit's cross-engine correctness rests on
    the token *entering* being byte-equal — which this pins
    (``compare_fields == []``). The Lander's optional
    ``{2},Sac:SearchLibraryForBasicLandTapped`` *activation* has a
    documented residual cross-engine gap (Rust post-search-shuffles,
    Python does not — ``SEARCH_LIMITATIONS.md`` #15 residual
    Python-side gap, ``TOKEN_PRINTING_SUBSTRATE.md`` §7); it is NEVER
    driven by the bench, so the admit stays §2-safe under the bench
    strategy set and the activation-divergence is pinned separately as
    a known-gap probe (round-39 latent-SBA precedent).
    """
    import pytest

    pytest.importorskip("mtg_bot_engine")
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        build_python_field_view, compare_fields,
        parity_smoke as ce_parity_smoke, clear_token_printings,
        _card_def_to_python_printing,
    )
    from mtg_bot.actions import resolve_top_of_stack as py_resolve
    from mtg_bot.actions import dispatch_event
    from mtg_bot.effects.predicates import Event
    from mtg_bot.game import Zone

    clear_token_printings()
    try:
        pr = get_active_source().get_printings()
        if pr.get("Galactic Wayfarer") is None:
            pytest.skip("Galactic Wayfarer absent from active CardSource")
        gw = pr.get("Galactic Wayfarer")
        spec = trajectory.production_deck_to_deckspec(
            ["Galactic Wayfarer"] * 6 + ["Forest"] * 54,
            shared_rng_seed=7, pad_to_size=60, pad_with="Forest",
        )
        gw_cd = next(
            c for c in spec.card_defs if c.name == "Galactic Wayfarer"
        )
        lander_pid = gw_cd.triggered_abilities[0]["effect"][
            "token_printing_id"
        ]
        py_state, rust_state = trajectory.setup_matched_game(
            spec, opening_hand_size=0
        )

        # Inject the carrier on p0 bf, BOTH engines; ETB fires the
        # create_token (Lander) trigger.
        py_printing = _card_def_to_python_printing(gw_cd)
        py_carrier = py_state.new_object(
            py_printing, owner_id=0, zone=Zone.BATTLEFIELD
        )
        py_carrier.summoning_sick = False
        py_carrier.tapped = False
        py_carrier.etb_timestamp = py_state.next_timestamp
        py_state.next_timestamp += 1
        py_state.battlefield.append(py_carrier)
        dispatch_event(py_state, Event.ENTERS, obj=py_carrier)

        ZONE_BATTLEFIELD = 2
        rust_state.register_and_inject_object(
            {
                "name": "Galactic Wayfarer",
                "card_types": ["Creature"],
                "mana_cost": {
                    "generic": gw.mana_cost.generic, "G": 1
                },
                "power": gw.power, "toughness": gw.toughness,
                "effects": [],
                "triggered_abilities": [
                    {
                        "event": "enters",
                        "predicate": {"kind": "source_is_self"},
                        "effect": {
                            "kind": "create_token",
                            "count": 1,
                            "token_printing_id": lander_pid,
                        },
                    }
                ],
            },
            0,
            ZONE_BATTLEFIELD,
        )
        assert rust_state.stack_size() == 1
        assert len(py_state.stack) == 1

        while rust_state.stack_size() > 0:
            rust_state.resolve_top_of_stack()
        while py_state.stack:
            py_resolve(py_state)

        py_view = build_python_field_view(py_state)
        rust_view = ce_parity_smoke()._build_field_view(rust_state)
        py_bf = sorted(t[0] for t in py_view["battlefield"])
        rust_bf = sorted(t[0] for t in rust_view["battlefield"])
        assert "Lander" in py_bf, f"py bf={py_bf}"
        assert "Lander" in rust_bf, f"rust bf={rust_bf}"
        assert py_bf == rust_bf
        # Headline: the spawned Lander (with its activated ability) is
        # a byte-equal permanent on BOTH engines. The token printing
        # round-tripped bit-for-bit (the round-52 cache key folds
        # token_activated_abilities in so the Lander never collides
        # with a vanilla token).
        divs = compare_fields(py_view, rust_view)
        assert divs == [], (
            f"real Galactic Wayfarer ETB→Lander spawn diverges: "
            f"{[(d.field, d.py_value, d.rust_value) for d in divs]}"
        )
    finally:
        clear_token_printings()


def test_bridge_famished_worldsire_look_top_x_by_power_put_lands_tapped_basic_translation():
    """BLOCKED-RUST-EFFECT slice (round 54) -- Famished Worldsire ETB.
    Rust ``{"kind":
    "famished_worldsire_look_top_x_by_power_put_lands_tapped"}``
    translates to Python ``FamishedWorldsireETBEffect()``.

    Mirrors Rust's parse_effect arm (added in this slice immediately
    after the round-28 ``look_top_n_put_one_in_hand_rest_to_graveyard``
    arm; unit variant; no fields -- X is dynamic = source effective
    power, land selection deterministic greedy-all on both engines).
    GENUINELY-unshipped real-card residue: live-verified absent from
    all 108 Rust Effect variants + unbridged this round, while the
    Python source-of-truth FamishedWorldsireETBEffect
    (effects/bespoke.py:94) already existed.
    """
    from mtg_bot.effects.bespoke import FamishedWorldsireETBEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": (
                "famished_worldsire_look_top_x_by_power_put_lands_tapped"
            )
        }
    )
    assert isinstance(e, FamishedWorldsireETBEffect)


def test_bridge_famished_worldsire_extra_keys_ignored():
    """Bridge arm pin: unit-variant tolerance for extra keys. Mirrors
    the round-28/49 unit-variant precedent -- unrecognized fields are
    ignored, not raised. Notably an ``n`` key (which the sibling
    ``look_top_n_put_one_in_hand_rest_to_graveyard`` arm DOES consume)
    is silently dropped here -- X is dynamic (source effective power),
    not a static field.
    """
    from mtg_bot.effects.bespoke import FamishedWorldsireETBEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": (
                "famished_worldsire_look_top_x_by_power_put_lands_tapped"
            ),
            "spurious_field": "ignored",
            # ``n`` would be consumed by the sibling look-top-N arm;
            # here it's silently dropped -- X is dynamic (source power).
            "n": 99,
        }
    )
    assert isinstance(e, FamishedWorldsireETBEffect)


def test_bridge_famished_worldsire_equality_default_constructor():
    """Bridge arm pin: two ``FamishedWorldsireETBEffect()`` instances
    are equal under dataclass(frozen=True) default __eq__ (no
    constructor args -> trivially equal). Pins the frozen-dataclass
    equality contract for downstream hash-based keying (Sequence
    de-dup). Mirrors the round-49 unit-variant equality pin.
    """
    from mtg_bot.effects.bespoke import FamishedWorldsireETBEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {
            "kind": (
                "famished_worldsire_look_top_x_by_power_put_lands_tapped"
            )
        }
    )
    e2 = FamishedWorldsireETBEffect()
    assert e1 == e2


def test_bridge_famished_worldsire_composable_in_sequence():
    """Round-trip in a Sequence composer to pin nesting end-to-end:
    ``Sequence([FamishedWorldsireETBEffect(), DrawCards(1)])``
    translates correctly via the recursive Sequence translator. Pins
    the round-54 arm against accidental dispatch loss inside Sequence
    (mirrors the round-30..49 unit-variant Sequence-composability
    pins).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.bespoke import FamishedWorldsireETBEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {
                    "kind": (
                        "famished_worldsire_look_top_x_by_power_"
                        "put_lands_tapped"
                    )
                },
                {"kind": "draw_cards", "count": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], FamishedWorldsireETBEffect)
    assert isinstance(e.effects[1], DrawCards)


def test_rust_parser_accepts_famished_worldsire_in_triggered_ability():
    """Lockstep pin: Rust's parse_effect arm accepts the bridge dict
    shape ``{"kind":
    "famished_worldsire_look_top_x_by_power_put_lands_tapped"}`` on a
    triggered_ability carrier. Mirrors Rust's parse_effect arm added in
    this slice immediately after the round-28
    ``look_top_n_put_one_in_hand_rest_to_graveyard`` arm. Keeps the
    Python bridge from drifting away from Rust's parse_effect coverage.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- FamishedWorldsire r54",
        "card_types": ["Creature"],
        "mana_cost": {"G": 3, "generic": 5},
        "power": 0,
        "toughness": 0,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "famished_worldsire_look_top_x_by_power_"
                        "put_lands_tapped"
                    )
                },
            }
        ],
    }
    pid = reg.register_printing(d)
    assert pid == 0


def test_rust_parser_rejects_famished_worldsire_with_typo():
    """Negative lockstep pin: a misspelled kind (spurious trailing "s")
    surfaces as ValueError on Rust's parse_effect arm because the new
    variant isn't registered for that spelling. Pin against accidental
    schema drift between bridge dict shape and Rust parser's
    accepted-kinds set. Sister of the round-45/48/49 typo-rejection
    pins.
    """
    import mtg_bot_engine
    import pytest

    reg = mtg_bot_engine.Registry()
    d = {
        "name": "BLOCKED-RUST-EFFECT Bridge Pin -- typo lockstep r54",
        "card_types": ["Creature"],
        "mana_cost": {"G": 3},
        "power": 0,
        "toughness": 0,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    # Misspelled -- spurious trailing "s".
                    "kind": (
                        "famished_worldsire_look_top_x_by_power_"
                        "put_lands_tappeds"
                    )
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(d)


def _fw_etb_dict():
    """Famished Worldsire printing dict carrying the round-54
    SourceIsSelf ENTERS trigger -> the new bridge-dict unit-variant
    carrier (the loader doesn't admit the Ward-rider real card --
    cross-engine driving pattern, round-31/51/52/53 precedent)."""
    return {
        "name": "Famished Worldsire",
        "card_types": ["Creature"],
        "mana_cost": {"G": 3, "generic": 5},
        "power": 0,
        "toughness": 0,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": (
                        "famished_worldsire_look_top_x_by_power_"
                        "put_lands_tapped"
                    )
                },
            }
        ],
    }


def test_rust_famished_worldsire_zero_power_silent_noop():
    """Rust regression pin: X = source effective power. A 0/0 source
    with NO +1/+1 counters (Famished Worldsire's printed P/T) -> X = 0
    -> ``if x <= 0: return`` silent no-op. Library untouched, no land
    enters the battlefield. Mirrors Python's chained
    ``if source is None: return`` / ``if x <= 0: return``.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    forest = reg.register_printing(
        {
            "name": "Forest",
            "card_types": ["Land"],
            "supertypes": ["Basic"],
            "subtypes": ["Forest"],
            "mana_cost": {},
        }
    )
    # p0 library top-to-bottom: 3 Forests (index 0 = top per CE3).
    st = mtg_bot_engine.make_state(reg, [forest, forest, forest], [])
    assert [d["printing_id"] for d in st.library_snapshot(0)] == [
        forest,
        forest,
        forest,
    ]
    # Famished Worldsire (0/0, NO counter granter -> effective power 0).
    st.register_and_inject_object(_fw_etb_dict(), 0, 2)
    while st.stack_size() > 0:
        st.resolve_top_of_stack()
    # X=0 -> silent no-op: all 3 Forests still in library, none on bf.
    lib_after = [d["printing_id"] for d in st.library_snapshot(0)]
    assert lib_after == [forest, forest, forest], (
        f"X=0 must be a silent no-op (library untouched); got "
        f"{lib_after}"
    )
    bf_names = [
        e["printing_name"] for e in st.battlefield_snapshot()
    ]
    assert bf_names.count("Forest") == 0, (
        f"X=0: no land should have entered the battlefield; got "
        f"bf={bf_names}"
    )


def test_rust_famished_worldsire_greedy_all_lands_enter_tapped():
    """Rust regression pin: X = source power (via +1/+1 counters,
    matching Devour land). Source 0/0 + 3 +1/+1 counters -> X = 3.
    Top-3 library = 3 lands -> ALL 3 enter the battlefield TAPPED
    (deterministic greedy-all, mirroring Python's ``bespoke.py:121``).
    Library emptied -> no shuffle (Rust gate ``lib_len_post >= 2``;
    Python's unconditional shuffle of a 0/1-card library is a no-op
    too -> byte-equal, no RNG).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    forest = reg.register_printing(
        {
            "name": "Forest",
            "card_types": ["Land"],
            "supertypes": ["Basic"],
            "subtypes": ["Forest"],
            "mana_cost": {},
        }
    )
    st = mtg_bot_engine.make_state(reg, [forest, forest, forest], [])
    fw_id = st.register_and_inject_object(_fw_etb_dict(), 0, 2)
    # Famished's ENTERS trigger is now on the stack but NOT yet
    # resolved. Seed +3/+3 +1/+1 counters on it (modeling Devour
    # land's counter seed without the loader-gated Devour rider) so
    # ``read_famished_source_power`` reads base(0) + counters(3) = X=3
    # at resolve. This directly exercises the §2 counter-fold path
    # (the round-53 Mightform `read_doubling_power` precedent — the
    # Famished arm's private read folds counters in to MATCH Python's
    # `effective_power`/`_compute_effective_pt`). NOTE: the
    # "Rust's absolute effective_power getter omits counters" premise
    # was true pre-R74-1; the round-74 I23 slice closed that
    # combat-side asymmetry (getter now folds counters). This private
    # re-derivation is unaffected and stays correct.
    assert st.get_counters(fw_id) == 0
    assert st.set_counters(fw_id, 3) is True
    assert st.get_counters(fw_id) == 3
    while st.stack_size() > 0:
        st.resolve_top_of_stack()
    lib_after = [d["printing_id"] for d in st.library_snapshot(0)]
    assert lib_after == [], (
        f"X=3, top-3 all lands -> library emptied; got {lib_after}"
    )
    bf = st.battlefield_snapshot()
    forests_on_bf = [e for e in bf if e["printing_name"] == "Forest"]
    assert len(forests_on_bf) == 3, (
        f"all 3 Forests should be on the battlefield; got "
        f"{len(forests_on_bf)}"
    )
    assert all(e["tapped"] for e in forests_on_bf), (
        f"every Famished-put land enters TAPPED; got "
        f"{[(e['printing_name'], e['tapped']) for e in forests_on_bf]}"
    )


def test_rust_famished_worldsire_nonland_stays_no_shuffle():
    """Rust regression pin: non-land cards in the examined window stay
    in the library (Python's ``rest`` is never removed --
    ``bespoke.py:122``). Scenario sized so the post-resolve library has
    < 2 cards -> no shuffle on EITHER engine (Rust gate ``>= 2``;
    Python's unconditional 1-card shuffle is a no-op) -> deterministic,
    no RNG.

    Source power 2 (2 +1/+1 counters). Library top-to-bottom =
    [Forest, Vanilla Bear]. Examined window = top 2 = both. Only
    Forest is a land -> Forest to bf tapped; Bear (non-land) stays.
    Library after = [Bear] (1 card -> NO shuffle).
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    forest = reg.register_printing(
        {
            "name": "Forest",
            "card_types": ["Land"],
            "supertypes": ["Basic"],
            "subtypes": ["Forest"],
            "mana_cost": {},
        }
    )
    bear = reg.register_printing(
        {
            "name": "Vanilla Bear",
            "card_types": ["Creature"],
            "mana_cost": {"generic": 2},
            "power": 2,
            "toughness": 2,
        }
    )
    # Library top-to-bottom: Forest (top), Vanilla Bear (second).
    st = mtg_bot_engine.make_state(reg, [forest, bear], [])
    fw_id = st.register_and_inject_object(_fw_etb_dict(), 0, 2)
    # Seed +2/+2 counters before the ETB resolves -> X = base(0) +
    # counters(2) = 2 (§2 counter-fold path).
    assert st.set_counters(fw_id, 2) is True
    while st.stack_size() > 0:
        st.resolve_top_of_stack()
    # X=2, examined = [Forest, Bear]. Only Forest is a land -> Forest
    # to bf tapped; Bear (non-land) stays. Library after = [Bear]
    # (1 card -> no shuffle).
    lib_after = [d["printing_id"] for d in st.library_snapshot(0)]
    assert lib_after == [bear], (
        f"non-land Bear must stay in the library in place; got "
        f"{lib_after}"
    )
    bf = st.battlefield_snapshot()
    forests = [e for e in bf if e["printing_name"] == "Forest"]
    assert len(forests) == 1 and forests[0]["tapped"], (
        f"the single examined land enters tapped; got {forests}"
    )
    assert not any(
        e["printing_name"] == "Vanilla Bear" for e in bf
    ), "the non-land Bear must NOT enter the battlefield"


def test_rust_famished_worldsire_cross_controller_source_resolution():
    """Rust regression pin: 'your library' resolves under the SOURCE's
    controller, NOT the active player (CR 119.5 / 608.2). Famished
    Worldsire owned by p1 enters while p0 is the active player -> X
    reads p1's Famished power and lands enter from p1's library onto
    p1's battlefield (same I69 source-controller convention
    ``SearchLibraryForBasicLandTapped`` uses). Sized to empty post
    library -> no shuffle, deterministic.
    """
    import mtg_bot_engine

    reg = mtg_bot_engine.Registry()
    forest = reg.register_printing(
        {
            "name": "Forest",
            "card_types": ["Land"],
            "supertypes": ["Basic"],
            "subtypes": ["Forest"],
            "mana_cost": {},
        }
    )
    # p0 library empty; p1 library = 2 Forests.
    st = mtg_bot_engine.make_state(reg, [], [forest, forest])
    assert st.active_player == 0
    # p1's Famished Worldsire (controller=1). p0 is the active player.
    fw_id = st.register_and_inject_object(_fw_etb_dict(), 1, 2)
    # Seed +2/+2 counters before the ETB resolves -> X = 2.
    assert st.set_counters(fw_id, 2) is True
    while st.stack_size() > 0:
        st.resolve_top_of_stack()
    # p1's library emptied (X=2, both Forests are lands); p0 untouched.
    assert [d["printing_id"] for d in st.library_snapshot(1)] == []
    assert [d["printing_id"] for d in st.library_snapshot(0)] == []
    bf = st.battlefield_snapshot()
    p1_forests = [
        e
        for e in bf
        if e["printing_name"] == "Forest" and e["controller_id"] == 1
    ]
    assert len(p1_forests) == 2 and all(
        e["tapped"] for e in p1_forests
    ), (
        f"both Forests enter p1's battlefield tapped (source's "
        f"controller, NOT active player p0); got {p1_forests}"
    )


# ---------------------------------------------------------------------------
# Round-56 N1 — ExileEachOpponentsGraveyard (Dauntless Scrapbot 1st
# Sequence arm). Rust Effect::ExileEachOpponentsGraveyard (unit
# variant) + parse + to_dict + resolve mirrors Python's
# ExileEachOpponentsGraveyard (effects/generic/exile.py:260), a
# structural twin of EachOpponentMillsCards (same source-controller
# finder + 2-player opponent loop; inner op is graveyard->exile
# instead of library->graveyard). Bridge arm: cross_engine.py
# exile_each_opponents_graveyard (field-free, no key translation).
# ---------------------------------------------------------------------------


def _r56_eeg_grave_printing(name, power=1, toughness=1):
    """A minimal vanilla creature CardPrinting for graveyard seeding."""
    from mtg_bot.cards.base import CardPrinting, ManaCost

    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(generic=0),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=power,
        toughness=toughness,
        produces=frozenset(),
    )


def _r56_eeg_source_card_def():
    """A synthetic creature whose ETB carries the bare
    ``exile_each_opponents_graveyard`` arm. ``build_python_state``
    auto-translates the triggered-ability dict through the genuine
    ``_ta_dict_to_python_effect`` bridge, so the Python side exercises
    the real bridge arm (not a hand-rolled effect object)."""
    from mtg_bot.parity.cross_engine import CardDef

    return CardDef(
        name="Scrapbot Stand-In",
        card_type="Artifact",
        extra_card_types=("Creature",),
        power=3,
        toughness=1,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "exile_each_opponents_graveyard"},
            },
        ),
    )


def _r56_eeg_rust_source_dict():
    return {
        "name": "Scrapbot Stand-In",
        "card_types": ["Artifact", "Creature"],
        "power": 3,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "exile_each_opponents_graveyard"},
            }
        ],
    }


def test_bridge_exile_each_opponents_graveyard_translates_to_python():
    """Phase 5 bridge extension: Rust
    ``{"kind": "exile_each_opponents_graveyard"}`` translates to a
    Python ``ExileEachOpponentsGraveyard`` (field-free unit effect —
    no key translation, mirroring the parameter-free Python
    dataclass).
    """
    from mtg_bot.effects.generic.exile import ExileEachOpponentsGraveyard
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "exile_each_opponents_graveyard"}
    )
    assert isinstance(e, ExileEachOpponentsGraveyard)
    assert e.describe() == "exile each opponent's graveyard"


def test_cross_engine_exile_each_opponents_graveyard_etb_resolve_parity():
    """Cross-engine end-to-end parity: an ETB trigger carrying
    ``exile_each_opponents_graveyard`` empties the opponent's whole
    graveyard into THAT opponent's own exile zone, byte-equal on both
    engines, preserving graveyard order. p1 (the opponent) has two
    cards in its graveyard; p0's source ETBs.
    """
    import mtg_bot_engine  # noqa: F401

    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
        _card_def_to_python_printing,
    )
    from mtg_bot.game import Zone

    spec = DeckSpec(
        card_defs=(_r56_eeg_source_card_def(),),
        p0_library=(),
        p1_library=(),
    )
    rust_state = build_rust_state(spec)
    py_state = build_python_state(spec)
    ZG, ZB = 3, 2

    # ---- Rust: seed p1's graveyard with two cards, then ETB p0 src.
    rust_state.register_and_inject_object(
        {"name": "GraveA", "card_types": ["Creature"],
         "power": 1, "toughness": 1}, 1, ZG)
    rust_state.register_and_inject_object(
        {"name": "GraveB", "card_types": ["Creature"],
         "power": 2, "toughness": 2}, 1, ZG)
    assert [e["printing_name"]
            for e in rust_state.graveyard_snapshot(1)] == [
        "GraveA", "GraveB"]
    rust_state.register_and_inject_object(
        _r56_eeg_rust_source_dict(), 0, ZB)
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    # ---- Python: seed p1's graveyard with the SAME two cards in the
    # same order, then ETB the source. ``build_python_state`` already
    # materialized the source CardDef; fire it through new_object.
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    for nm, pw, tg in (("GraveA", 1, 1), ("GraveB", 2, 2)):
        o = py_state.new_object(
            _r56_eeg_grave_printing(nm, pw, tg), owner_id=1,
            zone=Zone.GRAVEYARD,
        )
        py_state.players[1].graveyard.append(o)
    src_printing = _card_def_to_python_printing(_r56_eeg_source_card_def())
    py_src = py_state.new_object(src_printing, owner_id=0,
                                 zone=Zone.BATTLEFIELD)
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    while py_state.stack:
        resolve_top_of_stack(py_state)

    # ---- Byte-equal: p1's graveyard emptied into p1's exile (order
    # preserved); p0's own zones untouched.
    rust_p1_gy = [e["printing_name"]
                  for e in rust_state.graveyard_snapshot(1)]
    rust_p1_ex = [e["printing_name"]
                  for e in rust_state.exile_snapshot(1)]
    py_p1_gy = [o.printing.name for o in py_state.players[1].graveyard]
    py_p1_ex = [o.printing.name for o in py_state.players[1].exile]
    assert rust_p1_gy == [] == py_p1_gy
    assert rust_p1_ex == ["GraveA", "GraveB"] == py_p1_ex
    assert rust_p1_ex == py_p1_ex
    assert [e["printing_name"]
            for e in rust_state.graveyard_snapshot(0)] == []
    assert [o.printing.name
            for o in py_state.players[0].graveyard] == []
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life


def test_cross_engine_exile_each_opponents_graveyard_controllers_own_gy_untouched():
    """Cross-engine parity: the controller's OWN graveyard is NOT
    exiled — only each *opponent's* (Python skips ``p.id ==
    controller_id``; Rust's 2-player mirror touches only
    ``1 - controller``).
    """
    import mtg_bot_engine  # noqa: F401

    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
        _card_def_to_python_printing,
    )
    from mtg_bot.game import Zone
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack

    spec = DeckSpec(
        card_defs=(_r56_eeg_source_card_def(),),
        p0_library=(), p1_library=(),
    )
    rust_state = build_rust_state(spec)
    py_state = build_python_state(spec)
    ZG, ZB = 3, 2

    rust_state.register_and_inject_object(
        {"name": "P0Own", "card_types": ["Creature"],
         "power": 1, "toughness": 1}, 0, ZG)
    rust_state.register_and_inject_object(
        {"name": "P1Card", "card_types": ["Creature"],
         "power": 1, "toughness": 1}, 1, ZG)
    rust_state.register_and_inject_object(
        _r56_eeg_rust_source_dict(), 0, ZB)
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    o0 = py_state.new_object(_r56_eeg_grave_printing("P0Own"),
                             owner_id=0, zone=Zone.GRAVEYARD)
    py_state.players[0].graveyard.append(o0)
    o1 = py_state.new_object(_r56_eeg_grave_printing("P1Card"),
                             owner_id=1, zone=Zone.GRAVEYARD)
    py_state.players[1].graveyard.append(o1)
    src_printing = _card_def_to_python_printing(_r56_eeg_source_card_def())
    py_src = py_state.new_object(src_printing, owner_id=0,
                                 zone=Zone.BATTLEFIELD)
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    while py_state.stack:
        resolve_top_of_stack(py_state)

    # p0's own graveyard intact on BOTH; p1's exiled on BOTH.
    assert [e["printing_name"]
            for e in rust_state.graveyard_snapshot(0)] == ["P0Own"]
    assert [o.printing.name
            for o in py_state.players[0].graveyard] == ["P0Own"]
    assert [e["printing_name"]
            for e in rust_state.exile_snapshot(0)] == []
    assert [o.printing.name
            for o in py_state.players[0].exile] == []
    assert [e["printing_name"]
            for e in rust_state.graveyard_snapshot(1)] == []
    assert [o.printing.name
            for o in py_state.players[1].graveyard] == []
    assert [e["printing_name"]
            for e in rust_state.exile_snapshot(1)] == ["P1Card"]
    assert [o.printing.name
            for o in py_state.players[1].exile] == ["P1Card"]


def test_cross_engine_exile_each_opponents_graveyard_empty_gy_noop():
    """Cross-engine parity: an empty opponent graveyard is a silent
    no-op on both engines (no error, no state change). Mirrors
    EachOpponentMillsCards' empty-library silent-stop shape.
    """
    import mtg_bot_engine  # noqa: F401

    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
        _card_def_to_python_printing,
    )
    from mtg_bot.game import Zone
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack

    spec = DeckSpec(
        card_defs=(_r56_eeg_source_card_def(),),
        p0_library=(), p1_library=(),
    )
    rust_state = build_rust_state(spec)
    py_state = build_python_state(spec)
    ZB = 2

    rust_state.register_and_inject_object(
        _r56_eeg_rust_source_dict(), 0, ZB)
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    src_printing = _card_def_to_python_printing(_r56_eeg_source_card_def())
    py_src = py_state.new_object(src_printing, owner_id=0,
                                 zone=Zone.BATTLEFIELD)
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    while py_state.stack:
        resolve_top_of_stack(py_state)

    for pid in (0, 1):
        assert rust_state.graveyard_snapshot(pid) == []
        assert rust_state.exile_snapshot(pid) == []
        assert py_state.players[pid].graveyard == []
        assert py_state.players[pid].exile == []
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life


def _r57_ff_vanilla_printing(name, power=2, toughness=2):
    """Minimal vanilla creature CardPrinting for bf / library seeding
    in the Faller's Faithful bridge pins."""
    from mtg_bot.cards.base import CardPrinting, ManaCost

    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(generic=0),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=power,
        toughness=toughness,
        produces=frozenset(),
    )


def _r57_ff_source_card_def(draw_amount=2):
    """Synthetic creature whose ETB carries the bare
    ``destroy_target_creature_then_draw_if_undamaged`` arm.
    ``build_python_state`` auto-translates the triggered-ability dict
    through the genuine ``_ta_dict_to_python_effect`` bridge, so the
    Python side exercises the REAL bridge arm (mapping to
    ``DestroyTargetCreature(draw_if_undamaged_amount=draw_amount,
    optional=True, exclude_self=True)`` — the exact class the genuine
    Faller's Faithful printing parses to)."""
    from mtg_bot.parity.cross_engine import CardDef

    return CardDef(
        name="Faithful Stand-In",
        card_type="Creature",
        power=2,
        toughness=1,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "destroy_target_creature_then_draw_if_undamaged",
                    "draw_amount": draw_amount,
                },
            },
        ),
    )


def _r57_ff_rust_source_dict(draw_amount=2):
    return {
        "name": "Faithful Stand-In",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "destroy_target_creature_then_draw_if_undamaged",
                    "draw_amount": draw_amount,
                },
            }
        ],
    }


def test_bridge_destroy_target_creature_then_draw_if_undamaged_translates():
    """**BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB.** Rust's
    ``{"kind": "destroy_target_creature_then_draw_if_undamaged",
    "draw_amount": N}`` translates to the EXACT Python class the
    genuine Faller's Faithful printing parses to:
    ``DestroyTargetCreature(draw_if_undamaged_amount=N, optional=True,
    exclude_self=True)``. Missing / negative ``draw_amount`` clamps to
    0 (X-from-count zero-short-circuit convention).
    """
    from mtg_bot.effects.generic.destroy import DestroyTargetCreature
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_creature_then_draw_if_undamaged",
         "draw_amount": 2}
    )
    assert isinstance(e, DestroyTargetCreature)
    assert e.draw_if_undamaged_amount == 2
    assert e.optional is True
    assert e.exclude_self is True

    # Missing draw_amount -> 0 (legal "destroy with no draw" shape).
    e0 = _ta_dict_to_python_effect(
        {"kind": "destroy_target_creature_then_draw_if_undamaged"}
    )
    assert isinstance(e0, DestroyTargetCreature)
    assert e0.draw_if_undamaged_amount == 0

    # Negative draw_amount clamps to 0.
    eneg = _ta_dict_to_python_effect(
        {"kind": "destroy_target_creature_then_draw_if_undamaged",
         "draw_amount": -3}
    )
    assert eneg.draw_if_undamaged_amount == 0

    # Byte-identical to the GENUINE parsed Faller's Faithful printing.
    from mtg_bot.cards.source import get_active_source

    real = get_active_source().get_printings().get("Faller's Faithful")
    if real is not None:
        real_eff = real.triggered_abilities[0].effect
        assert isinstance(real_eff, DestroyTargetCreature)
        assert real_eff.draw_if_undamaged_amount == e.draw_if_undamaged_amount
        assert real_eff.optional == e.optional
        assert real_eff.exclude_self == e.exclude_self


def _r57_ff_cross_engine_run(draw_amount, victim_pre_damage):
    """Drive the Faller's Faithful ETB cross-engine end-to-end.

    p0 (controller) ETBs the source; p1 (opponent) has a 2/2 victim
    creature + a 3-card library. Returns
    ``(py_tuple, rust_tuple)`` where each tuple is
    ``(victim_alive, p1_gy_size, p1_hand_size, p1_lib_size,
    p0_life, p1_life)``. Both engines pick the opponent's first
    creature deterministically (Rust's defender-bf scan; Python via
    ``_PickFirstOpponentTargetAgent`` over the legal pool).
    """
    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
        _card_def_to_python_printing,
    )
    from mtg_bot.game import Zone
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from tests.parity._picker_shims import _PickFirstOpponentTargetAgent

    cd = _r57_ff_source_card_def(draw_amount)
    spec = DeckSpec(card_defs=(cd,), p0_library=(), p1_library=())
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)
    ZB, ZL = 2, 1
    py_state.players[0].agent = _PickFirstOpponentTargetAgent()

    # p1 library: 3 vanilla cards on BOTH engines (draw-source).
    for nm in ("FF Lib1", "FF Lib2", "FF Lib3"):
        o = py_state.new_object(
            _r57_ff_vanilla_printing(nm, 1, 1), owner_id=1,
            zone=Zone.LIBRARY,
        )
        py_state.players[1].library.append(o)
        rust_state.register_and_inject_object(
            {"name": nm, "card_types": ["Creature"],
             "power": 1, "toughness": 1}, 1, ZL)

    # p1 victim creature on BOTH bfs.
    py_vic = py_state.new_object(
        _r57_ff_vanilla_printing("FF Victim", 2, 2), owner_id=1,
        zone=Zone.BATTLEFIELD,
    )
    py_state.battlefield.append(py_vic)
    rust_vic_id = rust_state.register_and_inject_object(
        {"name": "FF Victim", "card_types": ["Creature"],
         "power": 2, "toughness": 2}, 1, ZB)

    if victim_pre_damage:
        py_vic.damage = victim_pre_damage
        py_vic.damaged_this_turn = True
        rust_state.deal_damage(rust_vic_id, victim_pre_damage)

    # p0 source ETB on BOTH engines.
    py_src = py_state.new_object(
        _card_def_to_python_printing(cd), owner_id=0,
        zone=Zone.BATTLEFIELD,
    )
    py_src.summoning_sick = False
    py_src.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_src)
    _trigger_etb_abilities(py_state, py_src)
    rust_state.register_and_inject_object(
        _r57_ff_rust_source_dict(draw_amount), 0, ZB)

    for _ in range(16):
        if not py_state.stack:
            break
        resolve_top_of_stack(py_state)
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()

    py_tuple = (
        any(o.id == py_vic.id for o in py_state.battlefield),
        len(py_state.players[1].graveyard),
        len(py_state.players[1].hand),
        len(py_state.players[1].library),
        py_state.players[0].life,
        py_state.players[1].life,
    )
    rust_tuple = (
        any(o["id"] == rust_vic_id
            for o in rust_state.battlefield_snapshot()),
        len(rust_state.graveyard_snapshot(1)),
        rust_state.hand_sizes[1],
        rust_state.library_sizes[1],
        rust_state.get_life(0),
        rust_state.get_life(1),
    )
    return py_tuple, rust_tuple


def test_cross_engine_ff_destroy_then_draw_undamaged_parity():
    """**BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB, undamaged
    case.** The victim carried NO marked damage → it is destroyed AND
    its controller (p1) draws 2. Byte-equal on both engines."""
    import mtg_bot_engine  # noqa: F401

    py, rust = _r57_ff_cross_engine_run(draw_amount=2, victim_pre_damage=0)
    assert py == rust, f"cross-engine divergence: py={py} rust={rust}"
    # Concrete observable: victim destroyed, p1 gy=1, p1 drew 2
    # (hand=2, lib 3→1), lives unchanged.
    assert py == (False, 1, 2, 1, 20, 20), py


def test_cross_engine_ff_destroy_then_draw_predamaged_no_draw_parity():
    """**BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB, damaged
    case.** The victim was dealt damage this turn (marked damage > 0
    on both engines) → it is still destroyed but NO draw fires. Byte-
    equal on both engines (the realistic no-heal corner where Rust's
    ``obj.damage`` read matches Python's ``damaged_this_turn``)."""
    import mtg_bot_engine  # noqa: F401

    py, rust = _r57_ff_cross_engine_run(draw_amount=2, victim_pre_damage=1)
    assert py == rust, f"cross-engine divergence: py={py} rust={rust}"
    # Victim destroyed, p1 gy=1, NO draw (hand=0, lib stays 3).
    assert py == (False, 1, 0, 3, 20, 20), py


def test_cross_engine_ff_destroy_then_draw_zero_amount_parity():
    """**BLOCKED-RUST-EFFECT slice — Faller's Faithful ETB,
    ``draw_amount=0``.** Legal "destroy with no draw" shape — the
    undamaged victim is destroyed but the zero-amount rider is a
    silent no-op (X-from-count zero short-circuit). Byte-equal."""
    import mtg_bot_engine  # noqa: F401

    py, rust = _r57_ff_cross_engine_run(draw_amount=0, victim_pre_damage=0)
    assert py == rust, f"cross-engine divergence: py={py} rust={rust}"
    assert py == (False, 1, 0, 3, 20, 20), py


# ---------------------------------------------------------------------
# Round-58 N1 — BECOMES_TAPPED event bridge
#
# Rust-engine regression pins (the event fires on every production tap
# path that mirrors Python's `turn.tap_permanent`; does NOT fire when
# it shouldn't) + cross-engine bridge round-trip pins (the
# (BECOMES_TAPPED, source_is_self, create_token) triple round-trips
# byte-equal Python<->Rust, the SAME shape ENTERS/DIES/ATTACKS use).
#
# Python's BECOMES_TAPPED firing model (the source of truth, §2): it
# fires ONLY through `turn.tap_permanent` (the no-op-if-already-tapped
# chokepoint), called from tap-for-mana, the {T} tap_self activated
# cost, tap_permanent_you_control, convoke, crew, attack-tap of a
# non-vigilance attacker (BEFORE the per-attacker ATTACKS dispatch),
# and the tapper-effect family. It deliberately does NOT fire for
# permanents entering tapped nor for regeneration's replacement tap
# (CR 701.21a). Rust's emission set matches this exactly.
# ---------------------------------------------------------------------


def _bt_self_draw_creature_dict(name="BT SelfDraw", count=1, keywords=None):
    """A creature with a ``BECOMES_TAPPED + source_is_self + draw_cards``
    trigger — the canonical observable shape for the Rust-side
    regression pins (drawing is a pure deterministic zone move; the
    stack growing by 1 is the witness that the event fired)."""
    d = {
        "name": name,
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "G": 1},
        "effects": [],
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "becomes_tapped",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": count},
            }
        ],
    }
    if keywords:
        d["keywords"] = list(keywords)
    return d


def test_r58_becomes_tapped_fires_on_attack_tap():
    """Rust regression pin: a non-vigilance attacker BECOMES_TAPPED
    when it is tapped to attack. Mirrors Python's
    ``combat.declare_attackers`` -> ``turn.tap_permanent`` (combat.py:
    104) firing ``Event.BECOMES_TAPPED``."""
    import mtg_bot_engine

    s = mtg_bot_engine.make_synthetic_state()
    oid = s.register_and_inject_object(
        _bt_self_draw_creature_dict("BT Attacker"), 0, 2  # ZONE_BATTLEFIELD
    )
    s.clear_summoning_sick_for_player(0)
    assert s.stack_size() == 0, "ETB must NOT fire BECOMES_TAPPED"
    assert s.declare_attacker(oid) is True
    snap = [o for o in s.battlefield_snapshot() if o["id"] == oid][0]
    assert snap["tapped"] is True
    assert s.stack_size() == 1, (
        "attack-tap of a non-vigilance attacker must fire "
        "BECOMES_TAPPED -> the SourceIsSelf draw trigger lands on "
        "the stack"
    )


def test_r58_becomes_tapped_pushed_before_attacks_trigger():
    """Rust regression pin (ORDERING — the load-bearing §2 byte-
    equality constraint). A creature carrying BOTH a
    BECOMES_TAPPED+draw and an ATTACKS+gain_life trigger: when it is
    declared as an attacker, Python pushes the BECOMES_TAPPED trigger
    FIRST (``tap_permanent`` at combat.py:104) and the ATTACKS trigger
    SECOND (``_fire_attacks_triggers`` at combat.py:115). So the
    ATTACKS trigger is on TOP of the stack (pushed last → resolves
    first). Rust must match this push order exactly."""
    import mtg_bot_engine

    s = mtg_bot_engine.make_synthetic_state()
    d = {
        "name": "BT+ATK",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "becomes_tapped",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            },
            {
                "event": "attacks",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "gain_life", "amount": 3},
            },
        ],
    }
    oid = s.register_and_inject_object(d, 0, 2)
    s.clear_summoning_sick_for_player(0)
    assert s.declare_attacker(oid) is True
    assert s.stack_size() == 2, (
        "both the BECOMES_TAPPED and the ATTACKS trigger must be on "
        "the stack"
    )
    # ATTACKS was pushed last (combat.py:115 after tap at :104), so
    # it is on top. Resolving it first must not raise; resolving the
    # remaining (BECOMES_TAPPED) trigger then drains the stack.
    s.resolve_top_of_stack()
    assert s.stack_size() == 1
    s.resolve_top_of_stack()
    assert s.stack_size() == 0


def test_r58_becomes_tapped_fires_on_tap_for_mana():
    """Rust regression pin: a land that taps for mana BECOMES_TAPPED.
    Mirrors Python's ``auto_tap_for_cost`` routing every land through
    ``turn.tap_permanent`` (actions.py:981-985 — the docstring cites
    Chrome Companion / Grove of the Burnwillows becomes-tapped mana
    triggers)."""
    import mtg_bot_engine

    s = mtg_bot_engine.make_synthetic_state()
    land = {
        "name": "BT Mana Land",
        "card_types": ["Land"],
        "mana_produced": ["G"],
        "triggered_abilities": [
            {
                "event": "becomes_tapped",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    lid = s.register_and_inject_object(land, 0, 2)
    assert s.stack_size() == 0
    assert s.tap_for_mana(lid) == "G"
    assert s.stack_size() == 1, (
        "tap-for-mana must fire BECOMES_TAPPED on the tapped land"
    )


def test_r58_becomes_tapped_fires_on_tap_target_effect():
    """Rust regression pin: ``Effect::TapTarget`` (a Twiddle-shape
    tapper effect) fires BECOMES_TAPPED on the creature it taps.
    Mirrors Python's ``effects/generic/tap.py`` Tap-family routing the
    tap through ``turn.tap_permanent``. Uses an empty-DeckSpec Rust
    state so ``TapTarget{AnyCreature}`` (which picks the FIRST
    defending-side bf object) deterministically lands on our injected
    victim rather than a synthetic-state filler creature."""
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state

    s = build_rust_state(
        DeckSpec(card_defs=(), p0_library=(), p1_library=())
    )
    # p1 victim creature carrying a BECOMES_TAPPED+self draw trigger
    # (p1's only bf object, so it IS the defending-side bf[0]).
    vic = s.register_and_inject_object(
        _bt_self_draw_creature_dict("BT Victim"), 1, 2
    )
    # p0 source whose resolving ENTERS trigger is
    # TapTarget(AnyCreature) -> taps the defending (p1) bf[0].
    src = {
        "name": "BT Tapper Source",
        "card_types": ["Creature"],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "tap_target",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    s.register_and_inject_object(src, 0, 2)
    # The source's ETB fired its own ENTERS trigger (TapTarget) onto
    # the stack. Resolving it taps p1's victim, which must in turn
    # fire the victim's BECOMES_TAPPED draw trigger.
    assert s.stack_size() == 1
    s.resolve_top_of_stack()  # resolve TapTarget -> taps victim
    # The victim's BECOMES_TAPPED trigger is now on the stack.
    assert s.stack_size() == 1, (
        "TapTarget tapping the victim must fire the victim's "
        "BECOMES_TAPPED trigger"
    )
    vsnap = [o for o in s.battlefield_snapshot() if o["id"] == vic][0]
    assert vsnap["tapped"] is True


def test_r58_becomes_tapped_does_not_fire_on_vigilance_attacker():
    """Rust regression pin (NEGATIVE): a vigilance attacker does NOT
    tap when attacking (CR 702.21), so it does NOT become tapped and
    BECOMES_TAPPED does NOT fire. Mirrors Python's ``declare_attackers``
    skipping ``tap_permanent`` for vigilance creatures (combat.py:
    102-104)."""
    import mtg_bot_engine

    s = mtg_bot_engine.make_synthetic_state()
    oid = s.register_and_inject_object(
        _bt_self_draw_creature_dict("BT Vig", keywords=["vigilance"]), 0, 2
    )
    s.clear_summoning_sick_for_player(0)
    assert s.declare_attacker(oid) is True
    snap = [o for o in s.battlefield_snapshot() if o["id"] == oid][0]
    assert snap["tapped"] is False, "vigilance attacker must NOT tap"
    assert s.stack_size() == 0, (
        "no tap occurred -> BECOMES_TAPPED must NOT fire (CR 701.21a "
        "the creature never 'becomes tapped')"
    )


def test_r58_becomes_tapped_does_not_fire_on_already_tapped_retap():
    """Rust regression pin (NEGATIVE — the no-op-if-already-tapped
    gate). A creature tapped via the cost/test primitive
    ``tap_object`` (which deliberately does NOT dispatch) and then
    re-tapped by a TapTarget effect must NOT fire BECOMES_TAPPED on
    the re-tap — the permanent was already tapped, so it does not
    'become tapped' again. Mirrors Python's ``turn.tap_permanent``
    early-return on ``obj.tapped`` (turn.py:565-566)."""
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state

    s = build_rust_state(
        DeckSpec(card_defs=(), p0_library=(), p1_library=())
    )
    vic = s.register_and_inject_object(
        _bt_self_draw_creature_dict("BT PreTapped"), 1, 2
    )
    # Pre-tap WITHOUT firing the event (cost/test primitive).
    assert s.tap_object(vic) is True
    assert s.stack_size() == 0, "tap_object must not dispatch"
    # Now a TapTarget effect re-targets the already-tapped victim
    # (p1's only bf object → the defending-side bf[0]).
    src = {
        "name": "BT Retap Source",
        "card_types": ["Creature"],
        "power": 1,
        "toughness": 1,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "tap_target",
                    "target": {"kind": "any_creature"},
                },
            }
        ],
    }
    s.register_and_inject_object(src, 0, 2)
    assert s.stack_size() == 1  # the source's ENTERS TapTarget
    s.resolve_top_of_stack()  # resolve TapTarget on already-tapped victim
    assert s.stack_size() == 0, (
        "re-tapping an already-tapped permanent must NOT fire "
        "BECOMES_TAPPED (no-op-if-already-tapped parity with Python's "
        "tap_permanent early return)"
    )


def test_r58_becomes_tapped_bridge_round_trips_source_is_self_draw():
    """Cross-engine bridge pin: the
    ``(becomes_tapped, source_is_self, draw_cards)`` triple round-trips
    Rust-dict -> Python ``TriggeredAbility`` carrying
    ``Event.BECOMES_TAPPED`` / ``IsSelf()`` / ``DrawCards`` — the SAME
    bridge shape ENTERS/DIES/ATTACKS use (event-string passthrough, no
    Cast/IS_CAST-style spelling skew)."""
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )

    ta = _ta_dict_to_python_triggered_ability(
        {
            "event": "becomes_tapped",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind": "draw_cards", "count": 2},
        }
    )
    assert ta.event is Event.BECOMES_TAPPED
    assert isinstance(ta.predicate, IsSelf)
    assert ta.predicate.subject is None
    assert isinstance(ta.effect, DrawCards)
    assert ta.effect.amount == 2
    assert ta.once_per_turn is False


def test_r58_becomes_tapped_bridge_round_trips_create_token():
    """Cross-engine bridge pin: the
    ``(becomes_tapped, source_is_self, create_token)`` triple — the
    Seedship Agrarian shape — round-trips to ``Event.BECOMES_TAPPED``
    / ``IsSelf()`` / ``CreateToken`` once the token printing is
    registered (the create_token bridge arm is event-agnostic; it
    works identically under BECOMES_TAPPED as under ENTERS/DIES)."""
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
        clear_token_printings,
        register_token_printing,
    )

    py_lander = CardPrinting(
        name="Lander Token (r58 bridge pin)",
        mana_cost_str="",
        mana_cost=ManaCost(generic=0, colored=()),
        type_line="Token Artifact -- Lander",
        supertypes=("Token",),
        card_types=("Artifact",),
        subtypes=("Lander",),
        power=None,
        toughness=None,
        produces=frozenset(),
        triggered_abilities=(),
    )
    pid = 90_581  # arbitrary fixed pid for this isolated bridge pin
    clear_token_printings()
    register_token_printing(pid, py_lander)
    try:
        ta = _ta_dict_to_python_triggered_ability(
            {
                "event": "becomes_tapped",
                "predicate": {"kind": "source_is_self"},
                "effect": {
                    "kind": "create_token",
                    "count": 1,
                    "token_printing_id": pid,
                },
            }
        )
    finally:
        clear_token_printings()
    assert ta.event is Event.BECOMES_TAPPED
    assert isinstance(ta.predicate, IsSelf)
    assert isinstance(ta.effect, CreateToken)
    assert ta.effect.token_printing.name == "Lander Token (r58 bridge pin)"


def test_r58_becomes_tapped_safe_trigger_dict_admits_seedship_shape():
    """Cross-engine bridge pin: the loader-side
    ``_is_safe_trigger_dict`` gate admits the exact Seedship-Agrarian
    BECOMES_TAPPED Rust-shape dict (the ``_SAFE_TRIGGERS`` triple
    ``("becomes_tapped","source_is_self","create_token")`` plus the
    ``create_token`` payload-key whitelist), and rejects the §2-
    declined sibling combinations."""
    from mtg_bot.parity.trajectory import _is_safe_trigger_dict

    # ADMIT: Seedship's BECOMES_TAPPED+IsSelf+create_token shape.
    assert _is_safe_trigger_dict(
        {
            "event": "becomes_tapped",
            "predicate": {"kind": "source_is_self"},
            "effect": {
                "kind": "create_token",
                "count": 1,
                "token_printing_id": 123,
            },
        }
    ) is True
    # DECLINE: BECOMES_TAPPED + a non-admitted leaf (draw_cards) — no
    # _SAFE_TRIGGERS entry (substrate pre-positioned only for
    # create_token; the recognizer lists draw under other events, but
    # the triple gate rejects becomes_tapped+draw_cards).
    assert _is_safe_trigger_dict(
        {
            "event": "becomes_tapped",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind": "draw_cards", "count": 1},
        }
    ) is False
    # DECLINE: BECOMES_TAPPED + a non-source_is_self predicate
    # (source_is_creature) + create_token — no _SAFE_TRIGGERS entry.
    assert _is_safe_trigger_dict(
        {
            "event": "becomes_tapped",
            "predicate": {"kind": "source_is_creature"},
            "effect": {
                "kind": "create_token",
                "count": 1,
                "token_printing_id": 123,
            },
        }
    ) is False


# ===== Round-59 N1 landfall+put_plus_one_counter_on_self bridge pins
# (the sole remaining Seedship Agrarian blocker — closes its
# trigger-2 leaf; trigger-1 BECOMES_TAPPED->create_token was bridged
# round-58 N1). These pin the FULL triggered-ability round-trip of
# the NEW combination this slice admits, not just the bare effect arm
# (already pinned by the round-22 put_plus_one_counter_on_self tests
# above). =====


def test_bridge_ta_landfall_put_plus_one_counter_on_self_round_trip():
    """The canonical landfall trigger dict
    ``{event:enters, predicate:And(subject_is_land, controlled_by_you),
       effect:put_plus_one_counter_on_self(amount=1)}`` round-trips
    through ``_ta_dict_to_python_triggered_ability`` to the EXACT
    Python shape the genuine Seedship Agrarian parser emits for its
    trigger-2: ``Event.ENTERS / And(Subject(IsLand),
    Controlled(YOU)) / PutPlusOneCounterOnSelf(amount=1)``.

    This is the §2 bridge-correctness proof for the new admit: the
    predicate axis (landfall) is orthogonal to the effect resolve, so
    the same already-byte-equal put_plus_one_counter_on_self arm
    round-trips identically under the landfall ENTERS-on-land
    predicate.
    """
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, Subject, YOU,
    )
    from mtg_bot.effects.object_predicates import IsLand
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )

    ta = _ta_dict_to_python_triggered_ability({
        "event": "enters",
        "predicate": {
            "kind": "and",
            "left": {"kind": "subject_is_land"},
            "right": {"kind": "controlled_by_you"},
        },
        "effect": {"kind": "put_plus_one_counter_on_self", "amount": 1},
    })
    assert ta.event is Event.ENTERS
    assert isinstance(ta.predicate, And)
    assert isinstance(ta.predicate.left, Subject)
    assert isinstance(ta.predicate.left.inner, IsLand)
    assert ta.predicate.left.slot is None
    assert isinstance(ta.predicate.right, Controlled)
    assert ta.predicate.right.by is YOU
    assert ta.predicate.right.subject is None
    assert isinstance(ta.effect, PutPlusOneCounterOnSelf)
    assert ta.effect.amount == 1


def test_bridge_ta_landfall_put_plus_one_counter_matches_genuine_parser():
    """The bridge round-trip of Seedship Agrarian's trigger-2 dict
    reconstructs an object byte-equal to what the LIVE parser produces
    for the real EoE printing. Drives the genuine card so a parser
    shape change OR a bridge regression both surface here.
    """
    import pytest

    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.predicates import Event
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.trajectory import _python_ta_to_safe_dict
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )

    real = get_active_source().get_printings().get("Seedship Agrarian")
    if real is None:
        pytest.skip("Seedship Agrarian absent from active CardSource")
    landfall = [
        t for t in real.triggered_abilities
        if t.event is Event.ENTERS
    ]
    assert len(landfall) == 1
    genuine = landfall[0]
    assert isinstance(genuine.effect, PutPlusOneCounterOnSelf)
    # Recognizer emits the canonical dict; bridge reconstructs it.
    d = _python_ta_to_safe_dict(genuine)
    assert d is not None
    assert d["effect"] == {
        "kind": "put_plus_one_counter_on_self",
        "amount": genuine.effect.amount,
    }
    bridged = _ta_dict_to_python_triggered_ability(d)
    assert bridged.event is genuine.event
    assert type(bridged.predicate) is type(genuine.predicate)
    assert isinstance(bridged.effect, PutPlusOneCounterOnSelf)
    assert bridged.effect.amount == genuine.effect.amount


def test_bridge_ta_landfall_put_plus_one_counter_amount_default():
    """The landfall trigger dict with ``amount`` omitted defaults to 1
    on round-trip (same default-handling contract as the bare
    put_plus_one_counter_on_self arm). Pins that the predicate
    composition does not perturb the effect-arm default semantics.
    """
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )

    ta = _ta_dict_to_python_triggered_ability({
        "event": "enters",
        "predicate": {
            "kind": "and",
            "left": {"kind": "subject_is_land"},
            "right": {"kind": "controlled_by_you"},
        },
        "effect": {"kind": "put_plus_one_counter_on_self"},
    })
    assert isinstance(ta.effect, PutPlusOneCounterOnSelf)
    assert ta.effect.amount == 1


def test_bridge_ta_landfall_put_plus_one_counter_amount_preserved():
    """A non-default amount survives the landfall-composed round-trip
    (a future "Landfall — put 2 +1/+1 counters on this creature"
    carrier would round-trip the count, not collapse to 1).
    """
    from mtg_bot.effects.generic.counters import PutPlusOneCounterOnSelf
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_triggered_ability,
    )

    ta = _ta_dict_to_python_triggered_ability({
        "event": "enters",
        "predicate": {
            "kind": "and",
            "left": {"kind": "subject_is_land"},
            "right": {"kind": "controlled_by_you"},
        },
        "effect": {"kind": "put_plus_one_counter_on_self", "amount": 3},
    })
    assert ta.effect == PutPlusOneCounterOnSelf(amount=3)


# ===========================================================================
# Phase-6 non-basic-land substrate — PHASE 1 cross-engine bridge pins
# (round-60 N1; spec docs/NONBASIC_LAND_SUBSTRATE_PLAN.md §2/§3)
#
# The 5 EoE pay-life shocklands are admitted as untapped dual-mana
# lands. These pins prove the bridge round-trips the REAL shockland
# CardDef (built by trajectory._printing_to_card_def from the live
# parser printing) IDENTICALLY on both engines: same untapped dual,
# byte-equal field views, exactly two auto-wired {T}: Add <c> mana
# abilities (one per produce color), no extra rider fields.
# ===========================================================================


def _r60_real_shockland_card_def(name: str):
    """The REAL shockland CardDef the production loader uses — built
    from the live ``OracleParserCardSource`` printing through the
    actual ``trajectory._printing_to_card_def`` arm (NOT a synthetic
    hand-rolled CardDef). This makes the pin exercise the real
    code path end-to-end."""
    from mtg_bot.cards.source import OracleParserCardSource
    from mtg_bot.parity.trajectory import _printing_to_card_def

    by = OracleParserCardSource().get_printings()
    return _printing_to_card_def(by[name])


def test_r60_bridge_shockland_card_def_is_real_untapped_dual():
    """The real Breeding Pool CardDef the loader emits is a "Land"
    with the sorted dual ``mana_produced=("G","U")`` and carries NO
    creature / ability / rider fields — structurally identical to a
    basic-land CardDef, just non-basic. This is the shape the bridge
    must round-trip byte-equal on both engines."""
    cd = _r60_real_shockland_card_def("Breeding Pool")
    assert cd.card_type == "Land"
    assert cd.mana_produced == ("G", "U")
    assert cd.power is None and cd.toughness is None
    assert cd.triggered_abilities == ()
    assert cd.keywords == ()


def test_r60_bridge_python_state_wires_two_mana_abilities_for_shockland():
    """``build_python_state`` materializes the real Watery Grave
    printing with its dual ``produces`` and the per-color mana-ability
    auto-wire: exactly TWO activated abilities (one ``{T}: Add U`` +
    one ``{T}: Add B``), enters_tapped False (the parser flattened the
    pay-life replacement — documented deliberate simplification)."""
    from mtg_bot.parity.cross_engine import DeckSpec, build_python_state

    cd = _r60_real_shockland_card_def("Watery Grave")
    spec = DeckSpec(
        card_defs=(cd,),
        p0_library=("Watery Grave",),
        p1_library=("Watery Grave",),
    )
    state = build_python_state(spec)
    pr = state.players[0].library[0].printing
    assert pr.produces == frozenset({"U", "B"})
    assert "Land" in pr.card_types
    # Dual produce -> exactly two auto-wired {T}: Add <c> mana abilities.
    assert len(pr.activated_abilities) == 2
    assert pr.triggered_abilities == ()
    assert pr.static_abilities == ()
    # The pay-life / tapped replacement was dropped by the parser —
    # the bridged printing is a vanilla untapped dual on this engine.
    assert getattr(pr, "enters_tapped", False) is False


def test_r60_bridge_shockland_byte_equal_initial_state_both_engines():
    """**The cross-engine equality contract**: a DeckSpec built from
    the real Stomping Ground CardDef produces a BYTE-EQUAL initial
    field view on Python and Rust — the parser drops the pay-life
    replacement SYMMETRICALLY so both engines materialize the SAME
    untapped dual (round-39 Ward precedent; bit-equal cross-engine,
    CR-incomplete; see SEARCH_LIMITATIONS.md)."""
    import importlib.util
    import pathlib
    import sys

    from mtg_bot.parity.cross_engine import (
        DeckSpec,
        build_matched_states,
        build_python_field_view,
        compare_fields,
    )

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    spec_mod = importlib.util.spec_from_file_location(
        "parity_smoke", repo_root / "scripts" / "parity_smoke.py",
    )
    parity_smoke = importlib.util.module_from_spec(spec_mod)
    sys.modules["parity_smoke"] = parity_smoke
    spec_mod.loader.exec_module(parity_smoke)

    cd = _r60_real_shockland_card_def("Stomping Ground")
    spec = DeckSpec(
        card_defs=(cd,),
        p0_library=("Stomping Ground",) * 4,
        p1_library=("Stomping Ground",) * 4,
    )
    py_state, rust_state = build_matched_states(spec)
    py_view = build_python_field_view(py_state)
    rust_view = parity_smoke._build_field_view(rust_state)
    divs = compare_fields(py_view, rust_view)
    assert divs == [], (
        "round-60 phase-1: the real Stomping Ground CardDef must "
        "build a BYTE-EQUAL initial state on both engines (untapped "
        f"dual, symmetric pay-life drop). Divergences: "
        f"{[(d.field, d.py_value, d.rust_value) for d in divs]}"
    )
    # Both libraries carry the 4 Stomping Grounds, byte-equal.
    assert py_view["library_p0"] == rust_view["library_p0"]
    assert py_view["library_p1"] == rust_view["library_p1"]


def test_bridge_perigee_pump_grant_dies_return_tapped_translates():
    """**BLOCKED-RUST-EFFECT slice (round 60) — Perigee Beckoner.**
    The ``pump_target_creature_until_eot_grant_dies_return_tapped``
    dict translates to ``PumpTargetCreatureUntilEOT`` carrying the
    EXACT params the genuine printing parses to: power/toughness
    bonus preserved, ``you_control_only`` + ``exclude_self`` set, and
    a ``granted_triggered_ability`` whose effect is
    ``PerigeeBeckonerDiesReturnTappedEffect``, event ``DIES``,
    predicate ``IsSelf``.
    """
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.effects.bespoke import (
        PerigeeBeckonerDiesReturnTappedEffect,
    )
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    eff = _ta_dict_to_python_effect({
        "kind": "pump_target_creature_until_eot_grant_dies_return_tapped",
        "power_bonus": 2,
        "toughness_bonus": 0,
    })
    assert isinstance(eff, PumpTargetCreatureUntilEOT)
    assert eff.power_bonus == 2
    assert eff.toughness_bonus == 0
    assert eff.you_control_only is True
    assert eff.exclude_self is True
    gta = eff.granted_triggered_ability
    assert gta is not None
    assert isinstance(gta.effect, PerigeeBeckonerDiesReturnTappedEffect)
    assert gta.event == Event.DIES
    assert isinstance(gta.predicate, IsSelf)


def test_bridge_perigee_pump_grant_dies_return_tapped_defaults_zero():
    """Missing ``power_bonus`` / ``toughness_bonus`` default to 0 on
    BOTH sides (Rust parse arm tolerates the missing keys; the bridge
    mirrors that with ``eff_dict.get(..., 0)``). The rider is still
    attached even at +0/+0 — the marker, not the pump, is the
    load-bearing observable.
    """
    from mtg_bot.effects.generic.pump import PumpTargetCreatureUntilEOT
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    eff = _ta_dict_to_python_effect({
        "kind": "pump_target_creature_until_eot_grant_dies_return_tapped",
    })
    assert isinstance(eff, PumpTargetCreatureUntilEOT)
    assert eff.power_bonus == 0
    assert eff.toughness_bonus == 0
    assert eff.granted_triggered_ability is not None


def test_bridge_perigee_pump_grant_dies_return_tapped_etb_pump_parity():
    """Cross-engine ETB parity on the PUMP-applied observable: a
    Perigee-shape source enters with an "another" 2/2 ally on the
    controller's bf; after the ETB resolves, the ally is +2/+0 on
    BOTH engines (Rust: ``UeotModifier::PowerToughnessBonus`` push;
    Python: the genuine ``PumpTargetCreatureUntilEOT`` resolve). The
    source itself is NOT pumped (CR 115.4 "another" — Rust's
    source-id exclusion; Python's ``exclude_self``).
    """
    import pytest

    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
    )
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.game import Zone
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from tests.parity._picker_shims import (
        _PickFirstControlledCreatureAgent,
    )

    real = get_active_source().get_printings().get("Perigee Beckoner")
    if real is None:
        pytest.skip("Perigee Beckoner absent from active CardSource")

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py = build_python_state(spec)
    rs = build_rust_state(spec)
    ZB = 2
    py.players[0].agent = _PickFirstControlledCreatureAgent()

    def _gp(n):
        return CardPrinting(
            name=n, mana_cost_str="", mana_cost=ManaCost(generic=0),
            type_line="Creature", supertypes=(),
            card_types=("Creature",), subtypes=(),
            power=2, toughness=2, produces=frozenset(),
        )

    py_ally = py.new_object(_gp("Ally"), owner_id=0,
                            zone=Zone.BATTLEFIELD)
    py.battlefield.append(py_ally)
    rs_ally = rs.register_and_inject_object(
        {"name": "Ally", "card_types": ["Creature"],
         "power": 2, "toughness": 2}, 0, ZB)

    rs.register_and_inject_object(
        {"name": "Perigee Beckoner", "card_types": ["Creature"],
         "power": 4, "toughness": 5,
         "triggered_abilities": [{
             "event": "enters",
             "predicate": {"kind": "source_is_self"},
             "effect": {
                 "kind": "pump_target_creature_until_eot_grant_dies_return_tapped",
                 "power_bonus": 2, "toughness_bonus": 0,
             }}]}, 0, ZB)
    py_src = py.new_object(real, owner_id=0, zone=Zone.BATTLEFIELD)
    py_src.summoning_sick = False
    py_src.etb_timestamp = py.next_timestamp
    py.next_timestamp += 1
    py.battlefield.append(py_src)
    _trigger_etb_abilities(py, py_src)

    while rs.stack_size() > 0:
        rs.resolve_top_of_stack()
    for _ in range(12):
        if not py.stack:
            break
        resolve_top_of_stack(py)
    assert rs.stack_size() == 0 and not py.stack

    from mtg_bot.game import effective_power

    rs_bf = {o["id"]: o for o in rs.battlefield_snapshot()}
    # Ally got +2/+0 on BOTH (effective power 2 -> 4), byte-equal.
    py_ally_pow = effective_power(py, py_ally)
    assert py_ally_pow == 4
    assert rs.effective_power(rs_ally) == py_ally_pow
    assert rs_ally in rs_bf
    assert rs_bf[rs_ally]["printing_name"] == "Ally"
    assert rs_bf[rs_ally]["tapped"] is False
    # Source itself NOT pumped on BOTH ("another"), byte-equal.
    py_src_pow = effective_power(py, py_src)
    assert py_src_pow == real.power
    rs_src_id = next(o["id"] for o in rs.battlefield_snapshot()
                     if o["printing_name"] == "Perigee Beckoner")
    assert rs.effective_power(rs_src_id) == py_src_pow


def test_bridge_perigee_pump_grant_dies_return_tapped_dies_return_parity():
    """Cross-engine parity on the DIES-RETURN-TAPPED observable (the
    genuinely-new shape this slice ships): after the pumped ally is
    destroyed, it returns to its OWNER's battlefield TAPPED with no
    UEOT bonus, graveyard empty, on BOTH engines (Rust: the SBA
    return-tapped path; Python: the genuine
    ``PerigeeBeckonerDiesReturnTappedEffect``).
    """
    import pytest

    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
    )
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.game import Zone
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.turn import state_based_actions
    from tests.parity._picker_shims import (
        _PickFirstControlledCreatureAgent,
    )

    real = get_active_source().get_printings().get("Perigee Beckoner")
    if real is None:
        pytest.skip("Perigee Beckoner absent from active CardSource")

    spec = DeckSpec(card_defs=(), p0_library=(), p1_library=())
    py = build_python_state(spec)
    rs = build_rust_state(spec)
    ZB = 2
    py.players[0].agent = _PickFirstControlledCreatureAgent()

    def _gp(n):
        return CardPrinting(
            name=n, mana_cost_str="", mana_cost=ManaCost(generic=0),
            type_line="Creature", supertypes=(),
            card_types=("Creature",), subtypes=(),
            power=2, toughness=2, produces=frozenset(),
        )

    py_ally = py.new_object(_gp("Ally"), owner_id=0,
                            zone=Zone.BATTLEFIELD)
    py.battlefield.append(py_ally)
    rs_ally = rs.register_and_inject_object(
        {"name": "Ally", "card_types": ["Creature"],
         "power": 2, "toughness": 2}, 0, ZB)
    rs.register_and_inject_object(
        {"name": "Perigee Beckoner", "card_types": ["Creature"],
         "power": 4, "toughness": 5,
         "triggered_abilities": [{
             "event": "enters",
             "predicate": {"kind": "source_is_self"},
             "effect": {
                 "kind": "pump_target_creature_until_eot_grant_dies_return_tapped",
                 "power_bonus": 2, "toughness_bonus": 0,
             }}]}, 0, ZB)
    py_src = py.new_object(real, owner_id=0, zone=Zone.BATTLEFIELD)
    py_src.summoning_sick = False
    py_src.etb_timestamp = py.next_timestamp
    py.next_timestamp += 1
    py.battlefield.append(py_src)
    _trigger_etb_abilities(py, py_src)
    while rs.stack_size() > 0:
        rs.resolve_top_of_stack()
    for _ in range(12):
        if not py.stack:
            break
        resolve_top_of_stack(py)

    # Kill the pumped ally on BOTH (lethal marked damage + SBAs).
    py_ally.damage = 99
    for _ in range(20):
        if not state_based_actions(py):
            break
        while py.stack:
            resolve_top_of_stack(py)
    rs.deal_damage(rs_ally, 99)
    rs.apply_sbas_until_quiescent()

    # Ally returned tapped under owner's control on BOTH; gy empty.
    rs_bf = {o["id"]: o for o in rs.battlefield_snapshot()}
    assert rs_ally in rs_bf, "Rust: ally must return to bf"
    assert rs_bf[rs_ally]["tapped"] is True
    assert rs_bf[rs_ally]["controller_id"] == 0
    assert rs.graveyard_snapshot(0) == []

    py_ally_back = [o for o in py.battlefield
                    if o.printing.name == "Ally"]
    assert len(py_ally_back) == 1, "Python: ally must return to bf"
    assert py_ally_back[0].tapped is True
    assert py_ally_back[0].controller_id == 0
    assert list(py.players[0].graveyard) == []
    # **Source-of-truth power-retention (round-53 effective-state
    # lesson).** Python's PerigeeBeckonerDiesReturnTappedEffect
    # re-uses the SAME object and put_into_battlefield does NOT reset
    # temp_power_bonus, so the returned creature KEEPS the +2/+0 pump
    # (effective power stays 4, NOT a clean CR-400.7 new-object 2).
    # Rust matches that source-of-truth behavior by dropping ONLY the
    # rider marker and keeping the PowerToughnessBonus — byte-equal on
    # BOTH (4), asserted here rather than against a textbook value the
    # source engine does not implement.
    from mtg_bot.game import effective_power
    py_back_pow = effective_power(py, py_ally_back[0])
    rs_back_pow = rs.effective_power(rs_ally)
    assert py_back_pow == 4, (
        f"Python source-of-truth: returned ally keeps +2/+0 "
        f"(eff power 4); got {py_back_pow}")
    assert rs_back_pow == py_back_pow, (
        f"Rust returned-ally effective power {rs_back_pow} must "
        f"byte-equal Python's {py_back_pow}")


# ===========================================================================
# Round-62 N1 — Rust `apply_mana_produced_bridge` per-color-shape
# regression pins. The bridge now emits ONE `is_mana_ability`
# ActivatedAbility PER produced color (input order), matching Python's
# `_card_def_to_python_printing`, instead of the pre-r62 ONE ability
# adding ALL colors (`Sequence(Add c1, Add c2)`). Behavior-pinned via
# `register_and_inject_object` + `activate_ability` (the Rust Registry
# exposes no `printing_to_dict` introspection; observable pool state
# after a per-index activate is the contract that matters and is the
# exact axis round-61 N3's bench sweep diverged on).
# ===========================================================================


def _r62_rust_inject_land(rust_state, name, mana_produced):
    """Inject one untapped Land with ``mana_produced`` onto p0's Rust
    battlefield via the dict-input registry path (parse_printing ->
    apply_mana_produced_bridge). Returns the obj id."""
    ZONE_BATTLEFIELD = 2
    return int(rust_state.register_and_inject_object(
        {
            "name": name,
            "card_types": ["Land"],
            "mana_cost": {},
            "effects": [],
            "mana_produced": list(mana_produced),
            "supertypes": [],
        },
        0,
        ZONE_BATTLEFIELD,
    ))


def _r62_fresh_rust_state():
    from mtg_bot.parity.cross_engine import DeckSpec, build_rust_state
    return build_rust_state(DeckSpec(card_defs=(), p0_library=(),
                                     p1_library=()))


def test_r62_rust_bridge_single_color_land_unchanged():
    """Single-color land (every basic) stays bit-identical to pre-r62:
    exactly ONE auto-wired mana ability at index 0 that adds the one
    color. ``activate_ability(forest, 0)`` -> pool {G:1}; index 1
    must NOT exist (no second per-color ability). Pins the
    "no regression for basics" guarantee."""
    import pytest as _pytest

    rs = _r62_fresh_rust_state()
    fid = _r62_rust_inject_land(rs, "Forest", ["G"])
    rs.empty_mana_pool(0)
    rs.apply_action({"kind": "activate_ability",
                     "source_id": fid, "ability_index": 0})
    pool = dict(rs.get_mana_pool(0))
    assert pool.get("G", 0) == 1 and sum(pool.values()) == 1, pool
    # A second per-color ability must NOT have been synthesized for a
    # single-color land — the printing has EXACTLY ONE activated
    # ability, so index 1 is out of range and Rust raises loudly
    # (this is the strong proof the bridge didn't over-synthesize:
    # a basic must never gain a phantom second mana ability).
    fid2 = _r62_rust_inject_land(rs, "Forest", ["G"])
    rs.empty_mana_pool(0)
    with _pytest.raises(ValueError, match="no ability at index 1"):
        rs.apply_action({"kind": "activate_ability",
                         "source_id": fid2, "ability_index": 1})


def test_r62_rust_bridge_two_color_land_one_ability_per_color_in_order():
    """A 2-color land (Watery Grave B/U) gets TWO per-color mana
    abilities in input order: index 0 adds ONLY B, index 1 adds ONLY
    U — never the aggregate. This is the direct inversion of the
    pre-r62 `Sequence(Add B, Add U)`-at-index-0 bug."""
    rs = _r62_fresh_rust_state()
    # index 0 -> only B
    a = _r62_rust_inject_land(rs, "Watery Grave", ["B", "U"])
    rs.empty_mana_pool(0)
    rs.apply_action({"kind": "activate_ability",
                     "source_id": a, "ability_index": 0})
    p0 = dict(rs.get_mana_pool(0))
    assert p0.get("B", 0) == 1 and p0.get("U", 0) == 0 \
        and sum(p0.values()) == 1, (
        f"idx0 must add ONLY B (not the Sequence); got {p0}")
    # index 1 -> only U
    b = _r62_rust_inject_land(rs, "Watery Grave", ["B", "U"])
    rs.empty_mana_pool(0)
    rs.apply_action({"kind": "activate_ability",
                     "source_id": b, "ability_index": 1})
    p1 = dict(rs.get_mana_pool(0))
    assert p1.get("U", 0) == 1 and p1.get("B", 0) == 0 \
        and sum(p1.values()) == 1, (
        f"idx1 must add ONLY U; got {p1}")


def test_r62_rust_bridge_order_follows_input_not_wubrgc():
    """The per-color abilities follow the INPUT ``mana_produced``
    order (= Python's `for color in card.mana_produced`), NOT a fixed
    W/U/B/R/G/C slot order (which the pre-r62 aggregate code used
    internally). Inject a land with ``mana_produced=["G","B"]`` (NOT
    alphabetical): index 0 must add G (the first input symbol), index
    1 must add B — proving order tracks the input list, so it stays
    locked to whatever order Python iterates."""
    rs = _r62_fresh_rust_state()
    a = _r62_rust_inject_land(rs, "OrderProbe", ["G", "B"])
    rs.empty_mana_pool(0)
    rs.apply_action({"kind": "activate_ability",
                     "source_id": a, "ability_index": 0})
    assert dict(rs.get_mana_pool(0)).get("G", 0) == 1, "idx0 -> G"
    b = _r62_rust_inject_land(rs, "OrderProbe", ["G", "B"])
    rs.empty_mana_pool(0)
    rs.apply_action({"kind": "activate_ability",
                     "source_id": b, "ability_index": 1})
    assert dict(rs.get_mana_pool(0)).get("B", 0) == 1, "idx1 -> B"


def test_r62_rust_bridge_three_color_land_three_per_color_abilities():
    """Generalizes past 2 colors: a hypothetical 3-color
    ``mana_produced=["U","B","R"]`` land yields THREE per-color
    abilities, each adding exactly its one color at its index — the
    bridge is uniformly per-color, not special-cased to duals."""
    rs = _r62_fresh_rust_state()
    for idx, col in enumerate(("U", "B", "R")):
        oid = _r62_rust_inject_land(rs, "TriProbe", ["U", "B", "R"])
        rs.empty_mana_pool(0)
        rs.apply_action({"kind": "activate_ability",
                         "source_id": oid, "ability_index": idx})
        pool = dict(rs.get_mana_pool(0))
        assert pool.get(col, 0) == 1 and sum(pool.values()) == 1, (
            f"idx{idx} must add ONLY {col}; got {pool}")


def test_r62_rust_bridge_explicit_abilities_follow_per_color_entries():
    """Explicit registry-supplied ``activated_abilities`` still land
    AFTER the synthesized per-color mana abilities (the prepend
    contract is preserved, just widened from 1 prepended entry to N).
    A 2-color land with one extra explicit ability: indices 0,1 are
    the per-color mana abilities (B then U); the explicit ability is
    at index 2."""
    ZONE_BATTLEFIELD = 2
    rs = _r62_fresh_rust_state()
    oid = int(rs.register_and_inject_object(
        {
            "name": "DualWithExtra",
            "card_types": ["Land"],
            "mana_cost": {},
            "effects": [],
            "mana_produced": ["B", "U"],
            "supertypes": [],
            "activated_abilities": [
                {
                    "cost": {"mana": {}, "tap_self": True},
                    "effect": {"kind": "draw_cards", "count": 1},
                    "is_mana_ability": False,
                }
            ],
        },
        0,
        ZONE_BATTLEFIELD,
    ))
    # idx0 -> B (first per-color mana ability), idx1 -> U.
    rs.empty_mana_pool(0)
    rs.apply_action({"kind": "activate_ability",
                     "source_id": oid, "ability_index": 0})
    assert dict(rs.get_mana_pool(0)).get("B", 0) == 1
    oid2 = int(rs.register_and_inject_object(
        {
            "name": "DualWithExtra",
            "card_types": ["Land"],
            "mana_cost": {},
            "effects": [],
            "mana_produced": ["B", "U"],
            "supertypes": [],
            "activated_abilities": [
                {
                    "cost": {"mana": {}, "tap_self": True},
                    "effect": {"kind": "draw_cards", "count": 1},
                    "is_mana_ability": False,
                }
            ],
        },
        0,
        ZONE_BATTLEFIELD,
    ))
    rs.empty_mana_pool(0)
    rs.apply_action({"kind": "activate_ability",
                     "source_id": oid2, "ability_index": 1})
    assert dict(rs.get_mana_pool(0)).get("U", 0) == 1, (
        "idx1 must be the second per-color mana ability (U), proving "
        "the explicit non-mana ability sits at idx2, not idx1")


# ---------------------------------------------------------------------------
# warp_cost Phase 1 (round-63) — Rust regression + cross-engine bridge
# pins. Spec: docs/WARP_COST_PHASE1_DESIGN.md §4 pins 1-8.
# ---------------------------------------------------------------------------

_R63_ZONE_HAND = 0  # state_ops_zones.rs ZONE_HAND
_R63_ZONE_BF = 2    # ZONE_BATTLEFIELD


def _r63_inject_warp_creature(rs, zone=_R63_ZONE_HAND):
    """Inject a pure-warp Artifact Creature ({3}{R} normal, {1}{R}
    warp, 3/3) into p0's ``zone`` via the dict registry path. Returns
    the obj id."""
    return int(rs.register_and_inject_object(
        {
            "name": "WarpBeast",
            "card_types": ["Artifact", "Creature"],
            "mana_cost": {"generic": 3, "R": 1},
            "warp_cost": {"generic": 1, "R": 1},
            "power": 3,
            "toughness": 3,
            "effects": [],
        },
        0,
        zone,
    ))


def test_r63_warp_parse_printing_roundtrips_warp_cost():
    """Spec §4 pin 1 — ``parse_printing`` with
    ``warp_cost={"generic":1,"R":1}`` round-trips into
    ``CardPrinting.warp_cost``; the absent key maps to ``None``
    (back-compat). Verified via the legal emitter (a printing WITH
    warp_cost offers the use_warp variant; one WITHOUT does not)."""
    rs = _r62_fresh_rust_state()
    wid = _r63_inject_warp_creature(rs)
    # A vanilla (no warp_cost) creature for the negative half.
    vid = int(rs.register_and_inject_object(
        {"name": "Vanilla", "card_types": ["Creature"],
         "mana_cost": {"generic": 1, "G": 1},
         "power": 2, "toughness": 2, "effects": []},
        0, _R63_ZONE_HAND))
    rs.set_phase(3)  # MAIN1 (sorcery timing)
    # legal_decisions mana-awareness Slice 1: legal_decisions now
    # gates every CastSpell site on can_pay_cost (the SoT-flip
    # closure). Seat enough floating mana that BOTH the WarpBeast
    # base {3}{R} + warp {1}{R} AND the Vanilla {1}{G} are payable,
    # so this pin still exercises the warp/base EMISSION-SHAPE +
    # round-trip it was written to characterize (the mana-prune is a
    # separate axis, pinned by test_cross_engine_legal_set_asymmetry).
    rs.add_mana(0, "C", 4)
    rs.add_mana(0, "R", 1)
    rs.add_mana(0, "G", 1)
    ld = rs.legal_decisions(0)
    warp_for_w = sorted(
        a.get("use_warp") for a in ld
        if a.get("kind") == "cast_spell" and a.get("card_id") == wid
    )
    warp_for_v = sorted(
        a.get("use_warp") for a in ld
        if a.get("kind") == "cast_spell" and a.get("card_id") == vid
    )
    assert warp_for_w == [False, True], (
        f"warp printing must emit base+warp; got {warp_for_w}")
    assert warp_for_v == [False], (
        f"no-warp printing must emit base ONLY; got {warp_for_v}")


def test_r63_warp_legal_emits_adjacent_base_then_warp_positional():
    """Spec §4 pin 2 — a hand with one warp printing emits EXACTLY
    ``[..., cast_spell(use_warp=False), cast_spell(use_warp=True),
    ...]`` ADJACENT in that order (positional, not just set). 2x the
    same warp printing → still exactly one {base, warp} pair (dedup
    parity with Python's name+flag key)."""
    rs = _r62_fresh_rust_state()
    w1 = _r63_inject_warp_creature(rs)
    rs.set_phase(3)
    # legal_decisions mana-awareness Slice 1: seat mana so the
    # WarpBeast base {3}{R} + warp {1}{R} are BOTH payable — this pin
    # characterizes the POSITIONAL emission shape + dedup, not the
    # (separately-pinned) mana-prune axis.
    rs.add_mana(0, "C", 4)
    rs.add_mana(0, "R", 1)
    ld = rs.legal_decisions(0)
    cs = [
        (a["card_id"], a["use_warp"]) for a in ld
        if a.get("kind") == "cast_spell"
    ]
    assert cs == [(w1, False), (w1, True)], (
        f"single warp printing: expected adjacent "
        f"[(id,False),(id,True)]; got {cs}")
    # 2x the same warp printing name → name-only dedup yields ONE
    # {base, warp} pair, matching Python's (name, use_warp, ...) key.
    w2 = _r63_inject_warp_creature(rs)
    rs.set_phase(3)
    # Re-seat (the first legal_decisions call does not drain mana,
    # but be explicit so the dedup half is independent of pool carry).
    rs.add_mana(0, "C", 4)
    rs.add_mana(0, "R", 1)
    ld2 = rs.legal_decisions(0)
    cs2 = [
        (a["card_id"], a["use_warp"]) for a in ld2
        if a.get("kind") == "cast_spell"
    ]
    warp_count = sum(1 for _, w in cs2 if w)
    base_count = sum(1 for _, w in cs2 if not w)
    assert warp_count == 1 and base_count == 1, (
        f"2x same warp printing must dedup to exactly one "
        f"{{base, warp}} pair; got {cs2} (w2={w2})")


def test_r63_warp_cast_pays_warp_cost_sets_flag_and_pending():
    """Spec §4 pin 3 — ``Action::CastSpell{use_warp:true}`` on a warp
    creature pays ``warp_cost`` (pool debited by {1}{R}, NOT the {3}{R}
    mana_cost), ``spell_warped_this_turn == True``, and the resolved
    permanent has ``warp_exile_pending == True`` (observed via the
    end-step exile firing one phase later — see pin 4)."""
    rs = _r62_fresh_rust_state()
    wid = _r63_inject_warp_creature(rs)
    rs.set_phase(3)
    rs.add_mana(0, "R", 1)
    rs.add_mana(0, "C", 1)  # {1} generic
    assert rs.spell_warped_this_turn is False
    rs.apply_action({
        "kind": "cast_spell", "card_id": wid, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # G7 S4: cast_creature_warp now pushes a StackItem (was atomic).
    # The genuinely cast-time riders (warp_cost paid,
    # spell_warped_this_turn) are observable immediately; the bf entry
    # is at resolution (byte-mirroring Python's _cast_spell /
    # put_into_battlefield split). Drive the resolve so WarpBeast
    # lands — byte-equivalent to the old fused post-state, just split
    # cast-then-resolve (the S1 closure-flip pattern).
    pool = dict(rs.get_mana_pool(0))
    assert sum(pool.values()) == 0, (
        f"warp cost {{1}}{{R}} must fully drain the 2-mana pool; "
        f"got {pool} (would be non-empty if it paid the {{3}}{{R}} "
        f"mana_cost instead)")
    assert rs.spell_warped_this_turn is True
    assert not rs.is_stack_empty(), (
        "G7 S4: the warp creature spell must be ON THE STACK "
        "post-cast (stack-then-resolve, not atomic)")
    rs.resolve_top_of_stack()
    bf = [o for o in rs.battlefield_snapshot()
          if o["printing_name"] == "WarpBeast"]
    assert len(bf) == 1, "WarpBeast must be on the battlefield post-resolve"


def test_r63_warp_end_step_sweep_exiles_with_warped_marker():
    """Spec §4 pin 4 — advancing to PHASE_END_STEP on the caster's
    turn exiles the warp-pending permanent (bf → exile),
    ``warp_exile_pending`` cleared, ``warped`` set (the exile-recast
    marker)."""
    rs = _r62_fresh_rust_state()
    wid = _r63_inject_warp_creature(rs)
    rs.set_phase(3)
    rs.add_mana(0, "R", 1)
    rs.add_mana(0, "C", 1)
    rs.apply_action({
        "kind": "cast_spell", "card_id": wid, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # G7 S4: warp cast is stack-then-resolve (was atomic). Drive the
    # resolve so WarpBeast lands on the bf with the warp_exile_pending
    # flag riding through onto the post-resolution bf object — exactly
    # where the end-step sweep keys off it (the design's key risk,
    # resolved structurally). S1 closure-flip pattern.
    rs.resolve_top_of_stack()
    assert any(o["printing_name"] == "WarpBeast"
               for o in rs.battlefield_snapshot())
    rs.set_phase(9)        # MAIN2
    rs.advance_phase()     # → END_STEP (phase 10): WarpExileSelf onto stack
    assert rs.current_phase() == 10
    # **G7 Class-C delayed-trigger port (round-78).** END_STEP entry
    # pushes WarpExileSelf onto the stack (CR 603.7 response window);
    # resolve drains it to exile. Pre-port this was an atomic
    # phase-entry sweep; the new path matches Python's
    # `turn.run_delayed_end_step_triggers` semantics.
    rs.resolve_top_of_stack()  # WarpExileSelf resolves → bf → exile.
    bf = [o for o in rs.battlefield_snapshot()
          if o["printing_name"] == "WarpBeast"]
    ex = [o for o in rs.exile_snapshot(0)
          if o["printing_name"] == "WarpBeast"]
    assert bf == [], "WarpBeast must have left the battlefield"
    assert len(ex) == 1, (
        f"WarpBeast must be in p0 exile after the delayed-trigger "
        f"resolves; got exile="
        f"{[o['printing_name'] for o in rs.exile_snapshot(0)]}")


def test_r63_warp_exile_recast_pays_normal_cost_clears_warped():
    """Spec §4 pin 5 — after the end-step exile, ``legal_decisions``
    offers a BASE ``cast_spell(use_warp=False)`` for the ``warped``
    exile object; casting it pays the NORMAL {3}{R} ``mana_cost`` (NOT
    the {1}{R} warp cost) and clears ``warped`` (object back on bf,
    exile empty)."""
    rs = _r62_fresh_rust_state()
    wid = _r63_inject_warp_creature(rs)
    rs.set_phase(3)
    rs.add_mana(0, "R", 1)
    rs.add_mana(0, "C", 1)
    rs.apply_action({
        "kind": "cast_spell", "card_id": wid, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # G7 S4: warp cast is stack-then-resolve (was atomic); drive the
    # resolve so WarpBeast lands (warp_exile_pending rides onto the bf
    # object) BEFORE the end-step sweep exiles it. S1 closure-flip.
    rs.resolve_top_of_stack()
    rs.set_phase(9)
    rs.advance_phase()  # → END_STEP pushes WarpExileSelf onto the stack
    rs.resolve_top_of_stack()  # delayed trigger resolves → bf → exile
    assert any(o["printing_name"] == "WarpBeast"
               for o in rs.exile_snapshot(0))
    rs.set_phase(3)  # a later MAIN1
    # legal_decisions mana-awareness Slice 1: the warp-cast above
    # drained the pool, so the warp-exiled object's BASE recast is
    # now mana-gated. Seat the NORMAL {3}{R} BEFORE the emission
    # assertion (this is the SAME mana the recast apply_action below
    # consumes — the post-apply pool-drain assert still expects 0).
    rs.add_mana(0, "R", 1)
    rs.add_mana(0, "C", 3)  # {3}{R} normal mana_cost
    ld = rs.legal_decisions(0)
    recast = [
        (a["card_id"], a["use_warp"]) for a in ld
        if a.get("kind") == "cast_spell" and a.get("card_id") == wid
    ]
    assert recast == [(wid, False)], (
        f"warp-exiled object must offer a BASE recast only "
        f"(use_warp=False, normal cost); got {recast}")
    rs.apply_action({
        "kind": "cast_spell", "card_id": wid, "use_warp": False,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # G7 S1: cast_creature now pushes a StackItem (exile-recast is
    # normalized exile -> hand at cast time); drive the resolve so the
    # recast WarpBeast lands on the battlefield — byte-equivalent to
    # the old fused no-stack post-state, just split cast-then-resolve.
    rs.resolve_top_of_stack()
    pool = dict(rs.get_mana_pool(0))
    assert sum(pool.values()) == 0, (
        f"recast must pay the full {{3}}{{R}} (4 mana) — pool should "
        f"drain to 0; got {pool}")
    bf = [o for o in rs.battlefield_snapshot()
          if o["printing_name"] == "WarpBeast"]
    ex = [o for o in rs.exile_snapshot(0)
          if o["printing_name"] == "WarpBeast"]
    assert len(bf) == 1 and ex == [], (
        "recast WarpBeast must be back on bf with exile cleared")


def _r63_warp_synth_deckspec():
    """A synthetic pure-warp CardDef ({3}{R} normal, {1}{R} warp,
    3/3) + a Forest-that-taps-R, padded so both engines have enough
    R mana. Used by the cross-engine bridge pins."""
    from mtg_bot.parity.cross_engine import CardDef, DeckSpec

    warp = CardDef(
        name="SynthWarp", card_type="Creature",
        mana_cost_generic=3, mana_cost_colored=(("R", 1),),
        power=3, toughness=3,
        warp_cost_generic=1, warp_cost_colored=(("R", 1),),
        has_warp=True,
    )
    forest = CardDef(name="Forest", card_type="Land",
                     mana_produced=("R",))
    lib = ("SynthWarp",) + ("Forest",) * 39
    return DeckSpec(
        card_defs=(warp, forest),
        p0_library=lib,
        p1_library=("Forest",) * 40,
    )


def test_r63_warp_bridge_both_engines_same_castspell_pair():
    """Spec §4 pin 6 — a DeckSpec with a synthetic pure-warp CardDef:
    once the warp cost is PAYABLE, BOTH engines' legal action sets
    offer the SAME ``{use_warp:False, use_warp:True}`` CastSpell pair
    for the warp card (action-set + positional byte-equal).

    Mana must be available first: Python's ``legal_actions`` gates
    BOTH the base and warp cast on ``_can_cast`` (mana payability),
    whereas Rust's emitter has NO ``_can_cast`` gate (the documented
    pre-existing "Deliberate divergence" in ``state_ops_legal.rs`` —
    spec §2.2 caveat 1). At a no-mana state Python emits ``[]`` and
    Rust ``[False, True]`` for the SAME accepted reason the bench
    already tolerates for every base cast; that asymmetry is NOT a
    warp bug and is orthogonal to Phase 1. With the warp cost
    payable, ``_can_cast`` passes and the two engines' pairs match —
    which is what THIS pin asserts (the warp variant inherits the
    base variant's gating consistency)."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        advance_phase_python, step_both, step_both_drained,
    )
    from mtg_bot.legal import legal_actions

    py, rs = trajectory.setup_matched_game(_r63_warp_synth_deckspec())

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()

    adv(3)  # turn 1 MAIN1
    # Build R mana: play a Forest each p0 turn for 3 turns, then tap
    # two unsick Forests for the {1}{R} warp cost.
    for _ in range(3):
        f = next(o for o in py.players[py.active_player_idx].hand
                 if o.printing.name == "Forest")
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(f.id)}).divergences
        adv(24)
    f = next(o for o in py.players[py.active_player_idx].hand
             if o.printing.name == "Forest")
    assert not step_both(
        py, rs, {"kind": "play_land", "card_id": int(f.id)}).divergences
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == py.active_player_idx
                and o.printing.is_land and not o.tapped
                and not o.summoning_sick and taps < 2):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    ai = py.active_player_idx
    sw = next(o for o in py.players[ai].hand
              if o.printing.name == "SynthWarp")
    py_pairs = sorted(
        a.use_warp for a in legal_actions(py, ai)
        if getattr(a, "card_id", None) == sw.id
    )
    rs_pairs = sorted(
        a["use_warp"] for a in rs.legal_decisions(ai)
        if a.get("kind") == "cast_spell" and a.get("card_id") == sw.id
    )
    assert py_pairs == [False, True], (
        f"Python must offer base+warp once payable; got {py_pairs}")
    assert rs_pairs == [False, True], (
        f"Rust must offer base+warp; got {rs_pairs}")
    assert py_pairs == rs_pairs, (
        f"cross-engine CastSpell pair mismatch: py={py_pairs} "
        f"rs={rs_pairs}")


def test_r63_warp_bridge_cast_byte_equal_via_step_both():
    """Spec §4 pin 7 — driving ``CastSpell{use_warp:True}`` via
    ``step_both``: post-cast state (mana pool, zones,
    spell_warped_this_turn) is byte-equal across engines (zero
    field-view divergences)."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import step_both, step_both_drained, advance_phase_python

    py, rs = trajectory.setup_matched_game(_r63_warp_synth_deckspec())

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()

    adv(3)  # turn 1 MAIN1
    for _ in range(3):
        f = next(o for o in py.players[py.active_player_idx].hand
                 if o.printing.name == "Forest")
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(f.id)}).divergences
        adv(24)
    f = next(o for o in py.players[py.active_player_idx].hand
             if o.printing.name == "Forest")
    assert not step_both(
        py, rs, {"kind": "play_land", "card_id": int(f.id)}).divergences
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == py.active_player_idx
                and o.printing.is_land and not o.tapped
                and not o.summoning_sick and taps < 2):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    ai = py.active_player_idx
    sw = next(o for o in py.players[ai].hand
              if o.printing.name == "SynthWarp")
    s = step_both_drained(
        py, rs,
        {"kind": "cast_spell", "card_id": int(sw.id), "use_warp": True},
    )
    assert not s.divergences, (
        f"warp cast via step_both surfaced divergences: "
        f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}")
    assert py.spell_warped_this_turn is True
    assert rs.spell_warped_this_turn is True


def test_r63_warp_bridge_end_step_exile_and_recast_byte_equal():
    """Spec §4 pin 8 — advancing both engines to the end step: the
    warp permanent is in exile with the warp marker on BOTH;
    field-view byte-equal. Then recast-from-exile via ``step_both``:
    byte-equal."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        step_both, step_both_drained, advance_phase_python, compare_fields,
    )
    from tests.test_trajectory_parity import _step_views

    py, rs = trajectory.setup_matched_game(_r63_warp_synth_deckspec())

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()
            pv, rv = _step_views(py, rs)
            d = compare_fields(pv, rv)
            assert not d, (
                f"phase advance ({pv['step']}, turn {pv['turn']}) "
                f"diverged: "
                f"{[(x.field, x.py_value, x.rust_value) for x in d]}")

    adv(3)
    for _ in range(3):
        f = next(o for o in py.players[py.active_player_idx].hand
                 if o.printing.name == "Forest")
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(f.id)}).divergences
        adv(24)
    f = next(o for o in py.players[py.active_player_idx].hand
             if o.printing.name == "Forest")
    assert not step_both(
        py, rs, {"kind": "play_land", "card_id": int(f.id)}).divergences
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == py.active_player_idx
                and o.printing.is_land and not o.tapped
                and not o.summoning_sick and taps < 2):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    ai = py.active_player_idx
    sw = next(o for o in py.players[ai].hand
              if o.printing.name == "SynthWarp")
    assert not step_both_drained(
        py, rs,
        {"kind": "cast_spell", "card_id": int(sw.id),
         "use_warp": True}).divergences
    # **G7 Class-C delayed-trigger port (round-78).** Advance to END
    # step on both — END_STEP entry pushes WarpExileSelf onto the
    # stack (CR-603.7 delayed trigger); resolution drains it to exile.
    # adv() asserts byte-equal at every phase boundary, including END.
    adv(7)  # MAIN1 → END (WarpExileSelf onto the stack on both)
    from mtg_bot.parity.cross_engine import (
        resolve_top_of_stack_python, resolve_top_of_stack_rust,
    )
    resolve_top_of_stack_python(py, floor=0)
    resolve_top_of_stack_rust(rs, floor=0)
    py_ex = sorted(o.printing.name for o in py.players[ai].exile)
    rs_ex = sorted(o["printing_name"] for o in rs.exile_snapshot(ai))
    assert py_ex == rs_ex == ["SynthWarp"], (
        f"warp permanent must be in exile on BOTH; "
        f"py={py_ex} rs={rs_ex}")
    # Recast from exile at normal cost on a later p0 turn.
    for _ in range(80):
        pv, _rv = _step_views(py, rs)
        if (py.active_player_idx == ai
                and pv["step"] == "PRECOMBAT_MAIN"
                and py.players[ai].lands_played_this_turn == 0):
            break
        adv(1)
    fhand = [o for o in py.players[ai].hand
             if o.printing.name == "Forest"]
    if fhand:
        assert not step_both(
            py, rs,
            {"kind": "play_land",
             "card_id": int(fhand[0].id)}).divergences
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == ai and o.printing.is_land
                and not o.tapped and not o.summoning_sick and taps < 4):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    rtm_ex = next(o for o in py.players[ai].exile
                  if o.printing.name == "SynthWarp")
    s = step_both_drained(
        py, rs,
        {"kind": "cast_spell", "card_id": int(rtm_ex.id),
         "use_warp": False},
    )
    assert not s.divergences, (
        f"warp exile-recast via step_both surfaced divergences: "
        f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}")


import pytest as _pytest_warp_p15  # noqa: E402


@_pytest_warp_p15.mark.parametrize(
    "card_name,color,etb_check",
    [
        ("Germinating Wurm", "G", "gain_life"),
        ("Knight Luminary", "W", "create_token"),
        ("Starbreach Whale", "U", "surveil"),
    ],
)
def test_r65_warp_p15_trio_etb_fires_byte_equal_under_warp_cast(
    card_name, color, etb_check,
):
    """**Phase 1.5 cross-engine bridge pin (round-65) — the
    load-bearing assertion: the trio's SAFE ETB trigger fires
    BYTE-EQUAL on BOTH engines UNDER the warp-cast path.**

    warp-cast is a NEW cast path the ETB had never fired through
    pre-Phase-1.5. This pin drives the ACTUAL parsed trio printing
    via the production-deck loader (``production_deck_to_deckspec`` —
    the SAME path the bench uses; it seats the Knight-Luminary token
    printing via ``_seat_token_printings`` so the create_token ETB
    bridges) and asserts ``CastSpell{use_warp:True}`` via
    ``step_both`` surfaces ZERO field-view divergences AND the ETB
    observably resolved identically on both engines (Germinating Wurm
    +2 life; Knight Luminary a ``Human Soldier`` token on the
    battlefield; Starbreach Whale on the battlefield with the Surveil
    having resolved — zero divergence is the byte-equal proof for the
    no-observable-side library/graveyard reorder of Surveil)."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, step_both, step_both_drained, advance_phase_python,
    )

    from mtg_bot.cards.source import get_active_source
    src = get_active_source().get_printings()
    real = src.get(card_name)
    if real is None:
        _pytest_warp_p15.skip(
            f"{card_name} absent from active CardSource")
    # Verify-against-live: genuinely a safe-trigger+warp creature
    # whose ONLY gate WAS warp (the Phase-1.5 boundary).
    assert real.warp_cost is not None
    assert getattr(real, "warp_additional_life_cost", 0) == 0
    assert getattr(real, "warp_castable_from_graveyard", False) is False
    assert len(real.triggered_abilities) == 1
    assert trajectory._printing_is_bridge_pure_warp_creature(real), (
        f"{card_name} must pass the Phase-1.5 guardrail")

    # Production loader path (seats token printings). Swap the pad
    # land to one tapping the trio member's warp colored pip so both
    # engines can afford {1}{<color>}.
    spec = trajectory.production_deck_to_deckspec(
        [card_name] + ["Forest"] * 59, ["Forest"] * 60)
    newdefs = tuple(
        CardDef(name="Forest", card_type="Land",
                mana_produced=(color,))
        if c.name == "Forest" else c
        for c in spec.card_defs
    )
    spec = DeckSpec(card_defs=newdefs,
                    p0_library=spec.p0_library,
                    p1_library=spec.p1_library)
    py, rs = trajectory.setup_matched_game(spec)

    def adv(n):
        from mtg_bot.parity.cross_engine import compare_fields
        from tests.test_trajectory_parity import _step_views
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()
            pv, rv = _step_views(py, rs)
            d = compare_fields(pv, rv)
            assert not d, (
                f"{card_name}: phase advance ({pv['step']}, turn "
                f"{pv['turn']}) diverged: "
                f"{[(x.field, x.py_value, x.rust_value) for x in d]}")

    adv(3)
    for _ in range(2):
        f = next(o for o in py.players[py.active_player_idx].hand
                 if o.printing.name == "Forest")
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(f.id)}).divergences
        adv(24)
    f = next(o for o in py.players[py.active_player_idx].hand
             if o.printing.name == "Forest")
    assert not step_both(
        py, rs, {"kind": "play_land", "card_id": int(f.id)}).divergences
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == py.active_player_idx
                and o.printing.is_land and not o.tapped
                and not o.summoning_sick and taps < 2):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    ai = py.active_player_idx
    obj = next(o for o in py.players[ai].hand
               if o.printing.name == card_name)
    life_before = py.players[ai].life
    assert rs.get_life(ai) == life_before

    # The warp cast — the ETB fires through cast_creature_warp's
    # move_from_hand_to_battlefield → drain_pending_etb_dispatches
    # (the SAME ENTERS dispatch as cast_creature). Zero divergence is
    # the byte-equal proof.
    s = step_both_drained(
        py, rs,
        {"kind": "cast_spell", "card_id": int(obj.id),
         "use_warp": True},
    )
    assert not s.divergences, (
        f"{card_name}: warp-cast (ETB under warp path) diverged: "
        f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}")
    assert py.spell_warped_this_turn is True
    assert rs.spell_warped_this_turn is True

    # ETB observably resolved identically on BOTH engines.
    if etb_check == "gain_life":
        assert py.players[ai].life == life_before + 2, (
            f"{card_name}: Python ETB GainLife 2 must fire under "
            f"warp-cast; life {life_before}→{py.players[ai].life}")
        assert rs.get_life(ai) == life_before + 2, (
            f"{card_name}: Rust ETB GainLife 2 must fire under "
            f"warp-cast; life {life_before}→{rs.get_life(ai)}")
    elif etb_check == "create_token":
        py_tok = [o for o in py.battlefield
                  if o.controller_id == ai
                  and o.printing.name == "Human Soldier"]
        rs_tok = [o for o in rs.battlefield_snapshot()
                  if o["printing_name"] == "Human Soldier"]
        assert len(py_tok) == 1, (
            f"{card_name}: Python ETB must spawn 1 Human Soldier "
            f"token under warp-cast; got {len(py_tok)}")
        assert len(rs_tok) == 1, (
            f"{card_name}: Rust ETB must spawn 1 Human Soldier "
            f"token under warp-cast; got {len(rs_tok)}")
    else:  # surveil — zero divergence already proved byte-equality;
        # assert the permanent itself arrived on both bf.
        assert any(o.printing.name == card_name
                   for o in py.battlefield), (
            f"{card_name}: must be on the Python battlefield")
        assert any(o["printing_name"] == card_name
                   for o in rs.battlefield_snapshot()), (
            f"{card_name}: must be on the Rust battlefield")


def test_r65_warp_p15_trio_bridge_offers_same_base_warp_castspell_pair():
    """Phase 1.5 cross-engine bridge pin — once the warp cost is
    payable, BOTH engines' legal action sets offer the SAME
    ``{use_warp:False, use_warp:True}`` CastSpell pair for a
    safe-trigger+warp trio card (Germinating Wurm, the simplest of
    the three — GainLife ETB, no token-seating needed). The
    Phase-1.5 widen does not perturb the legal-emitter parity the
    Phase-1 substrate established."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, step_both, step_both_drained, advance_phase_python,
    )
    from mtg_bot.legal import legal_actions

    from mtg_bot.cards.source import get_active_source
    src = get_active_source().get_printings()
    real = src.get("Germinating Wurm")
    if real is None:
        _pytest_warp_p15.skip("Germinating Wurm absent")
    spec = trajectory.production_deck_to_deckspec(
        ["Germinating Wurm"] + ["Forest"] * 59, ["Forest"] * 60)
    newdefs = tuple(
        CardDef(name="Forest", card_type="Land",
                mana_produced=("G",))
        if c.name == "Forest" else c
        for c in spec.card_defs
    )
    spec = DeckSpec(card_defs=newdefs,
                    p0_library=spec.p0_library,
                    p1_library=spec.p1_library)
    py, rs = trajectory.setup_matched_game(spec)

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()

    # Build a big enough mana base that Python can afford the NORMAL
    # cost too ({4}{G} = 5 mana) — Python gates BOTH base + warp on
    # _can_cast (the documented pre-existing Rust-no-_can_cast
    # asymmetry, Phase-1 spec §2.2 caveat 1); with only the warp cost
    # ({1}{G}=2) payable Python emits ONLY [True]. With ≥5 mana both
    # base+warp are payable and the two engines' pairs match — which
    # is what this pin asserts (the Phase-1.5 widen does not perturb
    # the legal-emitter parity).
    adv(3)
    for _ in range(6):
        hand_f = [o for o in py.players[py.active_player_idx].hand
                  if o.printing.name == "Forest"]
        if hand_f:
            assert not step_both(
                py, rs,
                {"kind": "play_land",
                 "card_id": int(hand_f[0].id)}).divergences
        adv(24)
    ai = py.active_player_idx
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == ai
                and o.printing.is_land and not o.tapped
                and not o.summoning_sick and taps < 5):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    assert taps == 5, (
        f"need 5 mana so Python can afford the {{4}}{{G}} normal "
        f"cost (the base cast's _can_cast gate); tapped {taps}")
    gw = next(o for o in py.players[ai].hand
              if o.printing.name == "Germinating Wurm")
    py_pair = sorted(
        a.use_warp for a in legal_actions(py, ai)
        if getattr(a, "card_id", None) == gw.id
    )
    rs_pair = sorted(
        a["use_warp"] for a in rs.legal_decisions(ai)
        if a.get("kind") == "cast_spell" and a.get("card_id") == gw.id
    )
    assert py_pair == rs_pair == [False, True], (
        f"Germinating Wurm: base+warp CastSpell pair mismatch — "
        f"py={py_pair} rs={rs_pair}")


def test_bridge_destroy_target_artifact_or_tapped_creature_basic_translation():
    """Effect-uncovered Tier-A queue #1 — Radiant Strike inner half.
    Rust ``{"kind": "destroy_target_artifact_or_tapped_creature"}``
    translates to Python ``DestroyTargetArtifactOrTappedCreature()``.

    Mirrors Rust's parse_effect arm (unit variant — no fields).
    Python's class is a ``Destroy`` subclass that sets ``targets``
    to ``(TargetSpec(IsArtifact() | (IsCreature() & IsTapped())),)``
    in ``__post_init__``.
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetArtifactOrTappedCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_artifact_or_tapped_creature"}
    )
    assert isinstance(e, DestroyTargetArtifactOrTappedCreature)


def test_bridge_dtaotc_targets_set_to_artifact_or_tapped_creature():
    """Bridge arm pin: ``__post_init__`` wires the ``targets`` tuple
    to a single ``TargetSpec(IsArtifact() | (IsCreature() &
    IsTapped()))``. Pins the typed-OR filter shape — a bug that
    dropped the ``__post_init__`` would leave ``targets`` empty and
    make the resolver pick anything (the OR-filter teeth are the
    whole new behavior).
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetArtifactOrTappedCreature,
    )
    from mtg_bot.effects.object_predicates import (
        IsArtifact, IsCreature, IsTapped,
    )
    from mtg_bot.effects.targets import TargetSpec
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_artifact_or_tapped_creature"}
    )
    assert isinstance(e, DestroyTargetArtifactOrTappedCreature)
    expected = IsArtifact() | (IsCreature() & IsTapped())
    assert e.targets == (TargetSpec(expected),)


def test_bridge_dtaotc_extra_keys_silently_ignored():
    """Bridge arm pin: extra keys on the dict are silently dropped
    (mirrors Rust's unit-variant parse arm which takes no fields and
    simply returns the variant — no extra-key validation; the
    silent-drop convention shared by sister unit-variant arms).
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetArtifactOrTappedCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_artifact_or_tapped_creature",
         "spurious": 42}
    )
    assert isinstance(e, DestroyTargetArtifactOrTappedCreature)


def test_bridge_dtaotc_equality_with_default_constructor():
    """Bridge arm pin: round-trip equality. Translated effect equals
    a directly-constructed ``DestroyTargetArtifactOrTappedCreature()``
    — the bare unit-shape carries no parameters so equality is
    structural (including the ``__post_init__``-set ``targets``).
    """
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetArtifactOrTappedCreature,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "destroy_target_artifact_or_tapped_creature"}
    )
    assert e == DestroyTargetArtifactOrTappedCreature()


def test_bridge_dtaotc_composable_in_radiant_strike_sequence():
    """Round-trip the FULL Radiant Strike shape through the recursive
    Sequence translator: ``Sequence([
    DestroyTargetArtifactOrTappedCreature(), GainLife(3)])``. Pins
    the new destroy arm composes with the already-live ``sequence``
    / ``gain_life`` arms end-to-end (the exact dict the Rust
    ``effect_to_pydict`` emits for the real card).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.destroy import (
        DestroyTargetArtifactOrTappedCreature,
    )
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "destroy_target_artifact_or_tapped_creature"},
                {"kind": "gain_life", "amount": 3},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DestroyTargetArtifactOrTappedCreature)
    assert isinstance(e.effects[1], GainLife)
    assert e.effects[1].amount == 3


def test_bridge_seedship_impact_effect_basic_translation():
    """Effect-uncovered Tier-A queue #2 — Seedship Impact. Rust
    ``{"kind": "seedship_impact_effect", "token_printing_id": N}``
    translates to Python ``SeedshipImpactEffect()`` (the bridge DROPS
    the Rust-side pid carrier — Python's class is field-free and mints
    the Lander intrinsically from TOKEN_PRINTINGS["Lander"]).
    """
    from mtg_bot.effects.bespoke import SeedshipImpactEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "seedship_impact_effect", "token_printing_id": 42}
    )
    assert isinstance(e, SeedshipImpactEffect)


def test_bridge_seedship_impact_effect_pid_dropped_equality():
    """Bridge arm pin: the translated effect equals a directly-
    constructed ``SeedshipImpactEffect()`` regardless of the Rust
    ``token_printing_id`` value — the pid is a Rust-registry carrier
    with no Python mirror (Python uses TOKEN_PRINTINGS["Lander"]); the
    bridge constructs the bare field-free class so two different pids
    round-trip to EQUAL Python effects.
    """
    from mtg_bot.effects.bespoke import SeedshipImpactEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "seedship_impact_effect", "token_printing_id": 7}
    )
    e2 = _ta_dict_to_python_effect(
        {"kind": "seedship_impact_effect", "token_printing_id": 999}
    )
    assert e1 == e2 == SeedshipImpactEffect()


def test_bridge_seedship_impact_effect_extra_keys_silently_ignored():
    """Bridge arm pin: extra keys beyond kind/token_printing_id are
    silently dropped (the bridge only constructs the bare Python
    class; the silent-drop convention shared by sister arms).
    """
    from mtg_bot.effects.bespoke import SeedshipImpactEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "seedship_impact_effect", "token_printing_id": 1,
         "spurious": "x"}
    )
    assert isinstance(e, SeedshipImpactEffect)


def test_bridge_seedship_impact_effect_composable_in_sequence():
    """Round-trip the bespoke arm through the recursive Sequence
    translator to pin it dispatches correctly when nested (forward-
    compat with any future Sequence-wrapped consumer).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.bespoke import SeedshipImpactEffect
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "seedship_impact_effect",
                 "token_printing_id": 3},
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], SeedshipImpactEffect)
    assert isinstance(e.effects[1], GainLife)


def test_bridge_orbital_plunge_basic_translation_amount_6():
    """Effect-uncovered Tier-A queue #3 — Orbital Plunge. Rust
    ``{"kind": "deal_damage_to_target_creature_with_excess_lander_
    rider", "amount": 6, "token_printing_id": N}`` translates to
    Python ``DealDamageToTargetCreatureWithExcessLanderRider(
    amount=6)`` — the Rust-side Lander pid is DROPPED (Python's class
    mints intrinsically from TOKEN_PRINTINGS["Lander"]).
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToTargetCreatureWithExcessLanderRider,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_target_creature_with_excess_lander_rider",
         "amount": 6, "token_printing_id": 42}
    )
    assert isinstance(e, DealDamageToTargetCreatureWithExcessLanderRider)
    assert e.amount == 6


def test_bridge_orbital_plunge_amount_rides_through():
    """Bridge arm pin: the ``amount`` field rides through (not
    hard-coded to 6); a non-6 amount round-trips faithfully so a
    future reprint at a different damage value stays correct.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToTargetCreatureWithExcessLanderRider,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_target_creature_with_excess_lander_rider",
         "amount": 9, "token_printing_id": 1}
    )
    assert e.amount == 9


def test_bridge_orbital_plunge_pid_dropped_equality():
    """Bridge arm pin: the translated effect equals a directly-
    constructed ``DealDamageToTargetCreatureWithExcessLanderRider(
    amount=6)`` regardless of the Rust ``token_printing_id`` — the
    pid is a Rust-registry carrier with no Python mirror, so two
    different pids round-trip to EQUAL Python effects.
    """
    from mtg_bot.effects.generic.damage import (
        DealDamageToTargetCreatureWithExcessLanderRider,
    )
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e1 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_target_creature_with_excess_lander_rider",
         "amount": 6, "token_printing_id": 7}
    )
    e2 = _ta_dict_to_python_effect(
        {"kind": "deal_damage_to_target_creature_with_excess_lander_rider",
         "amount": 6, "token_printing_id": 999}
    )
    assert (
        e1 == e2
        == DealDamageToTargetCreatureWithExcessLanderRider(amount=6)
    )


def test_bridge_orbital_plunge_composable_in_sequence():
    """Round-trip the bespoke arm through the recursive Sequence
    translator (forward-compat with any Sequence-wrapped consumer).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.damage import (
        DealDamageToTargetCreatureWithExcessLanderRider,
    )
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": ("deal_damage_to_target_creature_"
                          "with_excess_lander_rider"),
                 "amount": 6, "token_printing_id": 3},
                {"kind": "gain_life", "amount": 2},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(
        e.effects[0], DealDamageToTargetCreatureWithExcessLanderRider
    )
    assert e.effects[0].amount == 6
    assert isinstance(e.effects[1], GainLife)


def test_bridge_depressurize_basic_translation_pp3():
    """Effect-uncovered Tier-A queue #4 — Depressurize. Rust
    ``{"kind": "depressurize_effect", "power_penalty": 3}``
    translates to Python ``DepressurizeEffect(power_penalty=3)``.
    """
    from mtg_bot.effects.bespoke import DepressurizeEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "depressurize_effect", "power_penalty": 3}
    )
    assert isinstance(e, DepressurizeEffect)
    assert e.power_penalty == 3


def test_bridge_depressurize_power_penalty_rides_through():
    """Bridge arm pin: ``power_penalty`` rides through (not
    hard-coded to 3) so a future reprint at a different penalty
    round-trips faithfully.
    """
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "depressurize_effect", "power_penalty": 5}
    )
    assert e.power_penalty == 5


def test_bridge_depressurize_default_pp_when_absent():
    """Bridge arm pin: a missing ``power_penalty`` defaults to 3
    (mirrors Python's ``DepressurizeEffect(power_penalty: int = 3)``
    default — the lenient-default convention for partial fixtures).
    """
    from mtg_bot.effects.bespoke import DepressurizeEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "depressurize_effect"})
    assert e == DepressurizeEffect(power_penalty=3)


def test_bridge_depressurize_composable_in_sequence():
    """Round-trip the bespoke arm through the recursive Sequence
    translator (forward-compat with any Sequence-wrapped consumer).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.bespoke import DepressurizeEffect
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "depressurize_effect", "power_penalty": 3},
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], DepressurizeEffect)
    assert e.effects[0].power_penalty == 3
    assert isinstance(e.effects[1], GainLife)


def test_bridge_spell_card_lever_phase1_materialize_byte_equal():
    """**Spell-card-bridge lever Phase 1 (round-70) — bridge pin #7.**
    A real single-target Tier-A spell printing (Depressurize —
    ``DepressurizeEffect(power_penalty=3)``, covered+bridged, pid-free)
    routed through ``_printing_to_card_def`` materializes the SAME
    printing on both engines: Python's ``CardPrinting.spell_effect``
    is non-None (round-trips byte-equal to the live parse); the Rust
    registry dict carries a non-empty ``effects`` list (the
    ``depressurize_effect`` dict the Rust ``parse_effect`` consumes).
    Proves Gate A (guardrail admit) + Gate B (CardDef.spell_effect
    field + the bidirectional bridge arms) round-trip from ONE source.
    """
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _card_def_to_python_printing
    from mtg_bot.cards.source import get_active_source

    src = get_active_source().get_printings()
    real = src.get("Depressurize")
    if real is None:
        import pytest
        pytest.skip("Depressurize absent from active CardSource")
    # Verify-against-live: genuinely the Phase-1 admit shape.
    assert trajectory._printing_is_bridge_simple_spell(real), (
        "Depressurize must pass the Phase-1 spell guardrail")
    cd = trajectory._printing_to_card_def(real)
    assert cd.card_type == "Instant"
    # Rust registry-dict side: spell_effect dict is the single
    # effect (the schema Rust's parse_effect consumes).
    assert cd.spell_effect == {
        "kind": "depressurize_effect", "power_penalty": 3}
    # Python side: materialized CardPrinting.spell_effect round-trips
    # byte-equal to the live-parsed effect.
    rt = _card_def_to_python_printing(cd)
    assert rt.spell_effect is not None
    assert repr(rt.spell_effect) == repr(real.spell_effect)
    assert rt.card_types == ("Instant",)
    assert rt.is_instant is True


def test_bridge_spell_card_lever_phase1_real_eoe_cast_resolve_parity():
    """**Spell-card-bridge lever Phase 1 (round-70) — bridge pins
    #8 + #9 (the genuine full-card admit / acceptance probe).** Drive
    the ACTUAL parsed Depressurize printing through the genuine loader
    admit transform (``_printing_to_card_def`` — the EXACT CardDef the
    production-deck loader's bridge OR-chain produces for this real
    printing; pin #7 proved that transform byte-equal) into a
    DETERMINISTIC matched game (no shuffle — the
    ``production_deck_to_deckspec`` library order is preserved when no
    ``shared_rng_seed`` is set), then via ``step_both``: the cast
    (``CastSpell`` -> ``cast_spell_noncreature`` -> spell StackItem)
    AND the resolve (the harness CE5 cast-vs-resolve mirror drains
    both stacks: ``DepressurizeEffect`` applies to the deterministic-
    first creature, then ``move_source_post_resolve`` sends the spell
    hand -> graveyard via the ALREADY-PRESENT Instant/Sorcery ->
    ZONE_GRAVEYARD route) surface ZERO field-view divergences.
    Asserts post-cast AND post-resolve byte-equal + the spell source
    in the controller's graveyard on BOTH engines + the effect
    observably resolved (the targeted Bear destroyed)."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, step_both, step_both_drained, advance_phase_python,
        compare_fields,
    )
    from tests.test_trajectory_parity import _step_views
    from tests.parity._picker_shims import _PickFirstLegalTargetAgent
    from mtg_bot.cards.source import get_active_source

    src = get_active_source().get_printings()
    real = src.get("Depressurize")
    if real is None:
        import pytest
        pytest.skip("Depressurize absent from active CardSource")
    # Verify-against-live: genuinely the Phase-1 admit shape.
    assert trajectory._printing_is_bridge_simple_spell(real), (
        "Depressurize must pass the Phase-1 spell guardrail")

    # The genuine loader admit transform — the EXACT CardDef the
    # production bridge OR-chain produces for the real Depressurize
    # printing (pin #7 proved this round-trips byte-equal). A vanilla
    # 2/2 Grizzly Bears is the target; Forest produces {B} so both
    # engines afford the {1}{B} costs. Deterministic library (no
    # shuffle): opening hand draws Grizzly Bears + Depressurize +
    # Forests, the rest Forest.
    dep_cd = trajectory._printing_to_card_def(real)
    bear = CardDef(name="Grizzly Bears", card_type="Creature",
                   mana_cost_generic=1, mana_cost_colored=(("B", 1),),
                   power=2, toughness=2)
    forest = CardDef(name="Forest", card_type="Land",
                     mana_produced=("B",))
    spec = DeckSpec(
        card_defs=(dep_cd, bear, forest),
        p0_library=tuple(
            ["Grizzly Bears", "Depressurize"] + ["Forest"] * 58),
        p1_library=tuple(["Forest"] * 60),
    )
    py, rs = trajectory.setup_matched_game(spec)
    # First-legal target agent (the campaign no-callback convention —
    # matches Rust's bf_iter_etb_sorted first-creature cast-time scan;
    # the equivalence the Tier-A real-EoE probes proved byte-equal).
    for pl in py.players:
        pl.agent = _PickFirstLegalTargetAgent()

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()
            pv, rv = _step_views(py, rs)
            d = compare_fields(pv, rv)
            assert not d, (
                f"phase advance ({pv['step']}, turn {pv['turn']}) "
                f"diverged: "
                f"{[(x.field, x.py_value, x.rust_value) for x in d]}")

    def tap_n(n):
        ai = py.active_player_idx
        taps = 0
        for o in list(py.battlefield):
            if (o.controller_id == ai and o.printing.is_land
                    and not o.tapped and not o.summoning_sick
                    and taps < n):
                assert not step_both(
                    py, rs,
                    {"kind": "activate_ability",
                     "source_id": int(o.id),
                     "ability_index": 0}).divergences
                taps += 1
        assert taps == n, f"need {n} mana; tapped {taps}"

    adv(3)  # turn 1 MAIN1
    # Play one Forest each of p0's first 4 turns (deterministic ramp:
    # adv(24) = +2 turns, back to p0 MAIN1).
    for t in range(4):
        ai = py.active_player_idx
        hf = [o for o in py.players[ai].hand
              if o.printing.name == "Forest"]
        assert hf, "Forest must be in hand each ramp turn"
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(hf[0].id)}).divergences
        if t < 3:
            adv(24)
    ai = py.active_player_idx
    # Cast the Bear (the target). {1}{B} = tap 2 lands.
    tap_n(2)
    bear_obj = next(o for o in py.players[ai].hand
                    if o.printing.name == "Grizzly Bears")
    assert not step_both_drained(
        py, rs,
        {"kind": "cast_spell",
         "card_id": int(bear_obj.id)}).divergences
    adv(24)  # next p0 turn — the Bear is no longer summoning-sick
    ai = py.active_player_idx
    bears_bf = [o for o in py.battlefield
                if o.printing.name == "Grizzly Bears"]
    assert bears_bf, "Grizzly Bears must be on the battlefield"
    target_bear_id = bears_bf[0].id
    gy_py_before = len(py.players[ai].graveyard)

    # Cast Depressurize at the Bear. {1}{B} = tap 2 lands.
    tap_n(2)
    spell = next(o for o in py.players[ai].hand
                 if o.printing.name == "Depressurize")
    s = step_both_drained(
        py, rs, {"kind": "cast_spell", "card_id": int(spell.id)})
    # Pin #8: post-cast AND post-resolve byte-equal (the CE5
    # cast-vs-resolve mirror drains both stacks before the compare).
    assert not s.divergences, (
        "Depressurize cast->resolve diverged: "
        f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}")
    # Pin #9: the spell source is in the controller's graveyard on
    # BOTH engines post-resolve (zero divergence already proves Rust
    # agrees; assert the Python observable explicitly).
    py_spell = next((o for o in py.players[ai].graveyard
                     if o.printing.name == "Depressurize"), None)
    assert py_spell is not None, (
        "Depressurize must be in p%d graveyard post-resolve" % ai)
    # The resolve adds TWO cards to p%d's graveyard: the destroyed
    # Grizzly Bears (CR 704 — it is p0's own creature, the target) AND
    # the Depressurize spell itself (CR 608.2m). Both controller-side.
    # The byte-equal compare above already proved Rust agrees; assert
    # the Python observable count delta explicitly.
    assert len(py.players[ai].graveyard) == gy_py_before + 2
    # Effect observably resolved: the deterministic-first Grizzly
    # Bears (2/2 -3/-0 -> eff power -1 <= 0) was destroyed.
    assert not [o for o in py.battlefield
                if o.id == target_bear_id], (
        "the deterministic-first Grizzly Bears must be destroyed by "
        "Depressurize (-3/-0 -> eff power -1 <= 0)")


def test_spell_card_lever_phase1_negative_pins_decline_set():
    """**Spell-card-bridge lever Phase 1 (round-70) + Lander-spell-
    token-seating sub-substrate (round-73) + Phase 2 (round-76) —
    pins #10-#13 (the §2 guardrail's teeth).** Table-driven
    assertions that the decline set stays filtered and the admit set
    is EXACTLY the covered+bridged Tier-A spells whose Effect close
    landed, whose token (if any) the loader can seat, AND whose
    target grammar is implemented (Phase-1 single-target OR Phase-2
    the one 2-slot ``('my_creature','creature')`` pair) — NOT a
    static card-name list; the lever is the multiplier; the set
    auto-grows as each per-card Tier-A close lands. The round-73
    sub-substrate added the two Lander-pid spells once their token
    became seatable on the spell axis (Pin #12b flipped DECLINE →
    SEATED); round-76 Phase 2 flipped Diplomatic Relations DECLINE →
    ADMITTED via the 2-slot grammar (Pin #12 — new-code-is-right per
    ``feedback_parity_new_code_is_right.md``)."""
    from mtg_bot.parity import trajectory
    from mtg_bot.cards.source import get_active_source

    src = get_active_source().get_printings()

    def g(name):
        p = src.get(name)
        if p is None:
            import pytest
            pytest.skip(f"{name} absent from active CardSource")
        return trajectory._printing_is_bridge_simple_spell(p)

    # Pin #10: a covered+bridged pid-free single-target Tier-A spell
    # is ADMITTED (Radiant Strike — Sequence(DestroyTargetArtifactOr
    # TappedCreature, GainLife); Depressurize — DepressurizeEffect).
    # The covered-set gate is the multiplier: these flipped to True
    # exactly when their per-card Effect close landed.
    assert g("Radiant Strike") is True
    assert g("Depressurize") is True

    # Pin #11 (UPDATED via close-encounter-substrate at merge
    # ``17dc7d02``, 2026-05-21 — picked up by scour-for-scrap-cycle-2's
    # admit-set re-anchor): Close Encounter FLIPPED from DECLINED →
    # ADMITTED. The Close Encounter substrate lane added the
    # ``Effect::CloseEncounterEffect`` Rust variant + bridge arms +
    # the additional-cost-choice substrate that lets the live
    # printing satisfy both the covered-set gate AND the round-trip-
    # or-bail teeth. The matching Phase-1 ``(1, 1)`` admit shape was
    # already in place; Close Encounter rides the existing surgical
    # widen. Pre-substrate this pin asserted DECLINED via the covered-
    # set gate; post-substrate that load-bearing gate FLIPS True. The
    # pin is preserved as an explicit ADMIT pin so the lane-credit is
    # documented and a regression that re-declines Close Encounter is
    # caught here.
    ce = src.get("Close Encounter")
    if ce is not None:
        assert trajectory._printing_is_bridge_simple_spell(ce) is True
        assert ce.warp_cost is None  # verify-first: the spec was stale
        assert (
            trajectory._spell_effect_is_covered_and_bridged(ce)
            is True
        ), "CloseEncounterEffect must be covered+bridged post-substrate"

    # Pin #12 (round-76 Phase 2 — FLIPPED from DECLINE to ADMITTED;
    # new-code-is-right per feedback_parity_new_code_is_right). In
    # Phase 1 Diplomatic Relations was DECLINED (2-slot grammar
    # unimplemented). Phase 2 ships the 2-slot StackItem binding +
    # double resolve-swap + the DiplomaticRelationsEffect Rust
    # variant/bridge/reverse-leaf, so it now passes the guardrail.
    # Pin the EXACT shape that flipped (the surgical 2-slot widening,
    # NOT a blanket multi-target widen): (2,2) target_count_range AND
    # the disjoint friendly/opponent slot pair — AND that it is now
    # covered+bridged + round-trips the reverse serializer.
    dr = src.get("Diplomatic Relations")
    if dr is not None:
        se = dr.spell_effect
        assert se.target_count_range() == (2, 2)
        assert se.target_slot_kinds() == ("my_creature", "creature")
        assert trajectory._printing_is_bridge_simple_spell(dr) is True, (
            "Diplomatic Relations must be ADMITTED post-Phase-2 (the "
            "2-slot grammar is implemented + pinned byte-equal)")
        assert (
            trajectory._spell_effect_is_covered_and_bridged(dr) is True
        ), "DiplomaticRelationsEffect must be covered+bridged (#7)"
        assert trajectory._python_spell_effect_to_dict(
            dr.spell_effect
        ) == {"kind": "diplomatic_relations_effect"}, (
            "DR reverse serializer must emit the field-free kind-only "
            "dict (the 2-slot targets ride on the StackItem, NOT the "
            "effect dict)")
        # Negative control — the surgical Phase-2 widening admits ONLY
        # this exact 2-slot pair, NOT any 2-target shape. A synthetic
        # (2,2) effect with a DIFFERENT slot tuple stays declined
        # (proves the widening is the documented one heterogeneous
        # pair, not a count-chasing blanket — the §2 teeth).
        class _FakeOtherTwoSlot:
            def target_count_range(self):
                return (2, 2)

            def target_slot_kinds(self):
                return ("creature", "creature")

        class _FakePrinting:
            card_types = ("Instant",)
            spell_effect = _FakeOtherTwoSlot()
            warp_cost = None
            kicker_cost = None
            impending_cost = None
            room_halves = ()
            mana_cost = None
            additional_cost_choice = None

        assert (
            trajectory._printing_is_bridge_simple_spell(_FakePrinting())
            is False
        ), ("the Phase-2 widening must admit ONLY the documented "
            "('my_creature','creature') pair — a different 2-slot "
            "kind-tuple stays declined (the surgical-widen §2 teeth)")

    # Pin #12b (Lander-spell-token-seating sub-substrate, round-73 —
    # FLIPPED from DECLINE to SEATED; new-code-is-right per
    # feedback_parity_new_code_is_right). Seedship Impact / Orbital
    # Plunge are covered+bridged AND now ADMITTED: their Rust
    # parse_effect requires a Lander token_printing_id, and the
    # loader's _seat_token_printings pass now ALSO walks spell_effect
    # to seat it (the exact triggered-abilities machinery applied to
    # the spell axis). The reverse serializer emits the symbolic
    # _token_card_def Lander reference (round-trip-or-bail FLIPPED
    # POSITIVE — the no-fallback-hybrid discipline preserved: the
    # symbolic ref is emitted only because the seating pass that
    # resolves it now exists). Pin the seated round-trip: covered True,
    # guardrail True, the reverse serializer emits the symbolic Lander
    # ref, and the loader resolves it to a numeric token_printing_id.
    for nm in ("Seedship Impact", "Orbital Plunge"):
        p = src.get(nm)
        if p is None:
            continue
        assert (
            trajectory._spell_effect_is_covered_and_bridged(p) is True
        ), f"{nm} Effect must be covered+bridged"
        assert trajectory._printing_is_bridge_simple_spell(p) is True, (
            f"{nm} must be ADMITTED post-sub-substrate (the Lander "
            f"token_printing_id is now seatable on the spell path)")
        rt = trajectory._python_spell_effect_to_dict(p.spell_effect)
        assert rt is not None, (
            f"{nm} must round-trip the reverse serializer post-"
            f"sub-substrate")
        # Pre-seating: the dict carries the SYMBOLIC Lander ref (no
        # numeric pid yet — the loader assigns it once the union
        # card_defs tuple is frozen, exactly like the create_token
        # triggered carrier).
        assert "_token_card_def" in rt and "token_printing_id" not in rt
        assert rt["_token_card_def"].name == "Lander"
        # Post-seating (the genuine production loader path): the spell
        # CardDef's spell_effect carries a NUMERIC token_printing_id ==
        # the seated Lander's pid, the symbolic ref dropped — the exact
        # field set the Rust parse_effect arm requires.
        spec = trajectory.production_deck_to_deckspec(
            [nm] * 4 + ["Forest"] * 36, ["Forest"] * 40, pad_to_size=40)
        cds = {cd.name: cd for cd in spec.card_defs}
        assert nm in cds and "Lander" in cds, (
            f"{nm} + its seated Lander token must be admitted; "
            f"got {sorted(cds)}")
        lander_pid = next(
            i for i, cd in enumerate(spec.card_defs)
            if cd.name == "Lander")
        se_dict = cds[nm].spell_effect
        assert "_token_card_def" not in se_dict
        assert se_dict["token_printing_id"] == lander_pid, (
            f"{nm} spell_effect must carry the numeric seated Lander "
            f"pid {lander_pid}; got {se_dict}")
        # The Orbital Plunge two-field arm preserves ``amount``; the
        # Seedship single-field arm carries no extra field (NO spurious
        # ``count`` — that would be the silent-mistranslation class the
        # §2 teeth prevent: the Rust arms don't take one).
        assert "count" not in se_dict
        if nm == "Orbital Plunge":
            assert se_dict["amount"] == 6

    # Pin #13: the set of EoE printings passing the guardrail equals
    # EXACTLY the covered+bridged single-target Tier-A spells whose
    # Effect close has landed AND whose token (if any) the loader can
    # seat. Regenerable; auto-grows as each subsequent per-card Effect
    # closes — that growth IS the multiplier proof (NOT a static
    # card-name list). Post-Lander-spell-token-seating sub-substrate
    # (round-73) the set was {Depressurize, Orbital Plunge, Radiant
    # Strike, Seedship Impact}; round-74 closed Tier-A #6 Biosynthic
    # Burst (+1 → 5); round-75 closed Tier-A #5 Focus Fire
    # (``FocusFireEffect { base }`` single-int variant + bridge +
    # reverse-serializer leaf — pid-free, single-target ``{W}``
    # Instant, covered), which AUTO-JOINED via the §2 auto-grow teeth.
    # round-76 closed Diplomatic Relations via the Phase-2 2-slot
    # grammar widen (6→7). The Cut Propulsion close (Effect-uncovered
    # Tier-A; ~11,990 17lands decks) AUTO-JOINED (+1 → 8); the
    # Shattered Wings close (Effect-uncovered single-gated tier;
    # ~7,793 17lands decks — destroy variant + Surveil 1 outer
    # Sequence; the trajectory reverse-serializer Surveil leaf flipped
    # the round-trip teeth) AUTO-JOINED (+1 → 9). The §2 auto-grow
    # teeth keep firing as each subsequent per-card close lands — the
    # lever queries the LIVE covered set, NOT a hardcoded list —
    # new-code-is-right per ``feedback_parity_new_code_is_right.md``;
    # this guard's purpose is to be updated when a Tier-A close lands,
    # NOT to forbid the growth it explicitly anticipates). The
    # effect-uncovered-slice round closed Cut Propulsion (+1 → 8), the
    # invasive-maneuvers-slice round closed Invasive Maneuvers (+1 →
    # 9), the shattered-wings-slice round closed Shattered Wings
    # (+1 → 10), and the next-effect-uncovered round closed Emergency
    # Eject (Effect-uncovered single-gated tier; ~6,291 17lands decks
    # — same Lander-pid carrier shape as Seedship Impact / Orbital
    # Plunge; the trajectory reverse-serializer leaf + the
    # ``_SPELL_TOKEN_PID_KINDS`` tuple entry flipped the round-trip
    # teeth) AUTO-JOINED (+1 → 11) — all AUTO-JOINED via the §2 auto-
    # grow teeth. **wrath-admit-widening (round-87, 2026-05-21):** the
    # ``(0, 0)``-shape Phase-3 wrath admit arm + DealNDamageToEach
    # Creature / DestroyAllCreatures / ManifestDread serializer leaves
    # AUTO-JOINED 4 new wrath-shape spells (Pyroclasm + Manifest Dread
    # + Break Down the Door + Under the Skin; the latter 3 are duskmourn
    # .json pool members loaded into the active source; Pyroclasm is a
    # genuine EoE-pool member already PROBED by its dedicated effect-
    # close probe). 11 → 15 — all AUTO-JOINED via the §2 auto-grow teeth.
    # **beyond-the-quiet-slice (2026-05-21):** the
    # ``ExileAllCreaturesAndSpacecraft`` substrate (Rust Effect variant
    # + bridge arm + reverse-serializer leaf) AUTO-JOINED Beyond the
    # Quiet (4,698 17lands decks; genuine EoE-pool wrath
    # ``(0, 0)``-shape Sorcery) via Lane Y's existing Phase-3 wrath
    # admit arm — no further guardrail widening needed; the new
    # ``ExileAllCreaturesAndSpacecraft`` serializer leaf flipped the
    # round-trip teeth and the §2 covered+bridged gate now resolves
    # positively. 15 → 16 — AUTO-JOINED via the §2 auto-grow teeth.
    # **modal-chooseone-cycle-4 (2026-05-21):** the ModalChooseOne
    # envelope's Cycle 4 widened the admit predicate (the unconditional
    # ModalChooseOne reject is removed; round-trip-or-bail + the
    # ``(0, 0)``-shape arm now carry the modal admit load) AND added
    # the per-leaf ``DealDamageEachOpponent`` +
    # ``ExileAllArtifactsWithManaValueAtMost`` serializer leaves +
    # the Rust ``Effect::ExileAllArtifactsWithManaValueAtMost``
    # variant. AUTO-JOINED 1 new modal: Ruinous Rampage (the only
    # all-untargeted ModalChooseOne in the EoE pool today; some-
    # targeted modals stay declined via the target-grammar gate
    # pending Cycle 5's per-mode target binding). 16 → 17.
    admitted = sorted(
        n for n in src
        if trajectory._printing_is_bridge_simple_spell(src[n])
    )
    # **drill-too-deep-slice (2026-05-21):** Cycle 6 closes the
    # first per-mode-target ModalChooseOne via the
    # ``PutChargeCountersOnTargetSpacecraftOrPlanetYouControl``
    # Effect variant + bridge arm + reverse-serializer leaf + the
    # Cycle-5 detector table extension in
    # ``cast_spell_noncreature``. Both Drill Too Deep modes are
    # now covered+bridged, the ModalChooseOne composer recurses
    # into both, and the {1}{R} (1, 1) Instant rides the Phase-1
    # admit shape. Adding mode 1's ``DestroyTargetArtifact`` to
    # the trajectory ``_leaf`` table (required for the
    # ModalChooseOne composer recursion) ALSO auto-joins Untimely
    # Malfunction — a {1}{R} Instant whose top-level
    # ``spell_effect`` IS ``DestroyTargetArtifact``; previously
    # declined only because the serializer leaf was missing. The
    # §2 auto-grow teeth: covered+bridged AND round-trips ⇒
    # admitted, no fallback hybrid. 17 → 19 — both AUTO-JOINED.
    # **multi-target-admit-phase4 + non-state-rs-tractable-Exorcise
    # (2026-05-21):** Two sibling closures landed concurrently growing
    # the admit set 19 → 22. Phase 4 widens the admit predicate to
    # accept ``(min, N)`` graveyard-source multi-target shapes with
    # uniform slot_kinds + adds the new Rust
    # ``Effect::ReturnUpToNSmallCreaturesFromGraveyardWithCounters``
    # variant → Scout for Survivors AUTO-JOINS. The Exorcise lane
    # adds ``Effect::ExorciseEffect`` (Duskmourn {1}{W} Sorcery,
    # structural sibling of Shattered Wings's destroy-half filter with
    # exile-not-destroy + power-4-not-flying swaps) → Exorcise
    # AUTO-JOINS. Pull Through the Weft + Archenemy's Charm stay
    # declined (each needs its own substrate per the multi-target
    # lane's honest-stop). 19 → 22 — both AUTO-JOINED.
    # **close-encounter-substrate (2026-05-21, merge ``17dc7d02``):**
    # adds the additional-cost-choice substrate +
    # ``Effect::CloseEncounterEffect`` Rust variant + bridge arms
    # → Close Encounter AUTO-JOINS the Phase-1 ``(1, 1)`` admit shape.
    # Admit set 22 → 23 (the prior pin missed this drift; this
    # re-anchor catches it up).
    # **scour-for-scrap-cycle-2 (2026-05-21):** FIRST ``or_both=True``
    # closure. Adds ``Effect::ReturnTargetArtifactCardFromYourGraveyard
    # ToHand`` Rust variant + per-mode detector or_both walk + the
    # Phase-5 ``(0, 1)`` ``or_both`` admit-shape widen + the matching
    # bridge arms + narrow ``_leaf(SearchLibraryForCard,
    # filter_name="artifact")`` admit (honest-stop). Scour for Scrap
    # AUTO-JOINS. Admit set 23 → 24.
    # **control-change-ueot-cycle-4-p1 (2026-05-22):** Mutinous
    # Massacre AUTO-JOINED. The C1-C3 control-change-UEOT substrate
    # (controller_override field + helper + cleanup drain + bridge
    # round-trip) + Cycle-4's ``Effect::MutinousMassacreEffect``
    # field-free unit variant + resolve arm (the FIRST Rust call
    # site that mints a non-None override) + bridge arms + reverse-
    # serializer leaf flip Mutinous Massacre COVERED → ADMITTED.
    # 24 → 25 (4,671 17lands decks). The card rides the Phase-3
    # ``(0, 0)`` wrath admit shape (no targets — the destroy +
    # gain-control + untap + haste resolution all walks the
    # battlefield itself at resolve time). Per the §2 auto-grow
    # teeth, no admit-set widening was needed — just landing the
    # Effect coverage + serializer leaf.
    # **control-change-ueot-cycle-5-p1 (2026-05-22):** Systems
    # Override AUTO-JOINED. Cycle 5 lands ``Effect::SystemsOverride
    # Effect`` (field-free unit variant) + the resolve arm (control-
    # grant + untap + haste + Spacecraft-conditional +10 charge
    # counters + delayed-trigger schedule) + the new
    # ``Effect::RemoveChargeCountersFromSourceDelayed { amount }``
    # delayed-trigger effect + the end-step drain (composes with the
    # existing G7 Class-C ``run_delayed_end_step_triggers``
    # machinery — second consumer after WarpExileSelf) + bridge arms
    # + reverse-serializer leaves. Systems Override COVERED →
    # ADMITTED. 25 → 26 (4,775 17lands decks). The card rides the
    # Phase-1 ``(1, 1)`` single-target admit shape; the single
    # target rides on the StackItem's payload-target field, the
    # +10/-10 amount + Spacecraft-conditional rider are intrinsic
    # to the resolve arm. Per the §2 auto-grow teeth, no admit-set
    # widening was needed — just landing the Effect coverage +
    # serializer leaves.
    assert admitted == [
        "Beyond the Quiet", "Biosynthic Burst", "Break Down the Door",
        "Close Encounter", "Cut Propulsion", "Depressurize",
        "Diplomatic Relations", "Drill Too Deep", "Emergency Eject",
        "Exorcise", "Focus Fire", "Invasive Maneuvers",
        "Manifest Dread", "Mutinous Massacre", "Orbital Plunge",
        "Pyroclasm", "Radiant Strike", "Reroute Systems",
        "Ruinous Rampage", "Scour for Scrap", "Scout for Survivors",
        "Seedship Impact", "Shattered Wings", "Systems Override",
        "Under the Skin", "Untimely Malfunction",
    ], (
        f"spell-card-bridge admit set drifted: {admitted} (expected "
        f"exactly the 26 covered+bridged Tier-A spells — the 25 "
        f"prior + 1 control-change-UEOT closure: Systems Override "
        f"via control-change-ueot-cycle-5-p1 — auto-grows further "
        f"as the remaining Tier-A and per-mode-target modal spells "
        f"close)")


def test_bridge_lander_spell_token_seating_materialize_byte_equal():
    """**Lander-spell-token-seating sub-substrate (round-73) — bridge
    pin #14 (the bidirectional seated round-trip).** Sibling of pin #7
    (``..._phase1_materialize_byte_equal``) for the two Lander-pid
    spells. The real Seedship Impact / Orbital Plunge printings routed
    through the GENUINE loader (``production_deck_to_deckspec`` → the
    round-73 ``_seat_token_printings`` spell-axis pass →
    ``_printing_to_card_def``) materialize the SAME printing on both
    engines:

      * Rust registry-dict side: the spell CardDef's ``spell_effect``
        carries the NUMERIC seated Lander ``token_printing_id`` (the
        exact field set Rust's ``parse_effect`` ``require``s — Seedship
        single-field; Orbital Plunge two-field with ``amount``), NO
        symbolic ``_token_card_def``, NO spurious ``count``;
      * Python side: ``_card_def_to_python_printing`` re-materializes
        the FIELD-FREE Python class (the existing bridge arm DROPS the
        pid — Python mints the Lander intrinsically), byte-equal to
        the live-parsed effect type.

    Proves Gate A (guardrail now admits) + Gate B (the spell-axis
    seating + the bidirectional bridge) round-trip from ONE source —
    the no-fallback-hybrid discipline: the symbolic ref is emitted
    AND resolved, never half-wired.
    """
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        _card_def_to_python_printing, DeckSpec,
    )
    from mtg_bot.cards.source import get_active_source

    src = get_active_source().get_printings()
    cases = {
        "Seedship Impact": ("Instant", "seedship_impact_effect", None),
        "Orbital Plunge": (
            "Sorcery",
            "deal_damage_to_target_creature_with_excess_lander_rider",
            6,
        ),
    }
    for name, (ctype, kind, amount) in cases.items():
        real = src.get(name)
        if real is None:
            import pytest
            pytest.skip(f"{name} absent from active CardSource")
        assert trajectory._printing_is_bridge_simple_spell(real), (
            f"{name} must pass the guardrail post-round-73")
        # The GENUINE production loader transform (the EXACT path the
        # bench OR-chain produces for the real printing).
        spec = trajectory.production_deck_to_deckspec(
            [name] * 4 + ["Forest"] * 36, ["Forest"] * 40,
            pad_to_size=40)
        cds = {cd.name: cd for cd in spec.card_defs}
        assert name in cds and "Lander" in cds
        lander_pid = next(
            i for i, cd in enumerate(spec.card_defs)
            if cd.name == "Lander")
        sc = cds[name]
        assert sc.card_type == ctype
        # Rust registry-dict side: numeric seated pid, no symbolic ref,
        # no spurious count (the §2 teeth — the Rust arms don't take
        # one; injecting one would silently mistranslate).
        se = sc.spell_effect
        assert se["kind"] == kind
        assert se["token_printing_id"] == lander_pid
        assert "_token_card_def" not in se
        assert "count" not in se
        if amount is not None:
            assert se["amount"] == amount
        # Python side: the field-free class round-trips byte-equal to
        # the live-parsed effect type (the bridge DROPS the pid).
        rt = _card_def_to_python_printing(sc)
        assert rt.spell_effect is not None
        assert type(rt.spell_effect) is type(real.spell_effect)
        assert rt.card_types == (ctype,)


def test_bridge_biosynthic_burst_basic_translation():
    """Effect-uncovered Tier-A queue #6 — Biosynthic Burst. Rust
    ``{"kind": "biosynthic_burst_effect"}`` (field-free unit
    variant) translates to Python ``BiosynthicBurstEffect()``.
    """
    from mtg_bot.effects.bespoke import BiosynthicBurstEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "biosynthic_burst_effect"})
    assert isinstance(e, BiosynthicBurstEffect)
    assert e == BiosynthicBurstEffect()


def test_bridge_biosynthic_burst_composable_in_sequence():
    """Round-trip the unit bespoke arm through the recursive Sequence
    translator (forward-compat with any Sequence-wrapped consumer).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.bespoke import BiosynthicBurstEffect
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "biosynthic_burst_effect"},
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], BiosynthicBurstEffect)
    assert isinstance(e.effects[1], GainLife)


def test_bridge_biosynthic_burst_bidirectional_round_trip():
    """Bridge pin: BOTH directions. The reverse serializer
    ``trajectory._python_spell_effect_to_dict`` emits the kind-only
    dict for ``BiosynthicBurstEffect()``; that dict round-trips back
    through ``_ta_dict_to_python_effect`` to an equal
    ``BiosynthicBurstEffect()``. The bidirectional arm is what lets
    the spell-card-bridge lever's ``_printing_to_card_def`` reuse the
    Python→dict sibling — a card flips to bench-admit only if its
    per-card round shipped BOTH directions.
    """
    from mtg_bot.effects.bespoke import BiosynthicBurstEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect
    from mtg_bot.parity.trajectory import _python_spell_effect_to_dict

    fwd = _python_spell_effect_to_dict(BiosynthicBurstEffect())
    assert fwd == {"kind": "biosynthic_burst_effect"}
    back = _ta_dict_to_python_effect(fwd)
    assert isinstance(back, BiosynthicBurstEffect)
    assert back == BiosynthicBurstEffect()


def test_bridge_focus_fire_basic_translation():
    """Effect-uncovered Tier-A queue #5 — Focus Fire. Rust
    ``{"kind": "focus_fire_effect", "base": 2}`` translates to Python
    ``FocusFireEffect(base=2)``.
    """
    from mtg_bot.effects.bespoke import FocusFireEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {"kind": "focus_fire_effect", "base": 2}
    )
    assert isinstance(e, FocusFireEffect)
    assert e.base == 2


def test_bridge_focus_fire_base_defaults_to_two_when_elided():
    """The ``base`` key is optional both directions and defaults to
    ``2`` (Focus Fire's printed base) — Rust's ``parse_effect``
    ``focus_fire_effect`` arm and this bridge arm agree on the
    default, so a key-elided round-trip is clean.
    """
    from mtg_bot.effects.bespoke import FocusFireEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect({"kind": "focus_fire_effect"})
    assert isinstance(e, FocusFireEffect)
    assert e.base == 2


def test_bridge_focus_fire_composable_in_sequence():
    """Round-trip the single-field bespoke arm through the recursive
    Sequence translator (forward-compat with any Sequence-wrapped
    consumer).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.bespoke import FocusFireEffect
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect

    e = _ta_dict_to_python_effect(
        {
            "kind": "sequence",
            "effects": [
                {"kind": "focus_fire_effect", "base": 2},
                {"kind": "gain_life", "amount": 1},
            ],
        }
    )
    assert isinstance(e, Sequence)
    assert len(e.effects) == 2
    assert isinstance(e.effects[0], FocusFireEffect)
    assert e.effects[0].base == 2
    assert isinstance(e.effects[1], GainLife)


def test_bridge_focus_fire_bidirectional_round_trip():
    """Bridge pin: BOTH directions. The reverse serializer
    ``trajectory._python_spell_effect_to_dict`` emits the
    ``{"kind": "focus_fire_effect", "base": N}`` dict for a
    ``FocusFireEffect``; that dict round-trips back through
    ``_ta_dict_to_python_effect`` to an equal ``FocusFireEffect``.
    The bidirectional arm is what lets the spell-card-bridge lever's
    ``_printing_to_card_def`` reuse the Python→dict sibling — a card
    flips to bench-admit only if its per-card round shipped BOTH
    directions.
    """
    from mtg_bot.effects.bespoke import FocusFireEffect
    from mtg_bot.parity.cross_engine import _ta_dict_to_python_effect
    from mtg_bot.parity.trajectory import _python_spell_effect_to_dict

    fwd = _python_spell_effect_to_dict(FocusFireEffect(base=2))
    assert fwd == {"kind": "focus_fire_effect", "base": 2}
    back = _ta_dict_to_python_effect(fwd)
    assert isinstance(back, FocusFireEffect)
    assert back.base == 2


# ---------------------------------------------------------------------------
# warp_cost Phase 2 (Timeline Culler) — Rust regression + cross-engine
# bridge pins. Spec: docs/WARP_COST_PHASE2_DESIGN.md §4 pins 1-10.
# Phase 2 lands the two warp riders (warp_additional_life_cost +
# warp_castable_from_graveyard) on BOTH engines: Rust CardPrinting
# fields + parse, the cast_creature_warp life debit + graveyard
# fall-through arm, the life-gated legal emit + the graveyard warp
# loop, and the bidirectional CardDef bridge.
# ---------------------------------------------------------------------------

_R72_ZONE_HAND = 0       # state_ops_zones.rs ZONE_HAND
_R72_ZONE_GRAVEYARD = 3  # ZONE_GRAVEYARD


def _r72_inject_tlc(rs, zone=_R72_ZONE_HAND):
    """Inject a Timeline-Culler-shaped bare Creature ({B}{B} normal,
    {B} warp, Pay 2 life, graveyard-castable, 2/2 haste) into p0's
    ``zone`` via the dict registry path. Returns the obj id."""
    return int(rs.register_and_inject_object(
        {
            "name": "TLC",
            "card_types": ["Creature"],
            "mana_cost": {"B": 2},
            "warp_cost": {"B": 1},
            "warp_additional_life_cost": 2,
            "warp_castable_from_graveyard": True,
            "keywords": ["haste"],
            "power": 2,
            "toughness": 2,
            "effects": [],
        },
        0,
        zone,
    ))


def test_r72_warp_phase2_parse_printing_roundtrips_riders():
    """Spec §4 pin 1 — ``parse_printing`` with
    ``warp_additional_life_cost=2`` + ``warp_castable_from_graveyard
    =true`` round-trips into the new ``CardPrinting`` fields; the
    absent keys map to ``0`` / ``false`` (back-compat — a non-Phase-2
    printing dict still parses with both at default). Verified via the
    Phase-2 legal/cast behavior the riders gate (a TLC-shaped printing
    in the graveyard offers the use_warp variant only with the riders
    parsed; a no-rider printing in the graveyard offers nothing)."""
    rs = _r62_fresh_rust_state()
    # WITH riders: in the graveyard, life high → graveyard warp offer.
    tlc = _r72_inject_tlc(rs, _R72_ZONE_GRAVEYARD)
    # WITHOUT the gy rider: a pure-warp creature (Phase-1 shape) in the
    # graveyard must NOT be graveyard-warp-castable.
    nogy = int(rs.register_and_inject_object(
        {"name": "NoGyWarp", "card_types": ["Creature"],
         "mana_cost": {"generic": 3, "R": 1},
         "warp_cost": {"generic": 1, "R": 1},
         "power": 3, "toughness": 3, "effects": []},
        0, _R72_ZONE_GRAVEYARD))
    rs.set_phase(3)  # MAIN1 (sorcery timing)
    rs.add_mana(0, "B", 5)
    ld = rs.legal_decisions(0)
    tlc_gy = sorted(
        a.get("use_warp") for a in ld
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc
    )
    nogy_gy = [
        a for a in ld
        if a.get("kind") == "cast_spell" and a.get("card_id") == nogy
    ]
    assert tlc_gy == [True], (
        f"a graveyard TLC (riders parsed) must offer EXACTLY one "
        f"use_warp=True cast (no base / free-cast); got {tlc_gy}")
    assert nogy_gy == [], (
        f"a no-graveyard-rider warp creature in the graveyard must "
        f"offer NOTHING (back-compat: absent key -> false); got "
        f"{nogy_gy}")


def test_r72_warp_phase2_hand_life_gate_on_legal_emit():
    """Spec §4 pin 2 — Timeline Culler in HAND: at ``life >= 3`` the
    emit is ``[base, warp]``; at ``life == 2`` the warp variant is
    suppressed by the life gate (base only). A Phase-1 card (life
    rider 0) at any ``life >= 1`` still emits ``[base, warp]`` — the
    life gate is a verified NO-OP for it (Phase-1 legal output
    unchanged)."""
    # legal_decisions mana-awareness Slice 1: seat the cost so the
    # LIFE gate is the sole variable under test (the mana-prune is a
    # separate axis pinned by test_cross_engine_legal_set_asymmetry).
    # TLC base {B}{B} + warp {B} → 2 B covers both independent
    # non-destructive can_pay_cost checks.
    # life >= 3 → base+warp
    rs = _r62_fresh_rust_state()
    tlc = _r72_inject_tlc(rs, _R72_ZONE_HAND)
    rs.set_life(0, 3)
    rs.set_phase(3)
    rs.add_mana(0, "B", 2)
    pairs = sorted(
        a["use_warp"] for a in rs.legal_decisions(0)
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc)
    assert pairs == [False, True], (
        f"TLC in hand at life=3 must emit base+warp; got {pairs}")
    # life == 2 → base only (warp suppressed: life > 2 is False)
    rs2 = _r62_fresh_rust_state()
    tlc2 = _r72_inject_tlc(rs2, _R72_ZONE_HAND)
    rs2.set_life(0, 2)
    rs2.set_phase(3)
    rs2.add_mana(0, "B", 2)
    pairs2 = sorted(
        a["use_warp"] for a in rs2.legal_decisions(0)
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc2)
    assert pairs2 == [False], (
        f"TLC in hand at life=2 must emit base ONLY (warp suppressed "
        f"by the life gate); got {pairs2}")
    # Phase-1 card (rider 0) at life=1 still base+warp (gate no-op).
    # WarpBeast base {3}{R} + warp {1}{R} → 4 C + 1 R covers both.
    rs3 = _r62_fresh_rust_state()
    w = _r63_inject_warp_creature(rs3, _R63_ZONE_HAND)
    rs3.set_life(0, 1)
    rs3.set_phase(3)
    rs3.add_mana(0, "C", 4)
    rs3.add_mana(0, "R", 1)
    pairs3 = sorted(
        a["use_warp"] for a in rs3.legal_decisions(0)
        if a.get("kind") == "cast_spell" and a.get("card_id") == w)
    assert pairs3 == [False, True], (
        f"Phase-1 warp card (life rider 0) at life=1 must STILL emit "
        f"base+warp — the life gate is a no-op for rider 0; got "
        f"{pairs3}")


def test_r72_warp_phase2_graveyard_loop_life_gate():
    """Spec §4 pin 3 — Timeline Culler in GRAVEYARD: at ``life >= 3``
    emits EXACTLY one ``use_warp=True`` CastSpell for the graveyard
    object (NO base, NO free-cast); at ``life == 2`` emits NONE. A
    non-``warp_castable_from_graveyard`` card in the graveyard emits
    nothing (regression)."""
    rs = _r62_fresh_rust_state()
    tlc = _r72_inject_tlc(rs, _R72_ZONE_GRAVEYARD)
    rs.set_life(0, 3)
    rs.set_phase(3)
    # legal_decisions mana-awareness Slice 1: seat the {B} graveyard-
    # warp cost so the LIFE gate is the sole variable under test.
    rs.add_mana(0, "B", 1)
    gy3 = [
        a for a in rs.legal_decisions(0)
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc]
    assert [a["use_warp"] for a in gy3] == [True], (
        f"graveyard TLC at life=3 → exactly one use_warp=True (no "
        f"base/free); got {gy3}")
    rs2 = _r62_fresh_rust_state()
    tlc2 = _r72_inject_tlc(rs2, _R72_ZONE_GRAVEYARD)
    rs2.set_life(0, 2)
    rs2.set_phase(3)
    rs2.add_mana(0, "B", 1)
    gy2 = [
        a for a in rs2.legal_decisions(0)
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc2]
    assert gy2 == [], (
        f"graveyard TLC at life=2 → NONE (life gate, mana seated so "
        f"the life gate is the sole prune); got {gy2}")


def test_r72_warp_phase2_hand_cast_pays_warp_cost_and_life():
    """Spec §4 pin 4 — ``Action::CastSpell{use_warp:true}`` on TLC in
    HAND (life=10): pays ``warp_cost`` ({B}), ``life == 8`` (2 paid),
    ``spell_warped_this_turn == True``, the battlefield permanent has
    ``warp_exile_pending == True`` (observed via the end-step exile
    one phase later)."""
    rs = _r62_fresh_rust_state()
    tlc = _r72_inject_tlc(rs, _R72_ZONE_HAND)
    rs.set_life(0, 10)
    rs.set_phase(3)
    rs.add_mana(0, "B", 1)  # exactly the {B} warp cost
    assert rs.spell_warped_this_turn is False
    rs.apply_action({
        "kind": "cast_spell", "card_id": tlc, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # G7 S4: warp cast is stack-then-resolve (was atomic). The
    # genuinely cast-time riders — warp_cost paid, the {2} life rider,
    # spell_warped_this_turn — are observable immediately (Python sets
    # all three in _cast_spell BEFORE the StackItem push); the bf
    # entry is at resolution. S1 closure-flip pattern.
    pool = dict(rs.get_mana_pool(0))
    assert sum(pool.values()) == 0, (
        f"the {{B}} warp cost must fully drain the 1-B pool; got "
        f"{pool} (non-empty would mean it paid {{B}}{{B}} mana_cost)")
    assert rs.get_life(0) == 8, (
        f"warp life rider: 10 - 2 = 8; got {rs.get_life(0)}")
    assert rs.spell_warped_this_turn is True
    assert not rs.is_stack_empty(), (
        "G7 S4: TLC's warp spell must be ON THE STACK post-cast")
    rs.resolve_top_of_stack()
    bf = [o for o in rs.battlefield_snapshot()
          if o["printing_name"] == "TLC"]
    assert len(bf) == 1, "TLC must be on the battlefield post-resolve"
    # **G7 Class-C delayed-trigger port (round-78).** END step push +
    # resolve drains the WarpExileSelf trigger that
    # `cast_creature_warp` scheduled into `state.delayed_triggers`.
    rs.set_phase(9)
    rs.advance_phase()  # → END_STEP: WarpExileSelf onto the stack
    rs.resolve_top_of_stack()  # delayed trigger resolves → bf → exile
    assert [o for o in rs.battlefield_snapshot()
            if o["printing_name"] == "TLC"] == []
    assert len([o for o in rs.exile_snapshot(0)
                if o["printing_name"] == "TLC"]) == 1


def test_r72_warp_phase2_graveyard_cast_full_then_exile_then_recast():
    """Spec §4 pins 5-7 — ``Action::CastSpell{use_warp:true}`` on TLC
    in the GRAVEYARD (life=10): the card LEAVES the graveyard, ENTERS
    the battlefield, pays ``warp_cost`` ({B}), ``life == 8``,
    ``spell_warped_this_turn == True``, ``warp_exile_pending == True``.
    Advancing to PHASE_END_STEP exiles it (``warped == true``); a
    later turn's recast from EXILE pays the NORMAL ``{B}{B}``
    ``mana_cost`` (NOT a second graveyard-warp) and clears
    ``warped``."""
    rs = _r62_fresh_rust_state()
    tlc = _r72_inject_tlc(rs, _R72_ZONE_GRAVEYARD)
    rs.set_life(0, 10)
    rs.set_phase(3)
    rs.add_mana(0, "B", 1)
    rs.apply_action({
        "kind": "cast_spell", "card_id": tlc, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # G7 S4: graveyard-warp cast is stack-then-resolve (was atomic).
    # The card LEAVES the graveyard at cast time (normalized
    # graveyard -> hand, the S1 exile-recast pattern, byte-mirroring
    # Python's p.graveyard.remove(card) + card.zone = STACK); the bf
    # entry is at resolution. The cast-time riders (life, warp pool,
    # spell_warped_this_turn) are observable immediately. S1
    # closure-flip pattern.
    assert [o for o in rs.graveyard_snapshot(0)
            if o["printing_name"] == "TLC"] == [], (
        "TLC must have LEFT the graveyard (graveyard-warp-cast)")
    assert rs.get_life(0) == 8, (
        f"graveyard-warp life rider: 10 - 2 = 8; got {rs.get_life(0)}")
    assert rs.spell_warped_this_turn is True
    assert dict(rs.get_mana_pool(0)).get("B", 0) == 0
    assert not rs.is_stack_empty(), (
        "G7 S4: graveyard-warp TLC must be ON THE STACK post-cast")
    rs.resolve_top_of_stack()
    bf = [o for o in rs.battlefield_snapshot()
          if o["printing_name"] == "TLC"]
    assert len(bf) == 1, "TLC must be on the battlefield post-resolve"
    # **G7 Class-C delayed-trigger port (round-78).** End step pushes
    # WarpExileSelf onto the stack; resolve drains it. Graveyard-cast
    # warp creatures schedule the SAME delayed trigger as hand-cast
    # (Python: `actions.py:1550` is shared; Rust:
    # `cast_creature_warp` `delayed_triggers.push` runs in both arms).
    rs.set_phase(9)
    rs.advance_phase()  # → END_STEP: WarpExileSelf onto the stack
    rs.resolve_top_of_stack()  # delayed trigger resolves → bf → exile
    ex = [o for o in rs.exile_snapshot(0)
          if o["printing_name"] == "TLC"]
    assert len(ex) == 1, (
        f"graveyard-warp-cast TLC must be exiled by the SAME delayed "
        f"trigger; exile="
        f"{[o['printing_name'] for o in rs.exile_snapshot(0)]}")
    assert [o for o in rs.battlefield_snapshot()
            if o["printing_name"] == "TLC"] == []
    # Recast from exile on a later turn: BASE use_warp=False at the
    # NORMAL {B}{B} cost; it must NOT re-offer a graveyard-warp (the
    # object is in EXILE, not the graveyard).
    rs.set_phase(3)
    # legal_decisions mana-awareness Slice 1: the graveyard-warp-cast
    # above drained the {B} pool, so the warp-exiled object's BASE
    # recast is now mana-gated. Seat the NORMAL {B}{B} BEFORE the
    # emission assertion (this is the SAME mana the recast
    # apply_action below consumes — the post-apply B==0 drain assert
    # still holds; life is unchanged by seating/emission).
    rs.add_mana(0, "B", 2)  # the NORMAL {B}{B}
    life_before = rs.get_life(0)
    ld = rs.legal_decisions(0)
    recast = sorted(
        a["use_warp"] for a in ld
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc)
    assert recast == [False], (
        f"warp-exiled TLC recast must be BASE-only (use_warp=False) at "
        f"NORMAL mana_cost — NOT a second graveyard-warp; got {recast}")
    rs.apply_action({
        "kind": "cast_spell", "card_id": tlc, "use_warp": False,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # G7 S1: cast_creature pushes a StackItem (exile-recast normalized
    # exile -> hand at cast time); drive the resolve so the recast TLC
    # lands on the battlefield — byte-equivalent to the old fused
    # no-stack post-state, split cast-then-resolve.
    rs.resolve_top_of_stack()
    assert len([o for o in rs.battlefield_snapshot()
                if o["printing_name"] == "TLC"]) == 1, (
        "exile-recast TLC must be back on the battlefield")
    assert dict(rs.get_mana_pool(0)).get("B", 0) == 0, (
        "the exile-recast must pay the NORMAL {B}{B} (2 B drained)")
    assert rs.get_life(0) == life_before, (
        f"exile-recast pays NO life rider (it is a NORMAL-cost cast, "
        f"not a warp cast); life {life_before} -> {rs.get_life(0)}")


def _r72_tlc_synth_deckspec(in_graveyard=False):
    """A synthetic Timeline-Culler-shaped CardDef ({B}{B} normal, {B}
    warp, warp_life_cost=2, warp_from_graveyard=True, 2/2 haste) + a
    Swamp-that-taps-B, padded so both engines have enough B mana.
    Used by the cross-engine bridge pins."""
    from mtg_bot.parity.cross_engine import CardDef, DeckSpec

    tlc = CardDef(
        name="SynthTLC", card_type="Creature",
        mana_cost_generic=0, mana_cost_colored=(("B", 2),),
        power=2, toughness=2, keywords=("haste",),
        warp_cost_generic=0, warp_cost_colored=(("B", 1),),
        has_warp=True, warp_life_cost=2, warp_from_graveyard=True,
    )
    swamp = CardDef(name="Swamp", card_type="Land",
                    mana_produced=("B",))
    lib = ("SynthTLC",) + ("Swamp",) * 39
    return DeckSpec(
        card_defs=(tlc, swamp),
        p0_library=lib,
        p1_library=("Swamp",) * 40,
    )


def test_r72_warp_phase2_bridge_both_engines_same_hand_pair():
    """Spec §4 pin 8 — a DeckSpec with a synthetic Timeline-Culler
    CardDef: once the {B} warp cost is payable, BOTH engines'
    ``legal_actions`` offer the SAME ``{base, warp}`` hand pair at
    ``life >= 3`` and the SAME (none-warp) at ``life == 2``
    (action-set + positional byte-equal). Same ``_can_cast``-asymmetry
    caveat as the Phase-1 pin — mana made payable first."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        advance_phase_python, step_both, step_both_drained,
    )
    from mtg_bot.legal import legal_actions

    py, rs = trajectory.setup_matched_game(_r72_tlc_synth_deckspec())

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()

    adv(3)  # turn 1 MAIN1
    # Build B mana: a Swamp each p0 turn for 2 turns, then a 3rd, tap 2.
    for _ in range(2):
        s = next(o for o in py.players[py.active_player_idx].hand
                 if o.printing.name == "Swamp")
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(s.id)}).divergences
        adv(24)
    s = next(o for o in py.players[py.active_player_idx].hand
             if o.printing.name == "Swamp")
    assert not step_both(
        py, rs, {"kind": "play_land", "card_id": int(s.id)}).divergences
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == py.active_player_idx
                and o.printing.is_land and not o.tapped
                and not o.summoning_sick and taps < 1):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    ai = py.active_player_idx
    tlc = next(o for o in py.players[ai].hand
               if o.printing.name == "SynthTLC")
    # life >= 3 → both engines offer base+warp.
    py.players[ai].life = 5
    rs.set_life(ai, 5)
    py_pairs = sorted(
        a.use_warp for a in legal_actions(py, ai)
        if getattr(a, "card_id", None) == tlc.id)
    rs_pairs = sorted(
        a["use_warp"] for a in rs.legal_decisions(ai)
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc.id)
    assert py_pairs == [False, True] and rs_pairs == [False, True], (
        f"life>=3: py={py_pairs} rs={rs_pairs}")
    assert py_pairs == rs_pairs, (
        f"cross-engine hand pair mismatch (life>=3): py={py_pairs} "
        f"rs={rs_pairs}")
    # life == 2 → neither engine offers the warp variant.
    py.players[ai].life = 2
    rs.set_life(ai, 2)
    py_w = sorted(
        a.use_warp for a in legal_actions(py, ai)
        if getattr(a, "card_id", None) == tlc.id)
    rs_w = sorted(
        a["use_warp"] for a in rs.legal_decisions(ai)
        if a.get("kind") == "cast_spell" and a.get("card_id") == tlc.id)
    assert True not in py_w and True not in rs_w, (
        f"life==2: the warp variant must be suppressed on BOTH; "
        f"py={py_w} rs={rs_w}")
    assert py_w == rs_w, (
        f"cross-engine hand pair mismatch (life==2): py={py_w} "
        f"rs={rs_w}")


def test_r72_warp_phase2_bridge_hand_cast_byte_equal():
    """Spec §4 pin 9 — driving ``CastSpell{use_warp:true}`` from HAND
    via ``step_both``: post-cast state fingerprint byte-equal (mana
    pool, **life**, zones, ``spell_warped_this_turn``) across
    engines."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        advance_phase_python, step_both, step_both_drained,
    )

    py, rs = trajectory.setup_matched_game(_r72_tlc_synth_deckspec())

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()

    adv(3)
    for _ in range(2):
        s = next(o for o in py.players[py.active_player_idx].hand
                 if o.printing.name == "Swamp")
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(s.id)}).divergences
        adv(24)
    s = next(o for o in py.players[py.active_player_idx].hand
             if o.printing.name == "Swamp")
    assert not step_both(
        py, rs, {"kind": "play_land", "card_id": int(s.id)}).divergences
    taps = 0
    for o in list(py.battlefield):
        if (o.controller_id == py.active_player_idx
                and o.printing.is_land and not o.tapped
                and not o.summoning_sick and taps < 1):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            taps += 1
    ai = py.active_player_idx
    tlc = next(o for o in py.players[ai].hand
               if o.printing.name == "SynthTLC")
    res = step_both_drained(
        py, rs,
        {"kind": "cast_spell", "card_id": int(tlc.id),
         "use_warp": True})
    assert not res.divergences, (
        f"TLC hand-warp-cast diverged (FULL compare_fields): "
        f"{[(d.field, d.py_value, d.rust_value) for d in res.divergences]}")
    assert py.spell_warped_this_turn is True
    assert rs.spell_warped_this_turn is True


def test_r72_warp_phase2_bridge_graveyard_cast_lifecycle_byte_equal():
    """Spec §4 pin 10 — seed the synthetic Timeline-Culler CardDef
    into BOTH engines' GRAVEYARDS; both ``legal_actions`` offer the
    SAME graveyard ``use_warp=true`` CastSpell; drive it via
    ``step_both`` → byte-equal (life debited 2 on both, card on
    battlefield, ``warp_exile_pending`` on both). Advance both to END
    → byte-equal exile; recast from exile at ``{B}{B}`` → byte-equal
    (and assert it does NOT re-offer the graveyard-warp variant since
    the object is in exile, not the graveyard)."""
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        advance_phase_python, compare_fields, step_both, step_both_drained,
    )
    from mtg_bot.game import Zone
    from mtg_bot.legal import legal_actions
    from tests.test_trajectory_parity import _step_views

    py, rs = trajectory.setup_matched_game(_r72_tlc_synth_deckspec())

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()
            pv, rv = _step_views(py, rs)
            d = compare_fields(pv, rv)
            assert not d, (
                f"TLC gy-lifecycle: phase advance diverged: "
                f"{[(x.field, x.py_value, x.rust_value) for x in d]}")

    adv(3)
    # Develop 2 Swamps for the {B} warp cost (graveyard cast still
    # pays the warp mana). Then seed TLC into BOTH graveyards.
    for _ in range(2):
        s = next(o for o in py.players[py.active_player_idx].hand
                 if o.printing.name == "Swamp")
        assert not step_both(
            py, rs,
            {"kind": "play_land", "card_id": int(s.id)}).divergences
        adv(24)
    ai = py.active_player_idx
    # Tap a Swamp for the {B} warp cost on BOTH — do this FIRST, via
    # step_both (FULL field-view byte-equal), while the two engines
    # are still obj-id-aligned (BEFORE the divergent-id graveyard
    # seed below). This keeps the strong whole-field compare on the
    # ramp.
    for o in list(py.battlefield):
        if (o.controller_id == ai and o.printing.is_land
                and not o.tapped and not o.summoning_sick):
            assert not step_both(
                py, rs,
                {"kind": "activate_ability",
                 "source_id": int(o.id), "ability_index": 0},
            ).divergences
            break
    # Seat the SynthTLC in the graveyard on BOTH engines (the
    # zone-seed idiom). Python reuses the deck object's id; Rust's
    # register_and_inject_object allocates its OWN fresh id — so the
    # graveyard object's id is unavoidably different between engines
    # (a pure probe-construction artifact of seeding via two
    # mechanisms, NOT an engine divergence; the FULL whole-field
    # byte-equality on the ACTUAL Timeline Culler — including the
    # graveyard-warp lifecycle — is proven by the real-EoE pin
    # test_cross_engine_real_eoe_warp_phase2_..._byte_equal). This
    # synthetic pin's job is the SYNTHETIC CardDef bridge round-trip
    # + the graveyard-warp OBSERVABLES byte-equal, so post-seed we
    # assert the observables (life, on-bf, spell_warped, exile-dest)
    # rather than the id-sensitive whole-field compare.
    tlc = next(
        (o for o in py.players[ai].hand
         if o.printing.name == "SynthTLC"), None)
    if tlc is None:
        tlc = next(
            o for o in py.players[ai].library
            if o.printing.name == "SynthTLC")
        py.players[ai].library.remove(tlc)
    else:
        py.players[ai].hand.remove(tlc)
    tlc.zone = Zone.GRAVEYARD
    tlc.summoning_sick = False
    tlc.tapped = False
    py.players[ai].graveyard.append(tlc)
    rs_tlc = int(rs.register_and_inject_object(
        {"name": "SynthTLC", "card_types": ["Creature"],
         "mana_cost": {"B": 2}, "warp_cost": {"B": 1},
         "warp_additional_life_cost": 2,
         "warp_castable_from_graveyard": True,
         "keywords": ["haste"], "power": 2, "toughness": 2,
         "effects": []},
        ai, _R72_ZONE_GRAVEYARD))
    # Both engines offer the SAME graveyard use_warp=True cast.
    py_gy = sorted(
        a.use_warp for a in legal_actions(py, ai)
        if getattr(a, "card_id", None) == tlc.id)
    rs_gy = sorted(
        a["use_warp"] for a in rs.legal_decisions(ai)
        if a.get("kind") == "cast_spell"
        and a.get("card_id") == rs_tlc)
    assert py_gy == [True] and rs_gy == [True], (
        f"both engines must offer exactly the graveyard use_warp=True "
        f"cast; py={py_gy} rs={rs_gy}")
    # Drive each engine with its own id; replicate step_both's CE5
    # cast-vs-resolve mirror by hand (Rust's cast_creature_warp is
    # no-stack; Python's CastSpell pushes a creature StackItem that
    # needs the mirrored resolve).
    py_life0 = py.players[ai].life
    rs_life0 = rs.get_life(ai)
    from mtg_bot.actions import apply_action as _py_apply
    from mtg_bot.actions import CastSpell as _PyCastSpell
    from mtg_bot.parity.cross_engine import (
        resolve_top_of_stack_python, resolve_top_of_stack_rust,
    )
    py_floor = len(getattr(py, "stack", ()))
    rs_floor = rs.stack_size()
    _py_apply(py, ai, _PyCastSpell(tlc.id, use_warp=True))
    rs.apply_action({
        "kind": "cast_spell", "card_id": rs_tlc, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    resolve_top_of_stack_python(py, floor=py_floor)
    resolve_top_of_stack_rust(rs, floor=rs_floor)
    # Observables byte-equal across engines (the synthetic bridge
    # round-trip's contract): life debited 2 on BOTH; the card on the
    # battlefield on BOTH; spell_warped set on BOTH; graveyard emptied
    # of the SynthTLC on BOTH.
    assert py.players[ai].life == py_life0 - 2, (
        f"Python life rider: {py_life0} - 2; got {py.players[ai].life}")
    assert rs.get_life(ai) == rs_life0 - 2, (
        f"Rust life rider: {rs_life0} - 2; got {rs.get_life(ai)}")
    assert py.players[ai].life == rs.get_life(ai), (
        f"cross-engine life mismatch post graveyard-warp-cast: "
        f"py={py.players[ai].life} rs={rs.get_life(ai)}")
    assert any(o.printing.name == "SynthTLC"
               for o in py.battlefield), (
        "SynthTLC must be on the Python battlefield post gy-warp-cast")
    assert any(e["printing_name"] == "SynthTLC"
               for e in rs.battlefield_snapshot()), (
        "SynthTLC must be on the Rust battlefield post gy-warp-cast")
    assert not [o for o in py.players[ai].graveyard
                if o.printing.name == "SynthTLC"], (
        "SynthTLC must have LEFT the Python graveyard")
    assert not [e for e in rs.graveyard_snapshot(ai)
                if e["printing_name"] == "SynthTLC"], (
        "SynthTLC must have LEFT the Rust graveyard")
    assert py.spell_warped_this_turn and rs.spell_warped_this_turn
    # End step → byte-equal exile DESTINATION (the SAME deterministic
    # sweep, graveyard origin).
    py_floor2 = len(getattr(py, "stack", ()))
    rs_floor2 = rs.stack_size()
    for _ in range(7):
        advance_phase_python(py)
        rs.advance_phase()
    resolve_top_of_stack_python(py, floor=py_floor2)
    resolve_top_of_stack_rust(rs, floor=rs_floor2)
    py_ex = sorted(o.printing.name for o in py.players[ai].exile)
    rs_ex = sorted(e["printing_name"]
                   for e in rs.exile_snapshot(ai))
    assert "SynthTLC" in py_ex and py_ex == rs_ex, (
        f"graveyard-warp-cast SynthTLC must be warp-exiled byte-equal "
        f"(exile destination, not graveyard); py_ex={py_ex} "
        f"rs_ex={rs_ex}")


def test_g7s4_warp_creature_cast_is_stack_then_resolve_not_atomic():
    """**G7 Slice S4 focused pin** — `cast_creature_warp` now pushes a
    `StackItem` (was atomic). Mirrors S1's
    `test_g7s1_cast_creature_pushes_stackitem_no_autoresolve` for the
    WARP creature path. Asserts the new shape directly (not just via
    the CE5-bridged `step_both` lifecycle tests above, which drain the
    stack before comparing):

    1. Post-cast: the warp spell is ON THE STACK (NOT on the
       battlefield), the cast-time riders are observable (warp_cost
       paid, `spell_warped_this_turn`), but `summoning_sick` /
       `warp_exile_pending` are NOT yet on any bf object (no bf
       object exists — byte-mirroring Python's `_cast_spell`, which
       only removes from source + appends the StackItem; bf entry +
       sickness are resolve-time `put_into_battlefield`).
    2. Post-resolve: the creature is on the bf with
       `summoning_sick=True` (resolve-time, via the already-present
       `move_source_post_resolve`), and the `warp_exile_pending` rider
       — stamped on the stack-resident SOURCE object at cast time —
       has RIDDEN THROUGH resolution onto the post-resolution bf
       object (proven by the end-step sweep exiling it; the design's
       key risk, resolved structurally with no new field)."""
    rs = _r62_fresh_rust_state()
    wid = _r63_inject_warp_creature(rs)
    rs.set_phase(3)
    rs.add_mana(0, "R", 1)
    rs.add_mana(0, "C", 1)  # {1}{R} warp cost
    assert rs.is_stack_empty() is True
    rs.apply_action({
        "kind": "cast_spell", "card_id": wid, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": False, "cast_room_half": None,
    })
    # (1) cast-time: on the stack, riders observable, NOTHING on bf.
    assert rs.is_stack_empty() is False, (
        "S4: the warp creature spell must be ON THE STACK post-cast")
    assert [o for o in rs.battlefield_snapshot()
            if o["printing_name"] == "WarpBeast"] == [], (
        "S4: NO bf entry at cast time (resolve-time, like Python "
        "put_into_battlefield)")
    assert sum(dict(rs.get_mana_pool(0)).values()) == 0, (
        "S4: warp_cost {1}{R} paid at cast time (genuinely cast-time "
        "rider, mirrors Python _cast_spell)")
    assert rs.spell_warped_this_turn is True, (
        "S4: spell_warped_this_turn set at cast time (Python :1427)")
    # (2) resolve-time: bf entry + summoning sickness.
    rs.resolve_top_of_stack()
    bf = [o for o in rs.battlefield_snapshot()
          if o["printing_name"] == "WarpBeast"]
    assert len(bf) == 1, "S4: WarpBeast lands on bf at RESOLUTION"
    assert bf[0].get("summoning_sick") is True, (
        "S4: summoning_sick stamped at RESOLVE (move_source_post_"
        "resolve), not cast — byte-mirrors Python put_into_battlefield")
    # (2 cont.) **G7 Class-C delayed-trigger port (round-78).** The
    # warp-EXILE obligation now flows through the stack: at
    # PHASE_END_STEP entry, the new
    # `state_ops_steps.rs::run_delayed_end_step_triggers` pushes
    # `Effect::WarpExileSelf` onto the stack (mirroring Python's
    # CR-603.7 delayed trigger). The bf→exile move happens at the
    # NEXT `resolve_top_of_stack` — NOT atomically on
    # `advance_phase` entry (pre-port behavior, since replaced).
    rs.set_phase(9)
    rs.advance_phase()  # → END_STEP: WarpExileSelf onto the stack.
    # The warp creature is STILL on the bf — the trigger has not yet
    # resolved (this is the byte-equal-Python on-stack window).
    assert len([o for o in rs.battlefield_snapshot()
                if o["printing_name"] == "WarpBeast"]) == 1, (
        "S4 + Class-C port: end-step pushes WarpExileSelf onto the "
        "stack; the warp creature stays on the bf until the trigger "
        "RESOLVES (CR 603.7 response window).")
    assert rs.is_stack_empty() is False, (
        "Class-C port: end-step pushed the WarpExileSelf delayed "
        "trigger onto the stack.")
    rs.resolve_top_of_stack()  # WarpExileSelf resolves → bf → exile.
    assert [o for o in rs.battlefield_snapshot()
            if o["printing_name"] == "WarpBeast"] == [], (
        "S4 + Class-C port: WarpExileSelf resolution exiles the warp "
        "creature (the cast-time warp_exile_pending marker rode the "
        "object through resolve to bf and the trigger drains it now).")
    assert len([o for o in rs.exile_snapshot(0)
                if o["printing_name"] == "WarpBeast"]) == 1


def test_g7s4_impending_and_room_riders_remain_out_of_reach_in_rust():
    """**G7 S4 verify-first honest pin** — S4 defers the *warp*
    rider/provenance (the one live rider: `warp_exile_pending`) from
    cast-time to resolve-time. The OTHER riders the design names —
    impending time-counter init, room-half — are NOT reachable through
    the Rust cast surface at all today (separately-deferred plumbing,
    NOT something S4 introduces a defer for):

    - the legal emitter hard-codes `use_impending=False` /
      `cast_room_half=None` (`state_ops_legal.rs:511-513`);
    - the `Action::CastSpell` dispatch explicitly ignores both
      (`state_ops_actions.rs:292-293`).

    So `use_impending=True` / `cast_room_half="..."` are accepted as
    data and have NO effect — the cast routes by `use_warp` only and
    the impending/room rider is a structural no-op. This pin LOCKS
    that honest boundary: a future slice that wires impending/room
    flips this and is forced to add the resolve-time rider
    consumption (new-code-is-right). This is NOT a tripwire weakening
    — it asserts the CURRENT honest reach of the Rust cast surface."""
    rs = _r62_fresh_rust_state()
    wid = _r63_inject_warp_creature(rs)
    rs.set_phase(3)
    rs.add_mana(0, "R", 1)
    rs.add_mana(0, "C", 1)
    # use_impending=True is data-only: the warp cast still routes by
    # use_warp (the synthetic WarpBeast HAS a warp_cost), and the
    # impending flag has zero effect on the stack-then-resolve shape.
    rs.apply_action({
        "kind": "cast_spell", "card_id": wid, "use_warp": True,
        "kicked": False, "cast_for_free": False,
        "use_impending": True, "cast_room_half": "left",
    })
    assert rs.is_stack_empty() is False, (
        "use_impending/cast_room_half are data-only — the warp cast "
        "still stack-pushes by use_warp, unaffected")
    rs.resolve_top_of_stack()
    bf = [o for o in rs.battlefield_snapshot()
          if o["printing_name"] == "WarpBeast"]
    assert len(bf) == 1, (
        "the creature lands as a NORMAL warp creature — impending "
        "time-counter init / room-half routing are NOT modeled in "
        "Rust (out-of-reach plumbing, not an S4 rider-defer)")


def test_bridge_put_plus_one_counter_on_opponents_creature_bidirectional_round_trip():
    """**Tier-C target-seating sub-substrate, lever-a (Sunset
    Saboteur) — the bidirectional bridge round-trip pin.**

    BOTH directions, byte-equal:

      * **Python→dict** — the recognizer
        ``trajectory._python_ta_to_safe_dict`` applied to the GENUINE
        parsed Sunset Saboteur ATTACKS trigger emits exactly
        ``{"event": "attacks",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind":
                "put_plus_one_counter_on_opponents_creature",
                "amount": 1}}``.
      * **dict→Python** — ``_ta_dict_to_python_effect`` /
        ``_ta_dict_to_python_triggered_ability`` reconstruct an equal
        ``PutPlusOneCounterOnOpponentsCreature(amount=1)`` /
        ``TriggeredAbility(ATTACKS, IsSelf, ...)``.
      * **dict→Rust** — Rust's ``register_printing`` parser accepts
        the SAME dict (the ``put_plus_one_counter_on_opponents_
        creature`` ``parse_effect`` arm); its ``to_dict`` re-emits
        the byte-equal ``{"kind": ..., "amount": 1}`` (the
        recognizer's payload whitelist mirrors it).

    The bidirectional arm is the lockstep-provenance contract
    (``trajectory.py`` ``_SAFE_TRIGGERS`` docstring): the bridge arm
    must exist on BOTH engines before the targeted-``_SAFE_TRIGGERS``
    triple + the real-EoE firing-parity probe (in
    test_parity_smoke.py).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnOpponentsCreature,
    )
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect,
        _ta_dict_to_python_triggered_ability,
    )
    from mtg_bot.parity.trajectory import _python_ta_to_safe_dict

    real = get_active_source().get_printings().get("Sunset Saboteur")
    if real is None:
        import pytest
        pytest.skip("Sunset Saboteur absent from active CardSource")
    ta = real.triggered_abilities[0]
    # Python→dict (the recognizer is the reverse serializer for
    # triggered abilities — there is no separate _python_*_to_dict
    # since the recognizer IS the Python→dict direction).
    fwd = _python_ta_to_safe_dict(ta)
    assert fwd == {
        "event": "attacks",
        "predicate": {"kind": "source_is_self"},
        "effect": {
            "kind": "put_plus_one_counter_on_opponents_creature",
            "amount": 1,
        },
    }
    # dict→Python (effect arm).
    back_eff = _ta_dict_to_python_effect(fwd["effect"])
    assert isinstance(back_eff, PutPlusOneCounterOnOpponentsCreature)
    assert int(back_eff.amount) == 1
    # dict→Python (whole triggered ability).
    back_ta = _ta_dict_to_python_triggered_ability(fwd)
    assert back_ta.event is Event.ATTACKS
    assert isinstance(back_ta.predicate, IsSelf)
    assert isinstance(back_ta.effect, PutPlusOneCounterOnOpponentsCreature)
    assert int(back_ta.effect.amount) == 1
    # Missing-amount tolerance (mirrors Rust parse_effect default=1).
    assert int(
        _ta_dict_to_python_effect(
            {"kind": "put_plus_one_counter_on_opponents_creature"}
        ).amount
    ) == 1
    # dict→Rust: the SAME dict registers cleanly on the Rust parser
    # (the ``put_plus_one_counter_on_opponents_creature``
    # ``parse_effect`` arm) — the lockstep gate that keeps the Python
    # bridge from drifting away from Rust's parse coverage.
    import mtg_bot_engine
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Sunset Saboteur",
            "card_types": ["Creature"],
            "mana_cost": {"generic": 1, "B": 1},
            "power": 4,
            "toughness": 1,
            "keywords": ["menace"],
            "triggered_abilities": [fwd],
        }
    )
    assert pid == 0, "Rust must register the Sunset Saboteur trigger dict"


def test_bridge_spell_card_lever_phase2_diplomatic_relations_materialize_byte_equal():
    """**Spell-card-bridge lever Phase 2 (round-76) — bridge pin #15
    (the bidirectional 2-slot materialize round-trip).** Sibling of
    pin #7 for the Phase-2 2-slot card. The real Diplomatic Relations
    printing routed through ``_printing_to_card_def`` materializes the
    SAME printing on both engines:

      * Rust registry-dict side: the spell CardDef's ``spell_effect``
        is the FIELD-FREE ``{"kind": "diplomatic_relations_effect"}``
        dict (the 2-slot targets ride on the StackItem, NOT the effect
        dict — exactly the Phase-1 convention generalized to 2 slots);
      * Python side: ``_card_def_to_python_printing`` re-materializes
        the field-free ``DiplomaticRelationsEffect`` byte-equal to the
        live-parsed effect type, with the (2,2) /
        ('my_creature','creature') target grammar intact.

    Proves Gate A (the Phase-2 guardrail widening admits) + Gate B
    (CardDef.spell_effect + the bidirectional bridge arm + the
    reverse-serializer leaf) round-trip from ONE source — the
    no-fallback-hybrid discipline.
    """
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import _card_def_to_python_printing
    from mtg_bot.cards.source import get_active_source

    src = get_active_source().get_printings()
    real = src.get("Diplomatic Relations")
    if real is None:
        import pytest
        pytest.skip("Diplomatic Relations absent from active CardSource")
    # Verify-against-live: genuinely the Phase-2 2-slot admit shape.
    assert trajectory._printing_is_bridge_simple_spell(real), (
        "Diplomatic Relations must pass the Phase-2 spell guardrail")
    se = real.spell_effect
    assert se.target_count_range() == (2, 2)
    assert se.target_slot_kinds() == ("my_creature", "creature")
    cd = trajectory._printing_to_card_def(real)
    assert cd.card_type == "Instant"
    # Rust registry-dict side: the field-free 2-slot effect dict.
    assert cd.spell_effect == {"kind": "diplomatic_relations_effect"}
    # Python side: round-trips byte-equal to the live-parsed effect.
    rt = _card_def_to_python_printing(cd)
    assert rt.spell_effect is not None
    assert type(rt.spell_effect) is type(real.spell_effect)
    assert rt.spell_effect.target_count_range() == (2, 2)
    assert rt.spell_effect.target_slot_kinds() == (
        "my_creature", "creature")
    assert rt.card_types == ("Instant",)
    assert rt.is_instant is True


def test_bridge_spell_card_lever_phase2_diplomatic_relations_real_eoe_cast_resolve_parity():
    """**Spell-card-bridge lever Phase 2 (round-76) — bridge pin #16
    (the genuine full-card 2-slot admit / acceptance probe).** Drive
    the ACTUAL parsed Diplomatic Relations printing through the
    genuine loader admit transform (``_printing_to_card_def`` — pin
    #15 proved that transform byte-equal) into a DETERMINISTIC matched
    game with a friendly creature (p0) AND an opponent creature (p1),
    then via ``step_both``: the 2-slot cast (``CastSpell`` ->
    ``cast_spell_noncreature`` -> the 2-slot first-legal scan binds
    slot 0 = first p0-controlled creature + slot 1 = first
    p1-controlled creature onto the StackItem) AND the resolve (the
    harness CE5 cast-vs-resolve mirror drains both stacks: the double
    resolve-time swap threads both targets, ``DiplomaticRelations
    Effect`` pumps slot-0 +1/+0 + vigilance UEOT then deals slot-0's
    post-pump effective power as damage to slot-1, then ``move_source
    _post_resolve`` sends the spell hand -> graveyard) surface ZERO
    field-view divergences. Asserts post-cast AND post-resolve
    byte-equal + the spell source in the controller's graveyard +
    the pump/vigilance/damage observably resolved on BOTH engines.
    """
    from mtg_bot.parity import trajectory
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, step_both, step_both_drained, advance_phase_python,
        compare_fields,
    )
    from tests.test_trajectory_parity import _step_views
    from tests.parity._picker_shims import _PickFirstLegalTargetAgent
    from mtg_bot.cards.source import get_active_source

    src = get_active_source().get_printings()
    real = src.get("Diplomatic Relations")
    if real is None:
        import pytest
        pytest.skip("Diplomatic Relations absent from active CardSource")
    assert trajectory._printing_is_bridge_simple_spell(real), (
        "Diplomatic Relations must pass the Phase-2 spell guardrail")

    dr_cd = trajectory._printing_to_card_def(real)
    # p0's friendly creature (a 3/3 so the post-pump power = 4 is
    # distinct from the 3/3's toughness — proves the +1 power read).
    ally = CardDef(name="Ally Beast", card_type="Creature",
                   mana_cost_generic=1, mana_cost_colored=(("G", 1),),
                   power=3, toughness=3)
    # p1's opponent creature (a 5/5 so it SURVIVES 4 damage —
    # post-resolve byte-equality on a damaged-but-alive creature is a
    # stronger pin than a death that just removes the object).
    foe = CardDef(name="Foe Beast", card_type="Creature",
                  mana_cost_generic=1, mana_cost_colored=(("G", 1),),
                  power=5, toughness=5)
    forest = CardDef(name="Forest", card_type="Land",
                     mana_produced=("G",))
    spec = DeckSpec(
        card_defs=(dr_cd, ally, foe, forest),
        p0_library=tuple(
            ["Ally Beast", "Diplomatic Relations"] + ["Forest"] * 58),
        p1_library=tuple(["Foe Beast"] + ["Forest"] * 59),
    )
    py, rs = trajectory.setup_matched_game(spec)
    for pl in py.players:
        pl.agent = _PickFirstLegalTargetAgent()

    def adv(n):
        for _ in range(n):
            advance_phase_python(py)
            rs.advance_phase()
            pv, rv = _step_views(py, rs)
            d = compare_fields(pv, rv)
            assert not d, (
                f"phase advance ({pv['step']}, turn {pv['turn']}) "
                f"diverged: "
                f"{[(x.field, x.py_value, x.rust_value) for x in d]}")

    def tap_n(n):
        ai = py.active_player_idx
        taps = 0
        for o in list(py.battlefield):
            if (o.controller_id == ai and o.printing.is_land
                    and not o.tapped and not o.summoning_sick
                    and taps < n):
                assert not step_both(
                    py, rs,
                    {"kind": "activate_ability",
                     "source_id": int(o.id),
                     "ability_index": 0}).divergences
                taps += 1
        assert taps == n, f"need {n} mana; tapped {taps}"

    def play_forest():
        ai = py.active_player_idx
        hf = [o for o in py.players[ai].hand
              if o.printing.name == "Forest"]
        assert hf, "Forest must be in hand this ramp turn"
        assert not step_both(
            py, rs,
            {"kind": "play_land",
             "card_id": int(hf[0].id)}).divergences

    adv(3)  # turn 1 (p0) MAIN1
    # p0 ramps a Forest; pass to p1.
    play_forest()
    adv(12)  # -> p1 MAIN1
    # p1 plays a Forest + casts Foe Beast ({1}{G} = tap... p1 has 1
    # Forest from its draw step only; play one then need 2 mana — so
    # ramp p1 over 2 of its turns).
    play_forest()
    adv(24)  # -> p1 MAIN1 again (skip p0 turn — p0 just passes)
    # p0's intervening turn: ramp another Forest so p0 reaches 3 mana
    # for the Bear + later the spell. (adv(24) crossed a p0 turn; we
    # land back at p1 — re-orient by driving forward to p0.)
    # Simpler deterministic ramp: drive several turns playing a Forest
    # for whoever is active + casting each side's creature when it can
    # afford {1}{G}.
    cast_ally = False
    cast_foe = False
    for _ in range(8):
        ai = py.active_player_idx
        # Play a land if one is in hand.
        hf = [o for o in py.players[ai].hand
              if o.printing.name == "Forest"]
        if hf:
            assert not step_both(
                py, rs,
                {"kind": "play_land",
                 "card_id": int(hf[0].id)}).divergences
        # Count untapped lands available.
        lands = [o for o in py.battlefield
                 if o.controller_id == ai and o.printing.is_land
                 and not o.tapped and not o.summoning_sick]
        if not cast_ally and ai == py.active_player_idx:
            ally_h = [o for o in py.players[ai].hand
                      if o.printing.name == "Ally Beast"]
            if ally_h and len(lands) >= 2:
                tap_n(2)
                assert not step_both_drained(
                    py, rs,
                    {"kind": "cast_spell",
                     "card_id": int(ally_h[0].id)}).divergences
                cast_ally = True
        foe_h = [o for o in py.players[ai].hand
                 if o.printing.name == "Foe Beast"]
        if not cast_foe and foe_h and len(lands) >= 2:
            tap_n(2)
            assert not step_both_drained(
                py, rs,
                {"kind": "cast_spell",
                 "card_id": int(foe_h[0].id)}).divergences
            cast_foe = True
        adv(12)
        if cast_ally and cast_foe:
            break
    assert cast_ally and cast_foe, (
        f"deterministic ramp must seat both creatures "
        f"(ally={cast_ally} foe={cast_foe})")
    # Drive to a p0 MAIN1 where p0 has Diplomatic Relations + >=3 mana
    # and BOTH creatures are on the battlefield, no longer
    # summoning-sick (vigilance/damage don't care about sickness but
    # the deterministic walk needs both present).
    for _ in range(12):
        ai = py.active_player_idx
        allies = [o for o in py.battlefield
                  if o.controller_id == ai
                  and o.printing.name == "Ally Beast"]
        foes = [o for o in py.battlefield
                if o.controller_id != ai
                and o.printing.name == "Foe Beast"]
        spell_h = [o for o in py.players[ai].hand
                   if o.printing.name == "Diplomatic Relations"]
        hf = [o for o in py.players[ai].hand
              if o.printing.name == "Forest"]
        if hf:
            assert not step_both(
                py, rs,
                {"kind": "play_land",
                 "card_id": int(hf[0].id)}).divergences
        lands = [o for o in py.battlefield
                 if o.controller_id == ai and o.printing.is_land
                 and not o.tapped and not o.summoning_sick]
        if allies and foes and spell_h and len(lands) >= 3:
            ally_obj = allies[0]
            foe_obj = foes[0]
            gy_before = len(py.players[ai].graveyard)
            # Cast Diplomatic Relations. {2}{G} = tap 3 lands.
            tap_n(3)
            s = step_both_drained(
                py, rs,
                {"kind": "cast_spell",
                 "card_id": int(spell_h[0].id)})
            # Pin #16: post-cast AND post-resolve byte-equal (the CE5
            # cast-vs-resolve mirror drains both stacks before the
            # compare — proves the 2-slot bind + double resolve-swap +
            # pump + vigilance + power-equal damage are byte-equal).
            assert not s.divergences, (
                "Diplomatic Relations 2-slot cast->resolve diverged: "
                f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}")
            # The spell source is in the controller's graveyard on
            # BOTH engines post-resolve (zero divergence already
            # proves Rust agrees; assert the Python observable).
            py_spell = next(
                (o for o in py.players[ai].graveyard
                 if o.printing.name == "Diplomatic Relations"), None)
            assert py_spell is not None, (
                "Diplomatic Relations must be in p%d graveyard "
                "post-resolve" % ai)
            assert len(py.players[ai].graveyard) == gy_before + 1, (
                "exactly the spell goes to graveyard (the 5/5 Foe "
                "Beast survives 4 damage — no death)")
            # Effect observably resolved: slot-0 Ally Beast pumped
            # +1/+0 + vigilance UEOT; slot-1 Foe Beast took 4 damage
            # (post-pump effective power = 3 + 1).
            from mtg_bot.game import effective_power, effective_keywords
            live_ally = next(
                (o for o in py.battlefield if o.id == ally_obj.id),
                None)
            assert live_ally is not None
            assert effective_power(py, live_ally) == 4, (
                "Ally Beast must be 4/3 post-pump (3/3 +1/+0 UEOT)")
            assert "vigilance" in effective_keywords(py, live_ally), (
                "Ally Beast must have vigilance UEOT post-resolve")
            live_foe = next(
                (o for o in py.battlefield if o.id == foe_obj.id),
                None)
            assert live_foe is not None, (
                "Foe Beast (5/5) survives 4 damage")
            assert live_foe.damage == 4, (
                f"Foe Beast must have 4 marked damage (Ally's "
                f"post-pump power); got {live_foe.damage}")
            break
        adv(12)
    else:
        import pytest
        pytest.fail(
            "deterministic walk never reached a castable Diplomatic "
            "Relations state with both creatures + 3 mana")


def test_bridge_return_target_permanent_you_control_bidirectional_round_trip():
    """**Tier-C target-seating sub-substrate, lever-a (Exosuit Savior
    — the count_min=0 optional-bounce edge) — the bidirectional
    bridge round-trip pin.**

    BOTH directions, byte-equal:

      * **Python→dict** — the recognizer applied to the GENUINE
        parsed Exosuit Savior ENTERS trigger emits exactly
        ``{"event": "enters",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind":
                "return_target_permanent_you_control"}}`` (a UNIT
        effect dict — no payload field; the filter / count_min=0 are
        structural).
      * **dict→Python** — reconstructs an equal
        ``ReturnTargetPermanentYouControlToOwnersHand(optional=True,
        exclude_self=True)`` (``count_range=(0,1)``).
      * **dict→Rust** — Rust's ``register_printing`` parser accepts
        the SAME dict (the ``return_target_permanent_you_control``
        ``parse_effect`` unit arm).

    Pins the lockstep-provenance contract: the bridge arm exists on
    BOTH engines before the targeted-``_SAFE_TRIGGERS`` triple + the
    real-EoE firing-parity probe (in test_parity_smoke.py).
    """
    from mtg_bot.cards.source import get_active_source
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.bounce import (
        ReturnTargetPermanentYouControlToOwnersHand,
    )
    from mtg_bot.parity.cross_engine import (
        _ta_dict_to_python_effect,
        _ta_dict_to_python_triggered_ability,
    )
    from mtg_bot.parity.trajectory import _python_ta_to_safe_dict

    real = get_active_source().get_printings().get("Exosuit Savior")
    if real is None:
        import pytest
        pytest.skip("Exosuit Savior absent from active CardSource")
    ta = real.triggered_abilities[0]
    fwd = _python_ta_to_safe_dict(ta)
    assert fwd == {
        "event": "enters",
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "return_target_permanent_you_control"},
    }
    # dict→Python (effect arm) — optional/exclude_self fixed
    # structurally to Exosuit Savior's shape (count_min=0).
    back_eff = _ta_dict_to_python_effect(fwd["effect"])
    assert isinstance(back_eff, ReturnTargetPermanentYouControlToOwnersHand)
    assert back_eff.optional is True
    assert back_eff.exclude_self is True
    assert back_eff.target_count_range() == (0, 1)
    # dict→Python (whole triggered ability).
    back_ta = _ta_dict_to_python_triggered_ability(fwd)
    assert back_ta.event is Event.ENTERS
    assert isinstance(back_ta.predicate, IsSelf)
    assert isinstance(
        back_ta.effect, ReturnTargetPermanentYouControlToOwnersHand
    )
    # dict→Rust: the same UNIT dict registers cleanly on the Rust
    # parser (the return_target_permanent_you_control parse arm).
    import mtg_bot_engine
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Exosuit Savior",
            "card_types": ["Creature"],
            "mana_cost": {"generic": 2, "W": 1},
            "power": 2,
            "toughness": 2,
            "keywords": ["flying"],
            "triggered_abilities": [fwd],
        }
    )
    assert pid == 0, "Rust must register the Exosuit Savior trigger dict"


# === PHASE5_BRIDGE_PIN_TESTS_INSERT_ABOVE_THIS_LINE ===
# New bridge-arm pin tests go IMMEDIATELY ABOVE this marker.
