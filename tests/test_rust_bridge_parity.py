"""Parity tests: drive the Python <-> Rust card bridge with cards
the *existing* Python parser produces, not hand-crafted dict literals.

Path A is taken: **Bombard** (EoE ``{2}{R}`` instant — "Bombard deals
4 damage to target creature.") parses to
``DealDamageToTargetCreature(amount=4, targets=(TargetSpec(IsCreature()),))``,
which fits the Rust bridge's first-slice vocabulary cleanly:

- Effect is ``DealDamage`` (the sugar subclass inherits it).
- Filter atom is bare ``IsCreature()`` -> Rust ``AnyCreature``.
- Cost ``{2}{R}`` -> ``{generic: 2, R: 1}``, no hybrid / X.

The bridge is exercised end-to-end: real Scryfall JSON -> production
parser -> ``CardPrinting`` -> ``tests/_rust_bridge_serialize.py``
-> ``Registry.register_printing`` -> ``Registry`` getters -> equality
on the dict shape we sent in. Any drift in either side surfaces here.

Test 2 verifies the serializer's rejection path is wired up so future
Rust-vocabulary additions surface as a NotImplementedError on the
real card that needs them, instead of silently mismatching.
"""

from __future__ import annotations

import pytest

mtg_bot_engine = pytest.importorskip(
    "mtg_bot_engine",
    reason="rust extension not built; run `maturin develop --release -m rust/Cargo.toml`",
)

from mtg_bot.cards.base import CardPrinting, ManaCost  # noqa: E402
from mtg_bot.cards.source import get_active_source  # noqa: E402
from mtg_bot.effects.generic.damage import (  # noqa: E402
    DealDamage,
    DealDamageToTargetCreature,
    DealNDamageToEachCreature,
)
from mtg_bot.effects.object_predicates import IsCreature, IsPlayer  # noqa: E402
from mtg_bot.effects.targets import TargetSpec  # noqa: E402

from tests._rust_bridge_serialize import (  # noqa: E402
    serialize_effect,
    serialize_mana_cost,
    serialize_printing,
    serialize_target_filter,
)


# ---------------------------------------------------------------------------
# Test 1 — real EoE card round-trip (path A: Bombard)
# ---------------------------------------------------------------------------


def test_bombard_round_trips_through_rust_registry():
    """Real EoE Bombard, parsed by the production parser, survives
    a serialize -> register -> getter round-trip with structural
    equality on the dict shape."""
    prints = get_active_source().get_printings()
    bombard = prints["Bombard"]
    # Sanity-check the parser's output matches what the bridge can ingest.
    # If a future parser change re-shapes Bombard (e.g. adds a rider),
    # this assertion + the round-trip below catch it loudly.
    assert isinstance(bombard.spell_effect, DealDamage)
    assert isinstance(bombard.spell_effect.targets[0].filter, IsCreature)

    payload = serialize_printing(bombard)
    assert payload == {
        "name": "Bombard",
        "mana_cost": {"generic": 2, "R": 1},
        "effects": [
            {
                "kind": "deal_damage",
                "amount": 4,
                "target": {"kind": "any_creature"},
            }
        ],
    }

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(payload)
    assert reg.get_printing_name(pid) == "Bombard"
    assert dict(reg.get_printing_mana_cost(pid)) == {"generic": 2, "R": 1}
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert len(effects) == 1
    e = effects[0]
    assert e["kind"] == "deal_damage"
    assert e["amount"] == 4
    assert dict(e["target"]) == {"kind": "any_creature"}


# ---------------------------------------------------------------------------
# Test 2 — rejection path: serializer raises on out-of-vocabulary effects
# ---------------------------------------------------------------------------


def test_serializer_rejects_unsupported_effect_type():
    """Sanity-check the rejection path: an effect type the Rust
    bridge has no variant for must surface as NotImplementedError
    at serialize time, not silently drop or mismatch at register
    time. ``DealNDamageToEachCreature`` (untargeted, hits every
    creature on the battlefield) is a real engine effect that Rust
    has no slot for today — it'll need its own variant when the
    Rust enum grows."""
    eff = DealNDamageToEachCreature(amount=2)
    with pytest.raises(NotImplementedError, match="DealNDamageToEachCreature"):
        serialize_effect(eff)


def test_serializer_rejects_oneof_filter():
    """A real ``DealDamage`` constructed with the "any target"
    expansion (``OneOf(IsCreature, IsPlaneswalker, IsBattle,
    IsPlayer)``) is out of scope for the first-slice vocabulary
    and must reject loudly. This is the shape ``DealDamageAnyTarget``
    emits — locking the rejection in here means whoever ports
    ``OneOf`` to Rust will see their parity test go from raise to
    pass with a single line change."""
    from mtg_bot.effects.targets import any_target_filter

    eff = DealDamage(amount=3, targets=(TargetSpec(any_target_filter()),))
    with pytest.raises(NotImplementedError, match="OneOf"):
        serialize_effect(eff)


def test_serializer_rejects_x_cost():
    """X-cost spells have no Rust slot for ``x_pips``; serializer
    must reject rather than drop it on the floor."""
    cost = ManaCost(generic=0, colored=(("R", 1),), x_pips=1)
    with pytest.raises(NotImplementedError, match="X-cost"):
        serialize_mana_cost(cost)


# ---------------------------------------------------------------------------
# Test 3 — independent (mana_cost / effect) round-trip symmetry
# ---------------------------------------------------------------------------


def test_mana_cost_round_trip_is_symmetric():
    """Serialize a multi-color ``ManaCost`` and confirm the dict
    survives the Rust register-and-emit pipeline byte-for-byte.
    Locks in: zero-count entries are dropped on both sides, and
    the W/U/B/R/G key spelling is shared."""
    cost = ManaCost(
        generic=4,
        colored=(("U", 1), ("R", 1)),
    )
    cost_dict = serialize_mana_cost(cost)
    assert cost_dict == {"generic": 4, "U": 1, "R": 1}

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing({"name": "probe", "mana_cost": cost_dict})
    assert dict(reg.get_printing_mana_cost(pid)) == cost_dict


def test_target_filter_round_trip_creature_and_player():
    """Both supported target-filter atoms (``IsCreature`` ->
    ``any_creature``, ``IsPlayer`` -> ``any_player``) round-trip
    independently of the effect that wraps them."""
    creature_spec = TargetSpec(IsCreature())
    player_spec = TargetSpec(IsPlayer())
    assert serialize_target_filter(creature_spec) == {"kind": "any_creature"}
    assert serialize_target_filter(player_spec) == {"kind": "any_player"}

    # Round-trip via the registry — the dict-emission path on the
    # Rust side should mirror exactly the dict we sent in.
    reg = mtg_bot_engine.Registry()
    payload = {
        "name": "probe",
        "effects": [
            {
                "kind": "deal_damage",
                "amount": 1,
                "target": serialize_target_filter(creature_spec),
            },
            {
                "kind": "deal_damage",
                "amount": 2,
                "target": serialize_target_filter(player_spec),
            },
        ],
    }
    pid = reg.register_printing(payload)
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert [dict(e["target"])["kind"] for e in effects] == [
        "any_creature",
        "any_player",
    ]


# ---------------------------------------------------------------------------
# Test 4 — synthetic CardPrinting via the real classes (path-B-style sanity)
# ---------------------------------------------------------------------------


def test_monoist_sentry_creature_pt_round_trips():
    """Round-trip a real EoE creature's printed name + mana cost +
    P/T through the bridge. Picked **Monoist Sentry** because it has
    the shortest oracle text among EoE creatures (single keyword
    ``Defender``); EoE has no truly-vanilla creatures, but Defender
    is purely a state-based-action / combat marker and the bridge
    has no slot for it yet — the round-trip therefore deliberately
    drops the ability and asserts only the bridge-vocabulary fields
    survive. The Rust side stores `spell_effect=None` for permanent
    spells, which matches what the parser emits. When the abilities
    schema lands (a future slice), the rejection-path test will
    guide the move to a richer vocabulary.

    Bombard's parity test (above) already locks in the spell-effect
    path; this one extends coverage to the creature-shaped path
    that the round-3 P/T extension introduced.
    """
    prints = get_active_source().get_printings()
    sentry = prints["Monoist Sentry"]
    # Sanity-check the parser's output matches what the bridge ingests.
    assert sentry.spell_effect is None
    assert sentry.power == 4
    assert sentry.toughness == 1

    payload = serialize_printing(sentry)
    assert payload == {
        "name": "Monoist Sentry",
        "mana_cost": {"B": 1},
        "power": 4,
        "toughness": 1,
    }

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(payload)
    assert reg.get_printing_name(pid) == "Monoist Sentry"
    assert dict(reg.get_printing_mana_cost(pid)) == {"B": 1}
    assert reg.get_printing_power(pid) == 4
    assert reg.get_printing_toughness(pid) == 1
    # No mana_produced on a creature parsed from oracle text — the
    # Python parser doesn't emit one, the serializer omits it, and
    # the bridge defaults to None.
    assert reg.get_printing_mana_produced(pid) is None
    # Permanent spells carry no `effects` row in this vocabulary.
    assert reg.get_printing_effects(pid) == []


def test_bombard_pt_fields_omitted_after_extension():
    """Locks in: the round-3 bridge extension does NOT regress
    Bombard (an instant). Its serialized payload still has no
    ``power`` / ``toughness`` / ``mana_produced`` keys, and the
    registry returns None for all three lookups."""
    prints = get_active_source().get_printings()
    bombard = prints["Bombard"]
    payload = serialize_printing(bombard)
    assert "power" not in payload
    assert "toughness" not in payload
    assert "mana_produced" not in payload

    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(payload)
    assert reg.get_printing_power(pid) is None
    assert reg.get_printing_toughness(pid) is None
    assert reg.get_printing_mana_produced(pid) is None


def test_synthetic_dealdamage_to_target_creature_round_trips():
    """Construct a ``CardPrinting`` directly via the production
    classes (no oracle-text parsing) carrying a
    ``DealDamageToTargetCreature`` spell effect, and confirm the
    serializer + Rust bridge handle it identically to the
    parsed-from-Scryfall Bombard case. Keeps the parity test alive
    even if Bombard is one day removed from the EoE pool or its
    parsed shape drifts to something out-of-vocabulary."""
    printing = CardPrinting(
        name="Synth-Bolt",
        mana_cost_str="{R}",
        mana_cost=ManaCost(generic=0, colored=(("R", 1),)),
        type_line="Instant",
        supertypes=(),
        card_types=("Instant",),
        subtypes=(),
        spell_effect=DealDamageToTargetCreature(amount=3),
    )
    payload = serialize_printing(printing)
    assert payload == {
        "name": "Synth-Bolt",
        "mana_cost": {"R": 1},
        "effects": [
            {
                "kind": "deal_damage",
                "amount": 3,
                "target": {"kind": "any_creature"},
            }
        ],
    }
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(payload)
    assert reg.get_printing_name(pid) == "Synth-Bolt"
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["amount"] == 3
    assert dict(effects[0]["target"]) == {"kind": "any_creature"}
