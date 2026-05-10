"""Round-trip + error-path tests for the Python <-> Rust card bridge.

Each test builds a card definition as a plain Python dict, registers
it through `mtg_bot_engine.Registry.register_printing`, then queries
the registry to make sure the parsed-and-re-emitted dict matches the
input shape. The bridge is the single point of validation, so the
error cases below cover each rejection path.
"""

from __future__ import annotations

import pytest

mtg_bot_engine = pytest.importorskip(
    "mtg_bot_engine",
    reason="rust extension not built; run `maturin develop --release -m rust/Cargo.toml`",
)


SHOCK = {
    "name": "Shock",
    "mana_cost": {"R": 1},
    "effects": [
        {"kind": "deal_damage", "amount": 2, "target": {"kind": "any_creature"}}
    ],
}

LIGHTNING_BOLT_ISH = {
    "name": "Lightning Bolt-ish",
    "mana_cost": {"R": 1},
    "effects": [
        {"kind": "deal_damage", "amount": 3, "target": {"kind": "any_player"}}
    ],
}

VANILLA_BLANK = {
    "name": "Blank Card",
}


def test_registry_starts_empty():
    reg = mtg_bot_engine.Registry()
    assert len(reg) == 0


def test_register_printing_returns_sequential_ids():
    reg = mtg_bot_engine.Registry()
    assert reg.register_printing(SHOCK) == 0
    assert reg.register_printing(LIGHTNING_BOLT_ISH) == 1
    assert reg.register_printing(VANILLA_BLANK) == 2
    assert len(reg) == 3


def test_round_trip_shock():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(SHOCK)
    assert reg.get_printing_name(pid) == "Shock"
    assert dict(reg.get_printing_mana_cost(pid)) == {"R": 1}
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert len(effects) == 1
    e = effects[0]
    assert e["kind"] == "deal_damage"
    assert e["amount"] == 2
    assert dict(e["target"]) == {"kind": "any_creature"}


def test_round_trip_any_player_target():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(LIGHTNING_BOLT_ISH)
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert dict(effects[0]["target"]) == {"kind": "any_player"}


def test_round_trip_draw_cards_effect():
    """Round-8 bridge extension: DrawCards { count } effect."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Concentrate-ish",
            "mana_cost": {"U": 2, "generic": 2},
            "card_types": ["Sorcery"],
            "effects": [{"kind": "draw_cards", "count": 3}],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert len(effects) == 1
    e = effects[0]
    assert e["kind"] == "draw_cards"
    assert e["count"] == 3


def test_round_trip_gain_life_effect():
    """Round-8 bridge extension: GainLife { amount } effect."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Healing Salve-ish",
            "mana_cost": {"W": 1},
            "card_types": ["Instant"],
            "effects": [{"kind": "gain_life", "amount": 3}],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert len(effects) == 1
    e = effects[0]
    assert e["kind"] == "gain_life"
    assert e["amount"] == 3


def test_round_trip_multiple_effect_kinds_preserve_order():
    """Round-8 sanity: multiple variants of Effect can land on one card."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Mixed Spell",
            "effects": [
                {"kind": "draw_cards", "count": 1},
                {"kind": "gain_life", "amount": 2},
                {"kind": "deal_damage", "amount": 1, "target": {"kind": "any_player"}},
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert [e["kind"] for e in effects] == ["draw_cards", "gain_life", "deal_damage"]


def test_draw_cards_missing_count_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="count"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "draw_cards"}]}
        )


def test_gain_life_missing_amount_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="amount"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "gain_life"}]}
        )


def test_round_trip_return_to_hand_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Unsummon-ish",
            "mana_cost": {"U": 1},
            "card_types": ["Instant"],
            "effects": [
                {"kind": "return_to_hand", "target": {"kind": "any_creature"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "return_to_hand"
    assert dict(effects[0]["target"]) == {"kind": "any_creature"}


def test_round_trip_mill_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Tome Scour-ish",
            "mana_cost": {"U": 1},
            "card_types": ["Sorcery"],
            "effects": [{"kind": "mill", "count": 5}],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0] == {"kind": "mill", "count": 5}


def test_return_to_hand_missing_target_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="target"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "return_to_hand"}]}
        )


def test_mill_missing_count_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="count"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "mill"}]}
        )


def test_round_trip_exile_target_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Path-ish",
            "mana_cost": {"W": 1},
            "card_types": ["Instant"],
            "effects": [
                {"kind": "exile_target", "target": {"kind": "any_creature"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "exile_target"
    assert dict(effects[0]["target"]) == {"kind": "any_creature"}


def test_exile_target_missing_target_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="target"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "exile_target"}]}
        )


def test_round_trip_tap_target_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Twiddle-ish",
            "mana_cost": {"U": 1},
            "card_types": ["Instant"],
            "effects": [
                {"kind": "tap_target", "target": {"kind": "any_creature"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "tap_target"
    assert dict(effects[0]["target"]) == {"kind": "any_creature"}


def test_tap_target_missing_target_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="target"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "tap_target"}]}
        )


def test_round_trip_untap_target_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Untap-ish",
            "mana_cost": {"U": 1},
            "card_types": ["Instant"],
            "effects": [
                {"kind": "untap_target", "target": {"kind": "any_creature"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "untap_target"
    assert dict(effects[0]["target"]) == {"kind": "any_creature"}


def test_untap_target_missing_target_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="target"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "untap_target"}]}
        )


def test_round_trip_sacrifice_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Sacrifice-ish",
            "mana_cost": {"B": 1},
            "card_types": ["Instant"],
            "effects": [
                {"kind": "sacrifice", "target": {"kind": "any_creature"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "sacrifice"
    assert dict(effects[0]["target"]) == {"kind": "any_creature"}


def test_sacrifice_missing_target_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="target"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "sacrifice"}]}
        )


def test_round_trip_add_counters_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Counterspell-ish-buff",
            "mana_cost": {"G": 1},
            "card_types": ["Instant"],
            "effects": [
                {"kind": "add_counters", "count": 2, "target": {"kind": "any_creature"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "add_counters"
    assert effects[0]["count"] == 2
    assert dict(effects[0]["target"]) == {"kind": "any_creature"}


def test_add_counters_missing_count_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="count"):
        reg.register_printing(
            {
                "name": "Broken",
                "effects": [
                    {"kind": "add_counters", "target": {"kind": "any_creature"}}
                ],
            }
        )


def test_add_counters_missing_target_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="target"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "add_counters", "count": 1}]}
        )


def test_round_trip_discard_card_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Distress-ish",
            "mana_cost": {"B": 1},
            "card_types": ["Sorcery"],
            "effects": [
                {"kind": "discard_card", "target": {"kind": "any_player"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "discard_card"
    assert dict(effects[0]["target"]) == {"kind": "any_player"}


def test_discard_card_missing_target_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="target"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "discard_card"}]}
        )


def test_negative_add_counters_round_trips():
    """-1/-1 counters via negative count — same Effect, same dispatch."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Wither-ish",
            "effects": [
                {"kind": "add_counters", "count": -1, "target": {"kind": "any_creature"}}
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["count"] == -1


def test_round_trip_scry_n_effect():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Preordain-ish",
            "mana_cost": {"U": 1},
            "card_types": ["Sorcery"],
            "effects": [{"kind": "scry_n", "count": 2}],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["kind"] == "scry_n"
    assert effects[0]["count"] == 2


def test_scry_n_missing_count_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="count"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "scry_n"}]}
        )


def test_scry_n_zero_count_round_trips():
    """Scry 0 is a degenerate but legal value — should round-trip."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "ScryZero",
            "effects": [{"kind": "scry_n", "count": 0}],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert effects[0]["count"] == 0


def test_vanilla_card_with_no_mana_cost_or_effects():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(VANILLA_BLANK)
    assert reg.get_printing_name(pid) == "Blank Card"
    assert dict(reg.get_printing_mana_cost(pid)) == {}
    assert reg.get_printing_effects(pid) == []


def test_multi_color_mana_cost_round_trips():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Niv-Mizzet-ish",
            "mana_cost": {"generic": 4, "U": 1, "R": 1},
        }
    )
    assert dict(reg.get_printing_mana_cost(pid)) == {"generic": 4, "U": 1, "R": 1}


def test_zero_mana_cost_keys_are_omitted_on_round_trip():
    """Round-trip drops zero counts — `{R: 0}` registers as nothing."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing({"name": "Zero", "mana_cost": {"R": 0, "G": 2}})
    assert dict(reg.get_printing_mana_cost(pid)) == {"G": 2}


def test_multiple_effects_preserve_order():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {
            "name": "Two Bolts",
            "effects": [
                {"kind": "deal_damage", "amount": 1, "target": {"kind": "any_player"}},
                {"kind": "deal_damage", "amount": 2, "target": {"kind": "any_creature"}},
            ],
        }
    )
    effects = [dict(e) for e in reg.get_printing_effects(pid)]
    assert [e["amount"] for e in effects] == [1, 2]
    assert [dict(e["target"])["kind"] for e in effects] == [
        "any_player",
        "any_creature",
    ]


# --- error paths -------------------------------------------------


def test_missing_name_raises_key_error():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="name"):
        reg.register_printing({"mana_cost": {"R": 1}})


def test_missing_effect_kind_raises_key_error():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(KeyError, match="kind"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"amount": 2}]}
        )


def test_unknown_effect_kind_raises_value_error():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="unknown effect kind"):
        reg.register_printing(
            {"name": "Broken", "effects": [{"kind": "summon_unicorn"}]}
        )


def test_unknown_target_kind_raises_value_error():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="unknown target filter kind"):
        reg.register_printing(
            {
                "name": "Broken",
                "effects": [
                    {
                        "kind": "deal_damage",
                        "amount": 2,
                        "target": {"kind": "any_planeswalker"},
                    }
                ],
            }
        )


def test_unknown_mana_color_raises_value_error():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="unknown mana_cost key"):
        reg.register_printing({"name": "Broken", "mana_cost": {"Q": 1}})


def test_lookup_out_of_range_id_raises_key_error():
    reg = mtg_bot_engine.Registry()
    reg.register_printing(SHOCK)
    with pytest.raises(KeyError, match="no printing with id 99"):
        reg.get_printing_name(99)
    with pytest.raises(KeyError):
        reg.get_printing_mana_cost(99)
    with pytest.raises(KeyError):
        reg.get_printing_effects(99)


def test_target_must_be_a_dict():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="target must be a dict"):
        reg.register_printing(
            {
                "name": "Broken",
                "effects": [{"kind": "deal_damage", "amount": 2, "target": "any"}],
            }
        )


def test_mana_cost_value_must_be_non_negative_integer():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="non-negative integer"):
        reg.register_printing({"name": "Broken", "mana_cost": {"R": "hot"}})


# --- power / toughness / mana_produced (round-3 extension) ------


GRIZZLY_BEARS = {
    "name": "Grizzly Bears",
    "mana_cost": {"generic": 1, "G": 1},
    "power": 2,
    "toughness": 2,
}

FOREST_PRINTING = {
    "name": "Forest",
    "mana_produced": ["G"],
}


def test_round_trip_creature_power_toughness():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(GRIZZLY_BEARS)
    assert reg.get_printing_power(pid) == 2
    assert reg.get_printing_toughness(pid) == 2
    # Non-creature default for the same call surface returns None.
    sid = reg.register_printing(SHOCK)
    assert reg.get_printing_power(sid) is None
    assert reg.get_printing_toughness(sid) is None


def test_round_trip_mana_produced_forest():
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(FOREST_PRINTING)
    assert reg.get_printing_mana_produced(pid) == ["G"]


def test_mana_produced_default_is_none():
    """Spells (and creatures without mana production) round-trip as None."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(SHOCK)
    assert reg.get_printing_mana_produced(pid) is None


def test_negative_power_round_trips():
    """`power: i16` permits negatives (no real card has them by
    print, but anthem-like effects produce them on objects; the
    field bound is intentionally i16 so the bridge doesn't need
    to grow another type later)."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {"name": "Pseudo", "power": -1, "toughness": 0}
    )
    assert reg.get_printing_power(pid) == -1
    assert reg.get_printing_toughness(pid) == 0


def test_mana_produced_multi_pip_round_trip():
    """Vec<String> tolerates >1-pip producers at the bridge level
    even though `tap_for_mana` declines to commit on them this slice."""
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(
        {"name": "Hypothetical Bicycle Land", "mana_produced": ["G", "G"]}
    )
    assert reg.get_printing_mana_produced(pid) == ["G", "G"]


# --- error paths for the new fields ------------------------------


def test_invalid_mana_symbol_in_mana_produced_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="unknown mana symbol"):
        reg.register_printing({"name": "Broken", "mana_produced": ["Q"]})


def test_lowercase_mana_symbol_in_mana_produced_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="unknown mana symbol"):
        reg.register_printing({"name": "Broken", "mana_produced": ["g"]})


def test_non_iterable_mana_produced_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError):
        reg.register_printing({"name": "Broken", "mana_produced": 7})


def test_non_string_mana_produced_entry_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="must be strings"):
        reg.register_printing({"name": "Broken", "mana_produced": [1]})


def test_non_integer_power_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="power must be an integer"):
        reg.register_printing({"name": "Broken", "power": "huge"})


def test_non_integer_toughness_raises():
    reg = mtg_bot_engine.Registry()
    with pytest.raises(ValueError, match="toughness must be an integer"):
        reg.register_printing({"name": "Broken", "toughness": "small"})


def test_lookup_oob_for_new_getters_raises():
    reg = mtg_bot_engine.Registry()
    reg.register_printing(GRIZZLY_BEARS)
    with pytest.raises(KeyError):
        reg.get_printing_power(99)
    with pytest.raises(KeyError):
        reg.get_printing_toughness(99)
    with pytest.raises(KeyError):
        reg.get_printing_mana_produced(99)


# --- bench helper ------------------------------------------------


def test_register_n_times_micros_returns_positive():
    reg = mtg_bot_engine.Registry()
    micros = reg.register_n_times_micros(SHOCK, 100)
    assert micros > 0.0
    # Sanity bound: parsing one Shock-like dict shouldn't exceed
    # ~50us even on a slow box; release-mode CPython is usually
    # under 5us.
    assert micros < 200.0, f"register too slow: {micros:.2f}us"
