"""Tests for ``mtg_bot.rust_bridge`` -- the Python-side parity utility module.

The Rust extension is required for every test here. If it isn't built,
``pytest.importorskip`` short-circuits the module so a fresh checkout
doesn't break the gate.
"""

from __future__ import annotations

import pytest

mtg_bot_engine = pytest.importorskip(
    "mtg_bot_engine",
    reason="rust extension not built; run `maturin develop --release -m rust/Cargo.toml`",
)

from mtg_bot.rust_bridge import (  # noqa: E402  (after importorskip)
    FieldDiff,
    SNAPSHOT_FIELDS,
    compare_snapshots,
    format_diff_table,
    get_builtin_card_recipe,
    make_rust_state_from_python_decklist,
    make_rust_state_from_python_decks,
    python_card_to_rust_recipe,
    rust_state_to_snapshot_dict,
)


def _synthetic():
    """Default synthetic state: 10 battlefield / 7 hand / 50 library."""
    return mtg_bot_engine.make_synthetic_state()


def test_snapshot_dict_has_every_canonical_field():
    """``rust_state_to_snapshot_dict`` returns exactly the SNAPSHOT_FIELDS
    key set in order (no extras, no missing)."""
    snap = rust_state_to_snapshot_dict(_synthetic())
    assert tuple(snap.keys()) == SNAPSHOT_FIELDS
    # Spot-check a scalar + a zone field landed with sensible values.
    assert snap["life_p0"] == 20
    assert snap["life_p1"] == 20
    # synthetic state seeds turn / step at fixed-but-non-default values; we
    # don't pin the literal here -- just that the field is plumbed through.
    assert isinstance(snap["turn"], int) and snap["turn"] >= 1
    assert isinstance(snap["step"], str) and snap["step"]
    # synthetic state has 7 cards in each hand.
    assert len(snap["hand_p0"]) == 7
    assert len(snap["hand_p1"]) == 7


def test_compare_snapshots_identical_returns_all_equal():
    """Two snapshots of the same state diff to zero non-equal rows."""
    snap = rust_state_to_snapshot_dict(_synthetic())
    diffs = compare_snapshots(snap, dict(snap))
    assert len(diffs) == len(SNAPSHOT_FIELDS)
    assert all(d.equal for d in diffs)
    # Subset that's non-equal should be empty.
    assert [d for d in diffs if not d.equal] == []


def test_compare_snapshots_flags_mismatched_fields():
    """Mutating two specific fields surfaces exactly those rows as DIFFER
    while everything else stays OK."""
    snap_a = rust_state_to_snapshot_dict(_synthetic())
    snap_b = dict(snap_a)
    snap_b["life_p0"] = 19
    snap_b["turn"] = snap_a["turn"] + 1

    diffs = compare_snapshots(snap_a, snap_b)
    differing = {d.name for d in diffs if not d.equal}
    assert differing == {"life_p0", "turn"}
    # Mismatched rows carry both values.
    life_row = next(d for d in diffs if d.name == "life_p0")
    assert life_row.value_a == 20
    assert life_row.value_b == 19


def test_format_diff_table_includes_names_values_and_differ_marker():
    """Rendered table contains every field name, the differing values, the
    DIFFER marker, and a PARITY: FAIL summary line for a mismatch."""
    snap_a = rust_state_to_snapshot_dict(_synthetic())
    snap_b = dict(snap_a)
    snap_b["life_p1"] = 7

    diffs = compare_snapshots(snap_a, snap_b)
    table = format_diff_table(diffs)
    # Every canonical field shows up as a row label.
    for field in SNAPSHOT_FIELDS:
        assert field in table
    # The DIFFER marker and FAIL summary line fire for the mutated field.
    assert "DIFFER" in table
    assert "PARITY: FAIL" in table
    # Both the old and the new value are present somewhere in the rendering.
    assert "20" in table
    assert "7" in table
    # An equal row uses "(same)" as the right-column shorthand.
    assert "(same)" in table


def test_make_rust_state_from_python_decks_uses_card_names():
    """The Python-name wrapper produces a state whose hand and library
    contents match the requested card names in caller order."""
    state = make_rust_state_from_python_decks(
        ["Forest", "Mountain", "Llanowar Elves"],
        ["Plains", "Island", "Bears"],
        p0_hand_card_names=["Lightning Bolt"],
        p1_hand_card_names=["Swamp"],
    )
    snap = rust_state_to_snapshot_dict(state)
    lib_p0_names = [obj[0] for obj in snap["library_p0"]]
    lib_p1_names = [obj[0] for obj in snap["library_p1"]]
    hand_p0_names = [obj[0] for obj in snap["hand_p0"]]
    hand_p1_names = [obj[0] for obj in snap["hand_p1"]]

    assert lib_p0_names == ["Forest", "Mountain", "Llanowar Elves"]
    assert lib_p1_names == ["Plains", "Island", "Bears"]
    assert hand_p0_names == ["Lightning Bolt"]
    assert hand_p1_names == ["Swamp"]


def test_get_builtin_card_recipe_unknown_card_raises():
    """Unknown card names are a hard error -- the wrapper would otherwise
    fail downstream in the Rust registry with a less actionable message."""
    with pytest.raises(KeyError, match="no built-in card recipe"):
        get_builtin_card_recipe("NotARealCard")


def test_round_trip_snapshot_format_is_human_readable():
    """End-to-end smoke: state -> snapshot dict -> compare-with-self ->
    format_diff_table produces a non-empty string with the PASS summary."""
    state = make_rust_state_from_python_decks(
        ["Forest"] * 3, ["Mountain"] * 3,
        p0_hand_card_names=["Llanowar Elves"],
    )
    snap = rust_state_to_snapshot_dict(state)
    diffs = compare_snapshots(snap, dict(snap), label_a="rust", label_b="rust2")
    table = format_diff_table(diffs, label_a="rust", label_b="rust2")
    assert isinstance(table, str)
    assert table.count("\n") > 5  # multi-line output
    assert "PARITY: PASS" in table
    # Label thread-through ends up in the header row.
    assert "rust" in table
    # FieldDiff dataclass shape sanity: name/value_a/value_b/equal accessible.
    first = diffs[0]
    assert isinstance(first, FieldDiff)
    assert first.equal is True


# ---- python_card_to_rust_recipe + make_rust_state_from_python_decklist ----

# Build a synthetic 60-card decklist that exercises the translator's
# supported / unsupported branches. Using a mix of basics + a real
# card with a translatable effect + a real card with an UN-translatable
# effect is the smoke target -- the stats roll-up measures the
# vocabulary gap directly.
_SYNTHETIC_DECK_60 = (
    ["Forest"] * 16
    + ["Mountain"] * 4
    + ["Lightning Bolt"] * 4  # builtin, translates clean
    + ["Llanowar Elves"] * 4  # builtin, vanilla creature
    + ["Bears"] * 4  # builtin, vanilla creature
    + ["Plains"] * 4
    + ["Island"] * 4
    + ["Swamp"] * 4
    + ["Beyond the Quiet"] * 4  # real EoE card, untranslated spell_effect
    + ["Dual-Sun Technique"] * 4  # real, also untranslated
    + ["Anticausal Vestige"] * 4  # real, has triggered ability (non-ETB)
    + ["Honor"] * 4  # Sequence spell_effect
)
assert len(_SYNTHETIC_DECK_60) == 60


def test_python_card_to_rust_recipe_forest_round_trips_cleanly():
    """A real Python Forest translates to a Rust recipe with the
    Land card type, an empty mana_cost, and {G} mana_produced. The
    recipe registers successfully in a fresh Rust registry."""
    from mtg_bot.cards.source import get_active_source

    forest = get_active_source().get_printings()["Forest"]
    recipe = python_card_to_rust_recipe(forest)
    # Out-of-band _unsupported_effects tracker is empty for a vanilla land.
    assert recipe.get("_unsupported_effects") == []
    assert recipe["name"] == "Forest"
    assert recipe["card_types"] == ["Land"]
    assert recipe["mana_cost"] == {}
    assert recipe["mana_produced"] == ["G"]
    assert recipe["effects"] == []
    assert recipe["etb_effects"] == []
    # Strip the bridge-internal marker before registering and verify
    # the Rust side accepts the recipe shape.
    recipe.pop("_unsupported_effects", None)
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(recipe)
    assert reg.get_printing_name(pid) == "Forest"


def test_python_card_to_rust_recipe_marks_unsupported_effects():
    """A real card whose spell_effect / triggered abilities aren't in
    the Rust closed vocabulary registers its class name in the
    ``_unsupported_effects`` marker list. Recipe still parses and
    registers (empty effects on the Rust side) -- the smoke can run."""
    from mtg_bot.cards.source import get_active_source

    printings = get_active_source().get_printings()
    # Beyond the Quiet's spell_effect is ExileAllCreaturesAndSpacecraft,
    # which the bridge doesn't translate.
    btq = printings["Beyond the Quiet"]
    recipe = python_card_to_rust_recipe(btq)
    unsup = recipe.get("_unsupported_effects", [])
    assert "ExileAllCreaturesAndSpacecraft" in unsup
    assert recipe["effects"] == []  # nothing translated into Rust effects
    # Recipe still registers (this is the parity-vocabulary point --
    # cards with unsupported effects are still legal deck members,
    # they just resolve to no-ops when cast).
    recipe.pop("_unsupported_effects", None)
    reg = mtg_bot_engine.Registry()
    pid = reg.register_printing(recipe)
    assert reg.get_printing_name(pid) == "Beyond the Quiet"


def test_make_rust_state_from_decklist_returns_state_with_full_library():
    """End-to-end: a 60-card decklist (mix of basics + real EoE cards
    with both supported and unsupported effects) loads into a Rust
    state. Both seats get the full deck in their library; nothing is
    pre-drawn to hand (caller controls the draw schedule)."""
    state, stats = make_rust_state_from_python_decklist(
        deck_card_names=_SYNTHETIC_DECK_60, shuffle_seed=123,
    )
    snap = rust_state_to_snapshot_dict(state)
    assert len(snap["library_p0"]) == 60
    assert len(snap["library_p1"]) == 60
    # Hands start empty -- the Rust make_state path doesn't auto-draw
    # when no explicit hand list is supplied. Caller decides.
    assert len(snap["hand_p0"]) == 0
    assert len(snap["hand_p1"]) == 0


def test_decklist_stats_report_unsupported_effect_classes_accurately():
    """The stats dict from a real-card-bearing decklist reports
    non-zero ``cards_loaded`` and surfaces the unsupported effect
    classes encountered. THIS is the parity-vocabulary measurement."""
    _, stats = make_rust_state_from_python_decklist(
        deck_card_names=_SYNTHETIC_DECK_60, shuffle_seed=7,
    )
    # 12 unique card names in the 60-card list above.
    assert stats["cards_loaded"] == 12
    assert stats["deck_size"] == 60
    # Basics + builtins translate clean. The real EoE cards in the
    # deck are picked because their spell_effect / triggered abilities
    # exercise the unsupported path -- expect at least one entry in
    # ``unsupported_effect_classes``, and a corresponding
    # cards_with_unsupported_effects count.
    assert stats["cards_with_unsupported_effects"] >= 3
    classes = stats["unsupported_effect_classes"]
    assert "ExileAllCreaturesAndSpacecraft" in classes  # Beyond the Quiet
    # Stats dict is well-formed (sorted-dedupe contract on the list).
    assert classes == sorted(set(classes))


def test_decklist_load_deterministic_across_two_loads_with_same_seed():
    """Loading the same deck with the same shuffle_seed produces a
    state whose snapshot fingerprints to the same value. Determinism
    is parity-load-bearing: the smoke needs reproducible deck
    schedules to diff against the Python engine cleanly."""
    state_a, _ = make_rust_state_from_python_decklist(
        deck_card_names=_SYNTHETIC_DECK_60, shuffle_seed=999,
    )
    state_b, _ = make_rust_state_from_python_decklist(
        deck_card_names=_SYNTHETIC_DECK_60, shuffle_seed=999,
    )
    snap_a = rust_state_to_snapshot_dict(state_a)
    snap_b = rust_state_to_snapshot_dict(state_b)
    # All snapshot fields except the registry-stamped GameObject ids
    # match. Library order in particular -- the seed makes shuffles
    # bit-identical.
    lib_a_names = tuple(obj[0] for obj in snap_a["library_p0"])
    lib_b_names = tuple(obj[0] for obj in snap_b["library_p0"])
    assert lib_a_names == lib_b_names
    # Across-seat parity too: same deck on p1.
    lib_a_p1 = tuple(obj[0] for obj in snap_a["library_p1"])
    lib_b_p1 = tuple(obj[0] for obj in snap_b["library_p1"])
    assert lib_a_p1 == lib_b_p1


def test_decklist_load_from_text_file_round_trips(tmp_path):
    """The file-based ``deck_path`` entry point reads a one-card-per-
    line text file (with leading-count syntax) and produces the same
    state as the inline ``deck_card_names`` path."""
    deck_file = tmp_path / "deck.txt"
    deck_file.write_text(
        "# basic land manifold\n"
        "20 Forest\n"
        "4 Lightning Bolt\n"
        "4 Llanowar Elves\n"
        "4 Bears\n"
        "\n"
        "20 Mountain\n"
        "4 Plains\n"
        "4 Island\n",
        encoding="utf-8",
    )
    state, stats = make_rust_state_from_python_decklist(
        deck_path=deck_file, shuffle_seed=1,
    )
    assert stats["deck_size"] == 60
    # All cards in the deck file are builtins or supported Python
    # printings (basics) -- no unsupported effects expected.
    assert stats["cards_with_unsupported_effects"] == 0
    snap = rust_state_to_snapshot_dict(state)
    assert len(snap["library_p0"]) == 60
    assert len(snap["library_p1"]) == 60
