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
    make_rust_state_from_python_decks,
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
