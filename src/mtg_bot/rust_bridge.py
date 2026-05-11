"""Python-side wrappers around common Rust engine snapshot / compare patterns.

The Rust engine (``mtg_bot_engine``) exposes a GameState with a family of
snapshot getters (``hand_snapshot``, ``library_snapshot``,
``battlefield_snapshot``, ``stack_snapshot``, ``mana_pool_snapshot``,
``get_life``, ``turn_number``, ``step``, ``active_player_idx``, ...). The
parity tooling stitches those into a per-field view of the state; today
that recipe is duplicated across ``scripts/parity_smoke.py``
(``_build_field_view`` + ``FIELD_DIFF_FIELDS`` + ``_field_repr`` + inline
Registry / make_state construction). This module is the forward home for
those patterns; parity_smoke keeps its in-script versions for compat.

Public surface: ``SNAPSHOT_FIELDS``, ``rust_state_to_snapshot_dict``,
``FieldDiff`` + ``compare_snapshots``, ``format_diff_table``,
``get_builtin_card_recipe``, ``make_rust_state_from_python_decks``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Canonical ordering of fields a Rust state snapshot reports on.
# Mirrors (and is intended to replace) ``FIELD_DIFF_FIELDS`` in
# scripts/parity_smoke.py.
SNAPSHOT_FIELDS: tuple[str, ...] = (
    "turn", "step", "phase", "active_player", "priority_player",
    "is_over", "is_draw", "winner_id",
    "life_p0", "life_p1",
    "hand_p0", "hand_p1",
    "library_p0", "library_p1",
    "graveyard_p0", "graveyard_p1",
    "exile_p0", "exile_p1",
    "mana_pool_p0", "mana_pool_p1",
    "battlefield", "stack",
)


def _zone_obj_sig(d: dict) -> tuple:
    """Per-object 8-tuple signature for a zone-snapshot dict.

    Mirrors ``_rust_zone_obj_sig`` in parity_smoke. ``attached_to`` is
    always ``None`` -- Rust does not yet model auras.
    """
    return (
        d["printing_name"], int(d["tapped"]), int(d["summoning_sick"]),
        d["damage"], d["counters"], None, d["controller_id"], d["id"],
    )


def _stack_obj_sig(d: dict) -> tuple:
    """Stack-item signature using the same 8-tuple shape as zone objects."""
    return (None, 0, 0, 0, 0, None, d["controller"], d["source_obj_id"])


def rust_state_to_snapshot_dict(rust_state) -> dict:
    """Assemble a Rust GameState's snapshot getters into a single dict.

    Keys are exactly ``SNAPSHOT_FIELDS`` (insertion order preserved).
    Scalars carry raw values; zone fields carry tuples of per-object
    8-tuples; mana pools carry sorted ``(symbol, count)`` tuples.
    """
    s = rust_state
    view: dict[str, Any] = {
        "turn": s.turn_number, "step": s.step, "phase": s.step,
        "active_player": s.active_player_idx, "priority_player": s.priority_player_idx,
        "is_over": s.is_over(), "is_draw": s.is_draw, "winner_id": s.winner_id,
        "life_p0": s.get_life(0), "life_p1": s.get_life(1),
        "battlefield": tuple(_zone_obj_sig(d) for d in s.battlefield_snapshot()),
        "stack": tuple(_stack_obj_sig(d) for d in s.stack_snapshot()),
    }
    for pid in (0, 1):
        view[f"hand_p{pid}"] = tuple(_zone_obj_sig(d) for d in s.hand_snapshot(pid))
        view[f"library_p{pid}"] = tuple(_zone_obj_sig(d) for d in s.library_snapshot(pid))
        view[f"graveyard_p{pid}"] = tuple(_zone_obj_sig(d) for d in s.graveyard_snapshot(pid))
        view[f"exile_p{pid}"] = tuple(_zone_obj_sig(d) for d in s.exile_snapshot(pid))
        view[f"mana_pool_p{pid}"] = tuple(sorted(s.mana_pool_snapshot(pid).items()))
    # Re-order to canonical SNAPSHOT_FIELDS order.
    return {k: view[k] for k in SNAPSHOT_FIELDS}


@dataclass(frozen=True)
class FieldDiff:
    """One row of a snapshot comparison."""
    name: str
    value_a: Any
    value_b: Any
    equal: bool


def compare_snapshots(
    snap_a: dict, snap_b: dict, *, label_a: str = "A", label_b: str = "B",
) -> list[FieldDiff]:
    """Structural per-field diff. Walks ``SNAPSHOT_FIELDS`` first, then any
    extra keys present in either snapshot. Labels are accepted for
    signature symmetry with ``format_diff_table`` (caller-side only)."""
    del label_a, label_b  # accepted for caller-side label threading
    diffs: list[FieldDiff] = []
    seen: set[str] = set()
    extras = [k for k in (*snap_a.keys(), *snap_b.keys()) if k not in SNAPSHOT_FIELDS]
    for field in list(SNAPSHOT_FIELDS) + extras:
        if field in seen:
            continue
        seen.add(field)
        if field not in snap_a and field not in snap_b:
            continue
        va = snap_a.get(field)
        vb = snap_b.get(field)
        diffs.append(FieldDiff(field, va, vb, va == vb))
    return diffs


def _short_repr(value: Any, max_len: int = 35) -> str:
    """Truncated single-line repr for table rendering."""
    rendered = repr(value)
    if len(rendered) > max_len:
        return rendered[: max_len - 3] + "..."
    return rendered


def format_diff_table(
    diffs: Sequence[FieldDiff], *, label_a: str = "A", label_b: str = "B",
) -> str:
    """Render a diff list as ``field | status | left | right`` rows plus
    a ``PARITY: PASS/FAIL`` summary line. Mirrors parity_smoke's
    ``--mode field-diff`` table format."""
    lines: list[str] = [
        f"field             | status | {label_a:<37} | {label_b}",
        "------------------|--------|---------------------------------------|--------",
    ]
    n_differ = 0
    for d in diffs:
        if d.equal:
            lines.append(f"{d.name:<17} | OK     | {_short_repr(d.value_a):<37} | (same)")
        else:
            n_differ += 1
            lines.append(
                f"{d.name:<17} | DIFFER | "
                f"{_short_repr(d.value_a):<37} | {_short_repr(d.value_b)}"
            )
    lines.append("")
    if n_differ == 0:
        lines.append(f"PARITY: PASS -- all {len(diffs)} fields match")
    else:
        lines.append(f"PARITY: FAIL -- {n_differ} of {len(diffs)} fields differ")
    return "\n".join(lines)


# ---- deck construction --------------------------------------------------

# Built-in card recipes. Short list of "cards parity tests already lean
# on" -- extend by adding entries to ``_BUILT_IN_CARDS``.
def _land(name: str, color: str) -> dict:
    return {"name": name, "card_types": ["Land"], "mana_cost": {}, "effects": [], "mana_produced": [color]}


_BUILT_IN_CARDS: dict[str, dict] = {
    "Forest":   _land("Forest",   "G"),
    "Mountain": _land("Mountain", "R"),
    "Plains":   _land("Plains",   "W"),
    "Island":   _land("Island",   "U"),
    "Swamp":    _land("Swamp",    "B"),
    "Llanowar Elves": {
        "name": "Llanowar Elves", "card_types": ["Creature"],
        "mana_cost": {"G": 1}, "effects": [], "power": 1, "toughness": 1,
    },
    "Bears": {
        "name": "Bears", "card_types": ["Creature"],
        "mana_cost": {"G": 1, "generic": 1}, "effects": [], "power": 2, "toughness": 2,
    },
    "Lightning Bolt": {
        "name": "Lightning Bolt", "card_types": ["Instant"], "mana_cost": {"R": 1},
        "effects": [{"kind": "deal_damage", "amount": 3, "target": {"kind": "any_player"}}],
    },
}


def get_builtin_card_recipe(name: str) -> dict:
    """Return a copy of the built-in recipe for ``name``; raises ``KeyError``
    if not registered. Copy so caller mutations don't corrupt the seed."""
    if name not in _BUILT_IN_CARDS:
        raise KeyError(
            f"no built-in card recipe for {name!r}; known cards: "
            f"{sorted(_BUILT_IN_CARDS)}"
        )
    return dict(_BUILT_IN_CARDS[name])


def make_rust_state_from_python_decks(
    p0_lib_card_names: Iterable[str],
    p1_lib_card_names: Iterable[str],
    *,
    p0_hand_card_names: Iterable[str] | None = None,
    p1_hand_card_names: Iterable[str] | None = None,
    shuffle_seed: int | None = None,
):
    """Build a Rust GameState from Python card-name strings.

    Looks each name up in the built-in recipe table, registers each
    unique recipe once, then assembles per-player printing-id lists in
    caller order and calls ``mtg_bot_engine.make_state``. Unknown names
    raise ``KeyError``; the engine import is deferred so this module
    can be imported without the extension built.
    """
    import mtg_bot_engine  # local import: keep module-import-time clean

    reg = mtg_bot_engine.Registry()
    name_to_pid: dict[str, int] = {}

    def _pid(name: str) -> int:
        if name not in name_to_pid:
            name_to_pid[name] = reg.register_printing(get_builtin_card_recipe(name))
        return name_to_pid[name]

    p0_lib = [_pid(n) for n in p0_lib_card_names]
    p1_lib = [_pid(n) for n in p1_lib_card_names]
    p0_hand = [_pid(n) for n in (p0_hand_card_names or [])]
    p1_hand = [_pid(n) for n in (p1_hand_card_names or [])]

    return mtg_bot_engine.make_state(
        reg, p0_lib, p1_lib,
        p0_hand_printing_ids=p0_hand or None,
        p1_hand_printing_ids=p1_hand or None,
        shuffle_seed=shuffle_seed,
    )
