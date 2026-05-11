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
``get_builtin_card_recipe``, ``make_rust_state_from_python_decks``,
``python_card_to_rust_recipe``, ``make_rust_state_from_python_decklist``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


# ---- Python card -> Rust recipe translation ------------------------------

# Rust knows a fixed closed-vocabulary of Effect variants (see
# rust/src/cards.rs). The translator maps Python ``Effect`` instance
# class names to a (kind, field-extractor) pair. Unmapped classes are
# the parity-vocabulary gap that ``cards_with_unsupported_effects``
# measures.
#
# Each entry's value is a callable ``(py_effect_instance) -> dict``
# returning the dict that Rust's ``parse_effect`` consumes. The
# callable returns ``None`` if the effect translates structurally but
# carries fields the Rust slice can't yet honor (e.g. a SearchLibrary
# variant constrained by a predicate the Rust version doesn't accept);
# the caller then treats the effect as unsupported.
def _classify_target_filter(filt) -> str | None:
    """Heuristic mapping Python TargetFilter -> Rust target-filter kind.

    Returns ``"any_creature"`` for creature-only filters, ``"any_player"``
    for player-only filters, ``"any_target"`` for filters that admit
    both players and creatures, and ``None`` when the filter doesn't
    resolve to a Rust-supported shape (e.g. lands, artifacts only).

    Class-name / repr-string inspection is the cheapest discriminator
    that doesn't pull the predicates sub-package into this bridge
    module's import surface. The Rust closed vocab knows two filter
    shapes: ``AnyPlayer`` and ``AnyCreature`` — anything narrower
    (e.g. "controlled by you" / "with power N or less") is dropped to
    the broader Rust shape, which over-targets relative to Python.
    """
    if filt is None:
        return None
    repr_str = repr(filt)
    has_creature = "IsCreature" in repr_str
    has_player = "IsPlayer" in repr_str
    if has_creature and not has_player:
        return "any_creature"
    if has_player and not has_creature:
        return "any_player"
    if has_creature and has_player:
        return "any_target"
    return None


def _single_target_filter_kind(eff) -> str | None:
    """Extract the Rust target-filter kind from a single-slot effect.

    Returns the Rust filter kind (``any_creature`` / ``any_player`` /
    ``any_target``) for effects whose ``targets`` tuple holds exactly
    one TargetSpec, or ``None`` if the effect doesn't have a clean
    single-target shape.
    """
    targets = getattr(eff, "targets", ())
    if len(targets) != 1:
        return None
    slot = targets[0]
    return _classify_target_filter(getattr(slot, "filter", None))


def _t_deal_damage(eff) -> dict | None:
    # The unified ``DealDamage`` carries an ``amount`` and a tuple of
    # target slots. The Rust ``deal_damage`` variant takes one target
    # filter; if Python's targets resolve to a single creature or
    # player slot we can translate, else we treat as unsupported.
    kind = _single_target_filter_kind(eff)
    if kind is None:
        return None
    if kind == "any_target":
        # any-target shape — Python's any_target_filter() is IsPlayer
        # | IsCreature. Rust's ``deal_damage_any_target`` exists for
        # this and doesn't carry a target field.
        return {"kind": "deal_damage_any_target", "amount": int(eff.amount)}
    return {
        "kind": "deal_damage", "amount": int(eff.amount),
        "target": {"kind": kind},
    }


def _t_deal_damage_any_target(eff) -> dict | None:
    return {"kind": "deal_damage_any_target", "amount": int(eff.amount)}


def _t_draw_cards(eff) -> dict | None:
    return {"kind": "draw_cards", "count": int(eff.amount)}


def _t_gain_life(eff) -> dict | None:
    return {"kind": "gain_life", "amount": int(eff.amount)}


def _t_lose_life(eff) -> dict | None:
    if int(eff.amount) < 0:
        return None
    return {"kind": "lose_life", "amount": int(eff.amount)}


def _t_each_opponent_loses_life(eff) -> dict | None:
    return {"kind": "each_opponent_loses_life", "amount": int(eff.amount)}


def _t_target_opponent_loses_life(eff) -> dict | None:
    return {"kind": "target_opponent_loses_life", "amount": int(eff.amount)}


def _t_each_opponent_discards(eff) -> dict | None:
    return {"kind": "each_opponent_discards", "count": int(eff.count)}


def _t_discard_self(eff) -> dict | None:
    return {"kind": "discard_self", "amount": int(eff.amount)}


def _t_mill_target_player(eff) -> dict | None:
    # ``MillTargetPlayer.use_controller_life`` is a Python-only widening
    # the Rust slice doesn't honor — treat as unsupported when set.
    if getattr(eff, "use_controller_life", False):
        return None
    return {"kind": "mill_target_player", "count": int(eff.amount)}


def _t_deal_damage_each_opponent(eff) -> dict | None:
    return {"kind": "damage_each_opponent", "amount": int(eff.amount)}


def _t_deal_n_damage_each_creature(eff) -> dict | None:
    return {"kind": "damage_each_creature", "amount": int(eff.amount)}


def _t_destroy_all_creatures(eff) -> dict | None:
    # Python's DestroyAllCreatures has optional narrowing fields
    # (tapped_only / untapped_only / min_power). Treat any non-default
    # as unsupported -- the Rust unit variant is unconditional.
    if (
        getattr(eff, "tapped_only", False)
        or getattr(eff, "untapped_only", False)
        or getattr(eff, "min_power", 0)
    ):
        return None
    return {"kind": "destroy_all_creatures"}


def _t_destroy_target_artifact(eff) -> dict | None:
    return {"kind": "destroy_target_artifact"}


def _t_destroy_target_land(eff) -> dict | None:
    return {"kind": "destroy_target_land"}


def _t_destroy_target_nonland_permanent(eff) -> dict | None:
    return {"kind": "destroy_target_nonland_permanent"}


def _t_counter_target_spell(eff) -> dict | None:
    # Python's CounterTargetSpell.filter_types narrows what spell types
    # are legal; Rust first slice is unrestricted. Translate
    # unconditionally — the resulting Rust effect over-targets relative
    # to Python (counters anything on top of stack), which the parity
    # smoke's per-step diff will surface if it matters.
    return {"kind": "counter_target_spell"}


def _t_return_creature_from_graveyard(eff) -> dict | None:
    # ``ReturnCreatureFromYourGraveyardToBattlefield`` inherits widening
    # rider fields (or_enchantment / or_spacecraft / max_mv / max_n).
    # When any rider is non-default, treat as unsupported.
    if (
        getattr(eff, "or_enchantment", False)
        or getattr(eff, "or_spacecraft", False)
        or getattr(eff, "or_artifact", False)
        or getattr(eff, "max_mv", 0)
        or int(getattr(eff, "max_n", 1)) > 1
    ):
        return None
    return {"kind": "return_creature_from_your_graveyard_to_battlefield"}


def _t_search_library_for_basic_land(eff) -> dict | None:
    return {
        "kind": "search_library_for_basic_land_tapped",
        "count": int(getattr(eff, "count", 1)),
    }


def _t_put_plus_one_counter_on_self(eff) -> dict | None:
    return {"kind": "put_plus_one_counter_on_self", "amount": int(eff.amount)}


def _t_surveil(eff) -> dict | None:
    if getattr(eff, "use_artifact_count", False):
        return None
    return {"kind": "surveil", "count": int(eff.amount)}


def _t_add_mana_to_pool(eff) -> dict | None:
    # Python's AddManaToPool can mint multiple colors per resolution
    # (the ``produces`` tuple of (color, count) pairs). Rust's
    # ``add_mana_to_pool`` is one (color, amount). If Python lists a
    # single (color, amount) we can translate; else treat as
    # unsupported (caller can register multiple effects in series via
    # the ``mana_produced`` field instead, which IS what land bridges
    # do).
    produces = getattr(eff, "produces", ())
    if len(produces) != 1:
        return None
    color, amount = produces[0]
    # "any" wildcard color is Python-only; Rust closed vocab is W/U/B/R/G/C.
    if color not in ("W", "U", "B", "R", "G", "C"):
        return None
    if getattr(eff, "spend_restriction", None) is not None:
        return None
    return {"kind": "add_mana_to_pool", "color": color, "amount": int(amount)}


# ---- additional effect translators (gap-coverage round) ----------------
#
# These cover Rust Effect variants the first translator landing missed.
# Each maps a Python ``Effect`` subclass to the Rust ``parse_effect``
# dict shape. Single-target effects use ``_single_target_filter_kind``
# to map Python's predicate AST to Rust's closed filter vocab
# (any_player / any_creature). When Python's filter is narrower than
# Rust's vocab (e.g. ``IsCreature & ControlledByYou``), the translation
# is lossy — Rust will over-target ("any creature" includes opponent's
# creatures) — which is intentional under the parity-smoke-first stance:
# a recipe that lands with a slightly-too-broad filter still resolves
# without parse errors, and the per-step diff surfaces any divergence.


def _t_destroy_target(eff) -> dict | None:
    """Translate a single-slot ``Destroy``-family effect to ``destroy_target``.

    Covers ``DestroyTargetCreature`` (incl. the ``or_spacecraft`` /
    ``or_enchantment`` widenings), ``DestroyUpToOneTargetNonlandPermanent``,
    ``DestroyTargetArtifactOrTappedCreature``, and other ``Destroy``
    subclasses whose single-slot filter resolves to a creature / player /
    any-target shape. Filters carrying rider checks the Rust vocab can't
    honor (e.g. ``draw_if_undamaged_amount``, attacking-only) fall through
    to the broader Rust filter — over-destroys lossily but lands without
    parse errors.
    """
    # Rider effects the Rust unit variant can't express -- skip those
    # so the loss is recorded as "unsupported" rather than silently
    # destroying without the draw / attached follow-up.
    if getattr(eff, "draw_if_undamaged_amount", 0) > 0:
        return None
    kind = _single_target_filter_kind(eff)
    if kind is None:
        return None
    if kind == "any_target":
        # Rust's destroy_target needs a TargetFilter; "any target"
        # (player or creature) isn't a Rust shape. Default to creature.
        kind = "any_creature"
    return {"kind": "destroy_target", "target": {"kind": kind}}


def _t_exile_target(eff) -> dict | None:
    """Translate a single-slot ``Exile``-family effect to ``exile_target``.

    Covers ``ExileTargetCreature`` and any ``Exile`` subclass whose
    single-slot filter resolves to creature / player / any-target.
    Narrower filters (e.g. ``HasPowerAtMost``) fall through to the
    broader Rust filter — over-exiles but lands without parse errors.
    """
    kind = _single_target_filter_kind(eff)
    if kind is None:
        return None
    if kind == "any_target":
        kind = "any_creature"
    return {"kind": "exile_target", "target": {"kind": kind}}


def _t_return_to_hand(eff) -> dict | None:
    """Translate ``Bounce`` / ``ReturnTargetCreatureToOwnersHand`` /
    ``ReturnTargetPermanentYouControlToOwnersHand`` to ``return_to_hand``.

    Single-target only (Rust's variant takes one TargetFilter).
    """
    # Bounce subclasses with the drone-create rider can't be expressed.
    if getattr(eff, "create_drone_if_was_attacking", False):
        return None
    kind = _single_target_filter_kind(eff)
    if kind is None:
        return None
    if kind == "any_target":
        kind = "any_creature"
    return {"kind": "return_to_hand", "target": {"kind": kind}}


def _t_tap_target(eff) -> dict | None:
    """Translate ``Tap`` / ``TapAnotherTargetCreature`` / etc to ``tap_target``."""
    kind = _single_target_filter_kind(eff)
    if kind is None:
        return None
    if kind == "any_target":
        kind = "any_creature"
    return {"kind": "tap_target", "target": {"kind": kind}}


def _t_put_plus_one_counter_on_target(eff) -> dict | None:
    """Translate ``PutPlusOneCounters`` / ``PutPlusOneCounterOnTargetCreature``
    / ``PutPlusOneCounterOnCreatureYouControl`` / ``PutPlusOneCounterOnOpponentsCreature``
    to ``add_counters``."""
    # Carry-rider variants (granted keywords UEOT, etc.) handled by
    # their own classes -- this only covers the simple base.
    kind = _single_target_filter_kind(eff)
    if kind is None:
        return None
    if kind == "any_target":
        kind = "any_creature"
    return {
        "kind": "add_counters",
        "count": int(getattr(eff, "amount", 1)),
        "target": {"kind": kind},
    }


def _t_self_mill(eff) -> dict | None:
    """Translate ``SelfMill`` to Rust ``mill`` (mill yourself N)."""
    return {"kind": "mill", "count": int(eff.amount)}


def _t_target_opponent_discards(eff) -> dict | None:
    """Translate ``TargetOpponentDiscards`` to ``discard_card`` with any_player.

    Rust's ``discard_card`` takes a target filter; "any player" is the
    closest fit since "target opponent" can't be expressed precisely.
    The lossy over-target (could resolve against the controller) is
    accepted per the parity-smoke-first stance.
    """
    if int(getattr(eff, "count", 1)) != 1:
        # Rust's discard_card discards one card per resolution; multi-
        # count would need a Sequence of discard_card effects, which
        # is out of scope for the first translation slice.
        return None
    return {"kind": "discard_card", "target": {"kind": "any_player"}}


def _t_target_player_loses_life(eff) -> dict | None:
    """Translate ``TargetPlayerLosesLife`` to ``target_opponent_loses_life``.

    Python's ``TargetPlayerLosesLife`` takes a pre-selected target via
    ``targets[0]`` (set by a triggered ability), while Rust's variant
    is targeted-at-resolve. In a 2-player game, "that target player"
    of an opponent-side trigger reduces to "target opponent loses N"
    — the closest Rust shape.
    """
    return {"kind": "target_opponent_loses_life", "amount": int(eff.amount)}


def _t_create_token(eff, *, token_registrar=None) -> dict | None:
    """Translate ``CreateToken`` to Rust ``create_token``.

    Rust's ``create_token`` carries a ``token_printing_id`` referring to
    a registry-side printing — the token must be pre-registered. The
    ``token_registrar`` argument is a callable that takes the Python
    ``CardPrinting`` of the token and returns the Rust printing id;
    when ``None`` (e.g. the translator is called without a registry
    context), this falls through to unsupported.

    The ``count`` field is preserved; the ``tapped`` / ``attacking`` /
    ``each_player`` / ``schedule_end_of_combat_sacrifice`` riders are
    Python-only widenings the Rust unit shape doesn't honor — when any
    rider is set, treat as unsupported.
    """
    if token_registrar is None:
        return None
    if (
        getattr(eff, "tapped", False)
        or getattr(eff, "attacking", False)
        or getattr(eff, "each_player", False)
        or getattr(eff, "schedule_end_of_combat_sacrifice", False)
    ):
        return None
    token_printing = getattr(eff, "token_printing", None)
    if token_printing is None:
        return None
    try:
        token_pid = token_registrar(token_printing)
    except Exception:
        return None
    return {
        "kind": "create_token",
        "count": int(getattr(eff, "count", 1)),
        "token_printing_id": int(token_pid),
    }


# Class-name -> translator dispatch. Keys are bare class names (no
# module prefix) so a refactor of the Python-side module layout doesn't
# break the bridge.
_EFFECT_TRANSLATORS: dict[str, Any] = {
    "DealDamage": _t_deal_damage,
    "DealDamageAnyTarget": _t_deal_damage_any_target,
    "DrawCards": _t_draw_cards,
    "GainLife": _t_gain_life,
    "LoseLife": _t_lose_life,
    "EachOpponentLosesLife": _t_each_opponent_loses_life,
    "TargetOpponentLosesLife": _t_target_opponent_loses_life,
    "EachOpponentDiscards": _t_each_opponent_discards,
    "DiscardSelf": _t_discard_self,
    "MillTargetPlayer": _t_mill_target_player,
    "DealDamageEachOpponent": _t_deal_damage_each_opponent,
    "DealNDamageToEachCreature": _t_deal_n_damage_each_creature,
    "DestroyAllCreatures": _t_destroy_all_creatures,
    "DestroyTargetArtifact": _t_destroy_target_artifact,
    "DestroyTargetLand": _t_destroy_target_land,
    "DestroyTargetNonlandPermanent": _t_destroy_target_nonland_permanent,
    "CounterTargetSpell": _t_counter_target_spell,
    "ReturnCreatureFromYourGraveyardToBattlefield": _t_return_creature_from_graveyard,
    "SearchLibraryForBasicLandTapped": _t_search_library_for_basic_land,
    "PutPlusOneCounterOnSelf": _t_put_plus_one_counter_on_self,
    "Surveil": _t_surveil,
    "AddManaToPool": _t_add_mana_to_pool,
    # ---- gap-coverage entries (gap-translator slice) ----
    # Destroy family -- single-slot subclasses go through the unified
    # destroy_target arm with filter classification.
    "Destroy": _t_destroy_target,
    "DestroyTargetCreature": _t_destroy_target,
    "DestroyUpToOneTargetNonlandPermanent": _t_destroy_target,
    "DestroyTargetArtifactOrTappedCreature": _t_destroy_target,
    "DestroyTargetArtifactEnchantmentOrFlyingCreature": _t_destroy_target,
    "DestroyTargetNoncreaturePermanent": _t_destroy_target,
    # Exile family -- single-slot Exile subclasses.
    "Exile": _t_exile_target,
    "ExileTargetCreature": _t_exile_target,
    "ExileTargetCreatureWithPowerOrToughnessAtMost": _t_exile_target,
    # Bounce / return-to-hand family.
    "Bounce": _t_return_to_hand,
    "ReturnTargetCreatureToOwnersHand": _t_return_to_hand,
    "ReturnTargetPermanentYouControlToOwnersHand": _t_return_to_hand,
    # Tap family.
    "Tap": _t_tap_target,
    "TapAnotherTargetCreature": _t_tap_target,
    "TapUpToOneTargetCreature": _t_tap_target,
    "TapTargetArtifactOrCreature": _t_tap_target,
    # +1/+1 counter family (single-target subclasses).
    "PutPlusOneCounters": _t_put_plus_one_counter_on_target,
    "PutPlusOneCounterOnTargetCreature": _t_put_plus_one_counter_on_target,
    "PutPlusOneCounterOnCreatureYouControl": _t_put_plus_one_counter_on_target,
    "PutPlusOneCounterOnOpponentsCreature": _t_put_plus_one_counter_on_target,
    # Discard / mill family.
    "SelfMill": _t_self_mill,
    "TargetOpponentDiscards": _t_target_opponent_discards,
    # Targeted-life-loss variant.
    "TargetPlayerLosesLife": _t_target_player_loses_life,
}


# Translators that need extra context (e.g. a token registrar) live
# here; they accept the standard ``eff`` plus context keyword args.
_CONTEXTUAL_EFFECT_TRANSLATORS: dict[str, Any] = {
    "CreateToken": _t_create_token,
}


def _translate_one_effect(py_eff, *, token_registrar=None) -> dict | None:
    """Translate a single Python ``Effect`` instance to a Rust dict.

    Returns ``None`` when the effect's class is not in the closed
    translation table or when translation finds a Python-only widening
    (e.g. ``use_controller_life`` on MillTargetPlayer) the Rust slice
    can't honor.

    The ``token_registrar`` is threaded to translators that need to
    pre-register Rust printings (CreateToken). When ``None`` (e.g.
    the translator is called without a registry context), CreateToken
    and other registry-bound translators fall through to unsupported.
    """
    cls_name = type(py_eff).__name__
    fn = _EFFECT_TRANSLATORS.get(cls_name)
    if fn is not None:
        return fn(py_eff)
    ctx_fn = _CONTEXTUAL_EFFECT_TRANSLATORS.get(cls_name)
    if ctx_fn is not None:
        return ctx_fn(py_eff, token_registrar=token_registrar)
    return None


def _translate_effect_or_sequence(
    py_eff, *, token_registrar=None,
) -> tuple[list[dict], list[str]]:
    """Translate one Python effect (or a ``Sequence`` combinator) to a
    list of Rust effect dicts.

    Returns ``(translated_dicts, unsupported_class_names)``.

    ``Sequence`` is unpacked one level deep: each sub-effect is run
    through ``_translate_one_effect``; sub-effects that fail to
    translate are captured in the unsupported list (so the per-card
    stats correctly attribute the gap to the inner class, not the
    outer Sequence). Nested Sequences (rare) are not recursively
    unpacked — they fall through as unsupported under "Sequence".

    Non-Sequence effects pass through the single-effect translator
    directly.
    """
    if py_eff is None:
        return [], []
    cls_name = type(py_eff).__name__
    if cls_name == "Sequence":
        sub_effects = getattr(py_eff, "effects", ())
        translated: list[dict] = []
        unsupported: list[str] = []
        for sub in sub_effects:
            t = _translate_one_effect(sub, token_registrar=token_registrar)
            if t is not None:
                translated.append(t)
            else:
                unsupported.append(type(sub).__name__)
        return translated, unsupported
    t = _translate_one_effect(py_eff, token_registrar=token_registrar)
    if t is not None:
        return [t], []
    return [], [cls_name]


def _mana_cost_to_dict(mc) -> dict:
    """Translate a Python ``ManaCost`` -> Rust dict.

    Rust knows generic / W / U / B / R / G. Python's hybrid mana
    (``{W/U}`` style symbols) and x_pips (``{X}``) are Python-only —
    hybrid contributions land as +1 generic per hybrid symbol
    (matches the Python engine's own simplification), x_pips are
    dropped (no X-cost casting in the Rust first slice).
    """
    out: dict[str, int] = {}
    generic = int(getattr(mc, "generic", 0))
    # Each hybrid symbol contributes 1 to the mana value -- mirror that.
    generic += len(getattr(mc, "hybrid", ()))
    if generic > 0:
        out["generic"] = generic
    for color, count in getattr(mc, "colored", ()):
        if color in ("W", "U", "B", "R", "G"):
            out[color] = int(count)
        else:
            # Phyrexian / other non-WUBRG colored mana isn't in the
            # Rust vocab -- fold to generic as a coarse approximation.
            out["generic"] = out.get("generic", 0) + int(count)
    return out


def python_card_to_rust_recipe(py_card, *, token_registrar=None) -> dict:
    """Translate one Python ``CardPrinting`` to a Rust registry dict.

    Returned dict carries the keys ``parse_printing`` in rust/src/cards.rs
    consumes: ``name``, ``mana_cost``, ``card_types``, ``mana_produced``,
    ``keywords``, ``power``, ``toughness``, ``effects``, ``etb_effects``.
    Any Python-side concept the Rust closed vocab doesn't model
    (oracle_text, set_code, supertypes, subtypes, triggered_abilities
    beyond plain ETBs, activated_abilities beyond tap-for-mana,
    spell_effect nesting via ModalChooseOne) is silently dropped. The
    returned dict also carries an out-of-band ``_unsupported_effects``
    list of bare class names whose translation failed -- callers
    surface that in their stats roll-up.

    ``token_registrar`` is an optional callable used by translators
    that need to pre-register a sub-printing (CreateToken) with the
    enclosing Rust registry. It takes the Python ``CardPrinting`` of
    the token and returns the Rust printing id assigned to it. When
    ``None`` (e.g. ad-hoc translator calls without a registry yet),
    CreateToken effects fall through to unsupported.

    The translator is best-effort and lossy; per parity-smoke's
    measurement-first stance, a recipe that lands SOMETHING (an empty
    effects list with the right card_types / mana_cost / mana_produced)
    is more useful for the 1000-game smoke than refusing to translate
    cards with effects Rust doesn't know about yet.

    ``Sequence`` spell_effects are unpacked one level: each translatable
    sub-effect lands as a separate Rust effect; failed sub-effects are
    recorded individually in ``_unsupported_effects`` (so the per-card
    stat correctly attributes the gap to the inner class).
    """
    # Accept either a Python ``CardPrinting`` (real path) or a plain
    # dict (lets tests stub a card without instantiating the dataclass).
    if isinstance(py_card, dict):
        # Already in Rust-recipe shape -- pass through but stamp the
        # ``_unsupported_effects`` field as empty.
        out = dict(py_card)
        out.setdefault("_unsupported_effects", [])
        return out

    unsupported: list[str] = []

    # name -- always present on CardPrinting.
    name = py_card.name

    # mana_cost.
    mana_cost = _mana_cost_to_dict(py_card.mana_cost)

    # card_types.
    card_types = list(py_card.card_types)

    # mana_produced: Python ``produces`` is a frozenset[str] for lands
    # (e.g. {"G"}). Map directly. Empty set means "doesn't tap for mana".
    produced = sorted(py_card.produces) if py_card.produces else []
    # "any" wildcard color isn't in the Rust closed vocab -- drop it
    # to keep parse_printing happy. Same for nonstandard symbols.
    produced = [c for c in produced if c in ("W", "U", "B", "R", "G", "C")]

    # keywords: Python frozenset[str]. Rust expects lowercase strings.
    keywords = [k.lower() for k in py_card.keywords]

    # power / toughness: pass through; None when not a creature.
    power = py_card.power
    toughness = py_card.toughness

    # effects: from spell_effect (instants/sorceries) — walk one level
    # via the Sequence-unpacking helper. A bare effect yields [one_dict];
    # a Sequence yields the per-sub translation list; both update the
    # unsupported list with class names of sub-effects that couldn't
    # translate.
    effects: list[dict] = []
    spell_effect = getattr(py_card, "spell_effect", None)
    if spell_effect is not None:
        translated_list, sub_unsupported = _translate_effect_or_sequence(
            spell_effect, token_registrar=token_registrar,
        )
        effects.extend(translated_list)
        unsupported.extend(sub_unsupported)

    # etb_effects: walk triggered_abilities, pick out the ones whose
    # event is "enters the battlefield" and whose effect translates.
    # We discriminate ETB triggers cheaply by checking the
    # ``TriggeredAbility.event`` field's name attr; non-ETB triggers
    # (attacks, dies, end-of-step, etc.) are unsupported here.
    etb_effects: list[dict] = []
    for ta in getattr(py_card, "triggered_abilities", ()):
        event = getattr(ta, "event", None)
        event_name = getattr(event, "name", "") if event is not None else ""
        if event_name != "ENTERS":
            # Non-ETB triggers (IS_CAST, ATTACKS, DIES, etc.) have no
            # Rust counterpart yet -- record the inner-effect class
            # name as unsupported so the gap is attributed correctly.
            unsupported.append(type(ta.effect).__name__)
            continue
        translated_list, sub_unsupported = _translate_effect_or_sequence(
            ta.effect, token_registrar=token_registrar,
        )
        etb_effects.extend(translated_list)
        unsupported.extend(sub_unsupported)

    # activated_abilities: harvested implicitly via ``mana_produced``.
    # Non-mana activated abilities are unsupported and counted.
    for aa in getattr(py_card, "activated_abilities", ()):
        if getattr(aa, "is_mana_ability", False):
            continue
        unsupported.append(type(aa.effect).__name__)

    recipe: dict[str, Any] = {
        "name": name,
        "mana_cost": mana_cost,
        "card_types": card_types,
        "effects": effects,
        "etb_effects": etb_effects,
        "keywords": keywords,
        "_unsupported_effects": unsupported,
    }
    if produced:
        recipe["mana_produced"] = produced
    if power is not None:
        recipe["power"] = int(power)
    if toughness is not None:
        recipe["toughness"] = int(toughness)
    return recipe


def _load_python_decklist(deck_path: Path | str) -> list[str]:
    """Read a deck file and return a flat list of card names.

    Supported formats:

    - ``.txt`` / no-suffix / one-card-per-line: lines starting with ``#``
      ignored, blank lines ignored, otherwise the WHOLE LINE is the
      card name (so a card like "Bonecrusher Giant // Stomp" stays one
      token). The leading "4 " count syntax is recognized: a line like
      ``4 Forest`` is expanded to four ``Forest`` entries.
    - ``.json`` / ``.json.gz``: either a flat list of strings (the
      direct decklist) or a list of Deck dicts (uses the first deck's
      ``cards`` field via ``decks.Deck.from_dict``). Choice is by
      shape inspection.
    """
    path = Path(deck_path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data and isinstance(data[0], dict):
            from mtg_bot.decks import Deck
            return list(Deck.from_dict(data[0]).cards)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError(f"unsupported JSON deck shape at {path}: {type(data).__name__}")
    # Fallback: text-based one-per-line (or "<count> <name>") format.
    cards: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # "4 Forest" / "1 Lightning Bolt" -- parse leading count.
        head, _, rest = line.partition(" ")
        if head.isdigit() and rest:
            for _ in range(int(head)):
                cards.append(rest)
        else:
            cards.append(line)
    return cards


def make_rust_state_from_python_decklist(
    deck_path: Path | str | None = None,
    *,
    deck_card_names: Iterable[str] | None = None,
    shuffle_seed: int | None = None,
):
    """Build a Rust GameState by translating a Python deck file's cards
    through the Python card-data pipeline into Rust registry recipes.

    Either ``deck_path`` (read from disk) or ``deck_card_names`` (an
    iterable of card-name strings) must be provided; if both are given,
    ``deck_card_names`` wins. The same translated card pool is used as
    BOTH players' libraries (so the two seats share the same deck,
    Forest-mirror style); a future slice can split p0/p1.

    Each unique card name is looked up in the Python card source
    (``mtg_bot.cards.source.get_active_source().get_printings()``),
    translated to a Rust recipe via ``python_card_to_rust_recipe``, and
    registered in a fresh Rust registry. Built-in card names (Forest /
    Mountain / Plains / Island / Swamp / Llanowar Elves / Bears /
    Lightning Bolt) fall back to the curated builtin recipes when the
    Python source doesn't have them -- keeps the in-test
    "synthetic 60-card deck" path working without depending on the
    Scryfall data being current.

    Returns ``(rust_state, stats)`` where ``stats`` is a dict:

    - ``cards_loaded``: int -- number of unique card NAMES registered
      (not total cards in deck).
    - ``cards_with_unsupported_effects``: int -- number of unique
      card NAMES whose recipe carried at least one unsupported
      Python effect class.
    - ``unsupported_effect_classes``: list[str] -- sorted, deduped
      list of Python-effect class names encountered that the Rust
      bridge couldn't translate. THIS is the parity-vocabulary
      measurement: "100 of 60 cards have unsupported effects" is
      itself a real signal.
    - ``deck_size``: int -- total number of cards in the deck
      (= len of the loaded decklist).
    """
    import mtg_bot_engine

    if deck_card_names is not None:
        card_names = list(deck_card_names)
    elif deck_path is not None:
        card_names = _load_python_decklist(deck_path)
    else:
        raise ValueError(
            "make_rust_state_from_python_decklist: either deck_path or "
            "deck_card_names must be provided"
        )

    if not card_names:
        raise ValueError("deck is empty -- nothing to register")

    # Pull the Python card source once; fall back to builtins for
    # names not present (mirrors the synthetic-decklist test path).
    try:
        from mtg_bot.cards.source import get_active_source
        py_printings = get_active_source().get_printings()
    except Exception:  # pragma: no cover -- defensive: don't take the import seam down
        py_printings = {}

    reg = mtg_bot_engine.Registry()
    name_to_pid: dict[str, int] = {}
    unsupported_classes: set[str] = set()
    cards_with_unsupported = 0

    # Token printings are sub-printings referenced by CreateToken
    # effects on top-level cards. They need their own Rust registry
    # entries before any effect that references them. Cache by token
    # name so the same Drone / Robot / Soldier token (cross-card) only
    # registers once.
    token_name_to_pid: dict[str, int] = {}

    def _token_registrar(token_printing) -> int:
        tname = token_printing.name
        if tname in token_name_to_pid:
            return token_name_to_pid[tname]
        # Translate the token recursively (no nested CreateToken
        # expected; if one shows up, it falls through to unsupported
        # at the inner level rather than recursing further).
        sub_recipe = python_card_to_rust_recipe(
            token_printing, token_registrar=None,
        )
        sub_recipe.pop("_unsupported_effects", None)
        sub_pid = reg.register_printing(sub_recipe)
        token_name_to_pid[tname] = sub_pid
        return sub_pid

    unique_names: list[str] = []
    seen: set[str] = set()
    for n in card_names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)

    for name in unique_names:
        if name in py_printings:
            recipe = python_card_to_rust_recipe(
                py_printings[name], token_registrar=_token_registrar,
            )
        else:
            try:
                recipe = get_builtin_card_recipe(name)
                recipe = dict(recipe)
                recipe.setdefault("_unsupported_effects", [])
            except KeyError as e:
                raise KeyError(
                    f"card {name!r} not in Python card source and not in "
                    f"the built-in recipe table"
                ) from e
        per_card_unsup = recipe.pop("_unsupported_effects", [])
        if per_card_unsup:
            cards_with_unsupported += 1
            unsupported_classes.update(per_card_unsup)
        pid = reg.register_printing(recipe)
        name_to_pid[name] = pid

    p0_lib = [name_to_pid[n] for n in card_names]
    p1_lib = list(p0_lib)  # mirror -- both seats run the same deck

    state = mtg_bot_engine.make_state(
        reg, p0_lib, p1_lib, shuffle_seed=shuffle_seed,
    )

    stats = {
        "cards_loaded": len(unique_names),
        "cards_with_unsupported_effects": cards_with_unsupported,
        "unsupported_effect_classes": sorted(unsupported_classes),
        "deck_size": len(card_names),
    }
    return state, stats
