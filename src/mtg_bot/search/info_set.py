"""Information-set node identity for ISMCTS.

A single function ``info_set_key(state, viewer_id)`` that returns a
stable, collision-resistant byte digest. Two engine states (or
adapter states) the viewer cannot distinguish — i.e. ones that
differ only in cards in opponent hands or in either player's
library — must hash equal. Anything the viewer CAN distinguish must
hash to a different key with very high probability.

See ``docs/ISMCTS_MIGRATION.md`` (Phase 1) for the design rationale
and ``docs/HIDDEN_INFO_MODEL.md`` for the underlying knowledge model.

What the viewer knows is delegated to ``determinize.viewer_knows``:
the same predicate the determinizer uses to decide which cards may
be permuted. Anything ``viewer_knows`` returns True for is folded
into the hash by printing identity; for hidden cards we fold in only
the count.

Adapter-side state (combat phase, pending attackers/blockers,
resolution-pause kind) is NOT viewer-hidden in our model — both
players see who is attacking, who is being asked to block what, and
which spell currently has a pending resolution callback. Same
engine state at "declare attackers, one creature pending" vs "main
phase" is a different decision moment, so it must hash differently.

This module is read-only over the state — never mutates anything.

Performance target: well under 100us on a typical mid-game state
(50-100 game objects, ~30 hand+library cards per player).
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from mtg_bot.game import GameObject, GameState, Player
from mtg_bot.search.determinize import viewer_knows


_DIGEST_SIZE = 16


class _HashBuilder:
    """Length-prefixed byte buffer hashed once at the end with
    ``blake2b``.

    Length-prefixing each chunk prevents two different field
    sequences from colliding — e.g. ("", "abc") vs ("ab", "c") feed
    the same bytes without prefixes but different bytes with them.
    Type-prefixing (``b"i"``, ``b"s"``, ``b"b"``, ``b"n"``) similarly
    prevents collisions between different "shapes" with the same
    payload bytes.

    The buffer-then-digest design (rather than streaming
    ``hashlib.blake2b.update`` calls per field) is the perf
    critical-path choice — ``bytearray.extend`` of a small bytes
    object is roughly 2-3x cheaper than two ``update`` calls for
    each field, and ``info_set_key`` is on every ISMCTS iteration's
    hot path.
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def add_bytes(self, b: bytes) -> None:
        self._buf.extend(len(b).to_bytes(4, "big"))
        self._buf.extend(b)

    def add_int(self, v: int) -> None:
        self._buf.extend(b"\x00\x00\x00\x09i")
        self._buf.extend(v.to_bytes(8, "big", signed=True))

    def add_bool(self, v: bool) -> None:
        if v:
            self._buf.extend(b"\x00\x00\x00\x02b\x01")
        else:
            self._buf.extend(b"\x00\x00\x00\x02b\x00")

    def add_str(self, s: str) -> None:
        encoded = s.encode("utf-8")
        self._buf.extend((1 + len(encoded)).to_bytes(4, "big"))
        self._buf.extend(b"s")
        self._buf.extend(encoded)

    def add_none(self) -> None:
        self._buf.extend(b"\x00\x00\x00\x01n")

    def add_opt_int(self, v: int | None) -> None:
        if v is None:
            self._buf.extend(b"\x00\x00\x00\x01n")
        else:
            self._buf.extend(b"\x00\x00\x00\x09i")
            self._buf.extend(v.to_bytes(8, "big", signed=True))

    def add_opt_str(self, v: str | None) -> None:
        if v is None:
            self._buf.extend(b"\x00\x00\x00\x01n")
        else:
            encoded = v.encode("utf-8")
            self._buf.extend((1 + len(encoded)).to_bytes(4, "big"))
            self._buf.extend(b"s")
            self._buf.extend(encoded)

    def add_iter(self, marker: str, items: Iterable[Any]) -> None:
        """Mark the start of a collection, count its items, then feed
        each as raw bytes. The caller is expected to encode each item;
        this just wraps the iteration with a delimiter and a count so
        an empty list and a list-of-one-empty-element don't collide."""
        items_list = list(items)
        self.add_str(marker)
        self.add_int(len(items_list))
        for item in items_list:
            self.add_bytes(item)

    def section(self, label: str) -> None:
        """Emit a section delimiter so successive sections never
        collide structurally."""
        self.add_bytes(b"|" + label.encode("utf-8"))

    def digest(self) -> bytes:
        return hashlib.blake2b(
            bytes(self._buf), digest_size=_DIGEST_SIZE,
        ).digest()


def info_set_key(state: Any, viewer_id: int) -> bytes:
    """Stable, hashable identity for an information set.

    Accepts either a raw ``mtg_bot.game.GameState`` or an
    ``MtgGameStateAdapter``. Adapters carry combat-phase /
    resolution-pause bookkeeping that is folded into the hash on
    top of the engine state.

    Two states the viewer cannot distinguish (differ only in cards
    in opponent hands or in either library) return equal bytes.
    Anything the viewer CAN distinguish — public zones, life
    totals, mana pools, phase/step/priority, viewer's own hand
    contents, combat declarations, resolution pauses — affects the
    hash.

    Returns a 16-byte ``blake2b`` digest. Birthday-collision
    threshold is sqrt(2^64) ~= 4 billion items, far past any
    realistic search-tree size.
    """
    engine_state, adapter = _split_state(state)
    h = _HashBuilder()
    _hash_engine(h, engine_state, viewer_id)
    if adapter is not None:
        _hash_adapter(h, adapter)
    return h.digest()


def _split_state(state: Any) -> tuple[GameState, Any]:
    """Return ``(engine_state, adapter_or_None)``. Accepts a raw
    ``GameState`` or an ``MtgGameStateAdapter`` — the latter exposes
    the engine state via its ``engine_state`` property."""
    if isinstance(state, GameState):
        return state, None
    engine = getattr(state, "_state", None)
    if engine is None:
        engine = getattr(state, "engine_state", None)
    if engine is None or not isinstance(engine, GameState):
        raise TypeError(
            f"info_set_key expects GameState or MtgGameStateAdapter, "
            f"got {type(state).__name__}"
        )
    return engine, state


def _hash_engine(
    h: _HashBuilder, state: GameState, viewer_id: int
) -> None:
    h.section("turn")
    h.add_int(state.turn_number)
    h.add_int(state.active_player_idx)
    h.add_int(state.priority_player_idx)
    h.add_str(state.step.value if state.step is not None else "")

    h.section("flags")
    h.add_bool(state.permanent_left_battlefield_this_turn)
    h.add_bool(state.spell_warped_this_turn)
    h.add_bool(state.prevent_combat_damage_this_turn)
    h.add_bool(state.is_draw)
    h.add_opt_int(state.winner_id)

    h.section("players")
    for p in sorted(state.players, key=lambda x: x.id):
        _hash_player(h, p, viewer_id)

    h.section("battlefield")
    h.add_int(len(state.battlefield))
    for obj in state.battlefield:
        _hash_battlefield_object(h, obj)

    h.section("stack")
    h.add_int(len(state.stack))
    for item in state.stack:
        _hash_stack_item(h, item)


def _hash_player(
    h: _HashBuilder, p: Player, viewer_id: int
) -> None:
    h.section(f"p{p.id}")
    h.add_int(p.id)
    h.add_int(p.life)
    h.add_int(p.starting_life)
    h.add_int(p.poison_counters)
    h.add_bool(p.has_lost)
    h.add_str(p.lost_reason)
    h.add_int(p.lands_played_this_turn)
    h.add_int(p.lands_entered_this_turn)
    h.add_int(p.spells_cast_this_turn)
    h.add_int(p.noncreature_spells_cast_this_turn)
    h.add_int(p.artifacts_entered_under_your_control_this_turn)
    h.add_int(p.life_gained_this_turn)

    h.section("mana")
    h.add_int(len(p.mana_pool))
    for coin in _canonicalize_mana_pool(p.mana_pool):
        h.add_str(coin[0])
        h.add_opt_str(coin[1])

    h.section("hand")
    h.add_int(len(p.hand))
    if p.id == viewer_id:
        for card in p.hand:
            _hash_hidden_or_visible_card(h, card, viewer_id)
    else:
        for card in p.hand:
            if viewer_knows(card, viewer_id):
                _hash_hidden_or_visible_card(h, card, viewer_id)
            else:
                h.add_str("?hidden")

    h.section("library")
    h.add_int(len(p.library))
    for card in p.library:
        if viewer_knows(card, viewer_id):
            _hash_hidden_or_visible_card(h, card, viewer_id)
        else:
            h.add_str("?hidden")

    h.section("graveyard")
    h.add_int(len(p.graveyard))
    for card in p.graveyard:
        _hash_zone_card(h, card)

    h.section("exile")
    h.add_int(len(p.exile))
    for card in p.exile:
        _hash_zone_card(h, card)

    h.section("attacked_this_turn")
    h.add_int(len(p.attacked_this_turn))
    for obj in p.attacked_this_turn:
        h.add_int(obj.id)


def _canonicalize_mana_pool(
    coins: Iterable[Any],
) -> list[tuple[str, str | None]]:
    """Canonicalize the mana pool to a sorted list of (color, restriction)
    pairs. Mana coins are interchangeable within a color+restriction
    bucket, so order in the underlying list is not viewer-distinguishing
    information."""
    out = [(c.color, c.spend_restriction) for c in coins]
    out.sort(key=lambda t: (t[0], t[1] or ""))
    return out


def _hash_hidden_or_visible_card(
    h: _HashBuilder, card: GameObject, viewer_id: int,
) -> None:
    """Used in the viewer's own hand and for explicitly-revealed cards
    in opponent hands or libraries — places where we know the printing
    but the card is otherwise face-down (no battlefield-side state)."""
    h.add_str(card.printing.name)
    if card.original_printing is not None:
        h.add_str(card.original_printing.name)
    else:
        h.add_none()
    h.add_int(card.id)
    h.add_bool(card.warped)
    h.add_bool(card.cast_from_exile_with_lightstall_surcharge)
    h.add_bool(card.play_from_exile_requires_kavu)
    h.add_bool(card.playable_from_exile_this_turn)
    h.add_bool(card.cast_via_warp)


def _hash_zone_card(h: _HashBuilder, card: GameObject) -> None:
    """Hash a card in a public-but-non-battlefield zone (graveyard,
    exile). Printing is public; we also fold in flags relevant to
    cast-from-here legality so two same-printing cards with different
    play permissions hash differently."""
    h.add_str(card.printing.name)
    if card.original_printing is not None:
        h.add_str(card.original_printing.name)
    else:
        h.add_none()
    h.add_int(card.id)
    h.add_int(card.owner_id)
    h.add_int(card.controller_id)
    h.add_str(card.zone.value)
    h.add_bool(card.warped)
    h.add_bool(card.cast_from_exile_with_lightstall_surcharge)
    h.add_bool(card.play_from_exile_requires_kavu)
    h.add_bool(card.playable_from_exile_this_turn)
    h.add_bool(card.face_down)
    h.add_opt_int(card.exiled_by_source)


def _hash_battlefield_object(h: _HashBuilder, obj: GameObject) -> None:
    """Hash a battlefield permanent. Battlefield is fully public, so
    every public flag goes into the hash. We over-include rather than
    under-include — over-inclusion makes information sets finer
    (acts like the viewer is slightly more informed); under-inclusion
    silently merges sets and is the dangerous direction for ISMCTS
    soundness."""
    h.add_int(obj.id)
    h.add_str(obj.printing.name)
    if obj.original_printing is not None:
        h.add_str(obj.original_printing.name)
    else:
        h.add_none()
    h.add_int(obj.owner_id)
    h.add_int(obj.controller_id)
    h.add_str(obj.zone.value)
    h.add_bool(obj.tapped)
    h.add_int(obj.damage)
    h.add_bool(obj.summoning_sick)
    h.add_bool(obj.attacking)
    h.add_opt_int(obj.blocking)
    h.add_iter(
        "blocked_by",
        (i.to_bytes(8, "big", signed=True) for i in obj.blocked_by),
    )
    h.add_bool(obj.marked_lethal)
    h.add_bool(obj.damaged_this_turn)
    h.add_int(obj.plus_one_counters)
    h.add_int(obj.minus_one_counters)
    h.add_int(obj.charge_counters)
    h.add_int(obj.time_counters)
    h.add_bool(obj.impending_active)
    h.add_int(obj.stun_counters)
    h.add_int(obj.temp_power_bonus)
    h.add_int(obj.temp_toughness_bonus)
    h.add_opt_int(obj.attached_to)
    h.add_opt_int(obj.exiled_by_source)
    h.add_opt_int(obj.controller_id_before_any_aura_swap)
    h.add_int(obj.loyalty_counters)
    h.add_int(obj.lore_counters)
    h.add_bool(obj.regeneration_shield)
    h.add_bool(obj.phased_out)
    h.add_int(obj.etb_timestamp)
    h.add_opt_str(obj.painter_chosen_color)
    h.add_bool(obj.face_down)
    h.add_iter(
        "unlocked_halves",
        (s.encode("utf-8") for s in sorted(obj.unlocked_halves)),
    )
    h.add_bool(obj.enduring_returned_as_enchantment)
    h.add_int(obj.flying_counters)
    h.add_bool(obj.has_activated_once_only_ability)
    h.add_bool(obj.permanent_is_creature)
    h.add_bool(obj.warped)
    h.add_bool(obj.cast_via_warp)
    h.add_bool(obj.cast_via_impending)
    h.add_bool(obj.cast_via_hand)
    h.add_bool(obj.playable_from_exile_this_turn)
    h.add_bool(obj.play_from_exile_requires_kavu)
    h.add_bool(obj.cast_from_exile_with_lightstall_surcharge)


def _hash_stack_item(h: _HashBuilder, item: Any) -> None:
    """Hash a stack item. Source identity, controller, ability index,
    chosen targets/modes, and X are all public from the moment the
    spell/ability hits the stack."""
    h.add_int(item.source.id)
    h.add_str(item.source.printing.name)
    h.add_int(item.controller_id)
    if item.ability is None:
        h.add_str("spell")
    else:
        h.add_str("ability")
        h.add_str(type(item.ability).__name__)
    h.add_iter(
        "targets",
        (
            (
                t[0].encode("utf-8") + b":" +
                t[1].to_bytes(8, "big", signed=True)
            )
            for t in item.targets
        ),
    )
    h.add_bool(item.countered)
    h.add_int(item.x_value)
    h.add_bool(item.kicked)
    h.add_opt_int(item.additional_cost_choice_id)
    h.add_int(item.combat_damage_amount)
    h.add_opt_int(item.combat_damage_source_controller_id)
    h.add_int(item.counter_amount)
    h.add_iter(
        "chosen_modes",
        (
            i.to_bytes(8, "big", signed=True)
            for i in item.chosen_modes
        ),
    )
    h.add_opt_str(item.cast_room_half)
    h.add_iter(
        "cost_paid",
        (
            o.id.to_bytes(8, "big", signed=True)
            for o in item.cost_paid_objects
        ),
    )


def _hash_adapter(h: _HashBuilder, adapter: Any) -> None:
    """Fold the adapter's combat-phase / resolution-pause state in.

    Two states with the same engine state but different combat-phase
    are NOT information-set-equivalent: a "main phase, no attackers
    pending" state has a different decision space than "declare
    attackers, one creature pending". Same for resolution-pause —
    a paused-on-target-choice adapter and a not-paused one expose
    completely different legal_decisions().
    """
    h.section("adapter")
    h.add_int(adapter._pass_count)
    h.add_bool(adapter._step_entered)

    combat_phase = adapter._combat_phase
    h.add_opt_str(combat_phase)
    h.add_iter(
        "pending_attackers",
        (
            i.to_bytes(8, "big", signed=True)
            for i in adapter._pending_attackers
        ),
    )
    h.add_iter(
        "chosen_attackers",
        (
            i.to_bytes(8, "big", signed=True)
            for i in adapter._chosen_attackers
        ),
    )
    h.add_iter(
        "pending_blockers",
        (
            i.to_bytes(8, "big", signed=True)
            for i in adapter._pending_blockers
        ),
    )
    h.add_iter(
        "chosen_blocks",
        (
            (
                k.to_bytes(8, "big", signed=True) + b":" +
                v.to_bytes(8, "big", signed=True)
            )
            for k, v in sorted(adapter._chosen_blocks.items())
        ),
    )
    h.add_opt_int(adapter._combat_defender_id)

    h.add_bool(adapter._resolution_paused)
    h.add_opt_str(adapter._resolution_kind)
    h.add_opt_int(adapter._resolution_player)
    # Resolution options are arbitrary tuples/values — use repr for
    # determinism. Engine emits ints/tuples-of-ints/booleans; repr
    # is stable across runs for those.
    h.add_str(repr(adapter._resolution_options))
    # Answers already given within the pending action (see the same
    # note in ``canonical_state``).
    channel = getattr(adapter, "_channel_agent", None)
    h.add_str(repr(channel.queue) if channel is not None else "")


__all__ = ["info_set_key"]
