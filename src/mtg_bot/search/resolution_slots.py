"""Policy-slot layout of the resolution block (action encoding
``canonical_v3``) and the per-pause option -> slot assignment.

``mtg_bot.training.action_encoding`` appends this block after the 517
``canonical_v2`` slots (see ``docs/RESOLUTION_DECISIONS.md`` §4). The
layout and the slot assignment live here, on the light engine-only side
of the package, because the assignment is computed by
``ResolutionChannelAgent`` at the moment the engine asks -- from the
LIVE mid-resolution state. The adapter then rolls back to its
pre-action snapshot, in which the objects a choice refers to may not be
where they were (an ETB trigger's own creature is not on the snapshot's
battlefield yet; a card drawn earlier in the same resolution is still in
the snapshot's library). Recording each option's slot at pause time
keeps ``encode`` / ``decode`` / ``legal_action_mask`` exact without
touching either state again.

Offsets returned here are relative to the start of the resolution
block; ``action_encoding`` adds its base (517). ``None`` means the
option has no slot (past a cap, or a target outside the vocabulary):
the option stays a legal search child with prior 0, and the event is
counted (:func:`overflow_counts`) and logged once per kind -- never a
silent drop.

Positions are relative to the DECIDER (the player the engine callback
asked).
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Optional

from mtg_bot.search.canonical_state import (
    _build_canonical_battlefield_index, _canonical_zone_card,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Caps. The battlefield / graveyard / exile / hand caps equal the
# canonical_v2 constants in ``action_encoding`` (a test pins that).
# ---------------------------------------------------------------------

BF_CAP = 24
GY_CAP = 24
EXILE_CAP = 24
HAND_CAP = 16
STACK_CAP = 8
MODE_CAP = 8
MODES_CAP = 4
X_CAP = 20
LIBRARY_OPTION_CAP = 16
SACRIFICE_CARD_CAP = 8

# ---------------------------------------------------------------------
# Block layout (offsets from the resolution-block base).
# ---------------------------------------------------------------------

TGT_PLAYER = 0                              # 0 = decider, 1 = opponent
TGT_BF_OWN = TGT_PLAYER + 2
TGT_BF_OPP = TGT_BF_OWN + BF_CAP
TGT_STACK = TGT_BF_OPP + BF_CAP             # position from the top
TGT_GY_OWN = TGT_STACK + STACK_CAP
TGT_GY_OPP = TGT_GY_OWN + GY_CAP
TGT_EXILE_OWN = TGT_GY_OPP + GY_CAP
TGT_EXILE_OPP = TGT_EXILE_OWN + EXILE_CAP
TGT_DECLINE = TGT_EXILE_OPP + EXILE_CAP
DISCARD = TGT_DECLINE + 1
MODE = DISCARD + HAND_CAP
MODES = MODE + MODE_CAP                     # (bitmask - 1)
SACRIFICE = MODES + (2 ** MODES_CAP - 1)
# ``choose_sacrifice`` doubles as the engine's generic "pick an id"
# hook (Close Encounter designates a warped creature card in exile,
# Anticausal Vestige puts a card from hand onto the battlefield):
# options that are not the decider's permanents rank by canonical key.
SACRIFICE_CARD = SACRIFICE + BF_CAP
X = SACRIFICE_CARD + SACRIFICE_CARD_CAP     # X = 0 .. X_CAP
SURVEIL = X + X_CAP + 1                     # 0 = mill, 1 = keep
SHOCK_LIFE = SURVEIL + 2                    # 0 = pay, 1 = don't
YES_NO = SHOCK_LIFE + 2                     # 0 = yes, 1 = no
CARD_FROM_LIBRARY = YES_NO + 2              # rank; +LIBRARY_OPTION_CAP = decline
CARD_FROM_LIBRARY_DECLINE = CARD_FROM_LIBRARY + LIBRARY_OPTION_CAP
CARD_FROM_OPP_HAND = CARD_FROM_LIBRARY_DECLINE + 1
DEVOUR = CARD_FROM_OPP_HAND + HAND_CAP      # 0 = none, 1 = all
RESOLUTION_BLOCK_SIZE = DEVOUR + 2
"""Number of slots in the resolution block. Pinned by a test."""

# Family name -> (offset, size), for docs / tests / diagnostics.
FAMILIES: dict[str, tuple[int, int]] = {
    "target_player": (TGT_PLAYER, 2),
    "target_bf_own": (TGT_BF_OWN, BF_CAP),
    "target_bf_opp": (TGT_BF_OPP, BF_CAP),
    "target_stack": (TGT_STACK, STACK_CAP),
    "target_gy_own": (TGT_GY_OWN, GY_CAP),
    "target_gy_opp": (TGT_GY_OPP, GY_CAP),
    "target_exile_own": (TGT_EXILE_OWN, EXILE_CAP),
    "target_exile_opp": (TGT_EXILE_OPP, EXILE_CAP),
    "target_decline": (TGT_DECLINE, 1),
    "discard": (DISCARD, HAND_CAP),
    "mode": (MODE, MODE_CAP),
    "modes": (MODES, 2 ** MODES_CAP - 1),
    "sacrifice": (SACRIFICE, BF_CAP),
    "sacrifice_card": (SACRIFICE_CARD, SACRIFICE_CARD_CAP),
    "x": (X, X_CAP + 1),
    "surveil": (SURVEIL, 2),
    "shock_life": (SHOCK_LIFE, 2),
    "yes_no": (YES_NO, 2),
    "card_from_library": (CARD_FROM_LIBRARY, LIBRARY_OPTION_CAP + 1),
    "card_from_opponent_hand": (CARD_FROM_OPP_HAND, HAND_CAP),
    "devour_lands": (DEVOUR, 2),
}

# ---------------------------------------------------------------------
# Overflow / collision accounting (process-wide, per pause kind).
# ---------------------------------------------------------------------

_OVERFLOWS: collections.Counter = collections.Counter()
_COLLISIONS: collections.Counter = collections.Counter()
_WARNED: set[str] = set()


def overflow_counts() -> dict[str, int]:
    """Options (summed over pause events) that got no slot, by kind."""
    return dict(_OVERFLOWS)


def collision_counts() -> dict[str, int]:
    """Options (summed over pause events) that shared a slot with an
    earlier option of the same pause, by kind."""
    return dict(_COLLISIONS)


def reset_counts() -> None:
    """Test hook: clear the process-wide counters and warn-once set."""
    _OVERFLOWS.clear()
    _COLLISIONS.clear()
    _WARNED.clear()


def _note(counter: collections.Counter, kind: str, what: str,
          detail: str) -> None:
    counter[kind] += 1
    key = f"{what}:{kind}"
    if key not in _WARNED:
        _WARNED.add(key)
        _log.warning(
            "resolution slot %s (%s): %s -- the option stays a legal "
            "search child with prior 0; counted in "
            "mtg_bot.search.resolution_slots.%s_counts() (warned once "
            "per kind)", what, kind, detail, what,
        )


# ---------------------------------------------------------------------
# Option normalisation (dedup) -- applied by the channel agent before
# the options are exposed.
# ---------------------------------------------------------------------


def card_key(state: Any, card_id: int) -> Optional[tuple]:
    """Canonical zone key (name + modifier flags, no id) of the card
    with ``card_id`` in any zone of ``state``; ``None`` if absent."""
    card = find_card(state, card_id)
    return None if card is None else _canonical_zone_card(card)


def find_card(state: Any, card_id: int):
    """The ``GameObject`` with ``card_id`` in any player zone or on the
    battlefield, else ``None``."""
    for p in state.players:
        for zone in (p.hand, p.library, p.graveyard, p.exile):
            for c in zone:
                if c.id == card_id:
                    return c
    for o in state.battlefield:
        if o.id == card_id:
            return o
    return None


def dedupe_cards(state: Any, card_ids) -> tuple:
    """Keep the first id per distinct canonical card key (two same-key
    cards are interchangeable). ``None`` entries (decline) pass
    through once. Ids whose card can't be found are kept as-is."""
    out: list = []
    seen: set = set()
    for cid in card_ids:
        if cid is None:
            key: Any = ("<decline>",)
        else:
            k = card_key(state, cid)
            key = ("<id>", cid) if k is None else k
        if key in seen:
            continue
        seen.add(key)
        out.append(cid)
    return tuple(out)


def dedupe_exact(options) -> tuple:
    """Drop exact duplicate options, keeping order."""
    out: list = []
    seen: set = set()
    for o in options:
        if o in seen:
            continue
        seen.add(o)
        out.append(o)
    return tuple(out)


# ---------------------------------------------------------------------
# Slot assignment
# ---------------------------------------------------------------------


class _Positions:
    """Lazily-built canonical positions for one pause's live state."""

    def __init__(self, state: Any, decider: int) -> None:
        self.state = state
        self.decider = decider
        self._bf: Optional[dict[int, tuple[bool, int]]] = None
        self._zone: dict[tuple[str, int], dict[int, int]] = {}

    def battlefield(self, obj_id: int) -> Optional[tuple[bool, int]]:
        """``(is_own, position)`` of a battlefield object in the
        canonical battlefield order, per controller side."""
        if self._bf is None:
            bf = self.state.battlefield
            indices, _, _ = _build_canonical_battlefield_index(self.state)
            counts = {True: 0, False: 0}
            self._bf = {}
            for i in indices:
                o = bf[i]
                own = o.controller_id == self.decider
                self._bf[o.id] = (own, counts[own])
                counts[own] += 1
        return self._bf.get(obj_id)

    def zone(self, zone: str, player_id: int, card_id: int) -> Optional[int]:
        key = (zone, player_id)
        table = self._zone.get(key)
        if table is None:
            cards = list(getattr(
                self.state.player_by_id(player_id), zone,
            ))
            cards.sort(key=lambda c: (_canonical_zone_card(c), c.id))
            table = {c.id: i for i, c in enumerate(cards)}
            self._zone[key] = table
        return table.get(card_id)


def _target_offset(
    target, pos: _Positions,
) -> tuple[Optional[int], str]:
    state = pos.state
    if target is None:
        return TGT_DECLINE, ""
    kind, tid = target
    player_ids = {p.id for p in state.players}
    if kind == "player":
        if tid not in player_ids:
            return None, f"unknown player id {tid}"
        return TGT_PLAYER + (0 if tid == pos.decider else 1), ""
    if kind == "stack":
        n = len(state.stack)
        if not (0 <= tid < n):
            return None, f"stack index {tid} outside a {n}-item stack"
        from_top = n - 1 - tid
        if from_top >= STACK_CAP:
            return None, f"stack position {from_top} >= STACK_CAP"
        return TGT_STACK + from_top, ""
    on_bf = pos.battlefield(tid)
    if on_bf is not None:
        own, p = on_bf
        if p >= BF_CAP:
            return None, f"battlefield position {p} >= BF_CAP"
        return (TGT_BF_OWN if own else TGT_BF_OPP) + p, ""
    for zone, own_base, opp_base, cap in (
        ("graveyard", TGT_GY_OWN, TGT_GY_OPP, GY_CAP),
        ("exile", TGT_EXILE_OWN, TGT_EXILE_OPP, EXILE_CAP),
    ):
        for pl in state.players:
            p = pos.zone(zone, pl.id, tid)
            if p is None:
                continue
            if p >= cap:
                return None, f"{zone} position {p} >= cap {cap}"
            own = pl.id == pos.decider
            return (own_base if own else opp_base) + p, ""
    if tid in player_ids:
        # ``any_damage_target``-style kinds carry player ids too; the
        # engine resolves the battlefield first, as above.
        return TGT_PLAYER + (0 if tid == pos.decider else 1), ""
    return None, f"target {target!r} is not a player, stack item, " \
                 f"battlefield, graveyard or exile object"


def _rank_offsets(
    state: Any, options: tuple, base: int, cap: int,
    decline: Optional[int] = None,
) -> list[tuple[Optional[int], str]]:
    keyed = []
    for cid in options:
        if cid is None:
            continue
        k = card_key(state, cid)
        # Unknown cards (not found in any zone) sort last, by id.
        keyed.append(((k is None, k if k is not None else (), cid), cid))
    keyed.sort(key=lambda kv: kv[0])
    rank = {cid: i for i, (_, cid) in enumerate(keyed)}
    out: list[tuple[Optional[int], str]] = []
    for cid in options:
        if cid is None:
            out.append((decline, "" if decline is not None else "decline"))
            continue
        r = rank[cid]
        if r >= cap:
            out.append((None, f"rank {r} >= cap {cap}"))
        else:
            out.append((base + r, ""))
    return out


def compute_offsets(
    kind: str, state: Any, decider: int, options: tuple,
) -> tuple[Optional[int], ...]:
    """Slot offset (within the resolution block) for every option of a
    pause, computed from the LIVE ``state`` the engine passed to the
    callback. Overflows and codec collisions are counted."""
    results: list[tuple[Optional[int], str]]
    if kind == "target":
        pos = _Positions(state, decider)
        results = [_target_offset(t, pos) for t in options]
    elif kind == "sacrifice":
        pos = _Positions(state, decider)
        results = []
        cards = tuple(
            oid for oid in options if pos.battlefield(oid) is None
        )
        card_offsets = dict(zip(cards, _rank_offsets(
            state, cards, SACRIFICE_CARD, SACRIFICE_CARD_CAP,
        )))
        for oid in options:
            on_bf = pos.battlefield(oid)
            if on_bf is None:
                results.append(card_offsets[oid])
            elif not on_bf[0]:
                results.append((None, f"object {oid} not the decider's"))
            elif on_bf[1] >= BF_CAP:
                results.append((None, f"position {on_bf[1]} >= BF_CAP"))
            else:
                results.append((SACRIFICE + on_bf[1], ""))
    elif kind == "discard":
        results = _rank_offsets(state, options, DISCARD, HAND_CAP)
    elif kind == "card_from_opponent_hand":
        results = _rank_offsets(state, options, CARD_FROM_OPP_HAND, HAND_CAP)
    elif kind == "card_from_library":
        results = _rank_offsets(
            state, options, CARD_FROM_LIBRARY, LIBRARY_OPTION_CAP,
            decline=CARD_FROM_LIBRARY_DECLINE,
        )
    elif kind == "mode":
        results = [
            (MODE + i, "") if 0 <= i < MODE_CAP
            else (None, f"mode {i} >= MODE_CAP")
            for i in options
        ]
    elif kind == "modes":
        results = []
        for subset in options:
            if not subset or any(not (0 <= i < MODES_CAP) for i in subset):
                results.append((None, f"modes {subset!r} outside 0..3"))
            else:
                mask = sum(1 << i for i in set(subset))
                results.append((MODES + mask - 1, ""))
    elif kind == "x":
        results = [
            (X + v, "") if 0 <= v <= X_CAP else (None, f"X={v} > X_CAP")
            for v in options
        ]
    elif kind in ("surveil", "shock_life", "yes_no"):
        base = {"surveil": SURVEIL, "shock_life": SHOCK_LIFE,
                "yes_no": YES_NO}[kind]
        results = [(base + (0 if v else 1), "") for v in options]
    elif kind == "devour_lands":
        results = [(DEVOUR + (1 if subset else 0), "") for subset in options]
    else:
        raise ValueError(f"unknown resolution kind {kind!r}")
    offsets: list[Optional[int]] = []
    seen: set[int] = set()
    for off, why in results:
        if off is None:
            _note(_OVERFLOWS, kind, "overflow", why)
        elif off in seen:
            _note(_COLLISIONS, kind, "collision",
                  f"two options share offset {off}")
        else:
            seen.add(off)
        offsets.append(off)
    return tuple(offsets)


__all__ = [
    "FAMILIES",
    "RESOLUTION_BLOCK_SIZE",
    "card_key",
    "collision_counts",
    "compute_offsets",
    "dedupe_cards",
    "dedupe_exact",
    "find_card",
    "overflow_counts",
    "reset_counts",
]
