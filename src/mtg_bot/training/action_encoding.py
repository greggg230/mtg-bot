"""Bucketed fixed-vocab encoding of MTG ``Decision`` objects.

Converts the engine-side ``Decision`` surface (priority actions plus
factored combat sub-decisions) into a fixed-size integer index space
suitable for an AlphaZero-style policy head. Provides three operations:

- ``encode(decision, state, viewer)`` → int slot index.
- ``decode(slot_idx, state, viewer)`` → ``Decision`` (or raises).
- ``legal_action_mask(state, viewer)`` → bool array of size
  ``ACTION_VOCAB_SIZE``.

The codec is bijective on the legal set within a single state: for
every legal decision the engine produces, ``decode(encode(d, s, v), s,
v) == d``. Slot indices reference *positions* within the viewing
player's zones (or the shared battlefield) rather than concrete card
ids — so the encoding is stable across different printings, IDs, and
engine refactors that don't change zone semantics.

Slot taxonomy (see ``ACTION_VOCAB_SIZE`` total):

    [    0]                                            PASS_PRIORITY
    [    1 ..   16]   16 slots   LAND_FROM_HAND[hand_pos in 0..15]
    [   17 ..   40]   24 slots   LAND_FROM_GRAVEYARD[gy_pos in 0..23]
    [   41 ..   64]   24 slots   LAND_FROM_EXILE[exile_pos in 0..23]
    [   65 ..  176]  112 slots   CAST_FROM_HAND[hand_pos][variant]
                                  variant: 0=normal 1=kicked 2=warp
                                           3=cast_for_free 4=use_impending
                                           5=cast_room_half[0]
                                           6=cast_room_half[1]
    [  177 ..  224]   48 slots   CAST_FROM_GRAVEYARD[gy_pos][variant]
                                  variant: 0=normal/warp 1=cast_for_free
    [  225 ..  272]   48 slots   CAST_FROM_EXILE[exile_pos][variant]
                                  variant: 0=normal 1=cast_for_free
    [  273]                                            CAST_FROM_LIBRARY_TOP
    [  274 ..  321]   48 slots   UNLOCK_ROOM_HALF[bf_own_pos][half_idx]
    [  322 ..  417]   96 slots   ACTIVATE_ABILITY[bf_own_pos][ability_idx]
    [  418 ..  481]   64 slots   ACTIVATE_STATION[ship_idx in 0..3][crew_pos in 0..15]
    [  482 ..  505]   24 slots   TURN_FACE_UP[bf_own_pos]
    [  506]                                            ATTACKER_ATTACK
    [  507]                                            ATTACKER_SKIP
    [  508 ..  516]    9 slots   BLOCKER[attacker_pos in 0..7] (+ 1 skip)

Total: 517 slots. ``ACTION_VOCAB_SIZE`` is the public constant (the
``positional_v1`` / ``canonical_v2`` vocabulary).

``canonical_v3`` appends the RESOLUTION block (280 slots, 517..796;
``ACTION_VOCAB_SIZE_V3`` = 797): one family per mid-resolution choice
kind (targets by zone and side, discard, mode(s), sacrifice, X,
surveil, shock life, yes/no, tutor / opponent-hand picks, devour).
Layout, caps and canonicalisation: ``mtg_bot.search.resolution_slots``
and ``docs/RESOLUTION_DECISIONS.md``. Each option's slot is computed by
the resolution channel agent from the live state at pause time and
read back here from the paused adapter
(``adapter.resolution_slot_offsets``). Use
``action_vocab_size(version)`` for a version's width.

Position conventions (referenced by all slot families above):

- ``hand_pos``: index into ``state.player_by_id(viewer).hand``.
- ``gy_pos`` / ``exile_pos``: index into the viewer's graveyard / exile.
- ``bf_own_pos``: index into ``[o for o in state.battlefield if
  o.controller_id == viewer]`` (controller-filtered, in battlefield order).
- ``ship_idx``: index into ``[o for o in viewer's battlefield if
  o.printing.is_spacecraft]``.
- ``crew_pos``: index into ``[o for o in viewer's battlefield if
  o.printing.is_creature]``. (NB: creatures form a different slot
  space than ``bf_own_pos``; both are used because Spacecraft slot
  one ship and one crewing creature.)
- ``half_idx``: 0 or 1, indexing into the printing's ``room_halves``
  sorted by name.
- ``attacker_pos``: index into the currently-attacking creatures sorted
  by ``object id`` ascending. Only valid in the BLOCKERS combat phase.

Set-valued combat decisions (``DeclareAttackers`` / ``DeclareBlockers``)
are emitted by the *adapter*, not by the policy. The policy emits one
factored sub-decision per pending creature (``AttackerDecision`` /
``BlockerDecision``); the adapter accumulates and finalizes.

State-tied identity: encoding is *positional*. If a hand is reordered
between encode and decode the round-trip would break — but the engine
never reorders zones mid-decision, so within a single priority moment
the codec is bijective on the legal set. Slot legality (i.e.
``legal_action_mask``) is recomputed from scratch every call so it
always matches the engine's current ``legal_decisions``.

Versioning (canonicality):

The codec carries a ``version`` parameter selecting the slot-mapping
convention:

- ``"positional_v1"`` (default, back-compat with v13/v15/v16 weights):
  hand / graveyard / exile positions are taken from the engine's raw
  zone list. Two states differing only in the ordering of card.ids
  (canonically equivalent transpositions) produce DIFFERENT slot-to-
  decision mappings even though their value/policy logits are
  identical. The MCTS NN-eval cache (``mcts._cache_key``) defends
  against the resulting drift via a slot_fingerprint extension to its
  cache key — see commit 2f50fca.

- ``"canonical_v2"``: hand / graveyard / exile positions are taken
  AFTER sorting each zone by the same forward-relevant tuple
  ``canonical_state._canonical_zone_card`` uses (printing.name +
  per-card modifier flags), with raw card.id as the final tiebreaker
  for determinism within identical canonicalized cards. Two states
  whose canonical hash collides also carry identical slot-to-decision
  mappings. v2 makes the slot_fingerprint cache workaround a no-op.

Migration: v13/v15/v16 metadata.json doesn't carry an
``action_encoding_version`` field, so loaders default it to
``"positional_v1"`` and behavior is bit-identical to before this
change. v17+ training runs opt into v2 by passing
``--action-encoding-version positional_v1|canonical_v2`` to
``mtg_bot train`` (or by writing the field directly into
``ModelMetadata.action_encoding_version``); the registry persists the
choice and ``NNEvaluator`` / ``NNPimcConfig`` thread it through to
every ``legal_action_mask`` call at inference time.

Stability guarantees:

- ``ACTION_VOCAB_SIZE`` is module-level and immutable. Pinned by a
  test so we notice if a cap edit changes it.
- Slot semantics are positional, not card-id-based. New EoE printings
  do not move existing slots.
- Caps (HAND_SIZE_CAP, BATTLEFIELD_OWN_CAP, etc.) are documented and
  surveyed against real games (50-game sweep against random agents
  showed: max hand=9, max own-controller battlefield=23, max graveyard=19,
  max exile=25, max attackers in any block phase=4 — all within caps).
- Going over a cap during encode raises ``EncodingOverflowError`` so the
  caller can decide (training-time: drop the sample; inference-time:
  fall back to tactical policy). Decoding an illegal slot raises
  ``IllegalSlotError``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mtg_bot.actions import (
    Action, ActivateAbility, ActivateStation, CastSpell, PassPriority,
    PlayLand, TurnFaceUp, UnlockRoomHalf,
)
from mtg_bot.combat import eligible_attackers, eligible_blockers
from mtg_bot.game import GameState as EngineGameState
from mtg_bot.legal import legal_actions
from mtg_bot.search import resolution_slots
from mtg_bot.search.combat_decisions import AttackerDecision, BlockerDecision
from mtg_bot.search.protocol import PlayerID
from mtg_bot.search.resolution_decisions import (
    ResolutionCardFromLibraryDecision,
    ResolutionCardFromOpponentHandDecision,
    ResolutionDevourLandsDecision, ResolutionDiscardDecision,
    ResolutionModeDecision, ResolutionModesDecision,
    ResolutionSacrificeDecision, ResolutionShockLifeDecision,
    ResolutionSurveilDecision, ResolutionTargetDecision,
    ResolutionXDecision, ResolutionYesNoDecision,
)

# ---------------------------------------------------------------------
# Capacity caps. Picked once, pinned by tests. Going over a cap at
# encode-time raises ``EncodingOverflowError``; the legal-mask helper
# silently truncates (anything beyond the cap is unreachable through
# the policy head).
# ---------------------------------------------------------------------

HAND_SIZE_CAP = 16
"""Hand positions slotted. Observed max ~9; 16 is generous."""

GRAVEYARD_CAP = 24
"""Graveyard positions slotted. Observed max ~19."""

EXILE_CAP = 24
"""Exile positions slotted. Observed max ~25 in pathological games;
extend if a card pool change pushes deeper."""

BATTLEFIELD_OWN_CAP = 24
"""Per-controller battlefield positions slotted. Observed max ~23."""

ABILITY_INDEX_CAP = 4
"""Activated abilities per source. Engine sources today have ≤2."""

CAST_VARIANT_HAND_CAP = 7
"""Cast variants from hand: normal / kicked / warp / cast_for_free /
use_impending / cast_room_half[0] / cast_room_half[1]."""

CAST_VARIANT_GY_EXILE_CAP = 2
"""Cast variants from graveyard / exile: normal / cast_for_free."""

ROOM_HALF_CAP = 2
"""Halves on a Room. CR 715 fixes this at 2."""

SPACECRAFT_CAP = 4
"""Spacecraft permanents controlled. Typically ≤1."""

CREW_CAP = 16
"""Creatures eligible to crew. Capped low; observed max ~4 attackers
in any block phase, similar order of magnitude here."""

BLOCK_ATTACKERS_CAP = 8
"""Distinct attackers a blocker may choose between (+1 for ``skip``)."""

# Cast variant ids (CAST_FROM_HAND).
_VAR_NORMAL = 0
_VAR_KICKED = 1
_VAR_WARP = 2
_VAR_FREE = 3
_VAR_IMPENDING = 4
_VAR_ROOM_HALF_0 = 5
_VAR_ROOM_HALF_1 = 6

# Cast variant ids (CAST_FROM_GRAVEYARD / CAST_FROM_EXILE).
_GE_VAR_NORMAL = 0  # whatever the zone's normal-cast path is
_GE_VAR_FREE = 1


# ---------------------------------------------------------------------
# Slot ranges, computed once. Ordering matches the docstring.
# ---------------------------------------------------------------------

SLOT_PASS = 0

SLOT_LAND_FROM_HAND_BASE = SLOT_PASS + 1
SLOT_LAND_FROM_HAND_END = SLOT_LAND_FROM_HAND_BASE + HAND_SIZE_CAP

SLOT_LAND_FROM_GY_BASE = SLOT_LAND_FROM_HAND_END
SLOT_LAND_FROM_GY_END = SLOT_LAND_FROM_GY_BASE + GRAVEYARD_CAP

SLOT_LAND_FROM_EXILE_BASE = SLOT_LAND_FROM_GY_END
SLOT_LAND_FROM_EXILE_END = SLOT_LAND_FROM_EXILE_BASE + EXILE_CAP

SLOT_CAST_FROM_HAND_BASE = SLOT_LAND_FROM_EXILE_END
SLOT_CAST_FROM_HAND_END = (
    SLOT_CAST_FROM_HAND_BASE + HAND_SIZE_CAP * CAST_VARIANT_HAND_CAP
)

SLOT_CAST_FROM_GY_BASE = SLOT_CAST_FROM_HAND_END
SLOT_CAST_FROM_GY_END = (
    SLOT_CAST_FROM_GY_BASE + GRAVEYARD_CAP * CAST_VARIANT_GY_EXILE_CAP
)

SLOT_CAST_FROM_EXILE_BASE = SLOT_CAST_FROM_GY_END
SLOT_CAST_FROM_EXILE_END = (
    SLOT_CAST_FROM_EXILE_BASE + EXILE_CAP * CAST_VARIANT_GY_EXILE_CAP
)

SLOT_CAST_FROM_LIBRARY_TOP = SLOT_CAST_FROM_EXILE_END

SLOT_UNLOCK_ROOM_BASE = SLOT_CAST_FROM_LIBRARY_TOP + 1
SLOT_UNLOCK_ROOM_END = (
    SLOT_UNLOCK_ROOM_BASE + BATTLEFIELD_OWN_CAP * ROOM_HALF_CAP
)

SLOT_ACTIVATE_ABILITY_BASE = SLOT_UNLOCK_ROOM_END
SLOT_ACTIVATE_ABILITY_END = (
    SLOT_ACTIVATE_ABILITY_BASE
    + BATTLEFIELD_OWN_CAP * ABILITY_INDEX_CAP
)

SLOT_ACTIVATE_STATION_BASE = SLOT_ACTIVATE_ABILITY_END
SLOT_ACTIVATE_STATION_END = (
    SLOT_ACTIVATE_STATION_BASE + SPACECRAFT_CAP * CREW_CAP
)

SLOT_TURN_FACE_UP_BASE = SLOT_ACTIVATE_STATION_END
SLOT_TURN_FACE_UP_END = SLOT_TURN_FACE_UP_BASE + BATTLEFIELD_OWN_CAP

SLOT_ATTACKER_ATTACK = SLOT_TURN_FACE_UP_END
SLOT_ATTACKER_SKIP = SLOT_ATTACKER_ATTACK + 1

SLOT_BLOCKER_BASE = SLOT_ATTACKER_SKIP + 1
# BLOCK_ATTACKERS_CAP attackers + 1 skip slot.
SLOT_BLOCKER_SKIP = SLOT_BLOCKER_BASE + BLOCK_ATTACKERS_CAP
SLOT_BLOCKER_END = SLOT_BLOCKER_SKIP + 1

ACTION_VOCAB_SIZE: int = SLOT_BLOCKER_END
"""Slot count of ``positional_v1`` / ``canonical_v2`` (517). Pinned by
a test. ``canonical_v3`` extends it -- see ``action_vocab_size``."""

SLOT_RESOLUTION_BASE: int = ACTION_VOCAB_SIZE
"""First slot of the ``canonical_v3`` resolution block."""

ACTION_VOCAB_SIZE_V3: int = (
    SLOT_RESOLUTION_BASE + resolution_slots.RESOLUTION_BLOCK_SIZE
)
"""Slot count of ``canonical_v3`` (797). Pinned by a test."""


# ---------------------------------------------------------------------
# Versioning. See module docstring "Versioning (canonicality)" for the
# semantic difference between v1 and v2.
# ---------------------------------------------------------------------

ACTION_ENCODING_VERSION_V1 = "positional_v1"
"""Legacy positional-by-card.id slot mapping. Default; v13/v15/v16
trained weights memorized this convention."""

ACTION_ENCODING_VERSION_V2 = "canonical_v2"
"""Canonical sort-by-(printing.name, modifier_tuple, card.id) slot
mapping. v17+ training runs opt in via metadata."""

ACTION_ENCODING_VERSION_V3 = "canonical_v3"
"""``canonical_v2`` + the resolution block (targets, modes, X, ...).
v2's 517 slots keep their indices and meaning; see
``docs/RESOLUTION_DECISIONS.md``."""

SUPPORTED_ACTION_ENCODING_VERSIONS = (
    ACTION_ENCODING_VERSION_V1,
    ACTION_ENCODING_VERSION_V2,
    ACTION_ENCODING_VERSION_V3,
)

_CANONICAL_ZONE_VERSIONS = frozenset({
    ACTION_ENCODING_VERSION_V2, ACTION_ENCODING_VERSION_V3,
})
"""Versions whose hand / graveyard / exile positions are canonically
sorted (v3 inherits v2's base families unchanged)."""

_RESOLUTION_DECISION_TYPES = (
    ResolutionTargetDecision, ResolutionDiscardDecision,
    ResolutionModeDecision, ResolutionModesDecision,
    ResolutionSacrificeDecision, ResolutionXDecision,
    ResolutionSurveilDecision, ResolutionShockLifeDecision,
    ResolutionYesNoDecision, ResolutionCardFromLibraryDecision,
    ResolutionCardFromOpponentHandDecision,
    ResolutionDevourLandsDecision,
)


def action_vocab_size(version: str) -> int:
    """Policy-head width of an action-encoding version: 517 for
    ``positional_v1`` / ``canonical_v2``, 797 for ``canonical_v3``."""
    _check_version(version)
    if version == ACTION_ENCODING_VERSION_V3:
        return ACTION_VOCAB_SIZE_V3
    return ACTION_VOCAB_SIZE


def has_resolution_slots(version: str) -> bool:
    """Whether ``version`` can encode ``Resolution*Decision``s."""
    _check_version(version)
    return version == ACTION_ENCODING_VERSION_V3


def is_resolution_decision(decision: Any) -> bool:
    """True for every search-side mid-resolution answer type."""
    return isinstance(decision, _RESOLUTION_DECISION_TYPES)


def _is_resolution_paused(state: Any) -> bool:
    """Adapter at a mid-resolution pause (duck-typed; raw engine
    states and the Rust provider never pause)."""
    return bool(getattr(state, "_resolution_paused", False))


def _check_version(version: str) -> None:
    """Validate ``version`` against the supported set. Raises
    ``ValueError`` early with a helpful message rather than silently
    treating an unknown string as v1."""
    if version not in SUPPORTED_ACTION_ENCODING_VERSIONS:
        raise ValueError(
            f"action_encoding_version must be one of "
            f"{SUPPORTED_ACTION_ENCODING_VERSIONS}; got {version!r}"
        )


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class EncodingOverflowError(ValueError):
    """A decision references a position past one of the caps. Bumping
    the offending cap and rerunning ``test_vocab_size_constant_is_fixed``
    is the fix; that test is what alerts you that ``ACTION_VOCAB_SIZE``
    has shifted."""


class IllegalSlotError(ValueError):
    """The slot index has no legal decoded interpretation in the
    current state. Raised by ``decode`` when the slot doesn't decode
    to a legal decision (e.g., empty hand position, no permanent at
    that battlefield index)."""


# ---------------------------------------------------------------------
# State accessors. The codec accepts either an engine ``GameState`` or
# an ``MtgGameStateAdapter`` — duck-typed via ``getattr``. The adapter
# is preferred because combat-phase decoding needs its
# ``_pending_attackers`` / ``_pending_blockers`` lists; the raw engine
# state is fine for non-combat priority moments.
# ---------------------------------------------------------------------


def _engine_state(state: Any) -> EngineGameState:
    """Return the wrapped engine state from either the adapter or the
    state itself. Lets a single helper handle both call sites."""
    return getattr(state, "engine_state", state)


def _combat_phase(state: Any) -> Optional[str]:
    """Adapter-aware combat-phase tag. Engine states (no adapter) can't
    surface combat sub-decisions, so they return ``None``."""
    return getattr(state, "combat_phase", None)


def _own_battlefield(eng: EngineGameState, viewer: PlayerID) -> list:
    """Permanents the viewer controls, in battlefield order."""
    return [o for o in eng.battlefield if o.controller_id == viewer]


def _own_creatures(eng: EngineGameState, viewer: PlayerID) -> list:
    return [
        o for o in eng.battlefield
        if o.controller_id == viewer and o.printing.is_creature
    ]


def _own_spacecraft(eng: EngineGameState, viewer: PlayerID) -> list:
    return [
        o for o in eng.battlefield
        if o.controller_id == viewer and o.printing.is_spacecraft
    ]


def _canonical_zone_sort_key(card: Any) -> tuple:
    """Sort key applied to hand / graveyard / exile entries under v2.

    Mirrors the forward-relevant tuple ``canonical_state._canonical_
    zone_card`` builds (printing.name + per-card modifier flags); the
    raw ``card.id`` is appended as the final tiebreaker so two
    canonically-equivalent cards (same printing.name + same modifier
    tuple) get a deterministic slot ordering. Without the id tiebreak
    the v2 sort would still be canonical-equivalent (the two slots
    would be interchangeable from any forward-play perspective), but a
    deterministic order keeps unit-test asserts simple and matches the
    tie-breaking convention the engine uses elsewhere
    (``_attacking_creatures`` sorts by id).
    """
    return (
        card.printing.name,
        getattr(card, "warped", False),
        getattr(card, "cast_via_warp", False),
        getattr(card, "face_down", False),
        getattr(card, "playable_from_exile_this_turn", False),
        getattr(card, "play_from_exile_requires_kavu", False),
        getattr(card, "cast_from_exile_with_lightstall_surcharge", False),
        card.id,
    )


def _hand(
    eng: EngineGameState, viewer: PlayerID,
    version: str = ACTION_ENCODING_VERSION_V1,
) -> list:
    """Return the viewer's hand ordered per ``version``.

    v1: raw engine list order (unchanged from pre-canonical days).
    v2: sorted by ``_canonical_zone_sort_key`` so two states that
    differ only by card.id permutation produce the same slot mapping.
    """
    cards = list(eng.player_by_id(viewer).hand)
    if version in _CANONICAL_ZONE_VERSIONS:
        return sorted(cards, key=_canonical_zone_sort_key)
    return cards


def _graveyard(
    eng: EngineGameState, viewer: PlayerID,
    version: str = ACTION_ENCODING_VERSION_V1,
) -> list:
    cards = list(eng.player_by_id(viewer).graveyard)
    if version in _CANONICAL_ZONE_VERSIONS:
        return sorted(cards, key=_canonical_zone_sort_key)
    return cards


def _exile(
    eng: EngineGameState, viewer: PlayerID,
    version: str = ACTION_ENCODING_VERSION_V1,
) -> list:
    cards = list(eng.player_by_id(viewer).exile)
    if version in _CANONICAL_ZONE_VERSIONS:
        return sorted(cards, key=_canonical_zone_sort_key)
    return cards


def _library_top(eng: EngineGameState, viewer: PlayerID):
    lib = eng.player_by_id(viewer).library
    return lib[0] if lib else None


def _attacking_creatures(eng: EngineGameState) -> list:
    """Sort by id for clone determinism, matching the adapter's
    ``_pending_blockers`` enumeration convention."""
    return sorted(
        (o for o in eng.battlefield if o.attacking),
        key=lambda o: o.id,
    )


def _cast_variant_for_hand(d: CastSpell, card) -> int:
    """Pick which CAST_FROM_HAND variant a CastSpell maps to. Order
    matches the precedence in ``legal.py`` — kicked / warp /
    cast_for_free / impending are mutually exclusive, then room
    halves are a separate axis."""
    if d.cast_room_half is not None:
        # Resolve to half index via sorted printing halves.
        halves = sorted(
            h.name for h in card.printing.room_halves
        )
        try:
            half_idx = halves.index(d.cast_room_half)
        except ValueError as exc:
            raise EncodingOverflowError(
                f"cast_room_half={d.cast_room_half!r} not in printing's "
                f"halves {halves}"
            ) from exc
        if half_idx == 0:
            return _VAR_ROOM_HALF_0
        if half_idx == 1:
            return _VAR_ROOM_HALF_1
        raise EncodingOverflowError(
            f"Room has >2 halves, cap is {ROOM_HALF_CAP}"
        )
    if d.kicked:
        return _VAR_KICKED
    if d.use_warp:
        return _VAR_WARP
    if d.cast_for_free:
        return _VAR_FREE
    if d.use_impending:
        return _VAR_IMPENDING
    return _VAR_NORMAL


def _cast_variant_for_gy_or_exile(d: CastSpell) -> int:
    """Cast-from-graveyard / cast-from-exile share a 2-variant axis:
    the zone's normal cast path (which may itself be warp/lightstall/
    impulse depending on the source flag) vs. a free cast offered
    when Weftwalking is active."""
    if d.cast_for_free:
        return _GE_VAR_FREE
    return _GE_VAR_NORMAL


def _find_card_zone(
    eng: EngineGameState, viewer: PlayerID, card_id: int,
    version: str = ACTION_ENCODING_VERSION_V1,
) -> tuple[str, int, Any]:
    """Locate ``card_id`` in the viewer's hand / graveyard / exile or
    on top of library. Returns ``(zone_name, position, card)`` or
    raises if not found.

    ``position`` is taken from the version-appropriate zone view: raw
    engine order under v1, ``_canonical_zone_sort_key``-sorted order
    under v2. Library top is unaffected (it's slot
    ``SLOT_CAST_FROM_LIBRARY_TOP``, not a positional family).

    Zones searched mirror ``legal.py``'s CastSpell sources."""
    p = eng.player_by_id(viewer)
    for i, c in enumerate(_hand(eng, viewer, version=version)):
        if c.id == card_id:
            return ("hand", i, c)
    for i, c in enumerate(_graveyard(eng, viewer, version=version)):
        if c.id == card_id:
            return ("graveyard", i, c)
    for i, c in enumerate(_exile(eng, viewer, version=version)):
        if c.id == card_id:
            return ("exile", i, c)
    if p.library and p.library[0].id == card_id:
        return ("library_top", 0, p.library[0])
    raise EncodingOverflowError(
        f"card_id={card_id} not found in viewer={viewer}'s hand/gy/exile/"
        f"library top"
    )


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------


def encode(
    decision: Any, state: Any, viewer: PlayerID,
    version: str = ACTION_ENCODING_VERSION_V1,
) -> int:
    """Convert ``decision`` into a slot index in ``[0, ACTION_VOCAB_SIZE)``.

    ``state`` may be an engine ``GameState`` or an ``MtgGameStateAdapter``;
    both are duck-typed for the engine bits. Combat-phase factored
    decisions (``AttackerDecision`` / ``BlockerDecision``) work with
    either, but their *legality* is only correctly enumerable from the
    adapter's pending lists, so prefer the adapter for the legal-mask
    helper.

    ``version`` selects the slot-mapping convention; defaults to
    ``ACTION_ENCODING_VERSION_V1`` for backward compatibility. See the
    module docstring "Versioning (canonicality)" for the v1-vs-v2
    semantic difference. Raises ``ValueError`` for an unknown version
    string.

    Raises ``EncodingOverflowError`` if a referenced position exceeds
    its cap (see module-level cap constants)."""
    _check_version(version)
    if is_resolution_decision(decision):
        return _encode_resolution(decision, state, viewer, version)
    eng = _engine_state(state)

    if isinstance(decision, PassPriority):
        return SLOT_PASS

    if isinstance(decision, PlayLand):
        zone, pos, _ = _find_card_zone(
            eng, viewer, decision.card_id, version=version,
        )
        if zone == "hand":
            if pos >= HAND_SIZE_CAP:
                raise EncodingOverflowError(
                    f"hand_pos={pos} exceeds HAND_SIZE_CAP={HAND_SIZE_CAP}"
                )
            return SLOT_LAND_FROM_HAND_BASE + pos
        if zone == "graveyard":
            if pos >= GRAVEYARD_CAP:
                raise EncodingOverflowError(
                    f"gy_pos={pos} exceeds GRAVEYARD_CAP={GRAVEYARD_CAP}"
                )
            return SLOT_LAND_FROM_GY_BASE + pos
        if zone == "exile":
            if pos >= EXILE_CAP:
                raise EncodingOverflowError(
                    f"exile_pos={pos} exceeds EXILE_CAP={EXILE_CAP}"
                )
            return SLOT_LAND_FROM_EXILE_BASE + pos
        raise EncodingOverflowError(
            f"PlayLand from zone={zone!r} not in vocab"
        )

    if isinstance(decision, CastSpell):
        zone, pos, card = _find_card_zone(
            eng, viewer, decision.card_id, version=version,
        )
        if zone == "hand":
            if pos >= HAND_SIZE_CAP:
                raise EncodingOverflowError(
                    f"hand_pos={pos} exceeds HAND_SIZE_CAP={HAND_SIZE_CAP}"
                )
            v = _cast_variant_for_hand(decision, card)
            return (
                SLOT_CAST_FROM_HAND_BASE + pos * CAST_VARIANT_HAND_CAP + v
            )
        if zone == "graveyard":
            if pos >= GRAVEYARD_CAP:
                raise EncodingOverflowError(
                    f"gy_pos={pos} exceeds GRAVEYARD_CAP={GRAVEYARD_CAP}"
                )
            v = _cast_variant_for_gy_or_exile(decision)
            return (
                SLOT_CAST_FROM_GY_BASE
                + pos * CAST_VARIANT_GY_EXILE_CAP + v
            )
        if zone == "exile":
            if pos >= EXILE_CAP:
                raise EncodingOverflowError(
                    f"exile_pos={pos} exceeds EXILE_CAP={EXILE_CAP}"
                )
            v = _cast_variant_for_gy_or_exile(decision)
            return (
                SLOT_CAST_FROM_EXILE_BASE
                + pos * CAST_VARIANT_GY_EXILE_CAP + v
            )
        if zone == "library_top":
            return SLOT_CAST_FROM_LIBRARY_TOP
        raise EncodingOverflowError(
            f"CastSpell from zone={zone!r} not in vocab"
        )

    if isinstance(decision, UnlockRoomHalf):
        own = _own_battlefield(eng, viewer)
        try:
            bf_pos = next(
                i for i, o in enumerate(own) if o.id == decision.card_id
            )
        except StopIteration as exc:
            raise EncodingOverflowError(
                f"UnlockRoomHalf source id={decision.card_id} not in "
                f"viewer's battlefield"
            ) from exc
        if bf_pos >= BATTLEFIELD_OWN_CAP:
            raise EncodingOverflowError(
                f"bf_own_pos={bf_pos} exceeds BATTLEFIELD_OWN_CAP="
                f"{BATTLEFIELD_OWN_CAP}"
            )
        room = own[bf_pos]
        halves = sorted(h.name for h in room.printing.room_halves)
        try:
            half_idx = halves.index(decision.half_name)
        except ValueError as exc:
            raise EncodingOverflowError(
                f"half_name={decision.half_name!r} not in {halves}"
            ) from exc
        if half_idx >= ROOM_HALF_CAP:
            raise EncodingOverflowError(
                f"half_idx={half_idx} exceeds ROOM_HALF_CAP={ROOM_HALF_CAP}"
            )
        return (
            SLOT_UNLOCK_ROOM_BASE + bf_pos * ROOM_HALF_CAP + half_idx
        )

    if isinstance(decision, ActivateAbility):
        own = _own_battlefield(eng, viewer)
        try:
            bf_pos = next(
                i for i, o in enumerate(own) if o.id == decision.source_id
            )
        except StopIteration as exc:
            raise EncodingOverflowError(
                f"ActivateAbility source id={decision.source_id} not in "
                f"viewer's battlefield"
            ) from exc
        if bf_pos >= BATTLEFIELD_OWN_CAP:
            raise EncodingOverflowError(
                f"bf_own_pos={bf_pos} exceeds BATTLEFIELD_OWN_CAP="
                f"{BATTLEFIELD_OWN_CAP}"
            )
        if decision.ability_index >= ABILITY_INDEX_CAP:
            raise EncodingOverflowError(
                f"ability_index={decision.ability_index} exceeds "
                f"ABILITY_INDEX_CAP={ABILITY_INDEX_CAP}"
            )
        return (
            SLOT_ACTIVATE_ABILITY_BASE
            + bf_pos * ABILITY_INDEX_CAP + decision.ability_index
        )

    if isinstance(decision, ActivateStation):
        ships = _own_spacecraft(eng, viewer)
        try:
            ship_idx = next(
                i for i, o in enumerate(ships)
                if o.id == decision.spacecraft_id
            )
        except StopIteration as exc:
            raise EncodingOverflowError(
                f"ActivateStation spacecraft id={decision.spacecraft_id} "
                f"not in viewer's spacecraft"
            ) from exc
        if ship_idx >= SPACECRAFT_CAP:
            raise EncodingOverflowError(
                f"ship_idx={ship_idx} exceeds SPACECRAFT_CAP={SPACECRAFT_CAP}"
            )
        creatures = _own_creatures(eng, viewer)
        try:
            crew_pos = next(
                i for i, o in enumerate(creatures)
                if o.id == decision.crew_id
            )
        except StopIteration as exc:
            raise EncodingOverflowError(
                f"ActivateStation crew id={decision.crew_id} not in "
                f"viewer's creatures"
            ) from exc
        if crew_pos >= CREW_CAP:
            raise EncodingOverflowError(
                f"crew_pos={crew_pos} exceeds CREW_CAP={CREW_CAP}"
            )
        return (
            SLOT_ACTIVATE_STATION_BASE + ship_idx * CREW_CAP + crew_pos
        )

    if isinstance(decision, TurnFaceUp):
        own = _own_battlefield(eng, viewer)
        try:
            bf_pos = next(
                i for i, o in enumerate(own) if o.id == decision.card_id
            )
        except StopIteration as exc:
            raise EncodingOverflowError(
                f"TurnFaceUp card id={decision.card_id} not in "
                f"viewer's battlefield"
            ) from exc
        if bf_pos >= BATTLEFIELD_OWN_CAP:
            raise EncodingOverflowError(
                f"bf_own_pos={bf_pos} exceeds BATTLEFIELD_OWN_CAP="
                f"{BATTLEFIELD_OWN_CAP}"
            )
        return SLOT_TURN_FACE_UP_BASE + bf_pos

    if isinstance(decision, AttackerDecision):
        # The current attacker is implicit (the head of the adapter's
        # pending list). We only record attack/skip; decode rebuilds
        # the attacker_id from state.
        return SLOT_ATTACKER_ATTACK if decision.attack else SLOT_ATTACKER_SKIP

    if isinstance(decision, BlockerDecision):
        if decision.attacker_id is None:
            return SLOT_BLOCKER_SKIP
        attackers = _attacking_creatures(eng)
        try:
            atk_pos = next(
                i for i, o in enumerate(attackers)
                if o.id == decision.attacker_id
            )
        except StopIteration as exc:
            raise EncodingOverflowError(
                f"BlockerDecision attacker_id={decision.attacker_id} "
                f"not in attacking creatures"
            ) from exc
        if atk_pos >= BLOCK_ATTACKERS_CAP:
            raise EncodingOverflowError(
                f"attacker_pos={atk_pos} exceeds BLOCK_ATTACKERS_CAP="
                f"{BLOCK_ATTACKERS_CAP}"
            )
        return SLOT_BLOCKER_BASE + atk_pos

    raise EncodingOverflowError(
        f"don't know how to encode decision type {type(decision).__name__}: "
        f"{decision!r}"
    )


# ---------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------


def _slot_in(slot: int, base: int, end: int) -> bool:
    return base <= slot < end


def decode(
    slot_idx: int, state: Any, viewer: PlayerID,
    version: str = ACTION_ENCODING_VERSION_V1,
) -> Any:
    """Reconstruct the ``Decision`` corresponding to ``slot_idx``.

    ``version`` must match the version used at encode-time. Decoding a
    slot encoded under v2 with a v1 decoder (or vice versa) is
    undefined behavior — the decoder will return the wrong card.id at
    that position. The caller is responsible for threading the version
    through (``NNEvaluator`` does this for inference; tests pass it
    explicitly).

    Raises ``IllegalSlotError`` when the slot's referenced position
    isn't filled in the current state (e.g., the hand has fewer cards
    than ``hand_pos``). Use ``legal_action_mask`` first if you want to
    avoid this — it's exact."""
    _check_version(version)
    vocab = action_vocab_size(version)
    if slot_idx < 0 or slot_idx >= vocab:
        raise IllegalSlotError(
            f"slot_idx={slot_idx} not in [0, {vocab}) for {version}"
        )
    if slot_idx >= SLOT_RESOLUTION_BASE:
        return _decode_resolution(slot_idx, state, viewer)
    eng = _engine_state(state)

    if slot_idx == SLOT_PASS:
        return PassPriority()

    if _slot_in(slot_idx, SLOT_LAND_FROM_HAND_BASE, SLOT_LAND_FROM_HAND_END):
        pos = slot_idx - SLOT_LAND_FROM_HAND_BASE
        hand = _hand(eng, viewer, version=version)
        if pos >= len(hand):
            raise IllegalSlotError(
                f"LAND_FROM_HAND pos={pos} but hand has {len(hand)} cards"
            )
        return PlayLand(card_id=hand[pos].id)

    if _slot_in(slot_idx, SLOT_LAND_FROM_GY_BASE, SLOT_LAND_FROM_GY_END):
        pos = slot_idx - SLOT_LAND_FROM_GY_BASE
        gy = _graveyard(eng, viewer, version=version)
        if pos >= len(gy):
            raise IllegalSlotError(
                f"LAND_FROM_GRAVEYARD pos={pos} but gy has {len(gy)} cards"
            )
        return PlayLand(card_id=gy[pos].id)

    if _slot_in(slot_idx, SLOT_LAND_FROM_EXILE_BASE, SLOT_LAND_FROM_EXILE_END):
        pos = slot_idx - SLOT_LAND_FROM_EXILE_BASE
        exile = _exile(eng, viewer, version=version)
        if pos >= len(exile):
            raise IllegalSlotError(
                f"LAND_FROM_EXILE pos={pos} but exile has {len(exile)} cards"
            )
        return PlayLand(card_id=exile[pos].id)

    if _slot_in(slot_idx, SLOT_CAST_FROM_HAND_BASE, SLOT_CAST_FROM_HAND_END):
        offset = slot_idx - SLOT_CAST_FROM_HAND_BASE
        pos, variant = divmod(offset, CAST_VARIANT_HAND_CAP)
        hand = _hand(eng, viewer, version=version)
        if pos >= len(hand):
            raise IllegalSlotError(
                f"CAST_FROM_HAND pos={pos} but hand has {len(hand)} cards"
            )
        card = hand[pos]
        return _cast_from_variant_hand(card, variant)

    if _slot_in(slot_idx, SLOT_CAST_FROM_GY_BASE, SLOT_CAST_FROM_GY_END):
        offset = slot_idx - SLOT_CAST_FROM_GY_BASE
        pos, variant = divmod(offset, CAST_VARIANT_GY_EXILE_CAP)
        gy = _graveyard(eng, viewer, version=version)
        if pos >= len(gy):
            raise IllegalSlotError(
                f"CAST_FROM_GRAVEYARD pos={pos} but gy has {len(gy)} cards"
            )
        card = gy[pos]
        if variant == _GE_VAR_FREE:
            return CastSpell(card_id=card.id, cast_for_free=True)
        # Normal cast from graveyard means warp-cast (Timeline Culler).
        # The engine accepts CastSpell(id, use_warp=True) at sorcery
        # speed; legality is checked separately by the mask helper.
        return CastSpell(card_id=card.id, use_warp=True)

    if _slot_in(slot_idx, SLOT_CAST_FROM_EXILE_BASE, SLOT_CAST_FROM_EXILE_END):
        offset = slot_idx - SLOT_CAST_FROM_EXILE_BASE
        pos, variant = divmod(offset, CAST_VARIANT_GY_EXILE_CAP)
        exile = _exile(eng, viewer, version=version)
        if pos >= len(exile):
            raise IllegalSlotError(
                f"CAST_FROM_EXILE pos={pos} but exile has {len(exile)} cards"
            )
        card = exile[pos]
        if variant == _GE_VAR_FREE:
            return CastSpell(card_id=card.id, cast_for_free=True)
        return CastSpell(card_id=card.id)

    if slot_idx == SLOT_CAST_FROM_LIBRARY_TOP:
        top = _library_top(eng, viewer)
        if top is None:
            raise IllegalSlotError("library is empty")
        return CastSpell(card_id=top.id)

    if _slot_in(slot_idx, SLOT_UNLOCK_ROOM_BASE, SLOT_UNLOCK_ROOM_END):
        offset = slot_idx - SLOT_UNLOCK_ROOM_BASE
        bf_pos, half_idx = divmod(offset, ROOM_HALF_CAP)
        own = _own_battlefield(eng, viewer)
        if bf_pos >= len(own):
            raise IllegalSlotError(
                f"UNLOCK_ROOM bf_pos={bf_pos} but own bf has {len(own)}"
            )
        room = own[bf_pos]
        halves = sorted(h.name for h in room.printing.room_halves)
        if half_idx >= len(halves):
            raise IllegalSlotError(
                f"UNLOCK_ROOM half_idx={half_idx} but printing has "
                f"{len(halves)} halves"
            )
        return UnlockRoomHalf(card_id=room.id, half_name=halves[half_idx])

    if _slot_in(
        slot_idx, SLOT_ACTIVATE_ABILITY_BASE, SLOT_ACTIVATE_ABILITY_END,
    ):
        offset = slot_idx - SLOT_ACTIVATE_ABILITY_BASE
        bf_pos, ability_idx = divmod(offset, ABILITY_INDEX_CAP)
        own = _own_battlefield(eng, viewer)
        if bf_pos >= len(own):
            raise IllegalSlotError(
                f"ACTIVATE_ABILITY bf_pos={bf_pos} but own bf has {len(own)}"
            )
        return ActivateAbility(
            source_id=own[bf_pos].id, ability_index=ability_idx,
        )

    if _slot_in(
        slot_idx, SLOT_ACTIVATE_STATION_BASE, SLOT_ACTIVATE_STATION_END,
    ):
        offset = slot_idx - SLOT_ACTIVATE_STATION_BASE
        ship_idx, crew_pos = divmod(offset, CREW_CAP)
        ships = _own_spacecraft(eng, viewer)
        creatures = _own_creatures(eng, viewer)
        if ship_idx >= len(ships):
            raise IllegalSlotError(
                f"ACTIVATE_STATION ship_idx={ship_idx} but viewer has "
                f"{len(ships)} spacecraft"
            )
        if crew_pos >= len(creatures):
            raise IllegalSlotError(
                f"ACTIVATE_STATION crew_pos={crew_pos} but viewer has "
                f"{len(creatures)} creatures"
            )
        return ActivateStation(
            spacecraft_id=ships[ship_idx].id,
            crew_id=creatures[crew_pos].id,
        )

    if _slot_in(slot_idx, SLOT_TURN_FACE_UP_BASE, SLOT_TURN_FACE_UP_END):
        bf_pos = slot_idx - SLOT_TURN_FACE_UP_BASE
        own = _own_battlefield(eng, viewer)
        if bf_pos >= len(own):
            raise IllegalSlotError(
                f"TURN_FACE_UP bf_pos={bf_pos} but own bf has {len(own)}"
            )
        return TurnFaceUp(card_id=own[bf_pos].id)

    if slot_idx == SLOT_ATTACKER_ATTACK or slot_idx == SLOT_ATTACKER_SKIP:
        # The attacker_id is implicit: the head of the adapter's
        # pending list.
        head_id = _pending_attacker_head(state, eng, viewer)
        if head_id is None:
            raise IllegalSlotError(
                "ATTACKER slot but not in the attacker decision phase"
            )
        return AttackerDecision(
            attacker_id=head_id,
            attack=(slot_idx == SLOT_ATTACKER_ATTACK),
        )

    if _slot_in(slot_idx, SLOT_BLOCKER_BASE, SLOT_BLOCKER_END):
        head_id = _pending_blocker_head(state, eng, viewer)
        if head_id is None:
            raise IllegalSlotError(
                "BLOCKER slot but not in the blocker decision phase"
            )
        if slot_idx == SLOT_BLOCKER_SKIP:
            return BlockerDecision(blocker_id=head_id, attacker_id=None)
        atk_pos = slot_idx - SLOT_BLOCKER_BASE
        attackers = _attacking_creatures(eng)
        if atk_pos >= len(attackers):
            raise IllegalSlotError(
                f"BLOCKER atk_pos={atk_pos} but {len(attackers)} attackers"
            )
        return BlockerDecision(
            blocker_id=head_id, attacker_id=attackers[atk_pos].id,
        )

    raise IllegalSlotError(
        f"slot_idx={slot_idx} fell through every range — vocab math is "
        "off; rerun test_vocab_size_constant_is_fixed"
    )


def _cast_from_variant_hand(card, variant: int) -> CastSpell:
    """Build the right ``CastSpell`` flavor for hand-cast variant N."""
    if variant == _VAR_NORMAL:
        return CastSpell(card_id=card.id)
    if variant == _VAR_KICKED:
        return CastSpell(card_id=card.id, kicked=True)
    if variant == _VAR_WARP:
        return CastSpell(card_id=card.id, use_warp=True)
    if variant == _VAR_FREE:
        return CastSpell(card_id=card.id, cast_for_free=True)
    if variant == _VAR_IMPENDING:
        return CastSpell(card_id=card.id, use_impending=True)
    if variant in (_VAR_ROOM_HALF_0, _VAR_ROOM_HALF_1):
        halves = sorted(h.name for h in card.printing.room_halves)
        idx = 0 if variant == _VAR_ROOM_HALF_0 else 1
        if idx >= len(halves):
            raise IllegalSlotError(
                f"CAST_FROM_HAND room_half variant {idx} but printing has "
                f"{len(halves)} halves"
            )
        return CastSpell(card_id=card.id, cast_room_half=halves[idx])
    raise IllegalSlotError(f"unknown CAST_FROM_HAND variant {variant}")


def _pending_attacker_head(
    state: Any, eng: EngineGameState, viewer: PlayerID,
) -> Optional[int]:
    """Return the id of the next-pending attacker the adapter is
    asking about, or ``None`` if not in the attacker phase. Falls
    back to the engine's ``eligible_attackers`` (smallest id) when
    the adapter has no pending list yet — that's the same convention
    the adapter uses on entry to the phase."""
    pending = getattr(state, "_pending_attackers", None)
    if pending:
        return pending[0]
    if _combat_phase(state) == "attackers":
        # Adapter has consumed all pending — defensive None.
        return None
    # No adapter context. If it's the declare-attackers step, the
    # head would be the smallest-id eligible attacker; otherwise
    # there's no valid head.
    from mtg_bot.turn import Step
    if eng.step == Step.DECLARE_ATTACKERS:
        elig = eligible_attackers(eng, viewer)
        if elig:
            return min(o.id for o in elig)
    return None


def _pending_blocker_head(
    state: Any, eng: EngineGameState, viewer: PlayerID,
) -> Optional[int]:
    """Mirror of ``_pending_attacker_head`` for blockers."""
    pending = getattr(state, "_pending_blockers", None)
    if pending:
        return pending[0]
    if _combat_phase(state) == "blockers":
        return None
    from mtg_bot.turn import Step
    if eng.step == Step.DECLARE_BLOCKERS:
        elig = eligible_blockers(eng, viewer)
        if elig:
            return min(o.id for o in elig)
    return None


# ---------------------------------------------------------------------
# Legal-action mask
# ---------------------------------------------------------------------


def legal_action_mask(
    state: Any, viewer: PlayerID,
    version: str = ACTION_ENCODING_VERSION_V1,
) -> np.ndarray:
    """Return a bool array of length ``ACTION_VOCAB_SIZE`` where True
    marks slots that decode to a legal decision in the current state.

    ``version`` selects the slot-mapping convention; defaults to
    ``ACTION_ENCODING_VERSION_V1`` for backward compat. The mask is
    bit-equivalent under v1 vs v2 only up to a permutation of the
    positional slot families (hand / graveyard / exile) — the SET of
    legal slots may differ between versions for the same state, but
    each version's mask exactly matches that version's
    ``encode``/``decode`` pair.

    Implementation: enumerate the adapter's ``legal_decisions`` (or
    when given an engine state, enumerate ``legal_actions`` directly)
    and encode each, setting the corresponding bit. Anything that
    can't be encoded (because of cap overflow) is silently skipped —
    those decisions are unreachable through the policy head, which is
    the only contract the mask supports.
    """
    _check_version(version)
    mask = np.zeros(action_vocab_size(version), dtype=bool)
    if _is_resolution_paused(state):
        # Mid-resolution pause: only the decider has decisions, and
        # only ``canonical_v3`` has slots for them. Each option's slot
        # was fixed at pause time; options without one (past a cap /
        # outside the vocabulary) were counted then by
        # ``resolution_slots`` and stay prior-0 search children.
        if (
            has_resolution_slots(version)
            and viewer == getattr(state, "_resolution_player", None)
        ):
            for off in state._resolution_slot_offsets:
                if off is not None:
                    mask[SLOT_RESOLUTION_BASE + off] = True
        return mask
    legals = _enumerate_legal(state, viewer)
    for d in legals:
        try:
            slot = encode(d, state, viewer, version=version)
        except EncodingOverflowError:
            # The position exceeded a cap. Skip — this slot is
            # unreachable by the policy. Increase the cap and rerun
            # the vocab-size test if you need to cover this case.
            continue
        mask[slot] = True
    return mask


def _resolution_option(decision: Any) -> Any:
    """The engine-side option value a ``Resolution*Decision`` answers
    with (the element of ``adapter.resolution_options`` it maps to)."""
    if isinstance(decision, ResolutionTargetDecision):
        return decision.target
    if isinstance(decision, (
        ResolutionDiscardDecision, ResolutionCardFromLibraryDecision,
        ResolutionCardFromOpponentHandDecision,
    )):
        return decision.card_id
    if isinstance(decision, ResolutionModeDecision):
        return decision.mode_index
    if isinstance(decision, ResolutionModesDecision):
        return tuple(decision.mode_indices)
    if isinstance(decision, ResolutionSacrificeDecision):
        return decision.object_id
    if isinstance(decision, ResolutionXDecision):
        return decision.x_value
    if isinstance(decision, ResolutionSurveilDecision):
        return decision.mill
    if isinstance(decision, ResolutionShockLifeDecision):
        return decision.pay
    if isinstance(decision, ResolutionYesNoDecision):
        return decision.yes
    if isinstance(decision, ResolutionDevourLandsDecision):
        return tuple(decision.land_ids)
    raise EncodingOverflowError(  # pragma: no cover - guarded by caller
        f"not a resolution decision: {decision!r}"
    )


def _encode_resolution(
    decision: Any, state: Any, viewer: PlayerID, version: str,
) -> int:
    """Slot of a ``Resolution*Decision`` at the adapter's current
    pause. Raises ``EncodingOverflowError`` when the version has no
    resolution block, the state is not paused for ``viewer``, the
    decision is not one of the pause's options, or the option has no
    slot (counted at pause time)."""
    if not has_resolution_slots(version):
        raise EncodingOverflowError(
            f"{type(decision).__name__} has no slot under {version} "
            f"(resolution decisions need {ACTION_ENCODING_VERSION_V3})"
        )
    if not _is_resolution_paused(state):
        raise EncodingOverflowError(
            f"{type(decision).__name__} encoded outside a resolution "
            f"pause (pass the paused adapter, not its engine state)"
        )
    if viewer != getattr(state, "_resolution_player", None):
        raise EncodingOverflowError(
            f"P{viewer} is not the decider of the pending "
            f"{state._resolution_kind} pause"
        )
    option = _resolution_option(decision)
    for opt, off in zip(
        state._resolution_options, state._resolution_slot_offsets,
    ):
        if type(opt) is type(option) and opt == option:
            if off is None:
                raise EncodingOverflowError(
                    f"{decision!r} has no slot in the resolution block "
                    f"(past a cap or outside the vocabulary; counted in "
                    f"resolution_slots.overflow_counts())"
                )
            return SLOT_RESOLUTION_BASE + off
    raise EncodingOverflowError(
        f"{decision!r} is not an option of the pending "
        f"{state._resolution_kind} pause"
    )


def _decode_resolution(slot_idx: int, state: Any, viewer: PlayerID) -> Any:
    """Inverse of ``_encode_resolution``: the first option of the
    pending pause whose slot is ``slot_idx``."""
    if not _is_resolution_paused(state):
        raise IllegalSlotError(
            f"resolution slot {slot_idx} but the state is not paused"
        )
    if viewer != getattr(state, "_resolution_player", None):
        raise IllegalSlotError(
            f"resolution slot {slot_idx} but P{viewer} is not the decider"
        )
    off = slot_idx - SLOT_RESOLUTION_BASE
    decisions = state._resolution_legal_decisions()
    for d, o in zip(decisions, state._resolution_slot_offsets):
        if o == off:
            return d
    raise IllegalSlotError(
        f"resolution slot {slot_idx} is not an option of the pending "
        f"{state._resolution_kind} pause"
    )


def _enumerate_legal(state: Any, viewer: PlayerID) -> list:
    """Return the legal decisions at the current decision moment.

    Adapters expose three flavors of decision moment: a normal
    priority window (``legal_actions``), the per-attacker phase
    (``AttackerDecision`` × 2), and the per-blocker phase
    (``BlockerDecision`` × N+1). When given a raw engine state, only
    the normal priority window is supported.

    When ``state`` is an adapter, the priority-window branch routes
    through the adapter's per-decision-moment ``legal_actions`` cache
    (``_legal_actions_cached``) so a search loop that calls both
    ``legal_decisions`` AND ``legal_action_mask`` for the same priority
    moment only enumerates once. The cache is invalidated by
    ``adapter.apply``, so freshness is automatic. Engine-state inputs
    (no adapter context) fall through to ``legal_actions`` directly —
    same behavior as before, no caching available.
    """
    # B3-S2: the Rust-backed provider already maps its Rust
    # legal_decisions dicts to search Decisions through the
    # non-vestigial rust_decision_adapter. Source the legal set
    # straight from it (the codec then encodes each Decision against
    # the RustZoneView via `encode`, unchanged). The provider's own
    # combat-window emission is the B3-S3 factored-combat scope; for
    # the S2 priority-window axis legal_decisions() is exactly the
    # set to encode. Detected by the provider's marker attribute so
    # no isinstance import cycle is introduced.
    if getattr(state, "_is_rust_decision_provider", False):
        return list(state.legal_decisions())

    if _is_resolution_paused(state):
        # Only the decider acts at a pause; its options are the
        # adapter's resolution decisions (never the snapshot's
        # priority actions).
        if viewer != getattr(state, "_resolution_player", None):
            return []
        return list(state.legal_decisions())

    eng = _engine_state(state)
    phase = _combat_phase(state)
    if phase == "attackers":
        head = _pending_attacker_head(state, eng, viewer)
        if head is None:
            return []
        return [
            AttackerDecision(attacker_id=head, attack=True),
            AttackerDecision(attacker_id=head, attack=False),
        ]
    if phase == "blockers":
        head = _pending_blocker_head(state, eng, viewer)
        if head is None:
            return []
        out: list = []
        for atk in _attacking_creatures(eng):
            out.append(
                BlockerDecision(blocker_id=head, attacker_id=atk.id)
            )
        out.append(BlockerDecision(blocker_id=head, attacker_id=None))
        return out
    # Normal priority window. Prefer the adapter's cached enumerator
    # when available so concurrent ``legal_decisions`` /
    # ``legal_action_mask`` calls in the ISMCTS inner loop share one
    # underlying ``legal.legal_actions`` invocation per priority
    # moment. Falls back to a direct call when ``state`` is a raw
    # engine state.
    # First: honour the adapter's opp-capability auto-pass — when
    # the non-active player has priority and the capability check
    # says "no instant-speed action available", the adapter returns
    # ``[PassPriority()]`` from ``legal_decisions`` without calling
    # ``legal_actions``. The mask must agree, otherwise the mask
    # would expose phantom "legal" slots the engine would refuse.
    skip_empty_combat_fn = getattr(state, "_should_skip_empty_combat", None)
    if skip_empty_combat_fn is not None and skip_empty_combat_fn():
        return [PassPriority()]
    skip_opp_pcm_fn = getattr(
        state, "_should_skip_opp_postcombat_main", None,
    )
    if skip_opp_pcm_fn is not None and skip_opp_pcm_fn():
        return [PassPriority()]
    skip_dead_main1_fn = getattr(state, "_should_skip_dead_main1", None)
    if skip_dead_main1_fn is not None and skip_dead_main1_fn():
        return [PassPriority()]
    short_circuit_fn = getattr(state, "_can_short_circuit_to_pass", None)
    if short_circuit_fn is not None and short_circuit_fn():
        return [PassPriority()]
    cached_fn = getattr(state, "_legal_actions_cached", None)
    if cached_fn is not None:
        decisions = list(cached_fn(viewer))
    else:
        decisions = list(legal_actions(eng, viewer))
    # Honour the adapter's forced-landdrop filter: at second main
    # with unspent land drop and ≥1 land in hand and <6 lands in
    # play, ``legal_decisions`` strips ``PassPriority`` from the
    # returned list. The mask must agree, otherwise the mask would
    # expose a phantom "pass" slot the policy could pick that the
    # engine adapter would never actually let it apply. See
    # ``MtgGameStateAdapter._should_force_landdrop``.
    force_landdrop_fn = getattr(state, "_should_force_landdrop", None)
    if force_landdrop_fn is not None and force_landdrop_fn():
        decisions = [
            d for d in decisions if not isinstance(d, PassPriority)
        ]
    return decisions
