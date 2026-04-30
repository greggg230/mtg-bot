"""Game state: zones, game objects, players, stack, and GameState."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from mtg_bot.cards import CardPrinting

if TYPE_CHECKING:
    from mtg_bot.agents import Agent
    from mtg_bot.effects import ActivatedAbility, TriggeredAbility
    from mtg_bot.turn import Step


class Zone(Enum):
    LIBRARY = "library"
    HAND = "hand"
    BATTLEFIELD = "battlefield"
    GRAVEYARD = "graveyard"
    EXILE = "exile"
    STACK = "stack"


@dataclass
class GameObject:
    """An instance of a card in the game. Uniquely identified by ``id``."""

    id: int
    printing: CardPrinting
    owner_id: int
    controller_id: int
    zone: Zone
    tapped: bool = False
    damage: int = 0
    summoning_sick: bool = True
    attacking: bool = False
    blocking: Optional[int] = None
    blocked_by: list[int] = field(default_factory=list)
    # Marked lethal by a deathtouch source this turn — destroyed by SBAs
    # regardless of damage vs toughness. Cleared when damage is cleared.
    marked_lethal: bool = False
    # "Was this creature dealt damage this turn?" — set True at every site
    # that applies damage (combat damage, _apply_damage, bespoke pingers).
    # Cleared at cleanup. Used by effects with riders gated on
    # "if that creature wasn't dealt damage this turn" (Faller's Faithful).
    damaged_this_turn: bool = False
    # +1/+1 counters raise effective power and toughness by equal amounts.
    # Persist across turns (never wear off on their own).
    plus_one_counters: int = 0
    # -1/-1 counters lower effective power and toughness by equal
    # amounts. Persist across turns. CR 704.5n: at the next SBA
    # check, +1/+1 and -1/-1 counters cancel pairwise — N pairs
    # of each remove min(+1/+1, -1/-1) from each side. The
    # cancellation runs in ``state_based_actions``; the readers
    # in ``_compute_effective_pt`` use the post-cancellation
    # values via this field.
    minus_one_counters: int = 0
    # "Can't be blocked this turn" marker (cleared at cleanup).
    temp_unblockable: bool = False
    # "Until end of turn, this land becomes a creature." Manlands
    # (Celestial Colonnade, Blinkmoth Nexus, etc.). When True, the
    # object counts as a creature for combat/targeting purposes.
    # Cleared at cleanup. Pair with temp_base_pt_override + temp_keywords.
    temp_is_creature: bool = False
    # Subtype granted UEOT — typically by an animation effect like
    # Blinkmoth Nexus's "becomes a 1/1 Blinkmoth artifact creature".
    # Cleared at cleanup. Consumed by target-subtype filters that
    # need to recognize creatures whose subtype is granted at runtime
    # (e.g. PumpTargetCreatureUntilEOT.target_subtype="Blinkmoth").
    temp_subtype: str | None = None
    # "You may play this card this turn" — set on cards exiled by
    # impulse-style effects (Territorial Bruntar landfall, Memorial
    # Vault activated). Cleared at cleanup. legal_actions honors it
    # for cast-from-exile at normal cost.
    playable_from_exile_this_turn: bool = False
    # Persistent cast-from-exile permission gated on the controller
    # controlling a Kavu. Set on cards exiled by Possibility
    # Technician's ETB / Kavu observer trigger ("for as long as that
    # card remains exiled, you may play it if you control a Kavu").
    # NOT cleared at cleanup — persists until the card leaves exile.
    # legal_actions honors it for cast-from-exile at normal cost when
    # the controller currently controls at least one Kavu.
    play_from_exile_requires_kavu: bool = False
    # Base P/T override ("has base power and toughness 1/1 until end of
    # turn") — Atomic Microsizer. None = use printed P/T. Cleared at
    # cleanup.
    temp_base_pt_override: "tuple[int, int] | None" = None
    # Warp-cast state. ``cast_via_warp`` is set True when this card was cast
    # for its warp cost (from hand or, for Timeline Culler, from graveyard).
    # The end-step exile is scheduled as a proper delayed triggered ability
    # on ``state.delayed_triggers`` at cast time (see ``_cast_spell``); this
    # flag is read by effects that key off "if cast via warp" — Astelli
    # Reclaimer's X computation and the ``granted_keywords_if_warp_cast``
    # rider on pump effects (Full Bore). Cleared by ``WarpExileSelf`` when
    # the delayed trigger fires.
    # ``warped`` is set True when this card sits in exile having been exiled
    # via that trigger — it can be cast from exile at its normal cost.
    cast_via_warp: bool = False
    warped: bool = False
    # EoE Spacecraft charge counters. A Spacecraft at ``station_threshold`` or
    # more is a creature (see ``is_creature_now``).
    charge_counters: int = 0
    # Duskmourn Impending: time counters on the permanent and a flag
    # indicating the printed creature type is suppressed. When a card
    # is cast for its impending cost, ``time_counters`` is initialized
    # to ``printing.impending_count`` and ``impending_active`` is set
    # True at ETB. While ``impending_active`` is True ``is_creature_now``
    # returns False even on a printed-creature card type — the
    # permanent is the enchantment half only (no combat, no creature-
    # type targeting). The end-step turn-based action
    # ``run_impending_decrement_turn_based_action`` removes one time
    # counter from each impending-active permanent the active player
    # controls; when the count hits 0 it clears ``impending_active``
    # and the permanent becomes a creature.
    time_counters: int = 0
    impending_active: bool = False
    # Set True at cast time when ``use_impending=True`` so
    # ``put_into_battlefield`` can initialize time counters and
    # ``impending_active`` as the permanent transitions onto the
    # battlefield. Cleared at ETB after the counter init runs (so
    # a blink-and-recast under the printed cost doesn't reactivate
    # the suppression). Mirrors the ``cast_via_warp`` precedent.
    cast_via_impending: bool = False
    # Set True at cast time when this card was cast from its
    # controller's hand (the normal cast path), distinct from casts
    # originating from exile (warped, impulse, Possibility-Technician),
    # the top of the library (Mm'menon grant), or the graveyard
    # (Timeline Culler warp-from-graveyard). Cleared at ETB after the
    # cast-from-hand-only ETB hooks read it (so a blink-and-recast
    # path doesn't carry stale provenance). Consumed by the
    # ``enters_with_minus_one_counters_from_hand_cast`` ETB
    # replacement (Patched Plaything). Default False — only the
    # ``actions._cast_spell`` cast-from-hand branch flips it on.
    cast_via_hand: bool = False
    # Stun counters: a permanent with stun counters doesn't untap during
    # its controller's untap step — instead one stun counter is removed.
    stun_counters: int = 0
    # Until-end-of-turn bonuses applied by pump effects (e.g. "+3/+0 until
    # end of turn"). Cleared on the cleanup step. Stacked additively when
    # multiple pumps hit the same object.
    temp_power_bonus: int = 0
    temp_toughness_bonus: int = 0
    temp_keywords: frozenset[str] = field(default_factory=frozenset)
    # Extended-temporal-scope bonuses: "until your next turn" — expire
    # at the START (untap step) of player ``extended_temp_expires_at_player_id``,
    # NOT at this turn's cleanup. Stacked additively, contribute to
    # effective_power / effective_toughness / effective_keywords just
    # like the regular temp_* fields. Matches Mouth of the Storm's
    # "creatures your opponents control get -3/-0 until your next turn."
    # Set ``extended_temp_expires_at_player_id`` to the casting/source
    # controller's id; their next untap clears these.
    extended_temp_power_bonus: int = 0
    extended_temp_toughness_bonus: int = 0
    extended_temp_keywords: frozenset[str] = field(default_factory=frozenset)
    extended_temp_expires_at_player_id: Optional[int] = None
    # UEOT triggered abilities granted by other effects (e.g. Requiem
    # Monolith's "Whenever this creature is dealt damage, you draw that
    # many cards and lose that much life", Raging Ravine's "Whenever
    # this creature attacks, put a +1/+1 counter on it"). Cleared at
    # cleanup. Fire helpers iterate this alongside
    # ``printing.triggered_abilities``; the once-per-turn ledger uses
    # synthetic indices (``("temp", i)`` or a 1000+ offset depending on
    # call site) to avoid colliding with printed-ability indices.
    temp_triggered_abilities: "tuple[TriggeredAbility, ...]" = ()
    # Granted activated abilities that exist only "while animated" — a
    # manland's UEOT activation can fold per-card abilities here (e.g.
    # Lavaclaw Reaches' "{X}: This creature gets +X/+0 UEOT", Wandering
    # Fumarole's "{0}: Switch P/T UEOT"). Cleared at cleanup. Read by
    # ``effective_activated_abilities`` and surfaced to legal_actions /
    # _activate_ability via the same index space (printed first, then
    # static-grant, then equipment-grant, then temp_*).
    temp_activated_abilities: "tuple[ActivatedAbility, ...]" = ()
    # Equipment attachment: if this object is Equipment, ``attached_to``
    # is the id of the creature it's attached to (or None if unattached).
    # Plays no role if the object isn't Equipment.
    attached_to: Optional[int] = None
    # Oblivion-Ring-style "until X leaves the battlefield" tracking: when
    # a card is exiled by a specific source, its ``exiled_by_source`` is
    # set to that source's id. When the source leaves the battlefield,
    # move_off_battlefield returns all cards tagged with its id from
    # exile back to their owner's battlefield.
    exiled_by_source: Optional[int] = None
    # "Gain control of ~ until end of turn" tracking: when this object
    # is controlled UEOT, ``temp_original_controller_id`` stores the
    # controller it came from. run_cleanup_step swaps ``controller_id``
    # back and clears this field. None = no temporary-control effect.
    temp_original_controller_id: Optional[int] = None
    # Baseline controller captured the first time a layer-2
    # control-change continuous effect targets this permanent
    # (Tractor Beam attaching). When the registry has no
    # ``"control_change"`` entries for this object,
    # ``effective_controller`` falls back to this value. Cleared by
    # ``move_off_battlefield``'s revert path when the last
    # control-grant entry is removed. Distinct from the UEOT
    # ``temp_original_controller_id`` mechanism — this baseline is
    # specifically the "controller before any aura-grant continuous
    # effect" and outlives end-of-turn cleanup.
    controller_id_before_any_aura_swap: Optional[int] = None
    # Planeswalker loyalty counters. Initialized to printing.loyalty
    # when a planeswalker enters the battlefield. State-based actions
    # send planeswalkers with 0 loyalty to the graveyard.
    loyalty_counters: int = 0
    # Saga lore counters. Sagas would get a lore counter at the
    # start of each precombat main phase as a turn-based action;
    # the state-based action (CR 704.5r in audit numbering)
    # sacrifices a saga whose ``lore_counters >=
    # printing.chapter_count``. The chapter-triggered abilities
    # that fire when the lore counter advances aren't modeled in
    # scope.
    lore_counters: int = 0
    # Snapshot of this object's original CardPrinting, taken the first
    # time a "becomes a copy of target X" effect mutates ``printing``.
    # Restored on cleanup? No — "becomes a copy" is a continuous effect
    # with no end condition, so the mutation persists. The snapshot is
    # kept so a second copy activation can re-derive "retained own
    # abilities" (Thespian's Stage: "except it has this ability").
    original_printing: Optional["CardPrinting"] = None
    # Regenerate shield: "The next time this creature would be
    # destroyed this turn, instead remove all damage from it, tap it,
    # and remove it from combat." Set by RegenerateTargetCreature;
    # consumed by state_based_actions when destruction is averted;
    # cleared at cleanup if unconsumed (regeneration is UEOT).
    regeneration_shield: bool = False
    # Phasing — "treat it and anything attached to it as though they
    # don't exist until its controller's next turn." While True, this
    # object is skipped in combat damage assignment. Set by
    # PhaseOutEquippedCreature (Robe of Stars) and cleared at the
    # start of its controller's untap step. Simplification: only
    # combat-damage resolution filters phased objects — triggers,
    # anthems, and legal_actions targeting don't (yet) hide them.
    phased_out: bool = False
    # Reveal-tracking for hidden-information determinization. Each
    # player id in this set has been shown this card's identity in a
    # way that gives them perfect knowledge of where it currently is —
    # e.g. it's their own card, it's in a public zone, or they were
    # shown it as part of resolution (tutor reveal, look-at-hand,
    # cast-from-exile-revealed). When the card moves into a face-down
    # zone whose ordering becomes uncertain (library shuffle, hand
    # mixed back into library), this is reset to ``frozenset()`` to
    # avoid the engine making misinformed claims about what other
    # players still know. See ``docs/HIDDEN_INFO_MODEL.md`` for the
    # design rationale and the under-informed edge cases this
    # representation cannot capture (multi-zone positional uncertainty,
    # cardinality constraints).
    publicly_known_to: frozenset[int] = field(default_factory=frozenset)
    # Timestamp (= ``state.next_timestamp`` snapshot) at the moment
    # this object entered the battlefield. Used to assign a stable
    # "static-effect timestamp" to layered effects that are
    # generated by a permanent's static ability rather than written
    # to the registry directly. The ``effective_subtypes`` walker
    # uses this for Magus-of-the-Moon and Dryad-of-the-Ilysian-Grove
    # ordering: each source's static contributes synthetic layer-4
    # entries with ``timestamp == source.etb_timestamp`` so the
    # within-layer timestamp order matches CR 613.4. 0 means "not
    # yet entered" — orders before any registry entry.
    etb_timestamp: int = 0
    # Painter's Servant: the color chosen as the printing's
    # ``paints_all_objects_chosen_color_at_etb`` ETB resolves.
    # Stored on the GameObject because the choice is per-instance.
    # Consumed by ``effective_colors`` when synthesizing layer-5
    # ``"add_colors"`` entries from this Painter source.
    painter_chosen_color: Optional[str] = None
    # Face-down marker. CR 701.34 / 707.2: a face-down permanent has
    # no name, no subtypes, and no abilities; it's a 2/2 creature.
    # We keep ``printing`` intact (so ``TurnFaceUp`` can read the
    # underlying card's mana cost and properties), but the engine's
    # effective_* helpers (``effective_power`` / ``effective_toughness``
    # / ``effective_keywords`` / ``effective_subtypes`` /
    # ``is_creature_now``) short-circuit to the face-down 2/2 vanilla
    # view when this flag is True. Set by the ``ManifestDread`` effect
    # at the moment a card becomes a face-down permanent; cleared by
    # the ``TurnFaceUp`` action.
    face_down: bool = False
    # DSK Rooms: set of half-names currently unlocked on this Room
    # permanent. When a Room is cast for half A's mana cost, the
    # entering permanent has ``unlocked_halves = frozenset({A})``;
    # ``UnlockRoomHalf`` adds the named half to the set. Locked
    # halves' abilities are filtered out by the effective-ability
    # iterators (``effective_triggered_abilities`` /
    # ``effective_activated_abilities`` / static iteration). When
    # ``len(unlocked_halves) == len(printing.room_halves)``, the Room
    # is fully unlocked — the cast/unlock sites log "Room fully
    # unlocked" so a future Eerie hook can observe it (Eerie itself
    # isn't wired yet). Empty frozenset for non-Room objects.
    unlocked_halves: frozenset[str] = field(default_factory=frozenset)
    # DSK Enduring cycle (Enduring Innocence / Curiosity / Tenacity /
    # Courage / Vitality): when an Enduring creature dies, its
    # ``When this creature dies`` trigger returns the card to the
    # battlefield "as an enchantment" — its printed Creature card type
    # is suppressed for the return. The printing still carries
    # ``"Creature"`` in ``card_types``; this per-instance flag is the
    # in-game suppression marker. Set by the resolving Enduring
    # effect (``EnduringDiesReturnAsEnchantmentEffect``) at the moment
    # the card transits graveyard → battlefield; consumed by
    # ``is_creature_now``, which short-circuits to ``False`` when
    # the flag is True so the returned permanent counts as an
    # enchantment only. Mirrors the ``impending_active`` precedent —
    # same shape (printing keeps the type, GameObject suppresses it
    # at runtime), different mechanic.
    enduring_returned_as_enchantment: bool = False
    # DSK Ghost Vacuum / "flying counter" custom counter type. CR 122
    # recognizes named counters whose name and printed wording grant the
    # named keyword to the bearer ("a flying counter"); engine-side we
    # carry the count as its own integer field on ``GameObject`` parallel
    # to ``plus_one_counters`` / ``stun_counters`` etc. ``effective_keywords``
    # adds ``"flying"`` to the bearer's keyword set whenever this is
    # ``> 0`` so combat / blocker rules pick the granted keyword up
    # automatically — same write-through-cache shape the +1/+1 counter
    # pump path uses for layer-7d P/T contributions.
    flying_counters: int = 0
    # DSK Haunted Screen "Activate only once" tracking. The printed
    # rider on Haunted Screen's ``{7}: …`` activation reads "Activate
    # only once." which the engine simulates by latching this flag to
    # ``True`` the first time the bespoke ``HauntedScreenActivatedEffect``
    # resolves; subsequent activations of the same ability silently
    # no-op at resolve time. The flag is per-``GameObject`` so a
    # blink-and-recast (a fresh ``GameObject`` instance) gets a fresh
    # one-shot — matches CR 706.2 ("a permanent that has changed zones
    # is a new object").
    has_activated_once_only_ability: bool = False
    # Permanent layer-4 ``add_type "creature"`` write-through cache.
    # Set by effects that turn a non-creature permanent into a
    # creature INDEFINITELY (Haunted Screen's ``It becomes a 0/0
    # Spirit creature in addition to its other types.``), as opposed
    # to the UEOT manland-animation flow that uses ``temp_is_creature``
    # and is cleared at cleanup. ``is_creature_now`` consults both
    # caches; cleanup leaves this one alone so the type-change persists
    # across turns. Mirrors a layer-4 ``add_type`` registry entry with
    # ``expires_at=None`` — the cache is the read path, the registry
    # entry holds the layer metadata for future layer-aware ordering
    # work. Distinct from ``temp_is_creature`` because each one's
    # cleanup story is different.
    permanent_is_creature: bool = False


def is_void_active(state: GameState) -> bool:
    """EoE Void condition: a nonland permanent left the battlefield this turn
    OR a spell was warped this turn."""
    return (
        state.permanent_left_battlefield_this_turn
        or state.spell_warped_this_turn
    )


def _active_static_abilities(source: GameObject) -> list:
    """Return ``source.printing.static_abilities`` filtered for DSK
    Room half-locking: a static tagged with a ``room_half`` is inert
    while that half is locked (not in ``source.unlocked_halves``).
    Non-Room statics (``room_half is None``) pass through unchanged.

    Engine call sites that walk static abilities (anthem layer 7c
    pump, layer-6 keyword grants) route through this helper so a
    locked-half anthem (Lecture Hall's "Other permanents you control
    have hexproof") doesn't apply while its half is locked.
    """
    if not source.printing.is_room:
        return list(source.printing.static_abilities)
    return [
        a for a in source.printing.static_abilities
        if getattr(a, "room_half", None) is None
        or a.room_half in source.unlocked_halves
    ]


def reveal_to(obj: GameObject, *player_ids: int) -> None:
    """Mark ``obj`` as publicly known to each of ``player_ids``.

    Idempotent. Use this whenever a card's identity is shown to a
    specific set of players in a way that gives them perfect knowledge
    of where the card currently is — e.g. an opponent looks at your
    hand (reveal_to(card, looker_id)) or you tutor with a reveal step
    and put the card into your hand publicly (reveal_to_all(state, card)).

    See ``docs/HIDDEN_INFO_MODEL.md`` for the design and its caveats.
    """
    if not player_ids:
        return
    obj.publicly_known_to = obj.publicly_known_to | frozenset(player_ids)


def reveal_to_all(state: GameState, obj: GameObject) -> None:
    """Mark ``obj`` as publicly known to every player in ``state``.

    Use at every site where a card is revealed publicly — bounce
    (battlefield → hand), tutor-with-reveal landing in hand, casting
    a spell from a private zone (it briefly transits the public
    stack), etc.
    """
    obj.publicly_known_to = frozenset(p.id for p in state.players)


def reset_public_knowledge(obj: GameObject) -> None:
    """Drop everyone from ``obj.publicly_known_to``.

    Call at "disturbance" sites — library shuffles, hand mixed back
    into library, scry-then-shuffle. Conservatively forgets all
    revealed-knowledge state; the bot becomes under-informed rather
    than misinformed about what other players still know.
    """
    obj.publicly_known_to = frozenset()


def reset_library_public_knowledge(state: GameState, player_id: int) -> None:
    """After shuffling ``player_id``'s library, forget who knew what
    about each card in it. Convenience wrapper around
    ``reset_public_knowledge`` for the common "shuffle the whole
    library" case.
    """
    p = state.player_by_id(player_id)
    for card in p.library:
        reset_public_knowledge(card)


def shuffle_library(state: GameState, player_id: int) -> None:
    """Shuffle ``player_id``'s library and reset reveal-tracking on
    every card in it.

    Always use this helper instead of calling ``state.rng.shuffle``
    directly on a library — it keeps shuffle and knowledge-reset paired
    so the engine can never produce a state where the determinizer
    "remembers" a card whose position was actually scrambled.
    """
    p = state.player_by_id(player_id)
    state.rng.shuffle(p.library)
    for card in p.library:
        reset_public_knowledge(card)


def is_stationed(obj: GameObject) -> bool:
    """True when this Spacecraft has enough charge counters to be a creature."""
    threshold = obj.printing.station_threshold
    return threshold is not None and obj.charge_counters >= threshold


def is_creature_now(obj: GameObject) -> bool:
    """Whether ``obj`` currently counts as a creature on the battlefield.

    True for printed creatures, for Spacecraft whose charge-counter
    threshold has been crossed (they're artifact creatures while
    stationed), and for lands animated by a manland activation
    (``temp_is_creature`` set until cleanup).

    A face-down permanent (``obj.face_down=True``, set by Manifest
    Dread / Morph) is always a 2/2 creature regardless of the
    underlying printing's types — CR 701.34. Face-down identity
    masking takes precedence over Impending suppression below.

    Impending-cast permanents (Duskmourn Overlord cycle) suppress
    the printed creature type while ``impending_active`` is True —
    the printing's ``card_types`` still includes "Creature" but
    this helper returns False until the last time counter is
    removed at the controller's end step (see
    ``run_impending_decrement_turn_based_action``). Impending takes
    precedence over ``temp_is_creature`` / ``is_stationed`` so a
    future card animating an impending Overlord wouldn't wake it
    early — those are independent mechanics.
    """
    if obj.face_down:
        return True
    if obj.impending_active:
        return False
    # DSK Enduring cycle: after the dies-trigger has returned the card
    # to the battlefield "as an enchantment", the printing's Creature
    # card type is suppressed at runtime. The flag is sticky for the
    # remainder of the object's life on the battlefield (CR 400.7 — a
    # new object once it changes zones; clearing happens at the next
    # zone change implicitly via a fresh GameObject instance).
    if obj.enduring_returned_as_enchantment:
        return False
    return (
        obj.printing.is_creature
        or is_stationed(obj)
        or obj.temp_is_creature
        or obj.permanent_is_creature
    )


def _source_layer6_wiped(
    state: "GameState", source: "GameObject"
) -> bool:
    """True if the source has any layer-6 ``"lose_all_abilities"``
    registry entry. Used to gate post-layer-6 reads (layer 7c P/T
    bonuses from a static anthem) — by the time layer 7 runs, layer
    6 has wiped the source's abilities, so its static doesn't
    generate the layer-7c effect anymore.

    Layer-6 effects (keyword grants, ability grants) still apply
    via the within-layer dependency rule even when this returns
    True — the dependency reorders the wipe and the grant within
    layer 6 so the recipients keep the granted abilities.
    """
    return any(
        ce.layer == 6
        and ce.kind == "lose_all_abilities"
        and ce.target_id == source.id
        for ce in state.continuous_effects
    )


def _static_bonuses(
    state: GameState, obj: GameObject
) -> tuple[int, int, frozenset[str]]:
    """Walk the battlefield for static abilities that buff ``obj``.

    Returns (power_bonus, toughness_bonus, granted_keywords). Used by
    the effective_* helpers. Also folds in bonuses from any
    Equipment attached to ``obj``.

    Bifurcated by layer:

    - Granted keywords (Sliver lord's "Slivers have haste/double
      strike", Equipment riders) are layer-6 effects. A Lignified
      / Humility'd source STILL contributes them via the CR 613.7
      within-layer dependency rule — the wipe targets the source,
      not the recipients of its grants.
    - P/T bonuses (Lord of Atlantis's "+1/+1 to other Merfolk",
      Memorial Team Leader's "+1/+0 to other creatures",
      Bonesplitter's equipped-creature +3/+0) are layer-7c effects.
      By the time layer 7c is computed, layer 6 has run and the
      source's static doesn't exist anymore — so a Lignified
      source DOES NOT contribute its P/T bonus.

    The famous case: Lord of Atlantis + Humility. Coralhelm
    Commander becomes 1/1, not 2/2 — the Lord's +1/+1 doesn't
    apply because the Lord's static is wiped before layer 7c.
    """
    power_bonus = 0
    toughness_bonus = 0
    granted: set[str] = set()
    for source in state.battlefield:
        wiped = _source_layer6_wiped(state, source)
        for ability in _active_static_abilities(source):
            if not ability.applies_to(source, obj, state):
                continue
            granted |= ability.granted_keywords
            if not wiped:
                power_bonus += ability.power_bonus
                toughness_bonus += ability.toughness_bonus
    # Equipment attached to obj contributes its equipment_bonuses.
    # The keyword grants and P/T bonuses on the equipment riders
    # follow the same layer split as static abilities above.
    for source in state.battlefield:
        if source.attached_to != obj.id:
            continue
        eb = source.printing.equipment_bonuses
        if eb is None:
            continue
        ep, et, ek = eb
        granted |= ek
        if not _source_layer6_wiped(state, source):
            power_bonus += ep
            toughness_bonus += et
    return power_bonus, toughness_bonus, frozenset(granted)


def effective_controller(state: GameState, obj: GameObject) -> int:
    """Return the controller id of ``obj`` after applying CR 613
    layer 2 (control-changing effects).

    Walks ``state.continuous_effects`` for entries with
    ``layer == 2`` and ``kind == "control_change"`` whose
    ``target_id`` matches ``obj.id``; the latest-timestamp entry
    wins (most recently created control-grant). When no entries
    apply, returns ``obj.controller_id_before_any_aura_swap`` if set
    (the captured-baseline controller from the first time a control
    grant was applied), else ``obj.controller_id`` (the printed
    baseline).

    Phase 2 of the layered-effects refactor: this replaces the
    bespoke ``aura_control_grants`` LIFO stack on GameObject. The
    stored ``obj.controller_id`` field is now a write-through cache
    of this function's value — every registry-mutating site
    (``apply_aura_static_grants_on_attach``,
    ``move_off_battlefield``'s revert path) calls
    ``_recompute_obj_controller`` to keep the cache in sync, so
    existing reads of ``obj.controller_id`` continue to return the
    right value.
    """
    # Layer 2 runs before layer 6 in CR 613.1 strict layer order, so
    # a Humility / Lignify wipe of the source (layer 6) does NOT
    # retroactively undo this layer-2 control grant: the layer-2
    # effect was already applied. We do NOT skip on
    # _source_abilities_suppressed here.
    latest: int | None = None
    latest_ts: int = -1
    for ce in state.continuous_effects:
        if (
            ce.layer == 2
            and ce.kind == "control_change"
            and ce.target_id == obj.id
            and ce.timestamp > latest_ts
        ):
            latest_ts = ce.timestamp
            latest = ce.params[0]
    if latest is not None:
        return latest
    if obj.controller_id_before_any_aura_swap is not None:
        return obj.controller_id_before_any_aura_swap
    return obj.controller_id


def _layer6_entries_for(
    state: "GameState", obj: "GameObject",
) -> list["ContinuousEffect"]:
    """Collect every layer-6 continuous-effect entry that targets
    ``obj`` — both real registry entries AND synthesized entries
    from static-anthem sources on the battlefield. Synthesized
    entries carry the source's ``etb_timestamp`` so the within-
    layer-6 timestamp walk in ``effective_keywords`` /
    ``effective_activated_abilities`` /
    ``effective_triggered_abilities`` interleaves them correctly
    with registry-based grants and ``"lose_all_abilities"`` wipes.

    Synthesized sources:
    - Static anthems with ``granted_keywords`` (Sliver lord style).
    - Equipment ``equipment_bonuses`` keyword riders attached to
      ``obj``. The Equipment's ``etb_timestamp`` is the rider's
      timestamp.
    - Humility-style global wipes: when any source on the
      battlefield has the ``wipes_all_creature_abilities`` flag
      and ``obj`` is a creature, synthesize a
      ``"lose_all_abilities"`` entry sourced by that permanent
      with its ``etb_timestamp``.
    """
    entries: list["ContinuousEffect"] = [
        ce for ce in state.continuous_effects
        if ce.layer == 6 and ce.target_id == obj.id
    ]
    # Static-anthem keyword grants from sources on the battlefield.
    for source in state.battlefield:
        for ability in _active_static_abilities(source):
            if not ability.applies_to(source, obj, state):
                continue
            for kw in ability.granted_keywords:
                entries.append(ContinuousEffect(
                    source_id=source.id,
                    layer=6,
                    timestamp=source.etb_timestamp,
                    kind="grant_keyword",
                    target_id=obj.id,
                    params=(kw,),
                    description=(
                        f"{source.printing.name} grants {kw}"
                    ),
                ))
    # Equipment keyword riders.
    for source in state.battlefield:
        if source.attached_to != obj.id:
            continue
        eb = source.printing.equipment_bonuses
        if eb is None:
            continue
        _, _, ek = eb
        for kw in ek:
            entries.append(ContinuousEffect(
                source_id=source.id,
                layer=6,
                timestamp=source.etb_timestamp,
                kind="grant_keyword",
                target_id=obj.id,
                params=(kw,),
                description=(
                    f"{source.printing.name} grants {kw} via equip"
                ),
            ))
    # Humility-style global wipe: applies only to creatures.
    if is_creature_now(obj):
        for source in state.battlefield:
            if source.printing.wipes_all_creature_abilities:
                entries.append(ContinuousEffect(
                    source_id=source.id,
                    layer=6,
                    timestamp=source.etb_timestamp,
                    kind="lose_all_abilities",
                    target_id=obj.id,
                    params=(),
                    description=(
                        f"{source.printing.name}: "
                        f"creatures lose all abilities"
                    ),
                ))
    return entries


def effective_keywords(state: GameState, obj: GameObject) -> frozenset[str]:
    """Keywords in effect right now, computed in CR-613 layer-6
    timestamp order.

    Base set (treated as before any layer-6 entry, since these
    inherent properties don't carry a layer-6 timestamp in our
    model):
    - ``obj.printing.keywords``
    - ``obj.printing.stationed_keywords`` if stationed
    - ``obj.temp_keywords`` and ``obj.extended_temp_keywords``
      (legacy UEOT grants from non-registry effects)

    Layer-6 entries (real + synthesized via
    ``_layer6_entries_for``) walked in timestamp order:
    - ``"lose_all_abilities"`` clears the set entirely.
    - ``"grant_keyword"`` adds ``params[0]``.

    Static-anthem keyword grants from a Sliver-lord-style source
    are synthesized as layer-6 ``grant_keyword`` entries with the
    source's ``etb_timestamp``. So a Lignify-OLDER-than-lord
    correctly grants the keyword (Sliver lord's anthem applies
    after the older Lignify wipe). Similarly Equipment keyword
    riders carry the equipment's etb_timestamp.

    A face-down permanent (CR 701.34 / 707.2) has no abilities,
    so it carries no keywords — short-circuit before walking the
    layer-6 entries.
    """
    if obj.face_down:
        return frozenset()
    kws: set[str] = set(obj.printing.keywords)
    if is_stationed(obj):
        kws |= set(obj.printing.stationed_keywords)
    kws |= set(obj.temp_keywords)
    kws |= set(obj.extended_temp_keywords)
    layer6 = sorted(_layer6_entries_for(state, obj), key=lambda ce: ce.timestamp)
    for ce in layer6:
        if ce.kind == "lose_all_abilities":
            kws.clear()
        elif ce.kind == "grant_keyword" and ce.params:
            kws.add(ce.params[0])
    # ContinuousModifier contributions from StaticAbility declarations.
    # Walked after the registry-based layer-6 pass so the existing
    # grant_keyword / lose_all_abilities ordering is preserved.  The
    # modifier grammar's GrantKeyword / RemoveKeyword payloads are
    # layer-6 operations; for the current EoE pool (Mechan Shieldmate
    # RemoveKeyword("defender"), Illvoi Infiltrator
    # GrantKeyword("unblockable")) no interaction with lose_all_abilities
    # exists, so post-walk application is CR-correct.
    from mtg_bot.effects.modifiers import GrantKeyword, RemoveKeyword, walk_continuous_modifiers
    for _src_id, mod in walk_continuous_modifiers(state, obj):
        if isinstance(mod.payload, GrantKeyword):
            kws.add(mod.payload.keyword)
        elif isinstance(mod.payload, RemoveKeyword):
            kws.discard(mod.payload.keyword)
    # CR 122 named counters that grant keywords — currently just the
    # "flying counter" custom counter type (DSK Ghost Vacuum). The
    # counter is its own integer field on ``GameObject``; a positive
    # count contributes the matching keyword to the bearer regardless
    # of how the counter got there. Folded in AFTER the layer-6 walk so
    # a Lignify wipe on the bearer doesn't cancel the counter's grant
    # — counters are not abilities (CR 122) so layer-6 ``lose_all_
    # abilities`` doesn't touch them. A future named-counter type
    # (haste counter, shield counter, …) lands as a sibling field
    # with the same shape rather than a generalized counter-name map,
    # mirroring the per-counter-type field precedent (``stun_counters``
    # / ``time_counters`` / ``charge_counters``).
    if obj.flying_counters > 0:
        kws.add("flying")
    return frozenset(kws)


def effective_ward_cost(
    state: GameState, obj: GameObject
) -> tuple["ManaCost | None", str]:  # noqa: F821
    """Return ``(ward_cost, ward_cost_str)`` currently in effect on
    ``obj`` — the printed ward stacked with any ward riders granted by
    Auras/Equipment attached to it (Hardlight Containment's "Enchanted
    permanent has ward {1}").

    Multiple ward instances stack additively (CR 702.21d "Multiple
    instances of ward on the same object are each separate triggered
    abilities."). This implementation aggregates them into a single
    additional cost since the simplified ward model auto-pays the whole
    sum at cast time. Returns ``(None, "")`` when nothing has ward.
    """
    from mtg_bot.cards.base import ManaCost  # local: avoid cycle

    parts: list[tuple[ManaCost, str]] = []
    if obj.printing.ward_cost is not None:
        parts.append((obj.printing.ward_cost, obj.printing.ward_cost_str))
    for source in state.battlefield:
        if source.attached_to != obj.id:
            continue
        granted = source.printing.attached_grants_ward_cost
        if granted is None:
            continue
        parts.append((granted, source.printing.attached_grants_ward_str))
    if not parts:
        return None, ""
    if len(parts) == 1:
        return parts[0]
    # Sum components across all ward instances. ManaCost stores generic
    # as an int and per-color counts in a (color, n) tuple — combine both.
    generic = sum(c.generic for c, _ in parts)
    x_pips = sum(c.x_pips for c, _ in parts)
    color_totals: dict[str, int] = {}
    for c, _ in parts:
        for color, n in c.colored:
            color_totals[color] = color_totals.get(color, 0) + n
    colored = tuple(sorted(color_totals.items()))
    total = ManaCost(generic=generic, colored=colored, x_pips=x_pips)
    label = " + ".join(s for _, s in parts if s)
    return total, label


def _dynamic_base_pt(
    state: GameState, obj: GameObject
) -> tuple[int, int] | None:
    """Return (base_power, base_toughness) for dynamic-P/T creatures,
    or None if this creature has no dynamic formula and should use
    its printed P/T.
    """
    formula = obj.printing.dynamic_pt_formula
    if formula is None:
        return None
    if formula == "cosmogoyf":
        # Cards owned in exile (by the creature's controller).
        controller = state.player_by_id(obj.controller_id)
        n = len(controller.exile)
        return n, n + 1
    if formula == "harmonious_grovestrider":
        # P = T = number of lands the controller controls.
        n = sum(
            1 for o in state.battlefield
            if o.controller_id == obj.controller_id and o.printing.is_land
        )
        return n, n
    if formula == "emissary_escort":
        # "Gets +X/+0 where X is the greatest mana value among other
        # artifacts you control." We encode that as a dynamic base
        # derived from printed P/T plus X on power only.
        mvs = [
            o2.printing.mana_cost.total
            for o2 in state.battlefield
            if o2.controller_id == obj.controller_id
            and o2.printing.is_artifact
            and o2.id != obj.id
        ]
        x = max(mvs) if mvs else 0
        printed_p = obj.printing.power or 0
        printed_t = obj.printing.toughness or 0
        return printed_p + x, printed_t
    return None  # pragma: no cover - defensive fallback


def gain_life(state: "GameState", player_id: int, amount: int) -> int:
    """Add ``amount`` life to ``player_id`` and track it in
    life_gained_this_turn. Returns the amount actually gained (same as
    ``amount`` for non-negative values; zero for non-positive).

    Use this helper instead of mutating ``p.life`` directly whenever
    an effect "gains life" — keys like Ragost's "if you gained life
    this turn" rely on the tracker being kept up-to-date.
    """
    if amount <= 0:
        return 0
    p = state.player_by_id(player_id)
    p.life += amount
    p.life_gained_this_turn += amount
    return amount


_BASIC_LAND_TYPE_TO_COLOR: dict[str, str] = {
    "Plains": "W",
    "Island": "U",
    "Swamp": "B",
    "Mountain": "R",
    "Forest": "G",
}


def effective_subtypes(
    state: "GameState", obj: "GameObject"
) -> tuple[str, ...]:
    """Return ``obj``'s subtypes after applying CR-613 layer 3 and
    layer 4 effects in order on top of the printed subtype list.

    Sources walked:

    - Static "artifacts you control are Foods" (Ragost). Adds "Food"
      to artifacts; not registry-driven (predates the refactor).
    - Layer 3 ``"replace_basic_land_type"`` registry entries
      (Magical Hack), in timestamp order — substitutes one basic
      land type for another in the subtype list.
    - Layer 4 entries, walked together in CR 613.4 timestamp order:
        - Real registry ``"add_subtype"`` entries (Lignify
          granting Treefolk).
        - Synthesized ``"set_basic_land_subtype"`` entries from
          permanents with ``nonbasic_lands_are_mountains=True``
          (Magus of the Moon / Blood Moon) — REPLACES all
          basic-land-type subtypes with "Mountain". Synthesized
          per call with ``timestamp == source.etb_timestamp``.
        - Synthesized ``"add_subtype"`` entries from permanents
          with ``grants_all_basic_land_types_to_lands_you_control
          =True`` (Dryad of the Ilysian Grove) — adds all five
          basic land types to lands you control. Synthesized
          per call with ``timestamp == source.etb_timestamp``.

    The Dryad-vs-Magus interaction is the famous case this models:
    older Magus + newer Dryad = Mountain replaced first, then five
    basics added on top, so all 5 colors. Older Dryad + newer
    Magus = five basics added first, then SET to Mountain, so just
    Mountain.

    Face-down permanents have no subtypes (CR 701.34) — short-circuit
    the layered walk.
    """
    if obj.face_down:
        return ()
    subtypes = list(obj.printing.subtypes)
    if obj.printing.is_artifact:
        for o in state.battlefield:
            if o.id == obj.id:
                continue
            if o.controller_id != obj.controller_id:
                continue
            # Ragost's "artifacts you control are Foods" is a layer-4
            # type-changing effect. Layer 4 runs before layer 6 (CR
            # 613.1), so a Lignified Ragost still contributes the
            # Food subtype add — strict layer order, no source-
            # suppression check here.
            if o.printing.grants_food_to_your_artifacts:
                if "Food" not in subtypes:
                    subtypes.append("Food")
                break
    layer3 = sorted(
        (
            ce for ce in state.continuous_effects
            if ce.layer == 3
            and ce.kind == "replace_basic_land_type"
            and ce.target_id == obj.id
        ),
        key=lambda ce: ce.timestamp,
    )
    for ce in layer3:
        if not ce.params or len(ce.params) != 2:
            continue
        old, new = ce.params
        subtypes = [new if s == old else s for s in subtypes]
    layer4: list[ContinuousEffect] = [
        ce for ce in state.continuous_effects
        if ce.layer == 4
        and ce.kind in ("add_subtype", "set_basic_land_subtype")
        and ce.target_id == obj.id
    ]
    # Synthesize Magus-of-the-Moon / Blood Moon entries: nonbasic
    # lands get a layer-4 set_basic_land_subtype("Mountain") effect
    # sourced by each Magus permanent on the battlefield.
    #
    # Strict layer order (CR 613.1): layer 4 runs before layer 6, so
    # a Lignified / Humility'd Magus STILL contributes its layer-4
    # type-changing effect — layer 6 hasn't run yet to wipe its
    # ability. The famous Dryad + Humility ruling: lands keep the
    # basic land types Dryad granted, even after Humility removes
    # Dryad's ability, because layer 4 already applied.
    if obj.printing.is_land and not obj.printing.is_basic_land:
        for src in state.battlefield:
            if src.printing.nonbasic_lands_are_mountains:
                layer4.append(ContinuousEffect(
                    source_id=src.id, layer=4,
                    timestamp=src.etb_timestamp,
                    kind="set_basic_land_subtype",
                    target_id=obj.id,
                    params=("Mountain",),
                    description=(
                        f"{src.printing.name}: nonbasic lands are Mountains"
                    ),
                ))
    # Synthesize Dryad-of-the-Ilysian-Grove entries: any land its
    # controller controls gets layer-4 add_subtype effects for all
    # five basic land types, sourced by Dryad with its etb_timestamp.
    # Same strict-layer-order reasoning — a Lignified Dryad still
    # contributes (layer 4 < layer 6).
    if obj.printing.is_land:
        for src in state.battlefield:
            if (
                src.printing.grants_all_basic_land_types_to_lands_you_control
                and src.controller_id == obj.controller_id
            ):
                for basic in (
                    "Plains", "Island", "Swamp", "Mountain", "Forest"
                ):
                    layer4.append(ContinuousEffect(
                        source_id=src.id, layer=4,
                        timestamp=src.etb_timestamp,
                        kind="add_subtype",
                        target_id=obj.id,
                        params=(basic,),
                        description=(
                            f"{src.printing.name}: lands are {basic}"
                        ),
                    ))
    # Synthesize Urborg-Tomb-of-Yawgmoth / Yavimaya-Cradle-of-Growth
    # entries: any land on the battlefield gets a layer-4 add_subtype
    # entry for the named basic land type, sourced by the
    # ``grants_basic_land_type_to_all_lands`` permanent.
    #
    # CR 613.7 within-layer dependency: if the source is itself
    # subject to a layer-4 ``set_basic_land_subtype`` effect (Blood
    # Moon / Magus targeting it as a nonbasic land), CR 305.6 wipes
    # the source's ability and the contribution is suppressed.
    # This is the famous "Blood Moon beats Urborg" ruling — Blood
    # Moon ALWAYS wins regardless of timestamps because Urborg's
    # ability depends on Blood Moon's effect.
    if obj.printing.is_land:
        for src in state.battlefield:
            granted = src.printing.grants_basic_land_type_to_all_lands
            if granted is None:
                continue
            if _has_basic_land_type_set_effect(state, src):
                continue  # CR 613.7 dependency suppresses
            layer4.append(ContinuousEffect(
                source_id=src.id, layer=4,
                timestamp=src.etb_timestamp,
                kind="add_subtype",
                target_id=obj.id,
                params=(granted,),
                description=(
                    f"{src.printing.name}: each land is a {granted}"
                ),
            ))
    layer4.sort(key=lambda ce: ce.timestamp)
    for ce in layer4:
        if ce.kind == "set_basic_land_subtype":
            new_sub = ce.params[0] if ce.params else None
            if new_sub is None:
                continue
            # Drop existing basic-land-type subtypes and add the new
            # one (preserves non-basic-land subtypes unchanged).
            subtypes = [
                s for s in subtypes
                if s not in _BASIC_LAND_TYPE_TO_COLOR
            ]
            if new_sub not in subtypes:
                subtypes.append(new_sub)
        elif ce.kind == "add_subtype":
            sub = ce.params[0] if ce.params else None
            if sub and sub not in subtypes:
                subtypes.append(sub)
    return tuple(subtypes)


def effective_colors(
    state: "GameState", obj: "GameObject"
) -> frozenset[str]:
    """Return ``obj``'s effective colors as a frozenset of color
    letters (W/U/B/R/G), applying CR 613 layer 5 (color-changing
    effects). Base colors come from ``printing.mana_cost.colors``.

    Layer-5 entries are walked in timestamp order:
      - ``kind="set_colors"`` REPLACES the color set with
        ``params[0]`` (frozenset). Matches Scuttlemutt's "Target
        creature becomes the color or colors of your choice UEOT".
      - ``kind="add_colors"`` UNIONS ``params[0]`` into the current
        color set. Matches Painter's Servant-style "is the chosen
        color in addition to its other colors" clauses.
    """
    colors: set[str] = set(obj.printing.mana_cost.colors)
    # Real registry layer-5 entries on this object.
    layer5: list[ContinuousEffect] = [
        ce for ce in state.continuous_effects
        if ce.layer == 5 and ce.target_id == obj.id
    ]
    # Synthesize Painter's-Servant-style ``add_colors`` entries from
    # any source on the battlefield with the painter flag and a
    # captured chosen color. Source's ``etb_timestamp`` is the
    # entry timestamp so it interleaves with Scuttlemutt-style
    # ``set_colors`` per CR 613 layer-5 timestamp rules.
    for src in state.battlefield:
        if (
            src.printing.paints_all_objects_chosen_color_at_etb
            and src.painter_chosen_color
        ):
            layer5.append(ContinuousEffect(
                source_id=src.id,
                layer=5,
                timestamp=src.etb_timestamp,
                kind="add_colors",
                target_id=obj.id,
                params=(frozenset({src.painter_chosen_color}),),
                description=(
                    f"{src.printing.name} paints "
                    f"{src.painter_chosen_color}"
                ),
            ))
    layer5.sort(key=lambda ce: ce.timestamp)
    for ce in layer5:
        if not ce.params:
            continue
        added = ce.params[0]
        if not isinstance(added, (frozenset, set)):
            continue
        if ce.kind == "set_colors":
            colors = set(added)
        elif ce.kind == "add_colors":
            colors |= set(added)
    return frozenset(colors)


def effective_protection_from_colors(
    state: "GameState", obj: "GameObject"
) -> frozenset[str]:
    """Return the colors ``obj`` has protection from.

    Sources:
    - Printing's ``protection_from_color`` field (single color).
    - Layer-3 ``"replace_color_word"`` registry entries on this
      object (Sleight of Mind, Mind Bend) — substitute one color
      letter for another, in timestamp order so the most-recent
      rewrite wins for chained substitutions.

    Returned as a frozenset for clean set-intersection in
    ``can_block`` and for future extension to multi-color
    protection.
    """
    pc = obj.printing.protection_from_color
    if not pc:
        return frozenset()
    layer3 = sorted(
        (
            ce for ce in state.continuous_effects
            if ce.layer == 3
            and ce.kind == "replace_color_word"
            and ce.target_id == obj.id
        ),
        key=lambda ce: ce.timestamp,
    )
    current = pc
    for ce in layer3:
        if not ce.params or len(ce.params) != 2:
            continue
        old, new = ce.params
        if current == old:
            current = new
    return frozenset({current})


def magus_of_the_moon_active(state: "GameState") -> bool:
    """True when any permanent on the battlefield has the
    ``nonbasic_lands_are_mountains`` flag. Thin wrapper kept for
    backward compatibility with code that wants a quick "is any
    Magus / Blood Moon source on the battlefield" check.

    Strict layer order (CR 613.1) puts layer 4 before layer 6, so
    a Lignified / Humility'd Magus STILL applies its layer-4
    nonbasic-lands-are-Mountains effect — its abilities haven't
    been wiped yet at layer 4 walking time. No source-suppression.
    """
    return any(
        o.printing.nonbasic_lands_are_mountains for o in state.battlefield
    )


def revert_printing_if_no_layer1_copies(
    state: "GameState", obj: "GameObject"
) -> None:
    """Restore ``obj.printing`` from ``obj.original_printing`` when
    no layer-1 ``"is_copy"`` registry entries remain on it.

    Used by:
    - ``run_cleanup_step``: after ``expire_continuous_effects(
      "cleanup")`` drops UEOT layer-1 entries (Cytoshape, Mirage
      Mirror), revert printing on creatures still on the
      battlefield.
    - ``move_off_battlefield``: when a permanent leaves play, its
      layered effects stop applying. The leaving object's printing
      reverts to its printed self before the zone change so a
      future reanimation enters as the printed object (CR
      400.7-style "becomes a new object on zone change").

    No-op if ``original_printing`` is unset (the object was never
    a copy) or if at least one layer-1 ``is_copy`` entry on
    ``obj`` is still active (a longer-lived copy effect like
    Clone's persistent ``expires_at=None`` entry).
    """
    if obj.original_printing is None:
        return
    has_layer1 = any(
        ce.layer == 1
        and ce.kind == "is_copy"
        and ce.target_id == obj.id
        for ce in state.continuous_effects
    )
    if has_layer1:
        return
    obj.printing = obj.original_printing
    obj.original_printing = None


def _has_basic_land_type_set_effect(
    state: "GameState", obj: "GameObject"
) -> bool:
    """True if any layer-4 ``"set_basic_land_subtype"`` effect
    (real registry entry or synthesized from a Magus-of-the-Moon /
    Blood Moon flag source) targets ``obj``. Per CR 305.6, when a
    land's basic land type is set by an effect, it loses abilities
    derived from its rules text and its previous land types — the
    suppression is part of layer-4 type-changing semantics, not a
    separate layer-6 ability wipe.

    Used by ``effective_activated_abilities`` to gate nonbasic
    lands' activated abilities under Magus / Blood Moon. Routes
    through the same registry/synthesis path as
    ``effective_subtypes`` so the two stay consistent.
    """
    # Real registry entries (no card writes these directly today,
    # but the path is open for future cards).
    for ce in state.continuous_effects:
        if (
            ce.layer == 4
            and ce.kind == "set_basic_land_subtype"
            and ce.target_id == obj.id
        ):
            return True
    # Synthesized Magus / Blood Moon entries: per the
    # ``effective_subtypes`` synthesis, these target nonbasic lands.
    if obj.printing.is_land and not obj.printing.is_basic_land:
        for src in state.battlefield:
            if src.printing.nonbasic_lands_are_mountains:
                return True
    return False


def effective_land_produces(
    state: "GameState", land_obj: "GameObject"
) -> frozenset[str]:
    """Return the effective colors a land taps for.

    Computed from ``effective_subtypes`` whenever any layer-3 or
    layer-4 effect is changing the land's basic-land-type subtypes
    — Magical Hack rewrites, Magus of the Moon ("nonbasic lands
    are Mountains"), and Dryad of the Ilysian Grove ("lands are
    every basic land type") all flow through that path. The
    produced color is derived from the resulting basic-land-type
    subtype set via ``_BASIC_LAND_TYPE_TO_COLOR``.

    Falls back to printed ``produces`` for lands whose effective
    subtypes contain no basic land type (e.g., colorless utility
    lands like Strip Mine without any layer-4 effect on them).
    """
    eff_subtypes = effective_subtypes(state, land_obj)
    produced = {
        _BASIC_LAND_TYPE_TO_COLOR[s]
        for s in eff_subtypes
        if s in _BASIC_LAND_TYPE_TO_COLOR
    }
    if produced:
        return frozenset(produced)
    return land_obj.printing.produces


def effective_activated_abilities(
    state: "GameState", obj: "GameObject"
) -> tuple:
    """Return ``obj``'s activated abilities plus any granted.

    Current grants:
    - Ragost: artifacts you control gain {2}, {T}, Sac: gain 3 life.
    - Mm'menon, the Right Hand: artifacts you control gain {T}: Add {U}
      with the "cast_from_outside_hand" spend restriction.
    - The Dominion Bracelet: the equipped creature gains the
      control-opponent-next-turn activated ability.

    Magus of the Moon: nonbasic lands lose their activated abilities
    entirely (their {T}: Add {R} tap is handled via
    ``effective_land_produces``, not as an activated ability).

    Lignify-style "lose all abilities" (layer 6) is processed in
    CR-613 timestamp order alongside ``grant_activated_ability``
    entries: an older lose-all-abilities followed by a newer grant
    re-adds the granted ability; a newer lose-all-abilities wipes
    everything that came before within layer 6.
    """
    from mtg_bot.cards import (
        DOMINION_BRACELET_GRANTED_ABILITY,
        MMNEMON_BLUE_MANA_GRANTED_ABILITY,
        RAGOST_FOOD_GRANTED_ABILITY,
    )

    # CR 305.6: a land whose basic land type has been SET by a
    # layer-4 effect (Magus / Blood Moon set_basic_land_subtype)
    # loses its abilities derived from rules text and old land
    # types. Routes through the same registry/synthesis path as
    # ``effective_subtypes`` so the two stay consistent.
    if (
        obj.printing.is_land
        and _has_basic_land_type_set_effect(state, obj)
    ):
        return ()

    # DSK Rooms: filter out activated abilities whose ``room_half``
    # is not in ``obj.unlocked_halves`` — those halves are locked,
    # so their abilities are inert. Non-Room abilities (room_half ==
    # None) are never filtered.
    if obj.printing.is_room:
        base = [
            a for a in obj.printing.activated_abilities
            if getattr(a, "room_half", None) is None
            or a.room_half in obj.unlocked_halves
        ]
    else:
        base = list(obj.printing.activated_abilities)
    if obj.printing.is_artifact:
        seen_ragost = False
        seen_mmnemon = False
        for o in state.battlefield:
            if o.id == obj.id:
                continue
            if o.controller_id != obj.controller_id:
                continue
            # Static-driven artifact-ability grants from sources like
            # Ragost / Mm'menon are layer-6 effects; per CR 613.7
            # within-layer dependency, they apply even when their
            # source has been Humility'd / Lignified — the wipe
            # targets the source, not the recipients of the grant.
            if (
                not seen_ragost
                and o.printing.grants_food_ability_to_your_artifacts
            ):
                base.append(RAGOST_FOOD_GRANTED_ABILITY)
                seen_ragost = True
            if (
                not seen_mmnemon
                and o.printing.grants_blue_mana_ability_to_your_artifacts
            ):
                base.append(MMNEMON_BLUE_MANA_GRANTED_ABILITY)
                seen_mmnemon = True
            if seen_ragost and seen_mmnemon:
                break
    # Equipment-granted abilities: walk attached equipment. The
    # equipment-keyword grant is layer 6 just like the static
    # anthem grants above — it survives a Lignify on the equipment
    # itself by the same dependency logic.
    for o in state.battlefield:
        if o.attached_to != obj.id:
            continue
        if o.printing.grants_control_opponent_ability_when_equipped:
            base.append(DOMINION_BRACELET_GRANTED_ABILITY)
    # UEOT-granted activated abilities (manland animations).
    if obj.temp_activated_abilities:
        base.extend(obj.temp_activated_abilities)
    # Walk layer-6 entries in timestamp order, applying
    # lose_all_abilities (clears) and grant_activated_ability (adds).
    layer6 = sorted(
        (
            ce for ce in state.continuous_effects
            if ce.layer == 6 and ce.target_id == obj.id
        ),
        key=lambda ce: ce.timestamp,
    )
    for ce in layer6:
        if ce.kind == "lose_all_abilities":
            base.clear()
        elif ce.kind == "grant_activated_ability" and ce.params:
            base.append(ce.params[0])
    return tuple(base)


def effective_triggered_abilities(
    state: "GameState", obj: "GameObject",
) -> tuple:
    """Return the printing's triggered abilities + any UEOT-granted
    triggered abilities, including layer-6 registry entries.

    Phase 2 of the layered-effects refactor: callers that previously
    iterated ``obj.temp_triggered_abilities`` directly should now go
    through this helper so registry-granted triggers (Requiem
    Monolith's UEOT trigger, Raging Ravine's animated attack
    trigger, Perigee Beckoner's granted dies-rider) are picked up.
    Indices match the order the engine fires them: printed first,
    then ``temp_*`` field grants (legacy path), then registry
    grants.

    Lignify-style "lose all abilities" (layer 6) is processed in
    CR-613 timestamp order alongside grant_triggered_ability
    entries: an older lose-all-abilities followed by a newer grant
    re-adds the granted trigger; a newer lose-all-abilities wipes
    everything that came before within layer 6.
    """
    # DSK Rooms: filter out triggered abilities whose ``room_half``
    # is not in ``obj.unlocked_halves`` — those halves are locked,
    # so their abilities are inert. Non-Room abilities (room_half ==
    # None) are never filtered.
    if obj.printing.is_room:
        base = [
            a for a in obj.printing.triggered_abilities
            if getattr(a, "room_half", None) is None
            or a.room_half in obj.unlocked_halves
        ]
    else:
        base = list(obj.printing.triggered_abilities)
    if obj.temp_triggered_abilities:
        base.extend(obj.temp_triggered_abilities)
    layer6 = sorted(
        (
            ce for ce in state.continuous_effects
            if ce.layer == 6 and ce.target_id == obj.id
        ),
        key=lambda ce: ce.timestamp,
    )
    for ce in layer6:
        if ce.kind == "lose_all_abilities":
            base.clear()
        elif ce.kind == "grant_triggered_ability" and ce.params:
            base.append(ce.params[0])
    return tuple(base)


def effective_draw_count(
    state: "GameState", player_id: int, base_count: int
) -> int:
    """Effective number of cards a "draw N" event should draw for
    ``player_id``, accounting for active CR 614 replacement effects.

    Routes through :func:`mtg_bot.effects.modifiers
    .dispatch_replacement` on ``WOULD_DRAW_CARDS``. The legacy
    Quantum-Riddler-shape flag (``replaces_draw_with_extra_when_hand
    _at_most_one``) synthesises an ``AddToCount(1)`` replacement
    gated on the drawing player's hand size ≤ 1; future replacement
    effects (Quantum-Riddler-style or otherwise) layer on by adding
    a ``ReplacementEffect`` to the printing without touching this
    site.
    """
    if base_count < 1:
        return base_count
    from mtg_bot.effects.modifiers import dispatch_replacement
    from mtg_bot.effects.predicates import Event

    p = state.player_by_id(player_id)
    out = dispatch_replacement(
        state, Event.WOULD_DRAW_CARDS,
        affected_player_id=player_id,
        subjects={"player": p},
        payload={"count": base_count},
    )
    if out.get("prevented"):
        return 0
    return out.get("count", base_count)


def etb_trigger_multiplier(state: "GameState", controller_id: int) -> int:
    """Return how many times an ETB-caused triggered ability on a
    permanent controlled by ``controller_id`` should fire.

    Routes through :func:`mtg_bot.effects.modifiers
    .dispatch_replacement` on ``TRIGGERED_ABILITY_TRIGGERS`` with a
    base ``instances=1`` payload. The Starfield-Vocalist-shape
    ``extra_etb_trigger`` flag synthesises an ``AddInstances(N)``
    replacement keyed on the carrier's controller; the post-
    replacement ``instances`` count drives the per-trigger fire
    loop.
    """
    from mtg_bot.effects.modifiers import dispatch_replacement
    from mtg_bot.effects.predicates import Event

    # The "ability_source" subject is what gates the replacement.
    # Our caller doesn't know the source object yet (this helper
    # answers a per-controller question), so we synthesise a
    # placeholder: a sentinel object whose controller_id matches
    # the question. The synthesizer reads ``ability_source
    # .controller_id`` off it. This keeps the Starfield-Vocalist
    # match shape consistent with how a real source-aware caller
    # would pass the ETB-target permanent.
    class _SourceSentinel:
        def __init__(self, controller_id):
            self.controller_id = controller_id
    sentinel = _SourceSentinel(controller_id)
    out = dispatch_replacement(
        state, Event.TRIGGERED_ABILITY_TRIGGERS,
        affected_player_id=controller_id,
        subjects={"ability_source": sentinel, "triggering_obj": sentinel},
        payload={"instances": 1},
    )
    return out.get("instances", 1)


def _counter_doubling_factor(state: GameState, obj: GameObject) -> int:
    """Return the post-replacement factor for +1/+1 counters being
    placed on ``obj``.

    Routes through :func:`mtg_bot.effects.modifiers
    .dispatch_replacement` on ``WOULD_PLACE_COUNTER`` with a base
    ``count=1`` payload; the post-replacement count IS the factor.
    Loading-Zone-shape doublers (``is_counter_doubler``) synthesise
    a ``MultiplyCount(2)`` replacement gated on the recipient's
    controller matching the doubler's controller. Recipients that
    aren't creatures or Spacecraft (Loading Zone's printed scope)
    short-circuit to 1 — the dispatch is skipped entirely so future
    replacements with broader scopes don't accidentally fire.
    """
    if not (obj.printing.is_creature or obj.printing.is_spacecraft):
        return 1
    from mtg_bot.effects.modifiers import dispatch_replacement
    from mtg_bot.effects.predicates import Event

    out = dispatch_replacement(
        state, Event.WOULD_PLACE_COUNTER,
        affected_player_id=obj.controller_id,
        subjects={"obj": obj, "kind": "+1/+1"},
        payload={"count": 1},
    )
    return out.get("count", 1)


def add_plus_one_counters(
    state: GameState, obj: GameObject, amount: int
) -> int:
    """Add ``amount`` +1/+1 counters to ``obj``, applying Loading Zone-
    style doubling replacement effects for each active doubler
    controlled by the recipient. Only applies to creatures and
    Spacecraft (the Planet type isn't modeled yet) — matches the real
    Loading Zone scope.

    Fires "whenever you put one or more +1/+1 counters on a creature
    you control" observer triggers (Terrasymbiosis) for counters added
    to creatures.

    Returns the effective count actually added (after doubling). Safe
    when amount <= 0 (no-op).
    """
    if amount <= 0:
        return 0
    effective = amount * _counter_doubling_factor(state, obj)
    obj.plus_one_counters += effective
    if obj.printing.is_creature and effective > 0:
        from mtg_bot.turn import fire_counters_added_to_your_creature_triggers
        fire_counters_added_to_your_creature_triggers(
            state, obj.controller_id, effective, obj=obj,
        )
    return effective


def add_minus_one_counters(
    state: GameState, obj: GameObject, amount: int
) -> int:
    """Add ``amount`` -1/-1 counters to ``obj``. Counters reduce
    effective P/T by 1 per counter (layer 7d). +1/+1 and -1/-1
    counters cancel pairwise via the CR 704.5n state-based
    action — the cancellation runs in ``state_based_actions``,
    not here.

    No doubling applied (no in-pool doublers target -1/-1
    counters). Safe when amount <= 0.
    """
    if amount <= 0:
        return 0
    obj.minus_one_counters += amount
    return amount


def add_charge_counters(
    state: GameState, obj: GameObject, amount: int
) -> int:
    """Add ``amount`` charge counters to ``obj``. Charge counters are
    Spacecraft-only in the EoE engine, so doubling via Loading Zone
    applies (Spacecraft is in scope). Returns the effective count
    actually added.
    """
    if amount <= 0:
        return 0
    effective = amount * _counter_doubling_factor(state, obj)
    obj.charge_counters += effective
    return effective


def add_stun_counters(
    state: GameState, obj: GameObject, amount: int
) -> int:
    """Add ``amount`` stun counters to ``obj``, applying Loading Zone
    doubling when ``obj`` is in scope (creature/Spacecraft). Returns
    the effective count actually added.
    """
    if amount <= 0:
        return 0
    effective = amount * _counter_doubling_factor(state, obj)
    obj.stun_counters += effective
    return effective


def _registry_set_base_pt(
    state: "GameState", obj: "GameObject"
) -> "tuple[int, int] | None":
    """Highest-timestamp layer-7 sublayer-1 ``"set_base_pt"`` entry
    targeting ``obj``, or None if no such entry exists.

    Walks both real registry entries (Lignify, manland animation,
    Atomic Microsizer) AND synthesized entries from Humility-style
    sources on the battlefield (any permanent with the
    ``sets_all_creature_pt_to_1_1`` flag synthesizes a 1/1 set on
    every creature with the source's ``etb_timestamp``). The most-
    recent timestamp wins (CR 613 within-sublayer rule).
    """
    matches: list[ContinuousEffect] = [
        ce for ce in state.continuous_effects
        if ce.layer == 7
        and ce.sublayer == 1
        and ce.kind == "set_base_pt"
        and ce.target_id == obj.id
    ]
    # Synthesize Humility-style global 1/1 set for creatures.
    if is_creature_now(obj):
        for source in state.battlefield:
            if source.printing.sets_all_creature_pt_to_1_1:
                matches.append(ContinuousEffect(
                    source_id=source.id,
                    layer=7, sublayer=1, kind="set_base_pt",
                    target_id=obj.id,
                    params=(1, 1),
                    description=(
                        f"{source.printing.name}: creatures are 1/1"
                    ),
                    timestamp=source.etb_timestamp,
                ))
    if not matches:
        return None
    matches.sort(key=lambda ce: ce.timestamp)
    return matches[-1].params  # (power, toughness)


def _compute_effective_pt(
    state: GameState, obj: GameObject
) -> tuple[int, int]:
    """Compute (effective_power, effective_toughness) walking layer 7
    sublayers in CR-613 order:

      - 7b (sublayer 1): set base P/T — most-recent registry
        ``"set_base_pt"`` wins; falls back to legacy
        ``temp_base_pt_override``, then dynamic P/T formula, then
        printed.
      - 7c (sublayer 2): modify P/T — anthem static bonuses + UEOT
        pump bonuses (legacy fields).
      - 7d (sublayer 3): counters — +1/+1 counters add equally.
      - 7e (sublayer 4): switch P/T — each layer-7 sublayer-4
        ``"switch_pt"`` registry entry contributes one swap, walked
        in timestamp order (so two swaps cancel; three swaps =
        one swap).

    Single function returning both so the sublayer walk happens
    once per query — saves the duplicate work of the prior
    ``effective_power`` / ``effective_toughness`` split.

    A face-down permanent (CR 701.34) is a 2/2 vanilla creature.
    +1/+1 / -1/-1 counters still modify the printed-2/2 base, so
    we apply layer 7d on top; layers 7b (set base) and 7c (anthem
    pumps from this source's ignored static abilities) and the
    legacy temp pump-bonus fields don't apply because face-down
    objects have no abilities and no name to anchor static effects
    that target by name.
    """
    if obj.face_down:
        # 2/2 base + counters (layer 7d). Switch entries (7e) still
        # affect a face-down permanent — a swap is a creature-state
        # mutation, not an ability. Anthems applying to "creatures
        # you control" still pump face-down creatures (they're
        # creatures), so we still walk static bonuses, but the
        # face-down object's OWN granted_keywords contributions are
        # already filtered out via ``effective_keywords`` /
        # ``_static_bonuses`` reading the printing's static_abilities
        # — those abilities are wiped by face-down (CR 707.2). So
        # we apply only counters here and skip anthems too: anthem
        # contributions from OTHER sources are correctly handled by
        # ``_static_bonuses`` walking the battlefield, but applying
        # them here would mismatch the "no abilities" reading. Keep
        # this simple: face-down = 2/2 plus +1/+1 counters, with
        # external anthems and switches optionally folded.
        base_p, base_t = 2, 2
        # External anthems still buff a face-down creature (it IS a
        # creature, and the anthem applies to "creatures you
        # control"). The face-down source's own static abilities
        # are already gated by being-face-down everywhere they
        # matter, so this is the sole non-counter contributor here.
        power_bonus, toughness_bonus, _ = _static_bonuses(state, obj)
        p = base_p + power_bonus
        t = base_t + toughness_bonus
        p += obj.plus_one_counters - obj.minus_one_counters
        t += obj.plus_one_counters - obj.minus_one_counters
        # Layer 7e switches.
        layer7e = sorted(
            (
                ce for ce in state.continuous_effects
                if ce.layer == 7 and ce.sublayer == 4
                and ce.kind == "switch_pt"
                and ce.target_id == obj.id
            ),
            key=lambda ce: ce.timestamp,
        )
        for _ in layer7e:
            p, t = t, p
        return p, t
    # 7b: base P/T.
    layer7b = _registry_set_base_pt(state, obj)
    if layer7b is not None:
        base_p, base_t = layer7b
    elif obj.temp_base_pt_override is not None:
        base_p, base_t = obj.temp_base_pt_override
    else:
        dynamic = _dynamic_base_pt(state, obj)
        if dynamic is not None:
            base_p, base_t = dynamic
        else:
            base_p = obj.printing.power or 0
            base_t = obj.printing.toughness or 0
    # 7c: modify P/T (anthems, pumps).
    power_bonus, toughness_bonus, _ = _static_bonuses(state, obj)
    p = (
        base_p + power_bonus
        + obj.temp_power_bonus + obj.extended_temp_power_bonus
    )
    t = (
        base_t + toughness_bonus
        + obj.temp_toughness_bonus + obj.extended_temp_toughness_bonus
    )
    # 7d: counters. +1/+1 and -1/-1 are normally cancelled out by
    # the CR 704.5n SBA before this read, but if they coexist
    # transiently (between counter additions and the next SBA pass)
    # we still apply both signs.
    p += obj.plus_one_counters - obj.minus_one_counters
    t += obj.plus_one_counters - obj.minus_one_counters
    # 7e: switches. Each entry contributes a single swap.
    layer7e = sorted(
        (
            ce for ce in state.continuous_effects
            if ce.layer == 7 and ce.sublayer == 4
            and ce.kind == "switch_pt"
            and ce.target_id == obj.id
        ),
        key=lambda ce: ce.timestamp,
    )
    for _ in layer7e:
        p, t = t, p
    return p, t


def effective_power(state: GameState, obj: GameObject) -> int:
    """Power after walking layer 7 sublayers (see
    ``_compute_effective_pt`` for the full ordering)."""
    return _compute_effective_pt(state, obj)[0]


def effective_toughness(state: GameState, obj: GameObject) -> int:
    """Toughness after walking layer 7 sublayers (see
    ``_compute_effective_pt`` for the full ordering)."""
    return _compute_effective_pt(state, obj)[1]


@dataclass
class StackItem:
    """A spell or triggered ability on the stack.

    When ``ability`` is None the stack item represents a spell being cast;
    resolving it moves ``source`` to the battlefield. When ``ability`` is set,
    the item represents a triggered ability; resolving it runs the effect.

    ``targets`` holds (kind, id) pairs chosen when the ability went on the
    stack — empty for spells and for untargeted abilities.
    """

    source: GameObject
    controller_id: int
    # Either a TriggeredAbility or an ActivatedAbility — both have an
    # ``.effect`` attribute and resolve the same way.
    ability: "object | None" = None
    targets: tuple[tuple[str, int], ...] = ()
    # True when the spell was countered at cast time (currently by unpaid
    # Ward). A countered spell resolves as a no-op, with the card going
    # to its owner's graveyard (for instants/sorceries) or vanishing
    # (for permanents) without firing any effects/ETBs.
    countered: bool = False
    # The value chosen for {X} in this spell's cost (0 for non-X spells).
    # Set by _cast_spell via agent.choose_x; exposed to the resolving
    # effect via state.resolving_x_value.
    x_value: int = 0
    # True when the spell was cast with its kicker cost paid. Exposed
    # to the resolving effect via state.resolving_kicked.
    kicked: bool = False
    # Id of the creature the caster designated as a
    # choose-your-creature additional cost. None for spells that
    # don't have such a cost. Mirrored to
    # state.resolving_additional_cost_choice_id during resolution.
    additional_cost_choice_id: Optional[int] = None
    # Combat damage amount carried by a combat-damage-triggered ability
    # while it sits on the stack. Mirrored to
    # state.resolving_combat_damage_amount during resolution so effects
    # like DrawCardsEqualToResolvingCombatDamage can read "that many"
    # even though the ability resolves after the damage step ends.
    # 0 for non-combat-damage triggers and for spells.
    combat_damage_amount: int = 0
    # Controller id of the creature dealing combat damage in a
    # combat-damage-triggered ability while it sits on the stack.
    # Mirrored to state.resolving_combat_damage_source_controller_id
    # during resolution so observer effects like Contested War Zone's
    # ChangeControllerToCombatDamageSource can read the attacker's
    # controller without having to thread it through the targets tuple.
    # None for non-combat-damage triggers.
    combat_damage_source_controller_id: Optional[int] = None
    # Counter amount carried by a counters-added-to-creature observer
    # trigger. Mirrored to state.resolving_counter_amount during
    # resolution so effects like Terrasymbiosis ("draw that many cards")
    # can read the count even though the ability resolves later.
    counter_amount: int = 0
    # Modal-spell mode picks chosen at cast time (CR 601.2e: modes
    # are picked BEFORE total cost is determined at 601.2f). For a
    # plain "Choose one" this is a 1-tuple; for "Choose one or both"
    # it's a 1- or 2-tuple in resolution order. Empty for non-modal
    # spells. Mirrored to state.resolving_modes during resolution so
    # ModalChooseOne.resolve can read the pre-picked choice instead
    # of prompting the agent again at resolve time.
    chosen_modes: tuple[int, ...] = ()
    # DSK Rooms: when a Room spell is cast, the half-name the
    # controller chose to cast for. Resolved into
    # ``GameObject.unlocked_halves`` when the Room enters the
    # battlefield. ``None`` for non-Room spells.
    cast_room_half: "str | None" = None


@dataclass
class Player:
    id: int
    life: int = 20
    # Starting life total — used by effects that reference it directly
    # (e.g. "your life total becomes half your starting life total").
    # Defaults to 20 (MTG standard); adjust for alternate formats.
    starting_life: int = 20
    # Poison counters (MTG rule 704.5c). A player with 10+ poison
    # counters loses the game — enforced via state-based actions.
    poison_counters: int = 0
    library: list[GameObject] = field(default_factory=list)
    hand: list[GameObject] = field(default_factory=list)
    graveyard: list[GameObject] = field(default_factory=list)
    exile: list[GameObject] = field(default_factory=list)
    lands_played_this_turn: int = 0
    # Number of lands that have entered the battlefield under this player's
    # control during the current turn. Distinct from lands_played_this_turn
    # (which counts the per-turn land-drop action only): incremented for
    # *every* land ETB, including basics fetched by Lander tokens or other
    # search-and-put-onto-battlefield effects. Reset at cleanup. Used by
    # cards that read "for each land that entered the battlefield under your
    # control this turn" (Bioengineered Future).
    lands_entered_this_turn: int = 0
    # Number of spells this player has cast in the current turn. Reset at
    # cleanup. Used by "whenever you cast your Nth spell each turn" triggers.
    spells_cast_this_turn: int = 0
    # Number of noncreature spells this player has cast this turn. Reset
    # at cleanup. Used by the Deafening Silence restriction ("each player
    # can't cast more than one noncreature spell each turn") — legal_actions
    # gates additional noncreature casts when any permanent on the
    # battlefield has ``restricts_noncreature_spells_to_one_per_turn``.
    noncreature_spells_cast_this_turn: int = 0
    # The ``GameObject``s this player has declared as attackers this
    # turn (in declaration order). Populated by
    # ``combat.declare_attackers`` immediately after each attacker has
    # ``attacking=True`` set. Reset at the cleanup step in
    # ``turn.run_cleanup_step``. Drives the
    # ``AttackedWithThisTurn(filter)`` cost-reduction condition — the
    # condition filters this list at evaluation time, so any future
    # "you attacked with a [type] this turn" gate composes by swapping
    # the filter rather than adding a new latch.
    attacked_this_turn: list["GameObject"] = field(default_factory=list)
    # Number of artifacts that entered the battlefield under this
    # player's control this turn. Reset at cleanup. Incremented by
    # ``actions.put_into_battlefield`` whenever an artifact (including
    # artifact creatures and artifact tokens) enters under their
    # control. Read by ``ControlledArtifactEnteredThisTurn`` condition
    # which gates Mechan Shieldmate's defender-removing
    # ``ContinuousModifier`` via ``effective_keywords``.
    artifacts_entered_under_your_control_this_turn: int = 0
    has_lost: bool = False
    lost_reason: str = ""
    agent: Optional["Agent"] = None
    # Mana pool: list of ManaCoin entries (color + optional spend
    # restriction). Populated by mana-ability activations; consumed
    # before lands when paying costs; emptied at cleanup step.
    mana_pool: list["ManaCoin"] = field(default_factory=list)
    # Total life gained this turn. Reset at cleanup. Consumed by
    # effects that key on "if you gained life this turn" (Ragost,
    # Deft Gastronaut's end-step untap). Incremented by gain_life().
    life_gained_this_turn: int = 0
    # Pending agent swap scheduled by The Dominion Bracelet's
    # "You control target opponent during their next turn". When set,
    # run_untap_step installs the override at the start of this
    # player's next turn (saving the original agent in
    # ``agent_override_saved`` so cleanup can revert).
    scheduled_agent_override: Optional["Agent"] = None
    scheduled_agent_override_by: Optional[int] = None
    agent_override_saved: Optional["Agent"] = None


@dataclass
class ManaCoin:
    """A unit of mana in a player's pool.

    ``color`` is one of "W", "U", "B", "R", "G", "C" (colorless), or
    "any" (a wildcard produced by effects like All-Fates Scroll's
    "{T}: Add one mana of any color"). An "any" coin can satisfy any
    colored requirement; colorless coins only satisfy generic
    requirements.

    ``spend_restriction`` gates what the coin can pay for. None =
    unrestricted. Known values:
      - "artifact_spell" — Steelswarm Operator / Power Depot
        ("Spend this mana only to cast an artifact spell").
      - "artifact_activation" — Steelswarm Operator's second mana
        ability ("Spend this mana only to activate abilities of
        artifact sources").
      - "colorless_eldrazi" — Eldrazi Temple's second mana ability
        ("Spend this mana only to cast colorless Eldrazi spells or
        activate abilities of colorless Eldrazi").
      - "legendary_spell" — Plaza of Heroes's second mana ability
        ("Spend this mana only to cast a legendary spell").
      - "legendary_permanent_ability" — Plaza of Heroes's third mana
        ability ("Add one mana of any color among legendary
        permanents you control").
      - "land_activation" — Sunken Citadel's second mana ability
        ("Spend this mana only to activate abilities of land
        sources").
      - "cast_from_outside_hand" — Mm'menon, the Right Hand's
        artifact-grant mana ability ("Spend this mana only to cast a
        spell from anywhere other than your hand").
    See ``actions._coin_satisfies_context`` for how each tag matches
    a payment context.
    """

    color: str
    spend_restriction: Optional[str] = None


def _clone_mana_coin(coin: ManaCoin) -> ManaCoin:
    """Fast clone for ManaCoin — same ``__new__`` + ``__dict__`` copy
    pattern as ``GameState.clone``'s per-object cloners. ManaCoin's
    two fields are immutable strings, so the shared __dict__ values
    are safe; only the dict itself is duplicated."""
    new = ManaCoin.__new__(ManaCoin)
    new.__dict__ = coin.__dict__.copy()
    return new


class LogBuffer(list[str]):
    """A list[str] with a global silent-mode switch.

    ``append(msg)`` early-returns when ``enabled`` is False. All other
    list operations behave exactly like a normal ``list``. The engine
    appends to ``state.log`` from hundreds of sites; turning this
    buffer off lets search/rollout code skip those string allocations
    without touching any call site.

    Reads still work — tests that inspect log contents continue to
    pass while the buffer is enabled (the default).
    """

    enabled: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enabled = True

    def append(self, msg: str) -> None:  # type: ignore[override]
        if not self.enabled:
            return
        list.append(self, msg)


@dataclass(frozen=True)
class ContinuousEffect:
    """A continuous effect record in the layered-effects registry.

    Models CR 613's "active continuous effects." Effects are stored
    as data (not closures) so the registry survives ``GameState.clone``
    and pickling for MCTS spawn. The layer-walker (Phase 2+)
    dispatches on ``kind`` to compute characteristics.

    Fields:

    - ``source_id``: the GameObject that produced this effect (the
      aura, the static-ability source, the spell that resolved it).
      When the source leaves play (or for one-shots that complete,
      when explicitly removed), the entry is dropped from the
      registry.
    - ``layer``: 1–7 per CR 613.1. 1=copy, 2=control, 3=text,
      4=type, 5=color, 6=ability, 6.5=ability-modifying-PT (aka
      part of layer 7 but historically grouped), 7=P/T.
    - ``sublayer``: 0 by default. For layer 7 the spec defines
      a–d (we use 0–3): 0 = base P/T set, 1 = base P/T modified
      by counters/effects, 2 = +N/+N modifications, 3 = switch
      P/T. Within the same (layer, sublayer), ties resolve by
      ``timestamp``.
    - ``timestamp``: monotonically increasing per ``GameState``;
      lower = older = applied first within a layer.
    - ``kind``: dispatch key, e.g. ``"control_change"``,
      ``"grant_keyword"``, ``"set_base_pt"``, ``"pump_pt"``,
      ``"add_subtype"``. Layer-2 control-change is the first kind
      that will get migrated in Phase 2.
    - ``target_filter``: predicate name describing which permanents
      this effect applies to. Empty means ``params`` carries an
      explicit ``target_id``. Examples:
      ``"self"`` (the source), ``"attached_to_source"`` (the
      permanent the source is attached to — auras),
      ``"creatures_you_control"``, ``"all_artifacts_you_control"``.
    - ``params``: tuple of kind-specific values. For
      ``"control_change"`` the tuple is ``(new_controller_id,)``.
      For ``"grant_keyword"`` it's ``(keyword_str,)``. Etc.
    - ``description``: human-readable summary for log lines and
      debugging.
    """

    source_id: int
    layer: int
    timestamp: int
    kind: str
    sublayer: int = 0
    target_filter: str = ""
    target_id: Optional[int] = None
    params: tuple = ()
    description: str = ""
    # Lifetime marker. None = entry persists until the source leaves
    # the battlefield (the default for static abilities — Tractor
    # Beam's control grant). Other values:
    #   - "cleanup": drop at the active player's cleanup step (UEOT
    #     pump-grants — Squire's Lightblade's first strike, manland
    #     animation, granted-triggers UEOT, etc.).
    #   - "controller_untap": drop at the source-controller's next
    #     untap step (extended-temporal pumps — Mouth of the Storm).
    #     ``expires_at_player_id`` carries the controller id whose
    #     untap step terminates the entry.
    expires_at: Optional[str] = None
    expires_at_player_id: Optional[int] = None


@dataclass
class GameState:
    players: list[Player] = field(default_factory=list)
    battlefield: list[GameObject] = field(default_factory=list)
    stack: list[StackItem] = field(default_factory=list)
    active_player_idx: int = 0
    priority_player_idx: int = 0
    turn_number: int = 1
    step: Optional["Step"] = None
    next_object_id: int = 0
    # Layered-effects registry (CR 613). Phase 1 introduces it as
    # additive scaffolding — the list is empty by default and no
    # current code consults it. Phase 2 will start migrating
    # field-mutation-based continuous effects (aura control swaps,
    # anthem statics, manland animation, granted abilities, etc.)
    # to live as ``ContinuousEffect`` records here, with
    # ``effective_*`` helpers walking the registry in layer +
    # timestamp order. ``next_timestamp`` is the monotonic counter
    # allocated by ``add_continuous_effect``; lower = older = applied
    # first within a layer.
    continuous_effects: list[ContinuousEffect] = field(default_factory=list)
    next_timestamp: int = 0
    log: LogBuffer = field(default_factory=LogBuffer)
    # Canonical game-wide RNG. Engine sites that need randomness (library
    # shuffles from Nexus, tutor shuffles, ward random-discard, etc.) pull
    # from ``state.rng`` so a run is reproducible from a single seed and
    # so search can fork a clone's RNG cleanly. ``simulate.setup_game``
    # seeds this from the caller-supplied seed.
    rng: random.Random = field(default_factory=random.Random)
    winner_id: Optional[int] = None
    is_draw: bool = False
    # EoE "Void" turn-state flags. Both are reset on cleanup (end of turn).
    # ``permanent_left_battlefield_this_turn`` is set whenever any nonland
    # permanent leaves the battlefield (destroy, exile, bounce, cease-to-
    # exist, warp-exile). ``spell_warped_this_turn`` is set when a spell
    # was cast via its warp alt-cost. The ``is_void_active`` helper returns
    # True when either flag is set, which is the Void trigger condition.
    permanent_left_battlefield_this_turn: bool = False
    spell_warped_this_turn: bool = False
    # Tracks (source_id, ability_index) pairs for triggered abilities that
    # have already fired this turn under the ``once_per_turn`` restriction.
    # Reset by run_cleanup_step.
    once_per_turn_fired: set[tuple[int, int]] = field(default_factory=set)
    # Delayed triggered abilities scheduled for a future end step. Each
    # entry is a dict: {"fire_at_turn": int, "controller_id": int,
    # "action": str, "target_id": int}.
    # Actions currently supported: "sacrifice".
    delayed_triggers: list[dict] = field(default_factory=list)
    # Source GameObject of the currently-resolving ability, if any. Set by
    # resolve_top_of_stack during ability resolution so effects like
    # "put a +1/+1 counter on this creature" can reach the source without
    # having to thread it through every resolve() signature.
    resolving_source: Optional[GameObject] = None
    # Controller id pinned for the current legal-targets enumeration.
    # Set transiently by ``Effect._enumerate_for_filter`` (and by
    # any peer call that consults a ``TargetFilter``) so controller-
    # aware atoms (``ControlledByOpponent`` / ``ControlledByYou``)
    # can compare an object's ``controller_id`` against the spell /
    # ability's controller without threading the id through every
    # ``ObjectPredicate.test`` signature. Cleared back to ``None``
    # after the enumeration unwinds.
    resolving_controller_id: Optional[int] = None
    # Value of {X} for the currently-resolving spell or ability. Mirrors
    # StackItem.x_value during resolution so effects can read X without
    # threading it through every resolve() signature. 0 when no X spell
    # is resolving.
    resolving_x_value: int = 0
    # Mirrors StackItem.kicked during resolution so effects can branch
    # on whether the spell was kicked without threading it through
    # every resolve() signature. False outside resolution.
    resolving_kicked: bool = False
    # Mirrors StackItem.chosen_modes during resolution so a modal
    # spell's resolve() reads the cast-time pick (CR 601.2e) rather
    # than prompting the agent again at resolve time. Empty tuple
    # outside modal-spell resolution.
    resolving_modes: tuple[int, ...] = ()
    # Mirrors StackItem.additional_cost_choice_id during resolution so
    # effects (e.g. Close Encounter) can read the creature the caster
    # designated as an additional cost. None outside resolution.
    resolving_additional_cost_choice_id: Optional[int] = None
    # Amount of combat damage dealt by the source that is currently
    # firing a combat-damage trigger. 0 outside combat-trigger
    # resolution. Used by effects like Starwinder's "may draw that
    # many cards" that need to know the damage amount.
    resolving_combat_damage_amount: int = 0
    # Controller id of the creature dealing combat damage in a
    # currently-firing combat-damage-to-you trigger. None outside such
    # resolution. Used by Contested War Zone's "that creature's
    # controller gains control of this land".
    resolving_combat_damage_source_controller_id: Optional[int] = None
    # Number of +1/+1 counters being added during a resolving
    # counters-added observer trigger (Terrasymbiosis: "draw that many
    # cards"). 0 outside such resolution.
    resolving_counter_amount: int = 0
    # IDs of permanents scheduled to be sacrificed at end of combat.
    # Cleared during clear_combat_state after processing.
    # Used by Kavaron Harrier's attack token.
    pending_end_of_combat_sacrifices: list[int] = field(default_factory=list)
    # Player ids who have already created one or more tokens this turn.
    # Gates "the first time you would create tokens each turn"
    # replacement effects (Moonlit Meditation). Cleared at cleanup.
    players_who_created_tokens_this_turn: set[int] = field(default_factory=set)
    # Per-turn counter of how many times a tagged ability has resolved.
    # Keys are effect-specific labels (typically ("tag", source_id)).
    # Reset by run_cleanup_step. Used by Tannuk, Memorial Ensign's
    # "if this is the second time this ability has resolved" rider.
    ability_resolutions_this_turn: dict = field(default_factory=dict)
    # When True, combat damage assignment in combat._resolve_damage_step
    # is skipped entirely for the remainder of the turn. Set by Darkness
    # ({B} instant) and reset by run_cleanup_step. Non-combat damage
    # (direct damage spells, pingers, infect from non-combat sources)
    # is unaffected.
    prevent_combat_damage_this_turn: bool = False
    # FIFO queue of player ids scheduled to take an extra turn after
    # the current one wraps up. Populated by TakeExtraTurnAfterThisOne
    # (Nexus of Fate). Consumed by simulate.simulate_game between
    # turns: if the queue is non-empty, the next turn's active player
    # is the head of the queue instead of the natural next alive player.
    extra_turn_queue: list[int] = field(default_factory=list)
    # Per-turn count of creatures that have died (moved from battlefield
    # to graveyard while reading as creatures) regardless of controller.
    # Incremented in ``turn.move_off_battlefield`` for every dies event
    # the engine fires. Reset by ``run_cleanup_step``. Consumed by
    # ETB replacement effects keyed on ``a creature died this turn``
    # (Cackling Slasher's printed +1/+1 counter rider). The tracker
    # COUNTS deaths from any player — the printed clause doesn't
    # specify "you control", so a dies on either side suffices.
    creatures_died_this_turn: int = 0

    def active_player(self) -> Player:
        return self.players[self.active_player_idx]

    def player_by_id(self, player_id: int) -> Player:
        for p in self.players:
            if p.id == player_id:
                return p
        raise KeyError(f"no player with id={player_id}")

    def index_of(self, player_id: int) -> int:
        for i, p in enumerate(self.players):
            if p.id == player_id:
                return i
        raise KeyError(f"no player with id={player_id}")

    def new_object(
        self, printing: CardPrinting, owner_id: int, zone: Zone
    ) -> GameObject:
        obj = GameObject(
            id=self.next_object_id,
            printing=printing,
            owner_id=owner_id,
            controller_id=owner_id,
            zone=zone,
        )
        self.next_object_id += 1
        return obj

    def is_over(self) -> bool:
        return self.winner_id is not None or self.is_draw

    def add_continuous_effect(
        self,
        *,
        source_id: int,
        layer: int,
        kind: str,
        sublayer: int = 0,
        target_filter: str = "",
        target_id: Optional[int] = None,
        params: tuple = (),
        description: str = "",
        expires_at: Optional[str] = None,
        expires_at_player_id: Optional[int] = None,
    ) -> ContinuousEffect:
        """Allocate a fresh timestamp and append a continuous-effect
        record to the registry. Returns the appended record so callers
        can hold onto it (e.g., to reference its source_id later).
        ``expires_at`` defaults to None (lives until the source
        leaves play); pass ``"cleanup"`` for UEOT effects, or
        ``"controller_untap"`` (with ``expires_at_player_id`` set to
        the relevant controller id) for extended-temporal pumps.
        """
        ts = self.next_timestamp
        self.next_timestamp += 1
        effect = ContinuousEffect(
            source_id=source_id,
            layer=layer,
            timestamp=ts,
            kind=kind,
            sublayer=sublayer,
            target_filter=target_filter,
            target_id=target_id,
            params=params,
            description=description,
            expires_at=expires_at,
            expires_at_player_id=expires_at_player_id,
        )
        self.continuous_effects.append(effect)
        return effect

    def grant_triggered_ability_ueot(
        self,
        *,
        source_id: int,
        target_id: int,
        ability,
        description: str = "",
    ) -> ContinuousEffect:
        """Helper for UEOT triggered-ability grants. Writes a layer-6
        ``"grant_triggered_ability"`` entry with ``expires_at="cleanup"``.
        Read via ``effective_triggered_abilities``."""
        return self.add_continuous_effect(
            source_id=source_id,
            layer=6,
            kind="grant_triggered_ability",
            target_id=target_id,
            params=(ability,),
            expires_at="cleanup",
            description=description,
        )

    def grant_activated_ability_ueot(
        self,
        *,
        source_id: int,
        target_id: int,
        ability,
        description: str = "",
    ) -> ContinuousEffect:
        """Helper for UEOT activated-ability grants (manland
        animations). Writes a layer-6 ``"grant_activated_ability"``
        entry with ``expires_at="cleanup"``. Read via
        ``effective_activated_abilities``."""
        return self.add_continuous_effect(
            source_id=source_id,
            layer=6,
            kind="grant_activated_ability",
            target_id=target_id,
            params=(ability,),
            expires_at="cleanup",
            description=description,
        )

    def grant_pt_pump_ueot(
        self,
        *,
        source_id: int,
        target_id: int,
        power: int,
        toughness: int,
        description: str = "",
    ) -> ContinuousEffect:
        """Helper for the layer-7 sublayer-2 ``"pump_pt"`` UEOT-grant
        pattern. Writes a single registry entry with
        ``params=(power_delta, toughness_delta)`` and
        ``expires_at="cleanup"``, AND mirrors the deltas onto the
        target's ``temp_power_bonus`` / ``temp_toughness_bonus``
        fields as a write-through cache. ``effective_power`` /
        ``effective_toughness`` continue to read from the field
        (the cache is the read path), while the registry holds the
        layer-7 metadata for future layer-aware ordering work.
        ``run_cleanup_step`` zeros both. No-op when both deltas
        are zero."""
        if power == 0 and toughness == 0:
            return None  # type: ignore[return-value]
        target = next(
            (o for o in self.battlefield if o.id == target_id), None,
        )
        if target is not None:
            target.temp_power_bonus += power
            target.temp_toughness_bonus += toughness
        return self.add_continuous_effect(
            source_id=source_id,
            layer=7,
            sublayer=2,
            kind="pump_pt",
            target_id=target_id,
            params=(power, toughness),
            expires_at="cleanup",
            description=description,
        )

    def grant_become_creature_ueot(
        self,
        *,
        source_id: int,
        target_id: int,
        power: int,
        toughness: int,
        subtype: Optional[str] = None,
        description: str = "",
    ) -> None:
        """Helper for the manland-animation pattern: target becomes a
        creature with base P/T (and optional subtype) until end of
        turn. Writes three registry entries —

          - layer 4 ``"add_type"`` params=("creature",)
          - layer 4 ``"add_subtype"`` params=(subtype,) if given
          - layer 7 sublayer 1 ``"set_base_pt"`` params=(power, toughness)

        — all with ``expires_at="cleanup"``. Mirrors the deltas onto
        the legacy ``temp_is_creature`` / ``temp_base_pt_override`` /
        ``temp_subtype`` cache fields so existing read sites
        (``is_creature_now``, ``effective_power``, etc.) keep
        working unchanged. The split-by-layer recording lets a
        future layer-aware refactor replay the manland animation in
        proper CR-613 order (layer 4 type before layer 7 P/T).
        """
        target = next(
            (o for o in self.battlefield if o.id == target_id), None,
        )
        if target is not None:
            target.temp_is_creature = True
            target.temp_base_pt_override = (power, toughness)
            if subtype is not None:
                target.temp_subtype = subtype
        self.add_continuous_effect(
            source_id=source_id, layer=4, kind="add_type",
            target_id=target_id, params=("creature",),
            expires_at="cleanup", description=description,
        )
        if subtype is not None:
            self.add_continuous_effect(
                source_id=source_id, layer=4, kind="add_subtype",
                target_id=target_id, params=(subtype,),
                expires_at="cleanup", description=description,
            )
        self.add_continuous_effect(
            source_id=source_id, layer=7, sublayer=1, kind="set_base_pt",
            target_id=target_id, params=(power, toughness),
            expires_at="cleanup", description=description,
        )

    def grant_pt_pump_until_player_next_turn(
        self,
        *,
        source_id: int,
        target_id: int,
        power: int,
        toughness: int,
        until_player_id: int,
        description: str = "",
    ) -> ContinuousEffect:
        """Helper for the layer-7 sublayer-2 ``"pump_pt"`` extended-
        temporal grant — "until your next turn" pumps. Mouth of the
        Storm's "creatures your opponents control get -3/-0 until
        your next turn" is the canonical case. Writes a registry
        entry with ``expires_at="controller_untap"`` and
        ``expires_at_player_id=until_player_id``, AND mirrors the
        deltas onto the legacy ``extended_temp_*`` cache fields. The
        entry expires at ``until_player_id``'s next untap step —
        ``run_untap_step`` calls ``expire_continuous_effects(
        "controller_untap", player_id=ap.id)``."""
        if power == 0 and toughness == 0:
            return None  # type: ignore[return-value]
        target = next(
            (o for o in self.battlefield if o.id == target_id), None,
        )
        if target is not None:
            target.extended_temp_power_bonus += power
            target.extended_temp_toughness_bonus += toughness
            target.extended_temp_expires_at_player_id = until_player_id
        return self.add_continuous_effect(
            source_id=source_id,
            layer=7,
            sublayer=2,
            kind="pump_pt",
            target_id=target_id,
            params=(power, toughness),
            expires_at="controller_untap",
            expires_at_player_id=until_player_id,
            description=description,
        )

    def grant_keywords_ueot(
        self,
        *,
        source_id: int,
        target_id: int,
        keywords,
        log_source_name: str = "",
        log_target_name: str = "",
    ) -> None:
        """Helper for the common layer-6 UEOT-keyword-grant pattern.
        Writes one ``ContinuousEffect`` entry per keyword
        (``layer=6, kind="grant_keyword", expires_at="cleanup"``).
        Effects that grant UEOT keywords (PumpTargetCreatureUntilEOT,
        BecomeLandCreatureUEOT, AttachToTargetCreatureYouControl, etc.)
        call this rather than mutating ``temp_keywords`` directly, so
        all UEOT keyword grants share one cleanup mechanism and one
        query path through ``effective_keywords``.
        """
        if not keywords:
            return
        for kw in sorted(keywords):
            desc = ""
            if log_source_name and log_target_name:
                desc = (
                    f"{log_source_name} grants {kw} UEOT to {log_target_name}"
                )
            self.add_continuous_effect(
                source_id=source_id,
                layer=6,
                kind="grant_keyword",
                target_id=target_id,
                params=(kw,),
                expires_at="cleanup",
                description=desc,
            )

    def grant_control_change_ueot(
        self,
        *,
        source_id: int,
        target_id: int,
        new_controller_id: int,
        description: str = "",
    ) -> None:
        """Helper for "Until end of turn, you gain control of target
        permanent" effects (Mutinous Massacre, Systems Override,
        Threaten / Act of Treason shape).

        Writes a layer-2 ``"control_change"`` registry entry with
        ``params=(new_controller_id,)`` and
        ``expires_at="cleanup"``. Captures the captured-baseline
        controller in ``controller_id_before_any_aura_swap`` if
        this is the first control-grant on the target, mirroring
        ``apply_aura_static_grants_on_attach``'s logic so the
        revert-to-baseline machinery in
        ``move_off_battlefield`` and ``run_cleanup_step``'s
        controller refresh works uniformly.

        Updates the ``target.controller_id`` write-through cache
        immediately via ``effective_controller`` so downstream
        reads (haste grants, attack eligibility) see the new
        controller right away.
        """
        target = next(
            (o for o in self.battlefield if o.id == target_id), None,
        )
        if target is None:
            return
        if effective_controller(self, target) == new_controller_id:
            return
        has_prior_grant = any(
            ce.layer == 2
            and ce.kind == "control_change"
            and ce.target_id == target.id
            for ce in self.continuous_effects
        )
        if not has_prior_grant:
            target.controller_id_before_any_aura_swap = target.controller_id
        self.add_continuous_effect(
            source_id=source_id,
            layer=2,
            kind="control_change",
            target_id=target.id,
            params=(new_controller_id,),
            expires_at="cleanup",
            description=description,
        )
        target.controller_id = effective_controller(self, target)

    def expire_continuous_effects(
        self, expires_at: str, *, player_id: Optional[int] = None,
    ) -> list[ContinuousEffect]:
        """Drop every registry entry whose ``expires_at`` matches
        the given marker (``"cleanup"`` for UEOT, ``"controller_untap"``
        for extended-temporal). Returns the dropped entries — useful
        for tests and for downstream cleanup hooks that need to react
        to specific kinds going away.

        For ``"controller_untap"`` the caller passes ``player_id`` —
        only entries whose ``expires_at_player_id`` matches are
        dropped (a Mouth-of-the-Storm-style "until your next turn"
        effect ends at the controller's untap step, not just any
        untap step). For ``"cleanup"`` the player_id is ignored
        because cleanup is a once-per-turn event tied to whoever
        the active player is at end-of-turn.
        """
        kept: list[ContinuousEffect] = []
        dropped: list[ContinuousEffect] = []
        for ce in self.continuous_effects:
            if ce.expires_at != expires_at:
                kept.append(ce)
                continue
            if (
                expires_at == "controller_untap"
                and player_id is not None
                and ce.expires_at_player_id != player_id
            ):
                kept.append(ce)
                continue
            dropped.append(ce)
        self.continuous_effects = kept
        return dropped

    def remove_continuous_effects_from_source(
        self, source_id: int, *, only_static: bool = False,
    ) -> list[ContinuousEffect]:
        """Drop every registry entry produced by ``source_id`` (called
        when that source leaves the battlefield, or when a one-shot
        effect's window closes). Returns the dropped entries — useful
        for log/debug and for tests that want to assert which effects
        were torn down.

        ``only_static`` (default False): when True, only drop entries
        with ``expires_at is None`` — the static-while-source-on-the-
        battlefield kind (Tractor Beam's control grant, Mm'menon's
        ability grant, etc.). UEOT entries (``expires_at="cleanup"``)
        survive the source leaving — they end at the next cleanup
        step regardless of whether the granting source is still on
        the battlefield. ``move_off_battlefield`` uses
        ``only_static=True`` so a UEOT-grant from a permanent that
        subsequently dies still fires its dependent triggers /
        queries before cleanup wipes it (Perigee Beckoner's granted
        dies-rider being the canonical case).
        """
        kept: list[ContinuousEffect] = []
        dropped: list[ContinuousEffect] = []
        for ce in self.continuous_effects:
            matches = ce.source_id == source_id
            if only_static and ce.expires_at is not None:
                matches = False
            if matches:
                dropped.append(ce)
            else:
                kept.append(ce)
        self.continuous_effects = kept
        return dropped

    def creatures_controlled_by(self, player_id: int) -> list[GameObject]:
        return [
            o for o in self.battlefield
            if o.controller_id == player_id and o.printing.is_creature
        ]

    def lands_controlled_by(self, player_id: int) -> list[GameObject]:
        return [
            o for o in self.battlefield
            if o.controller_id == player_id and o.printing.is_land
        ]

    def clone(self) -> "GameState":
        """Return an independent deep copy of this state, suitable for
        search rollouts.

        Semantics vs. ``copy.deepcopy(state)``:

        - CardPrinting refs are shared (frozen dataclasses with no
          mutable state). Same for frozensets and tuples of primitives.
        - All GameObjects are deep-copied. A ``{old_id: new_obj}`` map
          drives cross-reference rewiring so ``state.stack``'s StackItem
          sources and ``state.resolving_source`` point at the cloned
          objects.
        - Every zone list (battlefield + each player's library / hand /
          graveyard / exile / mana_pool) is a fresh list containing the
          cloned GameObjects.
        - ``state.log`` is a fresh empty ``LogBuffer`` with the same
          ``enabled`` flag — search rollouts don't need log history.
        - ``state.rng`` is a fresh ``random.Random`` seeded from the
          original's state, so the clone advances independently but
          deterministically.
        - ``Player.agent`` is **not** cloned — agents are decision-
          policy objects, not game state, and cloning them would both
          be wasteful and break mutable-agent contracts (RandomAgent's
          RNG, scheduled agent overrides). Search callers that need
          to vary the policy per rollout should swap the agent ref
          on the clone after calling ``.clone()``.

        GameObject integer id references (``attached_to``,
        ``exiled_by_source``, ``temp_original_controller_id``) carry
        integer ids — those stay valid without remapping because the
        new objects keep the same id.

        Performance: this is a search hot path (thousands of clones
        per MCTS decision). Per-object cloning uses a shallow
        ``object.__new__`` + ``__dict__`` copy rather than the
        dataclass __init__ with 30+ kwargs — the latter spends
        almost all its time in name-resolution overhead.
        ``GameObject.blocked_by`` is the only mutable per-object
        field; we duplicate that explicitly. Every other field is
        either an int / bool / None / frozenset / Enum (immutable)
        or a reference to a frozen-dataclass CardPrinting (also
        safe to share).
        """
        # Build the id->new_obj map by deep-copying every GameObject
        # referenced anywhere in the state. We walk zones in a fixed
        # order and copy each object exactly once.
        id_map: dict[int, GameObject] = {}

        def clone_obj(o: GameObject) -> GameObject:
            existing = id_map.get(o.id)
            if existing is not None:
                return existing
            # Fast clone: skip __init__ entirely, copy the __dict__ in
            # one C-level call, then duplicate the only mutable field.
            new = GameObject.__new__(GameObject)
            new.__dict__ = o.__dict__.copy()
            new.blocked_by = list(o.blocked_by)
            id_map[o.id] = new
            return new

        # Clone players (and every zone list on each player). Same
        # __new__ + __dict__ copy trick as clone_obj — avoids the
        # dataclass __init__'s 18+ kwarg dispatch. Agent is shared
        # (not game state). ManaCoin is a dataclass with two
        # immutable string fields and is never mutated post-creation
        # in the engine, so we copy via the same fast path.
        new_players: list[Player] = []
        for p in self.players:
            np = Player.__new__(Player)
            np.__dict__ = p.__dict__.copy()
            np.library = [clone_obj(o) for o in p.library]
            np.hand = [clone_obj(o) for o in p.hand]
            np.graveyard = [clone_obj(o) for o in p.graveyard]
            np.exile = [clone_obj(o) for o in p.exile]
            np.mana_pool = [_clone_mana_coin(c) for c in p.mana_pool]
            new_players.append(np)

        new_battlefield = [clone_obj(o) for o in self.battlefield]

        # Stack items must point at the cloned GameObject for their
        # source. All other fields are immutable; share by reference.
        new_stack: list[StackItem] = []
        for item in self.stack:
            ni = StackItem.__new__(StackItem)
            ni.__dict__ = item.__dict__.copy()
            ni.source = clone_obj(item.source)
            new_stack.append(ni)

        new_log = LogBuffer()
        new_log.enabled = self.log.enabled

        new_rng = random.Random()
        new_rng.setstate(self.rng.getstate())

        new_resolving_source = (
            clone_obj(self.resolving_source)
            if self.resolving_source is not None else None
        )

        new_state = GameState(
            players=new_players,
            battlefield=new_battlefield,
            stack=new_stack,
            active_player_idx=self.active_player_idx,
            priority_player_idx=self.priority_player_idx,
            turn_number=self.turn_number,
            step=self.step,
            next_object_id=self.next_object_id,
            continuous_effects=list(self.continuous_effects),
            next_timestamp=self.next_timestamp,
            log=new_log,
            rng=new_rng,
            winner_id=self.winner_id,
            is_draw=self.is_draw,
            permanent_left_battlefield_this_turn=(
                self.permanent_left_battlefield_this_turn
            ),
            spell_warped_this_turn=self.spell_warped_this_turn,
            once_per_turn_fired=set(self.once_per_turn_fired),
            delayed_triggers=[dict(d) for d in self.delayed_triggers],
            resolving_source=new_resolving_source,
            resolving_controller_id=self.resolving_controller_id,
            resolving_x_value=self.resolving_x_value,
            resolving_kicked=self.resolving_kicked,
            resolving_modes=self.resolving_modes,
            resolving_additional_cost_choice_id=(
                self.resolving_additional_cost_choice_id
            ),
            resolving_combat_damage_amount=(
                self.resolving_combat_damage_amount
            ),
            resolving_combat_damage_source_controller_id=(
                self.resolving_combat_damage_source_controller_id
            ),
            resolving_counter_amount=self.resolving_counter_amount,
            pending_end_of_combat_sacrifices=list(
                self.pending_end_of_combat_sacrifices
            ),
            players_who_created_tokens_this_turn=set(
                self.players_who_created_tokens_this_turn
            ),
            ability_resolutions_this_turn=dict(
                self.ability_resolutions_this_turn
            ),
            prevent_combat_damage_this_turn=(
                self.prevent_combat_damage_this_turn
            ),
            extra_turn_queue=list(self.extra_turn_queue),
            creatures_died_this_turn=self.creatures_died_this_turn,
        )
        return new_state
