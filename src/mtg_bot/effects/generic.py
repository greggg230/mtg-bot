"""Generic, reusable effects — damage, life, counters, zone moves,
tokens, pumps, search, draw, tap/untap, etc. Card-agnostic primitives that
many printings reuse. Card-specific one-off effects live in ``bespoke.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mtg_bot.effects.base import (
    Effect, Target, TriggeredAbility,
    _apply_damage, _multicolored_permanents_count,
    _find_moonlit_replacement_source,
)

if TYPE_CHECKING:
    from mtg_bot.cards import ManaCost
    from mtg_bot.game import GameState


def _eff_subtypes(state: "GameState", obj) -> tuple[str, ...]:
    """Local alias for ``mtg_bot.game.effective_subtypes`` so
    subtype-keyed legal-target filters (anti-subtype, allowed-subtype,
    target-subtype) can read the layered-effects view of a battlefield
    permanent's subtypes without circular-import gymnastics. Lazy
    import — the function-level fallthrough avoids importing game at
    module load."""
    from mtg_bot.game import effective_subtypes
    return effective_subtypes(state, obj)


@dataclass(frozen=True)
class DealDamage(Effect):
    """Unified deal-damage primitive driven by ``targets``.

    Replaces the eleven pre-PR-#3 deal-damage recognizer-classes at
    the static-amount level (``DealDamageAnyTarget``,
    ``DealDamageToTargetCreature``, ``DealDamageToOpponentCreature``,
    ``DealDamageToTargetTappedCreature``,
    ``DealDamageToTargetOpponent``). The ``targets`` field — typically
    a single ``TargetSpec`` for damage today — declares the legal-
    target shape via the ``TargetFilter`` grammar; the resolve walk
    re-verifies legality at fire time per CR 608.2b and applies the
    damage to whichever slots survive.

    Bespoke-amount damage classes (count-source artifact / cards-in-
    hand, X-base-plus-creature-count Focus Fire, excess-rider
    Lander, Spacecraft-boosted) keep their own engine classes —
    they need bespoke amount logic — but consult ``targets[0].filter``
    via ``Effect._enumerate_for_filter`` to share the same target-
    grammar plumbing.
    """

    amount: int = 0
    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        # Re-verify each chosen target against its slot's filter at
        # resolution per CR 608.2b. Slots whose object is no longer
        # legal (creature untapped, blocking creature stopped blocking,
        # etc.) log a fizzle line and skip — the user-visible feedback
        # mirrors the legacy per-class messages ("is no longer tapped
        # — damage fizzles") with a generic phrasing the unified
        # dispatcher can produce without introspecting the filter.
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    # Back-references aren't in the targets tuple
                    # under PR #3 — defensively skip.
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(_fizzle_message(state, chosen))
                    continue
                _apply_damage(state, chosen, self.amount)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return f"deals {self.amount} damage to {_describe_targets(self.targets)}"


def _slot_target_still_legal(state, slot, chosen):
    """Return True if ``chosen`` (engine ``("kind", id)`` tuple) still
    satisfies ``slot.filter`` against the current ``state``. Pure
    helper used by ``DealDamage.resolve`` and any future new-shape
    re-verification path."""
    kind, target_id = chosen
    if kind == "player":
        try:
            obj = state.player_by_id(target_id)
        except KeyError:
            return False
        if getattr(obj, "has_lost", False):
            return False
    else:
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return False
    return slot.filter.test(obj, state)


def _fizzle_message(state, chosen, action: str = "damage"):
    """Produce the user-visible log line for a target whose at-resolve
    legality re-check failed. Names the target where possible so the
    log entry reads like the legacy per-class fizzle messages without
    each effect having to compose its own string. ``action`` slots into
    the trailing "...fizzles" phrase ("damage fizzles", "destroy
    fizzles", "exile fizzles") so cross-family migrations can share
    the helper."""
    kind, target_id = chosen
    if kind == "player":
        return (
            f"P{target_id} is no longer a legal target — {action} fizzles"
        )
    obj = next(
        (o for o in state.battlefield if o.id == target_id), None
    )
    if obj is None:
        return (
            f"target is no longer on the battlefield — {action} fizzles"
        )
    name = obj.printing.name
    if action == "damage" and not getattr(obj, "tapped", False):
        # Common case for damage targeting tapped creatures: the slot's
        # filter was IsCreature() & IsTapped() and the creature
        # untapped between target-pick and resolve. Naming the specific
        # change reads cleanest in the log; falls back to the generic
        # phrasing for other re-check failures.
        return f"{name} is no longer tapped — damage fizzles"
    return f"{name} is no longer a legal target — {action} fizzles"


def _describe_targets(targets) -> str:
    """Coarse human-readable rendering of a target-slot tuple. PR #3
    only ships single-slot damage but the helper handles the multi-
    slot case for forward compat."""
    from mtg_bot.effects.targets import SlotRef, TargetSpec
    if not targets:
        return "any target"
    parts: list[str] = []
    for slot in targets:
        if isinstance(slot, SlotRef):
            parts.append(f"that target (slot {slot.idx})")
        elif isinstance(slot, TargetSpec):
            parts.append(_describe_filter(slot.filter))
        else:
            parts.append("?")
    return " and ".join(parts)


def _describe_filter(target_filter) -> str:
    """Render a ``TargetFilter`` as a printed-style noun phrase for
    log lines / describe(). Coarse — exact wording is owned by the
    parser's phrasebook on the way in."""
    from mtg_bot.effects.object_predicates import (
        ControlledByOpponent, ControlledByYou,
        IsArtifact, IsAttacking, IsBasic, IsBlocking, IsCreature,
        IsEnchantment, IsLand, IsPlayer, IsTapped, ObjectAnd,
        ObjectNot, ObjectOr,
    )
    from mtg_bot.effects.target_filter import OneOf
    from mtg_bot.effects.targets import any_target_filter
    if target_filter == any_target_filter():
        return "any target"
    if isinstance(target_filter, IsCreature):
        return "target creature"
    if isinstance(target_filter, IsPlayer):
        return "target player"
    if isinstance(target_filter, IsArtifact):
        return "target artifact"
    if isinstance(target_filter, IsLand):
        return "target land"
    if isinstance(target_filter, IsEnchantment):
        return "target enchantment"
    if isinstance(target_filter, ObjectAnd):
        l = target_filter.left
        r = target_filter.right
        if isinstance(l, IsCreature) and isinstance(r, IsTapped):
            return "target tapped creature"
        if isinstance(l, IsCreature) and isinstance(r, ControlledByOpponent):
            return "target creature an opponent controls"
        if isinstance(l, IsCreature) and isinstance(r, ControlledByYou):
            return "target creature you control"
        if isinstance(l, IsPlayer) and isinstance(r, ControlledByOpponent):
            return "target opponent"
        if isinstance(l, IsLand) and isinstance(r, ObjectNot) and isinstance(
            r.inner, IsBasic
        ):
            return "target nonbasic land"
        if isinstance(l, IsCreature) and isinstance(r, ObjectOr):
            inner_l, inner_r = r.left, r.right
            if (isinstance(inner_l, IsAttacking)
                    and isinstance(inner_r, IsBlocking)):
                return "target attacking or blocking creature"
    if isinstance(target_filter, OneOf):
        return "any target"  # default rendering for OneOf disjuncts
    return "target"


def _bounce_to_hand(state: "GameState", obj) -> None:
    """Common bounce-side-effects helper. Removes ``obj`` from the
    battlefield, resets combat / counters / temp_* state, marks the
    Void-tracking flag, and either lets a token cease to exist or
    moves the card to its owner's hand and reveals it. Pure helper —
    used by every variant of the bounce family so each subclass
    needs only the per-card filter and any rider snapshot logic.
    """
    from mtg_bot.game import Zone, reveal_to_all

    state.battlefield.remove(obj)
    obj.damage = 0
    obj.marked_lethal = False
    obj.damaged_this_turn = False
    obj.tapped = False
    obj.attacking = False
    obj.blocking = None
    obj.blocked_by = []
    obj.plus_one_counters = 0
    obj.charge_counters = 0
    obj.temp_power_bonus = 0
    obj.temp_toughness_bonus = 0
    obj.temp_keywords = frozenset()
    if not obj.printing.is_land:
        state.permanent_left_battlefield_this_turn = True
    if obj.printing.is_token:
        state.log.append(
            f"{obj.printing.name} token ceases to exist (bounced)"
        )
        return
    obj.zone = Zone.HAND
    state.player_by_id(obj.owner_id).hand.append(obj)
    reveal_to_all(state, obj)
    state.log.append(
        f"{obj.printing.name} returns to P{obj.owner_id}'s hand"
    )


@dataclass(frozen=True)
class PutPlusOneCounters(Effect):
    """Unified put-+1/+1-counters primitive driven by ``targets``.

    Sibling of :class:`Destroy` / :class:`Exile` / :class:`Tap` /
    :class:`Pump` / :class:`Bounce` for the +1/+1 counter family.
    The unified resolve walks each slot, re-verifies legality at
    resolution per CR 608.2b, and adds ``amount`` +1/+1 counters to
    each surviving target via ``add_plus_one_counters`` — the
    counter-doubling replacement effect on the recipient's
    controller is honored inside the helper, not here.

    Variants with extra rider effects (granted keywords UEOT,
    multi-slot up-to-N, sweeps, bespoke amount) override ``resolve``
    or stay on the old-shape ``Effect`` path.
    """

    amount: int = 1
    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        from mtg_bot.game import add_plus_one_counters

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(
                            state, chosen, action="counter placement",
                        )
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                add_plus_one_counters(state, obj, self.amount)
                state.log.append(
                    f"{obj.printing.name} gets {self.amount} +1/+1 "
                    f"counter(s) (now {obj.plus_one_counters})"
                )
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        if self.amount == 1:
            return f"put a +1/+1 counter on {_describe_targets(self.targets)}"
        return (
            f"put {self.amount} +1/+1 counters on "
            f"{_describe_targets(self.targets)}"
        )


@dataclass(frozen=True)
class Bounce(Effect):
    """Unified return-to-owner's-hand primitive driven by ``targets``.

    Sibling of :class:`Destroy` / :class:`Exile` / :class:`Tap` /
    :class:`Pump` for the bounce family. The unified resolve walks
    each slot, re-verifies legality at resolution per CR 608.2b, and
    bounces each surviving target via :func:`_bounce_to_hand` (which
    handles token cease-to-exist, hand placement, reveal, and the
    permanent-left-battlefield-this-turn bookkeeping for Void).

    Variants with riders that need to snapshot battlefield state
    BEFORE the bounce (e.g., the "if it was attacking, create a
    Drone token" rider on Desculpting Blast) override ``resolve`` so
    they can capture ``obj.attacking`` before ``_bounce_to_hand``
    clears it.
    """

    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(state, chosen, action="bounce")
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                _bounce_to_hand(state, obj)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return f"return {_describe_targets(self.targets)} to its owner's hand"


@dataclass(frozen=True)
class Pump(Effect):
    """Unified pump primitive — target gets ``+power_bonus/+toughness_bonus``
    and gains ``granted_keywords`` until end of turn.

    Sibling of :class:`Destroy` / :class:`Exile` / :class:`Tap` for the
    pump family. The unified resolve walks each slot, re-verifies
    legality at resolution per CR 608.2b, and stamps the bonus +
    keywords onto each surviving target's ``temp_*`` fields (cleared
    at cleanup). The pump amount and granted keywords are uniform
    across slots — multi-slot pumps with per-slot bonuses would need
    a different shape.

    Variants with bespoke amount (Spacecraft-power-X, draw-rider) or
    extra side effects (granted UEOT triggered ability, warp-cost
    rider, draw-if-counter rider) override ``resolve``; the bare
    ``Pump.resolve`` covers the simplest "target X gets +N/+M and
    gains KW UEOT" oracle shape.
    """

    power_bonus: int = 0
    toughness_bonus: int = 0
    granted_keywords: frozenset[str] = frozenset()
    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(state, chosen, action="pump")
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                obj.temp_power_bonus += self.power_bonus
                obj.temp_toughness_bonus += self.toughness_bonus
                if self.granted_keywords:
                    obj.temp_keywords = (
                        obj.temp_keywords | self.granted_keywords
                    )
                kw_note = (
                    f" and gains "
                    f"{', '.join(sorted(self.granted_keywords))}"
                    if self.granted_keywords else ""
                )
                state.log.append(
                    f"{obj.printing.name} gets "
                    f"+{self.power_bonus}/+{self.toughness_bonus}"
                    f"{kw_note} until end of turn"
                )
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        parts: list[str] = []
        if self.power_bonus or self.toughness_bonus:
            parts.append(
                f"{_describe_targets(self.targets)} gets "
                f"+{self.power_bonus}/+{self.toughness_bonus}"
            )
        if self.granted_keywords:
            kw = ", ".join(sorted(self.granted_keywords))
            if parts:
                parts.append(f"gains {kw}")
            else:
                parts.append(
                    f"{_describe_targets(self.targets)} gains {kw}"
                )
        return " and ".join(parts) + " until end of turn."


@dataclass(frozen=True)
class Tap(Effect):
    """Unified tap primitive driven by ``targets``.

    Sibling of :class:`Destroy` / :class:`Exile` for the tap family.
    The unified resolve walks each slot's chosen target, re-verifies
    legality at resolution per CR 608.2b, and taps whichever survive
    the recheck via ``turn.tap_permanent`` (no-op if already tapped).

    Variants that carry extra rider effects (stun counters, "and
    stun it" clauses) override ``resolve`` rather than reusing this
    base — the bare ``Tap.resolve`` covers the simple "tap target X"
    oracle shape.
    """

    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        from mtg_bot.turn import tap_permanent

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(state, chosen, action="tap")
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                tap_permanent(state, obj)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return f"tap {_describe_targets(self.targets)}"


@dataclass(frozen=True)
class Exile(Effect):
    """Unified exile primitive driven by ``targets``.

    Sibling of :class:`Destroy` for the exile family. The unified
    resolve walks each slot's chosen target, re-verifies legality at
    resolution per CR 608.2b, and exiles whichever survive the
    recheck via ``move_off_battlefield(..., destination="exile")``.

    Variants that need extra resolve logic (Mystifying Maze's
    delayed return-trigger, Oblivion-Ring's "until source leaves"
    bookkeeping) override ``resolve`` rather than reusing this base
    — ``Exile.resolve`` covers the bare "exile target X" oracle
    shape, not the until-clause/leave-trigger riders.
    """

    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        from mtg_bot.turn import move_off_battlefield

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(state, chosen, action="exile")
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                state.log.append(f"{obj.printing.name} is exiled")
                move_off_battlefield(state, obj, destination="exile")
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return f"exile {_describe_targets(self.targets)}"


@dataclass(frozen=True)
class Destroy(Effect):
    """Unified destroy primitive driven by ``targets``.

    Replaces the per-class ``legal_targets`` / ``resolve`` overrides
    on the simple destroy variants (``DestroyTargetArtifact``,
    ``DestroyTargetLand``, ``DestroyTargetNoncreaturePermanent``,
    ``DestroyTargetNonlandPermanent``, ``DestroyTargetArtifactOrTappedCreature``,
    ``DestroyTargetArtifactEnchantmentOrFlyingCreature``). The
    ``targets`` field declares legal-target shape via the
    ``TargetFilter`` grammar; the resolve walk re-verifies legality
    at fire time per CR 608.2b and destroys whichever slots survive
    the recheck.

    ``DestroyTargetCreature`` keeps its rider-class shape (Faller's
    Faithful's "if it wasn't dealt damage this turn, draws N") because
    the rider isn't expressible in the bare-grammar model — the rider
    semantically belongs to a sibling effect that pronouns "that
    creature". When the EffectSequence + SlotRef pillar lands the
    rider can move out into a follow-up effect; for now it's a
    ``Destroy`` subclass with extra fields.
    """

    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        from mtg_bot.turn import destroy

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(state, chosen, action="destroy")
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                state.log.append(f"{obj.printing.name} is destroyed")
                destroy(state, obj)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return f"destroy {_describe_targets(self.targets)}"



@dataclass(frozen=True)
class GainLife(Effect):
    amount: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import gain_life

        gain_life(state, controller_id, self.amount)
        p = state.player_by_id(controller_id)
        state.log.append(
            f"P{controller_id} gains {self.amount} life (life: {p.life})"
        )

    def describe(self) -> str:
        return f"you gain {self.amount} life"



@dataclass(frozen=True)
class SetLifeToHalfStartingRoundedUp(Effect):
    """Set the controller's life total to half their starting life, rounded up.

    Matches The Endstone's end-step trigger. In standard MTG,
    starting life = 20 → target becomes 10. Rounds up via
    ``(starting_life + 1) // 2``.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        p = state.player_by_id(controller_id)
        new_life = (p.starting_life + 1) // 2
        state.log.append(
            f"P{controller_id}'s life becomes {new_life} "
            f"(was {p.life}, starting {p.starting_life})"
        )
        p.life = new_life

    def describe(self) -> str:
        return (
            "your life total becomes half your starting life total, "
            "rounded up"
        )



@dataclass(frozen=True)
class TargetPlayerLosesLife(Effect):
    """Make the (pre-selected) target player lose ``amount`` life.

    Reads ``targets[0]`` as ("player", id). Used for triggers that
    identify a specific player via targeting (Entropic Battlecruiser's
    1+ tier "that player loses 3 life").
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, pid = targets[0]
        if kind != "player":
            return
        try:
            p = state.player_by_id(pid)
        except KeyError:
            return
        p.life -= self.amount
        state.log.append(
            f"P{pid} loses {self.amount} life (now {p.life})"
        )

    def describe(self) -> str:
        return f"that player loses {self.amount} life"



@dataclass(frozen=True)
class EachOpponentDiscardsOrLosesLife(Effect):
    """Each opponent discards a card. Each opponent who can't (because
    their hand is empty) loses ``life_if_cant`` life instead.

    Matches Entropic Battlecruiser's attacks trigger. Fires discard
    triggers per-discard via fire_opponent_discards_triggers.
    """

    life_if_cant: int = 3

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone
        from mtg_bot.turn import fire_opponent_discards_triggers

        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            if not p.hand:
                p.life -= self.life_if_cant
                state.log.append(
                    f"P{p.id} can't discard — loses {self.life_if_cant} "
                    f"life (now {p.life})"
                )
                continue
            hand_ids = [c.id for c in p.hand]
            chosen_id = p.agent.choose_discard(state, p.id, hand_ids)
            card = next(c for c in p.hand if c.id == chosen_id)
            p.hand.remove(card)
            card.zone = Zone.GRAVEYARD
            p.graveyard.append(card)
            state.log.append(f"P{p.id} discards {card.printing.name}")
            fire_opponent_discards_triggers(state, p.id)

    def describe(self) -> str:
        return (
            f"each opponent discards a card; each opponent who can't "
            f"loses {self.life_if_cant} life"
        )



@dataclass(frozen=True)
class PutPoisonCountersOnTargetPlayer(Effect):
    """Give ``amount`` poison counters to the (pre-selected) target
    player. Reads ``targets[0]`` as ("player", id); silently no-ops on
    mismatched shapes.

    Matches Virulent Silencer's trigger effect: the observer trigger
    wiring in combat._fire_combat_damage_to_player_triggers passes the
    hit player's id as the target.
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, pid = targets[0]
        if kind != "player":
            return
        try:
            p = state.player_by_id(pid)
        except KeyError:
            return
        p.poison_counters += self.amount
        state.log.append(
            f"P{pid} gets {self.amount} poison counter(s) "
            f"(now {p.poison_counters})"
        )

    def describe(self) -> str:
        return f"that player gets {self.amount} poison counter(s)"



@dataclass(frozen=True)
class LoseLife(Effect):
    """Controller loses ``amount`` life."""

    amount: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        p = state.player_by_id(controller_id)
        p.life -= self.amount
        state.log.append(
            f"P{controller_id} loses {self.amount} life (life: {p.life})"
        )

    def describe(self) -> str:
        return f"you lose {self.amount} life"



@dataclass(frozen=True)
class EachOpponentLosesLife(Effect):
    """Each opponent loses ``amount`` life."""

    amount: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            p.life -= self.amount
            state.log.append(
                f"P{p.id} loses {self.amount} life (life: {p.life})"
            )

    def describe(self) -> str:
        return f"each opponent loses {self.amount} life"



@dataclass(frozen=True)
class DiscardSelf(Effect):
    """Controller discards ``amount`` card(s) from their hand. The discarding
    player's agent picks which cards via ``choose_discard``."""

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone
        from mtg_bot.turn import fire_opponent_discards_triggers

        p = state.player_by_id(controller_id)
        for _ in range(self.amount):
            if not p.hand:
                break
            hand_ids = [c.id for c in p.hand]
            chosen_id = p.agent.choose_discard(state, p.id, hand_ids)
            card = next(c for c in p.hand if c.id == chosen_id)
            p.hand.remove(card)
            card.zone = Zone.GRAVEYARD
            p.graveyard.append(card)
            state.log.append(f"P{p.id} discards {card.printing.name}")
            fire_opponent_discards_triggers(state, p.id)

    def describe(self) -> str:
        if self.amount == 1:
            return "you discard a card"
        return f"you discard {self.amount} cards"



@dataclass(frozen=True)
class ReturnTargetCreatureToOwnersHand(Bounce):
    """``Bounce`` constructor sugar for "return target creature to its
    owner's hand" with optional widening / rider flags.

    Filter composition:
      - default: ``IsCreature()``
      - ``allow_nonland_permanent=True``: ``~IsLand() & ~IsPlayer()``
        (Desculpting Blast — broadens to any nonland permanent)
      - ``optional=True``: slot ``count_min=0`` ("up to one target")

    Rider: ``create_drone_if_was_attacking=True`` snapshots
    ``obj.attacking`` BEFORE the unified ``_bounce_to_hand`` clears
    it, then creates a Drone token after the bounce — needs a custom
    resolve since the snapshot can't happen inside the bare base.
    """

    allow_nonland_permanent: bool = False
    create_drone_if_was_attacking: bool = False
    optional: bool = False

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            IsCreature, IsLand, IsPlayer,
        )
        from mtg_bot.effects.targets import TargetSpec
        if self.allow_nonland_permanent:
            f = ~IsLand() & ~IsPlayer()
        else:
            f = IsCreature()
        spec = TargetSpec(
            f,
            count_min=0 if self.optional else 1,
            count_max=1,
        )
        object.__setattr__(self, "targets", (spec,))

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not self.create_drone_if_was_attacking:
            super().resolve(state, controller_id, targets)
            return
        # Drone-rider variant: snapshot ``obj.attacking`` before the
        # unified bounce clears it, then create the token after the
        # bounce settles.
        if not targets:
            return
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.cards.tokens import TOKEN_PRINTINGS
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        from mtg_bot.game import Zone

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(state, chosen, action="bounce")
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                was_attacking = obj.attacking
                _bounce_to_hand(state, obj)
                if was_attacking:
                    drone_printing = TOKEN_PRINTINGS["Drone"]
                    drone = state.new_object(
                        drone_printing,
                        owner_id=controller_id,
                        zone=Zone.STACK,
                    )
                    put_into_battlefield(state, drone, controller_id)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        scope = "nonland permanent" if self.allow_nonland_permanent else "creature"
        base = f"return target {scope} to its owner's hand"
        if self.create_drone_if_was_attacking:
            base += "; if it was attacking, create a Drone token"
        return base



@dataclass(frozen=True)
class ReturnTargetPermanentYouControlToOwnersHand(Bounce):
    """``Bounce`` constructor sugar for "return target permanent you
    control to its owner's hand" (Exosuit Savior's bounce arm).

    Filter is ``ControlledByYou() & ~IsPlayer()`` — covers any
    battlefield permanent (lands too) the controller controls.
    ``exclude_self=True`` adds ``& ~IsSource()`` for the "other
    target" qualifier. ``optional=True`` flips the slot's
    ``count_min`` to 0 for "up to one target" semantics.
    """

    optional: bool = False
    exclude_self: bool = False

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            ControlledByYou, IsPlayer, IsSource,
        )
        from mtg_bot.effects.targets import TargetSpec
        f = ControlledByYou() & ~IsPlayer()
        if self.exclude_self:
            f = f & ~IsSource()
        spec = TargetSpec(
            f,
            count_min=0 if self.optional else 1,
            count_max=1,
        )
        object.__setattr__(self, "targets", (spec,))

    def describe(self) -> str:
        return "return target permanent you control to its owner's hand"



@dataclass(frozen=True)
class OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary(Effect):
    """Target artifact or creature's owner puts it on their choice of the
    top or bottom of their library.

    Matches Lost in Space's main effect. The OWNER (not the spell's
    controller) makes the top-vs-bottom decision via their agent's
    ``choose_mode`` hook. Tokens cease to exist when bounced (rule
    704.5d) — they'd briefly land in the library, but a non-trivial
    state-based action removes them. Combat / counter state is cleared
    on the way off the battlefield as in other bounce effects.

    ``nonland_permanent`` widens the legal-target pool from "artifact
    or creature" to "any nonland permanent" — Vanish from Sight's
    printed scope. The same engine class is reused with a flag (rather
    than a sibling class) because every line below the legal-target
    filter is identical: same owner-choice top-vs-bottom prompt, same
    bounce-style cleanup, same token-ceases-to-exist handling.
    """

    nonland_permanent: bool = False

    def requires_target(self) -> bool:
        return True

    def _matches(self, printing) -> bool:
        if self.nonland_permanent:
            return not printing.is_land
        return printing.is_artifact or printing.is_creature

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if self._matches(o.printing)
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next((o for o in state.battlefield if o.id == target_id), None)
        if obj is None:
            return
        if not self._matches(obj.printing):
            return
        owner = state.player_by_id(obj.owner_id)
        # Owner picks top vs bottom via their agent's choose_mode.
        # Default to top when no agent is seated (e.g., in unit tests).
        if owner.agent is not None:
            choice = owner.agent.choose_mode(
                state, owner.id,
                ["top of library", "bottom of library"],
            )
        else:
            choice = 0
        choice = 0 if choice not in (0, 1) else choice

        # Strip the permanent off the battlefield with the same
        # housekeeping as other bounce-style effects.
        state.battlefield.remove(obj)
        obj.damage = 0
        obj.marked_lethal = False
        obj.damaged_this_turn = False
        obj.tapped = False
        obj.attacking = False
        obj.blocking = None
        obj.blocked_by = []
        obj.plus_one_counters = 0
        obj.charge_counters = 0
        obj.temp_power_bonus = 0
        obj.temp_toughness_bonus = 0
        obj.temp_keywords = frozenset()
        obj.attached_to = None
        if not obj.printing.is_land:
            state.permanent_left_battlefield_this_turn = True
        if obj.printing.is_token:
            state.log.append(
                f"{obj.printing.name} token ceases to exist (bounced)"
            )
            return
        obj.zone = Zone.LIBRARY
        if choice == 0:
            owner.library.insert(0, obj)
            destination = "top"
        else:
            owner.library.append(obj)
            destination = "bottom"
        state.log.append(
            f"{obj.printing.name} put on the {destination} of "
            f"P{owner.id}'s library"
        )

    def describe(self) -> str:
        scope = (
            "nonland permanent"
            if self.nonland_permanent
            else "artifact or creature"
        )
        return (
            f"target {scope}'s owner puts it on their choice of the "
            f"top or bottom of their library"
        )



@dataclass(frozen=True)
class OwnerPutsTargetWarpedExileCardOnBottomOfLibrary(Effect):
    """Put target exiled card with warp on the bottom of its owner's library.

    Matches mode 1 of Blade of the Swarm. Target pool is every card in
    any player's exile zone whose ``warped`` flag is set (the flag is
    raised when a warp-cast creature is exiled at the end step). The
    chosen card is moved to the bottom of its owner's library and its
    warp-related state is cleared (it's now a normal library card).

    Sibling of ``OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary``,
    but without the top/bottom owner choice (always bottom) and sourcing
    from exile rather than the battlefield. Targets carry kind
    ``"exile_card"`` — distinct from "permanent" because the card isn't
    on the battlefield and the cast pipeline shouldn't try to look it up
    there.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        out: list[Target] = []
        for p in state.players:
            for c in p.exile:
                if c.warped:
                    out.append(("exile_card", c.id))
        return out

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "exile_card":
            return
        # Locate the target across every player's exile.
        obj = None
        owner = None
        for p in state.players:
            for c in p.exile:
                if c.id == target_id:
                    obj = c
                    owner = p
                    break
            if obj is not None:
                break
        if obj is None or owner is None:
            return
        # Re-check the warped flag at resolution — if something stripped
        # it (or the card moved zones in response), the spell fizzles.
        if not obj.warped:
            return
        owner.exile.remove(obj)
        obj.warped = False
        obj.zone = Zone.LIBRARY
        owner.library.append(obj)
        state.log.append(
            f"{obj.printing.name} put on the bottom of P{owner.id}'s "
            f"library (was warped in exile)"
        )

    def describe(self) -> str:
        return (
            "put target exiled card with warp on the bottom of "
            "its owner's library"
        )



@dataclass(frozen=True)
class AttachToTargetCreatureYouControl(Effect):
    """Attach the source (Equipment) to target creature the controller
    controls. Matches the common ETB trigger on EoE Equipment.

    Reads state.resolving_source for the Equipment. No-op if the source
    isn't an Equipment (is_equipment check on subtypes).

    Optional ``granted_keywords_ueot``: keywords the chosen target also
    gains until end of turn. Matches oracle riders like Squire's
    Lightblade's "...attach it to target creature you control. That
    creature gains first strike until end of turn." and Illvoi Light
    Jammer's "...gains hexproof until end of turn." Adding the keyword
    here (rather than via a separate Sequence sub-effect) keeps the
    target shared between attach and the UEOT grant. The grant is
    applied after the attach succeeds; if the attach fizzles, no
    keywords are granted.
    """

    granted_keywords_ueot: frozenset[str] = frozenset()

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature and o.controller_id == controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        if "Equipment" not in source.printing.subtypes:
            return
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        target = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if target is None or not target.printing.is_creature:
            return
        if target.controller_id != controller_id:
            return
        source.attached_to = target_id
        state.log.append(
            f"{source.printing.name} is attached to {target.printing.name}"
        )
        if self.granted_keywords_ueot:
            state.grant_keywords_ueot(
                source_id=source.id,
                target_id=target.id,
                keywords=self.granted_keywords_ueot,
                log_source_name=source.printing.name,
                log_target_name=target.printing.name,
            )
            state.log.append(
                f"{target.printing.name} gains "
                f"{', '.join(sorted(self.granted_keywords_ueot))} "
                "until end of turn"
            )

    def describe(self) -> str:
        base = "attach this Equipment to target creature you control"
        if self.granted_keywords_ueot:
            return (
                f"{base}. That creature gains "
                f"{', '.join(sorted(self.granted_keywords_ueot))} "
                "until end of turn"
            )
        return base



@dataclass(frozen=True)
class EquipToTargetCreatureYouControl(AttachToTargetCreatureYouControl):
    """Equip activated-ability resolution effect (rule 702.6).

    Behaves identically to ``AttachToTargetCreatureYouControl`` (re-targets
    the source Equipment onto a creature the controller controls, replacing
    any prior attachment), but is given a distinct name and ``describe``
    string so logs and tooling can tell ETB-attach from a paid Equip
    activation. Pair with an ``ActivatedAbility`` whose cost is the printed
    Equip mana cost and ``sorcery_speed=True``.
    """

    def describe(self) -> str:
        return "equip: attach this Equipment to target creature you control"



@dataclass(frozen=True)
class ReturnUpToNTargetCreaturesToOwnersHands(Effect):
    """Return up to ``max_n`` target creatures to their owners' hands.

    Matches Specimen Freighter's "return up to two target non-Spacecraft
    creatures to their owners' hands". Set ``exclude_spacecraft=True`` to
    enforce the non-Spacecraft filter at target-selection time.
    """

    max_n: int = 2
    exclude_spacecraft: bool = False

    def requires_target(self) -> bool:
        return True

    def target_count_range(self) -> tuple[int, int]:
        return (0, self.max_n)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and (not self.exclude_spacecraft or not o.printing.is_spacecraft)
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        for kind, target_id in targets:
            if kind != "creature":
                continue
            obj = next(
                (o for o in state.battlefield if o.id == target_id), None
            )
            if obj is None:
                continue
            state.battlefield.remove(obj)
            obj.damage = 0
            obj.marked_lethal = False
            obj.damaged_this_turn = False
            obj.tapped = False
            obj.attacking = False
            obj.blocking = None
            obj.blocked_by = []
            obj.plus_one_counters = 0
            obj.charge_counters = 0
            obj.temp_power_bonus = 0
            obj.temp_toughness_bonus = 0
            obj.temp_keywords = frozenset()
            obj.attached_to = None
            if not obj.printing.is_land:
                state.permanent_left_battlefield_this_turn = True
            if obj.printing.is_token:
                state.log.append(
                    f"{obj.printing.name} token ceases to exist (bounced)"
                )
                continue
            from mtg_bot.game import reveal_to_all
            obj.zone = Zone.HAND
            state.player_by_id(obj.owner_id).hand.append(obj)
            reveal_to_all(state, obj)
            state.log.append(
                f"{obj.printing.name} returns to P{obj.owner_id}'s hand"
            )

    def describe(self) -> str:
        return (
            f"return up to {self.max_n} target creatures to their "
            f"owners' hands"
        )



@dataclass(frozen=True)
class ReturnUpToNSmallCreaturesFromGraveyardWithCounters(Effect):
    """Return up to ``max_n`` creature cards with **total mana value**
    ≤ ``max_total_mv`` from your graveyard to the battlefield. Put a
    +1/+1 counter on each.

    Matches Scout for Survivors. The constraint is on the sum of
    chosen targets' mana values, not on each individually — the
    cleanest place to enforce that is at resolution time, since the
    cast pipeline picks targets sequentially without a running budget.
    Single-target legality still filters to MV ≤ max_total_mv (a card
    bigger than the budget can never appear in any legal multi-pick).
    """

    max_n: int = 3
    max_total_mv: int = 3

    def requires_target(self) -> bool:
        return True

    def target_count_range(self) -> tuple[int, int]:
        return (0, self.max_n)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        p = state.player_by_id(controller_id)
        return [
            ("graveyard_creature", c.id)
            for c in p.graveyard
            if c.printing.is_creature
            and c.printing.mana_cost.total <= self.max_total_mv
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield

        p = state.player_by_id(controller_id)
        # Enforce the SUM-of-MV budget at resolve time. Walk picks in
        # pick order; admit each only if it fits the remaining budget,
        # else drop it. Targets that already left the graveyard are
        # silently skipped (standard targeting rule).
        remaining = self.max_total_mv
        for kind, target_id in targets:
            if kind != "graveyard_creature":
                continue
            card = next((c for c in p.graveyard if c.id == target_id), None)
            if card is None:
                continue
            mv = card.printing.mana_cost.total
            if mv > remaining:
                state.log.append(
                    f"{card.printing.name} skipped — exceeds total-MV "
                    f"budget (remaining {remaining}, needs {mv})"
                )
                continue
            remaining -= mv
            p.graveyard.remove(card)
            state.log.append(
                f"{card.printing.name} returns to the battlefield under "
                f"P{controller_id} with a +1/+1 counter"
            )
            put_into_battlefield(state, card, controller_id)
            from mtg_bot.game import add_plus_one_counters
            add_plus_one_counters(state, card, 1)

    def describe(self) -> str:
        return (
            f"return up to {self.max_n} target creature cards with total "
            f"mana value {self.max_total_mv} or less from your graveyard "
            f"to the battlefield with a +1/+1 counter on each"
        )



@dataclass(frozen=True)
class CreateLanderAndScheduleSacrifice(Effect):
    """Create a Lander token for the controller, then schedule a delayed
    trigger to sacrifice that specific Lander at the controller's next
    end step.

    Matches Kav Landseeker's ETB. "Next end step on your next turn" is
    interpreted in a 2-player game as the controller's next end step
    that isn't the current turn — turn_number+2 if it's currently the
    controller's turn, else turn_number+1.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.cards import TOKEN_PRINTINGS
        from mtg_bot.game import Zone

        lander = state.new_object(
            TOKEN_PRINTINGS["Lander"],
            owner_id=controller_id,
            zone=Zone.BATTLEFIELD,
        )
        put_into_battlefield(state, lander, controller_id)
        offset = 2 if state.active_player().id == controller_id else 1
        state.delayed_triggers.append({
            "fire_at_turn": state.turn_number + offset,
            "controller_id": controller_id,
            "source_id": lander.id,
            "effect": SacrificeSourceDelayed(),
        })
        state.log.append(
            f"P{controller_id} creates a Lander token (scheduled to be "
            f"sacrificed at end step of turn "
            f"{state.turn_number + offset})"
        )

    def describe(self) -> str:
        return (
            "create a Lander token; at the beginning of the end step on "
            "your next turn, sacrifice that token"
        )



@dataclass(frozen=True)
class CreateTokenAndAttachSelf(Effect):
    """Create a token and attach the source (Equipment) to it.

    Matches Auxiliary Boosters' ETB. Reads state.resolving_source for
    the Equipment. No-op if source isn't Equipment.
    """

    token_printing: "CardPrinting"

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.game import Zone

        source = state.resolving_source
        if source is None:
            return
        if "Equipment" not in source.printing.subtypes:
            return
        token = state.new_object(
            self.token_printing, owner_id=controller_id,
            zone=Zone.BATTLEFIELD,
        )
        state.log.append(
            f"P{controller_id} creates a {self.token_printing.name} token"
        )
        put_into_battlefield(state, token, controller_id)
        source.attached_to = token.id
        state.log.append(
            f"{source.printing.name} is attached to {token.printing.name}"
        )

    def describe(self) -> str:
        return (
            f"create a {self.token_printing.name} token and attach this "
            f"Equipment to it"
        )



@dataclass(frozen=True)
class ReturnCreatureFromYourGraveyardToBattlefield(Effect):
    """Target creature card from your graveyard enters the battlefield under
    your control. Fires ETB triggers (via put_into_battlefield).

    Target format: ``("graveyard_creature", id)`` where id is a GameObject
    in the controller's graveyard.

    ``max_mv``: when set, restricts the legal target pool to creature
    cards with mana value ≤ ``max_mv``. Models Surgical Suite's "with
    mana value 3 or less" rider. None = no mana-value restriction.

    ``or_enchantment`` widens the legal-target pool to also include
    enchantment cards in the graveyard. Matches oracle text "return
    target creature or enchantment card from your graveyard to the
    battlefield" — Rescue Skiff's ETB body. Mirrors the ``or_spacecraft``
    flag on ``ReturnCreatureFromYourGraveyardToHand`` (Fell Gravship).
    """

    max_mv: "int | None" = None
    # When True, enchantment cards in the graveyard are also legal
    # targets. Matches "return target creature or enchantment card from
    # your graveyard to the battlefield" — Rescue Skiff.
    or_enchantment: bool = False

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        p = state.player_by_id(controller_id)
        return [
            ("graveyard_creature", c.id)
            for c in p.graveyard
            if (
                c.printing.is_creature
                or (self.or_enchantment and c.printing.is_enchantment)
            )
            and (
                self.max_mv is None
                or c.printing.mana_cost.total <= self.max_mv
            )
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "graveyard_creature":
            return
        p = state.player_by_id(controller_id)
        card = next((c for c in p.graveyard if c.id == target_id), None)
        if card is None:
            return
        p.graveyard.remove(card)
        state.log.append(
            f"{card.printing.name} returns from P{controller_id}'s "
            f"graveyard to the battlefield"
        )
        put_into_battlefield(state, card, controller_id)

    def describe(self) -> str:
        type_phrase = (
            "creature or enchantment" if self.or_enchantment else "creature"
        )
        return (
            f"return target {type_phrase} card from your graveyard to the "
            f"battlefield"
        )



@dataclass(frozen=True)
class ReturnExiledPermanentToOwnerTapped(Effect):
    """Return a specific exiled card (``exile_card_id``) to the
    battlefield under its owner's control, tapped.

    Unlike the targeted-by-agent return effects, this one stores the
    exact card id at construction time. Intended for scheduled delayed
    triggers where the card was already selected at schedule time (see
    ExileTargetAttackingCreatureUntilEndStep / Mystifying Maze).
    """

    exile_card_id: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield

        owner = None
        card = None
        for p in state.players:
            for c in p.exile:
                if c.id == self.exile_card_id:
                    owner = p
                    card = c
                    break
            if card is not None:
                break
        if card is None or owner is None:
            return
        owner.exile.remove(card)
        put_into_battlefield(state, card, owner.id)
        card.tapped = True
        state.log.append(
            f"{card.printing.name} returns from exile to P{owner.id}'s "
            f"battlefield tapped"
        )

    def describe(self) -> str:
        return (
            f"return exiled card #{self.exile_card_id} to its owner's "
            f"battlefield tapped"
        )



@dataclass(frozen=True)
class WarpExileSelf(Effect):
    """Delayed-triggered exile of a warp-cast creature at end step.

    Scheduled at warp-cast time onto ``state.delayed_triggers`` with
    ``source_id`` set to the warped card's id. When the delayed-trigger
    machinery converts the entry into a real triggered ability and
    pushes it onto the stack, the StackItem's ``source`` is the warp
    permanent (or stays as the original GameObject reference even if
    the permanent has since left). On resolve, ``state.resolving_source``
    is that GameObject — but we re-check the battlefield by id before
    acting because the card may have left between the trigger going on
    the stack and resolving (countered, died, bounced, exiled).

    No-op if the source has left the battlefield — the trigger still
    "fires" per CR 603.6c, it just has no permanent to act on.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import move_off_battlefield

        # ``state.resolving_source`` is set to the StackItem's source
        # GameObject — that reference persists even if the warp creature
        # has since left the battlefield. Re-look up by id on the live
        # battlefield to honor the no-op-if-gone rule (CR 603.6c).
        ref = state.resolving_source
        if ref is None:
            return
        obj = next(
            (o for o in state.battlefield if o.id == ref.id), None,
        )
        if obj is None:
            return
        state.log.append(f"{obj.printing.name} is warp-exiled")
        move_off_battlefield(state, obj, destination="exile")
        if not obj.printing.is_token:
            obj.warped = True
        obj.cast_via_warp = False

    def describe(self) -> str:
        return "exile this warp-cast permanent"



@dataclass(frozen=True)
class SacrificeSourceDelayed(Effect):
    """Delayed-triggered "sacrifice this permanent" effect.

    Scheduled with ``source_id`` set to the permanent that should be
    sacrificed. Used by Kav Landseeker (sacrifice the Lander token at
    P0's next end step) and Devastating Onslaught (sacrifice the haste-
    token copies at the next end step).

    On resolve, look up the source on the battlefield by id and
    sacrifice it. No-op if it's already gone (CR 603.6c) — the trigger
    still fires, it just has nothing to act on.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import sacrifice

        ref = state.resolving_source
        if ref is None:
            return
        obj = next(
            (o for o in state.battlefield if o.id == ref.id), None,
        )
        if obj is None:
            return
        state.log.append(
            f"P{controller_id} sacrifices {obj.printing.name} "
            f"(delayed trigger)"
        )
        sacrifice(state, obj)

    def describe(self) -> str:
        return "sacrifice this permanent"



@dataclass(frozen=True)
class RemoveChargeCountersFromSourceDelayed(Effect):
    """Delayed-triggered "remove N charge counters from this permanent"
    effect.

    Scheduled with ``source_id`` set to the Spacecraft to be reset.
    Used by Systems Override (add 10 charge counters now, remove 10
    at the next end step).

    On resolve, look up the source on the battlefield by id and
    decrement charge counters (clamped at 0). No-op if the source has
    left the battlefield.
    """

    amount: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        ref = state.resolving_source
        if ref is None:
            return
        obj = next(
            (o for o in state.battlefield if o.id == ref.id), None,
        )
        if obj is None:
            return
        obj.charge_counters = max(0, obj.charge_counters - self.amount)
        state.log.append(
            f"remove {self.amount} charge counters from "
            f"{obj.printing.name} (now {obj.charge_counters})"
        )

    def describe(self) -> str:
        return (
            f"remove {self.amount} charge counters from this permanent"
        )



@dataclass(frozen=True)
class ExileTargetAttackingCreatureUntilEndStep(Effect):
    """Mystifying Maze: "Exile target attacking creature an opponent
    controls. At the beginning of the next end step, that player
    returns it to the battlefield under its owner's control tapped."

    On resolve: exiles the chosen creature (without the
    Oblivion-Ring-style ``exiled_by_source`` marker so the creature
    stays in exile even if Mystifying Maze later leaves the
    battlefield), then schedules a delayed end-step trigger that
    resolves a ReturnExiledPermanentToOwnerTapped for that card.

    The delayed trigger's ``controller_id`` is the CURRENT active
    player — Maze's activation is instant-speed on an opponent's
    combat, so "next end step" is the active player's own end step.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and o.attacking
            and o.controller_id != controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import move_off_battlefield

        if not targets:
            return
        _, tid = targets[0]
        target = next(
            (o for o in state.battlefield if o.id == tid), None,
        )
        if target is None:
            return
        # The Mystifying Maze itself is the source of the delayed
        # triggered ability: per CR 603.7c the ability is created by
        # Maze's resolution and goes on the stack at the next end step
        # under Maze's controller (the active player). We need a
        # source GameObject for the StackItem; ``state.resolving_source``
        # is Maze during this resolve.
        maze_source_id = (
            state.resolving_source.id
            if state.resolving_source is not None else None
        )
        state.log.append(
            f"{target.printing.name} is exiled until end of turn"
        )
        move_off_battlefield(state, target, destination="exile")
        entry: dict = {
            "fire_at_turn": state.turn_number,
            "controller_id": state.active_player().id,
            "effect": ReturnExiledPermanentToOwnerTapped(
                exile_card_id=target.id,
            ),
        }
        if maze_source_id is not None:
            entry["source_id"] = maze_source_id
        state.delayed_triggers.append(entry)

    def describe(self) -> str:
        return (
            "exile target attacking creature an opponent controls; its "
            "controller returns it to the battlefield under its "
            "owner's control tapped at the beginning of the next end "
            "step"
        )



@dataclass(frozen=True)
class ReturnTargetArtifactCardFromYourGraveyardToHand(Effect):
    """Target artifact card from your graveyard goes to your hand.

    Matches Scour for Scrap's second mode. Uses a dedicated target kind
    ``graveyard_artifact`` so the multi-zone target pool is distinct
    from battlefield targets.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        p = state.player_by_id(controller_id)
        return [
            ("graveyard_artifact", c.id)
            for c in p.graveyard
            if c.printing.is_artifact
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "graveyard_artifact":
            return
        p = state.player_by_id(controller_id)
        card = next((c for c in p.graveyard if c.id == target_id), None)
        if card is None:
            return
        from mtg_bot.game import reveal_to_all
        p.graveyard.remove(card)
        card.zone = Zone.HAND
        p.hand.append(card)
        reveal_to_all(state, card)
        state.log.append(
            f"{card.printing.name} returns from P{controller_id}'s "
            f"graveyard to hand"
        )

    def describe(self) -> str:
        return "return target artifact card from your graveyard to your hand"



@dataclass(frozen=True)
class ReturnCreatureFromYourGraveyardToHand(Effect):
    """Target creature (or artifact/Spacecraft) card from your graveyard
    goes to your hand.

    ``max_mv`` (default 0 = "no cap") narrows the legal-target pool to
    cards whose printed mana value is ≤ the threshold.

    ``or_spacecraft`` widens the legal-target pool to also include
    Spacecraft cards in the graveyard. Matches "return target creature
    or Spacecraft card" — Fell Gravship.

    ``or_artifact`` widens the pool to also include artifact cards.
    Matches Interceptor Mechan's "return target artifact or creature
    card from your graveyard to your hand."
    """

    max_mv: int = 0
    or_spacecraft: bool = False
    or_artifact: bool = False

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        p = state.player_by_id(controller_id)
        return [
            ("graveyard_creature", c.id)
            for c in p.graveyard
            if (
                c.printing.is_creature
                or (self.or_spacecraft and c.printing.is_spacecraft)
                or (self.or_artifact and c.printing.is_artifact)
            )
            and (self.max_mv == 0 or c.printing.mana_cost.total <= self.max_mv)
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "graveyard_creature":
            return
        p = state.player_by_id(controller_id)
        card = next((c for c in p.graveyard if c.id == target_id), None)
        if card is None:
            return
        from mtg_bot.game import reveal_to_all
        p.graveyard.remove(card)
        card.zone = Zone.HAND
        p.hand.append(card)
        reveal_to_all(state, card)
        state.log.append(
            f"{card.printing.name} returns from P{controller_id}'s "
            f"graveyard to hand"
        )

    def describe(self) -> str:
        extras = []
        if self.or_artifact:
            extras.append("artifact")
        if self.or_spacecraft:
            extras.append("Spacecraft")
        if extras:
            type_phrase = "artifact or creature" if "artifact" in extras else "creature or Spacecraft"
        else:
            type_phrase = "creature"
        if self.max_mv > 0:
            return (
                f"return target {type_phrase} card with mana value "
                f"{self.max_mv} or less from your graveyard to your hand"
            )
        return (
            f"return target {type_phrase} card from your graveyard "
            f"to your hand"
        )



@dataclass(frozen=True)
class ReturnUpToNCreaturesFromYourGraveyardToHand(Effect):
    """Return up to ``max_n`` creature cards from your graveyard to your hand.

    Models the "return one or two target creature [or planeswalker] cards
    from your graveyard to your hand" mode of Archenemy's Charm. Since
    planeswalkers aren't yet modeled in our card pool, the legal-target
    pool is creature-only; if/when planeswalker cards land in graveyards,
    they'd surface here through the same code path automatically.
    """

    max_n: int = 2
    min_n: int = 1

    def requires_target(self) -> bool:
        return True

    def target_count_range(self) -> tuple[int, int]:
        return (self.min_n, self.max_n)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        p = state.player_by_id(controller_id)
        return [
            ("graveyard_creature", c.id)
            for c in p.graveyard
            if c.printing.is_creature
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone, reveal_to_all

        if not targets:
            return
        p = state.player_by_id(controller_id)
        seen_ids: set[int] = set()
        for kind, target_id in targets:
            if kind != "graveyard_creature":
                continue
            if target_id in seen_ids:
                continue
            seen_ids.add(target_id)
            card = next((c for c in p.graveyard if c.id == target_id), None)
            if card is None:
                continue
            p.graveyard.remove(card)
            card.zone = Zone.HAND
            p.hand.append(card)
            reveal_to_all(state, card)
            state.log.append(
                f"{card.printing.name} returns from P{controller_id}'s "
                f"graveyard to hand"
            )

    def describe(self) -> str:
        return (
            f"return up to {self.max_n} target creature cards from your "
            f"graveyard to your hand"
        )



@dataclass(frozen=True)
class ReturnAllCreaturesFromYourGraveyardToBattlefield(Effect):
    """Return every creature card from your graveyard to the battlefield
    under your control. Untargeted mass-reanimation — Awakening Hall's
    "Return all creature cards from your graveyard to the battlefield"
    when its door is unlocked.

    Each creature is moved out of the graveyard and routed through
    ``put_into_battlefield`` so its ETB triggers fire normally. Order
    of resolution follows the graveyard's stored order (oldest first);
    we don't model APNAP for the iteration since the controller is the
    same player throughout.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield

        p = state.player_by_id(controller_id)
        # Snapshot the graveyard before mutation — entering creatures
        # could (in theory) trigger other effects that touch the
        # graveyard mid-resolution. We iterate the snapshot.
        creatures = [c for c in list(p.graveyard) if c.printing.is_creature]
        for card in creatures:
            if card not in p.graveyard:
                # Defensive: an ETB earlier in this loop already moved
                # the card (very rare; included for safety).
                continue
            p.graveyard.remove(card)
            state.log.append(
                f"{card.printing.name} returns from P{controller_id}'s "
                f"graveyard to the battlefield"
            )
            put_into_battlefield(state, card, controller_id)

    def describe(self) -> str:
        return (
            "return all creature cards from your graveyard to the "
            "battlefield"
        )


@dataclass(frozen=True)
class ExileEachOpponentsGraveyard(Effect):
    """Move every card in each opponent's graveyard to that player's exile."""

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        total = 0
        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            cards_to_exile = list(p.graveyard)
            for card in cards_to_exile:
                p.graveyard.remove(card)
                card.zone = Zone.EXILE
                p.exile.append(card)
                total += 1
        state.log.append(
            f"P{controller_id} exiles {total} card(s) from opponent graveyards"
        )

    def describe(self) -> str:
        return "exile each opponent's graveyard"



@dataclass(frozen=True)
class DealDamageToTargetOpponent(DealDamage):
    """``DealDamage`` constructor sugar for "deals N damage to target
    opponent".

    Filter is ``IsPlayer() & ControlledByOpponent()``; the
    ``ControlledByOpponent`` arm makes the spell's caster (the
    "you" in MTG parlance) ineligible.

    NB: ``amount`` defaults to 1 to mirror the legacy class's default;
    most call sites pass an explicit value but a few constructed it
    bare.
    """

    amount: int = 1

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            ControlledByOpponent, IsPlayer,
        )
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsPlayer() & ControlledByOpponent()),),
        )



@dataclass(frozen=True)
class TargetOpponentLosesNYouGainN(Effect):
    """Target opponent loses ``amount`` life; the ability's controller
    gains ``amount`` life. Matches Susurian Voidborn's
    your_creature_or_artifact_dies trigger.
    """

    amount: int = 1

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id) for p in state.players
            if p.id != controller_id and not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, pid = targets[0]
        if kind != "player":
            return
        try:
            opp = state.player_by_id(pid)
        except KeyError:
            return
        if opp.id == controller_id:
            return
        opp.life -= self.amount
        from mtg_bot.game import gain_life
        gain_life(state, controller_id, self.amount)
        you = state.player_by_id(controller_id)
        state.log.append(
            f"P{pid} loses {self.amount} life (now {opp.life}); "
            f"P{controller_id} gains {self.amount} life (now {you.life})"
        )

    def describe(self) -> str:
        return (
            f"target opponent loses {self.amount} life and you gain "
            f"{self.amount} life"
        )



@dataclass(frozen=True)
class TargetOpponentLosesLife(Effect):
    """Target opponent loses ``amount`` life.

    Matches oracle text like "Target opponent loses 3 life" — the second
    activated ability on Dubious Delicacy ({2}, {T}, Sacrifice this
    artifact: Target opponent loses 3 life). The legality pool is
    explicitly opponents-only, so the controller can never target
    themselves.
    """

    amount: int = 1

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id) for p in state.players
            if p.id != controller_id and not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, pid = targets[0]
        if kind != "player":
            return
        try:
            opp = state.player_by_id(pid)
        except KeyError:
            return
        if opp.id == controller_id:
            return
        opp.life -= self.amount
        state.log.append(
            f"P{pid} loses {self.amount} life (now {opp.life})"
        )

    def describe(self) -> str:
        return f"target opponent loses {self.amount} life"



@dataclass(frozen=True)
class PumpTargetOtherCreatureByXOverXWhereXIsSourcePower(Effect):
    """Target another creature you control gets +X/+X until end of turn,
    where X is the effect's source's effective power. Reads
    ``state.resolving_source``; target must be under the same
    controller and not be the source itself.

    Matches Hemosymbic Mite's becomes-tapped trigger.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        source = state.resolving_source
        source_id = source.id if source is not None else -1
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and o.controller_id == controller_id
            and o.id != source_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import effective_power

        source = state.resolving_source
        if source is None:
            return
        x = effective_power(state, source)
        if x <= 0:
            return
        if not targets:
            return
        kind, tid = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == tid), None
        )
        if obj is None:
            return
        state.grant_pt_pump_ueot(
            source_id=source.id, target_id=obj.id,
            power=x, toughness=x,
            description=(
                f"{source.printing.name} pumps {obj.printing.name} "
                f"+{x}/+{x} UEOT (source power)"
            ),
        )
        state.log.append(
            f"{obj.printing.name} gets +{x}/+{x} UEOT (source power)"
        )

    def describe(self) -> str:
        return (
            "another target creature you control gets +X/+X until end "
            "of turn, where X is this creature's power"
        )



@dataclass(frozen=True)
class DoubleTargetCreaturesPowerUEOT(Effect):
    """Target creature you control has its effective power doubled
    until end of turn. Implemented by adding the target's current
    effective power as a temp_power_bonus.

    Matches Mightform Harmonizer's landfall trigger. Simplification:
    the bonus is a one-time snapshot at resolution — further counter
    gains or other pumps don't re-double within the same turn.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and o.controller_id == controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import effective_power

        if not targets:
            return
        kind, tid = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == tid), None
        )
        if obj is None:
            return
        snapshot = effective_power(state, obj)
        if snapshot <= 0:
            return
        src = state.resolving_source
        state.grant_pt_pump_ueot(
            source_id=src.id if src is not None else obj.id,
            target_id=obj.id,
            power=snapshot, toughness=0,
            description=(
                f"{obj.printing.name}'s power doubled UEOT "
                f"(+{snapshot})"
            ),
        )
        state.log.append(
            f"{obj.printing.name}'s power is doubled UEOT "
            f"(+{snapshot} temp power)"
        )

    def describe(self) -> str:
        return "double the power of target creature you control until end of turn"



@dataclass(frozen=True)
class DiscardHandThenDrawN(Effect):
    """"You may discard your hand. If you do, draw ``amount`` cards."

    Matches Vaultguard Trooper's conditional end-step ability. Acts
    on the CONTROLLER's hand. The "may" is exposed via
    ``agent.choose_yes_no`` — if the controller declines, the hand is
    untouched and nothing is drawn (the dependent "If you do" rider
    does not fire).
    """

    amount: int = 2

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        p = state.player_by_id(controller_id)
        # Optional "may discard your hand" — controller decides.
        if p.agent is not None and not p.agent.choose_yes_no(
            state, controller_id,
            f"Discard your hand to draw {self.amount}?",
        ):
            state.log.append(
                f"P{controller_id} declines to discard hand "
                f"(DiscardHandThenDrawN)"
            )
            return
        # Discard every card.
        discarded_count = 0
        while p.hand:
            card = p.hand.pop(0)
            card.zone = Zone.GRAVEYARD
            p.graveyard.append(card)
            discarded_count += 1
        state.log.append(
            f"P{controller_id} discards hand ({discarded_count} cards)"
        )
        # Draw N cards.
        drew = 0
        for _ in range(self.amount):
            if not p.library:
                break
            card = p.library.pop(0)
            card.zone = Zone.HAND
            p.hand.append(card)
            drew += 1
        state.log.append(
            f"P{controller_id} draws {drew} card(s)"
        )

    def describe(self) -> str:
        return f"discard your hand, then draw {self.amount} cards"



@dataclass(frozen=True)
class PutPlusOneCountersOnEachCreatureYouControlEqualToSourcePower(Effect):
    """Put X +1/+1 counters on each creature you control, where X is
    this effect's source's effective power. Reads
    ``state.resolving_source`` for the power. No-op if source is None
    or X <= 0.

    Matches Ouroboroid's beginning-of-combat trigger.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import add_plus_one_counters, effective_power

        source = state.resolving_source
        if source is None:
            return
        x = effective_power(state, source)
        if x <= 0:
            return
        count = 0
        for obj in state.battlefield:
            if obj.controller_id != controller_id:
                continue
            if not obj.printing.is_creature:
                continue
            add_plus_one_counters(state, obj, x)
            count += 1
        state.log.append(
            f"P{controller_id}'s {count} creature(s) each get {x} "
            f"+1/+1 counter(s) (source power)"
        )

    def describe(self) -> str:
        return (
            "put X +1/+1 counters on each creature you control, where X "
            "is this creature's power"
        )



@dataclass(frozen=True)
class PutCardFromGraveyardOnBottomOfLibrary(Effect):
    """Put target card from a graveyard on the bottom of its owner's
    library. Matches Chrome Companion's activated ability. Any
    graveyard is a legal target source (either player's).
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        from mtg_bot.game import Zone

        return [
            ("card_in_graveyard", c.id)
            for p in state.players
            for c in p.graveyard
            if c.zone == Zone.GRAVEYARD
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "card_in_graveyard":
            return
        for p in state.players:
            card = next((c for c in p.graveyard if c.id == target_id), None)
            if card is None:
                continue
            p.graveyard.remove(card)
            card.zone = Zone.LIBRARY
            p.library.append(card)  # bottom = end of list
            state.log.append(
                f"{card.printing.name} put on the bottom of "
                f"P{p.id}'s library"
            )
            return

    def describe(self) -> str:
        return (
            "put target card from a graveyard on the bottom of its "
            "owner's library"
        )



@dataclass(frozen=True)
class ExileTargetCardFromGraveyard(Effect):
    """Exile target card from a graveyard. Either player's graveyard is a
    legal target source — the chosen card moves to its owner's exile zone
    and (when the source supports the linkage) carries
    ``exiled_by_source`` so a sibling activated ability can recall every
    card the same source exiled later. DSK Ghost Vacuum's
    ``{T}: Exile target card from a graveyard.`` is the introducing
    printing.

    Mirrors ``PutCardFromGraveyardOnBottomOfLibrary``'s
    ``("card_in_graveyard", id)`` target-kind shape — the same target
    kind the existing graveyard-bottom effect emits, so the targeting
    pipeline doesn't need a new kind for this primitive.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        from mtg_bot.game import Zone

        return [
            ("card_in_graveyard", c.id)
            for p in state.players
            for c in p.graveyard
            if c.zone == Zone.GRAVEYARD
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "card_in_graveyard":
            return
        source = state.resolving_source
        for p in state.players:
            card = next((c for c in p.graveyard if c.id == target_id), None)
            if card is None:
                continue
            p.graveyard.remove(card)
            card.zone = Zone.EXILE
            # Tag the exile site so a sibling activated ability on the
            # same source (Ghost Vacuum's ``{6}, {T}, Sacrifice this
            # artifact`` recall) can find every card the same source
            # exiled later. The link is harmless when the source has no
            # such recall ability; ``move_off_battlefield`` clears the
            # tag on any release path that returns the card from exile.
            if source is not None:
                card.exiled_by_source = source.id
            p.exile.append(card)
            state.log.append(
                f"{card.printing.name} exiled from P{p.id}'s graveyard"
            )
            return

    def describe(self) -> str:
        return "exile target card from a graveyard"



@dataclass(frozen=True)
class PutPlusOneCounterOnEachOfUpToNTargetCreatures(Effect):
    """Put a +1/+1 counter on each of up to ``max_n`` target creatures.

    Matches "put a +1/+1 counter on each of up to two target creatures"
    templating (Weftblade Enhancer). Targets are selected by the agent
    via repeated choose_target calls; effect resolves for each valid
    target that's still a creature on the battlefield.
    """

    max_n: int = 2

    def requires_target(self) -> bool:
        return True

    def target_count_range(self) -> tuple[int, int]:
        return (0, self.max_n)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import add_plus_one_counters

        for kind, target_id in targets:
            if kind != "creature":
                continue
            tgt = next(
                (o for o in state.battlefield if o.id == target_id), None
            )
            if tgt is None:
                continue
            add_plus_one_counters(state, tgt, 1)
            state.log.append(
                f"+1/+1 counter on {tgt.printing.name} "
                f"(now {tgt.plus_one_counters})"
            )

    def describe(self) -> str:
        return (
            f"put a +1/+1 counter on each of up to {self.max_n} target "
            f"creatures"
        )



@dataclass(frozen=True)
class AddManaToPool(Effect):
    """Add mana to the controller's mana pool. Paired with an
    ActivatedAbility flagged is_mana_ability=True so activation
    bypasses the stack.

    ``produces`` is a parsed list of (color, count) tuples. ``color``
    is one of "W", "U", "B", "R", "G", "C" (colorless), or "any"
    (wildcard satisfying any colored requirement; used by All-Fates
    Scroll's "Add one mana of any color").

    ``spend_restriction`` tags each generated coin for cost-payment
    filtering (e.g., "artifact_spell" for Steelswarm Operator's
    "Spend this mana only to cast an artifact spell";
    "artifact_activation" for Steelswarm's second mana ability
    "Spend this mana only to activate abilities of artifact
    sources").
    """

    produces: tuple[tuple[str, int], ...]
    spend_restriction: "str | None" = None

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import ManaCoin

        p = state.player_by_id(controller_id)
        minted: list[str] = []
        for color, count in self.produces:
            for _ in range(count):
                p.mana_pool.append(
                    ManaCoin(color=color, spend_restriction=self.spend_restriction)
                )
                minted.append(color)
        restriction = (
            f" ({self.spend_restriction})" if self.spend_restriction else ""
        )
        state.log.append(
            f"P{controller_id} adds {'+'.join(minted)}{restriction} to "
            f"mana pool"
        )

    def describe(self) -> str:
        parts = [
            (f"{{{c}}}" * n if c != "any" else "{any}") for c, n in self.produces
        ]
        base = "Add " + "".join(parts)
        if self.spend_restriction:
            base += f" (spend only to {self.spend_restriction})"
        return base



@dataclass(frozen=True)
class AddNManaOfChosenColor(Effect):
    """Add ``amount`` mana of a single color chosen by the controller
    at resolution time.

    Models oracle text like Lotus Field's "Add three mana of any one
    color" — the player picks ONE of W/U/B/R/G and gets ``amount``
    coins of that one color (not ``amount`` independent picks). This
    is strictly less flexible than minting "any"-coloured wildcards,
    so it is faithful to the printed clause.

    The choice is delegated to ``Agent.choose_mana_color``. If the
    controller has no agent seated (e.g. some unit-test setups), the
    first color in ``options`` is used as a deterministic fallback.

    Designed to be reusable for other "controller picks one color"
    mana abilities (Reflecting Pool, Meteor Crater, Sunken Citadel),
    though wiring those is left for a follow-up.
    """

    amount: int
    options: tuple[str, ...] = ("W", "U", "B", "R", "G")

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import ManaCoin

        p = state.player_by_id(controller_id)
        opts = list(self.options)
        chosen = opts[0]
        if p.agent is not None:
            picked = p.agent.choose_mana_color(state, controller_id, opts)
            if picked in opts:
                chosen = picked
        for _ in range(self.amount):
            p.mana_pool.append(ManaCoin(color=chosen))
        state.log.append(
            f"P{controller_id} adds {chosen}x{self.amount} to mana pool "
            f"(chosen color)"
        )

    def describe(self) -> str:
        return f"add {self.amount} mana of any one color"



@dataclass(frozen=True)
class AddOneManaFromControlledLandsProductions(Effect):
    """Reflecting Pool: "{T}: Add one mana of any type that a land you
    control could produce."

    On resolution, walk the controller's battlefield and union the
    ``produces`` color set of every land they control. Ask the
    controller's agent to pick one color from that union via
    ``choose_mana_color``, then mint one ``ManaCoin`` of that color.

    Defensive fallback: if the controller has no lands at all, or no
    land in play has any non-empty ``produces`` set (e.g., a colorless
    artifact-land cycle), mint a single ``"C"`` colorless coin so the
    ability still produces *something*. In normal play Reflecting Pool
    itself is on the battlefield, so the union is always non-empty for
    any deck containing a colored basic.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import ManaCoin

        p = state.player_by_id(controller_id)
        options: set[str] = set()
        for o in state.battlefield:
            if o.controller_id != controller_id:
                continue
            if not o.printing.is_land:
                continue
            options |= set(o.printing.produces)
        if not options:
            color = "C"
        else:
            sorted_options = sorted(options)
            if p.agent is not None:
                color = p.agent.choose_mana_color(
                    state, controller_id, sorted_options,
                )
            else:
                color = sorted_options[0]
            if color not in options:
                color = sorted_options[0]
        p.mana_pool.append(ManaCoin(color=color))
        state.log.append(
            f"P{controller_id} adds {color} to mana pool"
        )

    def describe(self) -> str:
        return "add one mana of any type a land you control could produce"



@dataclass(frozen=True)
class AddOneManaOfChosenPermanentColor(Effect):
    """Meteor Crater: "{T}: Choose a color of a permanent you control.
    Add one mana of that color."

    On resolution, walk the controller's battlefield and union the
    colors of every permanent they control (per ``mana_cost.colors``).
    Lands and other colorless permanents contribute nothing — colors
    come from colored mana symbols in the cost (rule 105.2). Ask the
    agent to pick one color from the union via ``choose_mana_color``,
    then mint one ``ManaCoin`` of that color.

    Defensive fallback: if the controller has no colored permanents
    (e.g., only lands and colorless artifacts), mint a single ``"C"``
    colorless coin so the ability still produces *something*.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import ManaCoin

        p = state.player_by_id(controller_id)
        options: set[str] = set()
        for o in state.battlefield:
            if o.controller_id != controller_id:
                continue
            options |= set(o.printing.mana_cost.colors)
        if not options:
            color = "C"
        else:
            sorted_options = sorted(options)
            if p.agent is not None:
                color = p.agent.choose_mana_color(
                    state, controller_id, sorted_options,
                )
            else:
                color = sorted_options[0]
            if color not in options:
                color = sorted_options[0]
        p.mana_pool.append(ManaCoin(color=color))
        state.log.append(
            f"P{controller_id} adds {color} to mana pool"
        )

    def describe(self) -> str:
        return (
            "add one mana of a color of a permanent you control"
        )



@dataclass(frozen=True)
class ControlTargetOpponentNextTurn(Effect):
    """The Dominion Bracelet: "You control target opponent during
    their next turn."

    Implementation: schedule the target player's ``scheduled_agent_
    override`` to the source-controller's agent, taking effect the
    next time it's the target's turn. On cleanup of that turn, the
    override is reverted automatically by run_cleanup_step.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id)
            for p in state.players
            if p.id != controller_id and not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "player":
            return
        target = state.player_by_id(target_id)
        caster = state.player_by_id(controller_id)
        # Schedule the override: the caster's agent will be swapped
        # into the target at the start of the target's next turn.
        target.scheduled_agent_override = caster.agent
        target.scheduled_agent_override_by = controller_id
        state.log.append(
            f"P{controller_id} will control P{target_id} during "
            f"P{target_id}'s next turn"
        )

    def describe(self) -> str:
        return "you control target opponent during their next turn"



@dataclass(frozen=True)
class ChangeControllerToCombatDamageSource(Effect):
    """Contested War Zone: "Whenever a creature deals combat damage to
    you, that creature's controller gains control of this land."

    Phase-9 wired the predicate-style ``Event.DEALS_COMBAT_DAMAGE``
    dispatch to thread the attacker's controller through
    ``state.resolving_combat_damage_source_controller_id`` (mirrored
    onto the StackItem) rather than ``fixed_targets``. The slot
    works the same way as ``state.resolving_combat_damage_amount``:
    set on push, mirrored at resolve time, cleared after. Permanent
    continuous effect (not UEOT); the new controller keeps the land
    until another control-change effect overrides it.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        new_cid = state.resolving_combat_damage_source_controller_id
        if new_cid is None or new_cid == source.controller_id:
            return
        old_cid = source.controller_id
        source.controller_id = new_cid
        state.log.append(
            f"P{new_cid} gains control of {source.printing.name} "
            f"(was P{old_cid})"
        )

    def describe(self) -> str:
        return "that creature's controller gains control of this land"



@dataclass(frozen=True)
class IfGainedLifeThisTurnUntapSelf(Effect):
    """Ragost, Deft Gastronaut's beginning-of-each-end-step rider:
    "If you gained life this turn, untap Ragost."

    Reads state.resolving_source for the permanent to untap. Checks
    the source-controller's life_gained_this_turn counter.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        p = state.player_by_id(source.controller_id)
        if p.life_gained_this_turn <= 0:
            return
        if source.tapped:
            source.tapped = False
            state.log.append(
                f"{source.printing.name} untaps ({p.life_gained_this_turn} "
                f"life gained this turn)"
            )

    def describe(self) -> str:
        return "if you gained life this turn, untap this permanent"



@dataclass(frozen=True)
class DefendingPlayerLosesHalfLifeRoundedUp(Effect):
    """Defending player loses half their life, rounded up. Matches
    Alpharael, Stonechosen's Void attack trigger. Simplified to "each
    opponent" — correct in 2-player, approximation in multiplayer.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            loss = (p.life + 1) // 2  # ceil(life / 2)
            p.life -= loss
            state.log.append(
                f"P{p.id} loses {loss} life (half, rounded up) — "
                f"life: {p.life}"
            )

    def describe(self) -> str:
        return "defending player loses half their life, rounded up"



@dataclass(frozen=True)
class EachOpponentMillsCards(Effect):
    """Each opponent puts the top ``amount`` cards of their library into
    their graveyard. Used as the 2-player simplification of "defending
    player mills N" attack triggers (Specimen Freighter).
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            milled = 0
            for _ in range(self.amount):
                if not p.library:
                    break
                card = p.library.pop(0)
                card.zone = Zone.GRAVEYARD
                p.graveyard.append(card)
                milled += 1
            state.log.append(f"P{p.id} mills {milled} cards")

    def describe(self) -> str:
        return f"defending player mills {self.amount} cards"



@dataclass(frozen=True)
class DrawCardsEqualToResolvingCounterAmount(Effect):
    """Draw a number of cards equal to
    ``state.resolving_counter_amount``. Matches Terrasymbiosis's "draw
    that many cards" observer. No-op if amount is 0.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import draw_n_cards

        amount = state.resolving_counter_amount
        if amount <= 0:
            return
        drawn = draw_n_cards(state, controller_id, amount)
        state.log.append(
            f"P{controller_id} draws {drawn} (counter-added trigger)"
        )

    def describe(self) -> str:
        return "you may draw that many cards"



@dataclass(frozen=True)
class DrawCardsEqualToResolvingCombatDamage(Effect):
    """Draw a number of cards equal to
    ``state.resolving_combat_damage_amount``. Matches Starwinder's
    "you may draw that many cards" observer trigger (the "may" is
    simplified to always yes, consistent with other OptionalXyz
    simplifications in this engine).

    No-op if the resolving amount is 0 (e.g. accidentally invoked
    outside a combat-damage trigger).
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import draw_n_cards

        amount = state.resolving_combat_damage_amount
        if amount <= 0:
            return
        drawn = draw_n_cards(state, controller_id, amount)
        state.log.append(
            f"P{controller_id} draws {drawn} cards (combat-damage trigger)"
        )

    def describe(self) -> str:
        return "you may draw that many cards"



@dataclass(frozen=True)
class CreateTokenPerMulticoloredPermanent(Effect):
    """Create a ``token_printing`` token (optionally tapped) for each
    multicolored permanent the controller controls.

    Matches Infinite Guideline Station's ETB.
    """

    token_printing: "CardPrinting"
    tapped: bool = True

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.game import Zone

        n = _multicolored_permanents_count(state, controller_id)
        if n <= 0:
            state.log.append(
                f"P{controller_id} controls 0 multicolored permanents "
                f"— no tokens created"
            )
            return
        for _ in range(n):
            obj = state.new_object(
                self.token_printing, owner_id=controller_id,
                zone=Zone.BATTLEFIELD,
            )
            put_into_battlefield(state, obj, controller_id)
            if self.tapped:
                obj.tapped = True
        state.log.append(
            f"P{controller_id} creates {n} "
            f"{self.token_printing.name} tokens "
            f"(multicolored-permanent count)"
        )

    def describe(self) -> str:
        return (
            f"create a tapped {self.token_printing.name} token for each "
            f"multicolored permanent you control"
        )



@dataclass(frozen=True)
class DrawCardPerMulticoloredPermanent(Effect):
    """Draw one card for each multicolored permanent you control.
    Matches Infinite Guideline Station's attack trigger."""

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        n = _multicolored_permanents_count(state, controller_id)
        p = state.player_by_id(controller_id)
        drew = 0
        for _ in range(n):
            if not p.library:
                break
            card = p.library.pop(0)
            card.zone = Zone.HAND
            p.hand.append(card)
            drew += 1
        state.log.append(
            f"P{controller_id} draws {drew} cards "
            f"(multicolored-permanent count)"
        )

    def describe(self) -> str:
        return "draw a card for each multicolored permanent you control"



@dataclass(frozen=True)
class PutLoyaltyCounterOnSelf(Effect):
    """Put ``amount`` loyalty counter(s) on the ability's source.

    Used by planeswalker "whenever ..." triggers that grow their own
    loyalty (Tezzeret's "whenever an artifact you control enters, put
    a loyalty counter on Tezzeret"). Reads state.resolving_source.
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        source.loyalty_counters += self.amount
        state.log.append(
            f"{source.printing.name} gains {self.amount} loyalty "
            f"counter(s) (now {source.loyalty_counters})"
        )

    def describe(self) -> str:
        return f"put {self.amount} loyalty counter(s) on ~"



@dataclass(frozen=True)
class LookAtTopNRevealOrCounter(Effect):
    """Look at the top ``n`` cards of your library. The controller may
    reveal one card matching ``filter_name`` and put it into hand.
    Put the rest on the bottom of the library in a random order. If no
    card was revealed (either because none matched or the agent
    declined), put a +1/+1 counter on the source.

    Matches Pulsar Squadron Ace's ETB with ``n=5``, ``filter_name=
    "spacecraft"``. Reads ``state.resolving_source`` for the counter
    placement.

    The "random order" shuffle is seeded deterministically on the turn
    number + controller + library length so tests remain reproducible.
    """

    n: int = 5
    filter_name: str = "spacecraft"
    # Inclusive upper bound on printed power for the ``creature`` filter
    # — used by Living Phone's ``creature card with power 2 or less``
    # tutor body. ``0`` (default) means "no power cap" so the existing
    # Pulsar Squadron Ace ``creature`` use-site keeps its unrestricted
    # behaviour. Ignored for non-creature filters.
    max_power: int = 0
    # When ``True`` (default, matches Pulsar Squadron Ace), failing to
    # reveal a card puts a +1/+1 counter on the resolving source. Living
    # Phone has no counter rider — its oracle text is the bare
    # ``Put the rest on the bottom of your library in a random order``
    # tail with no ``Otherwise, put a +1/+1 counter`` clause — so its
    # printing carries ``counter_on_miss=False`` and the resolution
    # short-circuits to bottoming the remainder when no card is revealed.
    counter_on_miss: bool = True

    def _matches(self, printing) -> bool:
        fn = self.filter_name
        if fn == "spacecraft":
            return printing.is_spacecraft
        if fn == "creature":
            if not printing.is_creature:
                return False
            if self.max_power > 0 and printing.power > self.max_power:
                return False
            return True
        if fn == "artifact":
            return printing.is_artifact
        return False

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        p = state.player_by_id(controller_id)
        examined = p.library[: self.n]
        options = [c for c in examined if self._matches(c.printing)]
        chosen_id = None
        if options and p.agent is not None:
            chosen_id = p.agent.choose_card_from_library(
                state, controller_id, [c.id for c in options]
            )
        revealed_card = None
        if chosen_id is not None:
            revealed_card = next(
                (c for c in options if c.id == chosen_id), None
            )
        if revealed_card is not None:
            from mtg_bot.game import reveal_to_all
            p.library.remove(revealed_card)
            revealed_card.zone = Zone.HAND
            p.hand.append(revealed_card)
            reveal_to_all(state, revealed_card)
            state.log.append(
                f"P{controller_id} reveals {revealed_card.printing.name} "
                f"and puts it into hand"
            )
        else:
            state.log.append(
                f"P{controller_id} reveals nothing from the top "
                f"{len(examined)} cards"
            )
            # Pulsar Squadron Ace's ``Otherwise, put a +1/+1 counter
            # on it`` rider. Living Phone's printing carries
            # ``counter_on_miss=False`` so a missed reveal there just
            # bottoms the remainder without a counter — matching its
            # printed text which has no such fallback clause.
            if self.counter_on_miss:
                source = state.resolving_source
                if source is not None and source in state.battlefield:
                    from mtg_bot.game import add_plus_one_counters
                    add_plus_one_counters(state, source, 1)
                    state.log.append(
                        f"{source.printing.name} gets a +1/+1 counter "
                        f"(now {source.plus_one_counters})"
                    )
        # Bottom the remainder in a random order. Take up to n cards off
        # the top (those we examined except the revealed one if any) and
        # shuffle them onto the bottom.
        from mtg_bot.game import reset_public_knowledge
        remaining_examined = [
            c for c in examined if c is not revealed_card
        ]
        for c in remaining_examined:
            p.library.remove(c)
            # Disturbance: any prior public knowledge of these cards'
            # top-of-library positions is now stale.
            reset_public_knowledge(c)
        state.rng.shuffle(remaining_examined)
        p.library.extend(remaining_examined)

    def describe(self) -> str:
        if self.filter_name == "creature" and self.max_power > 0:
            filter_phrase = f"creature card with power {self.max_power} or less"
        else:
            filter_phrase = f"{self.filter_name} card"
        base = (
            f"look at the top {self.n} cards of your library; you may "
            f"reveal a {filter_phrase} from among them and put "
            f"it into your hand; put the rest on the bottom of your "
            f"library in a random order"
        )
        if self.counter_on_miss:
            base += (
                "; if you didn't put a card into your hand this way, "
                "put a +1/+1 counter on ~"
            )
        return base



@dataclass(frozen=True)
class LookAtTopTwoSplitHandGraveyard(Effect):
    """Look at the top two cards of your library; put one into your
    hand, the other into your graveyard.

    Matches Codecracker Hound's ETB. The agent's choice via
    choose_surveil decides which pile each card goes to (surveil=True
    means graveyard, False means hand — mirrors the surveil semantics).
    Asymmetric: exactly one card ends up in hand, exactly one in
    graveyard (or fewer if the library has < 2 cards).
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        p = state.player_by_id(controller_id)
        if not p.library:
            return
        # Take the top 2 (or 1 if only one left).
        examined = p.library[:2]
        # Asks the agent per card; first "True" (graveyard) pick goes
        # there, then the remainder to hand. Fallback heuristic: top
        # card → hand, second → graveyard.
        if len(examined) == 1:
            card = examined[0]
            p.library.remove(card)
            card.zone = Zone.HAND
            p.hand.append(card)
            state.log.append(
                f"P{controller_id} draws {card.printing.name}"
            )
            return
        # Two cards; pick the one to send to hand. The other goes to
        # graveyard. Default: top card → hand.
        if p.agent is None:
            hand_card = examined[0]
            gy_card = examined[1]
        else:
            hand_card = examined[0]
            if p.agent.choose_surveil(state, p.id, hand_card.id):
                # Agent says "mill this one" → swap: other → hand.
                hand_card = examined[1]
                gy_card = examined[0]
            else:
                gy_card = examined[1]
        p.library.remove(hand_card)
        hand_card.zone = Zone.HAND
        p.hand.append(hand_card)
        p.library.remove(gy_card)
        gy_card.zone = Zone.GRAVEYARD
        p.graveyard.append(gy_card)
        state.log.append(
            f"P{controller_id} puts {hand_card.printing.name} into "
            f"hand and {gy_card.printing.name} into graveyard"
        )

    def describe(self) -> str:
        return (
            "look at the top two cards of your library; put one into "
            "your hand and the other into your graveyard"
        )



@dataclass(frozen=True)
class ManifestDread(Effect):
    """CR 701.55: To manifest dread, look at the top two cards of your
    library, manifest one of them, then put the other into your
    graveyard. To manifest a card (CR 701.34) is to put that card onto
    the battlefield face down. It's a 2/2 creature with no name, no
    subtypes, and no abilities.

    Repeats ``count`` times — for "Manifest dread X times" / "Manifest
    dread twice" surfaces. The agent's ``choose_card_from_library``
    hook picks which of the two examined cards is manifested
    (face-down) on each iteration; the other goes to the graveyard.
    With fewer than two cards left in the library the effect manifests
    one (if available) and skips the graveyard step — the simplest
    least-surprising fallback.
    """

    count: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.game import Zone

        n = self.count
        # ``count`` may be ``None`` for the X-times surface — read the
        # X-value off the state when present (Valgavoth's Onslaught).
        if n is None:
            n = state.resolving_x_value
        for _ in range(n):
            p = state.player_by_id(controller_id)
            if not p.library:
                return
            examined = p.library[:2]
            # With fewer than two cards left, manifest the only card
            # available and skip the graveyard step.
            if len(examined) == 1:
                card = examined[0]
                p.library.remove(card)
                card.face_down = True
                put_into_battlefield(state, card, controller_id)
                state.log.append(
                    f"P{controller_id} manifests a face-down card "
                    f"(library exhausted)"
                )
                continue
            # Two cards examined. Ask the agent which to manifest;
            # the other goes to graveyard. Default fallback: manifest
            # the top card.
            options = [c.id for c in examined]
            manifest_id = options[0]
            if p.agent is not None:
                chosen = p.agent.choose_card_from_library(
                    state, p.id, options
                )
                if chosen in options:
                    manifest_id = chosen
            manifest_card = next(c for c in examined if c.id == manifest_id)
            gy_card = next(c for c in examined if c.id != manifest_id)
            p.library.remove(manifest_card)
            manifest_card.face_down = True
            put_into_battlefield(state, manifest_card, controller_id)
            p.library.remove(gy_card)
            gy_card.zone = Zone.GRAVEYARD
            p.graveyard.append(gy_card)
            state.log.append(
                f"P{controller_id} manifests a face-down card; "
                f"{gy_card.printing.name} goes to the graveyard"
            )

    def describe(self) -> str:
        if self.count == 1:
            return "manifest dread"
        if self.count is None:
            return "manifest dread X times"
        if self.count == 2:
            return "manifest dread twice"
        if self.count == 3:
            return "manifest dread three times"
        return f"manifest dread {self.count} times"


@dataclass(frozen=True)
class ManifestDreadThenPutPlusOneCountersOnIt(Effect):
    """Manifest dread, then put ``counter_amount`` +1/+1 counter(s) on
    that creature.

    Matches the door triggers on Slimy Aquarium ("manifest dread, then
    put a +1/+1 counter on that creature" — counter_amount=1) and
    Weight Room ("manifest dread, then put three +1/+1 counters on
    that creature" — counter_amount=3) on Duskmourn Rooms.

    "That creature" refers to the just-manifested face-down 2/2 — the
    other examined card goes to the graveyard per ``ManifestDread``'s
    semantics. To track which permanent receives the counter we
    snapshot the battlefield ids before the manifest and pick the
    single id that wasn't there before. With fewer than two cards
    left in the library ``ManifestDread`` may manifest one or zero
    cards — the counter step then either lands on the lone manifested
    face-down (one new id) or no-ops (no new id).
    """

    counter_amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import add_plus_one_counters

        pre_ids = {o.id for o in state.battlefield}
        ManifestDread(count=1).resolve(state, controller_id, ())
        new_ids = [o.id for o in state.battlefield if o.id not in pre_ids]
        if not new_ids:
            return
        # Manifest dread enters exactly one face-down creature when it
        # successfully resolves, so pick that single new id.
        manifested_id = new_ids[-1]
        manifested = next(
            (o for o in state.battlefield if o.id == manifested_id), None,
        )
        if manifested is None:
            return
        add_plus_one_counters(state, manifested, self.counter_amount)
        state.log.append(
            f"{manifested.printing.name or 'manifested face-down creature'}"
            f" gets {self.counter_amount} +1/+1 counter(s) "
            f"(now {manifested.plus_one_counters})"
        )

    def describe(self) -> str:
        if self.counter_amount == 1:
            return (
                "manifest dread, then put a +1/+1 counter on that creature"
            )
        return (
            f"manifest dread, then put {self.counter_amount} +1/+1 "
            f"counters on that creature"
        )


@dataclass(frozen=True)
class ManifestDreadThenAttachSelf(Effect):
    """Manifest dread, then attach the source Equipment to that creature.

    Matches the Duskmourn Equipment ETB family — Conductive Machete,
    Killer's Mask, Cursed Windbreaker, Dissection Tools — whose oracle
    text reads ``"When this Equipment enters, manifest dread, then
    attach this Equipment to that creature."``. "That creature" refers
    to the just-manifested face-down 2/2 (the same back-reference
    ``ManifestDreadThenPutPlusOneCountersOnIt`` resolves), so the
    Equipment auto-attaches without a printed-target choice — no
    ``requires_target`` machinery needed.

    Implementation parallels ``ManifestDreadThenPutPlusOneCountersOnIt``:
    snapshot the battlefield ids before manifesting, run
    ``ManifestDread(count=1)``, then identify the new face-down
    permanent by id-set difference. With fewer than two cards left in
    the library, ``ManifestDread`` may manifest one or zero cards —
    the attach step then either lands on the lone manifested face-down
    (one new id) or no-ops (no new id, Equipment stays unattached).
    No-ops if the source isn't an Equipment (defensive — only Equipment
    builds wire this engine class today).
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        if "Equipment" not in source.printing.subtypes:
            return
        pre_ids = {o.id for o in state.battlefield}
        ManifestDread(count=1).resolve(state, controller_id, ())
        new_ids = [o.id for o in state.battlefield if o.id not in pre_ids]
        if not new_ids:
            return
        # Manifest dread enters exactly one face-down creature when it
        # successfully resolves; pick that single new id as the attach
        # target.
        manifested_id = new_ids[-1]
        manifested = next(
            (o for o in state.battlefield if o.id == manifested_id), None,
        )
        if manifested is None:
            return
        source.attached_to = manifested.id
        state.log.append(
            f"{source.printing.name} is attached to "
            f"{manifested.printing.name or 'manifested face-down creature'}"
        )

    def describe(self) -> str:
        return (
            "manifest dread, then attach this Equipment to that creature"
        )


@dataclass(frozen=True)
class Surveil(Effect):
    """Surveil N: look at the top N cards of your library; for each one, you
    may put it into your graveyard. The rest stay on top in their original
    order. Decision delegated to the controller's agent (choose_surveil).

    If ``use_artifact_count`` is True the amount is recomputed at
    resolution as the controller's current artifact count — matches
    Cerebral Download's "Surveil X, where X is the number of artifacts
    you control".
    """

    amount: int = 1
    use_artifact_count: bool = False

    def _amount(self, state: "GameState", controller_id: int) -> int:
        if not self.use_artifact_count:
            return self.amount
        return sum(
            1 for o in state.battlefield
            if o.controller_id == controller_id and o.printing.is_artifact
        )

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        p = state.player_by_id(controller_id)
        n = self._amount(state, controller_id)
        milled = 0
        # Examine the top N cards in order. Mill ones the agent chooses;
        # keep the rest on top in their original order.
        examined = p.library[: n]
        keep: list = []
        for card in examined:
            if p.agent is not None and p.agent.choose_surveil(state, p.id, card.id):
                p.library.remove(card)
                card.zone = Zone.GRAVEYARD
                p.graveyard.append(card)
                milled += 1
            else:
                keep.append(card)
        state.log.append(
            f"P{controller_id} surveils {n} ({milled} milled)"
        )

    def describe(self) -> str:
        if self.use_artifact_count:
            return "surveil X, where X is the number of artifacts you control"
        return f"surveil {self.amount}"



@dataclass(frozen=True)
class SelfMill(Effect):
    """Controller mills ``amount`` cards from the top of their own library.

    Matches oracle text like "mill three cards" on triggered abilities,
    where no target selection is needed.
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        p = state.player_by_id(controller_id)
        milled = 0
        for _ in range(self.amount):
            if not p.library:
                break
            card = p.library.pop(0)
            card.zone = Zone.GRAVEYARD
            p.graveyard.append(card)
            milled += 1
        state.log.append(f"P{controller_id} mills {milled} cards")

    def describe(self) -> str:
        return f"mill {self.amount} cards"



@dataclass(frozen=True)
class ExileTopOfLibraryMayPlayThisTurn(Effect):
    """Controller exiles the top ``amount`` cards of their own library
    and marks each with ``playable_from_exile_this_turn`` so they're
    castable / playable until cleanup of the resolving turn.

    Matches Charred Foyer's upkeep trigger ("At the beginning of your
    upkeep, exile the top card of your library. You may play it this
    turn.") and any other "exile the top N cards of your library; you
    may play them this turn" shape that doesn't carry an
    until-next-turn rider. The flag clears at cleanup of the casting
    turn via the existing turn.run_cleanup loop — same lifetime that
    impulse-cast flag uses everywhere else (Territorial Bruntar,
    Memorial Vault, Possibility Technician).

    ``amount`` defaults to 1 (the Charred Foyer shape); call sites
    that exile multiple cards (Painter's Studio's "exile the top two
    cards" — when "until end of next turn" support lands) pass a
    different value. Running out of cards mid-exile simply stops; no
    empty-library loss is triggered (impulse-cast at sorcery speed
    doesn't draw, so the empty-library loss check doesn't fire here).
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        p = state.player_by_id(controller_id)
        exiled = 0
        for _ in range(self.amount):
            if not p.library:
                break
            top = p.library.pop(0)
            top.zone = Zone.EXILE
            top.playable_from_exile_this_turn = True
            p.exile.append(top)
            exiled += 1
        state.log.append(
            f"P{controller_id} exiles {exiled} card(s) from top of "
            f"library (may play this turn)"
        )

    def describe(self) -> str:
        if self.amount == 1:
            return (
                "exile the top card of your library; you may play it "
                "this turn"
            )
        return (
            f"exile the top {self.amount} cards of your library; you "
            f"may play them this turn"
        )



@dataclass(frozen=True)
class MillTargetPlayer(Effect):
    """Target player puts the top N cards of their library into their
    graveyard.

    If ``use_controller_life`` is True, N is the controller's current life
    total (minimum 0) instead of ``amount`` — matches Space-Time Anomaly.
    Running out of cards mid-mill simply stops; empty-library loss is
    handled separately by the draw step.
    """

    amount: int = 1
    use_controller_life: bool = False

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id) for p in state.players if not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "player":
            return
        target = state.player_by_id(target_id)
        if self.use_controller_life:
            n = max(0, state.player_by_id(controller_id).life)
        else:
            n = self.amount
        milled = 0
        for _ in range(n):
            if not target.library:
                break
            card = target.library.pop(0)
            card.zone = Zone.GRAVEYARD
            target.graveyard.append(card)
            milled += 1
        state.log.append(f"P{target_id} mills {milled} cards")

    def describe(self) -> str:
        if self.use_controller_life:
            return (
                "target player mills cards equal to your life total"
            )
        return f"target player mills {self.amount} cards"



@dataclass(frozen=True)
class DrawCards(Effect):
    amount: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import draw_n_cards

        drawn = draw_n_cards(state, controller_id, self.amount)
        state.log.append(f"P{controller_id} draws {drawn} card(s)")

    def describe(self) -> str:
        if self.amount == 1:
            return "you draw a card"
        return f"you draw {self.amount} cards"



@dataclass(frozen=True)
class EachPlayerDraws(Effect):
    amount: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import draw_n_cards

        for i in range(len(state.players)):
            idx = (state.active_player_idx + i) % len(state.players)
            p = state.players[idx]
            if p.has_lost:
                continue
            draw_n_cards(state, p.id, self.amount)
        state.log.append(
            f"each player draws {self.amount} card(s) "
            f"(from {state.players[controller_id].id}'s trigger)"
        )

    def describe(self) -> str:
        if self.amount == 1:
            return "each player draws a card"
        return f"each player draws {self.amount} cards"



@dataclass(frozen=True)
class PumpTargetCreatureUntilEOT(Effect):
    """Target creature gets +power/+toughness and optionally gains keywords
    until end of turn.

    Matches oracle text like "Target creature gets +3/+0 and gains first
    strike and reach until end of turn." Bonuses stack additively and are
    cleared at cleanup. Negative bonuses model "target creature gets
    -N/-0" debuffs.

    Set ``optional=True`` for "up to one target creature" oracle text:
    the trigger / activation resolves as a no-op when no target is
    chosen / available (e.g. Dubious Delicacy's ETB).

    ``granted_triggered_ability`` (optional) appends a UEOT triggered
    ability to the target's ``temp_triggered_abilities``. Use for cards
    whose pump rider also grants "Whenever ~ does X, Y" until EOT.
    """

    power_bonus: int = 0
    toughness_bonus: int = 0
    granted_keywords: frozenset[str] = frozenset()
    you_control_only: bool = False  # restrict to "target creature you control"
    opponents_only: bool = False  # restrict to "an opponent controls"
    # Extra keywords granted UEOT only when the target was cast for its
    # warp cost (oracle: "If that creature was cast for its warp cost, it
    # also gains X"). Generic — any future card with the same rider uses it.
    granted_keywords_if_warp_cast: frozenset[str] = frozenset()
    # Conditional draw rider: "If [target] has a +1/+1 counter on it,
    # draw a card." Matches Dual-Sun Technique. The controller of the
    # spell does the drawing. Inspected at resolution time.
    draw_if_target_has_plus_one_counter: int = 0
    # Target subtype filter: when set, only creatures whose subtypes
    # (printed OR runtime-granted via ``GameObject.temp_subtype``)
    # include this value are legal targets. Matches oracle text like
    # "Target Blinkmoth creature gets +1/+1 until end of turn"
    # (Blinkmoth Nexus — the animated Nexus carries the Blinkmoth
    # subtype as a temp grant from BecomeLandCreatureUEOT).
    target_subtype: str | None = None
    # Granted triggered ability that's appended to the target's
    # ``temp_triggered_abilities`` UEOT. Matches oracle text like
    # Requiem Monolith's "Until end of turn, target creature gains
    # 'Whenever this creature is dealt damage, you draw that many
    # cards and lose that much life.'" Cleared at cleanup with the
    # rest of the temp_* fields.
    granted_triggered_ability: "TriggeredAbility | None" = None
    # "Up to one target" semantics — the trigger / activation resolves
    # as a no-op when no target is chosen or available (e.g. Dubious
    # Delicacy's ETB "up to one target creature gets -3/-3 UEOT").
    optional: bool = False
    # "Another target creature ..." — exclude state.resolving_source from
    # legal targets and from resolve. Matches oracle wording on cards
    # like Perigee Beckoner's ETB ("another target creature you control
    # gets +2/+0 and gains [dies-rider]").
    exclude_self: bool = False
    # "Target attacking creature ..." — restrict the legal-target pool to
    # creatures currently flagged ``attacking=True``. Matches the DSK
    # "Whenever you attack" rider on Most Valuable Slayer
    # ("...target attacking creature gets +1/+0 and gains first strike
    # until end of turn"). Independent of the controller filters
    # (``you_control_only`` / ``opponents_only``); attackers are always
    # the active player's so in practice the source's controller is
    # also the attackers' controller, but keeping the flag orthogonal
    # mirrors the engine class's other narrowing-flag conventions.
    attacking_only: bool = False

    def requires_target(self) -> bool:
        return not self.optional

    def target_count_range(self) -> tuple[int, int]:
        return (0, 1) if self.optional else (1, 1)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        from mtg_bot.game import effective_subtypes

        source = state.resolving_source if self.exclude_self else None
        source_id = source.id if source is not None else -1
        return [
            ("creature", o.id)
            for o in state.battlefield
            if (o.printing.is_creature or o.temp_is_creature)
            and (not self.you_control_only or o.controller_id == controller_id)
            and (not self.opponents_only or o.controller_id != controller_id)
            and (not self.attacking_only or o.attacking)
            and (
                self.target_subtype is None
                or self.target_subtype in effective_subtypes(state, o)
            )
            and o.id != source_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        src = state.resolving_source
        if self.power_bonus or self.toughness_bonus:
            state.grant_pt_pump_ueot(
                source_id=src.id if src is not None else obj.id,
                target_id=obj.id,
                power=self.power_bonus,
                toughness=self.toughness_bonus,
                description=(
                    f"{src.printing.name if src else obj.printing.name} "
                    f"pumps {obj.printing.name} "
                    f"+{self.power_bonus}/+{self.toughness_bonus} UEOT"
                ),
            )
        granted = self.granted_keywords
        if self.granted_keywords_if_warp_cast and obj.cast_via_warp:
            granted = granted | self.granted_keywords_if_warp_cast
        if granted:
            source = state.resolving_source
            state.grant_keywords_ueot(
                source_id=source.id if source is not None else obj.id,
                target_id=obj.id,
                keywords=granted,
                log_source_name=(
                    source.printing.name if source is not None
                    else obj.printing.name
                ),
                log_target_name=obj.printing.name,
            )
        if self.granted_triggered_ability is not None:
            src = state.resolving_source
            state.grant_triggered_ability_ueot(
                source_id=src.id if src is not None else obj.id,
                target_id=obj.id,
                ability=self.granted_triggered_ability,
                description=(
                    f"{src.printing.name if src else obj.printing.name} "
                    f"grants UEOT trigger to {obj.printing.name}"
                ),
            )
        state.log.append(
            f"{obj.printing.name} gets "
            f"+{self.power_bonus}/+{self.toughness_bonus} until end of turn"
            + (f" and gains {', '.join(sorted(granted))}"
               if granted else "")
            + (" and gains a granted triggered ability"
               if self.granted_triggered_ability is not None else "")
        )
        if (
            self.draw_if_target_has_plus_one_counter > 0
            and obj.plus_one_counters > 0
        ):
            from mtg_bot.turn import draw_n_cards
            draw_n_cards(
                state, controller_id,
                self.draw_if_target_has_plus_one_counter,
            )
            state.log.append(
                f"P{controller_id} draws "
                f"{self.draw_if_target_has_plus_one_counter} "
                f"(target had +1/+1 counter)"
            )

    def describe(self) -> str:
        parts: list[str] = []
        if self.power_bonus or self.toughness_bonus:
            parts.append(
                f"target creature gets +{self.power_bonus}/+{self.toughness_bonus}"
            )
        if self.granted_keywords:
            parts.append(
                f"gains {', '.join(sorted(self.granted_keywords))}"
            )
        if self.granted_triggered_ability is not None:
            parts.append(
                f"gains \"{self.granted_triggered_ability.describe()}\""
            )
        return " and ".join(parts) + " until end of turn."


@dataclass(frozen=True)
class GrantUnblockable(Effect):
    """Unified can't-be-blocked-this-turn primitive driven by ``targets``.

    Sibling of :class:`Destroy` / :class:`Exile` / :class:`Tap` /
    :class:`Pump` / :class:`Bounce` / :class:`PutPlusOneCounters`
    for the can't-be-blocked family. The unified resolve walks each
    slot, re-verifies legality at resolution per CR 608.2b, and sets
    ``temp_unblockable = True`` on each surviving target — the flag
    is read by ``combat.eligible_blockers`` at block-declaration
    time and cleared at cleanup.
    """

    targets: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(
                            state, chosen, action="unblockable grant",
                        )
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                obj.temp_unblockable = True
                state.log.append(
                    f"{obj.printing.name} can't be blocked this turn"
                )
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return f"{_describe_targets(self.targets)} can't be blocked this turn"


@dataclass(frozen=True)
class GrantUnblockableUEOT(GrantUnblockable):
    """``GrantUnblockable`` constructor sugar for "target creature can't
    be blocked this turn" with optional filter narrowing flags.

    Filter composition:
      - base: ``IsCreature()``
      - ``attacking_only=True``: ``& IsAttacking()`` (Clammy Prowler's
        attacks-trigger pool — declare-attackers-step shape)
      - ``exclude_self=True``: ``& ~IsSource()`` ("another target")
      - ``power_at_most=N``: ``& HasPowerAtMost(N)`` (Ragged Playmate's
        "with power 2 or less")
    """

    attacking_only: bool = False
    exclude_self: bool = False
    power_at_most: int | None = None

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            HasPowerAtMost, IsAttacking, IsCreature, IsSource,
        )
        from mtg_bot.effects.targets import TargetSpec
        f = IsCreature()
        if self.attacking_only:
            f = f & IsAttacking()
        if self.exclude_self:
            f = f & ~IsSource()
        if self.power_at_most is not None:
            f = f & HasPowerAtMost(self.power_at_most)
        object.__setattr__(self, "targets", (TargetSpec(f),))

    def describe(self) -> str:
        if self.attacking_only and self.exclude_self:
            return (
                "another target attacking creature can't be blocked this turn"
            )
        if self.attacking_only:
            return "target attacking creature can't be blocked this turn"
        if self.power_at_most is not None:
            return (
                f"target creature with power {self.power_at_most} or less "
                f"can't be blocked this turn"
            )
        return "target creature can't be blocked this turn"


@dataclass(frozen=True)
class GrantTriggeredAbilityUEOT(Effect):
    """Grant a triggered ability to target creature until end of turn.

    Appends ``ability`` to the target's ``temp_triggered_abilities``.
    Fire helpers iterate that tuple alongside the printed abilities.
    Cleared at cleanup. Use for "Until end of turn, target creature
    gains 'Whenever ~ does X, Y.'" — Requiem Monolith is the canonical
    EoE example.

    ``you_control_only`` mirrors the same flag on PumpTargetCreatureUntilEOT
    for "target creature you control" wording.
    """

    ability: "TriggeredAbility | None" = None
    you_control_only: bool = False
    opponents_only: bool = False

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and (not self.you_control_only or o.controller_id == controller_id)
            and (not self.opponents_only or o.controller_id != controller_id)
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if self.ability is None or not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        src = state.resolving_source
        state.grant_triggered_ability_ueot(
            source_id=src.id if src is not None else obj.id,
            target_id=obj.id,
            ability=self.ability,
            description=(
                f"{src.printing.name if src else obj.printing.name} "
                f"grants UEOT trigger to {obj.printing.name}"
            ),
        )
        state.log.append(
            f"{obj.printing.name} gains a granted triggered ability "
            f"until end of turn"
        )

    def describe(self) -> str:
        if self.ability is None:
            return "grant a triggered ability until end of turn"
        return (
            f"until end of turn, target creature gains "
            f"\"{self.ability.describe()}\""
        )



@dataclass(frozen=True)
class TargetArtifactBecomesCreatureUEOT(Effect):
    """Synthesizer Labship's 2+ tier trigger: ``up to one other target
    artifact you control becomes an artifact creature with base power
    and toughness ``power``/``toughness`` and gains ``granted_keywords``
    until end of turn.``

    The effect reuses the manland animation primitives — sets
    ``temp_is_creature``, overrides base P/T via ``temp_base_pt_override``,
    and folds keywords into ``temp_keywords``. Cleanup wipes all three.
    Targets exclude the source (``"other"``); both creature and non-
    creature artifacts are valid targets (a non-creature artifact
    gains creature status until end of turn; an already-creature
    artifact has its base P/T overridden — which can shrink a
    stationed Spacecraft).
    """

    power: int = 0
    toughness: int = 0
    granted_keywords: frozenset[str] = frozenset()

    def requires_target(self) -> bool:
        return True

    def target_count_range(self) -> tuple[int, int]:
        # "Up to one" — declining is legal.
        return (0, 1)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        source = state.resolving_source
        source_id = source.id if source is not None else -1
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.controller_id == controller_id
            and o.printing.is_artifact
            and o.id != source_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        src = state.resolving_source
        state.grant_become_creature_ueot(
            source_id=src.id if src is not None else obj.id,
            target_id=obj.id,
            power=self.power,
            toughness=self.toughness,
            description=(
                f"{obj.printing.name} becomes a "
                f"{self.power}/{self.toughness} artifact creature UEOT"
            ),
        )
        if self.granted_keywords:
            state.grant_keywords_ueot(
                source_id=src.id if src is not None else obj.id,
                target_id=obj.id,
                keywords=self.granted_keywords,
                log_source_name=(
                    src.printing.name if src is not None
                    else obj.printing.name
                ),
                log_target_name=obj.printing.name,
            )
        kw_note = (
            f" and gains {', '.join(sorted(self.granted_keywords))}"
            if self.granted_keywords else ""
        )
        state.log.append(
            f"{obj.printing.name} becomes a {self.power}/{self.toughness}"
            f" artifact creature{kw_note} until end of turn"
        )

    def describe(self) -> str:
        kw = ""
        if self.granted_keywords:
            kw = f" and gains {', '.join(sorted(self.granted_keywords))}"
        return (
            f"up to one other target artifact you control becomes a "
            f"{self.power}/{self.toughness} artifact creature{kw} until "
            f"end of turn"
        )



@dataclass(frozen=True)
class DealNDamageToEachCreature(Effect):
    """Deal ``amount`` damage to each creature on the battlefield.

    Snapshot semantics: the victim list is captured before any damage
    is applied, so any deaths mid-pass don't affect what gets damaged.
    Marks damage via _apply_damage (the same helper used by targeted
    burn), which routes through creature toughness for SBAs.
    """

    amount: int

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        victims = [
            o for o in state.battlefield if o.printing.is_creature
        ]
        for o in victims:
            _apply_damage(state, ("creature", o.id), self.amount)

    def describe(self) -> str:
        return f"deals {self.amount} damage to each creature"



@dataclass(frozen=True)
class PumpSelfUntilEOT(Effect):
    """The ability's source gets +power/+toughness (and optional keywords)
    until end of turn. Reads state.resolving_source."""

    power_bonus: int = 0
    toughness_bonus: int = 0
    granted_keywords: frozenset[str] = frozenset()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        if self.power_bonus or self.toughness_bonus:
            state.grant_pt_pump_ueot(
                source_id=source.id, target_id=source.id,
                power=self.power_bonus, toughness=self.toughness_bonus,
                description=(
                    f"{source.printing.name} pumps itself "
                    f"+{self.power_bonus}/+{self.toughness_bonus} UEOT"
                ),
            )
        if self.granted_keywords:
            state.grant_keywords_ueot(
                source_id=source.id,
                target_id=source.id,
                keywords=self.granted_keywords,
                log_source_name=source.printing.name,
                log_target_name=source.printing.name,
            )
        state.log.append(
            f"{source.printing.name} gets "
            f"+{self.power_bonus}/+{self.toughness_bonus} until end of turn"
        )

    def describe(self) -> str:
        return (
            f"this gets +{self.power_bonus}/+{self.toughness_bonus} "
            f"until end of turn"
        )



@dataclass(frozen=True)
class PumpCreaturesYouControlUntilEOT(Effect):
    """Each creature the controller controls gets +power/+toughness until
    end of turn (applied at resolution; creatures entering later don't get
    the bonus — Stage-3 simplification vs. a continuous effect).

    ``exclude_source=True`` models "other creatures you control" — reads
    ``state.resolving_source`` to skip the triggering permanent.

    ``attacking_only=True`` narrows the scope to attacking creatures
    regardless of controller — "Attacking creatures get +1/+0 until
    end of turn" (Contested War Zone). Attackers are always the
    active player's, so in practice this picks up your own attackers
    during your turn.

    ``also_untap_if_not_your_turn=True`` models the Zealous Display
    rider — "If it's not your turn, untap those creatures." When the
    spell resolves on an opponent's turn, every creature that received
    the pump is also untapped, letting them block (or, with vigilance/
    haste, attack again next turn fresh).
    """

    power_bonus: int = 0
    toughness_bonus: int = 0
    granted_keywords: frozenset[str] = frozenset()
    exclude_source: bool = False
    attacking_only: bool = False
    also_untap_if_not_your_turn: bool = False

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source if self.exclude_source else None
        untap_rider_active = (
            self.also_untap_if_not_your_turn
            and state.players[state.active_player_idx].id != controller_id
        )
        count = 0
        untapped_count = 0
        for obj in state.battlefield:
            if self.attacking_only:
                if not obj.attacking:
                    continue
            else:
                if obj.controller_id != controller_id:
                    continue
                if not obj.printing.is_creature:
                    continue
            if source is not None and obj is source:
                continue
            if self.power_bonus or self.toughness_bonus:
                src = state.resolving_source
                state.grant_pt_pump_ueot(
                    source_id=src.id if src is not None else obj.id,
                    target_id=obj.id,
                    power=self.power_bonus, toughness=self.toughness_bonus,
                    description=(
                        f"{src.printing.name if src else obj.printing.name} "
                        f"pumps {obj.printing.name} "
                        f"+{self.power_bonus}/+{self.toughness_bonus} UEOT"
                    ),
                )
            if self.granted_keywords:
                src = state.resolving_source
                state.grant_keywords_ueot(
                    source_id=src.id if src is not None else obj.id,
                    target_id=obj.id,
                    keywords=self.granted_keywords,
                    log_source_name=(
                        src.printing.name if src is not None
                        else obj.printing.name
                    ),
                    log_target_name=obj.printing.name,
                )
            if untap_rider_active and obj.tapped:
                obj.tapped = False
                untapped_count += 1
            count += 1
        if self.attacking_only:
            scope = "attacking creatures"
        elif self.exclude_source:
            scope = "other creatures"
        else:
            scope = "creatures"
        state.log.append(
            f"P{controller_id}'s {scope} ({count}) get "
            f"+{self.power_bonus}/+{self.toughness_bonus} until end of turn"
        )
        if untap_rider_active and untapped_count:
            state.log.append(
                f"P{controller_id}'s {scope} ({untapped_count}) untap "
                f"(not P{controller_id}'s turn)"
            )

    def describe(self) -> str:
        if self.attacking_only:
            scope = "attacking creatures"
        elif self.exclude_source:
            scope = "other creatures you control"
        else:
            scope = "creatures you control"
        return (
            f"{scope} get "
            f"+{self.power_bonus}/+{self.toughness_bonus} until end of turn"
        )



@dataclass(frozen=True)
class PumpCreaturesOpponentsControlUntilEOT(Effect):
    """Each creature an opponent of the controller controls gets
    +power/+toughness until end of turn (applied at resolution;
    creatures entering later don't get the bonus — Stage-3 snapshot vs.
    a true continuous effect).

    Negative bonuses model debuffs like "creatures your opponents
    control get -3/-0 until end of turn" (one-time snapshot).

    Set ``until_caster_next_turn=True`` for the "until your next turn"
    variant (Mouth of the Storm). The bonuses are written to the
    ``extended_temp_*`` fields on each affected GameObject and tagged
    with the resolving controller's id. They DON'T clear at the
    casting turn's cleanup; instead they expire at the start
    (``run_untap_step``) of that controller's NEXT turn, so the
    opponent's full subsequent turn keeps the debuff applied.
    """

    power_bonus: int = 0
    toughness_bonus: int = 0
    granted_keywords: frozenset[str] = frozenset()
    until_caster_next_turn: bool = False

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        count = 0
        for obj in state.battlefield:
            if obj.controller_id == controller_id:
                continue
            if not obj.printing.is_creature:
                continue
            if self.until_caster_next_turn:
                src = state.resolving_source
                if self.power_bonus or self.toughness_bonus:
                    state.grant_pt_pump_until_player_next_turn(
                        source_id=src.id if src is not None else obj.id,
                        target_id=obj.id,
                        power=self.power_bonus,
                        toughness=self.toughness_bonus,
                        until_player_id=controller_id,
                        description=(
                            f"opp creature {obj.printing.name} pumped "
                            f"{self.power_bonus:+d}/{self.toughness_bonus:+d} "
                            f"until P{controller_id}'s next turn"
                        ),
                    )
                if self.granted_keywords:
                    # Extended-temporal keywords still flow through
                    # the bespoke field path; the layered registry
                    # currently models only P/T pumps for this
                    # lifetime. Future work can mirror grant_keywords
                    # under expires_at="controller_untap".
                    obj.extended_temp_keywords = (
                        obj.extended_temp_keywords | self.granted_keywords
                    )
                    obj.extended_temp_expires_at_player_id = controller_id
            else:
                if self.power_bonus or self.toughness_bonus:
                    src = state.resolving_source
                    state.grant_pt_pump_ueot(
                        source_id=src.id if src is not None else obj.id,
                        target_id=obj.id,
                        power=self.power_bonus,
                        toughness=self.toughness_bonus,
                        description=(
                            f"opp creature {obj.printing.name} pumped "
                            f"{self.power_bonus:+d}/{self.toughness_bonus:+d} UEOT"
                        ),
                    )
                if self.granted_keywords:
                    src = state.resolving_source
                    state.grant_keywords_ueot(
                        source_id=src.id if src is not None else obj.id,
                        target_id=obj.id,
                        keywords=self.granted_keywords,
                        log_source_name=(
                            src.printing.name if src is not None
                            else obj.printing.name
                        ),
                        log_target_name=obj.printing.name,
                    )
            count += 1
        # Format signed bonuses ("+3"/"-3" rather than "+-3").
        p_str = f"{self.power_bonus:+d}"
        t_str = f"{self.toughness_bonus:+d}"
        scope = (
            f"until P{controller_id}'s next turn"
            if self.until_caster_next_turn
            else "until end of turn"
        )
        state.log.append(
            f"P{controller_id}'s opponents' creatures ({count}) get "
            f"{p_str}/{t_str} {scope}"
        )

    def describe(self) -> str:
        p_str = f"{self.power_bonus:+d}"
        t_str = f"{self.toughness_bonus:+d}"
        scope = (
            "until your next turn"
            if self.until_caster_next_turn
            else "until end of turn"
        )
        return (
            f"creatures your opponents control get {p_str}/{t_str} "
            f"{scope}"
        )



@dataclass(frozen=True)
class SearchLibraryForCard(Effect):
    """Search your library for a card matching the named filter, reveal
    it, put it into your hand, then shuffle.

    ``filter_name`` selects the filter applied to cards in the library:
      - ``"legendary_creature"`` — creature with Legendary supertype
      - ``"basic_plains"`` / ``"basic_island"`` / ... — specific basic
      - ``"creature_small"`` — creature with mana value ≤ ``max_mv``

    If ``may_fail`` is True, the controller may decline to fetch
    (shuffle happens either way). Always shuffles at the end — matches
    MTG wording.
    """

    filter_name: str
    max_mv: int = 0
    may_fail: bool = True

    def _matches(self, printing) -> bool:
        fn = self.filter_name
        if fn == "legendary_creature":
            return (
                printing.is_creature and "Legendary" in printing.supertypes
            )
        # Generic ``basic land card`` (any of the five basics) — Spineseeker
        # Centipede's ETB tutor. Distinct from the per-color
        # ``basic_plains`` / ``basic_island`` / ... filters which pin a
        # single basic by name.
        if fn == "basic_land":
            return printing.is_basic_land
        if fn.startswith("basic_"):
            wanted = fn[len("basic_"):].capitalize()
            return printing.is_basic_land and printing.name == wanted
        if fn == "creature_small":
            return (
                printing.is_creature
                and printing.mana_cost.total <= self.max_mv
            )
        if fn == "artifact":
            return printing.is_artifact
        if fn == "spacecraft":
            return printing.is_spacecraft
        if fn == "sliver":
            return "Sliver" in printing.subtypes
        # Demonic Counsel (Duskmourn) — tutor for a Demon creature
        # card. Mirrors the ``sliver`` shape (subtype-keyed filter on
        # creature printings); adding the filter here is purely
        # additive and reuses the existing tutor-via-subtype
        # convention.
        if fn == "demon":
            return "Demon" in printing.subtypes
        return False

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone, reveal_to_all, shuffle_library

        p = state.player_by_id(controller_id)
        options = [c for c in p.library if self._matches(c.printing)]
        if not options:
            state.log.append(
                f"P{controller_id} searches library — no "
                f"{self.filter_name} found"
            )
            shuffle_library(state, controller_id)
            return
        chosen_id = None
        if p.agent is not None:
            chosen_id = p.agent.choose_card_from_library(
                state, controller_id, [c.id for c in options]
            )
        if chosen_id is None:
            if self.may_fail:
                state.log.append(
                    f"P{controller_id} declines to fetch"
                )
                shuffle_library(state, controller_id)
                return
            chosen_id = options[0].id
        card = next((c for c in options if c.id == chosen_id), options[0])
        p.library.remove(card)
        card.zone = Zone.HAND
        p.hand.append(card)
        state.log.append(
            f"P{controller_id} fetches {card.printing.name} to hand"
        )
        reveal_to_all(state, card)
        shuffle_library(state, controller_id)

    def describe(self) -> str:
        return (
            f"search your library for a {self.filter_name} card, reveal "
            f"it, put it into your hand, then shuffle"
        )



@dataclass(frozen=True)
class SearchLibraryForBasicLandTapped(Effect):
    """Search the controller's library for up to ``count`` basic land
    cards, put them onto the battlefield tapped, then shuffle.

    Stage 3 simplification: picks the FIRST ``count`` basic lands in
    library order (no extra shuffle). If fewer basic lands exist,
    puts what it finds.
    """

    count: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone
        from mtg_bot.turn import fire_landfall_triggers

        p = state.player_by_id(controller_id)
        for _ in range(self.count):
            land = next(
                (c for c in p.library if c.printing.is_basic_land), None,
            )
            if land is None:
                state.log.append(
                    f"P{controller_id} searches but finds no more basic lands"
                )
                return
            p.library.remove(land)
            land.zone = Zone.BATTLEFIELD
            land.tapped = True
            land.controller_id = controller_id
            state.battlefield.append(land)
            state.log.append(
                f"P{controller_id} puts {land.printing.name} onto the "
                f"battlefield tapped"
            )
            # ``fire_landfall_triggers`` covers both the legacy
            # string-key path and the predicate-style ENTERS dispatch.
            fire_landfall_triggers(state, land)

    def describe(self) -> str:
        if self.count == 1:
            return (
                "search your library for a basic land card, put it onto "
                "the battlefield tapped, then shuffle"
            )
        return (
            f"search your library for up to {self.count} basic land "
            f"cards, put them onto the battlefield tapped, then shuffle"
        )



@dataclass(frozen=True)
class SearchLibraryForEquipmentToBattlefield(Effect):
    """Search the controller's library for an Equipment card, put it
    onto the battlefield (untapped, under the controller's control),
    then shuffle.

    Models Honored Knight-Captain's activated tutor: ``{4}{W}{W},
    Sacrifice this creature: Search your library for an Equipment card,
    put it onto the battlefield, then shuffle.``

    The agent picks which Equipment via ``choose_card_from_library``.
    If no Equipment exists in library, the search fails and a shuffle
    still happens. The card enters via ``put_into_battlefield`` so any
    ETB triggers (e.g., the auto-attach riders on EoE Equipment) fire
    normally.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.game import shuffle_library

        p = state.player_by_id(controller_id)
        options = [
            c for c in p.library
            if "Equipment" in c.printing.subtypes
        ]
        if not options:
            state.log.append(
                f"P{controller_id} searches but finds no Equipment"
            )
            shuffle_library(state, controller_id)
            return
        chosen_id = None
        if p.agent is not None:
            chosen_id = p.agent.choose_card_from_library(
                state, controller_id, [c.id for c in options]
            )
        card = next(
            (c for c in options if c.id == chosen_id), options[0],
        )
        p.library.remove(card)
        state.log.append(
            f"P{controller_id} fetches {card.printing.name} to "
            f"battlefield (Honored Knight-Captain)"
        )
        put_into_battlefield(state, card, controller_id)
        shuffle_library(state, controller_id)

    def describe(self) -> str:
        return (
            "search your library for an Equipment card, put it onto "
            "the battlefield, then shuffle"
        )



@dataclass(frozen=True)
class DealDamageEachOpponent(Effect):
    """Deal ``amount`` damage to each living opponent. Untargeted — mass
    burn half of Ruinous Rampage's choice.
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            p.life -= self.amount
            state.log.append(
                f"P{p.id} takes {self.amount} (life: {p.life})"
            )

    def describe(self) -> str:
        return f"deals {self.amount} damage to each opponent"



@dataclass(frozen=True)
class ExileAllArtifactsWithManaValueAtMost(Effect):
    """Exile every artifact on the battlefield with mana value ≤ ``max_mv``.

    Ruinous Rampage's anti-artifact mode. Lander tokens (MV 0) are hit.
    """

    max_mv: int = 3

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import move_off_battlefield

        victims = [
            o for o in state.battlefield
            if o.printing.is_artifact and o.printing.mana_cost.total <= self.max_mv
        ]
        for obj in victims:
            state.log.append(f"{obj.printing.name} is exiled (mass-artifact)")
            move_off_battlefield(state, obj, destination="exile")

    def describe(self) -> str:
        return (
            f"exile all artifacts with mana value {self.max_mv} or less"
        )



@dataclass(frozen=True)
class CreateToken(Effect):
    """Create ``count`` creature tokens of the given printing under the
    ability's controller.

    ``token_printing`` must be a CardPrinting with the Token supertype (built
    via ``cards.make_token_creature``). Tokens enter the battlefield through
    the same helper as spells, so any ETBs they themselves have fire
    correctly.

    Set ``tapped=True`` for effects whose oracle text says "enters tapped"
    (e.g. Gravpack Monoist's "create a tapped 2/2 Robot token").
    """

    token_printing: "CardPrinting"
    count: int = 1
    tapped: bool = False
    each_player: bool = False  # if True, every living player creates the token

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        import dataclasses as _dc
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.game import Zone

        if self.each_player:
            # Every living player creates the token(s) under their own control.
            recipients = [p.id for p in state.players if not p.has_lost]
        else:
            recipients = [controller_id]

        for recipient in recipients:
            # Moonlit Meditation replacement: "The first time you would
            # create one or more tokens each turn, you may instead
            # create that many tokens that are copies of enchanted
            # permanent." Use enchanted-permanent's printing (with
            # Token supertype added) for this batch if applicable.
            printing = self.token_printing
            moonlit = None
            if (
                recipient not in state.players_who_created_tokens_this_turn
            ):
                moonlit = _find_moonlit_replacement_source(state, recipient)
            if moonlit is not None:
                enchanted = next(
                    (
                        o for o in state.battlefield
                        if o.id == moonlit.attached_to
                    ),
                    None,
                )
                if enchanted is not None:
                    base = enchanted.printing
                    printing = _dc.replace(
                        base, supertypes=tuple(
                            list(base.supertypes)
                            + (["Token"] if "Token" not in base.supertypes else [])
                        ),
                    )
                    state.log.append(
                        f"Moonlit Meditation: tokens become copies of "
                        f"{base.name}"
                    )
            # CR 614 token-creation replacement effects (Exalted
            # Sunborn ×2, future Anointed Procession ×3, …) flow
            # through the unified ``dispatch_replacement``. The
            # dispatcher walks the battlefield collecting matching
            # ``ReplacementEffect`` instances (currently synthesised
            # from the legacy ``CardPrinting.is_token_doubler`` flag;
            # native declarations on each printing's static abilities
            # come in PR D), applies them in CR 616.1 order, and
            # returns the post-replacement count.
            from mtg_bot.effects.modifiers import dispatch_replacement
            from mtg_bot.effects.predicates import Event as _Event
            replaced = dispatch_replacement(
                state, _Event.WOULD_CREATE_TOKEN,
                affected_player_id=recipient,
                subjects={
                    "controller": state.player_by_id(recipient),
                    "kind": printing.name,
                    # ``controller_id`` is read by the legacy
                    # ``is_token_doubler`` synthesizer to gate the
                    # doubler to the recipient's own permanents.
                    "controller_id": recipient,
                },
                payload={"count": self.count},
            )
            if replaced.get("prevented"):
                continue
            effective_count = replaced.get("count", 0)
            doubled = effective_count != self.count
            for _ in range(effective_count):
                obj = state.new_object(
                    printing, owner_id=recipient, zone=Zone.BATTLEFIELD,
                )
                state.log.append(
                    f"P{recipient} creates a {printing.name} token"
                    + (" (tapped)" if self.tapped else "")
                    + (
                        f" (replaced x{effective_count}/{self.count})"
                        if doubled else ""
                    )
                )
                put_into_battlefield(state, obj, recipient)
                if self.tapped:
                    obj.tapped = True
            if effective_count > 0:
                state.players_who_created_tokens_this_turn.add(recipient)

    def describe(self) -> str:
        p = self.token_printing
        color_word = ""  # we don't track color on tokens explicitly yet
        pt = f"{p.power}/{p.toughness}" if p.is_creature else ""
        subtype = " ".join(p.subtypes)
        if self.count == 1:
            article = "a"
        else:
            article = str(self.count)
        pieces = [article, pt, color_word, subtype, "creature token"]
        return "create " + " ".join(x for x in pieces if x)



@dataclass(frozen=True)
class DestroyTargetCreature(Destroy):
    """``Destroy`` constructor sugar for "destroy target creature" with
    flag-driven filter widening and a Faller's-Faithful-style draw
    rider.

    Filter composition:
      - base: ``IsCreature()``
      - ``or_spacecraft=True``: union with ``HasSubtype("Spacecraft")``
        (Embrace Oblivion's "creature or Spacecraft")
      - ``or_enchantment=True``: union with ``IsEnchantment()``
        (Withering Torment's "creature or enchantment")
      - ``exclude_self=True``: intersect with ``~IsSource()`` so the
        trigger source isn't its own legal target ("another target")

    Rider: ``draw_if_undamaged_amount > 0`` adds a "if it wasn't dealt
    damage this turn, its controller draws N" follow-up — Faller's
    Faithful. The check snapshots ``obj.damaged_this_turn`` BEFORE the
    destroy because leaving the battlefield clears the flag. The
    drawing player is the destroyed creature's controller per the
    printed text. The custom resolve overrides ``Destroy.resolve``
    so the rider runs alongside the destruction; future migration
    will lift the rider into a sibling effect with a ``SlotRef`` back-
    reference once the EffectSequence consumer pillar lands.

    ``optional=True`` flips the slot's ``count_min`` to 0 ("up to one
    other target" / Faller's Faithful's optional shape).
    """

    draw_if_undamaged_amount: int = 0
    or_spacecraft: bool = False
    or_enchantment: bool = False
    optional: bool = False
    exclude_self: bool = False

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            HasSubtype, IsCreature, IsEnchantment, IsSource,
        )
        from mtg_bot.effects.target_filter import OneOf
        from mtg_bot.effects.targets import TargetSpec
        arms: list = [IsCreature()]
        if self.or_spacecraft:
            arms.append(HasSubtype("Spacecraft"))
        if self.or_enchantment:
            arms.append(IsEnchantment())
        f = arms[0] if len(arms) == 1 else OneOf(*arms)
        if self.exclude_self:
            f = f & ~IsSource()
        spec = TargetSpec(
            f,
            count_min=0 if self.optional else 1,
            count_max=1,
        )
        object.__setattr__(self, "targets", (spec,))

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        from mtg_bot.turn import destroy, draw_n_cards

        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            for slot, chosen in zip(self.targets, targets):
                if isinstance(slot, SlotRef):
                    continue
                if not isinstance(slot, TargetSpec):
                    continue
                if not _slot_target_still_legal(state, slot, chosen):
                    state.log.append(
                        _fizzle_message(state, chosen, action="destroy")
                    )
                    continue
                _, target_id = chosen
                obj = next(
                    (o for o in state.battlefield if o.id == target_id),
                    None,
                )
                if obj is None:
                    continue
                was_undamaged = not obj.damaged_this_turn
                victim_controller_id = obj.controller_id
                state.log.append(f"{obj.printing.name} is destroyed")
                destroy(state, obj)
                if self.draw_if_undamaged_amount > 0 and was_undamaged:
                    draw_n_cards(
                        state, victim_controller_id,
                        self.draw_if_undamaged_amount,
                    )
                    state.log.append(
                        f"P{victim_controller_id} draws "
                        f"{self.draw_if_undamaged_amount} "
                        f"(target wasn't dealt damage this turn)"
                    )
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        suffix_parts: list[str] = []
        if self.or_spacecraft:
            suffix_parts.append("Spacecraft")
        if self.or_enchantment:
            suffix_parts.append("enchantment")
        suffix = (" or " + " or ".join(suffix_parts)) if suffix_parts else ""
        base = f"destroy target creature{suffix}"
        if self.draw_if_undamaged_amount > 0:
            return (
                f"{base}; if it wasn't dealt damage this turn, its "
                f"controller draws {self.draw_if_undamaged_amount} card(s)"
            )
        return base



@dataclass(frozen=True)
class DestroyEnchantedPermanent(Effect):
    """Destroy the permanent enchanted by the resolving Aura.

    Untargeted: the source is the aura (read from
    ``state.resolving_source``), and the affected permanent is whatever
    that aura is currently attached to. Used by Cryoshatter's
    "when enchanted creature becomes tapped, destroy it" trigger.

    If the aura is no longer on the battlefield or has no host (e.g.,
    the host already left), the effect fizzles.
    """

    def requires_target(self) -> bool:
        return False

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import destroy

        aura = state.resolving_source
        if aura is None or aura.attached_to is None:
            return
        host = next(
            (o for o in state.battlefield if o.id == aura.attached_to),
            None,
        )
        if host is None:
            return
        state.log.append(
            f"{host.printing.name} is destroyed by {aura.printing.name}"
        )
        destroy(state, host)

    def describe(self) -> str:
        return "destroy enchanted permanent"



@dataclass(frozen=True)
class DestroyTargetArtifact(Destroy):
    """``Destroy`` constructor sugar for "destroy target artifact".

    Filter is :class:`IsArtifact`. The unified ``Destroy.resolve``
    re-verifies the slot's filter at resolution per CR 608.2b — a
    target that lost the Artifact card type between cast and resolve
    fails the recheck and the destruction is skipped.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsArtifact
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(self, "targets", (TargetSpec(IsArtifact()),))



@dataclass(frozen=True)
class ExileTargetCreatureUntilSourceLeaves(Effect):
    """Exile any target creature until the source leaves the battlefield.

    Matches All-Fates Stalker's ETB and Hardlight Containment's ETB.
    Uses the Oblivion-Ring-style ``exiled_by_source`` bookkeeping so
    the exiled creature returns when the source leaves play.

    ``excluded_subtypes`` filters out creatures whose printed subtypes
    overlap with any member (e.g., ``frozenset({"Assassin"})`` for
    All-Fates Stalker's "non-Assassin creature" clause). Empty = no
    subtype restriction.

    ``opponents_only`` restricts legal targets to creatures controlled
    by a player OTHER than the effect's controller (Hardlight
    Containment: "target creature an opponent controls"). Default False.

    ``yours_only`` restricts legal targets to creatures the effect's
    controller controls (Endless Sands: "target creature you control").
    Mutually exclusive with ``opponents_only``.
    """

    excluded_subtypes: frozenset[str] = frozenset()
    opponents_only: bool = False
    yours_only: bool = False

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and not (
                self.excluded_subtypes
                and any(
                    s in self.excluded_subtypes
                    for s in _eff_subtypes(state, o)
                )
            )
            and not (
                self.opponents_only and o.controller_id == controller_id
            )
            and not (
                self.yours_only and o.controller_id != controller_id
            )
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        from mtg_bot.turn import move_off_battlefield

        obj.exiled_by_source = source.id
        state.log.append(
            f"{obj.printing.name} is exiled by {source.printing.name}"
        )
        move_off_battlefield(state, obj, destination="exile")

    def describe(self) -> str:
        return (
            "exile target creature until this creature leaves the "
            "battlefield"
        )



@dataclass(frozen=True)
class ExileOpponentsNonlandPermanentUntilLeaves(Effect):
    """ETB exile target nonland permanent an opponent controls, optionally
    restricted by mana value. The exiled card returns when the source
    enchantment leaves the battlefield (Oblivion Ring / Banishing Light
    pattern). Uses state.resolving_source as the "source" for tracking.

    ``max_mv`` = 0 means no MV filter (Banishing Light); > 0 restricts
    to permanents with mana value ≤ max_mv (Seam Rip).
    """

    max_mv: int = 0

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        out: list[Target] = []
        for o in state.battlefield:
            if o.printing.is_land:
                continue
            if o.controller_id == controller_id:
                continue
            if self.max_mv and o.printing.mana_cost.total > self.max_mv:
                continue
            out.append(("permanent", o.id))
        return out

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None or obj.printing.is_land:
            return
        if obj.controller_id == controller_id:
            return
        if self.max_mv and obj.printing.mana_cost.total > self.max_mv:
            return
        from mtg_bot.turn import move_off_battlefield

        obj.exiled_by_source = source.id
        state.log.append(
            f"{obj.printing.name} is exiled by {source.printing.name}"
        )
        move_off_battlefield(state, obj, destination="exile")

    def describe(self) -> str:
        if self.max_mv:
            return (
                f"exile target nonland permanent an opponent controls "
                f"with mana value {self.max_mv} or less until this "
                f"enchantment leaves the battlefield"
            )
        return (
            "exile target nonland permanent an opponent controls until "
            "this enchantment leaves the battlefield"
        )



@dataclass(frozen=True)
class DestroyTargetNoncreaturePermanent(Destroy):
    """``Destroy`` constructor sugar for "destroy target noncreature
    permanent".

    Filter is ``~IsCreature() & ~IsPlayer()``. ``IsCreature`` reads
    ``is_creature_now``, so an Impending-suppressed Overlord (which
    is currently an enchantment, not a creature) IS a legal target —
    matches the legacy ``not is_creature_now(o)`` enumeration.
    Players are excluded by the ``~IsPlayer()`` arm.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsCreature, IsPlayer
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(~IsCreature() & ~IsPlayer()),),
        )

    def describe(self) -> str:
        return "destroy target noncreature permanent"



@dataclass(frozen=True)
class BecomeLandCreatureUEOT(Effect):
    """Animate a land into a creature until end of turn.

    The shared manland primitive: sets temp_is_creature, overrides
    the source's base P/T via temp_base_pt_override, and grants any
    keywords via temp_keywords. Also optionally sets temp_unblockable
    (Creeping Tar Pit: "It can't be blocked this turn").

    Per-manland abilities granted "while animated" are folded in via
    ``granted_activated_abilities`` (Lavaclaw Reaches' X-pump, Wandering
    Fumarole's P/T swap) and ``granted_triggered_abilities`` (Raging
    Ravine's "whenever this attacks, +1/+1 counter"). Both clear at
    cleanup with the rest of the temp_* fields.
    """

    power: int = 0
    toughness: int = 0
    granted_keywords: frozenset[str] = frozenset()
    unblockable: bool = False
    # Subtype the animated creature gains UEOT (e.g., "Blinkmoth" for
    # Blinkmoth Nexus, "Elemental" for Lavaclaw Reaches). Stored on
    # ``GameObject.temp_subtype`` and cleared at cleanup. Read by
    # target filters (e.g. PumpTargetCreatureUntilEOT.target_subtype).
    granted_subtype: str | None = None
    # Activated / triggered abilities granted UEOT to the source while
    # animated. Appended to ``temp_activated_abilities`` /
    # ``temp_triggered_abilities``; cleared at cleanup. Used by manlands
    # whose printed text grants per-card abilities only while the land
    # is a creature (Lavaclaw Reaches, Wandering Fumarole, Raging Ravine).
    granted_activated_abilities: tuple = ()
    granted_triggered_abilities: tuple = ()

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        state.grant_become_creature_ueot(
            source_id=source.id,
            target_id=source.id,
            power=self.power,
            toughness=self.toughness,
            subtype=self.granted_subtype,
            description=(
                f"{source.printing.name} becomes a "
                f"{self.power}/{self.toughness} creature UEOT"
            ),
        )
        if self.granted_keywords:
            state.grant_keywords_ueot(
                source_id=source.id,
                target_id=source.id,
                keywords=self.granted_keywords,
                log_source_name=source.printing.name,
                log_target_name=source.printing.name,
            )
        if self.unblockable:
            source.temp_unblockable = True
        for ab in self.granted_activated_abilities:
            state.grant_activated_ability_ueot(
                source_id=source.id,
                target_id=source.id,
                ability=ab,
                description=(
                    f"{source.printing.name} grants activated ability "
                    f"UEOT to itself (manland animation)"
                ),
            )
        for ab in self.granted_triggered_abilities:
            state.grant_triggered_ability_ueot(
                source_id=source.id,
                target_id=source.id,
                ability=ab,
                description=(
                    f"{source.printing.name} grants triggered ability "
                    f"UEOT to itself (manland animation)"
                ),
            )
        kw_note = (
            f" with {', '.join(sorted(self.granted_keywords))}"
            if self.granted_keywords else ""
        )
        state.log.append(
            f"{source.printing.name} becomes a {self.power}/{self.toughness}"
            f" creature UEOT{kw_note}"
        )

    def describe(self) -> str:
        kws = (
            " with " + ", ".join(sorted(self.granted_keywords))
            if self.granted_keywords else ""
        )
        unb = " (unblockable)" if self.unblockable else ""
        return (
            f"until end of turn, this land becomes a "
            f"{self.power}/{self.toughness} creature{kws}{unb}; "
            f"it's still a land"
        )


@dataclass(frozen=True)
class PumpSelfXPowerUEOT(Effect):
    """Source gets +X/+0 until end of turn, where X comes from
    ``state.resolving_x_value`` (set by the X-cost activation pipeline
    in ``_activate_ability``).

    Wires Lavaclaw Reaches' granted "{X}: This creature gets +X/+0 UEOT"
    while animated.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        x = state.resolving_x_value
        if x <= 0:
            return
        source.temp_power_bonus += x
        state.log.append(
            f"{source.printing.name} gets +{x}/+0 until end of turn"
        )

    def describe(self) -> str:
        return "this creature gets +X/+0 until end of turn"


@dataclass(frozen=True)
class SwitchSourcePowerToughnessUEOT(Effect):
    """Switch the source's power and toughness until end of turn.

    Implementation: writes a layer-7 sublayer-4 ``"switch_pt"``
    registry entry sourced by the source object with
    ``expires_at="cleanup"``. The entry carries no params — each
    such entry contributes a single swap, walked in timestamp order
    by ``_compute_effective_pt``. Two switches cancel; three
    switches is one swap.

    Per CR 613.4 layer 7e, switch operates on the post-7b/c/d
    effective P/T — so a Lignified manland (0/4 from layer 7b) with
    a +1/+1 counter (7d) becomes 1/5 before any switch, and 5/1
    after one switch. That matches MTG's "switch power and
    toughness" semantics on a creature whose P/T has already been
    modified by other effects.

    Wires Wandering Fumarole's granted "{0}: Switch this creature's
    power and toughness until end of turn" while animated. The {0}
    cost means the ability is freely re-activatable; each activation
    appends another switch entry, toggling the displayed P/T.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        state.add_continuous_effect(
            source_id=source.id,
            layer=7, sublayer=4, kind="switch_pt",
            target_id=source.id,
            params=(),
            expires_at="cleanup",
            description=(
                f"{source.printing.name} P/T switched UEOT"
            ),
        )
        state.log.append(
            f"{source.printing.name}'s power and toughness are switched "
            f"until end of turn"
        )

    def describe(self) -> str:
        return (
            "switch this creature's power and toughness until end of turn"
        )



@dataclass(frozen=True)
class BecomesCopyOfTargetLand(Effect):
    """Thespian's Stage: "This land becomes a copy of target land,
    except it has this ability." Continuous effect — persists until the
    source leaves play or another copy effect overrides.

    Implementation: mutates ``source.printing`` to point at a fresh
    CardPrinting derived from the target's, with the source's own
    activated abilities appended (so "it has this ability" is
    preserved across repeated re-copies).

    Simplifications:
    - Re-copying from the newly-mutated printing still re-derives
      "retained own abilities" from ``source.original_printing``, which
      is snapshotted on the first resolve. Prevents accidental
      ability accumulation across multiple copies.
    - Copying a token target (rare but possible) carries the token
      supertype through unchanged; the source itself doesn't become a
      token.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.printing.is_land
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        import dataclasses as _dc

        source = state.resolving_source
        if source is None or not targets:
            return
        _, tid = targets[0]
        target = next(
            (o for o in state.battlefield if o.id == tid), None,
        )
        if target is None:
            return
        # Snapshot the original printing the first time we copy — so
        # re-copy activations don't lose "this ability".
        if source.original_printing is None:
            source.original_printing = source.printing
        retained = source.original_printing.activated_abilities
        source.printing = _dc.replace(
            target.printing,
            activated_abilities=tuple(target.printing.activated_abilities)
            + tuple(retained),
        )
        state.log.append(
            f"{source.original_printing.name} becomes a copy of "
            f"{target.printing.name}"
        )

    def describe(self) -> str:
        return (
            "this land becomes a copy of target land, except it has "
            "this ability"
        )



@dataclass(frozen=True)
class CreateTokenCopyOfTargetCreatureYouControl(Effect):
    """Mirrorpool mode 2: "Create a token that's a copy of target
    creature you control."

    The token shares the target's printing with the Token supertype
    injected. Uses the existing CreateToken machinery so doublers and
    ETB triggers work normally.

    ``additional_subtype`` (default ``None``) injects a fixed extra
    creature subtype into the token's printing, on top of the
    target's printed subtypes. Matches Mirror Room's "create a token
    that's a copy of target creature you control, except it's a
    Reflection in addition to its other creature types" — the
    Reflection subtype is added to the copy's subtype tuple so
    type-line, oracle and any HasSubtype predicate sees the merged
    list. Idempotent: if the target's printing already lists the
    subtype, the injection is a no-op.
    """

    additional_subtype: "str | None" = None

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        from mtg_bot.game import is_creature_now
        return [
            ("creature", o.id)
            for o in state.battlefield
            if is_creature_now(o) and o.controller_id == controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        import dataclasses as _dc

        if not targets:
            return
        _, tid = targets[0]
        target = next(
            (o for o in state.battlefield if o.id == tid), None,
        )
        if target is None:
            return
        base = target.printing
        # Inject Token supertype if not already present.
        new_supertypes = tuple(
            list(base.supertypes)
            + (["Token"] if "Token" not in base.supertypes else [])
        )
        # Inject the additional creature subtype, if requested and
        # not already present on the target's printing — Mirror Room
        # adds "Reflection" to whatever the target already is.
        if (
            self.additional_subtype is not None
            and self.additional_subtype not in base.subtypes
        ):
            new_subtypes = tuple(
                list(base.subtypes) + [self.additional_subtype]
            )
        else:
            new_subtypes = base.subtypes
        token_printing = _dc.replace(
            base,
            supertypes=new_supertypes,
            subtypes=new_subtypes,
        )
        CreateToken(
            token_printing=token_printing, count=1,
        ).resolve(state, controller_id, ())

    def describe(self) -> str:
        if self.additional_subtype is not None:
            return (
                f"create a token that's a copy of target creature you "
                f"control, except it's a {self.additional_subtype} in "
                f"addition to its other creature types"
            )
        return (
            "create a token that's a copy of target creature you control"
        )



@dataclass(frozen=True)
class CopyTargetInstantOrSorceryYouControl(Effect):
    """Mirrorpool mode 1: "Copy target instant or sorcery spell you
    control. You may choose new targets for the copy."

    Simplification: rather than pushing a real stack copy (which would
    require mid-resolution target selection — not supported by the
    engine's cast-time target pipeline), we short-circuit by re-
    resolving the target spell's ``spell_effect`` with freshly-chosen
    targets. Functionally equivalent for anything our engine models.
    Close enough for flavor; counterspell-proofing isn't modeled.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        # Target key is the spell source's object id, NOT the stack
        # index — the cost-payment pipeline filters targets whose id
        # matches the ability source (``sacrifice_self``), so stack
        # indices (0, 1, …) would collide with low object ids.
        out: list[Target] = []
        for item in state.stack:
            if item.controller_id != controller_id:
                continue
            src = item.source
            if src is None:
                continue
            if not (src.printing.is_instant or src.printing.is_sorcery):
                continue
            out.append(("stack_spell", src.id))
        return out

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, src_id = targets[0]
        if kind != "stack_spell":
            return
        item = next(
            (
                i for i in state.stack
                if i.source is not None and i.source.id == src_id
            ),
            None,
        )
        if item is None:
            return
        src = item.source
        if src is None:
            return
        spell_effect = src.printing.spell_effect
        if spell_effect is None:
            return
        # Choose fresh targets for the copy.
        copy_targets: tuple[Target, ...] = ()
        if spell_effect.requires_target():
            legal = spell_effect.legal_targets(state, controller_id)
            if legal:
                p = state.player_by_id(controller_id)
                if p.agent is not None:
                    chosen = p.agent.choose_target(
                        state, controller_id, legal
                    )
                    if chosen is not None:
                        copy_targets = (chosen,)
                if not copy_targets:
                    copy_targets = (legal[0],)
        state.log.append(
            f"Mirrorpool copies {src.printing.name}"
        )
        # Short-circuit: resolve the copy by re-running the spell
        # effect with the freshly-chosen target.
        prior = state.resolving_source
        state.resolving_source = src
        try:
            spell_effect.resolve(state, controller_id, copy_targets)
        finally:
            state.resolving_source = prior

    def describe(self) -> str:
        return (
            "copy target instant or sorcery spell you control; you may "
            "choose new targets for the copy"
        )



@dataclass(frozen=True)
class UntapTargetLand(Effect):
    """Deserted Temple: `{1}, {T}: Untap target land.`"""

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.printing.is_land and o.tapped
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None or not obj.printing.is_land:
            return
        if not obj.tapped:
            return
        obj.tapped = False
        state.log.append(f"{obj.printing.name} is untapped")

    def describe(self) -> str:
        return "untap target land"



@dataclass(frozen=True)
class SacrificeNLandsYouControl(Effect):
    """Sacrifice ``count`` lands the effect's controller controls.

    Matches Lotus Field's "When this land enters, sacrifice two lands."
    Excludes the source (the ETB trigger is on the entering land
    itself, which is already on the battlefield at trigger time;
    oracle allows sac'ing the source in principle, but the engine
    never models that as a desirable choice).

    Simplification: the agent's choose_sacrifice picks; if the agent
    is None we pick the first ``count`` legal lands. If fewer than
    ``count`` legal lands exist, sacrifices all available.
    """

    count: int = 2

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import sacrifice as _sacrifice

        source = state.resolving_source
        source_id = source.id if source is not None else -1
        p = state.player_by_id(controller_id)
        remaining = self.count
        while remaining > 0:
            options = [
                o for o in state.battlefield
                if o.controller_id == controller_id
                and o.printing.is_land
                and o.id != source_id
            ]
            if not options:
                break
            chosen = options[0]
            if p.agent is not None:
                pick_id = p.agent.choose_sacrifice(
                    state, controller_id, [o.id for o in options]
                )
                chosen = next(
                    (o for o in options if o.id == pick_id), options[0]
                )
            state.log.append(
                f"P{controller_id} sacrifices {chosen.printing.name} "
                f"(Lotus Field ETB)"
            )
            _sacrifice(state, chosen)
            remaining -= 1

    def describe(self) -> str:
        if self.count == 1:
            return "sacrifice a land"
        return f"sacrifice {self.count} lands"



@dataclass(frozen=True)
class ExileAllGraveyards(Effect):
    """Scavenger Grounds: `{2}, {T}, Sacrifice a Desert: Exile all
    graveyards.` Every card in every player's graveyard is moved to
    that player's exile.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        total = 0
        for p in state.players:
            for c in list(p.graveyard):
                p.graveyard.remove(c)
                c.zone = Zone.EXILE
                p.exile.append(c)
                total += 1
        state.log.append(f"all graveyards exiled ({total} cards)")

    def describe(self) -> str:
        return "exile all graveyards"



@dataclass(frozen=True)
class DestroyTargetLand(Destroy):
    """``Destroy`` constructor sugar for "destroy target land" /
    "destroy target nonbasic land".

    ``nonbasic_only=True`` narrows the filter from ``IsLand()`` to
    ``IsLand() & ~IsBasic()`` — Strip Mine's bare form vs Dust Bowl's
    nonbasic-restricted form. The unified ``Destroy.resolve`` re-
    verifies the slot's filter at resolution.
    """

    nonbasic_only: bool = False

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsBasic, IsLand
        from mtg_bot.effects.targets import TargetSpec
        f = IsLand() & ~IsBasic() if self.nonbasic_only else IsLand()
        object.__setattr__(self, "targets", (TargetSpec(f),))

    def describe(self) -> str:
        return (
            "destroy target nonbasic land"
            if self.nonbasic_only
            else "destroy target land"
        )



@dataclass(frozen=True)
class PutBasicLandFromHandOntoBattlefieldTapped(Effect):
    """Terrain Generator's {2}, {T} activated: "You may put a basic
    land card from your hand onto the battlefield tapped."
    Simplification: "may" always yes when a basic is in hand.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone
        from mtg_bot.turn import fire_landfall_triggers

        p = state.player_by_id(controller_id)
        basic = next(
            (
                c for c in p.hand
                if c.printing.is_land and "Basic" in c.printing.supertypes
            ),
            None,
        )
        if basic is None:
            return
        p.hand.remove(basic)
        basic.zone = Zone.BATTLEFIELD
        basic.controller_id = controller_id
        basic.tapped = True
        state.battlefield.append(basic)
        state.log.append(
            f"P{controller_id} puts {basic.printing.name} onto the "
            f"battlefield tapped (Terrain Generator)"
        )
        # ``fire_landfall_triggers`` covers both the legacy
        # string-key path and the predicate-style ENTERS dispatch.
        fire_landfall_triggers(state, basic)

    def describe(self) -> str:
        return (
            "you may put a basic land card from your hand onto the "
            "battlefield tapped"
        )



@dataclass(frozen=True)
class ReturnTargetLandFromGraveyardToHand(Effect):
    """Petrified Field's {T}, Sacrifice: Return target land card from
    your graveyard to your hand.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        p = state.player_by_id(controller_id)
        return [
            ("graveyard_land", c.id)
            for c in p.graveyard
            if c.printing.is_land
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "graveyard_land":
            return
        p = state.player_by_id(controller_id)
        card = next((c for c in p.graveyard if c.id == target_id), None)
        if card is None:
            return
        from mtg_bot.game import reveal_to_all
        p.graveyard.remove(card)
        card.zone = Zone.HAND
        p.hand.append(card)
        reveal_to_all(state, card)
        state.log.append(
            f"{card.printing.name} returns from P{controller_id}'s "
            f"graveyard to hand (Petrified Field)"
        )

    def describe(self) -> str:
        return (
            "return target land card from your graveyard to your hand"
        )



@dataclass(frozen=True)
class DealDamageToControllerAndAddMana(Effect):
    """Ancient Tomb's mana ability: "{T}: Add {C}{C}. This land deals
    2 damage to you." Two-step: self-damage + pool fill. Paired with
    is_mana_ability=True so it bypasses the stack.
    """

    damage: int = 2
    produces: tuple[tuple[str, int], ...] = (("C", 2),)

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import ManaCoin

        p = state.player_by_id(controller_id)
        p.life -= self.damage
        state.log.append(
            f"{state.resolving_source.printing.name if state.resolving_source else 'land'} "
            f"deals {self.damage} to P{controller_id} (life: {p.life})"
        )
        for color, count in self.produces:
            for _ in range(count):
                p.mana_pool.append(ManaCoin(color=color))

    def describe(self) -> str:
        return (
            f"add "
            + "".join("{" + c + "}" for c, n in self.produces for _ in range(n))
            + f"; this land deals {self.damage} damage to you"
        )



@dataclass(frozen=True)
class GrantLifeToEachOpponent(Effect):
    """Grove of the Burnwillows' "Each opponent gains 1 life" clause
    (runs alongside the mana add via Sequence).
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import gain_life

        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            gain_life(state, p.id, self.amount)
            state.log.append(
                f"P{p.id} gains {self.amount} life (Grove) — life: {p.life}"
            )

    def describe(self) -> str:
        return f"each opponent gains {self.amount} life"



@dataclass(frozen=True)
class DestroyTargetNonlandPermanent(Destroy):
    """``Destroy`` constructor sugar for "destroy target nonland
    permanent".

    Filter is ``~IsLand() & ~IsPlayer()`` so the target pool covers
    every battlefield permanent that isn't a land while excluding
    players (which the unified target pipeline walks alongside the
    battlefield). Indestructible permanents survive at resolve time
    via the ``destroy`` helper's indestructible check.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsLand, IsPlayer
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(~IsLand() & ~IsPlayer()),),
        )

    def describe(self) -> str:
        return "destroy target nonland permanent"



@dataclass(frozen=True)
class MoveSelfCountersToTargetCreatureYouControl(Effect):
    """Move all +1/+1 counters from the source to target creature you
    control.

    Matches Dockworker Drone's dies trigger ("put its counters on target
    creature you control"). Reads state.resolving_source for the count;
    no-op if the source has zero counters or wasn't set.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature and o.controller_id == controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        count = source.plus_one_counters
        if count <= 0:
            return
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        from mtg_bot.game import add_plus_one_counters
        add_plus_one_counters(state, obj, count)
        # Clear source counters — they've been transferred.
        source.plus_one_counters = 0
        state.log.append(
            f"{count} +1/+1 counter(s) moved from {source.printing.name} "
            f"to {obj.printing.name}"
        )

    def describe(self) -> str:
        return (
            "move this creature's +1/+1 counters to target creature you "
            "control"
        )



@dataclass(frozen=True)
class RegenerateTargetCreature(Effect):
    """"Regenerate target creature" — sets ``regeneration_shield``
    on the target. The next time this turn state-based actions would
    destroy the creature from lethal damage (or a destroy effect;
    NOT from toughness <= 0), the destruction is replaced by:
    remove all damage, tap the creature, remove it from combat.
    The shield is consumed on use and cleared at cleanup if unused.

    ``allowed_subtypes``: when non-empty, restricts legal targets to
    creatures whose printed subtypes overlap with the set. Matches
    Swarmyard's "target Rat, Spider, Squirrel, Insect, or Snake."
    Empty tuple = any creature (not used by current cards).
    """

    allowed_subtypes: frozenset[str] = frozenset()

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and (
                not self.allowed_subtypes
                or any(
                    s in self.allowed_subtypes
                    for s in _eff_subtypes(state, o)
                )
            )
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        _, tid = targets[0]
        obj = next(
            (o for o in state.battlefield if o.id == tid), None
        )
        if obj is None:
            return
        obj.regeneration_shield = True
        state.log.append(
            f"{obj.printing.name} gains a regeneration shield"
        )

    def describe(self) -> str:
        if self.allowed_subtypes:
            subs = ", ".join(sorted(self.allowed_subtypes))
            return f"regenerate target {subs}"
        return "regenerate target creature"



@dataclass(frozen=True)
class MovePlusOneCounterBetweenPermanentsYouControl(Effect):
    """Nesting Grounds: "Move a counter from target permanent you
    control onto a second target permanent."

    Dual-target effect: targets[0] is the donor, targets[1] is the
    recipient. Both must be permanents the controller controls.

    Simplification: oracle says "a counter" (any type — +1/+1,
    charge, loyalty, stun, etc.). This implementation moves exactly
    one +1/+1 counter. If the donor has none, the ability fizzles.
    Charge/loyalty/stun counter movement isn't modeled — those
    counter types are tied to specific permanent types (Spacecraft,
    planeswalker, tapped creatures) where the recipient might not
    be a valid holder anyway.
    """

    def requires_target(self) -> bool:
        return True

    def target_count_range(self) -> tuple[int, int]:
        return (2, 2)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.controller_id == controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if len(targets) < 2:
            return
        (_, donor_id), (_, recipient_id) = targets[0], targets[1]
        if donor_id == recipient_id:
            return
        donor = next(
            (o for o in state.battlefield if o.id == donor_id), None
        )
        recipient = next(
            (o for o in state.battlefield if o.id == recipient_id), None
        )
        if donor is None or recipient is None:
            return
        if donor.plus_one_counters <= 0:
            state.log.append(
                f"Nesting Grounds fizzles — "
                f"{donor.printing.name} has no +1/+1 counters"
            )
            return
        from mtg_bot.game import add_plus_one_counters

        donor.plus_one_counters -= 1
        add_plus_one_counters(state, recipient, 1)
        state.log.append(
            f"+1/+1 counter moved from {donor.printing.name} "
            f"(now {donor.plus_one_counters}) to {recipient.printing.name}"
        )

    def describe(self) -> str:
        return (
            "move a +1/+1 counter from target permanent you control "
            "onto a second target permanent"
        )



@dataclass(frozen=True)
class PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact(Effect):
    """Put a +1/+1 counter on this creature for each OTHER creature and/or
    artifact you control.

    Matches Luxknight Breacher's "enters with" clause (modeled as an ETB
    trigger since we don't implement replacement effects). Counts are
    sampled at resolution; the source itself is excluded (it's on the
    battlefield by the time the ETB trigger resolves).
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        count = 0
        for o in state.battlefield:
            if o.id == source.id:
                continue
            if o.controller_id != controller_id:
                continue
            if o.printing.is_creature or o.printing.is_artifact:
                count += 1
        if count <= 0:
            return
        from mtg_bot.game import add_plus_one_counters
        add_plus_one_counters(state, source, count)
        state.log.append(
            f"{source.printing.name} enters with {count} +1/+1 counters"
        )

    def describe(self) -> str:
        return (
            "put a +1/+1 counter on this creature for each other creature "
            "and/or artifact you control"
        )



@dataclass(frozen=True)
class PutChargeCountersOnTargetSpacecraftYouControl(Effect):
    """Put ``amount`` charge counters on a Spacecraft you control.

    Spacecraft-only variant. Drill Too Deep uses the broader
    ``PutChargeCountersOnTargetSpacecraftOrPlanetYouControl`` (Planet is
    a Land subtype that IS modeled — see ``make_planet_land``).
    """

    amount: int = 5

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.printing.is_spacecraft
            and o.controller_id == controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None or not obj.printing.is_spacecraft:
            return
        from mtg_bot.game import add_charge_counters
        added = add_charge_counters(state, obj, self.amount)
        state.log.append(
            f"{obj.printing.name} gets {added} charge counters "
            f"(now {obj.charge_counters})"
        )

    def describe(self) -> str:
        return (
            f"put {self.amount} charge counters on target Spacecraft you "
            f"control"
        )



@dataclass(frozen=True)
class PutChargeCountersOnTargetSpacecraftOrPlanetYouControl(Effect):
    """Put ``amount`` charge counters on target Spacecraft or Planet you
    control.

    Matches Drill Too Deep's first mode in its full printed form. Planet
    is a Land subtype in this codebase (see ``make_planet_land``); a
    permanent qualifies if it's a Spacecraft (artifact subtype) OR has
    the ``Planet`` subtype, AND the controller matches.
    """

    amount: int = 5

    @staticmethod
    def _qualifies(printing) -> bool:
        return printing.is_spacecraft or "Planet" in printing.subtypes

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if self._qualifies(o.printing)
            and o.controller_id == controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None or not self._qualifies(obj.printing):
            return
        if obj.controller_id != controller_id:
            return
        obj.charge_counters += self.amount
        state.log.append(
            f"{obj.printing.name} gets {self.amount} charge counters "
            f"(now {obj.charge_counters})"
        )

    def describe(self) -> str:
        return (
            f"put {self.amount} charge counters on target Spacecraft or "
            f"Planet you control"
        )



@dataclass(frozen=True)
class PutChargeCountersOnSelf(Effect):
    """Put N charge counters on the resolving source.

    ``amount`` is the fixed count when ``amount_from_x`` is False.
    When ``amount_from_x`` is True, reads ``state.resolving_x_value``
    instead — used by Blast Zone's "{X}{X}, {T}: Put X charge counters
    on this land".

    Untargeted; the source is whatever's currently resolving
    (``state.resolving_source``). Used both as an ETB effect (entering
    with N charge counters) and as an activated-ability payload.
    """

    amount: int = 1
    amount_from_x: bool = False

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None:
            return
        n = state.resolving_x_value if self.amount_from_x else self.amount
        if n <= 0:
            return
        from mtg_bot.game import add_charge_counters
        added = add_charge_counters(state, source, n)
        state.log.append(
            f"{source.printing.name} gets {added} charge counter"
            f"{'s' if added != 1 else ''} (now {source.charge_counters})"
        )

    def describe(self) -> str:
        if self.amount_from_x:
            return "put X charge counters on this permanent"
        return (
            f"put {self.amount} charge counter"
            f"{'s' if self.amount != 1 else ''} on this permanent"
        )



@dataclass(frozen=True)
class DestroyEachNonlandPermanentWithMVEqualToSourceChargeCounters(Effect):
    """Blast Zone payoff: "Destroy each nonland permanent with mana
    value equal to the number of charge counters on this land."

    Reads ``state.resolving_source.charge_counters`` (snapshotted at
    resolution time). Tokens count too — they have a printed mana
    value of 0 and are destroyed when the source has 0 charge
    counters (rare edge case but oracle-compliant).
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import move_off_battlefield

        source = state.resolving_source
        if source is None:
            return
        target_mv = source.charge_counters
        # Snapshot the list — destroying mutates state.battlefield.
        for obj in list(state.battlefield):
            if obj.printing.is_land:
                continue
            if obj.printing.mana_cost.total != target_mv:
                continue
            if "indestructible" in obj.printing.keywords:
                continue
            state.log.append(
                f"{obj.printing.name} is destroyed by Blast Zone "
                f"(MV={target_mv})"
            )
            move_off_battlefield(state, obj, destination="graveyard")

    def describe(self) -> str:
        return (
            "destroy each nonland permanent with mana value equal to "
            "the number of charge counters on this land"
        )



@dataclass(frozen=True)
class DestroyTargetArtifactOrTappedCreature(Destroy):
    """``Destroy`` constructor sugar for "destroy target artifact, or
    tapped creature".

    Filter is ``IsArtifact() | (IsCreature() & IsTapped())``. The
    unified ``Destroy.resolve`` re-verifies the slot's filter at
    resolution per CR 608.2b — a creature that untapped between cast
    and resolve fails the recheck (an artifact target is unaffected
    by the untap and stays legal). Matches Radiant Strike's destroy
    clause.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            IsArtifact, IsCreature, IsTapped,
        )
        from mtg_bot.effects.targets import TargetSpec
        f = IsArtifact() | (IsCreature() & IsTapped())
        object.__setattr__(self, "targets", (TargetSpec(f),))

    def describe(self) -> str:
        return "destroy target artifact or tapped creature"



@dataclass(frozen=True)
class TapTargetOpponentPermanentAndStunIt(Effect):
    """Tap target artifact or creature an opponent controls, then put
    a stun counter on it. Matches Mechanozoa's ETB. Both effects apply
    to the same chosen target.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if (o.printing.is_artifact or o.printing.is_creature)
            and o.controller_id != controller_id
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import tap_permanent

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        if not obj.tapped:
            tap_permanent(state, obj)
        from mtg_bot.game import add_stun_counters
        add_stun_counters(state, obj, 1)
        state.log.append(
            f"{obj.printing.name} tapped and gains a stun counter "
            f"(now {obj.stun_counters})"
        )

    def describe(self) -> str:
        return (
            "tap target artifact or creature an opponent controls and "
            "put a stun counter on it"
        )



@dataclass(frozen=True)
class PutStunCounterOnTappedCreature(Effect):
    """Put a stun counter on up to one target tapped creature. Matches
    Cryogen Relic's activated ability. "Up to one" → required target
    in our simplified pipeline; silently fizzles if none legal."""

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature and o.tapped
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None or not obj.tapped:
            return
        from mtg_bot.game import add_stun_counters
        add_stun_counters(state, obj, 1)
        state.log.append(
            f"{obj.printing.name} gets a stun counter "
            f"(now {obj.stun_counters})"
        )

    def describe(self) -> str:
        return "put a stun counter on up to one target tapped creature"



@dataclass(frozen=True)
class TapAndStunUpToOneTargetCreature(Effect):
    """Tap up to one target creature and put ``stun_amount`` stun
    counters on it. Both effects apply to the same chosen target.

    Matches Meat Locker's "tap up to one target creature and put two
    stun counters on it." (DSK Room — door trigger). The "up to one"
    qualifier makes the target slot optional: ``target_count_range``
    flips to ``(0, 1)`` and ``requires_target`` flips to ``False`` so
    the cast-site target-selection pipeline allows zero-target casts.

    No controller restriction (unlike ``TapTargetOpponentPermanentAndStunIt``,
    which is opponent-only) and no taps-state restriction (unlike
    ``PutStunCounterOnTappedCreature``, which targets only already-tapped
    creatures): Meat Locker can tap any creature, friend or foe, tapped
    or untapped.
    """

    stun_amount: int = 2

    def requires_target(self) -> bool:
        return False

    def target_count_range(self) -> tuple[int, int]:
        return (0, 1)

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import tap_permanent

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        if not obj.tapped:
            tap_permanent(state, obj)
        from mtg_bot.game import add_stun_counters
        add_stun_counters(state, obj, self.stun_amount)
        state.log.append(
            f"{obj.printing.name} tapped and gains "
            f"{self.stun_amount} stun counters "
            f"(now {obj.stun_counters})"
        )

    def describe(self) -> str:
        return (
            f"tap up to one target creature and put {self.stun_amount} "
            f"stun counters on it"
        )



@dataclass(frozen=True)
class TapAnotherTargetCreature(Tap):
    """``Tap`` constructor sugar for "tap another target creature"
    (Starport Security's activated ability).

    Filter is ``IsCreature() & ~IsSource()`` — the ``~IsSource()``
    arm excludes the resolving source from its own legal-target pool
    so "another" reads CR-correct.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsCreature, IsSource
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsCreature() & ~IsSource()),),
        )

    def describe(self) -> str:
        return "tap another target creature"



@dataclass(frozen=True)
class TapUpToOneTargetCreature(Tap):
    """``Tap`` constructor sugar for "tap up to one target creature"
    (Baseball Bat's equipped-attack rider).

    Filter is :class:`IsCreature` with ``count_min=0`` so the
    targeting pipeline treats the slot as optional — a 0-target
    resolution is a silent no-op (the controller chose not to
    target).
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsCreature
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsCreature(), count_min=0, count_max=1),),
        )

    def describe(self) -> str:
        return "tap up to one target creature"



@dataclass(frozen=True)
class TapTargetArtifactOrCreature(Tap):
    """``Tap`` constructor sugar for "tap target artifact or creature"
    (Mental Modulation's middle clause).

    Filter is ``IsArtifact() | IsCreature()``. Already-tapped targets
    are no-ops via the underlying ``turn.tap_permanent``.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsArtifact, IsCreature
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsArtifact() | IsCreature()),),
        )

    def describe(self) -> str:
        return "tap target artifact or creature"



@dataclass(frozen=True)
class UntapTargetOtherPermanent(Effect):
    """Untap target permanent other than the source.

    Matches Nanoform Sentinel's "untap another target permanent" trigger.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        source = state.resolving_source
        source_id = source.id if source is not None else -1
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.id != source_id and o.tapped
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        if not obj.tapped:
            return
        obj.tapped = False
        state.log.append(f"{obj.printing.name} is untapped")

    def describe(self) -> str:
        return "untap another target permanent"



@dataclass(frozen=True)
class DestroyTargetArtifactEnchantmentOrFlyingCreature(Destroy):
    """``Destroy`` constructor sugar for "destroy target artifact,
    enchantment, or creature with flying".

    Filter is ``IsArtifact() | IsEnchantment() | (IsCreature() &
    HasKeyword("flying"))``. ``HasKeyword`` reads
    ``effective_keywords`` so anthem-granted / temp-granted flying
    counts toward legality. Matches Shattered Wings' destroy clause.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            HasKeyword, IsArtifact, IsCreature, IsEnchantment,
        )
        from mtg_bot.effects.targets import TargetSpec
        f = (
            IsArtifact()
            | IsEnchantment()
            | (IsCreature() & HasKeyword("flying"))
        )
        object.__setattr__(self, "targets", (TargetSpec(f),))

    def describe(self) -> str:
        return (
            "destroy target artifact, enchantment, or creature with flying"
        )



@dataclass(frozen=True)
class CounterTargetSpell(Effect):
    """Counter target spell on the stack.

    ``filter_types`` optionally restricts what card types can be
    countered (e.g., frozenset({"Artifact", "Enchantment"}) for Annul).
    Empty set means "any spell".

    Targets a stack item by its index via the new ``"stack"`` target kind.
    Triggered and activated abilities on the stack can't be countered by
    this effect (only spells — items where ``ability`` is None).
    """

    filter_types: frozenset[str] = frozenset()

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        out: list[Target] = []
        for i, item in enumerate(state.stack):
            if item.ability is not None:
                continue  # not a spell
            if item.countered:
                continue
            src = item.source.printing
            if self.filter_types:
                if not any(t in src.card_types for t in self.filter_types):
                    continue
            out.append(("stack", i))
        return out

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, idx = targets[0]
        if kind != "stack":
            return
        if not (0 <= idx < len(state.stack)):
            return
        item = state.stack[idx]
        if item.ability is not None:
            return
        item.countered = True
        state.log.append(
            f"counterspell: {item.source.printing.name} is countered"
        )

    def describe(self) -> str:
        if not self.filter_types:
            return "counter target spell"
        types = ", ".join(sorted(t.lower() for t in self.filter_types))
        return f"counter target {types} spell"



@dataclass(frozen=True)
class DestroyAllCreatures(Effect):
    """Destroy every creature on the battlefield. Wrath-style effect.

    Indestructible creatures survive (routes each kill through
    turn.destroy()). Tokens cease to exist normally.

    ``tapped_only`` / ``untapped_only`` narrow the wrath to only
    creatures matching that tap-state (Split Up's modal "destroy all
    tapped creatures" / "destroy all untapped creatures"). Mutually
    exclusive — at most one may be True.

    ``min_power`` (default 0 = no filter) narrows the wrath to only
    creatures whose ``effective_power`` is at least that threshold —
    Restricted Office's "destroy all creatures with power 3 or
    greater". The threshold is read against ``effective_power`` so
    any layer-7 pump or counters apply at resolve time. The default
    (all filters off) is the unrestricted Wrath shape.
    """

    tapped_only: bool = False
    untapped_only: bool = False
    min_power: int = 0

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import destroy
        from mtg_bot.game import effective_power

        victims = [o for o in state.battlefield if o.printing.is_creature]
        if self.tapped_only:
            victims = [o for o in victims if o.tapped]
        elif self.untapped_only:
            victims = [o for o in victims if not o.tapped]
        if self.min_power > 0:
            victims = [
                o for o in victims
                if effective_power(state, o) >= self.min_power
            ]
        for obj in victims:
            state.log.append(f"{obj.printing.name} is destroyed (wrath)")
            destroy(state, obj)

    def describe(self) -> str:
        if self.tapped_only:
            return "destroy all tapped creatures"
        if self.untapped_only:
            return "destroy all untapped creatures"
        if self.min_power > 0:
            return (
                f"destroy all creatures with power {self.min_power} or "
                f"greater"
            )
        return "destroy all creatures"



@dataclass(frozen=True)
class CreateXTokensOfTargetPermanentWithHasteEndStepSac(Effect):
    """Devastating Onslaught `{X}{X}{R}`:
    "Create X tokens that are copies of target artifact or creature you
    control. Those tokens gain haste until end of turn. Sacrifice them
    at the beginning of the next end step."

    Reads X from ``state.resolving_x_value``. Each token copies the
    target's printing directly (shared CardPrinting) and enters with a
    ``haste`` temp keyword. A delayed "sacrifice at end of turn"
    trigger is scheduled per token so they're cleaned up automatically.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.controller_id == controller_id
            and (o.printing.is_artifact or o.printing.is_creature)
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, tid = targets[0]
        target = next((o for o in state.battlefield if o.id == tid), None)
        if target is None:
            return
        x = state.resolving_x_value
        if x <= 0:
            return
        for _ in range(x):
            token = state.new_object(
                target.printing,
                owner_id=controller_id,
                zone=Zone.BATTLEFIELD,
            )
            # Temporarily suppress the "enters with X counters" replacement
            # when putting token copies into play — tokens copy printed
            # characteristics but X is 0 for them (they weren't cast).
            saved_x = state.resolving_x_value
            state.resolving_x_value = 0
            try:
                put_into_battlefield(state, token, controller_id)
            finally:
                state.resolving_x_value = saved_x
            token.temp_keywords = token.temp_keywords | frozenset({"haste"})
            state.delayed_triggers.append({
                "fire_at_turn": state.turn_number,
                "controller_id": controller_id,
                "source_id": token.id,
                "effect": SacrificeSourceDelayed(),
            })
        state.log.append(
            f"Devastating Onslaught creates {x} copies of "
            f"{target.printing.name} with haste (sac at end step)"
        )

    def describe(self) -> str:
        return (
            "create X tokens that are copies of target artifact or "
            "creature you control; they gain haste until end of turn; "
            "sacrifice them at the beginning of the next end step"
        )



@dataclass(frozen=True)
class ExileAllCreaturesAndSpacecraft(Effect):
    """Exile every creature on the battlefield, as well as Spacecraft (whether
    or not the Spacecraft has been stationed into creature mode).

    Matches Beyond the Quiet. Mass wrath-style effect: no targets, acts on
    the full battlefield snapshot at resolution. Tokens cease to exist per
    rule 704.5d.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import move_off_battlefield

        victims = [
            o for o in state.battlefield
            if o.printing.is_creature or o.printing.is_spacecraft
        ]
        for obj in victims:
            state.log.append(f"{obj.printing.name} is exiled (mass)")
            move_off_battlefield(state, obj, destination="exile")

    def describe(self) -> str:
        return "exile all creatures and Spacecraft"



@dataclass(frozen=True)
class ExileTargetCreature(Exile):
    """``Exile`` constructor sugar for "exile target creature".

    Filter is :class:`IsCreature` (which reads ``is_creature_now`` so
    Impending-suppressed Overlords aren't legal targets). Tokens
    exiled cease to exist per CR 704.5d.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsCreature
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(self, "targets", (TargetSpec(IsCreature()),))



@dataclass(frozen=True)
class PhaseOutEquippedCreature(Effect):
    """"Equipped creature phases out." — Robe of Stars' Astral Projection.

    Reads the Equipment source from ``state.resolving_source`` and
    phases out whatever creature it's attached to. Treat the creature
    (and anything attached to it) as though they don't exist until its
    controller's next turn — simplified here to filtering out of
    combat damage only; triggers/anthems/targeting still see phased
    objects in this engine's current scope.

    If the Equipment isn't attached to anything, the ability silently
    does nothing.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        source = state.resolving_source
        if source is None or source.attached_to is None:
            return
        creature = next(
            (o for o in state.battlefield if o.id == source.attached_to),
            None,
        )
        if creature is None:
            return
        creature.phased_out = True
        # Remove it from ongoing combat so the filter in
        # _resolve_damage_step is redundant-but-safe.
        creature.attacking = False
        creature.blocking = None
        creature.blocked_by = []
        state.log.append(
            f"{creature.printing.name} phases out (Astral Projection)"
        )

    def describe(self) -> str:
        return "equipped creature phases out"



@dataclass(frozen=True)
class TakeExtraTurnAfterThisOne(Effect):
    """"Take an extra turn after this one."

    Appends the controller to ``state.extra_turn_queue``; the sim loop
    in simulate.simulate_game consumes the queue between turns. Matches
    Nexus of Fate's extra-turn clause.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        state.extra_turn_queue.append(controller_id)
        state.log.append(
            f"P{controller_id} is scheduled to take an extra turn"
        )

    def describe(self) -> str:
        return "take an extra turn after this one"



@dataclass(frozen=True)
class SearchLibraryForGreenCreatureWithMVAtMostX(Effect):
    """Matches Green Sun's Zenith: "Search your library for a green
    creature card with mana value X or less, put it onto the battlefield,
    then shuffle." X is read from ``state.resolving_x_value``.

    Green creature: printing has "G" in ``mana_cost.colors``. Mana value
    uses ``mana_cost.total`` (includes the green pip). Declining to
    fetch is legal (matches the "search ... may fail" convention).
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.game import shuffle_library

        x = state.resolving_x_value
        p = state.player_by_id(controller_id)
        options = [
            c for c in p.library
            if c.printing.is_creature
            and "G" in c.printing.mana_cost.colors
            and c.printing.mana_cost.total <= x
        ]
        if not options:
            state.log.append(
                f"P{controller_id} searches for a green creature with "
                f"MV ≤ {x} — none found"
            )
            shuffle_library(state, controller_id)
            return
        chosen_id = None
        if p.agent is not None:
            chosen_id = p.agent.choose_card_from_library(
                state, controller_id, [c.id for c in options]
            )
        if chosen_id is None:
            state.log.append(
                f"P{controller_id} declines to fetch from Green Sun's Zenith"
            )
            shuffle_library(state, controller_id)
            return
        card = next((c for c in options if c.id == chosen_id), options[0])
        p.library.remove(card)
        state.log.append(
            f"P{controller_id} fetches {card.printing.name} onto the battlefield"
        )
        put_into_battlefield(state, card, controller_id)
        shuffle_library(state, controller_id)

    def describe(self) -> str:
        return (
            "search your library for a green creature card with mana "
            "value X or less, put it onto the battlefield, then shuffle"
        )



@dataclass(frozen=True)
class GainControlOfTargetSliverPermanently(Effect):
    """"Gain control of target Sliver. (This effect lasts indefinitely.)"

    Matches Sliver Overlord's second activated ability. Unlike temporary
    steal effects (Mutinous Massacre, Control-target-opponent), this
    one reassigns controller_id without scheduling a revert — it is
    not reset at cleanup. The Sliver becomes permanently the
    activator's.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and "Sliver" in _eff_subtypes(state, o)
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        if "Sliver" not in _eff_subtypes(state, obj):
            return
        if obj.controller_id == controller_id:
            # Already yours — no change, but log for visibility.
            state.log.append(
                f"P{controller_id} targets own Sliver "
                f"{obj.printing.name} — no change"
            )
            return
        obj.controller_id = controller_id
        # Permanent gain of control: do NOT set temp_original_controller_id.
        state.log.append(
            f"P{controller_id} permanently gains control of "
            f"{obj.printing.name}"
        )

    def describe(self) -> str:
        return "gain control of target Sliver (indefinitely)"



@dataclass(frozen=True)
class PreventAllCombatDamageThisTurn(Effect):
    """"Prevent all combat damage that would be dealt this turn."

    Matches Darkness ({B} instant). Sets state-level flag consumed by
    ``combat._resolve_damage_step``; the flag is cleared at end-of-turn
    cleanup. Non-combat damage (direct damage, pings) is unaffected.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        state.prevent_combat_damage_this_turn = True
        state.log.append("all combat damage prevented this turn")

    def describe(self) -> str:
        return "prevent all combat damage that would be dealt this turn"



@dataclass(frozen=True)
class ExileTargetCreatureWithPowerOrToughnessAtMost(Exile):
    """``Exile`` constructor sugar for "exile target creature with power
    or toughness N or less" (Warping Wail mode 1, Scryfall-updated
    oracle).

    Filter is ``IsCreature() & (HasPowerAtMost(n) | HasToughnessAtMost(n))``.
    Both predicates read ``effective_power`` / ``effective_toughness``
    so temp buffs and static anthems are respected at both targeting
    and the at-resolve recheck (a Giant-Growth-style pump between cast
    and resolve will fail the recheck and fizzle the exile).
    """

    max_value: int = 1

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            HasPowerAtMost, HasToughnessAtMost, IsCreature,
        )
        from mtg_bot.effects.targets import TargetSpec
        f = IsCreature() & (
            HasPowerAtMost(self.max_value)
            | HasToughnessAtMost(self.max_value)
        )
        object.__setattr__(self, "targets", (TargetSpec(f),))

    def describe(self) -> str:
        return (
            f"exile target creature with power or toughness "
            f"{self.max_value} or less"
        )



@dataclass(frozen=True)
class DestroyTargetNonlandPermanentWithLanderForController(Effect):
    """Destroy target nonland permanent. Its (former) controller creates a
    Lander token.

    Matches Emergency Eject's oracle text. Uses a new "permanent" target kind
    (any non-land battlefield object) so this can hit creatures, artifacts,
    and enchantments uniformly.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if not o.printing.is_land
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.cards import TOKEN_PRINTINGS
        from mtg_bot.game import Zone
        from mtg_bot.turn import destroy

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        former_controller = obj.controller_id
        state.log.append(f"{obj.printing.name} is destroyed")
        if not destroy(state, obj):
            return  # indestructible: no Lander consolation
        lander = state.new_object(
            TOKEN_PRINTINGS["Lander"],
            owner_id=former_controller,
            zone=Zone.BATTLEFIELD,
        )
        state.log.append(f"P{former_controller} creates a Lander token")
        put_into_battlefield(state, lander, former_controller)

    def describe(self) -> str:
        return (
            "destroy target nonland permanent; its controller creates a "
            "Lander token"
        )



@dataclass(frozen=True)
class PutPlusOneCounterOnTargetCreature(PutPlusOneCounters):
    """``PutPlusOneCounters`` constructor sugar for "put N +1/+1
    counter(s) on target creature" (Honor's bare form).

    Filter is :class:`IsCreature` (& :class:`IsAttacking` when
    ``attacking_only`` is True — Hospital Room half).
    """

    attacking_only: bool = False

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsAttacking, IsCreature
        from mtg_bot.effects.targets import TargetSpec
        f = IsCreature() & IsAttacking() if self.attacking_only else IsCreature()
        object.__setattr__(self, "targets", (TargetSpec(f),))

    def describe(self) -> str:
        scope = "target attacking creature" if self.attacking_only else "target creature"
        if self.amount == 1:
            return f"put a +1/+1 counter on {scope}"
        return f"put {self.amount} +1/+1 counters on {scope}"



@dataclass(frozen=True)
class PutPlusOneCountersAndGrantKeywordsUEOTOnTargetCreatureYouControl(Effect):
    """Put N +1/+1 counters on target creature you control AND grant it
    keywords until end of turn (a single chosen target).

    Models the third mode of Archenemy's Charm ("Put two +1/+1 counters
    on target creature you control. It gains lifelink until end of
    turn."). Done as a single combined effect rather than a Sequence
    because Sequence currently allows at most one targeting sub-effect.
    """

    amount: int = 2
    granted_keywords: frozenset[str] = frozenset()
    you_control_only: bool = True

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
            and (not self.you_control_only or o.controller_id == controller_id)
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "creature":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None:
            return
        from mtg_bot.game import add_plus_one_counters
        add_plus_one_counters(state, obj, self.amount)
        if self.granted_keywords:
            obj.temp_keywords = obj.temp_keywords | self.granted_keywords
        state.log.append(
            f"{obj.printing.name} gets {self.amount} +1/+1 counter(s) "
            f"(now {obj.plus_one_counters})"
            + (
                f" and gains {', '.join(sorted(self.granted_keywords))} "
                f"until end of turn"
                if self.granted_keywords else ""
            )
        )

    def describe(self) -> str:
        kw = (
            f". It gains {', '.join(sorted(self.granted_keywords))} "
            f"until end of turn"
            if self.granted_keywords else ""
        )
        return (
            f"put {self.amount} +1/+1 counters on target creature you "
            f"control{kw}"
        )



@dataclass(frozen=True)
class PutPlusOneCounterOnEachCreatureYouControl(Effect):
    """Put ``amount`` +1/+1 counter(s) on each creature the controller
    controls at resolution. Matches Atmospheric Greenhouse's ETB.

    ``exclude_source=True`` models "each other creature you control" — reads
    ``state.resolving_source`` to skip the triggering permanent (e.g.
    Dawnstrike Vanguard's end-step trigger).
    """

    amount: int = 1
    exclude_source: bool = False

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import add_plus_one_counters
        source = state.resolving_source if self.exclude_source else None
        count = 0
        for obj in state.battlefield:
            if obj.controller_id != controller_id:
                continue
            if not obj.printing.is_creature:
                continue
            if source is not None and obj is source:
                continue
            add_plus_one_counters(state, obj, self.amount)
            count += 1
        scope = "other creature(s)" if self.exclude_source else "creature(s)"
        state.log.append(
            f"P{controller_id}'s {count} {scope} each get "
            f"a +1/+1 counter"
        )

    def describe(self) -> str:
        scope = (
            "each other creature you control"
            if self.exclude_source
            else "each creature you control"
        )
        if self.amount == 1:
            return f"put a +1/+1 counter on {scope}"
        return f"put {self.amount} +1/+1 counters on {scope}"



@dataclass(frozen=True)
class PutPlusOneCounterOnSelf(Effect):
    """Put ``amount`` +1/+1 counter(s) on the ability's source ("this creature").

    Reads ``state.resolving_source`` to find the source — it's set by
    ``resolve_top_of_stack`` during ability resolution.
    """

    amount: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import add_plus_one_counters
        source = state.resolving_source
        if source is None:
            return
        add_plus_one_counters(state, source, self.amount)
        state.log.append(
            f"{source.printing.name} gets a +1/+1 counter "
            f"(now {source.plus_one_counters})"
        )

    def describe(self) -> str:
        if self.amount == 1:
            return "put a +1/+1 counter on this creature"
        return f"put {self.amount} +1/+1 counters on this creature"



@dataclass(frozen=True)
class PutPlusOneCounterOnOpponentsCreature(PutPlusOneCounters):
    """``PutPlusOneCounters`` constructor sugar for "put a +1/+1 counter
    on target creature an opponent controls" (Sunset Saboteur's
    attacks trigger).

    Filter is ``IsCreature() & ControlledByOpponent()``. The counter-
    doubling replacement (Loading Zone-style) is gated on the
    recipient's controller per the ``add_plus_one_counters`` helper —
    a doubler controlled by the spell's caster (the opponent of the
    target's controller) doesn't fire because doublers check the
    receiving creature's controller.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            ControlledByOpponent, IsCreature,
        )
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsCreature() & ControlledByOpponent()),),
        )

    def describe(self) -> str:
        return "put a +1/+1 counter on target creature an opponent controls"



@dataclass(frozen=True)
class PutPlusOneCounterOnCreatureYouControl(PutPlusOneCounters):
    """``PutPlusOneCounters`` constructor sugar for "put N +1/+1
    counter(s) on target creature you control" (Rayblade Trooper).

    Filter is ``IsCreature() & ControlledByYou()``.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            ControlledByYou, IsCreature,
        )
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsCreature() & ControlledByYou()),),
        )

    def describe(self) -> str:
        if self.amount == 1:
            return "put a +1/+1 counter on target creature you control"
        return (
            f"put {self.amount} +1/+1 counters on target creature you control"
        )



@dataclass(frozen=True)
class DealDamageEqualToSourcePowerAnyTarget(Effect):
    """Source deals damage equal to its (effective) power to any target
    (creature or player).

    Matches "this creature deals damage equal to its power to any
    target" — Infernal Phantom's dies trigger in DSK. The amount is
    snapshot from ``state.resolving_source.effective_power`` at resolve
    time. For a dies-trigger source the GameObject is already in the
    graveyard but ``state.resolving_source`` still references the
    pre-death object so its layer-7 power can be read; if the power is
    nonpositive the damage is a no-op (CR 119.5: "if the amount of
    damage that would be dealt is 0 or less, no damage is dealt").

    Bespoke amount (driven by source power) so the class survives the
    PR #3 collapse, but the legal-target pool now consults
    ``self.targets[0].filter`` via the structural ``TargetFilter``
    grammar — same any-target expansion ``DealDamage`` uses.
    """

    targets: tuple = ()

    def __post_init__(self) -> None:
        if not self.targets:
            from mtg_bot.effects.targets import TargetSpec, any_target_filter
            object.__setattr__(
                self, "targets",
                (TargetSpec(any_target_filter()),),
            )

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import effective_power
        if not targets:
            return
        source = state.resolving_source
        if source is None:
            return
        amount = effective_power(state, source)
        if amount <= 0:
            return
        if not _slot_target_still_legal(state, self.targets[0], targets[0]):
            return
        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            _apply_damage(state, targets[0], amount)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return (
            "deals damage equal to this creature's power to any target"
        )



# DealDamage and its helpers moved earlier in the file (above
# DealDamageToTargetOpponent at line ~1620 in the pre-collapse
# layout) so the sugar subclasses can reference DealDamage at class-
# definition time. See the canonical DealDamage definition near the
# top of the module.



@dataclass(frozen=True)
class DealDamageAnyTarget(DealDamage):
    """``DealDamage`` constructor sugar for "deals N damage to any
    target".

    Identical at the engine level to
    ``DealDamage(amount=N, targets=(TargetSpec(any_target_filter()),))``;
    the subclass is sugar so the parser can emit a self-describing
    name in the AST and existing parser tests' equality assertions
    keep their per-shape granularity. No extra fields, no behavioural
    overrides — every code path is the unified ``DealDamage`` resolve.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.targets import TargetSpec, any_target_filter
        # Frozen dataclass: bypass __setattr__ via __dict__ write.
        object.__setattr__(
            self, "targets",
            (TargetSpec(any_target_filter()),),
        )


@dataclass(frozen=True)
class DealDamageToTargetCreature(DealDamage):
    """``DealDamage`` constructor sugar for "deals N damage to (up to
    one) target creature".

    ``optional=True`` flips the slot's ``count_min`` to 0 so the
    targeting pipeline treats the slot as ``0..1`` ("up to one
    target creature"); default ``False`` is the bare 1..1 form.
    """

    optional: bool = False

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsCreature
        from mtg_bot.effects.targets import TargetSpec
        spec = TargetSpec(
            IsCreature(),
            count_min=0 if self.optional else 1,
            count_max=1,
        )
        object.__setattr__(self, "targets", (spec,))



@dataclass(frozen=True)
class DealDamageToTargetCreatureWithExcessLanderRider(Effect):
    """Orbital Plunge `{3}{R}`:
    "Orbital Plunge deals 6 damage to target creature. If excess damage
    was dealt this way, create a Lander token."

    Excess damage is the portion of the 6 that exceeded what was needed
    to mark lethal on the target — ``damage_amount - max(0, effective
    toughness - damage already marked)``. We check >0 after the damage
    is applied. Lifelink/deathtouch are not modeled here; in real MTG
    "excess" is resolved through rule 120.3e which handles that, but
    for our simplified model, comparing amount to remaining toughness
    is sufficient.

    Bespoke amount-and-rider so the class survives PR #3's damage
    collapse; legal-target pool now consults
    ``self.targets[0].filter`` (``IsCreature()``) via the structural
    target-grammar plumbing.
    """

    amount: int = 6
    targets: tuple = ()

    def __post_init__(self) -> None:
        if not self.targets:
            from mtg_bot.effects.object_predicates import IsCreature
            from mtg_bot.effects.targets import TargetSpec
            object.__setattr__(
                self, "targets", (TargetSpec(IsCreature()),),
            )

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.actions import put_into_battlefield
        from mtg_bot.cards import TOKEN_PRINTINGS
        from mtg_bot.game import Zone, effective_toughness

        if not targets:
            return
        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            if not _slot_target_still_legal(
                state, self.targets[0], targets[0]
            ):
                return
            kind, target_id = targets[0]
            obj = next(
                (o for o in state.battlefield if o.id == target_id), None
            )
            excess = 0
            if obj is not None:
                remaining = max(0, effective_toughness(state, obj) - obj.damage)
                excess = max(0, self.amount - remaining)
            _apply_damage(state, targets[0], self.amount)
            if excess > 0:
                lander = state.new_object(
                    TOKEN_PRINTINGS["Lander"],
                    owner_id=controller_id,
                    zone=Zone.BATTLEFIELD,
                )
                put_into_battlefield(state, lander, controller_id)
                state.log.append(
                    f"Orbital Plunge dealt {excess} excess damage; "
                    f"P{controller_id} creates a Lander token"
                )
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return (
            f"deals {self.amount} damage to target creature; if excess "
            f"damage was dealt this way, create a Lander token"
        )



@dataclass(frozen=True)
class DealDamageToOpponentCreatureByArtifactCount(Effect):
    """Deal damage equal to the number of artifacts you control to
    target creature an opponent controls.

    Matches Warmaker Gunship's ETB. Counts include lands that are also
    artifacts (there aren't any in EoE) and tokens like Lander.

    Bespoke amount (artifact count) so the class survives PR #3's
    damage collapse, but the legal-target pool now consults
    ``self.targets[0].filter`` via the structural ``TargetFilter``
    grammar — ``IsCreature() & ControlledByOpponent()``.
    """

    targets: tuple = ()

    def __post_init__(self) -> None:
        if not self.targets:
            from mtg_bot.effects.object_predicates import (
                ControlledByOpponent, IsCreature,
            )
            from mtg_bot.effects.targets import TargetSpec
            object.__setattr__(
                self, "targets",
                (TargetSpec(IsCreature() & ControlledByOpponent()),),
            )

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            if not _slot_target_still_legal(
                state, self.targets[0], targets[0]
            ):
                return
            artifact_count = sum(
                1 for o in state.battlefield
                if o.controller_id == controller_id and o.printing.is_artifact
            )
            if artifact_count <= 0:
                return
            _apply_damage(state, targets[0], artifact_count)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return (
            "deals damage equal to the number of artifacts you control to "
            "target creature an opponent controls"
        )



@dataclass(frozen=True)
class DealDamageToOpponentCreatureByCardsInHand(Effect):
    """Deal damage equal to the number of cards in your hand to target
    creature an opponent controls.

    Matches Roaring Furnace's door trigger oracle: "this Room deals
    damage equal to the number of cards in your hand to target creature
    an opponent controls". The hand size is read at resolve time from
    the controller's ``hand`` (post any cast/pay-cost effects, mirroring
    MTG's resolution-time evaluation of the count).

    The Room itself is NOT in the controller's hand at resolution time
    (it's already on the battlefield by the time the door trigger
    resolves), so a hand-of-N + the cast Room produces N damage, not
    N+1 — same hand count an opponent or a third-party observer would
    see at resolve time.

    Bespoke amount (hand size) so the class survives PR #3's damage
    collapse, but the legal-target pool now consults
    ``self.targets[0].filter`` — ``IsCreature() & ControlledByOpponent()``.
    """

    targets: tuple = ()

    def __post_init__(self) -> None:
        if not self.targets:
            from mtg_bot.effects.object_predicates import (
                ControlledByOpponent, IsCreature,
            )
            from mtg_bot.effects.targets import TargetSpec
            object.__setattr__(
                self, "targets",
                (TargetSpec(IsCreature() & ControlledByOpponent()),),
            )

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            if not _slot_target_still_legal(
                state, self.targets[0], targets[0]
            ):
                return
            hand_size = len(state.player_by_id(controller_id).hand)
            if hand_size <= 0:
                return
            _apply_damage(state, targets[0], hand_size)
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return (
            "deals damage equal to the number of cards in your hand to "
            "target creature an opponent controls"
        )



@dataclass(frozen=True)
class DealDamageToTargetCreatureSpacecraftBoosted(Effect):
    """Deal ``base`` damage to a target creature; ``boosted`` instead
    if you control a Spacecraft.

    Matches Invasive Maneuvers (3/5). Any-target creature.

    Bespoke amount (Spacecraft-conditional) so the class survives
    PR #3's damage collapse; legal-target pool now consults
    ``self.targets[0].filter`` (``IsCreature()``) via the structural
    target-grammar plumbing.
    """

    base: int = 3
    boosted: int = 5
    targets: tuple = ()

    def __post_init__(self) -> None:
        if not self.targets:
            from mtg_bot.effects.object_predicates import IsCreature
            from mtg_bot.effects.targets import TargetSpec
            object.__setattr__(
                self, "targets", (TargetSpec(IsCreature()),),
            )

    def _amount(self, state: "GameState", controller_id: int) -> int:
        has_spacecraft = any(
            o.controller_id == controller_id and o.printing.is_spacecraft
            for o in state.battlefield
        )
        return self.boosted if has_spacecraft else self.base

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        prior = getattr(state, "resolving_controller_id", None)
        try:
            state.resolving_controller_id = controller_id
            if not _slot_target_still_legal(
                state, self.targets[0], targets[0]
            ):
                return
            _apply_damage(
                state, targets[0], self._amount(state, controller_id),
            )
        finally:
            state.resolving_controller_id = prior

    def describe(self) -> str:
        return (
            f"deals {self.base} damage to target creature, {self.boosted} "
            f"instead if you control a Spacecraft"
        )



@dataclass(frozen=True)
class DealDamageToOpponentCreature(DealDamage):
    """``DealDamage`` constructor sugar for "deals N damage to target
    creature an opponent controls".

    The filter ``IsCreature() & ControlledByOpponent()`` reads
    ``state.resolving_controller_id`` (pinned by
    ``Effect._enumerate_for_filter``) so the legal-target pool is
    restricted to creatures the spell's caster doesn't control.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import (
            ControlledByOpponent, IsCreature,
        )
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsCreature() & ControlledByOpponent()),),
        )



@dataclass(frozen=True)
class EachOpponentDiscards(Effect):
    """Each opponent discards ``count`` cards (untargeted; no choice of target).

    Each discarding player chooses their own discard via their agent's
    ``choose_discard`` — the ability's controller doesn't pick for them.
    """

    count: int = 1

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone
        from mtg_bot.turn import fire_opponent_discards_triggers

        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            for _ in range(self.count):
                if not p.hand:
                    break
                hand_ids = [c.id for c in p.hand]
                chosen_id = p.agent.choose_discard(state, p.id, hand_ids)
                card = next(c for c in p.hand if c.id == chosen_id)
                p.hand.remove(card)
                card.zone = Zone.GRAVEYARD
                p.graveyard.append(card)
                state.log.append(
                    f"P{p.id} discards {card.printing.name}"
                )
                fire_opponent_discards_triggers(state, p.id)

    def describe(self) -> str:
        if self.count == 1:
            return "each opponent discards a card"
        return f"each opponent discards {self.count} cards"



@dataclass(frozen=True)
class EachOpponentSacrificesCreature(Effect):
    """Each opponent sacrifices a nontoken creature of their choice.

    Each sacrificing player picks via their agent's ``choose_sacrifice``.
    If an opponent has no legal nontoken creature, they skip — no target
    required to match the standard MTG "of their choice" wording.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import sacrifice as _sacrifice

        for p in state.players:
            if p.id == controller_id or p.has_lost:
                continue
            legal = [
                o for o in state.battlefield
                if o.controller_id == p.id
                and o.printing.is_creature
                and not o.printing.is_token
            ]
            if not legal or p.agent is None:
                continue
            chosen_id = p.agent.choose_sacrifice(
                state, p.id, [o.id for o in legal]
            )
            obj = next((o for o in legal if o.id == chosen_id), legal[0])
            state.log.append(f"P{p.id} sacrifices {obj.printing.name}")
            _sacrifice(state, obj)

    def describe(self) -> str:
        return "each opponent sacrifices a nontoken creature of their choice"



@dataclass(frozen=True)
class TapEnchantedPermanent(Effect):
    """Tap the source Aura's enchanted permanent.

    Used by Stay Hidden, Stay Silent's ETB ``When this Aura enters,
    tap enchanted creature.`` Identifies the host via
    ``state.resolving_source.attached_to`` and routes through
    ``mtg_bot.turn.tap_permanent`` so the tap goes through the same
    pipeline an action-tap would. Mirrors
    ``DestroyEnchantedPermanent`` 's host-resolution pattern; safe to
    re-use for any Aura ETB that taps the host.
    """

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import tap_permanent

        aura = state.resolving_source
        if aura is None or aura.attached_to is None:
            return
        host = next(
            (o for o in state.battlefield if o.id == aura.attached_to),
            None,
        )
        if host is None:
            return
        if host.tapped:
            return
        state.log.append(
            f"{host.printing.name} is tapped by {aura.printing.name}"
        )
        tap_permanent(state, host)

    def describe(self) -> str:
        return "tap enchanted permanent"



@dataclass(frozen=True)
class TargetOpponentSacrificesCreatureOtherThanEnchanted(Effect):
    """"Target opponent sacrifices a creature of their choice other
    than enchanted creature."

    Used by Sporogenic Infection's ETB. Identifies the source Aura's
    enchanted creature via ``state.resolving_source.attached_to`` and
    excludes it from the target's sac options. The target opponent
    picks via their own ``Agent.choose_sacrifice`` — same agent path
    ``EachOpponentSacrificesCreature`` rides.

    Simplification vs. printed ``target player``: targets are
    restricted to opponents only — the standard EoE / DSK ``target
    player`` simplification used by other Caster-controls-target
    effects in this codebase. Self-targeting is never strategically
    correct for the controller of an enchant-an-opponent's-creature
    Aura; narrowing legality doesn't affect rational play. The
    nontoken filter that ``EachOpponentSacrificesCreature`` ships is
    NOT applied here — the printed text says ``a creature of their
    choice``, not ``a nontoken creature``.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id) for p in state.players
            if p.id != controller_id and not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.turn import sacrifice as _sacrifice

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "player":
            return
        target = state.player_by_id(target_id)
        if target.has_lost:
            return
        # Excluded id: the Aura's enchanted creature, if any. When the
        # Aura is enchanting a creature controlled by the target (the
        # usual case for Sporogenic Infection), the host is excluded
        # from the sac options. When the Aura is enchanting one of
        # the controller's own creatures, the target's own creatures
        # are unaffected by the exclusion (the host isn't theirs).
        aura = state.resolving_source
        excluded_id = aura.attached_to if aura is not None else None
        legal = [
            o for o in state.battlefield
            if o.controller_id == target_id
            and o.printing.is_creature
            and o.id != excluded_id
        ]
        if not legal or target.agent is None:
            return
        chosen_id = target.agent.choose_sacrifice(
            state, target_id, [o.id for o in legal]
        )
        obj = next((o for o in legal if o.id == chosen_id), legal[0])
        state.log.append(
            f"P{target_id} sacrifices {obj.printing.name} "
            f"(Sporogenic Infection)"
        )
        _sacrifice(state, obj)

    def describe(self) -> str:
        return (
            "target opponent sacrifices a creature of their choice "
            "other than enchanted creature"
        )



@dataclass(frozen=True)
class TargetOpponentDiscardsCasterChoice(Effect):
    """Target opponent reveals their hand. You choose a nonland card
    from it. That player discards the chosen card.

    Matches Temporal Intervention's main effect. Unlike
    TargetOpponentDiscards (opponent picks their own discard), here the
    CASTER picks — a directed discard.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id) for p in state.players
            if p.id != controller_id and not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "player":
            return
        target = state.player_by_id(target_id)
        if target.has_lost:
            return
        # The reveal step is "Target opponent reveals their hand" — the
        # caster sees every card in target's hand, and remembers it
        # afterward. Mark every card currently in target's hand as
        # publicly known to the caster; whatever stays in hand after
        # the directed discard preserves that mark.
        from mtg_bot.game import reveal_to
        for c in target.hand:
            reveal_to(c, controller_id)
        # Nonland cards the target could be forced to discard.
        options = [
            c for c in target.hand if not c.printing.is_land
        ]
        if not options:
            state.log.append(
                f"P{target_id} has no nonland cards — directed discard "
                f"fizzles"
            )
            return
        caster = state.player_by_id(controller_id)
        if caster.agent is None:
            chosen = options[0]
        else:
            chosen_id = caster.agent.choose_card_from_opponent_hand(
                state, controller_id, [c.id for c in options]
            )
            chosen = next(
                (c for c in options if c.id == chosen_id), options[0]
            )
        target.hand.remove(chosen)
        chosen.zone = Zone.GRAVEYARD
        target.graveyard.append(chosen)
        state.log.append(
            f"P{controller_id} forces P{target_id} to discard "
            f"{chosen.printing.name}"
        )

    def describe(self) -> str:
        return (
            "target opponent reveals their hand; you choose a nonland "
            "card from it and they discard it"
        )



@dataclass(frozen=True)
class TargetOpponentLookHandMayDiscard(Effect):
    """"Look at target opponent's hand. You may choose a nonland card
    from it. That player discards that card."

    Variant of ``TargetOpponentDiscardsCasterChoice`` that honors the
    printed ``You may`` gate via ``Agent.choose_yes_no`` — the caster
    is OFFERED the directed discard rather than forced into it. Used
    by Cracked Skull's ETB. The reveal scope is ``look at`` (caster-
    only) rather than the public ``reveals their hand``, but the
    state-tracking primitive is the same: every card in the hand
    becomes ``publicly_known_to`` the caster (one-way reveal).

    Simplification vs. the printed ``target player`` scope on Cracked
    Skull: targets are restricted to opponents only (the same
    convention the original ``TargetOpponentDiscardsCasterChoice``
    rides). Self-targeting is never strategically correct for the
    controller, so the legality narrowing doesn't affect rational
    play; documented as the standard EoE / DSK target-player
    simplification.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id) for p in state.players
            if p.id != controller_id and not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone, reveal_to

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "player":
            return
        target = state.player_by_id(target_id)
        if target.has_lost:
            return
        # ``Look at`` step: caster sees every card in target's hand and
        # remembers it. Same one-way reveal primitive
        # ``TargetOpponentDiscardsCasterChoice`` uses for its public
        # ``reveals their hand`` step — engine-equivalent for the
        # discard-direction phase that follows.
        for c in target.hand:
            reveal_to(c, controller_id)
        # Nonland options for the may-discard.
        options = [c for c in target.hand if not c.printing.is_land]
        if not options:
            state.log.append(
                f"P{target_id} has no nonland cards — directed discard "
                f"declines silently"
            )
            return
        caster = state.player_by_id(controller_id)
        # The "may" gate. Default-decline if the caster has no agent;
        # otherwise the agent's ``choose_yes_no`` answers the prompt.
        if caster.agent is None:
            return
        if not caster.agent.choose_yes_no(
            state, controller_id,
            f"Force P{target_id} to discard a nonland card?",
        ):
            state.log.append(
                f"P{controller_id} declines the may-discard against "
                f"P{target_id}"
            )
            return
        chosen_id = caster.agent.choose_card_from_opponent_hand(
            state, controller_id, [c.id for c in options]
        )
        chosen = next(
            (c for c in options if c.id == chosen_id), options[0]
        )
        target.hand.remove(chosen)
        chosen.zone = Zone.GRAVEYARD
        target.graveyard.append(chosen)
        state.log.append(
            f"P{controller_id} chooses P{target_id}'s "
            f"{chosen.printing.name} for the may-discard"
        )

    def describe(self) -> str:
        return (
            "look at target opponent's hand; you may choose a nonland "
            "card from it and they discard it"
        )



@dataclass(frozen=True)
class TargetOpponentDiscards(Effect):
    """Target opponent discards ``count`` cards.

    The discarding player (the target) chooses which cards to discard — the
    target player's agent makes the pick via ``choose_discard``.
    """

    count: int = 1

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("player", p.id)
            for p in state.players
            if p.id != controller_id and not p.has_lost
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import Zone

        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "player":
            return
        p = state.player_by_id(target_id)
        if p.has_lost:
            return
        for _ in range(self.count):
            if not p.hand:
                break
            hand_ids = [c.id for c in p.hand]
            chosen_id = p.agent.choose_discard(state, p.id, hand_ids)
            card = next(c for c in p.hand if c.id == chosen_id)
            p.hand.remove(card)
            card.zone = Zone.GRAVEYARD
            p.graveyard.append(card)
            state.log.append(
                f"P{target_id} discards {card.printing.name}"
            )

    def describe(self) -> str:
        if self.count == 1:
            return "target opponent discards a card"
        return f"target opponent discards {self.count} cards"



@dataclass(frozen=True)
class PumpTargetArtifactOrCreatureUntilEOT(Pump):
    """``Pump`` constructor sugar for "target artifact or creature gets
    +N/+M and gains KW UEOT" (Reroute Systems mode 1, Mental
    Modulation's pump arm, …).

    Filter is ``IsArtifact() | IsCreature()``. The unified
    ``Pump.resolve`` re-verifies the slot's filter at resolution per
    CR 608.2b — a target that lost both relevant card types between
    cast and resolve fails the recheck and the pump is skipped.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsArtifact, IsCreature
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsArtifact() | IsCreature()),),
        )

    def describe(self) -> str:
        parts: list[str] = []
        if self.power_bonus or self.toughness_bonus:
            parts.append(
                f"target artifact or creature gets "
                f"+{self.power_bonus}/+{self.toughness_bonus}"
            )
        if self.granted_keywords:
            kw = ", ".join(sorted(self.granted_keywords))
            if parts:
                parts.append(f"gains {kw}")
            else:
                parts.append(
                    f"target artifact or creature gains {kw}"
                )
        return " and ".join(parts) + " until end of turn."



@dataclass(frozen=True)
class DealDamageToTargetTappedCreature(DealDamage):
    """``DealDamage`` constructor sugar for "deals N damage to target
    tapped creature".

    Filter is ``IsCreature() & IsTapped()``; the unified
    ``DealDamage.resolve`` re-verifies the slot's filter against the
    current battlefield at resolution time, so a target that
    untapped between cast and resolve fails the recheck and the
    damage application is skipped per CR 608.2b.
    """

    def __post_init__(self) -> None:
        from mtg_bot.effects.object_predicates import IsCreature, IsTapped
        from mtg_bot.effects.targets import TargetSpec
        object.__setattr__(
            self, "targets",
            (TargetSpec(IsCreature() & IsTapped()),),
        )



@dataclass(frozen=True)
class MagicalHackTextChangeEffect(Effect):
    """Magical Hack: 'Change the text of target spell or permanent
    by replacing all instances of one basic land type with another.'

    Implementation scope: targets a basic land on the battlefield.
    Picks the new basic land type via the controller's
    ``choose_mana_color`` callback, mapped W/U/B/R/G to
    Plains/Island/Swamp/Mountain/Forest. The OLD basic land type is
    derived automatically from the target's effective subtypes
    (basic lands typically have exactly one). Writes a layer-3
    ``"replace_basic_land_type"`` registry entry sourced by the
    resolving spell with ``params=(old, new)`` and
    ``expires_at=None`` — text changes persist indefinitely (CR
    612). The sorcery itself goes to graveyard, so the entry is
    never auto-dropped.

    Limitations: only handles basic-land-type substitution on basic
    lands. Real Magical Hack also rewrites rules text on non-basic
    permanents that mention basic land types (e.g., a creature with
    "Forestwalk"); this engine doesn't yet model landwalk so the
    extension is a no-op there.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.printing.is_basic_land
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import (
            _BASIC_LAND_TYPE_TO_COLOR, effective_subtypes,
        )

        if not targets:
            return
        _, target_id = targets[0]
        target = next(
            (o for o in state.battlefield if o.id == target_id), None,
        )
        if target is None:
            return
        # Pick the OLD basic land type from the target's current
        # effective subtypes. Basic lands typically have exactly one
        # basic-land-type subtype.
        current_subs = effective_subtypes(state, target)
        old = next(
            (s for s in current_subs if s in _BASIC_LAND_TYPE_TO_COLOR),
            None,
        )
        if old is None:
            state.log.append(
                f"{target.printing.name} has no basic land type — "
                f"Magical Hack fizzles"
            )
            return
        # Pick the NEW basic land type via choose_mana_color, excluding
        # the OLD color (per Magical Hack: "ONE basic land type with
        # ANOTHER" implies they must differ).
        color_to_type = {
            v: k for k, v in _BASIC_LAND_TYPE_TO_COLOR.items()
        }
        old_color = _BASIC_LAND_TYPE_TO_COLOR[old]
        options = [c for c in ["W", "U", "B", "R", "G"] if c != old_color]
        chosen = state.players[controller_id].agent.choose_mana_color(
            state, controller_id, options,
        )
        new = color_to_type.get(chosen)
        if new is None or new == old:
            state.log.append(
                f"Magical Hack: invalid new basic land type, fizzling"
            )
            return
        src = state.resolving_source
        source_id = src.id if src is not None else target.id
        state.add_continuous_effect(
            source_id=source_id,
            layer=3,
            kind="replace_basic_land_type",
            target_id=target.id,
            params=(old, new),
            description=(
                f"Magical Hack rewrites {target.printing.name}: "
                f"{old} -> {new}"
            ),
        )
        state.log.append(
            f"Magical Hack rewrites {target.printing.name}: "
            f"{old} -> {new}"
        )

    def describe(self) -> str:
        return (
            "change the text of target spell or permanent by replacing "
            "all instances of one basic land type with another"
        )



@dataclass(frozen=True)
class SleightOfMindTextChangeEffect(Effect):
    """Sleight of Mind: 'Change the text of target spell or
    permanent by replacing all instances of one color word with
    another.' Layer 3 text-changing.

    Implementation scope: rewrites color words in
    ``protection_from_color`` on the target's printing. The OLD
    color is auto-derived from the target's current effective
    protection (the only color-word reference our engine
    currently models on permanents); the controller picks the NEW
    color via ``choose_mana_color``. Writes a layer-3
    ``"replace_color_word"`` registry entry sourced by the
    resolving spell with ``params=(old, new)`` and
    ``expires_at=None`` (text changes persist per CR 612).

    Limitations: doesn't rewrite color words inside other rules
    text (e.g., "Whenever a black creature ETBs, ..."). The engine
    doesn't have such color-keyed triggers in scope, so the
    extension is a no-op there.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        # Permits any permanent on the battlefield. In practice
        # only the protection-from-color holders get observable
        # results; targeting elsewhere just writes a no-op entry.
        return [
            ("permanent", o.id) for o in state.battlefield
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        from mtg_bot.game import effective_protection_from_colors

        if not targets:
            return
        _, target_id = targets[0]
        target = next(
            (o for o in state.battlefield if o.id == target_id), None,
        )
        if target is None:
            return
        # Derive the OLD color from the target's current effective
        # protection set. If protection is empty or has multiple
        # colors, we can't unambiguously rewrite — fizzle.
        prot = effective_protection_from_colors(state, target)
        if len(prot) != 1:
            state.log.append(
                f"Sleight of Mind fizzles: {target.printing.name} "
                f"has no single color word to rewrite"
            )
            return
        (old,) = prot
        options = [c for c in ["W", "U", "B", "R", "G"] if c != old]
        chosen = state.players[controller_id].agent.choose_mana_color(
            state, controller_id, options,
        )
        if chosen not in options:
            state.log.append(
                f"Sleight of Mind fizzles: invalid new color choice"
            )
            return
        src = state.resolving_source
        source_id = src.id if src is not None else target.id
        state.add_continuous_effect(
            source_id=source_id,
            layer=3,
            kind="replace_color_word",
            target_id=target.id,
            params=(old, chosen),
            description=(
                f"Sleight of Mind rewrites {target.printing.name}: "
                f"{old} -> {chosen}"
            ),
        )
        state.log.append(
            f"Sleight of Mind rewrites {target.printing.name}: "
            f"{old} -> {chosen}"
        )

    def describe(self) -> str:
        return (
            "change the text of target spell or permanent by replacing "
            "all instances of one color word with another"
        )



@dataclass(frozen=True)
class CytoshapeBecomeCopyUEOT(Effect):
    """Cytoshape (Saviors of Kamigawa, simplified): "Until end of
    turn, target creature becomes a copy of another target
    creature." Layer 1 copy effect with ``expires_at="cleanup"``.

    The architectural payoff over Clone (which is permanent /
    ``expires_at=None``): the printing mutation must REVERT at
    cleanup. Wired via the new ``revert_printing_if_no_layer1_copies``
    helper called from ``run_cleanup_step`` after layer-1 entries
    expire.

    Demo simplifications:
    - Drops the real card's "Sacrifice a Spirit" cost.
    - Drops the "except it isn't a Spirit" rider on the resulting
      copy.
    - Activate-as-sorcery is moot because we ship as a sorcery
      directly.

    Two creature targets: slot 0 is the creature being changed;
    slot 1 is the source whose printing is copied. The two must
    differ ("another target creature") — fizzles if same.
    """

    def requires_target(self) -> bool:
        return True

    def target_count_range(self) -> tuple[int, int]:
        return (2, 2)

    def target_slot_kinds(self) -> tuple[str, ...]:
        return ("creature", "creature")

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if len(targets) < 2:
            return
        first_id = targets[0][1]
        second_id = targets[1][1]
        if first_id == second_id:
            state.log.append(
                "Cytoshape fizzles: targets must be different creatures"
            )
            return
        first = next(
            (o for o in state.battlefield if o.id == first_id), None,
        )
        second = next(
            (o for o in state.battlefield if o.id == second_id), None,
        )
        if first is None or second is None:
            return
        if first.original_printing is None:
            first.original_printing = first.printing
        copied_printing = second.printing
        first.printing = copied_printing
        src = state.resolving_source
        source_id = src.id if src is not None else first.id
        state.add_continuous_effect(
            source_id=source_id,
            layer=1,
            kind="is_copy",
            target_id=first.id,
            params=(copied_printing,),
            expires_at="cleanup",
            description=(
                f"{first.original_printing.name} becomes a copy of "
                f"{copied_printing.name} UEOT"
            ),
        )
        state.log.append(
            f"{first.original_printing.name} becomes a copy of "
            f"{copied_printing.name} until end of turn"
        )

    def describe(self) -> str:
        return (
            "until end of turn, target creature becomes a copy of "
            "another target creature"
        )



@dataclass(frozen=True)
class ScuttlemuttColorChangeUEOT(Effect):
    """Scuttlemutt: '{T}: Target creature becomes the color or colors
    of your choice until end of turn.' Layer 5 color-changing effect.

    Demo scope: picks exactly one color via ``choose_mana_color``.
    Real Scuttlemutt allows zero, one, or multiple colors; the demo
    keeps it to a single color since that's enough to exercise
    block legality with protection-from-color.

    Resolution writes a layer-5 ``"set_colors"`` registry entry
    sourced by Scuttlemutt with ``params=(frozenset({chosen}),)``
    and ``expires_at="cleanup"``. The entry REPLACES the target's
    color set (matching real-card "becomes" semantics, not
    "becomes ... in addition to"). At cleanup the entry is dropped
    via ``expire_continuous_effects("cleanup")`` and the target
    reverts to its printed colors.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("creature", o.id)
            for o in state.battlefield
            if o.printing.is_creature
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        _, target_id = targets[0]
        target = next(
            (o for o in state.battlefield if o.id == target_id), None,
        )
        if target is None:
            return
        chosen = state.players[controller_id].agent.choose_mana_color(
            state, controller_id, ["W", "U", "B", "R", "G"],
        )
        if chosen not in ("W", "U", "B", "R", "G"):
            return
        src = state.resolving_source
        source_id = src.id if src is not None else target.id
        state.add_continuous_effect(
            source_id=source_id,
            layer=5,
            kind="set_colors",
            target_id=target.id,
            params=(frozenset({chosen}),),
            expires_at="cleanup",
            description=(
                f"{target.printing.name} becomes {chosen} until end "
                f"of turn"
            ),
        )
        state.log.append(
            f"{target.printing.name} becomes {chosen} until end of turn"
        )

    def describe(self) -> str:
        return (
            "target creature becomes the color of your choice until "
            "end of turn"
        )



@dataclass(frozen=True)
class LockOrUnlockDoorOfTargetRoomYouControl(Effect):
    """Keys to the House: "Lock or unlock a door of target Room you
    control. Activate only as a sorcery."

    Resolves on a target Room permanent the controller controls. The
    controller chooses one of the Room's two doors to toggle (via
    ``choose_target`` over the legal-target pool — Rooms — and then
    ``choose_mode`` over a per-Room ``["lock <X>", "unlock <X>", ...]``
    label list). The chosen half flips its membership in
    ``GameObject.unlocked_halves``: a locked half being unlocked is
    added to the set; an unlocked half being locked is removed.

    The ``Activate only as a sorcery.`` rider is honored at the
    activated-ability level (``sorcery_speed=True``), not in the
    effect; the effect itself is happy to resolve at any point.

    Unlike the printed ``UnlockRoomHalf`` action (sorcery-speed special
    action that pays the half's mana cost), this effect skips the cost
    payment — the controller's "Keys" sacrifice already paid for the
    privilege. The unlock side-effects (Eerie observers, etc.) follow
    the same path the action takes: the predicate-style ENTERS event
    on a half-name filter does NOT re-fire here because the engine
    layer's ``_unlock_room_half_with_triggers`` is the canonical entry
    point. We mirror its observable state change (modify
    ``unlocked_halves``) without re-firing the per-half on-unlock
    triggered abilities — keeps the effect surgical.
    """

    def requires_target(self) -> bool:
        return True

    def legal_targets(
        self, state: "GameState", controller_id: int
    ) -> list[Target]:
        return [
            ("permanent", o.id)
            for o in state.battlefield
            if o.controller_id == controller_id
            and o.printing.is_room
        ]

    def resolve(
        self, state: "GameState", controller_id: int,
        targets: tuple[Target, ...] = (),
    ) -> None:
        if not targets:
            return
        kind, target_id = targets[0]
        if kind != "permanent":
            return
        obj = next(
            (o for o in state.battlefield if o.id == target_id), None
        )
        if obj is None or obj.controller_id != controller_id:
            return
        if not obj.printing.is_room:
            return
        # Build the choose_mode label list: one entry per door, each
        # pre-prefixed with "lock" or "unlock" depending on the door's
        # current state. The controller's pick maps directly to the
        # door name and the new state.
        labels: list[str] = []
        actions: list[tuple[str, str]] = []  # (door_name, "lock"/"unlock")
        for half in obj.printing.room_halves:
            if half.name in obj.unlocked_halves:
                labels.append(f"lock {half.name}")
                actions.append((half.name, "lock"))
            else:
                labels.append(f"unlock {half.name}")
                actions.append((half.name, "unlock"))
        if not labels:
            return
        p = state.player_by_id(controller_id)
        pick = 0
        if p.agent is not None:
            pick = p.agent.choose_mode(state, controller_id, labels)
            pick = max(0, min(pick, len(labels) - 1))
        door_name, op = actions[pick]
        if op == "unlock":
            obj.unlocked_halves = obj.unlocked_halves | {door_name}
            state.log.append(
                f"P{controller_id} unlocks {door_name} on "
                f"{obj.printing.name}"
            )
        else:
            obj.unlocked_halves = obj.unlocked_halves - {door_name}
            state.log.append(
                f"P{controller_id} locks {door_name} on "
                f"{obj.printing.name}"
            )

    def describe(self) -> str:
        return "lock or unlock a door of target Room you control"


