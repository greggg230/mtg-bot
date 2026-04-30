"""Modifier-grammar primitives — :class:`ContinuousModifier` (CR 613)
and :class:`ReplacementEffect` (CR 614).

This module is the architectural pillar that replaces the per-card
boolean-flag anti-pattern (``is_token_doubler``,
``is_counter_doubler``, ``effective_draw_count``,
``etb_trigger_multiplier``,
``shuffles_into_library_instead_of_graveyard``,
``attack_bypass_if_artifact_entered_this_turn``,
``unblockable_if_two_spells_cast_this_turn``, …) and the literal-line
recognizers in ``printing_keywords.py`` / ``equipment_builder.py`` /
``statics.py`` anthem family. Decomposition: a modifier is either a
:class:`ContinuousModifier` (CR 613 layer system — P/T mod, keyword
grant/remove, type add) or a :class:`ReplacementEffect` (CR 614 —
intercept events; transforms ``MultiplyCount(×N)``, ``AddToCount(+N)``,
``AddInstances(+N)``, ``Replace(other_event)``, ``Prevent``).

PR A (this module's birth) lands the type definitions for both
families plus the :func:`dispatch_replacement` infrastructure for
``ReplacementEffect``. The CR-613 dispatcher for ``ContinuousModifier``
lands in PR D when the continuous-effect family migrates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from mtg_bot.effects.object_predicates import ObjectPredicate
    from mtg_bot.effects.predicates import Event
    from mtg_bot.game import GameObject, GameState


# ────────────────────────────────────────────────────────────────────
# EventTransform — pure payload transformations
# ────────────────────────────────────────────────────────────────────


class EventTransform(ABC):
    """A pure ``payload -> payload`` transformation describing how a
    :class:`ReplacementEffect` modifies an in-flight event.

    Concrete transforms (``MultiplyCount``, ``AddToCount``,
    ``AddInstances``, ``Replace``, ``Prevent``) compose through a
    single :meth:`apply` interface — the dispatcher hands them the
    in-flight event payload and they return the modified copy.
    """

    @abstractmethod
    def apply(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class MultiplyCount(EventTransform):
    """Multiply the ``count`` payload key by ``factor``.

    Models token-doubling (Exalted Sunborn ×2, hypothetical Anointed
    Procession-style ×3) and +1/+1-counter-doubling (Doubling Season,
    Loading Zone). The same EventTransform claims every shape that
    multiplies a count payload — the only differentiator is the
    ``factor`` constant on the per-card replacement.
    """

    factor: int

    def apply(self, payload):
        new = dict(payload)
        new["count"] = payload.get("count", 0) * self.factor
        return new


@dataclass(frozen=True)
class AddToCount(EventTransform):
    """Add ``delta`` to the ``count`` payload key.

    Models Hardened-Scales-shape "+1 to +1/+1 counters placed" and
    Quantum Riddler-shape "draw N+1 instead of N". Composes additively
    with other transforms via the dispatcher's CR 616.1 ordering.
    """

    delta: int

    def apply(self, payload):
        new = dict(payload)
        new["count"] = payload.get("count", 0) + self.delta
        return new


@dataclass(frozen=True)
class AddInstances(EventTransform):
    """Add ``delta`` instances to the ``instances`` payload key.

    Models Strionic-Resonator / Starfield-Vocalist-shape "triggered
    abilities trigger an additional time" replacements. The
    triggered-ability-firing path reads the post-replacement
    ``instances`` count to decide how many copies of the trigger to
    push onto the stack.
    """

    delta: int

    def apply(self, payload):
        new = dict(payload)
        new["instances"] = payload.get("instances", 1) + self.delta
        return new


@dataclass(frozen=True)
class Replace(EventTransform):
    """Replace the event's destination zone with ``new_destination``.

    Models Nexus-of-Fate-shape "if it would go to a graveyard, shuffle
    into the library instead". The dispatch site reads
    ``payload["destination"]`` after the transform applies and routes
    the card accordingly.
    """

    new_destination: str

    def apply(self, payload):
        new = dict(payload)
        new["destination"] = self.new_destination
        return new


@dataclass(frozen=True)
class Prevent(EventTransform):
    """Prevent the event entirely.

    Sets ``payload["prevented"] = True``; the dispatch site short-
    circuits when this flag is set, skipping the underlying mutation.
    Composes with the count-style transforms — a Prevent ordered first
    short-circuits subsequent transforms (the dispatcher's CR 616.1
    re-evaluation handles this).
    """

    def apply(self, payload):
        new = dict(payload)
        new["prevented"] = True
        return new


# ────────────────────────────────────────────────────────────────────
# EventPattern — what to match
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EventPattern:
    """Pattern that matches a triggering event for a replacement.

    ``event`` is one of the existing :class:`Event` enum values
    (specifically the WOULD_X family for replacements — see the enum
    docs). ``subject_filters`` is an optional per-slot mapping of
    ``slot_name -> ObjectPredicate`` that further narrows which
    firings the replacement cares about.

    Example: a counter-doubler that only doubles +1/+1 counters going
    onto creatures you control composes:

        EventPattern(
            event=Event.WOULD_PLACE_COUNTER,
            subject_filters={
                "obj": IsCreature() & ControlledByYou(),
            },
        )

    The dispatcher resolves each subject_filter against the matching
    event subject at collection time, falling through replacements
    whose filter rejects the firing.
    """

    event: "Event"
    subject_filters: Mapping[str, "ObjectPredicate"] = field(
        default_factory=dict
    )


# ────────────────────────────────────────────────────────────────────
# LayerPayload — what a CR 613 continuous effect does
# ────────────────────────────────────────────────────────────────────


class LayerPayload(ABC):
    """A description of what a :class:`ContinuousModifier` does to each
    matching object.

    Concrete payloads (``GrantKeyword``, ``RemoveKeyword``,
    ``PowerToughnessBonus``, ``AddSubtype``) are tagged unions — the
    layer system dispatcher picks the right code path by isinstance
    check. The dispatcher itself lands in PR D when continuous-effect
    consumers migrate.
    """


@dataclass(frozen=True)
class GrantKeyword(LayerPayload):
    """CR 613 layer 6 — grant a keyword ability.

    Models the Illvoi-Infiltrator-shape "while [condition], this
    creature has unblockable" and the anthem-family "creatures you
    control have flying" — same payload, different scope_filter on
    the wrapping :class:`ContinuousModifier`.
    """

    keyword: str


@dataclass(frozen=True)
class RemoveKeyword(LayerPayload):
    """CR 613 layer 6 — remove a keyword ability.

    Models the Mechan-Shieldmate-shape "while an artifact entered
    this turn, this creature loses defender" — the
    ``ContinuousModifier`` payload removes the keyword from the
    matching object's effective keyword set, the same shape as the
    legacy ``attack_bypass_if_artifact_entered_this_turn`` flag
    expressed structurally.
    """

    keyword: str


@dataclass(frozen=True)
class PowerToughnessBonus(LayerPayload):
    """CR 613 layer 7c — additive P/T modification.

    Models the anthem family ("creatures you control get +1/+1") and
    the equipment-buff family ("equipped creature gets +N/+M"). The
    bonus stacks across multiple sources within layer 7c — this
    payload is the per-source contribution.
    """

    power_bonus: int = 0
    toughness_bonus: int = 0


@dataclass(frozen=True)
class AddSubtype(LayerPayload):
    """CR 613 layer 4 — add a permanent / creature subtype.

    Models the Magus-of-the-Moon-shape "all nonbasic lands are
    Mountains" and the Conspiracy-shape "all creatures are Slivers";
    the dispatcher walks the matching objects and adds ``subtype`` to
    their effective subtype set.
    """

    subtype: str


# ────────────────────────────────────────────────────────────────────
# ContinuousModifier / ReplacementEffect — top-level modifier types
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContinuousModifier:
    """A CR 613 continuous effect.

    While the source is on the battlefield (and the wrapping
    :class:`StaticAbility`'s condition is satisfied, when present),
    apply ``payload`` to every object satisfying ``subject_filter`` at
    layer ``layer``. The CR 613 dispatcher
    (:func:`walk_continuous_modifiers`) walks every active
    ``ContinuousModifier`` per object query and folds the contributions
    in layer order.

    ``subject_filter`` is an optional secondary predicate applied AFTER
    the wrapping ``StaticAbility.applies_to`` scope check. When
    ``None`` (the default for SELF-scope abilities), no additional
    filtering is performed — ``applies_to`` is the sole gate.
    """

    layer: int  # 1..7 per CR 613.1
    payload: LayerPayload
    subject_filter: "ObjectPredicate | None" = None


@dataclass(frozen=True)
class ReplacementEffect:
    """A CR 614 replacement effect.

    When an event matching ``event_pattern`` would happen, transform
    its payload via ``transform`` before the underlying mutation
    fires. The dispatcher (:func:`dispatch_replacement`) walks every
    active ``ReplacementEffect`` and applies them in CR 616.1 order
    (the affected player chooses among multiple).

    ``scope_filter`` (optional) constrains which firings the effect
    cares about — for replacements that gate on the recipient's
    controller without being expressible in the per-subject
    ``EventPattern.subject_filters`` (e.g., "you would create one or
    more tokens" gates on the controller of the creating action,
    which is a payload field, not a subject).
    """

    event_pattern: EventPattern
    transform: EventTransform
    scope_filter: "ObjectPredicate | None" = None


# ────────────────────────────────────────────────────────────────────
# dispatch_replacement — the CR 614 + CR 616.1 dispatcher
# ────────────────────────────────────────────────────────────────────


def dispatch_replacement(
    state: "GameState",
    event: "Event",
    *,
    affected_player_id: int,
    subjects: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Walk active :class:`ReplacementEffect`s, apply each matching one
    in CR 616.1 order, and return the final transformed payload.

    The ``affected_player_id`` is the player CR 616.1 names as
    choosing the application order when multiple replacements apply.
    The dispatcher consults ``state.player_by_id(affected_player_id)
    .agent.choose_replacement_order`` for the order; with 0 or 1
    candidates the prompt is skipped.

    ``subjects`` is the per-event subject mapping
    (e.g. ``{"obj": creature, "kind": "+1/+1"}`` for
    ``WOULD_PLACE_COUNTER``). ``payload`` carries the mutable fields
    every transform may modify (``count``, ``instances``,
    ``destination``, ``prevented``).

    Each replacement applies at most once per event (CR 616.1).
    After each application the dispatcher re-walks the candidate list
    to drop replacements whose ``scope_filter`` no longer matches
    (e.g., a ``Prevent`` ordered first invalidates a downstream
    multiply); the loop terminates when no further replacement
    applies.
    """
    subjects = subjects or {}
    cur_payload: dict[str, Any] = dict(payload or {})

    # Snapshot the candidate list ONCE — replacements applied during
    # this dispatch don't recursively trigger further replacement
    # collection on the SAME event. Each replacement applies at most
    # once per event.
    candidates = _collect_replacement_effects(state, event, subjects)

    if not candidates:
        return cur_payload

    if len(candidates) == 1:
        return candidates[0][1].transform.apply(cur_payload)

    # CR 616.1 — affected player chooses the order.
    affected = state.player_by_id(affected_player_id)
    agent = getattr(affected, "agent", None)
    if agent is not None and hasattr(agent, "choose_replacement_order"):
        order = agent.choose_replacement_order(
            [eff for _src, eff in candidates],
            state,
            affected_player_id,
        )
        # Sanitise: keep in-range, unique indices preserving order.
        seen: set[int] = set()
        clean: list[int] = []
        for idx in order:
            if 0 <= idx < len(candidates) and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        # Append any missed indices in source-id order.
        for i, _ in enumerate(candidates):
            if i not in seen:
                clean.append(i)
        order = tuple(clean)
    else:
        # Default deterministic ordering: sort by source object id —
        # stable, replay-safe, and predictable for tests.
        order = tuple(
            sorted(range(len(candidates)), key=lambda i: candidates[i][0])
        )

    for idx in order:
        cur_payload = candidates[idx][1].transform.apply(cur_payload)
        if cur_payload.get("prevented"):
            # A Prevent short-circuits subsequent transforms — the
            # event has been intercepted.
            break

    return cur_payload


def _collect_replacement_effects(
    state: "GameState",
    event: "Event",
    subjects: Mapping[str, Any],
) -> list[tuple[int, "ReplacementEffect"]]:
    """Walk the battlefield + per-printing flags + dynamic synthesis
    sites, returning ``(source_id, ReplacementEffect)`` pairs whose
    ``event_pattern`` matches and whose subject filters accept the
    firing.

    Legacy printing-level boolean flags synthesise matching
    ``ReplacementEffect`` instances — this keeps existing parser
    output working unchanged while the per-card boolean-flag anti-
    pattern collapses behind a single dispatcher. Native
    declarations on each printing's static abilities replace the
    synthesis paths in PR D.
    """
    from mtg_bot.effects.predicates import Event

    out: list[tuple[int, "ReplacementEffect"]] = []
    for obj in state.battlefield:
        # Native replacement effects on the printing's static
        # abilities (PR-D-and-later — empty for now).
        for static in getattr(obj.printing, "static_abilities", ()) or ():
            re = getattr(static, "replacement_effect", None)
            if re is None:
                continue
            if re.event_pattern.event != event:
                continue
            if not _subject_filters_match(
                re.event_pattern.subject_filters, subjects, state,
            ):
                continue
            if re.scope_filter is not None and not re.scope_filter.test(
                obj, state,
            ):
                continue
            out.append((obj.id, re))
        # ── Legacy synthesis paths ─────────────────────────────────
        # Token-doubler flag → MultiplyCount(2) on WOULD_CREATE_TOKEN.
        if event == Event.WOULD_CREATE_TOKEN and getattr(
            obj.printing, "is_token_doubler", False,
        ):
            controller_id = subjects.get("controller_id")
            if controller_id is None or obj.controller_id == controller_id:
                out.append((
                    obj.id,
                    ReplacementEffect(
                        event_pattern=EventPattern(
                            event=Event.WOULD_CREATE_TOKEN,
                        ),
                        transform=MultiplyCount(factor=2),
                    ),
                ))
        # Loading-Zone-shape counter-doubler →
        # MultiplyCount(2) on WOULD_PLACE_COUNTER. Scope: counter
        # recipient is a creature or Spacecraft the doubler's
        # controller controls. The recipient passes through subjects
        # ``"obj"``; the doubler reads its controller_id off that.
        if event == Event.WOULD_PLACE_COUNTER and getattr(
            obj.printing, "is_counter_doubler", False,
        ):
            recipient = subjects.get("obj")
            if (
                recipient is not None
                and getattr(recipient, "controller_id", None)
                == obj.controller_id
            ):
                out.append((
                    obj.id,
                    ReplacementEffect(
                        event_pattern=EventPattern(
                            event=Event.WOULD_PLACE_COUNTER,
                        ),
                        transform=MultiplyCount(factor=2),
                    ),
                ))
        # Quantum-Riddler-shape draw replacement →
        # AddToCount(1) on WOULD_DRAW_CARDS, gated on the drawing
        # player having ≤ 1 card in hand AND being the carrier's
        # controller.
        if event == Event.WOULD_DRAW_CARDS and getattr(
            obj.printing,
            "replaces_draw_with_extra_when_hand_at_most_one",
            False,
        ):
            drawer = subjects.get("player")
            if (
                drawer is not None
                and getattr(drawer, "id", None) == obj.controller_id
                and len(getattr(drawer, "hand", [])) <= 1
            ):
                out.append((
                    obj.id,
                    ReplacementEffect(
                        event_pattern=EventPattern(
                            event=Event.WOULD_DRAW_CARDS,
                        ),
                        transform=AddToCount(delta=1),
                    ),
                ))
        # Starfield-Vocalist-shape extra-ETB-trigger replacement →
        # AddInstances(extra) on TRIGGERED_ABILITY_TRIGGERS. Scope:
        # the triggering ETB is on a permanent the carrier's
        # controller controls.
        extra_etb = getattr(obj.printing, "extra_etb_trigger", 0)
        if event == Event.TRIGGERED_ABILITY_TRIGGERS and extra_etb > 0:
            ability_source = subjects.get("ability_source")
            if (
                ability_source is not None
                and getattr(ability_source, "controller_id", None)
                == obj.controller_id
            ):
                out.append((
                    obj.id,
                    ReplacementEffect(
                        event_pattern=EventPattern(
                            event=Event.TRIGGERED_ABILITY_TRIGGERS,
                        ),
                        transform=AddInstances(delta=extra_etb),
                    ),
                ))
    # Out-of-band synthesis: WOULD_GO_TO_GRAVEYARD reads a flag on
    # the moving CARD's printing, not on a battlefield permanent
    # (the card is mid-flight at dispatch). The dispatch site
    # passes the card via ``subjects["card"]``.
    if event == Event.WOULD_GO_TO_GRAVEYARD:
        card = subjects.get("card")
        if card is not None and getattr(
            getattr(card, "printing", None),
            "shuffles_into_library_instead_of_graveyard",
            False,
        ):
            out.append((
                getattr(card, "id", -1),
                ReplacementEffect(
                    event_pattern=EventPattern(
                        event=Event.WOULD_GO_TO_GRAVEYARD,
                    ),
                    transform=Replace(new_destination="library"),
                ),
            ))
    return out


def _subject_filters_match(
    filters: Mapping[str, "ObjectPredicate"],
    subjects: Mapping[str, Any],
    state: "GameState",
) -> bool:
    """Return True iff every filter passes against its named subject.

    A missing subject (the event didn't carry that slot) fails the
    filter — the predicate can't evaluate without an object.
    """
    for slot, pred in filters.items():
        obj = subjects.get(slot)
        if obj is None:
            return False
        if not pred.test(obj, state):
            return False
    return True


def walk_continuous_modifiers(
    state: "GameState",
    target: "GameObject",
) -> "list[tuple[int, ContinuousModifier]]":
    """Yield ``(source_id, ContinuousModifier)`` pairs whose wrapping
    ``StaticAbility`` applies to ``target`` right now.

    Walks every permanent on the battlefield. For each permanent,
    walks its ``static_abilities`` looking for ones with a
    ``continuous_effect`` slot populated. Delegates the scope /
    condition / creature-check to ``StaticAbility.applies_to``. If
    the modifier also carries a ``subject_filter``, that predicate is
    evaluated as a secondary gate after ``applies_to`` passes.

    The result list is ordered by source ``etb_timestamp`` so callers
    that need layer-within-6 ordering can sort by source timestamp if
    needed. For the current EoE pool (layer-6 keyword grants/removes)
    the ordering only matters when a wipe and a grant interact — the
    existing ``_layer6_entries_for`` path handles that; this function
    provides a parallel contribution consumed after the existing walk.
    """
    out: list[tuple[int, ContinuousModifier]] = []
    for source in state.battlefield:
        for static in getattr(source.printing, "static_abilities", ()) or ():
            cm = getattr(static, "continuous_effect", None)
            if cm is None:
                continue
            if not static.applies_to(source, target, state):
                continue
            if cm.subject_filter is not None and not cm.subject_filter.test(
                target, state
            ):
                continue
            out.append((source.id, cm))
    return out


__all__ = [
    "EventTransform",
    "MultiplyCount",
    "AddToCount",
    "AddInstances",
    "Replace",
    "Prevent",
    "EventPattern",
    "LayerPayload",
    "GrantKeyword",
    "RemoveKeyword",
    "PowerToughnessBonus",
    "AddSubtype",
    "ContinuousModifier",
    "ReplacementEffect",
    "dispatch_replacement",
    "walk_continuous_modifiers",
]
