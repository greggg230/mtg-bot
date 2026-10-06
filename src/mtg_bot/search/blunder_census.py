"""Blunder census: a few precise detectors over a REAL game's decision
stream, for one seat (the "agent").

Feed it every decision moment of a game (the driver calls
:meth:`BlunderCensus.observe_moment` with the live adapter before each
decision and :meth:`BlunderCensus.after_decision` right after the
decision is applied, then :meth:`BlunderCensus.finish`). The census
reads the engine state only; it never mutates it.

Detectors (each is narrow on purpose; false-positive risks are listed
so a rate can be read honestly — also in ``docs/SEARCH_HORIZON.md``):

``self_target`` — a single-target spell / ability the agent controls
    whose effect includes damage, destroy or exile-a-permanent, aimed at
    the agent's own permanent or the agent's own face. Split by whether
    an opponent-side target was ALSO legal at that moment
    (``self_target_avoidable``) or only own targets existed
    (``self_target_forced`` — then the blunder is putting it on the
    stack at all; for triggers that is not a choice). Targets are picked
    by the adapter's tactical policy (``DefaultTacticalAgent.
    choose_target`` = first legal), not by search, in production play.
    FP risk: modal spells whose chosen mode is not the harmful one (the
    effect tree contains every mode); ETB-blink style "exile, then
    return" effects (excluded by class name where recognisable).

``chump_attack`` — an agent attacker that was blocked, left the
    battlefield during that combat, and none of its blockers left the
    battlefield (no trade), attacker without trample. FP risk: an
    alpha strike where other attackers got through for the win / big
    damage (``chump_attack_no_other_damage`` additionally requires that
    the defending player lost no life in that combat), a creature
    removed by an instant mid-combat (counted as died-to-block).

``fizzle`` — an agent spell / ability with >= 1 object target where
    every object target had already left the battlefield when the item
    reached the top of the stack and resolved (CR 608.2b: it does
    nothing). Countered items are excluded. FP risk: none known for
    "did nothing"; it under-counts spells that resolve with a legal but
    pointless target (e.g. a pump on a creature that is not in combat).

``missed_play`` — the agent passed priority in its own second main
    phase with an empty stack while ``legal_actions`` offered a
    sorcery-speed ``CastSpell`` (creature / sorcery / artifact /
    enchantment / planeswalker; instants and flash excluded because
    holding them up is legitimate). Counted once per turn; split into
    ``missed_play_creature`` (a creature was castable — the cleanest
    signal) and ``missed_play_other``. FP risk: deliberate sandbagging;
    a removal spell held because its only legal targets were own
    creatures (correct to hold — so ``missed_play_other`` over-counts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from mtg_bot.turn import STEP_ORDER, Step

DETECTORS: tuple[str, ...] = (
    "self_target_avoidable",
    "self_target_forced",
    "chump_attack",
    "chump_attack_no_other_damage",
    "fizzle",
    "missed_play_creature",
    "missed_play_other",
)

_STEP_INDEX = {s: i for i, s in enumerate(STEP_ORDER)}
_HARM_PREFIXES = ("DealDamage", "Destroy", "Exile", "MayHaveSourceDealDamage")
# Classes whose name starts with a harmful prefix but that never hit a
# chosen permanent / face of the caster's choosing (each-opponent,
# self-cost, graveyard / library exile, sweepers, linked returns).
_HARM_EXCLUDED = frozenset({
    "DealDamageEachOpponent",
    "DealDamageToControllerAndAddMana",
    "DealDamageToTargetOpponent",
    "DealNDamageToEachCreature",
    "DestroyEnchantedPermanent",
    "DestroyAllCreatures",
    "DestroyEachNonlandPermanentWithMVEqualToSourceChargeCounters",
    "ExileEachOpponentsGraveyard",
    "ExileTargetCardFromGraveyard",
    "ExileTopOfLibraryMayPlayThisTurn",
    "ExileTopXOfLibraryMayPlayThisTurn",
    "ExileAllArtifactsWithManaValueAtMost",
    "ExileAllGraveyards",
    "ExileAllCreaturesAndSpacecraft",
    "ExileAllMatchingUntilSourceLeaves",
})
_BENIGN_MARKERS = ("Return", "Flicker", "Blink")


def effect_harm_classes(effect: Any, _depth: int = 0) -> set[str]:
    """Names of the harmful primitive classes inside ``effect`` (walks
    dataclass fields / tuples recursively, bounded depth)."""
    out: set[str] = set()
    if effect is None or _depth > 8:
        return out
    name = type(effect).__name__
    if (
        name.startswith(_HARM_PREFIXES)
        and name not in _HARM_EXCLUDED
        and not any(m in name for m in _BENIGN_MARKERS)
    ):
        out.add(name)
    fields = getattr(effect, "__dataclass_fields__", None)
    if fields:
        for fname in fields:
            try:
                val = getattr(effect, fname)
            except AttributeError:  # pragma: no cover - defensive
                continue
            out |= _walk_value(val, _depth + 1)
    return out


def _walk_value(val: Any, depth: int) -> set[str]:
    if isinstance(val, (tuple, list)):
        out: set[str] = set()
        for v in val:
            out |= _walk_value(v, depth)
        return out
    if hasattr(val, "__dataclass_fields__") and hasattr(val, "resolve"):
        return effect_harm_classes(val, depth)
    return set()


def has_benign_return(effect: Any, _depth: int = 0) -> bool:
    """True when the effect tree contains a class whose name marks a
    linked return / blink (``Return*`` etc.) — those make exiling an
    own permanent a plausible play, so ``self_target`` skips them."""
    if effect is None or _depth > 8:
        return False
    name = type(effect).__name__
    if any(m in name for m in _BENIGN_MARKERS):
        return True
    fields = getattr(effect, "__dataclass_fields__", None)
    if fields:
        for fname in fields:
            val = getattr(effect, fname, None)
            vals = val if isinstance(val, (tuple, list)) else (val,)
            for v in vals:
                if hasattr(v, "__dataclass_fields__") and has_benign_return(
                    v, _depth + 1,
                ):
                    return True
    return False


def _item_effect(item: Any) -> Any:
    ability = getattr(item, "ability", None)
    if ability is not None:
        return getattr(ability, "effect", None)
    return getattr(item.source.printing, "spell_effect", None)


def _item_kind(item: Any) -> str:
    ability = getattr(item, "ability", None)
    if ability is None:
        return "spell"
    return "activated" if type(ability).__name__ == "ActivatedAbility" else "triggered"


def _controller_of(eng: Any, obj_id: int) -> Optional[int]:
    for o in eng.battlefield:
        if o.id == obj_id:
            return o.controller_id
    return None


def _on_battlefield(eng: Any, obj_id: int) -> bool:
    return any(o.id == obj_id for o in eng.battlefield)


def _item_key(item: Any) -> tuple:
    return (
        id(item), item.source.id, item.controller_id,
        tuple(getattr(item, "targets", ()) or ()),
    )


@dataclass
class BlunderCensus:
    """Per-game detector state for one agent seat (see module doc)."""

    seat: int
    counts: dict[str, int] = field(
        default_factory=lambda: {d: 0 for d in DETECTORS},
    )
    events: list[dict[str, Any]] = field(default_factory=list)
    # Denominators: agent-controlled stack items that carried >= 1
    # target (all auto-targeted by the tactical policy in production),
    # and those whose effect is damage / destroy / exile.
    targeted_items: int = 0
    harmful_targeted_items: int = 0
    # Stack tracking: key -> (item, last-seen info)
    _seen_items: dict[tuple, dict[str, Any]] = field(default_factory=dict)
    # Combat tracking for the agent's attacks.
    _combat: Optional[dict[str, Any]] = None
    _missed_turns: set[int] = field(default_factory=set)

    # ------------------------------------------------------------------
    def observe_moment(self, adapter: Any) -> None:
        """Call at every decision moment of the real game (before the
        decision is chosen)."""
        eng = adapter.engine_state
        self._track_stack(eng)
        self._track_combat(eng)

    def before_decision(
        self, adapter: Any, actor: int, decision: Any,
    ) -> None:
        """Call with the chosen decision right before it is applied."""
        if actor != self.seat:
            return
        eng = adapter.engine_state
        if type(decision).__name__ != "PassPriority":
            return
        if (
            eng.step is not Step.POSTCOMBAT_MAIN
            or eng.stack
            or eng.players[eng.active_player_idx].id != self.seat
            or eng.priority_player_idx != eng.active_player_idx
        ):
            return
        if eng.turn_number in self._missed_turns:
            return
        from mtg_bot.legal import legal_actions
        creature, other = [], []
        for a in legal_actions(eng, self.seat):
            if type(a).__name__ != "CastSpell":
                continue
            card = _find_card(eng, a.card_id)
            if card is None:
                continue
            pr = card.printing
            if pr.is_instant or "flash" in pr.keywords:
                continue
            (creature if pr.is_creature else other).append(pr.name)
        if not (creature or other):
            return
        self._missed_turns.add(eng.turn_number)
        key = "missed_play_creature" if creature else "missed_play_other"
        self._hit(key, turn=eng.turn_number, castable=sorted(set(creature + other)))

    def after_decision(self, adapter: Any) -> None:
        """Call right after a decision was applied (picks up stack items
        the decision created, e.g. a cast and its auto-chosen target)."""
        eng = adapter.engine_state
        self._track_stack(eng)

    def finish(self, adapter: Any) -> None:
        eng = adapter.engine_state
        self._track_stack(eng)
        self._close_combat(eng, final=True)

    # ------------------------------------------------------------------
    def _hit(self, detector: str, **detail: Any) -> None:
        self.counts[detector] += 1
        self.events.append({"detector": detector, **detail})

    def _track_stack(self, eng: Any) -> None:
        current: dict[tuple, Any] = {}
        for item in eng.stack:
            current[_item_key(item)] = item
        # New items: self-target check at first sighting.
        for key, item in current.items():
            if key in self._seen_items:
                info = self._seen_items[key]
            else:
                info = {"item": item}
                self._seen_items[key] = info
                if item.controller_id == self.seat:
                    if getattr(item, "targets", ()):
                        self.targeted_items += 1
                        if effect_harm_classes(_item_effect(item)):
                            self.harmful_targeted_items += 1
                    self._check_self_target(eng, item)
            # Refresh last-seen facts used by the fizzle detector.
            targets = tuple(getattr(item, "targets", ()) or ())
            obj_targets = [t for t in targets if t[0] != "player"]
            info["all_obj_targets_gone"] = bool(obj_targets) and all(
                not _on_battlefield(eng, t[1]) for t in obj_targets
            ) and len(obj_targets) == len(targets)
            info["countered"] = bool(getattr(item, "countered", False))
            info["turn"] = eng.turn_number
        # Items that left the stack since the last observation resolved
        # (or were countered / removed).
        for key in list(self._seen_items):
            if key in current:
                continue
            info = self._seen_items.pop(key)
            item = info["item"]
            if (
                item.controller_id == self.seat
                and info.get("all_obj_targets_gone")
                and not info.get("countered")
            ):
                self._hit(
                    "fizzle", turn=info.get("turn"),
                    source=item.source.printing.name, kind=_item_kind(item),
                )

    def _check_self_target(self, eng: Any, item: Any) -> None:
        targets = tuple(getattr(item, "targets", ()) or ())
        if len(targets) != 1:
            return
        effect = _item_effect(item)
        harm = effect_harm_classes(effect)
        if not harm or has_benign_return(effect):
            return
        kind, tid = targets[0]
        if kind == "player":
            own = tid == self.seat and any(
                "Damage" in h for h in harm
            )
        else:
            own = _controller_of(eng, tid) == self.seat
        if not own:
            return
        pool = []
        try:
            if effect is not None:
                pool = list(effect.legal_targets(eng, self.seat))
        except Exception:  # pragma: no cover - exotic effect surfaces
            pool = []
        opp_side = any(
            (k == "player" and i != self.seat)
            or (k != "player" and _controller_of(eng, i) not in (None, self.seat))
            for k, i in pool
        )
        detector = "self_target_avoidable" if opp_side else "self_target_forced"
        target_name = "own face"
        if kind != "player":
            for o in eng.battlefield:
                if o.id == tid:
                    target_name = o.printing.name
        self._hit(
            detector, turn=eng.turn_number,
            source=item.source.printing.name, kind=_item_kind(item),
            target=target_name, effects=sorted(harm),
        )

    def _track_combat(self, eng: Any) -> None:
        active_id = eng.players[eng.active_player_idx].id
        step_idx = _STEP_INDEX.get(eng.step, -1)
        if self._combat is not None:
            if (
                eng.turn_number != self._combat["turn"]
                or step_idx > _STEP_INDEX[Step.COMBAT_DAMAGE]
                or eng.is_over()
            ):
                self._close_combat(eng)
        if active_id != self.seat or eng.step is not Step.DECLARE_BLOCKERS:
            return
        attackers = [
            o for o in eng.battlefield
            if o.attacking and o.controller_id == self.seat
        ]
        if not attackers:
            return
        from mtg_bot.game import effective_keywords
        defender = next(p for p in eng.players if p.id != self.seat)
        snap = {
            "turn": eng.turn_number,
            "defender_life": defender.life,
            "defender_id": defender.id,
            "attackers": {
                o.id: {
                    "name": o.printing.name,
                    "blocked_by": list(o.blocked_by),
                    "trample": "trample" in effective_keywords(eng, o),
                }
                for o in attackers
            },
        }
        self._combat = snap

    def _close_combat(self, eng: Any, final: bool = False) -> None:
        snap = self._combat
        self._combat = None
        if snap is None:
            return
        defender = next(
            (p for p in eng.players if p.id == snap["defender_id"]), None,
        )
        no_damage = defender is not None and defender.life >= snap["defender_life"]
        for aid, a in snap["attackers"].items():
            if not a["blocked_by"] or a["trample"]:
                continue
            if _on_battlefield(eng, aid):
                continue
            if any(not _on_battlefield(eng, b) for b in a["blocked_by"]):
                continue  # traded (or a blocker died some other way)
            self._hit(
                "chump_attack", turn=snap["turn"], attacker=a["name"],
                n_blockers=len(a["blocked_by"]),
            )
            if no_damage:
                self._hit(
                    "chump_attack_no_other_damage", turn=snap["turn"],
                    attacker=a["name"],
                )


def _find_card(eng: Any, card_id: int) -> Optional[Any]:
    for p in eng.players:
        for zone in (p.hand, p.exile, p.graveyard):
            for o in zone:
                if o.id == card_id:
                    return o
        if p.library and p.library[0].id == card_id:
            return p.library[0]
    return None
