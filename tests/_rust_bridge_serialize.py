"""Python -> Rust-bridge dict serializer (test helper).

Converts a Python :class:`mtg_bot.cards.CardPrinting` into the dict
shape that ``mtg_bot_engine.Registry.register_printing`` expects.
This is a *test* helper that lives under ``tests/`` so it stays
outside the ``mtg_bot`` coverage scope and outside the production
import graph — its sole job is to prove the Rust bridge can ingest
something the existing Python parser actually produces, not just
hand-crafted dict literals.

Vocabulary supported by this serializer mirrors the closed-
vocabulary first slice of the Rust bridge at commit ``bf9c74e``:

- One ``Effect`` variant: ``DealDamage`` (and its sugar subclasses
  whose engine semantics are pure ``DealDamage``: today
  ``DealDamageToTargetCreature``, ``DealDamageToTargetOpponent``,
  ``DealDamageAnyTarget``).
- Two ``TargetFilter`` variants: ``IsCreature()`` -> ``"any_creature"``
  and ``IsPlayer()`` -> ``"any_player"``.
- ``ManaCost`` with five colors + generic, no hybrid / X.

Anything outside that vocabulary raises :class:`NotImplementedError`
with a precise reason, so adding a new Rust variant in a follow-up
commit lights up here as a clear next-card-to-port.
"""

from __future__ import annotations

from typing import Any

from mtg_bot.cards.base import CardPrinting, ManaCost
from mtg_bot.effects.generic.damage import DealDamage
from mtg_bot.effects.object_predicates import IsCreature, IsPlayer
from mtg_bot.effects.targets import TargetSpec


_COLOR_KEYS = {"W", "U", "B", "R", "G"}


def serialize_mana_cost(cost: ManaCost) -> dict[str, int]:
    """Return the dict shape Rust ``parse_mana_cost`` accepts.

    Hybrid / X costs raise — the Rust bridge has no slot for them
    yet, and silently dropping them would mismatch the printed cost.
    Zero-count entries are omitted to match the Rust round-trip
    (which writes only positive counts).
    """
    if cost.x_pips:
        raise NotImplementedError(
            f"X-cost spells not in Rust vocabulary (x_pips={cost.x_pips})"
        )
    if cost.hybrid:
        raise NotImplementedError(
            f"hybrid mana not in Rust vocabulary (hybrid={cost.hybrid!r})"
        )
    out: dict[str, int] = {}
    if cost.generic:
        out["generic"] = cost.generic
    for color, n in cost.colored:
        if color not in _COLOR_KEYS:
            raise NotImplementedError(
                f"unknown mana color {color!r}; Rust vocabulary is W/U/B/R/G"
            )
        if n:
            out[color] = n
    return out


def serialize_target_filter(spec: TargetSpec) -> dict[str, str]:
    """Map a ``TargetSpec.filter`` to the Rust ``target`` dict.

    Only the two atom filters the Rust side knows survive; ``OneOf``
    (any-target expansion), composed predicates, and every other
    filter atom raise ``NotImplementedError``.
    """
    if spec.count_min != 1 or spec.count_max != 1:
        raise NotImplementedError(
            f"target count {spec.count_min}..{spec.count_max} not supported "
            f"(Rust expects exactly 1)"
        )
    if spec.distinct_from:
        raise NotImplementedError(
            f"distinct_from constraint not supported: {spec.distinct_from!r}"
        )
    f = spec.filter
    if isinstance(f, IsCreature):
        return {"kind": "any_creature"}
    if isinstance(f, IsPlayer):
        return {"kind": "any_player"}
    raise NotImplementedError(
        f"target filter {type(f).__name__} not in Rust vocabulary "
        f"(supported: IsCreature, IsPlayer)"
    )


def serialize_effect(effect: Any) -> dict[str, Any]:
    """Map one Python effect to a Rust-bridge effect dict.

    Recognized: anything assignable to ``DealDamage`` whose
    ``targets`` is a single ``TargetSpec``. The sugar subclasses
    (``DealDamageToTargetCreature`` / ``...AnyTarget`` /
    ``...ToTargetOpponent``) all inherit from ``DealDamage`` and
    expose the same ``amount`` + ``targets`` fields, so this single
    isinstance branch covers them.
    """
    if isinstance(effect, DealDamage):
        if len(effect.targets) != 1:
            raise NotImplementedError(
                f"DealDamage with {len(effect.targets)} target slots not "
                f"supported (Rust expects exactly 1)"
            )
        slot = effect.targets[0]
        if not isinstance(slot, TargetSpec):
            raise NotImplementedError(
                f"target slot type {type(slot).__name__} not supported "
                f"(SlotRef back-references would need cross-effect plumbing)"
            )
        return {
            "kind": "deal_damage",
            "amount": int(effect.amount),
            "target": serialize_target_filter(slot),
        }
    raise NotImplementedError(
        f"effect type {type(effect).__name__} not in Rust vocabulary "
        f"(supported: DealDamage and its sugar subclasses)"
    )


def serialize_printing(printing: CardPrinting) -> dict[str, Any]:
    """Convert a ``CardPrinting`` to the Rust ``register_printing`` dict.

    Only the spell-effect path is wired up today — triggered
    abilities, activated abilities, static abilities, and every
    non-spell-effect printing field are ignored because the Rust
    side has no slot for them. This is correct for the first-slice
    vocabulary (one ``DealDamage`` variant); a richer Rust schema
    will need this serializer to grow per added field.

    Round-3 extension: when ``printing.power`` / ``printing.toughness``
    are set (creature printings), pass them through as integers.
    The Python ``CardPrinting`` doesn't carry a ``mana_produced``
    field — mana production is encoded as activated abilities and
    is out of vocabulary for this slice — so the bridge field is
    omitted when serializing real cards. Synthetic registry seeds
    in ``state.rs`` set it directly.
    """
    out: dict[str, Any] = {
        "name": printing.name,
        "mana_cost": serialize_mana_cost(printing.mana_cost),
    }
    if printing.spell_effect is not None:
        out["effects"] = [serialize_effect(printing.spell_effect)]
    if printing.power is not None:
        out["power"] = int(printing.power)
    if printing.toughness is not None:
        out["toughness"] = int(printing.toughness)
    return out
