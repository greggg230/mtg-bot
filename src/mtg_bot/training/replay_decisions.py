"""JSON serialization for ``Decision`` instances stored in replays.

A captured replay's per-priority-moment record carries a small JSON
shape describing the action that was taken at that moment. Phase 6's
particle-filter likelihood needs the exact action to look its slot
probability up in the policy distribution; reconstructing the action
from a state-diff is fragile (mana abilities leave no zone change,
declared-attackers sets only show the union, pass-priority cascades
mix multiple engine steps). Storing the full ``Decision`` shape
guarantees exact reconstruction.

Two functions: ``decision_to_json`` and ``decision_from_json``. They
cover every concrete subclass of ``Decision`` the engine adapter
emits — priority actions from ``mtg_bot.actions``, search-side
combat sub-decisions from ``search.combat_decisions``, and the
resolution-pause decisions from ``search.resolution_decisions``.

Design choices:

- ``"kind"`` is the class name (string). Tagged-union style — no
  inheritance assumed across ``Decision`` subclasses since they live
  in three different modules. The kind dispatches the
  reconstruction; field names match the dataclass's attributes.
- Tuples are serialized as lists (JSON has no tuple type).
  ``DeclareBlockers.assignments`` round-trips via
  ``tuple(tuple(pair) for pair in ...)``.
- ``DeclareAttackers.attacker_ids`` is a frozenset; serialized as a
  sorted list for determinism (so the same decision always produces
  the same JSON), reconstructed as a frozenset.
- ``ResolutionTargetDecision.target`` is ``Optional[tuple[str, int]]``;
  the ``None`` case stores ``null``, the tuple stores as a 2-element
  list.
"""

from __future__ import annotations

from typing import Any

from mtg_bot.actions import (
    ActivateAbility,
    ActivateStation,
    CastSpell,
    DeclareAttackers,
    DeclareBlockers,
    PassPriority,
    PlayLand,
    TurnFaceUp,
    UnlockRoomHalf,
)
from mtg_bot.search.combat_decisions import AttackerDecision, BlockerDecision
from mtg_bot.search.resolution_decisions import (
    ResolutionCardFromLibraryDecision,
    ResolutionCardFromOpponentHandDecision,
    ResolutionDevourLandsDecision,
    ResolutionDiscardDecision,
    ResolutionModeDecision,
    ResolutionModesDecision,
    ResolutionSacrificeDecision,
    ResolutionShockLifeDecision,
    ResolutionSurveilDecision,
    ResolutionTargetDecision,
    ResolutionXDecision,
    ResolutionYesNoDecision,
)


def decision_to_json(d: Any) -> dict[str, Any]:
    """Serialize a ``Decision`` to a JSON-safe dict."""
    if isinstance(d, PassPriority):
        return {"kind": "PassPriority"}
    if isinstance(d, PlayLand):
        return {"kind": "PlayLand", "card_id": d.card_id}
    if isinstance(d, CastSpell):
        return {
            "kind": "CastSpell",
            "card_id": d.card_id,
            "use_warp": d.use_warp,
            "kicked": d.kicked,
            "cast_for_free": d.cast_for_free,
            "use_impending": d.use_impending,
            "cast_room_half": d.cast_room_half,
        }
    if isinstance(d, UnlockRoomHalf):
        return {
            "kind": "UnlockRoomHalf",
            "card_id": d.card_id,
            "half_name": d.half_name,
        }
    if isinstance(d, ActivateAbility):
        return {
            "kind": "ActivateAbility",
            "source_id": d.source_id,
            "ability_index": d.ability_index,
        }
    if isinstance(d, ActivateStation):
        return {
            "kind": "ActivateStation",
            "spacecraft_id": d.spacecraft_id,
            "crew_id": d.crew_id,
        }
    if isinstance(d, TurnFaceUp):
        return {"kind": "TurnFaceUp", "card_id": d.card_id}
    if isinstance(d, DeclareAttackers):
        return {
            "kind": "DeclareAttackers",
            "attacker_ids": sorted(d.attacker_ids),
        }
    if isinstance(d, DeclareBlockers):
        return {
            "kind": "DeclareBlockers",
            "assignments": [list(pair) for pair in d.assignments],
        }
    if isinstance(d, AttackerDecision):
        return {
            "kind": "AttackerDecision",
            "attacker_id": d.attacker_id,
            "attack": d.attack,
        }
    if isinstance(d, BlockerDecision):
        return {
            "kind": "BlockerDecision",
            "blocker_id": d.blocker_id,
            "attacker_id": d.attacker_id,
        }
    if isinstance(d, ResolutionTargetDecision):
        target = d.target
        return {
            "kind": "ResolutionTargetDecision",
            "target": None if target is None else [target[0], target[1]],
        }
    if isinstance(d, ResolutionDiscardDecision):
        return {"kind": "ResolutionDiscardDecision", "card_id": d.card_id}
    if isinstance(d, ResolutionModeDecision):
        return {"kind": "ResolutionModeDecision", "mode_index": d.mode_index}
    if isinstance(d, ResolutionModesDecision):
        return {
            "kind": "ResolutionModesDecision",
            "mode_indices": list(d.mode_indices),
        }
    if isinstance(d, ResolutionSacrificeDecision):
        return {"kind": "ResolutionSacrificeDecision", "object_id": d.object_id}
    if isinstance(d, ResolutionXDecision):
        return {"kind": "ResolutionXDecision", "x_value": d.x_value}
    if isinstance(d, ResolutionSurveilDecision):
        return {"kind": "ResolutionSurveilDecision", "mill": d.mill}
    if isinstance(d, ResolutionShockLifeDecision):
        return {"kind": "ResolutionShockLifeDecision", "pay": d.pay}
    if isinstance(d, ResolutionCardFromLibraryDecision):
        return {
            "kind": "ResolutionCardFromLibraryDecision",
            "card_id": d.card_id,
        }
    if isinstance(d, ResolutionCardFromOpponentHandDecision):
        return {
            "kind": "ResolutionCardFromOpponentHandDecision",
            "card_id": d.card_id,
        }
    if isinstance(d, ResolutionYesNoDecision):
        return {"kind": "ResolutionYesNoDecision", "yes": d.yes}
    if isinstance(d, ResolutionDevourLandsDecision):
        return {
            "kind": "ResolutionDevourLandsDecision",
            "land_ids": list(d.land_ids),
        }
    raise TypeError(
        f"decision_to_json: unsupported decision type "
        f"{type(d).__name__}: {d!r}"
    )


def decision_from_json(d: dict[str, Any]) -> Any:
    """Reverse of ``decision_to_json``. Raises ``ValueError`` on
    unknown kinds (forward-compat: a future writer might emit a kind
    a current reader doesn't recognize)."""
    kind = d.get("kind")
    if kind == "PassPriority":
        return PassPriority()
    if kind == "PlayLand":
        return PlayLand(card_id=d["card_id"])
    if kind == "CastSpell":
        return CastSpell(
            card_id=d["card_id"],
            use_warp=d["use_warp"],
            kicked=d["kicked"],
            cast_for_free=d["cast_for_free"],
            use_impending=d["use_impending"],
            cast_room_half=d["cast_room_half"],
        )
    if kind == "UnlockRoomHalf":
        return UnlockRoomHalf(card_id=d["card_id"], half_name=d["half_name"])
    if kind == "ActivateAbility":
        return ActivateAbility(
            source_id=d["source_id"], ability_index=d["ability_index"],
        )
    if kind == "ActivateStation":
        return ActivateStation(
            spacecraft_id=d["spacecraft_id"], crew_id=d["crew_id"],
        )
    if kind == "TurnFaceUp":
        return TurnFaceUp(card_id=d["card_id"])
    if kind == "DeclareAttackers":
        return DeclareAttackers(
            attacker_ids=frozenset(d["attacker_ids"]),
        )
    if kind == "DeclareBlockers":
        return DeclareBlockers(
            assignments=tuple(
                (int(pair[0]), int(pair[1])) for pair in d["assignments"]
            ),
        )
    if kind == "AttackerDecision":
        return AttackerDecision(
            attacker_id=d["attacker_id"], attack=d["attack"],
        )
    if kind == "BlockerDecision":
        return BlockerDecision(
            blocker_id=d["blocker_id"], attacker_id=d["attacker_id"],
        )
    if kind == "ResolutionTargetDecision":
        target = d["target"]
        return ResolutionTargetDecision(
            target=None if target is None else (target[0], int(target[1])),
        )
    if kind == "ResolutionDiscardDecision":
        return ResolutionDiscardDecision(card_id=d["card_id"])
    if kind == "ResolutionModeDecision":
        return ResolutionModeDecision(mode_index=d["mode_index"])
    if kind == "ResolutionModesDecision":
        return ResolutionModesDecision(
            mode_indices=tuple(int(i) for i in d["mode_indices"]),
        )
    if kind == "ResolutionSacrificeDecision":
        return ResolutionSacrificeDecision(object_id=d["object_id"])
    if kind == "ResolutionXDecision":
        return ResolutionXDecision(x_value=d["x_value"])
    if kind == "ResolutionSurveilDecision":
        return ResolutionSurveilDecision(mill=d["mill"])
    if kind == "ResolutionShockLifeDecision":
        return ResolutionShockLifeDecision(pay=d["pay"])
    if kind == "ResolutionCardFromLibraryDecision":
        return ResolutionCardFromLibraryDecision(card_id=d["card_id"])
    if kind == "ResolutionCardFromOpponentHandDecision":
        return ResolutionCardFromOpponentHandDecision(card_id=d["card_id"])
    if kind == "ResolutionYesNoDecision":
        return ResolutionYesNoDecision(yes=d["yes"])
    if kind == "ResolutionDevourLandsDecision":
        return ResolutionDevourLandsDecision(
            land_ids=tuple(d["land_ids"]),
        )
    raise ValueError(f"decision_from_json: unknown kind {kind!r}")
