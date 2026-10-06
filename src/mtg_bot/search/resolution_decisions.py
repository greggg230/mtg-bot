"""Search-side decision types for mid-resolution callbacks.

When a spell or activated ability resolves, the engine asks the
controlling (or targeted) player to make tactical choices:

- Which target to pick for a burn spell's damage.
- Which cards to discard when hit by Mind Rot.
- Which mode of a modal spell to take.
- Which creature to sacrifice to a cost or effect.

By default, ``MtgGameStateAdapter`` routes these to
``DefaultTacticalAgent`` (first-legal, deterministic). When the
adapter is constructed with ``expose_resolution_decisions=True``,
these four choice points become search decisions instead, and MCTS
can explore them the same way it explores priority actions.

Mechanism: during ``apply_action`` / ``resolve_top_of_stack``, the
adapter's ``ResolutionChannelAgent`` raises a ``_ResolutionPause``
when it hits one of these callbacks. The adapter catches it,
rollbacks to a pre-action snapshot (cloned before the priority-action
was attempted), and exposes the pending resolution choice via
``legal_decisions()``. When the caller applies a
``ResolutionDecision``, the adapter stashes it in a replay queue and
retries the original priority action — this time the channel agent
pops the answer from the queue and the effect resolves with the
search-chosen value. A second pause-point during the same resolution
adds another queue entry; the third replay reaches farther; etc.

Covers all decision-making callbacks on ``Agent``:

- ``ResolutionTargetDecision``             (← ``agent.choose_target``)
- ``ResolutionDiscardDecision``            (← ``agent.choose_discard``)
- ``ResolutionModeDecision``               (← ``agent.choose_mode``)
- ``ResolutionModesDecision``              (← ``agent.choose_modes``)
- ``ResolutionSacrificeDecision``          (← ``agent.choose_sacrifice``)
- ``ResolutionXDecision``                  (← ``agent.choose_x``)
- ``ResolutionSurveilDecision``            (← ``agent.choose_surveil``)
- ``ResolutionShockLifeDecision``
                                   (← ``agent.choose_pay_shock_life``)
- ``ResolutionCardFromLibraryDecision``
                               (← ``agent.choose_card_from_library``)
- ``ResolutionCardFromOpponentHandDecision``
                         (← ``agent.choose_card_from_opponent_hand``)
- ``ResolutionDevourLandsDecision``      (← ``agent.choose_devour_lands``)
- ``ResolutionYesNoDecision``              (← ``agent.choose_yes_no``)

All decision types are frozen dataclasses so they're hashable (the
search protocol's ``Decision`` contract) and cheaply comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResolutionTargetDecision:
    """Answer to an ``agent.choose_target`` callback during
    resolution (or at cast time — the engine uses the same callback
    for both). ``target`` is a ``(kind, id)`` tuple matching one of
    the options the engine passed to ``legal_targets``, or ``None``
    for "decline to target" (some effects permit this)."""

    target: Optional[tuple[str, int]]


@dataclass(frozen=True)
class ResolutionDiscardDecision:
    """Answer to an ``agent.choose_discard`` callback. Picks one of
    the caller's own hand card ids. Each call corresponds to one
    card; effects that discard multiple cards raise one pause per
    card."""

    card_id: int


@dataclass(frozen=True)
class ResolutionModeDecision:
    """Answer to an ``agent.choose_mode`` callback for modal
    "choose one" spells/abilities. ``mode_index`` is the index into
    the ``mode_descriptions`` list the engine supplied."""

    mode_index: int


@dataclass(frozen=True)
class ResolutionModesDecision:
    """Answer to an ``agent.choose_modes`` callback ("choose one or
    both" / "choose one or more"). ``mode_indices`` is a non-empty
    tuple of mode indices in printed (ascending) order -- the order
    the modes resolve in. The adapter exposes every non-empty subset
    of the modes."""

    mode_indices: tuple[int, ...]


@dataclass(frozen=True)
class ResolutionSacrificeDecision:
    """Answer to an ``agent.choose_sacrifice`` callback. Picks which
    battlefield object id to sacrifice from the legal-ids list the
    engine supplied."""

    object_id: int


@dataclass(frozen=True)
class ResolutionXDecision:
    """Answer to an ``agent.choose_x`` callback for X-cost spells
    and abilities. ``x_value`` is an int in ``[0, upper_bound]``
    where ``upper_bound`` is the caller's affordable maximum at
    cast time."""

    x_value: int


@dataclass(frozen=True)
class ResolutionSurveilDecision:
    """Answer to an ``agent.choose_surveil`` callback.
    ``mill=True`` sends the top library card to the graveyard;
    ``mill=False`` leaves it on top."""

    mill: bool


@dataclass(frozen=True)
class ResolutionShockLifeDecision:
    """Answer to an ``agent.choose_pay_shock_life`` callback when
    playing a shock land. ``pay=True`` pays the life cost and the
    land enters untapped; ``pay=False`` enters tapped."""

    pay: bool


@dataclass(frozen=True)
class ResolutionCardFromLibraryDecision:
    """Answer to an ``agent.choose_card_from_library`` callback for
    tutor-style effects. ``card_id`` is the chosen library card, or
    ``None`` for "decline to fetch" (where the effect allows it)."""

    card_id: Optional[int]


@dataclass(frozen=True)
class ResolutionCardFromOpponentHandDecision:
    """Answer to an ``agent.choose_card_from_opponent_hand`` callback
    for directed-discard effects (Thoughtseize-style) where the
    CASTER chooses which card from the opponent's revealed hand."""

    card_id: int


@dataclass(frozen=True)
class ResolutionYesNoDecision:
    """Answer to an ``agent.choose_yes_no`` callback for "may"
    abilities — the controller decides whether to take the optional
    action. ``yes=True`` means take it; ``yes=False`` declines."""

    yes: bool


@dataclass(frozen=True)
class ResolutionDevourLandsDecision:
    """Answer to an ``agent.choose_devour_lands`` callback (Devour
    mechanic). ``land_ids`` is the subset of eligible lands to
    sacrifice — empty tuple for "devour none".

    V1 simplification: the adapter only surfaces two options — the
    empty set and the full set — rather than every subset of
    eligible lands. Subset enumeration blows up quickly (2^N) and
    the strategic question is almost always "how many?" rather
    than "which specific lands?" in practice. A prior/factored-
    decision refinement can expose intermediate counts later.
    """

    land_ids: tuple[int, ...]
