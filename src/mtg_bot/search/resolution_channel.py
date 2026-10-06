"""Agent used by ``MtgGameStateAdapter`` when resolution decisions
are exposed to search.

Replaces ``DefaultTacticalAgent`` as a player's agent when that seat's
resolution decisions are exposed. Has a per-resolution **replay
queue** of ``ResolutionDecision`` values supplied by search. Each
time the engine calls one of the exposed agent-callback methods
during ``apply_action``:

- If the queue has an answer at the current position -> pop it, return
  the engine-native value, resolution proceeds.
- If the queue is exhausted -> raise ``_ResolutionPause``, propagating
  up to the adapter. The adapter catches it, exposes the decision
  to search, and on ``apply(answer)`` replays the whole action from a
  pre-action clone with the queue extended by one.

At pause time the agent also (from the LIVE state the engine passed
in -- the adapter is about to roll it back):

- normalises the options (interchangeable hand / library cards are
  offered once; exact duplicate targets are dropped -- see
  ``mtg_bot.search.resolution_slots``),
- records which hidden cards the callback revealed to the decider
  (tutor matches, an opponent's hand, the surveilled top card) so
  PIMC / ISMCTS determinizations keep them fixed,
- computes each option's policy slot (``resolution_slots.
  compute_offsets``) for the ``canonical_v3`` action encoding.

Callbacks NOT exposed to search (cost-payment and SBA choices:
``choose_permanent_to_tap``, convoke, sac-for-mana, mana colour,
legend rule, replacement order, ``choose_card_from_hand``, mulligans)
delegate to ``fallback`` (a regular ``DefaultTacticalAgent``)
explicitly -- never to the ``Agent`` base-class defaults.

Why a replay queue: this lets the adapter treat the priority action
as a black box -- the engine's resolution code is untouched. A
different implementation (generator-based effect resolution) would
avoid the replays but require changing every ``effect.resolve()``
across the codebase. The queue trades compute for simplicity.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Optional

from mtg_bot.agents import Agent
from mtg_bot.search import resolution_slots
from mtg_bot.search.resolution_decisions import (
    ResolutionCardFromLibraryDecision,
    ResolutionCardFromOpponentHandDecision,
    ResolutionDevourLandsDecision, ResolutionDiscardDecision,
    ResolutionModeDecision, ResolutionModesDecision,
    ResolutionSacrificeDecision,
    ResolutionShockLifeDecision, ResolutionSurveilDecision,
    ResolutionTargetDecision, ResolutionXDecision,
    ResolutionYesNoDecision,
)
from mtg_bot.search.tactical import DefaultTacticalAgent


# Exposed callback kinds -- used as pause sentinels and as dispatch
# keys when decoding queue entries.
KIND_TARGET = "target"
KIND_DISCARD = "discard"
KIND_MODE = "mode"
KIND_MODES = "modes"
KIND_SACRIFICE = "sacrifice"
KIND_X = "x"
KIND_SURVEIL = "surveil"
KIND_SHOCK_LIFE = "shock_life"
KIND_CARD_FROM_LIBRARY = "card_from_library"
KIND_CARD_FROM_OPPONENT_HAND = "card_from_opponent_hand"
KIND_DEVOUR_LANDS = "devour_lands"
KIND_YES_NO = "yes_no"

# "choose one or more" subsets are enumerated exhaustively up to this
# many modes (63 options); beyond it only singletons + all-modes are
# offered. Every EoE "or both" spell has two modes.
_MAX_MODES_ENUMERATED = 6


def mode_subsets(n_modes: int) -> tuple[tuple[int, ...], ...]:
    """Every non-empty subset of ``range(n_modes)`` in ascending order,
    smaller subsets first (``(0,), (1,), (0, 1)`` for two modes)."""
    if n_modes <= 0:
        return ((0,),)
    if n_modes > _MAX_MODES_ENUMERATED:
        singles = tuple((i,) for i in range(n_modes))
        return singles + (tuple(range(n_modes)),)
    out: list[tuple[int, ...]] = []
    for size in range(1, n_modes + 1):
        out.extend(combinations(range(n_modes), size))
    return tuple(out)


class _ResolutionPause(Exception):
    """Raised by ``ResolutionChannelAgent`` when the engine calls an
    exposed callback and the replay queue has no answer for it.

    The adapter catches this one exception type, inspects ``kind`` /
    ``options`` / ``slot_offsets`` / ``revealed_ids``, exposes the
    decision to search, and resumes the action once the caller
    supplies an answer.
    """

    def __init__(
        self, kind: str, player_id: int, options: tuple,
        slot_offsets: Optional[tuple] = None,
        revealed_ids: tuple = (),
    ) -> None:
        super().__init__(
            f"resolution pause: {kind} for P{player_id} "
            f"with {len(options)} option(s)"
        )
        self.kind = kind
        self.player_id = player_id
        self.options = options
        self.slot_offsets = (
            tuple(slot_offsets) if slot_offsets is not None
            else (None,) * len(options)
        )
        self.revealed_ids = tuple(revealed_ids)


class ResolutionChannelAgent(Agent):
    """Channel between the engine's agent-callback interface and the
    search framework.

    Per-resolution state (the replay queue + cursor) is owned by
    this agent and mutated in place as the adapter runs and replays
    actions. Callers (i.e., the adapter) reset the queue before each
    new action and extend it on each pause.

    Non-exposed callbacks delegate to a ``DefaultTacticalAgent``
    (or whatever fallback is supplied).
    """

    name: str = "resolution_channel"

    def __init__(self, fallback: Agent | None = None) -> None:
        self._fallback = fallback or DefaultTacticalAgent()
        self._queue: list = []
        self._queue_idx: int = 0

    # -----------------------------------------------------------------
    # Public API used by the adapter
    # -----------------------------------------------------------------

    def reset_queue(self) -> None:
        """Start a fresh action. Called by the adapter before
        attempting each new priority / combat action."""
        self._queue = []
        self._queue_idx = 0

    def rewind(self) -> None:
        """Before a replay, move the cursor back to the beginning.
        The queue's contents are preserved -- the callback sequence
        will pop from position 0 again, replaying the saved answers
        in order."""
        self._queue_idx = 0

    def push_answer(self, decision) -> None:
        """Append a ``ResolutionDecision`` to the replay queue.
        Called by the adapter after search answers a pause."""
        self._queue.append(decision)

    @property
    def queue(self) -> tuple:
        """The answers given so far in the current action (read-only
        view; part of the adapter's canonical / info-set keys)."""
        return tuple(self._queue)

    def snapshot(self) -> dict:
        """Return a snapshot of the queue state for cloning.
        Adapter uses this to copy queue state to a sibling agent
        instance that will be installed on a cloned engine state."""
        return {
            "queue": list(self._queue),
            "queue_idx": self._queue_idx,
        }

    def restore(self, snapshot: dict) -> None:
        """Restore queue state from a snapshot produced by
        ``snapshot()``."""
        self._queue = list(snapshot["queue"])
        self._queue_idx = snapshot["queue_idx"]

    # -----------------------------------------------------------------
    # Agent-protocol overrides for EXPOSED callbacks
    # -----------------------------------------------------------------

    def choose_target(
        self, state, player_id, legal_targets,
    ):
        answer = self._next_queue_entry(
            expect=ResolutionTargetDecision,
            kind=KIND_TARGET,
            player_id=player_id,
            state=state,
            options=resolution_slots.dedupe_exact(legal_targets),
        )
        return answer.target

    def choose_discard(self, state, player_id, hand_card_ids):
        answer = self._next_queue_entry(
            expect=ResolutionDiscardDecision,
            kind=KIND_DISCARD,
            player_id=player_id,
            state=state,
            options=resolution_slots.dedupe_cards(state, hand_card_ids),
        )
        return answer.card_id

    def choose_mode(self, state, player_id, mode_descriptions):
        answer = self._next_queue_entry(
            expect=ResolutionModeDecision,
            kind=KIND_MODE,
            player_id=player_id,
            state=state,
            options=tuple(range(len(mode_descriptions))),
        )
        return answer.mode_index

    def choose_modes(self, state, player_id, mode_descriptions):
        answer = self._next_queue_entry(
            expect=ResolutionModesDecision,
            kind=KIND_MODES,
            player_id=player_id,
            state=state,
            options=mode_subsets(len(mode_descriptions)),
        )
        return list(answer.mode_indices)

    def choose_sacrifice(self, state, player_id, legal_object_ids):
        answer = self._next_queue_entry(
            expect=ResolutionSacrificeDecision,
            kind=KIND_SACRIFICE,
            player_id=player_id,
            state=state,
            options=resolution_slots.dedupe_exact(legal_object_ids),
        )
        return answer.object_id

    def choose_x(self, state, player_id, card_name, upper_bound):
        # Options are ``0`` through ``upper_bound`` inclusive.
        answer = self._next_queue_entry(
            expect=ResolutionXDecision,
            kind=KIND_X,
            player_id=player_id,
            state=state,
            options=tuple(range(upper_bound + 1)),
        )
        return answer.x_value

    def choose_surveil(self, state, player_id, top_card_id):
        # The decider has looked at the top card: it is revealed to
        # them for the determinizer.
        answer = self._next_queue_entry(
            expect=ResolutionSurveilDecision,
            kind=KIND_SURVEIL,
            player_id=player_id,
            state=state,
            options=(True, False),
            revealed_ids=(top_card_id,),
        )
        return answer.mill

    def choose_pay_shock_life(self, state, player_id, life_cost):
        answer = self._next_queue_entry(
            expect=ResolutionShockLifeDecision,
            kind=KIND_SHOCK_LIFE,
            player_id=player_id,
            state=state,
            options=(True, False),
        )
        return answer.pay

    def choose_card_from_library(self, state, player_id, option_ids):
        # ``None`` is a legal answer for "decline to fetch" -- thread
        # it through as the final option so search sees decline as
        # a real alternative. Every matching card was revealed to the
        # searcher.
        options = resolution_slots.dedupe_cards(
            state, tuple(option_ids),
        ) + (None,)
        answer = self._next_queue_entry(
            expect=ResolutionCardFromLibraryDecision,
            kind=KIND_CARD_FROM_LIBRARY,
            player_id=player_id,
            state=state,
            options=options,
            revealed_ids=tuple(option_ids),
        )
        return answer.card_id

    def choose_card_from_opponent_hand(
        self, state, player_id, opponent_hand_ids,
    ):
        answer = self._next_queue_entry(
            expect=ResolutionCardFromOpponentHandDecision,
            kind=KIND_CARD_FROM_OPPONENT_HAND,
            player_id=player_id,
            state=state,
            options=resolution_slots.dedupe_cards(
                state, opponent_hand_ids,
            ),
            revealed_ids=tuple(opponent_hand_ids),
        )
        return answer.card_id

    def choose_devour_lands(self, state, player_id, land_ids):
        # V1: expose just ``()`` (devour none) and
        # ``tuple(land_ids)`` (devour all). Intermediate subset
        # counts and per-land factoring are future refinements.
        options = ((), tuple(land_ids)) if land_ids else ((),)
        answer = self._next_queue_entry(
            expect=ResolutionDevourLandsDecision,
            kind=KIND_DEVOUR_LANDS,
            player_id=player_id,
            state=state,
            options=options,
        )
        return list(answer.land_ids)

    def choose_yes_no(self, state, player_id, prompt):
        answer = self._next_queue_entry(
            expect=ResolutionYesNoDecision,
            kind=KIND_YES_NO,
            player_id=player_id,
            state=state,
            options=(True, False),
        )
        return answer.yes

    # -----------------------------------------------------------------
    # Non-exposed callbacks -> tactical fallback (explicitly; the
    # ``Agent`` base class raises or returns its own defaults)
    # -----------------------------------------------------------------

    def choose_action(self, state, player_id, legal):
        # Should not be called during resolution. Delegate if it is
        # -- keeps us safe against unusual call paths.
        return self._fallback.choose_action(state, player_id, legal)

    def declare_attackers(self, state, player_id, eligible):
        return self._fallback.declare_attackers(state, player_id, eligible)

    def declare_blockers(self, state, player_id, eligible, attackers):
        return self._fallback.declare_blockers(
            state, player_id, eligible, attackers,
        )

    def choose_permanent_to_tap(self, state, player_id, legal_object_ids):
        return self._fallback.choose_permanent_to_tap(
            state, player_id, legal_object_ids,
        )

    def choose_card_from_hand(self, state, player_id, option_ids):
        return self._fallback.choose_card_from_hand(
            state, player_id, option_ids,
        )

    def choose_mana_color(self, state, player_id, options):
        return self._fallback.choose_mana_color(state, player_id, options)

    def choose_convoke_creatures(
        self, state, player_id, candidates, remaining_generic,
        remaining_colored,
    ):
        return self._fallback.choose_convoke_creatures(
            state, player_id, candidates, remaining_generic,
            remaining_colored,
        )

    def choose_sac_for_mana(
        self, state, player_id, candidates, remaining_generic,
        remaining_colored,
    ):
        return self._fallback.choose_sac_for_mana(
            state, player_id, candidates, remaining_generic,
            remaining_colored,
        )

    def choose_legend_keep(self, state, player_id, legend_object_ids):
        return self._fallback.choose_legend_keep(
            state, player_id, legend_object_ids,
        )

    def choose_mulligan(self, state, player_id, hand, mulls_taken):
        return self._fallback.choose_mulligan(
            state, player_id, hand, mulls_taken,
        )

    def choose_bottom(self, state, player_id, hand, n):
        return self._fallback.choose_bottom(state, player_id, hand, n)

    def choose_replacement_order(
        self, candidates, state, affected_player_id,
    ):
        return self._fallback.choose_replacement_order(
            candidates, state, affected_player_id,
        )

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _next_queue_entry(
        self, *,
        expect: type,
        kind: str,
        player_id: int,
        state: Any,
        options: tuple,
        revealed_ids: tuple = (),
    ):
        """Pop the next queue entry (for replay) OR raise
        ``_ResolutionPause`` if the queue is exhausted.

        Validates that the queue entry matches the callback kind --
        a mismatch means the adapter enqueued something inconsistent
        with what the engine is asking for, which is a bug."""
        if self._queue_idx < len(self._queue):
            entry = self._queue[self._queue_idx]
            self._queue_idx += 1
            if not isinstance(entry, expect):
                raise TypeError(
                    f"Resolution queue entry at index "
                    f"{self._queue_idx - 1} expected to be "
                    f"{expect.__name__} for a {kind} callback; got "
                    f"{type(entry).__name__}"
                )
            return entry
        # Queue exhausted -- pause resolution. Slots are computed from
        # the live state now; the adapter is about to roll it back.
        raise _ResolutionPause(
            kind=kind, player_id=player_id, options=options,
            slot_offsets=resolution_slots.compute_offsets(
                kind, state, player_id, options,
            ),
            revealed_ids=self._known_at_pause(
                state, player_id, kind, options, revealed_ids,
            ),
        )

    def _known_at_pause(
        self, state: Any, player_id: int, kind: str, options: tuple,
        revealed_ids: tuple,
    ) -> tuple:
        """Card ids the decider can identify at this callback that may
        be HIDDEN in the pre-action snapshot the adapter rolls back to:
        the callback's explicit reveals, every card option, the
        decider's live hand (cards drawn earlier in this action are
        still library cards in the snapshot -- Alpharael's "draw two,
        then discard two unless you discard an artifact"), and the
        cards earlier answers of this action named. PIMC / ISMCTS pin
        them so a determinized world replays to the same pause
        (``determinize.determinized_clone``)."""
        ids: list = list(revealed_ids)
        if kind in (
            KIND_DISCARD, KIND_CARD_FROM_LIBRARY,
            KIND_CARD_FROM_OPPONENT_HAND, KIND_SACRIFICE,
        ):
            ids.extend(o for o in options if isinstance(o, int))
        try:
            ids.extend(c.id for c in state.player_by_id(player_id).hand)
        except (AttributeError, KeyError, StopIteration):
            pass
        for answer in self._queue:
            cid = getattr(answer, "card_id", None)
            if isinstance(cid, int):
                ids.append(cid)
        return tuple(dict.fromkeys(ids))
