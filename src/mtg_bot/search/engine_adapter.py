"""Adapter bridging the real MTG engine (``mtg_bot.game.GameState``)
to the search-framework ``GameState`` protocol.

This is the one object that touches both sides of the wall. Search
algorithms (MCTS etc.) call its abstract protocol methods; internally
it drives the real engine one decision at a time.

Design choices for this first cut (documented so the next iteration
knows what it's stepping into):

- **Deterministic tactical policy.** Every player's ``agent`` is
  replaced with ``DefaultTacticalAgent`` (first-legal / conservative
  defaults). Mid-resolution callbacks (choose_target, choose_x,
  choose_sacrifice, etc.) all go through this policy rather than
  being exposed as search nodes. Same state + same sequence of
  top-level decisions → same resulting position, which is what MCTS
  needs.
- **Priority loop reimplemented here.** Rather than refactoring
  ``simulate.priority_loop`` to be pausable, the adapter runs its
  own priority-loop body in ``apply``. ``_pass_count`` lives on the
  adapter instance so it survives between ``apply()`` calls.
- **Non-priority steps auto-advance.** Between the caller's
  decisions, ``_advance_to_next_decision`` runs ``UNTAP`` /
  ``CLEANUP`` / step transitions / pre-priority triggers so the
  caller sees one decision moment after another instead of needing
  to manage turn structure.
- **Chance nodes not modeled.** Card draw uses the current library
  order deterministically. Full determinization — sampling possible
  library orderings for hidden-information search — is a later
  wave.
- **Combat declarations: opt-in factored decisions.** With
  ``expose_combat_decisions=False`` (default), declare-attackers and
  declare-blockers go through the tactical policy as a single
  atomic step. With ``expose_combat_decisions=True``, the adapter
  walks each eligible attacker (then each eligible blocker) one at
  a time and exposes ``AttackerDecision`` / ``BlockerDecision``
  search-decision values. This trades shallow-and-wide combat
  branching for deep-and-narrow per-creature decisions, which MCTS
  with progressive widening handles much better.

What IS exposed to search by default: priority-action choices —
casting spells, activating abilities, playing lands, passing
priority. With ``expose_combat_decisions=True``, also the per-
attacker / per-blocker decisions during combat.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

from mtg_bot.actions import (
    Action, DeclareAttackers, DeclareBlockers, PassPriority,
    apply_action, resolve_top_of_stack,
)
from mtg_bot.agents import Agent
from mtg_bot.combat import (
    clear_combat_state, eligible_attackers, eligible_blockers,
    resolve_combat_damage,
)
from mtg_bot.effects.generic.become_type import BecomeLandCreatureUEOT
from mtg_bot.game import GameState as EngineGameState, is_creature_now
from mtg_bot.legal import legal_actions
from mtg_bot.search.action_history import (
    ActionEvent, _is_strategically_relevant, record_action_event,
)
from mtg_bot.search._opp_capabilities import (
    opp_can_act, opp_capability_hash,
)
from mtg_bot.search._player_capabilities import (
    has_player_capability_to_act, player_capability_hash,
)
from mtg_bot.search.combat_decisions import (
    AttackerDecision, BlockerDecision,
)
from mtg_bot.search.heuristics_config import HeuristicsConfig
from mtg_bot.search.protocol import Decision, PlayerID
from mtg_bot.search.resolution_channel import (
    KIND_CARD_FROM_LIBRARY, KIND_CARD_FROM_OPPONENT_HAND,
    KIND_DEVOUR_LANDS, KIND_DISCARD, KIND_MODE, KIND_MODES,
    KIND_SACRIFICE, KIND_SHOCK_LIFE, KIND_SURVEIL, KIND_TARGET, KIND_X,
    KIND_YES_NO,
    ResolutionChannelAgent, _ResolutionPause,
)
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
from mtg_bot.simulate import _advance_to_next_active_player, _next_alive
from mtg_bot.turn import (
    STEP_ORDER,
    STEPS_WITH_PRIORITY,
    Step,
    run_beginning_of_combat_triggers,
    run_beginning_of_end_step_triggers,
    run_beginning_of_postcombat_main_triggers,
    run_beginning_of_upkeep_triggers,
    run_cleanup_step,
    run_delayed_end_step_triggers,
    run_draw_step,
    run_untap_step,
    state_based_actions,
)


# Combat-phase sentinel values for ``_combat_phase``.
_COMBAT_NONE = None
_COMBAT_ATTACKERS = "attackers"
_COMBAT_BLOCKERS = "blockers"


# Steps at which the active player's empty-stack priority is pruned to
# ``[PassPriority()]`` when ``phase_prune_enabled`` is True. These are
# the steps where casting an instant is *almost never* the right move
# for the active player: their own upkeep / draw / beginning-of-combat /
# end step. See ``docs/SEARCH_LIMITATIONS.md`` entry #6 for the
# acceptable-failure cases (pump-before-upkeep-trigger, mana-sink-end-
# step, end-step-flash-creature, BoC combat tricks). Note: opponent's
# priority during these same steps is NOT pruned — opp-side instants
# during your upkeep/end-step (responding to triggers, casting flash
# creatures) are real plays. The opp-capability auto-pass handles those
# moments separately.
_PHASE_PRUNE_STEPS: frozenset[Step] = frozenset({
    Step.UPKEEP,
    Step.DRAW,
    Step.BEGINNING_OF_COMBAT,
    Step.END,
})


# Steps at which the priority window is pruned to ``[PassPriority()]``
# when ``skip_empty_combat_enabled`` is True AND there are no attacking
# creatures on the battlefield. These are the post-declaration priority
# windows of the combat sub-steps — when no creature attacked, there's
# nothing to respond to with combat tricks (no creature to pump, no
# attack to redirect, no damage to prevent), and the step's priority
# windows can be cascaded through. See SEARCH_LIMITATIONS entry #9.
_EMPTY_COMBAT_PRUNE_STEPS: frozenset[Step] = frozenset({
    Step.DECLARE_ATTACKERS,
    Step.DECLARE_BLOCKERS,
    Step.COMBAT_DAMAGE,
    Step.END_OF_COMBAT,
})


# Maps the ``_ResolutionPause.kind`` value to the search-side
# decision class the adapter expects when search answers that pause.
_RESOLUTION_KIND_TO_DECISION_CLASS = {
    KIND_TARGET: ResolutionTargetDecision,
    KIND_DISCARD: ResolutionDiscardDecision,
    KIND_MODE: ResolutionModeDecision,
    KIND_MODES: ResolutionModesDecision,
    KIND_SACRIFICE: ResolutionSacrificeDecision,
    KIND_X: ResolutionXDecision,
    KIND_SURVEIL: ResolutionSurveilDecision,
    KIND_SHOCK_LIFE: ResolutionShockLifeDecision,
    KIND_CARD_FROM_LIBRARY: ResolutionCardFromLibraryDecision,
    KIND_CARD_FROM_OPPONENT_HAND: ResolutionCardFromOpponentHandDecision,
    KIND_DEVOUR_LANDS: ResolutionDevourLandsDecision,
    KIND_YES_NO: ResolutionYesNoDecision,
}


class _AdvanceSentinel:
    """Pseudo-decision replayed by the resolution harness when the
    adapter's own forward drive (the constructor's first
    ``_advance_to_next_decision``) hits an exposed callback."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<advance>"


_ADVANCE = _AdvanceSentinel()


def resolve_resolution_seats(
    expose: Union[bool, Iterable[int]], player_ids: Iterable[int],
) -> frozenset[int]:
    """Normalise ``expose_resolution_decisions`` to the set of seats
    whose resolution decisions are searched: ``True`` = every seat,
    ``False`` = none, an iterable = exactly those player ids."""
    if expose is True:
        return frozenset(player_ids)
    if expose is False or expose is None:
        return frozenset()
    return frozenset(int(p) for p in expose)


class MtgGameStateAdapter:
    """Wrap a real-engine ``GameState`` so MCTS (and friends) can
    drive it through the ``mtg_bot.search.protocol.GameState``
    interface.

    Ownership: the adapter owns the wrapped state — in particular it
    rewrites every player's ``agent`` to the tactical policy on
    construction. If the caller later hands the state to something
    else (a simulate loop, say), they should not expect the original
    agents to still be installed.
    """

    def __init__(
        self,
        engine_state: EngineGameState,
        tactical_agent: Optional[Agent] = None,
        expose_combat_decisions: bool = True,
        expose_resolution_decisions: Union[bool, Iterable[int]] = False,
        heuristics: Optional[HeuristicsConfig] = None,
    ) -> None:
        self._state = engine_state
        self._tactical = tactical_agent or DefaultTacticalAgent()
        # Seats whose mid-resolution callbacks are search decisions.
        # ``True`` = both seats (self-play), an iterable = only those
        # (a searched seat vs an auto-targeting one in an A/B), and
        # ``False`` = the legacy all-tactical behaviour.
        self._resolution_seats: frozenset[int] = resolve_resolution_seats(
            expose_resolution_decisions,
            (p.id for p in engine_state.players),
        )
        self._expose_resolution = bool(self._resolution_seats)
        # All five priority short-circuit / filter heuristics live on
        # one frozen ``HeuristicsConfig``. Default = production self-
        # play (every flag True). Eval / correctness-sensitive callers
        # pass ``HeuristicsConfig.disabled()``. Per-flag rationale and
        # acceptable-failure cases are documented at the per-helper
        # docstrings (``_should_phase_prune_priority`` etc.) and at
        # SEARCH_LIMITATIONS entries #6, #7, #9, #10, #11.
        self._heuristics = heuristics if heuristics is not None else (
            HeuristicsConfig()
        )
        # Agent installation. Exposed seats get the
        # ``ResolutionChannelAgent`` (which intercepts their callbacks;
        # non-exposed callbacks fall through to the tactical policy);
        # the other seats keep the tactical policy itself.
        if self._expose_resolution:
            self._channel_agent: Optional[ResolutionChannelAgent] = (
                ResolutionChannelAgent(fallback=self._tactical)
            )
        else:
            self._channel_agent = None
        self._install_agents(self._state)
        # Consecutive priority passes since the last non-pass or
        # stack resolution. Lives on the adapter (not the engine
        # state) because this is search-loop bookkeeping rather than
        # game state.
        self._pass_count = 0
        # Tracks whether the current step's pre-priority logic has
        # been run. False on fresh construction (nothing's run yet);
        # True once ``_enter_step`` has been invoked for the current
        # step; flipped back to False by ``_advance_to_next_step``.
        self._step_entered = False
        # Factored-combat exposure flag and per-phase state. Default
        # is True: search sees AttackerDecision / BlockerDecision
        # nodes (one branch per creature) at combat steps instead of
        # the adapter auto-resolving via DefaultTacticalAgent's
        # "attack-everything, block-nothing" policy. Pass False to
        # restore the auto-resolve path (used by a handful of tests
        # that drive non-combat behavior and want the simpler flow).
        self._expose_combat = expose_combat_decisions
        self._combat_phase: Optional[str] = _COMBAT_NONE
        # Order matters — search will see the next-pending creature
        # first. Sort by id for reproducibility across clones.
        self._pending_attackers: list[int] = []
        self._chosen_attackers: list[int] = []
        self._pending_blockers: list[int] = []
        self._chosen_blocks: dict[int, int] = {}
        self._combat_defender_id: Optional[PlayerID] = None
        # Resolution-pause state. When a priority action's resolution
        # raises ``_ResolutionPause``, the adapter rolls back to a
        # snapshot taken before the action and surfaces the decision
        # to search. ``_pre_action_clone`` is the snapshot;
        # ``_pending_replay_decision`` is the priority action being
        # replayed with growing resolution-queue answers.
        self._resolution_paused = False
        self._resolution_kind: Optional[str] = None
        self._resolution_player: Optional[PlayerID] = None
        self._resolution_options: tuple = ()
        # Per-option policy-slot offsets within the ``canonical_v3``
        # resolution block (``None`` = no slot), computed by the
        # channel agent from the LIVE state at pause time; and the
        # hidden card ids the paused callback revealed to the decider
        # (pinned by PIMC / ISMCTS determinizations).
        self._resolution_slot_offsets: tuple = ()
        self._resolution_revealed_ids: tuple = ()
        self._pre_action_clone: Optional[EngineGameState] = None
        # Adapter-side bookkeeping at the moment the pending action
        # was first attempted (``_bookkeeping_snapshot``). A rollback
        # restores it with the engine state; without it a pause during
        # the stack-resolving ``PassPriority`` replayed with a reset
        # ``_pass_count`` and passed priority forever.
        self._pre_action_bookkeeping: Optional[tuple] = None
        self._pending_replay_decision: Optional[Decision] = None
        # Structured-NN action history: the last
        # ``DEFAULT_ACTION_HISTORY_N`` strategically-relevant decisions
        # applied through this adapter (and its ancestors — ``clone()``
        # shares the tuple). Recorded by ``apply`` via
        # ``mtg_bot.search.action_history.record_action_event``; read
        # by ``data_gen`` (training rows) and ``NNEvaluator`` (search)
        # so the two see identical history. Immutable tuple: clones
        # share the reference, extension allocates a new tuple.
        self._recent_actions: tuple[ActionEvent, ...] = ()
        # Per-decision-moment cache of ``legal.legal_actions`` results.
        # ISMCTS calls ``legal_decisions`` AND ``legal_action_mask`` AND
        # ``_grow_union_mask`` (downstream of mask) all on the SAME
        # engine state per descent step — three identical enumerations
        # of a function that consumes 37% of self-play wall time per
        # ``docs/perf_self_play_20260504.md``. Cache one entry keyed by
        # the player whose priority-window legal set was computed; the
        # cache is invalidated whenever any engine-state mutation runs
        # (every ``apply()`` and every internal step transition that
        # alters ``self._state``). The cache stores the list reference
        # itself; callers that need a private copy do their own
        # ``list(...)`` of it. Lifetime: per priority moment, never
        # crosses an ``apply()`` boundary.
        self._legal_actions_cache_player: Optional[PlayerID] = None
        self._legal_actions_cache_value: Optional[list[Action]] = None
        # Per-decision-moment cache of the player-capability "can act"
        # bool. Keyed on a structural hash from
        # ``_player_capabilities.player_capability_hash``. Covers BOTH
        # the non-active-player priority moments (instant-speed only)
        # AND the active-player priority moments (sorcery-speed-aware
        # when stack is empty + step is main phase). When the priority
        # holder has no possible action, we short-circuit
        # ``legal_decisions`` to ``[PassPriority()]`` without
        # enumerating ``legal_actions``. With both players auto-passing
        # in their respective windows, "quiet" turns advance through
        # priority round in microseconds. See
        # ``_player_capabilities.py`` for the conservatism contract
        # and documented limitations (continuous-effect grants,
        # cost-reduction effects, etc.).
        # Stored as ``(player_id, is_active_player, hash, can_act)``;
        # one entry per adapter is sufficient (only one priority
        # holder at a time). Attribute name retained as
        # ``_opp_cap_cache`` for compat with tests that touched it
        # directly under the opp-only era.
        self._opp_cap_cache: Optional[
            tuple[PlayerID, bool, int, bool]
        ] = None
        # Drive forward until we either hit a terminal state or a
        # priority moment where the caller must decide. With exposure
        # on, the drive runs inside the resolution harness (a callback
        # fired by a pre-priority trigger becomes a pause).
        if self._expose_resolution:
            self._run_with_resolution_capture(_ADVANCE)
        else:
            self._advance_to_next_decision()

    # ---------------------------------------------------------------
    # Agent installation / resolution-pause introspection
    # ---------------------------------------------------------------

    def _install_agents(self, state: EngineGameState) -> None:
        """Give every player of ``state`` this adapter's agent for
        their seat: the channel agent for exposed seats, the tactical
        policy otherwise."""
        for p in state.players:
            if (
                self._channel_agent is not None
                and p.id in self._resolution_seats
            ):
                p.agent = self._channel_agent
            else:
                p.agent = self._tactical

    @property
    def resolution_seats(self) -> frozenset[int]:
        """Player ids whose resolution decisions are searched."""
        return self._resolution_seats

    @property
    def is_resolution_paused(self) -> bool:
        """True at a mid-resolution decision moment."""
        return self._resolution_paused

    @property
    def resolution_kind(self) -> Optional[str]:
        """The pending callback kind (``"target"``, ``"x"``, ...) at a
        pause, else ``None``."""
        return self._resolution_kind

    @property
    def resolution_options(self) -> tuple:
        """The raw engine-side options of the pending callback."""
        return self._resolution_options

    @property
    def resolution_slot_offsets(self) -> tuple:
        """Per-option offset into the ``canonical_v3`` resolution
        block (``None`` = the option has no slot), aligned with
        ``resolution_options``."""
        return self._resolution_slot_offsets

    @property
    def resolution_revealed_ids(self) -> tuple:
        """Hidden card ids the pending callback revealed to the
        decider (pinned by determinizers)."""
        return self._resolution_revealed_ids

    @property
    def resolution_queue(self) -> tuple:
        """Answers already given within the pending action."""
        if self._channel_agent is None:
            return ()
        return self._channel_agent.queue

    def resync_pause_snapshot(self) -> None:
        """Re-take the pre-action snapshot from the current engine
        state. Determinizers call this after permuting a PAUSED
        adapter's hidden zones in place: at a pause the live state IS
        a fresh copy of the snapshot, so copying it back keeps a later
        rollback inside the determinized world (otherwise the first
        resolution answer would roll back to the true, un-
        determinized state). No-op when not paused."""
        if self._resolution_paused and self._pre_action_clone is not None:
            self._pre_action_clone = self._state.clone()

    def _bookkeeping_snapshot(self) -> tuple:
        """Every adapter-side field an action can mutate besides the
        engine state itself."""
        return (
            self._pass_count,
            self._step_entered,
            self._combat_phase,
            tuple(self._pending_attackers),
            tuple(self._chosen_attackers),
            tuple(self._pending_blockers),
            tuple(self._chosen_blocks.items()),
            self._combat_defender_id,
        )

    def _restore_bookkeeping(self, snap: tuple) -> None:
        (
            self._pass_count,
            self._step_entered,
            self._combat_phase,
            pending_attackers,
            chosen_attackers,
            pending_blockers,
            chosen_blocks,
            self._combat_defender_id,
        ) = snap
        self._pending_attackers = list(pending_attackers)
        self._chosen_attackers = list(chosen_attackers)
        self._pending_blockers = list(pending_blockers)
        self._chosen_blocks = dict(chosen_blocks)

    # ---------------------------------------------------------------
    # Per-decision-moment legal_actions cache
    # ---------------------------------------------------------------

    def _invalidate_legal_actions_cache(self) -> None:
        """Drop any cached ``legal_actions`` result. Called from every
        site that mutates ``self._state`` (or replaces it with a new
        clone via the resolution-pause snapshot path). Cheap: two
        attribute writes.

        Also drops the opp-capability cache: any state mutation
        invalidates the same way (and the structural hash check on
        the next priority moment will re-validate or re-compute).
        """
        self._legal_actions_cache_player = None
        self._legal_actions_cache_value = None
        self._opp_cap_cache = None

    def _can_act_cached(
        self, player_id: PlayerID, *, is_active_player: bool,
    ) -> bool:
        """Return ``has_player_capability_to_act(state, player_id,
        is_active_player=...)``, memoized via the structural hash
        from ``player_capability_hash``.

        Cache structure: ``(player_id, is_active_player, hash,
        can_act_bool)`` — one entry per adapter, replaced on miss.
        The hash check is O(player's hand + battlefield + graveyard
        + exile); a hit avoids the full capability walk (which itself
        avoids ``legal_actions``).
        """
        h = player_capability_hash(
            self._state, player_id, is_active_player=is_active_player,
        )
        cached = self._opp_cap_cache
        if (
            cached is not None
            and cached[0] == player_id
            and cached[1] == is_active_player
            and cached[2] == h
        ):
            return cached[3]
        can_act = has_player_capability_to_act(
            self._state, player_id, is_active_player=is_active_player,
        )
        self._opp_cap_cache = (player_id, is_active_player, h, can_act)
        return can_act

    # Backwards-compat alias for the original opp-only cached check.
    def _opp_can_act_cached(self, opp_id: PlayerID) -> bool:
        """Legacy alias preserved for tests. Equivalent to
        ``_can_act_cached(opp_id, is_active_player=False)``."""
        return self._can_act_cached(opp_id, is_active_player=False)

    def _should_phase_prune_priority(self) -> bool:
        """Phase-based heuristic: active player auto-passes priority
        during their own UPKEEP / DRAW / BEGINNING_OF_COMBAT / END step
        when the stack is empty.

        This is a training-time heuristic (NOT CR-correct). It catches
        the "player HAS castable instants but it's a stupid phase to
        use them" case — composes ABOVE the opp-capability auto-pass
        which catches the "opp has nothing castable" case.

        Conditions to fire:
        * ``phase_prune_enabled`` is True.
        * Not terminal.
        * Not a resolution-pause moment.
        * Not a combat sub-decision moment.
        * Player with priority IS the active player (this is the
          active-side prune; opp-side instants during these steps are
          real plays — opp-capability handles those).
        * Stack is empty (a non-empty stack means there's something
          to potentially respond to; stay engaged).
        * Step is in ``_PHASE_PRUNE_STEPS``.

        Documented divergence from optimal play, see
        ``docs/SEARCH_LIMITATIONS.md`` entry #6.
        """
        if not self._heuristics.phase_prune_enabled:
            return False
        if self._state.is_over():
            return False
        if self._resolution_paused:
            return False
        if self._combat_phase != _COMBAT_NONE:
            return False
        # Active-only: opp's priority during these same steps still
        # exposes legitimate flash / triggered-response windows.
        if (
            self._state.priority_player_idx
            != self._state.active_player_idx
        ):
            return False
        # Stack non-empty means there's a spell / ability waiting to
        # resolve; the active player might want to respond to it.
        if len(self._state.stack) > 0:
            return False
        return self._state.step in _PHASE_PRUNE_STEPS

    def _should_force_landdrop(self) -> bool:
        """Forced second-main land drop heuristic: when the active
        player is at POSTCOMBAT_MAIN priority with empty stack, has
        not yet played a land this turn, has at least one land in
        hand, and controls fewer than six lands, ``PassPriority`` is
        filtered out of ``legal_decisions``.

        Rationale: at second main with a land in hand and unspent
        land drop, the optimal play is almost always "play the land"
        — leaving the land in hand wastes a permanent resource that
        the player can't recover later. The player MAY still want to
        cast a spell or activate an ability before playing the land
        (e.g. tap a creature for mana, then play the land), so we
        filter rather than force the decision down to PlayLand only.
        Stripping ``PassPriority`` makes the player play the land at
        SOME point during the priority window before passing.

        Conditions to fire:
        * ``force_landdrop_enabled`` is True.
        * Not terminal.
        * Not a resolution-pause moment.
        * Not a combat sub-decision moment.
        * Player with priority IS the active player.
        * Stack is empty.
        * Step is ``POSTCOMBAT_MAIN``.
        * Active player has ``lands_played_this_turn == 0``.
        * Active player has at least one land in hand.
        * Active player controls fewer than six lands.

        Documented divergence from optimal play, see
        ``docs/SEARCH_LIMITATIONS.md`` entry #7.
        """
        if not self._heuristics.force_landdrop_enabled:
            return False
        if self._state.is_over():
            return False
        if self._resolution_paused:
            return False
        if self._combat_phase != _COMBAT_NONE:
            return False
        if (
            self._state.priority_player_idx
            != self._state.active_player_idx
        ):
            return False
        if len(self._state.stack) > 0:
            return False
        if self._state.step != Step.POSTCOMBAT_MAIN:
            return False
        active = self._state.players[self._state.active_player_idx]
        if active.lands_played_this_turn != 0:
            return False
        # Need ≥1 land in hand (otherwise we'd filter out
        # PassPriority leaving an unplayable empty list).
        has_land_in_hand = any(
            obj.printing.is_land for obj in active.hand
        )
        if not has_land_in_hand:
            return False
        # Land flood guard: once you have 6+ lands in play, holding
        # the next land for a future turn is often correct (curve is
        # done, the next land just floods). Six is conservative —
        # most decks plateau on lands well before that.
        n_lands_in_play = sum(
            1 for obj in self._state.battlefield
            if obj.controller_id == active.id and obj.printing.is_land
        )
        if n_lands_in_play >= 6:
            return False
        return True

    def _should_skip_empty_combat(self) -> bool:
        """Skip-empty-combat heuristic: at any combat sub-step priority
        window with empty stack and no attacking creatures on the
        battlefield, return ``[PassPriority()]`` to cascade through
        without enumerating legal_actions for either side.

        Conditions to fire:
        * ``skip_empty_combat_enabled`` is True.
        * Not terminal.
        * Not a resolution-pause moment.
        * Not a combat sub-decision moment (factored attacker/blocker
          path is unaffected — those are pre-declaration windows).
        * Stack is empty.
        * Step is in ``_EMPTY_COMBAT_PRUNE_STEPS``
          (DECLARE_ATTACKERS, DECLARE_BLOCKERS, COMBAT_DAMAGE,
          END_OF_COMBAT).
        * No creature on the battlefield is currently attacking.

        Fires for BOTH active and opponent priority windows: with no
        attackers there are no combat tricks to play, so opp's
        instant-speed window is also dead.

        Documented divergence from optimal play, see
        ``docs/SEARCH_LIMITATIONS.md`` entry #9.
        """
        if not self._heuristics.skip_empty_combat_enabled:
            return False
        if self._state.is_over():
            return False
        if self._resolution_paused:
            return False
        if self._combat_phase != _COMBAT_NONE:
            return False
        if len(self._state.stack) > 0:
            return False
        if self._state.step not in _EMPTY_COMBAT_PRUNE_STEPS:
            return False
        # No attacking creatures → nothing to respond to with combat
        # tricks. Both players' priority windows can be cascaded
        # through.
        for obj in self._state.battlefield:
            if obj.attacking:
                return False
        return True

    def _should_skip_opp_postcombat_main(self) -> bool:
        """Skip our priority window during the opponent's POSTCOMBAT_MAIN
        when the stack is empty. The capability auto-pass already
        handles the "we have nothing castable" subset; this fires for
        the remaining case (we DO have a castable instant but
        responding at opp end-of-main-2 is essentially never useful —
        any reasonable instant timing is BEFORE their combat or AT
        their end step, not in the dead window between).

        Conditions to fire:
        * ``skip_opp_postcombat_main_enabled`` is True.
        * Not terminal / not resolution-paused / not in a factored
          combat sub-decision.
        * Stack is empty.
        * Step is ``POSTCOMBAT_MAIN``.
        * Player with priority is the NON-active player (it's opp's
          turn and we hold priority).

        Documented divergence from optimal play, see
        ``docs/SEARCH_LIMITATIONS.md`` entry #10.
        """
        if not self._heuristics.skip_opp_postcombat_main_enabled:
            return False
        if self._state.is_over():
            return False
        if self._resolution_paused:
            return False
        if self._combat_phase != _COMBAT_NONE:
            return False
        if len(self._state.stack) > 0:
            return False
        if self._state.step != Step.POSTCOMBAT_MAIN:
            return False
        # Active-only's complement: the priority holder must be the
        # NON-active player (we're being asked to act on opp's turn).
        return (
            self._state.priority_player_idx
            != self._state.active_player_idx
        )

    def _should_skip_dead_main1(self) -> bool:
        """Skip the active player's PRECOMBAT_MAIN priority window
        when the player has nothing useful to do that's specific to
        main 1 (vs main 2). Returns ``[PassPriority()]`` to cascade
        through the dead window.

        Conditions to fire:
        * ``skip_dead_main1_enabled`` is True.
        * Not terminal / not resolution-paused / not in a factored
          combat sub-decision.
        * Stack is empty.
        * Step is ``PRECOMBAT_MAIN``.
        * Player with priority IS the active player.
        * Active player controls 0 creatures on the battlefield.
        * Active player has no card in hand whose printing has the
          ``haste`` keyword (i.e. no haste creature to cast for an
          immediate attack).
        * Active player controls no noncreature permanent whose
          printing has an activated ability that animates it into a
          creature (e.g. manlands like Mishra's Factory).

        The "no creatures + no haste + no animator" conjunction means
        nothing the player does at main 1 has a different effect from
        doing it at main 2 — there's no creature to attack with even
        if a creature gets cast or animated. Auto-passing main 1 is
        functionally equivalent to playing through it.

        Documented divergence from optimal play, see
        ``docs/SEARCH_LIMITATIONS.md`` entry #11.
        """
        if not self._heuristics.skip_dead_main1_enabled:
            return False
        if self._state.is_over():
            return False
        if self._resolution_paused:
            return False
        if self._combat_phase != _COMBAT_NONE:
            return False
        if len(self._state.stack) > 0:
            return False
        if self._state.step != Step.PRECOMBAT_MAIN:
            return False
        if (
            self._state.priority_player_idx
            != self._state.active_player_idx
        ):
            return False
        active = self._state.players[self._state.active_player_idx]
        active_id = active.id
        # No creatures in play. Use ``is_creature_now`` (which respects
        # natively-printed creature types AND animation cache flag) —
        # ``permanent_is_creature`` alone is False on natively-printed
        # creatures and only True when an animation effect has run.
        for obj in self._state.battlefield:
            if obj.controller_id != active_id:
                continue
            if is_creature_now(obj):
                return False
        # No haste creature in hand.
        for card in active.hand:
            keywords = getattr(card.printing, "keywords", frozenset())
            if "haste" in keywords:
                return False
        # No noncreature animator on battlefield. We only check
        # ``BecomeLandCreatureUEOT`` here — the only self-animation
        # Effect class today. Future Effect classes that also animate
        # permanents would need to be added to this isinstance check
        # (e.g. an artifact-self-animator class). False-negative is
        # safe (heuristic doesn't fire when it could; just less
        # perf gain); false-positive is the dangerous direction
        # (heuristic fires when an animatable permanent exists), but
        # that requires a NEW effect class we haven't yet added.
        for obj in self._state.battlefield:
            if obj.controller_id != active_id:
                continue
            if is_creature_now(obj):
                continue
            for ability in obj.printing.activated_abilities:
                if ability.is_mana_ability:
                    continue
                if isinstance(ability.effect, BecomeLandCreatureUEOT):
                    return False
        return True

    def _can_short_circuit_to_pass(self) -> bool:
        """True iff the current decision moment is a priority window
        whose holder has no possible action. Auto-passes BOTH the
        active player AND the non-active player (each with their
        appropriate sorcery-window-aware capability check). When this
        returns True, ``legal_decisions`` may safely return
        ``[PassPriority()]`` without enumerating ``legal_actions``.

        Conditions:
        * Not terminal.
        * Not a resolution-pause moment.
        * Not a combat sub-decision moment.
        * The priority holder's capability check returns False.
        """
        if self._state.is_over():
            return False
        if self._resolution_paused:
            return False
        if self._combat_phase != _COMBAT_NONE:
            return False
        priority_idx = self._state.priority_player_idx
        active_idx = self._state.active_player_idx
        is_active_priority = (priority_idx == active_idx)
        priority_pid = self._state.players[priority_idx].id
        return not self._can_act_cached(
            priority_pid, is_active_player=is_active_priority,
        )

    def _legal_actions_cached(
        self, player_id: PlayerID,
    ) -> list[Action]:
        """Return ``legal.legal_actions(self._state, player_id)``,
        memoized for the current decision moment. The first call after
        an ``apply()`` (or other state-mutating event) computes and
        stores; subsequent calls within the same decision moment for
        the same ``player_id`` return the cached list directly.

        The cached list is the SAME object across cache hits — callers
        must treat it as read-only. ``legal_decisions`` already does
        ``list(...)`` on the result before exposing it to search; the
        mask helper iterates without mutating. If a hypothetical
        future caller needs a private copy, they should ``list(...)``
        the result themselves.
        """
        if (
            self._legal_actions_cache_value is not None
            and self._legal_actions_cache_player == player_id
        ):
            return self._legal_actions_cache_value
        result = legal_actions(self._state, player_id)
        self._legal_actions_cache_player = player_id
        self._legal_actions_cache_value = result
        return result

    # ---------------------------------------------------------------
    # search.GameState protocol
    # ---------------------------------------------------------------

    def clone(self) -> "MtgGameStateAdapter":
        c = MtgGameStateAdapter.__new__(MtgGameStateAdapter)
        c._state = self._state.clone()
        c._tactical = self._tactical
        c._expose_resolution = self._expose_resolution
        c._resolution_seats = self._resolution_seats
        # ``HeuristicsConfig`` is frozen, so sharing the reference
        # across the clone is safe (no aliasing hazard); no copy needed.
        c._heuristics = self._heuristics
        # Resolution-channel agent is per-adapter-instance so each
        # clone has its own replay queue. Snapshot/restore preserves
        # queue state from the source.
        if self._expose_resolution and self._channel_agent is not None:
            c._channel_agent = ResolutionChannelAgent(
                fallback=self._tactical,
            )
            c._channel_agent.restore(self._channel_agent.snapshot())
        else:
            c._channel_agent = None
        c._install_agents(c._state)
        c._pass_count = self._pass_count
        c._step_entered = self._step_entered
        c._expose_combat = self._expose_combat
        c._combat_phase = self._combat_phase
        c._pending_attackers = list(self._pending_attackers)
        c._chosen_attackers = list(self._chosen_attackers)
        c._pending_blockers = list(self._pending_blockers)
        c._chosen_blocks = dict(self._chosen_blocks)
        c._combat_defender_id = self._combat_defender_id
        # Resolution-pause state.
        c._resolution_paused = self._resolution_paused
        c._resolution_kind = self._resolution_kind
        c._resolution_player = self._resolution_player
        c._resolution_options = self._resolution_options
        c._resolution_slot_offsets = self._resolution_slot_offsets
        c._resolution_revealed_ids = self._resolution_revealed_ids
        c._pre_action_clone = (
            self._pre_action_clone.clone()
            if self._pre_action_clone is not None else None
        )
        c._pre_action_bookkeeping = self._pre_action_bookkeeping
        c._pending_replay_decision = self._pending_replay_decision
        # Immutable tuple — shared by reference (no per-clone cost).
        # ``apply`` on either side rebinds its own attribute.
        c._recent_actions = self._recent_actions
        # Cache lifetime is per-adapter-per-priority-moment: the clone
        # starts with an empty cache. Sharing the source's cached list
        # with the clone would be tempting (the engine_state at clone
        # time is a deep copy of the source's, so legal_actions would
        # match) — but ISMCTS calls ``_fresh_determinization`` which
        # MUTATES ``c.engine_state`` in place right after this clone
        # returns, permuting hidden-zone printings. After that mutation
        # a shared cache would silently return stale entries (e.g. for
        # the Mm'menon cast-from-library-top check, which inspects
        # ``library[0]``). Starting with an empty cache forces the
        # next ``legal_decisions`` call on the clone to re-enumerate
        # against the post-determinization state — which is what we
        # want and is no worse than the pre-cache behavior.
        c._legal_actions_cache_player = None
        c._legal_actions_cache_value = None
        # Opp-capability cache: same per-priority-moment lifetime as
        # the legal_actions cache. Sharing across a clone-then-mutate
        # boundary (ISMCTS's ``_fresh_determinization`` permutes
        # hidden zones in place) would risk a stale "no capability"
        # answer. Clone starts with an empty cache; the first
        # priority moment recomputes the hash and (likely) re-decides.
        c._opp_cap_cache = None
        return c

    def active_player(self) -> PlayerID:
        # During mid-resolution pauses, it's the player the engine
        # callback was asking — not whoever held engine priority.
        if self._resolution_paused:
            assert self._resolution_player is not None
            return self._resolution_player
        # During factored combat, the decision-maker isn't whoever
        # holds engine priority — it's the active player (for
        # attacker decisions) or the defender (for blocker decisions).
        if self._combat_phase == _COMBAT_ATTACKERS:
            return self._state.active_player_idx
        if self._combat_phase == _COMBAT_BLOCKERS:
            assert self._combat_defender_id is not None
            return self._combat_defender_id
        return self._state.priority_player_idx

    def legal_decisions(self) -> list[Decision]:
        if self._state.is_over():
            return []
        if self._resolution_paused:
            return self._resolution_legal_decisions()
        # Phase-based active-priority prune: cheap step-membership
        # check, fires BEFORE the capability hash compute. When the
        # active player has priority during their own UPKEEP / DRAW /
        # BEGINNING_OF_COMBAT / END step with empty stack, short-
        # circuit to ``[PassPriority()]`` without enumerating
        # ``legal_actions``. See ``docs/SEARCH_LIMITATIONS.md`` entry
        # #6 for the documented divergence from optimal play.
        if self._should_phase_prune_priority():
            return [PassPriority()]
        # Skip-empty-combat: at any combat sub-step priority window
        # (DECLARE_ATTACKERS / DECLARE_BLOCKERS / COMBAT_DAMAGE /
        # END_OF_COMBAT) with empty stack and no attacking creatures
        # on the battlefield, short-circuit to ``[PassPriority()]``.
        # Fires for both active and opp priority. See
        # ``docs/SEARCH_LIMITATIONS.md`` entry #9.
        if self._should_skip_empty_combat():
            return [PassPriority()]
        # Skip our priority during opp's POSTCOMBAT_MAIN with empty
        # stack. See ``docs/SEARCH_LIMITATIONS.md`` entry #10.
        if self._should_skip_opp_postcombat_main():
            return [PassPriority()]
        # Skip-dead-main1: at active PRECOMBAT_MAIN with no creatures
        # in play, no haste-creature in hand, and no animator
        # permanent. See ``docs/SEARCH_LIMITATIONS.md`` entry #11.
        if self._should_skip_dead_main1():
            return [PassPriority()]
        # Capability auto-pass: when the priority holder has no
        # possible action (instant-speed for opp / non-active priority
        # windows; sorcery-aware union for active priority windows in
        # main phases with empty stack), return PassPriority
        # immediately without enumerating ``legal_actions``. See
        # ``_player_capabilities.py`` for the conservatism contract.
        if self._can_short_circuit_to_pass():
            return [PassPriority()]
        if self._combat_phase == _COMBAT_ATTACKERS:
            # One pending attacker at a time; each can attack or not.
            next_id = self._pending_attackers[0]
            return [
                AttackerDecision(attacker_id=next_id, attack=True),
                AttackerDecision(attacker_id=next_id, attack=False),
            ]
        if self._combat_phase == _COMBAT_BLOCKERS:
            next_id = self._pending_blockers[0]
            attacking_ids = [
                o.id for o in self._state.battlefield
                if o.attacking
            ]
            options: list[Decision] = [
                BlockerDecision(blocker_id=next_id, attacker_id=atk_id)
                for atk_id in attacking_ids
            ]
            options.append(
                BlockerDecision(blocker_id=next_id, attacker_id=None)
            )
            return options
        decisions = list(self._legal_actions_cached(self.active_player()))
        # Forced second-main land drop: at active POSTCOMBAT_MAIN
        # priority with empty stack, unspent land drop, ≥1 land in
        # hand, and <6 lands in play, filter PassPriority OUT. The
        # player can still cast a spell or activate an ability, but
        # cannot end the priority window without playing a land.
        # See ``docs/SEARCH_LIMITATIONS.md`` entry #7.
        if self._should_force_landdrop():
            decisions = [d for d in decisions if not isinstance(d, PassPriority)]
        return decisions

    def _resolution_legal_decisions(self) -> list[Decision]:
        """Materialize resolution-pause options as search-side
        decision objects. Each option becomes one ``Resolution*``
        instance."""
        kind = self._resolution_kind
        opts = self._resolution_options
        if kind == KIND_TARGET:
            return [ResolutionTargetDecision(target=t) for t in opts]
        if kind == KIND_DISCARD:
            return [ResolutionDiscardDecision(card_id=cid) for cid in opts]
        if kind == KIND_MODE:
            return [ResolutionModeDecision(mode_index=i) for i in opts]
        if kind == KIND_MODES:
            return [
                ResolutionModesDecision(mode_indices=tuple(s)) for s in opts
            ]
        if kind == KIND_SACRIFICE:
            return [
                ResolutionSacrificeDecision(object_id=oid) for oid in opts
            ]
        if kind == KIND_X:
            return [ResolutionXDecision(x_value=v) for v in opts]
        if kind == KIND_SURVEIL:
            return [ResolutionSurveilDecision(mill=v) for v in opts]
        if kind == KIND_SHOCK_LIFE:
            return [ResolutionShockLifeDecision(pay=v) for v in opts]
        if kind == KIND_CARD_FROM_LIBRARY:
            return [
                ResolutionCardFromLibraryDecision(card_id=cid) for cid in opts
            ]
        if kind == KIND_CARD_FROM_OPPONENT_HAND:
            return [
                ResolutionCardFromOpponentHandDecision(card_id=cid)
                for cid in opts
            ]
        if kind == KIND_DEVOUR_LANDS:
            return [
                ResolutionDevourLandsDecision(land_ids=subset)
                for subset in opts
            ]
        if kind == KIND_YES_NO:
            # opts is always (True, False) — channel agent supplies
            # both branches; the prompt itself isn't part of the
            # decision space. Yes-first matches the order the channel
            # agent enumerates.
            return [ResolutionYesNoDecision(yes=v) for v in opts]
        raise RuntimeError(f"unknown resolution kind: {kind!r}")

    def chance_outcomes(self) -> list[tuple[Decision, float]]:
        # V1: no chance nodes. Determinization is a later wave.
        return []

    def is_terminal(self) -> bool:
        return self._state.is_over()

    def winner(self) -> Optional[PlayerID]:
        return self._state.winner_id

    def players(self) -> list[PlayerID]:
        return [p.id for p in self._state.players]

    def apply(self, decision: Decision) -> None:
        if self._state.is_over():
            raise ValueError("apply() called on terminal state")
        # Record the decision for the structured-NN action history
        # BEFORE applying (turn / step / actor and the snapshot are
        # the pre-apply moment), but only commit it AFTER the apply
        # succeeds so a raising apply leaves the history untouched.
        # PassPriority / mana abilities (~90% of applied decisions)
        # never enter the history, so gate on the §3 filter first and
        # only pay for ``active_player()`` + the snapshot clone on the
        # relevant ~10%.
        if _is_strategically_relevant(decision, self._state):
            history = record_action_event(
                self._recent_actions,
                action=decision,
                pre_apply_state=self._state,
                actor_player_id=self.active_player(),
            )
        else:
            history = self._recent_actions
        self._apply_decision(decision)
        self._recent_actions = history

    def _apply_decision(self, decision: Decision) -> None:
        """The dispatch body of ``apply`` (resolution-pause answers,
        factored-combat sub-decisions, priority actions). Split out so
        ``apply`` can wrap it with the action-history bookkeeping
        without touching the branchy early-return structure."""
        # Cache invariant: any ``apply()`` will mutate ``self._state``
        # (or replace it via resolution-pause rollback). Dropping the
        # cached legal_actions here guarantees the next
        # ``legal_decisions()`` call re-enumerates against the fresh
        # state. Putting the invalidation at the top of ``apply()``
        # rather than inside each mutator is the simplest invariant
        # that survives every dispatch branch below.
        self._invalidate_legal_actions_cache()

        # Resolution-pause answer routing — consume a Resolution*
        # decision, push to the channel agent's replay queue, rollback
        # state, and retry the saved action.
        if self._resolution_paused:
            self._apply_resolution_decision(decision)
            return

        # Validate BEFORE any snapshot / mutation so a wrong decision
        # type leaves the adapter untouched. Combat sub-decisions are
        # search-side types, not engine actions.
        if self._combat_phase == _COMBAT_ATTACKERS:
            self._check_attacker_decision(decision)
        elif self._combat_phase == _COMBAT_BLOCKERS:
            self._check_blocker_decision(decision)
        elif not isinstance(decision, Action):
            raise TypeError(
                f"MtgGameStateAdapter expects an mtg_bot.actions.Action "
                f"at priority moments, got {type(decision).__name__}: "
                f"{decision!r}"
            )

        # With resolution exposure, wrap the action (priority action
        # OR factored attacker / blocker decision -- attack triggers
        # and step transitions fire callbacks too) in the pause-catch
        # + replay harness. Without, just run it — no snapshot, no
        # replay.
        if self._expose_resolution:
            self._run_with_resolution_capture(decision)
        else:
            self._dispatch_live(decision)

    def _dispatch_live(self, decision: Decision) -> None:
        """Run ``decision`` on the live state and drive forward to the
        next decision moment. The single body both the plain path and
        every resolution-harness (re)play go through."""
        if decision is _ADVANCE:
            self._advance_to_next_decision()
            return
        if self._combat_phase == _COMBAT_ATTACKERS:
            self._apply_attacker_decision(decision)
        elif self._combat_phase == _COMBAT_BLOCKERS:
            self._apply_blocker_decision(decision)
        else:
            self._run_priority_action(decision)
        self._advance_to_next_decision()

    def _run_priority_action(self, decision: Decision) -> None:
        """The original priority-action body, extracted so it can be
        invoked from multiple call sites (normal path and replay
        from resolution rollback)."""
        pid = self._state.priority_player_idx

        if isinstance(decision, PassPriority):
            self._pass_count += 1
            alive = sum(
                1 for p in self._state.players if not p.has_lost
            )
            if self._pass_count >= alive:
                # Everyone passed in succession.
                self._pass_count = 0
                if self._state.stack:
                    resolve_top_of_stack(self._state)
                    state_based_actions(self._state)
                    if self._state.is_over():
                        return
                    # Priority returns to active player after each
                    # resolution. Still the same step.
                    self._state.priority_player_idx = _next_alive(
                        self._state, self._state.active_player_idx
                    )
                else:
                    # All passed with empty stack — step ends.
                    self._advance_to_next_step()
            else:
                # Pass to the next player.
                self._state.priority_player_idx = _next_alive(
                    self._state,
                    (pid + 1) % len(self._state.players),
                )
        else:
            apply_action(self._state, pid, decision)
            state_based_actions(self._state)
            if self._state.is_over():
                return
            self._pass_count = 0
            # Priority returns to active after any action.
            self._state.priority_player_idx = _next_alive(
                self._state, self._state.active_player_idx
            )

    # ---------------------------------------------------------------
    # Resolution decision handling
    # ---------------------------------------------------------------

    def _run_with_resolution_capture(self, decision: Decision) -> None:
        """Save a pre-action snapshot (engine state AND adapter
        bookkeeping), run ``decision`` (a priority action, a factored
        combat decision, or the ``_ADVANCE`` forward drive), and catch
        any ``_ResolutionPause``. On pause, roll back and store the
        pending decision for replay."""
        assert self._channel_agent is not None
        # Snapshot the full engine state BEFORE the action. On pause
        # we roll back to this clone (re-cloning each time so the
        # snapshot itself stays pristine for subsequent pauses).
        self._pre_action_clone = self._state.clone()
        self._pre_action_bookkeeping = self._bookkeeping_snapshot()
        self._pending_replay_decision = decision
        self._channel_agent.reset_queue()
        self._attempt_pending_action()

    def _rollback_to_snapshot(self) -> None:
        """Restore ``self._state`` from ``self._pre_action_clone`` and
        the adapter bookkeeping from ``_pre_action_bookkeeping``, and
        re-install this adapter's agents on the rolled-back players.

        The snapshot carries whatever agents were installed at the
        time it was taken -- typically the original adapter's channel
        agent. After cloning the adapter, the clone has its OWN
        channel agent but the snapshot still points at the original.
        Re-installing agents here keeps each adapter's resolution
        queue isolated from its siblings.
        """
        assert self._pre_action_clone is not None
        assert self._channel_agent is not None
        self._state = self._pre_action_clone.clone()
        self._install_agents(self._state)
        if self._pre_action_bookkeeping is not None:
            self._restore_bookkeeping(self._pre_action_bookkeeping)

    def _attempt_pending_action(self) -> None:
        """Try the current pending action. Handles pause, success
        (the live dispatch already advanced to the next decision),
        and cleanup."""
        assert self._pending_replay_decision is not None
        assert self._channel_agent is not None
        try:
            self._dispatch_live(self._pending_replay_decision)
        except _ResolutionPause as pause:
            # Engine asked for a callback we don't have in the queue.
            # Rollback to the pre-action snapshot and expose the
            # decision to search. Keep the snapshot around for the
            # next replay attempt.
            self._rollback_to_snapshot()
            self._channel_agent.rewind()
            self._resolution_paused = True
            self._resolution_kind = pause.kind
            self._resolution_player = pause.player_id
            self._resolution_options = pause.options
            self._resolution_slot_offsets = pause.slot_offsets
            self._resolution_revealed_ids = pause.revealed_ids
            return
        # Success -- discard the snapshot and replay state.
        self._pre_action_clone = None
        self._pre_action_bookkeeping = None
        self._pending_replay_decision = None

    def _apply_resolution_decision(self, decision: Decision) -> None:
        """Caller supplied an answer to the currently-paused
        resolution callback. Push it to the replay queue and retry."""
        assert self._channel_agent is not None
        expected_type = _RESOLUTION_KIND_TO_DECISION_CLASS[
            self._resolution_kind
        ]
        if not isinstance(decision, expected_type):
            raise TypeError(
                f"Resolution pause expects {expected_type.__name__} "
                f"for a {self._resolution_kind} callback; got "
                f"{type(decision).__name__}: {decision!r}"
            )
        self._channel_agent.push_answer(decision)
        # Clear pause metadata; state will be reset by the retry.
        self._resolution_paused = False
        self._resolution_kind = None
        self._resolution_player = None
        self._resolution_options = ()
        self._resolution_slot_offsets = ()
        self._resolution_revealed_ids = ()
        # Rollback and replay. ``_attempt_pending_action`` handles
        # the outcome (next pause or successful completion).
        self._rollback_to_snapshot()
        self._channel_agent.rewind()
        self._attempt_pending_action()

    # ---------------------------------------------------------------
    # Combat decision handling
    # ---------------------------------------------------------------

    def _check_attacker_decision(self, decision: Decision) -> None:
        if not isinstance(decision, AttackerDecision):
            raise TypeError(
                f"In attacker decision phase, adapter expects "
                f"AttackerDecision; got {type(decision).__name__}: "
                f"{decision!r}"
            )
        next_id = self._pending_attackers[0]
        if decision.attacker_id != next_id:
            raise ValueError(
                f"AttackerDecision is for attacker {decision.attacker_id}, "
                f"but the next pending attacker is {next_id}. The adapter "
                f"enumerates pending attackers in a fixed order; pass "
                f"a decision for the head of legal_decisions()."
            )

    def _apply_attacker_decision(self, decision: Decision) -> None:
        self._check_attacker_decision(decision)
        if decision.attack:
            self._chosen_attackers.append(decision.attacker_id)
        self._pending_attackers.pop(0)
        if not self._pending_attackers:
            self._finalize_attacker_phase()

    def _finalize_attacker_phase(self) -> None:
        ap = self._state.active_player()
        apply_action(
            self._state, ap.id,
            DeclareAttackers(
                attacker_ids=frozenset(self._chosen_attackers)
            ),
        )
        self._combat_phase = _COMBAT_NONE
        self._chosen_attackers = []
        # Now enter the step's priority window.
        self._enter_priority_for_current_step()

    def _check_blocker_decision(self, decision: Decision) -> None:
        if not isinstance(decision, BlockerDecision):
            raise TypeError(
                f"In blocker decision phase, adapter expects "
                f"BlockerDecision; got {type(decision).__name__}: "
                f"{decision!r}"
            )
        next_id = self._pending_blockers[0]
        if decision.blocker_id != next_id:
            raise ValueError(
                f"BlockerDecision is for blocker {decision.blocker_id}, "
                f"but the next pending blocker is {next_id}."
            )

    def _apply_blocker_decision(self, decision: Decision) -> None:
        self._check_blocker_decision(decision)
        if decision.attacker_id is not None:
            self._chosen_blocks[decision.blocker_id] = decision.attacker_id
        self._pending_blockers.pop(0)
        if not self._pending_blockers:
            self._finalize_blocker_phase()

    def _finalize_blocker_phase(self) -> None:
        assert self._combat_defender_id is not None
        apply_action(
            self._state, self._combat_defender_id,
            DeclareBlockers(
                assignments=tuple(self._chosen_blocks.items())
            ),
        )
        self._combat_phase = _COMBAT_NONE
        self._chosen_blocks = {}
        self._combat_defender_id = None
        self._enter_priority_for_current_step()

    def _enter_priority_for_current_step(self) -> None:
        """After finalizing a combat sub-phase, the current step
        still needs its priority window opened. Mirrors the tail of
        ``_enter_step``."""
        self._pass_count = 0
        self._state.priority_player_idx = _next_alive(
            self._state, self._state.active_player_idx
        )

    # ---------------------------------------------------------------
    # Engine-driving internals
    # ---------------------------------------------------------------

    def _advance_to_next_decision(self) -> None:
        """From whatever state we're in, drive the engine forward
        until either the game ends or the next decision is pending.
        Idempotent: safe to call when we're already at a decision
        moment (the while-loop exits immediately in that case)."""
        while not self._state.is_over():
            # Combat sub-decisions trump step-advancement — the caller
            # must clear them first.
            if self._combat_phase != _COMBAT_NONE:
                return

            if self._state.step is None:
                self._state.step = Step.UNTAP
                self._step_entered = False

            if not self._step_entered:
                self._enter_step(self._state.step)
                self._step_entered = True
                if self._state.is_over():
                    return
                # Entering a combat-decision step might have set up a
                # combat phase; if so, return to caller for the
                # sub-decision.
                if self._combat_phase != _COMBAT_NONE:
                    return

            if self._state.step in STEPS_WITH_PRIORITY:
                return
            # Non-priority step already fully handled by
            # _enter_step; move on.
            self._advance_to_next_step()

    def _advance_to_next_step(self) -> None:
        """Current step has concluded. Bump ``state.step`` to the
        next step (rolling over the turn after CLEANUP). Does NOT
        run the new step's pre-priority logic — that happens the
        next time _advance_to_next_decision is called."""
        current = self._state.step
        if current == Step.CLEANUP:
            self._state.turn_number += 1
            _advance_to_next_active_player(self._state)
            self._state.step = Step.UNTAP
        else:
            idx = STEP_ORDER.index(current)
            self._state.step = STEP_ORDER[idx + 1]
        self._step_entered = False

    def _enter_step(self, step: Step) -> None:
        """Run a step's pre-priority logic (or full logic for
        non-priority steps). Mirrors the step-specific dispatch
        inside ``simulate.run_step`` but stops just before
        ``priority_loop`` — the adapter drives priority itself.

        For priority steps: sets up ``priority_player_idx`` and
        ``_pass_count`` for the first decision of this step.
        For DECLARE_ATTACKERS / DECLARE_BLOCKERS with combat
        decisions exposed: sets up ``_combat_phase`` and pending
        decisions instead of entering priority directly.
        """
        if step == Step.UNTAP:
            run_untap_step(self._state)
            return

        if step == Step.UPKEEP:
            run_beginning_of_upkeep_triggers(self._state)
            state_based_actions(self._state)

        elif step == Step.DRAW:
            # Starting player skips their very first draw step —
            # same rule run_step enforces. ``turn_number == 1`` alone
            # is the correct condition: on turn 1 the active player
            # IS the starting player regardless of which seat they
            # occupy, so this works for both standard (seat 0 starts)
            # and alternating-seat (seat 1 starts) games.
            if self._state.turn_number != 1:
                run_draw_step(self._state)

        elif step == Step.BEGINNING_OF_COMBAT:
            run_beginning_of_combat_triggers(self._state)
            state_based_actions(self._state)

        elif step == Step.POSTCOMBAT_MAIN:
            # Mirror the simulate.run_step wiring for the DSK Survival
            # second-main-phase trigger family.
            run_beginning_of_postcombat_main_triggers(self._state)
            state_based_actions(self._state)

        elif step == Step.DECLARE_ATTACKERS:
            self._enter_declare_attackers()
            return  # _enter_declare_attackers handles the priority-tail.

        elif step == Step.DECLARE_BLOCKERS:
            self._enter_declare_blockers()
            return

        elif step == Step.COMBAT_DAMAGE:
            if any(o.attacking for o in self._state.battlefield):
                defender_idx = _next_alive(
                    self._state,
                    (self._state.active_player_idx + 1)
                    % len(self._state.players),
                )
                resolve_combat_damage(
                    self._state, self._state.players[defender_idx].id
                )
                state_based_actions(self._state)

        elif step == Step.END:
            run_beginning_of_end_step_triggers(self._state)
            state_based_actions(self._state)
            if not self._state.is_over():
                run_delayed_end_step_triggers(self._state)
                state_based_actions(self._state)

        elif step == Step.CLEANUP:
            run_cleanup_step(self._state)
            return

        if self._state.is_over():
            return

        # END_OF_COMBAT post-hook — matches tail of
        # simulate.run_step.
        if step == Step.END_OF_COMBAT:
            clear_combat_state(self._state)

        if step in STEPS_WITH_PRIORITY:
            self._enter_priority_for_current_step()

    def _enter_declare_attackers(self) -> None:
        """Set up DECLARE_ATTACKERS step. With combat exposed,
        switch into the per-attacker decision phase. Without,
        delegate to the tactical agent and apply the engine action
        immediately."""
        ap = self._state.active_player()
        eligible = eligible_attackers(self._state, ap.id)
        if not self._expose_combat or not eligible:
            # No eligible attackers OR combat not exposed — let the
            # tactical agent (or atomic-empty default) handle it and
            # enter the step's priority window directly.
            chosen = (
                ap.agent.declare_attackers(self._state, ap.id, eligible)
                if eligible else []
            )
            apply_action(
                self._state, ap.id,
                DeclareAttackers(attacker_ids=frozenset(chosen)),
            )
            if self._state.is_over():
                return
            self._enter_priority_for_current_step()
            return

        # Switch into the attacker-decision phase. Sort for clone
        # determinism.
        self._combat_phase = _COMBAT_ATTACKERS
        self._pending_attackers = sorted(o.id for o in eligible)
        self._chosen_attackers = []

    def _enter_declare_blockers(self) -> None:
        """Set up DECLARE_BLOCKERS step. With combat exposed AND
        there's something to block, switch into the per-blocker
        decision phase. Otherwise delegate / no-op."""
        attackers = [
            o for o in self._state.battlefield if o.attacking
        ]
        if not attackers:
            # No attackers — no blocking phase, no engine action,
            # just enter the step's priority window.
            self._enter_priority_for_current_step()
            return

        defender_idx = _next_alive(
            self._state,
            (self._state.active_player_idx + 1)
            % len(self._state.players),
        )
        defender = self._state.players[defender_idx]
        eligible = eligible_blockers(self._state, defender.id)

        if not self._expose_combat or not eligible:
            # Combat not exposed OR nothing eligible to block —
            # tactical assigns and we enter priority.
            assignments = (
                defender.agent.declare_blockers(
                    self._state, defender.id, eligible, attackers
                ) if eligible else {}
            )
            apply_action(
                self._state, defender.id,
                DeclareBlockers(
                    assignments=tuple(assignments.items())
                ),
            )
            if self._state.is_over():
                return
            self._enter_priority_for_current_step()
            return

        self._combat_phase = _COMBAT_BLOCKERS
        self._combat_defender_id = defender.id
        self._pending_blockers = sorted(o.id for o in eligible)
        self._chosen_blocks = {}

    # ---------------------------------------------------------------
    # Debug / test helpers
    # ---------------------------------------------------------------

    @property
    def recent_actions(self) -> tuple[ActionEvent, ...]:
        """The last ``DEFAULT_ACTION_HISTORY_N`` strategically-relevant
        decisions applied on this adapter's line (oldest first), as
        ``ActionEvent``s ready for ``tokenize_state``. Inherited by
        ``clone()`` (and so by every MCTS child and PIMC / ISMCTS
        determinized world). See ``mtg_bot.search.action_history``."""
        return self._recent_actions

    @property
    def engine_state(self) -> EngineGameState:
        """The wrapped engine state. MTG-aware search code (PIMC, the
        determinizer) reaches through this to operate on the engine
        directly; the rest of search should stay on the protocol
        surface. Mutating it bypasses adapter bookkeeping (priority
        counter, combat phase, resolution-pause snapshot,
        legal_actions cache) and is unsupported.

        Callers that DO need to mutate the engine state directly (the
        ISMCTS / PIMC determinizer being the canonical example) must
        either (a) only do so on a freshly-cloned adapter whose cache
        was empty to begin with — see ``clone()`` — or (b) call
        ``_invalidate_legal_actions_cache()`` after the mutation to
        drop any stale entries.
        """
        return self._state

    @property
    def combat_phase(self) -> Optional[str]:
        """Current combat sub-phase, or None if the adapter is at a
        priority moment. Useful for tests / introspection."""
        return self._combat_phase
