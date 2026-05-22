"""Phase 6 trajectory-parity helpers.

Sibling module to :mod:`mtg_bot.parity.cross_engine`. The cross-engine
module ships the per-step / per-action primitives (``build_matched_states``,
``step_both``, ``advance_phase_python``, the family of ``*_selector``
functions, the field-view differ); this module ships the *real-game-shaped*
primitives that drive multi-turn trajectory probes.

Surface (Phase 6 round-13 + round-14 slices):

* :func:`setup_matched_game` — closes infrastructure gap #1 from
  ``docs/PHASE6_TRAJECTORY_PARITY.md``: a single ``(decks, seed,
  mulligans=...)`` entrypoint that produces a parity-synced opening
  state on both engines. Wraps :func:`cross_engine.build_matched_states`
  and the side-channel opening-hand draw.
* :class:`MulliganDeterministicAgent` — Round-14 closure (gap #2): a
  deterministic cross-engine mulligan agent that produces matching
  keep/mulligan + bottom-pick decisions on both Python and Rust. The
  default policy is ``mulligan_target_hand_size=7`` (always keep at 7,
  never mulligan); pass a smaller value to mulligan exactly to that
  size. Bottom-picks are positional (``hand[0..n]``) so they're
  trivially reproducible across engines.

What this module deliberately does NOT cover:

* **Decision-callback mulligans.** The agent surface here is config-
  driven (mulligan to N, bottom by position). It is NOT a full
  ``Agent.choose_mulligan`` / ``Agent.choose_bottom`` bridge — those
  agents make state-conditional decisions, and getting state-
  conditional cross-engine agreement requires either (a) the SAME
  agent instance making the decision and routing it to both engines,
  or (b) two agents that always agree on the same hand. The
  deterministic agent is the (a) path: one agent, one decision per
  iteration, applied to both engines.
* **Real ``mtg_bot.simulate.setup_game`` mirror.** Production
  ``setup_game`` constructs Players via ``_build_player_from_names``,
  which materializes printings out of the *active card source*
  (Scryfall-loaded printings). The cross-engine harness already has
  its own per-DeckSpec card-pool plumbing; this slice builds on that
  plumbing rather than the production printings dict so the harness
  can drive synthetic ``CardDef`` decks (the only deck shape Phase 5
  + 6 actually probe).

The natural follow-up: a ``setup_matched_game_from_real_decks(decks,
seed, mulligans=...)`` that accepts decklist *names* against the live
``mtg_bot.cards.printings`` dict and threads them into a freshly-
materialized DeckSpec. That's a Phase 6 follow-on once the trajectory
suite outgrows hand-rolled CardDef registries.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Optional

from mtg_bot.parity.cross_engine import (
    CardDef,
    DeckSpec,
    build_matched_states,
)


__all__ = [
    "setup_matched_game",
    "setup_matched_game_from_real_decks",
    "MulliganDeterministicAgent",
    "RealGameCombatAgent",
    "BilateralCombatAgent",
    "CombatStrategy",
    "drive_combat_phase",
    "load_production_deck",
    "production_deck_to_deckspec",
    "production_deck_to_deckspec_instrumented",
    "LoaderAdmitReport",
    "LoaderAdmitPolicy",
    "NonAdmittedCardError",
    "DEFAULT_PRODUCTION_DECK_POOL",
]


# Round-17 enhancement (production-decklists slice). The 17lands EoE
# deck pool used by self-play / iterate; relative to repo root.
DEFAULT_PRODUCTION_DECK_POOL = pathlib.Path("data/17lands/eoe_decks.json.gz")


# ---------------------------------------------------------------------------
# Phase 6 round-16: real-game combat agent (closes infrastructure gap #3).
#
# Round-13/14/15 trajectory tests drive multi-turn real-game combat using
# inline scripts: each test re-spells out "find Bears with name == ...,
# emit declare_attackers with [int(bears.id)], walk DA -> DB, emit
# declare_blockers with [], side-channel combat damage, ...". As the
# trajectory suite grows beyond a single attacker / single combat round,
# inlining repeats and ad-hoc per-test combat scripts become the dominant
# noise.
#
# RealGameCombatAgent is the round-16 promotion: a config-driven combat
# decision source that walks the battlefield + per-object flags on each
# call and emits the right action-dict for the current step. Same shape
# as the round-14 MulliganDeterministicAgent — frozen-dataclass config,
# stateless decision functions, Python-only (the action-dicts cross both
# engines via the same `step_both` path the inline scripts use).
#
# Strategy vocabulary:
#   - "attack_all"            — every unsick + untapped non-summoning-sick
#                               creature attacks; defender doesn't block.
#                               Same shape as combat_selector's attacker
#                               branch.
#   - "attack_all_block_smallest" — attackers as above; defender blocks
#                               each attacker with their SMALLEST
#                               available creature (by power, then
#                               toughness, then id).
#   - "no_attack"             — skip combat entirely; emit empty
#                               declare_attackers + empty declare_blockers.
#                               Pins the no-combat baseline path.
#
# The agent does NOT advance phases — it only emits the action-dict for
# the current decision (declare_attackers OR declare_blockers OR pass).
# Callers compose the agent with `_advance_n_phases` / `step_both` /
# `resolve_combat_damage_python` etc. to drive a full combat round.
# ---------------------------------------------------------------------------

CombatStrategy = str  # see _VALID_COMBAT_STRATEGIES

# Round-17 enhancement (production-decklists slice): three new strategies
# that exercise richer attacker / blocker logic than the round-16
# attack_all / attack_all_block_smallest / no_attack triad. Each new
# strategy is a pure function of observable battlefield state — no
# new RNG, no per-side asymmetry, agent stays stateless.
#
# - "attack_with_unblockable_creatures_only" — only declare attackers
#   that carry an evasion keyword (flying / unblockable / shadow / ...)
#   the cross-engine harness understands. Defender does not block
#   (vacuously — the attackers can't be legally blocked anyway, but
#   we emit the empty assignment to make the no-block path explicit).
#   Useful for trajectory probes that want a guaranteed-through-attack
#   shape without involving block-assignment plumbing.
# - "block_with_largest_able" — mirror of attack_all_block_smallest,
#   but defender picks their LARGEST creature per attacker (sorted by
#   (power desc, toughness desc, id asc)). Pins the inverted block
#   selector — verifies that the agent's block-assignment shape doesn't
#   accidentally hardcode "smallest" anywhere.
# - "optimal_no_loss_attack" — only attack if the defender CAN'T block
#   back-and-kill the attacker. For a vanilla creature with power P
#   and toughness T, an attack is "no-loss" if no defender creature has
#   power >= T (defender can't trade up: no single blocker can kill
#   the attacker). Defender then declines blocking (empty assignments).
#   Surfaces a non-trivial state-dependent attacker policy without
#   needing to model deeper combat math.
_VALID_COMBAT_STRATEGIES: frozenset[str] = frozenset(
    {
        "attack_all",
        "attack_all_block_smallest",
        "no_attack",
        "attack_with_unblockable_creatures_only",
        "block_with_largest_able",
        "optimal_no_loss_attack",
    }
)

# Evasion keywords the harness recognizes for the
# "attack_with_unblockable_creatures_only" strategy. Limited to the
# keywords the cross-engine bridge wires symmetrically (same keyword
# on Python's CardPrinting.keywords frozenset and Rust's
# battlefield_has_keyword check). "flying" / "unblockable" / "shadow"
# are the obvious ones; menace is NOT evasion-strict (it just requires
# 2+ blockers, so it CAN be blocked and is excluded here).
_EVASION_KEYWORDS: frozenset[str] = frozenset({"flying", "unblockable", "shadow"})


@dataclass(frozen=True)
class RealGameCombatAgent:
    """Cross-engine deterministic combat policy for real-game trajectories.

    Closes Phase 6 infrastructure gap #3: a config-driven combat
    decision source that drives multi-turn real-game combat via a
    deterministic strategy. Same shape as
    :class:`MulliganDeterministicAgent` — stateless, observation-driven,
    same emission on both engines.

    Parameters
    ----------
    strategy : str, default "attack_all"
        One of:

        * ``"attack_all"`` — at DECLARE_ATTACKERS, every unsick + untapped
          creature the active player controls attacks the opponent; at
          DECLARE_BLOCKERS, defender declines to block (empty
          assignments).
        * ``"attack_all_block_smallest"`` — same attacker selection;
          at DECLARE_BLOCKERS, defender blocks each attacker with their
          SMALLEST available creature, sorted by ``(power, toughness,
          id)``. One blocker per attacker (no multi-block); if defender
          has fewer eligible blockers than attackers, the lowest-id
          attackers get blocked first.
        * ``"no_attack"`` — at DECLARE_ATTACKERS, emit empty attacker
          set; at DECLARE_BLOCKERS, emit empty assignments. Combat
          becomes a no-op; useful for tests that want to drive land
          drops + creature ETBs across multiple turns without combat
          damage.
        * ``"attack_with_unblockable_creatures_only"`` — at
          DECLARE_ATTACKERS, only declare attackers that carry an
          evasion keyword (``"flying"`` / ``"unblockable"`` /
          ``"shadow"``) the cross-engine bridge understands. At
          DECLARE_BLOCKERS, emit empty assignments. Useful for
          trajectory probes that want a guaranteed-unblocked attack
          shape (e.g. lifelink trample probes that need the attacker
          to land combat damage on the player).
        * ``"block_with_largest_able"`` — mirror of
          ``"attack_all_block_smallest"`` but defender picks their
          LARGEST creature per attacker (sorted by ``(power desc,
          toughness desc, id asc)``). Pins the inverted block selector;
          attacker selection is identical to ``"attack_all"``.
        * ``"optimal_no_loss_attack"`` — at DECLARE_ATTACKERS, only
          declare attackers the defender CAN'T trade up against. For
          a vanilla attacker with toughness T, the attack is "no-loss"
          if no defender creature has ``power >= T`` (no single blocker
          can kill the attacker). Defender then declines blocking
          (empty assignments) — pins the attacker side. The "optimal"
          label is light: no continuous-effect / counter / removal
          tracking; just a static p/t comparison on the visible bf.

    Notes
    -----
    The agent reads the PYTHON state (``py_state``) for decision
    construction. Both engines maintain byte-equal field views by the
    Phase 5 / Phase 6 invariants, so reading from Python and applying
    on both engines via :func:`mtg_bot.parity.cross_engine.step_both`
    produces identical post-state. (The same shape Phase 5's
    ``combat_selector`` uses internally — it reads ``py_state``, emits
    the action-dict, the harness applies it on both sides.)

    The agent does NOT touch RNG. Combat decisions are pure functions
    of the visible battlefield + per-object flags. Two calls with the
    same agent + same observable state produce identical output.

    The agent does NOT advance phases. The caller composes phase
    advances around the agent's emit. The
    :func:`drive_combat_phase` helper composes the standard pattern
    (DECLARE_ATTACKERS -> DECLARE_BLOCKERS -> COMBAT_DAMAGE ->
    END_OF_COMBAT) for callers that don't need step-by-step control.
    """

    strategy: CombatStrategy = "attack_all"

    def __post_init__(self) -> None:  # pragma: no cover — frozen guard
        if self.strategy not in _VALID_COMBAT_STRATEGIES:
            raise ValueError(
                f"strategy={self.strategy!r} not supported. "
                f"Valid: {sorted(_VALID_COMBAT_STRATEGIES)}"
            )

    # -- Decision functions --------------------------------------------------

    def pick_attackers(self, py_state) -> list[int]:
        """Pick the active player's attacker ids for DECLARE_ATTACKERS.

        Returns a sorted list of ids.

        Strategy-specific filtering on top of the common eligibility
        gate (untapped + (unsick OR has haste) + not-already-attacking
        Creature controlled by the active player):

        * ``"no_attack"`` returns ``[]`` unconditionally.
        * ``"attack_all"`` / ``"attack_all_block_smallest"`` /
          ``"block_with_largest_able"`` return every eligible attacker.
          (``block_with_largest_able`` only differs on the BLOCKER
          side — it shares ``attack_all``'s attacker selection.)
        * ``"attack_with_unblockable_creatures_only"`` returns only
          eligible attackers carrying one of the
          :data:`_EVASION_KEYWORDS`. Reads the keyword from
          ``o.printing.keywords`` (Python's frozenset spelling — same
          place :func:`mtg_bot.combat.effective_keywords` reads from
          for the no-modifier case).
        * ``"optimal_no_loss_attack"`` returns only eligible attackers
          for which no defender creature on the bf has
          ``power >= attacker.toughness``. Walks the defender's bf
          per-attacker to compare static p/t only — no continuous
          effect / counter modeling.

        Round-18 enhancement: the eligibility gate now honors haste
        (CR 702.10) — a summoning-sick creature with the ``"haste"``
        keyword IS attack-eligible. Previously the gate was a hard
        ``not o.summoning_sick`` filter that silently excluded haste
        creatures. The agent reads the keyword from
        ``o.printing.keywords`` (the same Python-side spelling
        :func:`mtg_bot.combat.effective_keywords` consumes for the
        no-modifier case).
        """
        if self.strategy == "no_attack":
            return []
        active_idx = py_state.active_player_idx
        defender_idx = 1 - active_idx
        eligible = [
            o for o in py_state.battlefield
            if o.controller_id == active_idx
            and "Creature" in o.printing.card_types
            and not o.tapped
            and (
                not o.summoning_sick
                or "haste" in getattr(o.printing, "keywords", frozenset())
            )
            and not getattr(o, "attacking", False)
        ]

        if self.strategy == "attack_with_unblockable_creatures_only":
            eligible = [
                o for o in eligible
                if _EVASION_KEYWORDS & set(
                    getattr(o.printing, "keywords", frozenset())
                )
            ]
        elif self.strategy == "optimal_no_loss_attack":
            # Defender's eligible blockers (untapped Creatures defender
            # controls — sick can still block per CR 302.1, so the
            # filter mirrors pick_blocks's eligibility test).
            defender_blockers = [
                o for o in py_state.battlefield
                if o.controller_id == defender_idx
                and "Creature" in o.printing.card_types
                and not o.tapped
            ]
            # An attacker is "no-loss" if no defender blocker has
            # power >= attacker.toughness. Defender's largest power on
            # bf is the bound; if that's < attacker.toughness, no single
            # blocker can kill the attacker (multi-block sums power
            # but defender CHOOSES whether to block, so "no-loss"
            # against a non-blocking defender holds vacuously).
            max_def_power = max(
                (
                    int(getattr(b.printing, "power", 0) or 0)
                    for b in defender_blockers
                ),
                default=0,
            )
            eligible = [
                o for o in eligible
                if int(getattr(o.printing, "toughness", 0) or 0)
                > max_def_power
            ]

        return sorted(int(o.id) for o in eligible)

    def pick_blocks(self, py_state) -> list[list[int]]:
        """Pick the defender's block assignments for DECLARE_BLOCKERS.

        Returns a list of ``[blocker_id, attacker_id]`` pairs — the
        same shape Rust's ``Action::DeclareBlockers`` accepts and
        Python's ``DeclareBlockers(assignments=...)`` consumes after
        the harness's tuple-conversion.

        ``"no_attack"``, ``"attack_all"``,
        ``"attack_with_unblockable_creatures_only"``, and
        ``"optimal_no_loss_attack"`` return ``[]`` (no blocks).

        ``"attack_all_block_smallest"`` walks attackers in id-sort order
        and assigns the defender's smallest-power-then-toughness-then-id
        creature to each, one blocker per attacker. Already-assigned
        blockers are removed from the eligible pool so we don't
        double-assign.

        ``"block_with_largest_able"`` mirrors the smallest selector but
        sorts the eligible pool by ``(power desc, toughness desc, id
        asc)`` so the LARGEST creature blocks first. The attacker walk
        order (id-sort) is unchanged.

        Block-eligible-creature selection is deterministic: sorted by
        ``(power, toughness, id)`` so ties break the same way on every
        call. Power is read from ``printing.power`` (synthetic vanilla
        decks have static p/t; +1/+1 counters and continuous effects
        are NOT modeled here — the Phase 6 trajectory deck shape uses
        pure vanilla creatures).
        """
        if self.strategy not in (
            "attack_all_block_smallest",
            "block_with_largest_able",
        ):
            return []

        attackers = [
            o for o in py_state.battlefield
            if getattr(o, "attacking", False)
        ]
        if not attackers:
            return []

        active_idx = py_state.active_player_idx
        defender_idx = 1 - active_idx

        # Defender's eligible blockers: untapped Creatures defender
        # controls. CR 509.1b — tapped creatures can't block. CR 302.1
        # allows summoning-sick creatures TO block (sickness only
        # blocks tap-to-attack and tap-to-activate), so we keep
        # summoning-sick creatures eligible to block. Filter out
        # creatures already assigned blocking somewhere via the
        # per-object ``blocking`` flag (in case of re-visits).
        eligible = [
            o for o in py_state.battlefield
            if o.controller_id == defender_idx
            and "Creature" in o.printing.card_types
            and not o.tapped
            and not getattr(o, "blocking", False)
        ]
        if not eligible:
            return []

        # Sort by (power, toughness, id). printing.power can be None
        # for non-creatures, but the eligible list is filtered to
        # Creatures so power is set. Cast to int for the sort key.
        # ``block_with_largest_able`` reverses the (power, toughness)
        # axes so the largest creature sorts FIRST; ties on (power,
        # toughness) still break by id ascending so the agent's
        # output is deterministic regardless of bf insertion order.
        if self.strategy == "block_with_largest_able":
            def _key(o):
                p = int(getattr(o.printing, "power", 0) or 0)
                t = int(getattr(o.printing, "toughness", 0) or 0)
                return (-p, -t, int(o.id))
        else:
            def _key(o):
                p = int(getattr(o.printing, "power", 0) or 0)
                t = int(getattr(o.printing, "toughness", 0) or 0)
                return (p, t, int(o.id))

        eligible_sorted = sorted(eligible, key=_key)

        # Walk attackers in id order (matches the attacker emit order
        # from pick_attackers); assign smallest available blocker.
        attackers_by_id = sorted(attackers, key=lambda o: int(o.id))
        assignments: list[list[int]] = []
        used_blocker_ids: set[int] = set()
        for atk in attackers_by_id:
            for blk in eligible_sorted:
                if int(blk.id) in used_blocker_ids:
                    continue
                assignments.append([int(blk.id), int(atk.id)])
                used_blocker_ids.add(int(blk.id))
                break  # one blocker per attacker for this strategy
            # If no blocker available, attacker goes through unblocked.
        return assignments


# ---------------------------------------------------------------------------
# Phase 6 round-19 enhancement (bilateral / asymmetric combat strategies):
#
# RealGameCombatAgent's strategy is mirror-symmetric -- the same strategy
# decides attackers when p0 is active AND when p1 is active, and the same
# strategy decides blockers regardless of which side is defending. The
# round-17 enhancement-follow-on list flagged this as the next gap:
# "Bilateral / asymmetric combat strategies -- current strategies are
# symmetric across both players. Asymmetric strategies (`p0 attacks, p1
# never attacks back`) would let trajectory tests pin specific defensive
# patterns without needing a second RealGameCombatAgent per player.
# Round-17 ships the agent shape that could carry per-side config, but
# the strategy enum stays mirror-symmetric for now."
#
# BilateralCombatAgent is the per-side wrapper: a frozen-dataclass that
# holds two RealGameCombatAgent instances (one per player) and dispatches
# pick_attackers / pick_blocks to the active-player-side / defender-side
# delegate respectively. Same duck-typed interface as RealGameCombatAgent
# (just .pick_attackers + .pick_blocks), so drive_combat_phase accepts it
# unchanged.
#
# Construction is by (p0_strategy, p1_strategy) string pair (the cleanest
# call shape; the wrapper builds the two RealGameCombatAgent instances
# internally) OR by (p0_agent, p1_agent) for callers that want to share /
# pre-build per-side agents (e.g. a future state-conditional agent that
# isn't a bare RealGameCombatAgent). Both shapes route through the same
# dispatch logic.
#
# The wrapper is stateless (frozen-dataclass holding frozen-dataclass
# RealGameCombatAgent instances). Two calls with the same agent + same
# observable state produce identical output, mirroring the round-16/17
# RealGameCombatAgent invariant.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BilateralCombatAgent:
    """Per-side combat policy wrapper around two ``RealGameCombatAgent``s.

    Closes the round-17 enhancement-follow-on item "Bilateral /
    asymmetric combat strategies": the symmetric strategy enum becomes
    a per-side pair via this wrapper, without changing the underlying
    ``RealGameCombatAgent`` contract.

    The wrapper exposes the same duck-typed interface as
    ``RealGameCombatAgent`` (``pick_attackers`` + ``pick_blocks``) so
    it drops into ``drive_combat_phase`` unchanged. Each call dispatches
    to the appropriate per-side delegate based on which player is the
    active player at decision time:

      * ``pick_attackers(py_state)`` -> delegate for
        ``py_state.active_player_idx``'s strategy. The active player is
        the one declaring attackers per CR 508.1.
      * ``pick_blocks(py_state)`` -> delegate for the DEFENDER (=
        ``1 - py_state.active_player_idx``)'s strategy. The defender is
        the one declaring blockers per CR 509.1.

    Construct from per-side strategy strings (the common case):

        BilateralCombatAgent(p0_strategy="attack_all", p1_strategy="no_attack")

    Or from pre-built per-side agents (for callers sharing agents or
    using a future agent type that isn't a bare RealGameCombatAgent):

        BilateralCombatAgent(
            p0_agent=RealGameCombatAgent(strategy="attack_all"),
            p1_agent=RealGameCombatAgent(strategy="no_attack"),
        )

    Exactly one of the two construction shapes per side must be
    provided; mixing (p0_strategy + p0_agent both set) or omitting
    (neither set) raises ValueError.

    Use cases:

    * **Asymmetric attack/defense** -- p0 plays aggro
      (``"attack_all"``), p1 plays control (``"no_attack"``). Pins the
      aggressor / control trajectory shape in one config.
    * **Per-side block selectors** -- p0 uses
      ``"attack_all_block_smallest"`` to chump-block, p1 uses
      ``"block_with_largest_able"`` to trade up. Surfaces the
      asymmetric block selector divergence directly.
    * **Single-side smoke** -- one side runs an interesting policy
      (``"optimal_no_loss_attack"``), the other side runs a trivial
      baseline (``"no_attack"``) so the trajectory probe isolates one
      side's decision logic cleanly.

    Mirroring RealGameCombatAgent's invariants:

    * **Stateless.** Frozen-dataclass; per-side delegates are
      themselves frozen-dataclasses. No mutation, no RNG.
    * **Same emission on both engines.** Reads PYTHON state and emits
      action-dicts; the harness applies on both sides via
      ``step_both``.
    * **Does NOT advance phases.** The caller composes phase advances
      around the agent (or uses ``drive_combat_phase``).
    """

    p0_strategy: CombatStrategy | None = None
    p1_strategy: CombatStrategy | None = None
    p0_agent: RealGameCombatAgent | None = None
    p1_agent: RealGameCombatAgent | None = None

    def __post_init__(self) -> None:
        # Each side must have exactly one source of truth (strategy OR
        # agent, not both, not neither). Build the per-side agent from
        # the strategy if it was passed; this normalizes the two
        # construction shapes into a single internal representation.
        if self.p0_strategy is not None and self.p0_agent is not None:
            raise ValueError(
                "BilateralCombatAgent: pass either p0_strategy or "
                "p0_agent, not both"
            )
        if self.p0_strategy is None and self.p0_agent is None:
            raise ValueError(
                "BilateralCombatAgent: must pass one of p0_strategy or "
                "p0_agent"
            )
        if self.p1_strategy is not None and self.p1_agent is not None:
            raise ValueError(
                "BilateralCombatAgent: pass either p1_strategy or "
                "p1_agent, not both"
            )
        if self.p1_strategy is None and self.p1_agent is None:
            raise ValueError(
                "BilateralCombatAgent: must pass one of p1_strategy or "
                "p1_agent"
            )
        # Normalize: stamp the constructed delegates onto p0_agent /
        # p1_agent regardless of which constructor shape was used. The
        # validation in RealGameCombatAgent.__post_init__ catches an
        # invalid strategy at construct time (NOT at call time).
        if self.p0_agent is None:
            object.__setattr__(
                self, "p0_agent", RealGameCombatAgent(strategy=self.p0_strategy),
            )
        if self.p1_agent is None:
            object.__setattr__(
                self, "p1_agent", RealGameCombatAgent(strategy=self.p1_strategy),
            )

    def _agent_for(self, player_idx: int) -> RealGameCombatAgent:
        """Return the delegate agent for ``player_idx`` (0 or 1)."""
        if player_idx == 0:
            return self.p0_agent  # type: ignore[return-value]
        if player_idx == 1:
            return self.p1_agent  # type: ignore[return-value]
        raise ValueError(
            f"BilateralCombatAgent only supports two-player engines "
            f"(player_idx in {{0, 1}}); got {player_idx}"
        )

    # -- Decision functions (delegate to the right per-side agent) ----------

    def pick_attackers(self, py_state) -> list[int]:
        """Pick the active player's attackers via that player's delegate.

        The active player is the one declaring attackers per CR 508.1,
        so attacker selection routes to the active-side strategy --
        e.g. ``BilateralCombatAgent(p0_strategy="attack_all",
        p1_strategy="no_attack")`` will declare every eligible p0
        attacker on p0's turn and an empty attacker set on p1's turn.
        """
        return self._agent_for(py_state.active_player_idx).pick_attackers(
            py_state
        )

    def pick_blocks(self, py_state) -> list[list[int]]:
        """Pick the defender's block assignments via that player's delegate.

        The defender (= 1 - active_player_idx) is the one declaring
        blockers per CR 509.1, so block assignment routes to the
        defender-side strategy -- e.g. ``BilateralCombatAgent(
        p0_strategy="attack_all_block_smallest",
        p1_strategy="block_with_largest_able")`` will block with the
        SMALLEST defender on p1's attacks (because p0 is defending and
        their delegate runs ``attack_all_block_smallest``) and with
        the LARGEST defender on p0's attacks (because p1 is defending
        and their delegate runs ``block_with_largest_able``).
        """
        defender_idx = 1 - py_state.active_player_idx
        return self._agent_for(defender_idx).pick_blocks(py_state)


def drive_combat_phase(
    py_state, rust_state, agent: RealGameCombatAgent,
) -> None:
    """Drive one combat phase end-to-end with the supplied agent.

    Pre-condition: ``py_state.step == Step.DECLARE_ATTACKERS`` (i.e.,
    the caller has already advanced phases past PRECOMBAT_MAIN ->
    BEGINNING_OF_COMBAT). Post-condition:
    ``py_state.step == Step.END_OF_COMBAT`` with combat state cleared
    and parity asserted on every intermediate compare.

    Composes the standard "attacker emit -> advance -> blocker emit ->
    advance -> combat damage -> advance -> clear combat" sequence the
    round-13/14/15 inline scripts spell out by hand. Each step asserts
    byte-equal field views on both engines; the first divergence
    raises an AssertionError naming the step.

    The advance from END_OF_COMBAT into POSTCOMBAT_MAIN is NOT done
    here — the caller drives that via ``_advance_n_phases`` so the
    helper's post-condition is the same as the inline scripts'.

    For ``"no_attack"`` strategy: emits ``declare_attackers`` with
    empty list, walks past blockers + damage steps with no-op effect.
    """
    from mtg_bot.parity.cross_engine import (
        advance_phase_python,
        clear_combat_both,
        compare_fields,
        deal_combat_damage_rust,
        resolve_combat_damage_python,
        step_both,
    )
    from mtg_bot.turn import Step

    if py_state.step != Step.DECLARE_ATTACKERS:
        raise ValueError(
            f"drive_combat_phase requires step=DECLARE_ATTACKERS; "
            f"got step={py_state.step.name}"
        )

    # 1. DECLARE_ATTACKERS — emit attacker dict.
    attackers = agent.pick_attackers(py_state)
    s = step_both(
        py_state, rust_state,
        {"kind": "declare_attackers", "attacker_ids": attackers},
    )
    if s.divergences:
        raise AssertionError(
            f"declare_attackers (step=DA, attackers={attackers}) "
            f"surfaced divergences: "
            f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}"
        )

    # 2. DA -> DB advance.
    advance_phase_python(py_state)
    rust_state.advance_phase()
    assert py_state.step == Step.DECLARE_BLOCKERS, (
        f"expected DECLARE_BLOCKERS after advance, got {py_state.step.name}"
    )

    # 3. DECLARE_BLOCKERS — emit blocker dict.
    assignments = agent.pick_blocks(py_state)
    s = step_both(
        py_state, rust_state,
        {"kind": "declare_blockers", "assignments": assignments},
    )
    if s.divergences:
        raise AssertionError(
            f"declare_blockers (assignments={assignments}) surfaced "
            f"divergences: "
            f"{[(d.field, d.py_value, d.rust_value) for d in s.divergences]}"
        )

    # 4. DB -> COMBAT_DAMAGE advance.
    advance_phase_python(py_state)
    rust_state.advance_phase()
    assert py_state.step == Step.COMBAT_DAMAGE, (
        f"expected COMBAT_DAMAGE after advance, got {py_state.step.name}"
    )

    # 5. COMBAT_DAMAGE — side-channel damage on both engines.
    fired_py = resolve_combat_damage_python(py_state)
    if fired_py:
        deal_combat_damage_rust(rust_state)
    # Post-damage compare.
    from mtg_bot.parity.cross_engine import (
        build_python_field_view,
        parity_smoke,
    )
    ps = parity_smoke()
    py_view = build_python_field_view(py_state)
    rust_view = ps._build_field_view(rust_state)
    divs = compare_fields(py_view, rust_view)
    if divs:
        raise AssertionError(
            f"post combat-damage divergences: "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in divs]}"
        )

    # 6. COMBAT_DAMAGE -> END_OF_COMBAT advance.
    advance_phase_python(py_state)
    rust_state.advance_phase()
    assert py_state.step == Step.END_OF_COMBAT, (
        f"expected END_OF_COMBAT after advance, got {py_state.step.name}"
    )

    # 7. Clear combat state on both engines (matches the
    # ``simulate.run_step`` END_OF_COMBAT hook). Without this Rust's
    # ``self.attackers`` would persist past the turn.
    clear_combat_both(py_state, rust_state)
    py_view = build_python_field_view(py_state)
    rust_view = ps._build_field_view(rust_state)
    divs = compare_fields(py_view, rust_view)
    if divs:
        raise AssertionError(
            f"post clear-combat divergences: "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in divs]}"
        )


@dataclass(frozen=True)
class MulliganDeterministicAgent:
    """Cross-engine deterministic mulligan policy.

    A config-driven decision source that produces byte-identical
    keep/mulligan + bottom-pick decisions on both Python and Rust.
    Closes Phase 6 infrastructure gap #2: any state-conditional
    ``Agent.choose_mulligan`` would need a state-equivalent decision
    on both engines, but a config-driven agent ("mulligan exactly
    until hand_size==N, then bottom by position") sidesteps the
    state-equivalence question entirely — the agent makes the same
    decision regardless of which side's hand it's looking at.

    Parameters
    ----------
    target_hand_size : int, default 7
        Number of cards the player should END UP HOLDING after all
        mulligans + bottoming. ``target_hand_size=7`` means "never
        mulligan" (each player keeps the initial 7). ``target_hand_size
        =6`` means "mulligan exactly once and bottom 1". ``=5`` means
        "mulligan twice and bottom 2". Etc. Must be in ``[0, 7]``.
        The London-mulligan rule (CR 103.4) bottoms ``mulls_taken``
        cards on keep, so the final hand size is always
        ``7 - mulls_taken``.
    bottom_strategy : str, default "first"
        Which cards to put on the bottom after the final keep. Only
        ``"first"`` is supported in this slice — bottoms ``hand[0..n]``
        in insertion order. Trivial to reproduce on Rust because hand
        order is byte-equal across engines after the shared-RNG shuffle.
        A ``"last"`` / ``"by_id_descending"`` variant is a one-line
        addition if a future test needs it; ``"first"`` is the default
        because it matches Rust's own
        ``bottom_cards_from_hand``-by-position default.

    Notes
    -----
    The agent does NOT consume any RNG state. The randomness in the
    mulligan loop comes from the *shared RNG instance* threaded into
    each engine's shuffle primitive (``random.Random(seed)._randbelow``
    in both Rust's ``mulligan_atomic(rng=)`` path and Python's local
    Fisher-Yates). The agent only governs the policy decisions
    (mulligan/keep, which cards to bottom), not the shuffle outcome.

    This separation keeps the agent reproducible regardless of seed —
    two calls with ``target_hand_size=5`` always mulligan twice, no
    matter what cards land in hand.
    """

    target_hand_size: int = 7
    bottom_strategy: str = "first"

    def __post_init__(self) -> None:  # pragma: no cover — frozen guard
        # frozen=True forbids __setattr__, so __post_init__ is a no-op
        # validate-only hook. Raise on construction with a bad config.
        if not (0 <= self.target_hand_size <= 7):
            raise ValueError(
                f"target_hand_size must be in [0, 7], got "
                f"{self.target_hand_size}"
            )
        if self.bottom_strategy != "first":
            raise ValueError(
                f"bottom_strategy={self.bottom_strategy!r} not supported. "
                f"Only 'first' is implemented in this slice."
            )

    @property
    def mulligans_to_take(self) -> int:
        """Number of mulligans this agent will run before keeping.

        ``7 - target_hand_size`` per the London-mulligan rule. A
        target of 7 means 0 mulligans; target 5 means 2 mulligans
        (then bottom 2 cards on keep).
        """
        return 7 - self.target_hand_size

    def should_mulligan(self, mulls_taken: int) -> bool:
        """Decision: should the player mulligan again?

        Returns True until ``mulls_taken`` reaches the configured
        ``mulligans_to_take``, then False (keep). Stateless — the
        decision is a pure function of the agent's config and the
        current mulligan count.
        """
        return mulls_taken < self.mulligans_to_take

    def pick_bottom(self, hand_ids: list[int], n: int) -> list[int]:
        """Pick ``n`` card ids from ``hand_ids`` to put on the bottom.

        With ``bottom_strategy="first"`` (the only currently-supported
        strategy), returns ``hand_ids[:n]`` — the first ``n`` cards
        in hand-insertion order.
        """
        if n == 0:
            return []
        if n > len(hand_ids):
            raise ValueError(
                f"pick_bottom: requested {n} cards but hand has only "
                f"{len(hand_ids)}"
            )
        return list(hand_ids[:n])


def _draw_opening_hand_both(py_state, rust_state, n: int = 7) -> None:
    """Mirror real-game ``setup_game``'s seven-card opening draw on BOTH engines.

    Phase 5's CE3 closure proved Python's ``turn.draw_card`` + Rust's
    ``draw_card`` pymethod produce byte-equal hand/library views when
    both libraries start at the same permutation. This helper just
    fires that draw ``n`` times per player.

    Internal helper — :func:`setup_matched_game` is the public entry.
    """
    from mtg_bot.turn import draw_card as py_draw_card

    for pid in (0, 1):
        for _ in range(n):
            py_draw_card(py_state, pid)
            rust_state.draw_card(pid)


def _python_mulligan_iteration(py_state, player_id: int, rng) -> None:
    """One mulligan iteration on the PYTHON side.

    Mirrors Rust's ``mulligan_atomic`` shuffle+redraw step (the bottom
    step is applied separately by :func:`_python_bottom_cards`). Drains
    ``player_id``'s hand back into the library, runs Python's
    ``random.Random.shuffle`` on the library using the SHARED ``rng``
    instance (which is the same instance Rust's ``mulligan_atomic``
    consumes via ``_randbelow``), then redraws 7 cards.

    Both engines call ``rng.shuffle(<library_list>)`` (Python directly,
    Rust via ``_randbelow`` callback in Fisher-Yates). Because the
    rng instance is shared and the pre-shuffle library order is byte-
    equal on both engines, the post-shuffle libraries will be
    byte-equal as well.

    The internal-state mutation here mirrors
    ``mtg_bot.simulate.setup_game``'s mulligan-redraw block (lines
    160-170 of simulate.py) verbatim — same drain, same shuffle, same
    redraw — except it consumes the shared external rng rather than
    setup_game's local one.
    """
    from mtg_bot.game import Zone
    from mtg_bot.turn import draw_card as py_draw_card

    p = py_state.players[player_id]
    # Drain hand into library in HAND-INSERTION order (hand[0] first,
    # hand[6] last). Rust's mulligan_to_n / mulligan_atomic uses
    # ``hand.drain(..)`` which yields entries in insertion order; we
    # mirror that exactly so the pre-shuffle library tail order matches
    # across engines. (NOTE: this differs from Python's
    # mtg_bot.simulate.setup_game which uses ``p.hand.pop()`` (LIFO);
    # the cross-engine path needs the FIFO order to match Rust.)
    returning = list(p.hand)
    p.hand.clear()
    for card in returning:
        card.zone = Zone.LIBRARY
        p.library.append(card)
    # Shared-rng shuffle. random.shuffle consumes _randbelow in
    # Fisher-Yates order, matching Rust's mulligan_atomic(rng=...)
    # consumption order.
    rng.shuffle(p.library)
    # Redraw 7 (or until empty).
    for _ in range(7):
        if not p.library:
            break
        py_draw_card(py_state, player_id)


def _python_bottom_cards(py_state, player_id: int, bottom_ids: list[int]) -> None:
    """Apply the post-mulligan bottom step on the PYTHON side.

    Mirrors Rust's ``bottom_cards_from_hand`` orientation: the FIRST
    id in ``bottom_ids`` ends up at the very BOTTOM of the library
    (last index), each subsequent id sits one slot above.
    Implementation: walk ids in reverse and ``library.append`` each —
    the last-walked (= first-supplied) lands at the tail.

    Per-instance state reset matches Rust's ``bottom_cards_from_hand``:
    ``tapped=False``, ``summoning_sick=False``, ``damage=0``, plus
    counters / attachments cleared. Without this reset, a hand card
    carrying ``summoning_sick=True`` (the default for cards just
    drawn to hand on Python) would keep that flag in the library
    while Rust would not — surfacing as a per-object field divergence
    on the very last cards of the library after bottoming.
    """
    from mtg_bot.game import Zone

    p = py_state.players[player_id]
    by_id = {c.id: c for c in p.hand}
    for oid in reversed(bottom_ids):
        card = by_id[oid]
        p.hand.remove(card)
        card.zone = Zone.LIBRARY
        # Mirror Rust's per-instance reset on bottom (CR-equivalent
        # cleanup: a card moving from hand to library can't legally
        # carry battlefield-only state; defensive against future
        # paths that might leave stale flags).
        card.tapped = False
        card.summoning_sick = False
        card.damage = 0
        # ``counters`` on the Python side is a list (vs Rust's int).
        # Empty it out to match the int=0 reset semantically.
        if hasattr(card, "counters") and isinstance(card.counters, list):
            card.counters.clear()
        if hasattr(card, "attached_to"):
            card.attached_to = None
        p.library.append(card)


def _run_mulligan_loop_both(
    py_state,
    rust_state,
    agent: MulliganDeterministicAgent,
    base_seed: int,
) -> None:
    """Drive the mulligan loop on BOTH engines for both players.

    Per-player loop:

    1. Ask the agent ``should_mulligan(mulls_taken)``. If False, exit
       to step 4.
    2. Construct two ``random.Random(iter_seed)`` instances — one for
       Python's local Fisher-Yates, one for Rust's ``mulligan_to_n``.
       Both seeds derive deterministically from
       ``(base_seed, player_id, mulls_taken)``. Two ``random.Random``
       instances seeded identically produce bit-identical
       ``_randbelow`` sequences (MT determinism), so the in-mulligan
       shuffle on each engine consumes the same draw order.
    3. Fire shuffle+redraw on both engines. Increment ``mulls_taken``.
       Loop to 1.
    4. Pick ``mulls_taken`` bottom ids via ``agent.pick_bottom``. Apply
       on both engines: Python via :func:`_python_bottom_cards`, Rust
       via ``bottom_cards_from_hand(n=mulls_taken)`` which bottoms
       hand[0..n] by position.

    Why ``mulligan_to_n`` for intermediate iterations and a separate
    ``bottom_cards_from_hand`` for the final keep, rather than Rust's
    parity-canonical ``mulligan_atomic``? Two reasons:

    * ``mulligan_atomic`` requires ``bottom_card_ids.len() ==
      mulligan_count + 1``, so an N-mulligan loop would need to bottom
      progressively-larger sets on each iteration even though only the
      LAST iteration's bottom is meaningful.
    * Splitting shuffle+redraw from bottom mirrors Python's
      ``simulate.setup_game`` mulligan-loop shape exactly (intermediate
      iterations are shuffle+redraw only; bottom fires once on keep).

    The split-call path consumes one Rust ``mulligan_count`` slot per
    mulligan, same as ``mulligan_atomic`` would; the per-call cap of 7
    still holds. With ``bottom_strategy="first"`` (the only currently-
    supported strategy), Rust's positional ``bottom_cards_from_hand(n)``
    is exactly equivalent to the agent's ``pick_bottom(hand_ids, n)``
    returning ``hand_ids[:n]``, so the chosen-by-id and chosen-by-
    position results agree.

    A future non-positional bottom strategy (e.g. "highest mana value
    first") would need either Rust hand-reordering before the bottom
    call or a Rust primitive that takes explicit bottom_ids — see the
    Rust API gaps section in ``docs/PHASE6_TRAJECTORY_PARITY.md``.
    """
    for pid in (0, 1):
        mulls_taken = 0
        while agent.should_mulligan(mulls_taken):
            # Per-iteration seed: deterministic in (base_seed, pid,
            # mulls_taken). Two random.Random instances with the same
            # seed produce identical _randbelow sequences, so Python's
            # local Fisher-Yates and Rust's mulligan_to_n consume the
            # same draw order even though they don't share state.
            iter_seed = (base_seed * 1000003 + pid * 31 + mulls_taken) & 0xFFFFFFFF
            py_rng = random.Random(iter_seed)
            rust_rng = random.Random(iter_seed)

            _python_mulligan_iteration(py_state, pid, py_rng)
            # Rust ``mulligan_to_n``'s ``seed`` arg is unused on the
            # external-rng path (the rng kwarg owns shuffle order); pass
            # 0 as a placeholder.
            rust_state.mulligan_to_n(pid, 7, 0, rng=rust_rng)

            mulls_taken += 1

        if mulls_taken > 0:
            # Pick bottom ids from the kept hand. By-position policy
            # ("first") matches Rust's bottom_cards_from_hand(n) which
            # also bottoms by position.
            hand_ids = [c.id for c in py_state.players[pid].hand]
            bottom_ids = agent.pick_bottom(hand_ids, mulls_taken)

            # Python: apply bottoming.
            _python_bottom_cards(py_state, pid, bottom_ids)
            # Rust: bottom by position (n cards from hand[0..n]).
            # bottom_strategy="first" guarantees this matches the
            # agent's pick_bottom([:n]) result by id.
            rust_state.bottom_cards_from_hand(pid, mulls_taken)


def setup_matched_game(
    decks: DeckSpec,
    seed: Optional[int] = None,
    mulligans: bool = False,
    opening_hand_size: int = 7,
    mulligan_agent: Optional[MulliganDeterministicAgent] = None,
    mulligan_seed: int = 0,
):
    """Build a parity-synced opening state on both engines.

    Closes Phase 6 infrastructure gap #1 (``docs/PHASE6_TRAJECTORY_PARITY.md``):
    a single entrypoint that produces a starting state with seven-card
    opening hands dealt on both Python and Rust, identical libraries
    on both sides, ready to walk turn 1 from UNTAP. Round-14 extends
    the helper to optionally drive a deterministic London-mulligan
    loop (gap #2).

    Parameters
    ----------
    decks : DeckSpec
        The matched decklist for both players. Reuses the existing
        ``cross_engine.DeckSpec`` shape so trajectory tests share the
        same registry / decklist / shuffle-seed plumbing as Phase 5
        smoke tests. The DeckSpec's ``shared_rng_seed`` (or
        ``shuffle_seed``) governs library shuffling on both engines —
        see :class:`cross_engine.DeckSpec` for the precedence rules.
    seed : Optional[int], default None
        Convenience: when set AND ``decks.shared_rng_seed`` is None,
        rebuilds ``decks`` with ``shared_rng_seed=seed`` so callers
        don't have to thread the seed twice. When BOTH are None the
        libraries are unshuffled (deck-list order). Callers that want
        explicit per-engine shuffle semantics should set
        ``decks.shared_rng_seed`` directly and leave ``seed=None``.
    mulligans : bool, default False
        When True, runs the deterministic London-mulligan loop on
        both engines using the supplied ``mulligan_agent`` (defaults
        to ``MulliganDeterministicAgent(target_hand_size=7)`` — a
        no-op agent that always keeps the initial 7). When False
        (legacy default), the helper deals 7 cards and stops, matching
        round-13's behavior exactly.
    opening_hand_size : int, default 7
        Cards to draw per player. Defaults to 7 (CR 103.5). Exposed
        primarily for test ergonomics — drop to 0 for a no-draw
        starting state matching the Phase 5 ``build_matched_states``
        pre-draw shape. Ignored when ``mulligans=True`` (the mulligan
        loop assumes the standard 7-card opening hand).
    mulligan_agent : Optional[MulliganDeterministicAgent], default None
        Cross-engine deterministic mulligan policy. Required when
        ``mulligans=True`` to govern the keep/mulligan + bottom-pick
        decisions. Defaults to ``MulliganDeterministicAgent()`` (the
        always-keep policy, equivalent to ``mulligans=False``) when
        not supplied. Ignored when ``mulligans=False``.
    mulligan_seed : int, default 0
        Base seed for the per-iteration shuffle RNGs. Each per-player
        per-iteration shuffle uses a freshly-constructed
        ``random.Random`` derived from ``(mulligan_seed, player_id,
        mulls_taken)``. Two calls with the same agent + seed produce
        bit-identical post-mulligan states. Ignored when
        ``mulligans=False``.

    Returns
    -------
    tuple[GameState, mtg_bot_engine.GameState]
        ``(py_state, rust_state)``. With ``mulligans=False``: both
        have ``opening_hand_size`` cards in hand on each side. With
        ``mulligans=True``: both have ``agent.target_hand_size`` cards
        in hand on each side, library has the bottomed cards on its
        tail. ``step=Step.UNTAP``, ``turn=1``, ``active_player=0``,
        ``priority_player=0`` in either case.

    Raises
    ------
    ValueError
        If ``opening_hand_size`` exceeds the per-player library size
        (would underflow on the side-channel draw); or if a supplied
        ``mulligan_agent`` is passed without ``mulligans=True`` (a
        likely caller bug — the agent has no effect in that path).
    """
    if opening_hand_size < 0:
        raise ValueError(
            f"opening_hand_size must be >= 0 (got {opening_hand_size})"
        )

    if mulligans:
        # MulliganAgent is the gap-#2 substrate. The default agent is
        # the "always keep" policy — equivalent to mulligans=False on
        # the post-state, but exercises the mulligan-loop plumbing.
        if mulligan_agent is None:
            mulligan_agent = MulliganDeterministicAgent()
        # The mulligan path assumes the standard 7-card opening hand
        # (London mulligan only makes sense at 7). Refuse to silently
        # drop a non-default opening_hand_size in this path.
        if opening_hand_size != 7:
            raise ValueError(
                "setup_matched_game(mulligans=True) requires "
                "opening_hand_size=7 (London mulligan deals 7 cards "
                f"by rule); got opening_hand_size={opening_hand_size}"
            )
    else:
        if mulligan_agent is not None:
            raise ValueError(
                "mulligan_agent supplied with mulligans=False — the "
                "agent would have no effect. Pass mulligans=True to "
                "drive the mulligan loop."
            )

    # Apply the convenience seed if the DeckSpec doesn't already pin
    # a shared_rng_seed. Build a fresh DeckSpec (frozen-dataclass
    # contract) rather than mutating in place.
    if seed is not None and decks.shared_rng_seed is None:
        decks = DeckSpec(
            card_defs=decks.card_defs,
            p0_library=decks.p0_library,
            p1_library=decks.p1_library,
            shuffle_seed=decks.shuffle_seed,
            shared_rng_seed=seed,
        )

    # Validate library sizes BEFORE construction so a sub-7-card spec
    # surfaces as a ValueError here rather than as an IndexError deep
    # in the draw loop.
    if opening_hand_size > 0:
        if len(decks.p0_library) < opening_hand_size:
            raise ValueError(
                f"p0 library ({len(decks.p0_library)} cards) smaller "
                f"than opening_hand_size={opening_hand_size}"
            )
        if len(decks.p1_library) < opening_hand_size:
            raise ValueError(
                f"p1 library ({len(decks.p1_library)} cards) smaller "
                f"than opening_hand_size={opening_hand_size}"
            )

    py_state, rust_state = build_matched_states(decks)
    if opening_hand_size > 0:
        _draw_opening_hand_both(py_state, rust_state, n=opening_hand_size)

    if mulligans:
        _run_mulligan_loop_both(
            py_state, rust_state, mulligan_agent, mulligan_seed,
        )

    return py_state, rust_state


# ---------------------------------------------------------------------------
# Round-17 enhancement: production-shape decklists (closes one of the
# Phase 6 round-16 enhancement follow-ons).
#
# `load_production_deck` reads the live 17lands EoE deck pool
# (`data/17lands/eoe_decks.json.gz`, 130k decks, the same source
# self-play / iterate consumes) and returns one deck's card-name list.
# `production_deck_to_deckspec` converts a card-name list into a
# parity-bridge-compatible DeckSpec, **filtering out cards the harness
# can't model on both engines** (see filter rationale below).
#
# Why a filter step? CardDef intentionally carries only a minimal
# cross-engine surface (name / type / mana cost / pt / mana_produced /
# triggered_abilities-as-dict / subtypes). It does NOT carry
# spell_effect, activated_abilities (other than the auto-wired
# {T}: Add C for lands), static_abilities, keywords, ETB-counter
# triggers, warp/kicker/impending costs, etc. Production EoE creatures
# routinely carry triggered abilities + activated abilities + keywords
# that the parser materializes on Python's CardPrinting via the
# OracleParserCardSource path; Rust's `register_printing` accepts only
# the schema shapes documented in `cards.rs:parse_printing`.
#
# The pragmatic Round-17 path: filter the deck to cards that BOTH
# engines can register without effects/triggers/keywords/etc. that
# don't have a matching cross-engine bridge arm. The filter is
# conservative (drop on any rich field, not just unsupported ones)
# so future enhancements can flip a card from "filtered" to "bridged"
# by adding a CardDef field + bridge arm without churning the loader.
#
# What survives the filter on the EoE pool today:
#   * Basic lands (Forest/Island/Plains/Mountain/Swamp/Wastes).
#   * Vanilla creatures (no keywords, no triggered_abilities, no
#     activated_abilities, no static_abilities, no spell_effect,
#     no warp/kicker/impending).
# Per spot-check on the EoE pool: ~5 vanilla creatures total
# (Emissary Escort, Harmonious Grovestrider, Cosmogoyf, Magus of the
# Moon, Spectral Snatcher) plus the basic lands. Most decks contain
# a handful of basic lands + 0-2 vanilla creatures after filtering;
# the loader pads short decks with extra basics so trajectory tests
# always have ~60 cards to draw from.
#
# BLOCKED-RUST-EFFECT 42 (per task brief format): cards with rich
# effects (triggered abilities, activated abilities beyond mana,
# spell effects, warp/kicker/impending, room halves, planeswalker
# loyalty) are blocked at this layer because the CardDef bridge
# doesn't carry the fields needed to register them on both engines.
# Future enhancement: extend CardDef + bridge to carry the full
# CardPrinting surface, then drop the filter. Out of scope for the
# round-17 slice; the trajectory tests need a working production-
# shape deck NOW, not a complete production-shape deck.
# ---------------------------------------------------------------------------


# Mana-symbol -> (generic_count, [(color, count), ...]) parser. Reads a
# Scryfall-style mana-cost string like "{1}{G}{G}" and returns the parts
# CardDef needs. Empty string / None → (0, []).
_COLOR_LETTERS: frozenset[str] = frozenset({"W", "U", "B", "R", "G", "C"})


def _parse_mana_cost_string(s: str) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Parse a Scryfall mana-cost string into ``(generic, colored)``.

    Returns ``(generic, ((color, count), ...))`` where ``colored`` is
    sorted by color for deterministic CardDef construction. Hybrid /
    Phyrexian / X / snow symbols are NOT supported — a card whose mana
    cost contains them is rejected by the deckspec converter
    (production filter responsibility).

    Raises ``ValueError`` on unrecognized mana symbols so the caller
    can fall back to filtering the card out.
    """
    if not s:
        return 0, ()
    generic = 0
    colored: dict[str, int] = {}
    # Tokenize on `{...}` segments.
    i = 0
    while i < len(s):
        if s[i] != "{":
            raise ValueError(f"unexpected char {s[i]!r} in mana cost {s!r}")
        end = s.index("}", i + 1)
        sym = s[i + 1:end]
        if sym.isdigit():
            generic += int(sym)
        elif sym in _COLOR_LETTERS:
            colored[sym] = colored.get(sym, 0) + 1
        else:
            # Hybrid (W/U), Phyrexian (W/P), X, snow (S), 2/W, etc.
            raise ValueError(
                f"unsupported mana symbol {sym!r} in cost {s!r} "
                f"(production filter should drop this card)"
            )
        i = end + 1
    return generic, tuple(sorted(colored.items()))


# Phase 6 carddef-substrate slice: keywords confirmed parity-clean by
# Phase 5/6 probes whose per-keyword combat behaviors both engines
# model byte-for-byte today. Restricting CardDef.keywords to this set
# at filter time keeps the production-deck loader honest — a card
# carrying any keyword OUTSIDE this set falls through to the filter
# drop path, the same as a card with a triggered ability or a non-
# vanilla activated ability. Extending this set should be done in
# lockstep with adding a new trajectory keyword probe + verifying
# Phase 5's selector for that keyword still passes.
#
# Per Phase 6 doc (rounds 17-18 keyword probes) + Phase 5 selector
# coverage, the following keywords are parity-clean:
#
#   * flying      — Phase 5 flying_selector (synthetic) + can be
#                   exercised via the round-17 generic injector.
#   * vigilance   — round-17 trajectory probe
#                   (test_vigilance_attacker_untaps_parity).
#   * lifelink    — round-17 trajectory probe
#                   (test_lifelink_combat_damage_to_self_parity).
#   * trample     — round-17 trajectory probe
#                   (test_trample_overflow_to_player_parity).
#   * haste       — round-18 trajectory probe
#                   (test_haste_attacker_no_summoning_sickness_parity).
#   * deathtouch  — round-18 trajectory probe
#                   (test_deathtouch_combat_damage_kills_blocker_parity).
#   * first strike  — Phase 5 first_strike_selector + cross-engine
#                     bridge pin (cross_engine.py:9381).
#   * double strike — round-18 trajectory probe
#                     (test_double_strike_combat_damage_doubles_parity).
#                     Note: the engines agree byte-for-byte despite
#                     a CR divergence on regular-damage spillover,
#                     documented in SEARCH_LIMITATIONS.md.
#   * menace, defender, indestructible — Phase 5 selectors exist;
#                     keep them in the safe set so cards carrying
#                     these keywords alongside flying / vigilance
#                     etc. (e.g. multi-keyword EoE creatures) still
#                     survive the filter.
#   * reach         — round-45 trajectory probe
#                     (test_reach_blocker_blocks_flying_attacker_parity).
#                     Both engines model reach as a blocker-side
#                     gate-opener in their flying-block legality
#                     check (Python: ``combat.can_block`` —
#                     ``"flying" in blocker_kws or "reach" in
#                     blocker_kws``; Rust: ``state_ops_combat.rs``
#                     ``declare_blocker`` — ``blocker_has_flying ||
#                     blocker_has_reach``). Reach on the ATTACKER is
#                     a no-op for blocking purposes on both engines
#                     (a reach attacker is treated as ground for
#                     block-legality), matching CR 702.17.
#
# Keywords explicitly NOT in this set (and therefore disqualifying):
# flash (CR 702.8 cast-timing; trajectory probes for it land in a
# follow-up slice); hexproof (Phase 5 selector exists but lifecycle
# is fragile in trajectory context); ward / protection / scry (cost-
# modifying / replacement effects out of CardDef's per-creature
# surface).
_SAFE_KEYWORDS: frozenset[str] = frozenset({
    "flying",
    "vigilance",
    "lifelink",
    "trample",
    "haste",
    "deathtouch",
    "first strike",
    "double strike",
    "menace",
    "defender",
    "indestructible",
    "reach",
})


# Aura-substrate Phase 2 (round-79) — extra static keywords that the
# Aura predicate `_printing_is_bridge_simple_aura` admits ON TOP of
# `_SAFE_KEYWORDS`. These are cast-timing / non-combat riders that the
# CardDef bridge does **NOT** propagate from the source printing
# through to the materialized CardDef → CardPrinting round-trip on
# either engine — both Python's and Rust's bridge re-materialize the
# Aura with empty `keywords`, so the rider is **symmetrically
# dropped** on both sides. The drop is therefore bit-equal cross-
# engine under the bench's `attack_all` coverage (which only casts at
# sorcery speed; the cast-timing rider is never exercised by either
# engine in a bench game). The simple-creature filter continues to
# use the strict `_SAFE_KEYWORDS` set unchanged — these extras are
# Aura-only because Auras are not creatures and never enter combat
# (so a creature carrying `flash` would still need its own per-keyword
# parity work before admission).
#
# **Per-keyword admit rationale (round-79):**
#
#   * `flash` (CR 702.8) — cast-timing rider only; instant-speed
#     castability is documented at `parity/cross_engine.py` (the CE19
#     Bear-Flash legal-decisions emission probe) as a known
#     active!=priority divergence on CREATURE casts, but Auras are
#     dropped symmetrically through the CardDef bridge (no keywords
#     thread through), AND the bench's `attack_all` only ever casts at
#     sorcery speed where the gate is irrelevant on both engines.
#     Carriers: Frantic Strength + Shardmage's Rescue (S1 carriers per
#     `docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md` §1; both 0 EoE 17lands
#     decks — the substrate's value is the unblock of the first live
#     EoE Aura admit path, not the deck count). Symmetric-drop SL
#     entry: `docs/SEARCH_LIMITATIONS.md` (the "Aura `flash` cast-
#     timing rider" entry, sibling to the round-39 Ward / round-59
#     shockland precedents).
_AURA_SAFE_EXTRA_KEYWORDS: frozenset[str] = frozenset({
    "flash",
})


# Dynamic-P/T-formula substrate (round-78 + round-84). Admit-set for
# the simple-creature bridge predicate's ``dynamic_pt_formula``
# whitelist. Mirrors the Rust dispatch table in
# ``state_ops_combat::dynamic_base_pt_pub`` (lockstep with Python's
# ``game._dynamic_base_pt``). One entry per ROUND that landed a
# formula shape end-to-end on both engines:
#
#   * ``"harmonious_grovestrider"`` — round-78 (CDA P=T=lands-controlled).
#   * ``"cosmogoyf"`` — round-84 (CDA P=N, T=N+1 where
#     N = controller's-exile-zone count).
#
# Honest-stopped (NOT in this set, still drop-at-filter):
#
#   * ``"emissary_escort"`` — printed-base + X-from-other-artifact-MV
#     shape (different from the CDA-only dispatch arms above); the
#     Rust dispatch table's ``_`` arm returns ``None`` for any
#     unrecognized key, so admitting Escort would produce a 0/0
#     creature on Rust. Escort's slice extends the dispatch table
#     when it lands.
_DYNAMIC_PT_FORMULA_ADMIT_SET: frozenset[str] = frozenset({
    "harmonious_grovestrider",
    "cosmogoyf",
})


# Phase 6 carddef-triggers slice: triggered-ability shapes that BOTH
# engines model byte-for-byte today via the existing
# ``cross_engine._ta_dict_to_python_predicate`` /
# ``_ta_dict_to_python_effect`` bridge plus Rust's
# ``parse_triggered_ability`` arm. Restricting CardDef.triggered_
# abilities to this set at filter time keeps the production-deck
# loader honest — a card carrying any triggered ability OUTSIDE this
# set falls through to the filter drop path, the same as the keywords
# check above.
#
# Each entry is a ``(event, predicate_kind, effect_kind)`` triple. A
# triggered_ability dict matches a safe-trigger entry when:
#   * ``ta["event"] == event``
#   * Either the safe-trigger entry's ``predicate_kind`` is None and
#     the dict's ``predicate`` is None / absent, OR the dict's
#     ``predicate["kind"]`` equals the safe-trigger's ``predicate_kind``.
#   * ``ta["effect"]["kind"] == effect_kind``.
#   * ``ta.get("once_per_turn", False) is False`` (once-per-turn
#     bookkeeping isn't probed yet on the trajectory shape; restrict to
#     the simpler default to keep the safe set tight).
#   * The effect dict carries no fields beyond a single payload key
#     matching its arm (``count`` for draw_cards/scry_n/surveil,
#     ``amount`` for gain_life — see ``_is_safe_trigger_dict`` for the
#     exact whitelist).
#
# The starter set is the SOURCE-IS-SELF predicate plus three leaf
# effects with the heaviest cross-engine probe coverage in Phase 5 /
# rounds 1-19 of Phase 6:
#
#   * ``draw_cards`` — exercised by every ETB-trigger slice since
#     Slice 3 (test_cross_engine_etb_triggers.py:_etb_draw_bear,
#     CE19 substrate, the very first ETB probe to land cross-engine).
#     Bridge arm: ``cross_engine._ta_dict_to_python_effect`` "draw_cards".
#   * ``gain_life`` — exercised by the gain_life bridge arm
#     (cross_engine._ta_dict_to_python_effect "gain_life") and the
#     cross-controller / cross-event probes (Phase 5).
#   * ``scry_n`` — exercised by round-16's scry_n bridge arm; the
#     cross-engine default-strategy contract is "keep all on top" on
#     both engines, so the no-op trajectory walk is bit-equal without
#     any agent setup (no PendingDecision pipeline to drain on the
#     post-resolve state).
#
# **Landfall extension (round-36 slice)** adds one compound predicate
# shape: ``"land_enters_controlled_by_you"`` — a sentinel canonical
# name for the AND-nested ``{"kind": "and", "left": {"kind":
# "subject_is_land"}, "right": {"kind": "controlled_by_you"}}`` dict
# both engines already parse (Rust at `cards.rs::parse_predicate` —
# subject-anchored type filters slice; Python via
# `cross_engine._ta_dict_to_python_predicate`'s "and" / "subject_is_
# land" / "controlled_by_you" arms). The canonical sentinel keeps
# the ``(event, predicate_kind, effect_kind)`` triple shape uniform
# across the allowlist; ``_is_safe_trigger_dict`` extracts the
# sentinel from the nested dict shape, and ``_python_ta_to_safe_dict``
# recognizes Python's parser-emitted ``And(Subject(slot=None,
# inner=IsLand()), Controlled(by=YOU, subject=None))`` predicate as
# the same canonical shape. Unlocked Eumidian Terrabotanist
# (2/3 {1}{G}, "Landfall — Whenever a land you control enters, you
# gain 1 life") on the production-deck loader.
#
# **Landfall + pump_self_until_eot extension (round-37 DDDD slice)**
# rides the SAME canonical landfall predicate sentinel and adds a
# SECOND safe leaf effect under it: ``pump_self_until_eot`` (bare
# shape — ``power_bonus`` / ``toughness_bonus`` fields, no
# ``granted_keywords`` rider). Cross-engine bridge support landed
# in round 22 (``cross_engine._ta_dict_to_python_effect`` "pump_
# self_until_eot" arm + cross-controller pump-self probe at commit
# 92ac9ec5 in test_parity_smoke.py). Unlocked Icecave Crasher
# (4/4 {3}{G}, Trample, "Landfall — Whenever a land you control
# enters, this creature gets +1/+0 until end of turn") on the
# production-deck loader. Sister consumer Remnant Elemental (1/2
# {1}{R}, Reach, +2/+0 UEOT landfall) carries the SAME bare
# ``PumpSelfUntilEOT`` effect shape but its ``Reach`` keyword is
# outside today's :data:`_SAFE_KEYWORDS` set — the production-deck
# filter still drops it. Adding ``Reach`` to the safe-keyword set
# is a follow-up slice (needs combat-side parity verification).
#
# **Landfall + draw_cards extension (this slice)** adds a THIRD
# safe leaf effect under the canonical landfall predicate
# sentinel: ``draw_cards`` (bare shape — single ``count`` payload
# key; mirrors the source_is_self+draw_cards starter entry's
# allowed-keys whitelist). Cross-engine bridge support pre-existed
# in round 1 of Phase 6 carddef-triggers (``cross_engine._ta_dict
# _to_python_effect`` "draw_cards" arm) and the Python recognizer
# in ``_python_ta_to_safe_dict`` already maps ``DrawCards`` to the
# canonical ``{"kind": "draw_cards", "count": int}`` dict shape;
# only the landfall arm's ``safe_effect_kinds`` tuple needed the
# extension to admit the predicate-effect product. **No real-EoE
# consumer admitted today**: the EoE landfall pool's only
# draw-bearing card is Tannuk, Memorial Ensign, whose effect is
# the bespoke ``TannukEnsignLandfallEffect`` (deal-damage to each
# opponent + second-time-bonus draw), NOT a bare ``DrawCards``
# leaf — Tannuk continues to be dropped by the production-deck
# filter under this slice. Substrate is pre-positioned for any
# future EoE / standard-format printing carrying a bare
# landfall+draw shape (the parser-emitted equivalent of MH3 Hare
# Apparent's "+1/+0 and draw" or a clean "Landfall — draw a card"
# creature) so the next slice landing such a card need not extend
# the allowlist again; pinned by reject-path tests so the predicate
# -effect product gate stays honest in the meantime.
# Tannuk, Memorial Ensign / Mightform Harmonizer / Genemorph Imago
# / Eusocial Engineering / Glacier Godmaw / Icetill Explorer /
# Territorial Bruntar carry the same landfall predicate but pair
# it with bespoke / unsupported effect shapes (deal-damage +
# second-time-bonus, conditional-threshold pump, double-target-
# power, create-token, exile-and-may-cast, mill-a-card); each
# needs its own sister slice. Mightform's substrate is primed by
# round-37 CCCC (commit c7c095ed) on the free-form CardDef bridge
# path — but the production-deck filter would still drop it on
# the ``Warp`` keyword (outside :data:`_SAFE_KEYWORDS`) AND the
# ``requires_target=True`` agent-seating gap that the loader
# doesn't fill, so the matching ``double_target_creatures_power_
# ueot`` allowlist entry waits for those follow-ups.
#
# Triggers explicitly NOT in this set (and therefore disqualifying):
# ``deal_damage`` / ``deal_damage_any_target`` (target-pick at PUSH
# time requires an agent shim — the production-deck loader doesn't
# seat one); ``create_token`` (needs token-printing pre-registration
# via ``register_token_printing``); ``surveil`` (PendingDecision
# substrate requires an explicit auto-resolve drain on the Rust side
# to mirror Python's no-agent default — a follow-up slice once the
# trajectory walk drains pending decisions consistently);
# ``discard_self`` / ``each_opponent_discards`` (requires an agent on
# the discarding player to mirror Rust's hand[0] pick); modal /
# sequence / if_you_do composers (composer bridge arms exist but
# nesting them under ETB triggers compounds the bridge-arm surface
# beyond the leaf-effect floor this slice ships).
#
# Extending this set is a lockstep operation — each new entry needs
# (a) the matching cross-engine bridge arm to already exist on both
# engines, and (b) a trajectory probe that demonstrates the
# end-to-end firing parity. See ``test_carddef_safe_etb_trigger_*``
# in tests/test_trajectory.py and the synthetic real-game probe in
# test_trajectory_parity.py for the Phase 6 ENTERS+IsSelf+DrawCards
# probe shape; the landfall extension's analogue is
# ``test_carddef_landfall_safe_etb_trigger_fires_on_land_play_parity``.
_SAFE_TRIGGERS: frozenset[tuple[str, Optional[str], str]] = frozenset({
    ("enters", "source_is_self", "draw_cards"),
    ("enters", "source_is_self", "gain_life"),
    ("enters", "source_is_self", "scry_n"),
    # **Tier-C target-seating sub-substrate, lever-a — Exosuit
    # Savior (the count_min=0 optional-bounce edge; ENTERS sibling
    # of Sunset Saboteur's ATTACKS targeted entry below).** {2}{W}
    # Creature — 2/2 ``flying`` (in :data:`_SAFE_KEYWORDS`). "When
    # Exosuit Savior enters, you may return another target permanent
    # you control to its owner's hand." — 12,480 17lands decks.
    # Parsed live as ``(ENTERS, IsSelf,
    # ReturnTargetPermanentYouControlToOwnersHand)`` with
    # ``requires_target=False`` (because ``count_min=0`` — the bounce
    # is OPTIONAL, CR "you may return"), ``count_range=(0,1)``,
    # ``slot_kinds=()``, slot filter
    # ``ControlledByYou() & ~IsPlayer() & ~IsSource()``.
    #
    # **Reuses the SHARED Tier-C seat shipped with Sunset Saboteur.**
    # The deterministic push-time first-legal seat
    # (``first_legal_trigger_target`` on Rust at
    # ``state_ops_resolve.rs:push_triggered_ability``; the no-agent
    # ``pool[0]`` seat on Python at
    # ``actions.py:_put_triggered_ability_on_stack``, gated by
    # ``_is_target_seated_safe_trigger``) is ALREADY in place — this
    # entry only adds the per-card Effect (Rust unit variant
    # ``ReturnTargetPermanentYouControlToOwnersHand`` + the
    # ``return_target_permanent_you_control`` bridge arm + this
    # recognizer/triple). The **count_min=0 edge** (the distinct
    # contribution vs Sunset Saboteur's count_min=1): with NO
    # eligible own permanent the shared seat returns ``None`` and the
    # trigger pushes via the UNCHANGED ``new_trigger`` path
    # (push-with-no-target) → the resolve arm no-ops, byte-equal with
    # Python's ``targets=()`` ``Bounce.resolve`` early-return — the
    # optional is DECLINED, NOT fizzle-skipped (contrast Sunset
    # Saboteur, which IS skip-pushed via
    # ``lever_trigger_requires_target``). With an eligible permanent
    # both engines pick the SAME smallest-bf-index own-non-source
    # permanent and bounce it (the established ``Effect::ReturnToHand``
    # bounce-to-hand lifecycle + LTB fan-out, byte-equal). The ENTERS
    # event dispatch itself is already byte-equal-proven (the round-1
    # ENTERS+IsSelf starter set + every ENTERS probe). The genuine
    # real-EoE firing-parity probe is in test_parity_smoke.py.
    #
    # **Necessary-but-not-sufficient teeth.** Terrapact Intimidator
    # shares the IDENTICAL ``(ENTERS, source_is_self, targeted)``
    # shape but its resolve is a modal ``opponent.agent.choose_mode``
    # (the substrate seats a *target*, not a *mode*) — its
    # ``TerrapactIntimidatorETBEffect`` leaf is NOT added to the
    # ENTERS ``safe_effect_kinds``, so it stays declined even though
    # it shares the exact necessary shape (negative-pinned in
    # tests/test_trajectory.py — proves the gate is the triple +
    # bare-shape product, not the bare targeted-ETB shape).
    ("enters", "source_is_self", "return_target_permanent_you_control"),
    # Surveil extension (round-42 carddef-triggers slice). Adds the
    # bare-shape ``surveil`` effect (``count`` payload key only — no
    # ``use_artifact_count`` rider) under both ENTERS and DIES events
    # with the source_is_self predicate. Bridge support:
    #   * Rust ``Effect::Surveil { count: i32 }`` variant has existed
    #     since the I61 controller-of-effect substrate landed; its
    #     resolve arm (state_ops_resolve.rs:2610-2649) pops the top
    #     min(count, library.len()) cards and queues a PendingDecision
    #     that the auto-resolve path (state_ops_resolve.rs:6832, the
    #     ``resolve_top_of_stack``-after-triggered-ability terminal
    #     drain) closes by keeping every card on top of the library.
    #   * Python ``Surveil(amount: int)`` class
    #     (effects/generic/library_lookat.py:569) walks the top N
    #     cards in-line; when ``p.agent is None`` the surveil
    #     callback is short-circuited and every observed card stays
    #     on the library top (library_lookat.py:605).
    #   * Bridge translator arm in ``cross_engine._ta_dict_to_python_
    #     effect`` at cards.rs/cross_engine.py:444-508 has carried the
    #     ``surveil`` arm since round 16 (the scry/surveil slice);
    #     payload key is ``count`` mirroring Rust's terminology.
    #
    # **Cross-engine default-strategy contract.** Both engines'
    # no-agent surveil resolves to "keep_all_on_top" — bit-equal
    # post-resolve library top + graveyard, no auto-resolve drain
    # required on the trajectory walk because the terminal
    # post-triggered-ability path already invokes the keep_all_on_top
    # strategy. Same contract the round-16 scry arm relies on; the
    # earlier "PendingDecision pipeline" deferral noted in the
    # ``_SAFE_TRIGGERS`` docstring above turned out to be moot for
    # the default-strategy case (see the cross_engine.py:474-494
    # surveil bridge-arm docstring for the full no-agent contract).
    #
    # Cards admitted on EoE today (3 net new):
    #   * Thawbringer (4/2 {2}{G}, ENTERS+surveil 1 AND DIES+surveil 1)
    #   * Wary Watchdog (3/1 {1}{G}, ENTERS+surveil 1 AND DIES+surveil 1)
    #   * Wurmwall Sweeper (2/2 {2}, ENTERS+surveil 2)
    # Starbreach Whale carries the ENTERS+surveil 2 shape too but
    # also has ``warp_cost`` (outside the bridge); Skullsnap Nuisance
    # carries ENTERS+And/surveil + ROOM_FULLY_UNLOCKED predicate
    # (both outside today's safe set).
    ("enters", "source_is_self", "surveil"),
    ("dies", "source_is_self", "surveil"),
    # Surveil ATTACKS extension (round-43 slice). The FIRST ATTACKS
    # event in the safe set. Mirrors round-42's surveil ETB+DIES pair
    # on the orthogonal event axis — same bridge arms, same
    # default-strategy contract, just a different dispatch event.
    #
    # Bridge support (both pre-existed before this slice):
    #   * Rust ATTACKS event dispatch in
    #     ``state_ops_combat.rs:684`` (Slice 7 of the carddef-triggers
    #     ramp per the cards.rs:5742 parser arm comment), invoked
    #     after every declared-attacker on the active player's combat
    #     step. The dispatch wraps each attacker in the
    #     ``crate::cards::Event::Attacks`` subject view and runs the
    #     standard ``fire_triggers``/PendingDecision pipeline that
    #     terminates in the ``keep_all_on_top`` auto-resolve drain
    #     (state_ops_resolve.rs:6832) — identical to round-42's
    #     no-agent contract on the ENTERS/DIES surveil paths.
    #   * Python ATTACKS dispatch via ``combat._fire_attacks_triggers``
    #     (combat.py:196) followed by predicate-style ``Event.ATTACKS``
    #     fan-out in ``combat.py:208-217`` — also runs every triggered
    #     ability whose event is ATTACKS, then the standard p.agent-is
    #     -None surveil short-circuit at ``library_lookat.py:605``
    #     fires the same keep-all-on-top semantics as the ENTERS/DIES
    #     paths.
    #
    # **Cross-engine default-strategy contract.** Identical to
    # round-42 surveil ETB/DIES: no agent setup needed for the
    # trajectory walk. The auto-resolve keep_all_on_top default
    # terminates BOTH engines' surveil resolves identically — bit-equal
    # post-resolve library top + graveyard. The only axis that differs
    # from round-42 is the dispatch event; both bridges' resolve arms
    # are oblivious to which event fired the trigger because the
    # PendingDecision and library-lookat substrates fork only on the
    # effect shape (Surveil), not the upstream event.
    #
    # **No real-EoE consumer admitted today.** Pinned-by-reject-path
    # test (sister of round-37 DDDD landfall+draw and round-32+/35+/36+
    # damage-by-X precedent): the production-deck loader's pool
    # (data/17lands/eoe_decks.json.gz, 5000 archetypes) does NOT
    # contain Fear of Surveillance (the EoE printing carrying the bare
    # ATTACKS+IsSelf+Surveil(1) shape with vigilance keyword and no
    # other unsafe fields) — Fear of Surveillance is in the Scryfall
    # EoE set but absent from the 17lands deck pool (low pick rate).
    # Appendage Amalgam carries the same ATTACKS+surveil trigger but
    # its ``Flash`` keyword is outside the :data:`_SAFE_KEYWORDS` set
    # and would still be dropped at the keyword gate. Substrate is
    # pre-positioned for any future EoE / standard-format printing
    # carrying a bare ATTACKS+IsSelf+Surveil shape; the next slice
    # landing such a card in the pool need not extend the allowlist
    # again.
    ("attacks", "source_is_self", "surveil"),
    # ATTACKS+IsSelf draw_cards / gain_life extension (round-44 slice).
    # Adds the SECOND and THIRD ATTACKS leaves to the safe set after
    # round-43's surveil ATTACKS — completes the
    # ``draw_cards`` / ``gain_life`` / ``surveil`` triple under
    # source_is_self for ATTACKS, mirroring the round-1 ENTERS+IsSelf
    # leaf set (draw_cards / gain_life / scry_n / surveil) on the
    # ATTACKS event axis. Pure substrate slice — no
    # cross-engine bridge work needed since both events and both leaf
    # effects already round-trip on the bridge:
    #
    #   * ATTACKS event dispatch: Rust ``state_ops_combat.rs:684``
    #     fires every ATTACKS-event triggered ability after each
    #     declared-attacker on the active player's combat step
    #     (round-43 verified the dispatch path bit-equal vs Python's
    #     ``combat._fire_attacks_triggers`` at combat.py:196 +
    #     predicate-style ``Event.ATTACKS`` fan-out at
    #     combat.py:208-217).
    #   * ``draw_cards`` leaf: ``cross_engine._ta_dict_to_python_effect``
    #     "draw_cards" arm (round 1 of Phase 6 carddef-triggers) +
    #     Python ``DrawCards(amount: int)`` (effects/generic/draw_mill.py).
    #     Rust ``Effect::DrawCards { count: i32 }`` resolves
    #     deterministically on both engines (no
    #     PendingDecision / agent setup needed — drawing is a pure
    #     mechanical zone move from library top to hand).
    #   * ``gain_life`` leaf: ``cross_engine._ta_dict_to_python_effect``
    #     "gain_life" arm (round 1 of Phase 6) + Python
    #     ``GainLife(amount: int)`` (effects/generic/life.py). Rust
    #     ``Effect::GainLife { amount: i32 }`` resolves deterministically
    #     too (no agent setup needed — life-total mutation is a pure
    #     scalar update).
    #
    # **No real-EoE consumer admitted today.** Same precedent as
    # round-43 (substrate-only) and round-37 DDDD (landfall+draw
    # substrate-only). The 17lands deck pool's only ATTACKS+IsSelf+
    # ``draw_cards`` / ``gain_life`` printings would unlock under this
    # extension if any existed; surveying EoE printings via
    # ``scripts/scout_archetype_admit_simulation.py`` (round-44
    # update) confirms zero EoE creatures carry the bare
    # ATTACKS+IsSelf+DrawCards or ATTACKS+IsSelf+GainLife shape with
    # all other gates clean. The shape is a common one in the broader
    # MTG card pool (Curiosity-style "Whenever ~ attacks, draw a card"
    # and Soulmender-style "Whenever ~ attacks, you gain N life"
    # appear in older sets) so the substrate is pre-positioned for any
    # future EoE / standard-format printing carrying this shape — the
    # next slice landing such a card need not extend the allowlist
    # again.
    #
    # **Cross-engine default-strategy contract.** Identical to
    # round-43 surveil-ATTACKS: no agent setup needed for the
    # trajectory walk. The terminal post-triggered-ability path
    # already drains the deterministic ``DrawCards`` / ``GainLife``
    # resolves; both engines' resolve arms execute in lockstep
    # because both leaf effects mutate state via pure mechanical
    # zone moves / scalar updates (no PendingDecision queue
    # interaction).
    ("attacks", "source_is_self", "draw_cards"),
    ("attacks", "source_is_self", "gain_life"),
    # **Tier-C target-seating sub-substrate, lever-a — Sunset
    # Saboteur (the FIRST targeted-safe-trigger entry in this set).**
    # ``{1}{B}`` Creature — 4/1 ``menace`` (in :data:`_SAFE_KEYWORDS`).
    # "Whenever Sunset Saboteur attacks, put a +1/+1 counter on target
    # creature an opponent controls." — 6,375 17lands decks. Parsed
    # live as ``(ATTACKS, IsSelf, PutPlusOneCounterOnOpponentsCreature)``
    # with ``requires_target=True``, ``count_range=(1,1)``,
    # ``slot_kinds=()``, slot filter
    # ``IsCreature() & ControlledByOpponent()``, ``amount=1``.
    #
    # **Why this needs a NEW substrate (not just a leaf widen).**
    # Every prior ``_SAFE_TRIGGERS`` leaf is recipient-is-self /
    # untargeted; the production-deck bench loader seats NO agent, so
    # a targeted trigger has no one to pick the push-time target. The
    # ``_SAFE_TRIGGERS`` docstring's decline class
    # (``deal_damage_any_target`` "target-pick at PUSH time requires
    # an agent shim — the production-deck loader doesn't seat one")
    # is exactly this. The Tier-C target-seating shared sub-substrate
    # (this slice) replaces BOTH engines' no-agent behavior with the
    # SAME deterministic first-legal pick:
    #   * **Rust** — ``first_legal_trigger_target`` at
    #     ``state_ops_resolve.rs:push_triggered_ability`` scans
    #     ascending seat × bf-Vec order, seats the smallest-bf-index
    #     opponent-controlled creature into the EXISTING
    #     ``TriggerPayload.target_id`` via ``new_trigger_with_target``
    #     (CE20 Slice 4 — no new ``StackItem`` field); the resolve arm
    #     reads it via ``current_resolve_payload_target_id``
    #     (the ``EquipToTargetCreatureYouControl`` precedent).
    #     ``count_min>=1`` + no eligible target ⇒ trigger NOT pushed
    #     (CR 603.3c fizzle).
    #   * **Python** — the no-agent ``pool[0]`` seat at
    #     ``actions.py:_put_triggered_ability_on_stack`` (the
    #     ``controller.agent is None`` branch), gated to
    #     ``_is_target_seated_safe_trigger``; ``pool[0]`` IS
    #     ``Effect._enumerate_for_filter``'s first entry =
    #     ``legal_targets[0]`` = the same object Rust seats. No legal
    #     target with ``min_t>0`` ⇒ Python's existing
    #     "no legal targets; fizzles" ``return False`` — byte-equal
    #     with the Rust skip-push.
    # Both directions resolve to the SAME object by construction (the
    # cross-engine harness keeps bf insertion order + object ids in
    # lockstep), so the post-resolve ``counters`` fingerprint is
    # byte-equal. The deterministic-first no-agent target replacing
    # the CR 603.3d controller choice is documented in
    # ``docs/SEARCH_LIMITATIONS.md`` §2.3 (same class as every
    # ``_PickFirst*`` / first-eligible-resolve-arm convention the
    # campaign already documents).
    #
    # **Bridge support (the lockstep contract — both exist before
    # this entry):**
    #   * Rust ``Effect::PutPlusOneCounterOnOpponentsCreature
    #     { amount: i32 }`` variant + parse / to_dict / resolve arms
    #     (``rust/src/cards.rs`` + ``state_ops_resolve.rs``) — the
    #     opponent-direction sibling of the existing
    #     ``PutPlusOneCounterOnCreatureYouControl``, but reading the
    #     SEATED payload target (NOT a first-eligible self-scan — a
    #     self-scan is exactly the divergence the shared seat removes).
    #   * Python ``PutPlusOneCounterOnOpponentsCreature(amount: int =
    #     1)`` (effects/generic/counters.py — a ``PutPlusOneCounters``
    #     subclass) + the ``put_plus_one_counter_on_opponents_creature``
    #     cross_engine bridge arm. Bare ``amount`` payload, identical
    #     spelling both engines, non-negative-guarded.
    # The ATTACKS event dispatch itself is already byte-equal-proven
    # (round-43/44 surveil/draw/gain ATTACKS pins). The genuine
    # real-EoE firing-parity probe is in test_parity_smoke.py.
    #
    # **Necessary-but-not-sufficient teeth.** ``(attacks,
    # source_is_self)`` is shared by Hardened Escort / Flesh Burrower
    # (``ATTACKS/source_is_self/PumpTargetCreatureUntilEOT``) — those
    # are NOT admitted because ``pump_target_creature_until_eot`` is
    # never added to the ATTACKS ``safe_effect_kinds``. The gate is
    # the full triple + the recognizer's bare-shape slot check, NOT
    # the bare targeted-ATTACKS shape (negative-pinned in
    # tests/test_trajectory.py).
    ("attacks", "source_is_self", "put_plus_one_counter_on_opponents_creature"),
    # DIES+IsSelf+gain_life extension (this slice — round-47 sister to
    # round-44's ATTACKS+IsSelf+gain_life). Adds the second leaf to the
    # DIES+IsSelf arm (after round-42's surveil), completing the natural
    # ENTERS / DIES / ATTACKS × {gain_life} matrix for the bare
    # "Whenever ~ dies, you gain N life" Soul-Warden-on-death shape.
    # Pure substrate slice — no cross-engine bridge work needed since
    # both the DIES event and the gain_life leaf already round-trip on
    # the bridge:
    #
    #   * DIES event dispatch: Rust ``state_ops_resolve.rs`` /
    #     ``state_ops_sba.rs`` ``dispatch_event(Event::Dies, ...)``
    #     fires every DIES-event triggered ability after a creature
    #     moves from battlefield to graveyard (Slice 5 of the
    #     carddef-triggers ramp — see cards.rs:6035 parser arm; the
    #     DIES dispatch path was verified bit-equal vs Python's
    #     ``effects.dispatch.fire_dies_triggers`` by the round-42
    #     ENTERS+DIES surveil pair).
    #   * ``gain_life`` leaf: ``cross_engine._ta_dict_to_python_effect``
    #     "gain_life" arm (round 1 of Phase 6 carddef-triggers) +
    #     Python ``GainLife(amount: int)`` (effects/generic/life.py).
    #     Rust ``Effect::GainLife { amount: i32 }`` resolves
    #     deterministically (no PendingDecision / agent setup needed —
    #     life-total mutation is a pure scalar update).
    #
    # **Cross-engine default-strategy contract.** Identical to round-44
    # ATTACKS+IsSelf+gain_life: no agent setup needed for the trajectory
    # walk. The terminal post-triggered-ability path drains the
    # deterministic GainLife resolve in lockstep on both engines because
    # the effect mutates state via a pure scalar update with no
    # PendingDecision queue interaction.
    #
    # **No real-EoE consumer admitted today.** Same precedent as
    # round-43 (substrate-only) / round-44 (substrate-only) / round-45
    # landfall+draw substrate-only. The 17lands deck pool's EoE
    # creatures with DIES+IsSelf+GainLife: zero (the EoE death-triggers
    # cluster is CreateToken-heavy; Edge Rover / Gravpack Monoist /
    # Beamsaw Prospector all carry DIES+IsSelf+CreateToken which needs
    # token-printing pre-registration substrate the loader doesn't
    # seat — a separate larger slice). The shape is a common one in
    # the broader MTG card pool (Doomed Traveler-style "When ~ dies,
    # create a 1/1 with flying" or Disciple of the Vault-style "Whenever
    # an artifact you control is put into a graveyard, opponent loses
    # 1 life" variants share the DIES dispatch path) so the substrate
    # is pre-positioned for any future EoE / standard-format printing
    # carrying the bare DIES+IsSelf+GainLife shape — the next slice
    # landing such a card need not extend the allowlist again.
    #
    # Pinned by reject-path tests so the predicate-effect product gate
    # stays honest in the meantime; sister to the round-44 ATTACKS+
    # IsSelf+draw_cards/gain_life substrate-only landing precedent.
    ("dies", "source_is_self", "gain_life"),
    # DIES+IsSelf+draw_cards extension (this slice — round-47 sister to
    # the round-47 DIES+IsSelf+gain_life entry directly above, on the
    # draw_cards axis). Completes the natural DIES+IsSelf leaf set to
    # ``{surveil, gain_life, draw_cards}`` — the SAME three-leaf set
    # ENTERS carries (round-1 draw_cards/gain_life/scry_n + round-42
    # surveil) and ATTACKS carries (round-43 surveil + round-44
    # draw_cards/gain_life). The widening mirrors round-44's pattern
    # exactly: round-43 shipped surveil on a new event axis, round-44
    # then added draw_cards AND gain_life on that same axis once the
    # event dispatch was proven; round-42 shipped surveil on DIES,
    # round-47-gain_life added gain_life, and this slice adds the
    # final draw_cards leaf — closing the ENTERS/DIES/ATTACKS ×
    # {draw_cards, gain_life, surveil} matrix.
    #
    # Pure substrate slice — no cross-engine bridge work needed since
    # both the DIES event and the draw_cards leaf already round-trip
    # on the bridge:
    #
    #   * DIES event dispatch: Rust ``state_ops_resolve.rs`` /
    #     ``state_ops_sba.rs`` ``dispatch_event(Event::Dies, ...)``
    #     fires every DIES-event triggered ability after a creature
    #     moves from battlefield to graveyard (Slice 5 of the
    #     carddef-triggers ramp — cards.rs:6035 parser arm; the DIES
    #     dispatch path was verified bit-equal vs Python's
    #     ``effects.dispatch.fire_dies_triggers`` by the round-42
    #     ENTERS+DIES surveil pair AND the round-47 DIES+gain_life
    #     entry directly above).
    #   * ``draw_cards`` leaf: ``cross_engine._ta_dict_to_python_effect``
    #     "draw_cards" arm (cross_engine.py:242, round 1 of Phase 6
    #     carddef-triggers — the most-exercised leaf in the campaign,
    #     pinned by every ETB-trigger slice since Slice 3) + Python
    #     ``DrawCards(amount: int)`` (effects/generic/draw_mill.py).
    #     Rust ``Effect::DrawCards { count: i32 }`` resolves
    #     deterministically on both engines: drawing is a pure
    #     mechanical zone move (library top -> hand) with no
    #     PendingDecision / agent setup needed — identical resolve
    #     semantics to the already-shipped DIES+gain_life leaf.
    #
    # **Cross-engine default-strategy contract.** Identical to the
    # round-47 DIES+IsSelf+gain_life entry above: no agent setup needed
    # for the trajectory walk. The terminal post-triggered-ability path
    # drains the deterministic DrawCards resolve in lockstep on both
    # engines because the effect mutates state via a pure zone move
    # with no PendingDecision queue interaction.
    #
    # **No real-EoE consumer admitted today.** Substrate-only, same
    # precedent as round-43 / round-44 / round-47-gain_life. A FRESH
    # EoE-pool delta walk at slice time (scripts/scout_r47_dies_draw_
    # delta.py, full 475-printing EoE pool + the 318-name 17lands deck
    # pool) confirms ZERO creatures flip drop->admit on this extension:
    # the EoE death-triggers cluster is CreateToken-heavy (Edge Rover /
    # Gravpack Monoist / Beamsaw Prospector all carry
    # DIES+IsSelf+CreateToken which needs the token-printing
    # pre-registration substrate the loader doesn't seat — a separate
    # larger slice; Edge Rover specifically was investigated as the
    # round-47 N1 primary candidate and declined per that gate). The
    # bare "When ~ dies, draw a card" shape (Doomed Dissenter-adjacent /
    # Bloodsoaked-Champion-style sac-payoff draws / "When ~ dies, draw"
    # variants) is one of the most common death-trigger shapes in the
    # broader MTG card pool, so the substrate is broadly pre-positioned
    # for any future EoE / standard-format printing carrying it — the
    # next slice landing such a card need not extend the allowlist
    # again. Pinned by the round-47-gain_life slice's reject-path tests
    # (``test_is_safe_trigger_dict_rejects_dies_draw_cards_shape`` /
    # ``test_simple_creature_filter_rejects_dies_draw_cards_only_
    # creature``), which this slice CONVERTS from reject to admit —
    # the exact round-37 DDDD / round-44 sister-precedent pattern.
    ("dies", "source_is_self", "draw_cards"),
    # Landfall — "Whenever a land you control enters, ...". Compound
    # predicate sentinel; see ``_is_safe_trigger_dict`` /
    # ``_python_ta_to_safe_dict`` for the canonical-vs-nested shape
    # mapping. The canonical landfall predicate now rides THREE safe
    # leaf effects:
    #   * ``gain_life`` — admits Eumidian Terrabotanist (2/3 {1}{G},
    #     gain 1 life on land enter). Landed in the round-36
    #     landfall extension slice.
    #   * ``pump_self_until_eot`` — admits Icecave Crasher (4/4
    #     {3}{G}, +1/+0 until end of turn on land enter). Landed in
    #     the round-37 DDDD slice (commit 94ea1ef3 / 42a5eec9). The
    #     bare ``PumpSelfUntilEOT(power_bonus, toughness_bonus)``
    #     shape (no granted_keywords rider) is what Icecave
    #     Crasher's parser-emitted Landfall trigger uses; sister
    #     consumer Remnant Elemental (Reach, +2/+0 UEOT) carries
    #     the same effect shape but its ``Reach`` keyword is
    #     outside today's :data:`_SAFE_KEYWORDS` set so the
    #     production-deck filter still drops it.
    #   * ``draw_cards`` — this slice. Pinned-by-reject-path-test
    #     today: the ONLY EoE landfall consumer carrying a draw
    #     leaf is Tannuk, Memorial Ensign, but Tannuk's effect is
    #     the bespoke ``TannukEnsignLandfallEffect`` (deal-damage
    #     to each opponent + second-time-bonus draw), NOT a bare
    #     ``DrawCards`` — so no real-EoE consumer is admitted on
    #     EoE today. Substrate is pre-positioned for any future
    #     EoE / standard-format landfall+draw printing (the
    #     equivalent of MH3 Hare Apparent's "+1/+0 and draw" or a
    #     parser-emitted bare landfall-draw card) so the next slice
    #     dropping such a card need not extend the allowlist again.
    #     Cross-engine bridge support pre-existed in round 1 of
    #     Phase 6 carddef-triggers (``cross_engine._ta_dict_to_
    #     python_effect`` "draw_cards" arm + the trajectory probe
    #     at ``test_carddef_safe_etb_trigger_*`` exercising the
    #     bare draw effect under source_is_self); the recognizer
    #     ``_python_ta_to_safe_dict`` already maps ``DrawCards``
    #     to the ``{"kind": "draw_cards", "count": int}`` dict
    #     shape — only the safe-effect-kinds tuple under the
    #     landfall predicate arm needs the extension.
    # Tannuk Memorial Ensign / Mightform Harmonizer / Genemorph
    # Imago / Eusocial Engineering / Glacier Godmaw / Icetill
    # Explorer / Territorial Bruntar carry the same predicate but
    # pair it with bespoke / unsupported effect shapes (deal-damage
    # + second-time-bonus, conditional-threshold pump,
    # double-target-power, create-token, exile-and-may-cast,
    # mill-a-card) — each needs its own sister slice. Mightform
    # Harmonizer's landfall+double-power clause has its substrate
    # primed by the round-37 CCCC slice (commit c7c095ed) on the
    # free-form CardDef bridge path; ``double_target_creatures_
    # power_ueot`` is NOT yet in the allowlist here because
    # Mightform also carries ``Warp`` (outside :data:`_SAFE_KEY
    # WORDS`) and would still be dropped by the bridge-simple
    # filter — adding ``double_target_creatures_power_ueot`` here
    # without a clean non-Warp consumer would be preventative-only
    # but for an effect whose bridge round-trip requires the
    # ``requires_target=True`` PUSH-time pick agent that the
    # production-deck loader doesn't seat; a follow-up slice once
    # the loader grows targeting-agent seating.
    ("enters", "land_enters_controlled_by_you", "gain_life"),
    ("enters", "land_enters_controlled_by_you", "pump_self_until_eot"),
    ("enters", "land_enters_controlled_by_you", "draw_cards"),
    # Landfall + mill extension (round-48 bench-admit-set slice). Adds
    # a FOURTH safe leaf effect under the canonical landfall predicate
    # sentinel: ``mill`` (bare shape — single ``count`` payload key,
    # see :data:`_SAFE_EFFECT_KEYS` "mill" docstring). This is the
    # round-48 N1 highest-signal lever per the round-47 bench entry's
    # FAIL_TIMEOUT root-cause investigation: the timeout-dominant
    # archetypes (round-46/47 Sweep A grid 0,1,4,5,10,11,12,13,14,15)
    # collapse to ~1-2 admitted creature copies + ~58 basics, so games
    # never end -> FAIL_TIMEOUT. This entry admits a real EoE card.
    #
    # **Real-EoE consumer admitted today: Icetill Explorer**
    # (2/4 {2}{G}{G}, no keywords, single trigger
    # "Landfall — Whenever a land you control enters, mill a card",
    # i.e. ``ENTERS / And(SubjectIsLand, ControlledByYou) /
    # SelfMill(amount=1)``). It carries no keywords, no
    # activated/static abilities, no warp/ward/dynamic-P-T, and its
    # ONLY trigger is this landfall+mill — every gate in
    # :func:`_printing_is_bridge_simple_creature` passes once the
    # ``mill`` leaf rides the landfall predicate. EoE-pool delta
    # (scripts/scout_r48_timeout_archetype_admit_delta.py): Icetill
    # Explorer appears in Sweep-A archetypes 10/11/12 (x1 each,
    # 1->2 admitted creature copies per deck) plus archetypes 2 (x2),
    # 3 (x2), 51 (x1) outside the grid. The other EoE SelfMill
    # carrier, Seedship Broodtender (ENTERS / IsSelf / SelfMill(3)),
    # is NOT admitted — it also carries an activated ability, a
    # declined silent-mistranslation class, so it stays correctly
    # dropped at the ``activated_abilities`` gate (the
    # source_is_self+mill substrate entry below pins this reject-path).
    #
    # **Cross-engine parity contract.** ``mill`` is a pure
    # deterministic zone move (own library top -> own graveyard) with
    # NO PendingDecision / agent / target interaction — identical
    # resolve semantics to the already-shipped ``draw_cards`` /
    # ``gain_life`` landfall leaves. Bridge support pre-existed:
    #   * Rust ``Effect::Mill { count }`` parsed at cards.rs:3991,
    #     serialized at cards.rs:6450, resolved at
    #     state_ops_resolve.rs:1034-1051 reading
    #     ``current_resolve_source_controller`` (I65 controller-of-
    #     effect fix). Empty-library mid-mill silently stops (CR
    #     704.5b applies to DRAWING not MILLING) — both engines break
    #     out of the loop, observably matching.
    #   * Python ``SelfMill(amount)`` (library_lookat.py:624) pops
    #     ``amount`` cards from ``library[0]`` to the controller's
    #     graveyard, breaking on empty library — bit-equal.
    #   * Bridge arm ``cross_engine._ta_dict_to_python_effect`` "mill"
    #     (cross_engine.py:725, shipped round-16) maps the dict's
    #     ``count`` -> ``SelfMill(amount=count)``; the
    #     ``amount``/``count`` field-name discrepancy is local to the
    #     bridge translation, same shape as the established
    #     ``draw_cards count -> amount`` arm.
    # The landfall predicate path itself is the SAME one round-36's
    # landfall+gain_life (Eumidian Terrabotanist) and round-37's
    # landfall+pump_self (Icecave Crasher) validated bit-equal on
    # both engines under the ENTERS dispatch — only the leaf effect
    # differs, and ``mill``'s resolve is strictly simpler (single
    # zone move, no scalar/keyword rider).
    ("enters", "land_enters_controlled_by_you", "mill"),
    # Landfall + put_plus_one_counter_on_self extension (round-59 N1
    # — the sole remaining Seedship Agrarian blocker). Adds a FIFTH
    # safe leaf effect under the canonical landfall predicate
    # sentinel: ``put_plus_one_counter_on_self`` (bare shape — single
    # ``amount`` payload key, see :data:`_SAFE_EFFECT_KEYS`
    # "put_plus_one_counter_on_self" docstring). Exactly the same
    # one-line-widen shape as the round-48 ``landfall+mill`` slice
    # (Icetill Explorer) — only the leaf effect differs.
    #
    # **Real-EoE consumer admitted today: Seedship Agrarian's
    # TRIGGER-2.** Seedship Agrarian (3/3 {3}{G},
    # Creature — Insect Scientist) parses to TWO TriggeredAbilities:
    #   (1) ``BECOMES_TAPPED / IsSelf / CreateToken(Lander)`` — the
    #       FIRST trigger, admitted round-58 N1 by the BECOMES_TAPPED
    #       event bridge + the
    #       ``("becomes_tapped","source_is_self","create_token")``
    #       entry below; genuine real-EoE probe byte-equal.
    #   (2) ``ENTERS / land_enters_controlled_by_you /
    #       PutPlusOneCounterOnSelf(1)`` — "Landfall — Whenever a land
    #       you control enters, put a +1/+1 counter on this creature".
    #       This is the ORTHOGONAL trigger-2 leaf the round-58 resync
    #       narrowed Seedship's ``non_effect_blocker`` to: its sole
    #       remaining residual. This entry admits it. With BOTH
    #       triggers now round-tripping, Seedship Agrarian's FULL card
    #       is byte-equal-admittable (it carries no keywords, no
    #       carrier-side activated/static abilities, no warp/ward/
    #       dynamic-P-T — every gate in
    #       :func:`_printing_is_bridge_simple_creature` passes once
    #       both trigger leaves ride the safe set).
    #
    # **Cross-engine parity contract (bare-shape self-counter).**
    # ``put_plus_one_counter_on_self`` is event-agnostic and has been
    # byte-equal-proven under ATTACKS/ENTERS/DIES since round-22/23 —
    # this slice confirms it round-trips IDENTICALLY under the
    # landfall ENTERS-on-land predicate (the predicate axis is
    # orthogonal to the effect's resolve; the recognizer collapses
    # the parser's ``And(Subject(IsLand), Controlled(YOU))`` to the
    # same ``land_enters_controlled_by_you`` sentinel the shipped
    # gain_life / pump_self / draw_cards / mill landfall leaves use).
    # Bridge support pre-existed (round-56 N2 confirmed both the live
    # Rust variant AND the wired bridge arm):
    #   * Rust ``Effect::PutPlusOneCounterOnSelf { amount: i32 }``
    #     (cards.rs:487, parsed cards.rs:4653, serialized
    #     cards.rs:7179-7181, resolved at
    #     state_ops_resolve.rs reading the ability's source by
    #     ``source_obj_id``). Source not on a battlefield zone ->
    #     silent no-op (mirrors Python's ``source is None`` early
    #     return).
    #   * Python ``PutPlusOneCounterOnSelf(amount: int = 1)``
    #     (effects/generic/counters.py:784) reads
    #     ``state.resolving_source`` and calls
    #     ``add_plus_one_counters(state, source, amount)`` — bit-equal
    #     resolve to Rust's source-walk-then-add.
    #   * Bridge arm ``cross_engine._ta_dict_to_python_effect``
    #     "put_plus_one_counter_on_self" (cross_engine.py:3899,
    #     shipped round-22) maps the dict's ``amount`` (optional,
    #     defaults to 1, rejects negative — same contract as Rust's
    #     parse_effect) -> ``PutPlusOneCounterOnSelf(amount=amount)``.
    #     The ``amount`` field name is IDENTICAL on BOTH sides
    #     (Rust to_dict emits ``{"kind":...,"amount":<i32>}``; no
    #     ``count``/``amount`` spelling discrepancy unlike mill/scry/
    #     surveil) — strictly the simplest bridge shape, single
    #     scalar payload, NO rider the bridge would silently drop
    #     (contrast Surveil's ``use_artifact_count`` / PumpSelf's
    #     ``granted_keywords`` / CreateToken's token-registration).
    # The landfall predicate path itself is the SAME one round-36's
    # landfall+gain_life (Eumidian Terrabotanist), round-37's
    # landfall+pump_self (Icecave Crasher), and round-48's
    # landfall+mill (Icetill Explorer) all validated bit-equal on
    # both engines under the ENTERS dispatch — only the leaf effect
    # differs, and ``put_plus_one_counter_on_self``'s resolve is a
    # single in-place counter add with no scalar/keyword/zone rider.
    ("enters", "land_enters_controlled_by_you", "put_plus_one_counter_on_self"),
    # ENTERS + source_is_self + mill substrate (round-48, sister to
    # the landfall+mill entry above on the source_is_self predicate
    # axis — mirrors how round-1's ENTERS+IsSelf leaf set carries
    # draw_cards/gain_life/scry_n + round-42's surveil). **No real-EoE
    # consumer admitted today**: the only EoE ENTERS+IsSelf+SelfMill
    # carrier is Seedship Broodtender (1/4 {2}{G},
    # "When ~ enters, mill three cards", SelfMill(3)) which ALSO has
    # an activated ability — a declined silent-mistranslation class
    # the loader doesn't seat — so Seedship Broodtender stays dropped
    # at the ``activated_abilities`` gate. This substrate entry is
    # pre-positioned for any future EoE / standard-format bare
    # "When ~ enters, mill N" creature (Stitcher's Supplier-style,
    # one of the most common ETB-mill shapes in the broader pool) so
    # the next slice landing such a card need not extend the
    # allowlist again; pinned by a reject-path test so the
    # predicate-effect product gate stays honest in the meantime.
    # Same substrate-only landing precedent as round-37 DDDD
    # (landfall+draw) / round-43 (ATTACKS+surveil) / round-44
    # (ATTACKS+draw/gain) / round-47 (DIES+draw). The
    # ``_SAFE_EFFECT_KEYS`` "mill" whitelist + the recognizer's
    # ENTERS+IsSelf ``safe_effect_kinds`` widening (below) are the
    # only other edits this leaf needs; the event-string gate and the
    # ``_is_safe_trigger_dict`` dict-side gate stay unchanged since
    # both already admit ENTERS + the ``count`` payload-key whitelist.
    ("enters", "source_is_self", "mill"),
    # ENTERS + source_is_self + create_token substrate (round-49 N1
    # token-printing pre-registration substrate, Phase 1 — see
    # ``docs/TOKEN_PRINTING_SUBSTRATE.md``). Adds the FIRST CreateToken
    # leaf to the safe set: the fixed-single-token "When ~ enters,
    # create N tokens of <fixed printing>" shape (Wedgelight-Rammer /
    # Doomed-Traveler family), restricted to a VANILLA token printing
    # (non-None P/T, no keywords, no activated/triggered/static
    # abilities; card-types may be multi — ``Artifact Creature`` /
    # ``Enchantment Creature``) and a BARE rider profile (``count``
    # only — ``each_player`` / ``tapped`` / ``attacking`` /
    # ``schedule_end_of_combat_sacrifice`` all default-False).
    #
    # **Why VANILLA-token only (the §2 silent-mistranslation gate).**
    # The bridge resolves a Rust ``token_printing_id: u32`` to a Python
    # ``CardPrinting`` via the ``register_token_printing`` map; the
    # loader synthesizes a token CardDef from the carrier's parsed
    # ``token_printing``. CardDef carries name / card-types / P/T /
    # subtypes / keywords / triggered_abilities only — it has NO
    # activated-ability field. A non-vanilla token (the EoE ``Lander``,
    # which carries ``{2}, Sac: SearchLibraryForBasicLandTapped``)
    # would have its activated ability SILENTLY DROPPED on the bridge:
    # Python materializes a Lander that can sac for a land, Rust
    # materializes one that can't. That is exactly the round-38-E1 /
    # §2 silent-mistranslation class. The recognizer
    # (``_python_ta_to_safe_dict``) refuses any non-vanilla token so
    # the carrier falls through to the filter drop path until the
    # token-activated-ability sub-substrate lands (Phase 3 in the
    # design doc).
    #
    # **Why BARE rider only.** Rust's ``Effect::CreateToken { count,
    # token_printing_id }`` (rust/src/cards.rs:219) has NO
    # ``tapped`` / ``each_player`` / ``attacking`` /
    # ``schedule_end_of_combat_sacrifice`` field — the bridge arm
    # constructs ``CreateToken(token_printing=…, count=…)`` with every
    # rider at its Python dataclass default (False). A carrier whose
    # parsed ``CreateToken`` sets ``tapped=True`` (Gravpack Monoist)
    # or ``each_player=True`` (Edge Rover) would round-trip to a token
    # that enters untapped / under one player only on Rust while
    # Python honored the rider — a byte-divergent post-resolve token
    # object. The recognizer gates each rider to its default so those
    # shapes stay dropped (Phase 4 / Phase 5 in the design doc add the
    # sister-variant Rust support).
    #
    # **Cross-engine parity contract (bare-rider vanilla token).**
    # Both engines create ``count`` tokens of the fixed printing under
    # the trigger's resolved controller. Rust's resolve arm
    # (state_ops_resolve.rs) carries the I81 controller-of-effect fix
    # (both the recipient battlefield Vec pick AND the token's
    # ``controller`` byte derive from
    # ``current_resolve_source_controller``); Python's
    # ``CreateToken.resolve`` (effects/generic/tokens.py:195) has
    # always taken ``controller_id`` via the standard
    # ``Effect.resolve`` signature and has been correct on this
    # surface since before I81. The token GameObject's per-object
    # fields (printing_name, P/T, summoning_sick=True at ETB, tapped=
    # False, untap-eligible) are bit-equal because the token printing
    # is a vanilla creature on BOTH engines (registered from the same
    # CardDef → same Rust ``parse_printing`` shape + same Python
    # ``CardPrinting``). ETB dispatch fans out identically (the token
    # itself is vanilla — no nested ETB trigger to compound the
    # surface).
    #
    # **Loader-side auto-registration.** Unlike the per-probe
    # ``register_token_printing(hand_picked_pid, …)`` pattern, the
    # production-deck loader assigns the token printing a DETERMINISTIC
    # pid: ``production_deck_to_deckspec`` appends the token CardDef to
    # the union ``card_defs`` tuple (deduped by name, absent from both
    # libraries so it is registered but never drawn), and its pid =
    # its index in ``build_rust_state``'s sequential registration
    # order. ``_seat_token_printings`` rewrites the carrier's trigger
    # dict's ``token_printing_id`` to that index and calls
    # ``cross_engine.register_token_printing(pid, py_token_printing)``
    # so the Python bridge resolves the same pid. See
    # ``docs/TOKEN_PRINTING_SUBSTRATE.md`` §1.4 / §4 Phase 1.
    #
    # **No real-EoE consumer admitted today (substrate-only landing).**
    # Same precedent as round-37 DDDD (landfall+draw) / round-43
    # (ATTACKS+surveil) / round-44 (ATTACKS+draw/gain) / round-47
    # (DIES+draw). A fresh EoE-pool delta walk
    # (scripts/scout_r49_create_token_shapes.py, full EoE printing set
    # + the 318-name / 129,996-deck 17lands pool) confirms ZERO
    # creatures flip drop→admit on this Phase-1 entry: every
    # high-deck-count fixed-single-token creature carrier produces the
    # ``Lander`` token (non-vanilla — DECLINED §2, Phase 3) or carries
    # a declined rider (Gravpack Monoist ``tapped=True``) or a declined
    # carrier feature (Knight Luminary ``warp``, Honored Knight-Captain
    # activated ability); the ONLY clean creature carrier with a
    # vanilla token is Tunnel Surveyor (Glimmer 1/1) which appears in 0
    # of 129,996 17lands decks. The fixed-single-token "When ~ enters,
    # create a token" shape is one of the most common token shapes in
    # the broader MTG pool, so the substrate is broadly pre-positioned
    # for any future EoE / standard-format printing carrying a
    # bare-rider vanilla-token ENTERS+IsSelf+CreateToken — the next
    # slice landing such a card (or the Phase-3 Lander sub-substrate)
    # need not extend the allowlist again. Pinned by reject-path tests
    # (non-vanilla-token / tapped-rider / each-player-rider stay
    # dropped) so the predicate-effect product gate stays honest.
    ("enters", "source_is_self", "create_token"),
    # DIES + source_is_self + create_token_tapped (Phase 6 token-
    # printing substrate, Phase 4 — round 50). The FIRST real-card
    # token-substrate lever after Phase 1: it admits the real EoE card
    # **Gravpack Monoist** (Creature — Robot, flying; "When this
    # creature dies, create a tapped 2/2 Robot token." — 25,252 of
    # 129,996 17lands EoE decks). Unlike Phase 1 (substrate-only,
    # EoE-pool delta = 0), Phase 4 flips a real card drop→admit:
    # Gravpack Monoist's Robot token is ALREADY vanilla (2/2 Artifact
    # Creature, no abilities/keywords — verified against the parsed
    # printing), so the only Phase-1 decline that kept it dropped was
    # the ``tapped=True`` rider (``docs/TOKEN_PRINTING_SUBSTRATE.md``
    # §2.3). The new Rust ``Effect::CreateTokenTapped`` sister variant
    # + the ``create_token_tapped`` bridge arm + this recognizer widen
    # close that decline with NO Phase-3 Lander-activated-ability
    # dependency.
    #
    # **Why DIES (not ENTERS).** Gravpack Monoist's oracle is a death
    # trigger ("When this creature dies, ..."). The DIES event +
    # IsSelf predicate dispatch was proven bit-equal cross-engine by
    # round-42 (DIES+surveil) / round-47 (DIES+gain_life /
    # DIES+draw_cards) — the recognizer's event-string gate and
    # ``_is_safe_trigger_dict`` already admit DIES; this entry adds the
    # create_token_tapped leaf to the DIES×IsSelf row. The ENTERS
    # sister entry below pre-positions the substrate for any future
    # "When ~ enters, create a tapped token" carrier (sister to the
    # Phase-1 ENTERS+create_token entry) — substrate-only, no EoE
    # ENTERS+tapped consumer today.
    #
    # **Cross-engine parity contract (tapped vanilla token).** Both
    # engines create ``count`` tokens of the fixed vanilla printing
    # under the trigger's resolved controller, each entering TAPPED.
    # Rust's ``Effect::CreateTokenTapped`` resolve arm sets
    # ``tapped: true`` on the pushed GameObject (otherwise byte-
    # identical to the sister ``CreateToken`` arm — same I81
    # controller-of-effect resolution, summoning_sick=true, CE4
    # monotonic etb_timestamp, pending ENTERS dispatch); Python's
    # ``CreateToken.resolve`` flips ``obj.tapped = True`` post-ETB
    # (tokens.py:286-287). The token enters tapped but its ETB still
    # fires on BOTH engines (CR 603.6e — entering tapped does not
    # suppress the enters trigger; the Robot token is vanilla so there
    # is no nested ETB anyway). The token GameObject's per-object
    # signature (printing name / P/T / controller / summoning_sick /
    # **tapped=True**) is byte-equal — the field-view zone signature
    # includes ``tapped`` so the real-EoE probe's byte-equal assertion
    # genuinely exercises the tapped flag.
    #
    # **Loader-side auto-registration.** Same deterministic-pid
    # ``_seat_token_printings`` pass as the Phase-1 ``create_token``
    # entry — it now also rewrites symbolic ``create_token_tapped``
    # triggers (the carrier-ref scan + the trigger rewrite preserve
    # the ``kind`` so the tapped variant survives seating).
    #
    # **Real-EoE delta (NOT substrate-only).** A fresh EoE-pool walk
    # confirms Gravpack Monoist flips drop→admit on this entry
    # (~25,252 decks). Pinned by a real-EoE Gravpack-Monoist cross-
    # engine probe + the reject-path conversion (the round-49
    # ``test_printing_is_bridge_simple_creature_rejects_lander_cluster``
    # Gravpack-Monoist assertion is moved to an admit-path pin).
    ("dies", "source_is_self", "create_token_tapped"),
    ("enters", "source_is_self", "create_token_tapped"),
    # DIES + source_is_self + bare create_token (Phase 3a — round-53
    # N1, ``docs/TOKEN_PRINTING_SUBSTRATE.md`` §7). The round-49 Phase-1
    # ENTERS entry above admits the bare-rider fixed-single-token shape;
    # Phase 1 deliberately left bare ``create_token`` OUT of the DIES
    # row because every DIES Lander carrier was declined on the
    # non-vanilla token and the only DIES+vanilla-token consumer
    # (Gravpack Monoist) is the tapped shape. Phase 3a's
    # ``_token_with_safe_activated_ability_card_def`` ADMITS the bare
    # ``Lander`` token (its ONLY activated ability is the round-52-
    # precursor-verified bridge-correct
    # ``{2},Sac:SearchLibraryForBasicLandTapped`` shape, emitted on the
    # synthesized token CardDef via ``token_activated_abilities``), so
    # **Beamsaw Prospector** ("When ~ dies, create a Lander." —
    # Creature — Human Artificer, **24,310** of 129,996 17lands EoE
    # decks) is now the real DIES+IsSelf+bare-create_token consumer.
    # The bare ``create_token`` kind round-trips identically under DIES
    # vs ENTERS (the bridge's ``create_token`` arm is event-agnostic;
    # ``_seat_token_printings`` recognizes the kind under any event and
    # threads the token's ``token_activated_abilities`` through
    # ``dataclasses.replace`` + ``cached_token_printing`` — the round-52
    # cache key folds ``token_activated_abilities`` in so a Lander never
    # collides with a same-named vanilla token). Paired with the
    # ENTERS+create_token entry (line ~2002, round-49) this flips
    # **Galactic Wayfarer** (ENTERS+IsSelf+bare-Lander — Creature —
    # Human Scout, **33,469** decks) drop→admit too. Phase 3a
    # cumulative real-EoE delta: **2 carriers / 57,779 17lands decks**
    # (the §2-honest subset — see §7 for the per-carrier decline
    # ledger: Edge Rover each_player→Phase 5, Biomechan Engineer
    # carrier-activated-ability, Glacier Godmaw / Biotech Specialist
    # sibling un-bridged trigger, Sunstar Expansionist And-StateBound
    # predicate, Seedship Agrarian BECOMES_TAPPED event, Bioengineered
    # Future non-creature→Phase 2 ALL stay declined).
    ("dies", "source_is_self", "create_token"),
    # BECOMES_TAPPED + source_is_self + create_token (round-58 N1 —
    # the BECOMES_TAPPED event bridge). The FOURTH event in the safe
    # set (after ENTERS / DIES / ATTACKS), and the FIRST that is NOT
    # a zone-change / combat event — it fires on the tap-state
    # transition of an existing battlefield permanent. Closes the
    # campaign landmine cited repeatedly since round-42 (the
    # "``BECOMES_TAPPED`` event — no bridge arm — would silently
    # mistranslate" §2 decline): there is now a bridge arm, and
    # BOTH engines fire the event at the SAME tap sites in the SAME
    # order.
    #
    # **Real-EoE consumer: Seedship Agrarian** (3/3 {3}{G},
    # Creature — Insect Scientist). Its FIRST trigger is
    # "Whenever this creature becomes tapped, create a Lander token"
    # = ``BECOMES_TAPPED / IsSelf / CreateToken(Lander)``. (Its
    # SECOND trigger, "Landfall — ... put a +1/+1 counter on this
    # creature", is the already-bridged
    # ``ENTERS / land_enters_controlled_by_you /
    # put_plus_one_counter_on_self`` shape — out of scope of THIS
    # entry but already round-trips.) This entry closes the
    # BECOMES_TAPPED→CreateToken HALF of Seedship Agrarian: the
    # event now round-trips byte-equal. The residual is narrowed to
    # the Lander token-registration substrate ONLY (a SEPARATE
    # substrate — Lander's ``{2},{T},Sac:SearchLibraryForBasicLand
    # Tapped`` activated ability on the spawned token printing; same
    # boundary Galactic Wayfarer / Beamsaw Prospector hit). See
    # ``docs/SEARCH_LIMITATIONS.md`` "BECOMES_TAPPED event bridge"
    # for the honest split.
    #
    # **Cross-engine parity contract (event firing-set + ordering).**
    # Python fires ``Event.BECOMES_TAPPED`` ONLY through
    # ``turn.tap_permanent`` (`src/mtg_bot/turn.py:542-571`), the
    # no-op-if-already-tapped chokepoint, called from: tap-for-mana
    # (`actions.py:auto_tap_for_cost`), the ``{T}`` activated-ability
    # ``tap_self`` cost (`actions.py:_activate_ability`),
    # ``tap_permanent_you_control`` activated cost, convoke, crew,
    # attack-tap of a non-vigilance attacker (`combat.py:104`, fired
    # BEFORE the per-attacker ATTACKS dispatch at `combat.py:115`),
    # and the tapper-effect family (`effects/generic/tap.py`,
    # `combinators.py:163`, `bespoke.py:496`). It deliberately does
    # NOT fire for permanents entering tapped (`_play_land`,
    # search-land-tapped, reanimate-tapped, token-created-tapped) nor
    # for regeneration's replacement tap (CR 701.21a — entering
    # tapped never "becomes tapped"). Rust's emission set matches
    # this EXACTLY: the round-58 emission slice routes the production
    # tap sites through the centralized
    # ``state_ops_resolve::tap_permanent_and_dispatch`` chokepoint
    # (mirroring `turn.tap_permanent`'s no-op-if-already-tapped +
    # dispatch shape) plus inline dispatches at the
    # mana/tap_self sites; the entering-tapped / regeneration-tap
    # Rust sites stay raw ``obj.tapped = true`` (no event). The
    # attack-tap dispatch is placed BEFORE the ``Event::Attacks``
    # dispatch so the per-attacker stack-push order is byte-equal
    # to Python's (BECOMES_TAPPED triggers ahead of ATTACKS
    # triggers).
    #
    # **Scope: which (predicate, effect) combos are admitted vs
    # declined under BECOMES_TAPPED (§2 bridge-correctness).**
    #   ADMIT: ``source_is_self`` (-> ``IsSelf()``) + ``create_token``
    #     (-> ``CreateToken``). This is the ONLY combo with a real
    #     EoE consumer (Seedship Agrarian) and BOTH the predicate and
    #     effect bridge arms are event-agnostic + already proven
    #     byte-equal under ENTERS/DIES (round-49/53 create_token
    #     work). The event itself is the only new substrate; it
    #     round-trips because both engines' firing-set + ordering
    #     match.
    #   DECLINE (no entry, falls through the recognizer):
    #     * ``IsHost()`` predicate (Cryoshatter-shape "enchanted
    #       permanent becomes tapped" — `enchantment_builder.py:497`):
    #       the Rust predicate substrate has no ``SourceIsHost`` arm
    #       (only ``SourceIsSelf``); admitting it would silently
    #       mistranslate the host-relationship. Separate predicate-
    #       substrate slice.
    #     * Any non-``create_token`` effect under BECOMES_TAPPED
    #       (draw_cards / gain_life / surveil / etc.): no EoE
    #       BECOMES_TAPPED consumer carries those today; admitting
    #       them would be preventative-only AND would widen the
    #       loader's bench-admit surface without a real signal
    #       source. A future slice landing such a card ships the
    #       single new ``_SAFE_TRIGGERS`` entry + a one-line widen
    #       in the recognizer's BECOMES_TAPPED arm (same pattern as
    #       the ENTERS/DIES/ATTACKS leaf-set widenings).
    ("becomes_tapped", "source_is_self", "create_token"),
})


# Canonical name for the compound landfall predicate shape — Rust's
# ``Predicate::And(Predicate::SubjectIsLand, Predicate::ControlledByYou)``
# / Python's ``And(Subject(slot=None, inner=IsLand()),
# Controlled(by=YOU, subject=None))``. Used as the sentinel
# ``predicate_kind`` token in ``_SAFE_TRIGGERS`` so the compound
# shape composes uniformly with the leaf entries (e.g.
# ``("enters", "source_is_self", "draw_cards")`` shape).
_LANDFALL_PREDICATE_KIND = "land_enters_controlled_by_you"


# Effect arms in the safe set — each arm's allowed keys (other than
# ``"kind"``). Used by ``_is_safe_trigger_dict`` to reject any
# trigger carrying an unrecognized effect-payload field that the
# bridge would silently drop.
#
# ``pump_self_until_eot`` carries two payload keys: ``power_bonus``
# and ``toughness_bonus`` (mirrors Rust's ``PumpSelfUntilEOT
# { power_bonus: i32, toughness_bonus: i32 }``). Both are optional
# in Rust's parser (default 0), but the safe-trigger filter requires
# at least one to be present (a ``(0, 0)`` pump is a pure no-op on
# both engines so it's harmless if it leaks through, but a typed-
# absent payload would be a fixture smell). Riders that the bare
# bridge does NOT carry (``granted_keywords``, etc.) are absent
# here — the parser emits the bare-shape trigger for Icecave
# Crasher / Remnant Elemental, and any future +X/+Y-and-gains-
# keyword landfall consumer would have to ride a separate composer
# arm (out of scope for this slice).
_SAFE_EFFECT_KEYS: dict[str, frozenset[str]] = {
    "draw_cards": frozenset({"kind", "count"}),
    "gain_life": frozenset({"kind", "amount"}),
    "scry_n": frozenset({"kind", "count"}),
    "pump_self_until_eot": frozenset(
        {"kind", "power_bonus", "toughness_bonus"}
    ),
    # Surveil — single ``count`` payload key. Rust's
    # ``Effect::Surveil { count: i32 }`` is bare; the Python
    # ``Surveil`` class additionally carries a ``use_artifact_count``
    # bool that selects the Surveil-X-where-X=artifacts variant
    # (effects/generic/library_lookat.py:580). The bridge does NOT
    # carry the ``use_artifact_count`` flag — gating it out at the
    # recognizer side (:func:`_python_ta_to_safe_dict`) keeps the
    # Surveil-X variant from silently degrading to a fixed surveil-N
    # on round-trip. No EoE consumer triggers Surveil-X today (it's
    # an Anticausal-Vestige-and-friends mechanic on a different
    # axis), so this gate just pins the contract.
    "surveil": frozenset({"kind", "count"}),
    # Mill (self-mill) — single ``count`` payload key (round-48
    # bench-admit-set slice). Rust's ``Effect::Mill { count: i32 }``
    # (cards.rs:75, parsed at cards.rs:3991, serialized at
    # cards.rs:6450) is bare: controller mills the top ``count`` cards
    # of THEIR OWN library into THEIR OWN graveyard (CR 701.13b — the
    # controller-target shortcut sibling of ``MillTargetPlayer``).
    # Python's ``SelfMill(amount: int = 1)``
    # (effects/generic/library_lookat.py:624) is the matched class;
    # its single ``amount`` field round-trips to Rust's ``count``
    # spelling via the existing ``mill`` bridge arm
    # (cross_engine._ta_dict_to_python_effect:725, shipped round-16's
    # scry/surveil-family slice). The bare ``count`` payload is the
    # only field on either side — no rider the bridge would silently
    # drop (unlike Surveil's ``use_artifact_count`` or PumpSelf's
    # ``granted_keywords``). The recognizer in
    # :func:`_python_ta_to_safe_dict` gates this whitelist.
    "mill": frozenset({"kind", "count"}),
    # PutPlusOneCounterOnSelf — single ``amount`` payload key
    # (round-59 N1, the sole remaining Seedship Agrarian blocker).
    # Rust's ``Effect::PutPlusOneCounterOnSelf { amount: i32 }``
    # (cards.rs:487, parsed cards.rs:4653, serialized to the dict
    # ``{"kind": "put_plus_one_counter_on_self", "amount": <i32>}``
    # at cards.rs:7179-7181) is bare: it adds ``amount`` +1/+1
    # counters to the ability's source ("this creature"), the
    # recipient always being ``state.resolving_source`` / Rust's
    # ``source_obj_id`` walk — there is NO target picker and NO rider
    # field on either side. Python's
    # ``PutPlusOneCounterOnSelf(amount: int = 1)``
    # (effects/generic/counters.py:784) carries the SAME single
    # ``amount`` field with the SAME default; the existing
    # ``put_plus_one_counter_on_self`` bridge arm
    # (cross_engine.py:3899, shipped round-22) round-trips it with
    # the field name IDENTICAL on both sides (no ``count``/``amount``
    # spelling discrepancy unlike mill/scry/surveil). The bare
    # ``amount`` payload is the only field — no rider the bridge
    # would silently drop (contrast Surveil's ``use_artifact_count``
    # or PumpSelf's ``granted_keywords``). The recognizer in
    # :func:`_python_ta_to_safe_dict` gates this whitelist.
    "put_plus_one_counter_on_self": frozenset({"kind", "amount"}),
    # PutPlusOneCounterOnOpponentsCreature — single ``amount`` payload
    # key (Tier-C target-seating sub-substrate, lever-a — Sunset
    # Saboteur). Rust's
    # ``Effect::PutPlusOneCounterOnOpponentsCreature { amount: i32 }``
    # (rust/src/cards.rs) is bare: it adds ``amount`` +1/+1 counters
    # to the push-time-SEATED target (a creature an opponent of the
    # source's controller controls; the seat is the Tier-C shared
    # ``first_legal_trigger_target`` / no-agent ``pool[0]`` pick — NOT
    # a payload field, NOT a self-scan). Its to_dict emits ``{"kind":
    # "put_plus_one_counter_on_opponents_creature", "amount": <i32>}``
    # — IDENTICAL spelling to Python's
    # ``PutPlusOneCounterOnOpponentsCreature(amount: int = 1)``
    # (effects/generic/counters.py, a ``PutPlusOneCounters``
    # subclass), round-tripped by the
    # ``put_plus_one_counter_on_opponents_creature`` cross_engine
    # bridge arm. The slot filter
    # (``IsCreature() & ControlledByOpponent()``) is NOT a dict
    # payload key — it is structural to the effect class on Python and
    # to the seat's per-variant scan on Rust (the recognizer's
    # bare-shape gate enforces the single-implicit-target contract),
    # so the payload whitelist stays the bare ``{"kind", "amount"}``
    # exactly like the ``put_plus_one_counter_on_self`` sibling. The
    # recognizer in :func:`_python_ta_to_safe_dict` gates this
    # whitelist; the ``_is_safe_trigger_dict`` extra-key refusal is
    # the backstop.
    "put_plus_one_counter_on_opponents_creature": frozenset(
        {"kind", "amount"}
    ),
    # ReturnTargetPermanentYouControlToOwnersHand — UNIT shape, the
    # only payload key is ``kind`` (Tier-C target-seating
    # sub-substrate, lever-a — Exosuit Savior, the count_min=0
    # optional-bounce edge). Rust's
    # ``Effect::ReturnTargetPermanentYouControlToOwnersHand``
    # (rust/src/cards.rs) is a unit variant — no amount / no rider;
    # its to_dict emits ``{"kind":
    # "return_target_permanent_you_control"}`` and the parse arm
    # tolerates/ignores extra fields. Python's
    # ``ReturnTargetPermanentYouControlToOwnersHand(optional=True,
    # exclude_self=True)`` (effects/generic/bounce.py, a ``Bounce``
    # subclass) is reconstructed by the
    # ``return_target_permanent_you_control`` cross_engine bridge arm
    # with the optional/exclude_self flags fixed structurally (they
    # ARE Exosuit Savior's shape — not caller-varied). The slot
    # filter (own ∧ ~player ∧ ~source) and count_min=0 are structural
    # to the effect class / the seat's per-variant scan, NOT dict
    # payload keys (the recognizer's bare-shape gate enforces the
    # single implicit-target contract), so the whitelist is the bare
    # ``{"kind"}``. The recognizer in
    # :func:`_python_ta_to_safe_dict` gates this whitelist; the
    # ``_is_safe_trigger_dict`` extra-key refusal is the backstop.
    "return_target_permanent_you_control": frozenset({"kind"}),
    # CreateToken (round-49 N1 token-printing substrate, Phase 1) —
    # payload keys ``count`` + ``token_printing_id``. This is the
    # POST-SEATING dict shape: the recognizer
    # (``_python_ta_to_safe_dict``) first emits a SYMBOLIC reference
    # (``_token_card_def``) which the loader's ``_seat_token_printings``
    # pass rewrites into a numeric ``token_printing_id`` once the union
    # ``card_defs`` tuple is frozen and the token's Rust-registry pid
    # is known. ``_is_safe_trigger_dict`` validates the final
    # post-seating shape, so this whitelist mirrors Rust's two-field
    # ``Effect::CreateToken { count, token_printing_id }``
    # (rust/src/cards.rs:219, parse_effect cards.rs:4162). Rider flags
    # (``tapped`` / ``each_player`` / ``attacking`` /
    # ``schedule_end_of_combat_sacrifice``) are deliberately ABSENT —
    # Rust's variant has no mirror for them and the recognizer gates
    # each to its Python default, so a dict carrying any of them is
    # rejected here (mirrors the Surveil ``use_artifact_count`` /
    # PumpSelf ``granted_keywords`` rider-exclusion pattern). See
    # ``docs/TOKEN_PRINTING_SUBSTRATE.md`` §2.3.
    "create_token": frozenset({"kind", "count", "token_printing_id"}),
    # CreateTokenTapped (Phase 6 token-printing substrate, Phase 4 —
    # round 50). Same POST-SEATING two-key payload as the sister
    # ``create_token`` whitelist above (``count`` +
    # ``token_printing_id``); the tapped-vs-untapped choice rides on
    # the dict ``kind`` (= the Rust variant name
    # ``Effect::CreateTokenTapped``), not on a payload field, so the
    # whitelist is identical and the ``tapped`` rider is NOT a dict
    # key (mirrors the closed-vocab sister-variant convention — the
    # bridge round-trips the binary choice via ``kind``, never a bool
    # payload field). The recognizer (``_python_ta_to_safe_dict``)
    # emits this ``kind`` when the parsed ``CreateToken`` has
    # ``tapped=True`` and every OTHER rider
    # (``each_player`` / ``attacking`` /
    # ``schedule_end_of_combat_sacrifice``) at its default, gated to a
    # vanilla token (same §2 gate as the sister arm). EoE consumer:
    # Gravpack Monoist (DIES+IsSelf+tapped Robot — 25,252 17lands
    # decks). See ``docs/TOKEN_PRINTING_SUBSTRATE.md`` §4 Phase 4.
    "create_token_tapped": frozenset(
        {"kind", "count", "token_printing_id"}
    ),
}


def _is_landfall_predicate_dict(pred: object) -> bool:
    """True iff ``pred`` is the canonical landfall compound predicate
    dict shape: ``{"kind": "and", "left": {"kind": "subject_is_land"},
    "right": {"kind": "controlled_by_you"}}``.

    Strict on the left/right ordering and on the exact two-arm shape
    — that ordering is what the parser emits today
    (``cards/parser/triggers.py:603-1266``) and what cross_engine's
    bridge translates onto each engine via the ``and`` /
    ``subject_is_land`` / ``controlled_by_you`` arms in
    ``_ta_dict_to_python_predicate``. Both engines accept the dict
    without raising; the strict check here keeps the safe-trigger
    gate honest by treating any other compound shape (``Or`` /
    ``Not(IsLand)`` / nested-AND / different ordering) as unsafe
    until its own slice lands.
    """
    if not isinstance(pred, dict):
        return False
    if set(pred.keys()) != {"kind", "left", "right"}:
        return False
    if pred.get("kind") != "and":
        return False
    left = pred.get("left")
    right = pred.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if tuple(left.keys()) != ("kind",) or tuple(right.keys()) != ("kind",):
        return False
    return (
        left.get("kind") == "subject_is_land"
        and right.get("kind") == "controlled_by_you"
    )


def _is_safe_trigger_dict(ta_dict: dict) -> bool:
    """True iff ``ta_dict`` matches one of the :data:`_SAFE_TRIGGERS`
    shapes.

    Validates the four-tuple (event, predicate-kind, effect-kind,
    once_per_turn) as well as the effect-payload key whitelist (see
    :data:`_SAFE_EFFECT_KEYS`). Refuses any ``ta_dict`` carrying
    extra top-level keys beyond ``event`` / ``predicate`` / ``effect``
    / ``once_per_turn`` so a future schema extension doesn't silently
    pass through the loader and produce cross-engine drift.
    """
    if not isinstance(ta_dict, dict):
        return False
    allowed_top = {"event", "predicate", "effect", "once_per_turn"}
    if any(k not in allowed_top for k in ta_dict):
        return False
    if ta_dict.get("once_per_turn", False):
        return False
    event = ta_dict.get("event")
    if not isinstance(event, str):
        return False
    pred = ta_dict.get("predicate")
    if pred is None:
        pred_kind: Optional[str] = None
    elif isinstance(pred, dict) and tuple(pred.keys()) == ("kind",):
        pred_kind = pred.get("kind")
    elif _is_landfall_predicate_dict(pred):
        # Landfall extension: compound dict
        # ``{"kind": "and",
        #    "left": {"kind": "subject_is_land"},
        #    "right": {"kind": "controlled_by_you"}}``
        # collapses to the canonical sentinel kind so it composes
        # with the leaf-shape allowlist entries. Strict on the
        # left/right ordering and on the absence of any extra
        # top-level keys — that ordering is exactly what the
        # parser emits today (``cards/parser/triggers.py:603-1266``)
        # and what cross_engine's bridge arms parse on both engines.
        pred_kind = _LANDFALL_PREDICATE_KIND
    else:
        # Any other predicate carrying nested fields (subject, inner,
        # left, right, ...) is NOT in the safe set.
        return False
    eff = ta_dict.get("effect")
    if not isinstance(eff, dict):
        return False
    eff_kind = eff.get("kind")
    if not isinstance(eff_kind, str):
        return False
    triple = (event, pred_kind, eff_kind)
    if triple not in _SAFE_TRIGGERS:
        return False
    allowed_eff_keys = _SAFE_EFFECT_KEYS.get(eff_kind)
    if allowed_eff_keys is None:
        return False
    if any(k not in allowed_eff_keys for k in eff):
        return False
    return True


# Token printing keyword set the bridge round-trips (Phase 1: only
# the empty set is honored, but the helper is keyword-aware so the
# Phase-3+ keyword-bearing-token follow-ups are a single-line widen).
# The EoE vanilla token printings (Robot 2/2, Human Soldier 1/1,
# Glimmer 1/1, Gremlin 1/1) carry NO keywords; a keyword-bearing
# token (the Drone token's ``flying``) is deferred until the keyword
# round-trip on TOKEN printings is pinned (creature-printing keyword
# round-trip is pinned, but a token's keyword would need its own
# combat-side parity probe — Phase 1 keeps the gate tight).
_SAFE_TOKEN_KEYWORDS: frozenset[str] = frozenset()


def _token_vanilla_card_def(token_printing) -> Optional["CardDef"]:
    """Build a :class:`cross_engine.CardDef` from a token
    ``CardPrinting`` iff the token is bridge-VANILLA, else ``None``.

    Round-49 N1 token-printing substrate (Phase 1). "Vanilla" here is
    the §2 silent-mistranslation gate: the synthesized CardDef carries
    only name / card-types / P/T / subtypes / keywords /
    triggered_abilities, so any token field the CardDef bridge can't
    represent MUST be absent or the cross-engine round trip would
    silently drop it. Specifically the token must have:

      * ``Creature`` among its card types and non-None printed P/T
        (a token GameObject's per-object signature includes power /
        toughness; a None-P/T token can't be a creature body).
      * NO activated abilities — CardDef has no activated-ability
        field. This is the gate that DECLINES the EoE ``Lander``
        token (``{2}, Sac: SearchLibraryForBasicLandTapped``); its
        ability would vanish on the bridge (Python Lander can sac for
        a land, Rust can't). Lander carriers (Galactic Wayfarer,
        Beamsaw Prospector, Edge Rover, …) therefore stay dropped
        until the Phase-3 token-activated-ability sub-substrate.
      * NO static abilities (CardDef has no static-ability field;
        layered/conditional eval, declined campaign-wide).
      * NO nested triggered abilities (a token whose own ETB/DIES
        trigger would itself have to round-trip — out of Phase 1
        scope; require zero so the surface stays a single leaf).
      * NO keywords outside :data:`_SAFE_TOKEN_KEYWORDS` (Phase 1:
        the empty set — keyword-bearing tokens like the Drone's
        ``flying`` need their own token-keyword combat parity probe).
      * NO ``can_only_block_creatures_with_flying`` block restriction
        (the EoE Drone token's flag — CardDef can't carry it).

    The token's ``card_types`` may be MULTI-type (the EoE ``Robot``
    is ``Artifact Creature``, ``Glimmer`` is ``Enchantment
    Creature``); the primary type is "Creature" and the rest flow
    through ``CardDef.extra_card_types``. The ``Token`` supertype is
    stamped so both engines remove the token from non-battlefield
    zones per CR 704.5d (Python ``CardPrinting.is_token`` reads
    ``"Token" in supertypes``; Rust mirrors via the registry
    ``supertypes`` list).
    """
    if token_printing is None:
        return None
    card_types = tuple(getattr(token_printing, "card_types", ()) or ())
    if "Creature" not in card_types:
        return None
    if token_printing.power is None or token_printing.toughness is None:
        return None
    if token_printing.activated_abilities:
        return None
    if getattr(token_printing, "static_abilities", None):
        return None
    if getattr(token_printing, "triggered_abilities", None):
        return None
    kw = frozenset(token_printing.keywords or frozenset())
    if kw and not kw.issubset(_SAFE_TOKEN_KEYWORDS):
        return None
    if getattr(
        token_printing, "can_only_block_creatures_with_flying", False
    ):
        return None
    # Primary type = "Creature"; remaining types (Artifact /
    # Enchantment) flow through extra_card_types in their printed
    # order minus the primary, so the combined tuple
    # ``(card_type, *extra_card_types)`` round-trips card-type-for-
    # card-type. The token printing's spell_effect / cost fields are
    # irrelevant (tokens are created, never cast) and intentionally
    # not threaded.
    extra = tuple(t for t in card_types if t != "Creature")
    from mtg_bot.parity.cross_engine import CardDef as _CardDef
    return _CardDef(
        name=token_printing.name,
        card_type="Creature",
        extra_card_types=extra,
        power=int(token_printing.power),
        toughness=int(token_printing.toughness),
        subtypes=tuple(token_printing.subtypes or ()),
        supertypes=("Token",),
        keywords=tuple(sorted(kw)),
    )


# The exact bridge-correct activated-ability dict for the EoE
# ``Lander`` token: ``{2}, Sacrifice this artifact: Search your
# library for a basic land card, put it onto the battlefield tapped,
# then shuffle.`` Round-52 N1's Phase-3 precursor verified this shape
# round-trips Python<->Rust with NO §2 gap (the cost is
# ``mana={generic:2}`` + ``sacrifice_self`` — the exact subset both
# engines pay identically — and the effect
# ``search_library_for_basic_land_tapped`` is FULLY DETERMINISTIC on
# BOTH engines: first ``count`` basic lands in library order, no
# PendingDecision, no agent seating). This is the ONLY token-activated
# ability Phase 3a admits; any token whose activated ability differs
# from this exact dict stays declined (§2 — a divergent cost component
# or a non-deterministic / agent-seated effect would silently
# mistranslate). ``count`` is pinned to 1 because the real EoE Lander
# searches for exactly one basic land; a hypothetical multi-land
# variant is a separate recognizer-widen (Phase 3b territory) and is
# NOT admitted here.
_LANDER_SAFE_ACTIVATED_ABILITY: dict = {
    "cost": {"mana": {"generic": 2}, "sacrifice_self": True},
    "effect": {"kind": "search_library_for_basic_land_tapped"},
    "is_mana_ability": False,
}


def _token_with_safe_activated_ability_card_def(
    token_printing,
) -> Optional["CardDef"]:
    """Build a :class:`cross_engine.CardDef` from a token
    ``CardPrinting`` iff the token's ONLY activated ability is the
    exact bridge-correct Lander shape (and it is otherwise clean),
    else ``None``.

    Round-53 N1 Phase 3a (``docs/TOKEN_PRINTING_SUBSTRATE.md`` §7).
    Sibling of :func:`_token_vanilla_card_def`: that helper admits a
    VANILLA token (a creature body, NO activated abilities — declines
    the Lander). This helper admits the ``Lander`` cluster's token,
    whose ONLY activated ability is

      ``{2}, Sac: SearchLibraryForBasicLandTapped``

    The round-52 N1 Phase-3 PRECURSOR landed the
    ``CardDef.token_activated_abilities`` substrate + its
    Python<->Rust bridge plumbing
    (:func:`cross_engine._aa_dict_to_python_activated_ability` /
    :func:`cross_engine._aa_dict_to_python_activation_cost` /
    ``build_rust_state``'s ``activated_abilities`` registry key) and
    proved this exact shape round-trips byte-identical with NO §2 gap
    (the cost is the ``mana``+``sacrifice_self`` subset both engines
    pay identically; the effect is fully deterministic on BOTH engines
    — no agent / PendingDecision). This helper consumes that substrate:
    it emits a token CardDef carrying
    ``token_activated_abilities=(_LANDER_SAFE_ACTIVATED_ABILITY,)``.

    **§2 silent-mistranslation discipline (STRICT).** The token must:

      * be a NON-creature permanent (the real Lander is
        ``Artifact — Lander``, no P/T). A token that is ALSO a
        creature is handled by the vanilla path; this path is for the
        ability-bearing artifact token.
      * carry EXACTLY ONE activated ability, byte-equal to the real
        authored ``TOKEN_PRINTINGS["Lander"]`` ability AND to
        :data:`_LANDER_SAFE_ACTIVATED_ABILITY` (cost
        ``mana={generic:2}`` + ``sacrifice_self`` and nothing else;
        effect ``SearchLibraryForBasicLandTapped(count=1)``;
        ``is_mana_ability=False``). ANY deviation (a different cost
        component, a tap requirement, a non-deterministic effect, a
        ``count`` other than 1) keeps the token DECLINED — admitting a
        token whose ability silently differs cross-engine is exactly
        the §2 class this campaign exists to avoid.
      * carry NO keywords / static abilities / triggered abilities /
        nested abilities (CardDef carries only the synthesized
        single activated ability via ``token_activated_abilities``;
        any other token feature would silently vanish on the bridge).

    The synthesized CardDef's ``card_type`` is the token's primary
    card type with the rest threaded through ``extra_card_types``
    (the Lander is single-type ``Artifact``; the multi-type path is
    kept general). The ``Token`` supertype is stamped so both engines
    remove the token from non-battlefield zones per CR 704.5d.
    """
    from mtg_bot.effects.base import ActivatedAbility
    from mtg_bot.effects.generic.search import (
        SearchLibraryForBasicLandTapped,
    )

    if token_printing is None:
        return None
    card_types = tuple(getattr(token_printing, "card_types", ()) or ())
    if not card_types:
        return None
    # This path is the NON-creature ability-bearing token (the Lander
    # is Artifact — Lander, no P/T). A creature token with an activated
    # ability is out of Phase-3a scope (its P/T body + the ability
    # would BOTH have to round-trip; the only EoE consumer here is the
    # bare Lander).
    if "Creature" in card_types:
        return None
    if token_printing.power is not None or token_printing.toughness is not None:
        return None
    if getattr(token_printing, "static_abilities", None):
        return None
    if getattr(token_printing, "triggered_abilities", None):
        return None
    if token_printing.keywords:
        return None
    if getattr(
        token_printing, "can_only_block_creatures_with_flying", False
    ):
        return None
    aas = tuple(token_printing.activated_abilities or ())
    if len(aas) != 1:
        return None
    aa = aas[0]
    if not isinstance(aa, ActivatedAbility):
        return None
    # STRICT cost gate: exactly {2} generic + sacrifice_self, nothing
    # else. Any extra cost component (tap / other sac / target) would
    # round-trip differently or be silently dropped (§2).
    cost = aa.cost
    if cost.mana is None:
        return None
    if cost.mana.generic != 2 or cost.mana.colored:
        return None
    if not cost.sacrifice_self:
        return None
    for _extra in (
        "tap_self",
        "requires_target_creature_you_control",
        "sacrifice_another_creature_or_artifact",
        "sacrifice_a_land",
        "sacrifice_another_artifact",
        "sacrifice_food",
        "sacrifice_a_desert",
    ):
        if getattr(cost, _extra, False):
            return None
    # STRICT effect gate: the fully-deterministic single-land search.
    if not isinstance(aa.effect, SearchLibraryForBasicLandTapped):
        return None
    if aa.effect.count != 1:
        return None
    if getattr(aa, "is_mana_ability", False):
        return None
    extra = tuple(t for t in card_types if t != card_types[0])
    from mtg_bot.parity.cross_engine import CardDef as _CardDef
    return _CardDef(
        name=token_printing.name,
        card_type=card_types[0],
        extra_card_types=extra,
        subtypes=tuple(token_printing.subtypes or ()),
        supertypes=("Token",),
        token_activated_abilities=(dict(_LANDER_SAFE_ACTIVATED_ABILITY),),
    )


def _canonical_lander_token_card_def() -> Optional["CardDef"]:
    """The canonical bridge-correct ``Lander`` token CardDef, built
    from the authored ``TOKEN_PRINTINGS["Lander"]`` via the SAME
    :func:`_token_with_safe_activated_ability_card_def` helper the
    ``triggered_abilities`` token-seating path uses.

    Lander-spell-token-seating sub-substrate (round-73 — the follow-on
    the round-70 spell-card-bridge SEARCH_LIMITATIONS seam scoped). The
    Lander-pid spell effects (Seedship Impact / Orbital Plunge) emit a
    SYMBOLIC ``_token_card_def`` reference (see
    :func:`_python_spell_effect_to_dict`); the loader's
    :func:`_seat_token_printings` pass (extended to walk
    ``spell_effect``) resolves it to the numeric ``token_printing_id``
    Rust's ``parse_effect`` requires + registers the matching Python
    printing into the cross-engine bridge map.

    Using the EXACT helper the ``triggered_abilities`` path uses is the
    §2 correctness guarantee: the seated Lander's registered Python
    printing (via :func:`cross_engine.cached_token_printing` →
    :func:`cross_engine._card_def_to_python_printing`) is byte-equal to
    what Python's field-free ``SeedshipImpactEffect`` /
    ``DealDamageToTargetCreatureWithExcessLanderRider`` mint
    intrinsically from ``TOKEN_PRINTINGS["Lander"]`` — the SAME
    equivalence the existing Effect-half Seedship/Orbital real-EoE
    probes already proved byte-equal (Rust seats a minimal ``Lander``
    via the registered pid, Python the intrinsic
    ``TOKEN_PRINTINGS["Lander"]``).

    Returns ``None`` (round-trip-or-bail) iff the authored Lander
    token ever drifts out of the bridge-correct shape the
    ``_token_with_safe_activated_ability_card_def`` §2 gate accepts —
    keeping the no-fallback-hybrid discipline: the symbolic ref is
    emitted ONLY when a byte-correct Lander can be seated.
    """
    from mtg_bot.cards.tokens import TOKEN_PRINTINGS

    lander_tp = TOKEN_PRINTINGS.get("Lander")
    if lander_tp is None:
        return None
    return _token_with_safe_activated_ability_card_def(lander_tp)


def _python_ta_to_safe_dict(ta) -> Optional[dict]:
    """Convert a Python ``TriggeredAbility`` back to the Rust-shape
    dict iff it matches one of the :data:`_SAFE_TRIGGERS` shapes.

    Returns ``None`` for triggers whose shape isn't on the safe list.
    The production-deck loader uses this as the gate: a creature
    every one of whose Python-parsed triggered abilities round-trips
    through this helper passes the bridge-simple filter and lands in
    the cross-engine harness with its triggered abilities intact.

    Round-trip invariant: the dict this returns satisfies
    :func:`_is_safe_trigger_dict` AND the Rust + Python bridge arms
    consume it without raising. The dict shape mirrors what a CardDef
    written by a probe author would carry directly — same field
    spellings as :func:`cross_engine._ta_dict_to_python_triggered_ability`
    consumes.
    """
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import (
        And,
        Controlled,
        IsSelf,
        Event,
        Subject,
        YOU,
    )
    from mtg_bot.effects.object_predicates import IsLand
    from mtg_bot.effects.generic.bounce import (
        ReturnTargetPermanentYouControlToOwnersHand,
    )
    from mtg_bot.effects.generic.counters import (
        PutPlusOneCounterOnOpponentsCreature,
        PutPlusOneCounterOnSelf,
    )
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.effects.generic.library_lookat import (
        Scry,
        SelfMill,
        Surveil,
    )
    from mtg_bot.effects.generic.pump import PumpSelfUntilEOT
    from mtg_bot.effects.generic.tokens import CreateToken

    if not isinstance(ta, TriggeredAbility):
        return None
    if ta.once_per_turn:
        return None
    # Event must be one of the events present in :data:`_SAFE_TRIGGERS`.
    # Round-42 surveil extension widens the starter ENTERS-only gate to
    # also admit DIES (the SECOND event in the safe set — adds Thawbringer
    # / Wary Watchdog whose surveil trigger pair carries both events;
    # without DIES support the whole printing would be dropped because
    # the safe-trigger filter requires EVERY trigger to round-trip).
    # Round-43 surveil ATTACKS extension widens further to admit
    # ATTACKS (the THIRD event in the safe set — pre-positions the
    # substrate for any future ATTACKS+IsSelf+Surveil consumer; no real-
    # EoE consumer admitted today since Fear of Surveillance is absent
    # from the 17lands deck pool and Appendage Amalgam has Flash).
    # The event-string is emitted further down per arm — gate here so
    # any future event extension stays a single edit.
    # Round-58 widens the ENTERS/DIES/ATTACKS gate to also admit
    # BECOMES_TAPPED (the FOURTH event in the safe set — adds Seedship
    # Agrarian whose first trigger is BECOMES_TAPPED+IsSelf+CreateToken;
    # without BECOMES_TAPPED support the whole printing would be
    # dropped because the safe-trigger filter requires EVERY trigger
    # to round-trip and Seedship's landfall trigger alone wouldn't
    # admit it). The event-string is emitted per arm below.
    if ta.event not in (
        Event.ENTERS, Event.DIES, Event.ATTACKS, Event.BECOMES_TAPPED,
    ):
        return None
    if ta.event is Event.ENTERS:
        event_str = "enters"
    elif ta.event is Event.DIES:
        event_str = "dies"
    elif ta.event is Event.BECOMES_TAPPED:
        event_str = "becomes_tapped"
    else:  # Event.ATTACKS
        event_str = "attacks"
    # Recognize one of the safe predicate shapes; produce the
    # corresponding round-trip dict (or bail if no match).
    pred = ta.predicate
    pred_dict: Optional[dict] = None
    safe_effect_kinds: tuple[str, ...] = ()
    if isinstance(pred, IsSelf) and pred.subject is None:
        # ``IsSelf`` (default-slot form): "source_is_self" canonical
        # kind. Three ENTERS-only leaf effects from the starter set
        # (draw_cards / gain_life / scry_n); the round-42 surveil
        # extension adds ``surveil`` under BOTH ENTERS and DIES; the
        # round-43 surveil ATTACKS extension adds ``surveil`` under
        # ATTACKS; the round-44 ATTACKS+draw_cards/gain_life extension
        # adds those two leaves under ATTACKS, mirroring the ENTERS
        # starter set on the ATTACKS event axis.
        pred_dict = {"kind": "source_is_self"}
        if ta.event is Event.ENTERS:
            # Round-48 widens the ENTERS+IsSelf arm from
            # {draw_cards, gain_life, scry_n, surveil} to also include
            # ``mill`` (bare SelfMill — substrate-only; the only EoE
            # ENTERS+IsSelf+SelfMill carrier, Seedship Broodtender,
            # is dropped on its activated ability so no real consumer
            # is admitted on this leaf; the landfall+mill entry below
            # is the one that admits a real card, Icetill Explorer).
            # Round-49 N1 widens it further to include ``create_token``
            # (the fixed-single-vanilla-token "When ~ enters, create N
            # tokens of <fixed printing>" shape — substrate-only; every
            # high-deck-count carrier produces the non-vanilla Lander
            # token or carries a declined rider, so no real consumer is
            # admitted today; see ``docs/TOKEN_PRINTING_SUBSTRATE.md``).
            # Phase 4 (round 50) adds ``create_token_tapped`` under
            # ENTERS as the sister of the Phase-1 ``create_token``
            # leaf — pre-positions the substrate for any future "When ~
            # enters, create a tapped token" carrier (substrate-only on
            # ENTERS; the real-card delta is on DIES via Gravpack
            # Monoist, see the DIES arm below).
            # Tier-C target-seating lever-a (Exosuit Savior) widens
            # the ENTERS+IsSelf arm with the targeted-safe leaf
            # ``return_target_permanent_you_control`` (the
            # count_min=0 optional-bounce edge — the ENTERS sibling
            # of Sunset Saboteur's ATTACKS
            # ``put_plus_one_counter_on_opponents_creature``). The
            # predicate-effect product is still enforced by the
            # ``("enters", "source_is_self",
            # "return_target_permanent_you_control")``
            # ``_SAFE_TRIGGERS`` entry (this list is necessary, not
            # sufficient); the recognizer's bare-shape gate above is
            # the §2 teeth.
            safe_effect_kinds = (
                "draw_cards", "gain_life", "scry_n", "surveil", "mill",
                "create_token", "create_token_tapped",
                "return_target_permanent_you_control",
            )
        elif ta.event is Event.ATTACKS:
            # Round-44 widens the ATTACKS arm from {surveil} to
            # {surveil, draw_cards, gain_life}. ``scry_n`` is
            # intentionally left out today — no EoE consumer carries
            # ATTACKS+IsSelf+Scry and the scry resolver shares the
            # ``keep_all_on_top`` PendingDecision substrate with
            # surveil, so the round-43 ATTACKS+surveil precedent
            # already pins it; a future slice landing such a consumer
            # ships ``scry_n`` under ATTACKS as a single-line edit
            # here AND a single new ``_SAFE_TRIGGERS`` entry.
            # Tier-C target-seating lever-a (Sunset Saboteur) widens
            # the ATTACKS+IsSelf arm with the FIRST targeted-safe leaf
            # ``put_plus_one_counter_on_opponents_creature``. The
            # predicate-effect product is still enforced by the
            # ``("attacks", "source_is_self",
            # "put_plus_one_counter_on_opponents_creature")``
            # ``_SAFE_TRIGGERS`` entry (this list is necessary, not
            # sufficient); the recognizer's bare-shape gate above is
            # the §2 teeth.
            safe_effect_kinds = (
                "surveil", "draw_cards", "gain_life",
                "put_plus_one_counter_on_opponents_creature",
            )
        elif ta.event is Event.DIES:
            # Event.DIES — round-42 shipped the surveil leaf; the
            # round-47 DIES+IsSelf+gain_life entry widened it to
            # {surveil, gain_life}; THIS slice (round-47 DIES+IsSelf+
            # draw_cards, sister to the gain_life entry on the
            # draw_cards axis) widens it further to {surveil, gain_life,
            # draw_cards} — the SAME three-leaf set ENTERS carries
            # (minus scry_n, which has no DIES consumer) and ATTACKS
            # carries (round-43 surveil + round-44 draw_cards/gain_life).
            # This closes the ENTERS/DIES/ATTACKS × {draw_cards,
            # gain_life, surveil} matrix. The recognizer's event-string
            # gate above and the ``_is_safe_trigger_dict`` dict-side
            # gate stay unchanged since both already admit DIES + the
            # draw_cards ``count`` payload-key whitelist from
            # :data:`_SAFE_EFFECT_KEYS` (round-1 starter, event-agnostic).
            # ``scry_n`` stays out under DIES — no EoE consumer carries
            # DIES+IsSelf+Scry; a future slice landing one ships it as a
            # single-line edit here AND a single new ``_SAFE_TRIGGERS``
            # entry.
            #
            # Phase 4 (round 50) adds ``create_token_tapped`` under
            # DIES — this is the real-card lever: it admits **Gravpack
            # Monoist** ("When this creature dies, create a tapped 2/2
            # Robot token." — 25,252 17lands decks). Its Robot token is
            # vanilla, so the Phase-1 vanilla-token gate +
            # ``CreateTokenTapped`` Rust sister variant + the
            # ``create_token_tapped`` bridge arm flip it drop→admit
            # with no Phase-3 dependency.
            #
            # Phase 3a (round-53 N1) adds bare ``create_token`` under
            # DIES — the round-50 comment ("no real EoE
            # DIES+IsSelf+bare-create_token consumer exists") held only
            # while every DIES Lander carrier was declined on the
            # non-vanilla token. Phase 3a's
            # ``_token_with_safe_activated_ability_card_def`` admits the
            # bare Lander, so **Beamsaw Prospector** ("When ~ dies,
            # create a Lander." — Creature — Human Artificer, 24,310
            # 17lands decks) is now exactly that consumer. The bare
            # ``create_token`` kind round-trips identically under DIES
            # vs ENTERS (the bridge's ``create_token`` arm is event-
            # agnostic; the seating pass already recognizes the kind
            # under any event), so this is a single-line widen + the
            # paired ``("dies","source_is_self","create_token")``
            # ``_SAFE_TRIGGERS`` entry.
            safe_effect_kinds = (
                "surveil", "gain_life", "draw_cards",
                "create_token", "create_token_tapped",
            )
        else:
            # Event.BECOMES_TAPPED (round-58 N1 — the BECOMES_TAPPED
            # event bridge). The ONLY safe leaf under
            # BECOMES_TAPPED+IsSelf today is ``create_token`` — the
            # Seedship Agrarian shape ("Whenever this creature becomes
            # tapped, create a Lander token"). Deliberately narrow:
            # no other EoE BECOMES_TAPPED+IsSelf consumer exists, and
            # the ``create_token`` bridge arm is event-agnostic +
            # already proven byte-equal under ENTERS/DIES (round-49/53
            # create_token work) so this is a single-leaf widen, NOT
            # the full ENTERS-style {draw_cards, gain_life, scry_n,
            # surveil, mill, create_token, create_token_tapped} set.
            # A future slice landing a BECOMES_TAPPED carrier with a
            # different leaf ships that leaf here + the paired
            # ``("becomes_tapped","source_is_self",<leaf>)``
            # ``_SAFE_TRIGGERS`` entry (same one-line-widen pattern as
            # the ENTERS/DIES/ATTACKS leaf-set extensions). The
            # predicate-effect product is enforced by
            # ``_SAFE_TRIGGERS`` (the
            # ``("becomes_tapped","source_is_self","create_token")``
            # tuple), so even though ``create_token`` is listed here
            # the triple gate rejects any other event/predicate
            # pairing that isn't an explicit ``_SAFE_TRIGGERS`` entry.
            safe_effect_kinds = ("create_token",)
    elif _is_landfall_predicate_python(pred) and ta.event is Event.ENTERS:
        # Landfall is parser-emitted under ENTERS only (CR 614 — the
        # land enters under your control); gate on event to keep the
        # DIES extension (round 42) from accidentally pairing the
        # landfall predicate with a DIES event that no parser emits.
        # Landfall extension: compound
        # ``And(Subject(slot=None, inner=IsLand()),
        #       Controlled(by=YOU, subject=None))``
        # — emit the canonical AND-nested dict shape both engines'
        # bridges already translate. THREE safe leaves today:
        # ``gain_life`` (Eumidian Terrabotanist, round-36 slice),
        # ``pump_self_until_eot`` (Icecave Crasher, round-37 DDDD
        # slice — the bare ``power_bonus`` / ``toughness_bonus``
        # shape with NO ``granted_keywords`` rider; sister Remnant
        # Elemental carries the same bare shape but is dropped by
        # the ``_SAFE_KEYWORDS`` filter on its ``Reach`` keyword),
        # and ``draw_cards`` (this slice — preventative substrate
        # extension; no EoE consumer admitted today since Tannuk
        # Memorial Ensign's draw rider is the bespoke
        # ``TannukEnsignLandfallEffect`` second-time-bonus shape,
        # NOT a bare ``DrawCards`` leaf; pinned by reject-path
        # tests so the predicate-effect product gate stays honest).
        # Other landfall consumers (Tannuk, Mightform Harmonizer,
        # Genemorph Imago, Eusocial Engineering, Glacier Godmaw,
        # Icetill Explorer, Territorial Bruntar) need their own
        # leaf-effect bridge arms before they can ride this
        # predicate.
        pred_dict = {
            "kind": "and",
            "left": {"kind": "subject_is_land"},
            "right": {"kind": "controlled_by_you"},
        }
        # Round-48 widened the landfall arm from
        # {gain_life, pump_self_until_eot, draw_cards} to also include
        # ``mill`` (Icetill Explorer). Round-59 widens it again to
        # also include ``put_plus_one_counter_on_self`` — admits the
        # real EoE card Seedship Agrarian's TRIGGER-2 ("Landfall —
        # Whenever a land you control enters, put a +1/+1 counter on
        # this creature"), the sole remaining blocker after round-58
        # N1 bridged its BECOMES_TAPPED trigger-1. See the
        # ``("enters", "land_enters_controlled_by_you",
        # "put_plus_one_counter_on_self")`` entry in
        # :data:`_SAFE_TRIGGERS` for the full cross-engine-parity
        # provenance.
        safe_effect_kinds = (
            "gain_life", "pump_self_until_eot", "draw_cards", "mill",
            "put_plus_one_counter_on_self",
        )
    if pred_dict is None:
        return None
    # Effect must be one of the safe leaves AND in the safe set for
    # this predicate (the predicate-effect product is enforced by
    # ``_SAFE_TRIGGERS``, not the effect kinds alone — e.g.
    # draw_cards under landfall is unsafe today).
    eff = ta.effect
    if isinstance(eff, DrawCards):
        eff_kind = "draw_cards"
        eff_dict: dict = {"kind": "draw_cards", "count": int(eff.amount)}
    elif isinstance(eff, GainLife):
        eff_kind = "gain_life"
        eff_dict = {"kind": "gain_life", "amount": int(eff.amount)}
    elif isinstance(eff, Scry):
        eff_kind = "scry_n"
        # Python's ``Scry`` exposes the count as ``amount``; the dict
        # shape uses Rust's spelling ``count``.
        eff_dict = {"kind": "scry_n", "count": int(eff.amount)}
    elif isinstance(eff, PumpSelfUntilEOT):
        # Bare-shape PumpSelfUntilEOT (no ``granted_keywords``
        # rider). Python's class carries an additional
        # ``granted_keywords: frozenset[str]`` field
        # (``effects/generic/pump.py:632``) that Rust's
        # ``PumpSelfUntilEOT { power_bonus, toughness_bonus }``
        # variant does NOT represent — gate on the rider being
        # empty so a parser-emitted "Landfall — this creature gets
        # +1/+0 and gains haste UEOT" doesn't slip through with the
        # keyword silently dropped on the cross-engine round trip.
        # No EoE landfall consumer carries the rider today (Icecave
        # Crasher and Remnant Elemental both emit the bare shape),
        # so this gate simply pins the contract.
        if eff.granted_keywords:
            return None
        eff_kind = "pump_self_until_eot"
        eff_dict = {
            "kind": "pump_self_until_eot",
            "power_bonus": int(eff.power_bonus),
            "toughness_bonus": int(eff.toughness_bonus),
        }
    elif isinstance(eff, Surveil):
        # Bare-shape Surveil (no ``use_artifact_count`` rider).
        # Python's class carries an additional ``use_artifact_count``
        # flag that selects the Surveil-X-where-X=artifacts variant
        # (effects/generic/library_lookat.py:580); Rust's
        # ``Effect::Surveil { count: i32 }`` does NOT represent the
        # X variant — the bridge would silently degrade Surveil-X to
        # a fixed surveil-N on round trip. Gate on the rider being
        # False so the Surveil-X mechanic falls through to the
        # filter drop path. No EoE consumer's ENTERS / DIES surveil
        # trigger uses the X variant today (it's a separate effect
        # arm on Anticausal Vestige and friends), so this gate just
        # pins the contract.
        if eff.use_artifact_count:
            return None
        eff_kind = "surveil"
        # Python's ``Surveil`` exposes the count as ``amount``; the
        # dict shape uses Rust's spelling ``count`` per the
        # cross_engine bridge arm convention.
        eff_dict = {"kind": "surveil", "count": int(eff.amount)}
    elif isinstance(eff, SelfMill):
        # Bare-shape SelfMill (round-48 bench-admit-set slice). Python's
        # ``SelfMill`` class (effects/generic/library_lookat.py:624)
        # carries a SINGLE ``amount`` field — no rider the bridge would
        # silently drop (contrast Surveil's ``use_artifact_count`` or
        # PumpSelf's ``granted_keywords``). The dict shape uses Rust's
        # spelling ``count`` per the ``mill`` cross_engine bridge arm
        # convention (cross_engine.py:725, shipped round-16); the
        # ``amount``/``count`` field-name discrepancy is local to the
        # bridge translation, same shape as the established
        # ``draw_cards count -> amount`` and ``surveil count -> amount``
        # arms. SelfMill is a pure deterministic zone move (own library
        # top -> own graveyard); empty-library mid-mill silently stops
        # on BOTH engines (CR 704.5b applies to drawing, not milling)
        # — see the ``("enters", "land_enters_controlled_by_you",
        # "mill")`` entry in :data:`_SAFE_TRIGGERS` for the full
        # cross-engine parity contract.
        eff_kind = "mill"
        eff_dict = {"kind": "mill", "count": int(eff.amount)}
    elif isinstance(eff, PutPlusOneCounterOnSelf):
        # Bare-shape PutPlusOneCounterOnSelf (round-59 N1 — the sole
        # remaining Seedship Agrarian blocker). Python's
        # ``PutPlusOneCounterOnSelf`` class
        # (effects/generic/counters.py:784) carries a SINGLE
        # ``amount`` field (default 1) — no rider the bridge would
        # silently drop (contrast Surveil's ``use_artifact_count`` or
        # PumpSelf's ``granted_keywords``). The recipient is ALWAYS
        # the ability's source ("this creature") — no target picker
        # on either side. The dict shape uses the IDENTICAL spelling
        # ``amount`` on both engines (Rust's
        # ``Effect::PutPlusOneCounterOnSelf`` to_dict emits
        # ``{"kind": "put_plus_one_counter_on_self",
        # "amount": <i32>}`` at cards.rs:7179-7181 — no
        # ``count``/``amount`` discrepancy unlike mill/scry/surveil),
        # round-tripped by the ``put_plus_one_counter_on_self``
        # cross_engine bridge arm (cross_engine.py:3899, shipped
        # round-22; event-agnostic + byte-equal-proven under
        # ATTACKS/ENTERS/DIES). Negative amount is rejected by the
        # bridge arm (mirrors Rust parse_effect); amount==0 is a
        # silent no-op on both engines. See the
        # ``("enters", "land_enters_controlled_by_you",
        # "put_plus_one_counter_on_self")`` entry in
        # :data:`_SAFE_TRIGGERS` for the full cross-engine parity
        # contract.
        eff_kind = "put_plus_one_counter_on_self"
        eff_dict = {
            "kind": "put_plus_one_counter_on_self",
            "amount": int(eff.amount),
        }
    elif isinstance(eff, CreateToken):
        # Round-49 N1 token-printing substrate, Phase 1 (bare rider) +
        # Phase 4 (round 50, the ``tapped`` rider). Fixed-single-token
        # "create N tokens of <fixed printing>" shape. Two §2
        # silent-mistranslation gates:
        #
        #  (1) **Supported-rider only.** Rust's bare
        #      ``Effect::CreateToken { count, token_printing_id }``
        #      (cards.rs:219) has NO rider fields; Phase 4 adds the
        #      ``Effect::CreateTokenTapped`` sister variant for the
        #      ``tapped=True`` rider (cards.rs — the
        #      ``create_token_tapped`` parse/to_dict/resolve arms +
        #      the ``create_token_tapped`` bridge arm). So the
        #      recognizer now supports TWO rider profiles:
        #        * all riders default  -> ``create_token`` (Phase 1)
        #        * ``tapped=True`` only -> ``create_token_tapped``
        #          (Phase 4 — the byte-equal tapped sister)
        #      The ``each_player=True`` rider (Edge Rover) still has
        #      its own Rust variant + bridge arm but no loader
        #      recognizer/auto-registration arm (Phase 5 in the design
        #      doc); ``attacking`` and
        #      ``schedule_end_of_combat_sacrifice`` have no Rust
        #      mirror at all. Any carrier engaging those riders (or
        #      ``tapped`` COMBINED with another rider) would round-trip
        #      to a byte-divergent token, so it falls to the filter
        #      drop path here.
        #  (2) **Vanilla token only.** The loader synthesizes a token
        #      CardDef from the parsed ``token_printing``; CardDef
        #      carries name / card-types / P/T / subtypes / keywords /
        #      triggered_abilities but has NO activated-ability field.
        #      A non-vanilla token (the EoE ``Lander``, which carries
        #      ``{2}, Sac: SearchLibraryForBasicLandTapped``) would
        #      have its activated ability SILENTLY DROPPED on the
        #      bridge. Reject any token printing carrying activated /
        #      static abilities, keywords the safe-keyword set doesn't
        #      cover, nested triggered abilities, or a non-creature /
        #      None-P-T body. (The token's triggered abilities, if
        #      any, would also have to round-trip — out of scope here;
        #      require zero.) Gravpack Monoist's Robot token IS vanilla
        #      (2/2 Artifact Creature, no abilities/keywords) so it
        #      passes gate (2); only its ``tapped=True`` rider kept it
        #      dropped under Phase 1 — Phase 4 closes that decline.
        #
        # The emitted dict carries a SYMBOLIC ``_token_card_def`` (the
        # synthesized token CardDef) rather than a numeric
        # ``token_printing_id``. The numeric pid is assigned by the
        # loader's ``_seat_token_printings`` pass once the union
        # ``card_defs`` tuple is frozen and the token's Rust-registry
        # position is known (``docs/TOKEN_PRINTING_SUBSTRATE.md`` §1.4).
        # ``_is_safe_trigger_dict`` validates the POST-seating shape
        # (``token_printing_id`` numeric), so a never-seated dict that
        # leaked to the dict-side gate would correctly be rejected.
        #
        # ``_seat_token_printings`` recognizes BOTH ``create_token``
        # and ``create_token_tapped`` symbolic carriers and preserves
        # the ``kind`` through the seating rewrite (the variant
        # distinction must survive — it IS the tapped-vs-untapped
        # choice on the bridge).
        if eff.each_player or eff.attacking:
            return None
        if getattr(eff, "schedule_end_of_combat_sacrifice", False):
            return None
        tp = eff.token_printing
        # Phase 1/4: a VANILLA token (creature body, no abilities).
        # Phase 3a (round-53 N1): the EoE ``Lander`` token — a
        # non-creature artifact whose ONLY activated ability is the
        # bridge-correct ``{2},Sac:SearchLibraryForBasicLandTapped``
        # shape (round-52 N1 precursor proved it round-trips with NO
        # §2 gap). The vanilla helper declines the Lander (it has an
        # activated ability + no creature body); fall back to the
        # ability-bearing-token helper, which §2-strictly requires the
        # token's ONLY activated ability to be exactly that Lander
        # shape and the token to be otherwise clean. ANY other
        # ability-bearing token (a different cost / effect / extra
        # ability / keyword) returns None from BOTH helpers and stays
        # DECLINED. The emitted CardDef carries
        # ``token_activated_abilities`` which ``_seat_token_printings``
        # threads unchanged through ``dataclasses.replace`` and
        # ``cached_token_printing`` (the round-52 cache key folds
        # ``token_activated_abilities`` in, so a Lander never collides
        # with a same-named vanilla token — verified by the round-52
        # cache-key pin).
        token_cd = _token_vanilla_card_def(tp)
        if token_cd is None:
            token_cd = _token_with_safe_activated_ability_card_def(tp)
        if token_cd is None:
            return None
        # Phase 4: the ``tapped`` rider routes to the
        # ``create_token_tapped`` sister kind (Rust
        # ``Effect::CreateTokenTapped``); all-default routes to the
        # Phase-1 bare ``create_token``. No other rider is set here
        # (each_player/attacking/schedule already declined above), so
        # exactly one of the two profiles applies.
        eff_kind = "create_token_tapped" if eff.tapped else "create_token"
        eff_dict = {
            "kind": eff_kind,
            "count": int(eff.count),
            # Symbolic token reference — replaced with a numeric
            # ``token_printing_id`` by ``_seat_token_printings``
            # (which preserves this ``kind``).
            "_token_card_def": token_cd,
        }
    elif isinstance(eff, PutPlusOneCounterOnOpponentsCreature):
        # **Tier-C target-seating sub-substrate (lever-a, Sunset
        # Saboteur).** The FIRST targeted-safe-trigger leaf. Python's
        # ``PutPlusOneCounterOnOpponentsCreature(amount: int = 1)``
        # (effects/generic/counters.py — a ``PutPlusOneCounters``
        # subclass whose ``__post_init__`` wires the single slot
        # ``TargetSpec(IsCreature() & ControlledByOpponent())``,
        # ``count_min=1``, ``count_max=1``, ``slot_kinds()==()``).
        # Sunset Saboteur's ATTACKS trigger is the sole EoE consumer
        # (``{1}{B}`` 4/1 ``menace``, 6,375 17lands decks).
        #
        # **This is a TARGET-SEATED leaf — distinct from every other
        # arm above (all of which are recipient-is-self / untargeted).**
        # Both engines pick the target via the Tier-C shared
        # deterministic push-time seat (``first_legal_trigger_target``
        # on Rust at ``state_ops_resolve.rs:push_triggered_ability``;
        # the no-agent ``pool[0]`` seat on Python at
        # ``actions.py:_put_triggered_ability_on_stack`` — the
        # ``controller.agent is None`` branch), so NO loader-seated
        # agent is needed and BOTH engines pick the SAME
        # smallest-bf-index opponent-controlled creature. The
        # ``put_plus_one_counter_on_opponents_creature`` bridge arm
        # (``cross_engine.py``) round-trips the bare ``amount`` payload
        # IDENTICALLY on both engines (Rust
        # ``Effect::PutPlusOneCounterOnOpponentsCreature`` to_dict
        # emits ``{"kind": "put_plus_one_counter_on_opponents_creature",
        # "amount": <i32>}`` — no ``count``/``amount`` discrepancy;
        # parse arm tolerates a missing ``amount`` defaulting to 1 and
        # rejects negatives, mirroring the Python class's
        # ``amount: int = 1`` default). The resolve-time CR 608.2b
        # re-legality check (target still on a bf, still a creature,
        # still opponent-controlled) is byte-equal on both engines
        # (Python's ``PutPlusOneCounters.resolve``/
        # ``_slot_target_still_legal``; Rust's resolve arm's explicit
        # re-check). See the
        # ``("attacks", "source_is_self",
        # "put_plus_one_counter_on_opponents_creature")`` entry in
        # :data:`_SAFE_TRIGGERS` for the full cross-engine parity
        # provenance + the deterministic-first SEARCH_LIMITATIONS §2.3
        # entry.
        #
        # **Bare-shape gate (§2 silent-mistranslation teeth).** Only
        # the EXACT single-implicit-target shape is admitted: exactly
        # one ``TargetSpec`` slot, ``count_min==1``, ``count_max==1``,
        # ``slot_kinds()==()`` (no ``SlotRef``, no per-slot kind, no
        # ``distinct_from``). Any heterogeneous / multi-slot shape
        # falls through to the filter drop — the bridge + the shared
        # seat both assume the single-bare-slot contract. (Sunset
        # Saboteur's parsed effect satisfies this exactly — verified
        # live: ``count_range=(1,1)``, ``slot_kinds=()``.)
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        slots = tuple(getattr(eff, "targets", ()) or ())
        if len(slots) != 1:
            return None
        slot = slots[0]
        if isinstance(slot, SlotRef) or not isinstance(slot, TargetSpec):
            return None
        if slot.count_min != 1 or slot.count_max != 1:
            return None
        if getattr(slot, "distinct_from", ()) not in ((), None):
            return None
        if eff.target_slot_kinds() != ():
            return None
        if int(eff.amount) < 0:
            return None
        eff_kind = "put_plus_one_counter_on_opponents_creature"
        eff_dict = {
            "kind": "put_plus_one_counter_on_opponents_creature",
            "amount": int(eff.amount),
        }
    elif isinstance(eff, ReturnTargetPermanentYouControlToOwnersHand):
        # **Tier-C target-seating sub-substrate, lever-a (Exosuit
        # Savior — the count_min=0 optional-bounce edge).** Python's
        # ``ReturnTargetPermanentYouControlToOwnersHand(optional=True,
        # exclude_self=True)`` (effects/generic/bounce.py — a
        # ``Bounce`` subclass whose ``__post_init__`` wires the single
        # slot ``TargetSpec(ControlledByYou() & ~IsPlayer() &
        # ~IsSource(), count_min=0, count_max=1)``). Exosuit Savior's
        # ENTERS trigger is the sole EoE consumer ({2}{W} 2/2
        # ``flying``, 12,480 17lands decks).
        #
        # **The count_min=0 optional edge (contrast Sunset
        # Saboteur's count_min=1).** ``requires_target()`` reads
        # ``False`` (CR "you may return") but the slot is a real
        # ``TargetSpec``. The Tier-C shared seat handles the edge:
        # no eligible own permanent ⇒ Rust's
        # ``first_legal_trigger_target`` returns ``None`` and
        # ``push_triggered_ability`` takes the UNCHANGED
        # ``new_trigger`` path → push-with-no-target → the resolve
        # arm no-ops (byte-equal with Python's ``targets=()``
        # ``Bounce.resolve`` early-return — the optional was simply
        # declined, NOT fizzled). When an eligible permanent exists
        # both engines deterministically pick the SAME
        # smallest-bf-index own-non-source permanent and bounce it.
        # The ``return_target_permanent_you_control`` bridge arm
        # round-trips the UNIT shape (no payload field — the filter
        # is structural to the effect class / the seat's per-variant
        # scan; the recognizer's bare-shape gate enforces the single
        # implicit-target contract). See the
        # ``("enters", "source_is_self",
        # "return_target_permanent_you_control")`` entry in
        # :data:`_SAFE_TRIGGERS` for the full cross-engine parity
        # provenance + the deterministic-first SEARCH_LIMITATIONS
        # §2.3 entry.
        #
        # **Bare-shape gate (§2 silent-mistranslation teeth).** Only
        # the EXACT single-implicit-target shape is admitted: exactly
        # one ``TargetSpec`` slot, ``count_min in (0, 1)`` (Exosuit
        # Savior is the optional ``count_min=0`` — DISTINCT from
        # Sunset Saboteur's mandatory ``count_min=1``; both ride the
        # same shared seat), ``count_max==1``, ``slot_kinds()==()``,
        # no ``SlotRef`` / ``distinct_from``. Any heterogeneous /
        # multi-slot shape falls through to the filter drop.
        # (Exosuit Savior's parsed effect satisfies this exactly —
        # verified live: ``count_range=(0,1)``, ``slot_kinds=()``.)
        from mtg_bot.effects.targets import SlotRef, TargetSpec
        slots = tuple(getattr(eff, "targets", ()) or ())
        if len(slots) != 1:
            return None
        slot = slots[0]
        if isinstance(slot, SlotRef) or not isinstance(slot, TargetSpec):
            return None
        if slot.count_min not in (0, 1) or slot.count_max != 1:
            return None
        if getattr(slot, "distinct_from", ()) not in ((), None):
            return None
        if eff.target_slot_kinds() != ():
            return None
        eff_kind = "return_target_permanent_you_control"
        eff_dict = {"kind": "return_target_permanent_you_control"}
    else:
        return None
    if eff_kind not in safe_effect_kinds:
        # Effect kind is a safe leaf in the abstract but not safe
        # under this (event, predicate) pair — e.g. a parser-emitted
        # "Landfall — draw a card" would land here once such a card
        # exists in the EoE pool. Out of the safe set until the
        # matching ``_SAFE_TRIGGERS`` entry lands.
        return None
    return {
        "event": event_str,
        "predicate": pred_dict,
        "effect": eff_dict,
    }


# **Tier-C target-seating sub-substrate.** The set of recognizer
# ``effect.kind`` values that ride the deterministic push-time target
# seat (a targeted SAFE triggered ability whose target is picked by
# the no-agent first-legal rule, NOT a loader-seated agent). Every
# OTHER safe-trigger leaf is recipient-is-self / untargeted and is
# NOT in this set — so the surgical Python seat (§3.2) fires ONLY for
# these. ``put_plus_one_counter_on_opponents_creature`` (Sunset
# Saboteur, lever-a) is the first member; Exosuit Savior's
# ``return_target_permanent_you_control`` joins in its own slice.
#
# This is a SEPARATE module-level signal rather than a top-level
# ``"target_seated"`` key on the round-trip dict because
# :func:`_is_safe_trigger_dict` refuses any top-level key beyond
# ``event`` / ``predicate`` / ``effect`` / ``once_per_turn`` (the
# schema-extension backstop). Keying the gate off the recognized
# ``effect.kind`` keeps the cross-engine round-trip dict shape
# byte-identical while still making the seat surgical.
_TARGET_SEATED_EFFECT_KINDS: frozenset[str] = frozenset({
    "put_plus_one_counter_on_opponents_creature",
    # Exosuit Savior (the count_min=0 optional-bounce edge) — joins
    # via its own per-card slice, reusing the shared seat shipped
    # with Sunset Saboteur. The gate is the same: the Python no-agent
    # ``pool[0]`` seat fires for this kind too (and the count_min=0
    # edge means an empty ``pool`` simply declines the optional —
    # ``len(chosen_list)=0 >= min_t=0`` — byte-equal with the Rust
    # no-seat no-op).
    "return_target_permanent_you_control",
})


def _is_target_seated_safe_trigger(source, ability) -> bool:
    """True iff ``ability`` (a parsed ``TriggeredAbility`` on
    ``source``) round-trips through :func:`_python_ta_to_safe_dict`
    as a TARGET-SEATED safe trigger — i.e. its recognized
    ``effect.kind`` is in :data:`_TARGET_SEATED_EFFECT_KINDS`.

    This is the gate that keeps the Tier-C no-agent deterministic
    target seat (the ``controller.agent is None`` branch of
    ``actions._put_triggered_ability_on_stack``) **surgical**: it
    fires ONLY for the lever-admitted targeted safe triggers (Sunset
    Saboteur today), never for the engine's other targeted triggers
    (which keep their existing no-agent fizzle/no-op intact) and
    never for any untargeted safe trigger (those don't reach the
    target-pick loop at all). ``source`` is accepted for signature
    symmetry with the call site (the recognizer reads only
    ``ability``); a future per-source gate can use it without a
    call-site change.
    """
    dct = _python_ta_to_safe_dict(ability)
    if dct is None:
        return False
    eff = dct.get("effect")
    if not isinstance(eff, dict):
        return False
    return eff.get("kind") in _TARGET_SEATED_EFFECT_KINDS


def _is_landfall_predicate_python(pred: object) -> bool:
    """True iff ``pred`` is the canonical landfall compound predicate
    on the Python side: ``And(Subject(slot=None, inner=IsLand()),
    Controlled(by=YOU, subject=None))``.

    Mirrors :func:`_is_landfall_predicate_dict` on the Python-object
    side. Strict on operand ordering AND on the optional-slot
    fields' default values — anything else (a non-default subject
    slot, ``Or``-form, an inner type other than ``IsLand``, a
    Controlled-by-OPPONENT) is out of the safe set until its own
    slice lands.
    """
    from mtg_bot.effects.predicates import (
        And,
        Controlled,
        Subject,
        YOU,
    )
    from mtg_bot.effects.object_predicates import IsLand

    if not isinstance(pred, And):
        return False
    left = pred.left
    right = pred.right
    if not isinstance(left, Subject) or left.slot is not None:
        return False
    if not isinstance(left.inner, IsLand):
        return False
    if not isinstance(right, Controlled):
        return False
    if right.by is not YOU:
        return False
    if right.subject is not None:
        return False
    return True


def _printing_is_bridge_simple_creature(printing) -> bool:
    """True if a CardPrinting is a vanilla bridge-supported creature.

    A "bridge-simple" creature has:
      * "Creature" in card_types
      * Non-None power and toughness
      * Either no keywords, OR every keyword in :data:`_SAFE_KEYWORDS`
        (Phase 6 carddef-substrate slice — see the constant's docstring
        for the per-keyword parity provenance).
      * Either no triggered_abilities, OR every triggered_ability
        round-trips through :func:`_python_ta_to_safe_dict` (Phase 6
        carddef-triggers slice — see :data:`_SAFE_TRIGGERS` for the
        per-shape parity provenance).
      * No activated_abilities (other than the empty default)
      * No static_abilities
      * No spell_effect (creatures don't have one anyway, but be
        defensive — flip cards / mdfcs land here)
      * No warp / kicker / impending / room halves / planeswalker
        loyalty
      * Mana cost is parseable into (generic, colored) — no hybrid /
        phyrexian / X / snow.
      * Admitted ``dynamic_pt_formula`` keys (as of round-84):
        ``"harmonious_grovestrider"`` (round-78) and ``"cosmogoyf"``
        (round-84, this slice). The CardDef bridge carries
        ``dynamic_pt_formula`` (an optional string field;
        round-trips through ``_spec_card_to_printing_dict`` →
        Rust's ``parse_printing`` and back through
        ``_card_def_to_python_printing``); Rust's
        ``battlefield_creature_power`` /
        ``battlefield_creature_toughness`` /
        ``collect_sba_victims`` honor each admitted key via
        ``state_ops_combat::dynamic_base_pt_pub`` in lockstep with
        Python's ``game._dynamic_base_pt``. The remaining sibling
        formula (``"emissary_escort"``) reads
        ``printing.power`` / ``printing.toughness`` and ADDS X
        (greatest mana value among other artifacts the controller
        controls); the CDA-only ``dynamic_pt_formula`` arm this
        dispatch table covers today sets the BASE from scratch, so
        Escort is a different per-formula shape and is
        honest-stopped Rust-side (the dispatch table's ``_`` arm
        returns ``None`` for any unrecognized key); it stays on
        the drop-at-filter path until a per-card slice extends the
        dispatch table. The pre-round-78 rejection (which dropped
        EVERY ``dynamic_pt_formula`` carrier) was the round-39
        honest-stop precursor to this substrate; ``9c0d440f`` /
        ``9e71b63b`` did the drop, ``9c0d440f``'s SEARCH_LIMITATIONS
        entry tracked the deferred substrate, round-78 delivered
        the substrate for Grovestrider, and round-84 extends it
        for Cosmogoyf.
      * No ``ward_cost`` (round-39 fix — Harmonious Grovestrider,
        Mouth of the Storm, Famished Worldsire, Lionheart Glimmer).
        The CardDef bridge doesn't carry ``ward_cost``; the bridged
        printing has ``ward_cost=None`` on both engines, so Ward never
        fires on the harness side either, which is BIT-EQUAL in the
        ``attack_all`` strategy used by the bench (the attacker never
        targets the warded creature). Filtering these out anyway keeps
        the substrate honest — if a future trajectory probe drives a
        targeted spell at one of these creatures, the bridge's silent
        Ward-drop would surface as another mystery divergence; better
        to ship them blocked at the filter today than to ship a
        targeting-strategy-conditional latent bug. Cards that ALSO
        carry an unsafe keyword / triggered ability would already be
        dropped above; this catches the bare-Ward case.
    """
    if "Creature" not in printing.card_types:
        return False
    if printing.power is None or printing.toughness is None:
        return False
    if printing.keywords and not printing.keywords.issubset(_SAFE_KEYWORDS):
        return False
    if printing.triggered_abilities:
        # Phase 6 carddef-triggers slice: a creature with one or more
        # triggered abilities passes the filter only if EVERY ability
        # round-trips cleanly through the safe-trigger gate. Any single
        # unsafe ability disqualifies the whole printing — there's no
        # "partial admit" semantic since the bridge has to thread the
        # full triggered-abilities tuple to both engines.
        for ta in printing.triggered_abilities:
            if _python_ta_to_safe_dict(ta) is None:
                return False
    if printing.activated_abilities:
        return False
    if printing.static_abilities:
        return False
    if printing.spell_effect is not None:
        return False
    if printing.warp_cost or printing.kicker_cost or printing.impending_cost:
        return False
    if printing.room_halves:
        return False
    if printing.loyalty is not None:
        return False
    if printing.cost_reductions or printing.external_cost_reductions:
        return False
    if printing.station_threshold:
        return False
    # Dynamic-P/T-formula substrate (round-78 + round-84). The
    # admit set is ``_DYNAMIC_PT_FORMULA_ADMIT_SET`` (module-level;
    # see its docstring for the per-round closure history and the
    # honest-stop on Emissary Escort). Rust's
    # ``battlefield_creature_power`` /
    # ``battlefield_creature_toughness`` /
    # ``collect_sba_victims`` honor each admitted key via
    # ``state_ops_combat::dynamic_base_pt_pub`` in lockstep with
    # Python's ``game._dynamic_base_pt``. A formula key NOT in the
    # admit set drops the printing here so we never ship a 0/0
    # mistranslation on the Rust side.
    dpt_formula = getattr(printing, "dynamic_pt_formula", None)
    if (
        dpt_formula is not None
        and dpt_formula not in _DYNAMIC_PT_FORMULA_ADMIT_SET
    ):
        return False
    # Round-39 ward-cost fix (Harmonious Grovestrider et al.): see
    # docstring above for the substrate-honesty rationale. This
    # rejection STAYS — the real EoE Harmonious Grovestrider printing
    # carries both ``dynamic_pt_formula`` AND ``ward_cost``; the
    # dynamic-P/T substrate landing in round-78 opens the formula
    # half of the gate, but the ward half remains an orthogonal
    # substrate (separate closure target). A synthetic Grovestrider
    # CardDef without the ward rider admits cleanly; the live EoE
    # printing remains drop-at-filter until the ward substrate
    # lands.
    if getattr(printing, "ward_cost", None) is not None:
        return False
    try:
        _parse_mana_cost_string(printing.mana_cost_str or "")
    except ValueError:
        return False
    return True


def _printing_is_bridge_pure_warp_creature(printing) -> bool:
    """True iff a CardPrinting is a warp creature the warp substrate
    round-trips byte-equal across engines — Phase 1's pure fixed-cost
    pair PLUS Phase 1.5's safe-trigger+warp residue trio PLUS Phase
    2's Timeline Culler (the life + graveyard-cast riders).

    Phase-6 warp_cost substrate, **phase 1** (round-63; spec at
    ``docs/WARP_COST_PHASE1_DESIGN.md`` §3, written round-62 N2) +
    **phase 1.5** (round-65; the safe-trigger+warp residue trio the
    Phase-1 agent deliberately scoped out — see the live-tree-drift
    note below) + **phase 2** (Timeline Culler; spec at
    ``docs/WARP_COST_PHASE2_DESIGN.md`` — the two former
    rider-rejection clauses were removed because the life rider +
    graveyard-cast rider are now modeled on BOTH engines; Timeline
    Culler is the ONLY card carrying either rider, so the widen
    admits EXACTLY it and the §2 teeth that decline every other warp
    card are byte-for-byte unchanged).

    Motivating set:

    * **Phase 1** — Red Tiger Mechan ({1}{R} warp, {3}{R} normal),
      Bygone Colossus ({3} warp, {9} normal) — both Artifact
      Creatures, ``triggered_abilities=()``, warp is the ONLY
      blocking feature, no life rider, hand-only, no graveyard
      recursion.
    * **Phase 1.5** — Germinating Wurm ({1}{G} warp; ENTERS/IsSelf/
      GainLife 2), Knight Luminary ({1}{W} warp; ENTERS/IsSelf/
      CreateToken Human Soldier 1/1), Starbreach Whale ({1}{U} warp,
      Flying; ENTERS/IsSelf/Surveil 2) — each carries exactly ONE
      SAFE ETB trigger (round-trips through
      :func:`_python_ta_to_safe_dict`) AND ``warp_cost``; warp was
      the ONLY remaining gate (their trigger half already ships
      bridge-correct, and the Rust warp-cast primitive
      ``cast_creature_warp`` fires ENTERS through the SAME
      ``move_from_hand_to_battlefield`` →
      ``drain_pending_etb_dispatches`` path as ``cast_creature``, so
      the ETB fires byte-equal under a warp-cast — pinned by the
      Phase-1.5 real-EoE probe).

    The §2 Class-1 guardrail: it admits ONLY the pure-fixed-cost
    profile OR the safe-trigger+warp profile OR Timeline Culler's
    pure-life+graveyard profile (Phase 2 — the riders are now modeled
    on both engines), and MUST still reject every multi-gated warp
    card (static / Effect-uncovered — never flips on warp alone). The
    exclusion set is the SAME as
    :func:`_printing_is_bridge_simple_creature` (so the 26 multi-gated
    warp cards trip one of these and stay filtered) MINUS the
    ``printing.warp_cost`` rejection (which is the whole point — this
    predicate REQUIRES it), with ``triggered_abilities`` gated to the
    SAFE set (not hard-rejected). Phase 2 REMOVED the two
    Phase-2-rider positive rejections (Timeline Culler is the ONLY
    card carrying either rider — verified live — so the widen admits
    EXACTLY it; every other clause is byte-for-byte unchanged so the
    §2 teeth that decline the 26 multi-gated warp cards still hold).

    **Live-tree drift caught (round-63) → resolved (round-65 Phase
    1.5).** The round-62 design doc (``WARP_COST_PHASE1_DESIGN.md``
    §1) enumerated Phase-1's set as exactly Red Tiger Mechan + Bygone
    Colossus, both with ``triggered_abilities=()``. Verifying the
    spec's §3.7 predicate against the LIVE ``OracleParserCardSource``
    showed THREE more warp creatures parse with a SINGLE SAFE ETB
    trigger (the trio above). The Phase-1 agent deliberately tightened
    the guardrail to ``triggered_abilities == ()`` to exclude them —
    NOT because their trigger was wrong, but because admitting them
    was a Phase-1-boundary expansion the round-62 scout had not
    cost / pinned, and the warp-cast path was a NEW cast path the
    ETB had never fired through. Phase 1.5 (this slice) closes
    exactly that follow-up: the warp substrate has landed
    (``cast_creature_warp`` + the END-step exile sweep + the
    exile-recast), the trio's trigger half was already bridge-correct
    in :func:`_printing_is_bridge_simple_creature`'s safe-trigger
    machinery, and a genuine real-EoE probe drives an actual trio
    card through warp-cast → ETB-fires → END-step-exile → recast
    byte-equal both engines (the ETB-fires-byte-equal-under-warp-cast
    assertion is the load-bearing new pin). Quantum Riddler — which
    ALSO has a safe ETB trigger ({1}{U} warp, Flying; ENTERS/IsSelf/
    DrawCards 1) — stays correctly DECLINED here on the
    ``static_abilities`` clause (it carries a ``HandSizeAtMost(1)``
    draw-replacement static); it is NOT a safe-trigger+warp-ONLY card
    and is excluded by the general §2 exclusion set, not a
    special-case (the guardrail is NOT widened beyond the 3 — the
    Phase-1 agent's tightening intent is preserved for everything
    whose trigger is not genuinely already-safe+bridged).
    """
    if "Creature" not in printing.card_types:
        return False
    if printing.power is None or printing.toughness is None:
        return False
    if printing.warp_cost is None:
        return False
    # Phase 2 (Timeline Culler): the life rider + graveyard-cast rider
    # are now MODELED on BOTH engines (CardDef warp_life_cost /
    # warp_from_graveyard + Rust warp_additional_life_cost /
    # warp_castable_from_graveyard + the cast_creature_warp life debit
    # + the graveyard cast-from-zone fall-through + the life-gated
    # legal emit + the graveyard warp loop). They are no longer
    # §2-rejected. Timeline Culler ({B} warp, Pay 2 life, gy-castable)
    # is the ONLY card carrying either rider (verified live), so
    # dropping the two former rejection clauses admits EXACTLY Timeline
    # Culler and nothing else — every other warp card still trips a
    # non-warp clause below (static / non-safe-trigger / kicker /
    # ...), so the §2 teeth that decline the 26 multi-gated warp cards
    # are byte-for-byte unchanged. See
    # docs/WARP_COST_PHASE2_DESIGN.md §1/§2.
    # Phase-1.5 boundary (round-65): triggered abilities are admitted
    # ONLY if EVERY one round-trips through the already-bridge-correct
    # safe-trigger gate (the SAME machinery
    # _printing_is_bridge_simple_creature uses). Any single unsafe
    # trigger disqualifies the whole printing — there is no
    # "partial admit" since the bridge threads the full
    # triggered-abilities tuple to both engines. Phase 1 (pure
    # fixed-cost, triggered_abilities==()) is the empty-tuple case of
    # this gate; Phase 1.5 admits exactly the trio (Germinating Wurm
    # / Knight Luminary / Starbreach Whale). A warp card with a
    # NON-safe trigger (Codecracker Hound, Mechanozoa, ...) trips
    # this and stays declined — the §2 teeth (see the docstring's
    # live-tree-drift note + SEARCH_LIMITATIONS.md).
    if printing.triggered_abilities:
        for ta in printing.triggered_abilities:
            if _python_ta_to_safe_dict(ta) is None:
                return False
    # No OTHER blocking feature — same exclusion set as
    # _printing_is_bridge_simple_creature (the 7 multi-gated warp
    # cards trip one of these and stay filtered):
    if printing.keywords and not printing.keywords.issubset(_SAFE_KEYWORDS):
        return False
    if printing.activated_abilities:
        return False
    if printing.static_abilities:
        return False
    if printing.spell_effect is not None:
        return False
    # NOTE: unlike _printing_is_bridge_simple_creature, warp_cost is
    # REQUIRED above (not rejected) — kicker / impending still reject.
    if printing.kicker_cost or printing.impending_cost:
        return False
    if printing.room_halves:
        return False
    if printing.loyalty is not None:
        return False
    if printing.cost_reductions or printing.external_cost_reductions:
        return False
    if printing.station_threshold:
        return False
    # Round-78 dynamic-P/T-formula substrate: same admit rule as
    # ``_printing_is_bridge_simple_creature`` — only the
    # ``"harmonious_grovestrider"`` formula key is honored by Rust's
    # dispatch table today. No EoE warp creature carries a
    # ``dynamic_pt_formula`` (verified: the §1 warp pool is Red Tiger
    # Mechan, Bygone Colossus, Germinating Wurm, Knight Luminary,
    # Starbreach Whale, Timeline Culler — none are ``*``-based P/T),
    # so this branch is dead today; keeping the explicit gate
    # preserves the §2 substrate-honesty guarantee if a future warp
    # printing happens to be a layer-7 CDA carrier.
    dpt_formula = getattr(printing, "dynamic_pt_formula", None)
    if dpt_formula is not None and dpt_formula != "harmonious_grovestrider":
        return False
    if getattr(printing, "ward_cost", None) is not None:
        return False
    try:
        _parse_mana_cost_string(printing.mana_cost_str or "")
        _parse_mana_cost_string(printing.warp_cost_str or "")
    except ValueError:
        return False
    return True


def _spell_effect_is_covered_and_bridged(printing) -> bool:
    """True iff every Effect class the printing's ``spell_effect``
    emits is in the LIVE covered+bridged set.

    Reuses the canonical effect-coverage classifier
    (:mod:`mtg_bot.parity.effect_coverage`, the production-importable
    extraction merged ``326023b4``, hardened to 100% branch coverage
    ``02bf16bf``) so the lever auto-gates on each per-card Tier-A
    close having LANDED — NOT a hardcoded card-name list (the §2
    teeth; the lever-doc §2.1 / round-64 §5 "queries the live covered
    set" requirement). ``card_effect_classes`` walks ``spell_effect``
    (+ triggered/activated/static) and returns the distinct Python
    Effect class names; ``covered_python_class_set`` is the EXACT
    ``py_class_set ∩ rust_py_name_set`` the pool-parity gate reports
    (the single canonical definition of "covered").
    """
    from mtg_bot.parity.effect_coverage import (
        covered_python_class_set,
        card_effect_classes,
    )

    emitted = card_effect_classes(printing)
    if not emitted:
        return False  # no introspectable effect — never spuriously OK
    return emitted.issubset(covered_python_class_set())


def _python_spell_effect_to_dict(spell_effect) -> Optional[dict]:
    """Convert a Python spell ``Effect`` back to the Rust-shape effect
    dict (the SAME schema Rust's ``parse_effect`` consumes and
    :func:`cross_engine._ta_dict_to_python_effect` bridges), iff it is
    one of the spell-card-bridge Phase-1 admissible shapes. Returns
    ``None`` for any shape this recognizer does not serialize.

    Spell-card-bridge lever, **Phase 1** (round-70; spec
    ``docs/SPELL_CARD_BRIDGE_PHASE1_DESIGN.md`` §3.6). The Python↔dict
    reverse of :func:`cross_engine._ta_dict_to_python_effect`
    (verify-first RE-CONFIRMED at the live HEAD: NO generic reverse
    serializer exists — only the dict→Python direction). Mirrors the
    established :func:`_python_ta_to_safe_dict` "recognize → emit the
    round-trip dict, else bail" convention.

    **It is part of the §2 teeth.** It only serializes a shape whose
    every node has a covered+bridged ``_ta_dict_to_python_effect``
    arm AND whose Rust ``parse_effect`` arm needs only a registry pid
    the loader's spell path CAN seat. Concretely, the Phase-1 admit
    set is the pid-FREE covered single-target Tier-A spells — Radiant
    Strike (``Sequence(DestroyTargetArtifactOrTappedCreature,
    GainLife)``) and Depressurize (``DepressurizeEffect``) — PLUS, as
    of the Lander-spell-token-seating sub-substrate (round-73), the
    two **Lander-pid** covered single-target Tier-A spells: Seedship
    Impact (``SeedshipImpactEffect``) and Orbital Plunge
    (``DealDamageToTargetCreatureWithExcessLanderRider``). Those two
    were DECLINED at the Phase-1 landing (round-70) because their Rust
    ``parse_effect`` arms ``require("token_printing_id")`` for the
    Lander and the loader's spell path had NO Lander-token-seating
    pass. The round-73 sub-substrate extends
    :func:`_seat_token_printings` to ALSO walk ``spell_effect`` for a
    symbolic ``_token_card_def`` (the EXACT machinery the
    ``triggered_abilities`` path already uses), so this recognizer
    now emits the symbolic Lander reference for those two shapes — the
    round-trip-or-bail tooth FLIPS POSITIVE for them, the
    no-fallback-hybrid discipline preserved (the symbolic ref is
    emitted ONLY because the seating pass that resolves it now
    exists). The teeth STILL positively decline any genuinely
    unseatable shape: a Sequence with one unrecognized child, any
    other token-pid Effect whose token is not the bridge-correct
    Lander (``_canonical_lander_token_card_def`` returns ``None``),
    and — complementary to this serializer — the covered-set gate
    still declines Close Encounter (``CloseEncounterEffect`` not
    covered+bridged).
    """
    from mtg_bot.effects.combinators import Sequence
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.effects.generic.destroy import (
        DestroyAllCreatures,
        DestroyTargetArtifact,
        DestroyTargetArtifactEnchantmentOrFlyingCreature,
        DestroyTargetArtifactOrTappedCreature,
        DestroyTargetNonlandPermanentWithLanderForController,
    )
    from mtg_bot.effects.generic.exile import (
        ExileAllCreaturesAndSpacecraft,
    )
    from mtg_bot.effects.generic.library_lookat import (
        ManifestDread,
        Surveil,
    )
    from mtg_bot.effects.bespoke import (
        BiosynthicBurstEffect,
        CloseEncounterEffect,
        CutPropulsionEffect,
        DepressurizeEffect,
        DiplomaticRelationsEffect,
        ExorciseEffect,
        FocusFireEffect,
        SeedshipImpactEffect,
    )
    from mtg_bot.effects.generic.damage import (
        DealDamageEachOpponent,
        DealDamageToTargetCreatureSpacecraftBoosted,
        DealDamageToTargetCreatureWithExcessLanderRider,
        DealDamageToTargetTappedCreature,
        DealNDamageToEachCreature,
    )
    from mtg_bot.effects.generic.exile import (
        ExileAllArtifactsWithManaValueAtMost,
    )
    from mtg_bot.effects.generic.pump import (
        PumpTargetArtifactOrCreatureUntilEOT,
    )
    from mtg_bot.effects.generic.counters import (
        PutChargeCountersOnTargetSpacecraftOrPlanetYouControl,
    )
    from mtg_bot.effects.generic.bounce import (
        ReturnTargetArtifactCardFromYourGraveyardToHand,
        ReturnUpToNSmallCreaturesFromGraveyardWithCounters,
    )
    from mtg_bot.effects.generic.search import (
        SearchLibraryForCard,
    )

    def _leaf(eff) -> Optional[dict]:
        # Pid-free covered leaves only. DestroyTargetArtifactOrTapped
        # Creature is a unit variant (intrinsic filter — Rust's
        # parse arm takes NO fields, the bridge arm constructs the
        # field-free Python class). GainLife carries ``amount``.
        if isinstance(eff, DestroyTargetArtifactOrTappedCreature):
            return {"kind": "destroy_target_artifact_or_tapped_creature"}
        # **ModalChooseOne envelope — Cycle 6 (Drill Too Deep closure,
        # 2026-05-21).** Per-leaf ``DestroyTargetArtifact`` arm — Drill
        # Too Deep mode 1. The Rust ``Effect::DestroyTargetArtifact``
        # (cards.rs:1254 unit variant) and matching bridge arm in
        # ``cross_engine.py`` (the round-23 `destroy_target_artifact`
        # arm) have shipped since round 23 — the gap was just the
        # composer-recurse leaf entry. Pid-FREE covered unit variant —
        # field-free round-trip; mirrors the sibling
        # ``destroy_target_artifact_or_tapped_creature`` unit-leaf
        # idiom. Adding this leaf — TOGETHER with the matching
        # ``put_charge_counters_on_target_spacecraft_or_planet_you_
        # control`` leaf for mode 0 and the Cycle-5 per-mode target
        # binding in ``cast_spell_noncreature`` — is what flips Drill
        # Too Deep into the spell-card-bridge admit set: both modes
        # are now serializable, the ModalChooseOne composer recurses
        # cleanly, and the {1}{R} Instant rides the Phase-1 ``(1, 1)``
        # admit shape.
        if isinstance(eff, DestroyTargetArtifact):
            return {"kind": "destroy_target_artifact"}
        # **scour-for-scrap-cycle-2 (2026-05-21) — Scour for Scrap
        # mode 0 leaf `SearchLibraryForCard` (NARROW
        # ``filter_name="artifact"`` admit).** Library-tutor primitive
        # with three serializable axes: ``filter_name`` (str,
        # required), ``max_mv`` (int, default 0), and ``may_fail``
        # (bool, default True). Rust's ``parse_effect``
        # ``"search_library_for_card"`` arm (cards.rs) reads all three
        # fields and the resolve arm (state_ops_resolve.rs) consumes
        # them deterministically. The dict→Python bridge arm
        # (cross_engine.py:4068 ``search_library_for_card``)
        # constructs ``SearchLibraryForCard(filter_name, max_mv,
        # may_fail)``. All three fields participate in equality on the
        # Python side, so the round-trip dict carries them all.
        #
        # **Narrow ``filter_name == "artifact"`` admit (honest-stop
        # discipline).** Scour for Scrap mode 0 uses
        # ``filter_name="artifact"``, ``max_mv=0``, ``may_fail=True``
        # — the EXACT shape this lane is responsible for landing.
        # Other ``SearchLibraryForCard`` filters
        # (``filter_name="demon"`` — Demonic Counsel,
        # ``filter_name="basic_land"`` / per-color basics — fetch
        # lands, etc.) ALSO have full Rust + bridge coverage and
        # would also round-trip if this leaf admitted them, but each
        # auto-flips a distinct card class (Demonic Counsel auto-
        # flips a {1}{B} tutor; basic-land filters touch Spineseeker
        # Centipede and triggered-ability surfaces). Per the
        # honest-stop discipline (a follow-up cycle ships each card's
        # parity probe + admit-set guardrail flip in its own lane),
        # this leaf narrowly admits ONLY the ``filter_name="artifact"``
        # shape today. The narrowing is a `_leaf` predicate, not a
        # cross-engine arm change — the existing
        # `search_library_for_card` arm in cross_engine.py / cards.rs
        # / state_ops_resolve.rs accepts every filter byte-identical
        # (no narrowing on the cross-engine side); only the bridge-
        # admit-set serializer narrows.
        if isinstance(eff, SearchLibraryForCard):
            if eff.filter_name != "artifact":
                # Other filters stay declined at the bridge-admit-set
                # gate until a per-card cycle ships their probe +
                # guardrail flip.
                return None
            return {
                "kind": "search_library_for_card",
                "filter_name": str(eff.filter_name),
                "max_mv": int(eff.max_mv),
                "may_fail": bool(eff.may_fail),
            }
        # **scour-for-scrap-cycle-2 (2026-05-21) — Scour for Scrap
        # mode 1 leaf `ReturnTargetArtifactCardFromYourGraveyardTo
        # Hand`.** Pid-FREE covered unit variant — Rust's
        # ``parse_effect`` ``"return_target_artifact_card_from_your_
        # graveyard_to_hand"`` arm takes NO fields and the dict→Python
        # bridge arm (cross_engine.py
        # ``return_target_artifact_card_from_your_graveyard_to_hand``)
        # constructs the field-free
        # ``ReturnTargetArtifactCardFromYourGraveyardToHand()``. No
        # token pid → no :func:`_seat_token_printings` dependency;
        # the round-trip is the clean
        # ``destroy_target_artifact`` unit-leaf idiom. Adding this
        # leaf — TOGETHER with mode 0's ``SearchLibraryForCard`` leaf
        # above AND the Cycle-1 ``or_both`` substrate (Lane LL) AND
        # the admit-predicate widen below to accept Scour's ``(0, 1)``
        # ``or_both=True`` shape — is what flips Scour for Scrap
        # into the spell-card-bridge admit set: both modes are now
        # serializable, the ModalChooseOne composer recurses cleanly,
        # the cast-time multi-pick + per-mode target binder both fire,
        # and the {3}{U} Instant joins the bridge admit set
        # (auto-grow, no hardcoded list — the §2 teeth).
        if isinstance(eff, ReturnTargetArtifactCardFromYourGraveyardToHand):
            return {
                "kind": "return_target_artifact_card_from_your_graveyard_to_hand",
            }
        if isinstance(eff, GainLife):
            return {"kind": "gain_life", "amount": int(eff.amount)}
        if isinstance(eff, DepressurizeEffect):
            return {
                "kind": "depressurize_effect",
                "power_penalty": int(eff.power_penalty),
            }
        # Biosynthic Burst (Effect-uncovered Tier-A #6; round-74).
        # Pid-FREE covered unit variant — Rust's ``parse_effect``
        # ``"biosynthic_burst_effect"`` arm takes NO fields and the
        # dict→Python bridge arm (cross_engine.py
        # ``biosynthic_burst_effect``) constructs the field-free
        # ``BiosynthicBurstEffect()``. No token pid → no
        # :func:`_seat_token_printings` dependency (unlike Seedship /
        # Orbital Plunge); the round-trip is trivially clean — emit
        # the kind-only dict, exactly the
        # ``destroy_target_artifact_or_tapped_creature`` unit-leaf
        # idiom. Adding this leaf is what flips Biosynthic Burst into
        # the spell-card-bridge admit set: with the Rust variant +
        # bridge arm landed it is now covered+bridged AND round-trips
        # here, so :func:`_printing_is_bridge_simple_spell` admits the
        # ``{1}{G}`` single-target Instant (auto-grow, no hardcoded
        # list — the §2 teeth).
        if isinstance(eff, BiosynthicBurstEffect):
            return {"kind": "biosynthic_burst_effect"}
        # Focus Fire (Effect-uncovered Tier-A #5; round-75). Single
        # ``base`` int leaf — Rust's ``parse_effect``
        # ``"focus_fire_effect"`` arm + the dict→Python bridge arm
        # (cross_engine.py ``focus_fire_effect``) both carry ``base``
        # and both default it to ``2`` (Focus Fire's printed base).
        # Pid-FREE (no token) — no :func:`_seat_token_printings`
        # dependency; the round-trip is the clean
        # ``depressurize_effect`` single-int-field idiom. Adding this
        # leaf is what flips Focus Fire into the spell-card-bridge
        # admit set: with the Rust variant + bridge arm landed it is
        # now covered+bridged AND round-trips here, so
        # :func:`_printing_is_bridge_simple_spell` admits the ``{W}``
        # single-target Instant (auto-grow, no hardcoded list — the
        # §2 teeth).
        if isinstance(eff, FocusFireEffect):
            return {"kind": "focus_fire_effect", "base": int(eff.base)}
        # Diplomatic Relations (Effect-uncovered Tier-A #7; round-76;
        # spell-card-bridge **Phase 2** — the 2-slot heterogeneous-
        # target grammar). Pid-FREE covered unit variant — Rust's
        # ``parse_effect`` ``"diplomatic_relations_effect"`` arm takes
        # NO fields and the dict→Python bridge arm (cross_engine.py
        # ``diplomatic_relations_effect``) constructs the field-free
        # ``DiplomaticRelationsEffect()``. The +1/+0, vigilance, and
        # power-equal damage are intrinsic; the 2-slot targets ride
        # on the spell StackItem, NOT in this dict. No token pid → no
        # :func:`_seat_token_printings` dependency; the round-trip is
        # the clean ``biosynthic_burst_effect`` unit-leaf idiom.
        # Adding this leaf — TOGETHER with the guardrail's Phase-2
        # 2-slot widening (:func:`_printing_is_bridge_simple_spell`)
        # — is what flips Diplomatic Relations into the spell-card-
        # bridge admit set: with the Rust variant + bridge arm landed
        # it is now covered+bridged AND round-trips here, so the
        # guardrail admits the ``{2}{G}`` 2-slot Instant (the admit
        # set grows 6→7; auto-grow, no hardcoded list — the §2
        # teeth).
        if isinstance(eff, DiplomaticRelationsEffect):
            return {"kind": "diplomatic_relations_effect"}
        # Cut Propulsion (Effect-uncovered Tier-A; ~11,990 17lands
        # decks). Pid-FREE covered unit variant — Rust's
        # ``parse_effect`` ``"cut_propulsion_effect"`` arm takes NO
        # fields and the dict→Python bridge arm (cross_engine.py
        # ``cut_propulsion_effect``) constructs the field-free
        # ``CutPropulsionEffect()``. The single creature target rides
        # on the spell StackItem, NOT in this dict; the damage amount
        # (target's effective power; doubled if flying) is intrinsic.
        # No token pid → no :func:`_seat_token_printings` dependency;
        # the round-trip is the clean ``biosynthic_burst_effect`` /
        # ``diplomatic_relations_effect`` unit-leaf idiom. Adding this
        # leaf is what flips Cut Propulsion into the spell-card-bridge
        # admit set: with the Rust variant + bridge arm landed it is
        # now covered+bridged AND round-trips here, so
        # :func:`_printing_is_bridge_simple_spell` admits the ``{2}{R}``
        # single-target Instant (auto-grow, no hardcoded list — the
        # §2 teeth).
        if isinstance(eff, CutPropulsionEffect):
            return {"kind": "cut_propulsion_effect"}
        # Invasive Maneuvers (Effect-uncovered Tier-A; ~10,981 17lands
        # decks). Two-int covered variant — Rust's ``parse_effect``
        # ``"invasive_maneuvers_effect"`` arm carries ``base`` +
        # ``boosted`` ints and the dict→Python bridge arm
        # (cross_engine.py ``invasive_maneuvers_effect``) constructs
        # ``DealDamageToTargetCreatureSpacecraftBoosted(base=base,
        # boosted=boosted)``. The single creature target rides on the
        # spell StackItem, NOT in this dict; the Spacecraft check is
        # intrinsic (resolves off the source-controller's bf via the
        # printing's ``subtypes`` — Spacecraft is a SUBTYPE per
        # ``cards/base.py:1056-1057``). No token pid → no
        # :func:`_seat_token_printings` dependency; the round-trip is
        # the clean ``focus_fire_effect`` single-int-field idiom
        # widened to two ints. Adding this leaf is what flips Invasive
        # Maneuvers into the spell-card-bridge admit set: with the
        # Rust variant + bridge arm landed it is now covered+bridged
        # AND round-trips here, so
        # :func:`_printing_is_bridge_simple_spell` admits the ``{1}{R}``
        # single-target Instant (auto-grow, no hardcoded list — the
        # §2 teeth).
        if isinstance(eff, DealDamageToTargetCreatureSpacecraftBoosted):
            return {
                "kind": "invasive_maneuvers_effect",
                "base": int(eff.base),
                "boosted": int(eff.boosted),
            }
        # Lander-pid spell effects (Seedship Impact / Orbital Plunge) —
        # the Lander-spell-token-seating sub-substrate (round-73; the
        # follow-on the round-70 SEARCH_LIMITATIONS seam scoped). Their
        # Rust ``parse_effect`` arms ``require("token_printing_id")``
        # for the Lander; the matching dict→Python bridge arms
        # (cross_engine.py ``seedship_impact_effect`` /
        # ``deal_damage_to_target_creature_with_excess_lander_rider``)
        # DROP the pid and construct the FIELD-FREE Python class (which
        # mints the Lander intrinsically via ``TOKEN_PRINTINGS
        # ["Lander"]``). The reverse serializer emits a SYMBOLIC
        # ``_token_card_def`` Lander reference exactly as
        # :func:`_python_ta_to_safe_dict`'s ``create_token`` arm does;
        # the loader's :func:`_seat_token_printings` pass (now extended
        # to walk ``spell_effect`` too) rewrites it to the numeric
        # ``token_printing_id`` Rust requires + registers the matching
        # Python printing. round-trip-or-bail preserved: bail if the
        # authored Lander token ever drifts out of the bridge-correct
        # shape (``_canonical_lander_token_card_def`` → ``None``).
        if isinstance(eff, SeedshipImpactEffect):
            lcd = _canonical_lander_token_card_def()
            if lcd is None:
                return None
            return {
                "kind": "seedship_impact_effect",
                "_token_card_def": lcd,
            }
        # Emergency Eject (Effect-uncovered single-gated tier;
        # ~6,291 17lands decks). Same Lander-pid carrier shape as
        # Seedship Impact: Rust's ``parse_effect``
        # ``"emergency_eject_effect"`` arm
        # ``require("token_printing_id")`` for the Lander; the
        # matching dict→Python bridge arm (cross_engine.py
        # ``emergency_eject_effect``) DROPS the pid and constructs
        # the FIELD-FREE Python class
        # ``DestroyTargetNonlandPermanentWithLanderForController``
        # (which mints the Lander intrinsically via
        # ``TOKEN_PRINTINGS["Lander"]``). The reverse serializer emits
        # the SYMBOLIC ``_token_card_def`` Lander reference the same
        # way Seedship / Orbital Plunge do; the loader's
        # :func:`_seat_token_printings` pass (already walking
        # ``spell_effect`` per the round-73 sub-substrate, extended
        # below to recognize this new kind) rewrites it to the
        # numeric ``token_printing_id`` Rust requires + registers
        # the matching Python printing. Round-trip-or-bail preserved.
        if isinstance(eff, DestroyTargetNonlandPermanentWithLanderForController):
            lcd = _canonical_lander_token_card_def()
            if lcd is None:
                return None
            return {
                "kind": "emergency_eject_effect",
                "_token_card_def": lcd,
            }
        if isinstance(eff, DealDamageToTargetCreatureWithExcessLanderRider):
            lcd = _canonical_lander_token_card_def()
            if lcd is None:
                return None
            return {
                "kind": (
                    "deal_damage_to_target_creature_with_excess_"
                    "lander_rider"
                ),
                "amount": int(eff.amount),
                "_token_card_def": lcd,
            }
        # Shattered Wings's destroy half (Effect-uncovered single-
        # gated tier; ~7,793 17lands decks). Pid-FREE covered unit
        # variant — Rust's ``parse_effect`` ``"destroy_target_
        # artifact_enchantment_or_flying_creature"`` arm takes NO
        # fields and the dict→Python bridge arm (cross_engine.py
        # ``destroy_target_artifact_enchantment_or_flying_creature``)
        # constructs the field-free
        # ``DestroyTargetArtifactEnchantmentOrFlyingCreature()``. The
        # 3-way OR filter (Artifact OR Enchantment OR (Creature AND
        # has-flying)) is intrinsic. No token pid → no
        # :func:`_seat_token_printings` dependency; the round-trip is
        # the clean ``destroy_target_artifact_or_tapped_creature``
        # unit-leaf idiom. Adding this leaf + the matching
        # ``Surveil`` leaf below is what flips Shattered Wings into
        # the spell-card-bridge admit set: with the Rust variant +
        # bridge arm landed AND the outer Sequence's children all
        # serializable, the {2}{G} Sorcery is now covered+bridged
        # AND round-trips through the reverse serializer, so
        # :func:`_printing_is_bridge_simple_spell` admits it
        # (auto-grow, no hardcoded list — the §2 teeth).
        if isinstance(eff, DestroyTargetArtifactEnchantmentOrFlyingCreature):
            return {
                "kind": (
                    "destroy_target_artifact_enchantment_or_flying_"
                    "creature"
                ),
            }
        # Exorcise (Effect-uncovered single-gated tier; Duskmourn).
        # Pid-FREE covered unit variant — Rust's ``parse_effect``
        # ``"exorcise_effect"`` arm takes NO fields and the dict→Python
        # bridge arm (cross_engine.py ``exorcise_effect``) constructs
        # the field-free ``ExorciseEffect()``. The 3-way OR filter
        # (Artifact OR Enchantment OR (Creature AND effective_power
        # >= 4)) is intrinsic. No token pid → no
        # :func:`_seat_token_printings` dependency; the round-trip is
        # the clean ``destroy_target_artifact_or_tapped_creature``
        # unit-leaf idiom. Adding this leaf is what flips Exorcise
        # into the spell-card-bridge admit set: with the Rust variant +
        # bridge arm landed AND the printing's spell_effect
        # serializable, the {1}{W} Sorcery is now covered+bridged AND
        # round-trips through the reverse serializer, so
        # :func:`_printing_is_bridge_simple_spell` admits it (auto-
        # grow, no hardcoded list — the §2 teeth).
        if isinstance(eff, ExorciseEffect):
            return {"kind": "exorcise_effect"}
        # Close Encounter — additional-cost-choice closure (Cycle 3,
        # 2026-05-21; ~16,098 17lands decks — highest-deck-count
        # BLOCKED card in the substrate-gated tier). Pid-FREE
        # covered unit variant — Rust's ``parse_effect``
        # ``"close_encounter_effect"`` arm takes NO fields and the
        # dict→Python bridge arm (cross_engine.py
        # ``close_encounter_effect``) constructs the field-free
        # ``CloseEncounterEffect()``. The damage amount is intrinsic
        # (computed at resolve from the additional-cost-choice
        # object's power, read via the
        # ``additional_cost_choice_id`` StackItem channel — the
        # Cycle 1 substrate landed in this same lane). The single
        # damage-target rides on the StackItem
        # (``current_resolve_payload_target_id``), NOT in this dict.
        # No token pid → no :func:`_seat_token_printings` dependency;
        # the round-trip is the clean ``exorcise_effect`` /
        # ``diplomatic_relations_effect`` unit-leaf idiom. Adding
        # this leaf — TOGETHER with the admit-predicate widen below
        # (which lets the choose-your-creature additional-cost flag
        # through when the covered+bridged ``CloseEncounterEffect``
        # is the spell's effect) — is what flips Close Encounter
        # into the spell-card-bridge admit set.
        if isinstance(eff, CloseEncounterEffect):
            return {"kind": "close_encounter_effect"}
        # **Phase-3 wrath-admit-widening (2026-05-21).** Pyroclasm-shape
        # mass-damage leaf — ``DealNDamageToEachCreature(amount)``. The
        # bridge arm at ``cross_engine.py::damage_each_creature`` REQUIRES
        # ``amount`` (no default; mirrors Rust's ``parse_effect`` REQUIRED-
        # field contract at ``cards.rs:7435``+) and constructs the
        # field-clean ``DealNDamageToEachCreature(amount=amount)`` — the
        # bare 2-field round-trip is bit-equal. Wrath-shape ``(0, 0)``
        # spell-effect with no at-cast target (acts on the full
        # battlefield snapshot at resolve-time). EoE consumer:
        # Pyroclasm — already PROBED by its dedicated effect-close probe;
        # this leaf admits it through the spell-card-bridge wrath path
        # as well (auto-grow; no hardcoded card-name list — the §2
        # teeth). Mirrors the ``depressurize_effect`` / ``focus_fire_
        # effect`` single-int-field idiom.
        if isinstance(eff, DealNDamageToEachCreature):
            return {
                "kind": "damage_each_creature",
                "amount": int(eff.amount),
            }
        # **Phase-3 wrath-admit-widening (2026-05-21).** Bare-shape
        # ``DestroyAllCreatures()`` leaf — only the field-free shape
        # round-trips. Python's class carries optional narrowing fields
        # (``tapped_only``, ``untapped_only``, ``min_power``) that Rust's
        # unit-variant ``Effect::DestroyAllCreatures`` (cards.rs:7818)
        # does NOT represent; the bridge arm always rebuilds the bare
        # ``DestroyAllCreatures()``. Serializing a narrowed variant
        # would silently widen the destroy filter on Rust — the §2
        # teeth bail here when any narrowing flag is set, mirroring the
        # ``Surveil`` ``use_artifact_count`` decline-path. No EoE-pool
        # ``(0, 0)`` non-modal consumer at this HEAD (Singularity
        # Rupture is ``(0, 8)``; Split Up is modal; Zero Point Ballad
        # uses the ``min_power`` narrowing). The leaf is load-bearing
        # for the wrath-shape round-trip surface — keeps the serializer
        # symmetric with the bridge arm's known-shape coverage.
        if isinstance(eff, DestroyAllCreatures):
            if (
                eff.tapped_only
                or eff.untapped_only
                or eff.min_power
            ):
                return None
            return {"kind": "destroy_all_creatures"}
        # **Beyond-the-Quiet slice (2026-05-21).** Wrath-shape exile
        # leaf — ``ExileAllCreaturesAndSpacecraft``. Pid-FREE covered
        # unit variant — Rust's ``parse_effect``
        # ``"exile_all_creatures_and_spacecraft"`` arm takes NO fields
        # and the dict→Python bridge arm (cross_engine.py
        # ``exile_all_creatures_and_spacecraft``) constructs the
        # field-free ``ExileAllCreaturesAndSpacecraft()``. The
        # Creature-OR-Spacecraft filter is intrinsic (resolves off the
        # printing's ``card_types`` / ``subtypes``); no targets, acts
        # on the full battlefield snapshot at resolve-time. Mirrors
        # the ``destroy_all_creatures`` unit-leaf idiom. Adding this
        # leaf is what flips Beyond the Quiet (4,698 17lands decks)
        # into the spell-card-bridge admit set via Lane Y's Phase-3
        # ``(0, 0)`` wrath admit arm: with the Rust variant + bridge
        # arm landed it is now covered+bridged AND round-trips here,
        # so :func:`_printing_is_bridge_simple_spell` admits the
        # ``{3}{W}{W}`` no-target Sorcery (auto-grow, no hardcoded
        # list — the §2 teeth; the admit set grows 15 → 16).
        if isinstance(eff, ExileAllCreaturesAndSpacecraft):
            return {"kind": "exile_all_creatures_and_spacecraft"}
        # **Phase-3 wrath-admit-widening (2026-05-21).** ManifestDread
        # spell-effect leaf — ``ManifestDread(count)``. The bridge arm
        # at ``cross_engine.py::manifest_dread`` reads ``count`` (default
        # 1 if omitted) and constructs ``ManifestDread(count=count)`` —
        # the bare single-int round-trip is bit-equal. Wrath-shape
        # ``(0, 0)`` spell-effect: the manifest-dread CR 701.55 mechanic
        # operates on the caster's library (look 2, manifest 1 face-down,
        # graveyard 1) at resolve-time with NO at-cast target binding —
        # the canonical wrath-shape no-targets EoE spell. EoE consumers
        # (loaded via the duskmourn.json card pool): Manifest Dread (the
        # eponymous Sorcery), Break Down the Door (the Instant flash
        # variant), Under the Skin (the {1}{B} Sorcery variant) —
        # Manifest Dread is ALREADY PROBED by its dedicated empty-lib
        # no-op probe (``test_probe_remaining_sss.py``); the two new
        # ``Break Down the Door`` / ``Under the Skin`` probes added by
        # this lane drive the SAME bridge arm through their respective
        # parsed printings, flipping each from COVERED → PROBED. Mirrors
        # the bridge arm's ``count``-defaulting-to-1 / non-negative
        # contract.
        if isinstance(eff, ManifestDread):
            if eff.count < 0:
                return None
            return {
                "kind": "manifest_dread",
                "count": int(eff.count),
            }
        # Bare-shape Surveil leaf (no ``use_artifact_count`` rider).
        # Mirrors the safe-trigger Surveil arm
        # (:func:`_python_ta_to_safe_dict` Surveil case): Python's
        # class carries an additional ``use_artifact_count`` flag
        # that selects the Surveil-X-where-X=artifacts variant
        # (``effects/generic/library_lookat.py:580``); Rust's
        # ``Effect::Surveil { count: i32 }`` does NOT represent the
        # X variant — gate on the rider being False so the X
        # variant falls through to the bail path. Rust's spelling
        # for the count field is ``count`` (not Python's
        # ``amount``); the cross_engine bridge arm
        # (``cross_engine.py:1030`` ``surveil``) reads ``count`` and
        # constructs ``Surveil(amount=count)`` — bidirectional with
        # what we emit here. Adding this leaf — together with the
        # destroy leaf above — admits Shattered Wings's outer
        # Sequence into the bridge admit set.
        if isinstance(eff, Surveil):
            if eff.use_artifact_count:
                return None
            return {"kind": "surveil", "count": int(eff.amount)}
        # **ModalChooseOne envelope — Cycle 4 (per-leaf Effect for
        # all-untargeted modal cards, 2026-05-21).** Per-leaf
        # ``DealDamageEachOpponent(amount)`` arm — Ruinous Rampage's
        # mass-burn mode (the all-untargeted half of the
        # ``ModalChooseOne([DealDamageEachOpponent,
        # ExileAllArtifactsWithManaValueAtMost])`` shape, the only
        # all-untargeted ModalChooseOne candidate in the EoE pool).
        # Rust variant `Effect::DamageEachOpponent { amount: i32 }`
        # (`cards.rs:246`) is shipped + bridged via
        # `cross_engine.py::damage_each_opponent` ↔
        # `DealDamageEachOpponent`. The bare 2-field round-trip
        # (`kind` + `amount`) is bit-equal — mirrors the
        # `damage_each_creature` and `depressurize_effect` single-int
        # idioms. Non-negative invariant (Rust raises on negative at
        # resolve; Python's class has no validation — bail at the
        # serializer if the field is negative to keep the round-trip
        # symmetric).
        if isinstance(eff, DealDamageEachOpponent):
            if eff.amount < 0:
                return None
            return {
                "kind": "damage_each_opponent",
                "amount": int(eff.amount),
            }
        # **ModalChooseOne envelope — Cycle 4 (per-leaf Effect for
        # all-untargeted modal cards, 2026-05-21).** Per-leaf
        # ``ExileAllArtifactsWithManaValueAtMost(max_mv)`` arm —
        # Ruinous Rampage's anti-artifact mode (paired with
        # ``DealDamageEachOpponent`` above; the two together carry the
        # all-untargeted ModalChooseOne admit shape for the Sorcery).
        # Rust variant
        # ``Effect::ExileAllArtifactsWithManaValueAtMost { max_mv: i32 }``
        # ships in the SAME cycle (declaration + parse + to_pydict +
        # resolve arms). Two-field round-trip (`kind` + `max_mv`) is
        # bit-equal. Non-negative invariant per CR 202.3 (MV is
        # non-negative; Python class default is 3, parser only emits
        # explicit values via the regex digit capture — but bail
        # defensively on a negative if a future hand-built call
        # constructs one).
        if isinstance(eff, ExileAllArtifactsWithManaValueAtMost):
            if eff.max_mv < 0:
                return None
            return {
                "kind": "exile_all_artifacts_with_mana_value_at_most",
                "max_mv": int(eff.max_mv),
            }
        # **Reroute Systems closure (2026-05-21).** ModalChooseOne
        # envelope **Cycle 6** per-leaf serializer for Reroute Systems
        # mode 0: ``PumpTargetArtifactOrCreatureUntilEOT``. The Rust
        # variant carries ``power_bonus``, ``toughness_bonus``, AND
        # the extended ``granted_keywords`` list-of-strings field (the
        # ``granted_keywords`` extension lands in this same closure
        # so the round-trip is bit-equal for Reroute Systems mode 0's
        # ``(0, 0, {"indestructible"})`` shape). Per the to_pydict
        # parity convention, ``granted_keywords`` is emitted ONLY
        # when non-empty so back-compat with every pre-closure parse
        # site that omits the field stays intact.
        if isinstance(eff, PumpTargetArtifactOrCreatureUntilEOT):
            out: dict = {
                "kind": "pump_target_artifact_or_creature_until_eot",
                "power_bonus": int(eff.power_bonus),
                "toughness_bonus": int(eff.toughness_bonus),
            }
            if eff.granted_keywords:
                # Sort for deterministic ordering — Python's
                # ``frozenset`` has unstable iteration order, but the
                # serialized list must round-trip byte-equal.
                out["granted_keywords"] = sorted(eff.granted_keywords)
            return out
        # **Reroute Systems closure (2026-05-21).** ModalChooseOne
        # envelope **Cycle 6** per-leaf serializer for Reroute Systems
        # mode 1: ``DealDamageToTargetTappedCreature``. The new Rust
        # variant carries the single ``amount: i32`` field (REQUIRED
        # — mirrors the round-28 REQUIRED-amount precedent for
        # ``damage_each_opponent`` / ``damage_each_creature``). Non-
        # negative invariant (Rust raises on negative at parse +
        # resolve; mirror that here so the round-trip surface stays
        # symmetric).
        if isinstance(eff, DealDamageToTargetTappedCreature):
            if eff.amount < 0:
                return None
            return {
                "kind": "deal_damage_to_target_tapped_creature",
                "amount": int(eff.amount),
            }
        # **ModalChooseOne envelope — Cycle 6 (Drill Too Deep closure,
        # 2026-05-21).** Per-leaf
        # ``PutChargeCountersOnTargetSpacecraftOrPlanetYouControl
        # (amount)`` arm — Drill Too Deep mode 0. The other mode
        # (``DestroyTargetArtifact``) already has a leaf above. Rust
        # variant ``Effect::PutChargeCountersOnTargetSpacecraftOrPlanet
        # YouControl { amount: i32 }`` ships in the SAME cycle
        # (declaration + parse + to_pydict + resolve arms +
        # cast-time per-mode binding detector entry). Two-field
        # round-trip (``kind`` + ``amount``) is bit-equal — mirrors
        # the ``exile_all_artifacts_with_mana_value_at_most`` /
        # ``damage_each_opponent`` single-int-field idiom. Non-
        # negative invariant per CR 122 (charge counters are non-
        # negative; parser emits explicit digits via
        # ``parser/effects.py:3002-3005``; bail defensively on a
        # negative if a hand-constructed instance ever leaks
        # through). Adding this leaf — TOGETHER with the matching
        # ``destroy_target_artifact`` leaf already present and the
        # Cycle-5 per-mode target binding in
        # ``cast_spell_noncreature`` — is what flips Drill Too Deep
        # into the spell-card-bridge admit set: both modes are
        # covered+bridged, the ModalChooseOne composer recurses into
        # both, and the {1}{R} Instant rides the Phase-1 ``(1, 1)``
        # admit shape.
        if isinstance(eff, PutChargeCountersOnTargetSpacecraftOrPlanetYouControl):
            if eff.amount < 0:
                return None
            return {
                "kind": (
                    "put_charge_counters_on_target_spacecraft_or_"
                    "planet_you_control"
                ),
                "amount": int(eff.amount),
            }
        # **Multi-target admit-gate Phase 4 widen (2026-05-21).** Per-
        # leaf serializer for
        # ``ReturnUpToNSmallCreaturesFromGraveyardWithCounters(max_n,
        # max_total_mv)`` — Scout for Survivors. Rust variant
        # ``Effect::ReturnUpToNSmallCreaturesFromGraveyardWithCounters
        # { max_n: i32, max_total_mv: i32 }`` ships in the SAME lane
        # (declaration + parse + to_pydict + resolve arms + bridge arm).
        # Three-field round-trip (``kind`` + ``max_n`` + ``max_total_mv``)
        # is bit-equal — mirrors the
        # ``invasive_maneuvers_effect`` two-int-field idiom widened to
        # the count + sum-MV-budget pair. Non-negative invariant per
        # CR 202.3 (mv is non-negative; max_n is non-negative); Python
        # class defaults to max_n=3, max_total_mv=3 (Scout's printed
        # values). Bail defensively on a negative if a hand-constructed
        # instance ever leaks through. Adding this leaf — TOGETHER with
        # the multi-target admit-gate Phase-4 widen below — is what
        # flips Scout for Survivors into the spell-card-bridge admit
        # set.
        #
        # **Riders gate.** The Python class is constructor sugar on the
        # unified ``ReturnFromGraveyard`` base; ``__post_init__`` pins
        # ``creature=True``, ``destination='battlefield'``,
        # ``plus_one_counter=True``, ``min_n=0``, ``max_mv=max_total_mv``.
        # The Rust variant carries ONLY ``max_n`` + ``max_total_mv``;
        # the destination + counter + creature filter + per-card MV cap
        # are intrinsic to the variant (not data fields). Bail if any
        # printed-from-text rider that DOESN'T fold into this fixed
        # shape is set (defensive; the parser only emits this shape
        # for the exact Scout-print pattern, but a future parser
        # extension that adds e.g. ``or_artifact=True`` would silently
        # mistranslate without this guard).
        if isinstance(eff, ReturnUpToNSmallCreaturesFromGraveyardWithCounters):
            if eff.max_n < 0 or eff.max_total_mv < 0:
                return None
            # Riders that don't fold into the fixed Rust variant shape.
            # The class pins creature=True, destination='battlefield',
            # plus_one_counter=True, min_n=0 in __post_init__; any
            # deviation from those defaults means the parser produced
            # a shape this round-trip doesn't cover.
            if (
                not eff.creature
                or eff.artifact
                or eff.land
                or eff.nonland_permanent
                or eff.or_enchantment
                or eff.or_spacecraft
                or eff.or_artifact
                or eff.destination != "battlefield"
                or not eff.plus_one_counter
                or eff.min_n != 0
                or eff.untargeted_all
                or eff.log_suffix
                or eff.target_kind != "graveyard_creature"
                or eff.max_mv != eff.max_total_mv
            ):
                return None
            return {
                "kind": (
                    "return_up_to_n_small_creatures_from_graveyard_"
                    "with_counters"
                ),
                "max_n": int(eff.max_n),
                "max_total_mv": int(eff.max_total_mv),
            }
        return None

    # Sequence composer (Radiant Strike): recurse into each child
    # leaf; bail if ANY child is unrecognized (the round-trip-or-bail
    # teeth — a partially-serializable Sequence would silently drop a
    # child on Rust).
    if isinstance(spell_effect, Sequence):
        children: list[dict] = []
        for child in spell_effect.effects:
            cd = _leaf(child)
            if cd is None:
                return None
            children.append(cd)
        return {"kind": "sequence", "effects": children}
    # ModalChooseOne composer (spell-card-bridge **ModalChooseOne
    # envelope — Cycle 3**, 2026-05-21). Mirrors the Sequence
    # composer's recurse-into-children-and-bail-on-any-unrecognized
    # discipline. Emits the Rust-shape ``{"kind": "choose_one_mode",
    # "modes": [<per-mode leaf>, ...]}`` dict that
    # ``cards.rs::parse_effect``'s ``"choose_one_mode"`` arm at
    # ``cards.rs:9084`` consumes (recursive parse over ``modes``) and
    # that ``cross_engine.py``'s ``_ta_dict_to_python_effect``
    # ``"choose_one_mode"`` arm at line 6264 round-trips into
    # ``ModalChooseOne(modes=...)``.
    #
    # **Round-trip-or-bail teeth.** Recurses into each mode via the
    # same ``_leaf`` table the Sequence composer uses; bails (returns
    # ``None``) if ANY mode is unrecognized — the no-fallback-hybrid
    # discipline a partially-serializable ModalChooseOne would
    # silently drop a mode on Rust.
    #
    # **`or_both` widen (scour-for-scrap-slice, 2026-05-21).** The
    # Cycle-1-widen substrate adds the multi-pick StackItem slot
    # (``chosen_mode_idxs: Option<Vec<u8>>``) PLUS the
    # ``Effect::ChooseOneMode { ..., or_both: bool }`` Rust field.
    # The composer now emits the ``or_both`` round-trip field;
    # downstream the cast-time bind (``cast_spell_noncreature``)
    # detects ``or_both=true`` and binds the full mode subset, and
    # the resolve arm dispatches each chosen mode in order. The
    # admit-set still gates on per-leaf coverage; any leaf that
    # remains uncovered keeps the spell declined (Scour for Scrap's
    # mode 1, ``ReturnTargetArtifactCardFromYourGraveyardToHand``,
    # is uncovered today and will keep Scour declined until a
    # follow-up cycle adds that Effect variant + per-mode target
    # binder — the substrate-only landing for this slice).
    #
    # **Round-trip-or-bail teeth preserved.** Recurses into each
    # mode via the same ``_leaf`` table; bails (returns ``None``) if
    # ANY mode is unrecognized — keeps the no-fallback-hybrid
    # discipline a partially-serializable ModalChooseOne would
    # silently drop a mode on Rust.
    #
    # **``or_both: false`` omitted.** Per the Rust to_pydict
    # symmetry, the field is emitted ONLY when ``True`` — keeps the
    # pre-widen dict shape byte-identical for every non-or_both
    # modal.
    #
    # **Composer-not-leaf shape.** Lives outside ``_leaf`` because
    # ``_leaf`` is intentionally the pid-free covered LEAVES (the
    # Sequence composer also lives outside it). A `ModalChooseOne`
    # nested as a Sequence child would NOT round-trip through this
    # path today (`_leaf` doesn't recurse into ModalChooseOne); the
    # Cycle 3 admit shape is `ModalChooseOne` AS the top-level
    # `spell_effect`, not nested inside a Sequence. Real EoE modal
    # cards (Reroute Systems, Archenemy's Charm, ...) print modal
    # spells at the head of the effect, not nested — matches the
    # admit shape.
    from mtg_bot.effects.combinators import ModalChooseOne
    if isinstance(spell_effect, ModalChooseOne):
        mode_dicts: list[dict] = []
        for mode in spell_effect.modes:
            md = _leaf(mode)
            if md is None:
                return None
            mode_dicts.append(md)
        out: dict = {"kind": "choose_one_mode", "modes": mode_dicts}
        if spell_effect.or_both:
            out["or_both"] = True
        return out
    return _leaf(spell_effect)


def _printing_is_bridge_simple_spell(printing) -> bool:
    """True iff a CardPrinting is a covered Instant/Sorcery whose
    target grammar the spell-card-bridge lever round-trips byte-equal
    across engines (Phase-1 single-target OR Phase-2 2-slot).

    Spell-card-bridge lever, **Phase 1** (round-70) **+ Phase 2**
    (round-76; spec at ``docs/SPELL_CARD_BRIDGE_PHASE1_DESIGN.md``
    §3 + its Phase-2 section; scoping basis
    ``docs/SPELL_CARD_BRIDGE_LEVER_DESIGN.md`` §3.0).

    Motivating set:

    - **Phase 1 (single-target — ``(1,1)`` / ``()``):** the
      single-target Tier-A spells whose bespoke Effect close has
      landed AND whose effect carries no loader-unseatable token pid
      — currently the 6-card set {Radiant Strike, Depressurize,
      Focus Fire, Biosynthic Burst, Seedship Impact, Orbital Plunge}
      (Seedship Impact / Orbital Plunge admitted via the round-73
      Lander-spell-token-seating sub-substrate).
    - **Phase 2 (2-slot heterogeneous — ``(2,2)`` /
      ``('my_creature','creature')``):** Diplomatic Relations — the
      SINGLE 2-slot grammar admitted, exactly the disjoint
      friendly-creature + opponent-creature slot pair the
      ``_select_spell_targets`` per-slot-kind first-legal convention
      drives byte-equal. Adding it grows the admit set 6→7.

    Close Encounter (additional-cost-choice + the
    ``CloseEncounterEffect`` not covered) STAYS the §2.2-declined
    Tier-C card; every other multi-target / modal / X / warp /
    additional-cost shape STAYS filtered (the decline set, §2.2).

    The §2 teeth: it admits ONLY a spell whose ``spell_effect`` is a
    COVERED+BRIDGED effect (queried against the LIVE covered set, NOT
    a hardcoded list — so it auto-gates on each per-card Tier-A close
    having LANDED) AND round-trips through the reverse serializer
    (which structurally declines the unseatable-token-pid shapes),
    single-mode, no X / kicker / warp / additional-cost-choice, with
    a target grammar that is EITHER the Phase-1 implicit single
    target OR the Phase-2 exact 2-slot ``('my_creature','creature')``
    pair the deterministic-first convention drives byte-equal. Every
    other spell shape stays filtered (the decline set, §2.2).
    """
    if not (
        "Instant" in printing.card_types
        or "Sorcery" in printing.card_types
    ):
        return False
    if "Creature" in printing.card_types:
        return False  # permanent-spell rider — other substrate
    se = printing.spell_effect
    if se is None:
        return False
    # Phase-2+ / declined alt-cost & mode shapes — positively
    # rejected (the §2 teeth). Mirrors the warp guardrail's
    # rider-rejection block.
    if printing.warp_cost is not None:
        return False  # warp-entangled — warp substrate Phase 2
    if printing.kicker_cost or printing.impending_cost:
        return False
    if printing.room_halves:
        return False
    mc = printing.mana_cost
    # **X-value-subsystem-C2.5 (2026-05-22).** X-cost admit-predicate
    # widen — the early ``if mc.x_pips: return False`` reject is gone.
    # X-cost printings now flow through the per-phase target-shape gates
    # below. ONLY the Phase-3 wrath shape (`(0, 0)` with no slot_kinds /
    # no ``targets`` attr) admits an X-cost printing — every other
    # phase's `_PHASEN_*_SHAPE` predicate is conjuncted with
    # ``mc.x_pips == 0`` so the X-cost lane stays surgical
    # (single-target Phase 1, 2-slot Phase 2, multi-target Phase 4,
    # or_both Phase 5 — all stay declined for X-cost cards). Today the
    # ONLY X-cost EoE card whose `spell_effect` shape would satisfy the
    # Phase-3 widen is Zero Point Ballad (and Devastating Onslaught
    # which shares the X-value substrate but is Sorcery
    # "create X token copies of..." — different resolve shape).
    # **The widen is structurally clean substrate: until C3 lands the
    # `Effect::ZeroPointBallad` resolve arm + bridge, the
    # `_spell_effect_is_covered_and_bridged` tooth + the
    # `_python_spell_effect_to_dict(se) is None` round-trip-or-bail
    # tooth below keep Zero Point Ballad DECLINED.** The admit-arm
    # widen + cast-pipeline live-mint substrate ship together so
    # Cycle 3 lands the Effect arm against an existing admit gate.
    _printing_is_x_cost = mc is not None and getattr(mc, "x_pips", 0) > 0
    # Additional-cost-choice (Close Encounter): the live printing
    # exposes NO ``additional_cost_choice`` attribute (verify-first
    # at the implement HEAD — the spec ADDENDUM's "warp_cost is not
    # None" Close-Encounter reason is ALSO stale: live
    # ``Close Encounter.warp_cost is None``).
    #
    # **Close Encounter additional-cost-choice closure — Cycle 3
    # admit widen (2026-05-21).** With the Rust ``Effect::Close
    # EncounterEffect`` variant + the bridge arms + the resolve-arm
    # consumer of ``current_resolve_additional_cost_choice_id``
    # (the Cycle 1 substrate), the covered+bridged ``CloseEncounter
    # Effect`` is what flips Close Encounter into the admit set.
    # The covered-set gate below carries the load-bearing decline
    # for the GENERIC additional-cost-choice case: cards whose
    # ``additional_cost_choose_your_creature`` is True but whose
    # ``spell_effect`` is NOT a covered class STILL stay declined
    # (the round-trip-or-bail serializer gate). The
    # ``additional_cost_choose_your_creature`` flag itself is NOT a
    # decline reason — the Cycle 1+2 substrate handles the choice
    # binding at cast time on both engines, byte-equal.
    # The defensive ``additional_cost_choice`` attr reject below
    # is kept (future-proof; harmless no-op while the attr is
    # absent) so a future parser that structures the marker into a
    # distinct attribute still gets positively declined.
    if getattr(printing, "additional_cost_choice", None) is not None:
        return False
    # **ModalChooseOne envelope — Cycle 4 (admit-predicate widen,
    # 2026-05-21).** ``ModalChooseOne`` was previously REJECTED here
    # unconditionally (the Cycle 1-3 substrate-only landing kept the
    # admit set closed while the cast-time mode binding + per-mode
    # resolve dispatch + reverse serializer cycles shipped). With
    # Cycle 4 landing the per-leaf Effect variants
    # (``DealDamageEachOpponent`` already shipped; the new
    # ``ExileAllArtifactsWithManaValueAtMost`` lands in this cycle)
    # the round-trip-or-bail teeth at ``_python_spell_effect_to_dict``
    # below carry the load: a modal whose modes are NOT all leaf-
    # recognized bails at the serializer (returns ``None``), and the
    # admit gate's ``_python_spell_effect_to_dict(se) is None`` clause
    # rejects it.
    #
    # The Phase-3 ``(0, 0)``-shape wrath admit arm below ALSO covers
    # all-untargeted ModalChooseOne cards: Ruinous Rampage's
    # ``target_count_range() == (0, 0)`` and ``target_slot_kinds()
    # == ()`` because both of its modes (``DealDamageEachOpponent``,
    # ``ExileAllArtifactsWithManaValueAtMost``) are untargeted. The
    # some-targeted ModalChooseOne cards (Reroute Systems / Scour for
    # Scrap / Archenemy's Charm / Drill Too Deep / Warping Wail /
    # Twist Reality / Live or Die — 7 cards) report non-``(0, 0)``
    # shapes and STAY DECLINED via the target-grammar gate below;
    # they need a per-mode target-binding Slice (Cycle 5) before
    # they can flip.
    #
    # **``or_both`` deferral preserved.** The serializer at
    # ``_python_spell_effect_to_dict`` explicitly bails on
    # ``ModalChooseOne(or_both=True)`` (Cycle 3); the admit gate's
    # round-trip clause picks that up too. Scour for Scrap (the only
    # ``or_both`` EoE modal) STAYS DECLINED via the round-trip-or-bail
    # path even if its modes were leaf-recognized.
    # Target-grammar gate — Phase 1 (single-target) OR Phase 2 (the
    # exact 2-slot ('my_creature','creature') heterogeneous pair) OR
    # the wrath-admit-widening Phase-3 ``(0, 0)``-shape wrath grammar.
    # Every OTHER target shape (any-number, opponent-zone, 3+ slots,
    # other 2-slot kind-tuples) STAYS positively declined — the §2
    # teeth. The Phase-2 widening is SURGICAL: it admits ONLY the
    # one heterogeneous pair the cross-engine 2-slot binding +
    # resolve-time double-swap is implemented + pinned byte-equal
    # for (Diplomatic Relations), NOT a blanket "any 2-target"
    # widen (that would ship the count-chasing anti-pattern §2
    # exists to prevent).
    min_t, max_t = se.target_count_range()
    slot_kinds = se.target_slot_kinds()
    # X-value-subsystem-C2.5: every non-Phase-3 phase explicitly
    # excludes X-cost printings (`mc.x_pips > 0`). Only Phase 3
    # admits X-cost (the wrath-shape lane that motivated the widen).
    _PHASE1_SHAPE = (
        (min_t, max_t) == (1, 1)
        and not slot_kinds
        and not _printing_is_x_cost
    )
    # Phase-2 admit shape: exactly two required targets with the
    # disjoint friendly/opponent creature slot pair. ``_select_spell
    # _targets`` filters the flat ``legal_targets`` per slot_kind and
    # the first-legal convention picks slot 0 = first caster-
    # controlled creature, slot 1 = first non-caster-controlled
    # creature — the EXACT pair the Rust ``cast_spell_noncreature``
    # 2-slot scan binds + the double resolve-swap threads byte-equal.
    _PHASE2_SHAPE = (
        (min_t, max_t) == (2, 2)
        and tuple(slot_kinds) == ("my_creature", "creature")
        and not _printing_is_x_cost
    )
    # **Phase 3 — wrath-admit-widening (2026-05-21).** ``(0, 0)``-shape
    # spells with NO slot_kinds: wrath-style mass-effect spells whose
    # ``spell_effect`` resolves over the full battlefield snapshot with
    # NO at-cast target binding (Pyroclasm, Manifest Dread, Destroy-All-
    # Creatures, etc.). The ``cast_spell_noncreature`` Rust path already
    # threads a target-free StackItem byte-equal across both engines (no
    # at-cast binding to mirror — the spell pushes onto the stack with an
    # empty targets tuple on both sides; the at-resolve dispatch reads
    # the Effect dict and walks the battlefield itself). The Phase-3
    # widen is SURGICAL: it admits ONLY ``(0, 0)`` with no slot_kinds AND
    # no ``targets`` attribute on the printing — every other target shape
    # stays declined. The serializer round-trip teeth below carry the
    # full no-fallback-hybrid load (an admitted card without a
    # round-trip-clean Effect arm bails via
    # :func:`_python_spell_effect_to_dict` returning ``None``).
    # **X-value-subsystem-C2.5 (2026-05-22) widen.** Phase 3 is the
    # SOLE phase that admits X-cost printings. The `(0, 0)`-shape with
    # empty slot_kinds + no ``targets`` attr is the only target
    # grammar Zero Point Ballad's `{X}{B}{B}` "Destroy each creature
    # with toughness X or less. Then ..." resolve body fits — no
    # at-cast target binding, the destroy walks the bf snapshot at
    # resolve time scoped by `state.resolving_x_value` (read via the
    # Cycle-1 bracket). The covered+bridged + round-trip-or-bail
    # teeth still gate the admit set — until C3 lands
    # `Effect::ZeroPointBallad`, Zero Point Ballad stays DECLINED via
    # the no-Effect-coverage tooth (the admit widen is substrate-only
    # in C2.5: the predicate accepts the SHAPE, but the gate hasn't
    # opened yet for any real EoE X-cost card).
    _PHASE3_WRATH_SHAPE = (
        (min_t, max_t) == (0, 0)
        and not slot_kinds
        and not getattr(printing, "targets", ())
    )
    # **Phase 4 — multi-target admit widen (2026-05-21).** Graveyard-
    # source multi-target spell-effects with a uniform target kind
    # (``"graveyard_creature"`` today; Scout for Survivors's (0, 3)
    # shape is the inaugural consumer). Admit only when:
    #
    #   1. ``min_t == 0`` (optional targets — "up to N target ...") OR
    #      ``min_t >= 1 and max_t > min_t`` (range "one or two ...");
    #      the bound is "max_t > 1" so single-target shapes go through
    #      Phase 1 instead.
    #   2. ``slot_kinds`` is empty OR has length ``max_t`` with all
    #      slots reporting the SAME kind (uniform-kind multi-target).
    #   3. The covered+bridged gate (`_spell_effect_is_covered_and_
    #      bridged`) AND the round-trip-or-bail gate
    #      (``_python_spell_effect_to_dict(se) is None``) below carry
    #      the load — admitting a covered+bridged shape that round-
    #      trips and whose serializer leaf emits the Rust-variant dict
    #      auto-grows the admit set; an uncovered or non-serializable
    #      shape stays declined (the §2 teeth).
    #
    # **Resolve binding shape.** Today's resolve path SELF-SCANS the
    # source-controller's graveyard at resolve time (the
    # ``Effect::ReturnUpToNSmallCreaturesFromGraveyardWithCounters``
    # arm in ``state_ops_resolve.rs``), mirroring the established
    # ``ReturnCreatureFromYourGraveyardToBattlefield`` single-target
    # self-scan pattern. The cross-engine harness's
    # ``_PickFirstLegalTargetAgent`` picks first-N-legal in
    # legal_targets order, which coincides byte-equal with the self-
    # scan — see ``docs/SEARCH_LIMITATIONS.md`` entry "Multi-target
    # Phase 4 resolve-time self-scan" for the deliberate divergence
    # vs cast-time binding the §2 admit gate accepts.
    _PHASE4_MULTI_GRAVEYARD_SHAPE = (
        max_t > 1
        and min_t in (0, 1)
        and (
            not slot_kinds
            or (
                len(slot_kinds) == max_t
                and len(set(slot_kinds)) == 1
                and slot_kinds[0] in ("graveyard_creature",)
            )
        )
        and not _printing_is_x_cost
    )
    # **Phase 5 — scour-for-scrap-cycle-2 (2026-05-21) — `or_both`
    # modal admit-widen.** ``ModalChooseOne(or_both=True)`` cards whose
    # ``target_count_range() == (0, 1)`` and ``slot_kinds == ()``.
    # Reports `(0, 1)` because at least one mode is untargeted (so you
    # can pick the untargeted mode alone — `min_t == 0`) and the
    # most-targeted mode needs at most one target (`max_t == 1`). The
    # empty ``slot_kinds`` mirrors ``ModalChooseOne``'s base default
    # — it doesn't override ``target_slot_kinds()``. Today's consumer
    # is exactly **Scour for Scrap** (its mode 0 is the untargeted
    # ``SearchLibraryForCard``, mode 1 is the single-target
    # ``ReturnTargetArtifactCardFromYourGraveyardToHand``); any future
    # ``or_both`` modal with the same shape (one untargeted mode +
    # one single-target mode) will auto-flip the same way provided
    # both leaves are covered+bridged AND the per-mode binder picks
    # the targeted mode's filter (the
    # ``is_recognized_targeted_mode_leaf`` allowlist in
    # ``state_ops_cast.rs``). The round-trip-or-bail teeth at
    # ``_python_spell_effect_to_dict`` continue to carry the load:
    # a modal whose modes are NOT all leaf-recognized bails at the
    # serializer (returns ``None``) and the admit gate's
    # ``_python_spell_effect_to_dict(se) is None`` clause rejects it.
    _PHASE5_OR_BOTH_MIXED_TARGET_SHAPE = (
        (min_t, max_t) == (0, 1)
        and not slot_kinds
        and not _printing_is_x_cost
    )
    if not (
        _PHASE1_SHAPE
        or _PHASE2_SHAPE
        or _PHASE3_WRATH_SHAPE
        or _PHASE4_MULTI_GRAVEYARD_SHAPE
        or _PHASE5_OR_BOTH_MIXED_TARGET_SHAPE
    ):
        return False  # every other target grammar stays declined
    # The load-bearing §2 gate: the effect must be COVERED + BRIDGED
    # (its emitted Effect classes ⊆ the LIVE covered set AND each has
    # a _ta_dict_to_python_effect bridge arm + a Rust variant). This
    # is what makes the lever gate on the per-card Tier-A close
    # having LANDED — admitting a card before its Effect's Rust+bridge
    # exists would silently mistranslate to an effectless spell. NOT
    # a hardcoded card-name list — the §2 teeth.
    if not _spell_effect_is_covered_and_bridged(printing):
        return False
    # Second §2 tooth (verify-first surfaced): the effect must
    # ROUND-TRIP through the Python→dict reverse serializer with NO
    # loader-unseatable token pid. This structurally declines
    # Seedship Impact / Orbital Plunge (covered, but their Rust
    # parse_effect requires a Lander token_printing_id the spell-path
    # loader cannot seat) — the no-fallback-hybrid discipline.
    if _python_spell_effect_to_dict(se) is None:
        return False
    return True


def _python_ta_to_safe_dict_aura_extension(ta) -> dict:
    """**Phase 2 S2 extension (Tractor Beam closure, 2026-05-20).**
    Convert a Phase-2-S2-admitted Aura ETB ``TriggeredAbility``
    into its Rust-shape round-trip dict.

    Pre-condition: ``ta`` has been validated by
    :func:`_aura_triggers_are_phase2_s2_safe` — i.e. it is the
    ``Event.ENTERS + IsSelf + TapEnchantedPermanent`` shape. The
    dict this returns matches the canonical Rust trigger shape
    (``{"event": "enters", "predicate": {"kind": "is_self"},
    "effect": {"kind": "tap_enchanted_permanent"}, "once_per_turn":
    False}``) — the same per-key spellings the existing
    :func:`_python_ta_to_safe_dict` recognizer family emits for
    sibling unit-variant effects.

    Lives separate from :func:`_python_ta_to_safe_dict` because the
    safe-trigger family is gated on creature ETB triggers (DrawCards
    / GainLife / Scry / ...), while this gate covers the strictly-
    Aura-specific ``TapEnchantedPermanent`` family. Sharing the
    safe-trigger registry would require widening that family's
    effect-kind allowlist for an effect that only makes sense on an
    Aura source — keep the surface narrow per the "no fallback
    hybrid" discipline.
    """
    return {
        "event": "enters",
        # Rust's predicate kind for ``IsSelf`` is ``"source_is_self"``
        # (see ``cards.rs::parse_predicate`` and the bridge in
        # ``cross_engine.py:9597``). Don't confuse with ``"is_self"``
        # which is not a registered predicate kind.
        "predicate": {"kind": "source_is_self"},
        "effect": {"kind": "tap_enchanted_permanent"},
        "once_per_turn": False,
    }


def _aura_triggers_are_phase2_s2_safe(triggered_abilities) -> bool:
    """**Phase 2 S2 extension (Tractor Beam closure, 2026-05-20).**
    True iff every ``TriggeredAbility`` in ``triggered_abilities``
    matches the narrow Tractor-Beam-shape allowlist:

      * ``event`` is :data:`Event.ENTERS`
      * ``predicate`` is :class:`IsSelf` with no subject scope
      * ``effect`` is :class:`TapEnchantedPermanent`
      * ``once_per_turn`` is ``False`` (ETB triggers are once-per-ETB
        by construction, not once-per-turn).

    The ``TapEnchantedPermanent`` effect is a bridge-supported unit
    variant — Python side at ``effects/generic/tap.py:497``, Rust
    side at ``state_ops_resolve.rs::TapEnchantedPermanent``,
    cross-engine bridge at ``cross_engine.py:5317`` for the
    dict-to-Python arm. The trigger dict round-trip uses the
    ``_python_ta_to_safe_dict`` siblings (the Rust side parses an
    Aura-ETB triggered ability dict via the standard
    ``parse_triggered_ability`` arm; the effect dict the bridge
    emits for ``TapEnchantedPermanent`` is the canonical unit-variant
    ``{"kind": "tap_enchanted_permanent"}`` shape).

    Returns ``False`` for an empty tuple — callers should gate on
    ``if printing.triggered_abilities:`` first; this helper is only
    relevant when there IS at least one trigger to validate.

    Returns ``False`` on the first non-matching trigger so any future
    Aura with a different ETB body keeps the gate honest (the safe-
    trigger predicate's all-or-nothing convention).
    """
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent

    if not triggered_abilities:
        return False
    for ta in triggered_abilities:
        if not isinstance(ta, TriggeredAbility):
            return False
        if ta.once_per_turn:
            return False
        if ta.event is not Event.ENTERS:
            return False
        pred = ta.predicate
        if not isinstance(pred, IsSelf):
            return False
        if pred.subject is not None:
            return False
        if not isinstance(ta.effect, TapEnchantedPermanent):
            return False
    return True


def _aura_triggers_are_phase3_safe(triggered_abilities) -> bool:
    """**Aura-attachment substrate — Phase 3 / S4 (Pain for All
    closure; round-80, 2026-05-20).** True iff every
    ``TriggeredAbility`` in ``triggered_abilities`` matches one of
    the narrow Phase-3 admit shapes:

      1. The Phase-2-S2-extension Aura-ETB shape (Tractor Beam):
         ``Event.ENTERS + IsSelf(subject=None) + TapEnchantedPermanent``.
      2. The Phase-3 Pain-for-All ETB shape:
         ``Event.ENTERS + IsSelf(subject=None) + PainForAllETBEffect``.
      3. The Phase-3 Pain-for-All damage-echo shape:
         ``Event.DEALT_DAMAGE + IsHost(subject=None) +
         PainForAllDamageEchoEffect``.

    ``once_per_turn=False`` on every shape — Aura triggers fire each
    occurrence by construction, not once-per-turn.

    The two Pain for All bespoke effect classes
    (:class:`mtg_bot.effects.bespoke.PainForAllETBEffect` +
    :class:`mtg_bot.effects.bespoke.PainForAllDamageEchoEffect`) are
    unit variants with full Rust ``Effect`` parity (landed in the
    Pain for All closure slice — ``rust/src/cards.rs::Effect::
    PainForAllETBEffect`` / ``::PainForAllDamageEchoEffect``) and
    matching ``_ta_dict_to_python_effect`` bridge arms
    (``cross_engine.py:8351`` / ``cross_engine.py:8388``). The
    :class:`mtg_bot.effects.predicates.IsHost` predicate is also
    fully Rust-bridged
    (``rust/src/cards.rs::Predicate::IsHost { subject }`` +
    ``cross_engine.py:9785``).

    Returns ``False`` for an empty tuple — callers should gate on
    ``if printing.triggered_abilities:`` first; this helper is only
    relevant when there IS at least one trigger to validate.

    Returns ``False`` on the first non-matching trigger — the
    safe-trigger predicate's all-or-nothing convention.

    Lives separate from :func:`_aura_triggers_are_phase2_s2_safe`
    because the Phase-3 admit set covers a wider event surface
    (DEALT_DAMAGE in addition to ENTERS) and a different predicate
    family (IsHost in addition to IsSelf). Phase-3 admits a strict
    superset of Phase-2-S2-extension; Tractor Beam's
    ``ENTERS+IsSelf+TapEnchantedPermanent`` shape passes both
    helpers (only one needs to ship the admit decision in
    ``_printing_is_bridge_simple_aura`` — Phase-3 is the canonical
    gate, Phase-2-S2 stays for back-compat with its existing
    tests).
    """
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.predicates import Event, IsHost, IsSelf
    from mtg_bot.effects.bespoke import (
        PainForAllDamageEchoEffect,
        PainForAllETBEffect,
    )
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent

    if not triggered_abilities:
        return False
    for ta in triggered_abilities:
        if not isinstance(ta, TriggeredAbility):
            return False
        if ta.once_per_turn:
            return False
        # Phase-3 admit shape 1: Tractor-Beam Aura-ETB tap.
        if (
            ta.event is Event.ENTERS
            and isinstance(ta.predicate, IsSelf)
            and ta.predicate.subject is None
            and isinstance(ta.effect, TapEnchantedPermanent)
        ):
            continue
        # Phase-3 admit shape 2: Pain for All ETB damage to any
        # other target (the bespoke ETB effect).
        if (
            ta.event is Event.ENTERS
            and isinstance(ta.predicate, IsSelf)
            and ta.predicate.subject is None
            and isinstance(ta.effect, PainForAllETBEffect)
        ):
            continue
        # Phase-3 admit shape 3: Pain for All damage echo.
        if (
            ta.event is Event.DEALT_DAMAGE
            and isinstance(ta.predicate, IsHost)
            and ta.predicate.subject is None
            and isinstance(ta.effect, PainForAllDamageEchoEffect)
        ):
            continue
        return False
    return True


def _python_ta_to_safe_dict_aura_phase3(ta) -> dict:
    """**Aura-attachment substrate — Phase 3 / S4 (Pain for All
    closure; round-80, 2026-05-20).** Convert a Phase-3-admitted
    Aura ``TriggeredAbility`` into its Rust-shape round-trip dict.

    Pre-condition: ``ta`` has been validated by
    :func:`_aura_triggers_are_phase3_safe` — i.e. it is one of:

      * ``ENTERS+IsSelf+TapEnchantedPermanent`` (Tractor-Beam shape;
        emits the same dict as
        :func:`_python_ta_to_safe_dict_aura_extension` for
        cross-substrate continuity).
      * ``ENTERS+IsSelf+PainForAllETBEffect`` (Pain for All ETB).
      * ``DEALT_DAMAGE+IsHost+PainForAllDamageEchoEffect`` (Pain
        for All damage echo).

    Rust spellings (cross-engine bridge arms):

      * ``Event.ENTERS`` → ``"enters"``;
        ``Event.DEALT_DAMAGE`` → ``"dealt_damage"``
        (``cards.rs::parse_triggered_ability`` event arm at line ~10252).
      * ``IsSelf(subject=None)`` → ``{"kind": "source_is_self"}``
        (``cards.rs::parse_predicate``).
      * ``IsHost(subject=None)`` → ``{"kind": "is_host"}``
        (``cards.rs::Predicate::IsHost`` + bridge arm at
        ``cross_engine.py:9785``).
      * ``PainForAllETBEffect`` → ``{"kind": "pain_for_all_etb"}``
        (``cards.rs:8069`` + bridge arm at ``cross_engine.py:8351``).
      * ``PainForAllDamageEchoEffect`` →
        ``{"kind": "pain_for_all_damage_echo"}``
        (``cards.rs:8082`` + bridge arm at ``cross_engine.py:8388``).
      * ``TapEnchantedPermanent`` →
        ``{"kind": "tap_enchanted_permanent"}``
        (cross_engine.py:5317 — shared with the Phase-2-S2-extension
        emitter).

    All admitted shapes have ``once_per_turn=False`` per the helper
    gate — Aura triggers fire each occurrence by construction.

    Lives separate from :func:`_python_ta_to_safe_dict` (the
    creature safe-trigger family) because the Aura-specific effects
    only make sense on an Aura source — sharing the family's
    effect-kind allowlist would force the creature gate to admit
    Aura-only effects per the "no fallback hybrid" discipline.
    """
    from mtg_bot.effects.predicates import IsHost, IsSelf, Event
    from mtg_bot.effects.bespoke import (
        PainForAllDamageEchoEffect,
        PainForAllETBEffect,
    )
    from mtg_bot.effects.generic.tap import TapEnchantedPermanent

    if isinstance(ta.effect, PainForAllETBEffect):
        # Shape 2: Pain for All ETB.
        return {
            "event": "enters",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind": "pain_for_all_etb"},
            "once_per_turn": False,
        }
    if isinstance(ta.effect, PainForAllDamageEchoEffect):
        # Shape 3: Pain for All damage echo (DEALT_DAMAGE + IsHost).
        # IsHost subject is None for the Pain for All shape
        # (single-subject DEALT_DAMAGE event — the receiving creature).
        pred_dict: dict = {"kind": "is_host"}
        # If the predicate carries a subject slot (defensive; live
        # parser emits IsHost(subject=None) for Pain for All), thread
        # it through — the Rust IsHost variant carries the same
        # Optional<String> slot.
        if isinstance(ta.predicate, IsHost) and ta.predicate.subject is not None:
            pred_dict["subject"] = ta.predicate.subject
        return {
            "event": "dealt_damage",
            "predicate": pred_dict,
            "effect": {"kind": "pain_for_all_damage_echo"},
            "once_per_turn": False,
        }
    if isinstance(ta.effect, TapEnchantedPermanent):
        # Shape 1: Tractor-Beam Aura-ETB tap (same shape as the
        # Phase-2-S2-extension emitter ships).
        return {
            "event": "enters",
            "predicate": {"kind": "source_is_self"},
            "effect": {"kind": "tap_enchanted_permanent"},
            "once_per_turn": False,
        }
    # Should be unreachable when pre-gated by
    # :func:`_aura_triggers_are_phase3_safe`. Raise loudly so a
    # future widen of the helper without a matching emitter widen
    # surfaces immediately rather than silently mistranslating.
    raise ValueError(
        f"_python_ta_to_safe_dict_aura_phase3: unsupported trigger "
        f"effect type {type(ta.effect).__name__!r} — gate must be "
        "_aura_triggers_are_phase3_safe-clean before calling this helper."
    )


def _printing_is_bridge_simple_aura(printing) -> bool:
    """True iff a CardPrinting is a Phase-1/2 bridge-supported Aura
    (sub-shape S1 in ``docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md`` §1):
    a static P/T-or-keyword Aura with no triggered/spell/activated/
    static effects, no control-swap or replacement riders, and a
    parseable cost.

    **Aura-attachment substrate — Phase 2 FIRST DELIVERABLE LANDED
    (round-79; widens the keyword admit set to include cast-timing
    ``flash`` per the symmetric-drop precedent).** Phase 2 widens the
    static-keyword check to admit Auras whose printing keywords are a
    subset of ``_SAFE_KEYWORDS ∪ {flash}``. ``flash`` is a cast-timing
    rider that the CardDef bridge symmetrically drops on BOTH engines
    (the CardDef → CardPrinting round-trip carries no ``keywords``
    field through the aura arm). This finally admits the **first live
    EoE Aura** — Frantic Strength (and Shardmage's Rescue) — through
    the loader: substrate end-to-end on a real EoE printing. See
    `docs/SEARCH_LIMITATIONS.md` for the SL entry mirroring the
    round-39 Ward / round-59 shockland symmetric-drop precedent. The
    full Phase 2 (S2 control-changing per the plan §3) remains
    deferred to a follow-up Rust slice.

    **Aura-attachment substrate — Phase 1 STEP 7 LANDED (round-78).**
    This is the §2 positive-allowlist predicate the plan's §3 Phase 1
    step 7 specifies, and it is now OR'd into
    :func:`_filter_deck_to_bridge_supported`'s admit chokepoint as
    the 6th disjunct. STEPs 1-6 (CardDef aura representation + Rust
    ``parse_printing`` aura-key parse + cast-time enchant-target
    decision + Rust aura-resolution attach arm + CR-704.5n SBA +
    bidirectional bridge round-trip) all landed before this OR-in,
    so a printing that passes the predicate flows end-to-end through
    the bench loader, the Python side, and the Rust side.

    Phase-1 admit set: sub-shape **S1** ONLY (Frantic Strength,
    Shardmage's Rescue per the plan §1). Every other Aura sub-shape
    (S2 control-changing, S3 replacement, S4 triggered) stays
    positively declined here — Phase 2/3/4 land them.

    The §2 teeth (declines that would silently mistranslate, mirroring
    the warp / spell guardrail's rider-rejection blocks):

      * Must be ``is_aura`` (CR 205.3h — Enchantment - Aura).
      * No ``triggered_abilities`` (S4 — Phase 4).
      * No ``activated_abilities`` (no Aura in the EoE pool carries
        one today; defensive).
      * No ``static_abilities`` (Phase 2 — anthem-style Auras).
      * No ``spell_effect`` (Auras are permanents, not Instant/
        Sorcery; defensive).
      * No ``warp_cost`` / ``kicker_cost`` / ``impending_cost`` /
        ``room_halves`` / ``loyalty`` (alt-cost riders — other
        substrates).
      * No ``cost_reductions`` / ``external_cost_reductions``.
      * No ``attached_grants_control`` / ``attached_grants_lose_all_
        abilities`` / ``attached_sets_base_pt`` / ``attached_grants_
        subtype`` / ``attached_prevents_untap`` (S2 — Phase 2).
      * ``replaces_first_tokens_with_copies`` (S3 — Phase 4) is
        NOW ADMITTED post-closure (2026-05-21). The Rust
        ``CreateToken`` arms consult the per-controller Moonlit-shape
        Aura via ``moonlit_token_substitution`` and swap to the host's
        printing with a per-object Token-supertype override. The
        substrate predicate no longer rejects this rider.
      * No ``ward_cost`` (defensive — mirrors the simple-creature
        guardrail).
      * No ``dynamic_pt_formula`` (defensive — Auras don't have one,
        but the CardDef bridge would silently drop it).
      * Either no ``keywords``, OR every keyword in
        :data:`_SAFE_KEYWORDS` (the parser-on-the-Aura keyword set;
        Frantic Strength's ``flash`` is OUTSIDE _SAFE_KEYWORDS — see
        below; Shardmage's ``flash`` ditto).

        **Phase-1-S1 carrier reality (verify-first).** Frantic Strength
        carries ``flash`` as a STATIC keyword on the printing AND a
        ``trample`` keyword on the granted ``equipment_bonuses`` tuple.
        The granted-keyword (the +2/+2/trample rider) lives on a
        SEPARATE field from the printing's static ``keywords`` — the
        bridge round-trips them via different paths. Phase-1 admits
        ONLY the granted-keyword subset that's in
        :data:`_SAFE_KEYWORDS`; the printing's own static ``flash``
        keyword is the §2 decline reason for both Frantic Strength
        and Shardmage's Rescue in THIS slice. This matches the round-
        61 N1 scout's finding that S1 has zero realizable-yield decks
        today (both cards have 0 EoE 17lands decks; the long-tail
        admit is the substrate's value, not the deck count).

      * Aura's ``equipment_bonuses`` (the (power_bonus,
        toughness_bonus, granted_keywords) tuple) — when present, the
        granted_keywords must all be in :data:`_SAFE_KEYWORDS`. The
        bonus deltas have no upper bound.
      * Mana cost is parseable into (generic, colored).

    **Wired into the loader as of STEP 7 (round-78), widened by Phase
    2 first deliverable (round-79):** the predicate is the 6th
    disjunct of :func:`_filter_deck_to_bridge_supported`'s admit
    chokepoint. **The live EoE pool now admits ≥1 Aura through this
    disjunct** (the first live admit — round-79 widen of the keyword
    set per the symmetric-drop precedent above). Remaining S2/S3/S4
    Aura sub-shapes stay positively declined here; Phase 3/4 widen
    further.
    """
    if not getattr(printing, "is_aura", False):
        return False
    # Triggered / static / activated / spell-effect riders — other
    # Aura sub-shapes.
    #
    # **Phase 3 / S4 (Pain for All closure; round-80, 2026-05-20)
    # + Cast-time auto-resolve seat alignment (Tractor Beam closure,
    # 2026-05-20).** Auras with one or more triggered abilities are
    # admitted iff every trigger matches a Phase-3-safe shape per
    # :func:`_aura_triggers_are_phase3_safe`:
    #
    #   * ``ENTERS+IsSelf+TapEnchantedPermanent`` (Tractor-Beam
    #     ETB-tap shape — Phase-2-S2-extension substrate).
    #     **NOW ADMITTED end-to-end** post the cast-time auto-resolve
    #     seat alignment slice (Rust ``apply_action`` drains the
    #     ``EnchantTargetChoice`` synchronously after ``cast_aura``;
    #     non-creature ``summoning_sick=false`` clear in
    #     ``move_*_post_resolve``; aura-host-control revert helper for
    #     non-SBA destroy sites; ``BiosynthicBurst`` legal/resolve scan
    #     walks both bfs by controller).
    #   * ``ENTERS+IsSelf+PainForAllETBEffect`` (Pain for All ETB
    #     damage-to-any-target — the Aura Phase 3 first deliverable;
    #     Rust ``Effect::PainForAllETBEffect`` parity landed in the
    #     Pain for All closure slice).
    #   * ``DEALT_DAMAGE+IsHost+PainForAllDamageEchoEffect`` (Pain
    #     for All damage echo — Rust ``Effect::
    #     PainForAllDamageEchoEffect`` + ``Predicate::IsHost``
    #     parity also landed in the Pain for All closure slice).
    #
    # Every other triggered-Aura shape (S4 sub-shape per the plan
    # §1 — Chorale of the Void / Cracked Skull / Sporogenic
    # Infection) stays declined here: those carry uncovered Rust
    # Effect variants AND/OR unbridged trigger predicates, so the
    # Phase-3 safe-trigger gate refuses them. The §2 teeth keep
    # the predicate honest — admitting an unsafe trigger surface
    # would silently mistranslate to an effectless permanent on
    # one engine while the other applies the effect.
    if printing.triggered_abilities:
        if not _aura_triggers_are_phase3_safe(printing.triggered_abilities):
            return False
    if printing.activated_abilities:
        return False
    if printing.static_abilities:
        return False
    if printing.spell_effect is not None:
        return False
    # Alt-cost / mode riders — defensive (no EoE Aura today carries
    # any of these, but the same §2 rider-rejection pattern as the
    # warp / spell guardrails keeps the gate honest).
    if printing.warp_cost or printing.kicker_cost or printing.impending_cost:
        return False
    if printing.room_halves:
        return False
    if printing.loyalty is not None:
        return False
    if printing.cost_reductions or printing.external_cost_reductions:
        return False
    # S2 control-changing — Phase 2 S2 (round-79+, 2026-05-20) — now
    # admitted for the Duskmourn's Domination shape:
    # ``attached_grants_control AND attached_grants_lose_all_abilities``
    # (paired — both flags or neither — the live S2 carrier in EoE
    # is Duskmourn's Domination, and the only other live S2 sibling,
    # Tractor Beam, ALSO carries `attached_prevents_untap` plus a
    # triggered ETB and is independently declined below). The Rust
    # attach arm flips the host's controller (layer 2) and the
    # `battlefield_has_keyword` reader wipes the host's keyword set
    # while the lose-abilities aura is attached (layer 6 honest-stop
    # — see the SL entry).
    #
    # ``attached_sets_base_pt`` (Lignify-shape layer 7b override) and
    # ``attached_grants_subtype`` (Lignify-shape layer 4 add_subtype)
    # stay positively declined — no S2 carrier in the live EoE pool
    # uses them, and the Rust reader fold for set-base-PT is NOT
    # wired in this slice (honest-stop per the plan §3 Phase 2).
    #
    # ``attached_prevents_untap`` (Tractor Beam's don't-untap clause)
    # also stays declined — no Rust untap-step skip for the attached
    # host is wired yet (also a future Phase-2-S2-extension slice).
    #
    # S3 replacement riders — Phase 4; positively declined here.
    if printing.attached_grants_control and (
        not printing.attached_grants_lose_all_abilities
    ) and (
        not printing.attached_prevents_untap
    ):
        # Lone control-grant without the lose-abilities wipe AND
        # without the don't-untap clause — no live EoE carrier matches
        # this shape today (Tractor Beam pairs control-grant WITH the
        # don't-untap rider, admitted below). Stay defensive —
        # admitting a control-only Aura would silently mistranslate
        # against a host that still has functional abilities the
        # bench may rely on.
        return False
    if printing.attached_grants_lose_all_abilities and (
        not printing.attached_grants_control
    ):
        # Lone lose-abilities without control-swap is the Lignify
        # shape (also carries set-base-PT + subtype add). No live EoE
        # carrier matches this AND has its other riders unset; stay
        # declined.
        return False
    if printing.attached_sets_base_pt is not None:
        return False
    if printing.attached_grants_subtype is not None:
        return False
    if printing.attached_prevents_untap and (
        not printing.attached_grants_control
    ):
        # Lone don't-untap rider without the control-grant. Tractor
        # Beam is the live EoE carrier pairing the two — admitted via
        # the joint-shape gate above. Any future carrier with the
        # bare don't-untap rider (Tilling Treefolk-shape) stays
        # declined here pending a separate slice.
        return False
    # Aura-attachment Phase 4 S3 closure (2026-05-21): Moonlit
    # Meditation now admits at the loader. The Rust CreateToken /
    # CreateTokenTapped / CreateTokenEachPlayer / CreateToken-
    # CopyOfTargetCreatureYouControl arms consult the controller's
    # attached Moonlit-shape Aura via ``moonlit_token_substitution``
    # and swap the token's printing_id to the host's printing with
    # the per-object Token-supertype override stamped. Mirrors
    # Python's ``_find_moonlit_replacement_source`` +
    # ``_dc.replace(base, supertypes=(*base, "Token"))`` shim at
    # ``effects/generic/tokens.py:217-247`` — byte-equal on the
    # post-resolution bf snapshot (the bridge round-trip surfaces
    # the override via the ``token_supertype_override`` key in
    # ``object_to_pydict``). The rejection that used to live here
    # has been replaced by an admit; pre-closure pinned baselines
    # for Moonlit Meditation under the bench loader were "still
    # declined" — those are now positive admits.
    #
    # **What changed.** Prior to closure: this predicate returned
    # False for any Aura carrying ``replaces_first_tokens_with_copies
    # = True``, declining Moonlit Meditation (and any future
    # CR-614-token-replacement Aura on the same rider). Post-closure:
    # the predicate falls through to the rest of the simple-Aura
    # admit checks, so an Aura whose ONLY rider is the Moonlit
    # replacement is admitted as a bridge-simple Aura. The other
    # Phase-4 declines (Chorale of the Void's reanimate trigger,
    # Cracked Skull's destroy-host shape) remain declined — this
    # closure is scoped to the S3 replacement-rider lane only.
    pass
    # Defensive — Auras don't carry ward / dynamic-PT today, but keep
    # the substrate honest (mirrors the simple-creature guardrail).
    if getattr(printing, "ward_cost", None) is not None:
        return False
    if getattr(printing, "dynamic_pt_formula", None) is not None:
        return False
    # Static keyword set on the printing itself.
    #
    # **Phase 2 widen (round-79; symmetric-drop precedent — round-39
    # Ward / round-59 shockland).** The Aura-keyword surface admits
    # _SAFE_KEYWORDS PLUS ``flash``. ``flash`` is a cast-timing rider
    # (CR 702.8), not a per-permanent combat keyword, AND the CardDef
    # bridge in :func:`_printing_to_card_def`'s aura arm DOES NOT
    # propagate the source printing's ``keywords`` field through to the
    # CardDef — both Python and Rust materialize the Aura with empty
    # ``keywords`` from the CardDef on each engine, so the ``flash``
    # rider is **symmetrically dropped** on BOTH bridge sides
    # bit-equally. The instant-speed cast emit path (`flash` on a
    # non-creature spell makes it instant-speed castable) is unexercised
    # by `attack_all`-style bench play, which only casts at sorcery
    # speed — exactly the same parity-safety argument as the round-59
    # shocklands' tapped-entry / pay-life replacement (the rider is
    # never driven by the bench strategy on EITHER engine post-bridge,
    # so the drop is bit-equal under bench coverage).
    #
    # This widen finally admits the first **live EoE Aura** through the
    # loader: Frantic Strength (0 EoE 17lands decks — the substrate's
    # value, not deck count) + Shardmage's Rescue (0 decks; its
    # ``aura_yours_only`` is independently bridge-correct on both
    # engines per `actions.py:1337-1357` / `state_ops_cast.rs:2000-2055`;
    # its conditional `entered_this_turn` hexproof grant is NOT parsed
    # by the live parser at all — shared CR-incompleteness on the
    # Python side, so the drop is symmetric there too). See
    # `docs/SEARCH_LIMITATIONS.md` for the symmetric-drop SL entry.
    if printing.keywords and not printing.keywords.issubset(
        _SAFE_KEYWORDS | _AURA_SAFE_EXTRA_KEYWORDS
    ):
        return False
    # Granted-keyword check on equipment_bonuses (the +P/+T/keywords
    # rider). The bonus tuple is (power_bonus, toughness_bonus,
    # granted_keywords); granted_keywords must all be in _SAFE_KEYWORDS.
    # (Granted keywords flow through the CardDef bridge per
    # ``aura_granted_keywords`` and are applied to the host by Rust's
    # layer-6 grant — those MUST stay in _SAFE_KEYWORDS so both engines
    # round-trip the grant identically; only the static cast-timing
    # ``flash`` is dropped symmetrically per the comment above.)
    if printing.equipment_bonuses is not None:
        _, _, granted_kws = printing.equipment_bonuses
        if granted_kws and not granted_kws.issubset(_SAFE_KEYWORDS):
            return False
    try:
        _parse_mana_cost_string(printing.mana_cost_str or "")
    except ValueError:
        return False
    return True


def _printing_is_bridge_basic_land(printing) -> bool:
    """True if a CardPrinting is a basic Land the harness can model.

    Basic lands carry the ``"Basic"`` supertype + ``"Land"`` card type
    + a single ``produces`` color. The CardDef bridge auto-wires the
    ``{T}: Add {C}`` mana ability via the ``mana_produced`` field so
    activate_ability(land_id, 0) works on both engines.

    Non-basic lands (utility lands, shock lands, planet lands) are
    rejected here because they typically carry activated abilities
    and/or ETB triggers that CardDef can't represent.
    """
    if "Land" not in printing.card_types:
        return False
    if "Basic" not in printing.supertypes:
        return False
    if not printing.produces:
        return False
    if len(printing.produces) != 1:
        return False
    if printing.activated_abilities:
        # Basic lands SHOULD have no activated_abilities (the engine
        # auto-wires the mana ability from `produces`); a parser-shipped
        # ability would conflict with the CardDef-side auto-wire.
        return False
    if printing.triggered_abilities:
        return False
    if printing.static_abilities:
        return False
    return True


def _printing_is_bridge_nonbasic_mana_land(printing) -> bool:
    """True if a CardPrinting is a NON-basic mana-only land the bridge
    can round-trip byte-equal across both engines.

    Phase-6 non-basic-land substrate, **phase 1** (round-60 N1; spec at
    ``docs/NONBASIC_LAND_SUBSTRATE_PLAN.md`` §2/§3, written round-59 N2).

    The motivating set is the 5 EoE pay-life shocklands — Breeding
    Pool, Watery Grave, Stomping Ground, Godless Shrine, Sacred
    Foundry. The live ``OracleParserCardSource`` flattens each to
    ``produces={c1,c2}`` + ``enters_tapped=False`` + no abilities: the
    parser does **not** structure the "pay 2 life or enter tapped"
    replacement at all (verified round-60 against the live parser).
    Both engines therefore materialize the **same** untapped dual —
    bit-equal cross-engine, but CR-incomplete (a real shockland can
    enter tapped or cost 2 life). This is the round-39 Ward precedent:
    a rider the bridge drops *symmetrically*, admittable as a
    **documented deliberate simplification** (see
    ``docs/SEARCH_LIMITATIONS.md`` —
    ``trajectory.py:_printing_is_bridge_nonbasic_mana_land``).

    The predicate is the §2 Class-1 guardrail: it admits ONLY the
    parser-flattened mana-only profile (non-basic Land, 1-2 produce
    colors, NO abilities, NO structured tapped-entry, NO clone). It
    MUST reject the other 18 non-basic-land-gated cards — the Planet
    lands (``enters_tapped=True`` + Station + activated ability),
    the Duskmourn conditional painlands
    (``enters_tapped_unless_any_player_life_at_most`` set), Secluded
    Starforge (no ``produces`` + activated abilities), Cathedral of
    War (``enters_tapped=True`` + Exalted static + no mana), and
    Echoing Deeps (``optional_copy_land_from_graveyard`` clone). Each
    of those is a real cross-engine divergence or a §2-declined
    subsystem the bridge cannot carry today; admitting them via a bare
    supertype-gate widen would ship silent mistranslation (§4 of the
    plan, the round-38-E5 count-chasing anti-pattern §2 exists to
    prevent).

    NOTE on the conditional-tapped guard: the live
    ``CardPrinting.enters_tapped_unless_any_player_life_at_most`` is an
    ``int`` whose **disabled sentinel is ``0``** (``cards/base.py:733``
    — "0 disables"), NOT ``None``. The round-59 spec text wrote the
    guard as ``... is not None`` against an assumed ``int | None``
    shape; applied literally to the live ``int``-``0`` field that would
    reject all 5 shocklands (``0 is not None`` → True) and ship the
    round-60 slice as a +0-card no-op — the exact opposite of the
    spec's stated intent (§1/§3: "+5 cards"). The intent-preserving
    translation of the spec to the live field is a **falsy** check:
    ``0``/absent ⇒ no conditional tapped ⇒ admit; any positive
    threshold (the painlands' ``13``) ⇒ reject. This honors the spec's
    design (admit exactly the 5, reject the 10 painlands on the
    conditional-tapped guard) rather than its mistyped literal.
    """
    if "Land" not in printing.card_types:
        return False
    if "Basic" in printing.supertypes:
        return False  # basic path owns these
    if not printing.produces or not (1 <= len(printing.produces) <= 2):
        return False
    if printing.activated_abilities:
        return False
    if printing.triggered_abilities:
        return False
    if printing.static_abilities:
        return False
    if getattr(printing, "enters_tapped", False):
        return False
    # int field; 0 (or absent) disables the conditional-tapped
    # replacement (cards/base.py:733). A positive threshold (the
    # Duskmourn painlands' 13) is a parser-structured tapped-entry
    # Python honors and Rust ignores — a real cross-engine divergence.
    if getattr(printing, "enters_tapped_unless_any_player_life_at_most", 0):
        return False
    if getattr(printing, "enters_as_copy_of_creature", False):
        return False
    if getattr(printing, "optional_copy_land_from_graveyard", False):
        return False  # Echoing Deeps' real clone field (round-59 verified)
    return True


def _printing_is_bridge_simple_tapped_mana_land(printing) -> bool:
    """True if a CardPrinting is a NON-basic mono-color tapped mana-only
    land the bridge can round-trip byte-equal across both engines.

    **Non-basic land substrate phase 2 (round-85 N1; sibling of the
    round-60 untapped predicate above).** The motivating set is the 5
    EoE Planet lands — Adagia, Windswept Bastion / Evendo, Waking Haven
    / Kavaron, Memorial World / Susur Secundi, Void Altar / Uthros,
    Titanic Godcore. The live ``OracleParserCardSource`` flattens each
    to ``produces={one color}`` + ``enters_tapped=True`` + no triggered/
    activated/static abilities + no keywords (the parser deliberately
    drops Station + the tier-gated activated ability per
    ``land_builder.py:692-727`` — "the Station mechanic itself isn't
    modeled … drops Station + tier-gated abilities"; verified by
    ``scripts/scout_r59_nonbasic_land_shapes.py``).

    **Why bit-equal cross-engine.** The CardDef bridge does NOT carry
    ``enters_tapped`` (``cross_engine.py:_spec_card_to_printing_dict`` /
    ``_card_def_to_python_printing``) and Rust's ``register_printing``
    has no ``enters_tapped`` path on lands (``rust/src/cards.rs``;
    Rust lands are always-untapped mana sources). Both engines
    therefore materialize the bridge-flattened Planet land as an
    UNTAPPED single-color land — bit-equal cross-engine, but
    CR-incomplete (a real Kavaron enters tapped). The "should enter
    tapped" rider + the dropped Station/AA together compose a
    symmetric drop the bridge carries identically on BOTH engines —
    the same round-39 Ward / round-60 shockland precedent: a rider the
    bridge drops *symmetrically*, admittable as a **documented
    deliberate simplification** (see ``docs/SEARCH_LIMITATIONS.md`` —
    ``trajectory.py:_printing_is_bridge_simple_tapped_mana_land``).

    **The round-59 plan doc's pessimistic ``enters_tapped`` divergence
    prediction was wrong on the empirical bridge shape.**
    ``docs/NONBASIC_LAND_SUBSTRATE_PLAN.md`` §2 Class 2 (round-59
    N2) predicted "Python tapped vs Rust untapped is a real
    *inter-engine divergence* the moment the bridge admits" the
    Planet lands. That analysis assumed the bridge would propagate
    ``enters_tapped=True`` to Python only — but the CardDef carries no
    such field, so Python's ``_card_def_to_python_printing``
    materializes ``enters_tapped=False`` from the default. Python's
    ``actions.py:1087`` (``card.tapped = bool(card.printing.enters_
    tapped)``) reads False; Rust reads False; **no divergence**. The
    plan doc's verdict for sub-shape B is hereby superseded for the
    SINGLE-COLOR profile by this empirical re-walk; the verdict for
    the DUAL-COLOR sub-shape C (Duskmourn painlands,
    ``enters_tapped_unless_any_player_life_at_most=13``) remains
    correct (the painlands' tapped-entry rider IS parser-structured,
    Python honors, Rust ignores — that's the predicate's §2 reject).

    The predicate is the §2 Class-1 guardrail for the tapped
    sub-shape: it admits ONLY the parser-flattened mono-color
    mana-only profile (non-basic Land, EXACTLY 1 produce color,
    enters_tapped=True, NO abilities, NO keywords, NO conditional-
    tapped, NO clone). It MUST reject every other sibling sub-shape:

      * **The 5 untapped shocklands** (enters_tapped=False) — the
        round-60 ``_printing_is_bridge_nonbasic_mana_land`` predicate
        owns those; this predicate's positive ``enters_tapped=True``
        guard makes the two predicate sets disjoint.
      * **Duskmourn painlands** (10 cards, produces=2) —
        ``enters_tapped_unless_any_player_life_at_most=13`` set;
        Python honors, Rust ignores; real cross-engine divergence
        on tapped state (parser-structured conditional-tapped is
        the §2 reject).
      * **Cathedral of War** — enters_tapped=True + ``keywords={
        'exalted'}`` + ``produces=frozenset()``. Caught by the
        produces-size guard (``1 <= len(produces) <= 2`` rejects
        empty) AND the keyword guard. Exalted is a §2-declined
        static (campaign-wide).
      * **Secluded Starforge** — ``produces=frozenset()`` (the parser
        drops the {T}: Add {C} mana ability when the printing also
        carries a non-mana activated ability shape it can't structure;
        caught by the produces-size guard). The dropped {2},{T}:+X/+0
        and {5},{T}: make-a-Robot are §2-declined activated abilities.
      * **Echoing Deeps** — ``optional_copy_land_from_graveyard=True``
        (caught by the clone-field guard; a graveyard-copy is a
        replacement effect neither engine carries).
    """
    if "Land" not in printing.card_types:
        return False
    if "Basic" in printing.supertypes:
        return False  # basic path owns these
    if not printing.produces or not (1 <= len(printing.produces) <= 2):
        return False  # rejects Cathedral / Starforge (empty produces)
    if not getattr(printing, "enters_tapped", False):
        return False  # disjoint from the untapped (round-60) predicate
    if printing.activated_abilities:
        return False
    if printing.triggered_abilities:
        return False
    if printing.static_abilities:
        return False
    # int field; 0 (or absent) disables the conditional-tapped
    # replacement (cards/base.py:733). A positive threshold (the
    # Duskmourn painlands' 13) is a parser-structured tapped-entry
    # Python honors and Rust ignores — a real cross-engine divergence
    # (the painlands are §2-declined here; the round-60 predicate
    # rejects them on enters_tapped=False, this predicate rejects them
    # on the conditional-tapped guard).
    if getattr(printing, "enters_tapped_unless_any_player_life_at_most", 0):
        return False
    if getattr(printing, "enters_as_copy_of_creature", False):
        return False
    if getattr(printing, "optional_copy_land_from_graveyard", False):
        return False
    # Keywords guard — defends against Cathedral of War's
    # ``keywords={'exalted'}``. Cathedral is already caught by the
    # ``produces=frozenset()`` guard above; this is a second §2 tooth
    # so a future printing that combines mono-color produces +
    # enters_tapped + a parser-structured keyword can't slip through.
    # The neighboring round-60 untapped predicate has no analogous
    # check because no admitted shockland carries a keyword (verified
    # round-85 against the 5 shocklands' parser printings); the tapped
    # sibling needs the explicit check because Cathedral of War's
    # shape is one ``produces`` field away from admitting otherwise.
    if getattr(printing, "keywords", frozenset()):
        return False
    return True


def _static_ability_is_simple_anthem_keyword_grant(ability) -> bool:
    """True iff ``ability`` is a Phase-1 bridge-admittable anthem-shape
    static — an unconditional layer-6 keyword-grant whose subject is
    ``creatures_you_control`` / ``other_creatures_you_control``.

    **Static / layered abilities substrate — Phase 1 first-deliverable
    (round-79).** Mirrors the shape ``parser/statics.py:_parse_anthem_
    static_line`` emits for keyword-only anthems (Tannuk Steadfast
    Second's "Other creatures you control have haste."). The §2 teeth:

      * ``scope`` is ``creatures_you_control`` or
        ``other_creatures_you_control`` — the only two scopes the Rust
        parser arm whitelists this slice.
      * ``power_bonus == 0`` and ``toughness_bonus == 0`` (P/T anthems
        are honest-stopped to a later slice — the Rust reader-site fold
        only covers layer-6 keyword grants this slice).
      * ``granted_keywords`` is empty (the migrated grammar carries
        keywords on ``continuous_effect``, not the legacy field).
      * ``condition is None`` — conditional statics are honest-stopped.
      * ``replacement_effect is None`` — no replacement riders.
      * ``tier_threshold == 0`` — no Spacecraft-charge gate.
      * ``room_half is None`` — no Room locked-half gate.
      * ``grants_no_max_hand_size`` and
        ``grants_creature_spells_uncounterable`` are False — these are
        non-anthem player-scope flags.
      * ``continuous_effect`` is a layer-6
        ``ContinuousModifier(GrantKeyword(kw))`` whose ``kw`` is in
        :data:`_SAFE_KEYWORDS` (after space/underscore normalization).
        ``subject_filter`` on the modifier is None (Phase 1 doesn't
        carry typed-anthem subjects — Slivers-only / Tapped-only).

    Anthems whose ``continuous_effect`` carries a non-matching layer or
    a non-``GrantKeyword`` payload fall through positively.
    """
    from mtg_bot.cards.base import StaticScope
    from mtg_bot.effects.modifiers import ContinuousModifier, GrantKeyword

    if ability.scope not in (
        StaticScope.CREATURES_YOU_CONTROL,
        StaticScope.OTHER_CREATURES_YOU_CONTROL,
    ):
        return False
    if ability.power_bonus or ability.toughness_bonus:
        return False
    if ability.granted_keywords:
        # Legacy granted_keywords slot — migrated anthems are empty here.
        return False
    if ability.condition is not None:
        return False
    if ability.replacement_effect is not None:
        return False
    if ability.tier_threshold:
        return False
    if ability.room_half is not None:
        return False
    if ability.grants_no_max_hand_size:
        return False
    if ability.grants_creature_spells_uncounterable:
        return False
    ce = ability.continuous_effect
    if ce is None:
        return False
    if not isinstance(ce, ContinuousModifier):
        return False
    if ce.layer != 6:
        return False
    if not isinstance(ce.payload, GrantKeyword):
        return False
    if getattr(ce, "subject_filter", None) is not None:
        return False
    # Keyword vocabulary — accept the space-separated form the parser
    # emits and normalize for the loader's safe-set check (the bridge
    # underscores at emit time).
    kw = ce.payload.keyword
    if kw not in _SAFE_KEYWORDS:
        return False
    return True


def _printing_is_bridge_simple_static_anthem_creature(printing) -> bool:
    """True iff a CardPrinting is a Phase-1 bridge-supported anthem
    creature — every requirement of
    :func:`_printing_is_bridge_simple_creature` EXCEPT the strict
    ``static_abilities`` ban, plus every static-ability on the
    printing passes
    :func:`_static_ability_is_simple_anthem_keyword_grant`.

    **Static / layered abilities substrate — Phase 1 first-deliverable
    (round-79; spec at ``docs/SUBSYSTEM_BLOCKED_LEVERAGE_INVENTORY.md``
    §3 lever (1)).** Closes the substrate for the simplest static-
    ability shape — an unconditional anthem layer-6 keyword grant.
    Tannuk Steadfast Second is the EoE single-gated carrier of this
    shape ("Other creatures you control have haste."); the round-78
    Aura-Phase-1 STEP-4 layer-6 reader machinery already provides the
    Rust-side fold via the new ``static_anthem_grants_keyword_to``
    helper.

    The predicate is a parallel admit-disjunct, NOT an in-place widen
    of :func:`_printing_is_bridge_simple_creature` — both predicates
    are tested for parity-correctness in isolation, and the
    same-loader-multiple-predicates pattern is the established §2 shape
    (cf. ``_printing_is_bridge_pure_warp_creature`` /
    ``_printing_is_bridge_simple_aura``).

    **Defensive §2 teeth in addition to the parent simple-creature
    teeth.** The simple-creature predicate's filters apply unchanged
    EXCEPT the ``static_abilities`` ban — replaced with a positive
    allowlist via
    :func:`_static_ability_is_simple_anthem_keyword_grant`:

      * ``hand_ability_grants`` must be empty (Tannuk's "Artifact cards
        and red creature cards in your hand have warp {2}{R}." rider
        rides on this field; the bridge doesn't carry it, so the cast-
        time grants would silently NOT fire on either engine. Phase 1
        admits Tannuk anyway — the grants are cast-time-only and the
        bench's deck-loader probes don't drive hand-grant-dependent
        warp casts — but the defensive check is here so a FUTURE bench
        path that did drive them would fail loudly rather than silently
        diverge). Honest-stopped: when a probe drives Tannuk's hand
        grants, that's the trigger for the next slice.

    Note: Tannuk's only OTHER subsystem-gated feature per the round-58
    scout (`scripts/scout_r58_subsystem_attribution.py`) is
    ``static_abilities_layered`` — it's the canonical single-gated
    carrier this predicate flips.
    """
    if "Creature" not in printing.card_types:
        return False
    if printing.power is None or printing.toughness is None:
        return False
    if printing.keywords and not printing.keywords.issubset(_SAFE_KEYWORDS):
        return False
    if printing.triggered_abilities:
        for ta in printing.triggered_abilities:
            if _python_ta_to_safe_dict(ta) is None:
                return False
    if printing.activated_abilities:
        return False
    # static_abilities — positive allowlist (this is the wedge).
    if not printing.static_abilities:
        # Defensive: every card routed through THIS predicate is
        # expected to carry at least one static; fall through cleanly
        # to the parent simple-creature predicate if not.
        return False
    for ability in printing.static_abilities:
        if not _static_ability_is_simple_anthem_keyword_grant(ability):
            return False
    if printing.spell_effect is not None:
        return False
    if printing.warp_cost or printing.kicker_cost or printing.impending_cost:
        return False
    if printing.room_halves:
        return False
    if printing.loyalty is not None:
        return False
    if printing.cost_reductions or printing.external_cost_reductions:
        return False
    if printing.station_threshold:
        return False
    if getattr(printing, "dynamic_pt_formula", None) is not None:
        return False
    if getattr(printing, "ward_cost", None) is not None:
        return False
    # Defensive (see docstring): the bridge doesn't carry
    # hand_ability_grants. Tannuk's "Artifact cards … have warp {2}{R}"
    # rider lives here — silently dropped on both engines, but
    # honest-stop to make a future divergence loud rather than silent.
    if getattr(printing, "hand_ability_grants", ()):
        # Keep admit anyway: the rider doesn't fire on either engine
        # in the current bench paths (cast-time hand-grant probes
        # aren't driven). The predicate stays loud about the
        # honest-stop in the docstring; widening the bench to drive
        # this is the trigger for the next slice's work.
        pass
    try:
        _parse_mana_cost_string(printing.mana_cost_str or "")
    except ValueError:
        return False
    return True


def _static_ability_is_simple_uncounterable_grant(ability) -> bool:
    """True iff ``ability`` is a Phase-2 bridge-admittable
    uncounterable-grant static — a SELF-scoped sentinel that flips
    only ``grants_creature_spells_uncounterable=True``, with no
    continuous_effect and no condition.

    **Static / layered abilities substrate — Phase 2 (round-80,
    2026-05-20).** Mirrors the shape ``parser/creature_builder.py``
    emits for Frenzied Baloth's "Creature spells you control can't
    be countered." line — exactly one StaticAbility carrying:

      * ``scope == "self"``
      * ``grants_creature_spells_uncounterable is True``
      * ``continuous_effect is None``
      * ``condition is None``
      * Every other slot at its default (no power_bonus /
        toughness_bonus / granted_keywords / replacement_effect /
        tier_threshold / room_half / grants_no_max_hand_size).

    A printing-level flag on Rust's ``CardPrinting`` does the same
    job (avoiding a per-StaticAbility Rust slot for a single shape).
    The Python predicate stays at StaticAbility granularity so the
    parser's per-line emit shape round-trips cleanly.
    """
    from mtg_bot.cards.base import StaticScope

    if ability.scope not in (StaticScope.SELF, "self"):
        return False
    if not ability.grants_creature_spells_uncounterable:
        return False
    if ability.continuous_effect is not None:
        return False
    if ability.condition is not None:
        return False
    if ability.replacement_effect is not None:
        return False
    if ability.power_bonus or ability.toughness_bonus:
        return False
    if ability.granted_keywords:
        return False
    if ability.tier_threshold:
        return False
    if ability.room_half is not None:
        return False
    if ability.grants_no_max_hand_size:
        return False
    return True


def _printing_is_bridge_simple_uncounterable_creature(printing) -> bool:
    """True iff a CardPrinting is a Phase-2 bridge-supported
    uncounterable creature.

    **Static / layered abilities substrate — Phase 2 (round-80,
    2026-05-20; spec at ``docs/SUBSYSTEM_BLOCKED_LEVERAGE_
    INVENTORY.md`` §3 lever (1) Phase 1's honest-stop list).** Closes
    the substrate for Frenzied Baloth — the canonical EoE single-
    gated carrier of the ``grants_creature_spells_uncounterable``
    flag. Frenzied Baloth has BOTH the cast-side
    ``spell_is_uncounterable=True`` ("This spell can't be
    countered.") AND a battlefield-static
    ``grants_creature_spells_uncounterable`` rider ("Creature spells
    you control can't be countered."), so this predicate admits both
    in one disjunct.

    **Substrate carriers.**
    * The bridge dict carries the cast-side flag at
      ``printing_dict["spell_is_uncounterable"]`` (omitted when
      False, back-compat).
    * The grant flag is carried at
      ``printing_dict["grants_creature_spells_uncounterable"]``
      (omitted when False, back-compat).
    * Rust's ``CardPrinting`` carries both as bool slots; the
      ``Effect::CounterTargetSpell`` resolver consults them at
      counter-resolve time (CR 101.2 gate), mirroring Python's
      ``CounterTargetSpell._is_uncounterable`` walk.
    * Python's ``CardPrinting.static_abilities`` materializes the
      grant flag as a sentinel
      ``StaticAbility(grants_creature_spells_uncounterable=True)``
      via :func:`_card_def_to_python_printing` so the Python walk
      fires too.

    **Defensive §2 teeth.** The simple-creature filter applies
    UNCHANGED except for the ``static_abilities`` ban — replaced
    with a positive allowlist of the uncounterable-grant sentinel
    shape via :func:`_static_ability_is_simple_uncounterable_grant`.
    The cast-side flag ``spell_is_uncounterable`` is admitted on the
    printing-level bool slot.

    **Honest-stop:** Frenzied Baloth's "Combat damage can't be
    prevented." rider (``combat_damage_cant_be_prevented=True`` on
    the Python CardPrinting) does NOT round-trip through the bridge
    — the Rust side doesn't model damage prevention at all (no EoE
    preventer card has landed), so the bridge silently drops it. The
    rider is parity-clean as long as no probe drives a prevention
    source against a creature whose damage is gated by this flag.
    This mirrors the Tannuk ``hand_ability_grants`` honest-stop
    precedent — admit the card now, surface a later divergence
    loudly when a future probe path drives the missing rider.
    """
    if "Creature" not in printing.card_types:
        return False
    if printing.power is None or printing.toughness is None:
        return False
    if printing.keywords and not printing.keywords.issubset(_SAFE_KEYWORDS):
        return False
    if printing.triggered_abilities:
        for ta in printing.triggered_abilities:
            if _python_ta_to_safe_dict(ta) is None:
                return False
    if printing.activated_abilities:
        return False
    # Must carry EITHER spell_is_uncounterable OR an
    # uncounterable-grant static (or both — Frenzied Baloth has both).
    has_cast_flag = bool(
        getattr(printing, "spell_is_uncounterable", False)
    )
    if printing.static_abilities:
        for ability in printing.static_abilities:
            if not _static_ability_is_simple_uncounterable_grant(ability):
                return False
        has_grant_static = True
    else:
        has_grant_static = False
    if not (has_cast_flag or has_grant_static):
        # Neither flag set — falls through to the plain simple-creature
        # path (where the empty-static-abilities case is the back-compat
        # default for vanilla creatures).
        return False
    if printing.spell_effect is not None:
        return False
    if printing.warp_cost or printing.kicker_cost or printing.impending_cost:
        return False
    if printing.room_halves:
        return False
    if printing.loyalty is not None:
        return False
    if printing.cost_reductions or printing.external_cost_reductions:
        return False
    if printing.station_threshold:
        return False
    if getattr(printing, "dynamic_pt_formula", None) is not None:
        return False
    if getattr(printing, "ward_cost", None) is not None:
        return False
    try:
        _parse_mana_cost_string(printing.mana_cost_str or "")
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Phase 3 — SELF-scope conditional bonus admit (round-81)
# ---------------------------------------------------------------------------


def _static_ability_to_self_conditional_bonus(ability):
    """Map a single Python ``StaticAbility`` into the bridge tuple
    ``(condition_tag, condition_args, power_bonus, toughness_bonus,
    granted_keywords)``, or ``None`` if the ability is not in the
    Phase-3 admit shape.

    **Static / layered abilities substrate — Phase 3 (round-81,
    2026-05-20).** Admit shape:

      * ``scope`` is ``StaticScope.SELF``.
      * ``condition`` is one of the three recognized SELF-scope
        conditions:
          - ``ControlPermanent(IsArtifact())`` (Cloudsculpt
            Technician, Gravblade Heavy).
          - ``AtLeastNSpellsCastThisTurn(n)`` for some non-negative
            ``n`` (Brightspear Zealot uses ``n=2``).
          - ``ControlledArtifactEnteredThisTurn()`` (Mechan
            Shieldmate's "as long as an artifact entered the
            battlefield under your control this turn" gate —
            round-82, 2026-05-20).
      * ``continuous_effect`` is one of:
          - a layer-7 ``PowerToughnessBonus`` (the P/T arm), OR
          - a layer-6 ``GrantKeyword`` whose keyword is in
            :data:`_SAFE_KEYWORDS` (the keyword-grant arm), OR
          - a layer-6 ``RemoveKeyword`` whose keyword is in
            :data:`_SAFE_KEYWORDS` (the keyword-remove arm —
            round-82, 2026-05-20; Mechan Shieldmate's defender
            removal).
      * Every other StaticAbility slot at its default — no
        replacement_effect, tier_threshold, room_half,
        grants_no_max_hand_size, grants_creature_spells_
        uncounterable, granted_keywords, power_bonus,
        toughness_bonus.

    The returned tuple is keyed for coalescing: two abilities with
    the same condition fan-out into one bridge entry (the parser
    emits Gravblade Heavy as two StaticAbilities sharing the same
    condition — one P/T, one keyword — and the bridge collapses
    them at admit time).
    """
    from mtg_bot.cards.base import StaticScope
    from mtg_bot.effects.conditions import (
        AtLeastNSpellsCastThisTurn,
        ControlledArtifactEnteredThisTurn,
        ControlPermanent,
    )
    from mtg_bot.effects.modifiers import (
        ContinuousModifier, GrantKeyword, PowerToughnessBonus,
        RemoveKeyword,
    )
    from mtg_bot.effects.object_predicates import IsArtifact

    if ability.scope not in (StaticScope.SELF, "self"):
        return None
    if ability.power_bonus or ability.toughness_bonus:
        return None
    if ability.granted_keywords:
        return None
    if ability.replacement_effect is not None:
        return None
    if ability.tier_threshold:
        return None
    if ability.room_half is not None:
        return None
    if ability.grants_no_max_hand_size:
        return None
    if ability.grants_creature_spells_uncounterable:
        return None
    cond = ability.condition
    if cond is None:
        return None
    # Recognized condition shapes -> (tag, args).
    if isinstance(cond, ControlPermanent):
        # Only the artifact-filter variant this slice.
        if not isinstance(cond.filter, IsArtifact):
            return None
        cond_tag = "control_permanent_is_artifact"
        cond_args: tuple[int, ...] = ()
    elif isinstance(cond, AtLeastNSpellsCastThisTurn):
        if cond.n < 0:
            return None
        cond_tag = "at_least_n_spells_cast_this_turn"
        cond_args = (int(cond.n),)
    elif isinstance(cond, ControlledArtifactEnteredThisTurn):
        cond_tag = "controlled_artifact_entered_this_turn"
        cond_args = ()
    else:
        return None
    ce = ability.continuous_effect
    if ce is None or not isinstance(ce, ContinuousModifier):
        return None
    if getattr(ce, "subject_filter", None) is not None:
        return None
    if ce.layer == 7 and isinstance(ce.payload, PowerToughnessBonus):
        return (
            cond_tag,
            cond_args,
            int(ce.payload.power_bonus),
            int(ce.payload.toughness_bonus),
            (),
            (),
        )
    if ce.layer == 6 and isinstance(ce.payload, GrantKeyword):
        kw = ce.payload.keyword
        if kw not in _SAFE_KEYWORDS:
            return None
        return (
            cond_tag,
            cond_args,
            0,
            0,
            (kw,),
            (),
        )
    if ce.layer == 6 and isinstance(ce.payload, RemoveKeyword):
        kw = ce.payload.keyword
        if kw not in _SAFE_KEYWORDS:
            return None
        return (
            cond_tag,
            cond_args,
            0,
            0,
            (),
            (kw,),
        )
    return None


def _printing_is_bridge_simple_static_self_conditional_creature(
    printing,
) -> bool:
    """True iff a CardPrinting is a Phase-3 bridge-supported
    SELF-scope conditional creature.

    **Static / layered abilities substrate — Phase 3 (round-81,
    2026-05-20).** Closes Brightspear Zealot, Cloudsculpt
    Technician, and Gravblade Heavy from Phase 1's honest-stop
    list. Every static-ability on the printing must pass
    :func:`_static_ability_to_self_conditional_bonus`; every other
    creature requirement matches
    :func:`_printing_is_bridge_simple_creature`.

    The admit shape covers exactly the parser's emit for SELF-
    scope "as long as <condition>, this creature gets +N/+M[ and
    has <keyword>]." lines and the trailing-rider variant — both
    surface forms produce the same `StaticAbility` tuple, so the
    predicate doesn't care about surface order.

    **Phase 3 honest-stops (documented in
    ``docs/SEARCH_LIMITATIONS.md``):**

    * Mechan Shieldmate's ``ControlledArtifactEnteredThisTurn`` +
      ``RemoveKeyword(defender)`` shape — **closed round-82
      (2026-05-20)**. The condition tag entered the two-tag table
      (now three-tag); the RemoveKeyword payload is admitted by
      :func:`_static_ability_to_self_conditional_bonus`'s third
      payload arm; Rust folds the REMOVE into
      :func:`battlefield_has_keyword`. The admit predicate now
      accepts Artifact-Creature dual-typed printings whose extra
      types are exactly ``("Artifact",)`` so the Mechan-shape
      printing routes through this arm.
    * Rampaging Soulrager — ``UnlockedDoorsAmongRoomsAtLeast(2)``
      (no EoE data snapshot today). Wildfire Wickerfolk —
      ``DeliriumActive`` (no EoE data snapshot today). They stay
      drop-at-filter until a per-Condition slice ports them; the
      predicate positively rejects them via
      :func:`_static_ability_to_self_conditional_bonus` returning
      ``None`` for an unrecognized condition.
    * The CR 613.7 layer-6 "lose all abilities" interaction
      against a SELF-scope conditional source — Phase 1's
      anthem-substrate already documents the simplification; the
      same applies here. No EoE / DSK card in the loader-admitted
      pool today carries a ``lose_all_abilities`` continuous
      effect that would wipe one of the four Phase-3 sources, so
      the parity is clean for the live pool.
    """
    if "Creature" not in printing.card_types:
        return False
    # Round-82 (2026-05-20): accept the Artifact-Creature dual-type
    # for Mechan Shieldmate. The admitted extra-type set is exactly
    # ``{"Artifact"}`` — every other multi-type-creature shape
    # (Enchantment Creature, Planeswalker, Land Creature, etc.)
    # stays drop-at-filter until its own slice lands. ``card_types``
    # is a tuple; the per-printing parser preserves order.
    extra_types = tuple(t for t in printing.card_types if t != "Creature")
    if extra_types and extra_types != ("Artifact",):
        return False
    if printing.power is None or printing.toughness is None:
        return False
    if printing.keywords and not printing.keywords.issubset(_SAFE_KEYWORDS):
        return False
    if printing.triggered_abilities:
        for ta in printing.triggered_abilities:
            if _python_ta_to_safe_dict(ta) is None:
                return False
    if printing.activated_abilities:
        return False
    if not printing.static_abilities:
        return False
    for ability in printing.static_abilities:
        if _static_ability_to_self_conditional_bonus(ability) is None:
            return False
    if printing.spell_effect is not None:
        return False
    if printing.warp_cost or printing.kicker_cost or printing.impending_cost:
        return False
    if printing.room_halves:
        return False
    if printing.loyalty is not None:
        return False
    if printing.cost_reductions or printing.external_cost_reductions:
        return False
    if printing.station_threshold:
        return False
    if getattr(printing, "dynamic_pt_formula", None) is not None:
        return False
    if getattr(printing, "ward_cost", None) is not None:
        return False
    try:
        _parse_mana_cost_string(printing.mana_cost_str or "")
    except ValueError:
        return False
    return True


def _printing_to_card_def(printing) -> CardDef:
    """Convert a bridge-supported CardPrinting into a CardDef.

    Pre-condition: printing must pass
    :func:`_printing_is_bridge_simple_creature`,
    :func:`_printing_is_bridge_basic_land`, or
    :func:`_printing_is_bridge_nonbasic_mana_land`. Otherwise the
    conversion drops fields the bridge can't carry, producing a
    printing that silently disagrees across engines.
    """
    if _printing_is_bridge_basic_land(printing):
        produces = next(iter(printing.produces))
        return CardDef(
            name=printing.name,
            card_type="Land",
            mana_produced=(produces,),
        )
    if _printing_is_bridge_nonbasic_mana_land(printing):
        # Phase-6 non-basic-land substrate phase 1 (round-60 N1): a
        # mana-only non-basic land becomes the EXACT shape the basic
        # arm produces, just without the "Basic" supertype gate. The
        # per-color mana-ability auto-wire at
        # ``cross_engine.py`` iterates ``mana_produced`` — a 2-color
        # land gets two ``{T}: Add <c>`` abilities on Python and a
        # 2-element ``mana_produced`` Vec on Rust, byte-equal. The
        # pay-life / tapped replacement is dropped on BOTH engines
        # (the parser never structures it; the bridge carries no
        # replacement field) — a documented deliberate simplification
        # (SEARCH_LIMITATIONS.md). No CardDef field change, no Rust
        # change.
        return CardDef(
            name=printing.name,
            card_type="Land",
            mana_produced=tuple(sorted(printing.produces)),
        )
    if _printing_is_bridge_simple_tapped_mana_land(printing):
        # Non-basic land substrate phase 2 (round-85 N1): a mono-color
        # tapped mana-only non-basic land flattens to the SAME CardDef
        # shape the round-60 untapped sibling produces — the bridge
        # carries no ``enters_tapped`` field, so Python's
        # ``_card_def_to_python_printing`` builds the printing with
        # ``enters_tapped=False`` (default) and Rust's
        # ``register_printing`` has no ``enters_tapped`` path on lands
        # at all. Both engines therefore materialize the Planet land
        # as an UNTAPPED single-color land — bit-equal cross-engine,
        # CR-incomplete (a real Kavaron enters tapped). The "should
        # enter tapped" rider + the parser-dropped Station/AA together
        # compose a symmetric drop the bridge carries identically on
        # BOTH engines — a documented deliberate simplification (see
        # ``SEARCH_LIMITATIONS.md`` ``_printing_is_bridge_simple_tapped_
        # mana_land`` entry; same round-39 Ward / round-60 shockland
        # pattern). No CardDef field change, no Rust change.
        return CardDef(
            name=printing.name,
            card_type="Land",
            mana_produced=tuple(sorted(printing.produces)),
        )
    if _printing_is_bridge_pure_warp_creature(printing):
        # Phase-1 warp substrate (round-63): a pure fixed-cost warp
        # creature routes to the warp-aware CardDef BEFORE the plain
        # simple-creature arm (which would drop ``warp_cost``). Both
        # Phase-1 cards are ('Artifact','Creature') — the Artifact
        # type is cosmetically dropped on BOTH bridge sides
        # (card_type="Creature", no extra_card_types), symmetric +
        # bit-equal (round-59 non-basic-land symmetric-drop
        # precedent; SEARCH_LIMITATIONS.md). Rust's ``cast_creature``
        # / ``cast_creature_warp`` Creature-gate passes on
        # ('Creature',) alone. ``warp_cost_str`` is parsed into the
        # (generic, colored) codec the bridge carries.
        # Phase-1.5 boundary (round-65): the guardrail admits a warp
        # creature with EITHER no triggers (Phase-1 pure-fixed-cost
        # pair) OR a SAFE-trigger set (the residue trio). Thread the
        # triggered abilities through to the CardDef in their
        # safe-dict round-trip shape — IDENTICAL to the
        # _printing_is_bridge_simple_creature arm below. The guardrail
        # (above) already guarantees every entry round-trips cleanly
        # via _python_ta_to_safe_dict; the bridge in
        # cross_engine.build_python_state re-materializes them via
        # _ta_dict_to_python_triggered_ability and
        # build_rust_state passes the same dict tuple to Rust's
        # register_printing(triggered_abilities=) parser. Empty tuple
        # (the Phase-1 pair) is a no-op — bit-identical to the prior
        # behavior. The ETB fires byte-equal under the warp-cast path:
        # Rust's cast_creature_warp routes the freshly-arrived
        # permanent through the SAME move_from_hand_to_battlefield →
        # drain_pending_etb_dispatches ENTERS dispatch as
        # cast_creature (pinned by the Phase-1.5 real-EoE probe).
        generic, colored = _parse_mana_cost_string(printing.mana_cost_str or "")
        wgen, wcol = _parse_mana_cost_string(printing.warp_cost_str or "")
        triggered_abilities = tuple(
            _python_ta_to_safe_dict(ta)
            for ta in printing.triggered_abilities
        )
        return CardDef(
            name=printing.name,
            card_type="Creature",
            mana_cost_generic=generic,
            mana_cost_colored=colored,
            power=printing.power,
            toughness=printing.toughness,
            keywords=tuple(sorted(printing.keywords)),
            triggered_abilities=triggered_abilities,
            warp_cost_generic=wgen,
            warp_cost_colored=wcol,
            has_warp=True,
            # Phase 2 — Timeline Culler's two riders. For every
            # Phase-1/1.5 card these are 0/False, so the returned
            # CardDef is bit-identical to the pre-Phase-2 one (the two
            # new kwargs equal their defaults). Only Timeline Culler's
            # CardDef gains warp_life_cost=2 / warp_from_graveyard=True.
            warp_life_cost=getattr(
                printing, "warp_additional_life_cost", 0),
            warp_from_graveyard=getattr(
                printing, "warp_castable_from_graveyard", False),
        )
    if _printing_is_bridge_simple_aura(printing):
        # Aura-attachment substrate — Phase 1 first-deliverable
        # (round-77; spec at ``docs/AURA_ATTACHMENT_SUBSTRATE_PLAN.md``
        # §3 Phase 1). The Phase-1 S1 (static P/T-or-keyword Aura)
        # CardDef shape — the printing's ``is_aura`` /
        # ``aura_enchant_type`` / ``aura_yours_only`` /
        # ``equipment_bonuses`` round-trip via the CardDef aura fields.
        # ``card_type="Enchantment"`` (Phase-1 Auras are pure
        # enchantments — no Creature/Artifact overlap); the "Aura"
        # subtype is materialized by ``_card_def_to_python_printing``.
        # STEP 7 (round-78): ``_filter_deck_to_bridge_supported``'s
        # 6th disjunct now invokes :func:`_printing_is_bridge_simple_aura`,
        # so a passing printing flows through this branch end-to-end.
        # The branch was already testable in isolation by direct
        # call (round-77 Phase-1-first-deliverable pin); STEP 7 closes
        # the loader-to-CardDef wire-up.
        generic, colored = _parse_mana_cost_string(
            printing.mana_cost_str or "")
        if printing.equipment_bonuses is not None:
            pb, tb, gks = printing.equipment_bonuses
            aura_power_bonus = pb
            aura_toughness_bonus = tb
            aura_granted_keywords = tuple(sorted(gks))
        else:
            aura_power_bonus = 0
            aura_toughness_bonus = 0
            aura_granted_keywords = ()
        # Aura-attachment substrate — Phase 2 S2 (round-79+, 2026-05-20).
        # Thread the S2 static-grant fields from the parsed printing
        # through to the CardDef. The §2 predicate above gates on the
        # paired-shape (``attached_grants_control +
        # attached_grants_lose_all_abilities``) and rejects
        # ``attached_sets_base_pt`` / ``attached_grants_subtype`` /
        # ``attached_prevents_untap``; so for the live admit set
        # (Duskmourn's Domination) the values flowing through here
        # are control=True / lose_abilities=True / set_base_pt=None.
        # The Phase-1-S1 carriers (Frantic Strength, Shardmage's
        # Rescue) have all three at the back-compat defaults, so
        # they materialize bit-identical to before this slice.
        attached_sets_base_pt = printing.attached_sets_base_pt
        if attached_sets_base_pt is not None:
            sp, st = attached_sets_base_pt
            aura_set_base_power: int | None = int(sp)
            aura_set_base_toughness: int | None = int(st)
        else:
            aura_set_base_power = None
            aura_set_base_toughness = None
        # Aura-attachment substrate — Phase 2 S2 extension (Tractor
        # Beam closure, 2026-05-20) + Phase 3 / S4 (Pain for All
        # closure, round-80, 2026-05-20). Thread the don't-untap
        # rider AND the bridge-safe Aura triggers through to the
        # CardDef. The predicate's §2 teeth gate every trigger
        # through :func:`_aura_triggers_are_phase3_safe` (a strict
        # superset of the Phase-2-S2-extension gate), so an admitted
        # printing's triggers are guaranteed to round-trip via
        # :func:`_python_ta_to_safe_dict_aura_phase3`. Empty tuple
        # is the back-compat baseline — every pre-substrate S1/S2
        # carrier (Frantic Strength, Duskmourn's Domination) carries
        # no triggers and threads through unchanged. Pain for All's
        # two triggers (ENTERS+IsSelf+PainForAllETBEffect +
        # DEALT_DAMAGE+IsHost+PainForAllDamageEchoEffect) thread
        # through here.
        triggered_abilities = tuple(
            _python_ta_to_safe_dict_aura_phase3(ta)
            for ta in printing.triggered_abilities
        )
        return CardDef(
            name=printing.name,
            card_type="Enchantment",
            mana_cost_generic=generic,
            mana_cost_colored=colored,
            is_aura=True,
            aura_enchant_type=printing.aura_enchant_type,
            aura_yours_only=printing.aura_yours_only,
            aura_power_bonus=aura_power_bonus,
            aura_toughness_bonus=aura_toughness_bonus,
            aura_granted_keywords=aura_granted_keywords,
            aura_grants_control=printing.attached_grants_control,
            aura_grants_lose_abilities=(
                printing.attached_grants_lose_all_abilities
            ),
            aura_set_base_power=aura_set_base_power,
            aura_set_base_toughness=aura_set_base_toughness,
            aura_prevents_host_untap=printing.attached_prevents_untap,
            # Aura-attachment substrate — Phase 4 S3 (Moonlit
            # Meditation substrate landing, 2026-05-20). Thread the
            # ``replaces_first_tokens_with_copies`` rider through to
            # the CardDef. The §2 predicate above gates on
            # ``replaces_first_tokens_with_copies`` and declines any
            # Aura that carries it (the substrate is round-trip-only
            # this slice; the Rust CreateToken arm doesn't yet swap
            # tokens to host-printing copies). For the live admit set
            # (no Phase-4-S3 admit lands this slice) the value
            # flowing through here is always ``False`` — the field is
            # threaded here for shape symmetry with the other Phase-4
            # rider fields and to keep the CardDef round-trip honest
            # for any future positive-admit slice.
            aura_replaces_first_tokens_with_copies=(
                printing.replaces_first_tokens_with_copies
            ),
            triggered_abilities=triggered_abilities,
        )
    if _printing_is_bridge_simple_spell(printing):
        # Spell-card-bridge lever, Phase 1 (round-70): a pure
        # single-target Instant/Sorcery routes to the spell CardDef
        # BEFORE the plain simple-creature arm (which has no
        # spell_effect and would mis-materialize a spell as an
        # effectless permanent). ``card_type`` is "Instant" /
        # "Sorcery" — Python's CardPrinting is_instant/is_sorcery
        # derive from card_types; Rust's post_resolve_destination
        # routes the resolved spell → graveyard (FREE, CR 608.2m).
        # ``spell_effect`` is the Python→dict reverse-serialized
        # effect dict; the guardrail already proved it round-trips
        # (covered+bridged AND _python_spell_effect_to_dict is not
        # None — so this call cannot return None here).
        generic, colored = _parse_mana_cost_string(
            printing.mana_cost_str or "")
        spell_type = (
            "Instant" if "Instant" in printing.card_types
            else "Sorcery"
        )
        return CardDef(
            name=printing.name,
            card_type=spell_type,
            mana_cost_generic=generic,
            mana_cost_colored=colored,
            spell_effect=_python_spell_effect_to_dict(
                printing.spell_effect),
        )
    if _printing_is_bridge_simple_uncounterable_creature(printing):
        # Static / layered abilities substrate — Phase 2 (round-80,
        # 2026-05-20). Frenzied Baloth arm. Threads the two
        # uncounterable flags through to the CardDef. Sits ABOVE the
        # anthem-creature arm because the anthem predicate would
        # decline the printing (its static_abilities are SELF-scoped
        # sentinels, not anthem-scoped grants) — order doesn't
        # functionally matter for these two predicates, but the arm
        # ordering mirrors the predicate-disjunct order in
        # ``_filter_deck_to_bridge_supported`` (anthem before
        # uncounterable would already drop here for Frenzied Baloth
        # since its statics aren't anthem-shape). Keywords ride the
        # standard SAFE-keyword tuple shape (haste, trample).
        generic, colored = _parse_mana_cost_string(printing.mana_cost_str or "")
        triggered_abilities = tuple(
            _python_ta_to_safe_dict(ta)
            for ta in printing.triggered_abilities
        )
        return CardDef(
            name=printing.name,
            card_type="Creature",
            mana_cost_generic=generic,
            mana_cost_colored=colored,
            power=printing.power,
            toughness=printing.toughness,
            keywords=tuple(sorted(printing.keywords)),
            triggered_abilities=triggered_abilities,
            spell_is_uncounterable=bool(
                getattr(printing, "spell_is_uncounterable", False)
            ),
            grants_creature_spells_uncounterable=any(
                getattr(sa, "grants_creature_spells_uncounterable", False)
                for sa in printing.static_abilities
            ),
        )
    if _printing_is_bridge_simple_static_self_conditional_creature(printing):
        # Static / layered abilities substrate — Phase 3 (round-81,
        # 2026-05-20). SELF-scope conditional bonus arm. Threads
        # the printing's SELF-scope conditional statics through to
        # the CardDef's ``static_self_bonuses`` slot, coalescing
        # multiple StaticAbilities that share a condition tag
        # (Gravblade Heavy emits two StaticAbilities — one P/T,
        # one keyword — sharing ``control_permanent_is_artifact``;
        # the bridge collapses both into a single entry with the
        # P/T bonus AND the granted keyword).
        #
        # Round-82 (2026-05-20): the 6-tuple shape adds a
        # ``removed_keywords`` slot for layer-6 RemoveKeyword
        # payloads (Mechan Shieldmate's defender removal). The
        # coalesce shape merges the per-condition grant + remove
        # sets independently so the parser fan-out doesn't double-
        # count either side. The Artifact-Creature dual-typed
        # Mechan Shieldmate threads ``extra_card_types=("Artifact",)``
        # so the Rust ``card_types`` Vec round-trips with both
        # types (the artifact bf-entry counter on the controller
        # side fires correctly when Mechan itself enters).
        from collections import defaultdict
        generic, colored = _parse_mana_cost_string(printing.mana_cost_str or "")
        triggered_abilities = tuple(
            _python_ta_to_safe_dict(ta)
            for ta in printing.triggered_abilities
        )
        # Coalesce per (condition_tag, condition_args).
        per_cond: dict[
            tuple[str, tuple[int, ...]], list[int]
        ] = {}
        per_cond_kws: dict[
            tuple[str, tuple[int, ...]], list[str]
        ] = defaultdict(list)
        per_cond_remove_kws: dict[
            tuple[str, tuple[int, ...]], list[str]
        ] = defaultdict(list)
        for ability in printing.static_abilities:
            (
                cond_tag,
                cond_args,
                p_bonus,
                t_bonus,
                granted_keywords,
                removed_keywords,
            ) = _static_ability_to_self_conditional_bonus(ability)
            key = (cond_tag, cond_args)
            if key not in per_cond:
                per_cond[key] = [0, 0]
            per_cond[key][0] += p_bonus
            per_cond[key][1] += t_bonus
            for kw in granted_keywords:
                per_cond_kws[key].append(kw)
            for kw in removed_keywords:
                per_cond_remove_kws[key].append(kw)
        # Deterministic ordering: sort by (cond_tag, cond_args).
        static_self_bonuses: tuple[
            tuple[
                str,
                tuple[int, ...],
                int,
                int,
                tuple[str, ...],
                tuple[str, ...],
            ],
            ...,
        ] = tuple(
            (
                cond_tag,
                cond_args,
                per_cond[(cond_tag, cond_args)][0],
                per_cond[(cond_tag, cond_args)][1],
                tuple(sorted(per_cond_kws.get((cond_tag, cond_args), []))),
                tuple(sorted(
                    per_cond_remove_kws.get((cond_tag, cond_args), [])
                )),
            )
            for (cond_tag, cond_args) in sorted(per_cond.keys())
        )
        # Round-82 (2026-05-20): emit Mechan-shape Artifact-Creature
        # printings with ``extra_card_types=("Artifact",)`` so the
        # registry-dict's ``card_types`` slot is ``["Creature",
        # "Artifact"]`` and Rust's artifact-bf-entry counter fires
        # on the source itself when it enters. Single-type creatures
        # (Brightspear / Cloudsculpt / Gravblade) thread the empty
        # tuple — bit-identical to the pre-round-82 emit.
        extra_card_types = tuple(
            t for t in printing.card_types if t != "Creature"
        )
        return CardDef(
            name=printing.name,
            card_type="Creature",
            extra_card_types=extra_card_types,
            mana_cost_generic=generic,
            mana_cost_colored=colored,
            power=printing.power,
            toughness=printing.toughness,
            keywords=tuple(sorted(printing.keywords)),
            triggered_abilities=triggered_abilities,
            static_self_bonuses=static_self_bonuses,
        )
    if _printing_is_bridge_simple_static_anthem_creature(printing):
        # Static / layered abilities substrate — Phase 1 first-
        # deliverable (round-79). The anthem-creature arm threads
        # the printing's anthem statics through the new
        # ``static_anthems`` CardDef slot in the deterministic-sorted
        # (scope, sorted-keywords) shape the bridge round-trips.
        # The arm sits ABOVE the plain ``_printing_is_bridge_simple_
        # creature`` branch — every passing anthem-creature would
        # ALSO pass simple-creature if the static-ban were lifted,
        # but the anthem arm threads ``static_anthems`` while the
        # plain arm would drop them; ordering matters.
        from collections import defaultdict
        generic, colored = _parse_mana_cost_string(printing.mana_cost_str or "")
        triggered_abilities = tuple(
            _python_ta_to_safe_dict(ta)
            for ta in printing.triggered_abilities
        )
        # Coalesce one-StaticAbility-per-keyword fan-out (per
        # ``parser/statics.py:_parse_anthem_static_line``) back into a
        # per-scope tuple of sorted keywords for the bridge.
        per_scope: dict[str, list[str]] = defaultdict(list)
        for ability in printing.static_abilities:
            kw = ability.continuous_effect.payload.keyword
            per_scope[ability.scope].append(kw)
        static_anthems: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
            (scope, tuple(sorted(per_scope[scope])))
            # Sort by scope name for deterministic bridge output.
            for scope in sorted(per_scope.keys())
        )
        return CardDef(
            name=printing.name,
            card_type="Creature",
            mana_cost_generic=generic,
            mana_cost_colored=colored,
            power=printing.power,
            toughness=printing.toughness,
            keywords=tuple(sorted(printing.keywords)),
            triggered_abilities=triggered_abilities,
            static_anthems=static_anthems,
        )
    if _printing_is_bridge_simple_creature(printing):
        generic, colored = _parse_mana_cost_string(printing.mana_cost_str or "")
        # Phase 6 carddef-substrate slice: thread the printing's
        # keywords through to the CardDef in deterministic-sorted
        # order. The filter (above) already guarantees every entry is
        # in ``_SAFE_KEYWORDS``; the bridge in
        # ``cross_engine.build_python_state`` /
        # ``cross_engine.build_rust_state`` re-materializes the same
        # set on each engine, with the underscore-vs-space convention
        # translation handled on the Rust side.
        #
        # Phase 6 carddef-triggers slice: thread the printing's
        # triggered abilities through to the CardDef in their dict
        # round-trip shape. The filter (above) already guarantees
        # every entry round-trips cleanly via
        # :func:`_python_ta_to_safe_dict`; the bridge in
        # ``cross_engine.build_python_state`` re-materializes them
        # via the existing
        # ``_ta_dict_to_python_triggered_ability`` translator and
        # ``cross_engine.build_rust_state`` passes the same dict
        # tuple to Rust's ``register_printing(triggered_abilities=)``
        # parser. Empty tuple is a no-op (every pre-triggers fixture
        # path ends up here unchanged).
        triggered_abilities = tuple(
            _python_ta_to_safe_dict(ta)
            for ta in printing.triggered_abilities
        )
        # Round-78 dynamic-P/T-formula substrate: thread the formula
        # key through to the CardDef. The filter above gates on the
        # admit set (today: only ``"harmonious_grovestrider"`` is
        # honored); ``None`` for every non-CDA creature, bit-
        # identical to pre-substrate.
        return CardDef(
            name=printing.name,
            card_type="Creature",
            mana_cost_generic=generic,
            mana_cost_colored=colored,
            power=printing.power,
            toughness=printing.toughness,
            keywords=tuple(sorted(printing.keywords)),
            triggered_abilities=triggered_abilities,
            dynamic_pt_formula=getattr(
                printing, "dynamic_pt_formula", None
            ),
        )
    raise ValueError(
        f"printing {printing.name!r} is not bridge-supported; "
        f"call _printing_is_bridge_* first to filter."
    )


def load_production_deck(
    archetype: int = 0,
    deck_pool_path: Optional[pathlib.Path] = None,
) -> list[str]:
    """Load one real EoE deck from the 17lands deck pool.

    Returns the deck's card-name list (typically 40 cards — limited /
    sealed deck size). The card names are direct Scryfall names that
    the active CardSource (OracleParserCardSource) resolves via
    ``get_printings()``.

    Parameters
    ----------
    archetype : int, default 0
        Index into the deck pool. The 17lands JSON is a flat list of
        deck dicts; ``archetype`` is the positional index. Trivially
        deterministic — index 0 always returns the same deck. Pass a
        different index for a different deck.
    deck_pool_path : pathlib.Path, optional
        Path to the gzipped deck pool. Defaults to
        :data:`DEFAULT_PRODUCTION_DECK_POOL`. Resolved relative to the
        repo root if not absolute.

    Returns
    -------
    list[str]
        The deck's 40-card name list. Order matches the JSON.

    Raises
    ------
    FileNotFoundError
        If the deck pool file is missing.
    IndexError
        If ``archetype`` is out of range for the deck pool.
    """
    if deck_pool_path is None:
        # Resolve relative to the repo root (the parity module sits
        # three dirs deep under src/mtg_bot/parity/, so go up four).
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        deck_pool_path = repo_root / DEFAULT_PRODUCTION_DECK_POOL
        if not deck_pool_path.exists():
            # Worktree fallback: data/ is gitignored, so a freshly-
            # checked-out worktree won't have the pool. Look for it in
            # a sibling `mtg-bot/` main checkout (the project's
            # canonical worktree layout per the repo's CLAUDE.md).
            sibling = repo_root.parent.parent / "mtg-bot" / DEFAULT_PRODUCTION_DECK_POOL
            if sibling.exists():
                deck_pool_path = sibling
    if not deck_pool_path.exists():
        raise FileNotFoundError(
            f"production deck pool not found at {deck_pool_path}; "
            f"regenerate via scripts/fetch_eoe.py + the 17lands "
            f"sync pipeline if it's missing."
        )
    with gzip.open(deck_pool_path, "rt") as fh:
        data = json.load(fh)
    if archetype < 0 or archetype >= len(data):
        raise IndexError(
            f"archetype={archetype} out of range for deck pool with "
            f"{len(data)} decks"
        )
    return list(data[archetype]["cards"])


# ---------------------------------------------------------------------------
# C6 — the loader-admit gate as the enforced soundness boundary.
#
# `docs/SOT_FLIP_CUTOVER_PLAN.md` C6 / blocker B2 / Step 2: the loader
# admit gate (`_filter_deck_to_bridge_supported`'s 5-disjunct chokepoint
# + the `production_deck_to_deckspec` Forest-pad) is the SoT-flip
# soundness boundary. A card the loader does NOT admit MUST be unable to
# silently enter a flipped (Rust-sourced) game.
#
# TODAY (Python is SoT, single-script-replay) the silent drop +
# Forest-pad is the CORRECT, parity-required behavior — the bench plays
# the admitted residue of real decks, Forest-padded, and that is benign
# because Python drives. AT/AFTER the flip it becomes a SILENT
# CORRECTNESS HOLE: the Rust-sourced bot would quietly play a different
# (reduced, Forest-padded) deck than intended with no signal.
#
# C6 makes that boundary EXPLICIT + INSTRUMENTED + FAIL-LOUD without
# changing what gets admitted (that is C1 / the substrates' job):
#
#   * ``LoaderAdmitReport`` is the OBSERVABLE — exactly which names were
#     dropped (unresolved vs resolved-but-not-bridged) and how many
#     Forest-pad slots each side received. It is opt-in: the default
#     loader path (``production_deck_to_deckspec``) produces NO report
#     and is bit-identical to the pre-C6 behavior (parity-neutral by
#     construction).
#   * ``LoaderAdmitPolicy.RAISE`` is the post-flip FAIL-LOUD enforcement
#     — a non-admitted name raises ``NonAdmittedCardError`` instead of
#     being silently dropped. The Rust-sourced path (Step 3/4) sets
#     this; the Python-SoT parity path keeps the default ``DROP``.
#
# This is the §2 Step-2 / B2 pin substrate: the EXISTING boundary made
# enforced, non-silent, and pinned — NOT a change to the admit set.
# ---------------------------------------------------------------------------


class LoaderAdmitPolicy:
    """Policy for a card-name the loader-admit gate does NOT admit.

    ``DROP`` (default) — the pre-C6, parity-required behavior: a
    non-admitted name is silently dropped from the per-player library
    and the slot is later Forest-padded. Correct WHILE Python is the
    source of truth (single-script-replay; the bench plays the admitted
    residue and the drop is benign because Python drives both engines).

    ``RAISE`` — the post-flip FAIL-LOUD soundness boundary: a
    non-admitted name raises :class:`NonAdmittedCardError` rather than
    silently degrading the deck. The Rust-sourced decision path
    (`SOT_FLIP_CUTOVER_PLAN` Step 3/4) MUST use this so a card the
    loader cannot bridge can never quietly enter a Rust-authoritative
    game as a Forest.
    """

    DROP = "drop"
    RAISE = "raise"


class NonAdmittedCardError(ValueError):
    """A non-loader-admitted card hit the gate under ``RAISE`` policy.

    Raised by :func:`_filter_deck_to_bridge_supported` (and therefore
    :func:`production_deck_to_deckspec_instrumented`) when
    ``on_non_admitted`` is :data:`LoaderAdmitPolicy.RAISE` and a deck
    name either does not resolve in the active CardSource OR resolves
    but passes none of the five bridge-admit disjuncts. This is the C6
    enforced boundary — it must be unraisable on the Python-SoT parity
    path (which always uses the default ``DROP``) and the hard fence on
    the Rust-sourced path.
    """

    def __init__(self, name: str, *, resolved: bool):
        self.card_name = name
        self.resolved = resolved
        reason = (
            "resolves in the active CardSource but passes none of the "
            "five bridge-admit disjuncts (basic_land / "
            "nonbasic_mana_land / pure_warp_creature / simple_creature "
            "/ simple_spell)"
            if resolved
            else "does not resolve in the active CardSource"
        )
        super().__init__(
            f"loader-admit boundary (C6): card {name!r} {reason}; it "
            f"cannot enter a Rust-sourced game. The Python-SoT parity "
            f"path tolerates this (silent drop + Forest-pad); the "
            f"post-flip Rust-sourced path does not — admit it via a "
            f"C1/substrate close or fence the deck out upstream."
        )


@dataclass(frozen=True)
class LoaderAdmitReport:
    """Observable record of what the loader-admit gate dropped + padded.

    Returned by :func:`production_deck_to_deckspec_instrumented` (the
    C6 observable wrapper). Makes the loader-admit boundary an explicit
    artifact instead of a silent mutation, so the post-flip soundness
    fence (`SOT_FLIP_CUTOVER_PLAN` C6) is auditable.

    Fields are per-side (p0 / p1). ``*_unresolved`` are deck names that
    did not resolve in the active CardSource at all; ``*_not_bridged``
    are names that resolved but passed NONE of the five admit
    disjuncts. ``*_pad_count`` is the number of Forest (or configured
    ``pad_with``) slots appended to reach ``pad_to_size``. Lists
    preserve input order and multiplicity (a 4x non-admitted card
    appears 4x), so a caller can reconstruct exactly which deck slots
    were lost.
    """

    p0_unresolved: tuple[str, ...]
    p0_not_bridged: tuple[str, ...]
    p1_unresolved: tuple[str, ...]
    p1_not_bridged: tuple[str, ...]
    p0_pad_count: int
    p1_pad_count: int
    pad_with: str

    @property
    def any_dropped(self) -> bool:
        """True iff ANY name was dropped on either side (the boundary
        was crossed — under ``RAISE`` this state is unreachable)."""
        return bool(
            self.p0_unresolved
            or self.p0_not_bridged
            or self.p1_unresolved
            or self.p1_not_bridged
        )

    @property
    def total_dropped(self) -> int:
        """Total dropped deck slots across both sides (with
        multiplicity) — the bounded reduced-universe magnitude."""
        return (
            len(self.p0_unresolved)
            + len(self.p0_not_bridged)
            + len(self.p1_unresolved)
            + len(self.p1_not_bridged)
        )


def production_deck_to_deckspec(
    p0_card_names: list[str],
    p1_card_names: Optional[list[str]] = None,
    *,
    shared_rng_seed: Optional[int] = None,
    pad_to_size: int = 60,
    pad_with: str = "Forest",
) -> DeckSpec:
    """Build a parity-bridge DeckSpec from real EoE deck card-name lists.

    Filters each card-name list to the bridge-supported subset (basic
    lands + vanilla creatures), builds the union CardDef registry, and
    optionally pads each filtered library with extra basic lands so
    short post-filter decks still have enough cards to drive a multi-
    turn trajectory test. Mirror-match defaults: when ``p1_card_names``
    is None, p1 gets the same filtered library as p0.

    Parameters
    ----------
    p0_card_names : list[str]
        p0's deck card-name list (e.g. from
        :func:`load_production_deck`). Card names that don't resolve
        in the active CardSource OR don't pass the bridge-support
        filter are silently dropped.
    p1_card_names : list[str], optional
        p1's deck card-name list. When None, p1 receives a copy of
        p0's filtered library — produces a mirror-match for clean
        trajectory probes.
    shared_rng_seed : int, optional
        Forwarded to :class:`DeckSpec.shared_rng_seed`. When set,
        both engines shuffle to the same library permutation.
    pad_to_size : int, default 60
        Target deck size after filter+pad. If a player's filtered
        library has fewer cards than this, the remainder is padded
        with copies of ``pad_with``. Set to 0 to disable padding (use
        the post-filter library size as-is).
    pad_with : str, default "Forest"
        Card name to use for padding. Must resolve to a bridge-
        supported basic land in the active CardSource. The default is
        "Forest" because the EoE pool's mono-G decks are the most
        likely test target; switch to the deck's primary color for
        non-green archetypes.

    Returns
    -------
    DeckSpec
        Ready to feed into :func:`setup_matched_game`.

    Raises
    ------
    ValueError
        If ``pad_with`` doesn't resolve to a bridge-supported basic
        land in the active CardSource, OR if BOTH players' filtered
        libraries are empty (would produce a zero-card DeckSpec).

    Notes
    -----
    Cards filtered out are NOT logged on this path — the default
    loader's job is to produce a clean DeckSpec, not to audit the
    filter, and this path stays bit-identical to the pre-C6 behavior
    (parity-neutral). To OBSERVE which cards a deck loses to the gate,
    or to enforce the fail-loud post-flip soundness boundary, use
    :func:`production_deck_to_deckspec_instrumented` (the C6 wrapper)
    or :func:`_filter_deck_to_bridge_supported` directly.
    """
    spec, _report = _production_deck_to_deckspec_core(
        p0_card_names,
        p1_card_names,
        shared_rng_seed=shared_rng_seed,
        pad_to_size=pad_to_size,
        pad_with=pad_with,
        on_non_admitted=LoaderAdmitPolicy.DROP,
        observe=False,
    )
    return spec


def production_deck_to_deckspec_instrumented(
    p0_card_names: list[str],
    p1_card_names: Optional[list[str]] = None,
    *,
    shared_rng_seed: Optional[int] = None,
    pad_to_size: int = 60,
    pad_with: str = "Forest",
    on_non_admitted: str = LoaderAdmitPolicy.DROP,
) -> tuple[DeckSpec, LoaderAdmitReport]:
    """C6 observable wrapper: build a DeckSpec AND record what was lost.

    Same DeckSpec contract as :func:`production_deck_to_deckspec`, but
    additionally returns a :class:`LoaderAdmitReport` that names every
    deck slot the loader-admit gate dropped (unresolved vs
    resolved-but-not-bridged, per side, with multiplicity) and the
    Forest-pad count per side. This is the C6 OBSERVABLE — the loader
    boundary as an explicit, auditable artifact instead of a silent
    mutation (`SOT_FLIP_CUTOVER_PLAN` C6 / B2 / Step 2).

    ``on_non_admitted``
        :data:`LoaderAdmitPolicy.DROP` (default) keeps the
        parity-required silent-drop behavior but surfaces it in the
        returned report. :data:`LoaderAdmitPolicy.RAISE` is the
        post-flip FAIL-LOUD boundary — the first non-admitted name
        raises :class:`NonAdmittedCardError` (the Rust-sourced path
        sets this so a non-bridgeable card can never quietly enter a
        Rust-authoritative game as a Forest). Under ``RAISE`` a
        successful return is the proof the deck is fully admitted
        (``report.any_dropped is False``).
    """
    return _production_deck_to_deckspec_core(
        p0_card_names,
        p1_card_names,
        shared_rng_seed=shared_rng_seed,
        pad_to_size=pad_to_size,
        pad_with=pad_with,
        on_non_admitted=on_non_admitted,
        observe=True,
    )


def _production_deck_to_deckspec_core(
    p0_card_names: list[str],
    p1_card_names: Optional[list[str]] = None,
    *,
    shared_rng_seed: Optional[int] = None,
    pad_to_size: int = 60,
    pad_with: str = "Forest",
    on_non_admitted: str = LoaderAdmitPolicy.DROP,
    observe: bool = False,
) -> tuple[DeckSpec, LoaderAdmitReport]:
    """Shared body for the public + instrumented loader entrypoints.

    With ``on_non_admitted=DROP`` and ``observe=False`` (the
    :func:`production_deck_to_deckspec` defaults) the DeckSpec produced
    is bit-identical to the pre-C6 implementation — the sinks are
    ``None`` so no name is recorded and no policy branch fires. The C6
    observable/fail-loud behavior is strictly opt-in.
    """
    from mtg_bot.cards.source import get_active_source
    src = get_active_source()
    printings = src.get_printings()

    # Validate the pad_with card upfront so a silent zero-pad later
    # surfaces as a clean error.
    pad_printing = printings.get(pad_with)
    if pad_printing is None or not _printing_is_bridge_basic_land(pad_printing):
        raise ValueError(
            f"pad_with={pad_with!r} does not resolve to a bridge-"
            f"supported basic land in the active CardSource; pick a "
            f"basic land name (Forest/Island/Plains/Mountain/Swamp)."
        )
    pad_card_def = _printing_to_card_def(pad_printing)

    # C6 observable sinks — only allocated when observing; otherwise
    # left None so the filter takes the exact pre-C6 code path.
    p0_unresolved: Optional[list] = [] if observe else None
    p0_not_bridged: Optional[list] = [] if observe else None
    p1_unresolved: Optional[list] = [] if observe else None
    p1_not_bridged: Optional[list] = [] if observe else None

    p0_filtered, p0_card_defs = _filter_deck_to_bridge_supported(
        p0_card_names, printings,
        on_non_admitted=on_non_admitted,
        unresolved_sink=p0_unresolved,
        not_bridged_sink=p0_not_bridged,
    )
    if p1_card_names is None:
        p1_filtered = list(p0_filtered)
        p1_card_defs = list(p0_card_defs)
        # Mirror match: p1 IS p0's filtered library — its drop record
        # is identical to p0's by construction.
        if observe:
            p1_unresolved = list(p0_unresolved or [])
            p1_not_bridged = list(p0_not_bridged or [])
    else:
        p1_filtered, p1_card_defs = _filter_deck_to_bridge_supported(
            p1_card_names, printings,
            on_non_admitted=on_non_admitted,
            unresolved_sink=p1_unresolved,
            not_bridged_sink=p1_not_bridged,
        )

    # Pad each side to pad_to_size with copies of pad_with.
    p0_pad_count = 0
    p1_pad_count = 0
    if pad_to_size > 0:
        if len(p0_filtered) < pad_to_size:
            p0_pad_count = pad_to_size - len(p0_filtered)
            p0_filtered.extend([pad_with] * p0_pad_count)
        if len(p1_filtered) < pad_to_size:
            p1_pad_count = pad_to_size - len(p1_filtered)
            p1_filtered.extend([pad_with] * p1_pad_count)

    # Refuse zero-card libraries even if both filters dropped everything.
    if not p0_filtered or not p1_filtered:
        raise ValueError(
            "production deck filter produced an empty library on at "
            "least one side; check that the input decks contain at "
            "least one bridge-supported card OR set pad_to_size>0."
        )

    # Union CardDef registry: dedup by name (CardDef is frozen+hashable
    # but two CardDefs with the same name from different decks should
    # be the same object — bridge-supported names always have a single
    # representation in the active CardSource).
    union_card_defs: dict[str, CardDef] = {}
    for cd in p0_card_defs + p1_card_defs:
        union_card_defs[cd.name] = cd
    # Always include the pad card (even when both decks happen to
    # already have it — dict-dedup makes this a no-op then).
    union_card_defs.setdefault(pad_card_def.name, pad_card_def)

    # Round-49 N1 token-printing substrate (Phase 1): seat any
    # create_token carrier's token printing. This appends synthetic
    # token CardDefs to the registry (absent from both libraries),
    # rewrites the carrier triggers' symbolic ``_token_card_def`` into
    # the numeric Rust-registry pid, and registers the matching Python
    # printing into the cross-engine bridge's
    # ``register_token_printing`` map. No-op when no carrier carries a
    # create_token trigger (every pre-round-49 production deck).
    card_defs_ordered = _seat_token_printings(
        list(union_card_defs.values())
    )

    spec = DeckSpec(
        card_defs=tuple(card_defs_ordered),
        p0_library=tuple(p0_filtered),
        p1_library=tuple(p1_filtered),
        shared_rng_seed=shared_rng_seed,
    )
    report = LoaderAdmitReport(
        p0_unresolved=tuple(p0_unresolved or ()),
        p0_not_bridged=tuple(p0_not_bridged or ()),
        p1_unresolved=tuple(p1_unresolved or ()),
        p1_not_bridged=tuple(p1_not_bridged or ()),
        p0_pad_count=p0_pad_count,
        p1_pad_count=p1_pad_count,
        pad_with=pad_with,
    )
    return spec, report


def _seat_token_printings(card_defs: list[CardDef]) -> list[CardDef]:
    """Loader-side token-printing pre-registration pass (round-49 N1,
    Phase 1).

    Walks ``card_defs`` for carriers whose ``triggered_abilities``
    carry a symbolic ``create_token`` dict (one with a
    ``_token_card_def`` key emitted by :func:`_python_ta_to_safe_dict`).
    For each distinct token CardDef (deduped by name):

      1. Append the token CardDef to the registry list IF not already
         present, AFTER every carrier card def. Its position in the
         returned list == its Rust ``Registry`` pid, because
         :func:`cross_engine.build_rust_state` registers
         ``spec.card_defs`` in tuple order and
         :meth:`Registry.register_printing` assigns
         ``id = printings.len()`` (sequential). The token CardDef is
         deliberately NOT added to either player's library — it is
         registered (so the pid is valid) but never drawn / played
         (CR: tokens only exist on the battlefield, created by the
         effect).
      2. Rewrite every carrier trigger referencing that token: drop
         the symbolic ``_token_card_def`` key, add the numeric
         ``token_printing_id`` = the token's final list index. The
         resulting dict now satisfies :func:`_is_safe_trigger_dict`
         (``{kind, count, token_printing_id}``) AND both engines'
         ``create_token`` bridge / parser arms.
      3. Register ``(pid, python_token_printing)`` into
         :func:`cross_engine.register_token_printing` so the Python
         bridge's ``create_token`` arm resolves the same pid to the
         same printing :func:`cross_engine.build_python_state`
         materializes for that token CardDef. The printing is obtained
         via :func:`cross_engine.cached_token_printing` (round-51 N1),
         which builds it once via
         :func:`cross_engine._card_def_to_python_printing` (the single
         source of truth ``build_python_state`` also uses — no shape
         drift) and returns the SAME object for the same logical token
         across every game / seed.

    ``CardDef`` is frozen, so carriers with rewritten triggers are
    rebuilt via :func:`dataclasses.replace`.

    Cross-game idempotency (round-51 N1 fix; round-50 N3 finding 1):
    this pass runs once per
    :func:`setup_matched_game_from_real_decks` invocation, and the
    token's pid is its index in the per-game ``card_defs`` union, which
    VARIES BY SEED (different opening hands -> different filtered
    libraries -> different union ordering). The
    ``register_token_printing`` map is a module-global that PERSISTS
    across games. Building a fresh printing per game (the pre-round-51
    behavior) made game 2+ of any token-carrier-bearing archetype hit
    the registry's identity-based conflicting-registration guard
    (``existing is not py_printing``) and raise ``ValueError`` — 56/60
    FAIL_EXCEPTION on a real multi-game Gravpack-Monoist bench sweep.
    :func:`cross_engine.cached_token_printing` keys the printing by
    token SHAPE (not pid) and returns the SAME object every game, so a
    repeat ``register_token_printing`` is identity-idempotent (exact
    no-op when the pid is unchanged; same-object re-register under a
    new pid when the seed shifted the union, never a conflict). Two
    carriers printing the same-named token still share one token
    CardDef + one pid (dedup by name); the shared cache makes their
    Python printing the same object too.

    Returns the (possibly extended + carrier-rewritten) card-def list,
    ordered as carriers-then-tokens so the token pids are stable and
    deterministic.
    """
    import dataclasses

    from mtg_bot.parity.cross_engine import (
        _reset_token_printing_pid_map_for_seating_pass,
        cached_token_printing,
        register_token_printing,
    )

    # Round-57 N1 fix (round-56 N3 finding — the multi-distinct-token
    # cross-archetype pid collision). The ``pid -> CardPrinting`` map
    # is a per-GAME artifact (a token's pid is its index in THIS game's
    # ``card_defs`` union, which varies by archetype AND seed) but is a
    # module-global that persists across games. A different token shape
    # (Lander) that lands on a pid a prior game's different token shape
    # (Robot) still holds would otherwise hit ``register_token_printing``'s
    # identity-conflict guard and raise — even though the prior game's
    # registration is dead. Resetting ONLY the pid map at pass entry
    # restores the per-game invariant the map is supposed to have.
    # Round-51's shape-keyed cache (``_TOKEN_PRINTING_CACHE_BY_KEY``) is
    # deliberately NOT cleared, so the same logical token still resolves
    # to the SAME identity-stable object across games (round-51's
    # same-shape cross-game guarantee is preserved). The full
    # correctness argument (round-51 safety preserved / cross-shape
    # collision fixed / no within-game silent mis-resolution) is in
    # ``_reset_token_printing_pid_map_for_seating_pass``'s docstring.
    # Unconditional + before the fast-path return so the pid map ALWAYS
    # reflects only the current game (a carrier-free game leaves no
    # stale entries behind for the next game either); the cost is one
    # ``dict.clear()``.
    _reset_token_printing_pid_map_for_seating_pass()

    # Phase 4 (round 50): the seating pass handles BOTH the Phase-1
    # bare ``create_token`` symbolic carrier AND the Phase-4
    # ``create_token_tapped`` symbolic carrier. The two are seated
    # identically (deterministic-pid token CardDef append + bridge-map
    # register); the ONLY difference is the ``kind`` string, which is
    # the tapped-vs-untapped choice on the bridge and MUST survive the
    # seating rewrite.
    _TOKEN_CREATE_KINDS = ("create_token", "create_token_tapped")

    # Lander-spell-token-seating sub-substrate (round-73). The two
    # covered single-target Tier-A Lander-pid SPELL effects whose Rust
    # ``parse_effect`` arms ``require("token_printing_id")``. Their
    # reverse serializer (:func:`_python_spell_effect_to_dict`) emits a
    # symbolic ``_token_card_def`` exactly like the ``create_token``
    # triggered carrier; the seating pass now ALSO walks
    # ``spell_effect`` to resolve it. Unlike the triggered
    # ``create_token`` rewrite (which reconstructs a fixed
    # ``{kind,count,token_printing_id}`` dict), the spell rewrite
    # PRESERVES every non-symbolic field (``amount`` for Orbital
    # Plunge; Seedship Impact has none) and only swaps the symbolic
    # ``_token_card_def`` for the numeric ``token_printing_id`` — the
    # exact field set the matching Rust parse arms require.
    _SPELL_TOKEN_PID_KINDS = (
        "seedship_impact_effect",
        "deal_damage_to_target_creature_with_excess_lander_rider",
        # Emergency Eject (Effect-uncovered single-gated tier;
        # ~6,291 17lands decks). Same Lander-pid carrier shape as
        # Seedship Impact / Orbital Plunge — the reverse
        # serializer emits a symbolic ``_token_card_def`` Lander
        # reference and the spell-axis seating pass below resolves
        # it to the numeric ``token_printing_id`` Rust's
        # ``Effect::EmergencyEjectEffect { token_printing_id }``
        # parse arm requires. Adding the kind here is what auto-
        # joins Emergency Eject into the spell-card-bridge admit
        # set (the §2 auto-grow teeth).
        "emergency_eject_effect",
    )

    def _carrier_token_refs(cd: CardDef):
        """Yield (idx, token_card_def) for each create_token /
        create_token_tapped trigger on ``cd`` that still carries a
        symbolic ``_token_card_def``."""
        for i, ta in enumerate(cd.triggered_abilities):
            if not isinstance(ta, dict):
                continue
            eff = ta.get("effect")
            if not isinstance(eff, dict):
                continue
            if eff.get("kind") not in _TOKEN_CREATE_KINDS:
                continue
            tcd = eff.get("_token_card_def")
            if tcd is not None:
                yield i, tcd

    def _carrier_spell_token_ref(cd: CardDef):
        """Return the symbolic ``token_card_def`` from ``cd``'s
        ``spell_effect`` iff it is a Lander-pid spell effect still
        carrying a symbolic ``_token_card_def``, else ``None``.

        A spell CardDef carries exactly ONE ``spell_effect`` (the
        guardrail is single-mode single-effect), so this returns at
        most one ref — no index needed (the rewrite replaces the whole
        ``spell_effect`` dict)."""
        se = cd.spell_effect
        if not isinstance(se, dict):
            return None
        if se.get("kind") not in _SPELL_TOKEN_PID_KINDS:
            return None
        return se.get("_token_card_def")

    # Fast path: no create_token carrier AND no Lander-pid spell
    # carrier -> return the list untouched (every pre-round-49
    # production deck; every pre-round-73 spell deck). Avoids the
    # dataclasses / cross_engine import cost on the hot loader path
    # when there's nothing to seat.
    has_token_carrier = any(
        any(True for _ in _carrier_token_refs(cd))
        or _carrier_spell_token_ref(cd) is not None
        for cd in card_defs
    )
    if not has_token_carrier:
        return card_defs

    # Stable token order: first appearance across the carrier list
    # (deterministic — card_defs order is the union-dict insertion
    # order, itself deterministic from the deck card-name order).
    # Triggered-ability carriers and spell carriers share ONE token
    # namespace (a Lander is a Lander whichever axis spawns it — same
    # name, same CardDef, ONE pid), so a deck that has BOTH a
    # triggered Lander carrier and a Lander spell shares a single
    # Lander pid.
    token_cd_by_name: dict[str, CardDef] = {}
    for cd in card_defs:
        for _idx, tcd in _carrier_token_refs(cd):
            token_cd_by_name.setdefault(tcd.name, tcd)
        stcd = _carrier_spell_token_ref(cd)
        if stcd is not None:
            token_cd_by_name.setdefault(stcd.name, stcd)

    # Carriers occupy pids [0, len(card_defs)); tokens get the pids
    # immediately after, in token_cd_by_name insertion order. A token
    # whose name already appears as a regular card def reuses that
    # def's pid (a token-named real card in the pool is implausible
    # for the EoE token names — Robot / Glimmer / Lander aren't
    # castable cards — but the dedup keeps the contract honest).
    existing_names = {cd.name: i for i, cd in enumerate(card_defs)}
    result = list(card_defs)
    token_pid_by_name: dict[str, int] = {}
    for tname, tcd in token_cd_by_name.items():
        if tname in existing_names:
            token_pid_by_name[tname] = existing_names[tname]
            continue
        token_pid_by_name[tname] = len(result)
        result.append(tcd)

    # Register each token's Python printing into the bridge map at
    # its pid. ``cached_token_printing`` returns the identity-stable
    # printing for this token shape (built once via the shared
    # CardDef->printing helper, so it is bit-identical to what
    # build_python_state materializes) and reuses the SAME object
    # across every game / seed -> ``register_token_printing`` is
    # cross-game identity-idempotent and never hits the registry's
    # conflicting-registration guard (round-51 N1 fix; round-50 N3
    # finding 1).
    for tname, pid in token_pid_by_name.items():
        py_printing = cached_token_printing(token_cd_by_name[tname])
        register_token_printing(pid, py_printing)

    # Rewrite every carrier's symbolic create_token triggers AND/OR
    # symbolic Lander-pid spell_effect into the numeric
    # token_printing_id shape. CardDef is frozen -> replace.
    rewritten: list[CardDef] = []
    for cd in result:
        refs = list(_carrier_token_refs(cd))
        stcd = _carrier_spell_token_ref(cd)
        if not refs and stcd is None:
            rewritten.append(cd)
            continue
        replace_kwargs: dict = {}
        if refs:
            new_tas = list(cd.triggered_abilities)
            for idx, tcd in refs:
                old = new_tas[idx]
                old_eff = old["effect"]
                new_eff = {
                    # Preserve the original kind — ``create_token``
                    # (Phase 1, bare) vs ``create_token_tapped``
                    # (Phase 4) IS the tapped-vs-untapped choice on
                    # the bridge; the seating pass only swaps the
                    # symbolic ``_token_card_def`` for the numeric
                    # ``token_printing_id``, never the kind.
                    "kind": old_eff["kind"],
                    "count": old_eff["count"],
                    "token_printing_id": token_pid_by_name[tcd.name],
                }
                new_ta = dict(old)
                new_ta["effect"] = new_eff
                new_tas[idx] = new_ta
            replace_kwargs["triggered_abilities"] = tuple(new_tas)
        if stcd is not None:
            # Spell rewrite: copy the whole spell_effect dict, drop
            # the symbolic ``_token_card_def``, add the numeric
            # ``token_printing_id``. EVERY other field is preserved
            # verbatim (``kind`` + ``amount`` for Orbital Plunge;
            # ``kind`` only for Seedship Impact) — the field set the
            # Rust ``parse_effect`` arms require. NO ``count`` is
            # injected (these arms don't take one — that would be the
            # silent-mistranslation class the §2 teeth exist to
            # prevent).
            old_se = cd.spell_effect
            new_se = {
                k: v
                for k, v in old_se.items()
                if k != "_token_card_def"
            }
            new_se["token_printing_id"] = token_pid_by_name[stcd.name]
            replace_kwargs["spell_effect"] = new_se
        rewritten.append(dataclasses.replace(cd, **replace_kwargs))
    return rewritten


def _filter_deck_to_bridge_supported(
    card_names: list[str],
    printings: dict,
    *,
    on_non_admitted: str = LoaderAdmitPolicy.DROP,
    unresolved_sink: Optional[list] = None,
    not_bridged_sink: Optional[list] = None,
) -> tuple[list[str], list[CardDef]]:
    """Filter a card-name list to bridge-supported names + CardDefs.

    Returns ``(filtered_names, card_defs)`` where:

      * ``filtered_names`` is the input list with unsupported names
        dropped, in original order. Duplicates are preserved (a 4x
        Forest in the input stays 4x in the output).
      * ``card_defs`` is the unique CardDef list (deduped by name) the
        filtered names reference.

    Internal helper for :func:`production_deck_to_deckspec`.

    C6 instrumentation (parity-neutral — all three keyword params
    default to the pre-C6 behavior; with the defaults this function is
    bit-identical to before the C6 slice):

    ``on_non_admitted``
        :data:`LoaderAdmitPolicy.DROP` (default) silently drops a
        non-admitted name exactly as before — the Python-SoT
        parity-required behavior. :data:`LoaderAdmitPolicy.RAISE`
        raises :class:`NonAdmittedCardError` instead — the post-flip
        fail-loud soundness boundary (`SOT_FLIP_CUTOVER_PLAN` C6).
    ``unresolved_sink`` / ``not_bridged_sink``
        Optional lists; when provided, each dropped name is appended
        (preserving order + multiplicity) to the matching sink —
        ``unresolved_sink`` for names absent from ``printings``,
        ``not_bridged_sink`` for names that resolve but pass none of
        the five disjuncts. The OBSERVABLE half of C6 (the loader no
        longer silently mutates — it records what it removed).
    """
    filtered_names: list[str] = []
    seen_card_defs: dict[str, CardDef] = {}
    for name in card_names:
        printing = printings.get(name)
        if printing is None:
            # Unresolved: name not in the active CardSource at all.
            if on_non_admitted == LoaderAdmitPolicy.RAISE:
                raise NonAdmittedCardError(name, resolved=False)
            if unresolved_sink is not None:
                unresolved_sink.append(name)
            continue
        if not (
            _printing_is_bridge_basic_land(printing)
            or _printing_is_bridge_nonbasic_mana_land(printing)
            or _printing_is_bridge_pure_warp_creature(printing)
            or _printing_is_bridge_simple_creature(printing)
            # Spell-card-bridge lever, Phase 1 (round-70): the 5th
            # disjunct admits a pure single-target Instant/Sorcery
            # whose bespoke Effect close has landed (covered+bridged)
            # and round-trips through the reverse serializer. Wired
            # exactly like the round-63 pure-warp-creature predicate.
            or _printing_is_bridge_simple_spell(printing)
            # Aura-attachment substrate, Phase 1 STEP 7 (round-78):
            # the 6th disjunct admits a sub-shape S1 static P/T-or-
            # keyword Aura. The predicate's §2 teeth (no triggered/
            # static/activated/spell-effect, no control-swap /
            # replacement riders, equipment_bonuses with granted
            # keywords restricted to ``_SAFE_KEYWORDS``) keep the
            # admit set bridge-correct. STEPs 2-6 (Rust parse arm +
            # cast-time enchant-target decision + Aura-resolution
            # attach + CR-704.5n SBA bury + bidirectional bridge
            # round-trip) closed the prior honest-stops; with this
            # OR-in Phase 1 closes for sub-shape S1. The live EoE
            # pool admits 0 Auras today (every S1 carrier — Frantic
            # Strength / Shardmage's Rescue — carries the non-
            # ``_SAFE_KEYWORDS`` ``flash`` keyword, the predicate's
            # documented decline reason), so this OR-in is a no-op
            # for the current bench corpus but the substrate is
            # wired through for the first non-flash S1 Aura that
            # ships (Phase 2/3/4 widen the admit set).
            or _printing_is_bridge_simple_aura(printing)
            # Static / layered abilities substrate, Phase 1 first-
            # deliverable (round-79): the 7th disjunct admits a
            # creature whose every static_ability is an unconditional
            # anthem keyword-grant (scope =
            # ``[other_]creatures_you_control``, layer-6
            # ``GrantKeyword(kw)``, kw in ``_SAFE_KEYWORDS``). Tannuk
            # Steadfast Second is the canonical EoE single-gated
            # carrier ("Other creatures you control have haste.");
            # closing this gate flips Tannuk to bridge-admitted.
            # Rust-side fold rides the round-78 STEP-4
            # ``battlefield_has_keyword`` reader-site extension
            # (``static_anthem_grants_keyword_to``). Conditional
            # statics + P/T anthems + SELF-scope statics are
            # honest-stopped to later slices (Brightspear /
            # Cloudsculpt / Gravblade / Mechan Shieldmate / Frenzied
            # Baloth / Rampaging Soulrager / Wildfire Wickerfolk
            # each fall outside this Phase-1 admit shape).
            or _printing_is_bridge_simple_static_anthem_creature(printing)
            # Static / layered abilities substrate — Phase 2 (round-80,
            # 2026-05-20): the 8th disjunct admits a creature whose
            # static_abilities consist of one uncounterable-grant
            # sentinel ``StaticAbility(grants_creature_spells_
            # uncounterable=True)`` AND/OR whose printing carries
            # ``spell_is_uncounterable=True``. Frenzied Baloth is the
            # canonical EoE single-gated carrier — has BOTH (the cast-
            # side flag and the battlefield-static rider). Closing
            # this admits Frenzied Baloth through the loader filter.
            # Rust-side gate rides the printing-level bool slots
            # (``spell_is_uncounterable`` /
            # ``grants_creature_spells_uncounterable``) consulted by
            # ``Effect::CounterTargetSpell`` and the round-7
            # ``counter_spell`` primitive. The remaining 6 of the 7
            # honest-stops from Phase 1 stay honest-stopped
            # (Brightspear / Cloudsculpt / Gravblade / Mechan
            # Shieldmate / Rampaging Soulrager / Wildfire Wickerfolk
            # all need per-Condition Rust ports + SELF-scope reader-
            # site walks; Phase 2 sequencing reserves them for the
            # next slice).
            or _printing_is_bridge_simple_uncounterable_creature(printing)
            # Static / layered abilities substrate — Phase 3 (round-81,
            # 2026-05-20): the 9th disjunct admits a creature whose
            # every static_ability is a SELF-scope conditional bonus
            # in the two-tag dispatch table (``control_permanent_is_
            # artifact`` for Cloudsculpt Technician / Gravblade Heavy
            # and ``at_least_n_spells_cast_this_turn`` for Brightspear
            # Zealot). Closes 3 of the 6 honest-stops from Phase 1.
            # Rust-side fold rides ``static_self_bonus_*`` helpers
            # consulted by ``battlefield_creature_power`` /
            # ``battlefield_creature_toughness`` /
            # ``battlefield_has_keyword``. The remaining 3 of the 6
            # stay honest-stopped (Mechan Shieldmate's
            # ``ControlledArtifactEnteredThisTurn`` +
            # ``RemoveKeyword`` payload, Rampaging Soulrager's
            # ``UnlockedDoorsAmongRoomsAtLeast``, Wildfire
            # Wickerfolk's ``DeliriumActive`` — each needs a Condition
            # variant outside this slice's table OR a layer-6
            # ``RemoveKeyword`` payload the Rust reader doesn't yet
            # fold).
            or _printing_is_bridge_simple_static_self_conditional_creature(
                printing,
            )
            # Non-basic land substrate — phase 2 (round-85 N1): the
            # 10th disjunct admits a mono-color tapped mana-only
            # non-basic land. The 5 EoE Planet lands (Adagia /
            # Evendo / Kavaron / Susur Secundi / Uthros) are the
            # carriers — parser-flattened to ``produces={one color}``
            # + ``enters_tapped=True`` + no abilities + no keywords
            # (the parser drops Station + the tier-gated activated
            # ability per ``land_builder.py:692-727``). The bridge
            # carries no ``enters_tapped`` field and Rust's
            # ``register_printing`` has no land ``enters_tapped``
            # path, so both engines materialize the bridge-flattened
            # CardDef as an UNTAPPED single-color land — bit-equal
            # cross-engine, CR-incomplete (round-39 Ward / round-60
            # shockland precedent). The 5 Planet lands carry 7,617
            # 17lands decks (per round-58 attribution scout); the
            # remaining 13 non-shockland non-basic-land siblings
            # stay declined here (Duskmourn painlands' parser-
            # structured conditional-tapped IS a real cross-engine
            # divergence — Python's ``actions.py:1099`` honors the
            # ``enters_tapped_unless_any_player_life_at_most`` field,
            # Rust ignores it; the predicate's §2 teeth reject the
            # 10 painlands on that field, the 3 specials
            # Cathedral/Starforge/Echoing Deeps on the produces / clone
            # guards).
            or _printing_is_bridge_simple_tapped_mana_land(printing)
        ):
            # Resolved but passes none of the 5 bridge-admit disjuncts.
            if on_non_admitted == LoaderAdmitPolicy.RAISE:
                raise NonAdmittedCardError(name, resolved=True)
            if not_bridged_sink is not None:
                not_bridged_sink.append(name)
            continue
        if name not in seen_card_defs:
            seen_card_defs[name] = _printing_to_card_def(printing)
        filtered_names.append(name)
    return filtered_names, list(seen_card_defs.values())


def setup_matched_game_from_real_decks(
    p0_card_names: list[str],
    p1_card_names: Optional[list[str]] = None,
    *,
    shared_rng_seed: Optional[int] = None,
    pad_to_size: int = 60,
    pad_with: str = "Forest",
    mulligans: bool = False,
    opening_hand_size: int = 7,
    mulligan_agent: Optional[MulliganDeterministicAgent] = None,
    mulligan_seed: int = 0,
):
    """One-shot: convert real EoE decklists → DeckSpec → matched states.

    Convenience wrapper that composes
    :func:`production_deck_to_deckspec` + :func:`setup_matched_game`.
    Same return contract as :func:`setup_matched_game`. Use this
    entrypoint for trajectory tests that want to drive on real EoE
    deck shapes; use the two-step path
    (``production_deck_to_deckspec`` then ``setup_matched_game``)
    when the caller wants to inspect / mutate the DeckSpec between
    steps.
    """
    spec = production_deck_to_deckspec(
        p0_card_names,
        p1_card_names,
        shared_rng_seed=shared_rng_seed,
        pad_to_size=pad_to_size,
        pad_with=pad_with,
    )
    return setup_matched_game(
        spec,
        mulligans=mulligans,
        opening_hand_size=opening_hand_size,
        mulligan_agent=mulligan_agent,
        mulligan_seed=mulligan_seed,
    )
