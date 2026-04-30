"""Condition grammar for cost-reduction rules.

A ``Condition`` is a structural predicate over the ``GameState`` viewed
from a particular caster's perspective. Conditions replace the historic
opaque-string ``CostReduction.condition`` keys: instead of a closed
whitelist of engine keys mapped to a ``_evaluate_cost_reduction`` ``elif``
cascade, each rule carries a tree of ``Condition`` objects whose
``matches(state, viewer_id)`` evaluates the gate directly.

Composition is the same ``& | ~`` pattern used by the predicate stack;
``AndCondition`` / ``OrCondition`` / ``NotCondition`` are frozen
dataclasses, so trees are value-equal for free.

Variable-amount conditions (Affinity, per-differently-named-land) override
``multiplier(state, viewer_id) -> int`` to return the count; the engine
multiplies ``rule.amount`` by ``condition.multiplier(...)`` whenever
``matches`` is true, replacing the old ``per_land_group: bool`` flag on
``CostReduction``.

Some conditions evaluate against the spell's chosen targets rather than
the static state — Dark Endurance's "if it targets a blocking creature"
gate. Those conditions take an optional ``chosen_targets`` argument on
``matches``; when ``None`` (the affordability-probe path through
``legal_actions`` before the caster has committed to a target), the
condition evaluates optimistically against the battlefield (does any
creature exist that the caster COULD pick?).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mtg_bot.effects.object_predicates import ObjectPredicate

if TYPE_CHECKING:
    from mtg_bot.game import GameState


# Targets are passed through the cast pipeline as ``(kind, id)`` tuples.
# Conditions that consult them re-shape their evaluation around the
# committed target; conditions that don't, ignore the parameter.
ChosenTargets = "tuple[tuple[str, int], ...] | None"


class Condition(ABC):
    """A boolean over ``(GameState, viewer_id, chosen_targets)``.

    Subclasses implement :meth:`matches`. Combine with ``&`` / ``|`` /
    ``~``; the operators return ``AndCondition`` / ``OrCondition`` /
    ``NotCondition`` instances that delegate to their operands.

    Variable-amount conditions (``PerDifferentlyNamedLand``,
    ``PerControlled``) override :meth:`multiplier` to return the
    multiplicand. Boolean-only conditions inherit the default
    ``multiplier`` of ``1``.
    """

    @abstractmethod
    def matches(
        self,
        state: "GameState",
        viewer_id: int,
        chosen_targets: ChosenTargets = None,
    ) -> bool:
        ...

    def multiplier(
        self,
        state: "GameState",
        viewer_id: int,
        chosen_targets: ChosenTargets = None,
    ) -> int:
        return 1

    def __and__(self, other: "Condition") -> "Condition":
        return AndCondition(self, other)

    def __or__(self, other: "Condition") -> "Condition":
        return OrCondition(self, other)

    def __invert__(self) -> "Condition":
        return NotCondition(self)


# ---------- boolean combinators ----------


@dataclass(frozen=True)
class AndCondition(Condition):
    left: Condition
    right: Condition

    def matches(self, state, viewer_id, chosen_targets=None):
        return (
            self.left.matches(state, viewer_id, chosen_targets)
            and self.right.matches(state, viewer_id, chosen_targets)
        )

    def multiplier(self, state, viewer_id, chosen_targets=None):
        # AND of two variable-amount conditions multiplies their factors.
        # In practice no card today carries a compound variable-amount
        # condition, but the algebra is the well-defined product so the
        # combinator stays composable rather than picking one side.
        return (
            self.left.multiplier(state, viewer_id, chosen_targets)
            * self.right.multiplier(state, viewer_id, chosen_targets)
        )


@dataclass(frozen=True)
class OrCondition(Condition):
    left: Condition
    right: Condition

    def matches(self, state, viewer_id, chosen_targets=None):
        return (
            self.left.matches(state, viewer_id, chosen_targets)
            or self.right.matches(state, viewer_id, chosen_targets)
        )


@dataclass(frozen=True)
class NotCondition(Condition):
    inner: Condition

    def matches(self, state, viewer_id, chosen_targets=None):
        return not self.inner.matches(state, viewer_id, chosen_targets)


# ---------- permanent-counting variants ----------


@dataclass(frozen=True)
class ControlPermanent(Condition):
    """``viewer_id`` controls at least one permanent matching ``filter``.

    Iterates the battlefield filtered to the viewer's permanents and
    tests each with ``filter.test(perm, state)``. The filter is an
    intrinsic ``ObjectPredicate`` — controller-scope is supplied by the
    iteration, so the predicate doesn't need its own controller check.

    Models the Starport Security activated-ability rider "if you control
    a creature with a +1/+1 counter on it" via
    ``ControlPermanent(IsCreature() & HasCounter("+1/+1"))``.
    """

    filter: ObjectPredicate

    def matches(self, state, viewer_id, chosen_targets=None):
        for obj in state.battlefield:
            if obj.controller_id != viewer_id:
                continue
            if self.filter.test(obj, state):
                return True
        return False


@dataclass(frozen=True)
class OpponentsControlAtLeast(Condition):
    """Some opponent of ``viewer_id`` controls at least ``n`` total
    permanents matching ``filter`` (summed across all opponents).

    Models "your opponents control three or more creatures" — counts
    every permanent matching ``filter`` whose ``controller_id`` is NOT
    ``viewer_id`` and tests the sum against the threshold.
    """

    filter: ObjectPredicate
    n: int

    def matches(self, state, viewer_id, chosen_targets=None):
        count = 0
        for obj in state.battlefield:
            if obj.controller_id == viewer_id:
                continue
            if self.filter.test(obj, state):
                count += 1
                if count >= self.n:
                    return True
        return False


# ---------- turn-state variants ----------


@dataclass(frozen=True)
class AnotherSpellCastThisTurn(Condition):
    """The viewer has cast at least one other spell this turn.

    Reads ``Player.spells_cast_this_turn`` pre-increment of the spell
    being evaluated — so "another" means strictly previous casts in
    the same turn, not including the spell whose cost is being computed.
    """

    def matches(self, state, viewer_id, chosen_targets=None):
        return state.player_by_id(viewer_id).spells_cast_this_turn >= 1


@dataclass(frozen=True)
class NthSpellThisTurn(Condition):
    """The viewer is about to cast their ``n``-th spell this turn —
    ``Player.spells_cast_this_turn`` is currently ``n - 1`` (the cast
    pipeline increments AFTER cost is paid).

    Models Uthros Psionicist's "the second spell you cast each turn
    costs {N} less to cast" via ``NthSpellThisTurn(2)``.
    """

    n: int

    def matches(self, state, viewer_id, chosen_targets=None):
        return state.player_by_id(viewer_id).spells_cast_this_turn == (
            self.n - 1
        )


@dataclass(frozen=True)
class AttackedWithThisTurn(Condition):
    """The viewer attacked with at least one permanent matching
    ``filter`` this turn.

    Reads the generalized ``Player.attacked_this_turn`` latch — a list
    of every permanent that was declared as an attacker by the viewer
    this turn (set in ``combat.declare_attackers``, cleared at the
    cleanup step in ``turn.run_cleanup_step``). Filters the latch's
    members by ``filter.test(perm, state)``; matches if any member
    qualifies.

    Models Thaumaton Torpedo's "if you attacked with a Spacecraft this
    turn" via ``AttackedWithThisTurn(HasSubtype("Spacecraft"))``.
    """

    filter: ObjectPredicate

    def matches(self, state, viewer_id, chosen_targets=None):
        player = state.player_by_id(viewer_id)
        for snapshot in getattr(player, "attacked_this_turn", ()):
            if self.filter.test(snapshot, state):
                return True
        return False


@dataclass(frozen=True)
class DuringYourTurn(Condition):
    """``viewer_id`` is the active player.

    Models Mental Modulation's "...less to cast during your turn." rider.
    Reads ``state.active_player()`` rather than walking phases — the
    discount is independent of which step or phase the spell is being
    cast in.
    """

    def matches(self, state, viewer_id, chosen_targets=None):
        return state.active_player().id == viewer_id


@dataclass(frozen=True)
class VoidActiveCondition(Condition):
    """The Void mechanic is active right now.

    Mirrors :class:`mtg_bot.effects.state_predicates.VoidActive` — Void
    is global, viewer-independent. Models the Temporal Intervention
    "Void — This spell costs {N} less to cast..." rider.
    """

    def matches(self, state, viewer_id, chosen_targets=None):
        from mtg_bot.game import is_void_active

        return is_void_active(state)


# ---------- target-dependent variants ----------


@dataclass(frozen=True)
class TargetMatches(Condition):
    """The spell's chosen target is a permanent matching ``filter``.

    Two evaluation modes:

    - **Cast-pipeline path** (``chosen_targets`` is not None): walks the
      committed targets, finds the ``("creature", id)`` entries, looks
      each one up on the battlefield, and tests ``filter.test(obj,
      state)``. Matches if any committed target qualifies.

    - **Affordability-probe path** (``chosen_targets is None``):
      ``legal_actions`` calls ``apply_cost_reductions`` BEFORE the
      caster has picked a target, so the gate evaluates optimistically:
      "does ANY creature on the battlefield satisfy ``filter``?" The
      probe answers "yes the discount could apply" so the spell isn't
      rejected as unaffordable. The actual cast pipeline re-checks
      with the committed targets and only sticks the discount if the
      caster commits to a qualifying target.

    Models Dark Endurance ("if it targets a blocking creature") via
    ``TargetMatches(IsCreature() & IsBlocking())`` and Seized from
    Slumber ("if it targets a tapped creature") via
    ``TargetMatches(IsCreature() & IsTapped())``.
    """

    filter: ObjectPredicate

    def matches(self, state, viewer_id, chosen_targets=None):
        if chosen_targets is None:
            # Optimistic affordability probe — does ANY battlefield
            # creature satisfy the filter?
            for obj in state.battlefield:
                if self.filter.test(obj, state):
                    return True
            return False
        for kind, tid in chosen_targets:
            if kind != "creature":
                continue
            obj = next(
                (o for o in state.battlefield if o.id == tid), None,
            )
            if obj is None:
                continue
            if self.filter.test(obj, state):
                return True
        return False


# ---------- variable-amount (multiplier) variants ----------


@dataclass(frozen=True)
class PerDifferentlyNamedLand(Condition):
    """Variable-amount: count distinctly-named lands the viewer
    controls. ``matches`` is true iff the count is at least 1;
    ``multiplier`` returns the actual count.

    Models Fungal Colossus's "This spell costs {X} less to cast,
    where X is the number of differently named lands you control."

    The historic engine key ``per_differently_named_land`` paired with
    ``per_land_group=True`` on ``CostReduction``; this condition
    subsumes both — the unit ``amount`` is multiplied by
    ``multiplier()`` directly in ``_evaluate_cost_reduction``, so the
    flag is redundant.
    """

    def _names(self, state, viewer_id) -> set[str]:
        names: set[str] = set()
        for o in state.battlefield:
            if o.controller_id == viewer_id and o.printing.is_land:
                names.add(o.printing.name)
        return names

    def matches(self, state, viewer_id, chosen_targets=None):
        return len(self._names(state, viewer_id)) >= 1

    def multiplier(self, state, viewer_id, chosen_targets=None):
        return len(self._names(state, viewer_id))


@dataclass(frozen=True)
class PerControlled(Condition):
    """Variable-amount: count permanents the viewer controls matching
    ``filter``. ``matches`` is true iff the count is at least 1;
    ``multiplier`` returns the actual count.

    Models the Affinity-style scalings:

    - Thrumming Hivepool's "Affinity for Slivers (this spell costs
      {1} less to cast for each Sliver you control)" via
      ``PerControlled(HasSubtype("Sliver"))``.
    - Sami, Wildcat Captain's external "Spells you cast have affinity
      for artifacts. (They cost {1} less to cast for each artifact
      you control.)" via ``PerControlled(IsArtifact())``.
    """

    filter: ObjectPredicate

    def _count(self, state, viewer_id) -> int:
        return sum(
            1 for o in state.battlefield
            if o.controller_id == viewer_id
            and self.filter.test(o, state)
        )

    def matches(self, state, viewer_id, chosen_targets=None):
        return self._count(state, viewer_id) >= 1

    def multiplier(self, state, viewer_id, chosen_targets=None):
        return self._count(state, viewer_id)


# ---------- continuous-effect gate variants ----------


@dataclass(frozen=True)
class ControlledArtifactEnteredThisTurn(Condition):
    """An artifact entered the battlefield under ``viewer_id``'s control
    this turn.

    Reads ``Player.artifacts_entered_under_your_control_this_turn``,
    which ``actions.put_into_battlefield`` increments whenever an
    artifact (including artifact creatures and tokens) enters under
    the player's control and resets at the cleanup step.

    Models Mechan Shieldmate's "as long as an artifact entered the
    battlefield under your control this turn" gate on its defender-
    removing ``ContinuousModifier``.
    """

    def matches(self, state, viewer_id, chosen_targets=None):
        return (
            state.player_by_id(viewer_id)
            .artifacts_entered_under_your_control_this_turn
            > 0
        )


@dataclass(frozen=True)
class AtLeastNSpellsCastThisTurn(Condition):
    """The viewer has cast at least ``n`` spells this turn.

    Reads ``Player.spells_cast_this_turn >= n``. Distinct from
    ``NthSpellThisTurn`` (which checks equality at cast time) and
    ``AnotherSpellCastThisTurn`` (which checks strictly positive):
    this variant is a ``>=`` threshold gate evaluated after the fact,
    suitable for static continuous effects whose condition is
    "you have cast N or more spells this turn."

    Models Illvoi Infiltrator's "if you've cast two or more spells
    this turn" gate on its unblockable ``ContinuousModifier`` via
    ``AtLeastNSpellsCastThisTurn(2)``.
    """

    n: int

    def matches(self, state, viewer_id, chosen_targets=None):
        return state.player_by_id(viewer_id).spells_cast_this_turn >= self.n
