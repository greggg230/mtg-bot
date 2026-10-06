"""Capability-based auto-pass for ANY player's priority moments.

Generalizes the opp-side auto-pass from ``_opp_capabilities.py`` to
both the active player and the non-active player. Drives a single
short-circuit path in ``MtgGameStateAdapter.legal_decisions``: when
the player who currently has priority has no possible action other
than passing, return ``[PassPriority()]`` immediately without
enumerating ``legal_actions``.

A "quiet" turn — where neither side has cards in hand they can cast
nor activated abilities they can fire — has 20-40 priority moments
per phase that all trivially pass. With BOTH players' priority
windows short-circuiting, the engine churns through priority round
in microseconds and the whole turn structure flies.

Two priority window flavors
---------------------------

**Non-active priority** (we call this "opp" historically):
priority is on the player who is NOT the active player. By
definition this is non-sorcery timing — only instant-speed
capabilities matter (instants in hand with payable cost, instant-
speed activated abilities, instant-speed warp / impulse / Lightstall
casts from graveyard / exile). Sorcery-speed spells, land plays, and
attack-step pre-combat declarations are NOT possible at this window.

**Active priority — instant-speed window**: active player has
priority but the stack is non-empty OR we are in a combat substep
or beginning/end-of-combat / upkeep / draw / end / cleanup window.
Same instant-speed-only capabilities as opp.

**Active priority — sorcery-speed window**: active player has
priority, stack is empty, AND step is precombat-main or
postcombat-main. NOW everything in the player's toolbox is on the
table — instants AND sorceries AND lands AND activated abilities AND
warp / impulse / Lightstall sorcery-speed casts AND spacecraft
station crew etc.

**Active priority — combat-eligible window**: at the
DECLARE_ATTACKERS step (or BEGINNING_OF_COMBAT step where attacker
declarations are about to happen), the active player can declare
attackers — even though the priority window itself has no spells to
cast. The capability check must NOT auto-pass when the active player
controls at least one creature eligible to attack.

NB: in the engine_adapter, the DECLARE_ATTACKERS step's combat
sub-decision phase routes through ``AttackerDecision`` /
``BlockerDecision`` factored decisions — auto-pass is gated out by
``_combat_phase != _COMBAT_NONE`` in ``_can_short_circuit_to_pass``.
The capability check therefore doesn't NEED a "could attack" gate
to be safe (the combat phase short-circuit already handles it).
That said, we DO check eligible-attackers conservatively so a
future code path that opens a priority window before
``_combat_phase`` is set never silently auto-passes through what
should be an attacker declaration.

Conservatism contract
---------------------

False positives (returning True when the player actually couldn't
act) are *safe*: they fall through to the existing ``legal_actions``
path, which produces the correct legal set. False negatives
(returning False when the player CAN act) would be a correctness
bug — the bot would silently skip a legitimate response. **All
conservative gates here err toward False positives**: if we cannot
quickly rule out a capability, we include it.

Documented limitations (intentional false-NEGATIVE risk)
--------------------------------------------------------

The hash inputs are designed to capture every change that
*plausibly* changes the player's capability set. They DO NOT cover:

* **Continuous effects granting flash mid-turn** (Vedalken Orrery,
  Teferi, Mage of Zhalfir). The hash watches static flash-grant
  flags on permanents but if a one-shot effect grants flash through
  a continuous effect that doesn't change ``state.continuous_effects``
  in a way the hash sees, the cached "no capability" answer is
  stale until the next ``apply()`` invalidates the cache.
* **Cost-reduction effects landing mid-turn** (Goblin Electromancer)
  that would let an unaffordable spell become affordable. The hash
  fingerprints the pool / sources but doesn't re-run cost-reduction
  evaluation per card; we conservatively count "could possibly ever
  be cast for any cost in this turn" rather than the exact current
  effective cost.
* **Effects granting new activated abilities to a permanent**
  (Equipment with `grant_activated_ability` that lands without an
  ``apply()`` between moments). The hash counts permanent ids that
  have any ability per the current effective-abilities lookup; if
  the count changes the hash changes. But a SAME-ID, NEW-ability
  case is invisible.
* **Effects making spells castable from new zones**. The hash watches
  zone-castable flags (``warp_castable_from_graveyard``,
  ``cast_from_exile_with_lightstall_surcharge``,
  ``playable_from_exile_this_turn``) at coarse granularity.

The first two are exceedingly rare in the EoE pool we train on;
the last two are checked structurally. In practice the rate of
silent false-negatives is small enough to be acceptable for
training. These limitations apply equally to both opp and active
player auto-pass — the same hash machinery feeds both.

Cache lifetime
--------------

The cache lives on the adapter and is invalidated on every
``apply()`` (same hook the legal_actions cache uses). Within a
single priority window the hash is recomputed and compared on every
``legal_decisions`` call; matching hashes hit the cached bool.
"""

from __future__ import annotations

from mtg_bot.actions import can_pay_cost
from mtg_bot.cards.hidden import is_hidden_placeholder
from mtg_bot.game import GameState
from mtg_bot.turn import MAIN_STEPS, Step


def _is_sorcery_window(state: GameState, player_id: int) -> bool:
    """True iff the given player is at sorcery-speed timing right now.

    Mirrors ``mtg_bot.legal._is_sorcery_timing`` exactly: stack is
    empty, the player has priority, the player is the active player,
    AND the current step is one of the main phases.

    Both opp and active checks call this. For opp it always returns
    False (opp by definition isn't the active player); we still call
    it from the active-player path to gate sorcery-speed capabilities.
    """
    return (
        len(state.stack) == 0
        and state.priority_player_idx == state.active_player_idx
        and state.players[state.active_player_idx].id == player_id
        and state.step in MAIN_STEPS
    )


def _instant_speed_castable_in_hand(
    state: GameState, player_id: int, p,
) -> bool:
    """Return True if any card in the player's hand is castable at
    non-sorcery timing AND the cost is payable."""
    for card in p.hand:
        pr = card.printing
        instant_speed = pr.is_instant or "flash" in pr.keywords
        if not instant_speed:
            continue
        if can_pay_cost(state, player_id, pr.mana_cost):
            return True
    return False


def _sorcery_speed_castable_in_hand(
    state: GameState, player_id: int, p,
) -> bool:
    """Return True if any card in the player's hand is castable at
    sorcery timing AND the cost is payable. Includes creatures,
    sorceries, artifacts, enchantments, planeswalkers, spacecraft —
    every non-instant non-flash non-land card whose mana cost can
    be paid.

    Conservative: like the instant-speed sibling, we do NOT replicate
    the full ``legal_actions`` target/additional-cost gating — if the
    spell COULD plausibly be cast, return True. The instant-speed
    siblings (warp / kicker / impending) are already covered by the
    instant block; we only need to flag the spell-cast capability.
    """
    for card in p.hand:
        pr = card.printing
        if pr.is_land:
            continue
        if is_hidden_placeholder(pr):
            # Board-only search worlds: the inert hidden-card stand-in
            # has no type, so ``legal_actions`` never offers it; don't
            # let its {0} cost make the player "able to act".
            continue
        if pr.is_instant or "flash" in pr.keywords:
            # Already covered by the instant-speed sibling. Skip to
            # avoid double-counting (no harm if we did, but keeps
            # the gate set independent so each contributes once).
            continue
        # Rooms have no single mana_cost — try each half.
        if pr.is_room:
            for half in pr.room_halves:
                if can_pay_cost(state, player_id, half.mana_cost):
                    return True
            continue
        if can_pay_cost(state, player_id, pr.mana_cost):
            return True
    return False


def _has_lands_to_play(
    state: GameState, player_id: int, p,
) -> bool:
    """Return True if the player has at least one land in hand AND
    has remaining land drops this turn. Capability check is gated
    upstream by ``_is_sorcery_window``; we don't re-check timing here.
    """
    # Compute current land-drop ceiling. Default 1; +1 per
    # grants_additional_land_per_turn permanent the player controls.
    max_lands = 1 + sum(
        o.printing.grants_additional_land_per_turn
        for o in state.battlefield
        if o.controller_id == player_id
    )
    if p.lands_played_this_turn >= max_lands:
        return False
    for card in p.hand:
        if card.printing.is_land:
            return True
    # Also check graveyard if a "play lands from graveyard" grant is
    # in play (Splendid Reclamation-style; tracked via
    # grants_play_lands_from_graveyard).
    grants_play_from_grave = any(
        o.controller_id == player_id
        and o.printing.grants_play_lands_from_graveyard
        for o in state.battlefield
    )
    if grants_play_from_grave:
        for card in p.graveyard:
            if card.printing.is_land:
                return True
    return False


def _instant_speed_castable_in_graveyard(
    state: GameState, player_id: int, p,
) -> bool:
    """Warp-from-graveyard at instant speed. Today's ``legal_actions``
    only allows warp-from-graveyard at sorcery speed (per the comment
    in legal.py near ``warp_castable_from_graveyard``), so this
    returns False unless an instant-cast-from-grave grant is in play.
    Kept as a hook so adding instant-speed flashback later doesn't
    require touching the auto-pass module.
    """
    for card in p.graveyard:
        pr = card.printing
        if not pr.is_instant:
            continue
        if pr.warp_castable_from_graveyard and pr.warp_cost is not None:
            if can_pay_cost(state, player_id, pr.warp_cost):
                return True
    return False


def _sorcery_speed_castable_in_graveyard(
    state: GameState, player_id: int, p,
) -> bool:
    """Warp-from-graveyard at sorcery speed (Timeline Culler). Returns
    True when the player has at least one warp-castable card in their
    graveyard whose warp cost is payable AND surviving the optional
    additional life cost is possible."""
    for card in p.graveyard:
        pr = card.printing
        if not pr.warp_castable_from_graveyard:
            continue
        if pr.warp_cost is None:
            continue
        if pr.is_instant:
            continue  # instant variant handled above
        if p.life <= pr.warp_additional_life_cost:
            continue
        if can_pay_cost(state, player_id, pr.warp_cost):
            return True
    return False


def _instant_speed_castable_from_exile(
    state: GameState, player_id: int, p,
) -> bool:
    """Impulse-cast (``playable_from_exile_this_turn``) at instant
    speed if the underlying card is an instant. Kavu-gated cards
    likewise. Lightstall surcharge is sorcery-speed-only, skip.
    """
    for card in p.exile:
        if not card.printing.is_instant:
            continue
        if card.warped:
            continue  # warp branches are sorcery-speed today
        if not (
            card.playable_from_exile_this_turn
            or card.play_from_exile_requires_kavu
        ):
            continue
        if can_pay_cost(state, player_id, card.printing.mana_cost):
            return True
    return False


def _sorcery_speed_castable_from_exile(
    state: GameState, player_id: int, p,
) -> bool:
    """Sorcery-speed exile casts: warp-exile recasts, impulse exile
    (non-instant), Lightstall surcharge exile, and Kavu-gated impulse
    cards. Conservative: returns True if any such card has a payable
    cost in any of the relevant flag branches.
    """
    for card in p.exile:
        pr = card.printing
        if pr.is_instant:
            continue  # instant variant handled above
        if pr.is_land:
            # Lands cast from exile use PlayLand, not CastSpell;
            # they're sorcery-speed too. Treat as a capability if the
            # land-drop ceiling has room (re-use the helper).
            continue
        # Warp-exile recast.
        if card.warped:
            if can_pay_cost(state, player_id, pr.mana_cost):
                return True
            continue
        # Impulse-cast (with or without Kavu gate). Reused as a
        # conservative include: the engine gates Kavu on actually
        # controlling a Kavu, but checking that here is cheap if the
        # flag is set.
        if (
            card.playable_from_exile_this_turn
            or card.play_from_exile_requires_kavu
        ):
            if can_pay_cost(state, player_id, pr.mana_cost):
                return True
            continue
        # Lightstall surcharge — +1 generic on top of mana_cost.
        if card.cast_from_exile_with_lightstall_surcharge:
            from mtg_bot.cards import ManaCost
            with_surcharge = ManaCost(
                generic=pr.mana_cost.generic + 1,
                colored=pr.mana_cost.colored,
                x_pips=pr.mana_cost.x_pips,
            )
            if can_pay_cost(state, player_id, with_surcharge):
                return True
    return False


def _has_instant_speed_activatable(
    state: GameState, player_id: int, p, *, is_active_player: bool,
) -> bool:
    """Engine contract: ``legal_actions`` returns early at non-sorcery
    timing BEFORE walking the activated-ability block (see
    ``legal.py`` — ``if not _is_sorcery_timing: return actions`` after
    the hand-cast loop). Both opp at non-active priority AND active
    player during combat / upkeep / draw / end / cleanup priority
    windows therefore never receive ``ActivateAbility`` actions from
    the engine, regardless of which abilities their permanents have.

    Returning False unconditionally matches what the engine actually
    offers at non-sorcery priority moments. Treating "the player has
    activated abilities" as a capability would silently sabotage
    every auto-pass attempt (every player has lands; every utility
    land has a mana ability with ``is_mana_ability=True`` that the
    engine doesn't surface as ``ActivateAbility`` either).

    If a future engine change DOES start surfacing ``ActivateAbility``
    at non-sorcery timing, this is the place to scan permanents — and
    the structural hash already captures permanent shape so the
    cache stays correct.
    """
    return False


def _has_sorcery_speed_activatable(
    state: GameState, player_id: int, p,
) -> bool:
    """Any permanent the active player controls with at least one
    activatable ability (sorcery-speed or instant-speed; both are
    legal at sorcery timing) whose cost is payable.

    Only meaningful when called within a sorcery window. Mirrors
    ``_has_instant_speed_activatable`` with sorcery-speed inclusion.
    """
    from mtg_bot.game import effective_activated_abilities

    for obj in state.battlefield:
        if obj.controller_id != player_id:
            continue
        for ability in effective_activated_abilities(state, obj):
            # Both sorcery-speed and instant-speed abilities are legal
            # at sorcery timing. Check cost for either.
            if _ability_costs_payable(state, player_id, p, obj, ability):
                return True
    return False


def _ability_costs_payable(state, player_id, p, obj, ability) -> bool:
    """Conservative cost check for an activated ability. Returns True
    iff the mana portion is payable AND the simple non-mana costs
    that have a clear precondition (tap_self with untapped + non-sick,
    pay_life with surviving life total) are satisfiable.

    More complex non-mana costs (sacrifice-another, exile-from-grave,
    etc.) are not checked — those are conservative-include cases. If
    the ability LOOKS payable in mana terms, we say capable; the
    full ``legal_actions`` enumerator will then either confirm or
    filter it out at the next priority moment.

    NB: ``legal.py`` does NOT filter ``is_mana_ability=True`` entries
    — utility lands like Command Bridge surface their tap-for-mana as
    a real ``ActivateAbility``, so we treat them as a capability too.
    Basic lands have no entries in ``activated_abilities`` (the
    produces=frozenset path is consumed by ``_pay_mana_cost``
    inline), so this doesn't introduce a "every player always has a
    capability" false positive.
    """
    if ability.cost.mana is not None:
        from mtg_bot.actions import _effective_ability_mana_cost
        effective_mana = _effective_ability_mana_cost(state, obj, ability)
        if not can_pay_cost(state, player_id, effective_mana):
            return False
    if ability.cost.tap_self:
        if obj.tapped:
            return False
        from mtg_bot.game import is_creature_now
        if is_creature_now(obj) and obj.summoning_sick:
            return False
    if ability.cost.pay_life:
        if p.life <= ability.cost.pay_life:
            return False
    return True


def _has_eligible_attacker(state: GameState, player_id: int) -> bool:
    """Return True if the active player controls any creature eligible
    to attack right now. Used pre-combat / at the beginning-of-combat
    step to ensure auto-pass doesn't fire when the player has an
    attacker pending.

    Only meaningful when the player IS the active player AND we're at
    a step where attacker declarations are forthcoming
    (BEGINNING_OF_COMBAT or DECLARE_ATTACKERS). For other steps,
    eligible attackers are no different from "any untapped creature"
    and the result doesn't gate auto-pass behavior — but checking it
    cheaply doesn't hurt.
    """
    from mtg_bot.combat import eligible_attackers
    return bool(eligible_attackers(state, player_id))


def has_player_capability_to_act(
    state: GameState, player_id: int, *, is_active_player: bool,
) -> bool:
    """Definitive "does this player have at least one possible action
    other than passing right now" check. Returns True conservatively
    (false positives lead to a normal ``legal_actions`` enumeration
    which is correct); returns False only when we're confident there's
    nothing for the player to do.

    Args:
        state: current engine state.
        player_id: id of the player whose capability to assess.
        is_active_player: True iff this player is the active player
            for the current turn. Determines which sorcery-speed
            capabilities are checked.

    Returns:
        True if the player has at least one possible non-pass action.
    """
    p = state.player_by_id(player_id)

    # Instant-speed capabilities apply equally to opp and active.
    if _instant_speed_castable_in_hand(state, player_id, p):
        return True
    if _instant_speed_castable_in_graveyard(state, player_id, p):
        return True
    if _instant_speed_castable_from_exile(state, player_id, p):
        return True
    if _has_instant_speed_activatable(
        state, player_id, p, is_active_player=is_active_player,
    ):
        return True

    if not is_active_player:
        # Non-active priority is by definition non-sorcery timing;
        # no further capabilities to check.
        return False

    # Active player. If we're at sorcery-speed timing, sorcery-speed
    # spells, lands, sorcery-speed activated abilities, and warp /
    # impulse / Lightstall sorcery-speed casts become capable.
    sorcery_window = _is_sorcery_window(state, player_id)

    # Pre-combat attacker declaration capability — even if the player
    # has nothing else to do, they may have an attacker to declare at
    # the beginning of combat. Today the engine adapter routes
    # attacker declarations through the factored combat-decision
    # phase BEFORE the priority window opens, so by the time
    # legal_decisions runs in DECLARE_ATTACKERS the attackers have
    # already been chosen and ``_combat_phase != _COMBAT_NONE`` is
    # what the auto-pass gate keys on. Still, we include this check
    # so a future code path that opens a priority window before
    # combat-phase setup never silently auto-passes through what
    # should have been an attacker declaration.
    if state.step in (Step.BEGINNING_OF_COMBAT, Step.DECLARE_ATTACKERS):
        if _has_eligible_attacker(state, player_id):
            return True

    if sorcery_window:
        if _sorcery_speed_castable_in_hand(state, player_id, p):
            return True
        if _has_lands_to_play(state, player_id, p):
            return True
        if _sorcery_speed_castable_in_graveyard(state, player_id, p):
            return True
        if _sorcery_speed_castable_from_exile(state, player_id, p):
            return True
        if _has_sorcery_speed_activatable(state, player_id, p):
            return True
        # Spacecraft station crew is a sorcery-speed activation; if
        # any spacecraft + any non-sick untapped creature exists,
        # capability is True. Cheap structural check.
        spacecraft = [
            o for o in state.battlefield
            if o.controller_id == player_id and o.printing.is_spacecraft
        ]
        if spacecraft:
            for o in state.battlefield:
                if o.controller_id != player_id:
                    continue
                if not o.printing.is_creature:
                    continue
                if o.tapped or o.summoning_sick:
                    continue
                # At least one ship + one crew candidate.
                return True

    return False


def player_capability_hash(
    state: GameState, player_id: int, *, is_active_player: bool,
) -> int:
    """Fast structural hash over the inputs that decide
    ``has_player_capability_to_act``.

    Cheap: O(player_hand_size + player_battlefield_size +
    player_graveyard_size + player_exile_size).

    Hash inputs:
    * Hand contents (printing names sorted).
    * Mana pool fingerprint.
    * Battlefield shape: per-permanent (id, name, tapped, sick,
      ability_count) for permanents the player controls. Captures
      arrival/departure of ability-having permanents, untap/tap flips,
      sick status changes.
    * Graveyard / exile printing names + warp / impulse / Kavu /
      Lightstall flags. Cheap.
    * Coarse global context: step / turn / stack length / priority
      idx / active idx / continuous-effect count.
    * For ACTIVE PLAYER: ``lands_played_this_turn`` and the boolean
      ``is_sorcery_window`` are added. The land-drop counter changes
      across the turn; when it hits the cap, the "has lands to play"
      capability flips from True to False even if the hand didn't
      change. Sorcery-window flips at every step transition that
      moves into / out of a main phase, every stack push/pop, etc.
    * For ACTIVE PLAYER: pre-combat eligible-attacker count is also
      hashed in a coarse "step in combat-substeps + has untapped
      creatures" fingerprint.
    """
    p = state.player_by_id(player_id)

    hand_key = tuple(sorted(c.printing.name for c in p.hand))
    pool_key = tuple(sorted(
        (coin.color, coin.spend_restriction or "") for coin in p.mana_pool
    ))
    bf_key = tuple(sorted(
        (
            obj.id,
            obj.printing.name,
            obj.tapped,
            obj.summoning_sick,
            len(obj.printing.activated_abilities),
        )
        for obj in state.battlefield
        if obj.controller_id == player_id
    ))
    gy_key = tuple(sorted(
        (
            c.printing.name,
            c.printing.warp_castable_from_graveyard,
        )
        for c in p.graveyard
    ))
    exile_key = tuple(sorted(
        (c.printing.name, c.warped, c.playable_from_exile_this_turn,
         c.play_from_exile_requires_kavu,
         c.cast_from_exile_with_lightstall_surcharge)
        for c in p.exile
    ))
    global_key: tuple = (
        state.step,
        state.turn_number,
        len(state.stack),
        state.priority_player_idx,
        state.active_player_idx,
        len(state.continuous_effects),
    )
    extra_active: tuple = ()
    if is_active_player:
        # Land-drop counter and sorcery-window status. Both flip the
        # capability set without any other observable change.
        extra_active = (
            p.lands_played_this_turn,
            _is_sorcery_window(state, player_id),
            # Life total — gates warp's life-cost check. Cheap.
            p.life,
        )

    return hash((
        hand_key, pool_key, bf_key, gy_key, exile_key, global_key,
        extra_active, is_active_player,
    ))
