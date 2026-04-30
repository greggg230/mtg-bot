"""Combat: eligibility, declaration, and simultaneous damage."""

from __future__ import annotations

from mtg_bot.game import (
    GameObject,
    GameState,
    effective_keywords,
    effective_power,
    effective_toughness,
    is_creature_now,
)


def eligible_attackers(state: GameState, player_id: int) -> list[GameObject]:
    return [
        o for o in state.battlefield
        if o.controller_id == player_id
        and is_creature_now(o)
        and not o.tapped
        and (not o.summoning_sick or "haste" in effective_keywords(state, o))
        and "defender" not in effective_keywords(state, o)
    ]


def eligible_blockers(state: GameState, player_id: int) -> list[GameObject]:
    return [
        o for o in state.battlefield
        if o.controller_id == player_id
        and is_creature_now(o)
        and not o.tapped
    ]


def can_block(
    state: GameState, blocker: GameObject, attacker: GameObject
) -> bool:
    """Return True if ``blocker`` may legally declare a block against ``attacker``.

    Flying: a creature with flying can only be blocked by creatures with flying
    or reach.

    Unblockable: an attacker with "unblockable" in its effective keywords
    rejects all blocks. The "unblockable" keyword is granted by
    ContinuousModifier declarations (e.g. Illvoi Infiltrator's
    "can't be blocked if you've cast two or more spells this turn") via
    walk_continuous_modifiers inside effective_keywords.

    "Can block only creatures with flying" (EoE Drone token): a blocker
    whose printing carries this flag may only declare blocks against
    attackers that have flying.

    Protection from color (CR 702.16): an attacker with protection
    from color X can't be blocked by a creature whose effective
    color includes X. Composes with layer 5 (Scuttlemutt) — once
    the blocker's effective color set drops X, the block becomes
    legal. (Damage-prevention and targeting protections are handled
    elsewhere; only the block-legality clause lives here.)
    """
    from mtg_bot.game import (
        effective_colors, effective_protection_from_colors,
    )

    attacker_kws = effective_keywords(state, attacker)
    blocker_kws = effective_keywords(state, blocker)
    if "flying" in attacker_kws:
        if not ("flying" in blocker_kws or "reach" in blocker_kws):
            return False
    if "unblockable" in attacker_kws:
        return False
    if blocker.printing.can_only_block_creatures_with_flying:
        if "flying" not in attacker_kws:
            return False
    attacker_protect = effective_protection_from_colors(state, attacker)
    if attacker_protect:
        blocker_colors = effective_colors(state, blocker)
        if attacker_protect & blocker_colors:
            return False
    return True


def declare_attackers(state: GameState, attacker_ids: list[int]) -> None:
    from mtg_bot.turn import tap_permanent

    declared: list = []
    for oid in attacker_ids:
        obj = next(
            (o for o in state.battlefield if o.id == oid), None
        )
        if obj is None:
            # A prior attacker's trigger may have removed this one
            # (e.g., a sacrifice effect chained from becomes_tapped).
            # Skip silently — we still respect the agent's intent for
            # the remaining attackers.
            continue
        obj.attacking = True
        declared.append(obj)
        # Vigilance: attacking doesn't tap
        if "vigilance" not in effective_keywords(state, obj):
            tap_permanent(state, obj)
        state.log.append(f"{obj.printing.name} attacks")
        # Append to the per-player "attacked this turn" latch — read by
        # ``effects.conditions.AttackedWithThisTurn(filter)`` to gate
        # cost reductions like Thaumaton Torpedo's "if you attacked
        # with a Spacecraft this turn". The latch records every
        # attacker (not just Spacecraft) so a future filter can ask
        # the same question against any subtype/predicate without a
        # new latch.
        state.player_by_id(obj.controller_id).attacked_this_turn.append(obj)
        # Fire "whenever this creature attacks" triggers on the attacker.
        _fire_attacks_triggers(state, obj)
    # After all attackers are declared, fire Exalted (rule 702.83):
    # "Whenever a creature you control attacks alone, that creature gets
    # +1/+1 until end of turn." Multiple instances of Exalted stack —
    # each Exalted permanent the attacker's controller has applies its
    # own +1/+1.
    _fire_exalted_pumps(state)
    # Once-per-pass ``Event.YOU_ATTACK`` dispatch (DSK "Whenever you
    # attack" family — Razorkin Hordecaller, Most Valuable Slayer,
    # Lionheart Glimmer). Distinct from ``Event.ATTACKS`` which fires
    # per-attacker: this one fires exactly ONCE when the active player
    # has declared at least one attacker, with the attacking player as
    # the subject. The trigger phrase "Whenever you attack" doesn't
    # multiply with attacker count, so a single dispatch is the right
    # semantic. Skipped entirely when no creature actually attacked
    # (every requested attacker was unresolvable / fizzled).
    _fire_you_attack_triggers(state, declared)


def _fire_you_attack_triggers(state: GameState, declared: list) -> None:
    """Fire ``Event.YOU_ATTACK`` once per declare-attackers pass when the
    active player has actually attacked with at least one creature.

    ``declared`` is the list of attacker ``GameObject``s that successfully
    flipped to ``attacking=True`` during this call. The subject is the
    common controller of those attackers (the active player); we read it
    off the first attacker rather than trusting ``state.active_player_idx``
    so a hypothetical edge case where the active player can attack with a
    creature controlled by someone else (Threaten-style temporary control)
    still routes the trigger through the right subject. In practice every
    attacker shares the active player's id.
    """
    from mtg_bot.actions import dispatch_event
    from mtg_bot.effects.predicates import Event

    if not declared:
        return
    attacking_player = state.player_by_id(declared[0].controller_id)
    dispatch_event(
        state, Event.YOU_ATTACK, label="you_attack",
        player=attacking_player,
    )


def _fire_exalted_pumps(state: GameState) -> None:
    """If exactly one creature is attacking, scan permanents its
    controller controls for the ``"exalted"`` keyword. Each instance
    grants +1/+1 until end of turn to the lone attacker.
    """
    attackers = [o for o in state.battlefield if o.attacking]
    if len(attackers) != 1:
        return
    attacker = attackers[0]
    controller_id = attacker.controller_id
    exalted_count = 0
    for source in state.battlefield:
        if source.controller_id != controller_id:
            continue
        if "exalted" in source.printing.keywords:
            exalted_count += 1
    if exalted_count == 0:
        return
    attacker.temp_power_bonus += exalted_count
    attacker.temp_toughness_bonus += exalted_count
    state.log.append(
        f"{attacker.printing.name} gets +{exalted_count}/+{exalted_count} "
        f"until end of turn (Exalted x{exalted_count})"
    )


def _fire_attacks_triggers(state: GameState, attacker) -> None:
    from mtg_bot.actions import dispatch_event, push_triggered_ability
    from mtg_bot.effects.predicates import Event
    from mtg_bot.game import effective_triggered_abilities

    # Predicate-style ``Event.ATTACKS`` dispatch (phase 5). Phase-5
    # commit 2 migrated every static ``trigger="attacks"`` site to the
    # predicate path, so the dispatcher subsumes all printed-ability
    # firings. The legacy walk over ``attacker.printing.triggered_abilities``
    # is gone.
    dispatch_event(
        state, Event.ATTACKS, label="attacks", attacker=attacker,
    )

    # UEOT-granted attacks-triggers (Raging Ravine's "+1/+1 counter
    # when this attacks", granted while animated). These live on
    # ``temp_triggered_abilities`` / layer-6 registry entries via
    # ``effective_triggered_abilities`` rather than on
    # ``printing.triggered_abilities``, so the dispatcher's
    # battlefield-walk doesn't see them. Use a 1000-offset
    # ability_index so the once_per_turn ledger key
    # ``(source_id, ability_index)`` can't collide with a printed-
    # ability index. None of the wired-in granted triggers are
    # once-per-turn, so the offset is purely defensive. Both legacy
    # ``trigger="attacks"`` and predicate-style ``event=Event.ATTACKS``
    # granted shapes are recognized.
    granted_only = [
        ab for ab in effective_triggered_abilities(state, attacker)
        if ab not in attacker.printing.triggered_abilities
    ]
    for i, ability in enumerate(granted_only):
        is_legacy_attacks = ability.trigger == "attacks"
        is_predicate_attacks = ability.matches_event(
            Event.ATTACKS, {"attacker": attacker}, attacker, state,
        )
        if not (is_legacy_attacks or is_predicate_attacks):
            continue
        push_triggered_ability(
            state,
            source=attacker,
            ability=ability,
            ability_index=1000 + i,
            controller_id=attacker.controller_id,
            label="attacks",
        )
    # Phase-7 commit 3 migrated every static
    # ``trigger="enchanted_creature_attacks"`` (Chorale of the Void)
    # and ``trigger="equipped_creature_attacks"`` (Atomic Microsizer)
    # site to predicate-style ``event=Event.ATTACKS,
    # predicate=IsHost(subject="attacker")``. The dispatcher's
    # ATTACKS sweep above already fires those — for an attached aura
    # or Equipment, ``IsHost(subject="attacker")`` evaluates true
    # exactly when ``source.attached_to == attacker.id``, the same
    # condition the legacy walk here gated on. The legacy walk had
    # no live entries left and was dropped.


def declare_blockers(state: GameState, assignments: dict[int, int]) -> None:
    """``assignments`` maps blocker id → attacker id.

    Phase-9 commit 3 collapsed the legacy
    ``trigger="blocks_creature_with_flying"`` walk (which used to
    push Skystinger's pump on the blocker) — every static site for
    that key was migrated to predicate-style ``event=Event.BLOCKS,
    predicate=IsSelf(subject="blocker") & HasKeyword("flying",
    subject="attacker")`` in commit 2, so the dispatcher's
    ``Event.BLOCKS`` sweep below is the single fire path.
    """
    from mtg_bot.actions import dispatch_event
    from mtg_bot.effects.predicates import Event

    for blocker_id, attacker_id in assignments.items():
        blocker = next(o for o in state.battlefield if o.id == blocker_id)
        attacker = next(o for o in state.battlefield if o.id == attacker_id)
        if attacker.temp_unblockable or "unblockable" in effective_keywords(
            state, attacker
        ):
            state.log.append(
                f"{attacker.printing.name} can't be blocked — "
                f"{blocker.printing.name}'s block is ignored"
            )
            continue
        blocker.blocking = attacker_id
        attacker.blocked_by.append(blocker_id)
        state.log.append(f"{blocker.printing.name} blocks {attacker.printing.name}")
        # Predicate-style ``Event.BLOCKS`` dispatch. Subjects are the
        # blocker and the attacker it blocked. Skystinger's
        # ``IsSelf(subject="blocker") & HasKeyword("flying", subject=
        # "attacker")`` is the one in-scope predicate today; future
        # block-filter triggers compose more keyword/type clauses on
        # the same event.
        dispatch_event(
            state, Event.BLOCKS, label="blocks",
            blocker=blocker, attacker=attacker,
        )


def _deals_in_step(kws: frozenset[str], *, first_strike_step: bool) -> bool:
    """Whether a creature with ``kws`` deals damage in the given step.

    First-strike step: creatures with first strike or double strike.
    Regular step: creatures with double strike OR without first strike.
    Creatures with first strike only don't deal regular damage.
    """
    has_first = "first strike" in kws
    has_double = "double strike" in kws
    if first_strike_step:
        return has_first or has_double
    return has_double or not has_first


def _any_first_or_double_strike_in_combat(state: GameState) -> bool:
    for o in state.battlefield:
        if not (o.attacking or o.blocking is not None):
            continue
        kws = effective_keywords(state, o)
        if "first strike" in kws or "double strike" in kws:
            return True
    return False


def resolve_combat_damage(state: GameState, defending_player_id: int) -> None:
    """Assign and deal combat damage.

    If any creature in combat has first or double strike, damage is split
    into two steps with state-based actions between them:
      1. First-strike step: first-strike and double-strike creatures deal.
      2. Regular step: creatures without first strike (and double-strike
         creatures again) deal.
    Otherwise there's just the single regular step (common case).
    """
    from mtg_bot.turn import state_based_actions

    if _any_first_or_double_strike_in_combat(state):
        _resolve_damage_step(state, defending_player_id, first_strike_step=True)
        state_based_actions(state)
        if state.is_over():
            return
    _resolve_damage_step(state, defending_player_id, first_strike_step=False)


def _resolve_damage_step(
    state: GameState, defending_player_id: int, *, first_strike_step: bool
) -> None:
    """Build and apply the damage events for one combat damage step."""
    # Darkness: "prevent all combat damage that would be dealt this
    # turn." Skip the entire damage pass — we still proceed through the
    # step so combat-damage-to-player triggers that key on actual
    # damage simply don't fire (no damage events to pair them with).
    if state.prevent_combat_damage_this_turn:
        state.log.append("combat damage prevented this turn (Darkness)")
        return
    # Event tuple layout:
    # (kind, target_id, amount, source_id, source_keywords, source_controller_id)
    damage_events_typed: list[
        tuple[str, int, int, int, frozenset[str], int]
    ] = []

    for attacker in state.battlefield:
        if not attacker.attacking:
            continue
        if attacker.phased_out:
            # Phased-out creatures don't exist for combat purposes.
            continue
        attacker_kws = effective_keywords(state, attacker)
        if not _deals_in_step(attacker_kws, first_strike_step=first_strike_step):
            # Attacker doesn't deal damage in this step; blockers of this
            # attacker may still deal damage back IF they qualify.
            blockers = [o for o in state.battlefield if o.id in attacker.blocked_by]
            for blocker in blockers:
                blocker_kws = effective_keywords(state, blocker)
                if not _deals_in_step(blocker_kws, first_strike_step=first_strike_step):
                    continue
                blocker_power = effective_power(state, blocker)
                if blocker_power > 0:
                    damage_events_typed.append(
                        ("creature", attacker.id, blocker_power,
                         blocker.id, blocker_kws, blocker.controller_id)
                    )
            continue

        blockers = [o for o in state.battlefield if o.id in attacker.blocked_by]
        attacker_power = effective_power(state, attacker)
        attacker_controller = attacker.controller_id
        has_trample = "trample" in attacker_kws
        has_deathtouch = "deathtouch" in attacker_kws
        if not blockers:
            damage_events_typed.append(
                ("player", defending_player_id, attacker_power,
                 attacker.id, attacker_kws, attacker_controller)
            )
        else:
            remaining = attacker_power
            for i, blocker in enumerate(blockers):
                if remaining <= 0:
                    break
                need = 1 if has_deathtouch else max(
                    0, effective_toughness(state, blocker) - blocker.damage
                )
                is_last = i == len(blockers) - 1
                if is_last and not has_trample:
                    assigned = remaining
                else:
                    assigned = min(remaining, need)
                damage_events_typed.append(
                    ("creature", blocker.id, assigned,
                     attacker.id, attacker_kws, attacker_controller)
                )
                remaining -= assigned
            if has_trample and remaining > 0:
                damage_events_typed.append(
                    ("player", defending_player_id, remaining,
                     attacker.id, attacker_kws, attacker_controller)
                )
            for blocker in blockers:
                blocker_kws = effective_keywords(state, blocker)
                if not _deals_in_step(blocker_kws, first_strike_step=first_strike_step):
                    continue
                blocker_power = effective_power(state, blocker)
                if blocker_power > 0:
                    damage_events_typed.append(
                        ("creature", attacker.id, blocker_power,
                         blocker.id, blocker_kws, blocker.controller_id)
                    )

    # Combat damage in MTG is simultaneous, so aggregate damage to each
    # player into a single end-of-step log line ("P0 takes 5 (life:
    # 15)") instead of one line per attacker. Creature damage,
    # lifelink, and infect-poison events stay per-source because each
    # one can independently matter for triggers and SBAs.
    player_life_damage: dict[int, int] = {}
    sources_hit_player: list[tuple[int, int, int]] = []
    for (
        kind, target_id, amount, source_id, source_kws, source_controller,
    ) in damage_events_typed:
        if amount <= 0:
            continue
        has_infect = "infect" in source_kws
        if kind == "player":
            p = state.player_by_id(target_id)
            if has_infect:
                # Infect: damage to a player becomes that many poison
                # counters instead of life loss. Per-event log preserved
                # so the source/amount remains traceable.
                p.poison_counters += amount
                state.log.append(
                    f"P{target_id} gets {amount} poison counter(s) "
                    f"from infect (now {p.poison_counters})"
                )
            else:
                p.life -= amount
                player_life_damage[target_id] = (
                    player_life_damage.get(target_id, 0) + amount
                )
            sources_hit_player.append((source_id, target_id, amount))
        else:
            obj = next((o for o in state.battlefield if o.id == target_id), None)
            if obj is not None:
                if has_infect:
                    # Infect: damage to a creature becomes -1/-1
                    # counters. The engine doesn't currently track
                    # -1/-1 counters as a distinct field; we fold
                    # them into temp_power/toughness bonuses (UEOT)
                    # as a simplification — functionally lethal for
                    # this-turn damage math, which is what infect
                    # generally ends up doing in our sim.
                    obj.temp_power_bonus -= amount
                    obj.temp_toughness_bonus -= amount
                    state.log.append(
                        f"{obj.printing.name} takes {amount} -1/-1 "
                        f"from infect"
                    )
                else:
                    obj.damage += amount
                    obj.damaged_this_turn = True
                if "deathtouch" in source_kws:
                    obj.marked_lethal = True
                # Fire "whenever enchanted creature is dealt damage"
                # observer triggers on auras attached to obj.
                _fire_enchanted_dealt_damage_triggers(
                    state, obj, amount,
                )
        # Lifelink: the source's controller gains life equal to the damage
        # dealt, regardless of target kind.
        if "lifelink" in source_kws:
            from mtg_bot.game import gain_life
            gain_life(state, source_controller, amount)
            gainer = state.player_by_id(source_controller)
            state.log.append(
                f"P{source_controller} gains {amount} life (life: {gainer.life})"
            )

    # Emit aggregated player-damage lines after all events applied so
    # the "(life: N)" reading reflects the final post-step total.
    for target_id, total in player_life_damage.items():
        p = state.player_by_id(target_id)
        state.log.append(
            f"P{target_id} takes {total} (life: {p.life})"
        )

    # Fire "whenever this creature deals combat damage to a player" triggers
    # for each (source, target_player) pair that actually dealt damage.
    _fire_combat_damage_to_player_triggers(state, sources_hit_player)


def _fire_enchanted_dealt_damage_triggers(
    state: GameState, obj: "GameObject", amount: int,
) -> None:
    """Fire "whenever enchanted creature is dealt damage" triggers on
    auras attached to ``obj``, plus any UEOT-granted
    ``this_dealt_damage`` triggers sitting in
    ``obj.temp_triggered_abilities`` (Requiem Monolith).

    Amount exposed via state.resolving_combat_damage_amount for effects
    that want to read it (e.g., Pain for All's
    echo-damage-to-each-opponent, Requiem Monolith's draw-and-lose).

    Phase-7 commit 3 migrated every static
    ``trigger="enchanted_creature_dealt_damage"`` site to predicate-
    style ``event=Event.DEALT_DAMAGE, predicate=IsHost()`` (Cryoshatter,
    Pain for All), so the legacy aura walk over
    ``aura.printing.triggered_abilities`` had no live entries left and
    was dropped. The aura-firing path is now the
    ``dispatch_event(state, Event.DEALT_DAMAGE, obj=obj, …)`` call
    below: ``IsHost()`` on an attached aura matches when
    ``aura.attached_to == obj.id``. The bespoke granted-ability walk
    for ``this_dealt_damage`` (Requiem Monolith) stays — it lives on
    ``temp_triggered_abilities`` / layer-6 registry entries rather than
    ``printing.triggered_abilities``, so the dispatcher's
    battlefield-walk doesn't see it.
    """
    from mtg_bot.actions import dispatch_event
    from mtg_bot.effects.predicates import Event

    # Predicate-style ``Event.DEALT_DAMAGE`` dispatch. Subject ``obj``
    # is the damaged permanent; ``IsHost()`` matches when an attached
    # aura's ``attached_to`` equals ``obj.id``. ``combat_damage_amount``
    # threads the damage amount through to each pushed StackItem so
    # resolving effects can read ``state.resolving_combat_damage_amount``.
    dispatch_event(
        state, Event.DEALT_DAMAGE,
        label="enchanted-creature-dealt-damage",
        combat_damage_amount=amount,
        obj=obj,
    )

    # UEOT-granted "Whenever this creature is dealt damage, ..." triggers
    # on the damaged object itself (Requiem Monolith). Direct resolve
    # rather than stack-push because the granted ability is ephemeral
    # and isn't represented in the printed-ability ledger; using
    # synthetic indices for push_triggered_ability would require
    # broadening the ledger key type. Sets resolving_source +
    # combat_damage_amount around the resolve so the inner effect can
    # read the damage amount.
    from mtg_bot.game import effective_triggered_abilities

    granted_only = [
        ab for ab in effective_triggered_abilities(state, obj)
        if ab not in obj.printing.triggered_abilities
    ]
    for ability in granted_only:
        if ability.trigger != "this_dealt_damage":
            continue
        state.log.append(
            f"{obj.printing.name}'s granted dealt-damage trigger fires"
        )
        state.resolving_source = obj
        state.resolving_combat_damage_amount = amount
        try:
            ability.effect.resolve(state, obj.controller_id, ())
        finally:
            state.resolving_source = None
            state.resolving_combat_damage_amount = 0


def _fire_combat_damage_to_player_triggers(
    state: GameState, sources_hit_player: list[tuple[int, int, int]]
) -> None:
    """For each (source_id, target_player_id, amount) triple, fire
    every triggered ability that listens for combat damage to a
    player.

    The single-source / your-creature / creature-deals-to-you families
    (Illvoi Infiltrator, Starwinder, Contested War Zone) flow through
    the predicate-trigger dispatcher (phase 9); see the
    ``Event.DEALS_COMBAT_DAMAGE`` dispatch below. The dispatcher
    walks every battlefield permanent, evaluates each one's
    predicate against the ``(source, target)`` subject mapping, and
    pushes a StackItem when the predicate matches. The attacker's
    controller is threaded via
    ``combat_damage_source_controller_id`` (mirrored to
    ``state.resolving_combat_damage_source_controller_id`` at resolve
    time) so observer effects like Contested War Zone's
    ``ChangeControllerToCombatDamageSource`` can read the new
    controller without having to thread it through ``fixed_targets``.

    Two specialty observer keys remain on the legacy string-key path
    because their semantics don't fit the per-event predicate model
    cleanly:

    - ``your_nontoken_artifact_creature_combat_damage_to_player``
      (Virulent Silencer): the source-side qualifier filters on the
      attacker being a nontoken artifact creature, but the
      registration also targets the damaged player as a fixed target
      slot — encoding "fire as observer on permanents controlled by
      the attacker's controller" with the predicate vocabulary would
      need an OBSERVER_CONTROLLED-style predicate that we haven't
      added yet.
    - ``your_counter_creature_combat_damage_to_player_aggregated``
      (Haliya, Ascendant Cadet): fires ONCE per (controller,
      target_player) pair regardless of how many qualifying sources
      damaged the player. The deduplication is fundamentally
      cross-event aggregation, not a per-event predicate.

    Both stay as legacy walks below; they have their own dedicated
    fire paths and are not migrated by phase 9.
    """
    from mtg_bot.actions import dispatch_event, push_triggered_ability
    from mtg_bot.turn import should_fire_trigger
    from mtg_bot.effects.predicates import Event

    for source_id, target_player, amount in sources_hit_player:
        source = next(
            (o for o in state.battlefield if o.id == source_id), None
        )
        if source is None:
            continue  # source died from simultaneous damage
        # Predicate-style ``Event.DEALS_COMBAT_DAMAGE`` dispatch.
        # Subsumes Illvoi Infiltrator
        # (``IsSelf(subject="source") & IsPlayer(subject="target")``),
        # Starwinder (``IsCreature(subject="source") & Controlled(YOU,
        # subject="source") & IsPlayer(subject="target")``), and
        # Contested War Zone (``IsCreature(subject="source") &
        # Controlled(YOU, subject="target")``).
        target_player_obj = state.player_by_id(target_player)
        attacker_controller_id = source.controller_id
        dispatch_event(
            state, Event.DEALS_COMBAT_DAMAGE,
            label="combat-damage-to-player",
            combat_damage_amount=amount,
            combat_damage_source_controller_id=attacker_controller_id,
            source=source,
            target=target_player_obj,
        )

        # Observer: "whenever a nontoken artifact creature you control
        # deals combat damage to a player" — Virulent Silencer.
        # Stays on the legacy string-key path; see docstring above.
        if (
            source.printing.is_artifact
            and source.printing.is_creature
            and not source.printing.is_token
        ):
            for obs in list(state.battlefield):
                if obs.controller_id != source.controller_id:
                    continue
                for i, ability in enumerate(obs.printing.triggered_abilities):
                    if ability.trigger != (
                        "your_nontoken_artifact_creature_"
                        "combat_damage_to_player"
                    ):
                        continue
                    push_triggered_ability(
                        state,
                        source=obs,
                        ability=ability,
                        ability_index=i,
                        controller_id=obs.controller_id,
                        label="artifact-creature-damage",
                        combat_damage_amount=amount,
                        fixed_targets=(("player", target_player),),
                    )

    # Aggregated observer: "whenever one or more creatures you control
    # with +1/+1 counters on them deal combat damage to a player"
    # (Haliya, Ascendant Cadet). Build the set of (controller,
    # target_player) pairs where at least one source carrying a
    # +1/+1 counter, controlled by ``controller``, hit ``target_player``;
    # then fire each matching trigger ONCE per pair (regardless of how
    # many qualifying sources damaged the player). Source-with-counter
    # state is snapshotted at lookup time from the live battlefield
    # entry, matching how the per-source observers above resolve.
    # Stays on the legacy string-key path; see docstring above.
    aggregated_pairs: set[tuple[int, int]] = set()
    for source_id, target_player, _amount in sources_hit_player:
        source = next(
            (o for o in state.battlefield if o.id == source_id), None
        )
        if source is None:
            continue  # source died; can't read counters
        if source.plus_one_counters <= 0:
            continue
        aggregated_pairs.add((source.controller_id, target_player))
    for controller_id, target_player in aggregated_pairs:
        for obs in list(state.battlefield):
            if obs.controller_id != controller_id:
                continue
            for i, ability in enumerate(obs.printing.triggered_abilities):
                if ability.trigger != (
                    "your_counter_creature_combat_damage_"
                    "to_player_aggregated"
                ):
                    continue
                if not should_fire_trigger(state, obs.id, i, ability):
                    continue
                state.log.append(
                    f"{obs.printing.name}'s aggregated counter-creature-"
                    f"damage-to-player trigger fires (P{target_player} was "
                    f"hit)"
                )
                state.resolving_source = obs
                try:
                    ability.effect.resolve(
                        state, obs.controller_id,
                        (("player", target_player),),
                    )
                finally:
                    state.resolving_source = None


def clear_combat_state(state: GameState) -> None:
    # Process "sacrifice at end of combat" schedules (Kavaron Harrier
    # attack token). Done before attacking flags clear so the sacrifice
    # still resolves with combat state intact if any observer cares.
    if state.pending_end_of_combat_sacrifices:
        from mtg_bot.turn import sacrifice as _sacrifice

        scheduled = list(state.pending_end_of_combat_sacrifices)
        state.pending_end_of_combat_sacrifices = []
        for tid in scheduled:
            obj = next((o for o in state.battlefield if o.id == tid), None)
            if obj is not None:
                state.log.append(
                    f"end of combat: sacrifice {obj.printing.name}"
                )
                _sacrifice(state, obj)
    for obj in state.battlefield:
        obj.attacking = False
        obj.blocking = None
        obj.blocked_by = []
