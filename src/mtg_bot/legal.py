"""Enumerate legal actions for the player who currently has priority."""

from __future__ import annotations

from mtg_bot.actions import (
    Action,
    ActivateAbility,
    ActivateStation,
    CastSpell,
    PassPriority,
    PlayLand,
    TurnFaceUp,
    UnlockRoomHalf,
    can_pay_cost,
)
from mtg_bot.cards import ManaCost
from mtg_bot.game import GameObject, GameState
from mtg_bot.turn import MAIN_STEPS


def _has_priority(state: GameState, player_id: int) -> bool:
    return state.players[state.priority_player_idx].id == player_id


def _is_sorcery_timing(state: GameState, player_id: int) -> bool:
    """Sorcery timing: your turn, main phase, empty stack, you have priority."""
    return (
        state.players[state.active_player_idx].id == player_id
        and state.step in MAIN_STEPS
        and not state.stack
        and _has_priority(state, player_id)
    )


def _spell_kinds_for_card(card: GameObject) -> "frozenset[str]":
    """Build the set of spend-restriction context tags for casting
    ``card``. Mirrors the logic in ``actions._cast_spell``: a card may
    satisfy multiple coin restrictions at once (e.g., a legendary
    artifact spell satisfies both "artifact_spell" and "legendary_spell"
    coin tags). Reads ``card.zone`` to detect non-hand origins
    (graveyard / exile / library top — Mm'menon's grant).
    """
    from mtg_bot.game import Zone

    kinds: set[str] = {"spell"}
    if card.printing.is_artifact:
        kinds.add("artifact_spell")
    if (
        "Eldrazi" in card.printing.subtypes
        and not card.printing.mana_cost.colored
    ):
        kinds.add("colorless_eldrazi_spell")
    if card.printing.is_legendary:
        kinds.add("legendary_spell")
    if card.zone != Zone.HAND:
        kinds.add("cast_from_outside_hand")
    return frozenset(kinds)


def _can_cast(
    state: GameState, player_id: int, card: GameObject, cost: ManaCost,
) -> bool:
    """Check whether ``player_id`` can pay ``cost`` to cast ``card``.

    Bundles spend-restriction context (spell_kinds from the card type and
    zone), convoke eligibility, and sac-for-mana eligibility so call sites
    don't have to reconstruct those three arguments on every check.
    ``sac_for_mana`` is always ``True`` — every spell-cast site permits
    Treasure-style sacrifice during cost payment.
    """
    return can_pay_cost(
        state, player_id, cost,
        spell_kind=_spell_kinds_for_card(card),
        convoke=card.printing.has_convoke,
        sac_for_mana=True,
    )


def _targets_are_satisfiable(
    state: GameState, player_id: int, card: GameObject
) -> bool:
    """For spells with targeted spell_effects, check that at least
    ``min_t`` legal targets exist (CR 601.2c — a spell with N
    required targets needs N legal targets to be cast at all). When
    ``target_slot_kinds`` is set, each kind's slot count must be
    backed by enough legal entries of that kind, since duplicates
    can't be picked across slots (CR 115.1b "different targets" rule).

    Non-targeting spells and creatures without spell_effect trivially
    satisfy this.
    """
    effect = card.printing.spell_effect
    if effect is None or not effect.requires_target():
        return True
    legal = effect.legal_targets(state, player_id)
    if not legal:
        return False
    min_t, _max_t = effect.target_count_range()
    if min_t <= 1:
        return True
    slot_kinds = effect.target_slot_kinds()
    if slot_kinds:
        # Per-slot kind enforcement: each unique kind needs as many
        # legal entries as it has slots.
        from collections import Counter
        slot_counts = Counter(slot_kinds[:min_t])
        legal_counts = Counter(t[0] for t in legal)
        for kind, needed in slot_counts.items():
            if legal_counts.get(kind, 0) < needed:
                return False
        return True
    # No per-slot kind enforcement: just ensure at least min_t
    # distinct legal targets exist (different-targets rule).
    return len(legal) >= min_t


def _additional_costs_payable(
    state: GameState, player_id: int, card: GameObject
) -> bool:
    """Check that any cast-time additional costs on ``card`` can be paid.
    Currently: additional_cost_sacrifice_creature_or_artifact requires a
    creature or artifact the caster controls.
    """
    if card.printing.additional_cost_sacrifice_creature_or_artifact:
        have_sac = any(
            o for o in state.battlefield
            if o.controller_id == player_id
            and (o.printing.is_creature or o.printing.is_artifact)
        )
        if not have_sac:
            return False
    if card.printing.additional_cost_sacrifice_creature_or_enchantment:
        # Final Vengeance: "As an additional cost to cast this spell,
        # sacrifice a creature or enchantment." Eligible permanents are
        # any creature or enchantment the caster controls. Auras count
        # (an Aura is an Enchantment subtype); the same controller-
        # filter as the artifact-or-creature flag applies.
        have_sac = any(
            o for o in state.battlefield
            if o.controller_id == player_id
            and (
                o.printing.is_creature
                or "Enchantment" in o.printing.card_types
            )
        )
        if not have_sac:
            return False
    if card.printing.additional_cost_choose_your_creature:
        # "Choose a creature you control or a warped creature card you
        # own in exile." Either source satisfies the cost.
        have_battlefield = any(
            o for o in state.battlefield
            if o.controller_id == player_id and o.printing.is_creature
        )
        if not have_battlefield:
            p = state.player_by_id(player_id)
            have_exiled = any(
                o for o in p.exile
                if o.owner_id == player_id
                and o.warped
                and o.printing.is_creature
            )
            if not have_exiled:
                return False
    return True


def legal_actions(state: GameState, player_id: int) -> list[Action]:
    """Return all legal actions for ``player_id`` right now.

    At any priority moment (on or off your turn), instants in hand whose cost
    is payable and whose targets are satisfiable are legal. At sorcery timing
    (your main phase, stack empty, your priority) creatures, sorceries, land
    plays, and warp-exile recasts are also legal. A card in hand with a warp
    cost can be cast either at its normal cost or for its warp alt-cost
    whenever a normal cast would be legal — proposed separately.
    """
    actions: list[Action] = [PassPriority()]

    if not _has_priority(state, player_id):
        return actions

    p = state.player_by_id(player_id)

    from mtg_bot.actions import apply_cost_reductions

    def _effective_cost(card, base_cost):
        return apply_cost_reductions(card.printing, state, player_id, base_cost)

    def _cost_plus_kicker(card, base_cost):
        """Combined base + kicker cost (colored pips merged)."""
        kc = card.printing.kicker_cost
        if kc is None:
            return base_cost
        merged: dict[str, int] = dict(base_cost.colored)
        for color, n in kc.colored:
            merged[color] = merged.get(color, 0) + n
        return ManaCost(
            generic=base_cost.generic + kc.generic,
            colored=tuple(sorted(merged.items())),
            x_pips=base_cost.x_pips,
        )

    # Weftwalking-style "first spell each player casts during each of
    # their turns may be cast without paying its mana cost." Only
    # offered to the active player when they haven't yet cast a spell
    # this turn AND any battlefield permanent grants it.
    free_first_cast_active = (
        state.players[state.active_player_idx].id == player_id
        and p.spells_cast_this_turn == 0
        and any(
            o.printing.grants_first_spell_each_turn_free
            for o in state.battlefield
        )
    )

    # Instant-speed-castable: instants AND any permanent spell with flash.
    # Hand-zone PlayLand / CastSpell dedup: two same-printing copies in hand
    # reach identical end-states modulo card.id (which is a label with no
    # future bearing on play) — no per-object state on hand-resident
    # GameObjects diverges between copies; counters / tapped / attached_to /
    # warped only attach in non-hand zones. Equivalence key is
    # (printing.name, *modifier_tuple), mirroring the precedent at
    # _player_capabilities.py:557. MAINTENANCE INVARIANT: if a future
    # mechanic ever attaches per-object state to hand-resident cards
    # (counters on cards in hand, "next time you cast THIS card" memory,
    # etc.), this key MUST absorb that state — otherwise legal actions are
    # silently dropped. Same comment applies to the PlayLand and sorcery-
    # speed CastSpell dedup blocks below.
    seen_instant: set[tuple] = set()

    def _emit_hand_cast(card, **kwargs):
        key = (
            card.printing.name,
            kwargs.get("use_warp", False),
            kwargs.get("kicked", False),
            kwargs.get("cast_for_free", False),
            kwargs.get("use_impending", False),
            kwargs.get("cast_room_half", None),
        )
        if key in seen_instant:
            return
        seen_instant.add(key)
        actions.append(CastSpell(card.id, **kwargs))

    for card in p.hand:
        instant_speed = (
            card.printing.is_instant or "flash" in card.printing.keywords
        )
        if not instant_speed:
            continue
        if _can_cast(
            state, player_id, card, _effective_cost(card, card.printing.mana_cost),
        ) and _targets_are_satisfiable(state, player_id, card) and _additional_costs_payable(state, player_id, card):
            _emit_hand_cast(card)
            # Kicked variant: can_pay the combined cost.
            if card.printing.kicker_cost is not None and _can_cast(
                state, player_id, card,
                _cost_plus_kicker(card, _effective_cost(card, card.printing.mana_cost)),
            ):
                _emit_hand_cast(card, kicked=True)
        if (
            card.printing.warp_cost is not None
            and _can_cast(
                state, player_id, card, _effective_cost(card, card.printing.warp_cost),
            )
            and _targets_are_satisfiable(state, player_id, card)
            and _additional_costs_payable(state, player_id, card)
            and p.life > card.printing.warp_additional_life_cost
        ):
            _emit_hand_cast(card, use_warp=True)
        # Battlefield-granted warp: a permanent the controller controls
        # may grant warp to cards in hand matching a filter (Tannuk,
        # Steadfast Second). The grant only applies when the card has
        # no innate warp_cost — innate warp wins at the legal-actions
        # layer so the player isn't offered both options for the same
        # spell. ``effective_warp_cost`` looks up the grant; if it
        # returns a cost (and there's no innate warp already proposed
        # above), offer warp-cast.
        if card.printing.warp_cost is None:
            from mtg_bot.actions import effective_warp_cost
            warp_lookup = effective_warp_cost(state, player_id, card)
            if (
                warp_lookup is not None
                and _can_cast(
                    state, player_id, card,
                    _effective_cost(card, warp_lookup[0]),
                )
                and _targets_are_satisfiable(state, player_id, card)
                and _additional_costs_payable(state, player_id, card)
            ):
                _emit_hand_cast(card, use_warp=True)
        # Weftwalking free-cast variant. Skips the mana-cost payable
        # check (it's free) but still gates on targets / additional
        # non-mana costs.
        if (
            free_first_cast_active
            and _targets_are_satisfiable(state, player_id, card)
            and _additional_costs_payable(state, player_id, card)
        ):
            _emit_hand_cast(card, cast_for_free=True)

    if not _is_sorcery_timing(state, player_id):
        return actions

    # Land plays per turn at sorcery speed. Base = 1, increased by
    # permanents granting "you may play an additional land" (Icetill
    # Explorer).
    max_lands = 1 + sum(
        o.printing.grants_additional_land_per_turn
        for o in state.battlefield
        if o.controller_id == player_id
    )
    grants_play_from_graveyard = any(
        o.controller_id == player_id
        and o.printing.grants_play_lands_from_graveyard
        for o in state.battlefield
    )
    if p.lands_played_this_turn < max_lands:
        # Hand-zone PlayLand dedup; see rationale + invariant on the
        # instant-speed dedup block above.
        seen_playland: set[tuple] = set()
        for card in p.hand:
            if card.printing.is_land:
                key = (card.printing.name,)
                if key in seen_playland:
                    continue
                seen_playland.add(key)
                actions.append(PlayLand(card.id))
        if grants_play_from_graveyard:
            for card in p.graveyard:
                if card.printing.is_land:
                    actions.append(PlayLand(card.id))

    # Creature / sorcery / other permanent (artifact, spacecraft, ...) casts
    # at sorcery speed. Creatures/permanents with flash were already proposed
    # in the instant-speed block above, so skip them here to avoid duplicate
    # CastSpell actions.
    def _is_sorcery_speed_spell(card) -> bool:
        pr = card.printing
        return (
            pr.is_creature
            or pr.is_sorcery
            or (pr.is_permanent and not pr.is_instant and not pr.is_land)
        )

    # Hand-zone CastSpell dedup, sorcery-speed mirror; see rationale +
    # invariant on the instant-speed dedup block above.
    seen_sorcery: set[tuple] = set()

    def _emit_hand_sorcery_cast(card, **kwargs):
        key = (
            card.printing.name,
            kwargs.get("use_warp", False),
            kwargs.get("kicked", False),
            kwargs.get("cast_for_free", False),
            kwargs.get("use_impending", False),
            kwargs.get("cast_room_half", None),
        )
        if key in seen_sorcery:
            return
        seen_sorcery.add(key)
        actions.append(CastSpell(card.id, **kwargs))

    for card in p.hand:
        if "flash" in card.printing.keywords:
            continue
        if card.printing.is_instant:
            continue
        # DSK Rooms: a Room is cast for the chosen half's mana cost,
        # not the printing's combined cost. Surface one CastSpell per
        # half whose cost is payable; skip the generic cast block
        # below (the Room has no single "mana_cost" cast site).
        if card.printing.is_room:
            for half in card.printing.room_halves:
                if _can_cast(
                    state, player_id, card,
                    _effective_cost(card, half.mana_cost),
                ):
                    _emit_hand_sorcery_cast(card, cast_room_half=half.name)
            continue
        if _is_sorcery_speed_spell(card):
            if _can_cast(
                state, player_id, card, _effective_cost(card, card.printing.mana_cost),
            ) and _targets_are_satisfiable(state, player_id, card) and _additional_costs_payable(state, player_id, card):
                _emit_hand_sorcery_cast(card)
                if card.printing.kicker_cost is not None and _can_cast(
                    state, player_id, card,
                    _cost_plus_kicker(card, _effective_cost(card, card.printing.mana_cost)),
                ):
                    _emit_hand_sorcery_cast(card, kicked=True)
            if (
                card.printing.warp_cost is not None
                and _can_cast(
                    state, player_id, card, _effective_cost(card, card.printing.warp_cost),
                )
                and _targets_are_satisfiable(state, player_id, card)
                and _additional_costs_payable(state, player_id, card)
                and p.life > card.printing.warp_additional_life_cost
            ):
                _emit_hand_sorcery_cast(card, use_warp=True)
            # Battlefield-granted warp (sorcery-speed mirror of the
            # instant-speed block above). Same priority: grant only
            # applies when no innate warp_cost is present.
            if card.printing.warp_cost is None:
                from mtg_bot.actions import effective_warp_cost
                warp_lookup = effective_warp_cost(state, player_id, card)
                if (
                    warp_lookup is not None
                    and _can_cast(
                        state, player_id, card,
                        _effective_cost(card, warp_lookup[0]),
                    )
                    and _targets_are_satisfiable(state, player_id, card)
                    and _additional_costs_payable(state, player_id, card)
                ):
                    _emit_hand_sorcery_cast(card, use_warp=True)
            # Impending alt-cost (Duskmourn Overlord cycle). Sorcery-
            # speed only — mirrors the printed text; offered when
            # the card has an impending cost the caster can pay,
            # alongside the normal cast. ``_targets_are_satisfiable`` /
            # ``_additional_costs_payable`` checks aren't strictly
            # needed today (no Overlord targets at cast time / has
            # additional costs) but the gates mirror the warp pattern
            # so a future printing with both Impending and a target /
            # additional cost composes cleanly.
            if (
                card.printing.impending_cost is not None
                and _can_cast(
                    state, player_id, card,
                    _effective_cost(card, card.printing.impending_cost),
                )
                and _targets_are_satisfiable(state, player_id, card)
                and _additional_costs_payable(state, player_id, card)
            ):
                _emit_hand_sorcery_cast(card, use_impending=True)
            # Weftwalking free-cast variant for this sorcery-speed
            # spell. Skips the mana-cost payable check (it's free)
            # but still gates on targets / additional non-mana costs.
            if (
                free_first_cast_active
                and _targets_are_satisfiable(state, player_id, card)
                and _additional_costs_payable(state, player_id, card)
            ):
                _emit_hand_sorcery_cast(card, cast_for_free=True)

    # Warp-cast from graveyard. Timeline Culler grants this; see
    # warp_castable_from_graveyard. Currently only sorcery-speed since
    # all such cards in the pool are creatures.
    for card in p.graveyard:
        if not card.printing.warp_castable_from_graveyard:
            continue
        if card.printing.warp_cost is None:
            continue
        if card.printing.is_instant:
            continue  # extend if a future card needs instant-speed grave warp
        if (
            _can_cast(state, player_id, card, _effective_cost(card, card.printing.warp_cost))
            and _targets_are_satisfiable(state, player_id, card)
            and _additional_costs_payable(state, player_id, card)
            and p.life > card.printing.warp_additional_life_cost
        ):
            actions.append(CastSpell(card.id, use_warp=True))
        # NOTE: no Weftwalking free-cast variant from graveyard. The
        # warp-from-graveyard permission (Timeline Culler) is itself an
        # alternative cost (the warp cost), and CR 117.9b allows only
        # one alternative cost per cast. ``_cast_spell``'s graveyard
        # branch is gated on ``use_warp=True`` for that reason; emitting
        # a ``cast_for_free=True`` variant here would route through the
        # apply path's defensive raise (no graveyard route accepts
        # ``cast_for_free`` without ``use_warp``). Confirmed by a
        # 0.03%-rate ValueError surfaced in v16-vs-v13 eval (1/3263
        # games) when Weftwalking was on the battlefield + Timeline
        # Culler in the graveyard. See test_weftwalking_and_
        # possibility.py::test_legal_actions_does_not_offer_cast_for_
        # free_for_warp_castable_card_in_graveyard.

    # Warp-exile recasts at sorcery speed (for creatures / sorceries).
    # Instants in warp-exile could be recast at instant speed, but we don't
    # have any such cards yet — extend if/when we do.
    for card in p.exile:
        if not card.warped:
            continue
        if card.printing.is_instant:
            continue  # handled above at instant speed (none yet in the pool)
        if _can_cast(
            state, player_id, card, _effective_cost(card, card.printing.mana_cost),
        ) and _targets_are_satisfiable(state, player_id, card) and _additional_costs_payable(state, player_id, card):
            actions.append(CastSpell(card.id))

    # Impulse-cast: cards in exile marked playable_from_exile_this_turn
    # (Territorial Bruntar landfall, Memorial Vault activated) — or
    # marked play_from_exile_requires_kavu (Possibility Technician),
    # which persists until the card leaves exile but is gated on the
    # controller currently controlling a Kavu. Both pay normal mana
    # cost and respect sorcery-speed rules. Lands use PlayLand;
    # non-land permanents / sorceries use CastSpell.
    from mtg_bot.game import effective_subtypes
    controls_kavu = any(
        o.controller_id == player_id
        and "Kavu" in effective_subtypes(state, o)
        for o in state.battlefield
    )
    for card in p.exile:
        impulse = card.playable_from_exile_this_turn
        kavu_gate = card.play_from_exile_requires_kavu and controls_kavu
        if not (impulse or kavu_gate):
            continue
        if card.warped:
            continue  # already handled above
        if card.printing.is_land:
            if p.lands_played_this_turn < (
                1 + sum(
                    o.printing.grants_additional_land_per_turn
                    for o in state.battlefield
                    if o.controller_id == player_id
                )
            ):
                actions.append(PlayLand(card.id))
            continue
        if card.printing.is_instant:
            # Instant speed — always castable if cost payable.
            if _can_cast(
                state, player_id, card, _effective_cost(card, card.printing.mana_cost),
            ) and _targets_are_satisfiable(state, player_id, card):
                actions.append(CastSpell(card.id))
            continue
        # Sorcery-speed cards: main phase, empty stack only.
        if not _is_sorcery_timing(state, player_id):
            continue
        if _can_cast(
            state, player_id, card, _effective_cost(card, card.printing.mana_cost),
        ) and _targets_are_satisfiable(state, player_id, card) and _additional_costs_payable(state, player_id, card):
            actions.append(CastSpell(card.id))

    # Lightstall Inquisitor exile-from-hand grant: the opponent may play
    # the exiled card for as long as it remains in exile, with a {1}
    # surcharge on spells and tapped-entry on lands. Sorcery-speed only
    # (oracle: "only if it's a land or they could cast it" — same timing
    # as a normal cast). Instants could technically be cast at instant
    # speed, but the oracle text's "could cast it" clause is interpreted
    # as "could cast it at sorcery timing" to keep the model simple; no
    # such card combination exists in the pool today.
    for card in p.exile:
        if not card.cast_from_exile_with_lightstall_surcharge:
            continue
        if card.warped:
            continue  # warp block above already handles this
        if card.printing.is_land:
            if _is_sorcery_timing(state, player_id) and p.lands_played_this_turn < (
                1 + sum(
                    o.printing.grants_additional_land_per_turn
                    for o in state.battlefield
                    if o.controller_id == player_id
                )
            ):
                actions.append(PlayLand(card.id))
            continue
        # Non-land: only castable at sorcery timing (mirrors the standard
        # "could cast it" restriction on the card's text).
        if not _is_sorcery_timing(state, player_id):
            continue
        # Apply +{1} surcharge on top of normal effective cost.
        base_with_surcharge = ManaCost(
            generic=_effective_cost(card, card.printing.mana_cost).generic + 1,
            colored=_effective_cost(card, card.printing.mana_cost).colored,
            x_pips=_effective_cost(card, card.printing.mana_cost).x_pips,
        )
        if (
            _can_cast(state, player_id, card, base_with_surcharge)
            and _targets_are_satisfiable(state, player_id, card)
            and _additional_costs_payable(state, player_id, card)
        ):
            actions.append(CastSpell(card.id))

    # Cast-from-library (top card only): Mm'menon, the Right Hand.
    # "You may cast artifact spells from the top of your library."
    # Offered when the controller has any permanent with the grant
    # flag AND the top card is an artifact at sorcery speed.
    has_cast_artifact_top = any(
        o.printing.grants_cast_artifact_from_top_of_library
        for o in state.battlefield
        if o.controller_id == player_id
    )
    if has_cast_artifact_top and p.library:
        top = p.library[0]
        if top.printing.is_artifact:
            # Sorcery timing check (all artifact tops, creature or not).
            if _is_sorcery_timing(state, player_id) and _can_cast(
                state, player_id, top, _effective_cost(top, top.printing.mana_cost),
            ) and _targets_are_satisfiable(state, player_id, top):
                actions.append(CastSpell(top.id))

    # DSK Rooms: ``UnlockRoomHalf`` is sorcery-speed only (CR 715.4).
    # Surface one action per Room the controller controls + locked
    # half whose mana cost is currently payable. The unlock action is
    # NOT a spell (no spell_kinds context), but the same coin-pool
    # check applies via ``can_pay_cost`` with the default tag.
    for obj in state.battlefield:
        if obj.controller_id != player_id:
            continue
        if not obj.printing.is_room:
            continue
        for half in obj.printing.room_halves:
            if half.name in obj.unlocked_halves:
                continue
            if can_pay_cost(state, player_id, half.mana_cost):
                actions.append(
                    UnlockRoomHalf(card_id=obj.id, half_name=half.name)
                )

    # Activated abilities on permanents this player controls.
    for obj in state.battlefield:
        if obj.controller_id != player_id:
            continue
        from mtg_bot.game import effective_activated_abilities
        for idx, ability in enumerate(
            effective_activated_abilities(state, obj)
        ):
            if ability.sorcery_speed and not _is_sorcery_timing(state, player_id):
                continue
            # ``Activate only once.`` rider — DSK Haunted Screen's
            # third activation. Once the source's
            # ``has_activated_once_only_ability`` flag is latched at
            # activation time, every later activation of an ability
            # with this rider is filtered out — including a second
            # attempt within the same priority window before the
            # first activation has resolved.
            if ability.activate_only_once and obj.has_activated_once_only_ability:
                continue
            # Cost payability checks. Apply ability-level mana
            # reductions (Dominion Bracelet: {X} less where X is
            # equipped creature's power) before checking.
            if ability.cost.mana is not None:
                from mtg_bot.actions import _effective_ability_mana_cost
                effective_mana = _effective_ability_mana_cost(state, obj, ability)
                # Activation spend-restriction context: combine all the
                # tags this source qualifies for so coins with any of
                # those restrictions are usable. Mirrors
                # actions._activate_ability.
                ability_kinds: set[str] = {"spell"}
                if (
                    "Eldrazi" in obj.printing.subtypes
                    and not obj.printing.mana_cost.colored
                ):
                    ability_kinds.add("colorless_eldrazi_ability")
                if obj.printing.is_artifact:
                    ability_kinds.add("artifact_activation")
                if obj.printing.is_legendary:
                    ability_kinds.add("legendary_permanent_ability")
                if obj.printing.is_land:
                    ability_kinds.add("land_activation")
                if not can_pay_cost(
                    state, player_id, effective_mana,
                    spell_kind=frozenset(ability_kinds),
                ):
                    continue
            if ability.cost.tap_self:
                if obj.tapped:
                    continue
                # Creatures tapping as part of a cost can't while summoning-sick.
                from mtg_bot.game import is_creature_now
                if is_creature_now(obj) and obj.summoning_sick:
                    continue
            if ability.cost.sacrifice_another_creature_or_artifact:
                has_another = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and o2.id != obj.id
                    and (o2.printing.is_creature or o2.printing.is_artifact)
                )
                if not has_another:
                    continue
            if ability.cost.sacrifice_another_artifact_or_land:
                has_another = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and o2.id != obj.id
                    and (o2.printing.is_artifact or o2.printing.is_land)
                )
                if not has_another:
                    continue
            if ability.cost.sacrifice_another_artifact:
                has_another = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and o2.id != obj.id
                    and o2.printing.is_artifact
                )
                if not has_another:
                    continue
            if ability.cost.remove_plus_one_counter_from_your_creature:
                has_counter_creature = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and o2.printing.is_creature
                    and o2.plus_one_counters > 0
                )
                if not has_counter_creature:
                    continue
            if ability.cost.sacrifice_food:
                has_food = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and "Food" in effective_subtypes(state, o2)
                )
                if not has_food:
                    continue
            if ability.cost.pay_life:
                if p.life <= ability.cost.pay_life:
                    # MTG lets you pay to 0 life (it kills you) but
                    # pragmatically our bot avoids that line — gate
                    # strictly to require surviving the payment.
                    continue
            if ability.cost.sacrifice_a_land:
                has_land = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and o2.printing.is_land
                    and o2.id != obj.id
                )
                if not has_land:
                    continue
            if ability.cost.sacrifice_a_desert:
                has_desert = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and "Desert" in effective_subtypes(state, o2)
                )
                if not has_desert:
                    continue
            if ability.cost.requires_own_creature_power_at_least:
                from mtg_bot.game import effective_power
                threshold = ability.cost.requires_own_creature_power_at_least
                has_big = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and o2.printing.is_creature
                    and effective_power(state, o2) >= threshold
                )
                if not has_big:
                    continue
            if ability.cost.requires_own_artifact_count_at_least:
                # Inventors' Fair: "Activate only if you control three
                # or more artifacts." Count artifacts the activator
                # controls; skip the activation if below the threshold.
                threshold = ability.cost.requires_own_artifact_count_at_least
                artifact_count = sum(
                    1 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and "Artifact" in o2.printing.card_types
                )
                if artifact_count < threshold:
                    continue
            if ability.cost.requires_control_of_subtype:
                # Duskmourn Verge cycle: "Activate only if you control
                # a <T1> or a <T2>." The activator must control at
                # least one permanent whose effective subtypes include
                # any of the listed names.
                wanted = ability.cost.requires_control_of_subtype
                has_subtype = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and any(
                        st in effective_subtypes(state, o2) for st in wanted
                    )
                )
                if not has_subtype:
                    continue
            if ability.cost.exile_granting_equipment:
                # The granting equipment must still be attached to
                # the source. Legal when there's any equipment with
                # a matching grant-flag attached to obj.
                has_granter = any(
                    eq for eq in state.battlefield
                    if eq.attached_to == obj.id
                    and eq.printing.grants_control_opponent_ability_when_equipped
                )
                if not has_granter:
                    continue
            # exile_self — source must be on the battlefield to exile
            # it. The battlefield presence check at the top of this
            # loop already guarantees that, so there is no additional
            # gate beyond what sacrifice_self requires. The field is
            # checked here for explicitness and future-proofing (e.g.
            # if a card ever has an exile_self restriction beyond the
            # normal "must be on battlefield" one).
            # (no extra gate beyond battlefield presence for exile_self)
            if ability.cost.tap_permanent_you_control:
                # "Tap an untapped permanent you control" — Gene
                # Pollinator. Requires at least one OTHER untapped
                # permanent under the activator's control. The source
                # itself cannot be used (same ruling as crewing: the
                # source is the thing benefiting from the cost).
                has_untapped_other = any(
                    o2 for o2 in state.battlefield
                    if o2.controller_id == player_id
                    and o2.id != obj.id
                    and not o2.tapped
                )
                if not has_untapped_other:
                    continue
            if ability.cost.exile_artifact_from_graveyard:
                # "Exile an artifact card from your graveyard" — Rust
                # Harvester. Requires at least one artifact card in the
                # activating player's graveyard.
                has_artifact_in_gy = any(
                    card for card in p.graveyard
                    if card.printing.is_artifact
                )
                if not has_artifact_in_gy:
                    continue
            # If the effect requires a target, at least one must be legal.
            # Set resolving_source so effects with self-exclusion filters
            # (e.g. "another target creature") compute the correct legal
            # pool during legality-check. Cleared immediately afterward.
            if ability.effect.requires_target():
                state.resolving_source = obj
                try:
                    has_legal = bool(
                        ability.effect.legal_targets(state, player_id)
                    )
                finally:
                    state.resolving_source = None
                if not has_legal:
                    continue
            actions.append(
                ActivateAbility(source_id=obj.id, ability_index=idx)
            )

    # Station activations: for each Spacecraft this player controls, each
    # untapped non-sick creature they control (not the Spacecraft itself) is
    # a legal crew choice.
    spacecraft = [
        o for o in state.battlefield
        if o.controller_id == player_id and o.printing.is_spacecraft
    ]
    if spacecraft:
        crew_candidates = [
            o for o in state.battlefield
            if o.controller_id == player_id
            and o.printing.is_creature
            and not o.tapped
            and not o.summoning_sick
        ]
        for ship in spacecraft:
            for crew in crew_candidates:
                if crew.id == ship.id:
                    continue
                actions.append(
                    ActivateStation(spacecraft_id=ship.id, crew_id=crew.id)
                )

    # Turn-face-up special action (CR 701.34 / 707.4): for each
    # face-down permanent the player controls whose underlying
    # printing is a creature card whose mana cost they can pay, offer
    # the action. Special action — does not use the stack — but it's
    # a sorcery-speed timing in our simplified model (CR 707.4
    # actually permits "any time", but the only paths into this engine
    # that matter resolve in main phases anyway, and tying it to
    # sorcery timing avoids interactions with priority windows we
    # don't yet model).
    for obj in state.battlefield:
        if obj.controller_id != player_id:
            continue
        if not obj.face_down:
            continue
        if not obj.printing.is_creature:
            continue
        if not can_pay_cost(state, player_id, obj.printing.mana_cost):
            continue
        actions.append(TurnFaceUp(card_id=obj.id))

    # Deafening Silence gate: post-filter CastSpell actions so
    # noncreature casts beyond the per-turn limit of 1 are removed
    # when any permanent with the restriction is on the battlefield.
    # Kept as a post-filter because CastSpell is proposed from many
    # sites (hand, warp-exile, impulse-exile, cast-from-library) and
    # touching every site would risk drift.
    if any(
        o.printing.restricts_noncreature_spells_to_one_per_turn
        for o in state.battlefield
    ):
        p = state.player_by_id(player_id)
        if p.noncreature_spells_cast_this_turn >= 1:
            filtered: list[Action] = []
            for action in actions:
                if isinstance(action, CastSpell):
                    card = _find_card_by_id(state, player_id, action.card_id)
                    if card is not None and not card.printing.is_creature:
                        continue
                filtered.append(action)
            actions = filtered

    return actions


def _find_card_by_id(state: GameState, player_id: int, card_id: int):
    """Look up a card by id across zones where CastSpell can source
    from: hand, exile (warp / impulse), and library (cast-from-top).
    Returns None if not found — defensive; CastSpell proposals should
    always reference a card the caster can see."""
    p = state.player_by_id(player_id)
    for zone in (p.hand, p.exile):
        for c in zone:
            if c.id == card_id:
                return c
    if p.library and p.library[0].id == card_id:
        return p.library[0]
    return None
