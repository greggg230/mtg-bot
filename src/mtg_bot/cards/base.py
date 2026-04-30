"""Card printings, mana cost parsing, basic lands, and the Stage-1 synthetic pool.

Real Edge of Eternities cards are sitting in `data/edge_of_eternities.json` but
none are playable yet — they all require keywords/triggers that Stage 1 doesn't
support. Stage 1 plays with invented vanilla creatures instead.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mtg_bot.effects.conditions import Condition

from mtg_bot.effects import (
    ActivatedAbility,
    ActivationCost,
    ConditionalIfOpponentHasMoreLands,
    ConditionalIfSourcePowerAtLeast,
    ConditionalIfVoid,
    ConditionalIfYouControlAtLeast,
    ConditionalIfYouGainedLifeAtLeast,
    ControlTargetOpponentNextTurn,
    DealDamageToControllerAndAddMana,
    BecomeLandCreatureUEOT,
    BecomesCopyOfTargetLand,
    ChangeControllerToCombatDamageSource,
    CopyTargetInstantOrSorceryYouControl,
    CreateTokenCopyOfTargetCreatureYouControl,
    DestroyEachNonlandPermanentWithMVEqualToSourceChargeCounters,
    DestroyTargetLand,
    ExileAllGraveyards,
    ExileTargetAttackingCreatureUntilEndStep,
    GrantLifeToEachOpponent,
    GrantUnblockableUEOT,
    LockOrUnlockDoorOfTargetRoomYouControl,
    PutBasicLandFromHandOntoBattlefieldTapped,
    ReturnTargetLandFromGraveyardToHand,
    SacrificeNLandsYouControl,
    UntapTargetLand,
    Sequence,
    AddManaToPool,
    AddNManaOfChosenColor,
    ConditionalOnTappedCreatures,
    CounterTargetSpell,
    CreateToken,
    DealDamage,
    DealDamageAnyTarget,
    DealDamageEachOpponent,
    DealDamageEqualToSourcePowerAnyTarget,
    DefendingPlayerLosesHalfLifeRoundedUp,
    DealDamageToOpponentCreature,
    DealDamageToOpponentCreatureByArtifactCount,
    DealDamageToOpponentCreatureByCardsInHand,
    DealDamageToTargetCreature,
    DealDamageToTargetOpponent,
    DealDamageToTargetTappedCreature,
    DestroyAllCreatures,
    DestroyEnchantedPermanent,
    DestroyTargetArtifact,
    DestroyTargetCreature,
    DestroyTargetNonlandPermanent,
    ExileOpponentsNonlandPermanentUntilLeaves,
    ChoraleReanimateDefenderGraveCreatureEffect,
    EnchantedCreatureFightsTargetEffect,
    EnduringDiesReturnAsEnchantmentEffect,
    KavaronHarrierAttackEffect,
    PainForAllDamageEchoEffect,
    PainForAllETBEffect,
    MemorialVaultActivatedEffect,
    PerigeeBeckonerDiesReturnTappedEffect,
    PossibilityTechnicianETBEffect,
    TerritorialBruntarLandfallEffect,
    TractorBeamETBEffect,
    SacrificeSelfUnlessTapPermanent,
    SacrificeSelfUnlessVoidActive,
    WeftwalkingETBEffect,
    PinnacleStarcageETBEffect,
    SotheraEndStepReanimateEffect,
    SotheraExileOpponentCreatureEffect,
    PinnacleStarcageSacEffect,
    RequiemMonolithActivatedEffect,
    RovingActuatorETBEffect,
    PutStunCounterOnTappedCreature,
    TapAndStunUpToOneTargetCreature,
    TapTargetOpponentPermanentAndStunIt,
    ExileTargetCreatureUntilSourceLeaves,
    SeedshipImpactEffect,
    MovePlusOneCounterBetweenPermanentsYouControl,
    MoveSelfCountersToTargetCreatureYouControl,
    RegenerateTargetCreature,
    UntapTargetOtherPermanent,
    PutPlusOneCountersOnSelfPerOtherCreatureOrArtifact,
    PutChargeCountersOnSelf,
    PutChargeCountersOnTargetSpacecraftOrPlanetYouControl,
    PutChargeCountersOnTargetSpacecraftYouControl,
    DiscardSelf,
    DrawCards,
    DrawCardsEqualToResolvingCombatDamage,
    DrawCardsEqualToResolvingCounterAmount,
    EachOpponentDiscards,
    EachOpponentMillsCards,
    EachOpponentSacrificesCreature,
    EachOpponentLosesLife,
    EachPlayerDraws,
    Effect,
    ExileEachOpponentsGraveyard,
    ExileTargetCardFromGraveyard,
    ExileTargetCreature,
    ExileTargetCreatureWithPowerOrToughnessAtMost,
    GainControlOfTargetSliverPermanently,
    GainLife,
    IfGainedLifeThisTurnUntapSelf,
    LoseLife,
    PutCardFromGraveyardOnBottomOfLibrary,
    PumpCreaturesOpponentsControlUntilEOT,
    PumpCreaturesYouControlUntilEOT,
    PutPlusOneCounterOnOpponentsCreature,
    PumpSelfUntilEOT,
    PumpSelfXPowerUEOT,
    PhaseOutEquippedCreature,
    PreventAllCombatDamageThisTurn,
    PumpTargetArtifactOrCreatureUntilEOT,
    PumpTargetCreatureUntilEOT,
    PutPlusOneCounterOnCreatureYouControl,
    PutPlusOneCounterOnEachCreatureYouControl,
    PutPlusOneCounterOnEachOfUpToNTargetCreatures,
    PutPlusOneCounterOnSelf,
    PutPlusOneCounterOnTargetCreature,
    PutPlusOneCountersAndGrantKeywordsUEOTOnTargetCreatureYouControl,
    PutPoisonCountersOnTargetPlayer,
    ReturnAllCreaturesFromYourGraveyardToBattlefield,
    ReturnCreatureFromYourGraveyardToBattlefield,
    ReturnCreatureFromYourGraveyardToHand,
    ReturnTargetCreatureToOwnersHand,
    ReturnTargetPermanentYouControlToOwnersHand,
    ReturnUpToNCreaturesFromYourGraveyardToHand,
    ReturnUpToNSmallCreaturesFromGraveyardWithCounters,
    ReturnUpToNTargetCreaturesToOwnersHands,
    DestroyTargetNonlandPermanentWithLanderForController,
    DestroyTargetArtifactOrTappedCreature,
    TapAnotherTargetCreature,
    TapTargetArtifactOrCreature,
    DepressurizeEffect,
    DestroyTargetArtifactEnchantmentOrFlyingCreature,
    CutPropulsionEffect,
    DiplomaticRelationsEffect,
    DealDamageToTargetCreatureSpacecraftBoosted,
    DealDamageToTargetCreatureWithExcessLanderRider,
    ExileAllArtifactsWithManaValueAtMost,
    AnticausalVestigeLeavesEffect,
    AddManaToPool,
    AddOneManaFromControlledLandsProductions,
    AddOneManaOfChosenPermanentColor,
    AllFatesScrollDrawEffect,
    AlpharaelDreamingAcolyteDiscardEffect,
    ArabellaAttackEffect,
    AttackInTheBoxAttackEffect,
    GhostVacuumActivatedEffect,
    HauntedScreenActivatedEffect,
    HouseCartographerSurvivalEffect,
    AstelliReclaimerETBEffect,
    AtomicMicrosizerAttackEffect,
    AttachToTargetCreatureYouControl,
    EquipToTargetCreatureYouControl,
    BiosynthicBurstEffect,
    BoilerbilgesRipperETBEffect,
    CreateLanderAndScheduleSacrifice,
    CreateTokenAndAttachSelf,
    CloseEncounterEffect,
    CommuneWithEvilEffect,
    ConsultTheStarChartsEffect,
    CreateTokenPerMulticoloredPermanent,
    DrawCardPerMulticoloredPermanent,
    DyadrineAttackEffect,
    EachOpponentDiscardsOrLosesLife,
    FamishedWorldsireETBEffect,
    FanaticOfTheHarrowingETBEffect,
    PutPlusOneCountersOnEachCreatureYouControlEqualToSourcePower,
    OptionalDiscardCardThenDrawN,
    OptionalSacLandOrLanderForBasicLandRamp,
    OptionalPutCreatureFromHandOntoBattlefield,
    OptionalPutLandFromHandOntoBattlefield,
    CreateXTokensOfTargetPermanentWithHasteEndStepSac,
    DealNDamageToEachCreature,
    DestroyTargetNoncreaturePermanent,
    DiscardHandThenDrawN,
    DoubleTargetCreaturesPowerUEOT,
    PumpTargetOtherCreatureByXOverXWhereXIsSourcePower,
    TargetOpponentLosesLife,
    TargetOpponentLosesNYouGainN,
    TargetPlayerLosesLife,
    ExileAllCreaturesAndSpacecraft,
    OptionalSacArtifactForEffect,
    PutLoyaltyCounterOnSelf,
    TargetArtifactBecomesCreatureUEOT,
    ReturnTargetArtifactCardFromYourGraveyardToHand,
    SetLifeToHalfStartingRoundedUp,
    SwitchSourcePowerToughnessUEOT,
    CoordinatedClobberingEffect,
    ExorciseEffect,
    SingularityRuptureEffect,
    SystemsOverrideEffect,
    TerrapactIntimidatorETBEffect,
    VengefulPossessionEffect,
    WickerfolkThresherAttackEffect,
    ZeroPointBalladEffect,
    ModalChooseOne,
    MonoistCircuitFeederETBEffect,
    FocusFireEffect,
    OptionalManaCostThenEffect,
    OptionalSacAnotherForEffect,
    OptionalSacAnotherForPumpSelfUEOT,
    OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    OwnerPutsTargetWarpedExileCardOnBottomOfLibrary,
    SelfMill,
    MillTargetPlayer,
    SearchLibraryForBasicLandTapped,
    SearchLibraryForCard,
    SearchLibraryForEquipmentToBattlefield,
    SearchLibraryForGreenCreatureWithMVAtMostX,
    TakeExtraTurnAfterThisOne,
    SelfcraftMechanETBEffect,
    SplitskinDollETBEffect,
    TannukEnsignLandfallEffect,
    Sequence,
    StaticAbility,
    StaticScope,
    LookAtTopNRevealOrCounter,
    LookAtTopTwoSplitHandGraveyard,
    ManifestDread,
    ManifestDreadThenPutPlusOneCountersOnIt,
    MutinousMassacreEffect,
    OsseousSticktwisterEndStepEffect,
    Surveil,
    TargetOpponentDiscards,
    TargetOpponentDiscardsCasterChoice,
    TriggeredAbility,
)


COLORS: tuple[str, ...] = ("W", "U", "B", "R", "G")


@dataclass(frozen=True)
class ManaCost:
    generic: int = 0
    colored: tuple[tuple[str, int], ...] = ()
    # Number of {X} symbols in the printed cost. The actual generic mana
    # paid at cast time is ``generic + x_pips * X`` where X is the
    # caster's chosen value. Stored separately so a card's "templated"
    # cost stays distinct from its paid cost.
    x_pips: int = 0

    @property
    def total(self) -> int:
        return self.generic + sum(n for _, n in self.colored)

    @property
    def colors(self) -> frozenset[str]:
        return frozenset(c for c, _ in self.colored)

    def reduced(self, amount: int) -> "ManaCost":
        """Return a new ManaCost with ``amount`` generic mana removed
        (minimum 0; colored costs unchanged — MTG rule 601.2f). The
        ``x_pips`` template is preserved: cost reductions only touch
        the generic side, so an X-spell stays an X-spell after a
        discount (CR 601.2f reduces generic mana, leaving {X} symbols
        intact for the controller to choose later)."""
        if amount <= 0:
            return self
        return ManaCost(
            generic=max(0, self.generic - amount),
            colored=self.colored,
            x_pips=self.x_pips,
        )


@dataclass(frozen=True)
class CostReduction:
    """A rule that reduces the generic-mana portion of a spell's cost.

    ``condition`` is a structural :class:`mtg_bot.effects.conditions.
    Condition` evaluated against game state at cast time. ``amount`` is
    the unit generic reduction; the engine multiplies it by
    ``condition.multiplier(state, caster_id, chosen_targets)`` so
    variable-amount conditions (Fungal Colossus's
    ``PerDifferentlyNamedLand``, Affinity-style ``PerControlled``) scale
    automatically without a separate flag.
    """

    amount: int
    condition: "Condition"


@dataclass(frozen=True)
class ExternalCostReduction:
    """Static cost-reduction granted by a permanent on the battlefield
    to OTHER spells the controller casts. Lives on a permanent's
    printing; ``apply_cost_reductions`` scans the caster's battlefield
    for these whenever they cast a spell.

    Distinct from ``CostReduction`` (which is baked into the spell
    being cast and reduces ITS own cost). Externals come from a
    different permanent than the spell, and apply to spells they
    don't share an id with.

    ``condition`` is a structural :class:`mtg_bot.effects.conditions.
    Condition` — the same grammar the spell-local ``CostReduction``
    uses. Variable-amount externals (Sami, Wildcat Captain's "Spells
    you cast have affinity for artifacts" → ``PerControlled(IsArtifact
    ())``) scale via the condition's ``multiplier``.
    """

    amount: int
    condition: "Condition"


def parse_mana_cost(s: str) -> ManaCost:
    if not s:
        return ManaCost()
    generic = 0
    colored: dict[str, int] = {}
    x_pips = 0
    for sym in re.findall(r"\{([^}]+)\}", s):
        if sym.isdigit():
            generic += int(sym)
        elif sym in COLORS:
            colored[sym] = colored.get(sym, 0) + 1
        elif sym == "C":
            generic += 1
        elif sym == "X":
            x_pips += 1
        else:
            raise ValueError(f"unsupported mana symbol: {{{sym}}}")
    return ManaCost(
        generic=generic,
        colored=tuple(sorted(colored.items())),
        x_pips=x_pips,
    )


@dataclass(frozen=True)
class RoomHalf:
    """One door/half of a Duskmourn Room card (CR 207.2g + 715).

    A Room is an enchantment with two halves; each has its own name,
    mana cost, and abilities. When a Room spell is cast, it's cast for
    one half's mana cost — that half enters unlocked, the other locked.
    The owner can later unlock the locked half at sorcery speed by
    paying its mana cost (see ``UnlockRoomHalf`` action).

    Locked halves' abilities are inert. The
    ``effective_triggered_abilities`` / ``effective_activated_abilities``
    iterators filter out abilities whose ``room_half`` name is not in
    the source's ``GameObject.unlocked_halves``. The static-ability
    iterator does the same.

    The ``triggered_abilities`` / ``static_abilities`` /
    ``activated_abilities`` here are also flattened onto the parent
    ``CardPrinting`` (with ``room_half`` set on each) so the existing
    ability iterators work without per-call-site special-casing.
    """

    name: str
    mana_cost_str: str
    mana_cost: ManaCost
    triggered_abilities: tuple["TriggeredAbility", ...] = ()
    static_abilities: tuple["StaticAbility", ...] = ()
    activated_abilities: tuple["ActivatedAbility", ...] = ()


@dataclass(frozen=True)
class CardPrinting:
    name: str
    mana_cost_str: str
    mana_cost: ManaCost
    type_line: str
    supertypes: tuple[str, ...]
    card_types: tuple[str, ...]
    subtypes: tuple[str, ...]
    power: int | None = None
    toughness: int | None = None
    produces: frozenset[str] = frozenset()
    keywords: frozenset[str] = frozenset()
    triggered_abilities: tuple[TriggeredAbility, ...] = ()
    # For instants and sorceries: the effect that runs when the spell
    # resolves. After resolution the spell goes to its owner's graveyard.
    # None for creatures, lands, and spells whose effect is implicit (rare).
    spell_effect: Effect | None = None
    # Edge of Eternities "warp" alt-cost. If set, the card may be cast from
    # hand for this cost instead of its normal cost; when cast via warp it's
    # exiled at the next end step and can then be cast from exile at its
    # normal cost on a later turn.
    warp_cost: ManaCost | None = None
    warp_cost_str: str = ""
    # Additional life cost paid alongside the warp mana cost. When > 0,
    # warp-casting deducts this life from the caster on top of paying
    # warp_cost. Matches Timeline Culler's "Warp—{B}, Pay 2 life."
    # legal_actions checks the controller has > life loss available.
    warp_additional_life_cost: int = 0
    # When True, the card is also castable via its warp ability from
    # the graveyard, not just the hand. Matches Timeline Culler's
    # "You may cast this card from your graveyard using its warp ability."
    warp_castable_from_graveyard: bool = False
    # Duskmourn "Impending N — {alt-cost}" alt-cost. If set, the card
    # may be cast from hand for ``impending_cost`` instead of its
    # normal cost; when cast that way, it enters with
    # ``impending_count`` time counters and is suppressed-as-non-
    # creature (``impending_active=True``) until the last counter is
    # removed at the controller's end step. The Overlord cycle
    # (Duskmourn) is the printing-source: each Overlord is an
    # Enchantment Creature with both a normal cost and an Impending
    # cost. When cast for the Impending cost the permanent's printed
    # creature type is suppressed via ``is_creature_now`` reading
    # ``GameObject.impending_active`` — the printing's
    # ``card_types`` still includes "Creature" but
    # ``is_creature_now`` returns False so combat / target /
    # creature-only triggers don't fire. The end-step turn-based
    # action ``run_impending_decrement_turn_based_action`` removes
    # one time counter from each impending-active permanent the
    # active player controls; when the count hits 0 it clears
    # ``impending_active`` and the permanent becomes a creature.
    impending_cost: ManaCost | None = None
    impending_cost_str: str = ""
    impending_count: int = 0
    # "Kicker {cost}" — an optional additional mana cost the caster may
    # pay while casting. If paid, the StackItem is marked kicked and
    # the resolving effect can branch on state.resolving_kicked.
    kicker_cost: ManaCost | None = None
    kicker_cost_str: str = ""
    # Planeswalker starting loyalty. None on non-planeswalkers. When
    # the planeswalker enters the battlefield it starts with this many
    # loyalty counters on it (GameObject.loyalty_counters).
    loyalty: int | None = None
    # Devour land N: "As this creature enters, you may sacrifice any
    # number of lands. It enters with N times that many +1/+1 counters
    # on it." Non-zero enables the replacement in put_into_battlefield.
    # 0 = no devour. Matches Famished Worldsire (N=3).
    devour_land_multiplier: int = 0
    # EoE Spacecraft Station mechanic. For a Spacecraft card: once it has
    # station_threshold or more charge counters on it, it becomes an artifact
    # creature with its printed P/T and gains stationed_keywords. None on
    # non-Spacecraft cards.
    station_threshold: int | None = None
    stationed_keywords: frozenset[str] = frozenset()
    # Saga (Enchantment — Saga). When > 0, this card is a Saga
    # with the given final chapter number. Sagas would get a lore
    # counter at the start of each precombat main phase as a
    # turn-based action, with chapter-numbered triggered
    # abilities firing when the lore counter advances (those
    # mechanics aren't modeled in scope). The state-based action
    # (CR 704.5r in audit numbering) sacrifices a saga whose
    # ``lore_counters >= chapter_count``. 0 = not a saga.
    chapter_count: int = 0
    # Ward {cost}: when this creature becomes the target of a spell or
    # ability an opponent controls, counter that spell/ability unless the
    # controller pays this cost. We simplify to "caster auto-pays if they
    # can; otherwise the spell is countered at cast time" — no player
    # choice, no pay-or-counter trigger on the stack.
    ward_cost: ManaCost | None = None
    ward_cost_str: str = ""
    # "Enchanted/equipped permanent has ward {cost}." Aura/Equipment grant.
    # When this printing is attached to another permanent, the enchanted
    # permanent gains ward {cost} on top of any printed ward. Combined by
    # game.effective_ward_cost and read by actions._apply_ward_check.
    # Hardlight Containment's "Enchanted permanent has ward {1}" rider.
    attached_grants_ward_cost: ManaCost | None = None
    attached_grants_ward_str: str = ""
    # "You may play an additional land on each of your turns." Non-zero
    # value contributes to the land-per-turn cap via legal_actions.
    # Icetill Explorer has this at 1.
    grants_additional_land_per_turn: int = 0
    # "You may play lands from your graveyard." When True on any
    # permanent the controller controls, legal_actions also lists
    # land cards in the controller's graveyard as PlayLand actions
    # (still subject to the land-per-turn cap). Matches Icetill
    # Explorer. _play_land removes the card from the graveyard.
    grants_play_lands_from_graveyard: bool = False
    # "If a permanent entering the battlefield causes a triggered
    # ability of a permanent you control to trigger, that ability
    # triggers an additional time." Starfield Vocalist. Summed across
    # permanents the acting player controls by etb_trigger_multiplier
    # — each one adds 1 extra firing.
    extra_etb_trigger: int = 0
    # "Enchanted/equipped permanent doesn't untap during its
    # controller's untap step." Tractor Beam's don't-untap clause.
    # When True on an aura/equipment attached to a permanent, the
    # untap step skips that permanent.
    attached_prevents_untap: bool = False
    # "You control enchanted permanent." Tractor Beam's static
    # control-grant clause. When True on an aura, the moment it
    # enters the battlefield attached to a target,
    # ``apply_aura_static_grants_on_attach`` writes a layer-2
    # ``"control_change"`` entry to ``state.continuous_effects`` —
    # applied BEFORE any ETB triggers go on the stack, since static
    # abilities are continuous (not stack-based).
    # ``effective_controller`` derives the target's controller from
    # the registry (most-recent timestamp wins), and
    # ``move_off_battlefield`` drops the entry when the aura leaves
    # play. Supports nested swaps in CR-613 timestamp order.
    attached_grants_control: bool = False
    # Lignify-style aura flags. When set on an Aura's printing,
    # ``apply_aura_static_grants_on_attach`` writes corresponding
    # registry entries on attach so the layered registry models the
    # full Lignify effect (CR 613, layers 4 + 6 + 7b):
    #   - ``attached_sets_base_pt`` — layer 7 sublayer 1 set-base-PT,
    #     params ``(power, toughness)``. Overrides the enchanted
    #     permanent's base power and toughness while attached.
    #   - ``attached_grants_lose_all_abilities`` — layer 6
    #     ``"lose_all_abilities"``. Suppresses every printed and
    #     granted ability/keyword/trigger on the enchanted permanent.
    #   - ``attached_grants_subtype`` — layer 4 ``"add_subtype"``
    #     adding the named subtype to the enchanted permanent (e.g.
    #     "Treefolk" for Lignify).
    # All three live with ``expires_at=None`` so they end when the
    # source aura leaves play (``remove_continuous_effects_from_source(
    # only_static=True)`` drops them in ``move_off_battlefield``).
    attached_sets_base_pt: "tuple[int, int] | None" = None
    attached_grants_lose_all_abilities: bool = False
    attached_grants_subtype: "str | None" = None
    # Clone-style ETB replacement: "You may have this creature enter
    # as a copy of any creature on the battlefield." When True, the
    # spell-resolution path prompts the controller for a target
    # creature and (if chosen) snapshots ``original_printing`` and
    # mutates ``printing`` to the target's PRINTED characteristics
    # (CR 706.2 copiable values). A layer-1 ``"is_copy"`` registry
    # entry is also written with ``expires_at=None`` and
    # ``source_id=target_id=clone.id`` so the layered system records
    # the copy event; the entry is dropped when the clone leaves
    # play. If no legal target exists, or the controller declines,
    # the clone enters as itself (its printed self).
    enters_as_copy_of_creature: bool = False
    # "The first time you would create one or more tokens each turn,
    # you may instead create that many tokens that are copies of
    # enchanted permanent." Moonlit Meditation. Aura-only flag.
    replaces_first_tokens_with_copies: bool = False
    # "You may cast artifact spells from the top of your library."
    # Mm'menon, the Right Hand. legal_actions offers a CastSpell for
    # the controller's top-of-library card when it's an artifact and
    # this flag is present on any permanent they control.
    grants_cast_artifact_from_top_of_library: bool = False
    # "The first spell each player casts during each of their turns
    # may be cast without paying its mana cost." Weftwalking. When any
    # permanent on the battlefield has this flag, legal_actions offers
    # a free-cost CastSpell variant (cast_for_free=True) to the active
    # player whenever they have not yet cast a spell this turn.
    # _cast_spell skips cost payment when cast_for_free is True.
    grants_first_spell_each_turn_free: bool = False
    # "Artifacts you control are Foods in addition to their other
    # types." Ragost, Deft Gastronaut. Consumed by
    # game.effective_subtypes to inject "Food" into artifact subtypes.
    grants_food_to_your_artifacts: bool = False
    # "Artifacts you control have '{2}, {T}, Sacrifice this artifact:
    # You gain 3 life.'" Ragost's companion ability-grant clause.
    # Consumed by game.effective_activated_abilities to append the
    # pre-built RAGOST_FOOD_GRANTED_ABILITY to artifact activations.
    grants_food_ability_to_your_artifacts: bool = False
    # "Artifacts you control have '{T}: Add {U}. Spend this mana only
    # to cast a spell from anywhere other than your hand.'" Mm'menon,
    # the Right Hand's static-grants-activated-ability clause.
    # Consumed by game.effective_activated_abilities to append the
    # pre-built MMNEMON_BLUE_MANA_GRANTED_ABILITY to artifact
    # activations.
    grants_blue_mana_ability_to_your_artifacts: bool = False
    # Equipment flag. When this equipment is attached to a creature,
    # that creature gains an activated ability to control the next
    # turn of a target opponent (The Dominion Bracelet). The full
    # ability is DOMINION_BRACELET_GRANTED_ABILITY.
    grants_control_opponent_ability_when_equipped: bool = False
    # Ward—Discard a card at random (non-mana ward). Alpharael, Stonechosen.
    # When True, the caster must randomly discard a card to pay the ward;
    # if their hand is empty, the spell is countered.
    ward_discard_random: bool = False
    # Ward—Discard a card (chosen, non-mana ward). Sunset Saboteur.
    # When True, the caster picks (via ``Agent.choose_discard``) which card
    # to discard from their hand to pay the ward; if their hand is empty,
    # the spell is countered.
    ward_discard_chosen: bool = False
    # Equipment: continuous bonuses granted to the equipped creature.
    # Tuple (power_bonus, toughness_bonus, granted_keywords). None on
    # non-Equipment cards. Consumed by game._static_bonuses.
    equipment_bonuses: tuple[int, int, frozenset[str]] | None = None
    # Shock-land-style optional ETB life cost. When nonzero, _play_land
    # asks the controller's agent whether to pay; yes pays the life and
    # enters untapped, no enters tapped. 0 means "no choice — enters
    # per its usual rules".
    shock_land_life_cost: int = 0
    # When True, this card always enters the battlefield tapped
    # (Planet-style utility lands and other "enters tapped" permanents).
    enters_tapped: bool = False
    # Duskmourn "horror dual" cycle (Bleeding Woods, Etched Cornfield,
    # …). When > 0, the land enters tapped unless any player's life
    # total is at or below this threshold at ETB time. No agent choice
    # — the condition is a deterministic state check. 0 disables.
    enters_tapped_unless_any_player_life_at_most: int = 0
    # Echoing Deeps: "You may have this land enter tapped as a copy of
    # any land card in a graveyard, except it's a Cave in addition to
    # its other types." When True, _play_land offers the controller's
    # agent a graveyard-land target. If they pick one, the played
    # land's printing is replaced with a Cave-injected copy of the
    # target and it enters tapped. If they decline (or no legal
    # target exists), the land enters per its normal rules.
    optional_copy_land_from_graveyard: bool = False
    # Additional cost to cast: sacrifice a creature or artifact the
    # caster controls (other than any source-of-the-spell). Matches
    # Embrace Oblivion's "As an additional cost to cast this spell,
    # sacrifice an artifact or creature."
    additional_cost_sacrifice_creature_or_artifact: bool = False
    # Additional cost to cast: sacrifice a creature or enchantment the
    # caster controls. Matches Final Vengeance's "As an additional cost
    # to cast this spell, sacrifice a creature or enchantment."
    # Parallel to ``additional_cost_sacrifice_creature_or_artifact``;
    # both flags route through the same ``_additional_costs_payable``
    # gate in ``legal.py`` and the same pay-time sacrifice path in
    # ``actions.py``, just with the eligible-permanent filter set per
    # the printed cost body.
    additional_cost_sacrifice_creature_or_enchantment: bool = False
    # Additional cost to cast: designate one of your creatures. The
    # chosen creature's id is stored on the StackItem and exposed via
    # state.resolving_additional_cost_choice_id for the effect to use
    # (e.g. Close Encounter's damage-equal-to-power).
    additional_cost_choose_your_creature: bool = False
    # Cost reductions: rules that lower this spell's generic mana cost
    # when specific conditions hold at cast time (e.g., "costs {3} less
    # if you've cast another spell this turn"). Consumed by
    # actions.apply_cost_reductions().
    cost_reductions: tuple[CostReduction, ...] = ()
    # Static cost-reduction effects this permanent provides to the
    # controller's OTHER spells while it's on the battlefield (e.g.,
    # Uthros Psionicist: "the second spell you cast each turn costs
    # {2} less"). apply_cost_reductions() walks the caster's battlefield
    # and applies any matching externals on top of the spell's own
    # cost_reductions.
    external_cost_reductions: tuple[ExternalCostReduction, ...] = ()
    # Continuous abilities that buff other permanents while this permanent
    # is on the battlefield (anthems, keyword grants).
    static_abilities: tuple[StaticAbility, ...] = ()
    # Abilities the controller may activate by paying a cost.
    activated_abilities: tuple["ActivatedAbility", ...] = ()
    # "Enters with +1/+1 counters" replacement effects tied to the cast-time
    # choice of X. Consumed by actions.put_into_battlefield.
    #   ``"x"`` — enters with state.resolving_x_value counters (Broodguard).
    #   ``"mana_spent"`` — enters with counters equal to total mana paid,
    #     i.e. X + base generic + colored pips (Dyadrine).
    #   None — no such replacement.
    enters_with_counters_from_x: str | None = None
    # External-anthem flag. While this permanent is on the battlefield,
    # each creature its controller plays enters with one additional +1/+1
    # counter for each land that entered the battlefield under that
    # controller this turn (read from Player.lands_entered_this_turn).
    # Multiple sources stack. Matches Bioengineered Future. Consumed in
    # actions.put_into_battlefield.
    grants_etb_counter_per_land_entered_this_turn: bool = False
    # ETB replacement: when this creature would enter the battlefield, if
    # ``state.creatures_died_this_turn > 0`` it enters with one +1/+1
    # counter on it instead. Matches Cackling Slasher's
    # ``This creature enters with a +1/+1 counter on it if a creature
    # died this turn.`` clause. Consumed in
    # ``actions.put_into_battlefield``; the death tracker increments in
    # ``turn.move_off_battlefield`` whenever any creature is graveyarded
    # and resets at cleanup. The check uses ``> 0`` rather than ``>= 1``
    # to defend against any future signed-int regressions, but the
    # tracker is always non-negative.
    enters_with_plus_one_counter_if_creature_died_this_turn: bool = False
    # ETB replacement: when this creature is cast from its controller's
    # hand and resolves onto the battlefield, it enters with N -1/-1
    # counters on it. Matches Patched Plaything's
    # ``This creature enters with two -1/-1 counters on it if you cast
    # it from your hand.`` clause. Consumed in
    # ``actions.put_into_battlefield``; the cast-from-hand origin is
    # recorded on the GameObject via ``cast_via_hand`` at cast time
    # (``actions._cast_spell``). 0 disables the replacement.
    enters_with_minus_one_counters_from_hand_cast: int = 0
    # Aura enchant restriction. Auras default to enchanting creatures;
    # Hardlight Containment enchants artifacts (and our machinery
    # wouldn't find a legal target otherwise). Accepts "creature" or
    # "artifact". Only consumed when is_aura is True.
    aura_enchant_type: str = "creature"
    # Aura target controller restriction — when True, the cast-time
    # target picker for ``is_aura`` cards only offers permanents the
    # caster controls. Mirrors the printed "Enchant creature you
    # control" / "Enchant artifact you control" line — the printed
    # restriction is independent of ``aura_enchant_type``. Default
    # False (the default Scryfall "Enchant creature" wording allows
    # any creature). Sheltered by Ghosts is the first DSK Aura that
    # carries this restriction in the parser pool.
    aura_yours_only: bool = False
    # Dynamic power/toughness formula. When set, overrides the printed
    # power/toughness inside effective_power and effective_toughness.
    # Currently supported:
    #   "cosmogoyf": power = number of cards the controller owns in
    #     exile; toughness = that number + 1.
    #   "harmonious_grovestrider": power = toughness = number of lands
    #     the controller controls.
    #   "emissary_escort": +X/+0 where X is the greatest mana value
    #     among OTHER artifacts the controller controls; base stays
    #     printed (0/4).
    # None = use printed P/T (default).
    dynamic_pt_formula: str | None = None
    # Replacement-effect flag: when any permanent with this flag is on
    # the battlefield under the token-creator's control, the number of
    # tokens created is doubled. Matches Exalted Sunborn.
    is_token_doubler: bool = False
    # Replacement-effect flag: when any permanent with this flag is on
    # the battlefield under the counter-recipient's control, +1/+1
    # counters going onto creatures (and Spacecraft) they control are
    # doubled. Matches Loading Zone. Multiple doublers stack
    # multiplicatively via add_plus_one_counters().
    is_counter_doubler: bool = False
    # Spell-cast restriction: when any permanent with this flag is on
    # the battlefield (regardless of controller), each player may cast
    # at most one noncreature spell per turn. Matches Deafening Silence.
    # Enforced by legal.legal_actions — noncreature CastSpell actions
    # aren't proposed once the caster's
    # Player.noncreature_spells_cast_this_turn is already ≥ 1.
    restricts_noncreature_spells_to_one_per_turn: bool = False
    # Type-change flag: while any permanent with this flag is on the
    # battlefield, every nonbasic land's mana-production colors are
    # overridden to {R} and its activated abilities are suppressed —
    # the Magus of the Moon "Nonbasic lands are Mountains" static.
    # Consumed by game.effective_land_produces and
    # game.effective_activated_abilities. Simplification: subtypes
    # aren't rewritten, and already-animated manlands keep their
    # temp_is_creature state (no layers yet).
    nonbasic_lands_are_mountains: bool = False
    # Dryad of the Ilysian Grove: "Lands you control are every basic
    # land type in addition to their other types." A printing with
    # this flag, while on the battlefield, contributes synthetic
    # layer-4 ``"add_subtype"`` entries (one per basic land type) to
    # ``effective_subtypes`` for every land its controller controls.
    # Synthesizing on-the-fly avoids ETB/leave plumbing; the
    # source's ``etb_timestamp`` is used so within-layer timestamp
    # order vs Blood Moon / Magus of the Moon matches CR 613.4.
    grants_all_basic_land_types_to_lands_you_control: bool = False
    # Urborg, Tomb of Yawgmoth / Yavimaya, Cradle of Growth: "Each
    # land is a [basic land type] in addition to its other types."
    # When this is set on a permanent on the battlefield,
    # ``effective_subtypes`` synthesizes a layer-4 ``"add_subtype"``
    # entry on every land, sourced by this permanent. The CR 613.7
    # dependency rule kicks in here: if the source itself is a
    # land that's been rewritten by a layer-4
    # ``"set_basic_land_subtype"`` effect (Blood Moon / Magus),
    # its ability is wiped at layer 4 (CR 305.6) and the
    # contribution doesn't apply. This is the famous Blood Moon
    # beats Urborg ruling.
    grants_basic_land_type_to_all_lands: "str | None" = None
    # Painter's Servant: "As Painter's Servant enters, choose a
    # color. All cards that aren't on the battlefield, spells, and
    # permanents are the chosen color in addition to their other
    # colors." When True on a permanent's printing,
    # ``effective_colors`` synthesizes a layer-5 ``"add_colors"``
    # entry on every battlefield permanent (and spell) with the
    # source's chosen color. The chosen color is captured at ETB
    # time on ``GameObject.painter_chosen_color``. The source's
    # ``etb_timestamp`` is the synthesized entry's timestamp so it
    # interleaves correctly with Scuttlemutt-style set_colors per
    # CR 613 layer-5 timestamp rules.
    paints_all_objects_chosen_color_at_etb: bool = False
    # Found Footage (Duskmourn): "You may look at face-down creatures
    # your opponents control any time." Today the engine doesn't model
    # hidden information for face-down permanents (the controller can
    # already inspect any face-down GameObject in the battlefield list),
    # so this flag is informational only — it carries the printed
    # static for fidelity but has no engine consequence. Reserved for
    # future hidden-information modeling.
    looks_at_opponents_face_down_creatures: bool = False
    # Humility (Tempest): "All creatures lose all abilities and have
    # base power and toughness 1/1." Decomposed into two flags so a
    # future card with only one half (just the wipe, just the 1/1)
    # composes cleanly:
    # - ``wipes_all_creature_abilities``: synthesizes a layer-6
    #   ``"lose_all_abilities"`` entry on every creature, sourced
    #   by this permanent with its ``etb_timestamp``.
    # - ``sets_all_creature_pt_to_1_1``: synthesizes a layer-7
    #   sublayer-1 ``"set_base_pt"`` entry with ``params=(1, 1)``
    #   on every creature, same source/timestamp.
    # Both contributions interleave with per-creature registry
    # entries (Lignify, anthem grants) by timestamp in the usual
    # CR 613 layer walks.
    wipes_all_creature_abilities: bool = False
    sets_all_creature_pt_to_1_1: bool = False
    # "This creature must be blocked if able." (CR 509.1c). When True,
    # while this creature is attacking, the defending player must
    # assign at least one of their untapped creatures that could
    # legally block it (per ``combat.can_block``) to block it. Enforced
    # in ``actions._declare_blockers``: if no validated blocker is
    # assigned to the attacker and at least one eligible defender
    # exists that could block it, the engine picks one (deterministic
    # — first by id) and adds the assignment. Differs from "menace" /
    # "unblockable" / per-block restrictions: those LIMIT the blocker
    # set; this requirement OBLIGATES at least one block. Matches DSK's
    # Fear of Being Hunted's static rider.
    must_be_blocked_if_able: bool = False
    # Protection from a single color (CR 702.16). When set on a
    # creature's printing, ``combat.can_block`` rejects any block
    # declared by an attacker (or against an attacker) whose
    # effective color includes this one. Matches the classic White
    # Knight ("Protection from black") / Black Knight ("Protection
    # from white") shape. Read via
    # ``game.effective_protection_from_colors`` so future text-/
    # ability-changing layers (Layer 3 color-word rewrites, Layer 6
    # ability grants/removals) compose cleanly. Single color only
    # for now; extend to a frozenset if real cards need protection
    # from multiple colors at once.
    protection_from_color: str | None = None
    # Zone-change replacement: when this card would be put into a
    # graveyard from ANY zone the engine currently models, instead it
    # is shuffled into its owner's library. Consumed by
    # actions._send_to_graveyard_or_library at the spell-resolve,
    # spell-countered, and failed-cast graveyard sites. Matches
    # Nexus of Fate's "from anywhere" clause and Green Sun's Zenith's
    # self-shuffle rider (GSZ's oracle is effectively the same
    # replacement after its tutor effect resolves). Simplification:
    # discard-from-hand and mill-from-library graveyard paths aren't
    # hooked (those aren't hit by any currently-modeled card that
    # targets Nexus / GSZ specifically).
    shuffles_into_library_instead_of_graveyard: bool = False
    # Convoke (CR 702.51): "Your creatures can help cast this spell.
    # Each creature you tap while casting this spell pays for {1} or
    # one mana of that creature's color." Wires the per-cast prompt
    # in ``_cast_spell``: after cost determination (601.2f) and
    # before paying the remaining cost from the mana pool / lands,
    # the controller's agent picks creatures to tap. Each tapped
    # creature pays {1} (or one of its colors) toward the cost. A
    # tapped creature can't separately activate its own
    # ``{T}, sacrifice: Add ...`` mana ability for the same cast,
    # so Convoke and a sac-for-mana ability on the same source are
    # mutually exclusive — matches CR.
    has_convoke: bool = False
    # Draw replacement: while any permanent with this flag is on the
    # battlefield under the drawing player's control AND that player
    # has one or fewer cards in hand, every "draw one or more cards"
    # event for that player draws one extra card. Matches Quantum
    # Riddler. Consumed by game.effective_draw_count and the
    # draw_n_cards helper in turn.py.
    replaces_draw_with_extra_when_hand_at_most_one: bool = False
    # Block restriction: "This creature can block only creatures with
    # flying." Sibling concept to reach (which loosens what *flying*
    # attackers can be blocked by). Matches the EoE Drone token printed
    # on Pinnacle Emissary, Station Monitor, etc. Consumed by
    # combat.can_block — when True on the blocker's printing, the
    # blocker may only declare blocks against attackers with flying.
    can_only_block_creatures_with_flying: bool = False
    # Basic-landcycling. CR 702.29 typecycling: while this card is in
    # the controller's hand, they may pay the cycling cost and discard
    # this card to search their library for a basic-land card of the
    # named subtype, reveal it, put it into their hand, then shuffle.
    # Tuple shape is ``(land_subtype, mana_cost_str)`` — e.g. DSK's
    # Plainscycling {2} cycle prints as ``("Plains", "{2}")``,
    # Islandcycling {2} as ``("Island", "{2}")``, etc. ``None`` when
    # the card has no landcycling.
    #
    # Today this field is **purely declarative**: the parser populates
    # it from the printed ``Plainscycling/Islandcycling/...`` line so
    # the printing accurately reflects oracle text, but the engine
    # does NOT yet enumerate hand-activated abilities, so the
    # cycling cost cannot actually be paid in simulation. Activation
    # support requires (1) ``legal_actions`` to enumerate from-hand
    # ability activations against this field and (2) an
    # ``ActivationCost.discard_self`` cost atom. Both lifts are
    # follow-up work that this batch deliberately scopes out — keeping
    # the printing-level field distinct from the activation pipeline
    # mirrors the precedent set by ``warp_castable_from_graveyard``
    # and ``warp_additional_life_cost`` (declarative metadata stored
    # on the printing; activation logic plumbs separately).
    landcycling: tuple[str, str] | None = None
    # DSK Rooms: a tuple of two ``RoomHalf`` entries when this printing
    # is a Room. Empty for non-Room cards. The Room's overall
    # ``mana_cost`` mirrors one half's cost at print time, but at cast
    # time the caster picks WHICH half to cast (and pays that half's
    # cost). The Room enters with that half ``unlocked``; the other is
    # locked. ``UnlockRoomHalf`` later pays the second half's cost to
    # unlock it. Each half's abilities are also flattened onto this
    # printing's ``triggered_abilities`` / ``static_abilities`` /
    # ``activated_abilities`` with each ability's ``room_half`` field
    # set to the half-name, so the existing iterators (filtered through
    # ``effective_*_abilities``) gate them by ``unlocked_halves``.
    room_halves: tuple[RoomHalf, ...] = ()

    @property
    def is_land(self) -> bool:
        return "Land" in self.card_types

    @property
    def is_creature(self) -> bool:
        return "Creature" in self.card_types

    @property
    def is_basic_land(self) -> bool:
        return self.is_land and "Basic" in self.supertypes

    @property
    def is_legendary(self) -> bool:
        return "Legendary" in self.supertypes

    @property
    def is_token(self) -> bool:
        return "Token" in self.supertypes

    @property
    def is_instant(self) -> bool:
        return "Instant" in self.card_types

    @property
    def is_sorcery(self) -> bool:
        return "Sorcery" in self.card_types

    @property
    def is_artifact(self) -> bool:
        return "Artifact" in self.card_types

    @property
    def is_enchantment(self) -> bool:
        return "Enchantment" in self.card_types

    @property
    def is_spacecraft(self) -> bool:
        return "Spacecraft" in self.subtypes

    @property
    def is_aura(self) -> bool:
        return "Aura" in self.subtypes

    @property
    def is_saga(self) -> bool:
        return self.chapter_count > 0

    @property
    def is_permanent(self) -> bool:
        """Resolves onto the battlefield when cast."""
        return (
            self.is_creature
            or self.is_artifact
            or self.is_land
            or self.is_enchantment
            or "Planeswalker" in self.card_types
        )

    @property
    def is_planeswalker(self) -> bool:
        return "Planeswalker" in self.card_types

    @property
    def is_room(self) -> bool:
        """A Duskmourn Room (Enchantment — Room) — has two halves."""
        return bool(self.room_halves)


_KNOWN_SUPERTYPES = {"Basic", "Legendary", "Snow", "World", "Token"}


def _parse_type_line(
    type_line: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    left, _, right = type_line.partition("—")
    left_words = left.strip().split()
    subtypes = tuple(right.strip().split()) if right else ()
    supertypes = tuple(w for w in left_words if w in _KNOWN_SUPERTYPES)
    card_types = tuple(w for w in left_words if w not in _KNOWN_SUPERTYPES)
    return supertypes, card_types, subtypes


def make_creature(
    name: str,
    cost: str,
    power: int,
    toughness: int,
    keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    triggered_abilities: tuple[TriggeredAbility, ...] = (),
    static_abilities: tuple[StaticAbility, ...] = (),
    activated_abilities: tuple[ActivatedAbility, ...] = (),
    warp: str | None = None,
    legendary: bool = False,
    ward: str | None = None,
    ward_discard_random: bool = False,
    ward_discard_chosen: bool = False,
    grants_additional_land_per_turn: int = 0,
    grants_play_lands_from_graveyard: bool = False,
    extra_etb_trigger: int = 0,
    grants_cast_artifact_from_top_of_library: bool = False,
    grants_food_to_your_artifacts: bool = False,
    grants_food_ability_to_your_artifacts: bool = False,
    grants_blue_mana_ability_to_your_artifacts: bool = False,
    artifact: bool = False,
    cost_reductions: tuple[CostReduction, ...] = (),
    external_cost_reductions: tuple[ExternalCostReduction, ...] = (),
    enters_with_counters_from_x: str | None = None,
    subtypes: tuple[str, ...] = (),
    dynamic_pt_formula: str | None = None,
    is_token_doubler: bool = False,
    devour_land_multiplier: int = 0,
    nonbasic_lands_are_mountains: bool = False,
    grants_all_basic_land_types_to_lands_you_control: bool = False,
    replaces_draw_with_extra_when_hand_at_most_one: bool = False,
    warp_additional_life_cost: int = 0,
    warp_castable_from_graveyard: bool = False,
    enters_as_copy_of_creature: bool = False,
    protection_from_color: str | None = None,
    impending: str | None = None,
    impending_count: int = 0,
    enchantment: bool = False,
    enters_tapped: bool = False,
    must_be_blocked_if_able: bool = False,
    landcycling: tuple[str, str] | None = None,
    enters_with_plus_one_counter_if_creature_died_this_turn: bool = False,
    enters_with_minus_one_counters_from_hand_cast: int = 0,
) -> CardPrinting:
    prefix = "Legendary " if legendary else ""
    artifact_word = "Artifact " if artifact else ""
    enchantment_word = "Enchantment " if enchantment else ""
    subtype_suffix = f" — {' '.join(subtypes)}" if subtypes else ""
    type_line = f"{prefix}{artifact_word}{enchantment_word}Creature{subtype_suffix}"
    supertypes, card_types, parsed_subtypes = _parse_type_line(type_line)
    subtypes = parsed_subtypes
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        power=power,
        toughness=toughness,
        keywords=frozenset(keywords),
        ward_cost=parse_mana_cost(ward) if ward else None,
        ward_cost_str=ward or "",
        ward_discard_random=ward_discard_random,
        ward_discard_chosen=ward_discard_chosen,
        grants_additional_land_per_turn=grants_additional_land_per_turn,
        grants_play_lands_from_graveyard=grants_play_lands_from_graveyard,
        extra_etb_trigger=extra_etb_trigger,
        grants_cast_artifact_from_top_of_library=(
            grants_cast_artifact_from_top_of_library
        ),
        grants_food_to_your_artifacts=grants_food_to_your_artifacts,
        grants_food_ability_to_your_artifacts=(
            grants_food_ability_to_your_artifacts
        ),
        grants_blue_mana_ability_to_your_artifacts=(
            grants_blue_mana_ability_to_your_artifacts
        ),
        triggered_abilities=tuple(triggered_abilities),
        static_abilities=tuple(static_abilities),
        activated_abilities=tuple(activated_abilities),
        warp_cost=parse_mana_cost(warp) if warp else None,
        warp_cost_str=warp or "",
        cost_reductions=tuple(cost_reductions),
        external_cost_reductions=tuple(external_cost_reductions),
        enters_with_counters_from_x=enters_with_counters_from_x,
        dynamic_pt_formula=dynamic_pt_formula,
        is_token_doubler=is_token_doubler,
        devour_land_multiplier=devour_land_multiplier,
        nonbasic_lands_are_mountains=nonbasic_lands_are_mountains,
        grants_all_basic_land_types_to_lands_you_control=(
            grants_all_basic_land_types_to_lands_you_control
        ),
        replaces_draw_with_extra_when_hand_at_most_one=(
            replaces_draw_with_extra_when_hand_at_most_one
        ),
        warp_additional_life_cost=warp_additional_life_cost,
        warp_castable_from_graveyard=warp_castable_from_graveyard,
        enters_as_copy_of_creature=enters_as_copy_of_creature,
        protection_from_color=protection_from_color,
        impending_cost=parse_mana_cost(impending) if impending else None,
        impending_cost_str=impending or "",
        impending_count=impending_count,
        enters_tapped=enters_tapped,
        must_be_blocked_if_able=must_be_blocked_if_able,
        landcycling=landcycling,
        enters_with_plus_one_counter_if_creature_died_this_turn=(
            enters_with_plus_one_counter_if_creature_died_this_turn
        ),
        enters_with_minus_one_counters_from_hand_cast=(
            enters_with_minus_one_counters_from_hand_cast
        ),
    )


def cost(
    mana: str = "",
    tap_self: bool = False,
    sacrifice_self: bool = False,
    remove_plus_one_counter_from_your_creature: bool = False,
) -> ActivationCost:
    """Small constructor for ``ActivationCost`` that handles mana parsing."""
    return ActivationCost(
        remove_plus_one_counter_from_your_creature=(
            remove_plus_one_counter_from_your_creature
        ),
        mana=parse_mana_cost(mana) if mana else None,
        mana_str=mana,
        tap_self=tap_self,
        sacrifice_self=sacrifice_self,
    )


def make_spacecraft(
    name: str,
    cost: str,
    power: int,
    toughness: int,
    station_threshold: int,
    stationed_keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    legendary: bool = False,
    triggered_abilities: tuple[TriggeredAbility, ...] = (),
    static_abilities: tuple[StaticAbility, ...] = (),
    activated_abilities: tuple[ActivatedAbility, ...] = (),
) -> CardPrinting:
    """Build a Spacecraft card (Artifact — Spacecraft).

    Spacecraft enter as non-creature artifacts. Once they accumulate
    ``station_threshold`` charge counters they become artifact creatures
    with the printed ``power``/``toughness`` and ``stationed_keywords``.
    Tier-gated static abilities (like Lumen-Class Frigate's "2+ anthem")
    use StaticAbility(tier_threshold=N).
    """
    prefix = "Legendary " if legendary else ""
    type_line = f"{prefix}Artifact — Spacecraft"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        power=power,
        toughness=toughness,
        station_threshold=station_threshold,
        stationed_keywords=frozenset(stationed_keywords),
        triggered_abilities=tuple(triggered_abilities),
        static_abilities=tuple(static_abilities),
        activated_abilities=tuple(activated_abilities),
    )


def make_instant(
    name: str, cost: str, effect: Effect,
    cost_reductions: tuple[CostReduction, ...] = (),
    kicker: str | None = None,
    additional_cost_choose_your_creature: bool = False,
    additional_cost_sacrifice_creature_or_enchantment: bool = False,
    shuffles_into_library_instead_of_graveyard: bool = False,
    has_convoke: bool = False,
) -> CardPrinting:
    type_line = "Instant"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        spell_effect=effect,
        cost_reductions=tuple(cost_reductions),
        kicker_cost=parse_mana_cost(kicker) if kicker else None,
        kicker_cost_str=kicker or "",
        additional_cost_choose_your_creature=(
            additional_cost_choose_your_creature
        ),
        additional_cost_sacrifice_creature_or_enchantment=(
            additional_cost_sacrifice_creature_or_enchantment
        ),
        shuffles_into_library_instead_of_graveyard=(
            shuffles_into_library_instead_of_graveyard
        ),
        has_convoke=has_convoke,
    )


def make_sorcery(
    name: str, cost: str, effect: Effect,
    cost_reductions: tuple[CostReduction, ...] = (),
    additional_cost_sacrifice_creature_or_artifact: bool = False,
    additional_cost_sacrifice_creature_or_enchantment: bool = False,
    shuffles_into_library_instead_of_graveyard: bool = False,
    has_convoke: bool = False,
) -> CardPrinting:
    type_line = "Sorcery"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        spell_effect=effect,
        cost_reductions=tuple(cost_reductions),
        additional_cost_sacrifice_creature_or_artifact=(
            additional_cost_sacrifice_creature_or_artifact
        ),
        additional_cost_sacrifice_creature_or_enchantment=(
            additional_cost_sacrifice_creature_or_enchantment
        ),
        shuffles_into_library_instead_of_graveyard=(
            shuffles_into_library_instead_of_graveyard
        ),
        has_convoke=has_convoke,
    )


def make_artifact(
    name: str,
    cost: str,
    triggered_abilities: tuple[TriggeredAbility, ...] = (),
    static_abilities: tuple[StaticAbility, ...] = (),
    activated_abilities: tuple[ActivatedAbility, ...] = (),
    legendary: bool = False,
    keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    cost_reductions: tuple[CostReduction, ...] = (),
    paints_all_objects_chosen_color_at_etb: bool = False,
    subtype: str | None = None,
    looks_at_opponents_face_down_creatures: bool = False,
) -> CardPrinting:
    """Build a non-creature, non-Spacecraft Artifact permanent.

    ``subtype`` (e.g. ``"Clue"``, ``"Food"``) appends an ``- <subtype>``
    suffix to the type line and lands on the printing's ``subtypes``
    tuple. Defaults to None (no subtype). DSK Found Footage is the
    first Clue subtype in the engine pool.

    ``looks_at_opponents_face_down_creatures`` mirrors the Duskmourn
    Found Footage static ``You may look at face-down creatures your
    opponents control any time.`` The flag is informational today —
    the engine doesn't model hidden information for face-down
    permanents (a controller can already inspect any face-down
    permanent), so this flag has no engine consequence beyond fidelity
    of the printing's static ability list. Reserved for future hidden-
    information modeling.
    """
    prefix = "Legendary " if legendary else ""
    if subtype is not None:
        type_line = f"{prefix}Artifact — {subtype}"
    else:
        type_line = f"{prefix}Artifact"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        keywords=frozenset(keywords),
        triggered_abilities=tuple(triggered_abilities),
        static_abilities=tuple(static_abilities),
        activated_abilities=tuple(activated_abilities),
        cost_reductions=tuple(cost_reductions),
        paints_all_objects_chosen_color_at_etb=(
            paints_all_objects_chosen_color_at_etb
        ),
        looks_at_opponents_face_down_creatures=(
            looks_at_opponents_face_down_creatures
        ),
    )


def make_equipment(
    name: str,
    cost: str,
    power_bonus: int = 0,
    toughness_bonus: int = 0,
    granted_keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    triggered_abilities: tuple[TriggeredAbility, ...] = (),
    activated_abilities: tuple[ActivatedAbility, ...] = (),
    equip_cost: str | None = None,
    grants_control_opponent_ability_when_equipped: bool = False,
    legendary: bool = False,
) -> CardPrinting:
    """Build an Artifact — Equipment card.

    ``power_bonus``/``toughness_bonus``/``granted_keywords`` define the
    continuous bonus granted to the equipped creature via the Equipment
    subsystem. ``keywords`` are on the Equipment itself (e.g., Flash).

    ``equip_cost`` (e.g. ``"{3}"``) wires the printed "Equip {N}" activated
    ability per rule 702.6: a sorcery-speed activated ability whose cost is
    the printed mana cost and whose effect re-attaches this Equipment to a
    target creature its controller controls. Omit it for Equipment that
    only auto-attaches via an ETB trigger and has no printed Equip cost.
    The synthesized ability is appended after any explicitly supplied
    ``activated_abilities``.
    """
    prefix = "Legendary " if legendary else ""
    type_line = f"{prefix}Artifact — Equipment"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    activated = tuple(activated_abilities)
    if equip_cost is not None:
        activated = activated + (
            ActivatedAbility(
                cost=ActivationCost(
                    mana=parse_mana_cost(equip_cost),
                    mana_str=equip_cost,
                ),
                effect=EquipToTargetCreatureYouControl(),
                sorcery_speed=True,
            ),
        )
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        keywords=frozenset(keywords),
        triggered_abilities=tuple(triggered_abilities),
        activated_abilities=activated,
        grants_control_opponent_ability_when_equipped=(
            grants_control_opponent_ability_when_equipped
        ),
        equipment_bonuses=(
            power_bonus, toughness_bonus, frozenset(granted_keywords)
        ),
    )


def make_aura(
    name: str,
    cost: str,
    power_bonus: int = 0,
    toughness_bonus: int = 0,
    granted_keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    triggered_abilities: tuple[TriggeredAbility, ...] = (),
    aura_enchant_type: str = "creature",
    aura_yours_only: bool = False,
    attached_prevents_untap: bool = False,
    attached_grants_control: bool = False,
    attached_sets_base_pt: "tuple[int, int] | None" = None,
    attached_grants_lose_all_abilities: bool = False,
    attached_grants_subtype: "str | None" = None,
    replaces_first_tokens_with_copies: bool = False,
    attached_grants_ward: str | None = None,
) -> CardPrinting:
    """Build an Enchantment — Aura card that attaches to a creature.

    ``power_bonus``/``toughness_bonus``/``granted_keywords`` define the
    continuous bonus granted to the enchanted creature; we reuse the
    Equipment bonus machinery (equipment_bonuses) so effective_* helpers
    pick them up automatically.

    ``keywords`` are on the Aura ITSELF (e.g. Flash on Frantic Strength /
    Shardmage's Rescue / Growing Dread). Mirrors ``make_equipment``'s
    same-named kwarg and routes through the same ``CardPrinting.keywords``
    field that gates instant-speed casting in ``legal.py``.

    ``attached_grants_ward`` (e.g. ``"{1}"``) wires the "Enchanted
    permanent has ward {cost}" rider — Hardlight Containment.

    Simplified Aura behavior: cast-time target must be a creature; on
    resolution the Aura enters attached to that target (or goes to the
    graveyard if the target is no longer legal). When the enchanted
    creature leaves the battlefield, the Aura is sent to the graveyard
    (unlike Equipment, which falls off but stays).
    """
    type_line = "Enchantment — Aura"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        triggered_abilities=tuple(triggered_abilities),
        keywords=frozenset(keywords),
        equipment_bonuses=(
            power_bonus, toughness_bonus, frozenset(granted_keywords)
        ),
        aura_enchant_type=aura_enchant_type,
        aura_yours_only=aura_yours_only,
        attached_prevents_untap=attached_prevents_untap,
        attached_grants_control=attached_grants_control,
        attached_sets_base_pt=attached_sets_base_pt,
        attached_grants_lose_all_abilities=attached_grants_lose_all_abilities,
        attached_grants_subtype=attached_grants_subtype,
        replaces_first_tokens_with_copies=replaces_first_tokens_with_copies,
        attached_grants_ward_cost=(
            parse_mana_cost(attached_grants_ward)
            if attached_grants_ward else None
        ),
        attached_grants_ward_str=attached_grants_ward or "",
    )


def make_planeswalker(
    name: str,
    cost: str,
    loyalty: int,
    subtype: str,
    triggered_abilities: tuple[TriggeredAbility, ...] = (),
    static_abilities: tuple[StaticAbility, ...] = (),
    activated_abilities: tuple[ActivatedAbility, ...] = (),
    legendary: bool = True,
) -> CardPrinting:
    """Build a Legendary Planeswalker card. Loyalty is the printed
    starting loyalty; it initializes ``GameObject.loyalty_counters``
    when the planeswalker enters the battlefield. SBAs send it to
    graveyard at 0 loyalty.

    Simplifications: loyalty-activated abilities aren't modeled yet
    (Tezzeret's 0/−3/−7 abilities are dropped); damage redirection
    to planeswalkers isn't modeled; attacking a planeswalker isn't
    modeled. Triggered abilities on the walker still fire normally.
    """
    prefix = "Legendary " if legendary else ""
    type_line = f"{prefix}Planeswalker — {subtype}"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        triggered_abilities=tuple(triggered_abilities),
        static_abilities=tuple(static_abilities),
        activated_abilities=tuple(activated_abilities),
        loyalty=loyalty,
    )


def make_enchantment(
    name: str,
    cost: str,
    triggered_abilities: tuple[TriggeredAbility, ...] = (),
    static_abilities: tuple[StaticAbility, ...] = (),
    activated_abilities: tuple[ActivatedAbility, ...] = (),
    legendary: bool = False,
    is_counter_doubler: bool = False,
    is_token_doubler: bool = False,
    restricts_noncreature_spells_to_one_per_turn: bool = False,
    grants_first_spell_each_turn_free: bool = False,
    warp: str | None = None,
    grants_etb_counter_per_land_entered_this_turn: bool = False,
    nonbasic_lands_are_mountains: bool = False,
    wipes_all_creature_abilities: bool = False,
    sets_all_creature_pt_to_1_1: bool = False,
    ward: str | None = None,
) -> CardPrinting:
    """Build a non-Aura Enchantment permanent. Casts onto the battlefield
    like any other permanent and sits there firing triggers or providing
    static bonuses. Auras (enchant creature) aren't modeled yet.

    ``ward`` (e.g. ``"{2}"``) wires the printed Ward keyword on the
    enchantment ITSELF (Trapped in the Screen). Mirrors the same-named
    kwarg on ``make_creature`` and routes through the same
    ``CardPrinting.ward_cost`` / ``ward_cost_str`` fields the engine's
    ward-check helpers (``effective_ward_cost``,
    ``_apply_ward_check``) already read. Auras carry their own
    ``attached_grants_ward`` kwarg on ``make_aura`` for the "Enchanted
    permanent has ward" rider; that targets the enchanted permanent
    rather than the Aura itself, so the two kwargs stay distinct.
    """
    prefix = "Legendary " if legendary else ""
    type_line = f"{prefix}Enchantment"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str=cost,
        mana_cost=parse_mana_cost(cost),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        triggered_abilities=tuple(triggered_abilities),
        static_abilities=tuple(static_abilities),
        activated_abilities=tuple(activated_abilities),
        is_counter_doubler=is_counter_doubler,
        is_token_doubler=is_token_doubler,
        restricts_noncreature_spells_to_one_per_turn=(
            restricts_noncreature_spells_to_one_per_turn
        ),
        grants_first_spell_each_turn_free=(
            grants_first_spell_each_turn_free
        ),
        warp_cost=parse_mana_cost(warp) if warp else None,
        warp_cost_str=warp or "",
        grants_etb_counter_per_land_entered_this_turn=(
            grants_etb_counter_per_land_entered_this_turn
        ),
        nonbasic_lands_are_mountains=nonbasic_lands_are_mountains,
        wipes_all_creature_abilities=wipes_all_creature_abilities,
        sets_all_creature_pt_to_1_1=sets_all_creature_pt_to_1_1,
        ward_cost=parse_mana_cost(ward) if ward else None,
        ward_cost_str=ward or "",
    )


def make_room(
    half_a: RoomHalf,
    half_b: RoomHalf,
    legendary: bool = False,
) -> CardPrinting:
    """Build a Duskmourn Room card (``Enchantment — Room // Enchantment — Room``).

    The combined-card name is ``"<A.name> // <B.name>"`` (Scryfall
    convention). The printing-level ``mana_cost`` and ``mana_cost_str``
    are the SUM of the two halves' costs — used only for things like
    ``cmc`` reporting; cast time uses the chosen half's cost via
    ``CastSpell``-with-half resolution. Each half's abilities are
    flattened onto the printing's ``triggered_abilities`` /
    ``static_abilities`` / ``activated_abilities`` tuples with
    ``room_half`` tagged so the engine's effective-ability iterators
    can gate them on ``unlocked_halves``.

    Simplification: the cast flow takes the controller's chosen half's
    cost; the half-choice is wired in ``_cast_spell`` via the
    ``cast_room_half`` argument on ``CastSpell``.
    """
    name = f"{half_a.name} // {half_b.name}"
    prefix = "Legendary " if legendary else ""
    type_line = f"{prefix}Enchantment — Room"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    # Tag each half's abilities with its ``room_half`` name and flatten
    # them onto the parent printing. Use ``dataclasses.replace`` so the
    # frozen dataclass instances on the half (which the half itself
    # also retains, untagged) get a tagged copy on the parent.
    from dataclasses import replace as _dc_replace

    triggered: list[TriggeredAbility] = []
    static: list[StaticAbility] = []
    activated: list[ActivatedAbility] = []
    for half in (half_a, half_b):
        for t in half.triggered_abilities:
            triggered.append(_dc_replace(t, room_half=half.name))
        for s in half.static_abilities:
            static.append(_dc_replace(s, room_half=half.name))
        for a in half.activated_abilities:
            activated.append(_dc_replace(a, room_half=half.name))
    # Combined mana cost: stored as a string of the two halves joined
    # by ``//`` (matches Scryfall's printed form). The parsed
    # ``mana_cost`` uses the COMBINED cost (sum of the two halves) so
    # any caller that asks for printing.mana_cost.total gets the
    # full-card mana value. Cast-time payment doesn't touch this
    # printing-level cost — it uses the chosen half's cost directly.
    combined_str = f"{half_a.mana_cost_str} // {half_b.mana_cost_str}"
    summed_generic = half_a.mana_cost.generic + half_b.mana_cost.generic
    summed_colored: dict[str, int] = {}
    for color, n in half_a.mana_cost.colored:
        summed_colored[color] = summed_colored.get(color, 0) + n
    for color, n in half_b.mana_cost.colored:
        summed_colored[color] = summed_colored.get(color, 0) + n
    summed_cost = ManaCost(
        generic=summed_generic,
        colored=tuple(sorted(summed_colored.items())),
    )
    return CardPrinting(
        name=name,
        mana_cost_str=combined_str,
        mana_cost=summed_cost,
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        triggered_abilities=tuple(triggered),
        static_abilities=tuple(static),
        activated_abilities=tuple(activated),
        room_halves=(half_a, half_b),
    )


def make_token_creature(
    name: str,
    power: int,
    toughness: int,
    subtypes: tuple[str, ...] = (),
    keywords: frozenset[str] | set[str] | tuple[str, ...] = (),
    artifact: bool = False,
    activated_abilities: tuple[ActivatedAbility, ...] = (),
    can_only_block_creatures_with_flying: bool = False,
) -> CardPrinting:
    """Build a CardPrinting representing a creature token.

    Tokens have no mana cost; they're created by effects rather than cast.
    The "Token" supertype distinguishes them so the engine can remove them
    from any non-battlefield zone per MTG rule 704.5d. Setting
    ``artifact=True`` adds the Artifact card type (e.g. for EoE's colorless
    Robot tokens). ``can_only_block_creatures_with_flying`` plumbs the
    block-restriction flag onto the resulting printing — used by the EoE
    Drone token.
    """
    subtypes_str = " ".join(subtypes)
    extra_type = "Artifact " if artifact else ""
    type_line = (
        f"Token {extra_type}Creature — {subtypes_str}"
        if subtypes else f"Token {extra_type}Creature"
    )
    supertypes, card_types, subtypes_tup = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes_tup,
        power=power,
        toughness=toughness,
        keywords=frozenset(keywords),
        activated_abilities=tuple(activated_abilities),
        can_only_block_creatures_with_flying=(
            can_only_block_creatures_with_flying
        ),
    )


def make_planet_land(
    name: str, color: str,
) -> CardPrinting:
    """EoE Planet land: enters tapped, taps for a single color.
    Simplification: drops the Station mechanic and all tiered abilities
    (Planets with Station are functionally a non-creature Spacecraft —
    not modeled for Planets here).
    """
    type_line = f"Land — Planet"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        produces=frozenset({color}),
        enters_tapped=True,
    )


def make_colorless_land(
    name: str, subtypes_str: str = "", legendary: bool = False,
) -> CardPrinting:
    """A non-basic land that taps for {C} only. Useful for utility
    lands whose activated abilities aren't modeled — taps-for-generic
    still works via auto_tap_for_cost's generic-mana fallback.
    """
    prefix = "Legendary " if legendary else ""
    type_line = (
        f"{prefix}Land — {subtypes_str}" if subtypes_str else f"{prefix}Land"
    )
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        produces=frozenset(),  # generic-only
    )


def make_utility_land(
    name: str,
    produces: "frozenset[str]" = frozenset(),
    subtypes_str: str = "",
    legendary: bool = False,
    triggered_abilities: tuple = (),
    activated_abilities: tuple = (),
    enters_tapped: bool = False,
    keywords: "frozenset[str] | set[str] | tuple[str, ...]" = (),
    optional_copy_land_from_graveyard: bool = False,
    enters_tapped_unless_any_player_life_at_most: int = 0,
) -> CardPrinting:
    """Build a non-basic land with arbitrary activated/triggered
    abilities and an explicit produces set. Used by Stellar Sights."""
    prefix = "Legendary " if legendary else ""
    type_line = (
        f"{prefix}Land — {subtypes_str}" if subtypes_str else f"{prefix}Land"
    )
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        produces=frozenset(produces),
        triggered_abilities=tuple(triggered_abilities),
        activated_abilities=tuple(activated_abilities),
        enters_tapped=enters_tapped,
        keywords=frozenset(keywords),
        optional_copy_land_from_graveyard=optional_copy_land_from_graveyard,
        enters_tapped_unless_any_player_life_at_most=(
            enters_tapped_unless_any_player_life_at_most
        ),
    )


def make_shock_land(
    name: str, colors: tuple[str, str], subtypes_str: str,
    life_cost: int = 2,
) -> CardPrinting:
    """Build a shock-land-style dual. ``subtypes_str`` is the type-line
    suffix (e.g. "Forest Island"). The land produces both ``colors`` and
    enters tapped unless the controller pays ``life_cost`` life on ETB.
    """
    type_line = f"Land — {subtypes_str}"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        produces=frozenset(colors),
        shock_land_life_cost=life_cost,
    )


def make_basic_land(name: str, color: str) -> CardPrinting:
    type_line = f"Basic Land — {name}"
    supertypes, card_types, subtypes = _parse_type_line(type_line)
    return CardPrinting(
        name=name,
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line=type_line,
        supertypes=supertypes,
        card_types=card_types,
        subtypes=subtypes,
        produces=frozenset({color}),
    )


# Pre-built activated ability granted to artifacts by Ragost, Deft
# Gastronaut's static "{2}, {T}, Sacrifice this artifact: You gain 3
# life." Exposed here so game.effective_activated_abilities can append
# it without each call-site re-importing effect classes.
RAGOST_FOOD_GRANTED_ABILITY = ActivatedAbility(
    cost=ActivationCost(
        mana=parse_mana_cost("{2}"),
        mana_str="{2}",
        tap_self=True,
        sacrifice_self=True,
    ),
    effect=GainLife(amount=3),
)


# Pre-built activated ability granted to artifacts by Mm'menon, the
# Right Hand's static: "Artifacts you control have '{T}: Add {U}.
# Spend this mana only to cast a spell from anywhere other than your
# hand.'" The "outside-of-hand" spend restriction is enforced via the
# ``"cast_from_outside_hand"`` coin tag — see
# ``actions._coin_satisfies_context``.
MMNEMON_BLUE_MANA_GRANTED_ABILITY = ActivatedAbility(
    cost=ActivationCost(tap_self=True),
    effect=AddManaToPool(
        produces=(("U", 1),),
        spend_restriction="cast_from_outside_hand",
    ),
    is_mana_ability=True,
)


# Pre-built activated ability granted to the equipped creature by
# The Dominion Bracelet: "{15}, Exile The Dominion Bracelet: You
# control target opponent during their next turn. This ability costs
# {X} less to activate, where X is this creature's power. Activate
# only as a sorcery."
DOMINION_BRACELET_GRANTED_ABILITY = ActivatedAbility(
    cost=ActivationCost(
        mana=parse_mana_cost("{15}"),
        mana_str="{15}",
        exile_granting_equipment=True,
        mana_reduced_by_source_power=True,
    ),
    effect=ControlTargetOpponentNextTurn(),
    sorcery_speed=True,
)
