"""Unified effect grammar shared by creature triggered abilities and
spells.

Each shape's regex matches the *normalized* effect phrase — no
source-subject prefix (``it`` for creature triggers or ``<Name>``
for spells), no trailing period, first letter lowercased. Surface
preprocessing (the ``_normalize_*`` helpers below) strips the
per-context noise before dispatch, so both contexts share one
effect grammar — adding a new effect here makes it available to
creature triggered abilities and to spells (single-line and
multi-line) at the same time.
"""

from __future__ import annotations

import re

from mtg_bot.cards.base import (
    AddManaToPool,
    CounterTargetSpell,
    CreateLanderAndScheduleSacrifice,
    CreateToken,
    CreateTokenPerMulticoloredPermanent,
    DealDamageAnyTarget,
    DealDamageEachOpponent,
    DealDamageEqualToSourcePowerAnyTarget,
    DealDamageToOpponentCreature,
    DealDamageToOpponentCreatureByArtifactCount,
    DealDamageToOpponentCreatureByCardsInHand,
    DealDamageToTargetCreature,
    DealDamageToTargetCreatureSpacecraftBoosted,
    DealDamageToTargetCreatureWithExcessLanderRider,
    DealDamageToTargetOpponent,
    DealDamageToTargetTappedCreature,
    DealNDamageToEachCreature,
    DestroyAllCreatures,
    DestroyTargetArtifact,
    DestroyTargetArtifactEnchantmentOrFlyingCreature,
    DestroyTargetArtifactOrTappedCreature,
    DestroyTargetCreature,
    DestroyTargetLand,
    DestroyTargetNoncreaturePermanent,
    DestroyTargetNonlandPermanent,
    DiscardHandThenDrawN,
    DiscardSelf,
    DoubleTargetCreaturesPowerUEOT,
    DrawCardPerMulticoloredPermanent,
    DrawCards,
    DrawCardsEqualToResolvingCombatDamage,
    EachOpponentDiscards,
    EachOpponentLosesLife,
    EachOpponentMillsCards,
    EachOpponentSacrificesCreature,
    EachPlayerDraws,
    Effect,
    EnduringDiesReturnAsEnchantmentEffect,
    ExileAllArtifactsWithManaValueAtMost,
    ExileAllCreaturesAndSpacecraft,
    ExileEachOpponentsGraveyard,
    ExileTargetCardFromGraveyard,
    ExileTargetCreature,
    ExorciseEffect,
    FocusFireEffect,
    GainLife,
    GrantUnblockableUEOT,
    LookAtTopNRevealOrCounter,
    LookAtTopTwoSplitHandGraveyard,
    LoseLife,
    ManifestDread,
    ManifestDreadThenPutPlusOneCountersOnIt,
    MillTargetPlayer,
    MoveSelfCountersToTargetCreatureYouControl,
    OptionalDiscardCardThenDrawN,
    OptionalManaCostThenEffect,
    OptionalPutCreatureFromHandOntoBattlefield,
    OptionalPutLandFromHandOntoBattlefield,
    OptionalSacAnotherForEffect,
    OptionalSacAnotherForPumpSelfUEOT,
    OptionalSacLandOrLanderForBasicLandRamp,
    OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary,
    PreventAllCombatDamageThisTurn,
    PumpCreaturesOpponentsControlUntilEOT,
    PumpCreaturesYouControlUntilEOT,
    PumpSelfUntilEOT,
    PumpTargetArtifactOrCreatureUntilEOT,
    PumpTargetCreatureUntilEOT,
    PumpTargetOtherCreatureByXOverXWhereXIsSourcePower,
    PutCardFromGraveyardOnBottomOfLibrary,
    PutChargeCountersOnTargetSpacecraftOrPlanetYouControl,
    PutStunCounterOnTappedCreature,
    PutPlusOneCounterOnCreatureYouControl,
    PutPlusOneCounterOnEachCreatureYouControl,
    PutPlusOneCounterOnEachOfUpToNTargetCreatures,
    PutPlusOneCountersOnEachCreatureYouControlEqualToSourcePower,
    PutPlusOneCounterOnOpponentsCreature,
    PutLoyaltyCounterOnSelf,
    PutPlusOneCounterOnSelf,
    PutPlusOneCounterOnTargetCreature,
    ReturnAllCreaturesFromYourGraveyardToBattlefield,
    ReturnCreatureFromYourGraveyardToBattlefield,
    ReturnCreatureFromYourGraveyardToHand,
    ReturnTargetArtifactCardFromYourGraveyardToHand,
    ReturnTargetCreatureToOwnersHand,
    ReturnTargetPermanentYouControlToOwnersHand,
    ReturnUpToNTargetCreaturesToOwnersHands,
    SearchLibraryForBasicLandTapped,
    SearchLibraryForCard,
    SearchLibraryForEquipmentToBattlefield,
    SeedshipImpactEffect,
    SelfMill,
    Sequence,
    SingularityRuptureEffect,
    Surveil,
    TapAndStunUpToOneTargetCreature,
    TapAnotherTargetCreature,
    TapTargetArtifactOrCreature,
    TapTargetOpponentPermanentAndStunIt,
    TargetOpponentLosesLife,
    UntapTargetOtherPermanent,
)
from mtg_bot.cards.parser.keywords import _KEYWORD_LINES
from mtg_bot.cards.tokens import TOKEN_PRINTINGS


# Spelled-out small-cardinal lookup for oracle phrasings like
# ``mill three cards``. Scryfall consistently spells out 1–10 as words
# in mill-style effects (e.g. "mill three cards" on Lithic Hierograph).
# Larger amounts use digits, so the table tops out at ten — extend if a
# future card ships an oracle line with "mill twelve cards".
_SMALL_CARDINAL_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_SMALL_CARDINAL_PATTERN = "|".join(_SMALL_CARDINAL_WORDS)


# ``manifest dread`` (CR 701.55) and its repeated forms ``manifest
# dread N times`` / ``manifest dread twice`` / ``manifest dread three
# times`` / ``manifest dread X times`` — DSK's Phase 3 mechanic. The
# bare form (no count tail) maps to ``ManifestDread(count=1)``; the
# spelled-out / digit forms map to ``ManifestDread(count=N)``; and the
# ``X times`` form maps to ``ManifestDread(count=None)`` so the engine
# class reads ``state.resolving_x_value`` at resolve time. The
# ``twice`` synonym short-circuits to amount 2, matching the same
# convention used for repeated triggers elsewhere in this grammar.
_MANIFEST_DREAD = re.compile(
    r"^manifest dread"
    r"(?:"
    r" (twice)"  # bare ``twice`` (no trailing ``times``)
    r"|"
    r" (X|" + _SMALL_CARDINAL_PATTERN + r"|\d+) times"
    r")?$"
)


# "Prevent all combat damage that would be dealt this turn." — bare
# global static spell effect. Maps to ``PreventAllCombatDamageThisTurn``
# whose ``resolve`` flips ``state.prevent_combat_damage_this_turn`` and
# the engine's combat damage step honors at apply time. Currently only
# Darkness in the EoE pool today, but the surface form is a parameterless
# whole-sentence shape so it lives in the per-clause grammar (rather
# than the whole-spell exact-text tier) per the parser's
# generic-templates-over-literals guidance — adding a future card with
# a similar parameterless global effect lands its own neighboring
# recognizer here.
_PREVENT_ALL_COMBAT_DAMAGE_THIS_TURN = re.compile(
    r"^prevent all combat damage that would be dealt this turn$"
)


# ``manifest dread, then put a +1/+1 counter on that creature`` (and the
# spelled-out / digit ``N +1/+1 counters`` plural variant) — a single
# coupled primitive in the engine (``ManifestDreadThenPutPlusOneCounters
# OnIt``) where ``that creature`` refers to the just-manifested face-
# down 2/2 (NOT a chosen target). The two halves can't be parsed as a
# compound ``Sequence`` because the second half's ``that creature``
# pronoun is anaphoric to the manifest's output, which the splitter
# can't reconstruct from per-piece parses; instead the whole phrase
# matches as one regex and lands on the dedicated engine class. Slimy
# Aquarium (``a +1/+1 counter`` → counter_amount=1) and Weight Room
# (``three +1/+1 counters`` → counter_amount=3) are the EoE/Duskmourn
# Room printings that ship this surface; the singular/plural arms are
# a single regex via ``a |N `` alternation so a future printing with
# a different counter amount lands on the same primitive without
# further grammar work.
_MANIFEST_DREAD_THEN_PUT_PLUS_ONE_COUNTERS_ON_THAT_CREATURE = re.compile(
    rf"^manifest dread, then put "
    rf"(?:a \+1/\+1 counter|"
    rf"(\d+|{_SMALL_CARDINAL_PATTERN}) \+1/\+1 counters)"
    rf" on that creature$"
)


_DEAL_DAMAGE_ANY_TARGET = re.compile(
    r"^deals (\d+) damage to any target$"
)
# "<source> deals damage equal to its power to any target" — Infernal
# Phantom's dies trigger body in DSK (after the ``it `` source-subject
# strip in ``_normalize_creature_effect_phrase``). Maps to
# ``DealDamageEqualToSourcePowerAnyTarget`` whose resolve reads the
# source's effective power at fire time and routes through
# ``_apply_damage`` against the chosen target. Distinct from the bare
# ``deals N damage to any target`` shape above (which carries a literal
# integer amount); this primitive uses the source's power as a dynamic
# integer. Same legal-target pool either way.
_DEAL_DAMAGE_EQUAL_TO_SOURCE_POWER_ANY_TARGET = re.compile(
    r"^deals damage equal to its power to any target$"
)
_DEAL_DAMAGE_TARGET_CREATURE = re.compile(
    # Optional ``it `` prefix accepts the activated-ability surface
    # where the source is referred to by pronoun — Bear Trap's
    # ``It deals 3 damage to target creature.`` body uses ``It`` for
    # the source artifact, and the activated normalizer lowercases
    # the first letter but doesn't strip the pronoun (unlike the
    # creature-trigger normalizer's ``it ``-strip). Same shape as
    # the ``(?:this creature )?`` prefix on
    # ``_DEAL_DAMAGE_TARGET_OPPONENT`` / ``_DEAL_N_DAMAGE_EACH_OPPONENT`` /
    # ``_DEAL_N_DAMAGE_EACH_CREATURE``.
    r"^(?:it )?deals (\d+) damage to target creature$"
)
# "<source> deals N damage to up to one target creature" — Pinnacle
# Kill-Ship's ETB body (``it deals 10 damage to up to one target
# creature``). Mirror of the bare ``deals N damage to target creature``
# shape with the ``up to one`` qualifier promoting the target count
# range from 1..1 to 0..1; maps to ``DealDamageToTargetCreature(amount=N,
# optional=True)`` whose engine class flips ``requires_target`` /
# ``target_count_range`` accordingly and resolves as a no-op when no
# target is chosen. The ``it `` prefix is the same source-pronoun the
# bare regex accepts; the creature-trigger normalizer doesn't strip it
# from this phrase because the phrase starts with the pronoun (the
# ``deals`` verb is right after).
_DEAL_DAMAGE_UP_TO_ONE_TARGET_CREATURE = re.compile(
    r"^(?:it )?deals (\d+) damage to up to one target creature$"
)
# "<source> deals N damage to target creature an opponent controls" —
# Nova Hellkite's ETB. Maps to ``DealDamageToOpponentCreature(amount=N)``
# whose ``legal_targets`` filters to creatures NOT controlled by the
# resolving controller. The ``deals`` form (no ``it`` / ``you`` subject)
# is what survives both creature-trigger normalization (the ``it ``
# prefix-strip) and spell normalization (the ``<Name> `` prefix-strip).
# Tried BEFORE the bare ``deals N damage to target creature`` regex
# which has a ``$`` anchor — phrases that end at ``creature`` with no
# qualifier match the bare form, while phrases with the ``an opponent
# controls`` tail land here.
#
# The optional ``this Room `` prefix accepts the surface where a
# Duskmourn Room half names itself as the damage source — Glassworks'
# door trigger prints ``this Room deals 4 damage to target creature an
# opponent controls`` (the ``this Room`` self-reference is the Room
# equivalent of the ``this creature`` self-reference on creature-source
# triggers). Same legal-target pool either way; the source identity
# check happens through the trigger's ``IsSelf()`` predicate, not the
# effect body.
_DEAL_DAMAGE_TARGET_OPPONENT_CREATURE = re.compile(
    r"^(?:this Room )?deals (\d+) damage to target creature "
    r"an opponent controls$"
)
# "[this Room ]deals damage equal to <count-source> to target creature
# an opponent controls" — Warmaker Gunship (artifacts you control,
# from EoE) and Roaring Furnace (cards in your hand, from DSK Room
# half). Parameterized over the count-source phrase via a captured
# group; the ``_DEAL_DAMAGE_COUNT_SOURCE_DISPATCH`` mapping below
# routes the captured surface text to the matching engine class.
# Same legal-target pool (creatures NOT controlled by the resolving
# controller) as the bare ``deals N damage to target creature an
# opponent controls`` shape above; differs only in the dynamic
# damage amount tied to a runtime count instead of a printed integer.
#
# The optional ``this Room `` prefix accepts the Duskmourn Room-half
# surface where the source names itself as ``this Room`` (mirrors the
# ``(?:this Room )?`` precedent on the integer-amount sibling above).
# Same source-identity check via the trigger's ``IsSelf()`` predicate;
# the prefix is purely surface noise that the regex consumes.
_DEAL_DAMAGE_BY_COUNT_TARGET_OPPONENT_CREATURE = re.compile(
    r"^(?:this Room )?deals damage equal to the number of "
    r"(artifacts you control|cards in your hand) "
    r"to target creature an opponent controls$"
)
# Maps the captured count-source surface text to its engine class.
# Adding a new count source (e.g. ``enchantments you control``) is
# one entry here plus extending the regex's alternation above — no
# new dispatch site required. Both engine classes share the
# ``target creature an opponent controls`` legal-target filter; they
# differ only in WHICH count drives the damage amount at resolve
# time (artifacts on the battlefield vs cards in hand).
_DEAL_DAMAGE_COUNT_SOURCE_DISPATCH: dict[str, type] = {
    "artifacts you control": DealDamageToOpponentCreatureByArtifactCount,
    "cards in your hand": DealDamageToOpponentCreatureByCardsInHand,
}
# "<source> deals X damage to target attacking or blocking creature,
# where X is N plus the number of creatures and/or Spacecraft you
# control." — Focus Fire `{R}` Sorcery: variable-X burn whose X is a
# fixed integer base plus a count of friendly creatures and Spacecraft
# at resolve time. The capture group is the integer base ``N`` (matches
# Focus Fire's printed ``2``); maps to ``FocusFireEffect(base=N)``
# whose ``_amount`` walks ``state.battlefield`` for friendly creatures
# and Spacecraft at resolve time.
#
# Currently only Focus Fire in the EoE pool fits this shape, but the
# regex captures the integer base as a parameter so a future printing
# with a different base (``where X is 3 plus ...`` etc.) lands on the
# same primitive without further grammar work — the
# parameterized-recognizer-preferred form per parser CLAUDE.md.
# Lives in the per-clause grammar (rather than the whole-spell exact-
# text tier) to claim future cards with this shape automatically.
_DEAL_DAMAGE_X_BASE_PLUS_FRIENDLY_CREATURE_OR_SPACECRAFT_COUNT = re.compile(
    r"^deals X damage to target attacking or blocking creature, "
    r"where X is (\d+) plus the number of creatures and/or "
    r"Spacecraft you control$"
)
# "<source> deals N damage to target creature. If excess damage was
# dealt this way, create a Lander token." — Orbital Plunge `{3}{R}`
# Sorcery: a fixed-amount burn whose Lander-creation rider fires only
# when the damage exceeded the target's lethal threshold (overkill).
# Two-sentence shape: the leading sentence is the bare damage; the
# trailing ``If excess damage was dealt this way, create a Lander
# token.`` clause is an anaphoric rider that doesn't parse as a
# standalone effect (no per-clause primitive recognizes the back-
# reference to the damage just dealt). The recognizer matches the
# unsplit two-sentence body verbatim — ``_parse_spell_oracle_text``
# tries the unsplit clause first via ``_parse_spell_effect_phrase``
# before falling through to the per-sentence splitter, so this regex
# claims Orbital Plunge before the splitter can carve the rider into
# an unparseable piece.
#
# The capture group is the integer damage amount ``N`` (matches
# Orbital Plunge's printed ``6``); maps to
# ``DealDamageToTargetCreatureWithExcessLanderRider(amount=N)`` whose
# resolve step deals the damage and gates Lander creation on the
# excess-damage check at resolve time. Currently only Orbital Plunge
# in the EoE pool fits this shape, but the regex captures the integer
# amount as a parameter so a future printing with a different damage
# value lands on the same primitive without further grammar work —
# the parameterized-recognizer-preferred form per parser CLAUDE.md.
_DEAL_DAMAGE_N_TARGET_CREATURE_WITH_EXCESS_LANDER_RIDER = re.compile(
    r"^deals (\d+) damage to target creature\. "
    r"If excess damage was dealt this way, create a Lander token$"
)
# "<source> deals N damage to target creature. It deals M damage
# instead if you control a Spacecraft." — Invasive Maneuvers `{1}{R}`
# Instant: a fixed-amount burn with a Spacecraft-state boost rider.
# Two-sentence shape: the leading sentence is the bare damage; the
# trailing ``It deals M damage instead if you control a Spacecraft.``
# clause is an anaphoric rider that doesn't parse as a standalone
# effect (the ``It deals M damage instead`` substitution-syntax has no
# per-clause primitive). The recognizer matches the unsplit two-
# sentence body verbatim — ``_parse_spell_oracle_text`` tries the
# unsplit clause first via ``_parse_spell_effect_phrase`` before
# falling through to the per-sentence splitter, so this regex claims
# Invasive Maneuvers before the splitter can carve the rider into an
# unparseable piece.
#
# Two capture groups: the base damage ``N`` (matches Invasive
# Maneuvers's printed ``3``) and the boosted damage ``M`` (matches
# the printed ``5``); maps to
# ``DealDamageToTargetCreatureSpacecraftBoosted(base=N, boosted=M)``
# whose resolve step picks ``base`` vs ``boosted`` by inspecting the
# controller's battlefield for a Spacecraft. Currently only Invasive
# Maneuvers in the EoE pool fits this shape, but the regex captures
# both integer amounts as parameters so a future printing with
# different base/boosted values lands on the same primitive without
# further grammar work — the parameterized-recognizer-preferred form
# per parser CLAUDE.md.
_DEAL_DAMAGE_N_TARGET_CREATURE_M_IF_SPACECRAFT = re.compile(
    r"^deals (\d+) damage to target creature\. "
    r"It deals (\d+) damage instead if you control a Spacecraft$"
)
# "<source> deals N damage to target opponent" — Biotech Specialist's
# sacrifice-an-artifact trigger body. Damage targets a player rather
# than a creature; ``DealDamageToTargetOpponent(amount=N)`` filters
# the legal-target pool to opponents (not the source's controller).
#
# The optional ``this creature `` prefix accepts the surface where the
# source is named explicitly inside the trigger body. Scryfall prints
# this form when the trigger's CONDITION already uses ``you`` as the
# subject (Biotech Specialist's ``Whenever you sacrifice an artifact``
# leads into ``this creature deals 2 damage to target opponent`` — the
# explicit ``this creature`` resolves what would otherwise be an
# ambiguous ``it`` reference). The
# ``_normalize_creature_effect_phrase`` strip only consumes ``it ``,
# so we accept the longer subject form here at the regex level
# rather than expanding the normalizer (which would also strip
# ``this creature`` from phrases where it's a self-object reference
# in primitives like ``put a +1/+1 counter on this creature``).
#
# Tried BEFORE the bare ``deals N damage to any target`` regex
# because the ``$`` anchor on the bare regex would not match the
# ``target opponent`` tail anyway, but ordering more-specific before
# more-general is the convention in this dispatcher.
_DEAL_DAMAGE_TARGET_OPPONENT = re.compile(
    r"^(?:this creature )?deals (\d+) damage to target opponent$"
)
# Controller draws cards. The optional ``you `` prefix surfaces in
# compound bodies like ``you draw a card and lose 1 life`` (Voidforged
# Titan, Decode Transmissions, Elegy Acolyte) — the leading ``you`` is
# carried over from the original sentence's subject when the printing
# joins two effects with `` and `` and the splitter would otherwise
# leave the first piece reading as ``you draw a card``. Mirrors the
# ``(?:you )?`` shape on ``_LOSE_N_LIFE`` (the same ``you`` carry-over
# on the other half of the canonical pair) — the ``you`` is purely
# surface noise because the controller is already implicit in every
# effect's resolve signature.
_DRAW_A_CARD = re.compile(r"^(?:you )?draw a card$")
# ``draw N cards`` — bare-amount draw effect. The ``N`` accepts either
# a digit or a spelled-out small cardinal (Glimmerburst's printed
# ``Draw two cards.`` is what introduced the cardinal-word form to
# this primitive; future ``draw three cards`` / ``draw four cards``
# printings parse without further grammar work). The captured token
# routes through ``_parse_amount`` to land as an integer on the engine
# class. Mirrors the digit-or-spelled-cardinal shape on the existing
# ``_MILL_N_CARDS`` and ``_OPTIONAL_DISCARD_THEN_DRAW`` primitives.
_DRAW_N_CARDS = re.compile(
    rf"^(?:you )?draw (\d+|{_SMALL_CARDINAL_PATTERN}) cards$"
)
# "Draw that many cards" — the variable-amount form whose count is
# resolved at fire time from ``state.resolving_combat_damage_amount``.
# Surfaces inside a triggered-ability body whose CONDITION names a
# combat-damage event; the ``that many`` pronoun is anaphoric to the
# damage amount latched on the trigger source. Fear of Failed Tests
# (Duskmourn) is the EoE/Duskmourn surface form: ``Whenever this
# creature deals combat damage to a player, draw that many cards.``
# Starwinder (EoE) prints the same shape with the ``you may`` may-rider:
# ``Whenever a creature you control deals combat damage to a player,
# you may draw that many cards.`` The ``may`` is engine-flattened to
# always-yes inside ``DrawCardsEqualToResolvingCombatDamage`` (see the
# class docstring), so the parser accepts the may-prefix as syntactic
# sugar that lands on the same primitive — same precedent as other
# OptionalXyz simplifications elsewhere in this engine.
# Maps to the existing ``DrawCardsEqualToResolvingCombatDamage``
# engine class — same primitive Starwinder uses via the bespoke
# table. The optional ``you `` / ``you may `` prefixes mirror the
# bare draw shapes above (the carry-over from compound ``you draw``
# sentences); both are accepted to keep the regex shape uniform with
# its bare-amount sibling.
_DRAW_THAT_MANY_CARDS = re.compile(
    r"^(?:you (?:may )?)?draw that many cards$"
)
_YOU_GAIN_N_LIFE = re.compile(r"^you gain (\d+) life$")
# "Put a +1/+1 counter on target creature" — bare-target counter
# (Honor-style). The optional ``attacking `` qualifier narrows the
# legal-target pool to currently-attacking creatures (Hospital
# Room's ``put a +1/+1 counter on target attacking creature`` door
# trigger). When present the captured group flips
# ``attacking_only=True`` on the resulting
# ``PutPlusOneCounterOnTargetCreature`` — same engine flag the
# bespoke factory we're replacing wired in. Generalized rather
# than split into a separate literal regex per the architectural
# guidance: a single parameterized recognizer covers both the bare
# Honor surface and Hospital Room's attacking-target form.
_PUT_PLUS_ONE_COUNTER_ON_TARGET_CREATURE = re.compile(
    r"^put a \+1/\+1 counter on target (attacking )?creature$"
)
# "Put a +1/+1 counter on target creature you control" — Rayblade
# Trooper's ETB. Restricted variant: legal targets are creatures the
# resolving controller controls. Maps to
# ``PutPlusOneCounterOnCreatureYouControl(amount=1)``. Tried BEFORE
# the bare ``target creature`` regex so the more-specific
# ``you control`` shape wins on phrases that carry it.
_PUT_PLUS_ONE_COUNTER_ON_TARGET_CREATURE_YOU_CONTROL = re.compile(
    r"^put a \+1/\+1 counter on target creature you control$"
)
# "Put a +1/+1 counter on target creature an opponent controls" —
# Sunset Saboteur's attack trigger. Mirror image of the
# ``you control`` restriction above: legal targets are creatures NOT
# controlled by the resolving controller. Maps to
# ``PutPlusOneCounterOnOpponentsCreature(amount=1)``. Tried BEFORE
# the bare ``target creature`` regex so the more-specific
# ``an opponent controls`` shape wins on phrases that carry it.
_PUT_PLUS_ONE_COUNTER_ON_TARGET_OPPONENT_CREATURE = re.compile(
    r"^put a \+1/\+1 counter on target creature an opponent controls$"
)
_PUT_PLUS_ONE_COUNTER_ON_THIS_CREATURE = re.compile(
    r"^put a \+1/\+1 counter on this creature$"
)
# Tezzeret, Cruel Captain's artifact-enters trigger body. The phrase
# routes through the same name-substitution pass as the +1/+1 counter
# variant above (``_substitute_self_name`` rewrites ``Tezzeret`` to
# ``this creature`` before lookup), so the planeswalker builder shares
# the unified trigger-body grammar with creature builders. Maps to
# ``PutLoyaltyCounterOnSelf(amount=1)``; the engine class reads
# ``state.resolving_source`` at resolve time and bumps its
# ``loyalty_counters`` field.
_PUT_LOYALTY_COUNTER_ON_THIS_CREATURE = re.compile(
    r"^put a loyalty counter on this creature$"
)
# "Put a +1/+1 counter on each of up to N target creatures" —
# multi-target counter spread (Weftblade Enhancer's ETB uses ``two``).
# The amount tail accepts spelled-out small cardinals (the only form
# that ships in EoE today; ``two`` for Weftblade) plus bare digits so
# a future ``up to three target creatures`` printing parses without
# further grammar work. Maps to
# ``PutPlusOneCounterOnEachOfUpToNTargetCreatures(max_n=N)``.
_PUT_PLUS_ONE_COUNTER_ON_EACH_OF_UP_TO_N_TARGET_CREATURES = re.compile(
    rf"^put a \+1/\+1 counter on each of up to "
    rf"(\d+|{_SMALL_CARDINAL_PATTERN}) target creatures$"
)
# "Put its counters on target creature you control." — Dockworker Drone's
# dies trigger. The ``its`` pronoun refers to the source creature (dying
# permanent). Maps to ``MoveSelfCountersToTargetCreatureYouControl()``,
# whose engine class reads ``state.resolving_source.plus_one_counters``
# and transfers them to the chosen target. Surface form is post-
# normalization (the ``it `` prefix-strip in ``_normalize_creature_
# effect_phrase`` doesn't fire because the phrase starts with the verb
# ``put``, not a pronoun subject — so the ``its`` here survives intact).
_PUT_ITS_COUNTERS_ON_TARGET_CREATURE_YOU_CONTROL = re.compile(
    r"^put its counters on target creature you control$"
)
_DESTROY_TARGET_CREATURE = re.compile(r"^destroy target creature$")
# "Destroy up to one other target creature. If that creature wasn't
# dealt damage this turn, its controller draws N cards." — Faller's
# Faithful's ETB. Single primitive with an embedded follow-up
# sentence; the captured amount lands on
# ``DestroyTargetCreature(draw_if_undamaged_amount=N)`` along with
# ``optional=True`` (the ``up to one`` fragment) and
# ``exclude_self=True`` (the ``other`` fragment). The amount tail
# accepts spelled-out small cardinals (``two`` is what Faller's
# prints) plus bare digits for a future printing.
_DESTROY_UP_TO_ONE_OTHER_TARGET_CREATURE_DRAW_IF_UNDAMAGED = re.compile(
    rf"^destroy up to one other target creature\. "
    rf"If that creature wasn't dealt damage this turn, "
    rf"its controller draws (\d+|{_SMALL_CARDINAL_PATTERN}) cards$"
)
# "Destroy target creature or Spacecraft." — Embrace Oblivion's
# destroy clause. Maps to ``DestroyTargetCreature(or_spacecraft=True)``
# whose ``legal_targets`` extends the bare ``destroy target creature``
# pool to also include unstationed Spacecraft (which aren't
# currently creatures). The ``Spacecraft`` capitalization is verbatim
# from the oracle text — the spell-effect normalizer only lowercases
# the first letter of the phrase, so inner type-name capitalization
# survives.
_DESTROY_TARGET_CREATURE_OR_SPACECRAFT = re.compile(
    r"^destroy target creature or Spacecraft$"
)
# "Destroy target creature or enchantment." — Withering Torment
# (Duskmourn). Maps to ``DestroyTargetCreature(or_enchantment=True)``
# whose ``legal_targets`` extends the bare ``destroy target creature``
# pool to also include any enchantment (Aura subtype or not). The
# enchantment branch emits ``("enchantment", id)`` tuples; the resolve
# accepts either kind and finds the object by id. The lowercase
# ``enchantment`` is the printed surface — unlike ``Spacecraft`` which
# is capitalized in oracle text, ``enchantment`` is lowercased.
_DESTROY_TARGET_CREATURE_OR_ENCHANTMENT = re.compile(
    r"^destroy target creature or enchantment$"
)
_COUNTER_TARGET_SPELL = re.compile(r"^counter target spell$")
# "Counter target artifact or enchantment spell." — Annul. Same engine
# class as the bare counterspell, just with a non-empty ``filter_types``
# frozenset that gates which stack items are legal targets. Tried before
# the bare ``counter target spell`` regex below because the bare regex
# requires the phrase to end at ``spell`` with no qualifier.
_COUNTER_TARGET_ARTIFACT_OR_ENCHANTMENT_SPELL = re.compile(
    r"^counter target artifact or enchantment spell$"
)
_EXILE_TARGET_CREATURE = re.compile(r"^exile target creature$")
# "Exile target artifact, enchantment, or creature with power 4 or
# greater." — Exorcise `{1}{W}` Sorcery (Duskmourn): a single-sentence
# three-way disjunctive exile (target an artifact OR an enchantment OR
# a creature whose effective power is at least 4). Maps to the
# parameterless ``ExorciseEffect`` engine class which evaluates the
# disjunctive filter at both target-pick and resolve time using
# ``effective_power`` for the creature arm.
#
# Currently only Exorcise in the EoE/DSK pool fits this shape, so the
# recognizer is a parameterless literal — broader generalization (e.g.
# ``exile target <type-list> with <p/t-threshold>``) is deferred until
# a second card with a similar three-way disjunctive permanent-type
# exile shape exists. Sibling to the bare ``exile target creature``
# branch above; lives in the per-clause grammar (rather than the
# whole-spell exact-text tier) per the parser's
# generic-templates-over-literals guidance.
_EXILE_TARGET_ARTIFACT_ENCHANTMENT_OR_BIG_CREATURE = re.compile(
    r"^exile target artifact, enchantment, or creature with power "
    r"4 or greater$"
)
# "Destroy all tapped creatures." / "Destroy all untapped creatures." —
# Split Up's modal mode lines. Two separate regexes (rather than one
# regex with a captured tap-state qualifier) because the matching engine
# class flag is a per-flag bool — cleaner to set the named bool from the
# matched regex than to dispatch on captured surface text. The bare
# ``destroy all creatures`` (untargeted Wrath) shape isn't used by any
# spell in the current pool, so it's not a regex here yet — add it as a
# sibling if/when one ships.
_DESTROY_ALL_TAPPED_CREATURES = re.compile(
    r"^destroy all tapped creatures$"
)
_DESTROY_ALL_UNTAPPED_CREATURES = re.compile(
    r"^destroy all untapped creatures$"
)
# "Destroy all creatures, then any number of target players each mill
# half their library, rounded down." — Singularity Rupture `{3}{U}{B}{B}`
# Sorcery: a board wrath followed by a multi-target half-library mill on
# any number of chosen players. Maps to the parameterless
# ``SingularityRuptureEffect`` engine class which folds the wrath and
# the per-target floor-divide mill into one resolve step (the cast-
# pipeline target-count range is 0..players-bounded, with each chosen
# target milling ``floor(library_size / 2)`` cards).
#
# Currently only Singularity Rupture in the EoE pool fits this shape, so
# the recognizer is a parameterless literal — broader generalization
# (e.g. ``destroy all creatures, then ... <any-mass-effect>``) is
# deferred until a second card with this two-clause wrath-then-multi-
# target-mass-effect shape exists. Sibling to the tapped/untapped
# wrath branches above; lives in the per-clause grammar (rather than
# the whole-spell exact-text tier) per the parser's
# generic-templates-over-literals guidance.
_SINGULARITY_RUPTURE = re.compile(
    r"^destroy all creatures, then any number of target players "
    r"each mill half their library, rounded down$"
)
# "Destroy all creatures with power N or greater." — Restricted Office's
# door trigger body in DSK ("destroy all creatures with power 3 or
# greater"). Parameterized: the captured group is the minimum power
# threshold (digit only — no card prints a spelled-out power bound on
# this surface). Maps to ``DestroyAllCreatures(min_power=N)`` whose
# ``effective_power`` walk at resolve time correctly accounts for
# layer-7 pump effects and +1/+1 counters when filtering victims.
# Sibling to the tapped/untapped branches above; sets a different
# engine-class flag (``min_power`` rather than the tap-state bools)
# so the dispatch is a separate ``re.match`` branch in the
# ``destroy``-family neighborhood.
_DESTROY_ALL_CREATURES_WITH_POWER_AT_LEAST = re.compile(
    r"^destroy all creatures with power (\d+) or greater$"
)
# "Exile all creatures and Spacecraft." — Beyond the Quiet wrath. Capital
# ``S`` on Spacecraft is preserved verbatim from the oracle text — the
# normalization layer only lowercases the first letter of the phrase, so
# inner type-name capitalization survives. Untargeted (mass) effect.
#
# Note this is matched here as a SINGLE-effect regex, before the
# compound-effect splitter would carve the phrase on `` and `` —
# ``_parse_compound_effect`` runs first in ``_parse_effect`` but punts
# (returns None) when any split-piece fails to parse, and ``Spacecraft``
# alone isn't a recognized effect, so the compound attempt rejects and
# falls through to the single-effect dispatcher where this regex wins.
_EXILE_ALL_CREATURES_AND_SPACECRAFT = re.compile(
    r"^exile all creatures and Spacecraft$"
)
# "Exile all artifacts with mana value N or less." — Ruinous Rampage's
# anti-artifact mode. The captured group is the max mana value (digit
# only — no card prints a spelled-out mana-value bound today). Maps to
# ``ExileAllArtifactsWithManaValueAtMost(max_mv=N)``.
_EXILE_ALL_ARTIFACTS_WITH_MANA_VALUE_AT_MOST = re.compile(
    r"^exile all artifacts with mana value (\d+) or less$"
)
_EACH_OPPONENT_LOSES_N_LIFE = re.compile(
    r"^each opponent loses (\d+) life$"
)
# "Each player draws a card" / "Each player draws N cards" — every
# player (active controller AND opponents) draws ``N`` cards.
# Friendly Teddy's ``When this creature dies, each player draws a
# card.`` is the EoE/Duskmourn surface form. ``a`` collapses to
# amount=1; digits and spelled-out small cardinals go through
# ``_parse_amount`` so a future ``each player draws three cards``
# printing parses without further grammar work. Maps to
# ``EachPlayerDraws(amount=N)`` whose ``resolve`` walks every living
# player and calls ``draw_n_cards``.
_EACH_PLAYER_DRAWS_N_CARDS = re.compile(
    rf"^each player draws "
    rf"(?:a card|(\d+|{_SMALL_CARDINAL_PATTERN}) cards)$"
)
# "Exile each opponent's graveyard" — untargeted graveyard wipe on
# every opponent (Dauntless Scrapbot's ETB compound first half).
# Maps to ``ExileEachOpponentsGraveyard()`` which sweeps every card
# in each opponent's graveyard into exile at resolve time.
_EXILE_EACH_OPPONENTS_GRAVEYARD = re.compile(
    r"^exile each opponent's graveyard$"
)
# "<source> deals N damage to each opponent." — mass burn aimed at
# every opponent (Ruinous Rampage's burn mode). The ``deals`` form
# (no ``it`` / ``you`` subject) is what Scryfall prints for spells
# that name themselves as the subject; the spell-effect normalizer
# strips the ``<Name>`` prefix before this regex sees the phrase, so
# the surface form here is exactly ``deals N damage to each opponent``.
# For creature-source triggered abilities the ``it`` prefix is also
# stripped first, leaving the same canonical ``deals …`` form.
#
# The optional ``this creature `` prefix accepts the surface where the
# trigger body names the source explicitly. Scryfall prints that form
# when the trigger CONDITION already uses a different subject (Weftstalker
# Ardent's ``Whenever another creature or artifact you control enters``
# leads into ``this creature deals 1 damage to each opponent`` — the
# explicit ``this creature`` resolves what would otherwise be an ambiguous
# ``it`` reference to the unrelated entering object). Mirrors the same
# ``(?:this creature )?`` shape on ``_DEAL_DAMAGE_TARGET_OPPONENT``
# (Biotech Specialist's ``Whenever you sacrifice an artifact, this
# creature deals 2 damage to target opponent``) — same justification:
# the normalizer only consumes the ``it `` pronoun, not the longer
# ``this creature`` self-object reference.
#
# The ``this Room `` prefix is the Duskmourn Room equivalent — Shattered
# Yard's end-step trigger prints ``this Room deals 1 damage to each
# opponent`` (the Room is the damage source; ``this Room`` reads the
# same way ``this creature`` does on a creature-source trigger). Same
# legal-target pool either way.
_DEAL_N_DAMAGE_EACH_OPPONENT = re.compile(
    r"^(?:this creature |this Room )?deals (\d+) damage to each opponent$"
)
# "<source> deals N damage to each creature." — untargeted board-wipe
# burn (Pyroclasm in Duskmourn; the same primitive surfaces inside
# Lithobraking's reflexive optional-sac sequence and Extinguisher
# Battleship's ETB sequence in EoE, where it's already exercised by
# the ``DealNDamageToEachCreature`` engine class). Mirrors the
# ``deals N damage to each opponent`` shape: ``deals`` form (no ``it``
# / ``you`` subject) is what Scryfall prints for spells that name
# themselves as the subject; the spell-effect normalizer strips the
# ``<Name>`` prefix before this regex sees the phrase. The optional
# ``this creature `` prefix accepts the surface where a creature-source
# trigger body names the source explicitly (same precedent as
# ``_DEAL_N_DAMAGE_EACH_OPPONENT`` and ``_DEAL_DAMAGE_TARGET_OPPONENT``
# — the ``it `` normalizer only consumes the bare pronoun).
#
# The optional ``this Spacecraft `` prefix accepts the surface where
# the source is a Spacecraft (Extinguisher Battleship's ETB body second
# sentence — ``Then this Spacecraft deals 4 damage to each creature.``;
# the leading ``Then `` connective is stripped by
# ``_split_effect_sentences`` before this regex sees the phrase).
# Mirrors the ``(?:this Room )?`` precedent on
# ``_DEAL_DAMAGE_TARGET_OPPONENT_CREATURE`` — same self-source pronoun
# pattern Rooms use, here applied to the Spacecraft type. Source
# identity is established by the enclosing trigger's ``IsSelf()``
# predicate, not the effect body.
_DEAL_N_DAMAGE_EACH_CREATURE = re.compile(
    r"^(?:this creature |this Spacecraft )?"
    r"deals (\d+) damage to each creature$"
)
# "Search your library for an artifact card, reveal it, put it into
# your hand, then shuffle." — Scour for Scrap mode 1 (and any future
# narrow tutor with the same surface). Maps to
# ``SearchLibraryForCard(filter_name="artifact", may_fail=True)``.
# The full surface form (with reveal/hand/shuffle tail) is required:
# this primitive doesn't claim a simpler ``search your library for an
# artifact card`` line because that bare shape isn't a complete oracle
# clause for any printed card — every artifact tutor in EoE prints the
# full reveal/hand/shuffle tail.
_SEARCH_LIBRARY_FOR_ARTIFACT_CARD = re.compile(
    r"^search your library for an artifact card, "
    r"reveal it, put it into your hand, then shuffle$"
)
# "Search your library for a basic land card, reveal it, put it into
# your hand, then shuffle." — Spineseeker Centipede's ETB tutor (and
# any future generic basic-land-to-hand tutor with the same surface).
# Maps to ``SearchLibraryForCard(filter_name="basic_land", may_fail=
# True)``. Like the artifact-tutor sibling above, the full surface
# form (reveal / put-into-hand / shuffle tail) is required — the bare
# ``search your library for a basic land card`` shape isn't a complete
# oracle clause for any DSK printing today, and the all-or-nothing
# tail keeps the regex's claim narrow enough that an unrelated tutor
# with a different reward (onto-the-battlefield, into-graveyard) won't
# accidentally match.
_SEARCH_LIBRARY_FOR_BASIC_LAND_CARD = re.compile(
    r"^search your library for a basic land card, "
    r"reveal it, put it into your hand, then shuffle$"
)
# "Search your library for a legendary creature card, reveal it, put
# it into your hand, then shuffle." — The Seriema's ETB body (the only
# EoE/DSK printing of this exact filter today). Maps to
# ``SearchLibraryForCard(filter_name="legendary_creature",
# may_fail=False)``. The printed text contains no "you may" rider, so
# ``may_fail=False`` is the faithful reading: when a legendary creature
# card exists in the library, the controller MUST take one. This
# differs from the ``may_fail=True`` semantics the older
# ``_SEARCH_LIBRARY_FOR_ARTIFACT_CARD`` /
# ``_SEARCH_LIBRARY_FOR_BASIC_LAND_CARD`` recognizers ship — those two
# pre-date the explicit-faithfulness rules and should likely flip to
# ``may_fail=False`` in a future pass; doing so here would change
# behaviour pinned by many existing tests, so it's out of scope for
# this batch. Literal-line recognizer because The Seriema is the only
# EoE/DSK card printing this exact "legendary creature card" filter
# in the reveal/hand/shuffle shape today; broader generalization
# (parameterizing the filter substring across artifact / basic land /
# legendary creature shapes into one regex) is deferred until the
# may_fail-faithfulness cleanup above merges the existing two literals
# at the same time.
_SEARCH_LIBRARY_FOR_LEGENDARY_CREATURE_CARD = re.compile(
    r"^search your library for a legendary creature card, "
    r"reveal it, put it into your hand, then shuffle$"
)
# "Search your library for a basic land card, put it onto the
# battlefield tapped, then shuffle." — Moldering Gym's door trigger
# (the tapped-onto-battlefield reward form). Same fetch-land payload
# as the Lander token's activated ability and the
# ``_FETCH_LAND_BODIES`` table in ``land_builder``; landing the
# surface here lets unified-grammar Room halves (and any future
# triggered/activated-ability surface that prints this exact
# sentence) claim through a single recognizer. Maps to
# ``SearchLibraryForBasicLandTapped()``. Distinct from the
# put-into-your-hand sibling above on the reward tail (battlefield-
# tapped vs hand-with-reveal); the two surfaces are mutually
# exclusive at the regex level so ordering between them doesn't
# matter. Literal-line recognizer with a single capture-less surface
# — Moldering Gym is the only printing in the EoE/DSK pool whose
# triggered-ability body prints this exact sentence today; broader
# grammar (parameterizing the count, ``land`` vs ``basic land``, or
# the tapped-vs-untapped tail) is deferred until a second example
# with a varying piece exists.
_SEARCH_LIBRARY_FOR_BASIC_LAND_CARD_TAPPED_ONTO_BATTLEFIELD = re.compile(
    r"^search your library for a basic land card, "
    r"put it onto the battlefield tapped, then shuffle$"
)
# "Search your library for an Equipment card, put it onto the
# battlefield, then shuffle." — Honored Knight-Captain's activated-
# ability body (sac-self → tutor for an Equipment). Maps to the
# existing ``SearchLibraryForEquipmentToBattlefield()`` engine class.
# Distinct from the ``_SEARCH_LIBRARY_FOR_BASIC_LAND_CARD_TAPPED_ONTO_
# BATTLEFIELD`` sibling above on the filter (Equipment vs basic land)
# AND the tap state (untapped vs tapped); the two shapes are mutually
# exclusive at the regex level so ordering doesn't matter. Literal-
# line recognizer with a single capture-less surface — Honored
# Knight-Captain is the only printing in the EoE/Duskmourn pool whose
# activated body prints this exact sentence today; broader grammar
# (parameterizing the filter-card-type — Aura, Vehicle, etc. — or the
# tapped-vs-untapped tail) is deferred until a second example with a
# varying piece exists.
_SEARCH_LIBRARY_FOR_EQUIPMENT_CARD_ONTO_BATTLEFIELD = re.compile(
    r"^search your library for an Equipment card, "
    r"put it onto the battlefield, then shuffle$"
)
# Token-creation effect grammar. Two related shapes co-exist in Scryfall
# oracle text:
#
# 1. **Stat-bearing creature tokens** — e.g.
#    ``create a tapped 2/2 colorless Robot artifact creature token`` or
#    ``create two 1/1 white Human Soldier creature tokens``. Carries
#    explicit power/toughness, a color word, one or more subtype words,
#    and a ``[artifact ]?creature token[s]?`` tail.
# 2. **Bare-name tokens** — e.g. ``create a Lander token``. No stats /
#    color / "creature" tail; the subtype word IS the token name and
#    fully identifies the registered printing (Lander, Munitions, etc.).
#
# Both paths look up the resolved token name in ``TOKEN_PRINTINGS``. The
# parser is NOT in the business of building token printings on the fly —
# if the registry doesn't already carry an entry for the matched name,
# parsing fails (returns ``None``) and the card stays unclaimed. Adding
# new token shapes is a separate concern handled in ``cards/tokens.py``.
#
# The leading count phrase is one of:
#   - ``a`` / ``an``    → 1
#   - a small spelled-out cardinal (``two``, ``three``, …)
#   - a bare digit
# The optional ``tapped`` keyword sits between the count and the stats
# (or directly before the bare-name in shape 2).
_TOKEN_COUNT_PATTERN = rf"a|an|\d+|{_SMALL_CARDINAL_PATTERN}"
# Color word — one of the five colors plus colorless. Not captured for
# correctness (the registry already encodes color via the printing's
# ``card_types``/``produces``); we just need to consume it so the regex
# matches the surface form.
_TOKEN_COLOR_PATTERN = r"white|blue|black|red|green|colorless"
# Stat-bearing creature-token shape. The captured groups are:
#   1. count word/digit
#   2. literal ``tapped`` if present (None otherwise)
#   3. power
#   4. toughness
#   5. color word (consumed only — see above)
#   6. subtype phrase (one or more space-separated capitalized words)
#   7. literal ``artifact`` if the token is also an artifact
#   8. literal ``enchantment`` if the token is also an enchantment
# The trailing ``creature token[s]?`` is required — bare-name tokens
# (Lander, Munitions) take a different surface form and parse via
# ``_CREATE_BARE_NAME_TOKEN`` below.
#
# Both ``artifact`` and ``enchantment`` qualifiers are consumed but not
# inspected — the registered ``TOKEN_PRINTINGS`` entry encodes the
# token's full type line, so the regex's job is just to recognize the
# surface and look up the subtype name. Glimmer (Duskmourn's
# enchantment-creature token created by Tunnel Surveyor and the other
# Glimmer-token cards) is the first ``enchantment creature`` shape in
# the EoE/Duskmourn pool; ``artifact`` came earlier with Robot/Drone.
# The Oxford comma between the two qualifiers (``2/2 white Cat
# artifact enchantment creature token``) is forward-looking grammar
# coverage — no card today carries both, but the regex shape allows
# either independently or both simultaneously without further work.
_CREATE_CREATURE_TOKEN = re.compile(
    rf"^create ({_TOKEN_COUNT_PATTERN}) (tapped )?(\d+)/(\d+) "
    rf"({_TOKEN_COLOR_PATTERN}) ([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*) "
    rf"(artifact )?(enchantment )?creature tokens?(?:\s+with\s+.+)?$",
    re.DOTALL,
)
# The optional trailing ``(?:\s+with\s+.+)?`` allows oracle text that
# appends the token's built-in abilities as a reminder (e.g. Station
# Monitor's "…Drone artifact creature token with flying and 'This token
# can block only creatures with flying.'"): the abilities are already
# encoded in ``TOKEN_PRINTINGS`` so the trailing description is
# redundant and can be safely ignored by the recognizer.
# Bare-name token shape — ``create a Lander token`` /
# ``create two Lander tokens``. The captured subtype IS the registry
# key. Optional ``tapped`` sits between the count and the name (no EoE
# card prints a tapped Lander today, but the slot is reserved so a
# future ``create a tapped Lander token`` line parses without further
# grammar work).
_CREATE_BARE_NAME_TOKEN = re.compile(
    rf"^create ({_TOKEN_COUNT_PATTERN}) (tapped )?"
    rf"([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*) tokens?$"
)
# "Create a tapped 2/2 colorless Robot artifact creature token for each
# multicolored permanent you control." — Infinite Guideline Station's
# ETB. Maps to ``CreateTokenPerMulticoloredPermanent(token_printing=
# TOKEN_PRINTINGS["Robot"], tapped=True)`` — same primitive the static
# registry / bespoke factory uses. Literal-line recognizer with a
# zero-capture surface: only Infinite Guideline Station in the EoE
# pool prints this exact shape today, so a parameterized
# ``create a tapped <P>/<T> <color> <Subtype> <type> token for each
# multicolored permanent you control`` regex would have nothing else
# to claim. Deferred generalization: when a second card ships with
# the ``... for each multicolored permanent you control`` token-
# creation tail (or with a sibling per-X gate like ``for each artifact
# you control``), fold both literals into one regex with captured
# count/stats/color/subtype groups dispatching through the existing
# ``_parse_create_token_effect`` body to share the
# ``TOKEN_PRINTINGS`` lookup with the bare-token shapes above.
_CREATE_TAPPED_ROBOT_TOKEN_PER_MULTICOLORED_LINE = re.compile(
    r"^create a tapped 2/2 colorless Robot artifact creature token "
    r"for each multicolored permanent you control$"
)
# "Draw a card for each multicolored permanent you control." —
# Infinite Guideline Station's attack trigger. Maps to
# ``DrawCardPerMulticoloredPermanent()``. Literal-line recognizer
# with a zero-capture surface: only Infinite Guideline Station in the
# EoE pool prints this exact shape today, so a parameterized
# ``<verb> for each multicolored permanent you control`` regex would
# have nothing else to claim. Deferred generalization: when a second
# card ships with a sibling per-X tail on a different verb (e.g.
# ``each opponent loses 1 life for each multicolored permanent you
# control``), fold the per-multicolored-permanent gate into a shared
# tail recognizer dispatching to the matching primitive — same
# approach the existing ``deal damage by artifact count`` recognizer
# took.
_DRAW_CARD_PER_MULTICOLORED_PERMANENT_LINE = re.compile(
    r"^draw a card for each multicolored permanent you control$"
)
_SURVEIL_N = re.compile(r"^surveil (\d+)$")
# "Look at the top two cards of your library. Put one into your hand
# and the other into your graveyard." — Codecracker Hound's ETB.
# Maps to ``LookAtTopTwoSplitHandGraveyard()``. The two-sentence
# surface is what Scryfall prints — the embedded period sits inside
# the effect phrase rather than being the trailing terminator. Compound
# splits don't apply (the second sentence's leading ``Put one`` doesn't
# parse as a standalone effect — see ``_parse_compound_effect``'s
# all-or-nothing rule), so the regex matches the literal two-sentence
# surface as a single primitive.
_LOOK_TOP_TWO_SPLIT_HAND_GRAVEYARD = re.compile(
    r"^look at the top two cards of your library\. "
    r"Put one into your hand and the other into your graveyard$"
)
# ``look at the top N cards of your library. You may reveal a <filter>
# card from among them and put it into your hand. Put the rest on the
# bottom of your library in a random order[. If you didn't put a card
# into your hand this way, put a +1/+1 counter on this creature]`` —
# parameterized peek-tutor-or-counter shape. Two cards in the EoE/DSK
# pool currently print this body:
#
#   - Living Phone (DSK) — dies trigger; filter "creature card with
#     power N or less"; no counter-rider tail (no ``If you didn't put
#     a card into your hand this way`` sentence).
#   - Pulsar Squadron Ace (EoE) — ETB; filter "Spacecraft card"; the
#     counter-rider tail IS present.
#
# Two filter shapes are captured: a bare type ("Spacecraft") OR
# "creature card with power N or less". Both land on the existing
# ``LookAtTopNRevealOrCounter`` engine class — the captured filter
# token routes to ``filter_name`` ("spacecraft" / "creature"), the
# optional ``max_power`` capture lands on the engine class's
# ``max_power=N`` kwarg (only meaningful for the creature filter; the
# class ignores it for non-creature filters), and the optional
# counter-rider tail flips ``counter_on_miss=True``. Replaces the two
# distinct literals (the prior Pulsar-specific bespoke and the prior
# ``_LOOK_TOP_FIVE_REVEAL_LOW_POWER_CREATURE`` Living Phone literal)
# with a single parameterized recognizer per the parser CLAUDE.md
# guidance.
_LOOK_TOP_N_REVEAL_OR_COUNTER = re.compile(
    rf"^look at the top (\d+|{_SMALL_CARDINAL_PATTERN}) cards of your "
    rf"library\. "
    rf"You may reveal a (Spacecraft|creature) card"
    rf"(?: with power (\d+) or less)? "
    rf"from among them and put it into your hand\. "
    rf"Put the rest on the bottom of your library in a random order"
    rf"(?P<rider>\. If you didn't put a card into your hand this way, "
    rf"put a \+1/\+1 counter on this creature)?$"
)
# Mill is "mill <amount> cards" where the amount is either a digit or
# one of the spelled-out small cardinals (Scryfall printing convention).
# Singular "mill 1 card" / "mill one card" is intentionally unsupported
# — no EoE card prints that surface form. Add it the day one ships.
_MILL_N_CARDS = re.compile(
    rf"^mill (\d+|{_SMALL_CARDINAL_PATTERN}) cards$"
)
_DESTROY_TARGET_ARTIFACT = re.compile(r"^destroy target artifact$")
# "Destroy target artifact, enchantment, or creature with flying." —
# Shattered Wings's destroy clause. Maps to
# ``DestroyTargetArtifactEnchantmentOrFlyingCreature()`` whose
# ``legal_targets`` filters to artifacts (any), enchantments (any),
# or creatures that currently have flying. The Oxford comma after
# ``enchantment`` is required (matches Scryfall's printed form).
_DESTROY_TARGET_ARTIFACT_ENCHANTMENT_OR_FLYING_CREATURE = re.compile(
    r"^destroy target artifact, enchantment, or creature with flying$"
)
_DESTROY_TARGET_LAND = re.compile(r"^destroy target land$")
_DESTROY_TARGET_NONCREATURE_PERMANENT = re.compile(
    r"^destroy target noncreature permanent$"
)
# "Destroy target nonland permanent." — Thaumaton Torpedo's activated
# ability. Maps to the existing ``DestroyTargetNonlandPermanent`` engine
# class whose ``legal_targets`` filters the battlefield to non-land
# permanents (creatures, artifacts, enchantments, planeswalkers).
# Distinct from ``destroy target noncreature permanent`` above — that
# one excludes creatures, this one excludes lands.
_DESTROY_TARGET_NONLAND_PERMANENT = re.compile(
    r"^destroy target nonland permanent$"
)
# "Destroy target artifact or enchantment. If its mana value was 2 or
# less, create a Lander token." — Seedship Impact `{1}{G}` Instant: a
# bare destroy on an artifact-or-enchantment target with a small-mana-
# value Lander rider. Two-sentence shape: the leading sentence is the
# bare destroy; the trailing ``If its mana value was 2 or less, create
# a Lander token.`` clause is an anaphoric rider that doesn't parse as
# a standalone effect (the ``its`` back-references the destroyed
# permanent's mana value, with no per-clause primitive recognizing the
# back-reference). The recognizer matches the unsplit two-sentence
# body verbatim — ``_parse_spell_oracle_text`` tries the unsplit
# clause first via ``_parse_spell_effect_phrase`` before falling
# through to the per-sentence splitter, so this regex claims Seedship
# Impact before the splitter can carve the rider into an unparseable
# piece.
#
# Currently only Seedship Impact in the EoE pool fits this shape, so
# the recognizer is a parameterless literal — broader generalization
# (e.g. parameterizing the mana-value threshold or the destroy-target
# type list) is deferred until a second card with a similar two-
# sentence destroy-with-Lander-on-small-mv-rider shape exists. Sibling
# to the destroy-target neighbors above; lives in the per-clause
# grammar (rather than the whole-spell exact-text tier) per the
# parser's generic-templates-over-literals guidance — a neighboring
# recognizer is the natural landing site for the next card with this
# shape rather than expanding the whole-spell tier.
_SEEDSHIP_IMPACT = re.compile(
    r"^destroy target artifact or enchantment\. "
    r"If its mana value was 2 or less, create a Lander token$"
)
# "Add {<color>}." — single coin of a single color. Mana ability surface
# form. The captured letter is one of WUBRG (no colorless single-coin
# add in EoE today; extend to ``[WUBRGC]`` if a future card needs it).
# The surrounding ActivatedAbility built by the activated-grammar layer
# auto-flags ``is_mana_ability=True`` once the resulting effect is
# recognized by ``_is_mana_ability_effect``, so the engine bypasses the
# stack on activation per CR 605.
_ADD_SINGLE_COLOR_MANA = re.compile(r"^add \{([WUBRG])\}$")
# "Add one mana of any color" — single coin of any color, the
# at-resolution choose-a-color mana ability surface form (All-Fates
# Scroll's ``{T}: Add one mana of any color.``). Maps to
# ``AddManaToPool(produces=(("any", 1),))`` — the engine treats the
# ``"any"`` color tag as a chosen-at-mint colored coin (the controller
# names the color when the ability resolves). Distinct from the
# bracketed ``add {<color>}`` shape: the latter mints a fixed-color
# coin known at parse time, this one defers the choice to resolution.
# Mana-source tracking treats ``"any"`` coins as untyped at compile
# time, mirroring ``land_builder``'s ``_fancy_pay_one_life_any_color``
# wiring on Mana Confluence.
_ADD_ONE_MANA_OF_ANY_COLOR = re.compile(r"^add one mana of any color$")
# "Add {<color>}{<color>}." — two coins of a single color (Steelswarm
# Operator's ``{T}: Add {U}{U}`` ability mints two restricted coins of
# the same color). The captured letter is reused — the engine class
# accepts a tuple of (color, amount) coins, so the produced effect is
# ``AddManaToPool(produces=((c, 1), (c, 1)))``. Two coins is what EoE
# prints today; if a future card uses ``Add {U}{U}{U}`` extend the
# regex with another optional brace pair (or generalize to a parser
# loop).
_ADD_TWO_SAME_COLOR_MANA = re.compile(
    r"^add \{([WUBRG])\}\{([WUBRG])\}$"
)
# "Add {<mana>}. Spend this mana only to <restriction>." —
# restricted-mana ability whose produced coins are tagged with an
# engine-side ``spend_restriction`` string at mint time. EoE wiring:
#
#   * ``Spend this mana only to cast an artifact spell`` →
#     ``"artifact_spell"`` (Steelswarm Operator's first ability).
#   * ``Spend this mana only to activate abilities of artifact
#     sources`` → ``"artifact_activation"`` (Steelswarm Operator's
#     second ability).
#
# Both restrictions are gated at cost-payment time by
# ``_coin_satisfies_context`` in ``actions.py``. The captured mana
# segment accepts either a single ``{<color>}`` or two of the same
# color (``{<color>}{<color>}``) — extend the inner pattern if a future
# printing mints a hybrid- or multi-color restricted batch. Tried
# BEFORE the bare ``add {<color>}`` regex so the qualifier wins; the
# bare regex's ``$`` anchor wouldn't match the multi-sentence shape
# anyway, but ordering more-specific before more-general is the
# convention here.
_ADD_MANA_WITH_SPEND_RESTRICTION = re.compile(
    r"^add (\{[WUBRG]\}(?:\{[WUBRG]\})?)\. "
    r"Spend this mana only to "
    r"(cast an artifact spell|activate abilities of artifact sources)$"
)
# Surface phrase → ``ActivationCost.spend_restriction`` string mapping
# for the restricted-mana ability above. Each tail phrase below comes
# from a real EoE printing's oracle text; the engine string is the key
# expected by ``_coin_satisfies_context`` (``actions.py``).
_SPEND_RESTRICTION_BY_PHRASE: dict[str, str] = {
    "cast an artifact spell": "artifact_spell",
    "activate abilities of artifact sources": "artifact_activation",
}
# "Target opponent loses N life" — pure life-loss spell/ability surface.
# TargetOpponentLosesLife filters its legal-targets pool to opponents.
_TARGET_OPPONENT_LOSES_N_LIFE = re.compile(
    r"^target opponent loses (\d+) life$"
)
# "Return up to one other target permanent you control to its owner's
# hand" — Exosuit Savior's ETB. Maps to
# ``ReturnTargetPermanentYouControlToOwnersHand(optional=True,
# exclude_self=True)``. The ``up to one other`` fragment combines the
# ``up to one`` optionality (resolves as a no-op if no target chosen)
# with the ``other`` self-exclusion (filters the trigger source out
# of the legal pool). Lands are legal targets — the engine class's
# ``legal_targets`` doesn't filter on type beyond the controller and
# self gate.
_RETURN_UP_TO_ONE_OTHER_PERMANENT_YOU_CONTROL_TO_OWNERS_HAND = re.compile(
    r"^return up to one other target permanent you control "
    r"to its owner's hand$"
)
# "Return up to one target creature to its owner's hand." — Bottomless
# Pool's door trigger. Optional-target bounce against any creature
# (friend or foe). Maps to
# ``ReturnTargetCreatureToOwnersHand(optional=True)`` — same engine
# class the spell-builder dispatch uses for the mandatory shape, here
# with the ``optional=True`` flag flipped so a no-target resolution
# is a clean no-op fizzle (engine-side
# ``target_count_range = (0, 1)``). Distinct from the ``up to one
# OTHER target permanent you control`` Exosuit Savior shape above on
# both target type (creature only vs any permanent) and controller
# qualifier (any controller vs you-control); the two are mutually
# exclusive at the regex level. Literal-line recognizer with no
# capture groups — Bottomless Pool is the only printing in the
# EoE/DSK pool whose triggered- or activated-ability body prints this
# exact sentence; broader grammar (parameterizing ``up to one`` vs
# bare ``target``, or fold into a single creature-bounce family with
# the existing ``ReturnTargetCreatureToOwnersHand`` paths in
# ``spell_builder``) is deferred until a second example with a
# varying piece exists.
_RETURN_UP_TO_ONE_TARGET_CREATURE_TO_OWNERS_HAND = re.compile(
    r"^return up to one target creature to its owner's hand$"
)
# "Return up to N target non-Spacecraft creatures to their owners'
# hands" — Specimen Freighter's ETB body. Maps to
# ``ReturnUpToNTargetCreaturesToOwnersHands(max_n=N,
# exclude_spacecraft=True)`` whose ``legal_targets`` filters to
# non-Spacecraft creatures and ``target_count_range`` returns
# ``(0, max_n)``. The amount tail accepts the spelled-out small
# cardinal Scryfall prints (``two`` on Specimen Freighter) plus
# bare digits for any future printing carrying a different bound.
# The ``non-Spacecraft`` qualifier is verbatim from the oracle text;
# only Specimen Freighter in the current pool prints this filter, but
# the bound is parameterized so a future ``return up to three target
# non-Spacecraft creatures …`` printing lands on the same regex.
_RETURN_UP_TO_N_NON_SPACECRAFT_CREATURES_TO_OWNERS_HANDS = re.compile(
    rf"^return up to (\d+|{_SMALL_CARDINAL_PATTERN}) target "
    rf"non-Spacecraft creatures to their owners' hands$"
)
# "Return target creature card from your graveyard to your hand" —
# graveyard-to-hand reanimation surface.
_RETURN_TARGET_CREATURE_GRAVEYARD_TO_HAND = re.compile(
    r"^return target creature card from your graveyard to your hand$"
)
# "Return a creature or Spacecraft card from your graveyard to your
# hand" — Fell Gravship's ETB body second sentence (after the
# ``mill three cards, then`` connective is split off by the compound
# parser). Maps to ``ReturnCreatureFromYourGraveyardToHand(
# or_spacecraft=True)``. The article is ``a`` rather than ``target``
# because Scryfall prints the choice as a non-target selection in
# this position; the engine class's ``requires_target`` /
# ``legal_targets`` still drive the choice through the agent at
# resolve time. Tried BEFORE the bare ``return target creature card``
# regex below so the more-specific ``or Spacecraft`` widening lands
# first; the bare regex's ``$`` anchor on ``hand`` would never match
# the ``or Spacecraft`` tail anyway, but ordering more-specific before
# more-general is the convention in this dispatcher. Mirrors the
# ``or_spacecraft`` widening on ``DestroyTargetCreature`` (Embrace
# Oblivion) and the ``or_enchantment`` widening on the same engine
# class (Withering Torment) — same precedent of adding a flag to an
# existing graveyard / battlefield operator rather than a new engine
# class. Literal-line recognizer with a single-card claim: only Fell
# Gravship in the EoE pool prints this exact ``a creature or
# Spacecraft card`` graveyard surface today; deferred generalization
# would fold this and a future ``return a creature or Spacecraft
# card from your graveyard to the battlefield`` sibling into one
# regex with a captured destination word, dispatching across both
# engine-class destinations.
_RETURN_CREATURE_OR_SPACECRAFT_GRAVEYARD_TO_HAND = re.compile(
    r"^return a creature or Spacecraft card "
    r"from your graveyard to your hand$"
)
# "Return target creature card with mana value N or less from your
# graveyard to your hand" — narrowed-by-MV graveyard-to-hand reanimation.
# Savior of the Small's Survival reward (``mana value 3 or less``); the
# threshold is the captured digit. Maps to
# ``ReturnCreatureFromYourGraveyardToHand(max_mv=N)``. Tried BEFORE the
# bare ``return target creature card from your graveyard to your hand``
# regex above so a future printing carrying the MV rider lands here
# rather than on the unbounded sibling. The bare regex still covers
# Interceptor Mechan and any other MV-unrestricted reanimator.
_RETURN_TARGET_CREATURE_GRAVEYARD_TO_HAND_WITH_MV = re.compile(
    r"^return target creature card with mana value (\d+) or less "
    r"from your graveyard to your hand$"
)
# "Return target creature card from your graveyard to the battlefield"
# — graveyard reanimation surface (Xu-Ifit and forward-compat for any
# future "raise dead" oracle line).
_RETURN_TARGET_CREATURE_GRAVEYARD_TO_BATTLEFIELD = re.compile(
    r"^return target creature card from your graveyard to the battlefield$"
)
# "Return target creature or enchantment card from your graveyard to
# the battlefield" — Rescue Skiff's ETB body (the only EoE/DSK printing
# of this exact widened-pool shape today). Maps to
# ``ReturnCreatureFromYourGraveyardToBattlefield(or_enchantment=True)``,
# mirroring the ``or_spacecraft=True`` rider on the to-hand sibling
# (Fell Gravship). Tried BEFORE the bare ``return target creature card
# from your graveyard to the battlefield`` regex below for the more-
# specific-first convention; the bare regex's ``$`` anchor on
# ``battlefield`` would not match the ``or enchantment`` rider anyway.
_RETURN_TARGET_CREATURE_OR_ENCHANTMENT_GRAVEYARD_TO_BATTLEFIELD = re.compile(
    r"^return target creature or enchantment card from your graveyard "
    r"to the battlefield$"
)
# "Return target creature card with mana value N or less from your
# graveyard to the battlefield" — narrowed-by-MV reanimation. Surgical
# Suite's door trigger ("with mana value 3 or less") is the EoE/DSK
# surface form; the captured digit lands as the engine class's
# ``max_mv`` rider, same flag the bare regex above leaves unset
# (None = unbounded). Tried BEFORE the bare regex so a phrase
# carrying the MV qualifier wins; the bare regex's ``$`` anchor on
# ``battlefield`` would never match the MV tail anyway, but ordering
# more-specific before more-general is the convention here. Mirrors
# the ``_RETURN_TARGET_CREATURE_GRAVEYARD_TO_HAND_WITH_MV`` shape on
# the to-hand sibling.
_RETURN_TARGET_CREATURE_GRAVEYARD_TO_BATTLEFIELD_WITH_MV = re.compile(
    r"^return target creature card with mana value (\d+) or less "
    r"from your graveyard to the battlefield$"
)
# Multi-sentence Xu-Ifit, Osteoharmonist variant — the printed oracle is
# ``Return target creature card from your graveyard to the battlefield.
# It's a Skeleton in addition to its other types and has no abilities.``
# The ``It's a Skeleton …`` rider is intentionally dropped (the engine
# doesn't yet model "has no abilities" suppression nor in-place subtype
# grants on an arriving creature). This mirrors the matching
# simplification in the static registry — the resulting primitive is
# the same bare ``ReturnCreatureFromYourGraveyardToBattlefield`` the
# bare regex above produces. Kept as its own regex (rather than a
# preprocessing strip) so the simplification is local and explicit:
# the only wrapping that matches this surface verbatim is Xu-Ifit's,
# and any future printing with a different rider tail will hit the
# bare regex's ``$`` anchor and surface a parser miss instead of being
# silently dropped.
_RETURN_TARGET_CREATURE_GRAVEYARD_TO_BATTLEFIELD_AS_SKELETON = re.compile(
    r"^return target creature card from your graveyard "
    r"to the battlefield\. "
    r"It's a Skeleton in addition to its other types "
    r"and has no abilities$"
)
# "Return all creature cards from your graveyard to the battlefield"
# — mass reanimation (Awakening Hall's door trigger; the half-card
# behind Funeral Room // Awakening Hall is the only EoE/DSK printing
# of this exact surface today). Maps to the existing
# ``ReturnAllCreaturesFromYourGraveyardToBattlefield`` engine class.
#
# Literal-line recognizer per the architectural guidance: only one
# card in the data pool fits this exact surface, and the engine has
# no parameterized "all <zone>" sibling primitive yet (no
# ``...ToHand`` mass-recursion engine class), so generalizing across
# destination zones isn't expressible today. Deferred generalization:
# when a future printing adds a sibling shape (e.g.
# ``return all creature cards from your graveyard to your hand``) and
# the engine grows the matching primitive, fold both into one regex
# with a captured destination group rather than copy-paste this
# literal.
_RETURN_ALL_CREATURE_CARDS_GRAVEYARD_TO_BATTLEFIELD = re.compile(
    r"^return all creature cards from your graveyard to the battlefield$"
)
# DSK Enduring cycle dies-trigger body: "return it to the battlefield
# under its owner's control. It's an enchantment." Every Enduring
# creature in DSK (Innocence, Curiosity, Tenacity, Courage, Vitality)
# prints the same self-return body verbatim. The phrase appears as
# the EFFECT tail of an intervening-if line ``When this creature
# dies, if it was a creature, …`` — the gate is handled by the
# ``_INTERVENING_IF_PHRASES`` ``"it was a creature"`` entry; this
# regex matches the bare effect body that lands in the trailing
# group.
#
# The body is two sentences glued together. Reminder text
# (``(It's not a creature.)``) has already been stripped by
# ``_strip_reminder_text`` before line-level parsing, so the
# parenthesized rule reminder doesn't appear here.
#
# Maps to ``EnduringDiesReturnAsEnchantmentEffect()`` — a
# parameterless engine class that reads ``state.resolving_source`` to
# locate the dying card in its owner's graveyard, moves it back onto
# the battlefield under the owner's control, and sets the
# ``enduring_returned_as_enchantment`` flag on the returned permanent
# so the engine's ``is_creature_now`` short-circuits to False.
_ENDURING_DIES_RETURN_AS_ENCHANTMENT = re.compile(
    r"^return it to the battlefield under its owner's control\. "
    r"It's an enchantment$"
)
# "Return target artifact card from your graveyard to your hand" —
# matches Scour for Scrap's mode (parsed via the modal combinator
# elsewhere, but a primitive worth surfacing in the unified grammar).
_RETURN_TARGET_ARTIFACT_GRAVEYARD_TO_HAND = re.compile(
    r"^return target artifact card from your graveyard to your hand$"
)
# "Put target card from a graveyard on the bottom of its owner's
# library" — Chrome Companion's tap-mana activated ability. Untyped
# grave-target (any card in any player's graveyard is legal); the
# resolution moves the chosen card to the bottom of its owner's
# library. Maps to the existing ``PutCardFromGraveyardOnBottomOfLibrary``
# engine class. Surface form is post-normalization (the ``it `` prefix-
# strip in ``_normalize_creature_effect_phrase`` doesn't fire because
# the phrase starts with the verb ``put``, not a pronoun subject — so
# the ``its`` here survives intact, mirroring the ``put its counters
# on …`` precedent).
_PUT_CARD_FROM_GRAVEYARD_ON_BOTTOM_OF_LIBRARY = re.compile(
    r"^put target card from a graveyard "
    r"on the bottom of its owner's library$"
)
# "Exile target card from a graveyard" — Ghost Vacuum's ``{T}`` activated
# ability. Sibling to the bottom-of-library shape above; same
# ``("card_in_graveyard", id)`` target-kind pool, different destination
# zone. Maps to the new ``ExileTargetCardFromGraveyard`` engine class
# whose resolve also tags the exiled card with the source's id via
# ``exiled_by_source`` so a sibling ``{6}, {T}, Sacrifice this artifact``
# recall ability (Ghost Vacuum's second activation) can find every card
# the same source exiled later. The tag is harmless on a source whose
# only graveyard interaction is this exile primitive — no recall.
_EXILE_CARD_FROM_GRAVEYARD = re.compile(
    r"^exile target card from a graveyard$"
)
# "Target creature gets +N/+M until end of turn" or
# "Target creature gets -N/-M until end of turn". Both signs supported
# (PumpTargetCreatureUntilEOT folds positive and negative bonuses
# through the same pump path). Mixed signs (``+1/-1``) are also
# accepted to keep the grammar uniform — no EoE card prints that today
# but the regex stays the simplest correct shape.
_PUMP_TARGET_CREATURE_UEOT = re.compile(
    r"^target creature gets ([+-]\d+)/([+-]\d+) until end of turn$"
)
# "Up to one target creature gets ±N/±M until end of turn" — optional
# pump/debuff with no fizzle when the battlefield has no creatures.
# Surfaces on Dubious Delicacy's ETB ("up to one target creature gets
# -3/-3 until end of turn"). Maps to
# ``PumpTargetCreatureUntilEOT(power_bonus=N, toughness_bonus=M,
# optional=True)`` — the ``optional`` flag flips the engine class's
# ``target_count_range`` to ``(0, 1)`` and ``requires_target`` to
# False, so a no-target resolution is a clean no-op rather than a
# fizzle.
_PUMP_UP_TO_ONE_TARGET_CREATURE_UEOT = re.compile(
    r"^up to one target creature gets ([+-]\d+)/([+-]\d+) "
    r"until end of turn$"
)
# "Target creature an opponent controls gets -N/-M until end of turn" —
# pump variant restricted to opposing creatures (Sinister Cryologist's
# ETB debuff). Maps to ``PumpTargetCreatureUntilEOT(opponents_only=True)``
# which gates ``legal_targets`` to creatures NOT controlled by the
# resolving controller. Negative bonuses are typical here (debuff
# semantics) but the regex accepts both signs so a future card that
# prints a positive opponent-only pump (rare but possible) parses
# without further grammar work — symmetric with the bare pump regex
# above.
_PUMP_TARGET_OPPONENT_CREATURE_UEOT = re.compile(
    r"^target creature an opponent controls gets ([+-]\d+)/([+-]\d+) "
    r"until end of turn$"
)
# "Creatures your opponents control get -N/-M until end of turn" /
# "...until your next turn" — mass debuff on every creature an opponent
# controls (Mouth of the Storm's ETB). Maps to
# ``PumpCreaturesOpponentsControlUntilEOT(power_bonus=N, toughness_bonus=M,
# until_caster_next_turn=...)``. The ``until your next turn`` tail
# extends the duration past the casting turn's cleanup; the bare
# ``until end of turn`` tail clears at the normal cleanup step.
_PUMP_OPPONENT_CREATURES_UEOT = re.compile(
    r"^creatures your opponents control get ([+-]\d+)/([+-]\d+) "
    r"until (end of turn|your next turn)$"
)
# "[Other ]creatures you control get +N/+M[ and gain <kw>[ and <kw2>...]]
# until end of turn" — anthem-style mass pump with optional
# source-self exclusion (``other``) and optional grant-keywords rider
# (``and gain <kw>``). Maps to ``PumpCreaturesYouControlUntilEOT``
# with ``exclude_source`` flipped by group-1 capture and
# ``granted_keywords`` populated from the body parsed by
# ``_parse_keyword_grant_body`` against the keyword-line whitelist.
#
# Surface coverage today:
#   * bare ``creatures you control get +N/+M until end of turn`` —
#     Dual-Sun Adepts' ``{5}`` mass pump (no exclusion, no grants).
#   * ``other creatures you control get +1/+0 and gain deathtouch
#     until end of turn`` — Syr Vondam, the Lucent's enters/attacks
#     trigger (exclude_source + granted_keywords={'deathtouch'}).
#
# Both signs accepted for the bonuses (no EoE printing uses negative
# self-mass pump today, but the uniform shape parses one without
# further grammar work). The ``other`` and ``and gain <kw>`` clauses
# are independently optional — every combination flows through the
# same engine class with the corresponding flags/fields populated.
_PUMP_YOUR_CREATURES_UEOT = re.compile(
    r"^(other )?creatures you control get ([+-]\d+)/([+-]\d+)"
    r"(?: and gain (.+))? until end of turn$"
)
# "Target creature you control gets +N/+M until end of turn" —
# you-control-only single-target pump (Zookeeper Mechan's ``{6}{R}``
# sorcery-speed activated ability). Maps to
# ``PumpTargetCreatureUntilEOT(you_control_only=True)`` whose
# ``legal_targets`` filters the standard creature pool to creatures the
# resolving controller controls. Tried BEFORE the bare ``target creature
# gets ... UEOT`` regex so the ``you control`` qualifier wins; the bare
# regex's ``$`` anchor wouldn't match the qualifier anyway, but
# ordering more-specific before more-general is the convention in this
# dispatcher.
_PUMP_TARGET_CREATURE_YOU_CONTROL_UEOT = re.compile(
    r"^target creature you control gets ([+-]\d+)/([+-]\d+) "
    r"until end of turn$"
)
# "Double the power of target creature you control until end of turn"
# — Mightform Harmonizer's landfall trigger body. The
# ``DoubleTargetCreaturesPowerUEOT`` engine class adds the target's
# current effective power as a temporary power bonus that clears at
# end-of-turn cleanup. Distinct from the ``gets +N/+0`` UEOT pump
# regexes above: the bonus amount is computed dynamically from the
# target's current power at resolve time, not a fixed constant. Maps
# to the engine class with no parameters — the legal-target predicate
# (``target creature you control``) and duration (``until end of
# turn``) are both encoded by the class.
_DOUBLE_TARGET_CREATURE_YOU_CONTROL_POWER_UEOT = re.compile(
    r"^double the power of target creature you control until end of "
    r"turn$"
)
# "Target creature can't be blocked this turn." — bare unblockable-
# UEOT grant on a creature target. Distinct from the larger pump-with-
# unblockable-rider shapes (Atomic Microsizer's attack trigger,
# manland animation's optional unblockable rider) — this primitive
# is the bare make-unblockable surface, with no P/T pump and no
# granted keywords. Matches Enter the Enigma's ``Target creature
# can't be blocked this turn.`` prefix; reusable for any future card
# that prints the same surface.
_TARGET_CREATURE_CANT_BE_BLOCKED_THIS_TURN = re.compile(
    r"^target creature can't be blocked this turn$"
)
# "Another target attacking creature can't be blocked this turn." —
# Clammy Prowler's attack-trigger body. Same engine class as the bare
# unblockable-UEOT primitive but with the ``attacking_only=True`` and
# ``exclude_self=True`` flags flipped — the legal-target pool is
# narrowed at selection time to "creatures currently flagged as
# attacking, excluding the source itself". Tried BEFORE the bare
# regex because the bare regex's ``$`` anchor would never match this
# longer form anyway, but ordering more-specific before more-general
# is the convention here.
_ANOTHER_TARGET_ATTACKING_CREATURE_CANT_BE_BLOCKED_THIS_TURN = re.compile(
    r"^another target attacking creature can't be blocked this turn$"
)
# "Target creature with power N or less can't be blocked this turn." —
# Ragged Playmate's activated ability body. Same engine class as the
# bare unblockable-UEOT primitive with the ``power_at_most=N`` filter
# narrowing the legal-target pool by effective power at selection
# time. Tried BEFORE the bare ``target creature can't be blocked``
# regex so the more-specific power-filter form wins.
_TARGET_CREATURE_WITH_POWER_AT_MOST_CANT_BE_BLOCKED_THIS_TURN = re.compile(
    r"^target creature with power (\d+) or less can't be blocked this turn$"
)
# Controller loses N life. Surface forms ``lose N life`` (post-compound
# split, where the carried-over subject is implicit — Decode
# Transmissions, Elegy Acolyte, Voidforged Titan all print
# ``you draw a card and lose 1 life``) and ``you lose N life`` (the
# explicit form, included for symmetry with ``you gain N life``). Both
# map to the same ``LoseLife`` primitive — the ``you`` is purely
# surface noise here because the controller is already implicit in
# every effect's resolve signature.
_LOSE_N_LIFE = re.compile(r"^(?:you )?lose (\d+) life$")
# "You may pay {COST}. If you do, <inner-effect>." — the ``OptionalManaCost
# ThenEffect`` wrapper. The inner-effect phrase is parsed recursively via
# the unified grammar, so any single-effect (or compound) primitive that
# parses on its own can serve as the ``If you do, …`` payload — e.g.
# ``put a +1/+1 counter on this creature`` (Oreplate Pangolin) or
# ``draw a card and you gain 1 life`` (synthetic compound).
#
# Cost shape is the standard ``{...}`` braced mana-cost mini-language
# (e.g. ``{2}`` or ``{1}{R}``); it's captured verbatim and handed to
# ``OptionalManaCostThenEffect`` which parses it via
# ``parse_mana_cost`` at resolve time.
#
# Note the embedded period after the cost: oracle text always prints
# ``You may pay {N}. If you do, …`` as TWO sentences. The
# normalization layers (``_normalize_creature_effect_phrase`` and
# ``_normalize_spell_effect_phrase``) only strip the OUTER trailing
# period — the inner sentence-boundary period stays as a literal
# match here.
_YOU_MAY_PAY_THEN_EFFECT = re.compile(
    r"^you may pay (\{[^}]+(?:\}\{[^}]+)*\})\. If you do, (.+)$"
)
# "Target creature gets +N/+M and gains <kw>[ and <kw2>...] until end of
# turn." — pump + grant-keywords compound. The granted-keywords body
# (between ``and gains `` and `` until end of turn``) is one or more
# evergreen-keyword tokens joined by `` and `` and validated against the
# keyword-line whitelist before the parsed effect is returned. Four
# subject-shapes share the same regex via the optional prefix groups:
#
# * Bare ``target creature gets ... UEOT`` (Rig for War, Dark Endurance).
# * ``target creature you control gets ... UEOT`` (Kavaron Turbodrone,
#   Molecular Modifier — activated/triggered abilities not yet wired by
#   the surrounding builders, but the effect primitive is in place).
# * ``another target creature you control gets ... UEOT`` (Hardened
#   Escort — Duskmourn). The ``another`` qualifier excludes the trigger
#   source from the legal pool, encoded by the engine class's
#   ``exclude_self=True`` flag.
# * ``target attacking creature gets ... UEOT`` (Most Valuable Slayer —
#   Duskmourn). The ``attacking`` qualifier narrows the legal pool to
#   creatures currently flagged ``attacking=True``, encoded by the
#   engine class's ``attacking_only=True`` flag. Mutually exclusive with
#   the ``another``/``you control`` qualifiers in print (no DSK card
#   layers them) but the regex captures each group independently so the
#   handler can flip flags orthogonally — adding a hypothetical
#   ``another target attacking creature you control gets ...`` line
#   later wouldn't need further grammar work.
#
# Future cards with multiple keywords (``... and gains first strike and
# reach UEOT``) parse through the same shape. The ``another``,
# ``attacking``, and ``you control`` groups are captured separately so
# the handler can flip the matching engine flags independently —
# ``another`` always co-occurs with ``you control`` in the surface
# forms in print today, but the regex doesn't enforce that pairing
# because the engine's ``exclude_self=True`` flag is meaningful even on
# bare-target effects.
_PUMP_AND_GAIN_KEYWORDS_UEOT = re.compile(
    r"^(another )?target (attacking )?creature (you control )?"
    r"gets ([+-]\d+)/([+-]\d+) and gains "
    r"(.+) until end of turn$"
)
# "You may discard a card. If you do, draw N cards." — the
# ``OptionalDiscardCardThenDrawN`` wrapper. The amount tail accepts the
# Scryfall standard variants for "1": the article ``a`` for the singular
# ``draw a card`` and the spelled-out small cardinals (``draw two
# cards`` is Melded Moxite's printed form) plus bare digits for any
# future card that uses ``draw 3 cards``. Inner-effect parsing is NOT
# delegated to ``_parse_effect`` here because the engine class hardcodes
# the inner draw — only the count needs to be extracted from the
# surface text.
_OPTIONAL_DISCARD_THEN_DRAW = re.compile(
    rf"^you may discard a card\. If you do, "
    rf"draw (a|\d+|{_SMALL_CARDINAL_PATTERN}) cards?$"
)
# "You may discard your hand. If you do, draw N cards." — Vaultguard
# Trooper's tapped-creatures-gated end-step trigger inner effect (the
# ``ConditionalOnTappedCreatures`` wrapper sits at the trigger-line
# level, not inside the effect-phrase grammar). Distinct from the
# single-card discard-then-draw above: discards the WHOLE hand, not a
# single chosen card. Maps to ``DiscardHandThenDrawN(amount=N)``. The
# amount tail accepts the Scryfall-printed forms — bare digits and
# spelled-out small cardinals (``two`` is what Vaultguard prints);
# ``a`` would collapse to 1 if a future card prints
# ``draw a card`` after a hand discard.
_OPTIONAL_DISCARD_HAND_THEN_DRAW = re.compile(
    rf"^you may discard your hand\. If you do, "
    rf"draw (a|\d+|{_SMALL_CARDINAL_PATTERN}) cards?$"
)
# "You may put a (creature|land) card from your hand onto the
# battlefield." — parameterized over the card-type word. Sledge-Class
# Seedship (creature, attack trigger) and Burgeoning (land, opponent-
# plays-land trigger) both fit this shape; the captured group drives
# the engine-class dispatch:
#   creature → ``OptionalPutCreatureFromHandOntoBattlefield()``
#   land     → ``OptionalPutLandFromHandOntoBattlefield()``
# Both classes delegate the optional choice through the resolving
# player's agent and treat declining as a legal no-op. A future
# printing with a different card-type word (artifact, enchantment, etc.)
# would extend the alternation and add a matching engine class.
_OPTIONAL_PUT_PERMANENT_FROM_HAND_ONTO_BATTLEFIELD = re.compile(
    r"^you may put a (creature|land) card from your hand "
    r"onto the battlefield$"
)
# "You may sacrifice another creature or artifact. If you do, this
# creature gets +N/+M until end of turn." — the
# ``OptionalSacAnotherForPumpSelfUEOT`` wrapper. Mirrors the surface
# form on Comet Crawler's attack trigger. Pump-self-only (no
# ``granted_keywords`` rider parsed today; no EoE printing pairs the
# self-sac pump with an inner keyword grant). Tried BEFORE the more
# general ``OptionalSacAnotherForEffect`` regex so the self-pump shape
# wins and produces the specialized engine class instead of falling
# through.
_OPTIONAL_SAC_ANOTHER_FOR_PUMP_SELF_UEOT = re.compile(
    r"^you may sacrifice another creature or artifact\. If you do, "
    r"this creature gets ([+-]\d+)/([+-]\d+) until end of turn$"
)
# "You may sacrifice a land or Lander. If you do, search your library
# for up to N basic land cards, put them onto the battlefield tapped,
# then shuffle." — Larval Scoutlander's ETB body (the only EoE/DSK
# printing fitting this exact shape today). Maps to
# ``OptionalSacLandOrLanderForBasicLandRamp(land_count=N)`` whose
# resolve handles the optional sac + tutor-tapped sequence via the
# existing combinator. Captures the count tail (spelled-out small
# cardinal, e.g. ``two``) for forward-compat with any future printing
# that varies the count. Literal-line otherwise: the leading "you may
# sacrifice a land or Lander" subject is unique to this card today;
# broader generalization (parameterizing the sacrificed object's
# noun phrase, or merging with the ``OptionalSacAnotherForEffect``
# shape) is deferred until a second printing with a varying
# sacrificed-object surface ships, at which point the right move is a
# parameterized regex over <subject> and <inner-effect> with the
# inner phrase parsed recursively through ``_parse_effect_phrase``.
_OPTIONAL_SAC_LAND_OR_LANDER_FOR_BASIC_LAND_RAMP = re.compile(
    rf"^you may sacrifice a land or Lander\. If you do, "
    rf"search your library for up to "
    rf"(\d+|{_SMALL_CARDINAL_PATTERN}) basic land cards, "
    rf"put them onto the battlefield tapped, then shuffle$"
)
# "You may sacrifice another creature or artifact. If you do, <inner-
# effect>." — the general ``OptionalSacAnotherForEffect`` wrapper.
# Inner-effect phrase is parsed recursively via the unified grammar so
# any single-effect (or compound) primitive that parses on its own can
# serve as the consequence (e.g. ``draw a card`` on Swarm Culler, or a
# ``put a +1/+1 counter on this creature`` on a future printing).
# Failure to parse the inner rejects the whole line so the wrapping
# card stays unclaimed rather than silently dropping the consequence.
_OPTIONAL_SAC_ANOTHER_FOR_EFFECT = re.compile(
    r"^you may sacrifice another creature or artifact\. If you do, (.+)$"
)
# "Each opponent discards a card" / "Each opponent discards N cards" —
# untargeted opponent-wide discard (Virus Beetle's ETB and any other
# wide-discard surface). The amount tail accepts the article ``a`` for
# the singular form (Scryfall's printed shape — every EoE card uses
# ``a card``) plus bare digits and spelled-out small cardinals so a
# future ``each opponent discards two cards`` line parses without
# further grammar work. ``a`` collapses to count=1 in the handler.
_EACH_OPPONENT_DISCARDS = re.compile(
    rf"^each opponent discards (a|\d+|{_SMALL_CARDINAL_PATTERN}) cards?$"
)
# "Each opponent sacrifices a nontoken creature of their choice." —
# Susurian Dirgecraft's ETB trigger body. Untargeted opponent-wide
# sacrifice; each affected opponent picks via their agent's
# ``choose_sacrifice``. Maps to ``EachOpponentSacrificesCreature()``
# whose engine class skips opponents with no legal nontoken creature.
# The ``of their choice`` tail is verbatim oracle text — the standard
# MTG wording that places target choice on the sacrificing player
# rather than the source's controller. No targeted variant exists in
# the current pool, so this is a single literal-line regex with no
# captured groups.
_EACH_OPPONENT_SACRIFICES_CREATURE = re.compile(
    r"^each opponent sacrifices a nontoken creature of their choice$"
)
# "Defending player mills N cards" — Specimen Freighter's attack
# trigger body. The defending-player surface is the multiplayer-aware
# wording for an attack-trigger mill; the engine simplifies this to
# "each opponent mills N" via ``EachOpponentMillsCards(amount=N)``,
# which is correct in 2-player and a documented approximation in
# multiplayer (per the engine class docstring). The amount tail
# accepts spelled-out small cardinals (``four`` on Specimen
# Freighter) plus bare digits for any future printing.
_DEFENDING_PLAYER_MILLS_N_CARDS = re.compile(
    rf"^defending player mills (\d+|{_SMALL_CARDINAL_PATTERN}) cards?$"
)
# "Exile target creature or Spacecraft." — Gravkill's printed surface.
# The static ``cards/eoe_spells.py`` Gravkill entry intentionally fudges
# this to ``ExileTargetCreature()`` because Spacecraft aren't yet a
# legal target type for the engine's exile path; the parser mirrors
# that fudge to stay in parity. When the engine grows a target kind
# that covers Spacecraft, both sides will need to switch in lockstep
# to a Spacecraft-aware engine class.
_EXILE_TARGET_CREATURE_OR_SPACECRAFT = re.compile(
    r"^exile target creature or Spacecraft$"
)
# "Destroy target artifact or tapped creature." — Radiant Strike's
# destroy clause. Maps to ``DestroyTargetArtifactOrTappedCreature()``
# whose ``legal_targets`` filters to artifacts (any) or creatures that
# are currently tapped. The "permanent" target kind is used because
# either an artifact or a creature can be hit; legality is policed by
# the engine class's predicate.
_DESTROY_TARGET_ARTIFACT_OR_TAPPED_CREATURE = re.compile(
    r"^destroy target artifact or tapped creature$"
)
# "Target artifact or creature's owner puts it on their choice of the
# top or bottom of their library." — Lost in Space's bounce-to-library
# clause. Maps to ``OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOf
# Library()`` whose target kind is ``("permanent", id)`` so artifacts
# (any) and creatures (any) are both legal; the chosen target's owner
# (not the spell's controller) decides top vs bottom at resolution.
_OWNER_PUTS_TARGET_ARTIFACT_OR_CREATURE_ON_TOP_OR_BOTTOM = re.compile(
    r"^target artifact or creature's owner puts it on their choice of "
    r"the top or bottom of their library$"
)
# "Target nonland permanent's owner puts it on their choice of the top
# or bottom of their library." — Vanish from Sight's bounce-to-library
# clause. Same engine class as the artifact-or-creature shape with the
# ``nonland_permanent=True`` flag set; the wider target pool is the
# only behavioral difference. The owner-decides top-vs-bottom path,
# token-ceases-to-exist handling, and bounce cleanup are all shared.
_OWNER_PUTS_TARGET_NONLAND_PERMANENT_ON_TOP_OR_BOTTOM = re.compile(
    r"^target nonland permanent's owner puts it on their choice of "
    r"the top or bottom of their library$"
)
# "Tap target artifact or creature." — Mental Modulation's middle
# clause and any future card that ships the same surface form. Maps to
# ``TapTargetArtifactOrCreature()``. No-op if the chosen target is
# already tapped at resolution.
_TAP_TARGET_ARTIFACT_OR_CREATURE = re.compile(
    r"^tap target artifact or creature$"
)
# "Tap another target creature." — Starport Security's activated
# ability. ``another`` excludes the source creature (reads
# ``state.resolving_source`` to filter). Maps to
# ``TapAnotherTargetCreature()``. No-op if the chosen target is already
# tapped at resolution. Tried as a separate primitive from the
# ``tap target artifact or creature`` shape because the engine class
# differs (creature-only legal pool, source-self exclusion) and the
# surface tail (no ``artifact or`` qualifier, ``another`` self-exclusion
# fragment) is also distinct.
_TAP_ANOTHER_TARGET_CREATURE = re.compile(
    r"^tap another target creature$"
)
# "Tap target artifact or creature an opponent controls and put a stun
# counter on it" — Mechanozoa's ETB. Single primitive (both clauses
# apply to the same target) — the ``and`` in this surface is a
# conjunction within ONE effect, not a compound joiner. Maps to
# ``TapTargetOpponentPermanentAndStunIt()``. The compound-effect
# splitter does try ``and`` first and produces two pieces, but
# neither piece is a recognized standalone effect, so the splitter
# rejects and we fall through to this single-effect regex.
_TAP_TARGET_OPPONENT_PERMANENT_AND_STUN_IT = re.compile(
    r"^tap target artifact or creature an opponent controls "
    r"and put a stun counter on it$"
)
# "Put a stun counter on up to one target tapped creature." — Cryogen
# Relic's activated ability. Maps to the existing
# ``PutStunCounterOnTappedCreature`` engine class whose
# ``legal_targets`` filters the battlefield to currently-tapped
# creatures and whose ``resolve`` is a no-op when no target is chosen
# (the ``up to one`` optionality fizzles cleanly if the pool is empty).
_PUT_STUN_COUNTER_UP_TO_ONE_TAPPED = re.compile(
    r"^put a stun counter on up to one target tapped creature$"
)
# "Tap up to one target creature and put N stun counters on it." —
# Meat Locker's door trigger (``put two stun counters``). Same
# tap-and-stun-the-same-target shape as
# ``_TAP_TARGET_OPPONENT_PERMANENT_AND_STUN_IT`` (Mechanozoa) — both
# halves apply to the SAME chosen target, with the ``and`` an
# intra-effect conjunction rather than a compound joiner. Differs
# from the Mechanozoa shape on three axes: target qualifier
# (``up to one`` vs ``target … an opponent controls``), target
# type pool (any creature vs artifact-or-creature opponent-controlled),
# and stun-counter count (parameterized digit/cardinal vs the
# Mechanozoa fixed singular). The captured count tail accepts the
# article ``a`` for the singular form (forward-looking — no current
# card prints ``a stun counter on it`` on this exact surface; that
# shape today only appears with the singular Mechanozoa wording),
# bare digits, and spelled-out small cardinals (``two`` is what Meat
# Locker prints). ``a`` collapses to amount=1 in the handler.
# Maps to ``TapAndStunUpToOneTargetCreature(stun_amount=N)``. The
# compound-effect splitter does try ``and`` first and produces two
# pieces, but neither piece is a recognized standalone effect, so
# the splitter rejects and we fall through to this single-effect
# regex.
_TAP_UP_TO_ONE_TARGET_CREATURE_AND_STUN = re.compile(
    rf"^tap up to one target creature and put "
    rf"(a|\d+|{_SMALL_CARDINAL_PATTERN}) stun counters? on it$"
)
# "Discard a card" / "you discard a card" / "you discard N cards" —
# controller-discards. The bare ``discard a card`` surface is what
# appears post-compound-split for ``draw a card, then discard a card``
# (Mechan Navigator's becomes-tapped trigger); the ``you `` prefix
# variant is included for symmetry with ``you gain N life`` /
# ``you lose N life``. Maps to ``DiscardSelf(amount=N)``. ``a``
# collapses to amount=1 in the handler.
_DISCARD_N_CARDS = re.compile(
    rf"^(?:you )?discard (a|\d+|{_SMALL_CARDINAL_PATTERN}) cards?$"
)
# "Untap another target permanent" — Nanoform Sentinel's becomes-tapped
# trigger. The ``another`` qualifier excludes the source permanent from
# the legal target pool — the engine class
# ``UntapTargetOtherPermanent`` reads ``state.resolving_source`` at
# resolve time to filter the chosen target. Maps to
# ``UntapTargetOtherPermanent()`` with no parameters (the engine class's
# semantics fully specify the legal-target predicate).
_UNTAP_ANOTHER_TARGET_PERMANENT = re.compile(
    r"^untap another target permanent$"
)
# "Another target creature you control gets +X/+X until end of turn,
# where X is this creature's power" — Hemosymbic Mite's becomes-tapped
# trigger. The variable-X bonus is fixed at "the source's effective
# power at resolve time" — both the regex and the engine class hardcode
# the source-power semantics; future cards with a different X-source
# (e.g. "where X is the number of artifacts you control") would land
# their own primitive. Maps to
# ``PumpTargetOtherCreatureByXOverXWhereXIsSourcePower()`` with no
# parameters — every aspect of the effect (target restriction to other
# creatures the controller controls, X-source from
# ``state.resolving_source``) is encoded by the engine class.
_PUMP_OTHER_TARGET_CREATURE_BY_X_OVER_X_SOURCE_POWER = re.compile(
    r"^another target creature you control gets \+X/\+X until end of "
    r"turn, where X is this creature's power$"
)
# "Put a +1/+1 counter on each creature you control" — Atmospheric
# Greenhouse's ETB trigger body. Mass +1/+1-counter spread on every
# creature the controller controls (the Spacecraft is itself an
# artifact, not a creature, when the ETB fires, so ``exclude_source``
# isn't needed here — the source is naturally outside the firing
# pool). Maps to ``PutPlusOneCounterOnEachCreatureYouControl(amount=1)``.
# Distinct from the ``other than this creature`` variant below, which
# carries an explicit source-exclusion tail.
_PUT_PLUS_ONE_COUNTER_ON_EACH_CREATURE_YOU_CONTROL = re.compile(
    r"^put a \+1/\+1 counter on each creature you control$"
)
# "Put X +1/+1 counters on each creature you control, where X is this
# creature's power" — Ouroboroid's beginning-of-combat trigger body.
# Mass per-controller-creature counter spread scaled by the source's
# current power; the X-source phrase ``where X is this creature's
# power`` reads ``state.resolving_source.power`` at resolve time
# (matching the engine class
# ``PutPlusOneCountersOnEachCreatureYouControlEqualToSourcePower``).
# Literal-line recognizer with no captures — only Ouroboroid in the
# pool prints this exact shape today; broader grammar deferred until
# a second example exists. A future printing whose X-source phrase
# differs (e.g. ``where X is the number of artifacts you control``)
# would land its own primitive rather than parameterizing this one,
# matching the precedent already documented for
# ``_PUMP_OTHER_TARGET_CREATURE_BY_X_OVER_X_SOURCE_POWER`` (where X is
# this creature's power).
_PUT_X_PLUS_ONE_COUNTERS_ON_EACH_CREATURE_YOU_CONTROL_WHERE_X_IS_SOURCE_POWER = (
    re.compile(
        r"^put X \+1/\+1 counters on each creature you control, "
        r"where X is this creature's power$"
    )
)
# "Put a +1/+1 counter on each creature you control other than this
# creature" — Dawnstrike Vanguard's tapped-creatures-gated end-step
# trigger body. Mass +1/+1-counter spread on each creature the
# controller controls except the source itself. Maps to
# ``PutPlusOneCounterOnEachCreatureYouControl(amount=1,
# exclude_source=True)`` — the ``exclude_source=True`` flag reads
# ``state.resolving_source`` at resolve time to skip the source from
# the firing pool. Distinct from the bare ``put a +1/+1 counter on
# each creature you control`` shape (Atmospheric Greenhouse) above,
# which omits the source-exclusion tail.
_PUT_PLUS_ONE_COUNTER_ON_EACH_OTHER_CREATURE_YOU_CONTROL = re.compile(
    r"^put a \+1/\+1 counter on each creature you control other than "
    r"this creature$"
)
# "This creature gets +N/+M until end of turn" — source pumps itself
# (no separate target). Surfaces on Skystinger's block trigger
# ("Whenever this creature blocks a creature with flying, this
# creature gets +5/+0 until end of turn") and the Landfall self-pump
# family (Icecave Crasher +1/+0, Remnant Elemental +2/+0). Both signs
# are accepted in case a future printing self-debuffs (the engine
# class folds positive and negative bonuses through the same path),
# matching the symmetry already documented for the
# ``_PUMP_TARGET_CREATURE_UEOT`` regex above.
#
# Maps to ``PumpSelfUntilEOT(power_bonus=N, toughness_bonus=M)``.
# Distinct from the ``put a +1/+1 counter on this creature`` shape
# (which produces ``PutPlusOneCounterOnSelf``) — counters persist past
# end of turn while the pump bonus clears at cleanup.
#
# The optional ``this Spacecraft `` prefix accepts the activated-
# ability surface where a Spacecraft body explicitly names its own
# source as ``this Spacecraft`` (Debris Field Crusher's ``{1}{R}: This
# Spacecraft gets +2/+0 until end of turn.`` body, post-activated
# normalization to ``this Spacecraft gets +2/+0 until end of turn``).
# Mirrors the ``(?:this Room )?`` precedent on
# ``_DEAL_DAMAGE_TARGET_OPPONENT_CREATURE`` — same self-source pronoun
# pattern Rooms use, here applied to the Spacecraft type. The source-
# identity check happens through the enclosing ability's wiring (an
# activated ability's source is implicitly the host), not the effect
# body — so the prefix is purely a surface-noise admission.
_PUMP_SELF_UEOT = re.compile(
    r"^(?:this creature |this Spacecraft )?"
    r"gets ([+-]\d+)/([+-]\d+) until end of turn$"
)
# "This creature gets +N/+M and gains <kw>[ and <kw2>...] until end of
# turn" — source pumps itself AND grants itself one or more evergreen
# keywords UEOT. Surfaces on Duskmourn Eerie creatures (Dashing
# Bloodsucker — ``this creature gets +2/+0 and gains lifelink until
# end of turn``). Maps to
# ``PumpSelfUntilEOT(power_bonus=N, toughness_bonus=M,
# granted_keywords=...)`` — same engine class as the bare self-pump,
# with the keyword-grant rider. The granted-keywords body is parsed
# via ``_parse_keyword_grant_body`` and validated against the
# keyword-line whitelist; rejection returns ``None`` so an unknown
# keyword keeps the wrapping card unclaimed (mirrors the discipline
# of every other grant-keywords primitive).
#
# Tried BEFORE the bare ``_PUMP_SELF_UEOT`` regex (its ``$`` anchor
# rejects the ``and gains …`` tail anyway, but ordering more-specific
# before more-general is the convention here).
_PUMP_AND_GAIN_KEYWORDS_SELF_UEOT = re.compile(
    r"^(?:this creature )?gets ([+-]\d+)/([+-]\d+) and gains "
    r"(.+) until end of turn$"
)
# "This creature gains <kw>[ and <kw2>...] until end of turn" — pure
# self-grant (no P/T pump). Surfaces on Duskmourn Eerie creatures
# (Cult Healer — ``this creature gains lifelink until end of turn``).
# Maps to ``PumpSelfUntilEOT(power_bonus=0, toughness_bonus=0,
# granted_keywords=...)`` — same engine class as the pump-and-gain
# shape just with zero bonuses, mirroring the
# ``_GRANT_KEYWORDS_TO_TARGET_CREATURE_UEOT`` precedent for the
# targeted variant. Validated via ``_parse_keyword_grant_body``.
_GAIN_KEYWORDS_SELF_UEOT = re.compile(
    r"^(?:this creature )?gains (.+) until end of turn$"
)
# "Another target creature you control gains <kw>[ and <kw2>...] until
# end of turn." — pure keyword-grant variant with no P/T pump,
# restricted to other creatures the controller controls (Blooming
# Stinger's ETB grants deathtouch to another creature you control).
# Maps to ``PumpTargetCreatureUntilEOT(power_bonus=0, toughness_bonus=0,
# granted_keywords=..., you_control_only=True, exclude_self=True)``.
# Distinct from the existing ``_PUMP_AND_GAIN_KEYWORDS_UEOT`` regex
# which requires a leading ``gets +N/+M and gains`` clause; this one
# is for pure-grant phrases. The granted-keywords body is validated
# against the keyword-line whitelist before the parsed effect is
# returned, mirroring the behavior of the pump-and-gain primitive.
_GRANT_KEYWORDS_TO_ANOTHER_TARGET_CREATURE_YOU_CONTROL_UEOT = re.compile(
    r"^another target creature you control gains (.+) until end of turn$"
)
# "Target creature gains <kw>[ and <kw2>...] until end of turn." —
# pure keyword-grant variant with no P/T pump and no control
# restriction (Horrid Vigor's ``Target creature gains deathtouch and
# indestructible until end of turn.`` in DSK). Maps to
# ``PumpTargetCreatureUntilEOT(power_bonus=0, toughness_bonus=0,
# granted_keywords=...)`` — bare-target legal pool, same engine class
# as the pump-and-gain shape just with zero bonuses. Tried AFTER the
# more-specific ``you control``/``opponent controls`` variants so the
# qualified shapes win on phrases that carry their qualifiers; this
# bare-target shape only claims phrases without such qualifiers. The
# granted-keywords body is validated against the keyword-line
# whitelist before the parsed effect is returned, mirroring the
# behavior of the other grant-keywords primitives.
#
# The optional ``attacking `` qualifier narrows the legal-target pool
# to creatures currently flagged ``attacking=True`` — same shape as
# the ``(attacking )?`` group on ``_PUMP_AND_GAIN_KEYWORDS_UEOT``,
# here lifted up to the no-pump grant-only sibling. Tunnel of Hate's
# door-trigger body ``target attacking creature gains double strike
# until end of turn`` is the surface form that motivated the
# parameterization; the engine class's ``attacking_only=True`` flag
# is the same one Hospital Room's pump uses, here flipped on by the
# captured qualifier rather than by a literal-line entry.
_GRANT_KEYWORDS_TO_TARGET_CREATURE_UEOT = re.compile(
    r"^target (attacking )?creature gains (.+) until end of turn$"
)
# "Target artifact or creature gains <kw>[ and <kw2>...] until end of
# turn." — pure keyword-grant on a permanent target whose legal pool
# spans both artifacts and creatures (Reroute Systems mode 0). Maps to
# ``PumpTargetArtifactOrCreatureUntilEOT(granted_keywords=...)`` which
# uses the ``("permanent", id)`` target kind so the chosen permanent
# may be either an artifact (any) or a creature.
#
# No P/T pump captured — the engine class doesn't ship one for this
# surface today (Reroute Systems' +0/+0 default is the only form in
# use). The keyword body is parsed via ``_parse_keyword_grant_body``
# and validated against the keyword-line whitelist; rejection returns
# ``None`` so an unknown keyword keeps the wrapping spell unclaimed.
_PUMP_TARGET_ARTIFACT_OR_CREATURE_GAINS_KEYWORDS_UEOT = re.compile(
    r"^target artifact or creature gains (.+) until end of turn$"
)
# "<source> deals N damage to target tapped creature" — Reroute Systems
# mode 1's tapped-only burn. Maps to ``DealDamageToTargetTappedCreature(
# amount=N)`` whose ``legal_targets`` filters to creatures that are
# currently tapped, with a re-check at resolution in case a between-
# cast-and-resolve untap removes the tapped creature from the legal
# pool. The ``deals`` form (no ``it`` / ``you`` subject) is what
# Scryfall prints for spells that name themselves as the subject; the
# spell-effect normalizer strips the ``<Name>`` prefix before this
# regex sees the phrase, so the surface form here is exactly
# ``deals N damage to target tapped creature``.
_DEAL_DAMAGE_TARGET_TAPPED_CREATURE = re.compile(
    r"^deals (\d+) damage to target tapped creature$"
)
# "Put N charge counters on target Spacecraft or Planet you control" —
# Drill Too Deep mode 0 (and any future card with the same surface).
# Maps to ``PutChargeCountersOnTargetSpacecraftOrPlanetYouControl(amount=
# N)``. Amount is a digit or one of the spelled-out small cardinals
# (Drill Too Deep prints ``Put five charge counters …``); the engine
# class's ``amount`` field is an int either way. The
# ``Spacecraft or Planet`` capitalization is verbatim from the oracle
# text — the spell-effect normalizer only lowercases the first letter
# of the phrase, so inner type-name capitalization survives.
_PUT_CHARGE_COUNTERS_ON_TARGET_SPACECRAFT_OR_PLANET_YOU_CONTROL = re.compile(
    rf"^put (\d+|{_SMALL_CARDINAL_PATTERN}) charge counters on "
    rf"target Spacecraft or Planet you control$"
)
# "Target player mills cards equal to your life total" — Space-Time
# Anomaly's printed form. Maps to
# ``MillTargetPlayer(use_controller_life=True)``. The engine class's
# ``use_controller_life`` flag substitutes the controller's current
# life total for the bare ``amount`` field at resolve time, so no
# numeric capture is needed here — the surface form is a literal
# match.
_MILL_TARGET_PLAYER_EQUAL_TO_YOUR_LIFE = re.compile(
    r"^target player mills cards equal to your life total$"
)
# "Target player mills N cards" — bare-amount mill at a single
# target player. Forward-looking surface form (no EoE card prints
# this exact shape today; Space-Time Anomaly is the variable-amount
# variant above), but the regex is included so a future printing of
# ``target player mills three cards`` parses without further grammar
# work. Tried AFTER the ``equal to your life total`` regex so the
# more-specific shape wins on phrases that carry it.
_MILL_TARGET_PLAYER_N_CARDS = re.compile(
    rf"^target player mills (\d+|{_SMALL_CARDINAL_PATTERN}) cards$"
)
# "Each player creates a <Name> token" — bare-name token-creation
# variant whose subject is each living player (every player creates
# the token under their own control). Surfaces on Edge Rover's
# dies trigger (``each player creates a Lander token``). Maps to
# ``CreateToken(token_printing=..., count=N, each_player=True)``.
# The captured groups mirror the shape of ``_CREATE_BARE_NAME_TOKEN``
# below — count word/digit, optional ``tapped`` slot, and the bare
# subtype name that doubles as the registry key — except the leading
# subject is ``each player creates`` instead of the implicit
# controller-creates default.
_EACH_PLAYER_CREATE_BARE_NAME_TOKEN = re.compile(
    rf"^each player creates ({_TOKEN_COUNT_PATTERN}) (tapped )?"
    rf"([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*) tokens?$"
)
# "Create a Lander token. At the beginning of the end step on your
# next turn, sacrifice that token." — Kav Landseeker's ETB. Single
# primitive that creates a specific Lander then schedules a delayed-
# trigger sac of that exact token at the controller's next end step.
# The two-sentence surface form is captured as a single literal regex
# — the second sentence's ``sacrifice that token`` references the
# token created in the first, so they can't be split into independent
# effects. Tried via the single-effect dispatcher AFTER the compound
# splitter rejects (the splitter tries `` and `` first — there's no
# ``and`` here — and falls through to the sentence-boundary split,
# whose second piece ``at the beginning of the end step on your next
# turn, sacrifice that token`` doesn't standalone-parse, so the
# splitter rejects and the single-effect dispatcher claims the whole
# phrase).
_CREATE_LANDER_AND_SCHEDULE_SAC = re.compile(
    r"^create a Lander token\. "
    r"At the beginning of the end step on your next turn, "
    r"sacrifice that token$"
)


def _parse_keyword_grant_body(body: str) -> frozenset[str] | None:
    """Parse the tail of a pump-and-gain phrase — one or more evergreen
    keywords joined by `` and `` — into the frozenset of engine
    spellings consumed by ``PumpTargetCreatureUntilEOT.granted_keywords``.

    Returns ``None`` if any piece is not a recognized standalone keyword
    so the caller can reject the whole pump-and-gain phrase rather than
    silently dropping a granted ability. Mirrors the keyword-line
    whitelist in ``parser.keywords`` (``_KEYWORD_LINES``) — adding a new
    evergreen keyword there automatically makes it grantable here.
    """
    pieces = [p.strip() for p in body.split(" and ")]
    if not all(p in _KEYWORD_LINES.values() for p in pieces):
        return None
    return frozenset(pieces)


def _parse_amount(token: str) -> int:
    """Convert a Scryfall amount token (digit or spelled-out small
    cardinal) into the integer N. Assumes the token already passed a
    regex that matched either ``\\d+`` or one of
    ``_SMALL_CARDINAL_WORDS``."""
    if token.isdigit():
        return int(token)
    return _SMALL_CARDINAL_WORDS[token]


def _parse_token_count(token: str) -> int:
    """Convert a token-count word into the integer N. Adds the singular
    articles ``a`` / ``an`` (= 1) to ``_parse_amount``'s digit-or-
    spelled-out-cardinal coverage. Assumes the token already passed
    ``_TOKEN_COUNT_PATTERN``."""
    if token in ("a", "an"):
        return 1
    return _parse_amount(token)


def _parse_create_token_effect(phrase: str) -> Effect | None:
    """Try to parse ``phrase`` as a token-creation effect (either the
    stat-bearing creature shape, the bare-name shape, or the
    each-player bare-name shape). Returns ``None`` if no shape
    matches OR if the resolved token name has no ``TOKEN_PRINTINGS``
    entry — the parser doesn't fabricate new token printings on
    the fly.
    """
    m = _CREATE_CREATURE_TOKEN.match(phrase)
    if m is not None:
        count = _parse_token_count(m.group(1))
        tapped = m.group(2) is not None
        # Stats and color word are consumed by the regex but not used
        # to look up the token — the registered ``TOKEN_PRINTINGS``
        # entry is the canonical source of those values. Mismatches
        # between oracle text and registry would be a static-vs-
        # parser parity bug caught downstream.
        subtype = m.group(6)
        if subtype not in TOKEN_PRINTINGS:
            return None
        return CreateToken(
            token_printing=TOKEN_PRINTINGS[subtype],
            count=count,
            tapped=tapped,
        )
    m = _CREATE_BARE_NAME_TOKEN.match(phrase)
    if m is not None:
        count = _parse_token_count(m.group(1))
        tapped = m.group(2) is not None
        name = m.group(3)
        if name not in TOKEN_PRINTINGS:
            return None
        return CreateToken(
            token_printing=TOKEN_PRINTINGS[name],
            count=count,
            tapped=tapped,
        )
    # ``each player creates a <Name> token`` — every living player
    # creates the bare-name token under their own control. Surfaces
    # on Edge Rover's dies trigger. The captured groups mirror the
    # bare-name shape; the only difference is the leading ``each
    # player creates`` subject (instead of the controller-implicit
    # default) and the resulting ``each_player=True`` flag on the
    # produced ``CreateToken`` engine class.
    m = _EACH_PLAYER_CREATE_BARE_NAME_TOKEN.match(phrase)
    if m is not None:
        count = _parse_token_count(m.group(1))
        tapped = m.group(2) is not None
        name = m.group(3)
        if name not in TOKEN_PRINTINGS:
            return None
        return CreateToken(
            token_printing=TOKEN_PRINTINGS[name],
            count=count,
            tapped=tapped,
            each_player=True,
        )
    return None


# Sentence-boundary split for compound effect phrases — period
# followed by whitespace and an uppercase letter, with a negative
# lookahead for ``If you do,`` so the
# ``You may pay/sacrifice/discard {COST}. If you do, …`` family
# stays welded together (the spell-builder splitter applies the same
# guard at the line level; we mirror it here at the effect-phrase
# level so triggered abilities with multi-sentence bodies — e.g.
# Dauntless Scrapbot's ``exile each opponent's graveyard. Create a
# Lander token.`` — split at the right boundary).
_EFFECT_SENTENCE_BOUNDARY = re.compile(r"(?<=\.) (?=[A-Z](?!f you do,))")


def _split_effect_sentences(phrase: str) -> list[str]:
    """Split a normalized effect phrase at internal sentence
    boundaries, preserving the trailing period on the leading piece(s)
    and lowercasing the first letter of each non-leading piece so each
    piece passes back through ``_parse_effect`` in normalized form.

    Returns ``[phrase]`` when no internal sentence boundary exists.
    The leading piece keeps its trailing period (so it can match
    primitives whose surface form ends mid-phrase, like Codecracker
    Hound's ``look at the top two cards of your library. Put one
    into your hand and the other into your graveyard``); we strip the
    trailing period from EVERY piece before returning so each one
    matches the normalized regex shape (no trailing period). The
    embedded period in Codecracker's surface is part of a single
    primitive's literal regex, so its phrase doesn't reach this
    splitter — it claims through the single-effect dispatcher first.

    Non-leading pieces also have a leading ``then `` connective stripped
    when present. Scryfall prints sentence-joined sequenced effects
    with ``Then <effect>.`` as the second sentence (Extinguisher
    Battleship's ETB body — ``destroy target noncreature permanent.
    Then this Spacecraft deals 4 damage to each creature.``); the
    ``Then`` is purely a sequencing connective with the same semantic
    weight as the ``, then `` joiner the compound parser accepts on
    comma-joined surfaces, so stripping it here lands the inner
    sentence on the same single-effect grammar without further
    branching.
    """
    pieces = _EFFECT_SENTENCE_BOUNDARY.split(phrase)
    if len(pieces) == 1:
        return pieces
    out: list[str] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        if piece.endswith("."):
            piece = piece[:-1]
        if i > 0 and piece and piece[0].isupper():
            piece = piece[0].lower() + piece[1:]
        if i > 0 and piece.startswith("then "):
            piece = piece[len("then "):]
        out.append(piece)
    return out


def _parse_compound_effect(phrase: str) -> Effect | None:
    """Try to parse ``phrase`` as a compound effect: two or more
    primitive effects joined by `` and ``, by ``, then ``, or by
    sentence boundary (``. ``).

    Splits on whichever joiner appears (`` and `` is preferred — see
    below), parses each piece via ``_parse_effect``, and returns a
    ``Sequence`` only if EVERY piece parses cleanly. Returns ``None``
    if the phrase contains no recognized joiner, or if any piece
    fails to parse — never falls back to single-effect parsing on the
    original (the caller does that).

    The strict all-or-nothing rule keeps us from accidentally
    consuming joiner-bearing single effects (e.g. the
    ``search your library for an artifact card, reveal it, put it into
    your hand, then shuffle`` artifact-tutor primitive whose ``, then
    `` is part of one effect's surface): if a split-and-parse attempt
    fails on any piece, we punt and let the caller try the single-
    effect grammar on the unsplit phrase. The artifact-tutor primitive
    in particular survives because its ``shuffle`` half doesn't parse
    as a standalone effect, so the ``, then `` split rejects and the
    full-phrase regex wins.

    Joiner precedence: `` and `` is tried first; ``, then `` second;
    sentence-boundary (``. ``) last. The order matters because the
    joiners can co-occur (a future ``draw a card and you gain 1 life,
    then discard a card`` would have both ``and`` and ``, then``) and
    splitting on the leftmost-listed joiner preserves the same
    left-to-right resolve order the engine expects from ``Sequence``
    regardless of which joiner the printing chose. The sentence-
    boundary split is last so single-effect primitives whose regex
    contains an internal period (Codecracker Hound's two-sentence
    look-at-top-two phrase) still match through the single-effect
    dispatcher first; only when that fails AND no other joiner
    applies do we carve at sentence boundaries.

    A compound where two or more sub-effects each require a target is
    also rejected (returns ``None``) — ``Sequence`` only supports a
    single targeting sub-effect (the cast-time target-selection
    pipeline is single-choice). A multi-target spell needs a
    different combinator and isn't claimed by this slice.
    """
    if " and " in phrase:
        pieces = phrase.split(" and ")
    elif ", then " in phrase:
        pieces = phrase.split(", then ")
    elif _EFFECT_SENTENCE_BOUNDARY.search(phrase):
        pieces = _split_effect_sentences(phrase)
        if len(pieces) < 2:
            return None
    else:
        return None
    parsed: list[Effect] = []
    for piece in pieces:
        eff = _parse_effect(piece)
        if eff is None:
            return None
        parsed.append(eff)
    if sum(1 for e in parsed if e.requires_target()) > 1:
        return None
    return Sequence(effects=tuple(parsed))


def _parse_effect(phrase: str) -> Effect | None:
    """Parse a *normalized* effect phrase into an ``Effect``.

    Normalized form has no source-subject prefix, no trailing period,
    and a lowercased first letter — see ``_normalize_*`` helpers below
    for the per-context surface stripping.

    Compound effects (two or more primitives joined by `` and ``) are
    tried first via ``_parse_compound_effect``; if every split-piece
    parses, returns a ``Sequence``. If the compound attempt fails
    (any piece doesn't parse), falls through to the single-effect
    grammar on the original phrase — preserving the ability of a
    single-effect regex to match a phrase that happens to contain
    `` and ``.
    """
    compound = _parse_compound_effect(phrase)
    if compound is not None:
        return compound
    if _PREVENT_ALL_COMBAT_DAMAGE_THIS_TURN.match(phrase) is not None:
        return PreventAllCombatDamageThisTurn()
    m = _DEAL_DAMAGE_ANY_TARGET.match(phrase)
    if m is not None:
        return DealDamageAnyTarget(amount=int(m.group(1)))
    # ``deals damage equal to its power to any target`` — Infernal
    # Phantom's dies trigger body in DSK. Mirrors the legal-target pool
    # of the bare ``deals N damage to any target`` shape but reads the
    # damage amount from the source's effective power at fire time
    # rather than a printed integer. Tried right after the bare shape
    # so a future ``deals damage equal to its power to target creature``
    # variant (no DSK printing today) can land its own primitive in the
    # same dispatch neighborhood.
    if _DEAL_DAMAGE_EQUAL_TO_SOURCE_POWER_ANY_TARGET.match(phrase) is not None:
        return DealDamageEqualToSourcePowerAnyTarget()
    # ``deals N damage to target creature an opponent controls`` — Nova
    # Hellkite's ETB. Tried BEFORE the bare ``deals N damage to target
    # creature`` regex because the bare regex's ``$`` anchor would not
    # match this surface anyway, but ordering more-specific before
    # more-general is the convention here.
    m = _DEAL_DAMAGE_TARGET_OPPONENT_CREATURE.match(phrase)
    if m is not None:
        return DealDamageToOpponentCreature(amount=int(m.group(1)))
    # ``[this Room ]deals damage equal to <count-source> to target
    # creature an opponent controls`` — Warmaker Gunship (count-source
    # ``artifacts you control``, EoE) and Roaring Furnace (count-source
    # ``cards in your hand``, DSK Room half). The captured count-source
    # surface text routes through
    # ``_DEAL_DAMAGE_COUNT_SOURCE_DISPATCH`` to the matching engine
    # class; both share the ``target creature an opponent controls``
    # legal-target pool (creatures NOT controlled by the resolving
    # controller) with the dynamic damage amount read at resolve time.
    m = _DEAL_DAMAGE_BY_COUNT_TARGET_OPPONENT_CREATURE.match(phrase)
    if m is not None:
        cls = _DEAL_DAMAGE_COUNT_SOURCE_DISPATCH.get(m.group(1))
        if cls is not None:
            return cls()
    # ``deals X damage to target attacking or blocking creature, where
    # X is N plus the number of creatures and/or Spacecraft you
    # control`` — Focus Fire's body (after the ``<Name> `` prefix
    # strip in ``_normalize_spell_effect_phrase``). The captured
    # integer base parameterizes ``FocusFireEffect(base=N)``; the
    # dynamic +creatures-and-Spacecraft count is read at resolve
    # time inside the engine class. See the regex's docstring for
    # the deferred-generalization plan when a second card with this
    # shape ships.
    m = _DEAL_DAMAGE_X_BASE_PLUS_FRIENDLY_CREATURE_OR_SPACECRAFT_COUNT.match(
        phrase
    )
    if m is not None:
        return FocusFireEffect(base=int(m.group(1)))
    # ``deals N damage to target creature. If excess damage was dealt
    # this way, create a Lander token`` — Orbital Plunge's full two-
    # sentence body. Compound parsing fails on this phrase because the
    # rider's anaphoric ``If excess damage was dealt this way, create
    # a Lander token`` doesn't parse as a standalone effect, so
    # ``_parse_compound_effect`` returns None and dispatch reaches this
    # multi-sentence literal-line recognizer. The captured integer
    # parameterizes ``DealDamageToTargetCreatureWithExcessLanderRider
    # (amount=N)`` whose resolve step gates Lander creation on the
    # excess-damage check at resolve time. See the regex's docstring
    # for the deferred-generalization plan when a second card with
    # this shape ships.
    m = _DEAL_DAMAGE_N_TARGET_CREATURE_WITH_EXCESS_LANDER_RIDER.match(phrase)
    if m is not None:
        return DealDamageToTargetCreatureWithExcessLanderRider(
            amount=int(m.group(1)),
        )
    # ``deals N damage to target creature. It deals M damage instead
    # if you control a Spacecraft`` — Invasive Maneuvers's full two-
    # sentence body. Compound parsing fails on this phrase because the
    # rider's anaphoric ``It deals M damage instead`` substitution-
    # syntax doesn't parse as a standalone effect, so
    # ``_parse_compound_effect`` returns None and dispatch reaches this
    # multi-sentence literal-line recognizer. The captured integers
    # parameterize ``DealDamageToTargetCreatureSpacecraftBoosted(
    # base=N, boosted=M)`` whose resolve step picks ``base`` vs
    # ``boosted`` by inspecting the controller's battlefield for a
    # Spacecraft. See the regex's docstring for the deferred-
    # generalization plan when a second card with this shape ships.
    m = _DEAL_DAMAGE_N_TARGET_CREATURE_M_IF_SPACECRAFT.match(phrase)
    if m is not None:
        return DealDamageToTargetCreatureSpacecraftBoosted(
            base=int(m.group(1)),
            boosted=int(m.group(2)),
        )
    # ``deals N damage to target opponent`` — Biotech Specialist's
    # sacrifice-an-artifact trigger body. Tried BEFORE the bare
    # ``deals N damage to target creature`` regex because the bare
    # regex's ``$`` anchor on ``target creature`` would not match the
    # ``target opponent`` tail anyway, but ordering more-specific
    # before more-general is the convention here. Maps to
    # ``DealDamageToTargetOpponent(amount=N)`` which gates the legal-
    # target pool to opponents (not the source's controller).
    m = _DEAL_DAMAGE_TARGET_OPPONENT.match(phrase)
    if m is not None:
        return DealDamageToTargetOpponent(amount=int(m.group(1)))
    m = _DEAL_DAMAGE_TARGET_CREATURE.match(phrase)
    if m is not None:
        return DealDamageToTargetCreature(amount=int(m.group(1)))
    # ``deals N damage to up to one target creature`` — Pinnacle
    # Kill-Ship's ETB body. Mirrors the bare ``deals N damage to
    # target creature`` shape but with the ``up to one`` qualifier
    # flipping ``DealDamageToTargetCreature.optional`` to True so the
    # engine treats target selection as 0..1 (resolves as a no-op
    # when no creature is chosen). Tried AFTER the bare-target form
    # because the bare regex's ``$`` anchor on ``target creature``
    # wouldn't match the ``up to one target creature`` tail anyway,
    # but ordering more-specific shapes after the more-general bare
    # form is the convention here for damage primitives — the bare
    # form acts as the default and modifiers (``up to one``,
    # ``an opponent controls``, ``tapped``) layer on top via
    # dedicated regexes.
    m = _DEAL_DAMAGE_UP_TO_ONE_TARGET_CREATURE.match(phrase)
    if m is not None:
        return DealDamageToTargetCreature(
            amount=int(m.group(1)), optional=True,
        )
    # ``deals N damage to each opponent`` — mass-burn opponents (Ruinous
    # Rampage's burn mode). Tried after the targeted ``deals N damage to
    # any target`` / ``… target creature`` regexes because those bind
    # tighter (the ``each opponent`` tail is an exact literal that won't
    # collide with the targeted shapes).
    m = _DEAL_N_DAMAGE_EACH_OPPONENT.match(phrase)
    if m is not None:
        return DealDamageEachOpponent(amount=int(m.group(1)))
    # ``deals N damage to each creature`` — untargeted board-wipe burn
    # (Pyroclasm). Tried after the targeted ``deals N damage to target
    # creature`` regex above because the bare creature form's ``$``
    # anchor wouldn't match the ``each creature`` tail anyway, but
    # ordering more-specific before more-general is the convention here.
    # Maps to the existing ``DealNDamageToEachCreature`` engine class,
    # already exercised inside Lithobraking's reflexive optional-sac
    # sequence and Extinguisher Battleship's ETB sequence in EoE.
    m = _DEAL_N_DAMAGE_EACH_CREATURE.match(phrase)
    if m is not None:
        return DealNDamageToEachCreature(amount=int(m.group(1)))
    # ``deals N damage to target tapped creature`` — Reroute Systems
    # mode 1's tapped-only burn. Tried AFTER the targeted ``deals N
    # damage to target creature`` regex above so the bare creature
    # form wins on phrases without the ``tapped`` qualifier; this
    # more-specific shape only claims phrases that carry the
    # ``tapped`` tail. Maps to ``DealDamageToTargetTappedCreature``
    # whose ``legal_targets`` gates on ``obj.tapped`` and re-checks
    # at resolution.
    m = _DEAL_DAMAGE_TARGET_TAPPED_CREATURE.match(phrase)
    if m is not None:
        return DealDamageToTargetTappedCreature(amount=int(m.group(1)))
    if _DRAW_CARD_PER_MULTICOLORED_PERMANENT_LINE.match(
        phrase
    ) is not None:
        # ``draw a card for each multicolored permanent you control``
        # — Infinite Guideline Station's attack trigger. Tried BEFORE
        # the bare ``draw a card`` shape because the bare regex's
        # ``draw a card$`` tail wouldn't match the ``for each
        # multicolored permanent you control`` rider anyway, but
        # more-specific before more-general is the convention here.
        # See the regex comment for the deferred-generalization plan.
        return DrawCardPerMulticoloredPermanent()
    if _DRAW_A_CARD.match(phrase) is not None:
        return DrawCards(amount=1)
    m = _DRAW_N_CARDS.match(phrase)
    if m is not None:
        return DrawCards(amount=_parse_amount(m.group(1)))
    if _DRAW_THAT_MANY_CARDS.match(phrase) is not None:
        # ``draw that many cards`` — the resolving-combat-damage form
        # used by Fear of Failed Tests's combat-damage trigger body.
        # Same engine class Starwinder uses via the bespoke table; the
        # source's combat-damage amount is latched at trigger time and
        # read at resolve time, so the engine class needs no
        # parameters.
        return DrawCardsEqualToResolvingCombatDamage()
    m = _YOU_GAIN_N_LIFE.match(phrase)
    if m is not None:
        return GainLife(amount=int(m.group(1)))
    # ``put a +1/+1 counter on each of up to N target creatures`` —
    # multi-target counter spread (Weftblade Enhancer). Tried BEFORE
    # the bare-target regex below because the bare regex's ``target
    # creature$`` tail won't match the ``each of up to`` shape anyway,
    # but ordering more-specific before more-general is the convention
    # here. The amount tail accepts digits and spelled-out small
    # cardinals; both go through ``_parse_amount`` to land as the
    # ``max_n`` int on the engine class.
    m = _PUT_PLUS_ONE_COUNTER_ON_EACH_OF_UP_TO_N_TARGET_CREATURES.match(
        phrase
    )
    if m is not None:
        return PutPlusOneCounterOnEachOfUpToNTargetCreatures(
            max_n=_parse_amount(m.group(1)),
        )
    # ``put a +1/+1 counter on target creature you control`` — Rayblade
    # Trooper's ETB. Restricted variant: legal targets are creatures
    # the resolving controller controls. Tried BEFORE the bare ``target
    # creature`` regex (which has a ``$`` anchor and so wouldn't match
    # the ``you control`` tail anyway) so the more-specific shape wins
    # by convention.
    if _PUT_PLUS_ONE_COUNTER_ON_TARGET_CREATURE_YOU_CONTROL.match(
        phrase
    ) is not None:
        return PutPlusOneCounterOnCreatureYouControl(amount=1)
    # ``put a +1/+1 counter on target creature an opponent controls`` —
    # Sunset Saboteur's attack trigger. Mirror image of the ``you
    # control`` restriction above: legal targets are creatures NOT
    # controlled by the resolving controller. Tried BEFORE the bare
    # ``target creature`` regex (which has a ``$`` anchor that
    # wouldn't match this tail anyway) so the more-specific shape wins
    # by convention.
    if _PUT_PLUS_ONE_COUNTER_ON_TARGET_OPPONENT_CREATURE.match(
        phrase
    ) is not None:
        return PutPlusOneCounterOnOpponentsCreature(amount=1)
    m = _PUT_PLUS_ONE_COUNTER_ON_TARGET_CREATURE.match(phrase)
    if m is not None:
        # Group 1 captures the optional ``attacking `` qualifier (None
        # otherwise). When present, ``attacking_only=True`` narrows the
        # legal-target pool to currently-attacking creatures — same
        # engine flag Hospital Room's door-trigger body uses.
        return PutPlusOneCounterOnTargetCreature(
            amount=1, attacking_only=m.group(1) is not None,
        )
    if _PUT_PLUS_ONE_COUNTER_ON_THIS_CREATURE.match(phrase) is not None:
        return PutPlusOneCounterOnSelf(amount=1)
    if _PUT_LOYALTY_COUNTER_ON_THIS_CREATURE.match(phrase) is not None:
        return PutLoyaltyCounterOnSelf(amount=1)
    # ``put a +1/+1 counter on each creature you control other than
    # this creature`` — Dawnstrike Vanguard's end-step trigger body
    # (gated by the tapped-creatures wrapper at the trigger-line
    # level, but the inner effect is a plain mass-counter spread).
    # Tried BEFORE the bare-target counter regexes because the bare
    # ones all carry a ``$`` anchor on ``target creature`` etc. that
    # wouldn't match the ``each creature`` subject anyway, but
    # ordering more-specific before more-general is the convention
    # here. Maps to ``PutPlusOneCounterOnEachCreatureYouControl(
    # amount=1, exclude_source=True)`` — mirrors the static-registry
    # layout exactly (the ``other than this creature`` tail is what
    # sets the engine class's ``exclude_source`` flag). Tried BEFORE
    # the bare ``each creature you control`` regex so the more-specific
    # ``other than this creature`` shape wins on phrases that carry it.
    if _PUT_PLUS_ONE_COUNTER_ON_EACH_OTHER_CREATURE_YOU_CONTROL.match(
        phrase
    ) is not None:
        return PutPlusOneCounterOnEachCreatureYouControl(
            amount=1, exclude_source=True,
        )
    # ``put X +1/+1 counters on each creature you control, where X is
    # this creature's power`` — Ouroboroid's beginning-of-combat
    # trigger body. Tried BEFORE the bare
    # ``_PUT_PLUS_ONE_COUNTER_ON_EACH_CREATURE_YOU_CONTROL`` regex —
    # the bare regex's ``$`` anchor would never match the X-form tail
    # anyway, but ordering more-specific before more-general is the
    # convention here. Maps to
    # ``PutPlusOneCountersOnEachCreatureYouControlEqualToSourcePower()``
    # with no parameters — every aspect of the effect (X-source from
    # ``state.resolving_source.power``, per-creature spread on every
    # creature the controller controls) is encoded by the engine
    # class.
    if (
        _PUT_X_PLUS_ONE_COUNTERS_ON_EACH_CREATURE_YOU_CONTROL_WHERE_X_IS_SOURCE_POWER
    ).match(phrase) is not None:
        return PutPlusOneCountersOnEachCreatureYouControlEqualToSourcePower()
    # ``put a +1/+1 counter on each creature you control`` —
    # Atmospheric Greenhouse's ETB trigger body. Mass +1/+1-counter
    # spread on every creature the controller controls. Tried AFTER
    # the ``other than this creature`` shape above (more-specific
    # before more-general). Maps to
    # ``PutPlusOneCounterOnEachCreatureYouControl(amount=1)`` — the
    # default ``exclude_source=False`` matches the bespoke factory's
    # output for Atmospheric Greenhouse exactly.
    if _PUT_PLUS_ONE_COUNTER_ON_EACH_CREATURE_YOU_CONTROL.match(
        phrase
    ) is not None:
        return PutPlusOneCounterOnEachCreatureYouControl(amount=1)
    # ``destroy target creature or Spacecraft`` — Embrace Oblivion.
    # Tried BEFORE the bare ``destroy target creature`` regex so the
    # Spacecraft-tail variant wins. Both produce the same engine class
    # today (``DestroyTargetCreature``), just with the
    # ``or_spacecraft=True`` flag flipped — the bare regex below would
    # never match this surface anyway (its ``$`` anchor rejects the
    # ``or Spacecraft`` tail) but ordering more-specific before
    # more-general is the convention here.
    if _DESTROY_TARGET_CREATURE_OR_SPACECRAFT.match(phrase) is not None:
        return DestroyTargetCreature(or_spacecraft=True)
    # ``destroy target creature or enchantment`` — Withering Torment
    # (Duskmourn). Tried BEFORE the bare ``destroy target creature``
    # regex so the more-specific subtype-tail variant wins. The bare
    # regex's ``$`` anchor would never match this surface anyway, but
    # ordering more-specific before more-general is the convention here.
    if _DESTROY_TARGET_CREATURE_OR_ENCHANTMENT.match(phrase) is not None:
        return DestroyTargetCreature(or_enchantment=True)
    if _PUT_ITS_COUNTERS_ON_TARGET_CREATURE_YOU_CONTROL.match(
        phrase
    ) is not None:
        # ``put its counters on target creature you control`` — Dockworker
        # Drone's dies trigger. ``its`` refers to the source creature
        # (dying permanent); the engine class reads
        # ``state.resolving_source.plus_one_counters`` at resolve time
        # and transfers them to the chosen target.
        return MoveSelfCountersToTargetCreatureYouControl()
    # ``destroy up to one other target creature. If that creature
    # wasn't dealt damage this turn, its controller draws N cards``
    # — Faller's Faithful's ETB. Tried BEFORE the bare destroy regex
    # below so the longer surface wins. The bare regex's ``$`` anchor
    # would never match this surface anyway, but ordering more-
    # specific before more-general is the convention here.
    m = _DESTROY_UP_TO_ONE_OTHER_TARGET_CREATURE_DRAW_IF_UNDAMAGED.match(
        phrase
    )
    if m is not None:
        return DestroyTargetCreature(
            optional=True,
            exclude_self=True,
            draw_if_undamaged_amount=_parse_amount(m.group(1)),
        )
    if _DESTROY_TARGET_CREATURE.match(phrase) is not None:
        return DestroyTargetCreature()
    # ``another target attacking creature can't be blocked this turn``
    # — Clammy Prowler's attack-trigger body. Same engine class as the
    # bare unblockable-UEOT primitive but with the ``attacking_only``
    # and ``exclude_self`` flags flipped to narrow the legal-target
    # pool at selection time. Tried BEFORE the bare regex because the
    # bare regex's ``$`` anchor would never match this longer form
    # anyway, but ordering more-specific before more-general is the
    # convention here.
    if _ANOTHER_TARGET_ATTACKING_CREATURE_CANT_BE_BLOCKED_THIS_TURN.match(
        phrase
    ) is not None:
        return GrantUnblockableUEOT(attacking_only=True, exclude_self=True)
    # ``target creature with power N or less can't be blocked this
    # turn`` — Ragged Playmate's activated body. Same engine class
    # as the bare unblockable-UEOT primitive with the
    # ``power_at_most=N`` filter narrowing the legal-target pool.
    # Tried BEFORE the bare regex so the more-specific power-filter
    # form wins.
    m = _TARGET_CREATURE_WITH_POWER_AT_MOST_CANT_BE_BLOCKED_THIS_TURN.match(
        phrase
    )
    if m is not None:
        return GrantUnblockableUEOT(power_at_most=int(m.group(1)))
    # ``target creature can't be blocked this turn`` — bare unblockable
    # grant. Pure UEOT primitive on a creature target; sets
    # ``temp_unblockable=True`` at resolve time. Enter the Enigma is
    # the EoE/Duskmourn surface form that introduced this primitive.
    if _TARGET_CREATURE_CANT_BE_BLOCKED_THIS_TURN.match(phrase) is not None:
        return GrantUnblockableUEOT()
    # ``counter target artifact or enchantment spell`` — Annul. Tried
    # BEFORE the bare ``counter target spell`` regex so the more
    # specific filter-typed shape wins. Both shapes return the same
    # engine class, just with different ``filter_types`` frozensets.
    if _COUNTER_TARGET_ARTIFACT_OR_ENCHANTMENT_SPELL.match(
        phrase
    ) is not None:
        return CounterTargetSpell(
            filter_types=frozenset({"Artifact", "Enchantment"})
        )
    if _COUNTER_TARGET_SPELL.match(phrase) is not None:
        return CounterTargetSpell()
    # ``exile target creature or Spacecraft`` — Gravkill's printed
    # surface. Tried BEFORE the bare ``exile target creature`` regex so
    # the Spacecraft-tail variant wins. Both produce the same engine
    # class today (``ExileTargetCreature()``) — the static side fudges
    # the same way pending engine support for Spacecraft as a target
    # kind. When the engine grows that, this primitive should switch to
    # a Spacecraft-aware engine class in lockstep with the static.
    if _EXILE_TARGET_CREATURE_OR_SPACECRAFT.match(phrase) is not None:
        return ExileTargetCreature()
    # ``exile target artifact, enchantment, or creature with power 4 or
    # greater`` — Exorcise's full single-sentence body. Maps to the
    # parameterless ``ExorciseEffect`` engine class which evaluates the
    # three-way disjunctive filter at both target-pick and resolve time.
    # Tried BEFORE the bare ``exile target creature`` branch below
    # because the bare regex's ``$`` anchor on ``creature`` wouldn't
    # match this longer tail anyway, but ordering more-specific before
    # more-general is the convention here. Sole card with this shape
    # today; broader generalization deferred — see
    # ``_EXILE_TARGET_ARTIFACT_ENCHANTMENT_OR_BIG_CREATURE`` regex
    # comment.
    if _EXILE_TARGET_ARTIFACT_ENCHANTMENT_OR_BIG_CREATURE.match(
        phrase
    ) is not None:
        return ExorciseEffect()
    if _EXILE_TARGET_CREATURE.match(phrase) is not None:
        return ExileTargetCreature()
    if _EXILE_ALL_CREATURES_AND_SPACECRAFT.match(phrase) is not None:
        return ExileAllCreaturesAndSpacecraft()
    # ``destroy all tapped creatures`` / ``destroy all untapped
    # creatures`` — Split Up's modal mode lines. Two separate
    # branches because the matching ``DestroyAllCreatures`` engine-
    # class flag is a per-flag bool. The bare ``destroy all creatures``
    # untargeted Wrath shape isn't used by any spell in the current
    # pool — add it as a sibling branch if/when one ships.
    if _DESTROY_ALL_TAPPED_CREATURES.match(phrase) is not None:
        return DestroyAllCreatures(tapped_only=True)
    if _DESTROY_ALL_UNTAPPED_CREATURES.match(phrase) is not None:
        return DestroyAllCreatures(untapped_only=True)
    # ``destroy all creatures, then any number of target players each
    # mill half their library, rounded down`` — Singularity Rupture's
    # full single-sentence body. Maps to the parameterless
    # ``SingularityRuptureEffect`` engine class which folds the wrath
    # and the per-target floor-divide mill into one resolve step. Sole
    # card with this shape today; broader generalization deferred —
    # see ``_SINGULARITY_RUPTURE`` regex comment.
    if _SINGULARITY_RUPTURE.match(phrase) is not None:
        return SingularityRuptureEffect()
    # ``destroy all creatures with power N or greater`` — Restricted
    # Office's door trigger body. Parameterized over the power
    # threshold; maps to ``DestroyAllCreatures(min_power=N)`` whose
    # resolve walks ``effective_power`` so layer-7 pumps and +1/+1
    # counters are accounted for when filtering victims.
    m = _DESTROY_ALL_CREATURES_WITH_POWER_AT_LEAST.match(phrase)
    if m is not None:
        return DestroyAllCreatures(min_power=int(m.group(1)))
    m = _EXILE_ALL_ARTIFACTS_WITH_MANA_VALUE_AT_MOST.match(phrase)
    if m is not None:
        return ExileAllArtifactsWithManaValueAtMost(max_mv=int(m.group(1)))
    if _SEARCH_LIBRARY_FOR_ARTIFACT_CARD.match(phrase) is not None:
        # ``may_fail=True`` mirrors the static Scour-for-Scrap printing —
        # the controller may decline the fetch (and shuffle anyway). No
        # EoE oracle text spells out ``may_fail=False`` for an artifact
        # tutor today; if one ships, gate it on a different surface form.
        return SearchLibraryForCard(
            filter_name="artifact", may_fail=True,
        )
    if _SEARCH_LIBRARY_FOR_BASIC_LAND_CARD.match(phrase) is not None:
        # ``may_fail=True`` mirrors the existing artifact-tutor sibling —
        # the controller may decline the fetch (and shuffle anyway). No
        # DSK oracle text spells out a non-may basic-land tutor today,
        # but the engine field defaults the way the mandatory phrasing
        # would expect, so any future ``... reveal it, put it into your
        # hand, then shuffle`` printing landing here will keep parity.
        return SearchLibraryForCard(
            filter_name="basic_land", may_fail=True,
        )
    if _SEARCH_LIBRARY_FOR_LEGENDARY_CREATURE_CARD.match(
        phrase
    ) is not None:
        # ``search your library for a legendary creature card, reveal
        # it, put it into your hand, then shuffle`` — The Seriema's ETB
        # body. Maps to
        # ``SearchLibraryForCard(filter_name="legendary_creature",
        # may_fail=False)``. ``may_fail=False`` reflects the faithful
        # reading of the printed text (no "you may" rider). See the
        # regex comment for the deferred unification with the
        # ``may_fail=True`` artifact / basic-land siblings above.
        return SearchLibraryForCard(
            filter_name="legendary_creature", may_fail=False,
        )
    if _SEARCH_LIBRARY_FOR_BASIC_LAND_CARD_TAPPED_ONTO_BATTLEFIELD.match(
        phrase
    ) is not None:
        # ``search your library for a basic land card, put it onto the
        # battlefield tapped, then shuffle`` — Moldering Gym's door
        # trigger. Same payload the Lander token's activated ability
        # uses (``SearchLibraryForBasicLandTapped()``); the duplicate
        # string-keyed entry in ``land_builder``'s
        # ``_FETCH_LAND_BODIES`` is the fetch-land surface for cycling-
        # land bodies and stays narrow to that builder's
        # tap-and-sacrifice activation shape.
        return SearchLibraryForBasicLandTapped()
    if _SEARCH_LIBRARY_FOR_EQUIPMENT_CARD_ONTO_BATTLEFIELD.match(
        phrase
    ) is not None:
        # ``search your library for an Equipment card, put it onto the
        # battlefield, then shuffle`` — Honored Knight-Captain's
        # activated-ability body (sac-self → tutor for Equipment).
        # Maps to the existing ``SearchLibraryForEquipmentToBattlefield``
        # engine class. Distinct from the basic-land-tapped sibling on
        # the filter (Equipment vs basic land) AND the tap state
        # (untapped vs tapped); a future ``put it onto the battlefield
        # tapped`` Equipment tutor would land its own surface and an
        # ``Equipment``-keyed payload class with a ``tapped=True``
        # construction (no DSK printing today carries that shape).
        return SearchLibraryForEquipmentToBattlefield()
    m = _EACH_OPPONENT_LOSES_N_LIFE.match(phrase)
    if m is not None:
        return EachOpponentLosesLife(amount=int(m.group(1)))
    m = _EACH_PLAYER_DRAWS_N_CARDS.match(phrase)
    if m is not None:
        # ``each player draws a card`` collapses to amount=1 (the
        # ``a card`` alternative has no capture group, so group(1)
        # is None); digits and spelled-out small cardinals on the
        # plural arm go through ``_parse_amount``. Friendly Teddy's
        # dies trigger uses the ``a card`` form; the plural arm is
        # forward-looking grammar coverage.
        token = m.group(1)
        amount = 1 if token is None else _parse_amount(token)
        return EachPlayerDraws(amount=amount)
    if _EXILE_EACH_OPPONENTS_GRAVEYARD.match(phrase) is not None:
        # ``exile each opponent's graveyard`` — Dauntless Scrapbot's
        # first-half compound ETB. Untargeted graveyard wipe on every
        # opponent.
        return ExileEachOpponentsGraveyard()
    m = _EACH_OPPONENT_DISCARDS.match(phrase)
    if m is not None:
        # ``each opponent discards a card`` / ``each opponent discards
        # N cards`` — controller-untargeted opponent-wide discard.
        # ``a`` collapses to count=1; digits and spelled-out small
        # cardinals go through ``_parse_amount``.
        token = m.group(1)
        count = 1 if token == "a" else _parse_amount(token)
        return EachOpponentDiscards(count=count)
    if _EACH_OPPONENT_SACRIFICES_CREATURE.match(phrase) is not None:
        # ``each opponent sacrifices a nontoken creature of their
        # choice`` — Susurian Dirgecraft's ETB trigger body.
        # Untargeted opponent-wide sacrifice. Maps to
        # ``EachOpponentSacrificesCreature()`` whose engine class
        # walks each opponent and routes through their agent's
        # ``choose_sacrifice`` (skipping opponents with no legal
        # nontoken creature).
        return EachOpponentSacrificesCreature()
    m = _DEFENDING_PLAYER_MILLS_N_CARDS.match(phrase)
    if m is not None:
        # ``defending player mills N cards`` — Specimen Freighter's
        # attack-trigger body. Maps to ``EachOpponentMillsCards(amount=N)``
        # which is correct in 2-player and a documented approximation
        # in multiplayer (per the engine class docstring). Amount
        # accepts digits and spelled-out small cardinals.
        return EachOpponentMillsCards(amount=_parse_amount(m.group(1)))
    # ``create a Lander token. At the beginning of the end step on
    # your next turn, sacrifice that token`` — Kav Landseeker's ETB.
    # Tried BEFORE the bare token-creation dispatcher because the
    # bare ``create a Lander token`` regex would match the first half
    # of this phrase if the compound splitter ran differently, but
    # this single primitive captures the two-sentence body as one
    # effect (the second sentence references the token created in
    # the first, so they can't be split into independent effects).
    if _CREATE_LANDER_AND_SCHEDULE_SAC.match(phrase) is not None:
        return CreateLanderAndScheduleSacrifice()
    if _CREATE_TAPPED_ROBOT_TOKEN_PER_MULTICOLORED_LINE.match(
        phrase
    ) is not None:
        # ``create a tapped 2/2 colorless Robot artifact creature
        # token for each multicolored permanent you control`` —
        # Infinite Guideline Station's ETB. Tried BEFORE the generic
        # ``_parse_create_token_effect`` because that function's
        # ``_CREATE_CREATURE_TOKEN`` regex also matches the head of
        # this phrase (``create a tapped 2/2 colorless Robot artifact
        # creature token``) but stops at the ``$`` anchor before the
        # ``for each multicolored permanent you control`` tail —
        # without this earlier dispatch, the generic shape would
        # silently drop the per-multicolored gate. See the regex
        # comment for the deferred-generalization plan.
        return CreateTokenPerMulticoloredPermanent(
            token_printing=TOKEN_PRINTINGS["Robot"],
            tapped=True,
        )
    token_eff = _parse_create_token_effect(phrase)
    if token_eff is not None:
        return token_eff
    m = _SURVEIL_N.match(phrase)
    if m is not None:
        return Surveil(amount=int(m.group(1)))
    if _LOOK_TOP_TWO_SPLIT_HAND_GRAVEYARD.match(phrase) is not None:
        # ``look at the top two cards of your library. Put one into
        # your hand and the other into your graveyard`` — Codecracker
        # Hound's ETB. The two-sentence surface form is matched as a
        # single primitive; the agent's ``choose_surveil`` hook
        # decides which card goes to which pile at resolve time.
        return LookAtTopTwoSplitHandGraveyard()
    m = _LOOK_TOP_N_REVEAL_OR_COUNTER.match(phrase)
    if m is not None:
        # Parameterized peek-tutor-or-counter recognizer covering both
        # Living Phone's dies trigger body (creature/power-capped, no
        # rider) and Pulsar Squadron Ace's ETB body (Spacecraft, with
        # the ``If you didn't put a card into your hand this way, put
        # a +1/+1 counter on this creature`` rider). Captures:
        #   group(1) — N (digit or spelled-out cardinal)
        #   group(2) — filter token ("Spacecraft" or "creature")
        #   group(3) — optional max-power cap (only on the creature
        #              filter form)
        #   group("rider") — present iff the +1/+1 counter rider is
        #              printed
        # The engine class's ``max_power`` is ignored for non-creature
        # filters, so passing ``int(group(3) or 0)`` is safe for the
        # Spacecraft path.
        filter_token = m.group(2)
        filter_name = (
            "spacecraft" if filter_token == "Spacecraft" else "creature"
        )
        max_power = int(m.group(3)) if m.group(3) else 0
        return LookAtTopNRevealOrCounter(
            n=_parse_amount(m.group(1)),
            filter_name=filter_name,
            max_power=max_power,
            counter_on_miss=m.group("rider") is not None,
        )
    m = _MILL_N_CARDS.match(phrase)
    if m is not None:
        return SelfMill(amount=_parse_amount(m.group(1)))
    if _DESTROY_TARGET_ARTIFACT.match(phrase) is not None:
        return DestroyTargetArtifact()
    if _DESTROY_TARGET_ARTIFACT_ENCHANTMENT_OR_FLYING_CREATURE.match(
        phrase
    ) is not None:
        # ``destroy target artifact, enchantment, or creature with
        # flying`` — Shattered Wings. The engine class's
        # ``legal_targets`` filters to artifacts (any), enchantments
        # (any), or creatures that currently have flying, with a
        # re-check at resolution in case a between-cast-and-resolve
        # ability removes flying. Tried after the bare
        # ``destroy target artifact`` regex would have matched (which
        # this surface text doesn't due to the ``$`` anchor) so the
        # more-specific shape only claims phrases that carry the
        # ``, enchantment, or creature with flying`` tail.
        return DestroyTargetArtifactEnchantmentOrFlyingCreature()
    if _DESTROY_TARGET_LAND.match(phrase) is not None:
        return DestroyTargetLand()
    if _DESTROY_TARGET_NONCREATURE_PERMANENT.match(phrase) is not None:
        return DestroyTargetNoncreaturePermanent()
    if _DESTROY_TARGET_NONLAND_PERMANENT.match(phrase) is not None:
        # ``destroy target nonland permanent`` — Thaumaton Torpedo's
        # activated ability. Tried after ``destroy target noncreature
        # permanent`` because the two surface forms are disjoint
        # (different exclusion words), but ordering more-specific before
        # more-general is the convention here. Maps to
        # ``DestroyTargetNonlandPermanent`` whose ``legal_targets``
        # filters the battlefield to permanents that aren't lands.
        return DestroyTargetNonlandPermanent()
    # ``destroy target artifact or enchantment. If its mana value was 2
    # or less, create a Lander token`` — Seedship Impact's full two-
    # sentence body. Compound parsing fails on this phrase because the
    # rider's anaphoric ``If its mana value was 2 or less`` back-
    # references the destroyed permanent's mana value with no per-
    # clause primitive recognizing the back-reference, so
    # ``_parse_compound_effect`` returns None and dispatch reaches this
    # multi-sentence literal-line recognizer. Maps to the parameterless
    # ``SeedshipImpactEffect`` engine class whose resolve step destroys
    # the artifact-or-enchantment target and gates Lander creation on
    # the destroyed permanent's printed mana value (≤ 2). See the
    # regex's docstring for the deferred-generalization plan when a
    # second card with this shape ships.
    if _SEEDSHIP_IMPACT.match(phrase) is not None:
        return SeedshipImpactEffect()
    if _DESTROY_TARGET_ARTIFACT_OR_TAPPED_CREATURE.match(phrase) is not None:
        # ``destroy target artifact or tapped creature`` — Radiant
        # Strike's destroy clause. The engine class's
        # ``legal_targets`` filters to artifacts (any) plus creatures
        # currently tapped, and re-checks the predicate at resolution
        # in case a between-cast-and-resolve untap removes the tapped
        # creature from the legal pool. Tried after the bare
        # ``destroy target artifact`` regex would have matched (which
        # this surface text doesn't) so the more-specific shape only
        # claims phrases that actually carry the ``or tapped
        # creature`` tail.
        return DestroyTargetArtifactOrTappedCreature()
    # ``tap another target creature`` — Starport Security's activated
    # ability. Tried BEFORE the ``tap target artifact or creature``
    # regex below because the bare regex's ``$`` anchor wouldn't match
    # the ``another`` qualifier anyway, but ordering more-specific
    # before more-general is the convention here.
    if _TAP_ANOTHER_TARGET_CREATURE.match(phrase) is not None:
        return TapAnotherTargetCreature()
    if _TAP_TARGET_ARTIFACT_OR_CREATURE.match(phrase) is not None:
        # ``tap target artifact or creature`` — Mental Modulation's
        # middle clause and any future card with the same surface.
        # No-op if the chosen target is already tapped at resolution.
        return TapTargetArtifactOrCreature()
    if _TAP_TARGET_OPPONENT_PERMANENT_AND_STUN_IT.match(phrase) is not None:
        # ``tap target artifact or creature an opponent controls and
        # put a stun counter on it`` — Mechanozoa's ETB. Both clauses
        # apply to the same chosen target — the ``and`` is an
        # intra-effect conjunction, not a compound joiner.
        return TapTargetOpponentPermanentAndStunIt()
    m = _TAP_UP_TO_ONE_TARGET_CREATURE_AND_STUN.match(phrase)
    if m is not None:
        # ``tap up to one target creature and put N stun counters on
        # it`` — Meat Locker's door trigger. Both clauses apply to
        # the same chosen target (the ``and`` is intra-effect, not a
        # compound joiner). The captured count tail collapses ``a``
        # to 1 and routes digits / spelled-out small cardinals
        # through ``_parse_amount`` — same shape as
        # ``_DISCARD_N_CARDS`` and the other count-tail primitives.
        # ``up to one`` resolves as a no-op when no target is chosen
        # (engine-side ``target_count_range = (0, 1)``).
        token = m.group(1)
        amount = 1 if token == "a" else _parse_amount(token)
        return TapAndStunUpToOneTargetCreature(stun_amount=amount)
    if _PUT_STUN_COUNTER_UP_TO_ONE_TAPPED.match(phrase) is not None:
        # ``put a stun counter on up to one target tapped creature`` —
        # Cryogen Relic's activated payload. ``up to one`` resolves as
        # a no-op fizzle when no legal target is chosen; the engine
        # class's ``legal_targets`` filters to currently-tapped
        # creatures.
        return PutStunCounterOnTappedCreature()
    if _UNTAP_ANOTHER_TARGET_PERMANENT.match(phrase) is not None:
        # ``untap another target permanent`` — Nanoform Sentinel's
        # becomes-tapped trigger body. The ``another`` qualifier
        # excludes the source from the legal target pool — encoded
        # by the engine class's reading of ``state.resolving_source``
        # at resolve time, not by a parser-level flag. Maps to
        # ``UntapTargetOtherPermanent()`` with no parameters.
        return UntapTargetOtherPermanent()
    if _OWNER_PUTS_TARGET_ARTIFACT_OR_CREATURE_ON_TOP_OR_BOTTOM.match(
        phrase
    ) is not None:
        # ``target artifact or creature's owner puts it on their
        # choice of the top or bottom of their library`` — Lost in
        # Space's bounce-to-library clause. The chosen target's
        # owner decides top vs bottom at resolution (not the
        # spell's controller). Maps to the engine class with no
        # parameters — the surface form is a literal match.
        return OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary()
    if _OWNER_PUTS_TARGET_NONLAND_PERMANENT_ON_TOP_OR_BOTTOM.match(
        phrase
    ) is not None:
        # ``target nonland permanent's owner puts it on their choice
        # of the top or bottom of their library`` — Vanish from
        # Sight's bounce-to-library clause. Same engine class as the
        # artifact-or-creature shape with the ``nonland_permanent``
        # flag flipped to widen the legal-target pool.
        return OwnerPutsTargetArtifactOrCreatureOnTopOrBottomOfLibrary(
            nonland_permanent=True,
        )
    # ``add {<mana>}. Spend this mana only to <restriction>`` —
    # restricted-mana ability (Steelswarm Operator). Tried BEFORE the
    # bare ``add {<color>}`` regex so the qualifier wins; the bare
    # regex's ``$`` anchor wouldn't match the multi-sentence shape
    # anyway, but ordering more-specific before more-general is the
    # convention here. The mana segment is parsed by inspecting the
    # captured brace expression — the regex enforces that it's exactly
    # one or two same-color brace pairs but doesn't decompose them
    # into engine coins, so we extract the colors directly.
    m = _ADD_MANA_WITH_SPEND_RESTRICTION.match(phrase)
    if m is not None:
        mana_str = m.group(1)
        restriction = _SPEND_RESTRICTION_BY_PHRASE[m.group(2)]
        # The mana segment is one or two ``{<color>}`` brace pairs;
        # ``parse_mana_cost`` would unpack them but for restricted-mana
        # abilities the engine wants a tuple of single-coin entries
        # (one ``(color, 1)`` per coin minted), so we walk the brace
        # pairs directly.
        colors = re.findall(r"\{([WUBRG])\}", mana_str)
        return AddManaToPool(
            produces=tuple((c, 1) for c in colors),
            spend_restriction=restriction,
        )
    m = _ADD_TWO_SAME_COLOR_MANA.match(phrase)
    if m is not None:
        # Two same-color coin shape (``add {U}{U}``) — only same-color
        # pairs reach this regex (the inner brace pattern captures both
        # letters and we reject mismatched pairs to keep the engine's
        # produced tuple simple). A mismatched pair like
        # ``add {U}{R}`` falls through to the bare single-color regex
        # and fails its anchor, so the card stays unclaimed (no EoE
        # printing today uses dual-color two-coin add anyway).
        if m.group(1) != m.group(2):
            return None
        c = m.group(1)
        return AddManaToPool(produces=((c, 1), (c, 1)))
    m = _ADD_SINGLE_COLOR_MANA.match(phrase)
    if m is not None:
        return AddManaToPool(produces=((m.group(1), 1),))
    if _ADD_ONE_MANA_OF_ANY_COLOR.match(phrase) is not None:
        # ``add one mana of any color`` — at-resolution choose-color
        # mana ability. Mints a single ``("any", 1)`` coin; the engine
        # prompts the controller to name the color at resolution. Same
        # production tuple ``land_builder`` builds for Mana Confluence.
        return AddManaToPool(produces=(("any", 1),))
    m = _TARGET_OPPONENT_LOSES_N_LIFE.match(phrase)
    if m is not None:
        return TargetOpponentLosesLife(amount=int(m.group(1)))
    if _RETURN_UP_TO_ONE_OTHER_PERMANENT_YOU_CONTROL_TO_OWNERS_HAND.match(
        phrase
    ) is not None:
        # ``return up to one other target permanent you control to its
        # owner's hand`` — Exosuit Savior's ETB. ``up to one`` →
        # ``optional=True`` (resolves as a no-op if no target chosen);
        # ``other`` → ``exclude_self=True`` (filters the trigger source
        # via ``state.resolving_source``).
        return ReturnTargetPermanentYouControlToOwnersHand(
            optional=True, exclude_self=True,
        )
    if _RETURN_UP_TO_ONE_TARGET_CREATURE_TO_OWNERS_HAND.match(
        phrase
    ) is not None:
        # ``return up to one target creature to its owner's hand`` —
        # Bottomless Pool's door trigger. ``up to one`` →
        # ``optional=True`` (resolves as a no-op fizzle if no target
        # chosen); no controller restriction (any creature, friend or
        # foe, is a legal target).
        return ReturnTargetCreatureToOwnersHand(optional=True)
    m = _RETURN_UP_TO_N_NON_SPACECRAFT_CREATURES_TO_OWNERS_HANDS.match(
        phrase
    )
    if m is not None:
        # ``return up to N target non-Spacecraft creatures to their
        # owners' hands`` — Specimen Freighter's ETB. The
        # ``non-Spacecraft`` qualifier flips
        # ``ReturnUpToNTargetCreaturesToOwnersHands.exclude_spacecraft``
        # to True (legal pool excludes stationed and unstationed
        # Spacecraft alike); ``max_n=N`` is the captured bound. Amount
        # accepts digits and spelled-out small cardinals.
        return ReturnUpToNTargetCreaturesToOwnersHands(
            max_n=_parse_amount(m.group(1)),
            exclude_spacecraft=True,
        )
    m = _RETURN_TARGET_CREATURE_GRAVEYARD_TO_HAND_WITH_MV.match(phrase)
    if m is not None:
        # ``return target creature card with mana value N or less from
        # your graveyard to your hand`` — Savior of the Small's
        # Survival reward. The MV cap narrows the legal-target pool to
        # graveyard creatures whose printed mana value clears the
        # threshold; ``max_mv=0`` (unset) keeps the original unbounded
        # behaviour for Interceptor Mechan and the bare regex below.
        return ReturnCreatureFromYourGraveyardToHand(max_mv=int(m.group(1)))
    if _RETURN_CREATURE_OR_SPACECRAFT_GRAVEYARD_TO_HAND.match(
        phrase
    ) is not None:
        # ``return a creature or Spacecraft card from your graveyard
        # to your hand`` — Fell Gravship's ETB body second sentence.
        # Tried before the bare ``return target creature card`` regex
        # below for the more-specific-first convention; the bare
        # regex's ``$`` anchor on ``hand`` would never match the
        # ``or Spacecraft`` rider anyway. ``or_spacecraft=True``
        # widens the engine class's legal-target pool to include
        # Spacecraft cards in the graveyard alongside creature cards,
        # mirroring the ``or_spacecraft`` flag on
        # ``DestroyTargetCreature`` (Embrace Oblivion).
        return ReturnCreatureFromYourGraveyardToHand(or_spacecraft=True)
    if _RETURN_TARGET_CREATURE_GRAVEYARD_TO_HAND.match(phrase) is not None:
        return ReturnCreatureFromYourGraveyardToHand()
    # Xu-Ifit, Osteoharmonist's reanimate-as-Skeleton multi-sentence
    # surface drops the ``has no abilities`` rider per the static-side
    # simplification — same engine class as the bare reanimate regex
    # but with its own pattern so the simplification stays local and
    # explicit.
    if _RETURN_TARGET_CREATURE_GRAVEYARD_TO_BATTLEFIELD_AS_SKELETON.match(
        phrase
    ) is not None:
        return ReturnCreatureFromYourGraveyardToBattlefield()
    # ``return all creature cards from your graveyard to the
    # battlefield`` — Awakening Hall's door trigger (Funeral Room //
    # Awakening Hall, the only EoE/DSK printing of this exact surface
    # today). Maps to ``ReturnAllCreaturesFromYourGraveyardToBattlefield()``.
    # Literal-line recognizer with a deferred-generalization note on
    # the regex above — fold this and any future ``...to your hand``
    # mass-recursion sibling into one parameterized regex when the
    # engine grows the matching sibling primitive.
    if _RETURN_ALL_CREATURE_CARDS_GRAVEYARD_TO_BATTLEFIELD.match(
        phrase
    ) is not None:
        return ReturnAllCreaturesFromYourGraveyardToBattlefield()
    m = _RETURN_TARGET_CREATURE_GRAVEYARD_TO_BATTLEFIELD_WITH_MV.match(phrase)
    if m is not None:
        # ``return target creature card with mana value N or less from
        # your graveyard to the battlefield`` — Surgical Suite's door
        # trigger. The MV cap narrows the legal-target pool to graveyard
        # creatures whose printed mana value clears the threshold;
        # ``max_mv=None`` (the default on the bare regex below) keeps
        # the unbounded behaviour for Xu-Ifit and any other unrestricted
        # reanimator. Mirrors the to-hand sibling's max_mv rider.
        return ReturnCreatureFromYourGraveyardToBattlefield(
            max_mv=int(m.group(1)),
        )
    if _RETURN_TARGET_CREATURE_OR_ENCHANTMENT_GRAVEYARD_TO_BATTLEFIELD.match(
        phrase
    ) is not None:
        # ``return target creature or enchantment card from your
        # graveyard to the battlefield`` — Rescue Skiff's ETB body.
        # Tried BEFORE the bare ``return target creature card`` regex
        # below for the more-specific-first convention; the bare regex's
        # ``$`` anchor on ``battlefield`` would not match the
        # ``or enchantment`` rider anyway. ``or_enchantment=True``
        # widens the engine class's legal-target pool to include
        # enchantment cards in the graveyard alongside creature cards,
        # mirroring the ``or_spacecraft`` flag on
        # ``ReturnCreatureFromYourGraveyardToHand`` (Fell Gravship).
        return ReturnCreatureFromYourGraveyardToBattlefield(
            or_enchantment=True,
        )
    if _RETURN_TARGET_CREATURE_GRAVEYARD_TO_BATTLEFIELD.match(
        phrase
    ) is not None:
        return ReturnCreatureFromYourGraveyardToBattlefield()
    if _RETURN_TARGET_ARTIFACT_GRAVEYARD_TO_HAND.match(phrase) is not None:
        return ReturnTargetArtifactCardFromYourGraveyardToHand()
    # DSK Enduring cycle dies-trigger body (Innocence / Curiosity /
    # Tenacity / Courage / Vitality). The phrase ``return it to the
    # battlefield under its owner's control. It's an enchantment``
    # appears verbatim as the effect tail of every Enduring creature's
    # intervening-if dies trigger; the parenthesized
    # ``(It's not a creature.)`` reminder is stripped before the
    # effect body reaches this dispatcher. Maps to the parameterless
    # ``EnduringDiesReturnAsEnchantmentEffect`` engine class.
    if _ENDURING_DIES_RETURN_AS_ENCHANTMENT.match(phrase) is not None:
        return EnduringDiesReturnAsEnchantmentEffect()
    # ``put target card from a graveyard on the bottom of its owner's
    # library`` — Chrome Companion's tap-mana activated ability. Untyped
    # grave-target across all players; resolution moves the chosen card
    # to the bottom of its owner's library. Maps to the existing
    # ``PutCardFromGraveyardOnBottomOfLibrary`` engine class — same
    # primitive the static registry uses for Chrome Companion today, so
    # parser/static parity holds out of the box.
    if _PUT_CARD_FROM_GRAVEYARD_ON_BOTTOM_OF_LIBRARY.match(
        phrase
    ) is not None:
        return PutCardFromGraveyardOnBottomOfLibrary()
    # ``exile target card from a graveyard`` — Ghost Vacuum's bare
    # tap activation. Sibling to the bottom-of-library shape above;
    # same target-kind pool but the chosen card moves to its owner's
    # exile zone, with the resolving source's id stamped on
    # ``exiled_by_source`` so a sibling recall ability can find every
    # card the same source exiled later (Ghost Vacuum's
    # ``{6}, {T}, Sacrifice this artifact`` recall is the only EoE/DSK
    # consumer of that linkage today).
    if _EXILE_CARD_FROM_GRAVEYARD.match(phrase) is not None:
        return ExileTargetCardFromGraveyard()
    # Pump + grant-keywords compound is tried BEFORE the bare-pump
    # regex so a phrase like ``+3/+0 and gains first strike and reach
    # until end of turn`` produces the keyword-bearing
    # ``PumpTargetCreatureUntilEOT``. The bare-pump regex below would
    # never match this surface (it requires the stats to be directly
    # followed by ``until end of turn``), but ordering more-specific
    # before more-general is the convention in this dispatcher.
    m = _PUMP_AND_GAIN_KEYWORDS_UEOT.match(phrase)
    if m is not None:
        # Group layout: 1=``another `` prefix, 2=``attacking `` prefix,
        # 3=``you control ``, 4=power, 5=toughness, 6=granted-keywords
        # body. The ``another`` qualifier excludes the trigger source
        # from the legal target pool via ``exclude_self=True``;
        # ``you control`` restricts targets to the controller's own
        # creatures via ``you_control_only=True``; ``attacking``
        # restricts targets to creatures currently flagged
        # ``attacking=True`` via ``attacking_only=True``. All three
        # flags compose so the ``another target creature you control``
        # surface lands with the first two flipped, and a hypothetical
        # ``another target attacking creature you control`` would land
        # with all three flipped.
        keywords = _parse_keyword_grant_body(m.group(6))
        if keywords is None:
            return None
        return PumpTargetCreatureUntilEOT(
            power_bonus=int(m.group(4)),
            toughness_bonus=int(m.group(5)),
            granted_keywords=keywords,
            you_control_only=m.group(3) is not None,
            exclude_self=m.group(1) is not None,
            attacking_only=m.group(2) is not None,
        )
    # ``target creature an opponent controls gets ±N/±M until end of
    # turn`` — opponent-controlled-only debuff/pump (Sinister
    # Cryologist's ETB). Tried BEFORE the bare-pump regex because that
    # bare regex's tail (``until end of turn$``) would never match
    # phrases with the ``an opponent controls`` qualifier in the
    # subject anyway, but ordering more-specific before more-general
    # is the convention here.
    m = _PUMP_TARGET_OPPONENT_CREATURE_UEOT.match(phrase)
    if m is not None:
        return PumpTargetCreatureUntilEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
            opponents_only=True,
        )
    # ``target creature you control gets ±N/±M until end of turn`` —
    # you-control-only single-target pump (Zookeeper Mechan's ``{6}{R}``
    # sorcery-speed activated ability). Tried BEFORE the bare
    # ``target creature gets … UEOT`` regex so the ``you control``
    # qualifier wins; the bare regex's ``$`` anchor wouldn't match the
    # qualifier anyway, but ordering more-specific before more-general
    # is the convention here.
    m = _PUMP_TARGET_CREATURE_YOU_CONTROL_UEOT.match(phrase)
    if m is not None:
        return PumpTargetCreatureUntilEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
            you_control_only=True,
        )
    # ``up to one target creature gets ±N/±M until end of turn`` —
    # optional pump/debuff (Dubious Delicacy's ETB ``-3/-3``). The
    # ``up to one`` qualifier flips the engine class's
    # ``target_count_range`` to ``(0, 1)`` and ``requires_target`` to
    # False, so a no-target resolution is a clean no-op rather than a
    # fizzle. Tried BEFORE the bare-pump regex because that bare regex's
    # ``$`` anchor wouldn't match the ``up to one`` prefix anyway, but
    # ordering more-specific before more-general is the convention here.
    m = _PUMP_UP_TO_ONE_TARGET_CREATURE_UEOT.match(phrase)
    if m is not None:
        return PumpTargetCreatureUntilEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
            optional=True,
        )
    if _DOUBLE_TARGET_CREATURE_YOU_CONTROL_POWER_UEOT.match(
        phrase
    ) is not None:
        # ``double the power of target creature you control until end
        # of turn`` — Mightform Harmonizer's landfall trigger. Distinct
        # from the bare ``gets +N/+0`` pump shapes above: the bonus is
        # computed at resolve time from the target's current power
        # rather than a fixed constant. Maps to the engine class with
        # no parameters — the surface form is a literal match.
        return DoubleTargetCreaturesPowerUEOT()
    m = _PUMP_TARGET_CREATURE_UEOT.match(phrase)
    if m is not None:
        return PumpTargetCreatureUntilEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
        )
    # ``[other ]creatures you control get ±N/±M[ and gain <kw>...]
    # until end of turn`` — anthem-style mass pump on every creature
    # the controller controls, with optional source-self exclusion
    # (``other``) and optional grant-keywords rider. The bare form is
    # Dual-Sun Adepts' ``{5}`` activated ability; the
    # exclude-source-and-grant-keywords form is Syr Vondam, the
    # Lucent's enters/attacks trigger ("other creatures you control
    # get +1/+0 and gain deathtouch until end of turn"). An
    # unrecognized granted-keyword body rejects the whole phrase
    # (returns ``None`` so the caller can fall through) — mirrors the
    # ``_PUMP_AND_GAIN_KEYWORDS_UEOT`` precedent: a missing grant is
    # never silently dropped.
    m = _PUMP_YOUR_CREATURES_UEOT.match(phrase)
    if m is not None:
        granted_body = m.group(4)
        granted_keywords: frozenset[str] = frozenset()
        if granted_body is not None:
            parsed_keywords = _parse_keyword_grant_body(granted_body)
            if parsed_keywords is None:
                return None
            granted_keywords = parsed_keywords
        return PumpCreaturesYouControlUntilEOT(
            power_bonus=int(m.group(2)),
            toughness_bonus=int(m.group(3)),
            exclude_source=m.group(1) is not None,
            granted_keywords=granted_keywords,
        )
    # ``creatures your opponents control get ±N/±M until end of turn`` /
    # ``... until your next turn`` — mass debuff/pump on every creature
    # an opponent controls (Mouth of the Storm's ETB uses the
    # ``until your next turn`` tail). The ``until_caster_next_turn``
    # flag flips when the tail matches the longer-duration form;
    # otherwise the bonus clears at the casting turn's cleanup.
    m = _PUMP_OPPONENT_CREATURES_UEOT.match(phrase)
    if m is not None:
        return PumpCreaturesOpponentsControlUntilEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
            until_caster_next_turn=(m.group(3) == "your next turn"),
        )
    # ``this creature gets +N/+M and gains <kw>[ and <kw2>...] until
    # end of turn`` — self-pump combined with a self keyword grant
    # (Dashing Bloodsucker's Eerie body: ``this creature gets +2/+0
    # and gains lifelink until end of turn``). Tried BEFORE the bare
    # ``_PUMP_SELF_UEOT`` regex so the more-specific compound shape
    # wins; the bare regex's ``$`` anchor would reject the
    # ``and gains …`` tail anyway, but ordering more-specific before
    # more-general is the convention here. The keyword body is parsed
    # via ``_parse_keyword_grant_body`` and validated against the
    # keyword-line whitelist; rejection returns ``None`` so an unknown
    # keyword keeps the wrapping card unclaimed.
    m = _PUMP_AND_GAIN_KEYWORDS_SELF_UEOT.match(phrase)
    if m is not None:
        keywords = _parse_keyword_grant_body(m.group(3))
        if keywords is None:
            return None
        return PumpSelfUntilEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
            granted_keywords=keywords,
        )
    # ``this creature gets +N/+M until end of turn`` — source pumps
    # itself UEOT. Tried AFTER the targeted ``target creature gets …
    # UEOT`` regex above because the ``this creature`` subject is a
    # different (source-self) primitive from a targeted bonus. Both
    # signs accepted, mirroring the symmetry of the targeted variant.
    m = _PUMP_SELF_UEOT.match(phrase)
    if m is not None:
        return PumpSelfUntilEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
        )
    # ``this creature gains <kw>[ and <kw2>...] until end of turn`` —
    # pure self-grant (no P/T pump). Cult Healer's Eerie body
    # (``this creature gains lifelink until end of turn``). Maps to
    # ``PumpSelfUntilEOT(power_bonus=0, toughness_bonus=0,
    # granted_keywords=...)`` — same engine class as the self-pump-
    # and-gain shape just with zero bonuses. Tried AFTER both self-
    # pump shapes so a phrase carrying a P/T pump prefix wins, and
    # AFTER the targeted bare-grant regexes (which all carry their
    # own ``target creature`` / ``another target creature you
    # control`` subject — the bare ``this creature`` subject doesn't
    # collide). Validated via ``_parse_keyword_grant_body``.
    m = _GAIN_KEYWORDS_SELF_UEOT.match(phrase)
    if m is not None:
        keywords = _parse_keyword_grant_body(m.group(1))
        if keywords is None:
            return None
        return PumpSelfUntilEOT(
            power_bonus=0,
            toughness_bonus=0,
            granted_keywords=keywords,
        )
    # ``another target creature you control gets +X/+X until end of
    # turn, where X is this creature's power`` — Hemosymbic Mite's
    # becomes-tapped trigger body. Variable-X bonus where X is fixed
    # to the source's effective power at resolve time; the engine
    # class ``PumpTargetOtherCreatureByXOverXWhereXIsSourcePower``
    # reads ``state.resolving_source.power`` and applies the bonus
    # symmetrically (P/T both gain the same X) to the chosen target,
    # which must be a creature the source's controller controls and
    # not be the source itself. Maps to the engine class with no
    # parameters — every aspect of the effect is encoded by the
    # class. Future "where X is …" forms with a different scaling
    # source (e.g. number of artifacts you control) would land their
    # own primitive rather than parameterizing this one.
    if _PUMP_OTHER_TARGET_CREATURE_BY_X_OVER_X_SOURCE_POWER.match(
        phrase
    ) is not None:
        return PumpTargetOtherCreatureByXOverXWhereXIsSourcePower()
    # ``another target creature you control gains <kw>[ and <kw2>...]
    # until end of turn`` — pure keyword grant to another creature you
    # control (Blooming Stinger's ETB grants deathtouch). No P/T pump
    # — only the keyword grant fires. Validated against the keyword-
    # line whitelist via ``_parse_keyword_grant_body``; rejection
    # returns ``None`` so an unknown keyword keeps the wrapping card
    # unclaimed rather than silently dropping the grant.
    m = _GRANT_KEYWORDS_TO_ANOTHER_TARGET_CREATURE_YOU_CONTROL_UEOT.match(
        phrase
    )
    if m is not None:
        keywords = _parse_keyword_grant_body(m.group(1))
        if keywords is None:
            return None
        return PumpTargetCreatureUntilEOT(
            power_bonus=0,
            toughness_bonus=0,
            granted_keywords=keywords,
            you_control_only=True,
            exclude_self=True,
        )
    # ``target creature gains <kw>[ and <kw2>...] until end of turn`` —
    # pure keyword grant on a bare creature target (Horrid Vigor's
    # ``Target creature gains deathtouch and indestructible until end
    # of turn.`` in DSK). No control restriction, no P/T pump — only
    # the keyword grant fires. Tried AFTER the ``another target
    # creature you control gains …`` shape above so the more-specific
    # qualifier wins; this bare-target shape only claims phrases that
    # don't carry the ``you control`` / ``another`` / ``an opponent
    # controls`` qualifiers. The keyword body is parsed via
    # ``_parse_keyword_grant_body`` and validated against the
    # keyword-line whitelist; rejection returns ``None`` so an
    # unknown keyword keeps the wrapping spell unclaimed.
    #
    # Group 1 captures the optional ``attacking `` qualifier (None
    # otherwise); when present, ``attacking_only=True`` flips on the
    # legal-target filter (same flag Hospital Room's counter trigger
    # uses) so Tunnel of Hate's ``target attacking creature gains
    # double strike until end of turn`` lands here without a separate
    # literal-line regex.
    m = _GRANT_KEYWORDS_TO_TARGET_CREATURE_UEOT.match(phrase)
    if m is not None:
        keywords = _parse_keyword_grant_body(m.group(2))
        if keywords is None:
            return None
        return PumpTargetCreatureUntilEOT(
            power_bonus=0,
            toughness_bonus=0,
            granted_keywords=keywords,
            attacking_only=m.group(1) is not None,
        )
    # ``target artifact or creature gains <kw>[ and <kw2>...] until
    # end of turn`` — pure keyword grant on a permanent target whose
    # legal pool spans both artifacts and creatures (Reroute Systems
    # mode 0 grants indestructible). Maps to
    # ``PumpTargetArtifactOrCreatureUntilEOT(granted_keywords=...)``.
    # Validated against the keyword-line whitelist; rejection returns
    # ``None`` so an unknown keyword keeps the wrapping spell
    # unclaimed rather than silently dropping the grant.
    m = _PUMP_TARGET_ARTIFACT_OR_CREATURE_GAINS_KEYWORDS_UEOT.match(phrase)
    if m is not None:
        keywords = _parse_keyword_grant_body(m.group(1))
        if keywords is None:
            return None
        return PumpTargetArtifactOrCreatureUntilEOT(
            granted_keywords=keywords,
        )
    m = _PUT_CHARGE_COUNTERS_ON_TARGET_SPACECRAFT_OR_PLANET_YOU_CONTROL.match(
        phrase
    )
    if m is not None:
        # ``put N charge counters on target Spacecraft or Planet you
        # control`` — Drill Too Deep mode 0 and any future card with
        # the same surface. Amount is a digit or one of the spelled-out
        # small cardinals; both go through ``_parse_amount`` to land
        # as an int on the engine class's ``amount`` field.
        return PutChargeCountersOnTargetSpacecraftOrPlanetYouControl(
            amount=_parse_amount(m.group(1)),
        )
    if _MILL_TARGET_PLAYER_EQUAL_TO_YOUR_LIFE.match(phrase) is not None:
        # ``target player mills cards equal to your life total`` —
        # Space-Time Anomaly. Maps to the engine class's
        # ``use_controller_life=True`` flag, which substitutes the
        # controller's current life total for the bare amount at
        # resolve time. Tried BEFORE the bare-amount mill regex so
        # the more-specific shape wins; the bare-amount form would
        # never match this surface anyway (its tail is a digit) but
        # ordering more-specific before more-general is the
        # convention here.
        return MillTargetPlayer(use_controller_life=True)
    m = _MILL_TARGET_PLAYER_N_CARDS.match(phrase)
    if m is not None:
        # ``target player mills N cards`` — bare-amount targeted
        # mill. Forward-looking surface form (no EoE card prints
        # this exact shape today; Space-Time Anomaly uses the
        # variable form above), but the regex accepts digits and
        # spelled-out small cardinals so a future printing parses
        # without further grammar work.
        return MillTargetPlayer(amount=_parse_amount(m.group(1)))
    m = _LOSE_N_LIFE.match(phrase)
    if m is not None:
        return LoseLife(amount=int(m.group(1)))
    # ``manifest dread, then put a +1/+1 counter on that creature`` /
    # ``manifest dread, then put N +1/+1 counters on that creature`` —
    # Slimy Aquarium / Weight Room door triggers. Coupled primitive in
    # the engine (``ManifestDreadThenPutPlusOneCountersOnIt``) where
    # ``that creature`` is anaphoric to the just-manifested face-down
    # 2/2 (NOT a chosen target). Tried BEFORE the bare ``_MANIFEST_DREAD``
    # regex below so the more-specific compound-with-counter shape
    # wins; the bare regex's ``$`` anchor on the count tail wouldn't
    # match this surface anyway, but ordering more-specific before
    # more-general keeps the dispatch narrative readable.
    m = _MANIFEST_DREAD_THEN_PUT_PLUS_ONE_COUNTERS_ON_THAT_CREATURE.match(
        phrase
    )
    if m is not None:
        # Group 1 is the plural counter amount (digit or spelled-out
        # cardinal); ``None`` means the singular ``a +1/+1 counter``
        # arm matched, which collapses to amount=1.
        amount_token = m.group(1)
        counter_amount = (
            1 if amount_token is None else _parse_amount(amount_token)
        )
        return ManifestDreadThenPutPlusOneCountersOnIt(
            counter_amount=counter_amount,
        )
    # ``manifest dread`` (count=1) or ``manifest dread <amount> times`` /
    # ``manifest dread twice`` / ``manifest dread three times`` /
    # ``manifest dread X times``. Maps to
    # ``ManifestDread(count=N)`` — N is 1 for the bare form, 2/3/N for
    # the spelled-out / digit forms, or ``None`` for the X-times form
    # (count is read from ``state.resolving_x_value`` at resolve time).
    # Tried BEFORE the generic compound-effect splitter would carve a
    # phrase like ``manifest dread X times, then put X +1/+1 counters on
    # each of those creatures`` (no card uses that as a top-level
    # primitive today; Valgavoth's Onslaught has it but the current
    # parser doesn't claim it because the second clause's ``those
    # creatures`` carry-over isn't yet supported).
    m = _MANIFEST_DREAD.match(phrase)
    if m is not None:
        twice_token = m.group(1)
        amount_token = m.group(2)
        if twice_token is not None:
            count: int | None = 2
        elif amount_token is None:
            count = 1
        elif amount_token == "X":
            count = None
        else:
            count = _parse_amount(amount_token)
        return ManifestDread(count=count)
    m = _DISCARD_N_CARDS.match(phrase)
    if m is not None:
        # ``discard a card`` / ``you discard a card`` / ``discard N
        # cards`` — controller-discards. Surfaces inside compound
        # ``draw a card, then discard a card`` (Mechan Navigator) and
        # any future card that prints a bare discard clause. ``a``
        # collapses to amount=1; digits and spelled-out small
        # cardinals go through ``_parse_amount``.
        token = m.group(1)
        amount = 1 if token == "a" else _parse_amount(token)
        return DiscardSelf(amount=amount)
    m = _YOU_MAY_PAY_THEN_EFFECT.match(phrase)
    if m is not None:
        # Parse the inner-effect phrase recursively. The inner phrase is
        # already in normalized form (lowercased first letter doesn't
        # apply here — Scryfall prints "If you do, <lowercase-inner>"
        # because the inner is a sub-clause, not a sentence start). If
        # the inner doesn't parse, the whole "you may pay …" line
        # rejects so the wrapping card stays unclaimed (rather than
        # silently dropping the consequence).
        inner = _parse_effect(m.group(2))
        if inner is None:
            return None
        return OptionalManaCostThenEffect(
            cost_str=m.group(1), inner_effect=inner,
        )
    m = _OPTIONAL_DISCARD_THEN_DRAW.match(phrase)
    if m is not None:
        # ``draw a card`` / ``draw N cards`` / ``draw <word> cards`` —
        # ``a`` collapses to 1, digits and spelled-out small cardinals
        # go through ``_parse_amount``.
        token = m.group(1)
        amount = 1 if token == "a" else _parse_amount(token)
        return OptionalDiscardCardThenDrawN(amount=amount)
    m = _OPTIONAL_DISCARD_HAND_THEN_DRAW.match(phrase)
    if m is not None:
        # ``you may discard your hand. If you do, draw N cards`` —
        # Vaultguard Trooper's tapped-creatures-gated end-step trigger
        # body. Distinct from the single-card discard above: discards
        # the WHOLE hand, not a single chosen card. Tried AFTER the
        # ``discard a card`` regex above because the two surface forms
        # are disjoint (``a card`` vs ``your hand`` are non-overlapping
        # tails) but ordering ``discard a card`` first matches the
        # convention of bare-shape-before-variant that the rest of
        # this dispatcher follows.
        token = m.group(1)
        amount = 1 if token == "a" else _parse_amount(token)
        return DiscardHandThenDrawN(amount=amount)
    m = _OPTIONAL_PUT_PERMANENT_FROM_HAND_ONTO_BATTLEFIELD.match(phrase)
    if m is not None:
        # ``you may put a (creature|land) card from your hand onto the
        # battlefield`` — Sledge-Class Seedship's attack-trigger body
        # (creature) and Burgeoning's opponent-plays-land trigger body
        # (land). The captured group drives the engine-class dispatch.
        permanent_type = m.group(1)
        if permanent_type == "creature":
            return OptionalPutCreatureFromHandOntoBattlefield()
        if permanent_type == "land":
            return OptionalPutLandFromHandOntoBattlefield()
        # The regex's alternation is closed; this branch is unreachable
        # but keeps the dispatch explicit if the regex grows.
        return None
    m = _OPTIONAL_SAC_ANOTHER_FOR_PUMP_SELF_UEOT.match(phrase)
    if m is not None:
        # More-specific shape (self-pump consequence) — produces the
        # specialized ``OptionalSacAnotherForPumpSelfUEOT`` engine
        # class. Tried BEFORE the general optional-sac regex below.
        return OptionalSacAnotherForPumpSelfUEOT(
            power_bonus=int(m.group(1)),
            toughness_bonus=int(m.group(2)),
        )
    m = _OPTIONAL_SAC_LAND_OR_LANDER_FOR_BASIC_LAND_RAMP.match(phrase)
    if m is not None:
        # ``you may sacrifice a land or Lander. If you do, search your
        # library for up to N basic land cards, put them onto the
        # battlefield tapped, then shuffle.`` — Larval Scoutlander's
        # ETB body. Tried BEFORE the generic
        # ``_OPTIONAL_SAC_ANOTHER_FOR_EFFECT`` so the specialized
        # land-or-Lander shape wins; the generic regex's
        # ``creature or artifact`` subject wouldn't match this anyway,
        # but the more-specific-first ordering is the convention here.
        # Maps to ``OptionalSacLandOrLanderForBasicLandRamp(land_count=
        # N)`` which already implements the optional sac + tutor-tapped
        # sequence as a single combinator.
        return OptionalSacLandOrLanderForBasicLandRamp(
            land_count=_parse_amount(m.group(1)),
        )
    m = _OPTIONAL_SAC_ANOTHER_FOR_EFFECT.match(phrase)
    if m is not None:
        # Generic optional-sac wrapper. The inner consequence is parsed
        # recursively via the unified grammar; if the inner doesn't
        # parse, the whole line rejects so the wrapping card stays
        # unclaimed rather than silently dropping the consequence.
        inner = _parse_effect(m.group(1))
        if inner is None:
            return None
        return OptionalSacAnotherForEffect(inner_effect=inner)
    return None


def _normalize_creature_effect_phrase(phrase: str) -> str:
    """Strip the ``it`` source-subject pronoun if present, leaving the
    normalized phrase ready for ``_parse_effect``."""
    phrase = phrase.strip()
    if phrase.startswith("it "):
        phrase = phrase[len("it "):]
    return phrase


def _normalize_spell_effect_phrase(phrase: str, name: str) -> str:
    """Strip the ``<Name>`` source-subject prefix (if present), trailing
    period, and lowercase the first letter — leaving the normalized
    phrase ready for ``_parse_effect``."""
    phrase = phrase.strip()
    if phrase.endswith("."):
        phrase = phrase[:-1]
    prefix = name + " "
    if phrase.startswith(prefix):
        phrase = phrase[len(prefix):]
    if phrase and phrase[0].isupper():
        phrase = phrase[0].lower() + phrase[1:]
    return phrase


def _parse_effect_phrase(phrase: str) -> Effect | None:
    """Parse a creature-triggered-ability effect phrase. Surface form
    is ``it deals N damage…`` / ``draw a card`` / etc. Delegates to
    the unified ``_parse_effect`` after stripping the ``it`` prefix."""
    return _parse_effect(_normalize_creature_effect_phrase(phrase))


def _parse_spell_effect_phrase(phrase: str, name: str) -> Effect | None:
    """Parse a spell's full-sentence effect text.

    ``phrase`` is one sentence (period included) of oracle text. ``name``
    is the card's own name and is stripped if it appears as the subject.
    Delegates to the unified ``_parse_effect`` after normalization, so
    every effect available to creature triggered abilities is also
    available to spells (and vice versa).
    """
    return _parse_effect(_normalize_spell_effect_phrase(phrase, name))
