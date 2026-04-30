"""Whole-card name-keyed dispatch for Edge of Eternities + Duskmourn
Enchantments.

Each enchantment whose oracle text maps directly to well-typed
``CardPrinting`` fields is claimed here via a name-keyed exact-dispatch
table — one factory per card, no grammar required.

This module covers:

Non-Aura enchantments (EoE):
- Banishing Light ``{2}{W}``
- Seam Rip ``{W}``
- Loading Zone ``{3}{G}``
- Weftwalking ``{4}{U}{U}``
- Sothera, the Supervoid ``{2}{B}{B}``
- Terrasymbiosis ``{2}{G}``
- Weapons Manufacturing ``{1}{R}``
- Bioengineered Future ``{1}{G}{G}``
- Eusocial Engineering ``{3}{G}{G}``

Non-Aura enchantments (DSK):
- Leyline of Mutation ``{2}{G}{G}`` (honest-no-op: opening-hand
  replacement isn't modeled and the printed ``You may pay {W}{U}{B}
  {R}{G} rather than pay the mana cost for spells you cast.`` alt
  cost is not offered by ``legal.py`` — so the printing sits there
  inert. Same permission-not-offered pattern Lecture Hall ships
  per dsk-rooms-batch6.)
- Trapped in the Screen ``{2}{W}`` (Ward {2} + ETB exile-opp-permanent-
  until-leaves; first non-Aura enchantment to ship the new ``ward``
  kwarg on ``make_enchantment``)

Aura enchantments (EoE):
- Moonlit Meditation ``{2}{U}``
- Tractor Beam ``{2}{U}{U}``
- Chorale of the Void ``{3}{B}``
- Cryoshatter ``{U}``
- Meltstrider's Resolve ``{G}``
- Hardlight Containment ``{W}``
- Pain for All ``{2}{R}``

Aura enchantments (DSK):
- Cracked Skull ``{2}{B}`` (ETB look-at-target-opp-hand may-discard
  + DEALT_DAMAGE-destroys-enchanted)
- Duskmourn's Domination ``{4}{U}{U}``
- Frantic Strength ``{1}{R}`` (Flash + +2/+2 + trample)
- Shardmage's Rescue ``{W}`` (Flash + Enchant-creature-you-control +
  +1/+1; ``as long as this Aura entered this turn, enchanted creature
  has hexproof`` is dropped — hexproof is engine-unenforced metadata
  in this codebase per the rooms-batch6 precedent, so a conditional
  hexproof grant is also a no-op.)
- Sheltered by Ghosts ``{1}{W}`` (Enchant-creature-you-control + ETB
  exile-opp-permanent-until-leaves + +1/+0 + lifelink + ward {2})
- Sporogenic Infection ``{1}{B}`` (ETB target-opp-sacs-creature-other-
  than-enchanted + DEALT_DAMAGE-destroys-enchanted)
- Stay Hidden, Stay Silent ``{1}{U}`` (ETB tap-enchanted +
  attached-prevents-untap; the {4}{U}{U} sorcery-speed activated
  ability is dropped as honest no-op for unoffered alt-cost)
"""

from __future__ import annotations

from mtg_bot.cards.base import (
    CardPrinting,
    TriggeredAbility,
    make_aura,
    make_enchantment,
    parse_mana_cost,
)
from mtg_bot.cards.tokens import TOKEN_PRINTINGS
from mtg_bot.effects import (
    ChoraleReanimateDefenderGraveCreatureEffect,
    CreateToken,
    DestroyEnchantedPermanent,
    DrawCardsEqualToResolvingCounterAmount,
    EnchantedCreatureFightsTargetEffect,
    ExileOpponentsNonlandPermanentUntilLeaves,
    ExileTargetCreatureUntilSourceLeaves,
    PainForAllDamageEchoEffect,
    PainForAllETBEffect,
    SacrificeSelfUnlessVoidActive,
    SotheraEndStepReanimateEffect,
    SotheraExileOpponentCreatureEffect,
    TapEnchantedPermanent,
    TargetOpponentLookHandMayDiscard,
    TargetOpponentSacrificesCreatureOtherThanEnchanted,
    TractorBeamETBEffect,
    WeftwalkingETBEffect,
)
from mtg_bot.effects.object_predicates import (
    IsArtifact,
    IsCreature,
    IsLand,
    IsNontoken,
)
from mtg_bot.effects.predicates import (
    YOU,
    Controlled,
    Event,
    IsHost,
    IsSelf,
    Subject,
)
from mtg_bot.cards.parser.printing_keywords import (
    _parse_restricts_noncreature_spells_line,
    _parse_ward_line,
)
from mtg_bot.cards.parser.text import _strip_reminder_text
from mtg_bot.cards.parser.triggers import _parse_triggered_ability_line


# ---------------------------------------------------------------------------
# Non-Aura enchantment factories
# ---------------------------------------------------------------------------

def _build_banishing_light() -> CardPrinting:
    return make_enchantment(
        "Banishing Light", "{2}{W}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=ExileOpponentsNonlandPermanentUntilLeaves(),
            ),
        ),
    )


def _build_seam_rip() -> CardPrinting:
    return make_enchantment(
        "Seam Rip", "{W}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=ExileOpponentsNonlandPermanentUntilLeaves(max_mv=2),
            ),
        ),
    )


def _build_loading_zone() -> CardPrinting:
    return make_enchantment(
        "Loading Zone", "{3}{G}",
        warp="{G}",
        is_counter_doubler=True,
    )


def _build_weftwalking() -> CardPrinting:
    return make_enchantment(
        "Weftwalking", "{4}{U}{U}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=WeftwalkingETBEffect(),
            ),
        ),
        grants_first_spell_each_turn_free=True,
    )


def _build_sothera_the_supervoid() -> CardPrinting:
    return make_enchantment(
        "Sothera, the Supervoid", "{2}{B}{B}",
        legendary=True,
        triggered_abilities=(
            TriggeredAbility(
                event=Event.DIES,
                predicate=Subject(None, IsCreature()) & Controlled(YOU),
                effect=SotheraExileOpponentCreatureEffect(),
            ),
            TriggeredAbility(
                event=Event.BEGINNING_OF_END_STEP, predicate=Controlled(YOU),
                effect=SotheraEndStepReanimateEffect(),
            ),
        ),
    )


def _build_terrasymbiosis() -> CardPrinting:
    return make_enchantment(
        "Terrasymbiosis", "{2}{G}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.COUNTERS_ADDED,
                predicate=Subject(None, IsCreature()) & Controlled(YOU),
                effect=DrawCardsEqualToResolvingCounterAmount(),
                once_per_turn=True,
            ),
        ),
    )


def _build_weapons_manufacturing() -> CardPrinting:
    return make_enchantment(
        "Weapons Manufacturing", "{1}{R}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=Subject(None, IsArtifact() & IsNontoken()) & Controlled(YOU) & ~IsSelf(),
                effect=CreateToken(
                    token_printing=TOKEN_PRINTINGS["Munitions"],
                    count=1,
                ),
            ),
        ),
    )


def _build_bioengineered_future() -> CardPrinting:
    return make_enchantment(
        "Bioengineered Future", "{1}{G}{G}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=CreateToken(
                    token_printing=TOKEN_PRINTINGS["Lander"],
                    count=1,
                ),
            ),
        ),
        grants_etb_counter_per_land_entered_this_turn=True,
    )


def _build_eusocial_engineering() -> CardPrinting:
    return make_enchantment(
        "Eusocial Engineering", "{3}{G}{G}",
        warp="{1}{G}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=Subject(None, IsLand()) & Controlled(YOU),
                effect=CreateToken(
                    token_printing=TOKEN_PRINTINGS["Robot"],
                    count=1,
                ),
            ),
        ),
    )


def _build_leyline_of_mutation() -> CardPrinting:
    """Leyline of Mutation ``{2}{G}{G}``: Enchantment.

    Oracle text:
        If this card is in your opening hand, you may begin the game
        with it on the battlefield.
        You may pay {W}{U}{B}{R}{G} rather than pay the mana cost for
        spells you cast.

    Honest no-op claim. Both printed lines are inert in this engine
    today:

    - The ``opening hand`` replacement (start-of-game-on-battlefield
      Leyline mechanic) isn't modeled — every Leyline starts the game
      in the controller's library / hand like any other card.
    - The ``You may pay {W}{U}{B}{R}{G} rather than pay the mana
      cost`` alt cost isn't offered by ``legal.py``'s cast pipeline
      today, so the controller is never given the option. Same
      permission-not-offered shape Lecture Hall's ``Other permanents
      you control have hexproof`` ships per dsk-rooms-batch6 — the
      static stays in place as a no-op until the engine wires the
      enabling check.

    The card therefore sits on the battlefield doing nothing, which
    is the rules-correct observable behaviour given that neither of
    its two surface clauses is wired through. Cost-and-name-only
    factory.
    """
    return make_enchantment(
        "Leyline of Mutation", "{2}{G}{G}",
    )


def _build_trapped_in_the_screen() -> CardPrinting:
    """Trapped in the Screen ``{2}{W}``: Enchantment.

    Oracle text:
        Ward {2}
        When this enchantment enters, exile target artifact, creature,
        or enchantment an opponent controls until this enchantment
        leaves the battlefield.

    Routes through the new ``ward`` kwarg on ``make_enchantment`` —
    which threads the printed cost onto the enchantment's own
    ``ward_cost`` / ``ward_cost_str`` fields, the same fields a
    creature with Ward already populates. The ETB rides the existing
    ``ExileOpponentsNonlandPermanentUntilLeaves`` engine class —
    ``artifact, creature, or enchantment`` an opponent controls is
    functionally equivalent to ``nonland permanent`` for the EoE/DSK
    pool (planeswalkers aren't fully modeled and lands are excluded
    either way), the same simplification Banishing Light's ETB
    already uses.
    """
    return make_enchantment(
        "Trapped in the Screen", "{2}{W}",
        ward="{2}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=ExileOpponentsNonlandPermanentUntilLeaves(),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Aura enchantment factories
# ---------------------------------------------------------------------------

def _build_moonlit_meditation() -> CardPrinting:
    return make_aura(
        "Moonlit Meditation", "{2}{U}",
        replaces_first_tokens_with_copies=True,
    )


def _build_tractor_beam() -> CardPrinting:
    return make_aura(
        "Tractor Beam", "{2}{U}{U}",
        attached_prevents_untap=True,
        attached_grants_control=True,
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=TractorBeamETBEffect(),
            ),
        ),
    )


def _build_chorale_of_the_void() -> CardPrinting:
    return make_aura(
        "Chorale of the Void", "{3}{B}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ATTACKS, predicate=IsHost(subject="attacker"),
                effect=ChoraleReanimateDefenderGraveCreatureEffect(),
            ),
            TriggeredAbility(
                event=Event.BEGINNING_OF_END_STEP, predicate=Controlled(YOU),
                effect=SacrificeSelfUnlessVoidActive(),
            ),
        ),
    )


def _build_cryoshatter() -> CardPrinting:
    return make_aura(
        "Cryoshatter", "{U}",
        power_bonus=-5, toughness_bonus=0,
        triggered_abilities=(
            TriggeredAbility(
                event=Event.BECOMES_TAPPED, predicate=IsHost(),
                effect=DestroyEnchantedPermanent(),
            ),
            TriggeredAbility(
                event=Event.DEALT_DAMAGE, predicate=IsHost(),
                effect=DestroyEnchantedPermanent(),
            ),
        ),
    )


def _build_meltstriders_resolve() -> CardPrinting:
    return make_aura(
        "Meltstrider's Resolve", "{G}",
        power_bonus=0, toughness_bonus=2,
        granted_keywords=frozenset({"menace"}),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=EnchantedCreatureFightsTargetEffect(),
            ),
        ),
    )


def _build_hardlight_containment() -> CardPrinting:
    return make_aura(
        "Hardlight Containment", "{W}",
        aura_enchant_type="artifact",
        power_bonus=0, toughness_bonus=0,
        granted_keywords=frozenset(),
        attached_grants_ward="{1}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=ExileTargetCreatureUntilSourceLeaves(
                    opponents_only=True,
                ),
            ),
        ),
    )


def _build_pain_for_all() -> CardPrinting:
    """Pain for All: Enchantment — Aura. No P/T bonus. Two triggers:
    - ETB: enchanted creature deals damage equal to its power to any
      other target (``PainForAllETBEffect``).
    - Whenever enchanted creature is dealt damage, it deals that much
      damage to each opponent (``PainForAllDamageEchoEffect``).
    """
    return make_aura(
        "Pain for All", "{2}{R}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=PainForAllETBEffect(),
            ),
            TriggeredAbility(
                event=Event.DEALT_DAMAGE, predicate=IsHost(),
                effect=PainForAllDamageEchoEffect(),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Duskmourn (DSK) Aura factories
# ---------------------------------------------------------------------------

def _build_frantic_strength() -> CardPrinting:
    """Frantic Strength ``{1}{R}``: Enchantment — Aura.

    Oracle text:
        Flash
        Enchant creature
        Enchanted creature gets +2/+2 and has trample.

    Maps onto the existing ``make_aura`` flag set with the new
    ``keywords`` kwarg carrying Flash on the Aura itself:
    - ``keywords=frozenset({"flash"})`` for instant-speed casting (the
      ``CardPrinting.keywords`` field is what ``legal.py`` reads to
      surface a permanent spell at instant speed).
    - ``power_bonus=2, toughness_bonus=2`` and
      ``granted_keywords=frozenset({"trample"})`` for the static
      ``Enchanted creature gets +2/+2 and has trample.`` rider —
      same shared ``equipment_bonuses`` machinery the rest of the
      EoE/DSK Auras already ride.

    No new engine fields are needed: Flash on permanents already
    gates instant-speed casting via the existing
    ``"flash" in printing.keywords`` check, the pump and granted
    keyword ride the existing aura-attached effective-PT layer.
    """
    return make_aura(
        "Frantic Strength", "{1}{R}",
        power_bonus=2, toughness_bonus=2,
        granted_keywords=frozenset({"trample"}),
        keywords=frozenset({"flash"}),
    )


def _build_sheltered_by_ghosts() -> CardPrinting:
    """Sheltered by Ghosts ``{1}{W}``: Enchantment — Aura.

    Oracle text:
        Enchant creature you control
        When this Aura enters, exile target nonland permanent an
        opponent controls until this Aura leaves the battlefield.
        Enchanted creature gets +1/+0 and has lifelink and ward {2}.

    Composes three existing engine pieces against the new
    ``aura_yours_only`` flag (which restricts the cast-time target
    picker to creatures the caster controls — the printed
    "Enchant creature you control" line):

    - ``aura_yours_only=True`` for the controller-restricted enchant
      target. First DSK Aura that ships this restriction; the flag
      defaults to False so existing "Enchant creature" Auras keep
      their previous behavior.
    - ``ExileOpponentsNonlandPermanentUntilLeaves()`` ETB — same
      Banishing-Light-style exile-until-leaves shape. The
      ``max_mv=0`` default disables the optional MV cap (Sheltered
      doesn't have one).
    - ``power_bonus=1, toughness_bonus=0`` and
      ``granted_keywords=frozenset({"lifelink"})`` for the static
      ``Enchanted creature gets +1/+0 and has lifelink`` rider —
      shared ``equipment_bonuses`` machinery.
    - ``attached_grants_ward="{2}"`` for the ``ward {2}`` rider —
      same Hardlight-Containment-style ward grant, just with a
      stronger cost.

    No new engine effect classes are needed: the new
    ``aura_yours_only`` flag is the only engine plumbing this card
    requires.
    """
    return make_aura(
        "Sheltered by Ghosts", "{1}{W}",
        power_bonus=1, toughness_bonus=0,
        granted_keywords=frozenset({"lifelink"}),
        aura_yours_only=True,
        attached_grants_ward="{2}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=ExileOpponentsNonlandPermanentUntilLeaves(),
            ),
        ),
    )


def _build_cracked_skull() -> CardPrinting:
    """Cracked Skull ``{2}{B}``: Enchantment — Aura.

    Oracle text:
        Enchant creature
        When this Aura enters, look at target player's hand. You may
        choose a nonland card from it. That player discards that card.
        When enchanted creature is dealt damage, destroy it.

    Composes two existing engine pieces against the new
    ``TargetOpponentLookHandMayDiscard`` ETB:

    - ``TargetOpponentLookHandMayDiscard()`` ETB — the new generic
      effect that honours the printed ``You may`` gate via
      ``Agent.choose_yes_no``. Opponent-only legality is the standard
      EoE / DSK ``target player`` simplification (the same convention
      ``TargetOpponentDiscardsCasterChoice`` already rides). Self-
      targeting is never strategically correct for the controller, so
      narrowing legality doesn't affect rational play.
    - ``DestroyEnchantedPermanent()`` on ``Event.DEALT_DAMAGE & IsHost()``
      — same Cryoshatter-style trigger that fires when the enchanted
      creature is dealt any damage and destroys it via the
      destroys-the-host primitive.
    """
    return make_aura(
        "Cracked Skull", "{2}{B}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=TargetOpponentLookHandMayDiscard(),
            ),
            TriggeredAbility(
                event=Event.DEALT_DAMAGE, predicate=IsHost(),
                effect=DestroyEnchantedPermanent(),
            ),
        ),
    )


def _build_stay_hidden_stay_silent() -> CardPrinting:
    """Stay Hidden, Stay Silent ``{1}{U}``: Enchantment — Aura.

    Oracle text:
        Enchant creature
        When this Aura enters, tap enchanted creature.
        Enchanted creature doesn't untap during its controller's
        untap step.
        {4}{U}{U}: Shuffle enchanted creature into its owner's
        library, then manifest dread. Activate only as a sorcery.

    Composes the new ``TapEnchantedPermanent`` ETB and the existing
    ``attached_prevents_untap`` static rider (Tractor Beam precedent):

    - ETB: ``TapEnchantedPermanent()`` — taps the host via
      ``mtg_bot.turn.tap_permanent``, idempotent if the host is
      already tapped.
    - Static: ``attached_prevents_untap=True`` for the printed
      ``doesn't untap during its controller's untap step`` clause —
      same flag Tractor Beam ships, gated by the engine's untap-step
      walk on every Aura attached to the host.

    Simplification — the activated ability ``{4}{U}{U}: Shuffle
    enchanted creature into its owner's library, then manifest
    dread. Activate only as a sorcery.`` is dropped as an honest
    no-op. The activated ability is OPTIONAL (the controller chooses
    whether to pay {4}{U}{U}), so not offering it through ``legal``
    means the controller misses the option but doesn't violate
    rules. Same permission-not-offered shape Lecture Hall ships per
    dsk-rooms-batch6, applied to an alt-cost activated ability
    rather than a static. The ETB tap and the don't-untap static are
    the load-bearing pieces of the printing — every game with this
    Aura on the battlefield has the host stuck-tapped indefinitely
    regardless of whether the activated ability is ever used.
    """
    return make_aura(
        "Stay Hidden, Stay Silent", "{1}{U}",
        attached_prevents_untap=True,
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=TapEnchantedPermanent(),
            ),
        ),
    )


def _build_sporogenic_infection() -> CardPrinting:
    """Sporogenic Infection ``{1}{B}``: Enchantment — Aura.

    Oracle text:
        Enchant creature
        When this Aura enters, target player sacrifices a creature of
        their choice other than enchanted creature.
        When enchanted creature is dealt damage, destroy it.

    Composes the new ``TargetOpponentSacrificesCreatureOtherThan
    Enchanted`` ETB and the existing Cryoshatter-style
    DEALT_DAMAGE-destroys-host trigger:

    - ETB: ``TargetOpponentSacrificesCreatureOtherThanEnchanted()`` —
      identifies the host via ``state.resolving_source.attached_to``
      and excludes it from the target's sac options. The simplified
      ``target player`` -> ``target opponent`` legality narrowing is
      the same convention Cracked Skull's may-discard ETB rides.
    - DEALT_DAMAGE: ``DestroyEnchantedPermanent()`` keyed on
      ``IsHost()`` — same shape Cryoshatter's bottom trigger uses.
    """
    return make_aura(
        "Sporogenic Infection", "{1}{B}",
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS, predicate=IsSelf(),
                effect=TargetOpponentSacrificesCreatureOtherThanEnchanted(),
            ),
            TriggeredAbility(
                event=Event.DEALT_DAMAGE, predicate=IsHost(),
                effect=DestroyEnchantedPermanent(),
            ),
        ),
    )


def _build_shardmages_rescue() -> CardPrinting:
    """Shardmage's Rescue ``{W}``: Enchantment — Aura.

    Oracle text:
        Flash
        Enchant creature you control
        As long as this Aura entered this turn, enchanted creature has
        hexproof.
        Enchanted creature gets +1/+1.

    Maps cleanly onto the existing ``make_aura`` flag set:

    - ``keywords=frozenset({"flash"})`` for the printed Flash on the
      Aura itself — same instant-speed gate Frantic Strength rides
      through ``CardPrinting.keywords`` and ``legal.py``.
    - ``aura_yours_only=True`` for the ``Enchant creature you control``
      target restriction — same flag Sheltered by Ghosts ships.
    - ``power_bonus=1, toughness_bonus=1`` for the unconditional
      ``Enchanted creature gets +1/+1`` rider — shared
      ``equipment_bonuses`` machinery.

    The conditional hexproof line (``As long as this Aura entered this
    turn, enchanted creature has hexproof``) is dropped as an honest
    no-op: hexproof is engine-unenforced metadata in this codebase
    today (no targeting site checks for it — same observation Lecture
    Hall's "other permanents you control have hexproof" rests on per
    the dsk-rooms-batch6 precedent). Since the unconditional grant
    would already be a no-op, the conditional ``as long as`` gated
    grant is just as much a no-op — not modelling the optional
    benefit makes the printing miss a defensive shield it never had
    anyway.
    """
    return make_aura(
        "Shardmage's Rescue", "{W}",
        power_bonus=1, toughness_bonus=1,
        keywords=frozenset({"flash"}),
        aura_yours_only=True,
    )


def _build_duskmourns_domination() -> CardPrinting:
    """Duskmourn's Domination ``{4}{U}{U}``: Enchantment — Aura.

    Oracle text:
        Enchant creature
        You control enchanted creature.
        Enchanted creature gets -3/-0 and loses all abilities.

    Maps onto the existing ``make_aura`` flag set:
    - ``attached_grants_control=True`` for the gain-control rider
      (Tractor Beam precedent: the EoE Aura that grants control of
      its enchanted creature).
    - ``attached_grants_lose_all_abilities=True`` for the lose-all-
      abilities rider (already wired on ``make_aura``; no DSK-
      specific engine field needed).
    - ``power_bonus=-3, toughness_bonus=0`` for the ``-3/-0``
      continuous bonus, routed through the shared
      ``equipment_bonuses`` machinery the same way Cryoshatter's
      ``-5/-0`` is.
    """
    return make_aura(
        "Duskmourn's Domination", "{4}{U}{U}",
        power_bonus=-3, toughness_bonus=0,
        attached_grants_control=True,
        attached_grants_lose_all_abilities=True,
    )


# ---------------------------------------------------------------------------
# Dispatch table: card name → factory
# ---------------------------------------------------------------------------

_ENCHANTMENT_BUILDERS: dict[str, "callable[[], CardPrinting]"] = {
    # Non-Aura
    "Loading Zone": _build_loading_zone,
    "Weftwalking": _build_weftwalking,
    "Sothera, the Supervoid": _build_sothera_the_supervoid,
    "Terrasymbiosis": _build_terrasymbiosis,
    "Weapons Manufacturing": _build_weapons_manufacturing,
    "Bioengineered Future": _build_bioengineered_future,
    "Eusocial Engineering": _build_eusocial_engineering,
    "Leyline of Mutation": _build_leyline_of_mutation,
    # Auras
    "Moonlit Meditation": _build_moonlit_meditation,
    "Tractor Beam": _build_tractor_beam,
    "Chorale of the Void": _build_chorale_of_the_void,
    "Cryoshatter": _build_cryoshatter,
    "Meltstrider's Resolve": _build_meltstriders_resolve,
    "Hardlight Containment": _build_hardlight_containment,
    "Pain for All": _build_pain_for_all,
    # Duskmourn (DSK) Auras
    "Cracked Skull": _build_cracked_skull,
    "Duskmourn's Domination": _build_duskmourns_domination,
    "Frantic Strength": _build_frantic_strength,
    "Shardmage's Rescue": _build_shardmages_rescue,
    "Sheltered by Ghosts": _build_sheltered_by_ghosts,
    "Sporogenic Infection": _build_sporogenic_infection,
    "Stay Hidden, Stay Silent": _build_stay_hidden_stay_silent,
}


def _build_enchantment_printing(card: dict) -> CardPrinting | None:
    """Claim an Enchantment card via name-keyed dispatch, falling back to
    a printing-flag-only oracle-text path for cards whose entire oracle
    text consists of recognized printing-level static lines (Deafening
    Silence's noncreature-spell restriction etc.).

    Returns the canonical ``CardPrinting`` if either path claims the
    card. Returns ``None`` otherwise so the builder chain continues.
    The exclusion guard rejects hybrid card types (Creature /
    Land / Artifact); their own builders own those.
    """
    type_line = card.get("type_line", "")
    if "Enchantment" not in type_line:
        return None
    # Exclude hybrid card types — let their own builders handle them.
    for exclusion in ("Creature", "Land", "Artifact"):
        if exclusion in type_line:
            return None
    factory = _ENCHANTMENT_BUILDERS.get(card.get("name", ""))
    if factory is not None:
        return factory()
    return _build_enchantment_via_oracle_text(card)


# Allowlist of names that may claim through the generic oracle-text
# fallback path below. Mirrors the room builder's
# ``_UNIFIED_ROOM_NAMES`` precedent — incremental migration: only
# names listed here can be reached by ``_build_enchantment_via_
# oracle_text``, so adding a new name to the allowlist is the
# explicit opt-in for that card. Other enchantments still take the
# name-dispatched bespoke factory path (or fall through to silent
# skip).
_UNIFIED_ENCHANTMENT_NAMES: frozenset[str] = frozenset({
    "Deafening Silence",
    "Burgeoning",
    # Banishing Light / Seam Rip / Trapped in the Screen — ETB exile-until-
    # source-leaves shapes.  The effect phrases are now in the grammar
    # (``_EXILE_OPPONENTS_NONLAND_UNTIL_LEAVES`` in effects.py).
    "Banishing Light",
    "Seam Rip",
    "Trapped in the Screen",
})


def _build_enchantment_via_oracle_text(
    card: dict,
) -> CardPrinting | None:
    """Fallback path: claim an Enchantment whose entire oracle text
    parses through recognized line shapes — printing-flag statics
    (Deafening Silence) or triggered-ability lines (Burgeoning).

    Gated on ``_UNIFIED_ENCHANTMENT_NAMES`` so the path only attempts
    cards that have been explicitly migrated. Returns ``None`` for
    any card with an unrecognized line so the silent-skip path
    mirrors the rest of the parser.
    """
    name = card.get("name", "")
    if name not in _UNIFIED_ENCHANTMENT_NAMES:
        return None

    text = card.get("oracle_text", "")
    legendary = "Legendary" in card.get("type_line", "")
    restricts_noncreature_spells = False
    ward: str | None = None
    triggers: list[TriggeredAbility] = []

    for raw_line in text.split("\n"):
        line = _strip_reminder_text(raw_line)
        if not line:
            continue
        if _parse_restricts_noncreature_spells_line(line):
            restricts_noncreature_spells = True
            continue
        ward_cost = _parse_ward_line(line)
        if ward_cost is not None:
            ward = ward_cost
            continue
        parsed_trigger = _parse_triggered_ability_line(line, name=name)
        if parsed_trigger is not None:
            if isinstance(parsed_trigger, tuple):
                triggers.extend(parsed_trigger)
            else:
                triggers.append(parsed_trigger)
            continue
        # Anything else — Aura grammar, activated abilities, unrecognized
        # statics — falls through to the silent-skip path.
        return None

    if not restricts_noncreature_spells and not triggers:
        # Nothing recognized — leave the card for a future grammar
        # slice rather than emit a blank Enchantment.
        return None

    return make_enchantment(
        name=card["name"],
        cost=card.get("mana_cost", ""),
        legendary=legendary,
        restricts_noncreature_spells_to_one_per_turn=(
            restricts_noncreature_spells
        ),
        ward=ward,
        triggered_abilities=tuple(triggers),
    )
