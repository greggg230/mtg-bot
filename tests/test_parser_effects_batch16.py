"""Tests for batch-16 whole-spell exact-text dispatch additions.

This batch extends the ``_WHOLE_SPELL_EXACT_TEXT`` dispatcher in
``parser.spell_builder`` with seven entries — the cluster of EoE
spells whose oracle text is a single sentence-block (or a tightly
bound two-/three-sentence block) that the static registry models
with one engine class. The per-clause grammar can't claim these
because at least one of their internal sentences (e.g. the
``Then if that creature's power is 0 or less, destroy it.`` rider
on Depressurize, or the ``If it has a +1/+1 counter on it, draw a
card.`` rider on Dual-Sun Technique) doesn't parse as a standalone
effect — and matching the per-sentence grammar to those forms
would yield primitives no other card uses.

Newly claimed spells:

* **Depressurize** ``{1}{B}`` Instant — target creature gets -3/-0
  UEOT plus a state-based death check. Maps to
  ``DepressurizeEffect(power_penalty=3)``.
* **Focus Fire** ``{W}`` Instant — variable-X burn whose X reads
  ``2 plus the number of creatures and/or Spacecraft you control``.
  Maps to ``FocusFireEffect(base=2)``.
* **Diplomatic Relations** ``{2}{G}`` Instant — pump-and-grant-
  vigilance on your creature plus a fight-shaped damage rider.
  Two-target spell. Maps to ``DiplomaticRelationsEffect()``.
* **Biosynthic Burst** ``{1}{G}`` Instant — counter + grant-three-
  keywords + untap on a creature you control. Maps to
  ``BiosynthicBurstEffect()``.
* **Zealous Display** ``{2}{W}`` Instant — anthem with an opp-turn-
  untap rider. Maps to
  ``PumpCreaturesYouControlUntilEOT(power_bonus=2,
  also_untap_if_not_your_turn=True)``.
* **Dual-Sun Technique** ``{1}{W}`` Instant — grant double strike
  + conditional draw rider. Maps to
  ``PumpTargetCreatureUntilEOT`` with
  ``draw_if_target_has_plus_one_counter=1``.
* **Invasive Maneuvers** ``{1}{R}`` Instant — bare burn with a
  Spacecraft-state boost. Maps to
  ``DealDamageToTargetCreatureSpacecraftBoosted(base=3, boosted=5)``.

Each spell is exercised by:
  - A claim test confirming the parser's ``claimed_names`` includes
    it.
  - A dispatch test confirming
    ``_try_whole_spell_exact_text`` returns the same engine-class
    instance the static side ships.
  - The shared parametrized parity test in
    ``tests/test_parser_parity.py`` enforces that the produced
    ``CardPrinting`` matches the static one (no separate per-card
    parity assert needed here).
"""

from __future__ import annotations

from mtg_bot.cards.base import (
    BiosynthicBurstEffect,
    ConditionalIfVoid,
    CreateXTokensOfTargetPermanentWithHasteEndStepSac,
    DealDamageToTargetCreatureSpacecraftBoosted,
    DealDamageToTargetCreatureWithExcessLanderRider,
    DepressurizeEffect,
    DestroyTargetNonlandPermanentWithLanderForController,
    DiplomaticRelationsEffect,
    DrawCards,
    EachOpponentLosesLife,
    FocusFireEffect,
    LoseLife,
    PumpCreaturesYouControlUntilEOT,
    PumpTargetCreatureUntilEOT,
    ReturnTargetCreatureToOwnersHand,
    ReturnUpToNSmallCreaturesFromGraveyardWithCounters,
    SeedshipImpactEffect,
    Sequence,
    Surveil,
    SystemsOverrideEffect,
    ZeroPointBalladEffect,
)
from mtg_bot.cards.parser.spell_builder import (
    _parse_spell_oracle_text,
    _try_whole_spell_exact_text,
)
from mtg_bot.cards.parser_source import OracleParserCardSource


def _claimed_names() -> frozenset[str]:
    return frozenset(OracleParserCardSource().claimed_names)


# ---------- claim tests (end-to-end via the parser source) ----------

def test_depressurize_claimed_by_parser():
    # Depressurize ``{1}{B}`` Instant — -3/-0 UEOT plus the
    # ``Then if that creature's power is 0 or less, destroy it.``
    # rider. The rider doesn't parse as a standalone effect, so the
    # spell is claimed via the whole-spell exact-text dispatcher.
    assert "Depressurize" in _claimed_names()


def test_focus_fire_claimed_by_parser():
    # Focus Fire ``{W}`` Instant — variable-X burn (2 plus creatures
    # and/or Spacecraft you control). The X-formula syntax has no
    # corresponding primitive in the per-clause grammar, so the spell
    # is claimed via the whole-spell exact-text dispatcher.
    assert "Focus Fire" in _claimed_names()


def test_diplomatic_relations_claimed_by_parser():
    # Diplomatic Relations ``{2}{G}`` Instant — two-target spell
    # (pump + grant-vigilance on yours, then fight-shaped damage at
    # an opponent's). Two-target compounds aren't supported by the
    # per-clause Sequence combinator, so the spell is claimed via
    # the whole-spell exact-text dispatcher.
    assert "Diplomatic Relations" in _claimed_names()


def test_biosynthic_burst_claimed_by_parser():
    # Biosynthic Burst ``{1}{G}`` Instant — counter + grant three
    # keywords + untap on a creature you control. The compound
    # spans more than the per-clause Sequence combinator handles
    # (the carry-over ``it`` subject across three sentences), so
    # the spell is claimed via the whole-spell exact-text dispatcher.
    assert "Biosynthic Burst" in _claimed_names()


def test_zealous_display_claimed_by_parser():
    # Zealous Display ``{2}{W}`` Instant — anthem with the
    # ``If it's not your turn, untap those creatures.`` rider. The
    # rider doesn't parse as a standalone effect, so the spell is
    # claimed via the whole-spell exact-text dispatcher.
    assert "Zealous Display" in _claimed_names()


def test_dual_sun_technique_claimed_by_parser():
    # Dual-Sun Technique ``{1}{W}`` Instant — grant double strike
    # plus conditional draw rider (``If it has a +1/+1 counter on
    # it, draw a card.``). The rider doesn't parse as a standalone
    # effect, so the spell is claimed via the whole-spell exact-
    # text dispatcher.
    assert "Dual-Sun Technique" in _claimed_names()


def test_invasive_maneuvers_claimed_by_parser():
    # Invasive Maneuvers ``{1}{R}`` Instant — bare burn with the
    # ``It deals 5 damage instead if you control a Spacecraft.``
    # boost rider. The rider's state-substitution syntax has no
    # corresponding primitive in the per-clause grammar, so the
    # spell is claimed via the whole-spell exact-text dispatcher.
    assert "Invasive Maneuvers" in _claimed_names()


# ---------- dispatch tests (direct lookup) ----------

def test_dispatch_depressurize_returns_engine_class():
    # Direct lookup against the dispatcher with the verbatim
    # Depressurize oracle text. Returns the static-side bespoke
    # engine class with the printed power penalty (3).
    eff = _try_whole_spell_exact_text(
        "Target creature gets -3/-0 until end of turn. "
        "Then if that creature's power is 0 or less, destroy it."
    )
    assert eff == DepressurizeEffect(power_penalty=3)


def test_dispatch_focus_fire_returns_engine_class():
    # Focus Fire's X-formula reads "2 plus the number of creatures
    # and/or Spacecraft you control"; the engine class's ``base=2``
    # field captures the additive constant and the rest is
    # computed at resolve time.
    eff = _try_whole_spell_exact_text(
        "Focus Fire deals X damage to target attacking or blocking "
        "creature, where X is 2 plus the number of creatures and/or "
        "Spacecraft you control."
    )
    assert eff == FocusFireEffect(base=2)


def test_dispatch_diplomatic_relations_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Target creature you control gets +1/+0 and gains vigilance "
        "until end of turn. It deals damage equal to its power to "
        "target creature an opponent controls."
    )
    assert eff == DiplomaticRelationsEffect()


def test_dispatch_biosynthic_burst_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Put a +1/+1 counter on target creature you control. "
        "It gains reach, trample, and indestructible until end of "
        "turn. Untap it."
    )
    assert eff == BiosynthicBurstEffect()


def test_dispatch_zealous_display_returns_engine_class():
    # ``also_untap_if_not_your_turn=True`` flips the engine class
    # into the opp-turn untap branch when the spell resolves on
    # an opponent's turn. The base anthem is +2/+0 UEOT.
    eff = _try_whole_spell_exact_text(
        "Creatures you control get +2/+0 until end of turn. "
        "If it's not your turn, untap those creatures."
    )
    assert eff == PumpCreaturesYouControlUntilEOT(
        power_bonus=2,
        toughness_bonus=0,
        also_untap_if_not_your_turn=True,
    )


def test_dispatch_dual_sun_technique_returns_engine_class():
    # The conditional draw rider lands on the engine class's
    # ``draw_if_target_has_plus_one_counter`` field; the keyword
    # grant rides the standard pump path. ``you_control_only=True``
    # restricts the legal-target pool to your own creatures.
    eff = _try_whole_spell_exact_text(
        "Target creature you control gains double strike until end "
        "of turn. If it has a +1/+1 counter on it, draw a card."
    )
    assert eff == PumpTargetCreatureUntilEOT(
        power_bonus=0,
        toughness_bonus=0,
        granted_keywords=frozenset({"double strike"}),
        you_control_only=True,
        draw_if_target_has_plus_one_counter=1,
    )


def test_dispatch_invasive_maneuvers_returns_engine_class():
    # The base burn is 3; the ``boosted=5`` field fires when the
    # controller has a Spacecraft on the battlefield at resolve
    # time.
    eff = _try_whole_spell_exact_text(
        "Invasive Maneuvers deals 3 damage to target creature. "
        "It deals 5 damage instead if you control a Spacecraft."
    )
    assert eff == DealDamageToTargetCreatureSpacecraftBoosted(
        base=3, boosted=5,
    )


# ---------- end-to-end through ``_parse_spell_oracle_text`` ----------

def test_parse_spell_oracle_text_routes_depressurize_through_whole_spell():
    # End-to-end: the spell-oracle-text parser MUST hit the whole-
    # spell exact-text branch BEFORE the per-clause splitter, since
    # Depressurize's second sentence (``Then if that creature's
    # power is 0 or less, destroy it.``) doesn't parse as a
    # standalone effect. If the dispatcher were tried after the
    # splitter, the per-clause Sequence path would reject and
    # leave the spell unclaimed.
    text = (
        "Target creature gets -3/-0 until end of turn. "
        "Then if that creature's power is 0 or less, destroy it."
    )
    assert _parse_spell_oracle_text(text, "Depressurize") == (
        DepressurizeEffect(power_penalty=3)
    )


def test_parse_spell_oracle_text_strips_biosynthic_burst_reminder():
    # Biosynthic Burst's printed oracle text carries a parenthesized
    # reminder tail describing indestructible (``(Damage and effects
    # that say "destroy" don't destroy it.)``). The parser must
    # strip that reminder BEFORE the dispatcher lookup, since the
    # dispatcher is exact-match.
    text = (
        "Put a +1/+1 counter on target creature you control. "
        "It gains reach, trample, and indestructible until end of "
        "turn. Untap it. (Damage and effects that say \"destroy\" "
        "don't destroy it.)"
    )
    assert _parse_spell_oracle_text(text, "Biosynthic Burst") == (
        BiosynthicBurstEffect()
    )


def test_dispatch_returns_none_for_depressurize_with_typo():
    # The dispatcher is exact-match, so a one-character paraphrase
    # of the Depressurize key rejects. Sanity-check that the new
    # entry doesn't accidentally fuzzy-match nearby surfaces.
    assert _try_whole_spell_exact_text(
        "Target creature gets -3/-0 until end of turn. "
        "Then if that creature's power is 0 or fewer, destroy it."
    ) is None


# ---------- Sequence-shaped whole-spell entries (Cerebral Download,
#            Decode Transmissions, Hymn of the Faller) ----------
#
# These three spells share the property that the static side ships
# them as a multi-effect ``Sequence(...)`` whose internal sentences
# don't all parse as standalone effects. Cerebral Download's first
# sentence is a variable surveil (``Surveil X, where X is the number
# of artifacts you control``) — no per-clause primitive captures the
# X-formula. Decode Transmissions has a Void rider that REPLACES
# the lose-life half of a compound base (the existing void-rider
# helpers swap whole effects, not half-effects). Hymn of the Faller
# has a Void rider that ADDS a draw on top of the base (the existing
# helpers swap, not add).
#
# All three are cleanly modeled by the static engine classes; the
# whole-spell exact-text dispatcher is the natural fit because each
# spell's surface text is a fixed verbatim block. Multi-line keys
# carry the literal newline between header and rider; reminder text
# is stripped per-clause before lookup.

def test_cerebral_download_claimed_by_parser():
    # Cerebral Download ``{4}{U}`` Instant — variable surveil + draw
    # 3. The static models the surveil with ``Surveil(use_artifact_
    # count=True)``; the X-formula is computed at resolve time.
    assert "Cerebral Download" in _claimed_names()


def test_decode_transmissions_claimed_by_parser():
    # Decode Transmissions ``{2}{B}`` Sorcery — draw 2 + Void-
    # replaceable lose-2 / each-opponent-loses-2.
    assert "Decode Transmissions" in _claimed_names()


def test_hymn_of_the_faller_claimed_by_parser():
    # Hymn of the Faller ``{1}{B}`` Sorcery — surveil 1, draw 1,
    # lose 1, plus a Void-conditional extra draw rider.
    assert "Hymn of the Faller" in _claimed_names()


def test_dispatch_cerebral_download_returns_sequence():
    # The dispatcher returns the static-side Sequence directly. The
    # ``amount=1`` on Surveil is the dataclass default and matches
    # the static initializer (which omits the kwarg, falling back to
    # the same default).
    eff = _try_whole_spell_exact_text(
        "Surveil X, where X is the number of artifacts you control. "
        "Then draw three cards."
    )
    assert eff == Sequence(effects=(
        Surveil(amount=1, use_artifact_count=True),
        DrawCards(amount=3),
    ))


def test_dispatch_decode_transmissions_returns_sequence_with_conditional():
    # The Void rider in Decode Transmissions only swaps the lose-life
    # half of the base sentence; the draw runs unconditionally. The
    # static models this as ``Sequence(DrawCards, ConditionalIfVoid(
    # then=EachOpponentLosesLife, else=LoseLife))``. The dispatch key
    # carries the literal newline between base and rider sentences.
    eff = _try_whole_spell_exact_text(
        "You draw two cards and lose 2 life.\n"
        "Void — If a nonland permanent left the battlefield this turn "
        "or a spell was warped this turn, instead you draw two cards "
        "and each opponent loses 2 life."
    )
    assert eff == Sequence(effects=(
        DrawCards(amount=2),
        ConditionalIfVoid(
            then_effect=EachOpponentLosesLife(amount=2),
            else_effect=LoseLife(amount=2),
        ),
    ))


def test_dispatch_hymn_of_the_faller_returns_sequence_with_void_rider():
    # The Void rider in Hymn of the Faller ADDS an extra draw on top
    # of the base (it doesn't swap an effect). The static models the
    # rider as ``ConditionalIfVoid(then=DrawCards(1), else=None)``:
    # the conditional resolves the extra draw only if the Void state
    # is active; otherwise it's a no-op (else=None).
    eff = _try_whole_spell_exact_text(
        "Surveil 1, then you draw a card and lose 1 life.\n"
        "Void — If a nonland permanent left the battlefield this turn "
        "or a spell was warped this turn, draw another card."
    )
    assert eff == Sequence(effects=(
        Surveil(amount=1, use_artifact_count=False),
        DrawCards(amount=1),
        LoseLife(amount=1),
        ConditionalIfVoid(
            then_effect=DrawCards(amount=1),
            else_effect=None,
        ),
    ))


def test_desculpting_blast_claimed_by_parser():
    # Desculpting Blast ``{1}{U}`` Instant — bounce-target-permanent
    # plus a Drone-token rider for the attacking-permanent branch.
    assert "Desculpting Blast" in _claimed_names()


def test_emergency_eject_claimed_by_parser():
    # Emergency Eject ``{2}{W}`` Instant — destroy + a Lander-for-
    # the-destroyed-permanent's-controller rider.
    assert "Emergency Eject" in _claimed_names()


def test_orbital_plunge_claimed_by_parser():
    # Orbital Plunge ``{3}{R}`` Sorcery — 6-damage burn + excess-
    # damage Lander rider.
    assert "Orbital Plunge" in _claimed_names()


def test_seedship_impact_claimed_by_parser():
    # Seedship Impact ``{1}{G}`` Instant — destroy artifact-or-
    # enchantment + small-mv Lander rider.
    assert "Seedship Impact" in _claimed_names()


def test_scout_for_survivors_claimed_by_parser():
    # Scout for Survivors ``{2}{W}`` Sorcery — return up to three
    # small creature cards from graveyard to battlefield with a
    # +1/+1 counter on each.
    assert "Scout for Survivors" in _claimed_names()


def test_zero_point_ballad_claimed_by_parser():
    # Zero Point Ballad ``{X}{B}`` Sorcery — variable-X wrath +
    # life-pay + reanimation-on-large-X rider.
    assert "Zero Point Ballad" in _claimed_names()


def test_systems_override_claimed_by_parser():
    # Systems Override ``{2}{R}`` Sorcery — gain-control + untap +
    # haste plus a Spacecraft-only ten-charge-counter rider.
    assert "Systems Override" in _claimed_names()


def test_devastating_onslaught_claimed_by_parser():
    # Devastating Onslaught ``{X}{X}{R}`` Sorcery — variable-X
    # token-copy spell with haste and end-step sac.
    assert "Devastating Onslaught" in _claimed_names()


def test_full_bore_claimed_by_parser():
    # Full Bore ``{R}`` Instant — pump-your-creature with warp-
    # conditional extra keywords (trample, haste).
    assert "Full Bore" in _claimed_names()


def test_dispatch_desculpting_blast_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Return target nonland permanent to its owner's hand. "
        "If it was attacking, create a 1/1 colorless Drone artifact "
        "creature token with flying and \"This token can block only "
        "creatures with flying.\""
    )
    assert eff == ReturnTargetCreatureToOwnersHand(
        allow_nonland_permanent=True,
        create_drone_if_was_attacking=True,
    )


def test_dispatch_emergency_eject_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Destroy target nonland permanent. "
        "Its controller creates a Lander token."
    )
    assert eff == DestroyTargetNonlandPermanentWithLanderForController()


def test_dispatch_orbital_plunge_returns_engine_class():
    # ``amount=6`` matches the printed damage; the bespoke class
    # gates the Lander rider on excess damage at resolve time.
    eff = _try_whole_spell_exact_text(
        "Orbital Plunge deals 6 damage to target creature. "
        "If excess damage was dealt this way, create a Lander token."
    )
    assert eff == DealDamageToTargetCreatureWithExcessLanderRider(amount=6)


def test_dispatch_seedship_impact_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Destroy target artifact or enchantment. "
        "If its mana value was 2 or less, create a Lander token."
    )
    assert eff == SeedshipImpactEffect()


def test_dispatch_scout_for_survivors_returns_engine_class():
    # ``max_n=3`` and ``max_total_mv=3`` from the printed bounds.
    eff = _try_whole_spell_exact_text(
        "Return up to three target creature cards with total mana "
        "value 3 or less from your graveyard to the battlefield. "
        "Put a +1/+1 counter on each of them."
    )
    assert eff == ReturnUpToNSmallCreaturesFromGraveyardWithCounters(
        max_n=3, max_total_mv=3,
    )


def test_dispatch_zero_point_ballad_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Destroy all creatures with toughness X or less. You lose "
        "X life. If X is 6 or more, return a creature card put into "
        "a graveyard this way to the battlefield under your control."
    )
    assert eff == ZeroPointBalladEffect()


def test_dispatch_systems_override_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Gain control of target artifact or creature until end of "
        "turn. Untap that permanent. It gains haste until end of "
        "turn. If it's a Spacecraft, put ten charge counters on it. "
        "If you do, remove ten charge counters from it at the "
        "beginning of the next end step."
    )
    assert eff == SystemsOverrideEffect()


def test_dispatch_devastating_onslaught_returns_engine_class():
    eff = _try_whole_spell_exact_text(
        "Create X tokens that are copies of target artifact or "
        "creature you control. Those tokens gain haste until end "
        "of turn. Sacrifice them at the beginning of the next end "
        "step."
    )
    assert eff == CreateXTokensOfTargetPermanentWithHasteEndStepSac()


def test_dispatch_full_bore_returns_engine_class():
    # The warp-conditional grant lands on the engine class's
    # ``granted_keywords_if_warp_cast`` field; the base pump is
    # +3/+2 UEOT on a creature you control.
    eff = _try_whole_spell_exact_text(
        "Target creature you control gets +3/+2 until end of turn. "
        "If that creature was cast for its warp cost, it also gains "
        "trample and haste until end of turn."
    )
    assert eff == PumpTargetCreatureUntilEOT(
        power_bonus=3,
        toughness_bonus=2,
        you_control_only=True,
        granted_keywords_if_warp_cast=frozenset({"trample", "haste"}),
    )


def test_dispatch_strips_hymn_of_the_faller_reminder():
    # End-to-end through ``_parse_spell_oracle_text``: the surveil-1
    # parenthesized reminder ``(To surveil 1, ...)`` is per-clause
    # stripped BEFORE the dispatcher key lookup. The Void rider line
    # has no parenthesized reminder, so it survives intact.
    text = (
        "Surveil 1, then you draw a card and lose 1 life. "
        "(To surveil 1, look at the top card of your library. "
        "You may put it into your graveyard.)\n"
        "Void — If a nonland permanent left the battlefield this turn "
        "or a spell was warped this turn, draw another card."
    )
    assert _parse_spell_oracle_text(text, "Hymn of the Faller") == (
        Sequence(effects=(
            Surveil(amount=1, use_artifact_count=False),
            DrawCards(amount=1),
            LoseLife(amount=1),
            ConditionalIfVoid(
                then_effect=DrawCards(amount=1),
                else_effect=None,
            ),
        ))
    )
