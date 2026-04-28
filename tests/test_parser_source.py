"""Tests for ``OracleParserCardSource`` and its clause parsers."""

from __future__ import annotations

import pytest

from mtg_bot.cards.base import (
    ActivatedAbility,
    ActivationCost,
    ConditionalIfVoid,
    CreateToken,
    DealDamageAnyTarget,
    DealDamageToTargetCreature,
    DrawCards,
    GainLife,
    PutPlusOneCounterOnTargetCreature,
    Sequence,
    StaticAbility,
    StaticScope,
    TriggeredAbility,
    parse_mana_cost,
)
from mtg_bot.cards.parser_source import (
    OracleParserCardSource,
    _build_creature_printing,
    _build_land_printing,
    _build_spell_printing,
    _normalize_activated_effect_phrase,
    _normalize_creature_effect_phrase,
    _normalize_spell_effect_phrase,
    _parse_activated_ability_line,
    _parse_clause,
    _parse_effect,
    _parse_effect_phrase,
    _parse_keyword_line,
    _parse_spell_effect_phrase,
    _parse_spell_oracle_text,
    _parse_static_ability_line,
    _parse_trigger_phrase,
    _parse_triggered_ability_line,
    _parse_type_line,
    _parse_void_damage_replacement,
    _parse_ward_line,
    _parse_warp_line,
)
from mtg_bot.cards.tokens import TOKEN_PRINTINGS


# ---------- _parse_clause ----------

def test_parse_clause_recognizes_flying():
    parsed = _parse_clause("Flying")
    assert parsed is not None
    assert parsed.keyword == "flying"
    assert parsed.trigger is None


def test_parse_clause_strips_surrounding_whitespace():
    assert _parse_clause("  Flying  ") == _parse_clause("Flying")


def test_parse_clause_recognizes_etb_damage_to_any_target():
    from mtg_bot.effects.predicates import Event, IsSelf
    parsed = _parse_clause(
        "When this creature enters, it deals 3 damage to any target."
    )
    assert parsed is not None
    assert parsed.keyword is None
    assert parsed.trigger == TriggeredAbility(
        event=Event.ENTERS, predicate=IsSelf(),
        effect=DealDamageAnyTarget(amount=3),
    )


def test_parse_clause_etb_damage_extracts_amount():
    parsed = _parse_clause(
        "When this creature enters, it deals 7 damage to any target."
    )
    assert parsed is not None
    assert parsed.trigger is not None
    assert parsed.trigger.effect == DealDamageAnyTarget(amount=7)


def test_parse_clause_returns_none_for_unknown_line():
    assert _parse_clause("Regenerate") is None
    assert _parse_clause("This creature has menace.") is None
    assert _parse_clause("") is None


# ---------- evergreen keyword lines ----------


@pytest.mark.parametrize(
    "line,engine_keyword",
    [
        ("Flying", "flying"),
        ("Trample", "trample"),
        ("Vigilance", "vigilance"),
        ("Haste", "haste"),
        ("Reach", "reach"),
        ("Deathtouch", "deathtouch"),
        ("Lifelink", "lifelink"),
        ("Defender", "defender"),
        ("Flash", "flash"),
        ("Menace", "menace"),
        ("Double strike", "double strike"),
        ("First strike", "first strike"),
        ("Prowess", "prowess"),
    ],
)
def test_parse_clause_recognizes_each_evergreen_keyword(line, engine_keyword):
    parsed = _parse_clause(line)
    assert parsed is not None
    assert parsed.keywords == (engine_keyword,)
    assert parsed.trigger is None


# ---------- unified effect grammar (_parse_effect + normalizers) ----------


def test_parse_effect_unknown_returns_none():
    assert _parse_effect("scry 1") is None
    assert _parse_effect("") is None


def test_parse_effect_recognizes_normalized_phrases():
    """Normalized form has no source-subject prefix, no trailing
    period, and a lowercased first letter."""
    assert _parse_effect("deals 3 damage to any target") == \
        DealDamageAnyTarget(amount=3)
    assert _parse_effect("deals 4 damage to target creature") == \
        DealDamageToTargetCreature(amount=4)
    assert _parse_effect("draw a card") == DrawCards(amount=1)
    assert _parse_effect("draw 2 cards") == DrawCards(amount=2)
    assert _parse_effect("you gain 2 life") == GainLife(amount=2)
    assert _parse_effect("put a +1/+1 counter on target creature") == \
        PutPlusOneCounterOnTargetCreature(amount=1)


def test_normalize_creature_effect_phrase_strips_it_prefix():
    assert _normalize_creature_effect_phrase(
        "it deals 3 damage to any target"
    ) == "deals 3 damage to any target"


def test_normalize_creature_effect_phrase_passthrough_no_it():
    """Effects without an ``it`` source-pronoun (draw, life-gain) pass
    through unchanged after whitespace strip."""
    assert _normalize_creature_effect_phrase("draw a card") == "draw a card"
    assert _normalize_creature_effect_phrase(
        "you gain 2 life"
    ) == "you gain 2 life"


def test_normalize_spell_effect_phrase_strips_name_period_and_lowercases():
    assert _normalize_spell_effect_phrase(
        "Bombard deals 4 damage to target creature.", "Bombard"
    ) == "deals 4 damage to target creature"
    assert _normalize_spell_effect_phrase(
        "Draw a card.", "Honor"
    ) == "draw a card"
    assert _normalize_spell_effect_phrase(
        "Put a +1/+1 counter on target creature.", "Honor"
    ) == "put a +1/+1 counter on target creature"


def test_normalize_spell_effect_phrase_leaves_phrase_when_name_absent():
    """If the card-name prefix isn't present (e.g. clause uses a
    generic shape), only the period is stripped."""
    assert _normalize_spell_effect_phrase(
        "Draw a card.", "Plasma Bolt"
    ) == "draw a card"


def test_unification_creature_and_spell_paths_agree_on_damage():
    """Same semantic effect via both surface contexts produces the same
    ``Effect`` — the whole point of the unified path."""
    via_creature = _parse_effect_phrase("it deals 3 damage to any target")
    via_spell = _parse_spell_effect_phrase(
        "Plasma Bolt deals 3 damage to any target.", "Plasma Bolt"
    )
    assert via_creature == via_spell == DealDamageAnyTarget(amount=3)


def test_unification_creature_and_spell_paths_agree_on_draw():
    via_creature = _parse_effect_phrase("draw a card")
    via_spell = _parse_spell_effect_phrase("Draw a card.", "Honor")
    assert via_creature == via_spell == DrawCards(amount=1)


# ---------- _parse_keyword_line / multi-keyword lines ----------


def test_parse_keyword_line_single_keyword():
    assert _parse_keyword_line("Flying") == ("flying",)


def test_parse_keyword_line_two_keywords():
    assert _parse_keyword_line("Flash, Flying") == ("flash", "flying")


def test_parse_keyword_line_three_keywords_preserves_order():
    assert _parse_keyword_line("Trample, Vigilance, Haste") == (
        "trample",
        "vigilance",
        "haste",
    )


def test_parse_keyword_line_returns_none_when_any_piece_unknown():
    # First piece known, second unknown → whole line fails so the
    # caller can fall back to other clause patterns.
    assert _parse_keyword_line("Flying, Bogus") is None
    assert _parse_keyword_line("Bogus, Flying") is None


def test_parse_keyword_line_returns_none_for_non_keyword_line():
    assert _parse_keyword_line(
        "When this creature enters, it deals 3 damage to any target."
    ) is None


def test_parse_clause_recognizes_multi_keyword_line():
    parsed = _parse_clause("Flash, Flying")
    assert parsed is not None
    assert parsed.keywords == ("flash", "flying")
    assert parsed.trigger is None


def test_parse_clause_keyword_shim_returns_none_for_multi_keyword():
    """The single-``keyword`` back-compat shim should not silently
    return one of several keywords; multi-keyword clauses must be
    consumed via ``keywords``."""
    parsed = _parse_clause("Flash, Flying")
    assert parsed is not None
    assert parsed.keyword is None
    assert parsed.keywords == ("flash", "flying")


def test_build_creature_printing_handles_multi_keyword_line():
    card = {
        "name": "Test Vanilla",
        "mana_cost": "{2}{G}",
        "type_line": "Creature — Beast",
        "power": "3",
        "toughness": "3",
        "oracle_text": "Trample, Vigilance",
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.keywords == frozenset({"trample", "vigilance"})
    assert printing.triggered_abilities == ()


def test_build_creature_printing_handles_each_evergreen_standalone():
    """Smoke-test: every evergreen keyword on its own oracle-text
    line produces the expected engine keyword on the printing."""
    base = {
        "name": "Test Creature",
        "mana_cost": "{1}",
        "type_line": "Creature — Human",
        "power": "1",
        "toughness": "1",
    }
    cases = [
        ("Flying", "flying"),
        ("Trample", "trample"),
        ("Vigilance", "vigilance"),
        ("Haste", "haste"),
        ("Reach", "reach"),
        ("Deathtouch", "deathtouch"),
        ("Lifelink", "lifelink"),
        ("Defender", "defender"),
        ("Flash", "flash"),
        ("Menace", "menace"),
        ("Double strike", "double strike"),
        ("First strike", "first strike"),
        ("Prowess", "prowess"),
    ]
    for line, engine_keyword in cases:
        printing = _build_creature_printing(dict(base, oracle_text=line))
        assert printing is not None, line
        assert printing.keywords == frozenset({engine_keyword}), line


# ---------- compositional triggered-ability parsing ----------

def test_parse_trigger_phrase_recognizes_etb():
    # Phase-2 migration: ``this creature enters`` is now claimed by
    # the predicate-style table (``Event.ENTERS, IsSelf()``) rather
    # than the legacy string-key dict. The string-key lookup returns
    # ``None`` for migrated phrases.
    from mtg_bot.cards.parser.triggers import _parse_predicate_trigger_phrase
    from mtg_bot.effects.predicates import Event, IsSelf
    assert _parse_trigger_phrase("this creature enters") is None
    assert _parse_predicate_trigger_phrase(
        "this creature enters"
    ) == (Event.ENTERS, IsSelf())


def test_parse_trigger_phrase_recognizes_dies():
    # Phase 4 migrated ``this creature dies`` from the legacy string-key
    # lookup to the predicate-style path: it now resolves to
    # ``(Event.DIES, IsSelf())``. The legacy lookup must therefore
    # return ``None`` so a card claiming the phrase doesn't accidentally
    # land on the obsolete string-key code path.
    from mtg_bot.cards.parser.triggers import _parse_predicate_trigger_phrase
    from mtg_bot.effects.predicates import Event, IsSelf
    assert _parse_trigger_phrase("this creature dies") is None
    assert _parse_predicate_trigger_phrase(
        "this creature dies"
    ) == (Event.DIES, IsSelf())


def test_parse_trigger_phrase_recognizes_attacks():
    # Phase 5 migrated ``this creature attacks`` from the legacy
    # string-key lookup to the predicate-style path: it now resolves
    # to ``(Event.ATTACKS, IsSelf())``. The legacy lookup must therefore
    # return ``None`` so a card claiming the phrase doesn't accidentally
    # land on the obsolete string-key code path.
    from mtg_bot.cards.parser.triggers import _parse_predicate_trigger_phrase
    from mtg_bot.effects.predicates import Event, IsSelf
    assert _parse_trigger_phrase("this creature attacks") is None
    assert _parse_predicate_trigger_phrase(
        "this creature attacks"
    ) == (Event.ATTACKS, IsSelf())


def test_parse_trigger_phrase_strips_whitespace():
    # The whitespace-stripping behavior applies to both the legacy
    # and the predicate-style lookups; here we exercise the predicate
    # path because ``this creature enters`` migrated.
    from mtg_bot.cards.parser.triggers import _parse_predicate_trigger_phrase
    from mtg_bot.effects.predicates import Event, IsSelf
    assert _parse_predicate_trigger_phrase(
        "  this creature enters  "
    ) == (Event.ENTERS, IsSelf())


def test_parse_trigger_phrase_returns_none_for_unknown():
    assert _parse_trigger_phrase("you draw a card") is None
    assert _parse_trigger_phrase("") is None


def test_parse_effect_phrase_recognizes_deal_damage_any_target():
    assert _parse_effect_phrase(
        "it deals 2 damage to any target"
    ) == DealDamageAnyTarget(amount=2)


def test_parse_effect_phrase_recognizes_create_tapped_robot_token():
    assert _parse_effect_phrase(
        "create a tapped 2/2 colorless Robot artifact creature token"
    ) == CreateToken(
        token_printing=TOKEN_PRINTINGS["Robot"],
        count=1,
        tapped=True,
    )


def test_parse_effect_phrase_recognizes_draw_a_card():
    assert _parse_effect_phrase("draw a card") == DrawCards(amount=1)


def test_parse_effect_phrase_recognizes_draw_n_cards_plural():
    assert _parse_effect_phrase("draw 2 cards") == DrawCards(amount=2)
    assert _parse_effect_phrase("draw 5 cards") == DrawCards(amount=5)


def test_parse_effect_phrase_recognizes_you_gain_n_life():
    assert _parse_effect_phrase("you gain 2 life") == GainLife(amount=2)
    assert _parse_effect_phrase("you gain 1 life") == GainLife(amount=1)


def test_parse_effect_phrase_returns_none_for_unknown():
    # Unknown shapes that should NOT be confused with the supported
    # singular/plural draw forms or the gain-life form.
    assert _parse_effect_phrase("draw cards") is None
    assert _parse_effect_phrase("you gain life") is None
    assert _parse_effect_phrase("scry 1") is None
    assert _parse_effect_phrase("") is None


def test_parse_triggered_ability_line_full_etb_damage():
    from mtg_bot.effects.predicates import Event, IsSelf
    result = _parse_triggered_ability_line(
        "When this creature enters, it deals 5 damage to any target."
    )
    assert result == TriggeredAbility(
        event=Event.ENTERS, predicate=IsSelf(),
        effect=DealDamageAnyTarget(amount=5),
    )


def test_parse_triggered_ability_line_full_dies_create_robot():
    """Gravpack Monoist's full dies-trigger line. Phase 4 migrated
    ``this creature dies`` to the predicate-style path; the parser now
    yields ``event=Event.DIES, predicate=IsSelf()``."""
    from mtg_bot.effects.predicates import Event, IsSelf
    result = _parse_triggered_ability_line(
        "When this creature dies, "
        "create a tapped 2/2 colorless Robot artifact creature token."
    )
    assert result == TriggeredAbility(
        event=Event.DIES, predicate=IsSelf(),
        effect=CreateToken(
            token_printing=TOKEN_PRINTINGS["Robot"],
            count=1,
            tapped=True,
        ),
    )


def test_parse_triggered_ability_line_accepts_whenever():
    """The line-level pattern matches both ``When`` and ``Whenever``;
    individual condition and effect parsers decide whether the
    surface pieces are recognized."""
    # No "Whenever this creature enters" trigger phrase, but the
    # line-level structural match should still be tried — failure
    # comes from the condition parser, not the line regex.
    assert _parse_triggered_ability_line(
        "Whenever you draw a card, it deals 1 damage to any target."
    ) is None  # condition phrase unknown


def test_parse_triggered_ability_line_rejects_non_trigger_lines():
    assert _parse_triggered_ability_line("Flying") is None
    assert _parse_triggered_ability_line(
        "This creature has menace."
    ) is None


def test_parse_triggered_ability_line_rejects_unknown_effect():
    """Trigger condition parses, effect doesn't → line as a whole
    fails (no fallback)."""
    assert _parse_triggered_ability_line(
        "When this creature enters, scry 1."
    ) is None


def test_parse_triggered_ability_line_full_etb_draw_a_card():
    from mtg_bot.effects.predicates import Event, IsSelf
    result = _parse_triggered_ability_line(
        "When this creature enters, draw a card."
    )
    assert result == TriggeredAbility(
        event=Event.ENTERS, predicate=IsSelf(),
        effect=DrawCards(amount=1),
    )


def test_parse_triggered_ability_line_full_etb_gain_life():
    from mtg_bot.effects.predicates import Event, IsSelf
    result = _parse_triggered_ability_line(
        "When this creature enters, you gain 2 life."
    )
    assert result == TriggeredAbility(
        event=Event.ENTERS, predicate=IsSelf(),
        effect=GainLife(amount=2),
    )


def test_parse_triggered_ability_line_full_attacks_deals_damage():
    """Phase 5: ``this creature attacks`` parses to predicate-style
    ``event=Event.ATTACKS, predicate=IsSelf()``."""
    from mtg_bot.effects.predicates import Event, IsSelf
    result = _parse_triggered_ability_line(
        "Whenever this creature attacks, "
        "it deals 1 damage to any target."
    )
    assert result == TriggeredAbility(
        event=Event.ATTACKS, predicate=IsSelf(),
        effect=DealDamageAnyTarget(amount=1),
    )


def test_parse_triggered_ability_line_full_dies_draw_cards_plural():
    from mtg_bot.effects.predicates import Event, IsSelf
    result = _parse_triggered_ability_line(
        "When this creature dies, draw 2 cards."
    )
    assert result == TriggeredAbility(
        event=Event.DIES, predicate=IsSelf(), effect=DrawCards(amount=2)
    )


# ---------- _parse_type_line ----------

def test_parse_type_line_simple_creature():
    assert _parse_type_line("Creature — Dragon") == (
        (), ("Creature",), ("Dragon",)
    )


def test_parse_type_line_legendary_multi_subtype():
    assert _parse_type_line("Legendary Creature — Human Soldier") == (
        ("Legendary",), ("Creature",), ("Human", "Soldier")
    )


def test_parse_type_line_no_subtypes():
    assert _parse_type_line("Sorcery") == ((), ("Sorcery",), ())


def test_parse_type_line_multi_card_type():
    assert _parse_type_line("Artifact Creature — Robot") == (
        (), ("Artifact", "Creature"), ("Robot",)
    )


def test_parse_type_line_empty():
    assert _parse_type_line("") == ((), (), ())


# ---------- _build_creature_printing ----------

_NEBULA_DRAGON_CARD = {
    "name": "Nebula Dragon",
    "mana_cost": "{6}{R}",
    "type_line": "Creature — Dragon",
    "power": "4",
    "toughness": "4",
    "oracle_text": (
        "Flying\n"
        "When this creature enters, it deals 3 damage to any target."
    ),
}


def test_build_creature_printing_handles_nebula_dragon():
    from mtg_bot.effects.predicates import Event, IsSelf
    printing = _build_creature_printing(_NEBULA_DRAGON_CARD)
    assert printing is not None
    assert printing.name == "Nebula Dragon"
    assert printing.power == 4
    assert printing.toughness == 4
    assert printing.subtypes == ("Dragon",)
    assert printing.keywords == frozenset({"flying"})
    assert printing.triggered_abilities == (
        TriggeredAbility(
            event=Event.ENTERS, predicate=IsSelf(),
            effect=DealDamageAnyTarget(amount=3),
        ),
    )


def test_build_creature_printing_skips_non_creatures():
    sorcery = {
        "name": "Plasma Bolt",
        "mana_cost": "{R}",
        "type_line": "Sorcery",
        "oracle_text": "...",
    }
    assert _build_creature_printing(sorcery) is None


def test_build_creature_printing_skips_unparsed_clause():
    card = dict(
        _NEBULA_DRAGON_CARD,
        # ``Regenerate`` is not in the keyword grammar, so this
        # whole card must be skipped.
        oracle_text="Flying\nRegenerate",
    )
    assert _build_creature_printing(card) is None


def test_build_creature_printing_skips_non_integer_power():
    card = dict(_NEBULA_DRAGON_CARD, power="*")
    assert _build_creature_printing(card) is None


def test_build_creature_printing_handles_empty_oracle_text():
    card = dict(_NEBULA_DRAGON_CARD, oracle_text="")
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.keywords == frozenset()
    assert printing.triggered_abilities == ()


# ---------- OracleParserCardSource ----------

def test_source_claims_nebula_dragon():
    source = OracleParserCardSource()
    assert "Nebula Dragon" in source.claimed_names
    assert "Nebula Dragon" in source.get_printings()


def test_source_claims_gravpack_monoist():
    """Flying + dies-trigger to create a tapped Robot is now in
    the parser's grammar."""
    source = OracleParserCardSource()
    assert "Gravpack Monoist" in source.claimed_names
    assert "Gravpack Monoist" in source.get_printings()


def test_source_skips_cards_with_unsupported_clauses():
    """A card whose oracle text uses any clause beyond what the parser
    supports must not be claimed. Consult the Star Charts has a Kicker
    preamble (unrecognized — only Convoke is on the spell-keyword
    preamble whitelist) AND a multi-clause body whose ``Look at the
    top X cards`` shape isn't a primitive in the unified grammar, so
    the whole spell is skipped."""
    source = OracleParserCardSource()
    assert "Consult the Star Charts" not in source.claimed_names


def test_source_get_printings_returns_only_claimed():
    source = OracleParserCardSource()
    assert set(source.get_printings()) == set(source.claimed_names)


def test_source_build_mono_color_deck_raises():
    with pytest.raises(NotImplementedError):
        OracleParserCardSource().build_mono_color_deck("R", seed=0)


# ---------- _parse_spell_effect_phrase ----------

def test_parse_spell_effect_phrase_recognizes_deal_damage_any_target():
    assert _parse_spell_effect_phrase(
        "Plasma Bolt deals 2 damage to any target.", "Plasma Bolt"
    ) == DealDamageAnyTarget(amount=2)


def test_parse_spell_effect_phrase_recognizes_deal_damage_target_creature():
    assert _parse_spell_effect_phrase(
        "Bombard deals 4 damage to target creature.", "Bombard"
    ) == DealDamageToTargetCreature(amount=4)


def test_parse_spell_effect_phrase_strips_whitespace():
    assert _parse_spell_effect_phrase(
        "  Bombard deals 4 damage to target creature.  ", "Bombard"
    ) == DealDamageToTargetCreature(amount=4)


def test_parse_spell_effect_phrase_returns_none_for_creature_style_it_text():
    """Creature-style 'it deals N damage to any target' has no card-name
    subject, so the spell parser must reject it. Spells use their own
    name as the subject; 'it' belongs to the creature parser."""
    assert _parse_spell_effect_phrase(
        "it deals 3 damage to any target.", "Bombard"
    ) is None


def test_parse_spell_effect_phrase_returns_none_for_unknown_text():
    assert _parse_spell_effect_phrase(
        "Bombard counters target spell.", "Bombard"
    ) is None
    assert _parse_spell_effect_phrase("", "Bombard") is None


def test_parse_spell_effect_phrase_name_substituted_exactly():
    """The card-name token in the regex is the actual card's name —
    a different name in the sentence body must not match."""
    assert _parse_spell_effect_phrase(
        "Bombard deals 4 damage to target creature.", "Plasma Bolt"
    ) is None


# ---------- _build_spell_printing ----------

_BOMBARD_CARD = {
    "name": "Bombard",
    "mana_cost": "{2}{R}",
    "type_line": "Instant",
    "oracle_text": "Bombard deals 4 damage to target creature.",
}


def test_build_spell_printing_handles_bombard():
    printing = _build_spell_printing(_BOMBARD_CARD)
    assert printing is not None
    assert printing.name == "Bombard"
    assert printing.is_instant
    assert printing.spell_effect == DealDamageToTargetCreature(amount=4)


def test_build_spell_printing_handles_sorcery_any_target():
    """Synthetic single-line sorcery — exercises the make_sorcery branch
    and the any-target damage shape together."""
    card = {
        "name": "Test Bolt",
        "mana_cost": "{R}",
        "type_line": "Sorcery",
        "oracle_text": "Test Bolt deals 3 damage to any target.",
    }
    printing = _build_spell_printing(card)
    assert printing is not None
    assert printing.is_sorcery
    assert printing.spell_effect == DealDamageAnyTarget(amount=3)


def test_build_spell_printing_skips_creatures():
    creature = {
        "name": "Nebula Dragon",
        "mana_cost": "{6}{R}",
        "type_line": "Creature — Dragon",
        "power": "4",
        "toughness": "4",
        "oracle_text": "Flying",
    }
    assert _build_spell_printing(creature) is None


def test_build_spell_printing_skips_unsupported_effect_text():
    # Use a phrase the unified effect grammar deliberately doesn't claim
    # (target self-card return — no engine class for it). "Counter
    # target spell" used to live here but is now claimed by batch-7's
    # CounterTargetSpell handler.
    card = dict(
        _BOMBARD_CARD,
        oracle_text="Return Bombard from your graveyard to your hand.",
    )
    assert _build_spell_printing(card) is None


def test_build_spell_printing_skips_empty_oracle_text():
    card = dict(_BOMBARD_CARD, oracle_text="")
    assert _build_spell_printing(card) is None


# ---------- OracleParserCardSource (spells) ----------

def test_source_claims_bombard():
    source = OracleParserCardSource()
    assert "Bombard" in source.claimed_names
    assert "Bombard" in source.get_printings()


def test_source_claims_plasma_bolt():
    """Plasma Bolt's two-clause Void rider parses to a
    ``ConditionalIfVoid`` switching the damage amount."""
    source = OracleParserCardSource()
    assert "Plasma Bolt" in source.claimed_names
    assert "Plasma Bolt" in source.get_printings()


# ---------- _build_land_printing ----------

_PLAINS_CARD = {
    "name": "Plains",
    "mana_cost": "",
    "type_line": "Basic Land — Plains",
    "oracle_text": "({T}: Add {W}.)",
}


@pytest.mark.parametrize(
    "name,subtype,reminder,produced",
    [
        ("Plains", "Plains", "({T}: Add {W}.)", "W"),
        ("Island", "Island", "({T}: Add {U}.)", "U"),
        ("Swamp", "Swamp", "({T}: Add {B}.)", "B"),
        ("Mountain", "Mountain", "({T}: Add {R}.)", "R"),
        ("Forest", "Forest", "({T}: Add {G}.)", "G"),
    ],
)
def test_build_land_printing_handles_each_basic(
    name, subtype, reminder, produced
):
    card = {
        "name": name,
        "mana_cost": "",
        "type_line": f"Basic Land — {subtype}",
        "oracle_text": reminder,
    }
    printing = _build_land_printing(card)
    assert printing is not None
    assert printing.name == name
    assert printing.is_basic_land
    assert printing.is_land
    assert printing.supertypes == ("Basic",)
    assert printing.card_types == ("Land",)
    assert printing.subtypes == (subtype,)
    assert printing.produces == frozenset({produced})
    # Basics are simple: no triggers, no activations, no enters-tapped.
    assert printing.triggered_abilities == ()
    assert printing.activated_abilities == ()
    assert printing.enters_tapped is False


def test_build_land_printing_skips_non_lands():
    creature = {
        "name": "Nebula Dragon",
        "mana_cost": "{6}{R}",
        "type_line": "Creature — Dragon",
        "power": "4",
        "toughness": "4",
        "oracle_text": "Flying",
    }
    assert _build_land_printing(creature) is None


def test_build_land_printing_skips_planet_lands():
    """Planet lands have ETB-tapped, mana-tap, Station, and a tier-gated
    activated ability — none of which the slice grammar yet models. The
    builder must skip silently rather than partially claim."""
    card = {
        "name": "Adagia, Windswept Bastion",
        "mana_cost": "",
        "type_line": "Land — Planet",
        "oracle_text": (
            "This land enters tapped.\n"
            "{T}: Add {W}.\n"
            "Station (Tap another creature you control: Put charge "
            "counters equal to its power on this Planet. Station only "
            "as a sorcery.)\n"
            "12+ | {3}{W}, {T}: Create a token that's a copy of target "
            "artifact or enchantment you control, except it's legendary. "
            "Activate only as a sorcery."
        ),
    }
    assert _build_land_printing(card) is None


@pytest.mark.parametrize(
    "name,subtypes_str,reminder,colors",
    [
        (
            "Breeding Pool", "Forest Island",
            "({T}: Add {G} or {U}.)", ("G", "U"),
        ),
        (
            "Godless Shrine", "Plains Swamp",
            "({T}: Add {W} or {B}.)", ("W", "B"),
        ),
        (
            "Sacred Foundry", "Mountain Plains",
            "({T}: Add {R} or {W}.)", ("R", "W"),
        ),
        (
            "Stomping Ground", "Mountain Forest",
            "({T}: Add {R} or {G}.)", ("R", "G"),
        ),
        (
            "Watery Grave", "Island Swamp",
            "({T}: Add {U} or {B}.)", ("U", "B"),
        ),
    ],
)
def test_build_land_printing_handles_each_shock_land(
    name, subtypes_str, reminder, colors
):
    """Each shock land's two-line oracle text parses to the expected
    shape: dual-mana producer, ETB-pay-2-life-or-tapped rider, no
    Basic supertype, subtypes lifted off the type line."""
    card = {
        "name": name,
        "mana_cost": "",
        "type_line": f"Land — {subtypes_str}",
        "oracle_text": (
            f"{reminder}\n"
            "As this land enters, you may pay 2 life. "
            "If you don't, it enters tapped."
        ),
    }
    printing = _build_land_printing(card)
    assert printing is not None
    assert printing.name == name
    assert printing.is_land
    assert printing.is_basic_land is False
    assert printing.supertypes == ()
    assert printing.card_types == ("Land",)
    assert printing.subtypes == tuple(subtypes_str.split())
    assert printing.produces == frozenset(colors)
    assert printing.shock_land_life_cost == 2
    # No granted triggers/activations beyond the engine's built-in
    # shock-land ETB handling.
    assert printing.triggered_abilities == ()
    assert printing.activated_abilities == ()


def test_build_land_printing_skips_shock_land_with_extra_line():
    """A shock-shaped card with a third line (hypothetical printed
    reminder or rider variation) must not partially claim — fall through."""
    card = {
        "name": "Hypothetical Shock",
        "mana_cost": "",
        "type_line": "Land — Forest Island",
        "oracle_text": (
            "({T}: Add {G} or {U}.)\n"
            "As this land enters, you may pay 2 life. "
            "If you don't, it enters tapped.\n"
            "When this land enters, draw a card."
        ),
    }
    assert _build_land_printing(card) is None


def test_build_land_printing_skips_shock_with_wrong_life_cost():
    """The fixed 2-life rider is what we recognize; a 3-life variant
    (hypothetical) falls through rather than silently parsing wrong."""
    card = {
        "name": "Hypothetical Shock",
        "mana_cost": "",
        "type_line": "Land — Forest Island",
        "oracle_text": (
            "({T}: Add {G} or {U}.)\n"
            "As this land enters, you may pay 3 life. "
            "If you don't, it enters tapped."
        ),
    }
    assert _build_land_printing(card) is None


def test_build_land_printing_skips_shock_with_missing_rider():
    """Just the dual-mana reminder line, no ETB rider, isn't a shock —
    skip rather than build a shock with shock_land_life_cost=2."""
    card = {
        "name": "Hypothetical Dual",
        "mana_cost": "",
        "type_line": "Land — Forest Island",
        "oracle_text": "({T}: Add {G} or {U}.)",
    }
    assert _build_land_printing(card) is None


def test_build_land_printing_skips_shock_dual_with_no_subtypes():
    """A non-basic land with the shock oracle shape but no subtypes on
    the type line has no ``subtypes_str`` to feed the factory; skip."""
    card = {
        "name": "Hypothetical Subtype-less",
        "mana_cost": "",
        "type_line": "Land",
        "oracle_text": (
            "({T}: Add {G} or {U}.)\n"
            "As this land enters, you may pay 2 life. "
            "If you don't, it enters tapped."
        ),
    }
    assert _build_land_printing(card) is None


def test_build_land_printing_skips_utility_lands():
    """Plain non-basic lands (no Basic supertype) without recognizable
    basic-reminder text are skipped."""
    card = {
        "name": "Command Bridge",
        "mana_cost": "",
        "type_line": "Land",
        "oracle_text": (
            "This land enters tapped.\n"
            "When this land enters, sacrifice it unless you tap an "
            "untapped permanent you control.\n"
            "{T}: Add one mana of any color."
        ),
    }
    assert _build_land_printing(card) is None


def test_build_land_printing_skips_basic_with_unrecognized_oracle_text():
    """A card flagged Basic Land whose oracle text doesn't match the
    reminder shape (hypothetical snow basic, etc.) must be skipped."""
    card = dict(_PLAINS_CARD, oracle_text="Some unsupported ability.")
    assert _build_land_printing(card) is None


def test_build_land_printing_skips_empty_oracle_text():
    """Defensive: a basic with no oracle text at all has no color signal,
    so the builder skips."""
    card = dict(_PLAINS_CARD, oracle_text="")
    assert _build_land_printing(card) is None


# ---------- OracleParserCardSource (lands) ----------


@pytest.mark.parametrize(
    "name", ["Plains", "Island", "Swamp", "Mountain", "Forest"]
)
def test_source_claims_each_basic_land(name):
    source = OracleParserCardSource()
    assert name in source.claimed_names
    assert name in source.get_printings()


def test_source_does_not_claim_planet_land():
    """Planet lands are out-of-slice; the source must not claim them."""
    source = OracleParserCardSource()
    assert "Adagia, Windswept Bastion" not in source.claimed_names


@pytest.mark.parametrize(
    "name",
    [
        "Breeding Pool", "Godless Shrine", "Sacred Foundry",
        "Stomping Ground", "Watery Grave",
    ],
)
def test_source_claims_each_shock_land(name):
    """Each EoE shock land gets claimed end-to-end from the cached
    Scryfall data."""
    source = OracleParserCardSource()
    assert name in source.claimed_names
    assert name in source.get_printings()


def test_source_does_not_claim_command_bridge():
    """Command Bridge needs an ETB sac-self trigger and an any-color
    mana ability that the parser doesn't yet wire end-to-end."""
    source = OracleParserCardSource()
    assert "Command Bridge" not in source.claimed_names


# ---------- Warp ----------

@pytest.mark.parametrize(
    "line,expected_cost",
    [
        ("Warp {3}", "{3}"),
        ("Warp {R}", "{R}"),
        ("Warp {1}{R}", "{1}{R}"),
        ("Warp {2}{U}{U}", "{2}{U}{U}"),
        ("Warp {X}{G}", "{X}{G}"),
        ("Warp {U/R}", "{U/R}"),
    ],
)
def test_parse_warp_line_simple_costs(line, expected_cost):
    """Bare ``Warp {cost}`` lines (no reminder text) extract the cost."""
    assert _parse_warp_line(line) == expected_cost


def test_parse_warp_line_with_reminder_text():
    """The printed line includes parenthesized reminder text — the
    recognizer ignores it and just returns the cost."""
    line = (
        "Warp {1}{R} (You may cast this card from your hand for its "
        "warp cost. Exile this creature at the beginning of the next "
        "end step, then you may cast it from exile on a later turn.)"
    )
    assert _parse_warp_line(line) == "{1}{R}"


def test_parse_warp_line_strips_whitespace():
    assert _parse_warp_line("  Warp {2}{G}  ") == "{2}{G}"


def test_parse_warp_line_rejects_pay_life_rider():
    """``Warp—{B}, Pay 2 life.`` (em-dash + additional cost rider) is
    out of scope this batch and must not match — the card should fall
    through and be skipped, not silently parsed as plain Warp."""
    rider = (
        "Warp—{B}, Pay 2 life. (You may cast this card from your "
        "hand or graveyard for its warp cost. If you do, exile this "
        "creature at the beginning of the next end step, then you may "
        "cast it from exile on a later turn.)"
    )
    assert _parse_warp_line(rider) is None


def test_parse_warp_line_rejects_em_dash_form():
    """Even the em-dash separator alone (no rider) is intentionally
    not matched — actual EoE simple-Warp printings use a space."""
    assert _parse_warp_line("Warp—{B}") is None


def test_parse_warp_line_rejects_non_warp_lines():
    assert _parse_warp_line("Flying") is None
    assert _parse_warp_line("Warp") is None
    assert _parse_warp_line("Warp {1}{R} extra junk") is None
    assert _parse_warp_line("") is None


def test_parse_warp_line_rejects_malformed_cost():
    """Cost segment must be one or more brace-wrapped symbols."""
    assert _parse_warp_line("Warp 3") is None
    assert _parse_warp_line("Warp {}") is None


def test_parse_clause_does_not_claim_warp_line():
    """Warp is handled by its own recognizer in
    ``_build_creature_printing``, not by ``_parse_clause``. A warp
    line passed directly to ``_parse_clause`` must return ``None``
    so the clause grammar stays orthogonal."""
    assert _parse_clause("Warp {1}{R}") is None


def test_build_creature_printing_handles_simple_warp():
    """Synthetic vanilla creature with only a Warp clause — exercises
    the warp passthrough without depending on any other clause."""
    card = {
        "name": "Test Warpling",
        "mana_cost": "{4}",
        "type_line": "Creature — Beast",
        "power": "3",
        "toughness": "3",
        "oracle_text": (
            "Warp {1}{G} (You may cast this card from your hand for "
            "its warp cost. Exile this creature at the beginning of "
            "the next end step, then you may cast it from exile on a "
            "later turn.)"
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.warp_cost_str == "{1}{G}"
    assert printing.warp_cost is not None


def test_build_creature_printing_handles_warp_with_keyword():
    """Red Tiger Mechan's exact shape: Haste + Warp."""
    card = {
        "name": "Red Tiger Mechan",
        "mana_cost": "{3}{R}",
        "type_line": "Artifact Creature — Robot Cat",
        "power": "3",
        "toughness": "3",
        "oracle_text": (
            "Haste\n"
            "Warp {1}{R} (You may cast this card from your hand for "
            "its warp cost. Exile this creature at the beginning of "
            "the next end step, then you may cast it from exile on a "
            "later turn.)"
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.keywords == frozenset({"haste"})
    assert printing.warp_cost_str == "{1}{R}"
    assert "Robot" in printing.subtypes
    assert "Cat" in printing.subtypes
    assert "Artifact" in printing.card_types


def test_build_creature_printing_skips_warp_with_pay_life_rider():
    """A creature whose Warp clause has a rider (additional life cost)
    falls through and the whole card must be skipped, not silently
    parsed as plain Warp."""
    card = {
        "name": "Test Rider",
        "mana_cost": "{B}",
        "type_line": "Creature — Zombie",
        "power": "1",
        "toughness": "1",
        "oracle_text": (
            "Warp—{B}, Pay 2 life. (You may cast this card from "
            "your hand or graveyard for its warp cost.)"
        ),
    }
    assert _build_creature_printing(card) is None


def test_source_claims_red_tiger_mechan():
    """Haste + Warp is now claimable."""
    source = OracleParserCardSource()
    assert "Red Tiger Mechan" in source.claimed_names


def test_source_claims_germinating_wurm():
    """ETB gain-2-life + Warp is now claimable."""
    source = OracleParserCardSource()
    assert "Germinating Wurm" in source.claimed_names


def test_source_claims_bygone_colossus():
    """Bare Warp-only oracle text is now claimable."""
    source = OracleParserCardSource()
    assert "Bygone Colossus" in source.claimed_names


def test_source_does_not_claim_timeline_culler():
    """Timeline Culler has the ``Warp—{B}, Pay 2 life.`` rider that's
    out of scope this batch — the card must remain unclaimed."""
    source = OracleParserCardSource()
    assert "Timeline Culler" not in source.claimed_names


# ---------- Ward ----------

@pytest.mark.parametrize(
    "line,expected_cost",
    [
        ("Ward {1}", "{1}"),
        ("Ward {2}", "{2}"),
        ("Ward {3}", "{3}"),
        ("Ward {R}", "{R}"),
        ("Ward {1}{R}", "{1}{R}"),
        ("Ward {2}{U}{U}", "{2}{U}{U}"),
        ("Ward {X}{G}", "{X}{G}"),
        ("Ward {U/R}", "{U/R}"),
    ],
)
def test_parse_ward_line_simple_costs(line, expected_cost):
    """Bare ``Ward {cost}`` lines (no reminder text) extract the cost."""
    assert _parse_ward_line(line) == expected_cost


def test_parse_ward_line_with_reminder_text():
    """The printed line includes parenthesized reminder text — the
    recognizer ignores it and just returns the cost."""
    line = (
        "Ward {2} (Whenever this creature becomes the target of a spell "
        "or ability an opponent controls, counter it unless that player "
        "pays {2}.)"
    )
    assert _parse_ward_line(line) == "{2}"


def test_parse_ward_line_strips_whitespace():
    assert _parse_ward_line("  Ward {2}  ") == "{2}"


def test_parse_ward_line_rejects_pay_life_rider():
    """``Ward — Pay 2 life.`` (em-dash + non-mana cost) is out of scope
    this batch and must not match — the card should fall through and
    be skipped, not silently parsed as plain Ward."""
    assert _parse_ward_line("Ward — Pay 2 life.") is None


def test_parse_ward_line_rejects_discard_random_rider():
    """``Ward — Discard a card at random.`` (em-dash + non-mana cost)
    is out of scope this batch — Alpharael, Stonechosen has this
    shape and must remain unclaimed."""
    assert _parse_ward_line("Ward—Discard a card at random.") is None


def test_parse_ward_line_rejects_discard_chosen_rider():
    """``Ward — Discard a card.`` (em-dash + non-mana cost) is out of
    scope this batch — Sunset Saboteur has this shape and must remain
    unclaimed."""
    assert _parse_ward_line("Ward—Discard a card.") is None


def test_parse_ward_line_rejects_em_dash_form():
    """Even the em-dash separator alone (no rider) is intentionally
    not matched — actual EoE simple-Ward printings use a space."""
    assert _parse_ward_line("Ward—{B}") is None


def test_parse_ward_line_rejects_non_ward_lines():
    assert _parse_ward_line("Flying") is None
    assert _parse_ward_line("Ward") is None
    assert _parse_ward_line("Ward {2} extra junk") is None
    assert _parse_ward_line("") is None


def test_parse_ward_line_rejects_malformed_cost():
    """Cost segment must be one or more brace-wrapped symbols."""
    assert _parse_ward_line("Ward 2") is None
    assert _parse_ward_line("Ward {}") is None


def test_parse_clause_does_not_claim_ward_line():
    """Ward is handled by its own recognizer in
    ``_build_creature_printing``, not by ``_parse_clause``. A ward
    line passed directly to ``_parse_clause`` must return ``None``
    so the clause grammar stays orthogonal."""
    assert _parse_clause("Ward {2}") is None


def test_build_creature_printing_handles_simple_ward():
    """Synthetic vanilla creature with only a Ward clause — exercises
    the ward passthrough without depending on any other clause."""
    card = {
        "name": "Test Wardling",
        "mana_cost": "{4}",
        "type_line": "Creature — Beast",
        "power": "3",
        "toughness": "3",
        "oracle_text": (
            "Ward {2} (Whenever this creature becomes the target of a "
            "spell or ability an opponent controls, counter it unless "
            "that player pays {2}.)"
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.ward_cost_str == "{2}"
    assert printing.ward_cost is not None


def test_build_creature_printing_handles_ward_with_keyword():
    """Synthetic creature with both Flying and Ward — both clauses
    contribute and the ward cost is plumbed through."""
    card = {
        "name": "Test Ward Flyer",
        "mana_cost": "{3}{U}",
        "type_line": "Creature — Spirit",
        "power": "2",
        "toughness": "3",
        "oracle_text": (
            "Flying\n"
            "Ward {1} (Whenever this creature becomes the target of a "
            "spell or ability an opponent controls, counter it unless "
            "that player pays {1}.)"
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.keywords == frozenset({"flying"})
    assert printing.ward_cost_str == "{1}"
    assert printing.ward_cost is not None


def test_build_creature_printing_skips_ward_with_pay_life_rider():
    """A creature whose Ward clause has an em-dash life-cost rider
    falls through and the whole card must be skipped, not silently
    parsed as plain Ward."""
    card = {
        "name": "Test Ward Rider",
        "mana_cost": "{B}",
        "type_line": "Creature — Zombie",
        "power": "1",
        "toughness": "1",
        "oracle_text": "Ward — Pay 2 life.",
    }
    assert _build_creature_printing(card) is None


def test_build_creature_printing_skips_ward_with_discard_rider():
    """``Ward—Discard a card.`` (Sunset Saboteur's shape) is out of
    scope and the creature must not be claimed."""
    card = {
        "name": "Test Discard Ward",
        "mana_cost": "{1}{B}",
        "type_line": "Creature — Human Rogue",
        "power": "4",
        "toughness": "1",
        "oracle_text": "Ward—Discard a card.",
    }
    assert _build_creature_printing(card) is None


def test_source_does_not_claim_alpharael_stonechosen():
    """Alpharael's ``Ward—Discard a card at random.`` rider is out of
    scope this batch — the card must remain unclaimed."""
    source = OracleParserCardSource()
    assert "Alpharael, Stonechosen" not in source.claimed_names


def test_source_does_not_claim_sunset_saboteur():
    """Sunset Saboteur's ``Ward—Discard a card.`` rider is out of
    scope this batch — the card must remain unclaimed."""
    source = OracleParserCardSource()
    assert "Sunset Saboteur" not in source.claimed_names


def test_source_does_not_claim_harmonious_grovestrider():
    """Harmonious Grovestrider has simple ``Ward {2}`` but its
    power/toughness are ``*/*`` (lands-you-control), so the parser
    rejects it at the integer-power check."""
    source = OracleParserCardSource()
    assert "Harmonious Grovestrider" not in source.claimed_names


def test_source_does_not_claim_famished_worldsire():
    """Famished Worldsire has simple ``Ward {3}`` but also Devour,
    which is out of scope — the card must remain unclaimed."""
    source = OracleParserCardSource()
    assert "Famished Worldsire" not in source.claimed_names


def test_source_claims_honor():
    """Honor's two-clause oracle text parses to a ``Sequence`` of
    counter-on-target-creature followed by draw-a-card."""
    source = OracleParserCardSource()
    assert "Honor" in source.claimed_names
    assert "Honor" in source.get_printings()


# ---------- _parse_spell_effect_phrase: generic shapes ----------

def test_parse_spell_effect_phrase_recognizes_draw_a_card_sentence():
    """The spell-side ``Draw a card.`` sentence (capital D, period)
    parses to ``DrawCards(amount=1)``. Distinct from the creature-side
    lowercase ``draw a card`` clause used inside trigger payloads."""
    assert _parse_spell_effect_phrase(
        "Draw a card.", "Honor"
    ) == DrawCards(amount=1)


def test_parse_spell_effect_phrase_recognizes_put_plus_one_counter():
    assert _parse_spell_effect_phrase(
        "Put a +1/+1 counter on target creature.", "Honor"
    ) == PutPlusOneCounterOnTargetCreature(amount=1)


# ---------- _parse_void_damage_replacement ----------

def test_parse_void_damage_replacement_recognizes_plasma_bolt_rider():
    base = DealDamageAnyTarget(amount=2)
    result = _parse_void_damage_replacement(
        "Void — Plasma Bolt deals 3 damage instead "
        "if a nonland permanent left the battlefield this turn "
        "or a spell was warped this turn.",
        "Plasma Bolt",
        base,
    )
    assert result == DealDamageAnyTarget(amount=3)


def test_parse_void_damage_replacement_returns_none_for_non_void_clause():
    base = DealDamageAnyTarget(amount=2)
    assert _parse_void_damage_replacement(
        "Plasma Bolt deals 3 damage to any target.",
        "Plasma Bolt",
        base,
    ) is None


def test_parse_void_damage_replacement_returns_none_for_unsupported_base():
    """Void-damage-replacement only pairs with ``DealDamageAnyTarget``
    today. A non-damage base effect can't carry a damage-amount Void
    rider, so the parse fails and the spell falls through to the
    Sequence path (which will also reject the unknown clause)."""
    base = DrawCards(amount=1)
    assert _parse_void_damage_replacement(
        "Void — Plasma Bolt deals 3 damage instead "
        "if a nonland permanent left the battlefield this turn "
        "or a spell was warped this turn.",
        "Plasma Bolt",
        base,
    ) is None


def test_parse_void_damage_replacement_requires_full_tail():
    """The Void rider regex is anchored to the full Plasma-Bolt-shape
    tail; truncated versions don't match."""
    base = DealDamageAnyTarget(amount=2)
    assert _parse_void_damage_replacement(
        "Void — Plasma Bolt deals 3 damage instead.",
        "Plasma Bolt",
        base,
    ) is None


# ---------- _parse_spell_oracle_text (multi-line spells) ----------

def test_parse_spell_oracle_text_single_line_passes_through():
    """A single-line oracle text returns the same effect as the
    underlying single-sentence parser."""
    assert _parse_spell_oracle_text(
        "Bombard deals 4 damage to target creature.", "Bombard"
    ) == DealDamageToTargetCreature(amount=4)


def test_parse_spell_oracle_text_plasma_bolt_void_conditional():
    """Plasma Bolt's exact Scryfall oracle text parses to a
    ``ConditionalIfVoid`` switching the damage amount."""
    text = (
        "Plasma Bolt deals 2 damage to any target.\n"
        "Void — Plasma Bolt deals 3 damage instead "
        "if a nonland permanent left the battlefield this turn "
        "or a spell was warped this turn."
    )
    assert _parse_spell_oracle_text(text, "Plasma Bolt") == ConditionalIfVoid(
        then_effect=DealDamageAnyTarget(amount=3),
        else_effect=DealDamageAnyTarget(amount=2),
    )


def test_parse_spell_oracle_text_honor_sequence():
    """Honor's exact Scryfall oracle text parses to a two-effect
    ``Sequence``."""
    text = "Put a +1/+1 counter on target creature.\nDraw a card."
    assert _parse_spell_oracle_text(text, "Honor") == Sequence(
        effects=(
            PutPlusOneCounterOnTargetCreature(amount=1),
            DrawCards(amount=1),
        )
    )


def test_parse_spell_oracle_text_skips_when_any_clause_unparseable():
    """If even one clause fails to parse, the whole spell is skipped."""
    # First clause OK, second clause unknown shape.
    assert _parse_spell_oracle_text(
        "Put a +1/+1 counter on target creature.\nScry 1.", "Honor"
    ) is None
    # First clause unknown, second clause OK.
    assert _parse_spell_oracle_text(
        "Scry 1.\nDraw a card.", "Honor"
    ) is None


def test_parse_spell_oracle_text_skips_truncated_void_rider_falls_to_sequence():
    """A 'Void — ... instead.' rider missing the full tail isn't a
    valid Void replacement, and isn't a plain spell-effect sentence
    either — so the Sequence fallback also fails and the whole spell
    is skipped."""
    text = (
        "Plasma Bolt deals 2 damage to any target.\n"
        "Void — Plasma Bolt deals 3 damage instead."
    )
    assert _parse_spell_oracle_text(text, "Plasma Bolt") is None


def test_parse_spell_oracle_text_empty_returns_none():
    assert _parse_spell_oracle_text("", "Bombard") is None
    # Whitespace-only clauses are filtered out — net result is empty.
    assert _parse_spell_oracle_text("\n\n", "Bombard") is None


def test_parse_spell_oracle_text_blank_lines_between_clauses_ignored():
    """Blank lines between real clauses don't change the effective
    clause list — the Honor pattern still parses cleanly with an
    extra blank line."""
    text = "Put a +1/+1 counter on target creature.\n\nDraw a card."
    assert _parse_spell_oracle_text(text, "Honor") == Sequence(
        effects=(
            PutPlusOneCounterOnTargetCreature(amount=1),
            DrawCards(amount=1),
        )
    )


# ---------- _build_spell_printing (multi-line) ----------

def test_build_spell_printing_handles_plasma_bolt():
    card = {
        "name": "Plasma Bolt",
        "mana_cost": "{R}",
        "type_line": "Sorcery",
        "oracle_text": (
            "Plasma Bolt deals 2 damage to any target.\n"
            "Void — Plasma Bolt deals 3 damage instead "
            "if a nonland permanent left the battlefield this turn "
            "or a spell was warped this turn."
        ),
    }
    printing = _build_spell_printing(card)
    assert printing is not None
    assert printing.is_sorcery
    assert printing.spell_effect == ConditionalIfVoid(
        then_effect=DealDamageAnyTarget(amount=3),
        else_effect=DealDamageAnyTarget(amount=2),
    )


def test_build_spell_printing_handles_honor():
    card = {
        "name": "Honor",
        "mana_cost": "{W}",
        "type_line": "Sorcery",
        "oracle_text": "Put a +1/+1 counter on target creature.\nDraw a card.",
    }
    printing = _build_spell_printing(card)
    assert printing is not None
    assert printing.is_sorcery
    assert printing.spell_effect == Sequence(
        effects=(
            PutPlusOneCounterOnTargetCreature(amount=1),
            DrawCards(amount=1),
        )
    )


def test_build_spell_printing_skips_multi_line_with_unparseable_clause():
    """A multi-line spell whose first clause is fine but second is an
    unsupported shape is skipped — the all-or-nothing rule from the
    creature builder applies to spells too. Consult the Star Charts'
    Kicker preamble would be dropped if it were on the recognized
    spell-keyword whitelist (it isn't, only Convoke is), and even
    if it were, the body's ``Look at the top X cards`` shape has no
    primitive in the unified grammar, so the spell is skipped."""
    card = {
        "name": "Consult the Star Charts",
        "mana_cost": "{1}{U}",
        "type_line": "Instant",
        "oracle_text": (
            "Kicker {1}{U} (You may pay an additional {1}{U} as you "
            "cast this spell.)\n"
            "Look at the top X cards of your library, where X is the "
            "number of lands you control. Put one of those cards into "
            "your hand. If this spell was kicked, put two of those "
            "cards into your hand instead. Put the rest on the "
            "bottom of your library in a random order."
        ),
    }
    assert _build_spell_printing(card) is None


# ---------- activated abilities ----------

def test_normalize_activated_effect_phrase_strips_period_and_lowercases():
    assert _normalize_activated_effect_phrase(
        "Put a +1/+1 counter on target creature."
    ) == "put a +1/+1 counter on target creature"
    assert _normalize_activated_effect_phrase(
        "Draw a card."
    ) == "draw a card"


def test_normalize_activated_effect_phrase_strips_whitespace():
    assert _normalize_activated_effect_phrase(
        "  Draw a card.  "
    ) == "draw a card"


def test_normalize_activated_effect_phrase_no_period_passthrough():
    """If the surface phrase already lacks a trailing period, just the
    lowercase + whitespace strip applies."""
    assert _normalize_activated_effect_phrase("Draw a card") == "draw a card"


def test_parse_activated_ability_line_simple_draw():
    result = _parse_activated_ability_line("{2}: Draw a card.")
    assert result == ActivatedAbility(
        cost=ActivationCost(mana=parse_mana_cost("{2}"), mana_str="{2}"),
        effect=DrawCards(amount=1),
    )


def test_parse_activated_ability_line_simple_gain_life():
    result = _parse_activated_ability_line("{1}{W}: You gain 2 life.")
    assert result == ActivatedAbility(
        cost=ActivationCost(
            mana=parse_mana_cost("{1}{W}"), mana_str="{1}{W}"
        ),
        effect=GainLife(amount=2),
    )


def test_parse_activated_ability_line_put_counter_on_target():
    result = _parse_activated_ability_line(
        "{3}: Put a +1/+1 counter on target creature."
    )
    assert result == ActivatedAbility(
        cost=ActivationCost(mana=parse_mana_cost("{3}"), mana_str="{3}"),
        effect=PutPlusOneCounterOnTargetCreature(amount=1),
    )


def test_parse_activated_ability_line_damage_to_any_target():
    result = _parse_activated_ability_line(
        "{1}{R}: Deals 2 damage to any target."
    )
    assert result == ActivatedAbility(
        cost=ActivationCost(
            mana=parse_mana_cost("{1}{R}"), mana_str="{1}{R}"
        ),
        effect=DealDamageAnyTarget(amount=2),
    )


def test_parse_activated_ability_line_strips_whitespace():
    assert _parse_activated_ability_line(
        "  {2}: Draw a card.  "
    ) == _parse_activated_ability_line("{2}: Draw a card.")


def test_parse_activated_ability_line_accepts_tap_only_cost():
    """``{T}: <effect>`` is the pure tap-cost shape — sets
    ``tap_self`` on the ``ActivationCost`` with no mana piece."""
    result = _parse_activated_ability_line("{T}: Draw a card.")
    assert result == ActivatedAbility(
        cost=ActivationCost(tap_self=True),
        effect=DrawCards(amount=1),
    )


def test_parse_activated_ability_line_accepts_tap_plus_mana_cost():
    """``{T}, {mana}: <effect>`` is the tap+mana shape — sets both
    ``tap_self`` and the mana cost on the ``ActivationCost``."""
    result = _parse_activated_ability_line("{2}, {T}: Draw a card.")
    assert result == ActivatedAbility(
        cost=ActivationCost(
            mana=parse_mana_cost("{2}"),
            mana_str="{2}",
            tap_self=True,
        ),
        effect=DrawCards(amount=1),
    )


def test_parse_activated_ability_line_accepts_tap_plus_mana_either_order():
    """``{T}, {mana}`` and ``{mana}, {T}`` produce the same cost — the
    parser is order-insensitive for the tap+mana shape."""
    a = _parse_activated_ability_line("{T}, {1}{R}: Deals 2 damage to any target.")
    b = _parse_activated_ability_line("{1}{R}, {T}: Deals 2 damage to any target.")
    assert a is not None
    assert a == b
    assert a.cost.tap_self is True
    assert a.cost.mana_str == "{1}{R}"


def test_parse_activated_ability_line_accepts_mana_plus_sacrifice():
    """``{2}, Sacrifice this creature: ...`` — mana plus sacrifice-self.
    Now supported (Illvoi Galeblade's activation shape). The new tests
    in ``test_parser_activated_sacrifice.py`` cover this branch in
    depth; this case stays here as a smoke test alongside the other
    cost shapes."""
    ab = _parse_activated_ability_line(
        "{2}, Sacrifice this creature: Draw a card."
    )
    assert ab is not None
    assert ab.cost == ActivationCost(
        mana=parse_mana_cost("{2}"),
        mana_str="{2}",
        sacrifice_self=True,
    )
    assert ab.effect == DrawCards(amount=1)


def test_parse_activated_ability_line_rejects_unknown_effect():
    """Cost parses, effect doesn't → whole line fails."""
    assert _parse_activated_ability_line("{2}: Scry 1.") is None


def test_parse_activated_ability_line_rejects_non_activated_lines():
    assert _parse_activated_ability_line("Flying") is None
    assert _parse_activated_ability_line("") is None
    assert _parse_activated_ability_line(
        "When this creature enters, draw a card."
    ) is None


def test_parse_activated_ability_line_rejects_missing_period():
    """The trailing period is part of the line shape — without it the
    line doesn't match the pattern."""
    assert _parse_activated_ability_line("{2}: Draw a card") is None


def test_parse_activated_ability_line_accepts_sorcery_rider_inline():
    """The inline ``Activate only as a sorcery.`` rider is the shape EoE
    actually uses — it's stripped from the end of the line, the rest is
    parsed as a normal activation, and the resulting ability is marked
    ``sorcery_speed=True``. The standalone-line rider variant is still
    handled separately by the creature builder."""
    ab = _parse_activated_ability_line(
        "{3}: Put a +1/+1 counter on target creature. "
        "Activate only as a sorcery."
    )
    assert ab == ActivatedAbility(
        cost=ActivationCost(mana=parse_mana_cost("{3}"), mana_str="{3}"),
        effect=PutPlusOneCounterOnTargetCreature(amount=1),
        sorcery_speed=True,
    )


def test_parse_clause_recognizes_activated_ability():
    parsed = _parse_clause("{2}: Draw a card.")
    assert parsed is not None
    assert parsed.activated == ActivatedAbility(
        cost=ActivationCost(mana=parse_mana_cost("{2}"), mana_str="{2}"),
        effect=DrawCards(amount=1),
    )
    assert parsed.keywords == ()
    assert parsed.trigger is None
    assert parsed.keyword is None


def test_parse_clause_does_not_claim_warp_does_not_collide_with_activated():
    """Warp lines (e.g. ``Warp {1}{R}``) lack the ``: <effect>.`` tail
    so the activated-ability matcher must not falsely claim them. Per
    the existing orthogonality rule, ``_parse_clause`` returns ``None``
    for a warp line — Warp is handled by the dedicated recognizer in
    ``_build_creature_printing``."""
    assert _parse_clause("Warp {1}{R}") is None


def test_build_creature_printing_handles_simple_activated_ability():
    """Synthetic creature with a pure ``{cost}: <effect>.`` activated
    ability — exercises the activated-ability passthrough end-to-end
    without depending on any other clause."""
    card = {
        "name": "Test Activator",
        "mana_cost": "{1}{G}",
        "type_line": "Creature — Insect",
        "power": "2",
        "toughness": "2",
        "oracle_text": "{3}: Put a +1/+1 counter on target creature.",
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.keywords == frozenset()
    assert printing.triggered_abilities == ()
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(mana=parse_mana_cost("{3}"), mana_str="{3}"),
            effect=PutPlusOneCounterOnTargetCreature(amount=1),
        ),
    )


def test_build_creature_printing_handles_activated_with_keyword():
    """Mixed clause types (keyword line + activated-ability line) on the
    same creature."""
    card = {
        "name": "Test Flyer",
        "mana_cost": "{2}{U}",
        "type_line": "Creature — Bird",
        "power": "1",
        "toughness": "1",
        "oracle_text": "Flying\n{2}: Draw a card.",
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.keywords == frozenset({"flying"})
    assert printing.triggered_abilities == ()
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(mana=parse_mana_cost("{2}"), mana_str="{2}"),
            effect=DrawCards(amount=1),
        ),
    )


def test_build_creature_printing_handles_multiple_activated_abilities():
    """Two pure activated abilities on the same creature both land in
    ``activated_abilities`` in oracle-text order."""
    card = {
        "name": "Test Multi-Activator",
        "mana_cost": "{2}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": (
            "{1}: You gain 1 life.\n"
            "{2}: Draw a card."
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(mana=parse_mana_cost("{1}"), mana_str="{1}"),
            effect=GainLife(amount=1),
        ),
        ActivatedAbility(
            cost=ActivationCost(mana=parse_mana_cost("{2}"), mana_str="{2}"),
            effect=DrawCards(amount=1),
        ),
    )


def test_build_creature_printing_attaches_sorcery_rider():
    """A creature whose activated ability has the
    ``Activate only as a sorcery.`` rider on a separate line — the
    rider attaches to the immediately preceding activated ability and
    sets ``sorcery_speed=True`` on it."""
    card = {
        "name": "Test Sorcery-Rider",
        "mana_cost": "{1}{G}",
        "type_line": "Creature — Insect",
        "power": "2",
        "toughness": "2",
        "oracle_text": (
            "{3}: Put a +1/+1 counter on target creature.\n"
            "Activate only as a sorcery."
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(mana=parse_mana_cost("{3}"), mana_str="{3}"),
            effect=PutPlusOneCounterOnTargetCreature(amount=1),
            sorcery_speed=True,
        ),
    )


def test_build_creature_printing_accepts_mana_plus_sacrifice_activated_ability():
    """A creature whose activated ability is ``{mana}, Sacrifice this
    creature: <effect>.`` is now claimable end-to-end through the
    creature builder — mana+sacrifice-self is a supported cost shape."""
    card = {
        "name": "Test Multi-Cost",
        "mana_cost": "{U}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": "{2}, Sacrifice this creature: Draw a card.",
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert len(printing.activated_abilities) == 1
    cost = printing.activated_abilities[0].cost
    assert cost.mana_str == "{2}"
    assert cost.sacrifice_self is True
    assert cost.tap_self is False


# ---------- tap costs and sorcery riders ----------


def test_parse_activated_ability_line_tap_only_draw():
    """``{T}: Draw a card.`` — pure tap-cost ability with no mana piece.
    The resulting ``ActivationCost`` has ``tap_self=True`` and a
    default-empty mana cost."""
    result = _parse_activated_ability_line("{T}: Draw a card.")
    assert result is not None
    assert result.cost.tap_self is True
    assert result.cost.mana is None
    assert result.cost.mana_str == ""
    assert result.effect == DrawCards(amount=1)
    # Sorcery-speed defaults to False — only the rider line flips it.
    assert result.sorcery_speed is False


def test_parse_activated_ability_line_tap_only_gain_life():
    result = _parse_activated_ability_line("{T}: You gain 1 life.")
    assert result == ActivatedAbility(
        cost=ActivationCost(tap_self=True),
        effect=GainLife(amount=1),
    )


def test_parse_activated_ability_line_tap_plus_mana_damage():
    """``{T}, {1}{R}: Deals 2 damage to any target.`` exercises the
    tap+mana shape with a non-trivial mana cost."""
    result = _parse_activated_ability_line(
        "{T}, {1}{R}: Deals 2 damage to any target."
    )
    assert result == ActivatedAbility(
        cost=ActivationCost(
            mana=parse_mana_cost("{1}{R}"),
            mana_str="{1}{R}",
            tap_self=True,
        ),
        effect=DealDamageAnyTarget(amount=2),
    )


def test_parse_activated_ability_line_accepts_three_piece_cost():
    """``{T}, {1}, Sacrifice this creature`` is a 3-piece cost — supported
    by the bag-merge classifier added in the multi-piece-costs slice.
    Detailed 3-piece coverage lives in
    ``tests/test_parser_multi_piece_costs.py``; keep a smoke check
    here so the dispatcher branch is exercised from the public entry
    point too."""
    ab = _parse_activated_ability_line(
        "{T}, {1}, Sacrifice this creature: Draw a card."
    )
    assert ab is not None
    assert ab.cost.tap_self is True
    assert ab.cost.sacrifice_self is True
    assert ab.cost.mana_str == "{1}"
    assert ab.effect == DrawCards(amount=1)


def test_parse_activated_ability_line_rejects_four_piece_cost():
    """4+ comma-separated pieces are out of scope this batch — the
    only EoE 4-piece cost (Susur Secundi's
    ``{1}{B}, {T}, Pay 2 life, Sacrifice a creature``) also needs the
    ``Sacrifice a creature`` target-sac atom which isn't wired yet,
    so the whole shape is left unsupported."""
    assert _parse_activated_ability_line(
        "{1}{B}, {T}, Pay 2 life, Sacrifice this creature: Draw a card."
    ) is None


def test_parse_activated_ability_line_accepts_tap_plus_sacrifice():
    """``{T}, Sacrifice this creature: ...`` — tap plus sacrifice-self.
    Now supported alongside the existing tap+mana shape."""
    ab = _parse_activated_ability_line(
        "{T}, Sacrifice this creature: Draw a card."
    )
    assert ab is not None
    assert ab.cost == ActivationCost(tap_self=True, sacrifice_self=True)
    assert ab.effect == DrawCards(amount=1)


def test_parse_activated_ability_line_accepts_pay_life_cost():
    """``Pay N life: ...`` — pure pay-life is now a supported cost
    shape. Detailed coverage lives in ``test_parser_pay_life.py``;
    keep a smoke check here so the dispatcher branch is exercised
    from the public entry point too."""
    ab = _parse_activated_ability_line("Pay 2 life: Draw a card.")
    assert ab is not None
    assert ab.cost.pay_life == 2
    assert ab.cost.tap_self is False
    assert ab.cost.mana is None


def test_parse_activated_ability_line_accepts_tap_plus_pay_life():
    """``{T}, Pay N life: ...`` — tap plus pay-life is now a
    supported cost shape. Detailed coverage lives in
    ``test_parser_pay_life.py``; keep a smoke check here so the
    dispatcher branch is exercised from the public entry point too."""
    ab = _parse_activated_ability_line(
        "{T}, Pay 2 life: Draw a card."
    )
    assert ab is not None
    assert ab.cost.pay_life == 2
    assert ab.cost.tap_self is True
    assert ab.cost.mana is None


def test_parse_activated_ability_line_rejects_tap_plus_remove_counter():
    """``{T}, Remove a +1/+1 counter from this creature: ...`` —
    tap plus a counter-removal cost. Out of scope; must not match."""
    assert _parse_activated_ability_line(
        "{T}, Remove a +1/+1 counter from this creature: Draw a card."
    ) is None


def test_parse_activated_ability_line_tap_only_strips_whitespace():
    assert _parse_activated_ability_line(
        "  {T}: Draw a card.  "
    ) == _parse_activated_ability_line("{T}: Draw a card.")


def test_parse_clause_recognizes_tap_only_activated_ability():
    """A pure tap line goes through ``_parse_clause`` and produces a
    ``_ParsedClause`` with the activated ability populated."""
    parsed = _parse_clause("{T}: Draw a card.")
    assert parsed is not None
    assert parsed.activated == ActivatedAbility(
        cost=ActivationCost(tap_self=True),
        effect=DrawCards(amount=1),
    )


def test_parse_clause_recognizes_tap_plus_mana_activated_ability():
    parsed = _parse_clause("{T}, {2}: Draw a card.")
    assert parsed is not None
    assert parsed.activated == ActivatedAbility(
        cost=ActivationCost(
            mana=parse_mana_cost("{2}"),
            mana_str="{2}",
            tap_self=True,
        ),
        effect=DrawCards(amount=1),
    )


def test_build_creature_printing_handles_tap_only_activated_ability():
    """Synthetic creature with a single ``{T}: <effect>.`` line — the
    ability lands in ``activated_abilities`` with ``tap_self=True``."""
    card = {
        "name": "Test Tapper",
        "mana_cost": "{1}{U}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": "{T}: Draw a card.",
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(tap_self=True),
            effect=DrawCards(amount=1),
        ),
    )


def test_build_creature_printing_handles_tap_plus_mana_activated_ability():
    """Synthetic creature whose activated ability has a tap+mana cost."""
    card = {
        "name": "Test TapMana",
        "mana_cost": "{1}{R}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": "{T}, {1}{R}: Deals 2 damage to any target.",
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(
                mana=parse_mana_cost("{1}{R}"),
                mana_str="{1}{R}",
                tap_self=True,
            ),
            effect=DealDamageAnyTarget(amount=2),
        ),
    )


def test_build_creature_printing_attaches_sorcery_rider_to_tap_only():
    """Tap-only activation followed by a separate-line sorcery rider
    flips ``sorcery_speed`` on the tap-only ability."""
    card = {
        "name": "Test Sorcery-Tap",
        "mana_cost": "{2}{B}",
        "type_line": "Creature — Wizard",
        "power": "2",
        "toughness": "2",
        "oracle_text": (
            "{T}: Draw a card.\n"
            "Activate only as a sorcery."
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(tap_self=True),
            effect=DrawCards(amount=1),
            sorcery_speed=True,
        ),
    )


def test_build_creature_printing_sorcery_rider_attaches_to_last_only():
    """When a creature has several activated abilities and a single
    rider line, the rider attaches only to the most recently parsed
    (immediately preceding) ability — the earlier ones stay
    instant-speed."""
    card = {
        "name": "Test Multi-Activated",
        "mana_cost": "{2}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": (
            "{1}: You gain 1 life.\n"
            "{2}: Draw a card.\n"
            "Activate only as a sorcery."
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.activated_abilities == (
        ActivatedAbility(
            cost=ActivationCost(mana=parse_mana_cost("{1}"), mana_str="{1}"),
            effect=GainLife(amount=1),
        ),
        ActivatedAbility(
            cost=ActivationCost(mana=parse_mana_cost("{2}"), mana_str="{2}"),
            effect=DrawCards(amount=1),
            sorcery_speed=True,
        ),
    )


def test_build_creature_printing_rejects_orphan_sorcery_rider():
    """A sorcery-rider line with no preceding activated ability on the
    card is malformed oracle text — there's nothing to mark
    sorcery-speed. Skip the card silently."""
    card = {
        "name": "Test Orphan Rider",
        "mana_cost": "{2}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": (
            "Flying\n"
            "Activate only as a sorcery."
        ),
    }
    assert _build_creature_printing(card) is None


def test_build_creature_printing_rejects_sorcery_rider_before_ability():
    """The rider attaches to the immediately PRECEDING activated
    ability — a rider that appears before any activated ability is an
    orphan and the card is skipped."""
    card = {
        "name": "Test Premature Rider",
        "mana_cost": "{2}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": (
            "Activate only as a sorcery.\n"
            "{2}: Draw a card."
        ),
    }
    assert _build_creature_printing(card) is None


def test_build_creature_printing_skips_tap_with_unsupported_effect():
    """A tap-only activation whose effect isn't in our grammar must not
    be partially claimed — the line fails and the card is skipped."""
    card = {
        "name": "Test Unsupported Tap",
        "mana_cost": "{1}{U}",
        "type_line": "Creature — Wizard",
        "power": "1",
        "toughness": "1",
        "oracle_text": "{T}: Scry 1.",
    }
    assert _build_creature_printing(card) is None


def test_parse_activated_ability_line_rejects_double_tap_cost():
    """``{T}{T}: ...`` is brace-only so passes the brace shape check but
    blows up in ``parse_mana_cost`` (``{T}`` is not a mana symbol).
    Defensive ``ValueError`` handling returns None."""
    assert _parse_activated_ability_line("{T}{T}: Draw a card.") is None


def test_parse_activated_ability_line_rejects_tap_plus_tap_cost():
    """``{T}, {T}: ...`` — comma-separated, both pieces are ``{T}``.
    The pieces-includes-{T} branch picks the OTHER piece (also ``{T}``)
    as the mana, which fails ``parse_mana_cost``."""
    assert _parse_activated_ability_line("{T}, {T}: Draw a card.") is None


def test_build_creature_printing_accepts_pure_sacrifice_cost_activated_ability():
    """Pure sacrifice-self costs are now supported. A creature whose
    only activated ability is ``Sacrifice this creature: <effect>.``
    is claimed end-to-end."""
    card = {
        "name": "Test Sac-Cost",
        "mana_cost": "{B}",
        "type_line": "Creature — Zombie",
        "power": "2",
        "toughness": "2",
        "oracle_text": "Sacrifice this creature: Draw a card.",
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert len(printing.activated_abilities) == 1
    cost = printing.activated_abilities[0].cost
    assert cost.sacrifice_self is True
    assert cost.tap_self is False
    assert cost.mana is None
    assert cost.mana_str == ""


# ---------- static abilities ----------

def test_parse_static_ability_line_artifact_pt_only():
    """Cloudsculpt Technician's exact static-ability line — bare
    P/T modifier, no granted keyword."""
    result = _parse_static_ability_line(
        "As long as you control an artifact, this creature gets +1/+0."
    )
    assert result == StaticAbility(
        scope=StaticScope.SELF,
        power_bonus=1,
        toughness_bonus=0,
        requires_artifact_you_control=True,
    )


def test_parse_static_ability_line_artifact_pt_and_keyword():
    """Gravblade Heavy's exact static-ability line — P/T modifier
    plus an ``and has <keyword>`` tail."""
    result = _parse_static_ability_line(
        "As long as you control an artifact, this creature gets +1/+0 "
        "and has deathtouch."
    )
    assert result == StaticAbility(
        scope=StaticScope.SELF,
        power_bonus=1,
        toughness_bonus=0,
        granted_keywords=frozenset({"deathtouch"}),
        requires_artifact_you_control=True,
    )


def test_parse_static_ability_line_extracts_pt_amounts():
    """The P/N capture groups should pick up the actual amounts, not
    just always 1/0."""
    result = _parse_static_ability_line(
        "As long as you control an artifact, this creature gets +2/+3."
    )
    assert result == StaticAbility(
        scope=StaticScope.SELF,
        power_bonus=2,
        toughness_bonus=3,
        requires_artifact_you_control=True,
    )


def test_parse_static_ability_line_strips_whitespace():
    line = (
        "  As long as you control an artifact, "
        "this creature gets +1/+0.  "
    )
    assert _parse_static_ability_line(line) == StaticAbility(
        scope=StaticScope.SELF,
        power_bonus=1,
        toughness_bonus=0,
        requires_artifact_you_control=True,
    )


def test_parse_static_ability_line_rejects_unknown_condition():
    """Conditions with no matching ``StaticAbility`` gate field must
    fall through — better to skip the card than silently mis-model it.
    ``you control a creature`` has no engine field, so it stays out of
    scope here."""
    assert _parse_static_ability_line(
        "As long as you control a creature, this creature gets +1/+0."
    ) is None


def test_parse_static_ability_line_recognizes_active_turn_condition():
    """``As long as it's your turn`` maps to ``active_turn_only=True``
    on the produced ``StaticAbility``. Forward-looking grammar
    coverage: the engine gate already exists and the surface phrase is
    unambiguous, so the parser surfaces a ``StaticAbility`` rather than
    falling through."""
    assert _parse_static_ability_line(
        "As long as it's your turn, this creature gets +2/+0."
    ) == StaticAbility(
        scope=StaticScope.SELF,
        power_bonus=2,
        toughness_bonus=0,
        active_turn_only=True,
    )


def test_parse_static_ability_line_rejects_unknown_keyword():
    """An ``and has <keyword>`` tail with a non-evergreen keyword
    fails the parse — the parser stays conservative."""
    assert _parse_static_ability_line(
        "As long as you control an artifact, this creature gets +1/+0 "
        "and has scry."
    ) is None


def test_parse_static_ability_line_rejects_non_static_lines():
    assert _parse_static_ability_line("Flying") is None
    assert _parse_static_ability_line(
        "When this creature enters, draw a card."
    ) is None
    assert _parse_static_ability_line("") is None


def test_parse_static_ability_line_rejects_missing_period():
    """The trailing period is part of the line shape — without it the
    line doesn't match the pattern."""
    assert _parse_static_ability_line(
        "As long as you control an artifact, this creature gets +1/+0"
    ) is None


def test_parse_clause_recognizes_static_ability():
    parsed = _parse_clause(
        "As long as you control an artifact, this creature gets +1/+0."
    )
    assert parsed is not None
    assert parsed.static == StaticAbility(
        scope=StaticScope.SELF,
        power_bonus=1,
        toughness_bonus=0,
        requires_artifact_you_control=True,
    )
    assert parsed.keywords == ()
    assert parsed.trigger is None
    assert parsed.activated is None
    assert parsed.keyword is None


def test_build_creature_printing_handles_simple_static_ability():
    """Synthetic creature with only a static-ability clause —
    exercises the static-ability passthrough end-to-end without
    depending on any other clause."""
    card = {
        "name": "Test Static",
        "mana_cost": "{2}{U}",
        "type_line": "Creature — Jellyfish",
        "power": "1",
        "toughness": "4",
        "oracle_text": (
            "As long as you control an artifact, "
            "this creature gets +1/+0."
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.static_abilities == (
        StaticAbility(
            scope=StaticScope.SELF,
            power_bonus=1,
            toughness_bonus=0,
            requires_artifact_you_control=True,
        ),
    )
    assert printing.triggered_abilities == ()
    assert printing.activated_abilities == ()


def test_build_creature_printing_handles_static_with_keyword_line():
    """Cloudsculpt Technician's exact two-line shape: a Flying
    keyword line plus the static-ability line. Both clauses
    contribute and end up on the resulting printing."""
    card = {
        "name": "Test Static Flyer",
        "mana_cost": "{2}{U}",
        "type_line": "Creature — Jellyfish Artificer",
        "power": "1",
        "toughness": "4",
        "oracle_text": (
            "Flying\n"
            "As long as you control an artifact, "
            "this creature gets +1/+0."
        ),
    }
    printing = _build_creature_printing(card)
    assert printing is not None
    assert printing.keywords == frozenset({"flying"})
    assert printing.static_abilities == (
        StaticAbility(
            scope=StaticScope.SELF,
            power_bonus=1,
            toughness_bonus=0,
            requires_artifact_you_control=True,
        ),
    )


def test_build_creature_printing_skips_unknown_static_condition():
    """A creature whose static-ability condition is out of scope
    falls through and the whole card must be skipped."""
    card = {
        "name": "Test Bad Static",
        "mana_cost": "{2}{G}",
        "type_line": "Creature — Beast",
        "power": "2",
        "toughness": "2",
        "oracle_text": (
            "As long as you control a creature, "
            "this creature gets +1/+0."
        ),
    }
    assert _build_creature_printing(card) is None


def test_source_claims_cloudsculpt_technician():
    """Cloudsculpt Technician's Flying + artifacts-matter static is
    now in the parser's grammar."""
    source = OracleParserCardSource()
    assert "Cloudsculpt Technician" in source.claimed_names
    assert "Cloudsculpt Technician" in source.get_printings()


def test_source_claims_gravblade_heavy():
    """Gravblade Heavy's artifacts-matter static with both a P/T
    bonus and a granted deathtouch keyword is now in the parser's
    grammar."""
    source = OracleParserCardSource()
    assert "Gravblade Heavy" in source.claimed_names
    assert "Gravblade Heavy" in source.get_printings()
