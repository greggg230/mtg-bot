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
    DealDamageToTargetCreatureSpacecraftBoosted,
    DepressurizeEffect,
    DiplomaticRelationsEffect,
    FocusFireEffect,
    PumpCreaturesYouControlUntilEOT,
    PumpTargetCreatureUntilEOT,
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
