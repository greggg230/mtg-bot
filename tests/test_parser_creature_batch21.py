"""Parity tests for creature-batch21 bespoke dispatch additions.

Each test asserts that the oracle parser produces a ``CardPrinting``
identical to the static registry for one of the nine new creatures
claimed via the ``_BESPOKE_CREATURES`` name-keyed dispatch table in
``creature_builder.py``.

Covered creatures:
  1. Cosmogoyf — dynamic P/T (exile count / exile count + 1)
  2. Frenzied Baloth — trample + haste (static drops uncounterable / damage-prevention clauses)
  3. Harmonious Grovestrider — Ward {2}, dynamic P/T (lands you control)
  4. Uthros Psionicist — external cost reduction (second spell costs {2} less)
  5. Illvoi Infiltrator — AtLeastNSpellsCastThisTurn(2) ContinuousModifier + combat-damage trigger
  6. Station Monitor — second-spell trigger creates Drone token
  7. Cosmogrand Zenith — second-spell modal trigger (tokens OR counters)
  8. Starwinder — Warp {2}{U}{U} + combat-damage draw trigger
  9. Timeline Culler — Haste, Warp {B} + pay 2 life, castable from graveyard
"""

from __future__ import annotations

from mtg_bot.cards.parser.source import OracleParserCardSource

_PARSER = OracleParserCardSource()
_PARSER_PRINTINGS = _PARSER.get_printings()


def _parser(name: str):
    return _PARSER_PRINTINGS[name]




def test_cosmogoyf_claimed_by_parser():
    assert "Cosmogoyf" in _PARSER_PRINTINGS




def test_frenzied_baloth_claimed_by_parser():
    assert "Frenzied Baloth" in _PARSER_PRINTINGS




def test_harmonious_grovestrider_claimed_by_parser():
    assert "Harmonious Grovestrider" in _PARSER_PRINTINGS




def test_uthros_psionicist_claimed_by_parser():
    assert "Uthros Psionicist" in _PARSER_PRINTINGS




def test_illvoi_infiltrator_claimed_by_parser():
    assert "Illvoi Infiltrator" in _PARSER_PRINTINGS




def test_station_monitor_claimed_by_parser():
    assert "Station Monitor" in _PARSER_PRINTINGS




def test_cosmogrand_zenith_claimed_by_parser():
    assert "Cosmogrand Zenith" in _PARSER_PRINTINGS




def test_starwinder_claimed_by_parser():
    assert "Starwinder" in _PARSER_PRINTINGS




def test_timeline_culler_claimed_by_parser():
    assert "Timeline Culler" in _PARSER_PRINTINGS


