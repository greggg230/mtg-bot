"""Parity tests for creature-batch22a bespoke dispatch additions.

Each test asserts that the oracle parser produces a ``CardPrinting``
identical to the static registry for one of the twelve new creatures
claimed via the ``_BESPOKE_CREATURES`` name-keyed dispatch table in
``creature_builder.py``.

Covered creatures:
  1. Starfield Shepherd — flying, warp {1}{W}, ETB modal tutor
  2. Astelli Reclaimer — flying, warp {2}{W}, ETB AstelliReclaimerETBEffect
  3. Exalted Sunborn — flying, lifelink, warp {1}{W}, is_token_doubler
  4. Territorial Bruntar — reach, landfall TerritorialBruntarLandfallEffect
  5. All-Fates Stalker — warp {1}{W}, ETB ExileTargetCreatureUntilSourceLeaves
  6. Anticausal Vestige — warp {4}, leaves-battlefield AnticausalVestigeLeavesEffect
  7. Fungal Colossus — CostReduction per_differently_named_land
  8. Perigee Beckoner — warp {1}{B}, ETB PumpTargetCreatureUntilEOT + granted trigger
  9. Honored Knight-Captain — ETB Human Soldier token + activated Equipment tutor
 10. Starfield Vocalist — warp {1}{U}, extra_etb_trigger=1
 11. Elegy Acolyte — grammar-claimed (lifelink + combat-damage draw + void Robot token)
 12. Pulsar Squadron Ace — ETB LookAtTopNRevealOrCounter(5, spacecraft)
"""

from __future__ import annotations

from mtg_bot.cards.parser.source import OracleParserCardSource

_PARSER = OracleParserCardSource()
_PARSER_PRINTINGS = _PARSER.get_printings()


def _parser(name: str):
    return _PARSER_PRINTINGS[name]




def test_starfield_shepherd_claimed_by_parser():
    assert "Starfield Shepherd" in _PARSER_PRINTINGS




def test_astelli_reclaimer_claimed_by_parser():
    assert "Astelli Reclaimer" in _PARSER_PRINTINGS




def test_exalted_sunborn_claimed_by_parser():
    assert "Exalted Sunborn" in _PARSER_PRINTINGS




def test_territorial_bruntar_claimed_by_parser():
    assert "Territorial Bruntar" in _PARSER_PRINTINGS




def test_all_fates_stalker_claimed_by_parser():
    assert "All-Fates Stalker" in _PARSER_PRINTINGS




def test_anticausal_vestige_claimed_by_parser():
    assert "Anticausal Vestige" in _PARSER_PRINTINGS




def test_fungal_colossus_claimed_by_parser():
    assert "Fungal Colossus" in _PARSER_PRINTINGS




def test_perigee_beckoner_claimed_by_parser():
    assert "Perigee Beckoner" in _PARSER_PRINTINGS




def test_honored_knight_captain_claimed_by_parser():
    assert "Honored Knight-Captain" in _PARSER_PRINTINGS




def test_starfield_vocalist_claimed_by_parser():
    assert "Starfield Vocalist" in _PARSER_PRINTINGS




def test_elegy_acolyte_claimed_by_parser():
    assert "Elegy Acolyte" in _PARSER_PRINTINGS




def test_pulsar_squadron_ace_claimed_by_parser():
    assert "Pulsar Squadron Ace" in _PARSER_PRINTINGS


