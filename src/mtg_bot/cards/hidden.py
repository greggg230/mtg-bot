"""The inert "hidden card" placeholder used by the board-only opponent
model (``opponent_hidden_model="board_only"``, see
``mtg_bot.search.determinize``).

In a board-only determinization the searching agent does not imagine
the opponent's unseen cards: every opponent hand / library card the
viewer cannot identify has its printing replaced by
:data:`HIDDEN_CARD_PRINTING`. The placeholder is built so that the
engine can never do anything with it:

* no card types -> ``is_instant`` / ``is_land`` / ``is_permanent`` /
  ``is_creature`` / ``is_sorcery`` are all False, so
  ``legal.legal_actions`` never offers ``CastSpell`` / ``PlayLand``
  for it (every hand cast / land path is gated on one of those);
* no keywords (no flash), no warp / kicker / impending / room halves,
  no abilities of any kind, no ``hand_ability_grants``;
* mana cost ``{}`` (mana value 0), colourless.

It still counts as a card for hand size, discard, mill, shuffle and
"cards in hand" effects, so hand / library cardinality stays exactly
the public count.

The name is deliberately not a legal Scryfall name, so it can never
collide with a card in any pool and is never in a card vocabulary:
``state_tokenizer.printing_id`` maps it to ``PRINTING_NA`` (the
redacted id), and the tokenizer additionally marks placeholder zone
cards non-public (see ``state_tokenizer._make_zone_card_token``).

Identity is by NAME (:func:`is_hidden_placeholder`), not by object
identity, so a pickled / copied placeholder is still recognised.
"""

from __future__ import annotations

from typing import Optional

from mtg_bot.cards.base import CardPrinting, ManaCost


HIDDEN_CARD_NAME = "<hidden card>"

HIDDEN_CARD_PRINTING = CardPrinting(
    name=HIDDEN_CARD_NAME,
    mana_cost_str="",
    mana_cost=ManaCost(),
    type_line="",
    supertypes=(),
    card_types=(),
    subtypes=(),
)


def is_hidden_placeholder(printing: Optional[CardPrinting]) -> bool:
    """True iff ``printing`` is the board-only hidden-card placeholder."""
    return printing is not None and printing.name == HIDDEN_CARD_NAME
