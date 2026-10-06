"""Root-level determinization: convert a god-view ``GameState`` into a
sample consistent with one player's knowledge.

The MCTS search currently sees every card in every zone. Without
determinization it can implicitly use information the viewer wouldn't
have in a real game (top of own library, opponent's exact hand) — the
search measures stronger than the agent would actually play. The fix
(per ``docs/HIDDEN_INFO_MODEL.md``) is to sample a world that's
consistent with what the viewer knows, then search that world.

This module provides the sampler. Integration into MCTS — running K
samples per search call and aggregating their visit counts (Perfect
Information Monte Carlo) — is the next layer up and lives in the
agent / strategy code.

Algorithm: for each player, pool the printings of every card in their
hand+library that the viewer doesn't know, shuffle the pool, reassign.
Zones, ids, owners, and per-zone counts are all preserved; only the
printing each ``GameObject`` carries is permuted. That matches the
viewer's epistemic state under our per-card flag model — they know how
many cards live in each hidden zone, they just don't know which is
which.
"""

from __future__ import annotations

import functools
import random
from typing import Callable

from mtg_bot.cards.hidden import HIDDEN_CARD_PRINTING
from mtg_bot.game import _CowList, GameObject, GameState, Zone


# How the searching agent models the opponent's hidden cards:
#
# * ``"sampled"`` (default; the behaviour since PR 2): re-permute the
#   opponent's unknown hand+library printings, i.e. imagine a concrete
#   hand drawn from their deck.
# * ``"board_only"``: don't imagine them at all -- every opponent
#   hand / library card the viewer can't identify becomes the inert
#   ``mtg_bot.cards.hidden.HIDDEN_CARD_PRINTING`` placeholder (never
#   castable / playable / activatable; still counts toward hand size,
#   discard, mill). The opponent can still act with everything public
#   (attack, block, activate permanents). The viewer's OWN library is
#   sampled exactly as under ``"sampled"`` (we know our decklist).
#   See ``docs/SEARCH_LIMITATIONS.md`` ("Board-only opponent model").
OPPONENT_HIDDEN_MODELS: tuple[str, ...] = ("sampled", "board_only")
DEFAULT_OPPONENT_HIDDEN_MODEL = "sampled"


def check_opponent_hidden_model(model: str) -> str:
    """Validate an ``opponent_hidden_model`` value; returns it."""
    if model not in OPPONENT_HIDDEN_MODELS:
        raise ValueError(
            f"opponent_hidden_model must be one of "
            f"{list(OPPONENT_HIDDEN_MODELS)}, got {model!r}"
        )
    return model


_PUBLIC_ZONES: frozenset[Zone] = frozenset(
    {Zone.BATTLEFIELD, Zone.GRAVEYARD, Zone.EXILE, Zone.STACK}
)


def viewer_knows(card: GameObject, viewer_id: int) -> bool:
    """Whether ``viewer_id`` has perfect-position knowledge of ``card``'s
    identity in its current zone right now.

    True when any of:
    - ``viewer_id`` is in ``card.publicly_known_to`` (explicitly tracked
      reveal).
    - ``card.zone`` is a public zone (battlefield, graveyard, exile,
      stack).
    - ``card`` is in ``viewer_id``'s own hand. (Viewer always knows
      their own hand. The reveal-tracking layer doesn't currently mark
      drawn cards as known to the drawer; this rule fills the gap so
      the determinizer doesn't permute the viewer's own hand.)

    False otherwise — chiefly: opponent hand cards that haven't been
    revealed, and library cards (including the viewer's own library,
    whose order is hidden from them).
    """
    if viewer_id in card.publicly_known_to:
        return True
    if card.zone in _PUBLIC_ZONES:
        return True
    if card.zone is Zone.HAND and card.owner_id == viewer_id:
        return True
    return False


def determinize(
    state: GameState,
    viewer_id: int,
    rng: random.Random | None = None,
    *,
    in_place: bool = False,
    opponent_hidden_model: str = DEFAULT_OPPONENT_HIDDEN_MODEL,
) -> GameState:
    """Return a state consistent with what ``viewer_id`` knows.

    For each player, every card in their hand or library that the
    viewer doesn't know has its ``printing`` (and ``original_printing``)
    re-shuffled within the per-player pool of unknown printings. Zones,
    ids, owners, and zone counts are unchanged.

    ``rng`` defaults to a fresh ``random.Random()`` — pass an explicit
    one for reproducibility, or call repeatedly with different rngs to
    draw independent samples for variance reduction (PIMC).

    ``in_place=False`` (default): clone the input first; the caller's
    state is untouched. Standard for callers holding a god-view state
    they want to preserve.

    ``in_place=True``: mutate ``state`` directly. Used by PIMC, which
    clones the source adapter (preserving its bookkeeping like
    ``_step_entered`` and ``_pass_count``) and then determinizes the
    cloned engine state in place — re-cloning here would be wasteful
    and the caller has already paid for one.

    Soundness: the result is *under-informed*, never *misinformed*.
    Whatever the viewer truly knows is preserved; everything else is
    a fresh sample from the deck-list-consistent unknown pool.

    Use at the root of an MCTS search, before the search clones for
    rollouts. The returned state's library order is then concrete, so
    no further chance-node modeling is required for card draws within
    the search.

    ``opponent_hidden_model="board_only"`` replaces (instead of
    permuting) every hidden opponent hand / library card with the inert
    placeholder; the viewer's own hidden cards are still permuted. See
    ``OPPONENT_HIDDEN_MODELS``.
    """
    check_opponent_hidden_model(opponent_hidden_model)
    board_only = opponent_hidden_model == "board_only"
    if rng is None:
        rng = random.Random()
    target = state if in_place else state.clone()
    for player in target.players:
        if board_only and player.id != viewer_id:
            _blank_player(player, viewer_id)
        else:
            _determinize_player(player, viewer_id, rng)
    return target


def make_determinize_fn(opponent_hidden_model: str) -> Callable:
    """The ``determinize_fn`` for PIMC / ISMCTS under the given
    opponent model. ``"sampled"`` returns :func:`determinize` itself
    (so the default search is bit-identical to before the option
    existed); ``"board_only"`` returns a picklable
    ``functools.partial`` of it."""
    check_opponent_hidden_model(opponent_hidden_model)
    if opponent_hidden_model == DEFAULT_OPPONENT_HIDDEN_MODEL:
        return determinize
    return functools.partial(
        determinize, opponent_hidden_model=opponent_hidden_model,
    )


def _blank_player(player, viewer_id: int) -> None:
    """Board-only: give every hand / library card of ``player`` that
    the viewer doesn't know the inert hidden-card placeholder. Same
    CoW fork as ``_determinize_player`` (the library's GameObjects are
    shared with the source state until forked)."""
    library = player.library
    if isinstance(library, _CowList):
        library.fork()
    for zone in (player.hand, player.library):
        for card in zone:
            if viewer_knows(card, viewer_id):
                continue
            card.printing = HIDDEN_CARD_PRINTING
            if card.original_printing is not None:
                card.original_printing = HIDDEN_CARD_PRINTING


def _determinize_player(
    player, viewer_id: int, rng: random.Random
) -> None:
    """Permute printings among ``player``'s hand+library cards that the
    viewer doesn't know.

    Pools hand and library together: the viewer can see the per-zone
    counts but can't tell which unknown card sits in which unknown
    position, so swaps across zones are valid. Counts stay constant
    because we only mutate ``printing`` / ``original_printing`` on
    existing objects — we don't move objects between zones.

    CoW share-group fork: library/graveyard/exile are ``_CowList``
    instances that share GameObject references with the source state's
    sibling lists. The list-level CoW intercepts list mutators
    (append/pop/etc.) but NOT in-place attribute writes on the
    elements themselves — so naive ``card.printing = new`` would leak
    back through the shared GameObject reference into the source
    state's library, accumulating phantom card identities across
    successive PIMC determinizations and corrupting future draws.
    Forcing the fork on ``player.library`` first isolates this
    determinization's GameObjects from the source. Hand stays deep-
    cloned by ``GameState.clone`` so it doesn't need the same
    treatment; we still read from it but the elements there are
    already independent.
    """
    library = player.library
    if isinstance(library, _CowList):
        library.fork()

    hidden: list[GameObject] = []
    for card in player.hand:
        if not viewer_knows(card, viewer_id):
            hidden.append(card)
    for card in player.library:
        if not viewer_knows(card, viewer_id):
            hidden.append(card)
    if len(hidden) <= 1:
        return
    # Snapshot the printings, shuffle, reassign. Working with the
    # printing-tuple keeps original_printing in lockstep with printing —
    # cards in hand/library haven't been "becomes a copy"-mutated, so
    # their original_printing typically equals printing or is None,
    # but we permute as a tuple to stay correct if either holds.
    printings = [(c.printing, c.original_printing) for c in hidden]
    rng.shuffle(printings)
    for card, (new_printing, new_original) in zip(hidden, printings):
        card.printing = new_printing
        card.original_printing = new_original
