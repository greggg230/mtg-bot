"""End-to-end tests for ``GameState.apply_action(action_dict)``.

Phase 3 of the Rust ``Action`` enum + apply_action dispatcher. Mirrors
the Python ``Action`` subclass hierarchy in ``src/mtg_bot/actions.py``:
each Python ``Action`` (and its dict serialization that the search
loop emits) gets a matching Rust variant inside ``rust/src/actions.rs``,
and ``apply_action`` is the bridge that turns a dict into a state
mutation.

**Slice scope.** Round 22 wired ``PassPriority`` (364 of 420 decisions
/ ~87% in the round-20 measurement). Round 23 adds ``PlayLand`` (10 of
420 / ~2.4% — the 4th-most-frequent variant). The ``apply_action``
dispatcher exists so callers can drive Rust state through the same
shape of dict the Python search loop already builds; future slices
grow the match arm (CastSpell, Activate, ...) plus a
``legal_decisions`` enumerator without changing the bridge shape.

**Bridge surface is end-to-end only.** ``parse_action`` /
``action_to_pydict`` are module-internal helpers in ``actions.rs``;
this slice exposes them only through ``apply_action`` (the actually-
useful pyfunction). A future slice can lift them to module-level
``#[pyfunction]`` if a test needs to round-trip a dict in isolation,
but today the end-to-end path is what matters and avoids touching
``lib.rs``.
"""

from __future__ import annotations

import pytest


def test_apply_action_pass_priority_toggles_priority(synthetic_state):
    """``apply_action({"kind": "pass_priority"})`` routes to the
    existing ``pass_priority()`` primitive — priority flips from
    active player (0) to the other player (1). A SINGLE pass does NOT
    fire the CR 117.4 step-advance (closes G3 at the action layer;
    see pass-priority-step-advance slice): the consecutive-passes
    counter is 1 after one pass, and step-advance fires only when
    both players have passed in succession with an empty stack."""
    assert synthetic_state.get_priority_player() == 0
    starting_phase = synthetic_state.current_phase()
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 1
    # Single pass: counter == 1, no step advance.
    assert synthetic_state.current_phase() == starting_phase


def test_apply_action_pass_priority_twice_advances_step(synthetic_state):
    """G3 close-out (pass-priority-step-advance slice): two
    apply_action pass_priority calls with an empty stack fire CR
    117.4 — the current step ends and the engine advances to the
    next step via ``advance_phase()``. Priority returns to player 0
    after the toggle (1→0); the new phase is UPKEEP (UNTAP→UPKEEP).
    Replaces the pre-slice "two applies return to the starting
    priority player" pin, which described the pure-toggle behavior
    that blocked the parity_smoke harness from ever reaching MAIN1.

    (The bare ``state.pass_priority()`` primitive in
    ``state_ops_priority.rs`` is unchanged — it remains a pure
    toggle that doesn't consult the stack or the phase. The CR 117.4
    composition is wired in apply_action; ``test_priority.py``'s
    primitive-shape pins still pass against the bare method.)"""
    starting_phase = synthetic_state.current_phase()
    synthetic_state.apply_action({"kind": "pass_priority"})
    synthetic_state.apply_action({"kind": "pass_priority"})
    # Priority is back to active player after the second toggle.
    assert synthetic_state.get_priority_player() == 0
    # Step advanced one tick.
    assert synthetic_state.current_phase() == starting_phase + 1


def test_apply_action_returns_none(synthetic_state):
    """``apply_action`` is a side-effecting mutator — Python
    convention is to return ``None``. Locks in that callers can't
    accidentally rely on a return value the future grow-the-arm
    slices might change."""
    result = synthetic_state.apply_action({"kind": "pass_priority"})
    assert result is None


def test_apply_action_unknown_kind_raises_value_error(synthetic_state):
    """Unknown ``kind`` strings raise ``ValueError`` with a helpful
    message listing the valid kinds. Single point of validation:
    ``parse_action`` rejects before dispatch."""
    with pytest.raises(ValueError, match="unknown action kind"):
        synthetic_state.apply_action({"kind": "frobnicate"})
    # State unchanged after the rejected call.
    assert synthetic_state.get_priority_player() == 0


def test_apply_action_missing_kind_raises_key_error(synthetic_state):
    """Missing the ``kind`` key raises ``KeyError`` — mirrors the
    ``cards.rs`` ``require`` helper's error shape so callers
    that handle one bridge's errors handle them all the same way."""
    with pytest.raises(KeyError, match="kind"):
        synthetic_state.apply_action({})
    assert synthetic_state.get_priority_player() == 0


def test_apply_action_does_not_affect_active_player(synthetic_state):
    """Active player and priority player are independent fields.
    ``apply_action({"kind": "pass_priority"})`` moves only
    ``priority_player``; ``active_player`` stays put (mirrors the
    underlying ``pass_priority()`` primitive's contract)."""
    starting_active = synthetic_state.active_player
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.active_player == starting_active


def test_apply_action_on_clone_does_not_affect_original(synthetic_state):
    """CoW canary via the Action bridge: applying an action to a clone
    mutates only the clone's priority. The original is untouched.
    The clone is a by-value copy of the ``priority_player`` u8 field
    here, but this test pins the behavior so a future slice that
    moves state into an Arc-backed structure doesn't silently leak
    mutations back."""
    cloned = synthetic_state.clone_once()
    cloned.apply_action({"kind": "pass_priority"})
    assert cloned.get_priority_player() == 1
    assert synthetic_state.get_priority_player() == 0


# --- G3: CR 117.4 step-advance composition on PassPriority ----------
#
# pass-priority-step-advance slice. Closes RUST_PARITY_GAPS.md G3 at
# the apply_action layer: the bare ``pass_priority()`` primitive is
# still a pure toggle (test_priority.py pins that), but
# ``apply_action({"kind": "pass_priority"})`` now wraps the toggle
# with the CR 117.4 step-advance composition — when both players
# pass priority in succession with an empty stack, the engine
# advances to the next step. Implementation lives in
# ``state_ops_actions.rs`` and uses a ``consecutive_passes`` counter
# on ``GameState``.


def test_apply_action_pass_priority_with_nonempty_stack_does_not_advance(
    synthetic_state,
):
    """G3 close-out, stack-aware leg: when the stack is non-empty
    (e.g. a spell or ability is waiting to resolve), two consecutive
    passes do NOT fire ``advance_phase``. This slice scopes the
    composition to the empty-stack/step-advance branch of CR 117.4;
    the stack-resolution branch is still reached via the explicit
    ``complete_priority_round()`` primitive. Pins that the dispatcher
    consults ``state.stack`` before firing the step-advance — a
    regression that ignored the stack would silently lose the in-
    flight stack item by advancing past it."""
    synthetic_state.push_to_stack(42, 0)
    assert synthetic_state.stack_size() == 1
    starting_phase = synthetic_state.current_phase()
    synthetic_state.apply_action({"kind": "pass_priority"})
    synthetic_state.apply_action({"kind": "pass_priority"})
    # No step advance with a non-empty stack.
    assert synthetic_state.current_phase() == starting_phase
    # Stack item still pending — neither pass resolved it.
    assert synthetic_state.stack_size() == 1
    # Priority sequence is the pure-toggle 0→1→0.
    assert synthetic_state.get_priority_player() == 0


def test_apply_action_pass_priority_then_play_land_then_pass_does_not_advance(
    synthetic_state,
):
    """G3 close-out, "intervening action resets the chain" leg: a
    non-pass action between two passes breaks the consecutive-pass
    chain. Pin: pass → play_land → pass leaves the phase unchanged
    (only one pass counts toward the round-completion counter after
    the play_land reset). A future slice may extend the
    ``legal_decisions`` enumerator to model this same chain-break;
    the dispatcher's behavior matters first.

    This pins that ``consecutive_passes`` is RESET (not just
    "incremented elsewhere") when a non-pass arm fires — a regression
    that only incremented on pass arms would still allow a
    "pass, play_land, pass" sequence to advance the step on the
    second pass."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    starting_phase = synthetic_state.current_phase()
    synthetic_state.apply_action({"kind": "pass_priority"})
    # play_land also flips priority (no — play_land doesn't touch
    # priority; G4 still open). But it MUST reset consecutive_passes.
    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_id})
    # One pass after the reset — counter == 1, no advance.
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.current_phase() == starting_phase


def test_apply_action_pass_priority_four_passes_advance_two_steps(synthetic_state):
    """G3 close-out, multi-step ladder: four consecutive passes walk
    the phase ladder by two ticks (UNTAP → UPKEEP after passes 1-2,
    UPKEEP → DRAW after passes 3-4). Locks in that the counter resets
    to 0 after each step-advance, so the next pair fires the next
    advance. Mirrors the production parity-smoke trajectory: the
    harness emits long pass-priority runs to walk past the pre-main
    steps and reach MAIN1 where actual decisions exist."""
    starting_phase = synthetic_state.current_phase()
    for _ in range(4):
        synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.current_phase() == starting_phase + 2
    # Priority is back to active player after each pair of toggles.
    assert synthetic_state.get_priority_player() == 0


def test_apply_action_pass_priority_round_trip_through_untap_fires_untap_sweep(
    synthetic_state,
):
    """G3 close-out, untap-sweep canary: walking via apply_action
    pass-priority pairs all the way around the phase ladder lands
    back on UNTAP for the OTHER player — and the cross-slice
    composition with ``state_ops_steps.rs`` fires the untap sweep on
    the new active player's permanents. Pins that the action-layer
    step-advance reaches the same downstream effects as a direct
    ``advance_phase()`` call (i.e. apply_action is a SUPER-set, not a
    SUB-set, of the bare primitive's effects).

    The synthetic seed seeds P1's battlefield with untapped
    permanents (Forest + Grizzly Bears at ids 67..71 / 72..76); tap
    one before the wrap to make the sweep observable. 12 phases × 2
    passes per phase = 24 passes to lap once and land on the next
    turn's UNTAP."""
    # Tap a P1 permanent so the untap sweep has something to flip.
    p1_forest_id = 67  # base 67 is the first P1 forest
    synthetic_state.tap_object(p1_forest_id)
    assert synthetic_state.is_tapped(p1_forest_id) is True
    # 24 passes = 12 step advances = full lap to next turn's UNTAP.
    for _ in range(24):
        synthetic_state.apply_action({"kind": "pass_priority"})
    # Back on UNTAP, active player flipped to 1, untap sweep fired.
    assert synthetic_state.current_phase() == 0
    assert synthetic_state.active_player == 1
    assert synthetic_state.is_tapped(p1_forest_id) is False


# --- PlayLand --------------------------------------------------------
#
# Round-23 slice: extend the Action enum + apply_action dispatcher
# with ``Action::PlayLand { card_id }``. The Python dataclass
# ``PlayLand(card_id: int)`` carries only the card id; the implicit
# actor is the priority player. The Rust ``play_land(obj_id,
# player_id)`` primitive itself takes both — apply_action reads
# ``state.priority_player`` and forwards.
#
# Synthetic-state reminder (cf. ``tests_rust/test_cast.py``):
# - Player 0 hand: ids 10..16, all Llanowar Elves (card_types =
#   ["Creature"]).
# - Player 0 battlefield: ids 0..4 Forest, 5..9 Grizzly Bears.
# - Player 0 library: ids 17..66, all Forest. Tests that need a
#   Forest in hand pull one over via ``move_object``.

ZONE_HAND = 0
ZONE_BATTLEFIELD = 2
P0_FIRST_BATTLEFIELD_ID = 0
P0_FIRST_BEARS_ID = 5
P0_FIRST_HAND_ID = 10  # Llanowar Elves
P0_FIRST_LIBRARY_ID = 17  # first Forest in library


def _move_forest_into_hand(state, player_id=0):
    """Move a Forest from ``player_id``'s library into their hand and
    return its id. Mirrors the helper in ``tests_rust/test_cast.py`` —
    the synthetic seed has only Llanowar Elves in hand, so PlayLand
    tests need to import a land first."""
    forest_id = P0_FIRST_LIBRARY_ID
    assert state.printing_name_of(forest_id) == "Forest"
    assert state.move_object(forest_id, ZONE_HAND, player_id) is True
    assert state.find_object_zone(forest_id) == ZONE_HAND
    return forest_id


def test_apply_action_play_land_dispatches_to_play_land(synthetic_state):
    """Happy path: a Forest in the priority player's hand moves to
    the battlefield after ``apply_action({"kind": "play_land",
    "card_id": forest_id})``. The dispatch routes through
    ``call_method1("play_land", (card_id, priority_player))``."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    assert synthetic_state.get_priority_player() == 0
    hand_before = synthetic_state.hand_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]

    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_id})

    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert synthetic_state.hand_sizes[0] == hand_before - 1
    assert synthetic_state.battlefield_sizes[0] == bf_before + 1


def test_apply_action_play_land_uses_priority_player_as_actor(synthetic_state):
    """The actor is implicit (priority player). Flip priority to
    player 1, import a Forest into player 1's hand, and verify
    apply_action moves it onto player 1's battlefield — not
    player 0's. Locks in that the dispatcher reads
    ``priority_player`` rather than a hardcoded 0."""
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 1

    forest_id = _move_forest_into_hand(synthetic_state, 1)
    p1_hand_before = synthetic_state.hand_sizes[1]
    p1_bf_before = synthetic_state.battlefield_sizes[1]
    p0_hand_before = synthetic_state.hand_sizes[0]
    p0_bf_before = synthetic_state.battlefield_sizes[0]

    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_id})

    # Forest landed on player 1's battlefield.
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert synthetic_state.hand_sizes[1] == p1_hand_before - 1
    assert synthetic_state.battlefield_sizes[1] == p1_bf_before + 1
    # Player 0 entirely untouched.
    assert synthetic_state.hand_sizes[0] == p0_hand_before
    assert synthetic_state.battlefield_sizes[0] == p0_bf_before


def test_apply_action_play_land_missing_card_id_raises_key_error(synthetic_state):
    """Missing the ``card_id`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``kind``-missing error shape)."""
    with pytest.raises(KeyError, match="card_id"):
        synthetic_state.apply_action({"kind": "play_land"})
    # No mutation.
    assert synthetic_state.get_priority_player() == 0


def test_apply_action_play_land_wrong_type_for_card_id_raises_value_error(
    synthetic_state,
):
    """``card_id`` must be a u32-extractable integer. A string raises
    ``TypeError`` from PyO3's ``extract::<u32>`` machinery (Python's
    standard mapping for failed numeric extraction)."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {"kind": "play_land", "card_id": "not-an-int"}
        )
    # No mutation.
    assert synthetic_state.get_priority_player() == 0


def test_apply_action_play_land_on_non_land_is_a_silent_no_op(synthetic_state):
    """The ``play_land`` primitive returns ``false`` (no raise) when
    the target card isn't a land — e.g. Grizzly Bears
    (card_types = ["Creature"]). ``apply_action`` discards the bool
    return; the test pins that no state mutation occurs and no
    exception escapes."""
    bears_id = P0_FIRST_BEARS_ID
    assert synthetic_state.printing_name_of(bears_id) == "Grizzly Bears"
    # Yank the Bears off the battlefield and into hand.
    assert synthetic_state.move_object(bears_id, ZONE_HAND, 0) is True
    hand_before = synthetic_state.hand_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]

    # No raise — the underlying primitive returns false silently.
    synthetic_state.apply_action({"kind": "play_land", "card_id": bears_id})

    # Bears stayed in hand; no battlefield growth.
    assert synthetic_state.find_object_zone(bears_id) == ZONE_HAND
    assert synthetic_state.hand_sizes[0] == hand_before
    assert synthetic_state.battlefield_sizes[0] == bf_before


def test_apply_action_play_land_on_card_not_in_hand_is_a_silent_no_op(
    synthetic_state,
):
    """The ``play_land`` primitive requires the card to be in the
    actor's hand; a card already on the battlefield (or any id not
    in hand) gets a ``false`` return with no mutation. The dispatch
    propagates the same shape — no exception, no state change."""
    forest_id = P0_FIRST_BATTLEFIELD_ID  # already on the battlefield
    assert synthetic_state.printing_name_of(forest_id) == "Forest"
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    hand_before = synthetic_state.hand_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]

    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_id})

    # Forest stayed on the battlefield, hand untouched.
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert synthetic_state.hand_sizes[0] == hand_before
    assert synthetic_state.battlefield_sizes[0] == bf_before


def test_apply_action_play_land_twice_in_a_row_both_succeed(synthetic_state):
    """The ``play_land`` primitive does not enforce one-land-per-turn
    (per the round-4 module docstring — that's the caller's job via
    ``legal_decisions``). ``apply_action`` mustn't add the gate
    either: two consecutive PlayLand dispatches both succeed,
    producing two new lands on the battlefield. A future
    legal_decisions slice will filter the second call out at the
    decision-enumeration layer; the dispatcher stays dumb."""
    forest_a = _move_forest_into_hand(synthetic_state, 0)
    # Pull a second Forest into hand from the library.
    forest_b = P0_FIRST_LIBRARY_ID + 1
    assert synthetic_state.printing_name_of(forest_b) == "Forest"
    assert synthetic_state.move_object(forest_b, ZONE_HAND, 0) is True

    bf_before = synthetic_state.battlefield_sizes[0]

    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_a})
    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_b})

    assert synthetic_state.find_object_zone(forest_a) == ZONE_BATTLEFIELD
    assert synthetic_state.find_object_zone(forest_b) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before + 2


def test_apply_action_unknown_kind_message_mentions_play_land(synthetic_state):
    """The unknown-kind error message lists the valid kinds. Round 23
    extends the list to include ``play_land``. Pin the substring so a
    future variant addition forces the message to grow."""
    with pytest.raises(ValueError, match=r"play_land"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- DeclareAttackers ------------------------------------------------
#
# Round-23 slice (Phase 3, batch 2): extend the Action enum +
# apply_action dispatcher with ``Action::DeclareAttackers {
# attacker_ids }``. The Python dataclass ``DeclareAttackers(
# attacker_ids: frozenset[int])`` carries a frozenset; the bridge
# serializes it to a list at the FFI boundary. Rust stores it as
# ``Vec<u32>`` to preserve iteration order (the dispatch loop calls
# ``declare_attacker(obj_id)`` per id, and tests want deterministic
# application order).
#
# Synthetic-state reminder (cf. ``tests_rust/test_combat.py``):
# - Player 0 battlefield: ids 0..4 Forest (non-creature, P=None),
#   5..9 Grizzly Bears (2/2 creature, not summoning-sick — seeded
#   onto the battlefield, not cast).
# - Player 0 hand: ids 10..16, all Llanowar Elves (1/1).
# - Player 1 battlefield: ids 67..71 Forest, 72..76 Grizzly Bears.

P0_FIRST_BEAR_ID = 5  # 5..9 are 2/2 Grizzly Bears, not summoning-sick


def test_apply_action_declare_attackers_empty_set_no_op(synthetic_state):
    """Empty ``attacker_ids`` set: the loop body never runs and
    ``state.attackers`` stays empty. No error raised. Locks in that
    the dispatcher tolerates the zero-attackers case (Python's
    ``DeclareAttackers()`` default is ``frozenset()``)."""
    assert synthetic_state.get_attackers() == []
    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": []}
    )
    assert synthetic_state.get_attackers() == []


def test_apply_action_declare_attackers_single_attacker(synthetic_state):
    """Single attacker id: one creature declares; ``state.attackers``
    contains that id; the creature is tapped (per the round-5
    primitive's tap-on-declare semantics)."""
    bear_id = P0_FIRST_BEAR_ID
    assert synthetic_state.is_tapped(bear_id) is False

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [bear_id]}
    )

    assert synthetic_state.get_attackers() == [bear_id]
    assert synthetic_state.is_tapped(bear_id) is True


def test_apply_action_declare_attackers_multiple(synthetic_state):
    """Multiple attacker ids: every eligible id is declared and the
    final ``state.attackers`` reflects all of them in dispatch order."""
    ids = [P0_FIRST_BEAR_ID, P0_FIRST_BEAR_ID + 1, P0_FIRST_BEAR_ID + 2]

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": ids}
    )

    assert synthetic_state.get_attackers() == ids
    for bear_id in ids:
        assert synthetic_state.is_tapped(bear_id) is True


def test_apply_action_declare_attackers_iteration_order_preserved(synthetic_state):
    """Passing the ids in a non-sorted order — [bear+2, bear, bear+1]
    — must produce a ``state.attackers`` list in the SAME order
    (Vec iteration is stable, not sorted). Locks in the deterministic
    application order — callers that want sorted dispatch sort
    BEFORE passing; the dispatcher doesn't impose an order."""
    a = P0_FIRST_BEAR_ID + 2
    b = P0_FIRST_BEAR_ID
    c = P0_FIRST_BEAR_ID + 1

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [a, b, c]}
    )

    # NOT sorted — exactly the input order.
    assert synthetic_state.get_attackers() == [a, b, c]


def test_apply_action_declare_attackers_missing_key_raises_key_error(synthetic_state):
    """Missing the ``attacker_ids`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``card_id``-missing error shape on
    ``play_land``)."""
    with pytest.raises(KeyError, match="attacker_ids"):
        synthetic_state.apply_action({"kind": "declare_attackers"})
    # No mutation.
    assert synthetic_state.get_attackers() == []


def test_apply_action_declare_attackers_accepts_python_list(synthetic_state):
    """``attacker_ids`` extracts as ``Vec<u32>`` from any Python
    iterable. A plain ``list`` works (this is the bridge's canonical
    on-the-wire shape — the dict serialization at the bridge
    boundary converts the dataclass's frozenset to a list)."""
    bear_id = P0_FIRST_BEAR_ID
    # list literal — the standard shape.
    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [bear_id]}
    )
    assert synthetic_state.get_attackers() == [bear_id]


def test_apply_action_declare_attackers_accepts_frozenset(synthetic_state):
    """``attacker_ids`` also works when passed as a Python
    ``frozenset`` — same shape as Python's dataclass field. PyO3's
    ``extract::<Vec<u32>>()`` accepts any iterable, so the bridge
    doesn't have to be the only on-the-wire shape. Order is not
    asserted because frozenset has no defined iteration order;
    membership is enough."""
    bear_id = P0_FIRST_BEAR_ID
    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": frozenset([bear_id])}
    )
    assert synthetic_state.get_attackers() == [bear_id]


def test_apply_action_declare_attackers_non_creature_silently_skipped(synthetic_state):
    """Non-creature ids (e.g. a Forest at id 0) are silently rejected
    by the underlying ``declare_attacker`` primitive (returns False).
    The dispatch discards the bool — no exception escapes, no entry
    added to ``state.attackers``. Eligible ids in the same call still
    get through."""
    forest_id = 0  # P0 Forest — not a creature
    bear_id = P0_FIRST_BEAR_ID
    assert synthetic_state.printing_name_of(forest_id) == "Forest"

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [forest_id, bear_id]}
    )

    # Only the eligible id landed on the attackers list.
    assert synthetic_state.get_attackers() == [bear_id]
    # Forest stayed untapped — no side effect from the rejected call.
    assert synthetic_state.is_tapped(forest_id) is False


def test_apply_action_declare_attackers_already_tapped_silently_skipped(
    synthetic_state,
):
    """An already-tapped creature in the set is silently rejected
    (the round-5 primitive's "must be untapped to attack" gate). The
    dispatch propagates the False return as a silent skip; remaining
    untapped creatures in the set still get declared."""
    tapped_bear = P0_FIRST_BEAR_ID
    fresh_bear = P0_FIRST_BEAR_ID + 1
    # Tap the first bear before the dispatch.
    assert synthetic_state.tap_object(tapped_bear) is True

    synthetic_state.apply_action(
        {
            "kind": "declare_attackers",
            "attacker_ids": [tapped_bear, fresh_bear],
        }
    )

    # Tapped bear skipped; fresh bear declared.
    assert synthetic_state.get_attackers() == [fresh_bear]
    assert synthetic_state.is_tapped(fresh_bear) is True


def test_apply_action_declare_attackers_wrong_inner_type_raises(synthetic_state):
    """A non-integer in the ``attacker_ids`` iterable fails PyO3's
    ``extract::<u32>`` machinery — ``TypeError`` propagates. The
    dispatch never runs; ``state.attackers`` stays empty."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {
                "kind": "declare_attackers",
                "attacker_ids": ["not-an-int"],
            }
        )
    # No mutation.
    assert synthetic_state.get_attackers() == []


def test_apply_action_unknown_kind_message_mentions_declare_attackers(synthetic_state):
    """The unknown-kind error message includes ``declare_attackers``
    after this slice extends the variant list. Pin the substring so
    a future variant addition forces the message to grow."""
    with pytest.raises(ValueError, match=r"declare_attackers"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- DeclareBlockers -------------------------------------------------
#
# Round-23 slice (Phase 3, batch 3): extend the Action enum +
# apply_action dispatcher with ``Action::DeclareBlockers {
# assignments }``. The Python dataclass ``DeclareBlockers(
# assignments: tuple[tuple[int, int], ...])`` carries a tuple of
# ``(blocker_id, attacker_id)`` pairs; Rust stores it as
# ``Vec<(u32, u32)>`` to preserve iteration order (the dispatch loop
# calls ``declare_blocker(blocker_id, attacker_id)`` per pair, and
# damage assignment when multiple blockers block one attacker
# depends on input order — the first declared blocker eats the
# attacker's power under the round-5 single-blocker damage model).
#
# Pair-shape note: the action's pair is ``(blocker_id, attacker_id)``
# (matches the underlying ``declare_blocker(blocker_id, attacker_id)``
# primitive signature). ``state.blockers`` STORES pairs in the
# opposite order — ``(attacker_id, blocker_id)`` — for fast
# attacker-keyed lookups during damage assignment. The primitive
# handles the storage reorder; ``apply_action`` passes the pair
# straight through to the primitive.
#
# Synthetic-state reminder (cf. ``tests_rust/test_combat.py``):
# - Player 0 battlefield: ids 0..4 Forest, 5..9 Grizzly Bears.
# - Player 1 battlefield: ids 67..71 Forest, 72..76 Grizzly Bears.
# A declare-blockers test needs a declared attacker first — call
# ``declare_attacker`` on a P0 bear, then declare a P1 bear as the
# blocker.

P1_FIRST_FOREST_ID = 67  # p1 battlefield 67..71 are Forest (non-creature)
P1_FIRST_BLOCKER_ID = 72  # p1 battlefield 72..76 are 2/2 Grizzly Bears


def _seed_one_attacker(state, attacker_id=P0_FIRST_BEAR_ID):
    """Declare one P0 bear as an attacker so blocker tests have a
    legal attacker to reference. Returns the attacker id."""
    assert state.declare_attacker(attacker_id) is True
    return attacker_id


def test_apply_action_declare_blockers_empty_no_op(synthetic_state):
    """Empty ``assignments`` list: the loop body never runs and
    ``state.blockers`` stays empty. No error raised. Locks in that
    the dispatcher tolerates the zero-blockers case (Python's
    ``DeclareBlockers()`` default is the empty tuple)."""
    _seed_one_attacker(synthetic_state)
    assert synthetic_state.get_blockers() == []

    synthetic_state.apply_action(
        {"kind": "declare_blockers", "assignments": []}
    )

    assert synthetic_state.get_blockers() == []


def test_apply_action_declare_blockers_single_pair(synthetic_state):
    """A single ``(blocker_id, attacker_id)`` pair: one block lands
    on ``state.blockers`` as ``(attacker_id, blocker_id)`` (storage
    order is the opposite of the action's input order — the
    primitive handles the reorder)."""
    attacker = _seed_one_attacker(synthetic_state)
    blocker = P1_FIRST_BLOCKER_ID

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            "assignments": [(blocker, attacker)],
        }
    )

    assert synthetic_state.get_blockers() == [(attacker, blocker)]


def test_apply_action_declare_blockers_multiple_pairs_single_blocker_each(
    synthetic_state,
):
    """Multiple pairs, each a distinct blocker on a distinct
    attacker: all recorded. Two P0 bears attack, two P1 bears block
    one each — ``state.blockers`` has both pairs in input order."""
    a0 = P0_FIRST_BEAR_ID
    a1 = P0_FIRST_BEAR_ID + 1
    assert synthetic_state.declare_attacker(a0) is True
    assert synthetic_state.declare_attacker(a1) is True
    b0 = P1_FIRST_BLOCKER_ID
    b1 = P1_FIRST_BLOCKER_ID + 1

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            "assignments": [(b0, a0), (b1, a1)],
        }
    )

    # storage shape is (attacker, blocker)
    assert synthetic_state.get_blockers() == [(a0, b0), (a1, b1)]


def test_apply_action_declare_blockers_multiple_blockers_on_one_attacker(
    synthetic_state,
):
    """Multi-block: one attacker blocked by three blockers. All
    pairs land with the same ``attacker_id``; blocker order matches
    the input order — load-bearing for damage assignment when the
    round-5 model gives the attacker's power to the first declared
    blocker."""
    attacker = _seed_one_attacker(synthetic_state)
    b0 = P1_FIRST_BLOCKER_ID
    b1 = P1_FIRST_BLOCKER_ID + 1
    b2 = P1_FIRST_BLOCKER_ID + 2

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            "assignments": [(b0, attacker), (b1, attacker), (b2, attacker)],
        }
    )

    assert synthetic_state.get_blockers() == [
        (attacker, b0),
        (attacker, b1),
        (attacker, b2),
    ]


def test_apply_action_declare_blockers_iteration_order_preserved(synthetic_state):
    """Passing pairs in a non-sorted order — [(b2, A), (b0, A),
    (b1, A)] — must produce ``state.blockers`` in the SAME input
    order. The Vec-based iteration is stable, not sorted; callers
    that want a canonical order sort BEFORE passing."""
    attacker = _seed_one_attacker(synthetic_state)
    b0 = P1_FIRST_BLOCKER_ID
    b1 = P1_FIRST_BLOCKER_ID + 1
    b2 = P1_FIRST_BLOCKER_ID + 2

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            "assignments": [(b2, attacker), (b0, attacker), (b1, attacker)],
        }
    )

    # NOT sorted by blocker id — exactly the input order.
    assert synthetic_state.get_blockers() == [
        (attacker, b2),
        (attacker, b0),
        (attacker, b1),
    ]


def test_apply_action_declare_blockers_missing_key_raises_key_error(synthetic_state):
    """Missing the ``assignments`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``attacker_ids``-missing shape on
    ``declare_attackers``)."""
    _seed_one_attacker(synthetic_state)
    with pytest.raises(KeyError, match="assignments"):
        synthetic_state.apply_action({"kind": "declare_blockers"})
    # No mutation.
    assert synthetic_state.get_blockers() == []


def test_apply_action_declare_blockers_accepts_list_of_lists(synthetic_state):
    """``assignments`` extracts as ``Vec<(u32, u32)>`` from any
    iterable-of-iterables. A plain ``list`` of 2-element ``list``s
    works — PyO3's tuple-extract accepts both Python tuples and
    fixed-length lists. The Python dataclass field is a tuple of
    tuples, but the JSON-like wire shape often uses lists; this
    test pins the relaxed acceptance."""
    attacker = _seed_one_attacker(synthetic_state)
    blocker = P1_FIRST_BLOCKER_ID

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            # list of lists, not list of tuples
            "assignments": [[blocker, attacker]],
        }
    )

    assert synthetic_state.get_blockers() == [(attacker, blocker)]


def test_apply_action_declare_blockers_non_creature_blocker_silently_skipped(
    synthetic_state,
):
    """A non-creature blocker (e.g. a Forest at id 67) is silently
    rejected by the underlying ``declare_blocker`` primitive
    (returns False). The dispatch discards the bool — no exception
    escapes, no entry added to ``state.blockers``. Eligible pairs
    in the same call still get through."""
    attacker = _seed_one_attacker(synthetic_state)
    forest_blocker = P1_FIRST_FOREST_ID
    bear_blocker = P1_FIRST_BLOCKER_ID
    assert synthetic_state.printing_name_of(forest_blocker) == "Forest"

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            "assignments": [
                (forest_blocker, attacker),
                (bear_blocker, attacker),
            ],
        }
    )

    # Only the eligible pair landed.
    assert synthetic_state.get_blockers() == [(attacker, bear_blocker)]


def test_apply_action_declare_blockers_tapped_blocker_silently_skipped(
    synthetic_state,
):
    """A tapped blocker is silently rejected (the round-5
    primitive's "blocker must be untapped" gate). Remaining
    untapped blockers in the same call still go through."""
    attacker = _seed_one_attacker(synthetic_state)
    tapped_blocker = P1_FIRST_BLOCKER_ID
    fresh_blocker = P1_FIRST_BLOCKER_ID + 1
    # Tap the first blocker before the dispatch.
    assert synthetic_state.tap_object(tapped_blocker) is True

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            "assignments": [
                (tapped_blocker, attacker),
                (fresh_blocker, attacker),
            ],
        }
    )

    # Tapped blocker skipped; fresh blocker landed.
    assert synthetic_state.get_blockers() == [(attacker, fresh_blocker)]


def test_apply_action_unknown_kind_message_mentions_declare_blockers(synthetic_state):
    """The unknown-kind error message includes ``declare_blockers``
    after this slice extends the variant list. Pin the substring so
    a future variant addition forces the message to grow."""
    with pytest.raises(ValueError, match=r"declare_blockers"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- ActivateAbility -------------------------------------------------
#
# Round-23 slice (Phase 3, batch 4): extended the Action enum +
# apply_action dispatcher with ``Action::ActivateAbility { source_id,
# ability_index }``. Round-24 (`feature/rust-legal-decisions-activate-
# ability`) wired the legal_decisions emission for the same shape
# (untapped permanents whose printing's mana_produced.is_some(), with
# ability_index=0). **Round-25 (this slice) flips the dispatch from
# a "not implemented yet" stub to a real call** for the same
# mana-ability scope.
#
# The Python dataclass ``ActivateAbility(source_id: int,
# ability_index: int)`` carries two ints — the GameObject id of the
# permanent whose ability fires and the 0-based index into the
# source's printing.activated_abilities list (CR 602.2 — printings
# can have multiple activated abilities).
#
# Round-20 frequency: 8 of 420 decisions (~1.9%) — 5th-most-frequent
# variant after ``PassPriority``, ``DeclareAttackers``, ``PlayLand``,
# and ``DeclareBlockers``.
#
# **Dispatch scope: mana abilities only.** The arm now:
#   1. Validates ``ability_index == 0`` (anything else raises — non-
#      mana activated abilities need richer primitives, future slice).
#   2. Locates the source on either player's battlefield (raises if
#      not found).
#   3. Validates the printing has ``mana_produced`` (raises if not —
#      this slice doesn't support non-mana activated abilities).
#   4. Validates the source is untapped (raises if tapped — a tapped
#      permanent can't pay the implicit {T} tap cost).
#   5. Taps the source via the per-zone CoW Arc::make_mut path and
#      credits the controller's mana pool one pip per element of
#      ``mana_produced`` (W/U/B/R/G/C — saturating u8 add).
#
# Mirrors the legal_decisions emission gate exactly — the harness
# can now emit-and-apply ActivateAbility round-trips without raising.


def test_apply_action_activate_ability_mana_taps_source_and_credits_pool(
    synthetic_state,
):
    """Happy path: a well-formed ActivateAbility dict on an untapped
    Forest on the priority player's battlefield taps the Forest and
    adds one G pip to the controller's mana pool.

    Pins the post-stub success contract — the round-23 "not
    implemented yet" path is gone."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    assert synthetic_state.printing_name_of(forest_id) == "Forest"
    assert synthetic_state.is_tapped(forest_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )

    assert synthetic_state.is_tapped(forest_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1


def test_apply_action_activate_ability_returns_none(synthetic_state):
    """``apply_action`` is a side-effecting mutator — the
    ActivateAbility arm returns ``None`` per the dispatcher's overall
    convention (locked in by the PassPriority test at the top of the
    file)."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    result = synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )
    assert result is None


def test_apply_action_activate_ability_on_tapped_source_raises(synthetic_state):
    """A source that's already tapped can't have its tap-for-mana
    ability activated (CR 602.2 — paying the implicit {T} cost
    requires an untapped permanent). Raises ValueError with the
    "already tapped" substring; no state mutation."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    # Pre-tap the Forest.
    assert synthetic_state.tap_object(forest_id) is True
    assert synthetic_state.is_tapped(forest_id) is True
    pool_before = synthetic_state.get_mana_pool(0)["G"]

    with pytest.raises(ValueError, match=r"already tapped"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": forest_id,
                "ability_index": 0,
            }
        )

    # Pool unchanged — the raise short-circuits before the credit.
    assert synthetic_state.get_mana_pool(0)["G"] == pool_before


def test_apply_action_activate_ability_on_non_mana_source_raises(synthetic_state):
    """A source whose printing has no ``mana_produced`` (e.g. Grizzly
    Bears) can't be tap-for-mana'd. Raises ValueError with the "no
    mana_produced" substring; no state mutation. Locks in the
    first-slice scope — non-mana activated abilities surface loudly
    rather than silently no-op'ing."""
    bears_id = P0_FIRST_BEARS_ID
    assert synthetic_state.printing_name_of(bears_id) == "Grizzly Bears"
    assert synthetic_state.is_tapped(bears_id) is False

    with pytest.raises(ValueError, match=r"no mana_produced"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": bears_id,
                "ability_index": 0,
            }
        )

    # Bears stayed untapped — the raise short-circuits before any
    # tap mutation.
    assert synthetic_state.is_tapped(bears_id) is False


def test_apply_action_activate_ability_on_source_not_on_battlefield_raises(
    synthetic_state,
):
    """A source id that isn't on either player's battlefield raises
    ValueError with the "not on any battlefield" substring. A hand
    object (e.g. one of the Llanowar Elves at id 10..16) is a clean
    no-match — it lives in the hand zone, not the battlefield zone.
    No state mutation."""
    hand_id = P0_FIRST_HAND_ID  # in hand, not on battlefield
    assert synthetic_state.find_object_zone(hand_id) == ZONE_HAND

    with pytest.raises(ValueError, match=r"not on any battlefield"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": hand_id,
                "ability_index": 0,
            }
        )


def test_apply_action_activate_ability_with_unknown_source_id_raises(
    synthetic_state,
):
    """A completely unknown source id (never assigned to any object)
    also takes the "not on any battlefield" path — the search is by
    id presence, not by object existence elsewhere. Pins that the
    error message is shape-invariant."""
    with pytest.raises(ValueError, match=r"not on any battlefield"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": 9999,  # no object has this id
                "ability_index": 0,
            }
        )


def test_apply_action_activate_ability_ability_index_one_raises(synthetic_state):
    """First-slice scope: only ``ability_index = 0`` is dispatched.
    Non-zero indices raise ValueError flagging that non-mana
    activated abilities are a future slice — the harness picks the
    raise up loudly rather than silently no-op'ing on an out-of-scope
    Action."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    with pytest.raises(ValueError, match=r"only ability_index=0"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": forest_id,
                "ability_index": 1,
            }
        )
    # Forest stayed untapped.
    assert synthetic_state.is_tapped(forest_id) is False


def test_apply_action_activate_ability_ability_index_two_raises_too(synthetic_state):
    """Locks in the ``ability_index != 0`` raise is shape-invariant
    over the specific non-zero value (i.e. not just a `== 1`
    guard)."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    with pytest.raises(ValueError, match=r"only ability_index=0"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": forest_id,
                "ability_index": 2,
            }
        )


def test_apply_action_activate_ability_opponent_source_is_allowed_by_dispatcher(
    synthetic_state,
):
    """The dispatcher is permissive about which side controls the
    source — it credits the **controller's** pool regardless of who
    has priority. ``legal_decisions`` only emits ActivateAbility for
    the priority player's permanents, so production callers won't
    hit this path; but a direct caller building the dict by hand
    gets the same successful behavior, with the mana landing in the
    controller's pool, not the priority player's. Locks in that the
    dispatcher reads the controller from the battlefield index
    rather than assuming `priority_player`."""
    # Find p1's first Forest via the unified battlefield_snapshot
    # (returns both players' battlefields; filter by controller_id).
    all_bf = synthetic_state.battlefield_snapshot()
    p1_forest = next(
        obj
        for obj in all_bf
        if obj["controller_id"] == 1
        and synthetic_state.printing_name_of(obj["id"]) == "Forest"
    )
    p1_forest_id = p1_forest["id"]
    assert synthetic_state.is_tapped(p1_forest_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0
    assert synthetic_state.get_mana_pool(1)["G"] == 0
    # Priority player is still 0; deliberately ActivateAbility a
    # p1-controlled Forest.
    assert synthetic_state.get_priority_player() == 0

    synthetic_state.apply_action(
        {
            "kind": "activate_ability",
            "source_id": p1_forest_id,
            "ability_index": 0,
        }
    )

    # Forest tapped; mana credited to p1 (the controller), NOT p0.
    assert synthetic_state.is_tapped(p1_forest_id) is True
    assert synthetic_state.get_mana_pool(1)["G"] == 1
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_apply_action_activate_ability_two_forests_sequentially_pools_two_green(
    synthetic_state,
):
    """Mana-pool persistence across consecutive ActivateAbility
    applies: tap two Forests in sequence, the controller's green
    pool ends at 2. Pins that the dispatch arm credits **into** the
    existing pool (saturating add) rather than overwriting."""
    forest_a = P0_FIRST_BATTLEFIELD_ID
    forest_b = P0_FIRST_BATTLEFIELD_ID + 1
    assert synthetic_state.printing_name_of(forest_a) == "Forest"
    assert synthetic_state.printing_name_of(forest_b) == "Forest"
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_a, "ability_index": 0}
    )
    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_b, "ability_index": 0}
    )

    assert synthetic_state.is_tapped(forest_a) is True
    assert synthetic_state.is_tapped(forest_b) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 2


def test_apply_action_activate_ability_clone_isolation(synthetic_state):
    """CoW canary via the ActivateAbility bridge: applying an
    ActivateAbility to a clone mutates **only the clone's**
    battlefield (the source becomes tapped + the pool gains a G) —
    the original's per-zone Arc<Vec<GameObject>> stays untouched.
    Locks in that the dispatcher's Arc::make_mut path on the
    battlefield zone honors the per-zone CoW invariant
    (state.players[N].battlefield is Arc<Vec<GameObject>>; mutating
    via Arc::make_mut clones the inner Vec when any other clone is
    observing it)."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    cloned = synthetic_state.clone_once()
    assert synthetic_state.is_tapped(forest_id) is False
    assert cloned.is_tapped(forest_id) is False

    cloned.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )

    # Clone mutated.
    assert cloned.is_tapped(forest_id) is True
    assert cloned.get_mana_pool(0)["G"] == 1
    # Original untouched — the per-zone Arc make_mut cloned, not shared.
    assert synthetic_state.is_tapped(forest_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_apply_action_activate_ability_bridge_roundtrip_via_legal_decisions(
    synthetic_state,
):
    """End-to-end harness round-trip: emit an ActivateAbility dict
    via ``legal_decisions``, pass it directly to ``apply_action``,
    verify state mutates. Pins that the dict shape
    legal_decisions emits (round 24) is exactly what apply_action
    consumes (round 25) — no key drift between emit and apply.

    legal_decisions only emits ActivateAbility under the sorcery-
    timing gate (main phase + empty stack — see test_legal.py's
    "off_sorcery_timing" / "nonempty_stack" suppression tests).
    The synthetic state's default phase is UNTAP, so we set
    PHASE_MAIN1=3 first; the priority player is already 0 with 5
    Forests on their battlefield."""
    PHASE_MAIN1 = 3
    synthetic_state.set_phase(PHASE_MAIN1)
    actions = synthetic_state.legal_decisions(0)
    # Find the first ActivateAbility action emitted.
    aa = next(a for a in actions if a.get("kind") == "activate_ability")
    source_id = aa["source_id"]
    assert aa["ability_index"] == 0
    assert synthetic_state.is_tapped(source_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    # Round-trip: apply the dict legal_decisions just emitted.
    synthetic_state.apply_action(aa)

    assert synthetic_state.is_tapped(source_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1


def test_apply_action_activate_ability_missing_source_id_raises_key_error(
    synthetic_state,
):
    """Missing the ``source_id`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``card_id``-missing shape on
    ``play_land``). The KeyError fires BEFORE any dispatch path, so
    this test pins the parse-time gate."""
    with pytest.raises(KeyError, match="source_id"):
        synthetic_state.apply_action(
            {"kind": "activate_ability", "ability_index": 0}
        )


def test_apply_action_activate_ability_missing_ability_index_raises_key_error(
    synthetic_state,
):
    """Missing the ``ability_index`` key raises ``KeyError`` — same
    parse-time gate as ``source_id``. Both keys are required; the
    parser reads them in order, so this test specifically exercises
    the second key's check (source_id present, ability_index
    missing)."""
    with pytest.raises(KeyError, match="ability_index"):
        synthetic_state.apply_action(
            {"kind": "activate_ability", "source_id": 5}
        )


def test_apply_action_activate_ability_wrong_type_for_source_id_raises(
    synthetic_state,
):
    """``source_id`` must be a u32-extractable integer. A string fails
    PyO3's ``extract::<u32>`` machinery → ``TypeError`` / ``ValueError``.
    Dispatch never runs; the parse-time gate fires first."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": "not-an-int",
                "ability_index": 0,
            }
        )


def test_apply_action_activate_ability_wrong_type_for_ability_index_raises(
    synthetic_state,
):
    """``ability_index`` must be a u32-extractable integer. A string
    fails the parse-time extract — ``TypeError`` / ``ValueError``
    before any dispatch happens."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": 5,
                "ability_index": "not-an-int",
            }
        )


def test_apply_action_unknown_kind_message_mentions_activate_ability(synthetic_state):
    """The unknown-kind error message includes ``activate_ability``
    after this slice extends the variant list. Pin the substring so
    a future variant addition forces the message to grow."""
    with pytest.raises(ValueError, match=r"activate_ability"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- CastSpell -------------------------------------------------------
#
# Round-23+ slice (Phase 3, batch 5): extend the Action enum +
# apply_action dispatcher with ``Action::CastSpell { card_id, use_warp,
# kicked, cast_for_free, use_impending, cast_room_half }``. The Python
# dataclass ``CastSpell(card_id: int, use_warp: bool = False, kicked:
# bool = False, cast_for_free: bool = False, use_impending: bool =
# False, cast_room_half: str | None = None)`` mirrors the Rust variant
# bit-for-bit so production CastSpell payloads round-trip losslessly
# through the bridge.
#
# Round-20 frequency: 17 of 420 decisions (~4.0%) — the
# second-most-frequent non-``PassPriority`` variant after
# ``DeclareAttackers``.
#
# **First-slice scope: flags are data-only.** The dispatch arm calls
# the existing round-4 ``cast_creature(card_id, player_id)`` primitive
# and IGNORES every flag value. Mirrors the round-23 ActivateAbility
# "scaffold the variant, defer the dispatch semantics" pattern.
# Future slices wire warp / kicker / impending / cast_for_free /
# cast_room_half through to richer primitives.
#
# Synthetic-state mana reminder: ``cast_creature`` pays the printing's
# mana_cost — Llanowar Elves costs ``{G}``, so the priority player
# needs at least 1 green pip in their mana pool to land a happy-path
# cast. Use ``synthetic_state.add_mana(player, "G", n)`` to prep.


def test_apply_action_cast_spell_minimal_dispatches_to_cast_creature(synthetic_state):
    """Happy path with a minimal dict (only ``card_id``, no flag
    keys): the Llanowar Elf in the priority player's hand pays its
    {G} cost and lands on the battlefield. Pins the
    ``apply_action({"kind": "cast_spell", "card_id": N})`` shape as
    the canonical minimal payload — equivalent to Python's
    ``CastSpell(card_id=N)`` call with all flags defaulted."""
    elf_id = P0_FIRST_HAND_ID
    assert synthetic_state.printing_name_of(elf_id) == "Llanowar Elves"
    synthetic_state.add_mana(0, "G", 1)
    hand_before = synthetic_state.hand_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]

    synthetic_state.apply_action({"kind": "cast_spell", "card_id": elf_id})

    # Elf landed; one G pip spent.
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    assert synthetic_state.hand_sizes[0] == hand_before - 1
    assert synthetic_state.battlefield_sizes[0] == bf_before + 1
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_apply_action_cast_spell_uses_priority_player_as_actor(synthetic_state):
    """The actor is implicit (priority player). Flip priority to
    player 1 and cast from player 1's hand — Locks in that the
    dispatcher reads ``priority_player`` rather than a hardcoded 0
    (mirrors the ``play_land`` actor-resolution convention)."""
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 1
    # Player 1 also has Llanowar Elves in hand at ids 77..83 in the
    # synthetic seed (10..16 are P0's elves; P1's hand starts at 77
    # = id_offset 67 + 10). Pull the first hand object's id from the
    # snapshot (each snapshot entry is a dict with an "id" key).
    p1_hand = synthetic_state.hand_snapshot(1)
    elf_id = p1_hand[0]["id"]
    assert synthetic_state.printing_name_of(elf_id) == "Llanowar Elves"
    synthetic_state.add_mana(1, "G", 1)
    p1_bf_before = synthetic_state.battlefield_sizes[1]

    synthetic_state.apply_action({"kind": "cast_spell", "card_id": elf_id})

    # Elf landed on player 1's battlefield, not player 0's.
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[1] == p1_bf_before + 1


def test_apply_action_cast_spell_with_all_flags_false_explicit(synthetic_state):
    """Explicit-defaults shape — every flag spelled out with its
    default value. Behavior must be bit-identical to the minimal
    shape (the flags are silently ignored by first-slice dispatch).
    Locks in the "round-trip" property: a dict that ``action_to_pydict``
    would emit (always all 6 keys) parses cleanly back through
    ``parse_action``."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    synthetic_state.apply_action(
        {
            "kind": "cast_spell",
            "card_id": elf_id,
            "use_warp": False,
            "kicked": False,
            "cast_for_free": False,
            "use_impending": False,
            "cast_room_half": None,
        }
    )

    # Same outcome as the minimal-shape test.
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_apply_action_cast_spell_use_warp_true_dispatches_unchanged(synthetic_state):
    """First-slice contract: ``use_warp=True`` is parsed as data but
    has NO observable effect on the dispatch. The cast still routes
    to ``cast_creature(card_id, priority_player)`` and pays the
    base mana cost (no exile-zone alt-cost, no end-of-turn exile).
    Pins the flag-ignored convention so a future warp-plumbing slice
    has a clear baseline to diff against."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    synthetic_state.apply_action(
        {"kind": "cast_spell", "card_id": elf_id, "use_warp": True}
    )

    # Same battlefield landing as the unflagged case — flag was ignored.
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_apply_action_cast_spell_kicked_true_dispatches_unchanged(synthetic_state):
    """First-slice contract: ``kicked=True`` is parsed as data but
    has NO observable effect — the base mana cost is paid, no
    additional kicker cost is debited. A future cost-stacking slice
    will wire the kicker pip through."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)
    # Pin: only ONE G pip — if the kicker were paid, the cast would
    # fail for insufficient mana. The flag-ignored contract makes
    # the cast succeed.
    synthetic_state.apply_action(
        {"kind": "cast_spell", "card_id": elf_id, "kicked": True}
    )
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_apply_action_cast_spell_cast_for_free_true_still_pays_mana(synthetic_state):
    """First-slice contract: ``cast_for_free=True`` is parsed as data
    but has NO observable effect — the base mana cost is STILL paid
    (the flag would normally bypass the mana payment, but the
    primitive can't honor that yet). A cast with the flag set but
    NO mana in pool still fails the same way an unflagged cast
    would. A future slice plumbing the flag will flip this test."""
    elf_id = P0_FIRST_HAND_ID
    # Pool is empty; flag would normally bypass payment.
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    # No exception — cast_creature returns False silently when mana
    # is insufficient; apply_action discards the bool.
    synthetic_state.apply_action(
        {"kind": "cast_spell", "card_id": elf_id, "cast_for_free": True}
    )

    # Cast did not happen — elf still in hand. Pins that
    # cast_for_free was ignored (flag's purpose would be to make
    # this succeed with empty pool).
    assert synthetic_state.find_object_zone(elf_id) == ZONE_HAND


def test_apply_action_cast_spell_use_impending_true_dispatches_unchanged(synthetic_state):
    """First-slice contract: ``use_impending=True`` is parsed as data
    but has NO observable effect — the creature lands as a normal
    creature (no impending counters, no creature-ness suppression).
    A future Effect dispatch slice will wire the impending mode."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    synthetic_state.apply_action(
        {"kind": "cast_spell", "card_id": elf_id, "use_impending": True}
    )

    # Lands as a normal creature (the elf would otherwise enter with
    # time counters under real impending semantics — but the flag is
    # ignored, so it's just a regular ETB).
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD


def test_apply_action_cast_spell_cast_room_half_string_preserved(synthetic_state):
    """``cast_room_half`` is ``Option<String>`` on the Rust side —
    a string value parses cleanly (no TypeError) and has no
    observable effect on dispatch (the flag is data-only). Pins
    that the Option<String> extract path handles a non-None string
    correctly (different code path from the bool flag extracts)."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    synthetic_state.apply_action(
        {
            "kind": "cast_spell",
            "card_id": elf_id,
            "cast_room_half": "left",
        }
    )

    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD


def test_apply_action_cast_spell_cast_room_half_none_parses(synthetic_state):
    """Explicit ``cast_room_half=None`` parses cleanly — same effect
    as omitting the key (the parser maps both to the Rust
    ``Option<String>::None``)."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    synthetic_state.apply_action(
        {
            "kind": "cast_spell",
            "card_id": elf_id,
            "cast_room_half": None,
        }
    )

    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD


def test_apply_action_cast_spell_missing_card_id_raises_key_error(synthetic_state):
    """Missing the required ``card_id`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``card_id``-missing shape on
    ``play_land``)."""
    with pytest.raises(KeyError, match="card_id"):
        synthetic_state.apply_action({"kind": "cast_spell"})


def test_apply_action_cast_spell_missing_flags_default_silently(synthetic_state):
    """Every flag is OPTIONAL — missing flag keys default to the
    Python dataclass defaults (False / None) silently, no KeyError.
    The minimal `{kind, card_id}` payload is the most common
    on-the-wire shape from the search loop, and it MUST work
    without spelling every flag out."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    # No flag keys at all. Must not raise.
    synthetic_state.apply_action({"kind": "cast_spell", "card_id": elf_id})

    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD


def test_apply_action_cast_spell_wrong_type_for_card_id_raises(synthetic_state):
    """``card_id`` must be a u32-extractable integer. A string fails
    PyO3's ``extract::<u32>`` machinery → ``TypeError`` / ``ValueError``
    before any dispatch happens (matches the ``play_land`` shape)."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {"kind": "cast_spell", "card_id": "not-an-int"}
        )


def test_apply_action_cast_spell_wrong_type_for_use_warp_raises(synthetic_state):
    """``use_warp`` must extract as a bool. A non-bool, non-int value
    like a list fails PyO3's ``extract::<bool>`` machinery →
    ``TypeError`` / ``ValueError``. Pins the type gate for one of the
    flag fields so future-slice flag wiring has a baseline to keep."""
    elf_id = P0_FIRST_HAND_ID
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {"kind": "cast_spell", "card_id": elf_id, "use_warp": [1, 2]}
        )


def test_apply_action_cast_spell_non_creature_in_hand_is_silent_no_op(
    synthetic_state,
):
    """The ``cast_creature`` primitive returns ``False`` silently
    when the target's printing isn't a Creature (the primitive's
    type-gate). ``apply_action`` discards the bool — no exception
    escapes; the test pins that flags-set-true on a non-Creature
    still does nothing (the gate fires before the flags would
    matter anyway)."""
    bears_id = P0_FIRST_BEARS_ID
    # Move bears to hand so the in-hand precondition passes; the
    # printing IS a creature though, so to test a non-creature we'd
    # need a non-creature in hand — the synthetic seed doesn't
    # have one. Instead, pin the wrong-zone failure path: bears
    # ON the battlefield can't be cast (not in hand).
    assert synthetic_state.find_object_zone(bears_id) == ZONE_BATTLEFIELD
    bf_before = synthetic_state.battlefield_sizes[0]

    # No raise — primitive returns False silently.
    synthetic_state.apply_action(
        {"kind": "cast_spell", "card_id": bears_id, "use_warp": True}
    )

    # No mutation.
    assert synthetic_state.find_object_zone(bears_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before


def test_apply_action_unknown_kind_message_mentions_cast_spell(synthetic_state):
    """The unknown-kind error message includes ``cast_spell`` after
    this slice extends the variant list. Pin the substring so a
    future variant addition forces the message to grow."""
    with pytest.raises(ValueError, match=r"cast_spell"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- TurnFaceUp ------------------------------------------------------
#
# Rolling-integration slice (post-round-23): extend the Action enum +
# apply_action dispatcher with ``Action::TurnFaceUp { card_id }``. The
# Python dataclass ``TurnFaceUp(card_id: int)`` carries a single int —
# the GameObject id of the face-down permanent on the priority
# player's battlefield to flip face up. Mirrors ``PlayLand``'s
# single-int shape exactly.
#
# CR 701.34 / 707.4 — Morph / Manifest Dread flip + Edge of Eternities
# Spacecraft station / launch. The EoE Spacecraft cycle is the
# production driver: Spacecraft enter face-down (as artifacts with no
# creature types) and accumulate charge counters via station-attaching;
# they turn face up (become creatures with their printed stats) when
# launched by paying a face-up cost. CR 701.34 is a special action —
# does not use the stack; costs are paid immediately and the flip
# resolves at once.
#
# **Dispatch is stubbed.** No Rust face-down GameObject state exists
# yet (the ``GameObject`` struct has no ``face_down`` flag today, and
# there's no Spacecraft charge-counter resolution primitive), so the
# variant + bridge round-trip ship now but the dispatch arm raises
# ``ValueError`` with a "not implemented yet" message. A future slice
# adds the per-object face-down state field + flip primitive
# (analogous to how ``tap`` / ``untap`` toggle the ``tapped`` flag)
# and flips this arm to a real ``call_method1("turn_face_up", ...)``
# dispatch. Until then, happy-path dispatch raises ``ValueError`` —
# these tests assert on the stubbed error path, NOT on a successful
# flip outcome. Mirrors the round-23 ActivateAbility "scaffold the
# variant, defer the dispatch semantics" pattern.


def test_apply_action_turn_face_up_stub_raises_value_error(synthetic_state):
    """Happy-path stub: a well-formed TurnFaceUp dict parses cleanly
    but the dispatch arm raises ``ValueError`` because no Rust face-
    down GameObject state / flip primitive exists yet. The error
    message names the variant + flags it as future work so a follow-
    up slice grepping for the string finds the dispatch site to
    flip."""
    with pytest.raises(ValueError, match=r"TurnFaceUp dispatch not implemented yet"):
        synthetic_state.apply_action({"kind": "turn_face_up", "card_id": 42})


def test_apply_action_turn_face_up_stub_error_mentions_spacecraft(synthetic_state):
    """The stub message names the missing Rust primitives (face-down
    GameObject state + Spacecraft) so a future slice grepping for
    "Spacecraft" finds the dispatch site to flip. Pin the substring
    so the message stays informative — a future slice that flips the
    dispatch to a real call must also delete this test (it documents
    the stub-state, not the eventual real behavior)."""
    with pytest.raises(ValueError, match=r"Spacecraft"):
        synthetic_state.apply_action({"kind": "turn_face_up", "card_id": 7})


def test_apply_action_turn_face_up_missing_card_id_raises_key_error(
    synthetic_state,
):
    """Missing the ``card_id`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``card_id``-missing shape on
    ``play_land``). The KeyError fires BEFORE the dispatch-stub
    ValueError, so this test pins the parse-time gate."""
    with pytest.raises(KeyError, match="card_id"):
        synthetic_state.apply_action({"kind": "turn_face_up"})


def test_apply_action_turn_face_up_wrong_type_for_card_id_raises(
    synthetic_state,
):
    """``card_id`` must be a u32-extractable integer. A string fails
    PyO3's ``extract::<u32>`` machinery → ``TypeError`` / ``ValueError``.
    The dispatch-stub never runs; the parse-time gate fires first."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {"kind": "turn_face_up", "card_id": "not-an-int"}
        )


def test_apply_action_unknown_kind_message_mentions_turn_face_up(synthetic_state):
    """The unknown-kind error message includes ``turn_face_up`` after
    this slice extends the variant list. Pin the substring so a future
    variant addition forces the message to grow."""
    with pytest.raises(ValueError, match=r"turn_face_up"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- UnlockRoomHalf --------------------------------------------------
#
# Rolling-integration slice (post-round-24): extend the Action enum +
# apply_action dispatcher with ``Action::UnlockRoomHalf { card_id,
# half_name }``. The Python dataclass ``UnlockRoomHalf(card_id: int,
# half_name: str)`` carries two fields — the GameObject id of the Room
# enchantment on the active player's battlefield, plus a string naming
# which of the two halves to unlock (per the Room printing).
#
# CR 715.3-4 (DSK Rooms): as a sorcery, the controller of an unlocked
# Room on the battlefield with at least one locked door may unlock a
# locked door of that Room by paying its mana cost as though casting an
# enchantment spell. The unlock action is NOT a spell — it doesn't go
# on the stack; the named half is added to ``unlocked_halves`` on the
# Room permanent on apply, and any newly-active triggered abilities on
# the entering half fire.
#
# **Dispatch is stubbed.** Rust has no Room-half state today — the
# per-object ``unlocked_halves: set[str]`` field that Python's
# ``GameObject`` carries is missing on the Rust side, and there's no
# half-specific effect activation primitive. The variant + bridge
# round-trip ship now, so callers building Action dicts can already
# use the ``unlock_room_half`` shape and hit a clear "not implemented
# yet" error path rather than the generic ``unknown action kind``
# rejection. A future slice adds the per-object Room state + the
# unlock primitive then flips the dispatch arm to a real
# ``call_method1("unlock_room_half", ...)`` call. Until then, happy-
# path dispatch raises ``ValueError`` — these tests assert on the
# stubbed error path, NOT on a successful unlock outcome. Mirrors the
# round-23 ActivateAbility and round-24 TurnFaceUp "scaffold the
# variant, defer the dispatch semantics" pattern.


def test_apply_action_unlock_room_half_stub_raises_value_error(synthetic_state):
    """Happy-path stub: a well-formed UnlockRoomHalf dict parses
    cleanly but the dispatch arm raises ``ValueError`` because no
    Rust Room-half state / unlock primitive exists yet. The error
    message names the variant + flags it as future work so a follow-
    up slice grepping for the string finds the dispatch site to
    flip."""
    with pytest.raises(
        ValueError, match=r"UnlockRoomHalf dispatch not implemented yet"
    ):
        synthetic_state.apply_action(
            {"kind": "unlock_room_half", "card_id": 42, "half_name": "left"}
        )


def test_apply_action_unlock_room_half_stub_error_mentions_room(synthetic_state):
    """The stub message names the missing Rust state (Room enchantment
    + half-name resolution) so a future slice grepping for "Room" finds
    the dispatch site to flip. Pin the substring so the message stays
    informative — a future slice that flips the dispatch to a real
    call must also delete this test (it documents the stub-state, not
    the eventual real behavior)."""
    with pytest.raises(ValueError, match=r"Room"):
        synthetic_state.apply_action(
            {"kind": "unlock_room_half", "card_id": 7, "half_name": "right"}
        )


def test_apply_action_unlock_room_half_missing_card_id_raises_key_error(
    synthetic_state,
):
    """Missing the ``card_id`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``card_id``-missing shape on
    ``play_land``). The KeyError fires BEFORE the dispatch-stub
    ValueError, so this test pins the parse-time gate."""
    with pytest.raises(KeyError, match="card_id"):
        synthetic_state.apply_action(
            {"kind": "unlock_room_half", "half_name": "left"}
        )


def test_apply_action_unlock_room_half_missing_half_name_raises_key_error(
    synthetic_state,
):
    """Missing the ``half_name`` key raises ``KeyError`` — same parse-
    time gate as ``card_id``. Both fields are required; the parser
    reads them in order, so this test specifically exercises the
    second key's check (card_id present, half_name missing)."""
    with pytest.raises(KeyError, match="half_name"):
        synthetic_state.apply_action(
            {"kind": "unlock_room_half", "card_id": 7}
        )


def test_apply_action_unlock_room_half_wrong_type_for_card_id_raises(
    synthetic_state,
):
    """``card_id`` must be a u32-extractable integer. A string fails
    PyO3's ``extract::<u32>`` machinery → ``TypeError`` / ``ValueError``.
    The dispatch-stub never runs; the parse-time gate fires first."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {
                "kind": "unlock_room_half",
                "card_id": "not-an-int",
                "half_name": "left",
            }
        )


def test_apply_action_unlock_room_half_wrong_type_for_half_name_raises(
    synthetic_state,
):
    """``half_name`` must be a Python string. A non-string value (e.g.
    an int) fails PyO3's ``extract::<String>`` machinery → ``TypeError``
    / ``ValueError``. The dispatch-stub never runs; the parse-time gate
    fires first."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {
                "kind": "unlock_room_half",
                "card_id": 7,
                "half_name": 123,
            }
        )


def test_apply_action_unknown_kind_message_mentions_unlock_room_half(
    synthetic_state,
):
    """The unknown-kind error message includes ``unlock_room_half``
    after this slice extends the variant list. Pin the substring so a
    future variant addition forces the message to grow."""
    with pytest.raises(ValueError, match=r"unlock_room_half"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- ActivateStation -------------------------------------------------
#
# Rolling-integration slice (post-UnlockRoomHalf): extend the Action
# enum + apply_action dispatcher with ``Action::ActivateStation {
# spacecraft_id, crew_id }``. The Python dataclass
# ``ActivateStation(spacecraft_id: int, crew_id: int)`` carries two
# ints — the GameObject id of the Spacecraft permanent and the
# GameObject id of the creature being tapped as cost. **This slice
# completes the Action enum scaffold: all 8 Python Action subclasses
# now have Rust variants.**
#
# EoE Spacecraft cycle: Spacecraft enter as artifacts with no creature
# types and accumulate charge counters via station-attaching (tap a
# creature, add charge counters equal to its power); they turn face
# up (become creatures with printed stats) via ``TurnFaceUp`` once
# enough charge counters have been added. Sorcery-speed only
# (enforced by the future ``legal_decisions`` enumerator, not by this
# dispatch arm).
#
# **Dispatch is stubbed.** Rust has no Spacecraft charge-counter state
# today, and there's no crew-tap-resolution primitive that ties the
# tap-as-cost to the counter-add-as-effect. The variant + bridge round-
# trip ship now, so callers building Action dicts can already use the
# ``activate_station`` shape and hit a clear "not implemented yet"
# error path rather than the generic ``unknown action kind`` rejection.
# A future slice adds the per-object Spacecraft charge state + the
# activate-station primitive then flips the dispatch arm to a real
# ``call_method1("activate_station", ...)`` call. Until then, happy-
# path dispatch raises ``ValueError`` — these tests assert on the
# stubbed error path, NOT on a successful station outcome. Mirrors the
# round-23 ActivateAbility and round-24 TurnFaceUp / UnlockRoomHalf
# "scaffold the variant, defer the dispatch semantics" pattern.


def test_apply_action_activate_station_stub_raises_value_error(synthetic_state):
    """Happy-path stub: a well-formed ActivateStation dict parses
    cleanly but the dispatch arm raises ``ValueError`` because no Rust
    Spacecraft charge-counter state / activate-station primitive
    exists yet. The error message names the variant + flags it as
    future work so a follow-up slice grepping for the string finds
    the dispatch site to flip."""
    with pytest.raises(
        ValueError, match=r"ActivateStation dispatch not implemented yet"
    ):
        synthetic_state.apply_action(
            {"kind": "activate_station", "spacecraft_id": 42, "crew_id": 5}
        )


def test_apply_action_activate_station_stub_error_mentions_spacecraft(
    synthetic_state,
):
    """The stub message names the missing Rust state (Spacecraft
    charge-counter + crew-tap resolution) so a future slice grepping
    for "Spacecraft" finds the dispatch site to flip. Pin the substring
    so the message stays informative — a future slice that flips the
    dispatch to a real call must also delete this test (it documents
    the stub-state, not the eventual real behavior)."""
    with pytest.raises(ValueError, match=r"Spacecraft"):
        synthetic_state.apply_action(
            {"kind": "activate_station", "spacecraft_id": 7, "crew_id": 8}
        )


def test_apply_action_activate_station_missing_spacecraft_id_raises_key_error(
    synthetic_state,
):
    """Missing the ``spacecraft_id`` key raises ``KeyError`` —
    ``parse_action`` rejects before dispatch (single point of
    validation, mirrors the ``source_id``-missing shape on
    ``activate_ability``). The KeyError fires BEFORE the dispatch-stub
    ValueError, so this test pins the parse-time gate."""
    with pytest.raises(KeyError, match="spacecraft_id"):
        synthetic_state.apply_action(
            {"kind": "activate_station", "crew_id": 5}
        )


def test_apply_action_activate_station_missing_crew_id_raises_key_error(
    synthetic_state,
):
    """Missing the ``crew_id`` key raises ``KeyError`` — same parse-
    time gate as ``spacecraft_id``. Both fields are required; the
    parser reads them in order, so this test specifically exercises
    the second key's check (spacecraft_id present, crew_id missing)."""
    with pytest.raises(KeyError, match="crew_id"):
        synthetic_state.apply_action(
            {"kind": "activate_station", "spacecraft_id": 7}
        )


def test_apply_action_activate_station_wrong_type_for_spacecraft_id_raises(
    synthetic_state,
):
    """``spacecraft_id`` must be a u32-extractable integer. A string
    fails PyO3's ``extract::<u32>`` machinery → ``TypeError`` /
    ``ValueError``. The dispatch-stub never runs; the parse-time gate
    fires first."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {
                "kind": "activate_station",
                "spacecraft_id": "not-an-int",
                "crew_id": 5,
            }
        )


def test_apply_action_activate_station_wrong_type_for_crew_id_raises(
    synthetic_state,
):
    """``crew_id`` must be a u32-extractable integer. A string fails
    PyO3's ``extract::<u32>`` machinery → ``TypeError`` / ``ValueError``.
    The dispatch-stub never runs; the parse-time gate fires first."""
    with pytest.raises((TypeError, ValueError)):
        synthetic_state.apply_action(
            {
                "kind": "activate_station",
                "spacecraft_id": 7,
                "crew_id": "not-an-int",
            }
        )


def test_apply_action_unknown_kind_message_mentions_activate_station(
    synthetic_state,
):
    """The unknown-kind error message includes ``activate_station``
    after this slice extends the variant list. Pin the substring so a
    future variant addition forces the message to grow. **This
    completes the Action enum scaffold — all 8 Python Action
    subclasses now have Rust variants.**"""
    with pytest.raises(ValueError, match=r"activate_station"):
        synthetic_state.apply_action({"kind": "frobnicate"})


# --- Cross-variant edge cases ---------------------------------------
#
# Rolling-integration slice (post-round-25 dispatch flip): broader
# edge-case coverage spanning all 8 Action variants. Motivation:
# round-25 (commit b155a91) flipped the ActivateAbility dispatch from
# a "not implemented yet" stub to a real mana-ability dispatch, so the
# cross-variant interactions (PlayLand-then-ActivateAbility on the
# just-played land, repeated DeclareAttackers semantics, the full
# legal_decisions round-trip surface) need a second look. The
# per-variant happy-path / missing-key / wrong-type tests above are
# in place; these add the cross-variant + sequencing + stubbed-
# dispatch-error breadth.

PHASE_MAIN1 = 3
ZONE_LIBRARY = 1


# Category 1: cross-action sequencing -------------------------------


def test_apply_action_play_land_then_activate_ability_on_same_land(
    synthetic_state,
):
    """Two-step composite sanity: PlayLand a Forest, then
    ActivateAbility on the just-played Forest. The Forest enters the
    battlefield untapped (per the round-4 play_land primitive); the
    immediate ActivateAbility taps it and credits one G to the
    priority player's pool. Locks in that the two arms compose
    cleanly — a future slice changing land-ETB or the mana-pool credit
    direction would break this end-to-end."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    pool_before = synthetic_state.get_mana_pool(0)["G"]

    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_id})
    # Forest is on the battlefield, untapped.
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_tapped(forest_id) is False

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )
    # Forest now tapped + one G in pool.
    assert synthetic_state.is_tapped(forest_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == pool_before + 1


def test_apply_action_activate_ability_same_forest_twice_second_raises_already_tapped(
    synthetic_state,
):
    """ActivateAbility on the SAME untapped Forest twice: the first
    call taps + credits a G; the second hits the "already tapped"
    guard in the dispatch and raises ValueError. Pins the second-call
    failure mode — locks in that the dispatcher checks the
    `tapped` flag on every call (no caching, no per-turn untap-on-
    activate shortcut)."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )
    assert synthetic_state.is_tapped(forest_id) is True

    with pytest.raises(ValueError, match=r"already tapped"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": forest_id,
                "ability_index": 0,
            }
        )


def test_apply_action_declare_attackers_twice_accumulates_not_overwrites(
    synthetic_state,
):
    """Sequencing semantics surprise: ``declare_attacker`` PUSHES onto
    ``state.attackers`` (see ``state_ops_combat.rs:501``); the apply_action
    DeclareAttackers arm iterates the dispatch list and calls the
    primitive per id, never clearing first. Therefore two consecutive
    DeclareAttackers([a]) + DeclareAttackers([b]) calls ACCUMULATE
    into [a, b] rather than the second OVERWRITING the first.

    Pins this so a future slice that switches the semantics (e.g.
    Python's `_declare_attackers` reassigns `state.attackers = list(
    eligible_set)` — an overwrite) trips loudly. Production callers
    that want overwrite semantics must reset_combat() between
    dispatches; the dispatcher doesn't impose that for them."""
    a0 = P0_FIRST_BEAR_ID
    a1 = P0_FIRST_BEAR_ID + 1

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [a0]}
    )
    assert synthetic_state.get_attackers() == [a0]

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [a1]}
    )
    # Accumulates — NOT overwrites. Both bears now on the attacker list.
    assert synthetic_state.get_attackers() == [a0, a1]


def test_apply_action_declare_attackers_then_empty_does_not_clear(
    synthetic_state,
):
    """Companion to the accumulate test: passing an empty
    ``attacker_ids`` set on the second call does NOT clear the prior
    declaration. The dispatcher iterates the empty list to zero
    primitive calls — a true no-op rather than a reset. Locks in that
    callers wanting to ``clear`` the attackers list must invoke a
    distinct reset primitive (reset_combat, end_combat_step, ...)
    rather than relying on an empty-set DeclareAttackers dispatch."""
    bear_id = P0_FIRST_BEAR_ID
    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [bear_id]}
    )
    assert synthetic_state.get_attackers() == [bear_id]

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": []}
    )
    # Untouched — the empty list is a no-op, not a reset.
    assert synthetic_state.get_attackers() == [bear_id]


# Category 2: unknown-kind error message completeness ---------------


def test_apply_action_unknown_kind_message_lists_all_eight_variants(
    synthetic_state,
):
    """The unknown-kind ValueError message lists EVERY currently
    supported kind so a caller building dicts by hand can spot a
    typo immediately. Single regression test that exercises the
    completeness of the listed-kinds substring — per-variant tests
    above each pin their own substring, but only this one fails if
    the message drops one in a careless refactor (e.g. an
    alphabetize-the-list edit that mistakenly drops a comma-
    separated entry)."""
    expected = [
        "pass_priority",
        "play_land",
        "declare_attackers",
        "declare_blockers",
        "activate_ability",
        "cast_spell",
        "turn_face_up",
        "unlock_room_half",
        "activate_station",
    ]
    try:
        synthetic_state.apply_action({"kind": "frobnicate"})
    except ValueError as exc:
        msg = str(exc)
    else:
        pytest.fail("expected ValueError for unknown kind")
    for kind in expected:
        assert kind in msg, f"unknown-kind message missing {kind!r}: {msg!r}"


# Category 3: bridge round-trip via legal_decisions ------------------


def test_apply_action_every_legal_decision_round_trips_without_raising(
    synthetic_state,
):
    """Round-25 broadens what ``legal_decisions`` emits to include
    real-dispatch ActivateAbility (mana abilities). Every dict that
    falls out of ``legal_decisions(0)`` on the default synthetic state
    in MAIN1 must apply cleanly via ``apply_action`` against a fresh
    clone of the state. Pins the emit/apply contract end-to-end
    across all the currently-emitted kinds (PassPriority, PlayLand
    [pending hand contents], CastSpell, ActivateAbility) — a
    future slice that adds an emission variant without also wiring
    the apply arm trips this test.

    Sets up the state with both a Forest (PlayLand) and an existing
    Llanowar Elf in hand (CastSpell). Uses clones for each apply so
    one action's mutation doesn't break a later action's
    preconditions."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    # Llanowar Elves are already in P0's hand at ids 10..16 — pin
    # presence rather than re-import.
    assert synthetic_state.printing_name_of(P0_FIRST_HAND_ID) == "Llanowar Elves"
    synthetic_state.add_mana(0, "G", 1)  # cover CastSpell mana cost
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)
    # Sanity: round-25 emits all four production kinds in this state.
    kinds = {a["kind"] for a in actions}
    assert {"pass_priority", "play_land", "cast_spell", "activate_ability"}.issubset(
        kinds
    ), f"missing emission kind(s) in {kinds!r}"

    # Apply each action against its own clone — no cross-action
    # interference (e.g. CastSpell pays the G, then ActivateAbility
    # would have nothing to apply if we shared the state).
    for action in actions:
        clone = synthetic_state.clone_once()
        # MUST NOT raise.
        clone.apply_action(action)
    # Original state is unchanged after the loop (all mutations
    # happened on clones).
    assert synthetic_state.find_object_zone(forest_id) == ZONE_HAND


# Category 5: CastSpell flag pass-through ---------------------------


def test_apply_action_cast_spell_with_warp_and_kicked_true_still_dispatches(
    synthetic_state,
):
    """``use_warp=True`` + ``kicked=True`` together: still routes to
    the base ``cast_creature`` primitive (round-25 flags-ignored
    contract). The Elf lands on the battlefield and only the base
    {G} mana cost is paid — neither alt-cost machinery fires. Locks
    in the combinatorial flag-ignored property across multiple
    True flags."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    synthetic_state.apply_action(
        {
            "kind": "cast_spell",
            "card_id": elf_id,
            "use_warp": True,
            "kicked": True,
        }
    )

    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    # Only the base G was spent — neither warp's alt-cost nor kicker's
    # extra pip was debited (would have failed for insufficient mana
    # if the kicker pip had been wired).
    assert synthetic_state.get_mana_pool(0)["G"] == 0


def test_apply_action_cast_spell_cast_room_half_left_routes_to_cast_creature(
    synthetic_state,
):
    """``cast_room_half="left"`` still routes to ``cast_creature`` —
    no special Room-half dispatch yet. The Llanowar Elf isn't a Room,
    but the dispatcher doesn't check for that; the flag is data-only
    and the base creature cast path runs unchanged. Pins that the
    Option<String> path with a non-None value is observationally
    identical to a None value (parity with the existing
    cast_room_half_string_preserved test, but specifically with
    "left" rather than an arbitrary value, since "left" is the
    canonical wire value the Room mechanic uses)."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)

    synthetic_state.apply_action(
        {
            "kind": "cast_spell",
            "card_id": elf_id,
            "cast_room_half": "left",
        }
    )

    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD


# Category 6: ActivateAbility ability_index gate --------------------


def test_apply_action_activate_ability_ability_index_large_value_raises(
    synthetic_state,
):
    """Large non-zero ``ability_index`` (999) — same dispatch raise
    as index=1 / index=2 (the per-variant tests above), pinning that
    the gate is `!= 0` rather than a small-int allowlist. A future
    slice that adds support for `ability_index in {1, 2}` (e.g.
    Spacecraft launch costs) would need to grow this gate; this
    test stays as a "did we forget to extend the gate?" canary."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    with pytest.raises(ValueError, match=r"only ability_index=0"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": forest_id,
                "ability_index": 999,
            }
        )


def test_apply_action_activate_ability_negative_ability_index_raises_at_parse(
    synthetic_state,
):
    """A negative ``ability_index`` (-1) fails PyO3's
    ``extract::<u32>`` machinery at PARSE time — before the dispatch
    gate ever runs. The error is OverflowError / ValueError / TypeError
    depending on PyO3's mood (the conversion path differs between
    PyLong subtypes); we accept the union. Locks in that the u32 type
    boundary at parse is the first defense — the negative value
    never reaches the dispatch arm's `!= 0` gate."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    with pytest.raises((TypeError, ValueError, OverflowError)):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": forest_id,
                "ability_index": -1,
            }
        )


# Category 7: bad inputs --------------------------------------------


def test_apply_action_empty_kind_string_is_unknown_kind(synthetic_state):
    """``apply_action({"kind": ""})`` falls through the
    ``parse_action`` match arms to the `other => Err(...)` arm — the
    empty string isn't a registered variant. Raises ValueError with
    the unknown-kind message. Locks in that the empty-string edge
    case takes the unknown-kind path, NOT a special-cased "missing
    kind" error (which would shadow the legitimate ``kind=""`` typo
    case behind a KeyError-looking message)."""
    with pytest.raises(ValueError, match=r"unknown action kind"):
        synthetic_state.apply_action({"kind": ""})


# Category 8: CoW canary on dispatched actions ----------------------


def test_apply_action_play_land_on_clone_leaves_original_unchanged(
    synthetic_state,
):
    """Cross-snapshot CoW canary for a real-dispatch action (not the
    PassPriority u8-toggle covered above): apply PlayLand on a clone
    and assert the ORIGINAL state's hand / battlefield / library
    snapshots are byte-for-byte identical pre- and post-clone-mutation.
    Locks in the per-zone Arc<Vec<GameObject>> CoW invariant survives
    a multi-zone mutation (PlayLand moves a card from hand to
    battlefield — two zones change).

    Round-25 motivation: dispatch flips that compose multiple
    primitives (PlayLand alone is one primitive but mutates two
    zones — the ActivateAbility flip in round-25 likewise mutates the
    battlefield Arc + the mana pool) need broader CoW coverage than
    the priority-player u8 test at the top of the file."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    p0_hand_before = synthetic_state.hand_snapshot(0)
    p0_bf_before = synthetic_state.battlefield_snapshot()
    p0_lib_before = synthetic_state.library_snapshot(0)
    p0_hand_size_before = synthetic_state.hand_sizes[0]
    p0_bf_size_before = synthetic_state.battlefield_sizes[0]

    cloned = synthetic_state.clone_once()
    cloned.apply_action({"kind": "play_land", "card_id": forest_id})

    # Clone reflects the mutation.
    assert cloned.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert cloned.hand_sizes[0] == p0_hand_size_before - 1
    assert cloned.battlefield_sizes[0] == p0_bf_size_before + 1

    # Original is untouched across every snapshot.
    assert synthetic_state.find_object_zone(forest_id) == ZONE_HAND
    assert synthetic_state.hand_sizes[0] == p0_hand_size_before
    assert synthetic_state.battlefield_sizes[0] == p0_bf_size_before
    assert synthetic_state.hand_snapshot(0) == p0_hand_before
    assert synthetic_state.battlefield_snapshot() == p0_bf_before
    assert synthetic_state.library_snapshot(0) == p0_lib_before


def test_apply_action_cast_spell_on_clone_leaves_original_unchanged(
    synthetic_state,
):
    """Companion CoW canary for the CastSpell arm: a clone-side cast
    of a Llanowar Elf mutates the clone's hand / battlefield / mana
    pool — the original's snapshots stay byte-identical. CastSpell
    routes to cast_creature, which spends mana AND moves the card —
    so the canary covers three independent piece of state (zone for
    the card, the mana pool, and battlefield sizes). Pins that the
    Arc-backed zones + mana pool both honor CoW under dispatch."""
    elf_id = P0_FIRST_HAND_ID
    synthetic_state.add_mana(0, "G", 1)
    pool_before = synthetic_state.get_mana_pool(0)["G"]
    hand_before = synthetic_state.hand_snapshot(0)
    bf_before = synthetic_state.battlefield_snapshot()

    cloned = synthetic_state.clone_once()
    cloned.apply_action({"kind": "cast_spell", "card_id": elf_id})

    # Clone reflects the mutation.
    assert cloned.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    assert cloned.get_mana_pool(0)["G"] == pool_before - 1

    # Original untouched across every observable surface.
    assert synthetic_state.find_object_zone(elf_id) == ZONE_HAND
    assert synthetic_state.get_mana_pool(0)["G"] == pool_before
    assert synthetic_state.hand_snapshot(0) == hand_before
    assert synthetic_state.battlefield_snapshot() == bf_before
