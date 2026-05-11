"""Cross-feature edge-case coverage for ACTION SEQUENCES on the Rust
``GameState.apply_action`` surface.

Companion file to ``tests_rust/test_actions.py``. That file covers the
**single-action dispatch shape** — one ``apply_action`` call per test
(plus a small handful of two-call sequences for "happens twice in a
row" / "clone vs original" canaries). This file covers **multi-action
chains**: each ``apply_action`` composes with the next, exercising
cross-variant interactions that aren't visible in any one single-
action test.

**Test-only slice. No Rust source changes.** Every primitive these
tests touch is already wired by an earlier slice (PlayLand,
ActivateAbility, CastSpell, PassPriority, DeclareAttackers,
DeclareBlockers). The Rust source isn't modified — these are the
"composition" tests that pin observed cross-variant behaviors so a
later refactor can't silently regress them.

## Why a separate file

``test_actions.py`` is per-variant: each section pins one variant's
contract in isolation, with a couple of "happens twice" sanity checks
inside the variant's section (e.g.
``test_apply_action_play_land_twice_in_a_row_both_succeed``). What's
missing from that organization is the **cross-variant** edge cases:
- a CastSpell after an ActivateAbility (mana) — pool persists across
  the variant boundary, then drains via cost payment;
- a DeclareAttackers followed by a DeclareBlockers — combat-state
  composition across two calls;
- a fingerprint-stable round trip across an entire chain.

Each test below is a 2–5-step chain. The synthetic state and helper
conventions match ``test_actions.py`` exactly so the file's "look and
feel" is consistent: same id constants, same ``_move_forest_into_hand``-
style helper, same zone constants.
"""

from __future__ import annotations

import pytest


# Zone constants — mirror ``test_actions.py`` / ``state_ops_zones.rs``.
ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3

# Synthetic-state id ranges (default 10/7/50 layout, mirrored from
# test_actions.py for consistency).
P0_FIRST_BATTLEFIELD_ID = 0  # 0..4 Forest, 5..9 Grizzly Bears
P0_FIRST_BEARS_ID = 5  # 5..9 Grizzly Bears (2/2, not summoning-sick)
P0_FIRST_HAND_ID = 10  # 10..16 Llanowar Elves (cost {G})
P0_FIRST_LIBRARY_ID = 17  # first Forest in library (17..66 all Forest)
P1_FIRST_BLOCKER_ID = 72  # p1 battlefield 72..76 Grizzly Bears


def _move_forest_into_hand(state, player_id=0):
    """Move a Forest from ``player_id``'s library into their hand and
    return its id. Same helper shape as ``test_actions.py``."""
    forest_id = P0_FIRST_LIBRARY_ID
    assert state.printing_name_of(forest_id) == "Forest"
    assert state.move_object(forest_id, ZONE_HAND, player_id) is True
    assert state.find_object_zone(forest_id) == ZONE_HAND
    return forest_id


# ---------------------------------------------------------------------
# 1. PlayLand → ActivateAbility (mana) chain
# ---------------------------------------------------------------------


def test_play_land_then_activate_mana_taps_just_played_land(synthetic_state):
    """PlayLand puts a Forest on the battlefield untapped; an immediate
    ActivateAbility on that same land taps it and credits one G to the
    controller's pool. Composes the round-23 PlayLand dispatch with the
    round-25 ActivateAbility(mana) dispatch — pins that a just-played
    land is immediately tappable (no summoning sickness for lands; CR
    302.1) AND that the priority-player → controller resolution works
    consistently across both arms (PlayLand uses priority_player as
    actor; ActivateAbility reads controller off the battlefield index,
    which equals priority_player here)."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_id})
    # Land is on battlefield, untapped.
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_tapped(forest_id) is False

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )

    assert synthetic_state.is_tapped(forest_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1


# ---------------------------------------------------------------------
# 2. PlayLand twice — pin observed (no one-land-per-turn gate at primitive)
# ---------------------------------------------------------------------


def test_play_land_twice_both_succeed_no_one_per_turn_gate(synthetic_state):
    """``test_actions.py`` already pins that two consecutive PlayLand
    dispatches both succeed (``test_apply_action_play_land_twice_in_a_
    row_both_succeed``). This sequence test broadens that to a 3-action
    chain — PlayLand, PlayLand, PlayLand — to lock in that the gate is
    NOT at the dispatch layer at every count, not just at 2. A future
    ``legal_decisions`` filter is the right home for the gate per CR
    305.2; until then, the dispatcher stays "dumb"."""
    f1 = _move_forest_into_hand(synthetic_state, 0)
    f2 = P0_FIRST_LIBRARY_ID + 1
    f3 = P0_FIRST_LIBRARY_ID + 2
    assert synthetic_state.move_object(f2, ZONE_HAND, 0) is True
    assert synthetic_state.move_object(f3, ZONE_HAND, 0) is True
    bf_before = synthetic_state.battlefield_sizes[0]

    synthetic_state.apply_action({"kind": "play_land", "card_id": f1})
    synthetic_state.apply_action({"kind": "play_land", "card_id": f2})
    synthetic_state.apply_action({"kind": "play_land", "card_id": f3})

    for fid in (f1, f2, f3):
        assert synthetic_state.find_object_zone(fid) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before + 3


# ---------------------------------------------------------------------
# 3. ActivateAbility on tapped source: failure leaves state unchanged
# ---------------------------------------------------------------------


def test_activate_then_activate_same_source_second_raises_state_intact(
    synthetic_state,
):
    """Two ActivateAbility calls on the SAME Forest in sequence: the
    first succeeds (forest becomes tapped, pool gains 1 G); the second
    raises ``ValueError`` ("already tapped") and leaves the pool at 1 —
    NOT 2. Composes the round-25 ActivateAbility dispatch with itself
    across a state-transition boundary (untapped → tapped). Pins that
    the raise short-circuits cleanly mid-sequence: a ValueError on the
    second call doesn't accidentally credit a second pip or untap the
    source."""
    forest_id = P0_FIRST_BATTLEFIELD_ID

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )
    assert synthetic_state.is_tapped(forest_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1

    with pytest.raises(ValueError, match=r"already tapped"):
        synthetic_state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": forest_id,
                "ability_index": 0,
            }
        )

    # State unchanged by the failed second call.
    assert synthetic_state.is_tapped(forest_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1


# ---------------------------------------------------------------------
# 4. CastSpell after ActivateAbility (mana): pool drains via cost
# ---------------------------------------------------------------------


def test_activate_mana_then_cast_spell_drains_pool(synthetic_state):
    """End-to-end mana economy across two dispatch arms: tap a Forest
    via ActivateAbility (pool: G=1), then cast a {G}-cost Llanowar Elf
    via CastSpell (pool: G=0, elf on battlefield). Pins that the mana
    pool is shared state across the variant boundary — the
    ActivateAbility's credit and CastSpell's debit see the SAME pool
    (no per-variant scratch buffer). Production search threads
    activate-then-cast sequences constantly; this is the irreducible
    smallest version of that pattern."""
    forest_id = P0_FIRST_BATTLEFIELD_ID
    elf_id = P0_FIRST_HAND_ID
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )
    assert synthetic_state.get_mana_pool(0)["G"] == 1

    synthetic_state.apply_action({"kind": "cast_spell", "card_id": elf_id})

    # Pool drained by exactly one G; elf landed on battlefield.
    assert synthetic_state.get_mana_pool(0)["G"] == 0
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    # Forest stayed tapped — CastSpell doesn't untap mana sources.
    assert synthetic_state.is_tapped(forest_id) is True


# ---------------------------------------------------------------------
# 5. PassPriority alternation: P0 → P1 → P0 → P1
# ---------------------------------------------------------------------


def test_pass_priority_four_times_walks_full_cycle(synthetic_state):
    """Four consecutive PassPriority dispatches walk the priority
    toggle through a full 0→1→0→1→0 cycle AT THE priority_player
    LEVEL. ``test_actions.py`` covers single + double dispatch;
    the four-step sequence pins that the priority toggle stays an
    involution under repeated application.

    G3 close-out aside (pass-priority-step-advance slice): in
    addition to the priority toggle, the apply_action layer now also
    fires ``advance_phase`` whenever both players pass in succession
    with an empty stack. Four passes therefore walk TWO steps on the
    phase ladder (UNTAP → UPKEEP after passes 1-2, UPKEEP → DRAW
    after passes 3-4) — the priority-toggle pattern still matches
    because each step-advance resets the consecutive-pass counter
    AND the priority sequence returns to active player between each
    pair of passes. See ``tests_rust/test_actions.py::test_apply_
    action_pass_priority_four_passes_advance_two_steps`` for the
    phase-side assertion that the step-advance fires. The bare
    ``state.pass_priority()`` primitive in
    ``state_ops_priority.rs`` remains a pure toggle that doesn't
    touch the counter; the round-complete bookkeeping lives only at
    the apply_action layer."""
    assert synthetic_state.get_priority_player() == 0
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 1
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 0
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 1
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 0
    # Active player never moved across the whole cycle (the
    # advance_phase calls move only `phase`, not `active_player` —
    # active rotates only at the CLEANUP→UNTAP wrap, which 4 passes
    # at the start of the turn don't reach).
    assert synthetic_state.active_player == 0


# ---------------------------------------------------------------------
# 6. DeclareAttackers + DeclareBlockers compose
# ---------------------------------------------------------------------


def test_declare_attackers_then_declare_blockers_both_stored(synthetic_state):
    """Round-23 combat composition: DeclareAttackers stores the
    attacker set on ``state.attackers`` AND taps each attacker;
    DeclareBlockers stores the (attacker, blocker) pairs on
    ``state.blockers``. After both apply_action calls, BOTH structures
    are populated; neither dispatch arm clobbers the other. Pins that
    the two arms address independent storage slots — a regression
    where DeclareBlockers cleared ``attackers`` (or vice versa) would
    surface here. Combat damage itself happens via a separate
    ``resolve_combat_damage`` primitive — neither dispatch arm
    short-circuits to damage."""
    attacker = P0_FIRST_BEARS_ID
    blocker = P1_FIRST_BLOCKER_ID

    synthetic_state.apply_action(
        {"kind": "declare_attackers", "attacker_ids": [attacker]}
    )
    assert synthetic_state.get_attackers() == [attacker]
    assert synthetic_state.is_tapped(attacker) is True

    synthetic_state.apply_action(
        {
            "kind": "declare_blockers",
            "assignments": [(blocker, attacker)],
        }
    )

    # Both combat structures populated. ``state.blockers`` storage
    # shape is (attacker, blocker) — the opposite of the action's
    # input shape — see ``test_apply_action_declare_blockers_single_
    # pair`` for the reorder rationale.
    assert synthetic_state.get_attackers() == [attacker]
    assert synthetic_state.get_blockers() == [(attacker, blocker)]
    # Attacker still tapped after blocker dispatch (no side effect).
    assert synthetic_state.is_tapped(attacker) is True


# ---------------------------------------------------------------------
# 7. Two CastSpell in succession — pin observed (no stack push today)
# ---------------------------------------------------------------------


def test_two_cast_spell_in_succession_no_stack_push_today(synthetic_state):
    """**Observed-behavior pin.** Per ``state_ops_cast.rs`` module
    docstring: "No stack yet. Casting a creature spell moves it
    directly hand → battlefield after paying mana." Two consecutive
    CastSpell dispatches therefore produce TWO new creatures on the
    battlefield AND leave the stack empty — NOT a stack with 2 entries
    (which would be the real-MTG / future-slice behavior).

    This test pins the current "no stack" semantics so a future
    cast-via-stack slice has to update both the test and the
    docstring together. The brief's "stack has 2 entries, top is the
    second cast" expectation describes the eventual real-MTG
    semantics; today the assertion is "stack empty, two creatures on
    battlefield"."""
    elf_a = P0_FIRST_HAND_ID
    elf_b = P0_FIRST_HAND_ID + 1
    synthetic_state.add_mana(0, "G", 2)
    bf_before = synthetic_state.battlefield_sizes[0]
    assert synthetic_state.stack_size() == 0

    synthetic_state.apply_action({"kind": "cast_spell", "card_id": elf_a})
    synthetic_state.apply_action({"kind": "cast_spell", "card_id": elf_b})

    # Both elves on battlefield; pool drained to 0.
    assert synthetic_state.find_object_zone(elf_a) == ZONE_BATTLEFIELD
    assert synthetic_state.find_object_zone(elf_b) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before + 2
    assert synthetic_state.get_mana_pool(0)["G"] == 0
    # Stack stayed empty — no creature cast pushed to the stack
    # under the current "no stack yet" cast-creature semantics.
    assert synthetic_state.stack_size() == 0


# ---------------------------------------------------------------------
# 8. ActivateAbility (mana) twice produces 2 mana pool entries
# ---------------------------------------------------------------------


def test_activate_mana_twice_distinct_sources_pools_two_g(synthetic_state):
    """Two ActivateAbility dispatches on distinct untapped Forests
    accumulate a G=2 pool — the round-25 dispatch arm credits the
    existing pool (saturating add) rather than overwriting on each
    call. ``test_actions.py`` covers this for two sources at the same
    time; the sequence here pins the same behavior across the dispatch
    layer (not the primitive directly), and additionally asserts BOTH
    sources end tapped — neither dispatch arm untapped the other
    Forest as a side effect."""
    forest_a = P0_FIRST_BATTLEFIELD_ID
    forest_b = P0_FIRST_BATTLEFIELD_ID + 1
    assert synthetic_state.get_mana_pool(0)["G"] == 0
    assert synthetic_state.is_tapped(forest_a) is False
    assert synthetic_state.is_tapped(forest_b) is False

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_a, "ability_index": 0}
    )
    assert synthetic_state.get_mana_pool(0)["G"] == 1
    assert synthetic_state.is_tapped(forest_a) is True
    # Forest B still untapped after Forest A's activation.
    assert synthetic_state.is_tapped(forest_b) is False

    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_b, "ability_index": 0}
    )
    assert synthetic_state.get_mana_pool(0)["G"] == 2
    assert synthetic_state.is_tapped(forest_a) is True
    assert synthetic_state.is_tapped(forest_b) is True


# ---------------------------------------------------------------------
# 9. Mixed sequence: PlayLand → ActivateAbility → CastSpell → PassPriority ×2
# ---------------------------------------------------------------------


def test_mixed_play_activate_cast_pass_pass_full_chain(synthetic_state):
    """A 5-action chain stringing every implemented happy-path variant
    through one state in production order:

    1. PlayLand: a Forest comes from hand to battlefield, untapped.
    2. ActivateAbility: tap the new Forest, pool G=1.
    3. CastSpell: pay the G to cast a Llanowar Elf — pool G=0, elf
       on battlefield.
    4. PassPriority: priority flips P0 → P1.
    5. PassPriority: priority flips P1 → P0.

    Pins that the full chain composes cleanly: no variant's dispatch
    arm clobbers state another arm reads. Production search loops
    drive exactly this kind of mixed-variant sequence on every node
    of the search tree, so an "everything composes" canary at the
    dispatch boundary is the cheapest safety net we can ship."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    elf_id = P0_FIRST_HAND_ID

    hand_before = synthetic_state.hand_sizes[0]
    bf_before = synthetic_state.battlefield_sizes[0]
    assert synthetic_state.get_mana_pool(0)["G"] == 0
    assert synthetic_state.get_priority_player() == 0

    # 1. PlayLand.
    synthetic_state.apply_action({"kind": "play_land", "card_id": forest_id})
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert synthetic_state.is_tapped(forest_id) is False

    # 2. ActivateAbility on the newly-played Forest.
    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )
    assert synthetic_state.is_tapped(forest_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1

    # 3. CastSpell — pay the G pip from step 2.
    synthetic_state.apply_action({"kind": "cast_spell", "card_id": elf_id})
    assert synthetic_state.find_object_zone(elf_id) == ZONE_BATTLEFIELD
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    # 4. PassPriority — P0 → P1.
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 1

    # 5. PassPriority — P1 → P0.
    synthetic_state.apply_action({"kind": "pass_priority"})
    assert synthetic_state.get_priority_player() == 0

    # Final accounting: hand lost 2 (forest + elf), battlefield gained
    # 2 (forest + elf), pool back to 0, priority back to 0.
    assert synthetic_state.hand_sizes[0] == hand_before - 2
    assert synthetic_state.battlefield_sizes[0] == bf_before + 2
    # Active player never moved; the chain stayed on P0's turn.
    assert synthetic_state.active_player == 0


# ---------------------------------------------------------------------
# 10. Determinism: same sequence on two clones → identical fingerprints
# ---------------------------------------------------------------------


def test_determinism_same_sequence_on_two_clones_identical_fingerprints(
    synthetic_state,
):
    """Apply the same 5-step action sequence to two independent clones
    of the same starting state; the fingerprint of each clone must be
    identical at EVERY intermediate step. Pins that ``apply_action``
    is deterministic with respect to the (state, action) pair — no
    randomness, no time-/address-derived bookkeeping, no per-clone
    init drift. This is the dispatch-layer companion to
    ``test_fingerprint_is_deterministic_across_runs`` (which pins
    fingerprint determinism for static states); this test pins it
    across multi-step mutation chains.

    Search MCTS rollouts depend on this property: two visits to the
    same node that apply the same action must reach an indistinguish-
    able child state. A regression where the dispatch arm leaks any
    non-deterministic byte into state would surface as a fingerprint
    mismatch here."""
    forest_id = _move_forest_into_hand(synthetic_state, 0)
    elf_id = P0_FIRST_HAND_ID
    actions = [
        {"kind": "play_land", "card_id": forest_id},
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0},
        {"kind": "cast_spell", "card_id": elf_id},
        {"kind": "pass_priority"},
        {"kind": "pass_priority"},
    ]

    a = synthetic_state.clone_once()
    b = synthetic_state.clone_once()
    assert a.fingerprint() == b.fingerprint()

    for action in actions:
        a.apply_action(action)
        b.apply_action(action)
        assert a.fingerprint() == b.fingerprint(), (
            f"fingerprint diverged after applying {action!r}: "
            f"a={a.fingerprint()} b={b.fingerprint()}"
        )

    # And one final equality after the full chain.
    assert a.fingerprint() == b.fingerprint()
