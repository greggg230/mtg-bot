"""End-to-end tests for ``GameState.legal_decisions(player_id)``.

The Phase 3 keystone: emits the closed set of legal Actions the
priority-holder can take right now. Python's MCTS / ISMCTS drives
off this enumeration; the Rust port must match Python's emission
order exactly (Action types in the same sequence with the same
per-type tie-breaking) for trajectory parity in
``scripts/parity_smoke.py``.

**Slice scope (rust(legal-decisions-skeleton + cast-spell)).**
``PassPriority`` + ``PlayLand`` + base ``CastSpell`` (no alt-cost
flags — all six defaulted: ``use_warp=False, kicked=False,
cast_for_free=False, use_impending=False, cast_room_half=None``).
``ActivateAbility``, ``DeclareAttackers``, ``DeclareBlockers``,
alt-cost CastSpell variants grow in follow-up slices.

**Emission-order audit vs Python.**  Mirrors
``src/mtg_bot/legal.py:legal_actions``:
- Line 178 — PassPriority is index 0 (prepended before any other
  variant).
- Lines 180-181 — off-priority callers see ``[PassPriority]`` only.
- Lines 25-32 + 303-304 — sorcery-timing gate (active == priority,
  main phase, empty stack) gates PlayLand emission.
- Lines 319-329 — PlayLand iterates hand in zone-insertion order,
  deduped by printing name.

**One-land-per-turn divergence.** Python's line 319 gates on
``p.lands_played_this_turn < max_lands``; the Rust ``GameState`` has
no ``lands_played_this_turn`` field yet (parallel-slice file-isolation
rule forbids adding it to ``state.rs`` from this slice). First-slice
behavior: emit a PlayLand action for every Land in hand regardless.
Caller / future slice enforcement is the Phase 5 fix.

Synthetic-state reminder (cf. ``tests_rust/test_actions.py``):
- Player 0 hand: ids 10..16, all Llanowar Elves (Creature).
- Player 0 battlefield: ids 0..4 Forest, 5..9 Grizzly Bears.
- Player 0 library: ids 17..66, all Forest.
- Player 1 mirrors with id_offset 67.
- Default phase = PHASE_UNTAP (0); tests that want sorcery timing
  must call ``set_phase(PHASE_MAIN1)``.
"""

from __future__ import annotations

import pytest

PHASE_UNTAP = 0
PHASE_MAIN1 = 3
PHASE_DECLARE_ATTACKERS = 5
PHASE_DECLARE_BLOCKERS = 6
PHASE_COMBAT_DAMAGE = 7
PHASE_MAIN2 = 9
PHASE_CLEANUP = 11

ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2

P0_FIRST_BATTLEFIELD_ID = 0
P0_FIRST_BEARS_ID = 5
P0_FIRST_HAND_ID = 10
P0_FIRST_LIBRARY_ID = 17  # first Forest in library


def _empty_hand(state, player_id=0):
    """Move every card out of ``player_id``'s hand into their library.
    Returns the count moved. Used for tests that want a clean
    starting hand (the synthetic seed gives 7 Llanowar Elves)."""
    hand = state.hand_snapshot(player_id)
    for obj in hand:
        # move to library (zone 1)
        ok = state.move_object(obj["id"], 1, player_id)
        assert ok is True
    return len(hand)


def _put_forest_in_hand(state, player_id=0, library_idx=0):
    """Move one Forest from ``player_id``'s library into their hand.
    Returns the Forest's id."""
    # The library snapshot is ordered bottom-to-top. The first Forest
    # in our P0 library starts at id P0_FIRST_LIBRARY_ID; the id
    # arithmetic + the snapshot agree.
    library = state.library_snapshot(player_id)
    forest_obj = library[library_idx]
    assert forest_obj["printing_name"] == "Forest"
    ok = state.move_object(forest_obj["id"], ZONE_HAND, player_id)
    assert ok is True
    return forest_obj["id"]


def _kinds(actions):
    """Pluck the kind strings out of a list of Action dicts."""
    return [a["kind"] for a in actions]


# --- 1. Empty hand --------------------------------------------------


def test_legal_decisions_empty_hand_in_main_phase_emits_only_pass_priority(
    synthetic_state,
):
    """Empty hand in PRECOMBAT_MAIN: only PassPriority emits + the 5
    Forests' tap-for-mana ActivateAbility (synthetic seed has 5
    Forests + 5 Grizzly Bears on p0 battlefield). PlayLand loop
    iterates an empty hand to zero actions (Python `legal.py:319-329`).
    Mirrors Python's `legal.py:178` (PassPriority always there)."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    # PassPriority + 5 ActivateAbility (one per untapped Forest).
    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("activate_ability") == 5
    assert "play_land" not in kinds
    assert "cast_spell" not in kinds


# --- 2. One Land in hand --------------------------------------------


def test_legal_decisions_one_land_in_hand_emits_pass_priority_then_play_land(
    synthetic_state,
):
    """One Forest in hand at sorcery timing -> 2 actions:
    PassPriority (Python `legal.py:178` — prepended) FIRST, then
    PlayLand (`:329` — appended inside the sorcery-timing block).
    Locks in the Python emission order: PassPriority at index 0,
    PlayLand at index 1."""
    _empty_hand(synthetic_state, 0)
    forest_id = _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    # PassPriority + 1 PlayLand + 5 ActivateAbility (one per
    # untapped Forest already on p0's battlefield).
    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds[1] == "play_land"
    assert kinds.count("play_land") == 1
    assert kinds.count("activate_ability") == 5
    # PlayLand carries the Forest's GameObject id.
    assert actions[1]["card_id"] == forest_id


# --- 3. Two Lands in hand -------------------------------------------


def test_legal_decisions_two_distinct_lands_in_hand_emit_two_play_lands(
    synthetic_state,
):
    """Two DIFFERENT-named lands in hand emit two PlayLands. The
    synthetic registry only has Forest; this test uses one Forest +
    one Grizzly Bears swapped into a land-type-by-card-types proxy
    isn't possible without registering a new printing, so we register
    a Mountain in a fresh Registry-style state instead. For the
    synthetic-state world, this test asserts the more realistic
    case: two Forests dedup to ONE PlayLand by Python's
    ``(printing.name,)`` dedup key — see legal.py:322-329. We pin
    the dedup behavior here and add the truly-two-lands case via
    a registered Mountain in a follow-up if/when the suite gets a
    Mountain-aware fixture."""
    _empty_hand(synthetic_state, 0)
    # Put two Forests in hand (same printing).
    forest_a = _put_forest_in_hand(synthetic_state, 0, library_idx=0)
    forest_b = _put_forest_in_hand(synthetic_state, 0, library_idx=0)
    assert forest_a != forest_b
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    # Dedup by printing name: two Forests collapse to one PlayLand.
    # Plus 5 ActivateAbility (per untapped Forest on p0 bf — these
    # are battlefield permanents, not deduped because each carries
    # its own per-object source_id).
    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("play_land") == 1
    assert kinds.count("activate_ability") == 5
    # The PlayLand carries the FIRST hand-order Forest's id.
    assert actions[1]["kind"] == "play_land"
    assert actions[1]["card_id"] == forest_a


# --- 4. Mixed hand: one Creature + one Land -------------------------


def test_legal_decisions_creature_plus_land_emits_play_land_and_cast_spell(
    synthetic_state,
):
    """Mixed hand (Llanowar Elves + Forest): the Forest emits a
    PlayLand, and the Elves (Creature) emits a CastSpell (post
    rust(legal-decisions-cast-spell) slice). The 7 Llanowar Elves in
    the synthetic seed all dedup to a single CastSpell by Python's
    `(printing.name,)` first-slice dedup key (Python `legal.py:352`).
    Locks in the mixed PlayLand + CastSpell coexistence and the
    PlayLand-comes-first emission order."""
    # Keep the existing Llanowar Elves in hand (don't empty it).
    forest_id = _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    kinds = _kinds(actions)
    # PassPriority then PlayLand then CastSpell — Python's natural
    # append order under sorcery timing. Plus 5 ActivateAbility (one
    # per untapped Forest on p0 battlefield).
    assert kinds[0] == "pass_priority"
    assert kinds.count("play_land") == 1
    assert kinds.count("cast_spell") == 1
    assert kinds.count("activate_ability") == 5
    # PlayLand carries the Forest's id (not any Llanowar Elves id).
    play_land = [a for a in actions if a["kind"] == "play_land"][0]
    assert play_land["card_id"] == forest_id


# --- 5. Non-main phase ----------------------------------------------


@pytest.mark.parametrize(
    "phase",
    [PHASE_UNTAP, PHASE_COMBAT_DAMAGE, PHASE_CLEANUP],
    ids=["UNTAP", "COMBAT_DAMAGE", "CLEANUP"],
)
def test_legal_decisions_non_main_phase_emits_only_pass_priority(
    synthetic_state, phase
):
    """Non-main phase (UNTAP / COMBAT_DAMAGE / CLEANUP): even with a
    Land in hand, no PlayLand emits — sorcery-timing gate (Python
    `legal.py:303-304`, `_is_sorcery_timing` lines 25-32) fails."""
    _empty_hand(synthetic_state, 0)
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(phase)

    actions = synthetic_state.legal_decisions(0)

    assert _kinds(actions) == ["pass_priority"]


# --- 6. Non-empty stack ---------------------------------------------


def test_legal_decisions_non_empty_stack_emits_only_pass_priority(
    synthetic_state,
):
    """Non-empty stack: even in PRECOMBAT_MAIN with a Land in hand,
    PlayLand can't emit — the sorcery-timing gate's empty-stack
    conjunct (Python `legal.py:30`) fails. Mirrors CR 305.2 (lands
    are played at sorcery speed, not as spells)."""
    _empty_hand(synthetic_state, 0)
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    # Push a dummy stack item. The legal_decisions enumerator only
    # cares whether the stack is empty.
    synthetic_state.push_to_stack(P0_FIRST_BEARS_ID, 0)

    actions = synthetic_state.legal_decisions(0)

    assert _kinds(actions) == ["pass_priority"]


# --- 7. Opponent's turn ---------------------------------------------


def test_legal_decisions_off_active_player_emits_only_pass_priority(
    synthetic_state,
):
    """Priority on opponent's turn (priority = active for opponent =
    P1, P0 is asking what's legal during P1's turn). With priority
    held but ``active_player != player_id``, sorcery-timing fails
    (Python `legal.py:28` — `state.players[active_player_idx].id ==
    player_id`). Only PassPriority emits."""
    _empty_hand(synthetic_state, 0)
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    # Flip active player to P1 but give priority back to P0.
    synthetic_state.advance_phase()  # advance to BEGIN_COMBAT
    # Walk to next turn: just brute-force the phase cycle to flip
    # active_player.
    while synthetic_state.active_player == 0:
        synthetic_state.advance_phase()
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.give_priority_to(0)
    assert synthetic_state.active_player == 1
    assert synthetic_state.get_priority_player() == 0

    actions = synthetic_state.legal_decisions(0)

    # P0 has priority but it's P1's turn — sorcery timing fails.
    assert _kinds(actions) == ["pass_priority"]


def test_legal_decisions_priority_off_emits_only_pass_priority(synthetic_state):
    """Priority on opponent (P1 has priority, P0 is asking). Python
    `legal.py:180-181` — off-priority callers immediately return
    the singleton `[PassPriority]` list before any sorcery-timing
    check runs."""
    _empty_hand(synthetic_state, 0)
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.pass_priority()  # P0 -> P1
    assert synthetic_state.get_priority_player() == 1

    actions = synthetic_state.legal_decisions(0)

    # P0 doesn't have priority — even with a Land in hand on P0's
    # turn, no PlayLand emits.
    assert _kinds(actions) == ["pass_priority"]


# --- 8. Out-of-range player_id --------------------------------------


def test_legal_decisions_out_of_range_player_id_raises_index_error(
    synthetic_state,
):
    """Mirrors the existing snapshot-getter convention: out-of-range
    ``player_id`` (anything > 1 in the 2-player engine) raises
    ``IndexError`` with a clear message. Locks in the same error
    shape as ``hand_snapshot`` / ``library_snapshot`` etc."""
    with pytest.raises(IndexError, match=r"player_id 2 out of range"):
        synthetic_state.legal_decisions(2)


# --- 9. Round-trip via apply_action ---------------------------------


def test_legal_decisions_actions_are_round_trip_able_via_apply_action(
    synthetic_state,
):
    """Every dict returned from ``legal_decisions`` is a valid input
    to ``apply_action`` (regression for the round-22/23 bridge). This
    locks in that the dispatch + enumeration ends round-trip cleanly
    — a future slice that adds a new Action type to the enumerator
    but forgets the matching ``apply_action`` arm trips this test."""
    _empty_hand(synthetic_state, 0)
    forest_id = _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    # Apply the PlayLand action: forest goes from hand to battlefield.
    play_land = [a for a in actions if a["kind"] == "play_land"][0]
    bf_before = synthetic_state.battlefield_sizes[0]
    synthetic_state.apply_action(play_land)
    assert synthetic_state.find_object_zone(forest_id) == ZONE_BATTLEFIELD
    assert synthetic_state.battlefield_sizes[0] == bf_before + 1


def test_legal_decisions_pass_priority_action_round_trips(synthetic_state):
    """The PassPriority emission (always at index 0) round-trips
    through ``apply_action`` — regression that locks in the dict shape
    matches what round-22 wired.

    CE2 closure: ``apply_action({"kind": "pass_priority"})`` is now a
    no-op (matches Python's ``apply_action(PassPriority)`` which is
    literally ``return``). The round-trip pin is "the dict shape
    parses cleanly and apply_action returns without raising" — the
    priority field stays at its starting value because the arm is a
    pure no-op. Priority management is owned by the composer
    (``complete_step_via_priority_round``), not by ``apply_action``."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)
    pass_priority = actions[0]
    assert pass_priority["kind"] == "pass_priority"

    assert synthetic_state.get_priority_player() == 0
    result = synthetic_state.apply_action(pass_priority)
    # No-op arm: returns None, priority unchanged.
    assert result is None
    assert synthetic_state.get_priority_player() == 0


# --- 10. PassPriority position --------------------------------------


def test_legal_decisions_pass_priority_is_always_first_emission(
    synthetic_state,
):
    """PassPriority is at index 0 in EVERY emission — mirrors Python's
    `legal.py:178` (`actions = [PassPriority()]` prepended). Pin
    this across multiple scenarios so a future slice that switches
    to a different position (e.g. appending PassPriority at the
    end) trips loudly."""
    # Empty hand, main phase.
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    assert synthetic_state.legal_decisions(0)[0]["kind"] == "pass_priority"

    # Off-priority.
    synthetic_state.pass_priority()
    assert synthetic_state.legal_decisions(0)[0]["kind"] == "pass_priority"

    # Off-main phase.
    synthetic_state.give_priority_to(0)
    synthetic_state.set_phase(PHASE_UNTAP)
    assert synthetic_state.legal_decisions(0)[0]["kind"] == "pass_priority"

    # With lands in hand at sorcery timing.
    synthetic_state.set_phase(PHASE_MAIN1)
    _put_forest_in_hand(synthetic_state, 0)
    actions = synthetic_state.legal_decisions(0)
    assert actions[0]["kind"] == "pass_priority"
    # And PlayLand is somewhere after it.
    assert "play_land" in _kinds(actions[1:])


# --- 11. PassPriority emitted exactly once --------------------------


def test_legal_decisions_pass_priority_emitted_exactly_once(synthetic_state):
    """PassPriority emits exactly once per call — Python's `legal.py:178`
    inserts the singleton at list-init time and never re-appends.
    Catches a regression where a future slice appends PassPriority
    at the end of the loop on top of the index-0 emission."""
    _empty_hand(synthetic_state, 0)
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    assert _kinds(actions).count("pass_priority") == 1


# --- 12. P1 main phase parity ---------------------------------------


def test_legal_decisions_works_for_player_1_on_their_turn(synthetic_state):
    """P1's turn, main phase, with a Forest in P1's hand: PlayLand
    emits for P1, mirroring P0's behavior. Locks in that the
    enumerator isn't hardcoded to player 0 — it indexes the
    right player's hand via ``player_id``."""
    _empty_hand(synthetic_state, 1)
    forest_id = _put_forest_in_hand(synthetic_state, 1)
    # Advance until active_player flips to 1, then set MAIN1 +
    # priority to 1.
    while synthetic_state.active_player == 0:
        synthetic_state.advance_phase()
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.give_priority_to(1)

    actions = synthetic_state.legal_decisions(1)

    # PassPriority + 1 PlayLand + 5 ActivateAbility (one per P1
    # untapped Forest on P1's battlefield, mirroring the P0 case).
    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("play_land") == 1
    assert kinds.count("activate_ability") == 5
    play_land = [a for a in actions if a["kind"] == "play_land"][0]
    assert play_land["card_id"] == forest_id


# --- 13. CastSpell: one Creature in main phase ----------------------
#
# rust(legal-decisions-cast-spell) — extends `legal_decisions` to emit
# CastSpell actions for non-Land cards in hand at sorcery timing.
# Mirrors Python's `legal.py:335-445` (the sorcery-speed CastSpell
# loop, `_is_sorcery_speed_spell` body). First-slice scope: base
# CastSpell with all alt-cost flags defaulted (use_warp=False,
# kicked=False, cast_for_free=False, use_impending=False,
# cast_room_half=None). Mana-affordability check is deferred — every
# non-Land in hand emits regardless of can_pay_cost.


def test_legal_decisions_one_creature_in_hand_emits_cast_spell(
    synthetic_state,
):
    """Single Llanowar Elves in hand at sorcery timing -> 2 actions:
    PassPriority FIRST (Python `legal.py:178`), then CastSpell
    (`legal.py:386` — `actions.append(CastSpell(card.id))` inside
    the sorcery-speed loop). No PlayLand because the Elves isn't a
    Land. Locks in CastSpell's emission position AFTER PlayLand /
    after PassPriority and the all-defaults flag shape."""
    _empty_hand(synthetic_state, 0)
    # Pull a single Elves out of the library back into hand. The
    # library is Forest-only; instead use the existing 7-elf hand
    # but clear all but one.
    library = synthetic_state.library_snapshot(0)
    # No Elves in library — populate hand directly via P0's stock
    # Llanowar Elves at id P0_FIRST_HAND_ID=10. Move 6 of them out
    # to library, leaving one in hand.
    hand = synthetic_state.hand_snapshot(0)
    assert hand == []  # _empty_hand cleared it
    # Move 1 Llanowar Elves from library (library starts with the
    # original 7 elves at the top? No — the seed puts elves in hand,
    # not library). Instead: undo the empty. We'll re-fill via
    # _empty_hand's reverse — just reset by getting a fresh state
    # is cleaner. Use the synthetic state straight without empty.
    # Re-make:
    import mtg_bot_engine
    state = mtg_bot_engine.make_synthetic_state()
    # Clear hand down to exactly 1 Llanowar Elves.
    hand = state.hand_snapshot(0)
    assert len(hand) == 7
    assert all(o["printing_name"] == "Llanowar Elves" for o in hand)
    # Move 6 out (to library).
    for obj in hand[1:]:
        assert state.move_object(obj["id"], 1, 0) is True
    elves_id = hand[0]["id"]
    state.set_phase(PHASE_MAIN1)

    actions = state.legal_decisions(0)

    # PassPriority + 1 CastSpell + 5 ActivateAbility (untapped Forests
    # on p0 battlefield each emit a tap-for-mana ActivateAbility).
    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("cast_spell") == 1
    assert kinds.count("activate_ability") == 5
    cs = [a for a in actions if a["kind"] == "cast_spell"][0]
    assert cs["card_id"] == elves_id


# --- 14. CastSpell: PlayLand + CastSpell emission order -------------


def test_legal_decisions_land_plus_creature_emit_in_python_order(
    synthetic_state,
):
    """Mixed hand at sorcery timing: PlayLand emits BEFORE CastSpell.
    Mirrors Python's natural append order — PlayLand loop at
    `legal.py:319-329` runs FIRST (lower line number), then the
    sorcery-speed CastSpell loop at `legal.py:365-445`. Final list
    is `[pass_priority, play_land, cast_spell_1, cast_spell_2, ...]`."""
    # Keep the 7 Llanowar Elves in hand and add a Forest.
    forest_id = _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    kinds = _kinds(actions)
    # 7 Llanowar Elves dedup to 1 CastSpell; 1 Forest -> 1 PlayLand;
    # 5 Forests on the battlefield -> 5 ActivateAbility. The 3 hand
    # zone actions come in PlayLand-then-CastSpell order; the 5
    # ActivateAbility come after (matching the emission-order
    # contract documented at the end of the slice).
    assert kinds[0] == "pass_priority"
    assert kinds[1] == "play_land"
    assert kinds[2] == "cast_spell"
    assert kinds.count("activate_ability") == 5
    assert actions[1]["card_id"] == forest_id
    assert actions[2]["card_id"] == P0_FIRST_HAND_ID


# --- 15. CastSpell: dedup on same printing --------------------------


def test_legal_decisions_two_same_printing_creatures_dedup_to_one_cast(
    synthetic_state,
):
    """Two Llanowar Elves (same printing) -> ONE CastSpell. Mirrors
    Python's dedup key at `legal.py:352-359`: `(printing.name, flag1,
    flag2, ..., flag5)`. With first-slice all-default flags, the key
    collapses to `(printing.name,)` and two-same-printing collapse
    to a single CastSpell — same shape as the PlayLand two-Forests
    dedup test above. The CastSpell carries the FIRST hand-order
    copy's id."""
    # Synthetic state already has 7 Llanowar Elves in hand. Trim
    # to exactly 2 so the assertion is sharp.
    hand = synthetic_state.hand_snapshot(0)
    assert len(hand) == 7
    for obj in hand[2:]:
        assert synthetic_state.move_object(obj["id"], 1, 0) is True
    elves_first_id = hand[0]["id"]
    elves_second_id = hand[1]["id"]
    assert elves_first_id != elves_second_id
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    # Dedup: two Elves -> ONE CastSpell. Plus 5 ActivateAbility
    # (untapped Forests on the p0 battlefield — these are
    # battlefield permanents, not deduped by name).
    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds[1] == "cast_spell"
    assert kinds.count("cast_spell") == 1
    assert kinds.count("activate_ability") == 5
    # The CastSpell carries the FIRST hand-order Elves' id.
    assert actions[1]["card_id"] == elves_first_id


# --- 16. CastSpell: dict shape carries all 6 flag fields ------------


def test_legal_decisions_cast_spell_dict_has_all_six_flag_fields(
    synthetic_state,
):
    """The CastSpell dict carries all 7 keys: kind + card_id + 5
    alt-cost flag fields. Mirrors `actions.rs:action_to_pydict`'s
    "always emit all 5 flag fields so round-trip is fully symmetric"
    contract — every CastSpell dict from `legal_decisions` must
    survive a `parse_action -> action_to_pydict` cycle without
    key drift."""
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)
    cast_spells = [a for a in actions if a["kind"] == "cast_spell"]
    assert len(cast_spells) == 1

    cs = cast_spells[0]
    # Every key present, defaults match Python's dataclass.
    assert set(cs.keys()) == {
        "kind",
        "card_id",
        "use_warp",
        "kicked",
        "cast_for_free",
        "use_impending",
        "cast_room_half",
    }
    assert cs["use_warp"] is False
    assert cs["kicked"] is False
    assert cs["cast_for_free"] is False
    assert cs["use_impending"] is False
    assert cs["cast_room_half"] is None


# --- 17. CastSpell: not emitted off sorcery timing ------------------


@pytest.mark.parametrize(
    "phase",
    [PHASE_UNTAP, PHASE_COMBAT_DAMAGE, PHASE_CLEANUP],
    ids=["UNTAP", "COMBAT_DAMAGE", "CLEANUP"],
)
def test_legal_decisions_non_main_phase_omits_cast_spell(
    synthetic_state, phase
):
    """Non-main phase: even with Creatures in hand, no CastSpell
    emits — sorcery-timing gate (Python `legal.py:303-304`) fails.
    First-slice ignores Python's instant-speed CastSpell block at
    lines 246-301 (instants / flash); when that block ports over, a
    flash creature in hand at non-main phase will emit, but this
    test still holds for the synthetic-state Llanowar Elves
    (no flash)."""
    synthetic_state.set_phase(phase)

    actions = synthetic_state.legal_decisions(0)

    assert _kinds(actions) == ["pass_priority"]
    assert "cast_spell" not in _kinds(actions)


# --- 18. CastSpell: not emitted with non-empty stack ----------------


def test_legal_decisions_non_empty_stack_omits_cast_spell(
    synthetic_state,
):
    """Non-empty stack: sorcery-timing gate's empty-stack conjunct
    (Python `legal.py:30`) fails, so no CastSpell emits at sorcery
    speed. First-slice has no instant-speed CastSpell block, so the
    list collapses to `[pass_priority]`."""
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.push_to_stack(P0_FIRST_BEARS_ID, 0)

    actions = synthetic_state.legal_decisions(0)

    assert _kinds(actions) == ["pass_priority"]


# --- 19. CastSpell: not emitted off-turn ----------------------------


def test_legal_decisions_off_active_player_omits_cast_spell(
    synthetic_state,
):
    """Priority on opponent's turn (P0 has priority during P1's
    main phase). Sorcery-timing fails (`active_player != player_id`,
    Python `legal.py:28`). No CastSpell emits, mirroring the same
    PlayLand-off-turn behavior in test 7."""
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.advance_phase()
    while synthetic_state.active_player == 0:
        synthetic_state.advance_phase()
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.give_priority_to(0)
    assert synthetic_state.active_player == 1
    assert synthetic_state.get_priority_player() == 0

    actions = synthetic_state.legal_decisions(0)

    # P0 has priority but it's P1's turn — sorcery timing fails.
    assert _kinds(actions) == ["pass_priority"]


# --- 20. CastSpell: dict round-trips through parse_action -----------


def test_legal_decisions_cast_spell_action_parses_via_apply_action(
    synthetic_state,
):
    """Every CastSpell dict from `legal_decisions` is accepted by
    `apply_action`'s dict parser. The bridge's `parse_action` arm
    for "cast_spell" (round-22 CastSpell agent) must accept the
    shape this slice emits without raising. The first-slice scope
    for `apply_action` on CastSpell may still mutate state (calling
    the existing `cast_creature` primitive) — we don't assert on
    the result, only that the parse + dispatch round-trip doesn't
    raise on the dict shape.

    This is the regression for a future slice that adds a new
    flag to the CastSpell variant but forgets to extend the
    emitter — the new key would mismatch parse_action."""
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)
    cast_spell = [a for a in actions if a["kind"] == "cast_spell"][0]

    # apply_action may either succeed (cast goes to stack /
    # battlefield) or raise on a downstream gate (mana cost not
    # paid — first-slice doesn't gate on affordability so the
    # primitive itself may refuse). Both outcomes are acceptable
    # here; what matters is that `parse_action` doesn't raise on
    # the dict shape. We catch ValueError as the "downstream
    # rejected the apply, not a parse failure" signal — anything
    # else (KeyError, TypeError) IS a regression.
    try:
        synthetic_state.apply_action(cast_spell)
    except ValueError:
        pass


# --- 21. CastSpell: emission order between PlayLand and PassPriority


def test_legal_decisions_cast_spell_after_play_land_after_pass_priority(
    synthetic_state,
):
    """Locks in the full Python emission order: PassPriority is at
    index 0 (Python's `legal.py:178` prepend), PlayLand actions
    come next (Python lines 319-329), then CastSpell actions
    (Python lines 365-445). Catches a regression where a future
    slice swaps the loop order and breaks parity with Python's
    `legal_actions`."""
    forest_id = _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    kinds = _kinds(actions)
    # Find the indices of each kind.
    pass_idx = kinds.index("pass_priority")
    play_idx = kinds.index("play_land")
    cast_idx = kinds.index("cast_spell")
    assert pass_idx < play_idx < cast_idx
    # And there is exactly one of each Action kind among the
    # hand-zone emissions. (Plus 5 ActivateAbility for the 5
    # untapped Forests on p0's battlefield — locked in below to
    # confirm ActivateAbility sits AFTER CastSpell in emission
    # order.)
    assert kinds.count("pass_priority") == 1
    assert kinds.count("play_land") == 1
    assert kinds.count("cast_spell") == 1
    assert kinds.count("activate_ability") == 5
    # ActivateAbility comes AFTER CastSpell in Python's natural
    # append order (legal.py:612 is downstream of the CastSpell
    # block at 335-445).
    aa_idx = kinds.index("activate_ability")
    assert cast_idx < aa_idx
    assert actions[play_idx]["card_id"] == forest_id


# --- 22. CastSpell: P1 parity on P1's turn --------------------------


def test_legal_decisions_cast_spell_works_for_player_1(synthetic_state):
    """P1's turn, main phase: P1's hand of Llanowar Elves emits a
    CastSpell action for P1. Locks in the enumerator's
    player-indexing — `self.players[idx].hand` walks the correct
    player's hand, mirroring the existing P1-PlayLand parity test."""
    # P1 starts with 7 Llanowar Elves in their hand (synthetic seed).
    while synthetic_state.active_player == 0:
        synthetic_state.advance_phase()
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.give_priority_to(1)

    actions = synthetic_state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    # P1's Elves dedup to 1 CastSpell.
    assert kinds.count("cast_spell") == 1
    cast_spell = [a for a in actions if a["kind"] == "cast_spell"][0]
    # The CastSpell carries one of P1's hand-card ids (id_offset = 67).
    p1_hand = synthetic_state.hand_snapshot(1)
    p1_hand_ids = {o["id"] for o in p1_hand}
    assert cast_spell["card_id"] in p1_hand_ids


# --- DeclareAttackers emission --------------------------------------
#
# rust(legal-decisions-declare-attackers) — extends `legal_decisions`
# to emit `DeclareAttackers` action dicts during the
# DECLARE_ATTACKERS phase. **Deliberate divergence from Python**:
# Python's `legal_actions` (in `src/mtg_bot/legal.py`) does NOT emit
# DeclareAttackers — the engine.py docstring (lines 10-19) explicitly
# says the combat-declaration actions "aren't enumerable as legal
# moves because their decision space is combinatorial," and Python
# uses a factored search-side approach in
# `src/mtg_bot/search/combat_decisions.py` (`AttackerDecision` —
# one per eligible attacker, 2 branches each — feeds MCTS at search
# time and the adapter accumulates them into a single engine-level
# `DeclareAttackers(frozenset[int])`).
#
# For the Rust port's first-slice we emit a powerset enumeration so
# `legal_decisions` covers the DECLARE_ATTACKERS phase uniformly.
# With N eligible attackers and N <= 4, all 2^N subsets emit. With
# N > 4, only the two canonical extremes (empty + full) emit so the
# dict count stays bounded. Future slices replace this with the
# factored-search shape (one AttackerDecision-style action per
# eligible attacker).
#
# Eligibility mirrors `src/mtg_bot/combat.py:eligible_attackers`
# (lines 15-23): on active player's battlefield, is a Creature, not
# tapped, not summoning-sick OR has haste. Defender (CR 702.3) and
# continuous "can't attack" restrictions are deferred to the
# `declare_attacker` primitive's per-id apply-time gate.


def _custom_registry_state(
    engine, *, p0_battlefield_printings, p0_hand_printings=None,
    p1_battlefield_printings=None,
):
    """Build a minimal GameState with a small custom registry that
    we control end-to-end (haste, defender, vanilla creature,
    Forest land). Returns (state, name_to_pid) so tests can refer
    to printings by name.

    Battlefield is seeded by hand-moving cards from library->bf via
    `move_object`; we put the printings in p0_hand initially and
    then move them to the battlefield.
    """
    import mtg_bot_engine as _engine
    registry = _engine.Registry()
    name_to_pid = {}
    name_to_pid["Forest"] = registry.register_printing({
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
    })
    name_to_pid["Vanilla Bear"] = registry.register_printing({
        "name": "Vanilla Bear",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
    })
    name_to_pid["Hasty Bear"] = registry.register_printing({
        "name": "Hasty Bear",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
        "keywords": ["haste"],
    })
    name_to_pid["Defender Wall"] = registry.register_printing({
        "name": "Defender Wall",
        "card_types": ["Creature"],
        "power": 0,
        "toughness": 4,
        "keywords": ["defender"],
    })

    # Build state with hand-cards seeded; then we move them onto the
    # battlefield to control object ids precisely. Library can be a
    # single Forest so the player has SOMETHING in their library
    # (move_object uses zone constants and works for non-empty
    # libraries too).
    p1_bf = p1_battlefield_printings or []
    p0_bf = p0_battlefield_printings
    p0_hand = p0_hand_printings or []
    state = _engine.make_state(
        registry,
        # Libraries: one Forest each, so the player exists with a
        # well-formed state. The id assignment is:
        #   p0_library ids 0..len-1, then p0_hand ids next, then p1.
        p0_library_printing_ids=[name_to_pid["Forest"]],
        p1_library_printing_ids=[name_to_pid["Forest"]],
        p0_hand_printing_ids=(
            [name_to_pid[name] for name in p0_bf]
            + [name_to_pid[name] for name in p0_hand]
        ),
        p1_hand_printing_ids=(
            [name_to_pid[name] for name in p1_bf]
        ),
    )

    # The cards we want on the battlefield were placed in hand;
    # move them to the battlefield. The hand_snapshot order matches
    # the order we passed them in; the first len(p0_bf) entries are
    # the battlefield targets.
    p0_hand_snap = state.hand_snapshot(0)
    bf_ids = []
    for obj in p0_hand_snap[: len(p0_bf)]:
        ok = state.move_object(obj["id"], ZONE_BATTLEFIELD, 0)
        assert ok is True
        bf_ids.append(obj["id"])

    # Move p1's "battlefield" cards too if any.
    p1_hand_snap = state.hand_snapshot(1)
    p1_bf_ids = []
    for obj in p1_hand_snap[: len(p1_bf)]:
        ok = state.move_object(obj["id"], ZONE_BATTLEFIELD, 1)
        assert ok is True
        p1_bf_ids.append(obj["id"])

    # CE1 (r28) — make_state now defaults summoning_sick=true to match
    # Python's universal default. Test fixture intent is "creatures
    # have been on the battlefield long enough to act"; clear sickness
    # explicitly on both sides (the test harness shortcut for the
    # untap-step sweep).
    state.clear_summoning_sick_for_player(0)
    state.clear_summoning_sick_for_player(1)

    return state, name_to_pid, bf_ids, p1_bf_ids


def _put_bf_into_declare_attackers(state, player_id=0):
    """Set the state up so `player_id` is the active player with
    priority during the DECLARE_ATTACKERS phase and the stack is
    empty. The synthetic state defaults to active_player=0,
    priority=0 — we just flip phase. For player 1 we walk the phase
    cycle until active_player flips.
    """
    if player_id == 1:
        while state.active_player == 0:
            state.advance_phase()
    state.set_phase(PHASE_DECLARE_ATTACKERS)
    state.give_priority_to(player_id)


# --- 23. DECLARE_ATTACKERS + 0 eligible attackers -------------------


def test_legal_decisions_declare_attackers_phase_zero_creatures_emits_empty_subset(
    engine,
):
    """DECLARE_ATTACKERS phase with NO eligible attackers (empty
    battlefield): emit ONE DeclareAttackers (empty set) + PassPriority
    = 2 actions total. The powerset of the empty set is {{}}, a
    one-element set containing only the empty subset — bit-for-bit
    what `for bitmask in 0..1` yields."""
    state, _, _, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=[],  # empty battlefield
    )
    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    assert kinds == ["pass_priority", "declare_attackers"]
    da = [a for a in actions if a["kind"] == "declare_attackers"][0]
    assert da["attacker_ids"] == []


# --- 24. DECLARE_ATTACKERS + 1 untapped creature --------------------


def test_legal_decisions_declare_attackers_phase_one_eligible_emits_two_subsets(
    engine,
):
    """DECLARE_ATTACKERS phase with 1 untapped non-sick creature:
    emit 2 DeclareAttackers dicts (empty + full = 2^1) +
    PassPriority = 3 total. Locks in: (a) emission count = 2^N
    when N <= 4, (b) PassPriority is still at index 0, (c) the full
    subset carries exactly the eligible creature's id."""
    state, _, bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
    )
    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("declare_attackers") == 2
    das = [a for a in actions if a["kind"] == "declare_attackers"]
    subsets = {tuple(sorted(d["attacker_ids"])) for d in das}
    # 2^1 powerset: empty + {bear}.
    assert subsets == {(), (bf_ids[0],)}


# --- 25. DECLARE_ATTACKERS + 2 untapped creatures -------------------


def test_legal_decisions_declare_attackers_phase_two_eligible_emits_four_subsets(
    engine,
):
    """DECLARE_ATTACKERS phase with 2 untapped non-sick creatures:
    emit 4 DeclareAttackers dicts (powerset of 2^2 = 4) +
    PassPriority. Subsets: {}, {A}, {B}, {A,B}. Locks in the cap-
    just-below-threshold behavior."""
    state, _, bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear", "Vanilla Bear"],
    )
    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("declare_attackers") == 4
    das = [a for a in actions if a["kind"] == "declare_attackers"]
    subsets = {tuple(sorted(d["attacker_ids"])) for d in das}
    a, b = bf_ids[0], bf_ids[1]
    assert subsets == {(), (a,), (b,), (a, b)}


# --- 26. DECLARE_ATTACKERS + tapped creature excluded ---------------


def test_legal_decisions_declare_attackers_tapped_creature_is_excluded(
    engine,
):
    """A tapped creature can't attack (CR 508.1a). With only one
    creature on the battlefield and it tapped, the eligible set is
    empty → emit only the empty subset, mirroring the zero-eligible
    case in test 23."""
    state, _, bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
    )
    # Tap the bear.
    state.tap_object(bf_ids[0])
    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    # Tapped creature -> no eligible -> just the empty subset.
    assert kinds == ["pass_priority", "declare_attackers"]
    da = [a for a in actions if a["kind"] == "declare_attackers"][0]
    assert da["attacker_ids"] == []


# --- 27. Summoning-sick creature WITHOUT haste excluded -------------


def test_legal_decisions_declare_attackers_summoning_sick_without_haste_excluded(
    engine,
):
    """A summoning-sick creature without haste can't attack (CR
    302.1). We exercise the exclusion by `cast_creature`-ing a
    Vanilla Bear (no haste) onto the battlefield, which stamps
    `summoning_sick = true` on the resulting permanent (see
    `state_ops_cast.rs:319-320`). The sick bear must NOT appear in
    the eligible set → only the empty subset emits."""
    # Build a registry with our custom printings + put a Vanilla
    # Bear in p0's hand. Then fund p0's mana pool and cast the bear
    # so the resulting battlefield slot carries summoning_sick=true.
    import mtg_bot_engine as _engine
    registry = _engine.Registry()
    forest_pid = registry.register_printing({
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
    })
    bear_pid = registry.register_printing({
        "name": "Vanilla Bear",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
        "mana_cost": {"G": 1, "generic": 1},
    })
    state = _engine.make_state(
        registry,
        p0_library_printing_ids=[forest_pid],
        p1_library_printing_ids=[forest_pid],
        p0_hand_printing_ids=[bear_pid],
        p1_hand_printing_ids=[],
    )
    # Find the bear in p0's hand.
    hand_snap = state.hand_snapshot(0)
    bear_obj = next(o for o in hand_snap if o["printing_name"] == "Vanilla Bear")
    bear_id = bear_obj["id"]
    # Fund the mana pool: 1G + 1 generic. We feed G + C.
    state.add_mana(0, "G", 1)
    state.add_mana(0, "C", 1)
    # Cast it — stamps summoning_sick = true on the battlefield slot.
    ok = state.cast_creature(bear_id, 0)
    assert ok is True
    # Confirm the bear is on the battlefield with summoning_sick =
    # true. We read via the snapshot (no player_id arg — emits both
    # players' bf).
    bf = state.battlefield_snapshot()
    bear_bf = next(o for o in bf if o["id"] == bear_id)
    assert bear_bf["summoning_sick"] is True

    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)

    # Sick bear without haste is excluded → only the empty subset
    # emits (n=0 eligible).
    kinds = _kinds(actions)
    assert kinds == ["pass_priority", "declare_attackers"]
    da = [a for a in actions if a["kind"] == "declare_attackers"][0]
    assert da["attacker_ids"] == []


# --- 28. Summoning-sick creature WITH haste included ----------------


def test_legal_decisions_declare_attackers_haste_creature_included(
    engine,
):
    """A creature with haste attacks the turn it enters (CR 702.10
    bypasses the CR 302.1 summoning-sickness restriction). We
    `cast_creature` a Hasty Bear so it lands on the battlefield
    with `summoning_sick = true`; the haste keyword on its printing
    must override the sickness exclusion → the bear emits in the
    eligible set, yielding 2 DeclareAttackers (empty + full)."""
    import mtg_bot_engine as _engine
    registry = _engine.Registry()
    forest_pid = registry.register_printing({
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
    })
    hasty_pid = registry.register_printing({
        "name": "Hasty Bear",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
        "mana_cost": {"G": 1, "generic": 1},
        "keywords": ["haste"],
    })
    state = _engine.make_state(
        registry,
        p0_library_printing_ids=[forest_pid],
        p1_library_printing_ids=[forest_pid],
        p0_hand_printing_ids=[hasty_pid],
        p1_hand_printing_ids=[],
    )
    hand_snap = state.hand_snapshot(0)
    hasty_obj = hand_snap[0]
    hasty_id = hasty_obj["id"]
    state.add_mana(0, "G", 1)
    state.add_mana(0, "C", 1)
    ok = state.cast_creature(hasty_id, 0)
    assert ok is True
    bf = state.battlefield_snapshot()
    hasty_bf = next(o for o in bf if o["id"] == hasty_id)
    # Stamped sick — but haste on the printing overrides.
    assert hasty_bf["summoning_sick"] is True

    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    # Haste creature is eligible despite sick → full 2^1 powerset.
    assert kinds.count("declare_attackers") == 2
    das = [a for a in actions if a["kind"] == "declare_attackers"]
    subsets = {tuple(sorted(d["attacker_ids"])) for d in das}
    assert subsets == {(), (hasty_id,)}


# --- 29. Non-DECLARE_ATTACKERS phase: no DeclareAttackers -----------


@pytest.mark.parametrize(
    "phase",
    [PHASE_UNTAP, PHASE_MAIN1, PHASE_MAIN2, PHASE_CLEANUP],
    ids=["UNTAP", "MAIN1", "MAIN2", "CLEANUP"],
)
def test_legal_decisions_non_declare_attackers_phase_emits_no_declare_attackers(
    engine, phase,
):
    """DeclareAttackers only emits during the DECLARE_ATTACKERS
    phase. In any other phase (including the main phases where
    PlayLand / CastSpell DO emit), no `declare_attackers` dict
    appears in the emission set."""
    state, _, _, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
    )
    state.set_phase(phase)
    state.give_priority_to(0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    assert "declare_attackers" not in kinds


# --- 30. Off-turn DECLARE_ATTACKERS: no DeclareAttackers ------------


def test_legal_decisions_declare_attackers_off_turn_emits_no_declare_attackers(
    engine,
):
    """You can't attack on the opponent's turn. Even during the
    DECLARE_ATTACKERS phase, if `active_player != player_id`, the
    enumerator must NOT emit DeclareAttackers for player_id.
    Mirrors CR 506.2 — only the active player declares attackers."""
    # Build state with bears on both battlefields; advance to P1's
    # DECLARE_ATTACKERS phase but query for P0.
    state, _, _, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    # Walk to P1's turn.
    while state.active_player == 0:
        state.advance_phase()
    state.set_phase(PHASE_DECLARE_ATTACKERS)
    # Priority on P0 (off-turn). P1 is active, P0 is off-active.
    state.give_priority_to(0)
    assert state.active_player == 1
    assert state.get_priority_player() == 0

    actions = state.legal_decisions(0)

    # P0 is off-active during P1's DECLARE_ATTACKERS — no
    # DeclareAttackers emitted for P0.
    kinds = _kinds(actions)
    assert "declare_attackers" not in kinds


# --- 31. N > cap: only empty + full subset emit ---------------------


def test_legal_decisions_declare_attackers_cap_above_threshold_emits_two_subsets(
    engine,
):
    """With N > DECLARE_ATTACKERS_POWERSET_CAP (= 4) eligible
    attackers, only the empty + full subsets emit. Confirms the cap
    keeps `legal_decisions` bounded at 2 dicts even on a wide
    board, instead of exploding to 2^N. Builds a 5-creature board
    (N = 5 > cap = 4)."""
    state, _, bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"] * 5,
    )
    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    # Capped: just 2 DeclareAttackers dicts (empty + everyone).
    assert kinds.count("declare_attackers") == 2
    das = [a for a in actions if a["kind"] == "declare_attackers"]
    subsets = [tuple(sorted(d["attacker_ids"])) for d in das]
    assert () in subsets
    assert tuple(sorted(bf_ids)) in subsets


# --- 32. Round-trip every emitted DeclareAttackers via apply_action -


def test_legal_decisions_declare_attackers_dicts_round_trip_via_apply_action(
    engine,
):
    """Every DeclareAttackers dict from `legal_decisions` must be a
    valid input to `apply_action` (regression for the round-23
    declare-attackers bridge). For the FULL-subset dict in particular,
    applying it should mark the attackers in `state.attackers`."""
    state, _, bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear", "Vanilla Bear"],
    )
    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)
    das = [a for a in actions if a["kind"] == "declare_attackers"]
    # Find the full-set DeclareAttackers (2 ids).
    full_da = max(das, key=lambda d: len(d["attacker_ids"]))
    assert sorted(full_da["attacker_ids"]) == sorted(bf_ids)

    # Apply it; the primitive must accept the dict shape without
    # raising. We don't assert on the precise battlefield mutation
    # (combat-state diagnostics live in test_combat.py); the
    # round-trip itself is the regression target.
    state.apply_action(full_da)


# --- 33. DeclareAttackers carries the right wire shape --------------


def test_legal_decisions_declare_attackers_dict_shape_is_minimal(
    engine,
):
    """Every DeclareAttackers dict carries exactly two keys: `kind`
    + `attacker_ids`. Mirrors `actions.rs:action_to_pydict`'s
    DeclareAttackers arm. Catches a future-slice regression that
    adds an extra wire key (e.g. `controller_id`) to the emitter
    but forgets to update the round-trip parser."""
    state, _, _, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
    )
    _put_bf_into_declare_attackers(state, 0)

    actions = state.legal_decisions(0)
    das = [a for a in actions if a["kind"] == "declare_attackers"]
    assert len(das) >= 1
    for d in das:
        assert set(d.keys()) == {"kind", "attacker_ids"}


# --- 34. P1 parity during their DECLARE_ATTACKERS ------------------


def test_legal_decisions_declare_attackers_works_for_player_1(engine):
    """P1's DECLARE_ATTACKERS phase: DeclareAttackers emits for P1's
    eligible battlefield creatures (not P0's). Locks in the
    enumerator's player-indexing — `self.players[idx].battlefield`
    walks the correct player's creatures."""
    state, _, _, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    # Walk to P1's turn + DECLARE_ATTACKERS phase.
    _put_bf_into_declare_attackers(state, 1)
    assert state.active_player == 1
    assert state.get_priority_player() == 1

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    # 2^1 = 2 DeclareAttackers (empty + P1's bear).
    assert kinds.count("declare_attackers") == 2
    das = [a for a in actions if a["kind"] == "declare_attackers"]
    full = [d for d in das if len(d["attacker_ids"]) == 1][0]
    # The single eligible attacker is P1's bear, not P0's.
    assert full["attacker_ids"] == [p1_bf_ids[0]]


# --- 35. DECLARE_ATTACKERS + non-empty stack: no DeclareAttackers --


def test_legal_decisions_declare_attackers_non_empty_stack_emits_no_declare_attackers(
    engine,
):
    """Non-empty stack during DECLARE_ATTACKERS: the
    `stack.is_empty()` gate fails, so DeclareAttackers does NOT
    emit (the active player must wait for the stack to resolve
    before declaring attackers). Mirrors CR 116.5 — declaration
    happens when the stack is empty."""
    state, _, bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
    )
    _put_bf_into_declare_attackers(state, 0)
    # Push something onto the stack.
    state.push_to_stack(bf_ids[0], 0)

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    assert "declare_attackers" not in kinds


# --- ActivateAbility emission ---------------------------------------
#
# rust(legal-decisions-activate-ability) — extends `legal_decisions`
# to emit `ActivateAbility` dicts for tap-for-mana abilities on
# untapped mana-producer permanents the priority player controls.
# Mirrors Python's `legal.py:612-829` (the activated-ability emission
# block) with a first-slice approximation: Rust's `CardPrinting` has
# no `activated_abilities` field (see `rust/src/cards.rs:361-396` and
# the docstring's "ActivateAbility extension" section), so we treat
# every untapped permanent whose printing has `mana_produced.is_some()`
# as having a single implicit ability at `ability_index=0`. The full
# `activated_abilities` bridge is a deferred slice.
#
# **Gating** (mirror Python):
# - Source on the priority player's battlefield (Python line 614
#   `obj.controller_id != player_id: continue`).
# - Source untapped (Python line 658-660 `if ability.cost.tap_self
#   and obj.tapped: continue`).
# - Sorcery timing — Python's early return at line 303-304 gates
#   the entire ActivateAbility block. Per-ability `sorcery_speed`
#   check at line 620 is a no-op once we're past 304.
#
# **Dispatch is a stub.** `apply_action` on an ActivateAbility dict
# raises ValueError ("ActivateAbility dispatch not implemented yet")
# at `rust/src/state_ops_actions.rs:271-276`. This slice ships
# emission only; the dispatch arm flips to a real
# `call_method1("activate_ability", ...)` once the Rust primitive
# lands in a future slice.


def test_legal_decisions_activate_ability_emits_for_untapped_forest(
    synthetic_state,
):
    """Default synthetic state has 5 Forests + 5 Grizzly Bears on
    P0's battlefield. At sorcery timing, each untapped Forest emits
    one ActivateAbility (ability_index=0); Grizzly Bears (no
    mana_produced) do not emit. Locks in the basic emission and the
    ability_index=0 first-slice constant."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    # 5 Forests on P0's battlefield, all untapped.
    assert len(aas) == 5
    # source_ids are the 5 Forest object ids (P0_FIRST_BATTLEFIELD_ID
    # through P0_FIRST_BATTLEFIELD_ID + 4).
    source_ids = {a["source_id"] for a in aas}
    assert source_ids == {
        P0_FIRST_BATTLEFIELD_ID + i for i in range(5)
    }
    # First-slice: ability_index always 0 (no activated_abilities
    # field yet — see module docstring).
    assert all(a["ability_index"] == 0 for a in aas)


def test_legal_decisions_activate_ability_excludes_tapped_forest(
    synthetic_state,
):
    """A tapped Forest can't pay its {T} cost (CR 605.1a — mana
    abilities tap the source). Tapping one of the 5 Forests drops
    the ActivateAbility count to 4; the tapped Forest's id is NOT
    in the emission set. Mirrors Python's line 658-660 cost-payability
    check (`if ability.cost.tap_self and obj.tapped: continue`)."""
    _empty_hand(synthetic_state, 0)
    # Tap the first Forest (id 0).
    synthetic_state.tap_object(P0_FIRST_BATTLEFIELD_ID)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    # 5 Forests, 1 tapped -> 4 ActivateAbility emit.
    assert len(aas) == 4
    source_ids = {a["source_id"] for a in aas}
    # The tapped Forest's id is excluded.
    assert P0_FIRST_BATTLEFIELD_ID not in source_ids
    assert source_ids == {
        P0_FIRST_BATTLEFIELD_ID + i for i in range(1, 5)
    }


def test_legal_decisions_activate_ability_excludes_non_mana_producer(
    synthetic_state,
):
    """Grizzly Bears (Creature, no mana_produced) does NOT emit
    ActivateAbility — the first-slice's mana-only filter excludes
    permanents whose `printing.mana_produced.is_none()`. Locks in
    that the 5 Bears on P0's battlefield (ids 5-9) emit zero
    ActivateAbility actions; only the 5 Forests do."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    source_ids = {a["source_id"] for a in aas}
    # Bears (ids P0_FIRST_BEARS_ID through P0_FIRST_BEARS_ID + 4)
    # are NOT in the emission set.
    for i in range(5):
        assert (P0_FIRST_BEARS_ID + i) not in source_ids
    # Only the 5 Forests.
    assert len(aas) == 5


def test_legal_decisions_activate_ability_emits_one_per_mana_producer(
    synthetic_state,
):
    """Multiple untapped mana producers each get their own
    ActivateAbility (no dedup by printing name — unlike PlayLand /
    CastSpell, each battlefield permanent has independent
    per-object state, so the ids are distinct). Locks in the
    `len == count_of_untapped_mana_producers` invariant."""
    _empty_hand(synthetic_state, 0)
    # Tap 3 of the 5 Forests; 2 remain untapped.
    for i in range(3):
        synthetic_state.tap_object(P0_FIRST_BATTLEFIELD_ID + i)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    assert len(aas) == 2
    source_ids = {a["source_id"] for a in aas}
    assert source_ids == {
        P0_FIRST_BATTLEFIELD_ID + 3,
        P0_FIRST_BATTLEFIELD_ID + 4,
    }


def test_legal_decisions_activate_ability_excludes_opponent_permanent(
    synthetic_state,
):
    """An untapped mana producer on the OPPONENT's battlefield does
    NOT emit an ActivateAbility for the priority player (you can
    only activate your own permanents' abilities — CR 605.1).
    Locks in the per-player-zone iteration: the enumerator walks
    `self.players[priority_idx].battlefield`, not the global
    battlefield. P1 has 5 Forests too (synthetic seed mirrors P0);
    none of them should appear in P0's ActivateAbility set."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    # P0 Forests are at ids 0-4. P1's battlefield ids start at the
    # synthetic seed's id_offset (67); P0 emissions must NOT include
    # any id >= 67.
    source_ids = {a["source_id"] for a in aas}
    assert all(sid < 67 for sid in source_ids)
    assert len(aas) == 5


@pytest.mark.parametrize(
    "phase",
    [PHASE_UNTAP, PHASE_COMBAT_DAMAGE, PHASE_CLEANUP],
    ids=["UNTAP", "COMBAT_DAMAGE", "CLEANUP"],
)
def test_legal_decisions_activate_ability_not_emitted_off_sorcery_timing(
    synthetic_state, phase,
):
    """First-slice parity with Python: ActivateAbility is gated by
    the sorcery-timing early-return at Python `legal.py:303-304`.
    In non-main phases (UNTAP / COMBAT_DAMAGE / CLEANUP) the function
    returns before reaching the activated-ability block at line 612,
    so no `activate_ability` dicts emit. This is a known parity
    artifact — see the module docstring's "ActivateAbility
    extension" section. Once Rust grows an instant-speed window for
    mana abilities, this test flips to "emit regardless of phase"."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(phase)

    actions = synthetic_state.legal_decisions(0)

    kinds = _kinds(actions)
    assert "activate_ability" not in kinds


def test_legal_decisions_activate_ability_not_emitted_with_nonempty_stack(
    synthetic_state,
):
    """Non-empty stack: sorcery-timing's `stack.is_empty()` conjunct
    fails (Python `legal.py:30`), so the activated-ability block at
    line 612 is unreachable. First-slice parity — no ActivateAbility
    emits. Mirrors the same gate that already excludes PlayLand /
    CastSpell at non-empty stack."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.push_to_stack(P0_FIRST_BEARS_ID, 0)

    actions = synthetic_state.legal_decisions(0)

    kinds = _kinds(actions)
    assert "activate_ability" not in kinds


def test_legal_decisions_activate_ability_emission_order_after_cast_spell(
    synthetic_state,
):
    """ActivateAbility emits AFTER PassPriority + PlayLand +
    CastSpell. Mirrors Python's natural append order: PlayLand at
    line 319-329, CastSpell at 335-445, ActivateAbility at 612 —
    each block runs in source order, so the wire shape is
    [pass_priority, play_land..., cast_spell..., activate_ability...].
    Catches a regression where a future slice reorders the loops."""
    # Synthetic seed has 7 Elves in hand + 5 Forests + 5 Bears bf.
    # Add one Forest to hand so we get all four Action kinds.
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    kinds = _kinds(actions)
    pass_idx = kinds.index("pass_priority")
    play_idx = kinds.index("play_land")
    cast_idx = kinds.index("cast_spell")
    aa_idx = kinds.index("activate_ability")
    assert pass_idx < play_idx < cast_idx < aa_idx


def test_legal_decisions_activate_ability_dict_shape_is_minimal(
    synthetic_state,
):
    """Every ActivateAbility dict carries exactly three keys: kind,
    source_id, ability_index. Mirrors `actions.rs:action_to_pydict`'s
    `Action::ActivateAbility` arm so a future-slice regression that
    adds a wire key here but forgets the parse_action side trips
    loudly."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)
    aas = [a for a in actions if a["kind"] == "activate_ability"]
    assert len(aas) >= 1
    for a in aas:
        assert set(a.keys()) == {"kind", "source_id", "ability_index"}


def test_legal_decisions_activate_ability_apply_action_taps_and_credits_pool(
    synthetic_state,
):
    """End-to-end bridge round-trip: legal_decisions emits an
    ``ActivateAbility`` dict (round 24 emission gate) → apply_action
    consumes it (round 25 — this slice flips the round-23 stub to a
    real dispatch). Locks in the emission-to-application contract so a
    future slice that drifts either side's dict shape trips the test.

    Pre-flip (round 23 / round 24) this test asserted the stub raised
    "ActivateAbility dispatch not implemented yet"; round 25 flips the
    dispatch arm to a real mana-ability call, so we now assert the
    successful mutation outcome — the source becomes tapped and the
    controller's mana pool gains one pip of the listed color."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)
    aa = [a for a in actions if a["kind"] == "activate_ability"][0]
    source_id = aa["source_id"]
    assert aa["ability_index"] == 0
    assert synthetic_state.is_tapped(source_id) is False
    assert synthetic_state.get_mana_pool(0)["G"] == 0

    synthetic_state.apply_action(aa)

    # Forest tapped, controller's G pool +1.
    assert synthetic_state.is_tapped(source_id) is True
    assert synthetic_state.get_mana_pool(0)["G"] == 1


def test_legal_decisions_activate_ability_works_for_player_1(
    synthetic_state,
):
    """P1's main phase: P1's 5 Forests on their battlefield each
    emit an ActivateAbility for P1. Locks in player-indexing —
    `self.players[idx].battlefield` walks the right player's
    permanents, mirroring the existing P1-PlayLand / P1-CastSpell
    parity tests."""
    while synthetic_state.active_player == 0:
        synthetic_state.advance_phase()
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.give_priority_to(1)

    actions = synthetic_state.legal_decisions(1)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    assert len(aas) == 5
    # All source_ids are P1 ids (offset 67); none are P0 ids.
    source_ids = {a["source_id"] for a in aas}
    assert all(sid >= 67 for sid in source_ids)


def test_legal_decisions_activate_ability_excludes_summoning_sick_creature(
    engine,
):
    """**Parity gap G6 closure (CR 302.1).** A just-cast creature
    mana source (Llanowar Elves) is summoning-sick on the turn it
    enters and cannot pay an activated mana ability's `{T}` cost.
    The Rust `legal_decisions` emitter mirrors Python's
    `legal.py:661-663` short-circuit and DOES NOT emit an
    ActivateAbility for the sick source. Pairs with the matching
    dispatch-side gate in `state_ops_actions.rs` (covered by
    `test_tap.py::test_activate_ability_on_summoning_sick_creature_with_tap_cost_is_rejected`).

    Setup: 5 Forests on P0's battlefield (non-creature, never
    summoning-sick) + 1 newly-cast Llanowar Elves (creature,
    summoning_sick=true). Expected emission: 5 ActivateAbility for
    the Forests + 0 for the sick Elves."""
    state = engine.make_synthetic_state()
    # Cast Llanowar Elves from hand (id 10) — stamps
    # summoning_sick=true on the new bf slot.
    state.add_mana(0, "G", 1)
    elves_id = 10
    assert state.printing_name_of(elves_id) == "Llanowar Elves"
    assert state.cast_creature(elves_id, 0) is True
    assert state.is_summoning_sick(elves_id) is True

    _empty_hand(state, 0)
    state.set_phase(PHASE_MAIN1)
    actions = state.legal_decisions(0)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    source_ids = {a["source_id"] for a in aas}
    # The sick Elves' id is NOT in the emission set.
    assert elves_id not in source_ids
    # The 5 (non-sick, non-creature) Forests still emit.
    assert source_ids == {
        P0_FIRST_BATTLEFIELD_ID + i for i in range(5)
    }
    assert len(aas) == 5


def test_legal_decisions_activate_ability_emits_for_summoning_sick_land(
    synthetic_state,
):
    """**CR 302.1 creature-only gating (companion of the
    creature-rejection test above).** Summoning sickness applies ONLY
    to creatures; a sick LAND can still tap for mana the turn it
    enters the battlefield. The Rust `legal_decisions` emitter
    must NOT skip a sick land's ActivateAbility — Python's
    `legal.py:661-663` gate reads `is_creature_now(obj) &&
    obj.summoning_sick`, so non-creature sources fall through to the
    tap-self cost-payability check regardless of summoning_sick.

    Setup: route Forest id 0 through graveyard so the
    `move_object_from_graveyard_to_battlefield` primitive stamps
    `summoning_sick=true` on the way back to the battlefield. The
    Forest is now on p0's bf with summoning_sick=true. The remaining
    4 Forests (ids 1..4) stay summoning_sick=false. Expected
    emission: 5 ActivateAbility (one per Forest, including the
    sick reanimated one).

    Pins the legal-emission half of the bug fix; the
    dispatch-arm half is pinned by
    `test_tap.py::test_activate_ability_on_summoning_sick_land_with_tap_cost_is_allowed`.
    """
    forest_id = P0_FIRST_BATTLEFIELD_ID  # id 0, a Forest
    assert synthetic_state.printing_name_of(forest_id) == "Forest"
    # Move Forest to graveyard then reanimate so it lands on bf with
    # summoning_sick=true (CR 302.1 / 400.7 reset on reanimation).
    ZONE_GRAVEYARD = 3
    assert synthetic_state.move_object(forest_id, ZONE_GRAVEYARD, 0) is True
    assert (
        synthetic_state.move_object_from_graveyard_to_battlefield(0, forest_id)
        is True
    )
    assert synthetic_state.is_summoning_sick(forest_id) is True

    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    actions = synthetic_state.legal_decisions(0)

    aas = [a for a in actions if a["kind"] == "activate_ability"]
    source_ids = {a["source_id"] for a in aas}
    # The sick Forest IS in the emission set (CR 302.1 is creature-only).
    assert forest_id in source_ids
    # All 5 Forests emit; the 5 Bears are creatures with no mana_produced,
    # so they remain excluded.
    assert source_ids == {
        P0_FIRST_BATTLEFIELD_ID + i for i in range(5)
    }
    assert len(aas) == 5


# --- DeclareBlockers emission --------------------------------------
#
# rust(legal-decisions-declare-blockers) — extends `legal_decisions`
# to emit `DeclareBlockers` action dicts during the
# DECLARE_BLOCKERS phase. **Deliberate divergence from Python**:
# Python's `legal_actions` (in `src/mtg_bot/legal.py`) does NOT emit
# DeclareBlockers — same factored-search story as DeclareAttackers
# (`src/mtg_bot/engine.py:10-19` is explicit). Python uses
# `BlockerDecision` (one per eligible blocker, one branch per
# attacker plus a "don't block" branch) in
# `src/mtg_bot/search/combat_decisions.py` and accumulates them into
# a single engine-level `DeclareBlockers(assignments=...)` at apply
# time.
#
# For the Rust port we emit a direct enumeration of assignments. With
# `n_blockers <= 3` AND `m_attackers <= 3` we emit all `(M+1)^N`
# possibilities — each blocker picks one of M attackers OR "don't
# block." With either dimension over its cap we emit only the
# canonical "no blocks" assignment (an empty Vec). Combinatorial
# growth is harder than DeclareAttackers's `2^N`: a 4×4 board is
# `5^4 = 625`. Future slices replace this with the factored-search
# shape.
#
# Eligibility mirrors `src/mtg_bot/combat.py:eligible_blockers`
# (lines 26-32): on the defender's battlefield, is a Creature, not
# tapped. Summoning sickness does NOT exclude (CR 302.1 only
# restricts attacking + tap-cost activated abilities, not blocking).
# Flying / reach / protection / skulk filters are deferred to the
# `declare_blocker` primitive's per-pair gate at apply time.


def _put_into_declare_blockers(state, attacker_ids, defender_id=1):
    """Set up the state so `defender_id` (the OFF-active player) has
    priority during the DECLARE_BLOCKERS phase with `attacker_ids`
    already declared as attackers. The synthetic state defaults to
    active_player=0; we declare attackers via the
    `declare_attacker` primitive (which gates on the creature being
    on some battlefield + untapped + non-sick or hasted), then walk
    to the DECLARE_BLOCKERS phase and flip priority to the defender.

    `attacker_ids` may be empty (no attackers declared — defender
    still has the chance to declare zero blocks).
    """
    # `declare_attacker` doesn't care about the current phase — it
    # gates on creature status alone. Declare each attacker; the
    # primitive pushes onto `state.attackers` and taps the creature
    # (unless vigilance). We don't care about the bool return — the
    # caller pre-validated eligibility before passing the ids.
    for aid in attacker_ids:
        state.declare_attacker(aid)
    state.set_phase(PHASE_DECLARE_BLOCKERS)
    state.give_priority_to(defender_id)


# --- 36. DECLARE_BLOCKERS + 0 eligible blockers, 1 attacker ---------


def test_legal_decisions_declare_blockers_phase_zero_blockers_emits_no_blocks(
    engine,
):
    """DECLARE_BLOCKERS phase with NO eligible blockers but 1
    attacker: emit ONE DeclareBlockers (empty assignments) +
    PassPriority = 2 actions total. `(M+1)^N = (1+1)^0 = 1`
    iteration of the product loop produces a single empty
    assignment list."""
    state, _, p0_bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        # P1 (defender) has NO creatures.
        p1_battlefield_printings=[],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds == ["pass_priority", "declare_blockers"]
    db = [a for a in actions if a["kind"] == "declare_blockers"][0]
    assert db["assignments"] == []


# --- 37. DECLARE_BLOCKERS + 1 blocker, 1 attacker -------------------


def test_legal_decisions_declare_blockers_one_one_emits_two_assignments(
    engine,
):
    """1 blocker, 1 attacker: `(M+1)^N = (1+1)^1 = 2` assignments —
    {} (don't block) and {(b, a)} (block). Plus PassPriority = 3
    total. Locks in the product-shape with the smallest non-trivial
    N×M."""
    state, _, p0_bf_ids, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    # 2 DeclareBlockers + 1 PassPriority = 3 total.
    assert kinds.count("declare_blockers") == 2
    dbs = [a for a in actions if a["kind"] == "declare_blockers"]
    # Set of (assignments-as-frozenset) — {} (no block) + {(b, a)}.
    assignments_sets = {
        frozenset(tuple(p) for p in d["assignments"]) for d in dbs
    }
    assert assignments_sets == {
        frozenset(),
        frozenset({(p1_bf_ids[0], p0_bf_ids[0])}),
    }


# --- 38. DECLARE_BLOCKERS + 2 blockers, 1 attacker ------------------


def test_legal_decisions_declare_blockers_two_blockers_one_attacker_emits_four(
    engine,
):
    """2 blockers, 1 attacker: `(M+1)^N = (1+1)^2 = 4` assignments:
    {}, {(b0,a)}, {(b1,a)}, {(b0,a),(b1,a)}. Locks in product growth
    in the N dimension."""
    state, _, p0_bf_ids, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear", "Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("declare_blockers") == 4
    dbs = [a for a in actions if a["kind"] == "declare_blockers"]
    assignments_sets = {
        frozenset(tuple(p) for p in d["assignments"]) for d in dbs
    }
    a = p0_bf_ids[0]
    b0, b1 = p1_bf_ids[0], p1_bf_ids[1]
    assert assignments_sets == {
        frozenset(),
        frozenset({(b0, a)}),
        frozenset({(b1, a)}),
        frozenset({(b0, a), (b1, a)}),
    }


# --- 39. DECLARE_BLOCKERS + 1 blocker, 2 attackers ------------------


def test_legal_decisions_declare_blockers_one_blocker_two_attackers_emits_three(
    engine,
):
    """1 blocker, 2 attackers: `(M+1)^N = (2+1)^1 = 3` assignments:
    {} (don't block), {(b,a0)}, {(b,a1)}. Locks in product growth
    in the M dimension — each blocker picks among M+1 choices."""
    state, _, p0_bf_ids, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear", "Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("declare_blockers") == 3
    dbs = [a for a in actions if a["kind"] == "declare_blockers"]
    assignments_sets = {
        frozenset(tuple(p) for p in d["assignments"]) for d in dbs
    }
    a0, a1 = p0_bf_ids[0], p0_bf_ids[1]
    b = p1_bf_ids[0]
    assert assignments_sets == {
        frozenset(),
        frozenset({(b, a0)}),
        frozenset({(b, a1)}),
    }


# --- 40. Tapped blocker is excluded ---------------------------------


def test_legal_decisions_declare_blockers_tapped_blocker_excluded(
    engine,
):
    """A tapped creature can't block (CR 509.1a). With P1's only
    creature tapped, eligible_blockers is empty → only the empty
    assignments emit. Mirrors the DeclareAttackers tapped-creature
    exclusion test."""
    state, _, p0_bf_ids, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    state.tap_object(p1_bf_ids[0])
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    # Tapped blocker -> no eligible -> just the empty assignment.
    assert kinds == ["pass_priority", "declare_blockers"]
    db = [a for a in actions if a["kind"] == "declare_blockers"][0]
    assert db["assignments"] == []


# --- 41. Non-creature is excluded -----------------------------------


def test_legal_decisions_declare_blockers_non_creature_excluded(
    engine,
):
    """Only creatures can block (CR 509.1). A Forest (Land) on the
    defender's battlefield does NOT count toward the eligible-
    blockers set. With only a Forest + no creature, eligible is
    empty → just the empty assignment."""
    state, _, p0_bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Forest"],  # land, not creature
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds == ["pass_priority", "declare_blockers"]
    db = [a for a in actions if a["kind"] == "declare_blockers"][0]
    assert db["assignments"] == []


# --- 42. Summoning-sick creature IS included ------------------------


def test_legal_decisions_declare_blockers_summoning_sick_creature_included(
    engine,
):
    """A summoning-sick creature CAN block — CR 302.1 only restricts
    attacking and activated abilities with tap costs, not blocking.
    We use `_custom_registry_state` to put a Vanilla Bear (P0
    attacker) + a Hasty Bear (P0 alt) + a sick Vanilla Bear (P1
    defender) on the boards. P1's sick bear is created by
    `cast_creature` — which stamps `summoning_sick=true` on the
    battlefield slot — and then we verify it appears as an eligible
    blocker. This is the headline rule difference vs the
    DeclareAttackers test 27 (which excludes sick attackers)."""
    import mtg_bot_engine as _engine
    registry = _engine.Registry()
    forest_pid = registry.register_printing({
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
    })
    hasty_pid = registry.register_printing({
        "name": "Hasty Bear",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
        "keywords": ["haste"],
    })
    bear_pid = registry.register_printing({
        "name": "Vanilla Bear",
        "card_types": ["Creature"],
        "power": 2,
        "toughness": 2,
        "mana_cost": {"G": 1, "generic": 1},
    })
    state = _engine.make_state(
        registry,
        p0_library_printing_ids=[forest_pid],
        p1_library_printing_ids=[forest_pid],
        # P0 hand: Hasty Bear — we move it to bf (bypasses sickness
        # stamping because move_object doesn't apply summoning
        # sickness; that's the `cast_creature` path). Even if it
        # WERE marked sick, haste bypasses the gate.
        p0_hand_printing_ids=[hasty_pid],
        # P1 hand: Vanilla Bear — we'll cast it so the resulting bf
        # slot carries `summoning_sick = true`.
        p1_hand_printing_ids=[bear_pid],
    )
    # Put P0's Hasty Bear on the battlefield via move_object — fast
    # path, no mana, no sickness stamping (move_object preserves the
    # default `summoning_sick=false` on the new bf slot).
    p0_hand = state.hand_snapshot(0)
    p0_hasty_id = p0_hand[0]["id"]
    assert state.move_object(p0_hasty_id, ZONE_BATTLEFIELD, 0) is True
    # Cast P1's Vanilla Bear — stamps summoning_sick=true.
    p1_hand = state.hand_snapshot(1)
    p1_bear_id = p1_hand[0]["id"]
    state.add_mana(1, "G", 1)
    state.add_mana(1, "C", 1)
    assert state.cast_creature(p1_bear_id, 1) is True
    bf = state.battlefield_snapshot()
    p1_bear_bf = next(o for o in bf if o["id"] == p1_bear_id)
    assert p1_bear_bf["summoning_sick"] is True
    # P0 declares the Hasty Bear as an attacker.
    assert state.declare_attacker(p0_hasty_id) is True
    # Set up DECLARE_BLOCKERS phase with priority on defender (P1).
    state.set_phase(PHASE_DECLARE_BLOCKERS)
    state.give_priority_to(1)
    assert p0_hasty_id in state.get_attackers()

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    # Sick bear IS eligible → full (1+1)^1 = 2 assignments emit.
    assert kinds.count("declare_blockers") == 2
    dbs = [a for a in actions if a["kind"] == "declare_blockers"]
    assignments_sets = {
        frozenset(tuple(p) for p in d["assignments"]) for d in dbs
    }
    assert frozenset() in assignments_sets
    assert frozenset({(p1_bear_id, p0_hasty_id)}) in assignments_sets


# --- 43. Cap kicks in: N > 3 OR M > 3 emits only no-blocks ---------


def test_legal_decisions_declare_blockers_n_over_cap_emits_only_no_blocks(
    engine,
):
    """`n_blockers > DECLARE_BLOCKERS_PRODUCT_CAP` (= 3): emit only
    the "no blocks" empty assignment. Builds a 4-blocker / 1-
    attacker board — full product would be `(1+1)^4 = 16` dicts,
    capped to 1."""
    state, _, p0_bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"] * 4,  # N=4 > cap=3
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    # Capped: just 1 DeclareBlockers (the empty "no blocks" choice).
    assert kinds == ["pass_priority", "declare_blockers"]
    db = [a for a in actions if a["kind"] == "declare_blockers"][0]
    assert db["assignments"] == []


def test_legal_decisions_declare_blockers_m_over_cap_emits_only_no_blocks(
    engine,
):
    """`m_attackers > DECLARE_BLOCKERS_MAX_ATTACKERS` (= 3): emit
    only the empty assignment. Builds a 1-blocker / 4-attacker
    board — full product would be `(4+1)^1 = 5` dicts, capped to 1.
    The M-dimension cap fires independently of the N-dimension cap."""
    state, _, p0_bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"] * 4,  # M=4 > cap=3
        p1_battlefield_printings=["Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert kinds == ["pass_priority", "declare_blockers"]
    db = [a for a in actions if a["kind"] == "declare_blockers"][0]
    assert db["assignments"] == []


# --- 44. Wrong phase: no DeclareBlockers ---------------------------


@pytest.mark.parametrize(
    "phase",
    [PHASE_UNTAP, PHASE_MAIN1, PHASE_DECLARE_ATTACKERS, PHASE_MAIN2,
     PHASE_CLEANUP],
    ids=["UNTAP", "MAIN1", "DECLARE_ATTACKERS", "MAIN2", "CLEANUP"],
)
def test_legal_decisions_declare_blockers_wrong_phase_emits_no_declare_blockers(
    engine, phase,
):
    """DeclareBlockers only emits during the DECLARE_BLOCKERS phase.
    In any other phase (including the main phases and the
    DECLARE_ATTACKERS phase) no `declare_blockers` dict emits, even
    if the priority/active gates would otherwise be satisfied."""
    state, _, _, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    state.set_phase(phase)
    # Priority on the defender (P1) so the priority gate passes.
    state.give_priority_to(1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert "declare_blockers" not in kinds


# --- 45. Priority on active player: no DeclareBlockers -------------


def test_legal_decisions_declare_blockers_priority_on_active_emits_no_declare_blockers(
    engine,
):
    """You can't declare blockers on your own attackers. Even
    during the DECLARE_BLOCKERS phase, if `priority_player ==
    active_player`, no `declare_blockers` dict emits for the active
    player. Mirrors CR 509.1 — the defending player declares
    blockers, NOT the attacking player."""
    state, _, p0_bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)
    # Now flip priority back to the active player (P0) and query
    # legal_decisions for them.
    state.give_priority_to(0)
    assert state.active_player == 0
    assert state.get_priority_player() == 0

    actions = state.legal_decisions(0)

    kinds = _kinds(actions)
    # No DeclareBlockers — active player can't declare them.
    assert "declare_blockers" not in kinds


# --- 46. Non-empty stack: no DeclareBlockers ----------------------


def test_legal_decisions_declare_blockers_non_empty_stack_emits_no_declare_blockers(
    engine,
):
    """Non-empty stack during DECLARE_BLOCKERS: the
    `stack.is_empty()` gate fails, so DeclareBlockers does NOT
    emit (declaration happens while the stack is empty)."""
    state, _, p0_bf_ids, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)
    state.push_to_stack(p1_bf_ids[0], 1)

    actions = state.legal_decisions(1)

    kinds = _kinds(actions)
    assert "declare_blockers" not in kinds


# --- 47. Round-trip emitted DeclareBlockers via apply_action -------


def test_legal_decisions_declare_blockers_dicts_round_trip_via_apply_action(
    engine,
):
    """Every DeclareBlockers dict from `legal_decisions` must be a
    valid input to `apply_action` (regression for the round-23
    declare-blockers bridge). The "block everything" dict in
    particular should apply without raising — the primitive accepts
    `[[blocker_id, attacker_id], ...]` and dispatches one
    `declare_blocker` per pair."""
    state, _, p0_bf_ids, p1_bf_ids = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)
    dbs = [a for a in actions if a["kind"] == "declare_blockers"]
    # Find the non-empty-assignments dict.
    full_db = max(dbs, key=lambda d: len(d["assignments"]))
    assert len(full_db["assignments"]) == 1
    # Apply it; the primitive must accept the dict shape without
    # raising. Combat-state diagnostics live in test_combat.py — the
    # round-trip itself is the regression target here.
    state.apply_action(full_db)


# --- 48. DeclareBlockers dict carries the right wire shape ---------


def test_legal_decisions_declare_blockers_dict_shape_is_minimal(
    engine,
):
    """Every DeclareBlockers dict carries exactly two keys: `kind`
    + `assignments`. Mirrors `actions.rs:action_to_pydict`'s
    DeclareBlockers arm so a future-slice regression that adds an
    extra wire key (e.g. `blocker_order`) to the emitter but
    forgets to update the round-trip parser trips loudly."""
    state, _, p0_bf_ids, _ = _custom_registry_state(
        engine,
        p0_battlefield_printings=["Vanilla Bear"],
        p1_battlefield_printings=["Vanilla Bear"],
    )
    _put_into_declare_blockers(state, attacker_ids=p0_bf_ids, defender_id=1)

    actions = state.legal_decisions(1)
    dbs = [a for a in actions if a["kind"] == "declare_blockers"]
    assert len(dbs) >= 1
    for d in dbs:
        assert set(d.keys()) == {"kind", "assignments"}


# --- Edge-case coverage spanning multiple emission types ------------
#
# The tests below extend the per-emission-type coverage above with
# cross-cutting scenarios: mixed emissions in one state, sorcery-timing
# gates that suppress multiple emission types simultaneously,
# determinism / mutation-invalidation behavior, and other
# regression-flavored edge cases. Each test references the
# Round-24 emission slice it pins (skeleton / cast-spell /
# activate-ability / declare-attackers / declare-blockers) and the
# Python `legal.py` line range its behavior mirrors.


# --- 49. Full integration: Land + Creature in hand + bf mana producers
#
# Extends tests 4 (`...creature_plus_land...`), 14 (PlayLand +
# CastSpell order), 21 (CastSpell vs PassPriority order), and the
# ActivateAbility-emission-order test (line 1546). All four emission
# kinds in one state at once, with exact ordering enforced. The
# expected ordering — PassPriority, PlayLand, CastSpell,
# ActivateAbility — pins Python's natural source-order append at
# `legal.py:178` (PassPriority prepend), `:319-329` (PlayLand loop),
# `:335-445` (CastSpell loop), `:612` (ActivateAbility loop).


def test_legal_decisions_all_four_emission_kinds_present_in_priority_order(
    synthetic_state,
):
    """Synthetic state at MAIN1 with a Forest added to P0's hand
    exercises all four currently-emitted Action kinds in a single
    `legal_decisions` call. Pins the full emission order and the
    invariant that each kind's count matches the per-loop expectations
    (1 PassPriority + 1 PlayLand + 1 CastSpell (Elves dedup) + 5
    ActivateAbility (one per untapped Forest on P0's battlefield))."""
    forest_id = _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)

    kinds = _kinds(actions)
    # Counts.
    assert kinds.count("pass_priority") == 1
    assert kinds.count("play_land") == 1
    assert kinds.count("cast_spell") == 1
    assert kinds.count("activate_ability") == 5
    # Total = 1 + 1 + 1 + 5 = 8.
    assert len(actions) == 8
    # Ordering: pass_priority at 0, play_land at 1, cast_spell at 2,
    # then 5 activate_ability dicts.
    assert kinds[0] == "pass_priority"
    assert kinds[1] == "play_land"
    assert kinds[2] == "cast_spell"
    assert kinds[3:] == ["activate_ability"] * 5
    # PlayLand carries the Forest's id.
    assert actions[1]["card_id"] == forest_id


# --- 50. Mixed emission shape on opponent's turn collapses to pass --


def test_legal_decisions_off_active_player_collapses_full_mix_to_pass_priority(
    synthetic_state,
):
    """Same shape as the all-four-kinds test, but query P0 during P1's
    main phase. Sorcery-timing fails (`active_player != player_id`,
    Python `legal.py:28`), so the early-return at `:303-304` short-
    circuits BEFORE the PlayLand / CastSpell / ActivateAbility blocks.
    Only PassPriority emits — even though the state has a Land in
    hand, Creatures in hand, AND untapped mana producers on P0's
    battlefield. Pins the sorcery-timing gate's reach across all
    three emission types simultaneously."""
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    # Walk to P1's turn, then back to MAIN1 with priority on P0.
    while synthetic_state.active_player == 0:
        synthetic_state.advance_phase()
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.give_priority_to(0)
    assert synthetic_state.active_player == 1
    assert synthetic_state.get_priority_player() == 0

    actions = synthetic_state.legal_decisions(0)

    # All three sorcery-timing-gated emissions are suppressed.
    assert _kinds(actions) == ["pass_priority"]


# --- 51. Non-empty stack suppresses PlayLand + CastSpell + ActAbility


def test_legal_decisions_non_empty_stack_suppresses_all_sorcery_gated_emissions(
    synthetic_state,
):
    """Round 24's `legal-decisions-activate-ability` slice cites that
    Python's `legal.py:303-304` early-return on sorcery-timing failure
    short-circuits the entire ActivateAbility block at `:612`. With a
    non-empty stack the sorcery-timing gate's `stack.is_empty()`
    conjunct (`:30`) fails, so PlayLand, CastSpell, AND ActivateAbility
    all suppress at once — only PassPriority emits. Pins that
    instant-speed ActivateAbility for mana abilities is NOT a first-
    slice feature (deferred per the module docstring's
    "ActivateAbility extension" section)."""
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)
    synthetic_state.push_to_stack(P0_FIRST_BEARS_ID, 0)

    actions = synthetic_state.legal_decisions(0)

    # PlayLand + CastSpell + ActivateAbility all suppressed.
    assert _kinds(actions) == ["pass_priority"]


# --- 52. UNTAP-phase suppresses every sorcery-gated emission --------


def test_legal_decisions_untap_phase_suppresses_all_sorcery_gated_emissions(
    synthetic_state,
):
    """Phase=PHASE_UNTAP (0): sorcery-timing fails on the main-phase
    conjunct (Python `_is_sorcery_timing` at `legal.py:25-32`), so
    PlayLand / CastSpell / ActivateAbility all suppress. Extends the
    parametrized non-main-phase tests above (5 / 17 / the
    `...not_emitted_off_sorcery_timing` parametrize for AA) to
    confirm the simultaneous-suppression invariant on a mixed-hand
    state with mana producers on bf."""
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_UNTAP)

    actions = synthetic_state.legal_decisions(0)

    assert _kinds(actions) == ["pass_priority"]


# --- 53. Determinism: same state twice yields equal output ---------


def test_legal_decisions_is_deterministic_on_same_state(synthetic_state):
    """Two back-to-back calls on the same state produce identical
    dict lists — order-stable, value-stable. Pins that `legal_decisions`
    has no hidden iteration-order randomness (e.g., HashSet iteration)
    that would defeat MCTS-trajectory replay parity."""
    synthetic_state.set_phase(PHASE_MAIN1)

    a1 = synthetic_state.legal_decisions(0)
    a2 = synthetic_state.legal_decisions(0)

    assert a1 == a2


# --- 54. Determinism: cloned state yields equal output -------------


def test_legal_decisions_is_deterministic_across_clone_once(synthetic_state):
    """A cloned state produces an identical `legal_decisions` output
    to the original. Pins that `clone_once` preserves every
    enumeration-relevant field (zones, phases, priority, attackers)
    and that the emission code is a pure function of state."""
    synthetic_state.set_phase(PHASE_MAIN1)

    original = synthetic_state.legal_decisions(0)
    cloned = synthetic_state.clone_once().legal_decisions(0)

    assert original == cloned


# --- 55. Mutation invalidates a previously-emitted ActivateAbility -


def test_legal_decisions_activate_ability_count_drops_after_apply(
    synthetic_state,
):
    """Applying an emitted ActivateAbility (which taps the source)
    drops the per-source count by exactly one. Pins the
    emission-vs-apply correspondence — emit reads `tapped`, apply
    flips `tapped`, re-emit sees one fewer producer. Mirrors the
    rebroadcast invariant that downstream MCTS expansion relies on."""
    _empty_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    before = synthetic_state.legal_decisions(0)
    aa_before = [a for a in before if a["kind"] == "activate_ability"]
    assert len(aa_before) == 5

    # Apply one ActivateAbility — first untapped Forest taps.
    synthetic_state.apply_action(aa_before[0])

    after = synthetic_state.legal_decisions(0)
    aa_after = [a for a in after if a["kind"] == "activate_ability"]
    assert len(aa_after) == 4
    # The tapped source's id is not in the new emission.
    tapped_source_id = aa_before[0]["source_id"]
    after_sources = {a["source_id"] for a in aa_after}
    assert tapped_source_id not in after_sources


# --- 56. Out-of-range player_id: large positive raises IndexError -


def test_legal_decisions_large_out_of_range_player_id_raises_index_error(
    synthetic_state,
):
    """Regression on the existing player_id validation: a far-out-of-
    range id (100) raises the same IndexError shape that id=2 raises,
    locking in the message format ("player_id N out of range...")
    rather than swallowing into a generic Rust panic. Extends the
    test_legal_decisions_out_of_range_player_id_raises_index_error
    test above to confirm the validation isn't a special-case at id=2."""
    with pytest.raises(IndexError, match=r"player_id 100 out of range"):
        synthetic_state.legal_decisions(100)


# --- 57. Empty-state edge: nothing in any zone emits only PassPriority


def test_legal_decisions_minimal_state_emits_only_pass_priority(engine):
    """A minimal hand-rolled state with one Forest each in both
    players' libraries, empty hands, empty battlefields, MAIN1 phase
    → ONLY PassPriority. Confirms the enumerator doesn't have a
    hidden floor (e.g., always-emit-one-DeclareAttackers) that would
    leak when every per-kind loop iterates zero items."""
    import mtg_bot_engine as _engine
    registry = _engine.Registry()
    forest_pid = registry.register_printing({
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
    })
    state = _engine.make_state(
        registry,
        p0_library_printing_ids=[forest_pid],
        p1_library_printing_ids=[forest_pid],
        p0_hand_printing_ids=[],
        p1_hand_printing_ids=[],
    )
    state.set_phase(PHASE_MAIN1)

    actions = state.legal_decisions(0)

    assert _kinds(actions) == ["pass_priority"]


# --- 58. DECLARE_ATTACKERS with no eligible attackers (empty bf) ---


def test_legal_decisions_declare_attackers_with_empty_bf_emits_pass_then_empty_subset(
    engine,
):
    """Synthetic-like edge case: DECLARE_ATTACKERS phase with NO
    creatures on the active player's battlefield → exactly
    `[PassPriority, DeclareAttackers([])]`. Extends test 23
    (`...declare_attackers_phase_zero_creatures_emits_empty_subset`)
    by also asserting the precise emission count = 2 and that no
    other Action kind sneaks in (the powerset-of-empty edge case is
    structurally identical to the cap-fires case, so this test
    also doubles as a regression on the cap-vs-empty divergence)."""
    import mtg_bot_engine as _engine
    registry = _engine.Registry()
    forest_pid = registry.register_printing({
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
    })
    state = _engine.make_state(
        registry,
        p0_library_printing_ids=[forest_pid],
        p1_library_printing_ids=[forest_pid],
        p0_hand_printing_ids=[],
        p1_hand_printing_ids=[],
    )
    state.set_phase(PHASE_DECLARE_ATTACKERS)
    state.give_priority_to(0)

    actions = state.legal_decisions(0)

    assert _kinds(actions) == ["pass_priority", "declare_attackers"]
    da = [a for a in actions if a["kind"] == "declare_attackers"][0]
    assert da["attacker_ids"] == []
    assert len(actions) == 2


# --- 59. PassPriority is ALWAYS emitted, exactly once, at index 0 --


@pytest.mark.parametrize(
    "phase,prio_player",
    [
        (PHASE_UNTAP, 0),
        (PHASE_MAIN1, 0),
        (PHASE_DECLARE_ATTACKERS, 0),
        (PHASE_DECLARE_BLOCKERS, 0),
        (PHASE_COMBAT_DAMAGE, 0),
        (PHASE_MAIN2, 0),
        (PHASE_CLEANUP, 0),
    ],
    ids=[
        "UNTAP", "MAIN1", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS",
        "COMBAT_DAMAGE", "MAIN2", "CLEANUP",
    ],
)
def test_legal_decisions_pass_priority_always_emitted_once_at_index_zero(
    synthetic_state, phase, prio_player,
):
    """Strong regression: across every phase, PassPriority is
    present exactly once and is at index 0. Pins Python's `legal.py:178`
    semantic (PassPriority is the universal prepend) across the
    full phase cycle, not just the main-phase + sorcery-timing
    happy paths. Catches a future-slice regression where a per-
    phase branch forgets to seed `actions = [PassPriority()]`
    before the per-kind loops."""
    synthetic_state.set_phase(phase)
    synthetic_state.give_priority_to(prio_player)

    actions = synthetic_state.legal_decisions(prio_player)

    kinds = _kinds(actions)
    assert kinds[0] == "pass_priority"
    assert kinds.count("pass_priority") == 1


# --- 60. Round-trip every Action kind via clone_once + apply_action


def test_legal_decisions_every_emitted_action_round_trips_via_apply_action(
    synthetic_state,
):
    """For each Action dict emitted by `legal_decisions`, applying it
    on a fresh `clone_once()` of the state must succeed without
    raising. Pins the round-23/24/25 emission-vs-dispatch contract
    across the full emission set (PassPriority, PlayLand, CastSpell,
    ActivateAbility — DeclareAttackers / DeclareBlockers are tested
    in their per-kind round-trip tests above, which use a custom-
    registry state with declared attackers). With the round-25
    ActivateAbility primitive flip, every dispatch arm now backs a
    real primitive — a future regression that adds a new Action kind
    to the enumerator but forgets the `apply_action` arm trips here."""
    _put_forest_in_hand(synthetic_state, 0)
    synthetic_state.set_phase(PHASE_MAIN1)

    actions = synthetic_state.legal_decisions(0)
    # Sanity: we expect all four currently-emitted kinds present.
    kinds = set(_kinds(actions))
    assert kinds == {
        "pass_priority", "play_land", "cast_spell", "activate_ability",
    }

    for action in actions:
        # Each Action applies to a fresh clone — the dispatcher must
        # accept the dict shape without raising. Downstream rejection
        # (e.g., mana cost not paid for the CastSpell) is acceptable
        # — we catch ValueError to allow that. KeyError / TypeError
        # / NotImplementedError would all signal a wire-shape or
        # missing-dispatch-arm regression.
        clone = synthetic_state.clone_once()
        try:
            clone.apply_action(action)
        except ValueError:
            # Downstream gate rejected the action — fine for the
            # round-trip check; the dict shape was accepted.
            pass
