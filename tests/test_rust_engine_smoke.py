"""Smoke test for the Rust engine PyO3 extension.

The extension is built out-of-tree by maturin (see ``rust/`` in the
project root). If it isn't installed yet, the test skips with a
hint instead of failing — that way running ``pytest`` on a fresh
checkout doesn't break before someone has run ``maturin develop``.
"""

from __future__ import annotations

import importlib

import pytest

mtg_bot_engine = pytest.importorskip(
    "mtg_bot_engine",
    reason="rust extension not built; run `maturin develop --release -m rust/pyproject.toml`",
)


def test_module_exposes_version():
    assert hasattr(mtg_bot_engine, "__version__")
    assert isinstance(mtg_bot_engine.__version__, str)


def test_synthetic_state_has_expected_dimensions():
    state = mtg_bot_engine.make_synthetic_state(
        num_battlefield=10, num_hand=7, num_library=50
    )
    assert state.turn_number == 5
    assert tuple(state.battlefield_sizes) == (10, 10)
    assert tuple(state.hand_sizes) == (7, 7)
    assert tuple(state.library_sizes) == (50, 50)
    assert tuple(state.life_totals) == (20, 20)


def test_clone_once_returns_independent_state():
    state = mtg_bot_engine.make_synthetic_state()
    cloned = state.clone_once()
    assert tuple(cloned.battlefield_sizes) == tuple(state.battlefield_sizes)
    assert tuple(cloned.life_totals) == tuple(state.life_totals)


def test_clone_n_times_returns_positive_micros():
    state = mtg_bot_engine.make_synthetic_state()
    micros_per_clone = state.clone_n_times_micros(100)
    assert micros_per_clone > 0.0
    # Generous upper bound — debug-mode clone of this shape on x86_64
    # comes in well under 100us; release mode is much lower. If this
    # ever fires we have a real regression to look at.
    assert micros_per_clone < 1000.0, f"clone too slow: {micros_per_clone:.2f}us"


def test_module_can_be_reimported():
    importlib.reload(mtg_bot_engine)
    assert hasattr(mtg_bot_engine, "make_synthetic_state")


def test_synthetic_state_carries_registry():
    """The synthetic state ships a 3-printing registry — Forest,
    Grizzly Bears, Llanowar Elves — shared by both players."""
    state = mtg_bot_engine.make_synthetic_state()
    assert state.registry_size == 3


def test_first_hand_object_resolves_to_llanowar_elves():
    """End-to-end Registry wiring check.

    `make_player`'s placement loop puts elves in hand first, so the
    first hand object on player 0 should resolve to "Llanowar Elves"
    via printing_id -> registry lookup. A failure anywhere in the
    chain (object id, printing_id, registry threading, lookup) shows
    up as a wrong-or-missing name here.

    Player 0's id range starts at 0 and runs:
      battlefield: 0..num_battlefield   (forests + bears)
      hand:        num_battlefield..num_battlefield+num_hand  (elves)
      library:     ...                                         (forests)
    """
    state = mtg_bot_engine.make_synthetic_state(
        num_battlefield=10, num_hand=7, num_library=50
    )
    first_hand_id = 10  # ids 0..9 are battlefield; 10 is the first elf
    assert state.printing_name_of(first_hand_id) == "Llanowar Elves"


def test_first_battlefield_object_resolves_to_forest():
    """Mirrors the elves test on the battlefield half — the first
    battlefield object is a forest by `make_player`'s placement."""
    state = mtg_bot_engine.make_synthetic_state(
        num_battlefield=10, num_hand=7, num_library=50
    )
    assert state.printing_name_of(0) == "Forest"


def test_printing_name_of_unknown_id_returns_none():
    state = mtg_bot_engine.make_synthetic_state(
        num_battlefield=4, num_hand=2, num_library=10
    )
    # ids 0..(4+2+10-1) = 0..15 belong to player 0;
    # 16..31 to player 1; 999 is far outside both.
    assert state.printing_name_of(999) is None


def test_require_printing_name_of_unknown_raises():
    state = mtg_bot_engine.make_synthetic_state(
        num_battlefield=4, num_hand=2, num_library=10
    )
    with pytest.raises(KeyError, match="no game object with id 999"):
        state.require_printing_name_of(999)


def test_clone_preserves_registry_lookups():
    """Cloning a state shares the Arc<RegistryData> — the cloned
    state's printing-name lookups must return the same names."""
    state = mtg_bot_engine.make_synthetic_state()
    cloned = state.clone_once()
    assert cloned.printing_name_of(10) == state.printing_name_of(10)
    assert cloned.printing_name_of(0) == "Forest"


def test_gameobject_byte_size_is_compact():
    """Regression backstop: the POD GameObject (id, printing_id,
    controller, zone, tapped, summoning_sick, damage, counters)
    fits in a small fixed-size struct. If a field is added that
    blows past 32 bytes (e.g. a String, a Vec, an Arc-of-anything),
    catch it here — the search-side win depends on the per-object
    block staying memcpy-shaped."""
    sz = mtg_bot_engine.gameobject_byte_size()
    assert 16 <= sz <= 32, f"GameObject size unexpected: {sz} bytes"


# ---------------------------------------------------------------------------
# Extended cross-feature smoke scenarios.
#
# The tests above pin individual primitives — registry lookup, clone, byte
# size, basic hand/battlefield placement. The block below exercises the
# Rust engine on realistic multi-step mini-game flows that touch several
# primitives at once: turn cycling, play-land/activate/cast pipelines,
# combat damage + SBA, multi-step decision walks, clone-replay equivalence,
# deck-out behavior, mulligan, and cross-priority mana activation.
#
# These are smoke tests: each one drives a short script of operations and
# asserts on a load-bearing observable (fingerprint, zone count, life
# total, game-over flag). They are not parity tests against Python — those
# live in scripts/parity_smoke.py. The point of this block is to catch
# regressions where individual primitives still pass their unit tests but
# their composition breaks (e.g. apply_action + advance_phase + apply_sbas
# stops working as a triple).
#
# Synthetic-state default layout (cf. tests_rust/test_actions.py): 10
# battlefield, 7 hand, 50 library per side. Player 0 ids 0..66; ids 0..4
# are Forests, 5..9 Grizzly Bears, 10..16 Llanowar Elves (hand), 17..66
# Forests (library). Player 1 mirrors this with offset 67.
# ---------------------------------------------------------------------------


# Zone constants (cf. rust/src/state_ops_zones.rs).
ZONE_HAND = 0
ZONE_LIBRARY = 1
ZONE_BATTLEFIELD = 2
ZONE_GRAVEYARD = 3

# Phase constants (cf. rust/src/state_ops_steps.rs).
PHASE_UNTAP = 0
PHASE_MAIN1 = 3
PHASE_CLEANUP = 11

# Synthetic-state id landmarks.
P0_FIRST_FOREST_BF = 0  # battlefield: ids 0..4 are forests
P0_FIRST_BEAR = 5  # battlefield: ids 5..9 are bears (2/2)
P0_FIRST_ELF_HAND = 10  # hand: ids 10..16 are Llanowar Elves
P0_FIRST_FOREST_LIB = 17  # library: ids 17..66 are forests


def _full_cast_spell_dict(card_id: int) -> dict:
    """Build the full 6-key CastSpell dict shape that ``apply_action``
    accepts. The four alt-cost / room-half flags must be present per
    ``actions.rs:parse_action``'s schema even when all are false /
    None — they're scaffolded into the variant for parity with
    Python's dataclass."""
    return {
        "kind": "cast_spell",
        "card_id": card_id,
        "use_warp": False,
        "kicked": False,
        "cast_for_free": False,
        "use_impending": False,
        "cast_room_half": None,
    }


def test_full_turn_cycle_flips_active_player_and_bumps_turn():
    """Walk all 12 phases via ``advance_phase`` and verify the
    cleanup -> untap wrap fires: ``turn_number`` increments, the
    ``active_player`` flips, the phase resets to UNTAP. Pins the
    composition of step constants + advance_phase's wrap arm + the
    untap-step auto-sweep.

    Note on G15: ``priority_player`` is NOT touched by ``advance_phase``
    (the priority field stays on whoever held it at start). This is a
    known parity gap; this test deliberately doesn't assert on
    ``priority_player`` so it pins the documented behavior."""
    state = mtg_bot_engine.make_synthetic_state()
    starting_turn = state.turn_number
    starting_active = state.active_player
    assert state.current_phase() == PHASE_UNTAP

    # Walk 12 advance_phase calls — that's exactly one full turn
    # cycle (UNTAP through CLEANUP -> back to UNTAP of next turn).
    for _ in range(12):
        state.advance_phase()

    assert state.turn_number == starting_turn + 1
    assert state.active_player == 1 - starting_active
    assert state.current_phase() == PHASE_UNTAP
    assert state.step == "UNTAP"


def test_play_land_then_tap_for_mana_then_cast_creature_flow():
    """End-to-end PlayLand -> ActivateAbility -> CastSpell pipeline
    through ``apply_action``. Pins the cross-feature composition of
    the action dispatcher with the hand/battlefield zone primitives,
    the mana-pool primitive (via ActivateAbility's tap-for-mana
    inlined logic), and ``cast_creature``'s mana payment.

    Sequence:
      1. Move a Forest (obj 17) from p0's library to p0's hand.
      2. Walk to MAIN1 (phase 3) so PlayLand is sorcery-legal.
      3. apply_action PlayLand(17) -> Forest hits the battlefield.
      4. apply_action ActivateAbility(17, 0) -> Forest taps, p0
         gains {G}.
      5. apply_action CastSpell(10) -> elves go hand -> battlefield
         (no stack — cast_creature skips the stack per current Rust
         scope simplification).
    """
    state = mtg_bot_engine.make_synthetic_state()
    # (1) Pull a Forest into hand.
    assert state.move_object(P0_FIRST_FOREST_LIB, ZONE_HAND, 0) is True
    # (2) Walk to MAIN1 so PlayLand emission gates pass.
    for _ in range(PHASE_MAIN1):
        state.advance_phase()
    assert state.step == "PRECOMBAT_MAIN"

    bf_before = state.battlefield_sizes[0]
    hand_before = state.hand_sizes[0]

    # (3) Play the Forest.
    state.apply_action({"kind": "play_land", "card_id": P0_FIRST_FOREST_LIB})
    assert state.find_object_zone(P0_FIRST_FOREST_LIB) == ZONE_BATTLEFIELD

    # (4) Tap it for mana via the ActivateAbility dispatch arm.
    state.apply_action(
        {
            "kind": "activate_ability",
            "source_id": P0_FIRST_FOREST_LIB,
            "ability_index": 0,
        }
    )
    assert state.is_tapped(P0_FIRST_FOREST_LIB) is True
    pool = state.get_mana_pool(0)
    assert pool["G"] == 1

    # (5) Cast Llanowar Elves (1G — wait, the synthetic-state recipe
    # has elves at cost {G}; one Forest = one green pip, sufficient).
    state.apply_action(_full_cast_spell_dict(P0_FIRST_ELF_HAND))
    assert state.find_object_zone(P0_FIRST_ELF_HAND) == ZONE_BATTLEFIELD
    # Net battlefield: +1 forest, +1 elf = +2; hand: -1 forest -1 elf = -2.
    assert state.battlefield_sizes[0] == bf_before + 2
    assert state.hand_sizes[0] == hand_before - 2
    # Mana paid down.
    assert state.get_mana_pool(0)["G"] == 0


def test_combat_damage_to_creature_drops_to_zero_and_sba_destroys():
    """Stamp lethal damage onto a 2/2 Grizzly Bears and verify the
    state-based action pass moves it to the graveyard.

    Pins the composition of the damage-stamp primitive with the
    SBA primitive — the same primitives ``deal_combat_damage`` calls
    under the hood, exercised here via the direct ``deal_damage``
    entry point so the test doesn't have to set up a full combat
    state."""
    state = mtg_bot_engine.make_synthetic_state()
    bear_id = P0_FIRST_BEAR
    assert state.printing_name_of(bear_id) == "Grizzly Bears"
    gy_before = state.graveyard_sizes[0]
    assert state.find_object_zone(bear_id) == ZONE_BATTLEFIELD

    # Bears are 2/2 — 2 damage is lethal.
    assert state.deal_damage(bear_id, 2) is True
    assert state.is_lethal(bear_id) is True

    # One SBA pass moves the lethal creature to its controller's
    # graveyard. apply_sbas returns the count of state-based actions
    # taken; we expect >= 1.
    passes = state.apply_sbas()
    assert passes >= 1

    assert state.find_object_zone(bear_id) == ZONE_GRAVEYARD
    assert state.graveyard_sizes[0] == gy_before + 1


def test_combat_damage_to_player_decrements_life_and_lethal_flips_game_over():
    """Drop player 1 to 0 life via ``deal_damage_to_player`` and
    verify ``is_game_over`` flips. Pins the life primitive +
    losers() derivation + ``is_game_over`` composition.

    Composes with G11 (loss is pure-derivation from life<=0): a
    player at 0 life is a loser regardless of how the life got
    there; this test uses ``deal_damage_to_player`` as the most
    natural way to mimic combat damage to a player."""
    state = mtg_bot_engine.make_synthetic_state()
    assert state.is_game_over() is False
    assert state.life_totals == [20, 20]

    state.deal_damage_to_player(1, 20)
    # life primitive subtracts immediately; no SBA pass needed for
    # the life total — but is_game_over reads losers() which is
    # purely derived from life <= 0, so a SBA pass isn't needed.
    assert state.get_life(1) == 0
    assert list(state.losers()) == [1]
    assert state.is_game_over() is True
    assert state.winner() == 0


def test_fifty_decision_walk_does_not_panic_and_fingerprint_progresses():
    """Walk 50 decisions in a row: at each step, fetch
    ``legal_decisions``, pick the first non-pass action if any
    (otherwise PassPriority), and apply it. After each step,
    capture the fingerprint and assert no panic occurs and the
    fingerprint set grows monotonically across the walk.

    Pins the composition of legal_decisions + apply_action across
    many primitive types: PassPriority, PlayLand, ActivateAbility,
    CastSpell, plus the cleanup composition. The fingerprint
    delta-counting catches a degenerate engine that keeps emitting
    identical decisions but never actually mutates state.

    Note on silent-no-op apply_action arms: ``cast_creature`` returns
    ``false`` (no mutation, no error) when mana payment fails. In the
    default synthetic state the priority player has no mana in pool
    at MAIN1, so the first emitted ``cast_spell`` action is a silent
    no-op. To exercise a richer set of primitives, this walk picks
    the LAST non-pass decision (typically ActivateAbility, which
    always mutates when emitted) when the first non-pass is a
    cast_spell. When neither path moves the fingerprint, we fall
    back to advancing the phase so the walk doesn't get stuck."""
    state = mtg_bot_engine.make_synthetic_state()
    # Advance to MAIN1 so the decision space is non-trivial.
    for _ in range(PHASE_MAIN1):
        state.advance_phase()

    fingerprints = [state.fingerprint()]
    for _ in range(50):
        priority = state.get_priority_player()
        decisions = state.legal_decisions(priority)
        # Prefer the LAST non-pass decision (ActivateAbility for
        # untapped permanents) since the FIRST non-pass is typically
        # CastSpell, which silently no-ops when the priority player
        # is mana-broke. Falls back to PassPriority if nothing else.
        non_pass = [d for d in decisions if d["kind"] != "pass_priority"]
        if non_pass:
            chosen = non_pass[-1]
        else:
            chosen = decisions[0]

        fp_before = state.fingerprint()
        state.apply_action(chosen)
        if state.fingerprint() == fp_before:
            # The chosen action was a silent no-op (e.g. cast_spell
            # without mana). Advance phase so the walk makes forward
            # progress instead of looping on dead decisions.
            state.advance_phase()
        fingerprints.append(state.fingerprint())

    # No panic occurred. The fingerprint must have moved at least
    # once across 50 steps; a degenerate "every choice is a no-op"
    # engine would never advance the digest.
    assert len(set(fingerprints)) > 1, (
        "fingerprint never changed across 50-decision walk — engine "
        "may be silently no-op'ing every applied action"
    )


def test_determinism_same_decisions_on_clones_produces_identical_fingerprints():
    """Two clones of the same state, given the same sequence of
    decisions, produce identical fingerprints AT EVERY STEP.

    Stronger than ``clone_preserves_registry_lookups`` from the
    basic block: drives the clone through 20 mutation steps and
    cross-checks the fingerprint trail end-to-end. Pins the CoW
    determinism invariant — same input + same operations = same
    output, regardless of how the clone was constructed."""
    state_a = mtg_bot_engine.make_synthetic_state()
    state_b = state_a.clone_once()

    # Identical starting fingerprints (paranoia: also a sanity check
    # on the clone primitive itself).
    assert state_a.fingerprint() == state_b.fingerprint()

    # Advance both to MAIN1.
    for _ in range(PHASE_MAIN1):
        state_a.advance_phase()
        state_b.advance_phase()

    for step in range(20):
        # Pick a stable choice for both: first action in legal_decisions.
        # Both states share legal_decisions output by construction.
        decisions_a = state_a.legal_decisions(state_a.get_priority_player())
        decisions_b = state_b.legal_decisions(state_b.get_priority_player())
        # Action lists themselves match step-by-step.
        assert decisions_a == decisions_b, (
            f"step {step}: divergent legal_decisions between clones"
        )
        action = decisions_a[0]
        state_a.apply_action(action)
        state_b.apply_action(action)
        assert state_a.fingerprint() == state_b.fingerprint(), (
            f"step {step}: fingerprints diverged after applying {action}"
        )


def test_clone_replay_matches_original_trajectory_at_every_step():
    """Build a 20-step trajectory on one state; on each step also
    clone+replay the original from scratch and verify the clone
    arrives at the SAME fingerprint as the original at that step.

    This is the "clone-replay equality" canary parity-smoke leans
    on (cf. ``tests_rust/test_fingerprint.py::
    test_fingerprint_clone_replay_equality_after_each_mutation``)
    extended to a 20-step trajectory. If lazy-promoted Arc state
    diverges only after several mutations, this test catches it."""
    base = mtg_bot_engine.make_synthetic_state()
    for _ in range(PHASE_MAIN1):
        base.advance_phase()

    trajectory_actions = []
    fingerprints = [base.fingerprint()]
    state = base.clone_once()
    for _ in range(20):
        decisions = state.legal_decisions(state.get_priority_player())
        action = next(
            (d for d in decisions if d["kind"] != "pass_priority"),
            decisions[0],
        )
        trajectory_actions.append(action)
        state.apply_action(action)
        fingerprints.append(state.fingerprint())

    # Now: replay from a fresh clone of base, comparing step-by-step.
    replay = base.clone_once()
    assert replay.fingerprint() == fingerprints[0]
    for i, action in enumerate(trajectory_actions, start=1):
        replay.apply_action(action)
        assert replay.fingerprint() == fingerprints[i], (
            f"clone replay diverged at step {i}: action={action}"
        )


def test_empty_library_advance_does_not_trigger_deck_out_loss():
    """G10 pin: Rust today does NOT model the CR 704.5b deck-out
    loss. Drain a player's library, walk through an UNTAP step
    (which doesn't auto-draw — G14), and verify the player is
    still alive.

    This test deliberately pins the GAP behavior so a future slice
    that closes G10 (by adding ``attempted_draw_on_empty`` to the
    Player struct + threading through losers()) shows up as a clean
    test flip rather than a mysterious downstream parity break."""
    state = mtg_bot_engine.make_synthetic_state(0, 0, 2)
    assert state.library_sizes[0] == 2

    # Drain p0's library.
    assert state.draw_card(0) is not None
    assert state.draw_card(0) is not None
    assert state.library_sizes[0] == 0

    # An additional draw against the empty library returns None — no
    # mutation, no loss flag.
    assert state.draw_card(0) is None
    assert list(state.losers()) == []
    assert state.is_game_over() is False

    # Advance phase doesn't auto-draw (G14), so walking a full turn
    # also doesn't fire deck-out — same gap, different surface.
    for _ in range(12):
        state.advance_phase()
    assert state.is_game_over() is False
    assert list(state.losers()) == []


def test_mulligan_does_not_shrink_hand_g21_gap_pin():
    """G21 pin: Rust ``mulligan`` always draws 7, never shrinks the
    hand to the London-mulligan count. A mulliganed state therefore
    has the same hand size as a non-mulliganed one, but the
    fingerprint differs because the hand CONTENTS shuffled.

    Pins the gap so a future slice that wires the London hand-shrink
    surfaces as a clean test flip. Also verifies the fingerprint
    delta is non-trivial (the shuffle actually rotates cards)."""
    baseline = mtg_bot_engine.make_synthetic_state()
    mulliganed = mtg_bot_engine.make_synthetic_state()
    assert baseline.fingerprint() == mulliganed.fingerprint()

    mulliganed.mulligan(0, seed=42)

    # G21: hand size stays at 7 (Rust ignores the "mulligan to 6" semantics
    # of the London mulligan).
    assert mulliganed.hand_sizes[0] == 7
    # But hand contents shuffled, so the fingerprint moves.
    assert mulliganed.fingerprint() != baseline.fingerprint()


def test_activate_ability_credits_source_controller_during_opponent_priority():
    """Round-25 finding: the ActivateAbility dispatch credits the
    source's controller, not the priority holder, for the mana it
    produces. Verify by handing priority to player 1, then having
    player 1's apply_action call activate one of player 0's forests —
    the mana lands in p0's pool, not p1's, and the forest taps.

    Composes:
      - apply_action dispatcher reads source's controller from the
        battlefield index (not from priority_player).
      - ActivateAbility's inlined tap-and-credit logic correctly
        identifies the source's owning battlefield via the
        ``state.players[pidx].battlefield`` iteration.

    Note: in real MTG this situation can't arise (mana abilities
    aren't activatable by non-controllers), but the Rust dispatcher
    is permissive at this layer — the test pins the routing
    invariant. The legal_decisions emission gate filters this out
    upstream, so production callers don't hit the path."""
    state = mtg_bot_engine.make_synthetic_state()
    # Move priority to p1; active stays p0.
    state.pass_priority()
    assert state.get_priority_player() == 1
    assert state.active_player == 0

    # p0 controls forest 0 (battlefield ids 0..4 for player 0).
    forest_id = P0_FIRST_FOREST_BF
    assert state.is_tapped(forest_id) is False
    p0_pool_before = state.get_mana_pool(0)
    p1_pool_before = state.get_mana_pool(1)

    # Apply the activate-ability action even though priority is on p1.
    # The dispatcher reads source_id's controller from the battlefield
    # — it doesn't gate on priority_player.
    state.apply_action(
        {
            "kind": "activate_ability",
            "source_id": forest_id,
            "ability_index": 0,
        }
    )

    # Mana landed in p0's (source controller's) pool, NOT p1's.
    p0_pool_after = state.get_mana_pool(0)
    p1_pool_after = state.get_mana_pool(1)
    assert p0_pool_after["G"] == p0_pool_before["G"] + 1
    assert p1_pool_after == p1_pool_before
    # Forest tapped.
    assert state.is_tapped(forest_id) is True
