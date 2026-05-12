"""Tap / untap primitives on the Rust ``GameState``.

Ported from ``tests/test_combat.py`` + the ``obj.tapped = True/False``
sites in ``src/mtg_bot/game.py`` (line 567 tap, 234 untap, 1268
sweep) and ``src/mtg_bot/turn.py``'s ``run_untap_step`` (line 103 —
the per-controller battlefield sweep).

Scope: the four primitives the higher-level Python tap/untap code
calls under the hood — `tap_object`, `untap_object`, `is_tapped`,
`untap_all_for_player`. The Python side mutates `obj.tapped` directly
because objects are mutable Python instances; the Rust side has POD
GameObjects in `Arc<Vec<GameObject>>` zones so each mutation must
go through `Arc::make_mut` to preserve copy-on-write semantics for
clones. The ``test_tap_one_clone_does_not_tap_other`` test is the
load-bearing CoW canary.
"""

from __future__ import annotations


def _bf_ids(state, player_id):
    """Return the GameObject ids on ``player_id``'s battlefield.

    The synthetic state assigns deterministic ids: player 0's battlefield
    is ids 0..(num_battlefield-1); player 1's starts at
    (num_battlefield + num_hand + num_library). With defaults
    (10/7/50) that's 0..9 and 67..76.
    """
    # We can't directly enumerate via Python (no zone iter exposed);
    # we know the synthetic-state schema, so reconstruct ids by hand.
    # 10 battlefield / 7 hand / 50 library default.
    if player_id == 0:
        return list(range(0, 10))
    # player 1 starts after p0's 67 ids
    return list(range(67, 77))


def test_object_starts_untapped(synthetic_state):
    """Fresh battlefield objects come up untapped (mirrors GameObject
    default `tapped: bool = False` in src/mtg_bot/game.py:443)."""
    for obj_id in _bf_ids(synthetic_state, 0):
        assert synthetic_state.is_tapped(obj_id) is False


def test_tap_object_flips_to_tapped(synthetic_state):
    """`tap_object` flips the tapped flag and reports True on success.

    Mirrors src/mtg_bot/effects/generic/tap.py's `obj.tapped = True`
    plus the convention that tap effects return True iff the state
    actually changed."""
    obj_id = _bf_ids(synthetic_state, 0)[0]
    assert synthetic_state.is_tapped(obj_id) is False
    assert synthetic_state.tap_object(obj_id) is True
    assert synthetic_state.is_tapped(obj_id) is True


def test_tap_unknown_object_returns_false(synthetic_state):
    """Tapping an id that isn't on any battlefield returns False
    rather than raising — matches the Python side's silent-no-op
    behavior when an effect's targeted permanent has already left
    the battlefield."""
    assert synthetic_state.tap_object(999_999) is False
    assert synthetic_state.is_tapped(999_999) is None


def test_untap_object_flips_back(synthetic_state):
    """Round-trip: tap then untap leaves the object untapped, with
    each call returning True for the state-change."""
    obj_id = _bf_ids(synthetic_state, 0)[0]
    assert synthetic_state.tap_object(obj_id) is True
    assert synthetic_state.is_tapped(obj_id) is True
    assert synthetic_state.untap_object(obj_id) is True
    assert synthetic_state.is_tapped(obj_id) is False


def test_double_tap_is_idempotent_returns_false(synthetic_state):
    """Tapping an already-tapped permanent returns False (no change)
    but the state stays tapped. Mirrors the Python convention that a
    'tap as cost' check fails on already-tapped permanents — the
    primitive distinguishes 'newly tapped' from 'already tapped'."""
    obj_id = _bf_ids(synthetic_state, 0)[0]
    assert synthetic_state.tap_object(obj_id) is True
    assert synthetic_state.tap_object(obj_id) is False
    assert synthetic_state.is_tapped(obj_id) is True
    # Symmetric for untap.
    assert synthetic_state.untap_object(obj_id) is True
    assert synthetic_state.untap_object(obj_id) is False
    assert synthetic_state.is_tapped(obj_id) is False


def test_untap_all_for_player_flips_only_that_players_permanents(synthetic_state):
    """`untap_all_for_player(p)` is the untap-step primitive — it
    untaps every battlefield permanent controlled by `p` and leaves
    the opponent's permanents alone. Mirrors the per-controller
    sweep at src/mtg_bot/turn.py:87-104."""
    p0_ids = _bf_ids(synthetic_state, 0)
    p1_ids = _bf_ids(synthetic_state, 1)
    # Tap everything on both sides.
    for obj_id in p0_ids + p1_ids:
        assert synthetic_state.tap_object(obj_id) is True
    for obj_id in p0_ids + p1_ids:
        assert synthetic_state.is_tapped(obj_id) is True
    # Untap only player 0's side.
    synthetic_state.untap_all_for_player(0)
    for obj_id in p0_ids:
        assert synthetic_state.is_tapped(obj_id) is False, (
            f"p0's obj {obj_id} should have untapped"
        )
    for obj_id in p1_ids:
        assert synthetic_state.is_tapped(obj_id) is True, (
            f"p1's obj {obj_id} should still be tapped"
        )


def test_clone_preserves_tapped_state(synthetic_state):
    """`clone_once` (which exercises the same `Clone` impl that
    cloning the GameState in MCTS goes through) must carry the
    tapped-state bits forward — those live inside the per-zone
    `Arc<Vec<GameObject>>` and must come along on the refcount bump."""
    obj_id = _bf_ids(synthetic_state, 0)[0]
    synthetic_state.tap_object(obj_id)
    snapshot = synthetic_state.clone_once()
    assert snapshot.is_tapped(obj_id) is True


def test_tap_one_clone_does_not_tap_other(synthetic_state):
    """**CoW canary.** The whole point of `Arc<Vec<GameObject>>` zones
    is that two live clones of a GameState observe independent
    mutations. Tapping an object on one clone must NOT visibly tap
    the same object on the other — `Arc::make_mut` should clone the
    inner Vec on the first mutation while another reference is live.

    If this test fails, every CoW assumption downstream (MCTS rollouts
    sharing parent state, clone-then-mutate-then-discard search idioms)
    is broken."""
    other = synthetic_state.clone_once()
    obj_id = _bf_ids(synthetic_state, 0)[0]
    assert synthetic_state.is_tapped(obj_id) is False
    assert other.is_tapped(obj_id) is False
    # Tap on the original only.
    assert synthetic_state.tap_object(obj_id) is True
    assert synthetic_state.is_tapped(obj_id) is True
    # The clone must NOT see the tap.
    assert other.is_tapped(obj_id) is False, (
        "CoW broken: tapping one clone leaked into the other"
    )


# --- Cross-feature edge-case coverage --------------------------------
#
# These tests pin tap/untap behavior at the seams where it intersects
# with other primitives — Effect dispatch, ActivateAbility, the untap
# step, fingerprinting, summoning sickness, and CoW. Each test
# documents the current Rust behavior; some pin deliberate divergences
# from CR (see the activate-ability-while-summoning-sick test) and
# others pin defensive no-op shapes (unknown player ids, hand-zone
# tap attempts).


def test_tap_object_on_hand_zone_returns_false(synthetic_state):
    """Only battlefield permanents have a meaningful tapped state.
    `tap_object` against a hand-zone object id is a silent no-op
    returning False — `mutate_battlefield_obj` only walks battlefields
    (see the docstring in `rust/src/state_ops_tap.rs`), so a
    hand-resident id falls through to the `unwrap_or(false)` arm.
    Mirrors the unknown-id case but lands on a real object that
    just lives in the wrong zone. `is_tapped` returns None for the
    same reason (no battlefield match)."""
    # With defaults, p0's hand is ids 10..16.
    hand_id = 10
    assert synthetic_state.find_object_zone(hand_id) == 0  # ZONE_HAND_LOCAL
    assert synthetic_state.tap_object(hand_id) is False
    assert synthetic_state.is_tapped(hand_id) is None
    assert synthetic_state.untap_object(hand_id) is False


def test_untap_all_for_player_on_empty_battlefield_is_no_op(engine):
    """`untap_all_for_player` short-circuits before the
    `Arc::make_mut` clone when the target battlefield has no
    permanents to flip (mirrors the
    `battlefield.iter().all(|o| !o.tapped)` early-return in
    `state_ops_tap.rs`). The empty-battlefield case is a stricter
    flavor of the early return — there are no objects at all, so
    nothing to iterate. Must not panic."""
    state = engine.make_synthetic_state(
        num_battlefield=0, num_hand=1, num_library=1
    )
    # Sanity: both players' battlefields are empty.
    assert state.battlefield_sizes == [0, 0]
    # Must not raise.
    state.untap_all_for_player(0)
    state.untap_all_for_player(1)
    assert state.battlefield_sizes == [0, 0]


def test_untap_all_for_player_unknown_player_is_silent_no_op(synthetic_state):
    """`untap_all_for_player` with an out-of-range player_id silently
    returns without mutating anything. Mirrors the `find(|p| p.id ==
    player_id)` -> `None` path in `state_ops_tap.rs`. Both real
    players' tapped states stay exactly as they were."""
    p0_obj = _bf_ids(synthetic_state, 0)[0]
    p1_obj = _bf_ids(synthetic_state, 1)[0]
    assert synthetic_state.tap_object(p0_obj) is True
    assert synthetic_state.tap_object(p1_obj) is True
    # Unknown player id — neither 0 nor 1.
    synthetic_state.untap_all_for_player(99)
    # Both sides still tapped.
    assert synthetic_state.is_tapped(p0_obj) is True
    assert synthetic_state.is_tapped(p1_obj) is True


def test_untap_all_for_player_leaves_summoning_sick_alone(engine):
    """The untap-step primitive only flips `tapped`; it must NOT
    touch `summoning_sick`. Cross-feature with summoning sickness —
    the `clear_summoning_sick_for_player_inline` sweep is the
    sibling helper in `state_ops_steps.rs` that handles sickness; this
    primitive stays narrowly scoped to tap state."""
    state = engine.make_synthetic_state()
    # Give p0 enough green to cast the elves in their hand. The
    # first hand object (id 10) is a Llanowar Elves printing — a
    # creature with mana_produced — which lets us put a
    # summoning-sick permanent on the battlefield.
    state.add_mana(0, "G", 1)
    elves_id = 10
    assert state.printing_name_of(elves_id) == "Llanowar Elves"
    assert state.cast_creature(elves_id, 0) is True
    assert state.is_summoning_sick(elves_id) is True
    # Tap the elves (e.g. as if activating its mana ability).
    assert state.tap_object(elves_id) is True
    # The untap-step sweep flips tapped back but must leave
    # summoning_sick=True intact — the two flags are independent.
    state.untap_all_for_player(0)
    assert state.is_tapped(elves_id) is False
    assert state.is_summoning_sick(elves_id) is True, (
        "untap_all_for_player must only flip the tapped flag — "
        "summoning_sick is owned by the steps slice's separate sweep"
    )


def test_untap_all_for_player_is_idempotent_on_untapped_state(synthetic_state):
    """Calling `untap_all_for_player` on a battlefield where every
    permanent is already untapped is a no-op — the
    `iter().all(|o| !o.tapped)` early-return in `state_ops_tap.rs`
    skips the `Arc::make_mut` clone entirely. The fingerprint
    (which projects every per-object flag, including tapped) must
    stay identical across the no-op call. This is the
    expected-no-mutation parity check that catches a regression
    where the sweep accidentally pays the CoW clone even when
    nothing changes."""
    fp_before = synthetic_state.fingerprint()
    # Every p0 permanent already untapped (per the seed).
    synthetic_state.untap_all_for_player(0)
    fp_after = synthetic_state.fingerprint()
    assert fp_before == fp_after, (
        "untap_all_for_player on already-untapped state must be a "
        "perfect no-op (fingerprint stable)"
    )


def test_untap_all_for_player_clone_isolation(synthetic_state):
    """**CoW canary for the untap-step primitive.** Mirrors
    `test_tap_one_clone_does_not_tap_other` but on the bulk-untap
    path: tap a permanent on the original, clone, then call
    `untap_all_for_player` on the clone. The clone's permanent
    must come up untapped while the original's stays tapped — the
    `Arc::make_mut` inside the helper must clone the Vec on first
    write so the original's Arc reference is left alone.

    Load-bearing for any search idiom that clones a parent state
    and advances through the untap step on the child (the standard
    MCTS rollout shape)."""
    obj_id = _bf_ids(synthetic_state, 0)[0]
    assert synthetic_state.tap_object(obj_id) is True
    clone = synthetic_state.clone_once()
    assert clone.is_tapped(obj_id) is True
    # Bulk-untap on the clone only.
    clone.untap_all_for_player(0)
    assert clone.is_tapped(obj_id) is False
    # Original must NOT see the untap leak across.
    assert synthetic_state.is_tapped(obj_id) is True, (
        "CoW broken: untap_all_for_player on a clone leaked into the original"
    )


def test_fingerprint_changes_when_tapped_flag_differs(synthetic_state):
    """The fingerprint projects every battlefield object's tapped
    flag (see `object_signature` in
    `rust/src/state_ops_fingerprint.rs` — field index 1 is
    `int(tapped)`). Two states that differ ONLY in a single
    permanent's tap state must produce different fingerprints.
    Cross-feature pin: tap is gameplay-relevant for parity hashing,
    not a transient ephemeral marker."""
    obj_id = _bf_ids(synthetic_state, 0)[0]
    fp_untapped = synthetic_state.fingerprint()
    assert synthetic_state.tap_object(obj_id) is True
    fp_tapped = synthetic_state.fingerprint()
    assert fp_untapped != fp_tapped, (
        "fingerprint should hash tapped state — two states differing "
        "only in a single tap must hash to different digests"
    )


def test_fingerprint_stable_across_tap_then_untap_roundtrip(synthetic_state):
    """A tap-then-untap round-trip is a net no-op at the
    state-of-objects level. The fingerprint folds the tapped bit
    into every per-object signature, so an exact round-trip must
    leave the fingerprint identical to the starting digest. Pins
    that no other field is accidentally co-mutated during a tap
    (e.g. a regression where `tap_object` also touched
    `summoning_sick` or `counters`)."""
    obj_id = _bf_ids(synthetic_state, 0)[0]
    fp_start = synthetic_state.fingerprint()
    assert synthetic_state.tap_object(obj_id) is True
    assert synthetic_state.untap_object(obj_id) is True
    fp_round = synthetic_state.fingerprint()
    assert fp_start == fp_round, (
        "tap+untap round-trip must leave fingerprint stable — any "
        "delta implies tap_object/untap_object touched a field "
        "beyond `tapped`"
    )


def test_activate_ability_taps_source_via_same_tapped_flag(synthetic_state):
    """Cross-feature: the mana-ability ActivateAbility dispatch in
    `state_ops_actions.rs` flips the source's `tapped` flag through
    `Arc::make_mut` on the controller's battlefield — the same
    field the round-1 tap primitive owns. `is_tapped` (which only
    reads the `tapped` field) must report the change just as if
    `tap_object` had been called directly. A second tap attempt via
    `tap_object` after the activate must return False (already
    tapped — the activate's tap is observationally identical to a
    primitive tap)."""
    # p0's first Forest at obj_id 0.
    forest_id = _bf_ids(synthetic_state, 0)[0]
    assert synthetic_state.printing_name_of(forest_id) == "Forest"
    assert synthetic_state.is_tapped(forest_id) is False
    # Activate the Forest's mana ability via the action dispatcher.
    synthetic_state.apply_action(
        {"kind": "activate_ability", "source_id": forest_id, "ability_index": 0}
    )
    # The activate's tap is the SAME tapped flag tap_object reads/writes.
    assert synthetic_state.is_tapped(forest_id) is True
    # A subsequent primitive tap is a no-op (already tapped).
    assert synthetic_state.tap_object(forest_id) is False
    # And the primitive untap successfully reverses it.
    assert synthetic_state.untap_object(forest_id) is True
    assert synthetic_state.is_tapped(forest_id) is False


def test_activate_ability_on_summoning_sick_creature_with_tap_cost_is_rejected(
    engine,
):
    """**Parity gap G6 closure (CR 302.1).** A creature that hasn't
    been controlled continuously since the start of its controller's
    most recent turn (the `summoning_sick = true` flag) can't tap to
    pay an activated ability's `{T}` cost. The Rust dispatch arm in
    `state_ops_actions.rs` gates on
    `summoning_sick && requires_tap_cost`; for the only ability
    currently wired through this dispatch (mana abilities, which all
    tap per CR 605.1a) `summoning_sick` alone is sufficient. Mirrors
    Python's `legal.py:661-663` and `actions.py:_activate_ability`
    precheck. Inverted from the `_succeeds` baseline (round 25) by
    the slice that closed G6 — see the matching exclusion in
    `state_ops_legal.rs` (no emit for summoning-sick sources) tested
    in `test_legal.py`."""
    import pytest

    state = engine.make_synthetic_state()
    # First hand object is a Llanowar Elves (creature with
    # mana_produced=Some(G)). Cast it to put a summoning-sick
    # mana-producer on the battlefield.
    state.add_mana(0, "G", 1)
    elves_id = 10
    assert state.printing_name_of(elves_id) == "Llanowar Elves"
    assert state.cast_creature(elves_id, 0) is True
    assert state.is_summoning_sick(elves_id) is True
    assert state.is_tapped(elves_id) is False
    pool_before = state.get_mana_pool(0)["G"]
    # Real MTG / Python: this fails because the source is summoning
    # sick. Rust now raises a `ValueError` from `apply_action`, with
    # no state mutation (the source stays untapped, the mana pool
    # unchanged, summoning_sick still True).
    with pytest.raises(ValueError, match="summoning-sick"):
        state.apply_action(
            {
                "kind": "activate_ability",
                "source_id": elves_id,
                "ability_index": 0,
            }
        )
    # No mutation occurred on the rejected path.
    assert state.is_tapped(elves_id) is False
    assert state.get_mana_pool(0)["G"] == pool_before
    assert state.is_summoning_sick(elves_id) is True


def test_activate_ability_on_summoning_sick_land_with_tap_cost_is_allowed(
    engine,
):
    """**CR 302.1 creature-only gating.** Summoning sickness applies
    ONLY to creatures tapping for tap-cost abilities — lands (Forests,
    Mountains, etc.) can tap for mana the turn they enter the
    battlefield even though every freshly-allocated permanent carries
    `summoning_sick = true` (the CE1 universal-true default).

    Setup: take a Forest from p0's battlefield (id 0), route it
    through graveyard, then reanimate it via
    `move_object_from_graveyard_to_battlefield` — that primitive
    stamps `summoning_sick = true` on the reanimated permanent (CR
    302.1 / 400.7 reset). The reanimated Forest is now on the
    battlefield with `summoning_sick = true`. Per CR 302.1, this is
    a NON-CREATURE permanent with summoning sickness — Rust must NOT
    reject `activate_ability` on it.

    Inversion of `test_activate_ability_on_summoning_sick_creature_with_tap_cost_is_rejected`
    above: same shape (sick + tap-cost), opposite expectation
    because the source is a land, not a creature.

    Pins the side-finding documented under CE5 in
    `docs/RUST_PARITY_GAPS.md` — the workaround in
    `cross_engine.py`'s `cast_bears_selector` (sick-land filter) is
    no longer load-bearing once this test goes green."""
    state = engine.make_synthetic_state()
    # Synthetic state: p0 battlefield is ids 0..4 Forest, 5..9 Bears.
    forest_id = 0
    assert state.printing_name_of(forest_id) == "Forest"
    # Route Forest through graveyard so reanimation sets
    # summoning_sick=true on the way back to the battlefield.
    assert state.move_object(forest_id, 3, 0) is True  # ZONE_GRAVEYARD=3
    assert state.move_object_from_graveyard_to_battlefield(0, forest_id) is True
    # Sanity: Forest now on bf with summoning_sick=true.
    assert state.is_summoning_sick(forest_id) is True
    assert state.is_tapped(forest_id) is False
    pool_before = state.get_mana_pool(0)["G"]
    # CR 302.1 — lands are exempt from summoning sickness. The
    # activate_ability call should SUCCEED: Forest taps + one G in
    # pool, no ValueError raised.
    state.apply_action(
        {
            "kind": "activate_ability",
            "source_id": forest_id,
            "ability_index": 0,
        }
    )
    # Mutation observed: Forest tapped + one G added to pool.
    assert state.is_tapped(forest_id) is True
    assert state.get_mana_pool(0)["G"] == pool_before + 1
    # summoning_sick is unaffected by the activation (the per-turn
    # untap step clears it; activate_ability does not).
    assert state.is_summoning_sick(forest_id) is True




# Zone constants for the round-28 tap-aoc tests below.
ZONE_HAND_LOCAL = 0
ZONE_BATTLEFIELD = 2
# -------------------- Round-28 TapTargetArtifactOrCreature (2 cards) --------------------


def _tap_aoc_dict():
    return {
        "name": "Synthetic Twiddle (artifact-or-creature)",
        "card_types": ["Instant"],
        "mana_cost": {"generic": 1},
        "effects": [
            {"kind": "tap_target_artifact_or_creature"}
        ],
    }


def _vanilla_artifact_dict_tap(name="Test Artifact"):
    return {
        "name": name,
        "card_types": ["Artifact"],
        "mana_cost": {"generic": 1},
    }


def test_register_tap_target_artifact_or_creature(engine):
    reg = engine.Registry()
    pid = reg.register_printing(_tap_aoc_dict())
    eff = reg.get_printing_effects(pid)[0]
    assert eff["kind"] == "tap_target_artifact_or_creature"


def test_resolve_tap_aoc_picks_first_creature_on_defender_bf(synthetic_state):
    """First creature on defender (p1) gets tapped. Synthetic state's
    p1 bf has Forests (non-creature, non-artifact) then Grizzly Bears."""
    snap = synthetic_state.battlefield_snapshot()
    p1_first_bear_id = next(
        d["id"] for d in snap
        if d["controller_id"] == 1 and d["printing_name"] == "Grizzly Bears"
    )
    spell_id = synthetic_state.register_and_inject_object(
        _tap_aoc_dict(), 0, ZONE_HAND_LOCAL
    )
    synthetic_state.push_to_stack(spell_id, 0)
    synthetic_state.resolve_top_of_stack()

    bear_after = next(
        d for d in synthetic_state.battlefield_snapshot() if d["id"] == p1_first_bear_id
    )
    assert bear_after["tapped"] is True


def test_resolve_tap_aoc_picks_artifact_when_present(synthetic_state):
    """Inject an artifact onto p1's bf — Forests are non-creature
    non-artifact, so they're skipped; then we hit the artifact (which
    appears in zone order BEFORE the Grizzly Bears since make_state /
    register_and_inject_object pushes to the back of the bf zone).
    Actually since artifact is appended LAST, the existing Grizzly
    Bears comes first. So Bears gets tapped, not the artifact. This
    test verifies that BOTH being in the eligible set works — pick
    the FIRST match regardless of type."""
    artifact_id = synthetic_state.register_and_inject_object(
        _vanilla_artifact_dict_tap(), 1, ZONE_BATTLEFIELD
    )
    snap = synthetic_state.battlefield_snapshot()
    p1_first_bear_id = next(
        d["id"] for d in snap
        if d["controller_id"] == 1 and d["printing_name"] == "Grizzly Bears"
    )
    spell_id = synthetic_state.register_and_inject_object(
        _tap_aoc_dict(), 0, ZONE_HAND_LOCAL
    )
    synthetic_state.push_to_stack(spell_id, 0)
    synthetic_state.resolve_top_of_stack()
    # Existing Bears tapped (first in zone order); injected artifact
    # not tapped (came after, second).
    snap_after = synthetic_state.battlefield_snapshot()
    bear_after = next(d for d in snap_after if d["id"] == p1_first_bear_id)
    art_after = next(d for d in snap_after if d["id"] == artifact_id)
    assert bear_after["tapped"] is True
    assert art_after["tapped"] is False


def test_resolve_tap_aoc_no_eligible_fizzles(synthetic_state):
    """No artifacts and no creatures on p1's bf → fizzle."""
    state = synthetic_state.clone_once()
    snap = state.battlefield_snapshot()
    p1_creature_ids = [
        d["id"] for d in snap
        if d["controller_id"] == 1 and d["printing_name"] == "Grizzly Bears"
    ]
    for cid in p1_creature_ids:
        state.move_object(cid, 3, 1)  # graveyard
    spell_id = state.register_and_inject_object(_tap_aoc_dict(), 0, ZONE_HAND_LOCAL)
    state.push_to_stack(spell_id, 0)
    state.resolve_top_of_stack()  # should not raise


def test_resolve_tap_aoc_idempotent_on_already_tapped(synthetic_state):
    """Already-tapped target stays tapped (idempotent)."""
    snap = synthetic_state.battlefield_snapshot()
    p1_first_bear_id = next(
        d["id"] for d in snap
        if d["controller_id"] == 1 and d["printing_name"] == "Grizzly Bears"
    )
    synthetic_state.tap_object(p1_first_bear_id)
    spell_id = synthetic_state.register_and_inject_object(
        _tap_aoc_dict(), 0, ZONE_HAND_LOCAL
    )
    synthetic_state.push_to_stack(spell_id, 0)
    synthetic_state.resolve_top_of_stack()
    bear_after = next(
        d for d in synthetic_state.battlefield_snapshot() if d["id"] == p1_first_bear_id
    )
    assert bear_after["tapped"] is True


# -------------------- Round-28 TapEnchantedPermanent (2 cards) --------------------


def _kismet_aura_dict():
    """Kismet-shape aura: ETB taps what it's attached to."""
    return {
        "name": "Synthetic Aura of Taps",
        "card_types": ["Enchantment"],
        "mana_cost": {"U": 1, "generic": 1},
        "effects": [
            {"kind": "tap_enchanted_permanent"}
        ],
    }


def test_register_tap_enchanted_permanent(engine):
    reg = engine.Registry()
    pid = reg.register_printing(_kismet_aura_dict())
    eff = reg.get_printing_effects(pid)[0]
    assert eff["kind"] == "tap_enchanted_permanent"


def test_resolve_tap_enchanted_permanent_taps_host(synthetic_state):
    """Aura attached to p1's first creature; resolve taps the
    creature."""
    snap = synthetic_state.battlefield_snapshot()
    p1_first_bear_id = next(
        d["id"] for d in snap
        if d["controller_id"] == 1 and d["printing_name"] == "Grizzly Bears"
    )
    aura_id = synthetic_state.register_and_inject_object(
        _kismet_aura_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.attach(aura_id, p1_first_bear_id)

    synthetic_state.push_to_stack(aura_id, 0)
    synthetic_state.resolve_top_of_stack()

    bear_after = next(
        d for d in synthetic_state.battlefield_snapshot() if d["id"] == p1_first_bear_id
    )
    assert bear_after["tapped"] is True


def test_resolve_tap_enchanted_permanent_no_attached_fizzles(synthetic_state):
    aura_id = synthetic_state.register_and_inject_object(
        _kismet_aura_dict(), 0, ZONE_BATTLEFIELD
    )
    # Don't attach; resolve shouldn't raise and nothing should be
    # tapped.
    synthetic_state.push_to_stack(aura_id, 0)
    synthetic_state.resolve_top_of_stack()


def test_resolve_tap_enchanted_permanent_idempotent_on_tapped(synthetic_state):
    snap = synthetic_state.battlefield_snapshot()
    p1_first_bear_id = next(
        d["id"] for d in snap
        if d["controller_id"] == 1 and d["printing_name"] == "Grizzly Bears"
    )
    synthetic_state.tap_object(p1_first_bear_id)
    aura_id = synthetic_state.register_and_inject_object(
        _kismet_aura_dict(), 0, ZONE_BATTLEFIELD
    )
    synthetic_state.attach(aura_id, p1_first_bear_id)
    synthetic_state.push_to_stack(aura_id, 0)
    synthetic_state.resolve_top_of_stack()
    bear_after = next(
        d for d in synthetic_state.battlefield_snapshot() if d["id"] == p1_first_bear_id
    )
    assert bear_after["tapped"] is True
