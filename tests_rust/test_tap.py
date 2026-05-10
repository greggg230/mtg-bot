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
