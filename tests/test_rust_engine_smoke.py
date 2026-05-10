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
