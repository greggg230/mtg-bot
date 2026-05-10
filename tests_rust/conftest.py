"""Shared fixtures for ``tests_rust/`` — the Rust-engine TDD test suite.

These tests target the ``mtg_bot_engine`` PyO3 extension, NOT the
pure-Python ``mtg_bot`` package. The extension is built out-of-tree
by maturin (see ``rust/`` and ``rust/README.md``); if it isn't
installed, the entire ``tests_rust/`` directory is skipped via
``pytest_collection_modifyitems`` so a fresh checkout doesn't
break the gate.

If you're writing a new Rust-engine test, prefer the ``engine``
fixture over a bare ``import mtg_bot_engine`` — it gives the test
a clean handle and a fast-fail message when the extension isn't
present.
"""

from __future__ import annotations

import importlib

import pytest


def _try_import_engine():
    try:
        return importlib.import_module("mtg_bot_engine")
    except ImportError:
        return None


_ENGINE = _try_import_engine()


def pytest_collection_modifyitems(config, items):
    """Skip every ``tests_rust/`` test if the extension isn't built."""
    if _ENGINE is not None:
        return
    skip_marker = pytest.mark.skip(
        reason=(
            "mtg_bot_engine PyO3 extension not installed; "
            "run `maturin develop --release -m rust/Cargo.toml`"
        )
    )
    for item in items:
        if "tests_rust" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip_marker)


@pytest.fixture
def engine():
    """The ``mtg_bot_engine`` module, or skip if not installed."""
    if _ENGINE is None:
        pytest.skip("mtg_bot_engine not installed")
    return _ENGINE


@pytest.fixture
def synthetic_state(engine):
    """A default synthetic state — 10 battlefield, 7 hand, 50 library per side."""
    return engine.make_synthetic_state()


@pytest.fixture
def empty_registry(engine):
    """A fresh, empty Registry — for bridge / card-data tests."""
    return engine.Registry()
