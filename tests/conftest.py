"""Session-scoped card-source swap driven by ``MTG_BOT_CARD_SOURCE``,
plus a session-scoped flip of all training-time priority short-circuit
heuristic defaults to ``False`` so legacy tests written before the
heuristics existed pass without modification, and of the searched-
resolution-decisions switch to OFF (tests of the new default opt in).

Default (env var unset) uses the production ``OracleParserCardSource``,
so ``pytest`` runs unchanged. The env-var hook is preserved as a seam
for any future alternative sources, but the parser source is the only
production source today.

Heuristic-default flip: production code (``SelfPlayJobSpec``,
``data_gen``, etc.) explicitly passes a ``HeuristicsConfig`` through
the call chain, so production behavior is unchanged. The adapter-
level default of ``HeuristicsConfig()`` (every flag True) exists for
ergonomic CLI use; flipping it to ``HeuristicsConfig.disabled()`` in
test sessions means tests that don't care about heuristics (the vast
majority) get the expected non-short-circuited behavior, and tests
that DO care about a specific heuristic (the per-heuristic
``test_<heuristic>.py`` files) can opt in by passing
``heuristics=HeuristicsConfig(...flag=True...)`` explicitly.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from mtg_bot.cards.source import (
    card_source_from_env,
    get_active_source,
    set_active_source,
)


@pytest.fixture(scope="session", autouse=True)
def _swap_card_source_from_env():
    requested = card_source_from_env(os.environ.get("MTG_BOT_CARD_SOURCE"))
    original = get_active_source()
    set_active_source(requested)
    try:
        yield
    finally:
        set_active_source(original)


@pytest.fixture(autouse=True)
def _isolate_runs_db_env(tmp_path, monkeypatch):
    """Per-test isolation of the runs-DB env vars.

    ``_ensure_master_runs_db_path_default()`` (in
    ``mtg_bot.training.distributed``) sets ``MTG_BOT_RUN_DB_PATH``
    to ``training_progress/runs.db`` (relative to CWD) at the top of
    ``run_distributed_*``. Without isolation that bleeds into other
    tests in the same process and points at the developer's actual
    runs.db file.

    We default to a per-test path so any code that auto-defaults the
    env var lands rows in an isolated DB. Individual tests that want
    to assert against the no-DB code path call
    ``monkeypatch.delenv("MTG_BOT_RUN_DB_PATH", raising=False)``;
    tests that pre-seed a master run row override the path explicitly.

    The default DB is created lazily (no file is written if no run is
    registered), so most tests pay zero cost.
    """
    monkeypatch.setenv(
        "MTG_BOT_RUN_DB_PATH", str(tmp_path / "runs.db"),
    )
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)


@pytest.fixture(autouse=True)
def _isolate_token_printing_registry():
    """Per-test isolation of the cross-engine token-printing registry.

    ``cross_engine._TOKEN_PRINTING_BY_PID`` (+ the round-51
    ``_TOKEN_PRINTING_CACHE_BY_KEY`` shape cache) is module-global —
    it persists across tests in the same process. ``trajectory.
    _seat_token_printings`` registers a token printing at a
    per-deck-deterministic pid (its index in the per-game ``card_defs``
    union). Before round-53 N1's Phase 3a, the production-deck loader
    seated a token printing only for the round-49/50 vanilla / tapped
    carriers (none of which any *real* archetype deck admitted, so the
    pre-existing real-deck loader tests never seated anything). Phase
    3a admits the real EoE Lander carriers (Galactic Wayfarer 33,469
    decks, Beamsaw Prospector 24,310 — these ARE in real archetype
    decks), so those pre-existing tests now seat a ``Lander`` printing
    into the module-global registry. A LATER test that seats a
    DIFFERENT-shape token (e.g. a ``Robot``) at the SAME pid then hits
    the identity-based conflicting-registration guard
    (``register_token_printing`` ``ValueError``).

    This autouse fixture clears the registry + the round-51 shape
    cache around EVERY test, generalizing the round-50-finding-1 /
    round-51 ``cached_token_printing`` cross-game-safety work from
    "same logical token across games in ONE test" to "no token
    registration leaks ACROSS tests". Tests that already wrap their
    own ``try/finally: clear_token_printings()`` stay correct (the
    clear is idempotent); pre-existing real-deck loader tests that
    never knew they needed it become leak-safe with no per-test edit.
    """
    from mtg_bot.parity.cross_engine import clear_token_printings

    clear_token_printings()
    try:
        yield
    finally:
        clear_token_printings()


@pytest.fixture(autouse=True)
def _shutdown_leaked_serve_apps():
    """Stop background threads started by ``serve.create_app`` in tests.

    ``mtg_bot.cli.serve.create_app`` can start up to three long-lived
    daemon resources (a ``JanitorThread`` poll loop, a
    ``HeartbeatPoller`` poll loop, and a report ``ThreadPoolExecutor``)
    and hangs them on ``app.extensions``. In production they live for
    the process lifetime; in tests, every ``create_app`` call in a
    worker process leaks more daemon threads. Under ``pytest -n`` the
    accumulated threads keep the worker from terminating cleanly, so
    xdist kills it ("node down: Not properly terminated") and dumps a
    faulthandler traceback of every leaked thread.

    This autouse fixture wraps ``create_app`` for the duration of each
    test, records every app it returns, and calls
    ``serve.shutdown_app`` on all of them in teardown — a generic
    safety net so individual tests don't each need bespoke cleanup.
    """
    from mtg_bot.cli import serve as _serve

    created: list = []
    real_create_app = _serve.create_app

    def _tracking_create_app(*args, **kwargs):
        app = real_create_app(*args, **kwargs)
        created.append(app)
        return app

    with mock.patch.object(_serve, "create_app", _tracking_create_app):
        try:
            yield
        finally:
            for app in created:
                _serve.shutdown_app(app)


@pytest.fixture(scope="session", autouse=True)
def _resolution_search_off_by_default_for_tests():
    """Production searches resolution decisions by default
    (``HeuristicsConfig.search_resolution_decisions``, default from
    ``$MTG_BOT_SEARCH_RESOLUTION_DECISIONS``; see
    ``docs/RESOLUTION_DECISIONS.md``). Same precedent as the heuristic
    flip below: legacy tests -- seeded outcome pins, positional_v1
    data-gen fixtures whose policy head has no resolution slots --
    run with the switch OFF for the session, and the tests that cover
    the new default turn it on explicitly
    (``monkeypatch.setenv(..., "1")`` or
    ``HeuristicsConfig(search_resolution_decisions=True)``). Set in
    ``os.environ`` so spawned pool workers inherit it."""
    from mtg_bot.search.heuristics_config import (
        SEARCH_RESOLUTION_DECISIONS_ENV,
    )

    previous = os.environ.get(SEARCH_RESOLUTION_DECISIONS_ENV)
    os.environ[SEARCH_RESOLUTION_DECISIONS_ENV] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SEARCH_RESOLUTION_DECISIONS_ENV, None)
        else:
            os.environ[SEARCH_RESOLUTION_DECISIONS_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _flip_heuristic_defaults_off_for_tests():
    """Wrap ``MtgGameStateAdapter.__init__`` so the ``heuristics``
    kwarg defaults to ``HeuristicsConfig.disabled()`` (every flag
    False) unless the test explicitly passes one. Production callers
    (``data_gen``, ``self_play``, ``distributed``) pass an explicit
    ``HeuristicsConfig`` and are unaffected.

    Tests that exercise a specific heuristic pass
    ``heuristics=HeuristicsConfig(<flag>=True)`` (or the all-True
    default) and continue to exercise that path.
    """
    from mtg_bot.search import engine_adapter as _ea
    from mtg_bot.search.heuristics_config import HeuristicsConfig

    real_init = _ea.MtgGameStateAdapter.__init__

    def _patched_init(self, *args, **kwargs):
        if kwargs.get("heuristics") is None:
            kwargs["heuristics"] = HeuristicsConfig.disabled()
        return real_init(self, *args, **kwargs)

    with mock.patch.object(
        _ea.MtgGameStateAdapter, "__init__", _patched_init,
    ):
        yield
