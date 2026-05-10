"""Tests for ``mtg_bot.cli.__init__`` — the top-level dispatcher.

The dispatcher routes ``mtg_bot <subcommand> ...`` to the right
``mtg_bot.cli.<name>.main(rest)``. These tests pin the routing
table and the help-output behavior; per-subcommand argparse
behavior is tested in the dedicated ``test_<name>.py`` files.
"""

from __future__ import annotations

import importlib
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from mtg_bot import cli as cli_pkg


def test_help_lists_all_subcommands(capsys):
    """``mtg_bot --help`` and bare ``mtg_bot`` must list every
    subcommand. Catches the regression where someone adds a new
    subcommand to the dispatch table but forgets to add it to the
    help-list block."""
    cli_pkg.main([])
    captured = capsys.readouterr().out
    for name in ("iterate", "train", "shard", "serve",
                 "capture", "eval", "bench", "db"):
        assert name in captured


def test_help_explicit_flag_also_lists_subcommands(capsys):
    """``--help`` should yield identical content to no-args (both
    are the explicit help-request paths)."""
    cli_pkg.main(["--help"])
    captured = capsys.readouterr().out
    assert "iterate" in captured
    assert "Subcommands:" in captured


def test_short_help_flag_also_lists_subcommands(capsys):
    cli_pkg.main(["-h"])
    captured = capsys.readouterr().out
    assert "Subcommands:" in captured


def test_dispatch_routes_to_subcommand_main():
    """``mtg_bot iterate <args>`` must call
    ``mtg_bot.cli.iterate.main(<args>)`` with the rest forwarded."""
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv

    fake_module = type("M", (), {"main": fake_main})
    with patch.dict(sys.modules, {"mtg_bot.cli.iterate": fake_module}):
        cli_pkg.main(["iterate", "--bootstrap-version", "v4"])
    assert captured["argv"] == ["--bootstrap-version", "v4"]


def test_dispatch_routes_each_subcommand():
    """Smoke each routing-table entry. Importlib caching means this
    pins the table contents — adding a new key without an entry in
    ``_SUBCOMMANDS`` would surface here."""
    for subname in ("iterate", "train", "shard", "serve",
                    "capture", "eval", "bench", "db"):
        called = {"with_argv": None}

        def make_fake(target=called):
            def fake_main(argv):
                target["with_argv"] = argv
            return fake_main

        fake_module = type("M", (), {"main": make_fake()})
        with patch.dict(
            sys.modules, {f"mtg_bot.cli.{subname}": fake_module},
        ):
            cli_pkg.main([subname, "--xyz", "foo"])
        assert called["with_argv"] == ["--xyz", "foo"], (
            f"{subname} routing didn't forward args"
        )


def test_unknown_subcommand_raises_systemexit():
    """argparse's ``choices=`` enforcement should fire on an unknown
    subcommand — argparse exits via SystemExit(2)."""
    with pytest.raises(SystemExit):
        cli_pkg.main(["nope-not-a-subcommand"])


def test_main_with_no_argv_uses_sys_argv(capsys):
    """``main()`` with no argument should fall back to ``sys.argv[1:]``.
    Pin this so it doesn't silently change to require explicit argv."""
    with patch.object(sys, "argv", ["mtg_bot"]):
        cli_pkg.main()
    captured = capsys.readouterr().out
    assert "Subcommands:" in captured


def test_module_underscore_main_executes_dispatch():
    """``python -m mtg_bot.cli`` runs ``mtg_bot.cli.__main__`` which
    just calls ``main``. Spot-import to confirm the wiring."""
    # Reload to make sure the module is fresh; importing it shouldn't
    # actually call main() (the call is gated on __name__ == "__main__").
    if "mtg_bot.cli.__main__" in sys.modules:
        del sys.modules["mtg_bot.cli.__main__"]
    mod = importlib.import_module("mtg_bot.cli.__main__")
    # The module exposes the imported main symbol so __main__.py is
    # something a test can introspect.
    assert mod.main is cli_pkg.main
