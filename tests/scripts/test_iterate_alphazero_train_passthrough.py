"""Tests for ``mtg_bot.cli.iterate`` -> ``scripts/train_nn.py`` shim
flag-passthrough plumbing (the iterate loop is now in
``mtg_bot.cli.iterate``; the script is a thin shim).

The iterate loop spawns ``scripts/train_nn.py`` via ``subprocess.run``;
the per-iteration train hyperparameters (``--epochs``,
``--policy-loss-weight``, ``--one-legal-keep-frac``, ...) need
to land in the subprocess argv exactly as the operator set them.

These tests stub ``subprocess.run`` so we capture the argv without
actually launching a training run, then assert the expected flags
are present.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    src_dir = _REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if "mtg_bot.cli.iterate" in sys.modules:
        del sys.modules["mtg_bot.cli.iterate"]
    return importlib.import_module("mtg_bot.cli.iterate")


def _captured_subprocess_call(**kwargs):
    """Invoke ``_train_next`` with the given kwargs and return the
    argv list passed to subprocess.run.

    ``subprocess`` is imported *inside* ``_train_next`` (lazy import
    to keep iterate_alphazero's import cost down on read-only paths),
    so we patch the global ``subprocess`` module directly — the lazy
    re-import inside ``_train_next`` resolves to the same already-
    imported module object that we've patched.
    """
    import subprocess as real_subprocess

    _load_module()  # ensure module imports cleanly
    captured: dict = {}

    def fake_run(cmd, check=False, **_kwargs):
        captured["cmd"] = list(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    with patch.object(real_subprocess, "run", side_effect=fake_run):
        # Reload the iterate_alphazero module so its lazy `import
        # subprocess` inside _train_next picks up our patched
        # subprocess.run.
        m = _load_module()
        try:
            m._train_next(**kwargs)
        except Exception:
            # Downstream registry lookups will fail since no version
            # was actually trained; we only care about argv capture.
            pass
    return captured["cmd"]


def test_train_next_threads_one_legal_keep_frac(tmp_path: Path):
    """``one_legal_keep_frac=0.1`` must appear as
    ``--one-legal-keep-frac 0.1`` in the train_nn argv."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_root = tmp_path / "weights"
    registry_root.mkdir()

    cmd = _captured_subprocess_call(
        data_dir=data_dir,
        registry_root=registry_root,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        val_split=0.1,
        seed=0,
        notes="test",
        window_recent=None,
        one_legal_keep_frac=0.1,
    )
    assert "--one-legal-keep-frac" in cmd
    idx = cmd.index("--one-legal-keep-frac")
    assert cmd[idx + 1] == "0.1"


def test_train_next_default_one_legal_keep_frac_is_one(tmp_path: Path):
    """Default ``one_legal_keep_frac=1.0`` is passed through; this
    pins the no-op default so existing iterate runs are unaffected."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_root = tmp_path / "weights"
    registry_root.mkdir()

    cmd = _captured_subprocess_call(
        data_dir=data_dir,
        registry_root=registry_root,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        val_split=0.1,
        seed=0,
        notes="test",
        window_recent=None,
    )
    assert "--one-legal-keep-frac" in cmd
    idx = cmd.index("--one-legal-keep-frac")
    assert cmd[idx + 1] == "1.0"


def test_train_next_threads_policy_loss_weight_with_keep_frac(
    tmp_path: Path,
):
    """Both flags can be set together — the data-side downsample and
    the loss-weight knob are orthogonal and must coexist on the argv."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_root = tmp_path / "weights"
    registry_root.mkdir()

    cmd = _captured_subprocess_call(
        data_dir=data_dir,
        registry_root=registry_root,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        val_split=0.1,
        seed=0,
        notes="test",
        window_recent=None,
        policy_loss_weight=5.0,
        one_legal_keep_frac=0.1,
    )
    assert "--policy-loss-weight" in cmd
    p_idx = cmd.index("--policy-loss-weight")
    assert cmd[p_idx + 1] == "5.0"
    assert "--one-legal-keep-frac" in cmd
    o_idx = cmd.index("--one-legal-keep-frac")
    assert cmd[o_idx + 1] == "0.1"


def test_train_next_threads_window_recent(tmp_path: Path):
    """``window_recent=50`` must appear as ``--window-recent 50`` on
    the train_nn argv. Pins the subprocess wiring that replaces
    the previous 'not yet wired' warning."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_root = tmp_path / "weights"
    registry_root.mkdir()

    cmd = _captured_subprocess_call(
        data_dir=data_dir,
        registry_root=registry_root,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        val_split=0.1,
        seed=0,
        notes="test",
        window_recent=50,
    )
    assert "--window-recent" in cmd
    idx = cmd.index("--window-recent")
    assert cmd[idx + 1] == "50"


def test_train_next_window_recent_none_omits_flag(tmp_path: Path):
    """``window_recent=None`` (the default) must NOT add
    ``--window-recent`` to the argv — train_nn's own default
    handles this, and an explicit ``--window-recent None`` would
    be argparse-invalid."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_root = tmp_path / "weights"
    registry_root.mkdir()

    cmd = _captured_subprocess_call(
        data_dir=data_dir,
        registry_root=registry_root,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        val_split=0.1,
        seed=0,
        notes="test",
        window_recent=None,
    )
    assert "--window-recent" not in cmd
