"""Tests for the ``action_encoding_version`` follow-up wiring inside
``mtg_bot.cli.iterate``.

Yesterday's commit shipped opt-in ``canonical_v2`` action encoding
behind a metadata-version flag (``ModelMetadata.action_encoding_version``,
``--action-encoding-version`` flag on ``mtg_bot train``, NNPimcConfig
field). Today's gap: ``mtg_bot iterate`` doesn't auto-thread the
version from bootstrap metadata into the per-iteration NN configs and
the train subprocess. A v17 trained at v2 via iterate would silently
regress to v1 self-play data because the iterate driver wouldn't know
to honor the bootstrap's metadata.

These tests pin:

A. ``Registry.load_metadata`` round-trips ``action_encoding_version``
   from metadata.json when the field is present, AND defaults to
   ``"positional_v1"`` when the field is absent (back-compat for
   v13/v15/v16 metadata files predating the field).
B. ``_train_next`` includes ``--action-encoding-version <version>`` in
   the spawned ``mtg_bot train`` subprocess command line, so the new
   model's metadata records the inherited convention.
C. ``_self_play`` constructs ``NNPimcConfig(...,
   action_encoding_version=...)`` AND passes the version through to
   ``generate_training_data_parallel`` (which threads it down to the
   per-game ``_play_one_game_recording`` policy-target encoder).
D. ``_eval_match`` constructs both NNPimcConfig instances (new + old
   seats) with the matching ``action_encoding_version``.

The tests use mocks/spies — no actual training, no actual self-play
games. Subprocess invocations are intercepted at ``subprocess.run``;
NNPimcConfig and the per-game data-gen / match drivers are stubbed
so we just assert the wiring, not the heavy downstream work.
"""

from __future__ import annotations

import json
import subprocess as _subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mtg_bot.cli import iterate as iz
from mtg_bot.training import data_gen as data_gen_mod
from mtg_bot.training.registry import ModelMetadata, Registry


# ----------------------------------------------------------------------
# Helpers — synthesize a fake registered version with a metadata.json
# carrying the action_encoding_version we want to test.
# ----------------------------------------------------------------------


def _make_version_dir(
    registry_root: Path,
    version: str,
    *,
    action_encoding_version: str | None = "canonical_v2",
    hidden_sizes: list[int] | None = None,
) -> Path:
    """Lay down ``<root>/<version>/{weights.pt,metadata.json}`` with a
    metadata.json that carries (or omits) the action_encoding_version
    field. Mirrors ``test_iterate_resume._make_version_dir`` plus the
    new field.

    ``action_encoding_version=None`` writes a metadata.json that OMITS
    the key entirely, exercising the back-compat default-to-v1 path
    (v13/v15/v16 lookalikes).
    """
    if hidden_sizes is None:
        hidden_sizes = [4]
    d = registry_root / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "weights.pt").write_bytes(b"\x00")
    payload: dict[str, Any] = {
        "version": version,
        "created_at": "2026-05-06T00:00:00Z",
        "feature_dim": 1,
        "hidden_sizes": hidden_sizes,
        "with_policy": True,
        "architecture": "shared",
    }
    if action_encoding_version is not None:
        payload["action_encoding_version"] = action_encoding_version
    (d / "metadata.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    return d


# ----------------------------------------------------------------------
# A. Registry round-trip + back-compat default
# ----------------------------------------------------------------------


def test_registry_load_metadata_returns_v2_when_present(tmp_path: Path):
    """Bootstrap metadata.json carrying
    ``action_encoding_version: "canonical_v2"`` round-trips through
    Registry.load_metadata so iterate can read it off the bootstrap
    version and thread it through."""
    registry_root = tmp_path / "trained_weights"
    _make_version_dir(
        registry_root, "v17", action_encoding_version="canonical_v2",
    )
    meta = Registry(registry_root).load_metadata("v17")
    assert isinstance(meta, ModelMetadata)
    assert meta.action_encoding_version == "canonical_v2"


def test_registry_load_metadata_defaults_when_field_missing(
    tmp_path: Path,
):
    """v13/v15/v16-style metadata.json (no action_encoding_version key)
    defaults to ``"positional_v1"``. This is the back-compat invariant
    that lets old bootstraps keep their bit-exact behavior — a missing
    field MUST resolve to v1, not raise or yield None."""
    registry_root = tmp_path / "trained_weights"
    _make_version_dir(
        registry_root, "v13", action_encoding_version=None,
    )
    meta = Registry(registry_root).load_metadata("v13")
    assert meta.action_encoding_version == "positional_v1"


# ----------------------------------------------------------------------
# B. _train_next subprocess command line carries the flag
# ----------------------------------------------------------------------


class _CmdRecorder:
    """Captures the cmd argv that ``subprocess.run`` was called with so
    the test can assert the iterate driver passed
    ``--action-encoding-version <X>`` through to ``mtg_bot train``."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd, *args, **kwargs):  # noqa: D401
        # ``check=False`` from caller; emulate a successful exit so
        # ``_train_next`` proceeds to the registry lookup.
        self.calls.append(list(cmd))

        class _Result:
            returncode = 0

        return _Result()


def test_train_next_passes_action_encoding_flag_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Calling ``_train_next(action_encoding_version="canonical_v2")``
    spawns ``mtg_bot train`` (via the train_nn.py shim) with
    ``--action-encoding-version canonical_v2`` on the command line. The
    spawned train CLI then writes the version into the new model's
    metadata.json — the next iter's bootstrap-read inherits cleanly."""
    registry_root = tmp_path / "trained_weights"
    # _train_next does a registry lookup at the end; pre-stage a v1 so
    # that lookup returns successfully without us having to actually
    # run the training.
    _make_version_dir(
        registry_root, "v1", action_encoding_version="canonical_v2",
    )

    rec = _CmdRecorder()
    monkeypatch.setattr(_subprocess, "run", rec)

    out = iz._train_next(
        data_dir=tmp_path / "iterate_data",
        registry_root=registry_root,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        val_split=0.1,
        seed=0,
        notes="iterate v2-wiring test",
        window_recent=None,
        action_encoding_version="canonical_v2",
    )
    assert out == "v1"  # registry lookup returned the staged version

    assert len(rec.calls) == 1
    cmd = rec.calls[0]
    # --action-encoding-version must appear immediately followed by the
    # opt-in value. argparse on the receiving side treats them as a
    # flag-value pair.
    assert "--action-encoding-version" in cmd, cmd
    idx = cmd.index("--action-encoding-version")
    assert cmd[idx + 1] == "canonical_v2"


def test_train_next_passes_action_encoding_flag_v1_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Default value (no kwarg from caller) passes
    ``--action-encoding-version positional_v1`` so iterate runs against
    a v13/v15/v16 bootstrap (with no metadata field) still produce the
    flag explicitly. Avoids relying on the train CLI's own default
    drifting silently."""
    registry_root = tmp_path / "trained_weights"
    _make_version_dir(
        registry_root, "v1", action_encoding_version=None,
    )

    rec = _CmdRecorder()
    monkeypatch.setattr(_subprocess, "run", rec)

    iz._train_next(
        data_dir=tmp_path / "iterate_data",
        registry_root=registry_root,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        val_split=0.1,
        seed=0,
        notes="iterate v1-default test",
        window_recent=None,
        # action_encoding_version omitted → falls through to the
        # default "positional_v1"
    )
    cmd = rec.calls[0]
    idx = cmd.index("--action-encoding-version")
    assert cmd[idx + 1] == "positional_v1"


# ----------------------------------------------------------------------
# C. _self_play threads version into NNPimcConfig + data-gen
# ----------------------------------------------------------------------


class _NNPimcSpy:
    """Replaces ``mtg_bot.cli.iterate.NNPimcConfig`` so we can assert
    the kwargs the iterate driver passed at construction time. Stores
    each instance + the original kwargs on the class for the test to
    inspect, and exposes ``.build`` so callers don't blow up at
    factory-construction."""

    instances: list["_NNPimcSpy"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        type(self).instances.append(self)

    def build(self, seed: int = 0):  # noqa: D401, ANN001
        # The dummy strategy returned never gets used because we also
        # stub out ``generate_training_data_parallel`` /
        # ``run_match_parallel``. Returning ``self`` keeps the factory
        # contract (callable returning an object).
        return self


def _empty_data_gen_result(*args: Any, **kwargs: Any) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Stand-in for ``generate_training_data_parallel`` — returns four
    empty arrays of the right dtype so ``_self_play``'s caller code
    (save_dataset, n_samples / n_policy counters) doesn't crash. The
    test only inspects the captured call kwargs, not the data."""
    f = np.zeros((0, 1), dtype=np.float32)
    o = np.zeros((0,), dtype=np.float32)
    p = np.zeros((0, 1), dtype=np.float32)
    m = np.zeros((0, 1), dtype=np.bool_)
    return f, o, p, m


def test_self_play_threads_v2_into_nnpimc_and_datagen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """``_self_play(..., action_encoding_version="canonical_v2")``
    constructs NNPimcConfig with the matching kwarg AND threads the
    same value into ``generate_training_data_parallel`` so the per-game
    encoder uses the v2 slot mapping for policy targets."""
    _NNPimcSpy.instances.clear()
    monkeypatch.setattr(iz, "NNPimcConfig", _NNPimcSpy)

    captured_kwargs: dict[str, Any] = {}

    def _spy_data_gen(*args: Any, **kwargs: Any):
        captured_kwargs.update(kwargs)
        return _empty_data_gen_result()

    monkeypatch.setattr(
        iz, "generate_training_data_parallel", _spy_data_gen,
    )
    # save_dataset would try to write to disk; stub it to a no-op.
    monkeypatch.setattr(iz, "save_dataset", lambda *a, **kw: None)

    iz._self_play(
        weights_path=tmp_path / "weights.pt",
        n_games=1,
        samples=1,
        nodes=1,
        seed=0,
        n_workers=1,
        output=tmp_path / "out.npz",
        deck_pool=[],
        action_encoding_version="canonical_v2",
    )

    assert len(_NNPimcSpy.instances) == 1, _NNPimcSpy.instances
    assert (
        _NNPimcSpy.instances[0].kwargs.get("action_encoding_version")
        == "canonical_v2"
    )
    assert (
        captured_kwargs.get("action_encoding_version") == "canonical_v2"
    )


def test_self_play_defaults_to_v1_when_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A v13/v15/v16-style invocation that omits the kwarg uses the
    "positional_v1" default end-to-end (mirrors the missing-field
    metadata default from the registry)."""
    _NNPimcSpy.instances.clear()
    monkeypatch.setattr(iz, "NNPimcConfig", _NNPimcSpy)

    captured_kwargs: dict[str, Any] = {}

    def _spy_data_gen(*args: Any, **kwargs: Any):
        captured_kwargs.update(kwargs)
        return _empty_data_gen_result()

    monkeypatch.setattr(
        iz, "generate_training_data_parallel", _spy_data_gen,
    )
    monkeypatch.setattr(iz, "save_dataset", lambda *a, **kw: None)

    iz._self_play(
        weights_path=tmp_path / "weights.pt",
        n_games=1,
        samples=1,
        nodes=1,
        seed=0,
        n_workers=1,
        output=tmp_path / "out.npz",
        deck_pool=[],
    )

    assert (
        _NNPimcSpy.instances[0].kwargs.get("action_encoding_version")
        == "positional_v1"
    )
    assert (
        captured_kwargs.get("action_encoding_version") == "positional_v1"
    )


# ----------------------------------------------------------------------
# D. _eval_match builds both NNPimcConfigs with the same version
# ----------------------------------------------------------------------


def test_eval_match_threads_v2_into_both_seats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """``_eval_match(..., action_encoding_version="canonical_v2")``
    constructs BOTH ``NNPimcConfig`` instances (new seat + old seat)
    with the matching version. Both seats share a slot mapping so PUCT
    priors line up across the head-to-head."""
    _NNPimcSpy.instances.clear()
    monkeypatch.setattr(iz, "NNPimcConfig", _NNPimcSpy)

    class _DummyStats:
        wins = (0, 0)
        draws = 0

    monkeypatch.setattr(
        iz, "run_match_parallel", lambda *a, **kw: _DummyStats(),
    )

    iz._eval_match(
        new_weights=tmp_path / "new.pt",
        old_weights=tmp_path / "old.pt",
        n_games=1,
        samples=1,
        nodes=1,
        n_workers=1,
        seed=0,
        deck_pool=[],
        action_encoding_version="canonical_v2",
    )

    # Two NNPimcConfigs: new seat first, old seat second.
    assert len(_NNPimcSpy.instances) == 2
    for inst in _NNPimcSpy.instances:
        assert (
            inst.kwargs.get("action_encoding_version") == "canonical_v2"
        )


def test_eval_match_heuristic_old_still_passes_v2_to_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When ``old_weights=None`` the old seat is a heuristic baseline
    (PimcConfig, not NNPimcConfig) — only the NEW seat's NNPimcConfig
    needs the version. Pin that the absent old NN doesn't break the
    threading and that the new seat still receives v2."""
    _NNPimcSpy.instances.clear()
    monkeypatch.setattr(iz, "NNPimcConfig", _NNPimcSpy)

    class _DummyStats:
        wins = (0, 0)
        draws = 0

    monkeypatch.setattr(
        iz, "run_match_parallel", lambda *a, **kw: _DummyStats(),
    )

    iz._eval_match(
        new_weights=tmp_path / "new.pt",
        old_weights=None,
        n_games=1,
        samples=1,
        nodes=1,
        n_workers=1,
        seed=0,
        deck_pool=[],
        action_encoding_version="canonical_v2",
    )

    # Only the new seat's NNPimcConfig was built; the old seat used
    # PimcConfig (heuristic baseline) which has no NN encoding to thread.
    assert len(_NNPimcSpy.instances) == 1
    assert (
        _NNPimcSpy.instances[0].kwargs.get("action_encoding_version")
        == "canonical_v2"
    )


# ----------------------------------------------------------------------
# E. data_gen.generate_training_data_parallel actually accepts and
#    threads the kwarg through (white-box check that the data_gen-side
#    edit completed — without it, _self_play's threading would TypeError
#    at runtime).
# ----------------------------------------------------------------------


def test_generate_training_data_parallel_accepts_action_encoding_version():
    """Unit-level smoke: the data-gen entry point accepts the kwarg
    introduced for this feature. Catches a regression where someone
    edits the iterate side but forgets the data_gen side (or vice
    versa) — without the data_gen kwarg, _self_play's call site would
    raise TypeError on the first iteration."""
    import inspect

    sig = inspect.signature(
        data_gen_mod.generate_training_data_parallel,
    )
    assert "action_encoding_version" in sig.parameters
    assert (
        sig.parameters["action_encoding_version"].default
        == "positional_v1"
    )


def test_play_one_game_recording_threads_through_action_encoding():
    """The downstream encoder function ``_play_one_game_recording``
    must accept the kwarg (yesterday's commit added it). Pin the
    signature so a downstream rename / removal would break this test
    rather than silently regressing the v2 wiring."""
    import inspect

    sig = inspect.signature(data_gen_mod._play_one_game_recording)
    assert "action_encoding_version" in sig.parameters
    assert (
        sig.parameters["action_encoding_version"].default
        == "positional_v1"
    )
