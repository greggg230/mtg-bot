"""``mtg_bot eval --hang-dump-dir`` plumbing tests.

Yesterday's v16-vs-v13 eval timed out 552 games and dropped the
forensic data because ``mtg_bot eval`` wasn't passing
``--hang-dump-dir`` through to the worker pool — only ``mtg_bot
iterate`` and ``mtg_bot capture`` were. These tests pin the wiring
so the regression can't recur:

1. ``EvalJobSpec`` round-trips ``hang_dump_root`` cleanly through
   ``to_dict``/``from_dict`` (already covered in
   ``tests/training/test_distributed.py``; we reassert here for
   localized coverage of the CLI flag's destination field).
2. ``mtg_bot eval --hang-dump-dir foo/bar`` populates
   ``EvalJobSpec.hang_dump_root`` (the on-spec field name; CLI flag
   uses ``--hang-dump-dir`` for parity with iterate/capture).
3. ``run_shard_eval_pull`` resolves ``spec.hang_dump_root`` into the
   per-shard ``hang_dump_dir`` and threads it into the worker pool's
   submitted args, so each timed-out game's snapshot lands on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mtg_bot.cli import eval as cli_eval
from mtg_bot.cli.runs import lifecycle
from mtg_bot.search.heuristics_config import HeuristicsConfig
from mtg_bot.training.distributed import EvalJobSpec


def test_eval_job_spec_round_trips_hang_dump_root():
    """The CLI flag's destination field on the spec — pin both
    set-and-cleared paths so a typo on either side surfaces here
    rather than at deploy time."""
    spec = EvalJobSpec(
        new_weights_path="new.pt",
        old_weights_path="old.pt",
        samples=2, inner_nodes=10, hidden_sizes=None,
        base_seed=1, enable_mulligans=False,
        per_game_timeout_sec=60.0, deck_pool_path=None,
        hang_dump_root="debug_hangs/eval_v16_v13",
    )
    d = spec.to_dict()
    assert d["hang_dump_root"] == "debug_hangs/eval_v16_v13"
    rt = EvalJobSpec.from_dict(d)
    assert rt.hang_dump_root == "debug_hangs/eval_v16_v13"

    spec_unset = EvalJobSpec(
        new_weights_path="new.pt", old_weights_path=None,
        samples=1, inner_nodes=1, hidden_sizes=None,
        base_seed=0, enable_mulligans=False,
        per_game_timeout_sec=10.0, deck_pool_path=None,
    )
    assert spec_unset.hang_dump_root is None
    assert EvalJobSpec.from_dict(spec_unset.to_dict()).hang_dump_root is None


def _run_eval_main_with_capture(
    monkeypatch, tmp_path, extra_argv,
):
    """Stub run_distributed_eval, run cli_eval.main, return the spec
    that was passed in.

    Shared between the populate-spec and disable-flag tests below."""
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("MTG_BOT_RUN_DB_PATH", str(db_path))
    monkeypatch.delenv("MTG_BOT_RUN_DB_URL", raising=False)
    monkeypatch.setattr(lifecycle, "git_commit_short", lambda: "abc1234")

    progress_dir = tmp_path / "training_progress"
    deck_pool = tmp_path / "decks.json.gz"
    deck_pool.write_bytes(b"")
    shards_config = tmp_path / "shards.json"
    shards_config.write_text(json.dumps({"shards": []}), encoding="utf-8")

    captured: dict = {}

    def fake_run_distributed_eval(*args, **kwargs):
        # ``run_distributed_eval(shards_cfg, spec, ...)`` — capture
        # the spec positional so the test can assert hang_dump_root.
        captured["spec"] = args[1]
        captured["args"] = args
        captured["kwargs"] = kwargs
        progress_json = (
            kwargs["progress_dir"]
            / f"{kwargs['run_id_prefix']}.json"
        )
        progress_json.parent.mkdir(parents=True, exist_ok=True)
        progress_json.write_text("{}", encoding="utf-8")
        return {"wins_new": 1, "wins_old": 0, "draws": 0, "crashed": 0}

    base_argv = [
        "--new", "v16", "--old", "v13",
        "--games", "10",
        "--shards-config", str(shards_config),
        "--progress-dir", str(progress_dir),
        "--deck-pool", str(deck_pool),
        "--queue-url", "http://master:5099",
        "--bundle-url", "http://master:5099/repo.bundle",
    ]

    with patch.object(
        cli_eval, "run_distributed_eval", fake_run_distributed_eval,
    ), patch.object(
        cli_eval, "load_shards_config", lambda p: {"shards": []},
    ):
        cli_eval.main(base_argv + extra_argv)

    return captured, db_path


def test_eval_main_populates_hang_dump_root_from_flag(
    monkeypatch, tmp_path,
):
    """``mtg_bot eval --hang-dump-dir foo/bar`` -> spec carries
    ``hang_dump_root='foo/bar'``."""
    captured, _db = _run_eval_main_with_capture(
        monkeypatch, tmp_path,
        ["--hang-dump-dir", "foo/bar"],
    )
    spec: EvalJobSpec = captured["spec"]
    # Path("foo/bar") → str either "foo/bar" (POSIX) or "foo\\bar"
    # (Windows). Normalize on the test side rather than re-encoding
    # in the CLI.
    assert Path(spec.hang_dump_root) == Path("foo/bar")


def test_eval_main_default_hang_dump_root_is_debug_hangs(
    monkeypatch, tmp_path,
):
    """No flag -> default ``debug_hangs`` (gitignored), matching the
    iterate-CLI default. Non-empty default means out-of-the-box
    invocations now capture forensic data on timeout, instead of
    silently dropping it the way yesterday's v16-vs-v13 did."""
    captured, _db = _run_eval_main_with_capture(
        monkeypatch, tmp_path, [],
    )
    spec: EvalJobSpec = captured["spec"]
    assert Path(spec.hang_dump_root) == Path("debug_hangs")


def test_eval_main_empty_hang_dump_root_normalizes_to_dot(
    monkeypatch, tmp_path,
):
    """Edge case: ``--hang-dump-dir ''`` is parsed as ``Path('')``
    which Python normalizes to ``Path('.')``. This mirrors the
    existing iterate-CLI behavior — both ship the help-text claim
    ``Pass empty string to disable``, and both actually produce the
    cwd as the dump root rather than disabling. Pinned here so a
    later "fix" to either side stays a paired change."""
    captured, _db = _run_eval_main_with_capture(
        monkeypatch, tmp_path,
        ["--hang-dump-dir", ""],
    )
    spec: EvalJobSpec = captured["spec"]
    # Path("") → Path(".") → str() == "."  -> still truthy, so the
    # CLI passes "." through rather than None. Document the quirk.
    assert spec.hang_dump_root == "."


def test_run_shard_eval_pull_threads_hang_dump_dir_into_worker_args(
    monkeypatch, tmp_path,
):
    """The end of the chain: ``run_shard_eval_pull`` builds
    ``hang_dump_dir`` from ``spec.hang_dump_root`` + run_id + label
    and includes it in the args submitted to the worker pool. Pin
    that args-tuple position so a future refactor of
    ``_play_one_match_game_alternating_indexed`` doesn't silently
    drop the forensic dir again.

    Mocking strategy: stub ``ProcessPoolExecutor`` with a fake that
    records submitted args, stub the queue HTTP to return one game
    and then 'closed', and rely on the executor never actually
    executing the worker (the recorded submit args is the
    observable)."""
    from mtg_bot.training import distributed as dist

    spec = EvalJobSpec(
        new_weights_path="new.pt",
        old_weights_path=None,  # heuristic opponent — no model load
        samples=1, inner_nodes=1, hidden_sizes=None,
        base_seed=0, enable_mulligans=False,
        per_game_timeout_sec=10.0, deck_pool_path=None,
        capture_every=0,
        opponent_label="heuristic",
        hang_dump_root="some_hangs",
        heuristics=HeuristicsConfig.disabled(),
    )

    submitted: list[tuple] = []

    class _FakeFuture:
        def __init__(self, args):
            self._args = args

        def result(self):
            # Return a sentinel ``(game_idx, (winner, decisions,
            # outcome_kind), wall_s)`` shape so the shard-loop's
            # accounting unpacks without raising. We pick winner=0,
            # outcome="completed" so neither crashed nor
            # timed_out branches fire — just a clean wins_new bump.
            game_idx = self._args[0]
            return game_idx, (0, 1, "completed"), 0.001

    class _FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def submit(self, fn, args):
            submitted.append(args)
            return _FakeFuture(args)

        def shutdown(self, wait=True):
            pass

    # ``run_shard_eval_pull`` does ``import concurrent.futures as _cf``
    # locally, so we patch the underlying module's symbols rather
    # than ``dist._cf``.
    import concurrent.futures as _cf
    monkeypatch.setattr(_cf, "ProcessPoolExecutor", _FakeExecutor)

    # _cf.wait should immediately surface "all done" so the shard
    # loop drains without blocking.
    def _fake_wait(futures, return_when=None):
        return set(futures), set()
    monkeypatch.setattr(_cf, "wait", _fake_wait)

    # Queue HTTP: first take returns one game (game_idx=0, remaining=0,
    # not closed), second take returns no games + closed=True.
    take_responses = iter([
        (json.dumps({
            "games": [0], "remaining": 0, "closed": False,
        }).encode("utf-8")),
        (json.dumps({
            "games": [], "remaining": 0, "closed": True,
        }).encode("utf-8")),
    ])

    class _FakeResp:
        def __init__(self, body):
            self.body = body
            self.status = 200

    def _fake_urlopen(req, **kw):
        return _FakeResp(next(take_responses))

    monkeypatch.setattr(dist, "_urlopen_with_retry", _fake_urlopen)

    # Skip NN weight loading (the spec.new_weights_path doesn't
    # exist on disk). The shard never actually invokes the factory
    # since the executor is faked, but `NNPimcConfig(...)` still
    # constructs without IO.

    # No progress dir = skip EvalProgress + EventLog.
    counts = dist.run_shard_eval_pull(
        spec,
        queue_url="http://master:5099",
        run_id="eval_v16_v13_test",
        n_workers=1,
        label="local",
        iter_total_games=1,
        progress_dir=None,
    )

    # One game submitted — args layout (see _build_args in
    # run_shard_eval_pull):
    #   (game_idx, p0_factory, p1_factory, base_seed, decision_cap,
    #    enable_mulligans, per_game_timeout_sec, deck_pool,
    #    replay_config, hang_dump_dir, heuristics)
    assert len(submitted) == 1
    args = submitted[0]
    hang_dump_dir = args[9]
    # Spec-level root + per-shard suffix.
    assert hang_dump_dir is not None
    assert Path(hang_dump_dir) == (
        Path("some_hangs") / "eval_v16_v13_test_shard_local"
    )
    assert counts["wins_new"] == 1


def test_run_shard_eval_pull_passes_none_when_root_unset(
    monkeypatch, tmp_path,
):
    """Counterpart: ``hang_dump_root=None`` -> ``hang_dump_dir=None``
    on the worker args (no per-shard subdir leak). Pins the disable
    path so an over-eager default value doesn't leak across the
    None boundary."""
    from mtg_bot.training import distributed as dist

    spec = EvalJobSpec(
        new_weights_path="new.pt",
        old_weights_path=None,
        samples=1, inner_nodes=1, hidden_sizes=None,
        base_seed=0, enable_mulligans=False,
        per_game_timeout_sec=10.0, deck_pool_path=None,
        capture_every=0,
        opponent_label="heuristic",
        hang_dump_root=None,
        heuristics=HeuristicsConfig.disabled(),
    )

    submitted: list[tuple] = []

    class _FakeFuture:
        def __init__(self, args):
            self._args = args

        def result(self):
            return self._args[0], (0, 1, "completed"), 0.001

    class _FakeExecutor:
        def __init__(self, max_workers):
            pass

        def submit(self, fn, args):
            submitted.append(args)
            return _FakeFuture(args)

        def shutdown(self, wait=True):
            pass

    import concurrent.futures as _cf
    monkeypatch.setattr(_cf, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(
        _cf, "wait",
        lambda futures, return_when=None: (set(futures), set()),
    )

    take_responses = iter([
        (json.dumps({
            "games": [0], "remaining": 0, "closed": False,
        }).encode("utf-8")),
        (json.dumps({
            "games": [], "remaining": 0, "closed": True,
        }).encode("utf-8")),
    ])

    class _FakeResp:
        def __init__(self, body):
            self.body = body
            self.status = 200

    monkeypatch.setattr(
        dist, "_urlopen_with_retry",
        lambda req, **kw: _FakeResp(next(take_responses)),
    )

    dist.run_shard_eval_pull(
        spec,
        queue_url="http://master:5099",
        run_id="eval_no_dump",
        n_workers=1,
        label="local",
        iter_total_games=1,
        progress_dir=None,
    )

    assert len(submitted) == 1
    assert submitted[0][9] is None
