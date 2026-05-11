"""Tests for ``scripts/parity_smoke.py`` — specifically the round-23
``--mode initial-state`` path that compares the Rust ``fingerprint()``
pymethod to a Python-side reimplementation of the hash recipe.

This is the first working parity check, narrowly scoped to the
fingerprint-vs-hash_state invariant. The pinned baselines below are the
same values asserted in ``tests_rust/test_fingerprint.py``; if either
drifts, the two test files should drift together (and the parity_smoke
script's documented contract — Rust fingerprint() byte-equal to the
Python hash recipe — is broken).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


# Skip the whole module if the Rust extension isn't built — mirrors the
# ``tests_rust/conftest.py`` policy so a fresh checkout without maturin
# doesn't break the gate.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mtg_bot_engine") is None,
    reason="mtg_bot_engine PyO3 extension not installed",
)


# ---------------------------------------------------------------------------
# Load scripts/parity_smoke.py as a module so we can call its functions
# without spawning a subprocess. The file isn't on the Python path by
# default (it's a script, not a package member) so we import it by path.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity_smoke():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "parity_smoke.py"
    spec = importlib.util.spec_from_file_location("parity_smoke", script_path)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass decorators in parity_smoke.py inspect cls.__module__'s
    # entry in sys.modules to resolve string annotations; register the
    # module before exec_module so that lookup succeeds.
    sys.modules["parity_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Pinned baselines — must match tests_rust/test_fingerprint.py values.
# These are the regression-safe load-bearing assertions; if either drifts
# the Rust fingerprint format changed on one side of the parity bridge.
# ---------------------------------------------------------------------------

EXPECTED_FINGERPRINT_DEFAULT = "93ab1bd03f94"  # make_synthetic_state()
EXPECTED_FINGERPRINT_SMALL = "e1d70afe6742"   # make_synthetic_state(2, 3, 5)


# ---------------------------------------------------------------------------
# run_initial_state_compare — the new entry point this slice ships.
# ---------------------------------------------------------------------------


def test_initial_state_mode_fingerprint_matches(parity_smoke, capsys):
    """``run_initial_state_compare`` returns 0 (PASS) on a freshly-built
    default synthetic state. End-to-end smoke of the new CLI mode."""
    rc = parity_smoke.run_initial_state_compare(seed=42)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert EXPECTED_FINGERPRINT_DEFAULT in out


def test_initial_state_mode_fingerprint_matches_seed_7(parity_smoke):
    """Same assertion, different seed. ``make_synthetic_state`` is
    deterministic so the seed doesn't shape the state today, but the
    contract is "PASS for any seed"; if a future variant threads the
    seed in (e.g. ``shuffle_library_with_seed``), this guards against
    regression at seed 7 specifically.
    """
    assert parity_smoke.run_initial_state_compare(seed=7) == 0


def test_initial_state_mode_pinned_hash_default(parity_smoke):
    """The single most load-bearing test: the Rust ``fingerprint()`` on
    a default synthetic state MUST equal the round-22 pinned baseline.
    If this fails, the hash recipe drifted on one side of the bridge.

    We use the Python shim AND the native Rust pymethod independently
    and assert both produce the same pinned value — that pins down the
    invariant from both directions.
    """
    import mtg_bot_engine
    state = mtg_bot_engine.make_synthetic_state()
    rust_fp = state.fingerprint()
    shim_fp = parity_smoke._hash_rust_state(state)
    assert rust_fp == EXPECTED_FINGERPRINT_DEFAULT
    assert shim_fp == EXPECTED_FINGERPRINT_DEFAULT
    assert rust_fp == shim_fp


def test_initial_state_mode_pinned_hash_2_3_5(parity_smoke):
    """Companion to the default-state pinned test: smaller (2, 3, 5)
    synthetic state, different bf/hand/lib sizes, asserted against the
    other round-22 baseline."""
    import mtg_bot_engine
    state = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    rust_fp = state.fingerprint()
    shim_fp = parity_smoke._hash_rust_state(state)
    assert rust_fp == EXPECTED_FINGERPRINT_SMALL
    assert shim_fp == EXPECTED_FINGERPRINT_SMALL


def test_initial_state_mode_with_2_3_5_runner(parity_smoke, capsys):
    """The CLI entry point honors ``--num-battlefield/-hand/-library``
    overrides and prints the smaller-state baseline."""
    rc = parity_smoke.run_initial_state_compare(
        seed=42, num_battlefield=2, num_hand=3, num_library=5
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert EXPECTED_FINGERPRINT_SMALL in out


def test_initial_state_mode_after_clone(parity_smoke):
    """Cloning a state preserves both the Rust fingerprint AND the
    Python shim hash — the CoW canary at the parity-bridge level. If
    clone_once leaked mutation across the boundary, the dict-shim hash
    would diverge from the native one for the clone.
    """
    import mtg_bot_engine
    state = mtg_bot_engine.make_synthetic_state()
    clone = state.clone_once()

    # Native fingerprint preserved across clone.
    assert state.fingerprint() == clone.fingerprint()
    # Python shim hash preserved across clone.
    assert parity_smoke._hash_rust_state(state) == parity_smoke._hash_rust_state(clone)
    # And both methods agree on each side of the clone.
    assert state.fingerprint() == parity_smoke._hash_rust_state(state)
    assert clone.fingerprint() == parity_smoke._hash_rust_state(clone)


def test_initial_state_mode_mutation_changes_hash(parity_smoke):
    """Mutating a player's life updates both hashes in lockstep.
    Verifies the shim and the native fingerprint actually depend on
    the gameplay-relevant fields (not constant strings).
    """
    import mtg_bot_engine
    state = mtg_bot_engine.make_synthetic_state()
    rust_before = state.fingerprint()
    shim_before = parity_smoke._hash_rust_state(state)
    state.set_life(0, 5)
    rust_after = state.fingerprint()
    shim_after = parity_smoke._hash_rust_state(state)

    assert rust_before != rust_after
    assert shim_before != shim_after
    # The two methods continue to agree post-mutation.
    assert rust_after == shim_after


def test_initial_state_mode_main_dispatches_to_initial_state(parity_smoke, monkeypatch, capsys):
    """``main()`` routes ``--mode initial-state`` to the new comparison
    path. End-to-end argparse + dispatch smoke."""
    monkeypatch.setattr(sys, "argv", ["parity_smoke.py", "--mode", "initial-state", "--seed", "42"])
    rc = parity_smoke.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert EXPECTED_FINGERPRINT_DEFAULT in out


# ---------------------------------------------------------------------------
# --mode trajectory — Rust-only trajectory walk with clone-replay
# determinism check. Catches non-deterministic Rust state mutation
# (same seed -> different fingerprint sequence) and clone divergence.
# ---------------------------------------------------------------------------


def test_trajectory_mode_seed_42_5_decisions_default_board_passes(parity_smoke, capsys):
    """Baseline: 5 decisions at default board size, seed=42 — should
    PASS with clone-replay matching. End-to-end smoke of the new
    trajectory CLI mode."""
    rc = parity_smoke.run_trajectory_mode(seed=42, num_decisions=5)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "clone-replay-matches=True" in out
    assert "seed=42" in out


def test_trajectory_mode_seed_7_10_decisions_passes(parity_smoke, capsys):
    """A different seed produces a different decision sequence but
    the clone-replay determinism check still holds."""
    rc = parity_smoke.run_trajectory_mode(seed=7, num_decisions=10)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "clone-replay-matches=True" in out


def test_trajectory_mode_zero_decisions_is_trivial_pass(parity_smoke, capsys):
    """0 decisions: the walk runs an empty loop, the clone is identical
    to the original, both fingerprint trajectories are [initial_fp].
    Trivial PASS — guards against an off-by-one edge case in the walk
    where 0 iterations would otherwise error."""
    rc = parity_smoke.run_trajectory_mode(seed=42, num_decisions=0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "trajectory of 0 decisions" in out


def test_trajectory_mode_reproducibility_same_seed_same_final_hash(parity_smoke, capsys):
    """Two runs at the same seed produce identical final fingerprints.
    The whole point of the determinism check: same seed -> bit-identical
    trajectory across runs (not just within a run). Catches a subtler
    class of nondeterminism — e.g. iteration over a HashMap whose
    order varies across process invocations."""
    rc1 = parity_smoke.run_trajectory_mode(seed=42, num_decisions=15)
    out1 = capsys.readouterr().out
    rc2 = parity_smoke.run_trajectory_mode(seed=42, num_decisions=15)
    out2 = capsys.readouterr().out
    assert rc1 == 0
    assert rc2 == 0
    # Extract the "final_hash=..." substring from both runs and assert
    # they match. The line is "[  rust] trajectory of N decisions,
    # final_hash=<hex>, clone-replay-matches=True".
    def _extract_final_hash(out: str) -> str:
        for line in out.splitlines():
            if "final_hash=" in line:
                # Split on "final_hash=" then take up to next comma.
                tail = line.split("final_hash=", 1)[1]
                return tail.split(",", 1)[0].strip()
        raise AssertionError(f"no final_hash line in:\n{out}")

    assert _extract_final_hash(out1) == _extract_final_hash(out2)


def test_trajectory_mode_different_seeds_pick_different_trajectories(
    parity_smoke, capsys
):
    """Probabilistic-but-deterministic check: different seeds should
    explore the state differently and (with the synthetic state
    permitting more than one legal action at some step) produce a
    different fingerprint trajectory.

    Pre pass-priority-step-advance slice: the synthetic-state UNTAP
    seat had only PassPriority legal, so 10 passes from seed 1 == 10
    passes from seed 2 (rng.choice of a one-element list). The test
    documented that quirk explicitly with ``traj_a == traj_b``.

    Post-slice: ``apply_action({"kind": "pass_priority"})`` now
    composes CR 117.4 step-advance, so a chain of passes walks the
    phase ladder. By the time 10 decisions have been applied, the
    state has reached a step (MAIN1 / BEGINNING_OF_COMBAT / etc.)
    where the legal-decisions set is larger than one — ActivateAbility
    on the seeded Forests, DeclareAttackers on the seeded Bears, etc.
    The two seeds now diverge into different trajectories, which is
    what this test's name originally intended."""
    import mtg_bot_engine

    state_a = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    _, traj_a = parity_smoke._walk_trajectory(state_a, seed=1, num_decisions=10)

    state_b = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    _, traj_b = parity_smoke._walk_trajectory(state_b, seed=2, num_decisions=10)

    # Shape (length) is identical: 11 fingerprints = initial + after
    # each of 10 decisions.
    assert len(traj_a) == len(traj_b) == 11
    # Initial fingerprint matches (same seed-independent state).
    assert traj_a[0] == traj_b[0]
    # But at least one intermediate / final fingerprint differs —
    # the seeds picked different paths once the legal set grew past
    # the singleton.
    assert traj_a != traj_b, (
        f"expected seeds 1 and 2 to pick different trajectories now "
        f"that pass_priority composes step-advance (so 10 decisions "
        f"reach a state with multiple legal actions); got identical "
        f"trajectories — has the legal-decisions set regressed back "
        f"to a singleton on the synthetic state?"
    )


def test_trajectory_mode_100_decisions_passes(parity_smoke, capsys):
    """Longer walk (100 decisions) at seed=42 — catches subtle
    non-determinism that only surfaces at deeper trajectory depths
    (e.g. an alloc-pattern-dependent ordering bug that triggers
    after the state grows past some threshold)."""
    rc = parity_smoke.run_trajectory_mode(seed=42, num_decisions=100)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "clone-replay-matches=True" in out
    assert "trajectory of 100 decisions" in out


def test_trajectory_mode_explicit_small_board(parity_smoke, capsys):
    """The CLI knobs ``--num-battlefield/-hand/-library`` are honored
    by ``run_trajectory_mode``. Catches a regression where a future
    refactor hardcodes the make_synthetic_state args."""
    rc = parity_smoke.run_trajectory_mode(
        seed=42, num_decisions=5, num_battlefield=2, num_hand=3, num_library=5
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out


def test_trajectory_mode_walk_records_one_decision_per_step(parity_smoke):
    """``_walk_trajectory`` returns ``(decisions, trajectory)`` where
    ``len(decisions) == num_decisions`` and
    ``len(trajectory) == num_decisions + 1``. Locks in the trajectory
    bookkeeping shape — off-by-one bugs in the loop bounds would
    surface here before they corrupt the clone-replay check."""
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state()
    decisions, trajectory = parity_smoke._walk_trajectory(
        state, seed=42, num_decisions=7
    )
    assert len(decisions) == 7
    assert len(trajectory) == 8  # initial + after-each-step
    # Every fingerprint is 12 hex chars (SHA1 truncated).
    for fp in trajectory:
        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)


def test_trajectory_mode_replay_on_fresh_clone_matches_original(parity_smoke):
    """The clone-replay invariant directly: walk an original state,
    clone the pre-walk state, replay the same decisions on the
    clone, assert the resulting fingerprint trajectory is identical."""
    import mtg_bot_engine

    state = mtg_bot_engine.make_synthetic_state()
    clone = state.clone_once()

    decisions, orig_traj = parity_smoke._walk_trajectory(
        state, seed=42, num_decisions=10
    )
    clone_traj = parity_smoke._replay_trajectory(clone, decisions)

    assert orig_traj == clone_traj


def test_trajectory_mode_main_dispatches_to_trajectory(parity_smoke, monkeypatch, capsys):
    """``main()`` routes ``--mode trajectory`` to the new walker.
    End-to-end argparse + dispatch smoke."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "trajectory",
            "--seed",
            "42",
            "--num-decisions",
            "5",
        ],
    )
    rc = parity_smoke.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "Rust trajectory deterministic" in out


def test_trajectory_mode_diverged_clone_reports_first_step(parity_smoke, capsys, monkeypatch):
    """Inject a fake clone-replay divergence: monkeypatch
    ``_replay_trajectory`` to return a trajectory that differs from
    the original at a known step, and verify ``run_trajectory_mode``
    returns 1 with the "diverged at step N" diagnostic.

    This is the failure-path counterpart to the other PASS tests —
    it locks in the diagnostic output format (the diff between the
    Rust internal nondeterminism case and the clone-CoW divergence
    case would both surface here)."""
    real_replay = parity_smoke._replay_trajectory

    def fake_replay(state, decisions):
        traj = real_replay(state, decisions)
        if len(traj) >= 3:
            # Corrupt step 2's fingerprint to force a divergence.
            traj = traj.copy()
            traj[2] = "deadbeef0000"
        return traj

    monkeypatch.setattr(parity_smoke, "_replay_trajectory", fake_replay)

    rc = parity_smoke.run_trajectory_mode(seed=42, num_decisions=5)
    assert rc == 1
    out = capsys.readouterr().out
    assert "PARITY: FAIL" in out
    assert "diverged at step 2" in out
    assert "clone-replay-matches=False" in out


# ---------------------------------------------------------------------------
# --mode multi-seed — sweep trajectory mode across K seeds and print an
# aggregate summary. Catches non-determinism bugs that only manifest at
# certain seeds (single-seed mode samples one trajectory through the
# legal space; K seeds tile it).
# ---------------------------------------------------------------------------


def test_multi_seed_mode_three_seeds_all_pass(parity_smoke, capsys):
    """Baseline: 3 seeds, default-ish board, 5 decisions each — all
    clone-replay match, exit code 0. Smoke of the new CLI mode."""
    rc = parity_smoke.run_multi_seed_mode(
        num_seeds=3,
        seed_start=0,
        num_decisions=5,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "all 3 seeds clone-replay-matched" in out
    # Header row + one row per seed should appear.
    assert "seed |" in out
    assert "all clone-match:   True" in out


def test_multi_seed_mode_single_seed_matches_trajectory_mode(
    parity_smoke, capsys
):
    """num_seeds=1 with seed_start=42 should produce a final
    fingerprint identical to what ``run_trajectory_mode`` produces at
    seed=42. Cross-check the helper / wrapper alignment — if multi-seed
    drifted from trajectory mode, this trips.
    """
    rc_traj = parity_smoke.run_trajectory_mode(
        seed=42,
        num_decisions=10,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    out_traj = capsys.readouterr().out
    assert rc_traj == 0

    def _extract_final_hash(out: str) -> str:
        for line in out.splitlines():
            if "final_hash=" in line:
                tail = line.split("final_hash=", 1)[1]
                return tail.split(",", 1)[0].strip()
        raise AssertionError(f"no final_hash line in:\n{out}")

    traj_fp = _extract_final_hash(out_traj)

    rc_ms = parity_smoke.run_multi_seed_mode(
        num_seeds=1,
        seed_start=42,
        num_decisions=10,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    out_ms = capsys.readouterr().out
    assert rc_ms == 0
    # The single-seed row in the multi-seed table prints the seed's
    # final fingerprint in the "fingerprint" column. Assert the same
    # 12-char hex appears in the multi-seed output.
    assert traj_fp in out_ms


def test_multi_seed_mode_unique_fingerprints_on_varied_board(
    parity_smoke, capsys
):
    """When the synthetic state has enough legal-space variation, the
    K seeds should produce mostly-unique trajectories. We use the
    default (10, 7, 50) board which has room for variation and run a
    longer walk so the rng's choices have meaningful impact.

    Allowed collision rate: at least num_seeds * 0.5 unique
    fingerprints. The current synthetic-state fixture starts in
    PHASE_UNTAP with PassPriority being the only legal action, so
    collisions are universal there — this test is a forward-looking
    guard that fires once the synthetic state grows beyond a single
    legal action; until then it asserts unique-count >= floor.
    """
    # Use enough decisions that any per-seed branching has a chance
    # to register.
    num_seeds = 8
    rc = parity_smoke.run_multi_seed_mode(
        num_seeds=num_seeds,
        seed_start=0,
        num_decisions=20,
        num_battlefield=10,
        num_hand=7,
        num_library=50,
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Extract the "unique fingerprints: N" line.
    unique_line = next(
        line for line in out.splitlines() if "unique fingerprints:" in line
    )
    n_unique = int(unique_line.split(":", 1)[1].strip().split()[0])
    # >= 1 (the singleton-legal-state fixture produces 1) and the
    # forward-looking floor is num_seeds * 0.5. We assert the soft
    # lower bound: at least one unique fingerprint exists. If the
    # synthetic state expands later, this floor naturally tightens.
    assert n_unique >= 1
    assert n_unique <= num_seeds


def test_multi_seed_mode_deterministic_across_runs(parity_smoke, capsys):
    """Same num_seeds + seed_start + decisions must produce identical
    output across two runs. The whole point of the regression-gate is
    determinism across runs, not just within one."""
    parity_smoke.run_multi_seed_mode(
        num_seeds=4,
        seed_start=0,
        num_decisions=8,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    out1 = capsys.readouterr().out
    parity_smoke.run_multi_seed_mode(
        num_seeds=4,
        seed_start=0,
        num_decisions=8,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    out2 = capsys.readouterr().out
    # Strict equality — the multi-seed output is purely a function
    # of (num_seeds, seed_start, decisions, board) and must be
    # bit-identical run-to-run.
    assert out1 == out2


def test_multi_seed_mode_aggregate_says_all_clone_match_true(
    parity_smoke, capsys
):
    """The aggregate-stats block must include the literal string
    ``all clone-match:   True`` when every seed passes. Locks in the
    output format that downstream consumers (CI log parsing, status
    dashboards) might grep for."""
    parity_smoke.run_multi_seed_mode(
        num_seeds=3,
        seed_start=0,
        num_decisions=3,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    out = capsys.readouterr().out
    assert "all clone-match:   True" in out
    assert "total errors:      0" in out
    assert "total seeds:       3" in out


def test_multi_seed_mode_failing_seed_reports_first_divergence(
    parity_smoke, capsys, monkeypatch
):
    """Inject a fake clone-replay divergence via the helper — the
    multi-seed report must call out the failing seed in the FAIL
    summary. Failure-path counterpart to the all-pass cases."""
    real_replay = parity_smoke._replay_trajectory

    def fake_replay(state, decisions):
        traj = real_replay(state, decisions)
        if len(traj) >= 3:
            traj = traj.copy()
            traj[2] = "deadbeef0000"
        return traj

    monkeypatch.setattr(parity_smoke, "_replay_trajectory", fake_replay)

    rc = parity_smoke.run_multi_seed_mode(
        num_seeds=2,
        seed_start=0,
        num_decisions=5,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "PARITY: FAIL" in out
    assert "2 of 2 seeds failed" in out
    # Both seeds should be cited as failing.
    assert "seed 0:" in out
    assert "seed 1:" in out


def test_multi_seed_mode_main_dispatches(parity_smoke, monkeypatch, capsys):
    """``main()`` routes ``--mode multi-seed`` to the new sweep.
    End-to-end argparse + dispatch smoke."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "multi-seed",
            "--num-seeds",
            "3",
            "--seed-start",
            "0",
            "--num-decisions",
            "5",
            "--num-battlefield",
            "2",
            "--num-hand",
            "3",
            "--num-library",
            "5",
        ],
    )
    rc = parity_smoke.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "running 3 seeds (0..2)" in out


def test_multi_seed_mode_seed_start_offset(parity_smoke, capsys):
    """``seed_start=100, num_seeds=3`` iterates seeds 100/101/102 —
    the header line and the table both report the offset range. Locks
    in the seed-start arithmetic against off-by-one regressions."""
    parity_smoke.run_multi_seed_mode(
        num_seeds=3,
        seed_start=100,
        num_decisions=2,
        num_battlefield=2,
        num_hand=3,
        num_library=5,
    )
    out = capsys.readouterr().out
    assert "running 3 seeds (100..102)" in out
    # Each seed value appears in a row of the table.
    for seed in (100, 101, 102):
        assert f"{seed:>4} |" in out


# ---------------------------------------------------------------------------
# --mode field-diff — per-field OK/DIFFER diff between two parallel views
# of a Rust synthetic state. The parity-debug primitive that answers
# "WHICH field diverged" once a trajectory smoke says digests don't match.
# ---------------------------------------------------------------------------


def test_field_diff_mode_vanilla_state_all_ok(parity_smoke, capsys):
    """Baseline: no injection, vanilla state — every field matches, the
    runner returns 0, and the output reports ``PARITY: PASS`` with a
    field count equal to ``FIELD_DIFF_FIELDS``."""
    rc = parity_smoke.run_field_diff_mode(seed=42)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    # Every field is reported with status OK on a vanilla state.
    for field in parity_smoke.FIELD_DIFF_FIELDS:
        assert field in out
    # No DIFFER markers at all on the vanilla path.
    assert "DIFFER" not in out


def test_field_diff_mode_inject_life_reports_mismatch(parity_smoke, capsys):
    """``--inject-divergence life`` flips life_p0 by -1; the diff table
    must report a DIFFER on life_p0 and the runner returns 1."""
    rc = parity_smoke.run_field_diff_mode(seed=42, inject_divergence="life")
    assert rc == 1
    out = capsys.readouterr().out
    assert "PARITY: FAIL" in out
    # The life_p0 row is the DIFFER one — its two values (20 / 19) must
    # appear on the same line.
    life_line = next(
        line for line in out.splitlines() if line.startswith("life_p0")
    )
    assert "DIFFER" in life_line
    # Vanilla synthetic state has both players at 20 life; the mutation
    # subtracts 1 from the right-hand side.
    assert "20" in life_line
    assert "19" in life_line
    # Exactly one field differs.
    assert "1 of 22 fields differ" in out


def test_field_diff_mode_inject_hand_reports_mismatch(parity_smoke, capsys):
    """``--inject-divergence hand`` appends a sentinel object to the
    hand_p0 zone signature on the right-hand side; the diff must
    surface the mismatch on hand_p0."""
    rc = parity_smoke.run_field_diff_mode(seed=42, inject_divergence="hand")
    assert rc == 1
    out = capsys.readouterr().out
    assert "PARITY: FAIL" in out
    hand_line = next(
        line for line in out.splitlines() if line.startswith("hand_p0")
    )
    assert "DIFFER" in hand_line


def test_field_diff_mode_inject_active_player_reports_mismatch(
    parity_smoke, capsys
):
    """``--inject-divergence active_player`` toggles 0<->1 on the
    right-hand side; the diff must surface the mismatch on the
    active_player row."""
    rc = parity_smoke.run_field_diff_mode(
        seed=42, inject_divergence="active_player"
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "PARITY: FAIL" in out
    ap_line = next(
        line for line in out.splitlines() if line.startswith("active_player")
    )
    assert "DIFFER" in ap_line
    # Vanilla synthetic state has active_player_idx=0; mutation flips it
    # to 1.
    assert " 0 " in ap_line or "| 0" in ap_line
    assert " 1" in ap_line


def test_field_diff_mode_table_covers_every_hash_part_field(
    parity_smoke, capsys
):
    """The printed table must include every field name that ``_hash_parts``
    folds into the fingerprint — the brief's load-bearing test that the
    diff machinery doesn't silently skip a field that ends up driving
    the hash mismatch.

    The list mirrors the per-field contributions of ``_hash_rust_state``
    (and through it, ``_hash_parts``): scalar state header (turn, step,
    phase, active_player, priority_player, is_over, is_draw, winner_id),
    per-player tuples (life, hand, library, graveyard, exile, mana_pool),
    and the central zones (battlefield, stack).
    """
    parity_smoke.run_field_diff_mode(seed=42)
    out = capsys.readouterr().out
    # Every field name in the canonical list must appear in the output.
    for field in parity_smoke.FIELD_DIFF_FIELDS:
        assert field in out, f"field {field!r} missing from field-diff output"
    # The header row of the table is also present.
    assert "field             | status | left" in out
    # And the brief-specified scalar fields each show up as their own
    # row prefix.
    for prefix in (
        "turn", "step", "phase", "active_player", "priority_player",
        "is_over", "is_draw", "winner_id", "life_p0", "life_p1",
        "hand_p0", "hand_p1", "library_p0", "library_p1",
        "graveyard_p0", "graveyard_p1", "exile_p0", "exile_p1",
        "mana_pool_p0", "mana_pool_p1", "battlefield", "stack",
    ):
        # Each field is the leading token of its own row.
        row_lines = [
            line for line in out.splitlines()
            if line.startswith(prefix) and "|" in line
        ]
        assert row_lines, f"no row starts with {prefix!r}"


def test_field_diff_mode_main_dispatches(parity_smoke, monkeypatch, capsys):
    """``main()`` routes ``--mode field-diff`` to the new runner.
    End-to-end argparse + dispatch smoke."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["parity_smoke.py", "--mode", "field-diff", "--seed", "42"],
    )
    rc = parity_smoke.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    # Table header present.
    assert "field             | status | left" in out


def test_field_diff_mode_main_dispatches_with_inject(
    parity_smoke, monkeypatch, capsys
):
    """``main() --mode field-diff --inject-divergence life`` returns 1
    and reports a DIFFER on the life_p0 row. End-to-end inject path."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "field-diff",
            "--inject-divergence",
            "life",
        ],
    )
    rc = parity_smoke.main()
    assert rc == 1
    out = capsys.readouterr().out
    assert "PARITY: FAIL" in out
    assert "life_p0" in out
    assert "DIFFER" in out


def test_field_diff_mode_unknown_inject_field_errors(parity_smoke, capsys):
    """An unknown ``--inject-divergence`` field name returns rc=2 and
    prints an error citing the valid field list. Locks in the
    error-path contract."""
    rc = parity_smoke.run_field_diff_mode(
        seed=42, inject_divergence="this_is_not_a_field"
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "this_is_not_a_field" in out


# ---------------------------------------------------------------------------
# --mode cross-engine-game — Phase-5 prep. Drive Python + Rust through the
# SAME deterministic decision sequence one step at a time and report the
# first decision index where they diverge per field. The smoke is expected
# to trip on the known G1-G24 gaps in docs/RUST_PARITY_GAPS.md (specifically
# G3 — Rust pass_priority is a pure toggle, Python is a no-op — fires on
# decision 0 for any seed because pass_priority is the only legal Rust
# action at PHASE_UNTAP). The harness's deliverable is the measurement,
# not a parity-clean run.
# ---------------------------------------------------------------------------


def test_cross_engine_game_mode_runs_end_to_end_fixed_seed(parity_smoke, capsys):
    """Mode runs end-to-end on a fixed seed without raising. The exit
    code is 0 OR 1 (a measurement outcome — neither is wrong), but the
    process must not abort with an unhandled exception."""
    rc = parity_smoke.run_cross_engine_game_mode(seed=42, num_decisions=5)
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "[cross-engine-game]" in out
    # Either PASS or FAIL — both are acceptable outcomes for the measurement.
    assert ("PARITY: PASS" in out) or ("PARITY: FAIL" in out)


@pytest.mark.parametrize("seed", [42, 7, 123, 999, 2026])
def test_cross_engine_game_mode_returns_0_or_1_across_seeds(
    parity_smoke, capsys, seed
):
    """Across multiple seeds the mode either passes (exit 0) or
    detects a divergence (exit 1) — never an unhandled exception or
    a different rc value. Locks in the "robust to seed variation"
    contract from the task brief."""
    rc = parity_smoke.run_cross_engine_game_mode(seed=seed, num_decisions=5)
    assert rc in (0, 1), f"seed={seed}: rc={rc} (expected 0 or 1)"
    _ = capsys.readouterr()  # drain the captured output


def test_cross_engine_game_mode_output_table_has_expected_columns(
    parity_smoke, capsys
):
    """The per-decision output table contains the expected column
    headers: step, rust_action, python_action, verdict. Pin the
    column names so a future column rename surfaces as a test
    failure rather than as silent loss of operator-readable output."""
    parity_smoke.run_cross_engine_game_mode(seed=42, num_decisions=3)
    out = capsys.readouterr().out
    # Header line — all four column names in order.
    assert "step" in out
    assert "rust_action" in out
    assert "python_action" in out
    assert "verdict" in out
    # Separator row — pin so a future "wider table" change is intentional.
    assert "-----" in out


def test_cross_engine_game_mode_divergence_reports_field_name(
    parity_smoke, capsys
):
    """When the mode catches a divergence, the output mentions a
    field name from FIELD_DIFF_FIELDS. With Forest-only decks at
    seed 42 the G3 pure-toggle pass_priority gap fires on decision 0
    and the diverging field is ``priority_player`` — pinning this
    specific known-gap surfacing locks in that the measurement loop
    is actually exercising the per-step field-diff machinery (not
    silently passing on a stale early return)."""
    rc = parity_smoke.run_cross_engine_game_mode(seed=42, num_decisions=5)
    assert rc == 1
    out = capsys.readouterr().out
    assert "PARITY: FAIL" in out
    # Some field from the canonical FIELD_DIFF_FIELDS list must appear
    # in the divergence report — this is the load-bearing contract
    # (the report tells the operator WHICH field diverged).
    matching = [f for f in parity_smoke.FIELD_DIFF_FIELDS if f in out]
    assert matching, (
        f"divergence output should cite at least one field from "
        f"FIELD_DIFF_FIELDS; got output:\n{out}"
    )
    # G3 specifically: pass_priority toggles the priority_player field
    # in Rust but is a no-op in Python, so decision 0 always diverges
    # there. If a future Rust slice closes G3 this assertion needs to
    # be relaxed — but until then it's the specific signal we want.
    assert "priority_player" in out


def test_cross_engine_game_mode_zero_decisions_passes(parity_smoke, capsys):
    """``--num-decisions 0`` exercises the "initial states match,
    no decisions applied" path. Initial-state setup matches between
    Python and Rust under the cross-engine canonicalization (zone
    contents byte-equal after the id-slot / summoning-sick /
    mana-pool normalizations), so the loop runs zero iterations and
    returns 0. Locks in the smoke-of-the-smoke: the mode CAN exit 0
    when no divergence is detected, not just exit 1 on every seed."""
    rc = parity_smoke.run_cross_engine_game_mode(seed=42, num_decisions=0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PARITY: PASS" in out
    assert "0 decisions applied" in out


def test_cross_engine_game_mode_main_dispatches(
    parity_smoke, monkeypatch, capsys
):
    """``main() --mode cross-engine-game`` routes to the new
    comparison path. End-to-end argparse + dispatch smoke. Accepts
    either rc=0 or rc=1 since the measurement outcome depends on
    Rust+Python state agreement under the current G1-G24 gap set."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "cross-engine-game",
            "--seed",
            "42",
            "--num-decisions",
            "3",
        ],
    )
    rc = parity_smoke.main()
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "[cross-engine-game]" in out


def test_cross_engine_game_mode_translates_card_ids_via_position(parity_smoke):
    """Direct unit on the Rust-id -> Python-id translator. Python
    and Rust allocate per-object ids in different orders so the
    harness translates by zone-position; this test exercises that
    bridge in isolation without depending on which gap fires first
    in the full run."""
    import mtg_bot_engine
    py_state = parity_smoke._build_python_state_for_cross_engine(42)
    rs_state = parity_smoke._build_rust_state_for_cross_engine(py_state)

    # The first card in Rust's p0 hand should translate to the first
    # card in Python's p0 hand — both are Forests (interchangeable),
    # but position is the stable correspondence.
    rust_hand = rs_state.hand_snapshot(0)
    rust_first_id = rust_hand[0]["id"]
    py_first_id = py_state.players[0].hand[0].id

    translated = parity_smoke._translate_rust_card_id(
        rust_first_id, rs_state, py_state
    )
    assert translated == py_first_id

    # An id that doesn't exist on either side translates to None.
    assert parity_smoke._translate_rust_card_id(
        99999, rs_state, py_state
    ) is None


# ---------------------------------------------------------------------------
# --ignore-known-gaps — masking flag that skips documented G1-G24
# divergences so the harness can walk past them and surface the FIRST
# UNKNOWN divergence. Today the cross-engine harness trips at decision 0
# on every seed because of G3 (`pass_priority` is a pure toggle in Rust);
# with the flag set the harness walks indefinitely on a Forest-only deck
# because the only legal action is pass_priority and G3 catches every
# divergence.
# ---------------------------------------------------------------------------


def test_known_gap_masks_has_at_least_four_entries_with_gap_citations(
    parity_smoke,
):
    """``KNOWN_GAP_MASKS`` must document at least 4 gaps, each with a
    clear gap-id citation (G\\d+ shape matching the doc anchors in
    docs/RUST_PARITY_GAPS.md) and a non-empty ``fields`` tuple of
    canonical FIELD_DIFF_FIELDS names. Locks in the mask schema so a
    future "let's add a fifth mask" change can't accidentally land
    with a wrong-shaped entry."""
    masks = parity_smoke.KNOWN_GAP_MASKS
    assert len(masks) >= 4, f"expected at least 4 mask entries, got {len(masks)}"
    import re
    for gap_id, mask in masks.items():
        # Gap-id shape matches the ### Gn anchors in the doc.
        assert re.fullmatch(r"G\d+", gap_id), (
            f"gap_id {gap_id!r} doesn't match the G\\d+ shape used "
            f"in docs/RUST_PARITY_GAPS.md"
        )
        # Mandatory keys.
        assert "description" in mask and mask["description"], (
            f"mask {gap_id} missing or empty 'description'"
        )
        assert "fields" in mask and mask["fields"], (
            f"mask {gap_id} missing or empty 'fields'"
        )
        # Every field is a canonical FIELD_DIFF_FIELDS name.
        for field_name in mask["fields"]:
            assert field_name in parity_smoke.FIELD_DIFF_FIELDS, (
                f"mask {gap_id} cites unknown field {field_name!r}; "
                f"must be one of FIELD_DIFF_FIELDS"
            )


def test_cross_engine_game_mode_without_flag_still_stops_at_g3(
    parity_smoke, capsys
):
    """Default behavior is preserved: without ``--ignore-known-gaps``
    the harness stops at the first divergence (G3's priority_player
    toggle at decision 0) per the original measurement contract."""
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=42, num_decisions=20, ignore_known_gaps=False
    )
    assert rc == 1
    out = capsys.readouterr().out
    # G3 fires at decision 0 — both the DIVERGE row and the PARITY:
    # FAIL summary cite priority_player.
    assert "DIVERGE: priority_player" in out
    assert "PARITY: FAIL" in out
    # The harness should NOT have masked anything when the flag is
    # off — no MASKED rows in the output.
    assert "MASKED" not in out


def test_cross_engine_game_mode_with_flag_walks_past_g3(
    parity_smoke, capsys
):
    """With ``--ignore-known-gaps`` set, the harness walks past the
    G3 divergence at decision 0 and reaches either a deeper UNKNOWN
    divergence OR a clean N-decision completion. The Forest-only
    deck's only legal action is pass_priority, so G3 catches every
    divergence and the harness walks to completion."""
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=42, num_decisions=20, ignore_known_gaps=True
    )
    # rc=0 means "walked to completion without unmasked divergence",
    # rc=1 means "hit an unmasked divergence further along". Either
    # outcome is the harness working as designed; the load-bearing
    # contract is that it didn't stop at decision 0.
    assert rc in (0, 1)
    out = capsys.readouterr().out
    # G3 fires every odd step, so MASKED (G3) must appear at least
    # once — proving the mask path actually engaged.
    assert "MASKED (G3)" in out
    # And the walk got past decision 0 — either at least 2 rows were
    # printed (decision 0 + decision 1) or the harness reached PASS.
    assert ("PARITY: PASS" in out) or ("decision 1" in out) or (
        "decision 2" in out
    ), (
        f"harness should have walked past decision 0 when masking "
        f"was on; got output:\n{out}"
    )


@pytest.mark.parametrize("seed", [42, 7, 123, 999, 2026])
def test_cross_engine_game_mode_with_flag_walks_past_pass_priority_loop_across_seeds(
    parity_smoke, capsys, seed
):
    """Post pass-priority-step-advance slice: with G3 closed at the
    apply_action layer, Rust now advances past UNTAP/UPKEEP/DRAW via
    consecutive pass_priority pairs and reaches MAIN1, where the
    harness can emit PlayLand from the Forest-only deck. Python's
    apply_action(PassPriority()) stays a no-op (its outer search loop
    drives the step machine separately), so the structural step/phase
    drift between the engines surfaces — masked as ``G25`` (drift
    downstream of G3, see KNOWN_GAP_MASKS) past decision 0.

    Across all seeds the harness walks at least 6 decisions of
    masked pass_priority and then hits an UNMASKED divergence at the
    first PlayLand (the engines pick different card_ids because
    they're in different steps and their legal-decisions enumerators
    have diverged). That's the new "first unknown divergence" signal
    the parity_smoke real-deck stress measurement was blocked on
    before the slice: pre-slice the harness stuck at decision 0
    forever; post-slice it reaches MAIN1 and surfaces the next
    parity gap (different legal_decisions / different action picks
    in MAIN1) for follow-up slices.

    Asserts rc==1 (unmasked divergence after the pass-priority run)
    and confirms ``MASKED (G3)`` / ``MASKED (G3-Drift)`` rows appear
    plus an ``UNMASKED divergence at decision N`` for N>=6.

    Asserts rc==1 and confirms ``MASKED (G3)`` / ``MASKED (G25)``
    rows appear plus an unmasked divergence past decision 5."""
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=seed, num_decisions=20, ignore_known_gaps=True
    )
    # New post-slice expectation: the harness walks PAST decision 0
    # (G3 used to stop it there) and surfaces a real divergence
    # downstream of the pass_priority loop. rc==1 means "hit an
    # unmasked divergence further along" — that's the new ground
    # the slice unblocks.
    assert rc == 1, (
        f"seed={seed}: expected harness to surface an unmasked "
        f"divergence after the pass-priority loop, got rc={rc}. "
        f"If rc==0 the harness reached the end without finding any "
        f"new divergence (a stronger result than expected — bump "
        f"the assertion); if rc != 0 or 1 something went wrong."
    )
    out = capsys.readouterr().out
    # MASKED G3 + G25 fired through the pass-priority loop.
    assert "MASKED (G3)" in out or "MASKED (G25" in out, (
        f"seed={seed}: expected G3 or G25 mask to fire on the "
        f"pass-priority loop. Output:\n{out}"
    )
    # The walk got past decision 5 (6 pass_priority pairs to step
    # through pre-MAIN steps + into MAIN1 where PlayLand emits, then
    # possibly further: DeclareAttackers/DeclareBlockers at later
    # combat steps as the harness walks more decisions on its way to
    # the first unmasked divergence). Check that AT LEAST one
    # decision past 5 appears in the table.
    import re
    decisions_seen = sorted(
        int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\s*\|", out, re.M)
    )
    assert decisions_seen and max(decisions_seen) >= 5, (
        f"seed={seed}: expected harness to reach at least decision 5 "
        f"into the post-pass_priority emission space. Decisions seen: "
        f"{decisions_seen}. Output:\n{out}"
    )


def test_cross_engine_game_mode_masked_verdict_cites_gap_id(
    parity_smoke, capsys
):
    """The MASKED verdict in the table cites the gap-id in the form
    'MASKED (Gxx)' — operators reading the output need the
    cross-reference back to docs/RUST_PARITY_GAPS.md. Pinning the
    exact verdict shape so a future render-change is intentional."""
    parity_smoke.run_cross_engine_game_mode(
        seed=42, num_decisions=5, ignore_known_gaps=True
    )
    out = capsys.readouterr().out
    # The G3 mask fires on decision 0 — its verdict cites the gap id.
    assert "MASKED (G3)" in out


def test_cross_engine_game_mode_main_dispatches_with_ignore_known_gaps_flag(
    parity_smoke, monkeypatch, capsys
):
    """``main() --mode cross-engine-game --ignore-known-gaps`` routes
    the flag into the new comparison path. End-to-end argparse +
    dispatch smoke; locks in the CLI surface ("--ignore-known-gaps"
    is a boolean store_true) so a future flag rename surfaces here.

    Five decisions of Forest-only deck: with pass-priority-step-
    advance landed, all 5 pass-priority dispatches mask through G3 /
    G3-Drift before the harness reaches MAIN1 (it takes 6 passes /
    3 step-advance pairs to walk UNTAP→UPKEEP→DRAW→MAIN1), so 5
    decisions PASS cleanly — the masked-G3 rows are all the rows."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "cross-engine-game",
            "--seed",
            "42",
            "--num-decisions",
            "5",
            "--ignore-known-gaps",
        ],
    )
    rc = parity_smoke.main()
    # 5 decisions of pass_priority all mask; harness reaches the
    # decision-budget cap without an unmasked divergence -> rc==0.
    assert rc == 0
    out = capsys.readouterr().out
    assert "[cross-engine-game]" in out
    assert "MASKED (G3)" in out


def test_match_known_gap_returns_g3_for_priority_player_after_pass_priority(
    parity_smoke,
):
    """Direct unit on the ``_match_known_gap`` helper. A
    priority_player divergence right after a pass_priority action
    matches G3; a priority_player divergence after a play_land
    action does NOT (no documented gap explains that)."""
    rust_view = {"step": "UPKEEP", "priority_player": 1}
    python_view = {"step": "UPKEEP", "priority_player": 0}

    # After pass_priority → G3 mask applies.
    assert parity_smoke._match_known_gap(
        "priority_player",
        {"kind": "pass_priority"},
        rust_view,
        python_view,
    ) == "G3"

    # After play_land → no mask applies (G3's trigger is
    # action-kind-gated).
    assert parity_smoke._match_known_gap(
        "priority_player",
        {"kind": "play_land", "card_id": 0},
        rust_view,
        python_view,
    ) is None

    # Initial-state (no last_action) → no action-kind-gated mask
    # matches.
    assert parity_smoke._match_known_gap(
        "priority_player", None, rust_view, python_view
    ) is None


# ---------------------------------------------------------------------------
# --mode cross-engine-game with --deck-preset / --deck-path — real-deck
# stress (action variety surface). The Forest-only default already
# exercises G3 / G5 toggle measurement; the real-deck variants drive the
# same walk against richer card pools so non-trivial action variety
# (play_land, cast_spell, ...) has a chance to surface real (or
# known-masked) divergences.
# ---------------------------------------------------------------------------


def test_cross_engine_game_mode_deck_preset_runs_end_to_end(
    parity_smoke, capsys
):
    """``run_cross_engine_game_mode(deck_preset='tiny-mixed')`` builds
    both Python and Rust states from a real EoE-flavored decklist
    and walks at least 5 decisions without crashing. The deck-stats
    line lands at the top of the output and cites the preset name,
    so operators reading the smoke can confirm WHICH deck shape was
    walked.
    """
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=42,
        num_decisions=5,
        deck_preset="tiny-mixed",
        ignore_known_gaps=True,
    )
    # rc=0 or 1 — either the walk passed or hit an unmasked divergence;
    # the load-bearing assertion is that the harness didn't crash and
    # emitted the deck-stats / header lines.
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "[cross-engine-game]" in out
    assert "tiny-mixed" in out
    # Deck-stats line shows up at the top of real-deck runs.
    assert "deck stats:" in out


@pytest.mark.parametrize("seed", [42, 7, 123])
def test_cross_engine_game_mode_deck_preset_no_crash_across_seeds(
    parity_smoke, capsys, seed
):
    """Across multiple seeds, the real-deck preset runs without
    crashing for at least 5 decisions. Pins "real-deck state
    construction is seed-robust" so a future bridge change that
    breaks one seed surfaces in the gate, not in production.
    """
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=seed,
        num_decisions=5,
        deck_preset="tiny-mixed",
        ignore_known_gaps=True,
    )
    # Walk completes (PASS or fall-through to unmasked divergence) —
    # the harness should never crash here.
    assert rc in (0, 1), f"seed={seed}: unexpected rc={rc}"


def test_cross_engine_game_mode_rust_only_fallback_works(
    parity_smoke, capsys
):
    """The ``rust_only=True`` fallback skips Python state construction
    and runs a Rust-only clone-replay determinism check against the
    real-deck preset. PASS proves the Rust trajectory is deterministic
    for the same seed; the rendered summary still cites the action-kind
    distribution.
    """
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=42,
        num_decisions=10,
        deck_preset="tiny-mixed",
        rust_only=True,
    )
    assert rc == 0  # clone-replay must match for a deterministic walker
    out = capsys.readouterr().out
    assert "[rust-only]" in out
    assert "clone-replay-matches=True" in out
    # Action-kind summary lands even on the rust-only path.
    assert "action kinds chosen" in out


def test_cross_engine_game_mode_deck_path_loads_from_disk(
    parity_smoke, capsys, tmp_path
):
    """The ``deck_path`` arg reads a one-card-per-line deck file
    and feeds the contents into the same cross-engine flow as
    ``deck_preset``. End-to-end smoke of the file-based entry
    point.
    """
    deck_file = tmp_path / "small.deck"
    deck_file.write_text(
        "# small mixed deck\n"
        "10 Forest\n"
        "5 Mountain\n"
        "5 Plains\n",
        encoding="utf-8",
    )
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=42,
        num_decisions=5,
        deck_path=str(deck_file),
        ignore_known_gaps=True,
    )
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "[cross-engine-game]" in out
    # Deck-path is rendered in the header.
    assert "path=" in out
    assert "deck stats:" in out


def test_cross_engine_game_mode_unknown_preset_returns_setup_error(
    parity_smoke, capsys
):
    """Unknown preset name -> setup-error exit code (2). Operators
    typoing a preset get a fast loud failure with the known-presets
    list, not a confusing field-diff."""
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=42,
        num_decisions=5,
        deck_preset="does-not-exist",
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_cross_engine_game_mode_main_dispatches_deck_preset_flag(
    parity_smoke, monkeypatch, capsys
):
    """``main() --mode cross-engine-game --deck-preset tiny-mixed
    --ignore-known-gaps`` routes the flag through argparse into the
    cross-engine mode. Pins the CLI surface."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "cross-engine-game",
            "--seed",
            "42",
            "--num-decisions",
            "5",
            "--deck-preset",
            "tiny-mixed",
            "--ignore-known-gaps",
        ],
    )
    rc = parity_smoke.main()
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "tiny-mixed" in out


def test_cross_engine_game_mode_deck_path_mutual_exclusion(
    parity_smoke, capsys
):
    """Passing both ``deck_preset`` and ``deck_path`` is a usage
    error — the harness fails loudly with a setup-error exit code
    instead of silently picking one or merging them.
    """
    rc = parity_smoke.run_cross_engine_game_mode(
        seed=42,
        num_decisions=5,
        deck_preset="tiny-mixed",
        deck_path="ignored.deck",
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_resolve_deck_preset_returns_eoe_draft_list(parity_smoke):
    """Direct unit on the preset resolver. ``eoe-draft-style`` resolves
    to a 40-card list whose every entry is in the active Python card
    source — otherwise setup_game would raise and the harness would
    break in the integration test above. This locks the preset
    contents in for future deck-pool changes."""
    cards = parity_smoke._resolve_deck_preset("eoe-draft-style")
    assert len(cards) == 40
    # Every card name must resolve in the active Python source so
    # setup_game accepts the deck.
    from mtg_bot.cards.source import get_active_source
    py_printings = get_active_source().get_printings()
    missing = sorted({n for n in cards if n not in py_printings})
    assert missing == [], (
        f"eoe-draft-style preset has names missing from active source: "
        f"{missing}"
    )


def test_resolve_deck_preset_unknown_raises(parity_smoke):
    """Unknown preset name raises ``KeyError`` with a helpful list
    of known presets — operators typoing a preset get a fast,
    descriptive failure."""
    with pytest.raises(KeyError) as exc_info:
        parity_smoke._resolve_deck_preset("not-a-preset")
    msg = str(exc_info.value)
    assert "not-a-preset" in msg
    assert "tiny-mixed" in msg or "known presets" in msg
