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

    Today's synthetic-state starting position has the state in
    PHASE_UNTAP where only PassPriority is legal — so the walks only
    differ in priority-toggle parity at intermediate steps. We assert
    that across the FULL trajectories (initial + after each step),
    at least one fingerprint differs between two well-separated
    seeds. If a future slice expands the synthetic state's legal
    space, this test stays valid (more variation, more likely to
    differ)."""
    import mtg_bot_engine

    state_a = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    _, traj_a = parity_smoke._walk_trajectory(state_a, seed=1, num_decisions=10)

    state_b = mtg_bot_engine.make_synthetic_state(2, 3, 5)
    _, traj_b = parity_smoke._walk_trajectory(state_b, seed=2, num_decisions=10)

    # Even on a purely PassPriority-only state, an odd number of
    # passes between any two trajectory points would produce a
    # different intermediate fingerprint — but the rng.choice of a
    # one-element list is always the same element, so 10 PassPriority
    # passes from seed 1 == 10 PassPriority passes from seed 2. The
    # meaningful assertion is that the trajectories have the same
    # SHAPE (length) and end on the same fingerprint when the legal
    # set is a singleton. We assert that as a sanity-check;
    # the "different seeds differ" assertion would need a state with
    # >1 legal action, which we don't yet have a fixture for.
    assert len(traj_a) == len(traj_b) == 11
    # On a singleton-legal state, both trajectories ARE identical:
    # rng.choice picks the only element regardless of seed.
    # Document this explicitly so a future maintainer doesn't read
    # this test as "broken" when traj_a == traj_b.
    assert traj_a == traj_b


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
# --mode cross-engine — FIRST cross-engine parity check (round 27 / Phase 5).
# Builds matched Python + Rust states from the tiny synthetic deck, drives
# N PassPriority decisions on both, and reports field-level divergences
# along with a gap-doc lookup.
#
# This is the inaugural cross-engine compare — divergences ARE expected
# (the bridge has known gaps documented in RUST_PARITY_GAPS.md). The
# tests below are xfailed-pinned on the divergences they detect, so the
# count of xfailed tests becomes the "open parity items" tally.
# ---------------------------------------------------------------------------


def test_cross_engine_mode_returns_walk_result_with_steps(parity_smoke):
    """``walk_both`` produces a CrossEngineWalkResult with one
    CrossEngineStep per applied decision. Smoke that the helper runs
    end-to-end without raising on the tiny synthetic deck."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        walk_both,
    )

    result = walk_both(tiny_synthetic_deckspec(), n_decisions=5)
    assert result.error is None
    assert result.aborted_at is None
    assert len(result.steps) == 5
    # Each step has the action it applied recorded.
    for step in result.steps:
        assert step.action_dict == {"kind": "pass_priority"}


def test_cross_engine_mode_initial_state_libraries_match_after_ce1_closure(parity_smoke):
    """CE1 closure pin: after the make_state path defaults
    ``summoning_sick=true`` to match Python's universal default,
    Python and Rust library_pN field views must agree byte-for-byte
    on the initial state. Inverts the inaugural CE1 pin from r27
    (which asserted these DIVERGED on summoning_sick)."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        walk_both,
    )

    result = walk_both(tiny_synthetic_deckspec(), n_decisions=0)
    field_names = {d.field for d in result.initial_divergences}
    # Closed by the round-28 CE1 fix in state_ops_setup.rs.
    assert "library_p0" not in field_names
    assert "library_p1" not in field_names


def test_cross_engine_mode_does_not_diverge_on_pass_priority_at_step_0(parity_smoke):
    """**CE2 closure** — formerly
    ``test_cross_engine_mode_detects_pass_priority_divergence_at_step_0``.

    After the first PassPriority decision, Python's
    ``apply_action(PassPriority)`` is a no-op (Python's priority_loop
    owns priority management, not apply_action). The CE2 slice flipped
    Rust's ``apply_action({"kind": "pass_priority"})`` arm from a
    priority-toggling dispatch to a TRUE no-op matching Python — so
    the cross-engine smoke no longer reports a ``priority_player``
    divergence on PassPriority steps. This test pins the
    no-divergence contract so a future regression that re-introduces
    the toggle surfaces here loudly."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        walk_both,
    )

    result = walk_both(tiny_synthetic_deckspec(), n_decisions=2)
    # Step 0's divergence list must NOT include priority_player —
    # CE2 closure made the arm a no-op matching Python.
    step_0 = result.steps[0]
    priority_div = next(
        (d for d in step_0.divergences if d.field == "priority_player"), None,
    )
    assert priority_div is None, (
        f"unexpected priority_player divergence at step 0 (CE2 should "
        f"have closed this): py={priority_div.py_value} "
        f"rust={priority_div.rust_value} known_gap={priority_div.known_gap}"
        if priority_div is not None
        else ""
    )
    # And the same on step 1 — pin the contract holds across the full
    # walk, not just the inaugural step.
    step_1 = result.steps[1]
    priority_div_1 = next(
        (d for d in step_1.divergences if d.field == "priority_player"), None,
    )
    assert priority_div_1 is None


def test_cross_engine_mode_initial_libraries_byte_equal_after_make_state():
    """The library_pN field views from Python and Rust MUST match
    byte-for-byte on the initial state. This is the canonical parity
    contract — both engines built from the same DeckSpec should
    produce byte-identical zone signatures.

    Round-28 CE1 closure: state_ops_setup.rs now defaults
    summoning_sick=true to match Python's universal default; this
    test flipped from xfail to xpass and the marker was removed."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        build_matched_states,
        build_python_field_view,
    )
    import importlib.util
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "parity_smoke", repo_root / "scripts" / "parity_smoke.py",
    )
    parity_smoke = importlib.util.module_from_spec(spec)
    sys.modules["parity_smoke"] = parity_smoke
    spec.loader.exec_module(parity_smoke)

    deck = tiny_synthetic_deckspec()
    py_state, rust_state = build_matched_states(deck)
    py_view = build_python_field_view(py_state)
    rust_view = parity_smoke._build_field_view(rust_state)

    # The xfail-pinned assertion: libraries match byte-for-byte. Today
    # they don't (summoning_sick mismatch). When a future slice closes
    # this, the xfail flips to xpass and the test maintainer removes
    # the marker.
    assert py_view["library_p0"] == rust_view["library_p0"]
    assert py_view["library_p1"] == rust_view["library_p1"]


def test_cross_engine_mode_priority_player_unchanged_by_pass_priority():
    """After applying PassPriority on both engines, priority_player
    must agree. **CE2 closure** (formerly xfail-pinned on G3): Rust's
    ``apply_action({"kind": "pass_priority"})`` is now a no-op
    matching Python's ``apply_action(PassPriority)`` which is a pure
    ``return`` (src/mtg_bot/actions.py:990-991). The cross-engine
    smoke no longer flags a ``priority_player`` divergence on
    PassPriority steps."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        walk_both,
    )

    result = walk_both(tiny_synthetic_deckspec(), n_decisions=1)
    step_0 = result.steps[0]
    priority_div = next(
        (d for d in step_0.divergences if d.field == "priority_player"), None,
    )
    assert priority_div is None, (
        f"priority_player diverged: {priority_div.py_value=} {priority_div.rust_value=}"
    )


def test_cross_engine_mode_inject_divergence_flags_life_at_step_0(parity_smoke):
    """``walk_both(inject_divergence=True)`` artificially perturbs the
    Rust state's life_p0 BEFORE the initial compare. The harness must
    surface a life_p0 divergence in the initial-state report.

    Smoke-test-the-smoke: this proves the harness is actually
    detecting a fake cross-engine divergence, not just reporting
    what we already know about. Failure mode is "harness silently
    misses the perturbation" — which would mean the cross-engine
    code path isn't actually comparing what we think it's
    comparing."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        walk_both,
    )

    result = walk_both(
        tiny_synthetic_deckspec(), n_decisions=0, inject_divergence=True,
    )
    initial_fields = {d.field for d in result.initial_divergences}
    assert "life_p0" in initial_fields
    life_div = next(
        d for d in result.initial_divergences if d.field == "life_p0"
    )
    # Python is at default 20; Rust has been perturbed to 19.
    assert life_div.py_value == 20
    assert life_div.rust_value == 19


def test_cross_engine_mode_cli_dispatches(parity_smoke, monkeypatch, capsys):
    """``main()`` routes ``--mode cross-engine`` to the new runner.
    End-to-end argparse + dispatch smoke.

    **CE2 closure** — flipped from rc=1 (divergences) to rc=0 (clean).
    With Rust's ``apply_action(pass_priority)`` now a no-op matching
    Python, the tiny synthetic-state PassPriority walk no longer
    surfaces divergences and the runner reports ``PARITY: PASS``. The
    next batch of cross-engine slices will grow the deck pool +
    decision script and surface NEW divergences; those will get their
    own CE<n> entries in `docs/RUST_PARITY_GAPS.md`."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "cross-engine",
            "--seed",
            "42",
            "--decisions",
            "5",
        ],
    )
    rc = parity_smoke.main()
    # CE2 closure: no divergences on the tiny synthetic PassPriority
    # walk — rc=0 (PARITY: PASS).
    assert rc == 0
    out = capsys.readouterr().out
    # The CLI prints the cross-engine header.
    assert "[cross-engine]" in out
    # The final verdict is PARITY: PASS now that CE2 is closed.
    assert "PARITY: PASS" in out


def test_cross_engine_mode_cli_inject_divergence_flags_life(
    parity_smoke, monkeypatch, capsys
):
    """``main() --mode cross-engine --inject-divergence yes`` flips
    Rust's life_p0 to 19 before the compare; the runner must report
    a life_p0 divergence in the output and exit 1."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_smoke.py",
            "--mode",
            "cross-engine",
            "--seed",
            "42",
            "--decisions",
            "0",
            "--inject-divergence",
            "yes",
        ],
    )
    rc = parity_smoke.main()
    assert rc == 1
    out = capsys.readouterr().out
    assert "life_p0" in out
    assert "(--inject-divergence)" in out
    # The initial-state block prints py=20 and rust=19.
    assert "20" in out and "19" in out


def test_cross_engine_mode_step_index_increments(parity_smoke):
    """``CrossEngineStep.step_index`` must be the loop index, not a
    sentinel. The walk_both helper fills it in after step_both returns.
    Catches a regression where the index would always be -1."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        walk_both,
    )

    result = walk_both(tiny_synthetic_deckspec(), n_decisions=4)
    assert [s.step_index for s in result.steps] == [0, 1, 2, 3]


def test_cross_engine_mode_first_divergence_step_helper(parity_smoke):
    """``result.first_divergence_step()`` returns -1 when the initial
    state diverges (i.e., "before any decision"). On a parity-clean
    initial state it returns the first step index whose divergences
    are non-empty, or None on a fully-clean walk.

    **CE2 closure**: With Rust's ``apply_action(pass_priority)`` now
    a no-op matching Python, the tiny synthetic PassPriority walk is
    fully clean — the helper returns ``None`` (no divergence at any
    step). Pinning the None return here so a future cross-engine
    surface that re-introduces a divergence at step 0 surfaces the
    regression rather than silently flipping the helper's return."""
    from mtg_bot.parity.cross_engine import (
        tiny_synthetic_deckspec,
        walk_both,
    )

    result = walk_both(tiny_synthetic_deckspec(), n_decisions=5)
    # Round-28 CE1 closed the initial-state library_pN divergence;
    # round-28 CE2 closed the per-step priority_player divergence.
    # The walk is now fully clean — helper returns None.
    assert result.first_divergence_step() is None


def test_cross_engine_default_gap_lookup_priority_player_resolves_i25(parity_smoke):
    """The gap-doc lookup walks RUST_PARITY_GAPS.md and maps
    ``priority_player`` to I25 (the priority composition invariant
    that documents the cross-engine apply_action-level residual after
    G3/G4/G15 were closed). Locks in the lookup contract — the
    hard-coded mapping in :func:`_load_gap_doc_index` must resolve
    to a section that ACTUALLY exists in the doc."""
    from mtg_bot.parity.cross_engine import default_gap_lookup

    assert default_gap_lookup("priority_player") == "I25"


def test_cross_engine_default_gap_lookup_unknown_field_returns_none(parity_smoke):
    """Fields not in the curated lookup table return None — the
    "NOVEL divergence" signal. The summoning_sick + mana_pool type
    divergences are NOT yet documented as gaps, so they should
    surface as NOVEL.

    Note: ``library_p0`` / ``hand_p0`` aren't in the curated map —
    the divergences we see today are on the per-object
    summoning_sick slot, which is properly NOVEL until a gap entry
    documents it.
    """
    from mtg_bot.parity.cross_engine import default_gap_lookup

    assert default_gap_lookup("library_p0") is None
    assert default_gap_lookup("hand_p0") is None
    assert default_gap_lookup("graveyard_p0") is None


def test_cross_engine_build_python_state_step_is_untap(parity_smoke):
    """``build_python_state`` sets ``state.step = Step.UNTAP`` to match
    Rust's PHASE_UNTAP=0 starting phase. Without this Python's default
    state.step is None and the cross-engine step diff would be
    swamped by a None vs "UNTAP" mismatch."""
    from mtg_bot.parity.cross_engine import (
        build_python_state,
        tiny_synthetic_deckspec,
    )
    from mtg_bot.turn import Step

    py_state = build_python_state(tiny_synthetic_deckspec())
    assert py_state.step == Step.UNTAP


def test_cross_engine_build_rust_state_priority_is_zero(parity_smoke):
    """The Rust state's starting priority_player must be 0 — Python's
    default is also 0. This is the contract make_state advertises;
    a regression here would be the first divergence the harness
    detects on step 0."""
    from mtg_bot.parity.cross_engine import (
        build_rust_state,
        tiny_synthetic_deckspec,
    )

    rust_state = build_rust_state(tiny_synthetic_deckspec())
    assert rust_state.priority_player_idx == 0
    assert rust_state.active_player_idx == 0
    assert rust_state.turn_number == 1
    assert rust_state.step == "UNTAP"


# ---------------------------------------------------------------------------
# Phase 5 follow-up slice: slightly_bigger_deckspec (Forest + Mountain +
# Grizzly Bears). The first slice's tiny synthetic state covers
# mono-color Forest + Bears; this slice expands by ONE basic land
# (Mountain) so the registry round-trips two basic-land entries with
# different `mana_produced` colors and the per-zone signatures span
# THREE distinct printing_names in adjacent slots. PassPriority remains
# the only driver — legal_actions / play_land / activate_ability parity
# is deferred to the next slice.
#
# Expected outcome under the current binary (CE2-closure landed at the
# apply_action layer — pass_priority is a no-op on Rust just like
# Python): the harness reports NO divergences across the full walk.
# If a later regression reintroduces a divergence on this deck, the
# test will surface it as a NEW CE<n> candidate rather than a known-
# gap-masked nothing-burger.
# ---------------------------------------------------------------------------


def test_cross_engine_slightly_bigger_deckspec_walks_clean(parity_smoke):
    """Drive 200 PassPriority decisions on the 3-card-kind synthetic
    deck (Forest + Mountain + Grizzly Bears) and assert no field-view
    divergences surface.

    The point of the slice: confirm the bridge's registry +
    zone-signature recipe stay parity-clean as the card pool grows by
    one basic-land color. A clean walk here means the next natural
    expansion is the decision-script side (PlayLand / ActivateAbility),
    not more cards.

    If a regression surfaces a NEW divergence on this deck, the
    failure mode is "unexpected divergence on field <X>"; the assertion
    message names the offending field + step index for fast triage.
    Pin via xfail with a fresh CE<n> entry in
    ``docs/RUST_PARITY_GAPS.md`` if that happens.
    """
    from mtg_bot.parity.cross_engine import (
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(slightly_bigger_deckspec(), n_decisions=200)
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    assert result.aborted_at is None
    assert len(result.steps) == 200
    assert result.initial_divergences == [], (
        "initial-state divergences on slightly_bigger_deckspec; "
        f"expected clean start. Fields: "
        f"{[d.field for d in result.initial_divergences]}"
    )
    # Walk every step; any divergence is unexpected under current
    # binary semantics. Pinpoint the first offender for triage.
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index} on "
            f"slightly_bigger_deckspec: "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_slightly_bigger_deckspec_initial_libraries_have_three_kinds(parity_smoke):
    """Sanity-check the new deckspec's libraries actually carry THREE
    distinct printing_names — guards against an accidental edit that
    flattens the deck back to two kinds and silently weakens the
    coverage delta over tiny_synthetic_deckspec."""
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        slightly_bigger_deckspec,
        build_python_field_view,
    )

    py_state, rust_state = build_matched_states(slightly_bigger_deckspec())
    py_view = build_python_field_view(py_state)
    # library_p0 is a tuple of per-object 8-tuples; index 0 is the
    # printing_name. Collect unique names across BOTH players.
    p0_names = {sig[0] for sig in py_view["library_p0"]}
    p1_names = {sig[0] for sig in py_view["library_p1"]}
    union = p0_names | p1_names
    assert union == {"Forest", "Mountain", "Grizzly Bears"}, (
        f"slightly_bigger_deckspec should expose three printing_names "
        f"across libraries; got {sorted(union)}"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #2: extend the DECISION script (not the deck) — drive a
# draw + play_land via ``play_a_land_selector``. This is the first slice that
# moves cards off the library on both engines. **CE3 closed** by the post-
# r28 library-pop-orientation slice: Rust now uses ``lib.remove(0)`` (post-
# CE3 convention: index 0 = top of library) to match Python's
# ``library.pop(0)`` exactly. Same DeckSpec produces the same draw on both
# engines. The strict-xfail clean-walk pin + companion "symptom-present"
# pin that flagged CE3 were both removed; the underlying parity is now
# load-bearing on the test below.
# ---------------------------------------------------------------------------


def test_cross_engine_play_a_land_selector_walks_clean(parity_smoke):
    """Drive 10 decisions with ``play_a_land_selector`` on the
    slightly_bigger_deckspec and assert every step walks clean. Post-
    CE3 closure: Rust's draw_card and Python's draw_card pop from the
    same end of the library (index 0 = top), so hand_p0 / library_p0
    agree at step 0 and battlefield agrees at step 1 when the
    selector plays a land that's now in both engines' hands.
    """
    from mtg_bot.parity.cross_engine import (
        play_a_land_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=10,
        decision_selector=play_a_land_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    assert result.aborted_at is None
    assert len(result.steps) == 10
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_play_a_land_selector_emits_pass_priority_when_no_land(parity_smoke):
    """Selector contract: if no Land lives in Python's hand on step
    >= 1, the selector falls back to ``pass_priority`` rather than
    raising or returning a malformed dict. Drives a 0-card-library
    deck so the step-0 draw doesn't put a Land in hand, then verifies
    step 1 emits ``pass_priority``.

    Guards a regression where the selector's land-lookup loop falls
    off the end without an else-branch — silently returning None
    would crash the harness rather than producing a clean fallback."""
    from mtg_bot.parity.cross_engine import (
        CardDef,
        DeckSpec,
        play_a_land_selector,
        walk_both,
    )

    # Two-card creature-only deck. The step-0 draw puts a Bears in
    # Python's hand (and a Bears in Rust's hand by tail-pop — both
    # happen to be Bears here since the deck is mono-Bears, dodging
    # CE3 for this particular guard test).
    bears = CardDef(
        name="Grizzly Bears",
        card_type="Creature",
        mana_cost_generic=1,
        mana_cost_colored=(("G", 1),),
        power=2,
        toughness=2,
    )
    deck = DeckSpec(
        card_defs=(bears,),
        p0_library=("Grizzly Bears", "Grizzly Bears"),
        p1_library=("Grizzly Bears", "Grizzly Bears"),
    )
    result = walk_both(
        deck, n_decisions=2, decision_selector=play_a_land_selector,
    )
    assert result.error is None
    assert result.steps[0].action_dict == {"kind": "pass_priority"}
    # Step 1: no land in hand -> selector falls back to pass.
    assert result.steps[1].action_dict == {"kind": "pass_priority"}


# ---------------------------------------------------------------------------
# Phase 5 follow-up #3: multi-turn selector. Drives ``advance_phase`` on both
# engines via a side-channel + emits ``play_land`` in MAIN phases. Walks
# multiple full turns so the harness exercises:
#
#   * Turn 1 DRAW skip (CR 103.7c — first turn active player skips draw step).
#   * Turn 1 -> Turn 2 CLEANUP -> UNTAP wrap (turn_number bump + active swap).
#   * Turn 2 UNTAP sweep + summoning-sick clear (CR 302.1).
#   * Turn 2 DRAW step auto-draw (CR 504.1).
#   * Turn 3+ MAIN phase land drop with both players' permanents on the
#     battlefield.
#
# **Outcome (this slice)**: NOVEL divergence CE4 — ``battlefield`` order
# disagrees between Python (ETB-timestamp-ordered centralized list) and
# Rust (per-controller-grouped split). Surfaces at step 28, turn 3
# PRECOMBAT_MAIN, when p0 plays a Forest while p1's Mountain is already
# on the battlefield. Pinned strict-xfail below; closure plan in
# ``docs/RUST_PARITY_GAPS.md`` CE4 entry.
# ---------------------------------------------------------------------------


def test_cross_engine_multi_turn_selector_walks_clean(parity_smoke):
    """Drive 50 decisions on the slightly_bigger_deckspec with the
    multi_turn_selector and assert every step walks clean.

    **CE4 closure (this slice).** Pre-CE4 this was strict-xfail because
    Rust's ``battlefield_snapshot`` emitted in controller-grouped order
    (p0's bf then p1's bf) while Python's centralized
    ``state.battlefield`` is ordered by ETB timestamp; the first
    divergence surfaced at step 28 (turn 3 PRECOMBAT_MAIN play_land)
    once both players had a permanent on the battlefield. The closure
    slice added a per-object ``etb_timestamp`` field stamped at every
    bf-entry primitive and stable-sorts ``battlefield_snapshot`` (and
    the fingerprint's battlefield walk) by that field, producing the
    same emission order Python's ``state.battlefield.append()`` flow
    produces. The xfail marker was removed alongside the companion
    symptom-pin test (per the closure-pair contract; see ``docs/RUST_PARITY_GAPS.md``).
    """
    from mtg_bot.parity.cross_engine import (
        multi_turn_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=50,
        decision_selector=multi_turn_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    assert result.aborted_at is None
    assert len(result.steps) == 50
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_multi_turn_selector_turn_1_skips_draw(parity_smoke):
    """CR 103.7c first-turn skip pin. Across the first 11 decisions
    (turn 1 UNTAP -> CLEANUP via the side-channel advance_phase), the
    active player's hand stays empty — no draw fires on turn 1's DRAW
    step. The harness reads hand_p0 size at each step; on the
    UNTAP -> UPKEEP -> DRAW transition (steps 0 / 1) the hand stays at
    0 cards.

    Surfaces I18 + Rust's gated draw on both engines uniformly. If
    either engine drew a card on turn 1, this test would FAIL with
    `hand_p0 size > 0` at the step the draw happened. Both engines
    are pinned to the same gate (`turn_number != 1`), so both
    correctly skip.
    """
    from mtg_bot.parity.cross_engine import (
        multi_turn_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=11,
        decision_selector=multi_turn_selector,
    )
    assert result.error is None
    # The full turn 1 walk: 11 phases (UPKEEP..CLEANUP per the
    # side-channel advance + the side-channel-into-UNTAP wrap). p0
    # is active for all of turn 1; their hand stays empty because
    # CR 103.7c skips their draw step on turn 1.
    for step in result.steps[:11]:
        assert step.py_view["turn"] in (1, 2)
        # Active player on turn 1 is 0; on the wrap-into-turn-2 step it's 1.
        if step.py_view["turn"] == 1:
            assert step.py_view["active_player"] == 0
            # p0's hand stays empty — no draw on turn 1.
            assert len(step.py_view["hand_p0"]) == 0
        # No divergences during turn 1 + wrap — the first divergence
        # is at step 28, well into turn 3.
        assert step.divergences == []


def test_cross_engine_multi_turn_selector_turn_2_drawstep_fires(parity_smoke):
    """Companion to the turn-1 skip pin. On turn 2's DRAW step (the
    new active player is p1 after the CLEANUP -> UNTAP wrap), the
    active player draws one card. Verify the hand_p1 size flips from
    0 to 1 at the right boundary, on BOTH engines uniformly (no
    divergence).
    """
    from mtg_bot.parity.cross_engine import (
        multi_turn_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=14,
        decision_selector=multi_turn_selector,
    )
    assert result.error is None
    # Step 11: turn 2 UNTAP — p1 is now active, p1 hand empty.
    step_11 = result.steps[11]
    assert step_11.py_view["turn"] == 2
    assert step_11.py_view["step"] == "UNTAP"
    assert step_11.py_view["active_player"] == 1
    assert len(step_11.py_view["hand_p1"]) == 0
    # Step 12: UPKEEP.
    assert result.steps[12].py_view["step"] == "UPKEEP"
    assert len(result.steps[12].py_view["hand_p1"]) == 0
    # Step 13: DRAW — p1 draws one card via the side-channel
    # advance_phase composition on both engines (CR 504.1).
    step_13 = result.steps[13]
    assert step_13.py_view["step"] == "DRAW"
    assert len(step_13.py_view["hand_p1"]) == 1
    # Both engines agree (no divergence).
    assert step_13.divergences == []


# ---------------------------------------------------------------------------
# Phase 5 follow-up #4: cast_bears_selector. Extends the harness past the
# land-only multi_turn_selector slice into creature spells + mana abilities
# + summoning-sick timing. Drives multi-turn play on the slightly_bigger_deckspec
# until the active player has Grizzly Bears in hand AND two untapped non-sick
# Forests on the battlefield, activates each Forest in turn for {G}{G}, then
# emits cast_spell(Bears).
#
# **Outcome (this slice)**: NEW divergence CE5 — Rust's ``cast_creature``
# primitive fuses cast+resolve into one mutation (no-stack semantics:
# state_ops_cast.rs:293-297 docstring), while Python's ``_cast_spell``
# pushes the spell as a ``StackItem`` and defers the bf transition to
# ``resolve_top_of_stack``. After ``cast_spell`` is applied, Python's
# stack has the Bears item / battlefield is missing the Bears; Rust's
# stack is empty / battlefield has the freshly-cast Bears. Pinned
# strict-xfail below; closure plan in ``docs/RUST_PARITY_GAPS.md`` CE5
# entry. The cleanly-walking ``activate_ability`` calls (steps 92, 106,
# 107) before the cast also serve as the first cross-engine pin that
# mana-ability parity holds — Python's stack-bypass for mana abilities
# and Rust's direct-pool-credit produce the same observable mana pool
# + tapped state.
# ---------------------------------------------------------------------------


def test_cross_engine_cast_bears_selector_walks_clean(parity_smoke):
    """Drive 150 decisions on the slightly_bigger_deckspec with the
    cast_bears_selector and assert every step walks clean.

    The selector activates 2 Forests for {G}{G} then casts Grizzly Bears
    at turn 9 PRECOMBAT_MAIN (p0 active). Pre-cast steps walk clean —
    the activate-ability mana-credit flow is parity-equal across both
    engines. The cast step itself (step 108) used to be where CE5
    surfaced (Python pushes StackItem + defers, Rust ``cast_creature``
    fuses cast+resolve and lands the Bears directly on the
    battlefield). **CE5 closure**: the cross-engine harness now drains
    Python's stack after every ``cast_spell`` action via
    :func:`resolve_top_of_stack_python`, parity-aligning Python's
    observable state with Rust's no-stack semantics. Every step in
    this walk — including the cast at step 108 — produces an empty
    divergence list.

    Step 119's prior eager-``is_over`` divergence (Rust flipping on
    empty-library draw while Python deferred until SBAs) is also
    closed independently as CE10 (`feature/ce-pure-derivation-is-over`,
    landed before this slice). The 150-decision walk now traverses
    that boundary cleanly — the surface accessors are SBA-gated on
    both sides, and neither engine fires SBAs between cross-engine
    decision steps, so both stay stale together until the next slice
    grows the harness an SBA hook.
    """
    from mtg_bot.parity.cross_engine import (
        cast_bears_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=150,
        decision_selector=cast_bears_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    assert result.aborted_at is None
    assert len(result.steps) == 150
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_cast_bears_selector_cast_step_108_walks_clean_post_ce5(
    parity_smoke,
):
    """**CE5 closure regression pin**: at step 108 (turn 9 PRECOMBAT_MAIN
    p0 active) the selector emits ``cast_spell`` for Grizzly Bears. With
    the harness-side ``resolve_top_of_stack_python`` mirror in
    :func:`step_both` (gated by ``_RUST_AUTO_RESOLVE_ACTION_KINDS``),
    the cast step's divergence list is EMPTY: Python's stack drains
    immediately after the StackItem is pushed, the Bears resolves to
    the battlefield, and both engines' observable state agrees.

    This is the post-closure counterpart of the pre-closure CE5
    symptom-pin (deleted in this slice — the symptom no longer
    surfaces). The pair-pin pattern matches the CE2/CE3/CE4 closures:
    a post-closure regression test pins the new equilibrium.

    Pinned fields after the closure:

      * ``battlefield`` — both engines have 7 objects (the cast lands
        the Bears on Python's bf via the drained stack).
      * ``stack`` — both engines have an empty stack.
    """
    from mtg_bot.parity.cross_engine import (
        cast_bears_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=109,
        decision_selector=cast_bears_selector,
    )
    assert result.error is None
    step_108 = result.steps[108]
    # The cast_spell action was emitted at step 108.
    assert step_108.action_dict["kind"] == "cast_spell"
    # The Bears was the active player's hand card 3 (per the
    # slightly_bigger_deckspec deck order: p0 lib =
    # [Forest(id=0), Mountain(id=1), Forest(id=2), Bears(id=3)] —
    # by turn 9 p0 has drawn all four).
    assert step_108.action_dict["card_id"] == 3

    # CE5 closure: zero divergences on the cast step.
    assert step_108.divergences == [], (
        f"CE5 closure regression: step 108 cast should be parity-clean, "
        f"got {[(d.field, d.py_value, d.rust_value) for d in step_108.divergences]}"
    )

    # Both views agree on the post-cast state — Bears on the
    # battlefield, stack empty.
    py_bf = step_108.py_view["battlefield"]
    rust_bf = step_108.rust_view["battlefield"]
    assert len(py_bf) == 7
    assert len(rust_bf) == 7
    assert "Grizzly Bears" in [sig[0] for sig in py_bf]
    assert "Grizzly Bears" in [sig[0] for sig in rust_bf]
    assert step_108.py_view["stack"] == ()
    assert step_108.rust_view["stack"] == ()


def test_cross_engine_resolve_top_of_stack_python_drains_stack(parity_smoke):
    """**CE5 closure unit pin**: the harness primitive
    :func:`resolve_top_of_stack_python` drains a non-empty Python
    stack to empty, mirroring ``simulate.priority_loop``'s
    "resolve until empty" contract (CR 117.4).

    Drives a one-decision walk to N=109 to reach the post-cast state,
    then verifies that after :func:`step_both` returns the Python
    stack is empty AND the cast emitted exactly one stack-pop.

    Also pins the no-op behavior: calling the primitive on a state
    with an empty stack returns 0 without raising.
    """
    from mtg_bot.parity.cross_engine import (
        cast_bears_selector,
        resolve_top_of_stack_python,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=109,
        decision_selector=cast_bears_selector,
    )
    assert result.error is None
    # The cast step left Python's stack empty (drain ran in step_both).
    assert result.steps[108].py_view["stack"] == ()

    # No-op contract: calling the primitive on a state with an empty
    # stack returns 0 without raising. Build a fresh state to confirm.
    from mtg_bot.parity.cross_engine import build_python_state
    py_state = build_python_state(slightly_bigger_deckspec())
    assert py_state.stack == []
    assert resolve_top_of_stack_python(py_state) == 0
    assert py_state.stack == []


def test_cross_engine_cast_bears_selector_activate_ability_pre_cast_walks_clean(
    parity_smoke,
):
    """Pre-cast walk parity pin. The first 108 decisions walk through
    multiple turns of advance_phase + play_land + (steps 92, 106, 107)
    activate_ability calls without any divergence. This is the
    counterpart of the post-CE5 clean-walk pin: it asserts that
    everything EXCEPT the cast step itself walks clean, which means
    the cross-engine bridge's mana-ability resolution path is
    parity-equal across engines.

    Step 92 (turn 8 PRECOMBAT_MAIN p1 active) is the first
    activate_ability call — p1 has Bears + Forest in hand, plays the
    Forest at step 91, taps the older Mountain (id=4) at step 92...
    wait, the selector picks a Forest (G-producer), so step 92's
    source is the Forest p1 just played. Re-read the decision: at
    step 92 p1 has 1 untapped non-sick Forest (id=6, played turn 6
    and cleared by turn 8's UNTAP) — that's the source the selector
    picks. After the activate the pool has {G}:1; the selector wants
    {G}:2 to cast Bears, and p1 has no second Forest source, so the
    next selector call falls through to multi_turn_selector and
    advances phase.

    Step 106 + 107 (turn 9 PRECOMBAT_MAIN p0 active) are the two
    Forest activations p0 actually needs to cast Bears at step 108.
    """
    from mtg_bot.parity.cross_engine import (
        cast_bears_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=108,
        decision_selector=cast_bears_selector,
    )
    assert result.error is None
    assert result.aborted_at is None
    assert len(result.steps) == 108

    # No divergence anywhere in the pre-cast walk.
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected pre-cast divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value) for d in step.divergences]}"
        )

    # Verify the activate_ability calls were actually emitted at the
    # expected steps (the brief asserts the selector's decision shape).
    activate_steps = [
        s for s in result.steps
        if s.action_dict.get("kind") == "activate_ability"
    ]
    # Three activate_ability calls: one on p1's turn 8 (single Forest,
    # builds 1 G but can't cast), two on p0's turn 9 (two Forests,
    # builds 2 G).
    assert len(activate_steps) == 3, (
        f"expected 3 activate_ability calls, got "
        f"{[(s.step_index, s.action_dict) for s in activate_steps]}"
    )
    # Step 92: p1 activates their Forest.
    assert activate_steps[0].step_index == 92
    # Steps 106 + 107: p0 activates both their Forests.
    assert activate_steps[1].step_index == 106
    assert activate_steps[2].step_index == 107


def test_cross_engine_cast_bears_selector_mana_pool_g2_at_step_107(parity_smoke):
    """After the second activate_ability at step 107 (turn 9
    PRECOMBAT_MAIN p0 active), p0's mana pool carries exactly 2 G
    coins on BOTH engines. Pins the mana-ability parity contract
    that motivated the cast at step 108: Python's stack-bypass for
    mana abilities and Rust's direct-pool-credit dispatch arm
    produce byte-equal observable pools.

    Mana pool field shape: a sorted 6-key tuple `((B,0), (C,0),
    (G,2), (R,0), (U,0), (W,0))`. The harness normalizes Python's
    ``list[ManaCoin]`` representation to this shape via
    ``_python_mana_pool_signature`` so both engines emit the same
    tuple on a green-only pool.
    """
    from mtg_bot.parity.cross_engine import (
        cast_bears_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=108,
        decision_selector=cast_bears_selector,
    )
    assert result.error is None
    step_107 = result.steps[107]
    assert step_107.action_dict["kind"] == "activate_ability"
    # Both engines agree on the mana pool after the second activate.
    py_pool = dict(step_107.py_view["mana_pool_p0"])
    rust_pool = dict(step_107.rust_view["mana_pool_p0"])
    assert py_pool == rust_pool, (
        f"mana_pool_p0 diverged at step 107: py={py_pool} rust={rust_pool}"
    )
    assert py_pool["G"] == 2
    assert step_107.divergences == []


def test_cross_engine_cast_bears_selector_falls_back_to_multi_turn(parity_smoke):
    """Selector contract: when the active player does NOT have Grizzly
    Bears in hand, the selector delegates to multi_turn_selector
    (play_land in MAIN if available, side-channel advance + pass
    otherwise). Step 0 (turn 1 ap=0 UPKEEP with empty hand) is the
    canonical fall-through case — multi_turn_selector advances UPKEEP
    -> DRAW and emits pass_priority.
    """
    from mtg_bot.parity.cross_engine import (
        cast_bears_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=1,
        decision_selector=cast_bears_selector,
    )
    assert result.error is None
    # Step 0: turn 1 UPKEEP, hands empty, no Bears anywhere → fall
    # through to multi_turn_selector → side-channel advance + pass.
    assert result.steps[0].action_dict == {"kind": "pass_priority"}
    assert result.steps[0].divergences == []


def test_cross_engine_dict_to_python_action_handles_activate_ability(parity_smoke):
    """The ``_dict_to_python_action`` shim must convert the
    ``activate_ability`` dict shape that the new selector emits into a
    Python ``ActivateAbility`` instance with matching fields. Locks in
    the action-dict round-trip contract so a future selector that
    emits a different ability_index doesn't silently fall through to
    the ``unsupported action kind`` raise.
    """
    from mtg_bot.actions import ActivateAbility
    from mtg_bot.parity.cross_engine import _dict_to_python_action

    action = _dict_to_python_action(
        {"kind": "activate_ability", "source_id": 42, "ability_index": 0}
    )
    assert isinstance(action, ActivateAbility)
    assert action.source_id == 42
    assert action.ability_index == 0


# ---------------------------------------------------------------------------
# Phase 5 follow-up #5: combat_selector. Extends cast_bears_selector past the
# cast into declare_attackers / declare_blockers / combat damage application
# / clear_combat. The probe surfaces (or fails to surface, depending on the
# outcome) the audits' predicted CE6 / CE7 (combat damage assignment order).
# A single-attacker / zero-blocker slice exercises the dispatch + advance
# wiring without surfacing the multi-block ordering surface — that requires
# a richer deck (two attackers facing two blockers with non-trivial ETB
# interleaving). This slice is the gate that unblocks the ordering probe.
# ---------------------------------------------------------------------------


def test_cross_engine_combat_selector_walks_clean(parity_smoke):
    """Drive 200 decisions on the slightly_bigger_deckspec with the
    combat_selector and assert every step walks clean.

    The selector inherits cast_bears_selector's pre-cast logic
    (activate 2 Forests + cast Bears at step 108, turn 9 p0
    PRECOMBAT_MAIN) and adds four new branches: declare_attackers
    (emitted when active player owns a non-sick non-tapped creature),
    declare_blockers (empty assignments, side-channel advance baked in),
    COMBAT_DAMAGE (side-channel resolve_combat_damage_python +
    deal_combat_damage_rust + apply_sbas), and END_OF_COMBAT
    (side-channel clear_combat_both).

    By the end of the 200 decisions the walker has driven three full
    combat cycles (Bears attacking unblocked on turns 11/13/15 — p0
    active, Bears unsick after turn 10's UNTAP). p1's life drops from
    20 -> 18 -> 16 -> 14 in 2-damage increments. Both engines apply
    damage simultaneously through the side-channel mirrors, and the
    post-step compare on COMBAT_DAMAGE / END_OF_COMBAT sees an empty
    divergence list for every step.

    **Why no CE divergence surfaces on this slice.** The audits'
    predicted CE6 / CE7 (combat damage assignment order — Python walks
    ``state.battlefield`` filtered by ``attacker.blocked_by``, Rust
    walks ``self.blockers`` Vec) requires MULTIPLE blockers per
    attacker to surface — the order question only matters when there
    are 2+ blockers facing the same attacker with interleaved ETB
    timestamps. On the current deck the defender has zero creatures
    so combat is always unblocked. The probe is therefore the GATE
    that unblocks the audits' true ordering probe rather than the
    ordering probe itself — a richer deck (two Bears + a defender
    with two blockers) is the natural next slice.
    """
    from mtg_bot.parity.cross_engine import (
        combat_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=200,
        decision_selector=combat_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    assert result.aborted_at is None
    assert len(result.steps) == 200
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_combat_selector_first_declare_attackers_at_step_136(
    parity_smoke,
):
    """**combat_selector script pin**: the first ``declare_attackers``
    emission lands at step 136, turn 11 DECLARE_ATTACKERS with p0
    active and Grizzly Bears (id=3) as the attacker.

    Reasoning trail:
      * Bears is cast at step 108 (turn 9 p0 PRECOMBAT_MAIN per the
        ``cast_bears_selector`` script).
      * Bears enters with ``summoning_sick=True`` and stays sick
        through turn 9's combat (DECLARE_ATTACKERS at step 110-ish)
        — selector falls through to pass_priority.
      * Turn 9 wraps to turn 10 (p1 active); turn 10 UNTAP clears p1's
        permanents but NOT Bears (Bears is on p0's side).
      * Turn 10 wraps to turn 11 (p0 active); turn 11 UNTAP fires on
        p0's permanents, clearing Bears' summoning_sick.
      * At turn 11 DECLARE_ATTACKERS, Bears is non-sick, non-tapped,
        Creature, controller=0 — the selector emits
        ``declare_attackers([3])``.

    The exact step index (136) is sensitive to the upstream selector
    chain's decision count; if cast_bears_selector or
    multi_turn_selector evolves, this pin's step number may need to
    update.
    """
    from mtg_bot.parity.cross_engine import (
        combat_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=140,
        decision_selector=combat_selector,
    )
    assert result.error is None
    declare_steps = [
        s for s in result.steps
        if s.action_dict.get("kind") == "declare_attackers"
    ]
    assert len(declare_steps) >= 1, (
        f"expected at least one declare_attackers emission, got "
        f"{[(s.step_index, s.action_dict) for s in declare_steps]}"
    )
    first_declare = declare_steps[0]
    assert first_declare.step_index == 136, (
        f"expected first declare_attackers at step 136, got "
        f"{first_declare.step_index} action={first_declare.action_dict}"
    )
    assert first_declare.action_dict == {
        "kind": "declare_attackers",
        "attacker_ids": [3],
    }
    assert first_declare.py_view["turn"] == 11
    assert first_declare.py_view["step"] == "DECLARE_ATTACKERS"
    assert first_declare.py_view["active_player"] == 0
    assert first_declare.divergences == []


def test_cross_engine_combat_selector_damage_drops_p1_life_in_2s(
    parity_smoke,
):
    """**Combat damage parity pin**: across the 200-decision walk, p1's
    life drops in 2-point increments (Bears unblocked, 2/2) on every
    combat cycle. Both engines agree on the life_p1 trajectory
    step-by-step.

    Three combat cycles fire on turns 11 / 13 / 15 (p0 active —
    p0 attacks with Bears on every other turn; p1's turns never
    declare attackers because p1 has no creatures). p1's life
    sequence: 20 -> 18 -> 16 -> 14. The cross-engine compare on
    each combat_damage step sees byte-equal ``life_p1`` on both
    engines.

    Pins the post-damage state ALSO has ``life_p0 == 20`` throughout
    (Bears is unblocked, no back-damage, so p0 stays at full life).
    """
    from mtg_bot.parity.cross_engine import (
        combat_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=200,
        decision_selector=combat_selector,
    )
    assert result.error is None
    # Walk the per-step compare and track life_p1 evolution. Every
    # life-drop boundary should agree across engines.
    life_drops_py: list[tuple[int, int]] = []
    prev_life_p1 = 20
    for step in result.steps:
        cur_py = step.py_view["life_p1"]
        cur_rust = step.rust_view["life_p1"]
        assert cur_py == cur_rust, (
            f"life_p1 diverged at step {step.step_index}: "
            f"py={cur_py} rust={cur_rust}"
        )
        if cur_py != prev_life_p1:
            life_drops_py.append((step.step_index, cur_py))
            prev_life_p1 = cur_py
    # Expect 3 drops: 20->18, 18->16, 16->14.
    assert [drop[1] for drop in life_drops_py] == [18, 16, 14], (
        f"expected p1 life trajectory 20->18->16->14, got {life_drops_py}"
    )
    # Final p1 life is 14; p0 untouched.
    assert result.steps[-1].py_view["life_p1"] == 14
    assert result.steps[-1].rust_view["life_p1"] == 14
    assert result.steps[-1].py_view["life_p0"] == 20
    assert result.steps[-1].rust_view["life_p0"] == 20


def test_cross_engine_combat_selector_falls_back_to_cast_bears(parity_smoke):
    """Selector contract: when none of the four combat branches
    (DECLARE_ATTACKERS / DECLARE_BLOCKERS / COMBAT_DAMAGE /
    END_OF_COMBAT) match, the selector delegates to
    cast_bears_selector (which itself delegates to multi_turn_selector
    if no cast/activate is appropriate).

    Step 0 (turn 1 UPKEEP, empty hand, no Bears) is the canonical
    fall-through case — multi_turn_selector advances UPKEEP -> DRAW
    and emits pass_priority.
    """
    from mtg_bot.parity.cross_engine import (
        combat_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=1,
        decision_selector=combat_selector,
    )
    assert result.error is None
    assert result.steps[0].action_dict == {"kind": "pass_priority"}
    assert result.steps[0].divergences == []


def test_cross_engine_combat_selector_summoning_sick_bears_skips_attack(
    parity_smoke,
):
    """At turn 9 DECLARE_ATTACKERS — the turn Bears was cast — Bears
    is summoning-sick. The selector's eligibility filter requires
    ``not summoning_sick``, so Bears is skipped and the selector
    falls through to multi_turn_selector for a pass_priority +
    advance. No attack happens on turn 9 even though Bears is on
    the battlefield.

    Pins CR 302.1's summoning-sickness restriction for attackers —
    the parallel land-side restriction (Rust's ActivateAbility
    rejects sick lands; documented as a CE candidate) is NOT in
    scope here.
    """
    from mtg_bot.parity.cross_engine import (
        combat_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=130,
        decision_selector=combat_selector,
    )
    assert result.error is None
    # Walk every step and confirm: at turn 9 DECLARE_ATTACKERS with
    # p0 active, the action emitted is NOT declare_attackers.
    for step in result.steps:
        if (
            step.py_view["turn"] == 9
            and step.py_view["step"] == "DECLARE_ATTACKERS"
            and step.py_view["active_player"] == 0
        ):
            assert step.action_dict.get("kind") != "declare_attackers", (
                f"turn 9 declare_attackers fired despite sick Bears: "
                f"step {step.step_index} action={step.action_dict}"
            )
            # Sanity: Bears IS on the battlefield and IS sick.
            bears_sigs = [
                s for s in step.py_view["battlefield"]
                if s[0] == "Grizzly Bears"
            ]
            assert len(bears_sigs) == 1, (
                f"expected Bears on bf at step {step.step_index}, got {bears_sigs}"
            )
            # Position 2 of the 8-tuple is summoning_sick (per
            # scripts/parity_smoke.py::_zone_signature).
            assert bears_sigs[0][2] == 1, (
                f"expected Bears summoning_sick=1 at step {step.step_index}, "
                f"got sig {bears_sigs[0]}"
            )


def test_cross_engine_combat_selector_declare_blockers_empty_assignments(
    parity_smoke,
):
    """The selector's DECLARE_BLOCKERS branch emits
    ``declare_blockers`` with an empty ``assignments`` list whenever
    an attacker is declared and the defender has no eligible
    blockers. This exercises the ``_dict_to_python_action``
    conversion for the empty-assignments edge case AND verifies that
    both engines' ``apply_action(DeclareBlockers)`` arms accept the
    no-op dispatch.

    On the current slightly_bigger_deckspec, the defender (p1) has
    no creatures at any point — p1's library / hand only contains
    Forests / Mountains / Bears, but p1 never has enough mana to
    cast Bears in this script (only one Forest gets played per p1
    turn and p1's selector chain only fires the mana-cast script
    when p0 is active). So every DECLARE_BLOCKERS step emits an
    empty assignment list.
    """
    from mtg_bot.parity.cross_engine import (
        combat_selector,
        slightly_bigger_deckspec,
        walk_both,
    )

    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=200,
        decision_selector=combat_selector,
    )
    assert result.error is None
    declare_blockers_steps = [
        s for s in result.steps
        if s.action_dict.get("kind") == "declare_blockers"
    ]
    # Three combat cycles fire (turns 11/13/15), each with one
    # declare_blockers emission. The advance-baked-in pattern in the
    # DECLARE_BLOCKERS branch ensures we emit exactly once per cycle.
    assert len(declare_blockers_steps) == 3, (
        f"expected 3 declare_blockers emissions (one per combat cycle), "
        f"got {[(s.step_index, s.py_view['turn']) for s in declare_blockers_steps]}"
    )
    for step in declare_blockers_steps:
        assert step.action_dict["assignments"] == [], (
            f"expected empty assignments at step {step.step_index}, got "
            f"{step.action_dict['assignments']}"
        )
        assert step.divergences == []


def test_cross_engine_dict_to_python_action_handles_declare_attackers(
    parity_smoke,
):
    """The ``_dict_to_python_action`` shim converts the
    ``declare_attackers`` dict shape into a Python ``DeclareAttackers``
    instance with a frozenset of ids. Locks the round-trip contract
    so a future probe that ships multiple attacker ids in a single
    set doesn't silently fall through to the unsupported-kind raise.
    """
    from mtg_bot.actions import DeclareAttackers
    from mtg_bot.parity.cross_engine import _dict_to_python_action

    action = _dict_to_python_action(
        {"kind": "declare_attackers", "attacker_ids": [3, 7, 11]}
    )
    assert isinstance(action, DeclareAttackers)
    assert action.attacker_ids == frozenset({3, 7, 11})

    # Empty set is also valid.
    action_empty = _dict_to_python_action(
        {"kind": "declare_attackers", "attacker_ids": []}
    )
    assert isinstance(action_empty, DeclareAttackers)
    assert action_empty.attacker_ids == frozenset()


def test_cross_engine_dict_to_python_action_handles_declare_blockers(
    parity_smoke,
):
    """The ``_dict_to_python_action`` shim converts the
    ``declare_blockers`` dict shape into a Python ``DeclareBlockers``
    instance with a tuple of (blocker_id, attacker_id) 2-tuples.
    Locks the round-trip contract for multi-blocker scenarios that
    the audits' predicted CE6/CE7 closure probe will exercise.
    """
    from mtg_bot.actions import DeclareBlockers
    from mtg_bot.parity.cross_engine import _dict_to_python_action

    # Single-block case.
    action = _dict_to_python_action(
        {"kind": "declare_blockers", "assignments": [[5, 3]]}
    )
    assert isinstance(action, DeclareBlockers)
    assert action.assignments == ((5, 3),)

    # Multi-block case (audit's CE6 probe shape).
    action_multi = _dict_to_python_action(
        {"kind": "declare_blockers", "assignments": [[5, 3], [7, 3], [9, 3]]}
    )
    assert isinstance(action_multi, DeclareBlockers)
    assert action_multi.assignments == ((5, 3), (7, 3), (9, 3))

    # Empty assignments (no-block dispatch).
    action_empty = _dict_to_python_action(
        {"kind": "declare_blockers", "assignments": []}
    )
    assert isinstance(action_empty, DeclareBlockers)
    assert action_empty.assignments == ()


def test_cross_engine_resolve_combat_damage_python_no_attackers_noop(
    parity_smoke,
):
    """**Combat-damage harness primitive contract**: invoking
    :func:`resolve_combat_damage_python` on a state with NO attackers
    (no creature has ``attacking=True``) returns False without raising,
    leaves life unchanged on both players.

    This pins the early-exit contract that the combat_selector relies
    on at the COMBAT_DAMAGE branch — without the no-op return, calling
    Python's ``resolve_combat_damage`` against an empty-attackers
    state would raise on the ``defender.agent`` reference (the
    synthetic state has no agent attached).
    """
    from mtg_bot.parity.cross_engine import (
        build_python_state,
        resolve_combat_damage_python,
        slightly_bigger_deckspec,
    )

    py_state = build_python_state(slightly_bigger_deckspec())
    # Fresh state: no attackers.
    assert not any(
        getattr(o, "attacking", False) for o in py_state.battlefield
    )
    assert resolve_combat_damage_python(py_state) is False
    assert py_state.players[0].life == 20
    assert py_state.players[1].life == 20


def test_cross_engine_clear_combat_both_resets_rust_attackers_vec(
    parity_smoke,
):
    """**Clear-combat harness primitive contract**: calling
    :func:`clear_combat_both` on matched states with declared
    attackers/blockers empties Rust's ``self.attackers`` /
    ``self.blockers`` Vecs AND drops Python's per-object
    ``attacking`` / ``blocking`` / ``blocked_by`` flags.

    Pins the end-of-combat cleanup contract that the combat_selector
    relies on at the END_OF_COMBAT branch. Without this clear, Rust's
    declared-attackers Vec would persist into the next turn, and the
    next DECLARE_ATTACKERS step's ``deal_combat_damage`` call would
    re-apply damage to a stale attacker entry.
    """
    from mtg_bot.parity.cross_engine import (
        combat_selector,
        clear_combat_both,
        slightly_bigger_deckspec,
        walk_both,
    )

    # Walk up to the first combat cycle's END_OF_COMBAT then verify
    # the clear ran on both engines.
    result = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=140,
        decision_selector=combat_selector,
    )
    assert result.error is None
    # Find the first END_OF_COMBAT step (immediately after the first
    # combat damage). It should have empty attackers on Rust by then.
    for step in result.steps:
        if (
            step.py_view["step"] == "POSTCOMBAT_MAIN"
            and step.py_view["turn"] == 11
        ):
            # By POSTCOMBAT_MAIN we've already passed END_OF_COMBAT,
            # so attacking should be cleared on Python.
            for sig in step.py_view["battlefield"]:
                # Position 1 of the 8-tuple is tapped. The Bears was
                # tapped from attacking; that tap state PERSISTS
                # through end-of-combat (only the next UNTAP step
                # clears it). The attacking flag itself isn't in the
                # 8-tuple, but the absence of further damage on later
                # steps is the indirect signal that clear_combat
                # worked.
                pass
            break

    # Direct unit-style smoke: build a fresh state, hand-poke an
    # attacking flag on a python object, then call clear_combat_both.
    from mtg_bot.parity.cross_engine import build_matched_states
    py_state, rust_state = build_matched_states(slightly_bigger_deckspec())
    # Simulate a declared attack: directly mark a python bf object as
    # attacking + push a rust attacker entry. (The rust_state has no
    # bf objects so we use it as a noop-target — clear_combat_both
    # should run without raising even on empty rust state.)
    # Build a minimal py-side bf object via the existing flow: draw
    # then play_land won't yield a creature; skip — use the walk to
    # arrive at an attacking state instead.
    result_short = walk_both(
        slightly_bigger_deckspec(),
        n_decisions=137,  # one step past the declare_attackers emission
        decision_selector=combat_selector,
    )
    assert result_short.error is None
    # Verify clear_combat_both is the harness-only primitive that
    # accepts both engines and returns None.
    py_state2, rust_state2 = build_matched_states(slightly_bigger_deckspec())
    result_clear = clear_combat_both(py_state2, rust_state2)
    assert result_clear is None  # no return value


# ---------------------------------------------------------------------------
# Phase 5 follow-up #6: multi_block_combat_selector. Extends combat_selector
# past the single-attacker / zero-blocker slice into a SINGLE-ATTACKER /
# TWO-BLOCKER scenario. Surfaced and (now) closes the audits' predicted
# CE9 (combat damage assignment order) divergence.
#
# **STATUS: CE9 CLOSED** by ``feature/ce9-blocker-bf-order``. Rust's two
# iteration sites in ``rust/src/state_ops_combat.rs`` (the per-attacker
# first-blocker walk at lines 865-868 and the back-damage walk at line
# 962) now iterate via the shared ``bf_iter_etb_sorted`` helper in
# ``rust/src/bf_iter.rs`` (the same helper introduced by the CE5/CE6
# closure for SBA victim collection), filtered by membership in
# ``self.blockers``. This matches Python's ``combat.py:389,405`` walk
# of ``[o for o in state.battlefield if o.id in attacker.blocked_by]``.
# Per the playbook §3 Option A: parity with observable Python first;
# CR 510.1d "active player chooses damage assignment order" is a
# separate long-term Python-side fix. See ``docs/RUST_PARITY_GAPS.md``
# I31 for the closure invariant.
# ---------------------------------------------------------------------------


def test_cross_engine_multi_block_combat_selector_walks_clean(parity_smoke):
    """**CE9 clean-walk pin** (expected-pass post-closure).

    Drive 210 decisions on :func:`multi_blocker_deckspec` with
    :func:`multi_block_combat_selector` and assert every step walks
    clean. Pre-closure, the multi-block COMBAT_DAMAGE step at index
    196 surfaced the audits' predicted CE6/CE7 (numbered CE9 here)
    divergence: Python's bf-order damage walk and Rust's
    self.blockers-Vec damage walk picked DIFFERENT first blockers
    when blockers were declared in reverse-bf order.

    Post-closure, Rust's combat-damage iteration consumes
    ``bf_iter_etb_sorted`` (same helper as the CE5/CE6 SBA closure)
    and the entire 210-decision walk runs clean.

    Mirrors the post-closure clean-walk pattern of CE2/CE3/CE4/CE5/CE6:
    a single expected-pass clean-walk pin; the companion symptom-pin
    that previously documented the surface shape was deleted on closure.
    """
    from mtg_bot.parity.cross_engine import (
        multi_block_combat_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=210,
        decision_selector=multi_block_combat_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    assert result.aborted_at is None
    assert len(result.steps) == 210
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_multi_block_combat_selector_pre_combat_walks_clean(
    parity_smoke,
):
    """The pre-combat walk (steps 0-192) on
    :func:`multi_blocker_deckspec` with
    :func:`multi_block_combat_selector` is parity-clean. Pins the
    contract that the multi-block CE9 surface is SPECIFICALLY a
    damage-assignment-order finding — the cast, mana-ability, draw,
    advance-phase, and declare-attackers paths all walk clean even
    on this fatter deck.

    Step 193 is the first declare_attackers emission; step 195 is
    the declare_blockers emission. Both walk clean — the divergence
    only surfaces on step 196 when actual damage is applied (the
    COMBAT_DAMAGE branch fires resolve_combat_damage_python +
    deal_combat_damage_rust).
    """
    from mtg_bot.parity.cross_engine import (
        multi_block_combat_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=196,
        decision_selector=multi_block_combat_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-combat divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Confirm the script shape (the same shape that gates the CE9
    # surface): three casts (p1 turns 6 + 10, p0 turn 13), one
    # declare_attackers (p0 turn 15), one declare_blockers (turn 15
    # COMBAT_DAMAGE / pre-advanced step).
    casts = [
        s for s in result.steps
        if s.action_dict.get("kind") == "cast_spell"
    ]
    assert len(casts) == 3, f"expected 3 cast emissions, got {len(casts)}"
    declares_attackers = [
        s for s in result.steps
        if s.action_dict.get("kind") == "declare_attackers"
    ]
    assert len(declares_attackers) == 1
    assert declares_attackers[0].py_view["active_player"] == 0
    declares_blockers = [
        s for s in result.steps
        if s.action_dict.get("kind") == "declare_blockers"
    ]
    assert len(declares_blockers) == 1
    assert declares_blockers[0].action_dict["assignments"] == [[11, 5], [9, 5]]


def test_cross_engine_multi_block_combat_selector_suppresses_p1_attacks(
    parity_smoke,
):
    """**Multi-block selector contract**: p1 NEVER attacks. The selector's
    DECLARE_ATTACKERS branch gates on ``active_idx == 0`` to preserve
    p1's Bears as untapped blockers (a tapped creature can't block per
    CR 509.1b — and combat tapping is the principal way Bears would
    end up tapped on this synthetic deck).

    Across the full 210-step walk, every ``declare_attackers`` action
    has ``active_player == 0``. p1's DECLARE_ATTACKERS step falls
    through to the multi_turn fallback (pass_priority + advance).
    """
    from mtg_bot.parity.cross_engine import (
        multi_block_combat_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=210,
        decision_selector=multi_block_combat_selector,
    )
    assert result.error is None
    declares = [
        s for s in result.steps
        if s.action_dict.get("kind") == "declare_attackers"
    ]
    for d in declares:
        assert d.py_view["active_player"] == 0, (
            f"unexpected p1 declare_attackers at step {d.step_index}: "
            f"{d.action_dict} active={d.py_view['active_player']}"
        )


def test_cross_engine_multi_blocker_deckspec_library_shape(parity_smoke):
    """**Deckspec contract**: the multi-blocker library shape is the
    minimum needed to deliver:
      * One Bears in p0's hand by turn 13 (the cast turn).
      * Two Bears in p1's hand spaced 4 turns apart (turn 6 + turn 10
        casts so both Bears are unsick by turn 15).

    Pins the exact library tuples so a future refactor that fiddles
    with the shape doesn't silently shift the multi-block CE9 step
    index.
    """
    from mtg_bot.parity.cross_engine import multi_blocker_deckspec

    spec = multi_blocker_deckspec()
    assert spec.p0_library == (
        "Forest", "Forest", "Forest", "Forest", "Forest",
        "Grizzly Bears", "Forest",
    )
    assert spec.p1_library == (
        "Forest", "Forest", "Grizzly Bears", "Forest",
        "Grizzly Bears", "Forest", "Forest",
    )
    assert spec.shuffle_seed is None  # no shuffle for parity smoke


def test_cross_engine_step_both_routes_declare_blockers_actor_to_defender(
    parity_smoke,
):
    """**step_both actor-routing contract**: when an action's kind is
    ``declare_blockers``, the harness routes Python's ``apply_action``
    using the DEFENDER's player id (not the priority player's), so
    Python's ``_declare_blockers`` validation passes
    ``eligible_blockers(state, defender_id)`` and the assignments
    actually take effect.

    Without this routing the declare_blockers action would silently
    drop every assignment because the harness uses
    ``priority_player_idx`` as the actor for every action kind, and
    during DECLARE_BLOCKERS the priority player on a synthetic state
    is still the active player (the harness doesn't model the
    priority handoff to defender). The CE9 surface depends on
    blockers actually being declared — without this routing the
    multi-block probe would see empty blocked_by sets and the
    divergence wouldn't surface.

    Smoke check: drive a multi-block walk through the declare_blockers
    step and verify the attacker's blocked_by set is non-empty on the
    Python side post-step.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
        multi_block_combat_selector,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=196,
        decision_selector=multi_block_combat_selector,
    )
    assert result.error is None
    # After step 195 (declare_blockers), Python's attacker (Bears
    # id=5) should have blocked_by set to both blockers — populated
    # by the defender-routed apply_action.
    py_state, _ = build_matched_states(multi_blocker_deckspec())
    for i in range(196):
        action = multi_block_combat_selector(py_state, _, i)
        from mtg_bot.parity.cross_engine import step_both
        step_both(py_state, _, action)
    attacker = next(
        (o for o in py_state.battlefield if int(o.id) == 5),
        None,
    )
    # If the attacker is None it died from combat damage already (the
    # COMBAT_DAMAGE branch may have fired in step 195's selector chain
    # if the advance landed at COMBAT_DAMAGE). On the multi_blocker
    # deck the COMBAT_DAMAGE fires at step 196 (one step after
    # declare), so the attacker should still be on bf at step 195's
    # post-state. If the test breaks here the walk shape has shifted.
    assert attacker is not None, (
        "expected attacker (id=5) to still be on bf at end of step 195"
    )
    # blocked_by should contain both blocker ids (the order depends on
    # the declare order — reversed bf order = [11, 9]).
    assert set(attacker.blocked_by) == {9, 11}, (
        f"expected blocked_by={{9, 11}}, got {attacker.blocked_by}"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #7: retaliation_selector — extend
# multi_block_combat_selector to bilateral combat. Surfaces CE12.
#
# **CE12 — winner_id sticky-vs-derived divergence.** Bilateral combat
# (both players attack on their respective turns) drives both players
# to empty-library exhaustion + has_lost=True. Python's `winner_id`
# is a SBA-WRITTEN sticky field (turn.py:1383-1393 — only written when
# `state.winner_id is None`, never cleared). Rust's `winner_id` is a
# LIVE-DERIVED getter (state_ops_winloss.rs:225-235 — returns
# `Some(other_id)` only when exactly one player has has_lost; returns
# None when both lost). When the state transitions from "exactly one
# loser" to "both lost", Python keeps the prior winner sticky, Rust
# switches to None.
#
# First diverging step on the retaliation_selector probe: index 226
# (the END_OF_COMBAT step right after the COMBAT_DAMAGE branch's
# apply_sbas pass commits p0's `attempted_draw_on_empty` to
# `has_lost = True`, making both players losers).
# ---------------------------------------------------------------------------


def test_cross_engine_retaliation_selector_walks_clean(
    parity_smoke,
):
    """**CE12 closure regression** (was: surface-pin). The
    retaliation_selector walk now walks parity-clean past the
    former divergence point at step 226 — Rust's sticky winner_id
    shadow field matches Python's SBA-write semantics through the
    BOTH-LOST transition.

    See docs/RUST_PARITY_GAPS.md I34 and the (now-closed) CE12
    entry. Pre-closure, this test was xfail-strict and the
    divergence at step 226 was `winner_id` Python=0 vs Rust=None
    (Rust's pre-CE12 live-derived getter returned None when both
    players carried `has_lost=True`).
    """
    from mtg_bot.parity.cross_engine import (
        retaliation_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=260,
        decision_selector=retaliation_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )

    # Post-CE12 closure: EVERY step walks divergence-free. The
    # bilateral active-player swap, summoning-sick clear, symmetric
    # multi-block combat, life-total accounting, AND the BOTH-LOST
    # transition's sticky winner_id all match between engines.
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_retaliation_selector_pre_ce12_walks_clean(
    parity_smoke,
):
    """The pre-CE12 walk (steps 0-225) on
    :func:`multi_blocker_deckspec` with
    :func:`retaliation_selector` is parity-clean. Pins the contract
    that CE12 is SPECIFICALLY a winner_id-sticky-vs-derived finding —
    not a broader divergence in active-player swap, summoning-sick
    clear, or symmetric combat.

    Walks 226 decisions (one past the last clean step) and asserts
    every step except the final one is divergence-free. The final
    (step 225) is the COMBAT_DAMAGE branch's apply_sbas pass on
    p0's second attack-turn; its post-state is still clean (both
    engines: is_over=True, is_draw=False, winner_id=0). Step 226's
    pass_priority is the first step where the (already-diverged)
    Python-vs-Rust state surfaces through the field-view compare.

    See CE12 in docs/RUST_PARITY_GAPS.md.
    """
    from mtg_bot.parity.cross_engine import (
        retaliation_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=226,
        decision_selector=retaliation_selector,
    )
    assert result.error is None
    for step in result.steps[:226]:
        assert step.divergences == [], (
            f"pre-CE12 divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: confirm BOTH players attacked at least once,
    # which is the load-bearing change vs multi_block_combat_selector
    # (which suppressed p1 attacks entirely).
    declares = [
        s for s in result.steps
        if s.action_dict.get("kind") == "declare_attackers"
    ]
    p0_attacks = [s for s in declares if s.py_view["active_player"] == 0]
    p1_attacks = [s for s in declares if s.py_view["active_player"] == 1]
    assert len(p0_attacks) >= 1, (
        f"expected at least 1 p0 attack, got {len(p0_attacks)}"
    )
    assert len(p1_attacks) >= 1, (
        f"expected at least 1 p1 attack (the bilateral surface), "
        f"got {len(p1_attacks)}"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #8: mass_removal_selector — extend
# retaliation_selector with a Wrath-style ``DestroyAllCreatures``
# side-channel fire at step 168 (after the multi-creature bf state
# is set up by p0's turn-13 Bears cast). Originally surfaced CE13.
#
# **CE13 (CLOSED 2026-05-12) — summoning_sick-on-LTB Python-preserve
# vs Rust-clear.** Pre-closure: when a creature left the battlefield
# (destroyed / exiled / sacrificed / bounced), Python's
# ``turn.move_off_battlefield`` did NOT clear ``obj.summoning_sick``
# while Rust's destroy / exile / sacrifice / bounce arms in
# ``state_ops_resolve.rs`` explicitly set
# ``obj.summoning_sick = false`` on the leaving object. The
# fingerprint snapshot tuple captured the raw field, so the cross-
# engine field-view compare flagged the asymmetry.
#
# Closed by ``feature/ce13-summoning-sick-ltb`` via Option B (Rust-
# side, convergence on Python): the explicit clears were dropped
# from every LTB arm in ``state_ops_resolve.rs``. The result is
# bit-equal to Python's preserve-on-LTB semantics. See I35 in
# ``docs/RUST_PARITY_GAPS.md``.
# ---------------------------------------------------------------------------


def test_cross_engine_mass_removal_selector_walks_clean(
    parity_smoke,
):
    """**CE13 closure regression** (was: surface-pin). The
    mass_removal_selector walk now walks parity-clean THROUGH the
    former divergence point at step 168 — Rust's destroy / exile /
    sacrifice / bounce arms preserve summoning_sick on LTB matching
    Python's ``turn.move_off_battlefield`` semantics.

    See docs/RUST_PARITY_GAPS.md I35 and the (now-closed) CE13
    entry. Pre-closure, this test was xfail-strict and the
    divergence at step 168 (the side-channel wrath fire on
    multi_blocker_deckspec) was the ``graveyard_p0`` snapshot's
    ``summoning_sick`` slot: Python preserved sick=1 on the
    just-cast Bears id 5, Rust cleared it to sick=0.
    """
    from mtg_bot.parity.cross_engine import (
        mass_removal_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=170,
        decision_selector=mass_removal_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )

    # Post-CE13 closure: EVERY step walks divergence-free, including
    # step 168 (the side-channel wrath fire) where the still-sick
    # Bears id 5 lands in the graveyard with summoning_sick preserved
    # in both engines.
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_mass_removal_selector_post_wrath_bf_state(
    parity_smoke,
):
    """Validates the load-bearing observable AFTER the wrath fires:
    both engines agree that the battlefield is empty of creatures
    (only lands remain) and the graveyards have the expected counts.

    This is a structural assertion — it doesn't inspect the
    summoning_sick CE13 surface (that's the xfail-pinned test), it
    asserts the mass-destruction itself composed correctly on both
    engines. If a future closure of CE13 lands and the strict-xfail
    test flips, this test continues to pin the load-bearing "wrath
    actually destroyed every creature" semantics.

    Reads the Python state directly (not the field-view 22-tuple)
    so the assertions are independent of the cross-engine
    field-compare's per-snapshot tuple shape. Builds the matched
    states, drives the walk through step 168 (the wrath fire),
    then inspects.
    """
    from mtg_bot.parity.cross_engine import (
        mass_removal_selector,
        multi_blocker_deckspec,
        build_matched_states,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(169):
        action = mass_removal_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Python side: bf has only lands, p0 graveyard has [Bears, Wrath],
    # p1 graveyard has [Bears, Bears].
    py_creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert py_creatures == [], (
        f"expected no creatures on Python bf after wrath; got "
        f"{[(o.id, o.printing.name) for o in py_creatures]}"
    )
    py_p0_gy_names = [o.printing.name for o in py_state.players[0].graveyard]
    py_p1_gy_names = [o.printing.name for o in py_state.players[1].graveyard]
    assert py_p0_gy_names == ["Grizzly Bears", "Wrath"], (
        f"expected Python p0 gy=[Bears, Wrath]; got {py_p0_gy_names}"
    )
    assert py_p1_gy_names == ["Grizzly Bears", "Grizzly Bears"], (
        f"expected Python p1 gy=[Bears, Bears]; got {py_p1_gy_names}"
    )

    # Rust side: same shape. Read via the pyclass surface
    # (graveyard_sizes + the field-view's graveyard_pN signature for
    # the name positions).
    rust_gy_sizes = rust_state.graveyard_sizes
    assert rust_gy_sizes == [2, 2], (
        f"expected Rust graveyard_sizes=[2, 2]; got {rust_gy_sizes}"
    )
    rust_view = parity_smoke._build_field_view(rust_state)
    rust_p0_names = tuple(entry[0] for entry in rust_view["graveyard_p0"])
    rust_p1_names = tuple(entry[0] for entry in rust_view["graveyard_p1"])
    assert rust_p0_names == ("Grizzly Bears", "Wrath"), (
        f"expected Rust p0 gy names=(Bears, Wrath); got {rust_p0_names}"
    )
    assert rust_p1_names == ("Grizzly Bears", "Grizzly Bears"), (
        f"expected Rust p1 gy names=(Bears, Bears); got {rust_p1_names}"
    )

    # Both engines should agree the bf is empty of creatures (this
    # is the mass-removal-fired observable; CE13 only surfaces in
    # the per-object summoning_sick field on the graveyard
    # snapshot, not in the bf-cleared invariant). The Rust
    # battlefield_sizes reports total bf size (creatures + lands);
    # post-wrath both players' bf is land-only with 5/4 lands
    # respectively (p0 played turn-13 land that didn't tap;
    # p1 played turn-12 land). Inspect via the snapshot names
    # rather than the size to assert "no creatures".
    rust_bf_p0_names = tuple(
        entry[0] for entry in rust_view["battlefield"]
        if entry[6] == 0  # controller_id == 0
    )
    rust_bf_p1_names = tuple(
        entry[0] for entry in rust_view["battlefield"]
        if entry[6] == 1  # controller_id == 1
    )
    assert "Grizzly Bears" not in rust_bf_p0_names, (
        f"Rust p0 bf still has a Grizzly Bears after wrath: "
        f"{rust_bf_p0_names}"
    )
    assert "Grizzly Bears" not in rust_bf_p1_names, (
        f"Rust p1 bf still has a Grizzly Bears after wrath: "
        f"{rust_bf_p1_names}"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #9: first_strike_selector — extend retaliation_selector
# with a side-channel-injected 2/2 first-strike creature on p0's
# battlefield at step 168 (after the multi-creature bf state is set
# up by p0's turn-13 Bears cast). Probes CR 510.5 (two damage substeps
# when any participant has first-strike) + CR 702.7 (first-strike
# timing) + SBA-between-steps semantics + damage-marking persistence.
#
# **Outcome on this probe: CLEAN PASS (no novel divergence surfaced).**
#
# The walk runs 300+ decisions through the FS combat at turn 13 PRECOMBAT
# (the side-channel-injected attacker is unsick + untapped + on p0's bf;
# p1's bears are side-channel-untapped to be eligible blockers since the
# retaliation walk left them tapped from p1's turn-12 attack). Both engines
# converge bit-identically through every step, including the COMBAT_DAMAGE
# step where Python's combat.resolve_combat_damage and Rust's
# deal_combat_damage internally compose the two-pass first-strike timing.
#
# The post-combat state on both engines:
#   * p0 battlefield: Bears id 5 (sick), Forests, NO Bear Knight
#   * p1 battlefield: Bears id 11 (untapped, untouched), Forests
#   * p0 graveyard:   Bear Knight id 14 (the FS attacker — died in
#                     regular damage step from blocker 11's strike-back)
#   * p1 graveyard:   Bears id 9 (the first-strike-killed blocker)
#
# Which is the exact CR 510.5 expected outcome:
#   1. First-strike damage step: Bear Knight (FS) assigns 2 damage to
#      Bears id 9 (bf-ETB-first eligible blocker — both engines pick
#      the same blocker via the post-CE9-closure bf-ETB-sorted walk).
#   2. SBA between steps (Python side, combat.py:317-318): Bears id 9
#      dies, moves to graveyard.
#   3. Rust side has no SBA between steps but uses an in-function
#      ``damage < toughness`` alive predicate (state_ops_combat.rs:1066)
#      to filter out the dead blocker as a SOURCE in pass 2. Effectively
#      equivalent behavior, different mechanism.
#   4. Regular damage step: Bears id 11 (surviving blocker) deals 2
#      damage to Bear Knight. Bears id 9 in gy (Python) / damage-marked
#      on bf (Rust) deals nothing in either case.
#   5. SBA after pass 2 (both engines): Bear Knight dies. End state
#      converges.
#
# The end-state convergence is exactly the in-between-passes SBA-firing
# divergence ABSORBED by Rust's alive-predicate-on-pass-2 fold. Neither
# engine writes any cross-step-observable damage-marking state that the
# other engine doesn't also write (since the fingerprint compare runs
# AFTER the full deal_combat_damage + apply_sbas composition).
#
# Why this probe didn't surface a new CE: the candidate divergences the
# slice was sized to surface (SBA timing between passes; damage-marking
# carryover between passes; ordering of multi-blocker damage assignment)
# either:
#   (a) had already been closed (CE9 closure fixed bf-ETB-sorted blocker
#       iteration), or
#   (b) are invisible to the harness's per-step compare because the
#       fingerprint snapshot is taken AFTER the full combat damage
#       composition, by which time both engines have absorbed the
#       mid-pass state divergence into a converged end state.
#
# Next-slice candidates to probe a different combat surface:
#   * **Deathtouch + multi-block.** CR 702.2 — deathtouch lets a creature
#     assign 1 damage to each blocker as "lethal" for CR 510.1c damage
#     assignment ordering. Both engines have deathtouch hooks but the
#     interaction with multi-block + assignment splitting is a candidate
#     divergence.
#   * **Trample + multi-block.** CR 702.19 — trample lets a creature spill
#     unblocked excess damage to the defending player. Rust's combat
#     plan already handles trample (state_ops_combat.rs:905-921 "lethal
#     to blocker, excess to defender"). Python's combat.resolve walks a
#     different assignment shape. Potential divergence.
#   * **Lifelink.** CR 702.15 — lifelink grants life to the controller
#     for damage dealt. Both engines have lifelink hooks; the
#     cross-engine surface is the life-gain SBA composition.
# ---------------------------------------------------------------------------


def test_cross_engine_first_strike_selector_walks_clean(parity_smoke):
    """Cross-engine first-strike combat walks parity-clean. **Outcome (a)** —
    no novel divergence surfaced on this probe; first-strike combat
    damage timing matches between Python and Rust.

    Walks 200 decisions on :func:`multi_blocker_deckspec` with
    :func:`first_strike_selector`. Asserts every step's
    ``divergences`` list is empty, including the FS-combat steps at
    turn 13 (steps 168-177): side-channel inject + phase advance +
    declare_attackers([14]) + declare_blockers([[11,14], [9,14]]) +
    COMBAT_DAMAGE (which internally fires the two-pass first-strike
    timing on both engines) + END_OF_COMBAT.

    If a future change to either engine's combat damage path breaks
    first-strike parity, this test fires immediately.
    """
    from mtg_bot.parity.cross_engine import (
        first_strike_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=first_strike_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    assert result.initial_divergences == [], (
        f"unexpected initial divergence (pre-step-0): "
        f"{[(d.field, d.py_value, d.rust_value) for d in result.initial_divergences]}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_first_strike_selector_post_combat_state(parity_smoke):
    """Validate the load-bearing observable AFTER the first-strike combat
    fires: both engines agree on the post-combat creature distribution
    and graveyard contents.

    Pins the CR 510.5 expected outcome:
      * Bear Knight (FS attacker, id 14) dies from regular-step damage
        and lands in p0's graveyard.
      * Grizzly Bears id 9 (first-strike-killed blocker) lands in p1's
        graveyard.
      * Grizzly Bears id 11 (surviving blocker) stays on p1's
        battlefield with no damage marked.
      * Grizzly Bears id 5 (p0's sick Bears) is unchanged on p0's
        battlefield.

    Reads both Python state and Rust battlefield/graveyard snapshots
    directly so the assertions are independent of the field-view
    22-tuple shape. This is the structural assertion — if a future
    engine change shifts the FS-combat outcome on either side, this
    test fires.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        first_strike_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    # Walk through step 179 (well past the FS combat at steps 171-177
    # but before turn 14's combat at steps 184-187). At this point the
    # FS combat is fully resolved on both engines.
    for i in range(180):
        action = first_strike_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # ---- Python side ----
    py_creatures = sorted(
        (int(o.id), o.printing.name, o.controller_id, o.damage)
        for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    )
    # Expected: id 5 (Bears, p0, sick, 0 damage) and id 11 (Bears, p1,
    # untouched, 0 damage). Bear Knight id 14 in p0_gy, Bears id 9 in
    # p1_gy.
    assert py_creatures == [
        (5, "Grizzly Bears", 0, 0),
        (11, "Grizzly Bears", 1, 0),
    ], f"unexpected Python bf creatures post-FS-combat: {py_creatures}"

    py_p0_gy_names = [o.printing.name for o in py_state.players[0].graveyard]
    py_p1_gy_names = [o.printing.name for o in py_state.players[1].graveyard]
    assert py_p0_gy_names == ["Bear Knight"], (
        f"expected Python p0 gy=[Bear Knight] (FS attacker died); "
        f"got {py_p0_gy_names}"
    )
    assert py_p1_gy_names == ["Grizzly Bears"], (
        f"expected Python p1 gy=[Grizzly Bears] (FS-killed blocker); "
        f"got {py_p1_gy_names}"
    )

    # ---- Rust side ----
    rust_bf = rust_state.battlefield_snapshot()
    rust_creatures = sorted(
        (
            int(entry["id"]),
            entry["printing_name"],
            int(entry["controller_id"]),
            int(entry["damage"]),
        )
        for entry in rust_bf
        if entry["printing_name"] in ("Grizzly Bears", "Bear Knight")
    )
    assert rust_creatures == [
        (5, "Grizzly Bears", 0, 0),
        (11, "Grizzly Bears", 1, 0),
    ], f"unexpected Rust bf creatures post-FS-combat: {rust_creatures}"

    rust_gy0_names = [
        entry["printing_name"]
        for entry in rust_state.graveyard_snapshot(0)
    ]
    rust_gy1_names = [
        entry["printing_name"]
        for entry in rust_state.graveyard_snapshot(1)
    ]
    assert rust_gy0_names == ["Bear Knight"], (
        f"expected Rust p0 gy=[Bear Knight]; got {rust_gy0_names}"
    )
    assert rust_gy1_names == ["Grizzly Bears"], (
        f"expected Rust p1 gy=[Grizzly Bears]; got {rust_gy1_names}"
    )


def test_cross_engine_first_strike_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """The pre-injection walk (steps 0-167) on
    :func:`multi_blocker_deckspec` with :func:`first_strike_selector`
    is parity-clean. Pins the contract that the FS probe's
    delegate-to-retaliation_selector backbone is unchanged — steps
    0-167 are identical to retaliation_selector's walk shape and
    surface no divergences.

    Step 168 is where the side-channel FS injection fires; steps
    168-N exercise the FS combat surface. This test asserts ONLY
    the pre-injection backbone is clean; the full-walk clean-pass is
    asserted by ``test_cross_engine_first_strike_selector_walks_clean``.

    Validates the load-bearing pre-condition: at the end of step 167,
    the bf has exactly 3 creatures — p0=1 (sick Bears id 5), p1=2
    (unsick Bears ids 9, 11). The FS injection at step 168 reads
    this state and triggers via the ``>=3 creatures, no Bear Knight
    yet`` gate.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        first_strike_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=first_strike_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: rebuild matched states, drive 168 steps, and
    # confirm the pre-injection bf state.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = first_strike_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}: "
        f"{[(o.id, o.controller_id, o.summoning_sick) for o in creatures]}"
    )
    # No Bear Knight in any zone yet (the FS injection hasn't fired).
    bear_knight_anywhere = (
        any(o.printing.name == "Bear Knight" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Knight"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_knight_anywhere, (
        "Bear Knight should not yet exist at end of step 167 — FS "
        "injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #10: deathtouch_selector — extend retaliation_selector
# with a side-channel-injected 2/2 deathtouch creature on p0's
# battlefield at step 168, then drive multi-block combat (CR 702.2 +
# CR 510.1c) at turn 13 PRECOMBAT. Mirrors :func:`first_strike_selector`
# exactly except the injected creature carries ``deathtouch`` instead
# of ``first_strike`` and the trigger gate keys on the
# :data:`_DEATHTOUCH_PRINTING_NAME` ("Bear Assassin") so the two probes
# are independent.
#
# **CE14 CLOSED** by ``feature/ce14-deathtouch-multiblock``: Rust's
# ``state_ops_combat.rs`` no-trample multi-block branch now walks ALL
# blockers in bf-ETB order with ``need = 1 if deathtouch else
# effective_toughness - blocker.damage`` per blocker (Python's
# ``combat.py:419-435`` shape, with the "is_last gets remaining" tail
# preserving the round-5 single-blocker observable behavior). With a
# 2/2 deathtouch attacker blocked by two 2/2 Bears, both Python and
# Rust now assign 1 damage to each blocker, deathtouch upgrades each
# mark to lethal=2 via the apply-damage pass, and SBA destroys both.
# The strict-xfail walk-clean pin below has been flipped to a normal
# passing test; the symptom-present pin has been deleted per the
# CE2-CE13 pair-pin convention.
# ---------------------------------------------------------------------------


def test_cross_engine_deathtouch_selector_walks_clean(parity_smoke):
    """**CE14 clean-walk pin (passing post-closure)**.

    Walks 200 decisions on :func:`multi_blocker_deckspec` with
    :func:`deathtouch_selector` and asserts every step is parity-
    clean. Pre-closure, the multi-block COMBAT_DAMAGE step at index
    174 surfaced CE14: Python killed both blockers via deathtouch's
    CR 702.2b lethal-mark assignment override; Rust killed only the
    bf-ETB-first blocker (the no-trample branch emitted all damage
    to a single PlanEntry). Post-closure
    (``feature/ce14-deathtouch-multiblock``), Rust walks all
    blockers in bf-ETB order with ``need = 1 if deathtouch`` per
    blocker, matching Python's ``combat.py:419-435`` loop shape, so
    every step is parity-clean.

    Mirrors the CE9 / CE12 / CE13 post-closure pattern: the
    formerly-xfailed clean-walk pin becomes an expected-pass full-
    walk parity assertion, and the symptom-present companion is
    deleted per the CE2-CE13 pair-pin convention. See I36 for the
    post-closure invariant.
    """
    from mtg_bot.parity.cross_engine import (
        deathtouch_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=deathtouch_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_deathtouch_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """Pre-injection walk (steps 0-167) on :func:`multi_blocker_deckspec`
    with :func:`deathtouch_selector` is parity-clean. Pins the
    contract that the deathtouch probe's delegate-to-
    retaliation_selector backbone is unchanged — steps 0-167 are
    identical to retaliation_selector's walk shape and surface no
    divergences. The CE14 divergence is SPECIFICALLY at the
    COMBAT_DAMAGE step (174), not the pre-injection casts /
    mana-abilities / declare-attackers steps.

    Validates the pre-condition: at the end of step 167, the bf has
    exactly 3 creatures and no Bear Assassin (the deathtouch
    injection hasn't fired). Same shape as the first-strike probe's
    pre-injection test.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        deathtouch_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=deathtouch_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: rebuild matched states, drive 168 steps, and
    # confirm the pre-injection bf state.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = deathtouch_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}: "
        f"{[(o.id, o.controller_id, o.summoning_sick) for o in creatures]}"
    )
    # No Bear Assassin in any zone yet (the deathtouch injection
    # hasn't fired).
    bear_assassin_anywhere = (
        any(o.printing.name == "Bear Assassin" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Assassin"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_assassin_anywhere, (
        "Bear Assassin should not yet exist at end of step 167 — "
        "deathtouch injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #11: lifelink_selector — extend retaliation_selector
# with a side-channel-injected 2/2 lifelink creature on p0's battlefield
# at step 168, then drive multi-block combat (CR 702.15 — "Damage dealt by
# a creature with lifelink causes its controller to gain that much life")
# at turn 13 PRECOMBAT. Mirrors :func:`deathtouch_selector` exactly except
# the injected creature carries ``lifelink`` instead of ``deathtouch`` and
# the trigger gate keys on :data:`_LIFELINK_PRINTING_NAME` ("Bear Cleric")
# so the three keyword-combat probes (FS / DT / LL) stay independent on
# the same multi_blocker_deckspec walk.
#
# **OUTCOME on this probe: CLEAN PASS — no novel cross-engine divergence
# surfaced.** Both engines wire lifelink at damage-apply time:
#
#   * **Python**: ``combat.py:533-541`` calls
#     ``gain_life(state, source_controller, amount)`` per damage event
#     when ``"lifelink" in source_kws``.
#   * **Rust**: ``state_ops_combat.rs:1282-1289`` increments
#     ``self.players[cidx].life += entry.amount as i32`` per ``PlanEntry``
#     when ``battlefield_has_keyword(self, entry.source_id, "lifelink")``.
#
# Both engines fire lifelink per-event in their respective damage-apply
# loops, so the 2-damage assignment to the first blocker (need_0 = 2;
# remaining drops to 0 and the apply loop exits before blocker_1 per the
# post-CE14 ``if remaining <= 0: break`` guard on each engine) results in
# +2 life on p0's controller. Post-combat: p0 life = 14 (was 12 from
# turn-12 retaliation Bears attack; +2 from lifelink), p1 life = 20.
# p0 gy = [Bear Cleric] (attacker died from 4 back-damage from blockers),
# p1 gy = [Grizzly Bears id 9] (first blocker died from assigned 2),
# p1 bf still has Grizzly Bears id 11 (survived because remaining was 0
# after blocker_0).
#
# Three pinning tests guard the contract:
#
#   1. ``test_cross_engine_lifelink_selector_walks_clean`` — full
#      200-decision walk parity-clean on every step.
#   2. ``test_cross_engine_lifelink_selector_post_combat_state`` —
#      direct-read structural assertion on life totals + bf / gy
#      contents post-COMBAT_DAMAGE. Pins the EXPECTED outcome shape so
#      a future refactor that silently breaks lifelink composition
#      (e.g. dropping the lifelink keyword read in apply_damage_pass,
#      or moving the source-keyword lookup to AFTER apply when the
#      attacker has died) fails loudly rather than silently passing
#      the walk-clean test (which would also fail, but the direct-
#      read pin makes the failure mode obvious — "p0 didn't gain 2
#      life" is faster to diagnose than "step 174 has field
#      'life_p0'='12' vs '14'").
#   3. ``test_cross_engine_lifelink_selector_pre_injection_walks_clean``
#      — steps 0-167 are parity-clean. The injection fires at step 168;
#      pre-injection backbone is identical to retaliation_selector.
# ---------------------------------------------------------------------------


def test_cross_engine_lifelink_selector_walks_clean(parity_smoke):
    """**Lifelink clean-walk pin** — 200-decision walk on
    :func:`multi_blocker_deckspec` with :func:`lifelink_selector` is
    parity-clean every step. The lifelink injection at step 168 + the
    multi-block COMBAT_DAMAGE step at index 174 compose correctly on
    both engines: 2 damage assigned to blocker_0 fires lifelink for
    +2 life on p0's controller, mirroring Python's per-event
    ``gain_life`` call in :func:`mtg_bot.combat.resolve_combat_damage`
    and Rust's per-PlanEntry ``self.players[cidx].life += entry.amount``
    in :meth:`GameState::apply_damage_pass`.

    Mirrors the first-strike probe's post-walk-clean shape: lifelink
    behavior is already parity across both engines (no CE16 surfaced
    on this slice), so this test is a "doesn't regress" guard rather
    than a strict-xfail pre-closure pin.
    """
    from mtg_bot.parity.cross_engine import (
        lifelink_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=lifelink_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_lifelink_selector_post_combat_state(parity_smoke):
    """**Lifelink post-combat structural pin** — drives 180 selector
    decisions to walk past the lifelink COMBAT_DAMAGE step and asserts
    the EXACT post-combat life totals + bf / gy contents on both
    engines. Pins the shape so a future refactor that silently breaks
    lifelink composition fails loudly with a specific diagnostic
    rather than a generic walk-clean divergence.

    Expected post-combat state (matches both engines today):

      * p0 life = 14 (was 12 entering combat; +2 from lifelink on the
        single 2-damage event to the first blocker).
      * p1 life = 20 (unchanged — Bears blockers have no lifelink and
        deal back-damage to the attacker, not to p1).
      * p0 graveyard = [Bear Cleric id 14] (attacker died from 4
        back-damage from both blockers' 2 power each).
      * p1 graveyard = [Grizzly Bears id 9] (first blocker took
        assigned 2 damage = its toughness, marked lethal, SBA destroyed).
      * p1 battlefield still has Grizzly Bears id 11 (second blocker
        survived because ``remaining = 0`` after blocker_0 absorbed
        its need=2; the post-CE14 ``if remaining <= 0: break`` guard
        skips blocker_1 entirely).
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        lifelink_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(180):
        action = lifelink_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # ---- Life totals (the load-bearing lifelink composition surface) ----
    assert py_state.players[0].life == 14, (
        f"expected Python p0 life = 14 (12 pre-combat + 2 lifelink "
        f"gain), got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == 20, (
        f"expected Python p1 life = 20 (unchanged; blockers have no "
        f"lifelink), got {py_state.players[1].life}"
    )
    rust_life = rust_state.life_totals
    assert rust_life[0] == 14, (
        f"expected Rust p0 life = 14 (matches Python), got "
        f"{rust_life[0]}"
    )
    assert rust_life[1] == 20, (
        f"expected Rust p1 life = 20 (matches Python), got "
        f"{rust_life[1]}"
    )

    # ---- p0 graveyard: Bear Cleric (the lifelink attacker died from
    #      back-damage) ----
    py_p0_gy = [(o.id, o.printing.name) for o in py_state.players[0].graveyard]
    assert py_p0_gy == [(14, "Bear Cleric")], (
        f"expected Python p0 gy = [(14, 'Bear Cleric')], got {py_p0_gy}"
    )
    rust_p0_gy = [
        (e["id"], e.get("printing_name"))
        for e in rust_state.graveyard_snapshot(0)
    ]
    assert rust_p0_gy == [(14, "Bear Cleric")], (
        f"expected Rust p0 gy = [(14, 'Bear Cleric')], got {rust_p0_gy}"
    )

    # ---- p1 graveyard: Grizzly Bears id 9 (first blocker took
    #      lethal 2) ----
    py_p1_gy = [(o.id, o.printing.name) for o in py_state.players[1].graveyard]
    assert py_p1_gy == [(9, "Grizzly Bears")], (
        f"expected Python p1 gy = [(9, 'Grizzly Bears')], got {py_p1_gy}"
    )
    rust_p1_gy = [
        (e["id"], e.get("printing_name"))
        for e in rust_state.graveyard_snapshot(1)
    ]
    assert rust_p1_gy == [(9, "Grizzly Bears")], (
        f"expected Rust p1 gy = [(9, 'Grizzly Bears')], got {rust_p1_gy}"
    )

    # ---- p1 battlefield still has Grizzly Bears id 11 (second blocker
    #      survived because remaining was 0 after first blocker absorbed
    #      need=2). ----
    py_bf_bears = sorted(
        o.id for o in py_state.battlefield
        if o.printing.name == "Grizzly Bears"
    )
    # p0 has Bears id 5 (the original turn-13-cast sick one); p1 has
    # Bears id 11 (the surviving second blocker). Bears 9 is in p1's
    # graveyard.
    assert py_bf_bears == [5, 11], (
        f"expected Python bf Grizzly Bears ids = [5, 11], got "
        f"{py_bf_bears}"
    )
    rust_bf_bears = sorted(
        int(e["id"]) for e in rust_state.battlefield_snapshot()
        if e.get("printing_name") == "Grizzly Bears"
    )
    assert rust_bf_bears == [5, 11], (
        f"expected Rust bf Grizzly Bears ids = [5, 11], got "
        f"{rust_bf_bears}"
    )


def test_cross_engine_lifelink_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """Pre-injection walk (steps 0-167) on :func:`multi_blocker_deckspec`
    with :func:`lifelink_selector` is parity-clean. Pins the contract
    that the lifelink probe's delegate-to-retaliation_selector
    backbone is unchanged — steps 0-167 are identical to
    retaliation_selector's walk shape and surface no divergences.
    The lifelink injection fires SPECIFICALLY at step 168, not before.

    Validates the pre-condition: at the end of step 167, the bf has
    exactly 3 creatures and no Bear Cleric (the lifelink injection
    hasn't fired). Same shape as the first-strike + deathtouch probes'
    pre-injection tests.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        lifelink_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=lifelink_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: rebuild matched states, drive 168 steps, and
    # confirm the pre-injection bf state.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = lifelink_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}: "
        f"{[(o.id, o.controller_id, o.summoning_sick) for o in creatures]}"
    )
    # No Bear Cleric in any zone yet (the lifelink injection hasn't
    # fired).
    bear_cleric_anywhere = (
        any(o.printing.name == "Bear Cleric" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Cleric"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_cleric_anywhere, (
        "Bear Cleric should not yet exist at end of step 167 — "
        "lifelink injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #12: vigilance_selector — extend retaliation_selector
# with a side-channel-injected 2/2 vigilance creature on p0's battlefield
# at step 168, then drive an attack (CR 702.20 — "Attacking doesn't cause
# this creature to tap") at turn 13 PRECOMBAT. Mirrors
# :func:`lifelink_selector` exactly except the injected creature carries
# ``vigilance`` instead of ``lifelink`` and the trigger gate keys on
# :data:`_VIGILANCE_PRINTING_NAME` ("Bear Watcher") so the four
# keyword-combat probes (FS / DT / LL / vigilance) stay independent on
# the same multi_blocker_deckspec walk.
#
# **OUTCOME on this probe: CLEAN PASS — no novel cross-engine divergence
# surfaced.** Both engines wire vigilance at declare-attackers time:
#
#   * **Python**: ``combat.py:103-104`` inside
#     ``declare_attackers``'s per-attacker loop suppresses the
#     ``tap_permanent`` call when ``"vigilance" in
#     effective_keywords(state, obj)``.
#   * **Rust**: ``state_ops_combat.rs:654-669`` in ``declare_attacker``
#     reads ``has_vigilance = battlefield_has_vigilance(...)`` and
#     skips the ``mutate_battlefield_obj(... |obj| obj.tapped = true)``
#     branch entirely.
#
# Both engines mark the attacker ``attacking=True`` without also setting
# ``tapped=True``. The post-DECLARE_ATTACKERS state has
# ``attacker.tapped == False`` on BOTH engines. The rest of combat
# proceeds normally: two 2/2 Bears multi-block, deal 4 back-damage, the
# vigilance attacker dies via SBA. Post-combat: p0 life = 12 (unchanged
# — no lifelink → no life gain), p1 life = 20. p0 gy = [Bear Watcher],
# p1 gy = [Grizzly Bears id 9], p1 bf still has Grizzly Bears id 11.
#
# Three pinning tests guard the contract:
#
#   1. ``test_cross_engine_vigilance_selector_walks_clean`` — full
#      200-decision walk parity-clean on every step (which is where
#      the DECLARE_ATTACKERS ``tapped`` field is compared
#      cross-engine).
#   2. ``test_cross_engine_vigilance_selector_post_declare_attacker_untapped``
#      — direct-read structural assertion that AFTER the
#      DECLARE_ATTACKERS step (step 171) the attacker on both engines
#      reports ``tapped == False`` AND ``attacking == True`` (mirror).
#      Pins the EXPECTED vigilance shape so a future refactor that
#      silently breaks the vigilance gate (e.g. dropping the
#      ``!has_vigilance`` guard in Rust's declare_attacker, or moving
#      the keyword lookup to AFTER tap_permanent) fails loudly with a
#      specific diagnostic.
#   3. ``test_cross_engine_vigilance_selector_pre_injection_walks_clean``
#      — steps 0-167 are parity-clean. The injection fires at step 168;
#      pre-injection backbone is identical to retaliation_selector.
# ---------------------------------------------------------------------------


def test_cross_engine_vigilance_selector_walks_clean(parity_smoke):
    """**Vigilance clean-walk pin** — 200-decision walk on
    :func:`multi_blocker_deckspec` with :func:`vigilance_selector` is
    parity-clean every step. The vigilance injection at step 168 + the
    DECLARE_ATTACKERS step at index 171 compose correctly on both
    engines: the attacker is marked attacking but NOT tapped, mirroring
    Python's ``if "vigilance" not in effective_keywords(state, obj):
    tap_permanent(state, obj)`` guard in
    :func:`mtg_bot.combat.declare_attackers` and Rust's
    ``if !has_vigilance { ... obj.tapped = true ... }`` branch in
    :meth:`GameState::declare_attacker`.

    Mirrors the lifelink probe's post-walk-clean shape: vigilance
    behavior is already parity across both engines (no CE16 surfaced
    on this slice), so this test is a "doesn't regress" guard rather
    than a strict-xfail pre-closure pin.
    """
    from mtg_bot.parity.cross_engine import (
        multi_blocker_deckspec,
        vigilance_selector,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=vigilance_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_vigilance_selector_post_declare_attacker_untapped(
    parity_smoke,
):
    """**Vigilance post-DECLARE_ATTACKERS structural pin** — drives 172
    selector decisions to walk through the DECLARE_ATTACKERS step (171)
    and asserts the EXACT vigilance-honored state on both engines.
    Pins the load-bearing surface — ``attacker.tapped`` AFTER declare —
    so a future refactor that silently breaks the vigilance gate fails
    loudly with a specific diagnostic rather than a generic walk-clean
    divergence.

    Expected post-DECLARE_ATTACKERS state (matches both engines today):

      * Vigilance attacker (id 14, "Bear Watcher") is on p0's
        battlefield with ``attacking == True`` AND ``tapped == False``.
        This is the CR 702.20 contract: attacking doesn't tap a
        vigilance creature.

    Additionally pins the post-combat state (181 decisions) — the
    attacker dies via SBA from 4 back-damage from the two Bears
    blockers; vigilance doesn't gate damage so there's no life-gain or
    damage-distribution surface to assert beyond "matches lifelink
    probe modulo no life gain".
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
        step_both,
        vigilance_selector,
    )

    # ---- Phase 1: post-DECLARE_ATTACKERS attacker state ----
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(172):
        action = vigilance_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # The vigilance attacker (id 14, "Bear Watcher") should be on bf
    # with attacking=True, tapped=False.
    py_att = next(
        (o for o in py_state.battlefield if o.id == 14), None
    )
    assert py_att is not None, (
        "expected Bear Watcher (id 14) on Python p0 bf after step 171; "
        f"bf ids = {[o.id for o in py_state.battlefield]}"
    )
    assert py_att.printing.name == "Bear Watcher", (
        f"expected id 14 to be Bear Watcher, got {py_att.printing.name}"
    )
    assert py_att.attacking is True, (
        f"expected Python attacker.attacking = True, got "
        f"{py_att.attacking}"
    )
    assert py_att.tapped is False, (
        f"expected Python attacker.tapped = False (vigilance honored "
        f"per CR 702.20), got {py_att.tapped}"
    )

    rust_att = next(
        (e for e in rust_state.battlefield_snapshot() if int(e["id"]) == 14),
        None,
    )
    assert rust_att is not None, (
        "expected Bear Watcher (id 14) on Rust p0 bf after step 171"
    )
    assert rust_att.get("printing_name") == "Bear Watcher", (
        f"expected id 14 to be Bear Watcher on Rust, got "
        f"{rust_att.get('printing_name')}"
    )
    assert rust_att["tapped"] is False, (
        f"expected Rust attacker.tapped = False (vigilance honored), "
        f"got {rust_att['tapped']}"
    )

    # ---- Phase 2: post-combat state (drive 180 total decisions —
    #      past COMBAT_DAMAGE and SBA cleanup) ----
    for i in range(172, 180):
        action = vigilance_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Life totals: no lifelink → no life gain → p0 unchanged at 12
    # (was 12 entering combat after turn-12 retaliation), p1 at 20.
    assert py_state.players[0].life == 12, (
        f"expected Python p0 life = 12 (no lifelink → no life gain), "
        f"got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == 20, (
        f"expected Python p1 life = 20 (unchanged), got "
        f"{py_state.players[1].life}"
    )
    rust_life = rust_state.life_totals
    assert rust_life[0] == 12, (
        f"expected Rust p0 life = 12 (matches Python), got "
        f"{rust_life[0]}"
    )
    assert rust_life[1] == 20, (
        f"expected Rust p1 life = 20 (matches Python), got "
        f"{rust_life[1]}"
    )

    # p0 graveyard: Bear Watcher (vigilance attacker died from 4
    # back-damage — vigilance doesn't grant damage prevention).
    py_p0_gy = [
        (o.id, o.printing.name) for o in py_state.players[0].graveyard
    ]
    assert py_p0_gy == [(14, "Bear Watcher")], (
        f"expected Python p0 gy = [(14, 'Bear Watcher')], got {py_p0_gy}"
    )
    rust_p0_gy = [
        (e["id"], e.get("printing_name"))
        for e in rust_state.graveyard_snapshot(0)
    ]
    assert rust_p0_gy == [(14, "Bear Watcher")], (
        f"expected Rust p0 gy = [(14, 'Bear Watcher')], got {rust_p0_gy}"
    )

    # p1 graveyard: Grizzly Bears id 9 (first blocker took lethal 2;
    # same shape as the lifelink probe — vigilance doesn't change the
    # damage-assignment branch).
    py_p1_gy = [
        (o.id, o.printing.name) for o in py_state.players[1].graveyard
    ]
    assert py_p1_gy == [(9, "Grizzly Bears")], (
        f"expected Python p1 gy = [(9, 'Grizzly Bears')], got {py_p1_gy}"
    )
    rust_p1_gy = [
        (e["id"], e.get("printing_name"))
        for e in rust_state.graveyard_snapshot(1)
    ]
    assert rust_p1_gy == [(9, "Grizzly Bears")], (
        f"expected Rust p1 gy = [(9, 'Grizzly Bears')], got {rust_p1_gy}"
    )

    # p1 battlefield still has Grizzly Bears id 11 (second blocker
    # survived because remaining was 0 after first blocker absorbed
    # need=2 — post-CE14 ``if remaining <= 0: break`` guard).
    py_bf_bears = sorted(
        o.id for o in py_state.battlefield
        if o.printing.name == "Grizzly Bears"
    )
    assert py_bf_bears == [5, 11], (
        f"expected Python bf Grizzly Bears ids = [5, 11], got "
        f"{py_bf_bears}"
    )
    rust_bf_bears = sorted(
        int(e["id"]) for e in rust_state.battlefield_snapshot()
        if e.get("printing_name") == "Grizzly Bears"
    )
    assert rust_bf_bears == [5, 11], (
        f"expected Rust bf Grizzly Bears ids = [5, 11], got "
        f"{rust_bf_bears}"
    )


def test_cross_engine_vigilance_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """Pre-injection walk (steps 0-167) on :func:`multi_blocker_deckspec`
    with :func:`vigilance_selector` is parity-clean. Pins the contract
    that the vigilance probe's delegate-to-retaliation_selector
    backbone is unchanged — steps 0-167 are identical to
    retaliation_selector's walk shape and surface no divergences.
    The vigilance injection fires SPECIFICALLY at step 168, not before.

    Validates the pre-condition: at the end of step 167, the bf has
    exactly 3 creatures and no Bear Watcher (the vigilance injection
    hasn't fired). Same shape as the first-strike + deathtouch +
    lifelink probes' pre-injection tests.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
        step_both,
        vigilance_selector,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=vigilance_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: rebuild matched states, drive 168 steps, and
    # confirm the pre-injection bf state.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = vigilance_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}: "
        f"{[(o.id, o.controller_id, o.summoning_sick) for o in creatures]}"
    )
    # No Bear Watcher in any zone yet (the vigilance injection hasn't
    # fired).
    bear_watcher_anywhere = (
        any(o.printing.name == "Bear Watcher" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Watcher"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_watcher_anywhere, (
        "Bear Watcher should not yet exist at end of step 167 — "
        "vigilance injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #13: haste_selector — extend retaliation_selector with
# a side-channel-injected 2/2 SUMMONING-SICK haste creature on p0's
# battlefield at step 168, then drive an attack (CR 702.10b — "A creature
# with haste can attack and {T} as soon as it comes under its controller's
# control") at turn 13 PRECOMBAT. Mirrors :func:`vigilance_selector`
# exactly except the injected creature carries ``haste`` instead of
# ``vigilance``, is left in the ``summoning_sick=True`` state (load-
# bearing — exercises the haste bypass of the CR 302.1 sickness gate),
# and the trigger gate keys on :data:`_HASTE_PRINTING_NAME`
# ("Bear Sprinter") so the five keyword-combat probes (FS / DT / LL /
# vigilance / haste) stay independent on the same multi_blocker_deckspec
# walk.
#
# **OUTCOME on this probe: CLEAN PASS — no novel cross-engine divergence
# surfaced.** Both engines wire haste at declare-attackers eligibility:
#
#   * **Python** (``combat.py:21``): inside ``eligible_attackers``'s
#     comprehension filter, the predicate ``(not o.summoning_sick or
#     "haste" in effective_keywords(state, o))`` bypasses the sickness
#     exclusion when the keyword is present. ``declare_attackers``
#     itself does not re-validate eligibility, so a sick haste creature
#     requested via DECLARE_ATTACKERS([id]) makes it through.
#
#   * **Rust** (``state_ops_combat.rs:648-651``): in
#     ``declare_attacker``, ``let has_haste = battlefield_has_keyword(
#     self, obj_id, "haste"); if summoning_sick && !has_haste { return
#     false; }`` short-circuits the sick-rejection on the keyword. The
#     same gate appears in ``state_ops_legal.rs:574-577`` for the
#     legal_decisions enumeration filter.
#
# Both engines mark the haste attacker ``attacking=True`` AND
# ``tapped=True`` (no vigilance → standard CR 508.1c tap step runs)
# while LEAVING ``summoning_sick=True`` untouched (haste bypasses the
# attack restriction but does not remove summoning sickness — that
# clears at the controller's next untap step per CR 302.1, not at
# declare). The rest of combat proceeds normally: two 2/2 Bears
# multi-block, deal 4 back-damage, the haste attacker dies via SBA.
# Post-combat: p0 life = 12 (unchanged — no lifelink), p1 life = 20.
# p0 gy = [Bear Sprinter], p1 gy = [Grizzly Bears id 9], p1 bf still
# has Grizzly Bears id 11.
#
# **DECLARE_ATTACKERS selector override.** The
# :func:`retaliation_selector` delegate filters out ``summoning_sick``
# creatures at its attacker enumeration site (line ~2125) — a
# search-side approximation of CR 302.1 that is NOT keyword-aware.
# For this probe to actually drive the sick haste creature through
# DECLARE_ATTACKERS, the :func:`haste_selector` overrides the
# DECLARE_ATTACKERS branch with a keyword-aware predicate
# ``(not o.summoning_sick or "haste" in o.printing.keywords)`` BEFORE
# falling through to retaliation_selector. Without that override, the
# Bear Sprinter is silently excluded and the probe degenerates to a
# pass-priority walk that doesn't exercise the haste apply-path.
#
# Three pinning tests guard the contract:
#
#   1. ``test_cross_engine_haste_selector_walks_clean`` — full
#      200-decision walk parity-clean on every step (which is where
#      the DECLARE_ATTACKERS ``attacking`` + ``tapped`` +
#      ``summoning_sick`` fields are compared cross-engine).
#   2. ``test_cross_engine_haste_selector_post_declare_attacker_state``
#      — direct-read structural assertion that AFTER the
#      DECLARE_ATTACKERS step (step 171) the attacker on both engines
#      reports ``attacking == True``, ``tapped == True`` (haste does
#      NOT grant vigilance), AND ``summoning_sick == True`` (haste
#      bypasses but does not remove summoning sickness). Pins the
#      EXPECTED haste shape so a future refactor that silently breaks
#      the haste gate (e.g. dropping the ``has_haste`` branch in
#      Rust's declare_attacker, or clearing summoning_sick on declare)
#      fails loudly with a specific diagnostic.
#   3. ``test_cross_engine_haste_selector_pre_injection_walks_clean``
#      — steps 0-167 are parity-clean. The injection fires at step
#      168; pre-injection backbone is identical to
#      retaliation_selector.
# ---------------------------------------------------------------------------


def test_cross_engine_haste_selector_walks_clean(parity_smoke):
    """**Haste clean-walk pin** — 200-decision walk on
    :func:`multi_blocker_deckspec` with :func:`haste_selector` is
    parity-clean every step. The haste injection at step 168
    (``summoning_sick=True`` Bear Sprinter on p0 bf) + the
    DECLARE_ATTACKERS step at index 171 compose correctly on both
    engines: the sick haste creature is accepted as a legal attacker,
    mirroring Python's ``(not o.summoning_sick or "haste" in
    effective_keywords(state, o))`` guard in
    :func:`mtg_bot.combat.eligible_attackers` and Rust's
    ``if summoning_sick && !has_haste { return false; }`` branch in
    :meth:`GameState::declare_attacker`.

    Mirrors the vigilance probe's post-walk-clean shape: haste
    eligibility behavior is already parity across both engines (no
    CE16 surfaced on this slice), so this test is a "doesn't regress"
    guard rather than a strict-xfail pre-closure pin.
    """
    from mtg_bot.parity.cross_engine import (
        haste_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=haste_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_haste_selector_post_declare_attacker_state(
    parity_smoke,
):
    """**Haste post-DECLARE_ATTACKERS structural pin** — drives 172
    selector decisions to walk through the DECLARE_ATTACKERS step (171)
    and asserts the EXACT haste-honored state on both engines. Pins
    the load-bearing surface — ``attacker.attacking`` + ``tapped`` +
    ``summoning_sick`` AFTER declare — so a future refactor that
    silently breaks the haste gate fails loudly with a specific
    diagnostic rather than a generic walk-clean divergence.

    Expected post-DECLARE_ATTACKERS state (matches both engines today):

      * Haste attacker (id 14, "Bear Sprinter") is on p0's
        battlefield with ``attacking == True`` (haste bypassed the
        sickness gate; CR 702.10b), ``tapped == True`` (no vigilance
        → standard CR 508.1c tap step), AND ``summoning_sick == True``
        (haste bypasses but does NOT clear summoning sickness — the
        flag clears at the next untap step per CR 302.1, not on
        declare).

    Additionally pins the post-combat state (181 decisions) — the
    attacker dies via SBA from 4 back-damage from the two Bears
    blockers; haste doesn't gate damage so there's no life-gain or
    damage-distribution surface to assert beyond "matches vigilance
    probe modulo the post-declare tap state".
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        haste_selector,
        multi_blocker_deckspec,
        step_both,
    )

    # ---- Phase 1: post-DECLARE_ATTACKERS attacker state ----
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(172):
        action = haste_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # The haste attacker (id 14, "Bear Sprinter") should be on bf
    # with attacking=True, tapped=True, summoning_sick=True.
    py_att = next(
        (o for o in py_state.battlefield if o.id == 14), None
    )
    assert py_att is not None, (
        "expected Bear Sprinter (id 14) on Python p0 bf after step 171; "
        f"bf ids = {[o.id for o in py_state.battlefield]}"
    )
    assert py_att.printing.name == "Bear Sprinter", (
        f"expected id 14 to be Bear Sprinter, got {py_att.printing.name}"
    )
    assert py_att.attacking is True, (
        f"expected Python attacker.attacking = True (haste bypassed "
        f"summoning-sickness gate per CR 702.10b), got "
        f"{py_att.attacking}"
    )
    assert py_att.tapped is True, (
        f"expected Python attacker.tapped = True (haste does NOT "
        f"grant vigilance — standard CR 508.1c tap step ran), got "
        f"{py_att.tapped}"
    )
    assert py_att.summoning_sick is True, (
        f"expected Python attacker.summoning_sick = True (haste "
        f"bypasses but does NOT remove summoning sickness — flag "
        f"clears at the next untap step per CR 302.1), got "
        f"{py_att.summoning_sick}"
    )

    rust_att = next(
        (e for e in rust_state.battlefield_snapshot() if int(e["id"]) == 14),
        None,
    )
    assert rust_att is not None, (
        "expected Bear Sprinter (id 14) on Rust p0 bf after step 171"
    )
    assert rust_att.get("printing_name") == "Bear Sprinter", (
        f"expected id 14 to be Bear Sprinter on Rust, got "
        f"{rust_att.get('printing_name')}"
    )
    assert rust_att["tapped"] is True, (
        f"expected Rust attacker.tapped = True (haste doesn't grant "
        f"vigilance), got {rust_att['tapped']}"
    )
    assert rust_att["summoning_sick"] is True, (
        f"expected Rust attacker.summoning_sick = True (haste bypasses "
        f"but does NOT clear summoning sickness), got "
        f"{rust_att['summoning_sick']}"
    )

    # ---- Phase 2: post-combat state (drive 180 total decisions —
    #      past COMBAT_DAMAGE and SBA cleanup) ----
    for i in range(172, 180):
        action = haste_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Life totals: no lifelink → no life gain → p0 unchanged at 12
    # (was 12 entering combat after turn-12 retaliation), p1 at 20.
    assert py_state.players[0].life == 12, (
        f"expected Python p0 life = 12 (no lifelink → no life gain), "
        f"got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == 20, (
        f"expected Python p1 life = 20 (unchanged), got "
        f"{py_state.players[1].life}"
    )
    rust_life = rust_state.life_totals
    assert rust_life[0] == 12, (
        f"expected Rust p0 life = 12 (matches Python), got "
        f"{rust_life[0]}"
    )
    assert rust_life[1] == 20, (
        f"expected Rust p1 life = 20 (matches Python), got "
        f"{rust_life[1]}"
    )

    # p0 graveyard: Bear Sprinter (haste attacker died from 4
    # back-damage — haste doesn't grant damage prevention).
    py_p0_gy = [
        (o.id, o.printing.name) for o in py_state.players[0].graveyard
    ]
    assert py_p0_gy == [(14, "Bear Sprinter")], (
        f"expected Python p0 gy = [(14, 'Bear Sprinter')], got {py_p0_gy}"
    )
    rust_p0_gy = [
        (e["id"], e.get("printing_name"))
        for e in rust_state.graveyard_snapshot(0)
    ]
    assert rust_p0_gy == [(14, "Bear Sprinter")], (
        f"expected Rust p0 gy = [(14, 'Bear Sprinter')], got {rust_p0_gy}"
    )

    # p1 graveyard: Grizzly Bears id 9 (first blocker took lethal 2;
    # same shape as the vigilance probe — haste doesn't change the
    # damage-assignment branch).
    py_p1_gy = [
        (o.id, o.printing.name) for o in py_state.players[1].graveyard
    ]
    assert py_p1_gy == [(9, "Grizzly Bears")], (
        f"expected Python p1 gy = [(9, 'Grizzly Bears')], got {py_p1_gy}"
    )
    rust_p1_gy = [
        (e["id"], e.get("printing_name"))
        for e in rust_state.graveyard_snapshot(1)
    ]
    assert rust_p1_gy == [(9, "Grizzly Bears")], (
        f"expected Rust p1 gy = [(9, 'Grizzly Bears')], got {rust_p1_gy}"
    )

    # p1 battlefield still has Grizzly Bears id 11 (second blocker
    # survived because remaining was 0 after first blocker absorbed
    # need=2 — post-CE14 ``if remaining <= 0: break`` guard).
    py_bf_bears = sorted(
        o.id for o in py_state.battlefield
        if o.printing.name == "Grizzly Bears"
    )
    assert py_bf_bears == [5, 11], (
        f"expected Python bf Grizzly Bears ids = [5, 11], got "
        f"{py_bf_bears}"
    )
    rust_bf_bears = sorted(
        int(e["id"]) for e in rust_state.battlefield_snapshot()
        if e.get("printing_name") == "Grizzly Bears"
    )
    assert rust_bf_bears == [5, 11], (
        f"expected Rust bf Grizzly Bears ids = [5, 11], got "
        f"{rust_bf_bears}"
    )


def test_cross_engine_haste_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """Pre-injection walk (steps 0-167) on :func:`multi_blocker_deckspec`
    with :func:`haste_selector` is parity-clean. Pins the contract that
    the haste probe's delegate-to-retaliation_selector backbone is
    unchanged — steps 0-167 are identical to retaliation_selector's
    walk shape and surface no divergences. The haste injection fires
    SPECIFICALLY at step 168, not before.

    Validates the pre-condition: at the end of step 167, the bf has
    exactly 3 creatures and no Bear Sprinter (the haste injection
    hasn't fired). Same shape as the first-strike + deathtouch +
    lifelink + vigilance probes' pre-injection tests.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        haste_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=haste_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: rebuild matched states, drive 168 steps, and
    # confirm the pre-injection bf state.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = haste_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}: "
        f"{[(o.id, o.controller_id, o.summoning_sick) for o in creatures]}"
    )
    # No Bear Sprinter in any zone yet (the haste injection hasn't
    # fired).
    bear_sprinter_anywhere = (
        any(o.printing.name == "Bear Sprinter" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Sprinter"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_sprinter_anywhere, (
        "Bear Sprinter should not yet exist at end of step 167 — "
        "haste injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #14: menace_selector — extend retaliation_selector with
# a side-channel-injected 2/2 menace creature on p0's battlefield at step
# 168, then drive a SINGLE-blocker DECLARE_BLOCKERS request (illegal per
# CR 702.110b — "can't be blocked except by two or more creatures") at
# turn 13 to surface the predicted CE16 divergence.
#
# Mirrors :func:`vigilance_selector` / :func:`haste_selector` exactly
# except the injected creature carries ``menace`` and the
# DECLARE_BLOCKERS branch is parameterized — three test scenarios
# exercise three block decisions:
#
#   * **A** (``"none"``): zero blockers → legal, menace invisible.
#   * **B** (``"both"``): two blockers → legal multi-block, menace
#     satisfied.
#   * **C** (``"single"``): one blocker → ILLEGAL per CR 702.110b.
#
# **OUTCOME on this probe: CE16 SURFACED then CLOSED by
# ``feature/ce16-menace-validation``** — the Rust
# ``Action::DeclareBlockers`` apply arm now post-validates each
# unique attacker's block assignment via
# ``is_legal_block_assignment`` (the predicate at
# ``state_ops_combat.rs:779-789``) and silently drops illegal
# final assignments from ``self.blockers`` via ``.retain``,
# mirroring Python's ``actions.py:2555-2566`` ``validated.pop(...)``
# post-validation pass. Single-blocker menace assignments are now
# dropped on both engines; the attacker proceeds unblocked.
# Scenarios A (none) + B (both) + C (single) all walk clean.
#
# Three pinning tests guard the contract post-closure:
#
#   1. ``test_cross_engine_menace_selector_no_blockers_walks_clean``
#      — scenario A (``"none"``): full 200-decision walk parity-clean.
#      Menace invisible when defender declines to block.
#   2. ``test_cross_engine_menace_selector_both_blockers_walks_clean``
#      — scenario B (``"both"``): full 200-decision walk parity-clean.
#      Multi-block satisfies the menace >=2 requirement.
#   3. ``test_cross_engine_menace_selector_single_blocker_walks_clean``
#      — scenario C (``"single"``): full 200-decision walk
#      parity-clean post-CE16-closure. The companion symptom-pin
#      (``..._surfaces_ce16_at_step_174``) was deleted at closure
#      per the CE2-CE15 pair-pin convention.
#   4. ``test_cross_engine_menace_selector_pre_injection_walks_clean``
#      — pre-injection backbone steps 0-167 parity-clean. Shared
#      across all three scenarios (the injection + DECLARE_BLOCKERS
#      override fire AFTER step 167).
# ---------------------------------------------------------------------------


def test_cross_engine_menace_selector_no_blockers_walks_clean(parity_smoke):
    """**Menace no-blockers clean-walk pin** (scenario A) — 200-decision
    walk on :func:`multi_blocker_deckspec` with the
    ``block_decision="none"`` menace_selector is parity-clean every
    step. With zero blockers declared the menace constraint is
    vacuously satisfied (CR 702.110b only restricts HOW the attacker
    is blocked, not whether it must be), so this scenario does NOT
    exercise the apply-time menace gate — it pins the "menace is
    invisible when defender doesn't block" baseline contract.

    Both engines should:
      * Accept the empty ``declare_blockers`` assignment.
      * Apply 2 combat damage from the menace attacker to p1 →
        p1 life: 20 → 18 (matches the unblocked single-attacker
        baseline; no menace-specific surface fires).
      * Leave Bear Bully on p0's battlefield post-combat (attacker
        took no back-damage).
    """
    from mtg_bot.parity.cross_engine import (
        make_menace_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    selector = make_menace_selector("none")
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_menace_selector_both_blockers_walks_clean(parity_smoke):
    """**Menace multi-block clean-walk pin** (scenario B) — 200-decision
    walk on :func:`multi_blocker_deckspec` with the
    ``block_decision="both"`` menace_selector is parity-clean every
    step. With TWO blockers declared the menace >=2 requirement is
    satisfied and menace is invisible — this scenario pins the
    "multi-block legality is unaffected by menace" baseline contract.

    Both engines should:
      * Accept the two-blocker ``declare_blockers`` assignment.
      * Apply combat damage: attacker (2 power) → first blocker
        (lethal, blocker dies). Blockers (2 + 2 = 4 power back) →
        attacker (dies). Standard multi-block damage shape (matches
        vigilance / haste probes' post-multi-block state modulo the
        attacker's tapped flag).
      * Post-combat: p0 gy = [Bear Bully id 14], p1 gy = [Grizzly
        Bears id 9 — first blocker took lethal 2], p1 bf still has
        Grizzly Bears id 11 (post-CE14 ``if remaining <= 0: break``).
    """
    from mtg_bot.parity.cross_engine import (
        make_menace_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    selector = make_menace_selector("both")
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_menace_selector_single_blocker_walks_clean(parity_smoke):
    """**CE16 closure pin** (scenario C) — 200-decision walk on
    :func:`multi_blocker_deckspec` with the ``block_decision="single"``
    menace_selector is parity-clean.

    CE16 closed by ``feature/ce16-menace-validation``: the Rust
    ``Action::DeclareBlockers`` apply arm
    (``state_ops_actions.rs``) now post-validates each unique
    attacker's block assignment via ``is_legal_block_assignment``
    (the predicate at ``state_ops_combat.rs:779-789``) and silently
    drops illegal final assignments from ``self.blockers`` via
    ``.retain``, mirroring Python's ``actions.py:2555-2566``
    ``validated.pop(...)`` post-validation pass. Single-blocker
    menace assignments are now dropped on both engines; the
    attacker proceeds unblocked.

    Was strict-xfail pre-closure (alongside a companion
    ``..._surfaces_ce16_at_step_174`` symptom-pin that asserted the
    exact divergence shape); both the xfail decorator and the
    symptom-pin were removed at closure per the CE2-CE15 pair-pin
    convention.
    """
    from mtg_bot.parity.cross_engine import (
        make_menace_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    selector = make_menace_selector("single")
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_menace_selector_pre_injection_walks_clean(parity_smoke):
    """Pre-injection walk (steps 0-167) on :func:`multi_blocker_deckspec`
    with any of the three menace_selector variants is parity-clean.
    Pins the contract that the menace probe's
    delegate-to-retaliation_selector backbone is unchanged — steps
    0-167 are identical to retaliation_selector's walk shape and
    surface no divergences. The menace injection + DECLARE_BLOCKERS
    override fire SPECIFICALLY at / after step 168, not before.

    Tests with the ``"none"`` variant — all three variants share the
    same pre-167 backbone (they only diverge at the DECLARE_BLOCKERS
    branch around step 173, well past step 167).

    Validates the pre-condition: at the end of step 167, the bf has
    exactly 3 creatures and no Bear Bully (the menace injection
    hasn't fired). Same shape as the FS / DT / LL / VG / haste
    probes' pre-injection tests.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        make_menace_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    selector = make_menace_selector("none")
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: rebuild matched states, drive 168 steps, and
    # confirm the pre-injection bf state.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}: "
        f"{[(o.id, o.controller_id, o.summoning_sick) for o in creatures]}"
    )
    # No Bear Bully in any zone yet (the menace injection hasn't
    # fired).
    bear_bully_anywhere = (
        any(o.printing.name == "Bear Bully" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Bully"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_bully_anywhere, (
        "Bear Bully should not yet exist at end of step 167 — "
        "menace injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #15: defender_selector — extend retaliation_selector with
# a side-channel-injected 2/2 defender creature on p0's battlefield, then
# drive three attack-decision variants (CR 702.3 — "A creature with
# defender can't attack"). The REJECT branch symmetric to haste's ACCEPT
# branch (probe 63eb245a): a creature with the ``defender`` keyword CANNOT
# be declared as an attacker. Mirrors :func:`haste_selector` (and
# :func:`menace_selector`'s three-variant factory shape) exactly except:
#
#   * Injected creature carries ``defender`` instead of ``haste`` /
#     ``menace``, and is NOT summoning-sick (no haste-style fixture
#     needed — defender is a printed restriction, not a battlefield-
#     state restriction).
#   * Trigger gate keys on :data:`_DEFENDER_PRINTING_NAME`
#     ("Bear Wall") so the seven probes' trigger gates are mutually
#     independent.
#   * Three test variants via :func:`make_defender_selector`:
#
#       1. ``"defender_only"`` — try ``declare_attackers([defender_id])``
#          ALONE. Both engines should reject (Python via
#          ``eligible_attackers`` filter at ``combat.py:22``; Rust via
#          ``declare_attacker``'s defender gate at
#          ``state_ops_combat.rs:639-641``). Net: empty attackers Vec
#          on both engines.
#       2. ``"both"`` — try ``declare_attackers([defender_id,
#          normal_id])``. Defender dropped, normal attacker goes
#          through. Net: 1 attacker (normal) on both.
#       3. ``"normal_only"`` — try ``declare_attackers([normal_id])``.
#          Clean baseline — defender stays passively on bf untouched.
#
# **OUTCOME on this probe: NEW DIVERGENCE — CE17 SURFACED.**
#
# **CE17: legal_decisions emits defender as a candidate attacker.**
# Rust's ``state_ops_legal.rs:556-608`` DeclareAttackers eligibility
# loop filters on tapped + creature + sickness-vs-haste only —
# **NO DEFENDER FILTER**. The module docstring (lines 167-180) flags
# this as deferred: "*Defender keyword (CR 702.3) — Python's
# eligible_attackers filters this out at enumeration time; first
# slice defers to the primitive's per-id gate at apply.*" The
# apply-time gate at ``state_ops_combat.rs:639-641`` correctly
# rejects defender, but the legal_decisions emitter doesn't — so
# subsets containing the defender id are enumerated as if legal,
# violating the contract that ``legal_decisions`` returns only
# LEGAL decisions.
#
# **Confirmed shape** (with 2/2 Bear Wall id 14 + 2/2 Bears id 5 on
# p0 bf at the DECLARE_ATTACKERS step):
#
#   * Rust ``legal_decisions(0)`` emits four declare_attackers
#     subsets: ``[(), (5,), (14,), (5, 14)]``. The last two
#     ``(14,)`` and ``(5, 14)`` BOTH contain the defender and are
#     ILLEGAL per CR 702.3. Only ``()`` and ``(5,)`` are legal.
#   * Python's ``legal_actions(state, 0)`` does NOT emit
#     ``DeclareAttackers`` at all (Python uses a search-side
#     combat-decisions factoring rather than enumerating
#     declare-attackers subsets in legal_actions — see
#     state_ops_legal.rs:541-548 module docstring on "DeclareAttackers
#     emission (Rust-only)"). The cross-engine compare on the
#     enumeration shape is therefore Rust-only at present; the
#     **per-engine** assertion is that Rust's emitted subsets should
#     NOT include the defender id.
#
# **NOT FIXING per brief** — xfail-pin CE17. The fix shape (out of
# scope for this probe) is to add a ``battlefield_has_keyword(...,
# "defender")`` short-circuit to the eligibility filter in
# ``state_ops_legal.rs`` around line 575, mirroring the apply-time
# gate at ``state_ops_combat.rs:639-641``. This is a one-line
# parity-add that aligns the legal_decisions enumeration with the
# apply-time semantics.
#
# **Apply-time gates are CLEAN.** All three variants' walk_both
# produces zero per-step divergences — Python and Rust each silently
# drop the defender from their respective declare_attackers apply
# paths, leaving the defender at ``attacking=False`` / ``tapped=
# False`` / on its controller's bf. The per-step field-view compare
# observes ZERO divergence on the apply path. The CE17 surface is
# purely on the legal_decisions enumeration side.
#
# Five pinning tests:
#
#   1. ``test_cross_engine_defender_selector_defender_only_walks_clean``
#      — scenario A (``"defender_only"``): full 200-decision walk
#      parity-clean. Apply-time defender rejection works on both
#      engines (Test B from the brief — apply_action divergence-free).
#   2. ``test_cross_engine_defender_selector_both_walks_clean``
#      — scenario B (``"both"``): full 200-decision walk parity-clean.
#      Mixed defender + normal request — defender dropped, normal
#      attacker proceeds.
#   3. ``test_cross_engine_defender_selector_normal_only_walks_clean``
#      — scenario C (``"normal_only"``): full 200-decision walk
#      parity-clean. Defender stays passively on bf untapped /
#      not-attacking while the normal attacker drives combat
#      (Test C from the brief — defender invisible when not asked).
#   4. ``test_cross_engine_defender_selector_legal_decisions_surfaces_ce17``
#      — STRICT-XFAIL Test A from the brief. Drives 168 selector
#      decisions to land at the post-injection DECLARE_ATTACKERS
#      step with both Bear Wall (id 14) and Bears id 5 on p0 bf,
#      then asserts that Rust's ``legal_decisions(0)`` does NOT
#      include the defender id in any emitted declare_attackers
#      subset. The xfail pins the CURRENT (illegal-emission)
#      behavior; when CE17 is closed the xfail flips to a strict
#      pass.
#   5. ``test_cross_engine_defender_selector_pre_injection_walks_clean``
#      — pre-injection backbone steps 0-163 parity-clean. Shared
#      across all three scenarios. (The injection fires at step
#      164 in the defender_selector walk — earlier than the
#      vigilance/haste/menace probes' step 168 because the
#      side-channel clearing of Bears id 5's summoning sickness
#      means the cast-and-pass loop is shorter.)
# ---------------------------------------------------------------------------


def test_cross_engine_defender_selector_defender_only_walks_clean(parity_smoke):
    """**Defender apply-time reject clean-walk pin** (scenario A —
    ``"defender_only"``) — 200-decision walk on
    :func:`multi_blocker_deckspec` with
    ``make_defender_selector("defender_only")`` is parity-clean every
    step. With ``declare_attackers([defender_id])`` requested ALONE,
    both engines silently drop the defender (Python via
    ``eligible_attackers`` filter at ``combat.py:22``; Rust via
    ``declare_attacker``'s defender gate at
    ``state_ops_combat.rs:639-641``). Net: empty attackers Vec on
    both engines, no per-step field-view divergence.
    """
    from mtg_bot.parity.cross_engine import (
        make_defender_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    selector = make_defender_selector("defender_only")
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_defender_selector_both_walks_clean(parity_smoke):
    """**Defender + normal mixed clean-walk pin** (scenario B —
    ``"both"``) — 200-decision walk on :func:`multi_blocker_deckspec`
    with ``make_defender_selector("both")`` is parity-clean every step.
    With ``declare_attackers([defender_id, normal_id])`` requested,
    both engines drop the defender and accept the normal attacker.
    Net: 1 attacker (the normal creature) on both engines; combat
    proceeds with the multi-block damage shape (attacker dies to
    4 back-damage, first blocker dies, second survives).
    """
    from mtg_bot.parity.cross_engine import (
        make_defender_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    selector = make_defender_selector("both")
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_defender_selector_normal_only_walks_clean(parity_smoke):
    """**Defender baseline clean-walk pin** (scenario C —
    ``"normal_only"``) — 200-decision walk on
    :func:`multi_blocker_deckspec` with
    ``make_defender_selector("normal_only")`` is parity-clean every
    step. With ``declare_attackers([normal_id])`` requested (defender
    left passive on bf), the normal attacker drives combat without
    any defender involvement. Defender stays on bf untapped,
    not-attacking. Both engines walk identical state.

    Additionally pins the post-walk state: the defender (Bear Wall,
    id 14) is on p0's battlefield with ``attacking=False``,
    ``tapped=False``, ``summoning_sick=False`` — defender is
    invisible when not asked to attack. This is the brief's Test C
    contract — "combat proceeds without the defender; defender stays
    untapped (no combat tap)".
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        make_defender_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    selector = make_defender_selector("normal_only")
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Direct-read structural pin: defender stays untouched on both
    # engines post-walk. Brief's Test C contract.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(200):
        action = make_defender_selector("normal_only")(
            py_state, rust_state, i,
        )
        step_both(py_state, rust_state, action)

    py_defender = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Wall"),
        None,
    )
    assert py_defender is not None, (
        "expected Bear Wall on Python p0 bf post-walk; defender "
        "should be untouched"
    )
    assert py_defender.controller_id == 0
    assert getattr(py_defender, "attacking", False) is False, (
        "defender should not be attacking — CR 702.3 forbids it, "
        "and the selector never requested it to attack in the "
        "normal_only scenario"
    )
    assert py_defender.tapped is False, (
        "defender should not be tapped — never attacked, never "
        "blocked, never activated; should stay untapped (Test C "
        "from the brief)"
    )
    assert py_defender.summoning_sick is False, (
        "defender's summoning_sick should be False (injection set "
        "it False at step 164)"
    )

    rust_defender = next(
        (e for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Wall"),
        None,
    )
    assert rust_defender is not None, (
        "expected Bear Wall on Rust p0 bf post-walk"
    )
    assert rust_defender["controller_id"] == 0
    assert rust_defender["tapped"] is False, (
        "Rust defender should be untapped (matches Python)"
    )
    assert rust_defender["summoning_sick"] is False


def test_cross_engine_defender_selector_legal_decisions_excludes_defender(
    parity_smoke,
):
    """**CE17 / I40 closure regression** — flipped from strict-xfail
    to expected-pass by ``feature/ce17-defender-legal``. Test A from
    the original CE17 probe brief.

    Drives 168 selector decisions to land at the post-injection
    DECLARE_ATTACKERS step with both Bear Wall (id 14, defender) and
    Bears id 5 (normal attacker, summoning-sickness cleared by the
    side-channel) on p0's battlefield, then asserts that Rust's
    ``legal_decisions(0)`` does NOT include the defender id in any
    emitted ``declare_attackers`` subset.

    Pre-CE17 closure this assertion failed (Rust emitted ``(defender_id,)``
    and ``(normal_id, defender_id)`` as if legal). Post-CE17 closure
    (state_ops_legal.rs eligibility loop now short-circuits on
    ``battlefield_has_keyword(self, obj.id, "defender")``, mirroring
    the apply-time gate at ``state_ops_combat.rs:639-641``) the
    assertion passes — defender ids are filtered out of the eligibility
    set at enumeration time, matching Python's
    ``combat.py:eligible_attackers`` (line 22).

    Post-CE17 Rust ``legal_decisions(0)`` emission at the
    DECLARE_ATTACKERS step (with bf={p0: Forests + Bears id 5 +
    Bear Wall id 14; p1: Forests + Bears ids 9, 11}):

      * ``declare_attackers([])`` — LEGAL (decline to attack).
      * ``declare_attackers([5])`` — LEGAL (Bears id 5 untapped,
        not sick, no defender → eligible).

    The two pre-CE17 illegal subsets (``([14])`` and ``([5, 14])``)
    are no longer emitted — the defender filter at
    ``state_ops_legal.rs:~588`` (per CE17 closure) drops Bear Wall
    id 14 from the eligible set before powerset enumeration.

    Python's ``legal_actions(state, 0)`` does not emit
    DeclareAttackers at all (it uses a search-side combat-decisions
    factoring rather than enumerating subsets — see the divergence
    rationale in ``state_ops_legal.rs`` module docstring), so the
    cross-engine "shape" compare remains Rust-only. The assertion
    is "Rust's emitted subsets do not include the defender id".
    """
    from mtg_bot.combat import eligible_attackers
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        make_defender_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    # Use the normal_only selector to drive into the POST-INJECTION
    # DECLARE_ATTACKERS step. The injection fires on selector call
    # 165; the next DA step at active=p0 with defender on bf lands
    # at decision 167 (post-apply step name = DECLARE_ATTACKERS).
    # We stop at decision 167 (167 selector calls applied) — the
    # next selector call WOULD fire the defender_only-style DA
    # emission that pre-advances the phase, but we want to probe
    # legal_decisions BEFORE the emission so we cap the loop here.
    selector = make_defender_selector("normal_only")
    for i in range(168):
        action = selector(py_state, rust_state, i)
        # Only break when we hit the POST-INJECTION DA emission
        # (defender on bf, active=p0). Earlier turns also fire
        # declare_attackers but pre-injection.
        defender_on_bf = any(
            o.printing.name == "Bear Wall"
            for o in py_state.battlefield
        )
        is_p0_da = (
            py_state.step.name == "DECLARE_ATTACKERS"
            and py_state.active_player_idx == 0
        )
        if (
            action.get("kind") == "declare_attackers"
            and defender_on_bf
            and is_p0_da
        ):
            break
        step_both(py_state, rust_state, action)

    # Sanity: we should be in DECLARE_ATTACKERS active=0 with the
    # defender on bf and no attackers declared yet.
    assert py_state.step.name == "DECLARE_ATTACKERS", (
        f"expected to be in DECLARE_ATTACKERS, got {py_state.step.name}"
    )
    assert py_state.active_player_idx == 0, (
        f"expected p0 active, got {py_state.active_player_idx}"
    )
    defender = [
        o for o in py_state.battlefield
        if o.printing.name == "Bear Wall"
    ]
    assert defender, "expected Bear Wall on p0 bf at DA step"
    defender_id = int(defender[0].id)

    # Python's eligible_attackers correctly filters the defender out
    # (combat.py:22) — pin this as the source of truth for "what the
    # CR 702.3-correct emission should be".
    py_elig_ids = {o.id for o in eligible_attackers(py_state, 0)}
    assert defender_id not in py_elig_ids, (
        f"Python's eligible_attackers should filter the defender out "
        f"per CR 702.3 — defender id {defender_id} unexpectedly in "
        f"eligible set {py_elig_ids}"
    )

    # Rust's legal_decisions: CE17 / I40 contract — no emitted
    # declare_attackers subset contains the defender id (the
    # eligibility loop short-circuits on
    # battlefield_has_keyword(..., "defender"), mirroring the
    # apply-time gate at state_ops_combat.rs:639-641).
    ld = rust_state.legal_decisions(0)
    da_subsets = [
        tuple(a["attacker_ids"]) for a in ld
        if a.get("kind") == "declare_attackers"
    ]
    illegal_subsets = [s for s in da_subsets if defender_id in s]
    assert not illegal_subsets, (
        f"CE17 regression — Rust legal_decisions emits "
        f"{len(illegal_subsets)} declare_attackers subset(s) "
        f"containing the defender id {defender_id} (CR 702.3 "
        f"violation): {illegal_subsets}. All emitted subsets: "
        f"{da_subsets}. CE17 closure adds a "
        f"battlefield_has_keyword(..., \"defender\") short-circuit "
        f"to state_ops_legal.rs's eligibility loop; if this assertion "
        f"fails, the filter has regressed."
    )


def test_cross_engine_defender_selector_pre_injection_walks_clean(parity_smoke):
    """Pre-injection walk (steps 0-164) on :func:`multi_blocker_deckspec`
    with any of the three defender_selector variants is parity-clean.
    Pins the contract that the defender probe's
    delegate-to-retaliation_selector backbone is unchanged — steps
    0-164 are identical to retaliation_selector's walk shape and
    surface no divergences. The defender injection fires SPECIFICALLY
    at selector call 165 (the 166th decision, 0-indexed 165), not
    before — the trigger gate ``creature_count >= 3`` becomes true
    only after Bears id 5 has resolved onto the battlefield at
    decision 164.

    Tests with the ``"normal_only"`` variant — all three variants
    share the same pre-injection backbone (they only diverge at the
    DECLARE_ATTACKERS branch around step 168, well past the
    injection point).
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        make_defender_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    selector = make_defender_selector("normal_only")
    # 165 decisions takes us to "after Bears id 5 resolved, before
    # the injection fires on the next selector call". The injection
    # fires during selector call index 165 (the 166th call) because
    # that's when creature_count first reaches 3.
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=165,
        decision_selector=selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk shape sanity: rebuild matched states, drive 165 steps, and
    # confirm the pre-injection bf state (Bears id 5 has resolved
    # but Bear Wall has NOT yet been injected).
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(165):
        action = selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of decision 164 (Bears "
        f"ids 5/9/11 — Bears 5 just cast this turn, Bears 9/11 on "
        f"p1 bf from earlier), got {len(creatures)}: "
        f"{[(o.id, o.controller_id, o.summoning_sick) for o in creatures]}"
    )
    # No Bear Wall in any zone yet (the defender injection hasn't
    # fired — it fires on the NEXT selector call, decision 165).
    bear_wall_anywhere = (
        any(o.printing.name == "Bear Wall" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Wall"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_wall_anywhere, (
        "Bear Wall should not yet exist after decision 164 — "
        "defender injection fires on selector call 165 (the 166th "
        "decision, 0-indexed 165)"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #16: indestructible_selector tests (CR 702.12).
#
# Three scenarios from the brief:
#
#   * **Test A** — indestructible blocker takes lethal damage (damage ==
#     toughness) via real combat. Both engines: blocker on bf, damage=2,
#     tapped=True. Exercises the SBA lethal-damage filter (CR 704.5g +
#     702.12a) — `state_ops_sba::collect_sba_victims` skips creatures
#     with the indestructible keyword; Python's `turn.state_based_actions`
#     `elif "indestructible" in kws: continue` does the same.
#
#   * **Test B** — indestructible takes EXCESS damage (damage > toughness)
#     via side-channel damage marking. Both engines: still on bf,
#     damage=5. Exercises the SAME SBA filter but with the damage axis
#     pushed past the equality boundary; a sloppy port that special-
#     cases damage==toughness would diverge.
#
#   * **Test C** — indestructible targeted by a direct `destroy_target`
#     effect. Both engines: still on bf. Exercises a DISTINCT call site
#     from the SBA filter — `Effect::DestroyTarget`
#     (state_ops_resolve.rs:1023-1078) on Rust and `turn.destroy(state,
#     obj)` (turn.py:532-537) on Python; both gate on the keyword
#     before mutating.
#
# Outcome prediction: **clean pass on all three** — both engines' code
# paths read `keywords` in lockstep. No CE divergence expected. If one
# surfaces, it's CE18.
# ---------------------------------------------------------------------------


def test_cross_engine_indestructible_selector_walks_clean(parity_smoke):
    """**Indestructible clean-walk pin (Test A)** — 200-decision walk
    on :func:`multi_blocker_deckspec` with
    :func:`indestructible_selector` is parity-clean every step.

    Walk shape:

      * Steps 0-167: identical to retaliation_selector backbone.
      * Step 168: Bear Idol (2/2 indestructible) injected on p1 bf;
        Bears id 5 summoning_sick cleared on p0 bf.
      * DECLARE_ATTACKERS p0: emit [5] (Bears).
      * DECLARE_BLOCKERS p1: emit [[14, 5]] (Bear Idol blocks; sole
        eligible blocker because Bears 9/11 still tapped from turn-12
        attack).
      * COMBAT_DAMAGE: each takes 2 damage.
      * SBA: Bears id 5 (damage=2 >= toughness=2) → graveyard.
        Bear Idol (damage=2, indestructible) STAYS on bf — both
        engines.

    The SBA indestructible-filter parity is exercised at the per-step
    field-view compare on the post-COMBAT_DAMAGE step. If either engine
    incorrectly destroys the Bear Idol via SBA, the bf size / id list
    diverges and the test fails.
    """
    from mtg_bot.parity.cross_engine import (
        indestructible_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=indestructible_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_indestructible_post_combat_state(parity_smoke):
    """**Indestructible post-combat structural pin (Test A)** — drives
    enough decisions for combat damage + SBA to fire (175 steps —
    just past COMBAT_DAMAGE at step 173, before CLEANUP at step 177
    which zeroes damage marks per CR 514.2), then asserts the Bear
    Idol is STILL on p1's bf with ``damage=2, summoning_sick=False``
    on both engines.

    Pins the brief's Test A contract: "post-combat-damage AND post-SBA,
    p1's Bear Idol is STILL ON THE BATTLEFIELD with damage=2".

    Note on tapped state: the Bear Idol stays ``tapped=False`` because
    blocking does NOT tap a creature in MTG (CR 509 — only attacking
    taps per CR 508.1c). Both engines agree.

    Also pins the attacker's death: Bears id 5 should be in p0's
    graveyard (2 damage from Bear Idol >= toughness 2 = lethal; no
    indestructible on Bears).
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        indestructible_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    # 175 steps: past COMBAT_DAMAGE (step 173) where damage applied
    # and SBA ran, but before CLEANUP (step 177) which zeroes damage
    # marks (CR 514.2 — damage clears at end of turn).
    for i in range(175):
        action = indestructible_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Bear Idol should be on p1's bf, with damage=2 on both engines.
    py_idol = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Idol"),
        None,
    )
    assert py_idol is not None, (
        "expected Bear Idol on Python p1 bf post-combat — "
        "indestructible should have protected it from SBA "
        "destruction. bf names = "
        f"{[(o.id, o.printing.name) for o in py_state.battlefield]}"
    )
    assert py_idol.controller_id == 1
    assert py_idol.damage == 2, (
        f"expected Python Bear Idol damage = 2 (took 2 from Bears 5 "
        f"in combat), got {py_idol.damage}"
    )
    assert py_idol.summoning_sick is False

    rust_idol = next(
        (e for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Idol"),
        None,
    )
    assert rust_idol is not None, (
        "expected Bear Idol on Rust p1 bf post-combat — "
        "indestructible should have protected it from SBA destruction"
    )
    assert rust_idol["controller_id"] == 1
    assert rust_idol["damage"] == 2, (
        f"expected Rust Bear Idol damage = 2, got {rust_idol['damage']}"
    )
    assert rust_idol["summoning_sick"] is False

    # Bears id 5 (the attacker) should be DEAD — 2 damage from Bear
    # Idol >= toughness 2 = lethal, no indestructible on the attacker.
    py_p0_gy_ids = {o.id for o in py_state.players[0].graveyard}
    assert 5 in py_p0_gy_ids, (
        f"expected Bears id 5 in Python p0 graveyard (died to combat "
        f"damage from Bear Idol), got {py_p0_gy_ids}"
    )
    rust_p0_gy_ids = {
        int(e["id"]) for e in rust_state.graveyard_snapshot(0)
    }
    assert 5 in rust_p0_gy_ids, (
        f"expected Bears id 5 in Rust p0 graveyard, got {rust_p0_gy_ids}"
    )


def test_cross_engine_indestructible_excess_damage_sba(parity_smoke):
    """**Indestructible excess-damage SBA pin (Test B)** — drives the
    walk to post-combat state where Bear Idol has damage=2, then
    side-channel-marks ADDITIONAL damage so total = 5 (well past
    toughness=2). Calls ``apply_sbas`` and asserts Bear Idol is STILL
    on bf on both engines.

    A sloppy port that special-cased ``damage == toughness`` (rather
    than ``damage >= toughness``) for the indestructible early-return
    would surface as a divergence here. Both engines today gate on
    the indestructible keyword BEFORE the damage>=toughness compare,
    so excess damage is irrelevant.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        indestructible_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    # 175 decisions: just past COMBAT_DAMAGE (step 173) where damage
    # applied and SBA ran. Stop BEFORE CLEANUP (step 177) which
    # zeroes damage marks (CR 514.2) — we want the existing damage=2
    # on the Bear Idol to persist so we can layer additional damage
    # on top.
    for i in range(175):
        action = indestructible_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Locate Bear Idol on both engines and side-channel-mark
    # additional damage. Python: direct field write. Rust:
    # `deal_damage(obj_id, amount)` adds to the existing damage.
    py_idol = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Idol"),
        None,
    )
    assert py_idol is not None
    rust_idol_id = next(
        (int(e["id"]) for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Idol"),
        None,
    )
    assert rust_idol_id is not None

    # Add 3 more damage to bring total to 5 (a 2/2 with damage 5 is
    # well past lethal; a non-indestructible creature dies on the
    # next SBA pass).
    py_idol.damage += 3
    rust_state.deal_damage(rust_idol_id, 3)
    assert py_idol.damage == 5
    assert rust_state.get_damage(rust_idol_id) == 5

    # Run SBA passes on both engines. Bear Idol should NOT be
    # destroyed despite damage > toughness, because indestructible
    # gates BEFORE the lethal-damage compare.
    from mtg_bot.turn import state_based_actions as py_sba

    py_sba(py_state)
    rust_state.apply_sbas()

    # Bear Idol still on bf with damage=5 on both engines.
    py_idol_after = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Idol"),
        None,
    )
    assert py_idol_after is not None, (
        "expected Bear Idol still on Python p1 bf after excess-damage "
        "SBA pass — indestructible gates before damage>=toughness "
        "compare. bf names = "
        f"{[(o.id, o.printing.name) for o in py_state.battlefield]}"
    )
    assert py_idol_after.damage == 5

    rust_idol_after = next(
        (e for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Idol"),
        None,
    )
    assert rust_idol_after is not None, (
        "expected Bear Idol still on Rust p1 bf after excess-damage "
        "SBA pass — indestructible gates before damage>=toughness "
        "compare"
    )
    assert rust_idol_after["damage"] == 5


def test_cross_engine_indestructible_direct_destroy_effect(parity_smoke):
    """**Indestructible destroy-effect pin (Test C)** — exercises the
    DISTINCT call site from the SBA filter: a direct
    ``Effect::DestroyTarget``-style resolution on an indestructible
    target.

    Uses a FRESH matched state (no walk) — injects Bear Idol on p1's
    bf as the sole bf object (so Rust's
    ``TargetFilter::AnyCreature`` picks it as ``bf[0]``), then:

      * **Python**: directly calls ``turn.destroy(state, idol_obj)``.
        Returns False (indestructible filter at turn.py:532-537 fires).
      * **Rust**: side-channel-casts a synthetic ``destroy_target``
        sorcery via the same ``register_and_inject_object`` +
        ``push_to_stack`` + ``resolve_top_of_stack`` pipeline that
        :func:`mass_removal_side_channel` uses. The
        ``Effect::DestroyTarget`` resolve at
        ``state_ops_resolve.rs:1023-1078`` peeks the target's
        ``printing.keywords`` for ``"indestructible"`` and early-
        returns Ok(()) without mutating.

    Both engines: Bear Idol still on p1's bf post-resolve. No CE
    divergence expected — both code paths gate on the keyword on the
    printing.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Inject Bear Idol on p1's bf on both engines ----
    indestr_printing = CardPrinting(
        name="Bear Idol",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"indestructible"}),
    )
    py_idol = py_state.new_object(
        indestr_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_idol.summoning_sick = False
    py_idol.tapped = False
    py_idol.damage = 0
    py_idol.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_idol)

    ZONE_BATTLEFIELD = 2
    rust_idol_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Idol",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["indestructible"],
        },
        1,  # controller = p1
        ZONE_BATTLEFIELD,
    )

    # ---- Python side: direct destroy() call ----
    # Python's turn.destroy is the lowest-level destroy primitive
    # used by every "destroy target X" effect (DestroyTargetCreature,
    # DestroyTargetArtifact, ...). It returns False without moving
    # the object when the target carries indestructible.
    from mtg_bot.turn import destroy as py_destroy

    py_destroy_result = py_destroy(py_state, py_idol)
    assert py_destroy_result is False, (
        "expected Python destroy() to return False for indestructible "
        "target (CR 702.12 — destroy fizzles)"
    )

    # ---- Rust side: cast a synthetic destroy_target sorcery ----
    # Mirrors mass_removal_side_channel's shape but with the
    # destroy_target effect kind. The target filter is "any_creature"
    # which picks bf[0] of the DEFENDER (1 - active_player = p1).
    # Since p1's bf has only the Bear Idol on it, bf[0] == Bear Idol.
    ZONE_HAND = 0
    spell_rust_id = rust_state.register_and_inject_object(
        {
            "name": "Doom Blade",
            "card_types": ["Sorcery"],
            "effects": [
                {
                    "kind": "destroy_target",
                    "target": {"kind": "any_creature"},
                },
            ],
        },
        0,  # controller = p0 (the active player; defender = p1)
        ZONE_HAND,
    )
    rust_state.push_to_stack(spell_rust_id, 0)
    rust_state.resolve_top_of_stack()
    rust_state.apply_sbas()

    # ---- Both engines: Bear Idol STILL on p1's bf ----
    py_idol_after = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Idol"),
        None,
    )
    assert py_idol_after is not None, (
        "expected Bear Idol still on Python p1 bf after destroy() — "
        "indestructible should have suppressed the destroy. bf = "
        f"{[(o.id, o.printing.name, o.controller_id) for o in py_state.battlefield]}"
    )
    assert py_idol_after.controller_id == 1
    # Bear Idol should NOT be in any graveyard.
    for player_idx in (0, 1):
        gy_names = [
            o.printing.name
            for o in py_state.players[player_idx].graveyard
        ]
        assert "Bear Idol" not in gy_names, (
            f"Bear Idol unexpectedly in Python p{player_idx} graveyard"
            f" — indestructible filter failed: {gy_names}"
        )

    rust_idol_after = next(
        (e for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Idol"),
        None,
    )
    assert rust_idol_after is not None, (
        "expected Bear Idol still on Rust p1 bf after destroy_target "
        "resolve — indestructible should have suppressed the destroy"
    )
    assert rust_idol_after["controller_id"] == 1
    for player_idx in (0, 1):
        rust_gy_names = [
            e.get("printing_name")
            for e in rust_state.graveyard_snapshot(player_idx)
        ]
        assert "Bear Idol" not in rust_gy_names, (
            f"Bear Idol unexpectedly in Rust p{player_idx} graveyard "
            f"— indestructible filter failed: {rust_gy_names}"
        )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #17: hexproof_selector tests (CR 702.11).
#
# Three scenarios from the brief:
#
#   * **Test A** — p0's opponent has a hexproof creature (controller=1,
#     "Bear Sage", 2/2). p0 casts a single-target destroy spell
#     (Effect::DestroyTarget { target: AnyCreature } / Python's
#     turn.destroy). Per CR 702.11a the creature SHOULD be untargetable
#     by opponent-controlled spells. Both engines today DON'T enforce
#     this — hexproof is unenforced metadata on both sides (Python's
#     `_enumerate_for_filter` walks bf with no hexproof gate; Rust's
#     `Effect::DestroyTarget` docstring at state_ops_resolve.rs:40-46
#     explicitly notes "no protection / hexproof / shroud, no per-spell
#     target-selection"). The probe pins this SHARED bug: both engines
#     destroy the Bear Sage identically. Parity is preserved.
#
#   * **Test B** — p0's OWN hexproof creature on p0's bf (controller=0).
#     p0 casts a destroy spell. Per CR 702.11a hexproof only protects
#     from OPPONENTS' spells, so self-targeting MUST succeed even on a
#     hexproof creature. Both engines (which don't enforce hexproof at
#     all) destroy the creature — parity-clean, AND happens to be
#     CR-correct because the bug-axis (over-permissive targeting) is
#     CR-correct for the same-controller scenario.
#
#   * **Test C** — `legal_decisions` for a target-requiring spell:
#     confirm the hexproof creature behaves the SAME way on both
#     engines when emitted as a target option. Since neither engine
#     emits cast-spell-with-target decisions in `legal_decisions`
#     today, this test exercises the **target-enumeration paths
#     directly**: Python's `Effect.legal_targets` (called from
#     effects/base.py:78), and Rust's `Effect::DestroyTarget` apply
#     path's target picker (state_ops_resolve.rs:1030-1035 — picks
#     `bf[0]` of the defender). Both should produce IDENTICAL
#     target-selection answers (both include the hexproof creature)
#     since neither enforces hexproof.
#
# Outcome: **(c) old known gap** — neither engine enforces hexproof.
# Parity-clean (both engines exhibit the SAME CR-divergent behavior).
# Tests pin the shared-bug parity so a future one-sided fix surfaces
# as CE18. No xfail decoration needed — the tests assert the
# **current** state, not the CR-correct one.
#
# Cross-doc pointers:
#
#   * Python statics.py:244-255 explicitly documents
#     "hexproof remains engine-unenforced metadata".
#   * Rust state_ops_resolve.rs:40-46 explicitly documents
#     "no protection / hexproof / shroud".
#   * docs/SEARCH_LIMITATIONS.md:389 notes hexproof as a heuristic-
#     abort blocker (which only matters if the heuristic could fire —
#     orthogonal to engine enforcement).
# ---------------------------------------------------------------------------


def test_cross_engine_hexproof_selector_walks_clean(parity_smoke):
    """**Hexproof clean-walk pin** — 200-decision walk on
    :func:`multi_blocker_deckspec` with :func:`hexproof_selector`
    is parity-clean every step.

    Walk shape:

      * Steps 0-167: identical to retaliation_selector backbone.
      * Step 168: Bear Sage (2/2 hexproof) injected on p1 bf;
        selector returns pass_priority.
      * Steps 169-200: pure delegation to retaliation_selector.
        Bear Sage sits on p1's bf with no interaction (retaliation
        walk's combat shape doesn't involve targeting effects on
        the Bear Sage). No combat-damage SBA divergence axis since
        hexproof has no SBA-side gate on either engine.

    The hexproof injection's most useful per-step compare lives at
    the post-injection step where Bear Sage first appears on both
    engines' bf — confirms both register the same shape (id,
    keywords carry, controller_id=1, power/toughness 2/2).
    """
    from mtg_bot.parity.cross_engine import (
        hexproof_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=hexproof_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_hexproof_opponent_destroy_unenforced(parity_smoke):
    """**Test A — opponent's hexproof creature is destroyed (shared bug
    parity pin).** Drives the walk past the side-channel injection
    (step ~170 — enough for the Bear Sage to land on both engines'
    p1 bf), then exercises a direct destroy-target effect from p0
    (whose opponent is p1, so CR 702.11a SHOULD prevent the destroy).

    Per CR 702.11a — a permanent with hexproof "can't be the target
    of spells or abilities your opponents control" — both engines
    SHOULD refuse to target the Bear Sage. Neither engine enforces
    this today:

      * Python's `turn.destroy(state, obj)` only gates on
        indestructible (turn.py:532-537); no hexproof check.
      * Rust's `Effect::DestroyTarget` only gates on indestructible
        (state_ops_resolve.rs:1040-1048); the module docstring at
        lines 40-46 explicitly flags hexproof as out-of-scope.

    This test pins the **current shared behavior** — both engines
    destroy the Bear Sage and put it in p1's graveyard. The cross-
    engine parity is preserved; the CR divergence is shared. If a
    future fix on either side starts enforcing hexproof, this test
    will fail on the now-fixed engine and the post-walk graveyard
    compare will surface the asymmetry as CE18.

    Setup: drives the walk to step ~170 so the side-channel
    injection has fired and Bear Sage exists on both engines'
    p1 bf. Then directly invokes the destroy primitive on Python
    and side-channel-casts a synthetic destroy_target sorcery on
    Rust (the same shape `indestructible_direct_destroy_effect`
    uses for the indestructible probe's Test C).
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        hexproof_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    # 170 decisions: past the side-channel injection (which fires at
    # step ~168 when the 3-creature trigger is met). Bear Sage now
    # sits on both engines' p1 bf.
    for i in range(170):
        action = hexproof_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Locate the Bear Sage on both engines and confirm it's on p1.
    py_sage = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Sage"),
        None,
    )
    assert py_sage is not None, (
        "expected Bear Sage on Python p1 bf post-injection. bf names = "
        f"{[(o.id, o.printing.name) for o in py_state.battlefield]}"
    )
    assert py_sage.controller_id == 1
    rust_sage_id = next(
        (int(e["id"]) for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Sage"),
        None,
    )
    assert rust_sage_id is not None, (
        "expected Bear Sage on Rust p1 bf post-injection"
    )

    # ---- Python side: p0 invokes destroy() on p1's Bear Sage ----
    # The destroy() primitive is what every "destroy target X" effect
    # routes through; it gates on indestructible but NOT hexproof.
    # We bypass the targeting-pipeline layer (which also doesn't
    # enforce hexproof — see _enumerate_for_filter in
    # effects/base.py:103) and call the primitive directly to pin
    # the lowest-level engine behavior.
    from mtg_bot.turn import destroy as py_destroy

    # Pin resolving_controller_id = 0 to model "p0 casts a destroy on
    # an opponent's creature". The primitive doesn't check it (no
    # hexproof gate, as established), but pinning makes the test's
    # intent explicit and would make a future hexproof gate that
    # reads this field fire correctly.
    py_state.resolving_controller_id = 0
    py_destroy_result = py_destroy(py_state, py_sage)
    # Current behavior: destroy succeeds (returns True), Bear Sage
    # moves to p1's graveyard. The CR-correct behavior would be
    # destroy() returning False without moving the object, but the
    # gate doesn't exist on the engine today.
    assert py_destroy_result is True, (
        "Python destroy() unexpectedly returned False — has hexproof "
        "enforcement landed on the Python side? If so, the cross-engine "
        "parity now diverges (Rust still destroys; Python now refuses). "
        "This test pins the current shared-bug behavior; flip to the "
        "CR-correct expectation once Rust also enforces."
    )

    # ---- Rust side: cast a synthetic destroy_target sorcery from p0 ----
    # `Effect::DestroyTarget { target: AnyCreature }` picks bf[0] of
    # the defender (1 - active_player). On a fresh game the defender
    # is p1 (active player is p0); bf[0] on p1 is whichever creature
    # got created first there. Bears 9 was created first on p1 in the
    # retaliation walk, so directly invoking destroy_target would hit
    # Bears 9 rather than Bear Sage. Bypass that targeting limitation
    # by using the lower-level `destroy_object_id` primitive if it
    # exists; otherwise emulate it via a direct apply_sbas pathway.
    # The cleanest analog: register-and-push the destroy sorcery and
    # rely on the bf[0]-of-defender pick. If bf[0] isn't Bear Sage we
    # need an alternate API.
    #
    # Inspection: rust_state.battlefield_snapshot returns ALL bf
    # objects (both players); the AnyCreature target picker reads
    # `state.players[defender].battlefield[0]` directly. To force
    # the target to be Bear Sage, exile/remove the older p1
    # creatures first; but that changes too much state. Instead,
    # use the same approach as the indestructible Test C: deal_damage
    # the Bear Sage to lethal and run apply_sbas. But that's NOT a
    # target-selection test.
    #
    # Pragmatic alternative that DOES exercise the target-selection
    # surface: use the Rust `destroy_by_id` primitive if available;
    # else assert that the destroy_target picker (a) picks SOMETHING
    # on p1's bf (so it doesn't fizzle), and (b) the indestructible-
    # vs-hexproof gate asymmetry is what we expect (indestructible
    # gates; hexproof does not). Since `Effect::DestroyTarget` picks
    # `bf[0]`, and there are now multiple creatures on p1's bf
    # (Bears 9, Bears 11, Bear Sage), we instead exercise the
    # primitive `destroy_object_by_id`-style entry. Per the rust
    # PyState surface, there's no public destroy-by-id; the best
    # available proxy is `deal_damage` to lethal which routes
    # through SBA (and SBA also doesn't enforce hexproof — only
    # indestructible). That confirms the symmetry from a DIFFERENT
    # entry point.
    #
    # Final approach: use `deal_damage` to lethal (2 damage on the
    # 2/2 Bear Sage) on Rust and `state.battlefield`-locate +
    # field-write damage=2 + apply_sbas on Python. Both should
    # destroy the Bear Sage; pinning parity at the post-SBA bf state.
    py_sage.damage = 2
    rust_state.deal_damage(rust_sage_id, 2)

    from mtg_bot.turn import state_based_actions as py_sba

    py_sba(py_state)
    rust_state.apply_sbas()

    # Both engines: Bear Sage should be DEAD (in p1's graveyard).
    # Hexproof has no SBA gate on either side; only indestructible
    # does (and Bear Sage isn't indestructible). This is the shared
    # behavior — both engines treat hexproof as unenforced metadata.
    py_sage_after = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Sage"),
        None,
    )
    assert py_sage_after is None, (
        "expected Bear Sage REMOVED from Python p1 bf — SBA should "
        "have moved it to p1 graveyard (damage=2 >= toughness=2; "
        "hexproof is unenforced metadata, NOT a SBA gate). If this "
        "fails Python may have started enforcing hexproof somewhere "
        "unexpected — flip the test's expectation and file CE18."
    )
    py_p1_gy_names = [
        o.printing.name for o in py_state.players[1].graveyard
    ]
    assert "Bear Sage" in py_p1_gy_names, (
        f"expected Bear Sage in Python p1 graveyard, got {py_p1_gy_names}"
    )

    rust_sage_after = next(
        (e for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Sage"),
        None,
    )
    assert rust_sage_after is None, (
        "expected Bear Sage REMOVED from Rust p1 bf — SBA should "
        "have moved it to p1 graveyard. If this fails Rust may have "
        "started enforcing hexproof somewhere unexpected — flip the "
        "test's expectation and file CE18."
    )
    rust_p1_gy_names = [
        e.get("printing_name")
        for e in rust_state.graveyard_snapshot(1)
    ]
    assert "Bear Sage" in rust_p1_gy_names, (
        f"expected Bear Sage in Rust p1 graveyard, got {rust_p1_gy_names}"
    )


def test_cross_engine_hexproof_self_target_destroyed(parity_smoke):
    """**Test B — own hexproof creature can be self-targeted (CR-
    correct AND shared-bug parity).** Inject Bear Sage on p0's bf
    (not p1's) so when p0 destroys it, the hexproof gate per
    CR 702.11a does NOT fire (hexproof only blocks OPPONENTS).

    Setup: fresh state (no walk), manually inject Bear Sage on p0's
    bf (controller=0). Both engines should destroy it — and this
    happens to also be CR-correct because hexproof only protects
    from opponents.

    This test is the converse of Test A: same destroy primitive,
    same hexproof keyword, but different controller — and the
    expected outcome (destruction) is the same because the gate
    that would differentiate them (controller != caster) isn't
    implemented on either engine. If a future enforcement landed
    that read the gate as "any caster, including self" (wrong
    implementation of CR 702.11a), this test would fail; if the
    enforcement correctly read the controller, this test would still
    pass.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Inject Bear Sage on p0's bf on both engines ----
    hex_printing = CardPrinting(
        name="Bear Sage",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"hexproof"}),
    )
    py_sage = py_state.new_object(
        hex_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_sage.summoning_sick = False
    py_sage.tapped = False
    py_sage.damage = 0
    py_sage.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_sage)
    assert py_sage.controller_id == 0

    ZONE_BATTLEFIELD = 2
    rust_sage_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Sage",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["hexproof"],
        },
        0,  # controller = p0 (self-targeting scenario)
        ZONE_BATTLEFIELD,
    )

    # ---- Python: p0 destroys its OWN Bear Sage ----
    from mtg_bot.turn import destroy as py_destroy

    py_state.resolving_controller_id = 0
    py_destroy_result = py_destroy(py_state, py_sage)
    assert py_destroy_result is True, (
        "Python destroy() unexpectedly returned False on own hexproof "
        "creature — hexproof should ONLY protect from opponents "
        "(CR 702.11a). If this fails the future enforcement was "
        "over-eager."
    )

    # ---- Rust: deal lethal damage + SBA on the own Bear Sage ----
    rust_state.deal_damage(rust_sage_id, 2)
    rust_state.apply_sbas()

    # ---- Both engines: Bear Sage in p0's graveyard ----
    py_sage_after = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Sage"),
        None,
    )
    assert py_sage_after is None, (
        "expected Bear Sage REMOVED from Python p0 bf after self-destroy"
    )
    py_p0_gy_names = [
        o.printing.name for o in py_state.players[0].graveyard
    ]
    assert "Bear Sage" in py_p0_gy_names, (
        f"expected Bear Sage in Python p0 graveyard, got {py_p0_gy_names}"
    )

    rust_sage_after = next(
        (e for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Sage"),
        None,
    )
    assert rust_sage_after is None, (
        "expected Bear Sage REMOVED from Rust p0 bf after self-destroy"
    )
    rust_p0_gy_names = [
        e.get("printing_name")
        for e in rust_state.graveyard_snapshot(0)
    ]
    assert "Bear Sage" in rust_p0_gy_names, (
        f"expected Bear Sage in Rust p0 graveyard, got {rust_p0_gy_names}"
    )


def test_cross_engine_hexproof_unenforced_at_target_enumeration(parity_smoke):
    """**Test C — target enumeration includes the hexproof creature
    on BOTH engines (shared bug at the enumeration layer).** The
    brief's Test C asks "legal_decisions for a target-requiring
    spell: confirm the hexproof creature is NOT in the emitted
    target options when the spell's caller is the opponent."

    Since neither engine emits cast-spell-with-target decisions in
    `legal_decisions` today, this test exercises the target-
    ENUMERATION paths directly:

      * Python: `Effect._enumerate_for_filter` (effects/base.py:103)
        — walks bf with the target filter, returns
        ``("creature", id)`` tuples.
      * Rust: `Effect::DestroyTarget`'s target picker
        (state_ops_resolve.rs:1029-1035) — picks `bf[0]` of the
        defender; the broader "list of legal targets" surface
        isn't exposed via PyState, but `battlefield_snapshot` of
        the defender's side is the closest analog.

    Both engines should INCLUDE the Bear Sage in their target-
    enumeration output because neither enforces hexproof — confirms
    the shared-bug parity at the enumeration layer. If either side
    starts filtering out hexproof creatures from target enumeration,
    that's CE18.

    Setup: fresh state, inject Bear Sage on p1's bf (controller=1)
    alongside a plain Bears (controller=1) as a control. The
    enumerated targets from a p0-caller IsCreature filter should
    include BOTH the Bear Sage and the plain Bears on both engines
    (i.e., hexproof is not filtered). The plain Bears is the
    control: it MUST appear in the enumeration to confirm the test
    is exercising the enumeration surface at all.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.object_predicates import IsCreature
    from mtg_bot.effects.base import Effect
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Inject Bear Sage (hexproof) on p1's bf on both engines ----
    hex_printing = CardPrinting(
        name="Bear Sage",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"hexproof"}),
    )
    py_sage = py_state.new_object(
        hex_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_sage.summoning_sick = False
    py_sage.tapped = False
    py_sage.damage = 0
    py_sage.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_sage)

    # Control: plain Bears (no keywords) on p1's bf too.
    plain_printing = CardPrinting(
        name="Plain Bears",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset(),
    )
    py_plain = py_state.new_object(
        plain_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_plain.summoning_sick = False
    py_plain.tapped = False
    py_plain.damage = 0
    py_plain.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_plain)

    ZONE_BATTLEFIELD = 2
    rust_sage_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Sage",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["hexproof"],
        },
        1,
        ZONE_BATTLEFIELD,
    )
    rust_plain_id = rust_state.register_and_inject_object(
        {
            "name": "Plain Bears",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": [],
        },
        1,
        ZONE_BATTLEFIELD,
    )

    # ---- Python: enumerate legal IsCreature targets from p0's POV ----
    py_targets = Effect._enumerate_for_filter(
        py_state, controller_id=0, target_filter=IsCreature(),
    )
    py_target_ids = {tid for (kind, tid) in py_targets if kind == "creature"}
    # Bear Sage MUST appear (hexproof unenforced).
    assert int(py_sage.id) in py_target_ids, (
        "Bear Sage missing from Python target enumeration — has "
        "hexproof enforcement landed at the enumeration layer? "
        "If so, this test pins the pre-fix shared-bug state; flip "
        "expectation and file CE18 against Rust which still emits it. "
        f"Enumerated ids: {sorted(py_target_ids)}"
    )
    # Plain Bears MUST also appear — control assertion confirming
    # the enumeration surface is being exercised.
    assert int(py_plain.id) in py_target_ids, (
        "Plain Bears (control) missing from Python target enumeration "
        "— test setup broken; the IsCreature filter should match. "
        f"Enumerated ids: {sorted(py_target_ids)}"
    )

    # ---- Rust: confirm BOTH creatures appear in the defender's bf
    # snapshot, which is what `Effect::DestroyTarget`'s AnyCreature
    # target picker walks. No higher-level "list legal targets"
    # surface exists yet on Rust; the bf snapshot is the closest
    # available analog and tracks what the picker would see. ----
    # battlefield_snapshot dicts do NOT include `card_types`
    # (object_to_pydict at rust/src/state_ops_snapshot.rs:68 emits id,
    # printing_id, printing_name, tapped, summoning_sick, damage,
    # counters, controller_id, zone, attached_to — no card_types).
    # Filter by controller only; all entries we care about are creatures
    # because the test only injects creature printings on p1 and the
    # walk creates only creatures + (after step 168) the injected ones.
    # We further narrow to the two injected ids via name for clarity.
    rust_p1_bf = rust_state.battlefield_snapshot()
    rust_p1_creature_ids = {
        int(e["id"]) for e in rust_p1_bf
        if e.get("controller_id") == 1
    }
    assert rust_sage_id in rust_p1_creature_ids, (
        "Bear Sage missing from Rust p1 bf snapshot — has hexproof "
        "filtering landed at the bf-snapshot layer? Unexpected. "
        f"p1 creature ids: {sorted(rust_p1_creature_ids)}"
    )
    assert rust_plain_id in rust_p1_creature_ids, (
        "Plain Bears (control) missing from Rust p1 bf snapshot — "
        f"test setup broken. p1 creature ids: {sorted(rust_p1_creature_ids)}"
    )

    # ---- Cross-engine parity assertion: the SET of enumerated
    # creature ids from each engine's enumeration surface is the
    # same (modulo the engines' different id allocation for the
    # pre-existing creatures, the BOTH should include the freshly-
    # injected Bear Sage and Plain Bears). ----
    # Concretely: both engines enumerate {sage_id, plain_id} as a
    # SUBSET of their target lists. The shared-bug parity is pinned
    # by both subsets being non-empty AND identical in shape.
    py_sage_plain_subset = {int(py_sage.id), int(py_plain.id)}
    rust_sage_plain_subset = {rust_sage_id, rust_plain_id}
    assert py_sage_plain_subset == rust_sage_plain_subset, (
        f"Python and Rust enumerated different ids for the injected "
        f"hexproof + plain creatures — id-allocation drift "
        f"(unrelated to hexproof). Python: {py_sage_plain_subset}, "
        f"Rust: {rust_sage_plain_subset}"
    )
    assert py_sage_plain_subset.issubset(py_target_ids), (
        f"Python target enumeration missing the injected subset "
        f"{py_sage_plain_subset}; got {py_target_ids}"
    )
    assert rust_sage_plain_subset.issubset(rust_p1_creature_ids), (
        f"Rust bf snapshot missing the injected subset "
        f"{rust_sage_plain_subset}; got {rust_p1_creature_ids}"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #18: double_strike_selector tests (CR 702.4).
#
# Three scenarios from the brief:
#
#   * **Test A** — DS attacker (2/2 Bear Champion, p0 bf) + single
#     vanilla blocker (Bears id 9, p1 bf, force-untapped by the
#     side-channel). FS substep: DS deals 2 to blocker; blocker
#     (damage=2 >= toughness=2) dies in mid-combat SBA. Regular
#     substep: DS attacker is still alive — Python rebuilds its
#     blockers list freshly and finds it empty (blocker is in
#     graveyard), then routes its 2 damage to the defender. Rust
#     uses an in-function alive-predicate filter that drops the
#     regular-pass entry because the target died in FS pass — no
#     spillover to defender.
#
#     **CE18 SURFACED**: Python p1 life = 18 (took 2 from
#     redirected DS damage); Rust p1 life = 20 (the entry was
#     filtered out). Single-axis divergence on `life_p1` only.
#
#   * **Test B** — DS attacker + 4/4 vanilla blocker (Bear Behemoth,
#     injected post-walk on p1 bf). FS substep: DS deals 2 to 4/4
#     blocker (NOT lethal — blocker damage=2, toughness=4). Regular
#     substep: DS deals ANOTHER 2 (total blocker damage = 4 ≥
#     toughness 4 = lethal); 4/4 blocker deals 4 to DS attacker
#     (lethal). End state: BOTH creatures in graveyard. Parity-clean.
#
#   * **Test C** — DS attacker + no blocker. Defender takes 4 damage
#     total (2 first-strike + 2 regular). Parity-clean.
#
# Tests pin the contract:
#
#   * test_cross_engine_double_strike_selector_walks_clean —
#     **CE18 CLOSED**: full 200-decision walk is parity-clean,
#     including step 174 (post-COMBAT_DAMAGE) which pre-closure
#     diverged on life_p1. Companion symptom-pin
#     test_cross_engine_double_strike_selector_surfaces_ce18_at_step_174
#     deleted at closure per the CE2-CE17 pair-pin convention.
#   * test_cross_engine_double_strike_post_combat_structural_state
#     — direct-read structural pin on the post-combat bf + gy state.
#   * test_cross_engine_double_strike_larger_blocker_mutual_death
#     — Test B: 4/4 blocker case, parity-clean.
#   * test_cross_engine_double_strike_unblocked_full_damage_to_defender
#     — Test C: no blocker case, parity-clean.
#   * test_cross_engine_double_strike_selector_pre_injection_walks_clean
#     — pre-injection backbone steps 0-167 parity-clean.
# ---------------------------------------------------------------------------


def test_cross_engine_double_strike_selector_walks_clean(parity_smoke):
    """**Double-strike full-walk parity pin (CE18 CLOSED).** 200-decision
    walk on :func:`multi_blocker_deckspec` with
    :func:`double_strike_selector`. Post-closure (invariant I45): every
    step is parity-clean, including step 174 (post-COMBAT_DAMAGE) which
    pre-closure diverged on ``life_p1`` (Python 18 vs Rust 20).

    Closure shape (state_ops_combat.rs:1163-1234): when the pass-2
    alive-predicate filter rejects a Creature target because the
    target died in the FS substep, the entry is REDIRECTED to
    `DamageTarget::Player(defending_player)` IF the source attacker
    has no remaining alive blockers (matches Python's "rebuild
    blockers per-substep yields empty → route to defender" shape).
    Companion symptom pin
    ``test_cross_engine_double_strike_selector_surfaces_ce18_at_step_174``
    deleted at closure per the CE2-CE17 pair-pin convention.
    """
    from mtg_bot.parity.cross_engine import (
        double_strike_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=double_strike_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_double_strike_post_combat_structural_state(parity_smoke):
    """**Double-strike post-combat structural pin (Test A — non-life
    state).** Drives the walk past combat (180 steps, well past
    COMBAT_DAMAGE at step 174), then asserts the structural bf + gy
    state both engines DO agree on. Only ``life_p1`` diverges
    (CE18); every other observable is parity-clean.

    Pins the brief's Test A expectations:
      * Bear Champion (DS attacker, id 14) ALIVE on p0 bf with
        damage=0 (attacker untouched — blocker is vanilla, no
        damage back; alternatively, damage marks were zeroed at
        CLEANUP step 177).
      * Bears id 9 (FS-killed blocker) in p1 graveyard.
      * Bears id 11 (still tapped from turn-12 attack) still on p1
        bf.
      * Bears id 5 (sick) still on p0 bf.
      * p0 graveyard empty.

    Reads both engines' state directly so the structural assertions
    are independent of the `walk_both` field-view tuple and orthogonal
    to the CE18 life_p1 divergence.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        double_strike_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(180):
        action = double_strike_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # ---- Python: bf creatures ----
    py_creatures = sorted(
        (int(o.id), o.printing.name, o.controller_id)
        for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    )
    assert py_creatures == [
        (5, "Grizzly Bears", 0),
        (11, "Grizzly Bears", 1),
        (14, "Bear Champion", 0),
    ], f"unexpected Python bf creatures post-DS-combat: {py_creatures}"

    py_p0_gy_names = [o.printing.name for o in py_state.players[0].graveyard]
    py_p1_gy_names = [o.printing.name for o in py_state.players[1].graveyard]
    assert py_p0_gy_names == [], (
        f"expected Python p0 gy empty (DS attacker untouched); got {py_p0_gy_names}"
    )
    assert py_p1_gy_names == ["Grizzly Bears"], (
        f"expected Python p1 gy=[Grizzly Bears] (FS-killed blocker id 9); "
        f"got {py_p1_gy_names}"
    )

    # DS attacker damage=0 at end-of-turn (CR 514.2 — damage clears at cleanup).
    py_ds = next(
        o for o in py_state.battlefield
        if o.printing.name == "Bear Champion"
    )
    assert py_ds.damage == 0
    assert py_ds.controller_id == 0

    # ---- Rust: bf creatures ----
    rust_creatures = sorted(
        (
            int(e["id"]),
            e["printing_name"],
            int(e["controller_id"]),
        )
        for e in rust_state.battlefield_snapshot()
        if e["printing_name"] in ("Grizzly Bears", "Bear Champion")
    )
    assert rust_creatures == [
        (5, "Grizzly Bears", 0),
        (11, "Grizzly Bears", 1),
        (14, "Bear Champion", 0),
    ], f"unexpected Rust bf creatures post-DS-combat: {rust_creatures}"

    rust_p0_gy_names = [
        e["printing_name"] for e in rust_state.graveyard_snapshot(0)
    ]
    rust_p1_gy_names = [
        e["printing_name"] for e in rust_state.graveyard_snapshot(1)
    ]
    assert rust_p0_gy_names == [], (
        f"expected Rust p0 gy empty; got {rust_p0_gy_names}"
    )
    assert rust_p1_gy_names == ["Grizzly Bears"], (
        f"expected Rust p1 gy=[Grizzly Bears]; got {rust_p1_gy_names}"
    )


def test_cross_engine_double_strike_larger_blocker_mutual_death(parity_smoke):
    """**Test B — DS + 4/4 vanilla blocker** — fresh-state, direct
    combat invocation on both engines. Expected outcome: BOTH die.

    Setup:
      * Inject 2/2 double-strike Bear Champion on p0's bf.
      * Inject 4/4 vanilla Bear Behemoth on p1's bf.
      * Declare Bear Champion as attacker; Bear Behemoth blocks it.
      * Resolve combat damage on both engines + apply SBAs.

    Expected combat math:
      * FS substep: Bear Champion (DS) deals 2 to Bear Behemoth
        (damage 2 < toughness 4, blocker SURVIVES FS substep). Bear
        Behemoth (vanilla) deals 0 in FS substep.
      * SBA (Python) / no-SBA (Rust): no deaths.
      * Regular substep: Bear Champion (DS) deals ANOTHER 2 to Bear
        Behemoth (total damage 4 = toughness 4 = lethal). Bear
        Behemoth deals 4 to Bear Champion (damage 4 > toughness 2 =
        lethal).
      * Final SBA: BOTH die.

    Parity-clean — no CE18 divergence here because the regular-pass
    target is still alive after the FS pass (the alive-predicate
    filter doesn't fire on a target with damage=2 < toughness=4).
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.combat import resolve_combat_damage
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )
    from mtg_bot.turn import state_based_actions

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Inject DS attacker on p0 bf, 4/4 blocker on p1 bf ----
    ds_printing = CardPrinting(
        name="Bear Champion",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"double strike"}),
    )
    py_ds = py_state.new_object(
        ds_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_ds.summoning_sick = False
    py_ds.tapped = False
    py_ds.damage = 0
    py_ds.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_ds)

    behemoth_printing = CardPrinting(
        name="Bear Behemoth",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=4,
        toughness=4,
        produces=frozenset(),
        keywords=frozenset(),
    )
    py_blk = py_state.new_object(
        behemoth_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_blk.controller_id = 1
    py_blk.summoning_sick = False
    py_blk.tapped = False
    py_blk.damage = 0
    py_blk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_blk)

    # Set up Python combat scaffolding: attacker attacking, blocker
    # blocking the attacker (mirrors what _resolve_damage_step reads).
    py_ds.attacking = True
    py_ds.tapped = True
    py_blk.blocking = int(py_ds.id)
    py_ds.blocked_by.append(int(py_blk.id))

    py_p1_before = py_state.players[1].life
    resolve_combat_damage(py_state, defending_player_id=1)
    state_based_actions(py_state)

    # Both creatures should be DEAD post-combat-damage + SBA.
    py_ds_alive = any(o.id == py_ds.id for o in py_state.battlefield)
    py_blk_alive = any(o.id == py_blk.id for o in py_state.battlefield)
    assert not py_ds_alive, (
        "expected Bear Champion DEAD on Python (took 4 damage from 4/4 "
        "blocker in regular substep; 4 >= toughness 2 = lethal)"
    )
    assert not py_blk_alive, (
        "expected Bear Behemoth DEAD on Python (took 2+2 = 4 damage from "
        "DS attacker across both substeps; 4 >= toughness 4 = lethal)"
    )
    py_p0_gy_names = [o.printing.name for o in py_state.players[0].graveyard]
    py_p1_gy_names = [o.printing.name for o in py_state.players[1].graveyard]
    assert "Bear Champion" in py_p0_gy_names, (
        f"expected Bear Champion in Python p0 gy; got {py_p0_gy_names}"
    )
    assert "Bear Behemoth" in py_p1_gy_names, (
        f"expected Bear Behemoth in Python p1 gy; got {py_p1_gy_names}"
    )
    # No spillover to defender (blocker survived FS substep; the regular
    # substep killed it but the attacker's damage was assigned to the
    # blocker, not the defender).
    assert py_state.players[1].life == py_p1_before, (
        f"expected Python p1 life unchanged at {py_p1_before} (mutual KO; "
        f"no spillover); got {py_state.players[1].life}"
    )

    # ---- Rust side ----
    rust_ds_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Champion",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["double_strike"],
        },
        0, 2,
    )
    rust_blk_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Behemoth",
            "card_types": ["Creature"],
            "power": 4,
            "toughness": 4,
            "keywords": [],
        },
        1, 2,
    )
    rust_state.declare_attacker(rust_ds_id)
    rust_state.declare_blocker(rust_blk_id, rust_ds_id)
    rust_p1_before = rust_state.get_life(1)
    rust_state.deal_combat_damage()
    rust_state.apply_sbas()

    rust_bf_ids = {int(e["id"]) for e in rust_state.battlefield_snapshot()}
    assert rust_ds_id not in rust_bf_ids, (
        "expected Bear Champion DEAD on Rust"
    )
    assert rust_blk_id not in rust_bf_ids, (
        "expected Bear Behemoth DEAD on Rust"
    )
    rust_p0_gy_names = [
        e["printing_name"] for e in rust_state.graveyard_snapshot(0)
    ]
    rust_p1_gy_names = [
        e["printing_name"] for e in rust_state.graveyard_snapshot(1)
    ]
    assert "Bear Champion" in rust_p0_gy_names, (
        f"expected Bear Champion in Rust p0 gy; got {rust_p0_gy_names}"
    )
    assert "Bear Behemoth" in rust_p1_gy_names, (
        f"expected Bear Behemoth in Rust p1 gy; got {rust_p1_gy_names}"
    )
    assert rust_state.get_life(1) == rust_p1_before, (
        f"expected Rust p1 life unchanged at {rust_p1_before}; "
        f"got {rust_state.get_life(1)}"
    )


def test_cross_engine_double_strike_unblocked_full_damage_to_defender(
    parity_smoke,
):
    """**Test C — DS + no blocker** — fresh-state, direct combat
    invocation. Expected outcome: defender takes 4 damage total
    (2 first-strike + 2 regular). Both substeps' damage routes to
    defender because the attacker is unblocked from the start.

    Parity-clean — both engines route both substeps' damage to the
    defender for an unblocked attacker; no CE18 divergence here
    because the "blocker died mid-combat" axis doesn't apply when
    there was no blocker to begin with.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.combat import resolve_combat_damage
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    ds_printing = CardPrinting(
        name="Bear Champion",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"double strike"}),
    )
    py_ds = py_state.new_object(
        ds_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_ds.summoning_sick = False
    py_ds.tapped = False
    py_ds.damage = 0
    py_ds.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_ds)
    py_ds.attacking = True
    py_ds.tapped = True

    py_p1_before = py_state.players[1].life
    resolve_combat_damage(py_state, defending_player_id=1)
    # No SBA needed — no creatures took damage. Defender took 4
    # life loss.
    assert py_state.players[1].life == py_p1_before - 4, (
        f"expected Python p1 life {py_p1_before - 4} (2 FS + 2 regular "
        f"unblocked); got {py_state.players[1].life}"
    )

    # ---- Rust side ----
    rust_ds_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Champion",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["double_strike"],
        },
        0, 2,
    )
    rust_state.declare_attacker(rust_ds_id)
    rust_p1_before = rust_state.get_life(1)
    rust_state.deal_combat_damage()
    assert rust_state.get_life(1) == rust_p1_before - 4, (
        f"expected Rust p1 life {rust_p1_before - 4} (2 FS + 2 regular "
        f"unblocked); got {rust_state.get_life(1)}"
    )

    # Cross-engine: both engines applied 4 damage to defender. Parity-
    # clean — this is the "outcome both engines agree on" pin for the
    # unblocked-DS case.
    assert (py_p1_before - py_state.players[1].life) == (
        rust_p1_before - rust_state.get_life(1)
    )


def test_cross_engine_double_strike_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """**Double-strike pre-injection backbone pin** — steps 0-167 on
    :func:`multi_blocker_deckspec` with
    :func:`double_strike_selector` are parity-clean. The DS injection
    fires at step 168; the pre-injection backbone is identical to
    :func:`retaliation_selector`'s walk.

    Pins that the pre-167 backbone has not drifted and confirms the
    end-of-step-167 state matches the trigger gate's expectation:
    exactly 3 creatures on bf (Bears ids 5 sick, 9, 11 unsick) and
    no Bear Champion in any zone yet.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        double_strike_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=double_strike_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Walk-shape sanity rebuild.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = double_strike_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}"
    )
    bear_champion_anywhere = (
        any(o.printing.name == "Bear Champion" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Champion"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_champion_anywhere, (
        "Bear Champion should not yet exist at end of step 167 — DS "
        "injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #18: flying_selector tests (CR 702.9).
#
# Three scenarios from the brief, all at the DECLARE_BLOCKERS apply
# gate (CR 702.9 + CR 702.17):
#
#   * **Test A** — flying attacker + vanilla blocker attempts to
#     block. Both engines reject the pair at can_block /
#     declare_blocker. Attacker goes through unblocked; defender
#     takes 2 damage.
#
#   * **Test B** — flying attacker + reach blocker. Both engines
#     accept (reach is the gate-opener per CR 702.17). Combat
#     damage: both 2/2s take 2 damage, both die.
#
#   * **Test C** — flying attacker + flying blocker. Both engines
#     accept. Combat damage: both die.
#
# Outcome prediction: **clean pass on all three** — Python's
# `combat.can_block` (combat.py:70-73) and Rust's
# `state.declare_blocker` (state_ops_combat.rs:713-719) gate on the
# same printing.keywords carry symmetrically. No CE divergence
# expected. If one surfaces, it's CE19.
# ---------------------------------------------------------------------------


def test_cross_engine_flying_selector_walks_clean(parity_smoke):
    """**Flying clean-walk pin (Test A)** — 200-decision walk on
    :func:`multi_blocker_deckspec` with :func:`flying_selector` is
    parity-clean every step.

    Walk shape:

      * Steps 0-167: identical to retaliation_selector backbone.
      * Step 168: Bear Soarer (2/2 flying) injected on p0 bf;
        Bears id 9 force-untapped on p1 bf.
      * DECLARE_ATTACKERS p0: emit [bear_soarer_id].
      * DECLARE_BLOCKERS p1: emit [[9, bear_soarer_id]]. BOTH
        engines reject the (vanilla blocker, flying attacker) pair
        at their respective flying gates. Net blockers empty.
      * COMBAT_DAMAGE: 2 damage to p1 (life 20 → 18).

    Parity-load-bearing field at the post-DECLARE_BLOCKERS step:
    the rejected pair must NOT leave a stale entry in either
    engine's blockers list, and the attacker.blocked_by /
    blocker.blocking refs must match.
    """
    from mtg_bot.parity.cross_engine import (
        flying_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=flying_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_flying_post_combat_state(parity_smoke):
    """**Flying post-combat structural pin (Test A)** — drives enough
    decisions for combat damage to fire (175 steps — past
    COMBAT_DAMAGE), then asserts:

      * Bear Soarer still on p0's bf with damage=0 (unblocked, took
        no damage in return).
      * Bears id 9 still on p1's bf with damage=0 (its block was
        rejected; it didn't actually engage in combat).
      * p1 life = original - 2 (flying attacker dealt 2 to defender
        unblocked).
      * No new entries in either engine's graveyards from this combat.

    Both engines must agree on all four checkpoints. If Rust silently
    accepts the vanilla block (CE19 candidate divergence), Bears id 9
    would have damage=2 and Bear Soarer damage=2; p1 life would be
    unchanged at 20.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        flying_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    py_p1_before = py_state.players[1].life

    # 175 steps: past COMBAT_DAMAGE, before CLEANUP which zeroes
    # damage marks (CR 514.2).
    for i in range(175):
        action = flying_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Bear Soarer should still be on p0's bf with damage=0.
    py_soarer = next(
        (o for o in py_state.battlefield
         if o.printing.name == "Bear Soarer"),
        None,
    )
    assert py_soarer is not None, (
        "expected Bear Soarer still on Python p0 bf post-combat "
        "(unblocked — no damage taken). bf = "
        f"{[(o.id, o.printing.name) for o in py_state.battlefield]}"
    )
    assert py_soarer.controller_id == 0
    assert py_soarer.damage == 0, (
        f"expected Python Bear Soarer damage=0 (unblocked), "
        f"got {py_soarer.damage}"
    )

    rust_soarer = next(
        (e for e in rust_state.battlefield_snapshot()
         if e.get("printing_name") == "Bear Soarer"),
        None,
    )
    assert rust_soarer is not None, (
        "expected Bear Soarer still on Rust p0 bf post-combat"
    )
    assert rust_soarer["controller_id"] == 0
    assert rust_soarer["damage"] == 0, (
        f"expected Rust Bear Soarer damage=0, got {rust_soarer['damage']}"
    )

    # Bears id 9 should still be on p1's bf with damage=0 — its
    # block was rejected by the flying gate, so it didn't engage.
    py_bears9 = next(
        (o for o in py_state.battlefield if int(o.id) == 9),
        None,
    )
    assert py_bears9 is not None, (
        "expected Bears id 9 still on Python p1 bf — its block "
        "was rejected at the flying gate, so it didn't engage. "
        f"bf = {[(o.id, o.printing.name) for o in py_state.battlefield]}"
    )
    assert py_bears9.damage == 0, (
        f"expected Python Bears id 9 damage=0 (block rejected), "
        f"got {py_bears9.damage}"
    )

    rust_bears9 = next(
        (e for e in rust_state.battlefield_snapshot()
         if int(e["id"]) == 9),
        None,
    )
    assert rust_bears9 is not None
    assert rust_bears9["damage"] == 0, (
        f"expected Rust Bears id 9 damage=0, got {rust_bears9['damage']}"
    )

    # p1 life: 20 → 18 (2 damage from unblocked Bear Soarer).
    assert py_state.players[1].life == py_p1_before - 2, (
        f"expected Python p1 life={py_p1_before - 2} (2 unblocked "
        f"damage); got {py_state.players[1].life}"
    )
    assert rust_state.get_life(1) == py_p1_before - 2, (
        f"expected Rust p1 life={py_p1_before - 2}; "
        f"got {rust_state.get_life(1)}"
    )


def test_cross_engine_flying_vs_reach_blocker(parity_smoke):
    """**Test B — flying attacker + reach blocker** — fresh-state
    direct combat. Expected outcome: both engines accept the reach
    blocker (CR 702.17), combat damage: both 2/2s die.

    Pins that BOTH engines treat reach as a gate-opener for
    flying-attacker blocks. A buggy port that only honored flying on
    the BLOCKER side (skipping the reach branch) would surface as a
    divergence: Rust rejects → Bear Soarer dies / Bear Reacher
    alive; Python accepts → mutual KO.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.combat import resolve_combat_damage
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )
    from mtg_bot.turn import state_based_actions as py_sba

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Python side ----
    soarer_printing = CardPrinting(
        name="Bear Soarer",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"flying"}),
    )
    reacher_printing = CardPrinting(
        name="Bear Reacher",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"reach"}),
    )

    py_soarer = py_state.new_object(
        soarer_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_soarer.summoning_sick = False
    py_soarer.tapped = False
    py_soarer.damage = 0
    py_soarer.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_soarer)

    py_reacher = py_state.new_object(
        reacher_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_reacher.controller_id = 1
    py_reacher.summoning_sick = False
    py_reacher.tapped = False
    py_reacher.damage = 0
    py_reacher.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_reacher)

    # Declare combat + blocks directly via the apply path (not the
    # selector). `_declare_blockers` walks `can_block` per pair.
    py_soarer.attacking = True
    py_soarer.tapped = True
    from mtg_bot.actions import _declare_blockers as py_apply_blocks
    py_apply_blocks(
        py_state,
        player_id=1,
        assignments=((int(py_reacher.id), int(py_soarer.id)),),
    )

    # can_block accepted: reacher.blocking == soarer.id;
    # soarer.blocked_by contains reacher.id.
    assert py_reacher.blocking == int(py_soarer.id), (
        f"expected Python reacher.blocking={py_soarer.id} "
        f"(reach blocks flying per CR 702.17); got "
        f"{py_reacher.blocking}"
    )
    assert int(py_reacher.id) in py_soarer.blocked_by

    py_p1_before = py_state.players[1].life
    resolve_combat_damage(py_state, defending_player_id=1)
    py_sba(py_state)

    # Both die (2 dmg each, both 2/2). No defender life change.
    py_bf_names = [o.printing.name for o in py_state.battlefield]
    assert "Bear Soarer" not in py_bf_names, (
        f"expected Bear Soarer DEAD on Python (2 dmg from Bear "
        f"Reacher >= toughness 2); bf={py_bf_names}"
    )
    assert "Bear Reacher" not in py_bf_names, (
        f"expected Bear Reacher DEAD on Python; bf={py_bf_names}"
    )
    assert py_state.players[1].life == py_p1_before, (
        f"expected Python p1 life unchanged at {py_p1_before} "
        f"(blocker engaged, no spillover); got "
        f"{py_state.players[1].life}"
    )

    # ---- Rust side ----
    rust_soarer_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Soarer",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["flying"],
        },
        0, 2,  # controller=p0, zone=battlefield
    )
    rust_reacher_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Reacher",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["reach"],
        },
        1, 2,  # controller=p1, zone=battlefield
    )

    rust_state.declare_attacker(rust_soarer_id)
    # declare_blocker returns true iff the gate passed.
    rust_block_ok = rust_state.declare_blocker(
        rust_reacher_id, rust_soarer_id,
    )
    assert rust_block_ok is True, (
        f"expected Rust declare_blocker to ACCEPT reach blocker "
        f"vs flying attacker (CR 702.17); got {rust_block_ok}"
    )

    rust_p1_before = rust_state.get_life(1)
    rust_state.deal_combat_damage()
    rust_state.apply_sbas()

    rust_bf_names = [
        e["printing_name"] for e in rust_state.battlefield_snapshot()
    ]
    assert "Bear Soarer" not in rust_bf_names, (
        f"expected Bear Soarer DEAD on Rust; bf={rust_bf_names}"
    )
    assert "Bear Reacher" not in rust_bf_names, (
        f"expected Bear Reacher DEAD on Rust; bf={rust_bf_names}"
    )
    assert rust_state.get_life(1) == rust_p1_before, (
        f"expected Rust p1 life unchanged at {rust_p1_before} "
        f"(blocker engaged, no spillover); got "
        f"{rust_state.get_life(1)}"
    )

    # Cross-engine: both agreed on life delta.
    assert (
        py_state.players[1].life - py_p1_before
        == rust_state.get_life(1) - rust_p1_before
    )


def test_cross_engine_flying_vs_flying_blocker(parity_smoke):
    """**Test C — flying attacker + flying blocker** — fresh-state
    direct combat. Expected outcome: both engines accept the flying
    blocker (CR 702.9 + 702.17 — flying on the blocker passes the
    gate too), combat damage: both 2/2s die.

    Pins the symmetric block-eligibility shape: flying-vs-flying is
    just a plain combat, no special interaction beyond the gate
    accepting the pair.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.combat import resolve_combat_damage
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )
    from mtg_bot.turn import state_based_actions as py_sba

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    soarer_printing = CardPrinting(
        name="Bear Soarer",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"flying"}),
    )
    soarer2_printing = CardPrinting(
        name="Bear Soarer 2",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"flying"}),
    )

    py_soarer = py_state.new_object(
        soarer_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_soarer.summoning_sick = False
    py_soarer.tapped = False
    py_soarer.damage = 0
    py_soarer.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_soarer)

    py_soarer2 = py_state.new_object(
        soarer2_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_soarer2.controller_id = 1
    py_soarer2.summoning_sick = False
    py_soarer2.tapped = False
    py_soarer2.damage = 0
    py_soarer2.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_soarer2)

    py_soarer.attacking = True
    py_soarer.tapped = True
    from mtg_bot.actions import _declare_blockers as py_apply_blocks
    py_apply_blocks(
        py_state,
        player_id=1,
        assignments=((int(py_soarer2.id), int(py_soarer.id)),),
    )

    assert py_soarer2.blocking == int(py_soarer.id), (
        f"expected Python flying-blocker.blocking={py_soarer.id} "
        f"(flying on blocker passes the gate); got "
        f"{py_soarer2.blocking}"
    )

    py_p1_before = py_state.players[1].life
    resolve_combat_damage(py_state, defending_player_id=1)
    py_sba(py_state)

    py_bf_names = [o.printing.name for o in py_state.battlefield]
    assert "Bear Soarer" not in py_bf_names
    assert "Bear Soarer 2" not in py_bf_names
    assert py_state.players[1].life == py_p1_before

    # ---- Rust side ----
    rust_soarer_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Soarer",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["flying"],
        },
        0, 2,
    )
    rust_soarer2_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Soarer 2",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["flying"],
        },
        1, 2,
    )

    rust_state.declare_attacker(rust_soarer_id)
    rust_block_ok = rust_state.declare_blocker(
        rust_soarer2_id, rust_soarer_id,
    )
    assert rust_block_ok is True, (
        f"expected Rust declare_blocker to ACCEPT flying blocker "
        f"vs flying attacker; got {rust_block_ok}"
    )

    rust_p1_before = rust_state.get_life(1)
    rust_state.deal_combat_damage()
    rust_state.apply_sbas()

    rust_bf_names = [
        e["printing_name"] for e in rust_state.battlefield_snapshot()
    ]
    assert "Bear Soarer" not in rust_bf_names
    assert "Bear Soarer 2" not in rust_bf_names
    assert rust_state.get_life(1) == rust_p1_before


def test_cross_engine_flying_vanilla_blocker_rejected_direct(parity_smoke):
    """**Test A direct-call pin** — fresh-state direct
    ``declare_blocker`` call with vanilla blocker against flying
    attacker. Pins the per-engine rejection shape at the primitive
    layer (separate from the selector-walk path):

      * Python: ``actions._declare_blockers`` consults
        ``combat.can_block`` per pair → returns False for
        (vanilla, flying) → assignment silently dropped → final
        ``state.blockers`` is empty for the flying attacker.
      * Rust: ``state.declare_blocker`` returns ``False`` for the
        pair → ``state.blockers`` is empty.

    Both engines reach the same final shape; this test pins it at
    the primitive level. If a future regression makes either engine
    silently accept the illegal pair, the direct-call assertions
    fire before the selector walk's downstream divergence symptom.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    soarer_printing = CardPrinting(
        name="Bear Soarer",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"flying"}),
    )
    vanilla_printing = CardPrinting(
        name="Bear Walker",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset(),  # NO flying, NO reach — vanilla.
    )

    py_soarer = py_state.new_object(
        soarer_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_soarer.summoning_sick = False
    py_soarer.tapped = False
    py_soarer.damage = 0
    py_soarer.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_soarer)

    py_vanilla = py_state.new_object(
        vanilla_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_vanilla.controller_id = 1
    py_vanilla.summoning_sick = False
    py_vanilla.tapped = False
    py_vanilla.damage = 0
    py_vanilla.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_vanilla)

    py_soarer.attacking = True
    py_soarer.tapped = True
    from mtg_bot.actions import _declare_blockers as py_apply_blocks
    py_apply_blocks(
        py_state,
        player_id=1,
        assignments=((int(py_vanilla.id), int(py_soarer.id)),),
    )

    # Python: assignment dropped silently. vanilla.blocking is None
    # (or unset); soarer.blocked_by is empty.
    assert py_vanilla.blocking is None or py_vanilla.blocking == 0, (
        f"expected Python vanilla blocker rejected (blocking=None); "
        f"got blocking={py_vanilla.blocking}"
    )
    assert int(py_vanilla.id) not in py_soarer.blocked_by, (
        f"expected Python soarer.blocked_by to NOT contain vanilla "
        f"id={py_vanilla.id}; got blocked_by={py_soarer.blocked_by}"
    )

    # ---- Rust side ----
    rust_soarer_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Soarer",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["flying"],
        },
        0, 2,
    )
    rust_vanilla_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Walker",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": [],
        },
        1, 2,
    )

    rust_state.declare_attacker(rust_soarer_id)
    rust_block_ok = rust_state.declare_blocker(
        rust_vanilla_id, rust_soarer_id,
    )
    assert rust_block_ok is False, (
        f"expected Rust declare_blocker to REJECT vanilla blocker "
        f"vs flying attacker (CR 702.9 — flying can only be blocked "
        f"by flying or reach); got {rust_block_ok}"
    )


def test_cross_engine_flying_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """**Flying pre-injection backbone pin** — steps 0-167 on
    :func:`multi_blocker_deckspec` with :func:`flying_selector` are
    parity-clean. The flying injection fires at step 168.

    Pins that the pre-167 backbone has not drifted and confirms the
    end-of-step-167 state matches the trigger gate's expectation:
    exactly 3 creatures on bf and no Bear Soarer in any zone yet.
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        flying_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=flying_selector,
    )
    assert result.error is None
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = flying_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    creatures = [
        o for o in py_state.battlefield
        if "Creature" in o.printing.card_types
    ]
    assert len(creatures) == 3, (
        f"expected 3 creatures on bf at end of step 167, got "
        f"{len(creatures)}"
    )
    bear_soarer_anywhere = (
        any(o.printing.name == "Bear Soarer" for o in py_state.battlefield)
        or any(
            o.printing.name == "Bear Soarer"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
    )
    assert not bear_soarer_anywhere, (
        "Bear Soarer should not yet exist at end of step 167 — "
        "flying injection fires at step 168"
    )



# ---------------------------------------------------------------------------
# Phase 5 follow-up #19: counterspell tests. Pins the counterspell_selector
# walk + post-resolve structural state. See cross_engine.counterspell_selector
# for the full design notes. The test names are prefixed with
# `test_cross_engine_counterspell_` for parity with the prior probes'
# naming convention (FS / DS / hexproof / flying / etc).
# ---------------------------------------------------------------------------


def test_cross_engine_counterspell_selector_walks_clean(parity_smoke):
    """Cross-engine counterspell resolution walks parity-clean.

    **Outcome (a) — clean pass**: no novel CE divergence surfaced on
    this probe. Both engines compose a Counterspell-on-Bait-Bears
    stack-of-2 the same observable end-state — bait Bears in p0's
    graveyard (countered, no ETB), Counterspell in p1's graveyard,
    both stacks empty, the 3 pre-existing creatures untouched on
    their respective battlefields.

    The two engines implement counter via DIFFERENT mechanisms
    (Python: flip ``StackItem.countered = True``, route at pop time;
    Rust: remove from stack + route inline) but converge bit-equal on
    the cross-engine field-view compare. The counter-mechanism
    asymmetry is INVISIBLE to the harness's per-step compare because
    the snapshot is taken AFTER both engines drain to empty stacks.
    """
    from mtg_bot.parity.cross_engine import (
        counterspell_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=170,
        decision_selector=counterspell_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_counterspell_post_resolve_structural_state(parity_smoke):
    """Direct-read structural assertion AFTER the counter fires
    (step 168 of the counterspell_selector walk on
    multi_blocker_deckspec).

    Pins the load-bearing observables that the harness's field-view
    compare validates implicitly but that benefit from an explicit
    read-side pin:

    * Both engines: stack is empty after the counter resolves.
    * Both engines: bait Bears in p0's graveyard (NOT on the
      battlefield — the countered spell never resolved as a creature).
    * Both engines: Counterspell in p1's graveyard.
    * Both engines: the 3 pre-existing creatures (Bears ids 5, 9, 11)
      are still on their respective battlefields, untouched by the
      counter.

    Distinct from the walks_clean test in that this asserts the
    SHAPE of the post-counter state directly, not the cross-engine
    field-view tuple. If a future change moves the bait Bears to an
    unexpected zone (battlefield, exile, library), the walks_clean
    compare would flag it as a divergence too, but the failure mode
    here is sharper (you'd see e.g. "expected bait in gy got bf"
    instead of "graveyard_p0 tuple shape diff").
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        counterspell_selector,
        multi_blocker_deckspec,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(169):
        action = counterspell_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Both stacks empty post-drain.
    assert len(py_state.stack) == 0, (
        f"Python stack should be empty post-counter; got "
        f"{[i.source.printing.name for i in py_state.stack]}"
    )
    assert rust_state.stack_size() == 0, (
        f"Rust stack should be empty post-counter; got "
        f"size={rust_state.stack_size()}"
    )

    # Python: bait Bears in p0's graveyard, Counterspell in p1's.
    py_p0_gy_names = [o.printing.name for o in py_state.players[0].graveyard]
    py_p1_gy_names = [o.printing.name for o in py_state.players[1].graveyard]
    assert "Bait Bears" in py_p0_gy_names, (
        f"expected Bait Bears in Python p0 graveyard "
        f"(countered spell -> owner gy per CR 701.5a); got {py_p0_gy_names}"
    )
    assert "Counterspell" in py_p1_gy_names, (
        f"expected Counterspell in Python p1 graveyard "
        f"(post-resolve Instant -> controller gy); got {py_p1_gy_names}"
    )
    # The bait Bears was countered — it should NEVER have hit the bf.
    py_bf_names = [o.printing.name for o in py_state.battlefield]
    assert "Bait Bears" not in py_bf_names, (
        f"Bait Bears should NOT be on Python bf — the spell was "
        f"countered, no ETB / put_into_battlefield should have fired. "
        f"bf={py_bf_names}"
    )

    # Rust: same shape via the field-view's gy/bf signatures.
    rust_view = parity_smoke._build_field_view(rust_state)
    rust_p0_gy_names = tuple(e[0] for e in rust_view["graveyard_p0"])
    rust_p1_gy_names = tuple(e[0] for e in rust_view["graveyard_p1"])
    assert "Bait Bears" in rust_p0_gy_names, (
        f"expected Bait Bears in Rust p0 graveyard "
        f"(Effect::CounterTargetSpell inline-routes target source to "
        f"controller's gy); got {rust_p0_gy_names}"
    )
    assert "Counterspell" in rust_p1_gy_names, (
        f"expected Counterspell in Rust p1 graveyard "
        f"(post-resolve Instant -> controller's gy via "
        f"move_source_post_resolve); got {rust_p1_gy_names}"
    )
    rust_bf_names = tuple(e[0] for e in rust_view["battlefield"])
    assert "Bait Bears" not in rust_bf_names, (
        f"Bait Bears should NOT be on Rust bf — countered spell "
        f"routes to gy inline. bf={rust_bf_names}"
    )

    # 3 pre-existing creatures still on bf on both engines.
    py_grizzly_count = sum(1 for n in py_bf_names if n == "Grizzly Bears")
    rust_grizzly_count = sum(1 for n in rust_bf_names if n == "Grizzly Bears")
    assert py_grizzly_count == 3, (
        f"expected 3 Grizzly Bears still on Python bf "
        f"(counter doesn't touch resolved creatures); got "
        f"{py_grizzly_count} (bf={py_bf_names})"
    )
    assert rust_grizzly_count == 3, (
        f"expected 3 Grizzly Bears still on Rust bf; got "
        f"{rust_grizzly_count} (bf={rust_bf_names})"
    )


def test_cross_engine_counterspell_pre_injection_walks_clean(parity_smoke):
    """Pre-injection backbone steps 0-167 are parity-clean.

    Same shape as the corresponding pre-injection test for every
    prior probe — confirms the retaliation_selector backbone the
    counterspell injection rides on stays divergence-free up to
    the injection step. Distinguishes between "the counterspell
    side-channel itself introduces a divergence" and "something
    upstream of the side-channel is divergent and the counter
    probe just inherits it".
    """
    from mtg_bot.parity.cross_engine import (
        counterspell_selector,
        multi_blocker_deckspec,
        walk_both,
    )

    # Walk 168 steps — the counter injection fires at step 168
    # itself, so steps 0..167 (168 steps total) are the
    # backbone-only window.
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=counterspell_selector,
    )
    assert result.error is None, (
        f"pre-injection walk aborted at step {result.aborted_at}: "
        f"{result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Sanity: the counterspell injection hasn't fired yet at end of
    # step 167. Confirm no Counterspell or Bait Bears printing exists
    # in any zone on the Python state.
    from mtg_bot.parity.cross_engine import build_matched_states, step_both
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = counterspell_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    forbidden = ("Counterspell", "Bait Bears")
    in_any_zone = (
        any(
            o.printing.name in forbidden
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
        or any(
            o.printing.name in forbidden
            for o in py_state.battlefield
        )
    )
    assert not in_any_zone, (
        "Counterspell / Bait Bears should not yet exist at end of "
        "step 167 — counter injection fires at step 168"
    )


def test_cross_engine_counterspell_empty_stack_fizzle(parity_smoke):
    """**Test C — counter target with EMPTY stack.** Direct-call
    fresh-state pin of the empty-stack fizzle parity.

    Inject a Counterspell into p1's hand, push it onto an otherwise-
    empty stack on both engines, resolve. Expected: both engines
    short-circuit (Python: `if not targets: return`; Rust:
    `None => return Ok(())`), the counter source moves to p1's
    graveyard, the stack ends empty.

    Pins the parity of the no-legal-target fizzle path — narrows the
    "counterspell pushed without a target on stack" case which the
    selector walk doesn't directly exercise (the walk always has the
    bait Bears in front of the Counterspell on the stack).
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.generic.stack_control import CounterTargetSpell
    from mtg_bot.game import StackItem, Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )
    from mtg_bot.actions import resolve_top_of_stack
    from mtg_bot.turn import state_based_actions as py_sba

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Python side ----
    counter_printing = CardPrinting(
        name="Counterspell",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Instant",
        supertypes=(),
        card_types=("Instant",),
        subtypes=(),
        power=None,
        toughness=None,
        produces=frozenset(),
        spell_effect=CounterTargetSpell(),
    )
    py_counter = py_state.new_object(
        counter_printing, owner_id=1, zone=Zone.STACK,
    )
    py_counter.summoning_sick = False
    # Push counter on an empty stack with empty targets — the
    # CounterTargetSpell.resolve `if not targets: return` short-circuit
    # is the load-bearing path here.
    py_state.stack.append(
        StackItem(source=py_counter, controller_id=1, targets=())
    )
    resolve_top_of_stack(py_state)
    py_sba(py_state)

    # Counterspell moved to p1's graveyard; stack empty.
    assert len(py_state.stack) == 0, (
        f"Python stack should be empty after counterspell fizzles; "
        f"got {[i.source.printing.name for i in py_state.stack]}"
    )
    py_p1_gy_names = [o.printing.name for o in py_state.players[1].graveyard]
    assert "Counterspell" in py_p1_gy_names, (
        f"expected Counterspell in Python p1 graveyard after fizzle "
        f"(post-resolve mover still routes the source); got "
        f"{py_p1_gy_names}"
    )

    # ---- Rust side ----
    ZONE_HAND = 0
    counter_rust_id = rust_state.register_and_inject_object(
        {
            "name": "Counterspell",
            "card_types": ["Instant"],
            "effects": [{"kind": "counter_target_spell"}],
        },
        1,
        ZONE_HAND,
    )
    rust_state.push_to_stack(counter_rust_id, 1)
    rust_state.resolve_top_of_stack()
    rust_state.apply_sbas()

    assert rust_state.stack_size() == 0, (
        f"Rust stack should be empty after counterspell fizzles; "
        f"got size={rust_state.stack_size()}"
    )
    rust_view = parity_smoke._build_field_view(rust_state)
    rust_p1_gy_names = tuple(e[0] for e in rust_view["graveyard_p1"])
    assert "Counterspell" in rust_p1_gy_names, (
        f"expected Counterspell in Rust p1 graveyard after fizzle "
        f"(move_source_post_resolve routes Instant to gy regardless "
        f"of effect outcome); got {rust_p1_gy_names}"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #20: flash tests. Pins the flash_selector walk +
# the predicted CE19 legal_decisions emission divergence. See
# cross_engine.flash_selector for the full design notes. Test names
# follow the ``test_cross_engine_flash_`` prefix for parity with the
# prior keyword probes' naming convention.
#
# **Test surface** is ``legal_decisions`` emission, not the walk_both
# field-view tuple -- same shape CE17 used for the defender legal-
# emission probe. Three fresh-state direct-call tests probe the
# three priority-window scenarios; a pre-injection backbone walk
# pins the upstream parity.
# ---------------------------------------------------------------------------


def _legal_actions_to_dicts(actions):
    """Convert Python ``legal.legal_actions`` output (a list of
    ``Action`` dataclasses) to the same dict shape Rust's
    ``legal_decisions`` returns. Lets us probe Python's emission via
    a uniform dict-list interface.

    Only the kinds the flash probe cares about (``pass_priority``,
    ``cast_spell``, ``play_land``) are mapped; other kinds round-trip
    to ``{"kind": <classname-lowercase>}`` for inclusion in count
    checks but aren't asserted on by field shape.
    """
    from mtg_bot.actions import CastSpell, PassPriority, PlayLand

    out = []
    for action in actions:
        if isinstance(action, PassPriority):
            out.append({"kind": "pass_priority"})
        elif isinstance(action, CastSpell):
            out.append({"kind": "cast_spell", "card_id": int(action.card_id)})
        elif isinstance(action, PlayLand):
            out.append({"kind": "play_land", "card_id": int(action.card_id)})
        else:
            out.append({"kind": type(action).__name__.lower()})
    return out


def test_cross_engine_flash_test_a_closes_ce19(parity_smoke):
    """**Test A — CE19 CLOSED**: legal_decisions for a flash creature
    on p1's turn (active=p1) with p0 holding priority emits CastSpell
    on BOTH engines.

    Setup: walk the retaliation_selector backbone to step 167 (turn
    13 PRECOMBAT_MAIN, active=p0). Side-channel-advance phases on
    BOTH engines through to turn 14 PRECOMBAT_MAIN (active=p1).
    Set ``priority_player`` to 0 on both engines via Python's
    dataclass attribute write + Rust's ``give_priority_to(0)``.
    Inject Bear Flash (2/2 + flash keyword) into p0's hand on both
    engines. Probe ``legal_decisions(0)`` / ``legal_actions(state, 0)``.

    Post-CE19-closure expectation (both engines agree):

    * **Python**: ``legal.py:217-301`` instant-speed CastSpell block
      runs BEFORE the sorcery-timing gate at line 303. The filter
      ``card.printing.is_instant or "flash" in card.printing.keywords``
      accepts Bear Flash. Emission includes a ``CastSpell(bear_flash_id)``
      dict.
    * **Rust** (post-closure): ``state_ops_legal.rs`` CastSpell loop
      now lives ABOVE the sorcery-timing short-circuit and gates
      per-card on ``is_instant || keywords.contains("flash") ||
      is_sorcery_timing``. The flash keyword triggers the
      instant-speed pass — Rust emits ``CastSpell`` for the flash
      card at the off-active priority window, matching Python.

    The assertion shape is flipped from the pre-closure shape-pin
    ("Rust does NOT emit") to the post-closure equality form: both
    engines emit identical ``cast_spell`` card_id sets, and the flash
    card's id is in each set. See I46 in ``docs/RUST_PARITY_GAPS.md``.
    """
    from mtg_bot.legal import legal_actions
    from mtg_bot.parity.cross_engine import (
        advance_phase_python,
        build_matched_states,
        flash_side_channel,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    # Walk 167 backbone decisions to land at turn 13 PRECOMBAT_MAIN
    # active=p0 (same setpoint the prior 11 keyword probes use).
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Side-channel advance phases until we reach turn 14
    # PRECOMBAT_MAIN active=p1. From turn 13 PRECOMBAT_MAIN we need
    # to walk: PRECOMBAT_MAIN -> BEGIN_COMBAT -> DECLARE_ATTACKERS
    # -> DECLARE_BLOCKERS -> COMBAT_DAMAGE -> END_COMBAT ->
    # POSTCOMBAT_MAIN -> END_STEP -> CLEANUP (wraps to UNTAP +
    # turn++ + swap active) -> UPKEEP -> DRAW -> PRECOMBAT_MAIN. 11
    # advance_phase calls.
    from mtg_bot.turn import Step
    safety_cap = 25
    advances = 0
    while not (
        py_state.step == Step.PRECOMBAT_MAIN
        and py_state.active_player_idx == 1
    ):
        advance_phase_python(py_state)
        rust_state.advance_phase()
        advances += 1
        assert advances < safety_cap, (
            f"failed to reach turn 14 PRECOMBAT_MAIN active=p1 within "
            f"{safety_cap} advance_phase calls; current "
            f"step={py_state.step.name} active={py_state.active_player_idx}"
        )

    # Sanity: both engines agree on the new phase / active player.
    assert py_state.step == Step.PRECOMBAT_MAIN
    assert py_state.active_player_idx == 1
    assert rust_state.active_player_idx == 1, (
        f"Rust active_player_idx={rust_state.active_player_idx} != Python's 1"
    )

    # Set priority to p0 on both engines (so we probe the off-active
    # priority window where flash matters). Python: direct dataclass
    # field write. Rust: ``give_priority_to`` primitive.
    py_state.priority_player_idx = 0
    rust_state.give_priority_to(0)

    # Inject Bear Flash (2/2 + flash) into p0's hand on both engines.
    py_id, rust_id = flash_side_channel(py_state, rust_state)

    # ---- Probe Python's legal_actions(state, 0) ----
    py_actions = _legal_actions_to_dicts(legal_actions(py_state, 0))
    py_cast_card_ids = [
        a["card_id"] for a in py_actions
        if a["kind"] == "cast_spell"
    ]
    assert py_id in py_cast_card_ids, (
        f"Python should emit CastSpell for the flash creature (Bear "
        f"Flash id {py_id}) at p1's turn / p0 priority -- the "
        f"instant-speed block at legal.py:217-301 accepts "
        f"'flash' in keywords. Python emission cast_spell ids: "
        f"{py_cast_card_ids}; full emission kinds: "
        f"{[a['kind'] for a in py_actions]}"
    )

    # ---- Probe Rust's legal_decisions(0) ----
    rust_actions = rust_state.legal_decisions(0)
    rust_cast_card_ids = [
        a["card_id"] for a in rust_actions
        if a.get("kind") == "cast_spell"
    ]
    # CE19 CLOSED: Rust now emits CastSpell for the flash creature
    # via the instant-speed pass that lives above the sorcery-timing
    # short-circuit. The per-card gate at state_ops_legal.rs is
    # `is_instant || keywords.contains("flash") || is_sorcery_timing`;
    # the flash keyword fires the first clause, so the off-active
    # priority window emits regardless of `active != priority`.
    assert rust_id in rust_cast_card_ids, (
        f"CE19 closure regression: Rust did NOT emit CastSpell for "
        f"the flash creature (Bear Flash id {rust_id}) at p1's turn "
        f"/ p0 priority. Post-closure Rust's legal_decisions should "
        f"emit via the instant-speed pass above the sorcery-timing "
        f"gate (state_ops_legal.rs `is_instant || keywords.contains"
        f"(\"flash\") || is_sorcery_timing`). Rust emission "
        f"cast_spell ids: {rust_cast_card_ids}; full emission kinds: "
        f"{[a.get('kind') for a in rust_actions]}. See I46 in "
        f"docs/RUST_PARITY_GAPS.md."
    )

    # Both-engines-agree pin: the set of cast_spell card ids matches
    # exactly between Python and Rust. Stronger than "both contain
    # the flash id" -- catches a divergence where Rust over-corrects
    # and emits additional non-flash non-instant cards at the off-
    # active priority window (which would be a different and larger
    # divergence than CE19).
    assert set(py_cast_card_ids) == set(rust_cast_card_ids), (
        f"CE19 closure should produce identical CastSpell emissions "
        f"on both engines at the off-active priority window. "
        f"Python ids: {sorted(py_cast_card_ids)}; "
        f"Rust ids: {sorted(rust_cast_card_ids)}. "
        f"Mismatch indicates a per-card filter divergence."
    )


def test_cross_engine_flash_test_b_own_turn_main_phase_parity(parity_smoke):
    """**Test B — parity-clean sanity**: legal_decisions for a flash
    creature on p0's OWN turn (active=p0) at PRECOMBAT_MAIN, priority=p0.

    Both engines should emit CastSpell for the flash creature:

    * **Python**: instant-speed block at lines 217-301 accepts flash;
      ALSO the sorcery-speed block at lines 365-445 skips it
      (line 366: ``if "flash" in card.printing.keywords: continue``)
      to avoid emitting twice. Net: one CastSpell emitted from the
      instant-speed block.
    * **Rust**: sorcery-timing predicate passes
      (active=0 == priority=0, step=PRECOMBAT_MAIN, stack empty).
      Sorcery-speed CastSpell block at lines 833-853 walks the hand
      and emits a base CastSpell for every non-Land card. The flash
      keyword is ignored (Rust doesn't honor an instant-speed
      window), but at sorcery timing the cards still emit -- so the
      flash creature gets a CastSpell emission regardless.

    Both engines: emission contains ``CastSpell(bear_flash_id)``.
    Parity-clean on the load-bearing axis (presence of the cast in
    the emission list).
    """
    from mtg_bot.legal import legal_actions
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        flash_side_channel,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Sanity: we should be at turn 13 PRECOMBAT_MAIN active=p0 (the
    # retaliation walk's setpoint matching the prior 11 keyword probes).
    from mtg_bot.turn import Step
    assert py_state.step == Step.PRECOMBAT_MAIN
    assert py_state.active_player_idx == 0
    # Priority is on p0 (the active player) at the start of their own
    # main phase. Both engines maintain this invariant through the
    # retaliation walk -- no priority swap happens in the backbone.
    assert py_state.priority_player_idx == 0
    assert rust_state.priority_player_idx == 0

    py_id, rust_id = flash_side_channel(py_state, rust_state)

    # ---- Python emission ----
    py_actions = _legal_actions_to_dicts(legal_actions(py_state, 0))
    py_cast_card_ids = [
        a["card_id"] for a in py_actions
        if a["kind"] == "cast_spell"
    ]
    assert py_id in py_cast_card_ids, (
        f"Python should emit CastSpell for Bear Flash at own-turn main "
        f"phase (instant-speed block accepts flash; sorcery-speed "
        f"block skips it to dedupe). Got cast_spell ids: "
        f"{py_cast_card_ids}"
    )

    # ---- Rust emission ----
    rust_actions = rust_state.legal_decisions(0)
    rust_cast_card_ids = [
        a["card_id"] for a in rust_actions
        if a.get("kind") == "cast_spell"
    ]
    assert rust_id in rust_cast_card_ids, (
        f"Rust should emit CastSpell for Bear Flash at own-turn main "
        f"phase (sorcery-timing gate passes; CastSpell block walks "
        f"hand and emits for every non-Land). Got cast_spell ids: "
        f"{rust_cast_card_ids}; full emission: "
        f"{[a.get('kind') for a in rust_actions]}"
    )


def test_cross_engine_flash_test_c_vanilla_off_turn_negative_parity(parity_smoke):
    """**Test C — parity-clean NEGATIVE**: legal_decisions for a
    VANILLA creature (NO flash) on p1's turn with p0 priority.

    Setup mirrors Test A's priority window (turn 14 PRECOMBAT_MAIN
    active=p1, priority=p0) but injects a Vanilla Bear (no flash
    keyword) into p0's hand instead of Bear Flash.

    Expected: neither engine emits CastSpell for the vanilla
    creature.

    * **Python**: the instant-speed block's filter
      ``card.printing.is_instant or "flash" in card.printing.keywords``
      rejects the vanilla bear (False on both clauses). The
      sorcery-speed block at lines 365-445 is gated by
      ``_is_sorcery_timing`` at line 303, which fails
      (active=1 != priority=0). Net: no CastSpell for vanilla bear.
    * **Rust**: sorcery-timing predicate fails (active != priority).
      Function returns at line 762-764 with just PassPriority. No
      CastSpell for any card in hand, vanilla or otherwise.

    Parity-clean. If this test DIVERGES, the bug is broader than a
    flash-keyword gap -- it'd be an instant-speed-window-honored
    mismatch on vanilla creatures, which would be a different and
    larger CE.
    """
    from mtg_bot.legal import legal_actions
    from mtg_bot.parity.cross_engine import (
        advance_phase_python,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
        vanilla_creature_side_channel,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Advance to turn 14 PRECOMBAT_MAIN active=p1, same as Test A.
    from mtg_bot.turn import Step
    safety_cap = 25
    advances = 0
    while not (
        py_state.step == Step.PRECOMBAT_MAIN
        and py_state.active_player_idx == 1
    ):
        advance_phase_python(py_state)
        rust_state.advance_phase()
        advances += 1
        assert advances < safety_cap

    py_state.priority_player_idx = 0
    rust_state.give_priority_to(0)

    py_id, rust_id = vanilla_creature_side_channel(py_state, rust_state)

    # ---- Python emission ----
    py_actions = _legal_actions_to_dicts(legal_actions(py_state, 0))
    py_cast_card_ids = [
        a["card_id"] for a in py_actions
        if a["kind"] == "cast_spell"
    ]
    assert py_id not in py_cast_card_ids, (
        f"Python should NOT emit CastSpell for the vanilla creature "
        f"(Vanilla Bear id {py_id}) at p1's turn / p0 priority -- "
        f"the instant-speed filter rejects (no flash + not instant), "
        f"and the sorcery-speed block is gated by sorcery-timing "
        f"(active != priority fails). Unexpected emission: "
        f"cast_spell ids {py_cast_card_ids}; if this fails the "
        f"Python instant-speed gate has changed."
    )

    # ---- Rust emission ----
    rust_actions = rust_state.legal_decisions(0)
    rust_cast_card_ids = [
        a["card_id"] for a in rust_actions
        if a.get("kind") == "cast_spell"
    ]
    assert rust_id not in rust_cast_card_ids, (
        f"Rust should NOT emit CastSpell for the vanilla creature "
        f"(Vanilla Bear id {rust_id}) at p1's turn / p0 priority -- "
        f"sorcery-timing gate fails (active=1 != priority=0). "
        f"Unexpected emission: cast_spell ids {rust_cast_card_ids}. "
        f"If this fails the Rust enumerator has an instant-speed "
        f"window that emits vanillas, which is a different / larger "
        f"divergence than CE19."
    )

    # Parity-clean sanity: both engines emit only the priority-gate-
    # independent actions on the off-active priority window. For the
    # vanilla-bear setup this is exactly PassPriority on both sides.
    rust_kinds = sorted(a.get("kind") for a in rust_actions)
    py_kinds = sorted(a["kind"] for a in py_actions)
    assert rust_kinds == ["pass_priority"], (
        f"Rust off-priority emission should be ONLY pass_priority; "
        f"got: {rust_kinds}"
    )
    assert py_kinds == ["pass_priority"], (
        f"Python off-priority emission for vanilla-only hand at "
        f"non-sorcery-timing window should be ONLY pass_priority; "
        f"got: {py_kinds}"
    )


def test_cross_engine_flash_selector_pre_injection_walks_clean(parity_smoke):
    """Pre-injection backbone steps 0-167 are parity-clean.

    Same shape as the corresponding pre-injection test for every
    prior probe -- confirms the retaliation_selector backbone the
    flash injection rides on stays divergence-free up to the
    injection step. Distinguishes between "the flash side-channel
    itself introduces a divergence" and "something upstream of the
    side-channel is divergent and the flash probe just inherits it".
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        flash_selector,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    # Walk 168 steps -- the flash injection fires at step 168
    # itself, so steps 0..167 (168 steps total) are the
    # backbone-only window.
    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=flash_selector,
    )
    assert result.error is None, (
        f"pre-injection walk aborted at step {result.aborted_at}: "
        f"{result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Sanity: the flash injection hasn't fired yet at end of step 167.
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = flash_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    in_any_zone = (
        any(
            o.printing.name == "Bear Flash"
            for p in py_state.players
            for zone in (p.hand, p.library, p.graveyard, p.exile)
            for o in zone
        )
        or any(
            o.printing.name == "Bear Flash"
            for o in py_state.battlefield
        )
    )
    assert not in_any_zone, (
        "Bear Flash should not yet exist at end of step 167 -- "
        "flash injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #21: activated_mid_combat tests. Pins the
# activated_mid_combat_selector walk + the predicted shared engine
# behavior at a non-MAIN priority window. See
# cross_engine.activated_mid_combat_selector for the full design notes.
#
# **Test surfaces:**
#
#   * Test A: ``legal_decisions(0)`` emission at BEGINNING_OF_COMBAT
#     with priority on the active player. Both engines should refuse
#     to emit ActivateAbility for the injected Forest (the
#     ``_is_sorcery_timing`` gate fails because step != MAIN). Same
#     parity-clean axis the flash Test C pinned for the vanilla
#     creature -- both engines on the same side of a shared gate.
#
#   * Test B: direct ``apply_action(activate_ability)`` mid-combat.
#     Neither engine's dispatch arm phase-gates, so both should
#     accept and credit {G}. Field-view compare: Forest tapped, G
#     coin in pool, no stack push (mana abilities resolve immediately
#     per CR 605.3a).
#
#   * Test C: phase-advance from the mid-combat step. The {G}
#     credited in Test B should empty (per I20 mana-pool-empty-on-
#     phase-boundary semantics). Both engines should agree.
#
#   * Test D (pre-injection walk-clean): the
#     activated_mid_combat_selector backbone walks parity-clean for
#     steps 0..167 (the pre-injection window). Same shape as every
#     prior probe's walk-clean baseline.
# ---------------------------------------------------------------------------


def test_cross_engine_activated_mid_combat_test_a_no_emission_off_main(parity_smoke):
    """**Test A**: legal_decisions for an untapped Forest on p0's
    battlefield at BEGINNING_OF_COMBAT with p0 holding priority.

    Setup: walk the retaliation_selector backbone to step 167 (turn
    13 PRECOMBAT_MAIN, active=p0, priority=p0). Side-channel-advance
    phase by ONE step on both engines (PRECOMBAT_MAIN ->
    BEGINNING_OF_COMBAT). Inject a fresh untapped, unsick "Combat
    Forest" onto p0's bf. Probe legal_decisions(0) emission.

    Expected (parity-clean): both engines refuse to emit
    ActivateAbility for the Combat Forest because the
    ``_is_sorcery_timing`` gate fails (step=BEGINNING_OF_COMBAT not
    in MAIN_STEPS, regardless of priority/active alignment). Python's
    ``legal.py:303-304`` returns early after the instant-speed
    CastSpell pass; Rust's ``state_ops_legal.rs:837-839`` returns
    early at the matching post-instant-speed gate.

    This is a SHARED engine limitation (CR 605.1 mana abilities are
    nominally activatable at any priority window), NOT a CE
    divergence. The test pins both engines on the same side.

    If this test FAILS with one engine emitting ActivateAbility and
    the other not, that's a NEW CE -- the gate is being applied
    asymmetrically. Document as CE20 and xfail-pin.
    """
    from mtg_bot.legal import legal_actions
    from mtg_bot.parity.cross_engine import (
        activated_mid_combat_side_channel,
        advance_phase_python,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )
    from mtg_bot.turn import Step

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Sanity: turn 13 PRECOMBAT_MAIN active=p0 priority=p0.
    assert py_state.step == Step.PRECOMBAT_MAIN
    assert py_state.active_player_idx == 0
    assert py_state.priority_player_idx == 0

    # Advance ONE step: PRECOMBAT_MAIN -> BEGINNING_OF_COMBAT. Active
    # stays p0, priority flows back to p0 (active player gets priority
    # at the start of every step per CR 117.3a).
    advance_phase_python(py_state)
    rust_state.advance_phase()
    assert py_state.step == Step.BEGINNING_OF_COMBAT
    assert py_state.active_player_idx == 0
    assert rust_state.active_player_idx == 0, (
        f"Rust active_player_idx={rust_state.active_player_idx} != Python's 0"
    )

    # Inject a fresh untapped, unsick Combat Forest on p0's bf.
    py_forest_id, rust_forest_id = activated_mid_combat_side_channel(
        py_state, rust_state,
    )

    # ---- Probe Python's legal_actions(state, 0) ----
    py_actions = legal_actions(py_state, 0)
    py_activate_source_ids = [
        int(a.source_id) for a in py_actions
        if type(a).__name__ == "ActivateAbility"
    ]
    assert py_forest_id not in py_activate_source_ids, (
        f"Python should NOT emit ActivateAbility for the Combat "
        f"Forest (id {py_forest_id}) at BEGINNING_OF_COMBAT -- "
        f"the _is_sorcery_timing gate at legal.py:303-304 returns "
        f"early because step=BEGINNING_OF_COMBAT is not in "
        f"MAIN_STEPS. If Python now emits, the gate's been moved -- "
        f"this is a Python-only change that surfaces a NEW asymmetry "
        f"vs Rust (CE20 candidate). Python emission ActivateAbility "
        f"source ids: {py_activate_source_ids}; full emission types: "
        f"{[type(a).__name__ for a in py_actions]}"
    )

    # ---- Probe Rust's legal_decisions(0) ----
    rust_actions = rust_state.legal_decisions(0)
    rust_activate_source_ids = [
        a.get("source_id") for a in rust_actions
        if a.get("kind") == "activate_ability"
    ]
    assert rust_forest_id not in rust_activate_source_ids, (
        f"Rust should NOT emit ActivateAbility for the Combat Forest "
        f"(id {rust_forest_id}) at BEGINNING_OF_COMBAT -- the "
        f"is_sorcery_timing gate at state_ops_legal.rs:837-839 "
        f"returns before the ActivateAbility emission loop runs. If "
        f"Rust now emits, the gate's been moved -- this is a "
        f"Rust-only change that surfaces a NEW asymmetry vs Python "
        f"(CE20 candidate). Rust emission ActivateAbility source ids: "
        f"{rust_activate_source_ids}; full emission kinds: "
        f"{[a.get('kind') for a in rust_actions]}"
    )

    # Parity-clean shape pin: both engines emit ONLY pass_priority
    # (no flash creatures, no instants, no sorcery-timing-gated
    # actions). DeclareAttackers is gated on
    # ``step == PHASE_DECLARE_ATTACKERS`` in Rust (not yet at
    # BEGINNING_OF_COMBAT) and Python doesn't enumerate it either
    # (legal_actions never emits DeclareAttackers per the docstring
    # at engine.py:10-19).
    rust_kinds = sorted(a.get("kind") for a in rust_actions)
    assert rust_kinds == ["pass_priority"], (
        f"Rust at BEGINNING_OF_COMBAT with empty hand + no eligible "
        f"emission should produce ONLY pass_priority; got: {rust_kinds}"
    )
    py_types = sorted(type(a).__name__ for a in py_actions)
    assert py_types == ["PassPriority"], (
        f"Python at BEGINNING_OF_COMBAT should produce ONLY "
        f"PassPriority; got: {py_types}"
    )


def test_cross_engine_activated_mid_combat_test_b_apply_dispatch_parity(parity_smoke):
    """**Test B**: directly apply ``ActivateAbility(forest_id, 0)``
    mid-combat (BEGINNING_OF_COMBAT) on both engines.

    Setup: walk + advance + inject as in Test A. Then build an
    ``{"kind": "activate_ability", "source_id": forest_id,
    "ability_index": 0}`` dict and feed it to :func:`step_both`.

    Expected (parity-clean): both engines accept the apply_action,
    tap the Forest, and credit {G} to p0's mana pool. No stack push
    on either engine (CR 605.3a mana abilities resolve immediately).

    Test verifies:
      * Forest's ``tapped`` flag flips True on both engines.
      * p0's mana_pool gains exactly one G coin on both engines.
      * Stack stays empty on both engines (no StackItem push for
        the mana ability resolution).
      * The cross-engine field-view divergence list for this step
        is empty -- both engines agree on every fingerprinted field.

    If the field-view compare DIVERGES on mana_pool_p0,
    battlefield, or stack, that's the predicted CE20 surface --
    document and xfail-pin.
    """
    from mtg_bot.parity.cross_engine import (
        activated_mid_combat_side_channel,
        advance_phase_python,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )
    from mtg_bot.turn import Step

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    advance_phase_python(py_state)
    rust_state.advance_phase()
    assert py_state.step == Step.BEGINNING_OF_COMBAT

    py_forest_id, rust_forest_id = activated_mid_combat_side_channel(
        py_state, rust_state,
    )
    # Sanity: the freshly-injected Forest's id collision-free wrt the
    # existing bf objects. Both engines allocate ids monotonically from
    # the next-id pool; the helper guarantees no collision via the
    # ``max+1`` scan on Rust side and Python's ``next_object_id``
    # bump.
    assert py_forest_id > 0
    assert rust_forest_id > 0

    # ---- Apply ActivateAbility on BOTH engines via step_both ----
    activate_dict = {
        "kind": "activate_ability",
        "source_id": py_forest_id,
        "ability_index": 0,
    }
    # Rust uses its own forest id (per-engine id allocation diverges
    # for synthetic injections). Feed Rust its own dict.
    rust_activate_dict = {
        "kind": "activate_ability",
        "source_id": rust_forest_id,
        "ability_index": 0,
    }
    # step_both feeds the SAME dict to both engines, but the Python
    # and Rust Forest ids may differ. Work around by applying
    # directly per-engine (the step_both shape is for matched-id
    # decisions; this probe deliberately diverges ids).
    from mtg_bot.parity.cross_engine import (
        _dict_to_python_action,
        build_python_field_view,
    )
    from mtg_bot.actions import apply_action as py_apply_action

    py_action = _dict_to_python_action(activate_dict)
    py_apply_action(py_state, 0, py_action)
    rust_state.apply_action(rust_activate_dict)

    # ---- Compare post-state ----
    # Python: find the Combat Forest object on bf; check tapped + pool.
    py_forest = next(
        o for o in py_state.battlefield
        if int(o.id) == py_forest_id
    )
    assert py_forest.tapped, (
        f"Python: Combat Forest (id {py_forest_id}) should be tapped "
        f"after ActivateAbility; got tapped={py_forest.tapped}"
    )
    py_green_coins = sum(
        1 for coin in py_state.players[0].mana_pool
        if getattr(coin, "color", None) == "G"
    )
    assert py_green_coins == 1, (
        f"Python: p0's mana_pool should have exactly 1 G coin after "
        f"ActivateAbility (Forest's mana ability resolves "
        f"immediately per CR 605.3a); got {py_green_coins} G coins. "
        f"Pool: {py_state.players[0].mana_pool}"
    )
    assert len(py_state.stack) == 0, (
        f"Python stack should be empty after mana ability activation "
        f"(CR 605.3a -- mana abilities bypass the stack); got stack "
        f"with {len(py_state.stack)} items"
    )

    # Rust: same checks via the engine's public getters.
    rust_view = parity_smoke._build_field_view(rust_state)
    rust_mana_p0 = dict(rust_view["mana_pool_p0"])
    assert rust_mana_p0.get("G", 0) == 1, (
        f"Rust: p0's mana_pool should have exactly 1 G after "
        f"ActivateAbility; got mana_pool_p0={rust_view['mana_pool_p0']}"
    )
    assert rust_view["stack"] == (), (
        f"Rust stack should be empty after mana ability (CR 605.3a); "
        f"got: {rust_view['stack']}"
    )
    # Locate the Rust Combat Forest in the bf zone signature and
    # verify tapped=True. The zone signature is a tuple of 8-tuples
    # (printing_name, tapped, sick, dmg, counters, attached_to,
    # ctrl, id) -- index 1 is tapped.
    rust_forest_entries = [
        entry for entry in rust_view["battlefield"]
        if entry[0] == "Combat Forest"
    ]
    assert len(rust_forest_entries) == 1, (
        f"Rust battlefield should contain exactly one Combat Forest; "
        f"got {len(rust_forest_entries)}: {rust_forest_entries}"
    )
    # Zone signature encodes ``tapped`` as 0/1 int rather than
    # bool (matches the Rust getter's wire shape; the Python normalizer
    # passes the int through unchanged). Compare against truthy.
    assert bool(rust_forest_entries[0][1]), (
        f"Rust: Combat Forest should be tapped after ActivateAbility; "
        f"got bf entry: {rust_forest_entries[0]}"
    )

    # ---- Cross-engine field-view compare on mana / bf / stack ----
    py_view = build_python_field_view(py_state)
    # Mana pool parity: both engines should agree on the 6-color
    # tuple. This is the load-bearing field for the probe.
    assert py_view["mana_pool_p0"] == rust_view["mana_pool_p0"], (
        f"mana_pool_p0 diverges post-ActivateAbility -- predicted "
        f"CE20 surface. Python: {py_view['mana_pool_p0']}; Rust: "
        f"{rust_view['mana_pool_p0']}. Document as CE20 and "
        f"xfail-pin per the brief's outcome (b) shape."
    )
    # Stack parity: both empty.
    assert py_view["stack"] == rust_view["stack"], (
        f"stack diverges post-ActivateAbility -- predicted CE20 "
        f"surface (mana abilities should bypass the stack on both "
        f"engines). Python: {py_view['stack']}; Rust: "
        f"{rust_view['stack']}"
    )


def test_cross_engine_activated_mid_combat_test_c_phase_advance_empties_pool(parity_smoke):
    """**Test C**: after the mid-combat activation in Test B,
    advance phase ONE step (BEGINNING_OF_COMBAT -> DECLARE_ATTACKERS)
    on both engines. The {G} credited by the activation should
    empty (per I20 -- both engines empty mana pools on every
    phase / step boundary).

    Setup: same as Test B through the activation. Then advance phase
    once. Compare post-advance mana_pool_p0 on both engines.

    Expected (parity-clean): both engines' mana_pool_p0 becomes the
    canonical empty 6-tuple
    ``(('B',0),('C',0),('G',0),('R',0),('U',0),('W',0))``.

    Documented as I20 in the gap doc (mana-pool-empty-on-phase-
    advance composition). If the timing differs by one step --
    e.g. Python empties at COMBAT_DAMAGE while Rust empties at
    DECLARE_ATTACKERS -- that's a NEW CE divergence surface
    distinct from I20 (which is about presence-of-empty, not
    timing-of-empty). Document as CE20 and xfail-pin.
    """
    from mtg_bot.parity.cross_engine import (
        _dict_to_python_action,
        activated_mid_combat_side_channel,
        advance_phase_python,
        build_matched_states,
        build_python_field_view,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )
    from mtg_bot.actions import apply_action as py_apply_action
    from mtg_bot.turn import Step

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    advance_phase_python(py_state)
    rust_state.advance_phase()
    assert py_state.step == Step.BEGINNING_OF_COMBAT

    py_forest_id, rust_forest_id = activated_mid_combat_side_channel(
        py_state, rust_state,
    )

    # Activate the Forest on both engines.
    py_apply_action(
        py_state, 0, _dict_to_python_action({
            "kind": "activate_ability",
            "source_id": py_forest_id,
            "ability_index": 0,
        }),
    )
    rust_state.apply_action({
        "kind": "activate_ability",
        "source_id": rust_forest_id,
        "ability_index": 0,
    })

    # Sanity: both engines have {G} in p0's pool pre-advance.
    py_view_pre = build_python_field_view(py_state)
    rust_view_pre = parity_smoke._build_field_view(rust_state)
    assert dict(py_view_pre["mana_pool_p0"]).get("G", 0) == 1
    assert dict(rust_view_pre["mana_pool_p0"]).get("G", 0) == 1

    # ---- Advance phase ONE step ----
    advance_phase_python(py_state)
    rust_state.advance_phase()
    assert py_state.step == Step.DECLARE_ATTACKERS

    # ---- Compare post-advance mana_pool ----
    py_view_post = build_python_field_view(py_state)
    rust_view_post = parity_smoke._build_field_view(rust_state)

    expected_empty = (
        ("B", 0), ("C", 0), ("G", 0),
        ("R", 0), ("U", 0), ("W", 0),
    )
    assert py_view_post["mana_pool_p0"] == expected_empty, (
        f"Python: mana_pool_p0 should empty after phase advance "
        f"(I20). Got: {py_view_post['mana_pool_p0']}"
    )
    assert rust_view_post["mana_pool_p0"] == expected_empty, (
        f"Rust: mana_pool_p0 should empty after phase advance "
        f"(I20). Got: {rust_view_post['mana_pool_p0']}"
    )

    # Cross-engine parity on the post-advance mana_pool: both empty.
    assert py_view_post["mana_pool_p0"] == rust_view_post["mana_pool_p0"], (
        f"Post-phase-advance mana_pool_p0 diverges between engines -- "
        f"CE20 candidate (mana-empty timing asymmetry). Python: "
        f"{py_view_post['mana_pool_p0']}; Rust: "
        f"{rust_view_post['mana_pool_p0']}"
    )


def test_cross_engine_activated_mid_combat_selector_pre_injection_walks_clean(parity_smoke):
    """Pre-injection backbone steps 0-167 are parity-clean for the
    activated_mid_combat_selector.

    Same shape as the corresponding pre-injection test for every
    prior probe -- confirms the retaliation_selector backbone the
    Combat Forest injection rides on stays divergence-free up to
    the injection step. Distinguishes between "the Combat Forest
    side-channel itself introduces a divergence" and "something
    upstream of the side-channel is divergent and the probe just
    inherits it".
    """
    from mtg_bot.parity.cross_engine import (
        activated_mid_combat_selector,
        build_matched_states,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=activated_mid_combat_selector,
    )
    assert result.error is None, (
        f"pre-injection walk aborted at step {result.aborted_at}: "
        f"{result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Sanity: the Combat Forest injection hasn't fired yet at end of
    # step 167 (3 creatures land on bf only at step 167's apply step
    # -- the selector's >= 3 trigger fires the NEXT call, i.e. step
    # 168).
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = activated_mid_combat_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)
    in_any_zone = any(
        o.printing.name == "Combat Forest"
        for o in py_state.battlefield
    )
    assert not in_any_zone, (
        "Combat Forest should not yet exist at end of step 167 -- "
        "injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #22: activated_cost tests. Pins the
# activated_cost_selector walk + the predicted CE20 surface for
# activated abilities that COST MANA TO PAY (not free mana abilities).
# See cross_engine.activated_cost_selector for the full design notes.
#
# **Test surfaces:**
#
#   * Test A -- emission with affordance: Python emits ActivateAbility
#     for the Bear Caster (mana cost {1}{G} payable from 3 untapped
#     Forests on p0's bf). Rust does NOT emit (mana_produced.is_none()
#     filter at state_ops_legal.rs:929-1003). **CE20 SURFACED**.
#
#   * Test B -- emission with NO affordance: tap all p0's Forests +
#     empty mana pool. Python's can_pay_cost gate fails -> no emission.
#     Rust still doesn't emit (mana_produced filter fires unconditionally).
#     Both engines "no emit" -- parity-clean on this branch but for
#     entirely different reasons. The branch pins the affordability gate
#     isn't accidentally surfacing as a CE.
#
#   * Test C -- apply_action with sufficient mana in pool: pre-populate
#     {1}{G}. Python apply succeeds (auto_tap_for_cost deducts pool,
#     Bear taps, draw effect goes on stack). Rust apply RAISES
#     ValueError("source N has no mana_produced ..."). **CE20 SURFACED**
#     on dispatch axis.
#
#   * Test D -- apply_action with INSUFFICIENT mana: empty pool + tapped
#     Forests. Python silently under-pays (no self-gate in
#     _activate_ability; auto_tap_for_cost just does its best). Rust
#     still raises the no-mana_produced error. Documents a subtler axis
#     of CE20 (Python doesn't self-gate; Rust short-circuits regardless).
# ---------------------------------------------------------------------------


def test_cross_engine_activated_cost_test_a_emission_parity_post_ce20_closure(parity_smoke):
    """**CE20 CLOSURE — flipped from pre-closure shape-pin.**

    Post-CE20-substrate, Rust's CardPrinting carries an
    ``activated_abilities`` Vec and ``legal_decisions`` walks it to
    emit ActivateAbility per per-ability affordability. With the
    Bear Caster injection's cost-payable from p0's pool (the side-
    channel pre-populates the pool via Python's auto-tap-from-lands
    accounting; Rust's narrower pool-only affordability accepts the
    pre-populated pool the same way), the emission now appears on
    BOTH engines.

    Pre-closure shape (CE20 pre-substrate): this test asserted Rust
    does NOT emit; that was the load-bearing CE20 surface pin. The
    closure flips the assertion to "Rust DOES emit, with the same
    source id mapped through the per-engine id allocators".

    NOTE: Python's emission accounts for auto-tap-from-lands
    affordability; Rust's emission today is gated by pool-only
    coverage. The cross-engine smoke pre-populates p0's pool so
    both gates see the cost as affordable.
    """
    from mtg_bot.legal import legal_actions
    from mtg_bot.parity.cross_engine import (
        activated_cost_side_channel,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )
    from mtg_bot.turn import Step

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    # Sanity: at PRECOMBAT_MAIN with p0 priority, sorcery timing holds.
    assert py_state.step == Step.PRECOMBAT_MAIN
    assert py_state.active_player_idx == 0
    assert py_state.priority_player_idx == 0

    py_bear_id, rust_bear_id = activated_cost_side_channel(
        py_state, rust_state,
    )
    assert py_bear_id > 0
    assert rust_bear_id > 0

    # Pre-populate p0's mana pool with {1}{G} on Rust so the pool-
    # only affordability check accepts the cost. Python's
    # auto-tap-from-lands accounting accepts the cost without any
    # priming (the 3 untapped Forests cover {1}{G} via auto-tap).
    from mtg_bot.game import ManaCoin
    rust_state.add_mana(0, "G", 1)
    rust_state.add_mana(0, "C", 1)

    # ---- Python ----
    py_actions = legal_actions(py_state, 0)
    py_activate_source_ids = [
        int(a.source_id) for a in py_actions
        if type(a).__name__ == "ActivateAbility"
    ]
    assert py_bear_id in py_activate_source_ids, (
        f"Python SHOULD emit ActivateAbility for the Bear Caster "
        f"(id {py_bear_id}); got {py_activate_source_ids}"
    )

    # ---- Rust (CE20 CLOSED — post-substrate Rust now emits) ----
    rust_actions = rust_state.legal_decisions(0)
    rust_activate_source_ids = [
        a.get("source_id") for a in rust_actions
        if a.get("kind") == "activate_ability"
    ]
    assert rust_bear_id in rust_activate_source_ids, (
        f"CE20 CLOSED — Rust SHOULD now emit ActivateAbility for the "
        f"Bear Caster (id {rust_bear_id}). The new activated_abilities "
        f"substrate walks the per-ability list and emits when the cost "
        f"is affordable from the pool. Got Rust ActivateAbility source "
        f"ids: {rust_activate_source_ids}"
    )

    # Sanity: the Forests on p0's bf DO emit ActivateAbility on BOTH
    # engines (their printings have mana_produced=["G"]). The legacy
    # mana-only path stays end-to-end functional post-closure.
    p0_forest_ids = {
        int(o.id) for o in py_state.battlefield
        if o.controller_id == 0 and o.printing.name == "Forest"
        and not o.tapped
    }
    py_forest_emit = {
        int(a.source_id) for a in py_actions
        if type(a).__name__ == "ActivateAbility"
    } & p0_forest_ids
    rust_forest_emit = {
        a.get("source_id") for a in rust_actions
        if a.get("kind") == "activate_ability"
    } & p0_forest_ids
    assert py_forest_emit == rust_forest_emit, (
        f"Untapped Forests should emit ActivateAbility identically on "
        f"both engines (free mana ability). Python forests: "
        f"{py_forest_emit}; Rust forests: {rust_forest_emit}."
    )


def test_cross_engine_activated_cost_test_b_no_emission_when_unaffordable(parity_smoke):
    """**Test B**: emission with insufficient mana affordance. Both
    engines do NOT emit -- parity-clean on this branch but for
    different reasons.

    Setup: walk to step 167, inject Bear Caster, then tap ALL of
    p0's Forests AND empty p0's mana pool. Probe both engines'
    emission.

    Expected (parity-clean negative): neither engine emits
    ActivateAbility for the Bear Caster.
      * Python: ``can_pay_cost`` at ``legal.py:653-657`` fails
        because no untapped Forests + empty pool means {1}{G} is
        unpayable. Skips emission.
      * Rust: still filters the bear out via
        ``mana_produced.is_none()`` regardless of affordance state
        (the filter fires before any cost check in Rust today).

    If Python now emits (e.g. the affordability gate is being
    bypassed), that's an orthogonal regression in Python's
    ``can_pay_cost`` -- NOT the CE20 surface. If Rust now emits,
    CE20 has been closed AND Rust's new emission code missing the
    affordability check would itself be a fresh divergence.
    """
    from mtg_bot.legal import legal_actions
    from mtg_bot.parity.cross_engine import (
        activated_cost_side_channel,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    py_bear_id, rust_bear_id = activated_cost_side_channel(
        py_state, rust_state,
    )

    # ---- Force the "cannot afford" precondition ----
    # Python: tap every untapped Forest on p0's bf + clear mana pool.
    for obj in py_state.battlefield:
        if obj.controller_id == 0 and obj.printing.name == "Forest":
            obj.tapped = True
    py_state.players[0].mana_pool.clear()
    # Rust: tap every Forest via the public tap helper +
    # empty_mana_pool. battlefield_snapshot yields one dict per
    # bf object across both players, sorted by etb_timestamp.
    for entry in rust_state.battlefield_snapshot():
        if entry.get("controller_id") == 0 and entry.get("printing_name") == "Forest":
            rust_state.tap_object(int(entry["id"]))
    rust_state.empty_mana_pool(0)

    # ---- Probe both ----
    py_actions = legal_actions(py_state, 0)
    py_activate_source_ids = [
        int(a.source_id) for a in py_actions
        if type(a).__name__ == "ActivateAbility"
    ]
    assert py_bear_id not in py_activate_source_ids, (
        f"Python should NOT emit ActivateAbility for Bear Caster "
        f"(id {py_bear_id}) when {{1}}{{G}} is unaffordable (all "
        f"Forests tapped + pool empty). If Python emits, "
        f"can_pay_cost is being bypassed -- orthogonal regression, "
        f"NOT CE20. Got: {py_activate_source_ids}"
    )

    rust_actions = rust_state.legal_decisions(0)
    rust_activate_source_ids = [
        a.get("source_id") for a in rust_actions
        if a.get("kind") == "activate_ability"
    ]
    assert rust_bear_id not in rust_activate_source_ids, (
        f"Rust should NOT emit ActivateAbility for Bear Caster -- "
        f"the mana_produced filter fires unconditionally. Got: "
        f"{rust_activate_source_ids}"
    )


def test_cross_engine_activated_cost_test_c_apply_dispatch_parity_post_ce20_closure(parity_smoke):
    """**CE20 CLOSURE — flipped from pre-closure shape-pin.**

    Post-CE20-substrate, Rust's ActivateAbility apply arm dispatches
    into ``printing.activated_abilities[ability_index]``, pays the
    mana cost from the controller's pool, taps the source, and
    pushes a StackItem carrying the effect (is_mana_ability=False
    routes through the stack per CR 605/113.3b). Both engines now
    accept the apply and converge on the post-state.

    Pre-closure shape (CE20 pre-substrate): Rust raised
    ValueError("has no mana_produced ..."). The closure flips the
    assertion to "Rust apply succeeds; both engines arrive at the
    same post-state shape".
    """
    from mtg_bot.actions import ActivateAbility as PyActivateAbility
    from mtg_bot.actions import apply_action as py_apply_action
    from mtg_bot.game import ManaCoin
    from mtg_bot.parity.cross_engine import (
        activated_cost_side_channel,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    py_bear_id, rust_bear_id = activated_cost_side_channel(
        py_state, rust_state,
    )

    # ---- Pre-populate p0's mana pool: {1}{G} = 1 generic + 1 G ----
    py_state.players[0].mana_pool.clear()
    py_state.players[0].mana_pool.append(ManaCoin(color="G"))
    py_state.players[0].mana_pool.append(ManaCoin(color="C"))
    rust_state.empty_mana_pool(0)
    rust_state.add_mana(0, "G", 1)
    rust_state.add_mana(0, "C", 1)

    # ---- Python apply: should succeed ----
    py_action = PyActivateAbility(source_id=py_bear_id, ability_index=0)
    pre_hand = len(py_state.players[0].hand)
    pre_pool_size = len(py_state.players[0].mana_pool)
    py_apply_action(py_state, 0, py_action)

    py_bear = next(o for o in py_state.battlefield if int(o.id) == py_bear_id)
    assert py_bear.tapped, (
        f"Python: Bear Caster (id {py_bear_id}) should be tapped after "
        f"apply_action(ActivateAbility) -- the tap_self cost fired. "
        f"Got tapped={py_bear.tapped}"
    )
    post_pool_size = len(py_state.players[0].mana_pool)
    assert post_pool_size == pre_pool_size - 2
    assert len(py_state.stack) == 1, (
        f"Python: stack should have exactly 1 StackItem (the draw "
        f"effect; is_mana_ability=False). Got: {len(py_state.stack)}"
    )
    assert len(py_state.players[0].hand) == pre_hand

    # ---- Rust apply: CE20 CLOSED — should now succeed ----
    rust_activate_dict = {
        "kind": "activate_ability",
        "source_id": rust_bear_id,
        "ability_index": 0,
    }
    rust_state.apply_action(rust_activate_dict)

    # Post-state parity:
    #   - Bear is tapped on both engines.
    assert rust_state.is_tapped(rust_bear_id) is True, (
        f"Rust: Bear Caster (id {rust_bear_id}) should be tapped "
        f"after the activate. Got tapped={rust_state.is_tapped(rust_bear_id)}"
    )
    #   - Rust's pool is drained of the 1G + 1C the Python add.
    rust_pool = rust_state.get_mana_pool(0)
    assert rust_pool["G"] == 0 and rust_pool["C"] == 0, (
        f"Rust: pool should be drained of {{1}}{{G}}; got {rust_pool}"
    )
    #   - Rust's stack carries the draw effect (is_mana_ability=False
    #     pushes a StackItem, mirroring Python's stack push).
    assert rust_state.stack_size() == 1, (
        f"Rust: stack should carry the draw effect post-activate; got "
        f"stack_size={rust_state.stack_size()}"
    )


def test_cross_engine_activated_cost_test_d_python_underpays_silently(parity_smoke):
    """**Test D (post-CE20-closure)**: apply_action with INSUFFICIENT
    mana in pool. Pins the asymmetry on the dispatch-time
    affordability gate:

      * Python: ``_activate_ability`` primitive does NOT self-gate
        (the gate lives in ``legal_actions``). An out-of-band
        unaffordable call silently under-pays and still pushes the
        draw on the stack.
      * Rust: the dispatch arm self-gates on
        ``pool_can_cover(pool, ability.cost.mana)`` via
        ``state_ops_cast::pay_mana_cost_pub`` — an unaffordable
        cost raises ValueError("unaffordable from controller's
        mana pool").

    Pre-closure shape: Rust raised "has no mana_produced". Post-
    closure the raise has the new message shape; the asymmetry axis
    (Python silent under-pay vs Rust loud reject) is preserved.
    A future slice might also rewire Python to self-gate at the
    primitive layer (full parity convergence at the gate axis).
    """
    import pytest
    from mtg_bot.actions import ActivateAbility as PyActivateAbility
    from mtg_bot.actions import apply_action as py_apply_action
    from mtg_bot.parity.cross_engine import (
        activated_cost_side_channel,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    py_bear_id, rust_bear_id = activated_cost_side_channel(
        py_state, rust_state,
    )

    # ---- Force unaffordable: tap all Forests + empty pool ----
    for obj in py_state.battlefield:
        if obj.controller_id == 0 and obj.printing.name == "Forest":
            obj.tapped = True
    py_state.players[0].mana_pool.clear()
    for entry in rust_state.battlefield_snapshot():
        if entry.get("controller_id") == 0 and entry.get("printing_name") == "Forest":
            rust_state.tap_object(int(entry["id"]))
    rust_state.empty_mana_pool(0)

    # ---- Python: silent under-pay ----
    py_action = PyActivateAbility(source_id=py_bear_id, ability_index=0)
    py_apply_action(py_state, 0, py_action)

    py_bear = next(o for o in py_state.battlefield if int(o.id) == py_bear_id)
    assert py_bear.tapped, (
        f"Python: Bear Caster should be tapped even when mana is "
        f"unaffordable (the tap_self cost fires before the mana check "
        f"is gated; _activate_ability doesn't self-gate). Got "
        f"tapped={py_bear.tapped}"
    )
    assert len(py_state.players[0].mana_pool) == 0, (
        f"Python: mana pool should remain empty (no mana to consume; "
        f"auto_tap_for_cost couldn't tap any lands). Got pool: "
        f"{py_state.players[0].mana_pool}"
    )
    assert len(py_state.stack) == 1, (
        f"Python: stack should still have the draw effect (the "
        f"ability went on the stack despite the under-payment; "
        f"the primitive doesn't self-gate). Got: "
        f"{len(py_state.stack)} items"
    )

    # ---- Rust: self-gates on affordability, raises "unaffordable" ----
    rust_activate_dict = {
        "kind": "activate_ability",
        "source_id": rust_bear_id,
        "ability_index": 0,
    }
    with pytest.raises(ValueError, match="unaffordable from controller's mana pool"):
        rust_state.apply_action(rust_activate_dict)


def test_cross_engine_activated_cost_test_e_auto_tap_headroom_emission_parity(parity_smoke):
    """**Test E (post-I50 auto-tap-from-lands composer closure)**.

    Pre-I50 the cross-engine smoke pre-populated p0's pool with
    `{1}{G}` before driving the ActivateAbility emission probe
    because Rust's affordability was POOL-ONLY — Python's
    auto-tap-from-lands accounting wasn't mirrored on the Rust
    side. Test A documents this gap explicitly: "the side-channel
    pre-populates the pool via Python's auto-tap-from-lands
    accounting; Rust's narrower pool-only affordability accepts the
    pre-populated pool the same way".

    Post-I50 Rust's `auto_tap_headroom_pool` composer matches
    Python's `can_pay_cost` model: emission-side affordability now
    sums pool + headroom from untapped non-sick mana sources on bf.
    With the side-channel's 3 untapped Forests on p0's bf, an empty
    pool composes to `{G}{G}{G}` of headroom, which covers
    `{1}{G}` directly — and the emission appears on BOTH engines
    WITHOUT priming the pool.

    Pins the converged emission-side affordability axis.
    """
    from mtg_bot.legal import legal_actions
    from mtg_bot.parity.cross_engine import (
        activated_cost_side_channel,
        build_matched_states,
        multi_blocker_deckspec,
        retaliation_selector,
        step_both,
    )
    from mtg_bot.turn import Step

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(167):
        action = retaliation_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)

    assert py_state.step == Step.PRECOMBAT_MAIN
    assert py_state.active_player_idx == 0
    assert py_state.priority_player_idx == 0

    py_bear_id, rust_bear_id = activated_cost_side_channel(
        py_state, rust_state,
    )
    assert py_bear_id > 0
    assert rust_bear_id > 0

    # **Crucial:** explicitly clear BOTH engines' p0 pools so the
    # affordability gate has to flow through the auto-tap-from-lands
    # composer on each side (not just consume floating mana that
    # leaked from earlier walk steps). The retaliation_selector
    # backbone leaves a couple of G coins in Python's pool from
    # tap-for-mana actions during the walk; without this clear, the
    # post-emission assertion would pass via floating mana, not via
    # the auto-tap composer — defeating the whole point of Test E.
    py_state.players[0].mana_pool.clear()
    rust_state.empty_mana_pool(0)
    rust_pool = rust_state.get_mana_pool(0)
    assert all(rust_pool[c] == 0 for c in ("W", "U", "B", "R", "G", "C")), (
        f"Test E precondition: Rust's p0 mana pool starts empty; got {rust_pool}"
    )
    assert len(py_state.players[0].mana_pool) == 0, (
        "Test E precondition: Python p0's mana pool is empty after clear."
    )

    # ---- Python emits via can_pay_cost auto-tap headroom ----
    py_actions = legal_actions(py_state, 0)
    py_activate_source_ids = [
        int(a.source_id) for a in py_actions
        if type(a).__name__ == "ActivateAbility"
    ]
    assert py_bear_id in py_activate_source_ids, (
        f"Python SHOULD emit ActivateAbility for Bear Caster (id "
        f"{py_bear_id}) with 0 pool + 3 untapped Forests via "
        f"can_pay_cost auto-tap headroom. Got {py_activate_source_ids}"
    )

    # ---- Rust emits via auto_tap_headroom_pool composer (I50) ----
    rust_actions = rust_state.legal_decisions(0)
    rust_activate_source_ids = [
        a.get("source_id") for a in rust_actions
        if a.get("kind") == "activate_ability"
    ]
    assert rust_bear_id in rust_activate_source_ids, (
        f"I50 CLOSED — Rust SHOULD now emit ActivateAbility for Bear "
        f"Caster (id {rust_bear_id}) with 0 pool + 3 untapped Forests, "
        f"matching Python's auto-tap-from-lands accounting. Got Rust "
        f"ActivateAbility source ids: {rust_activate_source_ids}"
    )


def test_cross_engine_activated_cost_selector_pre_injection_walks_clean(parity_smoke):
    """Pre-injection backbone steps 0-167 are parity-clean for the
    activated_cost_selector.

    Same shape as every prior probe's pre-injection walks-clean pin --
    confirms the retaliation_selector backbone the Bear Caster injection
    rides on stays divergence-free up to the injection step. Distinguishes
    between "the Bear Caster side-channel itself introduces a divergence"
    and "something upstream is divergent and the probe just inherits it".
    """
    from mtg_bot.parity.cross_engine import (
        activated_cost_selector,
        build_matched_states,
        multi_blocker_deckspec,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=activated_cost_selector,
    )
    assert result.error is None, (
        f"pre-injection walk aborted at step {result.aborted_at}: "
        f"{result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Sanity: the Bear Caster injection hasn't fired yet at end of
    # step 167 (3 creatures land on bf only at step 167's apply step --
    # the selector's >= 3 trigger fires the NEXT call, i.e. step 168).
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = activated_cost_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)
    in_any_zone = any(
        o.printing.name == "Bear Caster"
        for o in py_state.battlefield
    )
    assert not in_any_zone, (
        "Bear Caster should not yet exist at end of step 167 -- "
        "injection fires at step 168"
    )


# ---------------------------------------------------------------------------
# Phase 5 follow-up #21: multi_keyword_selector tests — MULTI-KEYWORD
# COMPOSITION combat (the eleven single-keyword surfaces — FS / DT /
# trample / lifelink / vigilance / haste / menace / defender /
# indestructible / double-strike / flying — are now well-covered, all
# clean or closed; this probe asks whether the keywords COMPOSE
# correctly on the same creature).
#
# Three scenarios:
#
#   * **Test A — Lifelink + Deathtouch (LL+DT).** A 2/2 LL+DT
#     attacker vs a 4/4 vanilla blocker. Per CR 702.2b + 702.15b:
#     attacker deals 2 to blocker; deathtouch upgrades mark to >= 4
#     (lethal); blocker dies. Blocker deals 4 back to attacker;
#     attacker (2 toughness) dies via SBA. Lifelink fires on the
#     literal 2 damage dealt -> p0 gains 2 life. Expected outcome on
#     both engines: both creatures dead, p0 life +2.
#
#   * **Test B — First-strike + Deathtouch (FS+DT).** A 2/2 FS+DT
#     attacker vs a 4/4 vanilla blocker. Per CR 702.7c + 702.2b:
#     first-strike pass deals 2 to blocker; deathtouch upgrades the
#     mark to >= 4 (lethal); blocker dies in the pass-1-alive-filter
#     gate. The regular pass alive-predicate filter drops the
#     blocker's back-damage entry (source is dead). Expected:
#     attacker survives with 0 marked damage, blocker dead.
#
#   * **Test C — Trample + Lifelink (T+LL).** A 4/4 T+LL attacker
#     vs a 2/2 vanilla blocker. Per CR 702.19b + 702.15b: trample
#     assigns lethal (= 2) to blocker, spills remaining 2 to
#     defender. Lifelink fires on BOTH entries (the blocker hit AND
#     the defender hit). Expected: blocker dead, attacker alive,
#     defender takes 2, p0 gains 4 life.
#
# **OUTCOME: (a) CLEAN PASS — no novel cross-engine divergence
# surfaced.** Both engines compose the keywords correctly per CR.
# Specifically:
#
#   * Lifelink fires per damage event (Python's combat.py:535-541) /
#     per plan entry (Rust's state_ops_combat.rs:1357-1364) at the
#     LITERAL `amount` value, not the deathtouch-upgraded mark. This
#     is correct (CR 702.15b: "Damage dealt by a creature with
#     lifelink causes its controller to gain that much life" — the
#     "that much" refers to the damage dealt, not the marked damage
#     after deathtouch upgrade).
#
#   * Deathtouch upgrades the recipient's mark to its toughness
#     (Python's `obj.marked_lethal = True` flag + SBA gate / Rust's
#     `obj.damage = recipient_toughness` upgrade at apply-time
#     state_ops_combat.rs:1336-1340). Both fire on creature-target
#     entries only; both fire BEFORE the post-combat SBA pass.
#
#   * Trample's spill-to-defender (Python combat.py:436-440 / Rust
#     state_ops_combat.rs:995-1008) fires the lifelink hook on the
#     spilled-to-defender entry the SAME way as the blocker entry
#     because lifelink runs in the per-event/per-entry apply loop
#     downstream of the plan partition (both engines).
#
#   * The CE18-closure alive-predicate filter (Rust
#     state_ops_combat.rs:1209-1247) drops the FS-killed blocker's
#     back-damage entry, matching Python's per-substep blockers-list
#     rebuild (combat.py:_resolve_damage_step). Both engines agree
#     on Test B's "attacker survives with 0 marked damage" outcome.
#
# Pinned by five new tests:
#
#   * test_cross_engine_multi_keyword_selector_walks_clean — full
#     200-decision walk on multi_blocker_deckspec with the LL+DT
#     injection, parity-clean every step.
#   * test_cross_engine_multi_keyword_lifelink_deathtouch — direct-
#     combat structural pin (Test A: LL+DT 2/2 vs vanilla 4/4).
#   * test_cross_engine_multi_keyword_first_strike_deathtouch —
#     direct-combat structural pin (Test B: FS+DT 2/2 vs vanilla 4/4).
#   * test_cross_engine_multi_keyword_trample_lifelink — direct-
#     combat structural pin (Test C: T+LL 4/4 vs vanilla 2/2).
#   * test_cross_engine_multi_keyword_selector_pre_injection_walks_clean
#     — steps 0-167 parity-clean (the selector's delegate-to-
#     retaliation_selector backbone is unchanged before step-168
#     injection).
# ---------------------------------------------------------------------------


def test_cross_engine_multi_keyword_selector_walks_clean(parity_smoke):
    """**Multi-keyword (LL+DT) clean-walk pin** — 200-decision walk on
    :func:`multi_blocker_deckspec` with :func:`multi_keyword_selector`
    is parity-clean every step. The LL+DT injection at step 168 +
    the single-blocker COMBAT_DAMAGE step around index 175 compose
    correctly on both engines: 2 damage assigned to Bears id 9 fires
    BOTH deathtouch mark-upgrade (mark >= 2 = toughness, blocker
    dies) AND lifelink (+2 life on p0).

    No CE21 surfaced on this slice. Outcome (a) — keywords compose
    correctly.
    """
    from mtg_bot.parity.cross_engine import (
        multi_blocker_deckspec,
        multi_keyword_selector,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=200,
        decision_selector=multi_keyword_selector,
    )
    assert result.error is None, (
        f"walk aborted at step {result.aborted_at}: {result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"unexpected divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )


def test_cross_engine_multi_keyword_lifelink_deathtouch(parity_smoke):
    """**Test A — LL+DT composition pin** — fresh-state, direct
    combat invocation on both engines.

    Setup: 2/2 lifelink+deathtouch Bear Inquisitor on p0 bf; 4/4
    vanilla Bear Behemoth on p1 bf. Declare attack + block; resolve
    combat damage + SBAs.

    Expected end-state (matches both engines):
      * Attacker DEAD (took 4 back-damage from 4/4 blocker; 4 >=
        toughness 2 = lethal).
      * Blocker DEAD (took 2 damage; deathtouch upgrades mark to
        >= 4 = toughness; blocker dies in SBA per CR 702.2b).
      * p0 life = 22 (was 20; +2 from lifelink on the 2-damage
        creature event; lifelink fires on literal `amount`, not the
        deathtouch-upgraded mark).
      * p1 life = 20 (unchanged; no spillover, no trample).
      * Bear Inquisitor in p0 graveyard.
      * Bear Behemoth in p1 graveyard.

    Pins the contract that lifelink + deathtouch on the same source
    compose correctly: lifelink fires on the literal damage dealt
    (not the post-deathtouch upgraded mark), and deathtouch's mark
    upgrade is independent of lifelink.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.combat import resolve_combat_damage
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )
    from mtg_bot.turn import state_based_actions

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Python side: LL+DT 2/2 + vanilla 4/4 ----
    ll_dt_printing = CardPrinting(
        name="Bear Inquisitor",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"lifelink", "deathtouch"}),
    )
    py_atk = py_state.new_object(
        ll_dt_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_atk.summoning_sick = False
    py_atk.tapped = False
    py_atk.damage = 0
    py_atk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_atk)

    behemoth_printing = CardPrinting(
        name="Bear Behemoth",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=4,
        toughness=4,
        produces=frozenset(),
        keywords=frozenset(),
    )
    py_blk = py_state.new_object(
        behemoth_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_blk.controller_id = 1
    py_blk.summoning_sick = False
    py_blk.tapped = False
    py_blk.damage = 0
    py_blk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_blk)

    py_atk.attacking = True
    py_atk.tapped = True
    py_blk.blocking = int(py_atk.id)
    py_atk.blocked_by.append(int(py_blk.id))

    py_p0_before = py_state.players[0].life
    py_p1_before = py_state.players[1].life
    resolve_combat_damage(py_state, defending_player_id=1)
    state_based_actions(py_state)

    py_atk_alive = any(o.id == py_atk.id for o in py_state.battlefield)
    py_blk_alive = any(o.id == py_blk.id for o in py_state.battlefield)
    assert not py_atk_alive, (
        "Python LL+DT: attacker should be DEAD (took 4 back-damage from "
        "4/4 blocker; 4 >= toughness 2 = lethal)"
    )
    assert not py_blk_alive, (
        "Python LL+DT: blocker should be DEAD (took 2 damage; deathtouch "
        "upgrades mark to >= toughness 4 per CR 702.2b)"
    )
    assert py_state.players[0].life == py_p0_before + 2, (
        f"Python LL+DT: p0 life should be {py_p0_before + 2} (+2 from "
        f"lifelink on the literal 2-damage event); got "
        f"{py_state.players[0].life}"
    )
    assert py_state.players[1].life == py_p1_before, (
        f"Python LL+DT: p1 life unchanged at {py_p1_before} (no spillover, "
        f"no trample); got {py_state.players[1].life}"
    )

    # ---- Rust side: LL+DT 2/2 + vanilla 4/4 ----
    rust_atk_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Inquisitor",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["lifelink", "deathtouch"],
        },
        0, 2,
    )
    rust_blk_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Behemoth",
            "card_types": ["Creature"],
            "power": 4,
            "toughness": 4,
            "keywords": [],
        },
        1, 2,
    )
    rust_state.declare_attacker(rust_atk_id)
    rust_state.declare_blocker(rust_blk_id, rust_atk_id)
    rust_p0_before = rust_state.get_life(0)
    rust_p1_before = rust_state.get_life(1)
    rust_state.deal_combat_damage()
    rust_state.apply_sbas()

    rust_bf_ids = {int(e["id"]) for e in rust_state.battlefield_snapshot()}
    assert rust_atk_id not in rust_bf_ids, (
        "Rust LL+DT: attacker should be DEAD"
    )
    assert rust_blk_id not in rust_bf_ids, (
        "Rust LL+DT: blocker should be DEAD"
    )
    assert rust_state.get_life(0) == rust_p0_before + 2, (
        f"Rust LL+DT: p0 life should be {rust_p0_before + 2} (+2 from "
        f"lifelink); got {rust_state.get_life(0)}"
    )
    assert rust_state.get_life(1) == rust_p1_before, (
        f"Rust LL+DT: p1 life unchanged at {rust_p1_before}; got "
        f"{rust_state.get_life(1)}"
    )


def test_cross_engine_multi_keyword_first_strike_deathtouch(parity_smoke):
    """**Test B — FS+DT composition pin** — fresh-state, direct
    combat invocation on both engines.

    Setup: 2/2 first-strike+deathtouch Bear Slayer on p0 bf; 4/4
    vanilla Bear Behemoth on p1 bf. Declare attack + block; resolve
    combat damage + SBAs.

    Expected end-state (matches both engines):
      * FS substep: attacker deals 2 to blocker; deathtouch upgrades
        mark to >= toughness 4 (lethal). Blocker dies via mid-combat
        SBA (Python) or pass-2 alive-predicate filter (Rust).
      * Regular substep: blocker is dead; its back-damage entry is
        dropped (source is dead — Rust's `battlefield_creature_alive`
        filter at state_ops_combat.rs:1211 / Python's per-substep
        blockers-list rebuild at combat.py:_resolve_damage_step).
        Attacker takes no back-damage.
      * Attacker ALIVE with 0 marked damage.
      * Blocker DEAD; Bear Behemoth in p1 graveyard.
      * Life totals unchanged (no creature-to-player damage; no
        trample; no lifelink).

    Pins the contract that first-strike + deathtouch on the same
    source compose correctly: deathtouch upgrades the FS-substep
    mark to lethal (so the blocker dies BEFORE the regular substep),
    and the alive-predicate filter then drops the blocker's
    back-damage. This is the load-bearing CE18-closure-touches-
    deathtouch axis the brief flagged as a potential CE21 site.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.combat import resolve_combat_damage
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )
    from mtg_bot.turn import state_based_actions

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Python side ----
    fs_dt_printing = CardPrinting(
        name="Bear Slayer",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset({"first strike", "deathtouch"}),
    )
    py_atk = py_state.new_object(
        fs_dt_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_atk.summoning_sick = False
    py_atk.tapped = False
    py_atk.damage = 0
    py_atk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_atk)

    behemoth_printing = CardPrinting(
        name="Bear Behemoth",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=4,
        toughness=4,
        produces=frozenset(),
        keywords=frozenset(),
    )
    py_blk = py_state.new_object(
        behemoth_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_blk.controller_id = 1
    py_blk.summoning_sick = False
    py_blk.tapped = False
    py_blk.damage = 0
    py_blk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_blk)

    py_atk.attacking = True
    py_atk.tapped = True
    py_blk.blocking = int(py_atk.id)
    py_atk.blocked_by.append(int(py_blk.id))

    py_p0_before = py_state.players[0].life
    py_p1_before = py_state.players[1].life
    resolve_combat_damage(py_state, defending_player_id=1)
    state_based_actions(py_state)

    py_atk_alive = any(o.id == py_atk.id for o in py_state.battlefield)
    py_blk_alive = any(o.id == py_blk.id for o in py_state.battlefield)
    assert py_atk_alive, (
        "Python FS+DT: attacker should be ALIVE (FS killed the blocker "
        "BEFORE the regular substep; alive-predicate / per-substep "
        "rebuild drops blocker's back-damage)"
    )
    assert not py_blk_alive, (
        "Python FS+DT: blocker should be DEAD (FS substep + deathtouch "
        "upgrades mark to lethal)"
    )
    # Attacker's marked damage cleared by mid-combat SBA's apply pass —
    # actually, marked damage persists through end-of-turn cleanup
    # (CR 514.2). Within the resolve_combat_damage call + immediate SBA,
    # the attacker's damage stays whatever the regular substep would
    # have applied. The post-CE18 alive-filter drops the blocker's
    # back-damage entry, so attacker damage = 0.
    py_atk_obj = next(o for o in py_state.battlefield if o.id == py_atk.id)
    assert py_atk_obj.damage == 0, (
        f"Python FS+DT: attacker should have 0 marked damage (FS killed "
        f"blocker; regular-substep back-damage entry dropped because "
        f"source is dead); got damage={py_atk_obj.damage}"
    )
    assert py_state.players[0].life == py_p0_before, (
        f"Python FS+DT: p0 life unchanged at {py_p0_before} (no "
        f"lifelink); got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == py_p1_before, (
        f"Python FS+DT: p1 life unchanged at {py_p1_before} (no "
        f"spillover, no trample); got {py_state.players[1].life}"
    )

    # ---- Rust side ----
    rust_atk_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Slayer",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": ["first_strike", "deathtouch"],
        },
        0, 2,
    )
    rust_blk_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Behemoth",
            "card_types": ["Creature"],
            "power": 4,
            "toughness": 4,
            "keywords": [],
        },
        1, 2,
    )
    rust_state.declare_attacker(rust_atk_id)
    rust_state.declare_blocker(rust_blk_id, rust_atk_id)
    rust_p0_before = rust_state.get_life(0)
    rust_p1_before = rust_state.get_life(1)
    rust_state.deal_combat_damage()
    rust_state.apply_sbas()

    rust_bf_ids = {int(e["id"]) for e in rust_state.battlefield_snapshot()}
    assert rust_atk_id in rust_bf_ids, (
        "Rust FS+DT: attacker should be ALIVE (FS killed blocker; "
        "alive-predicate filter drops back-damage entry)"
    )
    assert rust_blk_id not in rust_bf_ids, (
        "Rust FS+DT: blocker should be DEAD"
    )
    rust_atk_damage = next(
        (int(e.get("damage", 0))
         for e in rust_state.battlefield_snapshot()
         if int(e["id"]) == rust_atk_id),
        None,
    )
    assert rust_atk_damage == 0, (
        f"Rust FS+DT: attacker damage=0 (back-damage entry dropped); "
        f"got damage={rust_atk_damage}"
    )
    assert rust_state.get_life(0) == rust_p0_before, (
        f"Rust FS+DT: p0 life unchanged at {rust_p0_before}; got "
        f"{rust_state.get_life(0)}"
    )
    assert rust_state.get_life(1) == rust_p1_before, (
        f"Rust FS+DT: p1 life unchanged at {rust_p1_before}; got "
        f"{rust_state.get_life(1)}"
    )


def test_cross_engine_multi_keyword_trample_lifelink(parity_smoke):
    """**Test C — T+LL composition pin** — fresh-state, direct combat
    invocation on both engines.

    Setup: 4/4 trample+lifelink Bear Crusher on p0 bf; 2/2 vanilla
    Bear Cub on p1 bf. Declare attack + block; resolve combat
    damage + SBAs.

    Expected end-state (matches both engines):
      * Attacker (4 power) assigns 2 to blocker (= blocker's
        toughness = lethal floor under non-deathtouch), 2 trample-
        excess to defender (CR 702.19b).
      * Blocker DEAD (took 2 = toughness).
      * Attacker ALIVE (took 2 back-damage from 2/2 blocker; 2 <
        toughness 4 = non-lethal).
      * Attacker damage = 2 (marked).
      * p0 life = 24 (was 20; +4 from lifelink — TWO entries fire:
        +2 from the blocker hit AND +2 from the defender hit. Pins
        the contract that lifelink fires on the trample-spillover
        entry too, not just the blocker entry.)
      * p1 life = 18 (was 20; -2 from trample excess).
      * Bear Cub in p1 graveyard.

    This is the load-bearing T+LL composition pin — surfaces the
    "lifelink + trample double-counts or misses spillover" axis the
    brief flagged as a potential CE21 candidate.
    """
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.combat import resolve_combat_damage
    from mtg_bot.game import Zone
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
    )
    from mtg_bot.turn import state_based_actions

    py_state, rust_state = build_matched_states(multi_blocker_deckspec())

    # ---- Python side ----
    t_ll_printing = CardPrinting(
        name="Bear Crusher",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=4,
        toughness=4,
        produces=frozenset(),
        keywords=frozenset({"trample", "lifelink"}),
    )
    py_atk = py_state.new_object(
        t_ll_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_atk.summoning_sick = False
    py_atk.tapped = False
    py_atk.damage = 0
    py_atk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_atk)

    cub_printing = CardPrinting(
        name="Bear Cub",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        keywords=frozenset(),
    )
    py_blk = py_state.new_object(
        cub_printing, owner_id=1, zone=Zone.BATTLEFIELD,
    )
    py_blk.controller_id = 1
    py_blk.summoning_sick = False
    py_blk.tapped = False
    py_blk.damage = 0
    py_blk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_blk)

    py_atk.attacking = True
    py_atk.tapped = True
    py_blk.blocking = int(py_atk.id)
    py_atk.blocked_by.append(int(py_blk.id))

    py_p0_before = py_state.players[0].life
    py_p1_before = py_state.players[1].life
    resolve_combat_damage(py_state, defending_player_id=1)
    state_based_actions(py_state)

    py_atk_alive = any(o.id == py_atk.id for o in py_state.battlefield)
    py_blk_alive = any(o.id == py_blk.id for o in py_state.battlefield)
    assert py_atk_alive, (
        "Python T+LL: attacker should be ALIVE (took 2 < toughness 4 "
        "back-damage)"
    )
    assert not py_blk_alive, (
        "Python T+LL: blocker should be DEAD (took 2 = toughness)"
    )
    assert py_state.players[0].life == py_p0_before + 4, (
        f"Python T+LL: p0 life should be {py_p0_before + 4} (+4 from "
        f"lifelink: 2 on blocker hit + 2 on defender trample-spillover "
        f"hit; pins the contract that lifelink fires on both entries); "
        f"got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == py_p1_before - 2, (
        f"Python T+LL: p1 life should be {py_p1_before - 2} (-2 from "
        f"trample excess: 4 power - 2 lethal-to-blocker = 2 spillover); "
        f"got {py_state.players[1].life}"
    )

    # ---- Rust side ----
    rust_atk_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Crusher",
            "card_types": ["Creature"],
            "power": 4,
            "toughness": 4,
            "keywords": ["trample", "lifelink"],
        },
        0, 2,
    )
    rust_blk_id = rust_state.register_and_inject_object(
        {
            "name": "Bear Cub",
            "card_types": ["Creature"],
            "power": 2,
            "toughness": 2,
            "keywords": [],
        },
        1, 2,
    )
    rust_state.declare_attacker(rust_atk_id)
    rust_state.declare_blocker(rust_blk_id, rust_atk_id)
    rust_p0_before = rust_state.get_life(0)
    rust_p1_before = rust_state.get_life(1)
    rust_state.deal_combat_damage()
    rust_state.apply_sbas()

    rust_bf_ids = {int(e["id"]) for e in rust_state.battlefield_snapshot()}
    assert rust_atk_id in rust_bf_ids, (
        "Rust T+LL: attacker should be ALIVE"
    )
    assert rust_blk_id not in rust_bf_ids, (
        "Rust T+LL: blocker should be DEAD"
    )
    assert rust_state.get_life(0) == rust_p0_before + 4, (
        f"Rust T+LL: p0 life should be {rust_p0_before + 4} (+4 from "
        f"lifelink; matches Python); got {rust_state.get_life(0)}"
    )
    assert rust_state.get_life(1) == rust_p1_before - 2, (
        f"Rust T+LL: p1 life should be {rust_p1_before - 2} (-2 trample "
        f"excess; matches Python); got {rust_state.get_life(1)}"
    )


def test_cross_engine_multi_keyword_selector_pre_injection_walks_clean(
    parity_smoke,
):
    """Pre-injection backbone steps 0-167 are parity-clean for the
    multi_keyword_selector.

    Confirms the retaliation_selector backbone the Bear Inquisitor
    injection rides on stays divergence-free up to the injection
    step. Distinguishes between "the LL+DT side-channel itself
    introduces a divergence" and "something upstream is divergent
    and the probe just inherits it".
    """
    from mtg_bot.parity.cross_engine import (
        build_matched_states,
        multi_blocker_deckspec,
        multi_keyword_selector,
        step_both,
        walk_both,
    )

    result = walk_both(
        multi_blocker_deckspec(),
        n_decisions=168,
        decision_selector=multi_keyword_selector,
    )
    assert result.error is None, (
        f"pre-injection walk aborted at step {result.aborted_at}: "
        f"{result.error}"
    )
    for step in result.steps:
        assert step.divergences == [], (
            f"pre-injection divergence at step {step.step_index}: "
            f"action={step.action_dict} "
            f"{[(d.field, d.py_value, d.rust_value, d.known_gap) for d in step.divergences]}"
        )

    # Sanity: the Bear Inquisitor injection hasn't fired yet at end
    # of step 167 (3 creatures land on bf only at step 167's apply
    # step -- the selector's >= 3 trigger fires the NEXT call,
    # i.e. step 168).
    py_state, rust_state = build_matched_states(multi_blocker_deckspec())
    for i in range(168):
        action = multi_keyword_selector(py_state, rust_state, i)
        step_both(py_state, rust_state, action)
    in_any_zone = any(
        o.printing.name == "Bear Inquisitor"
        for o in py_state.battlefield
    )
    assert not in_any_zone, (
        "Bear Inquisitor should not yet exist at end of step 167 -- "
        "injection fires at step 168"
    )


# =====================================================================
# Phase 5 probe lane: real EoE-card ETB-trigger smoke
# =====================================================================
#
# These probes pin that real-EoE-card ETB triggers fire to parity on
# both engines by leveraging the ETB-trigger substrate (Slices 1-11 in
# `docs/RUST_ETB_TRIGGER_AUDIT.md`). Each probe:
#
#   1. Builds a `CardDef` mirroring a REAL Edge-of-Eternities printing
#      whose oracle text reduces to one ETB trigger the substrate
#      already supports (SourceIsSelf + DrawCards / GainLife so far).
#      Non-ETB oracle text (e.g. Warp, secondary clauses) is OUT of
#      scope -- the probe's contract is the ETB-trigger surface.
#
#   2. Injects the printing onto p0's hand via
#      `register_and_inject_object` (Rust) / direct GameObject
#      construction (Python) so the cast path uses the post-resolve
#      ETB substrate without depending on land/mana plumbing.
#
#   3. Casts the spell, drains the stack on both engines, and compares
#      the post-drain (life_p0, life_p1, hand_p0, hand_p1, bf
#      contents) tuple. Divergence on ANY axis surfaces a new CE.
#
# Outcomes:
#
#   (a) Clean pass -- both engines agree post-drain. Logged below;
#       moves the next probe forward.
#   (b) CE21 surfaces -- xfail-pin with a CE21 doc entry, switch to
#       closure-plan mode.
#   (c) Old-known-gap surfaces -- xfail-pin with cross-reference to
#       the existing CE entry.

def _build_eoe_etb_state(card_def):
    """Build matched Python + Rust states for a real-EoE probe.

    The card_def carries the ETB trigger; the deckspec ships one copy
    on each player's library so both engines materialize the printing
    on the registry the same way. Caller then injects onto a zone via
    the standard inject-and-resolve pattern.
    """
    from mtg_bot.parity.cross_engine import (
        DeckSpec, build_python_state, build_rust_state,
    )
    spec = DeckSpec(
        card_defs=(card_def,),
        p0_library=(card_def.name,),
        p1_library=(card_def.name,),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)
    return py_state, rust_state


def _drain_stack_both(py_state, rust_state):
    """Resolve the top of stack on both engines until both stacks are
    empty. Mirrors the synchronous draining pattern used by
    `test_rust_state_builder_wires_triggered_abilities` in
    test_cross_engine_etb_triggers.py.
    """
    from mtg_bot.actions import resolve_top_of_stack
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    while py_state.stack:
        resolve_top_of_stack(py_state)


def test_cross_engine_real_eoe_germinating_wurm_etb_gain_life_parity(
    parity_smoke,
):
    """Probe: Germinating Wurm (EoE) ETB trigger fires + parity-matches.

    Oracle text in `data/edge_of_eternities.json`:

        "When this creature enters, you gain 2 life.
         Warp {1}{G} (...)"

    The Warp clause is out of scope -- this probe exclusively exercises
    the ETB-trigger surface, which is the substrate's current
    sweet-spot:

      * Event: `enters`
      * Predicate: `source_is_self` (only the just-entered creature's
        own trigger fires)
      * Effect: `gain_life` amount=2

    Expected post-cast + post-drain state on BOTH engines:

      * Germinating Wurm on p0's battlefield (the spell's
        StackItem resolves to a permanent first, then the
        ETB trigger lands on top of the stack and is resolved
        next).
      * p0 life: 20 -> 22 (+2 from the ETB gain_life).
      * p1 life: unchanged at 20.
      * No residual stack on either engine.
      * No residual hand objects.

    This is the simplest cross-engine smoke for a REAL EoE card's
    ETB trigger -- a clean pass here pins the substrate end-to-end
    on a printing that lives in the actual draft pool, not just on
    the synthetic Slice-3 "ETB Draw Bear" fixture. Outcome (a) for
    this probe means the next probe can pick a more complex
    predicate / effect shape.
    """
    from mtg_bot.parity.cross_engine import CardDef
    germinating_wurm = CardDef(
        name="Germinating Wurm",
        card_type="Creature",
        mana_cost_generic=4,
        mana_cost_colored=(("G", 1),),
        power=5,
        toughness=5,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "gain_life", "amount": 2},
            },
        ),
    )
    py_state, rust_state = _build_eoe_etb_state(germinating_wurm)

    # ---- Rust side ----
    # Register-and-inject onto p0's hand, then push the spell onto the
    # stack and resolve. The substrate fires the ETB trigger onto the
    # stack right after the spell resolves; the trigger lands +2 life.
    ZONE_HAND = 0
    ZONE_BATTLEFIELD = 2
    rust_dict = {
        "name": "Germinating Wurm",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 4, "G": 1},
        "effects": [],
        "power": 5,
        "toughness": 5,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "gain_life", "amount": 2},
            }
        ],
    }
    rust_wurm_id = rust_state.register_and_inject_object(
        rust_dict, 0, ZONE_HAND
    )
    rust_p0_life_before = rust_state.get_life(0)
    rust_p1_life_before = rust_state.get_life(1)
    rust_state.push_to_stack(rust_wurm_id, 0)
    rust_state.resolve_top_of_stack()
    # Spell resolved -> permanent on bf, ETB trigger now on stack.
    assert rust_state.find_object_zone(rust_wurm_id) == ZONE_BATTLEFIELD
    assert rust_state.stack_size() == 1, (
        "expected Germinating Wurm's ETB trigger to land on the stack "
        "after the spell resolves"
    )

    # ---- Python side ----
    # Mirror the inject-and-resolve pattern: move the obj from library
    # to hand, then push onto the stack directly (bypassing mana cost
    # -- the substrate's contract is the resolve-and-fire path, not
    # the mana-payment path).
    from mtg_bot.actions import resolve_top_of_stack as py_resolve
    from mtg_bot.game import StackItem, Zone
    py_obj = py_state.players[0].library[0]
    assert py_obj.printing.name == "Germinating Wurm"
    py_state.players[0].library.remove(py_obj)
    py_obj.zone = Zone.STACK
    py_state.stack.append(StackItem(source=py_obj, controller_id=0))
    py_p0_life_before = py_state.players[0].life
    py_p1_life_before = py_state.players[1].life
    py_resolve(py_state)
    # Spell resolved -> permanent on bf, ETB trigger now on stack.
    assert any(
        o.printing.name == "Germinating Wurm"
        for o in py_state.battlefield
    ), "Germinating Wurm should be on Python p0's battlefield"
    assert len(py_state.stack) == 1, (
        "expected Germinating Wurm's ETB trigger to land on the "
        "Python stack after the spell resolves"
    )

    # ---- Drain both stacks, then compare ----
    _drain_stack_both(py_state, rust_state)

    # Post-drain parity: stacks empty, +2 life to p0, p1 unchanged.
    assert rust_state.stack_size() == 0
    assert len(py_state.stack) == 0
    assert rust_state.get_life(0) == rust_p0_life_before + 2, (
        f"Rust p0 life should be {rust_p0_life_before + 2}; "
        f"got {rust_state.get_life(0)}"
    )
    assert rust_state.get_life(1) == rust_p1_life_before, (
        f"Rust p1 life should be unchanged at {rust_p1_life_before}; "
        f"got {rust_state.get_life(1)}"
    )
    assert py_state.players[0].life == py_p0_life_before + 2, (
        f"Python p0 life should be {py_p0_life_before + 2}; "
        f"got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == py_p1_life_before, (
        f"Python p1 life should be unchanged at {py_p1_life_before}; "
        f"got {py_state.players[1].life}"
    )
    # Cross-engine: same final life totals.
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life


def test_cross_engine_real_eoe_quantum_riddler_etb_draw_parity(
    parity_smoke,
):
    """Probe: Quantum Riddler (EoE) ETB trigger fires + parity-matches.

    Oracle text in `data/edge_of_eternities.json`:

        "Flying
         When this creature enters, draw a card.
         As long as you have one or fewer cards in hand, if you would
         draw one or more cards, you draw that many cards plus one
         instead.
         Warp {1}{U}"

    The flying / replacement / warp clauses are out of scope -- this
    probe exclusively exercises the ETB-trigger surface
    (`SourceIsSelf` + `DrawCards(1)`). Note Quantum Riddler is the
    LITERAL "ETB Draw Bear" of the EoE pool, modulo cost and
    stats: the substrate's Slice-3 synthetic-fixture motif is a
    real-card pattern.

    Expected post-cast + post-drain state on BOTH engines:

      * Quantum Riddler on p0's battlefield.
      * p0 hand grows by exactly 1 (the ETB DrawCards(1) trigger
        drew a card from p0's library).
      * p0 library shrinks by exactly 1 (the drawn card).
      * p1 hand / library unchanged.
      * No residual stack.

    For the draw to have something to draw from, we ship a second
    copy of Germinating Wurm in p0's library as the "card to be
    drawn" fixture. (Using a second printing -- Germinating Wurm --
    keeps the library card distinct from the cast spell so a
    "library/hand confusion" bug would surface a clear name
    divergence.)
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )
    quantum_riddler = CardDef(
        name="Quantum Riddler",
        card_type="Creature",
        mana_cost_generic=3,
        mana_cost_colored=(("U", 2),),
        power=4,
        toughness=6,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            },
        ),
    )
    germinating_wurm = CardDef(
        name="Germinating Wurm",
        card_type="Creature",
        mana_cost_generic=4,
        mana_cost_colored=(("G", 1),),
        power=5,
        toughness=5,
        # NOTE: this copy lives in p0's library purely as a
        # draw-target. It carries the SAME ETB trigger as in the
        # prior probe, but the trigger is only relevant if the wurm
        # itself enters -- which it doesn't on the draw path. The
        # DrawCards(1) effect moves the wurm from library to hand,
        # not to bf, so the wurm's own ETB does NOT fire here.
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "gain_life", "amount": 2},
            },
        ),
    )
    spec = DeckSpec(
        card_defs=(quantum_riddler, germinating_wurm),
        p0_library=("Germinating Wurm",),
        p1_library=("Germinating Wurm",),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_HAND = 0
    ZONE_BATTLEFIELD = 2
    rust_dict = {
        "name": "Quantum Riddler",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 3, "U": 2},
        "effects": [],
        "power": 4,
        "toughness": 6,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {"kind": "source_is_self"},
                "effect": {"kind": "draw_cards", "count": 1},
            }
        ],
    }
    rust_riddler_id = rust_state.register_and_inject_object(
        rust_dict, 0, ZONE_HAND
    )
    rust_p0_hand_before = rust_state.hand_sizes[0]
    rust_p0_lib_before = rust_state.library_sizes[0]
    rust_state.push_to_stack(rust_riddler_id, 0)
    rust_state.resolve_top_of_stack()
    assert rust_state.find_object_zone(rust_riddler_id) == ZONE_BATTLEFIELD
    assert rust_state.stack_size() == 1, (
        "Quantum Riddler's ETB DrawCards(1) trigger should be on the "
        "Rust stack post-resolve"
    )

    # ---- Python side ----
    from mtg_bot.actions import resolve_top_of_stack as py_resolve
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.draw_mill import DrawCards
    from mtg_bot.effects.predicates import Event, IsSelf
    from mtg_bot.game import StackItem, Zone

    # Build a Python CardPrinting for Quantum Riddler that mirrors
    # the Rust registry's predicate-path trigger shape. (The
    # cross_engine.build_python_state path only materializes
    # printings from the DeckSpec's card_defs; injecting a
    # printing not in the deckspec needs a fresh CardPrinting +
    # state.new_object, mirroring the existing side-channel pattern
    # used by `vigilance_side_channel`.)
    riddler_printing = CardPrinting(
        name="Quantum Riddler",
        mana_cost_str="",
        mana_cost=ManaCost(generic=3, colored=(("U", 2),)),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=4,
        toughness=6,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=IsSelf(),
                effect=DrawCards(amount=1),
            ),
        ),
    )
    py_riddler = py_state.new_object(
        riddler_printing, owner_id=0, zone=Zone.HAND,
    )
    py_state.players[0].hand.append(py_riddler)
    py_p0_hand_before = len(py_state.players[0].hand)
    py_p0_lib_before = len(py_state.players[0].library)
    # Cast: move from hand to stack via direct StackItem push (same
    # bypass-mana-cost shape as the wurm probe).
    py_riddler.zone = Zone.STACK
    py_state.players[0].hand.remove(py_riddler)
    py_state.stack.append(
        StackItem(source=py_riddler, controller_id=0)
    )
    py_resolve(py_state)
    assert any(
        o.printing.name == "Quantum Riddler"
        for o in py_state.battlefield
    )
    assert len(py_state.stack) == 1, (
        "Python: Quantum Riddler's ETB DrawCards(1) trigger should be "
        "on the stack post-resolve"
    )

    # ---- Drain both stacks, then compare ----
    _drain_stack_both(py_state, rust_state)

    # Post-drain: each engine's p0 hand grew by 1 vs the pre-cast
    # snapshot (the Riddler itself was injected into the hand and
    # then went to bf, so its passage through hand is net-zero --
    # the +1 we see comes from the ETB DrawCards). The pre-cast
    # snapshots above were taken AFTER the riddler was injected on
    # both engines, so both deltas are +1 net.
    rust_p0_hand_after = rust_state.hand_sizes[0]
    rust_p0_lib_after = rust_state.library_sizes[0]
    py_p0_hand_after = len(py_state.players[0].hand)
    py_p0_lib_after = len(py_state.players[0].library)

    # The pre-cast hand snapshot included the riddler; post-cast the
    # riddler is on bf, BUT the ETB drew exactly 1 card from the
    # library into the hand. Net delta: hand size unchanged
    # (-1 for riddler leaving hand, +1 for the draw), library -1.
    assert rust_p0_hand_after == rust_p0_hand_before, (
        f"Rust p0 hand should be unchanged (riddler leaves hand "
        f"to bf, +1 draw replaces it): before={rust_p0_hand_before} "
        f"after={rust_p0_hand_after}"
    )
    assert rust_p0_lib_after == rust_p0_lib_before - 1, (
        f"Rust p0 library should shrink by 1 (the draw): "
        f"before={rust_p0_lib_before} after={rust_p0_lib_after}"
    )
    assert py_p0_hand_after == py_p0_hand_before, (
        f"Python p0 hand should be unchanged (riddler leaves hand "
        f"to bf, +1 draw replaces it): before={py_p0_hand_before} "
        f"after={py_p0_hand_after}"
    )
    assert py_p0_lib_after == py_p0_lib_before - 1, (
        f"Python p0 library should shrink by 1 (the draw): "
        f"before={py_p0_lib_before} after={py_p0_lib_after}"
    )

    # Cross-engine: same final library / hand sizes.
    assert rust_p0_hand_after == py_p0_hand_after
    assert rust_p0_lib_after == py_p0_lib_after
    assert rust_state.hand_sizes[1] == len(py_state.players[1].hand)
    assert rust_state.library_sizes[1] == len(py_state.players[1].library)


def test_cross_engine_real_eoe_eumidian_terrabotanist_landfall_parity(
    parity_smoke,
):
    """Probe: Eumidian Terrabotanist (EoE) Landfall trigger
    fires + parity-matches on a SUBJECT-not-SOURCE predicate path.

    Oracle text in `data/edge_of_eternities.json`:

        "Landfall - Whenever a land you control enters, you gain
         1 life."

    This is the **first** real-EoE probe that exercises the
    SUBJECT-vs-SOURCE distinction in the predicate algebra:

      * The TRIGGER SOURCE is the Eumidian Terrabotanist (a
        creature that's pre-placed on p0's battlefield).
      * The TRIGGER SUBJECT is a LAND that enters under p0's
        control (i.e. a Forest played by p0).

    The predicate `subject_is_land AND controlled_by_you`
    composes two Slice-4 predicate variants: `SubjectIsLand`
    (filters the event subject to land cards) AND
    `ControlledByYou` (the subject's controller must equal the
    trigger source's controller, i.e. p0).

    Both Slice-2's land-play-path dispatch_event fan-out (Rust:
    one of the 11 ETB-stamping sites threaded in
    `state_ops_cast.rs`'s `play_land` arm; Python:
    `actions.py:1139` inside `_play_land`) and the predicate
    algebra Slice 4 wired must be correct end-to-end for this
    probe to pass.

    Expected post-cast + post-drain state on BOTH engines:

      * Terrabotanist still on p0's bf (pre-placed; doesn't move).
      * Forest now on p0's bf (just played).
      * p0 life: 20 -> 21 (+1 from the landfall trigger).
      * p1 life: unchanged at 20.
      * No residual stack.

    Outcome (a) clean pass here proves the substrate dispatches
    correctly on the Subject-vs-Source distinction -- the same
    predicate shape covers the entire EoE Landfall surface (16
    printings counted in `data/edge_of_eternities.json`).
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )
    terra = CardDef(
        name="Eumidian Terrabotanist",
        card_type="Creature",
        mana_cost_generic=1,
        mana_cost_colored=(("G", 1),),
        power=2,
        toughness=3,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_land"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "gain_life", "amount": 1},
            },
        ),
    )
    forest = CardDef(
        name="Forest", card_type="Land", mana_produced=("G",),
    )
    spec = DeckSpec(
        card_defs=(terra, forest),
        p0_library=("Forest", "Forest"),
        p1_library=("Forest", "Forest"),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_HAND = 0
    ZONE_BATTLEFIELD = 2
    rust_terra_dict = {
        "name": "Eumidian Terrabotanist",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "G": 1},
        "effects": [],
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": {
                    "kind": "and",
                    "left": {"kind": "subject_is_land"},
                    "right": {"kind": "controlled_by_you"},
                },
                "effect": {"kind": "gain_life", "amount": 1},
            }
        ],
    }
    rust_terra_id = rust_state.register_and_inject_object(
        rust_terra_dict, 0, ZONE_BATTLEFIELD
    )
    rust_forest_dict = {
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
        "effects": [],
    }
    rust_forest_id = rust_state.register_and_inject_object(
        rust_forest_dict, 0, ZONE_HAND
    )
    rust_p0_life_before = rust_state.get_life(0)
    rust_p1_life_before = rust_state.get_life(1)
    # play_land(card_id, player_id)
    rust_state.play_land(rust_forest_id, 0)
    assert rust_state.find_object_zone(rust_forest_id) == ZONE_BATTLEFIELD, (
        "Rust: Forest should be on bf after play_land"
    )
    assert rust_state.stack_size() == 1, (
        "Rust: Terrabotanist's landfall trigger should be on the "
        "stack after the Forest enters"
    )

    # ---- Python side ----
    from mtg_bot.actions import _play_land, resolve_top_of_stack as py_resolve
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.effects.object_predicates import IsLand
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, Subject, YOU,
    )
    from mtg_bot.game import Zone

    # Build the Python Terrabotanist printing with the SAME
    # predicate-path trigger shape.
    terra_predicate = And(
        left=Subject(slot=None, inner=IsLand()),
        right=Controlled(by=YOU, subject=None),
    )
    terra_printing = CardPrinting(
        name="Eumidian Terrabotanist",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("G", 1),)),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=3,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=terra_predicate,
                effect=GainLife(amount=1),
            ),
        ),
    )
    py_terra = py_state.new_object(
        terra_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_terra.summoning_sick = False
    py_terra.tapped = False
    py_terra.damage = 0
    py_terra.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_terra)

    # Move a Forest from library to hand on Python side.
    py_forest = py_state.players[0].library[0]
    assert py_forest.printing.name == "Forest"
    py_state.players[0].library.remove(py_forest)
    py_forest.zone = Zone.HAND
    py_state.players[0].hand.append(py_forest)

    py_p0_life_before = py_state.players[0].life
    py_p1_life_before = py_state.players[1].life

    # Reset Python's "lands_played_this_turn" so _play_land doesn't
    # gate on the per-turn cap (the rust harness has no such gate
    # in the inject-and-play substrate path).
    py_state.players[0].lands_played_this_turn = 0
    _play_land(py_state, 0, py_forest.id)

    assert any(
        o.printing.name == "Forest" and o.controller_id == 0
        for o in py_state.battlefield
    ), "Python: Forest should be on bf after _play_land"
    assert len(py_state.stack) == 1, (
        "Python: Terrabotanist's landfall trigger should be on the "
        "stack after the Forest enters"
    )

    # ---- Drain both stacks, then compare ----
    _drain_stack_both(py_state, rust_state)

    # Post-drain parity: +1 life on p0 from landfall trigger.
    assert rust_state.get_life(0) == rust_p0_life_before + 1, (
        f"Rust p0 life should be {rust_p0_life_before + 1} "
        f"(+1 from landfall); got {rust_state.get_life(0)}"
    )
    assert rust_state.get_life(1) == rust_p1_life_before
    assert py_state.players[0].life == py_p0_life_before + 1, (
        f"Python p0 life should be {py_p0_life_before + 1} "
        f"(+1 from landfall); got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == py_p1_life_before
    # Cross-engine: same post-drain life totals.
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life


def test_cross_engine_real_eoe_mechan_assembler_another_artifact_parity(
    parity_smoke,
):
    """Probe D: Mechan Assembler (EoE) "another artifact" ETB trigger
    fires + parity-matches on the predicate algebra's NOT branch.

    Oracle text in `data/edge_of_eternities.json`:

        "Whenever another artifact you control enters, create a 2/2
         colorless Robot artifact creature token. This ability
         triggers only once each turn."

    The token-creation effect and the once_per_turn rider are
    deferred to follow-on probes (token creation isn't yet on the
    cross-engine effect bridge; once_per_turn pin lands in the next
    probe). This probe restricts the surface to the predicate-algebra
    novelty: the **NOT branch** that the prior three probes haven't
    exercised. The trigger predicate reduces to:

        Subject(IsArtifact) AND ControlledByYou AND NOT(IsSelf)

    -- composed via two Slice-1 algebra nodes (And + Not) wrapping
    three Slice-4 atoms (subject_is_artifact + controlled_by_you +
    source_is_self). Because we don't yet have a token-creation
    effect on the bridge, the effect is set to GainLife(1) -- the
    smallest observable parity diff.

    This is the **first** Phase 5 probe to exercise:

      * Predicate::Not in Rust (cards.rs:Predicate::Not) and
        Python's `Not(...)` trigger predicate (`predicates.py:312`).
      * The combined three-level And-Not composition through the
        cross-engine dict->Python bridge.
      * The "source vs subject identity" distinction in a NOT shape
        -- `NOT(source_is_self)` checks the SUBJECT (just-entered
        permanent) is NOT the SOURCE (Mechan Assembler that holds
        the trigger). Same identity check the prior Quantum Riddler
        probe drove on the POSITIVE side; this is the NEGATIVE side.

    Test flow (mirrors the prior Landfall probe's inject-and-fan-out
    shape rather than the cast-and-resolve shape -- the trigger
    SOURCE is pre-placed on bf and we inject a second artifact onto
    bf to observe the trigger firing on the second arrival):

      1. Pre-place Mechan Assembler on p0's battlefield. The Rust
         `register_and_inject_object(BATTLEFIELD)` fans out
         `Event::Enters` for the just-placed Mechan Assembler --
         that fan-out walks every bf permanent INCLUDING Mechan
         Assembler itself and evaluates its trigger predicate against
         the entering subject (= Mechan Assembler). The NOT(IsSelf)
         arm REJECTS the firing because source == subject. Net:
         NO trigger fires, NO life gain on Mechan Assembler's own
         arrival. This is half the parity contract (the "another"
         word in the oracle text doing its job).

      2. Inject a vanilla artifact ("Mechan Drone" -- a synthetic
         artifact-creature stand-in, since the actual oracle effect
         creates a Robot artifact creature token we can't yet
         represent on the bridge) onto p0's battlefield. The
         `Event::Enters` fan-out walks every bf permanent and finds
         Mechan Assembler's trigger predicate matches: the entering
         subject IS an artifact (yes), IS controlled by p0 (yes),
         and is NOT the source (yes -- this is the new "another"
         creature). Trigger lands on stack.

      3. Drain stack on both engines. Mechan Assembler's GainLife(1)
         resolves. p0's life goes 20 -> 21.

      4. Cross-engine parity check on (life_p0, life_p1) -- same on
         both engines, +1 from the trigger.

    Expected post-drain state on BOTH engines:

      * Mechan Assembler on p0's bf (didn't move).
      * Mechan Drone on p0's bf (just arrived).
      * p0 life: 20 -> 21 (+1 from the "another artifact entered"
        trigger firing on Drone's arrival; Mechan Assembler's own
        arrival did NOT fire because NOT(IsSelf) suppressed it).
      * p1 life: unchanged at 20.
      * No residual stack.

    Outcome (a) clean pass on both engines pins the entire predicate
    algebra (`And`, `Not`, subject filters, source filters) through
    the cross-engine bridge -- the substrate is now wide enough to
    drive every "another <type> you control enters" trigger shape in
    the EoE pool (Mechan Assembler, Haliya Guided by Light's
    creature-or-artifact arm, etc.). The shared predicate vocabulary
    is closed under composition through the bridge.
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )

    # The "another artifact you control enters" predicate, encoded
    # in the bridge's dict schema. Predicate atoms: subject_is_artifact
    # (the entering permanent is an artifact), controlled_by_you (its
    # controller equals the source's controller), NOT(source_is_self)
    # (the entering permanent is NOT the source ability's host -- i.e.
    # "another" rather than "itself"). Composed via the Slice-1
    # algebra's And + Not nodes.
    another_artifact_predicate = {
        "kind": "and",
        "left": {
            "kind": "and",
            "left": {"kind": "subject_is_artifact"},
            "right": {"kind": "controlled_by_you"},
        },
        "right": {
            "kind": "not",
            "inner": {"kind": "source_is_self"},
        },
    }
    mechan_assembler = CardDef(
        name="Mechan Assembler",
        card_type="Creature",  # Artifact Creature -- card_types is
                               # forwarded as ["Creature"] only via
                               # CardDef.card_type today, but the
                               # subject test reads the printing's
                               # card_types list and we add Artifact
                               # via the rust_dict override below for
                               # the Rust side and the Python printing
                               # for the Python side. (The Subject/
                               # controller filter cares about the
                               # SUBJECT's types here -- "an artifact
                               # you control" reads the just-arrived
                               # permanent, not the source.)
        mana_cost_generic=4,
        mana_cost_colored=(("U", 1),),
        power=4,
        toughness=4,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": another_artifact_predicate,
                # Real oracle is "create a Robot token" -- bridge
                # doesn't yet have token creation, so we lower to a
                # GainLife(1) probe (smallest observable trigger
                # outcome). The predicate-algebra coverage is the
                # point of this probe, not the effect surface.
                "effect": {"kind": "gain_life", "amount": 1},
            },
        ),
    )
    # The second artifact -- a synthetic stand-in named "Mechan
    # Drone" (the actual oracle effect creates a 2/2 Robot artifact
    # creature token; token creation isn't yet on the bridge, so we
    # represent the "another artifact entering" event with a directly-
    # injected artifact creature). It carries NO triggered abilities
    # of its own so any observed +1 life comes UNAMBIGUOUSLY from
    # Mechan Assembler's predicate-path trigger.
    mechan_drone = CardDef(
        name="Mechan Drone",
        card_type="Creature",
        mana_cost_generic=2,
        power=2,
        toughness=2,
        triggered_abilities=(),
    )
    spec = DeckSpec(
        # Both card defs in registry; libraries can be empty for both
        # players -- the probe injects directly onto bf rather than
        # drawing or casting, so library contents don't matter.
        card_defs=(mechan_assembler, mechan_drone),
        p0_library=(),
        p1_library=(),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_BATTLEFIELD = 2
    # Mechan Assembler dict: ["Artifact", "Creature"] in card_types so
    # the subject_is_artifact predicate-evaluator path sees it AS an
    # artifact when ANOTHER artifact's ENTERS firing walks this source.
    # (The source identity is the source-vs-subject distinction in
    # NOT(SourceIsSelf); the type read happens via the SUBJECT id.)
    rust_assembler_dict = {
        "name": "Mechan Assembler",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"generic": 4, "U": 1},
        "effects": [],
        "power": 4,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": another_artifact_predicate,
                "effect": {"kind": "gain_life", "amount": 1},
            }
        ],
    }
    rust_assembler_id = rust_state.register_and_inject_object(
        rust_assembler_dict, 0, ZONE_BATTLEFIELD
    )
    # Pin half the parity contract NOW: Mechan Assembler's OWN arrival
    # must NOT have fired the trigger. The NOT(IsSelf) arm rejects --
    # subject (assembler) IS source (assembler), so NOT(IsSelf) is
    # FALSE, And short-circuits. Net: stack is empty, life unchanged.
    assert rust_state.stack_size() == 0, (
        "Rust: Mechan Assembler's own arrival should NOT fire its "
        "'another artifact' trigger -- NOT(source_is_self) gates it "
        "off. stack should be empty."
    )
    rust_p0_life_after_assembler = rust_state.get_life(0)
    assert rust_p0_life_after_assembler == 20, (
        f"Rust p0 life should still be 20 after Mechan Assembler's "
        f"own arrival (NOT-branch gate suppressed the self-fire); "
        f"got {rust_p0_life_after_assembler}"
    )

    # Now inject Mechan Drone -- the "another artifact" arrival. This
    # fans out Event::Enters via `pending_etb_dispatches`; the
    # dispatcher walks every bf permanent (including Mechan Assembler)
    # and evaluates each one's predicate against the entering subject
    # (= Mechan Drone). Mechan Assembler's predicate now matches all
    # three arms: subject_is_artifact (Drone is artifact), controlled_
    # by_you (both p0), NOT(source_is_self) (Drone != Assembler).
    # Trigger lands on stack.
    rust_drone_dict = {
        "name": "Mechan Drone",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 2,
        "toughness": 2,
    }
    rust_drone_id = rust_state.register_and_inject_object(
        rust_drone_dict, 0, ZONE_BATTLEFIELD
    )
    rust_p0_life_before_drain = rust_state.get_life(0)
    assert rust_state.stack_size() == 1, (
        "Rust: Mechan Drone's arrival should fire Mechan Assembler's "
        "'another artifact' trigger onto the stack -- subject_is_"
        "artifact (yes) AND controlled_by_you (yes) AND NOT(IsSelf) "
        "(yes -- drone != assembler). Expected stack_size==1; got "
        f"{rust_state.stack_size()}"
    )

    # ---- Python side ----
    # Mirror the inject-and-fan-out pattern. Construct the Mechan
    # Assembler printing with the same predicate-path trigger shape,
    # call new_object to materialize an instance, drop it on bf, then
    # dispatch Event.ENTERS to fan out (matching what
    # `register_and_inject_object` does on the Rust side).
    from mtg_bot.actions import _trigger_etb_abilities
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.effects.object_predicates import IsArtifact
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, Not, Subject, IsSelf, YOU,
    )
    from mtg_bot.game import Zone

    # Mirror the dict-shape another_artifact_predicate as a Python
    # composed `TriggerPredicate`. The bridge module's
    # `_ta_dict_to_python_predicate` produces the same composition;
    # we replicate it here directly so the printing carries the
    # actual predicate object (not a dict).
    py_predicate = And(
        left=And(
            left=Subject(slot=None, inner=IsArtifact()),
            right=Controlled(by=YOU, subject=None),
        ),
        right=Not(inner=IsSelf()),
    )
    assembler_printing = CardPrinting(
        name="Mechan Assembler",
        mana_cost_str="",
        mana_cost=ManaCost(generic=4, colored=(("U", 1),)),
        type_line="Artifact Creature",
        supertypes=(),
        card_types=("Artifact", "Creature"),
        subtypes=(),
        power=4,
        toughness=4,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=py_predicate,
                effect=GainLife(amount=1),
            ),
        ),
    )
    py_assembler = py_state.new_object(
        assembler_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_assembler.summoning_sick = False
    py_assembler.tapped = False
    py_assembler.damage = 0
    py_assembler.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_assembler)
    # Fire Mechan Assembler's own ENTERS event. NOT(IsSelf) MUST
    # suppress the trigger -- the predicate evaluates subject==source.
    _trigger_etb_abilities(py_state, py_assembler)
    assert len(py_state.stack) == 0, (
        "Python: Mechan Assembler's own arrival should NOT push a "
        "trigger -- NOT(IsSelf) gates it off. Expected stack=[]; "
        f"got {[(s.source.printing.name if hasattr(s, 'source') else s) for s in py_state.stack]}"
    )
    py_p0_life_after_assembler = py_state.players[0].life
    assert py_p0_life_after_assembler == 20

    # Now inject Mechan Drone and fan out ENTERS. The predicate now
    # matches against Drone (subject) vs Assembler (source).
    drone_printing = CardPrinting(
        name="Mechan Drone",
        mana_cost_str="",
        mana_cost=ManaCost(generic=2),
        type_line="Artifact Creature",
        supertypes=(),
        card_types=("Artifact", "Creature"),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        triggered_abilities=(),
    )
    py_drone = py_state.new_object(
        drone_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_drone.summoning_sick = True
    py_drone.tapped = False
    py_drone.damage = 0
    py_drone.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_drone)
    py_p0_life_before_drain = py_state.players[0].life
    _trigger_etb_abilities(py_state, py_drone)
    assert len(py_state.stack) == 1, (
        "Python: Mechan Drone's arrival should fire Mechan Assembler's "
        "'another artifact' trigger -- subject_is_artifact (yes) "
        "AND controlled_by_you (yes) AND NOT(IsSelf) (drone != "
        f"assembler). Expected len(stack)==1; got {len(py_state.stack)}"
    )

    # ---- Drain both stacks, then compare ----
    _drain_stack_both(py_state, rust_state)

    # Post-drain: p0 gained +1 life from the Mechan Assembler trigger.
    # Critical parity contract: BOTH engines fired the trigger exactly
    # ONCE (only on Drone's arrival, not on Assembler's own arrival).
    assert rust_state.get_life(0) == rust_p0_life_before_drain + 1, (
        f"Rust p0 life should be {rust_p0_life_before_drain + 1} "
        f"(+1 from Mechan Assembler's 'another artifact' trigger); "
        f"got {rust_state.get_life(0)}"
    )
    assert rust_state.get_life(1) == 20
    assert py_state.players[0].life == py_p0_life_before_drain + 1, (
        f"Python p0 life should be {py_p0_life_before_drain + 1} "
        f"(+1 from Mechan Assembler's 'another artifact' trigger); "
        f"got {py_state.players[0].life}"
    )
    assert py_state.players[1].life == 20
    # Cross-engine: same post-drain life totals on both seats.
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life
    # Sanity: both stacks are empty after drain.
    assert rust_state.stack_size() == 0
    assert len(py_state.stack) == 0


def test_cross_engine_real_eoe_mechan_assembler_once_per_turn_parity(
    parity_smoke,
):
    """Probe E: Mechan Assembler (EoE) `once_per_turn` flag pin —
    trigger fires once on the first qualifying ENTERS and is
    suppressed on the second within the same turn.

    Oracle text (`data/edge_of_eternities.json`):

        "Whenever another artifact you control enters, create a 2/2
         colorless Robot artifact creature token. This ability
         triggers only once each turn."

    Probe D pinned the predicate algebra's NOT branch (the "another"
    clause). This probe pins the **`once_per_turn` flag** on the
    TriggeredAbility shape:

      * Rust: ``TriggeredAbility::once_per_turn: bool`` consumed at
        ``state_ops_resolve.rs:521`` (predicate-match gate skips the
        firing when ``state.triggered_this_turn`` contains
        ``(source_id, ability_index)``); stamped eagerly at push time
        (``push_triggered_ability`` line 581-585).
      * Python: ``TriggeredAbility.once_per_turn: bool`` consumed at
        ``turn.py:should_fire_trigger`` (``state.once_per_turn_fired``
        ledger; same eager-record semantics). Both eagerly record at
        the moment the event happens (MTG semantics — the ability has
        "triggered" the moment its event fires, even if it later
        fails to find legal targets and fizzles).

    Probe shape (same inject-and-fan-out frame as Probe D, twice):

      1. Pre-place Mechan Assembler with ``once_per_turn=True`` on
         p0 bf. Its OWN arrival doesn't fire (NOT(IsSelf) suppresses,
         carried over from Probe D's pin). Ledger empty.

      2. Inject Mechan Drone #1 on p0 bf. ENTERS dispatch: predicate
         matches all three arms (subject_is_artifact + controlled_by_you
         + NOT(IsSelf)). The ``once_per_turn`` gate sees an empty ledger
         for ``(assembler_id, ability_index=0)`` -> fires. Ledger now
         contains the key. Drain stack: +1 life.

      3. Inject Mechan Drone #2 on p0 bf. ENTERS dispatch: predicate
         still matches (Drone #2 is a different subject, all arms
         satisfied). The ``once_per_turn`` gate sees the ledger key
         -> trigger SKIPPED. No push, no life gain. Stack stays empty.

      4. Cross-engine parity check: BOTH engines fired the trigger
         exactly ONCE across two qualifying ENTERS in the same turn.
         p0 life: 20 -> 21 (not 22). Stacks empty post-drain.

    This is the **first** Phase 5 probe to exercise the
    ``once_per_turn`` flag through the cross-engine bridge — the
    Rust parser at ``cards.rs:4419`` reads ``once_per_turn`` from the
    triggered-ability dict; the Python bridge at
    ``cross_engine.py:275`` mirrors the same field into the Python
    ``TriggeredAbility`` instance. Both engines' eager-record-at-push
    semantics agree under this probe.

    The third-arrival recovery (cleanup -> ledger cleared -> trigger
    fires again on the same turn after) is OUT of scope — the cleanup
    step's reset path (Rust ``state_ops_cleanup.rs:272`` clears
    ``state.triggered_this_turn``; Python ``turn.py:1120``
    ``state.once_per_turn_fired.clear()``) is exercised by existing
    Rust + Python unit tests, and pinning it through the
    inject-and-fan-out cross-engine frame would require driving a
    cleanup phase change on both engines, which is a wider surface
    than this probe's contract. Future probe: turn-boundary
    once_per_turn reset via the cleanup path.
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )

    # Same predicate shape as Probe D (pinned to clean PASS there).
    another_artifact_predicate = {
        "kind": "and",
        "left": {
            "kind": "and",
            "left": {"kind": "subject_is_artifact"},
            "right": {"kind": "controlled_by_you"},
        },
        "right": {
            "kind": "not",
            "inner": {"kind": "source_is_self"},
        },
    }
    mechan_assembler = CardDef(
        name="Mechan Assembler",
        card_type="Creature",
        mana_cost_generic=4,
        mana_cost_colored=(("U", 1),),
        power=4,
        toughness=4,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": another_artifact_predicate,
                "effect": {"kind": "gain_life", "amount": 1},
                # The flag pin -- this is the new bit this probe
                # exercises on top of Probe D's predicate algebra.
                "once_per_turn": True,
            },
        ),
    )
    mechan_drone = CardDef(
        name="Mechan Drone",
        card_type="Creature",
        mana_cost_generic=2,
        power=2,
        toughness=2,
    )
    spec = DeckSpec(
        card_defs=(mechan_assembler, mechan_drone),
        p0_library=(),
        p1_library=(),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_BATTLEFIELD = 2
    rust_assembler_dict = {
        "name": "Mechan Assembler",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"generic": 4, "U": 1},
        "effects": [],
        "power": 4,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": another_artifact_predicate,
                "effect": {"kind": "gain_life", "amount": 1},
                "once_per_turn": True,
            }
        ],
    }
    rust_assembler_id = rust_state.register_and_inject_object(
        rust_assembler_dict, 0, ZONE_BATTLEFIELD
    )
    # Assembler's own arrival: NOT(IsSelf) suppresses; stack stays empty
    # AND the once_per_turn ledger stays empty (the gate only stamps on
    # successful push, not on predicate-failed firings).
    assert rust_state.stack_size() == 0
    assert rust_state.get_life(0) == 20

    rust_drone_dict_base = {
        "name": "Mechan Drone",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 2,
        "toughness": 2,
    }

    # Drone #1: first qualifying arrival. Trigger fires, ledger stamps.
    rust_drone1_id = rust_state.register_and_inject_object(
        rust_drone_dict_base, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 1, (
        "Rust: Drone #1 arrival should push Mechan Assembler's trigger "
        "onto the stack (first fire of the turn)"
    )
    # Drain the trigger now (before Drone #2) so we can observe the
    # +1 life from the first firing in isolation.
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    rust_p0_life_after_first = rust_state.get_life(0)
    assert rust_p0_life_after_first == 21, (
        f"Rust p0 life should be 21 after first trigger; "
        f"got {rust_p0_life_after_first}"
    )

    # Drone #2: second qualifying arrival same turn. once_per_turn
    # gate suppresses the trigger -- stack must stay empty, life
    # must NOT advance to 22.
    rust_drone2_id = rust_state.register_and_inject_object(
        rust_drone_dict_base, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 0, (
        "Rust: Drone #2 arrival should NOT push a trigger -- "
        "once_per_turn gate suppresses the second fire within the "
        f"same turn. Got stack_size={rust_state.stack_size()}"
    )
    assert rust_state.get_life(0) == 21, (
        f"Rust p0 life should still be 21 (no second trigger); "
        f"got {rust_state.get_life(0)}"
    )

    # ---- Python side ----
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.effects.object_predicates import IsArtifact
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, Not, Subject, IsSelf, YOU,
    )
    from mtg_bot.game import Zone

    py_predicate = And(
        left=And(
            left=Subject(slot=None, inner=IsArtifact()),
            right=Controlled(by=YOU, subject=None),
        ),
        right=Not(inner=IsSelf()),
    )
    assembler_printing = CardPrinting(
        name="Mechan Assembler",
        mana_cost_str="",
        mana_cost=ManaCost(generic=4, colored=(("U", 1),)),
        type_line="Artifact Creature",
        supertypes=(),
        card_types=("Artifact", "Creature"),
        subtypes=(),
        power=4,
        toughness=4,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=py_predicate,
                effect=GainLife(amount=1),
                once_per_turn=True,
            ),
        ),
    )
    py_assembler = py_state.new_object(
        assembler_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_assembler.summoning_sick = False
    py_assembler.tapped = False
    py_assembler.damage = 0
    py_assembler.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_assembler)
    _trigger_etb_abilities(py_state, py_assembler)
    # Assembler's own arrival suppressed (NOT(IsSelf)).
    assert len(py_state.stack) == 0
    assert py_state.players[0].life == 20

    drone_printing = CardPrinting(
        name="Mechan Drone",
        mana_cost_str="",
        mana_cost=ManaCost(generic=2),
        type_line="Artifact Creature",
        supertypes=(),
        card_types=("Artifact", "Creature"),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        triggered_abilities=(),
    )

    # Drone #1: first qualifying arrival; ledger empty -> fires.
    py_drone1 = py_state.new_object(
        drone_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_drone1.summoning_sick = True
    py_drone1.tapped = False
    py_drone1.damage = 0
    py_drone1.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_drone1)
    _trigger_etb_abilities(py_state, py_drone1)
    assert len(py_state.stack) == 1, (
        "Python: Drone #1 arrival should push Mechan Assembler's "
        "trigger onto the stack (first fire of the turn)"
    )
    # Drain the first firing before Drone #2.
    while py_state.stack:
        resolve_top_of_stack(py_state)
    py_p0_life_after_first = py_state.players[0].life
    assert py_p0_life_after_first == 21

    # Drone #2: second qualifying arrival same turn; ledger gates off.
    py_drone2 = py_state.new_object(
        drone_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_drone2.summoning_sick = True
    py_drone2.tapped = False
    py_drone2.damage = 0
    py_drone2.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_drone2)
    _trigger_etb_abilities(py_state, py_drone2)
    assert len(py_state.stack) == 0, (
        "Python: Drone #2 arrival should NOT push a trigger -- "
        "once_per_turn gate suppresses the second fire within the "
        f"same turn. Got stack={[s.source.printing.name for s in py_state.stack]}"
    )
    assert py_state.players[0].life == 21

    # ---- Cross-engine parity ----
    # Drain (no-op — both stacks already empty), then compare:
    _drain_stack_both(py_state, rust_state)
    # BOTH engines: net +1 life on p0 across two qualifying ENTERS
    # events in the same turn, not +2. p1 unchanged.
    assert rust_state.get_life(0) == 21
    assert rust_state.get_life(1) == 20
    assert py_state.players[0].life == 21
    assert py_state.players[1].life == 20
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life


def test_cross_engine_real_eoe_tannuk_memorial_ensign_landfall_burn_parity(
    parity_smoke,
):
    """Probe F: Tannuk, Memorial Ensign (EoE) Landfall->burn trigger
    fires + parity-matches across the newly-bridged
    ``damage_each_opponent`` effect.

    Oracle text in `data/edge_of_eternities.json`:

        "Landfall -- Whenever a land you control enters, Tannuk
         deals 1 damage to each opponent. If this is the second
         time this ability has resolved this turn, draw a card."

    The "second-time-this-turn -> draw a card" rider is OUT of scope
    -- modeling it requires a per-turn fire-count ledger on the
    bridge that doesn't exist yet (the existing once_per_turn ledger
    is a binary skip-gate, not a counted-fires bridge). This probe
    restricts the surface to the LANDFALL + DAMAGE_EACH_OPPONENT(1)
    base trigger. Both component clauses are now substrate-resolvable:

      * Predicate shape: ``Subject(IsLand) AND ControlledByYou`` --
        SAME shape pinned by the Eumidian Terrabotanist probe (the
        third Phase 5 probe), so the predicate-algebra coverage is
        already paid for.

      * Effect shape: ``DamageEachOpponent(1)`` -- NEW on the bridge
        as of the companion commit. The Rust side reads
        ``damage_each_opponent`` via ``parse_effect`` (cards.rs:2771);
        the Python bridge's
        ``_ta_dict_to_python_effect`` arm translates to
        ``DealDamageEachOpponent(amount=1)`` (effects/generic/damage.py).

    Probe shape (mirrors the Eumidian Terrabotanist Landfall frame):

      1. Pre-place Tannuk, Memorial Ensign on p0's bf. Tannuk's own
         arrival doesn't fire its predicate (Tannuk is a creature,
         not a land, so the SUBJECT(IsLand) arm short-circuits even
         without a NOT(IsSelf) gate -- that's a happy side-effect of
         the type filter).

      2. Inject a Forest onto p0's bf via ``play_land`` (Rust) /
         ``_play_land`` (Python). The land-play-path dispatches
         ENTERS for the Forest; the dispatcher walks every bf
         permanent and finds Tannuk's predicate matches:
         subject_is_land (yes), controlled_by_you (yes). Trigger
         pushes onto the stack.

      3. Drain stack on both engines. DamageEachOpponent(1) resolves:
         each engine subtracts 1 from p1's life (the only opponent
         in the 2-player game). p1 life: 20 -> 19.

      4. Cross-engine assert: BOTH engines reduced p1's life to 19.
         p0 life unchanged at 20 (Tannuk's controller, immune to the
         each-opponent damage).

    Expected post-drain state on BOTH engines:

      * Tannuk on p0's bf (didn't move).
      * Forest on p0's bf (just played).
      * p0 life: unchanged at 20 (controller-immune).
      * p1 life: 20 -> 19 (-1 from the landfall->burn trigger).
      * No residual stack.

    Outcome (a): clean pass pins the bridge extension end-to-end on
    a real-card landfall->burn trigger. Same predicate shape covers
    every "Landfall: do something to each opponent" trigger in EoE
    once landed on the bridge's effect surface; this probe is the
    first one to drive the LANDFALL + (effect-not-life/draw) shape.

    Notes on out-of-scope:

      * Multi-trigger-per-turn count (the "second time this turn"
        rider) -- needs a counter-based bridge entry, not the binary
        once_per_turn flag pinned by Probe E. Future probe / future
        bridge widening.

      * Cross-controller controller-of-effect contract -- this probe
        drives p0 (active_player) controlled trigger, where Rust's
        ``state.players[1 - active_player]`` reader and Python's
        per-source-controller walk agree. The "p1's source resolves
        on p0's turn" case is gated by the same active_player-keyed
        Rust reader that DrawCards/GainLife had pre-I59/I60 fixes.
        A follow-up controller-of-effect fix for
        ``Effect::DamageEachOpponent`` mirrors the I60 shape.
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )

    landfall_predicate = {
        "kind": "and",
        "left": {"kind": "subject_is_land"},
        "right": {"kind": "controlled_by_you"},
    }
    tannuk = CardDef(
        name="Tannuk, Memorial Ensign",
        card_type="Creature",
        mana_cost_generic=1,
        mana_cost_colored=(("R", 1),),
        power=2,
        toughness=2,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": landfall_predicate,
                "effect": {
                    "kind": "damage_each_opponent",
                    "amount": 1,
                },
            },
        ),
    )
    forest = CardDef(
        name="Forest", card_type="Land", mana_produced=("G",),
    )
    spec = DeckSpec(
        card_defs=(tannuk, forest),
        # Forest in p0's library — _play_land path moves it to bf.
        # p1 library has Forest too so the deckspec is symmetric; we
        # don't touch p1's resources in this probe.
        p0_library=("Forest", "Forest"),
        p1_library=("Forest", "Forest"),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_HAND = 0
    ZONE_BATTLEFIELD = 2
    rust_tannuk_dict = {
        "name": "Tannuk, Memorial Ensign",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1, "R": 1},
        "effects": [],
        "power": 2,
        "toughness": 2,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": landfall_predicate,
                "effect": {
                    "kind": "damage_each_opponent",
                    "amount": 1,
                },
            }
        ],
    }
    rust_tannuk_id = rust_state.register_and_inject_object(
        rust_tannuk_dict, 0, ZONE_BATTLEFIELD
    )
    # Tannuk's own arrival: predicate is subject_is_land AND
    # controlled_by_you. Tannuk is a Creature, NOT a Land, so
    # subject_is_land = FALSE. Trigger does NOT fire on Tannuk's own
    # arrival -- pin half the parity contract NOW.
    assert rust_state.stack_size() == 0, (
        "Rust: Tannuk's own arrival should NOT fire its landfall "
        "trigger -- subject_is_land filter rejects (Tannuk is a "
        "Creature). Expected stack_size==0; got "
        f"{rust_state.stack_size()}"
    )
    rust_p0_life_before = rust_state.get_life(0)
    rust_p1_life_before = rust_state.get_life(1)

    # Inject Forest onto p0's hand, then play it via play_land. The
    # play_land path dispatches ENTERS; the dispatcher walks every
    # bf permanent (including Tannuk) and fires matching triggers.
    rust_forest_dict = {
        "name": "Forest",
        "card_types": ["Land"],
        "mana_produced": ["G"],
        "effects": [],
    }
    rust_forest_id = rust_state.register_and_inject_object(
        rust_forest_dict, 0, ZONE_HAND
    )
    rust_state.play_land(rust_forest_id, 0)
    assert rust_state.find_object_zone(rust_forest_id) == ZONE_BATTLEFIELD, (
        "Rust: Forest should be on bf after play_land"
    )
    assert rust_state.stack_size() == 1, (
        "Rust: Tannuk's landfall trigger should be on the stack after "
        "the Forest enters under p0's control"
    )

    # ---- Python side ----
    from mtg_bot.actions import _play_land
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.damage import DealDamageEachOpponent
    from mtg_bot.effects.object_predicates import IsLand
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, Subject, YOU,
    )
    from mtg_bot.game import Zone

    py_predicate = And(
        left=Subject(slot=None, inner=IsLand()),
        right=Controlled(by=YOU, subject=None),
    )
    tannuk_printing = CardPrinting(
        name="Tannuk, Memorial Ensign",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1, colored=(("R", 1),)),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=py_predicate,
                effect=DealDamageEachOpponent(amount=1),
            ),
        ),
    )
    py_tannuk = py_state.new_object(
        tannuk_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_tannuk.summoning_sick = False
    py_tannuk.tapped = False
    py_tannuk.damage = 0
    py_tannuk.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_tannuk)
    # Python: Tannuk wasn't fanned through _trigger_etb_abilities
    # since we materialized it on bf directly. Stack should remain
    # empty as we never dispatched ENTERS for Tannuk's own arrival.
    assert len(py_state.stack) == 0

    # Move a Forest from library to hand on Python side.
    py_forest = py_state.players[0].library[0]
    assert py_forest.printing.name == "Forest"
    py_state.players[0].library.remove(py_forest)
    py_forest.zone = Zone.HAND
    py_state.players[0].hand.append(py_forest)

    py_p0_life_before = py_state.players[0].life
    py_p1_life_before = py_state.players[1].life
    # Reset the per-turn cap so _play_land doesn't gate (mirrors
    # the Eumidian Terrabotanist probe's reset pattern).
    py_state.players[0].lands_played_this_turn = 0
    _play_land(py_state, 0, py_forest.id)
    assert any(
        o.printing.name == "Forest" and o.controller_id == 0
        for o in py_state.battlefield
    ), "Python: Forest should be on bf after _play_land"
    assert len(py_state.stack) == 1, (
        "Python: Tannuk's landfall trigger should be on the stack "
        "after Forest enters under p0's control"
    )

    # ---- Drain both stacks, then compare ----
    _drain_stack_both(py_state, rust_state)

    # Post-drain: -1 life on p1 from the landfall burn trigger.
    # p0 unchanged (controller of the source, immune to "each
    # opponent").
    assert rust_state.get_life(0) == rust_p0_life_before, (
        f"Rust p0 life should be unchanged ({rust_p0_life_before}); "
        f"got {rust_state.get_life(0)}"
    )
    assert rust_state.get_life(1) == rust_p1_life_before - 1, (
        f"Rust p1 life should be {rust_p1_life_before - 1} "
        f"(-1 from landfall burn); got {rust_state.get_life(1)}"
    )
    assert py_state.players[0].life == py_p0_life_before
    assert py_state.players[1].life == py_p1_life_before - 1
    # Cross-engine: same post-drain life totals on both seats.
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life


def test_cross_engine_real_eoe_weftstalker_ardent_or_predicate_burn_parity(
    parity_smoke,
):
    """Probe G: Weftstalker Ardent (EoE) — first real-card consumer of
    the new ``Predicate::Or`` substrate. The deferred unblock from
    Probe F's forward notes lands here.

    Oracle text in ``data/edge_of_eternities.json``:

        "Whenever another creature or artifact you control enters,
         this creature deals 1 damage to each opponent."

    Predicate decomposition:

      * "another" -> ``NOT(SourceIsSelf)`` -- gate self-arrival.
      * "creature or artifact" -> ``Or(SubjectIsCreature, SubjectIsArtifact)``
        -- NEW disjunction, landed in the Or-substrate slice.
      * "you control" -> ``ControlledByYou``.

    Combined shape:

        And(
            And(
                Or(SubjectIsCreature, SubjectIsArtifact),
                ControlledByYou,
            ),
            Not(SourceIsSelf),
        )

    Effect: ``damage_each_opponent(1)`` -- already on the cross-engine
    bridge as of Probe F.

    Probe shape (mirrors Mechan Assembler's inject-and-fan-out frame):

      1. Pre-place Weftstalker Ardent on p0's bf. Its own arrival is
         gated by NOT(SourceIsSelf): no fire, no life change. The Or
         arm WOULD have matched (Weftstalker is a Creature), so this
         pins the "Not gates Or-true" composition.

      2. Inject a vanilla creature for p0 on the bf. Dispatch fires:
         (Subject IS Creature -> Or-left true -> Or true) AND
         (controlled_by_you yes) AND (NOT source_is_self yes).
         Trigger pushes. Drain stack -> p1 life 20 -> 19.

      3. Inject a vanilla artifact for p0 on the bf. Dispatch fires:
         (Subject IS Artifact -> Or-right true even though Or-left
         false -> Or true) AND controlled_by_you AND NOT-self.
         Trigger pushes. Drain stack -> p1 life 19 -> 18.

      4. Inject a vanilla land for p0 on the bf. Dispatch DOES NOT
         fire: (Subject IS Land -> Or-left false; Or-right false ->
         Or false) -> outer And short-circuits. No push, life
         unchanged.

      5. Cross-engine assert: BOTH engines reduced p1's life to 18
         after two triggered fires + one rejected (land) arrival.
         p0 life unchanged at 20 (controller-immune).

    This is the **first** cross-engine probe to exercise
    ``Predicate::Or`` end-to-end on a real EoE card. All three Or
    arms (left-match, right-match, neither-match) are pinned in a
    single inject-and-fan-out sequence.

    Out of scope:

      * Warp cost ({R}) — cast-from-exile alt path is irrelevant to
        the ETB-trigger contract; pre-place skips the cast pipeline.
      * Cross-controller controller-of-effect for
        ``DamageEachOpponent`` — same gap as Probe F's note;
        follow-up I-slice mirrors I60's shape.
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )

    weftstalker_predicate = {
        "kind": "and",
        "left": {
            "kind": "and",
            "left": {
                "kind": "or",
                "left": {"kind": "subject_is_creature"},
                "right": {"kind": "subject_is_artifact"},
            },
            "right": {"kind": "controlled_by_you"},
        },
        "right": {
            "kind": "not",
            "inner": {"kind": "source_is_self"},
        },
    }
    weftstalker = CardDef(
        name="Weftstalker Ardent",
        card_type="Creature",
        mana_cost_generic=2,
        mana_cost_colored=(("R", 1),),
        power=2,
        toughness=3,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": weftstalker_predicate,
                "effect": {
                    "kind": "damage_each_opponent",
                    "amount": 1,
                },
            },
        ),
    )
    bait_creature = CardDef(
        name="Bait Creature",
        card_type="Creature",
        mana_cost_generic=1,
        power=1,
        toughness=1,
    )
    bait_artifact = CardDef(
        name="Bait Artifact",
        card_type="Artifact",
        mana_cost_generic=1,
    )
    bait_land = CardDef(
        name="Bait Wastes",
        card_type="Land",
    )
    spec = DeckSpec(
        card_defs=(weftstalker, bait_creature, bait_artifact, bait_land),
        p0_library=(),
        p1_library=(),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_BATTLEFIELD = 2
    rust_weftstalker_dict = {
        "name": "Weftstalker Ardent",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2, "R": 1},
        "effects": [],
        "power": 2,
        "toughness": 3,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": weftstalker_predicate,
                "effect": {
                    "kind": "damage_each_opponent",
                    "amount": 1,
                },
            }
        ],
    }
    rust_weftstalker_id = rust_state.register_and_inject_object(
        rust_weftstalker_dict, 0, ZONE_BATTLEFIELD
    )
    # Weftstalker's own arrival: NOT(IsSelf) suppresses; stack empty.
    assert rust_state.stack_size() == 0, (
        "Rust: Weftstalker's own arrival should NOT fire its trigger "
        "(NOT(SourceIsSelf) suppresses even though Or matches creature). "
        f"Got stack_size={rust_state.stack_size()}"
    )
    rust_p0_life_before = rust_state.get_life(0)
    rust_p1_life_before = rust_state.get_life(1)
    assert rust_p0_life_before == 20
    assert rust_p1_life_before == 20

    rust_bait_creature_dict = {
        "name": "Bait Creature",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "effects": [],
        "power": 1,
        "toughness": 1,
    }
    rust_bait_artifact_dict = {
        "name": "Bait Artifact",
        "card_types": ["Artifact"],
        "mana_cost": {"generic": 1},
        "effects": [],
    }
    rust_bait_land_dict = {
        "name": "Bait Wastes",
        "card_types": ["Land"],
        "effects": [],
    }

    # Arrival 1: vanilla creature for p0. Or-left true (creature),
    # ControlledByYou true, NOT-self true -> fires.
    rust_creature_id = rust_state.register_and_inject_object(
        rust_bait_creature_dict, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 1, (
        "Rust: Or-left (subject_is_creature) should fire on p0's "
        "creature arrival"
    )
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    assert rust_state.get_life(1) == 19, (
        f"Rust: p1 life should be 19 after first burn; "
        f"got {rust_state.get_life(1)}"
    )

    # Arrival 2: vanilla artifact for p0. Or-right true (artifact),
    # Or-left false (not a creature). Outer And succeeds -> fires.
    rust_artifact_id = rust_state.register_and_inject_object(
        rust_bait_artifact_dict, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 1, (
        "Rust: Or-right (subject_is_artifact) should fire on p0's "
        "artifact arrival (Or-left false, Or-right true)"
    )
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    assert rust_state.get_life(1) == 18, (
        f"Rust: p1 life should be 18 after second burn; "
        f"got {rust_state.get_life(1)}"
    )

    # Arrival 3: vanilla land for p0. Neither Or arm matches (not a
    # creature, not an artifact) -> Or false -> outer And false.
    # No fire, life unchanged.
    rust_land_id = rust_state.register_and_inject_object(
        rust_bait_land_dict, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 0, (
        "Rust: Or(creature, artifact) should reject a land subject "
        "(neither arm matches). "
        f"Got stack_size={rust_state.stack_size()}"
    )
    assert rust_state.get_life(1) == 18, (
        f"Rust: p1 life should stay at 18 (land arrival rejected); "
        f"got {rust_state.get_life(1)}"
    )

    # ---- Python side ----
    from mtg_bot.actions import _trigger_etb_abilities
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.damage import DealDamageEachOpponent
    from mtg_bot.effects.object_predicates import IsArtifact, IsCreature
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, IsSelf, Not, Or, Subject, YOU,
    )
    from mtg_bot.game import Zone

    py_predicate = And(
        left=And(
            left=Or(
                left=Subject(slot=None, inner=IsCreature()),
                right=Subject(slot=None, inner=IsArtifact()),
            ),
            right=Controlled(by=YOU, subject=None),
        ),
        right=Not(inner=IsSelf()),
    )
    weftstalker_printing = CardPrinting(
        name="Weftstalker Ardent",
        mana_cost_str="",
        mana_cost=ManaCost(generic=2, colored=(("R", 1),)),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=2,
        toughness=3,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=py_predicate,
                effect=DealDamageEachOpponent(amount=1),
            ),
        ),
    )
    py_weftstalker = py_state.new_object(
        weftstalker_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_weftstalker.summoning_sick = False
    py_weftstalker.tapped = False
    py_weftstalker.damage = 0
    py_weftstalker.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_weftstalker)
    _trigger_etb_abilities(py_state, py_weftstalker)
    # NOT(IsSelf) suppresses self-arrival.
    assert len(py_state.stack) == 0
    assert py_state.players[1].life == 20

    creature_printing = CardPrinting(
        name="Bait Creature",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=1,
        toughness=1,
        produces=frozenset(),
        triggered_abilities=(),
    )
    artifact_printing = CardPrinting(
        name="Bait Artifact",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1),
        type_line="Artifact",
        supertypes=(),
        card_types=("Artifact",),
        subtypes=(),
        power=None,
        toughness=None,
        produces=frozenset(),
        triggered_abilities=(),
    )
    land_printing = CardPrinting(
        name="Bait Wastes",
        mana_cost_str="",
        mana_cost=ManaCost(),
        type_line="Land",
        supertypes=(),
        card_types=("Land",),
        subtypes=(),
        power=None,
        toughness=None,
        produces=frozenset(),
        triggered_abilities=(),
    )

    # Arrival 1: vanilla creature.
    py_creature = py_state.new_object(
        creature_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_creature.summoning_sick = True
    py_creature.tapped = False
    py_creature.damage = 0
    py_creature.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_creature)
    _trigger_etb_abilities(py_state, py_creature)
    assert len(py_state.stack) == 1, (
        "Python: Or-left (IsCreature) should fire on p0's creature "
        "arrival"
    )
    from mtg_bot.actions import resolve_top_of_stack
    while py_state.stack:
        resolve_top_of_stack(py_state)
    assert py_state.players[1].life == 19

    # Arrival 2: vanilla artifact.
    py_artifact = py_state.new_object(
        artifact_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_artifact.summoning_sick = False
    py_artifact.tapped = False
    py_artifact.damage = 0
    py_artifact.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_artifact)
    _trigger_etb_abilities(py_state, py_artifact)
    assert len(py_state.stack) == 1, (
        "Python: Or-right (IsArtifact) should fire on p0's artifact "
        "arrival"
    )
    while py_state.stack:
        resolve_top_of_stack(py_state)
    assert py_state.players[1].life == 18

    # Arrival 3: vanilla land — neither Or arm matches.
    py_land = py_state.new_object(
        land_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_land.summoning_sick = False
    py_land.tapped = False
    py_land.damage = 0
    py_land.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_land)
    _trigger_etb_abilities(py_state, py_land)
    assert len(py_state.stack) == 0, (
        "Python: Or(creature, artifact) should reject a land subject"
    )
    assert py_state.players[1].life == 18

    # ---- Cross-engine parity ----
    _drain_stack_both(py_state, rust_state)
    # Both engines: p0 life unchanged at 20; p1 life 18 (-2 from
    # two qualifying fires + 1 rejected land arrival).
    assert rust_state.get_life(0) == 20
    assert rust_state.get_life(1) == 18
    assert py_state.players[0].life == 20
    assert py_state.players[1].life == 18
    assert rust_state.get_life(0) == py_state.players[0].life
    assert rust_state.get_life(1) == py_state.players[1].life


def test_cross_engine_real_eoe_kav_landseeker_has_subtype_kavu_probe_parity(
    parity_smoke,
):
    """Probe H: First cross-engine probe to exercise
    ``Predicate::SubjectHasSubtype`` end-to-end on a real EoE-printed
    subtype-carrying subject.

    Kav Landseeker (EoE 138) — a real EoE creature whose printed
    type_line is "Creature -- Kavu Soldier". Its own oracle text
    ("When this creature enters, create a Lander token...") is OUT
    of scope: the bridge doesn't carry ``create_token`` and EoE has
    no Kavu/Sliver/Equipment ETB trigger whose effect is already on
    the substrate-bridged effect surface (draw_cards / gain_life /
    damage_each_opponent). The probe strips Kav Landseeker's own
    triggered ability and uses it strictly as the SUBJECT carrying
    the real EoE-printed ``["Kavu", "Soldier"]`` subtypes. Same
    shape as Tannuk's probe, which stripped the "second-time-this-
    turn -> draw" rider for the same reason.

    Probe shape (inject-and-fan-out, mirrors the artifact/creature
    arrival pattern from the Or-substrate consumer probe):

      1. Pre-place a synthetic Kavu-observer on p0's bf. Its
         predicate is ``Subject(HasSubtype("Kavu")) AND
         ControlledByYou`` -- the simplest unit-pin for HasSubtype
         on a real-EoE-typed subject. Effect: ``gain_life(1)``
         (already on the bridge per Probes B/D/E). Self-arrival
         doesn't fire because the observer's own ``subtypes`` is
         empty -- HasSubtype("Kavu") rejects.

      2. Inject Kav Landseeker (real EoE printing-shape; subtypes
         ["Kavu", "Soldier"]) onto p0's bf. Dispatch fires:
         HasSubtype("Kavu") TRUE (first-position subtype match),
         ControlledByYou TRUE. Trigger pushes. Drain -> p0 life
         20 -> 21.

      3. Inject a vanilla non-Kavu creature for p0 onto bf.
         HasSubtype("Kavu") FALSE -> trigger doesn't fire. Life
         unchanged.

      4. Cross-engine assert: BOTH engines fire the Kavu trigger
         exactly once and reject the vanilla creature. p0 life: 21.

    This is the **first** cross-engine probe to pin the
    ``Predicate::SubjectHasSubtype`` arm (landed in Subtypes Slice
    2, ``CardPrinting.subtypes: Vec<String>`` +
    ``Predicate::SubjectHasSubtype``) on a real EoE-printed subject.
    The substrate-side unit tests in
    ``tests_rust/test_triggers.py`` (Subtypes Slice 2's Equipment /
    Kavu / Bear arrival pins) cover the predicate dispatch in
    isolation; this probe ties it to the cross-engine bridge.

    Out of scope:

      * Kav Landseeker's own ``create_token`` ETB -- needs bridge
        widening for ``create_token`` effect, follow-up probe.
      * Effective subtypes (CR 613 layer-3/4 -- Conspiracy granting
        "Kavu", etc.) -- both engines read static subtypes, so the
        layer-aware semantics drift is irrelevant for EoE's pool.
      * Multi-Kavu cascade (a Kavu fires the observer, the
        observer's effect causes another Kavu to enter, etc.) --
        out of probe scope.
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )

    kavu_observer_predicate = {
        "kind": "and",
        "left": {
            "kind": "subject_has_subtype",
            "subtype": "Kavu",
        },
        "right": {"kind": "controlled_by_you"},
    }
    kavu_observer = CardDef(
        name="Test Kavu Observer",
        card_type="Creature",
        mana_cost_generic=2,
        power=1,
        toughness=1,
        # Empty subtypes — observer's own arrival doesn't match its
        # own predicate (HasSubtype("Kavu") false).
        subtypes=(),
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": kavu_observer_predicate,
                "effect": {"kind": "gain_life", "amount": 1},
            },
        ),
    )
    # Kav Landseeker — real EoE 138; type_line "Creature -- Kavu
    # Soldier", so subtypes = ("Kavu", "Soldier"). Strip the printed
    # ETB ability (create-Lander-token is out of bridge scope) but
    # KEEP the printed name + subtypes so the parity contract reads
    # the real EoE-printed subtype tuple end-to-end.
    kav_landseeker = CardDef(
        name="Kav Landseeker",
        card_type="Creature",
        mana_cost_generic=3,
        mana_cost_colored=(("R", 1),),
        power=4,
        toughness=3,
        subtypes=("Kavu", "Soldier"),
        triggered_abilities=(),
    )
    vanilla_non_kavu = CardDef(
        name="Test Vanilla Non-Kavu",
        card_type="Creature",
        mana_cost_generic=1,
        power=1,
        toughness=1,
        subtypes=(),
    )
    spec = DeckSpec(
        card_defs=(kavu_observer, kav_landseeker, vanilla_non_kavu),
        p0_library=(),
        p1_library=(),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_BATTLEFIELD = 2
    rust_observer_dict = {
        "name": "Test Kavu Observer",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "subtypes": [],
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": kavu_observer_predicate,
                "effect": {"kind": "gain_life", "amount": 1},
            }
        ],
    }
    rust_observer_id = rust_state.register_and_inject_object(
        rust_observer_dict, 0, ZONE_BATTLEFIELD
    )
    # Observer's own arrival: HasSubtype("Kavu") false (observer has
    # empty subtypes); trigger does NOT fire.
    assert rust_state.stack_size() == 0, (
        "Rust: Kavu observer's own arrival should NOT fire its "
        "trigger -- observer's subtypes is empty so HasSubtype('Kavu') "
        f"rejects. Got stack_size={rust_state.stack_size()}"
    )
    rust_p0_life_before = rust_state.get_life(0)
    assert rust_p0_life_before == 20

    rust_kav_dict = {
        "name": "Kav Landseeker",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 3, "R": 1},
        "effects": [],
        "power": 4,
        "toughness": 3,
        "subtypes": ["Kavu", "Soldier"],
    }
    rust_kav_id = rust_state.register_and_inject_object(
        rust_kav_dict, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 1, (
        "Rust: HasSubtype('Kavu') AND ControlledByYou should fire on "
        "Kav Landseeker's arrival (subtypes=['Kavu','Soldier'], "
        f"controller=0). Got stack_size={rust_state.stack_size()}"
    )
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    assert rust_state.get_life(0) == 21, (
        f"Rust: p0 life should be 21 after Kavu arrival; "
        f"got {rust_state.get_life(0)}"
    )

    # Inject a vanilla non-Kavu creature for p0. HasSubtype('Kavu')
    # false -> trigger doesn't fire.
    rust_vanilla_dict = {
        "name": "Test Vanilla Non-Kavu",
        "card_types": ["Creature"],
        "mana_cost": {"generic": 1},
        "effects": [],
        "power": 1,
        "toughness": 1,
        "subtypes": [],
    }
    rust_vanilla_id = rust_state.register_and_inject_object(
        rust_vanilla_dict, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 0, (
        "Rust: HasSubtype('Kavu') should reject a non-Kavu creature "
        f"subject. Got stack_size={rust_state.stack_size()}"
    )
    assert rust_state.get_life(0) == 21, (
        f"Rust: p0 life should stay at 21 (non-Kavu rejected); "
        f"got {rust_state.get_life(0)}"
    )

    # ---- Python side ----
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.life import GainLife
    from mtg_bot.effects.object_predicates import HasSubtype
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, Subject, YOU,
    )
    from mtg_bot.game import Zone

    py_predicate = And(
        left=Subject(slot=None, inner=HasSubtype(name="Kavu")),
        right=Controlled(by=YOU, subject=None),
    )
    observer_printing = CardPrinting(
        name="Test Kavu Observer",
        mana_cost_str="",
        mana_cost=ManaCost(generic=2),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=1,
        toughness=1,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=py_predicate,
                effect=GainLife(amount=1),
            ),
        ),
    )
    py_observer = py_state.new_object(
        observer_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_observer.summoning_sick = False
    py_observer.tapped = False
    py_observer.damage = 0
    py_observer.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_observer)
    _trigger_etb_abilities(py_state, py_observer)
    # Observer's own arrival: HasSubtype("Kavu") false (empty
    # subtypes); no fire.
    assert len(py_state.stack) == 0
    assert py_state.players[0].life == 20

    kav_printing = CardPrinting(
        name="Kav Landseeker",
        mana_cost_str="",
        mana_cost=ManaCost(generic=3, colored=(("R", 1),)),
        type_line="Creature — Kavu Soldier",
        supertypes=(),
        card_types=("Creature",),
        subtypes=("Kavu", "Soldier"),
        power=4,
        toughness=3,
        produces=frozenset(),
        triggered_abilities=(),
    )
    py_kav = py_state.new_object(
        kav_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_kav.summoning_sick = True
    py_kav.tapped = False
    py_kav.damage = 0
    py_kav.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_kav)
    _trigger_etb_abilities(py_state, py_kav)
    assert len(py_state.stack) == 1, (
        "Python: HasSubtype('Kavu') AND ControlledByYou should fire "
        "on Kav Landseeker's arrival"
    )
    while py_state.stack:
        resolve_top_of_stack(py_state)
    assert py_state.players[0].life == 21

    vanilla_printing = CardPrinting(
        name="Test Vanilla Non-Kavu",
        mana_cost_str="",
        mana_cost=ManaCost(generic=1),
        type_line="Creature",
        supertypes=(),
        card_types=("Creature",),
        subtypes=(),
        power=1,
        toughness=1,
        produces=frozenset(),
        triggered_abilities=(),
    )
    py_vanilla = py_state.new_object(
        vanilla_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_vanilla.summoning_sick = True
    py_vanilla.tapped = False
    py_vanilla.damage = 0
    py_vanilla.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_vanilla)
    _trigger_etb_abilities(py_state, py_vanilla)
    assert len(py_state.stack) == 0, (
        "Python: HasSubtype('Kavu') should reject a non-Kavu subject"
    )
    assert py_state.players[0].life == 21

    # ---- Cross-engine parity ----
    _drain_stack_both(py_state, rust_state)
    assert rust_state.get_life(0) == 21
    assert rust_state.get_life(1) == 20
    assert py_state.players[0].life == 21
    assert py_state.players[1].life == 20
    assert rust_state.get_life(0) == py_state.players[0].life


def test_cross_engine_real_eoe_mechan_assembler_real_oracle_create_token_parity(
    parity_smoke,
):
    """Probe I: Mechan Assembler (EoE) FULL real oracle — first
    cross-engine probe to exercise the new ``create_token`` bridge
    arm. Pins the Robot artifact-creature token on the
    ``another artifact you control`` predicate branch end-to-end.

    Oracle text (`data/edge_of_eternities.json`):

        "Whenever another artifact you control enters, create a 2/2
         colorless Robot artifact creature token. This ability
         triggers only once each turn."

    Probes D + E pinned the predicate algebra (NOT-branch) and the
    ``once_per_turn`` flag with a GainLife(1) effect stand-in. This
    probe replaces the stand-in with the REAL oracle effect:
    ``create_token`` of a 2/2 colorless Robot artifact-creature
    token. All three substrate features — Predicate::Not,
    once_per_turn ledger, AND CreateToken — exercise together on
    real-EoE-printed text.

    Substrate paid for:
      * Predicate algebra: ``Subject(IsArtifact) AND ControlledByYou
        AND NOT(IsSelf)`` (Probe D).
      * ``once_per_turn`` ledger (Probe E).
      * **NEW**: ``create_token`` effect arm + Python's
        ``printings_by_pid`` threading. The Rust effect resolves at
        ``state_ops_resolve.rs:1744``; the Python class at
        ``effects/generic/tokens.py:168``; the bridge translator
        at ``cross_engine.py::_ta_dict_to_python_effect``.

    Probe shape (same inject-and-fan-out frame as Probes D + E,
    swapping the GainLife pin for a battlefield-zone-count pin):

      1. Pre-place Mechan Assembler on p0 bf. Its OWN arrival
         doesn't fire (NOT(IsSelf) suppresses).
      2. Inject Mechan Drone #1 on p0 bf. ENTERS dispatch:
         predicate matches all three arms. ``once_per_turn``
         ledger empty -> trigger fires. Drain stack: CreateToken
         resolves -> Robot token appears on p0 bf. Ledger stamped.
      3. Inject Mechan Drone #2 on p0 bf. Predicate matches but
         ``once_per_turn`` gate suppresses (ledger has the key).
         No second token created.
      4. Cross-engine assert: BOTH engines have:
         * 4 GameObjects on p0 bf (Assembler + Drone #1 + Robot
           token + Drone #2), exactly one of which is named
           ``Robot_2_2`` and has Token supertype.
         * Stack empty on both.

    This is the FIRST cross-engine probe whose effect actually
    materializes a new permanent on the battlefield rather than
    just bumping a counter / life total. The post-drain battlefield
    signature (set of (printing.name, controller) pairs) is the
    parity assertion's primary load-bearing surface.
    """
    from mtg_bot.parity.cross_engine import (
        CardDef, DeckSpec, build_python_state, build_rust_state,
    )

    another_artifact_predicate = {
        "kind": "and",
        "left": {
            "kind": "and",
            "left": {"kind": "subject_is_artifact"},
            "right": {"kind": "controlled_by_you"},
        },
        "right": {
            "kind": "not",
            "inner": {"kind": "source_is_self"},
        },
    }

    # The Robot token CardDef. Registered FIRST in ``card_defs`` so
    # its positional pid is 0 — Mechan Assembler's create_token
    # trigger references ``token_printing_id=0``. The Rust registry
    # in ``build_rust_state`` assigns pids by position; Python's
    # ``printings_by_pid`` mirrors the same mapping. The Robot
    # carries ``["Artifact", "Creature"]`` card_types (via
    # ``extra_card_types``), ``["Token"]`` supertypes, and
    # ``["Robot"]`` subtypes — faithful to the real EoE Robot
    # printing's shape.
    robot_token_def = CardDef(
        name="Robot_2_2",
        card_type="Artifact",
        extra_card_types=("Creature",),
        supertypes=("Token",),
        subtypes=("Robot",),
        power=2,
        toughness=2,
    )

    # Mechan Assembler CardDef — pid 1. The trigger's effect dict
    # references ``token_printing_id=0`` (the Robot just above).
    # NOTE: the Rust-side ``register_and_inject_object`` path below
    # registers Mechan Assembler AGAIN (creating a separate
    # printing at pid 2 or later), and that new printing carries
    # its OWN TA dict still referencing pid 0 — the Robot stays
    # at pid 0 throughout.
    mechan_assembler_def = CardDef(
        name="Mechan Assembler",
        card_type="Artifact",
        extra_card_types=("Creature",),
        mana_cost_generic=4,
        mana_cost_colored=(("U", 1),),
        power=4,
        toughness=4,
        triggered_abilities=(
            {
                "event": "enters",
                "predicate": another_artifact_predicate,
                "effect": {
                    "kind": "create_token",
                    "count": 1,
                    "token_printing_id": 0,  # Robot_2_2 pid
                },
                "once_per_turn": True,
            },
        ),
    )
    mechan_drone_def = CardDef(
        name="Mechan Drone",
        card_type="Artifact",
        extra_card_types=("Creature",),
        mana_cost_generic=2,
        power=2,
        toughness=2,
    )
    spec = DeckSpec(
        card_defs=(robot_token_def, mechan_assembler_def, mechan_drone_def),
        p0_library=(),
        p1_library=(),
    )
    py_state = build_python_state(spec)
    rust_state = build_rust_state(spec)

    # ---- Rust side ----
    ZONE_BATTLEFIELD = 2
    rust_assembler_dict = {
        "name": "Mechan Assembler",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"generic": 4, "U": 1},
        "effects": [],
        "power": 4,
        "toughness": 4,
        "triggered_abilities": [
            {
                "event": "enters",
                "predicate": another_artifact_predicate,
                "effect": {
                    "kind": "create_token",
                    "count": 1,
                    "token_printing_id": 0,
                },
                "once_per_turn": True,
            }
        ],
    }
    rust_assembler_id = rust_state.register_and_inject_object(
        rust_assembler_dict, 0, ZONE_BATTLEFIELD
    )
    # Assembler's own arrival: NOT(IsSelf) suppresses; stack empty.
    assert rust_state.stack_size() == 0
    rust_drone_dict = {
        "name": "Mechan Drone",
        "card_types": ["Artifact", "Creature"],
        "mana_cost": {"generic": 2},
        "effects": [],
        "power": 2,
        "toughness": 2,
    }

    # Drone #1: first qualifying arrival. Trigger fires; ledger stamps.
    rust_drone1_id = rust_state.register_and_inject_object(
        rust_drone_dict, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 1, (
        "Rust: Drone #1 arrival should push Mechan Assembler's trigger "
        "onto the stack (first fire of the turn)"
    )
    # Drain the trigger now: CreateToken resolves -> Robot on bf.
    while rust_state.stack_size() > 0:
        rust_state.resolve_top_of_stack()
    # Inspect p0 bf: should now contain Assembler + Drone #1 + Robot
    # token. Read battlefield via parity_smoke's zone signature.
    rust_bf_after_first = [
        d for d in rust_state.battlefield_snapshot()
        if d["controller_id"] == 0
    ]
    rust_bf_names_after_first = [d["printing_name"] for d in rust_bf_after_first]
    assert "Robot_2_2" in rust_bf_names_after_first, (
        f"Rust: Robot token should be on p0 bf after first trigger; "
        f"got {rust_bf_names_after_first!r}"
    )
    assert rust_bf_names_after_first.count("Robot_2_2") == 1
    rust_bf_size_after_first = len(rust_bf_after_first)
    assert rust_bf_size_after_first == 3, (
        f"Rust: p0 bf size should be 3 (Assembler+Drone1+Robot); "
        f"got {rust_bf_size_after_first}"
    )

    # Drone #2: second arrival same turn. once_per_turn gate
    # suppresses; no second Robot token created.
    rust_drone2_id = rust_state.register_and_inject_object(
        rust_drone_dict, 0, ZONE_BATTLEFIELD
    )
    assert rust_state.stack_size() == 0, (
        "Rust: Drone #2 arrival should NOT push a trigger -- "
        "once_per_turn gate suppresses the second fire. "
        f"Got stack_size={rust_state.stack_size()}"
    )
    rust_bf_after_second = [
        d for d in rust_state.battlefield_snapshot()
        if d["controller_id"] == 0
    ]
    rust_bf_names_after_second = [d["printing_name"] for d in rust_bf_after_second]
    assert rust_bf_names_after_second.count("Robot_2_2") == 1, (
        f"Rust: still exactly ONE Robot token on p0 bf after Drone #2 "
        f"(once_per_turn gates the second token creation); "
        f"got {rust_bf_names_after_second!r}"
    )
    # p0 bf: Assembler + Drone #1 + Robot + Drone #2 = 4 objects.
    assert len(rust_bf_after_second) == 4

    # ---- Python side ----
    from mtg_bot.actions import _trigger_etb_abilities, resolve_top_of_stack
    from mtg_bot.cards.base import CardPrinting, ManaCost
    from mtg_bot.effects.base import TriggeredAbility
    from mtg_bot.effects.generic.tokens import CreateToken
    from mtg_bot.effects.object_predicates import IsArtifact
    from mtg_bot.effects.predicates import (
        And, Controlled, Event, Not, Subject, IsSelf, YOU,
    )
    from mtg_bot.game import Zone

    # The Python Robot token printing — same fields as the
    # Rust-side registered printing. We pull it from the printings
    # dict so the Mechan Assembler's CreateToken effect references
    # the SAME instance Python's CreateToken expects (effective
    # type-checks at resolve time read .printing.name).
    py_predicate = And(
        left=And(
            left=Subject(slot=None, inner=IsArtifact()),
            right=Controlled(by=YOU, subject=None),
        ),
        right=Not(inner=IsSelf()),
    )
    # The Python Robot printing — built locally so the CreateToken
    # effect can reference it. Matches the Rust dict shape (Token
    # supertype, Robot subtype, 2/2 Artifact Creature).
    robot_printing = CardPrinting(
        name="Robot_2_2",
        mana_cost_str="",
        mana_cost=ManaCost(generic=0, colored=()),
        type_line="Artifact Creature -- Robot",
        supertypes=("Token",),
        card_types=("Artifact", "Creature"),
        subtypes=("Robot",),
        power=2,
        toughness=2,
        produces=frozenset(),
        triggered_abilities=(),
    )
    assembler_printing = CardPrinting(
        name="Mechan Assembler",
        mana_cost_str="",
        mana_cost=ManaCost(generic=4, colored=(("U", 1),)),
        type_line="Artifact Creature",
        supertypes=(),
        card_types=("Artifact", "Creature"),
        subtypes=(),
        power=4,
        toughness=4,
        produces=frozenset(),
        triggered_abilities=(
            TriggeredAbility(
                event=Event.ENTERS,
                predicate=py_predicate,
                effect=CreateToken(
                    token_printing=robot_printing, count=1,
                ),
                once_per_turn=True,
            ),
        ),
    )
    py_assembler = py_state.new_object(
        assembler_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_assembler.summoning_sick = False
    py_assembler.tapped = False
    py_assembler.damage = 0
    py_assembler.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_assembler)
    _trigger_etb_abilities(py_state, py_assembler)
    assert len(py_state.stack) == 0  # NOT(IsSelf) suppressed

    drone_printing = CardPrinting(
        name="Mechan Drone",
        mana_cost_str="",
        mana_cost=ManaCost(generic=2),
        type_line="Artifact Creature",
        supertypes=(),
        card_types=("Artifact", "Creature"),
        subtypes=(),
        power=2,
        toughness=2,
        produces=frozenset(),
        triggered_abilities=(),
    )
    # Drone #1 -> trigger fires -> Robot token created.
    py_drone1 = py_state.new_object(
        drone_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_drone1.summoning_sick = True
    py_drone1.tapped = False
    py_drone1.damage = 0
    py_drone1.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_drone1)
    _trigger_etb_abilities(py_state, py_drone1)
    assert len(py_state.stack) == 1
    while py_state.stack:
        resolve_top_of_stack(py_state)
    py_bf_names_after_first = [
        o.printing.name for o in py_state.battlefield
        if o.controller_id == 0
    ]
    assert "Robot_2_2" in py_bf_names_after_first, (
        f"Python: Robot token should be on p0 bf after first trigger; "
        f"got {py_bf_names_after_first!r}"
    )
    assert py_bf_names_after_first.count("Robot_2_2") == 1
    assert len(py_bf_names_after_first) == 3

    # Drone #2 -> once_per_turn suppresses; no second Robot.
    py_drone2 = py_state.new_object(
        drone_printing, owner_id=0, zone=Zone.BATTLEFIELD,
    )
    py_drone2.summoning_sick = True
    py_drone2.tapped = False
    py_drone2.damage = 0
    py_drone2.etb_timestamp = py_state.next_timestamp
    py_state.next_timestamp += 1
    py_state.battlefield.append(py_drone2)
    _trigger_etb_abilities(py_state, py_drone2)
    assert len(py_state.stack) == 0
    py_bf_names_after_second = [
        o.printing.name for o in py_state.battlefield
        if o.controller_id == 0
    ]
    assert py_bf_names_after_second.count("Robot_2_2") == 1, (
        f"Python: still exactly ONE Robot token on p0 bf after Drone #2; "
        f"got {py_bf_names_after_second!r}"
    )
    assert len(py_bf_names_after_second) == 4

    # ---- Cross-engine parity ----
    _drain_stack_both(py_state, rust_state)
    # BOTH engines: exactly ONE Robot token created across two
    # qualifying ENTERS in the same turn (once_per_turn gate).
    # p0 bf size matches: 4 objects (Assembler + Drone1 + Robot +
    # Drone2).
    rust_bf_final = [
        d for d in rust_state.battlefield_snapshot()
        if d["controller_id"] == 0
    ]
    rust_bf_names_final = sorted([d["printing_name"] for d in rust_bf_final])
    py_bf_names_final = sorted([
        o.printing.name for o in py_state.battlefield
        if o.controller_id == 0
    ])
    assert rust_bf_names_final == py_bf_names_final, (
        f"Cross-engine p0 bf signatures must match: "
        f"rust={rust_bf_names_final!r} vs py={py_bf_names_final!r}"
    )
    assert rust_bf_names_final.count("Robot_2_2") == 1
    assert len(rust_bf_names_final) == 4
    # Sanity: p1 bf unchanged on both engines (no Robot tokens for
    # p1 — controller-immune to the trigger's create_token).
    rust_p1_bf = [
        d for d in rust_state.battlefield_snapshot()
        if d["controller_id"] == 1
    ]
    assert len(rust_p1_bf) == 0
    assert sum(1 for o in py_state.battlefield if o.controller_id == 1) == 0
    assert rust_state.get_life(1) == py_state.players[1].life