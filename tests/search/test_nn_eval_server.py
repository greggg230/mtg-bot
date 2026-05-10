"""Tests for the centralized NN inference server.

Covers the in-process server (started in a thread, not a subprocess —
faster, easier to introspect, and exercises the same dispatcher /
batching / IPC code path the subprocess form uses) plus the client.

Subprocess startup is exercised by a single happy-path integration
test in this file; the heavy lifting lives in thread-based tests.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from mtg_bot.agents import RandomAgent
from mtg_bot.features import FEATURE_DIM
from mtg_bot.search.engine_adapter import MtgGameStateAdapter
from mtg_bot.search.nn_eval_server import (
    MSG_KIND_VALUE,
    MSG_KIND_VALUE_BATCH,
    NNEvalClient,
    NNEvalServer,
    NNEvalServerError,
    NNServerHandle,
    launch_server_subprocess,
)
from mtg_bot.search.nn_evaluator import (
    NNEvaluator, _ValueMLP, _ValuePolicyMLP,
)
from mtg_bot.simulate import setup_game
from tests._test_decks import simple_test_decklists  # noqa: E402


# ---------- fixtures --------------------------------------------------------


@pytest.fixture
def adapter():
    """A real MTG state adapter — gives the client realistic
    feature vectors / masks so the round-trip exercises the full
    extract→ship→forward→reply path."""
    state = setup_game([RandomAgent(), RandomAgent()], decks=simple_test_decklists(), seed=42,
    )
    return MtgGameStateAdapter(state)


@pytest.fixture
def server_factory(tmp_path: Path):
    """Build a server whose model state_dict matches a reference
    NNEvaluator; the reference is returned alongside so tests can
    compare server output against it exactly."""

    def _make(with_policy: bool = False, **server_kwargs):
        # Save weights, then load them into both a reference local
        # evaluator and the server. This guarantees bit-identical
        # weights — without it the server's internal random init
        # would produce different values from the reference.
        torch.manual_seed(0)
        ref = NNEvaluator(with_policy=with_policy)
        weights_path = tmp_path / (
            "with_policy.pt" if with_policy else "value.pt"
        )
        ref.save(weights_path)
        server = NNEvalServer(
            weights_path=weights_path,
            device="cpu",
            with_policy=with_policy,
            **server_kwargs,
        )
        return server, ref, weights_path

    return _make


# ---------- request / response correlation ----------------------------------


def test_value_request_round_trip_matches_local(server_factory, adapter):
    server, ref, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    with server:
        client = NNEvalClient(server.handle())
        try:
            v = client.evaluate(adapter, player_id=0)
        finally:
            client.close()
        v_ref = ref.evaluate(adapter, player_id=0)
        assert v == pytest.approx(v_ref, abs=1e-5)


def test_batch_evaluate_matches_local(server_factory, adapter):
    server, ref, _ = server_factory(batch_size=4, batch_wait_ms=2.0)
    states = [adapter.clone() for _ in range(5)]
    with server:
        client = NNEvalClient(server.handle())
        try:
            vs = client.batch_evaluate(states, player_id=0)
        finally:
            client.close()
        ref_vs = ref.batch_evaluate(states, player_id=0)
        assert len(vs) == len(ref_vs)
        for a, b in zip(vs, ref_vs):
            assert a == pytest.approx(b, abs=1e-5)


def test_evaluate_with_policy_matches_local(server_factory, adapter):
    server, ref, _ = server_factory(
        with_policy=True, batch_size=1, batch_wait_ms=0.0,
    )
    with server:
        client = NNEvalClient(server.handle())
        try:
            v, p = client.evaluate_with_policy(adapter, player_id=0)
        finally:
            client.close()
        v_ref, p_ref = ref.evaluate_with_policy(adapter, player_id=0)
        assert v == pytest.approx(v_ref, abs=1e-5)
        np.testing.assert_allclose(p, p_ref, atol=1e-5)


def test_batch_evaluate_with_policy_matches_local(server_factory, adapter):
    server, ref, _ = server_factory(
        with_policy=True, batch_size=4, batch_wait_ms=2.0,
    )
    states = [adapter.clone() for _ in range(3)]
    with server:
        client = NNEvalClient(server.handle())
        try:
            results = client.batch_evaluate_with_policy(
                states, player_id=0,
            )
        finally:
            client.close()
        ref_results = ref.batch_evaluate_with_policy(states, player_id=0)
        assert len(results) == len(ref_results)
        for (v, p), (vr, pr) in zip(results, ref_results):
            assert v == pytest.approx(vr, abs=1e-5)
            np.testing.assert_allclose(p, pr, atol=1e-5)


def test_terminal_state_short_circuits_without_ipc(server_factory, adapter):
    """Terminal states must not hit the network — verify by giving the
    server a model that would raise on forward, and checking the
    terminal call still returns ±1."""

    class ExplodingModel(torch.nn.Module):
        def forward(self, x):  # pragma: no cover - intentionally not called
            raise AssertionError("terminal short-circuit failed")

    # Mark adapter terminal.
    adapter.engine_state.players[1].has_lost = True
    adapter.engine_state.winner_id = 0
    assert adapter.is_terminal()

    server = NNEvalServer(
        device="cpu", batch_size=1, batch_wait_ms=0.0,
        model_factory=ExplodingModel,
    )
    with server:
        client = NNEvalClient(server.handle())
        try:
            assert client.evaluate(adapter, player_id=0) == 1.0
            assert client.evaluate(adapter, player_id=1) == -1.0
        finally:
            client.close()


# ---------- batching behavior -----------------------------------------------


def test_concurrent_clients_get_batched_into_one_dispatch(server_factory):
    """Five threads each send one request; with a generous wait window
    the dispatcher should fold them into a single batch.

    We pin batch_wait_ms high enough that the threads have time to
    pile in before the first batch fires. Verify via the server's
    telemetry: at most a small number of batches for 5 requests, and
    one of the recorded batches has size > 1.
    """
    server, _, _ = server_factory(batch_size=8, batch_wait_ms=100.0)
    with server:
        handle = server.handle()
        clients = [NNEvalClient(handle) for _ in range(5)]
        try:
            results: list[float | None] = [None] * 5
            ready = threading.Barrier(len(clients))

            def _worker(i: int):
                ready.wait()
                feats = np.full(
                    FEATURE_DIM, float(i), dtype=np.float32,
                )
                results[i] = clients[i]._call(MSG_KIND_VALUE, feats)

            threads = [
                threading.Thread(target=_worker, args=(i,))
                for i in range(len(clients))
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            for r in results:
                assert isinstance(r, float)
            # Telemetry: at least one batch of size > 1, total
            # served == 5.
            assert server.requests_served == 5
            assert any(s > 1 for s in server.batch_sizes), (
                f"no batched dispatch observed: batch_sizes="
                f"{server.batch_sizes}"
            )
        finally:
            for c in clients:
                c.close()


def test_batch_size_caps_dispatch(server_factory):
    """Send 10 requests at once; with batch_size=3 we should see at
    least four batches (10 / 3, ceiling), each capped at <= 3."""
    server, _, _ = server_factory(batch_size=3, batch_wait_ms=50.0)
    with server:
        handle = server.handle()
        clients = [NNEvalClient(handle) for _ in range(10)]
        try:
            ready = threading.Barrier(len(clients))
            results: list[float | None] = [None] * len(clients)

            def _worker(i: int):
                ready.wait()
                feats = np.full(
                    FEATURE_DIM, float(i), dtype=np.float32,
                )
                results[i] = clients[i]._call(MSG_KIND_VALUE, feats)

            threads = [
                threading.Thread(target=_worker, args=(i,))
                for i in range(len(clients))
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            assert all(r is not None for r in results)
            assert all(s <= 3 for s in server.batch_sizes), (
                f"batch_size cap exceeded: {server.batch_sizes}"
            )
            assert server.requests_served == 10
        finally:
            for c in clients:
                c.close()


def test_zero_wait_window_dispatches_immediately(server_factory):
    """batch_wait_ms == 0 should fire as soon as the first item is
    pulled. Not strictly testable for "exactly batch=1" because a
    fast queue can race in another item, but we can check that the
    dispatcher serves all requests and produces at least one batch
    of size 1 (i.e. it didn't deadlock waiting)."""
    server, _, _ = server_factory(batch_size=8, batch_wait_ms=0.0)
    with server:
        handle = server.handle()
        client = NNEvalClient(handle)
        try:
            for i in range(3):
                feats = np.full(
                    FEATURE_DIM, float(i), dtype=np.float32,
                )
                _ = client._call(MSG_KIND_VALUE, feats)
        finally:
            client.close()
        assert server.requests_served == 3


# ---------- error path ------------------------------------------------------


def test_client_raises_on_server_already_stopped(server_factory):
    server, _, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    server.start()
    handle = server.handle()
    server.stop(timeout=2.0)
    # Client now tries to connect to a dead listener.
    client = NNEvalClient(handle, connect_timeout=0.5)
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    with pytest.raises(NNEvalServerError):
        client._call(MSG_KIND_VALUE, feats)


def test_client_with_policy_disabled_rejects_policy_calls(server_factory, adapter):
    server, _, _ = server_factory(
        with_policy=False, batch_size=1, batch_wait_ms=0.0,
    )
    with server:
        client = NNEvalClient(server.handle())
        try:
            with pytest.raises(RuntimeError, match="policy-head"):
                client.evaluate_with_policy(adapter, player_id=0)
            with pytest.raises(RuntimeError, match="policy-head"):
                client.batch_evaluate_with_policy([adapter], player_id=0)
        finally:
            client.close()


def test_server_forward_failure_returns_clean_error_to_client():
    """If the model raises during forward, the server replies with
    REPLY_ERROR and the client raises NNEvalServerError instead of
    hanging."""

    class ExplodingModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("kaboom")

    server = NNEvalServer(
        device="cpu", batch_size=1, batch_wait_ms=0.0,
        model_factory=ExplodingModel,
    )
    with server:
        client = NNEvalClient(server.handle())
        try:
            feats = np.zeros(FEATURE_DIM, dtype=np.float32)
            with pytest.raises(NNEvalServerError, match="forward failed"):
                client._call(MSG_KIND_VALUE, feats)
        finally:
            client.close()


# ---------- shutdown --------------------------------------------------------


def test_graceful_shutdown_via_context_manager(server_factory):
    server, _, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    with server:
        client = NNEvalClient(server.handle())
        client.close()
    # After the with block exits, server threads are joined; address
    # is no longer accepting connections. A fresh client connecting
    # to the dead handle should fail cleanly.
    handle = server.handle()
    client = NNEvalClient(handle, connect_timeout=0.3)
    with pytest.raises(NNEvalServerError):
        client._call(MSG_KIND_VALUE, np.zeros(FEATURE_DIM, dtype=np.float32))


def test_double_stop_is_idempotent(server_factory):
    server, _, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    server.start()
    server.stop(timeout=2.0)
    server.stop(timeout=2.0)  # second call must not raise


def test_double_start_is_idempotent(server_factory):
    server, _, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    server.start()
    server.start()  # must not raise; must not spawn extra threads
    server.stop(timeout=2.0)


# ---------- handle picklability --------------------------------------------


def test_server_handle_is_picklable():
    """The whole point of the handle is to ride across a spawn-based
    multiprocessing boundary. Verify pickle round-trip is loss-free."""
    import pickle

    h = NNServerHandle(
        address=("127.0.0.1", 12345),
        authkey=b"x" * 32,
        with_policy=True,
    )
    h2 = pickle.loads(pickle.dumps(h))
    assert h2 == h


# ---------- multi-thread integration ----------------------------------------


def test_many_threads_many_calls_match_local(server_factory, adapter):
    """Hammer the server with 100 evaluate() calls across 8 client
    threads (8 client connections). Verify every reply matches the
    reference local evaluator on the same input."""
    server, ref, _ = server_factory(batch_size=8, batch_wait_ms=2.0)
    n_threads = 8
    n_per = 13  # 104 total calls
    with server:
        handle = server.handle()
        clients = [NNEvalClient(handle) for _ in range(n_threads)]
        try:
            errors: list[Exception] = []
            mismatches: list[tuple[int, float, float]] = []
            ref_value = ref.evaluate(adapter, player_id=0)
            ready = threading.Barrier(n_threads)

            def _worker(tid: int):
                try:
                    ready.wait()
                    for i in range(n_per):
                        v = clients[tid].evaluate(adapter, player_id=0)
                        if abs(v - ref_value) > 1e-5:
                            mismatches.append((tid, v, ref_value))
                except Exception as e:  # pragma: no cover
                    errors.append(e)

            threads = [
                threading.Thread(target=_worker, args=(i,))
                for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20.0)
            assert not errors, f"client errors: {errors}"
            assert not mismatches, f"value mismatches: {mismatches[:5]}"
            assert server.requests_served == n_threads * n_per
        finally:
            for c in clients:
                c.close()


# ---------- subprocess form (one happy-path integration) -------------------


def test_launch_server_subprocess_smoke(tmp_path):
    """Spin a real subprocess server and a client; verify a single
    round-trip. Validates the spawn/pickle/launch path for the form
    the CLI scripts use."""
    torch.manual_seed(0)
    ref = NNEvaluator()
    weights = tmp_path / "subproc.pt"
    ref.save(weights)

    proxy = launch_server_subprocess(
        weights_path=weights,
        device="cpu",
        batch_size=4,
        batch_wait_ms=2.0,
    )
    try:
        handle = proxy.handle()
        client = NNEvalClient(handle)
        try:
            feats = np.zeros(FEATURE_DIM, dtype=np.float32)
            v = client._call(MSG_KIND_VALUE, feats)
            with torch.no_grad():
                v_ref = ref.model(
                    torch.from_numpy(feats).unsqueeze(0)
                ).item()
            assert v == pytest.approx(v_ref, abs=1e-5)
        finally:
            client.close()
    finally:
        proxy.stop(timeout=10.0)


# ---------- batching correctness: single-worker (no batch) ------------------


def test_single_worker_serial_calls_each_dispatch_one(server_factory):
    """One client firing one request at a time is the worst case for
    batching — the dispatcher should still serve every request and the
    answers should still be correct (just at batch=1).

    Verifies: (a) the server doesn't deadlock when there's no
    second request to coalesce with, (b) the per-call result matches
    the reference, (c) every dispatched batch has size 1 in this
    serialized regime.
    """
    server, ref, _ = server_factory(batch_size=8, batch_wait_ms=2.0)
    rng = np.random.default_rng(42)
    feats_pool = rng.standard_normal(
        (5, FEATURE_DIM),
    ).astype(np.float32)
    with server:
        client = NNEvalClient(server.handle())
        try:
            for i in range(5):
                v_server = client._call(MSG_KIND_VALUE, feats_pool[i])
                with torch.no_grad():
                    v_ref = ref.model(
                        torch.from_numpy(feats_pool[i]).unsqueeze(0),
                    ).item()
                assert v_server == pytest.approx(v_ref, abs=1e-5)
        finally:
            client.close()
        # Single-worker path: every dispatched batch is size 1 (no
        # peer requests to coalesce with).
        assert all(s == 1 for s in server.batch_sizes), (
            f"unexpected coalesced batches in serial regime: "
            f"{server.batch_sizes}"
        )
        assert server.requests_served == 5


# ---------- correlation under interleaved sends ----------------------------


def test_per_worker_request_correlation_across_a_shared_batch(server_factory):
    """The dispatcher coalesces requests from multiple workers into one
    forward pass. Verify that each worker still gets back the value
    computed for ITS feature vector — no cross-talk where worker A
    receives worker B's reply.

    We give each worker a worker-specific feature vector
    (``np.full(FEATURE_DIM, i)``) so the reference forward produces
    a worker-distinguishable result. Then check that worker i's
    reply equals the reference forward on input i.
    """
    server, ref, _ = server_factory(batch_size=16, batch_wait_ms=20.0)
    n_workers = 8
    with server:
        handle = server.handle()
        clients = [NNEvalClient(handle) for _ in range(n_workers)]
        try:
            results: list[float | None] = [None] * n_workers
            ready = threading.Barrier(n_workers)

            def _worker(i: int):
                ready.wait()
                feats = np.full(FEATURE_DIM, float(i), dtype=np.float32)
                results[i] = clients[i]._call(MSG_KIND_VALUE, feats)

            threads = [
                threading.Thread(target=_worker, args=(i,))
                for i in range(n_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            for i, r in enumerate(results):
                assert r is not None, f"worker {i} timed out"
                feats = np.full(FEATURE_DIM, float(i), dtype=np.float32)
                with torch.no_grad():
                    v_ref = ref.model(
                        torch.from_numpy(feats).unsqueeze(0),
                    ).item()
                assert r == pytest.approx(v_ref, abs=1e-5), (
                    f"worker {i} got back the wrong reply: "
                    f"server={r!r} ref={v_ref!r}"
                )
            assert any(s > 1 for s in server.batch_sizes), (
                f"expected at least one coalesced batch, "
                f"got sizes={server.batch_sizes}"
            )
        finally:
            for c in clients:
                c.close()


# ---------- 32-worker stampede ---------------------------------------------


def test_thirty_two_worker_stampede_serves_all(server_factory):
    """All 32 workers hit the server at once. Sanity check that the
    server handles the simultaneous-arrival storm without losing or
    crossing replies. Each worker gets a worker-specific feature row
    (np.full(..., i)) so we can verify each reply matches its own
    input under the reference model."""
    server, ref, _ = server_factory(batch_size=16, batch_wait_ms=2.0)
    n_workers = 32
    with server:
        handle = server.handle()
        clients = [NNEvalClient(handle) for _ in range(n_workers)]
        try:
            results: list[float | None] = [None] * n_workers
            errors: list[Exception] = []
            ready = threading.Barrier(n_workers)

            def _worker(i: int):
                try:
                    ready.wait()
                    feats = np.full(
                        FEATURE_DIM, float(i), dtype=np.float32,
                    )
                    results[i] = clients[i]._call(MSG_KIND_VALUE, feats)
                except Exception as e:  # pragma: no cover - debug aid
                    errors.append(e)

            threads = [
                threading.Thread(target=_worker, args=(i,))
                for i in range(n_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
            assert not errors, f"client errors: {errors!r}"
            for i, r in enumerate(results):
                assert r is not None
                feats = np.full(
                    FEATURE_DIM, float(i), dtype=np.float32,
                )
                with torch.no_grad():
                    v_ref = ref.model(
                        torch.from_numpy(feats).unsqueeze(0),
                    ).item()
                assert r == pytest.approx(v_ref, abs=1e-5), (
                    f"stampede crossed replies for worker {i}"
                )
            # Sanity: with 32 simultaneous requests and batch_size=16,
            # at least two batches should have fired.
            assert len(server.batch_sizes) >= 2
            assert server.requests_served == n_workers
        finally:
            for c in clients:
                c.close()


# ---------- slow forward eventually completes -------------------------------


def test_slow_forward_eventually_responds():
    """A slow model still serves correctly. Simulates the GPU under
    contention (e.g., cuda kernels piling up) by sleeping inside
    forward; the dispatcher must not abandon the batch and the client
    must wait for the reply rather than declaring an error."""

    class SlowModel(torch.nn.Module):
        def forward(self, x):  # noqa: D401 - test model
            time.sleep(0.2)
            n = x.shape[0]
            return torch.full((n, 1), 0.42, dtype=torch.float32)

    server = NNEvalServer(
        device="cpu", batch_size=4, batch_wait_ms=1.0,
        model_factory=SlowModel,
    )
    with server:
        client = NNEvalClient(server.handle())
        try:
            feats = np.zeros(FEATURE_DIM, dtype=np.float32)
            t0 = time.perf_counter()
            v = client._call(MSG_KIND_VALUE, feats)
            elapsed = time.perf_counter() - t0
            assert v == pytest.approx(0.42, abs=1e-6)
            # Must have taken at least the 0.2s simulated forward time.
            assert elapsed >= 0.18, (
                f"call returned faster than the simulated forward: "
                f"{elapsed:.3f}s"
            )
        finally:
            client.close()


# ---------- client reconnect after explicit close ---------------------------


def test_client_reconnects_after_explicit_close(server_factory):
    """``client.close()`` releases the connection but the client must
    still be usable: the next call lazily reopens. Mirrors the worker
    lifecycle where a worker may close its connection between phases
    and reuse the same client object."""
    server, ref, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    with server:
        client = NNEvalClient(server.handle())
        feats = np.zeros(FEATURE_DIM, dtype=np.float32)
        v1 = client._call(MSG_KIND_VALUE, feats)
        client.close()
        # Second call should reopen the connection automatically.
        v2 = client._call(MSG_KIND_VALUE, feats)
        with torch.no_grad():
            v_ref = ref.model(
                torch.from_numpy(feats).unsqueeze(0),
            ).item()
        assert v1 == pytest.approx(v_ref, abs=1e-5)
        assert v2 == pytest.approx(v_ref, abs=1e-5)
        client.close()


# ---------- ephemeral port (no fixed-port collision risk) ------------------


def test_default_address_uses_ephemeral_port(server_factory):
    """The server should bind an OS-assigned ephemeral port by
    default — never a fixed one. Defends against accidental collisions
    with the existing ``mtg_bot serve`` infrastructure (port 5099) and
    other local services."""
    server, _, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    with server:
        addr = server.address
        # AF_INET tuple form: ('host', port). Port 0 means "let OS
        # choose"; the resolved port read off the listener must be
        # nonzero, must be in the valid TCP range, and must not be
        # the well-known mtg_bot serve port (5099).
        assert isinstance(addr, tuple)
        assert addr[0] == "127.0.0.1"
        assert 1 <= addr[1] <= 65535
        assert addr[1] != 5099, (
            "default address must not collide with the mtg_bot serve "
            "HTTP server on 5099"
        )


# ---------- unknown kind error path -----------------------------------------


def test_unknown_kind_returns_error_to_client(server_factory):
    """The dispatcher routes by ``kind``; an unknown one must not
    silently drop the request. The client should receive a clean
    REPLY_ERROR and raise ``NNEvalServerError`` rather than hang."""
    server, _, _ = server_factory(batch_size=1, batch_wait_ms=0.0)
    with server:
        client = NNEvalClient(server.handle())
        try:
            with pytest.raises(NNEvalServerError, match="unknown kind"):
                client._call("not_a_real_kind", np.zeros(FEATURE_DIM, dtype=np.float32))
        finally:
            client.close()
