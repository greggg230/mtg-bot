"""Centralized GPU inference server for NN evaluation.

The classic AlphaZero-at-scale shape: one server process loads the
network onto a GPU once, all worker processes ship pre-extracted
feature vectors (and, for the policy path, action masks) over IPC,
the server batches across workers and time, runs a single forward
pass per dispatch, and returns results to the originating worker.

Why: today's eval/data-gen worker pools spawn 24+ worker processes,
each loading the NN onto its own CPU process and running per-leaf
forwards locally. With 30 workers and 32 logical cores the box
thrashes (mitigated for now via ``torch.set_num_threads(1)``).
Per-leaf NN inference in PUCT mode is the dominant cost; workers
can't batch across each other so the RTX 5090 sits at ~10%.
Centralizing inference behind one process lets the server batch
across all workers — that's where the GPU win is.

Module layout:

- ``NNEvalServer``: spawns a daemon thread inside a server process
  that owns the model. Listens on a ``multiprocessing.connection``
  Listener for client connections. Each connection runs in its own
  reader thread; reader threads enqueue requests onto a shared
  ``queue.Queue``; a single dispatcher thread drains the queue,
  batches up to ``batch_size`` items within ``batch_wait_ms``, runs
  one forward pass, and writes replies back to the originating
  connection. Thread-per-connection is fine because each worker
  has a single connection; we don't expect thousands of clients.

- ``NNEvalClient``: drop-in for ``NNEvaluator``. Same public API
  (``evaluate``, ``batch_evaluate``, ``evaluate_with_policy``,
  ``batch_evaluate_with_policy``). On each call extracts features
  on the client side (and the action mask for the policy path), so
  the server is engine-agnostic — only feature vectors and masks
  cross the IPC boundary, never engine ``GameState`` objects.

- ``NNServerHandle``: a small picklable struct describing how to
  reach the server (address + authkey + with_policy flag). Workers
  receive this in their config, instantiate ``NNEvalClient`` from
  it on demand, and use the client like an ``Evaluator``.

Transport: ``multiprocessing.connection.Listener`` / ``Client``.
Reasons over ``multiprocessing.Queue``:
- Per-connection back-pressure: a slow worker doesn't slow the
  others. Each connection has its own send/recv buffer.
- Easier server start/stop: the listener accepts new clients without
  the server needing to know in advance how many workers will exist.
- Authkey-based authentication, picklable address, works on Windows
  (AF_INET on localhost or AF_PIPE on Windows-named pipes).

Default address is ``('127.0.0.1', 0)`` — IPv4 on an OS-picked
ephemeral port. The actual port is read off the listener after
bind and stored on the handle so it can be passed to workers.

Graceful shutdown: clients send a ``MSG_KIND_SHUTDOWN`` request,
or the server's ``stop()`` method is called externally. The
dispatcher drains in-flight batches before exiting; reader threads
die when the connection closes. SIGTERM handling on Windows is
known-flaky, so the public stop() is the supported path; it's
called automatically on context-manager exit.

Error path: if the server-side forward pass raises, the failing
batch's replies are sent back as ``MSG_KIND_ERROR`` with a short
diagnostic. The client raises ``NNEvalServerError`` so the worker
sees a clean failure rather than a hang.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mtg_bot.features import FEATURE_DIM, extract_features
from mtg_bot.search.evaluator import Evaluator
from mtg_bot.search.nn_evaluator import (
    HIDDEN_SIZES,
    _ValueMLP,
    _ValuePolicyMLP,
    _action_encoding,
)
from mtg_bot.search.protocol import GameState, PlayerID

# ---- wire format -----------------------------------------------------------

# Request/response messages are simple tuples on the wire. Tuples
# pickle fast and avoid the overhead of constructing a dataclass per
# request — the request rate is high (potentially thousands per
# second across 30 workers).
#
# Request:   (seq: int, kind: str, payload)
# Response:  (seq: int, kind: str, result)
#
# kind ∈ {"value", "value_policy", "value_batch", "value_policy_batch",
#         "shutdown"}.
# Reply kind ∈ {"ok", "error"}.

MSG_KIND_VALUE = "value"
MSG_KIND_VALUE_POLICY = "value_policy"
MSG_KIND_VALUE_BATCH = "value_batch"
MSG_KIND_VALUE_POLICY_BATCH = "value_policy_batch"
MSG_KIND_SHUTDOWN = "shutdown"

REPLY_OK = "ok"
REPLY_ERROR = "error"


class NNEvalServerError(RuntimeError):
    """Raised on the client side when the server reports an error
    or the connection is dropped mid-request. Wraps a short
    diagnostic string so workers can log it and decide whether to
    fall back to a local evaluator (caller's choice — this module
    doesn't auto-fallback)."""


# ---- handle dataclass ------------------------------------------------------


@dataclass(frozen=True)
class NNServerHandle:
    """Picklable connection details.

    Carried inside :class:`NNServerPimcConfig` so workers can rebuild
    a client. Frozen + only-builtin-types means it pickles cleanly
    across spawn boundaries on Windows.

    Attributes:
        address: ``(host, port)`` for AF_INET, or a string path for
            AF_UNIX / AF_PIPE. The form chosen by the server at start
            time. Round-trips through pickle.
        authkey: Bytes used by ``multiprocessing.connection`` to
            authenticate clients. Generated randomly at server start;
            uniform across all workers. NOT a security boundary
            (anyone with read access to a worker's pickled config
            sees it) — purely a sanity check that the client is from
            the same launcher run.
        with_policy: Whether the server's loaded model has a policy
            head. Stored on the handle so the client can refuse
            ``evaluate_with_policy`` calls early instead of relying
            on the server to surface the error mid-request.
    """

    address: Any
    authkey: bytes
    with_policy: bool


# ---- server implementation -------------------------------------------------


@dataclass
class _PendingRequest:
    """In-flight item sitting in the dispatcher's queue.

    The dispatcher drains these, batches them by kind, runs one
    forward, then writes replies via the per-connection lock.
    """

    seq: int
    kind: str
    payload: Any
    conn: Connection
    conn_lock: threading.Lock


class NNEvalServer:
    """In-process server that owns the model and a listener.

    Lifecycle:

    - ``__init__`` loads the model and binds the listener but does
      NOT start serving yet.
    - ``start()`` spawns the accept thread and the dispatcher thread.
    - ``stop()`` (or ``__exit__``) signals shutdown, joins threads,
      closes the listener.
    - ``handle()`` returns the picklable :class:`NNServerHandle`
      that workers consume to build clients.

    The class is intended to live inside a launcher process (the CLI
    script) and to run alongside a multiprocessing worker pool. Inside
    the launcher, multiple threads are fine — Python's GIL doesn't
    matter here because the bottleneck is GPU forward + IPC, both
    of which release it.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        with_policy: bool = False,
        batch_size: int = 32,
        batch_wait_ms: float = 5.0,
        address: Any = ("127.0.0.1", 0),
        authkey: Optional[bytes] = None,
        model_factory: Optional[Any] = None,
        hidden_sizes: Optional[tuple[int, ...]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._device = torch.device(device)
        self._with_policy = with_policy
        self._batch_size = max(1, int(batch_size))
        self._batch_wait_s = max(0.0, float(batch_wait_ms)) / 1000.0
        self._logger = logger or logging.getLogger(
            "mtg_bot.search.nn_eval_server"
        )

        # Build the model. ``model_factory`` is an injection point
        # for tests that want a deterministic / instrumented network
        # without touching the production load path. ``hidden_sizes``
        # threads through to the underlying MLP — the server must
        # match the trunk shape the saved weights were trained at,
        # mirroring the same knob on ``NNEvaluator`` and
        # ``NNPimcConfig``.
        mlp_kwargs: dict[str, Any] = {}
        if hidden_sizes is not None:
            mlp_kwargs["hidden_sizes"] = tuple(hidden_sizes)
        if model_factory is not None:
            self._model = model_factory()
        elif with_policy:
            self._model = _ValuePolicyMLP(**mlp_kwargs)
        else:
            self._model = _ValueMLP(**mlp_kwargs)
        if weights_path is not None and model_factory is None:
            state_dict = torch.load(
                weights_path,
                map_location=self._device,
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        # Cache action vocab size; resolved lazily so importing this
        # module does not eagerly trigger the action_encoding import.
        self._vocab_size: Optional[int] = None

        # Listener + auth.
        self._authkey = authkey if authkey is not None else secrets.token_bytes(32)
        self._listener = Listener(address, authkey=self._authkey)
        # Read back the bound address (port 0 → OS picks one).
        self._address = self._listener.address

        # Dispatcher state.
        self._pending: "queue.Queue[Optional[_PendingRequest]]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._reader_threads: list[threading.Thread] = []
        self._connections: list[tuple[Connection, threading.Lock]] = []
        self._connections_lock = threading.Lock()

        # Telemetry — useful for the bench. Plain ints / lists; the
        # caller reads them after stop().
        self.batches_dispatched = 0
        self.requests_served = 0
        self.batch_sizes: list[int] = []

    # ---- public lifecycle --------------------------------------------------

    @property
    def address(self) -> Any:
        return self._address

    @property
    def authkey(self) -> bytes:
        return self._authkey

    def handle(self) -> NNServerHandle:
        """Return picklable connection details for clients."""
        return NNServerHandle(
            address=self._address,
            authkey=self._authkey,
            with_policy=self._with_policy,
        )

    def start(self) -> None:
        """Spawn accept + dispatcher threads. Idempotent — calling
        twice is a no-op."""
        if self._accept_thread is not None:
            return
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="nn-eval-server-dispatch",
            daemon=True,
        )
        self._dispatch_thread.start()
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="nn-eval-server-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown; join threads; close the listener.

        The accept thread is woken by closing the listener (it's
        blocked in ``accept()`` otherwise — there is no sane way
        to interrupt it on Windows other than closing the socket).
        Reader threads die when their connection closes, which
        happens either because the client disconnected or because
        we close their connection here. Dispatcher exits when it
        sees the sentinel.
        """
        if self._stop_evt.is_set():
            return
        self._stop_evt.set()

        # Close listener first so accept() returns / raises.
        try:
            self._listener.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            self._logger.debug("listener.close raised", exc_info=True)

        # Close all live connections so reader threads exit.
        with self._connections_lock:
            for conn, _lock in self._connections:
                try:
                    conn.close()
                except Exception:  # pragma: no cover
                    pass

        # Sentinel: dispatcher exits when it sees None.
        self._pending.put(None)

        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=timeout)
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=timeout)
        for t in self._reader_threads:
            t.join(timeout=timeout)

    def __enter__(self) -> "NNEvalServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ---- accept / read loops ----------------------------------------------

    def _accept_loop(self) -> None:
        """Accept new client connections and spawn a reader per
        connection.

        Uses the listener's blocking ``accept()``. When ``stop()``
        closes the listener, ``accept()`` raises an OSError /
        ``OSError`` / ``EOFError`` depending on platform — we
        catch broadly and exit.
        """
        while not self._stop_evt.is_set():
            try:
                conn = self._listener.accept()
            except (OSError, EOFError):
                # Listener closed or shutdown raced — clean exit.
                return
            except Exception:  # pragma: no cover - defensive
                self._logger.exception("accept() failed")
                return
            conn_lock = threading.Lock()
            with self._connections_lock:
                self._connections.append((conn, conn_lock))
            t = threading.Thread(
                target=self._reader_loop,
                args=(conn, conn_lock),
                name=f"nn-eval-server-reader-{len(self._reader_threads)}",
                daemon=True,
            )
            self._reader_threads.append(t)
            t.start()

    def _reader_loop(
        self, conn: Connection, conn_lock: threading.Lock,
    ) -> None:
        """Drain requests off ``conn`` and enqueue them for the
        dispatcher.

        Runs until the connection is closed or a shutdown message is
        received from this client. Note: shutdown from a client only
        closes the client's own connection; it does NOT stop the
        whole server. The launcher uses ``stop()`` directly for
        global shutdown.
        """
        try:
            while not self._stop_evt.is_set():
                try:
                    msg = conn.recv()
                except (EOFError, OSError, ConnectionError, TypeError):
                    # TypeError shows up on Windows when the handle is
                    # closed mid-recv (the C-level read returns None,
                    # which the multiprocessing layer can't decode).
                    return
                if not isinstance(msg, tuple) or len(msg) != 3:
                    self._logger.warning(
                        "bad message shape from client: %r", msg,
                    )
                    continue
                seq, kind, payload = msg
                if kind == MSG_KIND_SHUTDOWN:
                    # Client-initiated graceful close: ack and return.
                    self._send(conn, conn_lock, (seq, REPLY_OK, None))
                    return
                self._pending.put(
                    _PendingRequest(
                        seq=seq, kind=kind, payload=payload,
                        conn=conn, conn_lock=conn_lock,
                    ),
                )
        finally:
            try:
                conn.close()
            except Exception:  # pragma: no cover
                pass

    def _send(
        self,
        conn: Connection,
        conn_lock: threading.Lock,
        msg: tuple,
    ) -> None:
        """Thread-safe send on a single connection. Connections are
        not safe for concurrent use; the per-connection lock
        serializes writes from the dispatcher (which may interleave
        replies from a single batch back across the same connection
        if multiple of that connection's requests landed in the
        same batch)."""
        with conn_lock:
            try:
                conn.send(msg)
            except (OSError, BrokenPipeError, ConnectionError):
                # Client gone — nothing we can do. The reader thread
                # will exit on next recv().
                self._logger.debug("send() to dead connection: %r", msg[0])

    # ---- dispatcher -------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """Drain the pending queue, build batches, run forwards.

        Batching strategy: pull the first item (blocking). Then drain
        up to ``batch_size - 1`` more items with a per-pull timeout
        of ``batch_wait_s``. The intent is "wait briefly to let the
        queue fill, then dispatch what's there" — so the first wait
        bounds tail latency and subsequent pulls collect anything
        that lands during that window. ``batch_wait_ms = 0`` degrades
        to "dispatch whatever's in the queue right now" (least
        latency, smallest batches).

        Items of different kinds within the same drained group cannot
        share a forward pass (different output shapes), so we
        partition by kind before running compute.
        """
        while not self._stop_evt.is_set():
            try:
                first = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            if first is None:
                # Sentinel from stop().
                return

            batch: list[_PendingRequest] = [first]
            deadline = time.monotonic() + self._batch_wait_s
            while len(batch) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._pending.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    # Re-enqueue so the outer check sees it.
                    self._pending.put(None)
                    break
                batch.append(item)

            self._dispatch_batch(batch)

    def _dispatch_batch(self, batch: list[_PendingRequest]) -> None:
        """Group requests by kind, run a forward per group, send
        replies. A "batched" group of single-state requests (kind ==
        VALUE / VALUE_POLICY) becomes a stacked tensor with one row
        per request; a "batched" group of batch requests (kind ==
        VALUE_BATCH / VALUE_POLICY_BATCH) gets concatenated and then
        split back out.
        """
        # Group by kind.
        groups: dict[str, list[_PendingRequest]] = {}
        for req in batch:
            groups.setdefault(req.kind, []).append(req)

        # Telemetry update BEFORE the per-request sends so a client
        # that's racing the dispatcher (e.g. round-trips fast and
        # stops the server immediately after the last reply) always
        # sees consistent counters by the time its reply arrives.
        self.batches_dispatched += 1
        self.requests_served += len(batch)
        self.batch_sizes.append(len(batch))

        for kind, reqs in groups.items():
            if kind == MSG_KIND_VALUE:
                self._run_value(reqs)
            elif kind == MSG_KIND_VALUE_POLICY:
                self._run_value_policy(reqs)
            elif kind == MSG_KIND_VALUE_BATCH:
                self._run_value_batch(reqs)
            elif kind == MSG_KIND_VALUE_POLICY_BATCH:
                self._run_value_policy_batch(reqs)
            else:
                for r in reqs:
                    self._send(
                        r.conn, r.conn_lock,
                        (r.seq, REPLY_ERROR,
                         f"unknown kind: {kind!r}"),
                    )

    # ---- per-kind handlers ------------------------------------------------

    def _vocab(self) -> int:
        if self._vocab_size is None:
            self._vocab_size = int(_action_encoding().ACTION_VOCAB_SIZE)
        return self._vocab_size

    def _run_value(self, reqs: list[_PendingRequest]) -> None:
        """VALUE: payload is a single feature vector (np.float32[51]).
        Stack and run a single forward pass over the group.
        """
        try:
            feats = np.stack(
                [r.payload for r in reqs], axis=0,
            ).astype(np.float32, copy=False)
            with torch.no_grad():
                x = torch.from_numpy(feats).to(self._device)
                out = self._model(x)
                value_t = out[0] if isinstance(out, tuple) else out
                values = value_t.squeeze(-1).cpu().tolist()
        except Exception as e:  # pragma: no cover - best-effort
            for r in reqs:
                self._send(
                    r.conn, r.conn_lock,
                    (r.seq, REPLY_ERROR, f"forward failed: {e!r}"),
                )
            return
        for r, v in zip(reqs, values):
            self._send(r.conn, r.conn_lock, (r.seq, REPLY_OK, float(v)))

    def _run_value_policy(self, reqs: list[_PendingRequest]) -> None:
        """VALUE_POLICY: payload is (features, mask). Each request
        contributes one row to the stacked forward; the mask is
        applied per-row on the host because the masks differ.
        """
        try:
            feats = np.stack(
                [r.payload[0] for r in reqs], axis=0,
            ).astype(np.float32, copy=False)
            masks = [r.payload[1] for r in reqs]
            with torch.no_grad():
                x = torch.from_numpy(feats).to(self._device)
                value_t, logits_t = self._model(x)
                values = value_t.squeeze(-1).cpu().tolist()
                logits_np = logits_t.cpu().numpy()
        except Exception as e:  # pragma: no cover
            for r in reqs:
                self._send(
                    r.conn, r.conn_lock,
                    (r.seq, REPLY_ERROR, f"forward failed: {e!r}"),
                )
            return
        vocab = self._vocab()
        for k, r in enumerate(reqs):
            mask = masks[k]
            if not mask.any():
                policy = np.zeros(vocab, dtype=np.float32)
            else:
                row = logits_np[k].astype(np.float32, copy=True)
                row[~mask] = -np.inf
                row -= row.max()
                exp = np.exp(row)
                policy = (exp / exp.sum()).astype(
                    np.float32, copy=False,
                )
            self._send(
                r.conn, r.conn_lock,
                (r.seq, REPLY_OK, (float(values[k]), policy)),
            )

    def _run_value_batch(self, reqs: list[_PendingRequest]) -> None:
        """VALUE_BATCH: payload is a list of feature vectors. Each
        request contributes ``len(payload)`` rows; replies are lists
        the same length as the request's input.
        """
        # Concat all rows into one mega-batch so one forward serves
        # the whole group.
        offsets: list[int] = [0]
        rows: list[np.ndarray] = []
        for r in reqs:
            for f in r.payload:
                rows.append(f)
            offsets.append(offsets[-1] + len(r.payload))
        if not rows:
            for r in reqs:
                self._send(r.conn, r.conn_lock, (r.seq, REPLY_OK, []))
            return
        try:
            feats = np.stack(rows, axis=0).astype(np.float32, copy=False)
            with torch.no_grad():
                x = torch.from_numpy(feats).to(self._device)
                out = self._model(x)
                value_t = out[0] if isinstance(out, tuple) else out
                values = value_t.squeeze(-1).cpu().tolist()
        except Exception as e:  # pragma: no cover
            for r in reqs:
                self._send(
                    r.conn, r.conn_lock,
                    (r.seq, REPLY_ERROR, f"forward failed: {e!r}"),
                )
            return
        for k, r in enumerate(reqs):
            start, end = offsets[k], offsets[k + 1]
            self._send(
                r.conn, r.conn_lock,
                (r.seq, REPLY_OK, [float(v) for v in values[start:end]]),
            )

    def _run_value_policy_batch(self, reqs: list[_PendingRequest]) -> None:
        """VALUE_POLICY_BATCH: payload is (features_list, masks_list).
        Same pattern as VALUE_BATCH but with the policy head and
        per-row masking on the host."""
        offsets: list[int] = [0]
        rows: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        for r in reqs:
            feats_list, masks_list = r.payload
            for f, m in zip(feats_list, masks_list):
                rows.append(f)
                all_masks.append(m)
            offsets.append(offsets[-1] + len(feats_list))
        if not rows:
            for r in reqs:
                self._send(r.conn, r.conn_lock, (r.seq, REPLY_OK, []))
            return
        try:
            feats = np.stack(rows, axis=0).astype(np.float32, copy=False)
            with torch.no_grad():
                x = torch.from_numpy(feats).to(self._device)
                value_t, logits_t = self._model(x)
                values = value_t.squeeze(-1).cpu().tolist()
                logits_np = logits_t.cpu().numpy()
        except Exception as e:  # pragma: no cover
            for r in reqs:
                self._send(
                    r.conn, r.conn_lock,
                    (r.seq, REPLY_ERROR, f"forward failed: {e!r}"),
                )
            return
        vocab = self._vocab()
        for k, r in enumerate(reqs):
            start, end = offsets[k], offsets[k + 1]
            out_list: list[tuple[float, np.ndarray]] = []
            for j in range(start, end):
                mask = all_masks[j]
                if not mask.any():
                    policy = np.zeros(vocab, dtype=np.float32)
                else:
                    row = logits_np[j].astype(np.float32, copy=True)
                    row[~mask] = -np.inf
                    row -= row.max()
                    exp = np.exp(row)
                    policy = (exp / exp.sum()).astype(
                        np.float32, copy=False,
                    )
                out_list.append((float(values[j]), policy))
            self._send(r.conn, r.conn_lock, (r.seq, REPLY_OK, out_list))


# ---- client ---------------------------------------------------------------


class NNEvalClient(Evaluator):
    """Drop-in replacement for ``NNEvaluator`` that ships requests
    over IPC instead of running the model locally.

    Public API matches ``NNEvaluator`` exactly so it can swap in via
    config. State serialization happens on the client side: features
    are extracted via ``mtg_bot.features.extract_features`` and the
    legal action mask via ``mtg_bot.training.action_encoding`` —
    only those numpy arrays cross the IPC boundary, never engine
    state. This keeps the server engine-agnostic and avoids shipping
    full ``GameState`` objects (large, expensive to pickle).

    Connection management:

    - Lazily connects on first use so the client is cheap to
      construct and survives the spawn boundary.
    - One connection per client instance, plus a lock around send/
      recv so multiple threads inside a single worker can share the
      client safely. (Most workers are single-threaded today, but
      MCTS internals could grow threading later — paying the lock
      cost per call is cheaper than serializing setup.)
    - Sequence numbers correlate replies. We block on the next reply
      that matches our seq; out-of-order replies for other in-flight
      requests are stashed and matched up later. With a single
      worker thread per client, out-of-order is rare; the dispatcher
      sends replies in the order the batch was processed.
    """

    def __init__(
        self,
        handle: NNServerHandle,
        *,
        connect_timeout: float = 30.0,
    ) -> None:
        self._handle = handle
        self._with_policy = handle.with_policy
        self._connect_timeout = connect_timeout
        self._conn: Optional[Connection] = None
        self._lock = threading.Lock()
        self._next_seq = 1
        self._stash: dict[int, tuple[str, Any]] = {}

    @property
    def with_policy(self) -> bool:
        return self._with_policy

    # ---- evaluator API ----------------------------------------------------

    def evaluate(self, state: GameState, player_id: PlayerID) -> float:
        # Terminal short-circuit on the client side, exactly like
        # NNEvaluator. Cheap, avoids an IPC round-trip.
        if state.is_terminal():
            winner = state.winner()
            if winner is None:
                return 0.0
            return 1.0 if winner == player_id else -1.0
        return self._evaluate_nonterminal(state, player_id)

    def _evaluate_nonterminal(
        self, state: GameState, player_id: PlayerID,
    ) -> float:
        engine = self._engine_state(state)
        feats = extract_features(engine, player_id)
        result = self._call(MSG_KIND_VALUE, feats)
        return float(result)

    def batch_evaluate(
        self, states: Sequence[GameState], player_id: PlayerID,
    ) -> list[float]:
        if not states:
            return []
        terminal_results: list[tuple[int, float]] = []
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
        for i, s in enumerate(states):
            if s.is_terminal():
                winner = s.winner()
                if winner is None:
                    terminal_results.append((i, 0.0))
                else:
                    terminal_results.append(
                        (i, 1.0 if winner == player_id else -1.0),
                    )
                continue
            engine = self._engine_state(s)
            nonterminal_indices.append(i)
            nonterminal_features.append(
                extract_features(engine, player_id),
            )

        out: list[float] = [0.0] * len(states)
        for i, v in terminal_results:
            out[i] = v
        if nonterminal_features:
            values = self._call(MSG_KIND_VALUE_BATCH, nonterminal_features)
            for idx, v in zip(nonterminal_indices, values):
                out[idx] = float(v)
        return out

    def evaluate_with_policy(
        self, state: GameState, player_id: PlayerID,
    ) -> tuple[float, np.ndarray]:
        if not self._with_policy:
            raise RuntimeError(
                "evaluate_with_policy requires a policy-head server",
            )
        ae = _action_encoding()
        vocab_size = ae.ACTION_VOCAB_SIZE
        if state.is_terminal():
            winner = state.winner()
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if winner == player_id else -1.0
            return value, np.zeros(vocab_size, dtype=np.float32)
        engine = self._engine_state(state)
        feats = extract_features(engine, player_id)
        mask = ae.legal_action_mask(state, player_id)
        result = self._call(MSG_KIND_VALUE_POLICY, (feats, mask))
        value, policy = result
        return float(value), policy

    def batch_evaluate_with_policy(
        self,
        states: Sequence[GameState],
        player_id: PlayerID,
    ) -> list[tuple[float, np.ndarray]]:
        if not self._with_policy:
            raise RuntimeError(
                "batch_evaluate_with_policy requires a policy-head server",
            )
        ae = _action_encoding()
        vocab_size = ae.ACTION_VOCAB_SIZE
        out: list[tuple[float, np.ndarray]] = [
            (0.0, np.zeros(vocab_size, dtype=np.float32))
            for _ in states
        ]
        if not states:
            return out
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
        nonterminal_masks: list[np.ndarray] = []
        for i, s in enumerate(states):
            if s.is_terminal():
                winner = s.winner()
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if winner == player_id else -1.0
                out[i] = (value, np.zeros(vocab_size, dtype=np.float32))
                continue
            engine = self._engine_state(s)
            nonterminal_indices.append(i)
            nonterminal_features.append(
                extract_features(engine, player_id),
            )
            nonterminal_masks.append(ae.legal_action_mask(s, player_id))
        if nonterminal_features:
            res = self._call(
                MSG_KIND_VALUE_POLICY_BATCH,
                (nonterminal_features, nonterminal_masks),
            )
            for idx, (v, p) in zip(nonterminal_indices, res):
                out[idx] = (float(v), p)
        return out

    # ---- IPC plumbing -----------------------------------------------------

    def _ensure_connected(self) -> Connection:
        if self._conn is None:
            deadline = time.monotonic() + self._connect_timeout
            last_err: Optional[Exception] = None
            while time.monotonic() < deadline:
                try:
                    self._conn = Client(
                        self._handle.address,
                        authkey=self._handle.authkey,
                    )
                    return self._conn
                except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
                    last_err = e
                    time.sleep(0.05)
            raise NNEvalServerError(
                f"could not connect to server at {self._handle.address!r}"
                f" within {self._connect_timeout}s: {last_err!r}",
            )
        return self._conn

    def _call(self, kind: str, payload: Any) -> Any:
        with self._lock:
            conn = self._ensure_connected()
            seq = self._next_seq
            self._next_seq += 1
            try:
                conn.send((seq, kind, payload))
            except (OSError, BrokenPipeError, ConnectionError) as e:
                self._conn = None
                raise NNEvalServerError(
                    f"send to server failed: {e!r}",
                ) from e

            # Receive replies until we see ours. Out-of-order replies
            # (rare with one connection / one in-flight call) get
            # stashed for whichever future call wants them.
            if seq in self._stash:
                kind_reply, result = self._stash.pop(seq)
                return _unwrap(kind_reply, result)
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError, ConnectionError) as e:
                    self._conn = None
                    raise NNEvalServerError(
                        f"recv from server failed: {e!r}",
                    ) from e
                if not isinstance(msg, tuple) or len(msg) != 3:
                    raise NNEvalServerError(
                        f"malformed reply from server: {msg!r}",
                    )
                rseq, rkind, rresult = msg
                if rseq == seq:
                    return _unwrap(rkind, rresult)
                # Stash for a peer call. Bounded only by traffic;
                # under the single-thread-per-client expectation the
                # stash is empty.
                self._stash[rseq] = (rkind, rresult)

    def close(self) -> None:
        """Release the underlying connection. Subsequent calls
        reconnect lazily."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send((0, MSG_KIND_SHUTDOWN, None))
                except Exception:  # pragma: no cover
                    pass
                try:
                    self._conn.close()
                except Exception:  # pragma: no cover
                    pass
                self._conn = None

    def __enter__(self) -> "NNEvalClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def _engine_state(state: GameState) -> Any:
        engine = getattr(state, "engine_state", None)
        if engine is not None:
            return engine
        return state


def _unwrap(kind: str, result: Any) -> Any:
    """Translate server reply kind to a Python return / raise."""
    if kind == REPLY_OK:
        return result
    if kind == REPLY_ERROR:
        raise NNEvalServerError(str(result))
    raise NNEvalServerError(f"unknown reply kind: {kind!r}")


# ---- convenience: launch a server in a child process ----------------------


def _server_subprocess_main(
    weights_path: Optional[str],
    device: str,
    with_policy: bool,
    batch_size: int,
    batch_wait_ms: float,
    address: Any,
    authkey: bytes,
    hidden_sizes: Optional[tuple[int, ...]],
    ready_event: Any,
    stop_event: Any,
    handle_box: Any,
) -> None:
    """Entry point for ``launch_server_subprocess``.

    Constructs an :class:`NNEvalServer`, publishes the resolved
    handle (now that the listener is bound, the actual port is
    known), then blocks until the parent sets ``stop_event``.
    """
    server = NNEvalServer(
        weights_path=Path(weights_path) if weights_path else None,
        device=device,
        with_policy=with_policy,
        batch_size=batch_size,
        batch_wait_ms=batch_wait_ms,
        address=address,
        authkey=authkey,
        hidden_sizes=hidden_sizes,
    )
    server.start()
    handle_box["address"] = server.address
    handle_box["authkey"] = server.authkey
    handle_box["with_policy"] = with_policy
    ready_event.set()
    try:
        stop_event.wait()
    finally:
        server.stop()


def launch_server_subprocess(
    weights_path: Optional[Path] = None,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    with_policy: bool = False,
    batch_size: int = 32,
    batch_wait_ms: float = 5.0,
    address: Any = ("127.0.0.1", 0),
    authkey: Optional[bytes] = None,
    hidden_sizes: Optional[tuple[int, ...]] = None,
) -> "_SubprocessServerProxy":
    """Spawn an :class:`NNEvalServer` in a child process.

    Returns a proxy with ``.handle()`` and ``.stop()``. Used by
    ``evaluate_nn.py``-style CLIs that want one server in front of
    a worker pool. The server runs in its own process so the GIL
    inside the launcher (which may be doing IPC + UI bookkeeping)
    doesn't fight with the inference loop.

    On Windows the spawn method requires the entry point to be
    importable; ``_server_subprocess_main`` lives at module scope
    and only takes pickle-friendly arguments for that reason.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    stop = ctx.Event()
    handle_box = ctx.Manager().dict()
    if authkey is None:
        authkey = secrets.token_bytes(32)
    proc = ctx.Process(
        target=_server_subprocess_main,
        args=(
            str(weights_path) if weights_path is not None else None,
            device,
            with_policy,
            batch_size,
            batch_wait_ms,
            address,
            authkey,
            tuple(hidden_sizes) if hidden_sizes is not None else None,
            ready,
            stop,
            handle_box,
        ),
        name="nn-eval-server",
        daemon=False,
    )
    proc.start()
    if not ready.wait(timeout=120.0):
        proc.terminate()
        raise NNEvalServerError("server subprocess did not become ready")
    return _SubprocessServerProxy(
        proc=proc, stop_event=stop, handle_box=handle_box,
    )


@dataclass
class _SubprocessServerProxy:
    """Lightweight handle on a subprocess server. Mirrors the
    surface :class:`NNEvalServer` exposes to the launcher: ``.handle()``
    and ``.stop()``. Internals are private."""

    proc: Any
    stop_event: Any
    handle_box: Any

    def handle(self) -> NNServerHandle:
        return NNServerHandle(
            address=self.handle_box["address"],
            authkey=self.handle_box["authkey"],
            with_policy=self.handle_box["with_policy"],
        )

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        self.proc.join(timeout=timeout)
        if self.proc.is_alive():  # pragma: no cover - defensive
            self.proc.terminate()
            self.proc.join(timeout=timeout)

    def __enter__(self) -> "_SubprocessServerProxy":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


__all__ = [
    "MSG_KIND_SHUTDOWN",
    "MSG_KIND_VALUE",
    "MSG_KIND_VALUE_BATCH",
    "MSG_KIND_VALUE_POLICY",
    "MSG_KIND_VALUE_POLICY_BATCH",
    "NNEvalClient",
    "NNEvalServer",
    "NNEvalServerError",
    "NNServerHandle",
    "REPLY_ERROR",
    "REPLY_OK",
    "launch_server_subprocess",
]
