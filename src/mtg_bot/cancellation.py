"""Cooperative cancellation for abandoned (timed-out) games.

Why this exists
---------------

``mtg_bot.training._timeout.run_with_timeout`` runs one game in a
daemon thread and gives up on it after a wall-clock budget. CPython
can't kill a thread, so before this module existed the abandoned
thread *kept running the game* inside the multiprocessing worker while
the pool handed that same worker the next game. Two full games then
ran concurrently in one process, sharing the interpreter, the numpy /
torch runtimes, the module-level tokenizer caches
(``card_vocab.current_card_vocab``'s per-source cache) and the process-global
search telemetry (``mtg_bot.search.eval_counters``).

That is what the 2026-09-23 production run showed:

- ``ladder_runs/prod_2026-09-23/run.log`` contains two
  ``Fatal Python error: _PyEval_EvalFrameDefault: Executing a cache.``
  faulthandler dumps whose thread lists show **two** live
  ``self_play._play_capped_game`` frames in one worker — the zombie
  game and the next game, running at the same time.
- Immediately after those, the same pool raised
  ``ValueError: could not broadcast input array from shape (59,7)
  into shape (58,7)`` out of
  ``nn_evaluator._collate_structured_token_dicts`` — a *torn*
  tokenizer output dict (``token_type`` and ``token_emb_ids`` have
  the same length by construction in ``tokenize_state``, so a
  mismatch is interpreter-level corruption, not a logic bug).

There's a second, quieter cost: every abandoned thread keeps burning a
core for the rest of the run. Each timeout therefore permanently
removes one core from a pool that is already sized to the core count,
which makes the *next* game slower, which makes the next timeout more
likely. Stopping the game for real breaks that feedback loop.

Design
------

A ``CancelToken`` is bound to the thread that runs one game, via a
``threading.local``. Search loops and game loops call
``raise_if_cancelled()``; when the parent flips the token the next
check raises ``GameCancelled`` and the whole call stack unwinds.

- **Zero cost when unused.** Threads with no bound token (the pool's
  own worker thread, unit tests, the ``per_game_timeout_sec=None``
  path) see one ``threading.local`` attribute read that returns
  ``None``. Measured at ~100 ns against a ~150 us -- 2 ms per-node
  MCTS iteration, i.e. well under 0.1 %.
- **No signature churn.** Strategy factories are
  ``Callable[[int], SearchStrategy]`` and are pickled across the
  spawn boundary; threading a token through them would mean changing
  that contract everywhere. Thread-affinity gives the same result
  with no plumbing.
- **``GameCancelled`` derives from ``BaseException``**, deliberately.
  The worker-crash sentinels in ``self_play`` / ``data_gen`` catch
  ``Exception`` to convert engine bugs into dropped games; a
  cancellation must not be laundered into "crashed". This mirrors
  ``KeyboardInterrupt`` / ``SystemExit``.
- **Tokens, not thread idents.** OS thread ids are recycled, so a
  registry keyed on ident could cancel an innocent later thread.

Search semantics are untouched: a cancelled search raises instead of
returning, so no visit distribution, prior, or budget is ever
observed by a caller in a cancelled state.
"""

from __future__ import annotations

import threading

__all__ = [
    "CancelToken",
    "GameCancelled",
    "bind_token",
    "current_token",
    "is_cancelled",
    "raise_if_cancelled",
    "unbind_token",
]


class GameCancelled(BaseException):
    """Raised inside a game thread whose token has been cancelled.

    Derives from ``BaseException`` so the broad ``except Exception``
    worker-crash handlers in ``training.self_play`` /
    ``training.data_gen`` don't reclassify an intentional stop as an
    engine crash.
    """


class CancelToken:
    """A one-way flag: not cancelled -> cancelled.

    ``cancel()`` is called from the *parent* (timeout) thread;
    ``cancelled`` is read from the *game* thread. A plain attribute
    is enough — CPython attribute load/store on a ``__slots__``
    object is atomic with respect to the GIL, and the flag only ever
    moves in one direction, so a stale read costs at most one more
    search iteration before the stop lands.
    """

    __slots__ = ("_cancelled", "reason")

    def __init__(self, reason: str = "cancelled") -> None:
        self._cancelled = False
        self.reason = reason

    @property
    def cancelled(self) -> bool:
        """True once ``cancel()`` has been called."""
        return self._cancelled

    def cancel(self, reason: str | None = None) -> None:
        """Request that the bound thread stop at its next check."""
        if reason is not None:
            self.reason = reason
        self._cancelled = True

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"CancelToken(cancelled={self._cancelled!r}, "
            f"reason={self.reason!r})"
        )


_LOCAL = threading.local()


def bind_token(token: CancelToken) -> None:
    """Attach ``token`` to the calling thread.

    Called by ``run_with_timeout``'s runner as its first act, so the
    token is live before any game code executes.
    """
    _LOCAL.token = token


def unbind_token() -> None:
    """Detach any token from the calling thread. Idempotent."""
    _LOCAL.token = None


def current_token() -> CancelToken | None:
    """The calling thread's token, or ``None`` when unbound."""
    return getattr(_LOCAL, "token", None)


def is_cancelled() -> bool:
    """True iff the calling thread has a token that was cancelled."""
    token = getattr(_LOCAL, "token", None)
    return token is not None and token.cancelled


def raise_if_cancelled() -> None:
    """Raise ``GameCancelled`` iff this thread's game was abandoned.

    The hot path is a single ``threading.local`` attribute read that
    returns ``None``; call it freely from search / game loops.
    """
    token = getattr(_LOCAL, "token", None)
    if token is not None and token._cancelled:
        raise GameCancelled(token.reason)
