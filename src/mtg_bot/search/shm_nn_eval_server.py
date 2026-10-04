"""Shared-memory NN eval server (Lever 2 v2).

The original :class:`NNEvalServer` in :mod:`mtg_bot.search.nn_eval_server`
uses Python's ``multiprocessing.connection`` for request/response wire
traffic. That protocol pickles every payload and uses one reader
thread per connection, with serialized GIL-held response writes.
The Phase 6 WSL2 migration bench (PR #39) measured ~2 ms p50 RTT
that floor — regardless of TCP vs UDS transport — because the cost
is in the Python protocol layer, not the socket.

This module replaces that wire protocol with shared-memory tensors:

- The parent allocates fixed-size shared tensors for inputs (features,
  optional mask) and outputs (value, optional policy), N rows each
  ("slots").
- Workers acquire a slot index from a ``free_slots`` mp.Queue (which
  doubles as backpressure: in-flight requests cap at N).
- Worker writes the input row in-place, puts the slot index on the
  ``request_queue``, then waits on a per-slot ``response_event``.
- A single dispatcher thread in the server pulls slot indices off
  ``request_queue`` (batched up to ``batch_size`` over a
  ``batch_wait_s`` window), runs ONE forward over the gathered batch,
  writes outputs back to the corresponding output-slot rows, and
  signals each slot's ``response_event``.
- Worker reads the output row, clears the event, returns the slot to
  ``free_slots``.

Substrate measurement (single client, 1000 iterations on WSL2 Ubuntu
24.04 + Python 3.13.13): p50 round-trip 74 us, vs the existing
mp.connection's 2 ms — ~27x lower per-request overhead. With batching
across multiple workers, aggregate throughput should easily exceed
the per-worker local-NN reference of 1,355 RPS (PR #39 bench).

Constraints:

- The substrate works on both Linux fork and Windows spawn. On Linux,
  workers fork-inherit the shared-tensor mmap + mp.Event fds (the
  original design). On Windows there is no ``fork`` start method, but
  ``torch.Tensor.share_memory_()`` allocates a named shared-memory
  region (via Windows file-mapping primitives) and the ``mp.Queue``
  / ``mp.Event`` objects from a ``spawn`` context are picklable, so
  passing the handle through ``ProcessPoolExecutor(initializer=
  set_worker_handle, initargs=(handle,), mp_context=
  mp.get_context("spawn"))`` reconstructs the shared state on each
  worker at startup. The picked start method is the host's natural
  default: fork on Linux, spawn on Windows.
- macOS support is implemented at the substrate level (spawn works
  the same as on Windows) but disabled at the call-site gate in
  ``mtg_bot.training.distributed`` because the SHM server's device
  routing doesn't currently auto-select MPS — running on Mac would
  fall back to CPU and regress the existing per-worker MPS path.
- One server per master process. Multiple instances on the same
  host work but each holds its own slot pool + dispatcher thread.

The API mirrors :class:`NNEvalServer` / :class:`NNEvalClient` so a
PIMC config can swap implementations transparently. Where the two
diverge: this server doesn't take ``address`` (no socket) and the
handle is a Python object holding shared-tensor + event references
that only work between processes of a single host (not pickle-across-
host).
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    from mtg_bot.search.evaluator import Evaluator as _EvaluatorBase
    from mtg_bot.search.types import GameState, PlayerID

# Re-use the model classes from the existing module so we don't
# duplicate the NN definition. Pull in extract_features + Evaluator
# base + action-encoding helpers to mirror NNEvalClient's API exactly.
from mtg_bot.features import FEATURE_DIM, extract_features
from mtg_bot.search import eval_counters
from mtg_bot.search.evaluator import Evaluator
from mtg_bot.search.nn_evaluator import (
    ARCHITECTURE_SHARED,
    ARCHITECTURE_STRUCTURED,
    HIDDEN_SIZES,
    NNEvaluator,
    SUPPORTED_ARCHITECTURES,
    _build_dual_head_model,
    _build_structured_model,
    _collate_structured_token_dicts,
    _meta_field,
    _ValueMLP,
    _ValuePolicyMLP,
    check_flat_metadata_value_options,
    structured_kwargs_from_metadata,
)


# ----------------------------------------------------------------------
# Structured-architecture shared-tensor layout.
#
# When the loaded model is ``StructuredModel`` (architecture='structured'),
# the SHM substrate hosts per-token tensors instead of the flat
# ``(B, FEATURE_DIM)`` MLP input. Each slot's per-request tensors are:
#
#   - ``token_type``     int64 [n_slots, MAX_SEQ_LEN]
#   - ``token_emb_ids``  int64 [n_slots, MAX_SEQ_LEN, N_EMB_FIELDS]
#   - ``token_scalars``  float32 [n_slots, MAX_SEQ_LEN, N_SCALAR_FIELDS]
#   - ``attention_mask`` bool  [n_slots, MAX_SEQ_LEN]
#
# Workers tokenize their state to a variable-length dict, then pad up
# to ``MAX_SEQ_LEN`` (padded positions get ``attention_mask=False``,
# zeroed categorical / scalar fields; the model's ``padding_idx=0`` +
# ``src_key_padding_mask`` handle them cleanly).
#
# The value / policy output slots stay fixed-shape: ``values`` is
# ``float32[n_slots, 1]`` and ``policies`` is
# ``float32[n_slots, ACTION_VOCAB_SIZE]``. The structured trunk always
# emits a policy head (``StructuredModel.forward`` returns both), so
# ``with_policy`` is forced True for structured servers — mirroring
# ``NNEvaluator``'s behaviour.
#
# MAX_SEQ_LEN tuning: real-state seq_len distribution from the
# Slice A+B agent measures median=53, p95=82. Default
# ``MAX_SEQ_LEN=128`` sits at ~1.5x p95 — plenty of headroom for the
# long tail without burning much memory. Four tensors per slot at
# MAX_SEQ_LEN=128 cost ~5.5 KB combined (8 + 56 + 12 + 1 bytes per
# token × 128 tokens), so a 64-slot pool burns ~350 KB total for the
# structured input substrate — trivial compared to the model itself.
#
# A request whose tokenized seq_len exceeds MAX_SEQ_LEN is an
# architectural-bound violation — fail loudly rather than silently
# truncate (truncation would drop tokens at the tail, which carry
# the most recent action history — exactly the signal Step 1 cares
# most about). The override knob is the ``max_seq_len`` server
# constructor kwarg if a config legitimately needs more headroom.
# ----------------------------------------------------------------------

DEFAULT_MAX_SEQ_LEN = 128
"""Default per-slot sequence-length cap for structured SHM servers.

Sized to ~1.5x the p95 of the real-state seq_len distribution
(measured median=53, p95=82 across the Slice A+B tokenizer). Bump
this only when a config-level test demonstrates the current cap
clips production sequences."""


def _action_encoding():
    """Lazy import so the module load doesn't pull in
    mtg_bot.training.action_encoding (a heavy chain via training.config)."""
    from mtg_bot.training import action_encoding as _ae
    return _ae


# Worker-process globals. The "default" handle is the back-compat
# single-server slot used by self-play (``run_shard_pull`` /
# ``mtg_bot bench nn-eval-server --mode shm``). The "by weights_path"
# dict supports the eval case where the master has TWO servers (one
# per weights set) and workers route requests by the cfg's
# ``weights_path``. Both are unset in the master parent and in
# non-SHM workers; :func:`maybe_get_worker_handle` returns ``None``
# in that case, which preserves the legacy ``NNEvaluator`` path.
_WORKER_SHM_DEFAULT_HANDLE: Optional["SharedTensorServerHandle"] = None
_WORKER_SHM_HANDLES_BY_PATH: dict[str, "SharedTensorServerHandle"] = {}


def set_worker_handle(
    handle: Optional["SharedTensorServerHandle"],
    weights_path: Optional[str] = None,
) -> None:
    """Install a fork-inherited SHM server handle on the calling
    worker process. Intended for use as an mp.Pool initializer when
    the master has spun up one or more :class:`SharedTensorNNEvalServer`.

    Two registration shapes:

    - ``weights_path=None`` (default) → install as the "singleton"
      handle, used by callers that don't disambiguate by weights
      path. ``handle=None`` clears it.
    - ``weights_path="...pt"`` → register under the given key so
      ``maybe_get_worker_handle(weights_path)`` returns this handle.
      Used by the eval path where the master holds two servers
      (new + old weights), workers route via the cfg's
      ``weights_path``.
    """
    global _WORKER_SHM_DEFAULT_HANDLE
    if weights_path is None:
        _WORKER_SHM_DEFAULT_HANDLE = handle
    else:
        if handle is None:
            _WORKER_SHM_HANDLES_BY_PATH.pop(weights_path, None)
        else:
            _WORKER_SHM_HANDLES_BY_PATH[weights_path] = handle


def maybe_get_worker_handle(
    weights_path: Optional[str] = None,
) -> Optional["SharedTensorServerHandle"]:
    """Return the worker's SHM handle if one was installed via
    :func:`set_worker_handle`, else ``None``.

    Lookup order:

    1. If ``weights_path`` is provided and matches a registered
       per-path handle, return that. This is the eval path's
       routing key.
    2. Otherwise return the singleton handle (or ``None`` if not
       set). Self-play registers only the singleton; legacy callers
       that don't pass ``weights_path`` continue to work unchanged.
    """
    if weights_path is not None:
        h = _WORKER_SHM_HANDLES_BY_PATH.get(weights_path)
        if h is not None:
            return h
    return _WORKER_SHM_DEFAULT_HANDLE


def clear_worker_handles() -> None:
    """Forget all installed handles. Used by tests to isolate state
    between cases."""
    global _WORKER_SHM_DEFAULT_HANDLE
    _WORKER_SHM_DEFAULT_HANDLE = None
    _WORKER_SHM_HANDLES_BY_PATH.clear()


def preferred_mp_context() -> "mp.context.BaseContext":
    """Return the preferred multiprocessing context for SHM-server
    workers on this host.

    - Linux / macOS: ``fork``. Workers inherit the shared-tensor mmap
      and mp.Event file descriptors directly — the original design.
    - Windows: ``spawn``. ``fork`` is not available on Windows; spawn
      workers pickle-reconstruct the handle from the executor's
      ``initargs``. torch's named shared-memory regions and mp.Queue
      / mp.Event from a spawn context all survive the round-trip.

    The selection is independent of the platform gate in
    :mod:`mtg_bot.training.distributed` — callers choose whether to
    install a SHM server per their own policy; this helper just picks
    the right child-process start method when they do.
    """
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    return mp.get_context("spawn")


@dataclasses.dataclass
class SharedTensorServerHandle:
    """Lightweight handle a worker uses to attach to the server.

    All fields are inter-process safe via the host's mp context:
    fork-inheritable on Linux, picklable on Windows (passed through
    ``ProcessPoolExecutor(initializer=set_worker_handle,
    initargs=(handle,))``). Either way, do not pickle this handle
    across host boundaries — the shared-memory regions are
    process-host local.
    """
    free_slots: "mp.Queue[int]"
    request_queue: "mp.Queue[int]"
    response_events: list[Any]  # list[mp.Event] — typing imprecise
    # Cross-process shutdown signal. The server sets this in ``stop()``
    # so that workers blocked in ``_call_value*`` notice the server is
    # gone and raise instead of hanging on a response event that will
    # never be signaled. Set once, never reset within a server's
    # lifetime; ``start()`` allocates a fresh handle if reused.
    server_stopped: Any  # mp.Event
    # Cross-process lock serializing the multi-slot acquire phase of
    # ``SharedTensorNNEvalClient._submit_batch``. Without this, N
    # concurrent workers each calling ``free_slots.get()`` in a tight
    # loop can each accumulate partial allocations, exhaust the pool
    # before any worker reaches its full count, and deadlock — the
    # multi-worker contention regression test pins this scenario.
    # The lock only gates the *acquire* phase (microseconds in the
    # uncontended case); writes, submits, waits, and reads all
    # proceed in parallel as before, so dispatcher fusion across
    # workers is unaffected.
    acquire_lock: Any  # mp.synchronize.Lock
    # Shared tensors. ``features``, ``values`` always present for
    # MLP-architecture servers; ``masks`` / ``policies`` only when
    # ``with_policy=True``. For structured-architecture servers
    # ``features`` is a zero-element tensor (kept on the handle for
    # field-uniformity; never written or read by structured callers);
    # ``values`` + ``policies`` stay fixed-shape because the model's
    # output side is identical across architectures.
    features: torch.Tensor
    values: torch.Tensor
    masks: Optional[torch.Tensor]
    policies: Optional[torch.Tensor]
    with_policy: bool
    n_slots: int
    feature_dim: int
    action_vocab_size: int
    # Architecture-routing metadata. ``architecture`` carries the
    # ``ARCHITECTURE_*`` string the server was built around so
    # clients pick the right submit path without having to consult
    # the loaded model. ``max_seq_len`` is the per-slot sequence-
    # length cap for the structured pool (unused / 0 for MLP).
    architecture: str = ARCHITECTURE_SHARED
    max_seq_len: int = 0
    # Structured-only per-slot shared tensors. Each is allocated
    # as zero-element when ``architecture != ARCHITECTURE_STRUCTURED``
    # so the dataclass stays a flat record. The model's
    # ``padding_idx=0`` + ``src_key_padding_mask`` handle padded
    # positions cleanly when these are populated.
    token_type: Optional[torch.Tensor] = None
    token_emb_ids: Optional[torch.Tensor] = None
    token_scalars: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    # Tokenizer hyperparameters the client threads into
    # ``state_tokenizer.tokenize_state`` so the emitted ids land in
    # the model's embedding tables instead of overflowing. Set from
    # the server's loaded ``StructuredModel.config`` at construction
    # time. ``n_printings_cap`` mirrors ``StructuredModelConfig
    # .n_printings`` (default 320); ``action_history_n`` mirrors
    # ``StructuredModelConfig.action_history_n`` (default 32).
    # Both default to 0 for non-structured handles so the field is
    # always present.
    structured_n_printings_cap: int = 0
    structured_action_history_n: int = 0
    # Scalar schema (``token_scalars`` row layout) the loaded model's
    # ``scalar_proj`` expects; the client threads it into
    # ``tokenize_state`` so the emitted width matches the slot
    # tensor. 0 for non-structured handles.
    structured_scalar_schema_version: int = 0


class SharedTensorNNEvalServer:
    """A single-server-per-master NN evaluator that batches requests
    from N worker processes via shared memory + mp.Event.

    Lifecycle:

        server = SharedTensorNNEvalServer(weights_path=..., with_policy=True)
        server.start()                 # spawns dispatcher thread
        handle = server.handle()       # pass to workers (fork-inheritable)
        # ... workers do their thing ...
        server.stop()                  # joins dispatcher

    Or as a context manager: ``with server: ...``.

    To host a REGISTERED checkpoint use :meth:`from_metadata`, which
    resolves the trunk dims, value-head input and value calibration
    through the same resolver as ``NNEvaluator.from_metadata`` /
    ``NNPimcConfig.build``.
    """

    @classmethod
    def from_metadata(
        cls,
        weights_path: Path,
        metadata: Any,
        *,
        apply_value_calibration: bool = True,
        **server_kwargs: Any,
    ) -> "SharedTensorNNEvalServer":
        """Host ``weights_path`` with every model knob read from its
        metadata (``ModelMetadata`` or the raw ``metadata.json`` dict)
        via :func:`~mtg_bot.search.nn_evaluator.structured_kwargs_from_metadata`
        (structured) or the MLP fields + the same refusal of value-head
        options an MLP cannot honour. ``server_kwargs`` are the
        server-only knobs (device, batch_size, n_slots, ...)."""
        architecture = _meta_field(
            metadata, "architecture", ARCHITECTURE_SHARED,
        )
        if architecture == ARCHITECTURE_STRUCTURED:
            return cls(
                weights_path=weights_path,
                architecture=architecture,
                card_vocab=(_meta_field(metadata, "card_vocab") or None),
                **structured_kwargs_from_metadata(
                    metadata,
                    apply_value_calibration=apply_value_calibration,
                ),
                **server_kwargs,
            )
        check_flat_metadata_value_options(
            metadata, apply_value_calibration=apply_value_calibration,
        )
        hidden = _meta_field(metadata, "hidden_sizes", ()) or ()
        training = _meta_field(metadata, "training", {}) or {}
        return cls(
            weights_path=weights_path,
            architecture=architecture,
            with_policy=bool(_meta_field(metadata, "with_policy", False)),
            hidden_sizes=tuple(hidden) if hidden else None,
            policy_hidden_sizes=(
                tuple(training.get("policy_hidden_sizes") or ()) or None
            ),
            **server_kwargs,
        )

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        *,
        device: Optional[str] = None,
        with_policy: bool = False,
        batch_size: int = 64,
        batch_wait_ms: float = 0.5,
        n_slots: int = 64,
        hidden_sizes: Optional[tuple[int, ...]] = None,
        architecture: Optional[str] = None,
        policy_hidden_sizes: Optional[tuple[int, ...]] = None,
        model_factory: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        # ----- GPU fast-path knobs (Slice "shm-gpu-opts") -----------
        # ``autocast_dtype`` enables mixed-precision in the dispatcher
        # forward. ``None`` (the back-compat default on non-CUDA hosts)
        # disables autocast; ``torch.bfloat16`` is the recommended
        # production setting on CUDA Blackwell+ (RTX 5090 has tensor
        # cores for both bf16/fp16; bf16's wider dynamic range avoids
        # the residual-stream overflow that fp16 hits on transformer
        # attention). When ``device='cuda'`` and ``autocast_dtype`` is
        # not explicitly set, we auto-enable bf16; pass
        # ``autocast_dtype=False`` to opt out (e.g. for fp32-vs-fp32
        # numerical regression tests).
        autocast_dtype: Any = "auto",
        # ``compile_model`` wraps the loaded model in
        # ``torch.compile(mode='reduce-overhead')`` after load. CUDA
        # graphs internally fuse the dispatch path's many-small-call
        # pattern; first forward pays the compile cost (~seconds),
        # subsequent calls amortize. Defaults to ``True`` on CUDA,
        # ``False`` elsewhere. Pass ``False`` to disable explicitly
        # (used in tests that want raw-eager output for numerical
        # comparison, and on hosts where Triton is unavailable).
        compile_model: Optional[bool] = None,
        # ``warmup`` runs a synthetic forward pass at
        # MAX_SEQ_LEN + batch_size during ``start()`` so CUDA kernels
        # (and torch.compile artifacts) JIT-compile before the first
        # real request. Adds ~hundreds of ms to startup but eliminates
        # the cold-cache spike on the first batch. Defaults to ``True``
        # on CUDA, ``False`` elsewhere.
        warmup: Optional[bool] = None,
        # ``pin_memory`` calls ``.pin_memory()`` on the slot tensors
        # so host->device copies use page-locked memory + can be
        # ``non_blocking=True``. Only meaningful on CUDA; ignored on
        # CPU. Defaults to ``True`` on CUDA, ``False`` elsewhere.
        pin_memory: Optional[bool] = None,
        # ----- structured-architecture knobs ------------------------
        # ``max_seq_len`` caps the per-slot sequence length for the
        # structured input pool. Requests that tokenize to seq_len >
        # max_seq_len are rejected (loud error) rather than silently
        # truncated — truncation would drop the most-recent action
        # tokens, which is the signal Step 1 cares most about.
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        # Optional config overrides for the StructuredModel trunk.
        # ``None`` falls through to StructuredModelConfig's defaults
        # (matching ``NNEvaluator(architecture='structured')``).
        structured_emb_dim: Optional[int] = None,
        structured_n_heads: Optional[int] = None,
        structured_n_layers: Optional[int] = None,
        structured_hidden_dim: Optional[int] = None,
        structured_action_history_n: Optional[int] = None,
        structured_n_printings: Optional[int] = None,
        structured_action_vocab_size: Optional[int] = None,
        structured_enable_hand_belief_head: Optional[bool] = None,
        structured_dropout: Optional[float] = None,
        structured_scalar_schema_version: Optional[int] = None,
        # Value-head routing + post-hoc value calibration; see
        # ``StructuredModelConfig.value_head_input`` /
        # ``.value_temperature``. ``None`` keeps the model defaults
        # (shared trunk, identity calibration). Hosting a calibrated
        # checkpoint here means the batched path returns the SAME
        # value the in-process ``NNEvaluator`` would.
        structured_value_head_input: Optional[str] = None,
        structured_value_temperature: Optional[float] = None,
        structured_value_bias: Optional[float] = None,
        # Stored card vocabulary of a structured ``weights_path`` —
        # same contract as ``NNEvaluator(card_vocab=...)``: ``None``
        # resolves it next to the weights, an unstamped checkpoint is
        # refused, a changed vocabulary is remapped by card name.
        card_vocab: Any = None,
    ) -> None:
        # Deferred CUDA resolution: same pattern as PR #36's fix to
        # NNEvalServer. Don't call torch.cuda.is_available() in any
        # function default or at module import — it primes CUDA on
        # the parent and breaks fork().
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._is_cuda = self._device.type == "cuda"
        # ----- GPU fast-path defaults --------------------------------
        # Resolve the autocast / compile / warmup / pin_memory knobs.
        # Default policy per knob:
        #
        # - ``autocast_dtype="auto"`` → OFF by default. Measurement
        #   on the current tiny structured trunk (emb_dim=128,
        #   2-layer, batch_size=16, seq_len=64) shows bf16 is 3.2x
        #   SLOWER than fp32 because per-op autocast dispatch
        #   overhead dominates the small kernel work. The
        #   infrastructure stays for opt-in via explicit
        #   ``autocast_dtype=torch.bfloat16`` once the trunk grows
        #   large enough to amortize tensor-core throughput. Set to
        #   ``"auto-cuda"`` to force-enable on CUDA.
        # - ``warmup`` → ON by default on CUDA. One-shot startup
        #   cost; eliminates first-batch latency spike.
        # - ``compile_model`` → OFF by default. Triton isn't
        #   packaged for Windows in the current cu128 wheel, and
        #   the structured trunk's dynamic input dict isn't a clean
        #   reduce-overhead target until Slice F's MAX_SEQ_LEN
        #   padding propagates further. Opt in explicitly.
        # - ``pin_memory`` → OFF by default. The added host-host
        #   copy through pinned staging buffers costs more than the
        #   non-blocking H2D speedup at our current per-request size
        #   (~6 KB structured input).
        if autocast_dtype == "auto":
            autocast_dtype = None
        elif autocast_dtype == "auto-cuda":
            autocast_dtype = torch.bfloat16 if self._is_cuda else None
        elif autocast_dtype is False:
            autocast_dtype = None
        self._autocast_dtype = autocast_dtype
        self._autocast_enabled = (
            self._is_cuda and self._autocast_dtype is not None
        )
        if compile_model is None:
            compile_model = False
        self._compile_model_requested = bool(compile_model)
        if warmup is None:
            warmup = self._is_cuda
        self._warmup_enabled = bool(warmup)
        if pin_memory is None:
            pin_memory = False
        self._pin_memory_enabled = bool(pin_memory) and self._is_cuda
        # Resolve the architecture string. ``None`` (the legacy default)
        # routes to the shared-trunk dual-head MLP, matching the
        # pre-Slice-F behaviour. The structured architecture is opted
        # into explicitly via ``architecture='structured'``.
        resolved_arch = (
            architecture if architecture is not None else ARCHITECTURE_SHARED
        )
        if resolved_arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {SUPPORTED_ARCHITECTURES}; "
                f"got {resolved_arch!r}"
            )
        self._architecture = resolved_arch
        self._is_structured = resolved_arch == ARCHITECTURE_STRUCTURED
        # The structured trunk always has a policy head; mirroring
        # ``NNEvaluator``, we silently upgrade ``with_policy=False``
        # to True for structured servers so the dispatch path always
        # writes both outputs.
        if self._is_structured:
            with_policy = True
        self._with_policy = with_policy
        self._batch_size = max(1, int(batch_size))
        self._batch_wait_s = max(0.0, float(batch_wait_ms)) / 1000.0
        self._n_slots = max(1, int(n_slots))
        self._max_seq_len = (
            max(1, int(max_seq_len)) if self._is_structured else 0
        )
        self._logger = logger or logging.getLogger(
            "mtg_bot.search.shm_nn_eval_server"
        )

        # Build the model. Dispatch by ``with_policy`` + ``architecture``
        # the same way NNEvaluator does so the SHM server can host any
        # production-shape weights (1024-trunk dual-head, separate-trunks,
        # value-only, structured-transformer). ``hidden_sizes=None``
        # falls through to the underlying class default (HIDDEN_SIZES)
        # — matches NNEvaluator.
        effective_hidden = (
            tuple(hidden_sizes) if hidden_sizes is not None else HIDDEN_SIZES
        )
        if model_factory is not None:
            self._model = model_factory()
        elif self._is_structured:
            self._model = _build_structured_model(
                emb_dim=structured_emb_dim,
                n_heads=structured_n_heads,
                n_layers=structured_n_layers,
                hidden_dim=structured_hidden_dim,
                action_history_n=structured_action_history_n,
                n_printings=structured_n_printings,
                action_vocab_size=structured_action_vocab_size,
                enable_hand_belief_head=(
                    structured_enable_hand_belief_head
                ),
                dropout=structured_dropout,
                scalar_schema_version=structured_scalar_schema_version,
                value_head_input=structured_value_head_input,
                value_temperature=structured_value_temperature,
                value_bias=structured_value_bias,
            )
        elif with_policy:
            self._model = _build_dual_head_model(
                resolved_arch,
                hidden_sizes=effective_hidden,
                policy_hidden_sizes=policy_hidden_sizes,
            )
        else:
            self._model = _ValueMLP(hidden_sizes=effective_hidden)
        self.card_vocab_report = None
        if (
            weights_path is not None and model_factory is None
            and self._is_structured
        ):
            # Same load path as NNEvaluator: refuse an unstamped
            # checkpoint, remap a changed card vocabulary by name.
            from mtg_bot.training.card_vocab import (
                load_structured_checkpoint,
            )

            self._model, self.card_vocab_report = load_structured_checkpoint(
                self._model, Path(weights_path),
                map_location=self._device, card_vocab=card_vocab,
            )
        elif weights_path is not None and model_factory is None:
            state_dict = torch.load(
                weights_path,
                map_location=self._device,
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        # ----- torch.compile (after .to(device) + .eval()) -----------
        # Wrap the loaded model in ``torch.compile(mode='reduce-
        # overhead')`` when requested. ``reduce-overhead`` is geared
        # for many small calls — exactly our dispatch pattern (~25
        # forwards per decision × hundreds of decisions per game).
        # CUDA-only by default; on CPU/MPS the cost > benefit.
        #
        # The padded structured input substrate already pins seq_len
        # at MAX_SEQ_LEN per slot, so compile sees a single static
        # shape across requests — no dynamic-shape recompiles. For
        # the flat-feature MLP path the input shape is already
        # ``(batch, FEATURE_DIM)`` so the same static-shape contract
        # holds.
        #
        # Triton is required on Linux (the inductor backend), and
        # historically wasn't packaged for Windows. As of torch
        # 2.11 + cu128 Triton ships with the Windows wheels, but
        # keep a soft fallback so a future split doesn't crash
        # startup.
        self._model_compiled = False
        self._uncompiled_model = self._model
        if self._compile_model_requested and self._is_cuda:
            try:
                self._model = torch.compile(
                    self._model, mode="reduce-overhead", dynamic=False,
                )
                # ``torch.compile`` is lazy — the wrap only swaps the
                # callable. The actual graph compile happens at the
                # first forward. Triton-missing failures don't surface
                # here. We trigger the eager probe at warmup time
                # (``_run_warmup_forward``) and back off there if
                # inductor errors out. Set ``_model_compiled=True``
                # optimistically; warmup will demote it on failure.
                self._model_compiled = True
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "torch.compile failed at wrap time (%s); falling "
                    "back to eager execution.", exc,
                )
                self._model = self._uncompiled_model
                self._model_compiled = False

        # Resolve dimensions for slot allocation.
        from mtg_bot.features import FEATURE_DIM
        self._feature_dim = FEATURE_DIM
        if with_policy:
            from mtg_bot.training.action_encoding import ACTION_VOCAB_SIZE
            self._action_vocab_size = ACTION_VOCAB_SIZE
        else:
            self._action_vocab_size = 0

        # Allocate shared tensors. share_memory_() puts the storage on
        # shm (anonymous-mmap on Linux); fork-inherited. For structured
        # servers the input substrate is the four per-token tensors
        # plus a zero-element ``features`` placeholder kept for
        # handle-record uniformity (the structured client never touches
        # it). For MLP servers ``features`` is the flat-feature slot
        # pool and the per-token tensors stay None on the handle.
        #
        # ``pin_memory_enabled`` (CUDA-only) allocates separate
        # PINNED host staging tensors alongside the slot pool. The
        # dispatcher uses these to bounce slot rows through pinned
        # memory before issuing ``.to(device, non_blocking=True)`` —
        # which is the H2D fast path on CUDA. Pinning the SHARED slot
        # tensors directly doesn't work: ``share_memory_()`` and
        # ``pin_memory()`` are mutually exclusive at the storage
        # level (each replaces the underlying storage with its own
        # backing). Two-buffer split is the standard pattern for
        # IPC + GPU.
        if self._is_structured:
            # Per-token vocab sizes pulled off the model so the slot
            # tensors line up with the trunk's embedding tables.
            from mtg_bot.training.structured_model import (
                N_EMB_FIELDS,
                N_SCALAR_FIELDS,
            )
            # Scalar-slot width follows the loaded model's scalar
            # schema (3 for v1 checkpoints, wider for v2). A
            # ``model_factory`` model without a ``config`` falls
            # back to the v1 width.
            _cfg = getattr(self._model, "config", None)
            scalar_width = int(
                getattr(_cfg, "scalar_width", N_SCALAR_FIELDS)
            )
            # Zero-element ``features`` keeps the handle's contract
            # (always-present tensor) without burning shared memory.
            self._features = torch.zeros(
                self._n_slots, 0, dtype=torch.float32,
            ).share_memory_()
            self._token_type = torch.zeros(
                self._n_slots, self._max_seq_len, dtype=torch.int64,
            ).share_memory_()
            self._token_emb_ids = torch.zeros(
                self._n_slots, self._max_seq_len, N_EMB_FIELDS,
                dtype=torch.int64,
            ).share_memory_()
            self._token_scalars = torch.zeros(
                self._n_slots, self._max_seq_len, scalar_width,
                dtype=torch.float32,
            ).share_memory_()
            self._attention_mask = torch.zeros(
                self._n_slots, self._max_seq_len, dtype=torch.bool,
            ).share_memory_()
        else:
            self._features = torch.zeros(
                self._n_slots, self._feature_dim, dtype=torch.float32,
            ).share_memory_()
            self._token_type = None
            self._token_emb_ids = None
            self._token_scalars = None
            self._attention_mask = None
        self._values = torch.zeros(
            self._n_slots, 1, dtype=torch.float32,
        ).share_memory_()
        if with_policy:
            self._masks = torch.zeros(
                self._n_slots, self._action_vocab_size, dtype=torch.bool,
            ).share_memory_()
            self._policies = torch.zeros(
                self._n_slots, self._action_vocab_size, dtype=torch.float32,
            ).share_memory_()
        else:
            self._masks = None
            self._policies = None

        # ----- Pinned host staging buffers (CUDA fast path) ---------
        # Pre-allocate one pinned staging tensor per input slot
        # tensor. The dispatcher copies the gathered batch into these
        # before issuing ``.to(device, non_blocking=True)`` — which
        # is what unlocks the asynchronous H2D transfer.
        #
        # Sized to the FULL slot pool so a single ``batch_size <=
        # n_slots`` dispatch can use a contiguous slice without
        # reallocation. Stays None when pin_memory is disabled
        # (CPU host, or explicit ``pin_memory=False``).
        if self._pin_memory_enabled:
            try:
                self._pinned_features = (
                    torch.zeros_like(self._features).pin_memory()
                    if self._features.numel() > 0 else None
                )
                self._pinned_masks = (
                    torch.zeros_like(self._masks).pin_memory()
                    if self._masks is not None else None
                )
                if self._is_structured:
                    self._pinned_token_type = (
                        torch.zeros_like(self._token_type).pin_memory()
                    )
                    self._pinned_token_emb_ids = (
                        torch.zeros_like(self._token_emb_ids).pin_memory()
                    )
                    self._pinned_token_scalars = (
                        torch.zeros_like(self._token_scalars).pin_memory()
                    )
                    self._pinned_attention_mask = (
                        torch.zeros_like(self._attention_mask).pin_memory()
                    )
                else:
                    self._pinned_token_type = None
                    self._pinned_token_emb_ids = None
                    self._pinned_token_scalars = None
                    self._pinned_attention_mask = None
            except (RuntimeError, NotImplementedError) as exc:
                # Older torch / exotic platform: fall back to the
                # unpinned slow path. The dispatcher consults
                # ``self._pinned_*`` and degrades to direct .to()
                # automatically when any of them is None.
                self._logger.warning(
                    "Pinned host buffer allocation failed (%s); "
                    "dispatcher will fall back to unpinned H2D "
                    "copies.", exc,
                )
                self._pin_memory_enabled = False
                self._pinned_features = None
                self._pinned_masks = None
                self._pinned_token_type = None
                self._pinned_token_emb_ids = None
                self._pinned_token_scalars = None
                self._pinned_attention_mask = None
        else:
            self._pinned_features = None
            self._pinned_masks = None
            self._pinned_token_type = None
            self._pinned_token_emb_ids = None
            self._pinned_token_scalars = None
            self._pinned_attention_mask = None

        # Sync primitives. Use the host's preferred context — fork on
        # Linux/macOS, spawn on Windows. The free_slots / request_queue
        # / response_events objects from a spawn context are picklable
        # so they survive the executor's initializer pickle round-trip
        # on Windows; on Linux they're fork-inherited as before. The
        # *fork-only* phrasing in earlier revisions of this module was
        # historical: torch.Tensor.share_memory_() works fine across
        # spawn on Windows too.
        ctx = preferred_mp_context()
        # ``free_slots`` is the slot allocator. Pre-fill with all N
        # indices. Workers do .get() to acquire and .put() to release.
        self._free_slots: "mp.Queue[int]" = ctx.Queue()
        for i in range(self._n_slots):
            self._free_slots.put(i)
        # ``request_queue`` is the dispatcher's input. Workers put
        # slot_idx after writing the input row.
        self._request_queue: "mp.Queue[int]" = ctx.Queue()
        # Per-slot response events. Worker waits on its slot's event
        # after submitting; server sets it after writing the output.
        self._response_events = [ctx.Event() for _ in range(self._n_slots)]
        # Cross-process shutdown signal — set in ``stop()`` so that
        # any worker mid-``_call_value*`` notices the server is gone
        # and raises ``RuntimeError`` instead of hanging on a response
        # event that will never be signaled. The dispatcher does NOT
        # consult this; the dispatcher's own loop is gated by
        # ``_stop_evt`` (a thread-local event), since the dispatcher
        # lives in the same process as the server.
        self._server_stopped = ctx.Event()
        # Cross-process lock used by clients to serialize the multi-
        # slot acquire phase of ``_submit_batch``. See
        # ``SharedTensorServerHandle.acquire_lock`` for rationale.
        self._acquire_lock = ctx.Lock()

        # Dispatcher state.
        self._stop_evt = threading.Event()
        self._dispatch_thread: Optional[threading.Thread] = None

        # Telemetry.
        self.batches_dispatched = 0
        self.requests_served = 0
        self.batch_sizes: list[int] = []

    # ---- public lifecycle --------------------------------------------------

    def start(self) -> None:
        if self._dispatch_thread is not None:
            return
        self._stop_evt.clear()
        # Clear the cross-process shutdown signal so a restart-after-
        # stop server doesn't immediately fail every client call.
        self._server_stopped.clear()
        # Run the synthetic warmup forward BEFORE the dispatcher
        # thread spins up so the first real request sees a pre-
        # compiled model + warm CUDA kernels. Skipped (no-op) when
        # warmup is disabled (CPU host, or explicit ``warmup=False``).
        # Counts the number of warmup forwards so tests can assert
        # exactly-one-call semantics.
        #
        # Warmup also acts as the compile-failure probe: if
        # ``torch.compile`` was applied at construction time but
        # inductor errors out at first-call time (e.g. Triton missing
        # on Windows), the compiled callable raises here. We catch
        # the failure, demote the model back to its uncompiled form,
        # and re-run the warmup forward through eager. The dispatcher
        # then sees an eager model on subsequent calls instead of
        # rethrowing on every request.
        self.warmup_calls = 0
        if self._warmup_enabled:
            try:
                self._run_warmup_forward()
            except Exception as exc:  # noqa: BLE001
                if self._model_compiled:
                    self._logger.warning(
                        "torch.compile failed at first forward (%s); "
                        "falling back to eager execution. Re-running "
                        "warmup against the uncompiled model.", exc,
                    )
                    self._model = self._uncompiled_model
                    self._model_compiled = False
                    try:
                        self._run_warmup_forward()
                    except Exception as exc2:  # noqa: BLE001
                        self._logger.warning(
                            "Eager-mode warmup failed too (%s); first "
                            "real request will pay the cold-cache "
                            "cost.", exc2,
                        )
                else:
                    self._logger.warning(
                        "SHM eval server warmup failed (%s); first "
                        "real request will pay the cold-cache cost.",
                        exc,
                    )
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="shm-nn-eval-server-dispatcher",
            daemon=True,
        )
        self._dispatch_thread.start()

    def _run_warmup_forward(self) -> None:
        """Issue one synthetic forward pass at the full
        ``(batch_size, MAX_SEQ_LEN, ...)`` shape so the model's CUDA
        kernels (and any ``torch.compile`` artifact) JIT-compile and
        cache before the first real request. Discards the output.

        For structured servers, the synthetic batch matches the
        ``StructuredModel.forward`` input dict shape — random
        integers in vocab ranges for ids, random floats for scalars,
        all-True attention mask. For flat-feature servers, a random
        ``(batch_size, FEATURE_DIM)`` tensor is enough.

        Runs under the same ``torch.no_grad()`` + autocast context
        the dispatcher uses, so the warmed kernels match the hot
        path bit-identically.
        """
        device = self._device
        batch_size = self._batch_size
        with torch.no_grad(), self._autocast_ctx():
            if self._is_structured:
                cfg = self._uncompiled_model.config
                from mtg_bot.training.structured_model import (
                    ACTION_KIND_VOCAB_SIZE,
                    CONTROLLER_VOCAB_SIZE,
                    EMB_FIELD_ACTION_KIND,
                    EMB_FIELD_CONTROLLER,
                    EMB_FIELD_PRINTING,
                    EMB_FIELD_STEP,
                    EMB_FIELD_TARGET_KIND,
                    EMB_FIELD_TARGET_PRINTING,
                    EMB_FIELD_VARIANT,
                    N_EMB_FIELDS,
                    N_SCALAR_FIELDS,
                    N_TOKEN_TYPES,
                    STEP_VOCAB_SIZE,
                    TARGET_KIND_VOCAB_SIZE,
                    VARIANT_VOCAB_SIZE,
                )
                seq_len = max(1, self._max_seq_len)
                # Each emb-field has its own sub-vocab size; using a
                # single max across all fields would overflow the
                # smaller tables (action_kind=23, controller=3,
                # target_kind=6, variant=9, step=13). Build the
                # tensor field-by-field so each column stays in range.
                token_emb_ids = torch.zeros(
                    (batch_size, seq_len, N_EMB_FIELDS),
                    dtype=torch.int64, device=device,
                )
                n_printings_cap = max(1, cfg.n_printings)
                token_emb_ids[..., EMB_FIELD_PRINTING] = torch.randint(
                    0, n_printings_cap,
                    size=(batch_size, seq_len),
                    dtype=torch.int64, device=device,
                )
                token_emb_ids[..., EMB_FIELD_ACTION_KIND] = torch.randint(
                    0, ACTION_KIND_VOCAB_SIZE,
                    size=(batch_size, seq_len),
                    dtype=torch.int64, device=device,
                )
                token_emb_ids[..., EMB_FIELD_CONTROLLER] = torch.randint(
                    0, CONTROLLER_VOCAB_SIZE,
                    size=(batch_size, seq_len),
                    dtype=torch.int64, device=device,
                )
                token_emb_ids[..., EMB_FIELD_TARGET_PRINTING] = (
                    torch.randint(
                        0, n_printings_cap,
                        size=(batch_size, seq_len),
                        dtype=torch.int64, device=device,
                    )
                )
                token_emb_ids[..., EMB_FIELD_TARGET_KIND] = torch.randint(
                    0, TARGET_KIND_VOCAB_SIZE,
                    size=(batch_size, seq_len),
                    dtype=torch.int64, device=device,
                )
                token_emb_ids[..., EMB_FIELD_VARIANT] = torch.randint(
                    0, VARIANT_VOCAB_SIZE,
                    size=(batch_size, seq_len),
                    dtype=torch.int64, device=device,
                )
                token_emb_ids[..., EMB_FIELD_STEP] = torch.randint(
                    0, STEP_VOCAB_SIZE,
                    size=(batch_size, seq_len),
                    dtype=torch.int64, device=device,
                )
                batch = {
                    "token_type": torch.randint(
                        0, N_TOKEN_TYPES,
                        size=(batch_size, seq_len),
                        dtype=torch.int64, device=device,
                    ),
                    "token_emb_ids": token_emb_ids,
                    "token_scalars": torch.randn(
                        (
                            batch_size, seq_len,
                            getattr(cfg, "scalar_width", N_SCALAR_FIELDS),
                        ),
                        dtype=torch.float32, device=device,
                    ),
                    "attention_mask": torch.ones(
                        (batch_size, seq_len),
                        dtype=torch.bool, device=device,
                    ),
                }
                _ = self._model(batch)
            else:
                feats = torch.randn(
                    (batch_size, self._feature_dim),
                    dtype=torch.float32, device=device,
                )
                _ = self._model(feats)
        # Optional CUDA-side sync so the warmup actually completes
        # before we return (otherwise the dispatcher thread might
        # race the kernel-compile that's still in flight).
        if self._is_cuda:
            torch.cuda.synchronize(device)
        self.warmup_calls += 1

    def stop(self, timeout: float = 5.0) -> None:
        if self._dispatch_thread is None:
            return
        # Order matters here. Before joining the dispatcher we must
        # unblock any worker stuck in ``_call_value*`` on a per-slot
        # response event. Otherwise the worker's wait() hangs forever
        # (the dispatcher won't dispatch the slot after we stop) and
        # at process exit ``concurrent.futures.process._python_exit``
        # deadlocks joining the worker proc, which deadlocks joining
        # its own internal threads — observed in the field as a 40+
        # minute hang on eval-shard cleanup.
        # Step 1: tell workers the server is going down so any in-
        # flight call raises after wake.
        self._server_stopped.set()
        # Step 2: signal every per-slot response event. Workers
        # currently in wait() return immediately and check
        # _server_stopped on the next line. Workers about to enter
        # wait() see the event already set and exit the same path.
        for ev in self._response_events:
            try:
                ev.set()
            except Exception:  # noqa: BLE001
                pass
        # Step 3: stop the dispatcher.
        self._stop_evt.set()
        # Wake the dispatcher if it's blocked on the request_queue.
        # Sentinel value -1 (slot indices are always >= 0).
        try:
            self._request_queue.put(-1)
        except Exception:  # noqa: BLE001
            pass
        self._dispatch_thread.join(timeout=timeout)
        self._dispatch_thread = None

    def __enter__(self) -> "SharedTensorNNEvalServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def handle(self) -> SharedTensorServerHandle:
        # Resolve the structured tokenizer knobs from the loaded
        # model's config so the client's tokenize calls produce ids
        # the model's embedding tables can index without overflow.
        if self._is_structured:
            cfg = self._model.config
            n_printings_cap = int(cfg.n_printings)
            action_history_n = int(cfg.action_history_n)
            scalar_schema_version = int(
                getattr(cfg, "scalar_schema_version", 1)
            )
        else:
            n_printings_cap = 0
            action_history_n = 0
            scalar_schema_version = 0
        return SharedTensorServerHandle(
            free_slots=self._free_slots,
            request_queue=self._request_queue,
            response_events=self._response_events,
            server_stopped=self._server_stopped,
            acquire_lock=self._acquire_lock,
            features=self._features,
            values=self._values,
            masks=self._masks,
            policies=self._policies,
            with_policy=self._with_policy,
            n_slots=self._n_slots,
            feature_dim=self._feature_dim,
            action_vocab_size=self._action_vocab_size,
            architecture=self._architecture,
            max_seq_len=self._max_seq_len,
            token_type=self._token_type,
            token_emb_ids=self._token_emb_ids,
            token_scalars=self._token_scalars,
            attention_mask=self._attention_mask,
            structured_n_printings_cap=n_printings_cap,
            structured_action_history_n=action_history_n,
            structured_scalar_schema_version=scalar_schema_version,
        )

    # ---- dispatcher --------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """Drain request_queue, batch up to batch_size, run forward,
        write outputs, signal response events."""
        while not self._stop_evt.is_set():
            try:
                first = self._request_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if first == -1:
                # Stop sentinel.
                return

            batch: list[int] = [first]
            deadline = time.monotonic() + self._batch_wait_s
            while len(batch) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item == -1:
                    # Stop sentinel mid-batch — process what we have,
                    # then exit on next loop.
                    self._request_queue.put(-1)
                    break
                batch.append(item)

            self._dispatch_batch(batch)

    def _dispatch_batch(self, slot_indices: list[int]) -> None:
        """Gather inputs from the shared tensors, run the forward,
        scatter outputs back, signal response events.

        If the forward pass raises (out-of-range embedding id, NaN,
        OOM, ...), we MUST still set every slot's response_event +
        ``server_stopped`` — otherwise the worker thread parked in
        ``_await_response`` waits forever and the whole eval shard
        deadlocks at cleanup. The dispatcher's swallowed exception
        is re-raised by the loop's caller via the dispatcher thread
        log; the cross-process worker observes a ``server_stopped``
        wake and raises ``RuntimeError`` from its call site, which
        is the loud-failure path callers already handle.
        """
        if not slot_indices:
            return
        try:
            if self._is_structured:
                self._dispatch_structured(slot_indices)
            else:
                self._dispatch_flat(slot_indices)
        except BaseException:
            # Forward-pass failure: log, signal shutdown so any worker
            # blocked on a per-slot event wakes and observes
            # ``server_stopped``, set every event so no one stays
            # parked. Re-raise so the dispatcher thread dies cleanly
            # with a stack-trace in stderr; the loop will not retry
            # (we expect ``stop()`` to be called via the parent).
            self._logger.exception(
                "SharedTensorNNEvalServer dispatch failed; signaling "
                "workers and re-raising"
            )
            self._server_stopped.set()
            for ev in self._response_events:
                try:
                    ev.set()
                except Exception:  # noqa: BLE001
                    pass
            raise
        # Telemetry.
        self.batches_dispatched += 1
        self.requests_served += len(slot_indices)
        self.batch_sizes.append(len(slot_indices))

    def _stage_to_device(
        self,
        src: torch.Tensor,
        idx_tensor: torch.Tensor,
        pinned: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Gather ``src[idx_tensor]`` and ship it to ``self._device``.

        Uses the pre-allocated pinned staging buffer (when present) so
        the H2D copy can run ``non_blocking=True`` and overlap with
        compute. Falls back to a direct ``.to(device)`` when pinning
        is disabled or unsupported on this host.
        """
        gathered = src.index_select(0, idx_tensor)
        if pinned is not None:
            # Copy gathered rows into the pinned staging window,
            # then non-blocking ship to device. The staging buffer is
            # sized to ``n_slots`` rows; the gathered slice fits.
            n = gathered.shape[0]
            pinned[:n].copy_(gathered)
            return pinned[:n].to(self._device, non_blocking=True)
        return gathered.to(self._device, non_blocking=self._is_cuda)

    def _autocast_ctx(self):
        """Return a torch.autocast context manager for the dispatch
        forward when autocast is enabled, else a no-op contextlib
        nullcontext. The CUDA-only gate prevents accidental MPS / CPU
        autocast paths from getting wired up here.
        """
        if self._autocast_enabled:
            return torch.autocast(
                device_type="cuda", dtype=self._autocast_dtype,
            )
        import contextlib
        return contextlib.nullcontext()

    def _dispatch_flat(self, slot_indices: list[int]) -> None:
        """Flat-feature MLP dispatch path. Identical to the pre-Slice-F
        behaviour: gather features (+ masks) from the slot tensors,
        run one forward pass, scatter outputs back, signal response
        events.

        GPU fast path: when ``self._pin_memory_enabled``, gathered
        rows go through pinned staging and ``non_blocking=True``
        H2D copies; when ``self._autocast_enabled``, the forward
        pass runs inside ``torch.autocast(device_type='cuda',
        dtype=bf16|fp16)``.
        """
        idx_tensor = torch.tensor(slot_indices, dtype=torch.long)
        with torch.no_grad():
            feat_dev = self._stage_to_device(
                self._features, idx_tensor, self._pinned_features,
            )
            if self._with_policy:
                assert self._masks is not None
                assert self._policies is not None
                mask_batch = self._stage_to_device(
                    self._masks, idx_tensor, self._pinned_masks,
                )
                with self._autocast_ctx():
                    value_t, policy_logits = self._model(feat_dev)
                # Apply the mask: -inf where mask is False, then softmax.
                # Use the policy_logits dtype (bf16/fp16/fp32) so the
                # masked-fill scalar matches.
                neg_inf = torch.tensor(
                    float("-inf"), dtype=policy_logits.dtype,
                    device=self._device,
                )
                masked_logits = torch.where(
                    mask_batch, policy_logits, neg_inf,
                )
                policy = torch.softmax(masked_logits, dim=-1)
                # Write outputs back to the original slots.
                value_cpu = value_t.detach().to(torch.float32).cpu()
                policy_cpu = policy.detach().to(torch.float32).cpu()
                for i, slot in enumerate(slot_indices):
                    self._values[slot, 0] = value_cpu[i].squeeze()
                    self._policies[slot].copy_(policy_cpu[i])
                    self._response_events[slot].set()
            else:
                with self._autocast_ctx():
                    value_t = self._model(feat_dev)
                if isinstance(value_t, tuple):
                    value_t = value_t[0]
                value_cpu = value_t.detach().to(torch.float32).cpu()
                for i, slot in enumerate(slot_indices):
                    self._values[slot, 0] = value_cpu[i].squeeze()
                    self._response_events[slot].set()

    def _dispatch_structured(self, slot_indices: list[int]) -> None:
        """Structured-input dispatch path. Gather the four per-token
        tensors from the slot pool, forward through ``StructuredModel``,
        apply the legal-action mask + softmax to the policy logits,
        scatter back into the fixed-shape ``values`` / ``policies`` slots.

        The model's value head is tanh-activated (per
        ``StructuredModel.forward``); no extra clamp is needed on this
        side.
        """
        assert self._token_type is not None
        assert self._token_emb_ids is not None
        assert self._token_scalars is not None
        assert self._attention_mask is not None
        assert self._masks is not None
        assert self._policies is not None
        idx_tensor = torch.tensor(slot_indices, dtype=torch.long)
        with torch.no_grad():
            batch = {
                "token_type": self._stage_to_device(
                    self._token_type, idx_tensor,
                    self._pinned_token_type,
                ),
                "token_emb_ids": self._stage_to_device(
                    self._token_emb_ids, idx_tensor,
                    self._pinned_token_emb_ids,
                ),
                "token_scalars": self._stage_to_device(
                    self._token_scalars, idx_tensor,
                    self._pinned_token_scalars,
                ),
                "attention_mask": self._stage_to_device(
                    self._attention_mask, idx_tensor,
                    self._pinned_attention_mask,
                ),
            }
            mask_batch = self._stage_to_device(
                self._masks, idx_tensor, self._pinned_masks,
            )
            with self._autocast_ctx():
                out = self._model(batch)
            value_t = out["value"]
            policy_logits = out["policy_logits"]
            # ``value_t`` is shaped ``(B,)`` per StructuredModel's
            # contract; some PyTorch versions return ``(B, 1)`` if a
            # squeeze axis collapses to 0-D — normalize so the write
            # below can index ``[i]``.
            if value_t.dim() == 0:
                value_t = value_t.unsqueeze(0)
            elif value_t.dim() > 1:
                value_t = value_t.squeeze(-1)
            neg_inf = torch.tensor(
                float("-inf"), dtype=policy_logits.dtype,
                device=self._device,
            )
            masked_logits = torch.where(
                mask_batch, policy_logits, neg_inf,
            )
            policy = torch.softmax(masked_logits, dim=-1)
            value_cpu = value_t.detach().to(torch.float32).cpu()
            policy_cpu = policy.detach().to(torch.float32).cpu()
            for i, slot in enumerate(slot_indices):
                self._values[slot, 0] = value_cpu[i]
                self._policies[slot].copy_(policy_cpu[i])
                self._response_events[slot].set()


class SharedTensorNNEvalClient(Evaluator):
    """Drop-in Evaluator that ships requests to a SharedTensorNNEvalServer
    via shared-memory tensors. Public API matches ``NNEvaluator`` /
    ``NNEvalClient`` exactly so PIMC configs can swap implementations
    transparently.

    Constructed in a worker process from a SharedTensorServerHandle
    inherited via fork. Stateless w.r.t. the server — each request
    acquires a slot, writes input rows, signals the server, waits for
    the response event, reads output rows, releases the slot.
    """

    def __init__(self, handle: SharedTensorServerHandle) -> None:
        self._h = handle
        self._with_policy = handle.with_policy
        self._architecture = getattr(handle, "architecture", ARCHITECTURE_SHARED)
        self._is_structured = (
            self._architecture == ARCHITECTURE_STRUCTURED
        )
        # Cap the per-request seq_len at the server's configured
        # MAX_SEQ_LEN. Requests with seq_len > cap are rejected with a
        # clear error rather than silently truncated (truncation would
        # drop the freshest action tokens, the signal Step 1 cares
        # most about).
        self._max_seq_len = getattr(handle, "max_seq_len", 0)

    @property
    def with_policy(self) -> bool:
        return self._with_policy

    @property
    def architecture(self) -> str:
        return self._architecture

    # ---- Evaluator API ---------------------------------------------------

    def _evaluate_nonterminal(
        self, state: "GameState", player_id: "PlayerID",
    ) -> float:
        """Concrete implementation of the abstract base method.
        Base class ``Evaluator.evaluate`` handles the terminal short-
        circuit; this method only ever sees non-terminal states.
        Server batches across worker-process boundaries already; the
        slot round-trip is the unit of work."""
        # Base ``Evaluator.evaluate`` filtered terminals; this is one
        # non-terminal state submitted to the network. Counted client-
        # side: the game's eval count is the same regardless of where
        # the server physically runs the matmul.
        eval_counters.record_nn_eval(1)
        if self._is_structured:
            # The structured server always has a policy head; route
            # through the with-policy path and discard the policy
            # array. This keeps both architectures' value-only callers
            # bit-equivalent (same tokenize + forward), at the cost of
            # one wasted policy-head materialization per call. The
            # alternative — a separate value-only-only structured
            # submit path — would diverge from the model's actual
            # forward graph and re-introduce the silent-output-drift
            # risk Slice E specifically guarded against.
            ae = _action_encoding()
            mask = ae.legal_action_mask(state, player_id)
            tokens = self._tokenize_state(state, player_id)
            value, _ = self._call_structured_value_policy(tokens, mask)
            return value
        engine = _engine_state(state)
        feats = extract_features(engine, player_id)
        return self._call_value(feats)

    # ``evaluate`` inherits the base-class implementation: terminal
    # short-circuit, then delegate to ``_evaluate_nonterminal``.
    # ``batch_evaluate`` is overridden below to submit all N requests
    # before waiting on any — without this, the dispatcher only ever
    # sees one in-flight request per worker, capping cross-worker
    # fusion at ``N_workers`` instead of ``N_workers * batch_size``.
    # See ``docs/LEVER2_INVESTIGATION_2026_05_10.md`` for the wire-up
    # numbers behind this design.

    def evaluate_with_policy(
        self, state: "GameState", player_id: "PlayerID",
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
        # Past the terminal short-circuit: one non-terminal
        # (value, policy) submit.
        eval_counters.record_nn_eval(1)
        if self._is_structured:
            mask = ae.legal_action_mask(state, player_id)
            tokens = self._tokenize_state(state, player_id)
            return self._call_structured_value_policy(tokens, mask)
        engine = _engine_state(state)
        feats = extract_features(engine, player_id)
        mask = ae.legal_action_mask(state, player_id)
        return self._call_value_policy(feats, mask)

    def batch_evaluate(
        self, states: "Sequence[GameState]", player_id: "PlayerID",
    ) -> list[float]:
        """Batched value-only evaluation.

        Submits all non-terminal slot indices to the request queue
        *before* waiting on any response, so the dispatcher fuses the
        N requests into one (or few) GPU forward passes. The legacy
        for-loop sent each request as its own round-trip and capped
        each worker's in-flight count at 1 — see the Lever 2
        investigation doc for the wall-time math.

        Terminal states short-circuit per-element to the engine's
        true result (matching ``Evaluator.evaluate``) without touching
        the slot machinery. The returned list preserves input order
        and mixes terminal + non-terminal values transparently.

        Backpressure: if the non-terminal subset exceeds the slot
        pool size, the call automatically chunks into ``ceil(N /
        n_slots)`` sub-batches run sequentially through the existing
        batched-submit path, then concatenates results in input order.
        ``N <= n_slots`` (the production-sized case) is unchanged —
        a single submit, identical to the pre-chunking behavior.
        """
        if not states:
            return []

        # Partition. Terminal states get their value immediately and
        # never acquire a slot. Non-terminal states extract features
        # up front so the slot's I/O window is as tight as possible.
        out: list[float] = [0.0] * len(states)
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
        nonterminal_token_dicts: list[dict[str, np.ndarray]] = []
        nonterminal_masks: list[np.ndarray] = []
        ae = _action_encoding() if self._is_structured else None
        for i, s in enumerate(states):
            if s.is_terminal():
                winner = s.winner()
                if winner is None:
                    out[i] = 0.0
                else:
                    out[i] = 1.0 if winner == player_id else -1.0
                continue
            nonterminal_indices.append(i)
            if self._is_structured:
                # Structured value-only path tokenizes and forwards
                # through the full structured submit (policy head is
                # part of the same forward pass), then drops the
                # policy. Same rationale as ``_evaluate_nonterminal``
                # — keeps bit-equivalence with the direct
                # NNEvaluator(architecture='structured') call.
                nonterminal_token_dicts.append(
                    self._tokenize_state(s, player_id),
                )
                assert ae is not None
                nonterminal_masks.append(
                    ae.legal_action_mask(s, player_id),
                )
            else:
                engine = _engine_state(s)
                nonterminal_features.append(
                    extract_features(engine, player_id),
                )

        if not nonterminal_indices:
            return out

        # One logical batched value-only submit over the non-terminal
        # subset (the client may internally chunk to slot-pool size;
        # still one logical evaluator invocation).
        eval_counters.record_nn_eval(len(nonterminal_indices))
        if self._is_structured:
            values, _ = self._call_chunked_structured(
                token_dicts=nonterminal_token_dicts,
                masks=nonterminal_masks,
            )
        else:
            values, _ = self._call_chunked(
                features=nonterminal_features,
                masks=None,
                vocab_size=0,
            )
        for k, v in enumerate(values):
            out[nonterminal_indices[k]] = v
        return out

    def batch_evaluate_with_policy(
        self, states: "Sequence[GameState]", player_id: "PlayerID",
    ) -> list[tuple[float, np.ndarray]]:
        """Batched (value + policy) evaluation. Same shape as
        :meth:`batch_evaluate`: submit all slot indices, then drain
        responses. See that method for the backpressure rationale.

        Terminal states short-circuit to ``(±1/0, all-zeros policy)``,
        matching the single-state ``evaluate_with_policy`` semantics.
        Result list preserves input order.
        """
        if not self._with_policy:
            raise RuntimeError(
                "batch_evaluate_with_policy requires a policy-head server",
            )
        ae = _action_encoding()
        vocab_size = ae.ACTION_VOCAB_SIZE
        if not states:
            return []

        # Partition. Terminal states get the synthetic (value, zeros)
        # answer immediately; non-terminals are bundled into one
        # batched submit.
        out: list[tuple[float, np.ndarray]] = [
            (0.0, np.zeros(vocab_size, dtype=np.float32))
            for _ in states
        ]
        nonterminal_indices: list[int] = []
        nonterminal_features: list[np.ndarray] = []
        nonterminal_token_dicts: list[dict[str, np.ndarray]] = []
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
            nonterminal_indices.append(i)
            nonterminal_masks.append(
                ae.legal_action_mask(s, player_id),
            )
            if self._is_structured:
                nonterminal_token_dicts.append(
                    self._tokenize_state(s, player_id),
                )
            else:
                engine = _engine_state(s)
                nonterminal_features.append(
                    extract_features(engine, player_id),
                )

        if not nonterminal_indices:
            return out

        # One logical batched (value, policy) submit over the
        # non-terminal subset.
        eval_counters.record_nn_eval(len(nonterminal_indices))
        if self._is_structured:
            values, policies = self._call_chunked_structured(
                token_dicts=nonterminal_token_dicts,
                masks=nonterminal_masks,
            )
        else:
            values, policies = self._call_chunked(
                features=nonterminal_features,
                masks=nonterminal_masks,
                vocab_size=vocab_size,
            )
        for k, v in enumerate(values):
            out[nonterminal_indices[k]] = (v, policies[k])
        return out

    def _call_chunked(
        self,
        *,
        features: list[np.ndarray],
        masks: Optional[list[np.ndarray]],
        vocab_size: int,
    ) -> tuple[list[float], list[np.ndarray]]:
        """Submit ``features`` (and optionally ``masks``) through the
        SHM slot machinery, chunking into sub-batches of at most
        ``self._h.n_slots`` so any oversize request degrades gracefully
        instead of crashing or blocking on a too-small slot pool.

        Returns ``(values, policies)``. ``policies`` is an empty list
        when ``masks is None`` (value-only path); otherwise one
        np.ndarray of length ``vocab_size`` per input row, in input
        order.

        Common case (``N <= n_slots``): one chunk, single submit —
        identical to the pre-chunking path. Only oversize batches
        take the multi-chunk route, which is the rare config-mismatch
        case PR #49 hit before n_slots was bumped.

        Invariants preserved across chunks:
        - Input order: results are appended in the order chunks are
          processed, which iterates ``features``/``masks`` in order.
        - Slot lifecycle: each chunk acquires + writes + submits +
          waits + releases its slots within a ``try/finally`` so
          exceptions (including ``server_stopped`` raised by
          ``_await_response``) don't leak slots.
        - ``server_stopped`` detection: the first sub-batch to hit a
          stopped server raises immediately; subsequent chunks are
          not started.
        """
        n = len(features)
        if masks is not None:
            assert len(masks) == n
        if n == 0:
            return [], []

        chunk_size = min(n, self._h.n_slots)
        values: list[float] = []
        policies: list[np.ndarray] = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_features = features[start:end]
            chunk_masks = (
                masks[start:end] if masks is not None else None
            )
            slots = self._submit_batch(
                features=chunk_features,
                masks=chunk_masks,
            )
            try:
                for slot in slots:
                    self._await_response(slot)
                    values.append(
                        float(self._h.values[slot, 0].item())
                    )
                    if masks is not None:
                        assert self._h.policies is not None
                        policies.append(
                            self._h.policies[slot].clone().numpy()
                        )
            finally:
                for slot in slots:
                    self._h.free_slots.put(slot)
        return values, policies

    def _submit_batch(
        self,
        *,
        features: list[np.ndarray],
        masks: Optional[list[np.ndarray]],
    ) -> list[int]:
        """Acquire ``len(features)`` slots, write all input rows, then
        submit every slot index to the request queue.

        The "submit all before waiting on any" ordering is the whole
        point: it gives the dispatcher up to ``len(features)`` items
        on its queue at once, which is what enables it to fuse them
        into a single forward pass.

        The acquire phase is serialized across workers by the
        handle's ``acquire_lock`` (an mp.Lock). Without that, N
        concurrent workers each grabbing slots one at a time can
        each accumulate a partial allocation, exhaust the pool, and
        deadlock — see the multi-worker contention regression test.
        Holding the lock only for the acquire is deliberate: writes,
        submits, waits, and reads all run in parallel as before, so
        cross-worker dispatcher fusion is unaffected.

        On any failure (e.g. interrupted ``free_slots.get()``), the
        already-acquired slots are released before re-raising so the
        pool doesn't leak. Slots returned by this method are owned by
        the caller until released.
        """
        n = len(features)
        if masks is not None:
            assert self._h.masks is not None
            assert self._h.policies is not None
            assert len(masks) == n

        # Acquire slots atomically (all N or none). The lock prevents
        # the partial-allocation deadlock pattern. If we're
        # interrupted partway through the get() loop (BaseException,
        # e.g. KeyboardInterrupt), release whatever we have before
        # re-raising so the pool doesn't leak.
        slots: list[int] = []
        try:
            with self._h.acquire_lock:
                for _ in range(n):
                    slots.append(self._h.free_slots.get())
            # Write all input rows. Tight loop — no waiting between
            # rows, no submissions yet.
            for k, slot in enumerate(slots):
                np_feats = np.ascontiguousarray(
                    features[k], dtype=np.float32,
                )
                self._h.features[slot].copy_(torch.from_numpy(np_feats))
                if masks is not None:
                    np_mask = np.ascontiguousarray(masks[k], dtype=bool)
                    self._h.masks[slot].copy_(torch.from_numpy(np_mask))
            # Submit ALL slot indices before returning so the
            # dispatcher sees them as a batched window.
            for slot in slots:
                self._h.request_queue.put(slot)
        except BaseException:
            for slot in slots:
                self._h.free_slots.put(slot)
            raise
        return slots

    # ---- raw request paths (for bench / testing) ------------------------

    # Poll interval for the shutdown-aware wait. Normal calls return in
    # microseconds-to-low-milliseconds (substrate p50 74us, production
    # p95 ~3-4 ms), so wait() almost always returns True before the
    # first tick elapses — zero perceptible cost on the happy path.
    # Worst case: a worker observes server shutdown within this many
    # seconds. 0.5s is comfortably below the per-game timeout and well
    # within ``stop(timeout=5.0)``'s budget.
    _SHUTDOWN_POLL_S = 0.5

    def _await_response(self, slot: int) -> None:
        """Wait for the dispatcher to write outputs for ``slot``, with
        shutdown detection. Raises ``RuntimeError`` if the server is
        stopped before the response is written."""
        while not self._h.response_events[slot].wait(
            timeout=self._SHUTDOWN_POLL_S,
        ):
            # wait() timed out without being set. If the server has
            # gone down, bail; otherwise loop and keep waiting.
            if self._h.server_stopped.is_set():
                raise RuntimeError(
                    "SharedTensorNNEvalServer stopped before response "
                    f"was written (slot {slot})"
                )
        # The event fired. Two cases:
        # (a) Dispatcher wrote outputs and set the event — normal path.
        # (b) Server's ``stop()`` set the event to wake us; outputs
        #     are NOT valid. Distinguish via ``server_stopped``.
        self._h.response_events[slot].clear()
        if self._h.server_stopped.is_set():
            raise RuntimeError(
                "SharedTensorNNEvalServer stopped while waiting for "
                f"response (slot {slot})"
            )

    def _call_value(self, feats: np.ndarray) -> float:
        slot = self._h.free_slots.get()
        try:
            np_feats = np.ascontiguousarray(feats, dtype=np.float32)
            self._h.features[slot].copy_(torch.from_numpy(np_feats))
            self._h.request_queue.put(slot)
            self._await_response(slot)
            return float(self._h.values[slot, 0].item())
        finally:
            self._h.free_slots.put(slot)

    def _call_value_policy(
        self, feats: np.ndarray, mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        slot = self._h.free_slots.get()
        try:
            np_feats = np.ascontiguousarray(feats, dtype=np.float32)
            self._h.features[slot].copy_(torch.from_numpy(np_feats))
            assert self._h.masks is not None
            assert self._h.policies is not None
            np_mask = np.ascontiguousarray(mask, dtype=bool)
            self._h.masks[slot].copy_(torch.from_numpy(np_mask))
            self._h.request_queue.put(slot)
            self._await_response(slot)
            value = float(self._h.values[slot, 0].item())
            policy = self._h.policies[slot].clone().numpy()
            return value, policy
        finally:
            self._h.free_slots.put(slot)

    # ---- structured (per-token) request paths ----------------------------

    def _tokenize_state(
        self,
        state: "GameState",
        viewer_id: "PlayerID",
    ) -> dict[str, np.ndarray]:
        """Run ``state_tokenizer.tokenize_state`` against the engine
        state behind ``state``. The recent-action history is the
        state's adapter-carried ``recent_actions`` (recorded by
        ``MtgGameStateAdapter.apply``, shared across clones — see
        ``mtg_bot.search.action_history``), exactly as the direct
        ``NNEvaluator(architecture='structured')`` path and the
        training writer (``data_gen``) tokenize it. States without
        the attribute fall back to an empty history.

        ``n_printings_cap`` is threaded from the handle's
        ``structured_n_printings_cap`` (resolved from the loaded
        model's ``StructuredModelConfig.n_printings`` at handle
        time). Without this, the tokenizer would default to its own
        constant (320) and an EoE-style large-card-pool deployment
        with a smaller-than-default model n_printings would overflow
        the embedding table mid-dispatch.
        """
        import importlib
        tokenizer = importlib.import_module(
            "mtg_bot.training.state_tokenizer",
        )
        engine = _engine_state(state)
        recent_actions = getattr(state, "recent_actions", ())
        n_cap = self._h.structured_n_printings_cap
        action_n = self._h.structured_action_history_n
        schema = getattr(self._h, "structured_scalar_schema_version", 0)
        kwargs: dict[str, Any] = {}
        if n_cap > 0:
            kwargs["n_printings_cap"] = n_cap
        if schema > 0:
            # Emit the scalar row width the server's slot tensor /
            # model ``scalar_proj`` were sized for.
            kwargs["scalar_schema_version"] = schema
        if action_n > 0:
            return tokenizer.tokenize_state(
                engine, int(viewer_id), recent_actions, action_n, **kwargs,
            )
        return tokenizer.tokenize_state(
            engine, int(viewer_id), recent_actions, **kwargs,
        )

    def _validate_seq_len(self, token_dict: dict[str, np.ndarray]) -> None:
        """Verify the tokenized sequence fits in the server's per-slot
        cap. Architectural-bound violation surfaces here rather than
        silently truncating the action history.
        """
        seq_len = int(token_dict["token_type"].shape[0])
        if seq_len > self._max_seq_len:
            raise ValueError(
                f"tokenized state seq_len={seq_len} exceeds the SHM "
                f"server's MAX_SEQ_LEN={self._max_seq_len}; raise the "
                f"server's ``max_seq_len`` kwarg or shrink the "
                f"per-state token set. Refusing to silently truncate "
                f"because the tail tokens carry the freshest action "
                f"history (the strongest signal Step 1's trunk has)."
            )

    def _write_structured_row(
        self,
        slot: int,
        token_dict: dict[str, np.ndarray],
        mask: np.ndarray,
    ) -> None:
        """Pad ``token_dict`` up to ``MAX_SEQ_LEN`` and write it into
        the slot's shared tensors, alongside the policy-head legal-
        action mask.

        Padded positions get ``attention_mask=False`` and zeroed
        categorical / scalar values; the model's ``padding_idx=0`` on
        every embedding plus the ``src_key_padding_mask`` flip in
        ``StructuredModel.forward`` together ensure padded rows
        contribute zero to the trunk's attention output.
        """
        self._validate_seq_len(token_dict)
        assert self._h.token_type is not None
        assert self._h.token_emb_ids is not None
        assert self._h.token_scalars is not None
        assert self._h.attention_mask is not None
        assert self._h.masks is not None
        seq_len = int(token_dict["token_type"].shape[0])
        max_len = self._max_seq_len
        # Zero the trailing pad region of every per-token tensor for
        # this slot so a stale write from a previous request can't
        # leak into the current forward pass.
        self._h.token_type[slot].zero_()
        self._h.token_emb_ids[slot].zero_()
        self._h.token_scalars[slot].zero_()
        self._h.attention_mask[slot].zero_()
        if seq_len > 0:
            self._h.token_type[slot, :seq_len].copy_(
                torch.from_numpy(
                    np.ascontiguousarray(
                        token_dict["token_type"], dtype=np.int64,
                    ),
                ),
            )
            self._h.token_emb_ids[slot, :seq_len].copy_(
                torch.from_numpy(
                    np.ascontiguousarray(
                        token_dict["token_emb_ids"], dtype=np.int64,
                    ),
                ),
            )
            self._h.token_scalars[slot, :seq_len].copy_(
                torch.from_numpy(
                    np.ascontiguousarray(
                        token_dict["token_scalars"], dtype=np.float32,
                    ),
                ),
            )
            self._h.attention_mask[slot, :seq_len].copy_(
                torch.from_numpy(
                    np.ascontiguousarray(
                        token_dict["attention_mask"], dtype=bool,
                    ),
                ),
            )
        # Pad-region of ``attention_mask`` stays False from the
        # zero-fill above — the trunk's key_padding_mask flip in
        # forward() turns those into "ignore" positions.
        # Suppress an unused-var hint about max_len; it pins the
        # shape contract the assertion below upholds.
        assert self._h.attention_mask[slot].shape[0] == max_len
        # Legal-action mask for the policy head.
        np_mask = np.ascontiguousarray(mask, dtype=bool)
        self._h.masks[slot].copy_(torch.from_numpy(np_mask))

    def _call_structured_value_policy(
        self,
        token_dict: dict[str, np.ndarray],
        mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Single-state structured submit. Mirrors
        ``_call_value_policy`` but writes the per-token tensors
        instead of the flat feature row."""
        slot = self._h.free_slots.get()
        try:
            self._write_structured_row(slot, token_dict, mask)
            self._h.request_queue.put(slot)
            self._await_response(slot)
            assert self._h.policies is not None
            value = float(self._h.values[slot, 0].item())
            policy = self._h.policies[slot].clone().numpy()
            return value, policy
        finally:
            self._h.free_slots.put(slot)

    def _call_chunked_structured(
        self,
        *,
        token_dicts: list[dict[str, np.ndarray]],
        masks: list[np.ndarray],
    ) -> tuple[list[float], list[np.ndarray]]:
        """Structured analogue of :meth:`_call_chunked`. Chunks
        oversized batches into sub-batches of at most ``n_slots`` so
        the call degrades gracefully when the request exceeds the
        slot pool's size.

        Invariants preserved:

        - Input order: results are appended in chunk-iteration order,
          which matches the input order.
        - Slot lifecycle: each chunk acquires + writes + submits +
          waits + releases its slots within a try/finally so
          exceptions (e.g. ``server_stopped`` raised by
          ``_await_response``) don't leak slots.
        - ``server_stopped`` detection: the first sub-batch to hit a
          stopped server raises immediately; subsequent chunks are
          not started.

        Pre-validates every request's seq_len up front so an oversize
        request fails fast (before acquiring any slots) rather than
        partway through writing.
        """
        n = len(token_dicts)
        assert len(masks) == n
        if n == 0:
            return [], []
        for td in token_dicts:
            self._validate_seq_len(td)

        chunk_size = min(n, self._h.n_slots)
        values: list[float] = []
        policies: list[np.ndarray] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_tokens = token_dicts[start:end]
            chunk_masks = masks[start:end]
            slots = self._submit_batch_structured(
                token_dicts=chunk_tokens,
                masks=chunk_masks,
            )
            try:
                for slot in slots:
                    self._await_response(slot)
                    values.append(float(self._h.values[slot, 0].item()))
                    assert self._h.policies is not None
                    policies.append(
                        self._h.policies[slot].clone().numpy(),
                    )
            finally:
                for slot in slots:
                    self._h.free_slots.put(slot)
        return values, policies

    def _submit_batch_structured(
        self,
        *,
        token_dicts: list[dict[str, np.ndarray]],
        masks: list[np.ndarray],
    ) -> list[int]:
        """Structured analogue of :meth:`_submit_batch`. Acquires
        ``len(token_dicts)`` slots under ``acquire_lock``, writes each
        slot's per-token tensors + mask, then submits every slot index
        to the request queue.

        The acquire-all-before-write ordering, the lock-only-on-acquire
        rule, and the BaseException-safe release-on-failure handling
        all mirror the flat-feature submit path. See its docstring for
        the deeper rationale.
        """
        n = len(token_dicts)
        assert len(masks) == n
        slots: list[int] = []
        try:
            with self._h.acquire_lock:
                for _ in range(n):
                    slots.append(self._h.free_slots.get())
            for k, slot in enumerate(slots):
                self._write_structured_row(slot, token_dicts[k], masks[k])
            for slot in slots:
                self._h.request_queue.put(slot)
        except BaseException:
            for slot in slots:
                self._h.free_slots.put(slot)
            raise
        return slots

    def close(self) -> None:
        # Nothing to close — no per-worker resources.
        pass


def _engine_state(state: "GameState") -> Any:
    """Unwrap ``state.engine_state`` if present (matches NNEvalClient's
    helper); ``GameState`` instances may carry an engine-state attribute
    for feature extraction, or pass themselves through."""
    engine = getattr(state, "engine_state", None)
    if engine is not None:
        return engine
    return state
