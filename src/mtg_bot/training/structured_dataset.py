"""Dataset + collate_fn for ``StructuredModel`` training.

The structured-token npz files written by ``data_gen.save_dataset`` are
padded *per file* to the file's max sequence length. At train time
we don't want to pay for any of that padding: the trainer's
``DataLoader`` calls ``__getitem__`` millions of times per epoch and
the per-row Python overhead dominates GPU compute on a small NN
trunk.

This module flattens every npz's padded tokens into a single packed
"flat" storage indexed by per-row ``row_offsets``:

- ``_token_type_flat``      shape (total_real_tokens,) int16
- ``_token_emb_ids_flat``   shape (total_real_tokens, N_EMB_FIELDS) int32
- ``_token_scalars_flat``   shape (total_real_tokens, N_SCALAR_FIELDS) float32
- ``_row_offsets``          shape (n_total + 1,) int64 — cumulative
                            real-token count; ``_row_offsets[i+1] -
                            _row_offsets[i]`` is row i's sequence
                            length, ``_token_type_flat[start:end]``
                            is row i's real tokens with NO padding.

``__getitem__(idx)`` becomes a single numpy slice + ``torch.from_numpy``
per array — one materializing copy per row (which also widens the
narrow token-id dtypes back to ``int64``, see below), no per-row
``.sum()`` to recompute the sequence length, no Python-level list
indexing.

**Stored token-id dtypes.** The token id columns are stored at the
smallest integer width their vocabularies can ever need (see
:data:`FLAT_STORE_DTYPES`): ``token_type`` is int16 (12 token types
today), ``token_emb_ids`` is int32 (printing ids — the widest field —
top out in the low hundreds today and the model's other id fields are
< 32). Both are widened to ``torch.int64`` in ``__getitem__`` so the
model's ``nn.Embedding`` layers still receive ``LongTensor``s and no
model / trainer code changes. Store-build time asserts every value
fits the narrowed dtype and fails loudly, naming the column and the
offending value, rather than letting numpy truncate silently. This
cuts 53% off the two token-id columns (64 -> 30 bytes per token):
19.0 GB -> 8.9 GB on the 296M-token / 4.76M-row ``iterate_data_s2``
pool — measured on a 500-shard subset in
``docs/BENCHMARKING_FINDINGS.md`` §8A. ``token_scalars`` stays float32: the
schema-v2 scalar fields include /7, /10, /20 and /40 normalizers whose
values are not exactly representable in float16, so narrowing it
would change the training input.

History note: an earlier eager-concat-with-global-max-padding loader
OOM'd at ~42 GiB on the iter_016 dataset (1701 files, 7M samples).
A first refactor switched to per-file ragged storage (one ndarray per
source npz, padded to that file's local max_len) which fit in memory
but introduced enough Python overhead per ``__getitem__`` —
``np.searchsorted`` to map global idx -> file_idx, list indexing into
the per-file arrays, ``.astype`` casts on every row — that an RTX 5090
ran at <10% utilization waiting on the dataloader. The current flat
storage is BOTH smaller (no per-file padding) AND faster (single
contiguous slice per array).

Public API
==========

- :func:`load_structured_dataset` — discover every ``*.npz`` under a
  directory, flatten each file's real tokens, and return a
  :class:`StructuredDataset` ready for the trainer's ``DataLoader``.
  The build is **streaming**: a cheap sizing pass (npy headers + the
  bool attention masks) gives the exact final shapes, the flat
  ``.npy`` files are pre-allocated with ``np.lib.format.open_memmap``,
  and each source file is decoded into its slice and dropped. Peak
  RSS is O(largest single npz), not O(pool). The pre-2026-09 path
  decoded every file into RAM and concatenated, peaking at ~24 GB per
  1M rows (a 4M-row pool committed 96 GB and effectively hung).
- :func:`sweep_stale_memmap_dirs` — remove flat stores orphaned by a
  killed run, identified by a dead owner pid (or, with no owner
  marker, by age). ``$MTG_BOT_STRUCTURED_SWEEP=0`` opts out.
- :func:`sweep_abandoned_build_dirs` — the same for half-built stores
  left inside a persistent ``cache_dir``.
- :func:`shutdown_dataloader_workers` — drop a ``DataLoader``'s
  persistent workers so their memmap handles stop pinning the store.
- :class:`StructuredDataset` — wraps the packed-flat storage. The
  legacy single-array constructor (used by tests + direct callers)
  still accepts file-padded inputs and flattens them at construction
  time. The :meth:`from_per_file_arrays` and
  :meth:`from_flat_arrays` classmethods cover the other ingest paths.
- :func:`structured_collate` — pads variable-length per-row tensors
  to the batch's max ``seq_len`` and stacks them into the dict
  :meth:`StructuredModel.forward` expects.

This module deliberately does NOT touch ``trainer.py``'s existing
``train_value_network`` path. The sibling ``train_structured_model``
function (in ``trainer.py``) wires this loader to the new model.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mtg_bot.training.action_encoding import (
    ACTION_ENCODING_VERSION_V2, ACTION_ENCODING_VERSION_V3,
    ACTION_VOCAB_SIZE, SLOT_RESOLUTION_BASE, action_vocab_size,
)
from mtg_bot.training.data_gen import file_group_ids

_log = logging.getLogger(__name__)


_STRUCT_NPZ_KEYS = (
    "structured_token_type",
    "structured_token_emb_ids",
    "structured_token_scalars",
    "structured_attention_mask",
)
"""Mirror of ``data_gen._STRUCT_NPZ_KEYS`` to avoid the private import."""


MEMMAP_DIR_PREFIX = "mtg_bot_structured_dataset_"
"""Prefix of every flat-store directory this module creates in %TEMP%."""

FLAT_STORE_VERSION = 3
"""Bump when the on-disk flat-store layout changes. Part of the
persistent-cache key so an old cache can never be mistaken for a new
one.

History: v1 = all token arrays int64. v2 (2026-09-24) = narrowed
token-id dtypes (see :data:`FLAT_STORE_DTYPES`); the tensors handed
to the model are unchanged (still int64), only the bytes on disk
shrink."""


TOKEN_TYPE_DTYPE = np.int16
"""On-disk dtype of ``token_type_flat``. ``N_TOKEN_TYPES`` is 12 and
grows by one per new token kind, so int16 has ~2700x headroom; int8 /
uint8 were rejected because they would leave only 100-odd spare ids
and (for uint8) wrap a negative sentinel silently."""

TOKEN_EMB_IDS_DTYPE = np.int32
"""On-disk dtype of ``token_emb_ids_flat``. The widest field is the
printing id (``DEFAULT_N_PRINTINGS`` = 320 today; a full-Magic
printing table is ~10^5), the rest are < 32. int32 covers any
plausible card database; int16 would not survive a multi-set registry
expansion."""

TOKEN_SCALARS_DTYPE = np.float32
"""On-disk dtype of ``token_scalars_flat``. NOT narrowed to float16:
schema-v2 fields normalize by 7 / 10 / 20 / 40, and those values are
not exactly representable in float16 — narrowing would silently
perturb the training input."""

FLAT_STORE_DTYPES: dict[str, np.dtype] = {
    "token_type_flat.npy": np.dtype(TOKEN_TYPE_DTYPE),
    "token_emb_ids_flat.npy": np.dtype(TOKEN_EMB_IDS_DTYPE),
    "token_scalars_flat.npy": np.dtype(TOKEN_SCALARS_DTYPE),
    "row_offsets.npy": np.dtype(np.int64),
    "outcomes.npy": np.dtype(np.float32),
    "root_values.npy": np.dtype(np.float32),
    "policy_targets.npy": np.dtype(np.float32),
    "action_masks.npy": np.dtype(np.bool_),
    "has_policy.npy": np.dtype(np.bool_),
    "group_ids.npy": np.dtype(np.int64),
}
"""The dtype of every array in a v2 flat store, by file name. The
on-disk contract; a test pins it."""


_DTYPE_CONSTANT_BY_COLUMN: dict[str, str] = {
    "token_type_flat": "TOKEN_TYPE_DTYPE",
    "token_emb_ids_flat": "TOKEN_EMB_IDS_DTYPE",
}


def _check_fits_dtype(
    column: str,
    arr: np.ndarray,
    dtype,
    *,
    source: object = None,
) -> np.ndarray:
    """Raise unless every value in ``arr`` fits ``dtype``.

    numpy truncates silently on an out-of-range integer assignment
    (``int16_store[:] = 40000`` stores -25536, no warning), which would
    corrupt a training corpus invisibly. Every write into a narrowed
    store column is checked here first: the min/max scan is one pass
    over an already-decompressed chunk, and the error names the
    column, the offending value and (for a streaming build) the source
    npz, so a bad file is identifiable in a 2500-file pool.

    Returns ``arr`` so callers can use it inline.
    """
    target = np.dtype(dtype)
    arr = np.asarray(arr)
    if not arr.size or arr.dtype == target:
        return arr
    info = np.iinfo(target)
    lo = int(arr.min())
    hi = int(arr.max())
    if lo < info.min or hi > info.max:
        bad = lo if lo < info.min else hi
        where = f" (source {source})" if source is not None else ""
        constant = _DTYPE_CONSTANT_BY_COLUMN.get(column, "the column's")
        raise ValueError(
            f"structured flat store: {column} value {bad} does not "
            f"fit {target.name} (range [{info.min}, {info.max}])"
            f"{where}. The vocabulary has outgrown the store's dtype "
            f"— widen structured_dataset.{constant} and bump "
            f"FLAT_STORE_VERSION."
        )
    return arr


def _checked_narrow(
    column: str,
    arr: np.ndarray,
    dtype,
    *,
    source: object = None,
) -> np.ndarray:
    """:func:`_check_fits_dtype` followed by a contiguous cast. Used
    by the in-RAM path, which needs a real array to ``np.save``; the
    streaming path only checks and lets the memmap assignment do the
    conversion in place."""
    return np.ascontiguousarray(
        _check_fits_dtype(column, arr, dtype, source=source), dtype=dtype,
    )


SWEEP_ENV = "MTG_BOT_STRUCTURED_SWEEP"
"""Set to ``0``/``false`` to disable the stale-temp-dir startup sweep."""

SWEEP_HOURS_ENV = "MTG_BOT_STRUCTURED_SWEEP_HOURS"
"""Override the stale-dir age threshold (float hours)."""

CACHE_DIR_ENV = "MTG_BOT_STRUCTURED_CACHE_DIR"
"""Default value for ``load_structured_dataset(cache_dir=...)``."""

DEFAULT_SWEEP_HOURS = 2.0
"""Age at which a flat store with no identifiable owner is swept.

Lowered from 12 h (2026-09-24) now that a single leaked store is tens
of GB. The 12 h margin existed to protect a long-running training job
whose store is old but still mapped; that protection is now explicit
(:data:`_OWNER_NAME` + a pid-liveness check), so the age threshold
only governs stores with no owner marker at all — pre-marker layouts
and dirs whose marker never made it to disk."""

_OWNER_NAME = "owner.json"
"""Marker file written into every process-owned store: the pid that
built it, that pid's host, and when. The sweep reclaims a store whose
owner process is gone AT ANY AGE — which is what makes a hard-killed
or ``os._exit``-ed run's store disappear on the next load instead of
12 h later."""

# (attribute name, file name) for every array in a flat store. The
# order is the write order; ``_open_memmaps`` reads the same list.
_ARRAY_FILES: tuple[tuple[str, str], ...] = (
    ("_token_type_flat", "token_type_flat.npy"),
    ("_token_emb_ids_flat", "token_emb_ids_flat.npy"),
    ("_token_scalars_flat", "token_scalars_flat.npy"),
    ("_row_offsets", "row_offsets.npy"),
    ("_outcomes", "outcomes.npy"),
    ("_root_values", "root_values.npy"),
    ("_policy_targets", "policy_targets.npy"),
    ("_action_masks", "action_masks.npy"),
    ("_has_policy", "has_policy.npy"),
    ("_group_ids", "group_ids.npy"),
)

_MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------
# Owned-directory bookkeeping + deterministic cleanup
# ---------------------------------------------------------------------

_OWNED_DIRS: set[str] = set()
"""Flat-store directories this process created and must delete. A
module-level set (not just ``StructuredDataset.__del__``) so an
``atexit`` hook can still reclaim them when the interpreter tears the
dataset down without running its finalizer."""


def _new_owned_dir() -> str:
    """``mkdtemp`` a flat-store directory and register it for cleanup.

    The owner marker is written immediately (before any array), so the
    window in which a concurrent sweep sees an unidentified directory
    is a few milliseconds wide — and that window is covered by the age
    threshold anyway.
    """
    path = tempfile.mkdtemp(prefix=MEMMAP_DIR_PREFIX)
    _OWNED_DIRS.add(path)
    _write_owner_marker(path)
    return path


def _write_owner_marker(path: str | Path) -> None:
    """Record which process owns this store. Best-effort: a store
    without a marker is still swept, just on the age threshold."""
    try:
        (Path(path) / _OWNER_NAME).write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "created_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:  # pragma: no cover - unwritable temp dir
        _log.debug("structured dataset: could not write owner marker")


def _read_owner_marker(directory: Path) -> Optional[dict]:
    """Owner marker as a dict, or None if absent / unreadable."""
    try:
        raw = (directory / _OWNER_NAME).read_text(encoding="utf-8")
        marker = json.loads(raw)
    except (OSError, ValueError):
        return None
    return marker if isinstance(marker, dict) else None


def _pid_is_running(pid: int) -> bool:
    """True if a process with ``pid`` exists on this host.

    Deliberately conservative — anything ambiguous (access denied, an
    API that won't answer) reports True, because the cost of a false
    "alive" is one sweep cycle of delay while the cost of a false
    "dead" is deleting a live training run's store.

    ``os.kill(pid, 0)`` is NOT usable on Windows: CPython maps any
    signal other than CTRL_C_EVENT / CTRL_BREAK_EVENT onto
    ``TerminateProcess``, i.e. the liveness probe would kill the
    process it asks about. Hence the ctypes ``OpenProcess`` path.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes  # local: only Windows needs it
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        error_invalid_parameter = 87
        # ``use_last_error`` keeps the error code with the call (a
        # bare ``GetLastError()`` can be clobbered by ctypes' own
        # bookkeeping), and the explicit HANDLE argtypes stop the
        # 64-bit handle being truncated through the default c_int.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, pid,
        )
        if not handle:
            # 87 = no such pid. Anything else (5 = access denied for a
            # process owned by another user) means it exists.
            return ctypes.get_last_error() != error_invalid_parameter
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == still_active
            return True  # pragma: no cover - API refused to answer
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - POSIX, other user
        return True
    except OSError:  # pragma: no cover - defensive
        return True
    return True


def _store_owner_is_live(directory: Path) -> Optional[bool]:
    """True / False if the store's owner process is alive / gone;
    None when the store carries no usable marker (pre-marker layout,
    a marker from another host, or a truncated write) and the caller
    must fall back to the age threshold."""
    marker = _read_owner_marker(directory)
    if marker is None or marker.get("host") != platform.node():
        return None
    try:
        pid = int(marker["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    return _pid_is_running(pid)


def _close_memmap(arr: object) -> None:
    """Release a numpy memmap's file handle *now*.

    numpy releases the underlying ``mmap`` lazily (on GC), and Windows
    refuses to delete a file while any mapping of it is open — which is
    exactly why the old ``TemporaryDirectory(ignore_cleanup_errors=True)``
    teardown leaked 20-50 GB per training run. Closing explicitly makes
    the subsequent ``rmtree`` succeed on the first try.
    """
    mm = getattr(arr, "_mmap", None)
    if mm is None:
        return
    try:
        mm.close()
    except (BufferError, ValueError, OSError):  # pragma: no cover
        # Exported buffers (a live view someone else holds) keep the
        # mapping alive; the rmtree retry loop + GC handles it.
        pass


def _remove_store_dir(
    path: str | Path, *, attempts: int = 6, delay: float = 0.02,
) -> bool:
    """``rmtree`` with GC-then-retry. Returns True if the dir is gone.

    On Windows a file that is still mapped (a DataLoader worker that has
    not exited, a stray array view) cannot be unlinked. Each retry runs
    ``gc.collect()`` first, which drops the lazily-released memmap
    handles, then tries again with a short backoff.
    """
    p = str(path)
    for attempt in range(attempts):
        if not os.path.isdir(p):
            return True
        try:
            shutil.rmtree(p)
            return True
        except OSError:
            gc.collect()
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
    gone = not os.path.isdir(p)
    if not gone:
        _log.warning(
            "structured dataset: could not remove flat-store dir %s "
            "(still in use?); the startup sweep will retry", p,
        )
    return gone


@atexit.register
def _cleanup_owned_dirs_atexit() -> None:  # pragma: no cover - atexit
    for path in list(_OWNED_DIRS):
        _OWNED_DIRS.discard(path)
        try:
            _remove_store_dir(path, attempts=2)
        except Exception:  # noqa: BLE001
            pass


def _sweep_enabled() -> bool:
    raw = os.environ.get(SWEEP_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def sweep_stale_memmap_dirs(
    *,
    max_age_hours: Optional[float] = None,
    root: Optional[Path] = None,
) -> int:
    """Delete leftover ``mtg_bot_structured_dataset_*`` dirs.

    A hard-killed or ``os._exit``-ed training run leaves its flat
    store behind (tens of GB; ``atexit`` covers every *normal* exit
    path — including an unhandled exception, ``sys.exit``,
    ``KeyboardInterrupt`` and pytest-xdist workers — but cannot run
    when the process is killed or calls ``os._exit``). This sweep runs
    at the top of :func:`load_structured_dataset` and removes, under
    ``root`` (default: the system temp dir):

    - any store whose owner process (see :data:`_OWNER_NAME`) is gone,
      **at any age** — that is the killed-run case, and waiting hours
      to reclaim tens of GB is what made this a disk-space incident;
    - any store with no usable owner marker whose mtime is older than
      ``max_age_hours`` (default: ``$MTG_BOT_STRUCTURED_SWEEP_HOURS``
      or :data:`DEFAULT_SWEEP_HOURS`).

    A store whose owner pid is still running is never removed, however
    old it is, so a multi-hour training run is safe on POSIX too —
    previously only Windows protected it (by refusing to unlink a
    mapped file) and the age threshold was the only POSIX margin.
    Directories owned by *this* process are never touched.

    Opt out with ``MTG_BOT_STRUCTURED_SWEEP=0``. Returns the number of
    directories removed.
    """
    if max_age_hours is None:
        raw = os.environ.get(SWEEP_HOURS_ENV)
        try:
            max_age_hours = (
                float(raw) if raw not in (None, "") else DEFAULT_SWEEP_HOURS
            )
        except ValueError:
            max_age_hours = DEFAULT_SWEEP_HOURS
    root_path = Path(root) if root is not None else Path(tempfile.gettempdir())
    if not root_path.is_dir():
        return 0
    cutoff = time.time() - max(0.0, float(max_age_hours)) * 3600.0
    removed = 0
    try:
        entries = list(root_path.iterdir())
    except OSError:  # pragma: no cover - unreadable temp root
        return 0
    for entry in entries:
        if not entry.name.startswith(MEMMAP_DIR_PREFIX):
            continue
        if str(entry) in _OWNED_DIRS:
            continue
        try:
            if not entry.is_dir():
                continue
            owner_live = _store_owner_is_live(entry)
            if owner_live:
                continue  # a live run owns this store
            if owner_live is None and entry.stat().st_mtime > cutoff:
                continue  # unidentified and still young
        except OSError:  # pragma: no cover - vanished mid-sweep
            continue
        if _remove_store_dir(entry, attempts=1):
            removed += 1
    if removed:
        _log.info(
            "structured dataset: swept %d stale flat-store dir(s) from %s",
            removed, root_path,
        )
    return removed


def sweep_abandoned_build_dirs(cache_dir: Path) -> int:
    """Remove ``.building-*`` stores in a persistent cache dir whose
    builder process is gone.

    The temp sweep cannot see these: a cached build writes into
    ``<cache_dir>/.building-<signature>-<pid>`` and only renames it to
    its final name when complete, so a killed build leaves a partial
    (tens of GB) directory that nothing ever reclaims. A live builder
    is identified by its owner marker and left alone.
    """
    removed = 0
    try:
        entries = list(Path(cache_dir).iterdir())
    except OSError:  # pragma: no cover - unreadable cache dir
        return 0
    for entry in entries:
        if not entry.name.startswith(".building-"):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:  # pragma: no cover - vanished mid-sweep
            continue
        if str(entry) in _OWNED_DIRS:
            continue
        if _store_owner_is_live(entry) is not False:
            # Live builder, or a marker we can't judge (another host's
            # cache on a share): leave it.
            continue
        if _remove_store_dir(entry, attempts=1):
            removed += 1
    if removed:
        _log.info(
            "structured dataset: removed %d abandoned build dir(s) in %s",
            removed, cache_dir,
        )
    return removed


class StructuredDataset(Dataset):
    """Variable-length per-row structured-token dataset.

    Each item is a Python dict with keys:

    - ``token_type``      ``LongTensor(seq_len,)``
    - ``token_emb_ids``   ``LongTensor(seq_len, N_EMB_FIELDS)``
    - ``token_scalars``   ``FloatTensor(seq_len, N_SCALAR_FIELDS)``
    - ``attention_mask``  ``BoolTensor(seq_len,)`` — all True
      (padding is dropped at file-load time so the collate_fn only
      sees real tokens; this keeps the per-batch max accurate.)
    - ``outcome``         ``FloatTensor(scalar)`` — in ``{-1, 0, +1}``
    - ``policy_target``   ``FloatTensor(action_vocab_size,)`` — visit
      distribution; all-zero if the row carries no policy supervision.
    - ``action_mask``     ``BoolTensor(action_vocab_size,)`` — True
      for legal slots; all-False if no supervision.
    - ``has_policy``      ``BoolTensor(scalar)`` — True iff
      ``policy_target.sum() > 0.5``.

    The arrays are kept on CPU; the trainer is responsible for moving
    each batch to its target device after collation.

    **Internal storage.** Token arrays are stored "flat" — every
    npz's real-token rows packed into a single contiguous numpy
    array, with per-row slice bounds given by ``_row_offsets``. This
    eliminates BOTH the per-file padding overhead (saves ~50% memory
    vs file-padded layouts) AND the per-row Python overhead in
    ``__getitem__`` (a single ``arr[start:end]`` materialization +
    ``torch.from_numpy`` per array, no list indexing).

    The two id columns are stored at their narrowest safe integer
    width (``token_type`` int16, ``token_emb_ids`` int32 — see
    :data:`FLAT_STORE_DTYPES`) and widened to int64 by the
    materializing copy in ``__getitem__``, so the dict above is
    dtype-identical to the all-int64 store it replaced.
    """

    def __init__(
        self,
        token_type: np.ndarray,
        token_emb_ids: np.ndarray,
        token_scalars: np.ndarray,
        attention_mask: np.ndarray,
        outcomes: np.ndarray,
        policy_targets: np.ndarray,
        action_masks: np.ndarray,
        group_ids: Optional[np.ndarray] = None,
    ) -> None:
        n = token_type.shape[0]
        if (
            token_emb_ids.shape[0] != n
            or token_scalars.shape[0] != n
            or attention_mask.shape[0] != n
            or outcomes.shape[0] != n
            or policy_targets.shape[0] != n
            or action_masks.shape[0] != n
        ):
            raise ValueError(
                "All structured-dataset arrays must share row count; "
                f"got {token_type.shape[0]} / {token_emb_ids.shape[0]} "
                f"/ {token_scalars.shape[0]} / {attention_mask.shape[0]} "
                f"/ {outcomes.shape[0]} / {policy_targets.shape[0]} "
                f"/ {action_masks.shape[0]}"
            )
        # Flatten the file-padded inputs at construction time so the
        # internal storage matches the ``from_flat_arrays`` shape.
        tt_flat, emb_flat, sca_flat, row_offsets = _flatten_file_padded(
            token_type, token_emb_ids, token_scalars, attention_mask,
        )
        self._init_flat(
            token_type_flat=tt_flat,
            token_emb_ids_flat=emb_flat,
            token_scalars_flat=sca_flat,
            row_offsets=row_offsets,
            outcomes=np.asarray(outcomes, dtype=np.float32),
            policy_targets=np.asarray(policy_targets, dtype=np.float32),
            action_masks=np.asarray(action_masks, dtype=np.bool_),
            group_ids=group_ids,
        )

    @classmethod
    def from_per_file_arrays(
        cls,
        *,
        token_type_per_file: list[np.ndarray],
        token_emb_ids_per_file: list[np.ndarray],
        token_scalars_per_file: list[np.ndarray],
        attention_mask_per_file: list[np.ndarray],
        outcomes_per_file: list[np.ndarray],
        policy_targets_per_file: list[np.ndarray],
        action_masks_per_file: list[np.ndarray],
        group_ids_per_file: Optional[list[np.ndarray]] = None,
    ) -> "StructuredDataset":
        """Build from a list of per-file padded arrays. Each list
        entry holds one source npz's arrays at the file's native
        ``max_seq_len`` padding. Files may have different
        ``max_seq_len`` values; rows are flattened to their real
        length at construction time."""
        n_files = len(token_type_per_file)
        if (
            len(token_emb_ids_per_file) != n_files
            or len(token_scalars_per_file) != n_files
            or len(attention_mask_per_file) != n_files
            or len(outcomes_per_file) != n_files
            or len(policy_targets_per_file) != n_files
            or len(action_masks_per_file) != n_files
        ):
            raise ValueError(
                "All per-file lists must share length; got "
                f"{n_files} / {len(token_emb_ids_per_file)} / "
                f"{len(token_scalars_per_file)} / "
                f"{len(attention_mask_per_file)} / "
                f"{len(outcomes_per_file)} / "
                f"{len(policy_targets_per_file)} / "
                f"{len(action_masks_per_file)}"
            )
        for fi in range(n_files):
            n = token_type_per_file[fi].shape[0]
            if (
                token_emb_ids_per_file[fi].shape[0] != n
                or token_scalars_per_file[fi].shape[0] != n
                or attention_mask_per_file[fi].shape[0] != n
                or outcomes_per_file[fi].shape[0] != n
                or policy_targets_per_file[fi].shape[0] != n
                or action_masks_per_file[fi].shape[0] != n
            ):
                raise ValueError(
                    f"Per-file arrays in file index {fi} must share "
                    "row count; got "
                    f"{token_type_per_file[fi].shape[0]} / "
                    f"{token_emb_ids_per_file[fi].shape[0]} / "
                    f"{token_scalars_per_file[fi].shape[0]} / "
                    f"{attention_mask_per_file[fi].shape[0]} / "
                    f"{outcomes_per_file[fi].shape[0]} / "
                    f"{policy_targets_per_file[fi].shape[0]} / "
                    f"{action_masks_per_file[fi].shape[0]}"
                )
        # Flatten each file's tokens, then concatenate the flat
        # arrays + the per-file row arrays.
        tt_parts: list[np.ndarray] = []
        emb_parts: list[np.ndarray] = []
        sca_parts: list[np.ndarray] = []
        row_len_parts: list[np.ndarray] = []
        for fi in range(n_files):
            tt_f, emb_f, sca_f, row_lens_f = _flatten_one_file(
                token_type_per_file[fi],
                token_emb_ids_per_file[fi],
                token_scalars_per_file[fi],
                attention_mask_per_file[fi],
            )
            tt_parts.append(tt_f)
            emb_parts.append(emb_f)
            sca_parts.append(sca_f)
            row_len_parts.append(row_lens_f)
        if tt_parts:
            tt_flat = np.concatenate(tt_parts)
            emb_flat = np.concatenate(emb_parts)
            sca_flat = np.concatenate(
                _normalize_scalar_parts(
                    sca_parts, [f"file index {fi}" for fi in range(n_files)],
                )
            )
            row_lens_all = np.concatenate(row_len_parts).astype(np.int64)
            outcomes = np.concatenate(outcomes_per_file).astype(np.float32)
            policy_targets = np.concatenate(
                policy_targets_per_file
            ).astype(np.float32)
            action_masks = np.concatenate(
                action_masks_per_file
            ).astype(np.bool_)
        else:
            # Empty input lists: produce a zero-row dataset.
            tt_flat = np.zeros((0,), dtype=TOKEN_TYPE_DTYPE)
            emb_flat = np.zeros((0, 0), dtype=TOKEN_EMB_IDS_DTYPE)
            sca_flat = np.zeros((0, 0), dtype=TOKEN_SCALARS_DTYPE)
            row_lens_all = np.zeros((0,), dtype=np.int64)
            outcomes = np.zeros((0,), dtype=np.float32)
            policy_targets = np.zeros(
                (0, ACTION_VOCAB_SIZE), dtype=np.float32,
            )
            action_masks = np.zeros(
                (0, ACTION_VOCAB_SIZE), dtype=np.bool_,
            )
        row_offsets = np.concatenate(
            ([0], np.cumsum(row_lens_all)),
        ).astype(np.int64)
        group_ids: Optional[np.ndarray] = None
        if group_ids_per_file is not None:
            if len(group_ids_per_file) != n_files:
                raise ValueError(
                    f"group_ids_per_file length {len(group_ids_per_file)} "
                    f"!= number of files {n_files}"
                )
            group_ids = (
                np.concatenate(group_ids_per_file).astype(np.int64)
                if group_ids_per_file else np.zeros((0,), dtype=np.int64)
            )
        obj = cls.__new__(cls)
        obj._init_flat(
            token_type_flat=tt_flat,
            token_emb_ids_flat=emb_flat,
            token_scalars_flat=sca_flat,
            row_offsets=row_offsets,
            outcomes=outcomes,
            policy_targets=policy_targets,
            action_masks=action_masks,
            group_ids=group_ids,
        )
        return obj

    @classmethod
    def from_flat_arrays(
        cls,
        *,
        token_type_flat: np.ndarray,
        token_emb_ids_flat: np.ndarray,
        token_scalars_flat: np.ndarray,
        row_offsets: np.ndarray,
        outcomes: np.ndarray,
        policy_targets: np.ndarray,
        action_masks: np.ndarray,
        group_ids: Optional[np.ndarray] = None,
    ) -> "StructuredDataset":
        """Build directly from pre-flattened arrays. The caller is
        responsible for ensuring ``row_offsets`` is monotone
        non-decreasing, starts at 0, and ends at
        ``token_type_flat.shape[0]``.

        ``group_ids`` (optional, ``(n_rows,)`` int) is the per-row
        game group id the trainer uses for a game-level train/val
        split. ``None`` = every row is its own group (no game
        structure known — only appropriate for synthetic data)."""
        n_total = int(row_offsets.shape[0]) - 1
        if n_total < 0:
            raise ValueError("row_offsets must have length >= 1")
        if int(row_offsets[0]) != 0:
            raise ValueError(
                f"row_offsets must start at 0; got {int(row_offsets[0])}"
            )
        if int(row_offsets[-1]) != int(token_type_flat.shape[0]):
            raise ValueError(
                f"row_offsets[-1] must equal token_type_flat.shape[0]; "
                f"got {int(row_offsets[-1])} vs {int(token_type_flat.shape[0])}"
            )
        if (
            token_emb_ids_flat.shape[0] != token_type_flat.shape[0]
            or token_scalars_flat.shape[0] != token_type_flat.shape[0]
        ):
            raise ValueError(
                "Flat token arrays must share first dimension; got "
                f"{token_type_flat.shape[0]} / "
                f"{token_emb_ids_flat.shape[0]} / "
                f"{token_scalars_flat.shape[0]}"
            )
        if (
            outcomes.shape[0] != n_total
            or policy_targets.shape[0] != n_total
            or action_masks.shape[0] != n_total
        ):
            raise ValueError(
                f"Per-row arrays must share row count {n_total}; got "
                f"{outcomes.shape[0]} / {policy_targets.shape[0]} / "
                f"{action_masks.shape[0]}"
            )
        if group_ids is not None and np.asarray(group_ids).shape[0] != n_total:
            raise ValueError(
                f"group_ids rows {np.asarray(group_ids).shape[0]} != "
                f"row count {n_total}"
            )
        obj = cls.__new__(cls)
        obj._init_flat(
            token_type_flat=token_type_flat,
            token_emb_ids_flat=token_emb_ids_flat,
            token_scalars_flat=token_scalars_flat,
            row_offsets=row_offsets,
            outcomes=outcomes,
            policy_targets=policy_targets,
            action_masks=action_masks,
            group_ids=group_ids,
        )
        return obj

    def _init_flat(
        self,
        *,
        token_type_flat: np.ndarray,
        token_emb_ids_flat: np.ndarray,
        token_scalars_flat: np.ndarray,
        row_offsets: np.ndarray,
        outcomes: np.ndarray,
        policy_targets: np.ndarray,
        action_masks: np.ndarray,
        group_ids: Optional[np.ndarray] = None,
        root_values: Optional[np.ndarray] = None,
    ) -> None:
        # **Memmap-backed storage.** Write each flat array to a
        # ``.npy`` file in a temp directory, then ``mmap_mode='r'``
        # reopens them so the dataset object itself holds only file
        # paths + tiny header bytes. This is the storage layout that
        # works with DataLoader workers on Windows: pickling the
        # dataset to a worker sends just the directory path; the
        # worker re-opens the memmaps and the OS page cache shares
        # the data across all readers. (``torch.Tensor.share_memory_()``
        # was tried first but doesn't shrink the raw pickle —
        # see commit history.)
        #
        # Memory ceiling: tracks only what's actively page-cached,
        # not the full 30 GiB flat storage. The 7M-sample dataset
        # works with ``num_workers=4`` because each worker mmaps the
        # same files (no per-worker copy in the pipe).
        # Pre-compute has_policy once so each ``__getitem__`` avoids
        # a per-row ``.sum()`` over the 517-wide policy vector.
        has_policy = (
            np.ascontiguousarray(policy_targets, dtype=np.float32).sum(
                axis=1
            )
            > 0.5
        ).astype(np.bool_)

        # Owned temp directory: removed by :meth:`close` (called from
        # ``__del__``, and from the module ``atexit`` hook as a
        # backstop). ``__setstate__`` (in DataLoader workers) does NOT
        # own the directory — workers just re-open the memmaps.
        # Cleanup closes every memmap handle first; numpy releases them
        # lazily on GC and Windows refuses to unlink a mapped file,
        # which is what made the old ``TemporaryDirectory`` teardown
        # silently leak the whole flat store.
        self._owned_dir: Optional[str] = _new_owned_dir()
        memmap_dir = Path(self._owned_dir)
        # Save each array as a separate ``.npy`` file. ``.npy`` is
        # the simplest format that supports ``mmap_mode='r'`` — it
        # writes a small header followed by the contiguous data
        # block.
        np.save(
            memmap_dir / "token_type_flat.npy",
            _checked_narrow(
                "token_type_flat", token_type_flat, TOKEN_TYPE_DTYPE,
            ),
        )
        np.save(
            memmap_dir / "token_emb_ids_flat.npy",
            _checked_narrow(
                "token_emb_ids_flat", token_emb_ids_flat,
                TOKEN_EMB_IDS_DTYPE,
            ),
        )
        np.save(
            memmap_dir / "token_scalars_flat.npy",
            np.ascontiguousarray(
                token_scalars_flat, dtype=TOKEN_SCALARS_DTYPE,
            ),
        )
        np.save(
            memmap_dir / "row_offsets.npy",
            np.ascontiguousarray(row_offsets, dtype=np.int64),
        )
        np.save(
            memmap_dir / "outcomes.npy",
            np.ascontiguousarray(outcomes, dtype=np.float32),
        )
        # Search root value per row; NaN = unknown (see data_gen).
        if root_values is None:
            root_values_arr = np.full(
                (int(np.shape(outcomes)[0]),), np.nan, dtype=np.float32,
            )
        else:
            root_values_arr = np.ascontiguousarray(
                root_values, dtype=np.float32,
            )
            if root_values_arr.shape != np.shape(outcomes):
                raise ValueError(
                    f"root_values shape {root_values_arr.shape} != "
                    f"outcomes shape {np.shape(outcomes)}"
                )
        np.save(memmap_dir / "root_values.npy", root_values_arr)
        np.save(
            memmap_dir / "policy_targets.npy",
            np.ascontiguousarray(policy_targets, dtype=np.float32),
        )
        np.save(
            memmap_dir / "action_masks.npy",
            np.ascontiguousarray(action_masks, dtype=np.bool_),
        )
        np.save(
            memmap_dir / "has_policy.npy",
            has_policy,
        )
        # Per-row game group id for the trainer's game-level split.
        # Default (no game structure known) = one group per row.
        n_rows_total = int(row_offsets.shape[0]) - 1
        if group_ids is None:
            group_ids_arr = np.arange(n_rows_total, dtype=np.int64)
        else:
            group_ids_arr = np.ascontiguousarray(group_ids, dtype=np.int64)
            if group_ids_arr.shape[0] != n_rows_total:
                raise ValueError(
                    f"group_ids rows {group_ids_arr.shape[0]} != "
                    f"row count {n_rows_total}"
                )
        np.save(memmap_dir / "group_ids.npy", group_ids_arr)
        self._memmap_dir = str(memmap_dir)
        self._open_memmaps()

    @classmethod
    def from_memmap_dir(
        cls,
        memmap_dir: str | Path,
        *,
        own_dir: bool = False,
    ) -> "StructuredDataset":
        """Wrap an already-built flat store on disk.

        ``own_dir=True`` transfers ownership: the returned dataset
        deletes ``memmap_dir`` in :meth:`close` / ``__del__``. Used by
        :func:`load_structured_dataset`, which streams the flat store
        straight to disk instead of building it in RAM first.
        """
        obj = cls.__new__(cls)
        obj._memmap_dir = str(memmap_dir)
        obj._owned_dir = str(memmap_dir) if own_dir else None
        if own_dir:
            _OWNED_DIRS.add(str(memmap_dir))
        obj._open_memmaps()
        return obj

    @property
    def group_ids(self) -> np.ndarray:
        """``(len(self),)`` int64 per-row game group ids. Rows that
        share a value came from the same self-play game and must land
        on the same side of the train/val split (see
        ``trainer.grouped_train_val_split``).

        Returns a plain in-RAM copy, not a view on the memmap: a live
        view would pin the mapping open and block temp-dir cleanup on
        Windows (the caller keeps this array for the whole training
        run)."""
        return np.array(self._group_ids)

    def _open_memmaps(self) -> None:
        """Open the on-disk ``.npy`` files as read-only memmaps.

        Called from ``_init_flat`` in the parent process and from
        ``__setstate__`` in each DataLoader worker. Each call opens
        new file handles in the calling process; the underlying file
        pages are shared via the OS page cache.
        """
        d = Path(self._memmap_dir)
        for attr, fname in _ARRAY_FILES:
            setattr(self, attr, np.load(d / fname, mmap_mode="r"))
        self._n_total = int(self._row_offsets.shape[0]) - 1

    def close(self) -> None:
        """Release every memmap handle and delete an owned flat store.

        Idempotent. Called from ``__del__``; call it explicitly (or via
        ``contextlib.closing``) when you want the tens of GB back
        immediately. After ``close()`` the dataset can no longer be
        indexed.
        """
        for attr, _fname in _ARRAY_FILES:
            arr = getattr(self, attr, None)
            if arr is None:
                continue
            setattr(self, attr, None)
            _close_memmap(arr)
            del arr
        owned = getattr(self, "_owned_dir", None)
        if owned:
            self._owned_dir = None
            _OWNED_DIRS.discard(owned)
            gc.collect()
            _remove_store_dir(owned)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001  # pragma: no cover
            pass

    def __getstate__(self) -> dict:
        # Pickle a minimal state — just the memmap directory path
        # and the row count. The actual data is on disk; workers
        # re-open the memmaps in ``__setstate__``.
        return {
            "_memmap_dir": self._memmap_dir,
            "_n_total": self._n_total,
        }

    def __setstate__(self, state: dict) -> None:
        self._memmap_dir = state["_memmap_dir"]
        # Workers do NOT own the temp directory — only the parent
        # process cleans it up. Setting ``_owned_dir`` to None makes
        # this explicit and prevents ``__del__`` from trying to
        # remove files still in use by other processes. Workers DO
        # still close their own memmap handles in ``close()``, which
        # is what lets the parent's cleanup succeed after the workers
        # shut down.
        self._owned_dir = None
        self._open_memmaps()
        # ``_n_total`` is also restored by ``_open_memmaps`` (from
        # row_offsets.shape), but assigning from state is a touch
        # cheaper and pins the contract.
        self._n_total = int(state["_n_total"])

    def __len__(self) -> int:
        return self._n_total

    @property
    def policy_width(self) -> int:
        """Width of the policy-target / action-mask rows (517 for a
        v1 / v2 window, 805 for canonical_v3)."""
        return int(self._policy_targets.shape[1])

    @property
    def policy_report(self) -> dict:
        """What the loader did with each file's policy columns
        (``_PolicyWidthPlan.report``): target encoding / width, files
        and rows zero-padded from the legacy 517 width, resolution
        policy rows found. ``{}`` for stores built before 2026-10-06 or
        by the in-memory constructors."""
        path = Path(self._memmap_dir) / POLICY_REPORT_NAME
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - corrupt sidecar
            return {}

    @property
    def scalar_width(self) -> int:
        """Width of every ``token_scalars`` row in this dataset —
        ``N_SCALAR_FIELDS`` (3) for scalar-schema-v1 data,
        ``N_SCALAR_FIELDS_V2`` for v2. ``train_structured_model``
        compares this against ``StructuredModelConfig.scalar_width``
        and refuses to train on a mismatch."""
        return int(self._token_scalars_flat.shape[1])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._n_total:
            raise IndexError(
                f"index {idx} out of range for dataset of length "
                f"{self._n_total}"
            )
        start = int(self._row_offsets[idx])
        end = int(self._row_offsets[idx + 1])
        seq_len = end - start
        # ``np.array(memmap_slice)`` materializes a writable copy of
        # the slice in the worker process — small (one row's worth
        # of tokens, ~50 entries on average). ``torch.from_numpy``
        # then wraps that copy as a tensor with zero further copy.
        # We can't pass a memmap directly to ``torch.from_numpy``
        # because that would produce a tensor pointing into a
        # read-only mapping, which collate_fn's ``in-place`` writes
        # would fail on.
        # The two id columns are stored narrow (int16 / int32) and
        # widened back to int64 here: ``.astype`` IS the materializing
        # copy (it does not add one), so the row still costs exactly
        # one copy per array and the model's embedding layers keep
        # getting ``LongTensor``s.
        tt = torch.from_numpy(
            self._token_type_flat[start:end].astype(np.int64)
        )
        emb = torch.from_numpy(
            self._token_emb_ids_flat[start:end].astype(np.int64)
        )
        sca = torch.from_numpy(np.array(self._token_scalars_flat[start:end]))
        am = torch.ones(seq_len, dtype=torch.bool)
        outcome = torch.tensor(
            float(self._outcomes[idx]), dtype=torch.float32,
        )
        root_value = torch.tensor(
            float(self._root_values[idx]), dtype=torch.float32,
        )
        policy_target = torch.from_numpy(
            np.array(self._policy_targets[idx]),
        )
        action_mask = torch.from_numpy(
            np.array(self._action_masks[idx]),
        )
        has_policy = torch.tensor(
            bool(self._has_policy[idx]), dtype=torch.bool,
        )
        return {
            "token_type": tt,
            "token_emb_ids": emb,
            "token_scalars": sca,
            "attention_mask": am,
            "outcome": outcome,
            "root_value": root_value,
            "policy_target": policy_target,
            "action_mask": action_mask,
            "has_policy": has_policy,
        }


def _normalize_scalar_parts(
    sca_parts: list[np.ndarray],
    labels: list[str],
) -> list[np.ndarray]:
    """Make a list of flat ``(n_i, width_i)`` scalar arrays safe to
    ``np.concatenate``.

    All parts that carry real tokens must share one ``token_scalars``
    width (= one scalar schema). Mixing widths means the npz files
    were produced under different ``scalar_schema_version``s — that
    is a data-provenance bug, so raise a loud, specific error naming
    the offending file rather than letting ``np.concatenate`` fail
    with "input array dimensions ... must match exactly".

    Zero-token parts (crash / timeout npzs, which ``data_gen`` writes
    with a placeholder width) can't be distinguished by content, so
    they are reshaped to the width the real parts use.
    """
    width: Optional[int] = None
    first_label = ""
    for part, label in zip(sca_parts, labels):
        if part.shape[0] == 0:
            continue
        if width is None:
            width = int(part.shape[1])
            first_label = label
        elif int(part.shape[1]) != width:
            raise ValueError(
                f"scalar schema mismatch across structured-token files: "
                f"{label} has token_scalars width {part.shape[1]} but "
                f"{first_label} has {width}. All files in one training "
                f"run must be generated under the same "
                f"scalar_schema_version — regenerate or quarantine the "
                f"odd files."
            )
    if width is None:
        return sca_parts
    return [
        part if part.shape[0] else np.zeros((0, width), dtype=part.dtype)
        for part in sca_parts
    ]


def _flatten_one_file(
    token_type: np.ndarray,
    token_emb_ids: np.ndarray,
    token_scalars: np.ndarray,
    attention_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten one file's padded token arrays into packed real-token
    arrays + a row-length vector.

    Inputs are file-padded:
    - ``token_type`` shape ``(n_rows, file_max_len)``
    - ``token_emb_ids`` shape ``(n_rows, file_max_len, emb_width)``
    - ``token_scalars`` shape ``(n_rows, file_max_len, scalar_width)``
    - ``attention_mask`` shape ``(n_rows, file_max_len)`` (bool)

    Outputs are packed:
    - ``token_type_flat`` shape ``(total_real,)``
    - ``token_emb_ids_flat`` shape ``(total_real, emb_width)``
    - ``token_scalars_flat`` shape ``(total_real, scalar_width)``
    - ``row_lengths`` shape ``(n_rows,)``

    Uses numpy boolean-indexing (``arr[mask]``) for the gather, which
    is the vectorized way to extract real tokens from a 2-D padded
    layout. C-order memory layout guarantees the output rows appear
    in row-major (i.e., file-row) order.
    """
    if attention_mask.dtype != np.bool_:
        attention_mask = attention_mask.astype(np.bool_)
    # ``token_type[mask]`` produces a 1-D array of real tokens in
    # row-major order (numpy boolean-indexing on a 2-D array).
    tt_flat = token_type[attention_mask]
    # For the 3-D arrays, numpy broadcasts the mask across the last
    # axis: ``emb_ids[mask]`` returns shape ``(total_real, emb_width)``.
    emb_flat = token_emb_ids[attention_mask]
    sca_flat = token_scalars[attention_mask]
    # Per-row real length = mask sum along axis 1.
    row_lengths = attention_mask.sum(axis=1).astype(np.int64)
    return tt_flat, emb_flat, sca_flat, row_lengths


def _flatten_file_padded(
    token_type: np.ndarray,
    token_emb_ids: np.ndarray,
    token_scalars: np.ndarray,
    attention_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-file flatten helper that also computes ``row_offsets``.

    Returned tuple is
    ``(token_type_flat, token_emb_ids_flat, token_scalars_flat,
    row_offsets)`` — same shapes as the dataset's internal storage.
    """
    tt_flat, emb_flat, sca_flat, row_lengths = _flatten_one_file(
        token_type, token_emb_ids, token_scalars, attention_mask,
    )
    row_offsets = np.concatenate(
        ([0], np.cumsum(row_lengths)),
    ).astype(np.int64)
    return tt_flat, emb_flat, sca_flat, row_offsets


def structured_collate(
    items: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad variable-length per-row tensors to the batch's max
    ``seq_len`` and stack into the dict :meth:`StructuredModel.forward`
    expects.

    The shape contract on the returned dict:

    - ``token_type``      ``LongTensor(B, max_len)``
    - ``token_emb_ids``   ``LongTensor(B, max_len, N_EMB_FIELDS)``
    - ``token_scalars``   ``FloatTensor(B, max_len, N_SCALAR_FIELDS)``
    - ``attention_mask``  ``BoolTensor(B, max_len)``
    - ``outcome``         ``FloatTensor(B,)``
    - ``policy_target``   ``FloatTensor(B, action_vocab_size)``
    - ``action_mask``     ``BoolTensor(B, action_vocab_size)``
    - ``has_policy``      ``BoolTensor(B,)``

    Empty rows (seq_len=0) are tolerated: the row's mask comes back
    all-False. ``StructuredModel`` guards against fully-padded rows
    inside ``forward`` because attention with no valid keys raises NaN
    on some PyTorch versions; the data path filters those rows before
    we get here (rows with no decisions don't produce a sample), so
    the smoke check is "no row in production has seq_len=0" rather
    than a runtime guard inside the collate.
    """
    if not items:
        raise ValueError("structured_collate received an empty batch")
    batch_size = len(items)
    seq_lens = [int(it["token_type"].shape[0]) for it in items]
    max_len = max(seq_lens) if seq_lens else 0
    if max_len == 0:
        max_len = 1
    emb_width = items[0]["token_emb_ids"].shape[-1]
    scalar_width = items[0]["token_scalars"].shape[-1]
    action_vocab = items[0]["policy_target"].shape[-1]

    token_type = torch.zeros((batch_size, max_len), dtype=torch.int64)
    token_emb_ids = torch.zeros(
        (batch_size, max_len, emb_width), dtype=torch.int64,
    )
    token_scalars = torch.zeros(
        (batch_size, max_len, scalar_width), dtype=torch.float32,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    outcome = torch.zeros((batch_size,), dtype=torch.float32)
    policy_target = torch.zeros(
        (batch_size, action_vocab), dtype=torch.float32,
    )
    action_mask = torch.zeros(
        (batch_size, action_vocab), dtype=torch.bool,
    )
    has_policy = torch.zeros((batch_size,), dtype=torch.bool)
    root_value = torch.full((batch_size,), float("nan"), dtype=torch.float32)
    for i, it in enumerate(items):
        n = int(it["token_type"].shape[0])
        if n > 0:
            token_type[i, :n] = it["token_type"]
            token_emb_ids[i, :n, :] = it["token_emb_ids"]
            token_scalars[i, :n, :] = it["token_scalars"]
            attention_mask[i, :n] = True
        outcome[i] = it["outcome"]
        rv = it.get("root_value")
        if rv is not None:
            root_value[i] = rv
        policy_target[i] = it["policy_target"]
        action_mask[i] = it["action_mask"]
        has_policy[i] = it["has_policy"]
    return {
        "token_type": token_type,
        "token_emb_ids": token_emb_ids,
        "token_scalars": token_scalars,
        "attention_mask": attention_mask,
        "outcome": outcome,
        "root_value": root_value,
        "policy_target": policy_target,
        "action_mask": action_mask,
        "has_policy": has_policy,
    }


# ---------------------------------------------------------------------
# Incremental flat-store construction
# ---------------------------------------------------------------------


def _npy_member_names(zf: zipfile.ZipFile) -> dict[str, str]:
    """Map npz array name -> zip member name (``<name>.npy``)."""
    out: dict[str, str] = {}
    for name in zf.namelist():
        out[name[:-4] if name.endswith(".npy") else name] = name
    return out


def _read_member_header(
    zf: zipfile.ZipFile, member: str,
) -> tuple[tuple[int, ...], np.dtype]:
    """Read one npz member's ``.npy`` header without decompressing the
    array body.

    Deflate is a stream, so reading the ~100-byte header only inflates
    the first block — this is what makes the sizing pass cheap enough
    to run over thousands of files.
    """
    with zf.open(member) as fp:
        version = np.lib.format.read_magic(fp)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fp)
        elif version == (2, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fp)
        else:  # pragma: no cover - numpy never writes other versions
            raise ValueError(
                f"unsupported .npy version {version} in member {member}"
            )
    return tuple(int(d) for d in shape), dtype


def _read_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Decompress exactly one npz member into a fresh array."""
    with zf.open(member) as fp:
        return np.lib.format.read_array(fp, allow_pickle=False)


CHUNK_BYTES = 64 * 1024 * 1024
"""Target size of one decoded row-chunk. Bounds peak RSS *within* a
single npz: the pool contains legacy monolithic files (one 747k-row
npz in ``iterate_data_s2`` decodes to ~20 GB of padded arrays), so
"one file at a time" is not by itself a small enough unit."""


def _read_exact(fp, n: int) -> bytes:
    """``fp.read(n)`` that tolerates short reads (ZipExtFile)."""
    parts = []
    remaining = n
    while remaining > 0:
        block = fp.read(remaining)
        if not block:
            raise ValueError(
                f"unexpected EOF in npz member: wanted {n} bytes, "
                f"short by {remaining}"
            )
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts) if len(parts) > 1 else parts[0]


def _iter_member_rows(
    zf: zipfile.ZipFile, member: str, *, chunk_bytes: Optional[int] = None,
):
    """Yield ``(row_offset, chunk)`` for one npz member, row-major.

    The whole point of the streaming loader is never to materialize a
    big array; ``np.lib.format.read_array`` would do exactly that. The
    npy body is a C-contiguous block, so a row range is a byte range:
    we read the header, then inflate ``chunk_bytes`` worth of rows at a
    time straight off the deflate stream.

    ``chunk_bytes`` resolves to the module constant at call time (not
    as a default argument) so tests can shrink it with monkeypatch.
    """
    if chunk_bytes is None:
        chunk_bytes = CHUNK_BYTES
    with zf.open(member) as fp:
        version = np.lib.format.read_magic(fp)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
        else:  # pragma: no cover - numpy never writes other versions
            raise ValueError(
                f"unsupported .npy version {version} in member {member}"
            )
        if fortran:  # pragma: no cover - savez never writes F-order
            raise ValueError(
                f"{member}: Fortran-ordered arrays are not supported"
            )
        if dtype.hasobject:  # pragma: no cover - would need pickle
            raise ValueError(f"{member}: object arrays are not supported")
        shape = tuple(int(d) for d in shape)
        n_rows = shape[0] if shape else 0
        row_items = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        row_bytes = row_items * dtype.itemsize
        if n_rows == 0 or row_bytes == 0:
            return
        chunk_rows = max(1, chunk_bytes // row_bytes)
        offset = 0
        while offset < n_rows:
            take = min(chunk_rows, n_rows - offset)
            buf = _read_exact(fp, take * row_bytes)
            yield offset, np.frombuffer(buf, dtype=dtype).reshape(
                (take,) + shape[1:]
            )
            offset += take


@dataclass
class _FileScan:
    """Sizing metadata for one source npz, from the cheap first pass."""

    path: Path
    n_rows: int
    row_lengths: np.ndarray  # (n_rows,) int64
    n_tokens: int
    emb_width: int
    scalar_width: int
    policy_width: Optional[int]  # None = file carries no policy arrays
    # Action-encoding version the policy columns were written under
    # (``data_gen.resolution_stats_arrays``); None = legacy unstamped.
    encoding_version: Optional[str] = None


POLICY_REPORT_NAME = "policy_report.json"
"""Flat-store sidecar recording the policy-width plan's report."""

LEGACY_POLICY_ENCODING_ENV = "MTG_BOT_LEGACY_POLICY_ENCODING"
"""Environment twin of ``mtg_bot train --legacy-policy-encoding``."""


class PolicyWidthError(ValueError):
    """The window's policy columns can't be lined up with the model's
    action encoding (unknown width, a stamped version that differs, or
    legacy 517-wide shards in a canonical_v3 window without the
    ``--legacy-policy-encoding`` assertion)."""


@dataclass
class _PolicyWidthPlan:
    """How each file's policy columns map into the store."""

    width: int
    pad_paths: set
    report: dict


def plan_policy_width(
    scans: Sequence["_FileScan"],
    *,
    action_encoding_version: str,
    legacy_policy_encoding: Optional[str] = None,
) -> _PolicyWidthPlan:
    """Resolve the store's policy width for a model trained under
    ``action_encoding_version`` and decide, per file, copy vs explicit
    zero-pad (``docs/RESOLUTION_DECISIONS.md`` §8).

    * Files carrying no policy arrays, and zero-row files (crashed /
      timed-out games), never constrain the width.
    * A STAMPED file must be exactly its version's width; it is copied
      when its version is the model's, zero-padded when it is
      ``canonical_v2`` and the model ``canonical_v3`` (v3's first 517
      slots ARE v2's), refused otherwise.
    * An UNSTAMPED (legacy) file: copied when its width already is the
      model's (the pre-2026-10-06 behaviour); a 517-wide legacy file in
      a ``canonical_v3`` window is zero-padded ONLY when the operator
      asserts ``legacy_policy_encoding="canonical_v2"`` -- width 517
      cannot tell ``positional_v1`` from ``canonical_v2``, and padding a
      v1 file would mislabel every slot. Anything else is refused.

    The padded columns are the resolution block: those games had no
    resolution supervision (they were played with auto-targeting), so
    an all-zero target / all-False mask there is the fact, not a guess.
    """
    target = action_vocab_size(action_encoding_version)
    pad_paths: set = set()
    counts = {
        "files": 0, "files_without_policy": 0, "zero_row_files": 0,
        "stamped_files": 0, "legacy_files": 0,
        "copied_files": 0, "padded_files": 0, "padded_rows": 0,
    }
    for scan in scans:
        counts["files"] += 1
        if scan.policy_width is None:
            counts["files_without_policy"] += 1
            continue
        if scan.n_rows == 0:
            counts["zero_row_files"] += 1
            continue
        stamped = scan.encoding_version
        if stamped is not None:
            counts["stamped_files"] += 1
            expected = action_vocab_size(stamped)
            if scan.policy_width != expected:
                raise PolicyWidthError(
                    f"{scan.path}: policy width {scan.policy_width} but "
                    f"it is stamped {stamped!r} ({expected} slots)"
                )
            if stamped == action_encoding_version:
                counts["copied_files"] += 1
                continue
            if (
                stamped == ACTION_ENCODING_VERSION_V2
                and action_encoding_version == ACTION_ENCODING_VERSION_V3
            ):
                pad_paths.add(scan.path)
                counts["padded_files"] += 1
                counts["padded_rows"] += scan.n_rows
                continue
            raise PolicyWidthError(
                f"{scan.path}: stamped action encoding {stamped!r} can't "
                f"train a {action_encoding_version!r} model"
            )
        counts["legacy_files"] += 1
        if scan.policy_width == target:
            counts["copied_files"] += 1
            continue
        if (
            scan.policy_width == ACTION_VOCAB_SIZE
            and action_encoding_version == ACTION_ENCODING_VERSION_V3
        ):
            if legacy_policy_encoding != ACTION_ENCODING_VERSION_V2:
                raise PolicyWidthError(
                    f"{scan.path}: legacy (unstamped) {ACTION_VOCAB_SIZE}-"
                    f"wide policy columns in a {action_encoding_version} "
                    f"window. Width {ACTION_VOCAB_SIZE} can't say whether "
                    f"they are positional_v1 or canonical_v2; if they are "
                    f"canonical_v2 (every s2-lineage shard is), assert it "
                    f"with --legacy-policy-encoding canonical_v2 (or "
                    f"${LEGACY_POLICY_ENCODING_ENV}=canonical_v2) and they "
                    f"are zero-padded into the resolution block."
                )
            pad_paths.add(scan.path)
            counts["padded_files"] += 1
            counts["padded_rows"] += scan.n_rows
            continue
        raise PolicyWidthError(
            f"{scan.path}: policy width {scan.policy_width} matches "
            f"neither the {action_encoding_version!r} width {target} nor "
            f"a paddable legacy width"
        )
    report = {
        "action_encoding_version": action_encoding_version,
        "policy_width": target,
        "legacy_policy_encoding": legacy_policy_encoding,
        **counts,
    }
    return _PolicyWidthPlan(width=target, pad_paths=pad_paths, report=report)


def _require_structured(path: Path, members: dict[str, str]) -> None:
    if any(key not in members for key in _STRUCT_NPZ_KEYS):
        raise ValueError(
            f"{path}: missing structured-token arrays. The "
            f"structured trainer requires data_gen to be run "
            f"with structured_tokens=True."
        )


def _scan_structured_npz(path: Path) -> _FileScan:
    """First pass: shapes + per-row real-token counts for one npz.

    Reads the ``.npy`` headers of the big arrays (no body inflation)
    and decompresses only ``structured_attention_mask`` — a bool array
    that is ~1% of the file's compressed bytes — to derive exact row
    lengths. That is everything the flat store needs to be
    pre-allocated at its final size.
    """
    with zipfile.ZipFile(path) as zf:
        members = _npy_member_names(zf)
        _require_structured(path, members)
        tt_shape, _ = _read_member_header(zf, members["structured_token_type"])
        emb_shape, _ = _read_member_header(
            zf, members["structured_token_emb_ids"],
        )
        sca_shape, _ = _read_member_header(
            zf, members["structured_token_scalars"],
        )
        mask = _read_member(zf, members["structured_attention_mask"])
        policy_width: Optional[int] = None
        if "policy_targets" in members:
            p_shape, _ = _read_member_header(zf, members["policy_targets"])
            policy_width = int(p_shape[1]) if len(p_shape) > 1 else 1
        encoding_version: Optional[str] = None
        if "action_encoding_version" in members:
            encoding_version = str(
                _read_member(zf, members["action_encoding_version"]).item()
            )
    n_rows = int(tt_shape[0])
    if mask.shape[0] != n_rows:
        raise ValueError(
            f"{path}: structured_attention_mask rows {mask.shape[0]} != "
            f"structured_token_type rows {n_rows}"
        )
    if emb_shape[0] != n_rows or sca_shape[0] != n_rows:
        raise ValueError(
            f"{path}: structured token arrays disagree on row count: "
            f"token_type {n_rows} / token_emb_ids {emb_shape[0]} / "
            f"token_scalars {sca_shape[0]}"
        )
    mask_b = mask if mask.dtype == np.bool_ else mask.astype(np.bool_)
    if n_rows:
        row_lengths = (
            mask_b.reshape(n_rows, -1).sum(axis=1).astype(np.int64)
        )
    else:
        row_lengths = np.zeros((0,), dtype=np.int64)
    return _FileScan(
        path=path,
        n_rows=n_rows,
        row_lengths=row_lengths,
        n_tokens=int(row_lengths.sum()),
        emb_width=int(emb_shape[2]) if len(emb_shape) > 2 else 0,
        scalar_width=int(sca_shape[2]) if len(sca_shape) > 2 else 0,
        policy_width=policy_width,
        encoding_version=encoding_version,
    )


@dataclass
class _StorePlan:
    """Global shapes of the flat store, derived from the scans."""

    n_rows: int
    n_tokens: int
    emb_width: int
    scalar_width: int
    policy_width: int
    row_offsets: np.ndarray


def _plan_store(
    scans: Sequence[_FileScan],
    *,
    policy_plan: Optional[_PolicyWidthPlan] = None,
) -> _StorePlan:
    """Resolve the global widths + row offsets from the per-file scans.

    Width conflicts raise here with the same message the old
    concatenate-everything path produced (see
    :func:`_normalize_scalar_parts`): mixing scalar schemas is a data
    provenance bug, not something to paper over.
    """
    scalar_width: Optional[int] = None
    scalar_label = ""
    emb_width: Optional[int] = None
    emb_label = ""
    policy_width: Optional[int] = None
    for scan in scans:
        if scan.n_tokens == 0:
            # Zero-real-token files (crash / timeout npzs) carry a
            # placeholder width — they contribute no flat rows, so they
            # can't constrain the schema.
            continue
        if scalar_width is None:
            scalar_width = scan.scalar_width
            scalar_label = str(scan.path)
        elif scan.scalar_width != scalar_width:
            raise ValueError(
                f"scalar schema mismatch across structured-token files: "
                f"{scan.path} has token_scalars width {scan.scalar_width} "
                f"but {scalar_label} has {scalar_width}. All files in one "
                f"training run must be generated under the same "
                f"scalar_schema_version — regenerate or quarantine the "
                f"odd files."
            )
        if emb_width is None:
            emb_width = scan.emb_width
            emb_label = str(scan.path)
        elif scan.emb_width != emb_width:
            raise ValueError(
                f"token_emb_ids width mismatch across structured-token "
                f"files: {scan.path} has width {scan.emb_width} but "
                f"{emb_label} has {emb_width}."
            )
    if scalar_width is None:
        # Every file is empty of real tokens; fall back to the first
        # file's declared widths so the store still has a schema.
        scalar_width = scans[0].scalar_width if scans else 0
        emb_width = scans[0].emb_width if scans else 0
    policy_label = ""
    for scan in scans:
        if policy_plan is not None:
            # ``plan_policy_width`` already validated every file
            # against the model's encoding.
            break
        if scan.policy_width is None:
            continue
        if policy_width is None:
            policy_width = scan.policy_width
            policy_label = str(scan.path)
        elif scan.policy_width != policy_width:
            raise ValueError(
                f"policy_targets width mismatch across structured-token "
                f"files: {scan.path} has width {scan.policy_width} but "
                f"{policy_label} has {policy_width}."
            )
    if policy_plan is not None:
        policy_width = policy_plan.width
    elif policy_width is None:
        policy_width = ACTION_VOCAB_SIZE
    elif (
        policy_width != ACTION_VOCAB_SIZE
        and any(s.policy_width is None for s in scans)
    ):
        raise ValueError(
            f"policy_targets width {policy_width} ({policy_label}) does "
            f"not match ACTION_VOCAB_SIZE={ACTION_VOCAB_SIZE}, but other "
            f"files carry no policy supervision and would be zero-filled "
            f"at the vocab width. Regenerate the odd files."
        )
    row_lengths = (
        np.concatenate([s.row_lengths for s in scans])
        if scans else np.zeros((0,), dtype=np.int64)
    )
    row_offsets = np.concatenate(
        ([0], np.cumsum(row_lengths)),
    ).astype(np.int64)
    return _StorePlan(
        n_rows=int(row_lengths.shape[0]),
        n_tokens=int(row_offsets[-1]) if row_offsets.size else 0,
        emb_width=int(emb_width or 0),
        scalar_width=int(scalar_width or 0),
        policy_width=int(policy_width),
        row_offsets=row_offsets,
    )


_NARROWED_TOKEN_COLUMNS: dict[str, str] = {
    "token_type": "token_type_flat",
    "token_emb_ids": "token_emb_ids_flat",
}
"""Builder attribute -> store column name, for the columns whose
on-disk dtype is narrower than the npz's int64 and therefore need a
range check on every write."""


class _FlatStoreBuilder:
    """Pre-allocate the flat ``.npy`` files and stream rows into them.

    Every array is created at its final size with
    ``np.lib.format.open_memmap`` and filled slice-by-slice, so peak
    RSS is O(one source file) no matter how many files the pool has.
    """

    def __init__(self, directory: Path, plan: _StorePlan) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Identify the builder, so a build abandoned by a killed
        # process is reclaimable by pid liveness rather than by age
        # (this covers the persistent cache's ``.building-*`` dirs,
        # which live outside the swept temp root).
        _write_owner_marker(self.directory)
        self.plan = plan
        self._arrays: dict[str, np.memmap] = {}
        alloc = self._alloc
        n_rows, n_tok = plan.n_rows, plan.n_tokens
        self.token_type = alloc(
            "token_type_flat.npy", TOKEN_TYPE_DTYPE, (n_tok,),
        )
        self.token_emb_ids = alloc(
            "token_emb_ids_flat.npy", TOKEN_EMB_IDS_DTYPE,
            (n_tok, plan.emb_width),
        )
        self.token_scalars = alloc(
            "token_scalars_flat.npy", TOKEN_SCALARS_DTYPE,
            (n_tok, plan.scalar_width),
        )
        self.outcomes = alloc("outcomes.npy", np.float32, (n_rows,))
        # NaN = "no search root value for this row" (legacy shards,
        # forced moves). Files that carry the column overwrite their
        # rows in ``_stream_file_into_store``.
        self.root_values = alloc("root_values.npy", np.float32, (n_rows,))
        self.root_values[:] = np.nan
        self.policy_targets = alloc(
            "policy_targets.npy", np.float32, (n_rows, plan.policy_width),
        )
        self.action_masks = alloc(
            "action_masks.npy", np.bool_, (n_rows, plan.policy_width),
        )
        self.has_policy = alloc("has_policy.npy", np.bool_, (n_rows,))
        self.group_ids = alloc("group_ids.npy", np.int64, (n_rows,))
        # Rows whose policy TARGET has mass in the canonical_v3
        # resolution block (counted while streaming; reported).
        self.resolution_target_rows = 0
        # ``row_offsets`` is fully known from the sizing pass.
        row_offsets = alloc("row_offsets.npy", np.int64, (n_rows + 1,))
        row_offsets[:] = plan.row_offsets

    def _alloc(self, name: str, dtype, shape) -> np.memmap:
        arr = np.lib.format.open_memmap(
            self.directory / name, mode="w+", dtype=dtype, shape=shape,
        )
        self._arrays[name] = arr
        return arr

    def write_tokens(
        self,
        dest_name: str,
        token_start: int,
        chunk_offset: int,
        flat_chunk: np.ndarray,
        *,
        source: object = None,
    ) -> None:
        """Write one chunk's packed real tokens at ``token_start +
        chunk_offset``. A zero-token chunk is skipped: files with no
        real tokens (crash / timeout npzs) carry a placeholder
        emb/scalar width that must not be broadcast against the store's
        schema width — the streaming equivalent of
        :func:`_normalize_scalar_parts`.

        The two id columns are stored narrow, and a plain numpy
        assignment into a narrower dest truncates silently, so each
        chunk is range-checked first (see :func:`_checked_narrow`).
        """
        n = int(flat_chunk.shape[0])
        if not n:
            return
        dest = getattr(self, dest_name)
        if dest_name in _NARROWED_TOKEN_COLUMNS:
            _check_fits_dtype(
                _NARROWED_TOKEN_COLUMNS[dest_name],
                flat_chunk,
                dest.dtype,
                source=source,
            )
        start = token_start + chunk_offset
        dest[start:start + n] = flat_chunk

    def write_rows(
        self,
        dest_name: str,
        row_start: int,
        chunk_offset: int,
        chunk: np.ndarray,
    ) -> None:
        n = int(chunk.shape[0])
        if not n:
            return
        dest = getattr(self, dest_name)
        start = row_start + chunk_offset
        dest[start:start + n] = chunk

    def write_policy_rows(
        self, row_start: int, chunk_offset: int, chunk: np.ndarray,
    ) -> None:
        """Policy targets + the derived ``has_policy`` flag.

        ``has_policy`` uses the same per-row formula the old
        whole-array precompute did; summing per chunk is bit-identical
        because the sum is per row.
        """
        n = int(chunk.shape[0])
        if not n:
            return
        start = row_start + chunk_offset
        # A legacy 517-wide chunk in an 805-wide store fills the
        # leading (canonical_v2) columns; the resolution block keeps
        # the store's zero-fill (explicit pad, ``plan_policy_width``).
        self.policy_targets[start:start + n, :chunk.shape[1]] = chunk
        self.has_policy[start:start + n] = (
            np.ascontiguousarray(chunk, dtype=np.float32).sum(axis=1) > 0.5
        )
        if chunk.shape[1] > SLOT_RESOLUTION_BASE:
            res_mass = np.asarray(chunk)[:, SLOT_RESOLUTION_BASE:].sum(axis=1)
            self.resolution_target_rows += int((res_mass > 0.5).sum())

    def write_mask_rows(
        self, row_start: int, chunk_offset: int, chunk: np.ndarray,
    ) -> None:
        """Action masks; a narrower (legacy) chunk fills the leading
        columns like ``write_policy_rows``."""
        n = int(chunk.shape[0])
        if not n:
            return
        start = row_start + chunk_offset
        self.action_masks[start:start + n, :chunk.shape[1]] = chunk

    def close(self) -> None:
        """Flush + unmap every array. Must run before the store dir is
        opened read-only (or deleted) on Windows."""
        for name in list(self._arrays):
            arr = self._arrays.pop(name)
            try:
                arr.flush()
            except (ValueError, OSError):  # pragma: no cover
                pass
            _close_memmap(arr)
            del arr
        for attr in (
            "token_type", "token_emb_ids", "token_scalars", "outcomes",
            "root_values",
            "policy_targets", "action_masks", "has_policy", "group_ids",
        ):
            setattr(self, attr, None)
        gc.collect()


def _stream_file_into_store(
    builder: _FlatStoreBuilder,
    scan: _FileScan,
    file_index: int,
    row_start: int,
    token_start: int,
) -> int:
    """Second pass for one npz: decode in row-chunks, flatten, write.

    Returns the number of distinct game groups the file contributed.

    Only the arrays the dataset actually needs are decompressed —
    ``features`` (unused by the structured trainer) is skipped
    entirely, which is a straight I/O + CPU saving over
    ``data_gen.load_dataset``. Each token array is inflated
    ``CHUNK_BYTES`` at a time and written straight into its slice of
    the store, so peak RSS is bounded by the chunk size even for the
    legacy monolithic npzs (one file in ``iterate_data_s2`` holds 747k
    rows / ~20 GB of padded arrays).
    """
    path = scan.path
    n_rows = scan.n_rows
    # Per-row token offsets *within this file*: chunk k's first token
    # lands at ``token_start + local_offsets[first_row_of_chunk]``.
    local_offsets = np.concatenate(
        ([0], np.cumsum(scan.row_lengths)),
    ).astype(np.int64)
    with zipfile.ZipFile(path) as zf:
        members = _npy_member_names(zf)
        _require_structured(path, members)
        mask = _read_member(zf, members["structured_attention_mask"])
        if mask.dtype != np.bool_:
            mask = mask.astype(np.bool_)
        row_lengths = (
            mask.reshape(n_rows, -1).sum(axis=1).astype(np.int64)
            if n_rows else np.zeros((0,), dtype=np.int64)
        )
        if not np.array_equal(row_lengths, scan.row_lengths):
            raise ValueError(
                f"{path} changed between the sizing pass and the read "
                f"pass (row lengths differ). Re-run once the data "
                f"generator has finished writing."
            )
        for dest_name, key in (
            ("token_type", "structured_token_type"),
            ("token_emb_ids", "structured_token_emb_ids"),
            ("token_scalars", "structured_token_scalars"),
        ):
            for offset, chunk in _iter_member_rows(zf, members[key]):
                builder.write_tokens(
                    dest_name,
                    token_start,
                    int(local_offsets[offset]),
                    chunk[mask[offset:offset + chunk.shape[0]]],
                    source=path,
                )
        del mask
        outcomes_rows = 0
        for offset, chunk in _iter_member_rows(zf, members["outcomes"]):
            builder.write_rows("outcomes", row_start, offset, chunk)
            outcomes_rows += int(chunk.shape[0])
        if outcomes_rows != n_rows:
            raise ValueError(
                f"{path}: outcomes rows {outcomes_rows} != structured "
                f"token rows {n_rows}"
            )
        if "root_values" in members:
            rv_rows = 0
            for offset, chunk in _iter_member_rows(
                zf, members["root_values"],
            ):
                builder.write_rows(
                    "root_values", row_start, offset,
                    np.ascontiguousarray(chunk, dtype=np.float32),
                )
                rv_rows += int(chunk.shape[0])
            if rv_rows != n_rows:
                raise ValueError(
                    f"{path}: root_values rows {rv_rows} != structured "
                    f"token rows {n_rows}"
                )
        # A file with no policy supervision leaves its rows at the
        # store's zero-fill: ``open_memmap`` allocates a fresh
        # zero-initialized file, which is exactly the all-zero
        # policy_targets / all-False action_masks the old global
        # zero-fill produced (and ``has_policy`` False with it).
        if "policy_targets" in members:
            for offset, chunk in _iter_member_rows(
                zf, members["policy_targets"],
            ):
                builder.write_policy_rows(row_start, offset, chunk)
        if "action_masks" in members:
            for offset, chunk in _iter_member_rows(
                zf, members["action_masks"],
            ):
                builder.write_mask_rows(row_start, offset, chunk)
        game_ids = (
            _read_member(zf, members["game_ids"])
            if "game_ids" in members else None
        )
    # Game-level group ids: ``file_index * 1e6 + game_id`` when the
    # file carries ``game_ids``; whole file as one group for legacy
    # files (logged at INFO inside ``file_group_ids``).
    group_ids = file_group_ids(game_ids, n_rows, file_index, path=path)
    builder.write_rows("group_ids", row_start, 0, group_ids)
    # Group ids never cross file boundaries (they are offset by
    # ``file_index * 1e6``), so per-file unique counts sum to the
    # global count — no second pass over the whole array needed.
    return int(np.unique(group_ids).shape[0]) if n_rows else 0


def _select_paths(
    directory: Path, max_files: Optional[int],
) -> list[Path]:
    all_paths = list(directory.glob("*.npz"))
    if max_files is not None and all_paths:
        all_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        kept = all_paths[:max_files]
        return sorted(kept)
    return sorted(all_paths)


def _store_signature(
    paths: Sequence[Path], *, policy_key: str = "",
) -> str:
    """Cache key: layout version + every source file's path, size and
    mtime (+ the policy-width plan's target encoding and legacy
    assertion). Any add / remove / rewrite changes the key."""
    h = hashlib.sha256()
    h.update(f"v{FLAT_STORE_VERSION}|vocab={ACTION_VOCAB_SIZE}\n".encode())
    if policy_key:
        h.update(f"policy={policy_key}\n".encode())
    for p in paths:
        st = p.stat()
        h.update(
            f"{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8")
        )
    return h.hexdigest()[:32]


def _store_is_complete(directory: Path, signature: str) -> bool:
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - corrupt manifest
        return False
    if manifest.get("signature") != signature:
        return False
    if int(manifest.get("version", -1)) != FLAT_STORE_VERSION:
        return False
    return all(
        (directory / fname).is_file() for _attr, fname in _ARRAY_FILES
    )


def load_structured_dataset(
    directory: Path,
    *,
    max_files: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    sweep_stale: bool = True,
    action_encoding_version: Optional[str] = None,
    legacy_policy_encoding: Optional[str] = None,
) -> StructuredDataset:
    """Load every ``*.npz`` in ``directory`` that contains the four
    structured-token arrays. Files missing those arrays raise.

    ``action_encoding_version`` (the model's) makes the policy columns
    follow ``plan_policy_width``: the store is that version's width,
    ``canonical_v2`` files are explicitly zero-padded into a
    ``canonical_v3`` store (legacy unstamped ones only under
    ``legacy_policy_encoding="canonical_v2"``), everything else that
    disagrees is refused (``PolicyWidthError``). The plan's report --
    padded files / rows and resolution-target rows -- is logged and
    exposed as ``dataset.policy_report``. ``None`` keeps the
    pre-2026-10-06 behaviour (one shared width, inferred).

    ``max_files`` (default ``None`` = all) caps to the N most recent
    files by mtime. Same semantics as
    :func:`data_gen.load_all_datasets`.

    Each file's padded tokens are flattened into a packed real-token
    layout at load time — the returned dataset's ``__getitem__`` is a
    single contiguous slice per array, no per-row ``.astype`` copies,
    no ``np.searchsorted`` lookup, no per-row ``mask.sum()``
    recomputation. See module docstring for the storage layout and
    the history note on why this matters.

    **Memory.** The load is two-pass and streaming: pass 1 reads each
    npz's array headers plus its (tiny, bool) attention mask to learn
    the exact final shapes; the flat ``.npy`` files are then
    pre-allocated at full size with ``np.lib.format.open_memmap`` and
    pass 2 writes each file's rows straight into its slice and drops
    the decoded arrays. Peak RSS is O(largest single npz + row
    metadata) — independent of the number of files. The old path
    decoded every file into RAM, concatenated (doubling), then wrote a
    third copy out, peaking at ~24 GB per 1M rows.

    ``cache_dir`` (or ``$MTG_BOT_STRUCTURED_CACHE_DIR``) enables a
    persistent flat store keyed by the exact file list + sizes +
    mtimes: an unchanged pool skips the rebuild entirely. Cached
    stores are NOT auto-deleted (the caller owns the directory).

    ``sweep_stale`` (default True) removes leftover flat stores from
    crashed runs; see :func:`sweep_stale_memmap_dirs`.
    """
    if max_files is not None and max_files <= 0:
        raise ValueError(
            f"max_files must be positive or None; got {max_files}"
        )
    if sweep_stale and _sweep_enabled():
        try:
            sweep_stale_memmap_dirs()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            _log.warning("structured dataset: temp sweep failed: %s", exc)
    paths = _select_paths(directory, max_files)
    if not paths:
        raise FileNotFoundError(
            f"No .npz files in {directory}"
        )

    if cache_dir is None:
        env_cache = os.environ.get(CACHE_DIR_ENV)
        cache_dir = Path(env_cache) if env_cache else None
    policy_key = (
        f"{action_encoding_version}|legacy={legacy_policy_encoding}"
        if action_encoding_version is not None else ""
    )
    signature = (
        _store_signature(paths, policy_key=policy_key)
        if cache_dir is not None else ""
    )
    if cache_dir is not None:
        cached = Path(cache_dir) / signature
        if _store_is_complete(cached, signature):
            _log.info(
                "structured dataset: reusing cached flat store %s "
                "(%d files unchanged)", cached, len(paths),
            )
            return StructuredDataset.from_memmap_dir(cached, own_dir=False)

    # ---- pass 1: sizes only (headers + bool masks) -----------------
    scans = [_scan_structured_npz(p) for p in paths]
    policy_plan = (
        plan_policy_width(
            scans,
            action_encoding_version=action_encoding_version,
            legacy_policy_encoding=legacy_policy_encoding,
        )
        if action_encoding_version is not None else None
    )
    plan = _plan_store(scans, policy_plan=policy_plan)

    # ---- allocate the flat store at its final size -----------------
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # A build abandoned by a killed process leaves a partial store
        # in the cache dir, which the temp sweep never sees.
        sweep_abandoned_build_dirs(Path(cache_dir))
        build_dir = Path(cache_dir) / f".building-{signature}-{os.getpid()}"
        _remove_store_dir(build_dir, attempts=1)
        owns = False
    else:
        build_dir = Path(_new_owned_dir())
        owns = True
    builder = _FlatStoreBuilder(build_dir, plan)

    # ---- pass 2: stream every file into its slice ------------------
    try:
        row_cursor = 0
        token_cursor = 0
        n_groups = 0
        for file_index, scan in enumerate(scans):
            n_groups += _stream_file_into_store(
                builder, scan, file_index, row_cursor, token_cursor,
            )
            row_cursor += scan.n_rows
            token_cursor += scan.n_tokens
        if policy_plan is not None:
            report = dict(policy_plan.report)
            report["resolution_target_rows"] = int(
                builder.resolution_target_rows
            )
            report["rows"] = int(plan.n_rows)
            (build_dir / POLICY_REPORT_NAME).write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _log.info(
                "structured dataset policy columns (%s, width %d): "
                "%d files copied, %d files / %d rows zero-padded from "
                "the legacy %d width (legacy encoding asserted: %s), "
                "%d rows with resolution-decision targets",
                report["action_encoding_version"], report["policy_width"],
                report["copied_files"], report["padded_files"],
                report["padded_rows"], ACTION_VOCAB_SIZE,
                report["legacy_policy_encoding"],
                report["resolution_target_rows"],
            )
        if row_cursor != plan.n_rows or token_cursor != plan.n_tokens:
            raise ValueError(  # pragma: no cover - defensive
                f"flat-store fill mismatch: wrote {row_cursor}/"
                f"{plan.n_rows} rows, {token_cursor}/{plan.n_tokens} "
                f"tokens"
            )
        builder.close()
    except BaseException:
        builder.close()
        _OWNED_DIRS.discard(str(build_dir))
        _remove_store_dir(build_dir)
        raise

    _log.info(
        "structured dataset: %d rows across %d files in %d game groups",
        plan.n_rows, len(paths), n_groups,
    )

    store_dir = build_dir
    if cache_dir is not None:
        (build_dir / _MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "version": FLAT_STORE_VERSION,
                    "signature": signature,
                    "n_rows": plan.n_rows,
                    "n_tokens": plan.n_tokens,
                    "files": [p.name for p in paths],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        final_dir = Path(cache_dir) / signature
        try:
            os.replace(build_dir, final_dir)
            store_dir = final_dir
        except OSError:
            # Another process published the same store first (or the
            # rename lost a race on Windows). Prefer theirs.
            if _store_is_complete(final_dir, signature):
                _remove_store_dir(build_dir)
                store_dir = final_dir
            else:  # pragma: no cover - rename failed for another reason
                store_dir = build_dir

    return StructuredDataset.from_memmap_dir(store_dir, own_dir=owns)


def shutdown_dataloader_workers(loader: object) -> None:
    """Stop a ``DataLoader``'s persistent workers *now*.

    ``persistent_workers=True`` keeps worker processes (and therefore
    their memmap handles on the flat store) alive until the loader is
    garbage-collected. On Windows those open handles block deletion of
    the store, so the trainer calls this before dropping the dataset.
    Safe to call on any object; a loader with no live iterator is a
    no-op.
    """
    it = getattr(loader, "_iterator", None)
    if it is None:
        return
    try:
        shutdown = getattr(it, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    except Exception:  # noqa: BLE001  # pragma: no cover
        pass
    try:
        loader._iterator = None  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001  # pragma: no cover
        pass


__all__ = [
    "FLAT_STORE_DTYPES",
    "FLAT_STORE_VERSION",
    "StructuredDataset",
    "load_structured_dataset",
    "structured_collate",
    "sweep_stale_memmap_dirs",
    "shutdown_dataloader_workers",
    "sweep_abandoned_build_dirs",
]
