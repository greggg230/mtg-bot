"""Card-vocabulary fingerprint for the structured NN's printing embeddings.

Why this exists
---------------

The structured model looks every card up by an integer *printing id* in
two per-printing tables: ``printing_embed.weight`` (the card embedding)
and ``hand_belief_queries`` (the Step-4 hand-belief stub). The id is a
pure function of the active card source:

* **Scheme ``sorted_names_v1``** (the only scheme; unchanged since the
  tokenizer landed on 2026-05-12): sort the names of every printing the
  active :class:`~mtg_bot.cards.source.CardSource` builds; the raw id of
  ``names[i]`` is ``i + 1`` (``0`` is ``PRINTING_NA``).
* The embedding **row** is ``raw`` when ``raw < n_printings`` and
  ``1 + (raw - 1) % (n_printings - 1)`` otherwise (the tokenizer's
  modulo bucketing — see :func:`row_for_raw_id`).

So adding one card to the pool, or teaching the parser to claim one
more card, renumbers every card that sorts after it. Before this module
a checkpoint recorded nothing about the list it was trained on, and a
checkpoint loaded under a different list read scrambled embeddings with
no error.

What it provides
----------------

* :class:`CardVocab` — the ordered name list plus a SHA-256 fingerprint
  (over the id scheme and the names). Serialised into every structured
  checkpoint's ``metadata.json`` under ``card_vocab``
  (:data:`METADATA_KEY`).
* :func:`resolve_checkpoint_card_vocab` — find a checkpoint's stored
  vocabulary (embedded metadata, or the stamping sidecar
  ``<weights>.card_vocab.json`` next to the weights file — see
  :func:`sidecar_path_for`). **No stored
  vocabulary is an error** (:class:`MissingCardVocabError`); nothing is
  ever inferred.
* :func:`adapt_structured_state_dict` — remap the per-printing rows of a
  loaded state_dict from the stored vocabulary to the current one, by
  card name. Shared cards keep their trained rows; new cards get exactly
  the initialisation a fresh model gives them; dropped cards are
  removed. Logged loudly.
* Shard helpers (:func:`shard_card_vocab_arrays`,
  :func:`resolve_shards_card_vocab`) — the training ``.npz`` files carry
  pre-tokenised printing ids too, so they record the vocabulary (and the
  bucketing cap) they were tokenised under, and the trainer refuses to
  mix id spaces or to train on unstamped shards without an explicit
  assertion.

See ``docs/CARD_VOCAB_FINGERPRINT.md`` for the full design and the
operator runbook (stamping the pre-fingerprint checkpoints).

The module's own top-level imports are stdlib only; torch and numpy
are imported lazily inside the remap and shard helpers, so the metadata
helpers (used by the coordinator's weights endpoint and the registry)
add no import cost of their own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

CARD_VOCAB_FORMAT = 1
"""Version of the serialised :class:`CardVocab` dict layout."""

ID_SCHEME_SORTED_NAMES_V1 = "sorted_names_v1"
"""raw id = 1 + index in the sorted name list; row = raw if raw <
n_printings else 1 + (raw - 1) % (n_printings - 1)."""

SUPPORTED_ID_SCHEMES: tuple[str, ...] = (ID_SCHEME_SORTED_NAMES_V1,)

METADATA_KEY = "card_vocab"
"""Top-level ``metadata.json`` key holding the serialised vocabulary."""

SIDECAR_SUFFIX = ".card_vocab.json"
"""A weights file ``X`` is stamped by ``X.card_vocab.json`` next to it
(per FILE, so two checkpoints in one directory never share a stamp).
Written by ``mtg_bot card-vocab stamp --sidecar`` for a legacy
checkpoint (whose ``weights.pt`` / ``metadata.json`` stay
byte-identical) and by ``NNEvaluator.save`` for a bare structured
weights file."""

SIDECAR_FILENAME = "weights.pt" + SIDECAR_SUFFIX
"""The sidecar's name inside a registry version directory."""

SIDECAR_FORMAT = 1

PER_PRINTING_PARAMS: tuple[str, ...] = (
    "printing_embed.weight",
    "hand_belief_queries",
)
"""Every ``StructuredModel`` parameter whose row index is a printing id.
``tests/training/test_card_vocab.py`` pins that this is exactly the set
of parameters whose first dimension is ``n_printings``."""

SHARD_KEY = "structured_card_vocab"
"""npz key: utf-8 JSON ``{"format", "id_scheme", "sha256", "names",
"n_printings_cap"}`` stored as a ``uint8`` array (no pickle)."""

LEGACY_SHARD_N_PRINTINGS_CAP = 320
"""The bucketing cap every pre-fingerprint structured shard was
tokenised under. ``data_gen`` never passed ``n_printings_cap`` before
this module, so the tokenizer's default applied, and
``structured_model.DEFAULT_N_PRINTINGS`` has been 320 since it was
introduced (commit 9118a41f, 2026-05-12) — a fact of history, not a
default that tracks the constant."""

SHARD_CARD_VOCAB_ENV = "MTG_BOT_SHARD_CARD_VOCAB"
"""Env default for ``mtg_bot train`` / ``calibrate-value``'s
``--shard-card-vocab`` (lets ``mtg_bot iterate``'s train subprocess
inherit the assertion without new plumbing)."""


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class CardVocabError(ValueError):
    """Base class for card-vocabulary failures."""


class MissingCardVocabError(CardVocabError):
    """A structured checkpoint carries no card vocabulary. Never
    guessed: stamp it first (``mtg_bot card-vocab stamp``)."""


class CardVocabIntegrityError(CardVocabError):
    """A stored vocabulary is malformed, its fingerprint does not match
    its names, or two stores for one checkpoint disagree."""


class ShardCardVocabError(CardVocabError):
    """Training shards disagree about, or do not record, the vocabulary
    their printing ids were tokenised under."""


# ---------------------------------------------------------------------
# Vocabulary value type
# ---------------------------------------------------------------------


def card_vocab_sha256(names: Sequence[str], id_scheme: str) -> str:
    """Fingerprint of an ordered vocabulary under ``id_scheme``.

    Covers the scheme so a future scheme change can never collide with
    a ``sorted_names_v1`` list of the same names."""
    h = hashlib.sha256()
    h.update(id_scheme.encode("utf-8"))
    h.update(b"\n")
    h.update("\n".join(names).encode("utf-8"))
    return h.hexdigest()


def row_for_raw_id(raw_id: int, n_printings: int) -> int:
    """Embedding row for raw printing id ``raw_id`` in a table of
    ``n_printings`` rows. Exactly the tokenizer's bucketing (single
    definition: ``state_tokenizer.printing_id`` calls this)."""
    if n_printings < 2:
        raise ValueError(f"n_printings must be >= 2; got {n_printings}")
    if raw_id < n_printings:
        return raw_id
    return 1 + ((raw_id - 1) % (n_printings - 1))


@dataclass(frozen=True)
class CardVocab:
    """An ordered card vocabulary (``names[i]`` has raw id ``i + 1``)."""

    names: tuple[str, ...]
    id_scheme: str = ID_SCHEME_SORTED_NAMES_V1

    def __post_init__(self) -> None:
        names = tuple(self.names)
        object.__setattr__(self, "names", names)
        if self.id_scheme not in SUPPORTED_ID_SCHEMES:
            raise CardVocabError(
                f"unsupported card-vocab id_scheme {self.id_scheme!r}; "
                f"this code knows {SUPPORTED_ID_SCHEMES}"
            )
        for name in names:
            if not isinstance(name, str) or not name or "\n" in name:
                raise CardVocabError(
                    f"card-vocab names must be non-empty single-line "
                    f"strings; got {name!r}"
                )
        if list(names) != sorted(set(names)):
            raise CardVocabError(
                "card-vocab names must be unique and sorted under "
                f"{ID_SCHEME_SORTED_NAMES_V1!r}"
            )

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "CardVocab":
        """Build the ``sorted_names_v1`` vocabulary of ``names``."""
        return cls(tuple(sorted(set(names))))

    @cached_property
    def sha256(self) -> str:
        return card_vocab_sha256(self.names, self.id_scheme)

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]

    @property
    def size(self) -> int:
        return len(self.names)

    @cached_property
    def id_map(self) -> dict[str, int]:
        """name -> raw printing id (1-based; 0 is ``PRINTING_NA``)."""
        return {name: i + 1 for i, name in enumerate(self.names)}

    def to_dict(
        self, *, provenance: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": CARD_VOCAB_FORMAT,
            "id_scheme": self.id_scheme,
            "size": self.size,
            "sha256": self.sha256,
            "names": list(self.names),
        }
        if provenance:
            out["provenance"] = dict(provenance)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CardVocab":
        """Parse + verify a serialised vocabulary (format, scheme, size
        and fingerprint must all agree with the names)."""
        if not isinstance(data, Mapping):
            raise CardVocabIntegrityError(
                f"card_vocab must be a JSON object; got {type(data).__name__}"
            )
        fmt = data.get("format")
        if fmt != CARD_VOCAB_FORMAT:
            raise CardVocabIntegrityError(
                f"card_vocab format {fmt!r} is not supported "
                f"(expected {CARD_VOCAB_FORMAT})"
            )
        names = data.get("names")
        if not isinstance(names, list):
            raise CardVocabIntegrityError("card_vocab.names must be a list")
        try:
            vocab = cls(tuple(names), str(data.get("id_scheme")))
        except CardVocabError as exc:
            raise CardVocabIntegrityError(str(exc)) from exc
        if data.get("size") != vocab.size:
            raise CardVocabIntegrityError(
                f"card_vocab.size {data.get('size')!r} != "
                f"{vocab.size} names"
            )
        if data.get("sha256") != vocab.sha256:
            raise CardVocabIntegrityError(
                f"card_vocab.sha256 {data.get('sha256')!r} does not match "
                f"its names (computed {vocab.sha256})"
            )
        return vocab


def coerce_card_vocab(value: Any) -> CardVocab:
    """Accept a :class:`CardVocab`, a serialised vocab dict, or a dict
    wrapping one under ``card_vocab`` (a sidecar or a metadata.json)."""
    if isinstance(value, CardVocab):
        return value
    if isinstance(value, Mapping):
        if "names" in value:
            return CardVocab.from_dict(value)
        inner = value.get(METADATA_KEY)
        if isinstance(inner, Mapping) and inner:
            return CardVocab.from_dict(inner)
    raise CardVocabIntegrityError(
        "expected a card vocabulary (a card_vocab dict, or a JSON object "
        "with a 'card_vocab' key)"
    )


def load_card_vocab_file(path: Path) -> CardVocab:
    """Read a vocabulary file: the output of ``mtg_bot card-vocab
    export``, a stamping sidecar, or a stamped ``metadata.json``."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise CardVocabError(f"cannot read card vocab file {path}: {exc}")
    try:
        return coerce_card_vocab(data)
    except CardVocabIntegrityError as exc:
        raise CardVocabIntegrityError(f"{path}: {exc}") from exc


def write_card_vocab_file(
    path: Path,
    vocab: CardVocab,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(vocab.to_dict(provenance=provenance), indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# ---------------------------------------------------------------------
# The current (active-source) vocabulary
# ---------------------------------------------------------------------

_CURRENT: Optional[tuple[object, CardVocab]] = None


def current_card_vocab() -> CardVocab:
    """The vocabulary the tokenizer uses right now: the sorted names of
    every printing the active card source builds. Cached per source
    object (tests swap the source via ``set_active_source``)."""
    global _CURRENT
    from mtg_bot.cards.source import get_active_source

    source = get_active_source()
    cached = _CURRENT
    if cached is not None and cached[0] is source:
        return cached[1]
    vocab = CardVocab.from_names(source.get_printings().keys())
    _CURRENT = (source, vocab)
    return vocab


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------
# Checkpoint-side: where a checkpoint's vocabulary lives
# ---------------------------------------------------------------------

_FILE_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, cached per ``(path, mtime_ns, size)`` so the
    per-game evaluator builds on a worker hash ``weights.pt`` once."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    cached = _FILE_SHA_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_SHA_CACHE[key] = digest
    return digest


def sidecar_path_for(weights_path: Path) -> Path:
    """Where the stamping sidecar of ``weights_path`` lives."""
    weights_path = Path(weights_path)
    return weights_path.with_name(weights_path.name + SIDECAR_SUFFIX)


def build_sidecar(
    vocab: CardVocab,
    weights_path: Path,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The stamping sidecar for ``weights_path``. It records the SHA-256
    of the weights it stamps so it can never silently apply to a
    different checkpoint that later lands in the same directory."""
    return {
        "format": SIDECAR_FORMAT,
        "weights_sha256": file_sha256(weights_path),
        "card_vocab": vocab.to_dict(provenance=provenance),
    }


def write_sidecar(
    vocab: CardVocab,
    weights_path: Path,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Stamp ``weights_path`` with ``vocab`` (atomic write of
    :func:`sidecar_path_for`). Overwrites an existing sidecar — callers
    that must not (``mtg_bot card-vocab stamp``) check first."""
    target = sidecar_path_for(weights_path)
    payload = build_sidecar(vocab, weights_path, provenance=provenance)
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_sidecar(
    sidecar_path: Path, weights_path: Path,
) -> tuple[CardVocab, dict[str, Any]]:
    """Parse + verify a stamping sidecar against ``weights_path``."""
    sidecar_path = Path(sidecar_path)
    try:
        with sidecar_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as exc:
        raise CardVocabIntegrityError(
            f"unreadable card-vocab sidecar {sidecar_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("format") != SIDECAR_FORMAT:
        raise CardVocabIntegrityError(
            f"{sidecar_path}: not a format-{SIDECAR_FORMAT} card-vocab "
            f"sidecar"
        )
    try:
        vocab = CardVocab.from_dict(payload.get("card_vocab") or {})
    except CardVocabIntegrityError as exc:
        raise CardVocabIntegrityError(f"{sidecar_path}: {exc}") from exc
    expected = payload.get("weights_sha256")
    if not Path(weights_path).is_file():
        raise CardVocabIntegrityError(
            f"{sidecar_path}: stamped weights {weights_path} not found"
        )
    actual = file_sha256(weights_path)
    if expected != actual:
        raise CardVocabIntegrityError(
            f"{sidecar_path} stamps weights with sha256 {expected!r}, but "
            f"{weights_path} has sha256 {actual!r}. The sidecar belongs "
            f"to a different checkpoint; re-stamp this one."
        )
    return vocab, payload


@dataclass(frozen=True)
class ResolvedCardVocab:
    """A checkpoint's stored vocabulary and where it was found."""

    vocab: CardVocab
    origin: str  # "explicit" | "metadata" | "sidecar"
    location: str


def _missing_message(weights_path: Path) -> str:
    return (
        f"structured checkpoint {weights_path} records no card vocabulary "
        f"(no '{METADATA_KEY}' in its metadata and no "
        f"{sidecar_path_for(weights_path).name} next to it). Its printing embeddings cannot be interpreted "
        f"without knowing which card list they were trained on, and that "
        f"is never guessed. Stamp it from the commit it was trained at, "
        f"e.g.\n"
        f"  mtg_bot card-vocab stamp --version-dir {weights_path.parent} "
        f"--use-metadata-commit --sidecar\n"
        f"See docs/CARD_VOCAB_FINGERPRINT.md."
    )


def _metadata_card_vocab(metadata: Any) -> Optional[Mapping[str, Any]]:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        raw = metadata.get(METADATA_KEY)
    else:
        raw = getattr(metadata, METADATA_KEY, None)
    return raw or None


def resolve_checkpoint_card_vocab(
    weights_path: Path,
    *,
    explicit: Any = None,
    metadata: Any = None,
) -> ResolvedCardVocab:
    """Find the vocabulary a structured checkpoint was trained on.

    Precedence: ``explicit`` (a vocab the caller already holds, e.g.
    ``ModelMetadata.card_vocab``) > ``metadata`` (dict or
    ``ModelMetadata``) > the ``metadata.json`` next to ``weights_path``.
    A stamping sidecar next to ``weights_path`` is always verified
    against the weights' SHA-256 and must agree with any embedded
    vocabulary. Nothing found -> :class:`MissingCardVocabError`.
    """
    weights_path = Path(weights_path)
    version_dir = weights_path.parent
    sidecar_path = sidecar_path_for(weights_path)
    sidecar_vocab: Optional[CardVocab] = None
    if sidecar_path.is_file():
        sidecar_vocab, _ = read_sidecar(sidecar_path, weights_path)

    embedded: Optional[CardVocab] = None
    origin = ""
    location = ""
    if explicit is not None:
        embedded = coerce_card_vocab(explicit)
        origin, location = "explicit", "caller-supplied card_vocab"
    else:
        raw = _metadata_card_vocab(metadata)
        location = "caller-supplied metadata"
        if raw is None and metadata is None:
            meta_path = version_dir / "metadata.json"
            if meta_path.is_file():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        raw = _metadata_card_vocab(json.load(f))
                except (OSError, ValueError) as exc:
                    raise CardVocabError(
                        f"cannot read {meta_path} to find the card "
                        f"vocabulary: {exc}"
                    ) from exc
                location = str(meta_path)
        if raw is not None:
            try:
                embedded = CardVocab.from_dict(raw)
            except CardVocabIntegrityError as exc:
                raise CardVocabIntegrityError(f"{location}: {exc}") from exc
            origin = "metadata"

    if (
        embedded is not None and sidecar_vocab is not None
        and embedded.sha256 != sidecar_vocab.sha256
    ):
        raise CardVocabIntegrityError(
            f"{location} records card vocabulary {embedded.short_sha} "
            f"({embedded.size} names) but {sidecar_path} stamps "
            f"{sidecar_vocab.short_sha} ({sidecar_vocab.size} names)"
        )
    if embedded is not None:
        return ResolvedCardVocab(embedded, origin, location)
    if sidecar_vocab is not None:
        return ResolvedCardVocab(sidecar_vocab, "sidecar", str(sidecar_path))
    raise MissingCardVocabError(_missing_message(weights_path))


def effective_metadata_dict(version_dir: Path) -> dict[str, Any]:
    """``<version_dir>/metadata.json`` with a stamping sidecar's
    vocabulary merged in under ``card_vocab``.

    Used by ``Registry.load_metadata`` and by the coordinator's
    ``/v2/weights/<v>/metadata`` endpoint, so every metadata consumer
    (and every worker's weights cache) sees a sidecar-stamped legacy
    checkpoint exactly as if the stamp were embedded. The files on disk
    are not modified. Without a sidecar the parsed metadata is returned
    unchanged.
    """
    version_dir = Path(version_dir)
    meta_path = version_dir / "metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    sidecar_path = version_dir / SIDECAR_FILENAME
    if not sidecar_path.is_file():
        return metadata
    vocab, payload = read_sidecar(sidecar_path, version_dir / "weights.pt")
    existing = _metadata_card_vocab(metadata)
    if existing is not None:
        embedded = CardVocab.from_dict(existing)
        if embedded.sha256 != vocab.sha256:
            raise CardVocabIntegrityError(
                f"{meta_path} records card vocabulary {embedded.short_sha} "
                f"but {sidecar_path} stamps {vocab.short_sha}"
            )
        return metadata
    merged = dict(metadata)
    inner = payload.get("card_vocab") or {}
    merged[METADATA_KEY] = vocab.to_dict(provenance=inner.get("provenance"))
    return merged


# ---------------------------------------------------------------------
# Remap on load
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class VocabRemapReport:
    """What happened to a checkpoint's per-printing rows at load."""

    stored_sha256: str
    current_sha256: str
    stored_size: int
    current_size: int
    n_kept: int
    n_new: int
    n_dropped: int
    old_n_printings: int
    new_n_printings: int
    new_names: tuple[str, ...] = field(default=())
    dropped_names: tuple[str, ...] = field(default=())
    origin: str = ""

    @property
    def identical(self) -> bool:
        return self.stored_sha256 == self.current_sha256

    def summary(self) -> str:
        if self.identical:
            return (
                f"card vocabulary {self.current_sha256[:12]} matches the "
                f"checkpoint ({self.current_size} names, n_printings "
                f"{self.old_n_printings}); no remap"
            )

        def _sample(names: tuple[str, ...]) -> str:
            head = ", ".join(names[:5])
            more = f", ... (+{len(names) - 5})" if len(names) > 5 else ""
            return f" [{head}{more}]" if names else ""

        return (
            f"CARD VOCABULARY REMAP: checkpoint vocab "
            f"{self.stored_sha256[:12]} ({self.stored_size} names, "
            f"n_printings {self.old_n_printings}) -> current vocab "
            f"{self.current_sha256[:12]} ({self.current_size} names, "
            f"n_printings {self.new_n_printings}): "
            f"kept {self.n_kept}, new {self.n_new}"
            f"{_sample(self.new_names)}, dropped {self.n_dropped}"
            f"{_sample(self.dropped_names)}. Kept cards keep their trained "
            f"embedding rows; new cards are freshly initialised."
        )


def plan_remap_n_printings(
    stored: CardVocab, current: CardVocab, old_n_printings: int,
) -> int:
    """Embedding-table size for a remapped checkpoint.

    Unchanged vocabulary -> unchanged size (bit-identical load). A
    changed vocabulary grows the table to hold every current card in
    its own row (``current.size + 1``, never shrinking): with modulo
    bucketing a new card set would share rows among cards whose trained
    rows differ, and no remap could keep every shared card's row."""
    if stored.sha256 == current.sha256:
        return int(old_n_printings)
    return max(int(old_n_printings), current.size + 1)


def remap_seed(stored: CardVocab, current: CardVocab, new_n: int) -> int:
    """Deterministic init seed for the new rows, so every process (every
    worker) that remaps the same checkpoint builds the same model."""
    digest = hashlib.sha256(
        f"{stored.sha256}->{current.sha256}|{new_n}".encode("ascii")
    ).hexdigest()
    return int(digest[:15], 16)


def fresh_per_printing_params(config: Any, *, seed: int) -> dict[str, Any]:
    """The per-printing tensors a freshly constructed
    ``StructuredModel(config)`` would have, under a fixed seed. Built
    inside ``torch.random.fork_rng`` so the caller's RNG stream is not
    disturbed."""
    import torch

    from mtg_bot.training.structured_model import StructuredModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = StructuredModel(config)
    state = model.state_dict()
    return {key: state[key].detach().clone() for key in PER_PRINTING_PARAMS}


def remap_printing_rows(
    old: Any,
    fresh: Any,
    *,
    stored: CardVocab,
    current: CardVocab,
    old_n_printings: int,
    new_n_printings: int,
) -> Any:
    """Return ``fresh`` with every shared card's row replaced by its
    trained row from ``old`` (and the NA row 0 copied across)."""
    import torch

    if fresh.shape[0] != new_n_printings:
        raise ValueError(
            f"fresh table has {fresh.shape[0]} rows, expected "
            f"{new_n_printings}"
        )
    if old.shape[0] != old_n_printings:
        raise ValueError(
            f"stored table has {old.shape[0]} rows, expected "
            f"{old_n_printings}"
        )
    out = fresh.clone().to(dtype=old.dtype, device=old.device)
    src_rows = [0]
    dst_rows = [0]
    old_ids = stored.id_map
    seen_dst: set[int] = {0}
    for name, new_raw in current.id_map.items():
        old_raw = old_ids.get(name)
        if old_raw is None:
            continue
        dst = row_for_raw_id(new_raw, new_n_printings)
        if dst in seen_dst:
            raise CardVocabIntegrityError(  # pragma: no cover - guarded
                f"remap target row {dst} assigned twice; the remapped "
                f"table must hold every current card in its own row"
            )
        seen_dst.add(dst)
        src_rows.append(row_for_raw_id(old_raw, old_n_printings))
        dst_rows.append(dst)
    src = torch.tensor(src_rows, dtype=torch.long, device=old.device)
    dst = torch.tensor(dst_rows, dtype=torch.long, device=old.device)
    out[dst] = old.index_select(0, src)
    return out


_LOGGED_REMAPS: set[tuple[str, str, str]] = set()


def adapt_structured_state_dict(
    state_dict: Mapping[str, Any],
    *,
    stored: CardVocab,
    current: CardVocab,
    config: Any,
    origin: str = "",
    log_key: str = "",
) -> tuple[Any, dict[str, Any], VocabRemapReport]:
    """Adapt a structured checkpoint's state_dict to ``current``.

    Returns ``(config, state_dict, report)``. ``config`` is the input
    config when the vocabulary is unchanged, else a copy with the
    planned ``n_printings``. Raises :class:`CardVocabIntegrityError` if
    the weights disagree with the config about the table size.
    """
    import dataclasses

    missing = [k for k in PER_PRINTING_PARAMS if k not in state_dict]
    if missing:
        raise CardVocabIntegrityError(
            f"structured state_dict lacks per-printing parameters {missing}"
        )
    old_n = int(state_dict["printing_embed.weight"].shape[0])
    for key in PER_PRINTING_PARAMS:
        if int(state_dict[key].shape[0]) != old_n:
            raise CardVocabIntegrityError(
                f"{key} has {int(state_dict[key].shape[0])} rows but "
                f"printing_embed.weight has {old_n}"
            )
    stored_names = set(stored.names)
    current_names = set(current.names)
    new_names = tuple(n for n in current.names if n not in stored_names)
    dropped = tuple(n for n in stored.names if n not in current_names)
    new_n = plan_remap_n_printings(stored, current, old_n)
    report = VocabRemapReport(
        stored_sha256=stored.sha256,
        current_sha256=current.sha256,
        stored_size=stored.size,
        current_size=current.size,
        n_kept=current.size - len(new_names),
        n_new=len(new_names),
        n_dropped=len(dropped),
        old_n_printings=old_n,
        new_n_printings=new_n,
        new_names=new_names,
        dropped_names=dropped,
        origin=origin,
    )
    if report.identical:
        # Unchanged vocabulary: hand the state_dict back untouched. A
        # config that disagrees with the checkpoint (any dim, including
        # n_printings) surfaces as load_state_dict's own size-mismatch
        # error, exactly as before the fingerprint existed.
        return config, dict(state_dict), report
    if int(config.n_printings) != old_n:
        raise CardVocabIntegrityError(
            f"checkpoint embedding table has {old_n} rows but the model "
            f"was built with n_printings={config.n_printings}; the "
            f"structured config passed to the loader does not match the "
            f"checkpoint's metadata, so its rows cannot be remapped"
        )

    new_config = dataclasses.replace(config, n_printings=new_n)
    fresh = fresh_per_printing_params(
        new_config, seed=remap_seed(stored, current, new_n),
    )
    adapted = dict(state_dict)
    for key in PER_PRINTING_PARAMS:
        adapted[key] = remap_printing_rows(
            state_dict[key], fresh[key],
            stored=stored, current=current,
            old_n_printings=old_n, new_n_printings=new_n,
        )
    once = (log_key, stored.sha256, current.sha256)
    if once not in _LOGGED_REMAPS:
        _LOGGED_REMAPS.add(once)
        where = f" ({log_key})" if log_key else ""
        _log.warning("%s%s", report.summary(), where)
    return new_config, adapted, report


def load_structured_checkpoint(
    model: Any,
    weights_path: Path,
    *,
    map_location: Any,
    card_vocab: Any = None,
    metadata: Any = None,
) -> tuple[Any, VocabRemapReport]:
    """Load ``weights_path`` into ``model`` (a ``StructuredModel``),
    remapped to the current card vocabulary.

    The single structured-weights load path: ``NNEvaluator`` and the
    SHM eval server both call it. Returns the model to use — a rebuilt
    one when the remap grew the embedding table — and the report.
    """
    import torch

    weights_path = Path(weights_path)
    state_dict = torch.load(
        weights_path, map_location=map_location, weights_only=True,
    )
    resolved = resolve_checkpoint_card_vocab(
        weights_path, explicit=card_vocab, metadata=metadata,
    )
    new_config, adapted, report = adapt_structured_state_dict(
        state_dict,
        stored=resolved.vocab,
        current=current_card_vocab(),
        config=model.config,
        origin=resolved.origin,
        log_key=str(weights_path),
    )
    if new_config.n_printings != model.config.n_printings:
        model = type(model)(new_config)
    model.load_state_dict(adapted)
    return model, report


# ---------------------------------------------------------------------
# Shards (training .npz files)
# ---------------------------------------------------------------------


def shard_card_vocab_arrays(
    vocab: CardVocab, n_printings_cap: int,
) -> dict[str, Any]:
    """npz entries recording the id space a shard was tokenised in."""
    import numpy as np

    payload = {
        "format": CARD_VOCAB_FORMAT,
        "id_scheme": vocab.id_scheme,
        "sha256": vocab.sha256,
        "n_printings_cap": int(n_printings_cap),
        "names": list(vocab.names),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return {SHARD_KEY: np.frombuffer(raw, dtype=np.uint8).copy()}


def decode_shard_card_vocab(raw: Any, *, label: str = "") -> tuple[CardVocab, int]:
    """Inverse of :func:`shard_card_vocab_arrays`."""
    import numpy as np

    where = f"{label}: " if label else ""
    try:
        payload = json.loads(np.asarray(raw, dtype=np.uint8).tobytes())
    except ValueError as exc:
        raise ShardCardVocabError(
            f"{where}unreadable {SHARD_KEY} entry: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ShardCardVocabError(f"{where}{SHARD_KEY} is not a JSON object")
    cap = payload.get("n_printings_cap")
    if not isinstance(cap, int) or cap < 2:
        raise ShardCardVocabError(
            f"{where}{SHARD_KEY}.n_printings_cap must be an int >= 2; "
            f"got {cap!r}"
        )
    try:
        vocab = CardVocab.from_dict({
            "format": payload.get("format"),
            "id_scheme": payload.get("id_scheme"),
            "size": len(payload.get("names") or []),
            "sha256": payload.get("sha256"),
            "names": payload.get("names"),
        })
    except CardVocabIntegrityError as exc:
        raise ShardCardVocabError(f"{where}{exc}") from exc
    return vocab, cap


@dataclass(frozen=True)
class ShardVocabResolution:
    """The single id space a set of training shards was tokenised in."""

    vocab: CardVocab
    n_printings_cap: int
    n_stamped: int
    n_legacy: int
    n_empty_legacy: int
    legacy_assumed: bool

    def provenance(self, legacy_source: Optional[str]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": "training_shards",
            "n_stamped_files": self.n_stamped,
            "n_legacy_files": self.n_legacy,
            "n_empty_legacy_files": self.n_empty_legacy,
            "n_printings_cap": self.n_printings_cap,
        }
        if self.legacy_assumed:
            out["legacy_shards_asserted_by"] = legacy_source
        return out


def _shard_rows(data: Any) -> int:
    """Row count of a structured shard, read off its smallest array
    (``outcomes``) when present, else the token ids it is judged on."""
    key = "outcomes" if "outcomes" in data.files else "structured_token_emb_ids"
    return int(data[key].shape[0])


def resolve_shards_card_vocab(
    paths: Sequence[Path],
    *,
    legacy_vocab: Optional[CardVocab] = None,
) -> Optional[ShardVocabResolution]:
    """Determine the one ``(vocabulary, cap)`` a set of shards uses.

    * Stamped shards must all agree (any second id space -> error naming
      an example file of each).
    * Unstamped (pre-fingerprint) shards with rows are refused unless
      ``legacy_vocab`` asserts their vocabulary; they were tokenised at
      :data:`LEGACY_SHARD_N_PRINTINGS_CAP`. If stamped shards are present
      too, the assertion must match them.
    * Unstamped zero-row shards (crashed / timed-out games) carry no
      printing ids and are counted but not judged.

    Returns ``None`` only when there is nothing to resolve: no shard
    carries structured-token rows at all (no structured arrays, or only
    unstamped empty ones). Callers then hit their own "not structured" /
    "no samples" errors; a caller that goes on to USE printing ids must
    treat ``None`` as an error.
    """
    import numpy as np

    groups: dict[tuple[str, int], tuple[CardVocab, list[Path]]] = {}
    legacy: list[Path] = []
    n_empty_legacy = 0
    for path in paths:
        path = Path(path)
        with np.load(path) as data:
            if "structured_token_emb_ids" not in data.files:
                # No printing ids at all (a flat-feature shard): not
                # this resolver's business — the structured loader
                # rejects it with its own error.
                continue
            if SHARD_KEY in data.files:
                vocab, cap = decode_shard_card_vocab(
                    data[SHARD_KEY], label=str(path),
                )
                key = (vocab.sha256, cap)
                if key not in groups:
                    groups[key] = (vocab, [])
                groups[key][1].append(path)
            elif _shard_rows(data) == 0:
                n_empty_legacy += 1
            else:
                legacy.append(path)
    if len(groups) > 1:
        detail = "; ".join(
            f"vocab {sha[:12]} ({vocab.size} names) cap {cap}: "
            f"{len(files)} files, e.g. {files[0].name}"
            for (sha, cap), (vocab, files) in groups.items()
        )
        raise ShardCardVocabError(
            f"training shards were tokenised under {len(groups)} different "
            f"card vocabularies / caps ({detail}). Their printing ids are "
            f"different id spaces and cannot be trained together. Train "
            f"on one set (e.g. quarantine the files from the stale "
            f"workers)."
        )
    stamped = next(iter(groups.values()), None)
    stamped_key = next(iter(groups), None)
    if legacy:
        if legacy_vocab is None:
            raise ShardCardVocabError(
                f"{len(legacy)} training shards (e.g. {legacy[0].name}) "
                f"predate the card-vocab fingerprint and record no "
                f"vocabulary. Their printing ids cannot be interpreted "
                f"without one, and it is never guessed. Pass "
                f"--shard-card-vocab FILE (or set {SHARD_CARD_VOCAB_ENV}) "
                f"naming the vocabulary they were generated under, e.g. "
                f"from `mtg_bot card-vocab export --git-commit <sha> --out "
                f"FILE`. See docs/CARD_VOCAB_FINGERPRINT.md."
            )
        if stamped is not None and (
            stamped_key[0] != legacy_vocab.sha256
            or stamped_key[1] != LEGACY_SHARD_N_PRINTINGS_CAP
        ):
            raise ShardCardVocabError(
                f"the asserted legacy shard vocabulary "
                f"{legacy_vocab.short_sha} (cap "
                f"{LEGACY_SHARD_N_PRINTINGS_CAP}) differs from the "
                f"stamped shards' {stamped_key[0][:12]} (cap "
                f"{stamped_key[1]}, e.g. {stamped[1][0].name}); they "
                f"cannot be trained together."
            )
        return ShardVocabResolution(
            vocab=legacy_vocab,
            n_printings_cap=LEGACY_SHARD_N_PRINTINGS_CAP,
            n_stamped=len(stamped[1]) if stamped else 0,
            n_legacy=len(legacy),
            n_empty_legacy=n_empty_legacy,
            legacy_assumed=True,
        )
    if stamped is None:
        return None
    return ShardVocabResolution(
        vocab=stamped[0],
        n_printings_cap=stamped_key[1],
        n_stamped=len(stamped[1]),
        n_legacy=0,
        n_empty_legacy=n_empty_legacy,
        legacy_assumed=False,
    )


def legacy_shard_vocab_from_arg(value: Optional[str]) -> Optional[CardVocab]:
    """Resolve ``--shard-card-vocab`` (falling back to the env var)."""
    chosen = value if value else os.environ.get(SHARD_CARD_VOCAB_ENV)
    if not chosen:
        return None
    return load_card_vocab_file(Path(chosen))


__all__ = [
    "CARD_VOCAB_FORMAT",
    "CardVocab",
    "CardVocabError",
    "CardVocabIntegrityError",
    "ID_SCHEME_SORTED_NAMES_V1",
    "LEGACY_SHARD_N_PRINTINGS_CAP",
    "METADATA_KEY",
    "MissingCardVocabError",
    "PER_PRINTING_PARAMS",
    "ResolvedCardVocab",
    "SHARD_CARD_VOCAB_ENV",
    "SHARD_KEY",
    "SIDECAR_FILENAME",
    "SIDECAR_SUFFIX",
    "sidecar_path_for",
    "write_sidecar",
    "ShardCardVocabError",
    "ShardVocabResolution",
    "VocabRemapReport",
    "adapt_structured_state_dict",
    "build_sidecar",
    "card_vocab_sha256",
    "coerce_card_vocab",
    "current_card_vocab",
    "decode_shard_card_vocab",
    "effective_metadata_dict",
    "file_sha256",
    "fresh_per_printing_params",
    "legacy_shard_vocab_from_arg",
    "load_card_vocab_file",
    "load_structured_checkpoint",
    "plan_remap_n_printings",
    "read_sidecar",
    "remap_printing_rows",
    "remap_seed",
    "resolve_checkpoint_card_vocab",
    "resolve_shards_card_vocab",
    "row_for_raw_id",
    "shard_card_vocab_arrays",
    "write_card_vocab_file",
]
