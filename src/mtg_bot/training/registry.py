"""Versioned model checkpoints + metadata.

Each trained network lives in its own subdirectory under the registry
root (default ``trained_weights/``):

    trained_weights/
      v1/
        weights.pt
        metadata.json
      v2/
        weights.pt
        metadata.json
      ...

``weights.pt`` is the standard PyTorch state_dict NNEvaluator loads.
``metadata.json`` captures everything needed to interpret a version
later: training hyperparameters, data sources, training-loss history,
evaluation results vs baselines, architecture shape, git commit.

Versioning is sequential: ``v1``, ``v2``, ... auto-incremented at
register time. Sequential rather than hash-based so "the last 32
versions" is meaningful (the tournament harness uses this).

Older versions are kept indefinitely — registry doesn't prune. The
tournament can ignore old/bad versions explicitly via ``--n-versions``.
Disk cost is small (~100 KB per version: 25k-param weights + JSON).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Filename convention. The version-id regex matches ``v1``, ``v23``,
# etc. — sequential integers. Anything that doesn't match gets
# ignored (so manual notes / scratch files in the registry don't
# break enumeration).
_VERSION_RE = re.compile(r"^v(\d+)$")
_WEIGHTS_FILENAME = "weights.pt"
_METADATA_FILENAME = "metadata.json"


@dataclass
class ModelMetadata:
    """One trained network's metadata. Serialized to ``metadata.json``
    alongside ``weights.pt``.

    All fields are JSON-serializable primitives or nested dicts.
    Schema is forward-compatible: missing keys default sensibly,
    extra keys are preserved on round-trip via ``extra``.
    """

    version: str  # e.g. "v3"
    created_at: str  # ISO 8601, UTC
    feature_dim: int
    hidden_sizes: list[int]
    # Whether this version was trained with the two-headed
    # value+policy architecture (``_ValuePolicyMLP``) or the older
    # value-only ``_ValueMLP``. Defaults to False so existing v1/v2
    # metadata files round-trip unchanged: when an old metadata.json
    # is read, the missing key falls through to the default and the
    # tournament harness keeps loading those weights into a value-
    # only NNEvaluator. v4+ writes True for AlphaZero-style runs.
    with_policy: bool = False
    # Dual-head model architecture: ``"shared"`` (single trunk feeds
    # both heads, original ``_ValuePolicyMLP`` — what v1-v15 used) or
    # ``"separate"`` (parallel value_trunk + policy_trunk feeding their
    # own heads, ``_ValuePolicyMLPSeparate``). Default ``"shared"`` so
    # old metadata.json files (no architecture key) round-trip
    # unchanged and load via the original class. Ignored when
    # ``with_policy=False`` (the value-only ``_ValueMLP`` has no
    # architecture choice).
    architecture: str = "shared"
    # ``action_encoding_version`` carries which slot-mapping convention
    # the network's policy head was trained against. Two values today:
    # - ``"positional_v1"`` (default, back-compat with v13/v15/v16):
    #   hand/graveyard/exile slot positions are taken from the engine's
    #   raw zone list (positional by ``card.id`` order). Two states
    #   differing only in card.id permutation produce different slot
    #   mappings — the MCTS NN-eval cache defends this with a
    #   slot_fingerprint cache-key extension (commit 2f50fca).
    # - ``"canonical_v2"``: zones are sorted by
    #   ``(printing.name, modifier_tuple..., card.id)`` before slot
    #   assignment, so two canonically-equivalent states share their
    #   slot mappings. Opt-in for v17+; the slot_fingerprint workaround
    #   becomes a no-op for v2 weights (positions are already
    #   canonical).
    # Default ``"positional_v1"`` so old metadata.json files (no
    # action_encoding_version key) load with bit-exact pre-canonical
    # behavior; loaders thread this through to NNEvaluator's
    # ``legal_action_mask`` calls so v13/v15/v16 keep their trained
    # slot semantics.
    action_encoding_version: str = "positional_v1"
    # Training inputs and hyperparameters.
    training: dict[str, Any] = field(default_factory=dict)
    # Per-epoch loss series + summary, from TrainingHistory.
    training_history: dict[str, Any] = field(default_factory=dict)
    # Evaluation results, keyed by comparison name. E.g.
    # ``{"vs_heuristic": {"win_rate": 0.25, "wins": 5, ...}}``.
    # Updated retroactively as more comparisons are run via
    # ``Registry.add_evaluation_result``.
    evaluation: dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None
    notes: str = ""
    # Anything we don't know about — kept on round-trip so future
    # readers don't lose data written by future writers.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelMetadata":
        known = {
            "version", "created_at", "feature_dim", "hidden_sizes",
            "with_policy", "architecture", "action_encoding_version",
            "training", "training_history", "evaluation", "git_commit",
            "notes", "extra",
        }
        # Stuff future-written keys we don't recognize into ``extra``.
        extra = dict(d.get("extra", {}))
        for k, v in d.items():
            if k not in known:
                extra[k] = v
        return cls(
            version=d["version"],
            created_at=d["created_at"],
            feature_dim=d["feature_dim"],
            hidden_sizes=list(d["hidden_sizes"]),
            # Default False so old metadata.json (no with_policy key)
            # loads as a value-only network — backward compat.
            with_policy=bool(d.get("with_policy", False)),
            # Default "shared" so old metadata.json (no architecture key)
            # loads as the original single-trunk dual-head model — back-
            # compat for v1-v15.
            architecture=str(d.get("architecture", "shared")),
            # Default "positional_v1" so old metadata.json (no
            # action_encoding_version key) loads with bit-exact pre-
            # canonical slot semantics — back-compat for v13/v15/v16
            # weights that were trained against the positional-by-
            # card.id mapping.
            action_encoding_version=str(
                d.get("action_encoding_version", "positional_v1"),
            ),
            training=dict(d.get("training", {})),
            training_history=dict(d.get("training_history", {})),
            evaluation=dict(d.get("evaluation", {})),
            git_commit=d.get("git_commit"),
            notes=d.get("notes", ""),
            extra=extra,
        )


class Registry:
    """File-system backed model registry.

    Atomic-write pattern: each ``register`` writes to a fresh
    ``v{N+1}/`` directory, so partial writes don't leave the registry
    in an inconsistent state. (A crash between ``weights.pt`` and
    ``metadata.json`` writes would leave a directory with one of two
    files; ``list_versions`` filters those out via the
    ``has_complete_files`` check.)
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- enumeration -------------------------------------------------------

    def list_versions(self) -> list[str]:
        """Return all complete version IDs sorted by version number
        (``v1``, ``v2``, ...). Directories that don't match the
        version pattern OR are missing weights/metadata are skipped."""
        versions: list[tuple[int, str]] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            match = _VERSION_RE.match(child.name)
            if match is None:
                continue
            if not self._is_complete(child):
                continue
            versions.append((int(match.group(1)), child.name))
        versions.sort()
        return [name for _, name in versions]

    def latest_versions(self, n: int) -> list[str]:
        """Return up to the last N versions, NEWEST FIRST.

        ``n`` is a soft cap — if fewer versions exist, returns
        whatever's there. Used by the tournament harness to limit
        round-robin scope. ``n=0`` returns an empty list (handled
        explicitly because Python's ``[-0:]`` slice is the whole
        list, not nothing)."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0:
            return []
        all_versions = self.list_versions()
        # all_versions is oldest→newest; we want newest→oldest, capped at n.
        return list(reversed(all_versions[-n:]))

    def has(self, version: str) -> bool:
        return self._is_complete(self.root / version)

    # ---- access ------------------------------------------------------------

    def weights_path(self, version: str) -> Path:
        path = self.root / version / _WEIGHTS_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"weights for version {version!r} not found at {path}"
            )
        return path

    def load_metadata(self, version: str) -> ModelMetadata:
        path = self.root / version / _METADATA_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"metadata for version {version!r} not found at {path}"
            )
        with path.open("r", encoding="utf-8") as f:
            return ModelMetadata.from_dict(json.load(f))

    # ---- mutation ----------------------------------------------------------

    def next_version_id(self) -> str:
        """Compute the next sequential version ID. v1 if the registry
        is empty; v{max+1} otherwise. Considers BOTH complete versions
        and partially-written ones to avoid collisions on retry."""
        max_n = 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            match = _VERSION_RE.match(child.name)
            if match is None:
                continue
            max_n = max(max_n, int(match.group(1)))
        return f"v{max_n + 1}"

    def register(
        self,
        weights_source: Path,
        metadata: ModelMetadata,
        version: Optional[str] = None,
    ) -> str:
        """Copy ``weights_source`` into a new versioned subdirectory
        and write the metadata sidecar. Returns the new version ID.

        ``version`` defaults to ``next_version_id()``; pass an
        explicit ID to overwrite (useful for tests). The metadata's
        ``version`` field is overwritten to match the chosen ID, so
        callers don't need to pre-set it correctly.
        """
        if version is None:
            version = self.next_version_id()
        elif _VERSION_RE.match(version) is None:
            raise ValueError(
                f"version must match v\\d+ pattern, got {version!r}"
            )
        target_dir = self.root / version
        target_dir.mkdir(parents=True, exist_ok=True)
        weights_target = target_dir / _WEIGHTS_FILENAME
        # Copy weights bytes (not move — the caller may still want
        # the source file).
        weights_target.write_bytes(Path(weights_source).read_bytes())
        # Persist metadata last so a crash mid-register leaves a
        # detectable incomplete directory (no metadata.json).
        metadata.version = version
        if not metadata.created_at:
            metadata.created_at = _now_iso()
        self._write_metadata(version, metadata)
        return version

    def add_evaluation_result(
        self, version: str, name: str, result: dict[str, Any],
    ) -> None:
        """Update a registered version's metadata to include an
        evaluation result. ``name`` is a short tag like
        ``"vs_heuristic"`` or ``"vs_v1"``; ``result`` is any JSON-
        serializable dict. Overwrites if already present."""
        metadata = self.load_metadata(version)
        metadata.evaluation[name] = result
        self._write_metadata(version, metadata)

    # ---- internals ---------------------------------------------------------

    def _is_complete(self, version_dir: Path) -> bool:
        return (
            (version_dir / _WEIGHTS_FILENAME).is_file()
            and (version_dir / _METADATA_FILENAME).is_file()
        )

    def _write_metadata(
        self, version: str, metadata: ModelMetadata,
    ) -> None:
        path = self.root / version / _METADATA_FILENAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, sort_keys=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
