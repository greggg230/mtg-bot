"""Distribute self-play across local + remote machines.

Today ``iterate_alphazero``'s self-play step runs ``mp.Pool(workers=24)``
on the master box. Each iter spends ~50 minutes here. With other
machines on the same Tailnet (a Windows box, an Apple Silicon Mac),
we can shard the work and roughly halve that wall-time.

Architecture: each shard produces ONE ``iter_NNN_<bootstrap>_shard_
<label>_<utc>.npz`` file in the master's iterate-data dir. The
orchestrator then concatenates them in the existing
``train_nn.py --data-dir`` glob path — no schema change there.

Code distribution: each remote has its own ``git clone`` of the repo;
the orchestrator runs ``git pull`` over SSH before each shard so
master commits to ``main`` propagate automatically. Weight
distribution: ``scp`` the latest version's weights at the start of
each self-play phase. Result collection: ``scp`` the shard npz back
to the master after the remote finishes.

This module is the importable / testable core. The thin CLI wrapper
lives in ``scripts/distributed_self_play.py``; ``iterate_alphazero``
calls into here directly via ``run_distributed_self_play``.
"""

from __future__ import annotations

import concurrent.futures
import json
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Optional

from mtg_bot.search.heuristics_config import HeuristicsConfig


# ---- queue HTTP retry policy -------------------------------------
#
# Tailscale local network; <100ms typical RTT. A momentary blip on
# the tailnet (lost ARP, reconnect, brief NIC stall) should NOT kill
# a shard that's been running for 30-60 minutes. We saw exactly that
# on 2026-05-04: a Mac shard hit a 60s OS-level TCP-connect ETIMEDOUT
# on `/queue/take` (errno 60 on macOS) — the urllib `timeout=30` only
# applies AFTER the TCP connection is established, so the OS default
# bit us first. The fix: a tight per-attempt timeout plus a small
# bounded number of retries with exponential backoff. Tight timeout
# + retries beats one long blocking call when the failure mode is a
# transient blip; for a real outage the shard still fails, just
# faster.
#
# Worst-case wall before final failure with the constants below:
#   10 + 0.5 + 10 + 1.0 + 10 = 31.5 seconds
# (vs the 60s OS-default that bit us). Most blips resolve on the
# first retry well under 1s.
_QUEUE_HTTP_ATTEMPT_TIMEOUT_S = 10.0  # connect-or-read-or-fail
_QUEUE_HTTP_RETRIES = 3              # total attempts incl. the first
_QUEUE_HTTP_BACKOFF_BASE_S = 0.5     # 0.5s, then 1.0s between attempts

# Hard floor timeout on a single shard's SSH session. The watchdog
# (see ``_run_ssh_with_terminal_status_watchdog``) collapses the
# typical Tailscale-keeps-TCP-alive hang from many minutes to ~10s
# by terminating SSH as soon as the per-shard progress JSON shows
# terminal status; THIS constant only fires when the watchdog never
# sees a terminal status (e.g. shard died before writing one).
# 90 min covers ~3x the typical 500-game self-play wall + rsync;
# tunable via ``iterate_alphazero --shard-ssh-timeout-min``.
_DEFAULT_SHARD_SSH_TIMEOUT_S = 90 * 60


@dataclass(frozen=True)
class _QueueHTTPResponse:
    """Minimal response shape returned by :func:`_urlopen_with_retry`:
    just the status code and the body bytes. Wrapping these here lets
    callers check ``resp.status`` without keeping the underlying
    ``http.client.HTTPResponse`` open across the retry boundary."""

    status: int
    body: bytes


def _urlopen_with_retry(
    req,
    *,
    timeout: float = _QUEUE_HTTP_ATTEMPT_TIMEOUT_S,
    retries: int = _QUEUE_HTTP_RETRIES,
    backoff_base: float = _QUEUE_HTTP_BACKOFF_BASE_S,
    sleep=time.sleep,
) -> _QueueHTTPResponse:
    """Open ``req`` via :mod:`urllib.request` with a tight per-attempt
    timeout and a small bounded number of retries with exponential
    backoff. Returns a :class:`_QueueHTTPResponse` (status + body
    bytes); the underlying response object is closed before return.

    On the cluster's Tailscale network typical RTT is 6-65ms, so a
    queue endpoint should resolve in well under 100ms. If a single
    attempt exceeds ``timeout`` seconds (or any ``URLError`` /
    ``ConnectionError`` / ``TimeoutError`` is raised), the helper
    sleeps ``backoff_base * 2**attempt`` and tries again. After
    ``retries`` total attempts it re-raises the last exception, so
    callers see the original failure mode (URLError, etc.) — only
    transient blips are absorbed silently.

    ``sleep`` is parameterized so tests can stub it without inserting
    real sleep delays.
    """
    import urllib.error
    import urllib.request as _urllib_request
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            with _urllib_request.urlopen(req, timeout=timeout) as resp:
                return _QueueHTTPResponse(
                    status=resp.status, body=resp.read(),
                )
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_exc = exc
            if attempt + 1 == retries:
                raise
            sleep(backoff_base * (2 ** attempt))
    # Unreachable: the loop either returns or re-raises.
    assert last_exc is not None  # pragma: no cover
    raise last_exc  # pragma: no cover


@dataclass(frozen=True)
class ShardConfig:
    """One shard of a distributed self-play run.

    A shard is a worker (local or SSH-reachable) that pulls games
    from the master's shared queue. The total number of games per
    iter comes from the iterate driver's ``--games-per-iteration``;
    this config carries only worker / SSH info, no per-shard quota
    (a per-shard ``n_games`` field used to live here, but in pull
    mode it had no effect — slow machines pull fewer games and fast
    ones pull more, automatically — so it's been removed).

    Each shard still gets its own seed-offset so seed collisions
    across shards are impossible (different shards play DIFFERENT
    games for the same game-index, not the same games).

    ``host="local"`` runs in the master process via the pull-mode
    runner. Any other ``host`` value triggers the SSH remote-shard
    path.
    """

    # Where the shard runs. ``"local"`` means in-process on the
    # master; anything else is a Tailscale-reachable hostname or IP.
    host: str
    # Free-form short label for naming output artifacts (e.g. the
    # shard's npz file). Defaults to ``host`` when omitted.
    label: str = ""
    # Per-shard seed offset. Final per-game seed is
    # ``base_seed + seed_offset + game_idx``. Caller should ensure
    # offsets across shards don't overlap. Idiomatic pattern: the
    # orchestrator allocates 1_000_000 per shard so 500-game shards
    # never collide.
    seed_offset: int = 0
    # Worker count for the shard's local mp.Pool. Defaults to None
    # (the script's own default, currently cpu_count() - 2).
    n_workers: Optional[int] = None

    # ---- remote-only fields (ignored when host == "local") ----
    # SSH user. Required for remote shards.
    user: Optional[str] = None
    # Absolute path to the repo's working tree on the remote.
    repo_path: Optional[str] = None
    # Path (relative to ``repo_path``) of the venv's python
    # executable. Windows: ``.venv-py313\\Scripts\\python.exe``.
    # macOS / Linux: ``.venv-py313/bin/python``. Different across
    # OSes by Python convention.
    venv_python: Optional[str] = None
    # SSH port; default 22.
    ssh_port: int = 22

    @property
    def is_local(self) -> bool:
        return self.host == "local"

    @property
    def display_label(self) -> str:
        return self.label or self.host

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "label": self.label,
            "seed_offset": self.seed_offset,
            "n_workers": self.n_workers,
            "user": self.user,
            "repo_path": self.repo_path,
            "venv_python": self.venv_python,
            "ssh_port": self.ssh_port,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShardConfig":
        # Hard removal: ``n_games`` was a per-shard quota in a pre-
        # pull-queue era. In pull mode it had no effect, so allowing
        # it to silently linger in shards-config files would mislead
        # the next reader. Reject up-front with a pointer to the
        # replacement.
        if "n_games" in d:
            raise ValueError(
                f"shard {d.get('host', '<unknown>')!r}: "
                "the 'n_games' field has been removed from "
                "ShardConfig. Total games per iter now comes from "
                "iterate_alphazero.py's --games-per-iteration "
                "(every shard pulls from one shared queue). Drop "
                "the 'n_games' key from your shards-config JSON."
            )
        return cls(
            host=d["host"],
            label=d.get("label", ""),
            seed_offset=int(d.get("seed_offset", 0)),
            n_workers=d.get("n_workers"),
            user=d.get("user"),
            repo_path=d.get("repo_path"),
            venv_python=d.get("venv_python"),
            ssh_port=int(d.get("ssh_port", 22)),
        )

    def __post_init__(self) -> None:
        # Validate up-front — easier to catch a malformed config at
        # load time than after several minutes of self-play.
        if not self.host:
            raise ValueError("host is required")
        if self.seed_offset < 0:
            raise ValueError(
                f"seed_offset must be non-negative, got {self.seed_offset}"
            )
        if not self.is_local:
            missing = [
                name for name in ("user", "repo_path", "venv_python")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"remote shard host={self.host!r} missing fields: "
                    f"{missing}"
                )


@dataclass(frozen=True)
class ShardsConfig:
    """Top-level config: a list of shards plus a small block of
    defaults common to all of them.

    Loaded from JSON. Hand-editing JSON is fine for the 2-3-machine
    setup we have today; if it grows we can migrate to YAML.

    There is no longer a per-config ``total_games`` — the iter total
    is owned by ``iterate_alphazero.py``'s ``--games-per-iteration``
    and threaded into the orchestrator at dispatch time.
    """

    shards: tuple[ShardConfig, ...]

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("shards list is empty")
        # Reject duplicate labels — they'd collide in the npz
        # filenames the orchestrator produces.
        labels = [s.display_label for s in self.shards]
        dupes = {x for x in labels if labels.count(x) > 1}
        if dupes:
            raise ValueError(
                f"duplicate shard labels: {sorted(dupes)}. "
                f"Set explicit ``label`` on each shard with the same "
                f"host (or use distinct hosts)."
            )


def load_shards_config(path: Path) -> ShardsConfig:
    """Read a shards-config JSON file from disk.

    Schema (one example)::

        {
          "shards": [
            {"host": "local"},
            {
              "host": "win-1080ti",
              "user": "gregg",
              "repo_path": "C:\\Users\\gregg\\Projects\\mtg-bot",
              "venv_python": ".venv-py313\\Scripts\\python.exe"
            },
            {
              "host": "macbook",
              "user": "gregg",
              "repo_path": "/Users/gregg/projects/mtg-bot",
              "venv_python": ".venv-py313/bin/python"
            }
          ]
        }

    seed_offset defaults to position * 1_000_000 (idiomatic pattern,
    no collisions for shards under 1M games each) but can be set
    explicitly per shard.

    The total games per iter is NOT in this config — it comes from
    iterate_alphazero.py's ``--games-per-iteration``. A legacy
    ``n_games`` field on any shard entry raises a ``ValueError``
    pointing at the new contract; see ``ShardConfig.from_dict``.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "shards" not in raw:
        raise ValueError(
            f"shards-config {path} missing top-level 'shards' key"
        )
    raw_shards = raw["shards"]
    if not isinstance(raw_shards, list):
        raise ValueError(
            f"shards-config {path}: 'shards' must be a list"
        )
    shards: list[ShardConfig] = []
    for i, item in enumerate(raw_shards):
        if not isinstance(item, dict):
            raise ValueError(
                f"shards-config {path}: shard #{i} must be an object"
            )
        # Auto-assign seed_offset if absent — first shard 0, second
        # shard 1_000_000, etc. Explicit offsets win over the default.
        if "seed_offset" not in item:
            item = {**item, "seed_offset": i * 1_000_000}
        # ``from_dict`` rejects legacy fields (e.g. ``n_games``) with a
        # pointer at the new contract; preferred over ``ShardConfig(
        # **item)`` which would raise a less actionable TypeError.
        shards.append(ShardConfig.from_dict(item))
    return ShardsConfig(shards=tuple(shards))


# -----------------------------------------------------------------
# Runtime orchestration: dispatch each shard's self-play, collect
# the resulting npz paths back to the master.
# -----------------------------------------------------------------


def _now_stamp() -> str:
    """ISO timestamp with no colons (filename-safe)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _shard_npz_path(
    output_dir: Path, run_id_prefix: str, shard: ShardConfig,
    stamp: str,
) -> Path:
    """Per-shard output filename. Includes the shard's display label
    so two shards from the same iter (or two iter runs ending in the
    same second) don't collide. Lives in ``output_dir`` (typically
    the master's ``iterate_data/``)."""
    return output_dir / (
        f"{run_id_prefix}_shard_{shard.display_label}_{stamp}.npz"
    )


def _shard_progress_path(
    progress_dir: Path, run_id_prefix: str, shard: ShardConfig,
) -> Path:
    """Per-shard progress JSON path; the training UI surfaces each
    shard as a separate card. Stable across the run (no timestamp)
    so the UI can re-find the card across polls."""
    return progress_dir / (
        f"{run_id_prefix}_shard_{shard.display_label}.json"
    )


@dataclass(frozen=True)
class SelfPlayJobSpec:
    """All the parameters a shard needs to play games. Same shape on
    every shard — only the per-shard count + seed-offset differs.
    Frozen so it crosses the multiprocessing pickle boundary cleanly
    when iterate hands it to local-shard workers via ``generate_
    training_data_parallel``.

    All path fields (``weights_path``, ``deck_pool_path``,
    ``replays_root``, ``hang_dump_root``) are RELATIVE to the repo
    root by convention, so the same spec works for both local and
    remote shards (which have different absolute paths but the same
    relative layout). Local shards run with cwd at the master's
    repo root; remote shards' wrapper script ``cd``s into the
    remote's repo root before resolving paths.
    """

    weights_path: str
    samples: int
    inner_nodes: int
    hidden_sizes: Optional[tuple[int, ...]]
    base_seed: int
    enable_mulligans: bool
    per_game_timeout_sec: Optional[float]
    deck_pool_path: Optional[str] = None
    with_policy: bool = True
    decision_cap: int = 4000
    sample_every: int = 1
    capture_every: int = 0  # 0 = no replay capture
    nn_version_label: Optional[str] = None
    replays_root: Optional[str] = None
    hang_dump_root: Optional[str] = None
    use_ismcts: bool = False
    use_belief_weighted: bool = False
    # Dual-head architecture string for the shard's NNEvaluator. ``None``
    # means "use NNEvaluator's default", which is "shared" — the back-
    # compat path. Iterate threads the bootstrap-or-promoted version's
    # metadata.architecture in here so remote shards instantiate the
    # right ``_ValuePolicyMLP*`` class.
    architecture: Optional[str] = None
    policy_hidden_sizes: Optional[tuple[int, ...]] = None
    # Slot-mapping convention the bootstrap weights were trained
    # against — threaded into the shard's NNPimcConfig so PUCT prior
    # lookups and policy-target encoding agree on which decision goes
    # to which slot. ``None`` defaults to NNEvaluator's
    # ``"positional_v1"`` (v13/v15/v16 metadata predates the field and
    # back-compat-defaults correctly). v17+ training under
    # ``canonical_v2`` requires this to round-trip across the SSH
    # spec boundary or remote shards silently fall back to v1 and
    # mix v1-encoded data into v2-trained weights.
    action_encoding_version: Optional[str] = None
    # Bundle of priority short-circuit / filter heuristics applied by
    # ``MtgGameStateAdapter`` during search. Self-play default = every
    # flag ON (see ``HeuristicsConfig`` defaults). Per-heuristic
    # rationale + acceptable-failure cases at SEARCH_LIMITATIONS
    # entries #6, #7, #9, #10, #11.
    heuristics: HeuristicsConfig = field(default_factory=HeuristicsConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict. Used by ``run_self_play_shard.py``
        to round-trip the spec across the SSH boundary — the remote
        receives the spec via ``--spec-json '<base64-encoded JSON>'``
        and reconstructs.

        Path fields are normalized to forward slashes here so the
        wire format is platform-neutral. iterate runs on Windows and
        produces ``tmp/foo\\bar.pt`` from ``str(Path(...))``; the Mac
        remote then opens a literal filename with embedded ``\\``
        characters and crashes. Forward slashes work on both Windows
        (cmd.exe + Python both accept them) and POSIX, so we settle
        on them universally for the wire format.
        """
        def _posix(p):
            return p.replace("\\", "/") if isinstance(p, str) else p
        return {
            "weights_path": _posix(self.weights_path),
            "samples": self.samples,
            "inner_nodes": self.inner_nodes,
            "hidden_sizes": (
                list(self.hidden_sizes) if self.hidden_sizes is not None
                else None
            ),
            "base_seed": self.base_seed,
            "enable_mulligans": self.enable_mulligans,
            "per_game_timeout_sec": self.per_game_timeout_sec,
            "deck_pool_path": _posix(self.deck_pool_path),
            "with_policy": self.with_policy,
            "decision_cap": self.decision_cap,
            "sample_every": self.sample_every,
            "capture_every": self.capture_every,
            "nn_version_label": self.nn_version_label,
            "replays_root": _posix(self.replays_root),
            "hang_dump_root": _posix(self.hang_dump_root),
            "use_ismcts": self.use_ismcts,
            "use_belief_weighted": self.use_belief_weighted,
            "heuristics": self.heuristics.to_dict(),
            "architecture": self.architecture,
            "policy_hidden_sizes": (
                list(self.policy_hidden_sizes)
                if self.policy_hidden_sizes is not None else None
            ),
            "action_encoding_version": self.action_encoding_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelfPlayJobSpec":
        hs = d.get("hidden_sizes")
        # ``heuristics`` is a nested dict on the wire; missing key →
        # production self-play default (every flag True). Older payloads
        # written before this field existed produce that same default.
        h = d.get("heuristics")
        heuristics = (
            HeuristicsConfig.from_dict(h) if h is not None
            else HeuristicsConfig()
        )
        return cls(
            weights_path=d["weights_path"],
            samples=int(d["samples"]),
            inner_nodes=int(d["inner_nodes"]),
            hidden_sizes=tuple(hs) if hs is not None else None,
            base_seed=int(d["base_seed"]),
            enable_mulligans=bool(d["enable_mulligans"]),
            per_game_timeout_sec=d.get("per_game_timeout_sec"),
            deck_pool_path=d.get("deck_pool_path"),
            with_policy=bool(d.get("with_policy", True)),
            decision_cap=int(d.get("decision_cap", 4000)),
            sample_every=int(d.get("sample_every", 1)),
            capture_every=int(d.get("capture_every", 0)),
            nn_version_label=d.get("nn_version_label"),
            replays_root=d.get("replays_root"),
            hang_dump_root=d.get("hang_dump_root"),
            use_ismcts=bool(d.get("use_ismcts", False)),
            use_belief_weighted=bool(d.get("use_belief_weighted", False)),
            heuristics=heuristics,
            architecture=d.get("architecture"),
            policy_hidden_sizes=(
                tuple(d["policy_hidden_sizes"])
                if d.get("policy_hidden_sizes") is not None else None
            ),
            action_encoding_version=d.get("action_encoding_version"),
        )


@dataclass(frozen=True)
class EvalJobSpec:
    """Eval-mode counterpart to ``SelfPlayJobSpec``: parameters for an
    acceptance-gate match between two NN versions across the cluster.

    Two weights paths because every eval game pits ``new`` (seat 0)
    against ``old`` (seat 1). ``old_weights_path=None`` falls back to a
    heuristic opponent — useful for sanity-check evals against the
    baseline. Path fields normalize to forward slashes in
    ``to_dict()`` for the same cross-platform reasons documented on
    ``SelfPlayJobSpec``.
    """

    new_weights_path: str
    old_weights_path: Optional[str]
    samples: int
    inner_nodes: int
    hidden_sizes: Optional[tuple[int, ...]]
    base_seed: int
    enable_mulligans: bool
    per_game_timeout_sec: Optional[float]
    deck_pool_path: Optional[str] = None
    capture_every: int = 0
    nn_new_version_label: Optional[str] = None
    nn_old_version_label: Optional[str] = None
    replays_root: Optional[str] = None
    hang_dump_root: Optional[str] = None
    opponent_label: str = "previous"
    use_ismcts: bool = False
    use_belief_weighted: bool = False
    # Architecture strings for the new and old NN sides of the eval
    # match. ``None`` defaults to NNEvaluator's "shared". Iterate threads
    # the metadata.architecture for each version in here so the eval
    # workers instantiate the right dual-head class for each side.
    new_architecture: Optional[str] = None
    old_architecture: Optional[str] = None
    policy_hidden_sizes: Optional[tuple[int, ...]] = None
    # Per-side slot-mapping convention the new / old weights were
    # trained against. Separate fields because cross-version evals
    # (e.g. v17[canonical_v2] vs v13[positional_v1]) genuinely need
    # each side to encode under its own version — using the wrong
    # version on either side scrambles that side's PUCT priors. ``None``
    # defaults to NNEvaluator's positional_v1, matching the v13/v15/v16
    # metadata-missing back-compat path.
    new_action_encoding_version: Optional[str] = None
    old_action_encoding_version: Optional[str] = None
    # Bundle of priority short-circuit / filter heuristics. Eval
    # default = every flag OFF (``HeuristicsConfig.disabled()``):
    # match outcomes determine acceptance, so the strength-vs-
    # throughput tradeoff that justifies the heuristics in self-play
    # does not apply here. See SEARCH_LIMITATIONS entries #6, #7, #9,
    # #10, #11 for the per-heuristic rationale and the niche plays
    # the eval-default avoids skipping.
    heuristics: HeuristicsConfig = field(
        default_factory=HeuristicsConfig.disabled,
    )

    def to_dict(self) -> dict[str, Any]:
        def _posix(p):
            return p.replace("\\", "/") if isinstance(p, str) else p
        return {
            "new_weights_path": _posix(self.new_weights_path),
            "old_weights_path": _posix(self.old_weights_path),
            "samples": self.samples,
            "inner_nodes": self.inner_nodes,
            "hidden_sizes": (
                list(self.hidden_sizes) if self.hidden_sizes is not None
                else None
            ),
            "base_seed": self.base_seed,
            "enable_mulligans": self.enable_mulligans,
            "per_game_timeout_sec": self.per_game_timeout_sec,
            "deck_pool_path": _posix(self.deck_pool_path),
            "capture_every": self.capture_every,
            "nn_new_version_label": self.nn_new_version_label,
            "nn_old_version_label": self.nn_old_version_label,
            "replays_root": _posix(self.replays_root),
            "hang_dump_root": _posix(self.hang_dump_root),
            "opponent_label": self.opponent_label,
            "use_ismcts": self.use_ismcts,
            "use_belief_weighted": self.use_belief_weighted,
            "heuristics": self.heuristics.to_dict(),
            "new_architecture": self.new_architecture,
            "old_architecture": self.old_architecture,
            "policy_hidden_sizes": (
                list(self.policy_hidden_sizes)
                if self.policy_hidden_sizes is not None else None
            ),
            "new_action_encoding_version": (
                self.new_action_encoding_version
            ),
            "old_action_encoding_version": (
                self.old_action_encoding_version
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalJobSpec":
        hs = d.get("hidden_sizes")
        # ``heuristics`` round-trips as a nested dict. Missing key →
        # eval default (every flag OFF). Older payloads written
        # before this field existed produce that same default.
        h = d.get("heuristics")
        heuristics = (
            HeuristicsConfig.from_dict(h) if h is not None
            else HeuristicsConfig.disabled()
        )
        return cls(
            new_weights_path=d["new_weights_path"],
            old_weights_path=d.get("old_weights_path"),
            samples=int(d["samples"]),
            inner_nodes=int(d["inner_nodes"]),
            hidden_sizes=tuple(hs) if hs is not None else None,
            base_seed=int(d["base_seed"]),
            enable_mulligans=bool(d["enable_mulligans"]),
            per_game_timeout_sec=d.get("per_game_timeout_sec"),
            deck_pool_path=d.get("deck_pool_path"),
            capture_every=int(d.get("capture_every", 0)),
            nn_new_version_label=d.get("nn_new_version_label"),
            nn_old_version_label=d.get("nn_old_version_label"),
            replays_root=d.get("replays_root"),
            hang_dump_root=d.get("hang_dump_root"),
            opponent_label=d.get("opponent_label", "previous"),
            use_ismcts=bool(d.get("use_ismcts", False)),
            use_belief_weighted=bool(d.get("use_belief_weighted", False)),
            heuristics=heuristics,
            new_architecture=d.get("new_architecture"),
            old_architecture=d.get("old_architecture"),
            policy_hidden_sizes=(
                tuple(d["policy_hidden_sizes"])
                if d.get("policy_hidden_sizes") is not None else None
            ),
            new_action_encoding_version=d.get(
                "new_action_encoding_version"
            ),
            old_action_encoding_version=d.get(
                "old_action_encoding_version"
            ),
        )


def run_shard_local(
    shard: ShardConfig,
    spec: SelfPlayJobSpec,
    *,
    n_games: int,
    progress_dir: Path,
    output_dir: Path,
    run_id_prefix: str,
    stamp: str,
) -> Path:
    """Run a single shard in-process with a fixed pre-allocated game
    count. Builds the same NNPimcConfig + DataGenProgress + (optional)
    ReplayCaptureConfig that iterate_alphazero's ``_self_play``
    builds, then calls ``generate_training_data_parallel`` for the
    shard's slice of the workload.

    ``n_games`` is the explicit per-shard quota for this push-mode
    invocation. The pull-mode runner (``run_shard_pull``) is what the
    orchestrator uses now; this function survives for tests and
    isolated direct callers that still want push-mode semantics.

    Returns the path of the produced ``.npz``.
    """
    # Imports are local to keep the module's import-cost low for
    # the common "load shards-config to validate" path that doesn't
    # need any of the training internals.
    from mtg_bot.decks import decks_from_json
    from mtg_bot.training import (
        DataGenProgress,
        NNPimcConfig,
        generate_training_data_parallel,
        save_dataset,
    )
    from mtg_bot.training.data_gen import ReplayCaptureConfig
    from mtg_bot.training.trainer import auto_device

    cfg = NNPimcConfig(
        weights_path=spec.weights_path,
        samples=spec.samples,
        inner_nodes=spec.inner_nodes,
        batch_size=32,
        with_policy=spec.with_policy,
        puct=True,
        hidden_sizes=spec.hidden_sizes,
        architecture=spec.architecture,
        policy_hidden_sizes=spec.policy_hidden_sizes,
        action_encoding_version=spec.action_encoding_version,
        use_ismcts=spec.use_ismcts,
        use_belief_weighted=spec.use_belief_weighted,
        # Resolve locally on this shard's host: master picks "cuda",
        # Mac picks "mps", Win10 picks "cuda" (or "cpu" if no GPU).
        # Tests construct NNPimcConfig() directly without ``device=``
        # and stay on the CPU default — production goes through this
        # spec-decoding path which always opts in.
        device=auto_device(),
    )

    deck_pool = None
    if spec.deck_pool_path is not None:
        deck_pool = decks_from_json(Path(spec.deck_pool_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    output_path = _shard_npz_path(
        output_dir, run_id_prefix, shard, stamp,
    )
    progress_path = _shard_progress_path(
        progress_dir, run_id_prefix, shard,
    )

    # Replay capture — opt-in via spec.capture_every. Each shard
    # writes to its own subdir under the master's replays-root so
    # the training UI's per-card 'View N games' link finds them.
    replay_config: Optional[ReplayCaptureConfig] = None
    replays_subdir: Optional[str] = None
    if spec.capture_every > 0 and spec.replays_root is not None:
        run_id = (
            f"{run_id_prefix}_shard_{shard.display_label}"
        )
        replays_subdir = str(
            Path(spec.replays_root) / run_id
        )
        Path(replays_subdir).mkdir(parents=True, exist_ok=True)
        replay_config = ReplayCaptureConfig(
            save_dir=replays_subdir,
            capture_every=spec.capture_every,
            nn_weights_path=spec.weights_path,
            nn_version_label=spec.nn_version_label,
            nn_hidden_sizes=spec.hidden_sizes,
            nn_with_policy=spec.with_policy,
        )

    hang_dump_dir: Optional[str] = None
    if spec.hang_dump_root is not None:
        hang_dump_dir = str(
            Path(spec.hang_dump_root)
            / f"{run_id_prefix}_shard_{shard.display_label}"
        )

    progress = DataGenProgress(
        progress_path=progress_path,
        run_id=f"{run_id_prefix}_shard_{shard.display_label}",
        n_games_total=n_games,
        n_workers=(
            shard.n_workers if shard.n_workers is not None
            else max(1, 0)  # generate_training_data_parallel will
                            # default-fill from cpu_count
        ),
        command="distributed_self_play (local shard)",
        notes=(
            f"distributed shard {shard.display_label} "
            f"({n_games} games)"
        ),
        replays_dir=replays_subdir,
    )

    try:
        features, outcomes, policy_targets, action_masks = (
            generate_training_data_parallel(
                p0_factory=cfg.build,
                p1_factory=cfg.build,
                n_games=n_games,
                base_seed=spec.base_seed + shard.seed_offset,
                n_workers=shard.n_workers,
                with_policy=True,
                decision_cap=spec.decision_cap,
                sample_every=spec.sample_every,
                enable_mulligans=spec.enable_mulligans,
                per_game_timeout_sec=spec.per_game_timeout_sec,
                progress=progress,
                deck_pool=deck_pool,
                replay_config=replay_config,
                hang_dump_dir=hang_dump_dir,
                heuristics=spec.heuristics,
                action_encoding_version=(
                    spec.action_encoding_version or "positional_v1"
                ),
            )
        )
    except BaseException:
        progress.finalize(status="failed")
        raise
    progress.finalize(status="completed")

    save_dataset(
        features, outcomes, output_path,
        policy_targets=policy_targets, action_masks=action_masks,
    )
    return output_path


class _AggregateProgressPoller:
    """Periodically reads all per-shard progress JSONs in
    ``progress_dir`` matching ``<run_id>_shard_*.json`` and writes a
    rolled-up ``<run_id>.json`` summing the counters across shards.

    The training UI (``training_progress/`` viewer) globs that
    directory for run cards; without this aggregate it would render
    one card per shard, each showing ``X / iter_total`` where X is
    that shard's share. The aggregate gives the user one coherent
    "iter card" with the cluster-wide totals; individual per-shard
    files still exist for diagnostic drill-down (the UI filters them
    out of the top-level listing by the ``_shard_`` substring).

    ``kind`` is ``"data_gen"`` for self-play or ``"eval"`` — controls
    which set of summed fields ends up in the JSON. Both share the
    progress / total / status / elapsed shape so the UI renderer
    doesn't need to special-case distributed runs.
    """

    _SUM_FIELDS = (
        "n_games_done", "n_samples", "n_workers", "crashed",
        "timed_out",
        "p0_wins", "p1_wins", "draws",
        "wins_new", "wins_old",
    )

    def __init__(
        self,
        progress_dir: Path,
        run_id: str,
        n_games_total: int,
        kind: str,
        *,
        opponent_label: Optional[str] = None,
        interval_s: float = 2.0,
    ) -> None:
        self._progress_dir = Path(progress_dir)
        self._run_id = run_id
        self._n_games_total = n_games_total
        self._kind = kind
        self._opponent_label = opponent_label
        self._interval_s = interval_s
        self._aggregate_path = self._progress_dir / f"{run_id}.json"
        self._started_perf = time.monotonic()
        self._started_iso = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._write("running")
        self._thread = threading.Thread(
            target=self._loop,
            name=f"agg-progress-{self._run_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, status: str) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._write(status)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._write("running")

    def _write(self, status: str) -> None:
        from mtg_bot.training.progress import atomic_write_json
        sums = {f: 0 for f in self._SUM_FIELDS}
        machines: list[dict[str, Any]] = []
        ended_ats: list[str] = []
        decisions_weighted_total = 0.0
        pattern = f"{self._run_id}_shard_*.json"
        for path in sorted(self._progress_dir.glob(pattern)):
            # Eval shards write a sibling ``..._shard_<label>_result.json``
            # carrying the same wins/draws/crashed counts. Including it
            # in the sum would double-count every remote shard's games.
            if path.stem.endswith("_result"):
                continue
            try:
                shard = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for f in self._SUM_FIELDS:
                v = shard.get(f, 0)
                if isinstance(v, (int, float)):
                    sums[f] += v
            shard_done = shard.get("n_games_done", 0)
            shard_avg_dec = shard.get("avg_decisions", 0)
            if (
                isinstance(shard_done, (int, float))
                and isinstance(shard_avg_dec, (int, float))
            ):
                decisions_weighted_total += shard_avg_dec * shard_done
            label = path.stem.split("_shard_", 1)[-1]
            shard_elapsed = shard.get("elapsed_seconds", 0)
            shard_done_int = shard.get("n_games_done", 0)
            # Per-host performance stat: secs/game from THIS shard's own
            # elapsed time, not the aggregator's wall clock. The shard's
            # elapsed reflects only that machine's actual work; the
            # aggregate elapsed includes orchestrator startup overhead
            # and is the same for every machine. Surfaced as a precomputed
            # field so the UI can render without doing the divide-by-zero
            # dance per row.
            if (
                isinstance(shard_elapsed, (int, float))
                and isinstance(shard_done_int, (int, float))
                and shard_done_int > 0
            ):
                per_game = float(shard_elapsed) / float(shard_done_int)
            else:
                per_game = None
            machines.append({
                "label": label,
                "status": shard.get("status", "unknown"),
                "n_games_done": shard.get("n_games_done", 0),
                "n_workers": shard.get("n_workers", 0),
                "crashed": shard.get("crashed", 0),
                "timed_out": shard.get("timed_out", 0),
                "elapsed_seconds": (
                    float(shard_elapsed)
                    if isinstance(shard_elapsed, (int, float))
                    else 0.0
                ),
                "per_game_seconds": per_game,
            })
            if shard.get("ended_at"):
                ended_ats.append(shard["ended_at"])
        elapsed = max(time.monotonic() - self._started_perf, 0.0)
        done = sums["n_games_done"]
        total = self._n_games_total
        avg = (elapsed / done) if done > 0 else 0.0
        eta = avg * max(total - done, 0)
        games_per_hour = (done / elapsed * 3600.0) if elapsed > 0 else 0.0
        ended_at = (
            max(ended_ats) if ended_ats and status != "running" else None
        )
        payload: dict[str, Any] = {
            "run_id": self._run_id,
            "kind": self._kind,
            "status": status,
            "started_at": self._started_iso,
            "ended_at": ended_at,
            "n_games_total": int(total),
            "n_games_done": int(done),
            "n_workers": int(sums["n_workers"]),
            "crashed": int(sums["crashed"]),
            "timed_out": int(sums["timed_out"]),
            "elapsed_seconds": float(elapsed),
            "estimated_remaining_seconds": float(eta),
            "per_game_seconds_avg": float(avg),
            "games_per_hour": float(games_per_hour),
            "machines": machines,
        }
        if self._kind == "data_gen":
            samples_per_hour = (
                sums["n_samples"] / elapsed * 3600.0 if elapsed > 0 else 0.0
            )
            payload["n_samples"] = int(sums["n_samples"])
            payload["p0_wins"] = int(sums["p0_wins"])
            payload["p1_wins"] = int(sums["p1_wins"])
            payload["draws"] = int(sums["draws"])
            payload["samples_per_hour"] = float(samples_per_hour)
        else:
            decided = sums["wins_new"] + sums["wins_old"]
            avg_decisions = (
                decisions_weighted_total / done if done > 0 else 0.0
            )
            payload["wins_new"] = int(sums["wins_new"])
            payload["wins_old"] = int(sums["wins_old"])
            payload["draws"] = int(sums["draws"])
            payload["win_rate"] = (
                sums["wins_new"] / decided if decided > 0 else 0.0
            )
            payload["avg_decisions"] = float(avg_decisions)
            if self._opponent_label is not None:
                payload["opponent_label"] = self._opponent_label
        atomic_write_json(self._aggregate_path, payload)


class _EarlyStopPoller:
    """Watches the per-shard progress JSONs and POSTs
    ``/queue/close/<run_id>`` once the cluster has accumulated
    ``target_games`` successful completions. "Successful" =
    ``n_games_done - crashed - timed_out`` summed across shards.

    Pairs with the over-queue mechanism in
    :func:`run_distributed_self_play` and :func:`run_distributed_eval`:
    those initialize the queue with ``target_games + slack_extra``;
    this poller closes the queue early once ``target_games`` are
    actually in. Stragglers' games are abandoned because we already
    have what we need. Defends against the long-tail "499/500 stuck
    for 5 min" pattern observed on the cluster.

    The closing POST fires AT MOST ONCE — once the threshold trips,
    subsequent ticks short-circuit. Network blips between threshold
    and POST are absorbed by the same retry helper used elsewhere;
    if the master is genuinely unreachable the poller logs a warning
    and gives up (the queue will drain on its own once the slack
    indices are consumed, just slower).

    Read path is filesystem only — no shard liveness check. A hung
    shard's stale JSON simply stops contributing to the sum; the
    other shards' progress carries the threshold over the line.

    Reads only shard JSONs whose ``started_at`` is at or after the
    poller's instantiation time. Stale JSONs from prior runs with
    the same ``run_id`` are silently ignored — without this filter,
    rerunning an iter (e.g. ``iter_003_self_play_v13``) with shard
    JSONs left over from a previous attempt makes the very first
    tick read the prior run's ``n_games_done`` and POST close
    immediately, dropping the new run's shards before they begin
    (observed 2026-05-05 with ``--queue-slack-pct 0.1``). Shards
    without a ``started_at`` field are also skipped, since we can't
    tell which run they belong to.
    """

    _SUM_FIELDS = ("n_games_done", "crashed", "timed_out")

    def __init__(
        self,
        progress_dir: Path,
        run_id: str,
        target_games: int,
        queue_url: str,
        *,
        interval_s: float = 3.0,
        urlopen=None,
    ) -> None:
        self._progress_dir = Path(progress_dir)
        self._run_id = run_id
        self._target_games = int(target_games)
        self._queue_url = queue_url.rstrip("/")
        self._interval_s = interval_s
        # Injectable for tests so we don't need a real HTTP server.
        self._urlopen = urlopen
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._close_failed = False
        # Stale-JSON filter: any shard JSON written before this
        # timestamp is from a prior run with the same ``run_id`` and
        # must NOT count toward the early-stop threshold. ISO 8601
        # with seconds precision matches what DataGenProgress /
        # EvalProgress write in the per-shard JSON's ``started_at``;
        # lexical comparison on those strings is order-preserving.
        self._start_iso = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        # Set on the first stale-JSON skip so we log once per run
        # rather than on every tick.
        self._logged_stale_skip = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name=f"early-stop-{self._run_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def closed(self) -> bool:
        """True once this poller has successfully POSTed
        ``/queue/close``. Tests inspect this; orchestrator code
        doesn't need it (the close is a fire-and-forget signal)."""
        return self._closed

    def _loop(self) -> None:
        # First check fires immediately so unit tests that pre-populate
        # the JSONs can observe the close without waiting an interval.
        # Subsequent ticks wait first, so we don't busy-loop.
        first = True
        while True:
            if not first and self._stop_event.wait(self._interval_s):
                return
            first = False
            if self._closed:
                return
            successful = self._read_successful()
            if successful >= self._target_games:
                self._send_close()
                return

    def _read_successful(self) -> int:
        """Sum ``n_games_done - crashed - timed_out`` across all
        ``<run_id>_shard_*.json`` files in ``progress_dir``. Skips
        unreadable / partially-written files (the next tick will
        catch them). Skips ``..._result.json`` files to avoid double-
        counting eval shards (mirrors _AggregateProgressPoller).

        Also skips shard JSONs whose ``started_at`` is missing or
        predates this poller's instantiation — those are leftovers
        from a prior run with the same ``run_id`` and would
        prematurely trip the close (see class docstring)."""
        sums = {f: 0 for f in self._SUM_FIELDS}
        pattern = f"{self._run_id}_shard_*.json"
        stale_paths: list[str] = []
        for path in self._progress_dir.glob(pattern):
            if path.stem.endswith("_result"):
                continue
            try:
                shard = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            shard_started = shard.get("started_at")
            # ISO 8601 strings compare lexically. A missing or older
            # timestamp means this JSON predates the current run.
            if (
                not isinstance(shard_started, str)
                or shard_started < self._start_iso
            ):
                stale_paths.append(path.name)
                continue
            for f in self._SUM_FIELDS:
                v = shard.get(f, 0)
                if isinstance(v, (int, float)):
                    sums[f] += int(v)
        if stale_paths and not self._logged_stale_skip:
            self._logged_stale_skip = True
            print(
                f"[early-stop] {self._run_id}: ignoring "
                f"{len(stale_paths)} stale shard JSON(s) from a prior "
                f"run (started_at < {self._start_iso}): "
                f"{', '.join(sorted(stale_paths))}",
                file=sys.stderr,
            )
        return sums["n_games_done"] - sums["crashed"] - sums["timed_out"]

    def _send_close(self) -> None:
        """Best-effort POST to ``/queue/close/<run_id>``. On failure
        the next tick retries — but :meth:`_loop` exits after a
        successful close, so a permanent failure is logged once and
        we move on (the queue will drain naturally)."""
        import urllib.request
        url = f"{self._queue_url}/queue/close/{self._run_id}"
        req = urllib.request.Request(
            url, data=b"", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            if self._urlopen is not None:
                resp = self._urlopen(req)
                status = getattr(resp, "status", 200)
            else:
                resp = _urlopen_with_retry(req)
                status = resp.status
            if 200 <= status < 300:
                self._closed = True
            else:
                self._close_failed = True
                print(
                    f"[early-stop] {url} returned HTTP {status}; "
                    f"queue will drain naturally.",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            self._close_failed = True
            print(
                f"[early-stop] failed to POST {url}: {exc}; queue "
                f"will drain naturally.",
                file=sys.stderr,
            )


class _StopCheckPoller:
    """Watches a user-supplied ``stop_check`` predicate and POSTs
    ``/queue/close/<run_id>`` once it fires. Companion to
    :class:`_EarlyStopPoller`: the early-stop poller closes the
    queue when enough successful results are in (statistical /
    target-met early termination); this poller closes the queue
    when the operator clicked Stop on the training UI (or any
    upstream stop sentinel went off, e.g. iterate's cycle-level
    sentinel propagating through to the in-flight eval).
    The two are independent — an iter run can have either, both,
    or neither poller active simultaneously without conflict.
    Both ultimately make the same fire-and-forget POST to
    ``/queue/close/<run_id>``; the queue server treats repeated
    closes as a no-op so two concurrent close POSTs are safe.

    Returns to the orchestrator's ``finally`` cleanup just like
    ``_EarlyStopPoller`` — the close is fire-and-forget, the
    poller doesn't need to be drained or joined for correctness
    (best-effort ``stop()`` just sets the event so the thread
    exits quickly).

    The default poll interval is short (0.5s) because the
    user-perceived latency of "I clicked Stop" → "the eval
    actually stopped dispatching new games" is the metric this
    poller exists to optimize. The early-stop poller's 3s tick
    is fine because its threshold is filesystem-derived, but a
    UI-driven stop should feel immediate.

    ``stop_check`` is called on the poller thread; it must be
    cheap and thread-safe. The standard
    ``mtg_bot.training.stop_signal.stop_requested`` predicate is
    a tiny ``Path.exists()`` probe and satisfies both.
    """

    def __init__(
        self,
        *,
        stop_check: Callable[[], bool],
        run_id: str,
        queue_url: str,
        interval_s: float = 0.5,
        urlopen=None,
    ) -> None:
        self._stop_check = stop_check
        self._run_id = run_id
        self._queue_url = queue_url.rstrip("/")
        self._interval_s = interval_s
        self._urlopen = urlopen
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._close_failed = False
        # Set on first observation that ``stop_check`` returned True;
        # helps callers / tests assert "the poller saw the stop"
        # independent of whether the close POST succeeded.
        self._stop_observed = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name=f"stop-check-{self._run_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to exit. Idempotent. Safe to
        call from the orchestrator's ``finally`` even if
        :meth:`start` was never called (test paths)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def closed(self) -> bool:
        """True once this poller has successfully POSTed
        ``/queue/close``. Tests inspect this; orchestrator code
        doesn't need it (the close is fire-and-forget)."""
        return self._closed

    @property
    def stop_observed(self) -> bool:
        """True once ``stop_check()`` returned True at least once.
        Distinct from :attr:`closed` because the close POST may fail
        even though the stop was observed; callers can use this to
        decide whether the run should be reported as ``stopped`` vs
        ``completed`` independent of whether the queue close
        succeeded."""
        return self._stop_observed

    def _loop(self) -> None:
        # Tick once immediately so a stop sentinel that's already
        # present at start-up is observed without waiting an interval.
        first = True
        while True:
            if not first and self._stop_event.wait(self._interval_s):
                return
            first = False
            if self._closed:
                return
            try:
                fired = self._stop_check()
            except Exception as exc:  # noqa: BLE001
                # A misbehaving stop_check shouldn't kill the
                # orchestrator. Log once and exit the polling loop;
                # the run will continue normally.
                print(
                    f"[stop-check] {self._run_id}: stop_check raised "
                    f"{type(exc).__name__}: {exc}; abandoning poll "
                    f"thread (run will continue without stop-check "
                    f"propagation).",
                    file=sys.stderr,
                )
                return
            if not fired:
                continue
            self._stop_observed = True
            self._send_close()
            return

    def _send_close(self) -> None:
        """Best-effort POST to ``/queue/close/<run_id>``. On failure
        we log and give up — the run will drain naturally as the
        existing queue indices empty (slower than ideal but the user
        already got a stop sentinel, so the iterate driver will
        short-circuit at the next stage boundary regardless)."""
        import urllib.request
        url = f"{self._queue_url}/queue/close/{self._run_id}"
        req = urllib.request.Request(
            url, data=b"", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            if self._urlopen is not None:
                resp = self._urlopen(req)
                status = getattr(resp, "status", 200)
            else:
                resp = _urlopen_with_retry(req)
                status = resp.status
            if 200 <= status < 300:
                self._closed = True
            else:
                self._close_failed = True
                print(
                    f"[stop-check] {url} returned HTTP {status}; "
                    f"queue will drain naturally.",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            self._close_failed = True
            print(
                f"[stop-check] failed to POST {url}: {exc}; queue "
                f"will drain naturally.",
                file=sys.stderr,
            )


@dataclass(frozen=True)
class ShardSelfPlayReport:
    """Outcome of a single self-play shard from the orchestrator's
    point of view. Returned by :func:`run_shard_remote` and the local
    runner wrapper inside :func:`run_distributed_self_play`.

    Per-shard SSH / rsync flows used to raise on the first non-zero
    exit, killing the entire master process and discarding the OTHER
    shards' completed work. The orchestrator now catches and converts
    every per-shard failure into a ``ShardSelfPlayReport`` so siblings'
    data is preserved; the aggregator drops the failed shard's
    contribution and continues.

    ``npz_paths`` is the list of master-local npz files that were
    successfully scp'd back from the shard — partial data is allowed
    (a shard that crashed mid-run after producing N batches is still
    worth N batches of training data). ``error`` is a short
    human-readable failure description; ``error_stage`` names which
    pipeline stage failed (e.g. ``"self_play"``, ``"weight_push"``)
    so the user's stderr summary points at the right place.
    """

    shard_label: str
    success: bool
    npz_paths: list[Path]
    error: Optional[str] = None
    error_stage: Optional[str] = None


@dataclass(frozen=True)
class ShardEvalReport:
    """Eval-side counterpart to :class:`ShardSelfPlayReport`. ``counts``
    is the parsed result-JSON dict (n_games / wins_new / wins_old /
    draws / crashed). On failure it is an all-zeros dict so the
    aggregator can sum it without special-casing.
    """

    shard_label: str
    success: bool
    counts: dict[str, int]
    error: Optional[str] = None
    error_stage: Optional[str] = None


def _mark_shard_progress_failed(
    progress_path: Path, *, shard_label: str, note: str,
) -> None:
    """Write (or merge into) the per-shard progress JSON to mark it
    ``status="failed"`` so the training UI's per-shard card shows the
    failure rather than freezing on its last "running" snapshot.

    If the file exists, merge — preserves the last counters the poller
    pulled back so the user can still see "shard finished N/M games
    before the SSH wrapper died". If absent, write a minimal payload
    just to register the failure. Errors here are swallowed: a master
    that can't write a status file is a much smaller problem than a
    master that abandons live shards mid-iter, and the loud stderr log
    in the caller already records what went wrong.
    """
    payload: dict[str, Any]
    try:
        if progress_path.is_file():
            payload = json.loads(
                progress_path.read_text(encoding="utf-8"),
            )
        else:
            payload = {
                "run_id": progress_path.stem,
                "kind": "data_gen",
                "n_games_total": 0,
                "n_games_done": 0,
            }
        payload["status"] = "failed"
        payload["error_note"] = note
        payload["ended_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(payload), encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[shard {shard_label}] failed to mark progress JSON "
            f"as failed at {progress_path}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )


def _loopback_queue_url(queue_url: str) -> str:
    """Rewrite the queue URL's host to ``127.0.0.1`` so the master's
    own local shard skips Tailscale name resolution. On Windows that
    resolution attempts IPv6 first against the tailnet hostname and
    burns ~4s per take before falling back; ``localhost`` also has a
    ~2s penalty. Loopback IP literal hits ~12ms. Remote shards keep
    the original URL — they reach master via the tailnet."""
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(queue_url)
    netloc = f"127.0.0.1:{parts.port}" if parts.port else "127.0.0.1"
    return urlunsplit((
        parts.scheme, netloc, parts.path,
        parts.query, parts.fragment,
    ))


def run_distributed_self_play(
    shards_config: ShardsConfig,
    spec: SelfPlayJobSpec,
    *,
    progress_dir: Path,
    output_dir: Path,
    run_id_prefix: str,
    total_games: int,
    queue_url: str,
    bundle_url: Optional[str] = None,
    slack_extra: int = 0,
    shard_ssh_timeout_s: float = _DEFAULT_SHARD_SSH_TIMEOUT_S,
    stop_check: Optional[Callable[[], bool]] = None,
) -> list[Path]:
    """Pull-model orchestrator. Initializes a queue of
    ``total_games + slack_extra`` on master, dispatches every shard
    concurrently to pull from it, and returns the list of every
    ``.npz`` produced. Master's training step globs ``output_dir`` so
    the union becomes the iter's training mix automatically.

    Each shard runs ``run_shard_pull``, which opens a pool of
    ``n_workers`` worker processes and continuously feeds them games
    one at a time from ``/queue/take``. As any worker finishes its
    game, the shard pulls the next available game and dispatches it
    to that worker. Slow machines pull fewer total games; fast ones
    pull more — wall ≈ ``total_games / sum(per_machine_throughput)``
    instead of being capped by the slowest shard.

    ``slack_extra`` enables straggler tolerance: the queue is
    over-sized so the orchestrator can early-stop once
    ``total_games`` successful completions are in, abandoning slack.
    Set to 0 (default) to retain the legacy "drain the whole queue"
    behavior.

    ``stop_check``: optional predicate polled by a background thread.
    On True, the orchestrator POSTs ``/queue/close/<run_id_prefix>``
    so the queue stops dispatching new games. Shards' workers see
    an empty queue at the next take, finish their in-flight games,
    flush their final batches, and shut down; the orchestrator
    returns whatever npz paths landed before the stop. Mirrors
    :func:`run_match_parallel`'s ``stop_check`` shape so iterate
    can pass the same callable through both paths. Defaults to
    ``None`` (no stop-check), preserving legacy behavior.
    """
    import threading as _threading
    import urllib.request
    import json as _json

    stamp = _now_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    queued_games = int(total_games) + max(0, int(slack_extra))

    # Initialize the queue. The run_id_prefix doubles as the queue
    # key — same value workers send back to /queue/take.
    init_url = f"{queue_url.rstrip('/')}/queue/init"
    init_body = _json.dumps({
        "run_id": run_id_prefix,
        "total_games": queued_games,
    }).encode("utf-8")
    req = urllib.request.Request(
        init_url, data=init_body,
        headers={"Content-Type": "application/json"},
    )
    # Tight per-attempt timeout + bounded retries — see comment on
    # ``_QUEUE_HTTP_ATTEMPT_TIMEOUT_S`` for the rationale.
    init_resp = _urlopen_with_retry(req)
    if init_resp.status != 200:
        raise RuntimeError(
            f"queue init failed at {init_url}: HTTP {init_resp.status}"
        )

    output_paths: list[Path] = []
    output_paths_lock = _threading.Lock()
    reports: list[ShardSelfPlayReport] = []
    reports_lock = _threading.Lock()

    def _local_runner(shard: ShardConfig) -> ShardSelfPlayReport:
        """Local shard wrapper. Mirrors :func:`run_shard_remote`'s
        contract: catches per-shard failures and converts them into
        a :class:`ShardSelfPlayReport`. Without this, a local-shard
        crash takes the whole iter down even when remotes finished
        cleanly.
        """
        n_workers = shard.n_workers
        if n_workers is None:
            import os
            n_workers = max(1, (os.cpu_count() or 1) - 2)
        label = shard.display_label
        local_progress_path = (
            progress_dir / f"{run_id_prefix}_shard_{label}.json"
        )
        try:
            paths = run_shard_pull(
                spec,
                queue_url=_loopback_queue_url(queue_url),
                run_id=run_id_prefix,
                n_workers=n_workers,
                output_dir=output_dir,
                label=label,
                stamp=stamp,
                iter_total_games=total_games,
                progress_dir=progress_dir,
            )
            return ShardSelfPlayReport(
                shard_label=label, success=True,
                npz_paths=list(paths),
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[shard {label}] FAILURE in local shard: "
                f"{type(e).__name__}: {e}. Continuing with sibling "
                f"shards' data.",
                file=sys.stderr,
            )
            _mark_shard_progress_failed(
                local_progress_path, shard_label=label,
                note=f"local_shard: {type(e).__name__}: {e}",
            )
            # Local shards write npzs to ``output_dir`` directly via
            # ``run_shard_pull``; recover whatever made it to disk.
            recovered: list[Path] = sorted(
                output_dir.glob(
                    f"{run_id_prefix}_shard_{label}_chunk_*.npz",
                ),
            )
            return ShardSelfPlayReport(
                shard_label=label, success=False,
                npz_paths=recovered,
                error=f"{type(e).__name__}: {e}",
                error_stage="local_shard",
            )

    def _remote_runner(shard: ShardConfig) -> ShardSelfPlayReport:
        return run_shard_remote(
            shard, spec,
            queue_url=queue_url,
            run_id=run_id_prefix,
            iter_total_games=total_games,
            progress_dir=progress_dir,
            output_dir=output_dir,
            run_id_prefix=run_id_prefix,
            stamp=stamp,
            bundle_url=bundle_url,
            shard_ssh_timeout_s=shard_ssh_timeout_s,
        )

    aggregator = _AggregateProgressPoller(
        progress_dir=progress_dir,
        run_id=run_id_prefix,
        n_games_total=total_games,
        kind="data_gen",
    )
    aggregator.start()

    # Early-stop: once `total_games` successful completions are in,
    # close the queue so shards drain. Only useful when `slack_extra
    # > 0`; with no slack the queue caps out at the target anyway.
    early_stop: Optional[_EarlyStopPoller] = None
    if slack_extra > 0:
        early_stop = _EarlyStopPoller(
            progress_dir=progress_dir,
            run_id=run_id_prefix,
            target_games=total_games,
            queue_url=queue_url,
        )
        early_stop.start()

    # Stop-check propagation: when the operator (or iterate's cycle
    # sentinel) requests a stop, close the queue server-side so
    # shards drain at their next take. Independent of early-stop;
    # both can run concurrently. POSTs the same /queue/close URL —
    # idempotent on the server side, so a race between the two
    # closures is harmless.
    stop_check_poller: Optional[_StopCheckPoller] = None
    if stop_check is not None:
        stop_check_poller = _StopCheckPoller(
            stop_check=stop_check,
            run_id=run_id_prefix,
            queue_url=queue_url,
        )
        stop_check_poller.start()

    # All shards run concurrently — they pull from the same queue,
    # so there's no static partitioning to worry about. Local
    # shards' mp.Pool runs in this process; remote shards block on
    # SSH (lots of idle wall time waiting on the network call to
    # finish before a thread can do useful work, but the master
    # CPU isn't held).
    # Skip dispatch entirely if stop was already requested before
    # we got here — the iter's stop sentinel may have landed during
    # queue init / aggregator startup. Without this guard we'd
    # spin up workers only to immediately drain them on first take,
    # which works but wastes ~1-2s of pool spawn cost per shard on
    # Windows.
    futures: dict[concurrent.futures.Future, ShardConfig] = {}
    n_threads = max(1, len(shards_config.shards))
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=n_threads, thread_name_prefix="dist-self-play",
    )
    pre_stop = stop_check is not None and stop_check()
    try:
        if not pre_stop:
            for shard in shards_config.shards:
                runner = (
                    _local_runner if shard.is_local else _remote_runner
                )
                futures[executor.submit(runner, shard)] = shard
    finally:
        executor.shutdown(wait=False)

    # Per-shard failures are converted to ShardSelfPlayReport inside
    # the runners; the orchestrator only sees raises here for things
    # the runners don't catch (KeyboardInterrupt, internal bugs).
    # Those still bubble up since they are not "one shard's SSH
    # glitch" — they're a genuine reason to abort the iter.
    interrupted = False
    try:
        for fut in concurrent.futures.as_completed(futures):
            try:
                report = fut.result()
            except BaseException:
                interrupted = True
                raise
            with reports_lock:
                reports.append(report)
            if report.success or report.npz_paths:
                with output_paths_lock:
                    output_paths.extend(report.npz_paths)
    finally:
        if early_stop is not None:
            early_stop.stop()
        if stop_check_poller is not None:
            stop_check_poller.stop()
        any_failed = any(not r.success for r in reports) or interrupted
        aggregator.stop("failed" if any_failed else "completed")

    # Loud post-iter summary so the user can see at a glance which
    # shards succeeded and which didn't.
    failed_reports = [r for r in reports if not r.success]
    if failed_reports:
        print(
            f"[orchestrator] {len(failed_reports)} of "
            f"{len(reports)} shards FAILED: "
            + ", ".join(
                f"{r.shard_label}({r.error_stage})"
                for r in failed_reports
            ),
            file=sys.stderr,
        )
        recovered_from_failed = sum(
            len(r.npz_paths) for r in failed_reports
        )
        if recovered_from_failed:
            print(
                f"[orchestrator] recovered "
                f"{recovered_from_failed} npz file(s) from failed "
                f"shards via post-failure rsync.",
                file=sys.stderr,
            )

    if reports and all(not r.success for r in reports) and not output_paths:
        # All shards failed AND we recovered no usable npzs — this
        # iter has nothing to train on. Raise a descriptive error so
        # the iterate driver fails the iteration cleanly rather than
        # silently producing a 0-game training step.
        raise RuntimeError(
            "all distributed self-play shards failed and no npz "
            "files were recovered; refusing to return an empty "
            f"training mix. Failures: "
            + "; ".join(
                f"{r.shard_label}({r.error_stage}): {r.error}"
                for r in failed_reports
            ),
        )
    return output_paths


def run_distributed_eval(
    shards_config: ShardsConfig,
    spec: EvalJobSpec,
    *,
    progress_dir: Path,
    run_id_prefix: str,
    total_games: int,
    queue_url: str,
    bundle_url: Optional[str] = None,
    slack_extra: int = 0,
    shard_ssh_timeout_s: float = _DEFAULT_SHARD_SSH_TIMEOUT_S,
    stop_check: Optional[Callable[[], bool]] = None,
) -> dict[str, int]:
    """Eval counterpart to ``run_distributed_self_play``. Initializes
    a queue of ``total_games + slack_extra``, dispatches every shard
    concurrently to play eval games (new vs old), and returns the
    cluster-aggregate counts as a dict::

        {"n_games": int, "wins_new": int, "wins_old": int,
         "draws": int, "crashed": int}

    The orchestrator's caller (typically iterate's eval-gate) maps
    this onto its existing ``MatchStats`` shape.

    ``slack_extra`` enables the same straggler-tolerance early-stop
    used by self-play: queue ``total + slack``, close once ``total``
    successful games are in.

    ``stop_check``: optional predicate polled by a background thread.
    On True, the orchestrator POSTs ``/queue/close/<run_id_prefix>``
    so the queue stops dispatching new eval games. Shards drain
    cleanly: their workers finish in-flight games, write the per-
    shard result JSON, and exit; the orchestrator returns the
    accumulated counts (which the eval-gate caller can then judge
    against its acceptance threshold on partial-but-conclusive
    data — same shape as :func:`run_match_parallel`'s ``stop_check``
    semantics). Defaults to ``None`` (no stop-check), preserving
    legacy behavior.

    The early-stop watcher (``scripts/eval_early_stop_watcher.py``)
    writes the same ``.STOP_<run_id>`` sentinel iterate's cycle-
    level stop uses, so the watcher's PASS / FAIL early-stop now
    propagates end-to-end through the distributed eval path.
    """
    import threading as _threading
    import urllib.request
    import json as _json

    stamp = _now_stamp()
    progress_dir.mkdir(parents=True, exist_ok=True)

    queued_games = int(total_games) + max(0, int(slack_extra))

    init_url = f"{queue_url.rstrip('/')}/queue/init"
    init_body = _json.dumps({
        "run_id": run_id_prefix,
        "total_games": queued_games,
    }).encode("utf-8")
    req = urllib.request.Request(
        init_url, data=init_body,
        headers={"Content-Type": "application/json"},
    )
    # Tight per-attempt timeout + bounded retries — see comment on
    # ``_QUEUE_HTTP_ATTEMPT_TIMEOUT_S`` for the rationale.
    init_resp = _urlopen_with_retry(req)
    if init_resp.status != 200:
        raise RuntimeError(
            f"queue init failed at {init_url}: HTTP {init_resp.status}"
        )

    totals = {
        "n_games": 0, "wins_new": 0, "wins_old": 0,
        "draws": 0, "crashed": 0,
    }
    totals_lock = _threading.Lock()
    reports: list[ShardEvalReport] = []
    reports_lock = _threading.Lock()
    zero_counts = {
        "n_games": 0, "wins_new": 0, "wins_old": 0,
        "draws": 0, "crashed": 0,
    }

    def _local_runner(shard: ShardConfig) -> ShardEvalReport:
        """Local-shard wrapper for eval. Same shape as the self-play
        side: convert per-shard exceptions into a failure
        :class:`ShardEvalReport` so siblings' counts survive."""
        n_workers = shard.n_workers
        if n_workers is None:
            import os
            n_workers = max(1, (os.cpu_count() or 1) - 2)
        label = shard.display_label
        local_progress_path = (
            progress_dir / f"{run_id_prefix}_shard_{label}.json"
        )
        try:
            counts = run_shard_eval_pull(
                spec,
                queue_url=_loopback_queue_url(queue_url),
                run_id=run_id_prefix,
                n_workers=n_workers,
                label=label,
                iter_total_games=total_games,
                progress_dir=progress_dir,
            )
            return ShardEvalReport(
                shard_label=label, success=True,
                counts=dict(counts),
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[shard {label}] FAILURE in local eval shard: "
                f"{type(e).__name__}: {e}. Continuing with sibling "
                f"shards' counts.",
                file=sys.stderr,
            )
            _mark_shard_progress_failed(
                local_progress_path, shard_label=label,
                note=f"local_shard: {type(e).__name__}: {e}",
            )
            return ShardEvalReport(
                shard_label=label, success=False,
                counts=dict(zero_counts),
                error=f"{type(e).__name__}: {e}",
                error_stage="local_shard",
            )

    def _remote_runner(shard: ShardConfig) -> ShardEvalReport:
        return run_shard_eval_remote(
            shard, spec,
            queue_url=queue_url,
            run_id=run_id_prefix,
            iter_total_games=total_games,
            progress_dir=progress_dir,
            stamp=stamp,
            bundle_url=bundle_url,
            shard_ssh_timeout_s=shard_ssh_timeout_s,
        )

    aggregator = _AggregateProgressPoller(
        progress_dir=progress_dir,
        run_id=run_id_prefix,
        n_games_total=total_games,
        kind="eval",
        opponent_label=spec.opponent_label,
    )
    aggregator.start()

    # Same straggler-tolerance scheme as self-play: close the queue
    # once `total_games` successful completions are in, abandoning
    # the slack indices.
    early_stop: Optional[_EarlyStopPoller] = None
    if slack_extra > 0:
        early_stop = _EarlyStopPoller(
            progress_dir=progress_dir,
            run_id=run_id_prefix,
            target_games=total_games,
            queue_url=queue_url,
        )
        early_stop.start()

    # Stop-check propagation: same as self-play, but for eval. The
    # early-stop watcher (eval_early_stop_watcher.py) calls
    # write_stop on PASS / FAIL, which fires this poller's stop_check
    # and closes the queue. Mirrors the rationale above; see the
    # self-play branch for the docstring.
    stop_check_poller: Optional[_StopCheckPoller] = None
    if stop_check is not None:
        stop_check_poller = _StopCheckPoller(
            stop_check=stop_check,
            run_id=run_id_prefix,
            queue_url=queue_url,
        )
        stop_check_poller.start()

    futures: dict[concurrent.futures.Future, ShardConfig] = {}
    n_threads = max(1, len(shards_config.shards))
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=n_threads, thread_name_prefix="dist-eval",
    )
    pre_stop = stop_check is not None and stop_check()
    try:
        if not pre_stop:
            for shard in shards_config.shards:
                runner = (
                    _local_runner if shard.is_local else _remote_runner
                )
                futures[executor.submit(runner, shard)] = shard
    finally:
        executor.shutdown(wait=False)

    interrupted = False
    try:
        for fut in concurrent.futures.as_completed(futures):
            try:
                report = fut.result()
            except BaseException:
                interrupted = True
                raise
            with reports_lock:
                reports.append(report)
            with totals_lock:
                for k in totals:
                    totals[k] += int(report.counts.get(k, 0))
    finally:
        if early_stop is not None:
            early_stop.stop()
        if stop_check_poller is not None:
            stop_check_poller.stop()
        any_failed = any(not r.success for r in reports) or interrupted
        aggregator.stop("failed" if any_failed else "completed")

    failed_reports = [r for r in reports if not r.success]
    if failed_reports:
        print(
            f"[orchestrator] {len(failed_reports)} of "
            f"{len(reports)} eval shards FAILED: "
            + ", ".join(
                f"{r.shard_label}({r.error_stage})"
                for r in failed_reports
            ),
            file=sys.stderr,
        )

    if reports and all(not r.success for r in reports):
        # No shard produced usable counts. The eval gate would
        # otherwise treat 0-of-0 as a draw and either gate or
        # un-gate based on tiebreaker rules — both are wrong when
        # the data is missing. Raise so the caller can decide.
        raise RuntimeError(
            "all distributed eval shards failed; no counts to "
            "aggregate. Failures: "
            + "; ".join(
                f"{r.shard_label}({r.error_stage}): {r.error}"
                for r in failed_reports
            ),
        )
    return totals


class EventLog:
    """JSONL event log per shard. One JSON object per line:
    ``{"ts": <iso8601>, "event": <type>, ...kwargs}``. Plain stdlib
    so it ships with the worker over SSH (no extra deps on remotes).
    Thread-safe for the in-process master shard's case where the
    orchestrator and the pool driver may both write.

    Sits next to the per-shard progress JSON in ``progress_dir`` so
    everything observability-related is in one place. ``run_shard_
    remote`` scps it back to the master at end-of-shard so analysis
    on the host has every machine's events in one file tree.

    Truncates the file on construction: re-running the same iter
    (retries after a crash, repeated test runs) replaces the prior
    events file rather than appending to it.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: str, **kwargs: Any) -> None:
        """Append one line. ``default=str`` swallows non-JSON-native
        values (Path, datetime, etc.) into their string repr — the
        log is for human + ad-hoc analysis, not strict round-trip."""
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **kwargs,
            },
            default=str,
        )
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _machine_metadata() -> dict[str, Any]:
    """Best-effort machine info captured once per shard at startup.
    Useful when one shard's results look weird and you want to know
    "did this run on the M2 Mac or the 1080Ti box; same torch
    version?". Cheap enough that we always emit it."""
    import os
    import platform
    info: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:  # noqa: SIM105
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    return info


def _weights_digest(weights_path: str) -> Optional[str]:
    """SHA-256 hex prefix of the weights file. Lets cross-machine
    analysis confirm "same weights ran on every shard" — a stale
    weights file on one remote would silently bias the iter's
    training mix toward that worker's policy."""
    try:
        import hashlib
        h = hashlib.sha256()
        with open(weights_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _play_one_parallel_game_timed(args: tuple) -> tuple:
    """Wrap ``_play_one_parallel_game`` with a ``perf_counter`` pair
    so the pull-mode worker gets per-game wall time without engine
    instrumentation. Top-level so multiprocessing can pickle it
    across the spawn boundary.

    Returns ``(game_idx, chunk, wall_seconds)``. ``chunk`` is the
    same 5-tuple the original wrapper returns.
    """
    import time as _time

    from mtg_bot.training.data_gen import _play_one_parallel_game
    t0 = _time.perf_counter()
    game_idx, chunk = _play_one_parallel_game(args)
    return game_idx, chunk, _time.perf_counter() - t0


def _play_one_match_game_timed(args: tuple) -> tuple:
    """Eval shard worker. Wraps the seat-alternating match player
    with ``perf_counter`` so we get per-game wall time. The
    alternating variant swaps factories on odd ``game_idx`` and
    remaps the returned winner — neutralizes the engine's seat-0
    advantage so wins[0] across a match means "the original
    ``p0_factory`` won" without a structural bias.

    Returns ``(game_idx, (winner, decisions), wall_seconds)``.
    """
    import time as _time

    from mtg_bot.training.self_play import (  # type: ignore
        _play_one_match_game_alternating_indexed,
    )
    t0 = _time.perf_counter()
    game_idx, result = _play_one_match_game_alternating_indexed(args)
    return game_idx, result, _time.perf_counter() - t0


def run_shard_pull(
    spec: SelfPlayJobSpec,
    *,
    queue_url: str,
    run_id: str,
    n_workers: int,
    output_dir: Path,
    label: str,
    stamp: str,
    iter_total_games: int,
    progress_dir: Optional[Path] = None,
) -> list[Path]:
    """Continuously feed ``n_workers`` worker processes from the
    master's pull-queue. Each worker plays one game, returns its
    chunk, then is given the next available game. Loop until the
    queue is drained AND all in-flight games have finished.

    Maximum concurrency = ``n_workers``; each ``/queue/take`` pulls
    one game (the worker that just freed up gets the next idx). Per-
    game completions are accumulated; an npz is flushed every
    ``n_workers`` games for crash resilience and reasonable file
    granularity, plus one final flush at shard end.

    Per-shard progress JSON gets ``n_games_total = iter_total_games``
    so the training UI shows "this shard did X of Y" honestly — pull
    mode doesn't know in advance how many games each shard will run.
    """
    import concurrent.futures as _cf
    import json as _json
    import time as _time
    import urllib.request

    from mtg_bot.decks import decks_from_json
    from mtg_bot.training import (
        NNPimcConfig, save_dataset,
    )
    from mtg_bot.training.data_gen import (
        ReplayCaptureConfig, _concat_chunks,
        _winner_to_p0_outcome,
    )
    from mtg_bot.training.progress import DataGenProgress
    from mtg_bot.training.trainer import auto_device

    cfg = NNPimcConfig(
        weights_path=spec.weights_path,
        samples=spec.samples,
        inner_nodes=spec.inner_nodes,
        device=auto_device(),
        batch_size=32,
        with_policy=spec.with_policy,
        puct=True,
        hidden_sizes=spec.hidden_sizes,
        architecture=spec.architecture,
        policy_hidden_sizes=spec.policy_hidden_sizes,
        action_encoding_version=spec.action_encoding_version,
        use_ismcts=spec.use_ismcts,
        use_belief_weighted=spec.use_belief_weighted,
    )
    factory = cfg.build

    deck_pool = None
    if spec.deck_pool_path is not None:
        deck_pool = decks_from_json(Path(spec.deck_pool_path))

    replay_config: Optional[ReplayCaptureConfig] = None
    replays_subdir: Optional[str] = None
    if spec.capture_every > 0 and spec.replays_root is not None:
        replays_subdir = str(
            Path(spec.replays_root) / f"{run_id}_shard_{label}"
        )
        Path(replays_subdir).mkdir(parents=True, exist_ok=True)
        replay_config = ReplayCaptureConfig(
            save_dir=replays_subdir,
            capture_every=spec.capture_every,
            nn_weights_path=spec.weights_path,
            nn_version_label=spec.nn_version_label,
            nn_hidden_sizes=spec.hidden_sizes,
            nn_with_policy=spec.with_policy,
        )

    hang_dump_dir: Optional[str] = None
    if spec.hang_dump_root is not None:
        hang_dump_dir = str(
            Path(spec.hang_dump_root) / f"{run_id}_shard_{label}"
        )

    progress: Optional[DataGenProgress] = None
    events: Optional[EventLog] = None
    if progress_dir is not None:
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress = DataGenProgress(
            progress_path=progress_dir / f"{run_id}_shard_{label}.json",
            run_id=f"{run_id}_shard_{label}",
            n_games_total=iter_total_games,
            n_workers=n_workers,
            command="distributed_self_play (pull-mode shard)",
            notes=(
                f"pull-mode shard {label} (one of N workers in iter "
                f"{run_id})"
            ),
            replays_dir=replays_subdir,
        )
        progress.begin()
        events = EventLog(
            progress_dir / f"{run_id}_shard_{label}_events.jsonl"
        )
        events.record(
            "shard_start",
            run_id=run_id,
            label=label,
            iter_total_games=iter_total_games,
            n_workers=n_workers,
            base_seed=spec.base_seed,
            samples=spec.samples,
            inner_nodes=spec.inner_nodes,
            weights_path=spec.weights_path,
            weights_digest=_weights_digest(spec.weights_path),
            machine=_machine_metadata(),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    take_url = f"{queue_url.rstrip('/')}/queue/take"
    npz_paths: list[Path] = []
    chunk_idx = 0
    n_games_completed = 0
    n_games_crashed = 0
    n_games_timed_out = 0
    pending_chunks: list = []
    flush_every = max(1, n_workers)

    def _take_one() -> tuple[Optional[int], int, float, bool]:
        req_body = _json.dumps({
            "run_id": run_id, "batch_size": 1,
        }).encode("utf-8")
        req = urllib.request.Request(
            take_url, data=req_body,
            headers={"Content-Type": "application/json"},
        )
        take_t0 = _time.perf_counter()
        # Tight per-attempt timeout + bounded retries — see comment
        # on ``_QUEUE_HTTP_ATTEMPT_TIMEOUT_S`` for the rationale.
        resp_body = _urlopen_with_retry(req).body
        response = _json.loads(resp_body.decode("utf-8"))
        take_wall = _time.perf_counter() - take_t0
        games = response.get("games", [])
        # ``closed`` is the orchestrator's early-stop flag: queue
        # closed by /queue/close before its indices ran out. Distinct
        # from a natural drain (next == total, no ``closed`` key set).
        closed = bool(response.get("closed", False))
        return (
            games[0] if games else None,
            response.get("remaining", 0),
            take_wall,
            closed,
        )

    def _build_args(game_idx: int) -> tuple:
        # 13-element form so the worker's ``_play_one_parallel_game``
        # picks up the spec's slot-mapping convention rather than
        # silently defaulting to v1. ``None`` flows through as v1
        # there too — keeps the existing v13/v15/v16 behavior intact.
        return (
            game_idx, factory, factory, spec.base_seed,
            spec.decision_cap, spec.sample_every,
            replay_config, deck_pool,
            spec.enable_mulligans, spec.per_game_timeout_sec,
            hang_dump_dir, spec.heuristics,
            spec.action_encoding_version or "positional_v1",
        )

    def _flush() -> None:
        nonlocal chunk_idx
        if not pending_chunks:
            return
        f, o, p, m = _concat_chunks(
            [c[:4] for _, c, _ in pending_chunks]
        )
        output_path = output_dir / (
            f"{run_id}_shard_{label}_chunk_"
            f"{chunk_idx:04d}_{stamp}.npz"
        )
        save_dataset(
            f, o, output_path,
            policy_targets=p, action_masks=m,
        )
        npz_paths.append(output_path)
        if events is not None:
            events.record(
                "batch_saved",
                chunk_idx=chunk_idx,
                npz_path=str(output_path),
                n_samples=int(f.shape[0]) if f.size else 0,
                n_games=len(pending_chunks),
            )
        chunk_idx += 1
        pending_chunks.clear()

    shard_t0 = _time.perf_counter()
    # Don't use ``with ProcessPoolExecutor(...) as exe:`` here. The
    # context-manager exit calls ``shutdown(wait=True)``, which blocks
    # until every still-running future completes. Per-game timeouts
    # (``_timeout.run_with_timeout``) are thread-based — they raise
    # ``TimeoutError`` in the parent thread but the daemon work thread
    # in the worker process keeps running (Windows/CPython can't kill a
    # thread mid-C-extension call). Result: a Mac shard with 3 games
    # past the 300s budget will hang forever in ``__exit__`` even
    # though all completed games are already flushed to disk. Manual
    # ``shutdown(wait=False)`` returns as soon as no new tasks can be
    # submitted; the OS reaps the orphan workers when the parent exits.
    exe = _cf.ProcessPoolExecutor(max_workers=n_workers)
    try:
        in_flight: dict[_cf.Future, int] = {}
        queue_drained = False
        # ``queue_closed_early`` records whether the drain came from
        # the orchestrator's /queue/close (target reached early) or
        # the queue running out of indices on its own. We previously
        # distinguished the two — early-stop close abandoned in-flight
        # games, natural drain waited for them. In practice that
        # distinction made the long-tail wait painful: when the queue
        # is sized at target+slack but the timeout rate exceeds slack
        # (observed 2026-05-05: 13.5% timeouts, 10% slack ⇒ early-stop
        # never fires), shards drain naturally and then sat for up to
        # ``per_game_timeout_sec`` (5 min) waiting for stragglers.
        # Now both drains abandon in-flight after the queue empties:
        # any games already on disk are flushed below; the pending
        # ones go nowhere useful by waiting longer.
        queue_closed_early = False
        while True:
            while len(in_flight) < n_workers and not queue_drained:
                game_idx, remaining, take_wall, closed = _take_one()
                if events is not None:
                    events.record(
                        "game_take",
                        game_idx=game_idx,
                        remaining_after=remaining,
                        take_wall_s=take_wall,
                        in_flight=len(in_flight),
                    )
                if game_idx is None:
                    queue_drained = True
                    queue_closed_early = closed
                    if events is not None:
                        events.record(
                            "queue_closed" if closed
                            else "queue_drained",
                            in_flight=len(in_flight),
                            remaining=remaining,
                        )
                    break
                fut = exe.submit(
                    _play_one_parallel_game_timed,
                    _build_args(game_idx),
                )
                in_flight[fut] = game_idx
            if not in_flight:
                break
            if queue_drained:
                # Queue is empty (early-stop close OR natural drain):
                # abandon any in-flight games so the orchestrator can
                # move on. ``finally: exe.shutdown(wait=False)``
                # releases them. Already-completed pending_chunks get
                # flushed below. ``reason`` lets log readers tell the
                # two paths apart for diagnostics.
                if events is not None:
                    events.record(
                        "shard_abandoned_inflight",
                        n_abandoned=len(in_flight),
                        reason=(
                            "queue_closed_early"
                            if queue_closed_early
                            else "queue_drained"
                        ),
                    )
                break
            done, _pending = _cf.wait(
                list(in_flight.keys()),
                return_when=_cf.FIRST_COMPLETED,
            )
            for fut in done:
                in_flight.pop(fut)
                game_idx, chunk, wall_s = fut.result()
                n_samples = int(chunk[0].shape[0])
                # Worker tags each empty chunk with an outcome kind
                # ("crashed" / "timed_out" / "completed"). Pre-feature
                # shards may still produce 5-tuples (no kind element);
                # fall back to detecting the generic empty-chunk case
                # as "crashed" for back-compat with on-disk replays.
                outcome_kind = (
                    chunk[5] if len(chunk) > 5
                    else (
                        "crashed"
                        if n_samples == 0 and chunk[4] is None
                        else "completed"
                    )
                )
                crashed = outcome_kind == "crashed"
                timed_out = outcome_kind == "timed_out"
                if crashed:
                    n_games_crashed += 1
                elif timed_out:
                    n_games_timed_out += 1
                n_games_completed += 1
                if progress is not None:
                    if crashed:
                        progress.record_game(crashed=True)
                    elif timed_out:
                        progress.record_game(timed_out=True)
                    else:
                        progress.record_game(
                            n_samples=n_samples,
                            outcome_p0=_winner_to_p0_outcome(
                                chunk[4],
                            ),
                        )
                if events is not None:
                    events.record(
                        "game_end",
                        game_idx=int(game_idx),
                        wall_s=wall_s,
                        n_samples=n_samples,
                        winner=chunk[4],
                        outcome=outcome_kind,
                        crashed=crashed,
                        timed_out=timed_out,
                    )
                pending_chunks.append((game_idx, chunk, wall_s))
                if len(pending_chunks) >= flush_every:
                    _flush()
        _flush()
    except BaseException as exc:
        try:
            _flush()
        except Exception:  # noqa: BLE001
            pass
        if progress is not None:
            progress.finalize(status="failed")
        if events is not None:
            events.record(
                "shard_end",
                status="failed",
                error=repr(exc),
                n_games_completed=n_games_completed,
                n_games_crashed=n_games_crashed,
                n_games_timed_out=n_games_timed_out,
                n_batches=chunk_idx,
                wall_s=_time.perf_counter() - shard_t0,
            )
        raise
    finally:
        exe.shutdown(wait=False)
    if progress is not None:
        progress.finalize(status="completed")
    if events is not None:
        events.record(
            "shard_end",
            status="completed",
            n_games_completed=n_games_completed,
            n_games_crashed=n_games_crashed,
            n_games_timed_out=n_games_timed_out,
            n_batches=chunk_idx,
            wall_s=_time.perf_counter() - shard_t0,
        )
    return npz_paths


def run_shard_eval_pull(
    spec: EvalJobSpec,
    *,
    queue_url: str,
    run_id: str,
    n_workers: int,
    label: str,
    iter_total_games: int,
    progress_dir: Optional[Path] = None,
) -> dict[str, int]:
    """Eval counterpart to ``run_shard_pull``: continuously feeds
    n_workers worker processes from the master's pull-queue, where
    each game is one match (new-NN seat 0 vs old-NN-or-heuristic
    seat 1). Returns the shard's aggregated counts as a dict so the
    orchestrator can sum across shards into a single ``MatchStats``.
    """
    import concurrent.futures as _cf
    import json as _json
    import time as _time
    import urllib.request

    from mtg_bot.decks import decks_from_json
    from mtg_bot.training import NNPimcConfig
    from mtg_bot.training.config import HeuristicConfig, PimcConfig
    from mtg_bot.training.data_gen import ReplayCaptureConfig
    from mtg_bot.training.progress import EvalProgress
    from mtg_bot.training.trainer import auto_device

    _device = auto_device()
    new_cfg = NNPimcConfig(
        weights_path=spec.new_weights_path,
        samples=spec.samples, inner_nodes=spec.inner_nodes,
        batch_size=32, with_policy=True, puct=True,
        hidden_sizes=spec.hidden_sizes,
        architecture=spec.new_architecture,
        policy_hidden_sizes=spec.policy_hidden_sizes,
        action_encoding_version=spec.new_action_encoding_version,
        use_ismcts=spec.use_ismcts,
        use_belief_weighted=spec.use_belief_weighted,
        device=_device,
    )
    if spec.old_weights_path is not None:
        old_cfg = NNPimcConfig(
            weights_path=spec.old_weights_path,
            samples=spec.samples, inner_nodes=spec.inner_nodes,
            batch_size=32, with_policy=True, puct=True,
            hidden_sizes=spec.hidden_sizes,
            architecture=spec.old_architecture,
            policy_hidden_sizes=spec.policy_hidden_sizes,
            action_encoding_version=spec.old_action_encoding_version,
            use_ismcts=spec.use_ismcts,
            use_belief_weighted=spec.use_belief_weighted,
            device=_device,
        )
        p1_factory = old_cfg.build
    else:
        p1_factory = PimcConfig(
            samples=spec.samples, inner_nodes=spec.inner_nodes,
            heuristic=HeuristicConfig(), batch_size=1,
        ).build
    p0_factory = new_cfg.build

    deck_pool = None
    if spec.deck_pool_path is not None:
        deck_pool = decks_from_json(Path(spec.deck_pool_path))

    replay_config: Optional[ReplayCaptureConfig] = None
    replays_subdir: Optional[str] = None
    if spec.capture_every > 0 and spec.replays_root is not None:
        replays_subdir = str(
            Path(spec.replays_root) / f"{run_id}_shard_{label}"
        )
        Path(replays_subdir).mkdir(parents=True, exist_ok=True)
        replay_config = ReplayCaptureConfig(
            save_dir=replays_subdir,
            capture_every=spec.capture_every,
            nn_weights_path=spec.new_weights_path,
            nn_version_label=spec.nn_new_version_label,
            nn_hidden_sizes=spec.hidden_sizes,
            nn_with_policy=True,
        )

    hang_dump_dir: Optional[str] = None
    if spec.hang_dump_root is not None:
        hang_dump_dir = str(
            Path(spec.hang_dump_root) / f"{run_id}_shard_{label}"
        )

    progress: Optional[EvalProgress] = None
    events: Optional[EventLog] = None
    if progress_dir is not None:
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress = EvalProgress(
            progress_path=progress_dir / f"{run_id}_shard_{label}.json",
            run_id=f"{run_id}_shard_{label}",
            n_games_total=iter_total_games,
            n_workers=n_workers,
            opponent_label=spec.opponent_label,
            command="distributed_eval (pull-mode shard)",
            notes=(
                f"pull-mode eval shard {label} (one of N workers in "
                f"iter {run_id})"
            ),
            replays_dir=replays_subdir,
        )
        progress.begin()
        events = EventLog(
            progress_dir / f"{run_id}_shard_{label}_events.jsonl"
        )
        events.record(
            "shard_start",
            run_id=run_id, label=label, mode="eval",
            iter_total_games=iter_total_games,
            n_workers=n_workers,
            base_seed=spec.base_seed,
            samples=spec.samples, inner_nodes=spec.inner_nodes,
            new_weights_path=spec.new_weights_path,
            old_weights_path=spec.old_weights_path,
            new_weights_digest=_weights_digest(spec.new_weights_path),
            old_weights_digest=(
                _weights_digest(spec.old_weights_path)
                if spec.old_weights_path else None
            ),
            machine=_machine_metadata(),
        )

    take_url = f"{queue_url.rstrip('/')}/queue/take"
    wins_new = 0
    wins_old = 0
    draws = 0
    crashed = 0
    timed_out = 0
    n_games = 0

    def _take_one() -> tuple[Optional[int], int, float, bool]:
        req_body = _json.dumps({
            "run_id": run_id, "batch_size": 1,
        }).encode("utf-8")
        req = urllib.request.Request(
            take_url, data=req_body,
            headers={"Content-Type": "application/json"},
        )
        take_t0 = _time.perf_counter()
        # Tight per-attempt timeout + bounded retries — see comment
        # on ``_QUEUE_HTTP_ATTEMPT_TIMEOUT_S`` for the rationale.
        resp_body = _urlopen_with_retry(req).body
        response = _json.loads(resp_body.decode("utf-8"))
        take_wall = _time.perf_counter() - take_t0
        games = response.get("games", [])
        # See run_shard_pull._take_one for the closed/drained split.
        closed = bool(response.get("closed", False))
        return (
            games[0] if games else None,
            response.get("remaining", 0),
            take_wall,
            closed,
        )

    def _build_args(game_idx: int) -> tuple:
        return (
            game_idx, p0_factory, p1_factory, spec.base_seed,
            4000,  # decision_cap, mirrors run_match's default
            spec.enable_mulligans, spec.per_game_timeout_sec,
            deck_pool, replay_config, hang_dump_dir,
            spec.heuristics,
        )

    shard_t0 = _time.perf_counter()
    # See run_shard_pull above for why this isn't ``with ... as exe``:
    # ProcessPoolExecutor.__exit__ calls shutdown(wait=True), which
    # blocks forever on a worker stuck in C-extension code past the
    # per-game timeout. Manual shutdown(wait=False) returns promptly;
    # OS reaps orphan workers when the parent exits.
    exe = _cf.ProcessPoolExecutor(max_workers=n_workers)
    try:
        in_flight: dict[_cf.Future, int] = {}
        queue_drained = False
        # See run_shard_pull for the unified-abandon rationale; eval
        # mirrors self-play here. ``queue_closed_early`` records the
        # drain reason for the abandon-event log only.
        queue_closed_early = False
        while True:
            while len(in_flight) < n_workers and not queue_drained:
                game_idx, remaining, take_wall, closed = _take_one()
                if events is not None:
                    events.record(
                        "game_take",
                        game_idx=game_idx,
                        remaining_after=remaining,
                        take_wall_s=take_wall,
                        in_flight=len(in_flight),
                    )
                if game_idx is None:
                    queue_drained = True
                    queue_closed_early = closed
                    if events is not None:
                        events.record(
                            "queue_closed" if closed
                            else "queue_drained",
                            in_flight=len(in_flight),
                            remaining=remaining,
                        )
                    break
                fut = exe.submit(
                    _play_one_match_game_timed,
                    _build_args(game_idx),
                )
                in_flight[fut] = game_idx
            if not in_flight:
                break
            if queue_drained:
                # Queue is empty (early-stop close OR natural drain):
                # abandon any in-flight eval games. Counts already
                # accumulated for finished games are written to the
                # result JSON in the ``finally`` below, so we don't
                # lose data — just the abandoned games' would-be
                # results.
                if events is not None:
                    events.record(
                        "shard_abandoned_inflight",
                        n_abandoned=len(in_flight),
                        reason=(
                            "queue_closed_early"
                            if queue_closed_early
                            else "queue_drained"
                        ),
                    )
                break
            done, _pending = _cf.wait(
                list(in_flight.keys()),
                return_when=_cf.FIRST_COMPLETED,
            )
            for fut in done:
                in_flight.pop(fut)
                game_idx, result, wall_s = fut.result()
                if len(result) >= 3:
                    winner, decisions, outcome_kind = (
                        result[0], result[1], result[2],
                    )
                else:
                    winner, decisions = result[0], result[1]
                    outcome_kind = (
                        "crashed"
                        if winner is None and decisions == 0
                        else "completed"
                    )
                is_crash = outcome_kind == "crashed"
                is_timeout = outcome_kind == "timed_out"
                n_games += 1
                if is_crash:
                    crashed += 1
                elif is_timeout:
                    timed_out += 1
                elif winner == 0:
                    wins_new += 1
                elif winner == 1:
                    wins_old += 1
                else:
                    draws += 1
                if progress is not None:
                    progress.record_game(
                        winner=winner, decisions=decisions,
                        crashed=is_crash,
                        timed_out=is_timeout,
                    )
                if events is not None:
                    events.record(
                        "game_end",
                        game_idx=int(game_idx),
                        wall_s=wall_s,
                        winner=winner,
                        decisions=decisions,
                        outcome=outcome_kind,
                        crashed=is_crash,
                        timed_out=is_timeout,
                    )
    except BaseException as exc:
        if progress is not None:
            progress.finalize(status="failed")
        if events is not None:
            events.record(
                "shard_end",
                status="failed",
                error=repr(exc),
                n_games=n_games, wins_new=wins_new,
                wins_old=wins_old, draws=draws, crashed=crashed,
                timed_out=timed_out,
                wall_s=_time.perf_counter() - shard_t0,
            )
        raise
    finally:
        exe.shutdown(wait=False)
    if progress is not None:
        progress.finalize(status="completed")
    if events is not None:
        events.record(
            "shard_end",
            status="completed",
            n_games=n_games, wins_new=wins_new,
            wins_old=wins_old, draws=draws, crashed=crashed,
            timed_out=timed_out,
            wall_s=_time.perf_counter() - shard_t0,
        )
    return {
        "n_games": n_games,
        "wins_new": wins_new,
        "wins_old": wins_old,
        "draws": draws,
        "crashed": crashed,
        "timed_out": timed_out,
    }


def _is_windows_remote(shard: ShardConfig) -> bool:
    """Cheap heuristic to pick remote path semantics: ``.exe`` ending
    on the venv python path is unambiguously Windows. POSIX shells
    don't use that suffix."""
    return (shard.venv_python or "").endswith(".exe")


def _remote_join(shard: ShardConfig, *parts: str) -> str:
    """Join path parts using the remote's OS conventions. We don't
    actually run on the remote — this just produces the string the
    remote shell will interpret.

    POSIX: ``/`` separator via ``PurePosixPath``.

    Windows: all forward slashes, even inside ``C:\\Users\\...`` —
    cmd.exe handles ``C:/Users/...`` fine, AND scp's quoting layer
    rejects mixed-separator paths like ``C:\\Users\\.../sub/file``
    (it double-escapes the backslashes and the SSH server can't
    find the path). So we rewrite every backslash in each input to
    ``/`` and join with ``/``.
    """
    if _is_windows_remote(shard):
        return "/".join(p.replace("\\", "/") for p in parts)
    return str(PurePosixPath(*parts))


def _ssh_target(shard: ShardConfig) -> str:
    return f"{shard.user}@{shard.host}"


# TCP-keepalive options applied to every ssh / scp invocation. Without
# them, a Tailscale glitch that drops the network briefly leaves master
# holding a half-dead socket; the OS-level keepalive timeout is hours.
# With ServerAliveInterval=30 + ServerAliveCountMax=5, ssh itself sends
# a no-op every 30s and considers the connection dead after 5 misses
# (~150s). The shard's actual python work can still take 30+ minutes —
# these keepalives don't bound the legitimate work, only the time it
# takes to NOTICE a wedged socket.
#
# Observed 2026-05-05: master's SSH to Mac wedged for 90+ min after
# Mac dropped off Tailscale briefly mid-iteration; iter_002 hung in
# the rsync-back stage even though Mac's python process had finished.
_SSH_KEEPALIVE_OPTS = (
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=5",
)

def _ssh_cmd(shard: ShardConfig, remote_command: str) -> list[str]:
    """SSH invocation for a shell command on the remote. Sends one
    argument string the remote shell will interpret. The remote
    shell is bash on POSIX and cmd.exe on Windows OpenSSH-Server
    default — both handle ``&&`` and simple variable assignment, so
    a single command string portable to both is achievable.
    """
    return [
        "ssh",
        "-o", "BatchMode=yes",      # never prompt for password
        "-o", "StrictHostKeyChecking=accept-new",
        *_SSH_KEEPALIVE_OPTS,
        "-p", str(shard.ssh_port),
        _ssh_target(shard),
        remote_command,
    ]


def _scp_cmd(
    shard: ShardConfig, src: str, dst: str, *, from_remote: bool,
) -> list[str]:
    """``scp`` invocation. ``from_remote=True`` pulls from remote to
    local; ``False`` pushes."""
    target = _ssh_target(shard)
    if from_remote:
        return [
            "scp", "-P", str(shard.ssh_port),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            *_SSH_KEEPALIVE_OPTS,
            f"{target}:{src}", str(dst),
        ]
    return [
        "scp", "-P", str(shard.ssh_port),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        *_SSH_KEEPALIVE_OPTS,
        str(src), f"{target}:{dst}",
    ]


def _run_subprocess(
    cmd: list[str], *, timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Wrapper that captures both streams + raises on non-zero exit.
    Also gives us ONE place to instrument later (logging, retry)."""
    return subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, check=True,
    )


# ---- SSH-with-terminal-status watchdog ---------------------------
#
# The recurring "shard SSH hangs at 100%" symptom: every shard's
# per-shard progress JSON shows status="completed" with all data on
# disk, but ssh.exe keeps the orchestrator stuck for tens of minutes
# because Tailscale's WireGuard tunnel keeps the underlying TCP
# connection healthy at the network layer; SSH client never sees a
# clean close, and ServerAliveInterval/CountMax don't fire either.
#
# Root-cause fix: as soon as the LOCAL per-shard progress JSON (kept
# fresh by ``_ProgressPoller`` running concurrently) transitions to
# a terminal status, give a brief grace period for any final flushes
# (events.jsonl, npz saves) and forcibly terminate the SSH subprocess.
# The existing master-shard-resilience handler then catches the
# resulting CalledProcessError, and the post-failure rsync recovers
# any straggling data. Net effect: the hang collapses from 10-90 min
# to ~10s.
_WATCHDOG_POLL_INTERVAL_S = 5.0
_WATCHDOG_TERMINAL_GRACE_S = 10.0
# After terminate(), how long to wait before escalating to kill().
# terminate() is gentle (SIGTERM on POSIX, TerminateProcess on
# Windows); kill() is the hammer (SIGKILL on POSIX).
_WATCHDOG_TERMINATE_GRACE_S = 5.0


def _read_terminal_status(
    progress_path: Path,
    *,
    min_started_at: Optional[str] = None,
) -> Optional[str]:
    """Best-effort read of the local per-shard progress JSON's
    ``status`` field. Returns the string only if it is one of the
    terminal sentinels (``completed`` / ``failed``); returns None
    otherwise (file missing, partial write mid-flight, non-terminal
    status, malformed JSON). Swallows OSError + JSONDecodeError —
    a transient read failure on one tick is fine; the next tick
    retries.

    ``min_started_at`` (optional, ISO 8601 string): if provided,
    treat the file as not-yet-terminal whenever its ``started_at``
    is missing or lexically less than the floor. Defends against
    stale prior-run shard JSONs that say ``status="completed"``
    sitting on disk before the current run's ``_ProgressPoller`` has
    written a fresh copy. Same filter shape as ``_EarlyStopPoller``.
    """
    if not progress_path.is_file():
        return None
    try:
        d = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    status = d.get("status")
    if status not in ("completed", "failed"):
        return None
    if min_started_at is not None:
        started_at = d.get("started_at")
        if not isinstance(started_at, str) or started_at < min_started_at:
            return None
    return status


def _run_ssh_with_terminal_status_watchdog(
    cmd: list[str],
    *,
    progress_path: Path,
    hard_timeout_s: float,
    shard_label: str,
    poll_interval_s: float = _WATCHDOG_POLL_INTERVAL_S,
    terminal_grace_s: float = _WATCHDOG_TERMINAL_GRACE_S,
    terminate_grace_s: float = _WATCHDOG_TERMINATE_GRACE_S,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> subprocess.CompletedProcess:
    """Run an SSH subprocess with two complementary safety nets:

    1. **Terminal-status watchdog (proactive).** A daemon thread polls
       ``progress_path`` (the local per-shard progress JSON, kept
       fresh by :class:`_ProgressPoller`). As soon as the file's
       ``status`` field reaches a terminal value
       (``completed``/``failed``), the watchdog waits a small grace
       period (~10s) for any final flushes — events.jsonl writes,
       npz saves — then forcibly ``terminate()``s the SSH
       subprocess. If the process doesn't exit within
       ``terminate_grace_s``, escalate to ``kill()``.

    2. **Hard timeout (floor).** ``proc.communicate(timeout=...)``
       caps total wall time at ``hard_timeout_s``. Bounds the case
       where the watchdog never sees a terminal status (e.g. shard
       died before writing one). Raises
       :class:`subprocess.TimeoutExpired` exactly like
       :func:`_run_subprocess` would.

    On a clean exit (returncode 0): returns a
    :class:`subprocess.CompletedProcess`.
    On a non-zero exit caused by the watchdog terminating the SSH
    after the shard reported ``status='completed'``: also returns a
    :class:`subprocess.CompletedProcess` — the shard finished its
    work, the non-zero exit is just a side-effect of yanking the SSH
    out from under it.
    On any other non-zero exit (including watchdog-triggered
    termination after ``status='failed'``):
    raises :class:`subprocess.CalledProcessError`.
    On hard timeout: raises :class:`subprocess.TimeoutExpired`.

    These are the same exceptions ``_run_subprocess(check=True)``
    raises, so call sites' existing try/except blocks handle both.

    The ``popen_factory`` parameter is a seam for tests — production
    code passes ``subprocess.Popen``; tests pass a fake that returns
    a controllable Popen-like object.
    """
    # Stamp our start. Stale shard JSONs from prior runs of the
    # same run_id may still be on disk before _ProgressPoller writes
    # a fresh copy; treat any JSON whose started_at predates this
    # stamp as not-yet-terminal so we don't terminate the SSH
    # before the shard has even fired up. Same shape as the
    # _EarlyStopPoller stale-JSON filter.
    watchdog_min_started_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds",
    )
    proc = popen_factory(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ssh_done_event = threading.Event()
    # Records the terminal status the watchdog observed before
    # terminating SSH. Stays ``None`` if the watchdog never fired
    # (proc exited on its own). One-element list for closure mutation.
    watchdog_terminated_with: list[str | None] = [None]

    def _watchdog() -> None:
        """Polls ``progress_path``. Returns once SSH is done OR after
        terminating the process when terminal status is observed."""
        while not ssh_done_event.wait(timeout=poll_interval_s):
            status = _read_terminal_status(
                progress_path,
                min_started_at=watchdog_min_started_at,
            )
            if status is None:
                continue
            # Give the shard a brief moment to finish any
            # in-progress writes before we yank the SSH session
            # out from under it.
            if ssh_done_event.wait(timeout=terminal_grace_s):
                # SSH exited on its own during the grace period;
                # nothing to terminate.
                return
            print(
                f"[shard {shard_label}] shard reports "
                f"status={status!r}; terminating SSH session "
                f"after {terminal_grace_s:.0f}s grace.",
                file=sys.stderr,
            )
            watchdog_terminated_with[0] = status
            try:
                proc.terminate()
            except OSError:
                # Process already gone; nothing to do.
                return
            # Escalate to kill() if terminate() didn't take.
            # ssh.exe in particular has been observed to ignore
            # the gentle signal in the Tailscale-keeps-TCP-alive
            # scenario.
            if not ssh_done_event.wait(timeout=terminate_grace_s):
                print(
                    f"[shard {shard_label}] terminate() did not "
                    f"exit within {terminate_grace_s:.0f}s; "
                    f"escalating to kill().",
                    file=sys.stderr,
                )
                try:
                    proc.kill()
                except OSError:
                    pass
            return

    watchdog = threading.Thread(
        target=_watchdog,
        name=f"ssh-watchdog-{shard_label}",
        daemon=True,
    )
    watchdog.start()
    try:
        stdout, stderr = proc.communicate(timeout=hard_timeout_s)
    except subprocess.TimeoutExpired:
        # Hard floor fired. Reap the child so it doesn't linger.
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.communicate(timeout=5.0)
        except (subprocess.TimeoutExpired, OSError):
            pass
        raise
    finally:
        ssh_done_event.set()
        watchdog.join(timeout=2.0)

    if proc.returncode != 0:
        # When the watchdog killed SSH because the shard self-
        # reported ``status='completed'``, the shard already finished
        # its work cleanly — the non-zero exit is purely an artifact
        # of the forced terminate(). Treat as success so the
        # orchestrator doesn't mistakenly mark a successful shard as
        # FAILED (observed live: iter 3 of 2026-05-06 had Mac do
        # 314/500 games and report completed, watchdog terminated,
        # the parent self_play summary then showed Mac status=failed
        # despite all data being collected).
        if watchdog_terminated_with[0] == "completed":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        raise subprocess.CalledProcessError(
            proc.returncode, cmd,
            output=stdout, stderr=stderr,
        )
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _build_remote_pull_command(
    shard: ShardConfig,
    spec: SelfPlayJobSpec,
    *,
    queue_url: str,
    run_id: str,
    iter_total_games: int,
    stamp: str,
) -> str:
    """Produce the one-line shell command the remote runs to start
    its pull-mode worker. Same wrapper script as the master's local
    shard uses, just invoked over SSH.

    Spec + shard configs ride as base64-encoded JSON to avoid
    quoting hazards across cmd.exe vs bash. ``queue_url`` is plain
    ASCII (no quoting needed); the rest of the args are integers
    or short strings.
    """
    import base64
    spec_b64 = base64.b64encode(
        json.dumps(spec.to_dict()).encode("utf-8"),
    ).decode("ascii")
    shard_b64 = base64.b64encode(
        json.dumps(shard.to_dict()).encode("utf-8"),
    ).decode("ascii")

    args = (
        f"scripts/run_self_play_shard.py "
        f"--mode=pull "
        f"--spec-b64={spec_b64} "
        f"--shard-b64={shard_b64} "
        f"--queue-url={queue_url} "
        f"--run-id={run_id} "
        f"--iter-total-games={iter_total_games} "
        f"--stamp={stamp}"
    )
    repo_path = shard.repo_path or ""
    venv_python = shard.venv_python or ""
    if _is_windows_remote(shard):
        return (
            f'cd /d "{repo_path}" && '
            f'set PYTHONPATH=src && '
            f'"{venv_python}" {args}'
        )
    return (
        f'cd {shlex.quote(repo_path)} && '
        f'PYTHONPATH=src {shlex.quote(venv_python)} {args}'
    )


def _build_remote_eval_command(
    shard: ShardConfig,
    spec: EvalJobSpec,
    *,
    queue_url: str,
    run_id: str,
    iter_total_games: int,
    stamp: str,
) -> str:
    """Eval counterpart to ``_build_remote_pull_command``. Same shape;
    ``--mode=eval`` and an ``EvalJobSpec`` instead of ``SelfPlayJobSpec``.
    """
    import base64
    spec_b64 = base64.b64encode(
        json.dumps(spec.to_dict()).encode("utf-8"),
    ).decode("ascii")
    shard_b64 = base64.b64encode(
        json.dumps(shard.to_dict()).encode("utf-8"),
    ).decode("ascii")

    args = (
        f"scripts/run_self_play_shard.py "
        f"--mode=eval "
        f"--spec-b64={spec_b64} "
        f"--shard-b64={shard_b64} "
        f"--queue-url={queue_url} "
        f"--run-id={run_id} "
        f"--iter-total-games={iter_total_games} "
        f"--stamp={stamp}"
    )
    repo_path = shard.repo_path or ""
    venv_python = shard.venv_python or ""
    if _is_windows_remote(shard):
        return (
            f'cd /d "{repo_path}" && '
            f'set PYTHONPATH=src && '
            f'"{venv_python}" {args}'
        )
    return (
        f'cd {shlex.quote(repo_path)} && '
        f'PYTHONPATH=src {shlex.quote(venv_python)} {args}'
    )


class _ProgressPoller:
    """Periodically scp the remote's per-shard progress JSON back to
    the master so the training UI shows live updates. Stops when
    ``stop()`` is called. Errors are swallowed — a failed scp on
    one tick must not break the run; the next tick retries.
    """

    def __init__(
        self,
        shard: ShardConfig,
        remote_progress_path: str,
        local_progress_path: Path,
        interval_s: float = 5.0,
    ) -> None:
        self._shard = shard
        self._remote_path = remote_progress_path
        self._local_path = local_progress_path
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name=f"progress-poll-{self._shard.display_label}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                cmd = _scp_cmd(
                    self._shard,
                    self._remote_path, str(self._local_path),
                    from_remote=True,
                )
                # Don't raise on non-zero: the file may not yet
                # exist on the remote (worker startup).
                subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self._interval_s + 5,
                )
            except Exception:  # noqa: BLE001
                pass
            self._stop_event.wait(self._interval_s)


def _build_remote_mkdir_command(
    shard: ShardConfig, remote_dir: str,
) -> str:
    """Idempotent ``mkdir -p`` equivalent for the remote shell. cmd.exe
    ``mkdir`` is finicky over OpenSSH-Server (silently fails on
    existing dirs), so on Windows we route through PowerShell's
    ``New-Item -Force`` which is genuinely idempotent."""
    if _is_windows_remote(shard):
        return (
            f"powershell -Command \"New-Item -ItemType Directory "
            f"-Force -Path '{remote_dir}'\""
        )
    return f"mkdir -p {shlex.quote(remote_dir)}"


def _build_remote_bundle_sync_command(
    shard: ShardConfig, bundle_url: str,
) -> str:
    """Sync the remote's working tree to the latest master commits
    via a Tailscale-served git bundle. Avoids needing GitHub auth on
    the remote (curl gets the bundle over Tailscale; git fetches
    from the local bundle file; git reset --hard advances HEAD).

    Refspec ``main`` is hard-coded — that's what iterate runs from.
    Other branches stay on whatever they were.
    """
    repo_path = shard.repo_path or ""
    if _is_windows_remote(shard):
        return (
            f'cd /d "{repo_path}" && '
            f'curl.exe -sSL {bundle_url} -o "%TEMP%\\repo.bundle" && '
            f'git fetch "%TEMP%\\repo.bundle" main && '
            f'git reset --hard FETCH_HEAD'
        )
    return (
        f'cd {shlex.quote(repo_path)} && '
        f'curl -sSL {bundle_url} -o /tmp/repo.bundle && '
        f'git fetch /tmp/repo.bundle main && '
        f'git reset --hard FETCH_HEAD'
    )


def _push_weights_to_remote(
    shard: ShardConfig, remote_repo: str, weights_path: str,
) -> None:
    """scp ``<master_cwd>/<weights_path>`` to ``<repo_path>/<weights_path>``
    on the remote. Idempotent ``mkdir`` first so the parent dir exists.
    Failures propagate — a missing weights file on the remote silently
    breaks the shard."""
    print(
        f"[shard {shard.display_label}] pushing weights "
        f"{weights_path} to {shard.host}...",
        file=sys.stderr,
    )
    local_weights = (Path.cwd() / weights_path).resolve()
    weights_posix = weights_path.replace("\\", "/")
    remote_weights = _remote_join(shard, remote_repo, weights_posix)
    remote_weights_parent = _remote_join(
        shard, remote_repo, str(PurePosixPath(weights_posix).parent),
    )
    _run_subprocess(
        _ssh_cmd(shard, _build_remote_mkdir_command(
            shard, remote_weights_parent,
        )),
        timeout=30,
    )
    _run_subprocess(
        _scp_cmd(
            shard, str(local_weights), remote_weights,
            from_remote=False,
        ),
        timeout=300,
    )


def _pull_remote_replays(
    shard: ShardConfig,
    remote_repo: str,
    replays_root: Optional[str],
    capture_every: int,
    run_id_full: str,
) -> None:
    """scp every per-game ``.json.gz`` replay from the remote shard's
    replay subdir back to the master's matching subdir, mirroring what
    ``run_shard_local`` writes directly. No-op when capture is off
    (``capture_every <= 0`` or ``replays_root is None``).

    Mirrors the ls+scp loop used for npz collection. The remote shell
    listing returns no-match exit code (``ls`` / ``dir /b``) when the
    shard captured zero games; we treat that as zero files and return
    silently rather than raising.
    """
    if capture_every <= 0 or replays_root is None:
        return

    replays_posix = replays_root.replace("\\", "/")
    local_replays_dir = (
        Path.cwd() / replays_posix / run_id_full
    )
    print(
        f"[shard {shard.display_label}] collecting replays from "
        f"{shard.host}...",
        file=sys.stderr,
    )
    if _is_windows_remote(shard):
        replays_backslash = replays_posix.replace("/", "\\")
        list_cmd = (
            f'cd /d "{remote_repo}" && '
            f'dir /b {replays_backslash}\\{run_id_full}\\*.json.gz '
            f'2>nul'
        )
    else:
        list_cmd = (
            f'cd {shlex.quote(remote_repo)} && '
            f'ls {replays_posix}/{run_id_full}/*.json.gz '
            f'2>/dev/null | xargs -n1 basename'
        )
    list_proc = subprocess.run(
        _ssh_cmd(shard, list_cmd),
        capture_output=True, text=True, timeout=30,
    )
    basenames = [
        line.strip() for line in list_proc.stdout.splitlines()
        if line.strip()
    ]
    if not basenames:
        return
    local_replays_dir.mkdir(parents=True, exist_ok=True)
    for basename in basenames:
        remote_replay = _remote_join(
            shard, remote_repo, replays_posix, run_id_full, basename,
        )
        local_replay = local_replays_dir / basename
        _run_subprocess(
            _scp_cmd(
                shard, remote_replay, str(local_replay),
                from_remote=True,
            ),
            timeout=120,
        )


def run_shard_remote(
    shard: ShardConfig,
    spec: SelfPlayJobSpec,
    *,
    queue_url: str,
    run_id: str,
    iter_total_games: int,
    progress_dir: Path,
    output_dir: Path,
    run_id_prefix: str,
    stamp: str,
    bundle_url: Optional[str] = None,
    shard_ssh_timeout_s: float = _DEFAULT_SHARD_SSH_TIMEOUT_S,
) -> ShardSelfPlayReport:
    """Run one shard on a remote machine in pull mode.

    Sequence:
    1. Code sync from master's git bundle (if ``bundle_url`` is set).
    2. Push the master's current weights file to the remote at the
       same relative path. A stale weights file on the remote
       silently produces zero-sample games, so a failure here is
       fatal *for this shard*.
    3. Start a poller that periodically scps the remote progress
       JSON back for the master UI.
    4. Block on ``ssh ... run_self_play_shard.py --mode=pull`` —
       the wrapper loops fetching from /queue/take until empty.
    5. Stop poller; final progress + events scp.
    6. scp every per-batch npz back to ``output_dir``. Filenames
       follow ``{run_id}_shard_{label}_chunk_{NNNN}_{stamp}.npz``,
       which is what ``run_shard_pull`` writes server-side.
    7. scp any per-game replay JSON files back.

    Returns a :class:`ShardSelfPlayReport` describing the outcome
    (success flag, npzs collected, optional error/stage). Per-shard
    SSH / rsync failures do NOT propagate — they are caught, logged
    loudly to stderr, the per-shard progress JSON is marked
    ``status="failed"``, and the post-shard rsync stages still run
    so any data the shard managed to put on disk before the wrapper
    died is recovered. The orchestrator aggregates whatever it gets;
    one Tailscale glitch no longer kills 50 minutes of cluster work.

    The shard-level retry decision is OUT of scope here — the master
    moves on to the next shard either way. Re-runs of failed shards
    are a higher-level concern (re-queueing on the next iter).
    """
    label = shard.display_label
    run_id_full = f"{run_id}_shard_{label}"

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    local_progress_path = progress_dir / f"{run_id_full}.json"
    local_events_path = progress_dir / f"{run_id_full}_events.jsonl"

    remote_repo = shard.repo_path or ""
    remote_progress = _remote_join(
        shard, remote_repo, "training_progress", f"{run_id_full}.json",
    )
    remote_events = _remote_join(
        shard, remote_repo, "training_progress",
        f"{run_id_full}_events.jsonl",
    )

    # Track failure state so the post-shard data-collection block
    # still runs but knows whether to mark progress as failed.
    error: Optional[str] = None
    error_stage: Optional[str] = None
    poller: Optional[_ProgressPoller] = None
    local_paths: list[Path] = []

    try:
        # 1. Code sync.
        if bundle_url:
            print(
                f"[shard {label}] sync from bundle on "
                f"{shard.host}...", file=sys.stderr,
            )
            sync_cmd = _build_remote_bundle_sync_command(shard, bundle_url)
            try:
                _run_subprocess(_ssh_cmd(shard, sync_cmd), timeout=300)
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired, OSError) as e:
                error_stage = "bundle_sync"
                error = f"{type(e).__name__}: {e}"
                raise

        # 2. Weight push. Without this, iter #2+ silently produces
        # samples=0 games on remotes because their weights file is
        # stale or missing.
        try:
            _push_weights_to_remote(shard, remote_repo, spec.weights_path)
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired, OSError) as e:
            error_stage = "weight_push"
            error = f"{type(e).__name__}: {e}"
            raise

        # 3. Progress poller.
        poller = _ProgressPoller(
            shard,
            remote_progress_path=remote_progress,
            local_progress_path=local_progress_path,
        )
        poller.start()

        # 4. The actual pull-mode invocation. Blocks until queue drains.
        print(
            f"[shard {label}] pull-mode self-play on "
            f"{shard.host}...",
            file=sys.stderr,
        )
        try:
            run_cmd = _build_remote_pull_command(
                shard, spec,
                queue_url=queue_url,
                run_id=run_id,
                iter_total_games=iter_total_games,
                stamp=stamp,
            )
            # Two-layer defense against the post-shard SSH-still-
            # alive hang (Tailscale keeps TCP healthy after remote
            # explicit-exit; ssh client never sees protocol close):
            # 1. Watchdog terminates the SSH subprocess as soon as
            #    the local per-shard progress JSON shows terminal
            #    status — collapses ~10-90 min hang to ~10s grace.
            # 2. ``shard_ssh_timeout_s`` is the hard floor (only
            #    bounds the case where the JSON never reaches
            #    terminal status). On either exit path the outer
            #    per-shard try/except catches the exception and the
            #    rsync-recovery flow runs in finally.
            _run_ssh_with_terminal_status_watchdog(
                _ssh_cmd(shard, run_cmd),
                progress_path=local_progress_path,
                hard_timeout_s=shard_ssh_timeout_s,
                shard_label=label,
            )
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired, OSError) as e:
            error_stage = "self_play"
            error = f"{type(e).__name__}: {e}"
            raise
    except (subprocess.CalledProcessError,
            subprocess.TimeoutExpired, OSError) as e:
        # Loud log so the user sees the failure during a run; the
        # ShardSelfPlayReport carries the same info up to the
        # orchestrator's post-iter summary.
        print(
            f"[shard {label}] FAILURE during {error_stage}: "
            f"{type(e).__name__}: {e}. Continuing with rsync to "
            f"recover any data the shard already wrote.",
            file=sys.stderr,
        )
    finally:
        if poller is not None:
            poller.stop()
        # Final progress JSON snapshot. Best-effort even on failure:
        # the remote may have a useful progress snapshot we want to
        # mirror back before overwriting status.
        try:
            _run_subprocess(
                _scp_cmd(
                    shard, remote_progress, str(local_progress_path),
                    from_remote=True,
                ),
                timeout=30,
            )
        except Exception:  # noqa: BLE001
            pass
        # Pull the JSONL event log too — this is where machine
        # metadata, per-game wall times, and per-batch throughput
        # live for cross-machine analysis. Failure isn't fatal: if
        # the shard crashed before opening the file there's nothing
        # to fetch, but we still want the npzs back.
        try:
            _run_subprocess(
                _scp_cmd(
                    shard, remote_events, str(local_events_path),
                    from_remote=True,
                ),
                timeout=30,
            )
        except Exception:  # noqa: BLE001
            pass

    # 5. Pull all per-batch npzs back. Today (2026-05-05) we hit a
    # case where the Mac's run_self_play_shard.py finished all 329
    # games and wrote 24 npz files but ssh.exe lingered; the data
    # was on disk and the rsync below recovered every byte. So we
    # try this even on failure.
    print(
        f"[shard {label}] collecting per-batch npzs"
        f" from {shard.host}...",
        file=sys.stderr,
    )
    prefix = f"{run_id}_shard_{label}_chunk_"
    if _is_windows_remote(shard):
        # cmd.exe: ``dir /b`` lists basenames only.
        list_cmd = (
            f'cd /d "{remote_repo}" && '
            f'dir /b iterate_data\\{prefix}*.npz 2>nul'
        )
    else:
        list_cmd = (
            f'cd {shlex.quote(remote_repo)} && '
            f'ls iterate_data/{prefix}*.npz 2>/dev/null '
            f'| xargs -n1 basename'
        )
    # Don't ``check=True`` here: a shard that lost every batch race
    # (or whose self-play crashed before producing any output) is
    # not a bug at the listing layer. Both ``dir /b`` and ``ls`` exit
    # non-zero on no-match; we treat that as zero files and move on.
    try:
        list_proc = subprocess.run(
            _ssh_cmd(shard, list_cmd),
            capture_output=True, text=True, timeout=30,
        )
        basenames = [
            line.strip() for line in list_proc.stdout.splitlines()
            if line.strip()
        ]
    except (subprocess.SubprocessError, OSError) as e:
        # The listing itself failed (SSH unreachable). Record but
        # don't override an earlier-stage error.
        if error is None:
            error_stage = "list_npzs"
            error = f"{type(e).__name__}: {e}"
        print(
            f"[shard {label}] FAILURE listing remote npzs: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        basenames = []
    for basename in basenames:
        remote_npz = _remote_join(
            shard, remote_repo, "iterate_data", basename,
        )
        local_npz = output_dir / basename
        try:
            _run_subprocess(
                _scp_cmd(
                    shard, remote_npz, str(local_npz),
                    from_remote=True,
                ),
                timeout=120,
            )
            local_paths.append(local_npz)
        except (subprocess.SubprocessError, OSError) as e:
            # Record but continue — partial data from this shard is
            # still useful. Don't override an earlier-stage error.
            if error is None:
                error_stage = "scp_npz"
                error = (
                    f"{type(e).__name__}: {e} (basename={basename})"
                )
            print(
                f"[shard {label}] FAILURE pulling {basename}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )

    # 6. Pull any per-batch replay .json.gz files back. Same shape as
    # the local-shard path so the training UI's replay viewer
    # (which only knows about the master's replays_root) finds them.
    # Wrapped — replay-pull failures shouldn't sink an otherwise-
    # successful shard's data.
    try:
        _pull_remote_replays(
            shard, remote_repo,
            replays_root=spec.replays_root,
            capture_every=spec.capture_every,
            run_id_full=run_id_full,
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(
            f"[shard {label}] WARNING: replay pull failed: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )

    success = error is None
    if not success:
        _mark_shard_progress_failed(
            local_progress_path,
            shard_label=label,
            note=f"{error_stage}: {error}",
        )
    return ShardSelfPlayReport(
        shard_label=label,
        success=success,
        npz_paths=local_paths,
        error=error,
        error_stage=error_stage,
    )


def run_shard_eval_remote(
    shard: ShardConfig,
    spec: EvalJobSpec,
    *,
    queue_url: str,
    run_id: str,
    iter_total_games: int,
    progress_dir: Path,
    stamp: str,
    bundle_url: Optional[str] = None,
    shard_ssh_timeout_s: float = _DEFAULT_SHARD_SSH_TIMEOUT_S,
) -> ShardEvalReport:
    """Eval counterpart to :func:`run_shard_remote`. Mirrors the
    self-play remote runner but pushes BOTH NN weights files
    (new + old), invokes the wrapper in eval mode, and returns
    aggregated counts parsed from a JSON result file scp'd back from
    the remote.

    Like :func:`run_shard_remote`, per-shard SSH / rsync failures do
    NOT propagate. The function returns a :class:`ShardEvalReport`
    with ``success=False`` and zero counts on failure; the
    orchestrator aggregates whatever shards succeeded and decides
    whether to fail the eval (e.g. zero shards succeeded).
    """
    label = shard.display_label
    run_id_full = f"{run_id}_shard_{label}"
    progress_dir.mkdir(parents=True, exist_ok=True)
    local_progress_path = progress_dir / f"{run_id_full}.json"
    local_events_path = progress_dir / f"{run_id_full}_events.jsonl"
    local_result_path = progress_dir / f"{run_id_full}_result.json"

    remote_repo = shard.repo_path or ""
    remote_progress = _remote_join(
        shard, remote_repo, "training_progress", f"{run_id_full}.json",
    )
    remote_events = _remote_join(
        shard, remote_repo, "training_progress",
        f"{run_id_full}_events.jsonl",
    )
    remote_result = _remote_join(
        shard, remote_repo, "training_progress",
        f"{run_id_full}_result.json",
    )

    error: Optional[str] = None
    error_stage: Optional[str] = None
    poller: Optional[_ProgressPoller] = None
    zero_counts = {
        "n_games": 0, "wins_new": 0, "wins_old": 0,
        "draws": 0, "crashed": 0,
    }

    try:
        if bundle_url:
            print(
                f"[shard {label}] sync from bundle on "
                f"{shard.host}...", file=sys.stderr,
            )
            try:
                _run_subprocess(
                    _ssh_cmd(shard, _build_remote_bundle_sync_command(
                        shard, bundle_url,
                    )),
                    timeout=300,
                )
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired, OSError) as e:
                error_stage = "bundle_sync"
                error = f"{type(e).__name__}: {e}"
                raise

        try:
            _push_weights_to_remote(
                shard, remote_repo, spec.new_weights_path,
            )
            if spec.old_weights_path is not None:
                _push_weights_to_remote(
                    shard, remote_repo, spec.old_weights_path,
                )
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired, OSError) as e:
            error_stage = "weight_push"
            error = f"{type(e).__name__}: {e}"
            raise

        poller = _ProgressPoller(
            shard,
            remote_progress_path=remote_progress,
            local_progress_path=local_progress_path,
        )
        poller.start()

        print(
            f"[shard {label}] pull-mode eval on "
            f"{shard.host}...",
            file=sys.stderr,
        )
        try:
            run_cmd = _build_remote_eval_command(
                shard, spec,
                queue_url=queue_url,
                run_id=run_id,
                iter_total_games=iter_total_games,
                stamp=stamp,
            )
            # Watchdog + hard-timeout floor — same rationale as
            # run_shard_remote. Eval shards typically finish much
            # faster than self-play, so a 90-min hard floor with
            # only the SSH-protocol-close fallback was the worst
            # offender. The watchdog collapses it to ~10s grace.
            _run_ssh_with_terminal_status_watchdog(
                _ssh_cmd(shard, run_cmd),
                progress_path=local_progress_path,
                hard_timeout_s=shard_ssh_timeout_s,
                shard_label=label,
            )
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired, OSError) as e:
            error_stage = "eval"
            error = f"{type(e).__name__}: {e}"
            raise
    except (subprocess.CalledProcessError,
            subprocess.TimeoutExpired, OSError) as e:
        print(
            f"[shard {label}] FAILURE during {error_stage}: "
            f"{type(e).__name__}: {e}. Continuing with rsync to "
            f"recover any partial result.",
            file=sys.stderr,
        )
    finally:
        if poller is not None:
            poller.stop()
        for remote_path, local_path in (
            (remote_progress, local_progress_path),
            (remote_events, local_events_path),
            (remote_result, local_result_path),
        ):
            try:
                _run_subprocess(
                    _scp_cmd(
                        shard, remote_path, str(local_path),
                        from_remote=True,
                    ),
                    timeout=30,
                )
            except Exception:  # noqa: BLE001
                pass

    # If the shard never produced a result file we can't aggregate
    # its counts — but other shards may have, so this is no longer
    # fatal. Return a failure report and let the orchestrator decide.
    if not local_result_path.is_file():
        if error is None:
            error_stage = "no_result_file"
            error = (
                f"eval shard {label} produced no result file at "
                f"{local_result_path}"
            )
            print(
                f"[shard {label}] FAILURE: no result file at "
                f"{local_result_path}",
                file=sys.stderr,
            )
        _mark_shard_progress_failed(
            local_progress_path,
            shard_label=label,
            note=f"{error_stage}: {error}",
        )
        return ShardEvalReport(
            shard_label=label,
            success=False,
            counts=dict(zero_counts),
            error=error,
            error_stage=error_stage,
        )

    # Replay pull is best-effort: don't sink an otherwise-successful
    # shard's counts on a replay rsync hiccup.
    try:
        _pull_remote_replays(
            shard, remote_repo,
            replays_root=spec.replays_root,
            capture_every=spec.capture_every,
            run_id_full=run_id_full,
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(
            f"[shard {label}] WARNING: replay pull failed: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )

    try:
        counts = json.loads(
            local_result_path.read_text(encoding="utf-8"),
        )
    except (OSError, json.JSONDecodeError) as e:
        if error is None:
            error_stage = "result_parse"
            error = f"{type(e).__name__}: {e}"
            print(
                f"[shard {label}] FAILURE parsing result file: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
        _mark_shard_progress_failed(
            local_progress_path,
            shard_label=label,
            note=f"{error_stage}: {error}",
        )
        return ShardEvalReport(
            shard_label=label,
            success=False,
            counts=dict(zero_counts),
            error=error,
            error_stage=error_stage,
        )

    success = error is None
    if not success:
        _mark_shard_progress_failed(
            local_progress_path,
            shard_label=label,
            note=f"{error_stage}: {error}",
        )
    return ShardEvalReport(
        shard_label=label,
        success=success,
        counts=counts,
        error=error,
        error_stage=error_stage,
    )
