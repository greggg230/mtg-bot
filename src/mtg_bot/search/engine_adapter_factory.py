"""The single game-construction factory that selects the decision
source (B3-S1 / SOT_FLIP_CUTOVER_PLAN Step 3).

Per the umbrella's rollback constraint
(``docs/SOT_FLIP_CUTOVER_PLAN.md`` §3 + the B3 wiring spec
``docs/NN_RUST_DECISION_SOURCING_WIRING_DESIGN.md`` §3.1) the
decision-source switch MUST be a **single atomic-revert flag** read
once at game construction. This module is the ONE place the
``MtgGameStateAdapter`` vs ``RustGameStateAdapter`` choice is made; a
half-reverted state is unrepresentable because the whole game uses one
provider chosen here.

``decision_source`` lives on the frozen ``HeuristicsConfig`` (so it
travels through the existing JSON / cross-pickle plumbing for free):

* ``"python"`` (DEFAULT) — return the unchanged ``MtgGameStateAdapter``
  built exactly as the call sites built it before B3. **Byte-identical
  to pre-B3 behavior** — this module adds zero behavior change while
  the flag is off (the parity-safe landing).
* ``"rust"`` / ``"rust_shadow"`` — the Rust-backed provider. B3-S1
  ships the dispatch + the provider class; the Rust-state *loader*
  wiring at the production self-play construction sites
  (``data_gen``/``self_play`` build a Python ``engine`` from decks,
  not a Rust state) is sequenced into B3-S3/S4 (the
  ``build_matched_states`` loader path + the ``rust_shadow``
  shadow-checker). Until then, requesting a Rust source at a
  Python-engine construction site raises loudly rather than silently
  degrading (``feedback_no_fallback_hybrids`` — an honest hard
  failure is the real progress signal; NEVER a silent fallback to
  Python).

Tests / the B3-S4 bake-in construct the Rust provider directly from a
Rust ``GameState`` via :func:`make_rust_game_state_adapter`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Union

from mtg_bot.search.heuristics_config import (
    DEFAULT_DECISION_SOURCE, HeuristicsConfig,
    default_search_resolution_decisions,
)


class ResolutionPriorUnavailableError(ValueError):
    """A seat whose resolution decisions are searched uses a policy
    head (PUCT) that has no slots for them -- a ``positional_v1`` /
    ``canonical_v2`` checkpoint. It would search every resolution pause
    on all-zero priors. Extend the checkpoint (``mtg_bot
    extend-resolution-slots``) or run the seat with
    ``search_resolution_decisions=False`` (ladder ``+autotarget``)."""


def strategy_searches_resolution(strategy: Any) -> bool:
    """A built strategy's per-agent opt-in (``True`` unless its config
    set ``search_resolution_decisions=False``)."""
    return bool(getattr(strategy, "search_resolution_decisions", True))


def resolution_seats_for(
    strategies: Mapping[int, Any],
    heuristics: Optional[HeuristicsConfig],
) -> frozenset[int]:
    """The seats whose mid-resolution choices this game searches: none
    when the game-level switch
    (``HeuristicsConfig.search_resolution_decisions``, default ON) is
    off, else every seat whose strategy didn't opt out. Raises
    ``ResolutionPriorUnavailableError`` for a searched seat whose PUCT
    policy head has no resolution slots."""
    enabled = (
        heuristics.search_resolution_decisions
        if heuristics is not None
        else default_search_resolution_decisions()
    )
    if not enabled:
        return frozenset()
    seats: set[int] = set()
    for seat, strategy in strategies.items():
        if not strategy_searches_resolution(strategy):
            continue
        if not getattr(strategy, "can_prior_resolution_decisions", True):
            raise ResolutionPriorUnavailableError(
                f"seat {seat}: {getattr(strategy, 'resolution_prior_note', type(strategy).__name__)}"
                f" searches resolution decisions with a PUCT policy head "
                f"that has no resolution slots (all-zero priors). Extend "
                f"the checkpoint (mtg_bot extend-resolution-slots) or run "
                f"the seat with search_resolution_decisions=False "
                f"(ladder +autotarget)."
            )
        seats.add(int(seat))
    return frozenset(seats)


def _resolve_decision_source(
    heuristics: Optional[HeuristicsConfig],
    decision_source: Optional[str],
) -> str:
    """Resolve the effective decision source. An explicit
    ``decision_source`` argument overrides the heuristics field
    (the heuristics field is the production/distributed plumbing
    path; the explicit arg is the test / direct-construction path).
    Defaults to ``"python"`` — the safe-by-default landing."""
    if decision_source is not None:
        return decision_source
    if heuristics is not None:
        return heuristics.decision_source
    return DEFAULT_DECISION_SOURCE


def make_game_state_adapter(
    engine_state: Any,
    *,
    tactical_agent: Any = None,
    expose_combat_decisions: bool = True,
    expose_resolution_decisions: Union[bool, Iterable[int]] = False,
    heuristics: Optional[HeuristicsConfig] = None,
    decision_source: Optional[str] = None,
):
    """Construct the search ``GameState`` provider for a game.

    Under ``decision_source="python"`` (default) this returns exactly
    ``MtgGameStateAdapter(engine_state, ...)`` — the SAME object the
    call sites built before B3, with the SAME arguments. The only
    behavioral difference vs a direct ``MtgGameStateAdapter(...)``
    call is *nothing* when the flag is off.

    Under ``"rust"`` / ``"rust_shadow"`` at a Python-engine
    construction site, raises ``NotImplementedError`` (the Rust-state
    loader wiring at the production sites is B3-S3/S4). The Rust
    provider is constructed directly from a Rust state via
    :func:`make_rust_game_state_adapter` by tests + the B3-S4 bake-in.
    """
    src = _resolve_decision_source(heuristics, decision_source)

    if src == "python":
        # Lazy import to keep the factory import-cheap and avoid any
        # import cycle through the heavy engine_adapter module at
        # config-construction time.
        from mtg_bot.search.engine_adapter import MtgGameStateAdapter

        return MtgGameStateAdapter(
            engine_state,
            tactical_agent=tactical_agent,
            expose_combat_decisions=expose_combat_decisions,
            expose_resolution_decisions=expose_resolution_decisions,
            heuristics=heuristics,
        )

    if src in ("rust", "rust_shadow"):
        raise NotImplementedError(
            f"decision_source={src!r} requested at a Python-engine "
            f"construction site. A Rust-sourced game is built from a "
            f"DeckSpec via the surviving matched-state loader path, "
            f"NOT from a pre-built Python `engine` (the production "
            f"self-play sites pass a Python engine here). Build the "
            f"Rust-sourced provider from the spec instead: "
            f"make_rust_game_state_adapter(rust_state, spec=spec) for "
            f"`rust`, or build_shadow_checked_adapter(spec) for "
            f"`rust_shadow` (the B3-S4 bake-in). Enabling these at the "
            f"production construction sites is the orchestrator-"
            f"sequenced atomic flip (B3-S5) — explicitly NOT done by "
            f"B3-S1..S4. NO silent fallback to Python "
            f"(feedback_no_fallback_hybrids)."
        )

    raise ValueError(f"unknown decision_source: {src!r}")


def printing_resolver_from_spec(spec: Any) -> dict:
    """Build the B3-S2 ``name -> Python CardPrinting`` resolver from a
    ``cross_engine.DeckSpec``'s ``card_defs``, through the SAME
    ``_card_def_to_python_printing`` path ``build_python_state`` uses
    (the card parser/loader stays Python per the flip architecture,
    ``SOT_FLIP_CUTOVER_PLAN.md`` §5). The codec resolves
    ``printing.name`` / ``.is_creature`` / ``.is_spacecraft`` /
    ``.room_halves`` through this so its slot semantics are
    byte-identical to the Python path."""
    from mtg_bot.parity.cross_engine import _card_def_to_python_printing

    return {
        card.name: _card_def_to_python_printing(card)
        for card in spec.card_defs
    }


def make_rust_game_state_adapter(
    rust_state: Any,
    *,
    expose_combat_decisions: bool = True,
    printing_resolver: Optional[dict] = None,
    spec: Any = None,
):
    """Construct the Rust-backed provider directly from a live Rust
    ``GameState`` handle (e.g. from
    ``cross_engine.build_rust_state`` / ``build_matched_states`` — the
    surviving loader path per ``BRIDGE_VESTIGIALIZATION_PLAN.md``
    §2.1). Used by the B3 unit tests and the B3-S4 ``rust_shadow``
    bake-in; the production self-play sites adopt it in B3-S3/S4.

    B3-S2: pass ``spec`` (the ``DeckSpec`` the Rust state was built
    from) to auto-derive the codec ``printing_resolver``, or pass an
    explicit ``printing_resolver`` (a ``name -> CardPrinting`` map).
    Without either, the codec slot-source switch still works for the
    name+id positional axis (the Rust snapshots carry both) but a
    printing-typed lookup (``is_creature`` for an
    ``ActivateAbility`` / ``ActivateStation`` slot) raises loudly
    rather than silently mis-slotting — never a silent gap."""
    from mtg_bot.search.rust_engine_adapter import RustGameStateAdapter

    if printing_resolver is None and spec is not None:
        printing_resolver = printing_resolver_from_spec(spec)

    return RustGameStateAdapter(
        rust_state,
        expose_combat_decisions=expose_combat_decisions,
        printing_resolver=printing_resolver,
    )


def build_shadow_checked_adapter(spec: Any):
    """B3-S4 ``rust_shadow`` bake-in entry point: build a
    ``ShadowCheckedRustAdapter`` from a ``cross_engine.DeckSpec`` (Rust
    drives, Python checks in lock-step; the 4-assertion tripwire
    fires every step, raising loudly on any divergence — the umbrella's
    bad-flip detection, ``SOT_FLIP_CUTOVER_PLAN.md`` §3). The
    Step-0 C4 post-R76 scaled re-sweep + one real ``mtg_bot iterate``
    iteration run through this; enabling it at the production
    construction sites is the orchestrator-sequenced atomic flip
    (B3-S5), NOT done by B3-S1..S4."""
    from mtg_bot.search.rust_shadow import build_shadow_checked_adapter as _b

    return _b(spec)
