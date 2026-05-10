# `mtg_bot_engine` — Rust hot-path crate

Skeleton for the Rust rewrite of the engine + search hot path. Built
as a PyO3 extension and imported from Python as `import mtg_bot_engine`.

## Scope

**This crate owns:**

- `GameState`, `GameObject`, `Player` data shapes
- Clone (today)
- Future: legal-action enumeration, step / resolve / combat, MCTS

**Python keeps:**

- The oracle-text card parser (`src/mtg_bot/cards/parser/`)
- Effect / trigger / static grammar definitions
- The NN evaluator + AlphaZero training pipeline
- The Flask UI and distributed self-play orchestration
- The closed-vocabulary effect enum surfaces to Rust through the
  card-data bridge (TBD next branch)

## Build

From the worktree root, into the project's `.venv-py313`:

```bash
../../mtg-bot/.venv-py313/Scripts/maturin.exe develop \
    --release -m rust/pyproject.toml
```

That compiles and installs `mtg_bot_engine.pyd` into the venv's
`site-packages`. After that any Python in the venv can `import
mtg_bot_engine`.

If `maturin` isn't on PATH, run via the venv's python:

```bash
../../mtg-bot/.venv-py313/Scripts/python.exe -m maturin develop \
    --release -m rust/pyproject.toml
```

## What's here today

**Game state (skeleton, no game logic yet — `src/state.rs`)**

- `make_synthetic_state(num_battlefield, num_hand, num_library) -> GameState`
- `GameState.clone_once() -> GameState`
- `GameState.clone_n_times_micros(n) -> float` — micro-bench helper
- Read-only getters: `turn_number`, `battlefield_sizes`,
  `hand_sizes`, `library_sizes`, `life_totals`
- Zones are `Arc<Vec<GameObject>>` for CoW clone, matching Python's
  `_CowList` semantics — clone is an atomic refcount bump per zone.

**Card-data bridge (`src/cards.rs`)**

Closed-vocabulary Rust types + a Python-dict parser that's the
single point of validation. Python builds plain dicts describing
each card; the Rust dict parser converts to typed enums or fails
loudly (unknown `kind`, missing keys, type mismatches all raise at
`register_printing` time, not at game-resolution time).

- `Registry` (PyClass) — Python holds it explicitly. No global state.
  - `Registry().register_printing(dict) -> u32` — returns the
    stable printing id. Validates everything; raises `KeyError` for
    missing required keys, `ValueError` for unknown kinds /
    malformed values.
  - `Registry.get_printing_name(id) -> str`
  - `Registry.get_printing_mana_cost(id) -> dict` — round-trip
  - `Registry.get_printing_effects(id) -> list[dict]` — round-trip
  - `Registry.register_n_times_micros(dict, n) -> float` — bench helper
- Closed Rust enums:
  - `Effect::DealDamage { amount: i32, target: TargetFilter }`
  - `TargetFilter::AnyPlayer | AnyCreature`
- `ManaCost { generic, white, blue, black, red, green: u8 }` —
  hybrid / X / snow deferred until a card actually needs them.

The vocabulary will grow one variant per follow-up commit, each
with a parity test against the Python equivalent on a real EoE
card. **No fallback-to-opaque-blob path** — every variant the
Python parser emits has a Rust counterpart, and the parser fails
loudly on unknown `kind` so missing variants surface immediately.

The Python smoke + bridge tests at `tests/test_rust_engine_smoke.py`
and `tests/test_rust_card_bridge.py` skip gracefully when the
extension isn't installed.

Benches:

- `scripts/bench_rust_clone.py` — head-to-head clone perf vs
  Python's `GameState.clone()` and `copy.deepcopy`.
- `scripts/bench_rust_register_printings.py` — bridge parse cost.
  Headline: ~0.6us per Python `register_printing` call;
  5000 cards bulk-register in ~3.8ms.

## Why a separate `pyproject.toml`?

The project's main `pyproject.toml` uses setuptools and ships the
pure-Python `mtg_bot` package. Mixing setuptools and maturin in a
single project file is awkward; a sibling `rust/pyproject.toml`
keeps the Rust extension a separately-installable wheel that can be
developed and rebuilt without touching the main Python build.
