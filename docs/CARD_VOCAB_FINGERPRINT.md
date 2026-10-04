# Card-vocabulary fingerprint

Status: `feature/vocab-fingerprint` (2026-10-04). It has to merge
**before** any card-pool branch (`feature/cube-card-pool` →
`feature/cube-roto-coverage` → `feature/roto-integration` →
`feature/roto-default-pool-leyline-wizard`), because those branches
renumber every card the structured NN knows.

## 1. The problem

The structured network looks every card up by an integer **printing
id**. Two parameters are indexed by it:

| parameter | shape | used for |
|---|---|---|
| `printing_embed.weight` | `(n_printings, emb_dim)` | the card embedding; shared by the `PRINTING` and `TARGET_PRINTING` token fields |
| `hand_belief_queries` | `(n_printings, emb_dim)` | Step-4 hand-belief stub (trained as a parameter, loss not wired) |

`tests/training/test_card_vocab.py::test_per_printing_params_are_exactly_the_n_printings_rows`
pins that these are the only ones.

### How the id is built (scheme `sorted_names_v1`)

`state_tokenizer.printing_id` (unchanged since the tokenizer landed,
commit `46447461`, 2026-05-12):

1. **Source**: the active `CardSource`
   (`mtg_bot.cards.source.get_active_source()`), in production
   `OracleParserCardSource`. On `main` it walks four Scryfall dumps
   (`data/edge_of_eternities.json`, `stellar_sights.json`,
   `special_guests_eoe.json`, `duskmourn.json`) and keeps every card some
   builder **claims**. On the cube/roto branches the dumps come from
   the pool registry (`MTG_BOT_CARD_POOL` = `all` default / `cube` /
   `eoe` / `roto`).
2. **Ordering**: `sorted(source.get_printings().keys())`; the raw id of
   `names[i]` is `i + 1`; `0` is `PRINTING_NA` (unknown cards, tokens
   such as Robot/Lander, and the opponent's hidden hand cards).
3. **Bucketing**: the embedding row is `raw` if `raw < n_printings`,
   else `1 + (raw - 1) % (n_printings - 1)`. `n_printings` is the
   model's `StructuredModelConfig.n_printings` at inference and the
   tokenizer default `DEFAULT_N_PRINTINGS = 320` in data-gen.

So the vocabulary changes whenever the pool changes **or** the parser
claims one more card, and one added name renumbers every card that sorts
after it. Nothing recorded which list a checkpoint was trained on.

### What the existing checkpoints actually use

Reconstructed from every distinct `git_commit` in both registries
(`mtg_bot card-vocab export --git-commit <sha>`):

* **Every** checkpoint, `trained_weights_s2/v1`–`v26` (18 distinct
  commits, 2026-09-22 → 09-28, including the off-main `916b8972`
  root-value and `c25d2deb` aws-boot-fixes commits) **and** the old
  `trained_weights/` lineage (`v1`–`v14`, `v13r`, `v13s`; 14 commits,
  May 2026), uses one vocabulary: **475 names, fingerprint `25ac27bcad11…`**,
  which is also `main` today (`24db6758`). It is committed as
  `tests/fixtures/card_vocab/main_24db6758.json`.
* Every one has `n_printings = 320`, so raw ids 320–475 already fold
  onto rows 1–156: 156 rows are shared by two cards, 163 rows by one.
  The remap below preserves that sharing exactly for the old cards.

What the card-pool branches do to it:

| vocabulary | names | vs `main` (475) |
|---|---|---|
| `feature/cube-card-pool` `all` (`0b9050ff`) | 993 | +518, 475 renumbered |
| `feature/roto-default-pool-leyline-wizard` `all` (`d90323b9`) | 1323 | +848, **475 of 475 renumbered**, 474 change row at cap 320 |
| roto branch, `MTG_BOT_CARD_POOL=eoe` | 482 | +7 (newly claimed EoE-era cards + Leyline), 474 renumbered |
| cube `all` → roto `all` | 993 → 1323 | 992 of 993 renumbered |

No pool setting on the roto branch reproduces `main`'s id space: even
`eoe` claims 7 more cards.

## 2. Where the vocabulary is consumed

| consumer | how | covered by |
|---|---|---|
| `state_tokenizer.printing_id` / `tokenize_state` | name → raw id → row, at the caller's cap | single source: `card_vocab.current_card_vocab()`; bucketing is `card_vocab.row_for_raw_id` |
| `StructuredModel` (`printing_embed`, `hand_belief_queries`) | rows indexed by id | remap on load |
| `NNEvaluator` (search, self-play, eval, ladder, capture, bench, scripts) | builds the model, loads `weights.pt`, tokenizes at `model.config.n_printings` | `card_vocab.load_structured_checkpoint` |
| `NNPimcConfig.build` (every worker game, `mtg_bot ladder`, local eval) | `NNEvaluator(weights_path=...)` with dims read from the sibling `metadata.json` | vocabulary resolved next to the weights by `NNEvaluator` |
| `NNEvaluator.from_metadata` (`calibrate-value`, replay annotator) | passes `ModelMetadata.card_vocab` | same |
| `SharedTensorNNEvalServer` (bench only) | own model build + load; handle carries the cap | same loader; cap read from the loaded model |
| `NNEvalServer` | legacy flat MLP only (no printing identity) | n/a |
| data-gen (`data_gen._play_one_game_recording`) and the worker's per-game encoder (`worker_daemon_production_runner._encode_npz_bytes`) | **pre-tokenised** printing ids stored in every structured `.npz` | shards record vocabulary + cap (`structured_card_vocab`) |
| trainer (`mtg_bot train --architecture structured`) | trains on shard ids; registers the model | resolves the shards' single id space, stamps the model with it |
| `calibrate-value` | loaded model + held-out shards | refuses shards from another id space |
| coordinator (`mtg_bot serve`, `/v2/weights/<v>[/metadata]`) | serves `weights.pt` + `metadata.json`; does not tokenize | serves the sidecar-merged metadata |
| worker weights cache (`worker_daemon_claim_loop.fetch_weights_if_missing`) | re-fetches the pair whenever the served metadata's sha changes | gets the vocabulary inside `metadata.json`; no code change |
| `mtg_bot iterate` | reads bootstrap metadata, launches stages | preflight: refuses an unstamped structured bootstrap on the master |
| Rust engine | none: Rust states are projected to Python printings and tokenized by name in Python; `rust_determinize`'s `printing_id` is the engine's internal card-def id, not the NN id | n/a |
| `training/deck_belief.py` (Step 2, not landed) | its own corpus-built vocabulary, already saved inside its checkpoint | n/a |
| `features.py` + flat MLPs | 51 hand-built features, no card identity | n/a |

## 3. Metadata format

Every structured checkpoint's `metadata.json` carries:

```json
"card_vocab": {
  "format": 1,
  "id_scheme": "sorted_names_v1",
  "size": 475,
  "sha256": "25ac27bcad1112746e1e2c85cd530394bf9479098eb7a18842267921bd715312",
  "names": ["...", "..."],
  "provenance": {"source": "training_shards", "n_stamped_files": 3000, "n_printings_cap": 320, "git_commit": "...", "stamped_at": "..."}
}
```

* `sha256` = SHA-256 of `id_scheme + "\n" + "\n".join(names)`; a quick
  equality check. `from_dict` verifies format, scheme, sortedness, size
  and hash.
* The bucketing cap is the model's existing `n_printings` (in
  `training.structured_config`), so it is not duplicated here.
* `ModelMetadata.card_vocab` is a first-class field;
  `Registry.register` refuses a structured checkpoint without a valid one
  and writes nothing.
* **Sidecar stamp** for pre-fingerprint checkpoints:
  `<weights>.card_vocab.json` (in a registry: `vN/weights.pt.card_vocab.json`),
  `{"format": 1, "weights_sha256": "...", "card_vocab": {...}}`. Per weights
  FILE, so two checkpoints in one directory never share a stamp. The
  weights hash means a stale stamp can never apply to a different
  checkpoint. `Registry.load_metadata` and the coordinator merge it into
  `card_vocab`; the files on disk stay byte-identical.
* **Shards**: structured `.npz` files gain `structured_card_vocab`, a
  `uint8` UTF-8 JSON `{format, id_scheme, sha256, n_printings_cap, names}`
  (~7 KB compressed per file). Unknown keys are ignored by every reader,
  so old code still reads new shards.

## 4. Remap semantics

`card_vocab.adapt_structured_state_dict`, called by the single loader
`card_vocab.load_structured_checkpoint`:

* **Same vocabulary** (sha equal): the state_dict is handed through
  untouched, so the load is bit-identical to before. (Verified on copies of
  s2 `v1`, `v8`, `v22`, `v26`.)
* **Changed vocabulary**:
  * The table grows to `max(old n_printings, len(current) + 1)` so every
    current card gets its own row. It never shrinks. Keeping the old
    cap would put cards with different trained rows into one shared
    row, so no remap could keep them all.
  * **Kept** cards (name in both): the new row is a bit-exact copy of
    the row the old tokenizer used for that name, including a modulo-bucketed
    row. Row 0 (NA) is copied too.
  * **New** cards: the row is exactly what a freshly constructed
    `StructuredModel` of the new config holds there (`nn.Embedding`
    N(0,1) for `printing_embed`, N(0, 0.02) for `hand_belief_queries`). It
    is built under `torch.random.fork_rng` with a seed derived from both
    fingerprints, so every worker that remaps the same checkpoint gets
    the same network and the caller's RNG stream is untouched.
  * **Dropped** cards: their rows are simply not carried over.
  * All other parameters are passed through unchanged.
* Logged at WARNING, once per (weights, stored, current) per process:
  `CARD VOCABULARY REMAP: checkpoint vocab 25ac27bcad11 (475 names, n_printings 320) -> current vocab 2aee5d64a32e (1323 names, n_printings 1324): kept 475, new 848 [...], dropped 0. ...`
  The report is on `NNEvaluator.card_vocab_report` /
  `SharedTensorNNEvalServer.card_vocab_report`.
* The evaluator then tokenizes at the grown cap, so the remapped network
  computes exactly what the old one did for every state made only of
  kept cards. (`test_enlarged_vocabulary_preserves_network_outputs`
  asserts bit-equal value and policy logits on a real game state, and
  that a naive load does not.)
* **Refusal**: a structured checkpoint with neither an embedded
  vocabulary nor a sidecar raises `MissingCardVocabError` with the stamp
  command. A mismatched sidecar, or a sidecar for different weights, raises
  `CardVocabIntegrityError`. Nothing is ever inferred.
* `NNEvaluator.save` stamps structured weights with the evaluator's
  current vocabulary (sidecar), so a bare saved file round-trips.

### Training data

Shards are pre-tokenised and modulo-bucketed, so their ids cannot be
mapped back to names (row `r` may be two cards). They are never
remapped. Instead:

* `mtg_bot train` resolves the one `(vocabulary, cap)` of the training
  window, and **refuses**:
  * shards from two id spaces (e.g. a worker on a stale commit),
  * unstamped legacy shards, unless `--shard-card-vocab FILE` (or
    `$MTG_BOT_SHARD_CARD_VOCAB`, inherited by `mtg_bot iterate`'s train
    subprocess) asserts their vocabulary. Legacy shards were all tokenised
    at cap 320 (`LEGACY_SHARD_N_PRINTINGS_CAP`, a fact of history),
  * a model `n_printings` different from the shards' cap.
* The trained model is stamped with **the shards' vocabulary**, not the
  trainer's current pool. A model trained on the 475-name corpus after
  the roto merge is therefore correctly a 475-name model, and is remapped
  when loaded.
* `calibrate-value` refuses held-out shards whose `(vocabulary, cap)`
  differs from the loaded (possibly remapped) model's input space.
* Zero-row unstamped shards (crashed games) carry no ids and are
  counted, not judged.

## 5. Load paths covered

`NNEvaluator.__init__` (+ `from_metadata`), `NNPimcConfig.build` (worker
self-play/eval, ladder, local eval), the replay annotator and capture,
`calibrate-value`, `SharedTensorNNEvalServer`, the bench kinds and
`scripts/*` (all go through `NNEvaluator`), the coordinator's metadata
endpoint, the worker weights cache (via served metadata), `mtg_bot
iterate` (master preflight), `Registry.load_metadata` / `register`, and
the trainer and data-gen on the shard side. `NNEvalServer` and the
flat-MLP paths load no printing identity.

## 6. Runbook: stamping the existing checkpoints

The vocabulary for **all** existing checkpoints is the 475-name list,
`25ac27bcad11`. Stamp from each checkpoint's own recorded commit (the
tool also cross-checks it). This writes only
`vN/weights.pt.card_vocab.json` files; `weights.pt` and `metadata.json`
stay byte-identical:

```bash
# from the main checkout, after this branch is merged
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli card-vocab stamp \
    --registry-root trained_weights_s2 --all-versions --use-metadata-commit --sidecar
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli card-vocab show \
    --registry-root trained_weights_s2 --all-versions --compare-current
# optional, the archived lineage
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli card-vocab stamp \
    --registry-root trained_weights --all-versions --use-metadata-commit --sidecar
```

(~4 s per distinct commit: 18 for s2, 14 for the archived lineage.)
To keep the live registry untouched instead, write stamped copies with
`--out-root <dir>`.

**Legacy corpus.** Every shard in `iterate_data_s2/` predates the
fingerprint. To keep training on it, export the vocabulary once and
assert it:

```bash
PYTHONPATH=src .venv-py313/Scripts/python.exe -m mtg_bot.cli card-vocab export \
    --git-commit 24db6758 --out training_progress_s2/card_vocab_legacy_475.json
export MTG_BOT_SHARD_CARD_VOCAB=training_progress_s2/card_vocab_legacy_475.json   # or --shard-card-vocab
```

**Merge order / cluster.**

1. Merge this branch. Restart the coordinator and **every** worker
   (desktop daemon + Mac + AWS; check `git_commit` in each heartbeat).
   Run the stamp above **before** the next `iterate`; the master
   preflight refuses an unstamped bootstrap otherwise.
2. A worker still on pre-fingerprint code ignores the stamp and, like
   today, does not check. Its shards carry no stamp, so the trainer
   treats them as legacy: refused unless a legacy vocabulary is
   asserted, and the assertion must equal the stamped shards' vocabulary
   in the same window. A worker on a different card pool writes a
   different stamp, and the trainer refuses the mix.
3. When a card-pool branch merges, current checkpoints are remapped at
   load (kept 475 / new 848 for roto `all`). New shards carry the new
   vocabulary, so the trainer will refuse to mix them with the 475-name
   corpus. Train on one id space per run.

## 7. Encoding versions

The action-encoding codec and the printing-id scheme are unchanged, so
this is **not** a v-chain break. Existing checkpoints load bit-identically
once stamped. Future id schemes bump `card_vocab.id_scheme`. The hash
covers it and `CardVocab` refuses an unknown scheme.

## 8. Known limits / follow-ups

* **Modulo bucketing.** `DEFAULT_N_PRINTINGS = 320` is below every real
  vocabulary (475 today, 1323 after roto), so data-gen and freshly
  trained models share embedding rows between unrelated cards (~4 cards
  per row at 1323). The remap preserves whatever sharing a checkpoint
  had, but new training on the roto pool will collide heavily unless
  `n_printings` is raised. That is a training decision (a v-chain break),
  out of scope here.
* Remapped checkpoints (grown `n_printings`) cannot be fine-tuned on
  cap-320 shards: the trainer refuses the cap mismatch. Training is
  always from scratch today, so nothing does this.
* Separate pre-existing finding, not changed here: `NNPimcConfig.build`
  (every worker game, `mtg_bot ladder`) reads the trunk dims from
  `metadata.json` but not `value_calibration` / `value_head_input`, so
  those paths run the value head uncalibrated.

## 9. Compatibility check against the roto branch (2026-10-04)

Done in a throwaway **detached** worktree (no branch, nothing pushed):
`origin/feature/roto-default-pool-leyline-wizard` (`d90323b9`) with
`git merge --no-ff --no-commit feature/vocab-fingerprint`.

* **Merge**: clean, no conflicts. Only `CLAUDE.md` and
  `training/data_gen.py` were auto-merged, in regions far apart. At
  merge time, update the roto branch's CLAUDE.md line "Networks trained
  on the EoE pool ... do not transfer to the cube pool; retrain from
  scratch": they now load, with every shared card's embedding intact
  and new cards at fresh init.
* **Real checkpoints**: copies of s2 `v1`, `v8`, `v22`, `v26` were
  stamped on this branch (`card-vocab stamp --use-metadata-commit
  --sidecar`), then loaded in the merged tree via
  `Registry.load_metadata` + `NNEvaluator.from_metadata`. Each logged
  `CARD VOCABULARY REMAP ... kept 475, new 848, dropped 0` with
  `n_printings 320 -> 1324`. All 475 shared cards' rows are bit-exact in
  both `printing_embed` and `hand_belief_queries` (each copied from its
  old modulo-bucketed row), and every other parameter is identical.
* **Outputs**: v22 on a real game state (shared EoE cards) gives
  **bit-equal value and policy logits** in the remapped 1323-name space
  and in its original 475-name / cap-320 space. A naive load of the same
  weights under the roto ids differs (policy logits up to 1.16 apart).
* **Distributed path**: in the merged tree, the coordinator app served
  v22's sidecar-merged metadata, `fetch_weights_if_missing` cached it
  (the cached `metadata.json` carries `card_vocab`), and
  `NNPimcConfig.build` on that cache loaded the remapped net.
* **Tests in the merged tree**: full suite, 53 failed / 20691 passed.
  That is exactly `main`'s 52 baseline failures plus
  `test_shrunk_vocabulary_drops_removed_cards`, which assumed the
  vocabulary fits in 600 rows; it was fixed on this branch
  (`35d774e2`) by deriving table sizes from the live vocabulary. With
  that fix merged in, all 224 card-vocab-related tests pass in the
  merged tree.
