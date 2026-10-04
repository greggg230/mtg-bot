"""Tests for ``mtg_bot.training.card_vocab`` — the card-vocabulary
fingerprint, checkpoint resolution, and remap-on-load.

The load-bearing guarantees pinned here:

* every structured checkpoint resolves to exactly one stored vocabulary
  (embedded metadata, or a sidecar verified against the weights), and an
  unstamped one is REFUSED;
* remapping keeps every shared card's trained row bit-for-bit (including
  out of a modulo-bucketed table), gives new cards exactly a fresh
  model's initialisation, drops removed cards, and logs loudly;
* a checkpoint built on one vocabulary and loaded under an enlarged one
  produces the SAME network outputs for a game state of shared cards —
  the end-to-end property the fingerprint exists for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from mtg_bot.cards.source import get_active_source, set_active_source
from mtg_bot.training import card_vocab as cv
from mtg_bot.training.card_vocab import (
    CardVocab,
    CardVocabError,
    CardVocabIntegrityError,
    MissingCardVocabError,
    PER_PRINTING_PARAMS,
    adapt_structured_state_dict,
    current_card_vocab,
    effective_metadata_dict,
    resolve_checkpoint_card_vocab,
    row_for_raw_id,
    sidecar_path_for,
    write_sidecar,
)
from mtg_bot.training.structured_model import (
    StructuredModel,
    StructuredModelConfig,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "card_vocab"
MAIN_VOCAB_FILE = FIXTURES / "main_24db6758.json"
ROTO_VOCAB_FILE = FIXTURES / "roto_default_pool_d90323b9.json"

SMALL = dict(
    emb_dim=16, n_heads=2, n_layers=1, hidden_dim=32, action_history_n=4,
)


def _small_config(**overrides) -> StructuredModelConfig:
    return StructuredModelConfig(**{**SMALL, **overrides})


class _FakeSource:
    """A card source whose printing NAMES are chosen by the test (the
    values are real printings; only the keys feed the vocabulary)."""

    def __init__(self, printings: dict) -> None:
        self._printings = dict(printings)

    def get_printings(self) -> dict:
        return dict(self._printings)


@pytest.fixture
def restore_source():
    original = get_active_source()
    yield
    set_active_source(original)


# ---------------------------------------------------------------------
# CardVocab value type
# ---------------------------------------------------------------------


def test_from_names_sorts_dedups_and_numbers_from_one():
    v = CardVocab.from_names(["b", "a", "c", "a"])
    assert v.names == ("a", "b", "c")
    assert v.size == 3
    assert v.id_map == {"a": 1, "b": 2, "c": 3}
    assert len(v.sha256) == 64 and v.short_sha == v.sha256[:12]


def test_sha_is_order_and_scheme_sensitive():
    assert cv.card_vocab_sha256(["a", "b"], "s") != cv.card_vocab_sha256(
        ["b", "a"], "s",
    )
    assert cv.card_vocab_sha256(["a"], "s1") != cv.card_vocab_sha256(
        ["a"], "s2",
    )
    assert CardVocab(("a", "b")).sha256 == CardVocab(("a", "b")).sha256


@pytest.mark.parametrize(
    "names",
    [("b", "a"), ("a", "a"), ("",), ("a\nb",), (1,)],
)
def test_invalid_names_rejected(names):
    with pytest.raises(CardVocabError):
        CardVocab(names)


def test_unsupported_scheme_rejected():
    with pytest.raises(CardVocabError, match="unsupported"):
        CardVocab(("a",), "alphabet_soup_v9")


def test_dict_round_trip_with_provenance():
    v = CardVocab.from_names(["x", "y"])
    d = v.to_dict(provenance={"git_commit": "abc"})
    assert d["provenance"] == {"git_commit": "abc"}
    assert d["size"] == 2 and d["names"] == ["x", "y"]
    assert CardVocab.from_dict(json.loads(json.dumps(d))) == v
    assert "provenance" not in v.to_dict()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(sha256="0" * 64), "does not match"),
        (lambda d: d.update(size=99), "size"),
        (lambda d: d.update(format=2), "format"),
        (lambda d: d.update(names="x"), "list"),
        (lambda d: d.update(names=["y", "x"]), "sorted"),
        (lambda d: d.update(id_scheme="nope"), "unsupported"),
    ],
)
def test_from_dict_rejects_tampering(mutate, match):
    d = CardVocab.from_names(["x", "y"]).to_dict()
    mutate(d)
    with pytest.raises(CardVocabIntegrityError, match=match):
        CardVocab.from_dict(d)


def test_from_dict_rejects_non_mapping():
    with pytest.raises(CardVocabIntegrityError, match="JSON object"):
        CardVocab.from_dict(["x"])


def test_coerce_accepts_vocab_dict_and_wrappers():
    v = CardVocab.from_names(["x"])
    assert cv.coerce_card_vocab(v) is v
    assert cv.coerce_card_vocab(v.to_dict()) == v
    assert cv.coerce_card_vocab({"card_vocab": v.to_dict()}) == v
    with pytest.raises(CardVocabIntegrityError):
        cv.coerce_card_vocab({"card_vocab": {}})
    with pytest.raises(CardVocabIntegrityError):
        cv.coerce_card_vocab(42)


def test_vocab_file_round_trip_and_errors(tmp_path):
    v = CardVocab.from_names(["p", "q"])
    path = tmp_path / "sub" / "v.json"
    cv.write_card_vocab_file(path, v, provenance={"method": "test"})
    assert cv.load_card_vocab_file(path) == v
    assert json.loads(path.read_text())["provenance"] == {"method": "test"}
    with pytest.raises(CardVocabError, match="cannot read"):
        cv.load_card_vocab_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"names": ["b", "a"]}))
    with pytest.raises(CardVocabIntegrityError, match="bad.json"):
        cv.load_card_vocab_file(bad)


def test_committed_fixtures_are_valid_vocabularies():
    main = cv.load_card_vocab_file(MAIN_VOCAB_FILE)
    roto = cv.load_card_vocab_file(ROTO_VOCAB_FILE)
    assert main.size == 475
    assert roto.size == 1323
    assert set(main.names) <= set(roto.names)


# ---------------------------------------------------------------------
# Row bucketing + the current vocabulary
# ---------------------------------------------------------------------


def test_row_for_raw_id_matches_tokenizer_bucketing():
    assert row_for_raw_id(5, 320) == 5
    assert row_for_raw_id(319, 320) == 319
    assert row_for_raw_id(320, 320) == 1
    assert row_for_raw_id(475, 320) == 156
    with pytest.raises(ValueError):
        row_for_raw_id(3, 1)


def test_current_vocab_is_sorted_active_source_names():
    v = current_card_vocab()
    assert v.names == tuple(sorted(get_active_source().get_printings()))
    assert current_card_vocab() is v  # cached per source


def test_current_vocab_follows_source_swaps(restore_source):
    real = get_active_source().get_printings()
    names = sorted(real)[:5]
    set_active_source(_FakeSource({n: real[n] for n in names}))
    swapped = current_card_vocab()
    assert swapped.names == tuple(names)
    from mtg_bot.training.state_tokenizer import (
        _printing_to_id_map, printing_id,
    )

    assert _printing_to_id_map() == swapped.id_map
    assert printing_id(real[names[3]]) == 4
    assert printing_id(real[sorted(real)[-1]]) == 0  # not in vocab -> NA


# ---------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------


def _weights(tmp_path: Path, name: str = "weights.pt") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    torch.save({"x": torch.zeros(1)}, p)
    return p


def test_resolve_missing_is_refused_with_instructions(tmp_path):
    w = _weights(tmp_path)
    with pytest.raises(MissingCardVocabError) as exc:
        resolve_checkpoint_card_vocab(w)
    msg = str(exc.value)
    assert "never guessed" in msg and "card-vocab stamp" in msg


def test_resolve_from_explicit_metadata_and_sidecar(tmp_path):
    v = CardVocab.from_names(["a", "b"])
    w = _weights(tmp_path)
    r = resolve_checkpoint_card_vocab(w, explicit=v)
    assert (r.vocab, r.origin) == (v, "explicit")
    r = resolve_checkpoint_card_vocab(w, metadata={"card_vocab": v.to_dict()})
    assert (r.vocab, r.origin) == (v, "metadata")

    class _Meta:
        card_vocab = v.to_dict()

    assert resolve_checkpoint_card_vocab(w, metadata=_Meta()).vocab == v
    # metadata given but without a vocab -> sibling metadata.json is NOT
    # consulted; nothing else -> refused.
    with pytest.raises(MissingCardVocabError):
        resolve_checkpoint_card_vocab(w, metadata={"architecture": "structured"})
    (tmp_path / "metadata.json").write_text(json.dumps({"card_vocab": v.to_dict()}))
    r = resolve_checkpoint_card_vocab(w)
    assert (r.vocab, r.origin) == (v, "metadata")
    assert r.location.endswith("metadata.json")


def test_resolve_from_sidecar_verifies_weights(tmp_path):
    v = CardVocab.from_names(["a", "b"])
    w = _weights(tmp_path)
    target = write_sidecar(v, w, provenance={"k": 1})
    assert target == sidecar_path_for(w)
    assert target.name == "weights.pt.card_vocab.json"
    r = resolve_checkpoint_card_vocab(w)
    assert (r.vocab, r.origin) == (v, "sidecar")
    # Same vocab embedded too -> fine.
    assert resolve_checkpoint_card_vocab(w, explicit=v).vocab == v
    # Disagreeing embedded vocab -> integrity error.
    with pytest.raises(CardVocabIntegrityError, match="stamps"):
        resolve_checkpoint_card_vocab(w, explicit=CardVocab.from_names(["z"]))
    # The weights change under the sidecar -> the stamp no longer applies.
    torch.save({"x": torch.ones(3)}, w)
    with pytest.raises(CardVocabIntegrityError, match="different checkpoint"):
        resolve_checkpoint_card_vocab(w)


def test_sidecars_are_per_weights_file(tmp_path):
    a = _weights(tmp_path, "a.pt")
    b = _weights(tmp_path, "b.pt")
    torch.save({"x": torch.ones(2)}, b)
    va, vb = CardVocab.from_names(["a"]), CardVocab.from_names(["b"])
    write_sidecar(va, a)
    write_sidecar(vb, b)
    assert resolve_checkpoint_card_vocab(a).vocab == va
    assert resolve_checkpoint_card_vocab(b).vocab == vb


@pytest.mark.parametrize(
    "content, match",
    [
        ("{not json", "unreadable"),
        (json.dumps({"format": 9}), "not a format-1"),
        (json.dumps({"format": 1, "card_vocab": {"format": 1}}), "names"),
    ],
)
def test_bad_sidecars_raise(tmp_path, content, match):
    w = _weights(tmp_path)
    sidecar_path_for(w).write_text(content)
    with pytest.raises(CardVocabIntegrityError, match=match):
        resolve_checkpoint_card_vocab(w)


def test_sidecar_for_missing_weights(tmp_path):
    v = CardVocab.from_names(["a"])
    w = _weights(tmp_path)
    write_sidecar(v, w)
    sidecar = sidecar_path_for(w)
    with pytest.raises(CardVocabIntegrityError, match="not found"):
        cv.read_sidecar(sidecar, tmp_path / "gone.pt")


def test_resolve_unreadable_metadata_json(tmp_path):
    w = _weights(tmp_path)
    (tmp_path / "metadata.json").write_text("{nope")
    with pytest.raises(CardVocabError, match="cannot read"):
        resolve_checkpoint_card_vocab(w)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"card_vocab": {"format": 1, "names": ["b", "a"]}}),
    )
    with pytest.raises(CardVocabIntegrityError, match="metadata.json"):
        resolve_checkpoint_card_vocab(w)


def test_file_sha256_cache_tracks_rewrites(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"one")
    first = cv.file_sha256(p)
    assert cv.file_sha256(p) == first
    p.write_bytes(b"two-longer")
    assert cv.file_sha256(p) != first


def test_effective_metadata_merges_sidecar_without_touching_files(tmp_path):
    v = CardVocab.from_names(["a", "b"])
    w = _weights(tmp_path)
    meta = {"architecture": "structured", "version": "v1"}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    assert effective_metadata_dict(tmp_path) == meta
    write_sidecar(v, w, provenance={"git_commit": "c0ffee"})
    before = (tmp_path / "metadata.json").read_bytes()
    merged = effective_metadata_dict(tmp_path)
    assert (tmp_path / "metadata.json").read_bytes() == before
    assert CardVocab.from_dict(merged["card_vocab"]) == v
    assert merged["card_vocab"]["provenance"] == {"git_commit": "c0ffee"}
    # Embedded + same sidecar: returned as-is.
    (tmp_path / "metadata.json").write_text(
        json.dumps({**meta, "card_vocab": v.to_dict()}),
    )
    assert effective_metadata_dict(tmp_path)["card_vocab"]["sha256"] == v.sha256
    # Embedded + DIFFERENT sidecar: refused.
    (tmp_path / "metadata.json").write_text(
        json.dumps({**meta, "card_vocab": CardVocab.from_names(["q"]).to_dict()}),
    )
    with pytest.raises(CardVocabIntegrityError):
        effective_metadata_dict(tmp_path)


# ---------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------


def test_registry_refuses_unstamped_structured_and_writes_nothing(tmp_path):
    from mtg_bot.training.registry import ModelMetadata, Registry

    w = _weights(tmp_path / "src")
    reg = Registry(tmp_path / "reg")
    meta = ModelMetadata(
        version="", created_at="", feature_dim=0, hidden_sizes=[],
        architecture="structured",
    )
    with pytest.raises(MissingCardVocabError, match="refusing to register"):
        reg.register(w, meta)
    assert reg.list_versions() == [] and reg.next_version_id() == "v1"
    meta.card_vocab = {"format": 1, "names": ["b", "a"]}
    with pytest.raises(CardVocabIntegrityError):
        reg.register(w, meta)
    # MLP checkpoints carry no printing identity; no vocab needed.
    mlp = ModelMetadata(
        version="", created_at="", feature_dim=51, hidden_sizes=[8],
    )
    assert reg.register(w, mlp) == "v1"


def test_registry_round_trips_card_vocab_and_merges_sidecar(tmp_path):
    from mtg_bot.training.registry import ModelMetadata, Registry

    v = CardVocab.from_names(["a", "b"])
    w = _weights(tmp_path / "src")
    reg = Registry(tmp_path / "reg")
    version = reg.register(w, ModelMetadata(
        version="", created_at="", feature_dim=0, hidden_sizes=[],
        architecture="structured", card_vocab=v.to_dict(),
    ))
    loaded = reg.load_metadata(version)
    assert CardVocab.from_dict(loaded.card_vocab) == v
    assert ModelMetadata.from_dict(loaded.to_dict()).card_vocab == loaded.card_vocab
    # A legacy (unstamped) version stamped by sidecar: load_metadata
    # surfaces the stamp; the files on disk stay byte-identical.
    legacy = tmp_path / "reg" / "v9"
    legacy.mkdir()
    lw = _weights(legacy)
    (legacy / "metadata.json").write_text(json.dumps({
        "version": "v9", "created_at": "x", "feature_dim": 0,
        "hidden_sizes": [], "architecture": "structured",
    }))
    assert reg.load_metadata("v9").card_vocab == {}
    before = (legacy / "metadata.json").read_bytes()
    write_sidecar(v, lw)
    assert CardVocab.from_dict(reg.load_metadata("v9").card_vocab) == v
    assert (legacy / "metadata.json").read_bytes() == before


# ---------------------------------------------------------------------
# Remap
# ---------------------------------------------------------------------


def test_per_printing_params_are_exactly_the_n_printings_rows():
    """Guard: if a new parameter indexed by printing id is added to
    StructuredModel, the remap must learn about it."""
    distinctive = 37
    model = StructuredModel(_small_config(n_printings=distinctive))
    rows = {
        k for k, t in model.state_dict().items()
        if t.dim() >= 1 and t.shape[0] == distinctive
    }
    assert rows == set(PER_PRINTING_PARAMS)


def test_plan_remap_n_printings():
    a = CardVocab.from_names(["a", "b"])
    b = CardVocab.from_names(["a", "b", "c"])
    assert cv.plan_remap_n_printings(a, a, 2) == 2  # unchanged, even collided
    assert cv.plan_remap_n_printings(a, b, 2) == 4  # grow to hold all
    assert cv.plan_remap_n_printings(a, b, 50) == 50  # never shrink


def _state_dict(cfg: StructuredModelConfig, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return StructuredModel(cfg).state_dict()


def test_identical_vocab_passes_state_dict_through_untouched():
    v = CardVocab.from_names([f"c{i:02d}" for i in range(10)])
    cfg = _small_config(n_printings=8)  # collided table, unchanged vocab
    sd = _state_dict(cfg)
    new_cfg, out, rep = adapt_structured_state_dict(
        sd, stored=v, current=v, config=cfg,
    )
    assert new_cfg is cfg
    assert all(out[k] is sd[k] for k in sd)
    assert rep.identical and rep.n_kept == 10 and rep.n_new == 0
    assert "no remap" in rep.summary()


def test_remap_keeps_shared_rows_fresh_inits_new_and_drops_removed(caplog):
    old_names = [f"card{i:03d}" for i in range(0, 60, 2)]  # 30 names
    stored = CardVocab.from_names(old_names)
    dropped = {"card000", "card030"}
    added = ["card001", "card031", "card999", "aaa_first"]
    current = CardVocab.from_names(
        [n for n in old_names if n not in dropped] + added,
    )
    old_n = 12  # heavily bucketed: 30 names in 11 real rows
    cfg = _small_config(n_printings=old_n)
    sd = _state_dict(cfg, seed=3)
    rng_before = torch.get_rng_state()
    with caplog.at_level(logging.WARNING, logger="mtg_bot.training.card_vocab"):
        new_cfg, out, rep = adapt_structured_state_dict(
            sd, stored=stored, current=current, config=cfg,
            log_key="unit-remap",
        )
    assert torch.equal(torch.get_rng_state(), rng_before)  # RNG untouched
    new_n = current.size + 1
    assert new_cfg.n_printings == new_n and cfg.n_printings == old_n
    assert (rep.n_kept, rep.n_new, rep.n_dropped) == (28, 4, 2)
    assert set(rep.new_names) == set(added)
    assert set(rep.dropped_names) == dropped
    fresh = cv.fresh_per_printing_params(
        new_cfg, seed=cv.remap_seed(stored, current, new_n),
    )
    for key in PER_PRINTING_PARAMS:
        old_t, new_t = sd[key], out[key]
        assert new_t.shape[0] == new_n
        assert torch.equal(new_t[0], old_t[0])  # NA row
        for name, raw in current.id_map.items():
            if name in stored.id_map:
                src = row_for_raw_id(stored.id_map[name], old_n)
                assert torch.equal(new_t[raw], old_t[src]), (key, name)
            else:
                assert torch.equal(new_t[raw], fresh[key][raw]), (key, name)
    # Fresh rows are exactly what a fresh model of the new config gives.
    torch.manual_seed(cv.remap_seed(stored, current, new_n))
    pristine = StructuredModel(new_cfg).state_dict()
    for key in PER_PRINTING_PARAMS:
        assert torch.equal(fresh[key], pristine[key])
    # Every non-printing parameter is passed through unchanged.
    for key in sd:
        if key not in PER_PRINTING_PARAMS:
            assert out[key] is sd[key]
    # Loud log, once.
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "CARD VOCABULARY REMAP" in msgs[0]
    assert "kept 28, new 4" in msgs[0] and "dropped 2" in msgs[0]
    adapt_structured_state_dict(
        sd, stored=stored, current=current, config=cfg, log_key="unit-remap",
    )
    assert len(caplog.records) == 1
    # The remapped state_dict loads into a model of the planned config.
    StructuredModel(new_cfg).load_state_dict(out)


def test_remap_is_deterministic_across_calls():
    stored = CardVocab.from_names(["a", "c"])
    current = CardVocab.from_names(["a", "b", "c"])
    cfg = _small_config(n_printings=3)
    sd = _state_dict(cfg)
    _, one, _ = adapt_structured_state_dict(
        sd, stored=stored, current=current, config=cfg,
    )
    torch.manual_seed(12345)  # caller RNG state must not matter
    _, two, _ = adapt_structured_state_dict(
        sd, stored=stored, current=current, config=cfg,
    )
    for key in PER_PRINTING_PARAMS:
        assert torch.equal(one[key], two[key])


def test_remap_rejects_inconsistent_inputs():
    stored = CardVocab.from_names(["a"])
    current = CardVocab.from_names(["a", "b"])
    cfg = _small_config(n_printings=4)
    sd = _state_dict(cfg)
    with pytest.raises(CardVocabIntegrityError, match="n_printings=9"):
        adapt_structured_state_dict(
            sd, stored=stored, current=current,
            config=_small_config(n_printings=9),
        )
    bad = dict(sd)
    del bad["hand_belief_queries"]
    with pytest.raises(CardVocabIntegrityError, match="lacks"):
        adapt_structured_state_dict(
            bad, stored=stored, current=current, config=cfg,
        )
    bad = dict(sd)
    bad["hand_belief_queries"] = torch.zeros(7, 16)
    with pytest.raises(CardVocabIntegrityError, match="rows"):
        adapt_structured_state_dict(
            bad, stored=stored, current=current, config=cfg,
        )
    with pytest.raises(ValueError):
        cv.remap_printing_rows(
            torch.zeros(4, 2), torch.zeros(5, 2), stored=stored,
            current=current, old_n_printings=4, new_n_printings=3,
        )
    with pytest.raises(ValueError):
        cv.remap_printing_rows(
            torch.zeros(4, 2), torch.zeros(3, 2), stored=stored,
            current=current, old_n_printings=5, new_n_printings=3,
        )


def test_remap_main_vocab_checkpoint_into_roto_vocab():
    """The real case this feature exists for: a checkpoint on main's
    475-name vocabulary (bucketed into 320 rows, as every s2 checkpoint
    is) loaded under the roto branch's 1323-name default pool."""
    main = cv.load_card_vocab_file(MAIN_VOCAB_FILE)
    roto = cv.load_card_vocab_file(ROTO_VOCAB_FILE)
    cfg = _small_config(n_printings=320)
    sd = _state_dict(cfg, seed=7)
    new_cfg, out, rep = adapt_structured_state_dict(
        sd, stored=main, current=roto, config=cfg,
    )
    assert (rep.n_kept, rep.n_new, rep.n_dropped) == (475, 848, 0)
    assert new_cfg.n_printings == 1324
    emb_old, emb_new = sd["printing_embed.weight"], out["printing_embed.weight"]
    for name, raw in main.id_map.items():
        assert torch.equal(
            emb_new[roto.id_map[name]], emb_old[row_for_raw_id(raw, 320)],
        ), name
    # Without the remap, a roto-vocabulary tokenizer would read nearly
    # every main card's embedding from another card's row.
    moved = sum(
        row_for_raw_id(main.id_map[n], 320) != row_for_raw_id(roto.id_map[n], 320)
        for n in main.names
    )
    assert moved == 474


# ---------------------------------------------------------------------
# End to end through NNEvaluator
# ---------------------------------------------------------------------


def _real_state():
    from mtg_bot.agents import RandomAgent
    from mtg_bot.simulate import setup_game
    from tests._test_decks import simple_test_decklists

    return setup_game(
        agents=[RandomAgent(), RandomAgent()],
        decks=simple_test_decklists(), seed=11,
    )


def _evaluator_kwargs(n_printings: int = 320) -> dict:
    return dict(
        architecture="structured",
        structured_emb_dim=16, structured_n_heads=2, structured_n_layers=1,
        structured_hidden_dim=32, structured_action_history_n=4,
        structured_n_printings=n_printings,
    )


def _forward(ev, state):
    values, logits = ev._structured_forward([state], 0)
    return values.detach().clone(), logits.detach().clone()


def test_save_load_round_trip_stamps_and_reproduces(tmp_path):
    from mtg_bot.search.nn_evaluator import NNEvaluator

    torch.manual_seed(0)
    a = NNEvaluator(**_evaluator_kwargs())
    assert a.card_vocab == current_card_vocab()
    weights = tmp_path / "w.pt"
    a.save(weights)
    assert sidecar_path_for(weights).is_file()
    b = NNEvaluator(weights_path=weights, **_evaluator_kwargs())
    assert b.card_vocab_report.identical
    state = _real_state()
    va, pa = _forward(a, state)
    vb, pb = _forward(b, state)
    assert torch.equal(va, vb) and torch.equal(pa, pb)


def test_unstamped_structured_weights_refused_everywhere(tmp_path):
    from mtg_bot.search.nn_evaluator import NNEvaluator
    from mtg_bot.search.shm_nn_eval_server import SharedTensorNNEvalServer

    weights = tmp_path / "raw.pt"
    torch.save(StructuredModel(_small_config()).state_dict(), weights)
    with pytest.raises(MissingCardVocabError):
        NNEvaluator(weights_path=weights, **_evaluator_kwargs())
    with pytest.raises(MissingCardVocabError):
        SharedTensorNNEvalServer(
            weights_path=weights, device="cpu", warmup=False,
            **_evaluator_kwargs(),
        )
    # An MLP checkpoint has no printing identity and loads as before.
    from mtg_bot.search.nn_evaluator import _ValueMLP

    mlp = tmp_path / "mlp.pt"
    torch.save(_ValueMLP().state_dict(), mlp)
    assert NNEvaluator(weights_path=mlp).card_vocab is None


def test_enlarged_vocabulary_preserves_network_outputs(tmp_path, restore_source):
    """Build + save a checkpoint under the real vocabulary, then enlarge
    the vocabulary (new names sorted throughout, so every existing card
    is renumbered) and load it again: the network must produce the SAME
    value and policy logits for a state made of shared cards. A naive
    load under the enlarged vocabulary must NOT (that is the silent
    scramble the fingerprint prevents)."""
    from mtg_bot.search.nn_evaluator import NNEvaluator

    state = _real_state()
    torch.manual_seed(1)
    original = NNEvaluator(**_evaluator_kwargs(n_printings=320))
    weights = tmp_path / "w.pt"
    original.save(weights)
    v0, p0 = _forward(original, state)

    real = get_active_source().get_printings()
    filler = next(iter(real.values()))
    extra = {
        f"{prefix} Synthetic Card {i}": filler
        for i, prefix in enumerate(["A", "Ba", "Cr", "Gl", "Mo", "Sh", "Zz"] * 6)
    }
    set_active_source(_FakeSource({**real, **extra}))
    enlarged = current_card_vocab()
    assert enlarged.size == len(real) + len(extra)

    loaded = NNEvaluator(weights_path=weights, **_evaluator_kwargs(n_printings=320))
    rep = loaded.card_vocab_report
    assert not rep.identical
    assert (rep.n_kept, rep.n_new, rep.n_dropped) == (len(real), len(extra), 0)
    assert loaded.model.config.n_printings == enlarged.size + 1
    assert loaded._structured_n_printings_cap == enlarged.size + 1
    v1, p1 = _forward(loaded, state)
    assert torch.equal(v0, v1)
    assert torch.equal(p0, p1)

    # The unprotected load: same weights, enlarged vocabulary, no remap.
    naive = NNEvaluator(**_evaluator_kwargs(n_printings=320))
    naive.model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True),
    )
    v2, p2 = _forward(naive, state)
    assert not (torch.equal(v0, v2) and torch.equal(p0, p2))


def test_shrunk_vocabulary_drops_removed_cards(tmp_path, restore_source):
    from mtg_bot.search.nn_evaluator import NNEvaluator

    real = get_active_source().get_printings()
    # A collision-free table for the full vocabulary, whatever the pool.
    table = len(real) + 7
    torch.manual_seed(2)
    ev = NNEvaluator(**_evaluator_kwargs(n_printings=table))
    weights = tmp_path / "w.pt"
    ev.save(weights)
    keep = sorted(real)[::2]
    set_active_source(_FakeSource({n: real[n] for n in keep}))
    loaded = NNEvaluator(weights_path=weights, **_evaluator_kwargs(n_printings=table))
    rep = loaded.card_vocab_report
    assert (rep.n_kept, rep.n_new, rep.n_dropped) == (
        len(keep), 0, len(real) - len(keep),
    )
    assert loaded.model.config.n_printings == table  # never shrinks
    old = torch.load(weights, map_location="cpu", weights_only=True)[
        "printing_embed.weight"
    ]
    new = loaded.model.printing_embed.weight
    old_ids = CardVocab.from_names(real).id_map
    for name, raw in current_card_vocab().id_map.items():
        assert torch.equal(new[raw], old[old_ids[name]])


def test_shm_server_remaps_like_the_evaluator(tmp_path, restore_source):
    from mtg_bot.search.nn_evaluator import NNEvaluator
    from mtg_bot.search.shm_nn_eval_server import SharedTensorNNEvalServer

    torch.manual_seed(4)
    ev = NNEvaluator(**_evaluator_kwargs())
    weights = tmp_path / "w.pt"
    ev.save(weights)
    real = get_active_source().get_printings()
    filler = next(iter(real.values()))
    set_active_source(_FakeSource({**real, "Aaa Synthetic": filler}))
    server = SharedTensorNNEvalServer(
        weights_path=weights, device="cpu", warmup=False,
        **_evaluator_kwargs(),
    )
    rep = server.card_vocab_report
    assert (rep.n_kept, rep.n_new) == (len(real), 1)
    assert server.handle().structured_n_printings_cap == len(real) + 2
    direct = NNEvaluator(weights_path=weights, **_evaluator_kwargs())
    assert torch.equal(
        server._model.printing_embed.weight,
        direct.model.printing_embed.weight,
    )


def test_from_metadata_threads_stored_vocab(tmp_path):
    from mtg_bot.search.nn_evaluator import NNEvaluator
    from mtg_bot.training.registry import ModelMetadata

    torch.manual_seed(5)
    ev = NNEvaluator(**_evaluator_kwargs())
    weights = tmp_path / "w.pt"
    torch.save(ev.model.state_dict(), weights)  # bare: no sidecar
    meta = ModelMetadata(
        version="v1", created_at="x", feature_dim=0, hidden_sizes=[],
        architecture="structured", scalar_schema_version=2,
        card_vocab=current_card_vocab().to_dict(),
        training={"structured_config": {
            "emb_dim": 16, "n_heads": 2, "n_layers": 1, "hidden_dim": 32,
            "action_history_n": 4, "n_printings": 320,
        }},
    )
    loaded = NNEvaluator.from_metadata(weights, meta)
    assert loaded.card_vocab_report.identical
    assert loaded.card_vocab_report.origin == "explicit"
    meta.card_vocab = {}
    with pytest.raises(MissingCardVocabError):
        NNEvaluator.from_metadata(weights, meta)


def test_remap_report_summary_samples_names():
    rep = cv.VocabRemapReport(
        stored_sha256="a" * 64, current_sha256="b" * 64,
        stored_size=3, current_size=9, n_kept=3, n_new=6, n_dropped=0,
        old_n_printings=4, new_n_printings=10,
        new_names=tuple(f"n{i}" for i in range(6)),
    )
    s = rep.summary()
    assert "(+1)" in s and "n0" in s and "dropped 0" in s
