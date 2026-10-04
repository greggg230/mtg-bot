"""Tests for ``mtg_bot card-vocab`` (show / export / stamp / diff).

Git reconstruction is replaced by an in-memory fake here (it has its
own tests in ``tests/training/test_card_vocab_git.py``) so every case
runs in milliseconds; one test drives the real reconstruction end to
end through the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mtg_bot.cli import card_vocab as cli
from mtg_bot.training import card_vocab_git as cvg
from mtg_bot.training.card_vocab import (
    CardVocab,
    MissingCardVocabError,
    current_card_vocab,
    load_card_vocab_file,
    resolve_checkpoint_card_vocab,
    sidecar_path_for,
    write_card_vocab_file,
    write_sidecar,
)

SHA_A = "a" * 40
SHA_B = "b" * 40
VOCAB_A = CardVocab.from_names(["Alpha", "Beta"])
VOCAB_B = CardVocab.from_names(["Alpha", "Beta", "Gamma"])


@pytest.fixture
def fake_git(monkeypatch):
    calls: list[str] = []
    table = {SHA_A: VOCAB_A, SHA_B: VOCAB_B}

    def resolve(commit, *, repo=None):
        for sha in table:
            if sha.startswith(commit):
                return sha
        raise cvg.CardVocabReconstructionError(f"unknown git commit {commit!r}")

    def reconstruct(sha, *, pool=None, repo=None, python=None):
        calls.append(sha)
        return cvg.ReconstructedVocab(
            vocab=table[sha], git_commit=sha, pool=pool or "default",
            default_n_printings=320,
        )

    monkeypatch.setattr(cvg, "resolve_commit", resolve)
    monkeypatch.setattr(cvg, "reconstruct_card_vocab", reconstruct)
    return calls


def _version(
    root: Path, name: str, *, arch="structured", git_commit=None, extra=None,
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    torch.save({"w": torch.full((2,), float(len(name)))}, d / "weights.pt")
    meta = {"version": name, "architecture": arch}
    if git_commit is not None:
        meta["git_commit"] = git_commit
    meta.update(extra or {})
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


def _bytes(d: Path) -> tuple[bytes, bytes]:
    return (d / "weights.pt").read_bytes(), (d / "metadata.json").read_bytes()


# ---------------------------------------------------------------------
# show
# ---------------------------------------------------------------------


def test_show_reports_unstamped_stamped_invalid_and_mlp(tmp_path, capsys):
    reg = tmp_path / "reg"
    _version(reg, "v1", git_commit="aaaa")
    _version(reg, "v2", arch="shared")
    v3 = _version(reg, "v3")
    write_sidecar(current_card_vocab(), v3 / "weights.pt")
    v4 = _version(reg, "v4")
    sidecar_path_for(v4 / "weights.pt").write_text("{bad")
    rc = cli.main([
        "show", "--registry-root", str(reg), "--all-versions",
        "--compare-current",
    ])
    out = capsys.readouterr().out
    assert rc == 1
    assert "v1: UNSTAMPED (git_commit aaaa)" in out
    assert "architecture 'shared'" in out
    assert "from sidecar; matches current" in out
    assert "v4: INVALID" in out
    rc = cli.main(["show", "--version-dir", str(v3)])
    assert rc == 0 and "from sidecar" in capsys.readouterr().out
    other = _version(tmp_path / "x", "v9")
    write_sidecar(VOCAB_A, other / "weights.pt")
    cli.main(["show", "--version-dir", str(other), "--compare-current"])
    assert "DIFFERS from current" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["show"],
        ["show", "--versions", "v1"],
        ["show", "--registry-root", "REG"],
        ["show", "--version-dir", "NOT_A_VERSION"],
        ["show", "--registry-root", "MISSING", "--all-versions"],
    ],
)
def test_selection_errors(tmp_path, argv):
    (tmp_path / "reg").mkdir()
    argv = [
        a.replace("REG", str(tmp_path / "reg"))
        .replace("NOT_A_VERSION", str(tmp_path))
        .replace("MISSING", str(tmp_path / "missing"))
        for a in argv
    ]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    # A read-only tool never creates a registry root.
    assert not (tmp_path / "missing").exists()


def test_all_versions_skips_incomplete_and_non_version_dirs(tmp_path, capsys):
    reg = tmp_path / "reg"
    _version(reg, "v2")
    _version(reg, "v10")
    (reg / "v3").mkdir()  # incomplete
    (reg / "notes").mkdir()
    cli.main(["show", "--registry-root", str(reg), "--all-versions"])
    lines = capsys.readouterr().out.splitlines()
    shown = [Path(ln.split(": ", 1)[0]).name for ln in lines]
    assert shown == ["v2", "v10"]  # numeric order; v3 + notes skipped


# ---------------------------------------------------------------------
# export / diff
# ---------------------------------------------------------------------


def test_export_current_and_commit(tmp_path, fake_git, capsys):
    out = tmp_path / "cur.json"
    assert cli.main(["export", "--current", "--out", str(out)]) == 0
    assert load_card_vocab_file(out) == current_card_vocab()
    assert json.loads(out.read_text())["provenance"]["method"] == "current_checkout"
    out2 = tmp_path / "a.json"
    assert cli.main(["export", "--git-commit", "aaaa", "--out", str(out2)]) == 0
    assert load_card_vocab_file(out2) == VOCAB_A
    prov = json.loads(out2.read_text())["provenance"]
    assert prov["git_commit"] == SHA_A and prov["pool"] == "default"
    assert cli.main(["export", "--git-commit", "zzzz", "--out", str(out2)]) == 1
    assert "unknown git commit" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["export", "--current", "--pool", "cube", "--out", str(out)])


def test_diff_reports_kept_new_dropped_and_renumbering(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    write_card_vocab_file(old, CardVocab.from_names(["b", "c", "d"]))
    write_card_vocab_file(new, CardVocab.from_names(["a", "b", "c"]))
    assert cli.main(["diff", str(old), str(new), "--n-printings", "3"]) == 0
    out = capsys.readouterr().out
    assert "kept 2  new 1  dropped 1" in out
    assert "raw printing id changes: 2" in out
    assert "ROW changes at n_printings=3: 2" in out


def test_dispatcher_routes_card_vocab(tmp_path, capsys):
    from mtg_bot.cli import main as dispatch

    a = tmp_path / "a.json"
    write_card_vocab_file(a, VOCAB_A)
    dispatch(["card-vocab", "diff", str(a), str(a)])
    assert "kept 2  new 0  dropped 0" in capsys.readouterr().out


# ---------------------------------------------------------------------
# stamp
# ---------------------------------------------------------------------


def test_stamp_sidecar_from_metadata_commit_leaves_originals_untouched(
    tmp_path, fake_git, capsys,
):
    reg = tmp_path / "reg"
    dirs = [
        _version(reg, "v1", git_commit="aaaa"),
        _version(reg, "v2", git_commit="aaaa"),
        _version(reg, "v3", git_commit="bbbb"),
    ]
    before = {d: _bytes(d) for d in dirs}
    rc = cli.main([
        "stamp", "--registry-root", str(reg), "--all-versions",
        "--use-metadata-commit", "--sidecar",
    ])
    assert rc == 0
    assert fake_git == [SHA_A, SHA_B]  # memoised per commit
    for d in dirs:
        assert _bytes(d) == before[d]
        assert sidecar_path_for(d / "weights.pt").is_file()
    assert resolve_checkpoint_card_vocab(dirs[0] / "weights.pt").vocab == VOCAB_A
    assert resolve_checkpoint_card_vocab(dirs[2] / "weights.pt").vocab == VOCAB_B
    payload = json.loads(sidecar_path_for(dirs[0] / "weights.pt").read_text())
    prov = payload["card_vocab"]["provenance"]
    assert prov["checkpoint_git_commit"] == "aaaa"
    assert prov["commit_check"] == "is the checkpoint's own git_commit"
    assert prov["stamped_by"] == "mtg_bot card-vocab stamp"
    # Idempotent re-run.
    rc = cli.main([
        "stamp", "--version-dir", str(dirs[0]), "--use-metadata-commit",
        "--sidecar",
    ])
    assert rc == 0 and "already stamped" in capsys.readouterr().out


def test_stamp_cross_checks_the_checkpoint_commit(tmp_path, fake_git, capsys):
    d = _version(tmp_path / "reg", "v1", git_commit="aaaa")
    vb = tmp_path / "b.json"
    write_card_vocab_file(vb, VOCAB_B)
    rc = cli.main([
        "stamp", "--version-dir", str(d), "--vocab-file", str(vb), "--sidecar",
    ])
    assert rc == 1
    assert "differs from the one at the checkpoint's git_commit" in capsys.readouterr().err
    assert not sidecar_path_for(d / "weights.pt").exists()
    rc = cli.main([
        "stamp", "--version-dir", str(d), "--vocab-file", str(vb), "--sidecar",
        "--allow-commit-mismatch",
    ])
    assert rc == 0
    assert "DIFFERS from the checkpoint's git_commit (allowed)" in capsys.readouterr().out
    # A different vocabulary cannot overwrite an existing stamp.
    rc = cli.main([
        "stamp", "--version-dir", str(d), "--git-commit", "aaaa", "--sidecar",
    ])
    assert rc == 1 and "already stamps a different" in capsys.readouterr().err
    # Matching explicit commit passes the check.
    d2 = _version(tmp_path / "reg", "v2", git_commit="aaaa")
    assert cli.main([
        "stamp", "--version-dir", str(d2), "--git-commit", "aaaa", "--sidecar",
    ]) == 0
    assert "matches the checkpoint's git_commit" in capsys.readouterr().out


def test_stamp_without_or_with_unknown_metadata_commit(tmp_path, fake_git, capsys):
    reg = tmp_path / "reg"
    no_commit = _version(reg, "v1")
    unknown = _version(reg, "v2", git_commit="ffff")
    va = tmp_path / "a.json"
    write_card_vocab_file(va, VOCAB_A)
    assert cli.main([
        "stamp", "--version-dir", str(no_commit), "--use-metadata-commit",
        "--sidecar",
    ]) == 1
    assert "records no git_commit" in capsys.readouterr().err
    assert cli.main([
        "stamp", "--version-dir", str(no_commit), "--vocab-file", str(va),
        "--sidecar",
    ]) == 1
    assert "cannot cross-check" in capsys.readouterr().err
    assert cli.main([
        "stamp", "--version-dir", str(unknown), "--vocab-file", str(va),
        "--sidecar",
    ]) == 1
    assert "cannot reconstruct" in capsys.readouterr().err
    assert cli.main([
        "stamp", "--version-dir", str(no_commit), "--version-dir", str(unknown),
        "--vocab-file", str(va), "--sidecar", "--allow-commit-mismatch",
    ]) == 0
    out = capsys.readouterr().out
    assert "skipped (metadata records no git_commit)" in out
    assert "skipped (ffff not reconstructable)" in out


def test_stamp_copies_skip_and_dry_run(tmp_path, fake_git, capsys):
    reg = tmp_path / "reg"
    v1 = _version(reg, "v1", git_commit="aaaa")
    v2 = _version(reg, "v2", git_commit="aaaa")
    mlp = _version(reg, "v3", arch="shared")
    embedded = _version(
        reg, "v4", git_commit="aaaa",
        extra={"card_vocab": VOCAB_A.to_dict()},
    )
    before = _bytes(v1)
    # Dry run writes nothing.
    assert cli.main([
        "stamp", "--version-dir", str(v1), "--use-metadata-commit",
        "--sidecar", "--dry-run",
    ]) == 0
    assert "would write" in capsys.readouterr().out
    assert not sidecar_path_for(v1 / "weights.pt").exists()
    assert cli.main([
        "stamp", "--version-dir", str(v1), "--use-metadata-commit",
        "--out-dir", str(tmp_path / "c1"), "--dry-run",
    ]) == 0
    assert not (tmp_path / "c1").exists()
    # --out-dir writes a stamped copy; the original is untouched.
    out = tmp_path / "copy"
    assert cli.main([
        "stamp", "--version-dir", str(v1), "--use-metadata-commit",
        "--out-dir", str(out),
    ]) == 0
    assert _bytes(v1) == before
    meta = json.loads((out / "metadata.json").read_text())
    assert CardVocab.from_dict(meta["card_vocab"]) == VOCAB_A
    assert (out / "weights.pt").read_bytes() == before[0]
    assert resolve_checkpoint_card_vocab(out / "weights.pt").origin == "metadata"
    # Non-empty target refused.
    assert cli.main([
        "stamp", "--version-dir", str(v2), "--use-metadata-commit",
        "--out-dir", str(out),
    ]) == 1
    assert "not empty" in capsys.readouterr().err
    assert cli.main([
        "stamp", "--version-dir", str(v2), "--use-metadata-commit",
        "--out-dir", str(v2),
    ]) == 1
    assert "refusing to stamp in place" in capsys.readouterr().err
    # --out-root stamps several; MLP skipped; embedded noted.
    assert cli.main([
        "stamp", "--registry-root", str(reg), "--all-versions",
        "--use-metadata-commit", "--out-root", str(tmp_path / "root"),
    ]) == 0
    text = capsys.readouterr().out
    assert "v3: skipped (architecture 'shared')" in text
    assert "v4: already carries vocab" in text
    assert (tmp_path / "root" / "v2" / "metadata.json").is_file()
    assert embedded.is_dir() and mlp.is_dir()


def test_stamp_argument_errors(tmp_path, fake_git):
    v1 = _version(tmp_path / "reg", "v1", git_commit="aaaa")
    v2 = _version(tmp_path / "reg", "v2", git_commit="aaaa")
    va = tmp_path / "a.json"
    write_card_vocab_file(va, VOCAB_A)
    with pytest.raises(SystemExit):
        cli.main([
            "stamp", "--version-dir", str(v1), "--version-dir", str(v2),
            "--use-metadata-commit", "--out-dir", str(tmp_path / "o"),
        ])
    with pytest.raises(SystemExit):
        cli.main([
            "stamp", "--version-dir", str(v1), "--vocab-file", str(va),
            "--pool", "cube", "--sidecar",
        ])


def test_stamp_real_s2_style_checkpoint_end_to_end(tmp_path, capsys):
    """Real git reconstruction: a structured checkpoint whose metadata
    records HEAD as its git_commit is stamped from that commit, loads
    with no remap, and its files stay byte-identical."""
    from mtg_bot.search.nn_evaluator import NNEvaluator
    from mtg_bot.training.card_vocab_git import resolve_commit
    from mtg_bot.training.registry import Registry

    try:
        head = resolve_commit("HEAD")
    except Exception:  # noqa: BLE001
        pytest.skip("needs a git checkout")
    reg = tmp_path / "reg"
    d = reg / "v1"
    d.mkdir(parents=True)
    torch.manual_seed(0)
    ev = NNEvaluator(
        architecture="structured", structured_emb_dim=16, structured_n_heads=2,
        structured_n_layers=1, structured_hidden_dim=32,
        structured_action_history_n=4,
    )
    torch.save(ev.model.state_dict(), d / "weights.pt")
    (d / "metadata.json").write_text(json.dumps({
        "version": "v1", "created_at": "x", "feature_dim": 0,
        "hidden_sizes": [], "with_policy": True, "architecture": "structured",
        "scalar_schema_version": 2, "git_commit": head[:8],
        "training": {"structured_config": {
            "emb_dim": 16, "n_heads": 2, "n_layers": 1, "hidden_dim": 32,
            "action_history_n": 4, "n_printings": 320,
        }},
    }))
    before = _bytes(d)
    with pytest.raises(MissingCardVocabError):
        NNEvaluator.from_metadata(
            d / "weights.pt", Registry(reg).load_metadata("v1"),
        )
    assert cli.main([
        "stamp", "--registry-root", str(reg), "--versions", "v1",
        "--use-metadata-commit", "--sidecar",
    ]) == 0
    assert _bytes(d) == before
    loaded = NNEvaluator.from_metadata(
        d / "weights.pt", Registry(reg).load_metadata("v1"),
    )
    assert loaded.card_vocab_report.identical
