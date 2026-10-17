from __future__ import annotations

from pathlib import Path

import pytest

from archivist.cli import main


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_index_command_summarises_the_corpus(
    capsys: pytest.CaptureFixture[str], project_root: Path
) -> None:
    code, out = _run(capsys, "--corpus", str(project_root / "corpus"), "index")
    assert code == 0
    assert "documents: 5" in out
    assert "hashing-512d" in out


def test_ask_command_prints_sources(
    capsys: pytest.CaptureFixture[str], project_root: Path
) -> None:
    code, out = _run(
        capsys,
        "--corpus",
        str(project_root / "corpus"),
        "ask",
        "What is the retention period for events_raw?",
    )
    assert code == 0
    assert "30 days" in out
    assert "sources:" in out


def test_research_command_shows_its_steps(
    capsys: pytest.CaptureFixture[str], project_root: Path
) -> None:
    code, out = _run(
        capsys,
        "--corpus",
        str(project_root / "corpus"),
        "research",
        "Which job writes events_curated?",
    )
    assert code == 0
    assert "step 1: search_corpus" in out


def test_eval_command_passes_against_committed_baselines(
    capsys: pytest.CaptureFixture[str], project_root: Path
) -> None:
    code, out = _run(
        capsys,
        "--corpus",
        str(project_root / "corpus"),
        "eval",
        "--cases",
        str(project_root / "evals" / "qa.jsonl"),
        "--baselines",
        str(project_root / "evals" / "baselines"),
        "--strategy",
        "plain-rag",
        "agentic",
    )
    assert "REGRESSION" not in out
    assert code == 0


def test_missing_corpus_is_a_clean_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, _ = _run(capsys, "--corpus", str(tmp_path / "nope"), "index")
    assert code == 2
