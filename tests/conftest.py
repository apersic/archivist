from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from archivist.app import Archivist, build  # noqa: E402
from archivist.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def settings(project_root: Path) -> Settings:
    return Settings(corpus_dir=project_root / "corpus")


@pytest.fixture(scope="session")
def app(settings: Settings) -> Archivist:
    return build(settings)
