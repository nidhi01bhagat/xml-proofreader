"""Committed byte fixtures must match their generator exactly (no drift, no EOL conversion)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_xml_fixtures", FIXTURE_DIR / "generate_xml_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


@pytest.mark.parametrize("name", sorted(GEN.FIXTURES))
def test_committed_fixture_matches_generator(name: str) -> None:
    path = GEN.XML_DIR / name
    assert path.is_file(), f"missing fixture {name}; run tests/fixtures/generate_xml_fixtures.py"
    assert path.read_bytes() == GEN.FIXTURES[name]


def test_no_unlisted_fixture_files() -> None:
    on_disk = {p.name for p in GEN.XML_DIR.iterdir() if p.is_file()}
    assert on_disk == set(GEN.FIXTURES)
