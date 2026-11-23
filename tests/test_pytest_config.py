"""S4.1: test-dependency configuration and default exclusion of live tests."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

LIVE_AND_OFFLINE = '''
import pytest

@pytest.mark.live
def test_live_marked():
    pass

def test_offline():
    pass
'''


def test_http_mocking_dependency_is_declared_for_tests_only() -> None:
    dev = PYPROJECT["project"]["optional-dependencies"]["dev"]
    runtime = PYPROJECT["project"]["dependencies"]
    assert any(spec.replace(" ", "").startswith("httpx2>=2.0,<3") for spec in dev)
    assert not any(spec.startswith("httpx") for spec in runtime)  # comes with anthropic at runtime


def test_mock_transport_is_importable() -> None:
    import httpx2

    assert callable(httpx2.MockTransport)


def test_live_marker_is_declared_and_excluded_by_default() -> None:
    options = PYPROJECT["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("live:") for m in options["markers"])
    assert "-m 'not live'" in options["addopts"]


def _collect(tmp_path: Path, *extra: str) -> str:
    test_file = tmp_path / "test_marker_probe.py"
    test_file.write_text(LIVE_AND_OFFLINE, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-m", "pytest", "-c", str(ROOT / "pyproject.toml"), "--rootdir", str(tmp_path),
         "-p", "no:cacheprovider", "--collect-only", "-q", *extra, str(test_file)],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    return completed.stdout


def test_live_tests_are_deselected_by_default(tmp_path: Path) -> None:
    out = _collect(tmp_path)
    assert "test_offline" in out and "test_live_marked" not in out and "1 deselected" in out


def test_m_live_on_the_command_line_overrides_the_default(tmp_path: Path) -> None:
    out = _collect(tmp_path, "-m", "live")
    assert "test_live_marked" in out and "test_offline" not in out
