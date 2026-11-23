"""S0 foundation checks: package layout, repo hygiene, no secrets in source.

These tests exercise no production behaviour (none exists yet).
"""

from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "proofreader"

APPROVED_MODULES = [
    "cli", "config", "models", "encoding", "xml_source", "tokens", "quotes",
    "resolver", "annotation", "verify", "style_guide", "llm", "pipeline", "metrics",
]
SUPERSEDED_MODULES = ["xml_parser", "validator", "ai_client"]


def test_package_version() -> None:
    import proofreader

    assert proofreader.__version__ == "0.1.0"


@pytest.mark.parametrize("name", APPROVED_MODULES + ["__main__"])
def test_approved_module_imports(name: str) -> None:
    importlib.import_module(f"proofreader.{name}")


@pytest.mark.parametrize("name", SUPERSEDED_MODULES)
def test_superseded_module_absent(name: str) -> None:
    assert not (PKG / f"{name}.py").exists()


def test_no_src_main_py() -> None:
    """The only entry point is ``python -m proofreader`` (D10)."""
    assert not (ROOT / "src" / "main.py").exists()


def test_prompts_live_inside_package_only() -> None:
    assert (PKG / "prompts").is_dir()
    assert not (ROOT / "prompts").exists(), "no top-level prompts/ directory (D10)"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


# Directory-only patterns (trailing "/") are checked via a file path inside the directory,
# because `git check-ignore --no-index` cannot know that a non-existent path is a directory.
@pytest.mark.parametrize("path", [
    ".venv/pyvenv.cfg", ".env", ".env.local", "src/proofreader/__pycache__/x.pyc",
    ".pytest_cache/v/x", ".hypothesis/x", "src/proofreader.egg-info/PKG-INFO",
    "out.corrected.xml", "run.log",
    # Original assessment materials stay local only.
    "SSE Assignment/AI Engineer Assignment.docx", "SSE Assignment/Style Guide.docx",
    "SSE Assignment/example_input.xml", "SSE Assignment/example_output.xml",
    "SSE Assignment/sample_input.xml",
    "local/assessment-derived-analysis.md",
])
def test_gitignore_excludes(path: str) -> None:
    result = _git("check-ignore", "-q", "--no-index", path)
    assert result.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", [".env.example", ".gitignore", "pyproject.toml", "requirements.txt",
                                  "README.md", "src/proofreader/cli.py", "tests/fixtures/xml/minimal.xml",
                                  "docs/STATUS.md"])
def test_gitignore_keeps(path: str) -> None:
    result = _git("check-ignore", "-q", "--no-index", path)
    assert result.returncode == 1, f"{path} is unexpectedly ignored"


SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"ANTHROPIC_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # S5: Google (Gemini) API key shape
    re.compile(r"GEMINI_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{8,}"),
]


def test_no_secrets_in_tracked_source_files() -> None:
    candidates = [ROOT / ".env.example", ROOT / "pyproject.toml", ROOT / "requirements.txt"]
    candidates += list((ROOT / "src").rglob("*.py")) + list((ROOT / "src").rglob("*.md"))
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(text), f"possible secret in {path.relative_to(ROOT)}"
