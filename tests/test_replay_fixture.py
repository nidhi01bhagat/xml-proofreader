"""S4.8b: the checked-in replay fixture (AC-30). Offline: no API key, no network, no model output.

* The committed fixture is byte-identical to a fresh regeneration through the real CLI / client / SDK path with
  the in-memory transport (tests/fixtures/llm/generate_replay_fixtures.py).
* Its header matches the current prompt and fingerprint versions; a stale copy is refused.
* It contains no credential, path or private material; its documented SHA-256 matches.
* Replay through ``python -m proofreader`` in a subprocess - without credentials and without network (every socket
  operation is blocked and recorded; none is attempted) - reproduces the hand-written expected output
  byte-for-byte, identically on repeated runs. (The SDK package is imported by ``prompt.py`` for
  ``transform_schema``, an approved S4.3 design; replay never constructs the Anthropic client.)
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import docx_factory as f
import pytest

from proofreader import cli
from proofreader import llm
from proofreader import prompt as pr

TESTS = Path(__file__).resolve().parent
LLM_DIR = TESTS / "fixtures" / "llm"
FIXTURE = LLM_DIR / "golden-e2e-1.recording.json"
PROVENANCE = LLM_DIR / "PROVENANCE.md"
G1_IN = (TESTS / "fixtures" / "xml" / "golden-e2e-1-input.xml").read_bytes()
G1_OUT = (TESTS / "fixtures" / "xml" / "golden-e2e-1-expected.xml").read_bytes()
REPO = TESTS.parent


def _generator():
    spec = importlib.util.spec_from_file_location("generate_replay_fixtures", LLM_DIR / "generate_replay_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEN = _generator()


def guide_docx() -> bytes:
    return GEN._guide_docx()


# --------------------------------------------------------------------------------------------
# Regeneration, header, provenance
# --------------------------------------------------------------------------------------------


def test_committed_fixture_equals_a_fresh_regeneration() -> None:
    assert FIXTURE.read_bytes() == GEN.build_recording()


def test_regeneration_is_deterministic() -> None:
    assert GEN.build_recording() == GEN.build_recording()


def test_header_matches_current_versions_and_loads_strictly() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["format"] == llm.RECORDING_FORMAT and document["version"] == llm.RECORDING_VERSION
    assert document["fingerprint_version"] == pr.FINGERPRINT_VERSION
    assert document["prompt_version"] == pr.PROMPT_VERSION
    results = llm.load_recording(FIXTURE)
    assert len(results) == 8
    assert all(r.failure is None and r.attempts == 1 and r.latency_ms == 0.0 for r in results.values())
    assert sorted(r.request_id for r in results.values()) == [f"req_fixture_{p}" for p in (0, 1, 2, 3, 4, 5, 6, 8)]
    fingerprints = [e["fingerprint"] for e in document["entries"]]
    assert fingerprints == sorted(fingerprints) and len(set(fingerprints)) == 8


def test_provenance_documents_the_exact_fixture() -> None:
    text = PROVENANCE.read_text(encoding="utf-8")
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert f"`{digest}`" in text
    assert f"{len(FIXTURE.read_bytes())} bytes" in text
    assert f"`{pr.PROMPT_VERSION}`" in text and f"`{pr.FINGERPRINT_VERSION}`" in text


def test_fixture_is_committable_not_git_ignored() -> None:
    rel = FIXTURE.relative_to(REPO).as_posix()
    result = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=REPO, capture_output=True)
    assert result.returncode == 1  # 1 = not ignored


# --------------------------------------------------------------------------------------------
# Secret and content scan
# --------------------------------------------------------------------------------------------


def test_fixture_contains_no_credentials_paths_or_private_material() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    assert not any(p.search(text) for p in llm._SECRET_PATTERNS)  # the writer's own credential patterns
    lowered = text.lower()
    for forbidden in (GEN.DUMMY_KEY.lower(), "dummy", "sk-ant", "api_key", "x-api-key", "authorization", "bearer",
                      "sse assignment", "example_input", "sample_input", "style guide.docx", "\\users\\", "/users/",
                      "appdata", "\\temp\\", "/tmp/", "pytest-of-", ".docx", ".xml"):
        assert forbidden not in lowered, forbidden
    assert not re.search(r"[A-Za-z]:\\", text)  # no Windows absolute paths
    for entry in json.loads(text)["entries"]:
        assert set(entry) == {"fingerprint", "result", "failure", "payload", "usage", "attempts", "latency_ms",
                              "request_id", "stop_reason"}  # whitelisted fields only, no request content


def test_fixture_payloads_are_the_hand_written_invented_responses() -> None:
    payloads = sorted(json.dumps(e["payload"], sort_keys=True) for e in
                      json.loads(FIXTURE.read_text(encoding="utf-8"))["entries"])
    expected = sorted(json.dumps(payload, sort_keys=True) for _, payload in GEN.RESPONSES.values())
    assert payloads == expected


# --------------------------------------------------------------------------------------------
# Replay through the command line
# --------------------------------------------------------------------------------------------


def write_inputs(directory: Path) -> None:
    (directory / "doc.xml").write_bytes(G1_IN)
    (directory / "g.docx").write_bytes(guide_docx())


def test_in_process_replay_reproduces_the_expected_bytes_repeatedly(tmp_path) -> None:
    write_inputs(tmp_path)
    outputs = []
    for _ in range(3):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main([str(tmp_path / "doc.xml"), "--lang", "en", "--style-guide", str(tmp_path / "g.docx"),
                         "--replay", str(FIXTURE)], environ={"PYTHON_DOTENV_DISABLED": "1"}, cwd=tmp_path,
                        stdout=out, stderr=err)
        assert code == 0, err.getvalue()
        assert "mode replay" in err.getvalue()
        outputs.append((tmp_path / "doc.corrected.xml").read_bytes())
        (tmp_path / "doc.corrected.xml").unlink()
    assert outputs == [G1_OUT] * 3


# Runs the real entry point in a fresh interpreter with every socket operation blocked and recorded.
GUARDED_ENTRY = r"""
import json, os, runpy, socket, sys
attempts = []
def _blocked(name):
    def refuse(*args, **kwargs):
        attempts.append(name)
        raise OSError("network access is blocked in this test")
    return refuse
socket.socket.connect = _blocked("connect")
socket.socket.connect_ex = _blocked("connect_ex")
socket.create_connection = _blocked("create_connection")
socket.getaddrinfo = _blocked("getaddrinfo")
constructed = []
import proofreader.llm as _llm
_original_init = _llm.AnthropicProofreader.__init__
def _record_init(self, *args, **kwargs):
    constructed.append(True)
    return _original_init(self, *args, **kwargs)
_llm.AnthropicProofreader.__init__ = _record_init
code = 0
try:
    sys.argv = ["proofreader"] + sys.argv[1:]
    runpy.run_module("proofreader", run_name="__main__", alter_sys=True)
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 1
with open(os.environ["GUARD_REPORT"], "w", encoding="utf-8") as fh:
    json.dump({"socket_attempts": attempts, "exit": code, "anthropic_imported": "anthropic" in sys.modules,
               "client_constructed": bool(constructed)}, fh)
sys.exit(code)
"""


def guarded_replay(tmp_path: Path, replay: Path) -> tuple[subprocess.CompletedProcess, dict]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "PROOFREADER_")) and k.upper() not in {"HTTP_PROXY", "HTTPS_PROXY",
                                                                                      "ALL_PROXY"}}
    env.update({"PYTHON_DOTENV_DISABLED": "1", "GUARD_REPORT": str(tmp_path / "guard.json"),
                "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"})  # unreachable, belt and braces
    proc = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", "-c", GUARDED_ENTRY, "doc.xml", "--lang", "en",
                           "--style-guide", "g.docx", "--replay", str(replay)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    report = json.loads((tmp_path / "guard.json").read_text(encoding="utf-8"))
    return proc, report


def test_subprocess_replay_without_credentials_or_network_is_byte_identical(tmp_path) -> None:
    write_inputs(tmp_path)
    outputs = []
    for _ in range(3):
        proc, report = guarded_replay(tmp_path, FIXTURE)
        assert proc.returncode == 0, proc.stderr
        assert report["socket_attempts"] == [] and report["exit"] == 0  # no network operation at all
        assert report["client_constructed"] is False  # replay never builds the Anthropic client
        assert "mode replay" in proc.stderr and "ANTHROPIC_API_KEY" not in proc.stderr
        outputs.append((tmp_path / "doc.corrected.xml").read_bytes())
        (tmp_path / "doc.corrected.xml").unlink()
    assert outputs == [G1_OUT] * 3
    assert (tmp_path / "doc.xml").read_bytes() == G1_IN


def test_network_guard_in_the_subprocess_really_blocks(tmp_path) -> None:
    """Positive control: the same guard records and refuses a connection attempt made by the child."""
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")},
           "GUARD_REPORT": str(tmp_path / "guard.json")}
    probe = GUARDED_ENTRY.replace("code = 0\ntry:", "code = 0\ntry:\n    socket.create_connection(('192.0.2.1', 443), 1)\n", 1)
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", probe], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=60)
    report = json.loads((tmp_path / "guard.json").read_text(encoding="utf-8")) if (tmp_path / "guard.json").exists() \
        else None
    assert proc.returncode != 0 and "network access is blocked" in proc.stderr
    assert report is None  # the attempt aborted the child before it could write a clean report


def test_client_construction_detector_is_live(tmp_path) -> None:
    """Positive control for ``client_constructed``: live mode without a key constructs the client, which refuses
    (MissingApiKeyError) before any request - so the detector fires while no socket operation happens."""
    write_inputs(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "PROOFREADER_"))}
    env.update({"PYTHON_DOTENV_DISABLED": "1", "GUARD_REPORT": str(tmp_path / "guard.json")})
    proc = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", "-c", GUARDED_ENTRY, "doc.xml", "--lang", "en",
                           "--style-guide", "g.docx"], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)
    report = json.loads((tmp_path / "guard.json").read_text(encoding="utf-8"))
    assert proc.returncode == 1 and "ANTHROPIC_API_KEY is not set" in proc.stderr
    assert report["client_constructed"] is True and report["socket_attempts"] == []
    assert not (tmp_path / "doc.corrected.xml").exists()


# --------------------------------------------------------------------------------------------
# Drift and tampering
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [("prompt_version", "s0-old"), ("fingerprint_version", "old-v0")])
def test_stale_fixture_copy_is_refused_before_any_processing(tmp_path, field, value) -> None:
    write_inputs(tmp_path)
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document[field] = value
    stale = tmp_path / "stale.recording.json"
    stale.write_text(json.dumps(document), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([str(tmp_path / "doc.xml"), "--lang", "en", "--style-guide", str(tmp_path / "g.docx"),
                     "--replay", str(stale)], environ={"PYTHON_DOTENV_DISABLED": "1"}, cwd=tmp_path,
                    stdout=out, stderr=err)
    assert code == 1 and "stale" in err.getvalue() and not (tmp_path / "doc.corrected.xml").exists()


def test_changed_request_settings_miss_the_fixture(tmp_path) -> None:
    """A different model changes every fingerprint: replay refuses before any processing (no silent reuse)."""
    write_inputs(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([str(tmp_path / "doc.xml"), "--lang", "en", "--style-guide", str(tmp_path / "g.docx"),
                     "--replay", str(FIXTURE), "--model", "claude-other-model"],
                    environ={"PYTHON_DOTENV_DISABLED": "1"}, cwd=tmp_path, stdout=out, stderr=err)
    assert code == 1 and "no response for 8 of 8" in err.getvalue()
    assert not (tmp_path / "doc.corrected.xml").exists()


def test_a_different_locale_misses_the_fixture(tmp_path) -> None:
    """en-GB selects a different style-guide block, so the requests differ from the recorded ones."""
    write_inputs(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([str(tmp_path / "doc.xml"), "--lang", "en-GB", "--style-guide", str(tmp_path / "g.docx"),
                     "--replay", str(FIXTURE)], environ={"PYTHON_DOTENV_DISABLED": "1"}, cwd=tmp_path,
                    stdout=out, stderr=err)
    assert code == 1 and "no response for 8 of 8" in err.getvalue()
