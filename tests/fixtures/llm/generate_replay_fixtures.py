"""Deterministic generator for the committed replay fixture (S4.8b, AC-30). See PROVENANCE.md.

The fixture is produced through the REAL code path: ``proofreader.cli.main`` with ``--record``, the real
``AnthropicProofreader`` and the real Anthropic SDK, but every HTTP request is answered in memory by an
``httpx2.MockTransport`` with the HAND-WRITTEN, INVENTED model responses below (no network, no model output).
The API key is a dummy test value; it is never written to the fixture. Production request settings are used
unchanged (model claude-opus-5-5, effort low, max_tokens 8000, schema, prompts); the client clock is fixed so
every recorded latency is 0.

    python -I -B tests/fixtures/llm/generate_replay_fixtures.py            # (re)write the fixture
    python -I -B tests/fixtures/llm/generate_replay_fixtures.py --check    # exit 1 if it would change

tests/test_replay_fixture.py regenerates the fixture and requires it to be byte-identical to the committed file.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
FIXTURE = HERE / "golden-e2e-1.recording.json"
INPUT = TESTS / "fixtures" / "xml" / "golden-e2e-1-input.xml"
DUMMY_KEY = "dummy-fixture-key-not-real-0000"  # never recorded (RecordingWriter stores results only)
FIXED_CLOCK = 0.0

if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))  # docx_factory (test helper)

# Invented style guide (the same one tests/test_e2e_golden.py uses): its rendered text is part of every request
# fingerprint, so it is built here from the same invented paragraphs, never from a supplied document.
GUIDE_PARAGRAPHS = (("House rules", 0), ("Write per cent in running text, never the % sign.", None))


def _guide_docx() -> bytes:
    import docx_factory as f

    return f.make_docx(f.heading(GUIDE_PARAGRAPHS[0][0], 0) + f.para(GUIDE_PARAGRAPHS[1][0]))


def _f(original, before, after, start, end, correction, type_, reason):
    return {"original": original, "before": before, "after": after, "word_start": start, "word_end": end,
            "correction": correction, "type": type_, "reason": reason}


# Hand-written, invented model responses for golden-e2e-1-input.xml, keyed by paragraph id (see the fixture
# comment in tests/fixtures/generate_xml_fixtures.py). Paragraph 7 (existing <error>) and 9 (whitespace only)
# are never sent. Each key is matched against the start of the paragraph text in the request.
RESPONSES: dict[int, tuple[str, dict]] = {
    0: ("Our office", {"findings": [
        _f("recieve", "Our office will", "the new chairs", 3, 3, "receive", "spelling", "Misspelling"),
        _f("tuesday", "new chairs on", "morning.", 8, 8, "Tuesday", "capitalization", "Days of the week are capitalized"),
    ]}),
    1: ("Tea", {"findings": [
        _f("cofee", "Tea &", "are served in", 2, 2, "coffee", "spelling", "Misspelling"),
        _f("lobbby", "served in the", "daily.", 7, 7, "lobby", "spelling", "Misspelling"),
    ]}),
    2: ("The café", {"findings": [
        _f("then", "café opened earlier", "usual this week.", 4, 4, "than", "grammar", 'Use "than" for comparisons'),
    ]}),
    3: ("Sales", {"findings": [
        _f("12%", "Sales grew by", "in the spring.", 3, 3, "12 per cent", "styleguide", "House style: write per cent"),
    ]}),
    4: ("Outer text", {"findings": []}),
    5: ("Inner text", {"findings": [
        _f("a extra", "Inner text has", "word", 3, 4, "an extra", "grammar", "Use an before a vowel sound"),
    ]}),
    6: ("Everything", {"findings": []}),
    8: ("Data", {"findings": [
        _f("the the", "& stuff follows", "rules.", 5, 6, "the", "grammar", "Repeated word"),
    ]}),
}


def _pid_for(user_message: str) -> int:
    text = "".join(json.loads(user_message)["paragraph"]["segments"])
    matches = [pid for pid, (prefix, _) in RESPONSES.items() if text.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError("request does not match exactly one invented response")
    return matches[0]


def build_recording() -> bytes:
    """Run the real CLI with --record against the in-memory transport; return the recording bytes."""
    import httpx2

    from proofreader import cli, llm

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.host == "api.anthropic.com" and request.url.path == "/v1/messages"
        body = json.loads(request.content)
        pid = _pid_for(body["messages"][0]["content"])
        message = {"id": f"msg_fixture_{pid}", "type": "message", "role": "assistant", "model": body["model"],
                   "content": [{"type": "text", "text": json.dumps(RESPONSES[pid][1], ensure_ascii=False)}],
                   "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 1000 + pid, "output_tokens": 50 + pid,
                             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
        return httpx2.Response(200, json=message, headers={"request-id": f"req_fixture_{pid}"})

    def factory(cfg, recorder):
        http = httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)
        return llm.AnthropicProofreader(cfg, http_client=http, recorder=recorder, sleep=lambda s: None,
                                        clock=lambda: FIXED_CLOCK)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "doc.xml").write_bytes(INPUT.read_bytes())
        (tmp / "g.docx").write_bytes(_guide_docx())
        target = tmp / "fixture.recording.json"
        out, err = io.StringIO(), io.StringIO()
        code = cli.main([str(tmp / "doc.xml"), "--lang", "en", "--style-guide", str(tmp / "g.docx"),
                         "--record", str(target)],
                        environ={"ANTHROPIC_API_KEY": DUMMY_KEY, "PYTHON_DOTENV_DISABLED": "1"}, cwd=tmp,
                        proofreader_factory=factory, stdout=out, stderr=err)
        if code != 0:
            raise RuntimeError(f"fixture generation run failed with exit code {code}:\n{err.getvalue()}")
        return target.read_bytes()


def main(argv: list[str]) -> int:
    data = build_recording()
    if "--check" in argv:
        same = FIXTURE.is_file() and FIXTURE.read_bytes() == data
        print("fixture up to date" if same else "fixture differs from regeneration")
        return 0 if same else 1
    FIXTURE.write_bytes(data)
    print(f"wrote {FIXTURE.name} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
