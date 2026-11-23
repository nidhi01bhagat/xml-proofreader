"""S1 step 6: cluster-aware fine tokens and the whitespace word grid (decision C11).

Special characters are built with chr() so the source contains no invisible literals.
"""

from __future__ import annotations

import pytest

from proofreader import tokens as t

ACUTE = chr(0x0301)  # combining acute accent
ZWJ = chr(0x200D)
VS16 = chr(0xFE0F)  # emoji presentation selector
KEYCAP = chr(0x20E3)
MAN, WOMAN, GIRL = chr(0x1F468), chr(0x1F469), chr(0x1F467)
FAMILY = MAN + ZWJ + WOMAN + ZWJ + GIRL
THUMBS_MEDIUM = chr(0x1F44D) + chr(0x1F3FD)  # thumbs up + skin-tone modifier
HEART = chr(0x2764) + VS16
FLAG_FR = chr(0x1F1EB) + chr(0x1F1F7)
FLAG_DE = chr(0x1F1E9) + chr(0x1F1EA)
NBSP = chr(0x00A0)


def texts(s: str) -> list[str]:
    return [tok.text for tok in t.fine_tokens(s)]


# --------------------------------------------------------------------------------------------
# Clusters
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("e" + ACUTE, ["e" + ACUTE]),
        (FAMILY, [FAMILY]),
        (THUMBS_MEDIUM, [THUMBS_MEDIUM]),
        (HEART, [HEART]),
        ("1" + VS16 + KEYCAP, ["1" + VS16 + KEYCAP]),
        (FLAG_FR + FLAG_DE, [FLAG_FR, FLAG_DE]),  # regional indicators pair up, two flags
        ("ab", ["a", "b"]),
        (ACUTE + "a", [ACUTE, "a"]),  # a leading combining mark is its own cluster
    ],
)
def test_clusters(s: str, expected: list[str]) -> None:
    got = [s[a:b] for a, b in t.clusters(s)]
    assert got == expected
    assert "".join(got) == s


def test_odd_regional_indicator_stands_alone() -> None:
    s = FLAG_FR + chr(0x1F1E9)
    assert [s[a:b] for a, b in t.clusters(s)] == [FLAG_FR, chr(0x1F1E9)]


# --------------------------------------------------------------------------------------------
# Fine tokens
# --------------------------------------------------------------------------------------------


def test_words_and_punctuation() -> None:
    assert texts("Our team will recieve the report.") == ["Our", "team", "will", "recieve", "the", "report", "."]


def test_apostrophe_and_hyphen_split_words() -> None:
    assert texts("don't") == ["don", "'", "t"]
    assert texts("night-shift") == ["night", "-", "shift"]
    assert texts("boys" + chr(0x2019) + " bikes") == ["boys", chr(0x2019), "bikes"]


def test_numbers_with_separators() -> None:
    assert texts("1,000.50 $3") == ["1", ",", "000", ".", "50", "$", "3"]


def test_combining_marks_stay_inside_words() -> None:
    decomposed = "cafe" + ACUTE + " teh"
    toks = t.fine_tokens(decomposed)
    assert [x.text for x in toks] == ["cafe" + ACUTE, "teh"]
    assert (toks[0].start, toks[0].end) == (0, 5)
    assert 4 not in t.fine_boundaries(decomposed)  # never a boundary between base and mark


def test_emoji_sequences_are_single_tokens() -> None:
    s = "hi " + FAMILY + THUMBS_MEDIUM + " " + HEART + FLAG_FR
    assert texts(s) == ["hi", FAMILY, THUMBS_MEDIUM, HEART, FLAG_FR]


def test_whitespace_including_newline_and_nbsp_is_not_a_token() -> None:
    assert texts("a\nb\tc" + NBSP + "d") == ["a", "b", "c", "d"]
    assert t.fine_tokens("") == () and t.fine_tokens(" \n ") == ()


def test_underscore_and_symbols_are_separate_tokens() -> None:
    assert texts("a_b+c") == ["a", "_", "b", "+", "c"]


def test_scripts_without_spaces_form_one_word_run() -> None:
    s = chr(0x65E5) + chr(0x672C) + chr(0x8A9E)  # three CJK ideographs
    assert texts(s) == [s]  # documented limitation: no dictionary word segmentation


def test_token_offsets_are_exact() -> None:
    s = "a, " + FAMILY + " z"
    for tok in t.fine_tokens(s):
        assert s[tok.start : tok.end] == tok.text


# --------------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("s", "span", "ok"),
    [
        ("We saw a bright-red kite.", (9, 19), True),  # "bright-red"
        ("We saw a bright-red kite.", (14, 19), False),  # "t-red" starts inside a word
        ("Salt & peper", (0, 12), True),
        ("cafe" + ACUTE + " x", (0, 4), False),  # would split the accent from its base
        ("cafe" + ACUTE + " x", (0, 5), True),
        ("abc", (1, 1), False),  # empty span
        ("abc", (0, 4), False),  # out of range
    ],
)
def test_is_aligned(s: str, span: tuple[int, int], ok: bool) -> None:
    assert t.is_aligned(s, *span) is ok


# --------------------------------------------------------------------------------------------
# Word grid
# --------------------------------------------------------------------------------------------


def test_word_grid_indices_and_offsets() -> None:
    s = "Our team will recieve the final report on Monday."
    ws = t.words(s)
    assert [(w.index, w.text) for w in ws][:5] == [(0, "Our"), (1, "team"), (2, "will"), (3, "recieve"), (4, "the")]
    assert all(s[w.start : w.end] == w.text for w in ws)
    assert [w.text for w in ws] == s.split()


def test_word_grid_keeps_punctuation_attached() -> None:
    assert [w.text for w in t.words("  done.\n(next)  ")] == ["done.", "(next)"]


def test_word_grid_empty() -> None:
    assert t.words("") == () and t.words(" \t\n") == ()
