"""Decide whether a model's raw response is a clean single letter (A-E).

The experiment requires each answer as a bare letter. A response either is one
letter (optionally wrapped/punctuated, e.g. "C", "(C)", "**C**", "C.") or it is
not; anything that is not is recollected (see ``redo_nonsingle.py``) rather than
parsed further. The published dataset therefore contains only single-letter
responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNPARSEABLE = "unparseable"

# Strip leading bullet/markdown noise and trailing markdown/space from a line.
_LINE_NOISE = re.compile(r"^[\s>*#\-••\.\)]+|[\s*#`]+$")
# A line that is *only* a letter, optionally wrapped/punctuated: "C", "(C)", "**C**", "C.".
_BARE = re.compile(r"^\(?\[?([A-Ea-e])\]?\)?[.):\s]*$")


@dataclass
class ParseResult:
    letter: str    # 'A'..'E' or 'unparseable'
    method: str    # 'single_letter' or 'not_single_letter'


def parse_response_detailed(text: str, option_mapping: dict[str, str] | None = None) -> ParseResult:
    """Return the letter if ``text`` is a clean single letter, else unparseable.

    ``option_mapping`` is accepted for interface compatibility but unused.
    """
    if not text or not text.strip():
        return ParseResult(UNPARSEABLE, "not_single_letter")

    cleaned = _LINE_NOISE.sub("", text.strip()).strip()
    m = _BARE.match(cleaned)
    if m and len(cleaned) <= 4:
        return ParseResult(m.group(1).upper(), "single_letter")

    return ParseResult(UNPARSEABLE, "not_single_letter")


def parse_response(text: str, option_mapping: dict[str, str] | None = None) -> str:
    return parse_response_detailed(text, option_mapping).letter


if __name__ == "__main__":
    cases = [
        # (text, expected_letter, expected_method)
        ("A", "A", "single_letter"),
        (" B.", "B", "single_letter"),
        ("**C**", "C", "single_letter"),
        ("(D)", "D", "single_letter"),
        ("", UNPARSEABLE, "not_single_letter"),
        ("The answer is C.", UNPARSEABLE, "not_single_letter"),
        ("D)\n\nbecause it fits best", UNPARSEABLE, "not_single_letter"),
        ("Long reasoning...\n\n**Answer: E**", UNPARSEABLE, "not_single_letter"),
    ]
    failures = 0
    for text, exp_letter, exp_method in cases:
        res = parse_response_detailed(text)
        ok = res.letter == exp_letter and res.method == exp_method
        print(f"{'OK  ' if ok else 'FAIL'} parse({text!r:38}) -> {res.letter!r:14} {res.method}")
        failures += not ok
    print(f"\n{len(cases) - failures}/{len(cases)} passed.")
    if failures:
        raise SystemExit(1)
