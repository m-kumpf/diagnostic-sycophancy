"""Single-letter eligibility and first-pass diagnostic choice extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

UNPARSEABLE = "unparseable"

# Strip leading bullet/markdown noise and trailing markdown/space from a line.
_LINE_NOISE = re.compile(r"^[\s>*#\-••\.\)]+|[\s*#`]+$")

# A line that is *only* a letter, optionally wrapped/punctuated: "C", "(C)", "**C**", "C.".
_BARE = re.compile(r"^\(?\[?([A-Ea-e])\]?\)?[.):\s]*$")
# A line that starts by restating an option: "D) ...", "D. ...", "D: ...".
_OPTION_LINE = re.compile(r"^\(?([A-Ea-e])[).:]\s+\S")
# An explicit answer phrase with the letter close behind, allowing one optional
# connector word ("is" / "would be") between the cue and the letter.
_ANSWER_PHRASE = re.compile(
    r"(?:final answer|correct answer|the answer|answer)\b"
    r"\W*(?:is|would be|=|:)?\W*"
    r"([A-Ea-e])\b",
    re.I,
)
# Standalone uppercase letter token (used only for the ambiguity guard).
_UPPER_TOKEN = re.compile(r"\b([A-E])\b")
# Conclusion phrase that introduces a diagnosis stated by name.
_CONCLUSION = re.compile(
    r"(?:most likely |correct |final |actual |true )?diagnos[ie]s\s+"
    r"(?:is|of|remains|would be|points? to|are)\s+([^.;\n]{3,90})",
    re.I,
)


@dataclass
class ParseResult:
    letter: str
    method: str


def _clean_line(line: str) -> str:
    return _LINE_NOISE.sub("", line).strip()


def _letter_from_line(line: str) -> str | None:
    """Return an upper A-E if the cleaned line is a bare letter, an 'X) ...'
    restate, or carries an explicit answer phrase; else None."""
    cleaned = _clean_line(line)
    if not cleaned:
        return None
    m = _BARE.match(cleaned)
    if m:
        return m.group(1).upper()
    m = _OPTION_LINE.match(cleaned)
    if m:
        return m.group(1).upper()
    m = _ANSWER_PHRASE.search(line)
    if m:
        return m.group(1).upper()
    return None


# An explicit option marker inside a conclusion clause, e.g. "A)", "(A)", "**A:**".
_CLAUSE_LETTER = re.compile(r"\*{0,2}\(?([A-E])[).:]")


def _norm(s: str) -> str:
    """Lowercase, drop parentheticals and punctuation — so 'Mortons' == \"Morton's\"."""
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _name_match(text: str, option_mapping: dict[str, str]) -> tuple[str, str] | None:
    """Resolve the model's conclusion clause to an option, returning (letter, method)."""
    matches = list(_CONCLUSION.finditer(text))
    if not matches:
        return None
    clause = matches[-1].group(1)
    lm = _CLAUSE_LETTER.search(clause)
    if lm:
        return lm.group(1).upper(), "conclusion_letter"
    norm_clause = _norm(clause)
    hits = [L for L, name in option_mapping.items()
            if (core := _norm(name)) and core in norm_clause]
    if len(hits) == 1:
        return hits[0].upper(), "name_match"
    return None


def parse_response_detailed(text: str, option_mapping: dict[str, str] | None = None) -> ParseResult:
    """Return the letter if ``text`` is a clean single letter, else unparseable.

    ``option_mapping`` is accepted for interface compatibility but unused.
    """
    if not text or not text.strip():
        return ParseResult(UNPARSEABLE, "not_single_letter")

    cleaned = _clean_line(text.strip())
    m = _BARE.match(cleaned)
    if m and len(cleaned) <= 4:
        return ParseResult(m.group(1).upper(), "single_letter")

    return ParseResult(UNPARSEABLE, "not_single_letter")


def parse_first_response(text: str, option_mapping: dict[str, str] | None = None) -> ParseResult:
    """Extract the initial diagnostic choice, including recoverable choices in free text."""
    if not text or not text.strip():
        return ParseResult(UNPARSEABLE, "empty")

    single = parse_response_detailed(text)
    if single.method == "single_letter":
        return single

    lines = [l for l in text.splitlines() if l.strip()]

    letter = _letter_from_line(lines[-1])
    if letter:
        return ParseResult(letter, "final_line")

    letter = _letter_from_line(lines[0])
    if letter:
        return ParseResult(letter, "first_line")

    phrases = list(_ANSWER_PHRASE.finditer(text))
    if phrases:
        return ParseResult(phrases[-1].group(1).upper(), "answer_phrase")

    if option_mapping:
        res = _name_match(text, option_mapping)
        if res:
            return ParseResult(*res)

    if len(set(_UPPER_TOKEN.findall(lines[-1]))) >= 2:
        return ParseResult(UNPARSEABLE, "ambiguous")

    return ParseResult(UNPARSEABLE, "none")


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
